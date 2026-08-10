"""
Standalone training script for the Token AE (deterministic autoencoder).

Trains a Token AE to compress per-frame encoder tokens (256 -> num_latent_tokens)
with reconstruction loss (MSE + cosine). The frozen ViT encoder produces target
tokens; only the Token AE parameters receive gradients.

Entry point: ``main(args, resume_preempt=False)``
Called from ``app/scaffold.py`` via:
    python -m app.main --fname <config.yaml> --train-script train_token_ae

Architecture:
    frozen encoder: [B, C, T, H, W] -> [B, T*P, D]   (P=256, D=1408)
    token_ae:       [B, T*P, D]     -> [B, T*L, D]   (L=num_latent_tokens)
                    [B, T*L, D]     -> [B, T*P, D]   (reconstruction)
    loss = MSE(recon, target) + 0.25 * cosine_loss
"""

import gc
import logging
import math
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from app.vjepa_cowa_world_model.models.token_ae import TokenAE
from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.data import create_train_dataloader, create_transforms
from app.vjepa_cowa_world_model.training.models import get_encoder_embed_dim, init_encoder
from app.vjepa_cowa_world_model.training.runtimes.loop_runner import TrainingLoopRunner
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
log_timings = True
log_freq = 10
CHECKPOINT_FREQ = 1

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _configure_rank_logger(rank: int):
    """Silence logger on non-zero ranks."""
    if rank == 0:
        return
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())


# ===================================================================== #
#  Mixed SSL dataset helpers (reuses vjepa_2_1 dataset classes)           #
# ===================================================================== #


class _WeightedConcatDataset(ConcatDataset):
    """ConcatDataset with per-sample weights for weighted sampling."""

    def __init__(self, datasets: List[Dataset], dataset_weights: List[float]):
        super().__init__(datasets)
        assert len(datasets) == len(dataset_weights)
        weights = []
        for ds, w in zip(datasets, dataset_weights):
            n = len(ds)
            weights.extend([w / n] * n)
        self._sample_weights = np.array(weights, dtype=np.float64)

    @property
    def sample_weights(self) -> np.ndarray:
        return self._sample_weights


def _single_sample_collate(batch):
    """Return the single dataset sample unchanged (for batch_size=1 loaders)."""
    return batch[0]


def _build_single_dataset(
    ds_cfg: dict,
    resolved_fpc: int,
    target_fps: float,
    transform: Optional[Callable[..., Any]],
) -> Optional[Tuple[Dataset, float, str, int]]:
    """Build one sub-dataset from its config block.

    Returns ``(dataset, weight, type_name, fpc)`` or *None* if empty.
    """
    from app.vjepa_2_1.image_seq_ssl import ImageSequenceSSLDataset
    from app.vjepa_2_1.video_file_ssl import VideoFileSSLDataset

    ds_type = ds_cfg.get("type", "")
    root_dirs = ds_cfg.get("root_dirs", [])
    w = float(ds_cfg.get("weight", 1.0))
    fpc = int(ds_cfg.get("frames_per_clip", resolved_fpc))

    if ds_type == "image_sequence":
        ds = ImageSequenceSSLDataset(
            root_dirs=root_dirs,
            frames_per_clip=fpc,
            source_fps=float(ds_cfg.get("source_fps", 2.0)),
            target_fps=target_fps,
            transform=transform,
            camera_subdir=ds_cfg.get("camera_subdir"),
            window_stride=int(ds_cfg.get("window_stride", 1)),
            index_cache_path=ds_cfg.get("index_cache_path"),
        )
    elif ds_type == "video_file":
        ds = VideoFileSSLDataset(
            root_dirs=root_dirs,
            frames_per_clip=fpc,
            target_fps=target_fps,
            transform=transform,
            window_stride=int(ds_cfg.get("window_stride", 1)),
            index_cache_path=ds_cfg.get("index_cache_path"),
        )
    else:
        raise ValueError(f"Unknown mixed_ssl dataset type: {ds_type!r}")

    if len(ds) == 0:
        logger.warning("Dataset %s (fpc=%d) has 0 samples, skipping.", ds_type, fpc)
        return None
    return (ds, w, ds_type, fpc)


def _build_mixed_ssl_datasets(
    cfgs_mixed: dict,
    resolved_fpcs: List[int],
    target_fps: float,
    transform: Optional[Callable[..., Any]],
) -> Tuple[List[Dataset], List[float], List[int], List[str]]:
    """Build sub-datasets in parallel from mixed_ssl config."""
    ds_cfgs = cfgs_mixed.get("datasets", [])
    if not ds_cfgs:
        raise RuntimeError("No mixed SSL sub-datasets found in data.mixed_ssl.datasets")

    default_fpc = int(resolved_fpcs[0])
    if len(resolved_fpcs) == len(ds_cfgs):
        per_dataset_fpcs = [int(fpc) for fpc in resolved_fpcs]
    else:
        per_dataset_fpcs = [default_fpc] * len(ds_cfgs)

    logger.info("Building %d mixed SSL sub-datasets in parallel ...", len(ds_cfgs))
    index_to_future = {}
    with ThreadPoolExecutor(max_workers=len(ds_cfgs)) as pool:
        for idx, ds_cfg in enumerate(ds_cfgs):
            future = pool.submit(_build_single_dataset, ds_cfg, per_dataset_fpcs[idx], target_fps, transform)
            index_to_future[idx] = future
        for future in as_completed(index_to_future.values()):
            exc = future.exception()
            if exc is not None:
                logger.error("Sub-dataset build failed: %s", exc)

    datasets: List[Dataset] = []
    weights: List[float] = []
    dataset_clip_lengths: List[int] = []
    dataset_types: List[str] = []
    for idx in range(len(ds_cfgs)):
        result = index_to_future[idx].result()
        if result is None:
            continue
        ds, w, ds_type, fpc = result
        datasets.append(ds)
        weights.append(w)
        dataset_clip_lengths.append(fpc)
        dataset_types.append(ds_type)
        logger.info("Mixed SSL sub-dataset: type=%s, size=%d, weight=%.3f, fpc=%d", ds_type, len(ds), w, fpc)

    if not datasets:
        raise RuntimeError("All mixed SSL sub-datasets are empty")
    return datasets, weights, dataset_clip_lengths, dataset_types


def _mixed_ssl_collate_fn(batch):
    """Collate mixed_ssl samples into a single clip tensor.

    Each sample is ``([buffer], label, [clip_indices])`` where
    ``buffer`` is ``[C, T, H, W]``. We stack buffers into ``[B, C, T, H, W]``.

    For Token AE we only need the video clips (no masks).
    Samples with different ``T`` (``frames_per_clip``) are grouped by temporal
    length and returned as a dict so training can process each temporal bucket
    without dropping samples.
    """
    # Group by temporal length
    groups: dict = {}
    for sample in batch:
        buf = sample[0][0]  # [C, T, H, W]
        t = buf.shape[1]
        if t not in groups:
            groups[t] = []
        groups[t].append(buf)

    grouped_clips = {}
    for t, bufs in sorted(groups.items()):
        grouped_clips[t] = torch.stack(bufs, dim=0)  # [B_t, C, T, H, W]
    return grouped_clips


class _RoundRobinMixedSSLLoader:
    """Round-robin mixed_ssl loader with per-step image budget.

    Each step consumes frames from exactly one dataset in rotating order.
    Frames are read sequentially from the current clip; if a clip is shorter than
    the remaining budget, subsequent clips from the same dataset are appended in
    the same step. If a clip is longer than the remaining budget, the remainder
    is emitted on later steps.
    """

    def __init__(
        self,
        datasets: List[Dataset],
        dataset_types: List[str],
        dataset_clip_lengths: List[int],
        batch_image_budget: int,
        num_workers: int,
        pin_mem: bool,
        persistent_workers: bool,
        rank: int,
        world_size: int,
    ):
        self.datasets = datasets
        self.dataset_types = dataset_types
        self.dataset_clip_lengths = dataset_clip_lengths
        self.batch_image_budget = int(batch_image_budget)
        if self.batch_image_budget <= 0:
            raise ValueError("batch_image_budget must be > 0")
        self.num_datasets = len(datasets)

        workers_per_dataset = max(1, num_workers // max(1, self.num_datasets)) if num_workers > 0 else 0
        self.samplers: List[DistributedSampler] = []
        self.loaders: List[DataLoader] = []

        for dataset in datasets:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            loader = DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                collate_fn=_single_sample_collate,
                num_workers=workers_per_dataset,
                pin_memory=pin_mem,
                persistent_workers=(workers_per_dataset > 0) and persistent_workers,
                drop_last=False,
            )
            self.samplers.append(sampler)
            self.loaders.append(loader)

        self.dataset_total_frames = [
            len(sampler) * fpc for sampler, fpc in zip(self.samplers, self.dataset_clip_lengths)
        ]
        self.steps_per_epoch = sum(
            math.ceil(total_frames / self.batch_image_budget) for total_frames in self.dataset_total_frames
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int):
        for sampler in self.samplers:
            sampler.set_epoch(epoch)

    def __iter__(self):
        iterators = [iter(loader) for loader in self.loaders]
        active_clips: List[Optional[torch.Tensor]] = [None] * self.num_datasets
        clip_offsets = [0] * self.num_datasets
        remaining_frames = list(self.dataset_total_frames)

        dataset_cursor = 0
        emitted_steps = 0
        while emitted_steps < self.steps_per_epoch:
            chosen_idx = None
            for _ in range(self.num_datasets):
                idx = dataset_cursor % self.num_datasets
                dataset_cursor += 1
                if remaining_frames[idx] > 0:
                    chosen_idx = idx
                    break

            if chosen_idx is None:
                break

            remaining_budget = self.batch_image_budget
            grouped_chunks: Dict[int, List[torch.Tensor]] = {}

            while remaining_budget > 0 and remaining_frames[chosen_idx] > 0:
                if active_clips[chosen_idx] is None:
                    sample = next(iterators[chosen_idx])
                    active_clips[chosen_idx] = sample[0][0]  # [C, T, H, W]
                    clip_offsets[chosen_idx] = 0

                clip = active_clips[chosen_idx]
                assert clip is not None

                start = clip_offsets[chosen_idx]
                end = min(start + remaining_budget, clip.shape[1])
                clip_chunk = clip[:, start:end]  # [C, T_chunk, H, W]
                consumed_frames = int(clip_chunk.shape[1])
                if consumed_frames <= 0:
                    break

                grouped_chunks.setdefault(consumed_frames, []).append(clip_chunk)

                remaining_budget -= consumed_frames
                remaining_frames[chosen_idx] -= consumed_frames
                clip_offsets[chosen_idx] = end

                if end >= clip.shape[1]:
                    active_clips[chosen_idx] = None
                    clip_offsets[chosen_idx] = 0

            if not grouped_chunks:
                break

            emitted_steps += 1

            yield {
                "dataset_idx": chosen_idx,
                "dataset_type": self.dataset_types[chosen_idx],
                "clips": {t: torch.stack(chunks, dim=0) for t, chunks in grouped_chunks.items()},
            }


def create_mixed_ssl_dataloader(
    args: dict,
    transform: Any,
    rank: int,
    world_size: int,
) -> Tuple[_RoundRobinMixedSSLLoader, _RoundRobinMixedSSLLoader]:
    """Create a DataLoader for mixed_ssl datasets (Token AE variant, no masks).

    Parameters
    ----------
    args : dict
        Full parsed YAML config.
    transform : Any
        Data augmentation transform.
    rank : int
        Current process rank.
    world_size : int
        Total number of processes.

    Returns
    -------
    Tuple[DataLoader, DistributedSampler]
    """
    cfgs_data = args.get("data", {})
    cfgs_mixed = cfgs_data.get("mixed_ssl", {})
    if not cfgs_mixed.get("enabled", False):
        raise ValueError("mixed_ssl dataloader requires data.mixed_ssl.enabled=true")

    batch_size = cfgs_data.get("batch_size", 8)
    num_workers = cfgs_data.get("num_workers", 8)
    pin_mem = cfgs_data.get("pin_mem", True)
    persistent_workers = cfgs_data.get("persistent_workers", False)
    fps = cfgs_data.get("fps", 2)
    resolved_fpcs = cfgs_data.get("dataset_fpcs", [])
    if not resolved_fpcs:
        raise ValueError("data.dataset_fpcs must be specified for mixed_ssl")
    target_fps = float(fps)

    sub_datasets, _, dataset_clip_lengths, dataset_types = _build_mixed_ssl_datasets(
        cfgs_mixed,
        resolved_fpcs,
        target_fps,
        transform,
    )

    loader = _RoundRobinMixedSSLLoader(
        datasets=sub_datasets,
        dataset_types=dataset_types,
        dataset_clip_lengths=dataset_clip_lengths,
        batch_image_budget=batch_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        world_size=world_size,
    )
    logger.info(
        "Mixed SSL Token AE round-robin loader: datasets=%s, clip_lengths=%s, steps=%d, image_budget=%d, fps=%d",
        dataset_types,
        dataset_clip_lengths,
        len(loader),
        batch_size,
        fps,
    )
    return loader, loader


# ===================================================================== #
#  Entry point                                                           #
# ===================================================================== #
def main(args, resume_preempt=False):  # noqa: C901
    """Train Token AE on frozen encoder tokens.

    Parameters
    ----------
    args : dict
        Parsed YAML config (from ``app/scaffold.py``).
    resume_preempt : bool
        Unused, kept for interface compatibility.
    """

    selection_checkpoint_epochs = (args.get("meta") or {}).get("selection_checkpoint_epochs", ())
    if selection_checkpoint_epochs:
        raise ValueError(
            "train_token_ae does not support meta.selection_checkpoint_epochs because its artifacts use "
            "token_ae_eN.pt naming rather than selector-compatible eN.pt"
        )

    # ------------------------------------------------------------------ #
    # 1. Parse config                                                     #
    # ------------------------------------------------------------------ #
    folder = args.get("folder")

    # -- META
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file_full = cfgs_meta.get("pretrain_checkpoint_full", None)
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    which_dtype = cfgs_meta.get("dtype", "bfloat16")

    if which_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif which_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    mixed_precision = dtype in (torch.bfloat16, torch.float16)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_type = cfgs_data.get("dataset_type", "")
    batch_size = cfgs_data.get("batch_size")
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size")
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    tokens_per_frame = int((crop_size // patch_size) ** 2)
    use_mixed_ssl = dataset_type == "mixed_ssl"

    # -- TRAIN
    cfgs_train = args.get("train")
    num_encoder_frames = cfgs_train.get("num_encoder_frames", 4)
    # For Token AE training, we only need observed frames
    # num_predictor_frames is not needed

    # -- TOKEN_AE config section
    cfgs_token_ae = args.get("token_ae", {})
    num_latent_tokens = cfgs_token_ae.get("num_latent_tokens", 144)
    ae_num_heads = cfgs_token_ae.get("num_heads", 16)
    ae_encoder_depth = cfgs_token_ae.get("encoder_depth", 4)
    ae_decoder_depth = cfgs_token_ae.get("decoder_depth", 4)
    ae_mlp_ratio = cfgs_token_ae.get("mlp_ratio", 4.0)
    ae_dropout = cfgs_token_ae.get("dropout", 0.0)
    ae_encoder_mode = cfgs_token_ae.get("encoder_mode", "query")

    # -- Semantic supervision config section (optional)
    cfgs_sem = args.get("semantic_supervision", {})
    sem_enabled = bool(cfgs_sem.get("enabled", False))
    if sem_enabled:
        raise NotImplementedError(
            "semantic_supervision (token-AE detection-aux / det_head) was removed in the "
            "new-framework port; keep semantic_supervision.enabled=false (token-AE trains "
            "with reconstruction loss MSE + cosine only)."
        )

    # -- LOSS
    cfgs_loss = args.get("loss", {})
    normalize_reps = cfgs_loss.get("normalize_reps", True)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    wd = float(cfgs_opt.get("weight_decay", 0.04))
    final_wd = float(cfgs_opt.get("final_weight_decay", wd))
    num_epochs = cfgs_opt.get("epochs", 100)
    warmup = cfgs_opt.get("warmup", 2)
    start_lr = cfgs_opt.get("start_lr", 1e-6)
    lr = cfgs_opt.get("lr", 5e-5)
    final_lr = cfgs_opt.get("final_lr", 0.0)
    betas = tuple(cfgs_opt.get("betas", (0.9, 0.999)))
    eps = cfgs_opt.get("eps", 1e-8)
    grad_accum_steps = cfgs_opt.get("grad_accum_steps", 1)
    ipe = cfgs_opt.get("ipe", None)

    # ------------------------------------------------------------------ #
    # 2. Distributed setup                                                #
    # ------------------------------------------------------------------ #
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    _configure_rank_logger(rank)
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    # ------------------------------------------------------------------ #
    # 3. Logging                                                          #
    # ------------------------------------------------------------------ #
    os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"token_ae_log_r{rank}.txt")
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Output folder: {folder}")

    # ------------------------------------------------------------------ #
    # 4. Init models                                                      #
    # ------------------------------------------------------------------ #
    training_config = parse_training_config(args)

    # -- Frozen encoder
    encoder, _ = init_encoder(training_config, device)
    encoder_embed_dim = get_encoder_embed_dim(encoder)
    logger.info(f"encoder_embed_dim: {encoder_embed_dim}")
    logger.info(f"tokens_per_frame: {tokens_per_frame}")
    logger.info(f"num_encoder_frames: {num_encoder_frames}")

    # -- Token AE (trainable)
    token_ae = TokenAE(
        embed_dim=encoder_embed_dim,
        tokens_per_frame=tokens_per_frame,
        num_latent_tokens=num_latent_tokens,
        num_heads=ae_num_heads,
        encoder_depth=ae_encoder_depth,
        decoder_depth=ae_decoder_depth,
        mlp_ratio=ae_mlp_ratio,
        dropout=ae_dropout,
        encoder_mode=ae_encoder_mode,
    ).to(device=device, dtype=dtype)

    # Count parameters
    total_params = sum(p.numel() for p in token_ae.parameters())
    trainable_params = sum(p.numel() for p in token_ae.parameters() if p.requires_grad)
    logger.info(f"TokenAE total params: {total_params / 1e6:.2f}M")
    logger.info(f"TokenAE trainable params: {trainable_params / 1e6:.2f}M")
    logger.info(f"TokenAE config: {token_ae.extra_repr()}")

    # -- Optional semantic supervision modules (trainable)
    # ------------------------------------------------------------------ #
    # 5. Load pretrained encoder weights                                  #
    # ------------------------------------------------------------------ #
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")

    if p_file_full is not None and os.path.exists(p_file_full):
        logger.info(f"Loading pretrained checkpoint from {p_file_full}")
        checkpoint = torch.load(p_file_full, map_location="cpu")

        if load_encoder:
            # 支持多种 encoder key: 按优先级尝试
            encoder_state = None
            for key in [context_encoder_key, "encoder", "ema_encoder", "target_encoder"]:
                if key in checkpoint:
                    encoder_state = checkpoint[key]
                    logger.info(f"Found encoder weights under key '{key}'")
                    break

            if encoder_state is not None:
                encoder_unwrapped = encoder.module if hasattr(encoder, "module") else encoder
                new_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in encoder_state.items()}
                missing, unexpected = encoder_unwrapped.load_state_dict(new_state_dict, strict=False)
                logger.info(f"Loaded encoder: missing={len(missing)}, unexpected={len(unexpected)}")
                if unexpected:
                    logger.info(f"Unexpected keys (first 10): {unexpected[:10]}")
                if missing:  # strict=False tolerates ckpt extras, but unloaded encoder params = silent random
                    raise RuntimeError(
                        f"Frozen Token-AE encoder has {len(missing)} param(s) not loaded from {p_file_full!r} "
                        f"(would stay randomly initialized): {missing[:10]}"
                    )
            else:
                raise KeyError(
                    f"No encoder weights in checkpoint {p_file_full!r} under any of "
                    f"{[context_encoder_key, 'encoder', 'ema_encoder', 'target_encoder']}; "
                    f"available: {list(checkpoint.keys())}. Refusing to train Token-AE on a random frozen encoder."
                )
        else:
            logger.warning("load_encoder=False, skipping encoder weight loading")

        del checkpoint
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Pretrained encoder weights loaded.")
    elif load_encoder:
        raise FileNotFoundError(
            f"load_encoder=True but no usable pretrained checkpoint at "
            f"meta.pretrain_checkpoint_full={p_file_full!r} — Token-AE trains on a FROZEN encoder, "
            "so a missing/random encoder is meaningless."
        )

    # ------------------------------------------------------------------ #
    # 6. Freeze encoder, convert dtype                                    #
    # ------------------------------------------------------------------ #
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    if dtype == torch.bfloat16:
        encoder = encoder.to(dtype=torch.bfloat16)
        logger.info("Converted frozen encoder to bfloat16")

    # ------------------------------------------------------------------ #
    # 7. Data loading                                                     #
    # ------------------------------------------------------------------ #
    training_config.segmentation.use_segmentation = False
    transform = create_transforms(training_config)
    logger.info("Creating training dataloader...")

    if use_mixed_ssl:
        logger.info("Using mixed_ssl dataset pipeline for Token AE training")
        unsupervised_loader, unsupervised_sampler = create_mixed_ssl_dataloader(args, transform, rank, world_size)
    else:
        unsupervised_loader, unsupervised_sampler = create_train_dataloader(
            training_config,
            rank,
            world_size,
            transform,
        )
    logger.info("Training dataloader ready")

    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    ipe_opt = math.ceil(ipe / grad_accum_steps)  # optimizer steps/epoch incl. the final partial window
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")
    logger.info(f"grad_accum_steps={grad_accum_steps}, optimizer steps per epoch: {ipe_opt}")

    # ------------------------------------------------------------------ #
    # 8. Optimizer + schedulers                                           #
    # ------------------------------------------------------------------ #
    param_groups = [
        {
            "params": [p for p in token_ae.parameters() if p.requires_grad and p.dim() >= 2],
        },
        {
            "params": [p for p in token_ae.parameters() if p.requires_grad and p.dim() < 2],
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps, weight_decay=wd)

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * ipe_opt),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=int(num_epochs * ipe_opt),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * ipe_opt),
    )

    use_grad_scaler = mixed_precision and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda") if use_grad_scaler else None

    # ------------------------------------------------------------------ #
    # 9. DDP wrap (trainable model only)                                  #
    # ------------------------------------------------------------------ #
    token_ae = DistributedDataParallel(token_ae, find_unused_parameters=False)
    # ------------------------------------------------------------------ #
    # 10. Resume from checkpoint                                          #
    # ------------------------------------------------------------------ #
    latest_path = os.path.join(folder, "token_ae_latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu")

        ae_unwrapped = token_ae.module if hasattr(token_ae, "module") else token_ae
        if "token_ae" in ckpt:
            ae_unwrapped.load_state_dict(ckpt["token_ae"])
        if "opt" in ckpt:
            optimizer.load_state_dict(ckpt["opt"])
        start_epoch = ckpt.get("epoch", 0)
        if scaler is not None and "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        if ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler"])
            logger.info("Restored LR scheduler state")
        if ckpt.get("wd_scheduler") is not None and hasattr(wd_scheduler, "load_state_dict"):
            wd_scheduler.load_state_dict(ckpt["wd_scheduler"])
            logger.info("Restored WD scheduler state")

        del ckpt
        torch.cuda.empty_cache()
        logger.info(f"Resumed from epoch {start_epoch}")

    # ------------------------------------------------------------------ #
    # 11. Forward encoder helper (frozen, no grad)                        #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward_encoder(clips):
        """Encode video clips → per-frame tokens (frozen).

        Parameters
        ----------
        clips : [B, C, T, H, W]

        Returns
        -------
        z : [B, T * tokens_per_frame, D]
        """
        B, C, T, H, W = clips.shape
        encoder_dtype = next(encoder.parameters()).dtype
        encoder_input = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        encoder_input = encoder_input.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1)
        encoder_input = encoder_input.to(dtype=encoder_dtype)
        z = encoder([encoder_input])[0]
        z = z.view(B, T, -1, z.size(-1)).flatten(1, 2)
        if normalize_reps:
            z = F.layer_norm(z, (z.size(-1),))
        if dtype == torch.bfloat16:
            z = z.to(torch.bfloat16)
        return z

    # ------------------------------------------------------------------ #
    # 12. Checkpoint save                                                 #
    # ------------------------------------------------------------------ #
    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        ae_unwrapped = token_ae.module if hasattr(token_ae, "module") else token_ae
        save_dict = {
            "token_ae": ae_unwrapped.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "scheduler": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
            "wd_scheduler": wd_scheduler.state_dict() if hasattr(wd_scheduler, "state_dict") else None,
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "config": {
                "embed_dim": encoder_embed_dim,
                "tokens_per_frame": tokens_per_frame,
                "num_latent_tokens": num_latent_tokens,
                "num_heads": ae_num_heads,
                "encoder_depth": ae_encoder_depth,
                "decoder_depth": ae_decoder_depth,
                "mlp_ratio": ae_mlp_ratio,
                "dropout": ae_dropout,
                "normalize_reps": normalize_reps,
            },
        }
        # atomic write: tmp + fsync + os.replace, so preemption mid-write can't leave a half-written
        # checkpoint that resume then loads. Save failures propagate (a lost checkpoint must stop the run).
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as _f:
            torch.save(save_dict, _f)
            _f.flush()
            os.fsync(_f.fileno())
        os.replace(tmp_path, path)
        logger.info(f"Checkpoint saved to: {path}")

    # ------------------------------------------------------------------ #
    # 13. CSV logger                                                      #
    # ------------------------------------------------------------------ #
    csv_log_path = os.path.join(folder, f"token_ae_loss_r{rank}.csv")
    csv_logger = None
    if rank == 0:
        csv_logger = CSVLogger(
            csv_log_path,
            ("%d", "epoch"),
            ("%d", "iter"),
            ("%.6f", "total_loss"),
            ("%.6f", "recon_loss"),
            ("%.6f", "cos_loss"),
            ("%.6f", "sem_tl_loss"),
            ("%.6f", "sem_obs_loss"),
            ("%.6f", "loss_avg"),
            ("%.2e", "lr"),
            ("%.2f", "mem_gb"),
            ("%.1f", "iter_ms"),
            ("%.1f", "gpu_ms"),
            ("%.1f", "data_ms"),
        )

    # ------------------------------------------------------------------ #
    # 14. Training loop                                                   #
    # ------------------------------------------------------------------ #
    logger.info(
        f"Starting Token AE training: "
        f"epochs={num_epochs}, ipe={ipe}, batch_size={batch_size}, "
        f"tokens_per_frame={tokens_per_frame}, num_latent_tokens={num_latent_tokens}, "
        f"compression={tokens_per_frame / num_latent_tokens:.1f}x"
    )

    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)
    optimizer.zero_grad()
    current_lr = start_lr
    current_wd = wd
    training_start_time = time.time()

    total_train_iters = (num_epochs - start_epoch) * ipe
    if use_mixed_ssl and hasattr(unsupervised_loader, "dataset_total_frames"):
        epoch_total_frames = int(sum(getattr(unsupervised_loader, "dataset_total_frames")))
    else:
        epoch_total_frames = -1
    if epoch_total_frames > 0:
        logger.info(
            f"Training stats: epoch_total_frames(per-rank)={epoch_total_frames}, "
            f"iters_per_epoch={ipe}, total_iters={total_train_iters}"
        )
    else:
        logger.info(f"Training stats: iters_per_epoch={ipe}, total_iters={total_train_iters}")

    global_iter_time_meter = AverageMeter()

    def run_epoch(epoch):
        nonlocal current_lr, current_wd, loader

        unsupervised_sampler.set_epoch(epoch)
        loader = iter(unsupervised_loader)
        epoch_start_time = time.time()

        if use_mixed_ssl and hasattr(unsupervised_loader, "dataset_total_frames"):
            per_ds_frames = getattr(unsupervised_loader, "dataset_total_frames")
            per_ds_types = getattr(unsupervised_loader, "dataset_types", [])
            ds_msg = ", ".join([f"{name}:{int(frames)}" for name, frames in zip(per_ds_types, per_ds_frames)])
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | total_iters={ipe} | "
                f"epoch_total_frames(per-rank)={int(sum(per_ds_frames))} | per_dataset_frames=[{ds_msg}]"
            )
        else:
            logger.info(f"Epoch {epoch + 1}/{num_epochs} | total_iters={ipe}")

        loss_meter = AverageMeter()
        recon_loss_meter = AverageMeter()
        cos_loss_meter = AverageMeter()
        sem_tl_loss_meter = AverageMeter()
        sem_obs_loss_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            # -- Load data with retry logic
            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    if epoch == start_epoch and itr == 0 and iter_retries == 0:
                        logger.info("Fetching first training batch...")
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    if iter_retries < 5:
                        logger.warning(
                            "Data loading error at epoch=%d itr=%d retry=%d: %s\n%s",
                            epoch + 1,
                            itr,
                            iter_retries,
                            e,
                            traceback.format_exc(),
                        )
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.error(
                            "Data loading failed permanently at epoch=%d itr=%d after retries: %s\n%s",
                            epoch + 1,
                            itr,
                            e,
                            traceback.format_exc(),
                        )
                        raise e

            if use_mixed_ssl:
                # round-robin loader returns one dataset chunk per step
                dataset_name = sample["dataset_type"]
                context_clips_by_t = {t: clips.to(device, non_blocking=True) for t, clips in sample["clips"].items()}
                context_clips_tensor = None
                step_total_frames = int(sum(int(t) * int(clips.shape[0]) for t, clips in context_clips_by_t.items()))
                step_total_clips = int(sum(int(clips.shape[0]) for clips in context_clips_by_t.values()))
                step_bucket_desc = ",".join(
                    [f"T{int(t)}x{int(clips.shape[0])}" for t, clips in sorted(context_clips_by_t.items())]
                )
            else:
                dataset_name = None
                if not isinstance(sample, (tuple, list)):
                    raise TypeError("Expected tuple/list batch for non-mixed dataset")
                sample_tuple = cast(Tuple[Any, ...], sample)
                context_clips_tensor = sample_tuple[0].to(device, non_blocking=True)  # [B, C, T, H, W]
                context_clips_by_t = None
                step_total_frames = int(context_clips_tensor.shape[0] * context_clips_tensor.shape[2])
                step_total_clips = int(context_clips_tensor.shape[0])
                step_bucket_desc = f"T{int(context_clips_tensor.shape[2])}x{int(context_clips_tensor.shape[0])}"
            data_elapsed = (time.time() - itr_start_time) * 1000.0

            def train_step():
                _new_lr = None
                _new_wd = None

                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    if use_mixed_ssl:
                        total_frames = 0
                        loss = None
                        recon_loss = None
                        cos_loss = None
                        sem_tl_loss = None
                        sem_obs_loss = None

                        if context_clips_by_t is None:
                            raise TypeError("Expected dict clips for mixed_ssl")

                        for T_enc, clips in context_clips_by_t.items():
                            z_target = forward_encoder(clips)
                            ae_output = token_ae(z_target, num_frames=T_enc)

                            group_frames = clips.shape[0] * T_enc
                            total_frames += group_frames

                            group_loss = ae_output["loss"] * group_frames
                            group_recon = ae_output["recon_loss"] * group_frames
                            group_cos = ae_output["cos_loss"] * group_frames

                            loss = group_loss if loss is None else loss + group_loss
                            recon_loss = group_recon if recon_loss is None else recon_loss + group_recon
                            cos_loss = group_cos if cos_loss is None else cos_loss + group_cos

                        assert total_frames > 0, "Mixed SSL batch has no valid frames"
                        assert loss is not None and recon_loss is not None and cos_loss is not None
                        loss = loss / total_frames
                        recon_loss = recon_loss / total_frames
                        cos_loss = cos_loss / total_frames
                        sem_tl_loss = loss * 0.0
                        sem_obs_loss = loss * 0.0
                    else:
                        # Use ALL frames from the clip for training
                        clips = context_clips_tensor
                        if clips is None:
                            raise TypeError("Expected Tensor context_clips for non-mixed dataset")
                        T_enc = clips.shape[2]

                        z_target = forward_encoder(clips)
                        # z_target: [B, T_enc * tokens_per_frame, D]

                        # --- Token AE forward (trainable) ---
                        # No .detach() needed: forward_encoder is already @torch.no_grad()
                        ae_output = token_ae(z_target, num_frames=T_enc)

                        loss = ae_output["loss"]
                        recon_loss = ae_output["recon_loss"]
                        cos_loss = ae_output["cos_loss"]

                        sem_tl_loss = loss * 0.0
                        sem_obs_loss = loss * 0.0
                # ---- Backward ----
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss at epoch {epoch + 1}, iter {itr}, skipping step")
                    # Still call backward with zero-multiplied loss to keep DDP allreduce in sync
                    (loss * 0).backward()
                    optimizer.zero_grad()
                else:
                    scaled_loss = loss / grad_accum_steps
                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    if (itr + 1) % grad_accum_steps == 0 or (itr + 1) == ipe:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        grad_params = [p for p in token_ae.parameters() if p.requires_grad]
                        torch.nn.utils.clip_grad_norm_(
                            grad_params,
                            max_norm=1.0,
                        )
                        if scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()

                        # Step schedulers only when optimizer steps
                        _new_lr = scheduler.step()
                        _new_wd = wd_scheduler.step()

                return (
                    float(loss.detach()),
                    float(recon_loss.detach()),
                    float(cos_loss.detach()),
                    float(sem_tl_loss.detach()),
                    float(sem_obs_loss.detach()),
                    _new_lr,
                    _new_wd,
                )

            (loss_val, recon_val, cos_val, sem_tl_val, sem_obs_val, _new_lr, _new_wd), gpu_etime_ms = gpu_timer(
                train_step
            )
            if _new_lr is not None:
                current_lr = _new_lr
            if _new_wd is not None:
                current_wd = _new_wd
            iter_elapsed = (time.time() - itr_start_time) * 1000.0

            loss_meter.update(loss_val)
            recon_loss_meter.update(recon_val)
            cos_loss_meter.update(cos_val)
            sem_tl_loss_meter.update(sem_tl_val)
            sem_obs_loss_meter.update(sem_obs_val)
            iter_time_meter.update(iter_elapsed)
            gpu_time_meter.update(gpu_etime_ms)
            data_time_meter.update(data_elapsed)
            global_iter_time_meter.update(iter_elapsed)

            # -- Logging
            if (itr % log_freq == 0) or (itr == ipe - 1):
                dataset_msg = f" [ds: {dataset_name}]" if dataset_name is not None else ""
                iter_idx = itr + 1
                remaining_epoch_iters = ipe - iter_idx
                avg_iter_sec_epoch = iter_time_meter.avg / 1000.0
                avg_iter_sec_global = global_iter_time_meter.avg / 1000.0
                eta_epoch = _format_seconds(remaining_epoch_iters * avg_iter_sec_epoch)
                done_iters = (epoch - start_epoch) * ipe + iter_idx
                remaining_total_iters = max(0, total_train_iters - done_iters)
                eta_total = _format_seconds(remaining_total_iters * avg_iter_sec_global)
                logger.info(
                    "[%d, %5d/%5d] loss: %.4f (recon: %.4f, cos: %.4f, sem_tl: %.4f, sem_obs: %.4f) "
                    "[lr: %.2e] [wd: %.2e] "
                    "[mem: %.2fGB] "
                    "[iter: %.1fms] [gpu: %.1fms] [data: %.1fms] "
                    "[step_frames:%d step_clips:%d buckets:%s] "
                    "[eta_epoch:%s eta_total:%s]%s"
                    % (
                        epoch + 1,
                        iter_idx,
                        ipe,
                        loss_meter.avg,
                        recon_loss_meter.avg,
                        cos_loss_meter.avg,
                        sem_tl_loss_meter.avg,
                        sem_obs_loss_meter.avg,
                        current_lr,
                        current_wd,
                        torch.cuda.max_memory_allocated() / 1024.0**3,
                        iter_time_meter.avg,
                        gpu_time_meter.avg,
                        data_time_meter.avg,
                        step_total_frames,
                        step_total_clips,
                        step_bucket_desc,
                        eta_epoch,
                        eta_total,
                        dataset_msg,
                    )
                )
                if csv_logger is not None:
                    csv_logger.log(
                        epoch + 1,
                        itr,
                        loss_val,
                        recon_val,
                        cos_val,
                        sem_tl_val,
                        sem_obs_val,
                        loss_meter.avg,
                        current_lr,
                        torch.cuda.max_memory_allocated() / 1024.0**3,
                        iter_time_meter.avg,
                        gpu_time_meter.avg,
                        data_time_meter.avg,
                    )

        epoch_elapsed = time.time() - epoch_start_time
        remaining_epochs = num_epochs - (epoch + 1)
        eta_by_epoch = _format_seconds(remaining_epochs * epoch_elapsed)
        total_elapsed = _format_seconds(time.time() - training_start_time)
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} done | duration={_format_seconds(epoch_elapsed)} "
            f"| elapsed={total_elapsed} | eta_total(by_epoch)={eta_by_epoch} "
            f"| epoch_loss={loss_meter.avg:.4f}"
        )

        # -- End of epoch
        logger.info(
            f"Epoch {epoch + 1} avg loss: {loss_meter.avg:.4f} "
            f"(recon: {recon_loss_meter.avg:.4f}, cos: {cos_loss_meter.avg:.4f}, "
            f"sem_tl: {sem_tl_loss_meter.avg:.4f}, sem_obs: {sem_obs_loss_meter.avg:.4f})"
        )

        return False

    def _save_latest(epoch):
        save_checkpoint(epoch + 1, latest_path)
        # Token-AE's periodic cadence is (epoch + 1)-based unlike the shared runner's epoch-based
        # cadence, so it stays line-local to remain byte-identical; runner periodic saving is disabled.
        if save_every_freq > 0 and (epoch + 1) % save_every_freq == 0:
            save_checkpoint(epoch + 1, os.path.join(folder, f"token_ae_e{epoch + 1}.pt"))

    # TrainingLoopRunner (composition); run_epoch keeps the full per-epoch body verbatim. Byte-identical.
    # Token-AE had no epoch-end distributed barrier, so the runner receives a no-op barrier.
    TrainingLoopRunner(
        start_epoch=start_epoch,
        epochs=num_epochs,
        ipe=ipe,
        rank=rank,
        checkpoint_freq=CHECKPOINT_FREQ,
        save_every_freq=0,
        barrier=lambda: None,
    ).run(
        run_epoch=run_epoch,
        save_latest=_save_latest,
        save_periodic=None,
        run_validation=None,
    )

    logger.info("Token AE training completed!")
