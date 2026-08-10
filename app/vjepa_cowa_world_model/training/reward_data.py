"""Data helpers for predictor reward model training."""

from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Sampler

from app.vjepa_cowa_world_model.training.config import TrainingConfig
from app.vjepa_cowa_world_model.training.loop import make_dataloader_generator, seed_dataloader_worker
from app.vjepa_cowa_world_model.training.navsim_data import (
    NavSimWorldModelDataset,
    RootTaggedDataset,
    navsim_world_model_collate_fn,
)
from app.vjepa_cowa_world_model.training.samplers import ExactDistributedEvalSampler
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _copy_root(root: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    merged.update(dict(root))
    return merged


def _validate_reward_root_identity(root: Dict[str, Any], *, split: str, root_index: int) -> Dict[str, Any]:
    """Validate the explicit identity used by reward root tagging."""

    domain = root.get("domain")
    if domain not in {"real", "counterfactual"}:
        raise ValueError(
            f"reward.{split}_roots[{root_index}].domain must be 'real' or 'counterfactual', got {domain!r}"
        )
    name = root.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"reward.{split}_roots[{root_index}].name must be a non-empty string")
    return root


def resolve_reward_navsim_roots(config: TrainingConfig, *, split: str) -> List[Dict[str, Any]]:
    """Resolve NavSim roots for reward training or validation.

    ``reward.train_roots`` / ``reward.val_roots`` are used when present, so Adv-nuSc
    and original nuScenes/NavSim roots can be mixed without changing the existing
    ``data.navsim`` training path. If absent, this falls back to ``data.navsim``.
    """
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    navsim = config.data.navsim
    defaults = {
        "camera_name": getattr(navsim, "camera_name", "CAM_F0") if navsim is not None else "CAM_F0",
        "camera_names": getattr(navsim, "camera_names", ["CAM_F0"]) if navsim is not None else ["CAM_F0"],
        "index_cache": getattr(navsim, "index_cache", True) if navsim is not None else True,
        "window_stride": getattr(navsim, "window_stride", 1) if navsim is not None else 1,
        "max_frame_gap": getattr(navsim, "max_frame_gap", 3) if navsim is not None else 3,
    }

    reward_roots = config.reward.train_roots if split == "train" else config.reward.val_roots
    if reward_roots:
        return [
            _validate_reward_root_identity(
                _copy_root(root, defaults),
                split=split,
                root_index=root_index,
            )
            for root_index, root in enumerate(reward_roots)
        ]

    if navsim is None or not navsim.enabled:
        return []

    if split == "train":
        if not navsim.data_path or not navsim.sensor_blobs_path:
            return []
        return [
            _copy_root(
                {
                    "name": "navsim_train",
                    "domain": "real",
                    "data_path": navsim.data_path,
                    "sensor_blobs_path": navsim.sensor_blobs_path,
                    "max_scenes": navsim.max_scenes,
                },
                defaults,
            )
        ]

    if not navsim.val_data_path or not navsim.val_sensor_blobs_path:
        return []
    return [
        _copy_root(
            {
                "name": "navsim_val",
                "domain": getattr(navsim, "val_domain", None) or "real",
                "data_path": navsim.val_data_path,
                "sensor_blobs_path": navsim.val_sensor_blobs_path,
                "max_scenes": navsim.max_val_scenes,
                "window_stride": (
                    navsim.val_window_stride if navsim.val_window_stride is not None else navsim.window_stride
                ),
            },
            defaults,
        )
    ]


def create_reward_navsim_dataloader(
    config: TrainingConfig,
    *,
    split: str,
    rank: int,
    world_size: int,
    transform: Any = None,
    proposal_transform: Any = None,
) -> Tuple[Optional[DataLoader], Optional[Sampler[int]]]:
    """Create a reward-training dataloader from one or more NavSim roots."""
    roots = resolve_reward_navsim_roots(config, split=split)
    if not roots:
        if split == "val":
            logger.warning("No reward validation roots configured; validation will be skipped.")
            return None, None
        raise ValueError("No reward training roots configured. Set reward.train_roots or data.navsim paths.")
    if split == "val" and transform is not None and not bool(getattr(transform, "is_validation_transform", False)):
        raise ValueError("Reward validation requires a deterministic validation transform")

    datasets = []
    for root_index, root in enumerate(roots):
        data_path = root.get("data_path")
        sensor_blobs_path = root.get("sensor_blobs_path")
        if not data_path or not sensor_blobs_path:
            raise ValueError(f"Reward root requires data_path and sensor_blobs_path, got {root}")

        load_agent_annotations = bool(root.get("load_agent_annotations", True))
        base_dataset = NavSimWorldModelDataset(
            data_path=data_path,
            sensor_blobs_path=sensor_blobs_path,
            camera_name=root.get("camera_name", "CAM_F0"),
            camera_names=root.get("camera_names") if getattr(config.multiview, "enabled", False) else None,
            frames_per_clip=config.data.num_target_frames,
            fps=config.data.fps,
            tubelet_size=1,
            transform=transform,
            proposal_transform=proposal_transform,
            max_scenes=root.get("max_scenes"),
            action_dim=config.train.action_dim,
            index_cache=bool(root.get("index_cache", True)),
            window_stride=int(root.get("window_stride", 1)),
            max_frame_gap=int(root.get("max_frame_gap", 3)),
            load_agent_annotations=load_agent_annotations,
            is_validation=split == "val",
        )
        dataset = RootTaggedDataset(
            base_dataset,
            domain=root["domain"],
            dataset_root_name=root["name"],
            dataset_root_index=root_index,
            future_agent_geometry_valid=root["domain"] == "real" and load_agent_annotations,
        )
        datasets.append(dataset)
        logger.info(
            "Reward %s root ready: data=%s blobs=%s windows=%d",
            split,
            data_path,
            sensor_blobs_path,
            len(dataset),
        )

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    shuffle = split == "train"
    drop_last = split == "train"
    if split == "val":
        sampler = ExactDistributedEvalSampler(dataset, num_replicas=world_size, rank=rank)
    else:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        collate_fn=navsim_world_model_collate_fn,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_mem,
        persistent_workers=(config.data.num_workers > 0) and config.data.persistent_workers,
        drop_last=drop_last,
        worker_init_fn=seed_dataloader_worker,
        generator=make_dataloader_generator(rank=rank, stream=f"reward/{split}"),
    )
    logger.info("Reward %s dataloader created: roots=%d batches=%d", split, len(roots), len(loader))
    return loader, sampler
