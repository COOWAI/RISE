"""V-JEPA main encoder + LeWM predictor training entry.

This dedicated entry keeps the original train_le-wm path intact while using the
main-encoder runtime helpers required by V-JEPA chunked image encoders and
TokenAE-compressed predictor tokens.
"""

import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_cowa_world_model.losses import MultiSubspaceSIGReg, SIGReg  # noqa: E402
from app.vjepa_cowa_world_model.models import PredProjectorMLP, ProjectorMLP  # noqa: E402
from app.vjepa_cowa_world_model.training import (  # noqa: E402
    TrainingTimer,
    add_encoder_param_groups,
    calculate_iterations_per_epoch,
    compile_models,
    create_loss_meters,
    create_optimizer_and_scheduler,
    create_train_dataloader,
    create_transforms,
    create_val_dataloader,
    create_validation_transforms,
    extract_peft_predictor_state,
    extract_portable_predictor_state,
    freeze_parameters,
    get_encoder_embed_dim,
    get_next_batch,
    init_encoder,
    init_segmentation_modules,
    is_peft_wrapped,
    load_checkpoint,
    load_clips,
    load_pretrained_checkpoint,
    log_epoch_summary,
    log_predictor_validation_metrics,
    log_trainable_parameters,
    log_training_metrics,
    maybe_run_gc,
    parse_training_config,
    resume_from_checkpoint,
    wait_for_checkpoint_save,
    wrap_ddp_models,
)
from app.vjepa_cowa_world_model.training.config import is_vjepa_main_encoder_config  # noqa: E402
from app.vjepa_cowa_world_model.training.data import resolve_navsim_validation_data_semantics  # noqa: E402
from app.vjepa_cowa_world_model.training.models import (  # noqa: E402
    configure_vjepa_encoder_trainability,
    init_predictor_runtime_with_token_ae,
    resolve_main_predictor_runtime_overrides,
    should_save_main_encoder,
)
from app.vjepa_cowa_world_model.training.pipeline import setup_run
from app.vjepa_cowa_world_model.training.predictor_aux import (  # noqa: E402
    call_predictor_with_aux,
    prepare_predictor_aux_inputs,
)
from app.vjepa_cowa_world_model.training.predictor_aux import resolve_predictor_aux_policy as _get_predictor_state_mode
from app.vjepa_cowa_world_model.training.predictor_lora import apply_predictor_lora as _apply_predictor_lora
from app.vjepa_cowa_world_model.training.predictor_lora import (
    set_predictor_lora_trainable as _set_predictor_lora_trainable,
)
from app.vjepa_cowa_world_model.training.predictor_loss import (  # noqa: E402
    compute_lewm_projected_jepa_losses_from_config,
    predictor_needs_z_ar_rollout,
)
from app.vjepa_cowa_world_model.training.predictor_parallel import (  # noqa: E402
    forward_parallel_predictor,
    maybe_register_parallel_predictor_tokens,
    use_parallel_predictor,
)
from app.vjepa_cowa_world_model.training.predictor_stepping import rollout_latent_predictions  # noqa: E402
from app.vjepa_cowa_world_model.training.predictor_validation import run_predictor_validation  # noqa: E402
from app.vjepa_cowa_world_model.training.predictor_validation_suite import (  # noqa: E402
    build_predictor_validation_suite_signature,
    flatten_predictor_validation_suite_result,
    run_predictor_validation_suite,
    select_predictor_diagnostic_metrics,
)
from app.vjepa_cowa_world_model.training.runtimes.loop_runner import (  # noqa: E402
    TrainingLoopRunner,
    resolve_periodic_checkpoint_epoch,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (  # noqa: E402
    PredictorTimelineInputs,
    build_parallel_predictor_timeline_inputs,
    build_predictor_timeline_inputs,
    forward_main_context,
    resolve_main_timeline,
)
from app.vjepa_cowa_world_model.training.services import BestPredictorTracker  # noqa: E402
from app.vjepa_cowa_world_model.training.validation_capabilities import (  # noqa: E402
    validate_validation_suite_execution_contract,
)
from src.utils.logging import get_logger, gpu_timer  # noqa: E402

log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def _project_lewm_tokens_fp32(
    projector: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    encoder_embed_dim: int,
    lewm_embed_dim: int,
    batch_size: int,
) -> torch.Tensor:
    """
    Parameters
    ----------
    projector        : LeWM projector module.
    tokens           : [B, T, D] tokens to project.
    encoder_embed_dim: D input embedding dimension.
    lewm_embed_dim   : output embedding dimension.
    batch_size       : expected batch size B.

    Returns
    -------
    torch.Tensor:
        [B, T, lewm_embed_dim] projected tokens in fp32.
    """
    if tokens.ndim != 3:
        raise ValueError(f"tokens must have shape [B, T, D], got {tuple(tokens.shape)}")
    if tokens.shape[0] != batch_size:
        raise ValueError(f"tokens batch size mismatch: got {tokens.shape[0]}, expected {batch_size}")
    if tokens.shape[-1] != encoder_embed_dim:
        raise ValueError(f"tokens embedding dimension mismatch: got {tokens.shape[-1]}, expected {encoder_embed_dim}")

    with torch.amp.autocast(device_type=tokens.device.type, enabled=False):
        projected = projector(tokens.reshape(-1, encoder_embed_dim).float())
    return projected.reshape(batch_size, -1, lewm_embed_dim)


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """All-gather across ranks while preserving gradients for the local shard."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return x
    rank = dist.get_rank()
    gathered = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, x.contiguous())
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


def _maybe_ddp(module: torch.nn.Module) -> torch.nn.Module:
    if dist.is_available() and dist.is_initialized():
        return DistributedDataParallel(module)
    return module


def _barrier_if_needed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _strip_obsolete_projector_bn_buffers(state: dict) -> dict:
    """Drop BatchNorm running-stat buffers from a saved le-wm projector state.

    The projectors now use BatchNorm(track_running_stats=False) (no running buffers) to fix the
    inplace-autograd error from being applied multiple times per step. Older checkpoints still carry
    running_mean / running_var / num_batches_tracked; dropping them lets resume stay strict on the
    actual (linear + affine) weights while tolerating the now-removed buffers.
    """
    obsolete = (".running_mean", ".running_var", ".num_batches_tracked")
    return {k: v for k, v in state.items() if not k.endswith(obsolete)}


def _load_lewm_modules_from_checkpoint(
    path: Optional[str],
    projector: torch.nn.Module,
    pred_proj: torch.nn.Module,
    sigreg: Optional[torch.nn.Module],
    rank: int,
) -> None:
    checkpoint = load_checkpoint(path)
    # fail-loud (point 19): 该函数只在 resume 路径调用，ckpt 必须存在且含 LeWM 投影头权重。
    # 缺键或形状不符直接报错，禁止 .get()→None 跳过 + strict=False 后静默保持随机初始化。
    if checkpoint is None:
        raise FileNotFoundError(f"LeWM resume checkpoint could not be loaded: {path}")

    if "lewm_projector" not in checkpoint:
        raise KeyError(f"LeWM resume checkpoint missing 'lewm_projector': {path}")
    if "lewm_pred_proj" not in checkpoint:
        raise KeyError(f"LeWM resume checkpoint missing 'lewm_pred_proj': {path}")

    _unwrap(projector).load_state_dict(
        _strip_obsolete_projector_bn_buffers(
            {
                key[7:] if key.startswith("module.") else key: value
                for key, value in checkpoint["lewm_projector"].items()
            }
        ),
        strict=True,
    )
    _unwrap(pred_proj).load_state_dict(
        _strip_obsolete_projector_bn_buffers(
            {
                key[7:] if key.startswith("module.") else key: value
                for key, value in checkpoint["lewm_pred_proj"].items()
            }
        ),
        strict=True,
    )
    # if sigreg is not None:
    if isinstance(sigreg, MultiSubspaceSIGReg):
        if "sigreg" not in checkpoint:
            raise KeyError(f"LeWM resume checkpoint missing 'sigreg' but sigreg module is enabled: {path}")
        _unwrap(sigreg).load_state_dict(
            {key[7:] if key.startswith("module.") else key: value for key, value in checkpoint["sigreg"].items()},
            strict=True,
        )
    elif sigreg is not None and "sigreg" in checkpoint:
        _unwrap(sigreg).load_state_dict(
            {key[7:] if key.startswith("module.") else key: value for key, value in checkpoint["sigreg"].items()},
            strict=True,
        )
    if rank == 0:
        logger.info("Loaded LeWM projector/pred_proj/sigreg state from %s", path)


def _save_checkpoint(
    *,
    path: str,
    epoch: int,
    loss: float,
    config,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    projector: torch.nn.Module,
    pred_proj: torch.nn.Module,
    sigreg: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    wd_scheduler,
    rank: int,
    world_size: int,
    predictor_weights_updated: bool,
    extra_state: Optional[dict] = None,
) -> None:
    if rank != 0:
        return
    save_dict = {
        "opt": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "batch_size": config.data.batch_size,
        "world_size": world_size,
        "lr": config.optimization.lr,
        "lewm_projector": _unwrap(projector).state_dict(),
        "lewm_pred_proj": _unwrap(pred_proj).state_dict(),
    }
    if isinstance(sigreg, MultiSubspaceSIGReg):
        save_dict["sigreg"] = _unwrap(sigreg).state_dict()
    if should_save_main_encoder(config):
        save_dict["encoder"] = encoder.state_dict()
    if config.train.encoder_ema:
        save_dict["target_encoder"] = target_encoder.state_dict()
    # predictor 权重在 full train / LoRA 任一情形被更新时都必须保存（predictor_weights_updated
    # 由调用方按 predictor_train or use_predictor_lora 计算），否则微调结果静默丢失。
    # "predictor" 永远是 plain key（PEFT 时为 merged 权重）；LoRA 额外存精确的 "predictor_peft"
    # 供 resume_from_checkpoint 精确恢复。
    if predictor_weights_updated:
        save_dict["predictor"] = extract_portable_predictor_state(predictor)
        if is_peft_wrapped(predictor):
            save_dict["predictor_peft"] = extract_peft_predictor_state(predictor)
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if wd_scheduler is not None:
        save_dict["wd_scheduler"] = wd_scheduler.state_dict()
    save_dict["saved_modules"] = [
        key
        for key in [
            "encoder",
            "target_encoder",
            "predictor",
            "predictor_peft",
            "lewm_projector",
            "lewm_pred_proj",
            "sigreg",
        ]
        if key in save_dict
    ]
    # Merge caller-provided extra_state (e.g. best_predictor_metrics) at top level so
    # it can be read back via load_checkpoint(...).get("best_predictor_metrics").
    if extra_state:
        save_dict.update(extra_state)
    torch.save({key: value.cpu() if hasattr(value, "cpu") else value for key, value in save_dict.items()}, path)
    logger.info("Checkpoint saved: %s (modules: %s)", path, save_dict["saved_modules"])


def _forward_predictions(
    *,
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    predictor_inputs: PredictorTimelineInputs,
    config,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    pred_actions = predictor_inputs.actions
    pred_states = predictor_inputs.states
    pred_extrinsics = predictor_inputs.extrinsics
    pred_driving_command = predictor_inputs.driving_command
    pred_ego_dynamics = predictor_inputs.ego_dynamics
    num_obs = predictor_inputs.num_observed_steps

    def _step_predictor(_z, _actions, _states, _extrinsics):
        aux_inputs = prepare_predictor_aux_inputs(
            actions=_actions,
            states=_states,
            extrinsics=_extrinsics,
            config=config,
            num_observed_steps=num_obs,
            driving_command=pred_driving_command,
            ego_dynamics=pred_ego_dynamics,
        )
        pred = call_predictor_with_aux(predictor, _z, aux_inputs)
        if runtime_normalize_reps:
            pred = F.layer_norm(pred, (pred.size(-1),))
        return pred

    num_total = z_context.size(1) // tokens_per_frame
    return rollout_latent_predictions(
        _step_predictor,
        config=config,
        z_context=z_context,
        actions=pred_actions,
        states=pred_states,
        extrinsics=pred_extrinsics,
        num_obs=num_obs,
        tokens_per_frame=tokens_per_frame,
        num_total=num_total,
        compute_tf=True,
        needs_ar_rollout=predictor_needs_z_ar_rollout(config),
        planner_only_error_context=False,
        validate_ic_prefix=False,
    )


def _step_scheduler(scheduler, wd_scheduler):
    new_lr = scheduler.step() if scheduler is not None else 0.0
    new_wd = wd_scheduler.step() if wd_scheduler is not None else 0.0
    return new_lr, new_wd


def main(args, resume_preempt=False):
    """Train LeWM predictor with a V-JEPA/V-JEPA main encoder runtime."""
    del resume_preempt
    config = parse_training_config(args)
    lewm_cfg = config.world_model
    if not lewm_cfg.enabled:
        if config.train.predictor_train:
            raise ValueError(
                "train_world_model trains ONLY the LeWM predictor (requires world_model.enabled=true + the "
                "sigreg/projector config). For the EMA/standard world-model predictor (e.g. method: ema configs "
                "with no world_model section) use train_planner_world_model with planner.use_planner=false + "
                "train.predictor_train=true (that line handles predictor-only training)."
            )
        raise ValueError("lewm.enabled must be true for train_lewm_main_encoder")
    if config.planner.use_planner:
        raise ValueError(
            "train_lewm_main_encoder currently supports predictor-only LeWM training; set planner.use_planner=false"
        )
    if config.segmentation.use_segmentation:
        raise ValueError("train_lewm_main_encoder currently expects segmentation.use_segmentation=false")

    logger.info("%s", f"{config.meta.dtype=}")
    logger.info(
        "[lewm-main-encoder] sigreg_weight=%.4f embed_dim=%d projector_hidden=%d",
        lewm_cfg.sigreg_weight,
        lewm_cfg.embed_dim,
        lewm_cfg.projector_hidden_dim,
    )

    predictor_lora_cfg = args.get("predictor_lora", {})
    use_predictor_lora = bool(predictor_lora_cfg.get("enabled", False))
    predictor_lora_train_bias = bool(predictor_lora_cfg.get("train_bias", False))
    # full train 或 LoRA 任一情形 predictor 权重都会被更新，checkpoint 必须保存 predictor。
    predictor_weights_updated = bool(config.train.predictor_train) or use_predictor_lora

    predictor_validation_enabled = bool(getattr(config.train, "predictor_validation_enabled", True))
    validate_validation_suite_execution_contract(
        config,
        line_name="world_model",
        declared_executors={"predictor"},
        active_consumers={"predictor"} if predictor_validation_enabled and predictor_weights_updated else set(),
    )

    ctx = setup_run(config)
    world_size, rank, device, ckpt_paths = ctx.world_size, ctx.rank, ctx.device, ctx.ckpt_paths
    csv_logger, tb_writer = ctx.csv_logger, ctx.tb_writer
    latest_path, resume_path = ckpt_paths["latest"], ckpt_paths["resume"]
    # Best predictor checkpoint selection (by predictor validation loss).
    best_predictor_path = os.path.join(config.meta.folder, "best_predictor.pt")
    validation_suite_enabled = bool(config.validation_suite.enabled)
    predictor_validation_signature = None
    if validation_suite_enabled:
        if config.data.navsim is None:
            raise ValueError("predictor validation suite requires data.navsim configuration")
        predictor_validation_signature = build_predictor_validation_suite_signature(
            horizons=config.validation_suite.horizons,
            expected_weights=config.validation_suite.expected_weight_by_horizon,
            validation_data_semantics=resolve_navsim_validation_data_semantics(config),
            val_roots=config.data.navsim.val_roots,
        )
    best_predictor_tracker = BestPredictorTracker(
        best_predictor_path,
        validation_signature=predictor_validation_signature,
    )

    encoder, target_encoder = init_encoder(config, device)
    encoder_embed_dim = get_encoder_embed_dim(encoder)
    logger.info("encoder_embed_dim: %d", encoder_embed_dim)

    main_tokens_override, predictor_img_size_override = resolve_main_predictor_runtime_overrides(config, encoder)
    main_full_timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=config.data.num_target_frames)
    logger.info(
        "Main encoder timeline: raw_frames=%d stride=%d predictor_steps=%d observed_steps=%d "
        "future_steps=%d tokens_per_step=%d predictor_img_size=%s",
        main_full_timeline.raw_num_frames,
        main_full_timeline.frame_stride,
        main_full_timeline.num_time_steps,
        main_full_timeline.num_observed_steps,
        main_full_timeline.num_future_steps,
        main_full_timeline.tokens_per_frame,
        predictor_img_size_override if predictor_img_size_override is not None else config.data.crop_size,
    )

    predictor, token_ae, tokens_per_frame, runtime_normalize_reps = init_predictor_runtime_with_token_ae(
        config,
        device=device,
        encoder_embed_dim=encoder_embed_dim,
        raw_tokens_per_frame_override=main_tokens_override,
        predictor_img_size_override=predictor_img_size_override,
    )
    if use_predictor_lora:
        predictor = _apply_predictor_lora(predictor, predictor_lora_cfg)
    maybe_register_parallel_predictor_tokens(
        predictor=predictor,
        config=config,
        embed_dim=encoder_embed_dim,
        future_steps=main_full_timeline.num_future_steps,
        tokens_per_frame=tokens_per_frame,
        device=device,
    )
    seg_neck, seg_head = init_segmentation_modules(
        device,
        config.segmentation.use_segmentation,
        encoder_embed_dim=encoder_embed_dim,
        num_classes=config.segmentation.num_classes,
        loss_seg_weight=config.segmentation.loss_seg_weight,
        loss_dice_weight=config.segmentation.loss_dice_weight,
    )

    projector = ProjectorMLP(
        input_dim=encoder_embed_dim,
        hidden_dim=lewm_cfg.projector_hidden_dim,
        output_dim=lewm_cfg.embed_dim,
    ).to(device)
    pred_proj = PredProjectorMLP(
        input_dim=encoder_embed_dim,
        hidden_dim=lewm_cfg.projector_hidden_dim,
        output_dim=lewm_cfg.embed_dim,
    ).to(device)
    if lewm_cfg.num_subspaces > 1:
        sigreg = MultiSubspaceSIGReg(
            embed_dim=lewm_cfg.embed_dim,
            num_subspaces=lewm_cfg.num_subspaces,
            subspace_dim=lewm_cfg.subspace_dim,
            knots=lewm_cfg.sigreg_knots,
            num_proj=lewm_cfg.sigreg_num_proj,
            init_mode=lewm_cfg.init_mode,
        ).to(device)
        logger.info(
            "[lewm-main-encoder] MultiSubspaceSIGReg: num_subspaces=%d subspace_dim=%d init_mode=%s",
            sigreg.num_subspaces,
            sigreg.subspace_dim,
            sigreg.init_mode,
        )
    else:
        sigreg = SIGReg(knots=lewm_cfg.sigreg_knots, num_proj=lewm_cfg.sigreg_num_proj).to(device)
    logger.info(
        "[lewm-main-encoder] projector params %.2fM, pred_proj params %.2fM",
        sum(parameter.numel() for parameter in projector.parameters()) / 1e6,
        sum(parameter.numel() for parameter in pred_proj.parameters()) / 1e6,
    )

    configure_vjepa_encoder_trainability(encoder, config)
    configure_vjepa_encoder_trainability(target_encoder, config, trainable=False)
    compile_models(encoder, target_encoder, predictor, seg_head, config.model.compile_model)

    transform = create_transforms(config)
    validation_transform = create_validation_transforms(config)
    train_loader, train_sampler = create_train_dataloader(config, rank, world_size, transform)
    ipe = calculate_iterations_per_epoch(config, train_loader)

    # Build the predictor validation loader. create_val_dataloader returns (None, None)
    # gracefully when no val data is configured, so validation is simply skipped then.
    predictor_val_loaders = None
    if predictor_validation_enabled:
        if validation_suite_enabled:
            predictor_val_loaders = {
                domain: create_val_dataloader(
                    config,
                    rank,
                    world_size,
                    validation_transform,
                    validation_domain=domain,
                )
                for domain in ("real", "counterfactual")
            }
            val_loader, val_sampler = predictor_val_loaders["real"]
        else:
            val_loader, val_sampler = create_val_dataloader(config, rank, world_size, validation_transform)
    else:
        val_loader, val_sampler = None, None

    optimizer, scaler, scheduler, wd_scheduler = create_optimizer_and_scheduler(
        config, encoder, predictor, seg_neck, seg_head, None, ipe
    )
    extra_proj_params = list(projector.parameters()) + list(pred_proj.parameters())
    if isinstance(sigreg, MultiSubspaceSIGReg):
        extra_proj_params += list(sigreg.parameters())
    optimizer.add_param_group(
        {
            "params": extra_proj_params,
            "lr": config.optimization.lr,
            "weight_decay": config.optimization.weight_decay,
        }
    )
    logger.info(
        "[lewm-main-encoder] Added projector + pred_proj%s to optimizer (lr=%.2e)",
        " + sigreg" if isinstance(sigreg, MultiSubspaceSIGReg) else "",
        config.optimization.lr,
    )

    models = wrap_ddp_models(
        encoder,
        target_encoder,
        predictor,
        seg_neck,
        seg_head,
        None,
        encoder_train=config.train.encoder_train,
        use_planner=False,
        use_status_for_planner=False,
        use_temporal=False,
        use_z_context=False,
    )
    encoder = models["encoder"]
    target_encoder = models["target_encoder"]
    predictor = models["predictor"]
    seg_neck = models["seg_neck"]
    seg_head = models["seg_head"]

    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        projector = torch.nn.SyncBatchNorm.convert_sync_batchnorm(projector)
        pred_proj = torch.nn.SyncBatchNorm.convert_sync_batchnorm(pred_proj)
    projector = _maybe_ddp(projector)
    pred_proj = _maybe_ddp(pred_proj)
    if isinstance(sigreg, MultiSubspaceSIGReg):
        sigreg = _maybe_ddp(sigreg)

    freeze_parameters(
        encoder,
        target_encoder,
        predictor,
        seg_neck,
        seg_head,
        None,
        encoder_train=config.train.encoder_train,
        predictor_train=(config.train.predictor_train and not use_predictor_lora),
        seg_head_train=False,
    )
    if use_predictor_lora:
        _set_predictor_lora_trainable(predictor, train_bias=predictor_lora_train_bias)
    configure_vjepa_encoder_trainability(encoder, config)
    configure_vjepa_encoder_trainability(target_encoder, config, trainable=False)
    if config.train.encoder_train:
        add_encoder_param_groups(optimizer, encoder, config.optimization.enc_lr_scale)

    predictor_state_mode = _get_predictor_state_mode(config)
    if use_predictor_lora:
        predictor_state_mode += "+lora"
    log_trainable_parameters(encoder, predictor, seg_neck, seg_head, None, optimizer, False, predictor_state_mode)

    load_pretrained_checkpoint(
        config.meta.pretrain_checkpoint_full,
        encoder,
        target_encoder,
        predictor,
        seg_neck,
        seg_head,
        None,
        load_encoder=config.meta.load_encoder and not is_vjepa_main_encoder_config(config),
        load_predictor=config.meta.load_predictor,
        load_seg=config.meta.load_seg,
        load_planner=False,
        context_encoder_key=config.meta.context_encoder_key,
        target_encoder_key=config.meta.target_encoder_key,
        rank=rank,
        world_size=world_size,
        predictor_checkpoint=config.meta.predictor_checkpoint,
    )

    start_epoch = 0
    if resume_path:
        start_epoch = resume_from_checkpoint(
            resume_path,
            encoder,
            target_encoder,
            predictor,
            seg_head,
            seg_neck,
            None,
            optimizer,
            scaler,
            scheduler,
            wd_scheduler,
            use_planner=False,
            load_planner=False,
            rank=rank,
            world_size=world_size,
            use_broadcast=config.meta.resume_broadcast,
            model_only=config.meta.resume_model_only,
        )
        _load_lewm_modules_from_checkpoint(resume_path, projector, pred_proj, sigreg, rank)
        # Restore the best predictor tracker so a resumed run does not overwrite a real
        # best_predictor.pt on its first validation. best_predictor.pt embeds
        # best_predictor_metrics at top level (via extra_state); latest.pt does not.
        if not config.meta.resume_model_only:
            best_predictor_tracker.restore(load_checkpoint)

    loader = iter(train_loader)
    loss_meter = None

    def run_epoch(epoch):
        # loader is created once before the loop (line above) and advanced by get_next_batch across
        # epochs; nonlocal preserves that single persistent iterator (it is reassigned inside).
        nonlocal loss_meter, loader
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        loss_meters = create_loss_meters()
        loss_meter = loss_meters["loss"]
        jloss_meter = loss_meters["jloss"]
        sloss_meter = loss_meters["sloss"]
        sigreg_loss_meter = loss_meters["sigreg_loss"]
        ortho_loss_meter = loss_meters["ortho_loss"]
        seg_loss_meter = loss_meters["seg_loss"]
        mask_loss_meter = loss_meters["mask_loss"]
        dice_loss_meter = loss_meters["dice_loss"]
        traj_loss_meter = loss_meters["traj_loss"]
        reg_loss_meter = loss_meters["reg_loss"]
        conf_loss_meter = loss_meters["conf_loss"]
        cover_loss_meter = loss_meters["cover_loss"]
        cls_valid_ratio_meter = loss_meters["cls_valid_ratio"]
        iter_time_meter = loss_meters["iter_time"]
        gpu_time_meter = loss_meters["gpu_time"]
        data_elapsed_time_meter = loss_meters["data_load_time"]

        # Transient NaN/Inf loss (common early in high-LR diffusion) is skipped; but PERSISTENT
        # non-finite loss is real divergence and must fail loud rather than burn the run skipping.
        consecutive_nonfinite = 0
        max_consecutive_nonfinite = 25

        for itr in range(ipe):
            timer = TrainingTimer()
            timer.start_iteration()
            loader, sample, success = get_next_batch(loader, train_loader, train_sampler, epoch)
            if not success:
                continue
            context_clips, actions, states, extrinsics, _, driving_command, ego_dynamics = load_clips(
                sample, device, False, torch.float
            )
            batch_metadata = (
                sample[-1] if isinstance(sample, (list, tuple)) and sample and isinstance(sample[-1], dict) else {}
            )
            metadata_valid_mask = batch_metadata.get("metadata_valid_mask")
            observed_metadata_valid_mask = batch_metadata.get("observed_metadata_valid_mask")
            data_elapsed_time_ms = timer.record_data_load()
            maybe_run_gc(itr, GARBAGE_COLLECT_ITR_FREQ, config.meta.sync_gc)

            def train_step():
                nonlocal consecutive_nonfinite
                new_lr, new_wd = _step_scheduler(scheduler, wd_scheduler)
                with torch.cuda.amp.autocast(dtype=config.dtype, enabled=config.mixed_precision):
                    grad_ctx = torch.enable_grad() if config.train.encoder_train else torch.no_grad()
                    with grad_ctx:
                        z_context = forward_main_context(
                            encoder,
                            context_clips,
                            config=config,
                            runtime_normalize_reps=runtime_normalize_reps,
                            token_ae=token_ae,
                        )
                    if use_parallel_predictor(config):
                        predictor_inputs = build_parallel_predictor_timeline_inputs(
                            actions=actions,
                            states=states,
                            extrinsics=extrinsics,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            encoder=encoder,
                            dt=1.0 / float(max(config.data.fps, 1)),
                            metadata_valid_mask=metadata_valid_mask,
                            observed_metadata_valid_mask=observed_metadata_valid_mask,
                        )
                    else:
                        predictor_inputs = build_predictor_timeline_inputs(
                            actions=actions,
                            states=states,
                            extrinsics=extrinsics,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            encoder=encoder,
                            dt=1.0 / float(max(config.data.fps, 1)),
                            metadata_valid_mask=metadata_valid_mask,
                            observed_metadata_valid_mask=observed_metadata_valid_mask,
                        )
                    batch_timeline = resolve_main_timeline(
                        config, encoder=encoder, num_raw_frames=context_clips.shape[2]
                    )

                    if not (z_context.shape[1] == batch_timeline.num_time_steps * tokens_per_frame):
                        raise AssertionError(
                            f"z_context shape mismatch: {z_context.shape}; "
                            f"expected {batch_timeline.num_time_steps * tokens_per_frame} tokens"
                        )
                    expected_action_steps = (
                        batch_timeline.num_time_steps
                        if use_parallel_predictor(config)
                        else batch_timeline.num_time_steps - 1
                    )
                    if not (predictor_inputs.actions.shape[1] == expected_action_steps):
                        raise AssertionError(
                            f"predictor actions shape mismatch: {predictor_inputs.actions.shape}; "
                            f"expected {expected_action_steps}"
                        )
                    if not (predictor_inputs.states.shape[1] == batch_timeline.num_time_steps):
                        raise AssertionError(
                            "assertion failed: predictor_inputs.states.shape[1] == batch_timeline.num_time_steps"
                        )
                    if not (predictor_inputs.extrinsics.shape[1] == batch_timeline.num_time_steps):
                        raise AssertionError(
                            "assertion failed: predictor_inputs.extrinsics.shape[1] == batch_timeline.num_time_steps"
                        )

                    if use_parallel_predictor(config):
                        parallel_output = forward_parallel_predictor(
                            predictor=predictor,
                            observed_tokens=z_context,
                            actions=predictor_inputs.actions,
                            states=predictor_inputs.states,
                            extrinsics=predictor_inputs.extrinsics,
                            config=config,
                            tokens_per_frame=tokens_per_frame,
                            runtime_normalize_reps=runtime_normalize_reps,
                            num_observed_steps=batch_timeline.num_observed_steps,
                            driving_command=predictor_inputs.driving_command,
                            ego_dynamics=predictor_inputs.ego_dynamics,
                        )
                        z_tf, z_ar = parallel_output.z_pred, parallel_output.z_ar
                    else:
                        z_tf, z_ar = _forward_predictions(
                            predictor=predictor,
                            z_context=z_context,
                            predictor_inputs=predictor_inputs,
                            config=config,
                            tokens_per_frame=tokens_per_frame,
                            runtime_normalize_reps=runtime_normalize_reps,
                        )

                    batch_size = z_context.shape[0]
                    total_tokens = z_context.shape[1]
                    time_steps = total_tokens // tokens_per_frame
                    z_proj = _project_lewm_tokens_fp32(
                        projector,
                        z_context,
                        encoder_embed_dim=encoder_embed_dim,
                        lewm_embed_dim=lewm_cfg.embed_dim,
                        batch_size=batch_size,
                    )

                    def project_predictor_tokens(tokens: torch.Tensor) -> torch.Tensor:
                        return _project_lewm_tokens_fp32(
                            pred_proj,
                            tokens,
                            encoder_embed_dim=encoder_embed_dim,
                            lewm_embed_dim=lewm_cfg.embed_dim,
                            batch_size=batch_size,
                        )

                    def projected_loss_fn(z: torch.Tensor, h: torch.Tensor, offset: int = tokens_per_frame):
                        target = h[:, offset : z.size(1) + offset]
                        return torch.mean(torch.abs(z - target) ** config.loss.loss_exp) / config.loss.loss_exp

                    jepa_loss, jloss, sloss = compute_lewm_projected_jepa_losses_from_config(
                        z_tf=z_tf,
                        z_ar=z_ar,
                        h_target=z_proj,
                        config=config,
                        tokens_per_frame=tokens_per_frame,
                        project_fn=project_predictor_tokens,
                        loss_fn=projected_loss_fn,
                        num_observed_steps=batch_timeline.num_observed_steps,
                    )

                    z_tgt_btd = z_proj.reshape(batch_size, time_steps, tokens_per_frame, lewm_cfg.embed_dim).mean(
                        dim=2
                    )
                    z_tgt_gathered = all_gather_with_grad(z_tgt_btd)  # (B, T, D)

                    z_tf_proj = project_predictor_tokens(z_tf)
                    z_src_btd = z_tf_proj.reshape(batch_size, -1, tokens_per_frame, lewm_cfg.embed_dim).mean(dim=2)
                    z_src_gathered = all_gather_with_grad(z_src_btd)  # (B, T, D)

                    if isinstance(sigreg, MultiSubspaceSIGReg):
                        sigreg_loss_tgt = sigreg(z_tgt_gathered)
                        sigreg_loss_src = sigreg(z_src_gathered)
                        ortho_loss = sigreg.orthogonality_loss()
                    else:
                        z_tgt_for_sigreg = z_tgt_gathered.permute(1, 0, 2).contiguous()  # (T, B, D)
                        z_src_for_sigreg = z_src_gathered.permute(1, 0, 2).contiguous()
                        sigreg_loss_tgt = sigreg(z_tgt_for_sigreg)
                        sigreg_loss_src = sigreg(z_src_for_sigreg)
                        ortho_loss = torch.tensor(0.0, device=device)

                    sigreg_loss = 0.5 * (sigreg_loss_tgt + sigreg_loss_src)
                    loss = jepa_loss + lewm_cfg.sigreg_weight * sigreg_loss + lewm_cfg.theta * ortho_loss

                nonfinite_loss = torch.isnan(loss) or torch.isinf(loss)
                if config.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if nonfinite_loss:
                    consecutive_nonfinite += 1
                    if consecutive_nonfinite > max_consecutive_nonfinite:
                        raise RuntimeError(
                            f"[epoch {epoch + 1}, iter {itr}] {consecutive_nonfinite} consecutive NaN/Inf "
                            f"losses — training has diverged; refusing to keep skipping. Lower LR / check data."
                        )
                    logger.warning(
                        "[epoch %d, iter %d] NaN/Inf loss (loss=%.4g), skipping (%d consecutive)",
                        epoch + 1,
                        itr,
                        loss.item(),
                        consecutive_nonfinite,
                    )
                    optimizer.zero_grad()
                    if config.mixed_precision:
                        # scaler.unscale_(optimizer) ran above, leaving the GradScaler
                        # in the UNSCALED stage. Without update() the next iteration's
                        # unscale_() raises. Reset the scaler (matches train_navsim_v2).
                        scaler.update()
                else:
                    consecutive_nonfinite = 0  # a finite step clears the divergence streak
                    # Clip predictor grads for full fine-tune OR LoRA. Previously only
                    # the LoRA branch was clipped, so a full predictor fine-tune ran
                    # unclipped (the projector/pred_proj/sigreg clip below does NOT cover
                    # the predictor) — same late-training spike risk as train_navsim_v2.
                    if predictor is not None and (config.train.predictor_train or use_predictor_lora):
                        torch.nn.utils.clip_grad_norm_(
                            [parameter for parameter in predictor.parameters() if parameter.requires_grad],
                            max_norm=config.optimization.grad_clip_norm,
                        )
                    clip_params = list(projector.parameters()) + list(pred_proj.parameters())
                    if isinstance(sigreg, MultiSubspaceSIGReg) and any(p.requires_grad for p in sigreg.parameters()):
                        clip_params += list(sigreg.parameters())
                    torch.nn.utils.clip_grad_norm_(clip_params, config.optimization.grad_clip_norm)
                    if config.mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                def _to_float(value):
                    return float(value.detach()) if torch.is_tensor(value) else float(value)

                return (
                    _to_float(loss),
                    _to_float(jloss),
                    _to_float(sloss),
                    _to_float(sigreg_loss),
                    _to_float(ortho_loss),
                    new_lr,
                    new_wd,
                )

            (loss, jloss, sloss, sigreg_loss, ortho_loss, new_lr, new_wd), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms, _ = timer.stop_iteration()

            zero = 0.0
            for meter, value in [
                (loss_meter, loss),
                (jloss_meter, jloss),
                (sloss_meter, sloss),
                (sigreg_loss_meter, sigreg_loss),
                (ortho_loss_meter, ortho_loss),
                (seg_loss_meter, zero),
                (mask_loss_meter, zero),
                (dice_loss_meter, zero),
                (traj_loss_meter, zero),
                (reg_loss_meter, zero),
                (conf_loss_meter, zero),
                (cover_loss_meter, zero),
                (cls_valid_ratio_meter, zero),
                (iter_time_meter, iter_elapsed_time_ms),
                (gpu_time_meter, gpu_etime_ms),
                (data_elapsed_time_meter, data_elapsed_time_ms),
            ]:
                meter.update(value)

            log_training_metrics(
                tb_writer,
                loss_meter,
                jloss_meter,
                sloss_meter,
                seg_loss_meter,
                mask_loss_meter,
                dice_loss_meter,
                traj_loss_meter,
                reg_loss_meter,
                conf_loss_meter,
                cover_loss_meter,
                iter_time_meter,
                gpu_time_meter,
                data_elapsed_time_meter,
                epoch,
                itr,
                ipe,
                rank,
                new_lr,
                new_wd,
                log_freq,
                config.planner.wta_loss_version,
                config.planner.awta_init_temperature,
                config.planner.awta_exp_base,
                config.planner.awta_min_temperature,
                config.planner.num_modes,
                planner_type=getattr(config.planner, "planner_type", "transformer"),
                cls_valid_ratio_meter=cls_valid_ratio_meter,
                sigreg_loss_meter=sigreg_loss_meter,
                ortho_loss_meter=ortho_loss_meter,
            )

        log_epoch_summary(
            tb_writer,
            csv_logger,
            loss_meter,
            jloss_meter,
            sloss_meter,
            seg_loss_meter,
            mask_loss_meter,
            dice_loss_meter,
            traj_loss_meter,
            reg_loss_meter,
            conf_loss_meter,
            iter_time_meter,
            gpu_time_meter,
            data_elapsed_time_meter,
            epoch,
            rank,
            has_validation=False,
        )

        # Phase 5: compute the validation gate here so the runner knows whether to call run_validation.
        _val_freq = int(getattr(config.meta, "val_freq", 5) or 0)
        validation_due = val_loader is not None and _val_freq > 0 and (epoch + 1) % _val_freq == 0
        return predictor_validation_enabled and validation_due

    def _save_latest(epoch):
        _save_checkpoint(
            path=latest_path,
            epoch=epoch + 1,
            loss=loss_meter.avg if loss_meter is not None else 0.0,
            config=config,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            projector=projector,
            pred_proj=pred_proj,
            sigreg=sigreg,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            rank=rank,
            world_size=world_size,
            predictor_weights_updated=predictor_weights_updated,
        )

    def _save_periodic(epoch):
        periodic_epoch = resolve_periodic_checkpoint_epoch(
            epoch,
            periodic_one_based=bool(config.meta.selection_checkpoint_epochs),
            selection_checkpoint_epochs=config.meta.selection_checkpoint_epochs,
        )
        _save_checkpoint(
            path=os.path.join(config.meta.folder, f"e{periodic_epoch}.pt"),
            epoch=epoch + 1,
            loss=loss_meter.avg if loss_meter is not None else 0.0,
            config=config,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            projector=projector,
            pred_proj=pred_proj,
            sigreg=sigreg,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            rank=rank,
            world_size=world_size,
            predictor_weights_updated=predictor_weights_updated,
        )

    def _run_validation_isolated(epoch):
        # ---- Predictor-only validation + best-by-val-loss checkpoint selection ----
        _val_freq = int(getattr(config.meta, "val_freq", 5) or 0)
        validation_due = val_loader is not None and _val_freq > 0 and (epoch + 1) % _val_freq == 0
        if predictor_validation_enabled and validation_due:
            logger.info("Running predictor-only validation at epoch %d...", epoch + 1)
            predictor_suite_result = None
            if validation_suite_enabled:
                if predictor_val_loaders is None:
                    raise RuntimeError("predictor validation suite loaders were not initialized")

                def run_predictor_domain_protocol(domain, protocol):
                    domain_loader, domain_sampler = predictor_val_loaders[domain]
                    rows = run_predictor_validation(
                        encoder=encoder,
                        target_encoder=target_encoder,
                        predictor=predictor,
                        val_loader=domain_loader,
                        val_sampler=domain_sampler,
                        config=config,
                        epoch=epoch + 1,
                        rank=rank,
                        world_size=world_size,
                        token_ae=token_ae,
                        runtime_normalize_reps=runtime_normalize_reps,
                        rollout_future_steps=protocol.horizon,
                        validation_domain=domain,
                        return_cohort_metrics=True,
                    )
                    return select_predictor_diagnostic_metrics(rows)

                predictor_suite_result = run_predictor_validation_suite(
                    run_predictor_domain_protocol,
                    horizons=config.validation_suite.horizons,
                    expected_weights=config.validation_suite.expected_weight_by_horizon,
                    metric_directions={"predictor_rollout_mse": "lower"},
                )
                predictor_val_metrics = flatten_predictor_validation_suite_result(predictor_suite_result)
            else:
                predictor_val_metrics = run_predictor_validation(
                    encoder=encoder,
                    target_encoder=target_encoder,
                    predictor=predictor,
                    val_loader=val_loader,
                    val_sampler=val_sampler,
                    config=config,
                    epoch=epoch + 1,
                    rank=rank,
                    world_size=world_size,
                    token_ae=token_ae,
                    runtime_normalize_reps=runtime_normalize_reps,
                )
            log_predictor_validation_metrics(tb_writer, predictor_val_metrics, epoch, rank)
            # predictor_val_metrics is all-reduced inside run_predictor_validation, so this
            # decision is identical across ranks; _save_checkpoint writes on rank 0 only.
            is_best_predictor = (
                best_predictor_tracker.consider_suite_result(predictor_suite_result, epoch + 1)
                if predictor_suite_result is not None
                else best_predictor_tracker.consider(
                    float(predictor_val_metrics.get("predictor_loss", float("inf"))), epoch + 1
                )
            )
            if is_best_predictor:
                _save_checkpoint(
                    path=best_predictor_path,
                    epoch=epoch + 1,
                    loss=loss_meter.avg if loss_meter is not None else 0.0,
                    config=config,
                    encoder=encoder,
                    target_encoder=target_encoder,
                    predictor=predictor,
                    projector=projector,
                    pred_proj=pred_proj,
                    sigreg=sigreg,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    wd_scheduler=wd_scheduler,
                    rank=rank,
                    world_size=world_size,
                    predictor_weights_updated=predictor_weights_updated,
                    extra_state={
                        "best_predictor_metrics": predictor_val_metrics,
                        **(
                            {"predictor_validation_signature": predictor_validation_signature}
                            if predictor_validation_signature is not None
                            else {}
                        ),
                    },
                )
                logger.info(
                    "*** New best predictor checkpoint! epoch %d | loss: %.5f ***",
                    best_predictor_tracker.epoch,
                    best_predictor_tracker.loss,
                )

    def _run_validation(epoch):
        from app.vjepa_cowa_world_model.utils.eval_determinism import preserve_eval_rng_state

        with preserve_eval_rng_state(device):
            return _run_validation_isolated(epoch)

    # Phase 5: epoch iteration + checkpoint cadence + (custom) barrier + validation dispatch owned by
    # TrainingLoopRunner; run_epoch keeps the full per-epoch body verbatim. Byte-identical.
    TrainingLoopRunner(
        start_epoch=start_epoch,
        epochs=config.optimization.epochs,
        ipe=ipe,
        rank=rank,
        checkpoint_freq=CHECKPOINT_FREQ,
        save_every_freq=config.meta.save_every_freq,
        save_from_epoch=config.meta.save_from_epoch,
        periodic_one_based=bool(config.meta.selection_checkpoint_epochs),
        selection_checkpoint_epochs=config.meta.selection_checkpoint_epochs,
        barrier=_barrier_if_needed,
    ).run(
        run_epoch=run_epoch,
        save_latest=_save_latest,
        save_periodic=_save_periodic,
        run_validation=_run_validation,
    )

    wait_for_checkpoint_save()
    if rank == 0:
        logger.info("train_lewm_main_encoder finished; final loss=%.6f", 0.0 if loss_meter is None else loss_meter.avg)
