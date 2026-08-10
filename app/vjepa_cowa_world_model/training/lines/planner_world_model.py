# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Navsim Planner 训练脚本 - 重构版

使用 training 模块简化 main() 函数。
"""

import copy
import gc
import math  # noqa: F401
import os
import random
import time  # noqa: F401
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn  # noqa: F401
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.vjepa_cowa_world_model.losses import l1_length_normalized_loss  # noqa: E402,F401
from app.vjepa_cowa_world_model.losses import awta_temperature_schedule, convert_trajectory_3d_to_nd  # noqa: E402
from app.vjepa_cowa_world_model.models import MultiModalTemporalPlanner, PETRMultiViewFusion  # noqa: E402,F401
from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output  # noqa: E402
from app.vjepa_cowa_world_model.models.trajectory_value import TemporalTrajectoryValueHead  # noqa: E402
from app.vjepa_cowa_world_model.training import (  # noqa: E402
    OPEN_LOOP_SELECTION_RULE,
    TrainingTimer,
    add_encoder_param_groups,
    add_planner_param_groups,
    build_validation_record,
    calculate_iterations_per_epoch,
    compile_models,
    create_ema_update_fn,
    create_loss_meters,
    create_momentum_scheduler,
    create_optimizer_and_scheduler,
    create_train_dataloader,
    create_transforms,
    create_val_dataloader,
    create_validation_transforms,
    freeze_parameters,
    get_encoder_embed_dim,
    get_next_batch,
    init_encoder,
    init_encoder_for_full_state_warmstart,
    init_planner,
    init_segmentation_modules,
    init_training_loop,
    is_epoch_validation_due,
    load_checkpoint,
    load_clips,
    load_multiview_fusion_from_checkpoint,
    load_pretrained_checkpoint,
    log_epoch_summary,
    log_predictor_camera_training_metrics,
    log_predictor_validation_metrics,
    log_trainable_parameters,
    log_training_metrics,
    log_training_summary,
    log_validation_metrics,
    maybe_run_gc,
    parse_training_config,
    resolve_main_predictor_runtime_overrides,
    resume_from_checkpoint,
    save_training_checkpoint,
    wait_for_checkpoint_save,
    wrap_ddp_models,
)
from app.vjepa_cowa_world_model.training.budget_control import (  # noqa: E402
    load_budget_controller_from_checkpoint,
    resolve_controller_budget_profile,
)
from app.vjepa_cowa_world_model.training.budget_oracle_collection import run_budget_oracle_collection  # noqa: E402
from app.vjepa_cowa_world_model.training.config import (  # noqa: E402
    TrainingConfig,
    is_factory_pretrained_main_encoder_config,
    resolve_main_encoder_raw_tokens_per_frame,
)
from app.vjepa_cowa_world_model.training.counterfactual_supervision import (  # noqa: E402
    build_counterfactual_sample_masks,
    distributed_mask_normalization,
)
from app.vjepa_cowa_world_model.training.cvoi_execution import CvoiValueDtypeAdapter  # noqa: E402
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_planner_integration import (  # noqa: E402
    TorchDistributedAdapter,
    build_formal_v2_navsim_e120_planner_plan_on_rank0,
    formal_v2_navsim_e120_periodic_checkpoint_path,
    initialize_formal_v2_navsim_e120_planner_runtime,
    load_formal_v2_navsim_e120_calibration_model_direct,
    load_formal_v2_navsim_e120_field_model_on_rank0,
    planner_uses_legacy_open_loop_selection,
    planner_uses_navsim_e120_runtime,
    reconcile_formal_v2_navsim_e120_resume_milestone,
    record_formal_v2_navsim_e120_optimizer_exposure,
    save_formal_v2_navsim_e120_planner_checkpoint,
)
from app.vjepa_cowa_world_model.training.cvoi_runtime import (  # noqa: E402
    apply_cvoi_planner_guidance,
    cvoi_enabled,
    cvoi_guidance_enabled,
    load_cvoi_dual_value_model,
    require_cvoi_planner_stage,
    resolve_cvoi_planner_checkpoint_paths,
    resolve_cvoi_training_rollout_horizon,
    resolve_cvoi_validation_rollout_horizon,
)
from app.vjepa_cowa_world_model.training.data import (  # noqa: E402
    resolve_navsim_validation_data_semantics,
    resolve_planner_validation_domain,
)
from app.vjepa_cowa_world_model.training.latent_value_guidance import (  # noqa: E402
    apply_latent_value_guidance,
    should_apply_value_guidance,
)
from app.vjepa_cowa_world_model.training.model_factories.encoder import (  # noqa: E402
    validate_dinov2_checkpoint_source_separation,
)
from app.vjepa_cowa_world_model.training.models import (  # noqa: E402
    configure_pretrained_image_encoder_trainability,
    init_predictor_runtime_with_token_ae,
    should_save_main_encoder,
    validate_factory_pretrained_main_encoder_load_plan,
)
from app.vjepa_cowa_world_model.training.optimizer import resolve_optimization_schedule_epochs  # noqa: E402
from app.vjepa_cowa_world_model.training.per_camera_metrics import (  # noqa: E402
    compute_latent_dit_per_camera_losses,
    compute_predictor_per_camera_jepa_losses,
)
from app.vjepa_cowa_world_model.training.pipeline import setup_run  # noqa: E402
from app.vjepa_cowa_world_model.training.planner_anchor import build_ego_relative_diffusion_anchor  # noqa: E402
from app.vjepa_cowa_world_model.training.predictor_aux import (  # noqa: E402
    call_predictor_with_aux,
    prepare_predictor_aux_inputs,
)
from app.vjepa_cowa_world_model.training.predictor_aux import (  # noqa: E402
    resolve_predictor_aux_policy as _get_predictor_state_mode,
)
from app.vjepa_cowa_world_model.training.predictor_lora import (  # noqa: E402
    apply_predictor_lora as _apply_predictor_lora,
)
from app.vjepa_cowa_world_model.training.predictor_lora import (  # noqa: E402
    set_predictor_lora_trainable as _set_predictor_lora_trainable,
)
from app.vjepa_cowa_world_model.training.predictor_loss import (  # noqa: E402
    compute_predictor_jepa_losses_from_config,
    predictor_needs_z_ar_rollout,
    predictor_supervises_ar,
    select_dynamic_rollout_prefix,
    validate_ac_transformer_dynamic_rollout_config,
)
from app.vjepa_cowa_world_model.training.predictor_parallel import (  # noqa: E402
    forward_parallel_predictor,
    maybe_register_parallel_predictor_tokens,
    use_parallel_predictor,
)
from app.vjepa_cowa_world_model.training.predictor_split_backward import (  # noqa: E402
    backward_loss_outside_autocast,
    compute_split_ar_loss,
    compute_split_tf_loss,
    run_split_tf_ar_forward_backward,
)
from app.vjepa_cowa_world_model.training.predictor_stepping import (  # noqa: E402
    rollout_latent_predictions,
    validate_empty_future_planner_conditions,
)
from app.vjepa_cowa_world_model.training.predictor_validation import run_predictor_validation  # noqa: E402
from app.vjepa_cowa_world_model.training.predictor_validation_suite import (  # noqa: E402
    build_predictor_validation_suite_signature,
    flatten_predictor_validation_suite_result,
    run_predictor_validation_suite,
    select_predictor_diagnostic_metrics,
)
from app.vjepa_cowa_world_model.training.runtimes.latent_action_runtime import (  # noqa: E402
    actions_to_relative_trajectory as _integrate_joint_actions_to_relative_trajectory,
)
from app.vjepa_cowa_world_model.training.runtimes.latent_action_runtime import (  # noqa: E402
    build_joint_action_policy_output,
    denormalize_joint_actions,
)
from app.vjepa_cowa_world_model.training.runtimes.latent_diffusion_runtime import (  # noqa: E402
    forward_latent_dit_predictor_train,
    resolve_latent_dit_sampler_params,
    sample_latent_dit_joint_action_predictor,
    sample_latent_dit_predictor,
    use_latent_dit_predictor,
    use_sampled_latent_dit_planner_input,
)
from app.vjepa_cowa_world_model.training.runtimes.loop_runner import (  # noqa: E402
    TrainingLoopRunner,
    resolve_periodic_checkpoint_epoch,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (  # noqa: E402
    build_parallel_predictor_timeline_inputs,
    build_predictor_timeline_inputs,
    enforce_cvoi_zero_future_aux,
    forward_main_context,
    resolve_main_target_tokens,
    resolve_main_timeline,
    should_reuse_context_as_target,
)
from app.vjepa_cowa_world_model.training.services import (  # noqa: E402
    BestOpenLoopTracker,
    BestPredictorTracker,
    BestValueTracker,
    build_value_metric_signature,
)
from app.vjepa_cowa_world_model.training.validation_capabilities import (  # noqa: E402
    validate_validation_suite_execution_contract,
)
from app.vjepa_cowa_world_model.training.validation_suite import (  # noqa: E402
    build_validation_suite_compatibility_signature,
    build_validation_suite_signature,
    default_validation_metric_directions,
    flatten_validation_suite_result,
    require_checkpoint_validation_signature,
    run_rollout_validation_suite,
)
from app.vjepa_cowa_world_model.training.value_planning import (  # noqa: E402
    build_value_checkpoint_state,
    build_value_future_gt_trajectory,
    compute_value_head_loss_from_batch,
    copy_value_head_to_target_,
    distributed_optimizer_step_allowed,
    maybe_update_value_target_,
    optimizer_gradients_finite_flag,
    restore_value_lifecycle,
    skip_amp_optimizer_step_,
    value_optimizer_step_succeeded,
    value_planning_enabled,
)
from app.vjepa_cowa_world_model.training.value_validation import (  # noqa: E402
    build_value_validation_signature,
    run_value_validation,
)
from app.vjepa_cowa_world_model.training.wm_aux_losses import (  # noqa: E402
    compute_wm_contrastive_loss,
    compute_wm_reward_head_loss,
    init_wm_aux_modules,
    wm_aux_enabled,
)
from app.vjepa_cowa_world_model.utils import (  # noqa: E402
    build_observed_action_trajectory_history,
    prepare_inference_consistent_status_vector,
    prepare_seg_features,
    prepare_status_feature,
    resolve_planner_status_dim,
    resolve_planner_use_drive_command,
    save_training_visualization,
    select_best_trajectory,
    visualize_trajectory,
)
from app.vjepa_cowa_world_model.utils.planner_training import (  # noqa: E402
    _resolve_action_history_dt,
    compute_planner_wta_loss,
)
from app.vjepa_cowa_world_model.val_command import run_validation  # noqa: E402
from src.utils.logging import AverageMeter, get_logger, gpu_timer  # noqa: E402,F401

log_freq = 50
CHECKPOINT_FREQ = 1  # navsim-specific: save every epoch
GARBAGE_COLLECT_ITR_FREQ = 50

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def _format_tensor_finite_summary(name, value):
    """Return a compact finite-value summary for NaN diagnostics."""
    if value is None:
        return f"{name}=None"
    if not torch.is_tensor(value):
        return f"{name}={value}"

    with torch.no_grad():
        tensor = value.detach()
        total = tensor.numel()
        shape = tuple(tensor.shape)
        if total == 0:
            return f"{name}: empty shape={shape}"

        finite_mask = torch.isfinite(tensor)
        finite_count = int(finite_mask.sum().item())
        if finite_count == 0:
            return f"{name}: finite=0/{total} shape={shape}"

        finite_values = tensor[finite_mask].float()
        return (
            f"{name}: finite={finite_count}/{total} shape={shape} "
            f"min={finite_values.min().item():.4g} "
            f"max={finite_values.max().item():.4g} "
            f"mean={finite_values.mean().item():.4g}"
        )


def _build_future_gt_trajectory(states: torch.Tensor, config, num_poses: int) -> torch.Tensor:
    """Build future ego-relative trajectory [B, P, 3] through the shared strict helper."""

    return build_value_future_gt_trajectory(states, config, num_poses=num_poses)


def _actions_to_relative_trajectory(actions: torch.Tensor) -> torch.Tensor:
    """Integrate ego-frame [dx, dy, dyaw] action deltas into [B, H, 3] trajectory."""
    return _integrate_joint_actions_to_relative_trajectory(actions)


def _maybe_set_planner_awta_temperature(config, planner, epoch: int) -> float | None:
    """Update diffusion planner aWTA temperature when a learned planner exists."""
    if planner is None:
        return None
    if not (
        getattr(config.planner, "use_planner", False) and getattr(config.planner, "planner_type", "") == "diffusion"
    ):
        return None

    cur_awta_temp = awta_temperature_schedule(
        init_temperature=config.planner.awta_init_temperature,
        epoch=epoch,
        exp_base=config.planner.awta_exp_base,
        min_temperature=config.planner.awta_min_temperature,
    )
    planner_core = planner.module if hasattr(planner, "module") else planner
    planner_core.set_awta_temperature(cur_awta_temp)
    return cur_awta_temp


def _set_module_trainable(module: torch.nn.Module, trainable: bool) -> None:
    module = module.module if hasattr(module, "module") else module
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def resolve_main_encoder_pretrained_load_flag(
    config: TrainingConfig,
    *,
    requested_load_encoder: bool,
) -> bool:
    """Return whether generic checkpoint loading should replace the main image encoder."""
    return bool(requested_load_encoder and not is_factory_pretrained_main_encoder_config(config))


@dataclass(frozen=True)
class PretrainedCheckpointLoadDecision:
    """Exact generic-checkpoint loader inputs after factory-encoder routing."""

    should_call_generic_loader: bool
    checkpoint_path: str | None
    load_encoder: bool
    load_predictor: bool
    predictor_checkpoint: str | None


def _resolve_pretrained_checkpoint_load(
    config: TrainingConfig,
    *,
    checkpoint_path: str | None,
    predictor_checkpoint: str | None,
    requested_load_encoder: bool,
    load_predictor: bool,
    load_seg: bool,
    load_planner: bool,
) -> PretrainedCheckpointLoadDecision:
    load_encoder = resolve_main_encoder_pretrained_load_flag(config, requested_load_encoder=requested_load_encoder)
    generic_checkpoint_path = checkpoint_path
    generic_predictor_checkpoint = predictor_checkpoint
    is_dinov2_factory_config = (
        is_factory_pretrained_main_encoder_config(config) and config.model.backbone == "dinov2_img_encoder"
    )
    if is_dinov2_factory_config and load_predictor:
        validate_dinov2_checkpoint_source_separation(
            config.meta.pretrain_checkpoint_full,
            predictor_checkpoint,
        )
    is_dinov2_official_source = is_dinov2_factory_config and checkpoint_path == config.meta.pretrain_checkpoint_full
    if is_dinov2_official_source:
        conflicting = [
            name
            for name, enabled in (
                ("load_seg", load_seg),
                ("load_planner", load_planner),
            )
            if enabled
        ]
        if conflicting:
            raise ValueError(
                "DINOv2 official flat meta.pretrain_checkpoint_full cannot be used for generic full-checkpoint "
                f"components: {', '.join(conflicting)}"
            )
        if load_predictor:
            if not predictor_checkpoint:
                raise ValueError(
                    "DINOv2 load_predictor=true requires an independent meta.predictor_checkpoint; "
                    "the official flat encoder checkpoint cannot supply predictor weights"
                )
            generic_checkpoint_path = predictor_checkpoint
            generic_predictor_checkpoint = None

    should_call = bool(load_encoder or load_predictor or load_seg or load_planner)
    if config.model.backbone != "dinov2_img_encoder" and checkpoint_path:
        should_call = True
    return PretrainedCheckpointLoadDecision(
        should_call_generic_loader=should_call,
        checkpoint_path=generic_checkpoint_path,
        load_encoder=load_encoder,
        load_predictor=bool(load_predictor),
        predictor_checkpoint=generic_predictor_checkpoint,
    )


def resolve_normal_pretrained_checkpoint_load(config: TrainingConfig) -> PretrainedCheckpointLoadDecision:
    """Resolve the normal EMA checkpoint load before its direct loader callsite."""
    return _resolve_pretrained_checkpoint_load(
        config,
        checkpoint_path=config.meta.pretrain_checkpoint_full,
        predictor_checkpoint=config.meta.predictor_checkpoint,
        requested_load_encoder=config.meta.load_encoder,
        load_predictor=config.meta.load_predictor,
        load_seg=config.meta.load_seg,
        load_planner=config.meta.load_planner,
    )


def resolve_anneal_pretrained_checkpoint_load(
    config: TrainingConfig,
    *,
    load_predictor: bool = True,
) -> PretrainedCheckpointLoadDecision:
    """Resolve the fresh-anneal checkpoint load before its direct loader callsite."""
    return _resolve_pretrained_checkpoint_load(
        config,
        checkpoint_path=config.optimization.anneal_ckpt,
        predictor_checkpoint=config.meta.predictor_checkpoint,
        requested_load_encoder=True,
        load_predictor=load_predictor,
        load_seg=config.meta.load_seg,
        load_planner=config.meta.load_planner,
    )


def execute_anneal_pretrained_checkpoint_load(
    config,
    decision: PretrainedCheckpointLoadDecision,
    encoder,
    target_encoder,
    predictor,
    seg_neck,
    seg_head,
    planner,
    *,
    rank: int,
    world_size: int,
    suite_checkpoint_signatures,
    predictor_checkpoint_compatibility_signatures,
) -> None:
    """Execute the fresh-anneal generic checkpoint load only when components require it."""
    if decision.should_call_generic_loader:
        load_pretrained_checkpoint(
            decision.checkpoint_path,
            encoder,
            target_encoder,
            predictor,
            seg_neck,
            seg_head,
            planner,
            load_encoder=decision.load_encoder,
            load_predictor=decision.load_predictor,
            load_seg=config.meta.load_seg,
            load_planner=config.meta.load_planner,
            context_encoder_key=config.meta.context_encoder_key,
            target_encoder_key=config.meta.target_encoder_key,
            rank=rank,
            world_size=world_size,
            predictor_checkpoint=decision.predictor_checkpoint,
            expected_full_checkpoint_signatures=suite_checkpoint_signatures,
            expected_predictor_checkpoint_signatures=(
                predictor_checkpoint_compatibility_signatures if decision.load_predictor else None
            ),
            predictor_checkpoint_signature_match="compatible",
        )


def execute_normal_pretrained_checkpoint_load(
    config,
    decision: PretrainedCheckpointLoadDecision,
    encoder,
    target_encoder,
    predictor,
    seg_neck,
    seg_head,
    planner,
    *,
    rank: int,
    world_size: int,
    suite_checkpoint_signatures,
    predictor_checkpoint_compatibility_signatures,
) -> None:
    """Execute the normal generic checkpoint load only when components require it."""
    if decision.should_call_generic_loader:
        load_pretrained_checkpoint(
            decision.checkpoint_path,
            encoder,
            target_encoder,
            predictor,
            seg_neck,
            seg_head,
            planner,
            load_encoder=decision.load_encoder,
            load_predictor=decision.load_predictor,
            load_seg=config.meta.load_seg,
            load_planner=config.meta.load_planner,
            context_encoder_key=config.meta.context_encoder_key,
            target_encoder_key=config.meta.target_encoder_key,
            rank=rank,
            world_size=world_size,
            predictor_checkpoint=decision.predictor_checkpoint,
            expected_full_checkpoint_signatures=(suite_checkpoint_signatures if config.meta.load_planner else None),
            expected_predictor_checkpoint_signatures=(
                predictor_checkpoint_compatibility_signatures if decision.load_predictor else None
            ),
            predictor_checkpoint_signature_match="compatible",
        )


def _module_state_dict(module: torch.nn.Module) -> dict:
    module = module.module if hasattr(module, "module") else module
    return module.state_dict()


def _module_tensor_payload_nbytes(module: torch.nn.Module) -> int:
    """Return parameter and buffer payload bytes owned by a module."""
    module = module.module if hasattr(module, "module") else module
    tensors = tuple(module.parameters()) + tuple(module.buffers())
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _reuse_frozen_context_encoder_as_target(
    config: Any,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
) -> torch.nn.Module:
    """Offload a redundant target and return the frozen context encoder itself."""
    if not should_reuse_context_as_target(config, encoder):
        return target_encoder
    if target_encoder is encoder:
        return encoder
    target_encoder.to(torch.device("cpu"))
    return encoder


def _copy_module_state(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    src = src.module if hasattr(src, "module") else src
    dst = dst.module if hasattr(dst, "module") else dst
    dst.load_state_dict(src.state_dict())


def _update_module_ema(src: torch.nn.Module, dst: torch.nn.Module, momentum: float) -> None:
    src = src.module if hasattr(src, "module") else src
    dst = dst.module if hasattr(dst, "module") else dst
    with torch.no_grad():
        for src_param, dst_param in zip(src.parameters(), dst.parameters()):
            dst_param.data.mul_(momentum).add_(src_param.data, alpha=1.0 - momentum)
        for src_buffer, dst_buffer in zip(src.buffers(), dst.buffers()):
            dst_buffer.copy_(src_buffer)


def _move_camera_metadata_to_device(
    sample, device: torch.device, dtype: torch.dtype, *, require_camera_geometry: bool = False
) -> dict:
    camera_metadata = {}
    metadata = sample[-1] if isinstance(sample, (list, tuple)) and sample else None
    if isinstance(metadata, dict):
        for key in ("camera_intrinsics", "camera2ego"):
            value = metadata.get(key)
            if torch.is_tensor(value):
                camera_metadata[key] = value.to(device, dtype=dtype, non_blocking=True)
        if "camera_names" in metadata:
            camera_metadata["camera_names"] = metadata["camera_names"]
    # fail-loud (point 30): 多视角融合启用时缺相机几何直接报错，与 main_encoder_runtime 对齐，
    # 禁止静默返回 {} 后用 identity 几何（每个视角几何相同会污染融合 token）。
    if require_camera_geometry:
        missing = [k for k in ("camera_intrinsics", "camera2ego") if k not in camera_metadata]
        if missing:
            raise ValueError(
                "Multi-view fusion requires camera_metadata with tensor camera_intrinsics and camera2ego, "
                f"but missing/invalid: {missing}."
            )
    return camera_metadata


def _init_multiview_fusions(config, encoder_embed_dim: int, raw_tokens_per_frame: int, device: torch.device):
    if not bool(getattr(config.multiview, "enabled", False)):
        return None, None
    if config.multiview.fusion_type != "petr_cross_attn":
        raise ValueError(f"Unsupported multiview.fusion_type={config.multiview.fusion_type!r}")

    fusion = PETRMultiViewFusion(
        embed_dim=int(encoder_embed_dim),
        tokens_per_frame=int(raw_tokens_per_frame),
        hidden_dim=int(config.multiview.hidden_dim),
        num_heads=int(config.multiview.num_heads),
        dropout=float(config.multiview.dropout),
        output_mode=str(getattr(config.multiview, "output_mode", "fused")),
    ).to(device)
    target_fusion = PETRMultiViewFusion(
        embed_dim=int(encoder_embed_dim),
        tokens_per_frame=int(raw_tokens_per_frame),
        hidden_dim=int(config.multiview.hidden_dim),
        num_heads=int(config.multiview.num_heads),
        dropout=float(config.multiview.dropout),
        output_mode=str(getattr(config.multiview, "output_mode", "fused")),
    ).to(device)
    _copy_module_state(fusion, target_fusion)
    for parameter in target_fusion.parameters():
        parameter.requires_grad = False
    target_fusion.eval()
    logger.info(
        "Initialized PETR multi-view fusion: cameras=%s raw_tokens_per_frame=%d output_mode=%s hidden_dim=%d heads=%d",
        getattr(config.data.navsim, "camera_names", None),
        int(raw_tokens_per_frame),
        str(getattr(config.multiview, "output_mode", "fused")),
        int(config.multiview.hidden_dim),
        int(config.multiview.num_heads),
    )
    return fusion, target_fusion


def main(args, resume_preempt=False):
    """Navsim Planner 主训练函数 - 重构版"""
    config = parse_training_config(args)
    validate_factory_pretrained_main_encoder_load_plan(config)
    require_cvoi_planner_stage(config)
    navsim_e120_planner_active = planner_uses_navsim_e120_runtime(config.cvoi)
    navsim_e120_planner_plan = None
    logger.info(f"{config.meta.dtype=}")

    # -- Anneal (cooldown) validation
    is_anneal = config.optimization.is_anneal
    if navsim_e120_planner_active and is_anneal:
        raise ValueError("NavSim e120 Planner forbids anneal initialization; use the signed full-state lifecycle")
    if is_anneal and not config.optimization.anneal_ckpt:
        raise ValueError("is_anneal=True requires optimization.anneal_ckpt to be set")
    if is_anneal and resume_preempt:
        # 抢占恢复时，切换到 resume_anneal 模式。
        # 从 latest checkpoint 恢复，而非从 anneal_ckpt 重新开始。
        config.optimization.resume_anneal = True
        logger.info("[Anneal] resume_preempt=True → switching to resume_anneal mode")

    # -- Predictor LoRA config (read from raw args, not in TrainingConfig)
    cfgs_predictor_lora = args.get("predictor_lora", {})
    use_predictor_lora = bool(cfgs_predictor_lora.get("enabled", False))
    predictor_lora_train_bias = bool(cfgs_predictor_lora.get("train_bias", False))
    if use_predictor_lora:
        if bool(getattr(getattr(config, "cvoi", None), "enabled", False)):
            raise ValueError("CVoI planner stages require predictor_lora.enabled=false")
        logger.info("Predictor LoRA enabled: %s", cfgs_predictor_lora)
    planner_only_mode = bool(config.planner.use_planner) and not bool(config.train.predictor_train)
    if planner_only_mode:
        logger.info(
            "Planner-only training mode detected: predictor supervision/validation and target encoder forward "
            "will be skipped; predictor rollout is still used as planner conditioning."
        )
    validate_ac_transformer_dynamic_rollout_config(
        config,
        needs_planner_or_value=value_planning_enabled(config) or wm_aux_enabled(config),
    )

    # -- Predictor planner-finetune mode --------------------------------------
    # 解冻 predictor，仅由 planner 梯度（traj_loss -> planner -> z_ar -> predictor）微调它，
    # *不* 施加 jepa_loss。复用 planner_only 的「无 jepa 监督 / 跳过 target encoder / 跳过 predictor 验证」通路，
    # 区别仅在于此模式下 predictor 是可训练的。skip_predictor_supervision 统一驱动所有「关 jepa」分支。
    predictor_planner_finetune = bool(config.train.predictor_planner_finetune)
    skip_predictor_supervision = planner_only_mode or predictor_planner_finetune
    if predictor_planner_finetune:
        if not bool(config.planner.use_planner):
            raise ValueError(
                "predictor_planner_finetune=True 需要 planner.use_planner=True："
                "predictor 仅通过 planner 梯度微调，没有 planner 就没有监督信号。"
            )
        if use_predictor_lora:
            raise ValueError("predictor_planner_finetune 与 predictor_lora 互斥：请二选一。")
        if use_latent_dit_predictor(config):
            raise ValueError(
                "predictor_planner_finetune 不支持 latent_dit predictor："
                "其训练前向需要全时域 target image latents，与跳过 target encoder 冲突。"
            )
        if not navsim_e120_planner_active and not (config.meta.load_predictor or config.meta.predictor_checkpoint):
            raise ValueError(
                "predictor_planner_finetune=True 需要先加载训练好的 predictor："
                "请设 meta.load_predictor=True 或 meta.predictor_checkpoint=<ckpt>。"
            )
        if is_anneal and config.optimization.predictor_lr_scale != 1.0:
            raise ValueError(
                "anneal/cooldown 使用 LinearDecaySchedule，会忽略 predictor_lr_scale；"
                "请关闭 anneal，或将 optimization.predictor_lr_scale 设为 1.0。"
            )
        logger.info(
            "Predictor planner-finetune mode: predictor 解冻并仅由 planner 梯度训练（无 jepa_loss），"
            "predictor_lr_scale=%s",
            config.optimization.predictor_lr_scale,
        )

    latent_dit_action_source = str(getattr(config.planner, "latent_dit_action_source", "planner")).lower()
    if latent_dit_action_source not in {"planner", "joint_action"}:
        raise ValueError(
            "planner.latent_dit_action_source must be one of ['planner', 'joint_action'], "
            f"got {latent_dit_action_source!r}"
        )
    policy_output_source = str(getattr(config.planner, "policy_output_source", "planner")).lower()
    if policy_output_source not in {"planner", "joint_action"}:
        raise ValueError(
            "planner.policy_output_source must be one of ['planner', 'joint_action'], " f"got {policy_output_source!r}"
        )
    direct_joint_action_policy = policy_output_source == "joint_action"
    if latent_dit_action_source == "joint_action":
        if not use_latent_dit_predictor(config):
            raise ValueError("planner.latent_dit_action_source='joint_action' requires latent-DiT predictor")
        if not bool(getattr(config.predictor_dit, "joint_action_enabled", False)):
            raise ValueError(
                "planner.latent_dit_action_source='joint_action' requires " "predictor_dit.joint_action_enabled=True"
            )
        logger.info(
            "latent-DiT joint action source enabled: direct action proposals are available for "
            "metrics/visualization; planner replacement is intentionally not enabled in this training path."
        )
    if direct_joint_action_policy:
        if not bool(config.planner.use_planner):
            raise ValueError(
                "planner.policy_output_source='joint_action' requires planner.use_planner=True so the "
                "policy/trajectory objective and validation pipeline are enabled."
            )
        if not use_latent_dit_predictor(config):
            raise ValueError("planner.policy_output_source='joint_action' requires train.predictor_type='latent_dit'")
        if not bool(getattr(config.predictor_dit, "joint_action_enabled", False)):
            raise ValueError(
                "planner.policy_output_source='joint_action' requires predictor_dit.joint_action_enabled=True"
            )
        if int(getattr(config.predictor_dit, "joint_action_dim", 3)) != 3:
            raise ValueError(
                "planner.policy_output_source='joint_action' requires predictor_dit.joint_action_dim=3 "
                "for ego-frame [dx, dy, dyaw] actions"
            )
        if use_sampled_latent_dit_planner_input(config):
            raise ValueError(
                "planner.policy_output_source='joint_action' does not support "
                "train.latent_dit_planner_input='sample' during training; the no-grad joint sampler is "
                "validation/inference-only, while action-head training uses latent_output.action_loss."
            )
        logger.info(
            "Direct joint action policy mode enabled: action head / joint sampler provides planner-compatible "
            "single-mode trajectories; learned planner module will not be initialized."
        )

    # predictor 权重在以下任一情形被更新，checkpoint 必须保存 predictor（含 LoRA adapter），
    # 否则微调结果静默丢失。该布尔同时驱动保存闸门与 predictor 梯度裁剪。
    predictor_weights_updated = bool(config.train.predictor_train) or predictor_planner_finetune or use_predictor_lora

    predictor_validation_enabled = bool(getattr(config.train, "predictor_validation_enabled", True))
    if (
        bool(config.validation_suite.enabled)
        and predictor_validation_enabled
        and skip_predictor_supervision
        and not bool(config.planner.use_planner)
    ):
        raise ValueError(
            "value-only/frozen-predictor training must set train.predictor_validation_enabled=false and "
            "validation_suite.enabled=false"
        )
    active_validation_consumers = set()
    if predictor_validation_enabled and not skip_predictor_supervision:
        active_validation_consumers.add("predictor")
    if bool(config.planner.use_planner):
        active_validation_consumers.add("planner")
    validate_validation_suite_execution_contract(
        config,
        line_name="planner_world_model",
        declared_executors={"predictor", "planner"},
        active_consumers=active_validation_consumers,
    )

    # Phase 1 (doc 方向 D) wm_aux 前置校验:辅助监督的意义在于训练 world model，
    # 因此要求 predictor 确实在训练、且监督形态满足各组件的输入假设(fail-loud)。
    if wm_aux_enabled(config):
        if not bool(config.train.predictor_train):
            raise ValueError(
                "wm_aux (reward_head/contrastive) 需要 train.predictor_train=True："
                "辅助监督的目标是让 world model 学决策相关结构，冻结 predictor 时无意义。"
            )
        if use_latent_dit_predictor(config):
            raise ValueError("wm_aux 暂不支持 latent_dit predictor(rollout/监督路径不同)。")
        if config.wm_aux.reward_head_weight > 0.0 and not predictor_supervises_ar(config):
            raise ValueError(
                "wm_aux.reward_head_weight > 0 需要 AR 监督(predictor_supervision_mode 含 'ar')："
                "reward head 以 z_ar(AR rollout 未来 latent)为输入。"
            )

    ctx = setup_run(config)
    world_size, rank, device, ckpt_paths = ctx.world_size, ctx.rank, ctx.device, ctx.ckpt_paths
    if rank == 0 and config.train.predictor_split_tf_ar_backward:
        logger.info(
            "[predictor_split_tf_ar_backward] enabled: TF early backward; " "AR later backward; single optimizer step"
        )
    navsim_e120_distributed = (
        TorchDistributedAdapter(rank=rank, world_size=world_size) if navsim_e120_planner_active else None
    )
    if cvoi_enabled(config) and torch.device(device).type != "cuda":
        raise RuntimeError("CVoI Planner training requires CUDA")
    csv_logger, tb_writer = ctx.csv_logger, ctx.tb_writer

    latest_path, resume_path = ckpt_paths["latest"], ckpt_paths["resume"]
    latest_path, resume_path = resolve_cvoi_planner_checkpoint_paths(
        config,
        legacy_latest_path=latest_path,
        legacy_resume_path=resume_path,
    )
    best_path = os.path.join(config.meta.folder, "best_open_loop.pt")

    validation_suite_enabled = bool(config.validation_suite.enabled)
    validation_metric_directions = default_validation_metric_directions()
    validation_suite_signature = None
    validation_suite_compatibility_signature = None
    if validation_suite_enabled:
        if config.data.navsim is None:
            raise ValueError("validation_suite requires data.navsim configuration")
        validation_suite_signature = build_validation_suite_signature(
            horizons=config.validation_suite.horizons,
            expected_weights=config.validation_suite.expected_weight_by_horizon,
            metric_directions=validation_metric_directions,
            validation_data_semantics=resolve_navsim_validation_data_semantics(config),
            val_roots=config.data.navsim.val_roots,
        )
        validation_suite_compatibility_signature = build_validation_suite_compatibility_signature(
            validation_suite_signature
        )
    best_tracker = BestOpenLoopTracker(
        best_path,
        validation_signature=validation_suite_signature,
        selection_rule="legacy",
    )
    validation_history = []

    # Best predictor checkpoint, selected by predictor validation loss.
    # The open-loop (planner) selector below only runs when use_planner=True, so
    # for predictor-only runs this is the *only* meaningful "best" signal — without
    # it the summary reports "best epoch 0 / inf" and best_open_loop.pt is never written.
    best_predictor_path = os.path.join(config.meta.folder, "best_predictor.pt")
    predictor_validation_signature = None
    predictor_validation_compatibility_signature = None
    if validation_suite_signature is not None:
        predictor_suite_signature = build_predictor_validation_suite_signature(
            horizons=config.validation_suite.horizons,
            expected_weights=config.validation_suite.expected_weight_by_horizon,
            validation_data_semantics=resolve_navsim_validation_data_semantics(config),
            val_roots=config.data.navsim.val_roots,
        )
        predictor_validation_signature = {
            "validation_suite_signature": validation_suite_signature,
            "predictor_suite_signature": predictor_suite_signature,
            "selector": {"domain": "real", "cohort": "all", "protocol": "full"},
            "metric": "predictor_loss",
        }
        predictor_validation_compatibility_signature = {
            **copy.deepcopy(predictor_validation_signature),
            "validation_suite_signature": validation_suite_compatibility_signature,
        }
    best_predictor_tracker = BestPredictorTracker(
        best_predictor_path,
        validation_signature=predictor_validation_signature,
    )
    suite_checkpoint_signatures = (
        {"validation_suite_signature": validation_suite_signature} if validation_suite_signature is not None else None
    )
    predictor_checkpoint_signatures = (
        {"predictor_validation_signature": predictor_validation_signature}
        if predictor_validation_signature is not None
        else None
    )
    predictor_checkpoint_compatibility_signatures = (
        {"predictor_validation_signature": predictor_validation_compatibility_signature}
        if predictor_validation_compatibility_signature is not None
        else None
    )
    best_value_path = os.path.join(config.meta.folder, "best_value.pt")

    if navsim_e120_planner_active:
        encoder, target_encoder = init_encoder_for_full_state_warmstart(config, device)
    else:
        encoder, target_encoder = init_encoder(config, device)
    encoder_embed_dim = get_encoder_embed_dim(encoder)
    logger.info(f"encoder_embed_dim: {encoder_embed_dim}")

    main_tokens_override, predictor_img_size_override = resolve_main_predictor_runtime_overrides(config, encoder)
    main_full_timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=config.data.num_target_frames)
    value_metric_signature = build_value_metric_signature(
        config,
        frame_stride=main_full_timeline.frame_stride,
    )
    best_value_tracker = BestValueTracker(value_metric_signature)
    value_validation_signature = None
    value_validation_compatibility_signature = None
    if value_planning_enabled(config):
        if config.data.navsim is None:
            raise ValueError("value validation provenance requires data.navsim configuration")
        value_validation_signature = build_value_validation_signature(
            value_metric_signature=value_metric_signature,
            validation_data_semantics=resolve_navsim_validation_data_semantics(config),
            val_roots=config.data.navsim.val_roots,
        )
        value_validation_compatibility_signature = value_validation_signature
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
    raw_tokens_per_frame_for_fusion = resolve_main_encoder_raw_tokens_per_frame(config, encoder)
    multiview_fusion, target_multiview_fusion = _init_multiview_fusions(
        config,
        encoder_embed_dim=encoder_embed_dim,
        raw_tokens_per_frame=raw_tokens_per_frame_for_fusion,
        device=device,
    )
    if multiview_fusion is not None and config.multiview.load_from_predictor_checkpoint:
        load_multiview_fusion_from_checkpoint(
            config.meta.predictor_checkpoint,
            multiview_fusion=multiview_fusion,
            target_multiview_fusion=target_multiview_fusion,
            rank=rank,
            world_size=world_size,
            expected_validation_signatures=predictor_checkpoint_compatibility_signatures,
            validation_signature_match="compatible",
        )
    if multiview_fusion is not None and config.multiview.freeze_fusion:
        _set_module_trainable(multiview_fusion, False)
        logger.info("Frozen multiview_fusion parameters (multiview.freeze_fusion=True)")
    if use_predictor_lora:
        predictor = _apply_predictor_lora(predictor, cfgs_predictor_lora)
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

    # 计算 num_poses (planner 现在按 frame-level 轨迹工作，不再除以 tubelet_size)
    # Diff 2: use num_observed_frames instead of hardcoded -1
    total_frames = config.data.num_target_frames
    num_poses_init = total_frames - config.train.num_observed_frames
    planner = None
    if not direct_joint_action_policy:
        planner = init_planner(
            config,
            encoder_embed_dim,
            device,
            num_poses=num_poses_init,
            tokens_per_frame_override=tokens_per_frame,
        )

    # Phase 1 (doc 方向 D): wm_aux 模块(reward head 可训练;negative provider 无参数)
    wm_aux_modules = init_wm_aux_modules(config, encoder_dim=encoder_embed_dim, device=device)
    wm_reward_head = wm_aux_modules.get("reward_head") if wm_aux_modules else None
    wm_negative_provider = wm_aux_modules.get("negative_provider") if wm_aux_modules else None
    value_head = None
    target_value_head = None
    cvoi_guidance_active = cvoi_guidance_enabled(config)
    value_guidance_active = bool(getattr(getattr(config, "value_guidance", None), "enabled", False)) and not bool(
        getattr(getattr(config, "cvoi", None), "enabled", False)
    )
    if navsim_e120_planner_active:
        if navsim_e120_distributed is None:
            raise RuntimeError("NavSim e120 Field initialization requires a distributed adapter")
        cvoi_dual_value_model = load_formal_v2_navsim_e120_field_model_on_rank0(
            loader=lambda: load_formal_v2_navsim_e120_calibration_model_direct(
                config,
                embed_dim=int(encoder_embed_dim),
                device=torch.device(device),
            ),
            rank=rank,
            distributed=navsim_e120_distributed,
            device=torch.device(device),
        )
    else:
        cvoi_dual_value_model = load_cvoi_dual_value_model(
            config,
            embed_dim=int(encoder_embed_dim),
            device=device,
        )
    cvoi_dual_value = (
        None if cvoi_dual_value_model is None else CvoiValueDtypeAdapter(cvoi_dual_value_model).to(device)
    )
    navsim_e120_planner_runtime = None
    planner_exposure_recorder = None
    value_head_trainable = False
    if value_planning_enabled(config):
        value_head = TemporalTrajectoryValueHead(
            embed_dim=int(encoder_embed_dim),
            hidden_dim=int(getattr(config.value_planning, "hidden_dim", 512)),
            dropout=0.1,
        ).to(device)
        if value_guidance_active:
            _set_module_trainable(value_head, False)
            value_head.eval()
            logger.info("value_planning: TemporalTrajectoryValueHead 已启用为 latent guidance frozen critic")
        else:
            value_head_trainable = True
            # deepcopy avoids consuming a second random initialization stream;
            # the authoritative clone still happens after online DDP sync below.
            target_value_head = copy.deepcopy(value_head).to(device)
            _set_module_trainable(target_value_head, False)
            target_value_head.eval()
            logger.info("value_planning: TemporalTrajectoryValueHead 已启用 (Variant A Method 1 / value warmup)")

    configure_pretrained_image_encoder_trainability(encoder, config)
    configure_pretrained_image_encoder_trainability(target_encoder, config, trainable=False)

    compile_models(encoder, target_encoder, predictor, seg_head, config.model.compile_model)

    transform = create_transforms(config)
    validation_transform = create_validation_transforms(config)
    train_loader, train_sampler = create_train_dataloader(config, rank, world_size, transform)
    val_loader, val_sampler = create_val_dataloader(
        config,
        rank,
        world_size,
        validation_transform,
        validation_domain=resolve_planner_validation_domain(config),
    )
    predictor_val_loader, predictor_val_sampler = val_loader, val_sampler
    predictor_val_loaders = None
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
        predictor_val_loader, predictor_val_sampler = predictor_val_loaders["real"]
    value_val_loader, value_val_sampler = val_loader, val_sampler
    if value_planning_enabled(config) and not bool(config.planner.use_planner) and not validation_suite_enabled:
        value_val_loader, value_val_sampler = create_val_dataloader(
            config,
            rank,
            world_size,
            validation_transform,
            validation_domain=None,
        )
    ipe = calculate_iterations_per_epoch(config, train_loader)

    optimizer, scaler, scheduler, wd_scheduler = create_optimizer_and_scheduler(
        config, encoder, predictor, seg_neck, seg_head, planner, ipe
    )
    if config.planner.use_planner and planner is not None:
        add_planner_param_groups(optimizer, planner)
    if multiview_fusion is not None and any(p.requires_grad for p in multiview_fusion.parameters()):
        add_planner_param_groups(optimizer, multiview_fusion)
    elif multiview_fusion is not None:
        logger.info("Skipping multiview_fusion optimizer groups: no trainable parameters")
    if wm_reward_head is not None:
        add_planner_param_groups(optimizer, wm_reward_head)
        logger.info("wm_aux: PredictorRewardHead 参数已加入 optimizer(联合训练，梯度流入 predictor)")
    if value_head is not None and not value_guidance_active:
        add_planner_param_groups(optimizer, value_head)
        logger.info("value_planning: TemporalTrajectoryValueHead 参数已加入 optimizer")
    elif value_head is not None:
        logger.info("value_guidance: TemporalTrajectoryValueHead frozen; 不加入 optimizer")

    predictor_static_graph = (
        bool(getattr(config.train, "predictor_static_graph", False)) or float(config.wm_aux.contrastive_weight) > 0.0
    )
    models = wrap_ddp_models(
        encoder,
        target_encoder,
        predictor,
        seg_neck,
        seg_head,
        planner,
        encoder_train=config.train.encoder_train,
        use_planner=config.planner.use_planner,
        use_status_for_planner=config.planner.use_status_for_planner,
        use_temporal=config.planner.use_temporal,
        use_z_context=config.planner.use_z_context,
        predictor_static_graph=predictor_static_graph,
    )
    encoder, target_encoder, predictor, seg_neck, seg_head, planner = (
        models["encoder"],
        models["target_encoder"],
        models["predictor"],
        models["seg_neck"],
        models["seg_head"],
        models["planner"],
    )
    del models
    if (
        multiview_fusion is not None
        and any(p.requires_grad for p in multiview_fusion.parameters())
        and dist.is_available()
        and dist.is_initialized()
    ):
        multiview_fusion = DistributedDataParallel(multiview_fusion, find_unused_parameters=False)
    elif multiview_fusion is not None and dist.is_available() and dist.is_initialized():
        logger.info("[DDP] Skipping DDP wrap for multiview_fusion: no parameter requires grad.")
    if wm_reward_head is not None and dist.is_available() and dist.is_initialized():
        wm_reward_head = DistributedDataParallel(wm_reward_head, find_unused_parameters=False)
    if (
        value_head is not None
        and any(p.requires_grad for p in value_head.parameters())
        and dist.is_available()
        and dist.is_initialized()
    ):
        value_head = DistributedDataParallel(value_head, find_unused_parameters=False)
    elif value_head is not None and dist.is_available() and dist.is_initialized():
        logger.info("[DDP] Skipping DDP wrap for value_head: no parameter requires grad.")
    if value_head is not None and value_guidance_active:
        value_head.eval()
    if value_head_trainable:
        if target_value_head is None:
            raise RuntimeError("trainable value head requires a target_value_head")
        # DDP constructor has synchronized the online head from rank 0.  Clone
        # only now so every non-DDP target starts bit-identically on all ranks.
        copy_value_head_to_target_(value_head, target_value_head)

    budget_controller = None
    budget_controller_active = (
        bool(getattr(getattr(config, "budget_controller", None), "enabled", False))
        and getattr(config.budget_controller, "mode", None) == "eval"
    )
    if validation_suite_enabled and bool(getattr(getattr(config, "budget_controller", None), "enabled", False)):
        raise ValueError(
            "validation_suite cannot be combined with budget_controller modes: full and fixed-horizon "
            "protocols must not be redefined by a budget profile"
        )
    if budget_controller_active:
        if planner is None:
            raise ValueError("budget_controller.mode='eval' requires planner.use_planner=true")
        budget_controller_checkpoint = getattr(config.budget_controller, "controller_checkpoint", None)
        if not budget_controller_checkpoint:
            raise ValueError(
                "budget_controller.enabled=true with mode='eval' requires " "budget_controller.controller_checkpoint"
            )
        budget_controller = load_budget_controller_from_checkpoint(budget_controller_checkpoint, device=device)
        logger.info("budget_controller: loaded eval controller from %s", budget_controller_checkpoint)

    freeze_parameters(
        encoder,
        target_encoder,
        predictor,
        seg_neck,
        seg_head,
        planner,
        encoder_train=config.train.encoder_train,
        predictor_train=((config.train.predictor_train or predictor_planner_finetune) and not use_predictor_lora),
        seg_head_train=config.train.seg_head,
    )
    if use_predictor_lora:
        _set_predictor_lora_trainable(predictor, train_bias=predictor_lora_train_bias)
    configure_pretrained_image_encoder_trainability(encoder, config)
    configure_pretrained_image_encoder_trainability(target_encoder, config, trainable=False)
    if target_multiview_fusion is not None:
        target_multiview_fusion.eval()
    if config.train.encoder_train:
        add_encoder_param_groups(optimizer, encoder, config.optimization.enc_lr_scale)

    predictor_state_mode = _get_predictor_state_mode(config)
    if use_predictor_lora:
        predictor_state_mode += "+lora"
    log_trainable_parameters(
        encoder, predictor, seg_neck, seg_head, planner, optimizer, config.planner.use_planner, predictor_state_mode
    )

    momentum_scheduler = create_momentum_scheduler(
        config.ema.ema_start,
        config.ema.ema_end,
        ipe,
        resolve_optimization_schedule_epochs(config.optimization),
    )

    reuse_context_as_target = should_reuse_context_as_target(config, encoder)
    if reuse_context_as_target:
        logger.info(
            "Reusing frozen main-encoder context tokens as predictor targets; "
            "skipping target_encoder forward_main_target()."
        )

    # -- Checkpoint loading (anneal-aware)
    fresh_anneal = bool(is_anneal and not config.optimization.resume_anneal)
    value_resume_checkpoint = None
    if navsim_e120_planner_active:
        if navsim_e120_distributed is None or planner is None:
            raise RuntimeError("NavSim e120 Planner requires a distributed adapter and planner module")
        navsim_e120_planner_plan = build_formal_v2_navsim_e120_planner_plan_on_rank0(
            config,
            rank=rank,
            distributed=navsim_e120_distributed,
            resume_path=resume_path,
        )
        navsim_e120_planner_runtime = initialize_formal_v2_navsim_e120_planner_runtime(
            navsim_e120_planner_plan,
            modules={"encoder": encoder, "predictor": predictor, "planner": planner},
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            rank=rank,
            distributed=navsim_e120_distributed,
            resume_path=navsim_e120_planner_plan.resume_checkpoint_path,
            resume_model_only=bool(config.meta.resume_model_only),
        )
        start_epoch = navsim_e120_planner_runtime.start_epoch
        planner_exposure_recorder = navsim_e120_planner_runtime.exposure
        logger.info(
            "NavSim e120 Planner lifecycle initialized: run_id=%s stage=%s mode=%s start_epoch=%d",
            navsim_e120_planner_plan.run_id,
            navsim_e120_planner_plan.stage,
            navsim_e120_planner_runtime.initialization,
            start_epoch,
        )
    elif fresh_anneal:
        # Fresh anneal: load weights from anneal_ckpt, start from epoch 0
        logger.info(f"[Anneal] Loading anneal checkpoint from: {config.optimization.anneal_ckpt}")
        anneal_load_decision = resolve_anneal_pretrained_checkpoint_load(config)
        execute_anneal_pretrained_checkpoint_load(
            config,
            anneal_load_decision,
            encoder,
            target_encoder,
            predictor,
            seg_neck,
            seg_head,
            planner,
            rank=rank,
            world_size=world_size,
            suite_checkpoint_signatures=suite_checkpoint_signatures,
            predictor_checkpoint_compatibility_signatures=predictor_checkpoint_compatibility_signatures,
        )
        start_epoch = 0
        logger.info("[Anneal] Fresh anneal start: epoch=0, scheduler not restored")
    else:
        # Normal flow (or resume_anneal): load pretrained + resume from checkpoint
        normal_load_decision = resolve_normal_pretrained_checkpoint_load(config)
        execute_normal_pretrained_checkpoint_load(
            config,
            normal_load_decision,
            encoder,
            target_encoder,
            predictor,
            seg_neck,
            seg_head,
            planner,
            rank=rank,
            world_size=world_size,
            suite_checkpoint_signatures=suite_checkpoint_signatures,
            predictor_checkpoint_compatibility_signatures=predictor_checkpoint_compatibility_signatures,
        )
        resume_validation_signatures = dict(suite_checkpoint_signatures or {})
        resume_validation_signatures.update(predictor_checkpoint_signatures or {})
        if value_head is not None and value_validation_signature is not None:
            resume_validation_signatures["value_validation_signature"] = value_validation_signature

        start_epoch = resume_from_checkpoint(
            resume_path=resume_path,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            seg_head=seg_head,
            seg_neck=seg_neck,
            planner=planner,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            use_planner=config.planner.use_planner,
            load_planner=(not config.meta.resume_model_only) or config.meta.load_planner,
            rank=rank,
            world_size=world_size,
            use_broadcast=config.meta.resume_broadcast,
            model_only=config.meta.resume_model_only,
            expected_validation_signatures=resume_validation_signatures or None,
        )
        # wm_aux reward head 恢复:extra_state 在 checkpoint 顶层合并。完整 resume
        # 时 key 缺失 → raise(同一 run 续训必须有);resume_model_only(从老
        # checkpoint warm-start)允许 fresh head 并打日志——不属于静默兜底。
        if wm_reward_head is not None and resume_path is not None:
            _resume_ckpt_for_wm = load_checkpoint(resume_path)
            if _resume_ckpt_for_wm is not None and "wm_reward_head" in _resume_ckpt_for_wm:
                _wm_head_state = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in _resume_ckpt_for_wm["wm_reward_head"].items()
                }
                (wm_reward_head.module if hasattr(wm_reward_head, "module") else wm_reward_head).load_state_dict(
                    _wm_head_state
                )
                logger.info("wm_aux: 已从 resume checkpoint 恢复 wm_reward_head")
            elif not config.meta.resume_model_only:
                raise RuntimeError(
                    f"wm_aux.reward_head_weight > 0 但 resume checkpoint 缺少 'wm_reward_head'({resume_path})；"
                    "完整 resume 不允许静默重置 reward head。"
                )
            else:
                logger.info("wm_aux: resume_model_only 模式且 checkpoint 无 wm_reward_head — 头部从头训练")

        if value_head is not None and resume_path is not None:
            value_resume_checkpoint = load_checkpoint(resume_path)
            if value_resume_checkpoint is None:
                raise FileNotFoundError(f"value resume checkpoint does not exist: {resume_path}")
            if value_validation_signature is not None:
                require_checkpoint_validation_signature(
                    value_resume_checkpoint,
                    key="value_validation_signature",
                    expected=value_validation_signature,
                    checkpoint_name=resume_path,
                )

    if reuse_context_as_target:
        redundant_target_bytes = _module_tensor_payload_nbytes(target_encoder)
        target_encoder = _reuse_frozen_context_encoder_as_target(config, encoder, target_encoder)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "Released redundant frozen target_encoder after pretrained/resume restoration; "
            "target_encoder now aliases encoder (tensor payload %.2f MiB).",
            redundant_target_bytes / (1024**2),
        )

    update_ema = create_ema_update_fn(encoder, target_encoder)
    if multiview_fusion is not None and target_multiview_fusion is not None:
        base_update_ema = update_ema

        def update_ema(momentum):
            base_update_ema(momentum)
            _update_module_ema(multiview_fusion, target_multiview_fusion, momentum)

    # Value lifecycle is deliberately outside the fresh-anneal/normal branch.
    # Fresh anneal must not resume optimizer state, but it must still honor an
    # explicit value checkpoint (and frozen guidance must never use a random head).
    explicit_value_checkpoint = None
    if value_head is not None and value_resume_checkpoint is None and config.meta.value_checkpoint is not None:
        explicit_value_checkpoint = load_checkpoint(config.meta.value_checkpoint)
        if explicit_value_checkpoint is None:
            raise FileNotFoundError(f"meta.value_checkpoint does not exist: {config.meta.value_checkpoint}")
        if value_validation_signature is not None:
            require_checkpoint_validation_signature(
                explicit_value_checkpoint,
                key="value_validation_signature",
                expected=value_validation_compatibility_signature,
                checkpoint_name=config.meta.value_checkpoint,
                match_mode="compatible",
            )
    anneal_value_checkpoint = None
    if value_head is not None and fresh_anneal and explicit_value_checkpoint is None:
        anneal_value_checkpoint = load_checkpoint(config.optimization.anneal_ckpt)
        if anneal_value_checkpoint is None:
            raise FileNotFoundError(f"optimization.anneal_ckpt does not exist: {config.optimization.anneal_ckpt}")
        if value_validation_signature is not None:
            require_checkpoint_validation_signature(
                anneal_value_checkpoint,
                key="value_validation_signature",
                expected=value_validation_signature,
                checkpoint_name=config.optimization.anneal_ckpt,
            )
    if value_head is not None:
        loaded_value_source = restore_value_lifecycle(
            online_value_head=value_head,
            target_value_head=target_value_head,
            trainable=value_head_trainable,
            guidance_active=value_guidance_active,
            resume_checkpoint=value_resume_checkpoint,
            resume_source=resume_path if value_resume_checkpoint is not None else None,
            resume_model_only=bool(config.meta.resume_model_only),
            value_checkpoint=explicit_value_checkpoint,
            value_source=(config.meta.value_checkpoint if explicit_value_checkpoint is not None else None),
            best_value_tracker=best_value_tracker,
            fresh_anneal=fresh_anneal,
            anneal_checkpoint=anneal_value_checkpoint,
            anneal_source=(config.optimization.anneal_ckpt if anneal_value_checkpoint is not None else None),
        )
        if loaded_value_source is not None:
            logger.info("value_planning: restored value lifecycle from %s", loaded_value_source)
    elif value_guidance_active:
        raise RuntimeError("value_guidance.enabled=true requires an initialized value_head")

    # Dedicated best files carry the newest selection history. Do not restore
    # them for a fresh anneal, which intentionally starts a new training history.
    if planner_uses_legacy_open_loop_selection(config.cvoi) and not fresh_anneal and not config.meta.resume_model_only:
        best_tracker.restore(load_checkpoint)
        best_predictor_tracker.restore(load_checkpoint)
        best_value_checkpoint = load_checkpoint(best_value_path)
        if best_value_checkpoint is not None:
            if value_validation_signature is not None:
                require_checkpoint_validation_signature(
                    best_value_checkpoint,
                    key="value_validation_signature",
                    expected=value_validation_signature,
                    checkpoint_name=best_value_path,
                )
            tracker_state = best_value_checkpoint.get("best_value_tracker_state")
            if tracker_state is None:
                raise RuntimeError(f"best value checkpoint lacks best_value_tracker_state: {best_value_path}")
            best_value_tracker.load_state_dict(tracker_state)

    if budget_controller_active:
        logger.info("[budget_controller][eval] running one frozen-policy validation pass")
        return run_validation(
            encoder=encoder,
            predictor=predictor,
            planner=planner,
            val_loader=val_loader,
            val_sampler=val_sampler,
            config=config,
            epoch=0,
            rank=rank,
            world_size=world_size,
            use_tubelet_repeat=config.data.use_tubelet_repeat,
            vis_output_dir=os.path.join(config.meta.folder, "eval_vis"),
            token_ae=token_ae,
            runtime_normalize_reps=runtime_normalize_reps,
            multiview_fusion=multiview_fusion,
            value_head=value_head,
            cvoi_dual_value=cvoi_dual_value,
            budget_controller=budget_controller,
        )

    if (
        bool(getattr(getattr(config, "budget_controller", None), "enabled", False))
        and getattr(config.budget_controller, "mode", None) == "online_grpo"
    ):
        from app.vjepa_cowa_world_model.training.budget_controller_online_training import (
            configure_online_grpo_preempt_resume,
            train_budget_controller_online_grpo,
        )

        configure_online_grpo_preempt_resume(config, resume_preempt=resume_preempt)
        logger.info("[budget_controller][online_grpo] starting frozen-policy online rollout training")
        return train_budget_controller_online_grpo(
            config=config,
            encoder=encoder,
            predictor=predictor,
            planner=planner,
            train_loader=train_loader,
            train_sampler=train_sampler,
            device=device,
            rank=rank,
            world_size=world_size,
            tokens_per_frame=tokens_per_frame,
            runtime_normalize_reps=runtime_normalize_reps,
            token_ae=token_ae,
            multiview_fusion=multiview_fusion,
        )

    if (
        bool(getattr(getattr(config, "budget_controller", None), "enabled", False))
        and getattr(config.budget_controller, "mode", None) == "oracle_collection"
    ):
        logger.info("[budget_oracle] running Stage3B fixed-policy budget sweep")
        run_budget_oracle_collection(
            encoder=encoder,
            predictor=predictor,
            planner=planner,
            val_loader=val_loader,
            val_sampler=val_sampler,
            config=config,
            rank=rank,
            world_size=world_size,
            token_ae=token_ae,
            runtime_normalize_reps=runtime_normalize_reps,
            multiview_fusion=multiview_fusion,
            value_head=value_head,
        )
        return

    # -- Fast-forward momentum scheduler to match start_epoch
    for _ in range(start_epoch * ipe):
        next(momentum_scheduler)

    def save_checkpoint_fn(epoch, path, extra_state=None, *, replace=None):
        if navsim_e120_planner_active:
            if navsim_e120_planner_plan is None or navsim_e120_planner_runtime is None or planner is None:
                raise RuntimeError("NavSim e120 Planner checkpoint save requires initialized strict runtime state")
            if extra_state is not None:
                raise ValueError("NavSim e120 Planner checkpoints reject generic extra_state")
            if type(replace) is not bool:
                raise ValueError("NavSim e120 Planner checkpoint save requires explicit replace semantics")
            return save_formal_v2_navsim_e120_planner_checkpoint(
                plan=navsim_e120_planner_plan,
                modules={"encoder": encoder, "predictor": predictor, "planner": planner},
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                epoch=epoch,
                exposure=navsim_e120_planner_runtime.exposure,
                device=torch.device(device),
                rank=rank,
                distributed=navsim_e120_distributed,
                path=path,
                replace=replace,
            )
        checkpoint_extra_state = dict(extra_state or {})
        if validation_suite_signature is not None:
            checkpoint_extra_state["validation_suite_signature"] = validation_suite_signature
        if predictor_validation_signature is not None:
            checkpoint_extra_state["predictor_validation_signature"] = predictor_validation_signature
        if multiview_fusion is not None:
            checkpoint_extra_state["multiview_fusion"] = _module_state_dict(multiview_fusion)
            if target_multiview_fusion is not None:
                checkpoint_extra_state["target_multiview_fusion"] = _module_state_dict(target_multiview_fusion)
        if wm_reward_head is not None:
            checkpoint_extra_state["wm_reward_head"] = _module_state_dict(wm_reward_head)
        if value_head is not None:
            checkpoint_extra_state.update(
                build_value_checkpoint_state(
                    online_value_head=value_head,
                    target_value_head=target_value_head,
                    trainable=value_head_trainable,
                )
            )
            checkpoint_extra_state["best_value_tracker_state"] = best_value_tracker.state_dict()
            if value_validation_signature is not None:
                checkpoint_extra_state["value_validation_signature"] = value_validation_signature
        save_training_checkpoint(
            path=path,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            seg_neck=seg_neck,
            seg_head=seg_head,
            planner=planner,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            epoch=epoch,
            loss=loss_meter.avg,
            batch_size=config.data.batch_size,
            world_size=world_size,
            lr=config.optimization.lr,
            rank=rank,
            use_planner=config.planner.use_planner,
            encoder_train=should_save_main_encoder(config),
            encoder_ema=config.train.encoder_ema,
            predictor_train=predictor_weights_updated,
            seg_head_train=config.train.seg_head,
            extra_state=checkpoint_extra_state or None,
        )

    # Diff 12: log_optimizer_group_lrs helper
    def log_optimizer_group_lrs(epoch, itr):
        if rank != 0:
            return
        group_logs = []
        for group_idx, group in enumerate(optimizer.param_groups):
            params = group.get("params", [])
            num_tensors = len(params)
            num_params = sum(p.numel() for p in params if p is not None)
            group_logs.append(
                f"g{group_idx}: lr={group.get('lr', 0.0):.8e}, "
                f"lr_scale={group.get('lr_scale', 1.0):.4g}, "
                f"wd={group.get('weight_decay', 0.0):.8e}, "
                f"tensors={num_tensors}, params={num_params / 1e6:.2f}M"
            )
        logger.info(f"Epoch {epoch + 1} Iter {itr + 1} optimizer param group LRs:\n" + "\n".join(group_logs))

    logger.info("初始化数据加载器...")
    loader, _ = init_training_loop(
        train_loader, train_sampler, start_epoch, config.meta.skip_batches, config.meta.sync_gc
    )

    # Diff 2: num_poses uses num_observed_frames (consistent with num_poses_init above)
    num_poses = config.data.num_target_frames - config.train.num_observed_frames
    action_history_dt = _resolve_action_history_dt(config)

    train_start_time = time.time()

    # Phase 5: kept in main() scope so save_checkpoint_fn (closure over loss_meter.avg) and the runner's
    # validation gating see the latest epoch's values after run_epoch returns (the old for-loop leak).
    loss_meter = None
    has_validation = False
    has_predictor_validation = False
    has_value_validation = False

    if navsim_e120_planner_active:
        if navsim_e120_planner_plan is None or navsim_e120_planner_runtime is None:
            raise RuntimeError("NavSim e120 Planner milestone recovery requires initialized strict runtime state")
        reconcile_formal_v2_navsim_e120_resume_milestone(
            plan=navsim_e120_planner_plan,
            runtime_state=navsim_e120_planner_runtime,
            save_every_freq=config.meta.save_every_freq,
            rank=rank,
            distributed=navsim_e120_distributed,
            publish_checkpoint=lambda epoch, path, replace: save_checkpoint_fn(epoch, path, replace=replace),
        )

    def run_epoch(epoch):
        nonlocal loss_meter, has_validation, has_predictor_validation, has_value_validation
        logger.info(f"开始训练 Epoch {epoch + 1}")
        train_sampler.set_epoch(epoch)
        loader = iter(train_loader)

        # Diffusion planner (hybrid aWTA loss) 每 epoch 刷新温度，direct joint-action policy 没有 learned planner。
        cur_awta_temp = _maybe_set_planner_awta_temperature(config, planner, epoch)
        if cur_awta_temp is not None:
            logger.info(f"[aWTA] epoch={epoch} T={cur_awta_temp:.4f}")

        loss_meters = create_loss_meters()
        loss_meter, jloss_meter, sloss_meter = loss_meters["loss"], loss_meters["jloss"], loss_meters["sloss"]
        cls_valid_ratio_meter = loss_meters["cls_valid_ratio"]
        seg_loss_meter, mask_loss_meter, dice_loss_meter = (
            loss_meters["seg_loss"],
            loss_meters["mask_loss"],
            loss_meters["dice_loss"],
        )
        traj_loss_meter, reg_loss_meter, conf_loss_meter, cover_loss_meter = (
            loss_meters["traj_loss"],
            loss_meters["reg_loss"],
            loss_meters["conf_loss"],
            loss_meters["cover_loss"],
        )
        vel_loss_meter, yaw_loss_meter = loss_meters["vel_loss"], loss_meters["yaw_loss"]
        iter_time_meter, gpu_time_meter, data_elapsed_time_meter = (
            loss_meters["iter_time"],
            loss_meters["gpu_time"],
            loss_meters["data_load_time"],
        )
        wm_reward_loss_meter, wm_contrastive_loss_meter, wm_ranking_acc_meter = (
            AverageMeter(),
            AverageMeter(),
            AverageMeter(),
        )
        predictor_camera_loss_meters = {}

        for itr in range(ipe):
            timer = TrainingTimer()
            timer.start_iteration()
            loader, sample, success = get_next_batch(loader, train_loader, train_sampler, epoch)
            if not success:
                continue
            context_clips, actions, states, extrinsics, seg_targets, driving_command, ego_dynamics = load_clips(
                sample, device, config.segmentation.use_segmentation, torch.float
            )
            batch_metadata = (
                sample[-1] if isinstance(sample, (list, tuple)) and sample and isinstance(sample[-1], dict) else {}
            )
            if config.counterfactual_supervision.enabled:
                sample_masks = build_counterfactual_sample_masks(batch_metadata, device)
                imitation_mask = sample_masks.imitation
            else:
                sample_masks = None
                imitation_mask = torch.ones(actions.shape[0], dtype=torch.bool, device=device)
            metadata_valid_mask = batch_metadata.get("metadata_valid_mask")
            observed_metadata_valid_mask = batch_metadata.get("observed_metadata_valid_mask")
            camera_metadata = _move_camera_metadata_to_device(
                sample, device, torch.float, require_camera_geometry=bool(getattr(config.multiview, "enabled", False))
            )
            data_elapsed_time_ms = timer.record_data_load()
            maybe_run_gc(itr, GARBAGE_COLLECT_ITR_FREQ, config.meta.sync_gc)

            should_visualize = (rank == 0) and (itr % 100 == 0)
            vis_output_dir = os.path.join(config.meta.folder, "train_vis_debug")

            def train_step():
                formal_v2_sampled_horizon = None
                formal_v2_sampled_batch_size = None
                _new_lr, _new_wd = scheduler.step(), wd_scheduler.step()

                # Predictor 前向传播
                def forward_predictions(
                    z,
                    predictor_inputs,
                    rollout_end_step=None,
                    *,
                    compute_tf_override=None,
                    needs_ar_rollout_override=None,
                ):
                    pred_actions = predictor_inputs.actions
                    pred_states = predictor_inputs.states
                    pred_extrinsics = predictor_inputs.extrinsics
                    pred_driving_command = predictor_inputs.driving_command
                    pred_ego_dynamics = predictor_inputs.ego_dynamics
                    num_obs = predictor_inputs.num_observed_steps

                    # Diff 6: _step_predictor
                    def _step_predictor(_z, _a, _s, _e):
                        aux_inputs = prepare_predictor_aux_inputs(
                            actions=_a,
                            states=_s,
                            extrinsics=_e,
                            config=config,
                            num_observed_steps=num_obs,
                            driving_command=pred_driving_command,
                            ego_dynamics=pred_ego_dynamics,
                        )
                        _z = call_predictor_with_aux(predictor, _z, aux_inputs)
                        if runtime_normalize_reps:
                            _z = F.layer_norm(_z, (_z.size(-1),))
                        return _z

                    num_total = int(predictor_inputs.num_time_steps)
                    compute_tf = not planner_only_mode if compute_tf_override is None else bool(compute_tf_override)
                    needs_ar_rollout = (
                        predictor_needs_z_ar_rollout(config)
                        if needs_ar_rollout_override is None
                        else bool(needs_ar_rollout_override)
                    )
                    return rollout_latent_predictions(
                        _step_predictor,
                        config=config,
                        z_context=z,
                        actions=pred_actions,
                        states=pred_states,
                        extrinsics=pred_extrinsics,
                        num_obs=num_obs,
                        tokens_per_frame=tokens_per_frame,
                        num_total=num_total,
                        compute_tf=compute_tf,
                        needs_ar_rollout=needs_ar_rollout,
                        planner_only_error_context=True,
                        validate_ic_prefix=True,
                        rollout_end_step=rollout_end_step,
                    )

                def loss_fn(z, h, offset=tokens_per_frame):
                    _h = h[:, offset : z.size(1) + offset]
                    return torch.mean(torch.abs(z - _h) ** config.loss.loss_exp) / config.loss.loss_exp

                # Forward pass
                with torch.cuda.amp.autocast(dtype=config.dtype, enabled=config.mixed_precision):
                    grad_ctx = torch.enable_grad() if config.train.encoder_train else torch.no_grad()
                    with grad_ctx:
                        z_context = forward_main_context(
                            encoder,
                            context_clips,
                            config=config,
                            runtime_normalize_reps=runtime_normalize_reps,
                            token_ae=token_ae,
                            multiview_fusion=multiview_fusion,
                            camera_metadata=camera_metadata,
                        )
                    if skip_predictor_supervision:
                        h_target = None
                    else:
                        h_target = resolve_main_target_tokens(
                            reuse_context_as_target=reuse_context_as_target,
                            z_context=z_context,
                            target_encoder=target_encoder,
                            context_clips=context_clips,
                            config=config,
                            runtime_normalize_reps=runtime_normalize_reps,
                            token_ae=token_ae,
                            multiview_fusion=target_multiview_fusion,
                            camera_metadata=camera_metadata,
                        )
                    z_pred = z_context  # Diff 5: use z_pred variable
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
                    if cvoi_enabled(config):
                        predictor_inputs = enforce_cvoi_zero_future_aux(predictor_inputs)
                    raw_frame_dim = context_clips.shape[3] if context_clips.ndim == 6 else context_clips.shape[2]
                    context_timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=raw_frame_dim)
                    batch_timeline = resolve_main_timeline(
                        config,
                        encoder=encoder,
                        num_raw_frames=predictor_inputs.raw_num_frames,
                    )

                    # Diff 5: Shape assertions
                    if not (z_pred.shape[1] == context_timeline.num_time_steps * tokens_per_frame):
                        raise AssertionError(
                            f"z_pred shape mismatch: {z_pred.shape}, "
                            f"expected second dim to be {context_timeline.num_time_steps * tokens_per_frame}"
                        )
                    if not (context_timeline.num_time_steps >= batch_timeline.num_observed_steps):
                        raise AssertionError(
                            "context clip does not contain enough observed encoder steps: "
                            f"context_steps={context_timeline.num_time_steps}, "
                            f"observed_steps={batch_timeline.num_observed_steps}"
                        )
                    expected_action_steps = (
                        batch_timeline.num_time_steps
                        if use_parallel_predictor(config)
                        else batch_timeline.num_time_steps - 1
                    )
                    if not (predictor_inputs.actions.shape[1] == expected_action_steps):
                        raise AssertionError(
                            f"predictor actions shape mismatch: {predictor_inputs.actions.shape}, "
                            f"expected second dim to be {expected_action_steps}"
                        )
                    if not (predictor_inputs.states.shape[1] == batch_timeline.num_time_steps):
                        raise AssertionError(
                            f"predictor states shape mismatch: {predictor_inputs.states.shape}, "
                            f"expected second dim to be {batch_timeline.num_time_steps}"
                        )
                    if not (predictor_inputs.extrinsics.shape[1] == batch_timeline.num_time_steps):
                        raise AssertionError(
                            f"predictor extrinsics shape mismatch: {predictor_inputs.extrinsics.shape}, "
                            f"expected second dim to be {batch_timeline.num_time_steps}"
                        )

                    needs_planner_or_value = (
                        (config.planner.use_planner and planner is not None)
                        or value_head is not None
                        or wm_reward_head is not None
                        or wm_negative_provider is not None
                    )
                    dynamic_rollout_prefix = None
                    dynamic_rollout_end_step = None
                    validate_ac_transformer_dynamic_rollout_config(
                        config,
                        needs_planner_or_value=needs_planner_or_value,
                    )
                    if bool(config.predictor_dynamic_rollout.enabled) and not use_latent_dit_predictor(config):
                        total_future_steps = batch_timeline.num_time_steps - batch_timeline.num_observed_steps
                        rollout_horizon_steps = resolve_cvoi_training_rollout_horizon(
                            config,
                            total_future_steps=total_future_steps,
                        )
                        dynamic_rollout_prefix = select_dynamic_rollout_prefix(
                            enabled=True,
                            total_future_steps=rollout_horizon_steps,
                            full_prefix_prob=config.predictor_dynamic_rollout.full_prefix_prob,
                            min_prefix_steps=config.predictor_dynamic_rollout.min_prefix_steps,
                            max_non_full_prefix_steps=(config.predictor_dynamic_rollout.max_non_full_prefix_steps),
                            device=z_pred.device,
                            horizon_probabilities=config.predictor_dynamic_rollout.horizon_probabilities,
                        )
                        dynamic_rollout_end_step = dynamic_rollout_prefix.rollout_end_step(
                            num_observed_steps=batch_timeline.num_observed_steps
                        )
                        if planner_exposure_recorder is not None:
                            formal_v2_sampled_horizon = dynamic_rollout_prefix.prefix_steps
                            formal_v2_sampled_batch_size = int(actions.shape[0])
                        if rank == 0 and ((itr % log_freq == 0) or (itr == ipe - 1)):
                            logger.info(
                                "[predictor_dynamic_rollout] epoch %d iter %d: h=%d distribution=%s",
                                epoch,
                                itr,
                                dynamic_rollout_prefix.prefix_steps,
                                dynamic_rollout_prefix.distribution.probability_by_prefix_steps(),
                            )

                    budget_rollout_profile = None
                    if budget_controller is not None:
                        if bool(config.predictor_dynamic_rollout.enabled):
                            raise ValueError(
                                "budget_controller rollout budget must not be combined with "
                                "predictor_dynamic_rollout.enabled=true"
                            )
                        if use_latent_dit_predictor(config) or use_parallel_predictor(config):
                            raise ValueError(
                                "rollout budget controller currently supports only the non-parallel "
                                "ac_transformer autoregressive predictor path"
                            )
                        max_future_steps = batch_timeline.num_time_steps - batch_timeline.num_observed_steps
                        if max_future_steps <= 0:
                            raise ValueError(
                                "rollout budget requires num_time_steps > num_observed_steps, "
                                f"got {batch_timeline.num_time_steps} <= {batch_timeline.num_observed_steps}"
                            )
                        z_budget_obs = z_pred[:, : batch_timeline.num_observed_steps * tokens_per_frame]
                        controller_budget, budget_rollout_profile = resolve_controller_budget_profile(
                            budget_controller,
                            z_budget_obs,
                            config=config,
                            deterministic=True,
                            max_future_steps=max_future_steps,
                        )
                        dynamic_rollout_end_step = (
                            batch_timeline.num_observed_steps + budget_rollout_profile.rollout_future_steps
                        )
                        if rank == 0 and ((itr % log_freq == 0) or (itr == ipe - 1)):
                            logger.info(
                                "[budget_controller] epoch %d iter %d: budget=%.4f profile=%s " "rollout_end_step=%d",
                                epoch,
                                itr,
                                float(controller_budget[0].detach().cpu()),
                                budget_rollout_profile,
                                dynamic_rollout_end_step,
                            )

                    parallel_output = None
                    latent_output = None
                    split_tf_ar_output = None
                    if (
                        skip_predictor_supervision
                        and use_latent_dit_predictor(config)
                        and not use_sampled_latent_dit_planner_input(config)
                    ):
                        raise ValueError(
                            "Planner-only / predictor-planner-finetune training is not supported with latent-DiT "
                            "predictor training forward because it requires full-horizon target image latents."
                        )
                    if use_latent_dit_predictor(config) and not use_sampled_latent_dit_planner_input(config):
                        latent_output = forward_latent_dit_predictor_train(
                            predictor=predictor,
                            z_context=z_pred,
                            h_target=h_target,
                            predictor_inputs=predictor_inputs,
                            tokens_per_frame=tokens_per_frame,
                            num_observed_steps=batch_timeline.num_observed_steps,
                            runtime_normalize_reps=runtime_normalize_reps,
                            config=config,
                            imitation_mask=imitation_mask,
                        )
                        z_tf = latent_output.z_ar
                        z_ar = latent_output.z_ar
                    elif use_sampled_latent_dit_planner_input(config):
                        # Planner-only sampled latent-DiT mode has no target future latents.
                        # Keep placeholders for diagnostics/wm-aux-disabled code paths; the actual
                        # planner condition is sampled below via sample_latent_dit_predictor().
                        num_future_tokens = (
                            batch_timeline.num_time_steps - batch_timeline.num_observed_steps
                        ) * tokens_per_frame
                        z_tf = z_pred.new_zeros(z_pred.shape[0], num_future_tokens, z_pred.shape[-1])
                        z_ar = z_tf
                    elif use_parallel_predictor(config):
                        parallel_output = forward_parallel_predictor(
                            predictor=predictor,
                            observed_tokens=z_pred,
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
                    elif config.train.predictor_split_tf_ar_backward:

                        def forward_tf_only():
                            z_tf_only, unexpected_z_ar = forward_predictions(
                                z_pred,
                                predictor_inputs,
                                rollout_end_step=dynamic_rollout_end_step,
                                compute_tf_override=True,
                                needs_ar_rollout_override=False,
                            )
                            if z_tf_only is None or unexpected_z_ar is not None:
                                raise RuntimeError(
                                    "split TF forward must return z_tf and no z_ar, got "
                                    f"z_tf_is_none={z_tf_only is None}, "
                                    f"z_ar_is_none={unexpected_z_ar is None}"
                                )
                            return z_tf_only

                        def forward_ar_only():
                            _unused_z_tf, z_ar_only = forward_predictions(
                                z_pred,
                                predictor_inputs,
                                rollout_end_step=dynamic_rollout_end_step,
                                compute_tf_override=False,
                                needs_ar_rollout_override=True,
                            )
                            if z_ar_only is None:
                                raise RuntimeError("split AR forward must return z_ar")
                            return z_ar_only

                        split_tf_ar_output = run_split_tf_ar_forward_backward(
                            forward_tf=forward_tf_only,
                            forward_ar=forward_ar_only,
                            compute_jloss=lambda z_tf_tokens: compute_split_tf_loss(
                                z_tf=z_tf_tokens,
                                h_target=h_target,
                                tokens_per_frame=tokens_per_frame,
                                num_observed_steps=batch_timeline.num_observed_steps,
                                loss_fn=loss_fn,
                            ),
                            compute_sloss=lambda z_ar_tokens: compute_split_ar_loss(
                                z_ar=z_ar_tokens,
                                h_target=h_target,
                                tokens_per_frame=tokens_per_frame,
                                num_observed_steps=batch_timeline.num_observed_steps,
                                loss_fn=loss_fn,
                            ),
                            early_backward=lambda tf_loss: backward_loss_outside_autocast(
                                tf_loss,
                                scaler=scaler,
                                mixed_precision=config.mixed_precision,
                            ),
                        )
                        z_tf = split_tf_ar_output.z_tf
                        z_ar = split_tf_ar_output.z_ar
                    else:
                        z_tf, z_ar = forward_predictions(
                            z_pred,
                            predictor_inputs,
                            rollout_end_step=dynamic_rollout_end_step,
                        )

                    # JEPA / latent predictor loss
                    predictor_camera_metrics = {}
                    if skip_predictor_supervision:
                        # planner-only 或 predictor-planner-finetune：不施加 jepa_loss。
                        # finetune 时 predictor 仍可训练，梯度只来自下游 planner（z_ar）。
                        jepa_loss = z_pred.new_zeros(())
                        jloss = z_pred.new_zeros(())
                        sloss = z_pred.new_zeros(())
                    elif use_latent_dit_predictor(config):
                        jepa_loss = latent_output.loss
                        jloss = latent_output.flow_loss
                        sloss = latent_output.x0_loss
                        with torch.no_grad():
                            predictor_camera_metrics = compute_latent_dit_per_camera_losses(
                                latent_output=latent_output,
                                h_target=h_target,
                                config=config,
                                tokens_per_frame=tokens_per_frame,
                                num_observed_steps=batch_timeline.num_observed_steps,
                                predictor=predictor,
                            )
                    elif split_tf_ar_output is not None:
                        jepa_loss = split_tf_ar_output.jepa_loss
                        jloss = split_tf_ar_output.jloss
                        sloss = split_tf_ar_output.sloss
                        with torch.no_grad():
                            predictor_camera_metrics = compute_predictor_per_camera_jepa_losses(
                                z_tf=z_tf,
                                z_ar=z_ar,
                                h_target=h_target,
                                config=config,
                                tokens_per_frame=tokens_per_frame,
                                num_observed_steps=batch_timeline.num_observed_steps,
                            )
                    else:
                        jepa_loss, jloss, sloss = compute_predictor_jepa_losses_from_config(
                            z_tf=z_tf,
                            z_ar=z_ar,
                            h_target=h_target,
                            config=config,
                            tokens_per_frame=tokens_per_frame,
                            loss_fn=loss_fn,
                            num_observed_steps=batch_timeline.num_observed_steps,
                            active_prefix_steps=(
                                None if dynamic_rollout_prefix is None else dynamic_rollout_prefix.prefix_steps
                            ),
                        )
                        with torch.no_grad():
                            predictor_camera_metrics = compute_predictor_per_camera_jepa_losses(
                                z_tf=z_tf,
                                z_ar=z_ar,
                                h_target=h_target,
                                config=config,
                                tokens_per_frame=tokens_per_frame,
                                num_observed_steps=batch_timeline.num_observed_steps,
                                active_prefix_steps=(
                                    None if dynamic_rollout_prefix is None else dynamic_rollout_prefix.prefix_steps
                                ),
                            )

                    # Phase 1 (doc 方向 D): world-model 辅助监督。
                    # (a) λ^k 折扣已在 compute_predictor_jepa_losses_from_config 内生效;
                    # (b) reward/risk 头联合训练(z_ar 带梯度,梯度流入 predictor);
                    # (c) contrastive ranking(GT vs 反事实轨迹,同一 rollout 路径)。
                    wm_aux_loss = z_pred.new_zeros(())
                    _wm_reward_loss = z_pred.new_zeros(())
                    _wm_contrastive_loss = z_pred.new_zeros(())
                    _wm_ranking_acc = z_pred.new_zeros(())
                    if wm_reward_head is not None:
                        _wm_rh = compute_wm_reward_head_loss(
                            config=config,
                            reward_head=wm_reward_head,
                            z_ar=z_ar,
                            sample=sample,
                            timeline_states=predictor_inputs.states,
                            num_observed_steps=batch_timeline.num_observed_steps,
                            num_time_steps=batch_timeline.num_time_steps,
                            frame_stride=batch_timeline.frame_stride,
                            tokens_per_frame=tokens_per_frame,
                        )
                        _wm_reward_loss = _wm_rh["loss"]
                        wm_aux_loss = wm_aux_loss + config.wm_aux.reward_head_weight * _wm_reward_loss
                    if wm_negative_provider is not None:
                        _wm_ct = compute_wm_contrastive_loss(
                            config=config,
                            predictor=predictor,
                            negative_provider=wm_negative_provider,
                            z_context=z_pred,
                            h_target=h_target,
                            actions=actions,
                            states=states,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            tokens_per_frame=tokens_per_frame,
                            runtime_normalize_reps=runtime_normalize_reps,
                            num_observed_steps=batch_timeline.num_observed_steps,
                            frame_stride=batch_timeline.frame_stride,
                        )
                        _wm_contrastive_loss = _wm_ct["loss"]
                        _wm_ranking_acc = _wm_ct["ranking_acc"]
                        wm_aux_loss = wm_aux_loss + config.wm_aux.contrastive_weight * _wm_contrastive_loss

                    # 选择 planner 输入源: z_ar (自回归) 或 z_tf (teacher forcing)
                    if config.planner.planner_input_source == "z_tf":
                        if parallel_output is not None:
                            z_planner_input = parallel_output.z_future
                        elif config.train.predictor_inference_consistent:
                            _nobs_pi = batch_timeline.num_observed_steps
                            z_planner_input = z_tf[:, (_nobs_pi - 1) * tokens_per_frame :]
                        else:
                            z_planner_input = z_tf
                    else:
                        z_planner_input = z_ar

                    needs_learned_planner_or_value = (
                        config.planner.use_planner and planner is not None
                    ) or value_head is not None
                    needs_policy_or_value = (
                        config.planner.use_planner and (planner is not None or direct_joint_action_policy)
                    ) or value_head is not None
                    use_sampled_latent_dit_input = use_sampled_latent_dit_planner_input(config)
                    joint_policy_sample = None
                    if (
                        latent_output is not None
                        and getattr(latent_output, "future_token_indices", None) is not None
                        and needs_learned_planner_or_value
                        and not use_sampled_latent_dit_input
                    ):
                        raise ValueError(
                            "masked latent-DiT train helper returned a partial future window, but planner/value "
                            "requires full future tokens. Use train.latent_dit_planner_input='sample' with "
                            "train.predictor_train=false, or set predictor_dit.masked_train_full_prefix_prob=1.0 "
                            "for train-helper stages."
                        )
                    batch_value_guidance_active = value_guidance_active
                    batch_cvoi_guidance_active = cvoi_guidance_active
                    if use_sampled_latent_dit_input and needs_policy_or_value:
                        if (
                            direct_joint_action_policy
                            or str(getattr(config.planner, "latent_dit_action_source", "planner")).lower()
                            == "joint_action"
                        ):
                            joint_policy_sample = sample_latent_dit_joint_action_predictor(
                                predictor=predictor,
                                z_context=z_pred,
                                predictor_inputs=predictor_inputs,
                                tokens_per_frame=tokens_per_frame,
                                num_observed_steps=batch_timeline.num_observed_steps,
                                runtime_normalize_reps=runtime_normalize_reps,
                                config=config,
                                **resolve_latent_dit_sampler_params(config).as_kwargs(),
                            )
                            z_planner_input = joint_policy_sample.z_ar
                            if should_visualize:
                                build_joint_action_policy_output(
                                    joint_policy_sample.actions,
                                    num_poses=num_poses,
                                    frame_stride=max(int(batch_timeline.frame_stride), 1),
                                )
                        else:
                            z_planner_input = sample_latent_dit_predictor(
                                predictor=predictor,
                                z_context=z_pred,
                                predictor_inputs=predictor_inputs,
                                tokens_per_frame=tokens_per_frame,
                                num_observed_steps=batch_timeline.num_observed_steps,
                                runtime_normalize_reps=runtime_normalize_reps,
                                config=config,
                                **resolve_latent_dit_sampler_params(config).as_kwargs(),
                            )
                    value_z_future = z_planner_input if use_sampled_latent_dit_input else z_ar

                    gt_trajectory = None
                    if needs_policy_or_value:
                        gt_trajectory = _build_future_gt_trajectory(states, config, num_poses)

                    value_planning_loss = z_pred.new_zeros(())
                    if value_head is not None:
                        if gt_trajectory is None:
                            raise ValueError("value_planning enabled but gt_trajectory was not built")
                        if not value_guidance_active:
                            _vp = compute_value_head_loss_from_batch(
                                value_head=value_head,
                                target_value_head=target_value_head,
                                z_future=value_z_future,
                                gt_trajectory=gt_trajectory,
                                sample_masks=sample_masks,
                                tokens_per_frame=tokens_per_frame,
                                config=config,
                                frame_stride=batch_timeline.frame_stride,
                            )
                            value_planning_loss = _vp["loss"]
                    if batch_value_guidance_active and config.planner.use_planner and planner is not None:
                        if value_head is None:
                            raise ValueError("value_guidance.enabled=true requires a loaded value_head")
                        apply_guidance_this_batch = should_apply_value_guidance(
                            z_planner_input,
                            value_guidance_enabled=batch_value_guidance_active,
                            allow_empty_rollout_skip=bool(config.predictor_dynamic_rollout.enabled),
                        )
                        if apply_guidance_this_batch:
                            z_planner_input, _value_guidance_diag = apply_latent_value_guidance(
                                z_planner_input,
                                value_head,
                                tokens_per_frame=tokens_per_frame,
                                config=config,
                            )
                        if apply_guidance_this_batch and rank == 0 and ((itr % log_freq == 0) or (itr == ipe - 1)):
                            logger.info(
                                "[value_guidance] epoch %d iter %d: value_before=%.5f value_after=%.5f "
                                "delta_norm=%.5f steps=%.0f",
                                epoch,
                                itr,
                                _value_guidance_diag["value_before"],
                                _value_guidance_diag["value_after"],
                                _value_guidance_diag["delta_norm"],
                                _value_guidance_diag["guidance_steps"],
                            )
                        elif rank == 0 and ((itr % log_freq == 0) or (itr == ipe - 1)):
                            logger.info(
                                "[value_guidance] epoch %d iter %d: skipped for rollout_future_steps=0",
                                epoch,
                                itr,
                            )
                    if batch_cvoi_guidance_active and config.planner.use_planner and planner is not None:
                        z_observed = z_pred[:, : batch_timeline.num_observed_steps * tokens_per_frame]
                        z_planner_input, _cvoi_guidance_diag = apply_cvoi_planner_guidance(
                            z_observed,
                            z_planner_input,
                            cvoi_dual_value,
                            tokens_per_frame=tokens_per_frame,
                            config=config,
                        )
                        if rank == 0 and ((itr % log_freq == 0) or (itr == ipe - 1)):
                            logger.info(
                                "[cvoi_guidance] epoch %d iter %d: field_before=%.5f field_after=%.5f "
                                "delta_norm=%.5f steps=%.0f h0_skip=%.0f",
                                epoch,
                                itr,
                                _cvoi_guidance_diag["field_value_before"],
                                _cvoi_guidance_diag["field_value_after"],
                                _cvoi_guidance_diag["delta_norm"],
                                _cvoi_guidance_diag["guidance_steps"],
                                _cvoi_guidance_diag["guidance_skipped_h0"],
                            )

                    # Planner
                    traj_loss = torch.tensor(0.0, device=device)
                    _reg_loss = torch.tensor(0.0, device=device)
                    _conf_loss = torch.tensor(0.0, device=device)
                    _cover_loss = torch.tensor(0.0, device=device)
                    _vel_loss = torch.tensor(0.0, device=device)
                    _yaw_loss = torch.tensor(0.0, device=device)
                    _cls_valid_ratio = torch.tensor(0.0, device=device)
                    if config.planner.use_planner and direct_joint_action_policy:
                        planner_out = None
                        if use_sampled_latent_dit_input:
                            if joint_policy_sample is None:
                                raise ValueError(
                                    "Direct joint action policy sampled mode requires joint action sampler output"
                                )
                            planner_out = build_joint_action_policy_output(
                                joint_policy_sample.actions,
                                num_poses=num_poses,
                                frame_stride=max(int(batch_timeline.frame_stride), 1),
                            )
                        elif latent_output is not None and latent_output.action_x0_pred is not None:
                            if getattr(latent_output, "future_token_indices", None) is None:
                                normalized_actions = latent_output.action_x0_pred
                                direct_actions = denormalize_joint_actions(
                                    normalized_actions,
                                    tuple(float(value) for value in config.predictor_dit.joint_action_scale),
                                )
                                planner_out = build_joint_action_policy_output(
                                    direct_actions,
                                    num_poses=num_poses,
                                    frame_stride=max(int(batch_timeline.frame_stride), 1),
                                )
                        else:
                            raise ValueError(
                                "Direct joint action policy training requires latent-DiT joint action outputs; "
                                "use predictor_dit.joint_action_enabled=True with differentiable latent-DiT training."
                            )
                        if should_visualize and planner_out is not None and gt_trajectory is not None:
                            best_traj = select_best_trajectory(
                                planner_out["trajectories"],
                                planner_out["confidences"],
                            )
                            visualize_trajectory(
                                pred_traj=best_traj,
                                gt_traj=gt_trajectory,
                                output_dir=vis_output_dir,
                                epoch=epoch,
                                itr=itr,
                                limit=5,
                            )
                    elif config.planner.use_planner and planner is not None:
                        planner_imitation_has_samples = bool(imitation_mask.any().item())
                        planner_imitation_normalization = distributed_mask_normalization(
                            imitation_mask,
                            name="planner trajectory imitation",
                            dtype=z_planner_input.dtype,
                        )
                        if config.train.predictor_inference_consistent:
                            status_feature = prepare_inference_consistent_status_vector(
                                states,
                                num_observed=config.train.num_observed_frames,
                                driving_command=driving_command,
                                ego_dynamics=ego_dynamics,
                                state_dim=resolve_planner_status_dim(config),
                                use_drive_command=resolve_planner_use_drive_command(config),
                            )
                        else:
                            status_feature = prepare_status_feature(
                                states,
                                actions,
                                mode=config.planner.states_mode,
                                use_states_for_planner=config.planner.use_states_for_planner,
                                action_dim=config.train.action_dim,
                                driving_command=driving_command,
                                ego_dynamics=ego_dynamics,
                            )

                        z_first_frame = z_context[:, :tokens_per_frame] if config.planner.use_z_context else None
                        # 观测帧 tokens
                        if config.planner.use_observed_tokens:
                            num_obs = batch_timeline.num_observed_steps
                            z_observed = z_context[:, : num_obs * tokens_per_frame]
                        else:
                            z_observed = None
                        planner_action_history = None
                        if getattr(config.planner, "use_action_history_for_planner", False):
                            planner_action_history = build_observed_action_trajectory_history(
                                predictor_inputs.actions,
                                num_observed_frames=batch_timeline.num_observed_steps,
                                action_history_dim=int(getattr(config.planner, "action_history_dim", 3)),
                                dt=action_history_dt * max(int(predictor_inputs.frame_stride), 1),
                            )
                        validate_empty_future_planner_conditions(
                            z_planner_input,
                            z_context=z_first_frame,
                            z_observed=z_observed,
                            action_history=planner_action_history,
                        )

                        # Diff 8: Diffusion planner support
                        if config.planner.planner_type == "diffusion":
                            gt_traj_nd = convert_trajectory_3d_to_nd(
                                gt_trajectory,
                                dt=config.planner.diff_dt,
                                traj_dim=config.planner.diff_traj_dim,
                            )
                            anchor_observed_frames = (
                                config.train.num_observed_frames if config.train.predictor_inference_consistent else 1
                            )
                            action_states = build_ego_relative_diffusion_anchor(
                                planner,
                                ego_dynamics=ego_dynamics,
                                observed_frames=anchor_observed_frames,
                                reference=states,
                            )
                            planner_imitation_is_dummy = False
                            if planner_imitation_has_samples:
                                loss_z_planner_input = z_planner_input[imitation_mask]
                                loss_status_feature = status_feature[imitation_mask]
                                loss_z_first_frame = None if z_first_frame is None else z_first_frame[imitation_mask]
                                loss_z_observed = None if z_observed is None else z_observed[imitation_mask]
                                loss_action_history = (
                                    None if planner_action_history is None else planner_action_history[imitation_mask]
                                )
                                loss_gt_traj_nd = gt_traj_nd[imitation_mask]
                                loss_action_states = None if action_states is None else action_states[imitation_mask]
                                loss_gt_trajectory = gt_trajectory[imitation_mask]
                            else:
                                # DDP ranks must all enter the planner graph even when this local shard has
                                # no clone-eligible samples; the zeroed dummy loss preserves the no-clone rule.
                                planner_imitation_is_dummy = True
                                loss_z_planner_input = z_planner_input[:1]
                                loss_status_feature = status_feature[:1]
                                loss_z_first_frame = None if z_first_frame is None else z_first_frame[:1]
                                loss_z_observed = None if z_observed is None else z_observed[:1]
                                loss_action_history = (
                                    None if planner_action_history is None else planner_action_history[:1]
                                )
                                loss_gt_traj_nd = torch.zeros_like(gt_traj_nd[:1])
                                loss_action_states = None if action_states is None else action_states[:1]
                                loss_gt_trajectory = torch.zeros_like(gt_trajectory[:1])
                            diff_result = planner(
                                loss_z_planner_input,
                                loss_status_feature,
                                z_context=loss_z_first_frame,
                                z_observed=loss_z_observed,
                                action_history=loss_action_history,
                                gt_trajectory=loss_gt_traj_nd,
                                anchor_state=loss_action_states,
                            )
                            validate_planner_output(
                                diff_result,
                                mode="training",
                                required_training_keys=("reg_loss", "conf_loss", "cover_loss"),
                            )
                            if planner_imitation_is_dummy:
                                traj_loss = diff_result["loss"] * 0.0
                                _reg_loss = diff_result["reg_loss"] * 0.0
                                _conf_loss = diff_result["conf_loss"] * 0.0
                                _cover_loss = diff_result["cover_loss"] * 0.0
                                _vel_loss = diff_result.get("vel_loss", torch.tensor(0.0, device=device)) * 0.0
                                _yaw_loss = diff_result.get("yaw_loss", torch.tensor(0.0, device=device)) * 0.0
                                _cls_valid_ratio = torch.tensor(0.0, device=device)
                            else:
                                traj_loss = diff_result["loss"]
                                _reg_loss = diff_result["reg_loss"]
                                _conf_loss = diff_result["conf_loss"]
                                _cover_loss = diff_result["cover_loss"]
                                _vel_loss = diff_result.get("vel_loss", torch.tensor(0.0, device=device))
                                _yaw_loss = diff_result.get("yaw_loss", torch.tensor(0.0, device=device))
                                _cls_valid_ratio = diff_result.get(
                                    "cls_sample_valid_ratio", torch.tensor(0.0, device=device)
                                )
                            traj_loss = traj_loss * planner_imitation_normalization.mean_scale
                            _reg_loss = _reg_loss * planner_imitation_normalization.mean_scale
                            _conf_loss = _conf_loss * planner_imitation_normalization.mean_scale
                            _cover_loss = _cover_loss * planner_imitation_normalization.mean_scale
                            _vel_loss = _vel_loss * planner_imitation_normalization.mean_scale
                            _yaw_loss = _yaw_loss * planner_imitation_normalization.mean_scale
                            if should_visualize and not planner_imitation_is_dummy and "winner_traj_3d" in diff_result:
                                visualize_trajectory(
                                    pred_traj=diff_result["winner_traj_3d"],
                                    gt_traj=loss_gt_trajectory,
                                    output_dir=vis_output_dir,
                                    epoch=epoch,
                                    itr=itr,
                                    limit=5,
                                )
                        else:
                            planner_out = planner(
                                z_planner_input,
                                status_feature,
                                z_context=z_first_frame,
                                z_observed=z_observed,
                                action_history=planner_action_history,
                            )
                            validate_planner_output(planner_out, mode="inference")
                            pred_trajs = planner_out["trajectories"]
                            pred_conf = planner_out["confidences"]

                            if planner_imitation_has_samples:
                                # PyTorch bool indexing 会沿 batch 维筛样本：
                                # mask=[True, False, True] 时只保留第 0/2 个样本进入 WTA loss。
                                loss_pred_trajs = pred_trajs[imitation_mask]
                                loss_pred_conf = pred_conf[imitation_mask]
                                loss_gt_trajectory = gt_trajectory[imitation_mask]
                            else:
                                loss_pred_trajs = pred_trajs[:0]
                                loss_pred_conf = pred_conf[:0]
                                loss_gt_trajectory = gt_trajectory[:0]

                            # compute_planner_wta_loss is the shared fail-loud dispatch (num_modes==1 →
                            # single_model_loss; else v1/v2/v3 by wta_loss_version, raising on unknown;
                            # alpha=config.planner.wta_alpha, default 5.0) — equivalent to main's inline
                            # dispatch but without the duplicated branches / hardcoded alpha.
                            if planner_imitation_has_samples:
                                wta_result = compute_planner_wta_loss(
                                    config,
                                    pred_trajs=loss_pred_trajs,
                                    pred_conf=loss_pred_conf,
                                    gt_traj=loss_gt_trajectory,
                                    epoch=epoch,
                                    global_batch=bool(config.planner.wta_global_batch_norm),
                                )

                            if planner_imitation_has_samples:
                                traj_loss = wta_result["loss"]
                                _reg_loss = wta_result["reg_loss"]
                                _conf_loss = wta_result["conf_loss"]
                                _cover_loss = wta_result["cover_loss"]
                            else:
                                if bool(config.planner.wta_global_batch_norm) and config.planner.num_modes > 1:
                                    from app.vjepa_cowa_world_model.losses.wta_loss import (
                                        _normalize_arc_length_weights,
                                    )

                                    _normalize_arc_length_weights(
                                        pred_trajs.new_empty(0),
                                        eps=1e-6,
                                        global_batch=True,
                                    )
                                traj_loss = pred_trajs.sum() * 0.0
                                _reg_loss = traj_loss
                                _conf_loss = pred_conf.sum() * 0.0
                                _cover_loss = traj_loss
                            traj_loss = traj_loss * planner_imitation_normalization.mean_scale
                            _reg_loss = _reg_loss * planner_imitation_normalization.mean_scale
                            _conf_loss = _conf_loss * planner_imitation_normalization.mean_scale
                            _cover_loss = _cover_loss * planner_imitation_normalization.mean_scale

                            if should_visualize:
                                best_traj = select_best_trajectory(pred_trajs, pred_conf)
                                visualize_trajectory(
                                    pred_traj=best_traj,
                                    gt_traj=gt_trajectory,
                                    output_dir=vis_output_dir,
                                    epoch=epoch,
                                    itr=itr,
                                    limit=5,
                                )

                    loss = (
                        jepa_loss + config.planner.planner_loss_weight * traj_loss + wm_aux_loss + value_planning_loss
                    )

                    # Segmentation loss
                    seg_loss_value = 0.0
                    mask_loss_value = 0.0
                    dice_loss_value = 0.0
                    valid_samples = 0
                    neck_out = None
                    vis_meta = None

                    if config.segmentation.use_segmentation and seg_head is not None:
                        neck_out, batched_targets, valid_samples, vis_meta = prepare_seg_features(
                            context_clips=context_clips,
                            seg_targets=seg_targets,
                            z_perceiver=z_context,
                            seg_neck=seg_neck,
                            tubelet_size=config.data.tubelet_size,
                            tokens_per_frame=tokens_per_frame,
                            device=device,
                            mixed_precision=config.mixed_precision,
                            dtype=config.dtype,
                            normalize_reps=config.loss.normalize_reps,
                        )

                    if config.segmentation.use_segmentation and seg_head is not None and valid_samples > 0:
                        loss_dict = seg_head.module.get_loss(
                            inputs=neck_out, targets=batched_targets, input_query=None
                        )
                        total_seg_loss = sum(v for k, v in loss_dict.items() if "loss" in k)
                        seg_loss = total_seg_loss / valid_samples
                        loss = loss + (config.segmentation.seg_loss_weight * seg_loss)

                        with torch.no_grad():
                            last_layer_idx = 6
                            seg_loss_value = (
                                loss_dict.get(f"loss_seg_{last_layer_idx}", torch.tensor(0.0)).detach().item()
                            )
                            dice_loss_value = (
                                loss_dict.get(f"loss_dice_{last_layer_idx}", torch.tensor(0.0)).detach().item()
                            )
                            mask_loss_value = seg_loss_value

                # Diff 4: Always call backward to keep DDP gradient sync across all ranks
                _value_optimizer_step_successful = False

                if config.mixed_precision:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Keep the decision on-device until one cross-rank MIN and one
                # host read. Under AMP, inspect scaled gradients and defer
                # unscale until every rank has approved the optimizer step.
                _gradient_scale = float(scaler.get_scale()) if config.mixed_precision else None
                _local_step_allowed = torch.isfinite(loss.detach()).all()
                _local_step_allowed.logical_and_(
                    optimizer_gradients_finite_flag(
                        optimizer,
                        device=device,
                        scale=_gradient_scale,
                    )
                )
                _global_step_allowed = distributed_optimizer_step_allowed(_local_step_allowed)
                # Keep the historical variable/branch name because the branch
                # owns GradScaler's skipped-step update contract.  It now means
                # that any rank rejected the step before any optimizer stepped.
                _nan_detected = not _global_step_allowed
                if _nan_detected:
                    logger.warning(
                        f"[epoch {epoch + 1}, iter {itr}] global NaN/Inf detected before optimizer step "
                        f"(local_loss={loss.item():.4g}), all ranks skip"
                    )
                    if itr < 3:
                        diagnostic_items = [
                            _format_tensor_finite_summary("loss", loss),
                            _format_tensor_finite_summary("jepa_loss", jepa_loss),
                            _format_tensor_finite_summary("jloss", jloss),
                            _format_tensor_finite_summary("sloss", sloss),
                            _format_tensor_finite_summary("traj_loss", traj_loss),
                            _format_tensor_finite_summary("z_pred", z_pred),
                            _format_tensor_finite_summary("z_tf", z_tf),
                            _format_tensor_finite_summary("z_ar", z_ar),
                            _format_tensor_finite_summary("pred_actions", predictor_inputs.actions),
                            _format_tensor_finite_summary("pred_states", predictor_inputs.states),
                            _format_tensor_finite_summary("pred_extrinsics", predictor_inputs.extrinsics),
                            _format_tensor_finite_summary("pred_driving_command", predictor_inputs.driving_command),
                            _format_tensor_finite_summary("pred_ego_dynamics", predictor_inputs.ego_dynamics),
                        ]
                        if h_target is not None:
                            diagnostic_items.append(_format_tensor_finite_summary("h_target", h_target))
                        logger.warning(
                            "[epoch %d, iter %d] NaN diagnostics: %s",
                            epoch + 1,
                            itr,
                            " | ".join(diagnostic_items),
                        )
                    if config.mixed_precision:
                        skip_amp_optimizer_step_(scaler, optimizer)
                    else:
                        optimizer.zero_grad()
                    record_formal_v2_navsim_e120_optimizer_exposure(
                        planner_exposure_recorder,
                        optimizer_step_successful=False,
                        horizon=formal_v2_sampled_horizon,
                        batch_size=formal_v2_sampled_batch_size,
                    )
                else:
                    if config.mixed_precision:
                        scaler.unscale_(optimizer)
                    if config.planner.use_planner and planner is not None:
                        torch.nn.utils.clip_grad_norm_(
                            planner.parameters(), max_norm=config.optimization.grad_clip_norm
                        )
                    if multiview_fusion is not None:
                        torch.nn.utils.clip_grad_norm_(
                            multiview_fusion.parameters(), max_norm=config.optimization.grad_clip_norm
                        )
                    if value_head is not None:
                        torch.nn.utils.clip_grad_norm_(
                            value_head.parameters(), max_norm=config.optimization.grad_clip_norm
                        )
                    # Clip predictor grads whenever the predictor is being trained
                    # (full fine-tune OR LoRA). Previously this only fired for the
                    # LoRA branch, so full predictor fine-tuning ran with NO gradient
                    # clipping — a single large-but-finite step late in training could
                    # blow the weights out of the basin and trap the run at a degraded
                    # loss for the rest of training (observed: irreversible spike ~epoch 82).
                    if predictor is not None and predictor_weights_updated:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in predictor.parameters() if p.requires_grad],
                            max_norm=config.optimization.grad_clip_norm,
                        )
                    if seg_head is not None:
                        torch.nn.utils.clip_grad_norm_(
                            seg_head.parameters(), max_norm=config.optimization.grad_clip_norm
                        )

                    if config.mixed_precision:
                        _amp_scale_before = _gradient_scale
                        scaler.step(optimizer)
                        scaler.update()
                        _value_optimizer_step_successful = value_optimizer_step_succeeded(
                            mixed_precision=True,
                            scale_before=_amp_scale_before,
                            scale_after=float(scaler.get_scale()),
                        )
                        if not _value_optimizer_step_successful:
                            raise RuntimeError(
                                "GradScaler skipped after all ranks approved finite loss/gradients; "
                                "pre-step overflow consensus is inconsistent"
                            )
                    else:
                        optimizer.step()
                        _value_optimizer_step_successful = value_optimizer_step_succeeded(
                            mixed_precision=False,
                            scale_before=None,
                            scale_after=None,
                        )
                    optimizer.zero_grad()
                    record_formal_v2_navsim_e120_optimizer_exposure(
                        planner_exposure_recorder,
                        optimizer_step_successful=_value_optimizer_step_successful,
                        horizon=formal_v2_sampled_horizon,
                        batch_size=formal_v2_sampled_batch_size,
                    )

                if value_head_trainable:
                    # The non-DDP target follows the globally agreed pre-step
                    # decision, so every rank applies the same update or no update.
                    maybe_update_value_target_(
                        value_head,
                        target_value_head,
                        tau=float(config.value_planning.target_tau),
                        optimizer_step_successful=_value_optimizer_step_successful,
                    )

                # EMA 更新
                if config.train.encoder_ema:
                    m = next(momentum_scheduler)
                    update_ema(m)

                # 可视化
                if should_visualize and valid_samples > 0:
                    with torch.no_grad():
                        real_head = seg_head.module if isinstance(seg_head, DistributedDataParallel) else seg_head
                        real_head.eval()
                        pred_result = real_head(inputs=[x.float() for x in neck_out], input_query=None)
                        save_training_visualization(pred_result, vis_meta, vis_output_dir, epoch, itr)
                        real_head.train()

                def _to_float(value):
                    if torch.is_tensor(value):
                        return float(value.detach())
                    return float(value)

                predictor_camera_metric_values = {
                    key: _to_float(value) for key, value in predictor_camera_metrics.items()
                }

                return (
                    _to_float(loss),
                    _to_float(jloss),
                    _to_float(sloss),
                    _to_float(seg_loss_value),
                    _to_float(mask_loss_value),
                    _to_float(traj_loss),
                    _to_float(dice_loss_value),
                    _to_float(_reg_loss),
                    _to_float(_conf_loss),
                    _to_float(_cover_loss),
                    _to_float(_vel_loss),
                    _to_float(_yaw_loss),
                    _to_float(_cls_valid_ratio),
                    _to_float(_wm_reward_loss),
                    _to_float(_wm_contrastive_loss),
                    _to_float(_wm_ranking_acc),
                    predictor_camera_metric_values,
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                jloss,
                sloss,
                seg_loss_value,
                mask_loss_value,
                traj_loss,
                dice_loss_value,
                reg_loss_value,
                conf_loss_value,
                cover_loss_value,
                vel_loss_value,
                yaw_loss_value,
                cls_valid_ratio_value,
                wm_reward_loss_value,
                wm_contrastive_loss_value,
                wm_ranking_acc_value,
                predictor_camera_metrics,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)

            # Diff 12: Log optimizer group LRs at start of each epoch
            if itr == 0:
                log_optimizer_group_lrs(epoch, itr)

            iter_elapsed_time_ms, _ = timer.stop_iteration()

            for m, v in [
                (loss_meter, loss),
                (jloss_meter, jloss),
                (sloss_meter, sloss),
                (seg_loss_meter, seg_loss_value),
                (mask_loss_meter, mask_loss_value),
                (dice_loss_meter, dice_loss_value),
                (traj_loss_meter, traj_loss),
                (reg_loss_meter, reg_loss_value),
                (conf_loss_meter, conf_loss_value),
                (cover_loss_meter, cover_loss_value),
                (vel_loss_meter, vel_loss_value),
                (yaw_loss_meter, yaw_loss_value),
                (cls_valid_ratio_meter, cls_valid_ratio_value),
                (iter_time_meter, iter_elapsed_time_ms),
                (gpu_time_meter, gpu_etime_ms),
                (data_elapsed_time_meter, data_elapsed_time_ms),
                (wm_reward_loss_meter, wm_reward_loss_value),
                (wm_contrastive_loss_meter, wm_contrastive_loss_value),
                (wm_ranking_acc_meter, wm_ranking_acc_value),
            ]:
                m.update(v)
            if (
                (wm_reward_head is not None or wm_negative_provider is not None)
                and rank == 0
                and ((itr % log_freq == 0) or (itr == ipe - 1))
            ):
                logger.info(
                    "[wm_aux] epoch %d iter %d: reward_head=%.5f contrastive=%.5f ranking_acc=%.3f",
                    epoch,
                    itr,
                    wm_reward_loss_meter.avg,
                    wm_contrastive_loss_meter.avg,
                    wm_ranking_acc_meter.avg,
                )
                if tb_writer is not None:
                    _wm_step = epoch * ipe + itr
                    tb_writer.add_scalar("wm_aux/reward_head_loss", wm_reward_loss_meter.avg, _wm_step)
                    tb_writer.add_scalar("wm_aux/contrastive_loss", wm_contrastive_loss_meter.avg, _wm_step)
                    tb_writer.add_scalar("wm_aux/ranking_acc", wm_ranking_acc_meter.avg, _wm_step)
            for key, value in predictor_camera_metrics.items():
                if key.startswith("predictor_per_camera_num_tokens/"):
                    continue
                camera_name = key.rsplit("/", 1)[1]
                token_count = predictor_camera_metrics.get(f"predictor_per_camera_num_tokens/{camera_name}", 1.0)
                predictor_camera_loss_meters.setdefault(key, AverageMeter()).update(value, n=token_count)
            if (
                rank == 0
                and getattr(config.planner, "planner_type", "transformer") == "diffusion"
                and ((itr % log_freq == 0) or (itr == ipe - 1))
            ):
                logger.info(
                    "[diffusion gate] cls_sample_valid_ratio=%.2f%%",
                    100.0 * cls_valid_ratio_meter.avg,
                )
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
                _new_lr,
                _new_wd,
                train_start_time,
                start_epoch,
                config.optimization.epochs,
                log_freq,
                config.planner.wta_loss_version,
                config.planner.awta_init_temperature,
                config.planner.awta_exp_base,
                config.planner.awta_min_temperature,
                config.planner.num_modes,
                planner_type=getattr(config.planner, "planner_type", "transformer"),
                vel_loss_meter=vel_loss_meter,
                yaw_loss_meter=yaw_loss_meter,
                cls_valid_ratio_meter=cls_valid_ratio_meter,
            )
            log_predictor_camera_training_metrics(
                tb_writer,
                predictor_camera_loss_meters,
                epoch,
                itr,
                ipe,
                rank,
                log_freq=log_freq,
            )

        validation_due = is_epoch_validation_due(val_loader, epoch, config.meta.val_freq)
        has_validation = False if navsim_e120_planner_active else config.planner.use_planner and validation_due
        has_predictor_validation = (
            False
            if navsim_e120_planner_active
            else predictor_validation_enabled and validation_due and not skip_predictor_supervision
        )
        has_value_validation = (
            False
            if navsim_e120_planner_active
            else value_head_trainable and is_epoch_validation_due(value_val_loader, epoch, config.meta.val_freq)
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
            has_validation,
        )

        return has_validation or has_predictor_validation or has_value_validation

    def _save_latest(epoch):
        save_checkpoint_fn(epoch + 1, latest_path, replace=True)

    def _save_periodic(epoch):
        periodic_epoch = resolve_periodic_checkpoint_epoch(
            epoch,
            periodic_one_based=(navsim_e120_planner_active or bool(config.meta.selection_checkpoint_epochs)),
            selection_checkpoint_epochs=config.meta.selection_checkpoint_epochs,
        )
        if navsim_e120_planner_active:
            path = formal_v2_navsim_e120_periodic_checkpoint_path(config.meta.folder, epoch=periodic_epoch)
        else:
            path = os.path.join(config.meta.folder, f"e{periodic_epoch}.pt")
        save_checkpoint_fn(epoch + 1, path, replace=False)

    def _run_validation_isolated(epoch):
        if has_value_validation:
            logger.info(f"Running fixed value-head validation at epoch {epoch + 1}...")
            value_val_metrics = run_value_validation(
                encoder=encoder,
                predictor=predictor,
                value_head=value_head,
                val_loader=value_val_loader,
                val_sampler=value_val_sampler,
                config=config,
                epoch=epoch + 1,
                rank=rank,
                world_size=world_size,
                token_ae=token_ae,
                runtime_normalize_reps=runtime_normalize_reps,
                multiview_fusion=multiview_fusion,
            )
            if rank == 0 and tb_writer is not None:
                for metric_name, metric_value in value_val_metrics.items():
                    tb_writer.add_scalar(f"value_validation/{metric_name}", metric_value, epoch + 1)
            current_value_loss = float(value_val_metrics["value_validation_loss"])
            if best_value_tracker.consider(current_value_loss, epoch + 1):
                save_checkpoint_fn(
                    epoch + 1,
                    best_value_path,
                    extra_state={"best_value_metrics": value_val_metrics},
                )
                logger.info(
                    "*** New best value checkpoint: epoch=%d loss=%.6f path=%s ***",
                    best_value_tracker.epoch,
                    best_value_tracker.loss,
                    best_value_path,
                )

        if has_predictor_validation:
            logger.info(f"Running predictor-only validation at epoch {epoch + 1}...")
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
                        multiview_fusion=multiview_fusion,
                        target_multiview_fusion=target_multiview_fusion,
                        reuse_context_as_target=reuse_context_as_target,
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
                    val_loader=predictor_val_loader,
                    val_sampler=predictor_val_sampler,
                    config=config,
                    epoch=epoch + 1,
                    rank=rank,
                    world_size=world_size,
                    token_ae=token_ae,
                    runtime_normalize_reps=runtime_normalize_reps,
                    multiview_fusion=multiview_fusion,
                    target_multiview_fusion=target_multiview_fusion,
                    reuse_context_as_target=reuse_context_as_target,
                )
            log_predictor_validation_metrics(
                tb_writer,
                predictor_val_metrics,
                epoch,
                rank,
            )

            # Track the best predictor checkpoint by validation loss. predictor_val_metrics
            # is all-reduced, so this decision is identical across ranks;
            # save_training_checkpoint writes on rank 0 only.
            is_best_predictor = (
                best_predictor_tracker.consider_suite_result(predictor_suite_result, epoch + 1)
                if predictor_suite_result is not None
                else best_predictor_tracker.consider(
                    float(predictor_val_metrics.get("predictor_loss", float("inf"))), epoch + 1
                )
            )
            if is_best_predictor:
                save_checkpoint_fn(
                    epoch + 1,
                    best_predictor_path,
                    extra_state={
                        "best_predictor_metrics": predictor_val_metrics,
                        **(
                            {"predictor_validation_signature": predictor_validation_signature}
                            if predictor_validation_signature is not None
                            else {}
                        ),
                    },
                )
                _flow = predictor_val_metrics.get("predictor_flow_loss")
                _x0 = predictor_val_metrics.get("predictor_x0_loss")
                logger.info(
                    f"*** 新最佳 predictor checkpoint! Epoch {best_predictor_tracker.epoch} | "
                    f"loss: {best_predictor_tracker.loss:.5f}"
                    + (f", flow: {float(_flow):.5f}" if _flow is not None else "")
                    + (f", x0: {float(_x0):.5f}" if _x0 is not None else "")
                    + " ***"
                )

        # Diff 9: Pass config instead of args to run_validation
        if has_validation:
            logger.info(f"Running Navsim validation at epoch {epoch + 1}...")
            if validation_suite_enabled:

                def run_validation_protocol(protocol):
                    protocol_metrics = run_validation(
                        encoder=encoder,
                        predictor=predictor,
                        planner=planner,
                        val_loader=val_loader,
                        val_sampler=val_sampler,
                        config=config,
                        epoch=epoch + 1,
                        rank=rank,
                        world_size=world_size,
                        use_tubelet_repeat=config.data.use_tubelet_repeat,
                        vis_output_dir=os.path.join(config.meta.folder, "val_vis", protocol.name),
                        token_ae=token_ae,
                        runtime_normalize_reps=runtime_normalize_reps,
                        multiview_fusion=multiview_fusion,
                        value_head=value_head,
                        cvoi_dual_value=cvoi_dual_value,
                        validation_rollout_horizon=protocol.horizon,
                    )
                    cohort_metrics = protocol_metrics.get("cohort_metrics")
                    if not isinstance(cohort_metrics, dict):
                        raise RuntimeError(f"validation protocol {protocol.name} did not return cohort_metrics")
                    return cohort_metrics

                suite_result = run_rollout_validation_suite(
                    run_validation_protocol,
                    horizons=config.validation_suite.horizons,
                    expected_weights=config.validation_suite.expected_weight_by_horizon,
                    metric_directions=validation_metric_directions,
                )
                val_metrics = flatten_validation_suite_result(suite_result)
            else:
                val_metrics = run_validation(
                    encoder=encoder,
                    predictor=predictor,
                    planner=planner,
                    val_loader=val_loader,
                    val_sampler=val_sampler,
                    config=config,
                    epoch=epoch + 1,
                    rank=rank,
                    world_size=world_size,
                    use_tubelet_repeat=config.data.use_tubelet_repeat,
                    vis_output_dir=os.path.join(config.meta.folder, "val_vis"),
                    token_ae=token_ae,
                    runtime_normalize_reps=runtime_normalize_reps,
                    multiview_fusion=multiview_fusion,
                    value_head=value_head,
                    cvoi_dual_value=cvoi_dual_value,
                    budget_controller=budget_controller,
                    validation_rollout_horizon=resolve_cvoi_validation_rollout_horizon(config),
                )
            current_record = build_validation_record(epoch + 1, val_metrics)
            validation_history.append(current_record)
            log_validation_metrics(
                tb_writer,
                csv_logger,
                val_metrics,
                epoch,
                rank,
                validation_history=validation_history,
            )
            if rank == 0 and tb_writer is not None and validation_suite_enabled:
                for metric_name, metric_value in sorted(val_metrics.items()):
                    if metric_name.startswith("validation_suite/") and isinstance(metric_value, (int, float)):
                        tb_writer.add_scalar(metric_name, metric_value, epoch + 1)
                tb_writer.flush()

            best_tracker.consider(current_record, val_metrics, save_fn=save_checkpoint_fn)

    def _run_validation(epoch):
        from app.vjepa_cowa_world_model.utils.eval_determinism import preserve_eval_rng_state

        with preserve_eval_rng_state(device):
            return _run_validation_isolated(epoch)

    # Phase 5: outer epoch iteration + checkpoint cadence + barrier + validation dispatch owned by
    # TrainingLoopRunner (composition); run_epoch keeps the full per-epoch body verbatim. Byte-identical.
    TrainingLoopRunner(
        start_epoch=start_epoch,
        epochs=config.optimization.epochs,
        ipe=ipe,
        rank=rank,
        checkpoint_freq=CHECKPOINT_FREQ,
        save_every_freq=config.meta.save_every_freq,
        save_from_epoch=config.meta.save_from_epoch,
        periodic_one_based=(navsim_e120_planner_active or bool(config.meta.selection_checkpoint_epochs)),
        selection_checkpoint_epochs=config.meta.selection_checkpoint_epochs,
    ).run(
        run_epoch=run_epoch,
        save_latest=_save_latest,
        save_periodic=_save_periodic,
        run_validation=_run_validation,
    )

    wait_for_checkpoint_save()  # 确保最后一次异步 checkpoint 写入完成
    if planner_uses_legacy_open_loop_selection(config.cvoi):
        log_training_summary(
            best_tracker.epoch,
            best_tracker.ade,
            best_tracker.fde,
            best_tracker.minade_k,
            best_tracker.minfde_k,
            best_path,
            tb_writer,
            rank,
            best_l2_avg=best_tracker.l2_avg,
            best_collision_rate=best_tracker.collision_rate,
            validation_history=validation_history,
            best_selector_label=OPEN_LOOP_SELECTION_RULE,
            best_predictor_epoch=best_predictor_tracker.epoch,
            best_predictor_loss=best_predictor_tracker.loss,
            best_predictor_path=best_predictor_path,
        )
    elif rank == 0:
        logger.info("NavSim e120 Planner completed without L2/collision selection or summary")
