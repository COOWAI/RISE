# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

"""
轨迹预测评估模块 (for train_command.py)
与 train_command.py 的 encoder/predictor/planner 调用方式保持一致

相比 val_giant.py 的修复:
1. prepare_status_feature 支持 use_states_for_planner / action_dim
2. predictor forward 支持 predictor_inference_consistent 模式
3. predictor forward 支持 use_states_for_predictor

评估指标:
- ADE (Average Displacement Error): 所有预测点与GT点的平均L2距离
- FDE (Final Displacement Error): 最后一个预测点与GT点的L2距离
- minADE@K / minFDE@K: 多模态轨迹的最小ADE/FDE
"""

import os
from numbers import Integral
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel  # noqa: F401

from app.vjepa_cowa_world_model.training.budget_control import (
    BudgetProfile,
    load_budget_controller_from_checkpoint,
    resolve_controller_budget_profile,
)
from app.vjepa_cowa_world_model.training.config import (
    is_factory_pretrained_main_encoder_config,
    resolve_main_encoder_num_time_steps,
    resolve_main_encoder_tokens_per_frame,
    resolve_planner_observed_token_mode,
    resolve_planner_use_observed_tokens,
)
from app.vjepa_cowa_world_model.training.cvoi_execution import (
    cvoi_planner_inference_noise,
    cvoi_sample_seed,
    resolve_cvoi_evaluation_seed,
)
from app.vjepa_cowa_world_model.training.cvoi_runtime import (
    apply_cvoi_planner_guidance,
    cvoi_enabled,
    cvoi_guidance_enabled,
)
from app.vjepa_cowa_world_model.training.encoder_inputs import build_tubelet_encoder_input
from app.vjepa_cowa_world_model.training.latent_value_guidance import (
    apply_latent_value_guidance,
    should_apply_value_guidance,
)
from app.vjepa_cowa_world_model.training.predictor_parallel import forward_parallel_predictor, use_parallel_predictor
from app.vjepa_cowa_world_model.training.predictor_stepping import (
    make_predictor_step_fn,
    predictor_autoregressive_rollout,
    validate_empty_future_planner_conditions,
)
from app.vjepa_cowa_world_model.training.runtimes.latent_action_runtime import build_joint_action_policy_output
from app.vjepa_cowa_world_model.training.runtimes.latent_diffusion_runtime import (
    resolve_latent_dit_sampler_params,
    sample_latent_dit_joint_action_predictor,
    sample_latent_dit_predictor,
    use_latent_dit_predictor,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (
    build_parallel_predictor_timeline_inputs,
    build_predictor_timeline_inputs,
    enforce_cvoi_zero_future_aux,
    forward_main_context,
)
from app.vjepa_cowa_world_model.training.trajectory_quality_metrics import (
    compute_trajectory_quality_metrics_per_sample,
)
from app.vjepa_cowa_world_model.training.validation_cohorts import (
    ValidationCohortAccumulator,
    select_real_collision_inputs,
)
from app.vjepa_cowa_world_model.training.validation_distributed import (
    raise_if_validation_failed,
    reduce_open_loop_validation_totals,
    wrap_validation_batch_error,
)
from app.vjepa_cowa_world_model.training.validation_rng import resolve_stable_sample_ids, validation_randn
from app.vjepa_cowa_world_model.training.validation_suite import (
    resolve_validation_rollout_end_step,
    truncate_validation_future_tokens,
)
from app.vjepa_cowa_world_model.training.value_planning import (
    score_trajectories_method1,
    value_planning_method1_enabled,
)

# 使用重构后的公共模块
from app.vjepa_cowa_world_model.utils import (
    build_observed_action_trajectory_history,
    prepare_inference_consistent_status_vector,
    prepare_status_feature,
    resolve_planner_use_drive_command,
)
from app.vjepa_cowa_world_model.utils.eval_determinism import extract_batch_metadata
from app.vjepa_cowa_world_model.utils.metrics import (
    WORLD4DRIVE_REPORTED_SECONDS,
    compute_collision_rate,
    compute_world4drive_l2_metrics,
    populate_point_l2_horizons,
    populate_world4drive_collision_horizons,
    populate_world4drive_l2_horizons,
)
from app.vjepa_cowa_world_model.utils.planner_training import (
    resolve_validation_target_timeline,
    resolve_validation_timestep_sec,
)
from app.vjepa_cowa_world_model.utils.visualization import visualize_multimodal_trajectory, visualize_trajectory
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _select_validation_collision_inputs(
    *,
    metadata,
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    segmentation: torch.Tensor,
    ego_poses: torch.Tensor,
) -> dict:
    """Keep CF placeholder geometry out of every validation safety metric."""

    if isinstance(metadata, Mapping) and "geometry_present" in metadata:
        return select_real_collision_inputs(
            metadata=metadata,
            pred_traj=pred_traj,
            gt_traj=gt_traj,
            segmentation=segmentation,
            ego_poses=ego_poses,
        )
    return {
        "pred_traj": pred_traj.cpu().numpy(),
        "gt_traj": gt_traj.cpu().numpy(),
        "segmentation": segmentation,
        "ego_poses": ego_poses.cpu().numpy(),
        "sample_count": int(pred_traj.shape[0]),
    }


def _unwrap_validation_module(module):
    """Return the local module so unequal validation shards never run DDP forward collectives."""

    return module.module if module is not None and hasattr(module, "module") else module


def _get_config_value(config, key, default=None):
    """
    兼容 dict 和 dataclass 两种配置类型的访问方式
    """
    if hasattr(config, key):
        return getattr(config, key)
    elif isinstance(config, dict):
        return config.get(key, default)
    return default


def _get_nested_config(config, *keys, default=None):
    """
    获取嵌套配置值，兼容 dict 和 dataclass
    """
    current = config
    for key in keys:
        if current is None:
            return default
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    return current if current is not None else default


def _validation_empty_rollout_is_intentional(
    *,
    validation_rollout_horizon: int | None,
    budget_profile: BudgetProfile | None,
) -> bool:
    """Return whether the active protocol explicitly requests rollout horizon zero."""

    suite_horizon = None if validation_rollout_horizon is None else int(validation_rollout_horizon)
    if suite_horizon is not None and suite_horizon < 0:
        raise ValueError(f"validation_rollout_horizon must be non-negative or None, got {suite_horizon}")
    budget_horizon = None if budget_profile is None else int(budget_profile.rollout_future_steps)
    if budget_horizon is not None and budget_horizon < 0:
        raise ValueError(f"budget profile rollout_future_steps must be non-negative, got {budget_horizon}")
    if suite_horizon is not None and budget_horizon is not None and suite_horizon != budget_horizon:
        raise ValueError(
            "validation suite horizon and budget profile disagree: " f"{suite_horizon} != {budget_horizon}"
        )
    active_horizon = suite_horizon if suite_horizon is not None else budget_horizon
    return active_horizon == 0


def _resolve_validation_rng_horizon(config, semantic_horizon: int | None) -> int | None:
    """Resolve the horizon component of the sample-keyed validation RNG key.

    Validation-suite protocols deliberately use common random numbers: full and
    every fixed rollout horizon differ only in their available future prefix, not
    in predictor/action/planner sampling noise. Legacy non-suite validation keeps
    its previous horizon-keyed behavior.
    """

    if semantic_horizon is None:
        return None
    if not isinstance(semantic_horizon, Integral) or isinstance(semantic_horizon, bool):
        raise TypeError(f"validation RNG horizon must be an integer or None, got {type(semantic_horizon).__name__}")
    resolved_horizon = int(semantic_horizon)
    if resolved_horizon < 0:
        raise ValueError(f"validation RNG horizon must be non-negative or None, got {resolved_horizon}")
    if bool(_get_nested_config(config, "validation_suite", "enabled", default=False)):
        return None
    return resolved_horizon


def _sample_latent_dit_validation_horizon(
    *,
    predictor,
    z_context: torch.Tensor,
    predictor_inputs,
    tokens_per_frame: int,
    num_observed_steps: int,
    runtime_normalize_reps: bool,
    config,
    full_initial_noise: torch.Tensor,
    validation_rollout_horizon: int | None,
) -> torch.Tensor:
    """Sample the actual Latent-DiT prefix requested by one planner protocol."""

    tokens_per_frame = int(tokens_per_frame)
    if full_initial_noise.ndim != 3 or full_initial_noise.shape[1] % tokens_per_frame != 0:
        raise ValueError(
            "Latent-DiT validation noise must be [B, N, D] and frame aligned, "
            f"got {full_initial_noise.shape}, tokens_per_frame={tokens_per_frame}"
        )
    if validation_rollout_horizon is None:
        initial_noise = full_initial_noise
        future_steps = None
    else:
        future_steps = int(validation_rollout_horizon)
        available_steps = full_initial_noise.shape[1] // tokens_per_frame
        if future_steps < 0 or future_steps > available_steps:
            raise ValueError(
                f"Latent-DiT validation horizon {future_steps} exceeds available future steps {available_steps}"
            )
        if future_steps == 0:
            return z_context.new_empty(z_context.shape[0], 0, z_context.shape[-1])
        initial_noise = full_initial_noise[:, : future_steps * tokens_per_frame]

    sampled = sample_latent_dit_predictor(
        predictor=predictor,
        z_context=z_context,
        predictor_inputs=predictor_inputs,
        tokens_per_frame=tokens_per_frame,
        num_observed_steps=num_observed_steps,
        runtime_normalize_reps=runtime_normalize_reps,
        config=config,
        initial_noise=initial_noise,
        future_steps=future_steps,
        **resolve_latent_dit_sampler_params(config).as_kwargs(),
    )
    expected_tokens = initial_noise.shape[1]
    if sampled.ndim != 3 or sampled.shape[1] != expected_tokens:
        raise ValueError(
            "Latent-DiT planner validation must return the requested active prefix only: "
            f"expected {expected_tokens} tokens, got {sampled.shape}"
        )
    return sampled


def _validate_validation_suite_runtime_contract(
    config,
    *,
    validation_rollout_horizon: int | None,
    budget_requested: bool,
) -> None:
    """Reject runtime modes that would silently produce a mislabeled horizon curve."""

    if not bool(_get_nested_config(config, "validation_suite", "enabled", default=False)):
        return
    if budget_requested:
        raise ValueError(
            "validation_suite cannot be combined with a budget controller/profile because full must retain "
            "the unbudgeted rollout semantics"
        )
    z_ar_mode = str(_get_nested_config(config, "planner", "z_ar_mode", default="full"))
    if z_ar_mode != "full":
        raise ValueError("validation_suite requires planner.z_ar_mode='full' to distinguish h0/h1/h2/h3")
    policy_output_source = str(
        _get_nested_config(config, "planner", "policy_output_source", default="planner")
    ).lower()
    if validation_rollout_horizon is not None and policy_output_source == "joint_action":
        raise ValueError(
            "validation_suite fixed horizons are unsupported for planner.policy_output_source='joint_action': "
            "the direct action policy bypasses the truncated predictor latent"
        )


def _prepare_validation_planner_conditioning(
    *,
    z_ar_planner: torch.Tensor,
    z: torch.Tensor,
    predictor_actions: torch.Tensor,
    tokens_per_frame: int,
    observed_steps: int,
    use_z_context: bool,
    use_observed_tokens: bool,
    use_action_history: bool,
    action_history_dim: int,
    timestep_sec: float,
    predictor_frame_stride: int,
) -> dict:
    """Build non-future planner inputs that remain available for rollout horizon zero."""

    z_context = z[:, :tokens_per_frame] if use_z_context else None
    z_observed = z[:, : observed_steps * tokens_per_frame] if use_observed_tokens else None
    action_history = None
    if use_action_history:
        action_history = build_observed_action_trajectory_history(
            predictor_actions,
            num_observed_frames=observed_steps,
            action_history_dim=action_history_dim,
            dt=float(timestep_sec) * max(int(predictor_frame_stride), 1),
        )
    validate_empty_future_planner_conditions(
        z_ar_planner,
        z_context=z_context,
        z_observed=z_observed,
        action_history=action_history,
    )
    return {"z_context": z_context, "z_observed": z_observed, "action_history": action_history}


# =====================================================================
#  Metric functions (unchanged from val_giant.py)
# =====================================================================


def compute_ade(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    pred_xy = pred_traj[..., :2]
    gt_xy = gt_traj[..., :2]
    displacement = torch.norm(pred_xy - gt_xy, dim=-1)
    return displacement.mean(dim=-1)


def compute_fde(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    pred_xy_final = pred_traj[:, -1, :2]
    gt_xy_final = gt_traj[:, -1, :2]
    return torch.norm(pred_xy_final - gt_xy_final, dim=-1)


def compute_metrics(pred_traj: torch.Tensor, gt_traj: torch.Tensor) -> dict:
    ade_per_sample = compute_ade(pred_traj, gt_traj)
    fde_per_sample = compute_fde(pred_traj, gt_traj)
    return {
        "ade": ade_per_sample.mean().item(),
        "fde": fde_per_sample.mean().item(),
        "ade_per_sample": ade_per_sample,
        "fde_per_sample": fde_per_sample,
    }


def compute_minade_minfde_k(pred_trajs: torch.Tensor, gt_traj: torch.Tensor) -> dict:
    pred_xy = pred_trajs[..., :2]
    gt_xy = gt_traj[:, None, :, :2]
    displacement = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, K, num_poses]

    ade_k = displacement.mean(dim=-1)  # [B, K]
    fde_k = displacement[:, :, -1]  # [B, K]

    minade_per_sample = ade_k.min(dim=1).values  # [B]
    minfde_per_sample = fde_k.min(dim=1).values  # [B]

    return {
        "minade_k": minade_per_sample.mean().item(),
        "minfde_k": minfde_per_sample.mean().item(),
        "minade_per_sample": minade_per_sample,
        "minfde_per_sample": minfde_per_sample,
    }


def compute_oracle_l2_task_score(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    timestep_sec: float = 0.5,
) -> torch.Tensor:
    """Return per-sample oracle reward as negative World4Drive ``l2_avg``.

    Parameters
    ----------
    pred_traj    : [B, T, 3] predicted selected trajectories.
    gt_traj      : [B, T, 3] ground-truth trajectories.
    timestep_sec : seconds per trajectory step.

    Returns
    -------
    torch.Tensor:
        [B] task score, where larger is better and ``score = -l2_avg``.
    """
    if pred_traj.ndim != 3 or gt_traj.ndim != 3:
        raise ValueError(f"pred_traj and gt_traj must be [B, T, 3], got {pred_traj.shape} and {gt_traj.shape}")
    if pred_traj.shape != gt_traj.shape:
        raise ValueError(f"pred_traj and gt_traj shapes must match, got {pred_traj.shape} and {gt_traj.shape}")
    if pred_traj.shape[-1] < 2:
        raise ValueError(f"trajectory last dim must contain x/y coordinates, got {pred_traj.shape[-1]}")
    if float(timestep_sec) <= 0.0:
        raise ValueError(f"timestep_sec must be > 0, got {timestep_sec}")

    pred_xy = pred_traj[..., :2]
    gt_xy = gt_traj[..., :2]
    l2 = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, T]

    reported_horizons = []
    for sec in WORLD4DRIVE_REPORTED_SECONDS:
        horizon_steps = int(round(float(sec) / float(timestep_sec)))
        if 0 < horizon_steps <= l2.shape[1]:
            reported_horizons.append(l2[:, :horizon_steps].mean(dim=1))
    if not reported_horizons:
        raise ValueError(
            "compute_oracle_l2_task_score: no reportable horizon within the trajectory "
            f"(timestep_sec={timestep_sec}, reported_seconds={WORLD4DRIVE_REPORTED_SECONDS}, "
            f"num_steps={l2.shape[1]})"
        )

    l2_avg_per_sample = torch.stack(reported_horizons, dim=1).mean(dim=1)
    return -l2_avg_per_sample


# =====================================================================
#  Helper functions
# =====================================================================


def _prepare_encoder_input(context_clips: torch.Tensor) -> torch.Tensor:
    """
    与 train_command_v2.forward_context() 中的 encoder 输入构造保持完全一致。

    输入:  [B, C, T, H, W]
    输出:  [B*T, C, 2, H, W]
    """
    return build_tubelet_encoder_input(context_clips)


def _compress_tokens_with_token_ae(token_ae, tokens: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Apply frozen Token AE compression on concatenated frame tokens."""
    ae_tokens_per_frame = int(getattr(token_ae, "tokens_per_frame"))
    expected_tokens = int(num_frames) * ae_tokens_per_frame
    if tokens.size(1) != expected_tokens:
        if tokens.size(1) % ae_tokens_per_frame != 0:
            raise ValueError(
                "Cannot infer TokenAE frame count: "
                f"tokens={tokens.size(1)}, num_frames={num_frames}, ae_tokens_per_frame={ae_tokens_per_frame}"
            )
        num_frames = tokens.size(1) // ae_tokens_per_frame
    return token_ae.encode(tokens, num_frames=num_frames)


# =====================================================================
#  Core validation
# =====================================================================


@torch.no_grad()
def validate_one_epoch(
    encoder,
    predictor,
    planner,
    val_loader,
    val_sampler,
    device,
    dtype,
    mixed_precision,
    tubelet_size,
    tokens_per_frame,
    num_poses,
    num_time_steps,
    world_size,
    rank,
    epoch,
    normalize_reps: bool = True,
    status_mode: str = "first",
    z_ar_mode: str = "full",
    use_z_context: bool = False,
    use_tubelet_repeat: bool = False,
    # --- fix #1: planner status 参数 ---
    use_states_for_planner: bool = False,
    action_dim: int = 7,
    # --- fix #2: inference_consistent 参数 ---
    predictor_inference_consistent: bool = False,
    num_observed_frames: int = 2,
    # --- fix #3: predictor states 输入模式 ---
    use_states_for_predictor: bool = True,
    # --- fix #4: predictor 完全无辅助输入 (train_giant_first.py) ---
    predictor_no_aux_input: bool = False,
    # --- fix #5: 观测帧 tokens ---
    use_observed_tokens: bool = False,
    # --- fix #6: IC 状态维度 ---
    state_dim: int = 7,
    # --- fix #6b: planner status 维度（与 predictor state_dim 解耦；默认回落到 state_dim） ---
    planner_status_dim: int = 0,
    # --- fix #6c: predictor / planner use_drive_command 分离 ---
    predictor_use_drive_command: bool = True,
    planner_use_drive_command: bool = True,
    # --- fix #7: diffusion planner anchor_state ---
    planner_type: str = "transformer",
    # --- NuScenes L2 和碰撞指标参数 ---
    timestep_sec: float = 0.5,
    compute_collision: bool = True,
    # --- 可视化参数 ---
    vis_output_dir: str = None,
    vis_every_n_batches: int = 50,
    vis_samples_per_batch: int = 20,
    token_ae=None,
    config=None,
    multiview_fusion=None,
    value_head=None,
    cvoi_dual_value=None,
    budget_oracle_recorder=None,
    budget_controller=None,
    budget_oracle_profile: BudgetProfile = None,
    validation_rollout_horizon: int | None = None,
) -> dict:
    """
    执行一个完整 epoch 的验证，与 train_command.py / train_giant_first.py 的前向逻辑完全对齐。
    """
    encoder_unwrapped = _unwrap_validation_module(encoder)
    predictor_unwrapped = _unwrap_validation_module(predictor)
    policy_output_source = str(
        _get_nested_config(config, "planner", "policy_output_source", default="planner")
    ).lower()
    if policy_output_source not in {"planner", "joint_action"}:
        raise ValueError(
            "planner.policy_output_source must be one of ['planner', 'joint_action'], " f"got {policy_output_source!r}"
        )
    direct_joint_action_policy = policy_output_source == "joint_action"
    planner_unwrapped = _unwrap_validation_module(planner)
    if planner_unwrapped is None and not direct_joint_action_policy:
        raise ValueError(
            "validate_one_epoch requires a learned planner unless planner.policy_output_source='joint_action'"
        )
    value_head_unwrapped = _unwrap_validation_module(value_head)
    cvoi_dual_value_unwrapped = _unwrap_validation_module(cvoi_dual_value)
    multiview_fusion_unwrapped = _unwrap_validation_module(multiview_fusion)

    encoder_was_training = encoder_unwrapped.training
    predictor_was_training = predictor_unwrapped.training
    planner_was_training = planner_unwrapped.training if planner_unwrapped is not None else False
    value_head_was_training = value_head_unwrapped.training if value_head_unwrapped is not None else False
    cvoi_dual_value_was_training = (
        cvoi_dual_value_unwrapped.training if cvoi_dual_value_unwrapped is not None else False
    )
    multiview_fusion_was_training = (
        multiview_fusion_unwrapped.training if multiview_fusion_unwrapped is not None else False
    )

    def _restore_validation_training_states():
        encoder_unwrapped.train(encoder_was_training)
        predictor_unwrapped.train(predictor_was_training)
        if planner_unwrapped is not None:
            planner_unwrapped.train(planner_was_training)
        if value_head_unwrapped is not None:
            value_head_unwrapped.train(value_head_was_training)
        if cvoi_dual_value_unwrapped is not None:
            cvoi_dual_value_unwrapped.train(cvoi_dual_value_was_training)
        if multiview_fusion_unwrapped is not None:
            multiview_fusion_unwrapped.train(multiview_fusion_was_training)

    encoder_unwrapped.eval()
    predictor_unwrapped.eval()
    if planner_unwrapped is not None:
        planner_unwrapped.eval()
    if value_head_unwrapped is not None:
        value_head_unwrapped.eval()
    if cvoi_dual_value_unwrapped is not None:
        cvoi_dual_value_unwrapped.eval()
    if multiview_fusion_unwrapped is not None:
        multiview_fusion_unwrapped.eval()

    runtime_token_ae = token_ae
    runtime_normalize_reps = normalize_reps
    factory_pretrained_main = config is not None and is_factory_pretrained_main_encoder_config(config)

    total_ade = 0.0
    total_fde = 0.0
    total_minade_k = 0.0
    total_minfde_k = 0.0
    total_samples = 0
    failed_batches = 0

    # L2 per timestep 和碰撞率累加变量
    total_l2_per_step = None  # 延迟初始化为 list[float]
    l2_total_samples = 0
    total_collision_counts = None  # 延迟初始化为 np.ndarray (box collision)
    total_point_collision_counts = None  # 延迟初始化为 np.ndarray (point collision)
    total_gt_collision_counts = None  # 延迟初始化为 np.ndarray (GT 碰撞排除数)
    collision_total_samples = 0
    missing_bev_segmentation_warned = False
    local_error = None
    validation_suite_enabled = bool(_get_nested_config(config, "validation_suite", "enabled", default=False))
    cohort_accumulator = (
        ValidationCohortAccumulator(
            num_steps=num_poses,
            timestep_sec=timestep_sec,
            device=device,
            world_size=world_size,
        )
        if validation_suite_enabled
        else None
    )

    val_sampler.set_epoch(epoch)

    batch_idx = -1
    try:
        val_iterator = iter(val_loader)
    except Exception as error:
        local_error = RuntimeError("Planner validation dataloader initialization failed")
        local_error.__cause__ = error
        val_iterator = iter(())
    while local_error is None:
        try:
            sample = next(val_iterator)
        except StopIteration:
            break
        except Exception as error:
            local_error = RuntimeError("Planner validation dataloader iteration failed")
            local_error.__cause__ = error
            break
        batch_idx += 1
        try:
            metadata = extract_batch_metadata(sample)
            metadata_valid_mask = metadata.get("metadata_valid_mask") if isinstance(metadata, dict) else None
            observed_metadata_valid_mask = (
                metadata.get("observed_metadata_valid_mask") if isinstance(metadata, dict) else None
            )
            context_frames = sample[0].to(device, non_blocking=True)
            actions = sample[1].to(device, dtype=torch.float, non_blocking=True)
            states = sample[2].to(device, dtype=torch.float, non_blocking=True)
            extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)

            # driving_command / ego_dynamics (NavSim 7-tuple collate)
            # 槽位值本身可为 None（B2D 数据缺 command_near/ego_vel 时 collate 置 None）。
            driving_command = (
                sample[5].to(device, dtype=torch.float, non_blocking=True)
                if len(sample) > 5 and sample[5] is not None
                else None
            )
            ego_dynamics = (
                sample[6].to(device, dtype=torch.float, non_blocking=True)
                if len(sample) > 6 and sample[6] is not None
                else None
            )

            # agent annotations (index 7, 8) — not directly used for collision;
            # kept in collate for backward compatibility but not extracted here.

            # Pre-computed BEV segmentation for new collision rate API (kept as numpy, CPU-side)
            bev_segmentation = sample[9].numpy() if len(sample) > 9 and sample[9] is not None else None
            if compute_collision and bev_segmentation is None and not missing_bev_segmentation_warned:
                logger.warning(
                    "Validation batches do not include BEV segmentation; " "collision metrics will be reported as inf."
                )
                missing_bev_segmentation_warned = True

            B = context_frames.shape[0]
            T = context_frames.shape[3] if context_frames.ndim == 6 else context_frames.shape[2]
            camera_metadata = {}
            if isinstance(metadata, dict):
                for key in ("camera_intrinsics", "camera2ego"):
                    value = metadata.get(key)
                    if torch.is_tensor(value):
                        camera_metadata[key] = value.to(device, dtype=torch.float, non_blocking=True)
            # fail-loud (point 30): 多视角融合启用时缺相机几何直接报错，与训练侧对齐，禁止 eval 静默用空几何。
            if multiview_fusion_unwrapped is not None:
                _missing_geom = [k for k in ("camera_intrinsics", "camera2ego") if k not in camera_metadata]
                if _missing_geom:
                    raise ValueError(
                        "Multi-view fusion requires camera_metadata with tensor camera_intrinsics and "
                        f"camera2ego at eval, but missing/invalid: {_missing_geom}."
                    )

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                # ============ Encoder Forward ============
                parallel_predictor = config is not None and use_parallel_predictor(config)
                latent_dit_predictor = config is not None and use_latent_dit_predictor(config)
                if direct_joint_action_policy and not latent_dit_predictor:
                    raise ValueError(
                        "planner.policy_output_source='joint_action' requires train.predictor_type='latent_dit'"
                    )
                joint_policy_sample = None
                # Only the *parallel* predictor uses the full-step (padded-to-T)
                # timeline. latent_dit without use_parallel_predictor is trained with
                # the autoregressive builder (actions length T-1); eval must use the
                # same builder so the side-condition action mean keeps the same
                # denominator (bug #4 train/eval skew).
                parallel_timeline = parallel_predictor
                if factory_pretrained_main or context_frames.ndim == 6:
                    z = forward_main_context(
                        encoder=encoder_unwrapped,
                        context_clips=context_frames,
                        config=config,
                        runtime_normalize_reps=runtime_normalize_reps,
                        token_ae=runtime_token_ae,
                        multiview_fusion=multiview_fusion_unwrapped,
                        camera_metadata=camera_metadata,
                    )
                    if parallel_timeline:
                        batch_timeline = build_parallel_predictor_timeline_inputs(
                            actions=actions,
                            states=states,
                            extrinsics=extrinsics,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            encoder=encoder_unwrapped,
                            dt=timestep_sec,
                            metadata_valid_mask=metadata_valid_mask,
                            observed_metadata_valid_mask=observed_metadata_valid_mask,
                        )
                    else:
                        batch_timeline = build_predictor_timeline_inputs(
                            actions=actions,
                            states=states,
                            extrinsics=extrinsics,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            encoder=encoder_unwrapped,
                            dt=timestep_sec,
                            metadata_valid_mask=metadata_valid_mask,
                            observed_metadata_valid_mask=observed_metadata_valid_mask,
                        )
                    if cvoi_enabled(config):
                        batch_timeline = enforce_cvoi_zero_future_aux(batch_timeline)
                    predictor_actions = batch_timeline.actions
                    predictor_states = batch_timeline.states
                    predictor_extrinsics = batch_timeline.extrinsics
                    predictor_driving_command = batch_timeline.driving_command
                    predictor_ego_dynamics = batch_timeline.ego_dynamics
                    batch_tokens_per_frame = batch_timeline.tokens_per_frame
                    predictor_total_steps = batch_timeline.num_time_steps
                    predictor_observed_steps = batch_timeline.num_observed_steps
                    predictor_frame_stride = batch_timeline.frame_stride
                else:
                    encoder_input = _prepare_encoder_input(context_frames)
                    z_context_out = encoder_unwrapped([encoder_input])
                    z = z_context_out[0]
                    z = z.view(B, T, -1, z.size(-1)).flatten(1, 2)

                    if runtime_token_ae is None and _get_nested_config(config, "token_ae", "enabled", default=False):
                        from app.vjepa_cowa_world_model.training.models import load_frozen_token_ae

                        runtime_token_ae, runtime_normalize_reps = load_frozen_token_ae(
                            config,
                            device=device,
                            encoder_embed_dim=z.size(-1),
                            tokens_per_frame=z.size(1) // T,
                            normalize_reps=runtime_normalize_reps,
                            dtype=dtype,
                        )

                    if runtime_token_ae is not None:
                        z = _compress_tokens_with_token_ae(runtime_token_ae, z, num_frames=T)

                    if runtime_normalize_reps:
                        z = F.layer_norm(z, (z.size(-1),))

                    predictor_actions = actions
                    if parallel_timeline and predictor_actions.shape[1] == T - 1:
                        predictor_actions = torch.cat(
                            [predictor_actions, predictor_actions.new_zeros(B, 1, predictor_actions.shape[-1])],
                            dim=1,
                        )
                    predictor_states = states
                    predictor_extrinsics = extrinsics
                    predictor_driving_command = driving_command
                    predictor_ego_dynamics = ego_dynamics
                    batch_tokens_per_frame = tokens_per_frame
                    predictor_total_steps = T
                    predictor_observed_steps = num_observed_frames
                    predictor_frame_stride = 1
                    batch_timeline = SimpleNamespace(
                        actions=predictor_actions,
                        states=predictor_states,
                        extrinsics=predictor_extrinsics,
                    )

                if z.shape[1] % batch_tokens_per_frame != 0:
                    raise ValueError(
                        f"z token length ({z.shape[1]}) must be divisible by tokens_per_frame "
                        f"({batch_tokens_per_frame})"
                    )
                context_steps = z.shape[1] // batch_tokens_per_frame
                if context_steps < predictor_observed_steps:
                    raise ValueError(
                        "Observed token sequence is shorter than predictor observed prefix: "
                        f"context_steps={context_steps}, observed_steps={predictor_observed_steps}"
                    )
                if context_steps > predictor_total_steps:
                    raise ValueError(
                        "Context token sequence is longer than the predictor timeline: "
                        f"context_steps={context_steps}, predictor_total_steps={predictor_total_steps}"
                    )
                expected_action_steps = predictor_total_steps if parallel_timeline else predictor_total_steps - 1
                if not (predictor_actions.shape[1] == expected_action_steps):
                    raise AssertionError(
                        f"actions shape mismatch: {predictor_actions.shape}, expected second dim to be "
                        f"{expected_action_steps}"
                    )
                if not (predictor_states.shape[1] == predictor_total_steps):
                    raise AssertionError(
                        f"states shape mismatch: {predictor_states.shape}, expected second dim to be "
                        f"{predictor_total_steps}"
                    )
                if not (predictor_extrinsics.shape[1] == predictor_total_steps):
                    raise AssertionError(
                        f"extrinsics shape mismatch: {predictor_extrinsics.shape}, expected second dim to be "
                        f"{predictor_total_steps}"
                    )

                budget_rollout_end_step = None
                active_budget_profile = budget_oracle_profile
                if active_budget_profile is not None and budget_controller is not None:
                    raise ValueError("validation received both budget_oracle_profile and budget_controller")
                if active_budget_profile is not None or budget_controller is not None:
                    if latent_dit_predictor or parallel_predictor:
                        raise ValueError(
                            "rollout budget controller currently supports only the non-parallel "
                            "ac_transformer autoregressive predictor path"
                        )
                    max_future_steps = int(predictor_total_steps) - int(predictor_observed_steps)
                    if max_future_steps <= 0:
                        raise ValueError(
                            "rollout budget requires predictor_total_steps > predictor_observed_steps, "
                            f"got {predictor_total_steps} <= {predictor_observed_steps}"
                        )
                    controller_budget = None
                    if budget_controller is not None:
                        z_budget_obs = z[:, : predictor_observed_steps * batch_tokens_per_frame]
                        controller_budget, active_budget_profile = resolve_controller_budget_profile(
                            budget_controller,
                            z_budget_obs,
                            config=config,
                            deterministic=True,
                            max_future_steps=max_future_steps,
                        )
                    if active_budget_profile is None:
                        raise ValueError("budget controller did not produce a rollout profile")
                    budget_rollout_end_step = int(predictor_observed_steps) + int(
                        active_budget_profile.rollout_future_steps
                    )
                    if batch_idx == 0:
                        budget_text = (
                            "fixed"
                            if controller_budget is None
                            else f"{float(controller_budget[0].detach().cpu()):.4f}"
                        )
                        logger.info(
                            "[budget_controller][val] batch %d: budget=%s profile=%s rollout_end_step=%d",
                            batch_idx,
                            budget_text,
                            active_budget_profile,
                            budget_rollout_end_step,
                        )
                rollout_end_step = resolve_validation_rollout_end_step(
                    validation_horizon=validation_rollout_horizon,
                    observed_steps=predictor_observed_steps,
                    total_steps=predictor_total_steps,
                    budget_rollout_end_step=budget_rollout_end_step,
                )
                # ============ Predictor Forward (与 train_command.py 对齐) ============
                if latent_dit_predictor:
                    stable_sample_ids = resolve_stable_sample_ids(metadata, batch_size=B)
                    base_validation_seed = int(_get_nested_config(config, "meta", "seed", default=0))
                    predictor_rng_horizon = _resolve_validation_rng_horizon(config, validation_rollout_horizon)
                    predictor_core = predictor_unwrapped
                    predictor_noise = validation_randn(
                        (B, int(predictor_core.num_future_tokens), int(predictor_core.embed_dim)),
                        base_seed=base_validation_seed,
                        sample_ids=stable_sample_ids,
                        protocol="navsim/predictor",
                        horizon=predictor_rng_horizon,
                        stream="world_initial_noise",
                        device=z.device,
                        dtype=z.dtype,
                    )
                    if direct_joint_action_policy:
                        action_noise = validation_randn(
                            (
                                B,
                                int(predictor_core.num_future_steps),
                                int(predictor_core.joint_action_dim),
                            ),
                            base_seed=base_validation_seed,
                            sample_ids=stable_sample_ids,
                            protocol="navsim/predictor",
                            horizon=predictor_rng_horizon,
                            stream="action_initial_noise",
                            device=z.device,
                            dtype=z.dtype,
                        )
                        joint_policy_sample = sample_latent_dit_joint_action_predictor(
                            predictor=predictor_unwrapped,
                            z_context=z,
                            predictor_inputs=batch_timeline,
                            tokens_per_frame=batch_tokens_per_frame,
                            num_observed_steps=predictor_observed_steps,
                            runtime_normalize_reps=runtime_normalize_reps,
                            config=config,
                            initial_world_noise=predictor_noise,
                            initial_action_noise=action_noise,
                            **resolve_latent_dit_sampler_params(config).as_kwargs(),
                        )
                        z_ar = joint_policy_sample.z_ar
                    else:
                        z_ar = _sample_latent_dit_validation_horizon(
                            predictor=predictor_unwrapped,
                            z_context=z,
                            predictor_inputs=batch_timeline,
                            tokens_per_frame=batch_tokens_per_frame,
                            num_observed_steps=predictor_observed_steps,
                            runtime_normalize_reps=runtime_normalize_reps,
                            config=config,
                            full_initial_noise=predictor_noise,
                            validation_rollout_horizon=validation_rollout_horizon,
                        )
                elif parallel_predictor:
                    parallel_output = forward_parallel_predictor(
                        predictor=predictor_unwrapped,
                        observed_tokens=z,
                        actions=predictor_actions,
                        states=predictor_states,
                        extrinsics=predictor_extrinsics,
                        config=config,
                        tokens_per_frame=batch_tokens_per_frame,
                        runtime_normalize_reps=runtime_normalize_reps,
                        num_observed_steps=predictor_observed_steps,
                        driving_command=predictor_driving_command,
                        ego_dynamics=predictor_ego_dynamics,
                        predictor_no_aux_input=predictor_no_aux_input,
                    )
                    z_ar = parallel_output.z_future
                else:
                    full_context_available = context_steps == predictor_total_steps
                    num_obs = predictor_observed_steps

                    _step_predictor = make_predictor_step_fn(
                        predictor_unwrapped,
                        config,
                        num_obs,
                        driving_command=predictor_driving_command,
                        ego_dynamics=predictor_ego_dynamics,
                        predictor_no_aux_input=predictor_no_aux_input,
                        normalize_reps=runtime_normalize_reps,
                    )

                    z_tf = None
                    if full_context_available:
                        # Teacher forcing 只有在验证 dataloader 提供完整未来图像 token 时才可用。
                        _z_enc = z[:, :-batch_tokens_per_frame]
                        _s, _e = predictor_states[:, :-1], predictor_extrinsics[:, :-1]
                        z_tf = _step_predictor(_z_enc, predictor_actions, _s, _e)
                    elif not predictor_inference_consistent:
                        raise ValueError(
                            "Observed-image planner validation requires predictor_inference_consistent=True "
                            "because teacher forcing is unavailable without future image tokens."
                        )

                    # Autoregressive rollout (shared with viz via predictor_autoregressive_rollout).
                    z_ar = predictor_autoregressive_rollout(
                        _step_predictor,
                        z,
                        predictor_actions,
                        predictor_states,
                        predictor_extrinsics,
                        num_obs=num_obs,
                        tokens_per_frame=batch_tokens_per_frame,
                        num_total=predictor_total_steps,
                        predictor_inference_consistent=predictor_inference_consistent,
                        z_tf=z_tf,
                        rollout_end_step=rollout_end_step,
                    )

                if not latent_dit_predictor:
                    z_ar = truncate_validation_future_tokens(
                        z_ar,
                        validation_horizon=validation_rollout_horizon,
                        tokens_per_frame=batch_tokens_per_frame,
                    )

                z_ar_planner = z_ar if z_ar_mode == "full" else z_ar[:, :batch_tokens_per_frame]
                empty_rollout_is_intentional = _validation_empty_rollout_is_intentional(
                    validation_rollout_horizon=validation_rollout_horizon,
                    budget_profile=active_budget_profile,
                )
                if cvoi_guidance_enabled(config):
                    z_observed = z[:, : predictor_observed_steps * batch_tokens_per_frame]
                    z_ar_planner, _cvoi_guidance_diag = apply_cvoi_planner_guidance(
                        z_observed,
                        z_ar_planner,
                        cvoi_dual_value_unwrapped,
                        tokens_per_frame=batch_tokens_per_frame,
                        config=config,
                    )
                    if batch_idx == 0:
                        logger.info(
                            "[cvoi_guidance][val] batch %d: field_before=%.5f field_after=%.5f "
                            "delta_norm=%.5f steps=%.0f h0_skip=%.0f",
                            batch_idx,
                            _cvoi_guidance_diag["field_value_before"],
                            _cvoi_guidance_diag["field_value_after"],
                            _cvoi_guidance_diag["delta_norm"],
                            _cvoi_guidance_diag["guidance_steps"],
                            _cvoi_guidance_diag["guidance_skipped_h0"],
                        )
                elif bool(_get_nested_config(config, "value_guidance", "enabled", default=False)):
                    if value_head_unwrapped is None:
                        raise ValueError("value_guidance.enabled=true requires value_head during validation")
                    if not should_apply_value_guidance(
                        z_ar_planner,
                        value_guidance_enabled=True,
                        allow_empty_rollout_skip=empty_rollout_is_intentional,
                    ):
                        _value_guidance_diag = None
                    else:
                        z_ar_planner, _value_guidance_diag = apply_latent_value_guidance(
                            z_ar_planner,
                            value_head_unwrapped,
                            tokens_per_frame=batch_tokens_per_frame,
                            config=config,
                        )
                        if batch_idx == 0:
                            logger.info(
                                "[value_guidance][val] batch %d: value_before=%.5f value_after=%.5f "
                                "delta_norm=%.5f steps=%.0f",
                                batch_idx,
                                _value_guidance_diag["value_before"],
                                _value_guidance_diag["value_after"],
                                _value_guidance_diag["delta_norm"],
                                _value_guidance_diag["guidance_steps"],
                            )

                # ============ Planner Forward ============
                if direct_joint_action_policy:
                    if joint_policy_sample is None:
                        raise ValueError("Direct joint action policy validation requires joint action sampler output")
                    planner_output = build_joint_action_policy_output(
                        joint_policy_sample.actions,
                        num_poses=num_poses,
                        frame_stride=max(int(predictor_frame_stride), 1),
                    )
                else:
                    if predictor_inference_consistent:
                        _planner_sd = planner_status_dim if planner_status_dim > 0 else state_dim
                        status_feature = prepare_inference_consistent_status_vector(
                            states,
                            num_observed=num_observed_frames,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            state_dim=_planner_sd,
                            use_drive_command=planner_use_drive_command,
                        )
                    else:
                        status_feature = prepare_status_feature(
                            states,
                            actions,
                            status_mode=status_mode,
                            use_states_for_planner=use_states_for_planner,
                            action_dim=action_dim,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                        )
                    planner_conditioning = _prepare_validation_planner_conditioning(
                        z_ar_planner=z_ar_planner,
                        z=z,
                        predictor_actions=predictor_actions,
                        tokens_per_frame=batch_tokens_per_frame,
                        observed_steps=predictor_observed_steps,
                        use_z_context=use_z_context,
                        use_observed_tokens=use_observed_tokens,
                        use_action_history=bool(
                            _get_nested_config(config, "planner", "use_action_history_for_planner", default=False)
                        ),
                        action_history_dim=int(_get_nested_config(config, "planner", "action_history_dim", default=3)),
                        timestep_sec=timestep_sec,
                        predictor_frame_stride=predictor_frame_stride,
                    )
                    z_first_frame = planner_conditioning["z_context"]
                    z_observed = planner_conditioning["z_observed"]
                    planner_action_history = planner_conditioning["action_history"]
                    if planner_type == "diffusion":
                        stable_sample_ids = resolve_stable_sample_ids(metadata, batch_size=B)
                        planner_modes = (
                            int(planner_unwrapped.num_modes)
                            if int(planner_unwrapped.num_modes) > 1
                            else int(planner_unwrapped.num_samples)
                        )
                        if rollout_end_step is None and z_ar_mode == "full":
                            planner_horizon = None
                        else:
                            if z_ar_planner.shape[1] % batch_tokens_per_frame != 0:
                                raise ValueError(
                                    "Planner validation prefix must be frame aligned before RNG keying: "
                                    f"tokens={z_ar_planner.shape[1]}, tokens_per_frame={batch_tokens_per_frame}"
                                )
                            planner_horizon = int(z_ar_planner.shape[1]) // int(batch_tokens_per_frame)
                        planner_rng_horizon = _resolve_validation_rng_horizon(config, planner_horizon)
                        if cvoi_enabled(config):
                            base_seed = resolve_cvoi_evaluation_seed(config)
                            planner_noise = cvoi_planner_inference_noise(
                                planner_unwrapped,
                                seeds=[cvoi_sample_seed(base_seed, sample_id) for sample_id in stable_sample_ids],
                                device=z_ar_planner.device,
                            )
                        else:
                            planner_noise = validation_randn(
                                (
                                    B,
                                    planner_modes,
                                    int(planner_unwrapped.num_poses),
                                    int(planner_unwrapped.traj_dim),
                                ),
                                base_seed=int(_get_nested_config(config, "meta", "seed", default=0)),
                                sample_ids=stable_sample_ids,
                                protocol="navsim/planner",
                                horizon=planner_rng_horizon,
                                stream="trajectory_initial_noise",
                                device=z_ar_planner.device,
                                dtype=torch.float32,
                            )
                        # 构造 anchor_state，与 train_planner_world_model.py 对齐
                        _anchor_state = None
                        if hasattr(planner_unwrapped, "use_anchor_frame") and planner_unwrapped.use_anchor_frame:
                            _future_start = num_observed_frames if predictor_inference_consistent else 1
                            _origin_idx = _future_start - 1
                            # origin 帧即坐标系原点: x=0, y=0, yaw=0 → cos=1, sin=0
                            if planner_unwrapped.traj_dim == 4:
                                _anchor_state = torch.stack(
                                    [
                                        torch.zeros(B, device=device),  # x = 0
                                        torch.zeros(B, device=device),  # y = 0
                                        torch.ones(B, device=device),  # cos(0) = 1
                                        torch.zeros(B, device=device),  # sin(0) = 0
                                    ],
                                    dim=-1,
                                ).float()  # [B, 4]
                            else:
                                # vx, vy 从 ego_dynamics 取该帧的真实速度
                                # fail-loud (point 32): 缺 ego_dynamics 时不得把 6D anchor 速度静默置零。
                                if ego_dynamics is None:
                                    raise ValueError(
                                        "6D anchor needs real ego_dynamics for vx/vy; got None (禁止静默置零)"
                                    )
                                _a_vx = ego_dynamics[:, _origin_idx, 0].float()
                                _a_vy = ego_dynamics[:, _origin_idx, 1].float()
                                _anchor_state = torch.stack(
                                    [
                                        torch.zeros(B, device=device),  # x = 0
                                        torch.zeros(B, device=device),  # y = 0
                                        _a_vx,  # vx
                                        _a_vy,  # vy
                                        torch.ones(B, device=device),  # cos(0) = 1
                                        torch.zeros(B, device=device),  # sin(0) = 0
                                    ],
                                    dim=-1,
                                ).float()  # [B, 6]
                        planner_output = planner_unwrapped(
                            z_ar_planner,
                            status_feature,
                            z_context=z_first_frame,
                            z_observed=z_observed,
                            action_history=planner_action_history,
                            anchor_state=_anchor_state,
                            inference_noise=planner_noise,
                        )
                    else:
                        planner_output = planner_unwrapped(
                            z_ar_planner,
                            status_feature,
                            z_context=z_first_frame,
                            z_observed=z_observed,
                            action_history=planner_action_history,
                        )

                pred_trajs = None
                if "trajectories" in planner_output:
                    from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output

                    # Fail-loud contract check before indexing (clear error instead of a bare KeyError /
                    # silent shape drift) — same inference contract the training lines already enforce.
                    validate_planner_output(planner_output, mode="inference")
                    pred_trajs = planner_output["trajectories"]  # [B, K, num_poses, 3]
                    pred_conf = planner_output["confidences"]  # [B, K]
                    if value_planning_method1_enabled(config):
                        value_sample_ids = resolve_stable_sample_ids(metadata, batch_size=B)
                        value_result = score_trajectories_method1(
                            predictor=predictor_unwrapped,
                            value_head=value_head_unwrapped,
                            z_context=z,
                            trajs=pred_trajs.float(),
                            actions=actions,
                            states=states,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            tokens_per_frame=batch_tokens_per_frame,
                            runtime_normalize_reps=bool(runtime_normalize_reps),
                            dt=float(timestep_sec),
                            predictor_observed_steps=predictor_observed_steps,
                            predictor_frame_stride=predictor_frame_stride,
                            confidences=pred_conf.float(),
                            validation_sample_ids=value_sample_ids,
                            validation_base_seed=int(_get_nested_config(config, "meta", "seed", default=0)),
                            validation_protocol="navsim/value-method1",
                        )
                        traj_output = value_result["value_selected_trajectory"]
                    else:
                        best_idx = pred_conf.argmax(dim=1)
                        best_idx_exp = best_idx.view(-1, 1, 1, 1).expand(
                            -1, 1, pred_trajs.shape[2], pred_trajs.shape[3]
                        )
                        traj_output = pred_trajs.gather(1, best_idx_exp).squeeze(1)
                else:
                    traj_output = planner_output["trajectory"]

                # Cast to float32 to avoid BFloat16 errors in metrics / numpy
                traj_output = traj_output.float()
                if pred_trajs is not None:
                    pred_trajs = pred_trajs.float()

                # ============ GT 轨迹转换 ============
                # Use float64 for position diff to guard against large UTM values.
                StateSE2_indices = [0, 1, 5]
                states_se2 = states[:, :, StateSE2_indices].double()

                future_start_idx = num_observed_frames if predictor_inference_consistent else 1
                origin_idx = future_start_idx - 1
                origin_x = states_se2[:, origin_idx, 0]
                origin_y = states_se2[:, origin_idx, 1]
                origin_yaw = states_se2[:, origin_idx, 2]

                dx = states_se2[:, future_start_idx:, 0] - origin_x[:, None]
                dy = states_se2[:, future_start_idx:, 1] - origin_y[:, None]
                dyaw = states_se2[:, future_start_idx:, 2] - origin_yaw[:, None]

                cos_h = torch.cos(-origin_yaw)
                sin_h = torch.sin(-origin_yaw)
                ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
                ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
                ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

                gt_trajectory = torch.stack([ego_x, ego_y, ego_yaw], dim=-1).float()
                gt_trajectory = gt_trajectory[:, :num_poses]

                # ============ 计算指标 ============
                metrics = compute_metrics(traj_output, gt_trajectory)
                batch_ade = metrics["ade"]
                batch_fde = metrics["fde"]
                if pred_trajs is not None:
                    min_metrics = compute_minade_minfde_k(pred_trajs, gt_trajectory)
                    batch_minade_k = min_metrics["minade_k"]
                    batch_minfde_k = min_metrics["minfde_k"]
                else:
                    batch_minade_k = batch_ade
                    batch_minfde_k = batch_fde
                    min_metrics = {
                        "minade_per_sample": metrics["ade_per_sample"],
                        "minfde_per_sample": metrics["fde_per_sample"],
                    }

                if budget_oracle_recorder is not None:
                    oracle_score_per_sample = compute_oracle_l2_task_score(
                        traj_output,
                        gt_trajectory,
                        timestep_sec=timestep_sec,
                    ).detach()
                    z_observed_for_controller = z[:, : predictor_observed_steps * batch_tokens_per_frame]
                    budget_oracle_recorder.record_batch(
                        metadata=metadata,
                        pooled_latent=z_observed_for_controller.float().mean(dim=1),
                        task_score=oracle_score_per_sample.float(),
                        batch_idx=batch_idx,
                    )

                # ============ L2 per timestep ============
                l2_metrics = compute_world4drive_l2_metrics(
                    traj_output,
                    gt_trajectory,
                    timestep_sec=timestep_sec,
                )
                if cohort_accumulator is not None:
                    if ego_dynamics is None:
                        raise ValueError(
                            "validation suite progress/comfort metrics require observed ego_dynamics; got None"
                        )
                    if origin_idx < 1:
                        raise ValueError(
                            "validation suite yaw-rate boundary requires at least two observed states, "
                            f"got origin_idx={origin_idx}"
                        )
                    if ego_dynamics.ndim != 3 or ego_dynamics.shape[0] != B or ego_dynamics.shape[2] < 4:
                        raise ValueError(
                            "validation suite ego_dynamics must have shape [B, T, >=4] for vx/vy/ax/ay, "
                            f"got {ego_dynamics.shape}"
                        )
                    if ego_dynamics.shape[1] <= origin_idx or states.shape[1] <= origin_idx:
                        raise ValueError(
                            "validation suite observed boundary index exceeds state/dynamics timeline: "
                            f"origin_idx={origin_idx}, states={states.shape}, ego_dynamics={ego_dynamics.shape}"
                        )
                    observed_yaw_delta = states[:, origin_idx, 5] - states[:, origin_idx - 1, 5]
                    anchor_yaw_rate = torch.atan2(
                        torch.sin(observed_yaw_delta),
                        torch.cos(observed_yaw_delta),
                    ) / float(timestep_sec)
                    quality_metrics = compute_trajectory_quality_metrics_per_sample(
                        traj_output,
                        anchor_velocity=ego_dynamics[:, origin_idx, :2].float(),
                        anchor_acceleration=ego_dynamics[:, origin_idx, 2:4].float(),
                        anchor_yaw_rate=anchor_yaw_rate.float(),
                        timestep_sec=timestep_sec,
                    )
                    cohort_accumulator.add_batch(
                        metadata=metadata,
                        ade=metrics["ade_per_sample"],
                        fde=metrics["fde_per_sample"],
                        minade_k=min_metrics["minade_per_sample"],
                        minfde_k=min_metrics["minfde_per_sample"],
                        l2_per_step=torch.norm(traj_output[..., :2] - gt_trajectory[..., :2], dim=-1),
                        quality_metrics=quality_metrics,
                    )

                # ============ Collision rate ============
                if compute_collision and bev_segmentation is not None:
                    # 切出 future-only 的 BEV seg maps: [B, T_future, H, W]
                    bev_seg_future = bev_segmentation[:, future_start_idx:, :, :]
                    # 确保时间步对齐
                    T_future_traj = traj_output.shape[1]
                    bev_seg_future = bev_seg_future[:, :T_future_traj]
                    collision_inputs = _select_validation_collision_inputs(
                        metadata=metadata,
                        pred_traj=traj_output,
                        gt_traj=gt_trajectory,
                        segmentation=bev_seg_future,
                        ego_poses=states,
                    )
                    collision_batch_samples = int(collision_inputs["sample_count"])
                    collision_metrics = (
                        compute_collision_rate(
                            pred_traj=collision_inputs["pred_traj"],
                            gt_traj=collision_inputs["gt_traj"],
                            segmentation=collision_inputs["segmentation"],
                            ego_poses=collision_inputs["ego_poses"],
                            future_start_idx=future_start_idx,
                            timestep_sec=timestep_sec,
                            reference_frame_idx=origin_idx,
                        )
                        if collision_batch_samples > 0
                        else None
                    )
                else:
                    collision_metrics = None
                    collision_batch_samples = 0

                total_ade += batch_ade * B
                total_fde += batch_fde * B
                total_minade_k += batch_minade_k * B
                total_minfde_k += batch_minfde_k * B
                total_samples += B

                # L2: 累加每步 L2 × batch_size
                l2_values = l2_metrics["l2_per_step"]
                if l2_values and len(l2_values) != num_poses:
                    raise ValueError(f"Validation L2 horizon must be {num_poses}, got {len(l2_values)}")
                if l2_values:
                    if total_l2_per_step is None:
                        total_l2_per_step = [v * B for v in l2_values]
                    else:
                        for i, value in enumerate(l2_values):
                            total_l2_per_step[i] += value * B
                    l2_total_samples += B

                # 碰撞: 累加 raw counts (box, point, gt_exclusion)
                if collision_metrics is not None and "collision_counts" in collision_metrics:
                    cc = np.array(collision_metrics["collision_counts"], dtype=np.int64)
                    pc = np.array(collision_metrics["point_collision_counts"], dtype=np.int64)
                    gc = np.array(collision_metrics["gt_collision_counts"], dtype=np.int64)
                    collision_lengths = (len(cc), len(pc), len(gc))
                    if collision_lengths != (num_poses, num_poses, num_poses):
                        raise ValueError(
                            f"Validation collision horizons must each be {num_poses}, got {collision_lengths}"
                        )
                    if total_collision_counts is None:
                        total_collision_counts = cc
                        total_point_collision_counts = pc
                        total_gt_collision_counts = gc
                    else:
                        total_collision_counts += cc
                        total_point_collision_counts += pc
                        total_gt_collision_counts += gc
                    collision_total_samples += collision_batch_samples
                    if cohort_accumulator is not None:
                        cohort_accumulator.add_real_collision_counts(
                            metadata=metadata,
                            box_counts=cc,
                            point_counts=pc,
                            gt_counts=gc,
                        )

            if batch_idx % 50 == 0:
                logger.info(
                    f"Validation Epoch {epoch}, Batch {batch_idx}/{len(val_loader)}, "
                    f"ADE: {batch_ade:.4f}, FDE: {batch_fde:.4f}, "
                    f"minADE@K: {batch_minade_k:.4f}, minFDE@K: {batch_minfde_k:.4f}"
                )

            # Visualization: sample every N batches, rank 0 only
            if vis_output_dir and rank == 0 and batch_idx % vis_every_n_batches == 0:
                if pred_trajs is not None:
                    visualize_multimodal_trajectory(
                        pred_trajs=pred_trajs,
                        pred_conf=pred_conf,
                        gt_traj=gt_trajectory,
                        output_dir=vis_output_dir,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        limit=vis_samples_per_batch,
                    )
                else:
                    visualize_trajectory(
                        pred_traj=traj_output,
                        gt_traj=gt_trajectory,
                        output_dir=vis_output_dir,
                        epoch=epoch,
                        itr=batch_idx,
                        limit=vis_samples_per_batch,
                    )

        except Exception as e:
            local_error = wrap_validation_batch_error("Planner validation", batch_idx, e)
            break

    _restore_validation_training_states()
    raise_if_validation_failed(
        local_error,
        validation_name="Planner validation",
        device=device,
        world_size=world_size,
    )

    collision_sums = None
    if total_collision_counts is not None:
        collision_sums = (
            total_collision_counts,
            total_point_collision_counts,
            total_gt_collision_counts,
        )
    reduced = reduce_open_loop_validation_totals(
        metric_sums=[total_ade, total_fde, total_minade_k, total_minfde_k],
        total_samples=total_samples,
        l2_sums=total_l2_per_step,
        l2_samples=l2_total_samples,
        collision_sums=collision_sums,
        collision_samples=collision_total_samples,
        num_steps=num_poses,
        device=device,
        world_size=world_size,
    )
    total_ade, total_fde, total_minade_k, total_minfde_k = reduced["metric_sums"]
    total_samples = int(reduced["total_samples"])
    l2_total_samples = int(reduced["l2_samples"])
    total_l2_per_step = np.asarray(reduced["l2_sums"], dtype=np.float64)
    total_collision_counts = np.asarray(reduced["box_collision_sums"], dtype=np.float64)
    total_point_collision_counts = np.asarray(reduced["point_collision_sums"], dtype=np.float64)
    total_gt_collision_counts = np.asarray(reduced["gt_collision_sums"], dtype=np.float64)
    collision_total_samples = int(reduced["collision_samples"])

    if total_samples == 0:
        raise RuntimeError(
            f"Validation produced zero global successful samples (failed_batches={failed_batches}). "
            "Please check model interfaces and validation data pipeline."
        )

    avg_ade = total_ade / total_samples
    avg_fde = total_fde / total_samples
    avg_minade_k = total_minade_k / total_samples
    avg_minfde_k = total_minfde_k / total_samples

    # 计算最终 L2 per-timestep 指标
    result = {
        "ade": avg_ade,
        "fde": avg_fde,
        "minade_k": avg_minade_k,
        "minfde_k": avg_minfde_k,
    }

    if l2_total_samples > 0:
        total_l2_per_step = total_l2_per_step / l2_total_samples
        result["l2_per_step"] = total_l2_per_step.tolist()
        populate_world4drive_l2_horizons(result, total_l2_per_step, timestep_sec)
        populate_point_l2_horizons(result, total_l2_per_step, timestep_sec)

    if collision_total_samples > 0:
        box_collision_per_step = total_collision_counts / float(collision_total_samples)
        point_collision_per_step = total_point_collision_counts / float(collision_total_samples)
        result["collision_per_step"] = box_collision_per_step.tolist()
        result["point_collision_per_step"] = point_collision_per_step.tolist()
        result["collision_counts"] = total_collision_counts.tolist()
        result["point_collision_counts"] = total_point_collision_counts.tolist()
        result["gt_collision_counts"] = total_gt_collision_counts.tolist()
        populate_world4drive_collision_horizons(
            result,
            total_collision_counts,
            total_samples=collision_total_samples,
            timestep_sec=timestep_sec,
            metric_prefix="collision",
            avg_key="collision_rate",
        )
        populate_world4drive_collision_horizons(
            result,
            total_point_collision_counts,
            total_samples=collision_total_samples,
            timestep_sec=timestep_sec,
            metric_prefix="point_collision",
            avg_key="point_collision_rate",
        )
    elif compute_collision:
        result["collision_rate"] = float("inf")
        result["point_collision_rate"] = float("inf")

    if cohort_accumulator is not None:
        cohort_metrics = cohort_accumulator.finalize(
            require_primary_collision=validation_rollout_horizon is None and compute_collision,
        )
        result = dict(cohort_metrics["real"]["all"])
        result["cohort_metrics"] = cohort_metrics

    return result


# =====================================================================
#  Entry point
# =====================================================================


def run_validation(
    encoder,
    predictor,
    planner,
    val_loader,
    val_sampler,
    config: dict,
    epoch: int,
    rank: int,
    world_size: int,
    use_tubelet_repeat: bool = False,
    vis_output_dir: str = None,
    token_ae=None,
    runtime_normalize_reps=None,
    multiview_fusion=None,
    value_head=None,
    cvoi_dual_value=None,
    budget_oracle_recorder=None,
    budget_controller=None,
    budget_oracle_profile: BudgetProfile = None,
    validation_rollout_horizon: int | None = None,
) -> dict:
    """
    运行验证的入口函数 (for train_command.py)

    从 config 中自动读取所有与 train_command.py 对齐所需的参数，
    包括 predictor_inference_consistent 等。
    """
    # 设备
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    budget_controller_active = (
        bool(_get_nested_config(config, "budget_controller", "enabled", default=False))
        and _get_nested_config(config, "budget_controller", "mode", default=None) == "eval"
    )
    _validate_validation_suite_runtime_contract(
        config,
        validation_rollout_horizon=validation_rollout_horizon,
        budget_requested=bool(budget_controller_active or budget_controller is not None or budget_oracle_profile),
    )
    if budget_controller_active and budget_controller is None:
        budget_controller_checkpoint = _get_nested_config(
            config, "budget_controller", "controller_checkpoint", default=None
        )
        if not budget_controller_checkpoint:
            raise ValueError(
                "budget_controller.enabled=true with mode='eval' requires " "budget_controller.controller_checkpoint"
            )
        budget_controller = load_budget_controller_from_checkpoint(budget_controller_checkpoint, device=device)
        logger.info("budget_controller: loaded eval controller from %s", budget_controller_checkpoint)
    if budget_controller is not None:
        budget_controller.eval()

    # 数据类型 - 兼容 dict 和 dataclass
    which_dtype = _get_nested_config(config, "meta", "dtype", default="float32")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # 数据配置 - 兼容 dict 和 dataclass
    tubelet_size = _get_nested_config(config, "data", "tubelet_size", default=2)
    target_frame = _get_nested_config(config, "data", "num_target_frames", default=16)
    tokens_per_frame = resolve_main_encoder_tokens_per_frame(config, encoder)

    # Loss - representation normalization.
    # Resolve exactly as training does: the predictor pretrain config provides the
    # base value and an enabled TokenAE checkpoint overrides it. Reading the raw
    # experiment config here would cause a train/inference mismatch on the planner
    # input whenever loss.normalize_reps disagrees with the predictor/TokenAE.
    if runtime_normalize_reps is not None:
        normalize_reps = bool(runtime_normalize_reps)
    else:
        from app.vjepa_cowa_world_model.training.models import resolve_runtime_normalize_reps

        normalize_reps = resolve_runtime_normalize_reps(config, token_ae=token_ae)

    # Planner - 兼容 dict 和 dataclass
    status_mode = _get_nested_config(config, "planner", "states_mode", default="first")
    z_ar_mode = _get_nested_config(config, "planner", "z_ar_mode", default="full")
    use_z_context = _get_nested_config(config, "planner", "use_z_context", default=False)
    use_states_for_planner = _get_nested_config(config, "planner", "use_states_for_planner", default=True)
    observed_token_mode = resolve_planner_observed_token_mode(config)
    use_observed_tokens = resolve_planner_use_observed_tokens(config)
    planner_type = _get_nested_config(config, "planner", "planner_type", default="transformer")
    fps = _get_nested_config(config, "data", "fps", default=None)
    diff_dt = _get_nested_config(config, "planner", "diff_dt", default=None)
    timestep_sec = resolve_validation_timestep_sec(fps=fps, diff_dt=diff_dt, default=0.5)
    if not (z_ar_mode in ("full", "first_step")):
        raise AssertionError(f"Invalid planner.z_ar_mode={z_ar_mode}")
    if use_observed_tokens and z_ar_mode != "full":
        raise ValueError(
            f"planner.observed_token_mode={observed_token_mode!r} requires planner.z_ar_mode='full' in validation"
        )
    if fps is not None and fps > 0 and diff_dt is not None and diff_dt > 0:
        data_timestep_sec = 1.0 / float(fps)
        if abs(data_timestep_sec - float(diff_dt)) > 1e-6:
            logger.warning(
                "Validation timestep mismatch: data.fps=%.4f -> %.4fs, planner.diff_dt=%.4fs. "
                "Using %.4fs for metrics.",
                float(fps),
                data_timestep_sec,
                float(diff_dt),
                timestep_sec,
            )

    # Train 配置 — predictor 输入模式 - 兼容 dict 和 dataclass
    use_states_for_predictor = _get_nested_config(config, "train", "use_states_for_predictor", default=True)
    predictor_no_aux_input = _get_nested_config(config, "train", "predictor_no_aux_input", default=False)
    if predictor_no_aux_input:
        use_states_for_predictor = False
    action_dim = _get_nested_config(config, "train", "action_dim", default=7)
    configured_predictor_inference_consistent = _get_nested_config(
        config, "train", "predictor_inference_consistent", default=False
    )
    num_observed_frames = _get_nested_config(config, "train", "num_observed_frames", default=2)
    state_dim = _get_nested_config(config, "train", "state_dim", default=7)
    predictor_use_drive_command = _get_nested_config(config, "train", "use_drive_command", default=True)
    planner_use_drive_command = resolve_planner_use_drive_command(config)
    planner_status_dim_cfg = _get_nested_config(config, "planner", "status_dim", default=0)
    planner_target_timeline = resolve_validation_target_timeline(
        num_target_frames=target_frame,
        num_observed_frames=num_observed_frames,
        predictor_inference_consistent=configured_predictor_inference_consistent,
        predictor_no_aux_input=predictor_no_aux_input,
    )
    predictor_inference_consistent = bool(planner_target_timeline["predictor_inference_consistent"])
    num_poses = int(planner_target_timeline["num_poses"])
    num_time_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_poses, encoder=encoder)
    if predictor_no_aux_input and configured_predictor_inference_consistent:
        logger.warning(
            "predictor_no_aux_input=True and predictor_inference_consistent=True "
            "are mutually exclusive; disabling predictor_inference_consistent"
        )

    logger.info(f"Starting validation for epoch {epoch}...")
    logger.info(
        f"Validation config: tokens_per_frame={tokens_per_frame}, num_poses={num_poses}, "
        f"num_time_steps={num_time_steps}, normalize_reps={normalize_reps}, "
        f"status_mode={status_mode}, z_ar_mode={z_ar_mode}, use_z_context={use_z_context}, "
        f"use_tubelet_repeat={use_tubelet_repeat}, "
        f"predictor_no_aux_input={predictor_no_aux_input}, "
        f"use_states_for_predictor={use_states_for_predictor}, "
        f"use_states_for_planner={use_states_for_planner}, "
        f"predictor_inference_consistent={predictor_inference_consistent}, "
        f"num_observed_frames={num_observed_frames}, action_dim={action_dim}, "
        f"predictor_use_drive_command={predictor_use_drive_command}, "
        f"planner_use_drive_command={planner_use_drive_command}, "
        f"observed_token_mode={observed_token_mode}, use_observed_tokens={use_observed_tokens}, "
        f"planner_type={planner_type}, "
        f"timestep_sec={timestep_sec:.4f}"
    )

    metrics = validate_one_epoch(
        encoder=encoder,
        predictor=predictor,
        planner=planner,
        val_loader=val_loader,
        val_sampler=val_sampler,
        device=device,
        dtype=dtype,
        mixed_precision=mixed_precision,
        tubelet_size=tubelet_size,
        tokens_per_frame=tokens_per_frame,
        num_poses=num_poses,
        num_time_steps=num_time_steps,
        world_size=world_size,
        rank=rank,
        epoch=epoch,
        normalize_reps=normalize_reps,
        status_mode=status_mode,
        z_ar_mode=z_ar_mode,
        use_z_context=use_z_context,
        use_tubelet_repeat=use_tubelet_repeat,
        # fix #1
        use_states_for_planner=use_states_for_planner,
        action_dim=action_dim,
        # fix #2
        predictor_inference_consistent=predictor_inference_consistent,
        num_observed_frames=num_observed_frames,
        # fix #3
        use_states_for_predictor=use_states_for_predictor,
        # fix #4
        predictor_no_aux_input=predictor_no_aux_input,
        # fix #5: 观测帧 tokens
        use_observed_tokens=use_observed_tokens,
        state_dim=state_dim,
        planner_status_dim=planner_status_dim_cfg,
        predictor_use_drive_command=predictor_use_drive_command,
        planner_use_drive_command=planner_use_drive_command,
        # fix #7: diffusion planner type
        planner_type=planner_type,
        timestep_sec=timestep_sec,
        # visualization
        vis_output_dir=vis_output_dir,
        token_ae=token_ae,
        config=config,
        multiview_fusion=multiview_fusion,
        value_head=value_head,
        cvoi_dual_value=cvoi_dual_value,
        budget_oracle_recorder=budget_oracle_recorder,
        budget_controller=budget_controller,
        budget_oracle_profile=budget_oracle_profile,
        validation_rollout_horizon=validation_rollout_horizon,
    )

    if rank == 0:
        logger.info("=" * 50)
        logger.info(f"Validation Results - Epoch {epoch}:")
        logger.info(f"  ADE (Average Displacement Error): {metrics['ade']:.4f} m")
        logger.info(f"  FDE (Final Displacement Error):    {metrics['fde']:.4f} m")
        logger.info(f"  minADE@K:                         {metrics['minade_k']:.4f} m")
        logger.info(f"  minFDE@K:                         {metrics['minfde_k']:.4f} m")
        # L2 per timestep
        if "l2_avg" in metrics:
            logger.info(f"  avg L2: {metrics['l2_avg']:.4f} m")
            for sec in WORLD4DRIVE_REPORTED_SECONDS:
                key = f"l2_at_{sec}s"
                if key in metrics:
                    logger.info(f"  L2@{sec}s: {metrics[key]:.4f} m")
            if "l2_point_avg" in metrics:
                logger.info(f"  point L2 avg: {metrics['l2_point_avg']:.4f} m")
                for sec in WORLD4DRIVE_REPORTED_SECONDS:
                    key = f"l2_point_at_{sec}s"
                    if key in metrics:
                        logger.info(f"  PointL2@{sec}s: {metrics[key]:.4f} m")
        # Collision rate
        if "collision_rate" in metrics:
            logger.info(f"  Collision Rate: {metrics['collision_rate']:.4f}")
            for sec in WORLD4DRIVE_REPORTED_SECONDS:
                key = f"collision_at_{sec}s"
                if key in metrics:
                    logger.info(f"  Collision@{sec}s: {metrics[key]:.4f}")
        logger.info("=" * 50)

    return metrics
