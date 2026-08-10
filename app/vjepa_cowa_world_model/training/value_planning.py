"""Variant A Method 1 value planning utilities."""

import math
import struct
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Callable, Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.counterfactual_supervision import (
    CounterfactualSampleMasks,
    distributed_masked_mean,
)


@dataclass
class ValuePlanningLossConfig:
    gamma: float = 0.99
    lambda_return: float = 0.8
    bootstrap_horizon: int = 3
    progress_weight: float = 1.0
    comfort_weight: float = 0.2
    value_loss_weight: float = 1.0
    td_loss_weight: float = 1.0
    safe_floor_weight: float = 0.05
    episode_ranking_weight: float = 0.1
    episode_ranking_margin: float = 1.0
    srpo_shaping_weight: float = 0.0
    srpo_potential_based: bool = True
    pred_consistency_weight: float = 0.0


RolloutFn = Callable[..., torch.Tensor]


def _config_value(config: Any, name: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _value_cfg(config: Any) -> Any:
    return _config_value(config, "value_planning", config)


def _uses_latent_dit_predictor(config: Any) -> bool:
    train_cfg = _config_value(config, "train", None)
    predictor_type = str(_config_value(train_cfg, "predictor_type", "")).lower()
    return predictor_type == "latent_dit"


def _uses_fixed_future_predictor(config: Any) -> bool:
    if not _uses_latent_dit_predictor(config):
        return False
    predictor_dit = _config_value(config, "predictor_dit", None)
    return not bool(_config_value(predictor_dit, "masked_inpainting_enabled", False))


def build_value_future_gt_trajectory(
    states: torch.Tensor,
    config: Any,
    *,
    num_poses: Optional[int] = None,
    max_step_displacement: float = 20.0,
) -> torch.Tensor:
    """Build the shared training/validation ego-relative future trajectory."""

    if states.ndim != 3 or states.shape[-1] <= 5:
        raise ValueError(f"states must be [B, T, D>=6], got {tuple(states.shape)}")
    train_cfg = _config_value(config, "train", None)
    inference_consistent = bool(_config_value(train_cfg, "predictor_inference_consistent", False))
    num_observed = int(_config_value(train_cfg, "num_observed_frames", 1))
    future_start = num_observed if inference_consistent else 1
    if future_start < 1 or future_start >= states.shape[1]:
        raise ValueError(f"future_start={future_start} must leave at least one future state in T={states.shape[1]}")

    state_se2 = states[:, :, [0, 1, 5]].double()
    origin = state_se2[:, future_start - 1]
    delta = state_se2[:, future_start:] - origin[:, None]
    cos_heading = torch.cos(-origin[:, 2])
    sin_heading = torch.sin(-origin[:, 2])
    ego_x = cos_heading[:, None] * delta[..., 0] - sin_heading[:, None] * delta[..., 1]
    ego_y = sin_heading[:, None] * delta[..., 0] + cos_heading[:, None] * delta[..., 1]
    ego_yaw = torch.atan2(torch.sin(delta[..., 2]), torch.cos(delta[..., 2]))
    trajectory = torch.stack([ego_x, ego_y, ego_yaw], dim=-1).float()
    if num_poses is not None:
        num_poses = int(num_poses)
        if num_poses < 1:
            raise ValueError(f"num_poses must be >= 1, got {num_poses}")
        trajectory = trajectory[:, :num_poses]

    max_step_displacement = float(max_step_displacement)
    if not math.isfinite(max_step_displacement) or max_step_displacement <= 0.0:
        raise ValueError(f"max_step_displacement must be finite and positive, got {max_step_displacement}")
    with torch.no_grad():
        if trajectory.shape[1] > 1:
            step_displacement = torch.linalg.vector_norm(
                trajectory[:, 1:, :2] - trajectory[:, :-1, :2],
                dim=-1,
            )
            max_per_sample = step_displacement.max(dim=1).values
            anomaly_mask = max_per_sample > max_step_displacement
            if bool(anomaly_mask.any().item()):
                raise ValueError(
                    f"GT trajectory anomaly: {int(anomaly_mask.sum().item())}/{anomaly_mask.shape[0]} "
                    f"samples exceed {max_step_displacement:.1f} m/step "
                    f"(max={max_per_sample.max().item():.1f} m). Refusing to use anomalous GT for "
                    "value training or best-value selection."
                )
    return trajectory


def make_value_loss_config(config: Any) -> ValuePlanningLossConfig:
    """Extract a runtime loss config from ``TrainingConfig.value_planning``."""
    value_cfg = _value_cfg(config)
    return ValuePlanningLossConfig(
        gamma=float(_config_value(value_cfg, "gamma", 0.99)),
        lambda_return=float(_config_value(value_cfg, "lambda_return", 0.8)),
        bootstrap_horizon=int(_config_value(value_cfg, "bootstrap_horizon", 3)),
        progress_weight=float(_config_value(value_cfg, "progress_weight", 1.0)),
        comfort_weight=float(_config_value(value_cfg, "comfort_weight", 0.2)),
        value_loss_weight=float(_config_value(value_cfg, "value_loss_weight", 1.0)),
        td_loss_weight=float(_config_value(value_cfg, "td_loss_weight", 1.0)),
        safe_floor_weight=float(_config_value(value_cfg, "safe_floor_weight", 0.05)),
        episode_ranking_weight=float(_config_value(value_cfg, "episode_ranking_weight", 0.1)),
        episode_ranking_margin=float(_config_value(value_cfg, "episode_ranking_margin", 1.0)),
        srpo_shaping_weight=float(_config_value(value_cfg, "srpo_shaping_weight", 0.0)),
        srpo_potential_based=bool(_config_value(value_cfg, "srpo_potential_based", True)),
        pred_consistency_weight=float(_config_value(value_cfg, "pred_consistency_weight", 0.0)),
    )


def value_planning_enabled(config: Any) -> bool:
    """Whether a value head is enabled for value-planning/guidance."""
    return bool(_config_value(_value_cfg(config), "enabled", False))


def value_planning_method1_enabled(config: Any) -> bool:
    """Whether the old Variant A Method 1 candidate scorer should run."""
    value_cfg = _value_cfg(config)
    return (
        bool(_config_value(value_cfg, "enabled", False))
        and str(_config_value(value_cfg, "variant", "a_method1")) == "a_method1"
    )


def td_lambda_targets(
    rewards: torch.Tensor,
    target_values: torch.Tensor,
    *,
    gamma: float,
    lambda_return: float,
    rho: Optional[torch.Tensor] = None,
    srpo_shaping_weight: float = 0.0,
) -> torch.Tensor:
    """Compute non-terminal TD(lambda) targets bootstrapped by the EMA value head.

    The finite training window is a truncation, not an observed terminal.  In
    particular, counterfactual clip labels must never fabricate a terminal or
    overwrite the geometric progress/comfort reward.
    """
    if rewards.shape != target_values.shape:
        raise ValueError(f"rewards shape {tuple(rewards.shape)} != target_values shape {tuple(target_values.shape)}")
    gamma = float(gamma)
    lambda_return = float(lambda_return)
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lambda_return <= 1.0:
        raise ValueError(f"lambda_return must be in [0, 1], got {lambda_return}")

    shaped_rewards = rewards
    if rho is not None and float(srpo_shaping_weight) != 0.0:
        if rho.shape != rewards.shape:
            raise ValueError(f"rho shape {tuple(rho.shape)} != rewards shape {tuple(rewards.shape)}")
        rho_next = torch.cat([rho[:, 1:], torch.zeros_like(rho[:, -1:])], dim=1)
        shaping = gamma * rho_next - rho
        shaped_rewards = rewards + float(srpo_shaping_weight) * shaping

    targets = torch.zeros_like(target_values)
    next_return = target_values[:, -1].detach()
    for t in range(target_values.shape[1] - 1, -1, -1):
        next_value = (
            target_values[:, t].detach() if t == target_values.shape[1] - 1 else target_values[:, t + 1].detach()
        )
        bootstrap = (1.0 - lambda_return) * next_value + lambda_return * next_return
        next_return = shaped_rewards[:, t] + gamma * bootstrap
        targets[:, t] = next_return
    return targets


def compute_online_rewards(
    trajs: torch.Tensor,
    *,
    progress_weight: float,
    comfort_weight: float,
) -> torch.Tensor:
    """Compute simple online prefix rewards for candidate trajectories.

    Parameters
    ----------
    trajs : [B, K, P, 3] candidate trajectories in ego coordinates.
    """
    if trajs.ndim != 4 or trajs.shape[-1] != 3:
        raise ValueError(f"trajs must be [B, K, P, 3], got {tuple(trajs.shape)}")
    origin = torch.zeros(*trajs.shape[:2], 1, 3, device=trajs.device, dtype=trajs.dtype)
    full = torch.cat([origin, trajs], dim=2)
    step_delta = full[:, :, 1:, :] - full[:, :, :-1, :]
    progress = step_delta[..., 0]
    lateral = step_delta[..., 1].abs()
    yaw_delta = torch.atan2(torch.sin(step_delta[..., 2]), torch.cos(step_delta[..., 2])).abs()
    comfort_cost = lateral + yaw_delta
    rewards = float(progress_weight) * progress - float(comfort_weight) * comfort_cost
    return rewards


def _aggregate_rewards_for_frame_stride(
    raw_rewards: torch.Tensor,
    *,
    frame_stride: int,
    gamma: float,
    horizon: int,
) -> torch.Tensor:
    if raw_rewards.ndim != 2:
        raise ValueError(f"raw_rewards must be [B, T], got {tuple(raw_rewards.shape)}")
    frame_stride = int(frame_stride)
    horizon = int(horizon)
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    required_steps = horizon * frame_stride
    if raw_rewards.shape[1] < required_steps:
        raise ValueError(
            f"raw_rewards length ({raw_rewards.shape[1]}) is shorter than horizon*frame_stride ({required_steps})"
        )
    if frame_stride == 1:
        return raw_rewards[:, :horizon]

    discounts = torch.pow(
        raw_rewards.new_full((frame_stride,), float(gamma)),
        torch.arange(frame_stride, device=raw_rewards.device, dtype=raw_rewards.dtype),
    )
    chunks = raw_rewards[:, :required_steps].reshape(raw_rewards.shape[0], horizon, frame_stride)
    return (chunks * discounts.view(1, 1, frame_stride)).sum(dim=2)


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


@torch.no_grad()
def copy_value_head_to_target_(online_value_head: torch.nn.Module, target_value_head: torch.nn.Module) -> None:
    """Exactly copy a synchronized online value head into a frozen target head."""

    online = _unwrap_module(online_value_head)
    target = _unwrap_module(target_value_head)
    target.load_state_dict(online.state_dict(), strict=True)
    target.requires_grad_(False)
    target.eval()


@torch.no_grad()
def ema_update_value_head_(
    online_value_head: torch.nn.Module,
    target_value_head: torch.nn.Module,
    *,
    tau: float,
) -> None:
    """Apply ``target = tau * target + (1 - tau) * online`` to the full head state."""

    tau = float(tau)
    if not math.isfinite(tau) or not 0.0 <= tau < 1.0:
        raise ValueError(f"value target tau must be finite and in [0, 1), got {tau}")
    online = _unwrap_module(online_value_head)
    target = _unwrap_module(target_value_head)
    online_parameters = dict(online.named_parameters())
    target_parameters = dict(target.named_parameters())
    if online_parameters.keys() != target_parameters.keys():
        raise ValueError("online and target value heads have different parameter schemas")
    for name, target_parameter in target_parameters.items():
        online_parameter = online_parameters[name].detach()
        target_parameter.mul_(tau).add_(online_parameter, alpha=1.0 - tau)

    online_buffers = dict(online.named_buffers())
    target_buffers = dict(target.named_buffers())
    if online_buffers.keys() != target_buffers.keys():
        raise ValueError("online and target value heads have different buffer schemas")
    for name, target_buffer in target_buffers.items():
        online_buffer = online_buffers[name].detach()
        if target_buffer.is_floating_point() or target_buffer.is_complex():
            target_buffer.mul_(tau).add_(online_buffer, alpha=1.0 - tau)
        else:
            target_buffer.copy_(online_buffer)
    target.requires_grad_(False)
    target.eval()


def _portable_value_state(state: Any, *, key: str, source: str) -> Dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint {source!r} has invalid or empty {key!r} state")
    return {(name[7:] if name.startswith("module.") else name): value for name, value in state.items()}


def restore_value_heads_from_checkpoint(
    *,
    online_value_head: torch.nn.Module,
    target_value_head: Optional[torch.nn.Module],
    checkpoint: Dict[str, Any],
    source: str,
    trainable: bool,
    model_only: bool,
) -> None:
    """Restore online/target heads under strict resume and warm-start semantics."""

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"value checkpoint {source!r} is not a mapping")
    if "value_head" not in checkpoint:
        raise RuntimeError(f"value checkpoint {source!r} is missing 'value_head'")
    online = _unwrap_module(online_value_head)
    online.load_state_dict(
        _portable_value_state(checkpoint["value_head"], key="value_head", source=source),
        strict=True,
    )

    if not trainable:
        if target_value_head is not None:
            raise ValueError("frozen Stage3 value restore must not construct target_value_head")
        online.requires_grad_(False)
        online.eval()
        return
    if target_value_head is None:
        raise ValueError("trainable value restore requires target_value_head")
    if model_only:
        copy_value_head_to_target_(online, target_value_head)
        return
    if "target_value_head" not in checkpoint:
        raise RuntimeError(
            f"exact trainable value resume from {source!r} requires 'target_value_head'; "
            "use model-only warm start only when intentionally starting a new optimizer history"
        )
    target = _unwrap_module(target_value_head)
    target.load_state_dict(
        _portable_value_state(checkpoint["target_value_head"], key="target_value_head", source=source),
        strict=True,
    )
    target.requires_grad_(False)
    target.eval()


def restore_value_lifecycle(
    *,
    online_value_head: torch.nn.Module,
    target_value_head: Optional[torch.nn.Module],
    trainable: bool,
    guidance_active: bool,
    resume_checkpoint: Optional[Dict[str, Any]],
    resume_source: Optional[str],
    resume_model_only: bool,
    value_checkpoint: Optional[Dict[str, Any]],
    value_source: Optional[str],
    best_value_tracker: Optional[Any],
    fresh_anneal: bool = False,
    anneal_checkpoint: Optional[Dict[str, Any]] = None,
    anneal_source: Optional[str] = None,
) -> Optional[str]:
    """Apply value restore semantics after all outer checkpoint branches converge.

    Normal-flow exact resume has precedence over an explicit value warm-start.
    Fresh anneal rejects resume state, gives an explicit value checkpoint
    precedence over the anneal payload, and requires one of those sources for
    both trainable value and frozen guidance modes.
    """

    if (resume_checkpoint is None) != (resume_source is None):
        raise ValueError("resume_checkpoint and resume_source must be provided together")
    if (value_checkpoint is None) != (value_source is None):
        raise ValueError("value_checkpoint and value_source must be provided together")
    if (anneal_checkpoint is None) != (anneal_source is None):
        raise ValueError("anneal_checkpoint and anneal_source must be provided together")
    if not isinstance(fresh_anneal, bool):
        raise TypeError("fresh_anneal must be bool")
    if fresh_anneal and resume_checkpoint is not None:
        raise ValueError("fresh anneal must not restore a resume value checkpoint")

    loaded_source: Optional[str] = None
    if resume_checkpoint is not None:
        restore_value_heads_from_checkpoint(
            online_value_head=online_value_head,
            target_value_head=target_value_head,
            checkpoint=resume_checkpoint,
            source=str(resume_source),
            trainable=trainable,
            model_only=resume_model_only,
        )
        loaded_source = str(resume_source)
        if trainable and not resume_model_only:
            tracker_state = resume_checkpoint.get("best_value_tracker_state")
            if tracker_state is None:
                raise RuntimeError(
                    f"exact trainable value resume requires 'best_value_tracker_state': {resume_source}"
                )
            if best_value_tracker is None:
                raise ValueError("exact trainable value resume requires best_value_tracker")
            best_value_tracker.load_state_dict(tracker_state)
    elif value_checkpoint is not None:
        restore_value_heads_from_checkpoint(
            online_value_head=online_value_head,
            target_value_head=target_value_head,
            checkpoint=value_checkpoint,
            source=str(value_source),
            trainable=trainable,
            model_only=True,
        )
        loaded_source = str(value_source)
    elif fresh_anneal and anneal_checkpoint is not None:
        restore_value_heads_from_checkpoint(
            online_value_head=online_value_head,
            target_value_head=target_value_head,
            checkpoint=anneal_checkpoint,
            source=str(anneal_source),
            trainable=trainable,
            model_only=True,
        )
        loaded_source = str(anneal_source)

    if fresh_anneal and loaded_source is None:
        raise RuntimeError(
            "fresh anneal value planning requires value_head in explicit meta.value_checkpoint "
            "or optimization.anneal_ckpt"
        )
    if guidance_active and loaded_source is None:
        raise RuntimeError(
            "value_guidance.enabled=true requires loading value_head from resume or meta.value_checkpoint"
        )
    return loaded_source


def build_value_checkpoint_state(
    *,
    online_value_head: torch.nn.Module,
    target_value_head: Optional[torch.nn.Module],
    trainable: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Return the strict Stage2/Stage3 value-head checkpoint payload."""

    payload = {"value_head": _unwrap_module(online_value_head).state_dict()}
    if trainable:
        if target_value_head is None:
            raise ValueError("trainable Stage2 checkpoint requires target_value_head")
        payload["target_value_head"] = _unwrap_module(target_value_head).state_dict()
    elif target_value_head is not None:
        raise ValueError("frozen Stage3 checkpoint must not include target_value_head")
    return payload


def value_optimizer_step_succeeded(
    *,
    mixed_precision: bool,
    scale_before: Optional[float],
    scale_after: Optional[float],
) -> bool:
    """Return whether GradScaler executed the optimizer step (rather than overflow-skipping it)."""

    if not mixed_precision:
        return True
    if scale_before is None or scale_after is None:
        raise ValueError("AMP optimizer success detection requires scale_before and scale_after")
    scale_before = float(scale_before)
    scale_after = float(scale_after)
    if not math.isfinite(scale_before) or not math.isfinite(scale_after) or scale_before <= 0.0 or scale_after <= 0.0:
        raise ValueError(f"AMP scales must be finite and positive, got {scale_before} -> {scale_after}")
    return scale_after >= scale_before


def optimizer_gradients_finite_flag(
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    scale: Optional[float],
) -> torch.Tensor:
    """Aggregate whether scaled gradients can be unscaled without overflow."""

    if scale is None:
        normalized_scale = 1.0
    else:
        if isinstance(scale, bool) or not isinstance(scale, Real):
            raise TypeError(f"scale must be a real number or None, got {scale!r}")
        normalized_scale = float(scale)
    if not math.isfinite(normalized_scale) or normalized_scale <= 0.0:
        raise ValueError(f"scale must be finite and positive, got {normalized_scale}")
    try:
        float32_scale = struct.unpack("!f", struct.pack("!f", normalized_scale))[0]
    except OverflowError as exc:
        raise ValueError(f"scale must be finite and positive as float32, got {normalized_scale}") from exc
    if not math.isfinite(float32_scale) or float32_scale <= 0.0:
        raise ValueError(f"scale must be finite and positive as float32, got {float32_scale}")

    inverse_scale_float32 = None
    if float32_scale < 1.0:
        try:
            inverse_scale_float32 = struct.unpack("!f", struct.pack("!f", 1.0 / float32_scale))[0]
        except OverflowError as exc:
            raise ValueError(f"scale reciprocal must be finite as float32, got scale={float32_scale}") from exc
        if not math.isfinite(inverse_scale_float32):
            raise ValueError(f"scale reciprocal must be finite as float32, got scale={float32_scale}")

    finite_flag = torch.ones((), dtype=torch.bool, device=device)
    safe_thresholds: Dict[torch.dtype, torch.Tensor] = {}
    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group.get("params", []):
            if parameter is None or parameter.grad is None:
                continue
            gradient = parameter.grad
            gradient_values = gradient._values() if gradient.is_sparse else gradient
            if gradient_values.device != device:
                raise ValueError(
                    "optimizer gradient device must match the consensus device, "
                    f"got {gradient_values.device} and {device}"
                )
            values_allowed = torch.isfinite(gradient_values)
            if inverse_scale_float32 is not None and gradient_values.is_floating_point():
                safe_threshold = safe_thresholds.get(gradient_values.dtype)
                if safe_threshold is None:
                    safe_bound_float64 = torch.finfo(gradient_values.dtype).max / inverse_scale_float32
                    rounded_threshold = torch.tensor(
                        safe_bound_float64,
                        dtype=gradient_values.dtype,
                        device=device,
                    )
                    rounded_above_bound = rounded_threshold.to(torch.float64) > safe_bound_float64
                    threshold_toward_zero = torch.nextafter(
                        rounded_threshold,
                        torch.zeros((), dtype=gradient_values.dtype, device=device),
                    )
                    safe_threshold = torch.where(
                        rounded_above_bound,
                        threshold_toward_zero,
                        rounded_threshold,
                    )
                    safe_thresholds[gradient_values.dtype] = safe_threshold
                # The threshold is exclusive so a legacy max*scale boundary is
                # rejected even when it equals the largest conservative cast.
                values_allowed.logical_and_(gradient_values.abs() < safe_threshold)
            finite_flag.logical_and_(values_allowed.all())
    return finite_flag


def distributed_optimizer_step_allowed(local_step_allowed: torch.Tensor) -> bool:
    """Require every rank to approve a step with one collective and host sync."""

    if not torch.is_tensor(local_step_allowed):
        raise TypeError("local_step_allowed must be a torch.Tensor")
    if local_step_allowed.dtype != torch.bool or local_step_allowed.ndim != 0:
        raise ValueError("local_step_allowed must be a scalar bool tensor")
    step_flag = local_step_allowed.to(dtype=torch.int32)
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(step_flag, op=dist.ReduceOp.MIN)
    return bool(step_flag.item())


def skip_amp_optimizer_step_(
    scaler: Any,
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Use GradScaler's public overflow path to skip a globally rejected step."""

    scale_before = float(scaler.get_scale())
    materialized_gradient = None
    for parameter_group in optimizer.param_groups:
        for parameter in parameter_group.get("params", []):
            if parameter is None or parameter.grad is None:
                continue
            gradient = parameter.grad
            gradient_values = gradient._values() if gradient.is_sparse else gradient
            if gradient_values.numel() > 0:
                materialized_gradient = gradient_values
                break
        if materialized_gradient is not None:
            break
    if materialized_gradient is None:
        raise RuntimeError("cannot skip AMP optimizer step without a materialized gradient")

    with torch.no_grad():
        materialized_gradient.fill_(float("inf"))
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scale_after = float(scaler.get_scale())
    if not scale_after < scale_before:
        raise RuntimeError(
            "GradScaler did not back off after the forced overflow skip, " f"got scale {scale_before} -> {scale_after}"
        )
    return scale_before, scale_after


def maybe_update_value_target_(
    online_value_head: torch.nn.Module,
    target_value_head: torch.nn.Module,
    *,
    tau: float,
    optimizer_step_successful: bool,
) -> bool:
    """Update the EMA target only when the corresponding optimizer step executed."""

    if not isinstance(optimizer_step_successful, bool):
        raise TypeError("optimizer_step_successful must be bool")
    if not optimizer_step_successful:
        return False
    ema_update_value_head_(online_value_head, target_value_head, tau=tau)
    return True


def _gather_global_vector(vector: torch.Tensor) -> torch.Tensor:
    """Differentiably gather a 1-D vector, supporting uneven local batch sizes."""

    if vector.ndim != 1:
        raise ValueError(f"global vector must be 1-D, got {tuple(vector.shape)}")
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return vector

    world_size = int(dist.get_world_size())
    local_size = torch.tensor([vector.numel()], dtype=torch.long, device=vector.device)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)
    if max_size == 0:
        return vector.new_empty((0,))
    if vector.numel() < max_size:
        vector = F.pad(vector, (0, max_size - vector.numel()))
    from torch.distributed.nn.functional import all_gather

    gathered = all_gather(vector)
    return torch.cat([part[:size] for part, size in zip(gathered, sizes)], dim=0)


def _gather_global_mask(mask: torch.Tensor) -> torch.Tensor:
    """Gather a 1-D boolean mask with the same uneven-batch protocol as scores."""

    if mask.dtype != torch.bool or mask.ndim != 1:
        raise ValueError(f"global mask must be 1-D bool, got dtype={mask.dtype}, shape={tuple(mask.shape)}")
    dist = torch.distributed
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return mask
    world_size = int(dist.get_world_size())
    local_size = torch.tensor([mask.numel()], dtype=torch.long, device=mask.device)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(gathered_sizes, local_size)
    sizes = [int(size.item()) for size in gathered_sizes]
    max_size = max(sizes)
    padded = F.pad(mask, (0, max_size - mask.numel()), value=False)
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([part[:size] for part, size in zip(gathered, sizes)], dim=0)


def compute_episode_ranking_loss(
    episode_scores: torch.Tensor,
    hazard_mask: torch.Tensor,
    comparator_mask: torch.Tensor,
    *,
    margin: float,
) -> Dict[str, torch.Tensor]:
    """Compute a full-global-batch hazard-vs-comparator pairwise hinge loss.

    A batch without either category returns a graph-connected zero.  Validation
    applies the stricter full-dataset category requirement separately.
    """

    if episode_scores.ndim != 1:
        raise ValueError(f"episode_scores must be [B], got {tuple(episode_scores.shape)}")
    if hazard_mask.shape != episode_scores.shape or comparator_mask.shape != episode_scores.shape:
        raise ValueError("hazard/comparator masks must match episode_scores")
    if hazard_mask.dtype != torch.bool or comparator_mask.dtype != torch.bool:
        raise ValueError("hazard/comparator masks must be bool")
    if bool((hazard_mask & comparator_mask).any().item()):
        raise ValueError("hazard and comparator masks must be disjoint")
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"episode ranking margin must be finite and nonnegative, got {margin}")

    global_scores = _gather_global_vector(episode_scores)
    global_hazard = _gather_global_mask(hazard_mask.to(device=episode_scores.device))
    global_comparator = _gather_global_mask(comparator_mask.to(device=episode_scores.device))
    hazard_scores = global_scores[global_hazard]
    comparator_scores = global_scores[global_comparator]
    graph_zero = global_scores.sum() * 0.0
    if hazard_scores.numel() == 0 or comparator_scores.numel() == 0:
        loss = graph_zero
        ranking_accuracy = graph_zero.detach()
        margin_accuracy = graph_zero.detach()
    else:
        pairwise_gap = comparator_scores[None, :] - hazard_scores[:, None]
        loss = F.relu(margin - pairwise_gap).mean()
        ranking_accuracy = (pairwise_gap > 0.0).float().mean()
        margin_accuracy = (pairwise_gap >= margin).float().mean()
    return {
        "loss": loss,
        "ranking_accuracy": ranking_accuracy,
        "margin_accuracy": margin_accuracy,
        "hazard_count": episode_scores.new_tensor(float(hazard_scores.numel())),
        "comparator_count": episode_scores.new_tensor(float(comparator_scores.numel())),
    }


def _distributed_optional_masked_mean(
    per_sample_loss: torch.Tensor,
    mask: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """DDP-correct masked mean that returns graph-zero for a globally empty category."""

    if per_sample_loss.ndim != 1 or mask.shape != per_sample_loss.shape or mask.dtype != torch.bool:
        raise ValueError(f"{name} expects matching [B] loss and bool mask")
    local_count = mask.sum().to(dtype=per_sample_loss.dtype)
    global_count = local_count.detach().clone()
    world_size = 1
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        world_size = int(dist.get_world_size())
        if world_size > 1:
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    local_sum = (per_sample_loss * mask.to(dtype=per_sample_loss.dtype)).sum()
    if float(global_count.item()) == 0.0:
        return local_sum * 0.0
    return local_sum * (float(world_size) / global_count)


def compute_value_planning_loss(
    *,
    predicted_values: torch.Tensor,
    target_values: torch.Tensor,
    rewards: torch.Tensor,
    eligible_mask: torch.Tensor,
    hazard_mask: torch.Tensor,
    comparator_mask: torch.Tensor,
    config: ValuePlanningLossConfig,
    rho: Optional[torch.Tensor] = None,
    pred_consistency_loss: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute TD + conservative value losses for Variant A Method 1."""
    if predicted_values.ndim != 2:
        raise ValueError(f"predicted_values must be [B, F], got {tuple(predicted_values.shape)}")
    if rewards.shape != predicted_values.shape:
        raise ValueError(f"rewards shape {tuple(rewards.shape)} != predicted_values {tuple(predicted_values.shape)}")
    if target_values.shape != predicted_values.shape:
        raise ValueError(
            f"target_values shape {tuple(target_values.shape)} != predicted_values {tuple(predicted_values.shape)}"
        )
    for name, mask in (
        ("eligible_mask", eligible_mask),
        ("hazard_mask", hazard_mask),
        ("comparator_mask", comparator_mask),
    ):
        if mask.dtype != torch.bool or mask.shape != predicted_values.shape[:1]:
            raise ValueError(f"{name} must be bool [B], got dtype={mask.dtype}, shape={tuple(mask.shape)}")
    if bool((hazard_mask & comparator_mask).any().item()):
        raise ValueError("hazard and comparator masks must be disjoint")
    if bool(((hazard_mask | comparator_mask) & ~eligible_mask).any().item()):
        raise ValueError("hazard/comparator masks must be subsets of eligible_mask")
    if float(config.srpo_shaping_weight) != 0.0 and rho is None:
        raise ValueError("srpo_shaping_weight > 0 requires rho; refusing to silently disable SRPO shaping")
    if float(config.pred_consistency_weight) != 0.0 and pred_consistency_loss is None:
        raise ValueError(
            "pred_consistency_weight > 0 requires pred_consistency_loss; "
            "refusing to silently disable predictor consistency"
        )

    targets = td_lambda_targets(
        rewards,
        target_values,
        gamma=float(config.gamma),
        lambda_return=float(config.lambda_return),
        rho=rho,
        srpo_shaping_weight=float(config.srpo_shaping_weight),
    ).detach()
    td_per_sample = F.smooth_l1_loss(predicted_values, targets, reduction="none").mean(dim=1)
    td_loss = distributed_masked_mean(td_per_sample, eligible_mask, name="value TD")

    safe_floor_per_sample = F.relu(-predicted_values).mean(dim=1)
    safe_floor_loss = _distributed_optional_masked_mean(
        safe_floor_per_sample,
        comparator_mask,
        name="value safe floor",
    )

    ranking = compute_episode_ranking_loss(
        predicted_values.mean(dim=1),
        hazard_mask,
        comparator_mask,
        margin=float(config.episode_ranking_margin),
    )
    episode_ranking_loss = ranking["loss"]

    pred_loss = predicted_values.new_zeros(()) if pred_consistency_loss is None else pred_consistency_loss
    loss = (
        float(config.td_loss_weight) * td_loss
        + float(config.safe_floor_weight) * safe_floor_loss
        + float(config.episode_ranking_weight) * episode_ranking_loss
        + float(config.pred_consistency_weight) * pred_loss
    )
    return {
        "loss": float(config.value_loss_weight) * loss,
        "td_loss": td_loss,
        "safe_floor_loss": safe_floor_loss,
        "episode_ranking_loss": episode_ranking_loss,
        "ranking_accuracy": ranking["ranking_accuracy"],
        "margin_accuracy": ranking["margin_accuracy"],
        "hazard_count": ranking["hazard_count"],
        "comparator_count": ranking["comparator_count"],
        "pred_consistency_loss": pred_loss,
        "targets": targets,
    }


def compute_value_head_loss_from_batch(
    *,
    value_head: torch.nn.Module,
    target_value_head: torch.nn.Module,
    z_future: torch.Tensor,
    gt_trajectory: torch.Tensor,
    sample_masks: Optional[CounterfactualSampleMasks],
    tokens_per_frame: int,
    config: Any,
    frame_stride: int = 1,
) -> Dict[str, torch.Tensor]:
    """Train ``V(z_t)`` on imagined predictor future tokens from the current batch.

    ``z_future`` must be predictor-generated future tokens (normally ``z_ar``),
    not frozen target-encoder ``h_target``.
    """
    if gt_trajectory.ndim != 3 or gt_trajectory.shape[-1] != 3:
        raise ValueError(f"gt_trajectory must be [B, P, 3], got {tuple(gt_trajectory.shape)}")
    loss_config = make_value_loss_config(config)
    values_all = value_head(z_future, tokens_per_frame=int(tokens_per_frame))
    if values_all.ndim != 2:
        raise ValueError(f"value_head must return [B, F], got {tuple(values_all.shape)}")
    frame_stride = int(frame_stride)
    if frame_stride < 1:
        raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
    # horizon 是 value loss 真正监督的 predictor-step 数，不是 raw pose 数。
    # 例如 raw future 有 8 帧、frame_stride=2 时，最多只有 4 个 predictor value step。
    max_gt_value_steps = int(gt_trajectory.shape[1]) // frame_stride
    horizon = min(int(loss_config.bootstrap_horizon), int(values_all.shape[1]), max_gt_value_steps)
    if horizon < 1:
        raise ValueError(
            "value_planning needs at least one future value/pose; "
            f"values={tuple(values_all.shape)}, gt_trajectory={tuple(gt_trajectory.shape)}, "
            f"frame_stride={frame_stride}"
        )
    values = values_all[:, :horizon]
    with torch.no_grad():
        target_values_all = target_value_head(z_future, tokens_per_frame=int(tokens_per_frame))
    if target_values_all.shape != values_all.shape:
        raise ValueError(
            "target_value_head output shape does not match online value_head: "
            f"{tuple(target_values_all.shape)} != {tuple(values_all.shape)}"
        )
    target_values = target_values_all[:, :horizon].detach()
    raw_reward_traj = gt_trajectory[:, None, : horizon * frame_stride]
    raw_rewards = compute_online_rewards(
        raw_reward_traj,
        progress_weight=float(loss_config.progress_weight),
        comfort_weight=float(loss_config.comfort_weight),
    ).squeeze(1)
    rewards = _aggregate_rewards_for_frame_stride(
        raw_rewards,
        frame_stride=frame_stride,
        gamma=float(loss_config.gamma),
        horizon=horizon,
    )
    td_loss_config = replace(loss_config, gamma=float(loss_config.gamma) ** frame_stride)
    if rewards.shape != values.shape:
        raise ValueError(
            f"aggregated rewards shape {tuple(rewards.shape)} does not match value horizon {tuple(values.shape)}"
        )
    if sample_masks is None:
        eligible_mask = torch.ones(values.shape[0], dtype=torch.bool, device=values.device)
        hazard_mask = torch.zeros_like(eligible_mask)
        comparator_mask = eligible_mask
    else:
        eligible_mask = sample_masks.value.to(device=values.device)
        hazard_mask = sample_masks.cf_ego_hazard.to(device=values.device) & eligible_mask
        comparator_mask = eligible_mask & ~hazard_mask
    result = compute_value_planning_loss(
        predicted_values=values,
        target_values=target_values,
        rewards=rewards,
        eligible_mask=eligible_mask,
        hazard_mask=hazard_mask,
        comparator_mask=comparator_mask,
        config=td_loss_config,
    )
    result["predicted_values"] = values
    result["target_values"] = target_values
    result["rewards"] = rewards
    result["eligible_mask"] = eligible_mask
    result["hazard_mask"] = hazard_mask
    result["comparator_mask"] = comparator_mask
    return result


def _select_by_confidence(trajs: torch.Tensor, confidences: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
    if confidences is None:
        raise ValueError("confidences are required when value_planning.enabled=false")
    if confidences.ndim != 2 or confidences.shape != trajs.shape[:2]:
        raise ValueError(f"confidences must be [B, K]={tuple(trajs.shape[:2])}, got {tuple(confidences.shape)}")
    idx = confidences.argmax(dim=1)
    gather_idx = idx.view(-1, 1, 1, 1).expand(-1, 1, trajs.shape[2], trajs.shape[3])
    selected = trajs.gather(1, gather_idx).squeeze(1)
    return {
        "value_scores": confidences,
        "value_selected_idx": idx,
        "value_selected_trajectory": selected,
    }


def score_trajectories_method1(
    *,
    predictor: Optional[torch.nn.Module],
    value_head: Optional[torch.nn.Module],
    z_context: torch.Tensor,
    trajs: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
    config: Any,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    dt: float,
    predictor_observed_steps: Optional[int],
    predictor_frame_stride: int,
    confidences: Optional[torch.Tensor] = None,
    rollout_fn: Optional[RolloutFn] = None,
    validation_sample_ids: Optional[Sequence[str]] = None,
    validation_base_seed: Optional[int] = None,
    validation_protocol: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Score full trajectories by rolling their prefix and bootstrapping ``V(z_m)``.

    The returned selected trajectory is the original full candidate, not the
    rolled prefix.
    """
    if trajs.ndim != 4 or trajs.shape[-1] != 3:
        raise ValueError(f"trajs must be [B, K, P, 3], got {tuple(trajs.shape)}")
    value_cfg = _value_cfg(config)
    if not bool(_config_value(value_cfg, "enabled", True)):
        return _select_by_confidence(trajs, confidences)
    if value_head is None:
        raise ValueError("value_head is required when value_planning.enabled=true")

    prefix_steps = int(_config_value(value_cfg, "prefix_steps", 2))
    if prefix_steps < 1 or prefix_steps > trajs.shape[2]:
        raise ValueError(f"prefix_steps must be in [1, num_poses={trajs.shape[2]}], got {prefix_steps}")
    gamma = float(_config_value(value_cfg, "gamma", 0.99))
    progress_weight = float(_config_value(value_cfg, "progress_weight", 1.0))
    comfort_weight = float(_config_value(value_cfg, "comfort_weight", 0.2))
    prefix_trajs = trajs[:, :, :prefix_steps]
    rewards = compute_online_rewards(
        prefix_trajs,
        progress_weight=progress_weight,
        comfort_weight=comfort_weight,
    )

    if rollout_fn is None:
        from app.vjepa_cowa_world_model.training.runtimes.refinement_runtime import rollout_predictor_modes

        rollout_fn = rollout_predictor_modes
    fixed_future_predictor = _uses_fixed_future_predictor(config)
    rollout_trajs = trajs if fixed_future_predictor else prefix_trajs
    rollout_seconds = None if fixed_future_predictor else float(prefix_steps) * float(dt)
    rollout_kwargs = dict(
        predictor=predictor,
        z_context=z_context,
        future_trajs=rollout_trajs,
        actions=actions,
        states=states,
        driving_command=driving_command,
        ego_dynamics=ego_dynamics,
        config=config,
        tokens_per_frame=int(tokens_per_frame),
        runtime_normalize_reps=bool(runtime_normalize_reps),
        dt=float(dt),
        predictor_observed_steps=predictor_observed_steps,
        predictor_frame_stride=int(predictor_frame_stride),
        predictor_rollout_seconds=rollout_seconds,
    )
    validation_values = (validation_sample_ids, validation_base_seed, validation_protocol)
    if any(value is not None for value in validation_values) and not all(
        value is not None for value in validation_values
    ):
        raise ValueError(
            "validation_sample_ids, validation_base_seed, and validation_protocol must be provided together"
        )
    if validation_sample_ids is not None:
        rollout_kwargs.update(
            validation_sample_ids=validation_sample_ids,
            validation_base_seed=int(validation_base_seed),
            validation_protocol=str(validation_protocol),
            validation_stream="value_candidate_predictor_initial_noise",
        )
    z_prefix = rollout_fn(**rollout_kwargs)
    if z_prefix.ndim != 4 or z_prefix.shape[:2] != trajs.shape[:2]:
        raise ValueError(
            f"rollout_fn must return [B, K, N, D] aligned with trajs, got {tuple(z_prefix.shape)} "
            f"for trajs={tuple(trajs.shape)}"
        )

    batch_size, num_modes = trajs.shape[:2]
    z_flat = z_prefix.reshape(batch_size * num_modes, z_prefix.shape[2], z_prefix.shape[3])
    values = value_head(z_flat, tokens_per_frame=int(tokens_per_frame)).reshape(batch_size, num_modes, -1)
    if values.shape[2] < 1:
        raise ValueError("value_head returned no per-step values")
    if fixed_future_predictor or _uses_latent_dit_predictor(config):
        if values.shape[2] < prefix_steps:
            raise ValueError(
                "value_head returned fewer steps than value_planning.prefix_steps for latent-DiT predictor: "
                f"values_steps={values.shape[2]}, prefix_steps={prefix_steps}"
            )
        bootstrap = values[:, :, prefix_steps - 1]
    else:
        bootstrap = values[:, :, -1]
    discounts = torch.pow(
        rewards.new_full((prefix_steps,), gamma),
        torch.arange(prefix_steps, device=rewards.device, dtype=rewards.dtype),
    )
    prefix_score = (rewards * discounts.view(1, 1, -1)).sum(dim=2)
    scores = prefix_score + (gamma**prefix_steps) * bootstrap
    selected_idx = scores.argmax(dim=1)
    gather_idx = selected_idx.view(-1, 1, 1, 1).expand(-1, 1, trajs.shape[2], trajs.shape[3])
    selected = trajs.gather(1, gather_idx).squeeze(1)
    return {
        "value_scores": scores,
        "value_selected_idx": selected_idx,
        "value_selected_trajectory": selected,
    }
