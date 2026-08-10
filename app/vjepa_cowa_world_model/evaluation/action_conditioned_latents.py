"""Build hand-authored future action scenarios from a fixed observation window."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

SCENARIO_NAMES = ("factual", "brake", "left_turn", "right_turn", "accelerate")


class InsufficientActionAnchorError(ValueError):
    """The observed-prefix forward action is too small for manual futures."""


@dataclass(frozen=True)
class ActionScenarioConfig:
    """Parameters controlling manual future action scenarios."""

    min_anchor_forward_m: float = 0.25
    anchor_forward_override_m: Optional[float] = None
    brake_final_scale: float = 0.0
    accelerate_final_scale: float = 1.5
    turn_yaw_radians_per_step: float = 0.12


@dataclass(frozen=True)
class _CanonicalConfig:
    """Validated built-in Python values, without modifying the caller's config."""

    minimum: float
    override: Optional[float]
    brake: float
    accelerate: float
    turn: float


def _validate_actions(actions: Tensor) -> None:
    """Validate an action tensor with shape ``[B, T, 3]``."""
    if not isinstance(actions, Tensor):
        raise TypeError("actions must be a torch Tensor")
    if actions.dtype not in (torch.float32, torch.float64):
        raise TypeError("actions dtype must be torch.float32 or torch.float64")
    if actions.ndim != 3 or actions.shape[-1] != 3:
        raise ValueError("actions must have shape [B, T, 3]")
    if actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError("actions must have non-empty B and T dimensions")
    if not torch.isfinite(actions).all().item():
        raise ValueError("actions must contain only finite values")


def _validate_real(name: str, value: object) -> float:
    """Return a finite canonical built-in numeric configuration value."""
    if type(value) not in (int, float):
        raise TypeError(f"{name} must be a built-in int or float")
    try:
        value_float = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be representable as a finite float") from error
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be finite")
    return value_float


def _validate_config(config: ActionScenarioConfig) -> _CanonicalConfig:
    """Validate configuration and return canonical Python values."""
    if not isinstance(config, ActionScenarioConfig):
        raise TypeError("config must be an ActionScenarioConfig")
    minimum = _validate_real("min_anchor_forward_m", config.min_anchor_forward_m)
    brake = _validate_real("brake_final_scale", config.brake_final_scale)
    accelerate = _validate_real("accelerate_final_scale", config.accelerate_final_scale)
    turn = _validate_real("turn_yaw_radians_per_step", config.turn_yaw_radians_per_step)
    if minimum < 0:
        raise ValueError("min_anchor_forward_m must be >= 0")
    if not 0 <= brake < 1:
        raise ValueError("brake_final_scale must be in [0, 1)")
    if accelerate <= 1:
        raise ValueError("accelerate_final_scale must be > 1")
    if not 0 < turn < math.pi:
        raise ValueError("turn_yaw_radians_per_step must be in (0, pi)")
    override = None
    if config.anchor_forward_override_m is not None:
        override = _validate_real("anchor_forward_override_m", config.anchor_forward_override_m)
        if override <= 0:
            raise ValueError("anchor_forward_override_m must be > 0")
    return _CanonicalConfig(minimum, override, brake, accelerate, turn)


def _cast_config_value(name: str, value: float, actions: Tensor) -> Tensor:
    """Cast one canonical value to action dtype with an actionable validation error."""
    cast_value = actions.new_tensor(value)
    displayed = cast_value.item()
    if not torch.isfinite(cast_value).item():
        raise ValueError(f"{name} is not finite after dtype={actions.dtype} cast value={displayed}")
    return cast_value


def _validate_cast_range(name: str, value: Tensor, predicate: bool) -> None:
    """Fail before construction when dtype conversion invalidates a numeric contract."""
    if not predicate:
        raise ValueError(f"{name} is invalid after dtype={value.dtype} cast value={value.item()}")


def _validate_finite(name: str, value: Tensor) -> None:
    """Validate tensor output finiteness with an actionable field name."""
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _validate_observation_steps(num_observed_steps: int, total_steps: int) -> int:
    """Return the future boundary for a valid fixed observation window."""
    if isinstance(num_observed_steps, bool) or type(num_observed_steps) is not int:
        raise TypeError("num_observed_steps must be an int")
    if num_observed_steps < 2:
        raise ValueError("num_observed_steps must be >= 2")
    boundary = num_observed_steps - 1
    if boundary >= total_steps:
        raise ValueError("num_observed_steps must leave at least one future transition")
    return boundary


def build_action_scenarios(
    actions: Tensor, num_observed_steps: int, config: ActionScenarioConfig
) -> Dict[str, Tensor]:
    """Build full action sequences for factual and manually specified futures.

    Parameters
    ----------
    actions : [B, T, 3]
        Ego-frame action increments ``(forward, lateral, yaw)`` with dtype
        ``torch.float32`` or ``torch.float64``.
    num_observed_steps : int
        Fixed observation length; index ``num_observed_steps - 1`` starts the future.
    config : ActionScenarioConfig
        Manual-scenario parameters.

    Returns
    -------
    Dict[str, Tensor]
        Ordered factual, brake, left-turn, right-turn, and accelerate sequences, each
        with shape ``[B, T, 3]``.
    """
    _validate_actions(actions)
    canonical = _validate_config(config)
    boundary = _validate_observation_steps(num_observed_steps, actions.shape[1])
    future_steps = actions.shape[1] - boundary

    minimum = _cast_config_value("min_anchor_forward_m", canonical.minimum, actions)
    brake_final = _cast_config_value("brake_final_scale", canonical.brake, actions)
    accelerate_final = _cast_config_value("accelerate_final_scale", canonical.accelerate, actions)
    theta = _cast_config_value("turn_yaw_radians_per_step", canonical.turn, actions)
    _validate_cast_range("min_anchor_forward_m", minimum, minimum.item() >= 0)
    _validate_cast_range("brake_final_scale", brake_final, 0 <= brake_final.item() < 1)
    _validate_cast_range("accelerate_final_scale", accelerate_final, accelerate_final.item() > 1)
    _validate_cast_range("turn_yaw_radians_per_step", theta, 0 < theta.item() < math.pi)

    if canonical.override is None:
        anchor = actions[:, boundary - 1, 0]
        if not torch.all(anchor > minimum).item():
            raise InsufficientActionAnchorError("each inferred anchor forward action must exceed min_anchor_forward_m")
    else:
        override = _cast_config_value("anchor_forward_override_m", canonical.override, actions)
        _validate_cast_range("anchor_forward_override_m", override, override.item() > 0)
        anchor = override.expand(actions.shape[0])

    brake_scale = torch.linspace(
        1.0, brake_final.item(), future_steps + 1, dtype=actions.dtype, device=actions.device
    )[1:]
    accelerate_scale = torch.linspace(
        1.0, accelerate_final.item(), future_steps + 1, dtype=actions.dtype, device=actions.device
    )[1:]
    _validate_finite("brake_scale", brake_scale)
    _validate_finite("accelerate_scale", accelerate_scale)
    if not torch.all(brake_scale < 1).item() or not torch.all(brake_scale.diff() < 0).item():
        raise ValueError("brake_final_scale produces no decreasing scale after dtype cast")
    if not torch.all(accelerate_scale > 1).item() or not torch.all(accelerate_scale.diff() > 0).item():
        raise ValueError("accelerate_final_scale produces no increasing scale after dtype cast")

    turn_forward = anchor * (torch.sin(theta) / theta)
    turn_lateral = anchor * (2 * torch.sin(theta / 2).square() / theta)
    _validate_finite("turn_forward", turn_forward)
    _validate_finite("turn_lateral", turn_lateral)
    if not torch.all(turn_forward > 0).item():
        raise ValueError("turn_forward must be positive after dtype computation")
    if not torch.all(turn_lateral > 0).item():
        raise ValueError("turn_lateral must be positive after dtype computation")

    factual = actions.clone()
    brake = actions.clone()
    left_turn = actions.clone()
    right_turn = actions.clone()
    accelerate = actions.clone()
    brake[:, boundary:, 0] = anchor[:, None] * brake_scale
    brake[:, boundary:, 1:] = 0
    accelerate[:, boundary:, 0] = anchor[:, None] * accelerate_scale
    accelerate[:, boundary:, 1:] = 0
    left_turn[:, boundary:, 0] = turn_forward[:, None]
    left_turn[:, boundary:, 1] = turn_lateral[:, None]
    left_turn[:, boundary:, 2] = theta
    right_turn[:, boundary:, 0] = turn_forward[:, None]
    right_turn[:, boundary:, 1] = -turn_lateral[:, None]
    right_turn[:, boundary:, 2] = -theta
    scenarios = {
        "factual": factual,
        "brake": brake,
        "left_turn": left_turn,
        "right_turn": right_turn,
        "accelerate": accelerate,
    }
    for scenario_name, scenario in scenarios.items():
        _validate_finite(f"{scenario_name} scenario", scenario)
    return scenarios


def integrate_ego_actions(actions: Tensor) -> Tensor:
    """Integrate ego-frame action increments into global poses.

    Parameters
    ----------
    actions : [B, T, 3]
        Ego-frame action increments ``(forward, lateral, yaw)`` with dtype
        ``torch.float32`` or ``torch.float64``.

    Returns
    -------
    Tensor
        Global poses ``(x, y, yaw)`` with shape ``[B, T + 1, 3]``. The first pose
        is the origin, and yaw is wrapped to ``[-pi, pi]`` after each increment.
    """
    _validate_actions(actions)
    batch_size = actions.shape[0]
    position = torch.zeros((batch_size, 2), dtype=actions.dtype, device=actions.device)
    yaw = torch.zeros((batch_size,), dtype=actions.dtype, device=actions.device)
    poses = [torch.cat((position, yaw[:, None]), dim=1)]
    for step in range(actions.shape[1]):
        forward, lateral, yaw_delta = actions[:, step].unbind(dim=1)
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        position = position + torch.stack(
            (cos_yaw * forward - sin_yaw * lateral, sin_yaw * forward + cos_yaw * lateral), dim=1
        )
        yaw = torch.atan2(torch.sin(yaw + yaw_delta), torch.cos(yaw + yaw_delta))
        poses.append(torch.cat((position, yaw[:, None]), dim=1))
    integrated_poses = torch.stack(poses, dim=1)
    if not torch.isfinite(integrated_poses).all().item():
        max_abs_action = actions.abs().max().item()
        raise ValueError(
            "integrated trajectory contains non-finite values for "
            f"dtype={actions.dtype}; max_abs_action={max_abs_action}"
        )
    return integrated_poses


_LATENT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_LATENT_EPSILON = 1e-8


def _validate_latent_tensor(name: str, latents: object) -> Tensor:
    """Validate one evaluation latent tensor with the exact ``[B, F, P, D]`` contract."""
    if not isinstance(latents, Tensor):
        raise TypeError(f"{name} must be a torch Tensor")
    if latents.layout != torch.strided:
        raise TypeError(f"{name} must have strided layout")
    if latents.dtype not in _LATENT_DTYPES:
        raise TypeError(f"{name} dtype must be one of {_LATENT_DTYPES}")
    if latents.ndim != 4:
        raise ValueError(f"{name} must have shape [B, F, P, D]")
    if any(dimension == 0 for dimension in latents.shape):
        raise ValueError(f"{name} must have non-empty B, F, P, and D dimensions")
    if not torch.isfinite(latents).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return latents


def _validate_latent_pair(left: object, right: object) -> tuple[Tensor, Tensor]:
    """Validate two compatible latent tensors before any pairwise arithmetic."""
    left_tensor = _validate_latent_tensor("left", left)
    right_tensor = _validate_latent_tensor("right", right)
    if left_tensor.shape != right_tensor.shape:
        raise ValueError("left and right must have the same shape")
    if left_tensor.dtype != right_tensor.dtype:
        raise ValueError("left and right must have the same dtype")
    if left_tensor.device != right_tensor.device:
        raise ValueError("left and right must have the same device")
    return left_tensor, right_tensor


def _ensure_finite_metrics(metrics: Dict[str, Tensor]) -> None:
    """Ensure metric tensors are finite before conversion to Python summaries."""
    for name, metric in metrics.items():
        if not torch.isfinite(metric).all().item():
            raise ValueError(f"computed {name} contains non-finite values")


def _scaled_vector_norm(values: Tensor) -> tuple[Tensor, Tensor]:
    """Return max-absolute scale and normalized last-dimension L2 norm."""
    scale = values.abs().amax(dim=-1)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = values / safe_scale.unsqueeze(dim=-1)
    return scale, torch.linalg.vector_norm(normalized, dim=-1)


def _normalized_from_scale(values: Tensor, scale: Tensor) -> Tensor:
    """Normalize values by a zero-safe last-dimension scale."""
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return values / safe_scale.unsqueeze(dim=-1)


def _log_scaled_norm(scale: Tensor, normalized_norm: Tensor) -> Tensor:
    """Return log L2 norm, using ``-inf`` for exactly zero values."""
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    safe_normalized_norm = torch.where(normalized_norm > 0, normalized_norm, torch.ones_like(normalized_norm))
    nonzero = (scale > 0) & (normalized_norm > 0)
    return torch.where(nonzero, safe_scale.log() + safe_normalized_norm.log(), torch.full_like(scale, -math.inf))


def _stable_nonnegative_mean(values: Tensor, dims: tuple[int, ...]) -> Tensor:
    """Reduce finite nonnegative values without an overflowing intermediate sum."""
    scale = values.amax(dim=dims, keepdim=True)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized_mean = (values / safe_scale).mean(dim=dims)
    for dim in sorted(dims, reverse=True):
        scale = scale.squeeze(dim=dim)
    return normalized_mean * scale


def compute_pairwise_latent_metrics(left: Tensor, right: Tensor) -> Dict[str, Any]:
    """Compute non-differentiable pairwise summaries for ``[B, F, P, D]`` latents.

    Inputs are strictly validated, detached, and accumulated in ``torch.float64`` on
    their original device. This function is an evaluation summary: it does not
    participate in autograd and returns only JSON-safe Python scalars and lists.

    Parameters
    ----------
    left, right : [B, F, P, D]
        Matching finite, strided tensors with dtype float16, bfloat16, float32, or
        float64.

    Returns
    -------
    Dict[str, Any]
        Per-future-step and aggregate cosine, L2, relative-L2, and norm-drift
        measurements.
    """
    left_tensor, right_tensor = _validate_latent_pair(left, right)
    left_64 = left_tensor.detach().to(dtype=torch.float64)
    right_64 = right_tensor.detach().to(dtype=torch.float64)
    epsilon = left_64.new_tensor(_LATENT_EPSILON)

    left_scale, left_normalized_norm = _scaled_vector_norm(left_64)
    right_scale, right_normalized_norm = _scaled_vector_norm(right_64)
    left_scaled = _normalized_from_scale(left_64, left_scale)
    right_scaled = _normalized_from_scale(right_64, right_scale)
    scaled_dot = (left_scaled * right_scaled).sum(dim=-1)
    scaled_norm_product = left_normalized_norm * right_normalized_norm
    angle_cosine = (scaled_dot / scaled_norm_product.clamp_min(epsilon)).clamp(min=-1.0, max=1.0)
    left_nonzero = (left_scale > 0) & (left_normalized_norm > 0)
    right_nonzero = (right_scale > 0) & (right_normalized_norm > 0)
    nonzero_pair = left_nonzero & right_nonzero
    left_log_norm = _log_scaled_norm(left_scale, left_normalized_norm)
    right_log_norm = _log_scaled_norm(right_scale, right_normalized_norm)
    log_magnitude_factor = (left_log_norm + right_log_norm - epsilon.log()).clamp_max(0.0)
    magnitude_factor = torch.where(nonzero_pair, log_magnitude_factor.exp(), torch.zeros_like(left_scale))
    cosine = angle_cosine * magnitude_factor
    l2_scale, l2_normalized_norm = _scaled_vector_norm(left_64 - right_64)
    l2 = l2_scale * l2_normalized_norm
    l2_nonzero = (l2_scale > 0) & (l2_normalized_norm > 0)
    l2_log_norm = _log_scaled_norm(l2_scale, l2_normalized_norm)
    log_epsilon = epsilon.log().expand_as(left_log_norm)
    log_relative_denominator = torch.logsumexp(
        torch.stack((left_log_norm + math.log(0.5), right_log_norm + math.log(0.5), log_epsilon)), dim=0
    )
    relative_l2 = torch.where(
        l2_nonzero,
        (l2_log_norm - log_relative_denominator).exp(),
        torch.zeros_like(l2_scale),
    )
    common_scale = torch.maximum(left_scale, right_scale)
    safe_common_scale = torch.where(common_scale > 0, common_scale, torch.ones_like(common_scale))
    norm_drift = (
        (left_scale / safe_common_scale) * left_normalized_norm
        - (right_scale / safe_common_scale) * right_normalized_norm
    ).abs() * common_scale
    metric_tensors = {
        "cosine": cosine,
        "l2": l2,
        "relative_l2": relative_l2,
        "norm_drift": norm_drift,
    }
    _ensure_finite_metrics(metric_tensors)

    per_step = {
        "cosine_mean_per_step": cosine.mean(dim=(0, 2)),
        "l2_mean_per_step": _stable_nonnegative_mean(l2, dims=(0, 2)),
        "l2_max_per_step": l2.amax(dim=(0, 2)),
        "relative_l2_mean_per_step": _stable_nonnegative_mean(relative_l2, dims=(0, 2)),
        "norm_drift_mean_per_step": _stable_nonnegative_mean(norm_drift, dims=(0, 2)),
    }
    aggregates = {
        "cosine_mean": cosine.mean(),
        "l2_mean": _stable_nonnegative_mean(l2, dims=(0, 1, 2)),
        "l2_max": l2.max(),
        "relative_l2_mean": _stable_nonnegative_mean(relative_l2, dims=(0, 1, 2)),
        "norm_drift_mean": _stable_nonnegative_mean(norm_drift, dims=(0, 1, 2)),
    }
    _ensure_finite_metrics({**per_step, **aggregates})
    return {
        "future_steps": int(left_tensor.shape[1]),
        **{name: values.detach().cpu().tolist() for name, values in per_step.items()},
        **{name: float(value.detach().cpu().item()) for name, value in aggregates.items()},
    }


def compute_all_scenario_metrics(latents: Mapping[str, Tensor]) -> List[Dict[str, Any]]:
    """Compute flattened pairwise summaries for the ten fixed scenario pairs.

    The mapping must contain exactly ``SCENARIO_NAMES`` and every value must obey
    the same exact latent tensor contract. Output pair order is independent of the
    mapping insertion order and follows ``itertools.combinations(SCENARIO_NAMES, 2)``.
    """
    if not isinstance(latents, Mapping):
        raise TypeError("latents must be a Mapping")
    actual_keys = set(latents)
    expected_keys = set(SCENARIO_NAMES)
    missing = sorted(expected_keys - actual_keys, key=repr)
    extra = sorted(actual_keys - expected_keys, key=repr)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing keys: {missing}")
        if extra:
            details.append(f"extra keys: {extra}")
        raise ValueError("latents must have exactly SCENARIO_NAMES; " + "; ".join(details))

    ordered_latents = {name: _validate_latent_tensor(name, latents[name]) for name in SCENARIO_NAMES}
    reference = ordered_latents[SCENARIO_NAMES[0]]
    for name in SCENARIO_NAMES[1:]:
        candidate = ordered_latents[name]
        if candidate.shape != reference.shape:
            raise ValueError(f"{name} must have the same shape as {SCENARIO_NAMES[0]}")
        if candidate.dtype != reference.dtype:
            raise ValueError(f"{name} must have the same dtype as {SCENARIO_NAMES[0]}")
        if candidate.device != reference.device:
            raise ValueError(f"{name} must have the same device as {SCENARIO_NAMES[0]}")

    records = []
    for scenario_a, scenario_b in combinations(SCENARIO_NAMES, 2):
        records.append(
            {
                "scenario_a": scenario_a,
                "scenario_b": scenario_b,
                **compute_pairwise_latent_metrics(ordered_latents[scenario_a], ordered_latents[scenario_b]),
            }
        )
    return records
