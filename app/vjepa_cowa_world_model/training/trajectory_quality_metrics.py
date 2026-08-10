"""Route-free progress and comfort diagnostics for predicted ego trajectories."""

from __future__ import annotations

import math
from typing import Dict

import torch

TRAJECTORY_QUALITY_METRIC_KEYS = (
    "longitudinal_progress_m",
    "forward_progress_m",
    "reverse_distance_m",
    "path_length_m",
    "progress_efficiency",
    "accel_mean_mps2",
    "accel_violation_rate",
    "jerk_mean_mps3",
    "jerk_violation_rate",
    "yaw_rate_mean_radps",
    "yaw_rate_violation_rate",
    "comfort_risk",
)


def _validate_anchor(name: str, value: torch.Tensor, shape: tuple[int, ...], trajectory: torch.Tensor) -> None:
    if not torch.is_tensor(value) or value.shape != shape:
        actual = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        raise ValueError(f"{name} must have shape {shape}, got {actual}")
    if value.device != trajectory.device:
        raise ValueError(f"{name} device {value.device} must match trajectory device {trajectory.device}")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype, got {value.dtype}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")


def compute_trajectory_quality_metrics_per_sample(
    trajectory: torch.Tensor,
    *,
    anchor_velocity: torch.Tensor,
    anchor_acceleration: torch.Tensor,
    anchor_yaw_rate: torch.Tensor,
    timestep_sec: float,
    accel_threshold: float = 6.0,
    jerk_threshold: float = 8.0,
    yaw_rate_threshold: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute non-safety trajectory diagnostics for each sample.

    ``trajectory`` is the selected predicted ego path in its origin frame with
    shape ``[B, P, 3]`` and channels ``x, y, yaw``.
    """

    if not torch.is_tensor(trajectory) or trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        actual = tuple(trajectory.shape) if torch.is_tensor(trajectory) else type(trajectory).__name__
        raise ValueError(f"trajectory must have shape [B, P, 3], got {actual}")
    if trajectory.shape[0] < 1 or trajectory.shape[1] < 1:
        raise ValueError(f"trajectory batch and horizon must be non-empty, got {trajectory.shape}")
    if not trajectory.dtype.is_floating_point:
        raise TypeError(f"trajectory must use a floating dtype, got {trajectory.dtype}")
    if not bool(torch.isfinite(trajectory).all().item()):
        raise ValueError("trajectory must contain only finite values")

    batch_size = int(trajectory.shape[0])
    _validate_anchor("anchor_velocity", anchor_velocity, (batch_size, 2), trajectory)
    _validate_anchor("anchor_acceleration", anchor_acceleration, (batch_size, 2), trajectory)
    _validate_anchor("anchor_yaw_rate", anchor_yaw_rate, (batch_size,), trajectory)

    timestep_sec = float(timestep_sec)
    if not math.isfinite(timestep_sec) or timestep_sec <= 0.0:
        raise ValueError(f"timestep_sec must be finite and positive, got {timestep_sec}")
    thresholds = {
        "accel_threshold": float(accel_threshold),
        "jerk_threshold": float(jerk_threshold),
        "yaw_rate_threshold": float(yaw_rate_threshold),
    }
    invalid_thresholds = {name: value for name, value in thresholds.items() if not math.isfinite(value) or value <= 0}
    if invalid_thresholds:
        raise ValueError(f"trajectory quality thresholds must be finite and positive, got {invalid_thresholds}")

    origin_xy = trajectory.new_zeros(batch_size, 1, 2)
    origin_yaw = trajectory.new_zeros(batch_size, 1)
    xy = torch.cat([origin_xy, trajectory[..., :2]], dim=1)
    yaw = torch.cat([origin_yaw, trajectory[..., 2]], dim=1)
    delta_xy = xy[:, 1:] - xy[:, :-1]
    heading = yaw[:, :-1]
    longitudinal = torch.cos(heading) * delta_xy[..., 0] + torch.sin(heading) * delta_xy[..., 1]
    path_step = torch.linalg.vector_norm(delta_xy, dim=-1)
    path_length = path_step.sum(dim=1)
    progress = longitudinal.sum(dim=1)

    future_velocity = delta_xy / timestep_sec
    velocity = torch.cat([anchor_velocity[:, None], future_velocity], dim=1)
    future_acceleration = (velocity[:, 1:] - velocity[:, :-1]) / timestep_sec
    acceleration = torch.cat([anchor_acceleration[:, None], future_acceleration], dim=1)
    jerk = (acceleration[:, 1:] - acceleration[:, :-1]) / timestep_sec
    yaw_delta = torch.atan2(torch.sin(yaw[:, 1:] - yaw[:, :-1]), torch.cos(yaw[:, 1:] - yaw[:, :-1]))
    yaw_rate = yaw_delta / timestep_sec

    accel_magnitude = torch.linalg.vector_norm(future_acceleration, dim=-1)
    jerk_magnitude = torch.linalg.vector_norm(jerk, dim=-1)
    yaw_rate_magnitude = yaw_rate.abs()
    accel_risk = (accel_magnitude / thresholds["accel_threshold"]).clamp_min(0.0)
    jerk_risk = (jerk_magnitude / thresholds["jerk_threshold"]).clamp_min(0.0)
    yaw_risk = (yaw_rate_magnitude / thresholds["yaw_rate_threshold"]).clamp_min(0.0)
    boundary_yaw_risk = (yaw_rate[:, 0] - anchor_yaw_rate).abs() / thresholds["yaw_rate_threshold"]
    yaw_risk = torch.cat([torch.maximum(yaw_risk[:, 0], boundary_yaw_risk)[:, None], yaw_risk[:, 1:]], dim=1)
    comfort_risk = torch.maximum(torch.maximum(accel_risk, jerk_risk), yaw_risk).mean(dim=1)

    result = {
        "longitudinal_progress_m": progress,
        "forward_progress_m": longitudinal.clamp_min(0.0).sum(dim=1),
        "reverse_distance_m": (-longitudinal).clamp_min(0.0).sum(dim=1),
        "path_length_m": path_length,
        "progress_efficiency": torch.where(path_length > 0.0, progress / path_length, torch.zeros_like(progress)),
        "accel_mean_mps2": accel_magnitude.mean(dim=1),
        "accel_violation_rate": (accel_magnitude > thresholds["accel_threshold"]).float().mean(dim=1),
        "jerk_mean_mps3": jerk_magnitude.mean(dim=1),
        "jerk_violation_rate": (jerk_magnitude > thresholds["jerk_threshold"]).float().mean(dim=1),
        "yaw_rate_mean_radps": yaw_rate_magnitude.mean(dim=1),
        "yaw_rate_violation_rate": (yaw_rate_magnitude > thresholds["yaw_rate_threshold"]).float().mean(dim=1),
        "comfort_risk": comfort_risk,
    }
    if tuple(result) != TRAJECTORY_QUALITY_METRIC_KEYS:
        raise RuntimeError("trajectory quality metric schema drift")
    return result
