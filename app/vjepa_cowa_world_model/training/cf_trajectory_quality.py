"""Geometry-free trajectory-quality labels for counterfactual stress data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

from app.vjepa_cowa_world_model.losses.reward_selector import compute_comfort_risk
from app.vjepa_cowa_world_model.training.reward_labels import RewardLabelConfig

CF_QUALITY_SCHEMA = "cf_progress_reverse_comfort_efficiency_v1"
_PROGRESS_WEIGHT = 0.4
_REVERSE_WEIGHT = 0.2
_COMFORT_WEIGHT = 0.2
_EFFICIENCY_WEIGHT = 0.2


@dataclass(frozen=True)
class CounterfactualTrajectoryQuality:
    """Confidence-selected, geometry-free CF trajectory metrics in ``[0, 1]``."""

    selected_mode: torch.Tensor
    selected_trajectory: torch.Tensor
    progress_m: torch.Tensor
    progress_score: torch.Tensor
    reverse_risk: torch.Tensor
    comfort_score: torch.Tensor
    path_efficiency: torch.Tensor
    quality_score: torch.Tensor


def counterfactual_quality_schema(*, timestep_sec: float, max_progress_m: float) -> dict[str, object]:
    """Return the immutable, geometry-free CF label definition for checkpoints."""

    timestep_sec = float(timestep_sec)
    max_progress_m = float(max_progress_m)
    if not math.isfinite(timestep_sec) or timestep_sec <= 0.0:
        raise ValueError(f"timestep_sec must be finite and positive, got {timestep_sec}")
    if not math.isfinite(max_progress_m) or max_progress_m <= 0.0:
        raise ValueError(f"max_progress_m must be finite and positive, got {max_progress_m}")
    comfort_config = RewardLabelConfig(timestep_sec=timestep_sec)
    return {
        "name": CF_QUALITY_SCHEMA,
        "inputs": ["trajectory", "deployment_confidence"],
        "forbidden_inputs": ["agent_boxes", "collision", "ttc", "offroad"],
        "timestep_sec": timestep_sec,
        "max_progress_m": max_progress_m,
        "comfort": {
            "accel_threshold": comfort_config.accel_threshold,
            "accel_margin": comfort_config.accel_margin,
            "yaw_rate_threshold": comfort_config.yaw_rate_threshold,
            "yaw_rate_margin": comfort_config.yaw_rate_margin,
            "jerk_threshold": comfort_config.jerk_threshold,
            "jerk_margin": comfort_config.jerk_margin,
            "risk_boundary": "clamp((abs(value)-threshold)/margin,0,1)",
            "aggregation": "max_over_steps_and_accel_yaw_rate_jerk",
        },
        "weights": {
            "progress": _PROGRESS_WEIGHT,
            "non_reverse": _REVERSE_WEIGHT,
            "comfort": _COMFORT_WEIGHT,
            "path_efficiency": _EFFICIENCY_WEIGHT,
        },
    }


def compute_counterfactual_trajectory_quality(
    pred_trajs: torch.Tensor,
    confidences: torch.Tensor,
    *,
    dataset_domains: Sequence[str],
    timestep_sec: float = 0.5,
    max_progress_m: float = 20.0,
) -> CounterfactualTrajectoryQuality:
    """Score deployed CF trajectories without geometry or safety labels.

    Progress is measured along the current ego's forward x-axis. Reverse risk
    is the fraction of traveled distance with negative x displacement, and path
    efficiency is net displacement divided by traveled distance. The fixed
    schema combines progress, non-reverse behavior, comfort, and efficiency.
    """

    if not torch.is_tensor(pred_trajs) or pred_trajs.ndim != 4 or pred_trajs.shape[-1] != 3:
        shape = tuple(pred_trajs.shape) if torch.is_tensor(pred_trajs) else type(pred_trajs).__name__
        raise ValueError(f"pred_trajs must be [B, K, P, 3], got {shape}")
    if not pred_trajs.dtype.is_floating_point or not torch.isfinite(pred_trajs).all():
        raise ValueError("pred_trajs must contain finite floating-point values")
    batch_size, num_modes, num_poses, _ = pred_trajs.shape
    if batch_size < 1 or num_modes < 1 or num_poses < 1:
        raise ValueError("pred_trajs batch, mode, and pose dimensions must be non-empty")
    if not torch.is_tensor(confidences) or confidences.shape != (batch_size, num_modes):
        shape = tuple(confidences.shape) if torch.is_tensor(confidences) else type(confidences).__name__
        raise ValueError(f"confidences must be [{batch_size}, {num_modes}], got {shape}")
    if (
        not confidences.dtype.is_floating_point
        or not torch.isfinite(confidences).all()
        or confidences.device != pred_trajs.device
    ):
        raise ValueError("confidences must be finite floating point on the pred_trajs device")
    if isinstance(dataset_domains, (str, bytes)) or len(dataset_domains) != batch_size:
        raise ValueError("dataset_domains must contain one entry per sample")
    if any(str(domain) != "counterfactual" for domain in dataset_domains):
        raise ValueError("counterfactual trajectory-quality evaluation is counterfactual-only")
    timestep_sec = float(timestep_sec)
    max_progress_m = float(max_progress_m)
    if not math.isfinite(timestep_sec) or timestep_sec <= 0.0:
        raise ValueError(f"timestep_sec must be finite and positive, got {timestep_sec}")
    if not math.isfinite(max_progress_m) or max_progress_m <= 0.0:
        raise ValueError(f"max_progress_m must be finite and positive, got {max_progress_m}")

    selected_mode = confidences.argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=pred_trajs.device)
    selected = pred_trajs[batch_indices, selected_mode]

    origin_xy = selected.new_zeros((batch_size, 1, 2))
    points_xy = torch.cat([origin_xy, selected[..., :2]], dim=1)
    displacements = points_xy[:, 1:] - points_xy[:, :-1]
    segment_lengths = torch.linalg.vector_norm(displacements, dim=-1)
    path_length = segment_lengths.sum(dim=1)
    net_displacement = torch.linalg.vector_norm(selected[:, -1, :2], dim=-1)

    progress_m = selected[:, -1, 0]
    progress_score = torch.clamp(progress_m / max_progress_m, min=0.0, max=1.0)
    reverse_distance = torch.clamp(-displacements[..., 0], min=0.0).sum(dim=1)
    reverse_risk = torch.where(
        path_length > 1e-6,
        torch.clamp(reverse_distance / path_length.clamp_min(1e-6), min=0.0, max=1.0),
        torch.zeros_like(path_length),
    )
    path_efficiency = torch.where(
        path_length > 1e-6,
        torch.clamp(net_displacement / path_length.clamp_min(1e-6), min=0.0, max=1.0),
        torch.zeros_like(path_length),
    )
    comfort_modes = compute_comfort_risk(
        pred_trajs,
        timestep_sec=timestep_sec,
        config=RewardLabelConfig(timestep_sec=timestep_sec),
    )
    comfort_score = 1.0 - comfort_modes[batch_indices, selected_mode]
    quality_score = (
        _PROGRESS_WEIGHT * progress_score
        + _REVERSE_WEIGHT * (1.0 - reverse_risk)
        + _COMFORT_WEIGHT * comfort_score
        + _EFFICIENCY_WEIGHT * path_efficiency
    )
    quality_score = torch.clamp(quality_score, min=0.0, max=1.0)
    return CounterfactualTrajectoryQuality(
        selected_mode=selected_mode,
        selected_trajectory=selected,
        progress_m=progress_m,
        progress_score=progress_score,
        reverse_risk=reverse_risk,
        comfort_score=comfort_score,
        path_efficiency=path_efficiency,
        quality_score=quality_score,
    )
