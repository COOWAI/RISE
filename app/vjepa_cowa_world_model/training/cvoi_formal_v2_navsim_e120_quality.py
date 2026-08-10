"""Strict route-free supervision for the Formal-v2 NavSim-e120 lifecycle.

This module defines an internal training target.  It is deliberately separate
from both official NavSim evaluation protocols and from the legacy
geometry-based CVoI outcome path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Callable, Dict

import torch

from app.vjepa_cowa_world_model.losses.reward_selector import compute_comfort_risk
from app.vjepa_cowa_world_model.training.cvoi_execution import common_random_numbers
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_MAX_HORIZON
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import CvoiPlannerEvaluation
from app.vjepa_cowa_world_model.training.reward_labels import RewardLabelConfig
from app.vjepa_cowa_world_model.training.sequential_budget_control import CVOI_GUIDANCE_STEPS

CVOI_NAVSIM_E120_QUALITY_NAME = "navsim_e120_route_free_trajectory_quality_v1"
CVOI_NAVSIM_E120_TIMESTEP_SEC = 0.5
CVOI_NAVSIM_E120_MAX_PROGRESS_M = 20.0

_PROGRESS_WEIGHT = 0.4
_NON_REVERSE_WEIGHT = 0.2
_COMFORT_WEIGHT = 0.2
_EFFICIENCY_WEIGHT = 0.2


@dataclass(frozen=True)
class NavSimE120TrajectoryQuality:
    """Confidence-selected route-free trajectory quality in ``[0, 1]``."""

    selected_mode: torch.Tensor
    selected_trajectory: torch.Tensor
    progress_m: torch.Tensor
    progress: torch.Tensor
    reverse_risk: torch.Tensor
    comfort: torch.Tensor
    path_efficiency: torch.Tensor
    quality: torch.Tensor


@dataclass(frozen=True)
class NavSimE120QualitySample:
    """Identity and deterministic seed for one real NavSim sample."""

    sample_id: str
    source_scene_id: str
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("NavSim-e120 quality sample_id must be non-empty")
        if not isinstance(self.source_scene_id, str) or not self.source_scene_id:
            raise ValueError("NavSim-e120 quality source_scene_id must be non-empty")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("NavSim-e120 quality seed must be a non-negative integer")


@dataclass(frozen=True)
class NavSimE120DirectLocalQualityTargets:
    """Audit-free local targets for the direct manual training chain."""

    sample_id: str
    group_ids: tuple[str, ...]
    candidate_latents: torch.Tensor
    quality_targets: torch.Tensor


@dataclass(frozen=True)
class NavSimE120DirectStopQualityTargets:
    """Audit-free Stop targets for the direct manual training chain."""

    sample_id: str
    quality_targets: torch.Tensor
    latency_ms: torch.Tensor
    quality_components: Dict[str, torch.Tensor]


def navsim_e120_quality_schema(*, timestep_sec: float, max_progress_m: float) -> Dict[str, object]:
    """Return the canonical internal route-free target definition."""

    timestep_sec = float(timestep_sec)
    max_progress_m = float(max_progress_m)
    if not math.isfinite(timestep_sec) or timestep_sec <= 0.0:
        raise ValueError(f"timestep_sec must be finite and positive, got {timestep_sec}")
    if not math.isfinite(max_progress_m) or max_progress_m <= 0.0:
        raise ValueError(f"max_progress_m must be finite and positive, got {max_progress_m}")
    comfort_config = RewardLabelConfig(timestep_sec=timestep_sec)
    return {
        "name": CVOI_NAVSIM_E120_QUALITY_NAME,
        "scope": "internal_training_target",
        "official_evaluation": False,
        "inputs": ["planner_trajectory", "deployment_confidence"],
        "forbidden_inputs": ["ground_truth_trajectory", "agent_geometry", "map", "route"],
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
            "non_reverse": _NON_REVERSE_WEIGHT,
            "comfort": _COMFORT_WEIGHT,
            "path_efficiency": _EFFICIENCY_WEIGHT,
        },
    }


def navsim_e120_quality_schema_sha256(*, timestep_sec: float, max_progress_m: float) -> str:
    """Hash the canonical quality definition using deterministic JSON."""

    payload = navsim_e120_quality_schema(timestep_sec=timestep_sec, max_progress_m=max_progress_m)
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def score_navsim_e120_trajectory_quality(
    pred_trajs: torch.Tensor,
    confidences: torch.Tensor,
    *,
    timestep_sec: float,
    max_progress_m: float,
) -> NavSimE120TrajectoryQuality:
    """Score the deployed trajectory without reading future dataset geometry."""

    navsim_e120_quality_schema(timestep_sec=timestep_sec, max_progress_m=max_progress_m)
    if not isinstance(pred_trajs, torch.Tensor) or pred_trajs.ndim != 4 or pred_trajs.shape[-1] != 3:
        shape = tuple(pred_trajs.shape) if isinstance(pred_trajs, torch.Tensor) else type(pred_trajs).__name__
        raise ValueError(f"pred_trajs must be [B, K, P, 3], got {shape}")
    if not pred_trajs.is_floating_point() or not bool(torch.isfinite(pred_trajs).all().item()):
        raise ValueError("pred_trajs must contain finite floating-point values")
    batch_size, num_modes, num_poses, _ = pred_trajs.shape
    if batch_size < 1 or num_modes < 1 or num_poses < 1:
        raise ValueError("pred_trajs batch, mode, and pose dimensions must be non-empty")
    if (
        not isinstance(confidences, torch.Tensor)
        or confidences.shape != (batch_size, num_modes)
        or not confidences.is_floating_point()
        or not bool(torch.isfinite(confidences).all().item())
        or confidences.device != pred_trajs.device
    ):
        raise ValueError(f"confidences must be finite floating [{batch_size}, {num_modes}] on the trajectory device")

    selected_mode = confidences.argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=pred_trajs.device)
    selected = pred_trajs[batch_indices, selected_mode]
    origin_xy = selected.new_zeros((batch_size, 1, 2))
    points_xy = torch.cat((origin_xy, selected[..., :2]), dim=1)
    displacements = points_xy[:, 1:] - points_xy[:, :-1]
    segment_lengths = torch.linalg.vector_norm(displacements, dim=-1)
    path_length = segment_lengths.sum(dim=1)
    net_displacement = torch.linalg.vector_norm(selected[:, -1, :2], dim=-1)
    progress_m = selected[:, -1, 0]
    progress = torch.clamp(progress_m / float(max_progress_m), min=0.0, max=1.0)
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
    comfort_risk = compute_comfort_risk(
        pred_trajs,
        timestep_sec=float(timestep_sec),
        config=RewardLabelConfig(timestep_sec=float(timestep_sec)),
    )
    comfort = 1.0 - comfort_risk[batch_indices, selected_mode]
    quality = torch.clamp(
        _PROGRESS_WEIGHT * progress
        + _NON_REVERSE_WEIGHT * (1.0 - reverse_risk)
        + _COMFORT_WEIGHT * comfort
        + _EFFICIENCY_WEIGHT * path_efficiency,
        min=0.0,
        max=1.0,
    )
    return NavSimE120TrajectoryQuality(
        selected_mode=selected_mode,
        selected_trajectory=selected,
        progress_m=progress_m,
        progress=progress,
        reverse_risk=reverse_risk,
        comfort=comfort,
        path_efficiency=path_efficiency,
        quality=quality,
    )


def _validate_quality_sample(sample: NavSimE120QualitySample) -> None:
    if not isinstance(sample, NavSimE120QualitySample):
        raise TypeError("sample must be NavSimE120QualitySample")


def _validate_direct_controller_lineage(controller_lineage: str) -> str:
    if controller_lineage != "value_guided":
        raise ValueError("direct NavSim-e120 Stop quality collection requires controller_lineage='value_guided'")
    return controller_lineage


def _validate_base_future_latent(base_future_latent: torch.Tensor) -> torch.Tensor:
    if not isinstance(base_future_latent, torch.Tensor):
        raise TypeError("base_future_latent must be a torch.Tensor")
    if base_future_latent.ndim != 4 or base_future_latent.shape[0] != 1:
        raise ValueError("base_future_latent must have shape [1, F, T, D], " f"got {tuple(base_future_latent.shape)}")
    if any(size < 1 for size in base_future_latent.shape[1:]):
        raise ValueError("base_future_latent future, token, and embedding dimensions must be non-empty")
    if not base_future_latent.is_floating_point() or not bool(torch.isfinite(base_future_latent).all().item()):
        raise ValueError("base_future_latent must contain finite floating-point values")
    return base_future_latent.detach().clone()


def _positive_finite_float(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value}")
    return value


def _local_candidates(
    base: torch.Tensor,
    *,
    num_perturbations: int,
    perturbation_scale: float,
    max_delta_norm: float,
    seed: int,
) -> torch.Tensor:
    if type(num_perturbations) is not int or num_perturbations < 1:
        raise ValueError("num_perturbations must be a positive integer")
    perturbation_scale = _positive_finite_float(perturbation_scale, name="perturbation_scale")
    max_delta_norm = _positive_finite_float(max_delta_norm, name="max_delta_norm")
    _, num_future, num_tokens, embed_dim = base.shape
    generator = torch.Generator(device=base.device)
    generator.manual_seed(seed)
    deltas = torch.randn(
        (num_perturbations, num_future, num_tokens, embed_dim),
        dtype=base.dtype,
        device=base.device,
        generator=generator,
    )
    deltas.mul_(perturbation_scale)
    token_norms = torch.linalg.vector_norm(deltas.float(), dim=-1, keepdim=True)
    clip_factors = torch.clamp(
        max_delta_norm / token_norms.clamp_min(torch.finfo(torch.float32).eps),
        max=1.0,
    )
    deltas.mul_(clip_factors.to(dtype=deltas.dtype))
    perturbed = base.expand(num_perturbations, -1, -1, -1) + deltas
    return torch.cat((base, perturbed), dim=0)


def _validate_planner_result(
    result: object,
    *,
    expected_guidance_steps: int,
) -> CvoiPlannerEvaluation:
    if not isinstance(result, CvoiPlannerEvaluation):
        raise TypeError("quality callback must return CvoiPlannerEvaluation")
    if type(result.guidance_steps) is not int or result.guidance_steps != expected_guidance_steps:
        raise ValueError(
            "quality callback reported an unexpected guidance step count: "
            f"expected={expected_guidance_steps}, got={result.guidance_steps!r}"
        )
    latency_ms = float(result.latency_ms)
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise ValueError("quality callback latency_ms must be finite and non-negative")
    return result


def _collect_navsim_e120_local_quality_target_values(
    sample: NavSimE120QualitySample,
    *,
    base_future_latent: torch.Tensor,
    num_perturbations: int,
    perturbation_scale: float,
    max_delta_norm: float,
    evaluate_prefix: Callable[[torch.Tensor, int, int], CvoiPlannerEvaluation],
    timestep_sec: float,
    max_progress_m: float,
    calibration_mode: str,
) -> tuple[tuple[str, ...], torch.Tensor, torch.Tensor]:
    _validate_quality_sample(sample)
    if not callable(evaluate_prefix):
        raise TypeError("evaluate_prefix must be callable")
    base = _validate_base_future_latent(base_future_latent)
    if calibration_mode in {"local_geometry", "local_geometry_no_order"}:
        candidates = _local_candidates(
            base,
            num_perturbations=num_perturbations,
            perturbation_scale=perturbation_scale,
            max_delta_norm=max_delta_norm,
            seed=sample.seed,
        )
    elif calibration_mode == "factual_only":
        if type(num_perturbations) is not int or num_perturbations < 1:
            raise ValueError("num_perturbations must be a positive integer")
        candidates = base
    else:
        raise ValueError(
            "calibration_mode must be one of " "['factual_only', 'local_geometry', 'local_geometry_no_order']"
        )

    num_candidates, num_future, _, _ = candidates.shape
    targets = torch.empty((num_candidates, num_future), dtype=torch.float32, device=candidates.device)
    for candidate_index in range(num_candidates):
        for horizon in range(1, num_future + 1):
            prefix = candidates[candidate_index : candidate_index + 1, :horizon].clone()
            with torch.no_grad(), common_random_numbers(sample.seed):
                result = _validate_planner_result(
                    evaluate_prefix(prefix, horizon, sample.seed),
                    expected_guidance_steps=0,
                )
                quality = score_navsim_e120_trajectory_quality(
                    result.pred_trajs,
                    result.confidences,
                    timestep_sec=timestep_sec,
                    max_progress_m=max_progress_m,
                ).quality
            targets[candidate_index, horizon - 1] = quality[0].to(device=targets.device, dtype=targets.dtype)
    return (sample.sample_id,) * num_candidates, candidates, targets


def collect_navsim_e120_local_quality_targets_direct(
    sample: NavSimE120QualitySample,
    *,
    base_future_latent: torch.Tensor,
    num_perturbations: int,
    perturbation_scale: float,
    max_delta_norm: float,
    evaluate_prefix: Callable[[torch.Tensor, int, int], CvoiPlannerEvaluation],
    timestep_sec: float,
    max_progress_m: float,
    calibration_mode: str,
) -> NavSimE120DirectLocalQualityTargets:
    """Collect proof-free local Field targets for the direct manual chain."""

    group_ids, candidates, targets = _collect_navsim_e120_local_quality_target_values(
        sample,
        base_future_latent=base_future_latent,
        num_perturbations=num_perturbations,
        perturbation_scale=perturbation_scale,
        max_delta_norm=max_delta_norm,
        evaluate_prefix=evaluate_prefix,
        timestep_sec=timestep_sec,
        max_progress_m=max_progress_m,
        calibration_mode=calibration_mode,
    )
    return NavSimE120DirectLocalQualityTargets(
        sample_id=sample.sample_id,
        group_ids=group_ids,
        candidate_latents=candidates,
        quality_targets=targets,
    )


def _collect_navsim_e120_stop_quality_target_values(
    sample: NavSimE120QualitySample,
    *,
    max_horizon: int,
    evaluate_horizon: Callable[[int, bool, int], CvoiPlannerEvaluation],
    timestep_sec: float,
    max_progress_m: float,
    controller_lineage: str,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    _validate_quality_sample(sample)
    controller_lineage = _validate_direct_controller_lineage(controller_lineage)
    if not callable(evaluate_horizon):
        raise TypeError("evaluate_horizon must be callable")
    if int(max_horizon) != FORMAL_V2_NAVSIM_MAX_HORIZON:
        raise ValueError(f"NavSim-e120 Stop max_horizon must be {FORMAL_V2_NAVSIM_MAX_HORIZON}")
    qualities = []
    latencies = []
    components: Dict[str, list[torch.Tensor]] = {
        "progress": [],
        "non_reverse": [],
        "comfort": [],
        "path_efficiency": [],
    }
    for horizon in range(int(max_horizon) + 1):
        apply_guidance = controller_lineage == "value_guided" and horizon > 0
        expected_steps = CVOI_GUIDANCE_STEPS if apply_guidance else 0
        with torch.no_grad(), common_random_numbers(sample.seed):
            result = _validate_planner_result(
                evaluate_horizon(horizon, apply_guidance, sample.seed),
                expected_guidance_steps=expected_steps,
            )
            quality = score_navsim_e120_trajectory_quality(
                result.pred_trajs,
                result.confidences,
                timestep_sec=timestep_sec,
                max_progress_m=max_progress_m,
            )
        qualities.append(quality.quality)
        latencies.append(float(result.latency_ms))
        components["progress"].append(quality.progress)
        components["non_reverse"].append(1.0 - quality.reverse_risk)
        components["comfort"].append(quality.comfort)
        components["path_efficiency"].append(quality.path_efficiency)
    return (
        torch.stack(qualities, dim=1),
        torch.tensor(latencies, dtype=torch.float32),
        {name: torch.stack(values, dim=1) for name, values in components.items()},
    )


def collect_navsim_e120_stop_quality_target_direct(
    sample: NavSimE120QualitySample,
    *,
    max_horizon: int,
    evaluate_horizon: Callable[[int, bool, int], CvoiPlannerEvaluation],
    timestep_sec: float,
    max_progress_m: float,
    controller_lineage: str,
) -> NavSimE120DirectStopQualityTargets:
    """Collect proof-free Stop observations for the direct manual chain."""

    controller_lineage = _validate_direct_controller_lineage(controller_lineage)
    quality_targets, latency_ms, quality_components = _collect_navsim_e120_stop_quality_target_values(
        sample,
        max_horizon=max_horizon,
        evaluate_horizon=evaluate_horizon,
        timestep_sec=timestep_sec,
        max_progress_m=max_progress_m,
        controller_lineage=controller_lineage,
    )
    return NavSimE120DirectStopQualityTargets(
        sample_id=sample.sample_id,
        quality_targets=quality_targets,
        latency_ms=latency_ms,
        quality_components=quality_components,
    )
