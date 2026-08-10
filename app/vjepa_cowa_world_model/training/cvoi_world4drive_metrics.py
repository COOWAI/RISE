"""Canonical World4Drive metric adapter for CVoI horizon evaluation."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch

from app.vjepa_cowa_world_model.utils.metrics import BEV_SIZE, compute_collision_rate, compute_world4drive_l2_metrics

CVOI_WORLD4DRIVE_TASK_SCORE_SCHEMA = "cvoi_world4drive_l2_collision_task_score_v2"
_TASK_SCORE_L2_CLIP_METERS = 4.0
_TASK_SCORE_COLLISION_WEIGHT = 4.0
_TASK_SCORE_WEIGHT_SUM = 5.0


@dataclass(frozen=True)
class World4DriveSampleMetrics:
    """Serializable metrics for one selected Planner trajectory."""

    selected_mode: int
    selected_trajectory: tuple[tuple[float, float, float], ...]
    l2_per_step: tuple[float, ...]
    collision_counts: tuple[int, ...]
    gt_collision_counts: tuple[int, ...]
    l2_at_1s: float
    l2_at_2s: float
    l2_at_3s: float
    l2_avg: float
    collision_at_1s: float
    collision_at_2s: float
    collision_at_3s: float
    collision_rate: float


def _require_finite_real(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return normalized


def compute_canonical_task_score(l2_avg: object, collision_avg: object) -> float:
    """Compute the World4Drive task score for one window at one rollout horizon.

    Parameters
    ----------
    l2_avg       : non-negative World4Drive L2 average in meters
    collision_avg: canonical World4Drive collision fraction in ``[0, 1]``

    Returns
    -------
    float:
        Canonical score in ``[0, 1]``.
    """

    l2_value = _require_finite_real("l2_avg", l2_avg)
    collision_value = _require_finite_real("collision_avg", collision_avg)
    if l2_value < 0.0:
        raise ValueError(f"l2_avg must be non-negative meters, got {l2_avg!r}")
    if not 0.0 <= collision_value <= 1.0:
        raise ValueError(f"collision_avg must be a fraction in [0, 1], got {collision_avg!r}")
    clipped_l2_risk = min(l2_value / _TASK_SCORE_L2_CLIP_METERS, 1.0)
    return 1.0 - (clipped_l2_risk + _TASK_SCORE_COLLISION_WEIGHT * collision_value) / _TASK_SCORE_WEIGHT_SUM


def _require_metric_sequence(name: str, values: object) -> Sequence[object]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a finite metric sequence, got {type(values).__name__}")
    return values


def compute_canonical_task_scores(l2_avg: object, collision_avg: object) -> tuple[float, ...]:
    """Compute World4Drive scores independently for aligned window/horizon records."""

    l2_values = _require_metric_sequence("l2_avg", l2_avg)
    collision_values = _require_metric_sequence("collision_avg", collision_avg)
    if len(l2_values) != len(collision_values):
        raise ValueError(
            "l2_avg and collision_avg must have the same length, " f"got {len(l2_values)} and {len(collision_values)}"
        )
    if not l2_values:
        raise ValueError("l2_avg and collision_avg must be non-empty")
    return tuple(
        compute_canonical_task_score(l2_value, collision_value)
        for l2_value, collision_value in zip(l2_values, collision_values)
    )


def mean_canonical_task_score(l2_avg: object, collision_avg: object) -> float:
    """Average per-record World4Drive scores without aggregating metrics first."""

    scores = compute_canonical_task_scores(l2_avg, collision_avg)
    return math.fsum(scores) / len(scores)


def _require_tensor(name: str, value: object, *, ndim: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got shape {tuple(value.shape)}")
    return value


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def select_deployment_trajectory(
    trajectories: torch.Tensor,
    confidences: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select exactly the Planner mode deployed by confidence argmax.

    Parameters
    ----------
    trajectories : [B, K, T, 3]
    confidences  : [B, K]

    Returns
    -------
    selected_trajectories : [B, T, 3]
    selected_modes        : [B]
    """

    trajectories = _require_tensor("trajectories", trajectories, ndim=4)
    confidences = _require_tensor("confidences", confidences, ndim=2)
    if trajectories.shape[-1] != 3:
        raise ValueError(f"trajectories must have shape [B,K,T,3], got {tuple(trajectories.shape)}")
    if trajectories.shape[:2] != confidences.shape:
        raise ValueError(
            "confidences must match trajectories [B,K], "
            f"got trajectories={tuple(trajectories.shape)} confidences={tuple(confidences.shape)}"
        )
    if trajectories.shape[0] == 0 or trajectories.shape[1] == 0:
        raise ValueError(f"trajectories must contain at least one batch and mode, got {tuple(trajectories.shape)}")
    _require_finite_tensor("trajectories", trajectories)
    _require_finite_tensor("confidences", confidences)
    selected_modes = confidences.argmax(dim=1)
    batch_indices = torch.arange(trajectories.shape[0], device=trajectories.device)
    return trajectories[batch_indices, selected_modes], selected_modes


def _required_three_second_steps(timestep_sec: object) -> int:
    if isinstance(timestep_sec, bool) or not isinstance(timestep_sec, Real):
        raise ValueError(f"timestep_sec must be a finite positive real number, got {timestep_sec!r}")
    normalized = float(timestep_sec)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"timestep_sec must be a finite positive real number, got {timestep_sec!r}")
    exact_steps = 3.0 / normalized
    rounded_steps = int(round(exact_steps))
    if rounded_steps <= 0 or not math.isclose(exact_steps, rounded_steps, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"timestep_sec={normalized} does not exactly divide the 3-second reporting horizon")
    return rounded_steps


def _require_evaluation_shapes(
    *,
    selected: torch.Tensor,
    gt_trajectory: torch.Tensor,
    agent_seg: torch.Tensor,
    ego_poses: torch.Tensor,
    required_steps: int,
    future_start_idx: int,
    reference_frame_idx: int,
) -> None:
    if selected.shape[0] != 1:
        raise ValueError(f"World4Drive cache evaluation requires batch size 1, got {selected.shape[0]}")
    if gt_trajectory.shape[0] != 1 or agent_seg.shape[0] != 1 or ego_poses.shape[0] != 1:
        raise ValueError("World4Drive cache evaluation requires batch size 1 for every input")
    if gt_trajectory.shape[-1] != 3:
        raise ValueError(f"gt_trajectory must have shape [1,T,3], got {tuple(gt_trajectory.shape)}")
    if agent_seg.shape[-2:] != (BEV_SIZE, BEV_SIZE):
        raise ValueError(f"agent_seg must have shape [1,T,{BEV_SIZE},{BEV_SIZE}], got {tuple(agent_seg.shape)}")
    if ego_poses.shape[-1] != 7:
        raise ValueError(f"ego_poses must have shape [1,T,7], got {tuple(ego_poses.shape)}")
    temporal_lengths = {
        "trajectories": selected.shape[1],
        "gt_trajectory": gt_trajectory.shape[1],
        "agent_seg": agent_seg.shape[1],
    }
    too_short = {name: length for name, length in temporal_lengths.items() if length < required_steps}
    if too_short:
        raise ValueError(
            f"inputs do not cover the complete 3-second reporting horizon ({required_steps} steps): {too_short}"
        )
    if type(future_start_idx) is not int or future_start_idx < 0:
        raise ValueError(f"future_start_idx must be a non-negative integer, got {future_start_idx!r}")
    if type(reference_frame_idx) is not int or reference_frame_idx < 0:
        raise ValueError(f"reference_frame_idx must be a non-negative integer, got {reference_frame_idx!r}")
    if reference_frame_idx >= ego_poses.shape[1]:
        raise ValueError(f"reference_frame_idx={reference_frame_idx} is outside ego_poses length {ego_poses.shape[1]}")
    if future_start_idx + required_steps > ego_poses.shape[1]:
        raise ValueError(
            "ego_poses does not cover the complete 3-second reporting horizon: "
            f"future_start_idx={future_start_idx}, required_steps={required_steps}, length={ego_poses.shape[1]}"
        )


def evaluate_world4drive_sample(
    *,
    dataset_domain: str,
    future_agent_geometry_valid: bool,
    trajectories: torch.Tensor,
    confidences: torch.Tensor,
    gt_trajectory: torch.Tensor,
    agent_seg: torch.Tensor,
    ego_poses: torch.Tensor,
    future_start_idx: int,
    reference_frame_idx: int,
    timestep_sec: float,
) -> World4DriveSampleMetrics:
    """Evaluate one Real sample using the canonical World4Drive functions."""

    if dataset_domain != "real":
        raise ValueError(f"World4Drive evaluation is Real-only, got dataset_domain={dataset_domain!r}")
    if future_agent_geometry_valid is not True:
        raise ValueError("World4Drive evaluation requires future_agent_geometry_valid=True")
    selected, selected_modes = select_deployment_trajectory(trajectories, confidences)
    gt_trajectory = _require_tensor("gt_trajectory", gt_trajectory, ndim=3)
    agent_seg = _require_tensor("agent_seg", agent_seg, ndim=4)
    ego_poses = _require_tensor("ego_poses", ego_poses, ndim=3)
    _require_finite_tensor("gt_trajectory", gt_trajectory)
    _require_finite_tensor("ego_poses", ego_poses)
    required_steps = _required_three_second_steps(timestep_sec)
    _require_evaluation_shapes(
        selected=selected,
        gt_trajectory=gt_trajectory,
        agent_seg=agent_seg,
        ego_poses=ego_poses,
        required_steps=required_steps,
        future_start_idx=future_start_idx,
        reference_frame_idx=reference_frame_idx,
    )
    selected = selected[:, :required_steps]
    gt_trajectory = gt_trajectory[:, :required_steps]
    agent_seg = agent_seg[:, :required_steps]
    l2_metrics = compute_world4drive_l2_metrics(selected, gt_trajectory, timestep_sec=float(timestep_sec))
    collision_metrics = compute_collision_rate(
        selected.detach().cpu().numpy(),
        gt_trajectory.detach().cpu().numpy(),
        agent_seg.detach().cpu().numpy(),
        ego_poses.detach().cpu().numpy(),
        future_start_idx=future_start_idx,
        timestep_sec=float(timestep_sec),
        reference_frame_idx=reference_frame_idx,
    )
    selected_cpu = selected[0].detach().cpu().double().tolist()
    return World4DriveSampleMetrics(
        selected_mode=int(selected_modes[0].item()),
        selected_trajectory=tuple(tuple(float(component) for component in point) for point in selected_cpu),
        l2_per_step=tuple(float(value) for value in l2_metrics["l2_per_step"]),
        collision_counts=tuple(int(value) for value in collision_metrics["collision_counts"]),
        gt_collision_counts=tuple(int(value) for value in collision_metrics["gt_collision_counts"]),
        l2_at_1s=float(l2_metrics["l2_at_1s"]),
        l2_at_2s=float(l2_metrics["l2_at_2s"]),
        l2_at_3s=float(l2_metrics["l2_at_3s"]),
        l2_avg=float(l2_metrics["l2_avg"]),
        collision_at_1s=float(collision_metrics["collision_at_1s"]),
        collision_at_2s=float(collision_metrics["collision_at_2s"]),
        collision_at_3s=float(collision_metrics["collision_at_3s"]),
        collision_rate=float(collision_metrics["collision_rate"]),
    )
