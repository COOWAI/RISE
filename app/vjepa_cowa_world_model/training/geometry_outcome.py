"""Real-data geometry-anchored outcomes for deployed planner trajectories."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

import torch

from app.vjepa_cowa_world_model.losses.reward_selector import (
    compute_collision_components,
    compute_comfort_risk,
    compute_trajectory_error,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_metrics import (
    CVOI_WORLD4DRIVE_TASK_SCORE_SCHEMA,
    World4DriveSampleMetrics,
    compute_canonical_task_score,
    evaluate_world4drive_sample,
)
from app.vjepa_cowa_world_model.training.reward_labels import RewardLabelConfig
from app.vjepa_cowa_world_model.utils.metrics import EGO_LENGTH, EGO_LENGTH_OFFSET, EGO_WIDTH


def planning_outcome_evaluator_signature_sha256(signature: Mapping[str, object]) -> str:
    """Return the canonical identity used by Oracle and Gate provenance."""

    if not isinstance(signature, Mapping) or not signature:
        raise ValueError("PlanningOutcomeEvaluator signature must be a non-empty mapping")
    try:
        canonical = json.dumps(signature, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("PlanningOutcomeEvaluator signature must be JSON serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PlanningOutcome:
    """Per-sample outcome of the trajectory deployed by planner confidence."""

    selected_mode: torch.Tensor
    selected_trajectory: torch.Tensor
    collision_any: torch.Tensor
    near_miss_risk: torch.Tensor
    l2_risk: torch.Tensor
    comfort_risk: torch.Tensor
    task_score: torch.Tensor
    l2_avg: Optional[torch.Tensor] = None
    collision_avg: Optional[torch.Tensor] = None
    task_score_schema: Optional[str] = None

    @property
    def collision(self) -> torch.Tensor:
        """Compatibility alias for callers that use the shorter metric name."""

        return self.collision_any

    @property
    def selected_mode_idx(self) -> torch.Tensor:
        """Compatibility alias spelling out that the selected mode is an index."""

        return self.selected_mode


class CanonicalWorld4DrivePlanningOutcomeEvaluator:
    """Formal-v2 Real evaluator backed by canonical World4Drive L2/collision."""

    def __init__(self, *, timestep_sec: float) -> None:
        if isinstance(timestep_sec, bool) or not isinstance(timestep_sec, (int, float)):
            raise ValueError(f"timestep_sec must be a finite positive real number, got {timestep_sec!r}")
        self.timestep_sec = float(timestep_sec)
        if not math.isfinite(self.timestep_sec) or self.timestep_sec <= 0.0:
            raise ValueError(f"timestep_sec must be a finite positive real number, got {timestep_sec!r}")

    @property
    def signature(self) -> dict[str, object]:
        return {
            "schema": "canonical_world4drive_planning_outcome_v2",
            "task_score_schema": CVOI_WORLD4DRIVE_TASK_SCORE_SCHEMA,
            "mode_selection": "deployment_confidence_argmax",
            "l2": "canonical_world4drive_l2",
            "collision": "canonical_world4drive_bev_box_with_gt_mask",
            "l2_clip_m": 4.0,
            "collision_weight": 4.0,
            "denominator": 5.0,
            "timestep_sec": self.timestep_sec,
            "diagnostic_only": ["collision_any", "near_miss_risk", "comfort_risk"],
        }

    def evaluate_metrics(
        self,
        pred_trajs: torch.Tensor,
        confidences: torch.Tensor,
        gt_traj: torch.Tensor,
        agent_boxes: torch.Tensor,
        agent_mask: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> World4DriveSampleMetrics:
        """Return the canonical per-window metrics used by every Formal-v2 artifact."""

        del agent_boxes, agent_mask
        if not isinstance(metadata, Mapping):
            raise ValueError("Formal-v2 World4Drive evaluator metadata must be a mapping")
        domains = metadata.get("dataset_domain")
        if domains != ["real"]:
            raise ValueError("Formal-v2 World4Drive evaluator requires dataset_domain=['real']")
        geometry_valid = metadata.get("future_agent_geometry_valid")
        if not torch.is_tensor(geometry_valid) or geometry_valid.shape != (1,) or geometry_valid.dtype != torch.bool:
            raise ValueError("Formal-v2 World4Drive evaluator requires bool future_agent_geometry_valid [1]")
        agent_seg = metadata.get("world4drive_agent_seg")
        ego_poses = metadata.get("world4drive_ego_poses")
        future_start_idx = metadata.get("world4drive_future_start_idx")
        reference_frame_idx = metadata.get("world4drive_reference_frame_idx")
        return evaluate_world4drive_sample(
            dataset_domain="real",
            future_agent_geometry_valid=bool(geometry_valid[0].item()),
            trajectories=pred_trajs,
            confidences=confidences,
            gt_trajectory=gt_traj,
            agent_seg=agent_seg,
            ego_poses=ego_poses,
            future_start_idx=future_start_idx,
            reference_frame_idx=reference_frame_idx,
            timestep_sec=self.timestep_sec,
        )

    def evaluate(
        self,
        pred_trajs: torch.Tensor,
        confidences: torch.Tensor,
        gt_traj: torch.Tensor,
        agent_boxes: torch.Tensor,
        agent_mask: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> PlanningOutcome:
        """Evaluate one Real sample; box tensors remain diagnostics-only inputs."""

        metrics = self.evaluate_metrics(
            pred_trajs,
            confidences,
            gt_traj,
            agent_boxes,
            agent_mask,
            metadata,
        )
        device = pred_trajs.device
        dtype = pred_trajs.dtype
        l2_avg = torch.tensor([metrics.l2_avg], dtype=dtype, device=device)
        collision_avg = torch.tensor([metrics.collision_rate], dtype=dtype, device=device)
        task_score = torch.tensor(
            [compute_canonical_task_score(metrics.l2_avg, metrics.collision_rate)],
            dtype=dtype,
            device=device,
        )
        selected_mode = torch.tensor([metrics.selected_mode], dtype=torch.long, device=device)
        selected_trajectory = torch.tensor(metrics.selected_trajectory, dtype=dtype, device=device).unsqueeze(0)
        return PlanningOutcome(
            selected_mode=selected_mode,
            selected_trajectory=selected_trajectory,
            collision_any=collision_avg > 0.0,
            near_miss_risk=torch.zeros_like(l2_avg),
            l2_risk=torch.clamp(l2_avg / 4.0, min=0.0, max=1.0),
            comfort_risk=torch.zeros_like(l2_avg),
            task_score=task_score,
            l2_avg=l2_avg,
            collision_avg=collision_avg,
            task_score_schema=CVOI_WORLD4DRIVE_TASK_SCORE_SCHEMA,
        )

    def __call__(
        self,
        pred_trajs: torch.Tensor,
        confidences: torch.Tensor,
        gt_traj: torch.Tensor,
        agent_boxes: torch.Tensor,
        agent_mask: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> PlanningOutcome:
        return self.evaluate(pred_trajs, confidences, gt_traj, agent_boxes, agent_mask, metadata)


@dataclass(frozen=True)
class GuidanceOutcomeComparison:
    """Separate critic self-improvement from real external outcome changes."""

    value_self_delta: torch.Tensor
    value_self_improved: torch.Tensor
    external_task_delta: torch.Tensor
    externally_improved: torch.Tensor


def compare_guidance_outcomes(
    *,
    unguided: PlanningOutcome,
    guided: PlanningOutcome,
    field_value_before: torch.Tensor,
    field_value_after: torch.Tensor,
) -> GuidanceOutcomeComparison:
    """Audit Guidance using external real outcomes, never critic self-score alone."""

    if not isinstance(unguided, PlanningOutcome) or not isinstance(guided, PlanningOutcome):
        raise TypeError("unguided and guided must be PlanningOutcome instances from the real evaluator")
    if unguided.task_score.shape != guided.task_score.shape or unguided.task_score.ndim != 1:
        raise ValueError("unguided and guided task_score must be matching [B] tensors")
    if (
        not torch.is_tensor(field_value_before)
        or not torch.is_tensor(field_value_after)
        or field_value_before.shape != unguided.task_score.shape
        or field_value_after.shape != unguided.task_score.shape
    ):
        raise ValueError("field_value_before/after must match external task_score shape [B]")
    tensors = (
        unguided.task_score,
        guided.task_score,
        field_value_before,
        field_value_after,
    )
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("Guidance outcome comparison requires finite values")
    value_self_delta = field_value_after - field_value_before
    external_task_delta = guided.task_score - unguided.task_score
    return GuidanceOutcomeComparison(
        value_self_delta=value_self_delta,
        value_self_improved=value_self_delta > 0.0,
        external_task_delta=external_task_delta,
        externally_improved=external_task_delta > 0.0,
    )


def _require_finite_float_tensor(name: str, value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype, got {value.dtype}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_metadata(metadata: Mapping[str, Any], batch_size: int, num_poses: int) -> torch.Tensor:
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")

    domains = metadata.get("dataset_domain")
    if not isinstance(domains, (list, tuple)):
        raise TypeError("metadata['dataset_domain'] must be a list/tuple with one entry per sample")
    if len(domains) != batch_size:
        raise ValueError(
            "metadata['dataset_domain'] must have one entry per sample; " f"expected {batch_size}, got {len(domains)}"
        )
    invalid_domains = [domain for domain in domains if domain != "real"]
    if invalid_domains:
        raise ValueError(
            "PlanningOutcomeEvaluator is real-only; " f"metadata['dataset_domain'] contains {invalid_domains!r}"
        )

    bool_metadata = {}
    for name in ("geometry_present", "future_agent_geometry_valid"):
        value = metadata.get(name)
        if not torch.is_tensor(value):
            raise TypeError(f"metadata[{name!r}] must be a bool tensor [B]")
        if value.dtype != torch.bool:
            raise TypeError(f"metadata[{name!r}] must use dtype torch.bool, got {value.dtype}")
        if tuple(value.shape) != (batch_size,):
            raise ValueError(f"metadata[{name!r}] must have shape [{batch_size}], got {tuple(value.shape)}")
        bool_metadata[name] = value

    geometry_truncated_raw = metadata.get("agent_geometry_truncated")
    if torch.is_tensor(geometry_truncated_raw):
        if geometry_truncated_raw.dtype != torch.bool:
            raise TypeError(
                "metadata['agent_geometry_truncated'] must use dtype torch.bool, "
                f"got {geometry_truncated_raw.dtype}"
            )
        if tuple(geometry_truncated_raw.shape) != (batch_size,):
            raise ValueError(
                "metadata['agent_geometry_truncated'] must have shape "
                f"[{batch_size}], got {tuple(geometry_truncated_raw.shape)}"
            )
        geometry_truncated = geometry_truncated_raw
    elif isinstance(geometry_truncated_raw, (list, tuple)):
        if len(geometry_truncated_raw) != batch_size or any(
            type(value) is not bool for value in geometry_truncated_raw
        ):
            raise ValueError("metadata['agent_geometry_truncated'] must contain one explicit bool per real sample")
        geometry_truncated = torch.as_tensor(geometry_truncated_raw, dtype=torch.bool)
    else:
        raise TypeError("metadata['agent_geometry_truncated'] must be a bool tensor or bool sequence [B]")

    geometry_present = bool_metadata["geometry_present"]
    geometry_valid = bool_metadata["future_agent_geometry_valid"]
    if not bool(geometry_present.all().item()):
        raise ValueError("PlanningOutcomeEvaluator requires geometry_present=true for every sample")
    if not bool(geometry_valid.all().item()):
        raise ValueError("PlanningOutcomeEvaluator requires valid future agent geometry for every sample")
    if bool(geometry_truncated.any().item()):
        raise ValueError("PlanningOutcomeEvaluator rejects truncated agent geometry")
    expected_semantics = {
        "geometry_source": "logged_nuscenes_gt",
        "geometry_coordinate_frame": "per_frame_ego",
    }
    for name, expected in expected_semantics.items():
        value = metadata.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != batch_size or any(item != expected for item in value):
            raise ValueError(f"PlanningOutcomeEvaluator requires metadata[{name!r}]={expected!r}")

    raw_count_value = metadata.get("raw_agent_count")
    if torch.is_tensor(raw_count_value):
        raw_agent_count = raw_count_value
    elif isinstance(raw_count_value, (list, tuple)) and len(raw_count_value) == batch_size:
        if any(not torch.is_tensor(value) for value in raw_count_value):
            raise TypeError("metadata['raw_agent_count'] sequence must contain tensors")
        raw_agent_count = torch.stack(list(raw_count_value), dim=0)
    else:
        raise TypeError("metadata['raw_agent_count'] must be an integer tensor [B, P] or tensor sequence")
    if raw_agent_count.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("metadata['raw_agent_count'] must use an integer dtype")
    if tuple(raw_agent_count.shape) != (batch_size, num_poses):
        raise ValueError(
            "metadata['raw_agent_count'] must align with future geometry as "
            f"[{batch_size}, {num_poses}], got {tuple(raw_agent_count.shape)}"
        )
    if bool(((raw_agent_count < 0) | (raw_agent_count > 256)).any().item()):
        raise ValueError("metadata['raw_agent_count'] entries must be in [0, 256]")
    return raw_agent_count.to(device="cpu", dtype=torch.long)


class PlanningOutcomeEvaluator:
    """Evaluate confidence-selected planner modes against real future geometry."""

    def __init__(
        self,
        *,
        timestep_sec: float = 0.5,
        config: RewardLabelConfig | None = None,
    ) -> None:
        timestep_sec = float(timestep_sec)
        if not math.isfinite(timestep_sec) or timestep_sec <= 0.0:
            raise ValueError(f"timestep_sec must be finite and positive, got {timestep_sec}")
        if config is not None and not isinstance(config, RewardLabelConfig):
            raise TypeError(f"config must be RewardLabelConfig or None, got {type(config).__name__}")
        if config is not None and not math.isclose(
            float(config.timestep_sec),
            timestep_sec,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "PlanningOutcomeEvaluator timestep_sec must match RewardLabelConfig.timestep_sec; "
                f"got {timestep_sec} and {config.timestep_sec}"
            )
        self.timestep_sec = timestep_sec
        self.config = config or RewardLabelConfig(timestep_sec=timestep_sec)

    @property
    def signature(self) -> dict[str, object]:
        """Return the full checkpoint identity of this real-only evaluator."""

        return {
            "schema": "planning_outcome_real_geometry_v1",
            "geometry_source": "logged_nuscenes_gt",
            "geometry_coordinate_frame": "per_frame_ego",
            "agent_transport": {
                "capacity": 256,
                "raw_count_check": "exact_agent_mask_match",
            },
            "mode_selection": "deployment_confidence_argmax",
            "collision_algorithm": "exact_obb_sat_v1",
            "near_miss_clearance_algorithm": "exact_obb_euclidean_clearance_v1",
            "ego_box": {
                "length": float(EGO_LENGTH),
                "width": float(EGO_WIDTH),
                "length_offset": float(EGO_LENGTH_OFFSET),
            },
            "task_score_weights": {
                "collision": 4.0,
                "near_miss": 1.0,
                "l2": 1.0,
                "comfort": 0.2,
                "denominator": 6.2,
            },
            "l2_normalization_m": 4.0,
            "timestep_sec": self.timestep_sec,
            "reward_label_config": asdict(self.config),
        }

    def evaluate(
        self,
        pred_trajs: torch.Tensor,
        confidences: torch.Tensor,
        gt_traj: torch.Tensor,
        agent_boxes: torch.Tensor,
        agent_mask: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> PlanningOutcome:
        """Evaluate the confidence-selected deployment trajectory.

        Parameters
        ----------
        pred_trajs : [B, K, P, 3]
            Candidate ego trajectories.
        confidences : [B, K]
            Either confidence logits or probabilities. Only ``argmax`` is used.
        gt_traj : [B, P, 3]
            Logged real ego future used for geometry transforms and ADE.
        agent_boxes : [B, P, A, 7]
            Complete logged future-agent boxes; the agent axis is not truncated.
        agent_mask : [B, P, A]
            Valid-agent mask.
        metadata : mapping
            Must contain real-only ``dataset_domain`` entries and a true bool
            ``future_agent_geometry_valid`` vector. ``raw_agent_count [B, P]``
            must exactly match the valid-agent mask at every future frame.
        """

        pred_trajs = _require_finite_float_tensor("pred_trajs", pred_trajs)
        confidences = _require_finite_float_tensor("confidences", confidences)
        gt_traj = _require_finite_float_tensor("gt_traj", gt_traj)
        agent_boxes = _require_finite_float_tensor("agent_boxes", agent_boxes)

        if pred_trajs.ndim != 4 or pred_trajs.shape[-1] != 3:
            raise ValueError(f"pred_trajs must have shape [B, K, P, 3], got {tuple(pred_trajs.shape)}")
        batch_size, num_modes, num_poses, _ = pred_trajs.shape
        if batch_size < 1 or num_modes < 1 or num_poses < 1:
            raise ValueError(
                "pred_trajs batch, mode, and pose dimensions must be non-empty, " f"got {tuple(pred_trajs.shape)}"
            )
        if tuple(confidences.shape) != (batch_size, num_modes):
            raise ValueError(
                f"confidences must have shape [B, K]={tuple(pred_trajs.shape[:2])}, " f"got {tuple(confidences.shape)}"
            )
        if tuple(gt_traj.shape) != (batch_size, num_poses, 3):
            raise ValueError(
                f"gt_traj must have shape [B, P, 3]=({batch_size}, {num_poses}, 3), " f"got {tuple(gt_traj.shape)}"
            )
        if agent_boxes.ndim != 4 or agent_boxes.shape[-1] != 7:
            raise ValueError(f"agent_boxes must have shape [B, P, A, 7], got {tuple(agent_boxes.shape)}")
        if agent_boxes.shape[0] != batch_size or agent_boxes.shape[1] != num_poses:
            raise ValueError(
                "agent_boxes must align with pred_trajs on batch and pose dimensions; "
                f"got pred_trajs={tuple(pred_trajs.shape)}, agent_boxes={tuple(agent_boxes.shape)}"
            )
        if agent_boxes.shape[2] != 256:
            raise ValueError(
                "PlanningOutcomeEvaluator requires the complete max_agents=256 transport axis, "
                f"got {agent_boxes.shape[2]}"
            )
        if not torch.is_tensor(agent_mask):
            raise TypeError(f"agent_mask must be a torch.Tensor, got {type(agent_mask).__name__}")
        if agent_mask.dtype != torch.bool:
            raise TypeError(f"agent_mask must use dtype torch.bool, got {agent_mask.dtype}")
        if tuple(agent_mask.shape) != tuple(agent_boxes.shape[:3]):
            raise ValueError(
                f"agent_mask must have shape {tuple(agent_boxes.shape[:3])}, got {tuple(agent_mask.shape)}"
            )
        valid_agent_boxes = agent_boxes[agent_mask]
        if valid_agent_boxes.numel() and torch.any(valid_agent_boxes[:, 3:6] <= 0.0):
            raise ValueError("masked real agent boxes must have positive dimensions")

        tensors = {
            "confidences": confidences,
            "gt_traj": gt_traj,
            "agent_boxes": agent_boxes,
            "agent_mask": agent_mask,
        }
        device_mismatches = {
            name: value.device for name, value in tensors.items() if value.device != pred_trajs.device
        }
        if device_mismatches:
            raise ValueError(f"all tensors must be on pred_trajs device {pred_trajs.device}, got {device_mismatches}")
        raw_agent_count = _validate_metadata(metadata, batch_size, num_poses)
        mask_count = agent_mask.sum(dim=-1).to(device="cpu", dtype=torch.long)
        if not torch.equal(raw_agent_count, mask_count):
            raise ValueError("metadata['raw_agent_count'] must exactly match agent_mask counts for every future frame")

        selected_mode = confidences.argmax(dim=1)
        batch_indices = torch.arange(batch_size, device=pred_trajs.device)
        selected_trajectory = pred_trajs[batch_indices, selected_mode]

        collision_modes, near_miss_modes = compute_collision_components(
            pred_trajs,
            gt_traj,
            agent_boxes,
            agent_mask,
            config=self.config,
        )
        l2_modes = torch.clamp(compute_trajectory_error(pred_trajs, gt_traj) / 4.0, min=0.0, max=1.0)
        comfort_modes = compute_comfort_risk(
            pred_trajs,
            timestep_sec=self.timestep_sec,
            config=self.config,
        )

        collision_any = collision_modes[batch_indices, selected_mode]
        near_miss_risk = near_miss_modes[batch_indices, selected_mode]
        l2_risk = l2_modes[batch_indices, selected_mode]
        comfort_risk = comfort_modes[batch_indices, selected_mode]
        task_score = (
            1.0
            - (4.0 * collision_any.to(dtype=pred_trajs.dtype) + near_miss_risk + l2_risk + 0.2 * comfort_risk) / 6.2
        )

        outputs = {
            "near_miss_risk": near_miss_risk,
            "l2_risk": l2_risk,
            "comfort_risk": comfort_risk,
            "task_score": task_score,
        }
        invalid_outputs = [name for name, value in outputs.items() if not bool(torch.isfinite(value).all().item())]
        if invalid_outputs:
            raise ValueError(f"planning outcome contains non-finite values: {invalid_outputs}")
        out_of_range = [
            name for name, value in outputs.items() if not bool(((value >= 0.0) & (value <= 1.0)).all().item())
        ]
        if out_of_range:
            raise ValueError(f"planning outcome risks/scores must be in [0, 1]: {out_of_range}")

        return PlanningOutcome(
            selected_mode=selected_mode,
            selected_trajectory=selected_trajectory,
            collision_any=collision_any,
            near_miss_risk=near_miss_risk,
            l2_risk=l2_risk,
            comfort_risk=comfort_risk,
            task_score=task_score,
        )

    def __call__(
        self,
        pred_trajs: torch.Tensor,
        confidences: torch.Tensor,
        gt_traj: torch.Tensor,
        agent_boxes: torch.Tensor,
        agent_mask: torch.Tensor,
        metadata: Mapping[str, Any],
    ) -> PlanningOutcome:
        return self.evaluate(pred_trajs, confidences, gt_traj, agent_boxes, agent_mask, metadata)
