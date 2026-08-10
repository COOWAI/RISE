"""Fixed-shape multi-domain planner validation accumulation."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist

from app.vjepa_cowa_world_model.training.trajectory_quality_metrics import TRAJECTORY_QUALITY_METRIC_KEYS
from app.vjepa_cowa_world_model.training.validation_distributed import distributed_validation_active
from app.vjepa_cowa_world_model.utils.metrics import (
    populate_point_l2_horizons,
    populate_world4drive_collision_horizons,
    populate_world4drive_l2_horizons,
)

_COHORT_KEYS = ("real/all", "counterfactual/all", "counterfactual/safe", "counterfactual/hazard")
_METRIC_KEYS = ("ade", "fde", "minade_k", "minfde_k")


def _metadata_bool_vector(
    metadata: Mapping[str, object],
    key: str,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if key not in metadata:
        raise KeyError(f"validation cohort metadata is missing {key!r}")
    value = metadata[key]
    try:
        tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"validation cohort metadata {key!r} must be a boolean vector") from exc
    if tensor.shape != (batch_size,):
        raise ValueError(
            f"validation cohort metadata {key!r} must have shape [{batch_size}], got {tuple(tensor.shape)}"
        )
    return tensor


def build_validation_cohort_masks(
    metadata: Mapping[str, object],
    *,
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build strict per-sample domain/cohort and collision-valid masks."""

    if not isinstance(metadata, Mapping):
        raise TypeError("validation cohort metadata must be a mapping")
    raw_domains = metadata.get("dataset_domain")
    if not isinstance(raw_domains, (list, tuple)) or len(raw_domains) != batch_size:
        raise ValueError(
            "validation cohort metadata 'dataset_domain' must be a string list with " f"{batch_size} entries"
        )
    domains = tuple(str(value) for value in raw_domains)
    invalid_domains = sorted(set(domains) - {"real", "counterfactual"})
    if invalid_domains:
        raise ValueError(f"validation cohort metadata contains invalid dataset_domain values: {invalid_domains}")

    real = torch.tensor([value == "real" for value in domains], dtype=torch.bool, device=device)
    counterfactual = ~real
    hazard = _metadata_bool_vector(metadata, "cf_is_hazard", batch_size=batch_size, device=device)
    geometry_valid = _metadata_bool_vector(
        metadata,
        "future_agent_geometry_valid",
        batch_size=batch_size,
        device=device,
    )
    if bool(torch.any(counterfactual & geometry_valid).item()):
        raise ValueError("counterfactual samples must never set future_agent_geometry_valid=true")
    if bool(torch.any(real & hazard).item()):
        raise ValueError("real validation samples must never set cf_is_hazard=true")

    return {
        "real/all": real,
        "counterfactual/all": counterfactual,
        "counterfactual/safe": counterfactual & ~hazard,
        "counterfactual/hazard": counterfactual & hazard,
        "real/collision_valid": real & geometry_valid,
    }


def select_real_collision_inputs(
    *,
    metadata: Mapping[str, object],
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    segmentation: np.ndarray,
    ego_poses: torch.Tensor,
) -> Dict[str, object]:
    """Slice collision inputs before CPU conversion so invalid/CF samples cannot contribute."""

    if pred_traj.ndim != 3 or gt_traj.shape != pred_traj.shape:
        raise ValueError(
            f"collision pred/GT trajectories must share shape [B, T, D], got {pred_traj.shape}/{gt_traj.shape}"
        )
    batch_size = int(pred_traj.shape[0])
    if ego_poses.ndim != 3 or ego_poses.shape[0] != batch_size:
        raise ValueError(f"collision ego_poses must have batch size {batch_size}, got {ego_poses.shape}")
    segmentation = np.asarray(segmentation)
    if segmentation.ndim != 4 or segmentation.shape[0] != batch_size:
        raise ValueError(
            f"collision segmentation must have shape [B, T, H, W] with B={batch_size}, got {segmentation.shape}"
        )
    masks = build_validation_cohort_masks(metadata, batch_size=batch_size, device=pred_traj.device)
    eligible = masks["real/collision_valid"]
    indices = torch.nonzero(eligible, as_tuple=False).flatten()
    numpy_indices = indices.detach().cpu().numpy()
    return {
        "pred_traj": pred_traj.index_select(0, indices).detach().cpu().numpy(),
        "gt_traj": gt_traj.index_select(0, indices).detach().cpu().numpy(),
        "segmentation": segmentation[numpy_indices],
        "ego_poses": ego_poses.index_select(0, indices).detach().cpu().numpy(),
        "sample_count": int(indices.numel()),
    }


class ValidationCohortAccumulator:
    """Accumulate all validation slices and reduce them in one end collective."""

    def __init__(
        self,
        *,
        num_steps: int,
        timestep_sec: float,
        device: torch.device,
        world_size: int,
    ) -> None:
        self.num_steps = int(num_steps)
        self.timestep_sec = float(timestep_sec)
        self.device = device
        self.world_size = int(world_size)
        if self.num_steps < 1:
            raise ValueError(f"validation cohort num_steps must be positive, got {num_steps}")
        if self.timestep_sec <= 0.0:
            raise ValueError(f"validation cohort timestep_sec must be positive, got {timestep_sec}")

        num_cohorts = len(_COHORT_KEYS)
        self._metric_sums = torch.zeros(num_cohorts, len(_METRIC_KEYS), dtype=torch.float64, device=device)
        self._sample_counts = torch.zeros(num_cohorts, dtype=torch.float64, device=device)
        self._l2_sums = torch.zeros(num_cohorts, self.num_steps, dtype=torch.float64, device=device)
        self._quality_sums = torch.zeros(
            num_cohorts,
            len(TRAJECTORY_QUALITY_METRIC_KEYS),
            dtype=torch.float64,
            device=device,
        )
        self._box_collision_sums = torch.zeros(self.num_steps, dtype=torch.float64, device=device)
        self._point_collision_sums = torch.zeros(self.num_steps, dtype=torch.float64, device=device)
        self._gt_collision_sums = torch.zeros(self.num_steps, dtype=torch.float64, device=device)
        self._collision_sample_count = torch.zeros((), dtype=torch.float64, device=device)

    def add_batch(
        self,
        *,
        metadata: Mapping[str, object],
        ade: torch.Tensor,
        fde: torch.Tensor,
        minade_k: torch.Tensor,
        minfde_k: torch.Tensor,
        l2_per_step: torch.Tensor,
        quality_metrics: Mapping[str, torch.Tensor],
    ) -> None:
        """Add per-sample planner errors without any batch-time collective."""

        metric_values = (ade, fde, minade_k, minfde_k)
        if not all(torch.is_tensor(value) for value in metric_values):
            raise TypeError("validation cohort ADE/FDE metrics must be tensors")
        batch_size = int(ade.shape[0]) if ade.ndim == 1 else -1
        if batch_size < 0 or any(value.shape != (batch_size,) for value in metric_values):
            raise ValueError(
                "validation cohort ADE/FDE metrics must all have shape [B], got "
                f"{[tuple(value.shape) for value in metric_values]}"
            )
        if not torch.is_tensor(l2_per_step) or l2_per_step.shape != (batch_size, self.num_steps):
            shape = tuple(l2_per_step.shape) if torch.is_tensor(l2_per_step) else type(l2_per_step).__name__
            raise ValueError(
                f"validation cohort l2_per_step must have shape [{batch_size}, {self.num_steps}], got {shape}"
            )
        if set(quality_metrics) != set(TRAJECTORY_QUALITY_METRIC_KEYS):
            raise ValueError(
                "validation trajectory quality metric keys must exactly match schema: "
                f"expected={list(TRAJECTORY_QUALITY_METRIC_KEYS)}, got={sorted(quality_metrics)}"
            )
        quality_values = []
        for key in TRAJECTORY_QUALITY_METRIC_KEYS:
            value = quality_metrics[key]
            if not torch.is_tensor(value) or value.shape != (batch_size,):
                actual = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
                raise ValueError(
                    f"validation trajectory quality metric {key} must have shape [{batch_size}], got {actual}"
                )
            quality_values.append(value.to(self.device, dtype=torch.float64))
        masks = build_validation_cohort_masks(metadata, batch_size=batch_size, device=self.device)
        stacked_metrics = torch.stack([value.to(self.device, dtype=torch.float64) for value in metric_values], dim=1)
        l2_values = l2_per_step.to(self.device, dtype=torch.float64)
        stacked_quality = torch.stack(quality_values, dim=1)
        for cohort_index, cohort_key in enumerate(_COHORT_KEYS):
            mask = masks[cohort_key]
            mask_float = mask.to(dtype=torch.float64)
            self._metric_sums[cohort_index] += (stacked_metrics * mask_float[:, None]).sum(dim=0)
            self._l2_sums[cohort_index] += (l2_values * mask_float[:, None]).sum(dim=0)
            self._quality_sums[cohort_index] += (stacked_quality * mask_float[:, None]).sum(dim=0)
            self._sample_counts[cohort_index] += mask_float.sum()

    def add_real_collision_counts(
        self,
        *,
        metadata: Mapping[str, object],
        box_counts: Sequence[float],
        point_counts: Sequence[float],
        gt_counts: Sequence[float],
    ) -> None:
        """Add raw collision totals computed only on geometry-valid real samples."""

        raw_domains = metadata.get("dataset_domain") if isinstance(metadata, Mapping) else None
        if not isinstance(raw_domains, (list, tuple)):
            raise ValueError("collision cohort metadata requires dataset_domain")
        masks = build_validation_cohort_masks(metadata, batch_size=len(raw_domains), device=self.device)
        collision_count = masks["real/collision_valid"].to(dtype=torch.float64).sum()
        vectors = []
        for name, values in (
            ("box_counts", box_counts),
            ("point_counts", point_counts),
            ("gt_counts", gt_counts),
        ):
            vector = torch.as_tensor(values, dtype=torch.float64, device=self.device)
            if vector.shape != (self.num_steps,):
                raise ValueError(f"validation cohort {name} must have shape [{self.num_steps}], got {vector.shape}")
            vectors.append(vector)
        self._box_collision_sums += vectors[0]
        self._point_collision_sums += vectors[1]
        self._gt_collision_sums += vectors[2]
        self._collision_sample_count += collision_count

    def _reduced_payload(self) -> torch.Tensor:
        payload = torch.cat(
            [
                self._metric_sums.flatten(),
                self._sample_counts,
                self._l2_sums.flatten(),
                self._quality_sums.flatten(),
                self._box_collision_sums,
                self._point_collision_sums,
                self._gt_collision_sums,
                self._collision_sample_count.reshape(1),
            ]
        )
        if distributed_validation_active(self.world_size):
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)
        return payload

    def finalize(self, *, require_primary_collision: bool) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Reduce once and return scalar metrics for all required slices."""

        payload = self._reduced_payload().cpu()
        cursor = 0
        num_cohorts = len(_COHORT_KEYS)
        num_metrics = len(_METRIC_KEYS)
        metric_sums = payload[cursor : cursor + num_cohorts * num_metrics].view(num_cohorts, num_metrics)
        cursor += num_cohorts * num_metrics
        sample_counts = payload[cursor : cursor + num_cohorts]
        cursor += num_cohorts
        l2_sums = payload[cursor : cursor + num_cohorts * self.num_steps].view(num_cohorts, self.num_steps)
        cursor += num_cohorts * self.num_steps
        num_quality_metrics = len(TRAJECTORY_QUALITY_METRIC_KEYS)
        quality_sums = payload[cursor : cursor + num_cohorts * num_quality_metrics].view(
            num_cohorts, num_quality_metrics
        )
        cursor += num_cohorts * num_quality_metrics
        box_counts = payload[cursor : cursor + self.num_steps].numpy()
        cursor += self.num_steps
        point_counts = payload[cursor : cursor + self.num_steps].numpy()
        cursor += self.num_steps
        cursor += self.num_steps  # GT collision counts are retained in the fixed payload for diagnostics parity.
        collision_sample_count = float(payload[cursor])

        empty_cohorts = [
            cohort_key for cohort_index, cohort_key in enumerate(_COHORT_KEYS) if sample_counts[cohort_index] <= 0
        ]
        if empty_cohorts:
            raise RuntimeError(f"validation produced zero global samples for required cohorts: {empty_cohorts}")
        if require_primary_collision and collision_sample_count <= 0:
            raise RuntimeError(
                "primary validation slice real/all/full has no future-agent geometry valid for collision metrics"
            )

        result: Dict[str, Dict[str, Dict[str, float]]] = {
            "real": {"all": {}},
            "counterfactual": {"all": {}, "safe": {}, "hazard": {}},
        }
        for cohort_index, cohort_key in enumerate(_COHORT_KEYS):
            domain, cohort = cohort_key.split("/", 1)
            count = float(sample_counts[cohort_index])
            metrics = {
                metric_key: float(metric_sums[cohort_index, metric_index]) / count
                for metric_index, metric_key in enumerate(_METRIC_KEYS)
            }
            l2_per_step = l2_sums[cohort_index].numpy() / count
            populate_world4drive_l2_horizons(metrics, l2_per_step, self.timestep_sec)
            populate_point_l2_horizons(metrics, l2_per_step, self.timestep_sec)
            metrics.update(
                {
                    metric_key: float(quality_sums[cohort_index, metric_index]) / count
                    for metric_index, metric_key in enumerate(TRAJECTORY_QUALITY_METRIC_KEYS)
                }
            )
            result[domain][cohort] = metrics

        if collision_sample_count > 0:
            real_metrics = result["real"]["all"]
            populate_world4drive_collision_horizons(
                real_metrics,
                np.asarray(box_counts, dtype=np.float64),
                total_samples=collision_sample_count,
                timestep_sec=self.timestep_sec,
                metric_prefix="collision",
                avg_key="collision_rate",
            )
            populate_world4drive_collision_horizons(
                real_metrics,
                np.asarray(point_counts, dtype=np.float64),
                total_samples=collision_sample_count,
                timestep_sec=self.timestep_sec,
                metric_prefix="point_collision",
                avg_key="point_collision_rate",
            )
        return result
