"""Retained NavSim boundary for direct World4Drive real-geometry scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch

from app.vjepa_cowa_world_model.training.cvoi_execution import cvoi_execution_dtype_signature
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (  # noqa: F401
    CvoiDeploymentOutput,
    PreparedNavSimCvoiDeployment,
)
from app.vjepa_cowa_world_model.training.geometry_outcome import PlanningOutcome, PlanningOutcomeEvaluator
from app.vjepa_cowa_world_model.utils.status_features import build_future_gt_trajectory_from_states

_RUNTIME_ENVIRONMENT_SCHEMA = "cvoi_evaluation_runtime_environment_v1"
_LATENCY_MEASUREMENT_POLICY = {
    "sequential_gate": {
        "measurement_kind": "actual_short_circuit_adaptive_region_wall_time",
        "clock": "time.perf_counter",
        "synchronization": "cuda_synchronize_before_and_after_adaptive_region",
        "includes": ["predictor_rollout", "value", "gate", "guidance"],
        "excludes": ["encoder", "planner"],
    },
    "fixed_horizon_baselines": {
        "measurement_kind": "component_replay",
        "predictor": "cumulative_step_timings_from_one_full_horizon_rollout",
        "guidance": "per_horizon_guidance_operation_timing_for_p1_only",
        "synchronization": "cuda_synchronize_before_and_after_each_timed_component",
        "excludes": ["encoder", "planner"],
    },
    "warmup_samples": 0,
    "repetitions_per_sample_point": 1,
    "aggregation": "arithmetic_mean_over_evaluation_records",
}


def _json_copy(value: object) -> object:
    try:
        return json.loads(json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("CVoI evaluation runtime environment must be finite and JSON serializable") from exc


@dataclass(frozen=True)
class CvoiEvaluationRuntimeEnvironment:
    """Hardware, software, precision, and timing policy for one retained latency run."""

    device_type: str
    device_index: int | None
    gpu_model: str | None
    gpu_compute_capability: str | None
    torch_version: str
    cuda_runtime_version: str | None
    cudnn_version: int | None
    precision_contract: Mapping[str, object]
    test_only: bool = False

    def __post_init__(self) -> None:
        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError(f"unsupported CVoI evaluation device type: {self.device_type!r}")
        if not isinstance(self.test_only, bool):
            raise TypeError("test_only must be bool")
        if not isinstance(self.torch_version, str) or not self.torch_version.strip():
            raise ValueError("torch_version must be a non-empty string")
        precision_contract = _json_copy(self.precision_contract)
        if not isinstance(precision_contract, dict) or precision_contract.get("required_device") != "cuda":
            raise ValueError("precision_contract must be the signed CUDA CVoI execution contract")
        object.__setattr__(self, "precision_contract", precision_contract)

        if self.device_type == "cpu":
            if not self.test_only:
                raise RuntimeError("CPU CVoI evaluation runtime environments are test-only")
            if any(
                value is not None
                for value in (
                    self.device_index,
                    self.gpu_model,
                    self.gpu_compute_capability,
                    self.cuda_runtime_version,
                    self.cudnn_version,
                )
            ):
                raise ValueError("CPU test runtime environment cannot claim CUDA hardware or software")
            return

        if self.test_only:
            raise ValueError("CUDA CVoI evaluation runtime environment cannot be marked test-only")
        if isinstance(self.device_index, bool) or not isinstance(self.device_index, int) or self.device_index < 0:
            raise ValueError("CUDA CVoI evaluation requires a non-negative device_index")
        for name, value in (
            ("gpu_model", self.gpu_model),
            ("gpu_compute_capability", self.gpu_compute_capability),
            ("cuda_runtime_version", self.cuda_runtime_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"CUDA CVoI evaluation requires non-empty {name}")
        if isinstance(self.cudnn_version, bool) or not isinstance(self.cudnn_version, int) or self.cudnn_version <= 0:
            raise ValueError("CUDA CVoI evaluation requires a positive cuDNN version")

    def to_manifest(self) -> dict[str, object]:
        """Return a detached JSON-ready representation for latency identity."""

        return {
            "schema": _RUNTIME_ENVIRONMENT_SCHEMA,
            "hardware": {
                "device_type": self.device_type,
                "device_index": self.device_index,
                "gpu_model": self.gpu_model,
                "gpu_compute_capability": self.gpu_compute_capability,
                "test_only": self.test_only,
            },
            "software": {
                "torch_version": self.torch_version,
                "cuda_runtime_version": self.cuda_runtime_version,
                "cudnn_version": self.cudnn_version,
            },
            "precision_contract": _json_copy(self.precision_contract),
            "latency_measurement_policy": dict(_LATENCY_MEASUREMENT_POLICY),
        }


def build_cvoi_evaluation_runtime_environment(
    config: Any,
    *,
    device: torch.device | str,
    _allow_cpu_for_tests: bool = False,
) -> CvoiEvaluationRuntimeEnvironment:
    """Capture the exact CUDA environment used by direct World4Drive latency."""

    device = torch.device(device)
    if device.type != "cuda":
        if device.type != "cpu" or not _allow_cpu_for_tests:
            raise RuntimeError("formal CVoI evaluation runtime provenance requires a CUDA device")
        return CvoiEvaluationRuntimeEnvironment(
            device_type="cpu",
            device_index=None,
            gpu_model=None,
            gpu_compute_capability=None,
            torch_version=str(torch.__version__),
            cuda_runtime_version=None,
            cudnn_version=None,
            precision_contract=cvoi_execution_dtype_signature(config),
            test_only=True,
        )
    if _allow_cpu_for_tests:
        raise ValueError("_allow_cpu_for_tests is valid only with device='cpu'")
    if not torch.cuda.is_available():
        raise RuntimeError("formal CVoI evaluation runtime provenance requires an available CUDA device")

    device_index = int(torch.cuda.current_device() if device.index is None else device.index)
    capability = torch.cuda.get_device_capability(device_index)
    if (
        not isinstance(capability, Sequence)
        or len(capability) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in capability)
    ):
        raise RuntimeError(f"invalid CUDA compute capability for device {device_index}: {capability!r}")
    return CvoiEvaluationRuntimeEnvironment(
        device_type="cuda",
        device_index=device_index,
        gpu_model=str(torch.cuda.get_device_name(device_index)),
        gpu_compute_capability=f"{capability[0]}.{capability[1]}",
        torch_version=str(torch.__version__),
        cuda_runtime_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        precision_contract=cvoi_execution_dtype_signature(config),
        test_only=False,
    )


@runtime_checkable
class CvoiDeployment(Protocol):
    """Observed-only deployment operation used by retained evaluators."""

    def evaluate(
        self,
        navsim_batch: Sequence[object],
        *,
        lambda_compute: float,
        guidance_steps: int | None = None,
        forced_horizon: int | None = None,
    ) -> CvoiDeploymentOutput:
        raise NotImplementedError


def _require_standard_batch(navsim_batch: object, *, config: Any) -> tuple[Sequence[object], Mapping[str, object]]:
    if isinstance(navsim_batch, (str, bytes)) or not isinstance(navsim_batch, Sequence) or len(navsim_batch) != 12:
        raise ValueError("CVoI evaluation requires the standard 12-element NavSim batch tuple")
    states = navsim_batch[2]
    if not isinstance(states, torch.Tensor) or states.ndim != 3 or states.shape[0] != 1:
        raise ValueError("CVoI Controller evaluation requires NavSim batch size 1")
    if int(config.cvoi.controller_batch_size) != 1:
        raise ValueError("CVoI evaluation requires cvoi.controller_batch_size=1")
    metadata = navsim_batch[11]
    if not isinstance(metadata, Mapping):
        raise ValueError("CVoI evaluation requires metadata at NavSim batch element 11")
    domains = metadata.get("dataset_domain")
    if isinstance(domains, (str, bytes)) or not isinstance(domains, Sequence) or len(domains) != 1:
        raise ValueError("CVoI evaluation metadata['dataset_domain'] must contain one entry")
    if domains[0] not in {"real", "counterfactual"}:
        raise ValueError(f"unknown CVoI evaluation domain: {domains[0]!r}")
    return navsim_batch, metadata


def _metadata_entry(metadata: Mapping[str, object], name: str) -> object:
    value = metadata.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 1:
        raise ValueError(f"CVoI evaluation metadata[{name!r}] must contain one entry")
    return value[0]


def _real_outcome(
    navsim_batch: Sequence[object],
    metadata: Mapping[str, object],
    pred_trajs: torch.Tensor,
    confidences: torch.Tensor,
    *,
    config: Any,
    evaluator: PlanningOutcomeEvaluator,
) -> PlanningOutcome:
    geometry_present = metadata.get("geometry_present")
    future_valid = metadata.get("future_agent_geometry_valid")
    if (
        not isinstance(geometry_present, torch.Tensor)
        or geometry_present.dtype != torch.bool
        or geometry_present.shape != (1,)
        or not bool(geometry_present[0].item())
        or not isinstance(future_valid, torch.Tensor)
        or future_valid.dtype != torch.bool
        or future_valid.shape != (1,)
        or not bool(future_valid[0].item())
    ):
        raise ValueError("real CVoI evaluation requires valid logged future geometry")
    if _metadata_entry(metadata, "agent_geometry_truncated") is not False:
        raise ValueError("real CVoI evaluation rejects truncated agent geometry")

    num_observed = int(config.train.num_observed_frames)
    if not bool(config.train.predictor_inference_consistent):
        raise ValueError("CVoI evaluation requires predictor_inference_consistent=true")
    num_poses = int(pred_trajs.shape[2])
    states = navsim_batch[2]
    if not isinstance(states, torch.Tensor) or states.shape[1] < num_observed + num_poses:
        raise ValueError("real CVoI states do not cover the deployed planner horizon")
    gt_traj = build_future_gt_trajectory_from_states(states, num_observed)[:, :num_poses]

    agent_boxes = navsim_batch[7]
    agent_mask = navsim_batch[8]
    if (
        not isinstance(agent_boxes, torch.Tensor)
        or agent_boxes.ndim != 4
        or agent_boxes.shape[0] != 1
        or agent_boxes.shape[1] < num_observed + num_poses
        or agent_boxes.shape[2:] != (256, 7)
    ):
        raise ValueError("real CVoI evaluation requires complete agent_boxes [1,T,256,7]")
    if (
        not isinstance(agent_mask, torch.Tensor)
        or agent_mask.dtype != torch.bool
        or agent_mask.shape != agent_boxes.shape[:3]
    ):
        raise ValueError("real CVoI evaluation requires aligned bool agent_mask [1,T,256]")
    future_boxes = agent_boxes[:, num_observed : num_observed + num_poses].to(pred_trajs.device)
    future_mask = agent_mask[:, num_observed : num_observed + num_poses].to(pred_trajs.device)

    raw_agent_count = _metadata_entry(metadata, "raw_agent_count")
    if not isinstance(raw_agent_count, torch.Tensor) or raw_agent_count.ndim != 1:
        raise ValueError("real CVoI evaluation requires raw_agent_count [T]")
    future_counts = raw_agent_count[num_observed : num_observed + num_poses]
    if future_counts.shape != (num_poses,):
        raise ValueError("real CVoI raw_agent_count does not cover the deployed planner horizon")
    evaluator_metadata = {
        "dataset_domain": ["real"],
        "geometry_present": torch.ones(1, dtype=torch.bool, device=pred_trajs.device),
        "future_agent_geometry_valid": torch.ones(1, dtype=torch.bool, device=pred_trajs.device),
        "agent_geometry_truncated": torch.zeros(1, dtype=torch.bool, device=pred_trajs.device),
        "geometry_source": [_metadata_entry(metadata, "geometry_source")],
        "geometry_coordinate_frame": [_metadata_entry(metadata, "geometry_coordinate_frame")],
        "raw_agent_count": future_counts.unsqueeze(0),
    }
    return evaluator(
        pred_trajs,
        confidences,
        gt_traj.to(pred_trajs.device),
        future_boxes,
        future_mask,
        evaluator_metadata,
    )


def evaluate_navsim_real_planning_outcome(
    navsim_batch: Sequence[object],
    pred_trajs: torch.Tensor,
    confidences: torch.Tensor,
    *,
    config: Any,
    evaluator: PlanningOutcomeEvaluator,
) -> PlanningOutcome:
    """Evaluate one deployed NavSim trajectory through the canonical Real geometry adapter."""

    navsim_batch, metadata = _require_standard_batch(navsim_batch, config=config)
    if _metadata_entry(metadata, "dataset_domain") != "real":
        raise ValueError("NavSim real planning outcome evaluation rejects counterfactual samples")
    if not isinstance(evaluator, PlanningOutcomeEvaluator):
        raise TypeError("evaluator must be PlanningOutcomeEvaluator")
    return _real_outcome(
        navsim_batch,
        metadata,
        pred_trajs,
        confidences,
        config=config,
        evaluator=evaluator,
    )
