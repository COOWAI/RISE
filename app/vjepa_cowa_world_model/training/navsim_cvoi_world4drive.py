"""Exhaustive Real-only NavSim collection for CVoI World4Drive evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from app.vjepa_cowa_world_model.training.cvoi_execution import (
    CvoiValueDtypeAdapter,
    cvoi_sample_seed,
    resolve_cvoi_evaluation_seed,
)
from app.vjepa_cowa_world_model.training.cvoi_horizon_cache import (
    CVOI_HORIZON_CACHE_SCHEMA,
    CvoiHorizonCacheRecord,
    CvoiHorizonStudyPoint,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_identity import (  # noqa: F401
    WORLD4DRIVE_CODE_SIGNATURE_SOURCES,
    CvoiWorld4DriveCacheProvenance,
    build_cvoi_world4drive_cache_provenance,
    cvoi_world4drive_code_signature,
    cvoi_world4drive_evaluation_dataset_signature,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_metrics import evaluate_world4drive_sample
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (
    CvoiControllerTrace,
    CvoiDeploymentOutput,
    CvoiFixedComputeTrace,
    CvoiPlannerEvaluation,
    PreparedNavSimCvoiDeployment,
    build_navsim_cvoi_model_batch,
    navsim_cvoi_raw_prefix,
    validate_navsim_cvoi_encoded_batch,
)
from app.vjepa_cowa_world_model.training.latent_value_guidance import CVOI_LEGACY_EVALUATION_GUIDANCE_STEPS
from app.vjepa_cowa_world_model.training.navsim_cvoi_evaluation import CvoiDeployment
from app.vjepa_cowa_world_model.training.runtimes.sequential_rollout_runtime import run_sequential_rollout
from app.vjepa_cowa_world_model.training.sequential_budget_control import extract_prefix_gate_values
from app.vjepa_cowa_world_model.utils.status_features import build_future_gt_trajectory_from_states

World4DriveTaskScoreProvider = Callable[[Sequence[object], CvoiPlannerEvaluation], float]
World4DriveOnlineCall = Callable[[], "CvoiWorld4DriveOnlineResult"]

_WORLD4DRIVE_PROVENANCE_SOURCES = WORLD4DRIVE_CODE_SIGNATURE_SOURCES


@dataclass(frozen=True)
class CvoiWorld4DriveOnlineResult:
    """Ready Planner output and diagnostic identities from one full online call."""

    trajectories: torch.Tensor
    confidences: torch.Tensor
    hardware_signature: str
    runtime_signature: str
    component_latency_ms: Mapping[str, float]


@dataclass(frozen=True)
class CvoiWorld4DriveLatencyReport:
    """Paper latency statistics from synchronized end-to-end online calls."""

    schema: str
    warmup: int
    repetitions: int
    hardware_signature: str
    runtime_signature: str
    samples_ms: tuple[float, ...]
    mean_ms: float
    p50_ms: float
    p95_ms: float
    component_mean_ms: Mapping[str, float]


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_online_result(
    value: object,
    *,
    hardware_signature: str,
    runtime_signature: str,
    expected_components: tuple[str, ...],
) -> CvoiWorld4DriveOnlineResult:
    if not isinstance(value, CvoiWorld4DriveOnlineResult):
        raise TypeError("World4Drive online call must return CvoiWorld4DriveOnlineResult")
    if value.hardware_signature != hardware_signature:
        raise ValueError("World4Drive online result hardware signature changed within one latency report")
    if value.runtime_signature != runtime_signature:
        raise ValueError("World4Drive online result runtime signature changed within one latency report")
    trajectories = value.trajectories
    confidences = value.confidences
    if (
        not isinstance(trajectories, torch.Tensor)
        or trajectories.ndim != 4
        or trajectories.shape[0] < 1
        or trajectories.shape[1] < 1
        or trajectories.shape[-1] != 3
        or not trajectories.is_floating_point()
        or not bool(torch.isfinite(trajectories).all().item())
    ):
        raise ValueError("World4Drive online trajectories must be ready finite [B,K,T,3] floating tensors")
    if (
        not isinstance(confidences, torch.Tensor)
        or confidences.shape != trajectories.shape[:2]
        or not confidences.is_floating_point()
        or not bool(torch.isfinite(confidences).all().item())
    ):
        raise ValueError("World4Drive online confidences must be ready finite [B,K] floating tensors")
    diagnostics = value.component_latency_ms
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(expected_components):
        raise ValueError("World4Drive online component diagnostics do not match the expected component set")
    for name in expected_components:
        latency = diagnostics[name]
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            raise ValueError(f"World4Drive component {name!r} latency must be numeric")
        if not math.isfinite(float(latency)) or float(latency) < 0.0:
            raise ValueError(f"World4Drive component {name!r} latency must be finite and non-negative")
    return value


def measure_world4drive_online_latency(
    online_call: World4DriveOnlineCall,
    *,
    warmup: int,
    repetitions: int,
    hardware_signature: str,
    runtime_signature: str,
    expected_components: Sequence[str],
    device: torch.device | str | None = None,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CvoiWorld4DriveLatencyReport:
    """Measure synchronized latency around the complete online Planner path."""

    if not callable(online_call):
        raise TypeError("online_call must be callable")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if not isinstance(hardware_signature, str) or not hardware_signature:
        raise ValueError("hardware_signature must be a non-empty string")
    if not isinstance(runtime_signature, str) or not runtime_signature:
        raise ValueError("runtime_signature must be a non-empty string")
    components = tuple(expected_components)
    if not components or len(components) != len(set(components)) or any(not name for name in components):
        raise ValueError("expected_components must contain unique non-empty names")
    if synchronize is None:
        resolved_device = torch.device("cpu" if device is None else device)

        def synchronize() -> None:
            if resolved_device.type == "cuda":
                torch.cuda.synchronize(resolved_device)

    for _ in range(warmup):
        synchronize()
        _validate_online_result(
            online_call(),
            hardware_signature=hardware_signature,
            runtime_signature=runtime_signature,
            expected_components=components,
        )
        synchronize()

    samples = []
    component_samples = {name: [] for name in components}
    for _ in range(repetitions):
        synchronize()
        start = float(clock())
        result = _validate_online_result(
            online_call(),
            hardware_signature=hardware_signature,
            runtime_signature=runtime_signature,
            expected_components=components,
        )
        synchronize()
        elapsed_ms = (float(clock()) - start) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            raise RuntimeError("World4Drive measured latency must be finite and non-negative")
        samples.append(elapsed_ms)
        for name in components:
            component_samples[name].append(float(result.component_latency_ms[name]))
    return CvoiWorld4DriveLatencyReport(
        schema="cvoi_world4drive_latency_v1",
        warmup=warmup,
        repetitions=repetitions,
        hardware_signature=hardware_signature,
        runtime_signature=runtime_signature,
        samples_ms=tuple(samples),
        mean_ms=sum(samples) / len(samples),
        p50_ms=_percentile(samples, 0.5),
        p95_ms=_percentile(samples, 0.95),
        component_mean_ms={name: sum(values) / len(values) for name, values in component_samples.items()},
    )


def measure_world4drive_online_latency_calls(
    online_calls: Sequence[World4DriveOnlineCall],
    *,
    warmup: int,
    repetitions: int,
    hardware_signature: str,
    runtime_signature: str,
    expected_components: Sequence[str],
    device: torch.device | str | None = None,
    synchronize: Callable[[], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CvoiWorld4DriveLatencyReport:
    """Measure one prepared online call per matched warm-up/evaluation sample."""

    calls = tuple(online_calls)
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if len(calls) != warmup + repetitions:
        raise ValueError("online_calls must contain exactly warmup + repetitions entries")
    if any(not callable(call) for call in calls):
        raise TypeError("online_calls must contain only callables")
    iterator = iter(calls)
    return measure_world4drive_online_latency(
        lambda: next(iterator)(),
        warmup=warmup,
        repetitions=repetitions,
        hardware_signature=hardware_signature,
        runtime_signature=runtime_signature,
        expected_components=expected_components,
        device=device,
        synchronize=synchronize,
        clock=clock,
    )


def build_navsim_world4drive_online_call(
    *,
    deployment: object,
    prepared: PreparedNavSimCvoiDeployment,
    lambda_compute: float,
    guidance_steps: int | None,
    forced_horizon: int | None = None,
    hardware_signature: str,
    runtime_signature: str,
    clock: Callable[[], float] = time.perf_counter,
) -> World4DriveOnlineCall:
    """Bind one prepared sample to the complete deployed online Planner call."""

    evaluate_prepared = getattr(deployment, "evaluate_prepared", None)
    if not callable(evaluate_prepared):
        raise TypeError("World4Drive deployment must expose callable evaluate_prepared")
    controller_lineage = getattr(deployment, "controller_lineage", None)
    if controller_lineage not in {"p0_controller", "value_guided"}:
        raise TypeError("World4Drive deployment must expose an exact controller_lineage")
    if not isinstance(prepared, PreparedNavSimCvoiDeployment):
        raise TypeError("prepared must be a PreparedNavSimCvoiDeployment value")
    if isinstance(lambda_compute, bool) or not isinstance(lambda_compute, (int, float)):
        raise ValueError("World4Drive lambda_compute must be numeric")
    lambda_compute = float(lambda_compute)
    if not math.isfinite(lambda_compute) or lambda_compute < 0.0:
        raise ValueError("World4Drive lambda_compute must be finite and non-negative")
    if guidance_steps is not None and (
        type(guidance_steps) is not int or guidance_steps not in CVOI_LEGACY_EVALUATION_GUIDANCE_STEPS
    ):
        raise ValueError(f"World4Drive guidance_steps must be one of {CVOI_LEGACY_EVALUATION_GUIDANCE_STEPS} or None")
    if forced_horizon is not None and (type(forced_horizon) is not int or forced_horizon not in {0, 1, 2, 3}):
        raise ValueError("World4Drive forced_horizon must be one of {0, 1, 2, 3} or None")
    for name, value in (("hardware_signature", hardware_signature), ("runtime_signature", runtime_signature)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if not callable(clock):
        raise TypeError("clock must be callable")

    def online_call() -> CvoiWorld4DriveOnlineResult:
        start = float(clock())
        kwargs = {"lambda_compute": lambda_compute, "guidance_steps": guidance_steps}
        if forced_horizon is not None:
            kwargs["forced_horizon"] = forced_horizon
        deployed = evaluate_prepared(prepared, **kwargs)
        if not isinstance(deployed, CvoiDeploymentOutput) or len(deployed.controller_traces) != 1:
            raise TypeError("World4Drive prepared deployment must return one CvoiDeploymentOutput trace")
        if deployed.pred_trajs.shape[0] != 1 or deployed.pred_trajs.shape[2:] != (prepared.raw_poses, 3):
            raise ValueError(
                f"World4Drive raw planner candidates must contain exactly {prepared.raw_poses} poses for batch one"
            )
        if deployed.confidences.shape != deployed.pred_trajs.shape[:2]:
            raise ValueError("World4Drive raw planner confidences must match the exact candidate modes")
        raw_trace = deployed.controller_traces[0]
        if forced_horizon is not None:
            if not isinstance(raw_trace, CvoiFixedComputeTrace):
                raise TypeError("World4Drive forced-horizon deployment must return a fixed compute trace")
            if raw_trace.horizon != forced_horizon:
                raise ValueError("World4Drive fixed compute trace horizon disagrees with forced_horizon")
            expected_steps = 0 if forced_horizon == 0 else (2 if guidance_steps is None else guidance_steps)
            if raw_trace.guidance_steps != expected_steps:
                raise ValueError("World4Drive fixed compute trace Guidance disagrees with the requested policy")
            if forced_horizon == 0 and raw_trace.latency_ms != 0.0:
                raise ValueError("World4Drive forced H0 must report zero adaptive latency")
            adaptive_region_ms = raw_trace.latency_ms
        else:
            if isinstance(raw_trace, CvoiFixedComputeTrace):
                raise TypeError("World4Drive controller deployment cannot return a fixed compute trace")
            if isinstance(raw_trace, CvoiControllerTrace):
                adaptive_region_ms = raw_trace.rollout_latency_ms
                stop_horizon = raw_trace.stop_horizon
                guidance = raw_trace.guidance
                reported_steps = guidance["guidance_steps"]
            elif isinstance(raw_trace, Mapping):
                expected_trace_fields = {
                    "stop_horizon",
                    "decisions",
                    "predicted_deltas",
                    "rollout_latency_ms",
                    "guidance",
                }
                if set(raw_trace) != expected_trace_fields:
                    raise ValueError("World4Drive controller mapping trace fields are not exact")
                adaptive_region_ms = raw_trace.get("rollout_latency_ms")
                stop_horizon = raw_trace.get("stop_horizon")
                if type(stop_horizon) is not int or stop_horizon not in {0, 1, 2, 3}:
                    raise ValueError("World4Drive controller trace stop_horizon must be one of {0,1,2,3}")
                decisions = raw_trace.get("decisions")
                if (
                    isinstance(decisions, (str, bytes))
                    or not isinstance(decisions, Sequence)
                    or tuple(decisions) != ("ROLL",) * stop_horizon + ("STOP",)
                ):
                    raise ValueError("World4Drive controller trace decisions must end exactly at stop_horizon")
                predicted_deltas = raw_trace.get("predicted_deltas")
                if (
                    isinstance(predicted_deltas, (str, bytes))
                    or not isinstance(predicted_deltas, Sequence)
                    or len(predicted_deltas) != min(stop_horizon + 1, 3)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        for value in predicted_deltas
                    )
                ):
                    raise ValueError("World4Drive controller trace predicted_deltas must cover exact Gate decisions")
                guidance = raw_trace.get("guidance")
                guidance_fields = {
                    "guidance_steps",
                    "guidance_skipped_h0",
                    "delta_norm",
                    "field_value_before",
                    "field_value_after",
                }
                if not isinstance(guidance, Mapping) or set(guidance) != guidance_fields:
                    raise ValueError("World4Drive prepared deployment trace guidance fields are not exact")
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                    for value in guidance.values()
                ):
                    raise ValueError("World4Drive prepared deployment trace guidance values must be finite")
                reported_steps = guidance.get("guidance_steps")
                if isinstance(reported_steps, bool) or not isinstance(reported_steps, (int, float)):
                    raise ValueError("World4Drive prepared deployment trace guidance_steps must be numeric")
            else:
                raise TypeError("World4Drive controller deployment trace must be a mapping or CvoiControllerTrace")
            expected_steps = 2 if guidance_steps is None else guidance_steps
            if controller_lineage == "p0_controller":
                if float(reported_steps) != 0.0:
                    raise ValueError("World4Drive P0 controller trace must report Guidance K0")
                if any(
                    float(guidance[name]) != 0.0
                    for name in (
                        "delta_norm",
                        "field_value_before",
                        "field_value_after",
                    )
                ):
                    raise ValueError("World4Drive P0 controller trace must report zero Guidance diagnostics")
            elif controller_lineage == "value_guided":
                expected_actual = 0 if stop_horizon == 0 else expected_steps
                if float(reported_steps) != float(expected_actual):
                    raise ValueError(
                        "World4Drive Value controller trace guidance_steps disagrees with terminal horizon"
                    )
            expected_skip = float(stop_horizon == 0)
            if float(guidance["guidance_skipped_h0"]) != expected_skip:
                raise ValueError("World4Drive guidance_skipped_h0 disagrees with terminal horizon")
            if stop_horizon == 0 and float(guidance["delta_norm"]) != 0.0:
                raise ValueError("World4Drive H0 Guidance must report zero delta_norm")
        if isinstance(adaptive_region_ms, bool) or not isinstance(adaptive_region_ms, (int, float)):
            raise ValueError("World4Drive adaptive-region diagnostic latency must be numeric")
        adaptive_region_ms = float(adaptive_region_ms)
        if not math.isfinite(adaptive_region_ms) or adaptive_region_ms < 0.0:
            raise ValueError("World4Drive adaptive-region diagnostic latency must be finite and non-negative")
        elapsed_ms = (float(clock()) - start) * 1000.0
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0.0:
            raise RuntimeError("World4Drive full agent diagnostic latency must be finite and non-negative")
        return CvoiWorld4DriveOnlineResult(
            trajectories=deployed.pred_trajs,
            confidences=deployed.confidences,
            hardware_signature=hardware_signature,
            runtime_signature=runtime_signature,
            component_latency_ms={
                "full_agent_call": elapsed_ms,
                "adaptive_region": adaptive_region_ms,
            },
        )

    return online_call


def _metadata_entry(metadata: Mapping[str, object], name: str) -> object:
    value = metadata.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 1:
        raise ValueError(f"World4Drive metadata[{name!r}] must contain exactly one entry")
    return value[0]


def _require_real_batch(raw_batch: object, *, config: Any) -> tuple[Sequence[object], Mapping[str, object]]:
    if isinstance(raw_batch, (str, bytes)) or not isinstance(raw_batch, Sequence) or len(raw_batch) != 12:
        raise ValueError("World4Drive collection requires the standard 12-element NavSim batch")
    states = raw_batch[2]
    if not isinstance(states, torch.Tensor) or states.ndim != 3 or states.shape[0] != 1:
        raise ValueError("World4Drive collection requires NavSim batch size 1")
    if int(config.cvoi.controller_batch_size) != 1:
        raise ValueError("World4Drive collection requires cvoi.controller_batch_size=1")
    metadata = raw_batch[11]
    if not isinstance(metadata, Mapping):
        raise ValueError("World4Drive collection requires metadata at batch element 11")
    domain = _metadata_entry(metadata, "dataset_domain")
    if domain != "real":
        raise ValueError(f"World4Drive collection is Real-only, got dataset_domain={domain!r}")
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
        raise ValueError("World4Drive collection requires future_agent_geometry_valid=True")
    if _metadata_entry(metadata, "agent_geometry_truncated") is not False:
        raise ValueError("World4Drive collection rejects truncated agent geometry")
    return raw_batch, metadata


def _p0_controller_trace_horizon(raw_trace: object) -> int:
    """Validate the no-Guidance online trace emitted by the P0 Controller."""

    expected_fields = {"stop_horizon", "decisions", "predicted_deltas", "rollout_latency_ms", "guidance"}
    if not isinstance(raw_trace, Mapping) or set(raw_trace) != expected_fields:
        raise ValueError("P0 Controller trace fields do not match the no-Guidance protocol")
    horizon = raw_trace["stop_horizon"]
    if type(horizon) is not int or horizon not in {0, 1, 2, 3}:
        raise ValueError("P0 Controller trace stop_horizon must be in {0,1,2,3}")
    decisions = raw_trace["decisions"]
    if isinstance(decisions, (str, bytes)) or tuple(decisions) != ("ROLL",) * horizon + ("STOP",):
        raise ValueError("P0 Controller trace decisions must end at stop_horizon")
    deltas = raw_trace["predicted_deltas"]
    if isinstance(deltas, (str, bytes)) or not isinstance(deltas, Sequence) or len(deltas) != min(horizon + 1, 3):
        raise ValueError("P0 Controller trace predicted_deltas do not cover every Gate decision")
    numeric = [*deltas, raw_trace["rollout_latency_ms"]]
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in numeric
    ):
        raise ValueError("P0 Controller trace deltas and latency must be finite numeric values")
    if float(raw_trace["rollout_latency_ms"]) < 0.0:
        raise ValueError("P0 Controller trace latency must be non-negative")
    guidance = raw_trace["guidance"]
    guidance_fields = {
        "guidance_steps",
        "guidance_skipped_h0",
        "delta_norm",
        "field_value_before",
        "field_value_after",
    }
    if not isinstance(guidance, Mapping) or set(guidance) != guidance_fields:
        raise ValueError("P0 Controller trace guidance diagnostics are incomplete")
    expected_guidance = {
        "guidance_steps": 0.0,
        "guidance_skipped_h0": float(horizon == 0),
        "delta_norm": 0.0,
        "field_value_before": 0.0,
        "field_value_after": 0.0,
    }
    for name, expected in expected_guidance.items():
        value = guidance[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != expected:
            raise ValueError(f"P0 Controller trace requires guidance.{name}={expected}")
    return horizon


def collect_navsim_world4drive_controller_horizons(
    loader: Iterable[Sequence[object]],
    *,
    deployment: CvoiDeployment,
    config: Any,
    lambda_compute: float,
) -> dict[str, int]:
    """Collect the terminal horizon selected by the deployed online Gate for every Real sample."""

    if str(config.cvoi.stage) != "evaluation" or int(config.cvoi.max_horizon) != 3:
        raise ValueError("World4Drive Controller collection requires cvoi.stage='evaluation' and max_horizon=3")
    if isinstance(lambda_compute, bool) or not isinstance(lambda_compute, (int, float)):
        raise ValueError("World4Drive Controller collection requires lambda_compute=0.05")
    lambda_compute = float(lambda_compute)
    if not math.isfinite(lambda_compute) or lambda_compute != 0.05:
        raise ValueError("World4Drive Controller collection requires lambda_compute=0.05")
    if not isinstance(deployment, CvoiDeployment):
        raise TypeError("deployment must implement the CvoiDeployment protocol")

    horizons: dict[str, int] = {}
    for raw_batch in loader:
        navsim_batch, metadata = _require_real_batch(raw_batch, config=config)
        sample_id = str(_metadata_entry(metadata, "stable_sample_id"))
        if sample_id in horizons:
            raise ValueError(f"World4Drive Controller collection contains duplicate sample_id={sample_id!r}")
        deployed = deployment.evaluate(navsim_batch, lambda_compute=lambda_compute)
        if not isinstance(deployed, CvoiDeploymentOutput):
            raise TypeError("World4Drive Controller deployment must return CvoiDeploymentOutput")
        if deployed.pred_trajs.shape[0] != 1 or len(deployed.controller_traces) != 1:
            raise ValueError("World4Drive Controller deployment must preserve batch size 1")
        raw_trace = deployed.controller_traces[0]
        if getattr(config.cvoi, "controller_lineage", "value_guided") == "p0_controller":
            stop_horizon = _p0_controller_trace_horizon(raw_trace)
        elif isinstance(raw_trace, CvoiControllerTrace):
            stop_horizon = raw_trace.stop_horizon
        elif isinstance(raw_trace, Mapping):
            stop_horizon = CvoiControllerTrace.from_mapping(raw_trace).stop_horizon
        else:
            raise TypeError("World4Drive Controller trace must be a mapping or CvoiControllerTrace")
        horizons[sample_id] = stop_horizon
    if not horizons:
        raise ValueError("World4Drive Controller collection produced no horizons")
    return dict(sorted(horizons.items()))


def collect_navsim_world4drive_controller_horizons_from_runtime(
    loader: Iterable[Sequence[object]],
    *,
    runtime: object,
    gate: torch.nn.Module,
    config: Any,
    device: torch.device | str,
    lambda_compute: float,
) -> dict[str, int]:
    """Collect deployed Gate horizons without importing the external NavSim agent stack."""

    if str(config.cvoi.stage) != "evaluation" or int(config.cvoi.max_horizon) != 3:
        raise ValueError("World4Drive Controller collection requires cvoi.stage='evaluation' and max_horizon=3")
    if isinstance(lambda_compute, bool) or not isinstance(lambda_compute, (int, float)):
        raise ValueError("World4Drive Controller collection requires lambda_compute=0.05")
    lambda_compute = float(lambda_compute)
    if not math.isfinite(lambda_compute) or lambda_compute != 0.05:
        raise ValueError("World4Drive Controller collection requires lambda_compute=0.05")
    if not callable(getattr(runtime, "encode_batch", None)):
        raise TypeError("World4Drive runtime Controller collection requires runtime.encode_batch")
    embed_dim = getattr(runtime, "embed_dim", None)
    if type(embed_dim) is not int or embed_dim <= 0:
        raise ValueError("World4Drive runtime must declare a positive integer embed_dim")
    dual_value_model = getattr(runtime, "dual_value_model", None)
    if not isinstance(dual_value_model, torch.nn.Module):
        raise TypeError("World4Drive runtime Controller collection requires a dual Value model")
    if not isinstance(gate, torch.nn.Module):
        raise TypeError("World4Drive runtime Controller collection requires a Gate module")

    device = torch.device(device)
    gate = gate.to(device).eval()
    value_adapter = CvoiValueDtypeAdapter(dual_value_model).to(device).eval()
    tokens_per_frame = int(config.cvoi.tokens_per_frame)
    compute_costs = [float(value) for value in config.cvoi.compute_costs]
    signature = getattr(config.cvoi, "ablation_signature", None)
    gate_feature_mode = "full" if signature is None else str(signature.gate_feature_mode)
    controller_lineage = str(getattr(config.cvoi, "controller_lineage", "value_guided"))

    horizons: dict[str, int] = {}
    for raw_batch in loader:
        navsim_batch, metadata = _require_real_batch(raw_batch, config=config)
        sample_id = str(_metadata_entry(metadata, "stable_sample_id"))
        if sample_id in horizons:
            raise ValueError(f"World4Drive Controller collection contains duplicate sample_id={sample_id!r}")
        model_batch = build_navsim_cvoi_model_batch(navsim_batch, config=config, device=device)
        with torch.no_grad():
            encoded = runtime.encode_batch(model_batch)
        encoded = validate_navsim_cvoi_encoded_batch(
            encoded,
            batch_size=1,
            embed_dim=embed_dim,
            max_horizon=3,
            tokens_per_frame=tokens_per_frame,
        )
        observed = encoded.z_observed.reshape(1, -1, embed_dim)

        def rollout_step(raw_prefix: torch.Tensor, next_horizon: int) -> torch.Tensor:
            expected_tokens = (int(next_horizon) - 1) * tokens_per_frame
            if raw_prefix.shape != (1, expected_tokens, embed_dim):
                raise ValueError("World4Drive Gate raw prefix does not match next_horizon")
            return encoded.z_future[:, int(next_horizon) - 1].reshape(1, tokens_per_frame, embed_dim)

        def value_features(
            observed_latent: torch.Tensor,
            raw_prefix: torch.Tensor,
            horizon: int,
        ) -> dict[str, torch.Tensor]:
            if raw_prefix.shape[1] != int(horizon) * tokens_per_frame:
                raise ValueError("World4Drive Gate Value prefix does not match horizon")
            with torch.no_grad():
                values = extract_prefix_gate_values(
                    value_adapter(observed_latent, raw_prefix, tokens_per_frame=tokens_per_frame)
                )
            field_value = values["field_value"]
            if controller_lineage == "p0_controller":
                field_value = torch.zeros_like(field_value)
            return {"field_value": field_value, "stop_value": values["stop_value"]}

        def stop_at_horizon(_raw_prefix: torch.Tensor, horizon: int, apply_guidance: bool) -> int:
            if apply_guidance != (int(horizon) > 0):
                raise RuntimeError("World4Drive Gate produced an inconsistent terminal Guidance decision")
            return int(horizon)

        result = run_sequential_rollout(
            observed_latent=observed,
            gate=gate,
            max_horizon=3,
            lambda_compute=lambda_compute,
            compute_costs=compute_costs,
            rollout_step=rollout_step,
            value_features=value_features,
            stop_and_plan=stop_at_horizon,
            gate_feature_mode=gate_feature_mode,
        )
        result.require_finite_rollout_tokens()
        if result.planner_output != result.stop_horizon:
            raise RuntimeError("World4Drive runtime Controller terminal horizon is inconsistent")
        horizons[sample_id] = result.stop_horizon
    if not horizons:
        raise ValueError("World4Drive Controller collection produced no horizons")
    return dict(sorted(horizons.items()))


def _validate_protocol(
    *,
    config: Any,
    lineage: str,
    study_points: Sequence[CvoiHorizonStudyPoint],
) -> tuple[tuple[CvoiHorizonStudyPoint, ...], tuple[float, float, float, float]]:
    if str(config.cvoi.stage) != "evaluation" or int(config.cvoi.max_horizon) != 3:
        raise ValueError("World4Drive collection requires cvoi.stage='evaluation' and max_horizon=3")
    if lineage not in {"p0_controller", "real_only_value", "real_cf_value"}:
        raise ValueError(f"unsupported World4Drive lineage: {lineage!r}")
    points = tuple(study_points)
    if not points:
        raise ValueError("World4Drive study_points must not be empty")
    if any(point.lineage != lineage for point in points):
        raise ValueError(f"every study point lineage must equal {lineage!r}")
    identities = [(point.horizon, point.guidance_steps) for point in points]
    if len(identities) != len(set(identities)):
        raise ValueError("World4Drive study_points contains duplicate horizon/K identities")
    if not any(point.horizon == 0 for point in points):
        raise ValueError("World4Drive study_points must include h=0")
    raw_costs = getattr(config.cvoi, "compute_costs", None)
    if not isinstance(raw_costs, (list, tuple)) or len(raw_costs) != 4:
        raise ValueError("World4Drive collection requires four cvoi.compute_costs")
    costs = tuple(float(value) for value in raw_costs)
    if any(not math.isfinite(value) or value < 0.0 for value in costs):
        raise ValueError("World4Drive cvoi.compute_costs must be finite and non-negative")
    return tuple(sorted(points, key=lambda point: (point.horizon, point.guidance_steps))), costs


def _validate_planner_result(
    result: object,
    *,
    expected_guidance_steps: int,
) -> CvoiPlannerEvaluation:
    if not isinstance(result, CvoiPlannerEvaluation):
        raise TypeError("World4Drive runtime must return CvoiPlannerEvaluation")
    if result.guidance_steps != expected_guidance_steps:
        raise ValueError(
            f"World4Drive runtime must report guidance_steps={expected_guidance_steps}, "
            f"got {result.guidance_steps!r}"
        )
    if result.pred_trajs.shape[0] != 1 or result.pred_trajs.shape[2:] != (6, 3):
        raise ValueError("World4Drive Planner trajectories must be [1,K,6,3]")
    return result


def _confidence_digest(confidences: torch.Tensor) -> str:
    cpu = confidences.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(cpu.dtype), "shape": list(cpu.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + cpu.numpy().tobytes()).hexdigest()


def _external_metric_inputs(
    navsim_batch: Sequence[object],
    *,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, float]:
    num_observed = int(config.train.num_observed_frames)
    states = navsim_batch[2]
    segmentation = navsim_batch[9]
    if (
        not isinstance(states, torch.Tensor)
        or states.ndim != 3
        or states.shape[0] != 1
        or states.shape[1] < num_observed + 6
        or states.shape[2] != 7
        or not bool(torch.isfinite(states).all().item())
    ):
        raise ValueError("World4Drive states must provide finite [1,observed+6,7] ego poses")
    if (
        not isinstance(segmentation, torch.Tensor)
        or segmentation.ndim != 4
        or segmentation.shape[0] != 1
        or segmentation.shape[1] < num_observed + 6
    ):
        raise ValueError("World4Drive BEV segmentation must cover all six future steps")
    gt_trajectory = build_future_gt_trajectory_from_states(states, num_observed)[:, :6]
    future_segmentation = segmentation[:, num_observed : num_observed + 6]
    return (
        gt_trajectory,
        future_segmentation,
        states,
        num_observed,
        num_observed - 1,
        1.0 / float(config.data.fps),
    )


def collect_navsim_world4drive_lineage(
    loader: Iterable[Sequence[object]],
    *,
    runtime: object,
    config: Any,
    device: torch.device | str,
    lineage: str,
    study_points: Sequence[CvoiHorizonStudyPoint],
    provenance: CvoiWorld4DriveCacheProvenance,
    task_score_provider: World4DriveTaskScoreProvider | None,
) -> tuple[CvoiHorizonCacheRecord, ...]:
    """Encode each Real sample once and exhaustively evaluate one lineage."""

    points, compute_costs = _validate_protocol(config=config, lineage=lineage, study_points=study_points)
    if not isinstance(provenance, CvoiWorld4DriveCacheProvenance):
        raise TypeError("provenance must be CvoiWorld4DriveCacheProvenance")
    if not callable(task_score_provider):
        raise TypeError("task_score_provider must be callable")
    embed_dim = getattr(runtime, "embed_dim", None)
    if type(embed_dim) is not int or embed_dim <= 0:
        raise ValueError("World4Drive runtime must declare a positive integer embed_dim")
    for method in ("encode_batch", "evaluate_guided_horizon", "evaluate_unguided_prefix"):
        if not callable(getattr(runtime, method, None)):
            raise ValueError(f"World4Drive runtime is missing method {method}")
    records = []
    for raw_batch in loader:
        navsim_batch, metadata = _require_real_batch(raw_batch, config=config)
        sample_id = str(_metadata_entry(metadata, "stable_sample_id"))
        source_scene_id = str(_metadata_entry(metadata, "base_scene_id"))
        seed = cvoi_sample_seed(resolve_cvoi_evaluation_seed(config), sample_id)
        model_batch = build_navsim_cvoi_model_batch(
            navsim_batch,
            config=config,
            device=torch.device(device),
        )
        with torch.no_grad():
            encoded = runtime.encode_batch(model_batch)
        encoded = validate_navsim_cvoi_encoded_batch(
            encoded,
            batch_size=1,
            embed_dim=embed_dim,
            max_horizon=3,
            tokens_per_frame=int(config.cvoi.tokens_per_frame),
        )
        gt_trajectory, segmentation, ego_poses, future_start_idx, reference_frame_idx, timestep_sec = (
            _external_metric_inputs(navsim_batch, config=config)
        )
        for point in points:
            raw_prefix = navsim_cvoi_raw_prefix(
                encoded.z_future,
                point.horizon,
                tokens_per_frame=int(config.cvoi.tokens_per_frame),
            )
            if lineage == "p0_controller":
                result = runtime.evaluate_unguided_prefix(
                    context=encoded.model_contexts[0],
                    z_observed=encoded.z_observed,
                    prefix=raw_prefix,
                    horizon=point.horizon,
                    seed=seed,
                )
            else:
                result = runtime.evaluate_guided_horizon(
                    context=encoded.model_contexts[0],
                    z_observed=encoded.z_observed,
                    raw_prefix=raw_prefix,
                    horizon=point.horizon,
                    apply_guidance=point.horizon > 0,
                    seed=seed,
                    guidance_steps=None if point.horizon == 0 else point.guidance_steps,
                )
            result = _validate_planner_result(result, expected_guidance_steps=point.guidance_steps)
            metrics = evaluate_world4drive_sample(
                dataset_domain="real",
                future_agent_geometry_valid=True,
                trajectories=result.pred_trajs,
                confidences=result.confidences,
                gt_trajectory=gt_trajectory.to(result.pred_trajs.device),
                agent_seg=segmentation.to(result.pred_trajs.device),
                ego_poses=ego_poses.to(result.pred_trajs.device),
                future_start_idx=future_start_idx,
                reference_frame_idx=reference_frame_idx,
                timestep_sec=timestep_sec,
            )
            task_score = task_score_provider(navsim_batch, result)
            if isinstance(task_score, bool) or not isinstance(task_score, (int, float)):
                raise ValueError(f"task_score_provider must return one numeric score, got {task_score!r}")
            task_score = float(task_score)
            if not math.isfinite(task_score) or not 0.0 <= task_score <= 1.0:
                raise ValueError(f"task_score_provider must return a finite score in [0,1], got {task_score}")
            records.append(
                CvoiHorizonCacheRecord(
                    schema=CVOI_HORIZON_CACHE_SCHEMA,
                    dataset_domain="real",
                    sample_id=sample_id,
                    source_scene_id=source_scene_id,
                    seed=seed,
                    lineage=lineage,
                    horizon=point.horizon,
                    guidance_steps=point.guidance_steps,
                    selected_mode=metrics.selected_mode,
                    selected_trajectory=metrics.selected_trajectory,
                    confidence_digest=_confidence_digest(result.confidences),
                    l2_per_step=metrics.l2_per_step,
                    collision_counts=metrics.collision_counts,
                    gt_collision_counts=metrics.gt_collision_counts,
                    l2_at_1s=metrics.l2_at_1s,
                    l2_at_2s=metrics.l2_at_2s,
                    l2_at_3s=metrics.l2_at_3s,
                    l2_avg=metrics.l2_avg,
                    collision_at_1s=metrics.collision_at_1s,
                    collision_at_2s=metrics.collision_at_2s,
                    collision_at_3s=metrics.collision_at_3s,
                    collision_rate=metrics.collision_rate,
                    task_score=task_score,
                    compute_cost=compute_costs[point.horizon],
                    lambda_compute=0.05,
                    checkpoint_signature=provenance.checkpoint_signature,
                    dataset_signature=provenance.dataset_signature,
                    code_signature=provenance.code_signature,
                    converter_signature=provenance.converter_signature,
                    evaluator_signature=provenance.evaluator_signature,
                )
            )
    if not records:
        raise ValueError("World4Drive collection produced no records")
    keys = [record.cache_key for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("World4Drive collection produced duplicate cache identities")
    return tuple(sorted(records, key=lambda record: record.cache_key))
