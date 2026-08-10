"""Strict row routing for production World4Drive online latency collection."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from app.vjepa_cowa_world_model.training.cvoi_policy_replay import uniform_random_horizon
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import PreparedNavSimCvoiDeployment
from app.vjepa_cowa_world_model.training.navsim_cvoi_world4drive import (
    CvoiWorld4DriveLatencyReport,
    build_navsim_world4drive_online_call,
    measure_world4drive_online_latency_calls,
)

_ROW_IDS = {
    "p0_controller": ("main_p0_controller",),
    "real_only_value": (
        "main_no_controller_real_only",
        "main_real_only_controller",
    ),
    "real_cf_value": (
        "main_real_cf_controller",
        "stopping_uniform_random",
        "stopping_learned",
        "guidance_k1",
        "guidance_k2",
        "guidance_k3",
        "guidance_k4",
    ),
}


@dataclass(frozen=True)
class _LatencyRowPolicy:
    row_id: str
    guidance_steps: int | None
    forced_horizon: int | str | None


@dataclass(frozen=True)
class CvoiWorld4DriveLatencyLineageIdentity:
    """Six byte-authoritative identities shared by all three latency lineages."""

    warmup: int
    repetitions: int
    hardware_signature: str
    runtime_signature: str
    sample_ids: tuple[str, ...]
    runtime_environment_bytes: bytes


_ROW_POLICIES = {
    "p0_controller": (_LatencyRowPolicy("main_p0_controller", None, None),),
    "real_only_value": (
        _LatencyRowPolicy("main_no_controller_real_only", 2, 3),
        _LatencyRowPolicy("main_real_only_controller", None, None),
    ),
    "real_cf_value": (
        _LatencyRowPolicy("main_real_cf_controller", None, None),
        _LatencyRowPolicy("stopping_uniform_random", 2, "uniform_random"),
        _LatencyRowPolicy("stopping_learned", None, None),
        _LatencyRowPolicy("guidance_k1", 1, None),
        _LatencyRowPolicy("guidance_k2", 2, None),
        _LatencyRowPolicy("guidance_k3", 3, None),
        _LatencyRowPolicy("guidance_k4", 4, None),
    ),
}


def world4drive_latency_row_ids(lineage: object) -> tuple[str, ...]:
    """Return the exact paper latency rows owned by one matched lineage."""

    if not isinstance(lineage, str) or lineage not in _ROW_IDS:
        raise ValueError(f"unsupported World4Drive latency lineage: {lineage!r}")
    return _ROW_IDS[lineage]


def collect_cvoi_world4drive_latency_reports(
    *,
    lineage: str,
    prepared_samples: Sequence[tuple[str, PreparedNavSimCvoiDeployment]],
    deployment: object,
    random_stop_seed: int,
    lambda_compute: float,
    warmup: int,
    repetitions: int,
    hardware_signature: str,
    runtime_signature: str,
    device: torch.device | str | None = None,
    measure: Callable[..., CvoiWorld4DriveLatencyReport] = measure_world4drive_online_latency_calls,
) -> Mapping[str, CvoiWorld4DriveLatencyReport]:
    """Measure every row belonging to ``lineage`` on the same prepared Real samples."""

    world4drive_latency_row_ids(lineage)
    if type(random_stop_seed) is not int or random_stop_seed < 0:
        raise ValueError("random_stop_seed must be a non-negative integer")
    if isinstance(lambda_compute, bool) or not isinstance(lambda_compute, (int, float)):
        raise ValueError("World4Drive latency lambda_compute must be numeric")
    lambda_compute = float(lambda_compute)
    if not math.isfinite(lambda_compute) or lambda_compute != 0.05:
        raise ValueError("World4Drive latency requires lambda_compute=0.05")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    samples = tuple(prepared_samples)
    if len(samples) != warmup + repetitions:
        raise ValueError("prepared_samples must contain exactly warmup + repetitions entries")
    sample_ids = [sample_id for sample_id, _prepared in samples]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("prepared World4Drive latency sample IDs must be non-empty strings")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("prepared World4Drive latency sample IDs must be unique")
    if any(not isinstance(prepared, PreparedNavSimCvoiDeployment) for _sample_id, prepared in samples):
        raise TypeError("prepared World4Drive latency samples contain an invalid deployment payload")

    reports = {}
    for policy in _ROW_POLICIES[lineage]:
        calls = []
        for sample_id, prepared in samples:
            forced_horizon = policy.forced_horizon
            if forced_horizon == "uniform_random":
                forced_horizon = uniform_random_horizon(sample_id, random_stop_seed)
            calls.append(
                build_navsim_world4drive_online_call(
                    deployment=deployment,
                    prepared=prepared,
                    lambda_compute=lambda_compute,
                    guidance_steps=policy.guidance_steps,
                    forced_horizon=forced_horizon,
                    hardware_signature=hardware_signature,
                    runtime_signature=runtime_signature,
                )
            )
        reports[policy.row_id] = measure(
            calls,
            warmup=warmup,
            repetitions=repetitions,
            hardware_signature=hardware_signature,
            runtime_signature=runtime_signature,
            expected_components=("full_agent_call", "adaptive_region"),
            device=device,
        )
    return reports


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"World4Drive latency {name} is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"World4Drive latency {name} must contain a JSON mapping")
    return payload


def _normalize_latency_artifact_paths(paths: Mapping[str, Path]) -> Mapping[str, Path]:
    expected = {"latency_records.jsonl", "runtime_environment.json", "manifest.json"}
    if not isinstance(paths, Mapping) or set(paths) != expected:
        raise ValueError(f"World4Drive latency artifact paths must be exactly {sorted(expected)!r}")
    normalized = {name: Path(paths[name]) for name in expected}
    parents = {path.parent for path in normalized.values()}
    if len(parents) != 1 or any(path.name != name for name, path in normalized.items()):
        raise ValueError("World4Drive latency artifact paths must name one exact lineage directory")
    missing = [str(path) for path in normalized.values() if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError(f"World4Drive latency artifact file(s) do not exist: {missing}")
    return normalized


def validate_world4drive_latency_lineage_identity(
    paths: Mapping[str, Path],
    *,
    lineage: str,
    expected: CvoiWorld4DriveLatencyLineageIdentity | None,
) -> CvoiWorld4DriveLatencyLineageIdentity:
    """Validate a just-written lineage and compare its six cross-lineage identities."""

    row_ids = world4drive_latency_row_ids(lineage)
    normalized = _normalize_latency_artifact_paths(paths)
    manifest = _load_json_mapping(normalized["manifest.json"], name="manifest")
    manifest_fields = {
        "schema",
        "lineage",
        "row_ids",
        "sample_ids",
        "warmup",
        "repetitions",
        "hardware_signature",
        "runtime_signature",
        "artifacts",
    }
    if set(manifest) != manifest_fields or manifest.get("schema") != "cvoi_world4drive_latency_collection_v1":
        raise ValueError("World4Drive latency manifest fields or schema do not match v1")
    if manifest.get("lineage") != lineage or tuple(manifest.get("row_ids", ())) != row_ids:
        raise ValueError(f"World4Drive latency manifest does not own the exact {lineage!r} rows")

    artifacts = manifest.get("artifacts")
    artifact_names = {"latency_records.jsonl", "runtime_environment.json"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != artifact_names:
        raise ValueError("World4Drive latency manifest artifacts do not match the v1 contract")
    for name in sorted(artifact_names):
        if artifacts[name] != _sha256(normalized[name]):
            raise ValueError(f"World4Drive latency artifact SHA mismatch: {name}")

    records = []
    try:
        lines = normalized["latency_records.jsonl"].read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines, start=1):
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"line {index} is not a mapping")
            records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("World4Drive latency records are not valid JSONL mappings") from exc
    if tuple(record.get("row_id") for record in records) != row_ids:
        raise ValueError(f"World4Drive latency records do not own the exact {lineage!r} rows")
    if any(record.get("schema") != "cvoi_world4drive_latency_v1" for record in records):
        raise ValueError("World4Drive latency record schema does not match v1")

    warmup = manifest.get("warmup")
    repetitions = manifest.get("repetitions")
    hardware_signature = manifest.get("hardware_signature")
    runtime_signature = manifest.get("runtime_signature")
    sample_ids = manifest.get("sample_ids")
    if type(warmup) is not int or warmup < 0 or type(repetitions) is not int or repetitions <= 0:
        raise ValueError("World4Drive latency manifest timing counts are invalid")
    if not isinstance(hardware_signature, str) or not hardware_signature:
        raise ValueError("World4Drive latency manifest hardware signature is invalid")
    if not isinstance(runtime_signature, str) or not runtime_signature:
        raise ValueError("World4Drive latency manifest runtime signature is invalid")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != warmup + repetitions
        or len(sample_ids) != len(set(sample_ids))
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
    ):
        raise ValueError("World4Drive latency manifest sample IDs are invalid")

    expected_record_fields = {
        "row_id",
        "schema",
        "warmup",
        "repetitions",
        "hardware_signature",
        "runtime_signature",
        "samples_ms",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "component_mean_ms",
    }
    for record in records:
        if (
            set(record) != expected_record_fields
            or record.get("warmup") != warmup
            or record.get("repetitions") != repetitions
            or record.get("hardware_signature") != hardware_signature
            or record.get("runtime_signature") != runtime_signature
        ):
            raise ValueError("World4Drive latency record timing or runtime identity disagrees with its manifest")
        samples_ms = record.get("samples_ms")
        component_mean_ms = record.get("component_mean_ms")
        summary_values = (record.get("mean_ms"), record.get("p50_ms"), record.get("p95_ms"))
        if (
            not isinstance(samples_ms, list)
            or len(samples_ms) != repetitions
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in (*samples_ms, *summary_values)
            )
            or not isinstance(component_mean_ms, Mapping)
            or set(component_mean_ms) != {"full_agent_call", "adaptive_region"}
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in component_mean_ms.values()
            )
        ):
            raise ValueError("World4Drive latency record measurements do not match the v1 contract")

    environment_bytes = normalized["runtime_environment.json"].read_bytes()
    environment = _load_json_mapping(normalized["runtime_environment.json"], name="runtime environment")
    if environment.get("hardware_signature") != hardware_signature:
        raise ValueError("World4Drive latency runtime environment hardware signature mismatch")
    if environment.get("runtime_signature") != runtime_signature:
        raise ValueError("World4Drive latency runtime environment runtime signature mismatch")
    identity = CvoiWorld4DriveLatencyLineageIdentity(
        warmup=warmup,
        repetitions=repetitions,
        hardware_signature=hardware_signature,
        runtime_signature=runtime_signature,
        sample_ids=tuple(sample_ids),
        runtime_environment_bytes=environment_bytes,
    )
    if expected is not None:
        if not isinstance(expected, CvoiWorld4DriveLatencyLineageIdentity):
            raise TypeError("expected latency lineage identity must be CvoiWorld4DriveLatencyLineageIdentity")
        labels = {
            "warmup": "warmup",
            "repetitions": "repetitions",
            "hardware_signature": "hardware signature",
            "runtime_signature": "runtime signature",
            "sample_ids": "sample IDs",
            "runtime_environment_bytes": "runtime environment bytes",
        }
        mismatches = [labels[field] for field in labels if getattr(identity, field) != getattr(expected, field)]
        if mismatches:
            raise ValueError(f"World4Drive latency cross-lineage identity mismatch: {mismatches}")
    return identity


def write_cvoi_world4drive_latency_collection(
    output_dir: str | Path,
    *,
    lineage: str,
    reports: Mapping[str, CvoiWorld4DriveLatencyReport],
    sample_ids: Sequence[str],
    runtime_environment: Mapping[str, object],
) -> Mapping[str, Path]:
    """Write one lineage-owned latency subset with strict runtime provenance."""

    row_ids = world4drive_latency_row_ids(lineage)
    if not isinstance(reports, Mapping) or tuple(reports) != row_ids:
        raise ValueError(f"World4Drive latency reports for {lineage!r} must be exactly {row_ids!r}")
    values = tuple(reports[row_id] for row_id in row_ids)
    if any(not isinstance(report, CvoiWorld4DriveLatencyReport) for report in values):
        raise TypeError("World4Drive latency reports contain an invalid value")
    warmups = {report.warmup for report in values}
    repetitions = {report.repetitions for report in values}
    hardware_signatures = {report.hardware_signature for report in values}
    runtime_signatures = {report.runtime_signature for report in values}
    if any(len(group) != 1 for group in (warmups, repetitions, hardware_signatures, runtime_signatures)):
        raise ValueError("World4Drive latency reports contain mixed timing or runtime signatures")
    normalized_sample_ids = tuple(sample_ids)
    if (
        len(normalized_sample_ids) != next(iter(warmups)) + next(iter(repetitions))
        or len(normalized_sample_ids) != len(set(normalized_sample_ids))
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in normalized_sample_ids)
    ):
        raise ValueError("World4Drive latency sample IDs must be unique and match timing counts")
    if not isinstance(runtime_environment, Mapping) or not runtime_environment:
        raise ValueError("World4Drive latency runtime_environment must be a non-empty mapping")
    environment = json.loads(json.dumps(runtime_environment, ensure_ascii=True, allow_nan=False, sort_keys=True))
    if environment.get("hardware_signature") != next(iter(hardware_signatures)):
        raise ValueError("World4Drive latency environment hardware signature does not match reports")
    if environment.get("runtime_signature") != next(iter(runtime_signatures)):
        raise ValueError("World4Drive latency environment runtime signature does not match reports")

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"World4Drive latency collection output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    latency_path = output / "latency_records.jsonl"
    with latency_path.open("x", encoding="utf-8") as handle:
        for row_id in row_ids:
            handle.write(
                json.dumps(
                    {"row_id": row_id, **asdict(reports[row_id])},
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    environment_path = output / "runtime_environment.json"
    environment_path.write_text(_json_text(environment), encoding="utf-8")
    paths = {
        "latency_records.jsonl": latency_path,
        "runtime_environment.json": environment_path,
    }
    manifest = {
        "schema": "cvoi_world4drive_latency_collection_v1",
        "lineage": lineage,
        "row_ids": list(row_ids),
        "sample_ids": list(normalized_sample_ids),
        "warmup": next(iter(warmups)),
        "repetitions": next(iter(repetitions)),
        "hardware_signature": next(iter(hardware_signatures)),
        "runtime_signature": next(iter(runtime_signatures)),
        "artifacts": {name: _sha256(path) for name, path in sorted(paths.items())},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    paths["manifest.json"] = manifest_path
    return paths
