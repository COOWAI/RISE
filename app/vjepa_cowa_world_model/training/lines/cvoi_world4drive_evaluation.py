"""Deterministic report entry for the CVoI World4Drive evaluation protocol."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

from app.vjepa_cowa_world_model.training.configs.cvoi_world4drive import (
    CVOI_WORLD4DRIVE_LINEAGE_ORDER,
    CvoiWorld4DriveEvaluationConfig,
    parse_cvoi_world4drive_direct_config,
)
from app.vjepa_cowa_world_model.training.cvoi_horizon_cache import (
    CvoiExpectedCacheSample,
    CvoiHorizonCacheRecord,
    CvoiHorizonStudyPoint,
    parse_cvoi_horizon_cache_record,
    validate_cvoi_horizon_cache,
    write_cvoi_horizon_cache,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_input_assembly import assemble_cvoi_world4drive_inputs
from app.vjepa_cowa_world_model.training.cvoi_world4drive_report import (
    build_guidance_steps_rows,
    build_main_ablation_rows,
    build_oracle_matrices,
    build_stopping_strategy_rows,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (
    require_world4drive_inputs_present,
    require_world4drive_outputs_absent,
)
from app.vjepa_cowa_world_model.training.navsim_cvoi_world4drive import CvoiWorld4DriveLatencyReport
from src.utils.logging import get_logger

logger = get_logger(__name__)

_LATENCY_IDS = (
    "main_no_controller_real_only",
    "main_p0_controller",
    "main_real_only_controller",
    "main_real_cf_controller",
    "stopping_uniform_random",
    "stopping_learned",
    "guidance_k1",
    "guidance_k2",
    "guidance_k3",
    "guidance_k4",
)
_MAIN_LATENCY_IDS = _LATENCY_IDS[:4]
_STOPPING_LATENCY_IDS = _LATENCY_IDS[4:6]
_GUIDANCE_LATENCY_IDS = _LATENCY_IDS[6:]
_SUMMARY_FIELDS = (
    "sample_count",
    "scene_count",
    "l2_at_1s",
    "l2_at_2s",
    "l2_at_3s",
    "l2_avg",
    "collision_at_1s",
    "collision_at_2s",
    "collision_at_3s",
    "collision_rate",
    "average_rollout_count",
    "rollout_histogram",
)
_INPUT_FIELDS = frozenset(
    {
        "cache_path",
        "controller_horizons_path",
        "latency_records_path",
        "runtime_environment_path",
    }
)


@dataclass(frozen=True)
class CvoiWorld4DriveEvaluationInputs:
    """Validated model outputs and independent online timings consumed by writers."""

    records: Sequence[CvoiHorizonCacheRecord]
    controller_horizons: Mapping[str, Mapping[str, int]]
    latency_reports: Mapping[str, CvoiWorld4DriveLatencyReport]
    runtime_environment: Mapping[str, object]


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("World4Drive report artifacts reject non-finite floats")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"World4Drive report value is not JSON serializable: {type(value).__name__}")


def _json_text(value: object) -> str:
    return json.dumps(_json_value(value), ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def _write_text_exclusive(path: Path, payload: str) -> Path:
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)
    return path


def _csv_text(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for row in rows:
        normalized = {}
        for name in fieldnames:
            value = row[name]
            if value is None:
                value = "NA"
            elif isinstance(value, (dict, list, tuple)):
                value = json.dumps(_json_value(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            normalized[name] = value
        writer.writerow(normalized)
    return output.getvalue()


def world4drive_study_points() -> tuple[CvoiHorizonStudyPoint, ...]:
    """Return the exact cache product required by the approved protocol."""

    points = [CvoiHorizonStudyPoint("p0_controller", horizon, 0) for horizon in range(4)]
    points.extend(CvoiHorizonStudyPoint("real_only_value", horizon, 0 if horizon == 0 else 2) for horizon in range(4))
    points.append(CvoiHorizonStudyPoint("real_cf_value", 0, 0))
    points.extend(
        CvoiHorizonStudyPoint("real_cf_value", horizon, guidance_steps)
        for horizon in (1, 2, 3)
        for guidance_steps in (1, 2, 3, 4)
    )
    return tuple(points)


def _expected_samples(records: Sequence[CvoiHorizonCacheRecord]) -> tuple[CvoiExpectedCacheSample, ...]:
    by_lineage: dict[str, set[tuple[str, str, int]]] = {}
    for record in records:
        by_lineage.setdefault(record.lineage, set()).add((record.sample_id, record.source_scene_id, record.seed))
    if set(by_lineage) != {"p0_controller", "real_only_value", "real_cf_value"}:
        raise ValueError("World4Drive records must contain all three matched lineages")
    sample_sets = list(by_lineage.values())
    if any(values != sample_sets[0] for values in sample_sets[1:]):
        raise ValueError("World4Drive matched lineages must contain identical sample identities")
    return tuple(CvoiExpectedCacheSample(*identity) for identity in sorted(sample_sets[0]))


def _validate_controller_horizons(
    values: Mapping[str, Mapping[str, int]],
    *,
    sample_ids: set[str],
) -> dict[str, dict[str, int]]:
    expected_lineages = {"p0_controller", "real_only_value", "real_cf_value"}
    if not isinstance(values, Mapping) or set(values) != expected_lineages:
        raise ValueError("controller_horizons must contain exactly the three matched lineages")
    normalized = {}
    for lineage in sorted(expected_lineages):
        horizons = values[lineage]
        if not isinstance(horizons, Mapping) or set(horizons) != sample_ids:
            raise ValueError(f"controller_horizons[{lineage!r}] must match the cache sample IDs")
        if any(type(horizon) is not int or horizon not in {0, 1, 2, 3} for horizon in horizons.values()):
            raise ValueError(f"controller_horizons[{lineage!r}] contains an invalid horizon")
        normalized[lineage] = dict(horizons)
    return normalized


def _validate_latency_reports(
    reports: Mapping[str, CvoiWorld4DriveLatencyReport],
    *,
    config: CvoiWorld4DriveEvaluationConfig,
) -> dict[str, CvoiWorld4DriveLatencyReport]:
    if not isinstance(reports, Mapping) or set(reports) != set(_LATENCY_IDS):
        raise ValueError("latency_reports must contain the exact World4Drive policy rows")
    normalized = {}
    hardware_signatures = set()
    runtime_signatures = set()
    for row_id in _LATENCY_IDS:
        report = reports[row_id]
        if not isinstance(report, CvoiWorld4DriveLatencyReport):
            raise TypeError(f"latency_reports[{row_id!r}] must be CvoiWorld4DriveLatencyReport")
        if report.schema != "cvoi_world4drive_latency_v1":
            raise ValueError(f"latency_reports[{row_id!r}] has an invalid schema")
        if report.warmup != config.latency_warmup or report.repetitions != config.latency_repetitions:
            raise ValueError(f"latency_reports[{row_id!r}] timing counts do not match the evaluation config")
        if len(report.samples_ms) != report.repetitions:
            raise ValueError(f"latency_reports[{row_id!r}] samples do not match repetitions")
        _json_value(report)
        hardware_signatures.add(report.hardware_signature)
        runtime_signatures.add(report.runtime_signature)
        normalized[row_id] = report
    if len(hardware_signatures) != 1:
        raise ValueError("World4Drive latency reports contain mixed hardware signatures")
    if len(runtime_signatures) != 1:
        raise ValueError("World4Drive latency reports contain mixed runtime signatures")
    return normalized


def _table_rows(rows: Sequence[object], latency_ids: Sequence[str], reports: Mapping[str, object]) -> list[dict]:
    if len(rows) != len(latency_ids):
        raise ValueError("World4Drive table rows and latency IDs must have identical lengths")
    values = []
    for row, latency_id in zip(rows, latency_ids):
        payload = _json_value(row)
        if not isinstance(payload, dict):
            raise TypeError("World4Drive table row must serialize to a mapping")
        report = reports[latency_id]
        payload.update(
            {
                "latency_id": latency_id,
                "latency_mean_ms": report.mean_ms,
                "latency_p50_ms": report.p50_ms,
                "latency_p95_ms": report.p95_ms,
            }
        )
        values.append(payload)
    return values


def _matrix_rows(matrix: Sequence[Sequence[float | None]], sample_counts: Sequence[int], scene_counts: Sequence[int]):
    return [
        {
            "oracle_horizon": horizon,
            "sample_count": sample_counts[horizon],
            "scene_count": scene_counts[horizon],
            **{f"forced_h{forced}": matrix[horizon][forced] for forced in range(4)},
        }
        for horizon in range(4)
    ]


def _artifact_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_cvoi_world4drive_evaluation(
    output_root: str | Path,
    *,
    config: CvoiWorld4DriveEvaluationConfig,
    inputs: CvoiWorld4DriveEvaluationInputs,
) -> Mapping[str, Path]:
    """Validate complete inputs and write the approved artifact set once."""

    if not isinstance(config, CvoiWorld4DriveEvaluationConfig):
        raise TypeError("config must be CvoiWorld4DriveEvaluationConfig")
    if not isinstance(inputs, CvoiWorld4DriveEvaluationInputs):
        raise TypeError("inputs must be CvoiWorld4DriveEvaluationInputs")
    records = tuple(inputs.records)
    samples = _expected_samples(records)
    records = validate_cvoi_horizon_cache(
        records,
        expected_samples=samples,
        expected_study_points=world4drive_study_points(),
    )
    sample_ids = {sample.sample_id for sample in samples}
    controller_horizons = _validate_controller_horizons(inputs.controller_horizons, sample_ids=sample_ids)
    latency = _validate_latency_reports(inputs.latency_reports, config=config)
    environment = _json_value(inputs.runtime_environment)
    if not isinstance(environment, dict) or not environment:
        raise ValueError("runtime_environment must be a non-empty JSON mapping")

    main_rows = _table_rows(
        build_main_ablation_rows(records, controller_horizons=controller_horizons),
        _MAIN_LATENCY_IDS,
        latency,
    )
    stopping_rows = _table_rows(
        build_stopping_strategy_rows(
            records,
            controller_horizons=controller_horizons["real_cf_value"],
            random_stop_seed=config.random_stop_seed,
        ),
        _STOPPING_LATENCY_IDS,
        latency,
    )
    guidance_rows = _table_rows(
        build_guidance_steps_rows(records, controller_horizons=controller_horizons["real_cf_value"]),
        _GUIDANCE_LATENCY_IDS,
        latency,
    )
    oracle = build_oracle_matrices(records)
    l2_rows = _matrix_rows(oracle.l2_matrix, oracle.sample_counts, oracle.scene_counts)
    collision_rows = _matrix_rows(oracle.collision_matrix, oracle.sample_counts, oracle.scene_counts)

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"World4Drive output directory already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    paths: dict[str, Path] = {}
    paths["records.jsonl"] = write_cvoi_horizon_cache(output / "records.jsonl", records)
    table_specs = (
        ("main_ablation", main_rows),
        ("stopping_strategy", stopping_rows),
        ("guidance_steps_ablation", guidance_rows),
    )
    for name, rows in table_specs:
        paths[f"{name}.json"] = _write_text_exclusive(
            output / f"{name}.json",
            _json_text({"schema": f"cvoi_world4drive_{name}_v1", "rows": rows}),
        )
        paths[f"{name}.csv"] = _write_text_exclusive(
            output / f"{name}.csv",
            _csv_text(rows, tuple(rows[0])),
        )
    matrix_fields = (
        "oracle_horizon",
        "sample_count",
        "scene_count",
        "forced_h0",
        "forced_h1",
        "forced_h2",
        "forced_h3",
    )
    for metric_name, rows in (("l2", l2_rows), ("collision", collision_rows)):
        name = f"oracle_group_{metric_name}_matrix"
        paths[f"{name}.json"] = _write_text_exclusive(
            output / f"{name}.json",
            _json_text({"schema": f"cvoi_world4drive_{name}_v1", "rows": rows}),
        )
        paths[f"{name}.csv"] = _write_text_exclusive(output / f"{name}.csv", _csv_text(rows, matrix_fields))
    latency_payloads = [{"row_id": row_id, **_json_value(latency[row_id])} for row_id in _LATENCY_IDS]
    paths["latency_records.jsonl"] = _write_text_exclusive(
        output / "latency_records.jsonl",
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
            for row in latency_payloads
        ),
    )
    manifest = {
        "schema": "cvoi_world4drive_manifest_v1",
        "protocol": config.protocol,
        "cache_schema": config.cache_schema,
        "metric_authority": "canonical_world4drive_real_only",
        "sample_count": len(samples),
        "scene_count": len({sample.source_scene_id for sample in samples}),
        "record_count": len(records),
        "study_point_count": len(world4drive_study_points()),
        "lambda_compute": config.lambda_compute,
        "random_stop_seed": config.random_stop_seed,
        "runtime_environment": environment,
        "hardware_signature": next(iter(latency.values())).hardware_signature,
        "runtime_signature": next(iter(latency.values())).runtime_signature,
        "checkpoint_signatures": {
            lineage: sorted({record.checkpoint_signature for record in records if record.lineage == lineage})[0]
            for lineage in sorted(config.lineages)
        },
        "dataset_signature": records[0].dataset_signature,
        "code_signature": records[0].code_signature,
        "converter_signature": records[0].converter_signature,
        "evaluator_signature": records[0].evaluator_signature,
        "artifacts": {name: _artifact_digest(path) for name, path in sorted(paths.items())},
    }
    paths["manifest.json"] = _write_text_exclusive(output / "manifest.json", _json_text(manifest))
    return paths


def _load_json(path: Path, *, name: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc


def _load_cache_records(path: Path) -> tuple[CvoiHorizonCacheRecord, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"World4Drive cache does not exist: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"World4Drive cache contains a blank line at {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"World4Drive cache line {line_number} is not valid JSON") from exc
            records.append(parse_cvoi_horizon_cache_record(payload))
    if not records:
        raise ValueError("World4Drive cache must contain at least one record")
    return tuple(records)


def _load_controller_horizons(path: Path) -> Mapping[str, Mapping[str, int]]:
    payload = _load_json(path, name="World4Drive Controller horizon artifact")
    if not isinstance(payload, Mapping) or set(payload) != {"schema", "lineages"}:
        raise ValueError("World4Drive Controller horizon artifact must contain exactly schema and lineages")
    if payload["schema"] != "cvoi_world4drive_controller_horizons_v1":
        raise ValueError("World4Drive Controller horizon artifact has an invalid schema")
    lineages = payload["lineages"]
    if not isinstance(lineages, Mapping):
        raise ValueError("World4Drive Controller horizon lineages must be a mapping")
    return lineages


def _load_latency_reports(path: Path) -> Mapping[str, CvoiWorld4DriveLatencyReport]:
    if not path.is_file():
        raise FileNotFoundError(f"World4Drive latency artifact does not exist: {path}")
    reports = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"World4Drive latency artifact contains a blank line at {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"World4Drive latency line {line_number} is not valid JSON") from exc
            if not isinstance(payload, dict) or "row_id" not in payload:
                raise ValueError(f"World4Drive latency line {line_number} must contain row_id")
            row_id = payload.pop("row_id")
            if not isinstance(row_id, str) or row_id in reports:
                raise ValueError(f"World4Drive latency line {line_number} has a duplicate or invalid row_id")
            try:
                reports[row_id] = CvoiWorld4DriveLatencyReport(**payload)
            except TypeError as exc:
                raise ValueError(f"World4Drive latency line {line_number} has invalid fields") from exc
    return reports


def load_cvoi_world4drive_evaluation_inputs(values: object) -> CvoiWorld4DriveEvaluationInputs:
    """Load the four independently generated, provenance-bearing evaluation inputs."""

    if not isinstance(values, Mapping):
        raise ValueError("cvoi_world4drive_inputs must be a mapping")
    actual = frozenset(values)
    if actual != _INPUT_FIELDS:
        unknown = sorted(actual - _INPUT_FIELDS)
        missing = sorted(_INPUT_FIELDS - actual)
        raise ValueError(f"cvoi_world4drive_inputs fields mismatch: unknown={unknown}, missing={missing}")
    paths = {}
    for name in sorted(_INPUT_FIELDS):
        value = values[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"cvoi_world4drive_inputs.{name} must be a non-empty path")
        paths[name] = Path(value)
    environment = _load_json(paths["runtime_environment_path"], name="World4Drive runtime environment")
    if not isinstance(environment, Mapping):
        raise ValueError("World4Drive runtime environment must be a mapping")
    return CvoiWorld4DriveEvaluationInputs(
        records=_load_cache_records(paths["cache_path"]),
        controller_horizons=_load_controller_horizons(paths["controller_horizons_path"]),
        latency_reports=_load_latency_reports(paths["latency_records_path"]),
        runtime_environment=environment,
    )


def run_world4drive_report(args: Mapping[str, object]) -> Mapping[str, Mapping[str, Path]]:
    """Assemble the six fixed lineage directories and write the report in process."""

    direct = parse_cvoi_world4drive_direct_config(args.get("cvoi_world4drive"))
    if direct.job != "report":
        raise ValueError("evaluate_cvoi_world4drive requires cvoi_world4drive.job='report'")
    root = Path(direct.output_root)
    quality = {lineage: root / "quality" / lineage for lineage in CVOI_WORLD4DRIVE_LINEAGE_ORDER}
    latency = {lineage: root / "latency" / lineage for lineage in CVOI_WORLD4DRIVE_LINEAGE_ORDER}
    assembled_root = root / "assembled"
    report_root = root / "report"
    require_world4drive_inputs_present((*quality.values(), *latency.values()))
    require_world4drive_outputs_absent((assembled_root, report_root))
    assembled = assemble_cvoi_world4drive_inputs(
        assembled_root,
        quality_dirs=quality,
        latency_dirs=latency,
    )
    inputs = load_cvoi_world4drive_evaluation_inputs(
        {
            "cache_path": str(assembled["records.jsonl"]),
            "controller_horizons_path": str(assembled["controller_horizons.json"]),
            "latency_records_path": str(assembled["latency_records.jsonl"]),
            "runtime_environment_path": str(assembled["runtime_environment.json"]),
        }
    )
    report = write_cvoi_world4drive_evaluation(
        report_root,
        config=direct.evaluation_config(),
        inputs=inputs,
    )
    logger.info("[cvoi_world4drive] records=%d output=%s", len(inputs.records), report_root)
    return {"assembled": assembled, "report": report}


def main(args, resume_preempt=False):
    """Run the synchronous direct World4Drive report job."""

    if resume_preempt:
        raise ValueError("evaluate_cvoi_world4drive does not support resume_preempt")
    if not isinstance(args, Mapping):
        raise TypeError("evaluate_cvoi_world4drive args must be a mapping")
    return run_world4drive_report(args)
