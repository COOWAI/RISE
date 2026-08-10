"""Assemble matched World4Drive quality and latency collections for reporting."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from app.vjepa_cowa_world_model.training.cvoi_horizon_cache import (
    CvoiExpectedCacheSample,
    CvoiHorizonCacheRecord,
    CvoiHorizonStudyPoint,
    parse_cvoi_horizon_cache_record,
    validate_cvoi_horizon_cache,
    write_cvoi_horizon_cache,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_latency_collection import world4drive_latency_row_ids
from app.vjepa_cowa_world_model.training.navsim_cvoi_world4drive import CvoiWorld4DriveLatencyReport

_LINEAGES = ("p0_controller", "real_only_value", "real_cf_value")
_LATENCY_ROW_IDS = tuple(row_id for lineage in _LINEAGES for row_id in world4drive_latency_row_ids(lineage))
_QUALITY_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "dataset_domain",
        "lineage",
        "sample_count",
        "scene_count",
        "record_count",
        "study_points",
        "provenance",
        "artifacts",
    }
)
_LATENCY_MANIFEST_FIELDS = frozenset(
    {
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
)
_SHARED_QUALITY_SIGNATURES = (
    "dataset_signature",
    "code_signature",
    "converter_signature",
    "evaluator_signature",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc


def _require_lineage_dirs(values: Mapping[str, str | Path], *, name: str) -> dict[str, Path]:
    if not isinstance(values, Mapping) or tuple(sorted(values)) != tuple(sorted(_LINEAGES)):
        raise ValueError(f"{name} must contain exactly the three World4Drive lineages")
    paths = {}
    for lineage in _LINEAGES:
        path = Path(values[lineage])
        if not path.is_dir():
            raise FileNotFoundError(f"{name}[{lineage!r}] does not exist: {path}")
        paths[lineage] = path
    return paths


def _require_artifact_hashes(directory: Path, values: object, *, expected: Sequence[str], name: str) -> None:
    if not isinstance(values, Mapping) or set(values) != set(expected):
        raise ValueError(f"{name} artifact map must contain exactly {sorted(expected)!r}")
    for filename in expected:
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"{name} artifact does not exist: {path}")
        if values[filename] != _sha256(path):
            raise ValueError(f"{name} artifact SHA256 mismatch: {path}")


def _study_points(lineage: str) -> tuple[CvoiHorizonStudyPoint, ...]:
    if lineage == "p0_controller":
        return tuple(CvoiHorizonStudyPoint(lineage, horizon, 0) for horizon in range(4))
    if lineage == "real_only_value":
        return tuple(CvoiHorizonStudyPoint(lineage, horizon, 0 if horizon == 0 else 2) for horizon in range(4))
    if lineage == "real_cf_value":
        return (CvoiHorizonStudyPoint(lineage, 0, 0),) + tuple(
            CvoiHorizonStudyPoint(lineage, horizon, guidance_steps)
            for horizon in (1, 2, 3)
            for guidance_steps in (1, 2, 3, 4)
        )
    raise ValueError(f"unsupported World4Drive lineage: {lineage!r}")


def _load_records(path: Path) -> tuple[CvoiHorizonCacheRecord, ...]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"World4Drive quality cache contains a blank line at {line_number}: {path}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"World4Drive quality cache line {line_number} is invalid: {path}") from exc
            records.append(parse_cvoi_horizon_cache_record(payload))
    if not records:
        raise ValueError(f"World4Drive quality cache is empty: {path}")
    return tuple(records)


def _load_quality_collection(directory: Path, *, lineage: str):
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path, name=f"World4Drive {lineage} quality manifest")
    if not isinstance(manifest, Mapping) or frozenset(manifest) != _QUALITY_MANIFEST_FIELDS:
        raise ValueError(f"World4Drive {lineage} quality manifest fields are invalid")
    if (
        manifest["schema"] != "cvoi_world4drive_collection_v1"
        or manifest["dataset_domain"] != "real"
        or manifest["lineage"] != lineage
    ):
        raise ValueError(f"World4Drive {lineage} quality manifest identity is invalid")
    _require_artifact_hashes(
        directory,
        manifest["artifacts"],
        expected=("records.jsonl", "controller_horizons.json"),
        name=f"World4Drive {lineage} quality collection",
    )
    records = _load_records(directory / "records.jsonl")
    identities = sorted({(record.sample_id, record.source_scene_id, record.seed) for record in records})
    samples = tuple(CvoiExpectedCacheSample(*identity) for identity in identities)
    records = validate_cvoi_horizon_cache(
        records,
        expected_samples=samples,
        expected_study_points=_study_points(lineage),
    )
    if manifest["sample_count"] != len(samples) or manifest["record_count"] != len(records):
        raise ValueError(f"World4Drive {lineage} quality manifest counts do not match its cache")
    provenance = manifest["provenance"]
    expected_provenance = {
        name: getattr(records[0], name) for name in ("checkpoint_signature",) + _SHARED_QUALITY_SIGNATURES
    }
    if provenance != expected_provenance:
        raise ValueError(f"World4Drive {lineage} quality provenance does not match its cache")
    horizon_payload = _load_json(
        directory / "controller_horizons.json",
        name=f"World4Drive {lineage} Controller horizons",
    )
    if (
        not isinstance(horizon_payload, Mapping)
        or set(horizon_payload) != {"schema", "lineage", "horizons"}
        or horizon_payload["schema"] != "cvoi_world4drive_controller_horizons_v1"
        or horizon_payload["lineage"] != lineage
        or not isinstance(horizon_payload["horizons"], Mapping)
    ):
        raise ValueError(f"World4Drive {lineage} Controller horizon artifact is invalid")
    horizons = dict(horizon_payload["horizons"])
    sample_ids = {sample.sample_id for sample in samples}
    if set(horizons) != sample_ids or any(
        type(value) is not int or value not in range(4) for value in horizons.values()
    ):
        raise ValueError(f"World4Drive {lineage} Controller horizons do not match its cache")
    return records, samples, horizons, expected_provenance, manifest_path


def _require_finite_latency(report: CvoiWorld4DriveLatencyReport, *, row_id: str) -> None:
    values = (
        *report.samples_ms,
        report.mean_ms,
        report.p50_ms,
        report.p95_ms,
        *report.component_mean_ms.values(),
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError(f"World4Drive latency row {row_id!r} contains a non-finite value")


def _load_latency_collection(directory: Path, *, lineage: str):
    manifest_path = directory / "manifest.json"
    manifest = _load_json(manifest_path, name=f"World4Drive {lineage} latency manifest")
    if not isinstance(manifest, Mapping) or frozenset(manifest) != _LATENCY_MANIFEST_FIELDS:
        raise ValueError(f"World4Drive {lineage} latency manifest fields are invalid")
    expected_rows = world4drive_latency_row_ids(lineage)
    if (
        manifest["schema"] != "cvoi_world4drive_latency_collection_v1"
        or manifest["lineage"] != lineage
        or tuple(manifest["row_ids"]) != expected_rows
    ):
        raise ValueError(f"World4Drive {lineage} latency manifest identity is invalid")
    _require_artifact_hashes(
        directory,
        manifest["artifacts"],
        expected=("latency_records.jsonl", "runtime_environment.json"),
        name=f"World4Drive {lineage} latency collection",
    )
    reports = {}
    with (directory / "latency_records.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"World4Drive latency contains a blank line at {line_number}: {directory}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"World4Drive latency line {line_number} is invalid: {directory}") from exc
            if not isinstance(payload, dict) or "row_id" not in payload:
                raise ValueError(f"World4Drive latency line {line_number} is missing row_id: {directory}")
            row_id = payload.pop("row_id")
            if row_id in reports:
                raise ValueError(f"World4Drive latency row {row_id!r} is duplicated")
            try:
                report = CvoiWorld4DriveLatencyReport(**payload)
            except TypeError as exc:
                raise ValueError(f"World4Drive latency row {row_id!r} fields are invalid") from exc
            _require_finite_latency(report, row_id=row_id)
            reports[row_id] = report
    if tuple(reports) != expected_rows:
        raise ValueError(f"World4Drive {lineage} latency rows are incomplete or reordered")
    environment = _load_json(
        directory / "runtime_environment.json",
        name=f"World4Drive {lineage} runtime environment",
    )
    if not isinstance(environment, Mapping):
        raise ValueError(f"World4Drive {lineage} runtime environment must be a mapping")
    if (
        environment.get("hardware_signature") != manifest["hardware_signature"]
        or environment.get("runtime_signature") != manifest["runtime_signature"]
    ):
        raise ValueError(f"World4Drive {lineage} runtime environment signatures do not match its manifest")
    sample_ids = tuple(manifest["sample_ids"])
    if (
        len(sample_ids) != manifest["warmup"] + manifest["repetitions"]
        or len(sample_ids) != len(set(sample_ids))
        or any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids)
    ):
        raise ValueError(f"World4Drive {lineage} latency sample IDs are invalid")
    for row_id, report in reports.items():
        if (
            report.warmup != manifest["warmup"]
            or report.repetitions != manifest["repetitions"]
            or report.hardware_signature != manifest["hardware_signature"]
            or report.runtime_signature != manifest["runtime_signature"]
        ):
            raise ValueError(f"World4Drive latency row {row_id!r} does not match its manifest")
    return reports, sample_ids, dict(environment), manifest, manifest_path


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n"


def assemble_cvoi_world4drive_inputs(
    output_dir: str | Path,
    *,
    quality_dirs: Mapping[str, str | Path],
    latency_dirs: Mapping[str, str | Path],
) -> Mapping[str, Path]:
    """Validate and assemble the six formal lineage collections exactly once."""

    output = Path(output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"World4Drive assembly output already exists: {output}")
    quality_paths = _require_lineage_dirs(quality_dirs, name="quality_dirs")
    latency_paths = _require_lineage_dirs(latency_dirs, name="latency_dirs")

    all_records = []
    controller_horizons = {}
    quality_samples = None
    shared_quality_signatures = None
    source_manifest_digests = {"quality": {}, "latency": {}}
    for lineage in _LINEAGES:
        records, samples, horizons, provenance, manifest_path = _load_quality_collection(
            quality_paths[lineage],
            lineage=lineage,
        )
        identities = tuple((sample.sample_id, sample.source_scene_id, sample.seed) for sample in samples)
        if quality_samples is None:
            quality_samples = identities
        elif identities != quality_samples:
            raise ValueError("World4Drive quality lineages contain different matched sample identities")
        signatures = tuple(provenance[name] for name in _SHARED_QUALITY_SIGNATURES)
        if shared_quality_signatures is None:
            shared_quality_signatures = signatures
        elif signatures != shared_quality_signatures:
            raise ValueError("World4Drive quality lineages contain mismatched dataset/code/evaluator signatures")
        all_records.extend(records)
        controller_horizons[lineage] = horizons
        source_manifest_digests["quality"][lineage] = _sha256(manifest_path)

    latency_reports = {}
    latency_sample_ids = None
    latency_environment = None
    latency_identity = None
    for lineage in _LINEAGES:
        reports, sample_ids, environment, manifest, manifest_path = _load_latency_collection(
            latency_paths[lineage],
            lineage=lineage,
        )
        identity = (
            manifest["warmup"],
            manifest["repetitions"],
            manifest["hardware_signature"],
            manifest["runtime_signature"],
        )
        if latency_sample_ids is None:
            latency_sample_ids = sample_ids
            latency_environment = environment
            latency_identity = identity
        else:
            if sample_ids != latency_sample_ids:
                raise ValueError("World4Drive latency lineages contain different matched sample IDs")
            if identity[2] != latency_identity[2]:
                raise ValueError("World4Drive latency lineages contain mixed hardware signatures")
            if identity[3] != latency_identity[3]:
                raise ValueError("World4Drive latency lineages contain mixed runtime signatures")
            if identity[:2] != latency_identity[:2] or environment != latency_environment:
                raise ValueError("World4Drive latency lineages use different timing protocols")
        overlap = set(latency_reports).intersection(reports)
        if overlap:
            raise ValueError(f"World4Drive latency row IDs are duplicated across lineages: {sorted(overlap)!r}")
        latency_reports.update(reports)
        source_manifest_digests["latency"][lineage] = _sha256(manifest_path)
    if tuple(latency_reports) != _LATENCY_ROW_IDS:
        raise ValueError("World4Drive latency collections do not contain the exact ten formal rows")

    output.mkdir(parents=True, exist_ok=False)
    paths = {
        "records.jsonl": write_cvoi_horizon_cache(output / "records.jsonl", tuple(all_records)),
    }
    controller_path = output / "controller_horizons.json"
    controller_path.write_text(
        _json_text(
            {
                "schema": "cvoi_world4drive_controller_horizons_v1",
                "lineages": controller_horizons,
            }
        ),
        encoding="utf-8",
    )
    paths["controller_horizons.json"] = controller_path
    latency_path = output / "latency_records.jsonl"
    with latency_path.open("x", encoding="utf-8") as handle:
        for row_id in _LATENCY_ROW_IDS:
            handle.write(
                json.dumps(
                    {"row_id": row_id, **asdict(latency_reports[row_id])},
                    ensure_ascii=True,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    paths["latency_records.jsonl"] = latency_path
    environment_path = output / "runtime_environment.json"
    environment_path.write_text(_json_text(latency_environment), encoding="utf-8")
    paths["runtime_environment.json"] = environment_path
    manifest = {
        "schema": "cvoi_world4drive_input_assembly_v1",
        "quality_sample_ids": [identity[0] for identity in quality_samples],
        "latency_sample_ids": list(latency_sample_ids),
        "quality_record_count": len(all_records),
        "latency_row_ids": list(_LATENCY_ROW_IDS),
        "shared_quality_signatures": dict(zip(_SHARED_QUALITY_SIGNATURES, shared_quality_signatures)),
        "hardware_signature": latency_identity[2],
        "runtime_signature": latency_identity[3],
        "source_manifests": source_manifest_digests,
        "artifacts": {name: _sha256(path) for name, path in sorted(paths.items())},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(_json_text(manifest), encoding="utf-8")
    paths["manifest.json"] = manifest_path
    return paths
