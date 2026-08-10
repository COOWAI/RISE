"""Versioned exhaustive horizon cache for CVoI World4Drive evaluation."""

import json
import math
import os
import tempfile
from dataclasses import dataclass, fields
from numbers import Real
from pathlib import Path
from typing import Mapping, Sequence

from app.vjepa_cowa_world_model.training.configs.cvoi_world4drive import (
    CVOI_HORIZON_CACHE_SCHEMA,
    CVOI_WORLD4DRIVE_LAMBDA_COMPUTE,
    CVOI_WORLD4DRIVE_LINEAGES,
)

_TRAJECTORY_STEPS = 6
_SHA256_LENGTH = 64


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_sha256(name: str, value: object) -> str:
    value = _require_identifier(name, value)
    if len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest, got {value!r}")
    return value


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_finite(name: str, value: object, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized) or (minimum is not None and normalized < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}")
    return normalized


def _require_float_tuple(name: str, values: object, *, length: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values, got {values!r}")
    return tuple(_require_finite(f"{name}[{index}]", value, minimum=0.0) for index, value in enumerate(values))


def _require_count_tuple(name: str, values: object, *, length: int) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values, got {values!r}")
    return tuple(_require_int(f"{name}[{index}]", value) for index, value in enumerate(values))


def _require_trajectory(values: object) -> tuple[tuple[float, float, float], ...]:
    if not isinstance(values, (list, tuple)) or len(values) != _TRAJECTORY_STEPS:
        raise ValueError(f"selected_trajectory must contain exactly {_TRAJECTORY_STEPS} poses, got {values!r}")
    poses = []
    for index, pose in enumerate(values):
        if not isinstance(pose, (list, tuple)) or len(pose) != 3:
            raise ValueError(f"selected_trajectory[{index}] must contain exactly 3 values, got {pose!r}")
        poses.append(tuple(_require_finite(f"selected_trajectory[{index}]", value) for value in pose))
    return tuple(poses)


@dataclass(frozen=True)
class CvoiExpectedCacheSample:
    """Expected identity for one Real validation sample."""

    sample_id: str
    source_scene_id: str
    seed: int

    def __post_init__(self) -> None:
        _require_identifier("sample_id", self.sample_id)
        _require_identifier("source_scene_id", self.source_scene_id)
        _require_int("seed", self.seed)


@dataclass(frozen=True)
class CvoiHorizonStudyPoint:
    """One lineage/horizon/guidance combination required in the cache."""

    lineage: str
    horizon: int
    guidance_steps: int

    def __post_init__(self) -> None:
        if self.lineage not in CVOI_WORLD4DRIVE_LINEAGES:
            raise ValueError(f"lineage must be one of {sorted(CVOI_WORLD4DRIVE_LINEAGES)}, got {self.lineage!r}")
        _require_int("horizon", self.horizon)
        if self.horizon > 3:
            raise ValueError(f"horizon must be in [0,3], got {self.horizon}")
        _validate_guidance_steps(self.lineage, self.horizon, self.guidance_steps)


def _validate_guidance_steps(lineage: str, horizon: int, guidance_steps: object) -> int:
    guidance_steps = _require_int("guidance_steps", guidance_steps)
    if horizon == 0 and guidance_steps != 0:
        raise ValueError(f"h=0 must use guidance_steps=0, got {guidance_steps}")
    if lineage == "p0_controller" and guidance_steps != 0:
        raise ValueError(f"p0_controller must use guidance_steps=0, got {guidance_steps}")
    if lineage != "p0_controller" and horizon > 0 and guidance_steps not in {1, 2, 3, 4}:
        raise ValueError(f"positive Value horizon must use guidance_steps in [1,4], got {guidance_steps}")
    return guidance_steps


@dataclass(frozen=True)
class CvoiHorizonCacheRecord:
    """One cached selected trajectory and its canonical Real metrics."""

    schema: str
    dataset_domain: str
    sample_id: str
    source_scene_id: str
    seed: int
    lineage: str
    horizon: int
    guidance_steps: int
    selected_mode: int
    selected_trajectory: tuple[tuple[float, float, float], ...]
    confidence_digest: str
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
    task_score: float
    compute_cost: float
    lambda_compute: float
    checkpoint_signature: str
    dataset_signature: str
    code_signature: str
    converter_signature: str
    evaluator_signature: str

    def __post_init__(self) -> None:
        if self.schema != CVOI_HORIZON_CACHE_SCHEMA:
            raise ValueError(f"schema must be {CVOI_HORIZON_CACHE_SCHEMA!r}, got {self.schema!r}")
        if self.dataset_domain != "real":
            raise ValueError(f"CVoI horizon cache is Real-only, got dataset_domain={self.dataset_domain!r}")
        _require_identifier("sample_id", self.sample_id)
        _require_identifier("source_scene_id", self.source_scene_id)
        _require_int("seed", self.seed)
        CvoiHorizonStudyPoint(self.lineage, self.horizon, self.guidance_steps)
        _require_int("selected_mode", self.selected_mode)
        object.__setattr__(self, "selected_trajectory", _require_trajectory(self.selected_trajectory))
        _require_sha256("confidence_digest", self.confidence_digest)
        object.__setattr__(self, "l2_per_step", _require_float_tuple("l2_per_step", self.l2_per_step, length=6))
        object.__setattr__(
            self,
            "collision_counts",
            _require_count_tuple("collision_counts", self.collision_counts, length=6),
        )
        object.__setattr__(
            self,
            "gt_collision_counts",
            _require_count_tuple("gt_collision_counts", self.gt_collision_counts, length=6),
        )
        for name in (
            "l2_at_1s",
            "l2_at_2s",
            "l2_at_3s",
            "l2_avg",
            "collision_at_1s",
            "collision_at_2s",
            "collision_at_3s",
            "collision_rate",
            "compute_cost",
        ):
            _require_finite(name, getattr(self, name), minimum=0.0)
        task_score = _require_finite("task_score", self.task_score, minimum=0.0)
        if task_score > 1.0:
            raise ValueError(f"task_score must be <= 1.0, got {task_score}")
        lambda_compute = _require_finite("lambda_compute", self.lambda_compute, minimum=0.0)
        if lambda_compute != CVOI_WORLD4DRIVE_LAMBDA_COMPUTE:
            raise ValueError(f"lambda_compute must be exactly {CVOI_WORLD4DRIVE_LAMBDA_COMPUTE}, got {lambda_compute}")
        for name in (
            "checkpoint_signature",
            "dataset_signature",
            "code_signature",
            "converter_signature",
            "evaluator_signature",
        ):
            _require_sha256(name, getattr(self, name))

    @property
    def cache_key(self) -> tuple[str, str, int, str, int, int]:
        return (
            self.sample_id,
            self.source_scene_id,
            self.seed,
            self.lineage,
            self.horizon,
            self.guidance_steps,
        )

    def to_dict(self) -> dict[str, object]:
        return {field.name: _json_value(getattr(self, field.name)) for field in fields(CvoiHorizonCacheRecord)}


def _json_value(value: object) -> object:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


_RECORD_FIELDS = frozenset(field.name for field in fields(CvoiHorizonCacheRecord))


def parse_cvoi_horizon_cache_record(values: Mapping[str, object]) -> CvoiHorizonCacheRecord:
    """Parse one strict cache record without accepting unknown fields."""

    if not isinstance(values, Mapping):
        raise ValueError(f"cache record must be a mapping, got {type(values).__name__}")
    actual = frozenset(values)
    unknown = sorted(actual - _RECORD_FIELDS)
    missing = sorted(_RECORD_FIELDS - actual)
    if unknown:
        raise ValueError(f"cache record contains unknown fields: {unknown}")
    if missing:
        raise ValueError(f"cache record is missing fields: {missing}")
    return CvoiHorizonCacheRecord(**{name: values[name] for name in _RECORD_FIELDS})


def validate_cvoi_horizon_cache(
    records: Sequence[CvoiHorizonCacheRecord],
    *,
    expected_samples: Sequence[CvoiExpectedCacheSample],
    expected_study_points: Sequence[CvoiHorizonStudyPoint],
) -> tuple[CvoiHorizonCacheRecord, ...]:
    """Validate exact sample-by-study-point completeness and provenance."""

    if not expected_samples:
        raise ValueError("expected_samples must not be empty")
    if not expected_study_points:
        raise ValueError("expected_study_points must not be empty")
    sample_keys = [(sample.sample_id, sample.source_scene_id, sample.seed) for sample in expected_samples]
    point_keys = [(point.lineage, point.horizon, point.guidance_steps) for point in expected_study_points]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("expected_samples contains duplicate identities")
    if len(point_keys) != len(set(point_keys)):
        raise ValueError("expected_study_points contains duplicate identities")
    record_keys = [record.cache_key for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("cache contains duplicate record identities")
    expected_keys = {(*sample_key, *point_key) for sample_key in sample_keys for point_key in point_keys}
    actual_keys = set(record_keys)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys
    if missing:
        raise ValueError(f"cache is missing {len(missing)} expected records; first={sorted(missing)[0]!r}")
    if unexpected:
        raise ValueError(f"cache contains {len(unexpected)} unexpected records; first={sorted(unexpected)[0]!r}")
    for signature_name in (
        "dataset_signature",
        "code_signature",
        "converter_signature",
        "evaluator_signature",
    ):
        values = {getattr(record, signature_name) for record in records}
        if len(values) != 1:
            raise ValueError(f"cache contains mixed {signature_name} values: {sorted(values)}")
    checkpoint_by_lineage: dict[str, set[str]] = {}
    for record in records:
        checkpoint_by_lineage.setdefault(record.lineage, set()).add(record.checkpoint_signature)
    mixed_lineages = {lineage: values for lineage, values in checkpoint_by_lineage.items() if len(values) != 1}
    if mixed_lineages:
        raise ValueError(f"cache contains mixed checkpoint_signature values: {mixed_lineages}")
    return tuple(sorted(records, key=lambda record: record.cache_key))


def write_cvoi_horizon_cache(path: str | Path, records: Sequence[CvoiHorizonCacheRecord]) -> Path:
    """Write deterministic JSONL without replacing an existing output."""

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing horizon cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = tuple(sorted(records, key=lambda record: record.cache_key))
    payload = "".join(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n" for record in ordered)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.link(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def read_cvoi_horizon_cache(
    path: str | Path,
    *,
    expected_samples: Sequence[CvoiExpectedCacheSample],
    expected_study_points: Sequence[CvoiHorizonStudyPoint],
) -> tuple[CvoiHorizonCacheRecord, ...]:
    """Read and fully validate one JSONL horizon cache."""

    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"cache contains a blank line at {line_number}")
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"cache line {line_number} is not valid JSON") from exc
            records.append(parse_cvoi_horizon_cache_record(payload))
    return validate_cvoi_horizon_cache(
        records,
        expected_samples=expected_samples,
        expected_study_points=expected_study_points,
    )
