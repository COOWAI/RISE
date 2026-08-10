"""Configuration contract for the CVoI World4Drive evaluation protocol."""

import math
import os
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Mapping, Optional

CVOI_WORLD4DRIVE_PROTOCOL = "cvoi_world4drive_v1"
CVOI_WORLD4DRIVE_DIRECT_PROTOCOL = "cvoi_world4drive_direct_v1"
CVOI_HORIZON_CACHE_SCHEMA = "cvoi_horizon_cache_v1"
CVOI_WORLD4DRIVE_HORIZONS = (0, 1, 2, 3)
CVOI_WORLD4DRIVE_GUIDANCE_STEPS = (1, 2, 3, 4)
CVOI_WORLD4DRIVE_LAMBDA_COMPUTE = 0.05
CVOI_WORLD4DRIVE_LINEAGE_ORDER = (
    "p0_controller",
    "real_only_value",
    "real_cf_value",
)
CVOI_WORLD4DRIVE_LINEAGES = frozenset(CVOI_WORLD4DRIVE_LINEAGE_ORDER)
CVOI_WORLD4DRIVE_DIRECT_JOBS = frozenset({"collect", "latency", "report"})
CVOI_WORLD4DRIVE_FULL_REAL_VAL_PROTOCOL = "cvoi_world4drive_full_real_val_v1"

_COLLECTION_FIELDS = frozenset({"protocol", "audit_manifest_path", "real_val_root"})
_COLLECTION_REAL_ROOT_FIELDS = frozenset(
    {
        "name",
        "dataset_id",
        "domain",
        "data_path",
        "sensor_blobs_path",
        "max_agents",
        "load_agent_annotations",
        "index_cache",
        "window_stride",
        "base_fps",
        "max_frame_gap",
        "max_scenes",
        "annotation_selection",
        "pose_overlay_required",
    }
)

_TOP_LEVEL_FIELDS = frozenset(
    {
        "protocol",
        "cache_schema",
        "horizons",
        "guidance_steps",
        "random_stop_seed",
        "lambda_compute",
        "latency_warmup",
        "latency_repetitions",
        "lineages",
    }
)
_COMMON_ARTIFACT_FIELDS = frozenset(
    {
        "planner_checkpoint",
        "stop_checkpoint",
        "oracle_path",
        "gate_checkpoint",
    }
)
_DIRECT_TOP_LEVEL_FIELDS = frozenset(
    {
        "protocol",
        "job",
        "output_root",
        "checkpoint_audit_manifest_path",
        "dataset_audit_manifest_path",
        "common_artifacts",
        "real_val_root",
        "horizons",
        "guidance_steps",
        "random_stop_seed",
        "lambda_compute",
        "latency_warmup",
        "latency_repetitions",
        "lineages",
    }
)
_DIRECT_COMMON_ARTIFACT_FIELDS = frozenset(
    {
        "world_model_checkpoint",
        "token_ae_checkpoint",
        "unguided_planner_checkpoint",
    }
)


@dataclass(frozen=True)
class CvoiWorld4DriveLineageArtifacts:
    """Artifact paths for one matched World4Drive evaluation lineage."""

    planner_checkpoint: str
    stop_checkpoint: str
    oracle_path: str
    gate_checkpoint: str
    field_checkpoint: Optional[str] = None


@dataclass(frozen=True)
class CvoiWorld4DriveEvaluationConfig:
    """Fully parsed and validated World4Drive evaluation configuration."""

    protocol: str
    cache_schema: str
    horizons: tuple[int, ...]
    guidance_steps: tuple[int, ...]
    random_stop_seed: int
    lambda_compute: float
    latency_warmup: int
    latency_repetitions: int
    lineages: Mapping[str, CvoiWorld4DriveLineageArtifacts]


@dataclass(frozen=True)
class CvoiWorld4DriveCommonArtifacts:
    """Shared read-only checkpoints used by every direct evaluation lineage."""

    world_model_checkpoint: str
    token_ae_checkpoint: str
    unguided_planner_checkpoint: str


@dataclass(frozen=True)
class CvoiWorld4DriveDirectConfig:
    """Immutable three-lineage contract for one direct World4Drive job."""

    protocol: str
    job: str
    output_root: str
    checkpoint_audit_manifest_path: str
    dataset_audit_manifest_path: str
    common_artifacts: CvoiWorld4DriveCommonArtifacts
    real_val_root: Mapping[str, Any]
    horizons: tuple[int, ...]
    guidance_steps: tuple[int, ...]
    random_stop_seed: int
    lambda_compute: float
    latency_warmup: int
    latency_repetitions: int
    lineages: Mapping[str, CvoiWorld4DriveLineageArtifacts]

    def lineage_items(self) -> tuple[tuple[str, CvoiWorld4DriveLineageArtifacts], ...]:
        """Return lineage bindings in the protocol-authoritative execution order."""

        return tuple((name, self.lineages[name]) for name in CVOI_WORLD4DRIVE_LINEAGE_ORDER)

    def evaluation_config(self) -> CvoiWorld4DriveEvaluationConfig:
        """Project the direct contract onto the retained numerical v1 config."""

        return CvoiWorld4DriveEvaluationConfig(
            protocol=CVOI_WORLD4DRIVE_PROTOCOL,
            cache_schema=CVOI_HORIZON_CACHE_SCHEMA,
            horizons=self.horizons,
            guidance_steps=self.guidance_steps,
            random_stop_seed=self.random_stop_seed,
            lambda_compute=self.lambda_compute,
            latency_warmup=self.latency_warmup,
            latency_repetitions=self.latency_repetitions,
            lineages=self.lineages,
        )


@dataclass(frozen=True)
class CvoiWorld4DriveCollectionConfig:
    """Independent full Real validation cohort used only for model-output collection."""

    protocol: str
    audit_manifest_path: str
    real_val_root: Mapping[str, Any]


def _require_exact_fields(values: Mapping[str, Any], expected: frozenset[str], *, name: str) -> None:
    actual = frozenset(values)
    unknown = sorted(actual - expected)
    missing = sorted(expected - actual)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")


def _require_exact_value(name: str, value: Any, expected: Any) -> Any:
    if value != expected or type(value) is not type(expected):
        raise ValueError(f"{name} must be exactly {expected!r}, got {value!r}")
    return value


def _require_exact_sequence(name: str, value: Any, expected: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must be exactly {list(expected)!r}, got {value!r}")
    normalized = tuple(value)
    if normalized != expected:
        raise ValueError(f"{name} must be exactly {list(expected)!r}, got {value!r}")
    return normalized


def _require_positive_int(name: str, value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _require_nonnegative_int(name: str, value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return value


def _require_artifact_path(lineage: str, name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"lineages.{lineage}.{name} must be a non-empty path string, got {value!r}")
    return value


def _require_path(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string, got {value!r}")
    return value


def _require_absolute_path(name: str, value: Any) -> str:
    if not isinstance(value, str) or value != value.strip() or not value or not os.path.isabs(value):
        raise ValueError(f"{name} must be a non-empty absolute path string, got {value!r}")
    return os.path.normpath(value)


def _require_lambda_compute(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be exactly {CVOI_WORLD4DRIVE_LAMBDA_COMPUTE}, got {value!r}")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized != CVOI_WORLD4DRIVE_LAMBDA_COMPUTE:
        raise ValueError(f"{name} must be exactly {CVOI_WORLD4DRIVE_LAMBDA_COMPUTE}, got {value!r}")
    return normalized


def parse_cvoi_world4drive_collection_config(values: Any) -> CvoiWorld4DriveCollectionConfig:
    """Parse the evaluation-only complete Real validation cohort contract."""

    if not isinstance(values, Mapping):
        raise ValueError(f"cvoi_world4drive_collection must be a mapping, got {type(values).__name__}")
    _require_exact_fields(values, _COLLECTION_FIELDS, name="cvoi_world4drive_collection")
    protocol = _require_exact_value(
        "cvoi_world4drive_collection.protocol",
        values["protocol"],
        CVOI_WORLD4DRIVE_FULL_REAL_VAL_PROTOCOL,
    )
    root = values["real_val_root"]
    if not isinstance(root, Mapping):
        raise ValueError("cvoi_world4drive_collection.real_val_root must be a mapping")
    _require_exact_fields(root, _COLLECTION_REAL_ROOT_FIELDS, name="cvoi_world4drive_collection.real_val_root")
    if root["domain"] != "real":
        raise ValueError("cvoi_world4drive_collection.real_val_root.domain must be 'real'")
    if root["load_agent_annotations"] is not True:
        raise ValueError("cvoi_world4drive_collection.real_val_root.load_agent_annotations must be true")
    if root["pose_overlay_required"] is not False:
        raise ValueError("cvoi_world4drive_collection.real_val_root.pose_overlay_required must be false")
    if root["max_scenes"] is not None:
        raise ValueError("cvoi_world4drive_collection.real_val_root.max_scenes must be null")
    if type(root["window_stride"]) is not int or root["window_stride"] not in {1, 4}:
        raise ValueError("cvoi_world4drive_collection.real_val_root.window_stride must be 1 or 4")
    if root["max_agents"] != 256 or type(root["max_agents"]) is not int:
        raise ValueError("cvoi_world4drive_collection.real_val_root.max_agents must be exactly 256")
    if root["annotation_selection"] != "all_valid":
        raise ValueError("cvoi_world4drive_collection.real_val_root.annotation_selection must be 'all_valid'")
    for field in ("name", "dataset_id", "data_path", "sensor_blobs_path"):
        _require_path(f"cvoi_world4drive_collection.real_val_root.{field}", root[field])
    return CvoiWorld4DriveCollectionConfig(
        protocol=protocol,
        audit_manifest_path=_require_path(
            "cvoi_world4drive_collection.audit_manifest_path",
            values["audit_manifest_path"],
        ),
        real_val_root=MappingProxyType(dict(root)),
    )


def _parse_lineage_artifacts(lineage: str, values: Any) -> CvoiWorld4DriveLineageArtifacts:
    if not isinstance(values, Mapping):
        raise ValueError(f"lineages.{lineage} must be a mapping, got {type(values).__name__}")
    expected = (
        _COMMON_ARTIFACT_FIELDS if lineage == "p0_controller" else _COMMON_ARTIFACT_FIELDS | {"field_checkpoint"}
    )
    try:
        _require_exact_fields(values, frozenset(expected), name=f"lineages.{lineage}")
    except ValueError as exc:
        if lineage == "p0_controller" and "field_checkpoint" in values:
            raise ValueError("lineages.p0_controller must not define field_checkpoint") from exc
        if lineage != "p0_controller" and "field_checkpoint" not in values:
            raise ValueError(f"lineages.{lineage} requires field_checkpoint") from exc
        raise
    field_checkpoint = None
    if lineage != "p0_controller":
        field_checkpoint = _require_artifact_path(lineage, "field_checkpoint", values["field_checkpoint"])
    return CvoiWorld4DriveLineageArtifacts(
        planner_checkpoint=_require_artifact_path(lineage, "planner_checkpoint", values["planner_checkpoint"]),
        stop_checkpoint=_require_artifact_path(lineage, "stop_checkpoint", values["stop_checkpoint"]),
        oracle_path=_require_artifact_path(lineage, "oracle_path", values["oracle_path"]),
        gate_checkpoint=_require_artifact_path(lineage, "gate_checkpoint", values["gate_checkpoint"]),
        field_checkpoint=field_checkpoint,
    )


def _parse_direct_common_artifacts(values: Any) -> CvoiWorld4DriveCommonArtifacts:
    if not isinstance(values, Mapping):
        raise ValueError(f"common_artifacts must be a mapping, got {type(values).__name__}")
    _require_exact_fields(values, _DIRECT_COMMON_ARTIFACT_FIELDS, name="common_artifacts")
    return CvoiWorld4DriveCommonArtifacts(
        world_model_checkpoint=_require_absolute_path(
            "common_artifacts.world_model_checkpoint",
            values["world_model_checkpoint"],
        ),
        token_ae_checkpoint=_require_absolute_path(
            "common_artifacts.token_ae_checkpoint",
            values["token_ae_checkpoint"],
        ),
        unguided_planner_checkpoint=_require_absolute_path(
            "common_artifacts.unguided_planner_checkpoint",
            values["unguided_planner_checkpoint"],
        ),
    )


def _parse_direct_real_val_root(values: Any) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError(f"real_val_root must be a mapping, got {type(values).__name__}")
    _require_exact_fields(values, _COLLECTION_REAL_ROOT_FIELDS, name="real_val_root")
    exact_values = {
        "name": "nuscenes_real_val_full",
        "dataset_id": "nuscenes-real-val-full-cvoi-v1",
        "domain": "real",
        "max_agents": 256,
        "load_agent_annotations": True,
        "index_cache": True,
        "window_stride": 4,
        "base_fps": 2,
        "max_frame_gap": 1,
        "max_scenes": None,
        "annotation_selection": "all_valid",
        "pose_overlay_required": False,
    }
    for field, expected in exact_values.items():
        _require_exact_value(f"real_val_root.{field}", values[field], expected)
    normalized = dict(values)
    normalized["data_path"] = _require_absolute_path("real_val_root.data_path", values["data_path"])
    normalized["sensor_blobs_path"] = _require_absolute_path(
        "real_val_root.sensor_blobs_path",
        values["sensor_blobs_path"],
    )
    return MappingProxyType(normalized)


def _parse_direct_lineage_artifacts(lineage: str, values: Any) -> CvoiWorld4DriveLineageArtifacts:
    if not isinstance(values, Mapping):
        raise ValueError(f"lineages.{lineage} must be a mapping, got {type(values).__name__}")
    expected = _COMMON_ARTIFACT_FIELDS
    if lineage != "p0_controller":
        expected = expected | {"field_checkpoint"}
    try:
        _require_exact_fields(values, frozenset(expected), name=f"lineages.{lineage}")
    except ValueError as exc:
        if lineage == "p0_controller" and "field_checkpoint" in values:
            raise ValueError("lineages.p0_controller must not define field_checkpoint") from exc
        if lineage != "p0_controller" and "field_checkpoint" not in values:
            raise ValueError(f"lineages.{lineage} requires field_checkpoint") from exc
        raise
    field_checkpoint = None
    if lineage != "p0_controller":
        field_checkpoint = _require_absolute_path(
            f"lineages.{lineage}.field_checkpoint",
            values["field_checkpoint"],
        )
    return CvoiWorld4DriveLineageArtifacts(
        planner_checkpoint=_require_absolute_path(
            f"lineages.{lineage}.planner_checkpoint",
            values["planner_checkpoint"],
        ),
        stop_checkpoint=_require_absolute_path(
            f"lineages.{lineage}.stop_checkpoint",
            values["stop_checkpoint"],
        ),
        oracle_path=_require_absolute_path(
            f"lineages.{lineage}.oracle_path",
            values["oracle_path"],
        ),
        gate_checkpoint=_require_absolute_path(
            f"lineages.{lineage}.gate_checkpoint",
            values["gate_checkpoint"],
        ),
        field_checkpoint=field_checkpoint,
    )


def parse_cvoi_world4drive_direct_config(values: Any) -> CvoiWorld4DriveDirectConfig:
    """Parse the strict, evaluation-only, three-lineage World4Drive contract."""

    if not isinstance(values, Mapping):
        raise ValueError(f"cvoi_world4drive must be a mapping, got {type(values).__name__}")
    _require_exact_fields(values, _DIRECT_TOP_LEVEL_FIELDS, name="cvoi_world4drive")
    protocol = _require_exact_value("protocol", values["protocol"], CVOI_WORLD4DRIVE_DIRECT_PROTOCOL)
    job = values["job"]
    if type(job) is not str or job not in CVOI_WORLD4DRIVE_DIRECT_JOBS:
        raise ValueError(f"job must be one of {sorted(CVOI_WORLD4DRIVE_DIRECT_JOBS)!r}, got {job!r}")
    common_artifacts = _parse_direct_common_artifacts(values["common_artifacts"])
    raw_lineages = values["lineages"]
    if not isinstance(raw_lineages, Mapping):
        raise ValueError(f"lineages must be a mapping, got {type(raw_lineages).__name__}")
    actual_order = tuple(raw_lineages)
    if actual_order != CVOI_WORLD4DRIVE_LINEAGE_ORDER:
        raise ValueError(
            f"lineages must have exactly this order: {CVOI_WORLD4DRIVE_LINEAGE_ORDER!r}, got {actual_order!r}"
        )
    lineages = MappingProxyType(
        {
            lineage: _parse_direct_lineage_artifacts(lineage, raw_lineages[lineage])
            for lineage in CVOI_WORLD4DRIVE_LINEAGE_ORDER
        }
    )
    p0_checkpoint = lineages["p0_controller"].planner_checkpoint
    if p0_checkpoint != common_artifacts.unguided_planner_checkpoint:
        raise ValueError(
            "lineages.p0_controller.planner_checkpoint must equal "
            "common_artifacts.unguided_planner_checkpoint after path normalization"
        )
    checkpoint_audit_manifest_path = _require_absolute_path(
        "checkpoint_audit_manifest_path",
        values["checkpoint_audit_manifest_path"],
    )
    dataset_audit_manifest_path = _require_absolute_path(
        "dataset_audit_manifest_path",
        values["dataset_audit_manifest_path"],
    )
    if checkpoint_audit_manifest_path == dataset_audit_manifest_path:
        raise ValueError("checkpoint and dataset audit manifest paths must be distinct")
    return CvoiWorld4DriveDirectConfig(
        protocol=protocol,
        job=job,
        output_root=_require_absolute_path("output_root", values["output_root"]),
        checkpoint_audit_manifest_path=checkpoint_audit_manifest_path,
        dataset_audit_manifest_path=dataset_audit_manifest_path,
        common_artifacts=common_artifacts,
        real_val_root=_parse_direct_real_val_root(values["real_val_root"]),
        horizons=_require_exact_sequence("horizons", values["horizons"], CVOI_WORLD4DRIVE_HORIZONS),
        guidance_steps=_require_exact_sequence(
            "guidance_steps",
            values["guidance_steps"],
            CVOI_WORLD4DRIVE_GUIDANCE_STEPS,
        ),
        random_stop_seed=_require_nonnegative_int("random_stop_seed", values["random_stop_seed"]),
        lambda_compute=_require_lambda_compute("lambda_compute", values["lambda_compute"]),
        latency_warmup=_require_positive_int("latency_warmup", values["latency_warmup"]),
        latency_repetitions=_require_positive_int("latency_repetitions", values["latency_repetitions"]),
        lineages=lineages,
    )


def parse_cvoi_world4drive_evaluation_config(values: Any) -> CvoiWorld4DriveEvaluationConfig:
    """Parse the standalone CVoI World4Drive evaluation section."""

    if not isinstance(values, Mapping):
        raise ValueError(f"cvoi_world4drive_evaluation must be a mapping, got {type(values).__name__}")
    _require_exact_fields(values, _TOP_LEVEL_FIELDS, name="cvoi_world4drive_evaluation")
    protocol = _require_exact_value("protocol", values["protocol"], CVOI_WORLD4DRIVE_PROTOCOL)
    cache_schema = _require_exact_value("cache_schema", values["cache_schema"], CVOI_HORIZON_CACHE_SCHEMA)
    horizons = _require_exact_sequence("horizons", values["horizons"], CVOI_WORLD4DRIVE_HORIZONS)
    guidance_steps = _require_exact_sequence(
        "guidance_steps",
        values["guidance_steps"],
        CVOI_WORLD4DRIVE_GUIDANCE_STEPS,
    )
    lambda_compute = _require_lambda_compute("lambda_compute", values["lambda_compute"])
    raw_lineages = values["lineages"]
    if not isinstance(raw_lineages, Mapping):
        raise ValueError(f"lineages must be a mapping, got {type(raw_lineages).__name__}")
    actual_lineages = frozenset(raw_lineages)
    if actual_lineages != CVOI_WORLD4DRIVE_LINEAGES:
        raise ValueError(
            f"lineages must be exactly {sorted(CVOI_WORLD4DRIVE_LINEAGES)!r}, got {sorted(actual_lineages)!r}"
        )
    lineages = {
        lineage: _parse_lineage_artifacts(lineage, raw_lineages[lineage])
        for lineage in sorted(CVOI_WORLD4DRIVE_LINEAGES)
    }
    return CvoiWorld4DriveEvaluationConfig(
        protocol=protocol,
        cache_schema=cache_schema,
        horizons=horizons,
        guidance_steps=guidance_steps,
        random_stop_seed=_require_nonnegative_int("random_stop_seed", values["random_stop_seed"]),
        lambda_compute=lambda_compute,
        latency_warmup=_require_positive_int("latency_warmup", values["latency_warmup"]),
        latency_repetitions=_require_positive_int("latency_repetitions", values["latency_repetitions"]),
        lineages=MappingProxyType(lineages),
    )
