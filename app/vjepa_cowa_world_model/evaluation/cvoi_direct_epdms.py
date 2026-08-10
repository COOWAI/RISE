"""Strict configuration projections for one direct NavSim CVoI EPDMS run.

The public YAML describes one retained manual-chain branch.  It is never
passed to the NavSim Agent.  Instead, :func:`project_cvoi_direct_epdms_run`
creates one mode-specific effective projection.  For controller runs this is
also the only boundary that opens the Oracle: the projection contains its
validated digest, never its path.
"""

from __future__ import annotations

import csv
import errno
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import IO, Callable, Iterator, Mapping

import torch
import yaml

from app.vjepa_cowa_world_model.training import cvoi_manual_lineage
from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_runtime import (
    read_cvoi_direct_epdms_gate_checkpoint,
    read_cvoi_direct_epdms_planner_checkpoint,
    resolve_cvoi_direct_epdms_artifact_identity,
    resolve_formal_v2_navsim_e120_selected_checkpoint,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import V2_PROTOCOL_ID, get_cvoi_navsim_metric_protocol
from app.vjepa_cowa_world_model.training.cvoi_navsim_scores import _csv_components, _csv_number
from app.vjepa_cowa_world_model.training.cvoi_value import read_cvoi_navsim_e120_direct_value_checkpoint
from src.models.vision_transformer import VIT_EMBED_DIMS

CVOI_DIRECT_EPDMS_SCHEMA = "cvoi_direct_epdms"
CVOI_DIRECT_EPDMS_VERSION = 1
CVOI_DIRECT_EPDMS_EFFECTIVE_SCHEMA = "cvoi_direct_epdms_effective"
CVOI_DIRECT_EPDMS_EFFECTIVE_VERSION = 1
CVOI_DIRECT_EPDMS_TRACE_SCHEMA = "cvoi_direct_epdms_trace"
CVOI_DIRECT_EPDMS_TRACE_VERSION = 1
CVOI_DIRECT_EPDMS_SCENE_SCHEMA = "cvoi_direct_epdms_scene_v1"
CVOI_DIRECT_EPDMS_SUMMARY_SCHEMA = "cvoi_direct_epdms_summary_v1"

CVOI_DIRECT_EPDMS_BRANCHES = (
    "full",
    "no_cf",
    "hazard_only",
    "quality_only",
    "without_field",
    "without_stop",
    "without_value_summary",
)

_CONTROLLER_BRANCHES = frozenset(
    {
        "full",
        "no_cf",
        "without_field",
        "without_stop",
        "without_value_summary",
    }
)
_FORCED_BRANCHES = frozenset({"hazard_only", "quality_only"})
_GATE_FEATURE_MODE_BY_BRANCH = {
    "full": "full",
    "no_cf": "full",
    "without_field": "without_field",
    "without_stop": "without_stop",
    "without_value_summary": "without_value_summary",
}
_ORACLE_LINEAGE_BY_BRANCH = {
    "full": "p1_full",
    "no_cf": "p1_no_cf",
    "without_field": "p1_full",
    "without_stop": "p1_full",
    "without_value_summary": "p1_full",
}

_SPLIT = "navtest"
_PROTOCOL = V2_PROTOCOL_ID
_MAX_HORIZON = 4
_GUIDANCE_STEPS = 2
_GATE_CHECKPOINT_SCHEMA = "sequential_cvoi_gate_navsim_e120_v1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_REPOSITORY_TRAINING_CONFIG = Path("configs/train/navsim/cvoi_manual_full/05_p1_full.yaml")
_UNEDITED_PATH_ROOT = Path("/path/to")
_FULL_HANDOFF_NAME_BY_ARTIFACT_FIELD = {
    "p0_planner_checkpoint_path": "p0_handoff",
    "calibration_checkpoint_path": "calibration_handoff",
    "p1_planner_checkpoint_path": "p1_handoff",
    "stop_checkpoint_path": "stop_handoff",
    "gate_checkpoint_path": "gate_handoff",
    "oracle_path": "oracle_handoff",
}

_DIRECT_RECORD_FIELDS = frozenset(
    {
        "schema",
        "branch",
        "scenario_token",
        "evaluation_seed",
        "epdms",
        "final_horizon",
        "latency_ms",
    }
)
_DIRECT_TRACE_FIELDS = frozenset(
    {
        "schema",
        "version",
        "split",
        "protocol",
        "branch",
        "scenario_token",
        "evaluation_seed",
        "final_horizon",
        "latency_ms",
    }
)
_DIRECT_SCORE_COMPONENTS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
)
_DIRECT_SCORE_FIELDS = frozenset({"token", "valid", "score", *_DIRECT_SCORE_COMPONENTS})
_DIRECT_SCORE_INDEX_HEADERS = frozenset({"", "Unnamed: 0"})
_DIRECT_ROOT_REQUIRED_ENTRIES = frozenset({"scorer_output", "policy_traces"})
_DIRECT_ROOT_OPTIONAL_ENTRIES = frozenset({"cvoi_direct_epdms_effective.json"})
_DIRECT_OUTPUT_NAMES = ("records.jsonl", "summary.json")
_DIRECT_SUBPROCESS_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "CUDA_HOME",
        "CONDA_PREFIX",
        "VIRTUAL_ENV",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "TMPDIR",
        "OPENSCENE_DATA_ROOT",
        "NAVSIM_EXP_ROOT",
        "NUPLAN_MAPS_ROOT",
        "NAVSIM_DEVKIT_ROOT",
        "METRIC_CACHE_PATH",
        "PYTHON_BIN",
    }
)

_PUBLIC_KEYS = frozenset(
    {
        "schema",
        "version",
        "branch",
        "run_kind",
        "split",
        "protocol",
        "max_horizon",
        "guidance_steps",
        "training_config_path",
        "encoder_checkpoint_path",
        "scenario_manifest_path",
        "output_root",
        "artifacts",
    }
)
_CONTROLLER_ARTIFACT_KEYS = frozenset(
    {
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
        "stop_checkpoint_path",
        "gate_checkpoint_path",
        "oracle_path",
        "gate_feature_mode",
    }
)
_FORCED_ARTIFACT_KEYS = frozenset(
    {
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
    }
)
_EFFECTIVE_COMMON_KEYS = frozenset(
    {
        "schema",
        "version",
        "branch",
        "split",
        "protocol",
        "evaluation_mode",
        "horizon",
        "guidance_steps",
        "training_config_path",
        "encoder_checkpoint_path",
        "scenario_manifest_path",
        "output_directory",
    }
)
_EFFECTIVE_ARTIFACT_KEYS = {
    "controller": frozenset(
        {
            "p0_planner_checkpoint_path",
            "calibration_checkpoint_path",
            "p1_planner_checkpoint_path",
            "stop_checkpoint_path",
            "gate_checkpoint_path",
            "gate_feature_mode",
            "oracle_sha256",
        }
    ),
    "p0_forced": frozenset({"p0_planner_checkpoint_path"}),
    "p1_field_forced": frozenset(
        {
            "calibration_checkpoint_path",
            "p1_planner_checkpoint_path",
        }
    ),
}
_HEX_DIGITS = frozenset("0123456789abcdef")
_SCENARIO_ROW_FIELDS = frozenset(
    {
        "schema",
        "protocol_id",
        "scenario_token",
        "observation_key",
        "log_name",
        "current_camera_data_path",
    }
)
_SCENARIO_ROW_SCHEMA = "cvoi_navsim_scenario_v1"
_SCENARIO_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,198}[A-Za-z0-9])?")


@dataclass(frozen=True)
class CvoiDirectEpdmsArtifacts:
    """Exact public artifact fields for one controller or forced branch."""

    p0_planner_checkpoint_path: Path
    calibration_checkpoint_path: Path
    p1_planner_checkpoint_path: Path
    stop_checkpoint_path: Path | None = None
    gate_checkpoint_path: Path | None = None
    oracle_path: Path | None = None
    gate_feature_mode: str | None = None


@dataclass(frozen=True)
class CvoiDirectEpdmsConfig:
    """Strictly parsed public YAML configuration."""

    schema: str
    version: int
    branch: str
    run_kind: str
    split: str
    protocol: str
    max_horizon: int
    guidance_steps: int
    training_config_path: Path
    encoder_checkpoint_path: Path
    scenario_manifest_path: Path
    output_root: Path
    artifacts: CvoiDirectEpdmsArtifacts


@dataclass(frozen=True)
class CvoiDirectEpdmsProjection:
    """One immutable, mode-specific Agent input projection."""

    branch: str
    split: str
    protocol: str
    evaluation_mode: str
    horizon: int | None
    guidance_steps: int
    training_config_path: Path
    encoder_checkpoint_path: Path
    scenario_manifest_path: Path
    output_directory: Path
    p0_planner_checkpoint_path: Path | None = None
    calibration_checkpoint_path: Path | None = None
    p1_planner_checkpoint_path: Path | None = None
    stop_checkpoint_path: Path | None = None
    gate_checkpoint_path: Path | None = None
    gate_feature_mode: str | None = None
    oracle_sha256: str | None = None


@dataclass(frozen=True)
class DirectEpdmsSceneRecord:
    """One scored NavTest scene from a direct seven-branch CVoI run."""

    branch: str
    scenario_token: str
    evaluation_seed: int
    epdms: float
    final_horizon: int
    latency_ms: float

    def __post_init__(self) -> None:
        _require_direct_branch(self.branch)
        _require_navtest_scenario_token(self.scenario_token)
        _require_evaluation_seed(self.evaluation_seed)
        object.__setattr__(self, "epdms", _require_unit_float(self.epdms, field="epdms"))
        _require_final_horizon(self.final_horizon)
        object.__setattr__(
            self,
            "latency_ms",
            _require_finite_float(self.latency_ms, field="latency_ms", nonnegative=True),
        )


@dataclass(frozen=True)
class _DirectEpdmsTrace:
    branch: str
    scenario_token: str
    evaluation_seed: int
    final_horizon: int
    latency_ms: float


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ValueError("CVoI direct EPDMS YAML mapping keys must be scalar") from error
        if duplicate:
            raise ValueError(f"duplicate CVoI direct EPDMS YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _require_mapping(value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{context} keys must be strings")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: frozenset[str], *, context: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{context} fields mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _require_exact_string(value: object, *, field: str, expected: str | None = None) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if expected is not None and value != expected:
        raise ValueError(f"{field} must be exactly {expected!r}, got {value!r}")
    return value


def _require_exact_integer(value: object, *, field: str, expected: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer, got {value!r}")
    if expected is not None and value != expected:
        raise ValueError(f"{field} must be exactly {expected}, got {value!r}")
    return value


def _require_direct_branch(value: object) -> str:
    branch = _require_exact_string(value, field="branch")
    if branch not in CVOI_DIRECT_EPDMS_BRANCHES:
        raise ValueError(f"branch must be one of {list(CVOI_DIRECT_EPDMS_BRANCHES)}, got {branch!r}")
    return branch


def validate_cvoi_direct_epdms_scenario_token(value: object) -> str:
    """Return one path-safe NavTest token suitable for the trace filename."""

    token = _require_exact_string(value, field="scenario_token")
    if token != token.strip():
        raise ValueError("scenario_token must not contain surrounding whitespace")
    if token.lower().startswith("navtrain"):
        raise ValueError(f"scenario_token must identify a NavTest scene, not NavTrain: {token!r}")
    if _SCENARIO_TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError(f"scenario_token must be path-safe for direct EPDMS traces: {token!r}")
    return token


def _require_navtest_scenario_token(value: object) -> str:
    return validate_cvoi_direct_epdms_scenario_token(value)


def _require_evaluation_seed(value: object) -> int:
    seed = _require_exact_integer(value, field="evaluation_seed")
    if seed < 0:
        raise ValueError("evaluation_seed must be non-negative")
    return seed


def _require_final_horizon(value: object) -> int:
    horizon = _require_exact_integer(value, field="final_horizon")
    if not 0 <= horizon <= _MAX_HORIZON:
        raise ValueError(f"final_horizon must be an integer in H0--H{_MAX_HORIZON}, got {horizon!r}")
    return horizon


def _require_finite_float(value: object, *, field: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _require_unit_float(value: object, *, field: str) -> float:
    result = _require_finite_float(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _absolute_normalized_path(value: object, *, field: str) -> Path:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be a non-empty absolute path string")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain NUL bytes")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute: {path}")
    if str(path) != value or ".." in path.parts:
        raise ValueError(f"{field} must be a normalized absolute path: {value!r}")
    return path


def _repository_training_config_path(value: object) -> Path:
    """Resolve the one repository-owned public training config from the repository root."""

    if type(value) is not str or not value:
        raise ValueError("training_config_path must be a non-empty path string")
    path = Path(value)
    if path.is_absolute():
        return _absolute_normalized_path(value, field="training_config_path")
    if value != _REPOSITORY_TRAINING_CONFIG.as_posix():
        raise ValueError(
            "repository-relative training_config_path must be exactly "
            f"{_REPOSITORY_TRAINING_CONFIG.as_posix()!r}, got {value!r}"
        )
    return _REPOSITORY_ROOT / _REPOSITORY_TRAINING_CONFIG


def _canonical_alias_key(path: Path) -> Path:
    return path.resolve(strict=False)


def _reject_path_aliases(paths: Mapping[str, Path]) -> None:
    identities: dict[Path, str] = {}
    for field, path in paths.items():
        identity = _canonical_alias_key(path)
        previous = identities.get(identity)
        if previous is not None:
            raise ValueError(f"path fields {previous!r} and {field!r} alias the same path: {identity}")
        identities[identity] = field


def _read_regular_nonsymlink_text(path: Path, *, context: str) -> str:
    with _open_regular_nonsymlink_binary(path, context=context) as handle:
        data = handle.read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{context} must be UTF-8 text: {path}") from error


def _resolve_full_results_root_from_artifacts(artifact_paths: Mapping[str, Path]) -> Path:
    handoffs = {
        handoff_name: path
        for field, handoff_name in _FULL_HANDOFF_NAME_BY_ARTIFACT_FIELD.items()
        if (path := artifact_paths.get(field)) is not None
    }
    return cvoi_manual_lineage.resolve_cvoi_manual_full_results_root(handoffs)


def _full_results_root_from_validated_projection_paths(artifact_paths: Mapping[str, Path]) -> Path:
    """Recover the Full root after ``_direct_projection_artifact_paths`` validated suffixes."""

    p0_path = artifact_paths.get("p0_planner_checkpoint_path")
    if p0_path is None or not p0_path.is_absolute():
        raise RuntimeError("validated Full direct EPDMS paths lost the absolute P0 handoff")
    return p0_path.parent.parent


def _manual_artifact_authority(
    *,
    branch: str,
    run_kind: str,
) -> dict[str, Path]:
    """Return the exact retained handoff paths for one public branch."""

    full_root = cvoi_manual_lineage.CVOI_MANUAL_FULL_RESULTS_ROOT
    ablation_root = cvoi_manual_lineage.CVOI_MANUAL_ABLATION_RESULTS_ROOT
    if run_kind == "forced":
        value_root = ablation_root / branch
        return {
            "p0_planner_checkpoint_path": full_root / "handoff/p0_selected.pt",
            "calibration_checkpoint_path": value_root / "handoff/calibration.pt",
            "p1_planner_checkpoint_path": value_root / "handoff/p1_selected.pt",
        }

    value_root = ablation_root / "no_cf" if branch == "no_cf" else full_root
    gate_root = (
        ablation_root / branch
        if branch in {"no_cf", "without_field", "without_stop", "without_value_summary"}
        else full_root
    )
    return {
        "p0_planner_checkpoint_path": full_root / "handoff/p0_selected.pt",
        "calibration_checkpoint_path": value_root / "handoff/calibration.pt",
        "p1_planner_checkpoint_path": value_root / "handoff/p1_selected.pt",
        "stop_checkpoint_path": value_root / "handoff/stop.pt",
        "gate_checkpoint_path": gate_root / "handoff/gate.pt",
        "oracle_path": value_root / "handoff/oracle_full.sqlite3",
    }


def _require_manual_artifact_authority(
    artifact_paths: Mapping[str, Path],
    *,
    branch: str,
    run_kind: str,
) -> None:
    if branch == "full":
        _resolve_full_results_root_from_artifacts(artifact_paths)
        return
    expected = _manual_artifact_authority(branch=branch, run_kind=run_kind)
    drifted = {
        field: (artifact_paths.get(field), expected_path)
        for field, expected_path in expected.items()
        if artifact_paths.get(field) != expected_path
    }
    if drifted:
        raise ValueError(f"branch {branch!r} artifacts differ from the retained manual handoff authority: {drifted}")


def _parse_public_mapping(raw: object) -> CvoiDirectEpdmsConfig:
    payload = _require_mapping(raw, context="CVoI direct EPDMS public configuration")
    _require_exact_keys(payload, _PUBLIC_KEYS, context="CVoI direct EPDMS public configuration")

    schema = _require_exact_string(payload["schema"], field="schema", expected=CVOI_DIRECT_EPDMS_SCHEMA)
    version = _require_exact_integer(payload["version"], field="version", expected=CVOI_DIRECT_EPDMS_VERSION)
    branch = _require_exact_string(payload["branch"], field="branch")
    if branch not in CVOI_DIRECT_EPDMS_BRANCHES:
        raise ValueError(f"branch must be one of {list(CVOI_DIRECT_EPDMS_BRANCHES)}, got {branch!r}")
    run_kind = _require_exact_string(payload["run_kind"], field="run_kind")
    expected_run_kind = "controller" if branch in _CONTROLLER_BRANCHES else "forced"
    if run_kind != expected_run_kind:
        raise ValueError(f"branch {branch!r} requires run_kind={expected_run_kind!r}, got {run_kind!r}")

    split = _require_exact_string(payload["split"], field="split", expected=_SPLIT)
    protocol = _require_exact_string(payload["protocol"], field="protocol", expected=_PROTOCOL)
    max_horizon = _require_exact_integer(payload["max_horizon"], field="max_horizon", expected=_MAX_HORIZON)
    guidance_steps = _require_exact_integer(
        payload["guidance_steps"],
        field="guidance_steps",
        expected=_GUIDANCE_STEPS,
    )

    common_paths = {
        "training_config_path": _repository_training_config_path(payload["training_config_path"]),
        **{
            field: _absolute_normalized_path(payload[field], field=field)
            for field in (
                "encoder_checkpoint_path",
                "scenario_manifest_path",
                "output_root",
            )
        },
    }
    artifact_payload = _require_mapping(payload["artifacts"], context="artifacts")
    expected_artifact_keys = _CONTROLLER_ARTIFACT_KEYS if run_kind == "controller" else _FORCED_ARTIFACT_KEYS
    _require_exact_keys(artifact_payload, expected_artifact_keys, context=f"{run_kind} artifacts")

    artifact_path_fields = (
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
    )
    artifact_paths = {
        field: _absolute_normalized_path(artifact_payload[field], field=f"artifacts.{field}")
        for field in artifact_path_fields
    }
    if run_kind == "controller":
        for field in ("stop_checkpoint_path", "gate_checkpoint_path", "oracle_path"):
            artifact_paths[field] = _absolute_normalized_path(
                artifact_payload[field],
                field=f"artifacts.{field}",
            )
        gate_feature_mode = _require_exact_string(
            artifact_payload["gate_feature_mode"],
            field="artifacts.gate_feature_mode",
        )
        expected_feature_mode = _GATE_FEATURE_MODE_BY_BRANCH[branch]
        if gate_feature_mode != expected_feature_mode:
            raise ValueError(
                f"branch {branch!r} requires gate_feature_mode={expected_feature_mode!r}, "
                f"got {gate_feature_mode!r}"
            )
    else:
        gate_feature_mode = None

    _require_manual_artifact_authority(
        artifact_paths,
        branch=branch,
        run_kind=run_kind,
    )
    _reject_path_aliases({**common_paths, **artifact_paths})
    artifacts = CvoiDirectEpdmsArtifacts(
        p0_planner_checkpoint_path=artifact_paths["p0_planner_checkpoint_path"],
        calibration_checkpoint_path=artifact_paths["calibration_checkpoint_path"],
        p1_planner_checkpoint_path=artifact_paths["p1_planner_checkpoint_path"],
        stop_checkpoint_path=artifact_paths.get("stop_checkpoint_path"),
        gate_checkpoint_path=artifact_paths.get("gate_checkpoint_path"),
        oracle_path=artifact_paths.get("oracle_path"),
        gate_feature_mode=gate_feature_mode,
    )
    return CvoiDirectEpdmsConfig(
        schema=schema,
        version=version,
        branch=branch,
        run_kind=run_kind,
        split=split,
        protocol=protocol,
        max_horizon=max_horizon,
        guidance_steps=guidance_steps,
        training_config_path=common_paths["training_config_path"],
        encoder_checkpoint_path=common_paths["encoder_checkpoint_path"],
        scenario_manifest_path=common_paths["scenario_manifest_path"],
        output_root=common_paths["output_root"],
        artifacts=artifacts,
    )


def load_cvoi_direct_epdms_config(path: Path) -> CvoiDirectEpdmsConfig:
    """Load one strict public YAML without opening any declared artifact."""

    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"CVoI direct EPDMS public config path must be absolute: {source}")
    text = _read_regular_nonsymlink_text(source, context="CVoI direct EPDMS public config")
    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid CVoI direct EPDMS YAML: {source}") from error
    return _parse_public_mapping(raw)


def _public_config_mapping(config: CvoiDirectEpdmsConfig) -> dict[str, object]:
    if not isinstance(config, CvoiDirectEpdmsConfig):
        raise TypeError("config must be a CvoiDirectEpdmsConfig")
    artifacts: dict[str, object] = {
        "p0_planner_checkpoint_path": str(config.artifacts.p0_planner_checkpoint_path),
        "calibration_checkpoint_path": str(config.artifacts.calibration_checkpoint_path),
        "p1_planner_checkpoint_path": str(config.artifacts.p1_planner_checkpoint_path),
    }
    optional_artifacts = {
        "stop_checkpoint_path": config.artifacts.stop_checkpoint_path,
        "gate_checkpoint_path": config.artifacts.gate_checkpoint_path,
        "oracle_path": config.artifacts.oracle_path,
    }
    for field, value in optional_artifacts.items():
        if value is not None:
            artifacts[field] = str(value)
    if config.artifacts.gate_feature_mode is not None:
        artifacts["gate_feature_mode"] = config.artifacts.gate_feature_mode
    return {
        "schema": config.schema,
        "version": config.version,
        "branch": config.branch,
        "run_kind": config.run_kind,
        "split": config.split,
        "protocol": config.protocol,
        "max_horizon": config.max_horizon,
        "guidance_steps": config.guidance_steps,
        "training_config_path": str(config.training_config_path),
        "encoder_checkpoint_path": str(config.encoder_checkpoint_path),
        "scenario_manifest_path": str(config.scenario_manifest_path),
        "output_root": str(config.output_root),
        "artifacts": artifacts,
    }


def _require_edited_public_execution_paths(config: CvoiDirectEpdmsConfig) -> None:
    paths = {
        "training_config_path": config.training_config_path,
        "encoder_checkpoint_path": config.encoder_checkpoint_path,
        "scenario_manifest_path": config.scenario_manifest_path,
        "output_root": config.output_root,
        "artifacts.p0_planner_checkpoint_path": config.artifacts.p0_planner_checkpoint_path,
        "artifacts.calibration_checkpoint_path": config.artifacts.calibration_checkpoint_path,
        "artifacts.p1_planner_checkpoint_path": config.artifacts.p1_planner_checkpoint_path,
        "artifacts.stop_checkpoint_path": config.artifacts.stop_checkpoint_path,
        "artifacts.gate_checkpoint_path": config.artifacts.gate_checkpoint_path,
        "artifacts.oracle_path": config.artifacts.oracle_path,
    }
    unedited = sorted(
        field
        for field, path in paths.items()
        if path is not None and (path == _UNEDITED_PATH_ROOT or _UNEDITED_PATH_ROOT in path.parents)
    )
    if unedited:
        raise ValueError(
            "direct EPDMS public config contains unedited /path/to placeholders; "
            f"set deployment paths before execution: {unedited}"
        )


@contextmanager
def _open_regular_nonsymlink_binary(path: Path, *, context: str) -> Iterator[IO[bytes]]:
    try:
        before = os.lstat(path)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{context} does not exist: {path}") from error
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{context} must not be a symlink: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{context} must be a regular file: {path}")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EISDIR}:
            raise ValueError(f"{context} must be a regular non-symlink file: {path}") from error
        raise
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{context} must be a regular file: {path}")
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"{context} changed while it was opened: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha256_open_file(handle: IO[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _require_lowercase_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _HEX_DIGITS for character in value):
        raise ValueError(f"{field} must be a 64-character lowercase SHA-256 digest")
    return value


def _load_and_validate_gate_oracle_identity(
    *,
    branch: str,
    gate_feature_mode: str,
    oracle_path: Path,
    gate_checkpoint_path: Path,
) -> str:
    with _open_regular_nonsymlink_binary(oracle_path, context="controller Oracle") as oracle_handle:
        oracle_sha256 = _sha256_open_file(oracle_handle)

    with _open_regular_nonsymlink_binary(gate_checkpoint_path, context="controller Gate checkpoint") as gate_handle:
        try:
            payload = torch.load(gate_handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ValueError(f"failed to load controller Gate checkpoint: {gate_checkpoint_path}") from error
    gate_payload = _require_mapping(payload, context="controller Gate checkpoint")
    _require_exact_string(
        gate_payload.get("schema"),
        field="controller Gate checkpoint schema",
        expected=_GATE_CHECKPOINT_SCHEMA,
    )
    provenance = _require_mapping(gate_payload.get("provenance"), context="controller Gate checkpoint provenance")
    if any(type(key) is not str for key in provenance):
        raise ValueError("controller Gate checkpoint provenance keys must be strings")
    declared_sha256 = _require_lowercase_sha256(
        provenance.get("oracle_sha256"),
        field="controller Gate provenance oracle_sha256",
    )
    if declared_sha256 != oracle_sha256:
        raise ValueError(
            "controller Oracle SHA-256 does not match the Gate checkpoint provenance: "
            f"expected {declared_sha256}, got {oracle_sha256}"
        )
    expected_lineage = _ORACLE_LINEAGE_BY_BRANCH[branch]
    declared_lineage = _require_exact_string(
        provenance.get("oracle_lineage"),
        field="controller Gate provenance oracle_lineage",
    )
    if declared_lineage != expected_lineage:
        raise ValueError(
            f"branch {branch!r} requires Gate oracle_lineage={expected_lineage!r}, got {declared_lineage!r}"
        )
    declared_feature_mode = _require_exact_string(
        provenance.get("gate_feature_mode"),
        field="controller Gate provenance gate_feature_mode",
    )
    if declared_feature_mode != gate_feature_mode:
        raise ValueError(
            f"branch {branch!r} requires Gate gate_feature_mode={gate_feature_mode!r}, "
            f"got {declared_feature_mode!r}"
        )
    return oracle_sha256


def project_cvoi_direct_epdms_run(
    config: CvoiDirectEpdmsConfig,
    *,
    forced_horizon: int | None,
) -> CvoiDirectEpdmsProjection:
    """Project one public configuration into exactly one effective run."""

    validated = _parse_public_mapping(_public_config_mapping(config))
    artifacts = validated.artifacts
    common = {
        "branch": validated.branch,
        "split": validated.split,
        "protocol": validated.protocol,
        "training_config_path": validated.training_config_path,
        "encoder_checkpoint_path": validated.encoder_checkpoint_path,
        "scenario_manifest_path": validated.scenario_manifest_path,
    }
    if validated.run_kind == "controller":
        if forced_horizon is not None:
            raise ValueError("controller configuration forbids a forced horizon")
        if (
            artifacts.stop_checkpoint_path is None
            or artifacts.gate_checkpoint_path is None
            or artifacts.oracle_path is None
            or artifacts.gate_feature_mode is None
        ):
            raise ValueError("controller artifacts are incomplete")
        oracle_sha256 = _load_and_validate_gate_oracle_identity(
            branch=validated.branch,
            gate_feature_mode=artifacts.gate_feature_mode,
            oracle_path=artifacts.oracle_path,
            gate_checkpoint_path=artifacts.gate_checkpoint_path,
        )
        return CvoiDirectEpdmsProjection(
            **common,
            evaluation_mode="controller",
            horizon=None,
            guidance_steps=_GUIDANCE_STEPS,
            output_directory=validated.output_root,
            p0_planner_checkpoint_path=artifacts.p0_planner_checkpoint_path,
            calibration_checkpoint_path=artifacts.calibration_checkpoint_path,
            p1_planner_checkpoint_path=artifacts.p1_planner_checkpoint_path,
            stop_checkpoint_path=artifacts.stop_checkpoint_path,
            gate_checkpoint_path=artifacts.gate_checkpoint_path,
            gate_feature_mode=artifacts.gate_feature_mode,
            oracle_sha256=oracle_sha256,
        )

    if type(forced_horizon) is not int or not 0 <= forced_horizon <= _MAX_HORIZON:
        raise ValueError(f"forced configuration requires one integer horizon in H0--H4, got {forced_horizon!r}")
    if forced_horizon == 0:
        return CvoiDirectEpdmsProjection(
            **common,
            evaluation_mode="p0_forced",
            horizon=forced_horizon,
            guidance_steps=0,
            output_directory=validated.output_root / "h0",
            p0_planner_checkpoint_path=artifacts.p0_planner_checkpoint_path,
        )
    return CvoiDirectEpdmsProjection(
        **common,
        evaluation_mode="p1_field_forced",
        horizon=forced_horizon,
        guidance_steps=_GUIDANCE_STEPS,
        output_directory=validated.output_root / f"h{forced_horizon}",
        calibration_checkpoint_path=artifacts.calibration_checkpoint_path,
        p1_planner_checkpoint_path=artifacts.p1_planner_checkpoint_path,
    )


def _projection_mapping(projection: CvoiDirectEpdmsProjection) -> dict[str, object]:
    if not isinstance(projection, CvoiDirectEpdmsProjection):
        raise TypeError("projection must be a CvoiDirectEpdmsProjection")
    payload: dict[str, object] = {
        "schema": CVOI_DIRECT_EPDMS_EFFECTIVE_SCHEMA,
        "version": CVOI_DIRECT_EPDMS_EFFECTIVE_VERSION,
        "branch": projection.branch,
        "split": projection.split,
        "protocol": projection.protocol,
        "evaluation_mode": projection.evaluation_mode,
        "horizon": projection.horizon,
        "guidance_steps": projection.guidance_steps,
        "training_config_path": str(projection.training_config_path),
        "encoder_checkpoint_path": str(projection.encoder_checkpoint_path),
        "scenario_manifest_path": str(projection.scenario_manifest_path),
        "output_directory": str(projection.output_directory),
    }
    path_fields = (
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
        "stop_checkpoint_path",
        "gate_checkpoint_path",
    )
    for field in path_fields:
        value = getattr(projection, field)
        if value is not None:
            payload[field] = str(value)
    if projection.gate_feature_mode is not None:
        payload["gate_feature_mode"] = projection.gate_feature_mode
    if projection.oracle_sha256 is not None:
        payload["oracle_sha256"] = projection.oracle_sha256
    return payload


def _parse_effective_mapping(raw: object) -> CvoiDirectEpdmsProjection:
    payload = _require_mapping(raw, context="CVoI direct EPDMS effective configuration")
    mode = _require_exact_string(payload.get("evaluation_mode"), field="evaluation_mode")
    mode_keys = _EFFECTIVE_ARTIFACT_KEYS.get(mode)
    if mode_keys is None:
        raise ValueError(f"unsupported CVoI direct EPDMS evaluation mode: {mode!r}")
    _require_exact_keys(
        payload,
        _EFFECTIVE_COMMON_KEYS | mode_keys,
        context=f"effective {mode} mode",
    )
    _require_exact_string(
        payload["schema"],
        field="schema",
        expected=CVOI_DIRECT_EPDMS_EFFECTIVE_SCHEMA,
    )
    _require_exact_integer(
        payload["version"],
        field="version",
        expected=CVOI_DIRECT_EPDMS_EFFECTIVE_VERSION,
    )
    branch = _require_exact_string(payload["branch"], field="branch")
    if branch not in CVOI_DIRECT_EPDMS_BRANCHES:
        raise ValueError(f"branch must be one of {list(CVOI_DIRECT_EPDMS_BRANCHES)}, got {branch!r}")
    split = _require_exact_string(payload["split"], field="split", expected=_SPLIT)
    protocol = _require_exact_string(payload["protocol"], field="protocol", expected=_PROTOCOL)

    horizon = payload["horizon"]
    guidance_steps = _require_exact_integer(payload["guidance_steps"], field="guidance_steps")
    if mode == "controller":
        if branch not in _CONTROLLER_BRANCHES:
            raise ValueError(f"controller mode is incompatible with branch {branch!r}")
        if horizon is not None:
            raise ValueError("controller mode requires horizon=null")
        if guidance_steps != _GUIDANCE_STEPS:
            raise ValueError(f"controller mode requires guidance_steps={_GUIDANCE_STEPS}")
    elif mode == "p0_forced":
        if branch not in _FORCED_BRANCHES:
            raise ValueError(f"p0_forced mode is incompatible with branch {branch!r}")
        if type(horizon) is not int or horizon != 0:
            raise ValueError(f"p0_forced mode requires horizon=0, got {horizon!r}")
        if guidance_steps != 0:
            raise ValueError("p0_forced mode requires guidance_steps=0")
    else:
        if branch not in _FORCED_BRANCHES:
            raise ValueError(f"p1_field_forced mode is incompatible with branch {branch!r}")
        if type(horizon) is not int or not 1 <= horizon <= _MAX_HORIZON:
            raise ValueError(f"p1_field_forced mode requires an integer horizon in H1--H4, got {horizon!r}")
        if guidance_steps != _GUIDANCE_STEPS:
            raise ValueError(f"p1_field_forced mode requires guidance_steps={_GUIDANCE_STEPS}")

    path_fields = {
        field: _absolute_normalized_path(payload[field], field=field)
        for field in (
            "training_config_path",
            "encoder_checkpoint_path",
            "scenario_manifest_path",
            "output_directory",
        )
    }
    for field in mode_keys:
        if field.endswith("_path"):
            path_fields[field] = _absolute_normalized_path(payload[field], field=field)
    _reject_path_aliases(path_fields)
    if mode != "controller" and path_fields["output_directory"].name != f"h{horizon}":
        raise ValueError(f"{mode} output_directory must end in h{horizon}")

    gate_feature_mode = None
    oracle_sha256 = None
    if mode == "controller":
        gate_feature_mode = _require_exact_string(payload["gate_feature_mode"], field="gate_feature_mode")
        expected_feature_mode = _GATE_FEATURE_MODE_BY_BRANCH[branch]
        if gate_feature_mode != expected_feature_mode:
            raise ValueError(
                f"branch {branch!r} requires gate_feature_mode={expected_feature_mode!r}, "
                f"got {gate_feature_mode!r}"
            )
        oracle_sha256 = _require_lowercase_sha256(payload["oracle_sha256"], field="oracle_sha256")

    return CvoiDirectEpdmsProjection(
        branch=branch,
        split=split,
        protocol=protocol,
        evaluation_mode=mode,
        horizon=horizon,
        guidance_steps=guidance_steps,
        training_config_path=path_fields["training_config_path"],
        encoder_checkpoint_path=path_fields["encoder_checkpoint_path"],
        scenario_manifest_path=path_fields["scenario_manifest_path"],
        output_directory=path_fields["output_directory"],
        p0_planner_checkpoint_path=path_fields.get("p0_planner_checkpoint_path"),
        calibration_checkpoint_path=path_fields.get("calibration_checkpoint_path"),
        p1_planner_checkpoint_path=path_fields.get("p1_planner_checkpoint_path"),
        stop_checkpoint_path=path_fields.get("stop_checkpoint_path"),
        gate_checkpoint_path=path_fields.get("gate_checkpoint_path"),
        gate_feature_mode=gate_feature_mode,
        oracle_sha256=oracle_sha256,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"direct EPDMS JSON contains forbidden non-finite value {value!r}")


def _unique_json_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"duplicate CVoI direct EPDMS JSON key: {key!r}")
        mapping[key] = value
    return mapping


def read_cvoi_direct_epdms_scenario_manifest(path: Path) -> Mapping[str, str]:
    """Read the exact NavTest scenario-token/observation-key bijection."""

    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"direct EPDMS scenario manifest path must be absolute: {source}")
    try:
        with _open_regular_nonsymlink_binary(source, context="direct EPDMS scenario manifest") as handle:
            data = handle.read()
    except FileNotFoundError as error:
        raise ValueError(f"direct EPDMS scenario manifest must be a non-symlink regular file: {source}") from error
    if not data or not data.endswith(b"\n"):
        raise ValueError("direct EPDMS scenario manifest must be non-empty canonical JSONL")

    tokens: list[str] = []
    token_by_observation_key: dict[str, str] = {}
    for index, line in enumerate(data.splitlines(keepends=True)):
        if line == b"\n" or not line.endswith(b"\n"):
            raise ValueError(f"direct EPDMS scenario manifest row {index} is blank or lacks its newline")
        try:
            payload = json.loads(
                line[:-1].decode("utf-8"),
                object_pairs_hook=_unique_json_mapping,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"direct EPDMS scenario manifest row {index} is invalid JSON") from error
        if not isinstance(payload, Mapping) or set(payload) != _SCENARIO_ROW_FIELDS:
            actual = set(payload) if isinstance(payload, Mapping) else set()
            raise ValueError(
                "direct EPDMS scenario manifest row fields mismatch: "
                f"missing={sorted(_SCENARIO_ROW_FIELDS - actual)}, "
                f"unexpected={sorted(actual - _SCENARIO_ROW_FIELDS, key=str)}"
            )
        if line[:-1] != _canonical_json_bytes(payload):
            raise ValueError(f"direct EPDMS scenario manifest row {index} must use canonical JSON bytes")
        if payload["schema"] != _SCENARIO_ROW_SCHEMA or payload["protocol_id"] != _PROTOCOL:
            raise ValueError(f"direct EPDMS scenario manifest row {index} has the wrong schema or protocol")
        token = _require_navtest_scenario_token(payload["scenario_token"])
        observation_key = payload["observation_key"]
        for field in ("log_name", "current_camera_data_path"):
            value = _require_exact_string(
                payload[field],
                field=f"direct EPDMS scenario manifest row {index} {field}",
            )
            if value != value.strip():
                raise ValueError(
                    f"direct EPDMS scenario manifest row {index} {field} " "must not contain surrounding whitespace"
                )
        if (
            type(observation_key) is not str
            or len(observation_key) != 64
            or any(character not in _HEX_DIGITS for character in observation_key)
        ):
            raise ValueError(f"direct EPDMS scenario manifest row {index} has an invalid observation_key")
        if token in tokens:
            raise ValueError(f"direct EPDMS scenario manifest contains duplicate token {token!r}")
        if observation_key in token_by_observation_key:
            raise ValueError(f"direct EPDMS scenario manifest contains duplicate observation_key {observation_key!r}")
        tokens.append(token)
        token_by_observation_key[observation_key] = token
    if tokens != sorted(tokens):
        raise ValueError("direct EPDMS scenario manifest tokens must use canonical sorted order")
    return MappingProxyType(token_by_observation_key)


def _direct_projection_artifact_paths(
    projection: CvoiDirectEpdmsProjection,
) -> dict[str, Path]:
    """Resolve the exact mode-minimal handoffs from the manual lineage authority."""

    run_kind = "controller" if projection.evaluation_mode == "controller" else "forced"
    expected_fields_by_mode = {
        "controller": (
            "p0_planner_checkpoint_path",
            "calibration_checkpoint_path",
            "p1_planner_checkpoint_path",
            "stop_checkpoint_path",
            "gate_checkpoint_path",
        ),
        "p0_forced": ("p0_planner_checkpoint_path",),
        "p1_field_forced": (
            "calibration_checkpoint_path",
            "p1_planner_checkpoint_path",
        ),
    }
    expected_fields = expected_fields_by_mode[projection.evaluation_mode]
    artifact_fields = {
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
        "stop_checkpoint_path",
        "gate_checkpoint_path",
    }
    actual_fields = {field for field in artifact_fields if getattr(projection, field) is not None}
    if actual_fields != set(expected_fields):
        raise ValueError(
            f"direct EPDMS {projection.evaluation_mode} artifact fields mismatch: "
            f"missing={sorted(set(expected_fields) - actual_fields)}, "
            f"unexpected={sorted(actual_fields - set(expected_fields))}"
        )

    if projection.branch == "full":
        projected_paths = {field: getattr(projection, field) for field in expected_fields}
        full_results_root = _resolve_full_results_root_from_artifacts(projected_paths)
        handoffs = cvoi_manual_lineage.derive_cvoi_manual_full_handoffs(full_results_root)
        expected = {field: handoffs[_FULL_HANDOFF_NAME_BY_ARTIFACT_FIELD[field]] for field in expected_fields}
    else:
        authority = _manual_artifact_authority(branch=projection.branch, run_kind=run_kind)
        expected = {field: authority[field] for field in expected_fields}
    drifted = {
        field: (getattr(projection, field), path)
        for field, path in expected.items()
        if getattr(projection, field) != path
    }
    if drifted:
        raise ValueError(
            "direct EPDMS artifact paths differ from the retained manual handoff authority: " f"{drifted}"
        )

    resolved_identities: dict[Path, str] = {}
    for field, path in expected.items():
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"direct EPDMS artifact does not exist: {path}") from error
        if not resolved.is_file():
            raise ValueError(f"direct EPDMS artifact must resolve to a regular file: {path}")
        previous = resolved_identities.get(resolved)
        if previous is not None:
            raise ValueError(f"direct EPDMS artifacts {previous!r} and {field!r} alias {resolved}")
        resolved_identities[resolved] = field
    return expected


def _direct_state_sha256(state: object, *, role: str) -> str:
    """Hash one validated CPU state without copying its full byte stream."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"direct EPDMS {role} state must be a non-empty mapping")
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if type(key) is not str or not key or not torch.is_tensor(tensor):
            raise ValueError(f"direct EPDMS {role} state must map non-empty string keys to tensors")
        metadata = _canonical_json_bytes(
            {
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        )
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        byte_view = tensor.detach().cpu().reshape(-1).contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))
    return digest.hexdigest()


def _preflight_direct_training_config(
    projection: CvoiDirectEpdmsProjection,
    *,
    expected_p1_branch_id: str,
) -> int:
    """Parse the source training YAML and bind it to the projected P1 lineage."""

    source = projection.training_config_path
    text = _read_regular_nonsymlink_text(source, context="direct EPDMS training config")
    try:
        raw_config = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid direct EPDMS training YAML: {source}") from error
    if not isinstance(raw_config, Mapping):
        raise ValueError("direct EPDMS training config must contain one mapping")
    parsed = parse_training_config(dict(raw_config))
    cvoi = getattr(parsed, "cvoi", None)
    if cvoi is None or cvoi.protocol_version != "formal_v2_navsim_e120_h4_v3" or cvoi.stage != "guided_planner":
        raise ValueError(
            "direct EPDMS training_config_path must contain one parsed " "NavSim-e120 guided_planner config"
        )
    value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage(
        cvoi.ablation_signature,
        stage="guided_planner",
    )
    actual_p1_branch_id = value_lineage.checkpoint_branch_id("guided_planner")
    if actual_p1_branch_id != expected_p1_branch_id:
        raise ValueError(
            "direct EPDMS training config branch differs from the projected P1 lineage: "
            f"expected={expected_p1_branch_id!r}, actual={actual_p1_branch_id!r}"
        )
    model_name = getattr(parsed.model, "model_name", None)
    if type(model_name) is not str or model_name not in VIT_EMBED_DIMS:
        raise ValueError(
            "direct EPDMS training config must select one encoder with a known embed_dim, "
            f"got model.model_name={model_name!r}"
        )
    return int(VIT_EMBED_DIMS[model_name])


def preflight_cvoi_direct_epdms_projection(
    projection: CvoiDirectEpdmsProjection,
) -> None:
    """Validate every selected input before the runner creates its output root."""

    normalized = _parse_effective_mapping(_projection_mapping(projection))
    identity = resolve_cvoi_direct_epdms_artifact_identity(
        normalized.branch,
        evaluation_mode=normalized.evaluation_mode,
    )
    expected_encoder_embed_dim = _preflight_direct_training_config(
        normalized,
        expected_p1_branch_id=identity.p1_branch_id,
    )
    try:
        with _open_regular_nonsymlink_binary(
            normalized.encoder_checkpoint_path,
            context="direct EPDMS encoder checkpoint",
        ) as handle:
            if not handle.read(1):
                raise ValueError("direct EPDMS encoder checkpoint must not be empty")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"direct EPDMS encoder checkpoint does not exist: " f"{normalized.encoder_checkpoint_path}"
        ) from error
    read_cvoi_direct_epdms_scenario_manifest(normalized.scenario_manifest_path)
    paths = _direct_projection_artifact_paths(normalized)
    full_results_root = (
        _full_results_root_from_validated_projection_paths(paths) if normalized.branch == "full" else None
    )

    p0_shapes: object = None
    p0_encoder_sha256: str | None = None
    p0_path = paths.get("p0_planner_checkpoint_path")
    if p0_path is not None:
        resolved_p0 = resolve_formal_v2_navsim_e120_selected_checkpoint(
            p0_path,
            results_root=(
                full_results_root
                if full_results_root is not None
                else cvoi_manual_lineage.CVOI_MANUAL_FULL_RESULTS_ROOT
            ),
            stage="p0",
        )
        p0_payload = read_cvoi_direct_epdms_planner_checkpoint(
            resolved_p0,
            expected_stage="p0",
            expected_branch_id=identity.p0_branch_id,
        )
        p0_shapes = p0_payload["role_state_shapes"]
        p0_encoder_sha256 = _direct_state_sha256(
            p0_payload["encoder"],
            role="P0 encoder",
        )
        del p0_payload

    p1_path = paths.get("p1_planner_checkpoint_path")
    if p1_path is not None:
        value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
            phase="guided_planner",
            branch_id=identity.p1_branch_id,
            full_results_root=full_results_root,
        )
        resolved_p1 = resolve_formal_v2_navsim_e120_selected_checkpoint(
            p1_path,
            results_root=value_lineage.result_root,
            stage="p1",
        )
        p1_payload = read_cvoi_direct_epdms_planner_checkpoint(
            resolved_p1,
            expected_stage="p1",
            expected_branch_id=identity.p1_branch_id,
        )
        if p0_shapes is not None and p0_shapes != p1_payload["role_state_shapes"]:
            raise ValueError("direct EPDMS P0/P1 role architectures differ")
        if p0_encoder_sha256 is not None and p0_encoder_sha256 != _direct_state_sha256(
            p1_payload["encoder"], role="P1 encoder"
        ):
            raise ValueError("direct EPDMS P0/P1 encoder states differ")
        del p1_payload

    calibration_architecture: object = None
    calibration_path = paths.get("calibration_checkpoint_path")
    if calibration_path is not None:
        calibration_payload = read_cvoi_navsim_e120_direct_value_checkpoint(
            calibration_path,
            required_phase="field_calibrated",
            required_branch_id=identity.calibration_branch_id,
            map_location="cpu",
        )
        calibration_architecture = calibration_payload["architecture"]
        actual_calibration_embed_dim = (
            calibration_architecture.get("embed_dim") if isinstance(calibration_architecture, Mapping) else None
        )
        if actual_calibration_embed_dim != expected_encoder_embed_dim:
            raise ValueError(
                "direct EPDMS Calibration Value embed_dim differs from the training encoder: "
                f"expected={expected_encoder_embed_dim}, "
                f"actual={actual_calibration_embed_dim}"
            )
        del calibration_payload

    stop_path = paths.get("stop_checkpoint_path")
    if stop_path is not None:
        if identity.stop_branch_id is None:
            raise RuntimeError("direct EPDMS controller Stop identity was not resolved")
        stop_payload = read_cvoi_navsim_e120_direct_value_checkpoint(
            stop_path,
            required_phase="stop_calibrated",
            required_branch_id=identity.stop_branch_id,
            map_location="cpu",
        )
        if calibration_architecture != stop_payload["architecture"]:
            raise ValueError("direct EPDMS Calibration/Stop Value architectures differ")
        del stop_payload

    gate_path = paths.get("gate_checkpoint_path")
    if gate_path is not None:
        if normalized.oracle_sha256 is None or normalized.gate_feature_mode is None:
            raise RuntimeError("direct EPDMS controller Gate identity was not projected")
        gate_payload = read_cvoi_direct_epdms_gate_checkpoint(
            gate_path,
            branch=normalized.branch,
            oracle_sha256=normalized.oracle_sha256,
            gate_feature_mode=normalized.gate_feature_mode,
        )
        if isinstance(calibration_architecture, Mapping) and gate_payload[
            "latent_dim"
        ] != calibration_architecture.get("embed_dim"):
            raise ValueError("direct EPDMS Gate latent_dim differs from the Value architecture")
        del gate_payload


def load_cvoi_direct_epdms_projection(path: Path) -> CvoiDirectEpdmsProjection:
    """Load one exact effective JSON projection; public YAML is rejected."""

    source = Path(path)
    if not source.is_absolute():
        raise ValueError(f"CVoI direct EPDMS effective config path must be absolute: {source}")
    text = _read_regular_nonsymlink_text(source, context="CVoI direct EPDMS effective config")
    try:
        raw = json.loads(text, object_pairs_hook=_unique_json_mapping)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"invalid CVoI direct EPDMS effective JSON: {source}") from error
    return _parse_effective_mapping(raw)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise ValueError(f"effective config parent must be a directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_new_effective_target(path: Path) -> Path:
    target = _absolute_normalized_path(str(path), field="effective config target")
    parent = target.parent
    try:
        metadata = os.lstat(parent)
    except FileNotFoundError as error:
        raise ValueError(f"effective config parent directory does not exist: {parent}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"effective config parent must be a non-symlink directory: {parent}")
    if parent.resolve(strict=True) != parent:
        raise ValueError(f"effective config parent must be canonical: {parent}")
    if os.path.lexists(target):
        raise FileExistsError(f"effective config target already exists: {target}")
    return target


def write_cvoi_direct_epdms_projection(
    projection: CvoiDirectEpdmsProjection,
    path: Path,
) -> Path:
    """Failure-atomically create one new effective JSON projection."""

    normalized_projection = _parse_effective_mapping(_projection_mapping(projection))
    payload = _projection_mapping(normalized_projection)
    target = _validate_new_effective_target(Path(path))
    serialized = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"effective config target already exists: {target}") from error
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _require_absolute_existing_directory(path: Path, *, context: str) -> Path:
    if not isinstance(path, Path):
        raise TypeError(f"{context} must be a Path")
    if not path.is_absolute():
        raise ValueError(f"{context} must be absolute: {path}")
    try:
        metadata = os.lstat(path)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{context} must be an existing non-symlink directory: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{context} must be an existing non-symlink directory: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{context} must be an existing non-symlink directory: {path}") from error
    if resolved != path:
        raise ValueError(f"{context} must be a canonical absolute directory: {path}")
    return path


def _read_direct_json_object(data: bytes, *, context: str) -> Mapping[str, object]:
    try:
        payload = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_json_mapping,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context} is invalid JSON: {error}") from error
    return _require_mapping(payload, context=context)


def _direct_record_mapping(record: DirectEpdmsSceneRecord) -> dict[str, object]:
    return {
        "schema": CVOI_DIRECT_EPDMS_SCENE_SCHEMA,
        "branch": record.branch,
        "scenario_token": record.scenario_token,
        "evaluation_seed": record.evaluation_seed,
        "epdms": record.epdms,
        "final_horizon": record.final_horizon,
        "latency_ms": record.latency_ms,
    }


def _parse_direct_record(payload: Mapping[str, object], *, row_index: int) -> DirectEpdmsSceneRecord:
    _require_exact_keys(payload, _DIRECT_RECORD_FIELDS, context=f"direct EPDMS record row {row_index}")
    _require_exact_string(
        payload["schema"],
        field=f"direct EPDMS record row {row_index} schema",
        expected=CVOI_DIRECT_EPDMS_SCENE_SCHEMA,
    )
    return DirectEpdmsSceneRecord(
        branch=payload["branch"],  # type: ignore[arg-type]
        scenario_token=payload["scenario_token"],  # type: ignore[arg-type]
        evaluation_seed=payload["evaluation_seed"],  # type: ignore[arg-type]
        epdms=payload["epdms"],  # type: ignore[arg-type]
        final_horizon=payload["final_horizon"],  # type: ignore[arg-type]
        latency_ms=payload["latency_ms"],  # type: ignore[arg-type]
    )


def _validate_direct_record_inventory(
    records: list[DirectEpdmsSceneRecord],
) -> tuple[DirectEpdmsSceneRecord, ...]:
    if not records:
        raise ValueError("direct EPDMS records.jsonl must contain at least one record")
    tokens = [record.scenario_token for record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError("direct EPDMS scenario_token inventory must be unique")
    if tokens != sorted(tokens):
        raise ValueError("direct EPDMS scenario_token inventory must be sorted")
    branches = {record.branch for record in records}
    if len(branches) != 1:
        raise ValueError(f"direct EPDMS records must use one branch, got {sorted(branches)}")
    seeds = {record.evaluation_seed for record in records}
    if len(seeds) != 1:
        raise ValueError(f"direct EPDMS records must use one evaluation seed, got {sorted(seeds)}")
    return tuple(records)


def read_direct_epdms_records(result_root: Path) -> tuple[DirectEpdmsSceneRecord, ...]:
    """Read one immutable, sorted ``records.jsonl`` direct-run artifact."""

    root = _require_absolute_existing_directory(result_root, context="direct EPDMS result root")
    source = root / "records.jsonl"
    try:
        with _open_regular_nonsymlink_binary(source, context="direct EPDMS records.jsonl") as handle:
            data = handle.read()
    except FileNotFoundError as error:
        raise ValueError(f"direct EPDMS records.jsonl must be an existing regular file: {source}") from error
    if not data or not data.endswith(b"\n"):
        raise ValueError("direct EPDMS records.jsonl must be non-empty newline-terminated JSONL")

    records: list[DirectEpdmsSceneRecord] = []
    for row_index, line in enumerate(data.splitlines(keepends=True)):
        if line == b"\n" or not line.endswith(b"\n") or line.endswith(b"\r\n"):
            raise ValueError(f"direct EPDMS record row {row_index} must be non-blank LF-terminated JSON")
        payload = _read_direct_json_object(line[:-1], context=f"direct EPDMS record row {row_index}")
        records.append(_parse_direct_record(payload, row_index=row_index))
    return _validate_direct_record_inventory(records)


def _validate_direct_result_root_layout(root: Path) -> tuple[Path, Path]:
    entries = {entry.name: entry for entry in root.iterdir()}
    actual = frozenset(entries)
    missing = _DIRECT_ROOT_REQUIRED_ENTRIES - actual
    extra = actual - (_DIRECT_ROOT_REQUIRED_ENTRIES | _DIRECT_ROOT_OPTIONAL_ENTRIES)
    if missing or extra:
        raise ValueError(
            "direct EPDMS result root layout mismatch: " f"missing={sorted(missing)}, unexpected={sorted(extra)}"
        )

    scorer_output = _require_absolute_existing_directory(
        entries["scorer_output"],
        context="direct EPDMS scorer_output",
    )
    policy_traces = _require_absolute_existing_directory(
        entries["policy_traces"],
        context="direct EPDMS policy_traces",
    )
    effective = entries.get("cvoi_direct_epdms_effective.json")
    if effective is not None:
        try:
            with _open_regular_nonsymlink_binary(
                effective,
                context="direct EPDMS effective configuration",
            ):
                pass
        except FileNotFoundError as error:
            raise ValueError("direct EPDMS result root effective configuration must be a regular file") from error

    score_candidates: list[Path] = []
    for candidate in scorer_output.rglob("*.csv"):
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("direct EPDMS score CSV must be a contained non-symlink regular file")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(scorer_output)
        except ValueError as error:
            raise ValueError("direct EPDMS score CSV must be contained by scorer_output") from error
        score_candidates.append(resolved)
    if len(score_candidates) != 1:
        raise ValueError("direct EPDMS scorer_output must contain exactly one CSV, " f"found {len(score_candidates)}")
    return score_candidates[0], policy_traces


def _read_direct_score_rows(path: Path) -> dict[str, float]:
    protocol = get_cvoi_navsim_metric_protocol(_PROTOCOL)
    if protocol.split != _SPLIT or protocol.summary_token != "average_all_frames":
        raise RuntimeError("registered EPDMS V2 protocol no longer matches the direct-run contract")
    if protocol.required_components != frozenset(_DIRECT_SCORE_COMPONENTS):
        raise RuntimeError("registered EPDMS V2 score components no longer match the direct-run contract")

    try:
        with _open_regular_nonsymlink_binary(path, context="direct EPDMS score CSV") as handle:
            data = handle.read()
    except FileNotFoundError as error:
        raise ValueError(f"direct EPDMS score CSV must be an existing regular file: {path}") from error
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("direct EPDMS score CSV must be UTF-8") from error
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        fieldnames = reader.fieldnames
        rows = list(reader)
    except csv.Error as error:
        raise ValueError(f"direct EPDMS score CSV is malformed: {error}") from error
    if fieldnames is None or not fieldnames:
        raise ValueError("direct EPDMS score CSV must contain a header")
    if len(fieldnames) != len(set(fieldnames)):
        raise ValueError("direct EPDMS score CSV header contains duplicate columns")
    index_fields = [field for field in fieldnames if field in _DIRECT_SCORE_INDEX_HEADERS]
    if len(index_fields) > 1:
        raise ValueError("direct EPDMS score CSV may contain at most one pandas index column")
    actual = set(fieldnames) - set(index_fields)
    missing = _DIRECT_SCORE_FIELDS - actual
    extra = actual - _DIRECT_SCORE_FIELDS
    if missing or extra:
        raise ValueError(
            "direct EPDMS score CSV fields must match the exact EPDMS V2 schema: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if not rows:
        raise ValueError("direct EPDMS score CSV must contain scenario rows and average_all_frames")

    scores: dict[str, float] = {}
    summary_row: Mapping[str, str] | None = None
    index_field = index_fields[0] if index_fields else None
    for row_index, raw_row in enumerate(rows):
        if None in raw_row or any(value is None for value in raw_row.values()):
            raise ValueError(f"direct EPDMS score CSV row {row_index} has a malformed column count")
        row = {str(key): str(value) for key, value in raw_row.items()}
        if index_field is not None and row[index_field] != str(row_index):
            raise ValueError("direct EPDMS score CSV pandas index must contain consecutive zero-based values")
        token = row["token"]
        if not token or token != token.strip():
            raise ValueError(f"direct EPDMS score CSV row {row_index} has an invalid token")
        if row["valid"].strip().lower() != "true":
            raise ValueError(f"direct EPDMS score CSV token {token!r} is not valid=true")
        if token == protocol.summary_token:
            if summary_row is not None:
                raise ValueError("direct EPDMS score CSV contains duplicate average_all_frames rows")
            if row_index != len(rows) - 1:
                raise ValueError("direct EPDMS score CSV average_all_frames row must be last")
            summary_row = row
            continue
        _require_navtest_scenario_token(token)
        if token in scores:
            raise ValueError(f"direct EPDMS score CSV contains duplicate scenario token {token!r}")
        _csv_components(row, protocol=protocol, token=token)
        scores[token] = _csv_number(row["score"], field=f"CSV token {token!r} score", unit_score=True)

    if summary_row is None:
        raise ValueError("direct EPDMS score CSV is missing aggregate token average_all_frames")
    if not scores:
        raise ValueError("direct EPDMS score CSV must contain at least one scenario score")
    _csv_components(summary_row, protocol=protocol, token=protocol.summary_token)
    summary_score = _csv_number(
        summary_row["score"],
        field="CSV aggregate average_all_frames score",
        unit_score=True,
    )
    mean_score = sum(scores.values()) / len(scores)
    if not math.isclose(summary_score, mean_score, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(
            "direct EPDMS score CSV average_all_frames score must equal the scenario mean: "
            f"expected {mean_score}, got {summary_score}"
        )
    return scores


def _parse_direct_trace(payload: Mapping[str, object], *, filename: str) -> _DirectEpdmsTrace:
    _require_exact_keys(payload, _DIRECT_TRACE_FIELDS, context=f"direct EPDMS trace {filename!r}")
    _require_exact_string(
        payload["schema"],
        field=f"direct EPDMS trace {filename!r} schema",
        expected=CVOI_DIRECT_EPDMS_TRACE_SCHEMA,
    )
    _require_exact_integer(
        payload["version"],
        field=f"direct EPDMS trace {filename!r} version",
        expected=CVOI_DIRECT_EPDMS_TRACE_VERSION,
    )
    _require_exact_string(payload["split"], field="trace.split", expected=_SPLIT)
    _require_exact_string(payload["protocol"], field="trace.protocol", expected=_PROTOCOL)
    return _DirectEpdmsTrace(
        branch=_require_direct_branch(payload["branch"]),
        scenario_token=_require_navtest_scenario_token(payload["scenario_token"]),
        evaluation_seed=_require_evaluation_seed(payload["evaluation_seed"]),
        final_horizon=_require_final_horizon(payload["final_horizon"]),
        latency_ms=_require_finite_float(payload["latency_ms"], field="trace.latency_ms", nonnegative=True),
    )


def _read_direct_traces(trace_dir: Path) -> dict[str, _DirectEpdmsTrace]:
    entries = sorted(trace_dir.iterdir(), key=lambda entry: entry.name)
    if not entries:
        raise ValueError("direct EPDMS policy trace inventory must not be empty")
    traces: dict[str, _DirectEpdmsTrace] = {}
    filenames: dict[str, str] = {}
    for entry in entries:
        if entry.suffix != ".json":
            raise ValueError(f"direct EPDMS policy trace layout has an unexpected entry: {entry.name}")
        try:
            with _open_regular_nonsymlink_binary(entry, context="direct EPDMS policy trace") as handle:
                data = handle.read()
        except FileNotFoundError as error:
            raise ValueError(f"direct EPDMS policy trace must be an existing regular file: {entry.name}") from error
        payload = _read_direct_json_object(data, context=f"direct EPDMS policy trace {entry.name!r}")
        if data != _canonical_json_bytes(payload):
            raise ValueError(f"direct EPDMS policy trace {entry.name!r} must use canonical JSON bytes")
        trace = _parse_direct_trace(payload, filename=entry.name)
        token = trace.scenario_token
        if token in traces:
            raise ValueError(
                "direct EPDMS policy trace contains duplicate scenario token "
                f"{token!r} in {filenames[token]!r} and {entry.name!r}"
            )
        traces[token] = trace
        filenames[token] = entry.name

    branches = {trace.branch for trace in traces.values()}
    if len(branches) != 1:
        raise ValueError(f"direct EPDMS policy traces must use one branch, got {sorted(branches)}")
    seeds = {trace.evaluation_seed for trace in traces.values()}
    if len(seeds) != 1:
        raise ValueError(f"direct EPDMS policy traces must use one evaluation seed, got {sorted(seeds)}")
    for token, filename in filenames.items():
        if filename != f"{token}.json":
            raise ValueError(
                f"direct EPDMS policy trace filename must match scenario token {token!r}, got {filename!r}"
            )
    return traces


def _preflight_direct_output_targets(root: Path) -> tuple[Path, Path]:
    targets = tuple(root / name for name in _DIRECT_OUTPUT_NAMES)
    for target in targets:
        if os.path.lexists(target):
            raise FileExistsError(f"direct EPDMS aggregate output already exists: {target.name}")
    return targets


def _unlink_if_same_inode(target: Path, temporary: Path) -> None:
    try:
        target_metadata = os.lstat(target)
        temporary_metadata = os.lstat(temporary)
    except FileNotFoundError:
        return
    if (target_metadata.st_dev, target_metadata.st_ino) == (
        temporary_metadata.st_dev,
        temporary_metadata.st_ino,
    ):
        target.unlink()


def _write_direct_outputs_exclusively(
    root: Path,
    payloads: tuple[tuple[Path, bytes], tuple[Path, bytes]],
) -> None:
    temporaries: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path]] = []
    try:
        for target, data in payloads:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=root)
            temporary = Path(temporary_name)
            temporaries.append((target, temporary))
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        for target, temporary in temporaries:
            try:
                os.link(temporary, target)
            except FileExistsError as error:
                raise FileExistsError(f"direct EPDMS aggregate output already exists: {target.name}") from error
            published.append((target, temporary))
        _fsync_directory(root)
    except Exception:
        for target, temporary in reversed(published):
            _unlink_if_same_inode(target, temporary)
        if published:
            try:
                _fsync_directory(root)
            except OSError:
                pass
        raise
    finally:
        for _, temporary in temporaries:
            temporary.unlink(missing_ok=True)


def aggregate_direct_epdms_results(result_root: Path) -> tuple[DirectEpdmsSceneRecord, ...]:
    """Join one exact EPDMS V2 score CSV to direct per-scene traces."""

    root = _require_absolute_existing_directory(result_root, context="direct EPDMS result root")
    records_target, summary_target = _preflight_direct_output_targets(root)
    score_path, trace_dir = _validate_direct_result_root_layout(root)
    scores = _read_direct_score_rows(score_path)
    traces = _read_direct_traces(trace_dir)
    score_tokens = set(scores)
    trace_tokens = set(traces)
    if score_tokens != trace_tokens:
        raise ValueError(
            "direct EPDMS score/trace inventory coverage mismatch: "
            f"missing_traces={sorted(score_tokens - trace_tokens)}, "
            f"extra_traces={sorted(trace_tokens - score_tokens)}"
        )

    records = tuple(
        DirectEpdmsSceneRecord(
            branch=traces[token].branch,
            scenario_token=token,
            evaluation_seed=traces[token].evaluation_seed,
            epdms=scores[token],
            final_horizon=traces[token].final_horizon,
            latency_ms=traces[token].latency_ms,
        )
        for token in sorted(score_tokens)
    )
    records = _validate_direct_record_inventory(list(records))
    records_bytes = b"".join(_canonical_json_bytes(_direct_record_mapping(record)) + b"\n" for record in records)
    summary = {
        "schema": CVOI_DIRECT_EPDMS_SUMMARY_SCHEMA,
        "branch": records[0].branch,
        "evaluation_seed": records[0].evaluation_seed,
        "scenario_count": len(records),
        "mean_epdms": sum(record.epdms for record in records) / len(records),
        "mean_latency_ms": sum(record.latency_ms for record in records) / len(records),
    }
    summary_bytes = _canonical_json_bytes(summary) + b"\n"
    _write_direct_outputs_exclusively(
        root,
        (
            (records_target, records_bytes),
            (summary_target, summary_bytes),
        ),
    )
    return records


def run_cvoi_direct_epdms(
    config_path: Path,
    *,
    forced_horizon: int | None,
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environ: Mapping[str, str] = os.environ,
) -> int:
    """Project and execute exactly one direct NavTest EPDMS run."""

    source = Path(config_path)
    if not source.is_absolute():
        raise ValueError(f"direct EPDMS config path must be absolute: {source}")
    if not callable(subprocess_run):
        raise TypeError("subprocess_run must be callable")
    if not isinstance(environ, Mapping) or any(
        type(key) is not str or type(value) is not str for key, value in environ.items()
    ):
        raise TypeError("environ must map strings to strings")

    config = load_cvoi_direct_epdms_config(source)
    _require_edited_public_execution_paths(config)
    projection = project_cvoi_direct_epdms_run(
        config,
        forced_horizon=forced_horizon,
    )
    output_directory = projection.output_directory
    if not output_directory.is_absolute():
        raise ValueError(f"direct EPDMS selected output directory must be absolute: {output_directory}")
    if os.path.lexists(output_directory):
        raise FileExistsError(f"direct EPDMS selected output already exists: {output_directory}")
    preflight_cvoi_direct_epdms_projection(projection)

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(output_directory)
    scorer_output_directory = output_directory / "scorer_output"
    trace_output_directory = output_directory / "policy_traces"
    os.mkdir(scorer_output_directory)
    os.mkdir(trace_output_directory)
    effective_path = write_cvoi_direct_epdms_projection(
        projection,
        output_directory / "cvoi_direct_epdms_effective.json",
    )

    child_environment = {key: value for key, value in environ.items() if key in _DIRECT_SUBPROCESS_ENV_ALLOWLIST}
    child_environment.update(
        {
            "CVOI_DIRECT_EPDMS": "1",
            "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH": str(effective_path),
            "NAVSIM_OUTPUT_DIR": str(scorer_output_directory),
        }
    )
    repository_root = Path(__file__).resolve().parents[3]
    scorer_script = repository_root / "scripts/eval_navsim/eval_navsim_v2_pdms.sh"
    if not scorer_script.is_absolute() or not scorer_script.is_file():
        raise RuntimeError(f"direct EPDMS scorer script is unavailable: {scorer_script}")
    experiment_name = (
        f"cvoi_direct_epdms_{projection.branch}"
        if projection.horizon is None
        else f"cvoi_direct_epdms_{projection.branch}_h{projection.horizon}"
    )
    command = [
        str(scorer_script),
        str(projection.encoder_checkpoint_path),
        str(projection.training_config_path),
        experiment_name,
    ]
    completed = subprocess_run(
        command,
        check=False,
        env=child_environment,
    )
    returncode = getattr(completed, "returncode", None)
    if type(returncode) is not int:
        raise TypeError("direct EPDMS scorer subprocess must expose one integer returncode")
    if returncode != 0:
        return returncode
    aggregate_direct_epdms_results(output_directory)
    return 0


__all__ = [
    "CVOI_DIRECT_EPDMS_BRANCHES",
    "CVOI_DIRECT_EPDMS_EFFECTIVE_SCHEMA",
    "CVOI_DIRECT_EPDMS_EFFECTIVE_VERSION",
    "CVOI_DIRECT_EPDMS_SCENE_SCHEMA",
    "CVOI_DIRECT_EPDMS_SCHEMA",
    "CVOI_DIRECT_EPDMS_SUMMARY_SCHEMA",
    "CVOI_DIRECT_EPDMS_TRACE_SCHEMA",
    "CVOI_DIRECT_EPDMS_TRACE_VERSION",
    "CVOI_DIRECT_EPDMS_VERSION",
    "CvoiDirectEpdmsArtifacts",
    "CvoiDirectEpdmsConfig",
    "CvoiDirectEpdmsProjection",
    "DirectEpdmsSceneRecord",
    "aggregate_direct_epdms_results",
    "load_cvoi_direct_epdms_config",
    "load_cvoi_direct_epdms_projection",
    "preflight_cvoi_direct_epdms_projection",
    "project_cvoi_direct_epdms_run",
    "read_cvoi_direct_epdms_scenario_manifest",
    "read_direct_epdms_records",
    "run_cvoi_direct_epdms",
    "validate_cvoi_direct_epdms_scenario_token",
    "write_cvoi_direct_epdms_projection",
]
