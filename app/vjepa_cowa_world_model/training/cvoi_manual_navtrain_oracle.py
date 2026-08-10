"""Manual NavTrain Oracle environment, fixed paths, and raw manifest data plane."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Optional

import yaml

from app.vjepa_cowa_world_model.training import cvoi_manual_lineage
from app.vjepa_cowa_world_model.training.configs.parse import parse_training_config
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_runtime import (
    resolve_formal_v2_navsim_e120_selected_checkpoint,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_E120_LAMBDA_GRID
from app.vjepa_cowa_world_model.training.cvoi_manual_lineage import CvoiManualValueLineage
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import (
    NAVTRAIN_GATE_PROTOCOL_ID,
    FeatureRow,
    FeatureStoreMetadata,
    OracleStoreMetadata,
    ScoreIdentity,
    ScoreStoreMetadata,
    StoreReceipt,
    create_embedded_oracle_store_v2,
    create_feature_store,
    create_score_store_from_official_csv,
    open_embedded_oracle_store_v2,
    open_feature_store,
    open_score_store,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import V2_PROTOCOL_ID
from app.vjepa_cowa_world_model.training.cvoi_navsim_scenarios import (
    CvoiNavSimScenarioManifest,
    read_cvoi_navsim_raw_v2_authority,
)
from src.utils.logging import get_logger
from tools.build_cvoi_navsim_scenario_manifest import build_bundle

logger = get_logger(__name__)

MANUAL_NAVTRAIN_SCORER_SCHEMA = "cvoi_manual_navtrain_gate_scorer_v1"
MANUAL_NAVTRAIN_POLICY_TRACE_SCHEMA = "cvoi_manual_navtrain_gate_policy_trace_v1"

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SCORER_SCRIPT_PATH = _REPOSITORY_ROOT / "scripts/eval_navsim/eval_navsim_v2_pdms.sh"
_SCORER_CONFIG_NAME = "scorer_config.json"
_EFFECTIVE_CONFIG_NAME = "effective.yaml"
_FEATURE_SCHEMA = "sequential_cvoi_gate_features_lambda_independent_h4_v1"
_FEATURE_SOURCES = (
    "pooled_observed",
    "pooled_prefix",
    "field_value",
    "stop_value",
    "stop_value_delta",
    "horizon",
    "current_cost",
    "next_cost",
)
_COMMON_RANDOM_SEED = 239
_SCORE_SEMANTICS = "official_v2_one_stage_ordinary_row_score"
_MANUAL_ORACLE_VALUE_LINEAGES = frozenset({"full", "no_cf"})
_MANUAL_ORACLE_LAMBDA_GRID = FORMAL_V2_NAVSIM_E120_LAMBDA_GRID
_FORBIDDEN_SCORER_CONFIG_SUBSTRINGS = (
    "receipt",
    "task_id",
    "source_commit",
    "sha256",
    "provenance",
)
_HANDOFF_FILENAMES = MappingProxyType(
    {
        "p0_planner_checkpoint": "p0_selected.pt",
        "field_checkpoint": "field.pt",
        "calibration_checkpoint": "calibration.pt",
        "p1_planner_checkpoint": "p1_selected.pt",
        "stop_checkpoint": "stop.pt",
    }
)
_ENVIRONMENT_PATH_FIELDS = MappingProxyType(
    {
        "CVOI_NAVSIM_DATA_ROOT": "data_root",
        "CVOI_NAVSIM_EXP_ROOT": "navsim_exp_root",
        "CVOI_NUPLAN_MAPS_ROOT": "maps_root",
        "CVOI_NAVSIM_METRIC_CACHE_ROOT": "metric_cache_root",
        "CVOI_NAVSIM_DEVKIT_ROOT": "devkit_root",
        "CVOI_NAVSIM_PYTHON_BIN": "python_bin",
    }
)
_DIRECTORY_ENVIRONMENT_NAMES = frozenset(
    {
        "CVOI_NAVSIM_DATA_ROOT",
        "CVOI_NAVSIM_EXP_ROOT",
        "CVOI_NUPLAN_MAPS_ROOT",
        "CVOI_NAVSIM_METRIC_CACHE_ROOT",
        "CVOI_NAVSIM_DEVKIT_ROOT",
    }
)
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
)


def _strict_existing_directory(path: Path, *, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{field} must be an absolute Path")
    if path.is_symlink():
        raise ValueError(f"{field} must be a canonical existing non-symlink directory: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{field} must be an existing directory: {path}") from exc
    if resolved != path or not resolved.is_dir():
        raise ValueError(f"{field} must be a canonical existing non-symlink directory: {path}")
    return resolved


def _strict_existing_regular_file(path: Path, *, field: str, executable: bool = False) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{field} must be an absolute Path")
    if path.is_symlink():
        raise ValueError(f"{field} must be a canonical existing non-symlink regular file: {path}")
    try:
        resolved = path.resolve(strict=True)
        file_stat = path.stat()
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{field} must be an existing regular file: {path}") from exc
    if resolved != path or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{field} must be a canonical existing non-symlink regular file: {path}")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"{field} must be executable: {path}")
    return resolved


def _required_environment_value(environ: Mapping[str, str], name: str) -> str:
    try:
        value = environ[name]
    except KeyError as exc:
        raise ValueError(f"missing required environment variable {name}") from exc
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True)
class ManualOracleEnvironment:
    """Six explicit NavSim machine paths and optional scorer CUDA selection."""

    data_root: Path
    navsim_exp_root: Path
    maps_root: Path
    metric_cache_root: Path
    devkit_root: Path
    python_bin: Path
    cuda_visible_devices: Optional[str]

    def __post_init__(self) -> None:
        for field_name in (
            "data_root",
            "navsim_exp_root",
            "maps_root",
            "metric_cache_root",
            "devkit_root",
        ):
            _strict_existing_directory(getattr(self, field_name), field=f"ManualOracleEnvironment.{field_name}")
        _strict_existing_regular_file(
            self.python_bin,
            field="ManualOracleEnvironment.python_bin",
            executable=True,
        )
        if self.cuda_visible_devices is not None and (
            type(self.cuda_visible_devices) is not str
            or not self.cuda_visible_devices
            or self.cuda_visible_devices != self.cuda_visible_devices.strip()
        ):
            raise ValueError("ManualOracleEnvironment.cuda_visible_devices must be None or a non-empty trimmed string")


def load_manual_oracle_environment(
    environ: Mapping[str, str],
    *,
    require_cuda: bool,
) -> ManualOracleEnvironment:
    """Load only the explicit manual-Oracle machine environment."""

    if not isinstance(environ, Mapping):
        raise ValueError("environ must be a mapping")
    if type(require_cuda) is not bool:
        raise ValueError("require_cuda must be a bool")

    validated: dict[str, Path] = {}
    for environment_name, field_name in _ENVIRONMENT_PATH_FIELDS.items():
        raw_value = _required_environment_value(environ, environment_name)
        raw_path = Path(raw_value)
        if environment_name in _DIRECTORY_ENVIRONMENT_NAMES:
            validated[field_name] = _strict_existing_directory(raw_path, field=environment_name)
        else:
            validated[field_name] = _strict_existing_regular_file(
                raw_path,
                field=environment_name,
                executable=True,
            )

    cuda_visible_devices = _required_environment_value(environ, "CUDA_VISIBLE_DEVICES") if require_cuda else None
    return ManualOracleEnvironment(
        data_root=validated["data_root"],
        navsim_exp_root=validated["navsim_exp_root"],
        maps_root=validated["maps_root"],
        metric_cache_root=validated["metric_cache_root"],
        devkit_root=validated["devkit_root"],
        python_bin=validated["python_bin"],
        cuda_visible_devices=cuda_visible_devices,
    )


@dataclass(frozen=True)
class ManualOracleSource:
    """Frozen Full/no-CF source authority for one manual NavTrain Oracle."""

    source_config_path: Path
    results_root: Path
    value_lineage: CvoiManualValueLineage
    lineage: str
    p0_planner_checkpoint: Path
    field_checkpoint: Path
    calibration_checkpoint: Path
    p1_planner_checkpoint: Path
    stop_checkpoint: Path
    oracle_path: Path
    lambda_grid: tuple[float, ...]

    def __post_init__(self) -> None:
        _strict_existing_regular_file(
            self.source_config_path,
            field="ManualOracleSource.source_config_path",
        )
        if not isinstance(self.value_lineage, CvoiManualValueLineage):
            raise ValueError("ManualOracleSource.value_lineage must be a CvoiManualValueLineage")
        if self.value_lineage.name not in _MANUAL_ORACLE_VALUE_LINEAGES:
            raise ValueError("manual NavTrain Oracle supports only Full and no-CF Value lineages")
        expected_lineage = f"p1_{self.value_lineage.name}"
        if self.lineage != expected_lineage:
            raise ValueError(f"ManualOracleSource.lineage must be {expected_lineage!r}")
        canonical_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
            phase="guided_planner",
            branch_id=self.lineage,
            full_results_root=self.value_lineage.p0_result_root,
            ablation_results_root=self.value_lineage.result_root.parent,
        )
        if self.value_lineage != canonical_lineage:
            raise ValueError("ManualOracleSource.value_lineage differs from the canonical lineage authority")
        if self.results_root != self.value_lineage.result_root:
            raise ValueError("ManualOracleSource.results_root differs from the fixed Value-lineage root")
        expected_paths = {
            "p0_planner_checkpoint": self.value_lineage.p0_handoff,
            "field_checkpoint": self.value_lineage.field_handoff,
            "calibration_checkpoint": self.value_lineage.calibration_handoff,
            "p1_planner_checkpoint": self.value_lineage.p1_handoff,
            "stop_checkpoint": self.value_lineage.stop_handoff,
            "oracle_path": self.value_lineage.oracle_handoff,
        }
        for field_name, expected_path in expected_paths.items():
            if getattr(self, field_name) != expected_path:
                raise ValueError(f"ManualOracleSource.{field_name} differs from fixed lineage authority")
        if self.lambda_grid != _MANUAL_ORACLE_LAMBDA_GRID:
            raise ValueError("ManualOracleSource.lambda_grid differs from the fixed manual Oracle lambda grid")

    @property
    def artifacts(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                "p0_planner_checkpoint": self.p0_planner_checkpoint,
                "field_checkpoint": self.field_checkpoint,
                "calibration_checkpoint": self.calibration_checkpoint,
                "p1_planner_checkpoint": self.p1_planner_checkpoint,
                "stop_checkpoint": self.stop_checkpoint,
            }
        )


@dataclass(frozen=True)
class ManualOraclePaths:
    """The one fixed filesystem layout shared by all manual Oracle actions."""

    results_root: Path

    def __post_init__(self) -> None:
        _strict_existing_directory(self.results_root, field="results_root")

    @classmethod
    def from_results_root(cls, results_root: Path) -> "ManualOraclePaths":
        return cls(results_root=_strict_existing_directory(results_root, field="results_root"))

    @property
    def manifest_dir(self) -> Path:
        return self.results_root / "oracle/work/manifest"

    def horizon_dir(self, horizon: int) -> Path:
        if type(horizon) is not int or horizon not in range(5):
            raise ValueError("horizon must be one of 0, 1, 2, 3, or 4")
        return self.results_root / f"oracle/work/h{horizon}"

    def score_store(self, horizon: int) -> Path:
        return self.horizon_dir(horizon) / "score_store.sqlite3"

    def feature_store(self, horizon: int) -> Path:
        return self.horizon_dir(horizon) / "feature_store.sqlite3"

    @property
    def oracle_handoff(self) -> Path:
        return self.results_root / "handoff/oracle_full.sqlite3"


def _validate_manual_handoff_artifacts(
    artifacts: Mapping[str, Path],
    *,
    source: ManualOracleSource,
) -> dict[str, Path]:
    """Validate fixed handoffs while preserving the operator-facing selected paths."""

    if not isinstance(source, ManualOracleSource):
        raise ValueError("source must be a ManualOracleSource")
    expected = dict(source.artifacts)
    if set(artifacts) != set(expected):
        raise ValueError("manual scorer artifacts must contain exactly five fixed roles")
    normalized: dict[str, Path] = {}
    for role, expected_path in expected.items():
        configured = artifacts[role]
        if not isinstance(configured, Path) or configured != expected_path:
            raise ValueError(
                f"manual scorer artifacts.{role} must use fixed handoff path " f"{expected_path}, got {configured!r}"
            )
        if role == "p0_planner_checkpoint":
            resolve_formal_v2_navsim_e120_selected_checkpoint(
                configured,
                results_root=source.value_lineage.p0_result_root,
                stage="p0",
            )
        elif role == "p1_planner_checkpoint":
            resolve_formal_v2_navsim_e120_selected_checkpoint(
                configured,
                results_root=source.results_root,
                stage="p1",
            )
        else:
            _strict_existing_regular_file(
                configured,
                field=f"manual scorer artifacts.{role}",
            )
        normalized[role] = configured
    return normalized


@dataclass(frozen=True)
class ManualNavTrainAuthorityBundle:
    """Typed lookup over the builder's raw three-file NavTrain authority."""

    root: Path
    protocol_id: str
    split: str
    data_root: Path
    navsim_exp_root: Path
    maps_root: Path
    metric_cache_root: Path
    devkit_root: Path
    scenario_manifest: CvoiNavSimScenarioManifest
    metric_cache_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        _strict_existing_directory(self.root, field="ManualNavTrainAuthorityBundle.root")
        if self.protocol_id != V2_PROTOCOL_ID or self.split != "navtrain":
            raise ValueError("manual NavTrain authority must use the registered V2 navtrain protocol")
        for field_name in (
            "data_root",
            "navsim_exp_root",
            "maps_root",
            "metric_cache_root",
            "devkit_root",
        ):
            _strict_existing_directory(
                getattr(self, field_name),
                field=f"ManualNavTrainAuthorityBundle.{field_name}",
            )
        if not isinstance(self.scenario_manifest, CvoiNavSimScenarioManifest):
            raise ValueError("scenario_manifest must be a CvoiNavSimScenarioManifest")
        if self.scenario_manifest.protocol_id != self.protocol_id:
            raise ValueError("scenario_manifest protocol differs from manual authority protocol")
        if not isinstance(self.metric_cache_paths, Mapping):
            raise ValueError("metric_cache_paths must be a mapping")
        paths = dict(self.metric_cache_paths)
        if tuple(paths) != self.scenario_manifest.tokens:
            raise ValueError("metric_cache_paths must follow the exact scenario-token order")
        for token, path in paths.items():
            strict_path = _strict_existing_regular_file(
                path,
                field=f"metric_cache_paths[{token!r}]",
            )
            if (
                not strict_path.is_relative_to(self.metric_cache_root)
                or strict_path.name != "metric_cache.pkl"
                or strict_path.parent.name != token
            ):
                raise ValueError("metric_cache_paths must be contained <token>/metric_cache.pkl files")
        object.__setattr__(self, "metric_cache_paths", MappingProxyType(paths))


@dataclass(frozen=True)
class ManualNavTrainScorerConfig:
    """Typed, receipt-free authority for one manually requested forced horizon."""

    schema: str
    protocol_id: str
    policy_id: str
    lineage: str
    planner_stage: str
    policy_mode: str
    forced_horizon: int
    guidance_steps: int
    common_random_seed: int
    artifacts: Mapping[str, Path]
    environment: ManualOracleEnvironment
    authority: ManualNavTrainAuthorityBundle
    source: ManualOracleSource
    effective_config_path: Path
    output_dir: Path
    trace_output_dir: Path
    score_store_path: Path
    feature_store_path: Path

    def __post_init__(self) -> None:
        if self.schema != MANUAL_NAVTRAIN_SCORER_SCHEMA:
            raise ValueError("manual scorer schema differs")
        if self.protocol_id != NAVTRAIN_GATE_PROTOCOL_ID:
            raise ValueError("manual scorer protocol_id differs")
        if not isinstance(self.source, ManualOracleSource):
            raise ValueError("manual scorer source must be a ManualOracleSource")
        _require_horizon(self.forced_horizon)
        expected_policy_id, expected_guidance_steps = _manual_policy_identity(
            self.forced_horizon,
            lineage=self.source.lineage,
        )
        if (
            self.policy_id != expected_policy_id
            or self.lineage != self.source.lineage
            or self.planner_stage != "p1"
            or self.policy_mode != "fixed_horizon"
            or self.guidance_steps != expected_guidance_steps
            or self.common_random_seed != _COMMON_RANDOM_SEED
        ):
            raise ValueError("manual scorer policy identity differs")
        if not isinstance(self.artifacts, Mapping) or set(self.artifacts) != set(_HANDOFF_FILENAMES):
            raise ValueError("manual scorer artifacts must contain exactly five fixed roles")
        if not isinstance(self.effective_config_path, Path) or not self.effective_config_path.is_absolute():
            raise ValueError("manual scorer effective_config_path must be an absolute Path")
        results_root = self.effective_config_path.parent.parents[2]
        if results_root != self.source.results_root:
            raise ValueError("manual scorer runtime root differs from source lineage root")
        normalized_artifacts = _validate_manual_handoff_artifacts(
            self.artifacts,
            source=self.source,
        )
        if type(self.environment) is not ManualOracleEnvironment:
            raise ValueError("manual scorer environment must be a ManualOracleEnvironment")
        _validate_live_environment(self.environment)
        if not isinstance(self.authority, ManualNavTrainAuthorityBundle):
            raise ValueError("manual scorer authority must be a ManualNavTrainAuthorityBundle")
        expected_roots = (
            self.environment.data_root,
            self.environment.navsim_exp_root,
            self.environment.maps_root,
            self.environment.metric_cache_root,
            self.environment.devkit_root,
        )
        authority_roots = (
            self.authority.data_root,
            self.authority.navsim_exp_root,
            self.authority.maps_root,
            self.authority.metric_cache_root,
            self.authority.devkit_root,
        )
        if authority_roots != expected_roots:
            raise ValueError("manual scorer environment differs from raw authority roots")
        if self.environment.cuda_visible_devices is None:
            raise ValueError("manual scorer environment requires CUDA_VISIBLE_DEVICES")
        _strict_existing_regular_file(self.effective_config_path, field="manual scorer effective config")
        _strict_existing_directory(self.output_dir, field="manual scorer output directory")
        _strict_existing_directory(self.trace_output_dir, field="manual scorer trace directory")
        horizon_dir = self.effective_config_path.parent
        if (
            self.effective_config_path != horizon_dir / _EFFECTIVE_CONFIG_NAME
            or self.output_dir != horizon_dir / "scorer_output"
            or self.trace_output_dir != horizon_dir / "policy_traces"
            or self.score_store_path != horizon_dir / "score_store.sqlite3"
            or self.feature_store_path != horizon_dir / "feature_store.sqlite3"
        ):
            raise ValueError("manual scorer runtime paths must use the fixed horizon-local layout")
        for field_name, store_path in (
            ("score_store_path", self.score_store_path),
            ("feature_store_path", self.feature_store_path),
        ):
            if not isinstance(store_path, Path) or not store_path.is_absolute():
                raise ValueError(f"manual scorer {field_name} must be an absolute Path")
            if store_path.exists() or store_path.is_symlink():
                _strict_existing_regular_file(store_path, field=f"manual scorer {field_name}")
        object.__setattr__(self, "artifacts", MappingProxyType(normalized_artifacts))

    @property
    def source_config_path(self) -> Path:
        return self.source.source_config_path


@dataclass(frozen=True)
class ManualNavTrainHorizonArtifacts:
    """Paths produced by one successful manual forced-horizon scoring action."""

    horizon: int
    scorer_config_path: Path
    effective_config_path: Path
    official_csv_path: Path
    score_store_path: Path
    feature_store_path: Path


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains forbidden non-finite value {value!r}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(raw: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be duplicate-free finite UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    if raw != _canonical_json(value):
        raise ValueError(f"{label} must use canonical compact JSON without a trailing newline")
    return value


def _exact_mapping(
    value: object,
    *,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{label} must be a string-keyed mapping")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{label} fields differ: missing={sorted(fields - actual)}, unknown={sorted(actual - fields)}"
        )
    return value


def _nonempty_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _lower_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


def _manifest_path(value: object, *, field: str, directory: bool) -> Path:
    raw = _nonempty_string(value, field=field)
    path = Path(raw)
    if directory:
        return _strict_existing_directory(path, field=field)
    return _strict_existing_regular_file(path, field=field)


def read_manual_navtrain_authority_bundle(root: Path) -> ManualNavTrainAuthorityBundle:
    """Read the retained raw V2 NavTrain authority into the manual-oracle DTO."""

    authority = read_cvoi_navsim_raw_v2_authority(root, expected_split="navtrain")
    return ManualNavTrainAuthorityBundle(
        root=authority.root,
        protocol_id=authority.protocol_id,
        split=authority.split,
        data_root=authority.data_root,
        navsim_exp_root=authority.navsim_exp_root,
        maps_root=authority.maps_root,
        metric_cache_root=authority.metric_cache_root,
        devkit_root=authority.devkit_root,
        scenario_manifest=authority.scenario_manifest,
        metric_cache_paths=authority.metric_cache_paths,
    )


def _validate_live_environment(environment: ManualOracleEnvironment) -> None:
    if type(environment) is not ManualOracleEnvironment:
        raise ValueError("environment must be an exact ManualOracleEnvironment")
    for field_name in (
        "data_root",
        "navsim_exp_root",
        "maps_root",
        "metric_cache_root",
        "devkit_root",
    ):
        _strict_existing_directory(
            getattr(environment, field_name),
            field=f"environment.{field_name}",
        )
    _strict_existing_regular_file(
        environment.python_bin,
        field="environment.python_bin",
        executable=True,
    )


def _open_or_create_directory_at(parent_fd: int, name: str, *, field: str) -> int:
    created = False
    try:
        os.mkdir(name, mode=0o755, dir_fd=parent_fd)
        created = True
    except FileExistsError:
        pass
    except OSError as exc:
        raise ValueError(f"{field} could not be created") from exc
    if created:
        os.fsync(parent_fd)
    try:
        entry_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{field} must be an existing non-symlink directory") from exc
    if not stat.S_ISDIR(entry_stat.st_mode):
        raise ValueError(f"{field} must be an existing non-symlink directory")
    try:
        directory_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise ValueError(f"{field} must be an existing non-symlink directory") from exc
    opened_stat = os.fstat(directory_fd)
    if (entry_stat.st_dev, entry_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
        os.close(directory_fd)
        raise ValueError(f"{field} changed while being opened")
    return directory_fd


def _require_absent_entry(parent_fd: int, name: str, *, field: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ValueError(f"{field} could not be inspected") from exc
    raise FileExistsError(f"{field} already exists or is a symlink")


def build_manual_navtrain_manifest(
    results_root: Path,
    environment: ManualOracleEnvironment,
    *,
    bundle_builder: Callable[[argparse.Namespace], Path] = build_bundle,
) -> ManualNavTrainAuthorityBundle:
    """Build exactly one raw V2/navtrain authority bundle under the fixed layout."""

    _validate_live_environment(environment)
    if not callable(bundle_builder):
        raise ValueError("bundle_builder must be callable")
    paths = ManualOraclePaths.from_results_root(results_root)
    try:
        results_fd = os.open(paths.results_root, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise ValueError("results_root must be an openable non-symlink directory") from exc
    oracle_fd = -1
    work_fd = -1
    try:
        oracle_fd = _open_or_create_directory_at(results_fd, "oracle", field="results_root/oracle")
        work_fd = _open_or_create_directory_at(oracle_fd, "work", field="results_root/oracle/work")
        work_stat = os.fstat(work_fd)
        if work_stat.st_uid != os.geteuid() or work_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError("results_root/oracle/work must be owned by this uid and not group/world writable")
        _require_absent_entry(work_fd, "manifest", field="manual navtrain manifest")
        args = argparse.Namespace(
            protocol_id=V2_PROTOCOL_ID,
            split="navtrain",
            devkit_root=environment.devkit_root,
            data_root=environment.data_root,
            navsim_exp_root=environment.navsim_exp_root,
            maps_root=environment.maps_root,
            metric_cache_root=environment.metric_cache_root,
            output_dir=paths.manifest_dir,
            output_parent_fd=work_fd,
            token_subset_path=None,
        )
        built_path = bundle_builder(args)
        if not isinstance(built_path, Path) or built_path != paths.manifest_dir:
            raise ValueError("bundle_builder must return the exact fixed manifest_dir")
        authority = read_manual_navtrain_authority_bundle(paths.manifest_dir)
        expected_roots = (
            environment.data_root,
            environment.navsim_exp_root,
            environment.maps_root,
            environment.metric_cache_root,
            environment.devkit_root,
        )
        actual_roots = (
            authority.data_root,
            authority.navsim_exp_root,
            authority.maps_root,
            authority.metric_cache_root,
            authority.devkit_root,
        )
        if actual_roots != expected_roots:
            raise ValueError("raw manual authority roots differ from the validated environment")
        return authority
    finally:
        if work_fd >= 0:
            os.close(work_fd)
        if oracle_fd >= 0:
            os.close(oracle_fd)
        os.close(results_fd)


def _require_horizon(value: object) -> int:
    if type(value) is not int or value not in range(5):
        raise ValueError("horizon must be one of 0, 1, 2, 3, or 4")
    return value


def _manual_policy_identity(horizon: int, *, lineage: str) -> tuple[str, int]:
    horizon = _require_horizon(horizon)
    if lineage not in {"p1_full", "p1_no_cf"}:
        raise ValueError("manual NavTrain Oracle lineage must be 'p1_full' or 'p1_no_cf'")
    guidance_steps = 0 if horizon == 0 else 2
    return f"{lineage}__fixed_h{horizon}_k{guidance_steps}", guidance_steps


def _hash_file(path: Path, *, field: str) -> str:
    path = _strict_existing_regular_file(path, field=field)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader which also rejects duplicate mapping keys."""


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"YAML contains duplicate key {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


def _read_yaml_mapping(path: Path, *, field: str) -> dict[str, object]:
    path = _strict_existing_regular_file(path, field=field)
    try:
        value = yaml.load(path.read_bytes(), Loader=_UniqueKeySafeLoader)
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{field} must be valid duplicate-free YAML") from exc
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{field} must contain one string-keyed mapping")
    return dict(value)


def load_manual_oracle_source(
    source_config_path: Path,
    *,
    results_root: Path,
) -> ManualOracleSource:
    """Load one explicit Full/no-CF P1 YAML as the complete Oracle authority."""

    if not isinstance(source_config_path, Path):
        raise ValueError("manual Oracle source_config_path must be a Path")
    normalized_source_path = (
        source_config_path if source_config_path.is_absolute() else _REPOSITORY_ROOT / source_config_path
    )
    normalized_source_path = _strict_existing_regular_file(
        normalized_source_path,
        field="manual Oracle source config",
    )
    source_bytes = normalized_source_path.read_bytes()
    raw_source = _read_yaml_mapping(normalized_source_path, field="manual Oracle source config")
    parsed = parse_training_config(copy.deepcopy(raw_source))
    if parsed.cvoi.stage != "guided_planner":
        raise ValueError("manual Oracle source must be a guided_planner P1 config")
    signature = parsed.cvoi.ablation_signature
    full_results_root = cvoi_manual_lineage.resolve_cvoi_manual_full_results_root_from_config(parsed.cvoi)
    ablation_results_root = None
    if signature.experiment_role == "ablation":
        ablation_results_root = cvoi_manual_lineage.resolve_cvoi_manual_ablation_results_root_from_config(parsed.cvoi)
    value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage(
        signature,
        stage="guided_planner",
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )
    if value_lineage.name not in _MANUAL_ORACLE_VALUE_LINEAGES:
        raise ValueError("manual NavTrain Oracle supports only Full and no-CF Value lineages")

    normalized_results_root = _strict_existing_directory(results_root, field="results_root")
    if normalized_results_root != value_lineage.result_root:
        raise ValueError(
            "results_root differs from the fixed manual Oracle lineage root: "
            f"expected={value_lineage.result_root}, got={normalized_results_root}"
        )

    expected_config_paths = {
        "folder": value_lineage.result_root / "p1",
        "cvoi.unguided_planner_checkpoint": value_lineage.p0_handoff,
        "cvoi.field_checkpoint": value_lineage.calibration_handoff,
        "cvoi.output_checkpoint": value_lineage.result_root / "p1/p1_planner_checkpoint.pt",
    }
    configured_paths = {
        "folder": parsed.meta.folder,
        "cvoi.unguided_planner_checkpoint": parsed.cvoi.unguided_planner_checkpoint,
        "cvoi.field_checkpoint": parsed.cvoi.field_checkpoint,
        "cvoi.output_checkpoint": parsed.cvoi.output_checkpoint,
    }
    for field_name, expected_path in expected_config_paths.items():
        if configured_paths[field_name] != str(expected_path):
            raise ValueError(
                f"manual Oracle source {field_name} must be the fixed {value_lineage.name} path "
                f"{expected_path}, got {configured_paths[field_name]!r}"
            )

    lambda_grid = tuple(float(value) for value in parsed.cvoi.lambda_grid)
    if lambda_grid != _MANUAL_ORACLE_LAMBDA_GRID:
        raise ValueError(
            "manual Oracle source cvoi.lambda_grid must be exactly " f"{list(_MANUAL_ORACLE_LAMBDA_GRID)!r}"
        )
    if normalized_source_path.read_bytes() != source_bytes:
        raise RuntimeError("manual Oracle source config changed while loading")

    return ManualOracleSource(
        source_config_path=normalized_source_path,
        results_root=value_lineage.result_root,
        value_lineage=value_lineage,
        lineage=f"p1_{value_lineage.name}",
        p0_planner_checkpoint=value_lineage.p0_handoff,
        field_checkpoint=value_lineage.field_handoff,
        calibration_checkpoint=value_lineage.calibration_handoff,
        p1_planner_checkpoint=value_lineage.p1_handoff,
        stop_checkpoint=value_lineage.stop_handoff,
        oracle_path=value_lineage.oracle_handoff,
        lambda_grid=lambda_grid,
    )


def _materialize_manual_evaluation_projection(
    path: Path,
    *,
    source: ManualOracleSource,
) -> None:
    source_bytes = _strict_existing_regular_file(
        source.source_config_path,
        field="manual P1 source config",
    ).read_bytes()
    source_mapping = _read_yaml_mapping(source.source_config_path, field="manual P1 source config")
    cvoi = source_mapping.get("cvoi")
    if not isinstance(cvoi, Mapping):
        raise ValueError("manual P1 source config must contain cvoi")
    cvoi = dict(cvoi)
    signature = cvoi.get("ablation_signature")
    if (
        cvoi.get("stage") != "guided_planner"
        or cvoi.get("controller_lineage") != "value_guided"
        or not isinstance(signature, Mapping)
        or signature.get("branch_id") != source.lineage
    ):
        raise ValueError("manual P1 source config does not preserve its validated value-guided lineage")
    projected = copy.deepcopy(source_mapping)
    projected_cvoi = projected["cvoi"]
    if not isinstance(projected_cvoi, dict):
        raise RuntimeError("copied manual P1 cvoi config changed type")
    projected_cvoi.update(
        {
            "stage": "evaluation",
            "evaluation_mode": "p1_field_forced",
            "controller_lineage": "value_guided",
            "unguided_planner_checkpoint": str(source.p0_planner_checkpoint),
            "field_checkpoint": str(source.calibration_checkpoint),
            "guided_planner_checkpoint": str(source.p1_planner_checkpoint),
            "dual_value_checkpoint": str(source.stop_checkpoint),
            "output_checkpoint": None,
        }
    )
    if "forced_horizon" in projected_cvoi:
        raise ValueError("manual evaluation YAML must not carry a forced_horizon field")
    output = yaml.safe_dump(
        projected,
        allow_unicode=False,
        default_flow_style=False,
        sort_keys=False,
    ).encode("utf-8")
    try:
        with path.open("xb") as stream:
            stream.write(output)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise
    if source.source_config_path.read_bytes() != source_bytes:
        raise RuntimeError("manual P1 source config changed during projection")
    if _read_yaml_mapping(path, field="manual effective config") != projected:
        raise RuntimeError("manual evaluation projection changed during YAML reload")


def _manual_scorer_config_mapping(
    *,
    paths: ManualOraclePaths,
    source: ManualOracleSource,
    environment: ManualOracleEnvironment,
    authority: ManualNavTrainAuthorityBundle,
    horizon: int,
) -> dict[str, object]:
    policy_id, guidance_steps = _manual_policy_identity(horizon, lineage=source.lineage)
    horizon_dir = paths.horizon_dir(horizon)
    return {
        "schema": MANUAL_NAVTRAIN_SCORER_SCHEMA,
        "protocol_id": NAVTRAIN_GATE_PROTOCOL_ID,
        "policy": {
            "policy_id": policy_id,
            "lineage": source.lineage,
            "planner_stage": "p1",
            "policy_mode": "fixed_horizon",
            "forced_horizon": horizon,
            "guidance_steps": guidance_steps,
            "common_random_seed": _COMMON_RANDOM_SEED,
        },
        "artifacts": {role: str(path) for role, path in source.artifacts.items()},
        "environment": {
            "data_root": str(environment.data_root),
            "navsim_exp_root": str(environment.navsim_exp_root),
            "maps_root": str(environment.maps_root),
            "metric_cache_root": str(environment.metric_cache_root),
            "devkit_root": str(environment.devkit_root),
            "python_bin": str(environment.python_bin),
            "cuda_visible_devices": environment.cuda_visible_devices,
        },
        "scenario": {
            "authority_root": str(authority.root),
            "scenario_manifest_path": str(authority.root / "scenario_manifest.jsonl"),
            "metric_cache_inventory_path": str(authority.root / "metric_cache_inventory.json"),
            "scenario_count": len(authority.scenario_manifest.scenarios),
        },
        "runtime": {
            "source_config_path": str(source.source_config_path),
            "effective_config_path": str(horizon_dir / _EFFECTIVE_CONFIG_NAME),
            "output_dir": str(horizon_dir / "scorer_output"),
            "trace_output_dir": str(horizon_dir / "policy_traces"),
            "score_store_path": str(paths.score_store(horizon)),
            "feature_store_path": str(paths.feature_store(horizon)),
            "max_workers": 1,
            "use_process_pool": False,
            "forward_mode": "stage12",
        },
    }


def _path_from_json(value: object, *, field: str) -> Path:
    raw = _nonempty_string(value, field=field)
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path string")
    return path


def read_manual_navtrain_scorer_config(path: Path) -> ManualNavTrainScorerConfig:
    """Read one canonical manual scorer JSON into a strict typed runtime authority."""

    path = _strict_existing_regular_file(path, field="manual scorer config")
    raw_bytes = path.read_bytes()
    try:
        ascii_text = raw_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("manual scorer config must be ASCII JSON") from exc
    forbidden = [term for term in _FORBIDDEN_SCORER_CONFIG_SUBSTRINGS if term in ascii_text]
    if forbidden:
        raise ValueError(f"manual scorer config contains forbidden substrings: {forbidden}")
    raw = _exact_mapping(
        _parse_canonical_json(raw_bytes, label="manual scorer config"),
        fields=frozenset({"schema", "protocol_id", "policy", "artifacts", "environment", "scenario", "runtime"}),
        label="manual scorer config",
    )
    if raw["schema"] != MANUAL_NAVTRAIN_SCORER_SCHEMA:
        raise ValueError("manual scorer config schema differs")
    if raw["protocol_id"] != NAVTRAIN_GATE_PROTOCOL_ID:
        raise ValueError("manual scorer config protocol_id differs")

    policy = _exact_mapping(
        raw["policy"],
        fields=frozenset(
            {
                "policy_id",
                "lineage",
                "planner_stage",
                "policy_mode",
                "forced_horizon",
                "guidance_steps",
                "common_random_seed",
            }
        ),
        label="manual scorer config policy",
    )
    horizon = _require_horizon(policy["forced_horizon"])

    artifacts_raw = _exact_mapping(
        raw["artifacts"],
        fields=frozenset(_HANDOFF_FILENAMES),
        label="manual scorer config artifacts",
    )
    artifacts = {
        role: _path_from_json(artifacts_raw[role], field=f"manual scorer config artifacts.{role}")
        for role in _HANDOFF_FILENAMES
    }
    environment_raw = _exact_mapping(
        raw["environment"],
        fields=frozenset(
            {
                "data_root",
                "navsim_exp_root",
                "maps_root",
                "metric_cache_root",
                "devkit_root",
                "python_bin",
                "cuda_visible_devices",
            }
        ),
        label="manual scorer config environment",
    )
    environment = ManualOracleEnvironment(
        data_root=_path_from_json(environment_raw["data_root"], field="manual scorer config environment.data_root"),
        navsim_exp_root=_path_from_json(
            environment_raw["navsim_exp_root"],
            field="manual scorer config environment.navsim_exp_root",
        ),
        maps_root=_path_from_json(environment_raw["maps_root"], field="manual scorer config environment.maps_root"),
        metric_cache_root=_path_from_json(
            environment_raw["metric_cache_root"],
            field="manual scorer config environment.metric_cache_root",
        ),
        devkit_root=_path_from_json(
            environment_raw["devkit_root"],
            field="manual scorer config environment.devkit_root",
        ),
        python_bin=_path_from_json(
            environment_raw["python_bin"],
            field="manual scorer config environment.python_bin",
        ),
        cuda_visible_devices=_nonempty_string(
            environment_raw["cuda_visible_devices"],
            field="manual scorer config environment.cuda_visible_devices",
        ),
    )

    scenario = _exact_mapping(
        raw["scenario"],
        fields=frozenset(
            {
                "authority_root",
                "scenario_manifest_path",
                "metric_cache_inventory_path",
                "scenario_count",
            }
        ),
        label="manual scorer config scenario",
    )
    authority_root = _path_from_json(
        scenario["authority_root"],
        field="manual scorer config scenario.authority_root",
    )
    authority = read_manual_navtrain_authority_bundle(authority_root)
    if (
        _path_from_json(
            scenario["scenario_manifest_path"],
            field="manual scorer config scenario.scenario_manifest_path",
        )
        != authority.root / "scenario_manifest.jsonl"
        or _path_from_json(
            scenario["metric_cache_inventory_path"],
            field="manual scorer config scenario.metric_cache_inventory_path",
        )
        != authority.root / "metric_cache_inventory.json"
        or type(scenario["scenario_count"]) is not int
        or scenario["scenario_count"] != len(authority.scenario_manifest.scenarios)
    ):
        raise ValueError("manual scorer config scenario lookup differs from raw authority")

    runtime = _exact_mapping(
        raw["runtime"],
        fields=frozenset(
            {
                "source_config_path",
                "effective_config_path",
                "output_dir",
                "trace_output_dir",
                "score_store_path",
                "feature_store_path",
                "max_workers",
                "use_process_pool",
                "forward_mode",
            }
        ),
        label="manual scorer config runtime",
    )
    if (
        runtime["max_workers"] != 1
        or type(runtime["max_workers"]) is not int
        or runtime["use_process_pool"] is not False
        or runtime["forward_mode"] != "stage12"
    ):
        raise ValueError("manual scorer config runtime execution settings differ")
    source_config_path = _path_from_json(
        runtime["source_config_path"],
        field="manual scorer config runtime.source_config_path",
    )
    effective_config_path = _path_from_json(
        runtime["effective_config_path"],
        field="manual scorer config runtime.effective_config_path",
    )
    results_root = effective_config_path.parent.parents[2]
    source = load_manual_oracle_source(
        source_config_path,
        results_root=results_root,
    )
    expected_policy_id, expected_guidance_steps = _manual_policy_identity(
        horizon,
        lineage=source.lineage,
    )
    if policy != {
        "policy_id": expected_policy_id,
        "lineage": source.lineage,
        "planner_stage": "p1",
        "policy_mode": "fixed_horizon",
        "forced_horizon": horizon,
        "guidance_steps": expected_guidance_steps,
        "common_random_seed": _COMMON_RANDOM_SEED,
    }:
        raise ValueError("manual scorer config policy differs from its explicit P1 source")
    config = ManualNavTrainScorerConfig(
        schema=raw["schema"],
        protocol_id=raw["protocol_id"],
        policy_id=policy["policy_id"],
        lineage=policy["lineage"],
        planner_stage=policy["planner_stage"],
        policy_mode=policy["policy_mode"],
        forced_horizon=horizon,
        guidance_steps=policy["guidance_steps"],
        common_random_seed=policy["common_random_seed"],
        artifacts=artifacts,
        environment=environment,
        authority=authority,
        source=source,
        effective_config_path=effective_config_path,
        output_dir=_path_from_json(runtime["output_dir"], field="manual scorer config runtime.output_dir"),
        trace_output_dir=_path_from_json(
            runtime["trace_output_dir"],
            field="manual scorer config runtime.trace_output_dir",
        ),
        score_store_path=_path_from_json(
            runtime["score_store_path"],
            field="manual scorer config runtime.score_store_path",
        ),
        feature_store_path=_path_from_json(
            runtime["feature_store_path"],
            field="manual scorer config runtime.feature_store_path",
        ),
    )
    horizon_dir = config.effective_config_path.parent
    expected_paths = ManualOraclePaths.from_results_root(results_root)
    if (
        horizon_dir != expected_paths.horizon_dir(horizon)
        or path != horizon_dir / _SCORER_CONFIG_NAME
        or config.authority.root != expected_paths.manifest_dir
        or dict(config.artifacts) != dict(source.artifacts)
    ):
        raise ValueError("manual scorer config paths differ from the fixed results layout")
    _validate_manual_evaluation_projection(config)
    return config


def _validate_manual_evaluation_projection(config: ManualNavTrainScorerConfig) -> None:
    source = _read_yaml_mapping(config.source_config_path, field="manual scorer source config")
    projected = _read_yaml_mapping(config.effective_config_path, field="manual scorer effective config")
    source_cvoi = source.get("cvoi")
    projected_cvoi = projected.get("cvoi")
    if not isinstance(source_cvoi, Mapping) or not isinstance(projected_cvoi, Mapping):
        raise ValueError("manual scorer configs must contain cvoi mappings")
    expected = copy.deepcopy(source)
    expected_cvoi = expected["cvoi"]
    if not isinstance(expected_cvoi, dict):
        raise RuntimeError("manual scorer source cvoi changed type")
    expected_cvoi.update(
        {
            "stage": "evaluation",
            "evaluation_mode": "p1_field_forced",
            "controller_lineage": "value_guided",
            "unguided_planner_checkpoint": str(config.artifacts["p0_planner_checkpoint"]),
            "field_checkpoint": str(config.artifacts["calibration_checkpoint"]),
            "guided_planner_checkpoint": str(config.artifacts["p1_planner_checkpoint"]),
            "dual_value_checkpoint": str(config.artifacts["stop_checkpoint"]),
            "output_checkpoint": None,
        }
    )
    if projected != expected or "forced_horizon" in projected_cvoi:
        raise ValueError("manual scorer effective config is not the exact allowlisted P1 evaluation projection")


def _write_manual_scorer_config(
    path: Path,
    *,
    value: Mapping[str, object],
) -> ManualNavTrainScorerConfig:
    data = _canonical_json(value)
    text = data.decode("ascii")
    forbidden = [term for term in _FORBIDDEN_SCORER_CONFIG_SUBSTRINGS if term in text]
    if forbidden:
        raise ValueError(f"manual scorer config contains forbidden substrings: {forbidden}")
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return read_manual_navtrain_scorer_config(path)


def _manual_scorer_environment(
    config: ManualNavTrainScorerConfig,
    *,
    scorer_config_path: Path,
) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{_REPOSITORY_ROOT}:{config.environment.devkit_root}",
        "CUDA_VISIBLE_DEVICES": str(config.environment.cuda_visible_devices),
        "OPENSCENE_DATA_ROOT": str(config.environment.data_root),
        "NAVSIM_EXP_ROOT": str(config.environment.navsim_exp_root),
        "NUPLAN_MAPS_ROOT": str(config.environment.maps_root),
        "NAVSIM_DEVKIT_ROOT": str(config.environment.devkit_root),
        "METRIC_CACHE_PATH": str(config.environment.metric_cache_root),
        "PYTHON_BIN": str(config.environment.python_bin),
        "MAX_WORKERS": "1",
        "USE_PROCESS_POOL": "false",
        "FORWARD_MODE": "stage12",
        "PROPOSAL_CHECKPOINT": "",
        "NAVSIM_OUTPUT_DIR": str(config.output_dir),
        "CVOI_MANUAL_NAVTRAIN_GATE": "1",
        "CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH": str(scorer_config_path),
    }


def _one_official_csv(output_dir: Path) -> Path:
    output_dir = _strict_existing_directory(output_dir, field="manual scorer output directory")
    candidates: list[Path] = []
    for path in output_dir.rglob("*.csv"):
        candidates.append(_strict_existing_regular_file(path, field="manual official scorer CSV"))
    if len(candidates) != 1:
        raise ValueError(f"manual scorer must emit exactly one official CSV, found {len(candidates)}")
    return candidates[0]


def _manual_feature_rows(config: ManualNavTrainScorerConfig) -> Iterator[FeatureRow]:
    entries = tuple(config.trace_output_dir.iterdir())
    scenarios = config.authority.scenario_manifest.scenarios
    expected_names = {f"{scenario.observation_key}.json" for scenario in scenarios}
    if {entry.name for entry in entries} != expected_names:
        raise ValueError("manual policy traces must exactly cover the raw NavTrain cohort")
    expected_fields = frozenset(
        {
            "schema",
            "protocol_id",
            "scenario_token",
            "observation_key",
            "policy_id",
            "lineage",
            "horizon",
            "gate_features",
            "observed_feature_sha256",
        }
    )
    for row_index, scenario in enumerate(scenarios):
        trace_path = config.trace_output_dir / f"{scenario.observation_key}.json"
        trace = _exact_mapping(
            _parse_canonical_json(
                _strict_existing_regular_file(
                    trace_path,
                    field=f"manual policy trace {scenario.scenario_token}",
                ).read_bytes(),
                label=f"manual policy trace {scenario.scenario_token}",
            ),
            fields=expected_fields,
            label=f"manual policy trace {scenario.scenario_token}",
        )
        expected_identity = {
            "schema": MANUAL_NAVTRAIN_POLICY_TRACE_SCHEMA,
            "protocol_id": NAVTRAIN_GATE_PROTOCOL_ID,
            "scenario_token": scenario.scenario_token,
            "observation_key": scenario.observation_key,
            "policy_id": config.policy_id,
            "lineage": config.lineage,
            "horizon": config.forced_horizon,
        }
        if any(trace[field] != expected for field, expected in expected_identity.items()):
            raise ValueError(f"manual policy trace identity differs for token {scenario.scenario_token!r}")
        observed_digest = _lower_sha256(
            trace["observed_feature_sha256"],
            field=f"manual policy trace {scenario.scenario_token} observed_feature_sha256",
        )
        raw_features = trace["gate_features"]
        if isinstance(raw_features, (str, bytes)) or not isinstance(raw_features, Sequence) or not raw_features:
            raise ValueError("manual policy trace gate_features must be a non-empty sequence")
        features: list[float] = []
        for feature_index, raw_feature in enumerate(raw_features):
            if isinstance(raw_feature, bool) or not isinstance(raw_feature, (int, float)):
                raise ValueError(f"manual Gate feature {feature_index} must be a finite number")
            feature = float(raw_feature)
            if not math.isfinite(feature):
                raise ValueError(f"manual Gate feature {feature_index} must be finite")
            features.append(feature)
        yield FeatureRow(
            row_index=row_index,
            token=scenario.scenario_token,
            observation_key=scenario.observation_key,
            observed_feature_sha256=observed_digest,
            features=tuple(features),
        )


def _import_manual_horizon_outputs(
    config: ManualNavTrainScorerConfig,
    *,
    official_csv_path: Path,
) -> None:
    scenario_path = config.authority.root / "scenario_manifest.jsonl"
    inventory_path = config.authority.root / "metric_cache_inventory.json"
    scenario_digest = _hash_file(scenario_path, field="manual scenario manifest")
    inventory_digest = _hash_file(inventory_path, field="manual metric-cache inventory")
    create_feature_store(
        config.feature_store_path,
        FeatureStoreMetadata(
            protocol_id=NAVTRAIN_GATE_PROTOCOL_ID,
            policy_id=config.policy_id,
            lineage=config.lineage,
            horizon=config.forced_horizon,
            scenario_manifest_sha256=scenario_digest,
            metric_cache_inventory_sha256=inventory_digest,
            feature_schema=_FEATURE_SCHEMA,
            feature_sources=_FEATURE_SOURCES,
            common_random_seed=config.common_random_seed,
        ),
        _manual_feature_rows(config),
    )
    create_score_store_from_official_csv(
        config.score_store_path,
        ScoreStoreMetadata(
            protocol_id=NAVTRAIN_GATE_PROTOCOL_ID,
            policy_id=config.policy_id,
            lineage=config.lineage,
            horizon=config.forced_horizon,
            scenario_manifest_sha256=scenario_digest,
            metric_cache_inventory_sha256=inventory_digest,
            source_path=official_csv_path,
            source_sha256=_hash_file(official_csv_path, field="manual official scorer CSV"),
            score_semantics=_SCORE_SEMANTICS,
        ),
        identities=(
            ScoreIdentity(row_index, scenario.scenario_token, scenario.observation_key, scenario.log_name)
            for row_index, scenario in enumerate(config.authority.scenario_manifest.scenarios)
        ),
    )
    with (
        open_score_store(config.score_store_path) as score_store,
        open_feature_store(config.feature_store_path) as feature_store,
    ):
        expected_identity = (
            NAVTRAIN_GATE_PROTOCOL_ID,
            config.policy_id,
            config.lineage,
            config.forced_horizon,
            scenario_digest,
            inventory_digest,
        )
        if (
            score_store.metadata.protocol_id,
            score_store.metadata.policy_id,
            score_store.metadata.lineage,
            score_store.metadata.horizon,
            score_store.metadata.scenario_manifest_sha256,
            score_store.metadata.metric_cache_inventory_sha256,
        ) != expected_identity or (
            feature_store.metadata.protocol_id,
            feature_store.metadata.policy_id,
            feature_store.metadata.lineage,
            feature_store.metadata.horizon,
            feature_store.metadata.scenario_manifest_sha256,
            feature_store.metadata.metric_cache_inventory_sha256,
        ) != expected_identity:
            raise RuntimeError("manual score and feature store identities differ after import")
        expected_rows = len(config.authority.scenario_manifest.scenarios)
        if score_store.row_count != expected_rows or feature_store.row_count != expected_rows:
            raise RuntimeError("manual score and feature store row counts differ from raw authority")


def score_manual_navtrain_horizon(
    results_root: Path,
    environment: ManualOracleEnvironment,
    horizon: int,
    *,
    source_config_path: Path,
    runner: Callable[..., object] = subprocess.run,
) -> ManualNavTrainHorizonArtifacts:
    """Run exactly one requested H0--H4 scorer and import its v1 stores."""

    horizon = _require_horizon(horizon)
    _validate_live_environment(environment)
    if environment.cuda_visible_devices is None:
        raise ValueError("manual horizon scoring requires CUDA_VISIBLE_DEVICES")
    if not callable(runner):
        raise ValueError("runner must be callable")
    source = load_manual_oracle_source(
        source_config_path,
        results_root=results_root,
    )
    paths = ManualOraclePaths.from_results_root(results_root)
    try:
        authority = read_manual_navtrain_authority_bundle(paths.manifest_dir)
    except ValueError as exc:
        raise ValueError("manual NavTrain manifest is missing or invalid") from exc
    expected_roots = (
        environment.data_root,
        environment.navsim_exp_root,
        environment.maps_root,
        environment.metric_cache_root,
        environment.devkit_root,
    )
    authority_roots = (
        authority.data_root,
        authority.navsim_exp_root,
        authority.maps_root,
        authority.metric_cache_root,
        authority.devkit_root,
    )
    if authority_roots != expected_roots:
        raise ValueError("manual scoring environment differs from manifest authority")
    artifacts = _validate_manual_handoff_artifacts(
        source.artifacts,
        source=source,
    )
    scorer_script = _strict_existing_regular_file(
        _SCORER_SCRIPT_PATH,
        field="manual NavSim scorer script",
        executable=True,
    )
    horizon_dir = paths.horizon_dir(horizon)
    if horizon_dir.exists() or horizon_dir.is_symlink():
        raise FileExistsError(f"manual horizon work h{horizon} already exists")
    work_dir = _strict_existing_directory(horizon_dir.parent, field="manual Oracle work directory")
    if work_dir != paths.results_root / "oracle/work":
        raise ValueError("manual Oracle work directory differs from the fixed layout")
    horizon_dir.mkdir(mode=0o750)
    output_dir = horizon_dir / "scorer_output"
    trace_dir = horizon_dir / "policy_traces"
    output_dir.mkdir(mode=0o750)
    trace_dir.mkdir(mode=0o750)
    effective_path = horizon_dir / _EFFECTIVE_CONFIG_NAME
    _materialize_manual_evaluation_projection(effective_path, source=source)
    scorer_config_path = horizon_dir / _SCORER_CONFIG_NAME
    config = _write_manual_scorer_config(
        scorer_config_path,
        value=_manual_scorer_config_mapping(
            paths=paths,
            source=source,
            environment=environment,
            authority=authority,
            horizon=horizon,
        ),
    )
    command = [
        str(scorer_script),
        str(artifacts["p1_planner_checkpoint"]),
        str(effective_path),
        f"cvoi-manual-oracle-h{horizon}",
    ]
    completed = runner(
        command,
        cwd=_REPOSITORY_ROOT,
        env=_manual_scorer_environment(config, scorer_config_path=scorer_config_path),
        check=False,
    )
    returncode = getattr(completed, "returncode", None)
    if type(returncode) is not int or returncode != 0:
        raise RuntimeError(f"manual NavTrain scorer failed: exit={returncode!r}")
    official_csv_path = _one_official_csv(output_dir)
    _import_manual_horizon_outputs(config, official_csv_path=official_csv_path)
    logger.info("Imported manual NavTrain Oracle H%s stores at %s", horizon, horizon_dir)
    return ManualNavTrainHorizonArtifacts(
        horizon=horizon,
        scorer_config_path=scorer_config_path,
        effective_config_path=effective_path,
        official_csv_path=official_csv_path,
        score_store_path=paths.score_store(horizon),
        feature_store_path=paths.feature_store(horizon),
    )


def _manual_lambda_grid(source: ManualOracleSource) -> tuple[float, ...]:
    if not isinstance(source, ManualOracleSource):
        raise ValueError("source must be a ManualOracleSource")
    return source.lambda_grid


def aggregate_manual_navtrain_oracle(
    results_root: Path,
    *,
    source_config_path: Path,
) -> StoreReceipt:
    """Validate all ten H0--H4 stores and atomically publish one embedded Oracle."""

    source = load_manual_oracle_source(
        source_config_path,
        results_root=results_root,
    )
    paths = ManualOraclePaths.from_results_root(results_root)
    score_paths = {horizon: paths.score_store(horizon) for horizon in range(5)}
    feature_paths = {horizon: paths.feature_store(horizon) for horizon in range(5)}
    for horizon in range(5):
        try:
            _strict_existing_regular_file(
                score_paths[horizon],
                field=f"manual Oracle H{horizon} score store",
            )
        except ValueError as exc:
            raise ValueError(f"manual Oracle H{horizon} score store is missing or invalid") from exc
        try:
            _strict_existing_regular_file(
                feature_paths[horizon],
                field=f"manual Oracle H{horizon} feature store",
            )
        except ValueError as exc:
            raise ValueError(f"manual Oracle H{horizon} feature store is missing or invalid") from exc

    baseline_identity: Optional[tuple[str, str]] = None
    for horizon in range(5):
        expected_policy_id, _ = _manual_policy_identity(horizon, lineage=source.lineage)
        with (
            open_score_store(score_paths[horizon]) as score_store,
            open_feature_store(feature_paths[horizon]) as feature_store,
        ):
            expected_common = (
                NAVTRAIN_GATE_PROTOCOL_ID,
                expected_policy_id,
                source.lineage,
                horizon,
            )
            score_common = (
                score_store.metadata.protocol_id,
                score_store.metadata.policy_id,
                score_store.metadata.lineage,
                score_store.metadata.horizon,
            )
            feature_common = (
                feature_store.metadata.protocol_id,
                feature_store.metadata.policy_id,
                feature_store.metadata.lineage,
                feature_store.metadata.horizon,
            )
            if score_common != expected_common or feature_common != expected_common:
                raise ValueError(f"manual Oracle H{horizon} store policy identity differs")
            if (
                feature_store.metadata.feature_schema != _FEATURE_SCHEMA
                or feature_store.metadata.feature_sources != _FEATURE_SOURCES
                or feature_store.metadata.common_random_seed != _COMMON_RANDOM_SEED
                or score_store.metadata.score_semantics != _SCORE_SEMANTICS
            ):
                raise ValueError(f"manual Oracle H{horizon} store semantic contract differs")
            identity = (
                score_store.metadata.scenario_manifest_sha256,
                score_store.metadata.metric_cache_inventory_sha256,
            )
            if identity != (
                feature_store.metadata.scenario_manifest_sha256,
                feature_store.metadata.metric_cache_inventory_sha256,
            ) or (baseline_identity is not None and identity != baseline_identity):
                raise ValueError("manual Oracle forced-horizon authority identities differ")
            if baseline_identity is None:
                baseline_identity = identity
    if baseline_identity is None:
        raise RuntimeError("manual Oracle aggregation found no forced-horizon stores")
    receipt = create_embedded_oracle_store_v2(
        source.oracle_path,
        OracleStoreMetadata(
            protocol_id=NAVTRAIN_GATE_PROTOCOL_ID,
            lineage=source.lineage,
            scenario_manifest_sha256=baseline_identity[0],
            metric_cache_inventory_sha256=baseline_identity[1],
            lambda_grid=_manual_lambda_grid(source),
        ),
        score_store_paths=score_paths,
        feature_store_paths=feature_paths,
    )
    with open_embedded_oracle_store_v2(source.oracle_path, expected_sha256=receipt.sha256) as oracle:
        if oracle.policy_ids_by_horizon != tuple(
            _manual_policy_identity(horizon, lineage=source.lineage)[0] for horizon in range(5)
        ):
            raise RuntimeError("embedded manual Oracle policy vector differs after publication")
    logger.info("Published self-contained manual NavTrain Oracle at %s", source.oracle_path)
    return receipt


__all__ = [
    "MANUAL_NAVTRAIN_POLICY_TRACE_SCHEMA",
    "MANUAL_NAVTRAIN_SCORER_SCHEMA",
    "ManualNavTrainAuthorityBundle",
    "ManualNavTrainHorizonArtifacts",
    "ManualNavTrainScorerConfig",
    "ManualOracleEnvironment",
    "ManualOraclePaths",
    "ManualOracleSource",
    "aggregate_manual_navtrain_oracle",
    "build_manual_navtrain_manifest",
    "load_manual_oracle_environment",
    "load_manual_oracle_source",
    "read_manual_navtrain_authority_bundle",
    "read_manual_navtrain_scorer_config",
    "score_manual_navtrain_horizon",
]
