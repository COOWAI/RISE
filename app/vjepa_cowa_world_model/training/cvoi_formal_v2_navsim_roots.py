"""Self-contained NavSim root contract and stage projection for CVoI Formal-v2."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_MAX_AGENTS

FORMAL_V2_NAVSIM_ROOT_CATALOG_SCHEMA = "cvoi_formal_v2_navsim_root_catalog_v1"
FORMAL_V2_NAVSIM_EFFECTIVE_ROOT_SCHEMA = "cvoi_formal_v2_navsim_effective_runtime_root_v1"
FORMAL_V2_NAVSIM_RUNTIME_OVERLAY_SCHEMA = "cvoi_formal_v2_navsim_runtime_overlay_v1"
FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION = "trajectory_match_and_accident_type_allowlist"
FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST = ("自车行为引起", "非自车行为引起")
FORMAL_V2_NAVSIM_CF_ANNOTATION_FILTER_SCHEMA = "cvoi_navsim_cf_accident_type_filter_v1"
FORMAL_V2_NAVSIM_DIRECT_PREFLIGHT_ROOT = Path("/path/to/rise/results/cvoi_manual_full/preflight")

_FORMAL_V2_NAVSIM_TRAIN_SCENE_FILTER = "configs/navsim/scene_filters/navtrain.yaml"
_FORMAL_V2_NAVSIM_TEST_SCENE_FILTER = "configs/navsim/scene_filters/navtest.yaml"
_FORMAL_V2_NAVSIM_SCENE_FILTERS = frozenset(
    {_FORMAL_V2_NAVSIM_TRAIN_SCENE_FILTER, _FORMAL_V2_NAVSIM_TEST_SCENE_FILTER}
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

_EFFECTIVE_ROOT_COMMON: dict[str, object] = {
    "effective_runtime_root_schema": FORMAL_V2_NAVSIM_EFFECTIVE_ROOT_SCHEMA,
    "camera_name": "CAM_F0",
    "camera_names": ["CAM_F0"],
    "fps": 2,
    "base_fps": 2,
    "num_target_frames": 12,
    "num_observed_frames": 4,
    "max_frame_gap": 1,
    "image_require_policy": "observed_only",
    "max_agents": FORMAL_V2_NAVSIM_MAX_AGENTS,
    "max_scenes": None,
    "window_stride": 4,
    "annotation_selection": "all_valid",
    "repeat": 1,
    "tail_seconds": None,
}
_EFFECTIVE_ROOT_BY_DOMAIN: dict[str, dict[str, object]] = {
    "real": {
        "load_agent_annotations": True,
        "timestamp_policy": "eligible_window_boundary_v1",
        "window_start_policy": "sliding",
    },
    "counterfactual": {
        "annotation_selection": FORMAL_V2_NAVSIM_CF_ANNOTATION_SELECTION,
        "annotations_accident_type_allowlist": list(FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST),
        "annotations_drop_distorted": True,
        "annotations_require_trajectory_match": True,
        "load_agent_annotations": False,
        "timestamp_policy": "root_contiguous_v1",
        "window_start_policy": "counterfactual_scene_start",
        "pose_overlay_coord_frame": "opencv_first_frame",
        "pose_overlay_txt_start_seconds": 0.0,
    },
}
_EFFECTIVE_ROOT_HASH_EXCLUDED_FIELDS = frozenset(
    {
        "annotations_path",
        "annotations_source_path",
        "effective_runtime_root_sha256",
        "trajectory_quality_path",
    }
)
_RUNTIME_OVERLAY_COMMON_FIELDS = tuple(sorted(set(_EFFECTIVE_ROOT_COMMON) - {"effective_runtime_root_schema"}))
_RUNTIME_PROJECTION_BINDING_FIELDS = frozenset(
    {
        "annotations_path",
        "annotations_source_path",
        "trajectory_quality_path",
    }
)
_RUNTIME_PROVENANCE_FIELDS = frozenset(
    {
        "source_root_sha256",
        "runtime_overlay_schema",
        "runtime_overlay_allowed_fields",
    }
)

_CF_ANNOTATION_CONTRACT_ROWS = (
    (
        "cf_train",
        {
            "source_sha256": "cb4bb621761049ed7b124a9d0e1239e2b77481e72f20a261495a847bb37ccded",
            "source_record_count": 4433,
            "source_scene_set_sha256": "87a9ab78ab54f21933027e4c3c7596144ed7f20da2834d954269075ff9d99596",
            "source_trajectory_match_by_scene_sha256": (
                "53d25e2ca6728c39543fc86e6e617f4a38f35a63d47a902e48364c7d9bcf6cfa"
            ),
            "trajectory_match_counts": {"true": 3829, "false": 604, "null": 0},
            "trajectory_match_record_count": 3829,
            "trajectory_match_scene_set_sha256": "c0ae18676e5f7d1c9a4279f937434b9d68434ee7182650d567bdd6dae6a34bc8",
            "eligible_record_count": 3829,
            "selected_record_count": 2429,
            "selected_scene_set_sha256": "779d47d793d06db53447590854b116fa3898425bb7b5bcc401f8579bcae70ee8",
            "selected_records_sha256": "609eebeedd10cb7deffeb3be22ae3ef5edd5494848342a1be8f8d0cad371ec03",
            "output_sha256": "1f62e9abd15acc9cb87d580dc5cb302a81fc14add985201d0d7077cf90a4cf9e",
            "selected_accident_type_counts": {"自车行为引起": 1978, "非自车行为引起": 451},
        },
    ),
    (
        "cf_val",
        {
            "source_sha256": "2e7a8629830d2bb23aeb7defcebb6e7deaca6dbc10168c5b1dd72804d8c91456",
            "source_record_count": 921,
            "source_scene_set_sha256": "dbf7cd6d6bd2418c8e16dfe951ebb60d3b1240f48a95b63ab94de370e5380e24",
            "source_trajectory_match_by_scene_sha256": (
                "6e59227c3d272069ed348275c24282e5a2e25eb738794b96b3de1561ef99e21b"
            ),
            "trajectory_match_counts": {"true": 807, "false": 114, "null": 0},
            "trajectory_match_record_count": 807,
            "trajectory_match_scene_set_sha256": "bc6822dc270b7c3e2ef62c65d715f36bc23fce8f91007c5be310740b8923820c",
            "eligible_record_count": 807,
            "selected_record_count": 426,
            "selected_scene_set_sha256": "521be53c254e1ae993cf1ac41a4c4b100658bdfbb1e6f3bd169f3881787fa3fd",
            "selected_records_sha256": "9fca4eb76b9760340961cb5327de951b5dce1b5f639b263c3af5c5954d3dd29e",
            "output_sha256": "aa93308d5b1a3a1845801606e670ead25dd20c8e5a3a11cbf974128e85165a53",
            "selected_accident_type_counts": {"自车行为引起": 343, "非自车行为引起": 83},
        },
    ),
)


def _expected_cf_annotation_contracts() -> dict[str, dict[str, object]]:
    return {role: copy.deepcopy(contract) for role, contract in _CF_ANNOTATION_CONTRACT_ROWS}


def _freeze_cf_annotation_contract(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_cf_annotation_contract(child) for key, child in value.items()})
    return value


FORMAL_V2_NAVSIM_CF_ANNOTATION_CONTRACTS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {role: _freeze_cf_annotation_contract(contract) for role, contract in _expected_cf_annotation_contracts().items()}
)

_ROLE_NAMES = {
    "real_train": "navsim_real_train",
    "cf_train": "navsim_cf_train",
    "real_navtest": "navsim_real_navtest",
    "cf_val": "navsim_cf_val",
}
_ROLE_DOMAINS = {
    "real_train": "real",
    "cf_train": "counterfactual",
    "real_navtest": "real",
    "cf_val": "counterfactual",
}
_CONFIGURABLE_EXTERNAL_PATH_SUFFIXES: dict[str, dict[str, tuple[str, ...]]] = {
    "real_train": {
        "data_path": ("navsim_logs", "trainval"),
        "sensor_blobs_path": ("sensor_blobs", "trainval"),
    },
    "cf_train": {
        "data_path": ("navsim_logs", "trainval"),
        "sensor_blobs_path": ("sensor_blobs", "trainval"),
        "pose_overlay_path": ("pose_overlay", "trainval", "pred_pose"),
        "annotations_path": ("navsim_train.json",),
    },
    "real_navtest": {
        "data_path": ("navsim_logs", "test"),
        "sensor_blobs_path": ("sensor_blobs", "test"),
    },
    "cf_val": {
        "data_path": ("navsim_logs", "test"),
        "sensor_blobs_path": ("sensor_blobs", "test"),
        "pose_overlay_path": ("pose_overlay", "test", "pred_pose"),
        "annotations_path": ("navsim_test.json",),
    },
}
_ROOTS: dict[str, dict[str, object]] = {
    "real_train": {
        "name": "navsim_real_train",
        "domain": "real",
        "data_path": "/path/to/navsim/dataset/navsim_logs/trainval",
        "sensor_blobs_path": "/path/to/navsim/dataset/sensor_blobs/trainval",
        "scene_filter_yaml": _FORMAL_V2_NAVSIM_TRAIN_SCENE_FILTER,
        "index_cache": True,
        "window_stride": 1,
        "repeat": 1,
    },
    "cf_train": {
        "name": "navsim_cf_train",
        "domain": "counterfactual",
        "data_path": "/path/to/counterfactual/navsim_logs/trainval",
        "sensor_blobs_path": "/path/to/counterfactual/sensor_blobs/trainval",
        "pose_overlay_path": "/path/to/counterfactual/pose_overlay/trainval/pred_pose",
        "pose_overlay_coord_frame": "opencv_first_frame",
        "pose_overlay_txt_start_seconds": 0.0,
        "pose_overlay_required": True,
        "index_cache": False,
        "window_stride": 1,
        "tail_seconds": None,
        "annotations_path": "/path/to/counterfactual/annotations/navsim_train.json",
        "annotations_drop_distorted": True,
        "annotations_require_trajectory_match": True,
        "annotation_selection": "all_valid",
        "repeat": 1,
    },
    "real_navtest": {
        "name": "navsim_real_navtest",
        "dataset_id": "navsim-real-test-v1",
        "domain": "real",
        "data_path": "/path/to/navsim/dataset/navsim_logs/test",
        "sensor_blobs_path": "/path/to/navsim/dataset/sensor_blobs/test",
        "scene_filter_yaml": _FORMAL_V2_NAVSIM_TEST_SCENE_FILTER,
        "annotation_selection": "all_valid",
        "tail_seconds": None,
        "pose_overlay_required": False,
    },
    "cf_val": {
        "name": "navsim_cf_val",
        "dataset_id": "navsim-cf-test-v1",
        "domain": "counterfactual",
        "data_path": "/path/to/counterfactual/navsim_logs/test",
        "sensor_blobs_path": "/path/to/counterfactual/sensor_blobs/test",
        "pose_overlay_path": "/path/to/counterfactual/pose_overlay/test/pred_pose",
        "pose_overlay_coord_frame": "opencv_first_frame",
        "pose_overlay_txt_start_seconds": 0.0,
        "pose_overlay_required": True,
        "annotations_path": "/path/to/counterfactual/annotations/navsim_test.json",
        "annotations_drop_distorted": True,
        "annotations_require_trajectory_match": True,
        "annotation_selection": "all_valid",
        "tail_seconds": None,
    },
}
_CF_ANNOTATION_ROLE_BY_ROOT_NAME = {
    "navsim_cf_train": "cf_train",
    "navsim_cf_val": "cf_val",
}
_CF_ANNOTATION_ROOT_NAME_BY_FILENAME = {
    Path(str(_ROOTS[role]["annotations_path"])).name: root_name
    for root_name, role in _CF_ANNOTATION_ROLE_BY_ROOT_NAME.items()
}
_P0_BRANCHES = frozenset({"uniform", "extremes", "short_heavy", "no_full"})
_FIELD_BRANCHES = frozenset(
    {"full", "strict_real_only", "hazard_only", "quality_only", "without_local_order", "factual_only"}
)
_MIXED_FIELD_BRANCHES = frozenset({"full", "hazard_only", "quality_only"})
_REUSED_FIELD_BRANCHES = frozenset({"without_local_order", "factual_only"})
_STOP_ORACLE_BRANCHES = frozenset({"p0", "full", "no_cf", "real_only"})
_GATE_BRANCHES = frozenset(
    {
        "p0_full",
        "real_only",
        "full",
        "no_cf",
        "without_field",
        "without_stop",
        "without_value_summary",
    }
)
_SCORER_BRANCHES = {
    "selection_scorer": frozenset({"v2_h4_candidate"}),
    "final_scorer": frozenset({"pdms_v1", "epdms_v2"}),
}


def resolve_formal_v2_navsim_scene_filter_path(path: str | Path) -> Path:
    """Resolve one exact repository-owned Formal-v2 scene filter without using cwd."""

    if not isinstance(path, (str, Path)) or str(path) not in _FORMAL_V2_NAVSIM_SCENE_FILTERS:
        raise ValueError("Formal-v2 NavSim scene filter must be one of the two exact repository-relative paths")
    return _REPOSITORY_ROOT / Path(path)


def formal_v2_navsim_cf_root_name_for_annotation_path(path: str | Path) -> str:
    """Resolve one locked counterfactual root from its canonical annotation filename."""

    if not isinstance(path, (str, Path)) or not str(path).strip():
        raise ValueError("Formal-v2 NavSim annotation path must be a non-empty absolute path")
    normalized = Path(os.path.normpath(str(Path(path).expanduser())))
    if not normalized.is_absolute():
        raise ValueError("Formal-v2 NavSim annotation path must be absolute")
    root_name = _CF_ANNOTATION_ROOT_NAME_BY_FILENAME.get(normalized.name)
    if root_name is None:
        raise ValueError(
            "Formal-v2 NavSim annotation filename must be one of "
            f"{sorted(_CF_ANNOTATION_ROOT_NAME_BY_FILENAME)}, got {normalized.name!r}"
        )
    return root_name


def validate_formal_v2_navsim_cf_annotation_source(
    path: str | Path,
    *,
    root_name: str,
) -> dict[str, object]:
    """Validate one counterfactual annotation file against its locked data contract."""

    role = _CF_ANNOTATION_ROLE_BY_ROOT_NAME.get(root_name)
    if role is None:
        raise ValueError(
            "Formal-v2 NavSim counterfactual root_name must be one of "
            f"{sorted(_CF_ANNOTATION_ROLE_BY_ROOT_NAME)}, got {root_name!r}"
        )
    normalized = Path(os.path.normpath(str(Path(path).expanduser())))
    if not normalized.is_absolute():
        raise ValueError("Formal-v2 NavSim annotation path must be absolute")
    if normalized.name != Path(str(_ROOTS[role]["annotations_path"])).name:
        raise ValueError(
            f"Formal-v2 NavSim {root_name} annotation filename must be "
            f"{Path(str(_ROOTS[role]['annotations_path'])).name!r}"
        )
    if normalized.is_symlink():
        raise ValueError(f"Formal-v2 NavSim annotation path must not be a symlink: {normalized}")
    if not normalized.is_file():
        raise FileNotFoundError(f"Formal-v2 NavSim annotation file does not exist: {normalized}")

    source_bytes = normalized.read_bytes()
    source_sha256 = _sha256_bytes(source_bytes)
    contract = FORMAL_V2_NAVSIM_CF_ANNOTATION_CONTRACTS[role]
    expected_sha256 = contract["source_sha256"]
    if source_sha256 != expected_sha256:
        raise ValueError(
            f"Formal-v2 NavSim {root_name} source_sha256 mismatch: "
            f"expected={expected_sha256}, actual={source_sha256}"
        )
    try:
        records = json.loads(source_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Formal-v2 NavSim {root_name} annotations must be valid UTF-8 JSON") from exc
    if not isinstance(records, list):
        raise ValueError(f"Formal-v2 NavSim {root_name} annotations must contain a JSON list")
    source_record_count = len(records)
    expected_record_count = contract["source_record_count"]
    if source_record_count != expected_record_count:
        raise ValueError(
            f"Formal-v2 NavSim {root_name} source_record_count mismatch: "
            f"expected={expected_record_count}, actual={source_record_count}"
        )
    return {
        "role": role,
        "root_name": root_name,
        "source_sha256": source_sha256,
        "source_record_count": source_record_count,
        "selected_record_count": contract["selected_record_count"],
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: object) -> str:
    try:
        serialized = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Formal-v2 NavSim reference data must be canonical-JSON serializable") from exc
    return _sha256_bytes(serialized)


def _effective_root_sha256(root: Mapping[str, object]) -> str:
    payload = {
        key: copy.deepcopy(value) for key, value in root.items() if key not in _EFFECTIVE_ROOT_HASH_EXCLUDED_FIELDS
    }
    return _canonical_sha256(payload)


def formal_v2_navsim_effective_root_sha256(root: Mapping[str, object]) -> str:
    """Return the authoritative digest for one signed effective runtime root."""

    if not isinstance(root, Mapping):
        raise TypeError("Formal-v2 NavSim effective root must be a mapping")
    return _effective_root_sha256(root)


def _runtime_overlay_allowed_fields(domain: str) -> list[str]:
    fields = {*_RUNTIME_OVERLAY_COMMON_FIELDS, *_EFFECTIVE_ROOT_BY_DOMAIN[domain]}
    if domain == "counterfactual":
        fields.add("annotation_filter_contract")
    return sorted(fields)


def _counterfactual_annotation_filter_contract(name: object) -> dict[str, object]:
    role = {"navsim_cf_train": "cf_train", "navsim_cf_val": "cf_val"}.get(name)
    if role is None:
        raise ValueError(f"Formal-v2 NavSim counterfactual root name is invalid: {name!r}")
    contract = FORMAL_V2_NAVSIM_CF_ANNOTATION_CONTRACTS[role]
    return {
        "schema": FORMAL_V2_NAVSIM_CF_ANNOTATION_FILTER_SCHEMA,
        "field": "annos.accident_type",
        "allowlist": list(FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST),
        "source_sha256": contract["source_sha256"],
        "source_record_count": contract["source_record_count"],
        "trajectory_match_record_count": contract["trajectory_match_record_count"],
        "selected_record_count": contract["selected_record_count"],
        "drop_unlisted": True,
        "fail_on_unknown_or_missing": True,
    }


def formal_v2_navsim_effective_navsim_settings() -> dict[str, object]:
    """Return the public global values mirrored explicitly by every Formal-v2 root."""

    return {
        "camera_name": _EFFECTIVE_ROOT_COMMON["camera_name"],
        "camera_names": copy.deepcopy(_EFFECTIVE_ROOT_COMMON["camera_names"]),
        "max_agents": _EFFECTIVE_ROOT_COMMON["max_agents"],
        "max_scenes": _EFFECTIVE_ROOT_COMMON["max_scenes"],
        "window_stride": _EFFECTIVE_ROOT_COMMON["window_stride"],
        "max_frame_gap": _EFFECTIVE_ROOT_COMMON["max_frame_gap"],
        "image_require_policy": _EFFECTIVE_ROOT_COMMON["image_require_policy"],
    }


def validate_formal_v2_navsim_effective_root(root: Mapping[str, object]) -> dict[str, Any]:
    """Fail fast unless a projected root contains the exact signed runtime semantics."""

    if not isinstance(root, Mapping):
        raise ValueError("Formal-v2 NavSim effective root must be a mapping")
    normalized = copy.deepcopy(dict(root))
    domain = normalized.get("domain")
    if domain not in _EFFECTIVE_ROOT_BY_DOMAIN:
        raise ValueError(f"Formal-v2 NavSim effective root domain is invalid: {domain!r}")
    expected = {**_EFFECTIVE_ROOT_COMMON, **_EFFECTIVE_ROOT_BY_DOMAIN[str(domain)]}
    source_root_sha256 = normalized.get("source_root_sha256")
    if (
        type(source_root_sha256) is not str
        or len(source_root_sha256) != 64
        or any(char not in "0123456789abcdef" for char in source_root_sha256)
    ):
        raise ValueError("Formal-v2 NavSim effective root source_root_sha256 must be a SHA256 digest")
    if normalized.get("runtime_overlay_schema") != FORMAL_V2_NAVSIM_RUNTIME_OVERLAY_SCHEMA:
        raise ValueError(
            "Formal-v2 NavSim effective root runtime_overlay_schema must be exactly "
            f"{FORMAL_V2_NAVSIM_RUNTIME_OVERLAY_SCHEMA!r}"
        )
    expected_overlay_fields = _runtime_overlay_allowed_fields(str(domain))
    if normalized.get("runtime_overlay_allowed_fields") != expected_overlay_fields:
        raise ValueError(
            "Formal-v2 NavSim effective root runtime_overlay_allowed_fields must be exactly "
            f"{expected_overlay_fields!r}"
        )
    for field, expected_value in expected.items():
        if field not in normalized:
            raise ValueError(f"Formal-v2 NavSim effective root requires explicit {field}")
        actual = normalized[field]
        if actual != expected_value or type(actual) is not type(expected_value):
            raise ValueError(
                f"Formal-v2 NavSim effective root {field} must be exactly {expected_value!r}, got {actual!r}"
            )
    if domain == "real":
        scene_filter = normalized.get("scene_filter_yaml")
        if type(scene_filter) is not str or not scene_filter.strip():
            raise ValueError("Formal-v2 NavSim real root requires explicit scene_filter_yaml")
        if "pose_overlay_txt_start_seconds" in normalized:
            raise ValueError("Formal-v2 NavSim real root forbids pose_overlay_txt_start_seconds")
        if "annotations_accident_type_allowlist" in normalized:
            raise ValueError("Formal-v2 NavSim real root forbids annotations_accident_type_allowlist")
        if "annotation_filter_contract" in normalized:
            raise ValueError("Formal-v2 NavSim real root forbids annotation_filter_contract")
    else:
        if "scene_filter_yaml" in normalized:
            raise ValueError("Formal-v2 NavSim counterfactual root forbids scene_filter_yaml")
        expected_filter_contract = _counterfactual_annotation_filter_contract(normalized.get("name"))
        if normalized.get("annotation_filter_contract") != expected_filter_contract:
            raise ValueError(
                "Formal-v2 NavSim counterfactual root annotation_filter_contract must exactly bind "
                "the locked two-class selected-record contract"
            )
    digest = normalized.get("effective_runtime_root_sha256")
    if type(digest) is not str or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("Formal-v2 NavSim effective root effective_runtime_root_sha256 must be a SHA256 digest")
    actual_digest = formal_v2_navsim_effective_root_sha256(normalized)
    if digest != actual_digest:
        raise ValueError(
            "Formal-v2 NavSim effective root effective_runtime_root_sha256 mismatch: "
            f"expected={digest}, actual={actual_digest}"
        )
    return normalized


def _apply_effective_runtime_root(
    root: Mapping[str, Any],
    *,
    source_root: Mapping[str, Any],
) -> dict[str, Any]:
    materialized = copy.deepcopy(dict(root))
    source = copy.deepcopy(dict(source_root))
    domain = materialized.get("domain")
    if domain not in _EFFECTIVE_ROOT_BY_DOMAIN:
        raise ValueError(f"Formal-v2 NavSim root domain is invalid: {domain!r}")
    if source.get("domain") != domain or source.get("name") != materialized.get("name"):
        raise ValueError("Formal-v2 NavSim runtime projection source root identity mismatch")
    reserved = sorted(_RUNTIME_PROVENANCE_FIELDS & (set(source) | set(materialized)))
    if reserved:
        raise ValueError(f"Formal-v2 NavSim source/materialized root contains reserved provenance fields: {reserved}")
    changed_projection_fields = {
        field
        for field in set(source) | set(materialized)
        if source.get(field) != materialized.get(field) or type(source.get(field)) is not type(materialized.get(field))
    }
    unexpected_projection_fields = changed_projection_fields - _RUNTIME_PROJECTION_BINDING_FIELDS
    if unexpected_projection_fields:
        raise ValueError(
            "Formal-v2 NavSim runtime projection changed source fields outside the binding whitelist: "
            f"{sorted(unexpected_projection_fields)}"
        )
    materialized["source_root_sha256"] = _canonical_sha256(source)
    materialized["runtime_overlay_schema"] = FORMAL_V2_NAVSIM_RUNTIME_OVERLAY_SCHEMA
    materialized["runtime_overlay_allowed_fields"] = _runtime_overlay_allowed_fields(str(domain))
    materialized.update(copy.deepcopy(_EFFECTIVE_ROOT_COMMON))
    materialized.update(copy.deepcopy(_EFFECTIVE_ROOT_BY_DOMAIN[str(domain)]))
    if domain == "counterfactual":
        materialized["annotation_filter_contract"] = _counterfactual_annotation_filter_contract(
            materialized.get("name")
        )
    materialized["effective_runtime_root_sha256"] = formal_v2_navsim_effective_root_sha256(materialized)
    return validate_formal_v2_navsim_effective_root(materialized)


def _extract_split_roots(navsim: Mapping[str, Any], *, field: str) -> dict[str, dict[str, Any]]:
    raw_roots = navsim.get(field)
    if not isinstance(raw_roots, list) or len(raw_roots) != 2:
        raise ValueError(f"reference data.navsim.{field} must contain exactly real and counterfactual roots")
    by_domain: dict[str, dict[str, Any]] = {}
    for index, raw_root in enumerate(raw_roots):
        if not isinstance(raw_root, Mapping):
            raise ValueError(f"reference data.navsim.{field}[{index}] must be a mapping")
        domain = raw_root.get("domain")
        if domain not in {"real", "counterfactual"}:
            raise ValueError(f"reference data.navsim.{field}[{index}].domain is invalid: {domain!r}")
        if domain in by_domain:
            raise ValueError(f"reference data.navsim.{field} contains duplicate domain {domain!r}")
        by_domain[domain] = copy.deepcopy(dict(raw_root))
    if set(by_domain) != {"real", "counterfactual"}:
        raise ValueError(f"reference data.navsim.{field} must cover real and counterfactual exactly")
    return by_domain


def _require_nonempty_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    return value


def _validate_counterfactual_root(root: Mapping[str, Any], *, role: str) -> None:
    if root.get("domain") != "counterfactual":
        raise ValueError(f"Formal-v2 NavSim {role}.domain must be 'counterfactual'")
    if root.get("annotations_require_trajectory_match") is not True:
        raise ValueError(f"Formal-v2 NavSim {role}.annotations_require_trajectory_match must be true")
    if root.get("annotations_drop_distorted") is not True:
        raise ValueError(f"Formal-v2 NavSim {role}.annotations_drop_distorted must be true")
    _require_nonempty_path(root.get("annotations_path"), field=f"Formal-v2 NavSim {role}.annotations_path")
    _require_nonempty_path(root.get("pose_overlay_path"), field=f"Formal-v2 NavSim {role}.pose_overlay_path")
    if root.get("pose_overlay_required") is not True:
        raise ValueError(f"Formal-v2 NavSim {role}.pose_overlay_required must be true")
    if root.get("pose_overlay_coord_frame") != "opencv_first_frame":
        raise ValueError(f"Formal-v2 NavSim {role}.pose_overlay_coord_frame must be 'opencv_first_frame'")
    start_seconds = root.get("pose_overlay_txt_start_seconds")
    if isinstance(start_seconds, bool) or not isinstance(start_seconds, (int, float)) or float(start_seconds) != 0.0:
        raise ValueError(f"Formal-v2 NavSim {role}.pose_overlay_txt_start_seconds must be exactly 0.0")


def _validate_real_root(root: Mapping[str, Any], *, role: str) -> None:
    if root.get("domain") != "real":
        raise ValueError(f"Formal-v2 NavSim {role}.domain must be 'real'")
    if "annotations_require_trajectory_match" in root:
        raise ValueError(f"Formal-v2 NavSim {role} must not define annotations_require_trajectory_match")
    if "annotations_accident_type_allowlist" in root:
        raise ValueError(f"Formal-v2 NavSim {role} must not define annotations_accident_type_allowlist")


def _validate_configurable_external_path(
    value: object,
    *,
    field: str,
    expected_suffix: tuple[str, ...],
) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be a non-empty absolute path")
    if os.path.normpath(value) != value:
        raise ValueError(f"{field} must use canonical lexical path spelling")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    if ".." in path.parts:
        raise ValueError(f"{field} must not contain '..' traversal")
    if tuple(path.parts[-len(expected_suffix) :]) != expected_suffix:
        expected = "/".join(expected_suffix)
        raise ValueError(f"{field} must end with {expected!r}, got {value!r}")


def _validate_root_fields(
    root: Mapping[str, Any],
    *,
    expected: Mapping[str, object],
    role: str,
) -> None:
    actual_fields = set(root)
    expected_fields = set(expected)
    if actual_fields != expected_fields:
        raise ValueError(
            f"Formal-v2 NavSim {role} fields mismatch: "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"unexpected={sorted(actual_fields - expected_fields)}"
        )
    path_suffixes = _CONFIGURABLE_EXTERNAL_PATH_SUFFIXES[role]
    for field, expected_value in expected.items():
        actual_value = root[field]
        expected_suffix = path_suffixes.get(field)
        if expected_suffix is not None:
            _validate_configurable_external_path(
                actual_value,
                field=f"Formal-v2 NavSim {role}.{field}",
                expected_suffix=expected_suffix,
            )
            continue
        if actual_value != expected_value or type(actual_value) is not type(expected_value):
            raise ValueError(
                f"Formal-v2 NavSim {role}.{field} must be exactly {expected_value!r}, got {actual_value!r}"
            )


def _validate_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(catalog, Mapping):
        raise ValueError("Formal-v2 NavSim root catalog must be a mapping")
    normalized = copy.deepcopy(dict(catalog))
    expected_fields = {"schema", "roots"}
    if set(normalized) != expected_fields:
        raise ValueError(f"Formal-v2 NavSim root catalog fields must be exactly {sorted(expected_fields)}")
    if normalized["schema"] != FORMAL_V2_NAVSIM_ROOT_CATALOG_SCHEMA:
        raise ValueError("Formal-v2 NavSim root catalog schema mismatch")
    roots = normalized["roots"]
    if not isinstance(roots, Mapping) or set(roots) != set(_ROLE_NAMES):
        raise ValueError(f"Formal-v2 NavSim catalog roots must be exactly {list(_ROLE_NAMES)}")
    for role, expected_name in _ROLE_NAMES.items():
        root = roots[role]
        if not isinstance(root, Mapping):
            raise ValueError(f"Formal-v2 NavSim catalog root {role} must be a mapping")
        _validate_root_fields(root, expected=_ROOTS[role], role=role)
        if root.get("name") != expected_name:
            raise ValueError(f"Formal-v2 NavSim {role}.name must be {expected_name!r}")
        if _ROLE_DOMAINS[role] == "counterfactual":
            _validate_counterfactual_root(root, role=role)
        else:
            _validate_real_root(root, role=role)
    return normalized


def build_formal_v2_navsim_root_catalog() -> dict[str, Any]:
    """Return a fresh copy of the four roots embedded in this training contract."""

    return _validate_catalog(
        {
            "schema": FORMAL_V2_NAVSIM_ROOT_CATALOG_SCHEMA,
            "roots": copy.deepcopy(_ROOTS),
        }
    )


def _normalize_preflight_root(preflight_root: str | Path) -> Path:
    if not isinstance(preflight_root, (str, Path)) or not str(preflight_root).strip():
        raise ValueError("preflight_root must be an absolute path")
    source = Path(preflight_root)
    if ".." in source.parts:
        raise ValueError("preflight_root must not contain '..' traversal")
    path = Path(os.path.normpath(str(source)))
    if not path.is_absolute():
        raise ValueError("preflight_root must be an absolute path")
    return path


def _materialize_counterfactual_root(
    root: Mapping[str, Any],
    *,
    role: str,
    preflight_root: Path,
    include_source_path: bool,
) -> dict[str, Any]:
    materialized = copy.deepcopy(dict(root))
    source_path = materialized["annotations_path"]
    materialized.pop("annotations_source_path", None)
    if include_source_path:
        materialized["annotations_source_path"] = source_path
    materialized["trajectory_quality_path"] = str(preflight_root / "trajectory_quality" / f"navsim_{role}.json")
    _validate_counterfactual_root(materialized, role=role)
    return materialized


def _apply_direct_runtime_root(root: Mapping[str, Any]) -> dict[str, Any]:
    materialized = copy.deepcopy(dict(root))
    domain = materialized.get("domain")
    if domain not in _EFFECTIVE_ROOT_BY_DOMAIN:
        raise ValueError(f"Formal-v2 NavSim direct root domain is invalid: {domain!r}")
    materialized.update(
        {
            key: copy.deepcopy(value)
            for key, value in _EFFECTIVE_ROOT_COMMON.items()
            if key != "effective_runtime_root_schema"
        }
    )
    materialized.update(copy.deepcopy(_EFFECTIVE_ROOT_BY_DOMAIN[str(domain)]))
    return materialized


def _direct_runtime_roots(catalog: Mapping[str, Any], preflight_root: Path) -> dict[str, dict[str, Any]]:
    roots = catalog["roots"]
    return {
        "real_train": _apply_direct_runtime_root(roots["real_train"]),
        "cf_train": _apply_direct_runtime_root(
            _materialize_counterfactual_root(
                roots["cf_train"],
                role="cf_train",
                preflight_root=preflight_root,
                include_source_path=False,
            )
        ),
        "real_navtest": _apply_direct_runtime_root(roots["real_navtest"]),
        "cf_val": _apply_direct_runtime_root(
            _materialize_counterfactual_root(
                roots["cf_val"],
                role="cf_val",
                preflight_root=preflight_root,
                include_source_path=False,
            )
        ),
    }


def _projection(
    roots: Mapping[str, dict[str, Any]],
    *,
    train_roles: tuple[str, ...],
    val_roles: tuple[str, ...],
    balance_train_roots: bool,
) -> dict[str, object]:
    return {
        "train_roots": [copy.deepcopy(roots[role]) for role in train_roles],
        "val_roots": [copy.deepcopy(roots[role]) for role in val_roles],
        "balance_train_roots": balance_train_roots,
    }


def _build_formal_v2_navsim_task_projection(
    stage: str,
    branch: str,
    roots: Mapping[str, dict[str, Any]],
) -> dict[str, object]:
    if stage == "p0" and branch in _P0_BRANCHES:
        return _projection(
            roots,
            train_roles=("real_train",),
            val_roles=("real_navtest",),
            balance_train_roots=False,
        )
    if stage == "field" and branch in _REUSED_FIELD_BRANCHES:
        raise ValueError(f"Formal-v2 Field branch {branch!r} must reuse the full Field checkpoint")
    if stage == "field" and branch in _MIXED_FIELD_BRANCHES:
        return _projection(
            roots,
            train_roles=("real_train", "cf_train"),
            val_roles=("real_navtest", "cf_val"),
            balance_train_roots=True,
        )
    if stage == "field" and branch == "strict_real_only":
        return _projection(
            roots,
            train_roles=("real_train",),
            val_roles=("real_navtest",),
            balance_train_roots=False,
        )
    if stage in {"calibration", "p1"} and branch in _FIELD_BRANCHES:
        return _projection(
            roots,
            train_roles=("real_train",),
            val_roles=("real_navtest",),
            balance_train_roots=False,
        )
    if stage in {"stop", "oracle"} and branch in _STOP_ORACLE_BRANCHES:
        return _projection(
            roots,
            train_roles=("real_train",),
            val_roles=("real_navtest",),
            balance_train_roots=False,
        )
    if stage == "gate" and branch in _GATE_BRANCHES:
        return _projection(roots, train_roles=(), val_roles=(), balance_train_roots=False)
    if stage in _SCORER_BRANCHES and branch in _SCORER_BRANCHES[stage]:
        return _projection(
            roots,
            train_roles=(),
            val_roles=("real_navtest",),
            balance_train_roots=False,
        )
    raise ValueError(f"unsupported Formal-v2 NavSim stage/branch: stage={stage!r}, branch={branch!r}")


def build_formal_v2_navsim_direct_task_projection(
    stage: str,
    branch: str,
    catalog: Mapping[str, Any],
    preflight_root: str | Path,
) -> dict[str, object]:
    """Build one proof-free task projection from the embedded root contract."""

    normalized_catalog = _validate_catalog(catalog)
    root_path = _normalize_preflight_root(preflight_root)
    roots = _direct_runtime_roots(normalized_catalog, root_path)
    return _build_formal_v2_navsim_task_projection(stage, branch, roots)


def _validate_formal_v2_navsim_direct_root(
    root: object,
    *,
    expected: Mapping[str, object],
    field_name: str,
) -> dict[str, Any]:
    if not isinstance(root, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    normalized = copy.deepcopy(dict(root))
    expected_root = copy.deepcopy(dict(expected))
    actual_fields = frozenset(normalized)
    expected_fields = frozenset(expected_root)
    if actual_fields != expected_fields:
        raise ValueError(
            f"{field_name} direct authority fields mismatch: "
            f"missing={sorted(expected_fields - actual_fields)}, "
            f"unexpected={sorted(actual_fields - expected_fields)}"
        )
    name = expected_root.get("name")
    role = next((candidate for candidate, expected_name in _ROLE_NAMES.items() if expected_name == name), None)
    if role is None:
        raise ValueError(f"{field_name}.name is not a Formal-v2 NavSim root role: {name!r}")
    configurable_paths = _CONFIGURABLE_EXTERNAL_PATH_SUFFIXES[role]
    for key, expected_value in expected_root.items():
        actual_value = normalized[key]
        expected_suffix = configurable_paths.get(key)
        if expected_suffix is not None:
            _validate_configurable_external_path(
                actual_value,
                field=f"{field_name}.{key}",
                expected_suffix=expected_suffix,
            )
            continue
        if actual_value != expected_value or type(actual_value) is not type(expected_value):
            raise ValueError(
                f"{field_name}.{key} differs from the direct NavSim authority projection: "
                f"expected={expected_value!r}, got={actual_value!r}"
            )
    return normalized


def validate_formal_v2_navsim_direct_task_projection(
    stage: str,
    branch: str,
    configured_projection: Mapping[str, object],
    catalog: Mapping[str, Any],
    preflight_root: str | Path = FORMAL_V2_NAVSIM_DIRECT_PREFLIGHT_ROOT,
) -> dict[str, object]:
    """Validate configured roots against one independently built direct projection."""

    if not isinstance(configured_projection, Mapping):
        raise ValueError("Formal-v2 NavSim configured direct projection must be a mapping")
    expected = build_formal_v2_navsim_direct_task_projection(stage, branch, catalog, preflight_root)
    normalized: dict[str, object] = {}
    for collection in ("train_roots", "val_roots"):
        roots = configured_projection.get(collection)
        expected_roots = expected[collection]
        if not isinstance(roots, list):
            raise ValueError(f"Formal-v2 NavSim configured {collection} must be a list")
        if not isinstance(expected_roots, list):
            raise ValueError(f"Formal-v2 NavSim authority {collection} must be a list")
        if len(roots) != len(expected_roots):
            raise ValueError(
                f"Formal-v2 NavSim configured {collection} length mismatch: "
                f"expected={len(expected_roots)}, got={len(roots)}"
            )
        normalized[collection] = [
            _validate_formal_v2_navsim_direct_root(
                root,
                expected=expected_root,
                field_name=f"data.navsim.{collection}[{index}]",
            )
            for index, (root, expected_root) in enumerate(zip(roots, expected_roots))
        ]
    expected_balance = expected["balance_train_roots"]
    actual_balance = configured_projection.get("balance_train_roots")
    if actual_balance is not expected_balance:
        raise ValueError(
            "data.navsim.balance_train_roots differs from the direct NavSim authority projection: "
            f"expected={expected_balance!r}, got={actual_balance!r}"
        )
    normalized["balance_train_roots"] = actual_balance
    return normalized
