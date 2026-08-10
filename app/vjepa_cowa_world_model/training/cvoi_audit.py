"""Validated lineage for the NavSim CVoI geometry preflight."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (
    FORMAL_V2_NAVSIM_EFFECTIVE_ROOT_SCHEMA,
    resolve_formal_v2_navsim_scene_filter_path,
    validate_formal_v2_navsim_effective_root,
)
from app.vjepa_cowa_world_model.training.navsim_scene_filter_contract import load_navsim_scene_filter_contract

TIMESTAMP_POLICY_ROOT_CONTIGUOUS = "root_contiguous_v1"
TIMESTAMP_POLICY_ELIGIBLE_WINDOW_BOUNDARY = "eligible_window_boundary_v1"
TIMESTAMP_POLICIES = frozenset(
    {
        TIMESTAMP_POLICY_ROOT_CONTIGUOUS,
        TIMESTAMP_POLICY_ELIGIBLE_WINDOW_BOUNDARY,
    }
)
CVOI_AUDIT_VERIFICATION_LIVE = "live"
CVOI_AUDIT_VERIFICATION_RECEIPT_ONLY = "receipt_only"
CVOI_AUDIT_VERIFICATION_MODES = frozenset(
    {
        CVOI_AUDIT_VERIFICATION_LIVE,
        CVOI_AUDIT_VERIFICATION_RECEIPT_ONLY,
    }
)
CVOI_AUDIT_CONTRACT_GEOMETRY = "navsim_cvoi_geometry"
CVOI_AUDIT_CONTRACT_STATIC_INPUTS = "navsim_cvoi_static_inputs"
_STATIC_INPUT_INCLUDED_SCOPE = [
    "cf_runtime_annotation_selection",
    "cf_route_free_quality",
    "converter_sha256",
    "path_existence",
    "root_identity",
    "scene_filter_sha256",
    "split_manifest_sha256",
]
_STATIC_INPUT_EXCLUDED_SCOPE = [
    "agent_geometry",
    "l2_collision",
    "pkl_deserialization",
    "sensor_blob_hashing",
]


def _resolve_cvoi_configured_scene_filter_path(value: object, *, name: str) -> Path:
    try:
        return resolve_formal_v2_navsim_scene_filter_path(value)
    except ValueError as exc:
        raise ValueError(f"CVoI audit configured {name} must use an exact repository-relative path") from exc


def _select_cvoi_static_scene_filter_live_path(
    receipt_path: Path,
    *,
    configured_path: object | None,
    path_mode: str,
    name: str,
) -> Path:
    if configured_path is None:
        return receipt_path
    resolved_configured_path = _resolve_cvoi_configured_scene_filter_path(
        configured_path,
        name=f"{name}.scene_filter_yaml",
    )
    if path_mode == "exact":
        if receipt_path != resolved_configured_path:
            raise ValueError(f"CVoI static input root {name!r} scene_filter does not match current config")
        return receipt_path
    if path_mode == "portable_content":
        return resolved_configured_path
    raise ValueError(f"unsupported CVoI static scene-filter path_mode: {path_mode!r}")


def _sha256_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class CvoiAuditSignature:
    """Fingerprints proving the dataset passed the required preflight."""

    dataset_fingerprint: str
    converter_fingerprint: str
    split_manifest_fingerprint: str

    def __post_init__(self) -> None:
        _sha256_string(self.dataset_fingerprint, name="dataset_fingerprint")
        _sha256_string(self.converter_fingerprint, name="converter_fingerprint")
        _sha256_string(self.split_manifest_fingerprint, name="split_manifest_fingerprint")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def to_provenance(self) -> dict[str, str]:
        return {
            "dataset": self.dataset_fingerprint,
            "converter": self.converter_fingerprint,
            "split_manifest": self.split_manifest_fingerprint,
        }


def _timestamp_policy(value: object, *, name: str) -> str:
    if not isinstance(value, str) or value not in TIMESTAMP_POLICIES:
        raise ValueError(f"{name} must be one of {sorted(TIMESTAMP_POLICIES)}, got {value!r}")
    return value


def validate_cvoi_audit_signature(value: object) -> CvoiAuditSignature:
    if isinstance(value, CvoiAuditSignature):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("CVoI audit signature must be a mapping")
    expected = {"dataset_fingerprint", "converter_fingerprint", "split_manifest_fingerprint"}
    if set(value) != expected:
        raise ValueError(
            f"CVoI audit signature fields mismatch: missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )
    return CvoiAuditSignature(**dict(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CVoI audited directory no longer exists: {root}")
    digest = hashlib.sha256()
    for candidate in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def _sha256_relative_files(root: Path, relative_paths: Sequence[str]) -> str:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"CVoI audited directory no longer exists: {resolved_root}")
    if not relative_paths or list(relative_paths) != sorted(set(relative_paths)):
        raise ValueError("CVoI audited relative_paths must be non-empty, sorted, and unique")
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        candidate = (resolved_root / relative_path).resolve()
        try:
            canonical_relative = candidate.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"CVoI audited file escapes source root: {relative_path!r}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"CVoI audited file no longer exists: {candidate}")
        relative = canonical_relative.encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def _sha256_scene_subtrees(root: Path, scene_names: Sequence[str]) -> str:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"CVoI audited directory no longer exists: {resolved_root}")
    digest = hashlib.sha256()
    for scene_name in scene_names:
        scene_path = resolved_root / scene_name
        if not scene_path.is_dir():
            raise FileNotFoundError(f"CVoI audited sensor scene no longer exists: {scene_path}")
        encoded_name = scene_name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, byteorder="big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(_sha256_tree(scene_path)))
    return digest.hexdigest()


def _validate_file_provenance(provenance: object, *, name: str) -> tuple[Path, str]:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"CVoI audit manifest requires {name} file provenance")
    path_value = provenance.get("path")
    fingerprint = provenance.get("fingerprint")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"CVoI audit manifest {name} path must be non-empty")
    return Path(path_value).expanduser().resolve(), _sha256_string(fingerprint, name=f"{name}_fingerprint")


def _verify_live_file(provenance: object, *, name: str) -> None:
    resolved, fingerprint = _validate_file_provenance(provenance, name=name)
    if not resolved.is_file():
        raise FileNotFoundError(f"CVoI audited {name} file no longer exists: {resolved}")
    if _sha256_file(resolved) != fingerprint:
        raise ValueError(f"CVoI audited {name} file content has changed")


def _validate_tree_provenance(provenance: object, *, name: str) -> tuple[Path, str, str]:
    if not isinstance(provenance, Mapping):
        raise ValueError(f"CVoI audit manifest requires {name} directory provenance")
    path_value = provenance.get("path")
    fingerprint = provenance.get("fingerprint")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"CVoI audit manifest {name} path must be non-empty")
    fingerprint_scope = provenance.get("fingerprint_scope", "full_tree")
    if fingerprint_scope == "audited_scene_subtrees":
        scene_names = provenance.get("scene_names")
        if (
            not isinstance(scene_names, list)
            or not scene_names
            or any(not isinstance(value, str) or not value for value in scene_names)
            or scene_names != sorted(set(scene_names))
        ):
            raise ValueError(f"CVoI audit manifest {name} requires sorted unique scene_names")
    elif fingerprint_scope == "audited_relative_files":
        relative_paths = provenance.get("relative_paths")
        if (
            not isinstance(relative_paths, list)
            or not relative_paths
            or any(not isinstance(value, str) or not value for value in relative_paths)
            or relative_paths != sorted(set(relative_paths))
        ):
            raise ValueError(f"CVoI audit manifest {name} requires sorted unique relative_paths")
    elif fingerprint_scope != "full_tree":
        raise ValueError(f"CVoI audit manifest {name} has unknown fingerprint_scope={fingerprint_scope!r}")
    return (
        Path(path_value).expanduser().resolve(),
        _sha256_string(fingerprint, name=f"{name}_fingerprint"),
        str(fingerprint_scope),
    )


def _verify_live_tree(provenance: object, *, name: str) -> None:
    resolved, fingerprint, fingerprint_scope = _validate_tree_provenance(provenance, name=name)
    if fingerprint_scope == "full_tree":
        live_fingerprint = _sha256_tree(resolved)
    elif fingerprint_scope == "audited_scene_subtrees":
        live_fingerprint = _sha256_scene_subtrees(resolved, provenance["scene_names"])
    elif fingerprint_scope == "audited_relative_files":
        live_fingerprint = _sha256_relative_files(resolved, provenance["relative_paths"])
    else:  # pragma: no cover - _validate_tree_provenance owns the closed set.
        raise RuntimeError(f"unreachable CVoI fingerprint scope {fingerprint_scope!r}")
    if live_fingerprint != fingerprint:
        raise ValueError(f"CVoI audited {name} directory content has changed")


def _provenance_at_path(provenance: object, path: str) -> dict[str, object]:
    if not isinstance(provenance, Mapping):
        raise ValueError("CVoI audit path rebinding requires mapping provenance")
    rebound = dict(provenance)
    rebound["path"] = path
    return rebound


def _validate_timestamp_boundary(
    value: object,
    *,
    root_name: str,
    index: int,
    root_report: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"CVoI audit root {root_name!r} timestamp boundary {index} must be a mapping")
    expected_fields = {
        "scene",
        "previous_frame_index",
        "current_frame_index",
        "delta_us",
        "threshold_us",
    }
    if set(value) != expected_fields:
        raise ValueError(
            f"CVoI audit root {root_name!r} timestamp boundary {index} fields mismatch: "
            f"missing={sorted(expected_fields - set(value))}, unexpected={sorted(set(value) - expected_fields)}"
        )
    if not isinstance(value["scene"], str) or not value["scene"]:
        raise ValueError(f"CVoI audit root {root_name!r} timestamp boundary {index} requires a scene")
    previous_index = value["previous_frame_index"]
    current_index = value["current_frame_index"]
    if (
        isinstance(previous_index, bool)
        or not isinstance(previous_index, int)
        or isinstance(current_index, bool)
        or not isinstance(current_index, int)
        or previous_index < 0
        or current_index != previous_index + 1
    ):
        raise ValueError(
            f"CVoI audit root {root_name!r} timestamp boundary {index} requires consecutive frame indices"
        )
    delta_us = value["delta_us"]
    threshold_us = value["threshold_us"]
    if isinstance(delta_us, bool) or not isinstance(delta_us, int) or delta_us <= 0:
        raise ValueError(f"CVoI audit root {root_name!r} timestamp boundary {index} requires positive delta_us")
    if (
        isinstance(threshold_us, bool)
        or not isinstance(threshold_us, (int, float))
        or not math.isfinite(float(threshold_us))
        or threshold_us <= 0
    ):
        raise ValueError(f"CVoI audit root {root_name!r} timestamp boundary {index} requires positive threshold_us")
    if delta_us <= threshold_us:
        raise ValueError(f"CVoI audit root {root_name!r} timestamp boundary {index} delta_us must exceed threshold_us")

    scenes = root_report.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError(f"CVoI audit root {root_name!r} timestamp boundary {index} requires root scenes")
    scene_records = [scene for scene in scenes if isinstance(scene, Mapping) and scene.get("scene") == value["scene"]]
    if len(scene_records) != 1:
        raise ValueError(
            f"CVoI audit root {root_name!r} timestamp boundary {index} scene {value['scene']!r} "
            "must identify exactly one root scene"
        )
    frame_count = scene_records[0].get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(
            f"CVoI audit root {root_name!r} timestamp boundary {index} scene requires positive frame_count"
        )
    if current_index >= frame_count:
        raise ValueError(
            f"CVoI audit root {root_name!r} timestamp boundary {index} current_frame_index "
            f"must be below scene frame_count={frame_count}"
        )

    max_time_gap_us = root_report.get("max_time_gap_us")
    if (
        isinstance(max_time_gap_us, bool)
        or not isinstance(max_time_gap_us, (int, float))
        or not math.isfinite(float(max_time_gap_us))
        or max_time_gap_us <= 0
    ):
        raise ValueError(f"CVoI audit root {root_name!r} requires positive finite max_time_gap_us")
    if threshold_us != max_time_gap_us:
        raise ValueError(
            f"CVoI audit root {root_name!r} timestamp boundary {index} threshold_us must match "
            f"root max_time_gap_us={max_time_gap_us}"
        )


def _validated_manifest_timestamp_policies(manifest: Mapping[str, object], *, manifest_version: int) -> dict[str, str]:
    root_reports = manifest.get("roots")
    if manifest_version == 1:
        if root_reports is None:
            return {}
        if not isinstance(root_reports, list):
            raise ValueError("CVoI audit manifest roots must be a list")
        policies: dict[str, str] = {}
        for report in root_reports:
            if not isinstance(report, Mapping) or not isinstance(report.get("name"), str) or not report["name"]:
                raise ValueError("CVoI audit manifest contains an invalid root report")
            name = str(report["name"])
            if name in policies:
                raise ValueError(f"CVoI audit manifest has duplicate root name {name!r}")
            explicit_policy = report.get("timestamp_policy", TIMESTAMP_POLICY_ROOT_CONTIGUOUS)
            policy = _timestamp_policy(explicit_policy, name=f"CVoI audit root {name!r}.timestamp_policy")
            if policy != TIMESTAMP_POLICY_ROOT_CONTIGUOUS:
                raise ValueError("CVoI audit manifest v1 only supports root_contiguous_v1 timestamp semantics")
            policies[name] = policy
        return policies

    if not isinstance(root_reports, list) or not root_reports:
        raise ValueError("CVoI audit manifest v2 requires non-empty root reports")
    raw_policies = manifest.get("timestamp_policies")
    if not isinstance(raw_policies, Mapping) or not raw_policies:
        raise ValueError("CVoI audit manifest v2 requires a non-empty timestamp_policies mapping")
    reports_by_name: dict[str, Mapping[str, object]] = {}
    for report in root_reports:
        if not isinstance(report, Mapping) or not isinstance(report.get("name"), str) or not report["name"]:
            raise ValueError("CVoI audit manifest contains an invalid root report")
        name = str(report["name"])
        if name in reports_by_name:
            raise ValueError(f"CVoI audit manifest has duplicate root name {name!r}")
        reports_by_name[name] = report
    if set(raw_policies) != set(reports_by_name):
        raise ValueError(
            "CVoI audit manifest timestamp_policies roots mismatch: "
            f"policies={sorted(raw_policies)}, reports={sorted(reports_by_name)}"
        )
    policies: dict[str, str] = {}
    for name in sorted(reports_by_name):
        report = reports_by_name[name]
        policy = _timestamp_policy(raw_policies[name], name=f"CVoI audit timestamp_policies[{name!r}]")
        domain = report.get("domain")
        if domain not in {"real", "counterfactual"}:
            raise ValueError(f"CVoI audit root {name!r} domain must be real or counterfactual")
        if domain == "counterfactual" and policy != TIMESTAMP_POLICY_ROOT_CONTIGUOUS:
            raise ValueError(f"CVoI audit counterfactual root {name!r} requires root_contiguous_v1 timestamp_policy")
        if report.get("timestamp_policy") != policy:
            raise ValueError(f"CVoI audit manifest timestamp_policies disagrees with root report {name!r}")
        boundaries = report.get("timestamp_boundaries")
        boundary_count = report.get("timestamp_boundary_count")
        if not isinstance(boundaries, list):
            raise ValueError(f"CVoI audit root {name!r} requires timestamp_boundaries list")
        if (
            isinstance(boundary_count, bool)
            or not isinstance(boundary_count, int)
            or boundary_count != len(boundaries)
        ):
            raise ValueError(f"CVoI audit root {name!r} timestamp_boundary_count does not match its boundary list")
        if policy == TIMESTAMP_POLICY_ROOT_CONTIGUOUS and boundaries:
            raise ValueError(f"CVoI audit root {name!r} root_contiguous_v1 report cannot contain boundaries")
        for index, boundary in enumerate(boundaries):
            _validate_timestamp_boundary(
                boundary,
                root_name=name,
                index=index,
                root_report=report,
            )
        policies[name] = policy
    return policies


def _verify_configured_roots(
    manifest: Mapping[str, object],
    expected_roots: Sequence[Mapping[str, object]],
    *,
    path_mode: str,
    verification_mode: str,
    manifest_version: int,
    manifest_timestamp_policies: Mapping[str, str],
) -> None:
    verify_live = verification_mode == CVOI_AUDIT_VERIFICATION_LIVE
    if not isinstance(expected_roots, Sequence) or isinstance(expected_roots, (str, bytes)) or not expected_roots:
        raise ValueError("expected CVoI NavSim roots must be a non-empty sequence")
    root_reports = manifest.get("roots")
    if not isinstance(root_reports, list) or not root_reports:
        raise ValueError("CVoI audit manifest is missing root reports")
    reports_by_name = {}
    for report in root_reports:
        if not isinstance(report, Mapping) or not isinstance(report.get("name"), str):
            raise ValueError("CVoI audit manifest contains an invalid root report")
        if report["name"] in reports_by_name:
            raise ValueError(f"CVoI audit manifest has duplicate root name {report['name']!r}")
        reports_by_name[report["name"]] = report
    expected_by_name = {}
    for root in expected_roots:
        if not isinstance(root, Mapping) or not isinstance(root.get("name"), str) or not root["name"]:
            raise ValueError("configured CVoI root requires a non-empty name")
        if root["name"] in expected_by_name:
            raise ValueError(f"configured CVoI roots have duplicate name {root['name']!r}")
        expected_by_name[root["name"]] = root
    missing_reports = set(expected_by_name) - set(reports_by_name)
    if missing_reports:
        raise ValueError(
            "CVoI audit is missing roots required by the current config: "
            f"missing={sorted(missing_reports)}, audited={sorted(reports_by_name)}"
        )
    for name, root in expected_by_name.items():
        is_formal = root.get("effective_runtime_root_schema") == FORMAL_V2_NAVSIM_EFFECTIVE_ROOT_SCHEMA
        if is_formal:
            root = validate_formal_v2_navsim_effective_root(root)
        report = reports_by_name[name]
        if manifest_version == 2 and "timestamp_policy" not in root:
            raise ValueError(f"configured CVoI root {name!r} requires timestamp_policy for audit manifest v2")
        expected_timestamp_policy = _timestamp_policy(
            root.get("timestamp_policy", TIMESTAMP_POLICY_ROOT_CONTIGUOUS),
            name=f"configured CVoI root {name!r}.timestamp_policy",
        )
        expected_identity = {
            "domain": root.get("domain"),
            "data_path": str(Path(str(root.get("data_path", ""))).expanduser().resolve()),
            "max_agents": int(root.get("max_agents", 256)),
            "max_scenes": root.get("max_scenes"),
        }
        for temporal_field in ("base_fps", "max_frame_gap"):
            if temporal_field in root:
                expected_identity[temporal_field] = int(root[temporal_field])
        if is_formal:
            expected_identity.update(
                {
                    "effective_runtime_root_schema": root["effective_runtime_root_schema"],
                    "effective_runtime_root_sha256": root["effective_runtime_root_sha256"],
                }
            )
        identity_fields = set(expected_identity)
        if path_mode == "portable_content":
            identity_fields.remove("data_path")
        mismatched = {
            key: (report.get(key), expected_identity[key])
            for key in identity_fields
            if report.get(key) != expected_identity[key]
        }
        audited_timestamp_policy = manifest_timestamp_policies.get(name, TIMESTAMP_POLICY_ROOT_CONTIGUOUS)
        if audited_timestamp_policy != expected_timestamp_policy:
            mismatched["timestamp_policy"] = (audited_timestamp_policy, expected_timestamp_policy)
        if mismatched:
            raise ValueError(f"CVoI audit root {name!r} does not match current config: {mismatched}")
        scenes = report.get("scenes")
        if not isinstance(scenes, list) or not scenes:
            raise ValueError(f"CVoI audit root {name!r} is missing per-scene PKL fingerprints")
        data_path = Path(expected_identity["data_path"])
        audited_relative_paths: list[str] = []
        for scene in scenes:
            if not isinstance(scene, Mapping):
                raise ValueError(f"CVoI audit root {name!r} has an invalid scene record")
            relative_path = scene.get("relative_path")
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(f"CVoI audit root {name!r} scene is missing relative_path")
            _sha256_string(scene.get("file_sha256"), name=f"{name}.{relative_path}.file_sha256")
            audited_relative_paths.append(relative_path)
        if audited_relative_paths != sorted(set(audited_relative_paths)):
            raise ValueError(f"CVoI audit root {name!r} scene relative paths must be sorted and unique")

        scene_filter_contract = None
        scene_filter_path = root.get("scene_filter_yaml")
        configured_scene_filter_path = None
        if scene_filter_path is not None:
            configured_scene_filter_path = _resolve_cvoi_configured_scene_filter_path(
                scene_filter_path,
                name=f"{name}.scene_filter_yaml",
            )
        if verify_live:
            live_scene_paths = sorted(
                path for path in data_path.glob("*.pkl") if not path.name.startswith(".navsim_scene_index_cache_")
            )
        if verify_live and configured_scene_filter_path is not None:
            scene_filter_contract = load_navsim_scene_filter_contract(configured_scene_filter_path)
            paths_by_stem = {path.stem: path for path in live_scene_paths}
            missing_logs = sorted(set(scene_filter_contract.log_names) - set(paths_by_stem))
            if missing_logs:
                raise ValueError(f"CVoI audit root {name!r} scene_filter logs are missing PKLs: {missing_logs[:8]}")
            live_scene_paths = sorted(paths_by_stem[log_name] for log_name in scene_filter_contract.log_names)
        if verify_live:
            max_scenes = expected_identity["max_scenes"]
            if max_scenes is not None:
                live_scene_paths = live_scene_paths[: int(max_scenes)]
            live_relative_paths = [path.relative_to(data_path).as_posix() for path in live_scene_paths]
            if audited_relative_paths != live_relative_paths:
                raise ValueError(
                    f"CVoI audit root {name!r} selected scene set has changed: "
                    f"audited={audited_relative_paths}, live={live_relative_paths}"
                )
            for scene in scenes:
                relative_path = str(scene["relative_path"])
                _verify_live_file(
                    {"path": str(data_path / relative_path), "fingerprint": scene["file_sha256"]},
                    name=f"{name}.{relative_path}",
                )
        sidecars = report.get("sidecars")
        if not isinstance(sidecars, Mapping):
            raise ValueError(f"CVoI audit root {name!r} is missing sidecar provenance")
        sensor_path = root.get("sensor_blobs_path")
        if isinstance(sensor_path, str) and sensor_path:
            stored_sensor = sidecars.get("sensor_blobs")
            expected_sensor = str(Path(sensor_path).expanduser().resolve())
            if not isinstance(stored_sensor, Mapping):
                raise ValueError(f"CVoI audit root {name!r} is missing sensor_blobs provenance")
            if path_mode == "exact" and stored_sensor.get("path") != expected_sensor:
                raise ValueError(f"CVoI audit root {name!r} sensor_blobs_path does not match current config")
            sensor_provenance = (
                stored_sensor if path_mode == "exact" else _provenance_at_path(stored_sensor, expected_sensor)
            )
            _validate_tree_provenance(sensor_provenance, name=f"{name}.sensor_blobs")
            if verify_live:
                _verify_live_tree(sensor_provenance, name=f"{name}.sensor_blobs")
        configured_camera_names = root.get("camera_names")
        if configured_camera_names is not None and report.get("camera_names") != list(configured_camera_names):
            raise ValueError(f"CVoI audit root {name!r} camera_names/order does not match current config")
        if scene_filter_path is not None:
            stored_scene_filter = sidecars.get("scene_filter")
            if not isinstance(stored_scene_filter, Mapping):
                raise ValueError(f"CVoI audit root {name!r} is missing scene_filter provenance")
            expected_scene_filter_path = str(configured_scene_filter_path)
            if path_mode == "exact" and stored_scene_filter.get("path") != expected_scene_filter_path:
                raise ValueError(f"CVoI audit root {name!r} scene_filter path does not match current config")
            for digest_field in ("file_sha256", "log_name_set_sha256", "token_set_sha256"):
                _sha256_string(
                    stored_scene_filter.get(digest_field),
                    name=f"{name}.scene_filter.{digest_field}",
                )
            if verify_live:
                expected_scene_filter = scene_filter_contract.to_receipt()
                fields = set(expected_scene_filter)
                if path_mode == "portable_content":
                    fields.remove("path")
                drift = {
                    field: (stored_scene_filter.get(field), expected_scene_filter[field])
                    for field in fields
                    if stored_scene_filter.get(field) != expected_scene_filter[field]
                }
                if drift:
                    raise ValueError(f"CVoI audit root {name!r} scene_filter content has changed: {drift}")
        if root.get("domain") != "counterfactual":
            continue
        path_fields = {
            "pose_overlay_path": "pose_overlay_path",
            "annotations_source_path": "annotations_source",
            "annotations_path": "annotations",
            "trajectory_quality_path": "trajectory_quality",
        }
        for config_field, report_field in path_fields.items():
            if config_field not in root:
                continue
            expected_path = str(Path(str(root[config_field])).expanduser().resolve())
            stored = sidecars.get(report_field)
            stored_path = (
                stored if isinstance(stored, str) else stored.get("path") if isinstance(stored, Mapping) else None
            )
            if path_mode == "exact" and stored_path != expected_path:
                raise ValueError(
                    f"CVoI audit root {name!r} {config_field} does not match current config: "
                    f"audited={stored_path!r}, configured={expected_path!r}"
                )
            live_provenance = stored if path_mode == "exact" else _provenance_at_path(stored, expected_path)
            if report_field in {"annotations_source", "annotations", "trajectory_quality"}:
                _validate_file_provenance(live_provenance, name=f"{name}.{report_field}")
                if verify_live:
                    _verify_live_file(live_provenance, name=f"{name}.{report_field}")
            elif report_field == "pose_overlay_path":
                _validate_tree_provenance(live_provenance, name=f"{name}.pose_overlay")
                if verify_live:
                    _verify_live_tree(live_provenance, name=f"{name}.pose_overlay")


def _static_input_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CVoI static input {name} path must be non-empty")
    resolved = Path(value).expanduser().resolve()
    if value != str(resolved):
        raise ValueError(f"CVoI static input {name} path must be absolute and canonical")
    return resolved


def _validate_static_file_provenance(provenance: object, *, name: str) -> tuple[Path, str]:
    if not isinstance(provenance, Mapping) or set(provenance) != {"path", "fingerprint"}:
        raise ValueError(f"CVoI static input requires {name} file provenance with path and fingerprint")
    path = _static_input_path(provenance["path"], name=name)
    fingerprint = _sha256_string(provenance["fingerprint"], name=f"{name}_fingerprint")
    return path, fingerprint


def _validate_static_directory_binding(binding: object, *, name: str) -> Path:
    expected = {"path", "kind", "content_hashed"}
    if not isinstance(binding, Mapping) or set(binding) != expected:
        raise ValueError(f"CVoI static input requires exact {name} directory binding")
    if binding.get("kind") != "directory" or binding.get("content_hashed") is not False:
        raise ValueError(f"CVoI static input {name} must be an unhashed directory binding")
    return _static_input_path(binding["path"], name=name)


def _validate_static_scene_filter_receipt(receipt: object, *, name: str) -> tuple[Path, str]:
    expected = {
        "path",
        "file_sha256",
        "log_name_count",
        "log_name_set_sha256",
        "token_count",
        "token_set_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected:
        raise ValueError(f"CVoI static input requires exact {name} scene_filter provenance")
    path = _static_input_path(receipt["path"], name=f"{name}.scene_filter")
    file_sha256 = _sha256_string(receipt["file_sha256"], name=f"{name}.scene_filter.file_sha256")
    for digest_field in ("log_name_set_sha256", "token_set_sha256"):
        _sha256_string(receipt[digest_field], name=f"{name}.scene_filter.{digest_field}")
    for count_field in ("log_name_count", "token_count"):
        count = receipt[count_field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"CVoI static input {name}.scene_filter.{count_field} must be non-negative")
    return path, file_sha256


def _static_live_path(path: Path, *, expected_path: object | None, path_mode: str) -> Path:
    if path_mode == "portable_content" and expected_path is not None:
        return _static_input_path(expected_path, name="portable configured input")
    return path


def _require_static_live_file(
    path: Path,
    *,
    fingerprint: str,
    name: str,
    verify_live: bool,
    digest_cache: dict[Path, str],
) -> None:
    if not verify_live:
        return
    if not path.is_file():
        raise FileNotFoundError(f"CVoI static input {name} file does not exist: {path}")
    resolved = path.resolve()
    actual = digest_cache.get(resolved)
    if actual is None:
        actual = _sha256_file(resolved)
        digest_cache[resolved] = actual
    if actual != fingerprint:
        raise ValueError(f"CVoI static input {name} file content has changed")


def _require_static_live_directory(path: Path, *, name: str, verify_live: bool) -> None:
    if verify_live and not path.is_dir():
        raise FileNotFoundError(f"CVoI static input {name} directory does not exist: {path}")


def _static_expected_roots(
    expected_roots: Sequence[Mapping[str, object]] | None,
) -> dict[str, Mapping[str, object]]:
    if expected_roots is None:
        return {}
    if not isinstance(expected_roots, Sequence) or isinstance(expected_roots, (str, bytes)) or not expected_roots:
        raise ValueError("expected CVoI NavSim roots must be a non-empty sequence")
    roots_by_name: dict[str, Mapping[str, object]] = {}
    for root in expected_roots:
        if not isinstance(root, Mapping) or not isinstance(root.get("name"), str) or not root["name"]:
            raise ValueError("configured CVoI root requires a non-empty name")
        name = str(root["name"])
        if name in roots_by_name:
            raise ValueError(f"configured CVoI roots have duplicate name {name!r}")
        roots_by_name[name] = root
    return roots_by_name


def _validate_static_input_manifest(
    manifest: Mapping[str, object],
    *,
    expected_roots: Sequence[Mapping[str, object]] | None,
    path_mode: str,
    verification_mode: str,
) -> tuple[str, str]:
    """Validate the honest, non-geometry Formal-v2 static-input receipt."""

    if manifest.get("geometry_audit_performed") is not False:
        raise ValueError("CVoI static input manifest requires geometry_audit_performed=false")
    if manifest.get("sensor_content_hashed") is not False:
        raise ValueError("CVoI static input manifest requires sensor_content_hashed=false")
    if manifest.get("fingerprint_algorithm") != "sha256":
        raise ValueError("CVoI static input manifest requires fingerprint_algorithm=sha256")
    expected_scope = {
        "included": _STATIC_INPUT_INCLUDED_SCOPE,
        "excluded": _STATIC_INPUT_EXCLUDED_SCOPE,
    }
    if manifest.get("verification_scope") != expected_scope:
        raise ValueError("CVoI static input manifest verification_scope is incompatible")

    converter_path, converter_provenance_fingerprint = _validate_static_file_provenance(
        manifest.get("converter"),
        name="converter",
    )
    converter_fingerprint = _sha256_string(
        manifest.get("converter_fingerprint"),
        name="converter_fingerprint",
    )
    if converter_fingerprint != converter_provenance_fingerprint:
        raise ValueError("CVoI static input converter_fingerprint disagrees with converter provenance")
    split_manifest = manifest.get("split_manifest")
    if not isinstance(split_manifest, Mapping) or split_manifest.get("passed") is not True:
        raise ValueError("CVoI static input manifest requires a passing split_manifest")
    split_provenance = {key: split_manifest.get(key) for key in ("path", "fingerprint")}
    if set(split_manifest) != {"path", "fingerprint", "passed"}:
        raise ValueError("CVoI static input requires exact split_manifest provenance")
    split_path, split_fingerprint = _validate_static_file_provenance(
        split_provenance,
        name="split_manifest",
    )
    verify_live = verification_mode == CVOI_AUDIT_VERIFICATION_LIVE
    live_file_digests: dict[Path, str] = {}
    _require_static_live_file(
        converter_path,
        fingerprint=converter_provenance_fingerprint,
        name="converter",
        verify_live=verify_live,
        digest_cache=live_file_digests,
    )
    _require_static_live_file(
        split_path,
        fingerprint=split_fingerprint,
        name="split_manifest",
        verify_live=verify_live,
        digest_cache=live_file_digests,
    )

    root_reports = manifest.get("roots")
    if not isinstance(root_reports, list) or len(root_reports) != 4:
        raise ValueError("CVoI static input manifest requires exactly four root reports")
    reports_by_name: dict[str, Mapping[str, object]] = {}
    for report in root_reports:
        if not isinstance(report, Mapping) or not isinstance(report.get("name"), str) or not report["name"]:
            raise ValueError("CVoI static input manifest contains an invalid root report")
        name = str(report["name"])
        if name in reports_by_name:
            raise ValueError(f"CVoI static input manifest has duplicate root name {name!r}")
        reports_by_name[name] = report

    scene_list = manifest.get("scene_list")
    timestamp_policies = manifest.get("timestamp_policies")
    if not isinstance(scene_list, Mapping) or set(scene_list) != set(reports_by_name):
        raise ValueError("CVoI static input manifest scene_list must exactly cover all four roots")
    if not isinstance(timestamp_policies, Mapping) or set(timestamp_policies) != set(reports_by_name):
        raise ValueError("CVoI static input manifest timestamp_policies must exactly cover all four roots")
    max_agents = manifest.get("max_agents")
    if isinstance(max_agents, bool) or not isinstance(max_agents, int) or max_agents <= 0:
        raise ValueError("CVoI static input manifest max_agents must be a positive integer")

    configured_by_name = _static_expected_roots(expected_roots)
    missing = set(configured_by_name) - set(reports_by_name)
    if missing:
        raise ValueError(
            "CVoI static input audit is missing roots required by the current config: "
            f"missing={sorted(missing)}, audited={sorted(reports_by_name)}"
        )
    required_report_fields = {
        "name",
        "domain",
        "data_path",
        "sensor_blobs_path",
        "max_agents",
        "max_scenes",
        "timestamp_policy",
        "scene_names",
        "sidecars",
    }
    optional_report_fields = {
        "base_fps",
        "max_frame_gap",
        "effective_runtime_root_schema",
        "effective_runtime_root_sha256",
        "camera_names",
    }
    for name, report in reports_by_name.items():
        fields = set(report)
        if not required_report_fields <= fields or fields - required_report_fields - optional_report_fields:
            raise ValueError(f"CVoI static input root {name!r} identity fields are incompatible")
        domain = report.get("domain")
        if domain not in {"real", "counterfactual"}:
            raise ValueError(f"CVoI static input root {name!r} domain must be real or counterfactual")
        report_max_agents = report.get("max_agents")
        if report_max_agents != max_agents or type(report_max_agents) is not int:
            raise ValueError(f"CVoI static input root {name!r} max_agents disagrees with manifest")
        max_scenes = report.get("max_scenes")
        if max_scenes is not None and (
            isinstance(max_scenes, bool) or not isinstance(max_scenes, int) or max_scenes <= 0
        ):
            raise ValueError(f"CVoI static input root {name!r} max_scenes must be null or positive")
        timestamp_policy = _timestamp_policy(
            report.get("timestamp_policy"),
            name=f"CVoI static input root {name!r}.timestamp_policy",
        )
        if domain == "counterfactual" and timestamp_policy != TIMESTAMP_POLICY_ROOT_CONTIGUOUS:
            raise ValueError(f"CVoI static input counterfactual root {name!r} requires root_contiguous_v1")
        if timestamp_policies[name] != timestamp_policy:
            raise ValueError(f"CVoI static input timestamp_policies disagrees with root {name!r}")
        scenes = report.get("scene_names")
        if (
            not isinstance(scenes, list)
            or not scenes
            or any(not isinstance(scene, str) or not scene for scene in scenes)
            or scenes != sorted(set(scenes))
        ):
            raise ValueError(f"CVoI static input root {name!r} scene_names must be non-empty, sorted, and unique")
        if scene_list[name] != scenes:
            raise ValueError(f"CVoI static input scene_list disagrees with root {name!r}")
        for integer_field in ("base_fps", "max_frame_gap"):
            value = report.get(integer_field)
            if integer_field in report and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"CVoI static input root {name!r} {integer_field} must be positive")
        if "effective_runtime_root_schema" in report:
            if report["effective_runtime_root_schema"] != FORMAL_V2_NAVSIM_EFFECTIVE_ROOT_SCHEMA:
                raise ValueError(f"CVoI static input root {name!r} effective runtime schema is incompatible")
            _sha256_string(
                report.get("effective_runtime_root_sha256"),
                name=f"{name}.effective_runtime_root_sha256",
            )
        elif "effective_runtime_root_sha256" in report:
            raise ValueError(f"CVoI static input root {name!r} effective runtime identity is incomplete")
        if "camera_names" in report:
            camera_names = report["camera_names"]
            if (
                not isinstance(camera_names, list)
                or not camera_names
                or any(not isinstance(camera, str) or not camera for camera in camera_names)
                or len(camera_names) != len(set(camera_names))
            ):
                raise ValueError(f"CVoI static input root {name!r} camera_names must be non-empty and unique")

        data_path = _static_input_path(report.get("data_path"), name=f"{name}.data_path")
        sensor_path = _static_input_path(
            report.get("sensor_blobs_path"),
            name=f"{name}.sensor_blobs_path",
        )
        sidecars = report.get("sidecars")
        expected_sidecars = (
            {"data_path", "sensor_blobs", "scene_filter"}
            if domain == "real"
            else {
                "data_path",
                "sensor_blobs",
                "pose_overlay_path",
                "annotations_source",
                "annotations",
                "trajectory_quality",
            }
        )
        if not isinstance(sidecars, Mapping) or set(sidecars) != expected_sidecars:
            raise ValueError(f"CVoI static input root {name!r} sidecar provenance is incomplete")
        bound_data_path = _validate_static_directory_binding(sidecars["data_path"], name=f"{name}.data_path")
        bound_sensor_path = _validate_static_directory_binding(
            sidecars["sensor_blobs"],
            name=f"{name}.sensor_blobs",
        )
        if bound_data_path != data_path or bound_sensor_path != sensor_path:
            raise ValueError(f"CVoI static input root {name!r} directory binding disagrees with root identity")

        configured = configured_by_name.get(name)
        if configured is not None:
            configured_value: Mapping[str, object] = configured
            if configured.get("effective_runtime_root_schema") == FORMAL_V2_NAVSIM_EFFECTIVE_ROOT_SCHEMA:
                configured_value = validate_formal_v2_navsim_effective_root(configured)
            required_config_fields = {"domain", "data_path", "sensor_blobs_path", "max_agents", "timestamp_policy"}
            missing_config_fields = required_config_fields - set(configured_value)
            if missing_config_fields:
                raise ValueError(
                    f"configured CVoI root {name!r} is missing static identity fields: {sorted(missing_config_fields)}"
                )
            identity_fields = {"domain", "max_agents", "max_scenes", "timestamp_policy"}
            identity_fields.update(field for field in optional_report_fields if field in configured_value)
            mismatched = {
                field: (report.get(field), configured_value.get(field))
                for field in identity_fields
                if report.get(field) != configured_value.get(field)
            }
            if path_mode == "exact":
                for field in ("data_path", "sensor_blobs_path"):
                    configured_path = str(_static_input_path(configured_value[field], name=f"{name}.{field}"))
                    if report.get(field) != configured_path:
                        mismatched[field] = (report.get(field), configured_path)
            if mismatched:
                raise ValueError(f"CVoI static input root {name!r} does not match current config: {mismatched}")

        live_data_path = _static_live_path(
            data_path,
            expected_path=None if configured is None else configured.get("data_path"),
            path_mode=path_mode,
        )
        live_sensor_path = _static_live_path(
            sensor_path,
            expected_path=None if configured is None else configured.get("sensor_blobs_path"),
            path_mode=path_mode,
        )
        _require_static_live_directory(live_data_path, name=f"{name}.data_path", verify_live=verify_live)
        _require_static_live_directory(live_sensor_path, name=f"{name}.sensor_blobs", verify_live=verify_live)

        if domain == "real":
            scene_filter_path, scene_filter_fingerprint = _validate_static_scene_filter_receipt(
                sidecars["scene_filter"],
                name=name,
            )
            configured_scene_filter = None if configured is None else configured.get("scene_filter_yaml")
            scene_filter_path = _select_cvoi_static_scene_filter_live_path(
                scene_filter_path,
                configured_path=configured_scene_filter,
                path_mode=path_mode,
                name=name,
            )
            _require_static_live_file(
                scene_filter_path,
                fingerprint=scene_filter_fingerprint,
                name=f"{name}.scene_filter",
                verify_live=verify_live,
                digest_cache=live_file_digests,
            )
            continue

        pose_path = _validate_static_directory_binding(sidecars["pose_overlay_path"], name=f"{name}.pose_overlay")
        configured_pose = None if configured is None else configured.get("pose_overlay_path")
        if path_mode == "exact" and configured_pose is not None:
            expected_pose = _static_input_path(configured_pose, name=f"{name}.pose_overlay_path")
            if pose_path != expected_pose:
                raise ValueError(f"CVoI static input root {name!r} pose overlay does not match current config")
        live_pose_path = _static_live_path(pose_path, expected_path=configured_pose, path_mode=path_mode)
        _require_static_live_directory(live_pose_path, name=f"{name}.pose_overlay", verify_live=verify_live)
        static_file_provenance = {
            report_field: _validate_static_file_provenance(
                sidecars[report_field],
                name=f"{name}.{report_field}",
            )
            for report_field in ("annotations_source", "annotations", "trajectory_quality")
        }
        if static_file_provenance["annotations"] != static_file_provenance["annotations_source"]:
            raise ValueError(
                f"CVoI static input root {name!r} must use the original annotation for dataset runtime selection"
            )
        for config_field, report_field in (
            ("annotations_source_path", "annotations_source"),
            ("annotations_path", "annotations"),
            ("trajectory_quality_path", "trajectory_quality"),
        ):
            sidecar_path, sidecar_fingerprint = static_file_provenance[report_field]
            configured_path = None if configured is None else configured.get(config_field)
            if path_mode == "exact" and configured_path is not None:
                expected_sidecar = _static_input_path(configured_path, name=f"{name}.{config_field}")
                if sidecar_path != expected_sidecar:
                    raise ValueError(f"CVoI static input root {name!r} {config_field} does not match current config")
            live_sidecar_path = _static_live_path(
                sidecar_path,
                expected_path=configured_path,
                path_mode=path_mode,
            )
            _require_static_live_file(
                live_sidecar_path,
                fingerprint=sidecar_fingerprint,
                name=f"{name}.{report_field}",
                verify_live=verify_live,
                digest_cache=live_file_digests,
            )
    return converter_fingerprint, split_fingerprint


def load_cvoi_audit_manifest(
    path: str | Path,
    *,
    expected_roots: Sequence[Mapping[str, object]] | None = None,
    path_mode: str = "exact",
    verification_mode: str = CVOI_AUDIT_VERIFICATION_LIVE,
) -> CvoiAuditSignature:
    """Load a complete preflight manifest with explicit live or receipt-only verification."""

    if path_mode not in {"exact", "portable_content"}:
        raise ValueError("CVoI audit path_mode must be 'exact' or 'portable_content'")
    if verification_mode not in CVOI_AUDIT_VERIFICATION_MODES:
        raise ValueError(
            "CVoI audit verification_mode must be one of "
            f"{sorted(CVOI_AUDIT_VERIFICATION_MODES)}, got {verification_mode!r}"
        )
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CVoI audit manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"CVoI audit manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError("CVoI audit manifest must contain a mapping")
    manifest_version = manifest.get("manifest_version")
    contract = manifest.get("contract")
    is_geometry_manifest = contract == CVOI_AUDIT_CONTRACT_GEOMETRY and manifest_version in {1, 2}
    is_static_input_manifest = contract == CVOI_AUDIT_CONTRACT_STATIC_INPUTS and manifest_version == 1
    if (
        isinstance(manifest_version, bool)
        or not isinstance(manifest_version, int)
        or not (is_geometry_manifest or is_static_input_manifest)
    ):
        raise ValueError("CVoI audit manifest version/contract is incompatible")
    if manifest.get("passed") is not True:
        raise ValueError("CVoI audit manifest must have passed=true before label generation")
    if manifest.get("ready_for_labels") is not True:
        raise ValueError("CVoI audit manifest must be ready_for_labels=true before label generation")
    canonical_payload = dict(manifest)
    stored_fingerprint = canonical_payload.pop("fingerprint", None)
    canonical_payload.pop("dataset_fingerprint", None)
    try:
        canonical = json.dumps(
            canonical_payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("CVoI audit manifest must contain canonical finite JSON values") from exc
    recomputed_fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if stored_fingerprint != recomputed_fingerprint:
        raise ValueError("CVoI audit manifest content does not match its fingerprint")
    dataset_fingerprint = _sha256_string(
        manifest.get("dataset_fingerprint"),
        name="dataset_fingerprint",
    )
    if stored_fingerprint != dataset_fingerprint:
        raise ValueError("CVoI audit manifest fingerprint and dataset_fingerprint must match")
    if is_static_input_manifest:
        converter_fingerprint, split_fingerprint = _validate_static_input_manifest(
            manifest,
            expected_roots=expected_roots,
            path_mode=path_mode,
            verification_mode=verification_mode,
        )
        signature = CvoiAuditSignature(
            dataset_fingerprint=dataset_fingerprint,
            converter_fingerprint=converter_fingerprint,
            split_manifest_fingerprint=split_fingerprint,
        )
        return signature
    manifest_timestamp_policies = _validated_manifest_timestamp_policies(
        manifest,
        manifest_version=manifest_version,
    )
    converter_fingerprint = _sha256_string(
        manifest.get("converter_fingerprint"),
        name="converter_fingerprint",
    )
    split_manifest = manifest.get("split_manifest")
    if not isinstance(split_manifest, Mapping) or split_manifest.get("passed") is not True:
        raise ValueError("CVoI audit manifest requires a passing split_manifest")
    split_fingerprint = _sha256_string(
        split_manifest.get("fingerprint"),
        name="split_manifest_fingerprint",
    )
    if expected_roots is not None:
        converter = manifest.get("converter")
        split_provenance: object = split_manifest
        if path_mode == "portable_content":
            if not isinstance(converter, Mapping) or not isinstance(converter.get("path"), str):
                raise ValueError("CVoI audit portable path mode requires converter path provenance")
            if not isinstance(split_manifest.get("path"), str):
                raise ValueError("CVoI audit portable path mode requires split manifest path provenance")
            converter = _provenance_at_path(converter, str(manifest_path.parent / Path(converter["path"]).name))
            split_provenance = _provenance_at_path(
                split_manifest,
                str(manifest_path.parent / Path(str(split_manifest["path"])).name),
            )
        _validate_file_provenance(converter, name="converter")
        _validate_file_provenance(split_provenance, name="split_manifest")
        if verification_mode == CVOI_AUDIT_VERIFICATION_LIVE:
            _verify_live_file(converter, name="converter")
            _verify_live_file(split_provenance, name="split_manifest")
        _verify_configured_roots(
            manifest,
            expected_roots,
            path_mode=path_mode,
            verification_mode=verification_mode,
            manifest_version=manifest_version,
            manifest_timestamp_policies=manifest_timestamp_policies,
        )
    signature = CvoiAuditSignature(
        dataset_fingerprint=dataset_fingerprint,
        converter_fingerprint=converter_fingerprint,
        split_manifest_fingerprint=split_fingerprint,
    )
    return signature
