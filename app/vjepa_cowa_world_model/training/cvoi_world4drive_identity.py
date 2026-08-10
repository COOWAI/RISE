"""Focused identities for retained NuScenes H3 World4Drive evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.vjepa_cowa_world_model.training.cf_trajectory_quality import counterfactual_quality_schema
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (
    CvoiWorld4DriveRuntimeBinding,
    build_world4drive_evaluation_runtime_signature_payload,
)
from app.vjepa_cowa_world_model.training.geometry_outcome import (
    CanonicalWorld4DrivePlanningOutcomeEvaluator,
    PlanningOutcomeEvaluator,
    planning_outcome_evaluator_signature_sha256,
)

WORLD4DRIVE_CODE_SIGNATURE_SOURCES = (
    "app/vjepa_cowa_world_model/training/cvoi_horizon_cache.py",
    "app/vjepa_cowa_world_model/training/cvoi_policy_replay.py",
    "app/vjepa_cowa_world_model/training/cvoi_world4drive_identity.py",
    "app/vjepa_cowa_world_model/training/cvoi_world4drive_input_assembly.py",
    "app/vjepa_cowa_world_model/training/cvoi_world4drive_latency_collection.py",
    "app/vjepa_cowa_world_model/training/cvoi_world4drive_metrics.py",
    "app/vjepa_cowa_world_model/training/cvoi_world4drive_report.py",
    "app/vjepa_cowa_world_model/training/cvoi_world4drive_runtime.py",
    "app/vjepa_cowa_world_model/training/geometry_outcome.py",
    "app/vjepa_cowa_world_model/training/lines/cvoi_world4drive_collection.py",
    "app/vjepa_cowa_world_model/training/lines/cvoi_world4drive_evaluation.py",
    "app/vjepa_cowa_world_model/training/lines/cvoi_world4drive_latency_collection.py",
    "app/vjepa_cowa_world_model/training/navsim_cvoi_model_runtime.py",
    "app/vjepa_cowa_world_model/training/navsim_cvoi_world4drive.py",
)

_GEOMETRY_AUDIT_CONTRACT = "navsim_cvoi_geometry"
_TIMESTAMP_POLICIES = frozenset({"root_contiguous_v1", "eligible_window_boundary_v1"})


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest, got {value!r}")
    return value


@dataclass(frozen=True)
class CvoiWorld4DriveAuditSignature:
    """Dataset, converter, and split identities for the Real evaluation cohort."""

    dataset_fingerprint: str
    converter_fingerprint: str
    split_manifest_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "dataset_fingerprint",
            "converter_fingerprint",
            "split_manifest_fingerprint",
        ):
            _require_sha256(name, getattr(self, name))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def to_provenance(self) -> dict[str, str]:
        return {
            "dataset": self.dataset_fingerprint,
            "converter": self.converter_fingerprint,
            "split_manifest": self.split_manifest_fingerprint,
        }


@dataclass(frozen=True)
class CvoiWorld4DriveCacheProvenance:
    """Signatures attached to every record in one matched lineage cache."""

    checkpoint_signature: str
    dataset_signature: str
    code_signature: str
    converter_signature: str
    evaluator_signature: str

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_signature",
            "dataset_signature",
            "code_signature",
            "converter_signature",
            "evaluator_signature",
        ):
            _require_sha256(name, getattr(self, name))


@dataclass(frozen=True)
class CvoiWorld4DriveEvaluationProvenance:
    """Retained immutable model/evaluator identities for World4Drive caches."""

    artifact_signature: str
    model_signature: str
    real_evaluator_signature: str
    counterfactual_evaluator_signature: str
    fixed_model_signature: str | None = None
    p0_model_signature: str | None = None
    p1_unguided_model_signature: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "artifact_signature",
            "model_signature",
            "real_evaluator_signature",
            "counterfactual_evaluator_signature",
        ):
            _require_sha256(name, getattr(self, name))
        for name in ("fixed_model_signature", "p0_model_signature", "p1_unguided_model_signature"):
            value = getattr(self, name)
            if value is None:
                value = self.model_signature
                object.__setattr__(self, name, value)
            _require_sha256(name, value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    try:
        canonical = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("World4Drive evaluation provenance must be JSON serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_tree(path: Path) -> str:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"World4Drive audited directory no longer exists: {root}")
    digest = hashlib.sha256()
    for candidate in sorted(root.rglob("*")):
        try:
            candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"World4Drive audited file escapes source root: {candidate}") from exc
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def _sha256_relative_files(root: Path, relative_paths: Sequence[str]) -> str:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"World4Drive audited directory no longer exists: {resolved_root}")
    if not relative_paths or list(relative_paths) != sorted(set(relative_paths)):
        raise ValueError("World4Drive audited relative_paths must be non-empty, sorted, and unique")
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        candidate = (resolved_root / relative_path).resolve()
        try:
            canonical_relative = candidate.relative_to(resolved_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"World4Drive audited file escapes source root: {relative_path!r}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"World4Drive audited file no longer exists: {candidate}")
        relative = canonical_relative.encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return digest.hexdigest()


def _sha256_scene_subtrees(root: Path, scene_names: Sequence[str]) -> str:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"World4Drive audited directory no longer exists: {resolved_root}")
    digest = hashlib.sha256()
    for scene_name in scene_names:
        relative_scene = Path(scene_name)
        if relative_scene.is_absolute() or ".." in relative_scene.parts:
            raise ValueError(f"World4Drive audited sensor scene escapes source root: {scene_name!r}")
        scene_path = (resolved_root / relative_scene).resolve()
        try:
            scene_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"World4Drive audited sensor scene escapes source root: {scene_name!r}") from exc
        if not scene_path.is_dir():
            raise FileNotFoundError(f"World4Drive audited sensor scene no longer exists: {scene_path}")
        encoded_name = scene_name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, byteorder="big"))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(_sha256_tree(scene_path)))
    return digest.hexdigest()


def _validate_timestamp_boundary(
    value: object,
    *,
    root_name: str,
    index: int,
    root_report: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"World4Drive audit root {root_name!r} timestamp boundary {index} must be a mapping")
    expected_fields = {
        "scene",
        "previous_frame_index",
        "current_frame_index",
        "delta_us",
        "threshold_us",
    }
    if set(value) != expected_fields:
        raise ValueError(
            f"World4Drive audit root {root_name!r} timestamp boundary {index} fields mismatch: "
            f"missing={sorted(expected_fields - set(value))}, unexpected={sorted(set(value) - expected_fields)}"
        )
    if not isinstance(value["scene"], str) or not value["scene"]:
        raise ValueError(f"World4Drive audit root {root_name!r} timestamp boundary {index} requires a scene")
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
            f"World4Drive audit root {root_name!r} timestamp boundary {index} requires consecutive frame indices"
        )
    delta_us = value["delta_us"]
    threshold_us = value["threshold_us"]
    if isinstance(delta_us, bool) or not isinstance(delta_us, int) or delta_us <= 0:
        raise ValueError(f"World4Drive audit root {root_name!r} timestamp boundary {index} requires positive delta_us")
    if (
        isinstance(threshold_us, bool)
        or not isinstance(threshold_us, (int, float))
        or not math.isfinite(float(threshold_us))
        or threshold_us <= 0
    ):
        raise ValueError(
            f"World4Drive audit root {root_name!r} timestamp boundary {index} requires positive threshold_us"
        )
    if delta_us <= threshold_us:
        raise ValueError(
            f"World4Drive audit root {root_name!r} timestamp boundary {index} delta_us must exceed threshold_us"
        )
    scenes = root_report.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError(f"World4Drive audit root {root_name!r} timestamp boundary {index} requires root scenes")
    scene_records = [scene for scene in scenes if isinstance(scene, Mapping) and scene.get("scene") == value["scene"]]
    if len(scene_records) != 1:
        raise ValueError(
            f"World4Drive audit root {root_name!r} timestamp boundary {index} scene {value['scene']!r} "
            "must identify exactly one root scene"
        )
    frame_count = scene_records[0].get("frame_count")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(
            f"World4Drive audit root {root_name!r} timestamp boundary {index} scene requires positive frame_count"
        )
    if current_index >= frame_count:
        raise ValueError(
            f"World4Drive audit root {root_name!r} timestamp boundary {index} current_frame_index "
            f"must be below scene frame_count={frame_count}"
        )
    max_time_gap_us = root_report.get("max_time_gap_us")
    if (
        isinstance(max_time_gap_us, bool)
        or not isinstance(max_time_gap_us, (int, float))
        or not math.isfinite(float(max_time_gap_us))
        or max_time_gap_us <= 0
    ):
        raise ValueError(f"World4Drive audit root {root_name!r} requires positive finite max_time_gap_us")
    if threshold_us != max_time_gap_us:
        raise ValueError(
            f"World4Drive audit root {root_name!r} timestamp boundary {index} threshold_us must match "
            f"root max_time_gap_us={max_time_gap_us}"
        )


def _validate_file_provenance(value: object, *, name: str) -> tuple[Path, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"World4Drive audit manifest requires {name} file provenance")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"World4Drive audit manifest {name} path must be non-empty")
    return Path(path_value).expanduser().resolve(), _require_sha256(
        f"{name}_fingerprint",
        value.get("fingerprint"),
    )


def _verify_live_file(value: object, *, name: str) -> None:
    path, fingerprint = _validate_file_provenance(value, name=name)
    if not path.is_file():
        raise FileNotFoundError(f"World4Drive audited {name} file no longer exists: {path}")
    if _sha256_file(path) != fingerprint:
        raise ValueError(f"World4Drive audited {name} file content has changed")


def _validate_tree_provenance(value: object, *, name: str) -> tuple[Path, str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"World4Drive audit manifest requires {name} directory provenance")
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"World4Drive audit manifest {name} path must be non-empty")
    scope = value.get("fingerprint_scope", "full_tree")
    if scope == "audited_scene_subtrees":
        scene_names = value.get("scene_names")
        if (
            not isinstance(scene_names, list)
            or not scene_names
            or any(not isinstance(scene, str) or not scene for scene in scene_names)
            or scene_names != sorted(set(scene_names))
        ):
            raise ValueError(f"World4Drive audit manifest {name} requires sorted unique scene_names")
    elif scope == "audited_relative_files":
        relative_paths = value.get("relative_paths")
        if (
            not isinstance(relative_paths, list)
            or not relative_paths
            or any(not isinstance(relative, str) or not relative for relative in relative_paths)
            or relative_paths != sorted(set(relative_paths))
        ):
            raise ValueError(f"World4Drive audit manifest {name} requires sorted unique relative_paths")
    elif scope != "full_tree":
        raise ValueError(f"World4Drive audit manifest {name} has unknown fingerprint_scope={scope!r}")
    return (
        Path(path_value).expanduser().resolve(),
        _require_sha256(f"{name}_fingerprint", value.get("fingerprint")),
        str(scope),
    )


def _verify_live_tree(value: Mapping[str, object], *, name: str) -> None:
    path, fingerprint, scope = _validate_tree_provenance(value, name=name)
    if scope == "full_tree":
        live = _sha256_tree(path)
    elif scope == "audited_scene_subtrees":
        live = _sha256_scene_subtrees(path, value["scene_names"])
    elif scope == "audited_relative_files":
        live = _sha256_relative_files(path, value["relative_paths"])
    else:  # pragma: no cover - _validate_tree_provenance owns the closed set.
        raise RuntimeError(f"unreachable World4Drive fingerprint scope {scope!r}")
    if live != fingerprint:
        raise ValueError(f"World4Drive audited {name} directory content has changed")


def _validate_timestamp_contract(payload: Mapping[str, object], report: Mapping[str, object], *, version: int) -> None:
    if version == 1:
        policy = report.get("timestamp_policy", "root_contiguous_v1")
        if policy != "root_contiguous_v1":
            raise ValueError("World4Drive audit manifest v1 only supports root_contiguous_v1")
        return
    policies = payload.get("timestamp_policies")
    root_name = report.get("name")
    if not isinstance(policies, Mapping) or set(policies) != {root_name}:
        raise ValueError("World4Drive audit manifest timestamp_policies must match its sole Real root")
    policy = policies[root_name]
    if policy not in _TIMESTAMP_POLICIES or report.get("timestamp_policy") != policy:
        raise ValueError("World4Drive audit manifest timestamp policy is incompatible")
    boundaries = report.get("timestamp_boundaries")
    boundary_count = report.get("timestamp_boundary_count")
    if not isinstance(boundaries, list) or type(boundary_count) is not int or boundary_count != len(boundaries):
        raise ValueError("World4Drive audit manifest timestamp boundary inventory is invalid")
    if policy == "root_contiguous_v1" and boundaries:
        raise ValueError("World4Drive root_contiguous_v1 audit cannot contain timestamp boundaries")
    for index, boundary in enumerate(boundaries):
        _validate_timestamp_boundary(
            boundary,
            root_name=str(root_name),
            index=index,
            root_report=report,
        )


def _validate_real_root(
    payload: Mapping[str, object],
    *,
    expected_real_root: Mapping[str, object],
    verification_mode: str,
    manifest_version: int,
) -> None:
    if not isinstance(expected_real_root, Mapping) or expected_real_root.get("domain") != "real":
        raise ValueError("expected_real_root must be a Real root mapping")
    root_name = expected_real_root.get("name")
    if not isinstance(root_name, str) or not root_name:
        raise ValueError("expected_real_root requires a non-empty name")
    reports = payload.get("roots")
    if not isinstance(reports, list) or len(reports) != 1 or not isinstance(reports[0], Mapping):
        raise ValueError("World4Drive dataset audit must contain exactly one Real root report")
    report = reports[0]
    if report.get("name") != root_name:
        raise ValueError(f"World4Drive audit is missing expected Real root {root_name!r}")
    if report.get("domain") != "real":
        raise ValueError(f"World4Drive audit root {root_name!r} must be Real")
    _validate_timestamp_contract(payload, report, version=manifest_version)

    expected_identity = {
        "domain": "real",
        "data_path": str(Path(str(expected_real_root.get("data_path", ""))).expanduser().resolve()),
        "max_agents": expected_real_root.get("max_agents"),
        "max_scenes": expected_real_root.get("max_scenes"),
    }
    for field in ("base_fps", "max_frame_gap"):
        if field in expected_real_root:
            expected_identity[field] = expected_real_root[field]
    mismatched = {
        name: (report.get(name), expected)
        for name, expected in expected_identity.items()
        if report.get(name) != expected
    }
    if mismatched:
        raise ValueError(f"World4Drive audit root {root_name!r} does not match current config: {mismatched}")

    scenes = report.get("scenes")
    if not isinstance(scenes, list) or not scenes or any(not isinstance(scene, Mapping) for scene in scenes):
        raise ValueError(f"World4Drive audit root {root_name!r} is missing per-scene fingerprints")
    relative_paths = []
    for scene in scenes:
        relative_path = scene.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"World4Drive audit root {root_name!r} scene is missing relative_path")
        _require_sha256(f"{root_name}.{relative_path}.file_sha256", scene.get("file_sha256"))
        relative_paths.append(relative_path)
    if relative_paths != sorted(set(relative_paths)):
        raise ValueError(f"World4Drive audit root {root_name!r} scene paths must be sorted and unique")

    data_path = Path(expected_identity["data_path"])
    if verification_mode == "live":
        live_paths = sorted(
            path for path in data_path.glob("*.pkl") if not path.name.startswith(".navsim_scene_index_cache_")
        )
        max_scenes = expected_identity["max_scenes"]
        if max_scenes is not None:
            live_paths = live_paths[: int(max_scenes)]
        live_relative_paths = [path.relative_to(data_path).as_posix() for path in live_paths]
        if live_relative_paths != relative_paths:
            raise ValueError(f"World4Drive audit root {root_name!r} selected scene set has changed")
        for scene in scenes:
            relative_path = str(scene["relative_path"])
            _verify_live_file(
                {"path": str(data_path / relative_path), "fingerprint": scene["file_sha256"]},
                name=f"{root_name}.{relative_path}",
            )

    sidecars = report.get("sidecars")
    if not isinstance(sidecars, Mapping):
        raise ValueError(f"World4Drive audit root {root_name!r} is missing sidecar provenance")
    sensor = sidecars.get("sensor_blobs")
    sensor_path = expected_real_root.get("sensor_blobs_path")
    if not isinstance(sensor_path, str) or not sensor_path:
        raise ValueError("expected_real_root requires sensor_blobs_path")
    stored_sensor_path, _, _ = _validate_tree_provenance(sensor, name=f"{root_name}.sensor_blobs")
    expected_sensor_path = Path(sensor_path).expanduser().resolve()
    if stored_sensor_path != expected_sensor_path:
        raise ValueError(f"World4Drive audit root {root_name!r} sensor_blobs_path does not match current config")
    if verification_mode == "live":
        _verify_live_tree(sensor, name=f"{root_name}.sensor_blobs")


def validate_world4drive_audit_payload(
    payload: object,
    *,
    expected_real_root: Mapping[str, object],
    verification_mode: str = "live",
) -> CvoiWorld4DriveAuditSignature:
    """Validate one immutable, live Real-validation audit payload."""

    if verification_mode != "live":
        raise ValueError("World4Drive dataset audit verification_mode must be exactly 'live'")
    if not isinstance(payload, Mapping):
        raise ValueError("World4Drive audit manifest must contain a mapping")
    manifest_version = payload.get("manifest_version")
    if type(manifest_version) is not int or manifest_version not in {1, 2}:
        raise ValueError("World4Drive audit manifest version is incompatible")
    if payload.get("contract") != _GEOMETRY_AUDIT_CONTRACT:
        raise ValueError("World4Drive audit manifest contract is incompatible")
    if payload.get("passed") is not True or payload.get("ready_for_labels") is not True:
        raise ValueError("World4Drive audit manifest must be passing and ready for labels")

    canonical_payload = dict(payload)
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
        raise ValueError("World4Drive audit manifest must contain canonical finite JSON values") from exc
    recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if stored_fingerprint != recomputed:
        raise ValueError("World4Drive audit manifest content does not match its fingerprint")
    dataset_fingerprint = _require_sha256("dataset_fingerprint", payload.get("dataset_fingerprint"))
    if dataset_fingerprint != stored_fingerprint:
        raise ValueError("World4Drive audit fingerprint and dataset_fingerprint must match")

    converter_fingerprint = _require_sha256("converter_fingerprint", payload.get("converter_fingerprint"))
    converter_path, converter_provenance_fingerprint = _validate_file_provenance(
        payload.get("converter"),
        name="converter",
    )
    if converter_fingerprint != converter_provenance_fingerprint:
        raise ValueError("World4Drive audit converter fingerprints disagree")
    if not converter_path.is_file():
        raise FileNotFoundError(f"World4Drive audited converter file no longer exists: {converter_path}")
    _verify_live_file(payload["converter"], name="converter")

    split_manifest = payload.get("split_manifest")
    if not isinstance(split_manifest, Mapping) or split_manifest.get("passed") is not True:
        raise ValueError("World4Drive audit manifest requires a passing split_manifest")
    split_fingerprint = _require_sha256(
        "split_manifest_fingerprint",
        split_manifest.get("fingerprint"),
    )
    _verify_live_file(split_manifest, name="split_manifest")
    _validate_real_root(
        payload,
        expected_real_root=expected_real_root,
        verification_mode=verification_mode,
        manifest_version=manifest_version,
    )
    return CvoiWorld4DriveAuditSignature(
        dataset_fingerprint=dataset_fingerprint,
        converter_fingerprint=converter_fingerprint,
        split_manifest_fingerprint=split_fingerprint,
    )


def load_world4drive_audit_manifest(
    path: str | Path,
    *,
    expected_real_root: Mapping[str, object],
) -> CvoiWorld4DriveAuditSignature:
    """Load and live-verify the complete Real World4Drive cohort audit."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"World4Drive audit manifest does not exist: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"World4Drive audit manifest is not valid JSON: {manifest_path}") from exc
    return validate_world4drive_audit_payload(
        payload,
        expected_real_root=expected_real_root,
        verification_mode="live",
    )


def cvoi_world4drive_code_signature(source_root: str | Path) -> str:
    """Hash the fixed retained model, cache, replay, evaluator, and report sources."""

    root = Path(source_root)
    digest = hashlib.sha256()
    for relative_path in WORLD4DRIVE_CODE_SIGNATURE_SOURCES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"World4Drive provenance source does not exist: {path}")
        relative = relative_path.encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        digest.update(bytes.fromhex(hashlib.sha256(path.read_bytes()).hexdigest()))
    return digest.hexdigest()


def _normalize_audit_signature(value: object) -> CvoiWorld4DriveAuditSignature:
    if isinstance(value, CvoiWorld4DriveAuditSignature):
        return value
    fields = (
        "dataset_fingerprint",
        "converter_fingerprint",
        "split_manifest_fingerprint",
    )
    if any(not hasattr(value, field) for field in fields):
        raise TypeError("evaluation_audit must expose the exact World4Drive audit signature fields")
    return CvoiWorld4DriveAuditSignature(**{field: getattr(value, field) for field in fields})


def build_world4drive_evaluation_provenance(
    binding: CvoiWorld4DriveRuntimeBinding,
    evaluator: PlanningOutcomeEvaluator | CanonicalWorld4DrivePlanningOutcomeEvaluator,
) -> CvoiWorld4DriveEvaluationProvenance:
    """Preserve the legacy canonical hashes from retained, validated artifacts."""

    if not isinstance(binding, CvoiWorld4DriveRuntimeBinding):
        raise TypeError("binding must be a CvoiWorld4DriveRuntimeBinding")
    if not isinstance(evaluator, (PlanningOutcomeEvaluator, CanonicalWorld4DrivePlanningOutcomeEvaluator)):
        raise TypeError("evaluator must be a registered Real outcome evaluator")
    runtime_payload = dict(build_world4drive_evaluation_runtime_signature_payload(binding))
    if runtime_payload.get("stage") != "evaluation":
        raise ValueError("World4Drive evaluation runtime payload must declare stage='evaluation'")
    common_runtime = {
        key: value
        for key, value in runtime_payload.items()
        if key
        not in {
            "stage",
            "audit_signature",
            "guidance_steps",
            "guidance_objective",
            "guidance_step_size",
            "guidance_max_delta_norm",
            "guidance_detach_output",
            "parent_planner_sha256",
            "dual_value_sha256",
            "gate_sha256",
        }
    }
    guidance_contract = {
        "steps": runtime_payload["guidance_steps"],
        "objective": runtime_payload["guidance_objective"],
        "step_size": runtime_payload["guidance_step_size"],
        "max_delta_norm": runtime_payload["guidance_max_delta_norm"],
        "detach_output": runtime_payload["guidance_detach_output"],
        "h0": "skip",
    }
    planner_sha256 = _require_sha256(
        "evaluation Planner SHA-256",
        runtime_payload["parent_planner_sha256"],
    )
    dual_value_sha256 = _require_sha256(
        "evaluation dual Value SHA-256",
        runtime_payload["dual_value_sha256"],
    )
    gate_sha256 = _require_sha256("evaluation Gate SHA-256", runtime_payload["gate_sha256"])
    p0_sha256 = _sha256_file(Path(binding.unguided_planner_checkpoint))
    sequential_model = {
        "policy": "sequential_gate",
        "common_runtime": common_runtime,
        "planner_sha256": planner_sha256,
        "dual_value_sha256": dual_value_sha256,
        "value_usage": ["field", "stop"],
        "gate_sha256": gate_sha256,
        "guidance": guidance_contract,
    }
    fixed_model = {
        "policy": "fixed_guidance",
        "common_runtime": common_runtime,
        "planner_sha256": planner_sha256,
        "dual_value_sha256": dual_value_sha256,
        "value_usage": ["field"],
        "guidance": guidance_contract,
    }
    p0_model = {
        "policy": "p0_unguided",
        "common_runtime": common_runtime,
        "planner_sha256": p0_sha256,
        "value_usage": [],
        "guidance": None,
    }
    p1_unguided_model = {
        "policy": "p1_unguided",
        "common_runtime": common_runtime,
        "planner_sha256": planner_sha256,
        "value_usage": [],
        "guidance": None,
    }
    cf_schema = counterfactual_quality_schema(timestep_sec=binding.timestep_sec, max_progress_m=20.0)
    return CvoiWorld4DriveEvaluationProvenance(
        artifact_signature=_canonical_sha256(runtime_payload["audit_signature"]),
        model_signature=_canonical_sha256(sequential_model),
        real_evaluator_signature=planning_outcome_evaluator_signature_sha256(evaluator.signature),
        counterfactual_evaluator_signature=_canonical_sha256(cf_schema),
        fixed_model_signature=_canonical_sha256(fixed_model),
        p0_model_signature=_canonical_sha256(p0_model),
        p1_unguided_model_signature=_canonical_sha256(p1_unguided_model),
    )


def build_world4drive_cache_provenance(
    binding: CvoiWorld4DriveRuntimeBinding,
    *,
    evaluator: PlanningOutcomeEvaluator | CanonicalWorld4DrivePlanningOutcomeEvaluator,
    source_root: str | Path,
    evaluation_audit: object,
    evaluation_dataset_signature: str,
) -> CvoiWorld4DriveCacheProvenance:
    """Bind one cache lineage to its policy, audited data, code, and evaluator."""

    if not isinstance(binding, CvoiWorld4DriveRuntimeBinding):
        raise TypeError("binding must be a CvoiWorld4DriveRuntimeBinding")
    if not isinstance(evaluator, (PlanningOutcomeEvaluator, CanonicalWorld4DrivePlanningOutcomeEvaluator)):
        raise TypeError("evaluator must be a registered Real outcome evaluator")
    audit = _normalize_audit_signature(evaluation_audit)
    _require_sha256("evaluation_dataset_signature", evaluation_dataset_signature)
    policy = build_world4drive_evaluation_provenance(binding, evaluator)
    checkpoint_signature = (
        policy.p0_model_signature if binding.lineage == "p0_controller" else policy.fixed_model_signature
    )
    return CvoiWorld4DriveCacheProvenance(
        checkpoint_signature=checkpoint_signature,
        dataset_signature=evaluation_dataset_signature,
        code_signature=cvoi_world4drive_code_signature(source_root),
        converter_signature=audit.converter_fingerprint,
        evaluator_signature=policy.real_evaluator_signature,
    )


def cvoi_world4drive_evaluation_dataset_signature(
    audit: CvoiWorld4DriveAuditSignature,
    *,
    real_val_root: Mapping[str, object],
    config: Any,
) -> str:
    """Bind the audited Real data to the exact retained stride-4 window semantics."""

    audit = _normalize_audit_signature(audit)
    if not isinstance(real_val_root, Mapping) or real_val_root.get("domain") != "real":
        raise ValueError("real_val_root must be a Real root mapping")
    if real_val_root.get("window_stride") != 4 or type(real_val_root.get("window_stride")) is not int:
        raise ValueError("World4Drive direct real_val_root.window_stride must be exactly 4")
    if real_val_root.get("max_scenes") is not None:
        raise ValueError("World4Drive direct real_val_root.max_scenes must be null")
    if real_val_root.get("max_agents", 256) != 256:
        raise ValueError("World4Drive direct real_val_root.max_agents must be exactly 256")
    navsim = getattr(config.data, "navsim", None)
    payload = {
        "schema": "cvoi_world4drive_evaluation_dataset_v1",
        "audit_dataset_fingerprint": audit.dataset_fingerprint,
        "dataset_id": real_val_root["dataset_id"],
        "domain": "real",
        "window_stride": 4,
        "max_scenes": None,
        "base_fps": real_val_root["base_fps"],
        "max_frame_gap": real_val_root["max_frame_gap"],
        "max_agents": 256,
        "frames_per_clip": int(config.data.num_target_frames),
        "fps": int(config.data.fps),
        "num_observed_frames": int(config.train.num_observed_frames),
        "image_require_policy": real_val_root.get(
            "image_require_policy",
            getattr(navsim, "image_require_policy", None),
        ),
    }
    canonical = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


build_cvoi_world4drive_cache_provenance = build_world4drive_cache_provenance

__all__ = (
    "WORLD4DRIVE_CODE_SIGNATURE_SOURCES",
    "CvoiWorld4DriveAuditSignature",
    "CvoiWorld4DriveCacheProvenance",
    "CvoiWorld4DriveEvaluationProvenance",
    "build_cvoi_world4drive_cache_provenance",
    "build_world4drive_cache_provenance",
    "build_world4drive_evaluation_provenance",
    "cvoi_world4drive_code_signature",
    "cvoi_world4drive_evaluation_dataset_signature",
    "load_world4drive_audit_manifest",
    "validate_world4drive_audit_payload",
)
