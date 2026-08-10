"""Strict provenance contract for generated counterfactual quality sidecars."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

CF_QUALITY_SCHEMA = "cf_progress_reverse_comfort_efficiency_v1"
CF_QUALITY_GENERATOR_VERSION = 2
CF_QUALITY_POSE_OVERLAY_COORD_FRAME = "opencv_first_frame"
FORMAL_V2_CF_QUALITY_TIMELINE_SCHEMA = "navsim_formal_v2_cf_quality_12_4_8_v1"
FORMAL_V2_CF_QUALITY_TIMESTEP_SEC = 0.5
FORMAL_V2_CF_QUALITY_MAX_PROGRESS_M = 20.0
FORMAL_V2_CF_QUALITY_WEIGHTS = {
    "progress": 0.4,
    "non_reverse": 0.2,
    "comfort": 0.2,
    "path_efficiency": 0.2,
}
FORMAL_V2_CF_QUALITY_FINGERPRINT_SCOPES = {
    "navsim_pkl": "relative_path_identity",
    "pose_overlay": "content_sha256",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def formal_v2_cf_quality_timeline_contract() -> dict[str, Any]:
    """Return the immutable Formal-v2 12/4/8 CF quality timeline contract."""

    return {
        "schema": FORMAL_V2_CF_QUALITY_TIMELINE_SCHEMA,
        "num_total_frames": 12,
        "num_observed_frames": 4,
        "num_target_frames": 8,
        "window_start_policy": "counterfactual_scene_start",
        "window_start_frame_index": 0,
        "sample_step": 1,
        "anchor_frame_index": 3,
        "anchor_semantics": "last_observed_pose_local_frame",
        "scored_frame_indices": list(range(4, 12)),
    }


def _require_sha256(value: Any, *, field_name: str, source: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Counterfactual trajectory quality {field_name} must be lowercase SHA256: {source}")
    return value


def _strict_contract_equal(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_strict_contract_equal(actual[key], expected[key]) for key in expected)
        )
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(
                _strict_contract_equal(actual_item, expected_item)
                for actual_item, expected_item in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _validate_formal_v2_metadata(payload: Mapping[str, Any], *, source: str) -> dict[str, str]:
    scenes = payload.get("scenes")
    if not isinstance(scenes, Mapping) or not scenes:
        raise ValueError(f"Formal-v2 trajectory quality requires a non-empty scenes mapping: {source}")
    scene_count = payload.get("scene_count")
    if type(scene_count) is not int or scene_count != len(scenes):
        raise ValueError(
            "Formal-v2 trajectory quality scene_count must exactly match scenes: "
            f"scene_count={scene_count!r}, scenes={len(scenes)}, source={source}"
        )
    if payload.get("fingerprint_algorithm") != "sha256":
        raise ValueError(f"Formal-v2 trajectory quality fingerprint_algorithm must be 'sha256': {source}")
    if not _strict_contract_equal(payload.get("timestep_sec"), FORMAL_V2_CF_QUALITY_TIMESTEP_SEC):
        raise ValueError(
            "Formal-v2 trajectory quality timestep_sec must be exactly "
            f"{FORMAL_V2_CF_QUALITY_TIMESTEP_SEC}: {source}"
        )
    if not _strict_contract_equal(payload.get("max_progress_m"), FORMAL_V2_CF_QUALITY_MAX_PROGRESS_M):
        raise ValueError(
            "Formal-v2 trajectory quality max_progress_m must be exactly "
            f"{FORMAL_V2_CF_QUALITY_MAX_PROGRESS_M}: {source}"
        )
    if not _strict_contract_equal(payload.get("weights"), FORMAL_V2_CF_QUALITY_WEIGHTS):
        raise ValueError(f"Formal-v2 trajectory quality weights must be exactly canonical: {source}")
    if not _strict_contract_equal(payload.get("source_fingerprint_scopes"), FORMAL_V2_CF_QUALITY_FINGERPRINT_SCOPES):
        raise ValueError(f"Formal-v2 trajectory quality source_fingerprint_scopes must be exactly canonical: {source}")
    expected_timeline = formal_v2_cf_quality_timeline_contract()
    if not _strict_contract_equal(payload.get("timeline_contract"), expected_timeline):
        raise ValueError(
            f"Formal-v2 trajectory quality timeline_contract must be exactly {expected_timeline!r}: {source}"
        )

    root_fingerprints = payload.get("source_fingerprints")
    if not isinstance(root_fingerprints, Mapping) or set(root_fingerprints) != {
        "navsim_pkl_root",
        "pose_overlay_root",
    }:
        raise ValueError(f"Formal-v2 trajectory quality source_fingerprints roots must be exact: {source}")
    for root_name in ("navsim_pkl_root", "pose_overlay_root"):
        root_entry = root_fingerprints[root_name]
        if not isinstance(root_entry, Mapping) or set(root_entry) != {"fingerprint", "file_count"}:
            raise ValueError(f"Formal-v2 trajectory quality {root_name} fingerprint entry must be exact: {source}")
        _require_sha256(root_entry.get("fingerprint"), field_name=f"{root_name}.fingerprint", source=source)
        if type(root_entry.get("file_count")) is not int or root_entry["file_count"] != scene_count:
            raise ValueError(
                f"Formal-v2 trajectory quality {root_name}.file_count must equal scene_count={scene_count}: {source}"
            )

    pose_sha256_by_scene: dict[str, str] = {}
    for raw_scene_name, raw_entry in scenes.items():
        scene_name = str(raw_scene_name)
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"Formal-v2 trajectory quality scene {scene_name!r} must be a mapping: {source}")
        if raw_entry.get("pose_count") != 12 or raw_entry.get("scored_pose_count") != 8:
            raise ValueError(
                f"Formal-v2 trajectory quality scene {scene_name!r} must declare pose_count=12 and "
                f"scored_pose_count=8: {source}"
            )
        scene_fingerprints = raw_entry.get("source_fingerprints")
        if not isinstance(scene_fingerprints, Mapping) or set(scene_fingerprints) != {
            "navsim_pkl",
            "pose_overlay",
        }:
            raise ValueError(
                f"Formal-v2 trajectory quality scene {scene_name!r} source_fingerprints must be exact: {source}"
            )
        _require_sha256(
            scene_fingerprints.get("navsim_pkl"),
            field_name=f"scenes.{scene_name}.source_fingerprints.navsim_pkl",
            source=source,
        )
        pose_sha256_by_scene[scene_name] = _require_sha256(
            scene_fingerprints.get("pose_overlay"),
            field_name=f"scenes.{scene_name}.source_fingerprints.pose_overlay",
            source=source,
        )
    return pose_sha256_by_scene


def validate_counterfactual_quality_sidecar_metadata(
    payload: Mapping[str, Any],
    *,
    source: str,
    expected_pose_overlay_txt_start_seconds: float | None = None,
    require_formal_v2_contract: bool = False,
) -> dict[str, str]:
    """Validate the immutable generator and pose interpretation metadata."""

    if not isinstance(payload, Mapping):
        raise ValueError(f"Counterfactual trajectory quality must be a mapping: {source}")
    if payload.get("schema") != CF_QUALITY_SCHEMA:
        raise ValueError(f"Counterfactual trajectory quality schema must be {CF_QUALITY_SCHEMA!r}: {source}")
    generator_version = payload.get("generator_version")
    if type(generator_version) is not int or generator_version != CF_QUALITY_GENERATOR_VERSION:
        raise ValueError(
            "Counterfactual trajectory quality generator_version must be exactly "
            f"{CF_QUALITY_GENERATOR_VERSION}: {source}"
        )
    coord_frame = payload.get("pose_overlay_coord_frame")
    if type(coord_frame) is not str or coord_frame != CF_QUALITY_POSE_OVERLAY_COORD_FRAME:
        raise ValueError(
            "Counterfactual trajectory quality pose_overlay_coord_frame must be exactly "
            f"{CF_QUALITY_POSE_OVERLAY_COORD_FRAME!r}: {source}"
        )
    start_seconds = payload.get("pose_overlay_txt_start_seconds")
    if type(start_seconds) is not float or not math.isfinite(start_seconds):
        raise ValueError(
            "Counterfactual trajectory quality pose_overlay_txt_start_seconds must be a finite float: " f"{source}"
        )
    if start_seconds < 0.0:
        raise ValueError(
            "Counterfactual trajectory quality pose_overlay_txt_start_seconds must be non-negative: " f"{source}"
        )
    if expected_pose_overlay_txt_start_seconds is not None:
        if (
            isinstance(expected_pose_overlay_txt_start_seconds, bool)
            or not isinstance(expected_pose_overlay_txt_start_seconds, (int, float))
            or not math.isfinite(float(expected_pose_overlay_txt_start_seconds))
            or float(expected_pose_overlay_txt_start_seconds) < 0.0
        ):
            raise ValueError("expected_pose_overlay_txt_start_seconds must be finite and non-negative")
        expected_start = float(expected_pose_overlay_txt_start_seconds)
        if start_seconds != expected_start:
            raise ValueError(
                "Counterfactual trajectory quality pose_overlay_txt_start_seconds differs from runtime root: "
                f"expected={expected_start}, actual={start_seconds}, source={source}"
            )
    if type(require_formal_v2_contract) is not bool:
        raise TypeError("require_formal_v2_contract must be an exact boolean")
    if require_formal_v2_contract:
        return _validate_formal_v2_metadata(payload, source=source)
    return {}
