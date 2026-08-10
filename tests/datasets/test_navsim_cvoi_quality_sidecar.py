"""Counterfactual trajectory-quality sidecar contract for NavSim."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from app.vjepa_cowa_world_model.training.cf_trajectory_quality import CF_QUALITY_SCHEMA
from app.vjepa_cowa_world_model.training.navsim_data import load_counterfactual_trajectory_quality
from app.vjepa_cowa_world_model.training.pose_overlay import PoseOverlayReader


def _write_sidecar(tmp_path, scene_entry, **metadata_overrides):
    path = tmp_path / "trajectory_quality.json"
    metadata = {
        "generator_version": 2,
        "pose_overlay_coord_frame": "opencv_first_frame",
        "pose_overlay_txt_start_seconds": 0.0,
    }
    metadata.update(metadata_overrides)
    path.write_text(
        json.dumps(
            {
                "schema": CF_QUALITY_SCHEMA,
                **metadata,
                "scenes": {"scene-0001_cf_000010_000019": scene_entry},
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_formal_sidecar(tmp_path, scene_entry, *, pose_sha256="0" * 64, **metadata_overrides):
    scene_name = "scene-0001_cf_000010_000019"
    metadata = {
        "generator_version": 2,
        "fingerprint_algorithm": "sha256",
        "timestep_sec": 0.5,
        "pose_overlay_coord_frame": "opencv_first_frame",
        "pose_overlay_txt_start_seconds": 0.0,
        "max_progress_m": 20.0,
        "weights": {
            "progress": 0.4,
            "non_reverse": 0.2,
            "comfort": 0.2,
            "path_efficiency": 0.2,
        },
        "source_fingerprint_scopes": {
            "navsim_pkl": "relative_path_identity",
            "pose_overlay": "content_sha256",
        },
        "timeline_contract": {
            "schema": "navsim_formal_v2_cf_quality_12_4_8_v1",
            "num_total_frames": 12,
            "num_observed_frames": 4,
            "num_target_frames": 8,
            "window_start_policy": "counterfactual_scene_start",
            "window_start_frame_index": 0,
            "sample_step": 1,
            "anchor_frame_index": 3,
            "anchor_semantics": "last_observed_pose_local_frame",
            "scored_frame_indices": list(range(4, 12)),
        },
        "scene_count": 1,
        "source_fingerprints": {
            "navsim_pkl_root": {"fingerprint": "1" * 64, "file_count": 1},
            "pose_overlay_root": {"fingerprint": "2" * 64, "file_count": 1},
        },
    }
    metadata.update(metadata_overrides)
    enriched_entry = {
        **scene_entry,
        "pose_count": 12,
        "scored_pose_count": 8,
        "source_fingerprints": {
            "navsim_pkl": "3" * 64,
            "pose_overlay": pose_sha256,
        },
    }
    path = tmp_path / "formal_trajectory_quality.json"
    path.write_text(
        json.dumps({"schema": CF_QUALITY_SCHEMA, **metadata, "scenes": {scene_name: enriched_entry}}),
        encoding="utf-8",
    )
    return path


def test_load_quality_sidecar_derives_canonical_cf_quality_from_components(tmp_path) -> None:
    path = _write_sidecar(
        tmp_path,
        {
            "status": "passed",
            "metrics": {
                "progress_score": 0.8,
                "reverse_risk": 0.1,
                "comfort_score": 0.7,
                "path_efficiency": 0.9,
            },
            "quality_score": 0.82,
        },
    )

    sidecar = load_counterfactual_trajectory_quality(str(path))

    quality = sidecar["scene-0001_cf_000010_000019"]
    assert quality["cf_quality"] == pytest.approx(0.82)
    assert quality["cf_progress_score"] == pytest.approx(0.8)
    assert quality["cf_reverse_risk"] == pytest.approx(0.1)
    assert quality["cf_comfort_score"] == pytest.approx(0.7)
    assert quality["cf_path_efficiency"] == pytest.approx(0.9)
    assert quality["cf_quality_schema"] == CF_QUALITY_SCHEMA


def test_load_quality_sidecar_rejects_status_without_quality_components(tmp_path) -> None:
    path = _write_sidecar(tmp_path, {"status": "passed"})

    with pytest.raises(ValueError, match="progress_score"):
        load_counterfactual_trajectory_quality(str(path))


def test_load_quality_sidecar_rejects_inconsistent_aggregate_score(tmp_path) -> None:
    path = _write_sidecar(
        tmp_path,
        {
            "metrics": {
                "progress_score": 0.8,
                "reverse_risk": 0.1,
                "comfort_score": 0.7,
                "path_efficiency": 0.9,
            },
            "quality_score": 0.2,
        },
    )

    with pytest.raises(ValueError, match="quality_score.*canonical"):
        load_counterfactual_trajectory_quality(str(path))


@pytest.mark.parametrize("forbidden", ["collision", "min_ttc", "offroad", "agent_boxes"])
def test_load_quality_sidecar_rejects_safety_or_geometry_fields(tmp_path, forbidden) -> None:
    path = _write_sidecar(
        tmp_path,
        {
            "metrics": {
                "progress_score": 0.8,
                "reverse_risk": 0.1,
                "comfort_score": 0.7,
                "path_efficiency": 0.9,
                forbidden: 0.0,
            },
        },
    )

    with pytest.raises(ValueError, match="forbidden"):
        load_counterfactual_trajectory_quality(str(path))


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"generator_version": 1}, "generator_version.*2"),
        ({"generator_version": None}, "generator_version.*2"),
        ({"pose_overlay_coord_frame": "ego"}, "pose_overlay_coord_frame.*opencv_first_frame"),
        ({"pose_overlay_coord_frame": None}, "pose_overlay_coord_frame.*opencv_first_frame"),
        ({"pose_overlay_txt_start_seconds": -0.5}, "pose_overlay_txt_start_seconds.*non-negative"),
        ({"pose_overlay_txt_start_seconds": None}, "pose_overlay_txt_start_seconds.*finite"),
    ],
)
def test_load_quality_sidecar_rejects_noncanonical_pose_generation_semantics(
    tmp_path,
    metadata,
    message,
) -> None:
    path = _write_sidecar(
        tmp_path,
        {
            "metrics": {
                "progress_score": 0.8,
                "reverse_risk": 0.1,
                "comfort_score": 0.7,
                "path_efficiency": 0.9,
            }
        },
        **metadata,
    )

    with pytest.raises(ValueError, match=message):
        load_counterfactual_trajectory_quality(str(path))


def test_load_quality_sidecar_rejects_pose_start_that_differs_from_runtime_root(tmp_path) -> None:
    path = _write_sidecar(
        tmp_path,
        {
            "metrics": {
                "progress_score": 0.8,
                "reverse_risk": 0.1,
                "comfort_score": 0.7,
                "path_efficiency": 0.9,
            }
        },
        pose_overlay_txt_start_seconds=1.5,
    )

    with pytest.raises(ValueError, match="pose_overlay_txt_start_seconds.*expected=0.0"):
        load_counterfactual_trajectory_quality(
            str(path),
            expected_pose_overlay_txt_start_seconds=0.0,
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"scene_count": 2}, "scene_count.*scenes"),
        ({"fingerprint_algorithm": "md5"}, "fingerprint_algorithm.*sha256"),
        ({"timestep_sec": 1.0}, "timestep_sec.*0.5"),
        ({"max_progress_m": 10.0}, "max_progress_m.*20.0"),
        ({"weights": {"progress": 1.0}}, "weights"),
        (
            {"source_fingerprint_scopes": {"navsim_pkl": "content_sha256", "pose_overlay": "content_sha256"}},
            "source_fingerprint_scopes",
        ),
        (
            {
                "timeline_contract": {
                    "schema": "navsim_formal_v2_cf_quality_12_4_8_v1",
                    "num_total_frames": 12,
                    "num_observed_frames": 4,
                    "num_target_frames": 8,
                    "window_start_policy": "counterfactual_scene_start",
                    "window_start_frame_index": 0,
                    "sample_step": 1,
                    "anchor_frame_index": 0,
                    "anchor_semantics": "first_pose_local_frame",
                    "scored_frame_indices": list(range(4, 12)),
                }
            },
            "timeline_contract",
        ),
    ],
)
def test_formal_quality_sidecar_rejects_global_contract_drift(tmp_path, metadata, message) -> None:
    path = _write_formal_sidecar(
        tmp_path,
        {
            "status": "passed",
            "metrics": {
                "progress_score": 0.8,
                "reverse_risk": 0.1,
                "comfort_score": 0.7,
                "path_efficiency": 0.9,
            },
        },
        **metadata,
    )

    with pytest.raises(ValueError, match=message):
        load_counterfactual_trajectory_quality(
            str(path),
            expected_pose_overlay_txt_start_seconds=0.0,
            require_formal_v2_contract=True,
        )


def test_formal_quality_sidecar_pose_fingerprint_is_checked_when_overlay_is_preloaded(tmp_path) -> None:
    scene_name = "scene-0001_cf_000010_000019"
    pose_root = tmp_path / "poses"
    pose_root.mkdir()
    pose_path = pose_root / f"{scene_name}.npz"
    np.savez(
        pose_path,
        frame_indices=np.arange(12, dtype=np.int64),
        translation=np.zeros((12, 3), dtype=np.float64),
    )
    pose_sha256 = hashlib.sha256(pose_path.read_bytes()).hexdigest()
    path = _write_formal_sidecar(
        tmp_path,
        {
            "status": "passed",
            "metrics": {
                "progress_score": 0.8,
                "reverse_risk": 0.1,
                "comfort_score": 0.7,
                "path_efficiency": 0.9,
            },
        },
        pose_sha256=pose_sha256,
    )
    reader = PoseOverlayReader(pose_root, txt_start_seconds=0.0)
    load_counterfactual_trajectory_quality(
        str(path),
        expected_pose_overlay_txt_start_seconds=0.0,
        require_formal_v2_contract=True,
        pose_overlay_reader=reader,
    )
    with pose_path.open("ab") as handle:
        handle.write(b"drift")

    with pytest.raises(ValueError, match="pose overlay SHA256 mismatch.*scene-0001"):
        reader.preload_scenes([scene_name])
