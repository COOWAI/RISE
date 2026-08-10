"""Tests for geometry-free NavSim counterfactual trajectory-quality generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from app.vjepa_cowa_world_model.training.cf_trajectory_quality import CF_QUALITY_SCHEMA
from app.vjepa_cowa_world_model.training.navsim_data import load_counterfactual_trajectory_quality

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "generate_navsim_cf_trajectory_quality.py"
FORBIDDEN_OUTPUT_FRAGMENTS = ("agent_box", "collision", "ttc", "offroad")


def _load_module():
    assert MODULE_PATH.is_file(), "NavSim CF trajectory-quality generator must exist"
    spec = importlib.util.spec_from_file_location("generate_navsim_cf_trajectory_quality", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_cf_scene(pkl_root: Path, scene_name: str, *, payload: bytes = b"opaque-cf-navsim-pkl") -> Path:
    pkl_root.mkdir(parents=True, exist_ok=True)
    path = pkl_root / f"{scene_name}.pkl"
    path.write_bytes(payload)
    return path


def _write_pose_overlay(pose_root: Path, scene_name: str, translation: np.ndarray) -> Path:
    pose_root.mkdir(parents=True, exist_ok=True)
    path = pose_root / f"{scene_name}.npz"
    np.savez(
        path,
        frame_indices=np.arange(translation.shape[0], dtype=np.int64),
        translation=np.asarray(translation, dtype=np.float64),
    )
    return path


def test_cli_generates_formal_quality_from_future_eight_frames_anchored_at_last_observed(tmp_path: Path) -> None:
    module = _load_module()
    scene_name = "scene-0001_cf_000010_000019"
    pkl_root = tmp_path / "train"
    pose_root = tmp_path / "pose_overlay"
    output_path = tmp_path / "trajectory_quality.json"
    _write_cf_scene(pkl_root, scene_name)
    # OpenCV z maps to ego-forward x. Frames 0..2 and 12..13 are deliberately
    # extreme: Formal quality must use only frames 4..11 relative to frame 3.
    _write_pose_overlay(
        pose_root,
        scene_name,
        np.asarray(
            [
                [0.0, 0.0, -100.0],
                [0.0, 0.0, -50.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 10.0],
                [0.0, 0.0, 12.0],
                [0.0, 0.0, 11.0],
                [0.0, 0.0, 14.0],
                [0.0, 0.0, 15.0],
                [0.0, 0.0, 16.0],
                [0.0, 0.0, 17.0],
                [0.0, 0.0, 18.0],
                [0.0, 0.0, 20.0],
                [0.0, 0.0, -500.0],
                [0.0, 0.0, 1000.0],
            ]
        ),
    )

    result = module.main(
        [
            "--pkl-root",
            str(pkl_root),
            "--pose-overlay-root",
            str(pose_root),
            "--output",
            str(output_path),
            "--timestep-sec",
            "0.5",
            "--pose-overlay-txt-start-seconds",
            "0.0",
            "--pose-overlay-coord-frame",
            "opencv_first_frame",
            "--max-progress-m",
            "20",
            "--formal-v2-timeline",
            "--pkl-fingerprint-scope",
            "relative_path_identity",
        ]
    )

    assert result == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema"] == CF_QUALITY_SCHEMA
    assert payload["generator_version"] == 2
    assert payload["pose_overlay_coord_frame"] == "opencv_first_frame"
    assert payload["scene_count"] == 1
    assert payload["timestep_sec"] == pytest.approx(0.5)
    assert payload["pose_overlay_txt_start_seconds"] == 0.0
    assert payload["max_progress_m"] == pytest.approx(20.0)
    assert payload["fingerprint_algorithm"] == "sha256"
    assert payload["timeline_contract"] == {
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
    }
    assert set(payload["source_fingerprints"]) == {"navsim_pkl_root", "pose_overlay_root"}
    assert all(len(source["fingerprint"]) == 64 for source in payload["source_fingerprints"].values())

    scene = payload["scenes"][scene_name]
    assert scene["status"] == "passed"
    assert scene["pose_count"] == 12
    assert scene["scored_pose_count"] == 8
    assert scene["metrics"]["progress_score"] == pytest.approx(0.5)
    assert scene["metrics"]["reverse_risk"] == pytest.approx(1.0 / 12.0)
    assert scene["metrics"]["path_efficiency"] == pytest.approx(10.0 / 12.0)
    expected_quality = (
        0.4 * scene["metrics"]["progress_score"]
        + 0.2 * (1.0 - scene["metrics"]["reverse_risk"])
        + 0.2 * scene["metrics"]["comfort_score"]
        + 0.2 * scene["metrics"]["path_efficiency"]
    )
    assert scene["quality_score"] == pytest.approx(expected_quality)

    normalized = load_counterfactual_trajectory_quality(
        str(output_path),
        expected_pose_overlay_txt_start_seconds=0.0,
        require_formal_v2_contract=True,
    )
    assert normalized[scene_name]["cf_quality"] == pytest.approx(expected_quality)


def test_formal_generation_fails_when_pose_has_fewer_than_twelve_timeline_frames(tmp_path: Path) -> None:
    module = _load_module()
    scene_name = "scene-short_cf_000010_000019"
    pkl_root = tmp_path / "train"
    pose_root = tmp_path / "pose_overlay"
    output_path = tmp_path / "trajectory_quality.json"
    _write_cf_scene(pkl_root, scene_name)
    _write_pose_overlay(pose_root, scene_name, np.zeros((11, 3), dtype=np.float64))

    with pytest.raises((KeyError, ValueError), match="12-frame|frame.*11|missing frame"):
        module.generate_sidecar(
            pkl_root=pkl_root,
            pose_overlay_root=pose_root,
            output_path=output_path,
            pose_overlay_coord_frame="opencv_first_frame",
            pose_overlay_txt_start_seconds=0.0,
            pkl_fingerprint_scope="relative_path_identity",
            formal_v2_timeline=True,
        )

    assert not output_path.exists()


def test_generation_fails_when_a_scene_pose_overlay_is_missing(tmp_path: Path) -> None:
    module = _load_module()
    scene_name = "scene-0002_cf_000020_000029"
    pkl_root = tmp_path / "train"
    pose_root = tmp_path / "pose_overlay"
    output_path = tmp_path / "trajectory_quality.json"
    _write_cf_scene(pkl_root, scene_name)
    pose_root.mkdir()

    with pytest.raises(FileNotFoundError, match="missing NavSim pose overlay scene"):
        module.generate_sidecar(
            pkl_root=pkl_root,
            pose_overlay_root=pose_root,
            output_path=output_path,
            pose_overlay_coord_frame="opencv_first_frame",
            pose_overlay_txt_start_seconds=0.0,
        )

    assert not output_path.exists()


def test_generation_can_limit_sidecar_to_the_preflight_scene_manifest(tmp_path: Path) -> None:
    module = _load_module()
    selected = "scene-0005_cf_000050_000059"
    unselected = "scene-0006_cf_000060_000069"
    pkl_root = tmp_path / "train"
    pose_root = tmp_path / "pose_overlay"
    output_path = tmp_path / "trajectory_quality.json"
    _write_cf_scene(pkl_root, selected)
    _write_cf_scene(pkl_root, unselected)
    _write_pose_overlay(
        pose_root,
        selected,
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    )

    payload = module.generate_sidecar(
        pkl_root=pkl_root,
        pose_overlay_root=pose_root,
        output_path=output_path,
        pose_overlay_coord_frame="opencv_first_frame",
        pose_overlay_txt_start_seconds=0.0,
        scene_names=[selected],
    )

    assert payload["scene_count"] == 1
    assert list(payload["scenes"]) == [selected]
    assert payload["source_fingerprints"]["navsim_pkl_root"]["file_count"] == 1


def test_cli_selects_the_exact_formal_v2_hazard_cohort_from_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    selected = "scene-0005_cf_000050_000059"
    annotations_path = tmp_path / "navsim_train.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "scene": selected,
                    "annos": {
                        "distortion": False,
                        "trajectory_match": True,
                        "accident": True,
                        "accident_type": "自车行为引起",
                    },
                },
                {
                    "scene": "scene-0006_cf_000060_000069",
                    "annos": {
                        "distortion": False,
                        "trajectory_match": False,
                        "accident": True,
                        "accident_type": "非自车行为引起",
                    },
                },
                {
                    "scene": "scene-0007_cf_000070_000079",
                    "annos": {
                        "distortion": False,
                        "trajectory_match": True,
                        "accident": True,
                        "accident_type": "有事故但与自车无关",
                    },
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    validated: dict[str, object] = {}
    monkeypatch.setattr(
        module,
        "validate_formal_v2_navsim_cf_annotation_source",
        lambda path, *, root_name: validated.update(path=path, root_name=root_name) or {"selected_record_count": 1},
    )
    monkeypatch.setattr(module, "generate_sidecar", lambda **kwargs: captured.update(kwargs))

    assert (
        module.main(
            [
                "--pkl-root",
                str(tmp_path / "pkls"),
                "--pose-overlay-root",
                str(tmp_path / "poses"),
                "--output",
                str(tmp_path / "quality.json"),
                "--pose-overlay-coord-frame",
                "opencv_first_frame",
                "--pose-overlay-txt-start-seconds",
                "0.0",
                "--formal-v2-timeline",
                "--pkl-fingerprint-scope",
                "relative_path_identity",
                "--formal-v2-annotations",
                str(annotations_path),
                "--camera-name",
                "CAM_F0",
            ]
        )
        == 0
    )
    assert captured["scene_names"] == [selected]
    assert validated == {
        "path": annotations_path,
        "root_name": "navsim_cf_train",
    }


def test_formal_v2_annotation_selection_rejects_locked_selected_count_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    annotations_path = tmp_path / "navsim_train.json"
    annotations_path.write_text(
        json.dumps(
            [
                {
                    "scene": "scene-0005_cf_000050_000059",
                    "annos": {
                        "distortion": False,
                        "trajectory_match": True,
                        "accident": True,
                        "accident_type": "自车行为引起",
                    },
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "validate_formal_v2_navsim_cf_annotation_source",
        lambda *_args, **_kwargs: {"selected_record_count": 2},
    )

    with pytest.raises(ValueError, match=r"selected_record_count mismatch.*expected=2, actual=1"):
        module._formal_v2_selected_scene_names(
            annotations_path,
            camera_name="CAM_F0",
        )


def test_cli_rejects_drifted_formal_v2_annotations_before_sidecar_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    annotations_path = tmp_path / "navsim_test.json"
    annotations_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "validate_formal_v2_navsim_cf_annotation_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("source_sha256 mismatch")),
    )
    monkeypatch.setattr(
        module,
        "generate_sidecar",
        lambda **_kwargs: pytest.fail("sidecar generation must not run after annotation drift"),
    )

    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        module.main(
            [
                "--pkl-root",
                str(tmp_path / "pkls"),
                "--pose-overlay-root",
                str(tmp_path / "poses"),
                "--output",
                str(tmp_path / "quality.json"),
                "--pose-overlay-coord-frame",
                "opencv_first_frame",
                "--pose-overlay-txt-start-seconds",
                "0.0",
                "--formal-v2-timeline",
                "--pkl-fingerprint-scope",
                "relative_path_identity",
                "--formal-v2-annotations",
                str(annotations_path),
                "--camera-name",
                "CAM_F0",
            ]
        )


def test_generation_fails_on_nonfinite_full_pose_trajectory(tmp_path: Path) -> None:
    module = _load_module()
    scene_name = "scene-0003_cf_000030_000039"
    pkl_root = tmp_path / "train"
    pose_root = tmp_path / "pose_overlay"
    output_path = tmp_path / "trajectory_quality.json"
    _write_cf_scene(pkl_root, scene_name)
    _write_pose_overlay(
        pose_root,
        scene_name,
        np.asarray([[0.0, 0.0, 0.0], [0.0, np.nan, 1.0]], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="non-finite"):
        module.generate_sidecar(
            pkl_root=pkl_root,
            pose_overlay_root=pose_root,
            output_path=output_path,
            pose_overlay_coord_frame="opencv_first_frame",
            pose_overlay_txt_start_seconds=0.0,
        )

    assert not output_path.exists()


def test_generated_payload_never_contains_geometry_or_safety_fields(tmp_path: Path) -> None:
    module = _load_module()
    scene_name = "scene-0004_cf_000040_000049"
    pkl_root = tmp_path / "train"
    pose_root = tmp_path / "pose_overlay"
    output_path = tmp_path / "trajectory_quality.json"
    # Deliberately invalid pickle bytes prove the generator treats PKLs as
    # opaque scene identities/fingerprint sources and never deserializes them.
    _write_cf_scene(pkl_root, scene_name, payload=b"not-a-readable-pickle")
    _write_pose_overlay(
        pose_root,
        scene_name,
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    )

    payload = module.generate_sidecar(
        pkl_root=pkl_root,
        pose_overlay_root=pose_root,
        output_path=output_path,
        pose_overlay_coord_frame="opencv_first_frame",
        pose_overlay_txt_start_seconds=0.0,
    )

    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in FORBIDDEN_OUTPUT_FRAGMENTS:
        assert forbidden not in serialized


def test_relative_path_identity_mode_never_reads_navsim_pkl_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    scene_name = "scene-identity_cf_000001_000009"
    pkl_root = tmp_path / "train"
    pose_root = tmp_path / "pose_overlay"
    output_path = tmp_path / "trajectory_quality.json"
    _write_cf_scene(pkl_root, scene_name, payload=b"must-not-be-read")
    _write_pose_overlay(
        pose_root,
        scene_name,
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    )
    original_sha256_file = module._sha256_file

    def reject_pkl_read(path: Path) -> str:
        if path.suffix == ".pkl":
            pytest.fail("relative_path_identity mode must not read NavSim PKL bytes")
        return original_sha256_file(path)

    monkeypatch.setattr(module, "_sha256_file", reject_pkl_read)

    payload = module.generate_sidecar(
        pkl_root=pkl_root,
        pose_overlay_root=pose_root,
        output_path=output_path,
        pose_overlay_coord_frame="opencv_first_frame",
        pose_overlay_txt_start_seconds=0.0,
        pkl_fingerprint_scope="relative_path_identity",
    )

    assert payload["source_fingerprint_scopes"] == {
        "navsim_pkl": "relative_path_identity",
        "pose_overlay": "content_sha256",
    }
    assert len(payload["scenes"][scene_name]["source_fingerprints"]["navsim_pkl"]) == 64


def test_existing_output_fails_before_reading_sources_and_is_not_overwritten(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "trajectory_quality.json"
    original = b"existing-sidecar-must-remain-byte-identical"
    output_path.write_bytes(original)

    with pytest.raises(FileExistsError, match="already exists"):
        module.generate_sidecar(
            pkl_root=tmp_path / "missing-pkl-root",
            pose_overlay_root=tmp_path / "missing-pose-root",
            output_path=output_path,
            pose_overlay_coord_frame="opencv_first_frame",
            pose_overlay_txt_start_seconds=0.0,
        )

    assert output_path.read_bytes() == original


def test_generation_uses_explicit_zero_second_txt_timeline_without_skipping_bad_rows(tmp_path: Path) -> None:
    module = _load_module()
    scene_name = "scene-0007_cf_000070_000079"
    pkl_root = tmp_path / "train"
    pose_root = tmp_path / "pose_overlay"
    output_path = tmp_path / "trajectory_quality.json"
    _write_cf_scene(pkl_root, scene_name)
    pose_root.mkdir(parents=True)
    matrices = np.tile(np.eye(4, dtype=np.float64)[:3].reshape(1, 12), (21, 1))
    matrices[:15, 3] = np.nan
    pose_path = pose_root / (scene_name.replace("_cf_", "_CAM_F0_") + "_gen.txt")
    np.savetxt(pose_path, matrices)

    with pytest.raises(ValueError, match="non-finite"):
        module.generate_sidecar(
            pkl_root=pkl_root,
            pose_overlay_root=pose_root,
            output_path=output_path,
            pose_overlay_coord_frame="opencv_first_frame",
            pose_overlay_txt_start_seconds=0.0,
        )

    assert not output_path.exists()


def test_generation_rejects_noncanonical_pose_overlay_coordinate_frame(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "trajectory_quality.json"

    with pytest.raises(ValueError, match="pose_overlay_coord_frame.*opencv_first_frame"):
        module.generate_sidecar(
            pkl_root=tmp_path / "missing-pkl-root",
            pose_overlay_root=tmp_path / "missing-pose-root",
            output_path=output_path,
            pose_overlay_coord_frame="ego",
            pose_overlay_txt_start_seconds=0.0,
        )

    assert not output_path.exists()
