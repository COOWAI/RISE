"""Tests for NavSim multi-camera dataset outputs."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.vjepa_cowa_world_model.training.navsim_data import (
    NavSimWorldModelDataset,
    RootTaggedDataset,
    navsim_world_model_collate_fn,
)

CAMERA_NAMES = ["CAM_L0", "CAM_F0", "CAM_R0"]


def _make_frame(frame_idx: int) -> dict:
    cams = {}
    for cam_idx, name in enumerate(CAMERA_NAMES):
        intrinsic = np.array(
            [
                [100.0 + cam_idx, 0.0, 32.0],
                [0.0, 120.0 + cam_idx, 24.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        cams[name] = {
            "data_path": f"{name}/{frame_idx:03d}.jpg",
            "cam_intrinsic": intrinsic,
            "sensor2ego_translation": np.array([cam_idx, 0.0, 1.0], dtype=np.float32),
            "sensor2ego_rotation": np.eye(3, dtype=np.float32),
        }
    return {
        "cams": cams,
        "ego2global_translation": np.array([float(frame_idx), 0.0, 0.0], dtype=np.float64),
        "ego2global_rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ego_dynamic_state": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "driving_command": np.array([1, 0, 0, 0], dtype=np.float32),
    }


def _write_navsim_scene(root: Path, *, omit_camera: str | None = None) -> tuple[Path, Path]:
    logs_dir = root / "logs"
    blobs_dir = root / "sensor_blobs"
    scene_name = "scene_001"
    logs_dir.mkdir(parents=True)

    frames = [_make_frame(i) for i in range(4)]
    with open(logs_dir / f"{scene_name}.pkl", "wb") as handle:
        pickle.dump(frames, handle)

    for camera_name in CAMERA_NAMES:
        if camera_name == omit_camera:
            continue
        camera_dir = blobs_dir / scene_name / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(4):
            image = np.full((48, 64, 3), fill_value=frame_idx + len(camera_name), dtype=np.uint8)
            Image.fromarray(image).save(camera_dir / f"{frame_idx:03d}.jpg")
    return logs_dir, blobs_dir


class TestNavSimMultiViewData(unittest.TestCase):
    def test_single_view_shape_remains_compatible(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir, blobs_dir = _write_navsim_scene(Path(tmp_dir))

            dataset = NavSimWorldModelDataset(
                data_path=str(logs_dir),
                sensor_blobs_path=str(blobs_dir),
                camera_name="CAM_F0",
                frames_per_clip=4,
                fps=2,
                index_cache=False,
                load_agent_annotations=False,
            )
            sample = dataset[0]
            self.assertEqual(tuple(sample["buffer"].shape), (3, 4, 48, 64))

    def test_multiview_sample_and_collate_include_camera_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir, blobs_dir = _write_navsim_scene(Path(tmp_dir))

            raw_dataset = NavSimWorldModelDataset(
                data_path=str(logs_dir),
                sensor_blobs_path=str(blobs_dir),
                camera_names=CAMERA_NAMES,
                frames_per_clip=4,
                fps=2,
                index_cache=False,
                load_agent_annotations=False,
            )
            dataset = RootTaggedDataset(
                raw_dataset,
                domain="real",
                dataset_root_name="test-real",
                dataset_root_index=0,
                future_agent_geometry_valid=False,
                load_agent_annotations=False,
            )

            sample = dataset[0]
            self.assertEqual(tuple(sample["buffer"].shape), (3, 3, 4, 48, 64))
            self.assertEqual(sample["camera_names"], CAMERA_NAMES)
            self.assertEqual(tuple(sample["camera_intrinsics"].shape), (3, 4, 3, 3))
            self.assertEqual(tuple(sample["camera2ego"].shape), (3, 4, 4, 4))

            batch = navsim_world_model_collate_fn([sample, sample])
            context_frames = batch[0]
            metadata = batch[-1]
            self.assertEqual(tuple(context_frames.shape), (2, 3, 3, 4, 48, 64))
            self.assertEqual(tuple(metadata["camera_intrinsics"].shape), (2, 3, 4, 3, 3))
            self.assertEqual(tuple(metadata["camera2ego"].shape), (2, 3, 4, 4, 4))
            self.assertEqual(metadata["camera_names"], CAMERA_NAMES)

    def test_multiview_missing_camera_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir, blobs_dir = _write_navsim_scene(Path(tmp_dir), omit_camera="CAM_R0")

            with self.assertRaises(ValueError):
                NavSimWorldModelDataset(
                    data_path=str(logs_dir),
                    sensor_blobs_path=str(blobs_dir),
                    camera_names=CAMERA_NAMES,
                    frames_per_clip=4,
                    fps=2,
                    index_cache=False,
                    load_agent_annotations=False,
                )


if __name__ == "__main__":
    unittest.main()
