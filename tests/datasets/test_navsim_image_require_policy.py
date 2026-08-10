"""Tests for NavSim image requirements by training mode."""

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


def _make_frame(frame_idx: int) -> dict:
    return {
        "cams": {
            "CAM_F0": {
                "data_path": f"scene_001/CAM_F0/{frame_idx:03d}.jpg",
                "cam_intrinsic": np.eye(3, dtype=np.float32),
                "sensor2ego_translation": np.zeros(3, dtype=np.float32),
                "sensor2ego_rotation": np.eye(3, dtype=np.float32),
            }
        },
        "ego2global_translation": np.array([float(frame_idx), 0.0, 0.0], dtype=np.float64),
        "ego2global_rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ego_dynamic_state": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "driving_command": np.array([1, 0, 0, 0], dtype=np.float32),
        "timestamp": frame_idx * 500_000,
    }


def _write_scene(root: Path, *, num_frames: int, image_frames: range) -> tuple[Path, Path]:
    logs_dir = root / "logs"
    blobs_dir = root / "sensor_blobs"
    scene_name = "scene_001"
    camera_dir = blobs_dir / scene_name / "CAM_F0"
    logs_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)

    frames = [_make_frame(frame_idx) for frame_idx in range(num_frames)]
    with open(logs_dir / f"{scene_name}.pkl", "wb") as handle:
        pickle.dump(frames, handle)

    for frame_idx in image_frames:
        image = np.full((24, 32, 3), fill_value=frame_idx, dtype=np.uint8)
        Image.fromarray(image).save(camera_dir / f"{frame_idx:03d}.jpg")

    return logs_dir, blobs_dir


def _write_scene_with_metadata_valid(root: Path) -> tuple[Path, Path]:
    logs_dir = root / "logs"
    blobs_dir = root / "sensor_blobs"
    scene_name = "scene_001"
    camera_dir = blobs_dir / scene_name / "CAM_F0"
    logs_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)

    frames = []
    for frame_idx in range(11):
        frame = _make_frame(frame_idx)
        frame["metadata_valid"] = frame_idx < 4
        frames.append(frame)
    with open(logs_dir / f"{scene_name}.pkl", "wb") as handle:
        pickle.dump(frames, handle)

    for frame_idx in range(11):
        image = np.full((24, 32, 3), fill_value=frame_idx, dtype=np.uint8)
        Image.fromarray(image).save(camera_dir / f"{frame_idx:03d}.jpg")

    return logs_dir, blobs_dir


def _write_scene_with_future_agent_annotations(root: Path, *, include_empty_future_anns: bool) -> tuple[Path, Path]:
    logs_dir = root / "logs"
    blobs_dir = root / "sensor_blobs"
    scene_name = "scene_001"
    camera_dir = blobs_dir / scene_name / "CAM_F0"
    logs_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)

    frames = [_make_frame(frame_idx) for frame_idx in range(4)]
    if include_empty_future_anns:
        for frame in frames[2:]:
            frame["anns"] = {
                "gt_boxes": np.empty((0, 7), dtype=np.float32),
                "gt_names": [],
            }
    with open(logs_dir / f"{scene_name}.pkl", "wb") as handle:
        pickle.dump(frames, handle)

    for frame_idx in range(4):
        image = np.full((24, 32, 3), fill_value=frame_idx, dtype=np.uint8)
        Image.fromarray(image).save(camera_dir / f"{frame_idx:03d}.jpg")

    return logs_dir, blobs_dir


def _write_six_frame_scene_with_only_last_reduced_anns(root: Path) -> tuple[Path, Path]:
    logs_dir = root / "logs"
    blobs_dir = root / "sensor_blobs"
    scene_name = "scene_001"
    camera_dir = blobs_dir / scene_name / "CAM_F0"
    logs_dir.mkdir(parents=True)
    camera_dir.mkdir(parents=True)

    frames = [_make_frame(frame_idx) for frame_idx in range(6)]
    frames[4]["anns"] = {
        "gt_boxes": np.empty((0, 7), dtype=np.float32),
        "gt_names": [],
    }
    with open(logs_dir / f"{scene_name}.pkl", "wb") as handle:
        pickle.dump(frames, handle)
    for frame_idx in range(6):
        image = np.full((24, 32, 3), fill_value=frame_idx, dtype=np.uint8)
        Image.fromarray(image).save(camera_dir / f"{frame_idx:03d}.jpg")
    return logs_dir, blobs_dir


class TestNavSimImageRequirePolicy(unittest.TestCase):
    def _future_geometry_sample(self, *, include_empty_future_anns: bool) -> dict:
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        logs_dir, blobs_dir = _write_scene_with_future_agent_annotations(
            Path(tmp_dir.name),
            include_empty_future_anns=include_empty_future_anns,
        )
        dataset = NavSimWorldModelDataset(
            data_path=str(logs_dir),
            sensor_blobs_path=str(blobs_dir),
            camera_name="CAM_F0",
            frames_per_clip=4,
            fps=2,
            tubelet_size=1,
            index_cache=False,
            image_require_policy="all_frames",
            num_observed_frames=2,
            load_agent_annotations=True,
        )
        return dataset[0]

    def test_missing_future_anns_marks_agent_geometry_invalid(self):
        sample = self._future_geometry_sample(include_empty_future_anns=False)

        self.assertFalse(sample["future_agent_geometry_valid"])
        self.assertFalse(sample["agent_mask"][2:].any())

    def test_explicit_empty_future_anns_is_valid_no_agent_geometry(self):
        sample = self._future_geometry_sample(include_empty_future_anns=True)

        self.assertTrue(sample["future_agent_geometry_valid"])
        self.assertFalse(sample["agent_mask"][2:].any())

    def test_tubelet_reduction_checks_every_relevant_future_anns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir, blobs_dir = _write_six_frame_scene_with_only_last_reduced_anns(Path(tmp_dir))
            dataset = NavSimWorldModelDataset(
                data_path=str(logs_dir),
                sensor_blobs_path=str(blobs_dir),
                camera_name="CAM_F0",
                frames_per_clip=6,
                fps=2,
                tubelet_size=2,
                index_cache=False,
                image_require_policy="all_frames",
                num_observed_frames=2,
                load_agent_annotations=True,
            )

            self.assertFalse(dataset[0]["future_agent_geometry_valid"])

    def test_observed_only_keeps_window_when_future_images_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir, blobs_dir = _write_scene(Path(tmp_dir), num_frames=6, image_frames=range(3))

            dataset = NavSimWorldModelDataset(
                data_path=str(logs_dir),
                sensor_blobs_path=str(blobs_dir),
                camera_name="CAM_F0",
                frames_per_clip=6,
                fps=2,
                tubelet_size=1,
                index_cache=False,
                max_frame_gap=1,
                image_require_policy="observed_only",
                num_observed_frames=3,
                load_agent_annotations=False,
            )
            self.assertEqual(len(dataset), 1)
            sample = dataset[0]
            self.assertEqual(tuple(sample["buffer"].shape), (3, 3, 24, 32))
            self.assertEqual(tuple(sample["states"].shape), (6, 7))
            self.assertEqual(sample["sampled_frame_indices"].tolist(), [0, 1, 2, 3, 4, 5])
            self.assertEqual(sample["image_frame_indices"].tolist(), [0, 1, 2])

    def test_all_frames_rejects_window_when_future_images_are_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir, blobs_dir = _write_scene(Path(tmp_dir), num_frames=6, image_frames=range(3))

            with self.assertRaises(ValueError):
                NavSimWorldModelDataset(
                    data_path=str(logs_dir),
                    sensor_blobs_path=str(blobs_dir),
                    camera_name="CAM_F0",
                    frames_per_clip=6,
                    fps=2,
                    index_cache=False,
                    max_frame_gap=1,
                    image_require_policy="all_frames",
                    num_observed_frames=3,
                    load_agent_annotations=False,
                )

    def test_auto_policy_requires_all_images_for_predictor_training(self):
        from app.vjepa_cowa_world_model.training.config import (
            parse_training_config,
            resolve_navsim_image_require_policy,
        )

        config = parse_training_config(
            {
                "train": {"predictor_train": True},
                "planner": {"use_planner": False},
                "data": {"navsim": {"enabled": True, "image_require_policy": "auto"}},
            }
        )

        self.assertEqual(resolve_navsim_image_require_policy(config), "all_frames")

    def test_auto_policy_requires_all_images_for_joint_predictor_planner_training(self):
        from app.vjepa_cowa_world_model.training.config import (
            parse_training_config,
            resolve_navsim_image_require_policy,
        )

        config = parse_training_config(
            {
                "train": {"predictor_train": True},
                "planner": {"use_planner": True},
                "data": {"navsim": {"enabled": True, "image_require_policy": "auto"}},
            }
        )

        self.assertEqual(resolve_navsim_image_require_policy(config), "all_frames")

    def test_auto_policy_uses_observed_only_for_planner_only_training(self):
        from app.vjepa_cowa_world_model.training.config import (
            parse_training_config,
            resolve_navsim_image_require_policy,
        )

        config = parse_training_config(
            {
                "train": {"predictor_train": False},
                "planner": {"use_planner": True},
                "data": {"navsim": {"enabled": True, "image_require_policy": "auto"}},
            }
        )

        self.assertEqual(resolve_navsim_image_require_policy(config), "observed_only")

    def test_explicit_policy_overrides_auto_mode(self):
        from app.vjepa_cowa_world_model.training.config import (
            parse_training_config,
            resolve_navsim_image_require_policy,
        )

        config = parse_training_config(
            {
                "train": {"predictor_train": True},
                "planner": {"use_planner": True},
                "data": {"navsim": {"enabled": True, "image_require_policy": "observed_only"}},
            }
        )

        self.assertEqual(resolve_navsim_image_require_policy(config), "observed_only")

    def test_sample_and_collate_include_reduced_metadata_valid_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            logs_dir, blobs_dir = _write_scene_with_metadata_valid(Path(tmp_dir))

            raw_dataset = NavSimWorldModelDataset(
                data_path=str(logs_dir),
                sensor_blobs_path=str(blobs_dir),
                camera_name="CAM_F0",
                frames_per_clip=10,
                fps=2,
                tubelet_size=2,
                index_cache=False,
                max_frame_gap=1,
                image_require_policy="all_frames",
                num_observed_frames=4,
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
            self.assertEqual(sample["sampled_frame_indices"].tolist(), list(range(10)))
            self.assertEqual(sample["raw_metadata_valid_mask"].tolist(), [True, True, True, True] + [False] * 6)
            self.assertEqual(sample["metadata_valid_mask"].tolist(), [True, True, False, False, False])
            self.assertTrue(sample["observed_metadata_valid"])

            shifted_sample = dataset[1]
            self.assertEqual(shifted_sample["sampled_frame_indices"].tolist(), list(range(1, 11)))
            self.assertEqual(
                shifted_sample["raw_metadata_valid_mask"].tolist(), [True, True, True, False] + [False] * 6
            )
            self.assertFalse(shifted_sample["observed_metadata_valid"])

            batch = navsim_world_model_collate_fn([sample, shifted_sample])
            metadata = batch[-1]
            self.assertEqual(tuple(metadata["metadata_valid_mask"].shape), (2, 5))
            self.assertEqual(metadata["metadata_valid_mask"][0].tolist(), [True, True, False, False, False])
            self.assertEqual(tuple(metadata["raw_metadata_valid_mask"].shape), (2, 10))
            self.assertEqual(metadata["raw_metadata_valid_mask"][0].tolist(), [True, True, True, True] + [False] * 6)
            self.assertEqual(metadata["observed_metadata_valid_mask"].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
