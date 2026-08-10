"""Tests for official NavSim token-anchored window indexing."""

import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from app.vjepa_cowa_world_model.training.navsim_data import NavSimWorldModelDataset, load_navsim_scene_filter


def _make_frame(scene_name: str, frame_idx: int, timestamp_us: int) -> dict:
    return {
        "token": f"{scene_name}-tok{frame_idx:03d}",
        "timestamp": timestamp_us,
        "cams": {
            "CAM_F0": {
                "data_path": f"{scene_name}/CAM_F0/{frame_idx:03d}.jpg",
                "cam_intrinsic": np.eye(3, dtype=np.float32),
                "sensor2ego_translation": np.zeros(3, dtype=np.float32),
                "sensor2ego_rotation": np.eye(3, dtype=np.float32),
            }
        },
        "ego2global_translation": np.array([float(frame_idx), 0.0, 0.0], dtype=np.float64),
        "ego2global_rotation": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "ego_dynamic_state": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "driving_command": np.array([1, 0, 0, 0], dtype=np.float32),
    }


def _write_scene(
    root: Path,
    *,
    scene_name: str = "scene_001",
    num_frames: int = 12,
    image_frames=None,
    timestamps=None,
) -> tuple:
    logs_dir = root / "logs"
    blobs_dir = root / "sensor_blobs"
    camera_dir = blobs_dir / scene_name / "CAM_F0"
    logs_dir.mkdir(parents=True, exist_ok=True)
    camera_dir.mkdir(parents=True, exist_ok=True)

    if image_frames is None:
        image_frames = range(num_frames)
    if timestamps is None:
        timestamps = [frame_idx * 500_000 for frame_idx in range(num_frames)]

    frames = [_make_frame(scene_name, frame_idx, timestamps[frame_idx]) for frame_idx in range(num_frames)]
    with open(logs_dir / f"{scene_name}.pkl", "wb") as handle:
        pickle.dump(frames, handle)

    for frame_idx in image_frames:
        image = np.full((24, 32, 3), fill_value=frame_idx, dtype=np.uint8)
        Image.fromarray(image).save(camera_dir / f"{frame_idx:03d}.jpg")

    return logs_dir, blobs_dir


def _write_filter_yaml(root: Path, log_names, tokens) -> str:
    path = root / "scene_filter.yaml"
    with open(path, "w") as handle:
        yaml.safe_dump({"log_names": list(log_names), "tokens": list(tokens)}, handle)
    return str(path)


def _make_dataset(logs_dir: Path, blobs_dir: Path, scene_filter_yaml: str) -> NavSimWorldModelDataset:
    return NavSimWorldModelDataset(
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
        scene_filter_yaml=scene_filter_yaml,
    )


class TestSceneFilterYamlParsing(unittest.TestCase):
    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_navsim_scene_filter("/nonexistent/scene_filter.yaml")

    def test_missing_tokens_section_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bad.yaml"
            with open(path, "w") as handle:
                yaml.safe_dump({"log_names": ["scene_001"]}, handle)
            with self.assertRaisesRegex(ValueError, "tokens"):
                load_navsim_scene_filter(str(path))

    def test_empty_log_names_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_filter_yaml(Path(tmp_dir), [], ["tok-a"])
            with self.assertRaisesRegex(ValueError, "log_names"):
                load_navsim_scene_filter(path)

    def test_duplicate_tokens_raise(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_filter_yaml(Path(tmp_dir), ["scene_001"], ["tok-a", "tok-a"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_navsim_scene_filter(path)

    def test_valid_yaml_parses(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _write_filter_yaml(Path(tmp_dir), ["scene_001"], ["tok-a", "tok-b"])
            log_names, tokens = load_navsim_scene_filter(path)
            self.assertEqual(log_names, {"scene_001"})
            self.assertEqual(tokens, ["tok-a", "tok-b"])


class TestTokenAnchoredWindows(unittest.TestCase):
    def test_token_anchors_window_with_current_frame_last_observed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            # num_observed_frames=3, sample_step=1 -> start_pos = token_idx - 2.
            yaml_path = _write_filter_yaml(root, ["scene_001"], ["scene_001-tok004"])

            dataset = _make_dataset(logs_dir, blobs_dir, yaml_path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.windows[0].start_pos, 2)
            self.assertEqual(
                dataset.token_reject_counts,
                {
                    "out_of_bounds_history": 0,
                    "out_of_bounds_future": 0,
                    "image_missing": 0,
                    "outside_tail": 0,
                    "time_gap": 0,
                },
            )
            sample = dataset[0]
            self.assertEqual(sample["sampled_frame_indices"].tolist(), [2, 3, 4, 5, 6, 7])
            self.assertEqual(sample["image_frame_indices"].tolist(), [2, 3, 4])
            self.assertEqual(sample["sample_token"], "scene_001-tok004")
            self.assertEqual(sample["stable_sample_id"], "navsim:default:token:scene_001-tok004")

    def test_out_of_bounds_tokens_are_counted_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            # tok001 -> start_pos=-1 (history); tok010 -> window end 13 >= 12 (future).
            yaml_path = _write_filter_yaml(
                root,
                ["scene_001"],
                ["scene_001-tok001", "scene_001-tok004", "scene_001-tok010"],
            )

            dataset = _make_dataset(logs_dir, blobs_dir, yaml_path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.token_reject_counts["out_of_bounds_history"], 1)
            self.assertEqual(dataset.token_reject_counts["out_of_bounds_future"], 1)

    def test_unknown_token_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            yaml_path = _write_filter_yaml(root, ["scene_001"], ["scene_001-missing"])

            with self.assertRaisesRegex(ValueError, "not found in any kept log"):
                _make_dataset(logs_dir, blobs_dir, yaml_path)

    def test_log_name_without_pkl_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            yaml_path = _write_filter_yaml(root, ["scene_001", "scene_002"], ["scene_001-tok004"])

            with self.assertRaisesRegex(ValueError, "no pkl"):
                _make_dataset(logs_dir, blobs_dir, yaml_path)

    def test_log_dropped_for_missing_camera_dir_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            # scene_002 pkl exists but has no camera blob dir -> dropped at
            # scene-index build, which token mode must surface loudly.
            frames = [_make_frame("scene_002", idx, idx * 500_000) for idx in range(12)]
            with open(logs_dir / "scene_002.pkl", "wb") as handle:
                pickle.dump(frames, handle)
            yaml_path = _write_filter_yaml(root, ["scene_001", "scene_002"], ["scene_001-tok004", "scene_002-tok004"])

            with self.assertRaisesRegex(ValueError, "dropped while building the scene index"):
                _make_dataset(logs_dir, blobs_dir, yaml_path)

    def test_missing_observed_image_is_counted_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            # Frame 3 has no image: tok004 needs observed images [2, 3, 4].
            logs_dir, blobs_dir = _write_scene(root, image_frames=[i for i in range(12) if i != 3])
            yaml_path = _write_filter_yaml(root, ["scene_001"], ["scene_001-tok004", "scene_001-tok007"])

            dataset = _make_dataset(logs_dir, blobs_dir, yaml_path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.windows[0].start_pos, 5)
            self.assertEqual(dataset.token_reject_counts["image_missing"], 1)

    def test_time_gap_window_is_counted_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            # 2s recording gap between frames 6 and 7: tok004 spans [2, 7] and
            # crosses it; tok011 spans [9, 14] and stays clear.
            timestamps = [idx * 500_000 for idx in range(16)]
            for idx in range(7, 16):
                timestamps[idx] += 2_000_000
            logs_dir, blobs_dir = _write_scene(root, num_frames=16, timestamps=timestamps)
            yaml_path = _write_filter_yaml(root, ["scene_001"], ["scene_001-tok004", "scene_001-tok011"])

            dataset = _make_dataset(logs_dir, blobs_dir, yaml_path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.windows[0].start_pos, 9)
            self.assertEqual(dataset.token_reject_counts["time_gap"], 1)

    def test_zero_windows_raises(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            yaml_path = _write_filter_yaml(root, ["scene_001"], ["scene_001-tok000"])

            with self.assertRaisesRegex(ValueError, "No valid sliding windows"):
                _make_dataset(logs_dir, blobs_dir, yaml_path)

    def test_tokenless_pkl_raises_in_token_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            frames = pickle.load(open(logs_dir / "scene_001.pkl", "rb"))
            for frame in frames:
                del frame["token"]
            with open(logs_dir / "scene_001.pkl", "wb") as handle:
                pickle.dump(frames, handle)
            yaml_path = _write_filter_yaml(root, ["scene_001"], ["scene_001-tok004"])

            with self.assertRaisesRegex(ValueError, "no per-frame 'token'"):
                _make_dataset(logs_dir, blobs_dir, yaml_path)

    def test_max_scenes_conflicts_with_token_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            yaml_path = _write_filter_yaml(root, ["scene_001"], ["scene_001-tok004"])

            with self.assertRaisesRegex(ValueError, "max_scenes"):
                NavSimWorldModelDataset(
                    data_path=str(logs_dir),
                    sensor_blobs_path=str(blobs_dir),
                    camera_name="CAM_F0",
                    frames_per_clip=6,
                    fps=2,
                    tubelet_size=1,
                    index_cache=False,
                    max_scenes=1,
                    image_require_policy="observed_only",
                    num_observed_frames=3,
                    load_agent_annotations=False,
                    scene_filter_yaml=yaml_path,
                )

    def test_index_cache_roundtrip_in_token_mode(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)
            yaml_path = _write_filter_yaml(root, ["scene_001"], ["scene_001-tok004"])

            def _build() -> NavSimWorldModelDataset:
                return NavSimWorldModelDataset(
                    data_path=str(logs_dir),
                    sensor_blobs_path=str(blobs_dir),
                    camera_name="CAM_F0",
                    frames_per_clip=6,
                    fps=2,
                    tubelet_size=1,
                    index_cache=True,
                    max_frame_gap=1,
                    image_require_policy="observed_only",
                    num_observed_frames=3,
                    load_agent_annotations=False,
                    scene_filter_yaml=yaml_path,
                )

            first = _build()
            cache_files = list(logs_dir.glob(".navsim_scene_index_cache_*.pkl"))
            self.assertEqual(len(cache_files), 1)

            # Second build loads the cached scene index (frame_tokens included)
            # and must produce identical token-anchored windows.
            second = _build()
            self.assertEqual(
                [(window.scene_idx, window.start_pos) for window in second.windows],
                [(window.scene_idx, window.start_pos) for window in first.windows],
            )
            self.assertEqual(second.scenes[0].frame_tokens, first.scenes[0].frame_tokens)

    def test_stride_mode_unchanged_without_yaml(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            logs_dir, blobs_dir = _write_scene(root)

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

            self.assertFalse(dataset.token_mode)
            # 12 frames, frames_per_clip=6, stride=1 -> starts 0..6.
            self.assertEqual([window.start_pos for window in dataset.windows], list(range(7)))
            self.assertFalse(hasattr(dataset, "token_reject_counts"))


class TestConfigLevelValidation(unittest.TestCase):
    def _base_args(self, navsim: dict) -> dict:
        navsim = {"enabled": True, **navsim}
        return {
            "train": {"predictor_train": False},
            "planner": {"use_planner": True},
            "data": {"navsim": navsim},
        }

    def test_window_stride_conflict_raises(self):
        from app.vjepa_cowa_world_model.training.config import parse_training_config

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = _write_filter_yaml(Path(tmp_dir), ["scene_001"], ["tok-a"])
            with self.assertRaisesRegex(ValueError, "window_stride"):
                parse_training_config(self._base_args({"scene_filter_yaml": yaml_path, "window_stride": 4}))

    def test_max_scenes_conflict_raises(self):
        from app.vjepa_cowa_world_model.training.config import parse_training_config

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = _write_filter_yaml(Path(tmp_dir), ["scene_001"], ["tok-a"])
            with self.assertRaisesRegex(ValueError, "max_scenes"):
                parse_training_config(self._base_args({"scene_filter_yaml": yaml_path, "max_scenes": 8}))

    def test_nonexistent_yaml_raises(self):
        from app.vjepa_cowa_world_model.training.config import parse_training_config

        with self.assertRaisesRegex(ValueError, "does not exist"):
            parse_training_config(self._base_args({"scene_filter_yaml": "/nonexistent.yaml"}))

    def test_valid_token_mode_config_parses(self):
        from app.vjepa_cowa_world_model.training.config import parse_training_config

        with tempfile.TemporaryDirectory() as tmp_dir:
            train_yaml = _write_filter_yaml(Path(tmp_dir), ["scene_001"], ["tok-a"])
            config = parse_training_config(
                self._base_args({"scene_filter_yaml": train_yaml, "val_scene_filter_yaml": train_yaml})
            )
            self.assertEqual(config.data.navsim.scene_filter_yaml, train_yaml)
            self.assertEqual(config.data.navsim.val_scene_filter_yaml, train_yaml)

    def test_explicit_stride_one_allowed_with_token_mode(self):
        from app.vjepa_cowa_world_model.training.config import parse_training_config

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = _write_filter_yaml(Path(tmp_dir), ["scene_001"], ["tok-a"])
            config = parse_training_config(self._base_args({"scene_filter_yaml": yaml_path, "window_stride": 1}))
            self.assertEqual(config.data.navsim.scene_filter_yaml, yaml_path)


if __name__ == "__main__":
    unittest.main()
