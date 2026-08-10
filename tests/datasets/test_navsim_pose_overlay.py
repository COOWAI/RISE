import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from app.vjepa_cowa_world_model.training.navsim_data import NavSimWorldModelDataset, SceneRecord, WindowRecord
from app.vjepa_cowa_world_model.training.pose_overlay import PoseOverlayReader


class TestNavSimPoseOverlay(unittest.TestCase):
    def _write_overlay(self, root: Path) -> None:
        scene_dir = root / "scene_001"
        scene_dir.mkdir(parents=True)
        np.savez(
            scene_dir / "pred_pose.npz",
            frame_indices=np.asarray([0, 1, 2], dtype=np.int64),
            translation=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [2.0, 0.0, 5.0],
                    [4.0, 0.0, 9.0],
                ],
                dtype=np.float32,
            ),
        )

    def _write_minimal_navsim_scene(self, root: Path) -> tuple[Path, Path]:
        logs_dir = root / "logs"
        blobs_dir = root / "sensor_blobs"
        camera_dir = blobs_dir / "scene_001" / "CAM_F0"
        logs_dir.mkdir(parents=True)
        camera_dir.mkdir(parents=True)

        frames = []
        for frame_idx in range(3):
            frames.append(
                {
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
                }
            )
            Image.fromarray(np.full((8, 12, 3), frame_idx, dtype=np.uint8)).save(camera_dir / f"{frame_idx:03d}.jpg")

        with open(logs_dir / "scene_001.pkl", "wb") as handle:
            pickle.dump(frames, handle)
        return logs_dir, blobs_dir

    def test_opencv_first_frame_pose_converts_to_ego_states_actions_and_dynamics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_overlay(root)
            reader = PoseOverlayReader(root, coord_frame="opencv_first_frame", required=True)

            result = reader.build_states_actions_ego_dynamics("scene_001", [0, 1, 2], action_dim=3, dt=0.5)

            np.testing.assert_allclose(result.states[:, 0], [0.0, 5.0, 9.0], atol=1e-5)
            np.testing.assert_allclose(result.states[:, 1], [-0.0, -2.0, -4.0], atol=1e-5)
            np.testing.assert_allclose(result.states[:, 5], [0.0, np.arctan2(-2.0, 5.0), np.arctan2(-2.0, 4.0)])
            self.assertEqual(result.actions.shape, (2, 3))
            self.assertEqual(result.ego_dynamics.shape, (3, 4))
            np.testing.assert_allclose(result.ego_dynamics[1, :2], [10.0, -4.0], atol=1e-5)

    def test_missing_scene_raises_when_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            reader = PoseOverlayReader(tmp, coord_frame="opencv_first_frame", required=True)

            with self.assertRaisesRegex(FileNotFoundError, "missing NavSim pose overlay scene"):
                reader.get_pose_sequence("missing_scene", [0])

    def test_missing_frame_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_overlay(root)
            reader = PoseOverlayReader(root, coord_frame="opencv_first_frame", required=True)

            with self.assertRaisesRegex(KeyError, "missing frame"):
                reader.get_pose_sequence("scene_001", [0, 3])

    def test_bad_shape_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_dir = root / "scene_001"
            scene_dir.mkdir(parents=True)
            np.savez(
                scene_dir / "pred_pose.npz",
                frame_indices=np.asarray([0, 1], dtype=np.int64),
                translation=np.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=np.float32),
            )
            reader = PoseOverlayReader(root, coord_frame="opencv_first_frame", required=True)

            with self.assertRaisesRegex(ValueError, "translation.*\\[N, 3\\]"):
                reader.get_pose_sequence("scene_001", [0, 1])

    def test_nonfinite_pose_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_dir = root / "scene_001"
            scene_dir.mkdir(parents=True)
            np.savez(
                scene_dir / "pred_pose.npz",
                frame_indices=np.asarray([0], dtype=np.int64),
                translation=np.asarray([[0.0, np.nan, 0.0]], dtype=np.float32),
            )
            reader = PoseOverlayReader(root, coord_frame="opencv_first_frame", required=True)

            with self.assertRaisesRegex(ValueError, "non-finite"):
                reader.get_pose_sequence("scene_001", [0])

    def test_flat_counterfactual_txt_pose_resolves_and_samples_training_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for row_idx in range(100):
                matrix = np.eye(3, 4, dtype=np.float64)
                matrix[0, 3] = float(row_idx)
                matrix[2, 3] = float(row_idx * 2)
                rows.append(matrix.reshape(-1))
            np.savetxt(root / "scene-0001_CAM_F0_000010_000019_gen.txt", np.stack(rows, axis=0))
            reader = PoseOverlayReader(root, coord_frame="opencv_first_frame", required=True)

            pose = reader.get_pose_sequence("scene-0001_cf_000010_000019", [0, 7, 16])

            self.assertEqual(pose.frame_indices.tolist(), [0, 7, 16])
            np.testing.assert_allclose(pose.translation[:, 0], [15.0, 50.0, 95.0])
            np.testing.assert_allclose(pose.translation[:, 2], [30.0, 100.0, 190.0])

    def test_flat_counterfactual_txt_pose_can_sample_from_video_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for row_idx in range(60):
                matrix = np.eye(3, 4, dtype=np.float64)
                matrix[0, 3] = float(row_idx)
                matrix[2, 3] = float(row_idx * 2)
                rows.append(matrix.reshape(-1))
            np.savetxt(root / "scene-0001_CAM_F0_000010_000021_gen.txt", np.stack(rows, axis=0))
            reader = PoseOverlayReader(
                root,
                coord_frame="opencv_first_frame",
                required=True,
                txt_start_seconds=0.0,
            )

            pose = reader.get_pose_sequence("scene-0001_cf_000010_000021", [0, 11])

            self.assertEqual(pose.frame_indices.tolist(), [0, 11])
            np.testing.assert_allclose(pose.translation[:, 0], [0.0, 55.0])
            np.testing.assert_allclose(pose.translation[:, 2], [0.0, 110.0])

    def test_dataset_reads_pose_overlay_before_image_clip_allocation(self):
        order = []

        class FakeOverlayReader:
            def build_states_actions_ego_dynamics(self, scene_name, frame_indices, *, action_dim, dt):
                order.append("pose_overlay")
                return SimpleNamespace(
                    states=np.zeros((3, 7), dtype=np.float32),
                    actions=np.zeros((2, action_dim), dtype=np.float32),
                    ego_dynamics=np.zeros((3, 4), dtype=np.float32),
                )

        dataset = NavSimWorldModelDataset.__new__(NavSimWorldModelDataset)
        dataset.windows = [WindowRecord(scene_idx=0, start_pos=0)]
        dataset.scenes = [
            SceneRecord(
                scene_name="scene_001",
                pkl_path="/tmp/scene_001.pkl",
                camera_dir="/tmp/CAM_F0",
                valid_frame_indices=list(range(6)),
            )
        ]
        dataset.tubelet_size = 2
        dataset.transform = object()
        dataset.proposal_transform = None
        dataset.pose_overlay_reader = FakeOverlayReader()
        dataset.action_dim = 3
        dataset.num_observed_frames = 4
        dataset.load_agent_annotations = False
        dataset.max_agents = 8
        dataset.camera_names = ["CAM_F0"]
        # ``__new__`` bypasses the dataset constructor, so keep this focused
        # fixture aligned with the metadata/deterministic-validation fields that
        # production construction initializes.
        dataset._scene_annotations = None
        dataset._scene_trajectory_quality = None
        dataset.token_mode = False
        dataset.window_start_policy = "sliding"
        dataset.is_validation = True

        frames = [{"metadata_valid": True} for _ in range(6)]
        dataset._load_scene_frames = lambda pkl_path: frames
        dataset._sampled_frame_indices = lambda start_pos: list(range(6))
        dataset._required_image_frame_indices = lambda sampled: list(sampled)
        dataset._reduced_frame_dt = lambda reduced: 1.0
        dataset._infer_output_hw = lambda buffer: (4, 8)
        dataset._build_metadata_valid_mask = lambda frames, indices: np.ones(len(indices), dtype=np.bool_)
        dataset._build_driving_commands = lambda frames, indices: np.zeros((len(indices), 4), dtype=np.float32)
        dataset._build_camera_metadata = lambda frames, indices, **kwargs: (
            np.zeros((len(indices), 3, 3), dtype=np.float32),
            np.zeros((len(indices), 4, 4), dtype=np.float32),
        )

        def load_clip_images(scene, frames, sampled_frame_indices):
            order.append("load_images")
            return np.zeros((6, 4, 8, 3), dtype=np.uint8), np.zeros((6, 2), dtype=np.int64)

        def apply_clip_transform(buffer, transform):
            order.append("transform")
            return torch.zeros((3, 6, 4, 8), dtype=torch.float32)

        dataset._load_clip_images = load_clip_images
        dataset._apply_clip_transform = apply_clip_transform

        sample = dataset[0]

        self.assertEqual(tuple(sample["states"].shape), (3, 7))
        self.assertLess(order.index("pose_overlay"), order.index("load_images"))
        self.assertLess(order.index("pose_overlay"), order.index("transform"))

    def test_dataset_preloads_pose_overlay_tables_during_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir, blobs_dir = self._write_minimal_navsim_scene(root)
            overlay_root = root / "pose_overlay"
            self._write_overlay(overlay_root)

            dataset = NavSimWorldModelDataset(
                data_path=str(logs_dir),
                sensor_blobs_path=str(blobs_dir),
                camera_name="CAM_F0",
                frames_per_clip=3,
                fps=2,
                tubelet_size=1,
                index_cache=False,
                max_frame_gap=1,
                image_require_policy="all_frames",
                num_observed_frames=2,
                pose_overlay_path=str(overlay_root),
                pose_overlay_required=True,
                load_agent_annotations=False,
            )

            self.assertEqual(len(dataset), 1)
            self.assertIn("scene_001", dataset.pose_overlay_reader._cache)


if __name__ == "__main__":
    unittest.main()
