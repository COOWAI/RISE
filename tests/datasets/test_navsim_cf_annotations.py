import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.vjepa_cowa_world_model.training import navsim_data
from app.vjepa_cowa_world_model.training.navsim_data import (
    NavSimWorldModelDataset,
    load_counterfactual_annotations,
    navsim_world_model_collate_fn,
)


def _write_annos(path: Path, entries) -> str:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def _entry(scene, accident, accident_type, distortion=False, **extra):
    annos = {
        "accident": accident,
        "accident_type": accident_type,
        "suggested_action": [],
        "reverse": False,
        "static": False,
        "run_red_light": False,
        "distortion": distortion,
    }
    annos.update(extra)
    return {"scene": scene, "annos": annos}


class TestLoadCounterfactualAnnotations(unittest.TestCase):
    def test_maps_annotation_scene_to_pkl_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_annos(
                Path(tmp) / "annos.json",
                [
                    _entry("scene-0001_CAM_F0_000010_000019_gen", True, "自车行为引起"),
                    _entry("scene-0002_CAM_F0_000000_000009_gen", False, "正常"),
                ],
            )

            annotations = load_counterfactual_annotations(path, camera_name="CAM_F0")

        self.assertEqual(
            sorted(annotations.keys()),
            ["scene-0001_cf_000010_000019", "scene-0002_cf_000000_000009"],
        )
        self.assertTrue(annotations["scene-0001_cf_000010_000019"]["accident"])
        self.assertEqual(annotations["scene-0001_cf_000010_000019"]["accident_type"], "自车行为引起")
        self.assertFalse(annotations["scene-0002_cf_000000_000009"]["accident"])

    def test_accepts_already_normalized_pseudo_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_annos(
                Path(tmp) / "annos.json",
                [_entry("scene-0001_cf_000010_000019", True, "自车行为引起")],
            )

            annotations = load_counterfactual_annotations(path, camera_name="CAM_F0")

        self.assertEqual(list(annotations), ["scene-0001_cf_000010_000019"])
        self.assertEqual(
            annotations["scene-0001_cf_000010_000019"]["scene"],
            "scene-0001_cf_000010_000019",
        )
        self.assertTrue(annotations["scene-0001_cf_000010_000019"]["accident"])

    def test_distorted_entry_is_normalized_and_kept_for_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 导出约定：失真条目其余字段为空串
            path = _write_annos(
                Path(tmp) / "annos.json",
                [_entry("scene-0003_CAM_F0_000010_000019_gen", "", "", distortion=True)],
            )

            annotations = load_counterfactual_annotations(path, camera_name="CAM_F0")

        record = annotations["scene-0003_cf_000010_000019"]
        self.assertTrue(record["distortion"])
        self.assertFalse(record["accident"])
        self.assertEqual(record["accident_type"], "")

    def test_loads_optional_trajectory_match_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_annos(
                Path(tmp) / "annos.json",
                [
                    _entry(
                        "scene-0003_CAM_F0_000010_000019_gen",
                        True,
                        "自车行为引起",
                        trajectory_match=True,
                    ),
                    _entry(
                        "scene-0004_CAM_F0_000010_000019_gen",
                        True,
                        "自车行为引起",
                        trajectory_match=None,
                    ),
                ],
            )

            annotations = load_counterfactual_annotations(path, camera_name="CAM_F0")

        self.assertIs(annotations["scene-0003_cf_000010_000019"]["trajectory_match"], True)
        self.assertIsNone(annotations["scene-0004_cf_000010_000019"]["trajectory_match"])

    def test_legacy_trajectory_match_value_is_ignored_when_filter_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_annos(
                Path(tmp) / "annos.json",
                [
                    _entry(
                        "scene-0003_CAM_F0_000010_000019_gen",
                        "",
                        "",
                        distortion=True,
                        trajectory_match="",
                    )
                ],
            )

            annotations = load_counterfactual_annotations(path, camera_name="CAM_F0")

        self.assertEqual(annotations["scene-0003_cf_000010_000019"]["trajectory_match"], "")

    def test_raises_on_tri_state_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            # distortion=false 时 accident 必须是 bool，空串即数据错位
            path = _write_annos(
                Path(tmp) / "annos.json",
                [_entry("scene-0004_CAM_F0_000010_000019_gen", "", "", distortion=False)],
            )

            with self.assertRaisesRegex(ValueError, "non-bool accident"):
                load_counterfactual_annotations(path, camera_name="CAM_F0")

    def test_raises_on_accident_with_empty_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_annos(
                Path(tmp) / "annos.json",
                [_entry("scene-0005_CAM_F0_000010_000019_gen", True, "")],
            )

            with self.assertRaisesRegex(ValueError, "empty accident_type"):
                load_counterfactual_annotations(path, camera_name="CAM_F0")

    def test_raises_on_unexpected_scene_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_annos(
                Path(tmp) / "annos.json",
                [_entry("scene-0006_CAM_L0_000010_000019_gen", False, "正常")],
            )

            with self.assertRaisesRegex(ValueError, "pattern"):
                load_counterfactual_annotations(path, camera_name="CAM_F0")

    def test_raises_on_duplicate_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            entry = _entry("scene-0007_CAM_F0_000010_000019_gen", False, "正常")
            path = _write_annos(Path(tmp) / "annos.json", [entry, entry])

            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_counterfactual_annotations(path, camera_name="CAM_F0")

    def test_raises_when_raw_and_normalized_scenes_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_annos(
                Path(tmp) / "annos.json",
                [
                    _entry("scene-0007_CAM_F0_000010_000019_gen", False, "正常"),
                    _entry("scene-0007_cf_000010_000019", False, "正常"),
                ],
            )

            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_counterfactual_annotations(path, camera_name="CAM_F0")

    def test_raises_on_unknown_hazard_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_annos(
                Path(tmp) / "annos.json",
                [_entry("scene-0008_CAM_F0_000010_000019_gen", True, "未知事故类型")],
            )

            with self.assertRaisesRegex(ValueError, "hazard type"):
                load_counterfactual_annotations(path, camera_name="CAM_F0")

    def test_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_counterfactual_annotations("/nonexistent/annos.json", camera_name="CAM_F0")


def _make_frame(scene_name: str, frame_idx: int) -> dict:
    return {
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


def _write_cf_scenes(root: Path, scene_names, num_frames: int = 4):
    logs_dir = root / "logs"
    blobs_dir = root / "sensor_blobs"
    logs_dir.mkdir(parents=True)
    for scene_name in scene_names:
        camera_dir = blobs_dir / scene_name / "CAM_F0"
        camera_dir.mkdir(parents=True)
        frames = [_make_frame(scene_name, frame_idx) for frame_idx in range(num_frames)]
        with open(logs_dir / f"{scene_name}.pkl", "wb") as handle:
            pickle.dump(frames, handle)
        for frame_idx in range(num_frames):
            image = np.full((24, 32, 3), fill_value=frame_idx, dtype=np.uint8)
            Image.fromarray(image).save(camera_dir / f"{frame_idx:03d}.jpg")
    return logs_dir, blobs_dir


def _anno_scene(stem: str) -> str:
    return stem.replace("_cf_", "_CAM_F0_") + "_gen"


class TestNavSimDatasetWithAnnotations(unittest.TestCase):
    _STEMS = [
        "scene-0001_cf_000000_000009",  # accident/自车行为引起
        "scene-0002_cf_000000_000009",  # normal
        "scene-0003_cf_000000_000009",  # distorted -> dropped
    ]

    def _write_fixture(self, root: Path) -> tuple:
        logs_dir, blobs_dir = _write_cf_scenes(root, self._STEMS)
        annos_path = _write_annos(
            root / "annos.json",
            [
                _entry(_anno_scene(self._STEMS[0]), True, "自车行为引起"),
                _entry(_anno_scene(self._STEMS[1]), False, "正常"),
                _entry(_anno_scene(self._STEMS[2]), "", "", distortion=True),
            ],
        )
        return logs_dir, blobs_dir, annos_path

    def _build(self, logs_dir, blobs_dir, **kwargs):
        return NavSimWorldModelDataset(
            data_path=str(logs_dir),
            sensor_blobs_path=str(blobs_dir),
            camera_name="CAM_F0",
            frames_per_clip=4,
            fps=2,
            tubelet_size=1,
            index_cache=False,
            max_frame_gap=1,
            load_agent_annotations=False,
            **kwargs,
        )

    def test_drops_distorted_scenes_and_stamps_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, annos_path = self._write_fixture(Path(tmp))

            dataset = self._build(logs_dir, blobs_dir, annotations_path=annos_path, annotations_drop_distorted=True)

            self.assertEqual(sorted(scene.scene_name for scene in dataset.scenes), self._STEMS[:2])
            items = {dataset[i]["scene_name"]: dataset[i] for i in range(len(dataset))}
            self.assertTrue(items[self._STEMS[0]]["is_accident"])
            self.assertEqual(items[self._STEMS[0]]["accident_type"], "自车行为引起")
            self.assertFalse(items[self._STEMS[1]]["is_accident"])
            self.assertEqual(items[self._STEMS[1]]["accident_type"], "正常")
            for item in items.values():
                self.assertEqual(item["agent_boxes"].shape, (4, 256, 7))
                self.assertEqual(item["agent_mask"].shape, (4, 256))
                self.assertFalse(item["agent_boxes"].any())
                self.assertFalse(item["agent_mask"].any())
                self.assertFalse(item["bev_segmentation"].any())
                self.assertFalse(item["geometry_present"])
                self.assertIsNone(item["raw_agent_count"])
                self.assertIsNone(item["agent_geometry_truncated"])
                self.assertIsNone(item["geometry_coordinate_frame"])

            tagged = navsim_data.RootTaggedDataset(
                [items[self._STEMS[0]], items[self._STEMS[1]]],
                domain="counterfactual",
                dataset_root_name="synthetic",
                dataset_root_index=1,
                future_agent_geometry_valid=False,
            )
            batch = navsim_world_model_collate_fn([tagged[0], tagged[1]])
            metadata = batch[-1]
            self.assertEqual(metadata["is_accident"].tolist(), [True, False])
            self.assertEqual(metadata["accident_type"], ["自车行为引起", "正常"])
            self.assertEqual(metadata["dataset_domain"], ["counterfactual", "counterfactual"])
            self.assertEqual(metadata["dataset_root_name"], ["synthetic", "synthetic"])
            self.assertEqual(metadata["dataset_root_index"].tolist(), [1, 1])
            self.assertEqual(metadata["base_scene_id"], ["scene-0001", "scene-0002"])
            self.assertEqual(metadata["cf_annotation_valid"].tolist(), [True, True])
            self.assertEqual(metadata["cf_is_hazard"].tolist(), [True, False])
            self.assertEqual(metadata["cf_hazard_type"], ["自车行为引起", ""])
            self.assertEqual(metadata["future_agent_geometry_valid"].tolist(), [False, False])

    def test_safe_only_filters_hazards_after_distortion_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, annos_path = self._write_fixture(Path(tmp))

            dataset = self._build(
                logs_dir,
                blobs_dir,
                annotations_path=annos_path,
                annotations_drop_distorted=True,
                annotation_selection="safe_only",
            )

            self.assertEqual([scene.scene_name for scene in dataset.scenes], [self._STEMS[1]])
            sample = dataset[0]
            self.assertTrue(sample["cf_annotation_valid"])
            self.assertFalse(sample["cf_is_hazard"])
            self.assertEqual(sample["cf_hazard_type"], "")

    def test_requires_trajectory_match_filters_false_and_null(self):
        stems = [
            "scene-0010_cf_000000_000009",
            "scene-0011_cf_000000_000009",
            "scene-0012_cf_000000_000009",
            "scene-0013_cf_000000_000009",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir, blobs_dir = _write_cf_scenes(root, stems)
            annos_path = _write_annos(
                root / "annos.json",
                [
                    _entry(_anno_scene(stems[0]), True, "自车行为引起", trajectory_match=True),
                    _entry(_anno_scene(stems[1]), False, "正常", trajectory_match=False),
                    _entry(_anno_scene(stems[2]), False, "正常", trajectory_match=None),
                    _entry(_anno_scene(stems[3]), None, None, distortion=True, trajectory_match=None),
                ],
            )

            dataset = self._build(
                logs_dir,
                blobs_dir,
                annotations_path=annos_path,
                annotations_drop_distorted=True,
                annotations_require_trajectory_match=True,
            )

            self.assertEqual([scene.scene_name for scene in dataset.scenes], [stems[0]])

    def test_trajectory_match_filter_composes_with_safe_only(self):
        stems = [
            "scene-0020_cf_000000_000009",
            "scene-0021_cf_000000_000009",
            "scene-0022_cf_000000_000009",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir, blobs_dir = _write_cf_scenes(root, stems)
            annos_path = _write_annos(
                root / "annos.json",
                [
                    _entry(_anno_scene(stems[0]), True, "自车行为引起", trajectory_match=True),
                    _entry(_anno_scene(stems[1]), False, "正常", trajectory_match=True),
                    _entry(_anno_scene(stems[2]), False, "正常", trajectory_match=False),
                ],
            )

            dataset = self._build(
                logs_dir,
                blobs_dir,
                annotations_path=annos_path,
                annotations_drop_distorted=True,
                annotations_require_trajectory_match=True,
                annotation_selection="safe_only",
            )

            self.assertEqual([scene.scene_name for scene in dataset.scenes], [stems[1]])

    def test_formal_allowlist_selection_keeps_exactly_two_hazard_types(self):
        stems = [
            "scene-0030_cf_000000_000009",
            "scene-0031_cf_000000_000009",
            "scene-0032_cf_000000_000009",
            "scene-0033_cf_000000_000009",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir, blobs_dir = _write_cf_scenes(root, stems)
            annos_path = _write_annos(
                root / "annos.json",
                [
                    _entry(_anno_scene(stems[0]), True, "自车行为引起", trajectory_match=True),
                    _entry(_anno_scene(stems[1]), True, "非自车行为引起", trajectory_match=True),
                    _entry(_anno_scene(stems[2]), True, "有事故但与自车无关", trajectory_match=True),
                    _entry(_anno_scene(stems[3]), False, "正常", trajectory_match=True),
                ],
            )

            dataset = self._build(
                logs_dir,
                blobs_dir,
                annotations_path=annos_path,
                annotations_drop_distorted=True,
                annotations_require_trajectory_match=True,
                annotations_accident_type_allowlist=["自车行为引起", "非自车行为引起"],
                annotation_selection="trajectory_match_and_accident_type_allowlist",
            )

            self.assertEqual([scene.scene_name for scene in dataset.scenes], stems[:2])
            self.assertEqual(
                [dataset[index]["cf_hazard_type"] for index in range(2)], ["自车行为引起", "非自车行为引起"]
            )
            self.assertTrue(all(dataset[index]["cf_is_hazard"] for index in range(2)))

    def test_formal_allowlist_selection_rejects_expected_scene_silently_dropped_by_index(self):
        stems = [
            "scene-0034_cf_000000_000009",
            "scene-0035_cf_000000_000009",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir, blobs_dir = _write_cf_scenes(root, stems)
            annos_path = _write_annos(
                root / "annos.json",
                [
                    _entry(_anno_scene(stems[0]), True, "自车行为引起", trajectory_match=True),
                    _entry(_anno_scene(stems[1]), True, "非自车行为引起", trajectory_match=True),
                ],
            )
            for image_path in (blobs_dir / stems[1] / "CAM_F0").iterdir():
                image_path.unlink()

            with self.assertRaisesRegex(
                ValueError,
                "Formal-v2 CF annotation/index cohort mismatch.*expected_count=2.*actual_count=1.*scene-0035",
            ):
                self._build(
                    logs_dir,
                    blobs_dir,
                    annotations_path=annos_path,
                    annotations_drop_distorted=True,
                    annotations_require_trajectory_match=True,
                    annotations_accident_type_allowlist=["自车行为引起", "非自车行为引起"],
                    annotation_selection="trajectory_match_and_accident_type_allowlist",
                )

    def test_formal_allowlist_selection_rejects_any_allowlist_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, annos_path = self._write_fixture(Path(tmp))

            with self.assertRaisesRegex(ValueError, "annotations_accident_type_allowlist"):
                self._build(
                    logs_dir,
                    blobs_dir,
                    annotations_path=annos_path,
                    annotations_drop_distorted=True,
                    annotations_require_trajectory_match=True,
                    annotations_accident_type_allowlist=["自车行为引起"],
                    annotation_selection="trajectory_match_and_accident_type_allowlist",
                )

    def test_formal_quality_sidecar_rejects_dataset_timeline_other_than_12_4_8_before_file_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir, blobs_dir = _write_cf_scenes(root, [self._STEMS[0]])
            annos_path = _write_annos(
                root / "annos.json",
                [_entry(_anno_scene(self._STEMS[0]), True, "自车行为引起", trajectory_match=True)],
            )

            with self.assertRaisesRegex(ValueError, "Formal-v2 CF quality dataset timeline.*num_total_frames=4"):
                self._build(
                    logs_dir,
                    blobs_dir,
                    annotations_path=annos_path,
                    annotations_drop_distorted=True,
                    annotations_require_trajectory_match=True,
                    annotations_accident_type_allowlist=["自车行为引起", "非自车行为引起"],
                    annotation_selection="trajectory_match_and_accident_type_allowlist",
                    trajectory_quality_path=str(root / "must-not-be-read.json"),
                )

    def test_trajectory_match_filter_requires_annotations_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, _ = self._write_fixture(Path(tmp))

            with self.assertRaisesRegex(ValueError, "annotations_require_trajectory_match.*annotations_path"):
                self._build(
                    logs_dir,
                    blobs_dir,
                    annotations_require_trajectory_match=True,
                )

    def test_safe_only_rejects_disabled_distortion_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, annos_path = self._write_fixture(Path(tmp))

            with self.assertRaisesRegex(ValueError, "safe_only.*annotations_drop_distorted"):
                self._build(
                    logs_dir,
                    blobs_dir,
                    annotations_path=annos_path,
                    annotations_drop_distorted=False,
                    annotation_selection="safe_only",
                )

    def test_keeps_distorted_scenes_when_drop_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, annos_path = self._write_fixture(Path(tmp))

            dataset = self._build(logs_dir, blobs_dir, annotations_path=annos_path, annotations_drop_distorted=False)

            self.assertEqual(len(dataset.scenes), 3)

    def test_raises_when_scene_missing_annotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs_dir, blobs_dir = _write_cf_scenes(root, self._STEMS[:2])
            annos_path = _write_annos(
                root / "annos.json",
                [_entry(_anno_scene(self._STEMS[0]), True, "自车行为引起")],
            )

            with self.assertRaisesRegex(ValueError, "no entry in annotations_path"):
                self._build(logs_dir, blobs_dir, annotations_path=annos_path, annotations_drop_distorted=True)

    def test_raises_when_drop_flag_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, annos_path = self._write_fixture(Path(tmp))

            with self.assertRaisesRegex(ValueError, "annotations_drop_distorted"):
                self._build(logs_dir, blobs_dir, annotations_path=annos_path)

    def test_raises_when_drop_flag_set_without_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, _ = self._write_fixture(Path(tmp))

            with self.assertRaisesRegex(ValueError, "annotations_path is not"):
                self._build(logs_dir, blobs_dir, annotations_drop_distorted=True)

    def test_items_default_to_non_accident_without_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir, blobs_dir, _ = self._write_fixture(Path(tmp))

            dataset = self._build(logs_dir, blobs_dir)
            sample = dataset[0]

            self.assertFalse(sample["is_accident"])
            self.assertEqual(sample["accident_type"], "")


if __name__ == "__main__":
    unittest.main()
