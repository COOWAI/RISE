import numpy as np
import pytest
from torch.utils.data import Dataset

from app.vjepa_cowa_world_model.training import navsim_data
from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import _apply_effective_runtime_root


class _TinyDataset(Dataset):
    def __init__(self, name, length):
        self.name = name
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        is_counterfactual = self.name == "cf"
        scene_name = "scene-0001_cf_000010_000019" if is_counterfactual else "scene-0001"
        return {
            "name": self.name,
            "index": int(index),
            "scene_name": scene_name,
            "cf_annotation_valid": is_counterfactual,
            "cf_is_hazard": False,
            "cf_hazard_type": "",
            "future_agent_geometry_valid": not is_counterfactual,
            "agent_boxes": np.zeros((1, 256, 7), dtype=np.float32),
            "agent_mask": np.zeros((1, 256), dtype=np.bool_),
            "bev_segmentation": np.zeros((1, 200, 200), dtype=np.uint8),
            "raw_agent_count": None if is_counterfactual else np.zeros(1, dtype=np.int64),
        }


def _tagged_tiny_roots():
    real = navsim_data.RootTaggedDataset(
        _TinyDataset("real", 2),
        domain="real",
        dataset_root_name="real",
        dataset_root_index=0,
        future_agent_geometry_valid=True,
    )
    counterfactual = navsim_data.RootTaggedDataset(
        _TinyDataset("cf", 3),
        domain="counterfactual",
        dataset_root_name="counterfactual",
        dataset_root_index=1,
        future_agent_geometry_valid=False,
        load_agent_annotations=False,
    )
    return real, counterfactual


def test_balanced_root_concat_cycles_each_root_equally():
    dataset = navsim_data.BalancedRootConcatDataset(_tagged_tiny_roots())

    assert len(dataset) == 6
    samples = [dataset[i] for i in range(len(dataset))]

    assert [sample["name"] for sample in samples] == ["real", "cf", "real", "cf", "real", "cf"]
    assert [sample["index"] for sample in samples] == [0, 0, 1, 1, 0, 2]
    assert [sample["dataset_root_name"] for sample in samples] == [
        "real",
        "counterfactual",
        "real",
        "counterfactual",
        "real",
        "counterfactual",
    ]


def test_balanced_root_concat_supports_root_repeats_for_ratios():
    dataset = navsim_data.BalancedRootConcatDataset(
        _tagged_tiny_roots(),
        root_repeats=[3, 1],
    )

    assert len(dataset) == 12
    samples = [dataset[i] for i in range(8)]

    assert [sample["name"] for sample in samples] == ["real", "real", "real", "cf", "real", "real", "real", "cf"]
    assert [sample["index"] for sample in samples] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [sample["dataset_root_name"] for sample in samples] == [
        "real",
        "real",
        "real",
        "counterfactual",
        "real",
        "real",
        "real",
        "counterfactual",
    ]


def test_parse_training_config_reads_navsim_mixed_train_roots():
    cfg = parse_training_config(
        {
            "counterfactual_supervision": {"enabled": True},
            "data": {
                "navsim": {
                    "enabled": True,
                    "data_path": "/real/train",
                    "sensor_blobs_path": "/real/blobs",
                    "counterfactual_tail_seconds": 5.0,
                    "train_roots": [
                        {
                            "name": "real",
                            "domain": "real",
                            "data_path": "/real/train",
                            "sensor_blobs_path": "/real/blobs",
                        },
                        {
                            "name": "cf",
                            "domain": "counterfactual",
                            "data_path": "/cf/train",
                            "sensor_blobs_path": "/cf/blobs",
                            "tail_seconds": 4.0,
                            "pose_overlay_txt_start_seconds": 0.0,
                            "annotations_path": "/cf/annotations.json",
                            "annotations_drop_distorted": True,
                            "load_agent_annotations": False,
                        },
                    ],
                    "balance_train_roots": True,
                }
            },
        }
    )

    assert cfg.data.navsim.balance_train_roots is True
    assert cfg.data.navsim.counterfactual_tail_seconds == 5.0
    assert cfg.data.navsim.train_roots == [
        {
            "name": "real",
            "domain": "real",
            "data_path": "/real/train",
            "sensor_blobs_path": "/real/blobs",
            "annotation_selection": "all_valid",
        },
        {
            "name": "cf",
            "domain": "counterfactual",
            "data_path": "/cf/train",
            "sensor_blobs_path": "/cf/blobs",
            "tail_seconds": 4.0,
            "pose_overlay_txt_start_seconds": 0.0,
            "annotations_path": "/cf/annotations.json",
            "annotations_drop_distorted": True,
            "load_agent_annotations": False,
            "annotation_selection": "all_valid",
        },
    ]


def test_init_navsim_data_builds_balanced_mixed_roots(monkeypatch):
    created_roots = []

    class FakeNavSimWorldModelDataset(Dataset):
        def __init__(self, **kwargs):
            created_roots.append(kwargs)
            self.data_path = kwargs["data_path"]
            self.length = 2 if "real" in self.data_path else 3
            self.max_agents = kwargs["max_agents"]
            self.load_agent_annotations = kwargs["load_agent_annotations"]

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            is_counterfactual = "cf" in self.data_path
            return {
                "data_path": self.data_path,
                "index": int(index),
                "scene_name": ("scene-0001_cf_000010_000019" if is_counterfactual else "scene-0001"),
                "cf_annotation_valid": is_counterfactual,
                "cf_is_hazard": False,
                "cf_hazard_type": "",
                "future_agent_geometry_valid": not is_counterfactual,
                "agent_boxes": np.zeros((10, self.max_agents, 7), dtype=np.float32),
                "agent_mask": np.zeros((10, self.max_agents), dtype=np.bool_),
                "bev_segmentation": np.zeros((10, 200, 200), dtype=np.uint8),
                "raw_agent_count": None if is_counterfactual else np.zeros(10, dtype=np.int64),
            }

    monkeypatch.setattr(navsim_data, "NavSimWorldModelDataset", FakeNavSimWorldModelDataset)

    loader, sampler = navsim_data.init_navsim_data(
        data_path="",
        sensor_blobs_path="",
        batch_size=2,
        frames_per_clip=10,
        fps=2,
        tubelet_size=1,
        transform=None,
        num_workers=0,
        persistent_workers=False,
        rank=0,
        world_size=1,
        image_require_policy="all_frames",
        num_observed_frames=4,
        dataset_roots=[
            {
                "name": "real",
                "domain": "real",
                "data_path": "/real/train",
                "sensor_blobs_path": "/real/blobs",
                "index_cache": True,
                "repeat": 2,
            },
            {
                "name": "counterfactual",
                "domain": "counterfactual",
                "data_path": "/cf/train",
                "sensor_blobs_path": "/cf/blobs",
                "index_cache": False,
                "annotations_path": "/cf/annotations.json",
                "annotations_drop_distorted": True,
                "annotations_require_trajectory_match": True,
                "trajectory_quality_path": "/cf/trajectory_quality.json",
                "load_agent_annotations": False,
                "repeat": 1,
            },
        ],
        counterfactual_tail_seconds=5.0,
        balance_dataset_roots=True,
        counterfactual_supervision_v2=True,
    )

    assert isinstance(loader.dataset, navsim_data.BalancedRootConcatDataset)
    assert len(loader.dataset) == 9
    assert sampler.dataset is loader.dataset
    assert [root["data_path"] for root in created_roots] == ["/real/train", "/cf/train"]
    assert created_roots[1]["annotations_require_trajectory_match"] is True
    assert [root["index_cache"] for root in created_roots] == [True, False]
    assert [root["tail_seconds"] for root in created_roots] == [None, 5.0]
    assert [root["trajectory_quality_path"] for root in created_roots] == [None, "/cf/trajectory_quality.json"]
    assert [loader.dataset[index]["dataset_domain"] for index in (0, 2)] == ["real", "counterfactual"]
    assert [loader.dataset[index]["dataset_root_name"] for index in (0, 2)] == ["real", "counterfactual"]


def test_mixed_roots_do_not_inherit_global_pose_overlay_requirement(monkeypatch):
    created_roots = []

    class FakeNavSimWorldModelDataset(Dataset):
        def __init__(self, **kwargs):
            created_roots.append(kwargs)
            self.length = 1
            self.max_agents = kwargs["max_agents"]
            self.load_agent_annotations = kwargs["load_agent_annotations"]

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return {
                "index": int(index),
                "scene_name": "scene-0001",
                "cf_annotation_valid": False,
                "cf_is_hazard": False,
                "cf_hazard_type": "",
                "future_agent_geometry_valid": self.load_agent_annotations,
                "agent_boxes": np.zeros((10, self.max_agents, 7), dtype=np.float32),
                "agent_mask": np.zeros((10, self.max_agents), dtype=np.bool_),
                "bev_segmentation": np.zeros((10, 200, 200), dtype=np.uint8),
                "raw_agent_count": np.zeros(10, dtype=np.int64),
            }

    monkeypatch.setattr(navsim_data, "NavSimWorldModelDataset", FakeNavSimWorldModelDataset)

    navsim_data.init_navsim_data(
        data_path="",
        sensor_blobs_path="",
        batch_size=2,
        frames_per_clip=10,
        fps=2,
        tubelet_size=1,
        transform=None,
        num_workers=0,
        persistent_workers=False,
        rank=0,
        world_size=1,
        image_require_policy="all_frames",
        num_observed_frames=4,
        pose_overlay_required=True,
        dataset_roots=[
            {
                "name": "real",
                "domain": "real",
                "data_path": "/real/train",
                "sensor_blobs_path": "/real/blobs",
            },
            {
                "name": "counterfactual",
                "domain": "counterfactual",
                "data_path": "/cf/train",
                "sensor_blobs_path": "/cf/blobs",
                "pose_overlay_path": "/cf/poses",
                "pose_overlay_txt_start_seconds": 0.0,
                "pose_overlay_required": True,
                "annotations_path": "/cf/annotations.json",
                "annotations_drop_distorted": True,
                "trajectory_quality_path": "/cf/trajectory_quality.json",
            },
        ],
        counterfactual_supervision_v2=True,
    )

    assert [root["pose_overlay_path"] for root in created_roots] == [None, "/cf/poses"]
    assert [root["pose_overlay_txt_start_seconds"] for root in created_roots] == [1.5, 0.0]
    assert [root["pose_overlay_required"] for root in created_roots] == [False, True]


def _formal_real_runtime_root():
    source_root = {
        "name": "navsim_real_train",
        "domain": "real",
        "data_path": "/real/train",
        "sensor_blobs_path": "/real/blobs",
        "scene_filter_yaml": "/real/navtrain.yaml",
    }
    return _apply_effective_runtime_root(source_root, source_root=source_root)


def _install_capture_dataset(monkeypatch):
    created_roots = []

    class FakeNavSimWorldModelDataset(Dataset):
        def __init__(self, **kwargs):
            created_roots.append(kwargs)
            self.max_agents = kwargs["max_agents"]
            self.load_agent_annotations = kwargs["load_agent_annotations"]

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {
                "index": int(index),
                "scene_name": "scene-0001",
                "cf_annotation_valid": False,
                "cf_is_hazard": False,
                "cf_hazard_type": "",
                "future_agent_geometry_valid": True,
                "agent_boxes": np.zeros((12, self.max_agents, 7), dtype=np.float32),
                "agent_mask": np.zeros((12, self.max_agents), dtype=np.bool_),
                "bev_segmentation": np.zeros((12, 200, 200), dtype=np.uint8),
                "raw_agent_count": np.zeros(12, dtype=np.int64),
            }

    monkeypatch.setattr(navsim_data, "NavSimWorldModelDataset", FakeNavSimWorldModelDataset)
    return created_roots


def test_formal_root_drives_the_exact_dataset_timeline_and_sampling_contract(monkeypatch):
    created_roots = _install_capture_dataset(monkeypatch)

    navsim_data.init_navsim_data(
        data_path="",
        sensor_blobs_path="",
        batch_size=1,
        frames_per_clip=12,
        fps=2,
        base_fps=2,
        tubelet_size=1,
        num_workers=0,
        persistent_workers=False,
        num_observed_frames=4,
        dataset_roots=[_formal_real_runtime_root()],
    )

    assert len(created_roots) == 1
    runtime = created_roots[0]
    assert runtime["camera_name"] == "CAM_F0"
    assert runtime["camera_names"] == ["CAM_F0"]
    assert runtime["frames_per_clip"] == 12
    assert runtime["num_observed_frames"] == 4
    assert runtime["fps"] == 2
    assert runtime["base_fps"] == 2
    assert runtime["max_frame_gap"] == 1
    assert runtime["image_require_policy"] == "observed_only"
    assert runtime["max_agents"] == 1024
    assert runtime["max_scenes"] is None
    assert runtime["window_stride"] == 4
    assert runtime["load_agent_annotations"] is True
    assert runtime["window_start_policy"] == "sliding"
    assert runtime["timestamp_policy"] == "eligible_window_boundary_v1"


@pytest.mark.parametrize(
    ("argument", "value", "contract_field"),
    [
        ("frames_per_clip", 10, "num_target_frames"),
        ("num_observed_frames", 5, "num_observed_frames"),
        ("fps", 1, "fps"),
        ("base_fps", 4, "base_fps"),
    ],
)
def test_formal_root_rejects_global_timeline_drift(monkeypatch, argument, value, contract_field):
    _install_capture_dataset(monkeypatch)
    arguments = {
        "data_path": "",
        "sensor_blobs_path": "",
        "batch_size": 1,
        "frames_per_clip": 12,
        "fps": 2,
        "base_fps": 2,
        "tubelet_size": 1,
        "num_workers": 0,
        "persistent_workers": False,
        "num_observed_frames": 4,
        "dataset_roots": [_formal_real_runtime_root()],
    }
    arguments[argument] = value

    with pytest.raises(ValueError, match=contract_field):
        navsim_data.init_navsim_data(**arguments)
