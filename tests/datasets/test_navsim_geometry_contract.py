"""Strict Real/CF geometry transport contract for NavSim."""

import inspect

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from app.vjepa_cowa_world_model.training import config as config_module
from app.vjepa_cowa_world_model.training import data as data_module
from app.vjepa_cowa_world_model.training.config import NavSimConfig, parse_training_config
from app.vjepa_cowa_world_model.training.navsim_data import (
    NavSimWorldModelDataset,
    RootTaggedDataset,
    SceneRecord,
    WindowRecord,
    init_navsim_data,
    navsim_world_model_collate_fn,
)


def _dataset_for_agent_build(max_agents: int = 256) -> NavSimWorldModelDataset:
    dataset = NavSimWorldModelDataset.__new__(NavSimWorldModelDataset)
    dataset.max_agents = max_agents
    return dataset


def _dynamic_frame(count: int, *, invalid_box: np.ndarray | None = None) -> dict:
    boxes = np.zeros((count, 7), dtype=np.float32)
    boxes[:, 3:6] = np.asarray([4.0, 2.0, 1.5], dtype=np.float32)
    if invalid_box is not None:
        boxes[-1] = invalid_box
    return {
        "anns": {
            "gt_boxes": boxes,
            "gt_names": ["vehicle"] * count,
        }
    }


def test_navsim_capacity_defaults_to_256_everywhere() -> None:
    config = parse_training_config({"data": {"navsim": {"enabled": True}}})

    assert NavSimConfig().max_agents == 256
    assert config.data.navsim is not None
    assert config.data.navsim.max_agents == 256
    assert inspect.signature(NavSimWorldModelDataset).parameters["max_agents"].default == 256
    assert inspect.signature(init_navsim_data).parameters["max_agents"].default == 256


def test_pose_overlay_required_rejects_missing_path() -> None:
    with pytest.raises(ValueError, match="pose_overlay_required.*pose_overlay_path"):
        NavSimWorldModelDataset(
            data_path="/missing/logs",
            sensor_blobs_path="/missing/blobs",
            pose_overlay_required=True,
            pose_overlay_path=None,
        )


def test_training_data_forwards_navsim_geometry_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _Loader:
        def __len__(self) -> int:
            return 0

    def _fake_init_navsim_data(**kwargs):
        captured.update(kwargs)
        return _Loader(), None

    monkeypatch.setattr(data_module, "init_navsim_data", _fake_init_navsim_data)
    config = parse_training_config(
        {
            "data": {
                "navsim": {
                    "enabled": True,
                    "data_path": "/real/logs",
                    "sensor_blobs_path": "/real/blobs",
                    "max_agents": 123,
                    "load_agent_annotations": False,
                }
            }
        }
    )

    data_module.create_train_dataloader(config, rank=0, world_size=1, transform=object())

    assert captured["max_agents"] == 123
    assert captured["load_agent_annotations"] is False


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "256"])
def test_navsim_capacity_requires_a_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="max_agents.*positive integer"):
        parse_training_config({"data": {"navsim": {"enabled": True, "max_agents": value}}})


def test_mixed_roots_require_one_resolved_capacity() -> None:
    with pytest.raises(ValueError, match="max_agents.*same capacity"):
        parse_training_config(
            {
                "data": {
                    "navsim": {
                        "enabled": True,
                        "max_agents": 256,
                        "train_roots": [
                            {
                                "name": "real",
                                "domain": "real",
                                "data_path": "/real/logs",
                                "sensor_blobs_path": "/real/blobs",
                            },
                            {
                                "name": "cf",
                                "domain": "counterfactual",
                                "data_path": "/cf/logs",
                                "sensor_blobs_path": "/cf/blobs",
                                "max_agents": 128,
                            },
                        ],
                    }
                }
            }
        )


def test_mixed_roots_may_share_an_explicit_capacity() -> None:
    config = parse_training_config(
        {
            "data": {
                "navsim": {
                    "enabled": True,
                    "train_roots": [
                        {
                            "name": "real",
                            "domain": "real",
                            "data_path": "/real/logs",
                            "sensor_blobs_path": "/real/blobs",
                            "max_agents": 128,
                        },
                        {
                            "name": "cf",
                            "domain": "counterfactual",
                            "data_path": "/cf/logs",
                            "sensor_blobs_path": "/cf/blobs",
                            "max_agents": 128,
                        },
                    ],
                }
            }
        }
    )

    assert config.data.navsim is not None
    assert config.data.navsim.max_agents == 128
    assert [root["max_agents"] for root in config.data.navsim.train_roots] == [128, 128]


def test_256_dynamic_agents_are_preserved_and_counted() -> None:
    dataset = _dataset_for_agent_build()

    result = dataset._build_agent_annotations([_dynamic_frame(256)], [0])

    assert len(result) == 3
    agent_boxes, agent_mask, raw_agent_count = result
    assert agent_boxes.shape == (1, 256, 7)
    assert agent_mask.shape == (1, 256)
    assert agent_mask.all()
    assert raw_agent_count.tolist() == [256]


def test_257_dynamic_agents_fail_instead_of_truncating() -> None:
    dataset = _dataset_for_agent_build()

    with pytest.raises(ValueError, match="257.*max_agents=256"):
        dataset._build_agent_annotations([_dynamic_frame(257)], [0])


def test_training_sample_agent_overflow_fails_without_random_window_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = NavSimWorldModelDataset.__new__(NavSimWorldModelDataset)
    dataset.windows = [WindowRecord(scene_idx=0, start_pos=0)]
    dataset.scenes = [
        SceneRecord(
            scene_name="overflow-scene",
            pkl_path="/tmp/overflow-scene.pkl",
            camera_dir="/tmp/CAM_F0",
            valid_frame_indices=[0],
        )
    ]
    dataset.tubelet_size = 1
    dataset.transform = None
    dataset.proposal_transform = None
    dataset.pose_overlay_reader = None
    dataset.action_dim = 3
    dataset.num_observed_frames = 1
    dataset.load_agent_annotations = True
    dataset.max_agents = 256
    dataset.camera_names = ["CAM_F0"]
    dataset._scene_annotations = None
    dataset._scene_trajectory_quality = None
    dataset.token_mode = False
    dataset.is_validation = False

    frames = [_dynamic_frame(257)]
    dataset._load_scene_frames = lambda _path: frames
    dataset._sampled_frame_indices = lambda _start: [0]
    dataset._required_image_frame_indices = lambda sampled: list(sampled)
    dataset._build_metadata_valid_mask = lambda _frames, indices: np.ones(len(indices), dtype=np.bool_)
    dataset._validate_counterfactual_scene_start_metadata = lambda **_kwargs: None
    dataset._load_clip_images = lambda *_args: (
        np.zeros((1, 4, 8, 3), dtype=np.uint8),
        np.asarray([[4, 8]], dtype=np.int64),
    )
    dataset._default_image_tensor = lambda _buffer: torch.zeros((3, 1, 4, 8), dtype=torch.float32)
    dataset._infer_output_hw = lambda _buffer: (4, 8)
    dataset._build_camera_metadata = lambda *_args, **_kwargs: (
        np.zeros((1, 3, 3), dtype=np.float32),
        np.zeros((1, 4, 4), dtype=np.float32),
    )
    dataset._build_states = lambda _frames, indices: np.zeros((len(indices), 7), dtype=np.float32)
    dataset._build_actions = lambda _states: np.zeros((0, 3), dtype=np.float32)
    dataset._build_ego_dynamics = lambda _frames, indices: np.zeros((len(indices), 4), dtype=np.float32)
    dataset._build_driving_commands = lambda _frames, indices: np.zeros((len(indices), 4), dtype=np.float32)
    monkeypatch.setattr(
        "app.vjepa_cowa_world_model.training.navsim_data.random.randint",
        lambda *_args: (_ for _ in ()).throw(AssertionError("agent overflow must not select a replacement window")),
    )

    with pytest.raises(ValueError, match="257.*max_agents=256"):
        dataset[0]


@pytest.mark.parametrize(
    ("invalid_box", "message"),
    [
        (np.asarray([np.nan, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0], dtype=np.float32), "finite"),
        (np.asarray([0.0, 0.0, 0.0, 0.0, 2.0, 1.5, 0.0], dtype=np.float32), "positive dimensions"),
        (np.asarray([0.0, 0.0, 0.0, 4.0, -2.0, 1.5, 0.0], dtype=np.float32), "positive dimensions"),
    ],
)
def test_dynamic_agent_boxes_require_finite_positive_geometry(invalid_box: np.ndarray, message: str) -> None:
    dataset = _dataset_for_agent_build()

    with pytest.raises(ValueError, match=message):
        dataset._build_agent_annotations([_dynamic_frame(1, invalid_box=invalid_box)], [0])


class _GeometrySource(Dataset):
    def __init__(self, *, counterfactual: bool) -> None:
        self.counterfactual = counterfactual

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        del index
        scene_name = "scene-0001_cf_000010_000019" if self.counterfactual else "scene-0001"
        agent_boxes = np.zeros((2, 256, 7), dtype=np.float32)
        agent_mask = np.zeros((2, 256), dtype=np.bool_)
        raw_agent_count = None
        if not self.counterfactual:
            agent_boxes[:, :7, 3:6] = np.asarray([4.0, 2.0, 1.5], dtype=np.float32)
            agent_mask[:, :7] = True
            raw_agent_count = np.asarray([7, 7], dtype=np.int64)
        item = {
            "buffer": torch.zeros(3, 2, 8, 8),
            "proposal_buffer": None,
            "actions": np.zeros((1, 3), dtype=np.float32),
            "states": np.zeros((2, 7), dtype=np.float32),
            "extrinsics": np.zeros((2, 7), dtype=np.float32),
            "driving_command": np.zeros((2, 4), dtype=np.float32),
            "ego_dynamics": np.zeros((2, 4), dtype=np.float32),
            "agent_boxes": agent_boxes,
            "agent_mask": agent_mask,
            "bev_segmentation": np.zeros((2, 200, 200), dtype=np.uint8),
            "raw_agent_count": raw_agent_count,
            "scene_name": scene_name,
            "pkl_path": f"/tmp/{scene_name}.pkl",
            "window_start_pos": 0,
            "sampled_frame_indices": np.asarray([0, 1], dtype=np.int64),
            "cf_annotation_valid": self.counterfactual,
            "cf_is_hazard": False,
            "cf_hazard_type": "",
            "future_agent_geometry_valid": not self.counterfactual,
        }
        if not self.counterfactual:
            item["sample_token"] = "official-token-001"
        return item


def test_real_and_cf_geometry_metadata_collate_without_shape_drift() -> None:
    real = RootTaggedDataset(
        _GeometrySource(counterfactual=False),
        domain="real",
        dataset_root_name="real",
        dataset_root_index=0,
        future_agent_geometry_valid=True,
        load_agent_annotations=True,
    )[0]
    counterfactual = RootTaggedDataset(
        _GeometrySource(counterfactual=True),
        domain="counterfactual",
        dataset_root_name="cf",
        dataset_root_index=1,
        future_agent_geometry_valid=False,
        load_agent_annotations=False,
    )[0]

    assert real["geometry_present"] is True
    assert real["geometry_source"] == "logged_nuscenes_gt"
    assert real["geometry_coordinate_frame"] == "per_frame_ego"
    assert real["coordinate_frame"] == "per_frame_ego"
    assert real["agent_geometry_truncated"] is False
    assert real["raw_agent_count"].tolist() == [7, 7]
    assert counterfactual["geometry_present"] is False
    assert counterfactual["future_agent_geometry_valid"] is False
    assert counterfactual["raw_agent_count"] is None
    assert counterfactual["agent_geometry_truncated"] is None
    assert not counterfactual["agent_mask"].any()
    assert not counterfactual["agent_boxes"].any()
    assert not counterfactual["bev_segmentation"].any()

    batch = navsim_world_model_collate_fn([real, counterfactual])
    assert batch[7].shape == (2, 2, 256, 7)
    assert batch[8].shape == (2, 2, 256)
    metadata = batch[-1]
    assert metadata["sample_token"] == ["official-token-001", None]
    assert metadata["sample_token_valid_mask"].tolist() == [True, False]
    assert metadata["geometry_present"].tolist() == [True, False]
    assert metadata["future_agent_geometry_valid"].tolist() == [True, False]
    assert metadata["geometry_source"] == ["logged_nuscenes_gt", None]
    assert metadata["geometry_coordinate_frame"] == ["per_frame_ego", None]
    assert metadata["coordinate_frame"] == ["per_frame_ego", None]
    assert metadata["agent_geometry_truncated"] == [False, None]
    assert metadata["raw_agent_count"][0].tolist() == [7, 7]
    assert metadata["raw_agent_count"][1] is None


def test_counterfactual_root_rejects_future_geometry_claim() -> None:
    with pytest.raises(ValueError, match="counterfactual.*future_agent_geometry_valid"):
        RootTaggedDataset(
            _GeometrySource(counterfactual=True),
            domain="counterfactual",
            dataset_root_name="cf",
            dataset_root_index=0,
            future_agent_geometry_valid=True,
            load_agent_annotations=False,
        )


def test_counterfactual_sample_rejects_future_geometry_claim() -> None:
    source_item = _GeometrySource(counterfactual=True)[0]
    source_item["future_agent_geometry_valid"] = True

    class _SingleSample(Dataset):
        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            del index
            return source_item

    dataset = RootTaggedDataset(
        _SingleSample(),
        domain="counterfactual",
        dataset_root_name="cf",
        dataset_root_index=0,
        future_agent_geometry_valid=False,
        load_agent_annotations=False,
    )

    with pytest.raises(ValueError, match="counterfactual.*future_agent_geometry_valid"):
        dataset[0]


def _future_geometry_frame(timestamp: int, *, agent_count: int = 1) -> dict:
    frame = _dynamic_frame(agent_count)
    frame.update(
        {
            "timestamp": timestamp,
            "ego2global_translation": np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            "ego2global_rotation": np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
    )
    return frame


def _dataset_for_future_validity() -> NavSimWorldModelDataset:
    dataset = _dataset_for_agent_build()
    dataset.max_time_gap_us = 750_000.0
    return dataset


def test_future_geometry_validity_checks_full_geometry_contract() -> None:
    dataset = _dataset_for_future_validity()
    frames = [_future_geometry_frame(0), _future_geometry_frame(500_000)]

    assert dataset._future_agent_annotations_are_valid(frames, [0, 1]) is True


@pytest.mark.parametrize("failure", ["finite", "dimensions", "overflow", "pose", "time"])
def test_future_geometry_validity_rejects_invalid_geometry(failure: str) -> None:
    dataset = _dataset_for_future_validity()
    frames = [_future_geometry_frame(0), _future_geometry_frame(500_000)]
    if failure == "finite":
        frames[1]["anns"]["gt_boxes"][0, 0] = np.nan
    elif failure == "dimensions":
        frames[1]["anns"]["gt_boxes"][0, 3] = 0.0
    elif failure == "overflow":
        frames[1] = _future_geometry_frame(500_000, agent_count=257)
    elif failure == "pose":
        frames[1].pop("ego2global_translation")
    else:
        frames[1]["timestamp"] = -1

    assert dataset._future_agent_annotations_are_valid(frames, [0, 1]) is False


def _cvoi_geometry_validator():
    validator = getattr(config_module, "validate_navsim_cvoi_geometry_contract", None)
    assert callable(validator), "CVoI NavSim geometry contract validator must be exported"
    return validator


def test_cvoi_geometry_contract_accepts_real_and_cf_roots() -> None:
    capacity = _cvoi_geometry_validator()(
        [
            {"name": "real", "domain": "real", "load_agent_annotations": True},
            {
                "name": "cf",
                "domain": "counterfactual",
                "load_agent_annotations": False,
            },
        ],
        default_max_agents=256,
    )

    assert capacity == 256


@pytest.mark.parametrize(
    ("root", "message"),
    [
        ({"name": "real", "domain": "real", "load_agent_annotations": False}, "real.*true"),
        ({"name": "cf", "domain": "counterfactual"}, "counterfactual.*false"),
        ({"name": "cf", "domain": "counterfactual", "load_agent_annotations": True}, "counterfactual.*false"),
    ],
)
def test_cvoi_geometry_contract_enforces_domain_annotation_loading(root: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _cvoi_geometry_validator()([root], default_max_agents=256)


def test_cvoi_geometry_contract_enforces_one_transport_capacity() -> None:
    with pytest.raises(ValueError, match="same capacity"):
        _cvoi_geometry_validator()(
            [
                {
                    "name": "real",
                    "domain": "real",
                    "load_agent_annotations": True,
                    "max_agents": 256,
                },
                {
                    "name": "cf",
                    "domain": "counterfactual",
                    "load_agent_annotations": False,
                    "max_agents": 128,
                },
            ],
            default_max_agents=256,
        )
