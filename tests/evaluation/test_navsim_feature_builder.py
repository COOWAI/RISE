"""Tests for VJEPAFeatureBuilder driving_command/ego_dynamics extraction."""

import importlib
import importlib.util
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import numpy.testing as npt
import torch

from app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity import (
    decode_observation_key,
    decode_unsigned_seed,
    observation_key,
)
from app.vjepa_cowa_world_model.training.cvoi_execution import cvoi_sample_seed


def _setup_navsim_mocks():
    """Install minimal navsim mocks so feature_builder can be imported."""
    for mod_name in [
        "navsim",
        "navsim.common",
        "navsim.common.dataclasses",
        "navsim.planning",
        "navsim.planning.training",
        "navsim.planning.training.abstract_feature_target_builder",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    # Provide a real base class so VJEPAFeatureBuilder can inherit
    class _AbstractFeatureBuilder:
        def __init__(self):
            pass

    sys.modules["navsim.common.dataclasses"].AgentInput = MagicMock
    sys.modules["navsim.planning.training.abstract_feature_target_builder"].AbstractFeatureBuilder = (
        _AbstractFeatureBuilder
    )


_setup_navsim_mocks()


def _load_feature_builder_module():
    """Load navsim_feature_builder via importlib to bypass __init__ chains."""
    spec = importlib.util.spec_from_file_location(
        "navsim_feature_builder",
        "app/vjepa_cowa_world_model/evaluation/navsim_feature_builder.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FEATURE_BUILDER_MODULE = _load_feature_builder_module()
VJEPAFeatureBuilder = FEATURE_BUILDER_MODULE.VJEPAFeatureBuilder
_normalize_crop_size = FEATURE_BUILDER_MODULE._normalize_crop_size


def _make_ego_status(x=0.0, y=0.0, heading=0.0, vx=5.0, vy=1.0, ax=0.5, ay=0.1, driving_cmd_idx=0):
    """Create a mock EgoStatus with given values."""
    status = MagicMock()
    status.ego_pose = np.array([x, y, heading], dtype=np.float64)
    status.ego_velocity = np.array([vx, vy], dtype=np.float32)
    status.ego_acceleration = np.array([ax, ay], dtype=np.float32)
    cmd = np.zeros(4, dtype=np.int32)
    cmd[driving_cmd_idx] = 1
    status.driving_command = cmd
    return status


def _make_camera(image: np.ndarray):
    camera = MagicMock()
    camera.image = image
    camera.intrinsics = np.array(
        [
            [100.0, 0.0, image.shape[1] / 2.0],
            [0.0, 100.0, image.shape[0] / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera.camera2ego = np.eye(4, dtype=np.float32)
    return camera


def _make_agent_input(num_frames=4, vx=5.0, vy=1.0):
    """Create a mock AgentInput with consistent ego_statuses and cameras."""
    agent_input = MagicMock()
    statuses = []
    for i in range(num_frames):
        statuses.append(_make_ego_status(x=float(i), y=float(i * 0.5), heading=0.1 * i, vx=vx, vy=vy))
    agent_input.ego_statuses = statuses

    # Mock cameras with 64x128 images
    cameras = []
    for _ in range(num_frames):
        cam = MagicMock()
        image = np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8)
        cam.cam_f0 = _make_camera(image)
        cam.cam_l0 = _make_camera(image)
        cam.cam_r0 = _make_camera(image)
        cameras.append(cam)
    agent_input.cameras = cameras
    return agent_input


class TestSpeedIsVx(unittest.TestCase):
    """Speed field in states should be vx (longitudinal), not norm(vx, vy)."""

    def test_speed_is_vx_not_norm(self):
        """Verify states[:,6] = vx, not sqrt(vx^2 + vy^2)."""
        builder = VJEPAFeatureBuilder(crop_size=64)
        vx, vy = 5.0, 3.0
        agent_input = _make_agent_input(num_frames=4, vx=vx, vy=vy)

        features = builder.compute_features(agent_input)
        states = features["states"].numpy()

        # Speed should be vx (5.0), NOT norm(5.0, 3.0) = 5.83
        for t in range(states.shape[0]):
            npt.assert_almost_equal(
                states[t, 6], vx, decimal=5, err_msg=f"Frame {t}: speed should be vx={vx}, got {states[t, 6]}"
            )


class TestNormalizeCropSize(unittest.TestCase):
    def test_int_and_pair_inputs(self):
        self.assertEqual(_normalize_crop_size(64), (64, 64))
        self.assertEqual(_normalize_crop_size((32, 96)), (32, 96))
        self.assertEqual(_normalize_crop_size([48, 80]), (48, 80))

    def test_invalid_input_raises(self):
        with self.assertRaisesRegex(ValueError, "crop_size"):
            _normalize_crop_size((32, 64, 96))


class TestSingleViewFeatureAssembly(unittest.TestCase):
    def test_rectangular_crop_shapes_dtypes_and_intrinsics_scaling(self):
        builder = VJEPAFeatureBuilder(crop_size=(64, 32), camera_name="cam_f0")
        agent_input = _make_agent_input(num_frames=4)

        features = builder.compute_features(agent_input)

        self.assertEqual(tuple(features["video_clip"].shape), (3, 4, 64, 32))
        self.assertEqual(features["video_clip"].dtype, torch.float32)
        self.assertEqual(tuple(features["camera_intrinsics"].shape), (1, 4, 3, 3))
        self.assertEqual(features["camera_intrinsics"].dtype, torch.float32)
        self.assertEqual(tuple(features["camera2ego"].shape), (1, 4, 4, 4))
        self.assertEqual(features["actions"].dtype, torch.float32)
        npt.assert_allclose(
            features["camera_intrinsics"][0, 0].numpy(),
            np.array([[12.5, 0.0, 16.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_missing_single_view_camera_image_raises(self):
        builder = VJEPAFeatureBuilder(crop_size=64, camera_name="cam_f0")
        agent_input = _make_agent_input(num_frames=4)
        agent_input.cameras[2].cam_f0.image = None

        with self.assertRaisesRegex(ValueError, "Camera cam_f0 image is None at frame 2"):
            builder.compute_features(agent_input)


class TestOfficialCvoiIdentityFeatures(unittest.TestCase):
    def test_disabled_mode_preserves_exact_feature_schema_and_values(self):
        agent_input = _make_agent_input(num_frames=4)

        legacy_features = VJEPAFeatureBuilder(crop_size=(32, 48)).compute_features(agent_input)
        disabled_features = VJEPAFeatureBuilder(
            crop_size=(32, 48),
            official_cvoi_identity=False,
            cvoi_evaluation_seed=None,
        ).compute_features(agent_input)

        self.assertEqual(tuple(disabled_features), tuple(legacy_features))
        for name, legacy_value in legacy_features.items():
            self.assertTrue(torch.equal(disabled_features[name], legacy_value), name)

    def test_enabled_mode_hashes_raw_final_cam_f0_before_crop_and_emits_only_identity_tensors(self):
        agent_input = _make_agent_input(num_frames=4)
        raw_image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        agent_input.cameras[-1].cam_f0 = _make_camera(raw_image)
        expected_key = observation_key(raw_image)
        evaluation_seed = next(
            seed for seed in range(1 << 63, (1 << 63) + 1000) if cvoi_sample_seed(seed, expected_key) >= 1 << 63
        )

        baseline = VJEPAFeatureBuilder(crop_size=(2, 3)).compute_features(agent_input)
        features = VJEPAFeatureBuilder(
            crop_size=(2, 3),
            camera_name="cam_f0",
            official_cvoi_identity=True,
            cvoi_evaluation_seed=evaluation_seed,
        ).compute_features(agent_input)

        self.assertEqual(
            set(features) - set(baseline),
            {"cvoi_observation_key", "cvoi_rng_seed_bytes"},
        )
        self.assertEqual(features["cvoi_observation_key"].dtype, torch.uint8)
        self.assertEqual(tuple(features["cvoi_observation_key"].shape), (32,))
        self.assertEqual(decode_observation_key(features["cvoi_observation_key"]), expected_key)
        self.assertEqual(features["cvoi_rng_seed_bytes"].dtype, torch.uint8)
        self.assertEqual(tuple(features["cvoi_rng_seed_bytes"].shape), (8,))
        expected_sample_seed = cvoi_sample_seed(evaluation_seed, expected_key)
        self.assertGreaterEqual(expected_sample_seed, 1 << 63)
        self.assertEqual(decode_unsigned_seed(features["cvoi_rng_seed_bytes"]), expected_sample_seed)
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in features.values()))

    def test_enabled_mode_requires_raw_final_cam_f0_image_even_for_another_model_camera(self):
        agent_input = _make_agent_input(num_frames=4)
        agent_input.cameras[-1].cam_f0.image = None
        builder = VJEPAFeatureBuilder(
            crop_size=64,
            camera_name="cam_l0",
            official_cvoi_identity=True,
            cvoi_evaluation_seed=239,
        )

        with self.assertRaisesRegex(ValueError, "final-history cam_f0 image"):
            builder.compute_features(agent_input)

    def test_constructor_rejects_seed_when_identity_is_disabled(self):
        with self.assertRaisesRegex(ValueError, "must be None"):
            VJEPAFeatureBuilder(official_cvoi_identity=False, cvoi_evaluation_seed=239)

    def test_constructor_requires_boolean_identity_flag(self):
        for enabled in (None, 0, 1, "true"):
            with self.subTest(enabled=enabled), self.assertRaisesRegex(TypeError, "must be a bool"):
                VJEPAFeatureBuilder(official_cvoi_identity=enabled, cvoi_evaluation_seed=None)

    def test_constructor_requires_full_unsigned_64_bit_seed_when_identity_is_enabled(self):
        invalid_seeds = (None, True, -1, 1 << 64, 1.5, "239")
        for seed in invalid_seeds:
            with self.subTest(seed=seed), self.assertRaisesRegex((TypeError, ValueError), "unsigned 64-bit"):
                VJEPAFeatureBuilder(official_cvoi_identity=True, cvoi_evaluation_seed=seed)

        VJEPAFeatureBuilder(official_cvoi_identity=True, cvoi_evaluation_seed=0)
        VJEPAFeatureBuilder(official_cvoi_identity=True, cvoi_evaluation_seed=(1 << 64) - 1)


class TestDrivingCommandExtraction(unittest.TestCase):
    """Feature builder should extract driving_command from AgentInput."""

    def test_driving_command_shape(self):
        """driving_command should be [T, 4]."""
        builder = VJEPAFeatureBuilder(crop_size=64)
        agent_input = _make_agent_input(num_frames=4)
        features = builder.compute_features(agent_input)

        self.assertIn("driving_command", features)
        dc = features["driving_command"]
        self.assertEqual(dc.shape, (4, 4))  # [T, 4]

    def test_driving_command_values(self):
        """driving_command should match ego_status.driving_command."""
        builder = VJEPAFeatureBuilder(crop_size=64)
        agent_input = _make_agent_input(num_frames=4)
        # Set frame 2 to TURN_LEFT
        cmd = np.zeros(4, dtype=np.int32)
        cmd[1] = 1
        agent_input.ego_statuses[2].driving_command = cmd

        features = builder.compute_features(agent_input)
        dc = features["driving_command"].numpy()

        # Frame 0: GO_STRAIGHT
        npt.assert_array_equal(dc[0], [1, 0, 0, 0])
        # Frame 2: TURN_LEFT
        npt.assert_array_equal(dc[2], [0, 1, 0, 0])


class TestMissingEgoStatusContract(unittest.TestCase):
    """ego_status 缺失时的分层 fail-loud 契约。

    states 无条件必需 → 直接报错；driving_command/ego_dynamics 条件必需 →
    不伪造 zeros、省略 key，由消费端（status_features）在配置需要时报错。
    """

    def test_missing_ego_status_omits_conditional_features(self):
        builder = VJEPAFeatureBuilder(crop_size=64)
        agent_input = _make_agent_input(num_frames=4)
        agent_input.ego_statuses[1] = None

        self.assertIsNone(builder._build_driving_commands(agent_input))
        self.assertIsNone(builder._build_ego_dynamics(agent_input))

    def test_missing_ego_status_raises_for_states(self):
        builder = VJEPAFeatureBuilder(crop_size=64)
        agent_input = _make_agent_input(num_frames=4)
        agent_input.ego_statuses[1] = None

        with self.assertRaisesRegex(ValueError, "states"):
            builder._build_states(agent_input)


class TestEgoDynamicsExtraction(unittest.TestCase):
    """Feature builder should extract ego_dynamics from AgentInput."""

    def test_ego_dynamics_shape(self):
        """ego_dynamics should be [T, 4]."""
        builder = VJEPAFeatureBuilder(crop_size=64)
        agent_input = _make_agent_input(num_frames=4)
        features = builder.compute_features(agent_input)

        self.assertIn("ego_dynamics", features)
        ed = features["ego_dynamics"]
        self.assertEqual(ed.shape, (4, 4))  # [T, 4]

    def test_ego_dynamics_values(self):
        """ego_dynamics should be [vx, vy, ax, ay] from ego_status."""
        builder = VJEPAFeatureBuilder(crop_size=64)
        agent_input = _make_agent_input(num_frames=4)
        # Set specific values on frame 1
        agent_input.ego_statuses[1].ego_velocity = np.array([3.0, 0.5], dtype=np.float32)
        agent_input.ego_statuses[1].ego_acceleration = np.array([0.2, -0.1], dtype=np.float32)

        features = builder.compute_features(agent_input)
        ed = features["ego_dynamics"].numpy()

        npt.assert_almost_equal(ed[1], [3.0, 0.5, 0.2, -0.1], decimal=5)


class TestMultiViewFeatureBuilder(unittest.TestCase):
    """Feature builder should preserve training/eval camera alignment."""

    def test_multiview_video_clip_and_camera_metadata_shapes(self):
        builder = VJEPAFeatureBuilder(crop_size=64, camera_names=["cam_l0", "cam_f0", "cam_r0"])
        agent_input = _make_agent_input(num_frames=4)

        features = builder.compute_features(agent_input)

        self.assertEqual(tuple(features["video_clip"].shape), (3, 3, 4, 64, 64))
        self.assertEqual(tuple(features["camera_intrinsics"].shape), (3, 4, 3, 3))
        self.assertEqual(tuple(features["camera2ego"].shape), (3, 4, 4, 4))
        # camera_names must NOT be emitted as a feature: NavSim's AbstractAgent.compute_trajectory
        # batches every feature via `{k: v.unsqueeze(0).cuda() for k, v in features.items()}`, which
        # only works for tensors — a string list would crash PDMS eval. The view order/identity is
        # carried per-view by camera_intrinsics/camera2ego; camera_names lives on the builder.
        self.assertNotIn("camera_names", features)
        self.assertEqual(builder.camera_names, ["cam_l0", "cam_f0", "cam_r0"])

    def test_missing_multiview_camera_raises(self):
        builder = VJEPAFeatureBuilder(crop_size=64, camera_names=["cam_l0", "cam_f0", "cam_r0"])
        agent_input = _make_agent_input(num_frames=4)
        agent_input.cameras[0].cam_r0 = None

        with self.assertRaises(ValueError):
            builder.compute_features(agent_input)


if __name__ == "__main__":
    unittest.main()
