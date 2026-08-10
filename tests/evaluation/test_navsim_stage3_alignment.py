"""Regression tests for NavSim Stage-3 evaluation alignment."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.testing as npt
import torch

from app.vjepa_cowa_world_model.training.vjepa_transforms import VJEPAImageTransform

REPO_ROOT = Path(__file__).resolve().parents[2]


def _setup_navsim_mocks():
    """Install minimal navsim mocks so the feature builder can be imported."""
    for mod_name in [
        "navsim",
        "navsim.agents",
        "navsim.agents.abstract_agent",
        "navsim.common",
        "navsim.common.dataclasses",
        "navsim.planning",
        "navsim.planning.training",
        "navsim.planning.training.abstract_feature_target_builder",
        "nuplan",
        "nuplan.planning",
        "nuplan.planning.simulation",
        "nuplan.planning.simulation.trajectory",
        "nuplan.planning.simulation.trajectory.trajectory_sampling",
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = MagicMock()

    class _AbstractAgent:
        def __init__(self, requires_scene=False):
            self.requires_scene = requires_scene

    class _AbstractFeatureBuilder:
        def __init__(self):
            pass

    class _SensorConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _TrajectorySampling:
        def __init__(self, time_horizon, interval_length):
            self.time_horizon = time_horizon
            self.interval_length = interval_length

    sys.modules["navsim.agents.abstract_agent"].AbstractAgent = _AbstractAgent
    sys.modules["navsim.common.dataclasses"].AgentInput = MagicMock
    sys.modules["navsim.common.dataclasses"].SensorConfig = _SensorConfig
    sys.modules["navsim.planning.training.abstract_feature_target_builder"].AbstractFeatureBuilder = (
        _AbstractFeatureBuilder
    )
    sys.modules["nuplan.planning.simulation.trajectory.trajectory_sampling"].TrajectorySampling = _TrajectorySampling


def _make_agent_input(num_frames=4):
    agent_input = MagicMock()
    agent_input.ego_statuses = []
    agent_input.cameras = []
    for idx in range(num_frames):
        status = MagicMock()
        status.ego_pose = np.array([float(idx), 0.0, 0.0], dtype=np.float32)
        status.ego_velocity = np.array([1.0, 0.0], dtype=np.float32)
        status.ego_acceleration = np.array([0.0, 0.0], dtype=np.float32)
        status.driving_command = np.array([1, 0, 0, 0], dtype=np.float32)
        agent_input.ego_statuses.append(status)

        cam_f0 = SimpleNamespace(
            image=np.random.randint(0, 255, (128, 256, 3), dtype=np.uint8),
            intrinsics=np.array(
                [[200.0, 0.0, 128.0], [0.0, 200.0, 64.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            sensor2lidar_rotation=np.array(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            ),
            sensor2lidar_translation=np.array([1.5, -0.25, 1.2], dtype=np.float32),
        )
        camera_group = SimpleNamespace(cam_f0=cam_f0)
        agent_input.cameras.append(camera_group)
    return agent_input


def _make_gradient_agent_input(num_frames=4, height=128, width=256):
    agent_input = _make_agent_input(num_frames=num_frames)
    y = np.arange(height, dtype=np.uint8)[:, None]
    gradient = np.repeat(y, width, axis=1)
    image = np.stack([gradient, np.zeros_like(gradient), np.zeros_like(gradient)], axis=-1)
    for camera_group in agent_input.cameras:
        camera_group.cam_f0.image = image.copy()
    return agent_input


def _load_feature_builder():
    spec = importlib.util.spec_from_file_location(
        "navsim_feature_builder",
        REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_feature_builder.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VJEPAFeatureBuilder


def _load_navsim_agent_module():
    spec = importlib.util.spec_from_file_location(
        "navsim_agent",
        REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_navsim_agent():
    return _load_navsim_agent_module().VJEPAWorldModelAgent


def _fixed_rollout_config(**overrides):
    config = SimpleNamespace(
        train=SimpleNamespace(
            predictor_inference_consistent=True,
            predictor_no_aux_input=False,
            predictor_type="ac_transformer",
            use_parallel_predictor=False,
        ),
        planner=SimpleNamespace(
            use_planner=True,
            planner_type="diffusion",
            policy_output_source="planner",
            z_ar_mode="full",
            use_z_context=False,
        ),
        value_guidance=SimpleNamespace(enabled=False),
        value_planning=SimpleNamespace(enabled=False, variant="a_method1"),
        cvoi=SimpleNamespace(enabled=False),
        budget_controller=SimpleNamespace(enabled=False),
    )
    for path, value in overrides.items():
        section_name, field_name = path.split("__", 1)
        setattr(getattr(config, section_name), field_name, value)
    return config


_setup_navsim_mocks()
VJEPAFeatureBuilder = _load_feature_builder()


class TestStage3FeatureBuilder(unittest.TestCase):
    def test_navsim_camera_geometry_uses_sensor2lidar_extrinsics(self):
        agent_input = _make_agent_input(num_frames=1)
        cam_f0 = agent_input.cameras[0].cam_f0

        self.assertIsInstance(cam_f0, SimpleNamespace)
        self.assertFalse(hasattr(cam_f0, "camera2ego"))

        features = VJEPAFeatureBuilder(crop_size=(64, 128)).compute_features(agent_input)

        expected = np.array(
            [
                [0.0, -1.0, 0.0, 1.5],
                [1.0, 0.0, 0.0, -0.25],
                [0.0, 0.0, 1.0, 1.2],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.assertEqual(tuple(features["camera2ego"].shape), (1, 1, 4, 4))
        npt.assert_allclose(features["camera2ego"][0, 0].numpy(), expected, rtol=0.0, atol=0.0)

    def test_navsim_camera_geometry_rejects_missing_sensor2lidar_extrinsics(self):
        builder = VJEPAFeatureBuilder(crop_size=(64, 128))
        for missing_field in ("sensor2lidar_rotation", "sensor2lidar_translation"):
            with self.subTest(missing_field=missing_field):
                agent_input = _make_agent_input(num_frames=1)
                setattr(agent_input.cameras[0].cam_f0, missing_field, None)

                with self.assertRaisesRegex(ValueError, "missing sensor2lidar extrinsics"):
                    builder.compute_features(agent_input)

    def test_navsim_feature_builder_emits_only_tensor_features(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_feature_builder.py").read_text()
        compute_features_source = source.split("def compute_features", 1)[1].split(
            "# ------------------------------------------------------------------", 1
        )[0]

        self.assertNotIn('"camera_names": list(self.camera_names)', compute_features_source)

    def test_main_video_clip_uses_vjepa_crop_transform(self):
        builder = VJEPAFeatureBuilder(
            crop_size=(32, 64),
            crop_top_bottom=8,
        )
        agent_input = _make_gradient_agent_input(num_frames=4)

        features = builder.compute_features(agent_input)

        raw_clip = np.stack([camera_group.cam_f0.image for camera_group in agent_input.cameras], axis=0)
        expected = VJEPAImageTransform(resolution=(32, 64), crop_top_bottom=8)(raw_clip)
        npt.assert_allclose(features["video_clip"].numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)

    def test_emits_proposal_video_clip_with_separate_resolution(self):
        builder = VJEPAFeatureBuilder(crop_size=64, proposal_crop_size=(64, 128))

        features = builder.compute_features(_make_agent_input(num_frames=4))

        self.assertEqual(tuple(features["video_clip"].shape), (3, 4, 64, 64))
        self.assertIn("proposal_video_clip", features)
        self.assertEqual(tuple(features["proposal_video_clip"].shape), (3, 4, 64, 128))

    def test_proposal_video_clip_uses_vjepa_crop_transform(self):
        builder = VJEPAFeatureBuilder(
            crop_size=64,
            proposal_crop_size=(32, 64),
            proposal_crop_top_bottom=8,
        )
        agent_input = _make_gradient_agent_input(num_frames=4)

        features = builder.compute_features(agent_input)

        raw_clip = np.stack([camera_group.cam_f0.image for camera_group in agent_input.cameras], axis=0)
        expected = VJEPAImageTransform(resolution=(32, 64), crop_top_bottom=8)(raw_clip)
        npt.assert_allclose(features["proposal_video_clip"].numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)


class TestStage3AgentSource(unittest.TestCase):
    def test_stage12_fixed_rollout_resolves_predictor_end_step(self):
        agent_module = _load_navsim_agent_module()

        self.assertIsNone(agent_module._resolve_eval_rollout_end_step(None, num_observed_steps=2, num_future_steps=4))
        self.assertEqual(
            agent_module._resolve_eval_rollout_end_step(0, num_observed_steps=2, num_future_steps=4),
            2,
        )
        self.assertEqual(
            agent_module._resolve_eval_rollout_end_step(4, num_observed_steps=2, num_future_steps=4),
            6,
        )
        with self.assertRaisesRegex(ValueError, "exceeds available predictor future steps"):
            agent_module._resolve_eval_rollout_end_step(5, num_observed_steps=2, num_future_steps=4)

    def test_stage12_fixed_rollout_constructor_rejects_invalid_values_and_modes(self):
        agent_class = _load_navsim_agent()

        for value in (True, 1.5, "1"):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "non-negative integer"):
                agent_class(rollout_future_steps=value)
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            agent_class(rollout_future_steps=-1)
        with self.assertRaisesRegex(ValueError, "only supported with forward_mode='stage12'"):
            agent_class(forward_mode="stage3", rollout_future_steps=1)

        agent = agent_class(rollout_future_steps=2)
        self.assertEqual(agent._rollout_future_steps, 2)

    def test_stage12_fixed_rollout_setter_updates_and_validates_horizon(self):
        agent_class = _load_navsim_agent()
        agent = agent_class()

        agent.set_rollout_future_steps(3)
        self.assertEqual(agent._rollout_future_steps, 3)
        agent.set_rollout_future_steps(None)
        self.assertIsNone(agent._rollout_future_steps)

        for value in (True, -1):
            with self.subTest(value=value), self.assertRaisesRegex((TypeError, ValueError), "non-negative integer"):
                agent.set_rollout_future_steps(value)

        non_stage12_agent = agent_class(forward_mode="stage3")
        with self.assertRaisesRegex(ValueError, "only supported with forward_mode='stage12'"):
            non_stage12_agent.set_rollout_future_steps(1)

    def test_stage12_fixed_rollout_setter_revalidates_initialized_agent_capabilities(self):
        agent_class = _load_navsim_agent()
        agent = agent_class()
        agent._config = _fixed_rollout_config(planner__use_z_context=True)

        with self.assertRaisesRegex(ValueError, "planner.use_z_context=false"):
            agent.set_rollout_future_steps(1)

        self.assertIsNone(agent._rollout_future_steps)

    def test_stage12_fixed_rollout_capability_accepts_target_diffusion_contract(self):
        agent_module = _load_navsim_agent_module()

        agent_module._validate_fixed_rollout_capability(
            _fixed_rollout_config(),
            forward_mode="stage12",
        )

    def test_stage12_fixed_rollout_capability_rejects_unsupported_configs(self):
        agent_module = _load_navsim_agent_module()
        cases = (
            ({"train__predictor_inference_consistent": False}, "predictor_inference_consistent=true"),
            ({"planner__z_ar_mode": "first"}, "planner.z_ar_mode='full'"),
            ({"planner__use_z_context": True}, "planner.use_z_context=false"),
            ({"planner__planner_type": "transformer"}, "variable-length z_ar"),
            ({"value_guidance__enabled": True}, "value_guidance.enabled=false"),
            ({"value_planning__enabled": True}, "value_planning.enabled=false"),
        )

        for overrides, message in cases:
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                agent_module._validate_fixed_rollout_capability(
                    _fixed_rollout_config(**overrides),
                    forward_mode="stage12",
                )

    def test_stage12_fixed_rollout_trajectory_requires_exact_pose_count_before_normalization(self):
        agent_class = _load_navsim_agent()

        for num_poses in (0, 7, 9):
            with self.subTest(num_poses=num_poses), self.assertRaisesRegex(ValueError, "exactly 8 poses"):
                agent_class._validate_fixed_rollout_trajectory(
                    torch.zeros((1, num_poses, 3)),
                    num_poses=8,
                )

    def test_stage12_fixed_rollout_trajectory_rejects_nonfinite_trimmed_tail(self):
        agent_class = _load_navsim_agent()
        trajectory = torch.zeros((1, 9, 3))
        trajectory[:, -1] = torch.nan

        with self.assertRaisesRegex(ValueError, "exactly 8 poses|finite"):
            agent_class._validate_fixed_rollout_trajectory(trajectory, num_poses=8)

        exact_trajectory = torch.zeros((1, 8, 3))
        exact_trajectory[:, -1] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            agent_class._validate_fixed_rollout_trajectory(exact_trajectory, num_poses=8)

    def test_ordinary_navsim_compute_trajectory_uses_model_device_instead_of_cuda(self):
        agent_module = _load_navsim_agent_module()
        agent = object.__new__(agent_module.VJEPAWorldModelAgent)
        agent._cvoi_trajectory_sampling = SimpleNamespace(num_poses=8)
        agent._encoder = torch.nn.Linear(1, 1).cpu()
        agent.eval = MagicMock()
        agent.get_feature_builders = lambda: [
            SimpleNamespace(compute_features=lambda agent_input: {"video_clip": torch.ones(2, 3)})
        ]
        captured = {}

        def forward(features):
            captured.update(features)
            return {"trajectory": torch.zeros(1, 8, 3)}

        agent.forward = forward
        trajectory_type = MagicMock()
        agent_module.Trajectory = trajectory_type

        agent.compute_trajectory(object())

        self.assertEqual(captured["video_clip"].device.type, "cpu")
        self.assertEqual(tuple(captured["video_clip"].shape), (1, 2, 3))
        trajectory_type.assert_called_once()
        self.assertEqual(tuple(trajectory_type.call_args.kwargs["poses"].shape), (8, 3))

    def test_stage12_inference_consistent_rollout_calls_predictor_once_per_future_step(self):
        agent_module = _load_navsim_agent_module()
        agent = object.__new__(agent_module.VJEPAWorldModelAgent)
        agent._predictor = object()
        agent._config = SimpleNamespace()
        agent._runtime_normalize_reps = False
        agent._predictor_inference_consistent = True
        agent._num_observed_frames = 2
        agent._num_future_to_predict = 4
        z = torch.zeros(1, 4, 3)
        actions = torch.zeros(1, 5, 3)
        states = torch.zeros(1, 6, 8)
        extrinsics = torch.zeros(1, 6, 7)

        for horizon in range(5):
            with self.subTest(horizon=horizon):
                calls = []

                def recording_step(_z, _actions, _states, _extrinsics):
                    calls.append(1)
                    return _z[:, -2:]

                with patch.object(agent_module, "make_predictor_step_fn", return_value=recording_step):
                    future = agent._run_predictor_ar(
                        z,
                        actions,
                        states,
                        extrinsics,
                        tokens_per_frame=2,
                        num_observed_steps=2,
                        num_future_steps=4,
                        rollout_end_step=2 + horizon,
                    )

                self.assertEqual(len(calls), horizon)
                self.assertEqual(future.shape, (1, horizon * 2, 3))

    def test_stage12_fixed_rollout_rejects_incompatible_predictor_paths(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_source = source.split("def _forward_stage12", 1)[1].split("def _forward_stage2", 1)[0]

        self.assertIn("_resolve_eval_rollout_end_step(", stage12_source)
        self.assertIn("rollout_end_step=resolved_rollout_end_step", stage12_source)
        self.assertIn("fixed rollout evaluation does not support latent-DiT predictors", stage12_source)
        self.assertIn("fixed rollout evaluation does not support parallel predictors", stage12_source)
        self.assertIn("fixed rollout evaluation cannot be combined with CVoI", stage12_source)
        self.assertIn("fixed rollout evaluation cannot be combined with the budget controller", stage12_source)

    def test_cvoi_planner_output_getter_is_read_only_and_rejects_missing_forward(self):
        agent_class = _load_navsim_agent()
        agent = object.__new__(agent_class)
        agent._last_cvoi_planner_output = None

        with self.assertRaisesRegex(RuntimeError, "before the first trajectory"):
            agent.get_last_cvoi_planner_output()

        candidates = torch.arange(36, dtype=torch.float32).reshape(1, 2, 6, 3)
        confidences = torch.tensor([[0.2, 0.8]])
        agent._last_cvoi_planner_output = {
            "pred_trajs": candidates,
            "confidences": confidences,
        }
        output = agent.get_last_cvoi_planner_output()

        self.assertEqual(set(output), {"pred_trajs", "confidences"})
        output["pred_trajs"].zero_()
        output["confidences"].zero_()
        self.assertTrue(torch.equal(agent._last_cvoi_planner_output["pred_trajs"], candidates))
        self.assertTrue(torch.equal(agent._last_cvoi_planner_output["confidences"], confidences))

    def test_navsim_trajectory_output_is_padded_to_sampling_length(self):
        agent_module = _load_navsim_agent()
        trajectory = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 2.0, 0.3]]])

        normalized = agent_module._normalize_navsim_trajectory_length(trajectory)

        self.assertEqual(tuple(normalized.shape), (1, 8, 3))
        torch.testing.assert_close(normalized[:, :2], trajectory)
        expected_tail = trajectory[:, -1:, :].expand(-1, 6, -1)
        torch.testing.assert_close(normalized[:, 2:], expected_tail)

    def test_navsim_trajectory_output_is_trimmed_to_sampling_length(self):
        agent_module = _load_navsim_agent()
        trajectory = torch.arange(30, dtype=torch.float32).reshape(1, 10, 3)

        normalized = agent_module._normalize_navsim_trajectory_length(trajectory)

        self.assertEqual(tuple(normalized.shape), (1, 8, 3))
        torch.testing.assert_close(normalized, trajectory[:, :8])

    def test_stage3_legacy_refiner_checkpoint_sets_transformer_core_type(self):
        agent_module = _load_navsim_agent()
        cfg = SimpleNamespace(planner=SimpleNamespace(refinement_core_type=None))
        checkpoint = {
            "planner": {
                "refine_core.transformer.encoder.layers.0.self_attn.in_proj_weight": torch.empty(1),
            }
        }

        changed = agent_module._maybe_set_legacy_stage3_refinement_core_type(cfg, checkpoint)

        self.assertTrue(changed)
        self.assertEqual(cfg.planner.refinement_core_type, "transformer")

    def test_stage12_extracts_transformer_proposal_core_planner_state(self):
        agent_module = _load_navsim_agent()
        cfg = SimpleNamespace(proposal=SimpleNamespace(enabled=True, provider_type="transformer"))
        query_weight = torch.empty(1)
        planner_state = {
            "core.query_embedding.weight": query_weight,
            "core.transformer.encoder.layers.0.self_attn.in_proj_weight": torch.empty(1),
        }

        extracted = agent_module._extract_proposal_core_planner_state(planner_state, cfg)

        self.assertIsNotNone(extracted)
        self.assertIs(extracted["query_embedding.weight"], query_weight)
        self.assertIn("transformer.encoder.layers.0.self_attn.in_proj_weight", extracted)

    def test_stage12_proposal_core_extraction_ignores_normal_planner_config(self):
        agent_module = _load_navsim_agent()
        cfg = SimpleNamespace(proposal=SimpleNamespace(enabled=False, provider_type="transformer"))
        planner_state = {"core.query_embedding.weight": torch.empty(1)}

        extracted = agent_module._extract_proposal_core_planner_state(planner_state, cfg)

        self.assertIsNone(extracted)

    def test_stage12_proposal_core_extraction_rejects_mixed_planner_keys(self):
        agent_module = _load_navsim_agent()
        cfg = SimpleNamespace(proposal=SimpleNamespace(enabled=True, provider_type="transformer"))
        planner_state = {
            "core.query_embedding.weight": torch.empty(1),
            "query_embedding.weight": torch.empty(1),
        }

        with self.assertRaisesRegex(RuntimeError, "Refusing ambiguous stage12 planner load"):
            agent_module._extract_proposal_core_planner_state(planner_state, cfg)

    def test_stage12_proposal_core_checkpoint_rebuilds_provider_core(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_init_source = source.split("def _initialize_stage12", 1)[1].split("def _build_multiview_fusion", 1)[0]
        stage12_forward_source = source.split("def _forward_stage12", 1)[1].split("def _forward_stage2", 1)[0]

        self.assertIn("def _build_stage12_proposal_core_planner", source)
        self.assertIn("build_proposal_provider(", source)
        self.assertIn('proposal_core = getattr(proposal_planner, "core", None)', source)
        self.assertIn("self._build_stage12_proposal_core_planner(", stage12_init_source)
        self.assertIn("self._stage12_planner_uses_full_context = True", stage12_init_source)
        self.assertIn(
            'planner_context_steps = int(getattr(active_planner, "num_context_frames", 1))', stage12_forward_source
        )

    def test_stage3_vjepa_main_encoder_uses_vjepa_main_clip(self):
        agent_cls = _load_navsim_agent()
        agent = agent_cls(forward_mode="stage3", crop_size=32, device="cpu")
        agent._config = SimpleNamespace(
            model=SimpleNamespace(
                backbone="vjepa_img_encoder",
                vjepa_resolution=(32, 64),
                vjepa_crop_top_bottom=8,
            )
        )

        builder = agent._build_feature_builder()
        features = builder.compute_features(_make_agent_input(num_frames=4))

        self.assertEqual(tuple(features["video_clip"].shape), (3, 4, 32, 64))

    def test_encoder_direct_forward_mode_uses_dedicated_path(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()

        self.assertIn('"encoder_direct"', source)
        self.assertIn("def _initialize_encoder_direct", source)
        self.assertIn("def _forward_encoder_direct", source)
        self.assertIn('self._forward_mode == "encoder_direct"', source)

    def test_stage2_forward_mode_uses_staged_refinement_path(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()

        self.assertIn('"stage2"', source)
        self.assertIn("def _initialize_stage2", source)
        self.assertIn("def _forward_stage2", source)
        self.assertIn("run_stage2_refinement_for_validation", source)
        self.assertIn('stage="stage2"', source)

    def test_stage12_loads_predictor_checkpoint_before_main_checkpoint(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_source = source.split("def _initialize_stage12", 1)[1].split("def _build_multiview_fusion", 1)[0]

        self.assertIn("load_pretrained_checkpoint(", stage12_source)
        self.assertIn("predictor_checkpoint=self._config.meta.predictor_checkpoint", stage12_source)
        self.assertLess(
            stage12_source.index("load_pretrained_checkpoint("),
            stage12_source.index("torch.load(self.checkpoint_path"),
        )
        self.assertIn("Stage12 main checkpoint has no predictor; using pretrained predictor", stage12_source)

    def test_stage12_registers_parallel_predictor_tokens_before_loading_checkpoint(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_source = source.split("def _initialize_stage12", 1)[1].split("def _build_multiview_fusion", 1)[0]

        self.assertIn("maybe_register_parallel_predictor_tokens(", stage12_source)
        self.assertLess(
            stage12_source.index("maybe_register_parallel_predictor_tokens("),
            stage12_source.index("load_pretrained_checkpoint("),
        )

    def test_stage12_uses_token_ae_predictor_runtime(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_init_source = source.split("def _initialize_stage12", 1)[1].split("def _build_multiview_fusion", 1)[0]
        stage12_forward_source = source.split("def _forward_stage12", 1)[1].split("def _forward_stage2", 1)[0]

        self.assertIn("init_predictor_runtime_with_token_ae(", stage12_init_source)
        self.assertIn("self._token_ae = token_ae", stage12_init_source)
        self.assertIn("self._runtime_normalize_reps = runtime_normalize_reps", stage12_init_source)
        self.assertIn("token_ae=self._token_ae", stage12_forward_source)
        self.assertIn("runtime_normalize_reps=self._runtime_normalize_reps", stage12_forward_source)

    def test_stage12_forward_uses_parallel_predictor_path_when_configured(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_source = source.split("def _forward_stage12", 1)[1].split("def _forward_stage2", 1)[0]

        self.assertIn("use_parallel_predictor(self._config)", stage12_source)
        self.assertIn("build_parallel_predictor_timeline_inputs(", stage12_source)
        self.assertIn("forward_parallel_predictor(", stage12_source)

    def test_stage12_inference_consistent_states_honor_drive_command_flags(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        predictor_ar_source = source.split("def _run_predictor_ar", 1)[1].split("def _build_status_feature", 1)[0]
        status_source = source.split("def _build_status_feature", 1)[1].split("def _forward_stage2", 1)[0]

        self.assertIn("make_predictor_step_fn(", predictor_ar_source)
        self.assertIn("driving_command=driving_command", predictor_ar_source)
        self.assertIn("ego_dynamics=ego_dynamics", predictor_ar_source)
        self.assertIn("use_drive_command=resolve_planner_use_drive_command(self._config)", status_source)

    def test_stage12_aligns_predictor_inputs_to_main_encoder_timeline(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_source = source.split("def _forward_stage12", 1)[1].split("def _forward_stage2", 1)[0]
        predictor_ar_source = source.split("def _run_predictor_ar", 1)[1].split("def _build_status_feature", 1)[0]

        self.assertIn("build_predictor_timeline_inputs(", stage12_source)
        self.assertIn("predictor_inputs.num_observed_steps", stage12_source)
        self.assertIn("predictor_inputs.num_future_steps", stage12_source)
        self.assertIn("num_observed_steps", predictor_ar_source)
        self.assertIn("num_future_steps", predictor_ar_source)
        self.assertIn("predictor_inputs.num_observed_steps * tokens_per_frame", stage12_source)

    def test_stage12_passes_action_history_when_planner_requires_it(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_source = source.split("def _forward_stage12", 1)[1].split("def _forward_stage2", 1)[0]

        self.assertIn("build_observed_action_trajectory_history(", stage12_source)
        self.assertIn("use_action_history_for_planner", stage12_source)
        self.assertIn("predictor_inputs.actions", stage12_source)
        self.assertIn("num_observed_frames=predictor_inputs.num_observed_steps", stage12_source)
        self.assertIn("action_history=planner_action_history", stage12_source)

    def test_stage12_cvoi_diffusion_uses_shared_observed_velocity_anchor(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_source = source.split("def _forward_stage12", 1)[1].split("def _forward_stage2", 1)[0]

        self.assertIn("build_ego_relative_diffusion_anchor(", stage12_source)
        self.assertIn("ego_dynamics=ego_dynamics", stage12_source)
        self.assertIn("observed_frames=self._config.train.num_observed_frames", stage12_source)
        self.assertIn('planner_kwargs["anchor_state"] = planner_anchor_state', stage12_source)

    def test_stage12_value_planning_rollout_uses_raw_timeline_inputs(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()
        stage12_source = source.split("def _forward_stage12", 1)[1].split("def _forward_stage2", 1)[0]
        call_start = stage12_source.index("score_trajectories_method1(")
        call_end = stage12_source.index("confidences=pred_conf.float()", call_start)
        value_call = stage12_source[call_start:call_end]

        self.assertIn("actions=actions", value_call)
        self.assertIn("states=states", value_call)
        self.assertIn("driving_command=driving_command", value_call)
        self.assertIn("ego_dynamics=ego_dynamics", value_call)

    def test_stage3_predictor_rollout_is_truncated_like_validation(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()

        self.assertIn("build_stage_predictor_rollout_fn", source)
        self.assertIn('stage="stage3"', source)

    def test_stage3_forward_uses_proposal_video_clip_and_deterministic_rng(self):
        source = (REPO_ROOT / "app/vjepa_cowa_world_model/evaluation/navsim_agent.py").read_text()

        self.assertIn("proposal_video_clip", source)
        self.assertIn("_deterministic_navsim_eval_rng", source)


if __name__ == "__main__":
    unittest.main()
