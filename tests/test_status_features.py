"""Tests for mask_future_actions in status_features module."""

import importlib.util
import unittest
from types import SimpleNamespace

import torch


def _load_mask_future_actions():
    """直接加载 status_features 模块，避免 __init__ 依赖链。"""
    spec = importlib.util.spec_from_file_location(
        "status_features",
        "app/vjepa_cowa_world_model/utils/status_features.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.mask_future_actions


mask_future_actions = _load_mask_future_actions()


class TestMaskFutureActions(unittest.TestCase):
    """mask_future_actions 返回 (masked_actions, action_mask) 的单元测试。"""

    def setUp(self):
        self.B, self.T, self.D = 2, 5, 3
        self.actions = torch.randn(self.B, self.T, self.D)

    def test_returns_tuple(self):
        """验证返回值是长度为 2 的 tuple。"""
        result = mask_future_actions(self.actions, num_known_actions=3)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_mask_values_correct(self):
        """验证 mask 中 True/False 位置与已知/未知 action 对应。"""
        num_known = 3
        masked_actions, action_mask = mask_future_actions(self.actions, num_known)

        # 前 num_known 个位置: mask=False (已知), action 值保持
        self.assertFalse(action_mask[:, :num_known].any().item())
        torch.testing.assert_close(masked_actions[:, :num_known], self.actions[:, :num_known])

        # 后续位置: mask=True (未知), action 值为 0
        self.assertTrue(action_mask[:, num_known:].all().item())
        self.assertTrue((masked_actions[:, num_known:] == 0).all().item())

    def test_zero_known(self):
        """num_known_actions=0 时，所有位置都应被 mask。"""
        masked_actions, action_mask = mask_future_actions(self.actions, num_known_actions=0)

        self.assertTrue(action_mask.all().item())
        self.assertTrue((masked_actions == 0).all().item())

    def test_all_known(self):
        """num_known_actions=T 时，没有位置被 mask。"""
        masked_actions, action_mask = mask_future_actions(self.actions, num_known_actions=self.T)

        self.assertFalse(action_mask.any().item())
        torch.testing.assert_close(masked_actions, self.actions)

    def test_backward_compat_unpack(self):
        """确保结果可以通过 len() == 2 来解包。"""
        result = mask_future_actions(self.actions, num_known_actions=2)
        self.assertEqual(len(result), 2)
        masked_actions, action_mask = result
        self.assertEqual(masked_actions.shape, self.actions.shape)
        self.assertEqual(action_mask.shape, (self.B, self.T))
        self.assertEqual(action_mask.dtype, torch.bool)


def _load_status_module():
    """加载 status_features 模块。"""
    spec = importlib.util.spec_from_file_location(
        "status_features",
        "app/vjepa_cowa_world_model/utils/status_features.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_status_mod = _load_status_module()


class TestUtilsReexportsActionHistory(unittest.TestCase):
    """回归测试：utils 包应导出 observed action history helper。"""

    def test_utils_reexports_observed_action_history_builder(self):
        utils_mod = importlib.import_module("app.vjepa_cowa_world_model.utils")

        actions = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )

        history = utils_mod.build_observed_action_trajectory_history(actions, num_observed_frames=3)

        self.assertEqual(history.shape, (1, 3, 3))
        torch.testing.assert_close(history[0, -1], torch.zeros(3, dtype=history.dtype))


class TestObservedActionHistoryDimensions(unittest.TestCase):
    """Tests for configurable observed action history feature dimensions."""

    def test_history_dim_4_uses_cos_sin_yaw(self):
        actions = torch.zeros(1, 2, 3, dtype=torch.float32)

        history = _status_mod.build_observed_action_trajectory_history(
            actions,
            num_observed_frames=3,
            action_history_dim=4,
        )

        self.assertEqual(history.shape, (1, 3, 4))
        torch.testing.assert_close(history[0, -1], torch.tensor([0.0, 0.0, 1.0, 0.0]))

    def test_history_dim_6_adds_velocity_and_cos_sin_yaw(self):
        actions = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )

        history = _status_mod.build_observed_action_trajectory_history(
            actions,
            num_observed_frames=3,
            action_history_dim=6,
            dt=0.5,
        )

        self.assertEqual(history.shape, (1, 3, 6))
        torch.testing.assert_close(history[0, 0], torch.tensor([-2.0, 0.0, 2.0, 0.0, 1.0, 0.0]))
        torch.testing.assert_close(history[0, -1], torch.tensor([0.0, 0.0, 2.0, 0.0, 1.0, 0.0]))

    def test_history_dim_rejects_unsupported_values(self):
        actions = torch.zeros(1, 1, 3, dtype=torch.float32)

        with self.assertRaises(ValueError):
            _status_mod.build_observed_action_trajectory_history(
                actions,
                num_observed_frames=2,
                action_history_dim=5,
            )


class TestDriveCommand8(unittest.TestCase):
    """Tests for the 8-dim IC state vector."""

    def setUp(self):
        self.B, self.T = 4, 10
        # states: [B, T, 7] = [x, y, z, roll, pitch, yaw, velocity]
        self.states = torch.randn(self.B, self.T, 7)
        # Make x, y have increasing values for consistent test data
        self.states[:, :, 0] = torch.linspace(0, 10, self.T).unsqueeze(0).expand(self.B, -1)
        self.states[:, :, 1] = torch.linspace(0, 5, self.T).unsqueeze(0).expand(self.B, -1)
        self.driving_command = torch.zeros(self.B, 4)
        self.driving_command[:, 0] = 1.0  # GO_STRAIGHT
        self.ego_dynamics = torch.randn(self.B, self.T, 4)

    def test_output_shape(self):
        result = _status_mod._prepare_drive_command_8(
            self.states, num_observed=2, driving_command=self.driving_command, ego_dynamics=self.ego_dynamics
        )
        self.assertEqual(result.shape, (self.B, 8))

    def test_drive_command_prefix(self):
        """First 4 dims should be drive command one-hot."""
        result = _status_mod._prepare_drive_command_8(
            self.states, num_observed=2, driving_command=self.driving_command, ego_dynamics=self.ego_dynamics
        )
        torch.testing.assert_close(result[:, :4], self.driving_command.to(result.device))

    def test_ego_dynamics_used(self):
        """vx, vy, ax, ay should come from ego_dynamics when provided."""
        result = _status_mod._prepare_drive_command_8(
            self.states,
            num_observed=2,
            driving_command=self.driving_command,
            ego_dynamics=self.ego_dynamics,
        )
        obs_idx = 1  # num_observed - 1
        torch.testing.assert_close(result[:, 4], self.ego_dynamics[:, obs_idx, 0])  # vx
        torch.testing.assert_close(result[:, 5], self.ego_dynamics[:, obs_idx, 1])  # vy
        torch.testing.assert_close(result[:, 6], self.ego_dynamics[:, obs_idx, 2])  # ax
        torch.testing.assert_close(result[:, 7], self.ego_dynamics[:, obs_idx, 3])  # ay

    def test_raises_without_ego_dynamics(self):
        """fail-loud (point 4): 缺 ego_dynamics 时 8 维 status 直接报错，不再伪造 vy=ay=0。"""
        with self.assertRaises(ValueError):
            _status_mod._prepare_drive_command_8(self.states, num_observed=2)

    def test_raises_without_driving_command_for_8_dim(self):
        """fail-loud: 8 维 status 不得用全序列 yaw 伪造 drive command。"""
        with self.assertRaises(ValueError):
            _status_mod._prepare_drive_command_8(
                self.states,
                num_observed=2,
                ego_dynamics=self.ego_dynamics,
                driving_command=None,
                use_drive_command=True,
            )

    def test_raises_without_driving_command_for_12_dim(self):
        """fail-loud: 12 维 status 继承 8 维 command 语义，缺 command 也必须报错。"""
        with self.assertRaises(ValueError):
            _status_mod._prepare_drive_command_12(
                self.states,
                num_observed=2,
                ego_dynamics=self.ego_dynamics,
                driving_command=None,
                use_drive_command=True,
            )

    def test_ic_states_8_shape(self):
        """prepare_inference_consistent_states with state_dim=8 should return [B, T, 8]."""
        result = _status_mod.prepare_inference_consistent_states(
            self.states,
            num_observed=2,
            state_dim=8,
            driving_command=self.driving_command,
            ego_dynamics=self.ego_dynamics,
        )
        self.assertEqual(result.shape, (self.B, self.T, 8))

    def test_ic_states_7_backward_compat(self):
        """state_dim=7 + use_drive_command 默认 True 时需提供 driving_command（point 3），返回 [B,T,7]。"""
        result = _status_mod.prepare_inference_consistent_states(
            self.states, num_observed=2, driving_command=self.driving_command, ego_dynamics=self.ego_dynamics
        )
        self.assertEqual(result.shape, (self.B, self.T, 7))

    def test_ic_states_7_raises_without_driving_command(self):
        """fail-loud (point 3): state_dim=7 且 use_drive_command=True 时缺 driving_command 直接报错。"""
        with self.assertRaises(ValueError):
            _status_mod.prepare_inference_consistent_states(self.states, num_observed=2)

    def test_ic_status_vector_8_shape(self):
        """prepare_inference_consistent_status_vector with state_dim=8 should return [B, 8]."""
        result = _status_mod.prepare_inference_consistent_status_vector(
            self.states,
            num_observed=2,
            state_dim=8,
            driving_command=self.driving_command,
            ego_dynamics=self.ego_dynamics,
        )
        self.assertEqual(result.shape, (self.B, 8))

    def test_get_status_dim_8(self):
        """get_status_dim('drive_command_8') should return 8."""
        self.assertEqual(_status_mod.get_status_dim("drive_command_8"), 8)


class TestICConsistency(unittest.TestCase):
    """Verify IC functions produce identical output with and without driving_command/ego_dynamics."""

    def setUp(self):
        self.B, self.T = 2, 10
        self.states = torch.randn(self.B, self.T, 7)
        self.states[:, :, 0] = torch.linspace(0, 10, self.T).unsqueeze(0).expand(self.B, -1)
        self.states[:, :, 1] = torch.linspace(0, 5, self.T).unsqueeze(0).expand(self.B, -1)
        self.driving_command = torch.zeros(self.B, 4)
        self.driving_command[:, 1] = 1.0  # TURN_LEFT
        self.ego_dynamics = torch.randn(self.B, self.T, 4)

    def test_ic_states_8_requires_ego_dynamics(self):
        """state_dim=8: 提供 driving_command/ego_dynamics 返回 [B,T,8]；缺 ego_dynamics 直接报错（point 4）。"""
        result_with = _status_mod.prepare_inference_consistent_states(
            self.states,
            num_observed=2,
            state_dim=8,
            driving_command=self.driving_command,
            ego_dynamics=self.ego_dynamics,
        )
        self.assertEqual(result_with.shape, (self.B, self.T, 8))
        # fail-loud: 缺 ego_dynamics 时不再伪造 vy/ay，直接报错。
        with self.assertRaises(ValueError):
            _status_mod.prepare_inference_consistent_states(
                self.states, num_observed=2, state_dim=8, driving_command=self.driving_command, ego_dynamics=None
            )

    def test_ic_status_vector_with_dc_ed(self):
        """IC status vector with state_dim=8 and driving_command should have correct prefix."""
        result = _status_mod.prepare_inference_consistent_status_vector(
            self.states,
            num_observed=2,
            state_dim=8,
            driving_command=self.driving_command,
            ego_dynamics=self.ego_dynamics,
        )
        self.assertEqual(result.shape, (self.B, 8))
        # First 4 dims should be the driving command
        torch.testing.assert_close(result[:, :4], self.driving_command)

    def test_ic_status_vector_7dim_backward_compat(self):
        """state_dim=7 should produce [B,7] regardless of driving_command."""
        result = _status_mod.prepare_inference_consistent_status_vector(
            self.states,
            num_observed=2,
            state_dim=7,
            driving_command=self.driving_command,
            ego_dynamics=self.ego_dynamics,
        )
        self.assertEqual(result.shape, (self.B, 7))


class TestDriveCommandFrameSelection(unittest.TestCase):
    """3D driving_command 必须取当前帧（num_observed-1），不得取窗口首帧（滞后指令）。"""

    def setUp(self):
        self.B, self.T = 2, 6
        self.states = torch.randn(self.B, self.T, 7)
        self.ego_dynamics = torch.randn(self.B, self.T, 4)
        # 每帧不同的 one-hot 指令: frame 0 = GO_STRAIGHT, frame 2 = TURN_LEFT, ...
        self.driving_command = torch.zeros(self.B, self.T, 4)
        self.driving_command[:, 0, 0] = 1.0  # frame 0: GO_STRAIGHT
        self.driving_command[:, 1, 0] = 1.0
        self.driving_command[:, 2, 1] = 1.0  # frame 2: TURN_LEFT
        self.driving_command[:, 3:, 2] = 1.0  # frame 3+: TURN_RIGHT

    def test_7dim_uses_current_observed_frame(self):
        result = _status_mod._prepare_drive_command_7(
            self.states,
            num_observed=3,
            driving_command=self.driving_command,
            ego_dynamics=self.ego_dynamics,
        )
        # num_observed=3 → 当前帧 idx=2 → TURN_LEFT，而非 frame 0 的 GO_STRAIGHT
        torch.testing.assert_close(result[:, :4], self.driving_command[:, 2, :])

    def test_8dim_uses_current_observed_frame(self):
        result = _status_mod._prepare_drive_command_8(
            self.states,
            num_observed=3,
            driving_command=self.driving_command,
            ego_dynamics=self.ego_dynamics,
        )
        torch.testing.assert_close(result[:, :4], self.driving_command[:, 2, :])

    def test_cmd_index_clamped_to_available_frames(self):
        """driving_command 帧数不足时 clamp 到最后一帧（与 ego_dynamics 取帧惯例一致）。"""
        short_cmd = self.driving_command[:, :2]  # 只有 2 帧
        result = _status_mod._prepare_drive_command_7(
            self.states,
            num_observed=5,
            driving_command=short_cmd,
            ego_dynamics=self.ego_dynamics,
        )
        torch.testing.assert_close(result[:, :4], short_cmd[:, 1, :])

    def test_2d_command_unchanged(self):
        cmd_2d = torch.zeros(self.B, 4)
        cmd_2d[:, 3] = 1.0
        result = _status_mod._prepare_drive_command_7(
            self.states,
            num_observed=3,
            driving_command=cmd_2d,
            ego_dynamics=self.ego_dynamics,
        )
        torch.testing.assert_close(result[:, :4], cmd_2d)


class TestResolvePlannerStatusDim(unittest.TestCase):
    def test_inherits_train_drive_command_when_planner_flag_missing(self):
        cfg = SimpleNamespace(
            train=SimpleNamespace(state_dim=8, use_drive_command=False),
            planner=SimpleNamespace(status_dim=0, use_drive_command=None),
        )

        self.assertFalse(_status_mod.resolve_planner_use_drive_command(cfg))
        self.assertEqual(_status_mod.resolve_effective_planner_status_dim(cfg), 4)

    def test_planner_drive_command_can_override_train_flag(self):
        cfg = SimpleNamespace(
            train=SimpleNamespace(state_dim=8, use_drive_command=False),
            planner=SimpleNamespace(status_dim=0, use_drive_command=True),
        )

        self.assertTrue(_status_mod.resolve_planner_use_drive_command(cfg))
        self.assertEqual(_status_mod.resolve_effective_planner_status_dim(cfg), 8)

    def test_strips_drive_command_dims_when_disabled(self):
        cfg = SimpleNamespace(
            train=SimpleNamespace(state_dim=8, use_drive_command=False),
            planner=SimpleNamespace(status_dim=0, use_drive_command=None),
        )

        self.assertEqual(_status_mod.resolve_effective_planner_status_dim(cfg), 4)


def _load_planner_module():
    """直接加载 multimodal_planner 模块，避免 app 全量初始化。"""

    spec = importlib.util.spec_from_file_location(
        "multimodal_planner",
        "app/vjepa_cowa_world_model/models/multimodal_planner.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_planner_mod = _load_planner_module()


class TestSplitStatusEmbedding(unittest.TestCase):
    """验证 MultiModalTemporalPlanner 在 command_dim > 0 时正确拆分 status 嵌入。"""

    def _make_planner(self, status_dim, command_dim, use_temporal=True, use_status=True):
        """创建一个轻量级 planner 用于测试。"""
        return _planner_mod.MultiModalTemporalPlanner(
            encoder_dim=32,
            tf_d_model=16,
            tf_d_ffn=32,
            tf_num_layers=1,
            tf_num_head=2,
            tf_dropout=0.0,
            tokens_per_frame=4,
            num_poses=3,
            num_time_steps=3,
            status_dim=status_dim,
            use_spatial_tokens=False,
            num_modes=2,
            use_temporal=use_temporal,
            use_time_aligned_bias=False,
            use_status_for_planner=use_status,
            command_dim=command_dim,
        )

    def test_split_temporal_output_shape(self):
        """command_dim=4, status_dim=7: forward 应输出正确 shape。"""
        planner = self._make_planner(status_dim=7, command_dim=4, use_temporal=True)
        B = 2
        z_ar = torch.randn(B, 3 * 4, 32)  # num_time_steps * tokens_per_frame
        status = torch.randn(B, 7)
        out = planner(z_ar, status)
        self.assertEqual(out["trajectories"].shape, (B, 2, 3, 3))  # [B, K, P, 3]
        self.assertEqual(out["confidences"].shape, (B, 2))

    def test_split_temporal_output_shape_dim8(self):
        """command_dim=4, status_dim=8: forward 应输出正确 shape。"""
        planner = self._make_planner(status_dim=8, command_dim=4, use_temporal=True)
        B = 2
        z_ar = torch.randn(B, 3 * 4, 32)
        status = torch.randn(B, 8)
        out = planner(z_ar, status)
        self.assertEqual(out["trajectories"].shape, (B, 2, 3, 3))
        self.assertEqual(out["confidences"].shape, (B, 2))

    def test_split_single_frame_output_shape(self):
        """command_dim=4, use_temporal=False: 单帧模式也应正常工作。"""
        planner = self._make_planner(status_dim=8, command_dim=4, use_temporal=False)
        B = 2
        z_ar = torch.randn(B, 4, 32)  # tokens_per_frame
        status = torch.randn(B, 8)
        out = planner(z_ar, status)
        self.assertEqual(out["trajectories"].shape, (B, 2, 3, 3))

    def test_fallback_no_split(self):
        """command_dim=0: 旧行为，应存在 status_encoding 而非 command_encoding。"""
        planner = self._make_planner(status_dim=7, command_dim=0, use_temporal=True)
        self.assertTrue(hasattr(planner, "status_encoding"))
        self.assertFalse(hasattr(planner, "command_encoding"))
        self.assertFalse(hasattr(planner, "kinematics_encoding"))
        B = 2
        z_ar = torch.randn(B, 3 * 4, 32)
        status = torch.randn(B, 7)
        out = planner(z_ar, status)
        self.assertEqual(out["trajectories"].shape, (B, 2, 3, 3))

    def test_split_has_correct_modules(self):
        """command_dim > 0 时应有 command_encoding 和 kinematics_encoding，无 status_encoding。"""
        planner = self._make_planner(status_dim=8, command_dim=4, use_temporal=True)
        self.assertTrue(hasattr(planner, "command_encoding"))
        self.assertTrue(hasattr(planner, "kinematics_encoding"))
        self.assertFalse(hasattr(planner, "status_encoding"))

    def test_temporal_memory_token_count(self):
        """command_dim > 0 时，temporal memory 比 visual tokens 多 2 个 token。"""
        planner = self._make_planner(status_dim=7, command_dim=4, use_temporal=True)
        B = 2
        z_ar = torch.randn(B, 3 * 4, 32)
        status = torch.randn(B, 7)
        memory, step_idx = planner._build_memory_temporal(z_ar, status, num_steps=3)
        # use_spatial_tokens=False → 3 visual tokens + 2 status tokens = 5
        self.assertEqual(memory.shape[1], 5)
        self.assertEqual(step_idx.shape[0], 5)
        # 最后两个 step_idx 应为 -1 (不参与时间对齐)
        self.assertEqual(step_idx[-2].item(), -1)
        self.assertEqual(step_idx[-1].item(), -1)

    def test_temporal_memory_token_count_no_split(self):
        """command_dim=0 时，temporal memory 比 visual tokens 多 1 个 token。"""
        planner = self._make_planner(status_dim=7, command_dim=0, use_temporal=True)
        B = 2
        z_ar = torch.randn(B, 3 * 4, 32)
        status = torch.randn(B, 7)
        memory, step_idx = planner._build_memory_temporal(z_ar, status, num_steps=3)
        # 3 visual tokens + 1 status token = 4
        self.assertEqual(memory.shape[1], 4)
        self.assertEqual(step_idx.shape[0], 4)

    def test_kinematics_encoding_has_layernorm(self):
        """kinematics_encoding 第一个子模块应为 LayerNorm。"""
        planner = self._make_planner(status_dim=8, command_dim=4, use_temporal=True)
        first_layer = planner.kinematics_encoding[0]
        self.assertIsInstance(first_layer, torch.nn.LayerNorm)
        self.assertEqual(first_layer.normalized_shape[0], 4)  # status_dim - command_dim

    def test_gradient_flows_through_both_branches(self):
        """梯度应同时流过 command_encoding 和 kinematics_encoding。"""
        planner = self._make_planner(status_dim=8, command_dim=4, use_temporal=True)
        B = 2
        z_ar = torch.randn(B, 3 * 4, 32)
        status = torch.randn(B, 8, requires_grad=True)
        out = planner(z_ar, status)
        loss = out["trajectories"].sum() + out["confidences"].sum()
        loss.backward()
        # 检查 command_encoding 和 kinematics_encoding 的参数都有梯度
        for name, param in planner.named_parameters():
            if "command_encoding" in name or "kinematics_encoding" in name:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
                self.assertFalse(
                    torch.all(param.grad == 0).item(),
                    f"Zero gradient for {name}",
                )

    def test_temporal_memory_token_count_with_action_history(self):
        """新增 action history 后，planner memory 应显式增加历史轨迹 tokens。"""
        planner = _planner_mod.MultiModalTemporalPlanner(
            encoder_dim=32,
            tf_d_model=16,
            tf_d_ffn=32,
            tf_num_layers=1,
            tf_num_head=2,
            tf_dropout=0.0,
            tokens_per_frame=4,
            num_poses=3,
            num_time_steps=3,
            num_observed_frames=3,
            status_dim=7,
            use_spatial_tokens=False,
            num_modes=2,
            use_temporal=True,
            use_time_aligned_bias=False,
            use_status_for_planner=True,
            command_dim=0,
            use_action_history=True,
            action_history_dim=3,
        )
        B = 2
        z_ar = torch.randn(B, 3 * 4, 32)
        status = torch.randn(B, 7)
        action_history = torch.randn(B, 3, 3)

        memory, step_idx = planner._build_memory_temporal(
            z_ar,
            status,
            num_steps=3,
            action_history=action_history,
        )

        self.assertEqual(memory.shape[1], 7)
        self.assertEqual(step_idx.shape[0], 7)
        self.assertTrue(torch.equal(step_idx[-4:], torch.full((4,), -1, dtype=step_idx.dtype, device=step_idx.device)))

        out = planner(z_ar, status, action_history=action_history)
        self.assertEqual(out["trajectories"].shape, (B, 2, 3, 3))
        self.assertEqual(out["confidences"].shape, (B, 2))


if __name__ == "__main__":
    unittest.main()
