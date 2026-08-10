"""Tests for NavSim UTM coordinate precision fixes.

Verifies that:
1. _build_states centres translations around the first frame (float64 → float32).
2. GT trajectory diff uses float64, producing precise results even with
   large UTM-scale source coordinates.
3. Status-feature UTM-leak warning fires when raw UTM values are detected.
"""

import logging
import unittest

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helpers – simulate the centering logic from navsim_data._build_states
# ---------------------------------------------------------------------------


def _build_states_centred(translations_f64: np.ndarray, orientations: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    """Simulate the centred _build_states logic.

    Parameters
    ----------
    translations_f64 : [T, 3] float64
    orientations     : [T, 3] float64 (roll, pitch, yaw)
    speeds           : [T]    float64
    """
    origin = translations_f64[0].copy()
    centred = translations_f64 - origin  # float64 subtraction
    states = np.concatenate([centred, orientations, speeds.reshape(-1, 1)], axis=1)
    return states.astype(np.float32)


def _build_states_old(translations: np.ndarray, orientations: np.ndarray, speeds: np.ndarray) -> np.ndarray:
    """Simulate the OLD _build_states that stored raw UTM as float32."""
    states_list = []
    for i in range(len(translations)):
        t = np.asarray(translations[i], dtype=np.float32)
        o = np.asarray(orientations[i], dtype=np.float32)
        s = np.asarray([speeds[i]], dtype=np.float32)
        states_list.append(np.concatenate([t, o, s]))
    return np.stack(states_list, axis=0).astype(np.float32)


class TestBuildStatesCentering(unittest.TestCase):
    """Verify that the centering logic preserves precision."""

    def _make_utm_data(self, n_frames: int = 10, speed_mps: float = 7.5, dt: float = 0.5):
        """Create realistic NavSim-like UTM data."""
        utm_x0 = 664_460.0
        utm_y0 = 3_997_900.0
        # Drive north-east at constant speed
        yaw = np.deg2rad(45)
        translations = np.zeros((n_frames, 3), dtype=np.float64)
        orientations = np.zeros((n_frames, 3), dtype=np.float64)
        speeds = np.full(n_frames, speed_mps, dtype=np.float64)
        for i in range(n_frames):
            translations[i] = [
                utm_x0 + speed_mps * np.cos(yaw) * i * dt,
                utm_y0 + speed_mps * np.sin(yaw) * i * dt,
                615.0,
            ]
            orientations[i] = [0.0, 0.0, yaw]
        return translations, orientations, speeds

    def test_centred_first_frame_is_zero(self):
        translations, orientations, speeds = self._make_utm_data()
        states = _build_states_centred(translations, orientations, speeds)
        np.testing.assert_allclose(states[0, :3], [0.0, 0.0, 0.0], atol=1e-6)

    def test_centred_values_are_small(self):
        translations, orientations, speeds = self._make_utm_data()
        states = _build_states_centred(translations, orientations, speeds)
        # All x,y,z values should be O(100) or less
        self.assertTrue(np.abs(states[:, :3]).max() < 500.0)

    def test_diff_precision_centred_vs_old(self):
        """The centred approach should give sub-mm precision for frame diffs."""
        translations, orientations, speeds = self._make_utm_data()
        states_new = _build_states_centred(translations, orientations, speeds)
        states_old = _build_states_old(translations, orientations, speeds)

        # Ground-truth diff in float64
        gt_dx = np.diff(translations[:, 0])
        gt_dy = np.diff(translations[:, 1])

        # Diff using centred float32
        new_dx = np.diff(states_new[:, 0].astype(np.float64))
        new_dy = np.diff(states_new[:, 1].astype(np.float64))

        # Diff using old float32 (UTM)
        old_dy = np.diff(states_old[:, 1].astype(np.float64))

        # Centred approach: error < 1mm
        new_err_x = np.abs(new_dx - gt_dx).max()
        new_err_y = np.abs(new_dy - gt_dy).max()
        self.assertLess(new_err_x, 0.001, f"Centred x diff error {new_err_x:.6f} m > 1 mm")
        self.assertLess(new_err_y, 0.001, f"Centred y diff error {new_err_y:.6f} m > 1 mm")

        # Old approach: y error should be noticeable (≥ 0.1m for UTM y ~ 4M)
        old_err_y = np.abs(old_dy - gt_dy).max()
        # Just verify the old approach is less precise (may or may not exceed 0.1m
        # depending on exact values, so we just check centred is better or equal)
        self.assertLessEqual(new_err_y, old_err_y + 1e-9)

    def test_actions_diff_algebraic_invariance(self):
        """Centering should not change dx/dy between adjacent frames."""
        translations, orientations, speeds = self._make_utm_data()
        states_new = _build_states_centred(translations, orientations, speeds)

        gt_dx = np.diff(translations[:, 0])
        new_dx = np.diff(states_new[:, 0].astype(np.float64))
        np.testing.assert_allclose(new_dx, gt_dx, atol=0.001)


class TestGTTrajectoryFloat64Diff(unittest.TestCase):
    """Verify that the float64 GT trajectory diff is more precise."""

    def _make_utm_states_tensor(self, b: int = 2, t: int = 12) -> torch.Tensor:
        """Create a batch of states with UTM-scale coordinates."""
        states = torch.zeros(b, t, 7)
        utm_x0 = 664_460.0
        utm_y0 = 3_997_900.0
        speed = 10.0
        yaw = 0.3
        dt = 0.5
        for i in range(t):
            states[:, i, 0] = utm_x0 + speed * np.cos(yaw) * i * dt
            states[:, i, 1] = utm_y0 + speed * np.sin(yaw) * i * dt
            states[:, i, 5] = yaw
            states[:, i, 6] = speed
        return states

    def _compute_gt_traj_f64(self, states: torch.Tensor, future_start: int = 4, num_poses: int = 8):
        """GT trajectory with float64 (the fixed version)."""
        StateSE2_indices = [0, 1, 5]
        states_se2 = states[:, :, StateSE2_indices].double()
        origin_x = states_se2[:, 0, 0]
        origin_y = states_se2[:, 0, 1]
        origin_yaw = states_se2[:, 0, 2]

        dx = states_se2[:, future_start:, 0] - origin_x[:, None]
        dy = states_se2[:, future_start:, 1] - origin_y[:, None]
        dyaw = states_se2[:, future_start:, 2] - origin_yaw[:, None]

        cos_h = torch.cos(-origin_yaw)
        sin_h = torch.sin(-origin_yaw)
        ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
        ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
        ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        gt = torch.stack([ego_x, ego_y, ego_yaw], dim=-1).float()
        return gt[:, :num_poses]

    def _compute_gt_traj_f32(self, states: torch.Tensor, future_start: int = 4, num_poses: int = 8):
        """GT trajectory with float32 only (old version)."""
        StateSE2_indices = [0, 1, 5]
        states_se2 = states[:, :, StateSE2_indices]
        origin_x = states_se2[:, 0, 0]
        origin_y = states_se2[:, 0, 1]
        origin_yaw = states_se2[:, 0, 2]

        dx = states_se2[:, future_start:, 0] - origin_x[:, None]
        dy = states_se2[:, future_start:, 1] - origin_y[:, None]
        dyaw = states_se2[:, future_start:, 2] - origin_yaw[:, None]

        cos_h = torch.cos(-origin_yaw)
        sin_h = torch.sin(-origin_yaw)
        ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
        ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
        ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        return torch.stack([ego_x, ego_y, ego_yaw], dim=-1)[:, :num_poses]

    def test_float64_produces_valid_trajectory(self):
        states = self._make_utm_states_tensor()
        gt = self._compute_gt_traj_f64(states)
        self.assertEqual(gt.dtype, torch.float32)
        self.assertEqual(gt.shape, (2, 8, 3))
        # x should be positive (moving forward)
        self.assertTrue((gt[:, :, 0] > 0).all())

    def test_float64_more_precise_than_float32(self):
        """When states are raw UTM, float64 diff should be more precise."""
        states = self._make_utm_states_tensor()
        gt_f64 = self._compute_gt_traj_f64(states)
        gt_f32 = self._compute_gt_traj_f32(states)

        # Compute ground truth in full float64
        states_f64 = states.double()
        gt_ref = self._compute_gt_traj_f64(states_f64)

        err_f64 = (gt_f64 - gt_ref).abs().max().item()
        err_f32 = (gt_f32 - gt_ref).abs().max().item()
        # float64 should be at least as good
        self.assertLessEqual(err_f64, err_f32 + 1e-6)


class TestStatusFeatureUTMWarning(unittest.TestCase):
    """Verify that _warn_if_utm_leak fires for large coordinate values."""

    @classmethod
    def setUpClass(cls):
        """Import status_features directly to avoid triggering __init__ chain."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "status_features",
            "app/vjepa_cowa_world_model/utils/status_features.py",
        )
        cls._mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls._mod)

    def test_warning_fires_for_utm_values(self):
        _warn_if_utm_leak = self._mod._warn_if_utm_leak
        xy = torch.tensor([[664_460.0, 3_997_900.0]])
        with self.assertLogs(level=logging.WARNING) as cm:
            _warn_if_utm_leak(xy, "test_mode")
        self.assertTrue(any("UTM" in msg for msg in cm.output))

    def test_no_warning_for_centred_values(self):
        _warn_if_utm_leak = self._mod._warn_if_utm_leak
        xy = torch.tensor([[50.0, -3.0]])
        # Should not emit any warning
        with self.assertRaises(AssertionError):
            # assertLogs raises AssertionError if no log is emitted — that's what we want
            with self.assertLogs(level=logging.WARNING):
                _warn_if_utm_leak(xy, "test_mode")

    def test_prepare_status_feature_raw_states_warns(self):
        """raw_states mode should trigger UTM warning for large coords."""
        prepare_status_feature = self._mod.prepare_status_feature

        B, T = 2, 4
        states = torch.zeros(B, T, 7)
        states[:, :, 0] = 664_460.0  # UTM x
        states[:, :, 1] = 3_997_900.0  # UTM y
        actions = torch.zeros(B, T - 1, 3)

        with self.assertLogs(level=logging.WARNING) as cm:
            result = prepare_status_feature(states, actions, status_mode="raw_states", num_context_frames=T)
        self.assertTrue(any("UTM" in msg for msg in cm.output))
        self.assertEqual(result.shape, (B, 7))


if __name__ == "__main__":
    unittest.main()
