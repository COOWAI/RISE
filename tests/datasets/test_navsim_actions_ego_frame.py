"""Tests for NavSim _build_actions_3d ego-frame rotation.

Verifies that global-coordinate position differences are correctly rotated
into the ego frame of each timestep, so that dx ≈ forward and dy ≈ lateral
regardless of the vehicle's global heading.
"""

import unittest

import numpy as np


def _rotate_to_ego(dx_global: float, dy_global: float, yaw: float):
    """Rotate a global (dx, dy) into ego frame given current yaw."""
    cos_h = np.cos(-yaw)
    sin_h = np.sin(-yaw)
    dx_ego = cos_h * dx_global - sin_h * dy_global
    dy_ego = sin_h * dx_global + cos_h * dy_global
    return dx_ego, dy_ego


def _make_states(positions_yaws, speed: float = 10.0) -> np.ndarray:
    """Build a states array [T, 7] from list of (x, y, z, roll, pitch, yaw).

    Each element should be a tuple (x, y, z, roll, pitch, yaw).
    Speed is appended as the 7th dimension for all frames.
    """
    rows = []
    for x, y, z, roll, pitch, yaw in positions_yaws:
        rows.append([x, y, z, roll, pitch, yaw, speed])
    return np.array(rows, dtype=np.float32)


class TestNavSimActionsEgoFrame(unittest.TestCase):
    """Test that NavSim _build_actions rotates dx/dy to ego frame."""

    def _get_dataset_cls(self):
        """Import here to avoid top-level dependency on full project."""
        from app.vjepa_cowa_world_model.training.navsim_data import NavSimWorldModelDataset

        return NavSimWorldModelDataset

    def _build_actions_3d(self, states: np.ndarray) -> np.ndarray:
        """Call _build_actions_3d via a minimal dataset instance attribute."""
        # We only need the method; create a bare object and call it
        cls = self._get_dataset_cls()
        obj = cls.__new__(cls)
        return obj._build_actions_3d(states)

    # ------------------------------------------------------------------ #
    # 3D actions tests
    # ------------------------------------------------------------------ #
    def test_straight_east_3d(self):
        """Vehicle heading east (yaw=0), moving east → dx_ego > 0, dy_ego ≈ 0."""
        states = _make_states(
            [
                (100.0, 200.0, 0.0, 0.0, 0.0, 0.0),
                (103.75, 200.0, 0.0, 0.0, 0.0, 0.0),
            ]
        )
        actions = self._build_actions_3d(states)
        np.testing.assert_allclose(actions[0, 0], 3.75, atol=1e-4, err_msg="dx_ego should be ~3.75")
        np.testing.assert_allclose(actions[0, 1], 0.0, atol=1e-4, err_msg="dy_ego should be ~0")

    def test_straight_north_3d(self):
        """Vehicle heading north (yaw=π/2), moving north → dx_ego > 0, dy_ego ≈ 0."""
        yaw = np.pi / 2
        states = _make_states(
            [
                (100.0, 200.0, 0.0, 0.0, 0.0, yaw),
                (100.0, 203.75, 0.0, 0.0, 0.0, yaw),
            ]
        )
        actions = self._build_actions_3d(states)
        np.testing.assert_allclose(actions[0, 0], 3.75, atol=1e-4, err_msg="dx_ego should be ~3.75 (forward)")
        np.testing.assert_allclose(actions[0, 1], 0.0, atol=1e-4, err_msg="dy_ego should be ~0")

    def test_straight_west_3d(self):
        """Vehicle heading west (yaw=π), moving west → dx_ego > 0, dy_ego ≈ 0."""
        yaw = np.pi
        states = _make_states(
            [
                (100.0, 200.0, 0.0, 0.0, 0.0, yaw),
                (96.25, 200.0, 0.0, 0.0, 0.0, yaw),
            ]
        )
        actions = self._build_actions_3d(states)
        np.testing.assert_allclose(actions[0, 0], 3.75, atol=1e-3, err_msg="dx_ego should be ~3.75 (forward)")
        np.testing.assert_allclose(actions[0, 1], 0.0, atol=1e-3, err_msg="dy_ego should be ~0")

    def test_straight_south_3d(self):
        """Vehicle heading south (yaw=-π/2), moving south → dx_ego > 0."""
        yaw = -np.pi / 2
        states = _make_states(
            [
                (100.0, 200.0, 0.0, 0.0, 0.0, yaw),
                (100.0, 196.25, 0.0, 0.0, 0.0, yaw),
            ]
        )
        actions = self._build_actions_3d(states)
        np.testing.assert_allclose(actions[0, 0], 3.75, atol=1e-4, err_msg="dx_ego should be ~3.75 (forward)")
        np.testing.assert_allclose(actions[0, 1], 0.0, atol=1e-4, err_msg="dy_ego should be ~0")

    def test_lateral_left_3d(self):
        """Vehicle heading east, moving north → dx_ego ≈ 0, dy_ego > 0 (left)."""
        states = _make_states(
            [
                (100.0, 200.0, 0.0, 0.0, 0.0, 0.0),
                (100.0, 202.0, 0.0, 0.0, 0.0, 0.0),
            ]
        )
        actions = self._build_actions_3d(states)
        np.testing.assert_allclose(actions[0, 0], 0.0, atol=1e-4, err_msg="dx_ego should be ~0")
        np.testing.assert_allclose(actions[0, 1], 2.0, atol=1e-4, err_msg="dy_ego should be ~2.0 (left)")

    def test_dyaw_preserved_3d(self):
        """d_yaw (dim 2) should be relative heading change."""
        states = _make_states(
            [
                (100.0, 200.0, 0.0, 0.0, 0.0, 1.0),
                (103.0, 201.0, 0.0, 0.0, 0.0, 1.1),
            ]
        )
        actions = self._build_actions_3d(states)
        # d_yaw should be ~0.1 (angle_diff via relative rotation)
        np.testing.assert_allclose(actions[0, 2], 0.1, atol=1e-3, err_msg="d_yaw should be ~0.1")

    def test_consistency_across_headings_3d(self):
        """Same physical straight-line motion at different headings should produce same dx_ego."""
        speed = 10.0
        dist = 5.0

        results = []
        for yaw in [0.0, np.pi / 4, np.pi / 2, np.pi, -np.pi / 3, -np.pi / 2]:
            # Vehicle moves forward by `dist` in the heading direction
            dx_global = dist * np.cos(yaw)
            dy_global = dist * np.sin(yaw)
            states = _make_states(
                [
                    (1000.0, 2000.0, 0.0, 0.0, 0.0, yaw),
                    (1000.0 + dx_global, 2000.0 + dy_global, 0.0, 0.0, 0.0, yaw),
                ],
                speed=speed,
            )
            actions = self._build_actions_3d(states)
            results.append((actions[0, 0], actions[0, 1]))

        for i, (dx_ego, dy_ego) in enumerate(results):
            np.testing.assert_allclose(dx_ego, dist, atol=1e-3, err_msg=f"Heading {i}: dx_ego should be {dist}")
            np.testing.assert_allclose(dy_ego, 0.0, atol=1e-3, err_msg=f"Heading {i}: dy_ego should be 0")

    def test_multi_step_3d(self):
        """Test with 3 frames (2 action steps), each at different headings."""
        states = _make_states(
            [
                (100.0, 200.0, 0.0, 0.0, 0.0, 0.0),  # heading east
                (105.0, 200.0, 0.0, 0.0, 0.0, np.pi / 2),  # heading changes to north
                (105.0, 207.0, 0.0, 0.0, 0.0, np.pi / 2),  # moves north
            ]
        )
        actions = self._build_actions_3d(states)
        # Step 0: heading=0 (east), moves +5 in x → dx_ego=5, dy_ego=0
        np.testing.assert_allclose(actions[0, 0], 5.0, atol=1e-3)
        np.testing.assert_allclose(actions[0, 1], 0.0, atol=1e-3)
        # Step 1: heading=π/2 (north), moves +7 in y → dx_ego=7, dy_ego=0
        np.testing.assert_allclose(actions[1, 0], 7.0, atol=1e-3)
        np.testing.assert_allclose(actions[1, 1], 0.0, atol=1e-3)


class TestEgoRotationMath(unittest.TestCase):
    """Sanity checks for the ego rotation math itself."""

    def test_identity_rotation(self):
        """yaw=0 → no rotation, dx_ego == dx_global."""
        dx_e, dy_e = _rotate_to_ego(3.0, 4.0, 0.0)
        np.testing.assert_allclose(dx_e, 3.0, atol=1e-10)
        np.testing.assert_allclose(dy_e, 4.0, atol=1e-10)

    def test_90_degree_rotation(self):
        """yaw=π/2, moving +y in global → +x in ego (forward)."""
        dx_e, dy_e = _rotate_to_ego(0.0, 5.0, np.pi / 2)
        np.testing.assert_allclose(dx_e, 5.0, atol=1e-10)
        np.testing.assert_allclose(dy_e, 0.0, atol=1e-10)

    def test_180_degree_rotation(self):
        """yaw=π, moving -x in global → +x in ego (forward)."""
        dx_e, dy_e = _rotate_to_ego(-5.0, 0.0, np.pi)
        np.testing.assert_allclose(dx_e, 5.0, atol=1e-6)
        np.testing.assert_allclose(dy_e, 0.0, atol=1e-6)

    def test_minus_90_degree_rotation(self):
        """yaw=-π/2, moving -y in global → +x in ego (forward)."""
        dx_e, dy_e = _rotate_to_ego(0.0, -5.0, -np.pi / 2)
        np.testing.assert_allclose(dx_e, 5.0, atol=1e-10)
        np.testing.assert_allclose(dy_e, 0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
