"""Tests for NuScenes open-loop metrics: L2 per timestep and collision rate.

Tests cover:
- compute_l2_per_timestep: L2 displacement error at each time horizon
- _create_ego_footprint_pixels: ego box pixel footprint generation
- _rasterize_agents_to_bev: agent BEV rasterization
- _transform_traj_point_to_frame_t: trajectory coordinate transformation
- compute_collision_rate: ST-P3/VAD aligned collision (point + box)
- compute_collision_rate_legacy: backward-compatible interface
"""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

# Load the metrics module directly via importlib to avoid the full
# app.vjepa_cowa_world_model.__init__ import chain (which requires 'navsim').
_metrics_path = str(Path(__file__).resolve().parents[2] / "app" / "vjepa_cowa_world_model" / "utils" / "metrics.py")
_spec = importlib.util.spec_from_file_location("metrics", _metrics_path)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)

BEV_SIZE = _metrics.BEV_SIZE
BEV_RANGE = _metrics.BEV_RANGE
BEV_RESOLUTION = _metrics.BEV_RESOLUTION
compute_l2_per_timestep = _metrics.compute_l2_per_timestep
compute_world4drive_l2_metrics = _metrics.compute_world4drive_l2_metrics
compute_collision_rate = _metrics.compute_collision_rate
compute_collision_rate_legacy = _metrics.compute_collision_rate_legacy
populate_world4drive_collision_horizons = _metrics.populate_world4drive_collision_horizons
_create_ego_footprint_pixels = _metrics._create_ego_footprint_pixels
_rasterize_agents_to_bev = _metrics._rasterize_agents_to_bev
_transform_traj_point_to_frame_t = _metrics._transform_traj_point_to_frame_t
_point_to_bev_pixel = _metrics._point_to_bev_pixel


class TestComputeL2PerTimestep(unittest.TestCase):
    """Test compute_l2_per_timestep with known trajectories."""

    def test_zero_error(self):
        """Identical prediction and GT should produce zero L2 at all steps."""
        B, T = 4, 8
        traj = torch.randn(B, T, 3)
        result = compute_l2_per_timestep(traj, traj, timestep_sec=0.5)

        self.assertAlmostEqual(result["l2_avg"], 0.0, places=5)
        for v in result["l2_per_step"]:
            self.assertAlmostEqual(v, 0.0, places=5)
        # With 8 steps at 0.5s, we should report 1s, 2s, 3s, 4s
        for sec in [1, 2, 3, 4]:
            self.assertIn(f"l2_at_{sec}s", result)
            self.assertAlmostEqual(result[f"l2_at_{sec}s"], 0.0, places=5)

    def test_known_displacement(self):
        """Known constant displacement should produce predictable L2."""
        B, T = 2, 8
        pred = torch.zeros(B, T, 3)
        gt = torch.zeros(B, T, 3)
        # Set GT x=3.0, y=4.0 at all steps → L2 = 5.0
        gt[:, :, 0] = 3.0
        gt[:, :, 1] = 4.0

        result = compute_l2_per_timestep(pred, gt, timestep_sec=0.5)

        for v in result["l2_per_step"]:
            self.assertAlmostEqual(v, 5.0, places=4)
        self.assertAlmostEqual(result["l2_avg"], 5.0, places=4)
        self.assertAlmostEqual(result["l2_at_1s"], 5.0, places=4)
        self.assertAlmostEqual(result["l2_at_2s"], 5.0, places=4)
        self.assertAlmostEqual(result["l2_at_3s"], 5.0, places=4)
        self.assertAlmostEqual(result["l2_at_4s"], 5.0, places=4)

    def test_varying_displacement(self):
        """Different displacements at each step."""
        B, T = 1, 4  # 4 steps at 0.5s → can report 1s, 2s
        pred = torch.zeros(B, T, 3)
        gt = torch.zeros(B, T, 3)
        # Step 0: dx=1, dy=0 → L2=1
        # Step 1: dx=3, dy=4 → L2=5
        # Step 2: dx=0, dy=2 → L2=2
        # Step 3: dx=6, dy=8 → L2=10
        gt[0, 0, 0] = 1.0
        gt[0, 1, 0] = 3.0
        gt[0, 1, 1] = 4.0
        gt[0, 2, 1] = 2.0
        gt[0, 3, 0] = 6.0
        gt[0, 3, 1] = 8.0

        result = compute_l2_per_timestep(pred, gt, timestep_sec=0.5)

        self.assertAlmostEqual(result["l2_per_step"][0], 1.0, places=4)
        self.assertAlmostEqual(result["l2_per_step"][1], 5.0, places=4)
        self.assertAlmostEqual(result["l2_per_step"][2], 2.0, places=4)
        self.assertAlmostEqual(result["l2_per_step"][3], 10.0, places=4)
        # l2_at_1s = step_idx=1 → 5.0
        self.assertAlmostEqual(result["l2_at_1s"], 5.0, places=4)
        # l2_at_2s = step_idx=3 → 10.0
        self.assertAlmostEqual(result["l2_at_2s"], 10.0, places=4)

    def test_short_trajectory(self):
        """Trajectory shorter than 4s should only report available horizons."""
        B, T = 1, 2  # 2 steps at 0.5s → only 1s available
        pred = torch.zeros(B, T, 3)
        gt = torch.zeros(B, T, 3)

        result = compute_l2_per_timestep(pred, gt, timestep_sec=0.5)

        self.assertIn("l2_at_1s", result)
        self.assertNotIn("l2_at_2s", result)
        self.assertNotIn("l2_at_3s", result)
        self.assertNotIn("l2_at_4s", result)


class TestComputeWorld4DriveL2Metrics(unittest.TestCase):
    """World4Drive L2 reports cumulative ADE while preserving point horizons."""

    def test_cumulative_and_point_horizons_are_reported(self):
        pred = torch.zeros(1, 6, 3)
        gt = torch.zeros(1, 6, 3)
        gt[0, :, 0] = torch.tensor([1.0, 5.0, 2.0, 10.0, 6.0, 8.0])

        result = compute_world4drive_l2_metrics(pred, gt, timestep_sec=0.5)

        self.assertEqual(result["l2_per_step"], [1.0, 5.0, 2.0, 10.0, 6.0, 8.0])
        self.assertAlmostEqual(result["l2_at_1s"], 3.0, places=5)
        self.assertAlmostEqual(result["l2_at_2s"], 4.5, places=5)
        self.assertAlmostEqual(result["l2_at_3s"], 32.0 / 6.0, places=5)
        self.assertAlmostEqual(result["l2_avg"], (3.0 + 4.5 + 32.0 / 6.0) / 3.0, places=5)
        self.assertAlmostEqual(result["l2_point_at_1s"], 5.0, places=5)
        self.assertAlmostEqual(result["l2_point_at_2s"], 10.0, places=5)
        self.assertAlmostEqual(result["l2_point_at_3s"], 8.0, places=5)
        self.assertAlmostEqual(result["l2_point_avg"], (5.0 + 10.0 + 8.0) / 3.0, places=5)
        self.assertNotIn("l2_at_4s", result)


class TestTransformTrajPointToFrameT(unittest.TestCase):
    """Test coordinate transformation of trajectory points from frame-0 to frame-t."""

    def test_identity_transform(self):
        """When ego_pose_0 == ego_pose_t, point should not move."""
        point = np.array([10.0, 5.0])
        ego_pose = np.array([100.0, 200.0, 0.0, 0.0, 0.0, 0.5, 10.0], dtype=np.float64)

        result = _transform_traj_point_to_frame_t(point, ego_pose, ego_pose)

        np.testing.assert_allclose(result, point, atol=1e-5)

    def test_pure_translation(self):
        """When ego translates but doesn't rotate, point shifts accordingly."""
        point = np.array([0.0, 0.0])  # origin in frame-0

        # frame-0 at origin, frame-t at (10, 0), both yaw=0
        ego_pose_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float64)
        ego_pose_t = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float64)

        # Point at (0,0) in frame-0 → global (0,0) → frame-t local: (-10, 0)
        result = _transform_traj_point_to_frame_t(point, ego_pose_0, ego_pose_t)

        np.testing.assert_allclose(result, [-10.0, 0.0], atol=1e-5)

    def test_pure_rotation(self):
        """When ego rotates 90 degrees, point transforms accordingly."""
        point = np.array([1.0, 0.0])

        # frame-0 at origin yaw=0, frame-t at origin yaw=pi/2
        ego_pose_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float64)
        ego_pose_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, 10.0], dtype=np.float64)

        # Point (1,0) in frame-0 → global (1,0) → frame-t: rotate by -pi/2 → (0, -1)
        result = _transform_traj_point_to_frame_t(point, ego_pose_0, ego_pose_t)

        np.testing.assert_allclose(result, [0.0, -1.0], atol=1e-5)

    def test_round_trip(self):
        """Transform frame-0→frame-t→frame-0 should recover original point."""
        point = np.array([5.0, -3.0])

        ego_pose_0 = np.array([10.0, 20.0, 0.0, 0.0, 0.0, 0.3, 10.0], dtype=np.float64)
        ego_pose_t = np.array([15.0, 22.0, 0.0, 0.0, 0.0, 0.8, 10.0], dtype=np.float64)

        # frame-0 → frame-t
        point_t = _transform_traj_point_to_frame_t(point, ego_pose_0, ego_pose_t)
        # frame-t → frame-0 (swap poses)
        point_back = _transform_traj_point_to_frame_t(point_t, ego_pose_t, ego_pose_0)

        np.testing.assert_allclose(point_back, point, atol=1e-5)

    def test_combined_translation_rotation(self):
        """Test combined translation + rotation transform."""
        point = np.array([2.0, 1.0])

        # frame-0 at (5, 3) yaw=pi/4
        ego_pose_0 = np.array([5.0, 3.0, 0.0, 0.0, 0.0, np.pi / 4, 10.0], dtype=np.float64)
        # frame-t at (10, 8) yaw=pi/2
        ego_pose_t = np.array([10.0, 8.0, 0.0, 0.0, 0.0, np.pi / 2, 10.0], dtype=np.float64)

        result = _transform_traj_point_to_frame_t(point, ego_pose_0, ego_pose_t)

        # Verify by manual computation:
        # Step 1: frame-0 ego → global
        cos_0 = np.cos(np.pi / 4)
        sin_0 = np.sin(np.pi / 4)
        gx = 5.0 + cos_0 * 2.0 - sin_0 * 1.0
        gy = 3.0 + sin_0 * 2.0 + cos_0 * 1.0
        # Step 2: global → frame-t ego
        cos_t_inv = np.cos(-np.pi / 2)
        sin_t_inv = np.sin(-np.pi / 2)
        dx = gx - 10.0
        dy = gy - 8.0
        expected_x = cos_t_inv * dx - sin_t_inv * dy
        expected_y = sin_t_inv * dx + cos_t_inv * dy

        np.testing.assert_allclose(result, [expected_x, expected_y], atol=1e-5)


class TestPointToBevPixel(unittest.TestCase):
    """Test coordinate-to-pixel conversion."""

    def test_origin(self):
        """Origin (0,0) should map to center pixel."""
        row, col = _point_to_bev_pixel(0.0, 0.0)
        self.assertEqual(row, BEV_SIZE // 2)  # 100
        self.assertEqual(col, BEV_SIZE // 2)  # 100

    def test_forward(self):
        """Positive x (forward) should decrease row (up in image)."""
        row, col = _point_to_bev_pixel(10.0, 0.0)
        expected_col = int((10.0 + BEV_RANGE) / BEV_RESOLUTION)
        expected_row = int((BEV_RANGE - 0.0) / BEV_RESOLUTION)
        self.assertEqual(row, expected_row)
        self.assertEqual(col, expected_col)

    def test_left(self):
        """Positive y (left) should decrease row."""
        row, col = _point_to_bev_pixel(0.0, 5.0)
        expected_col = int((0.0 + BEV_RANGE) / BEV_RESOLUTION)
        expected_row = int((BEV_RANGE - 5.0) / BEV_RESOLUTION)
        self.assertEqual(row, expected_row)
        self.assertEqual(col, expected_col)


class TestEgoFootprint(unittest.TestCase):
    """Test ego footprint pixel creation."""

    def test_footprint_not_empty(self):
        """Ego footprint should produce non-empty pixel mask."""
        rr, cc = _create_ego_footprint_pixels()
        self.assertGreater(len(rr), 0)
        self.assertGreater(len(cc), 0)
        self.assertEqual(len(rr), len(cc))

    def test_footprint_reasonable_size(self):
        """Ego footprint pixel count should be reasonable for a ~4m x ~2m car."""
        rr, cc = _create_ego_footprint_pixels()
        # At 0.5m/pixel, a 4.084m x 1.85m car ≈ ~8 x 4 = 32 pixels
        # With some rounding it could be 20-60 pixels
        self.assertGreater(len(rr), 10)
        self.assertLess(len(rr), 100)


class TestRasterizeAgents(unittest.TestCase):
    """Test BEV rasterization of agent boxes."""

    def test_empty_agents(self):
        """No valid agents should produce empty BEV grid."""
        max_agents = 5
        boxes = np.zeros((max_agents, 7), dtype=np.float32)
        mask = np.zeros(max_agents, dtype=np.bool_)

        seg = _rasterize_agents_to_bev(boxes, mask)

        self.assertEqual(seg.shape, (BEV_SIZE, BEV_SIZE))
        self.assertEqual(seg.sum(), 0)

    def test_single_agent_at_origin(self):
        """A single agent at the origin should produce non-zero pixels."""
        max_agents = 5
        boxes = np.zeros((max_agents, 7), dtype=np.float32)
        mask = np.zeros(max_agents, dtype=np.bool_)

        # Agent at (0, 0) with size 4x2, heading 0
        boxes[0] = [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0]
        mask[0] = True

        seg = _rasterize_agents_to_bev(boxes, mask)

        self.assertGreater(seg.sum(), 0)

    def test_agent_out_of_range(self):
        """Agent outside BEV range should not appear."""
        max_agents = 5
        boxes = np.zeros((max_agents, 7), dtype=np.float32)
        mask = np.zeros(max_agents, dtype=np.bool_)

        # Agent at (60, 60) → outside [-50, 50]
        boxes[0] = [60.0, 60.0, 0.0, 4.0, 2.0, 1.5, 0.0]
        mask[0] = True

        seg = _rasterize_agents_to_bev(boxes, mask)

        self.assertEqual(seg.sum(), 0)


class TestComputeCollisionRate(unittest.TestCase):
    """Test collision rate computation with pre-computed segmentation maps."""

    def _make_seg_with_agent_at_origin(self, B, T_future):
        """Helper: create seg maps with a large agent at origin for all frames."""
        seg = np.zeros((B, T_future, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        max_agents = 1
        boxes = np.zeros((max_agents, 7), dtype=np.float32)
        mask = np.zeros(max_agents, dtype=np.bool_)
        boxes[0] = [0.0, 0.0, 0.0, 6.0, 4.0, 1.5, 0.0]
        mask[0] = True
        agent_seg = _rasterize_agents_to_bev(boxes, mask)
        for b in range(B):
            for t in range(T_future):
                seg[b, t] = agent_seg.copy()
        return seg

    def _make_seg_with_agent_far(self, B, T_future):
        """Helper: create seg maps with agents far from origin."""
        seg = np.zeros((B, T_future, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        max_agents = 1
        boxes = np.zeros((max_agents, 7), dtype=np.float32)
        mask = np.zeros(max_agents, dtype=np.bool_)
        boxes[0] = [40.0, 40.0, 0.0, 4.0, 2.0, 1.5, 0.0]
        mask[0] = True
        agent_seg = _rasterize_agents_to_bev(boxes, mask)
        for b in range(B):
            for t in range(T_future):
                seg[b, t] = agent_seg.copy()
        return seg

    def test_no_collision(self):
        """Predicted trajectory far from any agent should have zero collision."""
        B, T_future = 2, 4
        T_total = 6

        # Pred and GT at (0, 0) in frame-0 coords
        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)

        # Agents far away at (40, 40)
        seg = self._make_seg_with_agent_far(B, T_future)

        # States: ego at origin for all frames, no rotation
        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        self.assertAlmostEqual(result["collision_rate"], 0.0, places=5)
        self.assertAlmostEqual(result["point_collision_rate"], 0.0, places=5)

    def test_certain_box_collision(self):
        """Predicted trajectory on top of an agent should have box collision."""
        B, T_future = 1, 2
        T_total = 4

        # GT trajectory far from agent (no GT collision)
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj[:, :, 0] = 30.0

        # Pred trajectory at (0, 0) — on top of the agent
        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)

        # Agent at origin
        seg = self._make_seg_with_agent_at_origin(B, T_future)

        # States: ego at origin for all frames
        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        # All steps should have box collision
        for rate in result["collision_per_step"]:
            self.assertGreater(rate, 0.0)
        # Point collision should also fire (agent covers origin pixel)
        for rate in result["point_collision_per_step"]:
            self.assertGreater(rate, 0.0)

    def test_point_collision_only(self):
        """Agent covering only ego center pixel should trigger point but not box collision
        (unless agent is large enough). Here we test with a tiny 1-pixel agent."""
        B, T_future = 1, 2
        T_total = 4

        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj[:, :, 0] = 30.0  # GT far away

        # Pred at some offset where we place a single pixel
        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        pred_traj[:, :, 0] = 10.0  # forward 10m

        seg = np.zeros((B, T_future, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        # Place a single pixel at (10, 0) in ego coords → pixel (90, 120)
        p_row, p_col = _point_to_bev_pixel(10.0, 0.0)
        for b in range(B):
            for t in range(T_future):
                seg[b, t, p_row, p_col] = 1

        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        # Point collision should fire
        for rate in result["point_collision_per_step"]:
            self.assertGreater(rate, 0.0)
        # Box collision should also fire (that pixel is within the ego footprint)
        for rate in result["collision_per_step"]:
            self.assertGreater(rate, 0.0)

    def test_no_agents(self):
        """No valid agents should produce zero collision rate."""
        B, T_future = 2, 4
        T_total = 6

        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)

        # Empty segmentation
        seg = np.zeros((B, T_future, BEV_SIZE, BEV_SIZE), dtype=np.uint8)

        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        self.assertAlmostEqual(result["collision_rate"], 0.0, places=5)
        self.assertAlmostEqual(result["point_collision_rate"], 0.0, places=5)

    def test_gt_collision_excluded(self):
        """Steps where GT trajectory collides should be excluded from counting."""
        B, T_future = 1, 2
        T_total = 4

        # Both pred and GT at (0, 0) — on top of agent
        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)

        # Agent at (0, 0)
        seg = self._make_seg_with_agent_at_origin(B, T_future)

        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        # GT collides at all steps → gt_collision_counts should be [1, 1]
        for gc in result["gt_collision_counts"]:
            self.assertEqual(gc, 1)
        # Collision counts should be 0 (excluded samples)
        for cc in result["collision_counts"]:
            self.assertEqual(cc, 0)
        # Collision rate = 0 (no valid samples for collision)
        self.assertAlmostEqual(result["collision_rate"], 0.0, places=5)

    def test_world4drive_collision_horizons_are_cumulative(self):
        """World4Drive collision@horizon averages all steps up to the horizon."""
        metrics = {}
        populate_world4drive_collision_horizons(
            metrics,
            collision_counts=np.array([1, 1, 0, 0], dtype=np.int64),
            total_samples=1,
            timestep_sec=0.5,
            reported_seconds=(1, 2),
        )

        self.assertAlmostEqual(metrics["collision_at_1s"], 1.0, places=5)
        self.assertAlmostEqual(metrics["collision_at_2s"], 0.5, places=5)
        self.assertAlmostEqual(metrics["collision_rate"], 0.75, places=5)

    def test_compute_collision_rate_reports_cumulative_horizons(self):
        B, T_future = 1, 4
        T_total = 6

        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj[:, :, 0] = 30.0
        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)

        seg = np.zeros((B, T_future, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        agent_seg = self._make_seg_with_agent_at_origin(B, T_future)[0, 0]
        seg[0, 0] = agent_seg
        seg[0, 1] = agent_seg

        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        self.assertEqual(result["collision_counts"], [1, 1, 0, 0])
        self.assertAlmostEqual(result["collision_at_1s"], 1.0, places=5)
        self.assertAlmostEqual(result["collision_at_2s"], 0.5, places=5)
        self.assertAlmostEqual(result["collision_rate"], 0.75, places=5)

    def test_output_keys(self):
        """Verify all expected keys are in the result."""
        B, T_future = 1, 4
        T_total = 6

        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        seg = np.zeros((B, T_future, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        expected_keys = [
            "collision_rate",
            "collision_per_step",
            "collision_counts",
            "point_collision_rate",
            "point_collision_per_step",
            "point_collision_counts",
            "total_samples",
            "gt_collision_counts",
        ]
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_total_samples_equals_B(self):
        """total_samples should always equal B."""
        B, T_future = 3, 4
        T_total = 6

        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        seg = np.zeros((B, T_future, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        self.assertEqual(result["total_samples"], B)

    def test_with_ego_pose_transform(self):
        """Test that trajectory points are correctly transformed when ego moves."""
        B, T_future = 1, 1
        T_total = 3

        # Pred trajectory: (5, 0) in frame-0 ego coords
        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        pred_traj[0, 0, 0] = 5.0

        # GT far away
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj[:, :, 0] = 40.0

        # Ego moves forward: frame-0 at origin, frame-2 at (5, 0)
        # So pred point (5,0) in frame-0 → global (5,0) → frame-2 local: (0, 0)
        ego_poses = np.zeros((B, T_total, 7), dtype=np.float64)
        ego_poses[0, 2, 0] = 5.0  # frame-2 ego at x=5

        # Agent at (0, 0) in frame-2 ego coords (= the transformed pred location)
        seg = self._make_seg_with_agent_at_origin(B, T_future)

        result = compute_collision_rate(
            pred_traj,
            gt_traj,
            seg,
            ego_poses,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        # Should have collision because pred transforms to (0,0) in frame-2
        self.assertGreater(result["collision_per_step"][0], 0.0)


class TestComputeCollisionRateLegacy(unittest.TestCase):
    """Test backward-compatible collision rate interface."""

    def test_no_collision_legacy(self):
        """Legacy interface: no collision scenario."""
        B, T_future = 2, 4
        T_total = 6
        max_agents = 5

        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)

        agent_boxes = np.zeros((B, T_total, max_agents, 7), dtype=np.float32)
        agent_mask = np.zeros((B, T_total, max_agents), dtype=np.bool_)
        for b in range(B):
            for t in range(T_total):
                agent_boxes[b, t, 0] = [40.0, 40.0, 0.0, 4.0, 2.0, 1.5, 0.0]
                agent_mask[b, t, 0] = True

        states = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate_legacy(
            pred_traj,
            gt_traj,
            agent_boxes,
            agent_mask,
            states,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        self.assertAlmostEqual(result["collision_rate"], 0.0, places=5)
        self.assertIn("point_collision_rate", result)
        self.assertIn("total_samples", result)

    def test_certain_collision_legacy(self):
        """Legacy interface: certain collision scenario."""
        B, T_future = 1, 2
        T_total = 4
        max_agents = 5

        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj[:, :, 0] = 30.0

        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)

        agent_boxes = np.zeros((B, T_total, max_agents, 7), dtype=np.float32)
        agent_mask = np.zeros((B, T_total, max_agents), dtype=np.bool_)
        for t in range(T_total):
            agent_boxes[0, t, 0] = [0.0, 0.0, 0.0, 6.0, 4.0, 1.5, 0.0]
            agent_mask[0, t, 0] = True

        states = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate_legacy(
            pred_traj,
            gt_traj,
            agent_boxes,
            agent_mask,
            states,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        for rate in result["collision_per_step"]:
            self.assertGreater(rate, 0.0)

    def test_gt_collision_excluded_legacy(self):
        """Legacy interface: GT collision exclusion."""
        B, T_future = 1, 2
        T_total = 4
        max_agents = 5

        pred_traj = np.zeros((B, T_future, 3), dtype=np.float32)
        gt_traj = np.zeros((B, T_future, 3), dtype=np.float32)

        agent_boxes = np.zeros((B, T_total, max_agents, 7), dtype=np.float32)
        agent_mask = np.zeros((B, T_total, max_agents), dtype=np.bool_)
        for t in range(T_total):
            agent_boxes[0, t, 0] = [0.0, 0.0, 0.0, 6.0, 4.0, 1.5, 0.0]
            agent_mask[0, t, 0] = True

        states = np.zeros((B, T_total, 7), dtype=np.float64)

        result = compute_collision_rate_legacy(
            pred_traj,
            gt_traj,
            agent_boxes,
            agent_mask,
            states,
            future_start_idx=2,
            timestep_sec=0.5,
        )

        for gc in result["gt_collision_counts"]:
            self.assertEqual(gc, 1)
        self.assertAlmostEqual(result["collision_rate"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
