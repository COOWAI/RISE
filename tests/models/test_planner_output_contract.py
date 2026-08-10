"""Phase 0 safety net: lock the planner forward-output contract.

Two layers:
  1. Pure unit tests of ``validate_planner_output`` (synthetic dicts) — robust, no model build.
  2. Characterization that the REAL planners (a tiny DiffusionPlanner + MultiModalTemporalPlanner)
     return outputs that satisfy the contract, in both inference and training modes.

If a later refactor changes a planner's return shape, these fail immediately.
"""

import math
import unittest

import torch

from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner
from app.vjepa_cowa_world_model.models.multimodal_planner import MultiModalTemporalPlanner
from app.vjepa_cowa_world_model.models.planner_contracts import (
    PLANNER_OBSERVED_TOKEN_CONCAT,
    PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    PLANNER_OBSERVED_TOKEN_NONE,
    normalize_observed_token_mode,
    validate_planner_output,
)
from app.vjepa_cowa_world_model.models.trajectory_head import TrajectoryHead


class TestTrajectoryHead(unittest.TestCase):
    def test_state_dict_and_output_contract(self):
        head = TrajectoryHead(num_poses=2, d_ffn=2, d_model=1)

        self.assertEqual(
            list(head.state_dict()),
            ["_mlp.0.weight", "_mlp.0.bias", "_mlp.2.weight", "_mlp.2.bias"],
        )

        with torch.no_grad():
            head._mlp[0].weight.copy_(torch.tensor([[1.0], [-1.0]]))
            head._mlp[0].bias.zero_()
            head._mlp[2].weight.zero_()
            head._mlp[2].bias.zero_()
            head._mlp[2].weight[2].copy_(torch.tensor([100.0, -100.0]))

        object_queries = torch.tensor([[[1.0], [-1.0]]])
        out = head(object_queries)
        self.assertEqual(set(out), {"trajectory"})
        self.assertEqual(out["trajectory"].shape, (1, 2, 3))
        headings = out["trajectory"][..., 2]
        self.assertTrue(torch.all(headings.abs() <= math.pi))
        torch.testing.assert_close(headings, torch.tensor([[math.pi, -math.pi]]))

        gradient_input = torch.tensor([[[0.01], [-0.01]]], requires_grad=True)
        head(gradient_input)["trajectory"][..., 2].sum().backward()
        self.assertTrue(torch.isfinite(gradient_input.grad).all())
        self.assertTrue(torch.all(gradient_input.grad != 0))


class TestValidatePlannerOutput(unittest.TestCase):
    def test_accepts_well_formed_inference_output(self):
        out = {"trajectories": torch.zeros(2, 3, 5, 3), "confidences": torch.zeros(2, 3)}
        self.assertIs(validate_planner_output(out), out)
        validate_planner_output(out, mode="inference", num_poses=5)

    def test_accepts_well_formed_training_output(self):
        out = {"loss": torch.tensor(1.0), "reg_loss": torch.tensor(0.5)}
        validate_planner_output(out, mode="training")
        # auto-detected as training because 'loss' is present
        validate_planner_output(out)

    def test_rejects_non_mapping(self):
        with self.assertRaises(TypeError):
            validate_planner_output([1, 2, 3])

    def test_rejects_missing_inference_keys(self):
        with self.assertRaises(ValueError):
            validate_planner_output({"trajectories": torch.zeros(2, 3, 5, 3)}, mode="inference")

    def test_rejects_wrong_trajectory_rank_or_pose_dim(self):
        with self.assertRaises(ValueError):
            validate_planner_output(
                {"trajectories": torch.zeros(2, 3, 5), "confidences": torch.zeros(2, 3)}, mode="inference"
            )
        with self.assertRaises(ValueError):
            validate_planner_output(
                {"trajectories": torch.zeros(2, 3, 5, 6), "confidences": torch.zeros(2, 3)}, mode="inference"
            )

    def test_rejects_bk_mismatch(self):
        with self.assertRaises(ValueError):
            validate_planner_output(
                {"trajectories": torch.zeros(2, 3, 5, 3), "confidences": torch.zeros(2, 4)}, mode="inference"
            )

    def test_rejects_num_poses_mismatch(self):
        with self.assertRaises(ValueError):
            validate_planner_output(
                {"trajectories": torch.zeros(2, 3, 5, 3), "confidences": torch.zeros(2, 3)},
                mode="inference",
                num_poses=7,
            )


class TestNormalizeObservedTokenMode(unittest.TestCase):
    def test_none_falls_back_on_use_flag(self):
        self.assertEqual(normalize_observed_token_mode(None, True), PLANNER_OBSERVED_TOKEN_CONCAT)
        self.assertEqual(normalize_observed_token_mode(None, False), PLANNER_OBSERVED_TOKEN_NONE)

    def test_aliases_resolve(self):
        self.assertEqual(normalize_observed_token_mode("off", True), PLANNER_OBSERVED_TOKEN_NONE)
        self.assertEqual(normalize_observed_token_mode("type-embed", True), PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED)

    def test_unknown_fails_loud(self):
        with self.assertRaises(ValueError):
            normalize_observed_token_mode("banana", True)


def _tiny_diffusion_planner():
    return DiffusionPlanner(
        encoder_dim=8,
        num_poses=3,
        status_dim=4,
        hidden_dim=16,
        depth=1,
        heads=4,
        dropout=0.0,
        mlp_ratio=2.0,
        traj_dim=6,
        num_modes=2,
        inference_steps=10,  # DPM-solver requires steps >= solver order
        tokens_per_frame=3,
        use_last_frame_only=False,
        observed_token_mode="none",
    )


class TestRealPlannersSatisfyContract(unittest.TestCase):
    def test_diffusion_planner_inference_output(self):
        planner = _tiny_diffusion_planner().eval()
        z_ar = torch.randn(2, 4, 8)
        status = torch.randn(2, 4)
        with torch.no_grad():
            out = planner(z_ar, status)
        validate_planner_output(out, mode="inference", num_poses=3)

    def test_diffusion_planner_training_output(self):
        planner = _tiny_diffusion_planner().train()
        z_ar = torch.randn(2, 4, 8)
        status = torch.randn(2, 4)
        gt = torch.randn(2, 3, 6)
        out = planner(z_ar, status, gt_trajectory=gt)
        validate_planner_output(out, mode="training")

    def test_multimodal_planner_inference_output(self):
        planner = MultiModalTemporalPlanner(
            encoder_dim=8,
            tf_d_model=16,
            tf_d_ffn=32,
            tf_num_layers=1,
            tf_num_head=4,
            tokens_per_frame=3,
            num_poses=3,
            num_time_steps=1,
            status_dim=4,
            num_modes=2,
            use_temporal=False,
        ).eval()
        z_ar = torch.randn(2, 3, 8)
        status = torch.randn(2, 4)
        with torch.no_grad():
            out = planner(z_ar, status)
        validate_planner_output(out, mode="inference", num_poses=3)


if __name__ == "__main__":
    unittest.main()
