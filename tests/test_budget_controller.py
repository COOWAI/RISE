import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from app.vjepa_cowa_world_model.training.budget_control import (
    BudgetController,
    BudgetSchedule,
    budget_controller_bc_loss,
    budget_controller_grpo_loss,
    collect_budget_oracle,
    compute_budget_utility,
    load_budget_controller_from_checkpoint,
    resolve_controller_budget_profile,
    sample_beta_budget,
    select_oracle_budget,
)


class TestBudgetSchedule(unittest.TestCase):
    def test_maps_continuous_budget_to_rollout_future_steps(self):
        schedule = BudgetSchedule.from_mapping({"rollout_future_steps": [1, "full"]})

        low = schedule.profile(0.0, max_future_steps=6)
        mid = schedule.profile(0.5, max_future_steps=6)
        high = schedule.profile(1.0, max_future_steps=6)

        self.assertEqual(low.rollout_future_steps, 1)
        self.assertEqual(mid.rollout_future_steps, 4)
        self.assertEqual(high.rollout_future_steps, 6)

    def test_maps_zero_budget_to_zero_rollout_future_steps(self):
        schedule = BudgetSchedule.from_mapping({"rollout_future_steps": [0, "full"]})
        grid = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

        steps = [schedule.profile(budget, max_future_steps=3).rollout_future_steps for budget in grid]

        self.assertEqual(steps, [0, 1, 2, 3])

    def test_rejects_legacy_schedule_keys(self):
        for key in (
            "predictor_sampling_steps",
            "planner_inference_steps",
            "planner_num_candidates",
            "value_guidance_steps",
            "value_guidance_scale",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "no longer supported"):
                    BudgetSchedule.from_mapping({key: [1, 2]})

    def test_rejects_budget_outside_unit_interval(self):
        schedule = BudgetSchedule()

        with self.assertRaisesRegex(ValueError, "budget"):
            schedule.profile(-0.1, max_future_steps=6)
        with self.assertRaisesRegex(ValueError, "budget"):
            schedule.profile(1.1, max_future_steps=6)


class TestBudgetController(unittest.TestCase):
    def test_checkpoint_loader_can_require_stage4c_bc_provenance(self):
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "controller.pt"
            controller = BudgetController(latent_dim=2, hidden_dim=8)
            torch.save(
                {
                    "controller": controller.state_dict(),
                    "latent_dim": 2,
                    "feature_dim": 0,
                    "hidden_dim": 8,
                    "min_concentration": 1.0,
                    "mode": "grpo",
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "oracle_distillation"):
                load_budget_controller_from_checkpoint(
                    checkpoint_path,
                    device=torch.device("cpu"),
                    expected_mode="oracle_distillation",
                )

    def test_outputs_beta_action_and_log_prob_shapes(self):
        controller = BudgetController(latent_dim=8, feature_dim=3, hidden_dim=16)
        pooled_latent = torch.randn(5, 8)
        cheap_features = torch.randn(5, 3)

        result = controller.sample_budget(pooled_latent, cheap_features, deterministic=False)

        self.assertEqual(tuple(result.budget.shape), (5,))
        self.assertEqual(tuple(result.log_prob.shape), (5,))
        self.assertEqual(tuple(result.alpha.shape), (5,))
        self.assertEqual(tuple(result.beta.shape), (5,))
        self.assertTrue(torch.all(result.budget >= 0.0))
        self.assertTrue(torch.all(result.budget <= 1.0))

    def test_deterministic_budget_uses_beta_mean(self):
        controller = BudgetController(latent_dim=4, feature_dim=0, hidden_dim=8)
        pooled_latent = torch.randn(2, 4)

        result = controller.sample_budget(pooled_latent, deterministic=True)

        expected = result.alpha / (result.alpha + result.beta)
        self.assertTrue(torch.allclose(result.budget, expected))

    def test_stochastic_budget_is_score_function_sample(self):
        controller = BudgetController(latent_dim=4, feature_dim=0, hidden_dim=8)
        pooled_latent = torch.randn(2, 4)

        result = controller.sample_budget(pooled_latent, deterministic=False)

        self.assertFalse(result.budget.requires_grad)
        self.assertTrue(result.log_prob.requires_grad)

    def test_beta_sampling_from_forward_parameters_backpropagates(self):
        controller = BudgetController(latent_dim=4, hidden_dim=8)
        pooled_latent = torch.randn(3, 4)

        alpha, beta = controller(pooled_latent)
        output = sample_beta_budget(alpha, beta, deterministic=False)
        loss = -output.log_prob.mean()
        loss.backward()

        self.assertEqual(tuple(output.budget.shape), (3,))
        self.assertFalse(output.budget.requires_grad)
        self.assertTrue(output.log_prob.requires_grad)
        grad_norm = sum(
            float(parameter.grad.detach().norm())
            for parameter in controller.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_rejects_feature_shape_mismatches(self):
        controller = BudgetController(latent_dim=4, feature_dim=2, hidden_dim=8)
        pooled_latent = torch.randn(3, 4)

        with self.assertRaisesRegex(ValueError, "cheap_features is required"):
            controller.sample_budget(pooled_latent, deterministic=True)
        with self.assertRaisesRegex(ValueError, "cheap_features must be"):
            controller.sample_budget(pooled_latent, torch.randn(3, 3), deterministic=True)

    def test_bc_loss_backpropagates_to_controller(self):
        controller = BudgetController(latent_dim=2, hidden_dim=8)
        pooled_latent = torch.randn(4, 2)
        target_budget = torch.full((4,), 0.75)

        result = budget_controller_bc_loss(controller, pooled_latent, target_budget)
        result["loss"].backward()

        grad_norm = torch.stack([p.grad.detach().norm() for p in controller.parameters() if p.grad is not None]).sum()
        self.assertTrue(torch.isfinite(grad_norm))
        self.assertGreater(float(grad_norm), 0.0)

    def test_oracle_utility_prefers_low_budget_when_scores_tie(self):
        budgets = torch.tensor([0.0, 0.5, 1.0])
        scores = torch.tensor([0.8, 0.8, 0.8])
        costs = torch.tensor([1.0, 2.0, 3.0])

        utility = compute_budget_utility(scores, costs, lambda_compute=0.1)
        selected = select_oracle_budget(budgets, scores, costs, lambda_compute=0.1)

        self.assertTrue(torch.allclose(utility, torch.tensor([0.7, 0.6, 0.5])))
        self.assertAlmostEqual(float(selected.budget), 0.0)
        self.assertEqual(int(selected.index), 0)

    def test_oracle_utility_prefers_high_budget_when_score_gain_wins(self):
        budgets = torch.tensor([0.0, 0.5, 1.0])
        scores = torch.tensor([0.2, 0.6, 1.0])
        costs = torch.tensor([1.0, 2.0, 3.0])

        selected = select_oracle_budget(budgets, scores, costs, lambda_compute=0.1)

        self.assertAlmostEqual(float(selected.budget), 1.0)
        self.assertEqual(int(selected.index), 2)

    def test_budget_controller_bc_loss_optimizes_target_budget(self):
        controller = BudgetController(latent_dim=2, hidden_dim=8)
        pooled_latent = torch.randn(4, 2)
        target_budget = torch.tensor([0.0, 0.25, 0.75, 1.0])

        result = budget_controller_bc_loss(controller, pooled_latent, target_budget)

        self.assertIn("loss", result)
        self.assertIn("nll", result)
        self.assertIn("mse", result)
        self.assertEqual(tuple(result["pred_budget"].shape), (4,))
        self.assertTrue(torch.isfinite(result["loss"]))
        self.assertTrue(torch.isfinite(result["nll"]))

    def test_collect_budget_oracle_uses_fixed_policy_callback(self):
        schedule = BudgetSchedule(rollout_future_steps=(1, 3))

        def evaluate(scene_id, budget, profile):
            del scene_id
            return float(budget), float(profile.rollout_future_steps)

        records = collect_budget_oracle(
            scene_ids=["a", "b"],
            budget_grid=[0.0, 1.0],
            schedule=schedule,
            evaluate_fn=evaluate,
            lambda_compute=0.1,
        )

        self.assertEqual(len(records), 4)
        self.assertEqual(records[0].scene_id, "a")
        self.assertAlmostEqual(records[0].budget, 0.0)
        self.assertAlmostEqual(records[0].utility, -0.1)
        self.assertAlmostEqual(records[1].utility, 0.7)

    def test_grpo_loss_uses_group_baseline(self):
        log_prob = torch.tensor([-0.2, -0.4, -0.6, -0.8])
        reward = torch.tensor([1.0, 3.0, 2.0, 2.0])
        group_ids = torch.tensor([0, 0, 1, 1])

        result = budget_controller_grpo_loss(log_prob, reward, group_ids=group_ids)

        expected_adv = torch.tensor([-1.0, 1.0, 0.0, 0.0])
        expected_loss = -(log_prob * expected_adv).mean()
        self.assertTrue(torch.allclose(result["advantage"], expected_adv))
        self.assertTrue(torch.allclose(result["loss"], expected_loss))

    def test_loads_controller_checkpoint_and_resolves_runtime_profile(self):
        controller = BudgetController(latent_dim=4, hidden_dim=8)
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "budget_controller.pt"
            torch.save(
                {
                    "controller": controller.state_dict(),
                    "latent_dim": 4,
                    "feature_dim": 0,
                    "hidden_dim": 8,
                    "min_concentration": 1.0,
                },
                checkpoint_path,
            )

            loaded = load_budget_controller_from_checkpoint(checkpoint_path, device=torch.device("cpu"))
            config = SimpleNamespace(
                budget_controller=SimpleNamespace(
                    schedule={
                        "rollout_future_steps": [1, "full"],
                    },
                    feature_dim=0,
                )
            )

            z_obs = torch.randn(1, 3, 4)
            budget, profile = resolve_controller_budget_profile(
                loaded,
                z_obs,
                config=config,
                deterministic=True,
                max_future_steps=6,
            )

            self.assertEqual(tuple(budget.shape), (1,))
            self.assertGreaterEqual(profile.rollout_future_steps, 1)
            self.assertLessEqual(profile.rollout_future_steps, 6)

    def test_controller_runtime_profile_requires_single_sample_batch(self):
        controller = BudgetController(latent_dim=4, hidden_dim=8)
        config = SimpleNamespace(
            budget_controller=SimpleNamespace(
                schedule={"rollout_future_steps": [1, "full"]},
                feature_dim=0,
            )
        )

        with self.assertRaisesRegex(ValueError, "batch size 1"):
            resolve_controller_budget_profile(
                controller,
                torch.randn(2, 3, 4),
                config=config,
                deterministic=True,
                max_future_steps=6,
            )


if __name__ == "__main__":
    unittest.main()
