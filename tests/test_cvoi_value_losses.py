import unittest

import torch

from app.vjepa_cowa_world_model.training.cvoi_value import (
    CVoIValueLossOutput,
    compute_domain_routed_field_loss,
    compute_navsim_e120_quality_field_loss,
    compute_real_stop_value_loss,
)


class TestDomainRoutedFieldLoss(unittest.TestCase):
    def test_real_rows_use_absolute_huber_targets(self):
        values = torch.tensor([[0.0, 1.0], [0.8, 0.3]], requires_grad=True)
        targets = torch.tensor([[0.2, 0.8], [0.6, 0.4]])

        result = compute_domain_routed_field_loss(
            values,
            ["real", "real"],
            real_geometry_targets=targets,
            real_order_weight=0.0,
        )

        expected = torch.nn.functional.huber_loss(values, targets, reduction="mean", delta=1.0)
        self.assertIsInstance(result, CVoIValueLossOutput)
        self.assertTrue(torch.allclose(result.loss, expected))
        self.assertEqual(result.diagnostics["real_count"], 2.0)

    def test_optional_local_order_penalizes_reversed_local_candidates(self):
        values = torch.tensor([[0.9, 0.8], [0.5, 0.5], [0.1, 0.2]], requires_grad=True)
        targets = torch.tensor([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])

        result = compute_domain_routed_field_loss(
            values,
            ["real", "real", "real"],
            real_geometry_targets=targets,
            real_group_ids=["scene-a", "scene-a", "scene-a"],
            real_order_weight=1.0,
            real_order_margin=0.5,
        )

        self.assertGreater(result.diagnostics["real_order_loss"], 0.0)
        self.assertEqual(result.diagnostics["real_order_pairs"], 6.0)

    def test_counterfactual_rows_only_use_hazard_and_quality_ranking(self):
        values = torch.tensor(
            [
                [0.0, 0.0],
                [-2.0, -2.0],
                [2.0, 2.0],
                [-1.0, -1.0],
            ],
            requires_grad=True,
        )
        domains = ["real", "counterfactual", "counterfactual", "counterfactual"]
        geometry_a = torch.tensor([[1.0, 1.0], [float("nan"), float("nan")], [7.0, 8.0], [-9.0, 2.0]])
        geometry_b = geometry_a.clone()
        geometry_b[1:] = torch.tensor([[100.0, -100.0], [-200.0, 300.0], [400.0, -500.0]])
        hazards = torch.tensor([False, False, True, False])
        quality = torch.tensor([float("nan"), 0.9, 0.1, 0.5])
        hazard_types = ["", "", "自车行为引起", ""]

        result_a = compute_domain_routed_field_loss(
            values,
            domains,
            real_geometry_targets=geometry_a,
            cf_hazard=hazards,
            cf_hazard_types=hazard_types,
            cf_quality=quality,
            cf_ranking_margin=0.5,
        )
        result_b = compute_domain_routed_field_loss(
            values,
            domains,
            real_geometry_targets=geometry_b,
            cf_hazard=hazards,
            cf_hazard_types=hazard_types,
            cf_quality=quality,
            cf_ranking_margin=0.5,
        )

        self.assertTrue(torch.allclose(result_a.loss, result_b.loss))
        self.assertTrue(torch.isfinite(result_a.loss))
        result_a.loss.backward()
        self.assertGreater(float(values.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(values.grad[1:].abs().sum()), 0.0)
        self.assertEqual(result_a.diagnostics["cf_count"], 3.0)

    def test_all_counterfactual_batch_does_not_require_geometry_targets(self):
        values = torch.tensor([[1.0], [-1.0]], requires_grad=True)

        result = compute_domain_routed_field_loss(
            values,
            ["counterfactual", "counterfactual"],
            cf_hazard=torch.tensor([False, True]),
            cf_hazard_types=["", "自车行为引起"],
            cf_ranking_margin=1.0,
        )

        self.assertTrue(torch.allclose(result.loss, torch.tensor(0.0)))
        result.loss.backward()
        self.assertIsNotNone(values.grad)

    def test_counterfactual_batch_requires_ranking_supervision(self):
        with self.assertRaisesRegex(ValueError, "hazard or quality"):
            compute_domain_routed_field_loss(
                torch.zeros(2, 1),
                ["counterfactual", "counterfactual"],
            )

    def test_zero_cf_weights_allow_forward_only_counterfactual_rows(self):
        values = torch.tensor([[0.5], [-0.5]], requires_grad=True)

        result = compute_domain_routed_field_loss(
            values,
            ["counterfactual", "counterfactual"],
            cf_hazard_weight=0.0,
            cf_quality_weight=0.0,
        )

        self.assertTrue(torch.equal(result.loss, torch.tensor(0.0)))
        result.loss.backward()
        self.assertIsNotNone(values.grad)
        self.assertEqual(result.diagnostics["cf_hazard_pairs"], 0.0)
        self.assertEqual(result.diagnostics["cf_quality_pairs"], 0.0)

    def test_targets_and_quality_must_use_unit_interval(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            compute_domain_routed_field_loss(
                torch.zeros(1, 1),
                ["real"],
                real_geometry_targets=torch.tensor([[1.1]]),
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            compute_domain_routed_field_loss(
                torch.zeros(2, 1),
                ["counterfactual", "counterfactual"],
                cf_quality=torch.tensor([0.1, -0.1]),
            )

    def test_hazard_types_are_required_and_type_balanced(self):
        values = torch.tensor([[0.9], [0.1], [0.2]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "cf_hazard_types"):
            compute_domain_routed_field_loss(
                values,
                ["counterfactual"] * 3,
                cf_hazard=torch.tensor([False, True, True]),
            )

        result = compute_domain_routed_field_loss(
            values,
            ["counterfactual"] * 3,
            cf_hazard=torch.tensor([False, True, True]),
            cf_hazard_types=["", "自车行为引起", "有事故但与自车无关"],
        )
        self.assertEqual(result.diagnostics["cf_hazard_type_count"], 2.0)

    def test_navsim_e120_hazard_ranking_uses_only_explicit_matched_real_cf_pairs(self):
        values = torch.tensor([[0.9], [0.2], [0.1]], requires_grad=True)

        result = compute_navsim_e120_quality_field_loss(
            values,
            ["real", "counterfactual", "real"],
            real_quality_targets=torch.tensor([[0.9], [0.1]]),
            cf_hazard=torch.tensor([False, True, False]),
            cf_hazard_types=["", "非自车行为引起", ""],
            cf_hazard_pair_real_indices=torch.tensor([0]),
            cf_hazard_pair_counterfactual_indices=torch.tensor([1]),
            cf_quality_weight=0.0,
            cf_ranking_margin=1.0,
        )

        self.assertTrue(torch.allclose(result.loss, torch.tensor(0.3)))
        self.assertEqual(result.diagnostics["cf_hazard_pairs"], 1.0)
        self.assertEqual(result.diagnostics["cf_hazard_type_count"], 1.0)

    def test_navsim_e120_hazard_ranking_rejects_non_allowlisted_type(self):
        with self.assertRaisesRegex(ValueError, "exact accident_type allowlist"):
            compute_navsim_e120_quality_field_loss(
                torch.tensor([[0.9], [0.2]]),
                ["real", "counterfactual"],
                real_quality_targets=torch.tensor([[0.9]]),
                cf_hazard=torch.tensor([False, True]),
                cf_hazard_types=["", "有事故但与自车无关"],
                cf_hazard_pair_real_indices=torch.tensor([0]),
                cf_hazard_pair_counterfactual_indices=torch.tensor([1]),
                cf_quality_weight=0.0,
            )


class TestRealStopValueLoss(unittest.TestCase):
    def test_real_stop_calibration_uses_huber(self):
        stop_values = torch.tensor([[0.0, 0.4, 1.0]], requires_grad=True)
        targets = torch.tensor([[0.2, 0.5, 0.8]])

        result = compute_real_stop_value_loss(stop_values, targets, ["real"])

        expected = torch.nn.functional.huber_loss(stop_values, targets, reduction="mean", delta=1.0)
        self.assertTrue(torch.allclose(result.loss, expected))
        self.assertEqual(result.diagnostics["real_count"], 1.0)

    def test_counterfactual_stop_calibration_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "real-only"):
            compute_real_stop_value_loss(
                torch.zeros(2, 2),
                torch.zeros(2, 2),
                ["real", "counterfactual"],
            )

    def test_stop_targets_must_use_unit_interval(self):
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            compute_real_stop_value_loss(
                torch.zeros(1, 2),
                torch.tensor([[0.2, 1.2]]),
                ["real"],
            )


if __name__ == "__main__":
    unittest.main()
