import unittest

import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel, PrefixValueOutput


class TestPrefixDualValueModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = PrefixDualValueModel(embed_dim=4, hidden_dim=6, num_layers=1, dropout=0.0)

    def test_forward_returns_field_and_stop_values_for_every_prefix(self):
        observed = torch.randn(2, 3, 5, 4)
        future = torch.randn(2, 4, 5, 4)

        output = self.model(observed, future)

        self.assertIsInstance(output, PrefixValueOutput)
        self.assertEqual(tuple(output.field_values.shape), (2, 4))
        self.assertEqual(tuple(output.stop_values.shape), (2, 5))

    def test_empty_future_represents_h0(self):
        observed = torch.randn(2, 3, 5, 4)
        future = torch.empty(2, 0, 5, 4)

        output = self.model(observed, future)

        self.assertEqual(tuple(output.field_values.shape), (2, 0))
        self.assertEqual(tuple(output.stop_values.shape), (2, 1))
        self.assertTrue(torch.isfinite(output.stop_values).all())

    def test_future_outputs_are_causal(self):
        observed = torch.randn(2, 2, 3, 4)
        future = torch.randn(2, 4, 3, 4)
        changed_future = future.clone()
        changed_future[:, 2:] = torch.randn_like(changed_future[:, 2:]) * 20.0

        output = self.model(observed, future)
        changed_output = self.model(observed, changed_future)

        self.assertTrue(torch.allclose(output.field_values[:, :2], changed_output.field_values[:, :2]))
        self.assertTrue(torch.allclose(output.stop_values[:, :3], changed_output.stop_values[:, :3]))

    def test_incremental_extension_matches_full_forward(self):
        observed = torch.randn(2, 2, 3, 4)
        future = torch.randn(2, 4, 3, 4)

        full = self.model(observed, future)
        observed_state = self.model.encode_observed(observed)
        first, first_state = self.model.extend_prefix(observed_state, future[:, :1])
        rest, final_state = self.model.extend_prefix(first_state, future[:, 1:])

        incremental_fields = torch.cat([first.field_values, rest.field_values], dim=1)
        incremental_stops = torch.cat([first.stop_values, rest.stop_values[:, 1:]], dim=1)
        self.assertTrue(torch.allclose(full.field_values, incremental_fields, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(full.stop_values, incremental_stops, atol=1e-6, rtol=1e-6))
        _, full_state = self.model.extend_prefix(observed_state, future)
        self.assertTrue(torch.allclose(full_state, final_state, atol=1e-6, rtol=1e-6))

    def test_flat_tokens_use_spatial_frame_pooling(self):
        observed = torch.randn(2, 2, 3, 4)
        future = torch.randn(2, 4, 3, 4)
        observed_flat = observed.reshape(2, 6, 4)
        future_flat = future.reshape(2, 12, 4)

        framed = self.model(observed, future)
        flat = self.model(observed_flat, future_flat, tokens_per_frame=3)

        self.assertTrue(torch.allclose(framed.field_values, flat.field_values, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(framed.stop_values, flat.stop_values, atol=1e-6, rtol=1e-6))

    def test_rejects_empty_observed_prefix(self):
        observed = torch.empty(2, 0, 3, 4)
        future = torch.randn(2, 1, 3, 4)

        with self.assertRaisesRegex(ValueError, "observed.*at least one frame"):
            self.model(observed, future)


if __name__ == "__main__":
    unittest.main()
