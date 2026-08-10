import unittest

import torch

from app.vjepa_cowa_world_model.models.trajectory_value import TemporalTrajectoryValueHead


class TestTemporalTrajectoryValueHead(unittest.TestCase):
    def test_flat_future_tokens_return_per_step_values(self):
        head = TemporalTrajectoryValueHead(embed_dim=4, hidden_dim=8, dropout=0.0)
        z_future = torch.randn(2, 3 * 5, 4)

        values = head(z_future, tokens_per_frame=5)

        self.assertEqual(tuple(values.shape), (2, 3))

    def test_frame_token_input_return_per_step_values(self):
        head = TemporalTrajectoryValueHead(embed_dim=4, hidden_dim=8, dropout=0.0)
        z_future = torch.randn(2, 3, 5, 4)

        values = head(z_future)

        self.assertEqual(tuple(values.shape), (2, 3))

    def test_rejects_flat_tokens_not_divisible_by_tokens_per_frame(self):
        head = TemporalTrajectoryValueHead(embed_dim=4, hidden_dim=8, dropout=0.0)

        with self.assertRaisesRegex(ValueError, "divide"):
            head(torch.randn(2, 14, 4), tokens_per_frame=5)

    def test_rejects_wrong_embed_dim(self):
        head = TemporalTrajectoryValueHead(embed_dim=4, hidden_dim=8, dropout=0.0)

        with self.assertRaisesRegex(ValueError, "embed_dim"):
            head(torch.randn(2, 3, 5), tokens_per_frame=1)


if __name__ == "__main__":
    unittest.main()
