import unittest

import torch

from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.latent_value_guidance import (
    apply_latent_value_guidance,
    reduce_temporal_values,
)


class LinearToyValueHead(torch.nn.Module):
    def forward(self, z_future, *, tokens_per_frame=None):
        if z_future.ndim != 3:
            raise ValueError("test head expects [B, N, D]")
        if tokens_per_frame is None:
            raise ValueError("tokens_per_frame is required")
        frames = z_future.reshape(z_future.shape[0], z_future.shape[1] // tokens_per_frame, tokens_per_frame, -1)
        return frames[..., 0].mean(dim=2)


class TestLatentValueGuidance(unittest.TestCase):
    def test_reduce_temporal_values_objectives(self):
        values = torch.tensor([[1.0, 2.0, 4.0], [2.0, 4.0, 8.0]])

        self.assertTrue(torch.allclose(reduce_temporal_values(values, "last", gamma=0.5), torch.tensor(6.0)))
        self.assertTrue(torch.allclose(reduce_temporal_values(values, "mean", gamma=0.5), torch.tensor(3.5)))
        self.assertTrue(
            torch.allclose(
                reduce_temporal_values(values, "discounted", gamma=0.5),
                torch.tensor(((1.0 + 1.0 + 1.0) + (2.0 + 2.0 + 2.0)) / 2.0),
            )
        )

    def test_apply_latent_value_guidance_increases_toy_value_and_clamps_delta(self):
        cfg = parse_training_config(
            {
                "value_planning": {"enabled": True, "variant": "latent_guidance"},
                "value_guidance": {
                    "enabled": True,
                    "steps": 4,
                    "step_size": 0.2,
                    "max_delta_norm": 0.25,
                    "objective": "mean",
                    "detach_output": True,
                },
            }
        )
        z_future = torch.zeros(2, 3 * 2, 4)
        head = LinearToyValueHead()

        guided, diagnostics = apply_latent_value_guidance(
            z_future,
            head,
            tokens_per_frame=2,
            config=cfg,
        )

        before = head(z_future, tokens_per_frame=2).mean()
        after = head(guided, tokens_per_frame=2).mean()
        self.assertGreater(float(after), float(before))
        self.assertEqual(tuple(guided.shape), tuple(z_future.shape))
        self.assertFalse(guided.requires_grad)
        self.assertLessEqual(float((guided - z_future).norm(dim=-1).max()), 0.25 + 1e-6)
        self.assertEqual(diagnostics["guidance_steps"], 4.0)

    def test_disabled_guidance_returns_input_identity(self):
        cfg = parse_training_config({"value_guidance": {"enabled": False}})
        z_future = torch.randn(1, 4, 3)
        guided, diagnostics = apply_latent_value_guidance(
            z_future,
            LinearToyValueHead(),
            tokens_per_frame=2,
            config=cfg,
        )

        self.assertIs(guided, z_future)
        self.assertEqual(diagnostics["guidance_steps"], 0.0)

    def test_should_apply_value_guidance_skips_empty_dynamic_rollout_when_allowed(self):
        from app.vjepa_cowa_world_model.training.latent_value_guidance import should_apply_value_guidance

        z_future = torch.zeros(2, 0, 4)

        self.assertFalse(
            should_apply_value_guidance(
                z_future,
                value_guidance_enabled=True,
                allow_empty_rollout_skip=True,
            )
        )

    def test_should_apply_value_guidance_rejects_empty_without_dynamic_skip(self):
        from app.vjepa_cowa_world_model.training.latent_value_guidance import should_apply_value_guidance

        z_future = torch.zeros(2, 0, 4)

        with self.assertRaisesRegex(ValueError, "rollout_future_steps=0"):
            should_apply_value_guidance(
                z_future,
                value_guidance_enabled=True,
                allow_empty_rollout_skip=False,
            )

    def test_should_apply_value_guidance_returns_false_when_disabled(self):
        from app.vjepa_cowa_world_model.training.latent_value_guidance import should_apply_value_guidance

        z_future = torch.zeros(2, 0, 4)

        self.assertFalse(
            should_apply_value_guidance(
                z_future,
                value_guidance_enabled=False,
                allow_empty_rollout_skip=False,
            )
        )


if __name__ == "__main__":
    unittest.main()
