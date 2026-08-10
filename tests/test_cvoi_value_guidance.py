import unittest
from types import SimpleNamespace

import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixValueOutput
from app.vjepa_cowa_world_model.training.latent_value_guidance import (
    apply_cvoi_latent_value_guidance,
    cvoi_evaluation_guidance_steps,
)


def _guidance_config(*, protocol_version="formal_v2_navsim_e120_h4_v3", **overrides):
    values = {
        "enabled": True,
        "steps": 2,
        "step_size": 0.5,
        "max_delta_norm": 0.2,
        "objective": "last",
        "detach_output": True,
    }
    values.update(overrides)
    return SimpleNamespace(
        cvoi=SimpleNamespace(protocol_version=protocol_version),
        value_guidance=SimpleNamespace(**values),
    )


class OpposedDualValueHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.observed_requires_grad = []

    def forward(self, z_observed, z_future, *, tokens_per_frame=None):
        self.observed_requires_grad.append(z_observed.requires_grad)
        if tokens_per_frame is None:
            raise ValueError("tokens_per_frame is required")
        frames = z_future.reshape(
            z_future.shape[0],
            z_future.shape[1] // tokens_per_frame,
            tokens_per_frame,
            z_future.shape[-1],
        )
        observed_bias = z_observed[..., 0].mean(dim=1, keepdim=True)
        field_values = self.scale * (frames[..., 0].mean(dim=2) + observed_bias)
        h0 = -100.0 * observed_bias
        stop_values = torch.cat([h0, -100.0 * field_values], dim=1)
        return PrefixValueOutput(field_values=field_values, stop_values=stop_values)


class BombValueHead(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("h=0 guidance must not call the value model")


class EvalModeRecurrentValueHead(OpposedDualValueHead):
    def __init__(self):
        super().__init__()
        self.cudnn_enabled = []

    def forward(self, z_observed, z_future, *, tokens_per_frame=None):
        self.cudnn_enabled.append(torch.backends.cudnn.enabled)
        if not self.training and torch.backends.cudnn.enabled:
            raise RuntimeError("cudnn RNN backward can only be called in training mode")
        return super().forward(z_observed, z_future, tokens_per_frame=tokens_per_frame)


class TestCVoILatentValueGuidance(unittest.TestCase):
    def test_uses_last_field_value_and_isolates_predictor_and_observed_gradients(self):
        observed = torch.randn(2, 4, 3, requires_grad=True)
        future = torch.zeros(2, 3 * 2, 3, requires_grad=True)
        head = OpposedDualValueHead()

        guided, diagnostics = apply_cvoi_latent_value_guidance(
            observed,
            future,
            head,
            tokens_per_frame=2,
            config=_guidance_config(),
        )

        delta = (guided - future.detach()).reshape(2, 3, 2, 3)
        self.assertTrue(torch.equal(delta[:, :2], torch.zeros_like(delta[:, :2])))
        self.assertGreater(float(delta[:, -1, :, 0].min()), 0.0)
        self.assertLessEqual(float(delta.norm(dim=-1).max()), 0.2 + 1e-6)
        self.assertFalse(guided.requires_grad)
        self.assertIsNone(observed.grad)
        self.assertIsNone(future.grad)
        self.assertIsNone(head.scale.grad)
        self.assertTrue(head.observed_requires_grad)
        self.assertFalse(any(head.observed_requires_grad))
        self.assertGreater(diagnostics["field_value_after"], diagnostics["field_value_before"])
        self.assertEqual(diagnostics["guidance_steps"], 2.0)

    def test_h0_returns_detached_skip_without_calling_value_model(self):
        observed = torch.randn(2, 4, 3, requires_grad=True)
        empty_future = torch.empty(2, 0, 3, requires_grad=True)

        guided, diagnostics = apply_cvoi_latent_value_guidance(
            observed,
            empty_future,
            BombValueHead(),
            tokens_per_frame=2,
            config=_guidance_config(),
        )

        self.assertEqual(tuple(guided.shape), (2, 0, 3))
        self.assertFalse(guided.requires_grad)
        self.assertEqual(diagnostics["guidance_steps"], 0.0)
        self.assertEqual(diagnostics["guidance_skipped_h0"], 1.0)

    def test_guidance_update_is_invariant_to_batch_replication(self):
        observed = torch.randn(1, 4, 3)
        future = torch.zeros(1, 3 * 2, 3)
        head = OpposedDualValueHead()

        single, _ = apply_cvoi_latent_value_guidance(
            observed,
            future,
            head,
            tokens_per_frame=2,
            config=_guidance_config(),
        )
        repeated, _ = apply_cvoi_latent_value_guidance(
            observed.repeat(4, 1, 1),
            future.repeat(4, 1, 1),
            head,
            tokens_per_frame=2,
            config=_guidance_config(),
        )

        self.assertTrue(torch.allclose(repeated, single.repeat(4, 1, 1), atol=1e-6, rtol=1e-6))

    def test_eval_mode_recurrent_value_guidance_disables_cudnn_without_changing_model_mode(self):
        observed = torch.randn(1, 4, 3)
        future = torch.zeros(1, 3 * 2, 3)
        head = EvalModeRecurrentValueHead().eval()
        head.requires_grad_(False)

        with torch.backends.cudnn.flags(enabled=True):
            guided, diagnostics = apply_cvoi_latent_value_guidance(
                observed,
                future,
                head,
                tokens_per_frame=2,
                config=_guidance_config(),
            )
            self.assertTrue(torch.backends.cudnn.enabled)

        self.assertFalse(head.training)
        self.assertFalse(any(parameter.requires_grad for parameter in head.parameters()))
        self.assertTrue(head.cudnn_enabled)
        self.assertFalse(any(head.cudnn_enabled))
        self.assertGreater(diagnostics["field_value_after"], diagnostics["field_value_before"])
        self.assertFalse(guided.requires_grad)

    def test_requires_exactly_two_guidance_steps(self):
        with self.assertRaisesRegex(ValueError, "K=2"):
            apply_cvoi_latent_value_guidance(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 3),
                OpposedDualValueHead(),
                tokens_per_frame=2,
                config=_guidance_config(steps=1),
            )

    def test_evaluation_override_supports_navsim_e120_k8_with_eight_real_updates(self):
        for guidance_steps in (1, 2, 4, 8):
            with self.subTest(guidance_steps=guidance_steps):
                head = OpposedDualValueHead()
                guided, diagnostics = apply_cvoi_latent_value_guidance(
                    torch.zeros(1, 2, 3),
                    torch.zeros(1, 2, 3),
                    head,
                    tokens_per_frame=2,
                    config=_guidance_config(),
                    evaluation_guidance_steps=guidance_steps,
                )

                self.assertEqual(diagnostics["guidance_steps"], float(guidance_steps))
                self.assertEqual(len(head.observed_requires_grad), guidance_steps + 2)
                self.assertFalse(guided.requires_grad)

    def test_legacy_v1_evaluation_override_keeps_k3_compatibility(self):
        _, diagnostics = apply_cvoi_latent_value_guidance(
            torch.zeros(1, 2, 3),
            torch.zeros(1, 2, 3),
            OpposedDualValueHead(),
            tokens_per_frame=2,
            config=_guidance_config(protocol_version="legacy_v1"),
            evaluation_guidance_steps=3,
        )

        self.assertEqual(diagnostics["guidance_steps"], 3.0)

    def test_evaluation_override_rejects_invalid_k_and_config_drift(self):
        for guidance_steps in (0, 3, 5, True):
            with self.subTest(guidance_steps=guidance_steps):
                with self.assertRaisesRegex(ValueError, "evaluation_guidance_steps"):
                    apply_cvoi_latent_value_guidance(
                        torch.zeros(1, 2, 3),
                        torch.zeros(1, 2, 3),
                        OpposedDualValueHead(),
                        tokens_per_frame=2,
                        config=_guidance_config(),
                        evaluation_guidance_steps=guidance_steps,
                    )

        with self.assertRaisesRegex(ValueError, "configured K=2"):
            apply_cvoi_latent_value_guidance(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 3),
                OpposedDualValueHead(),
                tokens_per_frame=2,
                config=_guidance_config(steps=1),
                evaluation_guidance_steps=1,
            )

    def test_evaluation_grid_rejects_retired_generic_formal_v2_alias(self):
        with self.assertRaisesRegex(ValueError, "unsupported CVoI protocol_version"):
            cvoi_evaluation_guidance_steps(
                _guidance_config(protocol_version="formal_v2"),
                include_disabled=True,
            )

    def test_evaluation_override_is_still_skipped_at_h0(self):
        guided, diagnostics = apply_cvoi_latent_value_guidance(
            torch.zeros(1, 2, 3),
            torch.empty(1, 0, 3),
            BombValueHead(),
            tokens_per_frame=2,
            config=_guidance_config(),
            evaluation_guidance_steps=4,
        )

        self.assertEqual(tuple(guided.shape), (1, 0, 3))
        self.assertEqual(diagnostics["guidance_steps"], 0.0)

    def test_requires_last_objective(self):
        with self.assertRaisesRegex(ValueError, "objective.*last"):
            apply_cvoi_latent_value_guidance(
                torch.zeros(1, 2, 3),
                torch.zeros(1, 2, 3),
                OpposedDualValueHead(),
                tokens_per_frame=2,
                config=_guidance_config(objective="mean"),
            )

    def test_requires_detached_output_and_finite_hyperparameters(self):
        for overrides, message in (
            ({"detach_output": False}, "detach_output"),
            ({"step_size": float("nan")}, "step_size"),
            ({"max_delta_norm": float("inf")}, "max_delta_norm"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    apply_cvoi_latent_value_guidance(
                        torch.zeros(1, 2, 3),
                        torch.zeros(1, 2, 3),
                        OpposedDualValueHead(),
                        tokens_per_frame=2,
                        config=_guidance_config(**overrides),
                    )

    def test_disabled_guidance_preserves_input_identity(self):
        future = torch.randn(1, 2, 3)

        guided, diagnostics = apply_cvoi_latent_value_guidance(
            torch.randn(1, 2, 3),
            future,
            BombValueHead(),
            tokens_per_frame=2,
            config=_guidance_config(enabled=False),
        )

        self.assertIs(guided, future)
        self.assertEqual(diagnostics["guidance_steps"], 0.0)


if __name__ == "__main__":
    unittest.main()
