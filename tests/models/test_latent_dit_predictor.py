"""Tests for latent-token DiT predictor."""

import unittest
from types import MethodType

import torch

from app.vjepa_cowa_world_model.models.latent_dit_predictor import LatentDiTPredictor


class TestLatentDiTPredictor(unittest.TestCase):
    def test_forward_joint_returns_world_and_action_predictions(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
            joint_action_enabled=True,
            joint_action_dim=3,
        )

        output = predictor.forward_joint(
            noisy_future_tokens=torch.randn(2, 6, 8),
            timesteps=torch.full((2,), 0.5),
            observed_tokens=torch.randn(2, 4, 8),
            noisy_future_actions=torch.randn(2, 3, 3),
            action_timesteps=torch.full((2, 3), 0.5),
            action_state_tokens=torch.randn(2, 3, 7),
        )

        self.assertEqual(tuple(output["world_pred"].shape), (2, 6, 8))
        self.assertEqual(tuple(output["action_pred"].shape), (2, 3, 3))

    def test_forward_joint_accepts_partial_future_window(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=4,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
            joint_action_enabled=True,
            joint_action_dim=3,
        )

        output = predictor.forward_joint(
            noisy_future_tokens=torch.randn(2, 4, 8),
            timesteps=torch.full((2,), 0.5),
            observed_tokens=torch.randn(2, 4, 8),
            noisy_future_actions=torch.randn(2, 2, 3),
            action_timesteps=torch.full((2, 2), 0.5),
            action_state_tokens=torch.randn(2, 2, 7),
            future_token_indices=torch.arange(2, 6),
        )

        self.assertEqual(tuple(output["world_pred"].shape), (2, 4, 8))
        self.assertEqual(tuple(output["action_pred"].shape), (2, 2, 3))

    def test_forward_joint_requires_enabled_branch(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
        )

        with self.assertRaisesRegex(ValueError, "joint_action_enabled"):
            predictor.forward_joint(
                noisy_future_tokens=torch.randn(1, 6, 8),
                timesteps=torch.full((1,), 0.5),
                observed_tokens=torch.randn(1, 4, 8),
                noisy_future_actions=torch.randn(1, 3, 3),
                action_timesteps=torch.full((1, 3), 0.5),
                action_state_tokens=torch.randn(1, 3, 7),
            )

    def test_joint_action_attention_mask_blocks_future_latent_leakage(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
            joint_action_enabled=True,
            joint_action_dim=3,
        )

        mask = predictor._build_joint_action_attn_mask(device=torch.device("cpu"))

        latent0 = 0
        latent2 = predictor.tokens_per_frame * 2
        action0 = predictor.num_future_tokens
        state0 = predictor.num_future_tokens + predictor.num_future_steps
        self.assertFalse(mask[latent0, action0].item())
        self.assertFalse(mask[latent0, state0].item())
        self.assertTrue(mask[action0, latent2].item())

    def test_partial_joint_action_attention_mask_blocks_future_active_tokens(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=4,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
            joint_action_enabled=True,
            joint_action_dim=3,
        )

        mask = predictor._build_joint_action_attn_mask(device=torch.device("cpu"), active_future_tokens=4)

        action0 = 4
        action1 = 5
        state0 = 6
        self.assertEqual(tuple(mask.shape), (8, 8))
        self.assertFalse(mask[0, action0].item())
        self.assertFalse(mask[0, state0].item())
        self.assertTrue(mask[0, action1].item())
        self.assertTrue(mask[action0, 2].item())
        self.assertFalse(mask[action1, 2].item())

    def test_training_forward_returns_finite_losses_and_clean_tokens(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            x0_loss_weight=0.25,
        )
        observed_tokens = torch.randn(2, 4, 8)
        target_future_tokens = torch.randn(2, 6, 8)
        actions = torch.randn(2, 5, 3)
        states = torch.randn(2, 5, 7)
        extrinsics = torch.randn(2, 5, 7)

        output = predictor.forward_train(
            observed_tokens=observed_tokens,
            target_future_tokens=target_future_tokens,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
        )

        self.assertEqual(tuple(output["x0_pred"].shape), (2, 6, 8))
        self.assertEqual(tuple(output["velocity_pred"].shape), (2, 6, 8))
        self.assertEqual(tuple(output["velocity_target"].shape), (2, 6, 8))
        for key in ["loss", "flow_loss", "x0_loss"]:
            self.assertTrue(torch.isfinite(output[key]).all(), key)

    def test_x0_prediction_forward_train_directly_supervises_clean_tokens(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
        )
        observed_tokens = torch.randn(1, 2, 4)
        target_future_tokens = torch.randn(1, 4, 4)

        def perfect_x0(
            self,
            noisy_future_tokens,
            timesteps,
            observed_tokens,
            actions=None,
            states=None,
            extrinsics=None,
            condition_cache=None,
            anchor_tokens=None,
            future_token_indices=None,
            known_future_tokens=None,
            known_future_token_indices=None,
            metadata_condition_mask=None,
        ):
            del noisy_future_tokens, timesteps, observed_tokens, actions, states, extrinsics, condition_cache
            del anchor_tokens, future_token_indices, known_future_tokens, known_future_token_indices
            del metadata_condition_mask
            return target_future_tokens

        predictor.forward = MethodType(perfect_x0, predictor)

        output = predictor.forward_train(
            observed_tokens=observed_tokens,
            target_future_tokens=target_future_tokens,
        )

        self.assertEqual(output["objective"], "x0_prediction")
        self.assertEqual(tuple(output["x0_pred"].shape), (1, 4, 4))
        self.assertEqual(tuple(output["velocity_pred"].shape), (1, 4, 4))
        self.assertEqual(tuple(output["velocity_target"].shape), (1, 4, 4))
        self.assertLess(output["loss"].item(), 1e-10)
        self.assertLess(output["x0_loss"].item(), 1e-10)
        self.assertTrue(torch.allclose(output["loss"], output["objective_loss"]))

    def test_bottleneck_forward_keeps_external_token_dimension(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            bottleneck_dim=4,
            conditioning_mode="temporal_aux_tokens",
        )
        observed_tokens = torch.randn(2, 4, 8)
        target_future_tokens = torch.randn(2, 6, 8)

        output = predictor.forward_train(
            observed_tokens=observed_tokens,
            target_future_tokens=target_future_tokens,
        )

        self.assertEqual(predictor.bottleneck_dim, 4)
        self.assertEqual(tuple(output["x0_pred"].shape), (2, 6, 8))
        self.assertTrue(torch.isfinite(output["loss"]).all())

    def test_sample_returns_future_clean_tokens(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
        )
        observed_tokens = torch.randn(2, 4, 8)
        actions = torch.randn(2, 5, 3)
        states = torch.randn(2, 5, 7)
        extrinsics = torch.randn(2, 5, 7)

        sampled = predictor.sample(
            observed_tokens=observed_tokens,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            num_inference_steps=4,
        )

        self.assertEqual(tuple(sampled.shape), (2, 6, 8))
        self.assertTrue(torch.isfinite(sampled).all())

    def test_masked_sample_returns_only_requested_active_tokens(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            conditioning_mode="temporal_aux_tokens",
        )
        observed_tokens = torch.randn(1, 2, 4)
        calls = []

        def constant_velocity(
            self,
            noisy_future_tokens,
            timesteps,
            observed_tokens,
            actions=None,
            states=None,
            extrinsics=None,
            condition_cache=None,
            anchor_tokens=None,
            future_token_indices=None,
            known_future_tokens=None,
            known_future_token_indices=None,
        ):
            del timesteps, observed_tokens, actions, states, extrinsics, condition_cache, anchor_tokens
            del known_future_tokens, known_future_token_indices
            calls.append(tuple(future_token_indices.tolist()))
            return torch.ones_like(noisy_future_tokens)

        predictor.forward = MethodType(constant_velocity, predictor)

        sampled = predictor.sample(
            observed_tokens=observed_tokens,
            num_inference_steps=1,
            sampler_type="euler",
            schedule_type="uniform",
            temperature=0.0,
            future_token_indices=torch.tensor([2, 3]),
        )

        self.assertEqual(tuple(sampled.shape), (1, 2, 4))
        self.assertEqual(calls, [(2, 3)])
        self.assertTrue(torch.allclose(sampled, torch.ones_like(sampled)))

    def test_x0_prediction_sample_uses_ddim_update_and_masked_indices(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
            conditioning_mode="temporal_aux_tokens",
        )
        observed_tokens = torch.randn(1, 2, 4)
        calls = []

        def constant_x0(
            self,
            noisy_future_tokens,
            timesteps,
            observed_tokens,
            actions=None,
            states=None,
            extrinsics=None,
            condition_cache=None,
            anchor_tokens=None,
            future_token_indices=None,
            known_future_tokens=None,
            known_future_token_indices=None,
            metadata_condition_mask=None,
        ):
            del timesteps, observed_tokens, actions, states, extrinsics, condition_cache, anchor_tokens
            del known_future_tokens, known_future_token_indices, metadata_condition_mask
            calls.append(None if future_token_indices is None else tuple(future_token_indices.tolist()))
            return torch.full_like(noisy_future_tokens, 3.0)

        predictor.forward = MethodType(constant_x0, predictor)

        full = predictor.sample(
            observed_tokens=observed_tokens,
            num_inference_steps=2,
            sampler_type="euler",
            schedule_type="uniform",
            temperature=0.0,
        )
        masked = predictor.sample(
            observed_tokens=observed_tokens,
            num_inference_steps=2,
            sampler_type="euler",
            schedule_type="uniform",
            temperature=0.0,
            future_token_indices=torch.tensor([2, 3]),
        )

        self.assertEqual(tuple(full.shape), (1, 6, 4))
        self.assertEqual(tuple(masked.shape), (1, 2, 4))
        self.assertTrue(torch.allclose(full, torch.full_like(full, 3.0), atol=1e-6))
        self.assertTrue(torch.allclose(masked, torch.full_like(masked, 3.0), atol=1e-6))
        self.assertEqual(calls[-1], (2, 3))

    def test_x0_prediction_sample_rejects_heun_sampler(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
        )
        observed_tokens = torch.randn(1, 2, 4)

        with self.assertRaisesRegex(ValueError, "x0_prediction.*sampler_type='euler'"):
            predictor.sample(observed_tokens=observed_tokens, sampler_type="heun")

    def test_masked_forward_uses_absolute_future_position_embeddings(self):
        predictor = LatentDiTPredictor(
            embed_dim=3,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=3,
            depth=1,
            num_heads=1,
            dropout=0.0,
            conditioning_mode="temporal_aux_tokens",
        )
        captured = {}

        class CaptureBlock(torch.nn.Module):
            def forward(self, x, cond_tokens, timestep_embed, cond_key_padding_mask=None):
                del cond_key_padding_mask
                del cond_tokens, timestep_embed
                captured["x"] = x.detach().clone()
                return x

        with torch.no_grad():
            predictor.input_proj.weight.copy_(torch.eye(3))
            predictor.input_proj.bias.zero_()
            predictor.pos_embed.zero_()
            for idx in range(predictor.num_future_tokens):
                predictor.pos_embed[:, idx, :] = float(idx)
        predictor.blocks = torch.nn.ModuleList([CaptureBlock()])

        predictor(
            noisy_future_tokens=torch.zeros(1, 2, 3),
            timesteps=torch.tensor([0.5]),
            observed_tokens=torch.randn(1, 2, 3),
            future_token_indices=torch.tensor([2, 3]),
        )

        expected = torch.tensor([[[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]])
        self.assertTrue(torch.equal(captured["x"], expected))

    def test_known_future_prefix_is_added_to_condition_tokens(self):
        predictor = LatentDiTPredictor(
            embed_dim=3,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=3,
            depth=1,
            num_heads=1,
            dropout=0.0,
            conditioning_mode="temporal_aux_tokens",
        )
        captured = {}

        class CaptureBlock(torch.nn.Module):
            def forward(self, x, cond_tokens, timestep_embed, cond_key_padding_mask=None):
                del cond_key_padding_mask
                del x, timestep_embed
                captured["cond_tokens"] = cond_tokens.detach().clone()
                return torch.zeros(1, 2, 3)

        with torch.no_grad():
            predictor.context_proj.weight.copy_(torch.eye(3))
            predictor.context_proj.bias.zero_()
            predictor.pos_embed.zero_()
            predictor.future_condition_type_embed.zero_()
        predictor.blocks = torch.nn.ModuleList([CaptureBlock()])

        observed_tokens = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        known_future_tokens = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]])
        predictor(
            noisy_future_tokens=torch.zeros(1, 2, 3),
            timesteps=torch.tensor([0.5]),
            observed_tokens=observed_tokens,
            future_token_indices=torch.tensor([2, 3]),
            known_future_tokens=known_future_tokens,
            known_future_token_indices=torch.tensor([0, 1]),
        )

        self.assertEqual(tuple(captured["cond_tokens"].shape), (1, 4, 3))
        self.assertTrue(torch.equal(captured["cond_tokens"][:, :2], observed_tokens))
        self.assertTrue(torch.equal(captured["cond_tokens"][:, 2:], known_future_tokens))

    def test_masked_forward_rejects_invalid_future_indices(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
        )
        observed_tokens = torch.randn(1, 2, 4)

        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            predictor(
                noisy_future_tokens=torch.zeros(1, 2, 4),
                timesteps=torch.tensor([0.5]),
                observed_tokens=observed_tokens,
                future_token_indices=torch.tensor([1, 1]),
            )
        with self.assertRaisesRegex(ValueError, "out of range"):
            predictor(
                noisy_future_tokens=torch.zeros(1, 1, 4),
                timesteps=torch.tensor([0.5]),
                observed_tokens=observed_tokens,
                future_token_indices=torch.tensor([4]),
            )
        with self.assertRaisesRegex(ValueError, "known_future_tokens"):
            predictor(
                noisy_future_tokens=torch.zeros(1, 1, 4),
                timesteps=torch.tensor([0.5]),
                observed_tokens=observed_tokens,
                future_token_indices=torch.tensor([1]),
                known_future_tokens=torch.zeros(1, 2, 4),
                known_future_token_indices=torch.tensor([0]),
            )
        with self.assertRaisesRegex(ValueError, "prefix before active"):
            predictor(
                noisy_future_tokens=torch.zeros(1, 1, 4),
                timesteps=torch.tensor([0.5]),
                observed_tokens=observed_tokens,
                future_token_indices=torch.tensor([1]),
                known_future_tokens=torch.zeros(1, 1, 4),
                known_future_token_indices=torch.tensor([1]),
            )

    def test_anchor_frame_is_prepended_to_denoising_sequence_when_enabled(self):
        predictor = LatentDiTPredictor(
            embed_dim=3,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=3,
            depth=1,
            num_heads=1,
            dropout=0.0,
            use_anchor_frame=True,
            conditioning_mode="temporal_aux_tokens",
        )
        captured = {}

        class CaptureBlock(torch.nn.Module):
            def forward(self, x, cond_tokens, timestep_embed, cond_key_padding_mask=None):
                del cond_key_padding_mask, cond_tokens, timestep_embed
                captured["x"] = x.detach().clone()
                return x

        with torch.no_grad():
            predictor.input_proj.weight.copy_(torch.eye(3))
            predictor.input_proj.bias.zero_()
            predictor.pos_embed.zero_()
        predictor.blocks = torch.nn.ModuleList([CaptureBlock()])

        observed_tokens = torch.tensor(
            [
                [
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [10.0, 10.0, 10.0],
                    [20.0, 20.0, 20.0],
                ]
            ]
        )
        anchor_tokens = observed_tokens[:, -2:].clone()
        noisy_future_tokens = torch.zeros(1, 4, 3)

        output = predictor(
            noisy_future_tokens=noisy_future_tokens,
            timesteps=torch.tensor([0.5]),
            observed_tokens=observed_tokens,
            anchor_tokens=anchor_tokens,
        )

        self.assertEqual(tuple(captured["x"].shape), (1, 6, 3))
        self.assertTrue(torch.equal(captured["x"][:, :2], anchor_tokens))
        self.assertEqual(tuple(output.shape), (1, 4, 3))

    def test_sample_supports_heun_nonuniform_schedule_and_diagnostics(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            conditioning_mode="temporal_aux_tokens",
        )
        observed_tokens = torch.randn(1, 2, 4)
        calls = []

        def constant_velocity(
            self,
            noisy_future_tokens,
            timesteps,
            observed_tokens,
            actions=None,
            states=None,
            extrinsics=None,
            condition_cache=None,
            anchor_tokens=None,
        ):
            del anchor_tokens
            calls.append(timesteps.detach().clone())
            return torch.ones_like(noisy_future_tokens)

        predictor.forward = MethodType(constant_velocity, predictor)

        output = predictor.sample(
            observed_tokens=observed_tokens,
            num_inference_steps=4,
            sampler_type="heun",
            schedule_type="cosine",
            temperature=0.0,
            return_diagnostics=True,
        )

        self.assertEqual(tuple(output["samples"].shape), (1, 4, 4))
        self.assertTrue(torch.allclose(output["samples"], torch.ones_like(output["samples"]), atol=1e-6))
        self.assertEqual(output["sampler_type"], "heun")
        self.assertEqual(output["schedule_type"], "cosine")
        self.assertEqual(tuple(output["timesteps"].shape), (4,))
        self.assertEqual(tuple(output["deltas"].shape), (4,))
        self.assertGreater(len(calls), 4)

    def test_sample_applies_metadata_cfg_only_to_valid_samples(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            conditioning_mode="temporal_aux_tokens",
        )
        observed_tokens = torch.randn(2, 2, 4)
        actions = torch.randn(2, 2, 3)
        states = torch.randn(2, 2, 7)
        extrinsics = torch.randn(2, 2, 7)
        calls = []

        def conditional_velocity(
            self,
            noisy_future_tokens,
            timesteps,
            observed_tokens,
            actions=None,
            states=None,
            extrinsics=None,
            condition_cache=None,
            anchor_tokens=None,
        ):
            del timesteps, observed_tokens, states, extrinsics, condition_cache, anchor_tokens
            calls.append(actions is not None)
            if actions is None:
                return torch.zeros_like(noisy_future_tokens)
            return torch.ones_like(noisy_future_tokens)

        predictor.forward = MethodType(conditional_velocity, predictor)

        sampled = predictor.sample(
            observed_tokens=observed_tokens,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            num_inference_steps=1,
            sampler_type="euler",
            schedule_type="uniform",
            temperature=0.0,
            metadata_condition_mask=torch.tensor([True, False]),
            metadata_guidance_scale=2.0,
        )

        self.assertEqual(calls, [True, False])
        self.assertTrue(torch.allclose(sampled[0], torch.full_like(sampled[0], 2.0)))
        self.assertTrue(torch.allclose(sampled[1], torch.zeros_like(sampled[1])))

    def test_x0_prediction_sample_applies_metadata_cfg_in_x0_space(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
            conditioning_mode="temporal_aux_tokens",
        )
        observed_tokens = torch.randn(2, 2, 4)
        actions = torch.randn(2, 2, 3)
        states = torch.randn(2, 2, 7)
        extrinsics = torch.randn(2, 2, 7)
        calls = []

        def conditional_x0(
            self,
            noisy_future_tokens,
            timesteps,
            observed_tokens,
            actions=None,
            states=None,
            extrinsics=None,
            condition_cache=None,
            anchor_tokens=None,
        ):
            del timesteps, observed_tokens, states, extrinsics, condition_cache, anchor_tokens
            calls.append(actions is not None)
            if actions is None:
                return torch.zeros_like(noisy_future_tokens)
            return torch.ones_like(noisy_future_tokens)

        predictor.forward = MethodType(conditional_x0, predictor)

        sampled = predictor.sample(
            observed_tokens=observed_tokens,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            num_inference_steps=1,
            sampler_type="euler",
            schedule_type="uniform",
            temperature=0.0,
            metadata_condition_mask=torch.tensor([True, False]),
            metadata_guidance_scale=2.0,
        )

        self.assertEqual(calls, [True, False])
        self.assertTrue(torch.allclose(sampled[0], torch.full_like(sampled[0], 2.0)))
        self.assertTrue(torch.allclose(sampled[1], torch.zeros_like(sampled[1])))

    def test_temporal_aux_tokens_preserve_side_input_order(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            conditioning_mode="temporal_aux_tokens",
            max_steps=4,
        )
        observed_tokens = torch.randn(1, 2, 4)
        actions_a = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        actions_b = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
        states = torch.zeros(1, 2, 7)
        extrinsics = torch.zeros(1, 2, 7)

        cache_a = predictor._build_condition_cache(
            observed_tokens=observed_tokens,
            actions=actions_a,
            states=states,
            extrinsics=extrinsics,
        )
        cache_b = predictor._build_condition_cache(
            observed_tokens=observed_tokens,
            actions=actions_b,
            states=states,
            extrinsics=extrinsics,
        )

        self.assertEqual(tuple(cache_a["cond_tokens"].shape), (1, 4, 8))
        self.assertTrue(torch.allclose(cache_a["side_condition"], torch.zeros_like(cache_a["side_condition"])))
        self.assertFalse(torch.allclose(cache_a["cond_tokens"][:, 2:], cache_b["cond_tokens"][:, 2:]))

    def test_sample_rejects_unknown_sampler_or_schedule(self):
        predictor = LatentDiTPredictor(
            embed_dim=4,
            tokens_per_frame=2,
            num_future_steps=2,
            action_dim=3,
            state_dim=7,
            hidden_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
        )
        observed_tokens = torch.randn(1, 2, 4)

        with self.assertRaisesRegex(ValueError, "sampler_type"):
            predictor.sample(observed_tokens=observed_tokens, sampler_type="bad")
        with self.assertRaisesRegex(ValueError, "schedule_type"):
            predictor.sample(observed_tokens=observed_tokens, schedule_type="bad")

    def test_sample_joint_returns_world_and_action_shapes(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
            joint_action_enabled=True,
            joint_action_dim=3,
        )

        result = predictor.sample_joint(
            observed_tokens=torch.randn(2, 4, 8),
            action_state_tokens=torch.randn(2, 3, 7),
            num_inference_steps=2,
            sampler_type="euler",
            return_diagnostics=True,
        )

        self.assertEqual(tuple(result["samples"].shape), (2, 6, 8))
        self.assertEqual(tuple(result["actions"].shape), (2, 3, 3))
        self.assertEqual(result["objective"], "x0_prediction")

    def test_sample_joint_decoupled_inference_uses_distinct_world_and_action_schedules(self):
        predictor = LatentDiTPredictor(
            embed_dim=8,
            tokens_per_frame=2,
            num_future_steps=3,
            action_dim=3,
            state_dim=7,
            hidden_dim=16,
            depth=1,
            num_heads=2,
            dropout=0.0,
            objective="x0_prediction",
            joint_action_enabled=True,
            joint_action_dim=3,
            joint_action_inference_noise_mode="decoupled",
            joint_video_final_noise=0.8,
        )
        calls = []

        def fake_forward_joint(
            self,
            *,
            noisy_future_tokens,
            noisy_future_actions,
            timesteps,
            action_timesteps,
            **kwargs,
        ):
            del kwargs
            calls.append((timesteps.detach().clone(), action_timesteps.detach().clone()))
            return {
                "world_pred": torch.zeros_like(noisy_future_tokens),
                "action_pred": torch.zeros_like(noisy_future_actions),
            }

        predictor.forward_joint = MethodType(fake_forward_joint, predictor)

        result = predictor.sample_joint(
            observed_tokens=torch.randn(2, 4, 8),
            action_state_tokens=torch.randn(2, 3, 7),
            num_inference_steps=3,
            sampler_type="euler",
            return_diagnostics=True,
        )

        self.assertEqual(result["joint_action_inference_noise_mode"], "decoupled")
        self.assertAlmostEqual(result["joint_video_final_noise"], 0.8)
        self.assertAlmostEqual(result["world_final_t"], 0.2)
        self.assertTrue(calls)
        last_world_t, last_action_t = calls[-1]
        self.assertLess(last_world_t.max().item(), last_action_t.max().item())


if __name__ == "__main__":
    unittest.main()
