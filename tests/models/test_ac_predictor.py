import unittest

import torch
import torch.nn as nn

from src.models.ac_predictor import VisionTransformerPredictorAC


class TestACPredictorMaskToken(unittest.TestCase):
    def setUp(self):
        self.predictor = VisionTransformerPredictorAC(
            img_size=(64, 64),
            patch_size=16,
            num_frames=4,
            tubelet_size=2,
            embed_dim=192,
            predictor_embed_dim=128,
            depth=2,
            num_heads=4,
            action_embed_dim=3,
            use_extrinsics=False,
        )
        self.B = 2
        self.T = 2  # num_frames // tubelet_size = 4 // 2
        self.H, self.W = 4, 4
        self.P = self.H * self.W

    def test_has_action_mask_token(self):
        self.assertTrue(hasattr(self.predictor, "action_mask_token"))
        self.assertEqual(self.predictor.action_mask_token.shape, (1, 1, 128))

    @unittest.skip(
        "Per-layer output heads (layer_output_norms/layer_output_projs) are an optional, "
        "currently-disabled diagnostic feature: they are commented out in "
        "src/models/ac_predictor.py and checkpoint.py treats their keys as OPTIONAL "
        "(_OPTIONAL_OUTPUT_HEAD_PREFIXES). The model deliberately does not register them, so "
        "asserting their presence is stale. The active checkpoint-compat contract (old "
        "checkpoints without these heads load fine) lives in checkpoint.py, not here."
    )
    def test_has_per_layer_output_heads_for_checkpoint_compatibility(self):
        self.assertIsInstance(self.predictor.layer_output_norms, nn.ModuleList)
        self.assertIsInstance(self.predictor.layer_output_projs, nn.ModuleList)
        self.assertEqual(len(self.predictor.layer_output_norms), 2)
        self.assertEqual(len(self.predictor.layer_output_projs), 2)
        self.assertEqual(self.predictor.layer_output_norms[0].normalized_shape, (128,))
        self.assertEqual(self.predictor.layer_output_projs[0].in_features, 128)
        self.assertEqual(self.predictor.layer_output_projs[0].out_features, 192)

    @unittest.skip(
        "Per-layer output heads are an optional, currently-disabled feature (commented out in "
        "src/models/ac_predictor.py). The model does not register these modules, so a raw "
        "load_state_dict(strict=False) correctly reports them as 'unexpected'. Checkpoint "
        "compatibility for these prefixes is handled in checkpoint.py "
        "(_OPTIONAL_OUTPUT_HEAD_PREFIXES), not via raw load_state_dict on the bare model."
    )
    def test_accepts_layer_output_head_checkpoint_keys(self):
        state_dict = self.predictor.state_dict()
        for layer_idx in range(2):
            state_dict[f"layer_output_norms.{layer_idx}.weight"] = torch.ones(128)
            state_dict[f"layer_output_norms.{layer_idx}.bias"] = torch.zeros(128)
            state_dict[f"layer_output_projs.{layer_idx}.weight"] = torch.ones(192, 128)
            state_dict[f"layer_output_projs.{layer_idx}.bias"] = torch.zeros(192)

        missing, unexpected = self.predictor.load_state_dict(state_dict, strict=False)

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])

    def test_forward_without_mask(self):
        x = torch.randn(self.B, self.T * self.P, 192)
        actions = torch.randn(self.B, self.T, 3)
        states = torch.randn(self.B, self.T, 3)
        out = self.predictor(x, actions, states)
        self.assertEqual(out.shape, (self.B, self.T * self.P, 192))

    def test_forward_with_mask(self):
        x = torch.randn(self.B, self.T * self.P, 192)
        actions = torch.randn(self.B, self.T, 3)
        states = torch.randn(self.B, self.T, 3)
        action_mask = torch.zeros(self.B, self.T, dtype=torch.bool)
        action_mask[:, 1:] = True  # mask the second timestep onward
        out = self.predictor(x, actions, states, action_mask=action_mask)
        self.assertEqual(out.shape, (self.B, self.T * self.P, 192))

    def test_forward_builds_causal_mask_from_actual_multiview_token_grid(self):
        predictor = VisionTransformerPredictorAC(
            img_size=(64, 192),
            patch_size=16,
            num_frames=512,
            tubelet_size=2,
            embed_dim=192,
            predictor_embed_dim=128,
            depth=1,
            num_heads=4,
            action_embed_dim=3,
            use_extrinsics=False,
            is_frame_causal=True,
        )
        batch_size, num_steps, patches_per_step = 1, 2, 4 * 12
        x = torch.randn(batch_size, num_steps * patches_per_step, 192)
        actions = torch.randn(batch_size, num_steps, 3)
        states = torch.randn(batch_size, num_steps, 3)

        out = predictor(x, actions, states)

        self.assertIsNone(predictor.attn_mask)
        self.assertEqual(out.shape, (batch_size, num_steps * patches_per_step, 192))

    def test_mask_token_affects_output(self):
        torch.manual_seed(42)
        x = torch.randn(self.B, self.T * self.P, 192)
        actions = torch.randn(self.B, self.T, 3)
        states = torch.randn(self.B, self.T, 3)
        out_no_mask = self.predictor(x, actions, states)
        action_mask = torch.zeros(self.B, self.T, dtype=torch.bool)
        action_mask[:, 1:] = True  # mask the second timestep onward
        out_with_mask = self.predictor(x, actions, states, action_mask=action_mask)
        self.assertFalse(torch.allclose(out_no_mask, out_with_mask, atol=1e-5))


class TestACPredictorStateDim(unittest.TestCase):
    def test_separate_state_dim(self):
        predictor = VisionTransformerPredictorAC(
            img_size=(64, 64),
            patch_size=16,
            num_frames=4,
            tubelet_size=2,
            embed_dim=192,
            predictor_embed_dim=128,
            depth=2,
            num_heads=4,
            action_embed_dim=3,
            state_embed_dim=14,
            use_extrinsics=False,
        )
        B, T, P = 2, 2, 16  # T = num_frames // tubelet_size = 4 // 2
        x = torch.randn(B, T * P, 192)
        actions = torch.randn(B, T, 3)
        states = torch.randn(B, T, 14)
        out = predictor(x, actions, states)
        self.assertEqual(out.shape, (B, T * P, 192))

    def test_default_state_dim_equals_action_dim(self):
        predictor = VisionTransformerPredictorAC(
            img_size=(64, 64),
            patch_size=16,
            num_frames=4,
            tubelet_size=2,
            embed_dim=192,
            predictor_embed_dim=128,
            depth=2,
            num_heads=4,
            action_embed_dim=3,
            use_extrinsics=False,
        )
        self.assertEqual(predictor.state_encoder.in_features, 3)

    def test_state_dim_8_encoder(self):
        predictor = VisionTransformerPredictorAC(
            img_size=(64, 64),
            patch_size=16,
            num_frames=4,
            tubelet_size=2,
            embed_dim=192,
            predictor_embed_dim=128,
            depth=2,
            num_heads=4,
            action_embed_dim=3,
            state_embed_dim=8,
            use_extrinsics=False,
        )
        self.assertEqual(predictor.state_encoder.in_features, 8)
        self.assertEqual(predictor.action_encoder.in_features, 3)
