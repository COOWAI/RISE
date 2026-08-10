"""Tests for PETR-style multi-view token fusion."""

import unittest

import torch

from app.vjepa_cowa_world_model import models
from app.vjepa_cowa_world_model.models import PETRMultiViewFusion


class TestPETRMultiViewFusion(unittest.TestCase):
    def test_exported_from_models_package(self):
        self.assertIn("PETRMultiViewFusion", models.__all__)
        self.assertIs(models.PETRMultiViewFusion, PETRMultiViewFusion)

    def test_fuses_view_tokens_to_single_temporal_token_stream(self):
        fusion = PETRMultiViewFusion(
            embed_dim=8,
            tokens_per_frame=4,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
            depth_num=4,
        )
        raw_tokens = torch.randn(2, 3, 12, 8, requires_grad=True)
        intrinsics = torch.eye(3).view(1, 1, 1, 3, 3).repeat(2, 3, 3, 1, 1)
        camera2ego = torch.eye(4).view(1, 1, 1, 4, 4).repeat(2, 3, 3, 1, 1)

        fused = fusion(raw_tokens, camera_intrinsics=intrinsics, camera2ego=camera2ego)

        self.assertEqual(tuple(fused.shape), (2, 12, 8))
        loss = fused.square().mean()
        loss.backward()
        self.assertIsNotNone(raw_tokens.grad)
        self.assertGreater(float(raw_tokens.grad.abs().sum()), 0.0)

    def test_fuses_5d_vit_tokens_to_single_temporal_token_stream(self):
        fusion = PETRMultiViewFusion(
            embed_dim=8,
            tokens_per_frame=4,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
            depth_num=4,
        )
        raw_tokens = torch.randn(2, 3, 3, 4, 8, requires_grad=True)
        intrinsics = torch.eye(3).view(1, 1, 1, 3, 3).repeat(2, 3, 3, 1, 1)
        camera2ego = torch.eye(4).view(1, 1, 1, 4, 4).repeat(2, 3, 3, 1, 1)

        fused = fusion(raw_tokens, camera_intrinsics=intrinsics, camera2ego=camera2ego, image_shape=(2, 2))

        self.assertEqual(tuple(fused.shape), (2, 12, 8))
        loss = fused.square().mean()
        loss.backward()
        self.assertIsNotNone(raw_tokens.grad)
        self.assertGreater(float(raw_tokens.grad.abs().sum()), 0.0)

    def test_per_view_mode_keeps_each_camera_query_separate(self):
        fusion = PETRMultiViewFusion(
            embed_dim=8,
            tokens_per_frame=4,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0,
            depth_num=4,
            output_mode="per_view",
        )
        raw_tokens = torch.randn(2, 3, 3, 4, 8, requires_grad=True)
        intrinsics = torch.eye(3).view(1, 1, 1, 3, 3).repeat(2, 3, 3, 1, 1)
        camera2ego = torch.eye(4).view(1, 1, 1, 4, 4).repeat(2, 3, 3, 1, 1)

        per_view = fusion(raw_tokens, camera_intrinsics=intrinsics, camera2ego=camera2ego, image_shape=(2, 2))

        self.assertEqual(tuple(per_view.shape), (2, 3 * 3 * 4, 8))
        per_view.view(2, 3, 3, 4, 8).square().mean().backward()
        self.assertIsNotNone(raw_tokens.grad)
        self.assertGreater(float(raw_tokens.grad.abs().sum()), 0.0)

    def test_rejects_non_divisible_token_stream(self):
        fusion = PETRMultiViewFusion(embed_dim=8, tokens_per_frame=4, hidden_dim=16, num_heads=2, depth_num=4)
        raw_tokens = torch.randn(2, 3, 10, 8)

        with self.assertRaises(ValueError):
            fusion(raw_tokens)

    def test_rejects_missing_camera_geometry(self):
        # The fusion is geometry-driven; it must NOT silently fall back to identity geometry when
        # camera_intrinsics/camera2ego are missing (that would make every view identical and corrupt
        # the fused tokens at train/infer without any error).
        fusion = PETRMultiViewFusion(embed_dim=8, tokens_per_frame=4, hidden_dim=16, num_heads=2, depth_num=4)
        raw_tokens = torch.randn(2, 3, 12, 8)
        intrinsics = torch.eye(3).view(1, 1, 1, 3, 3).repeat(2, 3, 3, 1, 1)
        camera2ego = torch.eye(4).view(1, 1, 1, 4, 4).repeat(2, 3, 3, 1, 1)

        with self.assertRaises(ValueError):
            fusion(raw_tokens, camera_intrinsics=None, camera2ego=camera2ego)
        with self.assertRaises(ValueError):
            fusion(raw_tokens, camera_intrinsics=intrinsics, camera2ego=None)

    def test_frustum_depth_num_sets_position_mlp_input_dim(self):
        fusion = PETRMultiViewFusion(embed_dim=8, tokens_per_frame=4, hidden_dim=16, num_heads=2, depth_num=5)

        self.assertEqual(fusion.depth_num, 5)
        self.assertEqual(fusion.position_mlp[0].in_features, 15)

    def test_frustum_features_change_with_camera_translation(self):
        fusion = PETRMultiViewFusion(
            embed_dim=8,
            tokens_per_frame=4,
            hidden_dim=16,
            num_heads=2,
            depth_num=4,
        )
        intrinsics = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 1, 1, 1)
        camera2ego = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 2, 1, 1, 1)
        camera2ego[:, 1, :, 0, 3] = 5.0

        features = fusion._build_frustum_features(
            batch_size=1,
            num_views=2,
            num_steps=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            camera_intrinsics=intrinsics,
            camera2ego=camera2ego,
            image_shape=(2, 2),
        )

        self.assertEqual(tuple(features.shape), (1, 2, 1, 4, 12))
        self.assertGreater(float((features[:, 0] - features[:, 1]).abs().sum()), 0.0)

    def test_frustum_features_match_intrinsics_depth_and_camera2ego_geometry(self):
        fusion = PETRMultiViewFusion(
            embed_dim=8,
            tokens_per_frame=1,
            hidden_dim=16,
            num_heads=2,
            depth_num=2,
            depth_start=2.0,
            position_range=(-20.0, -20.0, -20.0, 20.0, 20.0, 20.0),
        )
        intrinsics = torch.tensor(
            [[[[[2.0, 0.0, 0.5], [0.0, 4.0, 0.25], [0.0, 0.0, 1.0]]]]],
            dtype=torch.float32,
        )
        camera2ego = torch.eye(4).view(1, 1, 1, 4, 4)
        camera2ego[..., 0, 3] = 1.0
        camera2ego[..., 1, 3] = -2.0
        camera2ego[..., 2, 3] = 3.0

        features = fusion._build_frustum_features(
            batch_size=1,
            num_views=1,
            num_steps=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
            camera_intrinsics=intrinsics,
            camera2ego=camera2ego,
            image_shape=(2, 2),
        )

        normalized = features.reshape(1, 1, 1, 1, 2, 3).sigmoid()
        range_min = fusion.position_range[:3].view(1, 1, 1, 1, 1, 3)
        range_max = fusion.position_range[3:].view(1, 1, 1, 1, 1, 3)
        ego_xyz = normalized * (range_max - range_min) + range_min
        expected = torch.tensor([[[[[[1.5, -1.625, 5.0], [3.75, 0.0625, 14.0]]]]]], dtype=torch.float32)
        torch.testing.assert_close(ego_xyz, expected, rtol=1e-5, atol=1e-5)

    def test_frustum_features_align_raw_camera_metadata_to_token_steps(self):
        fusion = PETRMultiViewFusion(
            embed_dim=8,
            tokens_per_frame=4,
            hidden_dim=16,
            num_heads=2,
            depth_num=2,
        )
        intrinsics = torch.eye(3).view(1, 1, 1, 3, 3).repeat(1, 2, 10, 1, 1)
        camera2ego = torch.eye(4).view(1, 1, 1, 4, 4).repeat(1, 2, 10, 1, 1)
        camera2ego[:, :, :, 0, 3] = torch.arange(10, dtype=torch.float32).view(1, 1, 10)
        aligned_indices = torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)

        raw_features = fusion._build_frustum_features(
            batch_size=1,
            num_views=2,
            num_steps=5,
            device=torch.device("cpu"),
            dtype=torch.float32,
            camera_intrinsics=intrinsics,
            camera2ego=camera2ego,
            image_shape=(2, 2),
        )
        aligned_features = fusion._build_frustum_features(
            batch_size=1,
            num_views=2,
            num_steps=5,
            device=torch.device("cpu"),
            dtype=torch.float32,
            camera_intrinsics=intrinsics.index_select(2, aligned_indices),
            camera2ego=camera2ego.index_select(2, aligned_indices),
            image_shape=(2, 2),
        )

        torch.testing.assert_close(raw_features, aligned_features)

    def test_frustum_features_accept_low_precision_inputs_on_cpu(self):
        fusion = PETRMultiViewFusion(
            embed_dim=8,
            tokens_per_frame=4,
            hidden_dim=16,
            num_heads=2,
            depth_num=4,
        )
        intrinsics = torch.eye(3, dtype=torch.float16).view(1, 1, 1, 3, 3).repeat(1, 2, 1, 1, 1)
        camera2ego = torch.eye(4, dtype=torch.float16).view(1, 1, 1, 4, 4).repeat(1, 2, 1, 1, 1)

        features = fusion._build_frustum_features(
            batch_size=1,
            num_views=2,
            num_steps=1,
            device=torch.device("cpu"),
            dtype=torch.float16,
            camera_intrinsics=intrinsics,
            camera2ego=camera2ego,
            image_shape=(2, 2),
        )

        self.assertEqual(features.dtype, torch.float16)
        self.assertEqual(tuple(features.shape), (1, 2, 1, 4, 12))

    def test_position_embedding_accepts_low_precision_geometry_inputs_on_cpu(self):
        fusion = PETRMultiViewFusion(
            embed_dim=8,
            tokens_per_frame=4,
            hidden_dim=16,
            num_heads=2,
            depth_num=4,
        )
        intrinsics = torch.eye(3, dtype=torch.float16).view(1, 1, 1, 3, 3).repeat(1, 2, 1, 1, 1)
        camera2ego = torch.eye(4, dtype=torch.float16).view(1, 1, 1, 4, 4).repeat(1, 2, 1, 1, 1)

        position = fusion._build_position_embedding(
            batch_size=1,
            num_views=2,
            num_steps=1,
            device=torch.device("cpu"),
            dtype=torch.float16,
            camera_intrinsics=intrinsics,
            camera2ego=camera2ego,
            image_shape=(2, 2),
        )

        self.assertEqual(position.dtype, torch.float16)
        self.assertEqual(tuple(position.shape), (1, 2, 1, 4, 8))

    def test_rejects_invalid_frustum_configuration(self):
        with self.assertRaises(ValueError):
            PETRMultiViewFusion(embed_dim=8, tokens_per_frame=4, hidden_dim=16, num_heads=2, depth_num=0)

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(embed_dim=8, tokens_per_frame=4, hidden_dim=16, num_heads=2, depth_num=True)

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(embed_dim=8, tokens_per_frame=4, hidden_dim=16, num_heads=2, depth_num=1.9)

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(
                embed_dim=8,
                tokens_per_frame=4,
                hidden_dim=16,
                num_heads=2,
                position_range=(-1.0, -1.0, -1.0, 1.0, 1.0),
            )

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(
                embed_dim=8,
                tokens_per_frame=4,
                hidden_dim=16,
                num_heads=2,
                position_range=(-1.0, -1.0, "bad", 1.0, 1.0, 1.0),
            )

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(
                embed_dim=8,
                tokens_per_frame=4,
                hidden_dim=16,
                num_heads=2,
                position_range=(-1.0, -1.0, float("nan"), 1.0, 1.0, 1.0),
            )

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(
                embed_dim=8,
                tokens_per_frame=4,
                hidden_dim=16,
                num_heads=2,
                position_range=(0.0, -1.0, -1.0, 0.0, 1.0, 1.0),
            )

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(
                embed_dim=8,
                tokens_per_frame=4,
                hidden_dim=16,
                num_heads=2,
                depth_start=float("nan"),
            )

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(
                embed_dim=8,
                tokens_per_frame=4,
                hidden_dim=16,
                num_heads=2,
                depth_start=0.0,
            )

        with self.assertRaises(ValueError):
            PETRMultiViewFusion(
                embed_dim=8,
                tokens_per_frame=4,
                hidden_dim=16,
                num_heads=2,
                depth_start=65.0,
            )

    def test_token_grid_matches_rectangular_encoder_layout(self):
        fusion = PETRMultiViewFusion(embed_dim=8, tokens_per_frame=512, hidden_dim=16, num_heads=2, depth_num=4)

        grid = fusion._token_grid(device=torch.device("cpu"), dtype=torch.float32, image_shape=(256, 512))

        self.assertEqual(tuple(grid.shape), (512, 3))
        self.assertEqual(grid[:, 0].unique().numel(), 32)
        self.assertEqual(grid[:, 1].unique().numel(), 16)
        torch.testing.assert_close(grid[0], torch.tensor([8.0, 8.0, 1.0]))
        torch.testing.assert_close(grid[-1], torch.tensor([504.0, 248.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
