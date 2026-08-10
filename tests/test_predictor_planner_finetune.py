# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Tests for the predictor planner-finetune mode.

This mode unfreezes the predictor during planner training and trains it *only*
from the planner gradient (through z_ar), without applying jepa_loss. The two
config knobs are:
  - train.predictor_planner_finetune (bool, default False)
  - optimization.predictor_lr_scale  (float, default 1.0)

These tests cover the config plumbing and the optimizer lr_scale wiring. The
loss-gating / unfreeze / grad-clip wiring inside train_navsim_v2.main() reuses
the already-tested planner_only_mode path, gated by skip_predictor_supervision.
"""

import unittest

import torch.nn as nn

from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.optimizer import init_opt_no_resample_world_model


class TestPredictorPlannerFinetuneConfig(unittest.TestCase):
    def test_defaults_preserve_existing_behavior(self):
        cfg = parse_training_config({})
        self.assertFalse(cfg.train.predictor_planner_finetune)
        self.assertEqual(cfg.optimization.predictor_lr_scale, 1.0)

    def test_flags_round_trip(self):
        cfg = parse_training_config(
            {
                "train": {"predictor_planner_finetune": True},
                "optimization": {"predictor_lr_scale": 0.1},
            }
        )
        self.assertTrue(cfg.train.predictor_planner_finetune)
        self.assertEqual(cfg.optimization.predictor_lr_scale, 0.1)


class TestPredictorLrScale(unittest.TestCase):
    def _first_step_lr(self, scale):
        predictor = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        optimizer, _scaler, scheduler, _wd = init_opt_no_resample_world_model(
            encoder=None,
            predictor=predictor,
            seg_head=None,
            seg_neck=None,
            iterations_per_epoch=10,
            start_lr=0.0,
            ref_lr=1.0,
            warmup=0,
            anneal=1,
            num_epochs=1,
            predictor_lr_scale=scale,
        )
        scheduler.step()
        scales = [g.get("lr_scale") for g in optimizer.param_groups]
        return optimizer.param_groups[0]["lr"], scales

    def test_default_scale_is_noop(self):
        lr, scales = self._first_step_lr(1.0)
        self.assertEqual(scales, [1.0, 1.0])
        # ref_lr=1.0 -> first cosine step lr; lr_scale=1.0 leaves it unchanged.
        self.assertGreater(lr, 0.0)

    def test_predictor_lr_scale_is_applied(self):
        base_lr, base_scales = self._first_step_lr(1.0)
        scaled_lr, scaled_scales = self._first_step_lr(0.1)
        self.assertEqual(base_scales, [1.0, 1.0])
        self.assertEqual(scaled_scales, [0.1, 0.1])
        # Both schedules are identical apart from the predictor lr_scale factor,
        # so the scaled lr must be exactly 0.1x the unscaled lr.
        self.assertAlmostEqual(scaled_lr / base_lr, 0.1, places=6)


if __name__ == "__main__":
    unittest.main()
