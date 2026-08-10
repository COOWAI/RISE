# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Prefix-conditioned diffusion planner variant.

Keeps the base DiffusionPlanner implementation unchanged and only alters the
training conditioning path: during training it randomly truncates future z_ar
tokens so the denoiser learns to improve as predictor rollout grows.
"""

from typing import Dict, Optional

import torch

from app.vjepa_cowa_world_model.training.prefix_schedule import (
    PrefixSample,
    resolve_prefix_distribution,
    sample_prefix,
    validate_sampled_h0_context,
)

from .diffusion_planner import DiffusionPlanner


class PrefixConditionedDiffusionPlanner(DiffusionPlanner):
    """Diffusion planner variant with random rollout-prefix conditioning."""

    def __init__(
        self,
        *args,
        train_min_prefix_frames: int = 1,
        train_full_prefix_prob: float = 0.25,
        train_max_non_full_prefix_frames: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_min_prefix_frames = int(train_min_prefix_frames)
        self.train_full_prefix_prob = float(train_full_prefix_prob)
        self.train_max_non_full_prefix_frames = (
            None if train_max_non_full_prefix_frames is None else int(train_max_non_full_prefix_frames)
        )
        self.last_train_prefix_sample: Optional[PrefixSample] = None
        if (
            self.train_min_prefix_frames == 0
            and self.train_full_prefix_prob < 1.0
            and not (self.use_z_context or self.use_observed_tokens or self.use_action_history)
        ):
            raise ValueError(
                "planner internal h=0 prefix conditioning requires z_context, observed tokens, or action_history"
            )

    def _maybe_apply_training_prefix_conditioning(self, z_ar: torch.Tensor) -> torch.Tensor:
        """Randomly truncate future predictor tokens during training."""
        if z_ar.size(1) % self.tokens_per_frame != 0:
            raise ValueError(
                f"z_ar token length {z_ar.size(1)} is not divisible by tokens_per_frame={self.tokens_per_frame}"
            )

        total_future_frames = z_ar.size(1) // self.tokens_per_frame
        distribution = resolve_prefix_distribution(
            enabled=True,
            horizon_steps=total_future_frames,
            full_prefix_prob=self.train_full_prefix_prob,
            min_prefix_steps=self.train_min_prefix_frames,
            max_non_full_prefix_steps=self.train_max_non_full_prefix_frames,
        )
        self.last_train_prefix_sample = sample_prefix(distribution, device=z_ar.device)
        prefix_tokens = self.last_train_prefix_sample.prefix_steps * self.tokens_per_frame
        return z_ar[:, :prefix_tokens]

    def forward(
        self,
        z_ar: torch.Tensor,
        status_feature: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
        z_observed: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
        gt_trajectory: Optional[torch.Tensor] = None,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if gt_trajectory is not None and self.training:
            z_ar = self._maybe_apply_training_prefix_conditioning(z_ar)
            validate_sampled_h0_context(
                prefix_steps=self.last_train_prefix_sample.prefix_steps,
                use_z_context=self.use_z_context,
                z_context=z_context,
                use_observed_tokens=self.use_observed_tokens,
                z_observed=z_observed,
                use_action_history=self.use_action_history,
                action_history=action_history,
            )

        return super().forward(
            z_ar,
            status_feature,
            z_context=z_context,
            z_observed=z_observed,
            action_history=action_history,
            gt_trajectory=gt_trajectory,
            anchor_state=anchor_state,
            inference_noise=inference_noise,
        )
