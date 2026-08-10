# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Seeded diffusion planners with structured K-trajectory initialization.

This module keeps the base DiffusionPlanner implementation untouched and
adds an opt-in subclass that supports non-random initial states (e.g.
kinematic priors) for inference-time x_T/x_t construction.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.prefix_schedule import (
    PrefixSample,
    resolve_prefix_distribution,
    sample_prefix,
    validate_sampled_h0_context,
)

from ..diffusion_utils import dpm_solver_pytorch as dpm
from ..diffusion_utils.sampling import dpm_sampler
from .diffusion_planner import DiffusionPlanner


class SeededDiffusionPlanner(DiffusionPlanner):
    """DiffusionPlanner with configurable structured trajectory seeds.

    Supported strategies:
    - ``gaussian``: legacy behavior (pure Gaussian future seeds)
    - ``kinematic``: constant-velocity prior + optional mode spread + noise
    """

    def __init__(
        self,
        *args,
        init_traj_strategy: str = "gaussian",
        init_traj_noise_scale: float = 1.0,
        init_traj_yaw_span_deg: float = 30.0,
        init_traj_speed_scale_span: float = 0.2,
        dt: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.init_traj_strategy = str(init_traj_strategy).lower()
        self.init_traj_noise_scale = float(init_traj_noise_scale)
        self.init_traj_yaw_span_deg = float(init_traj_yaw_span_deg)
        self.init_traj_speed_scale_span = float(init_traj_speed_scale_span)
        self.dt = float(dt)

        if self.init_traj_strategy not in {"gaussian", "kinematic"}:
            raise ValueError(
                f"Unsupported init_traj_strategy={self.init_traj_strategy!r}; "
                "expected one of {'gaussian', 'kinematic'}"
            )
        if self.init_traj_noise_scale < 0:
            raise ValueError(f"init_traj_noise_scale must be >= 0, got {self.init_traj_noise_scale}")
        if self.init_traj_speed_scale_span < 0:
            raise ValueError(f"init_traj_speed_scale_span must be >= 0, got {self.init_traj_speed_scale_span}")

    def _build_kinematic_future_prior(self, anchor: torch.Tensor, K: int) -> torch.Tensor:
        """Build deterministic future priors from anchor kinematics."""
        B = anchor.shape[0]
        T = self.num_poses
        td = self.traj_dim

        if td < 6:
            raise ValueError(f"kinematic init requires traj_dim>=6, got traj_dim={td}")

        vx0 = anchor[:, 0, 2]
        vy0 = anchor[:, 0, 3]
        cos0 = anchor[:, 0, 4]
        sin0 = anchor[:, 0, 5]

        speed = torch.sqrt(vx0.square() + vy0.square())
        yaw_from_vel = torch.atan2(vy0, vx0)
        yaw_from_heading = torch.atan2(sin0, cos0)
        yaw_base = torch.where(speed > 1e-3, yaw_from_vel, yaw_from_heading)

        if K > 1:
            yaw_span = math.radians(self.init_traj_yaw_span_deg)
            yaw_offsets = torch.linspace(-yaw_span, yaw_span, K, device=anchor.device, dtype=anchor.dtype)
            speed_span = self.init_traj_speed_scale_span
            speed_scales = torch.linspace(
                1.0 - speed_span,
                1.0 + speed_span,
                K,
                device=anchor.device,
                dtype=anchor.dtype,
            ).clamp_min(0.0)
        else:
            yaw_offsets = torch.zeros(1, device=anchor.device, dtype=anchor.dtype)
            speed_scales = torch.ones(1, device=anchor.device, dtype=anchor.dtype)

        yaw = yaw_base.unsqueeze(1) + yaw_offsets.view(1, K)
        speed_k = speed.unsqueeze(1) * speed_scales.view(1, K)
        vx = speed_k * torch.cos(yaw)
        vy = speed_k * torch.sin(yaw)

        steps = torch.arange(1, T + 1, device=anchor.device, dtype=anchor.dtype).view(1, 1, T)
        x = steps * (self.dt * vx.unsqueeze(-1))
        y = steps * (self.dt * vy.unsqueeze(-1))

        vx_t = vx.unsqueeze(-1).expand(-1, -1, T)
        vy_t = vy.unsqueeze(-1).expand(-1, -1, T)
        cos_t = torch.cos(yaw).unsqueeze(-1).expand(-1, -1, T)
        sin_t = torch.sin(yaw).unsqueeze(-1).expand(-1, -1, T)

        prior = torch.zeros(B, K, T, td, device=anchor.device, dtype=anchor.dtype)
        prior[..., 0] = x
        prior[..., 1] = y
        prior[..., 2] = vx_t
        prior[..., 3] = vy_t
        prior[..., 4] = cos_t
        prior[..., 5] = sin_t
        return prior

    def _sample_future_seeds(
        self,
        B: int,
        K: int,
        device: torch.device,
        anchor_state: Optional[torch.Tensor],
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample future seeds with optional structured kinematic prior."""
        anchor = self._get_anchor(anchor_state, B, device)
        if inference_noise is None:
            noise = torch.randn(
                B,
                K,
                self.num_poses,
                self.traj_dim,
                device=device,
                dtype=anchor.dtype,
            )
        else:
            noise = self._resolve_inference_noise(inference_noise, B=B, K=K, device=device)
            if noise.dtype != anchor.dtype:
                raise ValueError(f"inference_noise dtype must match anchor dtype: {noise.dtype} != {anchor.dtype}")
        if self.init_traj_strategy == "kinematic":
            prior = self._build_kinematic_future_prior(anchor, K)
            future = prior + noise * self.init_traj_noise_scale
        else:
            future = noise * self.init_traj_noise_scale
        return anchor, future

    def _build_initial_xt(
        self,
        B: int,
        K: int,
        device: torch.device,
        anchor_state: Optional[torch.Tensor],
        batch_expansion: bool,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build initial denoiser state x_T/x_t and optional anchor info."""
        anchor, future = self._sample_future_seeds(
            B,
            K,
            device,
            anchor_state,
            inference_noise=inference_noise,
        )

        if batch_expansion:
            future_bk = future.reshape(B * K, self.num_poses, self.traj_dim)
            if self.use_anchor_frame:
                anchor_bk = anchor.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, 1, self.traj_dim)
                x_init = torch.cat([anchor_bk, future_bk], dim=-2).reshape(B * K, -1)
                return x_init, anchor_bk
            return future_bk.reshape(B * K, -1), None

        if self.use_anchor_frame:
            anchor_k = anchor.unsqueeze(1).expand(-1, K, -1, -1)
            x_init = torch.cat([anchor_k, future], dim=-2).reshape(B, -1)
            return x_init, anchor_k
        return future.reshape(B, -1), None

    def init_interleaved_inference_state(
        self,
        status_feature: torch.Tensor,
        total_condition_updates: int,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Initialize interleaved inference state with structured seeds."""
        if total_condition_updates <= 0:
            raise ValueError(f"total_condition_updates must be positive, got {total_condition_updates}")

        B = status_feature.shape[0]
        device = status_feature.device
        status_emb = self._prepare_status(status_feature)

        if not self._uses_batch_expansion:
            K = self.num_modes
            x_t, anchor_info = self._build_initial_xt(
                B=B,
                K=K,
                device=device,
                anchor_state=anchor_state,
                batch_expansion=False,
                inference_noise=inference_noise,
            )
            status_k = status_emb
        else:
            K = self._batch_K
            x_t, anchor_info = self._build_initial_xt(
                B=B,
                K=K,
                device=device,
                anchor_state=anchor_state,
                batch_expansion=True,
                inference_noise=inference_noise,
            )
            status_k = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)

        noise_schedule = dpm.NoiseScheduleVP(schedule="linear")
        dummy_solver = dpm.DPM_Solver(lambda x, t: (None, x), noise_schedule, algorithm_type="dpmsolver++")
        timesteps = dummy_solver.get_time_steps(
            skip_type="logSNR",
            t_T=noise_schedule.T,
            t_0=1.0 / noise_schedule.total_N,
            N=self.inference_steps,
            device=device,
        )

        return {
            "x_t": x_t,
            "status_k": status_k,
            "noise_schedule": noise_schedule,
            "timesteps": timesteps,
            "batch_size": B,
            "total_condition_updates": total_condition_updates,
            "completed_condition_updates": 0,
            "completed_sampling_steps": 0,
            "anchor_info": anchor_info,
        }

    def _inference_forward(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Inference with structured seed initialization."""
        B = context_tokens.shape[0]
        device = context_tokens.device

        if self.num_modes > 1:
            if self.independent_modes:
                return self._inference_forward_independent(
                    context_tokens, status_emb, anchor_state, inference_noise=inference_noise
                )
            return self._inference_forward_multimodal(
                context_tokens, status_emb, anchor_state, inference_noise=inference_noise
            )

        K = self.num_samples
        context_k = context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)
        status_k = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)

        dpm_solver_params = {}
        x_T, anchor_info = self._build_initial_xt(
            B=B,
            K=K,
            device=device,
            anchor_state=anchor_state,
            batch_expansion=True,
            inference_noise=inference_noise,
        )
        if self.use_anchor_frame and anchor_info is not None:
            BK = B * K
            TF = self.total_frames
            td = self.traj_dim
            _anchor_exp = anchor_info

            def correcting_xt_fn(xt, t_val, step):
                xt_3d = xt.view(BK, TF, td)
                xt_3d[:, :1, :] = _anchor_exp
                return xt_3d.reshape(BK, -1)

            dpm_solver_params["correcting_xt_fn"] = correcting_xt_fn

        _, x_0 = dpm_sampler(
            model=self.dit,
            x_T=x_T,
            other_model_params={"cross_c": context_k, "status_emb": status_k},
            diffusion_steps=self.inference_steps,
            dpm_solver_params=dpm_solver_params,
        )
        x_0 = x_0.reshape(B, K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:, :]
        traj_3d = self._convert_nd_to_3d(x_0)
        confidences = traj_3d.new_full((B, K), 1.0 / K)
        return {"trajectories": traj_3d, "confidences": confidences}

    def _inference_forward_independent(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Independent-mode inference with structured seeds."""
        B = context_tokens.shape[0]
        K = self.num_modes
        device = context_tokens.device
        TF = self.total_frames
        td = self.traj_dim

        ctx_bk = context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)
        status_bk = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)

        dpm_solver_params = {}
        x_T, anchor_info = self._build_initial_xt(
            B=B,
            K=K,
            device=device,
            anchor_state=anchor_state,
            batch_expansion=True,
            inference_noise=inference_noise,
        )
        if self.use_anchor_frame and anchor_info is not None:
            _anchor_bk = anchor_info

            def correcting_xt_fn(xt, t_val, step):
                xt_3d = xt.view(B * K, TF, td)
                xt_3d[:, :1, :] = _anchor_bk
                return xt_3d.reshape(B * K, -1)

            dpm_solver_params["correcting_xt_fn"] = correcting_xt_fn

        _, x_0 = dpm_sampler(
            model=self.dit,
            x_T=x_T,
            other_model_params={"cross_c": ctx_bk, "status_emb": status_bk},
            diffusion_steps=self.inference_steps,
            dpm_solver_params=dpm_solver_params,
        )

        x_0 = x_0.reshape(B, K, TF, td)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:, :]

        traj_3d = self._convert_nd_to_3d(x_0)
        cls_pred = self.confidence_head(x_0, context_tokens)
        confidences = F.softmax(cls_pred, dim=-1)

        return {"trajectories": traj_3d, "confidences": confidences}

    def _inference_forward_multimodal(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Built-in K-mode inference with structured seeds."""
        B = context_tokens.shape[0]
        device = context_tokens.device
        K = self.num_modes

        dpm_solver_params = {}
        x_T, anchor_info = self._build_initial_xt(
            B=B,
            K=K,
            device=device,
            anchor_state=anchor_state,
            batch_expansion=False,
            inference_noise=inference_noise,
        )
        if self.use_anchor_frame and anchor_info is not None:
            TF = self.total_frames
            td = self.traj_dim
            _anchor_k = anchor_info

            def correcting_xt_fn(xt, t_val, step):
                xt_4d = xt.view(B, K, TF, td)
                xt_4d[:, :, :1, :] = _anchor_k
                return xt_4d.reshape(B, -1)

            dpm_solver_params["correcting_xt_fn"] = correcting_xt_fn

        cls_pred, x_0 = dpm_sampler(
            model=self.dit,
            x_T=x_T,
            other_model_params={"cross_c": context_tokens, "status_emb": status_emb},
            diffusion_steps=self.inference_steps,
            dpm_solver_params=dpm_solver_params,
        )

        x_0 = x_0.reshape(B, K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:, :]
        traj_3d = self._convert_nd_to_3d(x_0)

        if cls_pred is not None:
            confidences = F.softmax(cls_pred, dim=-1)
        else:
            confidences = traj_3d.new_full((B, K), 1.0 / K)

        return {"trajectories": traj_3d, "confidences": confidences}


class PrefixConditionedSeededDiffusionPlanner(SeededDiffusionPlanner):
    """Seeded planner variant with random rollout-prefix conditioning."""

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
