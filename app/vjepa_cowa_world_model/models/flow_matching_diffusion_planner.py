# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Flow-matching trajectory planner variants.

This module keeps the existing VP diffusion planner intact and adds an
opt-in planner that reuses the same TrajectoryDiT/context/status machinery
while replacing the denoising objective and DPM-Solver++ sampler with flow
matching objectives and ODE integration.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.prefix_schedule import (
    PrefixSample,
    resolve_prefix_distribution,
    sample_prefix,
    validate_sampled_h0_context,
)

from .seeded_diffusion_planner import SeededDiffusionPlanner


class FlowMatchingDiffusionPlanner(SeededDiffusionPlanner):
    """DiffusionPlanner-compatible trajectory planner trained by flow matching.

    Supported variants:
    - ``rectified``: straight line path ``x_t = (1-t) * noise + t * data``
      with target velocity ``data - noise``.
    - ``scheduler``: Wan/diffusers-style sigma path
      ``x_sigma = sigma * noise + (1-sigma) * data`` with target velocity
      ``noise - data`` and Euler/Heun integration over decreasing sigmas.
    """

    _RECTIFIED_ALIASES = {"rectified", "rectified_flow", "ot", "ot_cfm"}
    _SCHEDULER_ALIASES = {"scheduler", "flowmatch_euler", "wan", "wan_style"}

    def __init__(
        self,
        *args,
        flow_matching_variant: str = "rectified",
        flow_shift: float = 1.0,
        flow_sampler: str = "euler",
        flow_timestep_sampling: str = "logit_normal",
        train_prefix_conditioning: bool = False,
        train_min_prefix_frames: int = 1,
        train_full_prefix_prob: float = 0.25,
        train_max_non_full_prefix_frames: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flow_matching_variant = self._normalize_variant(flow_matching_variant)
        self.flow_shift = float(flow_shift)
        self.flow_sampler = str(flow_sampler).lower()
        self.flow_timestep_sampling = str(flow_timestep_sampling).lower()
        self.train_prefix_conditioning = bool(train_prefix_conditioning)
        self.train_min_prefix_frames = int(train_min_prefix_frames)
        self.train_full_prefix_prob = float(train_full_prefix_prob)
        self.train_max_non_full_prefix_frames = (
            None if train_max_non_full_prefix_frames is None else int(train_max_non_full_prefix_frames)
        )
        self.last_train_prefix_sample: Optional[PrefixSample] = None

        if self.flow_shift <= 0:
            raise ValueError(f"flow_shift must be positive, got {self.flow_shift}")
        if self.flow_sampler not in {"euler", "heun"}:
            raise ValueError(f"Unsupported flow_sampler={self.flow_sampler!r}; expected 'euler' or 'heun'")
        if self.flow_timestep_sampling not in {"uniform", "logit_normal"}:
            raise ValueError(
                f"Unsupported flow_timestep_sampling={self.flow_timestep_sampling!r}; "
                "expected 'uniform' or 'logit_normal'"
            )
        if (
            self.train_prefix_conditioning
            and self.train_min_prefix_frames == 0
            and self.train_full_prefix_prob < 1.0
            and not (self.use_z_context or self.use_observed_tokens or self.use_action_history)
        ):
            raise ValueError(
                "planner internal h=0 prefix conditioning requires z_context, observed tokens, or action_history"
            )

    @classmethod
    def _normalize_variant(cls, variant: str) -> str:
        value = str(variant).lower()
        if value in cls._RECTIFIED_ALIASES:
            return "rectified"
        if value in cls._SCHEDULER_ALIASES:
            return "scheduler"
        raise ValueError(
            f"Unsupported flow_matching_variant={variant!r}; expected one of "
            "{'rectified', 'scheduler', 'flowmatch_euler', 'wan'}"
        )

    def _maybe_apply_training_prefix_conditioning(self, z_ar: torch.Tensor) -> torch.Tensor:
        if not self.train_prefix_conditioning or not self.training:
            return z_ar
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
            if self.train_prefix_conditioning:
                validate_sampled_h0_context(
                    prefix_steps=self.last_train_prefix_sample.prefix_steps,
                    use_z_context=self.use_z_context,
                    z_context=z_context,
                    use_observed_tokens=self.use_observed_tokens,
                    z_observed=z_observed,
                    use_action_history=self.use_action_history,
                    action_history=action_history,
                )

        context_tokens = self._prepare_context(z_ar, z_context, z_observed, action_history)
        status_emb = self._prepare_status(status_feature)

        if gt_trajectory is not None and self.training:
            if inference_noise is not None:
                raise ValueError("inference_noise is only valid for planner inference")
            return self._training_forward(context_tokens, status_emb, gt_trajectory, anchor_state)
        return self._inference_forward(context_tokens, status_emb, anchor_state, inference_noise=inference_noise)

    def init_interleaved_inference_state(self, *args, **kwargs):
        raise NotImplementedError("FlowMatchingDiffusionPlanner does not support interleaved predictor sampling yet")

    def advance_interleaved_inference(self, *args, **kwargs):
        raise NotImplementedError("FlowMatchingDiffusionPlanner does not support interleaved predictor sampling yet")

    def finalize_interleaved_inference(self, *args, **kwargs):
        raise NotImplementedError("FlowMatchingDiffusionPlanner does not support interleaved predictor sampling yet")

    def _sample_flow_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.flow_timestep_sampling == "logit_normal":
            t = torch.sigmoid(torch.randn(batch_size, device=device))
        else:
            t = torch.rand(batch_size, device=device)
        if self.flow_matching_variant == "scheduler":
            t = self._shift_sigma(t)
        return t.clamp(1e-5, 1.0 - 1e-5)

    def _shift_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        if self.flow_shift == 1.0:
            return sigma
        return self.flow_shift * sigma / (1.0 + (self.flow_shift - 1.0) * sigma)

    def _flow_training_path(self, clean: torch.Tensor, time_value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build noised state and target velocity for future frames only."""
        noise = torch.randn_like(clean)
        view_shape = [clean.shape[0]] + [1] * (clean.ndim - 1)
        scalar = time_value.reshape(view_shape)
        if self.flow_matching_variant == "rectified":
            x_t = (1.0 - scalar) * noise + scalar * clean
            velocity = clean - noise
        else:
            sigma = time_value.reshape(view_shape)
            x_t = sigma * noise + (1.0 - sigma) * clean
            velocity = noise - clean
        return x_t, velocity

    def _predict_clean_from_velocity(
        self,
        x_t_future: torch.Tensor,
        velocity: torch.Tensor,
        time_value: torch.Tensor,
    ) -> torch.Tensor:
        view_shape = [x_t_future.shape[0]] + [1] * (x_t_future.ndim - 1)
        scalar = time_value.reshape(view_shape)
        if self.flow_matching_variant == "rectified":
            return x_t_future + (1.0 - scalar) * velocity
        sigma = time_value.reshape(view_shape)
        return x_t_future - sigma * velocity

    def _training_forward(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.num_modes > 1:
            if self.independent_modes:
                return self._training_forward_independent(context_tokens, status_emb, gt_trajectory, anchor_state)
            return self._training_forward_multimodal(context_tokens, status_emb, gt_trajectory, anchor_state)
        return self._training_forward_single(context_tokens, status_emb, gt_trajectory, anchor_state)

    def _training_forward_single(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = gt_trajectory.shape[0]
        device = gt_trajectory.device
        time_value = self._sample_flow_time(B, device)
        x_t_future, target_velocity = self._flow_training_path(gt_trajectory, time_value)

        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)
            x_t = torch.cat([anchor, x_t_future], dim=1).reshape(B, -1)
        else:
            x_t = x_t_future.reshape(B, -1)

        _, pred_velocity = self.dit(x_t, time_value, context_tokens, status_emb)
        pred_velocity = pred_velocity.view(B, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            pred_velocity = pred_velocity[:, 1:, :]
        x_pred = self._predict_clean_from_velocity(x_t_future, pred_velocity, time_value)

        flow_loss = F.mse_loss(pred_velocity, target_velocity)
        endpoint_reg_loss, vel_loss, yaw_loss = self._endpoint_losses(x_pred, gt_trajectory)
        # apply the configured endpoint vel/yaw weights (base diffusion does; flow-matching previously dropped
        # them, making diff_vel_loss_weight / diff_yaw_loss_weight silent no-ops in flow mode).
        total_loss = (
            self.reg_loss_weight * flow_loss + self.vel_loss_weight * vel_loss + self.yaw_loss_weight * yaw_loss
        )
        return {
            "loss": total_loss,
            "reg_loss": flow_loss,
            "conf_loss": torch.tensor(0.0, device=device),
            "cover_loss": torch.tensor(0.0, device=device),
            "vel_loss": vel_loss,
            "yaw_loss": yaw_loss,
            "flow_loss": flow_loss,
            "endpoint_reg_loss": endpoint_reg_loss,
            "winner_traj_3d": self._convert_nd_to_3d(x_pred.detach()),
            "cls_sample_valid_ratio": torch.tensor(0.0, device=device),
        }

    def _training_forward_independent(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = gt_trajectory.shape[0]
        K = self.num_modes
        device = gt_trajectory.device
        gt_rep = gt_trajectory.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, self.num_poses, self.traj_dim)
        time_value = self._sample_flow_time(B, device).unsqueeze(1).expand(-1, K).reshape(B * K)
        x_t_future, target_velocity = self._flow_training_path(gt_rep, time_value)

        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)
            anchor_bk = anchor.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, 1, self.traj_dim)
            x_t = torch.cat([anchor_bk, x_t_future], dim=1).reshape(B * K, -1)
        else:
            x_t = x_t_future.reshape(B * K, -1)

        ctx_bk = context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)
        status_bk = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)
        _, pred_velocity = self.dit(x_t, time_value, ctx_bk, status_bk)
        pred_velocity = pred_velocity.view(B * K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            pred_velocity = pred_velocity[:, 1:, :]
        x_pred = self._predict_clean_from_velocity(x_t_future, pred_velocity, time_value).view(
            B, K, self.num_poses, self.traj_dim
        )
        target_velocity = target_velocity.view(B, K, self.num_poses, self.traj_dim)
        pred_velocity = pred_velocity.view(B, K, self.num_poses, self.traj_dim)
        cls_pred = self.confidence_head(x_pred.detach(), context_tokens)
        return self._build_multimodal_loss_dict(pred_velocity, target_velocity, x_pred, cls_pred, gt_trajectory)

    def _training_forward_multimodal(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = gt_trajectory.shape[0]
        K = self.num_modes
        device = gt_trajectory.device
        gt_rep = gt_trajectory.unsqueeze(1).expand(-1, K, -1, -1)
        time_value = self._sample_flow_time(B, device)
        x_t_future, target_velocity = self._flow_training_path(gt_rep, time_value)

        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, B, device)
            anchor_k = anchor.unsqueeze(1).expand(-1, K, -1, -1)
            x_t = torch.cat([anchor_k, x_t_future], dim=2).reshape(B, -1)
        else:
            x_t = x_t_future.reshape(B, -1)

        cls_pred, pred_velocity = self.dit(x_t, time_value, context_tokens, status_emb)
        pred_velocity = pred_velocity.view(B, K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            pred_velocity = pred_velocity[:, :, 1:, :]
        x_pred = self._predict_clean_from_velocity(x_t_future, pred_velocity, time_value)
        return self._build_multimodal_loss_dict(pred_velocity, target_velocity, x_pred, cls_pred, gt_trajectory)

    def _endpoint_losses(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_xy = pred[..., :2]
        if pred_xy.ndim == 3:
            pred_xy = pred_xy.unsqueeze(1)
        reg_loss = self._xy_regression_loss_per_mode(pred_xy, gt[..., :2]).mean()
        if self._has_velocity:
            vel_loss = F.smooth_l1_loss(pred[..., 2], gt[..., 2], reduction="mean") + F.smooth_l1_loss(
                pred[..., 3], gt[..., 3], reduction="mean"
            )
        else:
            vel_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)
        ys = self._yaw_slice
        cos_sim = F.cosine_similarity(pred[..., ys], gt[..., ys], dim=-1)
        yaw_loss = ((1.0 - cos_sim) / 2.0).mean()
        return reg_loss, vel_loss, yaw_loss

    def _build_multimodal_loss_dict(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        x_pred: torch.Tensor,
        cls_pred: Optional[torch.Tensor],
        gt_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, K, _, _ = x_pred.shape
        device = x_pred.device
        gt_xy = gt_trajectory[:, :, :2]
        pred_xy = x_pred[:, :, :, :2]
        dist_xy = (pred_xy - gt_xy.unsqueeze(1)).norm(dim=-1).mean(dim=-1)
        min_dist, winner_idx = dist_xy.min(dim=1)

        logit_clamp = 50.0
        awta_T = float(torch.clamp(self.awta_temperature, min=self.awta_min_temperature).item())
        awta_w = F.softmax((-dist_xy / awta_T).clamp(-logit_clamp, logit_clamp), dim=1).detach()
        per_mode_flow = F.mse_loss(pred_velocity, target_velocity, reduction="none").mean(dim=(-1, -2))
        flow_loss = (awta_w * per_mode_flow).sum(dim=1).mean()

        gt_xy_k = gt_xy.unsqueeze(1).expand(-1, K, -1, -1)
        endpoint_reg = self._xy_regression_loss_per_mode(pred_xy, gt_xy_k)
        endpoint_reg_loss = (awta_w * endpoint_reg).sum(dim=1).mean()

        if self._has_velocity:
            gt_vx_k = gt_trajectory[:, :, 2].unsqueeze(1).expand(-1, K, -1)
            gt_vy_k = gt_trajectory[:, :, 3].unsqueeze(1).expand(-1, K, -1)
            per_mode_vx = F.smooth_l1_loss(x_pred[..., 2], gt_vx_k, reduction="none").mean(dim=-1)
            per_mode_vy = F.smooth_l1_loss(x_pred[..., 3], gt_vy_k, reduction="none").mean(dim=-1)
            vel_loss = (awta_w * (per_mode_vx + per_mode_vy)).sum(dim=1).mean()
        else:
            vel_loss = torch.zeros((), device=device, dtype=x_pred.dtype)

        ys = self._yaw_slice
        gt_ang_k = gt_trajectory[..., ys].unsqueeze(1).expand(-1, K, -1, -1)
        cos_sim = F.cosine_similarity(x_pred[..., ys], gt_ang_k, dim=-1)
        per_mode_yaw = ((1.0 - cos_sim) / 2.0).mean(dim=-1)
        yaw_loss = (awta_w * per_mode_yaw).sum(dim=1).mean()

        if cls_pred is not None:
            conf_logits_target = (-dist_xy / self.conf_temperature).clamp(-logit_clamp, logit_clamp)
            soft_target = F.softmax(conf_logits_target, dim=1).detach()
            sample_valid = min_dist < self.cls_th
            if bool(sample_valid.any()):
                mode_keep = (dist_xy - min_dist.unsqueeze(1)) > self.cls_ignore
                winner_onehot = F.one_hot(winner_idx, num_classes=K).bool()
                mode_keep = mode_keep | winner_onehot
                log_probs = F.log_softmax(cls_pred, dim=1)
                per_sample_ce = -(soft_target * log_probs * mode_keep.float()).sum(dim=1)
                cls_loss = per_sample_ce[sample_valid].mean()
            else:
                cls_loss = cls_pred.sum() * 0.0
        else:
            sample_valid = torch.zeros(B, dtype=torch.bool, device=device)
            cls_loss = torch.zeros((), device=device, dtype=x_pred.dtype)

        total_loss = (
            self.reg_loss_weight * flow_loss
            + self.cls_loss_weight * cls_loss
            + self.vel_loss_weight * vel_loss
            + self.yaw_loss_weight * yaw_loss
        )
        winner_traj = x_pred[torch.arange(B, device=device), winner_idx]
        return {
            "loss": total_loss,
            "reg_loss": flow_loss,
            "conf_loss": cls_loss,
            "cover_loss": torch.zeros((), device=device, dtype=x_pred.dtype),
            "vel_loss": vel_loss,
            "yaw_loss": yaw_loss,
            "flow_loss": flow_loss,
            "endpoint_reg_loss": endpoint_reg_loss,
            "winner_idx": winner_idx,
            "winner_traj_3d": self._convert_nd_to_3d(winner_traj.detach()),
            "awta_temperature": torch.tensor(awta_T, device=device),
            "cls_sample_valid_ratio": sample_valid.float().mean().detach(),
        }

    def _inference_forward(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
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
        x_t, anchor_info = self._build_initial_xt(
            B,
            K,
            device,
            anchor_state,
            batch_expansion=True,
            inference_noise=inference_noise,
        )
        x_0, _ = self._integrate_flow(x_t, context_k, status_k, anchor_info=anchor_info, batch_size=B)
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
        B = context_tokens.shape[0]
        K = self.num_modes
        device = context_tokens.device
        ctx_bk = context_tokens.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, -1, self.hidden_dim)
        status_bk = status_emb.unsqueeze(1).expand(-1, K, -1).reshape(B * K, self.hidden_dim)
        x_t, anchor_info = self._build_initial_xt(
            B,
            K,
            device,
            anchor_state,
            batch_expansion=True,
            inference_noise=inference_noise,
        )
        x_0, _ = self._integrate_flow(x_t, ctx_bk, status_bk, anchor_info=anchor_info, batch_size=B)
        x_0 = x_0.reshape(B, K, self.total_frames, self.traj_dim)
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
        B = context_tokens.shape[0]
        K = self.num_modes
        device = context_tokens.device
        x_t, anchor_info = self._build_initial_xt(
            B,
            K,
            device,
            anchor_state,
            batch_expansion=False,
            inference_noise=inference_noise,
        )
        x_0, cls_pred = self._integrate_flow(x_t, context_tokens, status_emb, anchor_info=anchor_info, batch_size=B)
        x_0 = x_0.reshape(B, K, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:, :]
        traj_3d = self._convert_nd_to_3d(x_0)
        if cls_pred is not None:
            confidences = F.softmax(cls_pred, dim=-1)
        else:
            confidences = traj_3d.new_full((B, K), 1.0 / K)
        return {"trajectories": traj_3d, "confidences": confidences}

    def _flow_time_grid(self, device: torch.device) -> torch.Tensor:
        steps = max(int(self.inference_steps), 1)
        if self.flow_matching_variant == "rectified":
            return torch.linspace(0.0, 1.0, steps + 1, device=device)
        base = torch.linspace(1.0, 0.0, steps + 1, device=device)
        return self._shift_sigma(base)

    @torch.no_grad()
    def _integrate_flow(
        self,
        x_t: torch.Tensor,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_info: Optional[torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        grid = self._flow_time_grid(x_t.device)
        x = x_t
        cls_pred = None
        for step_idx in range(grid.numel() - 1):
            current = grid[step_idx]
            nxt = grid[step_idx + 1]
            dt = nxt - current
            t = current.expand(x.shape[0])
            cls_pred, velocity = self.dit(x, t, context_tokens, status_emb)
            if self.flow_sampler == "heun" and step_idx < grid.numel() - 2:
                x_euler = x + dt * velocity
                if self.use_anchor_frame and anchor_info is not None:
                    x_euler = self._correct_anchor_xt(x_euler, anchor_info)
                _, velocity_next = self.dit(x_euler, nxt.expand(x.shape[0]), context_tokens, status_emb)
                x = x + dt * 0.5 * (velocity + velocity_next)
            else:
                x = x + dt * velocity
            if self.use_anchor_frame and anchor_info is not None:
                x = self._correct_anchor_xt(x, anchor_info)
        return x, cls_pred
