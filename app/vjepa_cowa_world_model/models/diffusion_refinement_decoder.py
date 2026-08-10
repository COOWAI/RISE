"""Diffusion-style iterative refinement decoder for staged training."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .diffusion_planner import DiffusionPlanner


class DiffusionRefinementDecoder(nn.Module):
    """Refine external proposal trajectories with a DiT-based update core.

    The public interface mirrors :class:`RefinementDecoder`: callers pass an
    external K-mode proposal and receive ``traj_rounds`` plus an optional single
    final trajectory.  Unlike ``RefinementDecoder``, this module does not build
    a ``MultiModalTemporalPlanner``; each refinement round feeds the current
    proposal through a ``DiffusionPlanner`` DiT core as a differentiable residual
    denoising/update step.
    """

    SOURCE_IMAGE = 0
    SOURCE_FUTURE = 1
    SOURCE_TRAJECTORY = 2
    SOURCE_PROPOSAL_FEATURE = 3

    def __init__(
        self,
        encoder_dim: int = 1024,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        tokens_per_frame: int = 256,
        num_poses: int = 7,
        status_dim: int = 7,
        num_modes: int = 6,
        traj_dim: int = 6,
        sde_beta_min: float = 0.1,
        sde_beta_max: float = 20.0,
        num_samples: Optional[int] = None,
        inference_steps: int = 2,
        trajectory_token_mode: str = "single_token",
        use_anchor_frame: bool = False,
        independent_modes: bool = False,
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 1.0,
        vel_loss_weight: float = 0.5,
        yaw_loss_weight: float = 0.5,
        awta_init_temperature: float = 8.0,
        awta_min_temperature: float = 0.1,
        conf_temperature: float = 1.5,
        cls_th: float = 2.0,
        cls_ignore: float = 0.2,
        command_dim: int = 0,
        adaln_version: str = "legacy",
        mode_token_expansion: bool = False,
        max_rounds: int = 4,
        dt: float = 0.2,
        refine_timestep: float = 0.5,
        observed_token_mode: str = "none",
    ):
        super().__init__()
        if traj_dim not in {4, 6}:
            raise ValueError(f"DiffusionRefinementDecoder supports traj_dim 4 or 6, got {traj_dim}")

        self.refine_core = DiffusionPlanner(
            encoder_dim=encoder_dim,
            num_poses=num_poses,
            status_dim=status_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            traj_dim=traj_dim,
            sde_beta_min=sde_beta_min,
            sde_beta_max=sde_beta_max,
            num_samples=num_samples if num_samples is not None else num_modes,
            inference_steps=inference_steps,
            use_z_context=False,
            tokens_per_frame=tokens_per_frame,
            trajectory_token_mode=trajectory_token_mode,
            use_last_frame_only=False,
            use_action_history=False,
            action_history_dim=3,
            num_observed_frames=1,
            observed_token_mode=observed_token_mode,
            num_modes=num_modes,
            independent_modes=independent_modes,
            use_anchor_frame=use_anchor_frame,
            cls_loss_weight=cls_loss_weight,
            reg_loss_weight=reg_loss_weight,
            vel_loss_weight=vel_loss_weight,
            yaw_loss_weight=yaw_loss_weight,
            awta_init_temperature=awta_init_temperature,
            awta_min_temperature=awta_min_temperature,
            conf_temperature=conf_temperature,
            cls_th=cls_th,
            cls_ignore=cls_ignore,
            command_dim=command_dim,
            adaln_version=adaln_version,
            mode_token_expansion=mode_token_expansion,
        )

        self.encoder_dim = encoder_dim
        self.tf_d_model = hidden_dim
        self.num_modes = num_modes
        self.num_poses = num_poses
        self.traj_dim = traj_dim
        self.max_rounds = max_rounds
        self.dt = float(dt)
        self.refine_timestep = float(refine_timestep)

        self.mode_embed = nn.Embedding(num_modes, encoder_dim)
        self.source_embed = nn.Embedding(4, encoder_dim)
        self.round_embed = nn.Embedding(max_rounds, encoder_dim)
        self.traj_condition_proj = nn.Sequential(
            nn.Linear(3, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, encoder_dim),
        )
        self.proposal_feature_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proposal_feature_condition_proj = nn.Linear(hidden_dim, encoder_dim)

    def encode_proposal_features(self, proposal_traj: torch.Tensor) -> torch.Tensor:
        return self.proposal_feature_encoder(proposal_traj)

    @staticmethod
    def _convert_3d_to_nd(traj_3d: torch.Tensor, dt: float, traj_dim: int) -> torch.Tensor:
        x = traj_3d[..., 0]
        y = traj_3d[..., 1]
        yaw = traj_3d[..., 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        if traj_dim == 4:
            return torch.stack([x, y, cos_yaw, sin_yaw], dim=-1)

        vx = torch.zeros_like(x)
        vy = torch.zeros_like(y)
        if x.shape[-1] > 0:
            vx[..., 0] = x[..., 0] / dt
            vy[..., 0] = y[..., 0] / dt
        if x.shape[-1] > 1:
            vx[..., 1:] = (x[..., 1:] - x[..., :-1]) / dt
            vy[..., 1:] = (y[..., 1:] - y[..., :-1]) / dt
        return torch.stack([x, y, vx, vy, cos_yaw, sin_yaw], dim=-1)

    def _check_mode_count(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.shape[1] != self.num_modes:
            raise ValueError(f"{name} has K={tensor.shape[1]}, expected num_modes={self.num_modes}")

    def _mode_round_tokens(self, tensor: torch.Tensor, round_index: int, source_id: int) -> torch.Tensor:
        self._check_mode_count(tensor, "refinement tensor")
        mode_tokens = self.mode_embed.weight.view(1, self.num_modes, 1, self.encoder_dim)
        source_tokens = self.source_embed.weight[source_id].view(1, 1, 1, self.encoder_dim)
        round_tokens = self.round_embed.weight[round_index].view(1, 1, 1, self.encoder_dim)
        return tensor + mode_tokens + source_tokens + round_tokens

    def _compose_condition_tokens(
        self,
        z_context: torch.Tensor,
        traj_m: torch.Tensor,
        z_fut_m: Optional[torch.Tensor],
        proposal_features: Optional[torch.Tensor],
        round_index: int,
    ) -> torch.Tensor:
        image_tokens = z_context + self.source_embed.weight[self.SOURCE_IMAGE].view(1, 1, self.encoder_dim)
        parts = [image_tokens]

        traj_tokens = self.traj_condition_proj(traj_m)
        traj_tokens = self._mode_round_tokens(traj_tokens, round_index, self.SOURCE_TRAJECTORY)
        parts.append(traj_tokens.flatten(1, 2))

        if z_fut_m is not None:
            future_tokens = self._mode_round_tokens(z_fut_m, round_index, self.SOURCE_FUTURE)
            parts.append(future_tokens.flatten(1, 2))

        if proposal_features is not None:
            feature_tokens = self.proposal_feature_condition_proj(proposal_features)
            feature_tokens = self._mode_round_tokens(feature_tokens, round_index, self.SOURCE_PROPOSAL_FEATURE)
            parts.append(feature_tokens.flatten(1, 2))

        return torch.cat(parts, dim=1)

    def _add_anchor_frame(self, traj_nd: torch.Tensor) -> torch.Tensor:
        if not self.refine_core.use_anchor_frame:
            return traj_nd
        batch_size = traj_nd.shape[0]
        anchor = self.refine_core._get_anchor(None, batch_size, traj_nd.device)
        anchor = anchor.unsqueeze(1).expand(-1, self.num_modes, -1, -1)
        return torch.cat([anchor.to(dtype=traj_nd.dtype), traj_nd], dim=2)

    def _raw_confidences(
        self,
        cls_pred: Optional[torch.Tensor],
        traj_nd: torch.Tensor,
        context_tokens: torch.Tensor,
        current_logits: torch.Tensor,
    ) -> torch.Tensor:
        if cls_pred is not None:
            return cls_pred
        if self.refine_core.independent_modes and self.refine_core.confidence_head is not None:
            return self.refine_core.confidence_head(traj_nd, context_tokens)
        return current_logits

    def _forward_mode_update(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        traj_m: torch.Tensor,
        logits_m: torch.Tensor,
        z_fut_m: Optional[torch.Tensor],
        proposal_features: Optional[torch.Tensor],
        round_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_mode_count(traj_m, "traj_m")
        condition_tokens = self._compose_condition_tokens(z_context, traj_m, z_fut_m, proposal_features, round_index)
        context_tokens = self.refine_core._prepare_context(condition_tokens)
        status_emb = self.refine_core._prepare_status(status_feature)

        batch_size = traj_m.shape[0]
        current_nd = self._convert_3d_to_nd(traj_m, self.dt, self.traj_dim)
        model_nd = self._add_anchor_frame(current_nd)
        t = torch.full(
            (batch_size,),
            max(min(self.refine_timestep, 1.0), 1e-3),
            device=traj_m.device,
            dtype=traj_m.dtype,
        )

        if self.refine_core.independent_modes and self.num_modes > 1:
            context_bk = (
                context_tokens.unsqueeze(1)
                .expand(-1, self.num_modes, -1, -1)
                .reshape(batch_size * self.num_modes, -1, self.tf_d_model)
            )
            status_bk = (
                status_emb.unsqueeze(1)
                .expand(-1, self.num_modes, -1)
                .reshape(batch_size * self.num_modes, self.tf_d_model)
            )
            x_t = model_nd.reshape(batch_size * self.num_modes, -1)
            t_bk = t.unsqueeze(1).expand(-1, self.num_modes).reshape(-1)
            cls_pred, delta = self.refine_core.dit(x_t, t_bk, context_bk, status_bk)
            delta = delta.reshape(batch_size, self.num_modes, self.refine_core.total_frames, self.traj_dim)
            cls_pred = None
        else:
            x_t = model_nd.reshape(batch_size, -1)
            cls_pred, delta = self.refine_core.dit(x_t, t, context_tokens, status_emb)
            delta = delta.reshape(batch_size, self.num_modes, self.refine_core.total_frames, self.traj_dim)

        updated_nd = model_nd + delta
        if self.refine_core.use_anchor_frame:
            updated_nd = updated_nd[:, :, 1:, :]
        traj_3d = self.refine_core._convert_nd_to_3d(updated_nd)
        confidences = self._raw_confidences(cls_pred, updated_nd, context_tokens, logits_m)
        features = self.encode_proposal_features(traj_3d)
        return traj_3d, confidences, features

    def _decode_final(self, traj_m: torch.Tensor, logits_m: torch.Tensor) -> torch.Tensor:
        weights = logits_m.softmax(dim=1).view(logits_m.shape[0], self.num_modes, 1, 1)
        return (traj_m * weights).sum(dim=1)

    def forward_iterative(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        proposal_traj: torch.Tensor,
        proposal_logits: torch.Tensor,
        proposal_features: Optional[torch.Tensor],
        predictor_rollout_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        num_rounds: int,
        grad_checkpoint: bool = False,
        detach_future: bool = True,
        use_initial_proposal_features: bool = True,
        return_single_final: bool = True,
    ) -> tuple[list[dict[str, torch.Tensor]], Optional[torch.Tensor]]:
        del grad_checkpoint
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1")

        current_traj = proposal_traj
        current_logits = proposal_logits
        current_features = proposal_features
        if current_features is None and use_initial_proposal_features:
            current_features = self.encode_proposal_features(proposal_traj)

        current_fut = None
        traj_rounds = [{"trajectories": current_traj, "confidences": current_logits}]

        for round_index in range(1, num_rounds):
            current_fut = predictor_rollout_fn(current_traj) if predictor_rollout_fn is not None else None
            if current_fut is not None and detach_future:
                current_fut = current_fut.detach()
            current_traj, current_logits, current_features = self._forward_mode_update(
                z_context,
                status_feature,
                current_traj,
                current_logits,
                current_fut,
                current_features,
                round_index=min(round_index, self.max_rounds - 1),
            )
            traj_rounds.append({"trajectories": current_traj, "confidences": current_logits})

        if not return_single_final:
            return traj_rounds, None
        return traj_rounds, self._decode_final(current_traj, current_logits)
