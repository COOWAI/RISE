"""Independent proposal provider modules for staged planner training."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class BaseProposalProvider(nn.Module):
    """Base interface for frozen proposal providers."""

    def forward(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        history_traj: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


def _trajectory_features_from_pose(trajectories: torch.Tensor, hidden_dim: int) -> torch.Tensor:
    x = trajectories[..., 0]
    y = trajectories[..., 1]
    yaw = trajectories[..., 2]
    base = torch.stack(
        [
            x,
            y,
            torch.sin(yaw),
            torch.cos(yaw),
            torch.sqrt(x.square() + y.square() + 1e-6),
            yaw,
        ],
        dim=-1,
    )
    repeat_factor = (hidden_dim + base.shape[-1] - 1) // base.shape[-1]
    expanded = base.repeat_interleave(repeat_factor, dim=-1)
    return expanded[..., :hidden_dim]


class TransformerProposalProvider(BaseProposalProvider):
    """Frozen transformer proposal provider backed by MultiModalTemporalPlanner."""

    def __init__(
        self,
        encoder_dim: int,
        tf_d_model: int,
        tf_d_ffn: int,
        tf_num_layers: int,
        tf_num_head: int,
        tf_dropout: float,
        tokens_per_frame: int,
        num_poses: int,
        num_time_steps: int,
        num_context_frames: int,
        status_dim: int,
        use_spatial_tokens: bool,
        num_modes: int,
        use_temporal: bool,
        use_time_aligned_bias: bool,
        time_aligned_bias_strength: float,
        use_status_for_planner: bool,
        command_dim: int,
        use_action_history: bool,
        action_history_dim: int,
        num_observed_frames: int,
        use_z_context: bool = True,
    ):
        super().__init__()
        from .multimodal_planner import MultiModalTemporalPlanner

        self.core = MultiModalTemporalPlanner(
            encoder_dim=encoder_dim,
            tf_d_model=tf_d_model,
            tf_d_ffn=tf_d_ffn,
            tf_num_layers=tf_num_layers,
            tf_num_head=tf_num_head,
            tf_dropout=tf_dropout,
            tokens_per_frame=tokens_per_frame,
            num_poses=num_poses,
            num_time_steps=num_time_steps,
            num_context_frames=num_context_frames,
            status_dim=status_dim,
            use_spatial_tokens=use_spatial_tokens,
            num_modes=num_modes,
            use_temporal=use_temporal,
            use_time_aligned_bias=use_time_aligned_bias,
            time_aligned_bias_strength=time_aligned_bias_strength,
            use_z_context=use_z_context,
            use_status_for_planner=use_status_for_planner,
            use_action_history=use_action_history,
            action_history_dim=action_history_dim,
            num_observed_frames=num_observed_frames,
            command_dim=command_dim,
        )
        self.num_modes = num_modes
        self.num_poses = num_poses
        self.hidden_dim = tf_d_model

    def _context_uses_temporal_path(self, z_context: torch.Tensor) -> bool:
        if self.core.use_temporal:
            return True
        return z_context.shape[1] > self.core.tokens_per_frame

    def _build_context_memory(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        history_traj: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._context_uses_temporal_path(z_context):
            num_steps = z_context.shape[1] // self.core.tokens_per_frame
            return self.core._build_memory_temporal(z_context, status_feature, num_steps, action_history=history_traj)
        return self.core._build_memory_single(z_context, status_feature, action_history=history_traj), None

    def forward(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        history_traj: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        memory, memory_step_idx = self._build_context_memory(z_context, status_feature, history_traj)
        query = self.core.query_embedding.weight.unsqueeze(0).expand(z_context.shape[0], -1, -1)
        if memory_step_idx is not None:
            memory_mask = self.core._build_time_aligned_memory_bias(
                self.core.query_step_idx, memory_step_idx, memory.dtype
            )
            query_out = self.core.transformer(src=memory, tgt=query, memory_mask=memory_mask)
        else:
            query_out = self.core.transformer(src=memory, tgt=query)

        batch_size = query_out.shape[0]
        query_out = query_out.view(batch_size, self.num_modes, self.num_poses, self.hidden_dim)
        trajectories = []
        for mode_idx in range(self.num_modes):
            head_out = self.core.trajectory_heads[mode_idx](query_out[:, mode_idx])
            trajectories.append(head_out["trajectory"] if isinstance(head_out, dict) else head_out)
        trajectories = torch.stack(trajectories, dim=1)
        conf_feat = query_out.mean(dim=2)
        confidences = self.core.confidence_head(conf_feat.reshape(batch_size, self.num_modes * self.hidden_dim))
        return {
            "trajectories": trajectories,
            "confidences": confidences,
            "proposal_features": query_out,
        }


class DiffusionProposalProvider(BaseProposalProvider):
    """Frozen diffusion proposal provider with deterministic feature projection."""

    def __init__(
        self,
        encoder_dim: int,
        num_poses: int,
        status_dim: int,
        hidden_dim: int,
        depth: int,
        heads: int,
        dropout: float,
        mlp_ratio: float,
        traj_dim: int,
        sde_beta_min: float,
        sde_beta_max: float,
        num_samples: int,
        inference_steps: int,
        tokens_per_frame: int,
        num_modes: int,
        use_last_frame_only: bool,
        independent_modes: bool,
        use_anchor_frame: bool,
        cls_loss_weight: float,
        reg_loss_weight: float,
        vel_loss_weight: float,
        yaw_loss_weight: float,
        conf_temperature: float,
        cls_th: float,
        cls_ignore: float,
        command_dim: int,
        trajectory_token_mode: str,
        adaln_version: str,
        mode_token_expansion: bool,
        proposal_hidden_dim: int,
        use_action_history: bool,
        action_history_dim: int,
        num_observed_frames: int,
        use_z_context: bool = True,
        observed_token_mode: str = "none",
    ):
        super().__init__()
        from .diffusion_planner import DiffusionPlanner

        self.core = DiffusionPlanner(
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
            num_samples=num_samples,
            inference_steps=max(2, inference_steps),
            use_z_context=use_z_context,
            tokens_per_frame=tokens_per_frame,
            use_last_frame_only=use_last_frame_only,
            use_action_history=use_action_history,
            action_history_dim=action_history_dim,
            num_observed_frames=num_observed_frames,
            observed_token_mode=observed_token_mode,
            num_modes=num_modes,
            independent_modes=independent_modes,
            use_anchor_frame=use_anchor_frame,
            cls_loss_weight=cls_loss_weight,
            reg_loss_weight=reg_loss_weight,
            vel_loss_weight=vel_loss_weight,
            yaw_loss_weight=yaw_loss_weight,
            conf_temperature=conf_temperature,
            cls_th=cls_th,
            cls_ignore=cls_ignore,
            command_dim=command_dim,
            trajectory_token_mode=trajectory_token_mode,
            adaln_version=adaln_version,
            mode_token_expansion=mode_token_expansion,
        )
        self.proposal_hidden_dim = proposal_hidden_dim

    def forward(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        history_traj: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = self.core(
            z_context,
            status_feature,
            z_context=z_context,
            action_history=history_traj,
            inference_noise=inference_noise,
        )
        output["proposal_features"] = _trajectory_features_from_pose(output["trajectories"], self.proposal_hidden_dim)
        output["confidences"] = output["confidences"].clamp_min(1e-8).log()
        return output


class HistoryKinematicProposalProvider(BaseProposalProvider):
    """Deterministic multi-modal proposal generator driven by observed history."""

    def __init__(
        self,
        num_modes: int,
        num_poses: int,
        hidden_dim: int,
        temperature: float = 1.0,
    ):
        super().__init__()
        if num_modes < 1:
            raise ValueError(f"num_modes must be >= 1, got {num_modes}")
        if num_poses < 1:
            raise ValueError(f"num_poses must be >= 1, got {num_poses}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.num_modes = int(num_modes)
        self.num_poses = int(num_poses)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)

        yaw_offsets = torch.linspace(-0.3, 0.3, steps=self.num_modes, dtype=torch.float32)
        speed_scales = torch.linspace(0.85, 1.15, steps=self.num_modes, dtype=torch.float32)
        confidence_logits = -torch.linspace(0.0, 1.0, steps=self.num_modes, dtype=torch.float32) / self.temperature
        self.register_buffer("yaw_offsets", yaw_offsets, persistent=False)
        self.register_buffer("speed_scales", speed_scales, persistent=False)
        self.register_buffer("confidence_logits", confidence_logits, persistent=False)

    def _build_mode_features(self, trajectories: torch.Tensor) -> torch.Tensor:
        return _trajectory_features_from_pose(trajectories, self.hidden_dim)

    def forward(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        history_traj: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del z_context, status_feature
        if history_traj is None:
            raise ValueError("history_traj is required for HistoryKinematicProposalProvider")
        if history_traj.ndim != 3 or history_traj.shape[-1] < 3:
            raise ValueError(f"Expected history_traj shape [B, T, 3+], got {tuple(history_traj.shape)}")

        batch_size = history_traj.shape[0]
        device = history_traj.device
        dtype = history_traj.dtype

        if history_traj.shape[1] > 1:
            delta_xy = history_traj[:, -1, :2] - history_traj[:, -2, :2]
            delta_yaw = history_traj[:, -1, 2] - history_traj[:, -2, 2]
        else:
            delta_xy = torch.zeros(batch_size, 2, device=device, dtype=dtype)
            delta_yaw = torch.zeros(batch_size, device=device, dtype=dtype)

        step_index = torch.arange(1, self.num_poses + 1, device=device, dtype=dtype).view(1, 1, self.num_poses, 1)
        yaw_offsets = self.yaw_offsets.to(device=device, dtype=dtype).view(1, self.num_modes, 1)
        speed_scales = self.speed_scales.to(device=device, dtype=dtype).view(1, self.num_modes, 1, 1)

        scaled_delta = delta_xy.view(batch_size, 1, 1, 2) * speed_scales
        xy = step_index * scaled_delta
        yaw = step_index.squeeze(-1) * delta_yaw.view(batch_size, 1, 1) + yaw_offsets
        yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
        trajectories = torch.cat([xy, yaw.unsqueeze(-1)], dim=-1)

        proposal_features = self._build_mode_features(trajectories)
        confidences = (
            self.confidence_logits.to(device=device, dtype=dtype).view(1, self.num_modes).expand(batch_size, -1)
        )
        return {
            "trajectories": trajectories,
            "confidences": confidences,
            "proposal_features": proposal_features,
        }


def build_proposal_provider(
    config,
    encoder_dim: int,
    tokens_per_frame: int,
    num_poses: int,
    status_dim: int,
    command_dim: int,
    num_context_frames: Optional[int] = None,
    num_observed_frames: Optional[int] = None,
):
    proposal_cfg = config.proposal
    provider_type = proposal_cfg.provider_type
    planner_cfg = getattr(config, "planner", None)
    train_cfg = getattr(config, "train", None)
    use_action_history = bool(getattr(planner_cfg, "use_action_history_for_planner", False))
    action_history_dim = int(getattr(planner_cfg, "action_history_dim", 3))
    num_context_steps = int(num_context_frames or getattr(train_cfg, "num_observed_frames", 1))
    num_observed_frames = int(num_observed_frames or getattr(train_cfg, "num_observed_frames", num_context_steps))
    provider_num_modes = int(getattr(proposal_cfg, "provider_num_modes", None) or proposal_cfg.num_modes)

    if provider_type == "history_kinematic":
        return HistoryKinematicProposalProvider(
            num_modes=provider_num_modes,
            num_poses=num_poses,
            hidden_dim=proposal_cfg.hidden_dim,
            temperature=proposal_cfg.history_temperature,
        )

    if provider_type == "transformer":
        return TransformerProposalProvider(
            encoder_dim=encoder_dim,
            tf_d_model=config.planner.tf_d_model,
            tf_d_ffn=config.planner.tf_d_ffn,
            tf_num_layers=config.planner.tf_num_layers,
            tf_num_head=config.planner.tf_num_head,
            tf_dropout=config.planner.tf_dropout,
            tokens_per_frame=tokens_per_frame,
            num_poses=num_poses,
            num_time_steps=num_context_steps,
            num_context_frames=num_context_steps,
            status_dim=status_dim,
            use_spatial_tokens=config.planner.use_spatial_tokens,
            num_modes=provider_num_modes,
            use_temporal=config.planner.use_temporal,
            use_time_aligned_bias=proposal_cfg.temporal_alignment,
            time_aligned_bias_strength=5.0,
            use_status_for_planner=config.planner.use_status_for_planner,
            command_dim=command_dim,
            use_action_history=use_action_history,
            action_history_dim=action_history_dim,
            num_observed_frames=num_observed_frames,
            use_z_context=proposal_cfg.use_z_context,
        )

    if provider_type == "diffusion":
        return DiffusionProposalProvider(
            encoder_dim=encoder_dim,
            num_poses=num_poses,
            status_dim=status_dim,
            hidden_dim=config.planner.diff_hidden_dim,
            depth=config.planner.diff_num_layers,
            heads=config.planner.diff_num_heads,
            dropout=config.planner.diff_dropout,
            mlp_ratio=config.planner.diff_mlp_ratio,
            traj_dim=config.planner.diff_traj_dim,
            sde_beta_min=config.planner.diff_sde_beta_min,
            sde_beta_max=config.planner.diff_sde_beta_max,
            num_samples=max(
                getattr(config.planner, "diff_num_samples", proposal_cfg.num_modes), proposal_cfg.num_modes
            ),
            inference_steps=config.planner.diff_inference_steps,
            tokens_per_frame=tokens_per_frame,
            num_modes=provider_num_modes,
            use_last_frame_only=config.planner.diff_use_last_frame_only,
            independent_modes=config.planner.diff_independent_modes,
            use_anchor_frame=config.planner.diff_use_anchor_frame,
            cls_loss_weight=config.planner.diff_cls_loss_weight,
            reg_loss_weight=config.planner.diff_reg_loss_weight,
            vel_loss_weight=config.planner.diff_vel_loss_weight,
            yaw_loss_weight=config.planner.diff_yaw_loss_weight,
            conf_temperature=config.planner.diff_conf_temperature,
            cls_th=config.planner.diff_cls_th,
            cls_ignore=config.planner.diff_cls_ignore,
            command_dim=command_dim,
            trajectory_token_mode=getattr(config.planner, "diff_trajectory_token_mode", "single_token"),
            adaln_version=config.planner.diff_adaln_version,
            mode_token_expansion=getattr(config.planner, "diff_mode_token_expansion", False),
            proposal_hidden_dim=proposal_cfg.hidden_dim,
            use_action_history=use_action_history,
            action_history_dim=action_history_dim,
            num_observed_frames=num_observed_frames,
            observed_token_mode="none",
            use_z_context=proposal_cfg.use_z_context,
        )

    raise ValueError(f"Unsupported proposal provider_type: {provider_type}")
