# Copyright (c) 2026 RISE Contributors
# RISE provenance: independent-diffusion-v1
"""Independent trajectory diffusion planner built from ordinary PyTorch modules."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..diffusion_utils import dpm_solver_pytorch as dpm
from ..diffusion_utils.sampling import dpm_sampler
from ..diffusion_utils.sde import VPSDE_linear
from .planner_contracts import (
    PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    PLANNER_OBSERVED_TOKEN_NONE,
    normalize_observed_token_mode,
)

_SUPPORTED_RETAINED_SOLVER_DTYPES = {torch.float32, torch.float64}


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class _ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        intermediate_dim = min(hidden_dim * 2, 256)
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float, dropout: float):
        super().__init__()
        expanded_dim = int(hidden_dim * mlp_ratio)
        self.fc1 = nn.Linear(hidden_dim, expanded_dim)
        self.fc2 = nn.Linear(expanded_dim, hidden_dim)
        self._dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self._dropout, training=self.training)
        return self.fc2(x)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(256, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if t.ndim != 1:
            raise ValueError(f"diffusion time must have shape [B], got {tuple(t.shape)}")
        half = 128
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        parameter = self.mlp[0].weight
        return self.mlp(embedding.to(dtype=parameter.dtype))


class _LegacyTrajectoryBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp1 = _FeedForward(hidden_dim, mlp_ratio, dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm4 = nn.LayerNorm(hidden_dim)
        self.mlp2 = _FeedForward(hidden_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, cross_c: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(condition).chunk(
            6, dim=-1
        )
        query = _modulate(self.norm1(x), shift_attn, scale_attn)
        x = x + gate_attn[:, None, :] * self.attn(query, query, query, need_weights=False)[0]
        x = x + gate_mlp[:, None, :] * self.mlp1(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        query = _modulate(self.norm3(x), shift_attn, scale_attn)
        x = x + gate_attn[:, None, :] * self.cross_attn(query, cross_c, cross_c, need_weights=False)[0]
        x = x + gate_mlp[:, None, :] * self.mlp2(_modulate(self.norm4(x), shift_mlp, scale_mlp))
        return x


class _V2TrajectoryBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float, mlp_ratio: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm_cond = nn.LayerNorm(hidden_dim)
        self.mlp = _FeedForward(hidden_dim, mlp_ratio, dropout)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 9))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, cross_c: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        values = self.adaLN_modulation(condition).chunk(9, dim=-1)
        shift_self, scale_self, gate_self = values[:3]
        shift_cross, scale_cross, gate_cross = values[3:6]
        shift_mlp, scale_mlp, gate_mlp = values[6:]
        normalized = F.layer_norm(x, (x.shape[-1],))
        query = _modulate(normalized, shift_self, scale_self)
        x = x + gate_self[:, None, :] * self.attn(query, query, query, need_weights=False)[0]
        query = _modulate(F.layer_norm(x, (x.shape[-1],)), shift_cross, scale_cross)
        context = self.norm_cond(cross_c)
        x = x + gate_cross[:, None, :] * self.cross_attn(query, context, context, need_weights=False)[0]
        normalized = _modulate(F.layer_norm(x, (x.shape[-1],)), shift_mlp, scale_mlp)
        return x + gate_mlp[:, None, :] * self.mlp(normalized)


class _TrajectoryFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        traj_dim: int,
        total_frames: int,
        num_modes: int,
        trajectory_token_mode: str,
        mode_token_expansion: bool,
        uses_batch_expansion: bool,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 2))

        self._single_token = trajectory_token_mode == "single_token"
        self._mode_token_expansion = mode_token_expansion
        self._uses_batch_expansion = uses_batch_expansion
        self._num_modes = num_modes
        self._total_frames = total_frames
        self._traj_dim = traj_dim

        if not self._single_token and num_modes > 1 and not uses_batch_expansion and not mode_token_expansion:
            self.proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, num_modes * hidden_dim),
            )
        if not uses_batch_expansion and num_modes > 1:
            self.cls = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        output_dim = total_frames * traj_dim if self._single_token else traj_dim
        self.reg = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, output_dim),
        )

    def _condition(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return _modulate(self.norm_final(tokens), shift, scale)

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        tokens = self._condition(tokens, condition)
        if self._single_token:
            if self._uses_batch_expansion or self._num_modes == 1:
                return None, self.reg(tokens[:, 0])
            if tokens.shape[1] != self._num_modes:
                raise ValueError(
                    f"joint single-token denoiser expected {self._num_modes} mode tokens, got {tokens.shape[1]}"
                )
            classification = self.cls(tokens).squeeze(-1)
            prediction = self.reg(tokens).reshape(tokens.shape[0], -1)
            return classification, prediction

        if self._mode_token_expansion:
            batch_size = tokens.shape[0]
            hidden_dim = tokens.shape[-1]
            mode_tokens = tokens.reshape(batch_size, self._num_modes, self._total_frames, hidden_dim)
            classification = self.cls(mode_tokens).squeeze(-1).mean(dim=-1)
            prediction = self.reg(mode_tokens).reshape(batch_size, -1)
            return classification, prediction

        if self._uses_batch_expansion:
            return None, self.reg(tokens).reshape(tokens.shape[0], -1)

        batch_size = tokens.shape[0]
        hidden_dim = tokens.shape[-1]
        mode_tokens = self.proj(tokens).reshape(batch_size, self._total_frames, self._num_modes, hidden_dim)
        mode_tokens = mode_tokens.permute(0, 2, 1, 3)
        classification = self.cls(mode_tokens).squeeze(-1).mean(dim=-1)
        prediction = self.reg(mode_tokens).reshape(batch_size, -1)
        return classification, prediction


class _TrajectoryDiT(nn.Module):
    model_type = "x_start"

    def __init__(
        self,
        *,
        hidden_dim: int,
        depth: int,
        heads: int,
        dropout: float,
        mlp_ratio: float,
        traj_dim: int,
        total_frames: int,
        num_modes: int,
        trajectory_token_mode: str,
        adaln_version: str,
        mode_token_expansion: bool,
        uses_batch_expansion: bool,
    ):
        super().__init__()
        self._hidden_dim = hidden_dim
        self._traj_dim = traj_dim
        self._total_frames = total_frames
        self._num_modes = num_modes
        self._trajectory_token_mode = trajectory_token_mode
        self._mode_token_expansion = mode_token_expansion
        self._uses_batch_expansion = uses_batch_expansion

        if trajectory_token_mode == "per_pose_token":
            self.pose_embed = nn.Parameter(torch.zeros(1, total_frames, hidden_dim))
        if mode_token_expansion:
            self.mode_embed = nn.Parameter(torch.zeros(1, num_modes, 1, hidden_dim))

        if trajectory_token_mode == "single_token":
            projection_input_dim = total_frames * traj_dim
        elif num_modes > 1 and not uses_batch_expansion and not mode_token_expansion:
            projection_input_dim = num_modes * traj_dim
        else:
            projection_input_dim = traj_dim
        self.preproj = _ProjectionMLP(projection_input_dim, hidden_dim)
        self.final_layer = _TrajectoryFinalLayer(
            hidden_dim,
            traj_dim,
            total_frames,
            num_modes,
            trajectory_token_mode,
            mode_token_expansion,
            uses_batch_expansion,
        )
        self.t_embedder = _TimestepEmbedder(hidden_dim)
        block_type = _V2TrajectoryBlock if adaln_version == "v2" else _LegacyTrajectoryBlock
        self.blocks = nn.ModuleList(block_type(hidden_dim, heads, dropout, mlp_ratio) for _ in range(depth))

    def _tokens_from_input(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        if self._trajectory_token_mode == "single_token":
            values_per_mode = self._total_frames * self._traj_dim
            if not self._uses_batch_expansion and self._num_modes > 1:
                expected = self._num_modes * values_per_mode
                if x.numel() != batch_size * expected:
                    raise ValueError(f"trajectory denoiser expected {expected} values per batch item")
                values = x.reshape(batch_size, self._num_modes, values_per_mode)
                return self.preproj(values)
            expected = values_per_mode
            if x.numel() != batch_size * expected:
                raise ValueError(f"trajectory denoiser expected {expected} values per batch item")
            return self.preproj(x.reshape(batch_size, expected)).unsqueeze(1)

        if self._mode_token_expansion:
            expected = self._num_modes * self._total_frames * self._traj_dim
            if x.numel() != batch_size * expected:
                raise ValueError(f"trajectory denoiser expected {expected} values per batch item")
            values = x.reshape(batch_size, self._num_modes, self._total_frames, self._traj_dim)
            tokens = self.preproj(values)
            tokens = tokens + self.pose_embed[:, None, :, :] + self.mode_embed
            return tokens.reshape(batch_size, self._num_modes * self._total_frames, self._hidden_dim)

        if not self._uses_batch_expansion and self._num_modes > 1:
            expected = self._num_modes * self._total_frames * self._traj_dim
            if x.numel() != batch_size * expected:
                raise ValueError(f"trajectory denoiser expected {expected} values per batch item")
            values = x.reshape(batch_size, self._num_modes, self._total_frames, self._traj_dim)
            values = values.permute(0, 2, 1, 3).reshape(
                batch_size, self._total_frames, self._num_modes * self._traj_dim
            )
            return self.preproj(values) + self.pose_embed

        expected = self._total_frames * self._traj_dim
        if x.numel() != batch_size * expected:
            raise ValueError(f"trajectory denoiser expected {expected} values per batch item")
        values = x.reshape(batch_size, self._total_frames, self._traj_dim)
        return self.preproj(values) + self.pose_embed

    def forward(self, x, t, cross_c, status_emb):
        tokens = self._tokens_from_input(x)
        if not isinstance(cross_c, torch.Tensor) or cross_c.ndim != 3:
            raise ValueError("cross_c must have shape [B, N, hidden_dim]")
        if not isinstance(status_emb, torch.Tensor) or status_emb.ndim != 2:
            raise ValueError("status_emb must have shape [B, hidden_dim]")
        if cross_c.shape[0] != tokens.shape[0] or status_emb.shape[0] != tokens.shape[0]:
            raise ValueError("denoiser conditioning batch size must match x")
        condition = self.t_embedder(t) + status_emb
        for block in self.blocks:
            tokens = block(tokens, cross_c, condition)
        return self.final_layer(tokens, condition)


class _IndependentConfidenceHead(nn.Module):
    def __init__(self, hidden_dim: int, num_poses: int, traj_dim: int):
        super().__init__()
        self.context_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.xy_norm = nn.LayerNorm(2)
        self.yaw_norm = nn.LayerNorm(2)
        self.traj_encoder = nn.Sequential(
            nn.Linear(num_poses * traj_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.scorer = nn.Sequential(nn.Linear(128 + hidden_dim, 128), nn.GELU(), nn.Linear(128, 1))
        self._num_poses = num_poses
        self._traj_dim = traj_dim
        self._yaw_slice = slice(4, 6) if traj_dim >= 6 else slice(2, 4)

    def forward(self, trajectories: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        if trajectories.ndim != 4:
            raise ValueError("confidence trajectories must have shape [B, K, P, traj_dim]")
        values = trajectories.clone()
        values[..., :2] = self.xy_norm(values[..., :2])
        values[..., self._yaw_slice] = self.yaw_norm(values[..., self._yaw_slice])
        encoded = self.traj_encoder(values.reshape(values.shape[0], values.shape[1], -1))
        query = self.context_query.expand(context_tokens.shape[0], -1, -1)
        attention = torch.softmax(torch.matmul(query, context_tokens.transpose(-1, -2)), dim=-1)
        context_summary = torch.matmul(attention, context_tokens).expand(-1, trajectories.shape[1], -1)
        return self.scorer(torch.cat((encoded, context_summary), dim=-1)).squeeze(-1)


class DiffusionPlanner(nn.Module):
    """VP-SDE trajectory planner with joint and independent multi-mode topologies."""

    def __init__(
        self,
        encoder_dim: int = 1408,
        num_poses: int = 7,
        status_dim: int = 7,
        hidden_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
        traj_dim: int = 6,
        sde_beta_min: float = 0.1,
        sde_beta_max: float = 20.0,
        num_samples: int = 6,
        inference_steps: int = 2,
        use_z_context: bool = False,
        tokens_per_frame: int = 256,
        trajectory_token_mode: str = "single_token",
        use_last_frame_only: bool = True,
        use_action_history: bool = False,
        action_history_dim: int = 3,
        num_observed_frames: int = 1,
        observed_token_mode: Optional[str] = None,
        num_modes: int = 1,
        independent_modes: bool = False,
        use_anchor_frame: bool = False,
        cls_loss_weight: float = 1.0,
        reg_loss_weight: float = 1.0,
        vel_loss_weight: float = 0.5,
        yaw_loss_weight: float = 0.5,
        reg_timestep_weights: Optional[torch.Tensor] = None,
        awta_init_temperature: float = 8.0,
        awta_min_temperature: float = 0.1,
        conf_temperature: float = 1.5,
        cls_th: float = 2.0,
        cls_ignore: float = 0.2,
        command_dim: int = 0,
        adaln_version: str = "legacy",
        mode_token_expansion: bool = False,
    ):
        super().__init__()
        if num_poses < 1:
            raise ValueError(f"num_poses must be positive, got {num_poses}")
        if num_modes < 1:
            raise ValueError(f"num_modes must be positive, got {num_modes}")
        if independent_modes and num_modes == 1:
            raise ValueError("independent_modes requires num_modes greater than one")
        if num_samples < 1:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if inference_steps < 2:
            raise ValueError(f"inference_steps must be at least 2, got {inference_steps}")
        if trajectory_token_mode not in {"single_token", "per_pose_token"}:
            raise ValueError(f"invalid trajectory_token_mode {trajectory_token_mode!r}")
        if adaln_version not in {"legacy", "v2", "v3"}:
            raise ValueError(f"invalid adaln_version {adaln_version!r}")
        if traj_dim not in {4, 6}:
            raise ValueError(f"traj_dim must be 4 or 6, got {traj_dim}")
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be positive and divisible by heads")
        if depth < 1:
            raise ValueError(f"depth must be positive, got {depth}")
        if command_dim < 0 or command_dim > status_dim:
            raise ValueError(f"command_dim must be within 0..status_dim, got {command_dim}")
        if mode_token_expansion and trajectory_token_mode != "per_pose_token":
            raise ValueError("mode_token_expansion requires trajectory_token_mode='per_pose_token'")
        if mode_token_expansion and independent_modes:
            raise ValueError("mode_token_expansion is incompatible with independent_modes")

        normalized_observed_mode = normalize_observed_token_mode(observed_token_mode, True)
        self.encoder_dim = int(encoder_dim)
        self.num_poses = int(num_poses)
        self.traj_dim = int(traj_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_samples = int(num_samples)
        self.num_modes = int(num_modes)
        self.independent_modes = bool(independent_modes)
        self.inference_steps = int(inference_steps)
        self.use_z_context = bool(use_z_context)
        self.tokens_per_frame = int(tokens_per_frame)
        self.trajectory_token_mode = trajectory_token_mode
        self.use_last_frame_only = bool(use_last_frame_only)
        self.use_action_history = bool(use_action_history)
        self.action_history_dim = int(action_history_dim)
        self.num_observed_frames = int(num_observed_frames)
        self.observed_token_mode = normalized_observed_mode
        self.use_observed_tokens = normalized_observed_mode != PLANNER_OBSERVED_TOKEN_NONE
        self.use_anchor_frame = bool(use_anchor_frame)
        self.total_frames = self.num_poses + int(self.use_anchor_frame)
        self.command_dim = int(command_dim)
        self.adaln_version = adaln_version
        self.mode_token_expansion = bool(mode_token_expansion)
        self.cls_loss_weight = float(cls_loss_weight)
        self.reg_loss_weight = float(reg_loss_weight)
        self.vel_loss_weight = float(vel_loss_weight)
        self.yaw_loss_weight = float(yaw_loss_weight)
        self.awta_min_temperature = float(awta_min_temperature)
        self.conf_temperature = float(conf_temperature)
        self.cls_th = float(cls_th)
        self.cls_ignore = float(cls_ignore)

        self._status_dim = int(status_dim)
        self._has_velocity = self.traj_dim >= 6
        self._yaw_slice = slice(4, 6) if self._has_velocity else slice(2, 4)
        self._uses_batch_expansion = self.num_modes == 1 or self.independent_modes
        self._batch_K = self.num_modes if self._uses_batch_expansion and self.num_modes > 1 else self.num_samples

        self.context_proj = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        if self.observed_token_mode == PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED:
            self.observed_source_embedding = nn.Embedding(2, self.encoder_dim)

        if self.command_dim == 0:
            self.status_proj = nn.Sequential(
                nn.Linear(self._status_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.command_proj = nn.Sequential(
                nn.Linear(self.command_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            kinematics_dim = self._status_dim - self.command_dim
            self.kinematics_proj = nn.Sequential(
                nn.LayerNorm(kinematics_dim),
                nn.Linear(kinematics_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

        if self.use_action_history:
            self.action_history_proj = nn.Sequential(
                nn.LayerNorm(self.action_history_dim),
                nn.Linear(self.action_history_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.action_history_pos_embedding = nn.Embedding(self.num_observed_frames, self.hidden_dim)

        self.dit = _TrajectoryDiT(
            hidden_dim=self.hidden_dim,
            depth=depth,
            heads=heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            traj_dim=self.traj_dim,
            total_frames=self.total_frames,
            num_modes=self.num_modes,
            trajectory_token_mode=self.trajectory_token_mode,
            adaln_version=self.adaln_version,
            mode_token_expansion=self.mode_token_expansion,
            uses_batch_expansion=self._uses_batch_expansion,
        )
        if self.independent_modes:
            self.confidence_head = _IndependentConfidenceHead(self.hidden_dim, self.num_poses, self.traj_dim)
        else:
            self.confidence_head = None

        if reg_timestep_weights is not None:
            if not isinstance(reg_timestep_weights, torch.Tensor):
                raise TypeError("reg_timestep_weights must be a torch.Tensor")
            if reg_timestep_weights.ndim != 1 or reg_timestep_weights.shape[0] != self.num_poses:
                raise ValueError(
                    f"reg_timestep_weights must have shape [{self.num_poses}], got {tuple(reg_timestep_weights.shape)}"
                )
            weights = reg_timestep_weights.to(dtype=torch.float32)
            if not bool(torch.isfinite(weights).all()):
                raise ValueError("reg_timestep_weights must contain only finite values")
            if bool((weights < 0.0).any()):
                raise ValueError("reg_timestep_weights must be non-negative")
            if not bool(weights.sum() > 0.0):
                raise ValueError("reg_timestep_weights must have a positive sum")
            self.register_buffer("reg_timestep_weights", weights, persistent=False)
        self.register_buffer(
            "awta_temperature", torch.tensor(float(awta_init_temperature), dtype=torch.float32), persistent=False
        )
        self.sde = VPSDE_linear(beta_min=sde_beta_min, beta_max=sde_beta_max)

    def set_awta_temperature(self, T: float) -> None:
        value = float(T)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"AWTA temperature must be finite and positive, got {T!r}")
        with torch.no_grad():
            self.awta_temperature.fill_(value)

    @staticmethod
    def convert_3d_to_6d(traj_3d: torch.Tensor, dt: float = 0.2) -> torch.Tensor:
        return DiffusionPlanner.convert_3d_to_nd(traj_3d, dt=dt, traj_dim=6)

    @staticmethod
    def convert_3d_to_nd(traj_3d: torch.Tensor, dt: float = 0.2, traj_dim: int = 6) -> torch.Tensor:
        if not isinstance(traj_3d, torch.Tensor) or traj_3d.ndim < 2 or traj_3d.shape[-1] != 3:
            raise ValueError("traj_3d must have shape [..., num_poses, 3]")
        if not traj_3d.is_floating_point():
            raise TypeError("traj_3d must have a floating dtype")
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt}")
        if traj_dim not in {4, 6}:
            raise ValueError(f"traj_dim must be 4 or 6, got {traj_dim}")
        x, y, yaw = traj_3d.unbind(dim=-1)
        if traj_dim == 4:
            return torch.stack((x, y, torch.cos(yaw), torch.sin(yaw)), dim=-1)
        vx = torch.zeros_like(x)
        vy = torch.zeros_like(y)
        vx[..., 0] = x[..., 0] / dt
        vy[..., 0] = y[..., 0] / dt
        if x.shape[-1] > 1:
            vx[..., 1:] = (x[..., 1:] - x[..., :-1]) / dt
            vy[..., 1:] = (y[..., 1:] - y[..., :-1]) / dt
        return torch.stack((x, y, vx, vy, torch.cos(yaw), torch.sin(yaw)), dim=-1)

    def _prepare_context(self, z_ar, z_context=None, z_observed=None, action_history=None):
        if not isinstance(z_ar, torch.Tensor) or z_ar.ndim != 3:
            raise ValueError("z_ar must have shape [B, N, encoder_dim]")
        if not z_ar.is_floating_point() or z_ar.shape[-1] != self.encoder_dim:
            raise ValueError(f"z_ar must be floating with encoder_dim={self.encoder_dim}")
        batch_size = z_ar.shape[0]
        if self.use_z_context:
            if not isinstance(z_context, torch.Tensor):
                raise ValueError("z_context is required when use_z_context=True")
            if z_context.ndim != 3 or z_context.shape[0] != batch_size or z_context.shape[-1] != self.encoder_dim:
                raise ValueError("z_context must have shape [B, N, encoder_dim]")
            base = z_context
        else:
            base = z_ar
        if base.device != z_ar.device or base.dtype != z_ar.dtype:
            raise ValueError("z_context and z_ar must share dtype and device")
        if self.use_last_frame_only:
            if 0 < base.shape[1] < self.tokens_per_frame:
                raise ValueError("planner context has fewer tokens than tokens_per_frame")
            if base.shape[1] > 0:
                base = base[:, -self.tokens_per_frame :]

        parts = []
        if self.use_observed_tokens:
            expected_tokens = self.num_observed_frames * self.tokens_per_frame
            if not isinstance(z_observed, torch.Tensor):
                raise ValueError("z_observed is required for the configured observed token mode")
            expected_shape = (batch_size, expected_tokens, self.encoder_dim)
            if tuple(z_observed.shape) != expected_shape:
                raise ValueError(f"z_observed must have shape {expected_shape}, got {tuple(z_observed.shape)}")
            if z_observed.dtype != z_ar.dtype or z_observed.device != z_ar.device:
                raise ValueError("z_observed must share z_ar dtype and device")
            if self.observed_token_mode == PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED:
                observed_embed = self.observed_source_embedding.weight[0].view(1, 1, self.encoder_dim)
                future_embed = self.observed_source_embedding.weight[1].view(1, 1, self.encoder_dim)
                parts.extend((z_observed + observed_embed, base + future_embed))
            else:
                parts.extend((z_observed, base))
        else:
            parts.append(base)
        context_tokens = self.context_proj(torch.cat(parts, dim=1))

        if self.use_action_history:
            expected_shape = (batch_size, self.num_observed_frames, self.action_history_dim)
            if not isinstance(action_history, torch.Tensor):
                raise ValueError("action_history is required when use_action_history=True")
            if tuple(action_history.shape) != expected_shape:
                raise ValueError(f"action_history must have shape {expected_shape}")
            if action_history.dtype != z_ar.dtype or action_history.device != z_ar.device:
                raise ValueError("action_history must share z_ar dtype and device")
            positions = torch.arange(self.num_observed_frames, device=z_ar.device)
            history_tokens = self.action_history_proj(action_history)
            history_tokens = history_tokens + self.action_history_pos_embedding(positions).unsqueeze(0)
            context_tokens = torch.cat((history_tokens, context_tokens), dim=1)
        if context_tokens.shape[1] == 0:
            raise ValueError("planner context must contain at least one token")
        return context_tokens

    def _prepare_status(self, status_feature):
        if not isinstance(status_feature, torch.Tensor) or status_feature.ndim != 2:
            raise ValueError("status_feature must have shape [B, status_dim]")
        if not status_feature.is_floating_point() or status_feature.shape[-1] != self._status_dim:
            raise ValueError(f"status_feature must be floating with status_dim={self._status_dim}")
        if self.command_dim == 0:
            return self.status_proj(status_feature)
        command = status_feature[:, : self.command_dim]
        kinematics = status_feature[:, self.command_dim :]
        return self.command_proj(command) + self.kinematics_proj(kinematics)

    def _get_anchor(self, anchor_state: Optional[torch.Tensor], B: int, device: torch.device) -> torch.Tensor:
        parameter = next(self.parameters())
        if device != parameter.device:
            raise ValueError("planner input device must match planner parameter device")
        if anchor_state is None:
            return torch.zeros(B, 1, self.traj_dim, device=parameter.device, dtype=parameter.dtype)
        if not isinstance(anchor_state, torch.Tensor) or not anchor_state.is_floating_point():
            raise TypeError("anchor_state must be a floating tensor")
        if anchor_state.device != parameter.device:
            raise ValueError("anchor_state device must match planner parameters")
        if anchor_state.dtype != parameter.dtype:
            raise ValueError("anchor_state dtype must match planner parameters")
        if tuple(anchor_state.shape) == (B, self.traj_dim):
            return anchor_state.unsqueeze(1)
        if tuple(anchor_state.shape) == (B, 1, self.traj_dim):
            return anchor_state
        raise ValueError(f"anchor_state must have shape [{B}, {self.traj_dim}]")

    def _resolve_inference_noise(
        self,
        inference_noise: Optional[torch.Tensor],
        *,
        B: int,
        K: int,
        device: torch.device,
    ) -> torch.Tensor:
        expected_shape = (B, K, self.num_poses, self.traj_dim)
        parameter = next(self.parameters())
        if inference_noise is None:
            return torch.randn(expected_shape, device=device, dtype=parameter.dtype)
        if not isinstance(inference_noise, torch.Tensor) or not inference_noise.is_floating_point():
            raise TypeError("inference_noise must be a floating tensor")
        if tuple(inference_noise.shape) != expected_shape:
            raise ValueError(f"inference_noise must have shape {expected_shape}, got {tuple(inference_noise.shape)}")
        if inference_noise.device != device:
            raise ValueError("inference_noise must be on the planner input device")
        if inference_noise.dtype != parameter.dtype:
            raise ValueError(f"inference_noise dtype must match planner dtype {parameter.dtype}")
        if not bool(torch.isfinite(inference_noise).all()):
            raise ValueError("inference_noise must contain only finite values")
        return inference_noise

    def _correct_anchor_xt(self, x_t: torch.Tensor, anchor_info: torch.Tensor) -> torch.Tensor:
        corrected = x_t.clone()
        if anchor_info.ndim == 3:
            values = corrected.reshape(anchor_info.shape[0], self.total_frames, self.traj_dim)
            values[:, :1] = anchor_info
        elif anchor_info.ndim == 4:
            values = corrected.reshape(anchor_info.shape[0], anchor_info.shape[1], self.total_frames, self.traj_dim)
            values[:, :, :1] = anchor_info
        else:
            raise ValueError("anchor_info must have rank 3 or 4")
        return values.reshape_as(x_t)

    @staticmethod
    def _convert_6d_to_3d(traj_6d: torch.Tensor) -> torch.Tensor:
        if traj_6d.shape[-1] != 6:
            raise ValueError("traj_6d last dimension must be 6")
        return torch.stack((traj_6d[..., 0], traj_6d[..., 1], torch.atan2(traj_6d[..., 5], traj_6d[..., 4])), dim=-1)

    @staticmethod
    def _convert_nd_to_3d(traj_nd: torch.Tensor) -> torch.Tensor:
        if traj_nd.shape[-1] not in {4, 6}:
            raise ValueError("trajectory last dimension must be 4 or 6")
        yaw_slice = slice(4, 6) if traj_nd.shape[-1] == 6 else slice(2, 4)
        return torch.stack(
            (
                traj_nd[..., 0],
                traj_nd[..., 1],
                torch.atan2(traj_nd[..., yaw_slice.stop - 1], traj_nd[..., yaw_slice.start]),
            ),
            dim=-1,
        )

    def _xy_regression_loss_per_mode(self, pred_xy: torch.Tensor, gt_xy: torch.Tensor) -> torch.Tensor:
        if pred_xy.ndim != 4 or pred_xy.shape[-1] != 2:
            raise ValueError("pred_xy must have shape [B, K, P, 2]")
        if gt_xy.ndim == 3:
            gt_xy = gt_xy.unsqueeze(1).expand(-1, pred_xy.shape[1], -1, -1)
        if gt_xy.shape != pred_xy.shape:
            raise ValueError("gt_xy must broadcast to pred_xy shape [B, K, P, 2]")
        per_timestep = (pred_xy - gt_xy).norm(dim=-1)
        weights = self._normalized_reg_timestep_weights(per_timestep)
        if weights is None:
            return per_timestep.mean(dim=-1)
        return (per_timestep * weights.view(1, 1, -1)).sum(dim=-1)

    def _normalized_reg_timestep_weights(self, reference: torch.Tensor) -> Optional[torch.Tensor]:
        weights = getattr(self, "reg_timestep_weights", None)
        if weights is None:
            return None
        weights = weights.to(device=reference.device, dtype=reference.dtype)
        denominator = weights.sum()
        if not bool(torch.isfinite(weights).all()) or not bool(torch.isfinite(denominator)):
            raise ValueError("reg_timestep_weights must contain only finite values")
        if bool((weights < 0.0).any()):
            raise ValueError("reg_timestep_weights must be non-negative")
        if not bool(denominator > 0.0):
            raise ValueError("reg_timestep_weights must have a positive sum")
        return weights / denominator

    def _regression_loss_per_mode(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.ndim != 4 or prediction.shape != target.shape:
            raise ValueError("prediction and target must have matching shape [B, K, P, traj_dim]")
        per_timestep = F.mse_loss(prediction, target, reduction="none").mean(dim=-1)
        weights = self._normalized_reg_timestep_weights(per_timestep)
        if weights is None:
            return per_timestep.mean(dim=-1)
        return (per_timestep * weights.view(1, 1, -1)).sum(dim=-1)

    def _validate_training_target(self, gt_trajectory: torch.Tensor, batch_size: int) -> None:
        expected = (batch_size, self.num_poses, self.traj_dim)
        if not isinstance(gt_trajectory, torch.Tensor) or not gt_trajectory.is_floating_point():
            raise TypeError("gt_trajectory must be a floating tensor")
        if tuple(gt_trajectory.shape) != expected:
            raise ValueError(f"gt_trajectory must have shape {expected}, got {tuple(gt_trajectory.shape)}")
        parameter = next(self.parameters())
        if gt_trajectory.device != parameter.device or gt_trajectory.dtype != parameter.dtype:
            raise ValueError("gt_trajectory dtype/device must match planner parameters")
        if not bool(torch.isfinite(gt_trajectory).all()):
            raise ValueError("gt_trajectory must contain only finite values")

    def _endpoint_losses(
        self, pred: torch.Tensor, gt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        reg_loss = self._xy_regression_loss_per_mode(pred[..., :2], gt[..., :2]).mean()
        if self._has_velocity:
            target_velocity = gt[..., 2:4].unsqueeze(1).expand_as(pred[..., 2:4])
            vel_loss = F.smooth_l1_loss(pred[..., 2:4], target_velocity)
        else:
            vel_loss = pred.sum() * 0.0
        target_yaw = gt[..., self._yaw_slice].unsqueeze(1).expand_as(pred[..., self._yaw_slice])
        cosine = F.cosine_similarity(pred[..., self._yaw_slice], target_yaw, dim=-1)
        yaw_loss = ((1.0 - cosine) * 0.5).mean()
        return reg_loss, vel_loss, yaw_loss

    def _perturb(self, clean: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean)
        mean, std = self.sde.marginal_prob(clean, t)
        x_t = mean + std * noise
        alpha, _ = self.sde.marginal_prob(torch.ones_like(clean), t)
        return x_t, noise, alpha

    def _build_multimodal_loss_dict(
        self,
        pred_noise: torch.Tensor,
        target_noise: torch.Tensor,
        x_pred: torch.Tensor,
        cls_pred: Optional[torch.Tensor],
        gt_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, num_modes = x_pred.shape[:2]
        gt_modes = gt_trajectory.unsqueeze(1).expand(-1, num_modes, -1, -1)
        dist_xy = (x_pred[..., :2] - gt_modes[..., :2]).norm(dim=-1).mean(dim=-1)
        min_dist, winner_idx = dist_xy.min(dim=1)
        temperature = torch.clamp(
            self.awta_temperature.to(device=x_pred.device, dtype=x_pred.dtype),
            min=self.awta_min_temperature,
        )
        awta_weights = torch.softmax(torch.clamp(-dist_xy / temperature, -50.0, 50.0), dim=1).detach()
        per_mode_regression = self._regression_loss_per_mode(pred_noise, target_noise)
        reg_loss = (awta_weights * per_mode_regression).sum(dim=1).mean()

        if self._has_velocity:
            per_mode_velocity = F.smooth_l1_loss(x_pred[..., 2:4], gt_modes[..., 2:4], reduction="none").mean(
                dim=(-1, -2)
            )
            vel_loss = (awta_weights * per_mode_velocity).sum(dim=1).mean()
        else:
            vel_loss = x_pred.sum() * 0.0
        cosine = F.cosine_similarity(x_pred[..., self._yaw_slice], gt_modes[..., self._yaw_slice], dim=-1)
        per_mode_yaw = ((1.0 - cosine) * 0.5).mean(dim=-1)
        yaw_loss = (awta_weights * per_mode_yaw).sum(dim=1).mean()

        if cls_pred is not None:
            soft_target = torch.softmax(torch.clamp(-dist_xy / self.conf_temperature, -50.0, 50.0), dim=1).detach()
            sample_valid = min_dist < self.cls_th
            if bool(sample_valid.any()):
                mode_keep = (dist_xy - min_dist.unsqueeze(1)) > self.cls_ignore
                mode_keep = mode_keep | F.one_hot(winner_idx, num_classes=num_modes).bool()
                per_sample = -(soft_target * F.log_softmax(cls_pred, dim=1) * mode_keep.to(x_pred.dtype)).sum(dim=1)
                conf_loss = per_sample[sample_valid].mean()
            else:
                conf_loss = cls_pred.sum() * 0.0
        else:
            sample_valid = torch.zeros(batch_size, device=x_pred.device, dtype=torch.bool)
            conf_loss = pred_noise.sum() * 0.0
        cover_loss = pred_noise.sum() * 0.0
        total_loss = (
            self.reg_loss_weight * reg_loss
            + self.cls_loss_weight * conf_loss
            + self.vel_loss_weight * vel_loss
            + self.yaw_loss_weight * yaw_loss
        )
        winner_traj = x_pred[torch.arange(batch_size, device=x_pred.device), winner_idx]
        return {
            "loss": total_loss,
            "reg_loss": reg_loss,
            "conf_loss": conf_loss,
            "cover_loss": cover_loss,
            "vel_loss": vel_loss,
            "yaw_loss": yaw_loss,
            "winner_idx": winner_idx,
            "winner_traj_3d": self._convert_nd_to_3d(winner_traj.detach()),
            "awta_temperature": temperature.detach(),
            "cls_sample_valid_ratio": sample_valid.to(x_pred.dtype).mean().detach(),
        }

    def _training_forward(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate_training_target(gt_trajectory, context_tokens.shape[0])
        if self.num_modes > 1:
            if self.independent_modes:
                return self._training_forward_independent(context_tokens, status_emb, gt_trajectory, anchor_state)
            return self._training_forward_multimodal(context_tokens, status_emb, gt_trajectory, anchor_state)

        batch_size = gt_trajectory.shape[0]
        time_value = torch.rand(batch_size, device=gt_trajectory.device, dtype=gt_trajectory.dtype).clamp_min(1e-5)
        clean = gt_trajectory
        anchor = None
        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, batch_size, gt_trajectory.device).to(dtype=gt_trajectory.dtype)
            clean = torch.cat((anchor, clean), dim=1)
        x_t, _, _ = self._perturb(clean, time_value)
        if anchor is not None:
            x_t[:, :1] = anchor
        _, predicted = self.dit(x_t, time_value, context_tokens, status_emb)
        predicted = predicted.reshape(batch_size, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            predicted_future = predicted[:, 1:]
            target_future = clean[:, 1:]
        else:
            predicted_future, target_future = predicted, clean
        x_pred = predicted_future
        reg_loss = self._regression_loss_per_mode(predicted_future.unsqueeze(1), target_future.unsqueeze(1)).mean()
        _, vel_loss, yaw_loss = self._endpoint_losses(x_pred, gt_trajectory)
        conf_loss = predicted_future.sum() * 0.0
        cover_loss = predicted_future.sum() * 0.0
        loss = self.reg_loss_weight * reg_loss + self.vel_loss_weight * vel_loss + self.yaw_loss_weight * yaw_loss
        return {
            "loss": loss,
            "reg_loss": reg_loss,
            "conf_loss": conf_loss,
            "cover_loss": cover_loss,
            "vel_loss": vel_loss,
            "yaw_loss": yaw_loss,
        }

    def _training_forward_independent(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = gt_trajectory.shape[0]
        num_modes = self.num_modes
        gt_modes = gt_trajectory.unsqueeze(1).expand(-1, num_modes, -1, -1)
        clean = gt_modes.reshape(batch_size * num_modes, self.num_poses, self.traj_dim)
        anchor_bk = None
        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, batch_size, gt_trajectory.device).to(dtype=gt_trajectory.dtype)
            anchor_bk = (
                anchor.unsqueeze(1).expand(-1, num_modes, -1, -1).reshape(batch_size * num_modes, 1, self.traj_dim)
            )
            clean = torch.cat((anchor_bk, clean), dim=1)
        time_batch = torch.rand(batch_size, device=gt_trajectory.device, dtype=gt_trajectory.dtype).clamp_min(1e-5)
        time_value = time_batch.unsqueeze(1).expand(-1, num_modes).reshape(-1)
        x_t, _, _ = self._perturb(clean, time_value)
        if anchor_bk is not None:
            x_t[:, :1] = anchor_bk
        context_bk = (
            context_tokens.unsqueeze(1)
            .expand(-1, num_modes, -1, -1)
            .reshape(batch_size * num_modes, -1, self.hidden_dim)
        )
        status_bk = status_emb.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch_size * num_modes, self.hidden_dim)
        _, predicted = self.dit(x_t, time_value, context_bk, status_bk)
        predicted = predicted.reshape(batch_size * num_modes, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            predicted = predicted[:, 1:]
            target = clean[:, 1:]
        else:
            target = clean
        x_pred = predicted.reshape(batch_size, num_modes, self.num_poses, self.traj_dim)
        predicted = predicted.reshape(batch_size, num_modes, self.num_poses, self.traj_dim)
        target = target.reshape_as(predicted)
        cls_pred = self.confidence_head(x_pred.detach(), context_tokens) if self.confidence_head is not None else None
        return self._build_multimodal_loss_dict(predicted, target, x_pred, cls_pred, gt_trajectory)

    def _training_forward_multimodal(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        gt_trajectory: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = gt_trajectory.shape[0]
        num_modes = self.num_modes
        clean = gt_trajectory.unsqueeze(1).expand(-1, num_modes, -1, -1)
        anchor_k = None
        if self.use_anchor_frame:
            anchor = self._get_anchor(anchor_state, batch_size, gt_trajectory.device).to(dtype=gt_trajectory.dtype)
            anchor_k = anchor.unsqueeze(1).expand(-1, num_modes, -1, -1)
            clean = torch.cat((anchor_k, clean), dim=2)
        time_value = torch.rand(batch_size, device=gt_trajectory.device, dtype=gt_trajectory.dtype).clamp_min(1e-5)
        x_t, _, _ = self._perturb(clean, time_value)
        if anchor_k is not None:
            x_t[:, :, :1] = anchor_k
        cls_pred, predicted = self.dit(x_t, time_value, context_tokens, status_emb)
        predicted = predicted.reshape(batch_size, num_modes, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            predicted = predicted[:, :, 1:]
            target = clean[:, :, 1:]
        else:
            target = clean
        x_pred = predicted
        return self._build_multimodal_loss_dict(predicted, target, x_pred, cls_pred, gt_trajectory)

    def _initial_solver_input(
        self, B: int, K: int, device: torch.device, anchor_state: Optional[torch.Tensor], inference_noise
    ):
        future = self._resolve_inference_noise(inference_noise, B=B, K=K, device=device)
        anchor = self._get_anchor(anchor_state, B, device).to(dtype=future.dtype)
        if self._uses_batch_expansion:
            future_bk = future.reshape(B * K, self.num_poses, self.traj_dim)
            if self.use_anchor_frame:
                anchor_bk = anchor.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, 1, self.traj_dim)
                return torch.cat((anchor_bk, future_bk), dim=1).reshape(B * K, -1), anchor_bk
            return future_bk.reshape(B * K, -1), None
        if self.use_anchor_frame:
            anchor_k = anchor.unsqueeze(1).expand(-1, K, -1, -1)
            return torch.cat((anchor_k, future), dim=2).reshape(B, -1), anchor_k
        return future.reshape(B, -1), None

    def _sample_with_solver(
        self,
        x_T: torch.Tensor,
        context: torch.Tensor,
        status: torch.Tensor,
        anchor_info: Optional[torch.Tensor],
    ):
        solver_options = {}
        if anchor_info is not None:
            solver_options["correcting_xt_fn"] = lambda x_t, time_value, step: self._correct_anchor_xt(
                x_t, anchor_info
            )
        return dpm_sampler(
            self.dit,
            x_T,
            other_model_params={"cross_c": context, "status_emb": status},
            diffusion_steps=self.inference_steps,
            noise_schedule_params={"continuous_beta_0": self.sde._beta_min, "continuous_beta_1": self.sde._beta_max},
            dpm_solver_params=solver_options,
        )

    def _inference_forward(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.num_modes > 1:
            if self.independent_modes:
                return self._inference_forward_independent(context_tokens, status_emb, anchor_state, inference_noise)
            return self._inference_forward_multimodal(context_tokens, status_emb, anchor_state, inference_noise)
        batch_size = context_tokens.shape[0]
        num_samples = self.num_samples
        context_bk = (
            context_tokens.unsqueeze(1)
            .expand(-1, num_samples, -1, -1)
            .reshape(batch_size * num_samples, -1, self.hidden_dim)
        )
        status_bk = (
            status_emb.unsqueeze(1).expand(-1, num_samples, -1).reshape(batch_size * num_samples, self.hidden_dim)
        )
        x_T, anchor_info = self._initial_solver_input(
            batch_size, num_samples, context_tokens.device, anchor_state, inference_noise
        )
        _, x_0 = self._sample_with_solver(x_T, context_bk, status_bk, anchor_info)
        trajectories = x_0.reshape(batch_size, num_samples, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            trajectories = trajectories[:, :, 1:]
        trajectories = self._convert_nd_to_3d(trajectories)
        confidences = torch.full(
            (batch_size, num_samples),
            1.0 / num_samples,
            device=trajectories.device,
            dtype=trajectories.dtype,
        )
        return {"trajectories": trajectories, "confidences": confidences}

    def _inference_forward_independent(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = context_tokens.shape[0]
        num_modes = self.num_modes
        context_bk = (
            context_tokens.unsqueeze(1)
            .expand(-1, num_modes, -1, -1)
            .reshape(batch_size * num_modes, -1, self.hidden_dim)
        )
        status_bk = status_emb.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch_size * num_modes, self.hidden_dim)
        x_T, anchor_info = self._initial_solver_input(
            batch_size, num_modes, context_tokens.device, anchor_state, inference_noise
        )
        _, x_0 = self._sample_with_solver(x_T, context_bk, status_bk, anchor_info)
        trajectories_nd = x_0.reshape(batch_size, num_modes, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            trajectories_nd = trajectories_nd[:, :, 1:]
        trajectories = self._convert_nd_to_3d(trajectories_nd)
        if self.confidence_head is None:
            confidences = torch.full(
                (batch_size, num_modes),
                1.0 / num_modes,
                device=trajectories.device,
                dtype=trajectories.dtype,
            )
        else:
            confidences = torch.softmax(self.confidence_head(trajectories_nd, context_tokens), dim=1)
        return {"trajectories": trajectories, "confidences": confidences}

    def _inference_forward_multimodal(
        self,
        context_tokens: torch.Tensor,
        status_emb: torch.Tensor,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = context_tokens.shape[0]
        num_modes = self.num_modes
        x_T, anchor_info = self._initial_solver_input(
            batch_size, num_modes, context_tokens.device, anchor_state, inference_noise
        )
        classification, x_0 = self._sample_with_solver(x_T, context_tokens, status_emb, anchor_info)
        trajectories = x_0.reshape(batch_size, num_modes, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            trajectories = trajectories[:, :, 1:]
        trajectories = self._convert_nd_to_3d(trajectories)
        if classification is None:
            confidences = torch.full(
                (batch_size, num_modes),
                1.0 / num_modes,
                device=trajectories.device,
                dtype=trajectories.dtype,
            )
        else:
            confidences = torch.softmax(classification, dim=1)
        return {"trajectories": trajectories, "confidences": confidences}

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
        context_tokens = self._prepare_context(z_ar, z_context, z_observed, action_history)
        status_emb = self._prepare_status(status_feature)
        if context_tokens.shape[0] != status_emb.shape[0]:
            raise ValueError("status_feature batch size must match z_ar")
        if self.use_anchor_frame and anchor_state is None:
            raise ValueError("anchor_state is required when use_anchor_frame=True")
        if gt_trajectory is not None and self.training:
            if inference_noise is not None:
                raise ValueError("inference_noise is only valid during planner inference")
            return self._training_forward(context_tokens, status_emb, gt_trajectory, anchor_state)
        return self._inference_forward(context_tokens, status_emb, anchor_state, inference_noise)

    def init_interleaved_inference_state(
        self,
        status_feature: torch.Tensor,
        total_condition_updates: int,
        anchor_state: Optional[torch.Tensor] = None,
        inference_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if total_condition_updates <= 0:
            raise ValueError("total_condition_updates must be positive")
        parameter = next(self.parameters())
        self._validate_retained_solver_tensor(parameter, "planner parameters")
        if isinstance(inference_noise, torch.Tensor):
            self._validate_retained_solver_tensor(inference_noise, "inference_noise")
        status_emb = self._prepare_status(status_feature)
        batch_size = status_feature.shape[0]
        num_modes = self.num_modes if self.num_modes > 1 else self.num_samples
        if self.use_anchor_frame and anchor_state is None:
            raise ValueError("anchor_state is required when use_anchor_frame=True")
        x_t, anchor_info = self._initial_solver_input(
            batch_size, num_modes, status_feature.device, anchor_state, inference_noise
        )
        if self._uses_batch_expansion:
            status_k = (
                status_emb.unsqueeze(1).expand(-1, num_modes, -1).reshape(batch_size * num_modes, self.hidden_dim)
            )
        else:
            status_k = status_emb
        noise_schedule = dpm.NoiseScheduleVP(
            schedule="linear",
            continuous_beta_0=self.sde._beta_min,
            continuous_beta_1=self.sde._beta_max,
        )
        placeholder = dpm.DPM_Solver(lambda x, t: (None, x), noise_schedule, algorithm_type="dpmsolver++")
        timesteps = placeholder.get_time_steps(
            "logSNR",
            noise_schedule.T,
            1.0 / noise_schedule.total_N,
            self.inference_steps,
            status_feature.device,
        )
        return {
            "x_t": x_t,
            "status_k": status_k,
            "noise_schedule": noise_schedule,
            "timesteps": timesteps,
            "batch_size": batch_size,
            "total_condition_updates": total_condition_updates,
            "completed_condition_updates": 0,
            "completed_sampling_steps": 0,
            "anchor_info": anchor_info,
        }

    def _run_interleaved_solver_step(
        self,
        x_t: torch.Tensor,
        context_k: torch.Tensor,
        status_k: torch.Tensor,
        noise_schedule: dpm.NoiseScheduleVP,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        model_fn = dpm.model_wrapper(
            self.dit,
            noise_schedule,
            model_type=self.dit.model_type,
            model_kwargs={"cross_c": context_k, "status_emb": status_k},
        )
        solver = dpm.DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
        _, model_s = solver.model_fn(x_t, s)
        return solver.dpm_solver_first_update(x_t, s, t, model_s=model_s)

    @staticmethod
    def _validate_retained_solver_tensor(value: torch.Tensor, name: str) -> None:
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor")
        if value.dtype not in _SUPPORTED_RETAINED_SOLVER_DTYPES:
            raise TypeError(f"{name} dtype must be float32 or float64 for the retained solver, got {value.dtype}")

    def _interleaved_target_sampling_steps(
        self, completed_condition_updates: int, total_condition_updates: int
    ) -> int:
        if total_condition_updates <= 0:
            raise ValueError("total_condition_updates must be positive")
        if completed_condition_updates < 0 or completed_condition_updates > total_condition_updates:
            raise ValueError("completed_condition_updates is outside the valid range")
        return min(
            self.inference_steps,
            (completed_condition_updates * self.inference_steps) // total_condition_updates,
        )

    def _interleaved_context(self, state, z_ar, z_context, z_observed, action_history):
        context = self._prepare_context(z_ar, z_context, z_observed, action_history)
        batch_size = state["batch_size"]
        if context.shape[0] != batch_size:
            raise ValueError("interleaved z_ar batch size changed")
        if self._uses_batch_expansion:
            num_modes = self.num_modes if self.num_modes > 1 else self.num_samples
            context = (
                context.unsqueeze(1).expand(-1, num_modes, -1, -1).reshape(batch_size * num_modes, -1, self.hidden_dim)
            )
        return context

    def advance_interleaved_inference(
        self,
        state: Dict[str, torch.Tensor],
        z_ar: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
        z_observed: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate_retained_solver_tensor(next(self.parameters()), "planner parameters")
        self._validate_retained_solver_tensor(state["x_t"], "interleaved state x_t")
        completed_updates = state["completed_condition_updates"]
        total_updates = state["total_condition_updates"]
        if completed_updates >= total_updates:
            raise ValueError("all interleaved condition updates are already complete")
        context_k = self._interleaved_context(state, z_ar, z_context, z_observed, action_history)
        completed_updates += 1
        target_steps = self._interleaved_target_sampling_steps(completed_updates, total_updates)
        while state["completed_sampling_steps"] < target_steps:
            step = state["completed_sampling_steps"]
            state["x_t"] = self._run_interleaved_solver_step(
                state["x_t"],
                context_k,
                state["status_k"],
                state["noise_schedule"],
                state["timesteps"][step],
                state["timesteps"][step + 1],
            )
            if state["anchor_info"] is not None:
                state["x_t"] = self._correct_anchor_xt(state["x_t"], state["anchor_info"])
            state["completed_sampling_steps"] += 1
        state["completed_condition_updates"] = completed_updates
        return state

    def finalize_interleaved_inference(
        self,
        state: Dict[str, torch.Tensor],
        z_ar: torch.Tensor,
        z_context: Optional[torch.Tensor] = None,
        z_observed: Optional[torch.Tensor] = None,
        action_history: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate_retained_solver_tensor(next(self.parameters()), "planner parameters")
        self._validate_retained_solver_tensor(state["x_t"], "interleaved state x_t")
        context_k = self._interleaved_context(state, z_ar, z_context, z_observed, action_history)
        while state["completed_sampling_steps"] < self.inference_steps:
            step = state["completed_sampling_steps"]
            state["x_t"] = self._run_interleaved_solver_step(
                state["x_t"],
                context_k,
                state["status_k"],
                state["noise_schedule"],
                state["timesteps"][step],
                state["timesteps"][step + 1],
            )
            if state["anchor_info"] is not None:
                state["x_t"] = self._correct_anchor_xt(state["x_t"], state["anchor_info"])
            state["completed_sampling_steps"] += 1
        state["completed_condition_updates"] = state["total_condition_updates"]

        batch_size = state["batch_size"]
        num_modes = self.num_modes if self.num_modes > 1 else self.num_samples
        final_time = state["timesteps"][-1].expand(state["x_t"].shape[0])
        classification, clean_prediction = self.dit(
            state["x_t"],
            final_time,
            context_k,
            state["status_k"],
        )
        x_0 = clean_prediction.reshape(batch_size, num_modes, self.total_frames, self.traj_dim)
        if self.use_anchor_frame:
            x_0 = x_0[:, :, 1:]
        trajectories = self._convert_nd_to_3d(x_0)
        if self.independent_modes and self.confidence_head is not None:
            unexpanded_context = self._prepare_context(z_ar, z_context, z_observed, action_history)
            confidences = torch.softmax(self.confidence_head(x_0, unexpanded_context), dim=1)
        elif not self._uses_batch_expansion and self.num_modes > 1:
            confidences = (
                torch.softmax(classification, dim=1)
                if classification is not None
                else torch.full_like(trajectories[:, :, 0, 0], 1.0 / num_modes)
            )
        else:
            confidences = torch.full_like(trajectories[:, :, 0, 0], 1.0 / num_modes)
        return {"trajectories": trajectories, "confidences": confidences}
