"""Latent-token DiT predictor trained with flow matching."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_cowa_world_model.models.diffusion_head import TimestepEmbedder

LATENT_DIT_OBJECTIVES = ("flow_matching", "x0_prediction")


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LatentDiTBlock(nn.Module):
    """Transformer block with adaLN and cross-attention to observed context tokens."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_cross = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 9 * hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        timestep_embed: torch.Tensor,
        cond_key_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(timestep_embed).chunk(9, dim=1)
        )

        x_sa = _modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.self_attn(x_sa, x_sa, x_sa, attn_mask=self_attn_mask, need_weights=False)
        x = x + gate_sa.unsqueeze(1) * attn_out

        x_ca = _modulate(self.norm_cross(x), shift_ca, scale_ca)
        cross_out, _ = self.cross_attn(
            x_ca,
            cond_tokens,
            cond_tokens,
            key_padding_mask=cond_key_padding_mask,
            need_weights=False,
        )
        x = x + gate_ca.unsqueeze(1) * cross_out

        x_mlp = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mlp)
        return x


class LatentDiTPredictor(nn.Module):
    """Predict future V-JEPA latent tokens with parallel flow-matching denoising.

    The model does not generate pixels. It denoises the full future token sequence
    in parallel and returns clean future tokens for downstream planner input.
    """

    supports_metadata_condition_mask = True

    def __init__(
        self,
        embed_dim: int,
        tokens_per_frame: int,
        num_future_steps: int,
        action_dim: int = 3,
        state_dim: int = 7,
        extrinsics_dim: int = 7,
        hidden_dim: int = 512,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        x0_loss_weight: float = 0.0,
        bottleneck_dim: Optional[int] = None,
        max_steps: int = 128,
        conditioning_mode: str = "mean",
        use_anchor_frame: bool = False,
        objective: str = "flow_matching",
        joint_action_enabled: bool = False,
        joint_action_dim: int = 3,
        joint_action_state_dim: Optional[int] = None,
        joint_action_inference_noise_mode: str = "shared",
        joint_video_final_noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.tokens_per_frame = int(tokens_per_frame)
        self.num_future_steps = int(num_future_steps)
        self.hidden_dim = int(hidden_dim)
        self.num_future_tokens = self.tokens_per_frame * self.num_future_steps
        self.use_anchor_frame = bool(use_anchor_frame)
        self.num_anchor_tokens = self.tokens_per_frame if self.use_anchor_frame else 0
        self.num_denoise_tokens = self.num_future_tokens + self.num_anchor_tokens
        self.x0_loss_weight = float(x0_loss_weight)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.extrinsics_dim = int(extrinsics_dim)
        self.max_steps = int(max_steps)
        self.conditioning_mode = str(conditioning_mode).lower()
        self.objective = str(objective).lower()
        self.joint_action_enabled = bool(joint_action_enabled)
        self.joint_action_dim = int(joint_action_dim)
        self.joint_action_state_dim = self.state_dim if joint_action_state_dim is None else int(joint_action_state_dim)
        self.joint_action_inference_noise_mode = str(joint_action_inference_noise_mode).lower()
        self.joint_video_final_noise = float(joint_video_final_noise)

        if self.num_future_tokens <= 0:
            raise ValueError("num_future_steps and tokens_per_frame must define at least one future token")
        if self.objective not in LATENT_DIT_OBJECTIVES:
            raise ValueError(f"objective must be one of {list(LATENT_DIT_OBJECTIVES)}, " f"got {objective!r}")
        if self.joint_action_inference_noise_mode not in {"shared", "decoupled"}:
            raise ValueError("joint_action_inference_noise_mode must be 'shared' or 'decoupled'")
        if not 0.0 <= self.joint_video_final_noise < 1.0:
            raise ValueError(f"joint_video_final_noise must be in [0.0, 1.0), got {self.joint_video_final_noise}")
        if self.joint_video_final_noise > 0.0 and self.joint_action_inference_noise_mode != "decoupled":
            raise ValueError("joint_video_final_noise > 0 requires joint_action_inference_noise_mode='decoupled'")
        if self.conditioning_mode not in {"mean", "temporal_aux_tokens"}:
            raise ValueError(
                "conditioning_mode must be one of ['mean', 'temporal_aux_tokens'], " f"got {conditioning_mode!r}"
            )

        self.bottleneck_dim = None if bottleneck_dim is None or int(bottleneck_dim) <= 0 else int(bottleneck_dim)
        predictor_input_dim = self.bottleneck_dim or self.embed_dim
        if self.bottleneck_dim is None:
            self.future_bottleneck = nn.Identity()
            self.context_bottleneck = nn.Identity()
            self.future_unbottleneck = nn.Identity()
        else:
            self.future_bottleneck = nn.Linear(self.embed_dim, self.bottleneck_dim)
            self.context_bottleneck = nn.Linear(self.embed_dim, self.bottleneck_dim)
            self.future_unbottleneck = nn.Linear(self.bottleneck_dim, self.embed_dim)

        self.input_proj = nn.Linear(predictor_input_dim, int(hidden_dim))
        self.context_proj = nn.Linear(predictor_input_dim, int(hidden_dim))
        self.output_proj = nn.Linear(int(hidden_dim), predictor_input_dim)
        if self.joint_action_enabled:
            self.action_input_proj = nn.Linear(self.joint_action_dim, int(hidden_dim))
            self.action_state_proj = nn.Linear(self.joint_action_state_dim, int(hidden_dim))
            self.action_t_embedder = TimestepEmbedder(int(hidden_dim))
            self.action_pos_embed = nn.Parameter(torch.zeros(1, self.num_future_steps, int(hidden_dim)))
            self.action_output_proj = nn.Linear(int(hidden_dim), self.joint_action_dim)
        else:
            self.action_input_proj = None
            self.action_state_proj = None
            self.action_t_embedder = None
            self.action_pos_embed = None
            self.action_output_proj = None

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_denoise_tokens, int(hidden_dim)))
        self.t_embedder = TimestepEmbedder(int(hidden_dim))
        self.side_embed = nn.Sequential(
            nn.Linear(self.action_dim + self.state_dim + self.extrinsics_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.future_condition_type_embed = nn.Parameter(torch.zeros(1, 1, int(hidden_dim)))
        if self.conditioning_mode == "temporal_aux_tokens":
            self.aux_token_embed = nn.Sequential(
                nn.Linear(self.action_dim + self.state_dim + self.extrinsics_dim, int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
            )
            self.aux_pos_embed = nn.Parameter(torch.zeros(1, self.max_steps, int(hidden_dim)))
        self.blocks = nn.ModuleList(
            [LatentDiTBlock(int(hidden_dim), int(num_heads), float(dropout)) for _ in range(int(depth))]
        )
        self.final_norm = nn.LayerNorm(int(hidden_dim), elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(int(hidden_dim), 2 * int(hidden_dim)))
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if hasattr(self, "aux_pos_embed"):
            nn.init.trunc_normal_(self.aux_pos_embed, std=0.02)
        if self.action_pos_embed is not None:
            nn.init.trunc_normal_(self.action_pos_embed, std=0.02)
        nn.init.zeros_(self.future_condition_type_embed)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.context_proj.weight)
        nn.init.zeros_(self.context_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        if self.action_input_proj is not None:
            nn.init.xavier_uniform_(self.action_input_proj.weight)
            nn.init.zeros_(self.action_input_proj.bias)
        if self.action_state_proj is not None:
            nn.init.xavier_uniform_(self.action_state_proj.weight)
            nn.init.zeros_(self.action_state_proj.bias)
        if self.action_output_proj is not None:
            nn.init.xavier_uniform_(self.action_output_proj.weight)
            nn.init.zeros_(self.action_output_proj.bias)
        if self.bottleneck_dim is not None:
            nn.init.xavier_uniform_(self.future_bottleneck.weight)
            nn.init.zeros_(self.future_bottleneck.bias)
            nn.init.xavier_uniform_(self.context_bottleneck.weight)
            nn.init.zeros_(self.context_bottleneck.bias)
            nn.init.xavier_uniform_(self.future_unbottleneck.weight)
            nn.init.zeros_(self.future_unbottleneck.bias)
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)

    def forward_train(
        self,
        observed_tokens: torch.Tensor,
        target_future_tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        anchor_tokens: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run one latent-DiT training step.

        Parameters
        ----------
        observed_tokens : [B, O*P, D]
            Clean observed context tokens.
        target_future_tokens : [B, F*P, D]
            Target encoder future tokens used as the clean endpoint.

        Returns
        -------
        dict
            ``loss``, ``flow_loss``, ``x0_loss`` and ``x0_pred``.
        """
        if target_future_tokens.shape[1] != self.num_future_tokens:
            raise ValueError(
                f"target_future_tokens length {target_future_tokens.shape[1]} does not match "
                f"configured num_future_tokens {self.num_future_tokens}"
            )
        batch_size = target_future_tokens.shape[0]
        t = self._sample_timestep(batch_size, target_future_tokens.device)
        noise = torch.randn_like(target_future_tokens)
        t_expand = t[:, None, None]
        x_t = (1.0 - t_expand) * noise + t_expand * target_future_tokens
        velocity_target = target_future_tokens - noise  # 一阶导

        model_pred = self.forward(
            noisy_future_tokens=x_t,
            timesteps=t,
            observed_tokens=observed_tokens,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            anchor_tokens=anchor_tokens,
        )

        if self.objective == "flow_matching":
            velocity_pred = model_pred
            x0_pred = x_t + (1.0 - t_expand) * velocity_pred  # 求target==x0_pred
            flow_loss = F.mse_loss(velocity_pred, velocity_target)
            x0_loss = F.mse_loss(x0_pred, target_future_tokens)
            loss = flow_loss + self.x0_loss_weight * x0_loss
        else:
            x0_pred = model_pred
            velocity_pred = (x0_pred - x_t) / (1.0 - t_expand).clamp_min(1e-5)
            flow_loss = F.mse_loss(velocity_pred, velocity_target)
            x0_loss = F.mse_loss(x0_pred, target_future_tokens)
            loss = x0_loss
        objective_loss = loss

        return {
            "loss": loss,
            "objective": self.objective,
            "objective_loss": objective_loss,
            "flow_loss": flow_loss,
            "x0_loss": x0_loss,
            "x0_pred": x0_pred,
            "velocity_pred": velocity_pred,
            "velocity_target": velocity_target,
        }

    def forward(
        self,
        noisy_future_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        observed_tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        condition_cache: Optional[Dict[str, torch.Tensor]] = None,
        anchor_tokens: Optional[torch.Tensor] = None,
        future_token_indices: Optional[torch.Tensor] = None,
        known_future_tokens: Optional[torch.Tensor] = None,
        known_future_token_indices: Optional[torch.Tensor] = None,
        metadata_condition_mask: Optional[torch.Tensor] = None,
        noisy_future_actions: Optional[torch.Tensor] = None,
        action_timesteps: Optional[torch.Tensor] = None,
        action_state_tokens: Optional[torch.Tensor] = None,
        target_future_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noisy_future_actions is not None or action_timesteps is not None or action_state_tokens is not None:
            if noisy_future_actions is None or action_timesteps is None or action_state_tokens is None:
                raise ValueError(
                    "noisy_future_actions, action_timesteps, and action_state_tokens must be provided together"
                )
            return self.forward_joint(
                noisy_future_tokens=noisy_future_tokens,
                timesteps=timesteps,
                observed_tokens=observed_tokens,
                noisy_future_actions=noisy_future_actions,
                action_timesteps=action_timesteps,
                action_state_tokens=action_state_tokens,
                target_future_actions=target_future_actions,
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                condition_cache=condition_cache,
                anchor_tokens=anchor_tokens,
                future_token_indices=future_token_indices,
                known_future_tokens=known_future_tokens,
                known_future_token_indices=known_future_token_indices,
                metadata_condition_mask=metadata_condition_mask,
            )
        if noisy_future_tokens.ndim != 3:
            raise ValueError("noisy_future_tokens must have shape [B, F*P, D]")
        if condition_cache is not None and (known_future_tokens is not None or known_future_token_indices is not None):
            raise ValueError("known_future_tokens must be included when building condition_cache, not passed with it")
        resolved_future_indices = self._resolve_future_token_indices(
            future_token_indices,
            active_tokens=int(noisy_future_tokens.shape[1]),
            device=noisy_future_tokens.device,
        )

        future_tokens = self.future_bottleneck(noisy_future_tokens)
        denoise_tokens = future_tokens
        future_pos_embed = self._future_position_embed(resolved_future_indices, noisy_future_tokens)
        denoise_pos_embed = future_pos_embed
        if self.use_anchor_frame:
            clean_anchor = self._resolve_anchor_tokens(
                observed_tokens=observed_tokens,
                anchor_tokens=anchor_tokens,
                reference=noisy_future_tokens,
            )
            denoise_tokens = torch.cat([self.future_bottleneck(clean_anchor), future_tokens], dim=1)
            anchor_pos_embed = self.pos_embed[:, : self.num_anchor_tokens].to(
                device=noisy_future_tokens.device,
                dtype=noisy_future_tokens.dtype,
            )
            denoise_pos_embed = torch.cat([anchor_pos_embed, future_pos_embed], dim=1)

        x = self.input_proj(denoise_tokens) + denoise_pos_embed
        if condition_cache is None:
            condition_cache = self._build_condition_cache(
                observed_tokens=observed_tokens,
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                known_future_tokens=known_future_tokens,
                known_future_token_indices=known_future_token_indices,
                active_future_token_indices=resolved_future_indices,
                metadata_condition_mask=metadata_condition_mask,
            )
        cond_tokens = condition_cache["cond_tokens"].to(device=x.device, dtype=x.dtype)
        cond_key_padding_mask = condition_cache.get("cond_key_padding_mask")
        if cond_key_padding_mask is not None:
            cond_key_padding_mask = cond_key_padding_mask.to(device=x.device, dtype=torch.bool)
        timestep_embed = self.t_embedder(timesteps)
        side_condition = condition_cache["side_condition"].to(device=timestep_embed.device, dtype=timestep_embed.dtype)
        timestep_embed = timestep_embed + side_condition

        for block in self.blocks:
            x = block(x, cond_tokens, timestep_embed, cond_key_padding_mask=cond_key_padding_mask)

        shift, scale = self.final_modulation(timestep_embed).chunk(2, dim=1)
        x = _modulate(self.final_norm(x), shift, scale)
        velocity = self.output_proj(x)
        if self.use_anchor_frame:
            velocity = velocity[:, self.num_anchor_tokens :]
        return self.future_unbottleneck(velocity)

    def forward_joint(
        self,
        *,
        noisy_future_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        observed_tokens: torch.Tensor,
        noisy_future_actions: torch.Tensor,
        action_timesteps: torch.Tensor,
        action_state_tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        condition_cache: Optional[Dict[str, torch.Tensor]] = None,
        anchor_tokens: Optional[torch.Tensor] = None,
        future_token_indices: Optional[torch.Tensor] = None,
        known_future_tokens: Optional[torch.Tensor] = None,
        known_future_token_indices: Optional[torch.Tensor] = None,
        metadata_condition_mask: Optional[torch.Tensor] = None,
        target_future_actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict world latents and normalized ego actions in one DiT sequence."""
        del target_future_actions
        if not self.joint_action_enabled:
            raise ValueError("forward_joint requires joint_action_enabled=True")
        if noisy_future_tokens.ndim != 3:
            raise ValueError("noisy_future_tokens must have shape [B, active_F*P, D]")
        resolved_future_indices = self._resolve_future_token_indices(
            future_token_indices,
            active_tokens=int(noisy_future_tokens.shape[1]),
            device=noisy_future_tokens.device,
        )
        if int(noisy_future_tokens.shape[1]) % self.tokens_per_frame != 0:
            raise ValueError(
                f"noisy_future_tokens length {noisy_future_tokens.shape[1]} must be divisible by "
                f"tokens_per_frame={self.tokens_per_frame}"
            )
        active_steps = int(noisy_future_tokens.shape[1]) // self.tokens_per_frame
        active_start_token = int(resolved_future_indices[0].item())
        if active_start_token % self.tokens_per_frame != 0:
            raise ValueError(
                f"joint action future_token_indices must start on a frame boundary, got token {active_start_token}"
            )
        expected_indices = torch.arange(
            active_start_token,
            active_start_token + int(noisy_future_tokens.shape[1]),
            device=resolved_future_indices.device,
            dtype=resolved_future_indices.dtype,
        )
        if not bool(torch.equal(resolved_future_indices, expected_indices)):
            raise ValueError("joint action future_token_indices must describe one contiguous frame-aligned window")
        active_start_step = active_start_token // self.tokens_per_frame
        if active_start_step + active_steps > self.num_future_steps:
            raise ValueError(
                "joint action active future window exceeds configured horizon: "
                f"start={active_start_step}, steps={active_steps}, num_future_steps={self.num_future_steps}"
            )
        if noisy_future_actions.ndim != 3 or noisy_future_actions.shape[-1] != self.joint_action_dim:
            raise ValueError(
                f"noisy_future_actions must have shape [B, H, {self.joint_action_dim}], "
                f"got {tuple(noisy_future_actions.shape)}"
            )
        if noisy_future_actions.shape[1] != active_steps:
            raise ValueError(
                f"noisy_future_actions horizon {noisy_future_actions.shape[1]} must match active "
                f"future steps={active_steps}"
            )
        if action_timesteps.shape != noisy_future_actions.shape[:2]:
            raise ValueError(
                f"action_timesteps must have shape {tuple(noisy_future_actions.shape[:2])}, "
                f"got {tuple(action_timesteps.shape)}"
            )
        if action_state_tokens.shape[:2] != noisy_future_actions.shape[:2]:
            raise ValueError(
                "action_state_tokens must align with noisy_future_actions horizon, got "
                f"{tuple(action_state_tokens.shape)} vs {tuple(noisy_future_actions.shape)}"
            )
        if action_state_tokens.shape[-1] != self.joint_action_state_dim:
            raise ValueError(
                f"action_state_tokens dim {action_state_tokens.shape[-1]} must match "
                f"joint_action_state_dim={self.joint_action_state_dim}"
            )

        future_tokens = self.future_bottleneck(noisy_future_tokens)
        denoise_tokens = future_tokens
        future_pos_embed = self._future_position_embed(resolved_future_indices, noisy_future_tokens)
        denoise_pos_embed = future_pos_embed
        if self.use_anchor_frame:
            clean_anchor = self._resolve_anchor_tokens(
                observed_tokens=observed_tokens,
                anchor_tokens=anchor_tokens,
                reference=noisy_future_tokens,
            )
            denoise_tokens = torch.cat([self.future_bottleneck(clean_anchor), future_tokens], dim=1)
            anchor_pos_embed = self.pos_embed[:, : self.num_anchor_tokens].to(
                device=noisy_future_tokens.device,
                dtype=noisy_future_tokens.dtype,
            )
            denoise_pos_embed = torch.cat([anchor_pos_embed, future_pos_embed], dim=1)

        x_world = self.input_proj(denoise_tokens) + denoise_pos_embed
        x_action = self.action_input_proj(noisy_future_actions)
        x_action = x_action + self.action_t_embedder(action_timesteps.reshape(-1)).reshape(
            noisy_future_actions.shape[0], noisy_future_actions.shape[1], -1
        )
        x_action = x_action + self.action_pos_embed[:, active_start_step : active_start_step + active_steps].to(
            device=x_action.device,
            dtype=x_action.dtype,
        )
        x_state = self.action_state_proj(action_state_tokens.to(device=x_action.device, dtype=x_action.dtype))
        x = torch.cat([x_world, x_action, x_state], dim=1)

        if condition_cache is None:
            condition_cache = self._build_condition_cache(
                observed_tokens=observed_tokens,
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                known_future_tokens=known_future_tokens,
                known_future_token_indices=known_future_token_indices,
                active_future_token_indices=resolved_future_indices,
                metadata_condition_mask=metadata_condition_mask,
            )
        cond_tokens = condition_cache["cond_tokens"].to(device=x.device, dtype=x.dtype)
        cond_key_padding_mask = condition_cache.get("cond_key_padding_mask")
        if cond_key_padding_mask is not None:
            cond_key_padding_mask = cond_key_padding_mask.to(device=x.device, dtype=torch.bool)
        timestep_embed = self.t_embedder(timesteps)
        side_condition = condition_cache["side_condition"].to(device=timestep_embed.device, dtype=timestep_embed.dtype)
        timestep_embed = timestep_embed + side_condition

        joint_attn_mask = self._build_joint_action_attn_mask(
            device=x.device,
            dtype=torch.bool,
            active_future_tokens=int(noisy_future_tokens.shape[1]),
            include_anchor=self.use_anchor_frame,
        )
        for block in self.blocks:
            x = block(
                x,
                cond_tokens,
                timestep_embed,
                cond_key_padding_mask=cond_key_padding_mask,
                self_attn_mask=joint_attn_mask,
            )

        shift, scale = self.final_modulation(timestep_embed).chunk(2, dim=1)
        x = _modulate(self.final_norm(x), shift, scale)
        denoise_len = x_world.shape[1]
        world_hidden = x[:, :denoise_len]
        if self.use_anchor_frame:
            world_hidden = world_hidden[:, self.num_anchor_tokens :]
        action_start = denoise_len
        action_end = action_start + active_steps
        world_pred = self.future_unbottleneck(self.output_proj(world_hidden))
        action_pred = self.action_output_proj(x[:, action_start:action_end])
        return {"world_pred": world_pred, "action_pred": action_pred}

    def _build_joint_action_attn_mask(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.bool,
        active_future_tokens: Optional[int] = None,
        include_anchor: Optional[bool] = None,
    ) -> torch.Tensor:
        if not self.joint_action_enabled:
            raise ValueError("_build_joint_action_attn_mask requires joint_action_enabled=True")
        active_future_tokens = self.num_future_tokens if active_future_tokens is None else int(active_future_tokens)
        if active_future_tokens % self.tokens_per_frame != 0:
            raise ValueError(
                f"active_future_tokens={active_future_tokens} must be divisible by "
                f"tokens_per_frame={self.tokens_per_frame}"
            )
        active_steps = active_future_tokens // self.tokens_per_frame
        if active_steps <= 0 or active_steps > self.num_future_steps:
            raise ValueError(
                "joint action attention mask active_steps must be within configured future horizon: "
                f"active_steps={active_steps}, num_future_steps={self.num_future_steps}"
            )
        include_anchor = self.use_anchor_frame if include_anchor is None else bool(include_anchor)
        anchor_tokens = self.num_anchor_tokens if include_anchor else 0
        action_start = anchor_tokens + active_future_tokens
        state_start = action_start + active_steps
        total = state_start + active_steps
        mask = torch.ones(total, total, device=device, dtype=dtype)
        if anchor_tokens > 0:
            mask[:anchor_tokens, :anchor_tokens] = False
        for step in range(active_steps):
            lat_start = anchor_tokens + step * self.tokens_per_frame
            lat_end = lat_start + self.tokens_per_frame
            prev_lat_end = lat_end
            action_idx = action_start + step
            state_idx = state_start + step

            for query in range(lat_start, lat_end):
                if anchor_tokens > 0:
                    mask[query, :anchor_tokens] = False
                mask[query, anchor_tokens:prev_lat_end] = False
                mask[query, action_idx] = False
                mask[query, state_idx] = False

            if anchor_tokens > 0:
                mask[action_idx, :anchor_tokens] = False
            mask[action_idx, anchor_tokens:prev_lat_end] = False
            mask[action_idx, action_idx] = False
            mask[action_idx, state_idx] = False

            mask[state_idx, state_idx] = False
        return mask

    @torch.no_grad()
    def sample(
        self,
        observed_tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        num_inference_steps: int = 8,
        sampler_type: str = "heun",
        schedule_type: str = "cosine",
        temperature: float = 1.0,
        return_diagnostics: bool = False,
        anchor_tokens: Optional[torch.Tensor] = None,
        metadata_condition_mask: Optional[torch.Tensor] = None,
        metadata_guidance_scale: float = 1.0,
        future_token_indices: Optional[torch.Tensor] = None,
        known_future_tokens: Optional[torch.Tensor] = None,
        known_future_token_indices: Optional[torch.Tensor] = None,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Dict[str, object]:
        """Sample clean future tokens from noise."""
        steps = max(1, int(num_inference_steps))
        sampler_type = str(sampler_type).lower()
        if sampler_type not in {"euler", "heun"}:
            raise ValueError(f"sampler_type must be one of ['euler', 'heun'], got {sampler_type!r}")
        if self.objective == "x0_prediction" and sampler_type != "euler":
            raise ValueError("LatentDiT objective='x0_prediction' currently requires sampler_type='euler'")

        timesteps, deltas = self._build_sampling_schedule(
            steps=steps,
            schedule_type=schedule_type,
            device=observed_tokens.device,
        )
        sample_token_count = self.num_future_tokens
        if future_token_indices is not None:
            if not torch.is_tensor(future_token_indices):
                raise TypeError("future_token_indices must be a torch.Tensor when provided")
            sample_token_count = int(future_token_indices.numel())
        resolved_future_indices = self._resolve_future_token_indices(
            future_token_indices,
            active_tokens=sample_token_count,
            device=observed_tokens.device,
        )
        forward_future_indices = resolved_future_indices if future_token_indices is not None else None
        expected_noise_shape = (observed_tokens.shape[0], sample_token_count, self.embed_dim)
        if initial_noise is None:
            x = torch.randn(
                expected_noise_shape,
                device=observed_tokens.device,
                dtype=observed_tokens.dtype,
            )
        else:
            if not torch.is_tensor(initial_noise):
                raise TypeError("initial_noise must be a torch.Tensor when provided")
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_noise_shape}, got {tuple(initial_noise.shape)}"
                )
            if initial_noise.device != observed_tokens.device:
                raise ValueError(
                    "initial_noise device must match observed_tokens: "
                    f"{initial_noise.device} != {observed_tokens.device}"
                )
            if initial_noise.dtype != observed_tokens.dtype:
                raise ValueError(
                    "initial_noise dtype must match observed_tokens: "
                    f"{initial_noise.dtype} != {observed_tokens.dtype}"
                )
            x = initial_noise
        x = x * float(temperature)
        metadata_condition_mask = self._resolve_metadata_condition_mask(
            metadata_condition_mask,
            batch_size=observed_tokens.shape[0],
            device=observed_tokens.device,
        )
        metadata_guidance_scale = float(metadata_guidance_scale)
        has_conditioned_samples = bool(metadata_condition_mask.any().item())
        needs_unconditioned = (
            bool((~metadata_condition_mask).any().item()) or abs(metadata_guidance_scale - 1.0) > 1e-8
        )
        condition_cache = (
            self._build_condition_cache(
                observed_tokens=observed_tokens,
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                known_future_tokens=known_future_tokens,
                known_future_token_indices=known_future_token_indices,
                active_future_token_indices=resolved_future_indices,
            )
            if has_conditioned_samples
            else None
        )
        unconditioned_cache = (
            self._build_condition_cache(
                observed_tokens=observed_tokens,
                actions=None,
                states=None,
                extrinsics=None,
                known_future_tokens=known_future_tokens,
                known_future_token_indices=known_future_token_indices,
                active_future_token_indices=resolved_future_indices,
            )
            if needs_unconditioned
            else None
        )
        num_function_evals = 0

        def guided_prediction(model_input: torch.Tensor, step_t: torch.Tensor) -> tuple[torch.Tensor, int]:
            eval_count = 0
            pred_cond = None
            if has_conditioned_samples:
                cond_kwargs = {
                    "noisy_future_tokens": model_input,
                    "timesteps": step_t,
                    "observed_tokens": observed_tokens,
                    "actions": actions,
                    "states": states,
                    "extrinsics": extrinsics,
                    "condition_cache": condition_cache,
                    "anchor_tokens": anchor_tokens,
                }
                if forward_future_indices is not None:
                    cond_kwargs["future_token_indices"] = forward_future_indices
                pred_cond = self.forward(**cond_kwargs)
                eval_count += 1
            if not needs_unconditioned:
                return pred_cond, eval_count

            uncond_kwargs = {
                "noisy_future_tokens": model_input,
                "timesteps": step_t,
                "observed_tokens": observed_tokens,
                "actions": None,
                "states": None,
                "extrinsics": None,
                "condition_cache": unconditioned_cache,
                "anchor_tokens": anchor_tokens,
            }
            if forward_future_indices is not None:
                uncond_kwargs["future_token_indices"] = forward_future_indices
            pred_uncond = self.forward(**uncond_kwargs)
            eval_count += 1
            if pred_cond is None:
                return pred_uncond, eval_count

            guided = pred_uncond + metadata_guidance_scale * (pred_cond - pred_uncond)
            return torch.where(metadata_condition_mask[:, None, None], guided, pred_uncond), eval_count

        # TODO: add multi-candidate sampling plus planner/downstream ranking once the scoring contract is defined.
        for t_start, dt in zip(timesteps, deltas):
            t = t_start.expand(observed_tokens.shape[0])
            model_pred, eval_count = guided_prediction(x, t)
            num_function_evals += eval_count
            if self.objective == "x0_prediction":
                t_expand = t[:, None, None].to(dtype=x.dtype)
                t_next = (t_start + dt).expand(observed_tokens.shape[0])
                t_next_expand = t_next[:, None, None].to(dtype=x.dtype)
                eps_pred = (x - t_expand * model_pred) / (1.0 - t_expand).clamp_min(1e-5)
                x = (1.0 - t_next_expand) * eps_pred + t_next_expand * model_pred
                continue

            velocity = model_pred
            if sampler_type == "euler":
                x = x + velocity * dt.to(dtype=x.dtype)
                continue

            x_euler = x + velocity * dt.to(dtype=x.dtype)
            t_next = (t_start + dt).expand(observed_tokens.shape[0])
            velocity_next, eval_count = guided_prediction(x_euler, t_next)
            num_function_evals += eval_count
            x = x + 0.5 * (velocity + velocity_next) * dt.to(dtype=x.dtype)
        if return_diagnostics:
            return {
                "samples": x,
                "timesteps": timesteps,
                "deltas": deltas,
                "sampler_type": sampler_type,
                "schedule_type": str(schedule_type).lower(),
                "objective": self.objective,
                "num_function_evals": num_function_evals,
            }
        return x

    @torch.no_grad()
    def sample_joint(
        self,
        *,
        observed_tokens: torch.Tensor,
        action_state_tokens: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        num_inference_steps: int = 8,
        sampler_type: str = "euler",
        schedule_type: str = "cosine",
        temperature: float = 1.0,
        return_diagnostics: bool = False,
        joint_action_inference_noise_mode: Optional[str] = None,
        joint_video_final_noise: Optional[float] = None,
        initial_world_noise: Optional[torch.Tensor] = None,
        initial_action_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Sample world latents and normalized ego actions with the joint branch."""
        if not self.joint_action_enabled:
            raise ValueError("sample_joint requires joint_action_enabled=True")
        sampler_type = str(sampler_type).lower()
        if sampler_type != "euler":
            raise ValueError("LatentDiT sample_joint currently supports sampler_type='euler' only")
        if self.objective == "x0_prediction" and sampler_type != "euler":
            raise ValueError("LatentDiT objective='x0_prediction' currently requires sampler_type='euler'")
        if action_state_tokens.ndim != 3 or action_state_tokens.shape[-1] != self.joint_action_state_dim:
            raise ValueError(
                f"action_state_tokens must have shape [B, H, {self.joint_action_state_dim}], "
                f"got {tuple(action_state_tokens.shape)}"
            )
        if action_state_tokens.shape[1] != self.num_future_steps:
            raise ValueError(
                f"action_state_tokens horizon {action_state_tokens.shape[1]} must match "
                f"num_future_steps={self.num_future_steps}"
            )

        mode = (
            self.joint_action_inference_noise_mode
            if joint_action_inference_noise_mode is None
            else str(joint_action_inference_noise_mode).lower()
        )
        final_noise = (
            self.joint_video_final_noise if joint_video_final_noise is None else float(joint_video_final_noise)
        )
        world_timesteps, world_deltas, action_timesteps, action_deltas, world_final_t = (
            self._build_joint_sampling_schedules(
                steps=max(1, int(num_inference_steps)),
                schedule_type=schedule_type,
                device=observed_tokens.device,
                joint_action_inference_noise_mode=mode,
                joint_video_final_noise=final_noise,
            )
        )
        world_shape = (observed_tokens.shape[0], self.num_future_tokens, self.embed_dim)
        action_shape = (observed_tokens.shape[0], self.num_future_steps, self.joint_action_dim)
        if initial_world_noise is None:
            x_world = torch.randn(world_shape, device=observed_tokens.device, dtype=observed_tokens.dtype)
        else:
            x_world = self._validate_initial_noise(
                initial_world_noise,
                expected_shape=world_shape,
                reference=observed_tokens,
                name="initial_world_noise",
            )
        if initial_action_noise is None:
            x_action = torch.randn(action_shape, device=observed_tokens.device, dtype=observed_tokens.dtype)
        else:
            x_action = self._validate_initial_noise(
                initial_action_noise,
                expected_shape=action_shape,
                reference=observed_tokens,
                name="initial_action_noise",
            )
        x_world = x_world * float(temperature)
        x_action = x_action * float(temperature)
        num_function_evals = 0

        for world_t, world_dt, action_t, action_dt in zip(
            world_timesteps, world_deltas, action_timesteps, action_deltas
        ):
            world_t_batch = world_t.expand(observed_tokens.shape[0])
            action_t_batch = torch.full(
                (observed_tokens.shape[0], self.num_future_steps),
                float(action_t),
                device=observed_tokens.device,
                dtype=x_action.dtype,
            )
            model_out = self.forward_joint(
                noisy_future_tokens=x_world,
                timesteps=world_t_batch,
                observed_tokens=observed_tokens,
                noisy_future_actions=x_action,
                action_timesteps=action_t_batch,
                action_state_tokens=action_state_tokens,
                actions=actions,
                states=states,
                extrinsics=extrinsics,
            )
            num_function_evals += 1
            world_pred = model_out["world_pred"]
            action_pred = model_out["action_pred"]
            if self.objective == "x0_prediction":
                world_next_t = world_t + world_dt
                world_eps_hat = (x_world - world_t.to(dtype=x_world.dtype) * world_pred) / max(
                    1.0 - float(world_t), 1e-5
                )
                x_world = (1.0 - world_next_t.to(dtype=x_world.dtype)) * world_eps_hat + world_next_t.to(
                    dtype=x_world.dtype
                ) * world_pred

                action_next_t = action_t + action_dt
                action_eps_hat = (x_action - action_t.to(dtype=x_action.dtype) * action_pred) / max(
                    1.0 - float(action_t), 1e-5
                )
                x_action = (1.0 - action_next_t.to(dtype=x_action.dtype)) * action_eps_hat + action_next_t.to(
                    dtype=x_action.dtype
                ) * action_pred
            else:
                x_world = x_world + world_pred * world_dt.to(dtype=x_world.dtype)
                x_action = x_action + action_pred * action_dt.to(dtype=x_action.dtype)

        result = {
            "samples": x_world,
            "actions": x_action,
            "objective": self.objective,
            "num_function_evals": num_function_evals,
            "joint_action_inference_noise_mode": mode,
            "joint_video_final_noise": final_noise,
            "world_final_t": world_final_t,
        }
        if return_diagnostics:
            result.update(
                {
                    "world_timesteps": world_timesteps,
                    "world_deltas": world_deltas,
                    "action_timesteps": action_timesteps,
                    "action_deltas": action_deltas,
                    "sampler_type": sampler_type,
                    "schedule_type": str(schedule_type).lower(),
                }
            )
        return result

    @staticmethod
    def _validate_initial_noise(
        noise: torch.Tensor,
        *,
        expected_shape: tuple[int, ...],
        reference: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if not torch.is_tensor(noise):
            raise TypeError(f"{name} must be a torch.Tensor when provided")
        if tuple(noise.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(noise.shape)}")
        if noise.device != reference.device:
            raise ValueError(f"{name} device must match observed_tokens: {noise.device} != {reference.device}")
        if noise.dtype != reference.dtype:
            raise ValueError(f"{name} dtype must match observed_tokens: {noise.dtype} != {reference.dtype}")
        return noise

    @staticmethod
    def _resolve_metadata_condition_mask(
        metadata_condition_mask: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if metadata_condition_mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if not torch.is_tensor(metadata_condition_mask):
            raise TypeError("metadata_condition_mask must be a torch.Tensor when provided")
        mask = metadata_condition_mask.to(device=device, dtype=torch.bool)
        if mask.ndim != 1 or mask.shape[0] != batch_size:
            raise ValueError(
                f"metadata_condition_mask must have shape [B], got {tuple(mask.shape)} for batch_size={batch_size}"
            )
        return mask

    def _build_joint_sampling_schedules(
        self,
        *,
        steps: int,
        schedule_type: str,
        device: torch.device,
        joint_action_inference_noise_mode: str,
        joint_video_final_noise: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
        mode = str(joint_action_inference_noise_mode).lower()
        if mode not in {"shared", "decoupled"}:
            raise ValueError("joint_action_inference_noise_mode must be 'shared' or 'decoupled'")
        final_noise = float(joint_video_final_noise)
        if not 0.0 <= final_noise < 1.0:
            raise ValueError(f"joint_video_final_noise must be in [0.0, 1.0), got {final_noise}")
        if final_noise > 0.0 and mode != "decoupled":
            raise ValueError("joint_video_final_noise > 0 requires joint_action_inference_noise_mode='decoupled'")
        base_timesteps, base_deltas = self._build_sampling_schedule(
            steps=int(steps),
            schedule_type=schedule_type,
            device=device,
        )
        action_timesteps = base_timesteps
        action_deltas = base_deltas
        if mode == "shared":
            return base_timesteps, base_deltas, action_timesteps, action_deltas, 1.0
        world_final_t = 1.0 - final_noise
        world_timesteps = base_timesteps * world_final_t
        world_deltas = base_deltas * world_final_t
        return world_timesteps, world_deltas, action_timesteps, action_deltas, world_final_t

    def _resolve_anchor_tokens(
        self,
        *,
        observed_tokens: torch.Tensor,
        anchor_tokens: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if anchor_tokens is None:
            if observed_tokens.shape[1] < self.tokens_per_frame:
                raise ValueError(
                    "observed_tokens must contain at least one frame when use_anchor_frame=True: "
                    f"got {tuple(observed_tokens.shape)} with tokens_per_frame={self.tokens_per_frame}"
                )
            anchor_tokens = observed_tokens[:, -self.tokens_per_frame :]
        if anchor_tokens.ndim != 3:
            raise ValueError("anchor_tokens must have shape [B, tokens_per_frame, D]")
        if anchor_tokens.shape[0] != reference.shape[0]:
            raise ValueError(
                f"anchor_tokens batch {anchor_tokens.shape[0]} does not match input batch {reference.shape[0]}"
            )
        if anchor_tokens.shape[1] != self.tokens_per_frame:
            raise ValueError(
                f"anchor_tokens length {anchor_tokens.shape[1]} does not match "
                f"tokens_per_frame={self.tokens_per_frame}"
            )
        if anchor_tokens.shape[2] != self.embed_dim:
            raise ValueError(f"anchor_tokens dim {anchor_tokens.shape[2]} does not match embed_dim={self.embed_dim}")
        return anchor_tokens.to(device=reference.device, dtype=reference.dtype)

    def _resolve_future_token_indices(
        self,
        future_token_indices: Optional[torch.Tensor],
        *,
        active_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        active_tokens = int(active_tokens)
        if active_tokens <= 0:
            raise ValueError(f"active future token count must be positive, got {active_tokens}")
        if future_token_indices is None:
            if active_tokens != self.num_future_tokens:
                raise ValueError(
                    f"noisy_future_tokens length {active_tokens} does not match configured "
                    f"num_future_tokens {self.num_future_tokens}; pass future_token_indices for masked inpainting"
                )
            return torch.arange(self.num_future_tokens, device=device, dtype=torch.long)
        if not torch.is_tensor(future_token_indices):
            raise TypeError("future_token_indices must be a torch.Tensor when provided")
        if future_token_indices.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(f"future_token_indices must have integer dtype, got {future_token_indices.dtype}")
        indices = future_token_indices.to(device=device, dtype=torch.long)
        if indices.ndim != 1:
            raise ValueError(f"future_token_indices must be 1D, got {tuple(indices.shape)}")
        if indices.numel() != active_tokens:
            raise ValueError(
                f"future_token_indices length {indices.numel()} does not match active token count {active_tokens}"
            )
        if bool((indices < 0).any().item()) or bool((indices >= self.num_future_tokens).any().item()):
            raise ValueError(
                f"future_token_indices out of range for num_future_tokens={self.num_future_tokens}: "
                f"{indices.detach().cpu().tolist()}"
            )
        if indices.numel() > 1 and not bool((indices[1:] > indices[:-1]).all().item()):
            raise ValueError("future_token_indices must be strictly increasing")
        return indices

    def _future_position_embed(self, future_token_indices: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        pos_indices = future_token_indices.to(device=self.pos_embed.device, dtype=torch.long) + self.num_anchor_tokens
        return self.pos_embed.index_select(1, pos_indices).to(device=reference.device, dtype=reference.dtype)

    def _resolve_known_future_tokens(
        self,
        known_future_tokens: Optional[torch.Tensor],
        known_future_token_indices: Optional[torch.Tensor],
        *,
        reference: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if known_future_tokens is None and known_future_token_indices is None:
            return None, None
        if known_future_tokens is None or known_future_token_indices is None:
            raise ValueError("known_future_tokens and known_future_token_indices must be provided together")
        if known_future_tokens.ndim != 3:
            raise ValueError(f"known_future_tokens must have shape [B, N, D], got {tuple(known_future_tokens.shape)}")
        if known_future_tokens.shape[0] != reference.shape[0]:
            raise ValueError(
                f"known_future_tokens batch {known_future_tokens.shape[0]} does not match {reference.shape[0]}"
            )
        if known_future_tokens.shape[2] != self.embed_dim:
            raise ValueError(
                f"known_future_tokens dim {known_future_tokens.shape[2]} does not match embed_dim={self.embed_dim}"
            )
        if not torch.is_tensor(known_future_token_indices):
            raise TypeError("known_future_token_indices must be a torch.Tensor when provided")
        if known_future_token_indices.ndim != 1 or known_future_token_indices.numel() != known_future_tokens.shape[1]:
            raise ValueError(
                "known_future_tokens length must match known_future_token_indices: "
                f"tokens={known_future_tokens.shape[1]}, indices_shape={tuple(known_future_token_indices.shape)}"
            )
        indices = self._resolve_future_token_indices(
            known_future_token_indices,
            active_tokens=int(known_future_tokens.shape[1]),
            device=reference.device,
        )
        return known_future_tokens.to(device=reference.device, dtype=reference.dtype), indices

    def _build_condition_cache(
        self,
        observed_tokens: torch.Tensor,
        actions: Optional[torch.Tensor],
        states: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
        known_future_tokens: Optional[torch.Tensor] = None,
        known_future_token_indices: Optional[torch.Tensor] = None,
        active_future_token_indices: Optional[torch.Tensor] = None,
        metadata_condition_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cond_tokens = self.context_proj(self.context_bottleneck(observed_tokens))
        cond_key_padding_mask = None
        metadata_condition_mask = (
            self._resolve_metadata_condition_mask(
                metadata_condition_mask,
                batch_size=cond_tokens.shape[0],
                device=cond_tokens.device,
            )
            if metadata_condition_mask is not None
            else None
        )
        if self.conditioning_mode == "temporal_aux_tokens":
            aux_tokens = self._temporal_aux_tokens(actions, states, extrinsics, cond_tokens)
            if aux_tokens is not None:
                if metadata_condition_mask is not None:
                    context_mask = torch.zeros(
                        cond_tokens.shape[0],
                        cond_tokens.shape[1],
                        dtype=torch.bool,
                        device=cond_tokens.device,
                    )
                    aux_mask = (~metadata_condition_mask)[:, None].expand(-1, aux_tokens.shape[1])
                    cond_key_padding_mask = torch.cat([context_mask, aux_mask], dim=1)
                cond_tokens = torch.cat([cond_tokens, aux_tokens], dim=1)
            side_condition = cond_tokens.new_zeros(cond_tokens.shape[0], cond_tokens.shape[-1])
        else:
            if metadata_condition_mask is not None and not bool(metadata_condition_mask.all().item()):
                raise ValueError(
                    "metadata_condition_mask requires conditioning_mode='temporal_aux_tokens' for "
                    "per-sample metadata-unconditional forwarding."
                )
            side_condition = self._side_condition(actions, states, extrinsics, cond_tokens.mean(dim=1))
        known_tokens, known_indices = self._resolve_known_future_tokens(
            known_future_tokens,
            known_future_token_indices,
            reference=observed_tokens,
        )
        if known_tokens is not None and known_indices is not None:
            if active_future_token_indices is not None:
                active_indices = active_future_token_indices.to(device=known_indices.device, dtype=torch.long)
                if active_indices.ndim != 1 or active_indices.numel() < 1:
                    raise ValueError(
                        f"active_future_token_indices must be non-empty 1D, got {tuple(active_indices.shape)}"
                    )
                if not bool((known_indices < active_indices[0]).all().item()):
                    raise ValueError(
                        "known_future_token_indices must describe a prefix before active future_token_indices"
                    )
            known_cond = self.context_proj(self.context_bottleneck(known_tokens))
            known_cond = known_cond + self._future_position_embed(known_indices, known_cond)
            known_cond = known_cond + self.future_condition_type_embed.to(
                device=known_cond.device, dtype=known_cond.dtype
            )
            if cond_key_padding_mask is not None:
                known_mask = torch.zeros(
                    cond_key_padding_mask.shape[0],
                    known_cond.shape[1],
                    dtype=torch.bool,
                    device=cond_key_padding_mask.device,
                )
                cond_key_padding_mask = torch.cat([cond_key_padding_mask, known_mask], dim=1)
            cond_tokens = torch.cat([cond_tokens, known_cond], dim=1)
        cache = {
            "cond_tokens": cond_tokens,
            "side_condition": side_condition,
        }
        if cond_key_padding_mask is not None:
            cache["cond_key_padding_mask"] = cond_key_padding_mask
        return cache

    def _fit_temporal_feature(
        self,
        tensor: Optional[torch.Tensor],
        *,
        batch_size: int,
        length: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Missing temporal side-input defaults to zeros (per owner: a missing dataset value means the
        # parameter is unimportant / not used downstream) rather than failing.
        if tensor is None:
            return torch.zeros(batch_size, length, feature_dim, device=device, dtype=dtype)
        values = tensor.to(device=device, dtype=dtype)
        if values.ndim != 3:
            raise ValueError(f"Temporal side input must have shape [B, T, D], got {tuple(values.shape)}")
        if values.shape[0] != batch_size:
            raise ValueError(f"Temporal side input batch {values.shape[0]} does not match {batch_size}")
        # fail-loud (point 8): feature 维度不符直接报错（一定是配置/数据错误）。
        # 时间维度保留 pad/truncate：actions 长度 T-1 vs states/extrinsics 长度 T 是 _temporal_aux_tokens
        # 注释中说明的既定对齐约定，不在此报错。
        if values.shape[2] != feature_dim:
            raise ValueError(
                f"Temporal side input feature dim {values.shape[2]} does not match expected {feature_dim}"
            )
        if values.shape[1] < length:
            pad = values.new_zeros(batch_size, length - values.shape[1], values.shape[2])
            values = torch.cat([values, pad], dim=1)
        elif values.shape[1] > length:
            values = values[:, :length]
        return values

    def _temporal_aux_tokens(
        self,
        actions: Optional[torch.Tensor],
        states: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        lengths = [x.shape[1] for x in (actions, states, extrinsics) if x is not None]
        if not lengths:
            return None
        # Alignment convention: aux_len = max temporal length; shorter inputs (e.g. actions of
        # length T-1 vs per-frame states/extrinsics of length T) are zero-padded at the tail, so
        # the last action token may not align frame-for-frame with state/extrinsics tokens. These
        # are unordered cross-attention context tokens, so strict per-step alignment is not required.
        aux_len = max(int(length) for length in lengths)
        if aux_len <= 0:
            return None
        if aux_len > self.max_steps:
            raise ValueError(f"Temporal side input length {aux_len} exceeds max_steps={self.max_steps}")

        batch_size = reference.shape[0]
        device = reference.device
        dtype = reference.dtype
        side = torch.cat(
            [
                self._fit_temporal_feature(
                    actions,
                    batch_size=batch_size,
                    length=aux_len,
                    feature_dim=self.action_dim,
                    device=device,
                    dtype=dtype,
                ),
                self._fit_temporal_feature(
                    states,
                    batch_size=batch_size,
                    length=aux_len,
                    feature_dim=self.state_dim,
                    device=device,
                    dtype=dtype,
                ),
                self._fit_temporal_feature(
                    extrinsics,
                    batch_size=batch_size,
                    length=aux_len,
                    feature_dim=self.extrinsics_dim,
                    device=device,
                    dtype=dtype,
                ),
            ],
            dim=-1,
        )
        aux_tokens = self.aux_token_embed(side)
        return aux_tokens + self.aux_pos_embed[:, :aux_len].to(device=device, dtype=dtype)

    @staticmethod
    def _build_sampling_schedule(
        steps: int,
        schedule_type: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        schedule_type = str(schedule_type).lower()
        u = torch.linspace(0.0, 1.0, int(steps) + 1, device=device, dtype=torch.float32)
        if schedule_type == "uniform":
            grid = u
        elif schedule_type == "cosine":
            grid = 0.5 - 0.5 * torch.cos(math.pi * u)
        elif schedule_type == "quadratic":
            grid = u.square()
        else:
            raise ValueError(f"schedule_type must be one of ['uniform', 'cosine', 'quadratic'], got {schedule_type!r}")
        return grid[:-1], grid[1:] - grid[:-1]

    def _side_condition(
        self,
        actions: Optional[torch.Tensor],
        states: Optional[torch.Tensor],
        extrinsics: Optional[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = reference.shape[0]
        device = reference.device
        dtype = reference.dtype

        # Missing data side-input defaults to zeros (per owner: a missing dataset value means the
        # parameter is unimportant / not used downstream), matching the states/extrinsics handling below.
        if actions is None:
            actions_mean = reference.new_zeros(batch_size, self.action_dim)
        else:
            actions_mean = actions.to(device=device, dtype=dtype).mean(dim=1)
        side_parts = [actions_mean]
        if states is not None:
            side_parts.append(states.to(device=device, dtype=dtype).mean(dim=1))
        else:
            side_parts.append(actions_mean.new_zeros(batch_size, self.state_dim))
        if extrinsics is not None:
            side_parts.append(extrinsics.to(device=device, dtype=dtype).mean(dim=1))
        else:
            side_parts.append(actions_mean.new_zeros(batch_size, self.extrinsics_dim))

        side = torch.cat(side_parts, dim=-1)
        expected = self.side_embed[0].in_features
        if side.shape[-1] < expected:
            side = torch.cat([side, side.new_zeros(batch_size, expected - side.shape[-1])], dim=-1)
        elif side.shape[-1] > expected:
            side = side[:, :expected]
        return self.side_embed(side)

    @staticmethod
    def _sample_timestep(batch_size: int, device: torch.device) -> torch.Tensor:
        t = torch.sigmoid(torch.randn(batch_size, device=device))
        return t.clamp(1e-5, 1.0 - 1e-5)
