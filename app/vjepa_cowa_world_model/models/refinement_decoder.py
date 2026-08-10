"""Trainable refinement decoder that consumes external proposal outputs."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .multimodal_planner import MultiModalTemporalPlanner
from .trajectory_head import TrajectoryHead


class RefinementDecoder(nn.Module):
    """Refine/final decoder for staged training with external proposal providers."""

    SOURCE_IMAGE = 0
    SOURCE_STATUS = 1
    SOURCE_FUTURE = 2
    SOURCE_TRAJECTORY = 3

    def __init__(
        self,
        encoder_dim: int = 1024,
        tf_d_model: int = 256,
        tf_d_ffn: int = 1024,
        tf_num_layers: int = 3,
        tf_num_head: int = 8,
        tf_dropout: float = 0.0,
        tokens_per_frame: int = 256,
        num_poses: int = 7,
        num_time_steps: int = 7,
        num_context_frames: Optional[int] = None,
        status_dim: int = 7,
        use_spatial_tokens: bool = False,
        num_modes: int = 6,
        use_temporal: bool = True,
        use_time_aligned_bias: bool = True,
        time_aligned_bias_strength: float = 5.0,
        use_status_for_planner: bool = True,
        command_dim: int = 0,
        max_rounds: int = 4,
    ):
        super().__init__()
        context_frames = num_context_frames or num_time_steps
        self.refine_core = MultiModalTemporalPlanner(
            encoder_dim=encoder_dim,
            tf_d_model=tf_d_model,
            tf_d_ffn=tf_d_ffn,
            tf_num_layers=tf_num_layers,
            tf_num_head=tf_num_head,
            tf_dropout=tf_dropout,
            tokens_per_frame=tokens_per_frame,
            num_poses=num_poses,
            num_time_steps=num_time_steps,
            num_context_frames=context_frames,
            status_dim=status_dim,
            use_spatial_tokens=use_spatial_tokens,
            num_modes=num_modes,
            use_temporal=use_temporal,
            use_time_aligned_bias=use_time_aligned_bias,
            time_aligned_bias_strength=time_aligned_bias_strength,
            use_z_context=True,
            use_status_for_planner=use_status_for_planner,
            command_dim=command_dim,
        )

        self.encoder_dim = encoder_dim
        self.tf_d_model = tf_d_model
        self.num_modes = num_modes
        self.num_poses = num_poses
        self.max_rounds = max_rounds

        self.mode_embed = nn.Embedding(num_modes, tf_d_model)
        self.source_embed = nn.Embedding(4, tf_d_model)
        self.round_embed = nn.Embedding(max_rounds, tf_d_model)
        self.fut_proj = nn.Linear(encoder_dim, tf_d_model)
        self.traj_tokenizer = nn.Sequential(
            nn.Linear(3, tf_d_model),
            nn.GELU(),
            nn.Linear(tf_d_model, tf_d_model),
        )
        self.final_query = nn.Embedding(num_poses, tf_d_model)
        self.final_head = TrajectoryHead(num_poses, tf_d_ffn, tf_d_model)

        obs_offset = self.refine_core.num_observed_frames if self.refine_core.use_observed_tokens else 0
        self.register_buffer(
            "final_query_step_idx",
            torch.arange(num_poses, dtype=torch.long) + obs_offset,
            persistent=False,
        )

    def encode_proposal_features(self, proposal_traj: torch.Tensor) -> torch.Tensor:
        return self.traj_tokenizer(proposal_traj)

    def _context_uses_temporal_path(self, z_context: torch.Tensor) -> bool:
        if self.refine_core.use_temporal:
            return True
        return z_context.shape[1] > self.refine_core.tokens_per_frame

    def _build_context_memory(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._context_uses_temporal_path(z_context):
            num_steps = z_context.shape[1] // self.refine_core.tokens_per_frame
            memory, memory_step_idx = self.refine_core._build_memory_temporal(z_context, status_feature, num_steps)
            source_ids = torch.full_like(memory_step_idx, self.SOURCE_IMAGE)
            source_ids = torch.where(
                memory_step_idx.eq(-1),
                torch.full_like(memory_step_idx, self.SOURCE_STATUS),
                source_ids,
            )
            memory = memory + self.source_embed(source_ids).unsqueeze(0)
            return memory, memory_step_idx

        memory = self.refine_core._build_memory_single(z_context, status_feature)
        img_tokens = self.refine_core.tokens_per_frame if self.refine_core.use_spatial_tokens else 1
        source_ids = torch.full((memory.shape[1],), self.SOURCE_STATUS, device=memory.device, dtype=torch.long)
        source_ids[:img_tokens] = self.SOURCE_IMAGE
        memory = memory + self.source_embed(source_ids).unsqueeze(0)
        return memory, None

    def _build_memory_mask(
        self,
        query_step_idx: torch.Tensor,
        memory_step_idx: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if memory_step_idx is None:
            return None
        return self.refine_core._build_time_aligned_memory_bias(query_step_idx, memory_step_idx, dtype)

    def _run_shared_transformer(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        def _forward(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
            if memory_mask is None:
                return self.refine_core.transformer(src=src, tgt=tgt)
            return self.refine_core.transformer(src=src, tgt=tgt, memory_mask=memory_mask)

        if use_checkpoint and (memory.requires_grad or query.requires_grad):
            return checkpoint(_forward, memory, query, use_reentrant=False)
        return _forward(memory, query)

    def _decode_mode_queries(self, query_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = query_out.shape[0]
        query_out = query_out.view(batch_size, self.num_modes, self.num_poses, self.tf_d_model)

        traj_list = []
        for mode_idx in range(self.num_modes):
            head_out = self.refine_core.trajectory_heads[mode_idx](query_out[:, mode_idx])
            traj_list.append(head_out["trajectory"] if isinstance(head_out, dict) else head_out)
        trajectories = torch.stack(traj_list, dim=1)

        conf_feat = query_out.mean(dim=2)
        confidences = self.refine_core.confidence_head(conf_feat.reshape(batch_size, self.num_modes * self.tf_d_model))
        return trajectories, confidences, query_out

    def _decode_final_query(self, query_out: torch.Tensor) -> torch.Tensor:
        head_out = self.final_head(query_out)
        return head_out["trajectory"] if isinstance(head_out, dict) else head_out

    def _compose_refinement_memory(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        traj_m: Optional[torch.Tensor],
        z_fut_m: Optional[torch.Tensor],
        round_index: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        memory, memory_step_idx = self._build_context_memory(z_context, status_feature)
        memory_parts = [memory]
        step_idx_parts = [memory_step_idx] if memory_step_idx is not None else []

        if z_fut_m is not None:
            mode_tokens = self.mode_embed.weight.view(1, self.num_modes, 1, self.tf_d_model)
            future_tokens = (
                self.fut_proj(z_fut_m) + mode_tokens + self.source_embed.weight[self.SOURCE_FUTURE].view(1, 1, 1, -1)
            )
            if round_index is not None:
                future_tokens = future_tokens + self.round_embed.weight[round_index].view(1, 1, 1, -1)
            memory_parts.append(future_tokens.flatten(1, 2))
            if memory_step_idx is not None:
                fut_idx = torch.full(
                    (self.num_modes * z_fut_m.shape[2],),
                    -1,
                    device=z_fut_m.device,
                    dtype=torch.long,
                )
                step_idx_parts.append(fut_idx)

        if traj_m is not None:
            mode_tokens = self.mode_embed.weight.view(1, self.num_modes, 1, self.tf_d_model)
            traj_tokens = (
                self.traj_tokenizer(traj_m)
                + mode_tokens
                + self.source_embed.weight[self.SOURCE_TRAJECTORY].view(1, 1, 1, -1)
            )
            if round_index is not None:
                traj_tokens = traj_tokens + self.round_embed.weight[round_index].view(1, 1, 1, -1)
            memory_parts.append(traj_tokens.flatten(1, 2))
            if memory_step_idx is not None:
                traj_idx = torch.full(
                    (self.num_modes * traj_m.shape[2],),
                    -1,
                    device=traj_m.device,
                    dtype=torch.long,
                )
                step_idx_parts.append(traj_idx)

        combined_memory = torch.cat(memory_parts, dim=1)
        if memory_step_idx is None:
            return combined_memory, None
        return combined_memory, torch.cat(step_idx_parts, dim=0)

    def _forward_mode_update(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        traj_m: torch.Tensor,
        z_fut_m: Optional[torch.Tensor],
        round_index: int,
        use_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, memory_step_idx = self._compose_refinement_memory(
            z_context,
            status_feature,
            traj_m,
            z_fut_m,
            round_index=round_index,
        )
        query = self.refine_core.query_embedding.weight.unsqueeze(0).expand(z_context.shape[0], -1, -1)
        query = query + self.round_embed.weight[round_index].view(1, 1, -1)
        memory_mask = self._build_memory_mask(self.refine_core.query_step_idx, memory_step_idx, memory.dtype)
        query_out = self._run_shared_transformer(memory, query, memory_mask=memory_mask, use_checkpoint=use_checkpoint)
        return self._decode_mode_queries(query_out)

    def forward_refine(
        self,
        z_context: torch.Tensor,
        status_feature: torch.Tensor,
        traj_m: torch.Tensor,
        z_fut_m: Optional[torch.Tensor],
        proposal_features: Optional[torch.Tensor] = None,
        round_index: Optional[int] = None,
    ) -> torch.Tensor:
        memory, memory_step_idx = self._compose_refinement_memory(
            z_context,
            status_feature,
            traj_m,
            z_fut_m,
            round_index=round_index,
        )
        query = self.final_query.weight.unsqueeze(0).expand(z_context.shape[0], -1, -1)
        if proposal_features is not None:
            query = query + proposal_features.mean(dim=1)
        if round_index is not None:
            query = query + self.round_embed.weight[round_index].view(1, 1, -1)
        memory_mask = self._build_memory_mask(self.final_query_step_idx, memory_step_idx, memory.dtype)
        query_out = self._run_shared_transformer(memory, query, memory_mask=memory_mask)
        return self._decode_final_query(query_out)

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
                current_fut,
                round_index=min(round_index, self.max_rounds - 1),
                use_checkpoint=grad_checkpoint and round_index < num_rounds - 1,
            )
            traj_rounds.append({"trajectories": current_traj, "confidences": current_logits})

        if not return_single_final:
            return traj_rounds, None

        traj_final = self.forward_refine(
            z_context,
            status_feature,
            current_traj,
            current_fut,
            current_features,
            round_index=min(num_rounds - 1, self.max_rounds - 1),
        )
        return traj_rounds, traj_final
