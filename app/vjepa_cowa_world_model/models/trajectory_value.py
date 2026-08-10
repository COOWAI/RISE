"""Temporal value head for predictor future tokens."""

from typing import Optional

import torch
import torch.nn as nn


class TemporalTrajectoryValueHead(nn.Module):
    """Predict per-future-step values from predictor latent tokens.

    Parameters
    ----------
    embed_dim  : predictor token dimension ``D``.
    hidden_dim : hidden MLP dimension.
    dropout    : dropout probability.

    Input shapes
    ------------
    ``[B, F*tokens_per_frame, D]`` or ``[B, F, tokens_per_frame, D]``.
    Output shape is ``[B, F]``.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.frame_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _pool_frames(self, z_future: torch.Tensor, tokens_per_frame: Optional[int]) -> torch.Tensor:
        if z_future.ndim == 4:
            if z_future.shape[-1] != self.embed_dim:
                raise ValueError(f"Expected embed_dim={self.embed_dim}, got {z_future.shape[-1]}")
            frame_tokens = z_future.mean(dim=2)
        elif z_future.ndim == 3:
            if z_future.shape[-1] != self.embed_dim:
                raise ValueError(f"Expected embed_dim={self.embed_dim}, got {z_future.shape[-1]}")
            if tokens_per_frame is None:
                raise ValueError("tokens_per_frame is required for flat [B, F*T, D] value-head input")
            tokens_per_frame = int(tokens_per_frame)
            if tokens_per_frame <= 0 or z_future.shape[1] % tokens_per_frame != 0:
                raise ValueError(
                    "tokens_per_frame must be positive and divide the flat token length; "
                    f"got tokens_per_frame={tokens_per_frame}, token_length={z_future.shape[1]}"
                )
            frame_tokens = z_future.reshape(
                z_future.shape[0],
                z_future.shape[1] // tokens_per_frame,
                tokens_per_frame,
                z_future.shape[-1],
            ).mean(dim=2)
        else:
            raise ValueError(f"z_future must be [B, N, D] or [B, F, T, D], got {tuple(z_future.shape)}")
        return self.frame_norm(frame_tokens)

    def forward(self, z_future: torch.Tensor, *, tokens_per_frame: Optional[int] = None) -> torch.Tensor:
        frame_tokens = self._pool_frames(z_future, tokens_per_frame)
        return self.mlp(frame_tokens).squeeze(-1)
