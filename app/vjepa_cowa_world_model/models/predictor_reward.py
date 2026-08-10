"""Reward head that scores future predictor tokens."""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictorRewardHead(nn.Module):
    """Pool future predictor tokens and predict safety reward logits.

    Parameters
    ----------
    embed_dim : int
        Predictor token embedding dimension.
    hidden_dim : int
        MLP hidden size.
    num_horizons : int
        Number of auxiliary horizon rewards.
    dropout : float
        Dropout probability inside the MLP.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 512, num_horizons: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_horizons = int(num_horizons)
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if self.num_horizons <= 0:
            raise ValueError("num_horizons must be positive")

        self.frame_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),
            nn.Linear(self.embed_dim * 2, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.scalar_head = nn.Linear(int(hidden_dim), 1)
        self.horizon_head = nn.Linear(int(hidden_dim), self.num_horizons)

    def _pool_future_tokens(self, z_future: torch.Tensor, tokens_per_frame: Optional[int]) -> torch.Tensor:
        if z_future.ndim == 4:
            if z_future.shape[-1] != self.embed_dim:
                raise ValueError(f"Expected embed_dim={self.embed_dim}, got {z_future.shape[-1]}")
            frame_tokens = z_future.mean(dim=2)
        elif z_future.ndim == 3:
            if z_future.shape[-1] != self.embed_dim:
                raise ValueError(f"Expected embed_dim={self.embed_dim}, got {z_future.shape[-1]}")
            if tokens_per_frame is None:
                frame_tokens = z_future
            else:
                tokens_per_frame = int(tokens_per_frame)
                if tokens_per_frame <= 0 or z_future.shape[1] % tokens_per_frame != 0:
                    raise ValueError(
                        "tokens_per_frame must be positive and divide the flat token length; "
                        f"got tokens_per_frame={tokens_per_frame}, token_length={z_future.shape[1]}"
                    )
                frame_tokens = z_future.view(z_future.shape[0], -1, tokens_per_frame, z_future.shape[-1]).mean(dim=2)
        else:
            raise ValueError(f"z_future must be [B, N, D] or [B, T, P, D], got {tuple(z_future.shape)}")

        frame_tokens = self.frame_norm(frame_tokens)
        mean_pool = frame_tokens.mean(dim=1)
        max_pool = frame_tokens.amax(dim=1)
        return torch.cat([mean_pool, max_pool], dim=-1)

    def forward(self, z_future: torch.Tensor, *, tokens_per_frame: Optional[int] = None) -> Dict[str, torch.Tensor]:
        pooled = self._pool_future_tokens(z_future, tokens_per_frame)
        hidden = self.mlp(pooled)
        return {
            "scalar_logit": self.scalar_head(hidden).squeeze(-1),
            "horizon_logits": self.horizon_head(hidden),
        }


def compute_reward_head_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    *,
    horizon_weight: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """Compute BCE-with-logits loss for scalar and horizon soft labels."""
    scalar_logit = outputs["scalar_logit"]
    horizon_logits = outputs["horizon_logits"]
    scalar_target = targets["scalar"].to(device=scalar_logit.device, dtype=scalar_logit.dtype)
    horizon_target = targets["horizons"].to(device=horizon_logits.device, dtype=horizon_logits.dtype)

    if scalar_target.shape != scalar_logit.shape:
        raise ValueError(f"scalar target shape {tuple(scalar_target.shape)} != logits {tuple(scalar_logit.shape)}")
    if horizon_target.shape != horizon_logits.shape:
        raise ValueError(f"horizon target shape {tuple(horizon_target.shape)} != logits {tuple(horizon_logits.shape)}")

    scalar_loss = F.binary_cross_entropy_with_logits(scalar_logit, scalar_target)
    horizon_loss = F.binary_cross_entropy_with_logits(horizon_logits, horizon_target)
    loss = scalar_loss + float(horizon_weight) * horizon_loss
    return {
        "loss": loss,
        "scalar_loss": scalar_loss,
        "horizon_loss": horizon_loss,
        "scalar_pred": torch.sigmoid(scalar_logit),
        "horizon_pred": torch.sigmoid(horizon_logits),
    }
