"""Prefix-aware dual value model for CVoI field and stopping values."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PrefixValueOutput:
    """Dual values for future prefixes.

    Attributes
    ----------
    field_values : [B, F]
        Value-field prediction after each future frame.
    stop_values : [B, F + 1]
        Stopping value at the observed-only state ``h=0`` and after every
        future prefix.
    """

    field_values: torch.Tensor
    stop_values: torch.Tensor


class PrefixDualValueModel(nn.Module):
    """Pool spatial frame tokens and encode all prefixes with one causal GRU.

    Observed frames and future frames pass through the same recurrent model.
    The final observed hidden state is therefore the exact ``h=0`` state used
    by both full and incremental evaluation.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        dropout = float(dropout)
        self.dropout = dropout
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.frame_norm = nn.LayerNorm(self.embed_dim)
        self.prefix_gru = nn.GRU(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.field_head = nn.Linear(self.hidden_dim, 1)
        self.stop_head = nn.Linear(self.hidden_dim, 1)

    def _pool_frames(
        self,
        tokens: torch.Tensor,
        *,
        tokens_per_frame: Optional[int],
        name: str,
    ) -> torch.Tensor:
        if tokens.ndim == 4:
            if tokens.shape[-1] != self.embed_dim:
                raise ValueError(f"{name} expected embed_dim={self.embed_dim}, got {tokens.shape[-1]}")
            if tokens.shape[2] < 1:
                raise ValueError(f"{name} must contain at least one spatial token per frame")
            frames = tokens.mean(dim=2)
        elif tokens.ndim == 3:
            if tokens.shape[-1] != self.embed_dim:
                raise ValueError(f"{name} expected embed_dim={self.embed_dim}, got {tokens.shape[-1]}")
            if tokens_per_frame is None:
                raise ValueError(f"tokens_per_frame is required for flat {name} input")
            tokens_per_frame = int(tokens_per_frame)
            if tokens_per_frame <= 0 or tokens.shape[1] % tokens_per_frame != 0:
                raise ValueError(
                    "tokens_per_frame must be positive and divide the flat token length; "
                    f"got tokens_per_frame={tokens_per_frame}, token_length={tokens.shape[1]}"
                )
            frames = tokens.reshape(
                tokens.shape[0],
                tokens.shape[1] // tokens_per_frame,
                tokens_per_frame,
                tokens.shape[-1],
            ).mean(dim=2)
        else:
            raise ValueError(f"{name} must be [B, N, D] or [B, F, T, D], got {tuple(tokens.shape)}")
        return self.frame_norm(frames)

    def encode_observed(
        self,
        z_observed: torch.Tensor,
        *,
        tokens_per_frame: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode observed frames and return recurrent state ``[L, B, H]``."""

        observed_frames = self._pool_frames(
            z_observed,
            tokens_per_frame=tokens_per_frame,
            name="z_observed",
        )
        if observed_frames.shape[1] < 1:
            raise ValueError("z_observed must contain at least one frame")
        _, state = self.prefix_gru(observed_frames)
        return state

    def extend_prefix(
        self,
        state: torch.Tensor,
        z_future: torch.Tensor,
        *,
        tokens_per_frame: Optional[int] = None,
    ) -> Tuple[PrefixValueOutput, torch.Tensor]:
        """Extend a recurrent prefix state by zero or more future frames.

        The returned stop sequence includes the input state's stopping value,
        which makes an empty extension a valid ``h=0`` evaluation.
        """

        if state.ndim != 3 or state.shape[0] != self.num_layers or state.shape[2] != self.hidden_dim:
            raise ValueError(f"state must be [{self.num_layers}, B, {self.hidden_dim}], got {tuple(state.shape)}")
        future_frames = self._pool_frames(
            z_future,
            tokens_per_frame=tokens_per_frame,
            name="z_future",
        )
        if future_frames.shape[0] != state.shape[1]:
            raise ValueError(f"z_future batch {future_frames.shape[0]} does not match state batch {state.shape[1]}")

        h0_stop = self.stop_head(state[-1]).squeeze(-1).unsqueeze(1)
        if future_frames.shape[1] == 0:
            empty_fields = h0_stop.new_empty((h0_stop.shape[0], 0))
            return PrefixValueOutput(field_values=empty_fields, stop_values=h0_stop), state

        future_states, next_state = self.prefix_gru(future_frames, state)
        field_values = self.field_head(future_states).squeeze(-1)
        future_stops = self.stop_head(future_states).squeeze(-1)
        stop_values = torch.cat([h0_stop, future_stops], dim=1)
        return PrefixValueOutput(field_values=field_values, stop_values=stop_values), next_state

    def forward(
        self,
        z_observed: torch.Tensor,
        z_future: torch.Tensor,
        *,
        tokens_per_frame: Optional[int] = None,
    ) -> PrefixValueOutput:
        """Evaluate field values ``[B, F]`` and stop values ``[B, F+1]``."""

        state = self.encode_observed(z_observed, tokens_per_frame=tokens_per_frame)
        output, _ = self.extend_prefix(state, z_future, tokens_per_frame=tokens_per_frame)
        return output
