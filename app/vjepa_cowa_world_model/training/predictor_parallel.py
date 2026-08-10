"""Future-query parallel predictor helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.models import (
    build_predictor_input_with_future_queries,
    register_predictor_future_query_tokens,
)
from app.vjepa_cowa_world_model.training.predictor_aux import call_predictor_with_aux, prepare_predictor_aux_inputs
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module


@dataclass(frozen=True)
class ParallelPredictorOutput:
    """Outputs from a single future-query predictor forward."""

    z_pred: torch.Tensor
    z_future: torch.Tensor
    z_ar: Optional[torch.Tensor]


def use_parallel_predictor(config) -> bool:
    """Return whether future-query parallel predictor mode is enabled."""
    return bool(config.train.use_parallel_predictor)


def has_parallel_predictor_tokens(predictor: torch.nn.Module) -> bool:
    predictor_core = unwrap_module(predictor)
    return getattr(predictor_core, "future_query_tokens", None) is not None


def maybe_register_parallel_predictor_tokens(
    predictor: torch.nn.Module,
    config,
    embed_dim: int,
    future_steps: int,
    tokens_per_frame: int,
    device: torch.device,
) -> None:
    """Register future query tokens when parallel predictor mode is enabled."""
    if not use_parallel_predictor(config):
        return
    register_predictor_future_query_tokens(
        predictor,
        embed_dim=embed_dim,
        future_tubelets=int(future_steps),
        tokens_per_frame=int(tokens_per_frame),
        device=device,
    )


def build_parallel_predictor_input(
    predictor: torch.nn.Module,
    observed_tokens: torch.Tensor,
    num_observed_steps: int,
    tokens_per_frame: int,
) -> torch.Tensor:
    """Build observed-token prefix plus learnable future query tokens."""
    num_observed_tokens = int(num_observed_steps) * int(tokens_per_frame)
    if observed_tokens.size(1) < num_observed_tokens:
        raise ValueError(
            "Observed token sequence is shorter than the requested observed prefix: "
            f"tokens={observed_tokens.size(1)}, requested={num_observed_tokens}"
        )
    observed_prefix = observed_tokens[:, :num_observed_tokens]
    return build_predictor_input_with_future_queries(predictor, observed_prefix)


def forward_parallel_predictor(
    predictor: torch.nn.Module,
    observed_tokens: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor,
    config,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    num_observed_steps: int,
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
    predictor_no_aux_input: Optional[bool] = None,
) -> ParallelPredictorOutput:
    """Run predictor once with future query tokens and return future tokens."""
    z_input = build_parallel_predictor_input(
        predictor,
        observed_tokens=observed_tokens,
        num_observed_steps=num_observed_steps,
        tokens_per_frame=tokens_per_frame,
    )
    if z_input.size(1) % tokens_per_frame != 0:
        raise ValueError(
            f"Parallel predictor input tokens ({z_input.size(1)}) must be divisible by tokens_per_frame "
            f"({tokens_per_frame})"
        )
    num_steps = z_input.size(1) // tokens_per_frame
    if actions.shape[1] != num_steps:
        raise ValueError(f"Parallel predictor actions length mismatch: {actions.shape[1]} vs {num_steps}")
    if states.shape[1] != num_steps:
        raise ValueError(f"Parallel predictor states length mismatch: {states.shape[1]} vs {num_steps}")
    if extrinsics.shape[1] != num_steps:
        raise ValueError(f"Parallel predictor extrinsics length mismatch: {extrinsics.shape[1]} vs {num_steps}")

    aux_inputs = prepare_predictor_aux_inputs(
        actions=actions,
        states=states,
        extrinsics=extrinsics,
        config=config,
        num_observed_steps=int(num_observed_steps),
        driving_command=driving_command,
        ego_dynamics=ego_dynamics,
        predictor_no_aux_input=predictor_no_aux_input,
    )

    z_pred = call_predictor_with_aux(predictor, z_input, aux_inputs)
    if runtime_normalize_reps:
        z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))

    future_start = int(num_observed_steps) * int(tokens_per_frame)
    z_future = z_pred[:, future_start:]
    return ParallelPredictorOutput(z_pred=z_pred, z_future=z_future, z_ar=z_future)
