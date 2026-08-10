"""Batch-one sequential rollout orchestration for a CVoI gate."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch

from app.vjepa_cowa_world_model.training.sequential_budget_control import (
    SequentialRolloutGate,
    apply_cvoi_formal_v2_navsim_e120_gate_feature_mask,
    apply_cvoi_gate_feature_mask,
    build_sequential_gate_features,
)


@dataclass(frozen=True)
class SequentialRolloutResult:
    """Terminal planner output and the raw-prefix decision trace."""

    stop_horizon: int
    raw_prefix: torch.Tensor
    planner_output: Any
    decisions: List[str]
    predicted_deltas: List[float]
    rollout_tokens_finite: torch.Tensor

    def require_finite_rollout_tokens(self) -> None:
        """Materialize the deferred Predictor-token check at the caller's synchronization boundary."""

        if (
            not isinstance(self.rollout_tokens_finite, torch.Tensor)
            or self.rollout_tokens_finite.ndim != 0
            or self.rollout_tokens_finite.dtype != torch.bool
        ):
            raise TypeError("sequential rollout finite-token diagnostic must be one boolean tensor")
        if not bool(self.rollout_tokens_finite):
            raise ValueError(
                "rollout_step must return finite "
                f"[1, N>0, {self.raw_prefix.shape[2]}] tokens, got {tuple(self.raw_prefix.shape)}"
            )


def _scalar_feature(name: str, value: Any, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.numel() != 1 or not torch.isfinite(tensor).all():
        raise ValueError(f"value_features[{name!r}] must be one finite scalar")
    return tensor.reshape(1)


def run_sequential_rollout(
    *,
    observed_latent: torch.Tensor,
    gate: torch.nn.Module,
    max_horizon: int,
    lambda_compute: float,
    compute_costs: List[float],
    rollout_step: Callable[[torch.Tensor, int], torch.Tensor],
    value_features: Callable[[torch.Tensor, torch.Tensor, int], Dict[str, Any]],
    stop_and_plan: Callable[[torch.Tensor, int, bool], Any],
    gate_feature_mode: str = "full",
    gate_feature_protocol: str = "legacy_v1",
) -> SequentialRolloutResult:
    """Roll one raw latent step at a time; guide only after the terminal STOP."""

    if observed_latent.ndim != 3 or int(observed_latent.shape[0]) != 1:
        raise ValueError(
            "sequential rollout runtime requires observed_latent [B,N,D] with batch size 1, "
            f"got {tuple(observed_latent.shape)}"
        )
    if not torch.isfinite(observed_latent).all():
        raise ValueError("observed_latent must contain only finite values")
    if type(max_horizon) is not int or max_horizon <= 0:
        raise ValueError(f"sequential CVoI runtime max_horizon must be a positive integer, got {max_horizon!r}")
    if len(compute_costs) != max_horizon + 1:
        raise ValueError(f"compute_costs must contain H+1={max_horizon + 1} entries")
    costs = torch.as_tensor(compute_costs, dtype=torch.float32, device=observed_latent.device)
    if not torch.isfinite(costs).all() or torch.any(costs < 0.0) or torch.any(costs[1:] < costs[:-1]):
        raise ValueError("compute_costs must be finite, non-negative, and non-decreasing")
    lambda_compute = float(lambda_compute)
    if not torch.isfinite(torch.tensor(lambda_compute)) or lambda_compute < 0.0:
        raise ValueError(f"lambda_compute must be finite and non-negative, got {lambda_compute}")
    if gate_feature_protocol == "legacy_v1":
        apply_gate_feature_mask = apply_cvoi_gate_feature_mask
    elif gate_feature_protocol == "formal_v2_navsim_e120_h4_v3":
        apply_gate_feature_mask = apply_cvoi_formal_v2_navsim_e120_gate_feature_mask
    else:
        raise ValueError("gate_feature_protocol must be exactly 'legacy_v1' or 'formal_v2_navsim_e120_h4_v3'")

    latent_dim = int(observed_latent.shape[2])
    expected_feature_dim = 2 * latent_dim + 7
    feature_dim = getattr(gate, "feature_dim", expected_feature_dim)
    if int(feature_dim) != expected_feature_dim:
        raise ValueError(f"gate feature_dim must match runtime schema {expected_feature_dim}, got {int(feature_dim)}")

    observed = observed_latent.detach()
    raw_prefix = observed.new_empty((1, 0, latent_dim))
    pooled_observed = observed.float().mean(dim=1)
    previous_stop = None
    decisions: List[str] = []
    predicted_deltas: List[float] = []
    rollout_tokens_finite = observed.new_ones((), dtype=torch.bool)

    for horizon in range(max_horizon + 1):
        if horizon == max_horizon:
            decisions.append("STOP")
            planner_output = stop_and_plan(raw_prefix, horizon, True)
            return SequentialRolloutResult(
                stop_horizon=horizon,
                raw_prefix=raw_prefix,
                planner_output=planner_output,
                decisions=decisions,
                predicted_deltas=predicted_deltas,
                rollout_tokens_finite=rollout_tokens_finite,
            )

        raw_features = value_features(observed, raw_prefix, horizon)
        if not isinstance(raw_features, dict) or set(raw_features) != {"field_value", "stop_value"}:
            raise ValueError("value_features must return exactly field_value and stop_value")
        field_value = _scalar_feature("field_value", raw_features["field_value"], device=observed.device)
        stop_value = _scalar_feature("stop_value", raw_features["stop_value"], device=observed.device)
        if horizon == 0 and not torch.equal(field_value, torch.zeros_like(field_value)):
            raise ValueError("CVoI h=0 field_value must use the explicit zero sentinel")
        if previous_stop is None:
            previous_stop = stop_value

        pooled_prefix = (
            raw_prefix.float().mean(dim=1) if raw_prefix.shape[1] > 0 else torch.zeros_like(pooled_observed)
        )
        next_cost = costs[min(horizon + 1, max_horizon)].reshape(1)
        features = build_sequential_gate_features(
            pooled_observed=pooled_observed,
            pooled_prefix=pooled_prefix,
            field_value=field_value,
            stop_value=stop_value,
            previous_stop_value=previous_stop,
            horizon=torch.tensor([horizon], device=observed.device),
            max_horizon=max_horizon,
            current_cost=costs[horizon].reshape(1),
            next_cost=next_cost,
            lambda_compute=torch.tensor([lambda_compute], device=observed.device),
        )
        features = apply_gate_feature_mask(
            features,
            latent_dim=latent_dim,
            mode=gate_feature_mode,
        )
        previous_stop = stop_value

        with torch.no_grad():
            predicted_delta = gate(features)
        if predicted_delta.shape != (1,) or not torch.isfinite(predicted_delta).all():
            raise ValueError(f"gate must return one finite delta, got {tuple(predicted_delta.shape)}")
        predicted_deltas.append(float(predicted_delta.item()))
        should_roll = bool(
            SequentialRolloutGate.should_roll(
                predicted_delta,
                horizon=horizon,
                max_horizon=max_horizon,
            ).item()
        )
        if not should_roll:
            decisions.append("STOP")
            planner_output = stop_and_plan(raw_prefix, horizon, horizon > 0)
            return SequentialRolloutResult(
                stop_horizon=horizon,
                raw_prefix=raw_prefix,
                planner_output=planner_output,
                decisions=decisions,
                predicted_deltas=predicted_deltas,
                rollout_tokens_finite=rollout_tokens_finite,
            )

        decisions.append("ROLL")
        next_tokens = rollout_step(raw_prefix, horizon + 1)
        if (
            not torch.is_tensor(next_tokens)
            or next_tokens.ndim != 3
            or next_tokens.shape[0] != 1
            or next_tokens.shape[1] <= 0
            or next_tokens.shape[2] != latent_dim
            or not next_tokens.is_floating_point()
        ):
            shape = tuple(next_tokens.shape) if torch.is_tensor(next_tokens) else type(next_tokens).__name__
            raise ValueError(f"rollout_step must return floating [1, N>0, {latent_dim}] tokens, got {shape}")
        next_tokens_finite = torch.isfinite(next_tokens).all()
        rollout_tokens_finite = torch.logical_and(rollout_tokens_finite, next_tokens_finite)
        raw_prefix = torch.cat([raw_prefix, next_tokens.detach()], dim=1)

    raise RuntimeError("unreachable sequential rollout state")
