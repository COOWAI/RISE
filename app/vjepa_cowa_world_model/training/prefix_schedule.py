"""Exact prefix-length distributions shared by predictor and planner training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class PrefixDistribution:
    """Resolved categorical distribution over cumulative prefix lengths ``h``."""

    horizon_steps: int
    prefix_steps: Tuple[int, ...]
    probabilities: Tuple[float, ...]

    def probability_by_prefix_steps(self) -> dict[int, float]:
        """Return a serializable theoretical distribution for logging/validation."""
        return dict(zip(self.prefix_steps, self.probabilities))


@dataclass(frozen=True)
class PrefixSample:
    """One rank-local batch prefix draw plus the distribution that produced it."""

    prefix_steps: int
    distribution: PrefixDistribution

    @property
    def future_start_step(self) -> int:
        """Prefix v2 is cumulative, so its future-relative start is always zero."""
        return 0

    def rollout_end_step(self, *, num_observed_steps: int) -> int:
        """Return the exclusive timeline rollout end for this cumulative prefix."""
        observed_steps = int(num_observed_steps)
        if observed_steps < 0:
            raise ValueError(f"num_observed_steps must be >= 0, got {num_observed_steps}")
        return observed_steps + self.prefix_steps


def normalize_horizon_probabilities(
    horizon_probabilities: object,
    *,
    horizon_steps: Optional[int] = None,
    field_name: str = "horizon_probabilities",
) -> Tuple[float, ...]:
    """Validate and normalize an explicit categorical distribution over ``h=0..H``."""

    if not isinstance(horizon_probabilities, (list, tuple)):
        raise ValueError(f"{field_name} must be a list or tuple, got {horizon_probabilities!r}")
    probabilities = []
    for index, value in enumerate(horizon_probabilities):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{field_name}[{index}] must be a finite non-negative real number, got {value!r}")
        probability = float(value)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(f"{field_name}[{index}] must be a finite non-negative real number, got {value!r}")
        probabilities.append(probability)
    if horizon_steps is not None:
        expected_length = int(horizon_steps) + 1
        if len(probabilities) != expected_length:
            raise ValueError(
                f"{field_name} must have length H+1={expected_length} for H={horizon_steps}, "
                f"got {len(probabilities)}"
            )
    probability_sum = math.fsum(probabilities)
    if not math.isclose(probability_sum, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{field_name} must sum to 1, got {probability_sum}")
    return tuple(probabilities)


def resolve_prefix_distribution(
    *,
    enabled: bool,
    horizon_steps: int,
    full_prefix_prob: float,
    min_prefix_steps: int,
    max_non_full_prefix_steps: Optional[int],
    horizon_probabilities: Optional[Sequence[float]] = None,
) -> PrefixDistribution:
    """Resolve the exact categorical prefix distribution without sampling.

    An explicit ``horizon_probabilities`` vector owns the full ``h=0..H``
    distribution. Otherwise, ``full_prefix_prob`` is assigned only to ``h == H``
    and the remaining mass is uniform over the configured non-full support.
    """
    horizon = int(horizon_steps)
    if horizon <= 0:
        raise ValueError(f"horizon_steps must be positive, got {horizon_steps}")

    explicit_probabilities = None
    if horizon_probabilities is not None:
        explicit_probabilities = normalize_horizon_probabilities(
            horizon_probabilities,
            horizon_steps=horizon,
        )

    if not bool(enabled):
        return PrefixDistribution(horizon_steps=horizon, prefix_steps=(horizon,), probabilities=(1.0,))

    if explicit_probabilities is not None:
        return PrefixDistribution(
            horizon_steps=horizon,
            prefix_steps=tuple(range(horizon + 1)),
            probabilities=explicit_probabilities,
        )

    full_prob = float(full_prefix_prob)
    if not math.isfinite(full_prob) or not 0.0 <= full_prob <= 1.0:
        raise ValueError(f"full_prefix_prob must be finite and in [0, 1], got {full_prefix_prob}")

    min_prefix = int(min_prefix_steps)
    if min_prefix < 0:
        raise ValueError(f"min_prefix_steps must be >= 0, got {min_prefix_steps}")

    max_non_full = horizon - 1 if max_non_full_prefix_steps is None else int(max_non_full_prefix_steps)
    if max_non_full < 0:
        raise ValueError("max_non_full_prefix_steps must be >= 0 when set, " f"got {max_non_full_prefix_steps}")
    if max_non_full >= horizon:
        raise ValueError(
            "max_non_full_prefix_steps must be < horizon_steps; "
            f"got max_non_full_prefix_steps={max_non_full}, horizon_steps={horizon}"
        )

    if full_prob == 1.0:
        return PrefixDistribution(horizon_steps=horizon, prefix_steps=(horizon,), probabilities=(1.0,))

    if min_prefix >= horizon:
        raise ValueError(
            "non-full prefix support is empty while full_prefix_prob < 1: "
            f"min_prefix_steps={min_prefix}, horizon_steps={horizon}"
        )
    if min_prefix > max_non_full:
        raise ValueError(
            "min_prefix_steps must be <= max_non_full_prefix_steps; "
            f"got min_prefix_steps={min_prefix}, max_non_full_prefix_steps={max_non_full}"
        )

    non_full_steps = tuple(range(min_prefix, max_non_full + 1))
    if not non_full_steps:
        raise ValueError("non-full prefix support is empty while full_prefix_prob < 1")
    non_full_prob = (1.0 - full_prob) / len(non_full_steps)
    return PrefixDistribution(
        horizon_steps=horizon,
        prefix_steps=non_full_steps + (horizon,),
        probabilities=(non_full_prob,) * len(non_full_steps) + (full_prob,),
    )


def sample_prefix(
    distribution: PrefixDistribution,
    *,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> PrefixSample:
    """Draw one prefix length for a rank-local batch."""
    probabilities = torch.tensor(distribution.probabilities, dtype=torch.float64, device=device)
    sampled_index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    return PrefixSample(
        prefix_steps=distribution.prefix_steps[sampled_index],
        distribution=distribution,
    )


def validate_sampled_h0_context(
    *,
    prefix_steps: int,
    use_z_context: bool,
    z_context: Optional[torch.Tensor],
    use_observed_tokens: bool,
    z_observed: Optional[torch.Tensor],
    use_action_history: bool,
    action_history: Optional[torch.Tensor],
) -> None:
    """Require an actual non-future runtime tensor when a planner samples h=0."""
    if int(prefix_steps) != 0:
        return

    def _is_enabled_nonempty_tensor(enabled: bool, value: Optional[torch.Tensor]) -> bool:
        return bool(
            enabled
            and torch.is_tensor(value)
            and value.ndim >= 2
            and int(value.shape[0]) > 0
            and int(value.shape[1]) > 0
            and value.numel() > 0
        )

    has_runtime_context = (
        _is_enabled_nonempty_tensor(use_z_context, z_context)
        or _is_enabled_nonempty_tensor(use_observed_tokens, z_observed)
        or _is_enabled_nonempty_tensor(use_action_history, action_history)
    )
    if not has_runtime_context:
        raise ValueError(
            "planner internal sampled h=0 requires at least one enabled context source with a provided, non-empty "
            "runtime tensor: use_z_context + z_context, observed_token_mode + z_observed, or "
            "use_action_history + action_history"
        )
