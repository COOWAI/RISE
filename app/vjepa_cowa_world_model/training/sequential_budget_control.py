"""Sequential compute-of-imagination gate and supervision utilities."""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import CVOI_GATE_FEATURE_MODES

CVOI_GUIDANCE_STEPS = 2
CVOI_MAX_HORIZON = 3
CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA = "cvoi_gate_online_features_v2"
CVOI_FORMAL_V2_GATE_FEATURE_MODES = frozenset({"full", "without_field", "without_stop"})
CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES = frozenset(
    {"full", "without_field", "without_stop", "without_value_summary"}
)
_SCALAR_FEATURE_COUNT = 7


@dataclass(frozen=True)
class CvoiTargets:
    """Oracle utilities and non-terminal marginal-compute labels."""

    utility: torch.Tensor
    delta_utility: torch.Tensor
    continue_target: torch.Tensor


@dataclass(frozen=True)
class SequentialGateLoss:
    """Loss terms used to distill an oracle utility curve."""

    loss: torch.Tensor
    classification: torch.Tensor
    regression: torch.Tensor
    sample_weight: torch.Tensor


def extract_prefix_gate_values(output: object) -> Dict[str, torch.Tensor]:
    """Map a dual-value prefix output to current online Gate scalars.

    ``field_values`` has no observed-only entry by contract. At ``h=0`` the
    fixed Guidance policy is skipped, so its Field feature is the explicit zero
    sentinel; the observed hidden still provides the absolute Stop value.
    """

    if not hasattr(output, "field_values") or not hasattr(output, "stop_values"):
        raise TypeError("prefix value output must expose field_values and stop_values")
    field_values = output.field_values
    stop_values = output.stop_values
    if not torch.is_tensor(field_values) or field_values.ndim != 2:
        raise ValueError("field_values must be a [B, F] tensor")
    if (
        not torch.is_tensor(stop_values)
        or stop_values.ndim != 2
        or stop_values.shape != (field_values.shape[0], field_values.shape[1] + 1)
    ):
        raise ValueError("stop_values must have shape [B, F+1] aligned with field_values")
    if not field_values.dtype.is_floating_point or not stop_values.dtype.is_floating_point:
        raise ValueError("field_values and stop_values must be floating point")
    if not torch.isfinite(field_values).all() or not torch.isfinite(stop_values).all():
        raise ValueError("field_values and stop_values must contain only finite values")
    stop_value = stop_values[:, -1]
    if field_values.shape[1] == 0:
        field_value = torch.zeros_like(stop_value)
        previous_stop_value = stop_value
    else:
        field_value = field_values[:, -1]
        previous_stop_value = stop_values[:, -2]
    return {
        "field_value": field_value,
        "stop_value": stop_value,
        "previous_stop_value": previous_stop_value,
    }


def _finite_vector(name: str, value: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    if not torch.is_tensor(value) or value.shape != (batch_size,):
        shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
        raise ValueError(f"{name} must have shape [{batch_size}], got {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def compute_cvoi_targets(
    task_scores: torch.Tensor,
    compute_costs: torch.Tensor,
    lambda_compute: torch.Tensor,
) -> CvoiTargets:
    """Build ``max-future minus stop-now`` labels from complete horizon curves."""

    if task_scores.ndim != 2 or task_scores.shape[1] < 2:
        raise ValueError(f"task_scores must be [B, H+1] with H>=1, got {tuple(task_scores.shape)}")
    if compute_costs.shape != task_scores.shape:
        raise ValueError(
            f"compute_costs must match task_scores shape {tuple(task_scores.shape)}, got {tuple(compute_costs.shape)}"
        )
    if not torch.isfinite(task_scores).all() or not torch.isfinite(compute_costs).all():
        raise ValueError("task_scores and compute_costs must contain only finite values")
    if torch.any(compute_costs < 0.0):
        raise ValueError("compute_costs must be non-negative")
    if torch.any(compute_costs[:, 1:] < compute_costs[:, :-1]):
        raise ValueError("compute_costs must be non-decreasing along the horizon axis")

    batch_size = int(task_scores.shape[0])
    lambda_compute = _finite_vector("lambda_compute", lambda_compute, batch_size=batch_size)
    if torch.any(lambda_compute < 0.0):
        raise ValueError("lambda_compute must be non-negative")

    lambda_value = lambda_compute.to(device=task_scores.device, dtype=task_scores.dtype).unsqueeze(1)
    utility = task_scores - lambda_value * compute_costs.to(device=task_scores.device, dtype=task_scores.dtype)
    future_utility = utility[:, 1:]
    best_future = torch.flip(torch.cummax(torch.flip(future_utility, dims=(1,)), dim=1).values, dims=(1,))
    delta_utility = best_future - utility[:, :-1]
    return CvoiTargets(
        utility=utility,
        delta_utility=delta_utility,
        continue_target=delta_utility > 0.0,
    )


def build_sequential_gate_features(
    *,
    pooled_observed: torch.Tensor,
    pooled_prefix: torch.Tensor,
    field_value: torch.Tensor,
    stop_value: torch.Tensor,
    previous_stop_value: torch.Tensor,
    horizon: torch.Tensor,
    max_horizon: int,
    current_cost: torch.Tensor,
    next_cost: torch.Tensor,
    lambda_compute: torch.Tensor,
) -> torch.Tensor:
    """Assemble the fixed online-only feature schema consumed by the gate."""

    if pooled_observed.ndim != 2:
        raise ValueError(f"pooled_observed must be [B, D], got {tuple(pooled_observed.shape)}")
    if pooled_prefix.shape != pooled_observed.shape:
        raise ValueError(
            f"pooled_prefix must match pooled_observed shape {tuple(pooled_observed.shape)}, "
            f"got {tuple(pooled_prefix.shape)}"
        )
    if not torch.isfinite(pooled_observed).all() or not torch.isfinite(pooled_prefix).all():
        raise ValueError("pooled observed/prefix features must contain only finite values")
    batch_size = int(pooled_observed.shape[0])
    max_horizon = int(max_horizon)
    if max_horizon <= 0:
        raise ValueError(f"max_horizon must be positive, got {max_horizon}")

    scalar_values = []
    for name, value in (
        ("field_value", field_value),
        ("stop_value", stop_value),
        ("previous_stop_value", previous_stop_value),
        ("current_cost", current_cost),
        ("next_cost", next_cost),
        ("lambda_compute", lambda_compute),
    ):
        scalar_values.append(_finite_vector(name, value, batch_size=batch_size))
    if not torch.is_tensor(horizon) or horizon.shape != (batch_size,):
        raise ValueError(f"horizon must have shape [{batch_size}], got {tuple(horizon.shape)}")
    if torch.any(horizon < 0) or torch.any(horizon > max_horizon):
        raise ValueError(f"horizon entries must be in [0, {max_horizon}]")
    if torch.any(current_cost < 0.0) or torch.any(next_cost < current_cost):
        raise ValueError("current/next compute costs must be non-negative and non-decreasing")
    if torch.any(lambda_compute < 0.0):
        raise ValueError("lambda_compute must be non-negative")

    dtype = pooled_observed.dtype
    device = pooled_observed.device
    scalars = torch.stack(
        [
            field_value,
            stop_value,
            stop_value - previous_stop_value,
            horizon.to(dtype=dtype) / float(max_horizon),
            current_cost,
            next_cost,
            lambda_compute,
        ],
        dim=1,
    ).to(device=device, dtype=dtype)
    return torch.cat([pooled_observed, pooled_prefix, scalars], dim=1)


def build_lambda_independent_sequential_gate_features(
    *,
    pooled_observed: torch.Tensor,
    pooled_prefix: torch.Tensor,
    field_value: torch.Tensor,
    stop_value: torch.Tensor,
    previous_stop_value: torch.Tensor,
    horizon: torch.Tensor,
    max_horizon: int,
    current_cost: torch.Tensor,
    next_cost: torch.Tensor,
) -> torch.Tensor:
    """Build the online Gate state without the lambda-compute conditioning column.

    Navtrain teacher traces are shared by every lambda in the later CPU utility
    join.  The regular online Gate schema remains unchanged and still appends
    lambda as its final scalar.
    """

    features = build_sequential_gate_features(
        pooled_observed=pooled_observed,
        pooled_prefix=pooled_prefix,
        field_value=field_value,
        stop_value=stop_value,
        previous_stop_value=previous_stop_value,
        horizon=horizon,
        max_horizon=max_horizon,
        current_cost=current_cost,
        next_cost=next_cost,
        lambda_compute=stop_value.new_zeros((int(pooled_observed.shape[0]),)),
    )
    return features[:, :-1]


def apply_cvoi_gate_feature_mask(
    features: torch.Tensor,
    *,
    latent_dim: int,
    mode: str,
) -> torch.Tensor:
    """Mask registered Value columns without changing Gate input capacity."""

    if mode not in CVOI_GATE_FEATURE_MODES:
        raise ValueError(f"CVoI gate feature mode must be one of {sorted(CVOI_GATE_FEATURE_MODES)}, got {mode!r}")
    expected_dim = 2 * int(latent_dim) + _SCALAR_FEATURE_COUNT
    if features.ndim != 2 or int(features.shape[1]) != expected_dim:
        raise ValueError(f"CVoI gate feature schema requires [B, {expected_dim}], got {tuple(features.shape)}")
    if mode == "full":
        return features
    masked = features.clone()
    scalar_start = 2 * int(latent_dim)
    if mode == "gate_no_stop_value":
        masked[:, scalar_start + 1] = 0.0
    else:
        masked[:, scalar_start : scalar_start + 3] = 0.0
    return masked


def apply_cvoi_formal_v2_gate_feature_mask(
    features: torch.Tensor,
    *,
    latent_dim: int,
    mode: str,
) -> torch.Tensor:
    """Apply a registered Formal v2 Value-feature ablation.

    This API is intentionally separate from :func:`apply_cvoi_gate_feature_mask`.
    The legacy ``gate_no_stop_value`` mode masks only the Stop level and must
    retain that historical behavior, whereas Formal v2 ``without_stop`` masks
    both the Stop level and Stop slope.
    """

    if mode not in CVOI_FORMAL_V2_GATE_FEATURE_MODES:
        raise ValueError(
            "CVoI Formal v2 Gate feature mode must be one of "
            f"{sorted(CVOI_FORMAL_V2_GATE_FEATURE_MODES)}, got {mode!r}"
        )
    latent_dim = int(latent_dim)
    expected_dim = 2 * latent_dim + _SCALAR_FEATURE_COUNT
    if features.ndim != 2 or int(features.shape[1]) != expected_dim:
        raise ValueError(
            f"CVoI Formal v2 Gate feature schema requires [B, {expected_dim}], got {tuple(features.shape)}"
        )
    if mode == "full":
        return features

    masked = features.clone()
    scalar_start = 2 * latent_dim
    if mode == "without_field":
        masked[:, scalar_start] = 0.0
    else:
        masked[:, scalar_start + 1 : scalar_start + 3] = 0.0
    return masked


def apply_cvoi_formal_v2_navsim_e120_gate_feature_mask(
    features: torch.Tensor,
    *,
    latent_dim: int,
    mode: str,
) -> torch.Tensor:
    """Apply the NavSim-e120 extension of the frozen Formal-v2 Gate masks."""

    if mode not in CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES:
        raise ValueError(
            "CVoI Formal v2 NavSim-e120 Gate feature mode must be one of "
            f"{sorted(CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES)}, got {mode!r}"
        )
    if mode != "without_value_summary":
        return apply_cvoi_formal_v2_gate_feature_mask(
            features,
            latent_dim=latent_dim,
            mode=mode,
        )

    validated = apply_cvoi_formal_v2_gate_feature_mask(
        features,
        latent_dim=latent_dim,
        mode="full",
    )
    masked = validated.clone()
    scalar_start = 2 * int(latent_dim)
    masked[:, scalar_start : scalar_start + 3] = 0.0
    return masked


class SequentialRolloutGate(nn.Module):
    """Predict marginal rollout utility with a monotone compute penalty.

    The final feature is ``lambda_compute``. It is deliberately excluded from
    the shared state encoder and enters only as ``-softplus(slope) * lambda``.
    Consequently, increasing the compute penalty cannot increase the predicted
    marginal utility for an otherwise identical online state.
    """

    def __init__(self, *, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        self.feature_dim = 2 * self.latent_dim + _SCALAR_FEATURE_COUNT
        self.net = nn.Sequential(
            nn.LayerNorm(self.feature_dim - 1),
            nn.Linear(self.feature_dim - 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must be [B, {self.feature_dim}], got {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise ValueError("gate features must contain only finite values")
        state_features = features[:, :-1]
        lambda_compute = features[:, -1]
        benefit_and_slope = self.net(state_features)
        benefit = benefit_and_slope[:, 0]
        penalty_slope = F.softplus(benefit_and_slope[:, 1])
        return benefit - penalty_slope * lambda_compute

    @staticmethod
    def should_roll(predicted_delta: torch.Tensor, *, horizon: int, max_horizon: int) -> torch.Tensor:
        if predicted_delta.ndim != 1 or not torch.isfinite(predicted_delta).all():
            raise ValueError(f"predicted_delta must be a finite [B] tensor, got {tuple(predicted_delta.shape)}")
        horizon = int(horizon)
        max_horizon = int(max_horizon)
        if max_horizon <= 0 or horizon < 0 or horizon > max_horizon:
            raise ValueError(f"invalid horizon/max_horizon pair: {horizon}/{max_horizon}")
        if horizon == max_horizon:
            return torch.zeros_like(predicted_delta, dtype=torch.bool)
        return predicted_delta > 0.0


def sequential_gate_loss(
    predicted_delta: torch.Tensor,
    *,
    target_delta: torch.Tensor,
    continue_target: torch.Tensor,
    temperature: float = 0.05,
    regression_weight: float = 0.5,
) -> SequentialGateLoss:
    """Combine regret-weighted sign classification with utility regression."""

    if predicted_delta.ndim != 1 or target_delta.shape != predicted_delta.shape:
        raise ValueError(
            f"predicted_delta and target_delta must be matching [B] tensors, got "
            f"{tuple(predicted_delta.shape)}/{tuple(target_delta.shape)}"
        )
    if continue_target.shape != predicted_delta.shape or continue_target.dtype != torch.bool:
        raise ValueError("continue_target must be a bool tensor matching predicted_delta")
    if not torch.isfinite(predicted_delta).all() or not torch.isfinite(target_delta).all():
        raise ValueError("predicted_delta and target_delta must contain only finite values")
    temperature = float(temperature)
    regression_weight = float(regression_weight)
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if regression_weight < 0.0:
        raise ValueError(f"regression_weight must be non-negative, got {regression_weight}")

    target_delta = target_delta.to(device=predicted_delta.device, dtype=predicted_delta.dtype)
    target_class = continue_target.to(device=predicted_delta.device, dtype=predicted_delta.dtype)
    sample_weight = torch.clamp(torch.abs(target_delta) / temperature, min=1.0, max=10.0)
    per_sample_classification = F.binary_cross_entropy_with_logits(
        predicted_delta / temperature,
        target_class,
        reduction="none",
    )
    classification = (sample_weight * per_sample_classification).mean()
    regression = F.smooth_l1_loss(predicted_delta, target_delta)
    return SequentialGateLoss(
        loss=classification + regression_weight * regression,
        classification=classification,
        regression=regression,
        sample_weight=sample_weight,
    )
