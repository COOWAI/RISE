"""Continuous rollout-budget mapping and controller modules."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class BudgetProfile:
    """Concrete rollout budget produced from a continuous budget scalar."""

    rollout_future_steps: int


RolloutUpper = Union[int, str]


_LEGACY_SCHEDULE_KEYS = {
    "predictor_sampling_steps",
    "planner_inference_steps",
    "planner_num_candidates",
    "value_guidance_steps",
    "value_guidance_scale",
}


def _validate_rollout_pair(pair: Sequence[Any]) -> Tuple[int, RolloutUpper]:
    if len(pair) != 2:
        raise ValueError(f"rollout_future_steps schedule entry must contain [min, max], got {pair!r}")
    low = pair[0]
    high = pair[1]
    if isinstance(low, bool) or int(low) != float(low):
        raise ValueError(f"rollout_future_steps min must be an integer, got {pair!r}")
    low = int(low)
    if low < 0:
        raise ValueError(f"rollout_future_steps min must be >= 0, got {pair!r}")
    if isinstance(high, str):
        if high != "full":
            raise ValueError("rollout_future_steps max must be an integer or 'full', got " f"{high!r}")
        return low, high
    if isinstance(high, bool) or int(high) != float(high):
        raise ValueError(f"rollout_future_steps max must be an integer or 'full', got {pair!r}")
    high = int(high)
    if high < low:
        raise ValueError(f"rollout_future_steps schedule max must be >= min, got {pair!r}")
    return low, high


@dataclass(frozen=True)
class BudgetSchedule:
    """Map ``b in [0, 1]`` to an autoregressive future-rollout length."""

    rollout_future_steps: Tuple[int, RolloutUpper] = (1, "full")

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rollout_future_steps",
            _validate_rollout_pair(self.rollout_future_steps),
        )

    @classmethod
    def from_mapping(cls, schedule: Optional[Mapping[str, Sequence[Any]]]) -> "BudgetSchedule":
        if not schedule:
            return cls()
        legacy = sorted(set(schedule) & _LEGACY_SCHEDULE_KEYS)
        if legacy:
            raise ValueError(
                "budget_controller.schedule keys "
                f"{legacy} are no longer supported; use rollout_future_steps: [0, 'full'] or [1, 'full']"
            )
        known = {"rollout_future_steps"}
        unknown = sorted(set(schedule) - known)
        if unknown:
            raise ValueError(f"budget_controller.schedule has unknown keys {unknown}; valid keys are {sorted(known)}")
        defaults = cls()
        values = {name: getattr(defaults, name) for name in known}
        values.update({name: tuple(value) for name, value in schedule.items()})
        return cls(**values)

    def _resolve_max_future_steps(self, max_future_steps: Optional[int]) -> int:
        low, high = self.rollout_future_steps
        if high == "full":
            if max_future_steps is None:
                raise ValueError("max_future_steps is required when rollout_future_steps max is 'full'")
            resolved_high = int(max_future_steps)
        else:
            resolved_high = int(high)
            if max_future_steps is not None and resolved_high > int(max_future_steps):
                raise ValueError(
                    "rollout_future_steps max exceeds the available future horizon: "
                    f"max={resolved_high}, available={int(max_future_steps)}"
                )
        if resolved_high <= 0:
            raise ValueError(f"max_future_steps must be > 0, got {resolved_high}")
        if resolved_high < int(low):
            raise ValueError(
                "rollout_future_steps max must be >= min after resolving 'full', "
                f"got min={int(low)}, max={resolved_high}"
            )
        return resolved_high

    def profile(self, budget: float, *, max_future_steps: Optional[int] = None) -> BudgetProfile:
        budget = float(budget)
        if not 0.0 <= budget <= 1.0:
            raise ValueError(f"budget must be in [0, 1], got {budget}")
        low = int(self.rollout_future_steps[0])
        high = self._resolve_max_future_steps(max_future_steps)
        steps = int(torch.floor(torch.tensor(float(low) + budget * float(high - low) + 0.5)).item())
        steps = max(low, min(high, steps))
        return BudgetProfile(rollout_future_steps=steps)


@dataclass(frozen=True)
class BudgetControllerOutput:
    """Sampled or deterministic budget action from ``BudgetController``."""

    budget: torch.Tensor
    log_prob: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


def sample_beta_budget(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    deterministic: bool,
) -> BudgetControllerOutput:
    """Sample a continuous budget from Beta parameters produced by a controller forward."""
    if alpha.ndim != 1 or beta.ndim != 1 or alpha.shape != beta.shape:
        raise ValueError(
            f"alpha and beta must be matching [B] tensors, got {tuple(alpha.shape)} and {tuple(beta.shape)}"
        )
    if not torch.isfinite(alpha).all() or not torch.isfinite(beta).all():
        raise ValueError("alpha and beta must contain only finite values")
    if not torch.all(alpha > 0.0) or not torch.all(beta > 0.0):
        raise ValueError("alpha and beta must be strictly positive")

    dist = torch.distributions.Beta(alpha, beta)
    budget = alpha / (alpha + beta) if deterministic else dist.sample()
    budget = budget.clamp(1e-6, 1.0 - 1e-6)
    return BudgetControllerOutput(
        budget=budget,
        log_prob=dist.log_prob(budget),
        alpha=alpha,
        beta=beta,
    )


@dataclass(frozen=True)
class OracleBudgetSelection:
    """Best budget entry from a fixed-policy oracle sweep."""

    index: torch.Tensor
    budget: torch.Tensor
    utility: torch.Tensor


@dataclass(frozen=True)
class BudgetOracleRecord:
    """One fixed-policy budget sweep result for a scene."""

    scene_id: str
    budget: float
    profile: BudgetProfile
    task_score: float
    compute_cost: float
    utility: float


def compute_budget_utility(
    task_score: torch.Tensor,
    compute_cost: torch.Tensor,
    *,
    lambda_compute: float,
) -> torch.Tensor:
    """Compute oracle utility ``task_score - lambda_compute * compute_cost``."""
    if task_score.shape != compute_cost.shape:
        raise ValueError(f"task_score shape {tuple(task_score.shape)} != compute_cost {tuple(compute_cost.shape)}")
    lambda_compute = float(lambda_compute)
    if lambda_compute < 0.0:
        raise ValueError(f"lambda_compute must be >= 0, got {lambda_compute}")
    return task_score - lambda_compute * compute_cost


def select_oracle_budget(
    budgets: torch.Tensor,
    task_scores: torch.Tensor,
    compute_costs: torch.Tensor,
    *,
    lambda_compute: float,
) -> OracleBudgetSelection:
    """Select the best continuous budget from an offline fixed-policy sweep."""
    if budgets.ndim != 1:
        raise ValueError(f"budgets must be [G], got {tuple(budgets.shape)}")
    if task_scores.shape != budgets.shape or compute_costs.shape != budgets.shape:
        raise ValueError(
            "task_scores and compute_costs must match budgets shape "
            f"{tuple(budgets.shape)}, got {tuple(task_scores.shape)} and {tuple(compute_costs.shape)}"
        )
    if not torch.all((budgets >= 0.0) & (budgets <= 1.0)):
        raise ValueError("budgets must all be in [0, 1]")
    utility = compute_budget_utility(task_scores, compute_costs, lambda_compute=lambda_compute)
    index = utility.argmax()
    return OracleBudgetSelection(index=index, budget=budgets[index], utility=utility[index])


def collect_budget_oracle(
    *,
    scene_ids: Sequence[Any],
    budget_grid: Sequence[float],
    schedule: BudgetSchedule,
    evaluate_fn: Callable[[Any, float, BudgetProfile], Tuple[float, float]],
    lambda_compute: float,
    max_future_steps: Optional[int] = None,
) -> List[BudgetOracleRecord]:
    """Collect utility records for a fixed main policy over a budget grid.

    ``evaluate_fn`` is the only project-specific hook: it runs the frozen main
    policy on ``scene_id`` with ``profile`` and returns ``(task_score,
    compute_cost)``. This keeps oracle collection decoupled from PDMS/open-loop
    metric implementations.
    """
    if not scene_ids:
        raise ValueError("scene_ids must contain at least one scene")
    if not budget_grid:
        raise ValueError("budget_grid must contain at least one budget")
    lambda_compute = float(lambda_compute)
    if lambda_compute < 0.0:
        raise ValueError(f"lambda_compute must be >= 0, got {lambda_compute}")
    records: List[BudgetOracleRecord] = []
    for scene_id in scene_ids:
        for raw_budget in budget_grid:
            budget = float(raw_budget)
            profile = schedule.profile(budget, max_future_steps=max_future_steps)
            task_score, compute_cost = evaluate_fn(scene_id, budget, profile)
            task_score = float(task_score)
            compute_cost = float(compute_cost)
            utility = task_score - lambda_compute * compute_cost
            records.append(
                BudgetOracleRecord(
                    scene_id=str(scene_id),
                    budget=budget,
                    profile=profile,
                    task_score=task_score,
                    compute_cost=compute_cost,
                    utility=utility,
                )
            )
    return records


class BudgetController(nn.Module):
    """Lightweight Beta-policy controller for continuous compute budgets."""

    def __init__(
        self,
        *,
        latent_dim: int,
        feature_dim: int = 0,
        hidden_dim: int = 128,
        min_concentration: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_concentration = float(min_concentration)
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if self.feature_dim < 0:
            raise ValueError(f"feature_dim must be >= 0, got {feature_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if self.min_concentration <= 0.0:
            raise ValueError(f"min_concentration must be > 0, got {min_concentration}")
        in_dim = self.latent_dim + self.feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def _features(self, pooled_latent: torch.Tensor, cheap_features: Optional[torch.Tensor]) -> torch.Tensor:
        if pooled_latent.ndim != 2 or pooled_latent.shape[1] != self.latent_dim:
            raise ValueError(f"pooled_latent must be [B, {self.latent_dim}], got {tuple(pooled_latent.shape)}")
        if self.feature_dim == 0:
            if cheap_features is not None and cheap_features.numel() > 0:
                raise ValueError("cheap_features must be None/empty when feature_dim=0")
            return pooled_latent
        if cheap_features is None:
            raise ValueError(f"cheap_features is required when feature_dim={self.feature_dim}")
        if cheap_features.ndim != 2 or cheap_features.shape != (pooled_latent.shape[0], self.feature_dim):
            raise ValueError(f"cheap_features must be [B, {self.feature_dim}], got {tuple(cheap_features.shape)}")
        return torch.cat(
            [pooled_latent, cheap_features.to(device=pooled_latent.device, dtype=pooled_latent.dtype)], dim=1
        )

    def forward(
        self,
        pooled_latent: torch.Tensor,
        cheap_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(self._features(pooled_latent, cheap_features))
        alpha = F.softplus(raw[:, 0]) + self.min_concentration
        beta = F.softplus(raw[:, 1]) + self.min_concentration
        return alpha, beta

    def sample_budget(
        self,
        pooled_latent: torch.Tensor,
        cheap_features: Optional[torch.Tensor] = None,
        *,
        deterministic: bool,
    ) -> BudgetControllerOutput:
        alpha, beta = self.forward(pooled_latent, cheap_features)
        return sample_beta_budget(alpha, beta, deterministic=deterministic)


def load_budget_controller_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    expected_mode: Optional[str] = None,
) -> BudgetController:
    """Load a standalone Stage4 ``BudgetController`` checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Path produced by ``train_budget_controller_from_oracle``.
    device:
        Target device for inference.
    expected_mode:
        Optional checkpoint provenance mode that must match exactly.

    Returns
    -------
    BudgetController
        Frozen eval-mode controller.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"budget_controller.controller_checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"budget controller checkpoint must be a mapping, got {type(checkpoint).__name__}: {path}")
    if expected_mode is not None and checkpoint.get("mode") != expected_mode:
        raise ValueError(
            f"budget controller checkpoint mode must be {expected_mode!r}, " f"got {checkpoint.get('mode')!r}: {path}"
        )
    required_keys = ("controller", "latent_dim", "feature_dim", "hidden_dim", "min_concentration")
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise RuntimeError(f"budget controller checkpoint missing key(s) {missing}: {path}")

    controller = BudgetController(
        latent_dim=int(checkpoint["latent_dim"]),
        feature_dim=int(checkpoint["feature_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        min_concentration=float(checkpoint["min_concentration"]),
    ).to(device)
    state = {
        (key[7:] if key.startswith("module.") else key): value for key, value in dict(checkpoint["controller"]).items()
    }
    controller.load_state_dict(state)
    controller.eval()
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller


def resolve_controller_budget_profile(
    controller: BudgetController,
    z_obs: torch.Tensor,
    *,
    config: Any,
    deterministic: bool = True,
    max_future_steps: int,
) -> Tuple[torch.Tensor, BudgetProfile]:
    """Predict a continuous budget from observed latent tokens and map it to rollout steps.

    The current validation runtime applies one rollout length at batch scope.
    To avoid silently applying one sample's budget to other samples, controller
    evaluation is intentionally restricted to batch size 1.
    """
    if z_obs.ndim != 3:
        raise ValueError(f"z_obs must be [B, N, D], got {tuple(z_obs.shape)}")
    if int(z_obs.shape[0]) != 1:
        raise ValueError(
            "controller-controlled runtime currently requires batch size 1 because rollout length is batch-global; "
            f"got batch size {int(z_obs.shape[0])}"
        )
    if int(z_obs.shape[2]) != int(controller.latent_dim):
        raise ValueError(
            f"z_obs last dim must match controller latent_dim={controller.latent_dim}, got {z_obs.shape[2]}"
        )

    budget_config = getattr(config, "budget_controller", None)
    if budget_config is None:
        raise ValueError("resolve_controller_budget_profile requires config.budget_controller")
    feature_dim = int(getattr(budget_config, "feature_dim", 0))
    if feature_dim != int(controller.feature_dim):
        raise ValueError(
            f"budget_controller.feature_dim={feature_dim} does not match checkpoint "
            f"feature_dim={controller.feature_dim}"
        )
    if feature_dim != 0:
        raise ValueError("controller-controlled runtime currently supports budget_controller.feature_dim=0 only")
    train_config = getattr(config, "train", None)
    predictor_type = getattr(train_config, "predictor_type", None)
    if predictor_type not in (None, "ac_transformer"):
        raise ValueError(
            "rollout budget controller currently supports train.predictor_type='ac_transformer' only, "
            f"got {predictor_type!r}"
        )

    pooled_latent = z_obs.detach().float().mean(dim=1)
    with torch.no_grad():
        output = controller.sample_budget(pooled_latent, deterministic=deterministic)
    schedule = BudgetSchedule.from_mapping(getattr(budget_config, "schedule", None))
    profile = schedule.profile(float(output.budget[0].detach().cpu()), max_future_steps=max_future_steps)
    return output.budget.detach(), profile


def budget_controller_bc_loss(
    controller: BudgetController,
    pooled_latent: torch.Tensor,
    target_budget: torch.Tensor,
    cheap_features: Optional[torch.Tensor] = None,
    *,
    mse_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Supervised oracle-distillation loss for the continuous budget policy."""
    if target_budget.ndim != 1 or target_budget.shape[0] != pooled_latent.shape[0]:
        raise ValueError(f"target_budget must be [B={pooled_latent.shape[0]}], got {tuple(target_budget.shape)}")
    if not torch.all((target_budget >= 0.0) & (target_budget <= 1.0)):
        raise ValueError("target_budget entries must be in [0, 1]")
    mse_weight = float(mse_weight)
    if mse_weight < 0.0:
        raise ValueError(f"mse_weight must be >= 0, got {mse_weight}")
    alpha, beta = controller(pooled_latent, cheap_features)
    dist = torch.distributions.Beta(alpha, beta)
    target = target_budget.to(device=pooled_latent.device, dtype=pooled_latent.dtype).clamp(1e-6, 1.0 - 1e-6)
    pred_budget = alpha / (alpha + beta)
    nll = -dist.log_prob(target).mean()
    mse = F.mse_loss(pred_budget, target)
    loss = nll + mse_weight * mse
    return {
        "loss": loss,
        "nll": nll,
        "mse": mse,
        "pred_budget": pred_budget,
        "target_budget": target,
    }


def budget_controller_grpo_loss(
    log_prob: torch.Tensor,
    reward: torch.Tensor,
    *,
    group_ids: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """GRPO/contextual-bandit loss with per-scene group baselines."""
    if log_prob.ndim != 1 or reward.ndim != 1 or log_prob.shape != reward.shape:
        raise ValueError(
            f"log_prob and reward must both be [B], got {tuple(log_prob.shape)} and {tuple(reward.shape)}"
        )
    if group_ids is None:
        baseline = reward.mean().expand_as(reward)
    else:
        if group_ids.ndim != 1 or group_ids.shape != reward.shape:
            raise ValueError(f"group_ids must be [B]={tuple(reward.shape)}, got {tuple(group_ids.shape)}")
        baseline = torch.empty_like(reward)
        for group_id in torch.unique(group_ids):
            mask = group_ids == group_id
            baseline[mask] = reward[mask].mean()
    advantage = reward - baseline
    loss = -(log_prob * advantage.detach()).mean()
    return {
        "loss": loss,
        "advantage": advantage,
        "baseline": baseline,
    }
