"""Latent-side value guidance for predictor-to-planner features."""

import math
from typing import Any, Dict, Optional, Tuple

import torch

CVOI_LEGACY_EVALUATION_GUIDANCE_STEPS = (1, 2, 3, 4)
CVOI_EVALUATION_GUIDANCE_STEPS = (1, 2, 4, 8)
CVOI_EVALUATION_GUIDANCE_OVERRIDES = (0,) + CVOI_EVALUATION_GUIDANCE_STEPS


def cvoi_evaluation_guidance_steps(config: Any, *, include_disabled: bool) -> tuple[int, ...]:
    """Return the evaluation K grid owned by the configured CVoI protocol."""

    cvoi = getattr(config, "cvoi", None)
    protocol_version = str(getattr(cvoi, "protocol_version", "legacy_v1"))
    if protocol_version == "formal_v2_navsim_e120_h4_v3":
        return CVOI_EVALUATION_GUIDANCE_OVERRIDES if include_disabled else CVOI_EVALUATION_GUIDANCE_STEPS
    if protocol_version == "legacy_v1":
        return CVOI_LEGACY_EVALUATION_GUIDANCE_STEPS
    raise ValueError(f"unsupported CVoI protocol_version for Guidance evaluation: {protocol_version!r}")


def _guidance_cfg(config: Any) -> Any:
    return getattr(config, "value_guidance", config)


def reduce_temporal_values(values: torch.Tensor, objective: str, gamma: float = 0.99) -> torch.Tensor:
    """Reduce per-step value predictions to one scalar objective.

    Parameters
    ----------
    values    : [B, F] per-future-step values.
    objective : one of ``last``, ``mean``, or ``discounted``.
    gamma     : discount used by ``discounted``.

    Returns
    -------
    torch.Tensor:
        Scalar objective, higher is better.
    """
    if values.ndim != 2:
        raise ValueError(f"values must be [B, F], got {tuple(values.shape)}")
    if values.shape[1] < 1:
        raise ValueError("values must contain at least one future step")
    objective = str(objective)
    if objective == "last":
        return values[:, -1].mean()
    if objective == "mean":
        return values.mean()
    if objective == "discounted":
        gamma = float(gamma)
        if not 0.0 < gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        discounts = torch.pow(
            values.new_full((values.shape[1],), gamma),
            torch.arange(values.shape[1], device=values.device, dtype=values.dtype),
        )
        return (values * discounts.view(1, -1)).sum(dim=1).mean()
    raise ValueError("objective must be one of {'last', 'mean', 'discounted'}, got " f"{objective!r}")


def _clip_delta_by_token_norm(delta: torch.Tensor, max_delta_norm: float) -> torch.Tensor:
    if max_delta_norm <= 0.0:
        raise ValueError(f"max_delta_norm must be > 0, got {max_delta_norm}")
    norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.clamp(float(max_delta_norm) / norms, max=1.0)
    return delta * scale


def should_apply_value_guidance(
    z_future: torch.Tensor,
    *,
    value_guidance_enabled: bool,
    allow_empty_rollout_skip: bool = False,
) -> bool:
    """Return whether latent value guidance should run for this future latent batch.

    Empty future latent is valid only when dynamic rollout intentionally sampled
    rollout_future_steps=0. In that case the caller can skip guidance and still
    train/evaluate the planner on non-future conditions. Otherwise empty z_future
    is a configuration/runtime error.
    """

    if not bool(value_guidance_enabled):
        return False
    if z_future.ndim not in (3, 4):
        raise ValueError(f"z_future must be [B, N, D] or [B, F, T, D], got {tuple(z_future.shape)}")
    future_length = int(z_future.shape[1])
    if future_length > 0:
        return True
    if bool(allow_empty_rollout_skip):
        return False
    raise ValueError("value_guidance is not defined for rollout_future_steps=0 (empty z_ar)")


def apply_latent_value_guidance(
    z_future: torch.Tensor,
    value_head: torch.nn.Module,
    *,
    tokens_per_frame: int,
    config: Any,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply local value-gradient ascent to predictor future latent tokens.

    The optimized variable is a local ``delta`` around ``z_future``. The base
    latent is detached so guidance cannot update predictor parameters.

    Parameters
    ----------
    z_future         : [B, N, D] or [B, F, T, D] planner-conditioning latent.
    value_head       : module returning per-step values [B, F].
    tokens_per_frame : token grouping for flat [B, N, D] inputs.
    config           : TrainingConfig or ValueGuidanceConfig.

    Returns
    -------
    (guided_z, diagnostics)
    """
    cfg = _guidance_cfg(config)
    if not bool(getattr(cfg, "enabled", False)):
        return z_future, {
            "guidance_steps": 0.0,
            "delta_norm": 0.0,
            "value_before": 0.0,
            "value_after": 0.0,
        }
    if z_future.ndim not in (3, 4):
        raise ValueError(f"z_future must be [B, N, D] or [B, F, T, D], got {tuple(z_future.shape)}")
    tokens_per_frame = int(tokens_per_frame)
    if tokens_per_frame <= 0:
        raise ValueError(f"tokens_per_frame must be > 0, got {tokens_per_frame}")
    steps = int(getattr(cfg, "steps", 1))
    step_size = float(getattr(cfg, "step_size", 0.05))
    max_delta_norm = float(getattr(cfg, "max_delta_norm", 0.25))
    objective = str(getattr(cfg, "objective", "last"))
    gamma = float(getattr(config, "value_planning", config).gamma) if hasattr(config, "value_planning") else 0.99
    detach_output = bool(getattr(cfg, "detach_output", True))
    if steps < 1:
        raise ValueError(f"value_guidance.steps must be >= 1, got {steps}")
    if step_size < 0.0:
        raise ValueError(f"value_guidance.step_size must be >= 0, got {step_size}")
    if max_delta_norm <= 0.0:
        raise ValueError(f"value_guidance.max_delta_norm must be > 0, got {max_delta_norm}")

    base = z_future.detach()
    delta = torch.zeros_like(base, requires_grad=True)

    def _score(candidate: torch.Tensor) -> torch.Tensor:
        values = value_head(candidate, tokens_per_frame=tokens_per_frame)
        return reduce_temporal_values(values, objective=objective, gamma=gamma)

    with torch.enable_grad():
        value_before = _score(base).detach()
        for _ in range(steps):
            candidate = base + delta
            score = _score(candidate)
            if not score.requires_grad:
                raise RuntimeError("value guidance objective has no gradient; value_head path is detached")
            (grad,) = torch.autograd.grad(score, delta, allow_unused=True)
            if grad is None:
                raise RuntimeError("value guidance objective produced no gradient for latent delta")
            if not torch.isfinite(grad).all():
                raise RuntimeError("value guidance gradient contains NaN/Inf")
            with torch.no_grad():
                next_delta = delta + step_size * grad
                next_delta = _clip_delta_by_token_norm(next_delta, max_delta_norm=max_delta_norm)
            delta = next_delta.requires_grad_(True)
        guided = base + delta
        value_after = _score(guided).detach()

    diagnostics = {
        "guidance_steps": float(steps),
        "delta_norm": float(delta.detach().norm(dim=-1).max()),
        "value_before": float(value_before),
        "value_after": float(value_after),
    }
    if detach_output:
        guided = guided.detach()
    return guided, diagnostics


def apply_cvoi_latent_value_guidance(
    z_observed: torch.Tensor,
    z_future: torch.Tensor,
    dual_value_model: torch.nn.Module,
    *,
    tokens_per_frame: int,
    config: Any,
    evaluation_guidance_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Apply the fixed CVoI ``STOP(2)`` field-value guidance policy.

    Unlike legacy temporal guidance, this path evaluates a prefix-aware dual
    value model with detached observed and future base latents. Only a local
    future delta is optimized, and the scalar objective is explicitly the last
    entry of ``PrefixValueOutput.field_values``.

    Parameters
    ----------
    z_observed       : [B, N_obs, D] or [B, O, T, D] observed latent tokens.
    z_future         : [B, N_future, D] or [B, F, T, D] future latent tokens.
    dual_value_model : module returning an object with ``field_values [B, F]``.
    tokens_per_frame : spatial token count for flat inputs.
    config           : object containing ``value_guidance`` or the guidance config itself.

    Returns
    -------
    (guided_future, diagnostics)
    """

    cfg = _guidance_cfg(config)
    disabled_diagnostics = {
        "guidance_steps": 0.0,
        "guidance_skipped_h0": 0.0,
        "delta_norm": 0.0,
        "field_value_before": 0.0,
        "field_value_after": 0.0,
    }
    if not bool(getattr(cfg, "enabled", False)):
        return z_future, disabled_diagnostics
    if z_observed.ndim not in (3, 4):
        raise ValueError(f"z_observed must be [B, N, D] or [B, O, T, D], got {tuple(z_observed.shape)}")
    if z_future.ndim not in (3, 4):
        raise ValueError(f"z_future must be [B, N, D] or [B, F, T, D], got {tuple(z_future.shape)}")
    if z_observed.shape[0] != z_future.shape[0]:
        raise ValueError(f"z_observed batch {z_observed.shape[0]} does not match z_future batch {z_future.shape[0]}")
    if z_observed.shape[-1] != z_future.shape[-1]:
        raise ValueError(f"z_observed embed dim {z_observed.shape[-1]} does not match z_future {z_future.shape[-1]}")
    tokens_per_frame = int(tokens_per_frame)
    if tokens_per_frame <= 0:
        raise ValueError(f"tokens_per_frame must be > 0, got {tokens_per_frame}")
    if z_future.ndim == 3:
        if z_future.shape[1] % tokens_per_frame != 0:
            raise ValueError(
                "tokens_per_frame must divide flat z_future length; "
                f"got tokens_per_frame={tokens_per_frame}, token_length={z_future.shape[1]}"
            )
        future_frames = int(z_future.shape[1] // tokens_per_frame)
    else:
        future_frames = int(z_future.shape[1])

    configured_steps = int(getattr(cfg, "steps", 2))
    if configured_steps != 2:
        raise ValueError(f"CVoI guidance requires configured K=2, got K={configured_steps}")
    if evaluation_guidance_steps is None:
        steps = configured_steps
    else:
        allowed_steps = cvoi_evaluation_guidance_steps(config, include_disabled=False)
        if type(evaluation_guidance_steps) is not int or evaluation_guidance_steps not in allowed_steps:
            raise ValueError(
                "evaluation_guidance_steps must be exactly one of "
                f"{list(allowed_steps)}, got {evaluation_guidance_steps!r}"
            )
        steps = evaluation_guidance_steps

    base = z_future.detach()
    observed_base = z_observed.detach()
    if future_frames == 0:
        diagnostics = dict(disabled_diagnostics)
        diagnostics["guidance_skipped_h0"] = 1.0
        return base, diagnostics

    objective = str(getattr(cfg, "objective", "last"))
    if objective != "last":
        raise ValueError(f"CVoI guidance objective must be 'last', got {objective!r}")
    step_size = float(getattr(cfg, "step_size", 0.05))
    max_delta_norm = float(getattr(cfg, "max_delta_norm", 0.25))
    detach_output = bool(getattr(cfg, "detach_output", True))
    if not math.isfinite(step_size) or step_size < 0.0:
        raise ValueError(f"value_guidance.step_size must be finite and >= 0, got {step_size}")
    if not math.isfinite(max_delta_norm) or max_delta_norm <= 0.0:
        raise ValueError(f"value_guidance.max_delta_norm must be finite and > 0, got {max_delta_norm}")
    if not detach_output:
        raise ValueError("CVoI guidance requires value_guidance.detach_output=true")

    delta = torch.zeros_like(base, requires_grad=True)

    def _last_field_values(candidate: torch.Tensor) -> torch.Tensor:
        output = dual_value_model(
            observed_base,
            candidate,
            tokens_per_frame=tokens_per_frame,
        )
        if not hasattr(output, "field_values"):
            raise TypeError("CVoI dual value model output must expose field_values")
        field_values = output.field_values
        if not isinstance(field_values, torch.Tensor) or field_values.ndim != 2:
            raise ValueError(
                "CVoI dual value field_values must be [B, F], got " f"{getattr(field_values, 'shape', None)}"
            )
        if field_values.shape != (base.shape[0], future_frames):
            raise ValueError(
                f"CVoI dual value field_values must be [{base.shape[0]}, {future_frames}], "
                f"got {tuple(field_values.shape)}"
            )
        if not bool(torch.isfinite(field_values).all().item()):
            raise RuntimeError("CVoI dual value field_values contains NaN/Inf")
        return field_values[:, -1]

    # cuDNN RNNs reject backward after an eval-mode forward. Guidance needs
    # gradients only with respect to the latent delta, so use the native GRU
    # autograd path while keeping the frozen Value model in eval mode.
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        field_values_before = _last_field_values(base)
        for _ in range(steps):
            # Samples are independent. Sum keeps each sample's delta invariant
            # to batch size, while a mean would divide every update by B.
            score = _last_field_values(base + delta).sum()
            if not score.requires_grad:
                raise RuntimeError("CVoI field guidance objective has no gradient")
            (gradient,) = torch.autograd.grad(score, delta, allow_unused=True)
            if gradient is None:
                raise RuntimeError("CVoI field guidance objective produced no gradient for future delta")
            if not torch.isfinite(gradient).all():
                raise RuntimeError("CVoI field guidance gradient contains NaN/Inf")
            with torch.no_grad():
                next_delta = delta + step_size * gradient
                next_delta = _clip_delta_by_token_norm(next_delta, max_delta_norm=max_delta_norm)
            delta = next_delta.requires_grad_(True)
        guided = base + delta
        field_values_after = _last_field_values(guided)

    diagnostics = {
        "guidance_steps": float(steps),
        "guidance_skipped_h0": 0.0,
        "delta_norm": float(delta.detach().norm(dim=-1).max()),
        "field_value_before": float(field_values_before.detach().mean()),
        "field_value_after": float(field_values_after.detach().mean()),
    }
    return guided.detach(), diagnostics
