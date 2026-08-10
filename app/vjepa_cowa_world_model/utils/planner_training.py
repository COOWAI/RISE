"""Helper utilities for planner-side training and validation."""

from typing import Optional, Sequence

import torch


def resolve_validation_target_timeline(
    *,
    num_target_frames: int,
    num_observed_frames: int,
    predictor_inference_consistent: bool,
    predictor_no_aux_input: bool = False,
) -> dict[str, int | bool]:
    """Resolve the exact GT origin and trajectory length used by planner validation."""

    num_target_frames = int(num_target_frames)
    num_observed_frames = int(num_observed_frames)
    if num_target_frames < 2:
        raise ValueError(f"num_target_frames must be at least 2, got {num_target_frames}")
    if num_observed_frames < 1 or num_observed_frames >= num_target_frames:
        raise ValueError(
            "num_observed_frames must satisfy 0 < observed < target, "
            f"got observed={num_observed_frames}, target={num_target_frames}"
        )
    effective_inference_consistent = bool(predictor_inference_consistent) and not bool(predictor_no_aux_input)
    future_start_index = num_observed_frames if effective_inference_consistent else 1
    return {
        "predictor_inference_consistent": effective_inference_consistent,
        "future_start_index": future_start_index,
        "origin_index": future_start_index - 1,
        "num_poses": num_target_frames - future_start_index,
    }


def resolve_validation_timestep_sec(
    fps: Optional[float] = None,
    diff_dt: Optional[float] = None,
    default: float = 0.5,
) -> float:
    """Resolve validation timestep seconds from config values.

    Prefer dataset fps because validation metrics are indexed by real trajectory
    timestamps. Fall back to planner.diff_dt, then to the provided default.
    """
    if fps is not None and fps > 0:
        return 1.0 / float(fps)
    if diff_dt is not None and diff_dt > 0:
        return float(diff_dt)
    return float(default)


def horizon_seconds_to_step_index(seconds: float, timestep_sec: float) -> int:
    """Map a future horizon in seconds to a 0-indexed trajectory step."""
    if timestep_sec <= 0:
        raise ValueError(f"timestep_sec must be positive, got {timestep_sec}")
    # round() (not int() truncation) to match utils/metrics.py — e.g. timestep_sec=0.2, 3/0.2=14.999... must
    # map to step 15, not 14. Integer divisions (e.g. navsim dt=0.5) are unaffected (byte-identical).
    return int(round(float(seconds) / float(timestep_sec))) - 1


def build_horizon_regression_timestep_weights(
    num_poses: int,
    timestep_sec: float,
    horizon_seconds: Sequence[float],
    horizon_weights: Sequence[float],
    normalize: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Optional[torch.Tensor]:
    """Build optional per-timestep regression weights for horizon-focused loss.

    Parameters
    ----------
    num_poses       : number of future trajectory poses.
    timestep_sec    : seconds represented by one future trajectory step.
    horizon_seconds : horizons to reweight, e.g. [2.0, 3.0].
    horizon_weights : raw weights for each horizon, e.g. [2.0, 2.0].
    normalize       : if True, scale the final vector to mean 1.

    Returns
    -------
    Optional[torch.Tensor]
        [num_poses] weights, or None when disabled or no horizons are in range.
    """
    if len(horizon_seconds) != len(horizon_weights):
        raise ValueError(
            "horizon_seconds and horizon_weights must have the same length, "
            f"got {len(horizon_seconds)} and {len(horizon_weights)}"
        )
    if num_poses <= 0:
        raise ValueError(f"num_poses must be positive, got {num_poses}")
    if not horizon_seconds:
        return None

    weights = torch.ones(num_poses, device=device, dtype=dtype)
    applied = False
    for seconds, raw_weight in zip(horizon_seconds, horizon_weights):
        raw_weight = float(raw_weight)
        if raw_weight < 0:
            raise ValueError(f"horizon regression weight must be non-negative, got {raw_weight}")
        step_idx = horizon_seconds_to_step_index(float(seconds), timestep_sec=timestep_sec)
        if 0 <= step_idx < num_poses:
            weights[step_idx] = raw_weight
            applied = True

    if not applied:
        return None
    if normalize:
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("horizon regression weights must sum to a positive value")
        weights = weights * (weights.numel() / weight_sum)
    return weights


def compute_planner_wta_loss(
    config,
    *,
    pred_trajs: torch.Tensor,
    pred_conf: torch.Tensor,
    gt_traj: torch.Tensor,
    epoch: int,
    alpha: Optional[float] = None,
    timestep_weights: Optional[torch.Tensor] = None,
    mode_scores: Optional[torch.Tensor] = None,
    global_batch: bool = False,
):
    """Single fail-loud WTA trajectory-loss dispatch shared by every planner training path.

    Replaces the previously-duplicated inline ``if/elif`` in the planner_world_model / planner_encoder_only
    lines, the refinement runtime, and the RL command. ``num_modes == 1`` uses ``single_model_loss``;
    otherwise the variant is selected by ``config.planner.wta_loss_version`` via ``get_loss_function()``,
    which **raises on an unknown version** (the old inline ``else`` silently fell back to v1).

    Optional ``timestep_weights`` (horizon reweighting) / ``mode_scores`` (world-model latent reward) are
    forwarded only to the variants that accept them, so each call site stays byte-identical to its prior
    inline dispatch.
    """
    from app.vjepa_cowa_world_model.losses import awta_temperature_schedule, get_loss_function, single_model_loss

    p = config.planner
    alpha = p.wta_alpha if alpha is None else alpha
    if p.num_modes == 1:
        kwargs = {} if timestep_weights is None else {"timestep_weights": timestep_weights}
        return single_model_loss(
            pred_trajs=pred_trajs,
            gt_traj=gt_traj,
            reg_loss_weight=p.reg_loss_weight,
            alpha=alpha,
            **kwargs,
        )

    loss_fn = get_loss_function(p.wta_loss_version)  # raises on unknown version (no silent v1 fallback)
    extra = {}
    if timestep_weights is not None:
        extra["timestep_weights"] = timestep_weights
    if mode_scores is not None:
        extra["mode_scores"] = mode_scores
    extra["global_batch"] = global_batch

    if p.wta_loss_version == "v2":
        return loss_fn(
            pred_trajs=pred_trajs,
            pred_conf_logits=pred_conf,
            gt_traj=gt_traj,
            reg_loss_weight=p.reg_loss_weight,
            conf_loss_weight=p.conf_loss_weight,
            cover_loss_weight=p.cover_loss_weight,
            alpha=alpha,
            temperature=p.wta_temperature,
            **extra,
        )
    if p.wta_loss_version == "v3":
        awta_temperature = awta_temperature_schedule(
            init_temperature=p.awta_init_temperature,
            epoch=epoch,
            exp_base=p.awta_exp_base,
            min_temperature=p.awta_min_temperature,
        )
        return loss_fn(
            pred_trajs=pred_trajs,
            pred_conf_logits=pred_conf,
            gt_traj=gt_traj,
            reg_loss_weight=p.reg_loss_weight,
            conf_loss_weight=p.conf_loss_weight,
            cover_loss_weight=p.cover_loss_weight,
            alpha=alpha,
            conf_temperature=p.wta_temperature,
            awta_temperature=awta_temperature,
            **extra,
        )
    # v1 — get_loss_function already validated wta_loss_version ∈ {v1, v2, v3}
    return loss_fn(
        pred_trajs=pred_trajs,
        pred_conf_logits=pred_conf,
        gt_traj=gt_traj,
        reg_loss_weight=p.reg_loss_weight,
        conf_loss_weight=p.conf_loss_weight,
        alpha=alpha,
        **extra,
    )


# shared by train_navsim_v2 / train_planner_encoder_only (moved verbatim)
def _resolve_action_history_dt(config) -> float:
    """Resolve timestep for 6D action-history velocity features."""
    fps = getattr(config.data, "fps", None)
    if fps is not None and float(fps) > 0:
        return 1.0 / float(fps)

    diff_dt = getattr(config.planner, "diff_dt", None)
    if diff_dt is not None and float(diff_dt) > 0:
        return float(diff_dt)

    raise ValueError(
        "Cannot resolve the action-history timestep: neither data.fps nor planner.diff_dt is set "
        "(both control velocity-feature scaling). Configure one — refusing to fall back to 1.0s, "
        "which would silently mis-scale velocities."
    )
