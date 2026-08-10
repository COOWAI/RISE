"""Predictor JEPA loss helpers."""

from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from app.vjepa_cowa_world_model.training.prefix_schedule import (
    PrefixSample,
    resolve_prefix_distribution,
    sample_prefix,
)


def select_dynamic_rollout_prefix(
    *,
    enabled: bool,
    total_future_steps: int,
    full_prefix_prob: float,
    min_prefix_steps: int,
    max_non_full_prefix_steps: Optional[int],
    device: torch.device,
    horizon_probabilities: Optional[Sequence[float]] = None,
) -> PrefixSample:
    """Draw the cumulative AC rollout prefix for one rank-local batch."""
    distribution = resolve_prefix_distribution(
        enabled=enabled,
        horizon_steps=total_future_steps,
        full_prefix_prob=full_prefix_prob,
        min_prefix_steps=min_prefix_steps,
        max_non_full_prefix_steps=max_non_full_prefix_steps,
        horizon_probabilities=horizon_probabilities,
    )
    return sample_prefix(distribution, device=device)


def validate_ac_transformer_dynamic_rollout_config(config: Any, *, needs_planner_or_value: bool = False) -> None:
    """Fail loud when AC dynamic prefix rollout is outside its supported envelope."""

    dynamic_cfg = getattr(config, "predictor_dynamic_rollout", None)
    if dynamic_cfg is None or not bool(dynamic_cfg.enabled):
        return
    if str(config.train.predictor_type).lower() != "ac_transformer":
        raise ValueError(
            "ac_transformer dynamic rollout requires predictor-only future_only inference-consistent training: "
            f"train.predictor_type={config.train.predictor_type!r}"
        )
    if bool(config.train.use_parallel_predictor):
        raise ValueError(
            "ac_transformer dynamic rollout requires predictor-only future_only inference-consistent training: "
            "train.use_parallel_predictor must be false"
        )
    if not bool(config.train.predictor_inference_consistent):
        raise ValueError(
            "ac_transformer dynamic rollout requires predictor-only future_only inference-consistent training: "
            "train.predictor_inference_consistent must be true"
        )
    if resolve_predictor_loss_scope(config) != "future_only":
        raise ValueError(
            "ac_transformer dynamic rollout requires predictor-only future_only inference-consistent training: "
            "train.predictor_loss_scope must resolve to 'future_only'"
        )
    predictor_weights_updated = bool(config.train.predictor_train) or bool(
        getattr(config.train, "predictor_planner_finetune", False)
    )
    if predictor_weights_updated and bool(config.train.predictor_static_graph):
        raise ValueError(
            "ac_transformer dynamic rollout requires predictor-only future_only inference-consistent training: "
            "train.predictor_static_graph must be false"
        )
    min_prefix_steps = int(dynamic_cfg.min_prefix_steps)
    horizon_probabilities = getattr(dynamic_cfg, "horizon_probabilities", None)
    can_sample_h0 = (
        bool(horizon_probabilities[0] > 0.0)
        if horizon_probabilities is not None
        else float(dynamic_cfg.full_prefix_prob) < 1.0 and min_prefix_steps == 0
    )
    predictor_supervision_enabled = bool(config.train.predictor_train) and not bool(
        getattr(config.train, "predictor_planner_finetune", False)
    )
    if can_sample_h0 and predictor_supervision_enabled:
        raise ValueError(
            "predictor_dynamic_rollout.min_prefix_steps=0 can sample h=0, which is forbidden with predictor "
            "supervision; use planner-only training or set min_prefix_steps >= 1"
        )
    if can_sample_h0 and bool(needs_planner_or_value):
        has_nonfuture_planner_tokens = bool(
            config.planner.use_z_context
            or config.planner.use_observed_tokens
            or config.planner.use_action_history_for_planner
        )
        if not bool(config.planner.use_planner) or not has_nonfuture_planner_tokens:
            raise ValueError(
                "predictor_dynamic_rollout h=0 requires planner-only training with at least one non-future "
                "planner context token source: use_z_context, observed tokens, or action_history"
            )


def _resolve_active_prefix_tokens(
    *,
    active_prefix_steps: Optional[int],
    tokens_per_frame: int,
) -> Optional[int]:
    if active_prefix_steps is None:
        return None
    prefix_steps = int(active_prefix_steps)
    if prefix_steps <= 0:
        raise ValueError(f"active_prefix_steps must be positive for predictor supervision, got {prefix_steps}")
    tpf = int(tokens_per_frame)
    if tpf <= 0:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
    return prefix_steps * tpf


def _slice_active_prefix_tokens(z: Any, *, name: str, active_prefix_tokens: Optional[int]) -> Any:
    """Slice an active prefix after verifying the predictor produced every requested token."""
    if active_prefix_tokens is None:
        return z
    available_tokens = int(z.shape[1])
    if available_tokens < active_prefix_tokens:
        raise ValueError(
            f"{name} must provide at least {active_prefix_tokens} tokens for the active prefix, "
            f"got {available_tokens}"
        )
    return z[:, :active_prefix_tokens]


def _resolve_ar_multistep_discount(config: Any) -> Optional[float]:
    """Return the AR multi-step discount λ from ``config.wm_aux`` (None = off)."""
    wm_aux = getattr(config, "wm_aux", None)
    if wm_aux is None:
        return None
    discount = wm_aux.multistep_discount
    if discount is None:
        return None
    discount = float(discount)
    if not 0.0 < discount <= 1.0:
        raise ValueError(f"wm_aux.multistep_discount must be in (0, 1], got {discount}")
    return discount


def _ar_multistep_weighted_loss(
    z_ar: Any,
    h_target: Any,
    *,
    base_offset: int,
    tokens_per_frame: int,
    discount: float,
    loss_fn: Callable[..., Any],
) -> Any:
    """λ^k-weighted AR rollout loss (doc 9.2): step k weighted by ``discount**k``.

    Weights are normalized (``Σ w_k·l_k / Σ w_k``) so the magnitude stays
    comparable to the unweighted mean; ``discount == 1.0`` reproduces the
    uniform per-step mean (numerically equivalent to the single-call loss,
    since all steps have equal token counts).
    """
    total_tokens = int(z_ar.shape[1])
    tpf = int(tokens_per_frame)
    if tpf <= 0 or total_tokens % tpf != 0:
        raise ValueError(
            f"z_ar token length {total_tokens} must be a positive multiple of tokens_per_frame={tpf} "
            "for multistep-discounted loss"
        )
    num_steps = total_tokens // tpf
    weight_sum = 0.0
    weighted = None
    weight = 1.0
    for step in range(num_steps):
        step_loss = loss_fn(
            z_ar[:, step * tpf : (step + 1) * tpf],
            h_target,
            offset=base_offset + step * tpf,
        )
        term = weight * step_loss
        weighted = term if weighted is None else weighted + term
        weight_sum += weight
        weight *= discount
    return weighted / weight_sum


def resolve_predictor_supervision_mode(config: Any) -> str:
    """Return normalized predictor supervision mode."""
    train = getattr(config, "train", config)
    # 直接索引（缺字段即 AttributeError）。None / "auto" 是文档化哨兵值（dataclass 默认），
    # 表示"未显式指定"——与 parse_training_config 相同，走 legacy flag 推导；拼错的值仍然报错。
    mode = train.predictor_supervision_mode
    if mode is not None and str(mode).strip().lower() != "auto":
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "z_tf": "tf",
            "tf_only": "tf",
            "teacher_forcing": "tf",
            "z_ar": "ar",
            "ar_only": "ar",
            "autoregressive": "ar",
            "both": "tf_ar",
            "tf+ar": "tf_ar",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {"tf", "ar", "tf_ar"}:
            return normalized
        # fail-loud (point 27): 写错/拼错 predictor_supervision_mode 直接报错，不再静默落到 legacy 默认。
        raise ValueError(
            "train.predictor_supervision_mode must be one of ['tf', 'ar', 'tf_ar'] (or aliases), " f"got {mode!r}"
        )
    return "tf_ar" if bool(train.predictor_use_z_ar_supervision) else "tf"


def predictor_supervises_tf(config: Any) -> bool:
    return resolve_predictor_supervision_mode(config) in {"tf", "tf_ar"}


def predictor_supervises_ar(config: Any) -> bool:
    return resolve_predictor_supervision_mode(config) in {"ar", "tf_ar"}


def resolve_predictor_loss_scope(config: Any) -> str:
    """Return normalized z_tf loss time-window scope."""
    train = getattr(config, "train", config)
    # 直接索引（缺字段即 AttributeError）。None / "auto" 是文档化哨兵值（dataclass 默认），
    # 表示"未显式指定"——与 parse_training_config 相同，走 legacy flag 推导；拼错的值仍然报错。
    scope = train.predictor_loss_scope
    if scope is not None and str(scope).strip().lower() != "auto":
        normalized = str(scope).strip().lower().replace("-", "_")
        aliases = {
            "next": "next_step",
            "all": "next_step",
            "all_next": "next_step",
            "all_next_steps": "next_step",
            "teacher_forcing": "next_step",
            "tf": "next_step",
            "future": "future_only",
            "future_tokens": "future_only",
            "future_steps": "future_only",
            "observed_future": "future_only",
            "inference_consistent": "future_only",
            "ic": "future_only",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in {"next_step", "future_only"}:
            return normalized
        # fail-loud (point 27): 写错/拼错 predictor_loss_scope 直接报错，不再静默落到 legacy 默认。
        raise ValueError(
            "train.predictor_loss_scope must be one of ['next_step', 'future_only'] (or aliases), " f"got {scope!r}"
        )
    return "future_only" if bool(train.predictor_inference_consistent) else "next_step"


def predictor_uses_future_only_loss_scope(config: Any) -> bool:
    return resolve_predictor_loss_scope(config) == "future_only"


def predictor_needs_z_ar_rollout(config: Any) -> bool:
    """Return whether the current config actually needs predictor z_ar outputs."""

    if predictor_supervises_ar(config):
        return True

    return bool(config.planner.use_planner and config.planner.planner_input_source != "z_tf")


def compute_predictor_jepa_losses_from_config(
    z_tf: Any,
    z_ar: Any,
    h_target: Any,
    config: Any,
    tokens_per_frame: int,
    loss_fn: Callable[..., Any],
    num_observed_steps: Optional[int] = None,
    active_prefix_steps: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Compute predictor JEPA losses from config.

    Returns
    -------
    tuple:
        jepa_loss, jloss, sloss
    """

    observed_steps = int(num_observed_steps) if num_observed_steps is not None else config.train.num_observed_frames

    supervision_mode = resolve_predictor_supervision_mode(config)

    future_only_loss = predictor_uses_future_only_loss_scope(config)
    active_prefix_tokens = _resolve_active_prefix_tokens(
        active_prefix_steps=active_prefix_steps,
        tokens_per_frame=tokens_per_frame,
    )
    if active_prefix_tokens is not None and not future_only_loss:
        raise ValueError("active prefixes require train.predictor_loss_scope='future_only'")

    # Phase 1 (doc 9.2): 可选 λ^k 逐步折扣，只作用于 AR rollout 的 sloss。
    ar_discount = _resolve_ar_multistep_discount(config)

    def _sloss(z_ar_tokens: Any, base_offset: int) -> Any:
        if ar_discount is None:
            return loss_fn(z_ar_tokens, h_target, offset=base_offset)
        return _ar_multistep_weighted_loss(
            z_ar_tokens,
            h_target,
            base_offset=base_offset,
            tokens_per_frame=tokens_per_frame,
            discount=ar_discount,
            loss_fn=loss_fn,
        )

    if bool(config.train.use_parallel_predictor):
        future_start_step = observed_steps if future_only_loss else 1
        future_offset = future_start_step * tokens_per_frame
        jloss = None
        if supervision_mode in ("tf", "tf_ar"):
            z_tf_for_loss = z_tf[:, future_offset:]
            z_tf_for_loss = _slice_active_prefix_tokens(
                z_tf_for_loss,
                name="z_tf",
                active_prefix_tokens=active_prefix_tokens,
            )
            jloss = loss_fn(
                z_tf_for_loss,
                h_target,
                offset=future_offset,
            )
        sloss = None
        if supervision_mode in ("ar", "tf_ar"):
            if z_ar is None:
                raise ValueError("z_ar must be provided when predictor_supervision_mode includes 'ar'")
            z_ar_for_loss = _slice_active_prefix_tokens(
                z_ar,
                name="z_ar",
                active_prefix_tokens=active_prefix_tokens,
            )
            sloss = _sloss(z_ar_for_loss, future_offset)
        if jloss is None:
            jloss = sloss * 0.0
        if sloss is None:
            sloss = jloss * 0.0
        jepa_loss = jloss + sloss
        return jepa_loss, jloss, sloss

    if future_only_loss:
        base_offset = observed_steps * tokens_per_frame
        observed_tokens = (observed_steps - 1) * tokens_per_frame
    else:
        base_offset = tokens_per_frame
        observed_tokens = 0
    target_offset = base_offset

    jloss = None
    if supervision_mode in ("tf", "tf_ar"):
        if z_tf is None:
            raise ValueError("z_tf must be provided when predictor_supervision_mode includes 'tf'")
        z_tf_for_loss = z_tf
        if observed_tokens != 0:
            z_tf_for_loss = z_tf_for_loss[:, observed_tokens:]
        z_tf_for_loss = _slice_active_prefix_tokens(
            z_tf_for_loss,
            name="z_tf",
            active_prefix_tokens=active_prefix_tokens,
        )
        jloss = loss_fn(z_tf_for_loss, h_target, offset=target_offset)

    sloss = None
    if supervision_mode in ("ar", "tf_ar"):
        if z_ar is None:
            raise ValueError("z_ar must be provided when predictor_supervision_mode includes 'ar'")
        z_ar_for_loss = _slice_active_prefix_tokens(
            z_ar,
            name="z_ar",
            active_prefix_tokens=active_prefix_tokens,
        )
        sloss = _sloss(z_ar_for_loss, target_offset)
    if jloss is None:
        jloss = sloss * 0.0
    if sloss is None:
        sloss = jloss * 0.0

    jepa_loss = jloss + sloss
    return jepa_loss, jloss, sloss


def compute_lewm_projected_jepa_losses_from_config(
    z_tf: Any,
    z_ar: Any,
    h_target: Any,
    config: Any,
    tokens_per_frame: int,
    project_fn: Callable[[Any], Any],
    loss_fn: Callable[..., Any],
    num_observed_steps: Optional[int] = None,
) -> Tuple[Any, Any, Any]:
    """Project predictor outputs into le-wm space, then compute JEPA losses."""

    z_tf_projected = project_fn(z_tf) if predictor_supervises_tf(config) else z_tf
    z_ar_projected = project_fn(z_ar) if predictor_supervises_ar(config) else None

    return compute_predictor_jepa_losses_from_config(
        z_tf=z_tf_projected,
        z_ar=z_ar_projected,
        h_target=h_target,
        config=config,
        tokens_per_frame=tokens_per_frame,
        loss_fn=loss_fn,
        num_observed_steps=num_observed_steps,
    )
