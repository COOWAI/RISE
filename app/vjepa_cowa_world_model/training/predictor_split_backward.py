"""Memory-bounded teacher-forcing/autoregressive predictor backpropagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class SplitTfArBackwardOutput:
    """Outputs after the TF graph has already been backpropagated and released."""

    z_tf: torch.Tensor
    z_ar: torch.Tensor
    jepa_loss: torch.Tensor
    jloss: torch.Tensor
    sloss: torch.Tensor


def _require_scalar_loss(loss: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(loss) or loss.ndim != 0:
        shape = None if not torch.is_tensor(loss) else tuple(loss.shape)
        raise ValueError(f"{name} must be a scalar tensor, got shape={shape}")


def compute_split_tf_loss(
    *,
    z_tf: torch.Tensor,
    h_target: torch.Tensor,
    tokens_per_frame: int,
    num_observed_steps: int,
    loss_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Compute the TF component for the supported future-only, non-parallel path."""

    tokens_per_frame = int(tokens_per_frame)
    num_observed_steps = int(num_observed_steps)
    if tokens_per_frame <= 0:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
    if num_observed_steps <= 0:
        raise ValueError(f"num_observed_steps must be positive, got {num_observed_steps}")
    observed_tokens = (num_observed_steps - 1) * tokens_per_frame
    target_offset = num_observed_steps * tokens_per_frame
    if int(z_tf.shape[1]) <= observed_tokens:
        raise ValueError(
            "z_tf must contain at least one future token after the observed prefix: "
            f"tokens={int(z_tf.shape[1])}, observed_prefix_tokens={observed_tokens}"
        )
    return loss_fn(z_tf[:, observed_tokens:], h_target, offset=target_offset)


def compute_split_ar_loss(
    *,
    z_ar: torch.Tensor,
    h_target: torch.Tensor,
    tokens_per_frame: int,
    num_observed_steps: int,
    loss_fn: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Compute the AR component for the supported future-only, non-parallel path."""

    tokens_per_frame = int(tokens_per_frame)
    num_observed_steps = int(num_observed_steps)
    if tokens_per_frame <= 0:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
    if num_observed_steps <= 0:
        raise ValueError(f"num_observed_steps must be positive, got {num_observed_steps}")
    if int(z_ar.shape[1]) <= 0:
        raise ValueError("z_ar must contain at least one future token")
    target_offset = num_observed_steps * tokens_per_frame
    return loss_fn(z_ar, h_target, offset=target_offset)


def backward_loss_outside_autocast(
    loss: torch.Tensor,
    *,
    scaler: Any,
    mixed_precision: bool,
) -> None:
    """Backpropagate with an explicitly disabled nested autocast context."""

    _require_scalar_loss(loss, name="loss")
    with torch.cuda.amp.autocast(enabled=False):
        if mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()


def run_split_tf_ar_forward_backward(
    *,
    forward_tf: Callable[[], torch.Tensor],
    forward_ar: Callable[[], torch.Tensor],
    compute_jloss: Callable[[torch.Tensor], torch.Tensor],
    compute_sloss: Callable[[torch.Tensor], torch.Tensor],
    early_backward: Callable[[torch.Tensor], None],
) -> SplitTfArBackwardOutput:
    """Backpropagate TF before constructing AR, retaining only detached TF diagnostics."""

    z_tf_with_graph = forward_tf()
    if not torch.is_tensor(z_tf_with_graph):
        raise TypeError(f"forward_tf must return a tensor, got {type(z_tf_with_graph).__name__}")
    jloss_with_graph = compute_jloss(z_tf_with_graph)
    _require_scalar_loss(jloss_with_graph, name="jloss")
    early_backward(jloss_with_graph)

    z_tf = z_tf_with_graph.detach()
    jloss = jloss_with_graph.detach()
    del z_tf_with_graph
    del jloss_with_graph

    z_ar = forward_ar()
    if not torch.is_tensor(z_ar):
        raise TypeError(f"forward_ar must return a tensor, got {type(z_ar).__name__}")
    sloss = compute_sloss(z_ar)
    _require_scalar_loss(sloss, name="sloss")
    return SplitTfArBackwardOutput(
        z_tf=z_tf,
        z_ar=z_ar,
        jepa_loss=jloss + sloss,
        jloss=jloss,
        sloss=sloss,
    )


def validate_predictor_split_tf_ar_backward_config(
    config: Any,
    *,
    predictor_lora_enabled: bool = False,
) -> None:
    """Fail fast outside the one graph-independent TF/AR training envelope."""

    if not bool(config.train.predictor_split_tf_ar_backward):
        return

    wm_aux_disabled = (
        config.wm_aux.multistep_discount is None
        and float(config.wm_aux.reward_head_weight) == 0.0
        and float(config.wm_aux.contrastive_weight) == 0.0
    )
    requirements = (
        (str(config.method).lower() == "ema", "method='ema'"),
        (not bool(config.world_model.enabled), "world_model.enabled=false"),
        (not bool(config.model.use_activation_checkpointing), "model.use_activation_checkpointing=false"),
        (not bool(config.model.compile_model), "model.compile_model=false"),
        (not bool(config.train.encoder_train), "train.encoder_train=false"),
        (not bool(config.train.encoder_ema), "train.encoder_ema=false"),
        (not bool(config.train.perceiver_ema), "train.perceiver_ema=false"),
        (bool(config.train.reuse_context_as_target_when_frozen), "train.reuse_context_as_target_when_frozen=true"),
        (bool(config.train.predictor_train), "train.predictor_train=true"),
        (not bool(config.train.predictor_planner_finetune), "train.predictor_planner_finetune=false"),
        (str(config.train.predictor_type).lower() == "ac_transformer", "train.predictor_type='ac_transformer'"),
        (not bool(config.train.use_parallel_predictor), "train.use_parallel_predictor=false"),
        (bool(config.train.predictor_inference_consistent), "train.predictor_inference_consistent=true"),
        (str(config.train.predictor_loss_scope).lower() == "future_only", "train.predictor_loss_scope='future_only'"),
        (
            str(config.train.predictor_supervision_mode).lower() == "tf_ar",
            "train.predictor_supervision_mode='tf_ar'",
        ),
        (not bool(config.train.predictor_static_graph), "train.predictor_static_graph=false"),
        (not bool(config.train.seg_head), "train.seg_head=false"),
        (not bool(config.planner.use_planner), "planner.use_planner=false"),
        (not bool(config.segmentation.use_segmentation), "segmentation.use_segmentation=false"),
        (not bool(config.multiview.enabled), "multiview.enabled=false"),
        (not bool(config.token_ae.enabled), "token_ae.enabled=false"),
        (not bool(config.predictor_dynamic_rollout.enabled), "predictor_dynamic_rollout.enabled=false"),
        (wm_aux_disabled, "wm_aux must be disabled"),
        (not bool(config.value_planning.enabled), "value_planning.enabled=false"),
        (not bool(config.value_guidance.enabled), "value_guidance.enabled=false"),
        (not bool(config.budget_controller.enabled), "budget_controller.enabled=false"),
        (
            not bool(config.counterfactual_supervision.enabled),
            "counterfactual_supervision.enabled=false",
        ),
        (not bool(config.optimization.is_anneal), "optimization.is_anneal=false"),
        (not bool(predictor_lora_enabled), "predictor_lora.enabled=false"),
    )
    violations = [requirement for is_valid, requirement in requirements if not is_valid]
    if violations:
        raise ValueError(
            "train.predictor_split_tf_ar_backward=true is supported only for the strict EMA "
            "predictor-only TF+AR envelope; requires " + ", ".join(violations)
        )
