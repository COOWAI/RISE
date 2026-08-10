"""Per-camera predictor loss diagnostics for multi-view token streams."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from app.vjepa_cowa_world_model.training.predictor_loss import (
    _resolve_active_prefix_tokens,
    predictor_uses_future_only_loss_scope,
    resolve_predictor_supervision_mode,
)
from app.vjepa_cowa_world_model.utils.module_utils import get_nested_config as _get_nested_config
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module as _unwrap_module


def resolve_per_camera_names(config: Any) -> List[str]:
    """Return camera names when multiview per-view tokens are active."""
    if not bool(_get_nested_config(config, "multiview", "enabled", default=False)):
        return []
    if str(_get_nested_config(config, "multiview", "output_mode", default="fused")).lower() != "per_view":
        return []
    camera_names = _get_nested_config(config, "data", "navsim", "camera_names", default=None)
    if not camera_names:
        return []
    return [str(name) for name in camera_names]


def _loss_exp(config: Any) -> float:
    return float(_get_nested_config(config, "loss", "loss_exp", default=1.0))


def _per_camera_pair_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    config: Any,
    tokens_per_frame: int,
    camera_names: List[str],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if not camera_names:
        return {}
    num_views = len(camera_names)
    if int(tokens_per_frame) % num_views != 0:
        raise ValueError(f"tokens_per_frame={tokens_per_frame} must be divisible by camera count {num_views}")
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes must match, got {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.size(1) % int(tokens_per_frame) != 0:
        raise ValueError(f"token length {pred.size(1)} is not divisible by tokens_per_frame={tokens_per_frame}")

    per_view_tokens = int(tokens_per_frame) // num_views
    frames = pred.size(1) // int(tokens_per_frame)
    pred_btvpd = pred.reshape(pred.size(0), frames, num_views, per_view_tokens, pred.size(-1))
    target_btvpd = target.reshape(target.size(0), frames, num_views, per_view_tokens, target.size(-1))
    abs_error = torch.abs(pred_btvpd - target_btvpd)
    per_view_loss = (abs_error ** _loss_exp(config)).mean(dim=(0, 1, 3, 4)) / _loss_exp(config)
    token_count = torch.tensor(
        float(pred.size(0) * frames * per_view_tokens),
        dtype=torch.float32,
        device=pred.device,
    )

    metrics: Dict[str, torch.Tensor] = {}
    for view_idx, camera_name in enumerate(camera_names):
        metrics[f"{prefix}/{camera_name}"] = per_view_loss[view_idx].detach()
        metrics[f"predictor_per_camera_num_tokens/{camera_name}"] = token_count
    return metrics


def _target_slice_for_prediction(h_target: torch.Tensor, pred: torch.Tensor, offset: int) -> torch.Tensor:
    return h_target[:, int(offset) : int(offset) + pred.size(1)]


def compute_predictor_per_camera_jepa_losses(
    *,
    z_tf: torch.Tensor,
    z_ar: Optional[torch.Tensor],
    h_target: torch.Tensor,
    config: Any,
    tokens_per_frame: int,
    num_observed_steps: int,
    active_prefix_steps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute AC/parallel predictor per-camera JEPA diagnostics."""
    camera_names = resolve_per_camera_names(config)
    if not camera_names:
        return {}

    observed_steps = int(num_observed_steps)
    future_only_loss = predictor_uses_future_only_loss_scope(config)
    active_prefix_tokens = _resolve_active_prefix_tokens(
        active_prefix_steps=active_prefix_steps,
        tokens_per_frame=tokens_per_frame,
    )
    if active_prefix_tokens is not None and not future_only_loss:
        raise ValueError("active prefixes require train.predictor_loss_scope='future_only'")
    if bool(_get_nested_config(config, "train", "use_parallel_predictor", default=False)):
        future_start_step = observed_steps if future_only_loss else 1
        base_offset = future_start_step * int(tokens_per_frame)
        observed_tokens = base_offset
    elif future_only_loss:
        base_offset = observed_steps * int(tokens_per_frame)
        observed_tokens = (observed_steps - 1) * int(tokens_per_frame)
    else:
        base_offset = int(tokens_per_frame)
        observed_tokens = 0
    target_offset = base_offset

    supervision_mode = resolve_predictor_supervision_mode(config)
    metrics: Dict[str, torch.Tensor] = {}
    if supervision_mode in ("tf", "tf_ar"):
        if z_tf is None:
            raise ValueError("z_tf must be provided when predictor_supervision_mode includes 'tf'")
        z_tf_for_loss = z_tf[:, observed_tokens:]
        if active_prefix_tokens is not None:
            z_tf_for_loss = z_tf_for_loss[:, :active_prefix_tokens]
        metrics.update(
            _per_camera_pair_loss(
                z_tf_for_loss,
                _target_slice_for_prediction(h_target, z_tf_for_loss, target_offset),
                config=config,
                tokens_per_frame=tokens_per_frame,
                camera_names=camera_names,
                prefix="predictor_per_camera_jloss",
            )
        )

    if supervision_mode in ("ar", "tf_ar") and z_ar is not None:
        z_ar_for_loss = z_ar
        if active_prefix_tokens is not None:
            z_ar_for_loss = z_ar[:, :active_prefix_tokens]
        metrics.update(
            _per_camera_pair_loss(
                z_ar_for_loss,
                _target_slice_for_prediction(h_target, z_ar_for_loss, target_offset),
                config=config,
                tokens_per_frame=tokens_per_frame,
                camera_names=camera_names,
                prefix="predictor_per_camera_sloss",
            )
        )

    for camera_name in camera_names:
        jloss_key = f"predictor_per_camera_jloss/{camera_name}"
        sloss_key = f"predictor_per_camera_sloss/{camera_name}"
        if jloss_key not in metrics and sloss_key in metrics:
            metrics[jloss_key] = metrics[sloss_key] * 0.0
        if sloss_key not in metrics and jloss_key in metrics:
            metrics[sloss_key] = metrics[jloss_key] * 0.0
        metrics[f"predictor_per_camera_loss/{camera_name}"] = metrics[jloss_key] + metrics[sloss_key]
    return metrics


def compute_latent_dit_per_camera_losses(
    *,
    latent_output: Any,
    h_target: torch.Tensor,
    config: Any,
    tokens_per_frame: int,
    num_observed_steps: int,
    predictor: Any,
) -> Dict[str, torch.Tensor]:
    """Compute latent-DiT per-camera flow/x0 diagnostics."""
    camera_names = resolve_per_camera_names(config)
    if not camera_names:
        return {}

    future_offset = int(num_observed_steps) * int(tokens_per_frame)
    x0_pred = latent_output.x0_pred
    future_token_indices = getattr(latent_output, "future_token_indices", None)
    if future_token_indices is None:
        target_future = _target_slice_for_prediction(h_target, x0_pred, future_offset)
    else:
        indices = future_token_indices.to(device=h_target.device, dtype=torch.long) + future_offset
        if bool((indices < 0).any().item()) or bool((indices >= h_target.shape[1]).any().item()):
            raise ValueError(
                f"latent_output.future_token_indices out of range for h_target shape {tuple(h_target.shape)}"
            )
        target_future = h_target.index_select(1, indices)
    metrics = _per_camera_pair_loss(
        x0_pred,
        target_future,
        config=config,
        tokens_per_frame=tokens_per_frame,
        camera_names=camera_names,
        prefix="predictor_per_camera_x0_loss",
    )

    velocity_pred = latent_output.velocity_pred
    velocity_target = latent_output.velocity_target
    if velocity_pred is not None and velocity_target is not None:
        metrics.update(
            _per_camera_pair_loss(
                velocity_pred,
                velocity_target,
                config=config,
                tokens_per_frame=tokens_per_frame,
                camera_names=camera_names,
                prefix="predictor_per_camera_flow_loss",
            )
        )
        objective = str(
            getattr(latent_output, "objective", getattr(_unwrap_module(predictor), "objective", "flow_matching"))
        ).lower()
        x0_weight = float(_unwrap_module(predictor).x0_loss_weight)
        for camera_name in camera_names:
            if objective == "x0_prediction":
                metrics[f"predictor_per_camera_loss/{camera_name}"] = metrics[
                    f"predictor_per_camera_x0_loss/{camera_name}"
                ]
            else:
                metrics[f"predictor_per_camera_loss/{camera_name}"] = (
                    metrics[f"predictor_per_camera_flow_loss/{camera_name}"]
                    + x0_weight * metrics[f"predictor_per_camera_x0_loss/{camera_name}"]
                )
    else:
        for camera_name in camera_names:
            metrics[f"predictor_per_camera_loss/{camera_name}"] = metrics[
                f"predictor_per_camera_x0_loss/{camera_name}"
            ]
    return metrics
