"""Predictor-only validation utilities."""

from __future__ import annotations

import os
from numbers import Integral
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.config import resolve_main_encoder_tokens_per_frame
from app.vjepa_cowa_world_model.training.counterfactual_supervision import build_counterfactual_sample_masks
from app.vjepa_cowa_world_model.training.models import resolve_runtime_normalize_reps
from app.vjepa_cowa_world_model.training.per_camera_metrics import (
    compute_latent_dit_per_camera_losses,
    compute_predictor_per_camera_jepa_losses,
    resolve_per_camera_names,
)
from app.vjepa_cowa_world_model.training.predictor_aux import call_predictor_with_aux, prepare_predictor_aux_inputs
from app.vjepa_cowa_world_model.training.predictor_loss import compute_predictor_jepa_losses_from_config
from app.vjepa_cowa_world_model.training.predictor_parallel import forward_parallel_predictor, use_parallel_predictor
from app.vjepa_cowa_world_model.training.runtimes.latent_diffusion_runtime import (
    forward_latent_dit_predictor_train,
    resolve_latent_dit_sampler_params,
    sample_latent_dit_predictor,
    use_latent_dit_predictor,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (
    build_parallel_predictor_timeline_inputs,
    build_predictor_timeline_inputs,
    forward_main_context,
    forward_main_target,
    should_reuse_context_as_target,
)
from app.vjepa_cowa_world_model.training.validation_distributed import (
    raise_if_validation_failed,
    wrap_validation_batch_error,
)
from app.vjepa_cowa_world_model.training.validation_rng import resolve_stable_sample_ids, validation_randn
from app.vjepa_cowa_world_model.training.validation_suite import truncate_validation_future_tokens
from app.vjepa_cowa_world_model.utils.eval_determinism import extract_batch_metadata
from app.vjepa_cowa_world_model.utils.module_utils import get_nested_config as _get_nested_config
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module as _unwrap_module
from app.vjepa_cowa_world_model.utils.planner_training import resolve_validation_timestep_sec
from src.utils.logging import get_logger

logger = get_logger(__name__)

PREDICTOR_VALIDATION_AVG_KEYS = (
    "predictor_loss",
    "predictor_jloss",
    "predictor_sloss",
    "predictor_objective_loss",
    "predictor_flow_loss",
    "predictor_x0_loss",
    "predictor_action_loss",
    "predictor_world_loss",
    "predictor_joint_loss",
    # Deployment-faithful metric: MSE of the multi-step sampled rollout (sample()) vs the target
    # future tokens. Unlike the teacher-forced flow_loss, this reflects what the planner / NavSim
    # eval actually consume. Only populated for the latent-DiT predictor.
    "predictor_sampled_mse",
    "predictor_rollout_mse",
)
_PREDICTOR_ROLLOUT_COHORTS = ("all", "safe", "hazard")


def predictor_rollout_mse_per_sample(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return deployment-rollout latent MSE for each sample."""

    if predicted.ndim != 3 or target.ndim != 3:
        raise ValueError("predictor rollout tensors must be [B, N, D]")
    if predicted.shape != target.shape or predicted.shape[1] == 0:
        raise ValueError(f"predictor rollout/target mismatch: {predicted.shape}/{target.shape}")
    return (predicted.float() - target.float()).square().mean(dim=(1, 2))


def _accumulate_predictor_rollout_cohorts(
    sums: Dict[str, float],
    counts: Dict[str, float],
    per_sample_mse: torch.Tensor,
    *,
    validation_domain: str,
    sample_masks: Any,
) -> None:
    """Accumulate raw local sums/counts for one exact-domain validation loader."""

    if per_sample_mse.ndim != 1:
        raise ValueError(f"predictor rollout cohort metric must have shape [B], got {per_sample_mse.shape}")
    if validation_domain == "real":
        masks = {"all": sample_masks.real}
    elif validation_domain == "counterfactual":
        masks = {
            "all": sample_masks.cf_safe | sample_masks.cf_hazard,
            "safe": sample_masks.cf_safe,
            "hazard": sample_masks.cf_hazard,
        }
    else:
        raise ValueError(f"validation_domain must be 'real' or 'counterfactual', got {validation_domain!r}")
    for cohort, mask in masks.items():
        if mask.shape != per_sample_mse.shape:
            raise ValueError(
                f"predictor cohort mask {validation_domain}/{cohort} shape {mask.shape} "
                f"does not match rollout MSE {per_sample_mse.shape}"
            )
        sums[cohort] += float(per_sample_mse[mask].sum().item())
        counts[cohort] += float(mask.sum().item())


def _resolve_predictor_rollout_future_steps(
    rollout_future_steps: Optional[int],
    *,
    observed_steps: int,
    total_steps: int,
) -> int:
    available = int(total_steps) - int(observed_steps)
    if available <= 0:
        raise ValueError(
            f"predictor validation requires future targets, got observed={observed_steps}, total={total_steps}"
        )
    if rollout_future_steps is None:
        return available
    if isinstance(rollout_future_steps, bool) or not isinstance(rollout_future_steps, Integral):
        raise TypeError(f"rollout_future_steps must be an integer or None, got {rollout_future_steps!r}")
    requested = int(rollout_future_steps)
    if requested <= 0:
        raise ValueError(f"predictor rollout_future_steps must be positive, got {requested}")
    if requested > available:
        raise ValueError(f"predictor horizon {requested} exceeds available future steps {available}")
    return requested


def reconstruct_joint_validation_loss(metrics: Dict[str, float], config: Any) -> None:
    """Rebuild the joint metric from independently globally reduced world/action means."""

    if "predictor_world_loss" not in metrics or "predictor_action_loss" not in metrics:
        return
    weight = float(_get_nested_config(config, "predictor_dit", "joint_action_loss_weight", default=0.0))
    joint_loss = float(metrics["predictor_world_loss"]) + weight * float(metrics["predictor_action_loss"])
    metrics["predictor_joint_loss"] = joint_loss
    metrics["predictor_loss"] = joint_loss


def _accumulate_validation_batch_metrics(
    metric_sums: Dict[str, float],
    metric_counts: Dict[str, float],
    avg_keys: tuple[str, ...],
    batch_metrics: Dict[str, torch.Tensor],
    *,
    batch_tokens: float,
) -> None:
    """Accumulate raw local validation numerators and denominators."""

    for key in avg_keys:
        value = batch_metrics.get(key)
        if value is None:
            continue
        if key.startswith("predictor_per_camera_"):
            camera_name = key.rsplit("/", 1)[1]
            count_value = batch_metrics.get(f"predictor_per_camera_num_tokens/{camera_name}")
            metric_count = float(count_value.item()) if count_value is not None else batch_tokens
            metric_sum = float(value.item()) * metric_count
        elif key == "predictor_action_loss":
            action_sum = batch_metrics.get("predictor_action_loss_sum")
            action_count = batch_metrics.get("predictor_action_num_samples")
            if action_sum is None or action_count is None:
                raise ValueError("predictor action validation requires raw local loss sum and sample count")
            metric_sum = float(action_sum.item())
            metric_count = float(action_count.item())
        else:
            metric_count = batch_tokens
            metric_sum = float(value.item()) * metric_count
        metric_sums[key] += metric_sum
        metric_counts[key] += metric_count


def _reduce_validation_totals(
    metric_sums: Dict[str, float],
    metric_counts: Dict[str, float],
    avg_keys: tuple[str, ...],
    *,
    total_tokens: float,
    total_samples: int,
    successful_batches: int,
    failed_batches: int,
    device: torch.device,
    world_size: int,
) -> tuple[Dict[str, float], Dict[str, float], float, int, int, int]:
    """All-reduce validation sums/counts exactly once and unpack them."""

    if world_size <= 1:
        return (
            metric_sums,
            metric_counts,
            total_tokens,
            total_samples,
            successful_batches,
            failed_batches,
        )

    reduce_values = []
    for key in avg_keys:
        reduce_values.extend([metric_sums[key], metric_counts[key]])
    reduce_values.extend([total_tokens, float(total_samples), float(successful_batches), float(failed_batches)])
    reduce_tensor = torch.tensor(reduce_values, dtype=torch.float64, device=device)
    dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)
    reduced = reduce_tensor.tolist()
    cursor = 0
    for key in avg_keys:
        metric_sums[key] = reduced[cursor]
        metric_counts[key] = reduced[cursor + 1]
        cursor += 2
    return (
        metric_sums,
        metric_counts,
        reduced[cursor],
        int(reduced[cursor + 1]),
        int(reduced[cursor + 2]),
        int(reduced[cursor + 3]),
    )


def _predictor_loss_fn(config: Any, tokens_per_frame: int):
    loss_exp = float(_get_nested_config(config, "loss", "loss_exp", default=1.0))

    def loss_fn(z: torch.Tensor, h: torch.Tensor, offset: int = tokens_per_frame) -> torch.Tensor:
        target = h[:, offset : z.size(1) + offset]
        return torch.mean(torch.abs(z - target) ** loss_exp) / loss_exp

    return loss_fn


def _forward_ac_predictor_for_validation(
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    predictor_inputs: Any,
    *,
    config: Any,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    rollout_future_steps: Optional[int] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the autoregressive AC predictor path used by predictor training."""

    pred_actions = predictor_inputs.actions
    pred_states = predictor_inputs.states
    pred_extrinsics = predictor_inputs.extrinsics
    pred_driving_command = getattr(predictor_inputs, "driving_command", None)
    pred_ego_dynamics = getattr(predictor_inputs, "ego_dynamics", None)
    predictor_inference_consistent = bool(
        _get_nested_config(config, "train", "predictor_inference_consistent", default=False)
    )
    num_obs = int(getattr(predictor_inputs, "num_observed_steps", 1))

    def step_predictor(
        tokens: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor,
    ) -> torch.Tensor:
        aux_inputs = prepare_predictor_aux_inputs(
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            config=config,
            num_observed_steps=num_obs,
            driving_command=pred_driving_command,
            ego_dynamics=pred_ego_dynamics,
        )
        z_pred = call_predictor_with_aux(predictor, tokens, aux_inputs)
        if runtime_normalize_reps:
            z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
        return z_pred

    z_tf = step_predictor(
        z_context[:, :-tokens_per_frame],
        pred_actions,
        pred_states[:, :-1],
        pred_extrinsics[:, :-1],
    )

    if not bool(_get_nested_config(config, "train", "predictor_use_z_ar_supervision", default=True)):
        return z_tf, None

    num_total = z_context.size(1) // tokens_per_frame
    requested_future_steps = _resolve_predictor_rollout_future_steps(
        rollout_future_steps,
        observed_steps=num_obs,
        total_steps=num_total,
    )
    rollout_end_step = num_obs + requested_future_steps
    if predictor_inference_consistent:
        z_rollout = z_context[:, : num_obs * tokens_per_frame]
        start_step = num_obs
    else:
        z_rollout = torch.cat([z_context[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
        start_step = 2

    for step_idx in range(start_step, rollout_end_step):
        if step_idx == num_total - 1:
            actions_full = pred_actions
            states_step = pred_states[:, :-1]
            extrinsics_step = pred_extrinsics[:, :-1]
        else:
            actions_full = pred_actions[:, :step_idx]
            states_step = pred_states[:, :step_idx]
            extrinsics_step = pred_extrinsics[:, :step_idx]

        next_tokens = step_predictor(
            z_rollout,
            actions_full,
            states_step,
            extrinsics_step,
        )[:, -tokens_per_frame:]
        z_rollout = torch.cat([z_rollout, next_tokens], dim=1)

    if predictor_inference_consistent:
        z_ar = z_rollout[:, num_obs * tokens_per_frame :]
    else:
        z_ar = z_rollout[:, tokens_per_frame:]
    return z_tf, z_ar


def _num_future_tokens_for_metrics(
    z_ar: Optional[torch.Tensor],
    h_target: torch.Tensor,
    *,
    tokens_per_frame: int,
    observed_steps: int,
) -> torch.Tensor:
    if z_ar is not None:
        value = z_ar.shape[0] * z_ar.shape[1]
    else:
        value = h_target.shape[0] * max(h_target.shape[1] - observed_steps * int(tokens_per_frame), 0)
    return torch.tensor(float(value), dtype=torch.float32, device=h_target.device)


def compute_predictor_validation_losses(
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    h_target: torch.Tensor,
    predictor_inputs: Any,
    *,
    config: Any,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    imitation_mask: Optional[torch.Tensor] = None,
    latent_dit_randomness: Optional[Dict[str, Any]] = None,
    rollout_future_steps: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Compute predictor-only validation losses for AC, parallel, or latent DiT predictors."""

    observed_steps = int(getattr(predictor_inputs, "num_observed_steps", 1))
    total_steps = h_target.shape[1] // int(tokens_per_frame)
    active_future_steps = _resolve_predictor_rollout_future_steps(
        rollout_future_steps,
        observed_steps=observed_steps,
        total_steps=total_steps,
    )
    if use_latent_dit_predictor(config):
        latent_randomness = {} if latent_dit_randomness is None else dict(latent_dit_randomness)
        supported_randomness = {
            "timesteps",
            "noise",
            "action_timesteps",
            "action_noise",
            "sample_initial_noise",
            "masked_prefix_steps",
        }
        unknown_randomness = sorted(set(latent_randomness) - supported_randomness)
        if unknown_randomness:
            raise ValueError(f"Unsupported latent-DiT validation randomness keys: {unknown_randomness}")
        if rollout_future_steps is not None:
            active_world_tokens = active_future_steps * int(tokens_per_frame)
            for key in ("noise", "sample_initial_noise"):
                value = latent_randomness.get(key)
                if value is not None:
                    latent_randomness[key] = value[:, :active_world_tokens]
            for key in ("action_timesteps", "action_noise"):
                value = latent_randomness.get(key)
                if value is not None:
                    latent_randomness[key] = value[:, :active_future_steps]
            latent_randomness["masked_prefix_steps"] = active_future_steps

        latent_output = forward_latent_dit_predictor_train(
            predictor=predictor,
            z_context=z_context,
            h_target=h_target,
            predictor_inputs=predictor_inputs,
            tokens_per_frame=tokens_per_frame,
            num_observed_steps=observed_steps,
            runtime_normalize_reps=runtime_normalize_reps,
            config=config,
            imitation_mask=imitation_mask,
            distributed_action_normalization=False,
            timesteps=latent_randomness.get("timesteps"),
            noise=latent_randomness.get("noise"),
            action_timesteps=latent_randomness.get("action_timesteps"),
            action_noise=latent_randomness.get("action_noise"),
            masked_prefix_steps=latent_randomness.get("masked_prefix_steps"),
            metadata_condition_training=False,
        )
        metrics = {
            "predictor_loss": latent_output.loss.detach(),
            "predictor_objective_loss": latent_output.objective_loss.detach(),
            "predictor_flow_loss": latent_output.flow_loss.detach(),
            "predictor_x0_loss": latent_output.x0_loss.detach(),
            "predictor_num_tokens": _num_future_tokens_for_metrics(
                latent_output.z_ar,
                h_target,
                tokens_per_frame=tokens_per_frame,
                observed_steps=observed_steps,
            ),
        }
        if latent_output.action_loss is not None:
            if latent_output.action_loss_sum is None or latent_output.action_num_samples is None:
                raise ValueError("latent-DiT runtime did not return local action validation totals")
            action_loss_sum = latent_output.action_loss_sum.detach()
            action_num_samples = latent_output.action_num_samples.detach()
            metrics["predictor_action_loss_sum"] = action_loss_sum
            metrics["predictor_action_num_samples"] = action_num_samples
            metrics["predictor_action_loss"] = action_loss_sum / action_num_samples.clamp_min(1.0)
            metrics["predictor_world_loss"] = latent_output.objective_loss.detach()
            metrics["predictor_joint_loss"] = latent_output.loss.detach()
        metrics.update(
            compute_latent_dit_per_camera_losses(
                latent_output=latent_output,
                h_target=h_target,
                config=config,
                tokens_per_frame=tokens_per_frame,
                num_observed_steps=observed_steps,
                predictor=predictor,
            )
        )
        # The teacher-forced flow_loss above does NOT reflect deployment quality: the planner and
        # NavSim eval consume tokens produced by the multi-step sampler (sample()), with the same
        # anchor convention. Measure the sampled rollout against the target future so validation
        # surfaces sampler/anchor regressions that the TF metric hides. The per-batch eval RNG is
        # already seeded by validate_predictor_one_epoch, so this metric is reproducible.
        if bool(_get_nested_config(config, "predictor_dit", "eval_sampled_rollout", default=True)):
            target_future_tokens = h_target[:, observed_steps * int(tokens_per_frame) :]
            sampled_future = sample_latent_dit_predictor(
                predictor=predictor,
                z_context=z_context,
                predictor_inputs=predictor_inputs,
                tokens_per_frame=tokens_per_frame,
                num_observed_steps=observed_steps,
                runtime_normalize_reps=runtime_normalize_reps,
                config=config,
                initial_noise=latent_randomness.get("sample_initial_noise"),
                future_steps=active_future_steps if rollout_future_steps is not None else None,
                **resolve_latent_dit_sampler_params(config).as_kwargs(),
            )
            target_future_tokens = target_future_tokens[:, : active_future_steps * int(tokens_per_frame)]
            rollout_mse = predictor_rollout_mse_per_sample(sampled_future, target_future_tokens).detach()
            metrics["predictor_rollout_mse_per_sample"] = rollout_mse
            metrics["predictor_rollout_mse"] = rollout_mse.mean()
            metrics["predictor_sampled_mse"] = metrics["predictor_rollout_mse"]
        return metrics

    if use_parallel_predictor(config):
        parallel_output = forward_parallel_predictor(
            predictor=predictor,
            observed_tokens=z_context,
            actions=predictor_inputs.actions,
            states=predictor_inputs.states,
            extrinsics=predictor_inputs.extrinsics,
            config=config,
            tokens_per_frame=tokens_per_frame,
            runtime_normalize_reps=runtime_normalize_reps,
            num_observed_steps=observed_steps,
            driving_command=getattr(predictor_inputs, "driving_command", None),
            ego_dynamics=getattr(predictor_inputs, "ego_dynamics", None),
        )
        z_tf, z_ar = parallel_output.z_pred, parallel_output.z_ar
        z_ar = truncate_validation_future_tokens(
            z_ar,
            validation_horizon=active_future_steps,
            tokens_per_frame=tokens_per_frame,
        )
    else:
        z_tf, z_ar = _forward_ac_predictor_for_validation(
            predictor,
            z_context,
            predictor_inputs,
            config=config,
            tokens_per_frame=tokens_per_frame,
            runtime_normalize_reps=runtime_normalize_reps,
            rollout_future_steps=active_future_steps,
        )

    jepa_loss, jloss, sloss = compute_predictor_jepa_losses_from_config(
        z_tf=z_tf,
        z_ar=z_ar,
        h_target=h_target,
        config=config,
        tokens_per_frame=tokens_per_frame,
        loss_fn=_predictor_loss_fn(config, tokens_per_frame),
        num_observed_steps=observed_steps,
        active_prefix_steps=active_future_steps if rollout_future_steps is not None else None,
    )
    metrics = {
        "predictor_loss": jepa_loss.detach(),
        "predictor_jloss": jloss.detach(),
        "predictor_sloss": sloss.detach(),
        "predictor_num_tokens": _num_future_tokens_for_metrics(
            z_ar,
            h_target,
            tokens_per_frame=tokens_per_frame,
            observed_steps=observed_steps,
        ),
    }
    metrics.update(
        compute_predictor_per_camera_jepa_losses(
            z_tf=z_tf,
            z_ar=z_ar,
            h_target=h_target,
            config=config,
            tokens_per_frame=tokens_per_frame,
            num_observed_steps=observed_steps,
            active_prefix_steps=active_future_steps if rollout_future_steps is not None else None,
        )
    )
    if z_ar is not None:
        target_future = h_target[
            :,
            observed_steps * int(tokens_per_frame) : (observed_steps + active_future_steps) * int(tokens_per_frame),
        ]
        rollout_mse = predictor_rollout_mse_per_sample(z_ar, target_future).detach()
        metrics["predictor_rollout_mse_per_sample"] = rollout_mse
        metrics["predictor_rollout_mse"] = rollout_mse.mean()
    return metrics


def _move_camera_metadata_to_device(
    metadata: Any, device: torch.device, *, require_camera_geometry: bool = False
) -> Dict[str, torch.Tensor]:
    camera_metadata: Dict[str, torch.Tensor] = {}
    if isinstance(metadata, dict):
        for key in ("camera_intrinsics", "camera2ego"):
            value = metadata.get(key)
            if torch.is_tensor(value):
                camera_metadata[key] = value.to(device, dtype=torch.float, non_blocking=True)
    # fail-loud (point 30): 多视角融合启用时缺相机几何直接报错，与训练侧 main_encoder_runtime 对齐，
    # 禁止 eval 静默返回 {} 后用 identity 几何跑出"看似正常"的指标。
    if require_camera_geometry:
        missing = [k for k in ("camera_intrinsics", "camera2ego") if k not in camera_metadata]
        if missing:
            raise ValueError(
                "Multi-view fusion requires camera_metadata with tensor camera_intrinsics and "
                f"camera2ego at eval, but missing/invalid: {missing}."
            )
    return camera_metadata


def _resolve_validation_dtype(config: Any) -> tuple[torch.dtype, bool]:
    which_dtype = str(_get_nested_config(config, "meta", "dtype", default="float32")).lower()
    if which_dtype == "bfloat16":
        return torch.bfloat16, True
    if which_dtype == "float16":
        return torch.float16, True
    return torch.float32, False


@torch.no_grad()
def validate_predictor_one_epoch(
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    val_loader,
    val_sampler,
    *,
    device: torch.device,
    dtype: torch.dtype,
    mixed_precision: bool,
    config: Any,
    epoch: int,
    rank: int,
    world_size: int,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    token_ae: Optional[torch.nn.Module] = None,
    multiview_fusion: Optional[torch.nn.Module] = None,
    target_multiview_fusion: Optional[torch.nn.Module] = None,
    reuse_context_as_target: Optional[bool] = None,
    timestep_sec: float = 0.5,
    rollout_future_steps: Optional[int] = None,
    validation_domain: Optional[str] = None,
    return_cohort_metrics: bool = False,
) -> Dict[str, Any]:
    """Validate predictor latent losses without running the planner head."""

    encoder_unwrapped = _unwrap_module(encoder)
    target_encoder_unwrapped = _unwrap_module(target_encoder)
    predictor_unwrapped = _unwrap_module(predictor)
    multiview_fusion_unwrapped = _unwrap_module(multiview_fusion) if multiview_fusion is not None else None
    target_multiview_fusion_unwrapped = (
        _unwrap_module(target_multiview_fusion) if target_multiview_fusion is not None else None
    )

    encoder_was_training = encoder_unwrapped.training
    target_encoder_was_training = target_encoder_unwrapped.training
    predictor_was_training = predictor_unwrapped.training
    fusion_was_training = multiview_fusion_unwrapped.training if multiview_fusion_unwrapped is not None else None
    target_fusion_was_training = (
        target_multiview_fusion_unwrapped.training if target_multiview_fusion_unwrapped is not None else None
    )

    encoder_unwrapped.eval()
    target_encoder_unwrapped.eval()
    predictor_unwrapped.eval()
    if multiview_fusion_unwrapped is not None:
        multiview_fusion_unwrapped.eval()
    if target_multiview_fusion_unwrapped is not None:
        target_multiview_fusion_unwrapped.eval()

    def restore_states() -> None:
        encoder_unwrapped.train(encoder_was_training)
        target_encoder_unwrapped.train(target_encoder_was_training)
        predictor_unwrapped.train(predictor_was_training)
        if multiview_fusion_unwrapped is not None and fusion_was_training is not None:
            multiview_fusion_unwrapped.train(fusion_was_training)
        if target_multiview_fusion_unwrapped is not None and target_fusion_was_training is not None:
            target_multiview_fusion_unwrapped.train(target_fusion_was_training)

    if val_sampler is not None:
        val_sampler.set_epoch(epoch)

    reuse_target = should_reuse_context_as_target(config, encoder_unwrapped)
    if reuse_context_as_target is not None:
        reuse_target = bool(reuse_context_as_target)

    camera_names = resolve_per_camera_names(config)
    if use_latent_dit_predictor(config):
        per_camera_metric_names = ("loss", "flow_loss", "x0_loss")
    else:
        per_camera_metric_names = ("loss", "jloss", "sloss")
    per_camera_avg_keys = tuple(
        f"predictor_per_camera_{metric_name}/{camera_name}"
        for camera_name in camera_names
        for metric_name in per_camera_metric_names
    )
    avg_keys = tuple(PREDICTOR_VALIDATION_AVG_KEYS) + per_camera_avg_keys
    metric_sums = {key: 0.0 for key in avg_keys}
    metric_counts = {key: 0.0 for key in avg_keys}
    total_tokens = 0.0
    total_samples = 0
    successful_batches = 0
    failed_batches = 0
    cohort_sums = {cohort: 0.0 for cohort in _PREDICTOR_ROLLOUT_COHORTS}
    cohort_counts = {cohort: 0.0 for cohort in _PREDICTOR_ROLLOUT_COHORTS}
    if return_cohort_metrics and validation_domain not in {"real", "counterfactual"}:
        raise ValueError("return_cohort_metrics requires validation_domain='real' or 'counterfactual'")
    local_error: Optional[BaseException] = None
    batch_idx = -1

    try:
        try:
            for batch_idx, sample in enumerate(val_loader):
                metadata = extract_batch_metadata(sample)
                counterfactual_config = getattr(config, "counterfactual_supervision", None)
                if bool(getattr(counterfactual_config, "enabled", False)) or return_cohort_metrics:
                    sample_masks = build_counterfactual_sample_masks(metadata, device)
                    imitation_mask = sample_masks.imitation
                else:
                    sample_masks = None
                    imitation_mask = None
                metadata_valid_mask = metadata.get("metadata_valid_mask") if isinstance(metadata, dict) else None
                observed_metadata_valid_mask = (
                    metadata.get("observed_metadata_valid_mask") if isinstance(metadata, dict) else None
                )
                context_frames = sample[0].to(device, non_blocking=True)
                actions = sample[1].to(device, dtype=torch.float, non_blocking=True)
                states = sample[2].to(device, dtype=torch.float, non_blocking=True)
                extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)
                # 槽位值本身可为 None（B2D 数据缺 command_near/ego_vel 时 collate 置 None）。
                driving_command = (
                    sample[5].to(device, dtype=torch.float, non_blocking=True)
                    if len(sample) > 5 and sample[5] is not None
                    else None
                )
                ego_dynamics = (
                    sample[6].to(device, dtype=torch.float, non_blocking=True)
                    if len(sample) > 6 and sample[6] is not None
                    else None
                )
                camera_metadata = _move_camera_metadata_to_device(
                    metadata, device, require_camera_geometry=multiview_fusion_unwrapped is not None
                )

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    z_context = forward_main_context(
                        encoder_unwrapped,
                        context_frames,
                        config=config,
                        runtime_normalize_reps=runtime_normalize_reps,
                        token_ae=token_ae,
                        multiview_fusion=multiview_fusion_unwrapped,
                        camera_metadata=camera_metadata,
                    )
                    if reuse_target:
                        h_target = z_context.detach()
                    else:
                        h_target = forward_main_target(
                            target_encoder_unwrapped,
                            context_frames,
                            config=config,
                            runtime_normalize_reps=runtime_normalize_reps,
                            token_ae=token_ae,
                            multiview_fusion=target_multiview_fusion_unwrapped,
                            camera_metadata=camera_metadata,
                        )

                    if use_parallel_predictor(config):
                        predictor_inputs = build_parallel_predictor_timeline_inputs(
                            actions=actions,
                            states=states,
                            extrinsics=extrinsics,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            encoder=encoder_unwrapped,
                            dt=timestep_sec,
                            metadata_valid_mask=metadata_valid_mask,
                            observed_metadata_valid_mask=observed_metadata_valid_mask,
                        )
                    else:
                        predictor_inputs = build_predictor_timeline_inputs(
                            actions=actions,
                            states=states,
                            extrinsics=extrinsics,
                            driving_command=driving_command,
                            ego_dynamics=ego_dynamics,
                            config=config,
                            encoder=encoder_unwrapped,
                            dt=timestep_sec,
                            metadata_valid_mask=metadata_valid_mask,
                            observed_metadata_valid_mask=observed_metadata_valid_mask,
                        )

                    latent_dit_randomness = None
                    if use_latent_dit_predictor(config):
                        batch_size = int(z_context.shape[0])
                        sample_ids = resolve_stable_sample_ids(metadata, batch_size=batch_size)
                        base_seed = int(_get_nested_config(config, "meta", "seed", default=0))
                        observed_steps = int(getattr(predictor_inputs, "num_observed_steps", 1))
                        future_offset = observed_steps * int(tokens_per_frame)
                        target_future_tokens = h_target[:, future_offset:]
                        if target_future_tokens.shape[1] <= 0:
                            raise ValueError("Latent-DiT validation requires at least one target future token")
                        if target_future_tokens.shape[1] % int(tokens_per_frame) != 0:
                            raise ValueError(
                                "Latent-DiT validation target future must be frame aligned: "
                                f"tokens={target_future_tokens.shape[1]}, tokens_per_frame={tokens_per_frame}"
                            )
                        full_horizon = int(target_future_tokens.shape[1]) // int(tokens_per_frame)
                        timestep_normal = validation_randn(
                            (batch_size,),
                            base_seed=base_seed,
                            sample_ids=sample_ids,
                            protocol="predictor-validation/full",
                            horizon=None,
                            stream="training_timestep",
                            device=target_future_tokens.device,
                            dtype=torch.float32,
                        )
                        world_timesteps = torch.sigmoid(timestep_normal).clamp(1e-5, 1.0 - 1e-5)
                        world_noise = validation_randn(
                            target_future_tokens.shape,
                            base_seed=base_seed,
                            sample_ids=sample_ids,
                            protocol="predictor-validation/full",
                            horizon=None,
                            stream="training_world_noise",
                            device=target_future_tokens.device,
                            dtype=target_future_tokens.dtype,
                        )
                        predictor_core = predictor_unwrapped
                        sample_initial_noise = validation_randn(
                            (
                                batch_size,
                                int(predictor_core.num_future_tokens),
                                int(predictor_core.embed_dim),
                            ),
                            base_seed=base_seed,
                            sample_ids=sample_ids,
                            protocol="predictor-validation/full",
                            horizon=None,
                            stream="sample_world_initial_noise",
                            device=z_context.device,
                            dtype=z_context.dtype,
                        )
                        latent_dit_randomness = {
                            "timesteps": world_timesteps,
                            "noise": world_noise,
                            "sample_initial_noise": sample_initial_noise,
                            "masked_prefix_steps": full_horizon,
                        }
                        if bool(getattr(predictor_core, "joint_action_enabled", False)):
                            action_shape = (
                                batch_size,
                                full_horizon,
                                int(predictor_core.joint_action_dim),
                            )
                            action_noise = validation_randn(
                                action_shape,
                                base_seed=base_seed,
                                sample_ids=sample_ids,
                                protocol="predictor-validation/full",
                                horizon=None,
                                stream="training_action_noise",
                                device=actions.device,
                                dtype=actions.dtype,
                            )
                            action_noise_mode = str(
                                _get_nested_config(
                                    config,
                                    "predictor_dit",
                                    "joint_action_noise_mode",
                                    default="shared",
                                )
                            ).lower()
                            if action_noise_mode == "shared":
                                action_timesteps = world_timesteps.to(dtype=actions.dtype)[:, None].expand(
                                    -1, full_horizon
                                )
                            elif action_noise_mode == "decoupled":
                                action_timestep_normal = validation_randn(
                                    (batch_size, full_horizon),
                                    base_seed=base_seed,
                                    sample_ids=sample_ids,
                                    protocol="predictor-validation/full",
                                    horizon=None,
                                    stream="training_action_timestep",
                                    device=actions.device,
                                    dtype=actions.dtype,
                                )
                                action_timesteps = torch.sigmoid(action_timestep_normal).clamp(1e-5, 1.0 - 1e-5)
                            else:
                                raise ValueError(
                                    "predictor_dit.joint_action_noise_mode must be 'shared' or 'decoupled', "
                                    f"got {action_noise_mode!r}"
                                )
                            latent_dit_randomness.update(
                                {
                                    "action_timesteps": action_timesteps,
                                    "action_noise": action_noise,
                                }
                            )

                    batch_metrics = compute_predictor_validation_losses(
                        predictor=predictor_unwrapped,
                        z_context=z_context,
                        h_target=h_target,
                        predictor_inputs=predictor_inputs,
                        config=config,
                        tokens_per_frame=tokens_per_frame,
                        runtime_normalize_reps=runtime_normalize_reps,
                        imitation_mask=imitation_mask,
                        latent_dit_randomness=latent_dit_randomness,
                        rollout_future_steps=rollout_future_steps,
                    )

                batch_tokens = float(batch_metrics["predictor_num_tokens"].item())
                total_tokens += batch_tokens
                total_samples += int(context_frames.shape[0])
                successful_batches += 1
                _accumulate_validation_batch_metrics(
                    metric_sums,
                    metric_counts,
                    avg_keys,
                    batch_metrics,
                    batch_tokens=batch_tokens,
                )
                if return_cohort_metrics:
                    rollout_mse = batch_metrics.get("predictor_rollout_mse_per_sample")
                    if rollout_mse is None:
                        raise ValueError("predictor cohort validation requires predictor_rollout_mse_per_sample")
                    _accumulate_predictor_rollout_cohorts(
                        cohort_sums,
                        cohort_counts,
                        rollout_mse,
                        validation_domain=str(validation_domain),
                        sample_masks=sample_masks,
                    )

                if rank == 0 and batch_idx % 50 == 0:
                    logger.info(
                        "Predictor validation Epoch %s, Batch %s/%s, loss=%.5f",
                        epoch,
                        batch_idx,
                        len(val_loader),
                        float(batch_metrics["predictor_loss"].item()),
                    )
        except Exception as exc:
            if batch_idx < 0:
                local_error = RuntimeError("Predictor validation dataloader iteration failed")
                local_error.__cause__ = exc
            else:
                local_error = wrap_validation_batch_error("Predictor validation", batch_idx, exc)
    finally:
        restore_states()

    raise_if_validation_failed(
        local_error,
        validation_name="Predictor validation",
        device=device,
        world_size=world_size,
    )

    (
        metric_sums,
        metric_counts,
        total_tokens,
        total_samples,
        successful_batches,
        failed_batches,
    ) = _reduce_validation_totals(
        metric_sums,
        metric_counts,
        avg_keys,
        total_tokens=total_tokens,
        total_samples=total_samples,
        successful_batches=successful_batches,
        failed_batches=failed_batches,
        device=device,
        world_size=world_size,
    )

    if successful_batches == 0 or total_tokens <= 0:
        raise RuntimeError(
            f"Predictor validation produced zero global successful batches "
            f"(failed_batches={failed_batches}, total_tokens={total_tokens})."
        )

    result: Dict[str, float] = {}
    for key in avg_keys:
        if metric_counts[key] > 0:
            result[key] = metric_sums[key] / metric_counts[key]
    reconstruct_joint_validation_loss(result, config)
    for camera_name in camera_names:
        token_key = f"predictor_per_camera_num_tokens/{camera_name}"
        count_key = f"predictor_per_camera_loss/{camera_name}"
        if metric_counts.get(count_key, 0.0) > 0:
            result[token_key] = metric_counts[count_key]
    result["predictor_num_tokens"] = total_tokens
    result["predictor_num_samples"] = float(total_samples)
    result["predictor_batches"] = float(successful_batches)
    result["predictor_failed_batches"] = float(failed_batches)
    if return_cohort_metrics:
        cohort_values = []
        for cohort in _PREDICTOR_ROLLOUT_COHORTS:
            cohort_values.extend((cohort_sums[cohort], cohort_counts[cohort]))
        cohort_tensor = torch.tensor(cohort_values, dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(cohort_tensor, op=dist.ReduceOp.SUM)
        reduced_cohorts = cohort_tensor.tolist()
        for index, cohort in enumerate(_PREDICTOR_ROLLOUT_COHORTS):
            cohort_sums[cohort] = reduced_cohorts[2 * index]
            cohort_counts[cohort] = reduced_cohorts[2 * index + 1]

        required_cohorts = ("all",) if validation_domain == "real" else _PREDICTOR_ROLLOUT_COHORTS
        rows: Dict[str, Dict[str, float]] = {"all": result}
        for cohort in required_cohorts:
            count = cohort_counts[cohort]
            if count <= 0.0:
                raise RuntimeError(f"predictor validation cohort {validation_domain}/{cohort} has zero samples")
            cohort_mse = cohort_sums[cohort] / count
            if cohort == "all":
                rows["all"]["predictor_rollout_mse"] = cohort_mse
            else:
                rows[cohort] = {"predictor_rollout_mse": cohort_mse}
        return rows
    return result


def run_predictor_validation(
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    val_loader,
    val_sampler,
    config: Any,
    epoch: int,
    rank: int,
    world_size: int,
    *,
    token_ae: Optional[torch.nn.Module] = None,
    runtime_normalize_reps: Optional[bool] = None,
    multiview_fusion: Optional[torch.nn.Module] = None,
    target_multiview_fusion: Optional[torch.nn.Module] = None,
    reuse_context_as_target: Optional[bool] = None,
    rollout_future_steps: Optional[int] = None,
    validation_domain: Optional[str] = None,
    return_cohort_metrics: bool = False,
) -> Dict[str, Any]:
    """Entry point for predictor-only validation."""

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    dtype, mixed_precision = _resolve_validation_dtype(config)
    tokens_per_frame = resolve_main_encoder_tokens_per_frame(config, encoder)
    normalize_reps = (
        bool(runtime_normalize_reps)
        if runtime_normalize_reps is not None
        else resolve_runtime_normalize_reps(config, token_ae=token_ae)
    )
    fps = _get_nested_config(config, "data", "fps", default=None)
    diff_dt = _get_nested_config(config, "planner", "diff_dt", default=None)
    timestep_sec = resolve_validation_timestep_sec(fps=fps, diff_dt=diff_dt, default=0.5)

    if rank == 0:
        logger.info(
            "Starting predictor validation for epoch %s: tokens_per_frame=%d, normalize_reps=%s, "
            "predictor_type=%s, use_parallel_predictor=%s",
            epoch,
            tokens_per_frame,
            normalize_reps,
            str(_get_nested_config(config, "train", "predictor_type", default="ac_transformer")),
            bool(_get_nested_config(config, "train", "use_parallel_predictor", default=False)),
        )

    metrics = validate_predictor_one_epoch(
        encoder=encoder,
        target_encoder=target_encoder,
        predictor=predictor,
        val_loader=val_loader,
        val_sampler=val_sampler,
        device=device,
        dtype=dtype,
        mixed_precision=mixed_precision,
        config=config,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
        tokens_per_frame=tokens_per_frame,
        runtime_normalize_reps=normalize_reps,
        token_ae=token_ae,
        multiview_fusion=multiview_fusion,
        target_multiview_fusion=target_multiview_fusion,
        reuse_context_as_target=reuse_context_as_target,
        timestep_sec=timestep_sec,
        rollout_future_steps=rollout_future_steps,
        validation_domain=validation_domain,
        return_cohort_metrics=return_cohort_metrics,
    )

    if rank == 0:
        log_metrics = metrics["all"] if return_cohort_metrics else metrics
        logger.info("=" * 50)
        logger.info("Predictor Validation Results - Epoch %s:", epoch)
        logger.info("  Predictor loss: %.5f", log_metrics["predictor_loss"])
        if "predictor_jloss" in log_metrics:
            logger.info("  Predictor jloss: %.5f", log_metrics["predictor_jloss"])
        if "predictor_sloss" in log_metrics:
            logger.info("  Predictor sloss: %.5f", log_metrics["predictor_sloss"])
        if "predictor_flow_loss" in log_metrics:
            logger.info("  Predictor flow loss: %.5f", log_metrics["predictor_flow_loss"])
        if "predictor_x0_loss" in log_metrics:
            logger.info("  Predictor x0 loss: %.5f", log_metrics["predictor_x0_loss"])
        if "predictor_action_loss" in log_metrics:
            logger.info("  Predictor action loss: %.5f", log_metrics["predictor_action_loss"])
        if "predictor_joint_loss" in log_metrics:
            logger.info("  Predictor joint loss: %.5f", log_metrics["predictor_joint_loss"])
        logger.info("  Predictor tokens: %.0f", log_metrics["predictor_num_tokens"])
        logger.info("=" * 50)
    return metrics
