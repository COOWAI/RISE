"""Fixed, target-independent validation for the temporal trajectory value head."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.config import resolve_main_encoder_tokens_per_frame
from app.vjepa_cowa_world_model.training.counterfactual_supervision import build_counterfactual_sample_masks
from app.vjepa_cowa_world_model.training.models import resolve_runtime_normalize_reps
from app.vjepa_cowa_world_model.training.predictor_parallel import forward_parallel_predictor, use_parallel_predictor
from app.vjepa_cowa_world_model.training.predictor_validation import (
    _forward_ac_predictor_for_validation,
    _move_camera_metadata_to_device,
)
from app.vjepa_cowa_world_model.training.runtimes.latent_diffusion_runtime import (
    resolve_latent_dit_sampler_params,
    sample_latent_dit_predictor,
    use_latent_dit_predictor,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (
    build_parallel_predictor_timeline_inputs,
    build_predictor_timeline_inputs,
    forward_main_context,
)
from app.vjepa_cowa_world_model.training.validation_distributed import (
    distributed_validation_active,
    raise_if_validation_failed,
    wrap_validation_batch_error,
)
from app.vjepa_cowa_world_model.training.validation_rng import resolve_stable_sample_ids, validation_randn
from app.vjepa_cowa_world_model.training.validation_suite import build_validation_data_signature
from app.vjepa_cowa_world_model.training.value_planning import (
    _aggregate_rewards_for_frame_stride,
    build_value_future_gt_trajectory,
    compute_online_rewards,
)
from app.vjepa_cowa_world_model.utils.eval_determinism import extract_batch_metadata
from app.vjepa_cowa_world_model.utils.module_utils import get_nested_config, unwrap_module
from src.utils.logging import get_logger

logger = get_logger(__name__)

VALUE_VALIDATION_SIGNATURE_VERSION = "value_validation_v2"


def build_value_validation_signature(
    *,
    value_metric_signature: Dict[str, Any],
    validation_data_semantics: Dict[str, Any],
    val_roots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build value checkpoint provenance independently of the rollout suite toggle."""

    if not isinstance(value_metric_signature, dict) or not value_metric_signature:
        raise ValueError("value_metric_signature must be a non-empty dict")
    return {
        "version": VALUE_VALIDATION_SIGNATURE_VERSION,
        "protocol": "full",
        "selector": "value_validation_loss",
        "domains": ["real", "counterfactual"],
        "cohorts": ["all", "safe", "hazard"],
        "value_metric_signature": dict(value_metric_signature),
        "validation_data": build_validation_data_signature(validation_data_semantics, val_roots),
    }


def _counterfactual_validation_cohorts_enabled(config: Any) -> bool:
    """Return whether validation requires real/safe/hazard cohort masks."""

    return bool(
        get_nested_config(config, "validation_suite", "enabled", default=False)
        or get_nested_config(config, "counterfactual_supervision", "enabled", default=False)
    )


def fixed_truncated_returns(rewards: torch.Tensor, *, gamma_stride: float) -> torch.Tensor:
    """Return ``G_t = r_t + gamma_stride * G_{t+1}`` with ``G_F = 0``.

    These targets intentionally do not reference the EMA value head.  They are
    therefore fixed across training and suitable for checkpoint selection.
    """

    if rewards.ndim != 2 or rewards.shape[1] < 1:
        raise ValueError(f"rewards must be non-empty [B, F], got {tuple(rewards.shape)}")
    gamma_stride = float(gamma_stride)
    if not math.isfinite(gamma_stride) or not 0.0 <= gamma_stride <= 1.0:
        raise ValueError(f"gamma_stride must be finite and in [0, 1], got {gamma_stride}")
    returns = torch.zeros_like(rewards)
    next_return = rewards.new_zeros(rewards.shape[0])
    for step in range(rewards.shape[1] - 1, -1, -1):
        next_return = rewards[:, step] + gamma_stride * next_return
        returns[:, step] = next_return
    return returns


def _pairwise_ranking_metrics(
    hazard_scores: torch.Tensor,
    comparator_scores: torch.Tensor,
    *,
    margin: float,
) -> tuple[float, float, float]:
    """Compute exact all-pairs ranking metrics in ``O(N log N)`` memory/time.

    For each comparator score ``c``, the hinge-active hazards satisfy
    ``h > c - margin``.  Sorting hazards therefore turns the dense pair matrix
    into searchsorted counts and prefix-sum tails.
    """

    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError(f"value validation margin must be finite and nonnegative, got {margin}")
    hazards = hazard_scores.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    comparators = comparator_scores.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    if hazards.numel() == 0 or comparators.numel() == 0:
        return 0.0, 0.0, 0.0
    if not bool(torch.isfinite(hazards).all().item()) or not bool(torch.isfinite(comparators).all().item()):
        raise ValueError("value validation episode scores contain NaN/Inf")

    sorted_hazards = torch.sort(hazards).values
    prefix_sums = torch.cat(
        [sorted_hazards.new_zeros(1), torch.cumsum(sorted_hazards, dim=0)],
        dim=0,
    )
    num_hazards = int(sorted_hazards.numel())
    num_pairs = float(num_hazards * int(comparators.numel()))

    active_start = torch.searchsorted(sorted_hazards, comparators - margin, right=True)
    active_count = num_hazards - active_start
    active_hazard_sum = prefix_sums[-1] - prefix_sums[active_start]
    hinge_sum = active_hazard_sum + active_count * (margin - comparators)
    ranking_loss = float(hinge_sum.clamp_min(0.0).sum().item() / num_pairs)

    positive_pairs = torch.searchsorted(sorted_hazards, comparators, right=False).sum()
    margin_pairs = torch.searchsorted(sorted_hazards, comparators - margin, right=True).sum()
    ranking_accuracy = float(positive_pairs.item() / num_pairs)
    margin_accuracy = float(margin_pairs.item() / num_pairs)
    return ranking_loss, ranking_accuracy, margin_accuracy


def _collective_device() -> torch.device:
    if dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@dataclass
class ValueValidationAccumulator:
    """Stream fixed metrics and retain only compact per-episode ranking scores."""

    _calibration_sum: float = 0.0
    _calibration_count: int = 0
    _eligible_count: int = 0
    _hazard_score_sum: float = 0.0
    _hazard_count: int = 0
    _comparator_score_sum: float = 0.0
    _comparator_count: int = 0
    _sample_count: int = 0
    _horizon: Optional[int] = None
    _hazard_scores: List[torch.Tensor] = field(default_factory=list)
    _comparator_scores: List[torch.Tensor] = field(default_factory=list)

    def update(
        self,
        *,
        predicted_values: torch.Tensor,
        returns: torch.Tensor,
        eligible_mask: torch.Tensor,
        hazard_mask: torch.Tensor,
        comparator_mask: torch.Tensor,
    ) -> None:
        """Add one local batch after strict shape/category validation."""

        if predicted_values.ndim != 2 or predicted_values.shape[1] < 1:
            raise ValueError(f"predicted_values must be non-empty [B, F], got {tuple(predicted_values.shape)}")
        if returns.shape != predicted_values.shape:
            raise ValueError(
                f"returns shape {tuple(returns.shape)} != predicted_values {tuple(predicted_values.shape)}"
            )
        for name, mask in (
            ("eligible_mask", eligible_mask),
            ("hazard_mask", hazard_mask),
            ("comparator_mask", comparator_mask),
        ):
            if mask.dtype != torch.bool or mask.shape != predicted_values.shape[:1]:
                raise ValueError(f"{name} must be bool [B], got dtype={mask.dtype}, shape={tuple(mask.shape)}")
        if bool((hazard_mask & comparator_mask).any().item()):
            raise ValueError("hazard and comparator masks must be disjoint")
        if bool(((hazard_mask | comparator_mask) & ~eligible_mask).any().item()):
            raise ValueError("hazard/comparator masks must be subsets of eligible_mask")
        if not bool(torch.isfinite(predicted_values).all().item()):
            raise ValueError("value validation predicted_values contain NaN/Inf")
        if not bool(torch.isfinite(returns).all().item()):
            raise ValueError("value validation returns contain NaN/Inf")

        horizon = int(predicted_values.shape[1])
        if self._horizon is None:
            self._horizon = horizon
        elif self._horizon != horizon:
            raise ValueError(
                "value validation batches produced inconsistent horizons: " f"expected {self._horizon}, got {horizon}"
            )

        predicted = predicted_values.detach().float()
        targets = returns.detach().float()
        eligible_count = int(eligible_mask.sum().item())
        if eligible_count:
            calibration_sum = F.smooth_l1_loss(
                predicted[eligible_mask],
                targets[eligible_mask],
                reduction="sum",
            )
            self._calibration_sum += float(calibration_sum.item())
            self._calibration_count += eligible_count * horizon
            self._eligible_count += eligible_count

        episode_scores = predicted.mean(dim=1).cpu()
        hazard_scores = episode_scores[hazard_mask.cpu()]
        comparator_scores = episode_scores[comparator_mask.cpu()]
        if hazard_scores.numel():
            self._hazard_scores.append(hazard_scores)
            self._hazard_score_sum += float(hazard_scores.double().sum().item())
            self._hazard_count += int(hazard_scores.numel())
        if comparator_scores.numel():
            self._comparator_scores.append(comparator_scores)
            self._comparator_score_sum += float(comparator_scores.double().sum().item())
            self._comparator_count += int(comparator_scores.numel())
        self._sample_count += int(predicted.shape[0])

    def _local_scores(self) -> tuple[torch.Tensor, torch.Tensor]:
        hazard_scores = torch.cat(self._hazard_scores) if self._hazard_scores else torch.empty(0)
        comparator_scores = torch.cat(self._comparator_scores) if self._comparator_scores else torch.empty(0)
        return hazard_scores, comparator_scores

    def finalize(
        self,
        *,
        margin: float,
        calibration_weight: float,
        ranking_weight: float,
        require_ranking_categories: bool,
    ) -> Dict[str, float]:
        """Compute full-dataset calibration, pairwise ranking, and composite loss."""

        for name, value in (
            ("margin", margin),
            ("calibration_weight", calibration_weight),
            ("ranking_weight", ranking_weight),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"value validation {name} must be finite and nonnegative, got {value}")
        active = distributed_validation_active(
            dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        )
        device = _collective_device()
        local_has_samples = int(self._sample_count > 0)
        local_horizon = int(self._horizon or 0)
        totals = torch.tensor(
            [
                self._calibration_sum,
                float(self._calibration_count),
                float(self._eligible_count),
                self._hazard_score_sum,
                float(self._hazard_count),
                self._comparator_score_sum,
                float(self._comparator_count),
                float(self._sample_count),
                float(local_has_samples),
                float(local_horizon * local_has_samples),
                float(local_horizon * local_horizon * local_has_samples),
            ],
            dtype=torch.float64,
            device=device,
        )
        if active:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        (
            calibration_sum,
            calibration_count,
            eligible_count_value,
            hazard_score_sum,
            hazard_count_value,
            comparator_score_sum,
            comparator_count_value,
            _sample_count,
            nonempty_rank_count,
            horizon_sum,
            horizon_sq_sum,
        ) = totals.tolist()
        eligible_count = int(eligible_count_value)
        if eligible_count == 0:
            raise ValueError("value validation has zero globally eligible samples")
        if calibration_count <= 0:
            raise ValueError("value validation has zero globally eligible timesteps")
        if nonempty_rank_count > 0 and (horizon_sq_sum * nonempty_rank_count - horizon_sum * horizon_sum) > 1e-9:
            raise ValueError("value validation ranks produced inconsistent horizons")

        calibration = calibration_sum / calibration_count
        hazard_count = int(hazard_count_value)
        comparator_count = int(comparator_count_value)
        if require_ranking_categories and hazard_count == 0:
            raise ValueError("value validation ranking requires at least one global ego-hazard sample")
        if require_ranking_categories and comparator_count == 0:
            raise ValueError("value validation ranking requires at least one global comparator sample")

        local_hazards, local_comparators = self._local_scores()
        if active:
            rank = int(dist.get_rank())
            gathered: Optional[List[tuple[List[float], List[float]]]] = (
                [([], []) for _ in range(dist.get_world_size())] if rank == 0 else None
            )
            dist.gather_object(
                (local_hazards.tolist(), local_comparators.tolist()),
                gathered,
                dst=0,
            )
            if rank == 0:
                if gathered is None:
                    raise RuntimeError("rank 0 did not receive value validation ranking payloads")
                hazard_scores = torch.tensor(
                    [score for hazard_values, _ in gathered for score in hazard_values],
                    dtype=torch.float64,
                )
                comparator_scores = torch.tensor(
                    [score for _, comparator_values in gathered for score in comparator_values],
                    dtype=torch.float64,
                )
                ranking_values = _pairwise_ranking_metrics(
                    hazard_scores,
                    comparator_scores,
                    margin=float(margin),
                )
            else:
                ranking_values = (0.0, 0.0, 0.0)
        else:
            ranking_values = _pairwise_ranking_metrics(
                local_hazards,
                local_comparators,
                margin=float(margin),
            )

        ranking_loss, ranking_accuracy, margin_accuracy = ranking_values
        composite = float(calibration_weight) * calibration + float(ranking_weight) * ranking_loss
        metric_values = [
            float(calibration),
            float(ranking_loss),
            float(ranking_accuracy),
            float(margin_accuracy),
            float(hazard_score_sum / hazard_count) if hazard_count else 0.0,
            float(comparator_score_sum / comparator_count) if comparator_count else 0.0,
            float(hazard_count),
            float(comparator_count),
            float(eligible_count),
            float(composite),
        ]
        if active:
            metrics_tensor = torch.tensor(metric_values, dtype=torch.float64, device=device)
            dist.broadcast(metrics_tensor, src=0)
            metric_values = metrics_tensor.tolist()
        if not all(math.isfinite(value) for value in metric_values):
            raise ValueError("value_validation_loss is NaN/Inf")
        (
            calibration,
            ranking_loss,
            ranking_accuracy,
            margin_accuracy,
            hazard_mean,
            comparator_mean,
            hazard_count_value,
            comparator_count_value,
            eligible_count_value,
            composite,
        ) = metric_values
        return {
            "value_calibration_loss": float(calibration),
            "value_ranking_loss": float(ranking_loss),
            "value_ranking_accuracy": float(ranking_accuracy),
            "value_margin_accuracy": float(margin_accuracy),
            "value_hazard_mean": float(hazard_mean),
            "value_comparator_mean": float(comparator_mean),
            "value_hazard_count": float(hazard_count_value),
            "value_comparator_count": float(comparator_count_value),
            "value_eligible_count": float(eligible_count_value),
            "value_validation_loss": float(composite),
        }


def _validation_dtype(config: Any) -> tuple[torch.dtype, bool]:
    dtype_name = str(get_nested_config(config, "meta", "dtype", default="float32")).lower()
    if dtype_name == "bfloat16":
        return torch.bfloat16, True
    if dtype_name == "float16":
        return torch.float16, True
    return torch.float32, False


@torch.no_grad()
def run_value_validation(
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    value_head: torch.nn.Module,
    val_loader: Any,
    val_sampler: Any,
    config: Any,
    epoch: int,
    rank: int,
    world_size: int,
    token_ae: Optional[torch.nn.Module] = None,
    runtime_normalize_reps: Optional[bool] = None,
    multiview_fusion: Optional[torch.nn.Module] = None,
) -> Dict[str, float]:
    """Evaluate a trainable value head against fixed GT truncated returns."""

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    encoder_unwrapped = unwrap_module(encoder)
    predictor_unwrapped = unwrap_module(predictor)
    value_head_unwrapped = unwrap_module(value_head)
    fusion_unwrapped = unwrap_module(multiview_fusion) if multiview_fusion is not None else None
    training_states = (
        encoder_unwrapped.training,
        predictor_unwrapped.training,
        value_head_unwrapped.training,
        fusion_unwrapped.training if fusion_unwrapped is not None else None,
    )
    encoder_unwrapped.eval()
    predictor_unwrapped.eval()
    value_head_unwrapped.eval()
    if fusion_unwrapped is not None:
        fusion_unwrapped.eval()
    if val_sampler is not None:
        val_sampler.set_epoch(epoch)

    tokens_per_frame = resolve_main_encoder_tokens_per_frame(config, encoder_unwrapped)
    normalize_reps = (
        bool(runtime_normalize_reps)
        if runtime_normalize_reps is not None
        else resolve_runtime_normalize_reps(config, token_ae=token_ae)
    )
    dtype, mixed_precision = _validation_dtype(config)
    fps = float(get_nested_config(config, "data", "fps", default=2.0))
    timestep_sec = 1.0 / max(fps, 1.0)
    accumulator = ValueValidationAccumulator()
    counterfactual_cohorts_enabled = _counterfactual_validation_cohorts_enabled(config)
    local_error: Optional[BaseException] = None
    batch_idx = -1

    try:
        try:
            for batch_idx, sample in enumerate(val_loader):
                metadata = extract_batch_metadata(sample)
                sample_masks = (
                    build_counterfactual_sample_masks(metadata, device) if counterfactual_cohorts_enabled else None
                )
                context_frames = sample[0].to(device, non_blocking=True)
                actions = sample[1].to(device, dtype=torch.float, non_blocking=True)
                states = sample[2].to(device, dtype=torch.float, non_blocking=True)
                extrinsics = sample[3].to(device, dtype=torch.float, non_blocking=True)
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
                metadata_valid_mask = metadata.get("metadata_valid_mask") if isinstance(metadata, dict) else None
                observed_metadata_valid_mask = (
                    metadata.get("observed_metadata_valid_mask") if isinstance(metadata, dict) else None
                )
                camera_metadata = _move_camera_metadata_to_device(
                    metadata,
                    device,
                    require_camera_geometry=fusion_unwrapped is not None,
                )

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    z_context = forward_main_context(
                        encoder_unwrapped,
                        context_frames,
                        config=config,
                        runtime_normalize_reps=normalize_reps,
                        token_ae=token_ae,
                        multiview_fusion=fusion_unwrapped,
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
                        z_future = forward_parallel_predictor(
                            predictor=predictor_unwrapped,
                            observed_tokens=z_context,
                            actions=predictor_inputs.actions,
                            states=predictor_inputs.states,
                            extrinsics=predictor_inputs.extrinsics,
                            config=config,
                            tokens_per_frame=tokens_per_frame,
                            runtime_normalize_reps=normalize_reps,
                            num_observed_steps=predictor_inputs.num_observed_steps,
                            driving_command=predictor_inputs.driving_command,
                            ego_dynamics=predictor_inputs.ego_dynamics,
                        ).z_future
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
                        if use_latent_dit_predictor(config):
                            batch_size = int(z_context.shape[0])
                            sample_ids = resolve_stable_sample_ids(metadata, batch_size=batch_size)
                            initial_noise = validation_randn(
                                (
                                    batch_size,
                                    int(predictor_unwrapped.num_future_tokens),
                                    int(predictor_unwrapped.embed_dim),
                                ),
                                base_seed=int(get_nested_config(config, "meta", "seed", default=0)),
                                sample_ids=sample_ids,
                                protocol="value-validation/full",
                                horizon=None,
                                stream="predictor_initial_noise",
                                device=z_context.device,
                                dtype=z_context.dtype,
                            )
                            z_future = sample_latent_dit_predictor(
                                predictor=predictor_unwrapped,
                                z_context=z_context,
                                predictor_inputs=predictor_inputs,
                                tokens_per_frame=tokens_per_frame,
                                num_observed_steps=predictor_inputs.num_observed_steps,
                                runtime_normalize_reps=normalize_reps,
                                config=config,
                                initial_noise=initial_noise,
                                **resolve_latent_dit_sampler_params(config).as_kwargs(),
                            )
                        else:
                            _, z_future = _forward_ac_predictor_for_validation(
                                predictor_unwrapped,
                                z_context,
                                predictor_inputs,
                                config=config,
                                tokens_per_frame=tokens_per_frame,
                                runtime_normalize_reps=normalize_reps,
                            )
                            if z_future is None:
                                raise ValueError("value validation requires an autoregressive predictor rollout")
                    predicted_all = value_head_unwrapped(z_future, tokens_per_frame=tokens_per_frame)

                frame_stride = int(predictor_inputs.frame_stride)
                gt_trajectory = build_value_future_gt_trajectory(states, config)
                horizon = min(
                    int(get_nested_config(config, "value_planning", "bootstrap_horizon", default=3)),
                    int(predicted_all.shape[1]),
                    int(gt_trajectory.shape[1]) // frame_stride,
                )
                if horizon < 1:
                    raise ValueError("value validation produced an empty prediction/reward horizon")
                predicted_values = predicted_all[:, :horizon].float()
                raw_rewards = compute_online_rewards(
                    gt_trajectory[:, None, : horizon * frame_stride],
                    progress_weight=float(get_nested_config(config, "value_planning", "progress_weight", default=1.0)),
                    comfort_weight=float(get_nested_config(config, "value_planning", "comfort_weight", default=0.2)),
                ).squeeze(1)
                gamma = float(get_nested_config(config, "value_planning", "gamma", default=0.99))
                rewards = _aggregate_rewards_for_frame_stride(
                    raw_rewards,
                    frame_stride=frame_stride,
                    gamma=gamma,
                    horizon=horizon,
                )
                returns = fixed_truncated_returns(rewards, gamma_stride=gamma**frame_stride)
                if sample_masks is None:
                    eligible = torch.ones(predicted_values.shape[0], dtype=torch.bool, device=device)
                    hazard = torch.zeros_like(eligible)
                    comparator = eligible
                else:
                    eligible = sample_masks.value
                    hazard = sample_masks.cf_ego_hazard & eligible
                    comparator = eligible & ~hazard
                accumulator.update(
                    predicted_values=predicted_values,
                    returns=returns,
                    eligible_mask=eligible,
                    hazard_mask=hazard,
                    comparator_mask=comparator,
                )
        except Exception as exc:
            if batch_idx < 0:
                local_error = RuntimeError("Value validation dataloader iteration failed")
                local_error.__cause__ = exc
            else:
                local_error = wrap_validation_batch_error("Value validation", batch_idx, exc)
    finally:
        encoder_unwrapped.train(training_states[0])
        predictor_unwrapped.train(training_states[1])
        value_head_unwrapped.train(training_states[2])
        if fusion_unwrapped is not None and training_states[3] is not None:
            fusion_unwrapped.train(training_states[3])

    raise_if_validation_failed(
        local_error,
        validation_name="Value validation",
        device=device,
        world_size=world_size,
    )

    calibration_weight = float(
        get_nested_config(config, "value_planning", "validation_calibration_weight", default=1.0)
    )
    ranking_weight = float(get_nested_config(config, "value_planning", "validation_ranking_weight", default=1.0))
    metrics = accumulator.finalize(
        margin=float(get_nested_config(config, "value_planning", "episode_ranking_margin", default=1.0)),
        calibration_weight=calibration_weight,
        ranking_weight=ranking_weight,
        require_ranking_categories=ranking_weight > 0.0,
    )
    if rank == 0:
        logger.info(
            "Value validation epoch %d: composite=%.6f calibration=%.6f ranking=%.6f "
            "ranking_acc=%.4f margin_acc=%.4f hazard=%d comparator=%d",
            epoch,
            metrics["value_validation_loss"],
            metrics["value_calibration_loss"],
            metrics["value_ranking_loss"],
            metrics["value_ranking_accuracy"],
            metrics["value_margin_accuracy"],
            int(metrics["value_hazard_count"]),
            int(metrics["value_comparator_count"]),
        )
    return metrics
