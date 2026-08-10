"""Frozen AC-predictor/planner runtime used by online budget-controller training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Tuple

import torch

from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output
from app.vjepa_cowa_world_model.training.budget_control import BudgetProfile
from app.vjepa_cowa_world_model.training.config import resolve_planner_use_observed_tokens
from app.vjepa_cowa_world_model.training.loop import load_clips
from app.vjepa_cowa_world_model.training.predictor_stepping import (
    make_predictor_step_fn,
    predictor_autoregressive_rollout,
    validate_empty_future_planner_conditions,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (
    build_predictor_timeline_inputs,
    forward_main_context,
    resolve_main_timeline,
)
from app.vjepa_cowa_world_model.utils import (
    build_future_gt_trajectory_from_states,
    build_observed_action_trajectory_history,
    prepare_inference_consistent_status_vector,
    resolve_planner_status_dim,
    resolve_planner_use_drive_command,
)
from app.vjepa_cowa_world_model.utils.eval_determinism import deterministic_eval_rng, extract_batch_metadata
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module
from app.vjepa_cowa_world_model.utils.planner_training import (
    _resolve_action_history_dt,
    resolve_validation_timestep_sec,
)
from app.vjepa_cowa_world_model.utils.trajectory import select_best_trajectory
from app.vjepa_cowa_world_model.val_command import compute_oracle_l2_task_score


@dataclass(frozen=True)
class PreparedBudgetedRolloutBatch:
    """One encoded scene with all budget-invariant planner inputs cached."""

    z_context: torch.Tensor
    predictor_inputs: Any
    step_predictor: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    tokens_per_frame: int
    status_feature: torch.Tensor
    z_first_frame: Optional[torch.Tensor]
    z_observed: Optional[torch.Tensor]
    action_history: Optional[torch.Tensor]
    anchor_state: Optional[torch.Tensor]
    gt_trajectory: torch.Tensor
    timestep_sec: float
    planner: Any
    planner_type: str
    z_ar_mode: str
    autocast_dtype: torch.dtype = torch.float32
    mixed_precision: bool = False


def _camera_metadata_to_device(
    metadata: Mapping[str, Any] | None,
    *,
    device: torch.device,
    require_geometry: bool,
) -> dict[str, torch.Tensor]:
    camera_metadata = {}
    if metadata:
        for key in ("camera_intrinsics", "camera2ego"):
            value = metadata.get(key)
            if torch.is_tensor(value):
                camera_metadata[key] = value.to(device=device, dtype=torch.float, non_blocking=True)
    if require_geometry:
        missing = [key for key in ("camera_intrinsics", "camera2ego") if key not in camera_metadata]
        if missing:
            raise ValueError(
                "online_grpo multi-view runtime requires camera metadata tensors "
                f"camera_intrinsics and camera2ego; missing {missing}"
            )
    return camera_metadata


def _build_diffusion_anchor_state(
    *,
    planner: Any,
    ego_dynamics: Optional[torch.Tensor],
    origin_idx: int,
    batch_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not bool(getattr(planner, "use_anchor_frame", False)):
        return None
    traj_dim = int(getattr(planner, "traj_dim", 0))
    if traj_dim == 4:
        return torch.stack(
            [
                torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, device=device),
                torch.ones(batch_size, device=device),
                torch.zeros(batch_size, device=device),
            ],
            dim=-1,
        ).float()
    if traj_dim == 6:
        if ego_dynamics is None:
            raise ValueError("online_grpo 6D diffusion anchor requires ego_dynamics for vx/vy")
        vx = ego_dynamics[:, int(origin_idx), 0].float()
        vy = ego_dynamics[:, int(origin_idx), 1].float()
        return torch.stack(
            [
                torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, device=device),
                vx,
                vy,
                torch.ones(batch_size, device=device),
                torch.zeros(batch_size, device=device),
            ],
            dim=-1,
        ).float()
    raise ValueError(f"online_grpo diffusion planner requires traj_dim 4 or 6, got {traj_dim}")


def prepare_budgeted_rollout_batch(
    sample: Any,
    *,
    encoder: Any,
    predictor: Any,
    planner: Any,
    config: Any,
    device: torch.device,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    token_ae: Any = None,
    multiview_fusion: Any = None,
) -> Tuple[PreparedBudgetedRolloutBatch, torch.Tensor, Mapping[str, Any]]:
    """Encode one real-data batch and cache every budget-invariant main-policy input."""
    encoder = unwrap_module(encoder)
    predictor = unwrap_module(predictor)
    planner = unwrap_module(planner)
    metadata = extract_batch_metadata(sample) or {}
    context_clips, actions, states, extrinsics, _, driving_command, ego_dynamics = load_clips(
        sample,
        device,
        use_segmentation=False,
        dtype=torch.float,
    )
    if int(context_clips.shape[0]) != 1:
        raise ValueError(f"online_grpo requires batch size 1, got {context_clips.shape[0]}")
    metadata_valid_mask = metadata.get("metadata_valid_mask")
    observed_metadata_valid_mask = metadata.get("observed_metadata_valid_mask")
    camera_metadata = _camera_metadata_to_device(
        metadata,
        device=device,
        require_geometry=bool(getattr(config.multiview, "enabled", False)),
    )

    autocast_enabled = bool(config.mixed_precision) and device.type == "cuda"
    with (
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=config.dtype,
            enabled=autocast_enabled,
        ),
    ):
        z_context = forward_main_context(
            encoder,
            context_clips,
            config=config,
            runtime_normalize_reps=bool(runtime_normalize_reps),
            token_ae=token_ae,
            multiview_fusion=multiview_fusion,
            camera_metadata=camera_metadata,
        )
        predictor_inputs = build_predictor_timeline_inputs(
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            config=config,
            encoder=encoder,
            dt=1.0 / float(max(config.data.fps, 1)),
            metadata_valid_mask=metadata_valid_mask,
            observed_metadata_valid_mask=observed_metadata_valid_mask,
        )
        raw_frames = context_clips.shape[3] if context_clips.ndim == 6 else context_clips.shape[2]
        context_timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=raw_frames)
        if int(z_context.shape[1]) != int(context_timeline.num_time_steps) * int(tokens_per_frame):
            raise ValueError(
                "online_grpo encoder token length does not match the resolved timeline: "
                f"tokens={z_context.shape[1]}, steps={context_timeline.num_time_steps}, "
                f"tokens_per_frame={tokens_per_frame}"
            )
        if int(context_timeline.num_time_steps) < int(predictor_inputs.num_observed_steps):
            raise ValueError(
                "online_grpo context does not contain the complete observed prefix: "
                f"context_steps={context_timeline.num_time_steps}, "
                f"observed_steps={predictor_inputs.num_observed_steps}"
            )

        observed_tokens = int(predictor_inputs.num_observed_steps) * int(tokens_per_frame)
        z_observed_for_controller = z_context[:, :observed_tokens]
        if int(z_observed_for_controller.shape[1]) != observed_tokens:
            raise ValueError(
                f"online_grpo observed latent is too short: got {z_observed_for_controller.shape[1]}, "
                f"expected {observed_tokens}"
            )
        pooled_latent = z_observed_for_controller.detach().float().mean(dim=1)
        step_predictor = make_predictor_step_fn(
            predictor,
            config,
            int(predictor_inputs.num_observed_steps),
            driving_command=predictor_inputs.driving_command,
            ego_dynamics=predictor_inputs.ego_dynamics,
            predictor_no_aux_input=bool(config.train.predictor_no_aux_input),
            normalize_reps=bool(runtime_normalize_reps),
            no_grad=True,
        )
        status_feature = prepare_inference_consistent_status_vector(
            states,
            num_observed=int(config.train.num_observed_frames),
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            state_dim=resolve_planner_status_dim(config),
            use_drive_command=resolve_planner_use_drive_command(config),
        )
        z_first_frame = z_context[:, : int(tokens_per_frame)] if bool(config.planner.use_z_context) else None
        z_observed = z_observed_for_controller if resolve_planner_use_observed_tokens(config) else None
        action_history = None
        if bool(config.planner.use_action_history_for_planner):
            action_history = build_observed_action_trajectory_history(
                predictor_inputs.actions,
                num_observed_frames=int(predictor_inputs.num_observed_steps),
                action_history_dim=int(config.planner.action_history_dim),
                dt=_resolve_action_history_dt(config) * max(int(predictor_inputs.frame_stride), 1),
            )
        num_poses = int(config.data.num_target_frames) - int(config.train.num_observed_frames)
        gt_trajectory = build_future_gt_trajectory_from_states(
            states,
            num_observed_frames=int(config.train.num_observed_frames),
            num_poses=num_poses,
        ).float()
        anchor_state = _build_diffusion_anchor_state(
            planner=planner,
            ego_dynamics=ego_dynamics,
            origin_idx=int(config.train.num_observed_frames) - 1,
            batch_size=1,
            device=device,
        )

    prepared = PreparedBudgetedRolloutBatch(
        z_context=z_context,
        predictor_inputs=predictor_inputs,
        step_predictor=step_predictor,
        tokens_per_frame=int(tokens_per_frame),
        status_feature=status_feature,
        z_first_frame=z_first_frame,
        z_observed=z_observed,
        action_history=action_history,
        anchor_state=anchor_state,
        gt_trajectory=gt_trajectory,
        timestep_sec=resolve_validation_timestep_sec(
            fps=float(config.data.fps),
            diff_dt=float(config.planner.diff_dt),
            default=0.5,
        ),
        planner=planner,
        planner_type=str(config.planner.planner_type),
        z_ar_mode=str(config.planner.z_ar_mode),
        autocast_dtype=config.dtype,
        mixed_precision=autocast_enabled,
    )
    return prepared, pooled_latent, metadata


def select_online_planner_task_score(
    planner_output: Any,
    gt_trajectory: torch.Tensor,
    *,
    timestep_sec: float,
) -> torch.Tensor:
    """Score the planner's highest-confidence trajectory with negative World4Drive L2 average."""
    output = validate_planner_output(
        planner_output,
        mode="inference",
        num_poses=int(gt_trajectory.shape[1]),
    )
    selected = select_best_trajectory(output["trajectories"].float(), output["confidences"].float())
    return compute_oracle_l2_task_score(
        selected,
        gt_trajectory.float(),
        timestep_sec=float(timestep_sec),
    )


def score_prepared_budget_profile(
    prepared: PreparedBudgetedRolloutBatch,
    profile: BudgetProfile,
    *,
    seed: int,
) -> torch.Tensor:
    """Run one frozen predictor/planner profile and return its scalar open-loop task score."""
    predictor_inputs = prepared.predictor_inputs
    rollout_end_step = int(predictor_inputs.num_observed_steps) + int(profile.rollout_future_steps)
    if rollout_end_step > int(predictor_inputs.num_time_steps):
        raise ValueError(
            "budget rollout exceeds the prepared timeline: "
            f"end={rollout_end_step}, total={predictor_inputs.num_time_steps}"
        )

    device = prepared.z_context.device
    with (
        deterministic_eval_rng(int(seed), device),
        torch.no_grad(),
        torch.autocast(
            device_type=device.type,
            dtype=prepared.autocast_dtype,
            enabled=prepared.mixed_precision,
        ),
    ):
        z_ar = predictor_autoregressive_rollout(
            prepared.step_predictor,
            prepared.z_context,
            predictor_inputs.actions,
            predictor_inputs.states,
            predictor_inputs.extrinsics,
            num_obs=int(predictor_inputs.num_observed_steps),
            tokens_per_frame=int(prepared.tokens_per_frame),
            num_total=int(predictor_inputs.num_time_steps),
            predictor_inference_consistent=True,
            rollout_end_step=rollout_end_step,
        )
        if prepared.z_ar_mode == "first_step":
            z_ar_planner = z_ar[:, : int(prepared.tokens_per_frame)]
        elif prepared.z_ar_mode == "full":
            z_ar_planner = z_ar
        else:
            raise ValueError(f"planner.z_ar_mode must be 'full' or 'first_step', got {prepared.z_ar_mode!r}")

        validate_empty_future_planner_conditions(
            z_ar_planner,
            z_context=prepared.z_first_frame,
            z_observed=prepared.z_observed,
            action_history=prepared.action_history,
        )
        planner_kwargs = {
            "z_context": prepared.z_first_frame,
            "z_observed": prepared.z_observed,
            "action_history": prepared.action_history,
        }
        if prepared.planner_type == "diffusion":
            planner_kwargs["anchor_state"] = prepared.anchor_state
        planner_output = prepared.planner(
            z_ar_planner,
            prepared.status_feature,
            **planner_kwargs,
        )
        task_score = select_online_planner_task_score(
            planner_output,
            prepared.gt_trajectory,
            timestep_sec=prepared.timestep_sec,
        )
    if task_score.shape != (1,):
        raise ValueError(f"online budget runtime requires batch size 1, got task_score shape {task_score.shape}")
    return task_score[0].detach()
