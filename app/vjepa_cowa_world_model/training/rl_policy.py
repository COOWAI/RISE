from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.encoder_inputs import build_tubelet_encoder_input
from app.vjepa_cowa_world_model.training.predictor_stepping import make_predictor_step_fn
from app.vjepa_cowa_world_model.utils import confidence_weighted_trajectory, trajectory_to_control_action
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module as _unwrap_module


def frame_history_to_context_clip(frame_history_tensor: torch.Tensor) -> torch.Tensor:
    """Convert `[T, C, H, W]` or `[B, T, C, H, W]` to `[B, C, T, H, W]`."""
    if frame_history_tensor.ndim == 4:
        return frame_history_tensor.permute(1, 0, 2, 3).unsqueeze(0)
    if frame_history_tensor.ndim == 5:
        return frame_history_tensor.permute(0, 2, 1, 3, 4)
    raise ValueError(
        f"Expected frame_history_tensor shape [T, C, H, W] or [B, T, C, H, W], got {tuple(frame_history_tensor.shape)}"
    )


def _repeat_step_frames(step_frames: torch.Tensor, tubelet_size: int) -> torch.Tensor:
    return build_tubelet_encoder_input(step_frames, tubelet_size)


def encode_context_clip(
    encoder,
    context_clip: torch.Tensor,
    *,
    tubelet_size: int,
    normalize_reps: bool,
    frame_stride: int = 1,
) -> torch.Tensor:
    """Encode clip frames with per-step tubelet repetition.

    `frame_stride` lets offline clips subsample raw video frames into predictor/planner
    timesteps while online RL clips can keep every simulator frame as one timestep.
    """
    encoder_module = _unwrap_module(encoder)
    if frame_stride > 1:
        context_clip = context_clip[:, :, ::frame_stride, :, :]

    batch_size, _, num_frames, _, _ = context_clip.shape
    with torch.no_grad():
        encoder_input = _repeat_step_frames(context_clip, tubelet_size)
        z_context = encoder_module([encoder_input])[0]
    z_context = z_context.view(batch_size, num_frames, -1, z_context.size(-1)).flatten(1, 2)

    if normalize_reps:
        z_context = F.layer_norm(z_context, (z_context.size(-1),))
    return z_context


def pad_temporal_sequence(sequence: torch.Tensor, target_len: int) -> torch.Tensor:
    if sequence.ndim < 2:
        raise ValueError(f"Expected sequence with temporal dimension, got shape {tuple(sequence.shape)}")
    current_len = sequence.shape[1]
    if current_len >= target_len:
        return sequence[:, :target_len]
    if current_len == 0:
        raise ValueError("Cannot pad an empty temporal sequence")
    pad_count = target_len - current_len
    pad_frame = sequence[:, -1:, ...].expand(-1, pad_count, *sequence.shape[2:])
    return torch.cat([sequence, pad_frame], dim=1)


def states_to_action_history(states: torch.Tensor) -> torch.Tensor:
    """Approximate dataset-style 7D actions from consecutive 7D states."""
    if states.ndim != 3 or states.shape[-1] != 7:
        raise ValueError(f"Expected states shape [B, T, 7], got {tuple(states.shape)}")
    if states.shape[1] < 2:
        return torch.zeros(states.shape[0], 0, states.shape[-1], device=states.device, dtype=states.dtype)

    xyz_diff = states[:, 1:, :3] - states[:, :-1, :3]
    rot_diff = states[:, 1:, 3:6] - states[:, :-1, 3:6]
    rot_diff = torch.atan2(torch.sin(rot_diff), torch.cos(rot_diff))
    velocity = states[:, :-1, 6:7]
    return torch.cat([xyz_diff, rot_diff, velocity], dim=-1)


def build_predictor_rollout_tensors(
    observed_states: torch.Tensor,
    *,
    num_total_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad observed states to rollout horizon and derive masked-action inputs."""
    if observed_states.ndim == 2:
        observed_states = observed_states.unsqueeze(0)
    states_full = pad_temporal_sequence(observed_states, num_total_frames)
    actions_full = states_to_action_history(states_full)
    return states_full, actions_full


def predictor_ar_rollout(
    predictor,
    z_context: torch.Tensor,
    *,
    actions: torch.Tensor,
    states: torch.Tensor,
    tokens_per_frame: int,
    num_observed_frames: int,
    num_total_frames: int,
    normalize_reps: bool,
    extrinsics: Optional[torch.Tensor] = None,
    predictor_inference_consistent: bool = True,
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
    state_dim: int = 7,
    use_drive_command: bool = True,
) -> torch.Tensor:
    """Run the frozen action-conditioned predictor autoregressively."""
    predictor_module = _unwrap_module(predictor)
    expected_state_dim = getattr(getattr(predictor_module, "state_encoder", None), "in_features", None)
    num_obs = int(num_observed_frames)
    num_total = int(num_total_frames)
    if num_obs < 1:
        raise ValueError(f"num_observed_frames must be >= 1, got {num_obs}")
    if num_total < num_obs:
        raise ValueError(f"num_total_frames ({num_total}) must be >= num_observed_frames ({num_obs})")

    actions = actions[:, : max(0, num_total - 1)]
    states = states[:, :num_total]
    if extrinsics is not None:
        extrinsics = extrinsics[:, :num_total]
    if driving_command is not None and driving_command.ndim >= 3:
        driving_command = driving_command[:, :num_total]
    if ego_dynamics is not None and ego_dynamics.ndim >= 3:
        ego_dynamics = ego_dynamics[:, :num_total]

    aux_config = SimpleNamespace(
        train=SimpleNamespace(
            predictor_aux_policy="inference_consistent" if predictor_inference_consistent else "full",
            predictor_inference_consistent=bool(predictor_inference_consistent),
            predictor_no_aux_input=False,
            use_states_for_predictor=True,
            state_dim=int(state_dim),
            use_drive_command=bool(use_drive_command),
        )
    )

    _step_predictor = make_predictor_step_fn(
        predictor_module,
        aux_config,
        num_obs,
        driving_command=driving_command,
        ego_dynamics=ego_dynamics,
        normalize_reps=normalize_reps,
        no_grad=True,
        expected_state_dim=expected_state_dim,
        state_dim=state_dim,
        use_drive_command=use_drive_command,
    )

    z_rollout = z_context[:, : num_obs * tokens_per_frame]
    for step_idx in range(num_obs, num_total):
        if step_idx == num_total - 1:
            action_seq = actions
            state_seq = states[:, :-1]
            extrinsic_seq = extrinsics[:, :-1] if extrinsics is not None else None
        else:
            action_seq = actions[:, :step_idx]
            state_seq = states[:, :step_idx]
            extrinsic_seq = extrinsics[:, :step_idx] if extrinsics is not None else None

        z_next = _step_predictor(
            z_rollout,
            action_seq,
            state_seq,
            extrinsic_seq,
        )[:, -tokens_per_frame:]
        z_rollout = torch.cat([z_rollout, z_next], dim=1)

    if predictor_inference_consistent:
        return z_rollout[:, num_obs * tokens_per_frame :]
    return z_rollout[:, tokens_per_frame:]


def planner_forward_from_z(
    planner,
    z_tokens: torch.Tensor,
    status_feature: torch.Tensor,
    *,
    use_z_context: bool,
):
    planner_module = _unwrap_module(planner)
    return planner_module(
        z_ar=z_tokens if not use_z_context else None,
        status_feature=status_feature,
        z_context=z_tokens if use_z_context else None,
    )


def compute_baseline_control(
    planner_out: dict,
    *,
    current_speed: torch.Tensor,
    current_steer: torch.Tensor,
    dt: float,
    wheel_base: float,
    action_low: torch.Tensor = None,
    action_high: torch.Tensor = None,
) -> torch.Tensor:
    pred_traj = confidence_weighted_trajectory(
        planner_out["trajectories"],
        planner_out["confidences"],
    )
    return trajectory_to_control_action(
        pred_traj,
        current_speed=current_speed,
        current_steer=current_steer,
        dt=dt,
        wheel_base=wheel_base,
        action_low=action_low,
        action_high=action_high,
    )


def compute_policy_mean(planner_out: dict, baseline_control: torch.Tensor) -> torch.Tensor:
    if "policy_mean" not in planner_out:
        raise KeyError("planner_out is missing policy_mean; enable planner actor-critic outputs first")
    return baseline_control + planner_out["policy_mean"]
