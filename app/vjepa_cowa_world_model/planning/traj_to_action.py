"""Trajectory rollout helpers for staged planner training."""

from typing import Optional

import torch

from app.vjepa_cowa_world_model.utils.status_features import prepare_inference_consistent_states


def _wrap_angle_delta(delta_yaw: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))


def _raw_states_from_traj(traj_xy_yaw: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    if traj_xy_yaw.ndim != 4 or traj_xy_yaw.shape[-1] != 3:
        raise ValueError(f"Expected trajectory shape [B, M, T, 3], got {tuple(traj_xy_yaw.shape)}")
    if traj_xy_yaw.shape[2] < 2:
        raise ValueError("Need at least 2 trajectory poses to derive actions/states")
    if dt <= 0:
        raise ValueError("dt must be > 0")

    delta_xy = traj_xy_yaw[:, :, 1:, :2] - traj_xy_yaw[:, :, :-1, :2]
    delta_yaw = _wrap_angle_delta(traj_xy_yaw[:, :, 1:, 2] - traj_xy_yaw[:, :, :-1, 2]).unsqueeze(-1)
    action_3d = torch.cat([delta_xy / dt, delta_yaw / dt], dim=-1)

    vx = torch.cat([action_3d[..., 0], action_3d[..., -1:, 0]], dim=-1)
    vy = torch.cat([action_3d[..., 1], action_3d[..., -1:, 1]], dim=-1)
    ax = torch.cat([vx[..., 1:] - vx[..., :-1], torch.zeros_like(vx[..., :1])], dim=-1) / dt
    ay = torch.cat([vy[..., 1:] - vy[..., :-1], torch.zeros_like(vy[..., :1])], dim=-1) / dt
    speed = torch.sqrt(torch.clamp(vx.square() + vy.square(), min=0.0))

    states = traj_xy_yaw.new_zeros(*traj_xy_yaw.shape[:3], 7)
    states[..., 0] = traj_xy_yaw[..., 0]
    states[..., 1] = traj_xy_yaw[..., 1]
    states[..., 5] = traj_xy_yaw[..., 2]
    states[..., 6] = speed

    ego_dynamics = torch.stack([vx, vy, ax, ay], dim=-1)
    return states, ego_dynamics


def _expand_driving_command(
    driving_command: Optional[torch.Tensor],
    batch_size: int,
    num_modes: int,
) -> Optional[torch.Tensor]:
    if driving_command is None:
        return None

    if driving_command.ndim == 2:
        expanded = driving_command[:, None, :].expand(batch_size, num_modes, -1)
        return expanded.reshape(batch_size * num_modes, -1)
    if driving_command.ndim == 3:
        expanded = driving_command[:, None, :, :].expand(batch_size, num_modes, -1, -1)
        return expanded.reshape(batch_size * num_modes, driving_command.shape[1], driving_command.shape[2])
    if driving_command.ndim == 4:
        return driving_command.reshape(batch_size * num_modes, driving_command.shape[2], driving_command.shape[3])
    raise ValueError(f"Unsupported driving_command shape: {tuple(driving_command.shape)}")


def traj_to_action(
    traj_xy_yaw: torch.Tensor,
    dt: float,
    action_dim: int,
    driving_command: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert ego-frame trajectory to differentiable finite-difference actions."""

    del driving_command
    raw_states, _ = _raw_states_from_traj(traj_xy_yaw, dt)
    delta_xy = raw_states[:, :, 1:, :2] - raw_states[:, :, :-1, :2]
    delta_yaw = _wrap_angle_delta(raw_states[:, :, 1:, 5] - raw_states[:, :, :-1, 5]).unsqueeze(-1)
    action_3d = torch.cat([delta_xy / dt, delta_yaw / dt], dim=-1)

    if action_dim <= 3:
        return action_3d[..., :action_dim]

    padded = traj_xy_yaw.new_zeros(*action_3d.shape[:3], action_dim)
    padded[..., :3] = action_3d
    return padded


def states_from_traj(
    traj_xy_yaw: torch.Tensor,
    dt: float,
    state_dim: int = 7,
    driving_command: Optional[torch.Tensor] = None,
    inference_consistent: bool = False,
    num_observed: Optional[int] = None,
    use_drive_command: bool = True,
) -> torch.Tensor:
    """Build predictor-ready states from trajectory proposals."""

    raw_states, ego_dynamics = _raw_states_from_traj(traj_xy_yaw, dt)
    batch_size, num_modes, num_steps = raw_states.shape[:3]

    if not inference_consistent:
        if state_dim == raw_states.shape[-1]:
            return raw_states
        if state_dim < raw_states.shape[-1]:
            return raw_states[..., :state_dim]
        padded = raw_states.new_zeros(batch_size, num_modes, num_steps, state_dim)
        padded[..., : raw_states.shape[-1]] = raw_states
        return padded

    observed_steps = min(num_steps, num_observed or 2)
    flat_states = raw_states.reshape(batch_size * num_modes, num_steps, raw_states.shape[-1])
    flat_dynamics = ego_dynamics.reshape(batch_size * num_modes, num_steps, ego_dynamics.shape[-1])
    flat_command = _expand_driving_command(driving_command, batch_size, num_modes)

    ic_states = prepare_inference_consistent_states(
        flat_states,
        num_observed=observed_steps,
        driving_command=flat_command,
        ego_dynamics=flat_dynamics,
        state_dim=state_dim,
        use_drive_command=use_drive_command,
    )
    return ic_states.reshape(batch_size, num_modes, num_steps, ic_states.shape[-1])
