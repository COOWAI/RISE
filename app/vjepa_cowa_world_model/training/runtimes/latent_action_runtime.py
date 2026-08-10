"""Utilities for DreamZero-style latent/action DiT training."""

from __future__ import annotations

from typing import Sequence

import torch


def _scale_tensor(scale: Sequence[float], reference: torch.Tensor, action_dim: int) -> torch.Tensor:
    if len(scale) != int(action_dim):
        raise ValueError(f"joint_action_scale length {len(scale)} must match action_dim={action_dim}")
    values = torch.as_tensor(scale, device=reference.device, dtype=reference.dtype)
    if bool((values <= 0).any().item()):
        raise ValueError("joint_action_scale entries must be positive")
    return values.view(1, 1, int(action_dim))


def normalize_joint_actions(actions: torch.Tensor, scale: Sequence[float]) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, H, A], got {tuple(actions.shape)}")
    return actions / _scale_tensor(scale, actions, int(actions.shape[-1]))


def denormalize_joint_actions(actions: torch.Tensor, scale: Sequence[float]) -> torch.Tensor:
    if actions.ndim != 3:
        raise ValueError(f"actions must have shape [B, H, A], got {tuple(actions.shape)}")
    return actions * _scale_tensor(scale, actions, int(actions.shape[-1]))


def actions_to_relative_trajectory(actions: torch.Tensor) -> torch.Tensor:
    """Integrate ego-frame [dx, dy, dyaw] action deltas into [B, H, 3] trajectories."""
    if actions.ndim != 3 or actions.shape[-1] != 3:
        raise ValueError(f"actions must have shape [B, H, 3], got {tuple(actions.shape)}")
    deltas = actions.float()
    x = deltas.new_zeros(deltas.shape[0])
    y = deltas.new_zeros(deltas.shape[0])
    yaw = deltas.new_zeros(deltas.shape[0])
    poses = []
    for step in range(deltas.shape[1]):
        dx = deltas[:, step, 0]
        dy = deltas[:, step, 1]
        dyaw = deltas[:, step, 2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        x = x + cos_yaw * dx - sin_yaw * dy
        y = y + sin_yaw * dx + cos_yaw * dy
        yaw = torch.atan2(torch.sin(yaw + dyaw), torch.cos(yaw + dyaw))
        poses.append(torch.stack([x, y, yaw], dim=-1))
    if not poses:
        return deltas.new_zeros(deltas.shape[0], 0, 3)
    return torch.stack(poses, dim=1)


def _interpolate_chunk_endpoints(chunk_endpoints: torch.Tensor, frame_stride: int) -> torch.Tensor:
    if frame_stride == 1:
        return chunk_endpoints
    previous = chunk_endpoints.new_zeros(chunk_endpoints.shape[0], 3)
    poses = []
    for chunk_idx in range(chunk_endpoints.shape[1]):
        endpoint = chunk_endpoints[:, chunk_idx]
        yaw_delta = torch.atan2(
            torch.sin(endpoint[:, 2] - previous[:, 2]),
            torch.cos(endpoint[:, 2] - previous[:, 2]),
        )
        for substep in range(1, frame_stride + 1):
            alpha = float(substep) / float(frame_stride)
            xy = previous[:, :2] + alpha * (endpoint[:, :2] - previous[:, :2])
            yaw = previous[:, 2] + alpha * yaw_delta
            yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
            poses.append(torch.cat([xy, yaw[:, None]], dim=-1))
        previous = endpoint
    if not poses:
        return chunk_endpoints.new_zeros(chunk_endpoints.shape[0], 0, 3)
    return torch.stack(poses, dim=1)


def build_joint_action_policy_output(
    actions: torch.Tensor,
    num_poses: int,
    *,
    frame_stride: int = 1,
) -> dict[str, torch.Tensor]:
    """Convert denormalized joint actions into the planner inference output contract."""
    num_poses = int(num_poses)
    if num_poses <= 0:
        raise ValueError(f"num_poses must be positive, got {num_poses}")
    frame_stride = int(frame_stride)
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")
    chunk_endpoints = actions_to_relative_trajectory(actions)
    trajectory = _interpolate_chunk_endpoints(chunk_endpoints, frame_stride)
    if trajectory.shape[1] < num_poses:
        raise ValueError(
            f"joint action policy action horizon {trajectory.shape[1]} is shorter than num_poses={num_poses}"
        )
    trajectory = trajectory[:, :num_poses].unsqueeze(1)
    confidences = trajectory.new_zeros(trajectory.shape[0], 1)
    return {"trajectories": trajectory, "confidences": confidences}


def build_future_action_targets(
    *,
    actions: torch.Tensor | None,
    num_observed_steps: int,
    num_future_steps: int,
    action_dim: int,
    future_start_step: int = 0,
) -> torch.Tensor:
    if actions is None:
        raise ValueError("joint action latent-DiT requires actions, got None")
    if actions.ndim != 3 or actions.shape[-1] != int(action_dim):
        raise ValueError(f"actions must have shape [B, T-1, {action_dim}], got {tuple(actions.shape)}")
    start = int(num_observed_steps) - 1 + int(future_start_step)
    end = start + int(num_future_steps)
    if start < 0 or end > actions.shape[1]:
        raise ValueError(
            "joint action target requires actions covering transitions "
            f"[{start}:{end}], got actions.shape={tuple(actions.shape)}"
        )
    return actions[:, start:end]


def build_last_observed_action_state_tokens(
    *,
    states: torch.Tensor | None,
    num_observed_steps: int,
    num_future_steps: int,
    state_dim: int,
) -> torch.Tensor:
    if states is None:
        raise ValueError("joint action latent-DiT requires states for state register tokens, got None")
    if states.ndim != 3 or states.shape[-1] < int(state_dim):
        raise ValueError(f"states must have shape [B, T, >= {state_dim}], got {tuple(states.shape)}")
    last_observed_idx = int(num_observed_steps) - 1
    if last_observed_idx < 0 or last_observed_idx >= states.shape[1]:
        raise ValueError(
            f"num_observed_steps={num_observed_steps} is incompatible with states.shape={tuple(states.shape)}"
        )
    last_state = states[:, last_observed_idx, : int(state_dim)]
    return last_state[:, None].expand(-1, int(num_future_steps), -1).contiguous()
