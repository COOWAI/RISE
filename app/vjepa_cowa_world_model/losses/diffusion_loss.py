# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Diffusion planner loss utilities.

The actual diffusion loss (denoising MSE) is computed inside DiffusionPlanner.forward()
during training. This module provides:
- convert_trajectory_3d_to_6d: Convert GT trajectory from 3-dim to 6-dim format
- compute_diffusion_metrics: Compute additional metrics for logging
"""

import torch


def convert_trajectory_3d_to_nd(
    gt_trajectory: torch.Tensor,
    dt: float = 0.2,
    traj_dim: int = 6,
) -> torch.Tensor:
    """
    Convert ground truth trajectory from 3-dim (x, y, yaw) to N-dim format.

    traj_dim=4: (x, y, cos_yaw, sin_yaw)  — no velocity
    traj_dim=6: (x, y, vx, vy, cos_yaw, sin_yaw)  — with finite-diff velocity

    Args:
        gt_trajectory: [B, T, 3] with (x, y, yaw) in ego-centric coordinates
        dt: time interval between frames (default 0.2s for 5fps)
        traj_dim: output dimension (4 or 6)

    Returns:
        [B, T, traj_dim]
    """
    x = gt_trajectory[..., 0]
    y = gt_trajectory[..., 1]
    yaw = gt_trajectory[..., 2]

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    if traj_dim == 4:
        return torch.stack([x, y, cos_yaw, sin_yaw], dim=-1)

    # 6D: compute velocities via finite differences
    vx = torch.zeros_like(x)
    vy = torch.zeros_like(y)
    vx[:, 1:] = (x[:, 1:] - x[:, :-1]) / dt
    vy[:, 1:] = (y[:, 1:] - y[:, :-1]) / dt
    if x.shape[1] > 0:
        vx[:, 0] = x[:, 0] / dt
        vy[:, 0] = y[:, 0] / dt

    return torch.stack([x, y, vx, vy, cos_yaw, sin_yaw], dim=-1)


# Backward-compatible alias
def convert_trajectory_3d_to_6d(
    gt_trajectory: torch.Tensor,
    dt: float = 0.2,
) -> torch.Tensor:
    return convert_trajectory_3d_to_nd(gt_trajectory, dt=dt, traj_dim=6)


def compute_diffusion_metrics(
    pred_trajectories: torch.Tensor,
    gt_trajectory: torch.Tensor,
) -> dict:
    """
    Compute evaluation metrics for diffusion planner predictions.

    Args:
        pred_trajectories: [B, K, T, 3] predicted trajectories (x, y, yaw)
        gt_trajectory: [B, T, 3] ground truth trajectory (x, y, yaw)

    Returns:
        Dict with:
            - min_ade: minimum Average Displacement Error across K modes
            - min_fde: minimum Final Displacement Error across K modes
    """
    # gt: [B, 1, T, 3] for broadcasting
    gt = gt_trajectory.unsqueeze(1)

    # Position error: [B, K, T]
    pos_error = torch.sqrt(
        (pred_trajectories[..., 0] - gt[..., 0]) ** 2 + (pred_trajectories[..., 1] - gt[..., 1]) ** 2
    )

    # ADE per mode: [B, K]
    ade = pos_error.mean(dim=-1)
    # FDE per mode: [B, K]
    fde = pos_error[..., -1]

    # Min over K modes: [B]
    min_ade = ade.min(dim=-1)[0].mean()
    min_fde = fde.min(dim=-1)[0].mean()

    return {
        "min_ade": min_ade,
        "min_fde": min_fde,
    }
