"""Shared ego-relative anchor construction for diffusion planner policies."""

from __future__ import annotations

from typing import Optional

import torch

CVOI_EGO_RELATIVE_ANCHOR_PROTOCOL = "ego_relative_last_observed_velocity_v1"


def build_ego_relative_diffusion_anchor(
    planner: torch.nn.Module,
    *,
    ego_dynamics: Optional[torch.Tensor],
    observed_frames: int,
    reference: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build the exact current-ego anchor shared by train, Oracle, and eval.

    The planner trajectory frame is centered at the last observed ego pose, so
    position and yaw are ``(0, 0, 0)``. A 6D planner additionally requires the
    measured ``vx, vy`` from that last observed frame; it never fabricates zero
    velocity.
    """

    core = planner.module if hasattr(planner, "module") else planner
    if not bool(getattr(core, "use_anchor_frame", False)):
        return None
    traj_dim = int(getattr(core, "traj_dim", 0))
    if traj_dim not in {4, 6}:
        raise ValueError(f"diffusion planner anchor requires traj_dim 4 or 6, got {traj_dim}")
    if isinstance(observed_frames, bool) or not isinstance(observed_frames, int) or observed_frames < 1:
        raise ValueError(f"observed_frames must be a positive integer, got {observed_frames!r}")
    if (
        not isinstance(reference, torch.Tensor)
        or reference.ndim < 2
        or reference.shape[0] < 1
        or not reference.dtype.is_floating_point
        or not bool(torch.isfinite(reference).all().item())
    ):
        raise ValueError("diffusion planner anchor reference must be a non-empty finite floating tensor")

    batch_size = int(reference.shape[0])
    zeros = torch.zeros(batch_size, device=reference.device, dtype=torch.float32)
    ones = torch.ones(batch_size, device=reference.device, dtype=torch.float32)
    if traj_dim == 4:
        return torch.stack([zeros, zeros, ones, zeros], dim=-1)

    if not isinstance(ego_dynamics, torch.Tensor) or ego_dynamics.ndim != 3 or ego_dynamics.shape[-1] < 2:
        raise ValueError("6D diffusion planner anchor requires observed ego_dynamics [B,T,D>=2]")
    if ego_dynamics.shape[0] != batch_size:
        raise ValueError("ego_dynamics batch must match diffusion anchor reference")
    if ego_dynamics.shape[1] < observed_frames:
        raise ValueError(f"ego_dynamics must cover {observed_frames} observed frames")
    if (
        not ego_dynamics.dtype.is_floating_point
        or ego_dynamics.device != reference.device
        or not bool(torch.isfinite(ego_dynamics[:, :observed_frames, :2]).all().item())
    ):
        raise ValueError("ego_dynamics must be finite floating point on the diffusion anchor reference device")
    velocity = ego_dynamics[:, observed_frames - 1, :2].to(dtype=torch.float32)
    return torch.stack([zeros, zeros, velocity[:, 0], velocity[:, 1], ones, zeros], dim=-1)
