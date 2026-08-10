"""First-party trajectory regression head for planner query embeddings."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

TRAJECTORY_STATE_SIZE = 3
TRAJECTORY_HEADING_INDEX = 2


class TrajectoryHead(nn.Module):
    """Regress planar poses while bounding heading angles to ``[-pi, pi]``."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int) -> None:
        super().__init__()
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._mlp = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, TRAJECTORY_STATE_SIZE),
        )

    def forward(self, object_queries: torch.Tensor) -> dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, TRAJECTORY_STATE_SIZE)
        poses[..., TRAJECTORY_HEADING_INDEX] = torch.tanh(poses[..., TRAJECTORY_HEADING_INDEX]) * math.pi
        return {"trajectory": poses}
