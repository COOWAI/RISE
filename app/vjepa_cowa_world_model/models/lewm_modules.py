"""
le-wm projection modules for JEPA training.

ProjectorMLP: post-encoder projection (hidden_dim → embed_dim)
PredProjectorMLP: post-predictor projection (hidden_dim → embed_dim)

Ported from le-wm MLP class (Maes et al., 2026).
"""

from torch import nn


class ProjectionMLP(nn.Module):
    """Two-layer MLP with BatchNorm1d, used for encoder/predictor output projection.

    Args:
        input_dim: input feature dimension (e.g. encoder hidden_dim = 1408)
        hidden_dim: hidden layer dimension (default 2048)
        output_dim: output embedding dimension (e.g. 192)
    """

    def __init__(self, input_dim, hidden_dim=2048, output_dim=192):
        super().__init__()
        # track_running_stats=False: this projector is a training-only SSL head (not saved in the
        # checkpoint, never run for inference). It is applied multiple times per step (z_tf / z_ar /
        # SIGReg src), and BatchNorm's in-place running_mean/var update bumps the buffer version on
        # each call, which breaks autograd ("variable ... modified by an inplace operation [..2048..]
        # is at version 3; expected version 2"). With track_running_stats=False there is no in-place
        # running-stat update, so the multiple forwards no longer conflict. Train-mode normalization
        # is unchanged (BatchNorm always uses batch statistics in train mode), so this is behaviorally
        # equivalent for training and resolves the pre-existing inplace-autograd RuntimeError.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, track_running_stats=False),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        """x: (N, input_dim) → (N, output_dim)"""
        return self.net(x)


# Alias for clarity — same architecture, independent parameters
ProjectorMLP = ProjectionMLP
PredProjectorMLP = ProjectionMLP
