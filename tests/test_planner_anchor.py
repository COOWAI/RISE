"""Shared diffusion anchor contract for training, Oracle, and deployment."""

from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.training.planner_anchor import build_ego_relative_diffusion_anchor


def _planner(*, enabled: bool, traj_dim: int) -> SimpleNamespace:
    return SimpleNamespace(use_anchor_frame=enabled, traj_dim=traj_dim)


def test_disabled_anchor_returns_none_without_fabricating_state() -> None:
    reference = torch.zeros(2, 4, 7)

    anchor = build_ego_relative_diffusion_anchor(
        _planner(enabled=False, traj_dim=6),
        ego_dynamics=None,
        observed_frames=4,
        reference=reference,
    )

    assert anchor is None


def test_four_dimensional_anchor_is_current_ego_origin() -> None:
    reference = torch.zeros(2, 4, 7, dtype=torch.bfloat16)

    anchor = build_ego_relative_diffusion_anchor(
        _planner(enabled=True, traj_dim=4),
        ego_dynamics=None,
        observed_frames=4,
        reference=reference,
    )

    assert anchor.dtype == torch.float32
    assert torch.equal(anchor, torch.tensor([[0.0, 0.0, 1.0, 0.0]]).expand(2, -1))


def test_six_dimensional_anchor_uses_last_observed_ego_velocity() -> None:
    dynamics = torch.zeros(2, 6, 4)
    dynamics[0, 3, :2] = torch.tensor([3.0, -1.0])
    dynamics[1, 3, :2] = torch.tensor([4.0, 2.0])

    anchor = build_ego_relative_diffusion_anchor(
        _planner(enabled=True, traj_dim=6),
        ego_dynamics=dynamics,
        observed_frames=4,
        reference=dynamics,
    )

    assert torch.equal(
        anchor,
        torch.tensor(
            [
                [0.0, 0.0, 3.0, -1.0, 1.0, 0.0],
                [0.0, 0.0, 4.0, 2.0, 1.0, 0.0],
            ]
        ),
    )


def test_six_dimensional_anchor_rejects_missing_or_short_dynamics() -> None:
    planner = _planner(enabled=True, traj_dim=6)
    reference = torch.zeros(1, 4, 7)

    with pytest.raises(ValueError, match="requires observed ego_dynamics"):
        build_ego_relative_diffusion_anchor(
            planner,
            ego_dynamics=None,
            observed_frames=4,
            reference=reference,
        )
    with pytest.raises(ValueError, match="cover 4 observed frames"):
        build_ego_relative_diffusion_anchor(
            planner,
            ego_dynamics=torch.zeros(1, 3, 4),
            observed_frames=4,
            reference=reference,
        )


@pytest.mark.parametrize("traj_dim", [0, 5, 7])
def test_anchor_rejects_unsupported_trajectory_dimension(traj_dim: int) -> None:
    with pytest.raises(ValueError, match="traj_dim 4 or 6"):
        build_ego_relative_diffusion_anchor(
            _planner(enabled=True, traj_dim=traj_dim),
            ego_dynamics=torch.zeros(1, 4, 4),
            observed_frames=4,
            reference=torch.zeros(1, 4, 7),
        )
