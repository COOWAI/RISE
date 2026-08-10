"""Unit tests for value-planning rewards and EMA-bootstrapped objectives."""

from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.training.value_planning import (
    ValuePlanningLossConfig,
    compute_online_rewards,
    compute_value_head_loss_from_batch,
    compute_value_planning_loss,
    td_lambda_targets,
)


def test_td_lambda_bootstraps_from_target_values_without_terminals() -> None:
    rewards = torch.tensor([[1.0, 1.0]])
    target_values = torch.tensor([[2.0, 4.0]])

    targets = td_lambda_targets(rewards, target_values, gamma=0.5, lambda_return=0.0)

    torch.testing.assert_close(targets, torch.tensor([[3.0, 3.0]]))


def test_srpo_shaping_disabled_keeps_td_targets_unchanged() -> None:
    rewards = torch.ones(1, 3)
    target_values = torch.zeros(1, 3)
    base = td_lambda_targets(rewards, target_values, gamma=0.9, lambda_return=0.8)
    shaped = td_lambda_targets(
        rewards,
        target_values,
        gamma=0.9,
        lambda_return=0.8,
        rho=torch.tensor([[0.1, 0.2, 0.4]]),
        srpo_shaping_weight=0.0,
    )
    torch.testing.assert_close(shaped, base)


def test_srpo_shaping_enabled_changes_td_targets() -> None:
    rewards = torch.ones(1, 3)
    target_values = torch.zeros(1, 3)
    base = td_lambda_targets(rewards, target_values, gamma=0.9, lambda_return=0.8)
    shaped = td_lambda_targets(
        rewards,
        target_values,
        gamma=0.9,
        lambda_return=0.8,
        rho=torch.tensor([[0.1, 0.2, 0.4]]),
        srpo_shaping_weight=0.5,
    )
    assert not torch.allclose(shaped, base)


def test_value_planning_loss_combines_td_floor_and_episode_ranking() -> None:
    config = ValuePlanningLossConfig(
        gamma=0.9,
        lambda_return=0.8,
        td_loss_weight=1.0,
        safe_floor_weight=0.1,
        episode_ranking_weight=0.2,
        episode_ranking_margin=1.0,
    )
    predicted = torch.tensor([[0.1, 0.2], [-2.0, -4.0]], requires_grad=True)
    result = compute_value_planning_loss(
        predicted_values=predicted,
        target_values=torch.zeros_like(predicted),
        rewards=torch.ones_like(predicted),
        eligible_mask=torch.tensor([True, True]),
        hazard_mask=torch.tensor([False, True]),
        comparator_mask=torch.tensor([True, False]),
        config=config,
    )
    assert result["loss"].item() > 0.0
    assert set(("td_loss", "safe_floor_loss", "episode_ranking_loss")) <= set(result)


def test_srpo_weight_requires_rho_tensor() -> None:
    with pytest.raises(ValueError, match="rho"):
        compute_value_planning_loss(
            predicted_values=torch.zeros(1, 2),
            target_values=torch.zeros(1, 2),
            rewards=torch.ones(1, 2),
            eligible_mask=torch.tensor([True]),
            hazard_mask=torch.tensor([False]),
            comparator_mask=torch.tensor([True]),
            config=ValuePlanningLossConfig(srpo_shaping_weight=0.1),
        )


def test_pred_consistency_weight_requires_loss_tensor() -> None:
    with pytest.raises(ValueError, match="pred_consistency_loss"):
        compute_value_planning_loss(
            predicted_values=torch.zeros(1, 2),
            target_values=torch.zeros(1, 2),
            rewards=torch.ones(1, 2),
            eligible_mask=torch.tensor([True]),
            hazard_mask=torch.tensor([False]),
            comparator_mask=torch.tensor([True]),
            config=ValuePlanningLossConfig(pred_consistency_weight=0.1),
        )


def test_compute_online_rewards_uses_progress_and_comfort() -> None:
    trajectories = torch.tensor([[[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 1.0, 0.1]]]])
    rewards = compute_online_rewards(trajectories, progress_weight=1.0, comfort_weight=0.2)
    assert rewards.shape == (1, 1, 3)
    assert rewards[0, 0, 1].item() > 0.0


class _ZeroHead(torch.nn.Module):
    def forward(self, z_future: torch.Tensor, *, tokens_per_frame: int) -> torch.Tensor:
        return z_future.new_zeros(z_future.shape[0], z_future.shape[1] // int(tokens_per_frame))


def test_value_head_loss_aggregates_raw_rewards_to_predictor_stride() -> None:
    value = SimpleNamespace(
        gamma=0.5,
        lambda_return=0.8,
        bootstrap_horizon=2,
        progress_weight=1.0,
        comfort_weight=0.0,
        value_loss_weight=1.0,
        td_loss_weight=1.0,
        safe_floor_weight=0.0,
        episode_ranking_weight=0.0,
        episode_ranking_margin=1.0,
        srpo_shaping_weight=0.0,
        srpo_potential_based=True,
        pred_consistency_weight=0.0,
    )
    trajectory = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    result = compute_value_head_loss_from_batch(
        value_head=_ZeroHead(),
        target_value_head=_ZeroHead(),
        z_future=torch.zeros(1, 2, 4),
        gt_trajectory=trajectory,
        sample_masks=None,
        tokens_per_frame=1,
        config=SimpleNamespace(value_planning=value),
        frame_stride=2,
    )
    torch.testing.assert_close(result["rewards"], torch.tensor([[1.5, 1.5]]))
    assert "terminals" not in result
