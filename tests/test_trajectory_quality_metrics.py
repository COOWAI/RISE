"""Route-free planner progress and comfort metric contracts."""

import math

import pytest
import torch

from app.vjepa_cowa_world_model.training.trajectory_quality_metrics import (
    compute_trajectory_quality_metrics_per_sample,
)

EXPECTED_KEYS = {
    "longitudinal_progress_m",
    "forward_progress_m",
    "reverse_distance_m",
    "path_length_m",
    "progress_efficiency",
    "accel_mean_mps2",
    "accel_violation_rate",
    "jerk_mean_mps3",
    "jerk_violation_rate",
    "yaw_rate_mean_radps",
    "yaw_rate_violation_rate",
    "comfort_risk",
}


def _metrics(trajectory, *, velocity=(0.0, 0.0), acceleration=(0.0, 0.0), yaw_rate=0.0, dt=1.0, **kwargs):
    batch_size = trajectory.shape[0]
    return compute_trajectory_quality_metrics_per_sample(
        trajectory,
        anchor_velocity=torch.tensor([velocity], dtype=trajectory.dtype).expand(batch_size, -1),
        anchor_acceleration=torch.tensor([acceleration], dtype=trajectory.dtype).expand(batch_size, -1),
        anchor_yaw_rate=torch.full((batch_size,), yaw_rate, dtype=trajectory.dtype),
        timestep_sec=dt,
        **kwargs,
    )


def test_straight_constant_speed_has_unit_efficiency_and_zero_comfort_cost() -> None:
    trajectory = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])
    result = _metrics(trajectory, velocity=(1.0, 0.0))

    assert set(result) == EXPECTED_KEYS
    assert result["longitudinal_progress_m"].item() == pytest.approx(3.0)
    assert result["forward_progress_m"].item() == pytest.approx(3.0)
    assert result["reverse_distance_m"].item() == pytest.approx(0.0)
    assert result["path_length_m"].item() == pytest.approx(3.0)
    assert result["progress_efficiency"].item() == pytest.approx(1.0)
    assert result["accel_mean_mps2"].item() == pytest.approx(0.0)
    assert result["jerk_mean_mps3"].item() == pytest.approx(0.0)
    assert result["comfort_risk"].item() == pytest.approx(0.0)


def test_stationary_and_reverse_progress_are_well_defined() -> None:
    stationary = _metrics(torch.zeros(1, 3, 3))
    assert stationary["progress_efficiency"].item() == 0.0

    reverse = _metrics(torch.tensor([[[-1.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]]))
    assert reverse["longitudinal_progress_m"].item() == pytest.approx(-2.0)
    assert reverse["forward_progress_m"].item() == 0.0
    assert reverse["reverse_distance_m"].item() == pytest.approx(2.0)
    assert reverse["progress_efficiency"].item() == pytest.approx(-1.0)


def test_known_acceleration_and_observed_boundary_jerk_are_included() -> None:
    trajectory = torch.tensor([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]])
    smooth = _metrics(trajectory, velocity=(0.0, 0.0), acceleration=(1.0, 0.0))
    discontinuous = _metrics(trajectory, velocity=(0.0, 0.0), acceleration=(0.0, 0.0))

    assert smooth["accel_mean_mps2"].item() == pytest.approx(1.0)
    assert smooth["jerk_mean_mps3"].item() == pytest.approx(0.0)
    assert discontinuous["jerk_mean_mps3"].item() == pytest.approx(1.0 / 3.0)


def test_constant_turn_and_yaw_wrap_use_wrapped_angle_differences() -> None:
    turn = torch.tensor([[[0.0, 0.0, 0.2], [0.0, 0.0, 0.4], [0.0, 0.0, 0.6]]])
    turn_result = _metrics(turn, yaw_rate=1.0, dt=0.2, yaw_rate_threshold=1.1)
    assert turn_result["yaw_rate_mean_radps"].item() == pytest.approx(1.0)
    assert turn_result["yaw_rate_violation_rate"].item() == 0.0

    wrapped = torch.tensor([[[0.0, 0.0, math.pi - 0.1], [0.0, 0.0, -math.pi + 0.1]]])
    wrapped_result = _metrics(wrapped)
    expected_mean = ((math.pi - 0.1) + 0.2) / 2.0
    assert wrapped_result["yaw_rate_mean_radps"].item() == pytest.approx(expected_mean, abs=1e-5)


def test_quality_metrics_never_claim_unavailable_safety_semantics() -> None:
    result = _metrics(torch.zeros(1, 2, 3))
    forbidden = ("safety", "collision", "ttc", "offroad", "red_light")
    assert not any(fragment in key for key in result for fragment in forbidden)


@pytest.mark.parametrize(
    ("trajectory", "velocity", "match"),
    [
        (torch.zeros(2, 3), torch.zeros(1, 2), "trajectory"),
        (torch.zeros(1, 2, 3), torch.zeros(1, 3), "anchor_velocity"),
    ],
)
def test_quality_metric_shapes_fail_loud(trajectory, velocity, match) -> None:
    with pytest.raises(ValueError, match=match):
        compute_trajectory_quality_metrics_per_sample(
            trajectory,
            anchor_velocity=velocity,
            anchor_acceleration=torch.zeros(1, 2),
            anchor_yaw_rate=torch.zeros(1),
            timestep_sec=0.5,
        )
