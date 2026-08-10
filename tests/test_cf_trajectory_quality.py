import pytest
import torch

from app.vjepa_cowa_world_model.training.cf_trajectory_quality import (
    CF_QUALITY_SCHEMA,
    compute_counterfactual_trajectory_quality,
    counterfactual_quality_schema,
)
from app.vjepa_cowa_world_model.training.reward_labels import RewardLabelConfig


def _trajectories():
    trajectories = torch.zeros(1, 2, 4, 3)
    trajectories[0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    trajectories[0, 1, :, 0] = torch.tensor([-1.0, -2.0, -3.0, -4.0])
    return trajectories


def test_cf_quality_uses_deployment_confidence_and_penalizes_reverse():
    trajectories = _trajectories()

    forward = compute_counterfactual_trajectory_quality(
        trajectories,
        torch.tensor([[0.9, 0.1]]),
        dataset_domains=["counterfactual"],
        timestep_sec=1.0,
    )
    reverse = compute_counterfactual_trajectory_quality(
        trajectories,
        torch.tensor([[0.1, 0.9]]),
        dataset_domains=["counterfactual"],
        timestep_sec=1.0,
    )

    assert CF_QUALITY_SCHEMA == "cf_progress_reverse_comfort_efficiency_v1"
    assert forward.selected_mode.tolist() == [0]
    assert reverse.selected_mode.tolist() == [1]
    assert forward.progress_m.item() == pytest.approx(4.0)
    assert forward.reverse_risk.item() == 0.0
    assert reverse.reverse_risk.item() == pytest.approx(1.0)
    assert forward.quality_score.item() > reverse.quality_score.item()


def test_cf_quality_reports_lower_path_efficiency_for_detour():
    trajectories = torch.zeros(1, 2, 3, 3)
    trajectories[0, 0, :, 0] = torch.tensor([1.0 / 3.0, 2.0 / 3.0, 1.0])
    trajectories[0, 1, :, 0] = torch.tensor([1.0, 0.0, 1.0])

    direct = compute_counterfactual_trajectory_quality(
        trajectories,
        torch.tensor([[1.0, 0.0]]),
        dataset_domains=["counterfactual"],
    )
    detour = compute_counterfactual_trajectory_quality(
        trajectories,
        torch.tensor([[0.0, 1.0]]),
        dataset_domains=["counterfactual"],
    )

    assert direct.path_efficiency.item() == pytest.approx(1.0)
    assert detour.path_efficiency.item() == pytest.approx(1.0 / 3.0)


def test_cf_quality_is_counterfactual_only_and_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="counterfactual-only"):
        compute_counterfactual_trajectory_quality(
            _trajectories(),
            torch.tensor([[1.0, 0.0]]),
            dataset_domains=["real"],
        )

    trajectories = _trajectories()
    trajectories[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_counterfactual_trajectory_quality(
            trajectories,
            torch.tensor([[1.0, 0.0]]),
            dataset_domains=["counterfactual"],
        )


def test_cf_quality_schema_fully_records_geometry_free_metric_definition():
    schema = counterfactual_quality_schema(timestep_sec=0.5, max_progress_m=20.0)
    comfort_config = RewardLabelConfig(timestep_sec=0.5)

    assert schema["name"] == CF_QUALITY_SCHEMA
    assert schema["inputs"] == ["trajectory", "deployment_confidence"]
    assert schema["forbidden_inputs"] == ["agent_boxes", "collision", "ttc", "offroad"]
    assert schema["comfort"] == {
        "accel_threshold": comfort_config.accel_threshold,
        "accel_margin": comfort_config.accel_margin,
        "yaw_rate_threshold": comfort_config.yaw_rate_threshold,
        "yaw_rate_margin": comfort_config.yaw_rate_margin,
        "jerk_threshold": comfort_config.jerk_threshold,
        "jerk_margin": comfort_config.jerk_margin,
        "risk_boundary": "clamp((abs(value)-threshold)/margin,0,1)",
        "aggregation": "max_over_steps_and_accel_yaw_rate_jerk",
    }
    assert sum(schema["weights"].values()) == pytest.approx(1.0)
