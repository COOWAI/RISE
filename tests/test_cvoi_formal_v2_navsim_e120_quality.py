"""NavSim-e120 direct quality targets are independent from official scoring."""

import inspect
import random

import numpy as np
import pytest
import torch

from app.vjepa_cowa_world_model.training import cvoi_formal_v2_navsim_e120_quality as quality_module
from app.vjepa_cowa_world_model.training.cvoi_execution import common_random_numbers
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_quality import (
    NavSimE120QualitySample,
    collect_navsim_e120_local_quality_targets_direct,
    collect_navsim_e120_stop_quality_target_direct,
    navsim_e120_quality_schema,
    score_navsim_e120_trajectory_quality,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import CvoiPlannerEvaluation
from app.vjepa_cowa_world_model.training.navsim_cvoi_batch import adapt_navsim_e120_quality_field_batch


def _planner_evaluation(*, horizon: int, apply_guidance: bool) -> CvoiPlannerEvaluation:
    trajectory = torch.zeros(1, 1, 4, 3)
    trajectory[0, 0, :, 0] = torch.arange(1, 5, dtype=torch.float32) * float(horizon + 1)
    return CvoiPlannerEvaluation(
        pred_trajs=trajectory,
        confidences=torch.ones(1, 1),
        latency_ms=1.0 + horizon,
        guidance_steps=2 if apply_guidance else 0,
    )


def test_quality_schema_declares_internal_route_free_semantics() -> None:
    schema = navsim_e120_quality_schema(timestep_sec=0.5, max_progress_m=20.0)

    assert schema["name"] == "navsim_e120_route_free_trajectory_quality_v1"
    assert schema["scope"] == "internal_training_target"
    assert schema["official_evaluation"] is False
    assert schema["inputs"] == ["planner_trajectory", "deployment_confidence"]
    assert schema["forbidden_inputs"] == ["ground_truth_trajectory", "agent_geometry", "map", "route"]


def test_quality_module_retains_only_direct_collection_surface() -> None:
    removed_names = (
        "CVOI_NAVSIM_E120_QUALITY_ORACLE_PROTOCOL",
        "NavSimE120QualityOracleJob",
        "NavSimE120QualityOracleRecord",
        "NavSimE120QualityOracleDataset",
        "NavSimE120LocalQualityTargets",
        "NavSimE120StopQualityTargets",
        "collect_navsim_e120_local_quality_targets",
        "collect_navsim_e120_stop_quality_target",
        "collect_navsim_e120_oracle_curve",
        "load_navsim_e120_quality_oracle",
        "write_navsim_e120_quality_oracle",
    )

    assert [name for name in removed_names if hasattr(quality_module, name)] == []
    assert "jsonl" not in inspect.getsource(quality_module).lower()


def test_common_random_numbers_is_execution_owned_and_preserves_rng_state() -> None:
    assert quality_module.common_random_numbers is common_random_numbers
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected_after = (random.random(), float(np.random.rand()), float(torch.rand(())))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    with common_random_numbers(19):
        first = (random.random(), float(np.random.rand()), float(torch.rand(())))
    actual_after = (random.random(), float(np.random.rand()), float(torch.rand(())))
    with common_random_numbers(19):
        second = (random.random(), float(np.random.rand()), float(torch.rand(())))

    assert first == second
    assert actual_after == expected_after
    with pytest.raises(ValueError, match="non-negative integer"):
        with common_random_numbers(-1):
            pass


def test_direct_local_and_stop_quality_collectors_are_audit_free() -> None:
    sample = NavSimE120QualitySample(sample_id="real:a", source_scene_id="scene-a", seed=17)

    def evaluate_prefix(prefix: torch.Tensor, horizon: int, seed: int) -> CvoiPlannerEvaluation:
        del prefix, seed
        return _planner_evaluation(horizon=horizon - 1, apply_guidance=False)

    def evaluate_horizon(horizon: int, apply_guidance: bool, seed: int) -> CvoiPlannerEvaluation:
        del seed
        return _planner_evaluation(horizon=horizon, apply_guidance=apply_guidance)

    local = collect_navsim_e120_local_quality_targets_direct(
        sample,
        base_future_latent=torch.zeros(1, 4, 2, 3),
        num_perturbations=4,
        perturbation_scale=0.05,
        max_delta_norm=0.25,
        evaluate_prefix=evaluate_prefix,
        timestep_sec=0.5,
        max_progress_m=20.0,
        calibration_mode="local_geometry",
    )
    stop = collect_navsim_e120_stop_quality_target_direct(
        sample,
        max_horizon=4,
        evaluate_horizon=evaluate_horizon,
        timestep_sec=0.5,
        max_progress_m=20.0,
        controller_lineage="value_guided",
    )

    assert not hasattr(local, "audit_signature")
    assert local.candidate_latents.shape == (5, 4, 2, 3)
    assert local.quality_targets.shape == (5, 4)
    assert not hasattr(stop, "audit_signature")
    assert stop.quality_targets.shape == (1, 5)
    assert stop.latency_ms.shape == (5,)


def test_direct_stop_quality_rejects_p0_before_callback() -> None:
    sample = NavSimE120QualitySample(sample_id="real:a", source_scene_id="scene-a", seed=17)
    calls = 0

    def poison(horizon: int, apply_guidance: bool, seed: int) -> CvoiPlannerEvaluation:
        nonlocal calls
        del horizon, apply_guidance, seed
        calls += 1
        raise AssertionError("direct P0 Stop must fail before planner evaluation")

    with pytest.raises(ValueError, match="value_guided"):
        collect_navsim_e120_stop_quality_target_direct(
            sample,
            max_horizon=4,
            evaluate_horizon=poison,
            timestep_sec=0.5,
            max_progress_m=20.0,
            controller_lineage="p0_controller",
        )
    assert calls == 0


def test_quality_uses_only_confidence_selected_planner_trajectory() -> None:
    trajectories = torch.zeros(1, 2, 4, 3)
    trajectories[0, 0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    trajectories[0, 1, :, 0] = torch.tensor([-1.0, -2.0, -3.0, -4.0])

    forward = score_navsim_e120_trajectory_quality(
        trajectories,
        torch.tensor([[0.9, 0.1]]),
        timestep_sec=0.5,
        max_progress_m=20.0,
    )
    reverse = score_navsim_e120_trajectory_quality(
        trajectories,
        torch.tensor([[0.1, 0.9]]),
        timestep_sec=0.5,
        max_progress_m=20.0,
    )

    assert forward.quality.item() > reverse.quality.item()
    assert forward.selected_mode.tolist() == [0]
    assert reverse.selected_mode.tolist() == [1]
    assert 0.0 <= reverse.quality.item() <= forward.quality.item() <= 1.0


def test_navsim_e120_field_batch_rejects_non_string_sample_identity() -> None:
    metadata = {
        "dataset_domain": ["real"],
        "stable_sample_id": [17],
        "base_scene_id": ["scene-a"],
    }
    navsim_batch = (*([object()] * 11), metadata)

    with pytest.raises(ValueError, match="stable_sample_id.*strings"):
        adapt_navsim_e120_quality_field_batch(
            navsim_batch,
            z_observed=torch.randn(1, 1, 1, 4),
            z_future=torch.randn(1, 2, 1, 4),
            real_quality_target_provider=lambda _request: torch.ones(1, 2),
            cf_field_supervision="none",
        )
