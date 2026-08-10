"""Tests for the retained World4Drive real-geometry boundary."""

from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.training import navsim_cvoi_evaluation as evaluation_module
from app.vjepa_cowa_world_model.training.geometry_outcome import PlanningOutcomeEvaluator


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        train=SimpleNamespace(num_observed_frames=4, predictor_inference_consistent=True),
        cvoi=SimpleNamespace(controller_batch_size=1),
    )


def _real_batch() -> tuple[object, ...]:
    metadata = {
        "dataset_domain": ["real"],
        "stable_sample_id": ["navsim:real:sample"],
        "base_scene_id": ["scene-real"],
        "geometry_present": torch.tensor([True]),
        "future_agent_geometry_valid": torch.tensor([True]),
        "agent_geometry_truncated": [False],
        "geometry_source": ["logged_nuscenes_gt"],
        "geometry_coordinate_frame": ["per_frame_ego"],
        "raw_agent_count": [torch.zeros(10, dtype=torch.long)],
    }
    return (
        torch.zeros(1, 3, 10, 8, 8),
        torch.zeros(1, 9, 3),
        torch.zeros(1, 10, 7),
        torch.zeros(1, 10, 7),
        [None],
        torch.zeros(1, 10, 4),
        torch.zeros(1, 10, 4),
        torch.zeros(1, 10, 256, 7),
        torch.zeros(1, 10, 256, dtype=torch.bool),
        torch.zeros(1, 10, 4, 4),
        None,
        metadata,
    )


def test_legacy_agent_adapter_is_removed_but_direct_latency_environment_remains() -> None:
    assert not hasattr(evaluation_module, "NavSimAgentCvoiDeployment")
    environment = evaluation_module.build_cvoi_evaluation_runtime_environment(
        SimpleNamespace(meta=SimpleNamespace(dtype="bfloat16")),
        device="cpu",
        _allow_cpu_for_tests=True,
    )
    assert environment.to_manifest()["hardware"]["test_only"] is True


def test_public_real_outcome_adapter_reuses_canonical_geometry_evaluator() -> None:
    outcome = evaluation_module.evaluate_navsim_real_planning_outcome(
        _real_batch(),
        torch.zeros(1, 1, 6, 3),
        torch.ones(1, 1),
        config=_config(),
        evaluator=PlanningOutcomeEvaluator(timestep_sec=0.5),
    )

    assert outcome.task_score.shape == (1,)
    assert float(outcome.task_score[0]) == pytest.approx(1.0)
