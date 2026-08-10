"""Real-object integration tests for interleaved diffusion inference."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner
from app.vjepa_cowa_world_model.models.seeded_diffusion_planner import SeededDiffusionPlanner

BATCH_SIZE = 2
BASE = {
    "encoder_dim": 8,
    "num_poses": 3,
    "status_dim": 4,
    "hidden_dim": 16,
    "depth": 1,
    "heads": 4,
    "dropout": 0.0,
    "mlp_ratio": 2.0,
    "traj_dim": 4,
    "num_samples": 2,
    "inference_steps": 3,
    "tokens_per_frame": 2,
    "use_last_frame_only": False,
    "observed_token_mode": "none",
}


def _build_planner(planner_cls: type[DiffusionPlanner], overrides: dict[str, Any]) -> DiffusionPlanner:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3201)
        return planner_cls(**{**BASE, **overrides}).eval()


def _interleaved_inputs(planner: DiffusionPlanner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_modes = planner.num_modes if planner.num_modes > 1 else planner.num_samples
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3202)
        status = torch.randn(BATCH_SIZE, 4)
        z_ar = torch.randn(
            BATCH_SIZE,
            planner.num_poses * planner.tokens_per_frame,
            planner.encoder_dim,
        )
        noise = torch.randn(
            BATCH_SIZE,
            output_modes,
            planner.num_poses,
            planner.traj_dim,
        )
    return status, z_ar, noise


@pytest.mark.parametrize(
    ("case_name", "planner_cls", "overrides"),
    (
        ("base_single", DiffusionPlanner, {}),
        (
            "base_joint",
            DiffusionPlanner,
            {"trajectory_token_mode": "per_pose_token", "adaln_version": "v2", "num_modes": 2},
        ),
        (
            "base_independent",
            DiffusionPlanner,
            {
                "trajectory_token_mode": "per_pose_token",
                "adaln_version": "v2",
                "num_modes": 2,
                "independent_modes": True,
            },
        ),
        (
            "seeded_joint",
            SeededDiffusionPlanner,
            {
                "trajectory_token_mode": "per_pose_token",
                "adaln_version": "v2",
                "num_modes": 2,
                "independent_modes": False,
                "init_traj_strategy": "gaussian",
            },
        ),
    ),
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_interleaved_public_api_advances_short_then_full_context_and_finalizes(
    case_name: str,
    planner_cls: type[DiffusionPlanner],
    overrides: dict[str, Any],
) -> None:
    del case_name
    planner = _build_planner(planner_cls, overrides)
    status, z_ar, noise = _interleaved_inputs(planner)

    with torch.no_grad():
        state = planner.init_interleaved_inference_state(
            status_feature=status,
            total_condition_updates=2,
            inference_noise=noise,
        )
        state = planner.advance_interleaved_inference(state, z_ar[:, :2])
        state = planner.advance_interleaved_inference(state, z_ar)
        output = planner.finalize_interleaved_inference(state, z_ar)

    expected_k = planner.num_modes if planner.num_modes > 1 else planner.num_samples
    assert state["completed_condition_updates"] == 2
    assert state["completed_sampling_steps"] == planner.inference_steps
    assert output["trajectories"].shape == (BATCH_SIZE, expected_k, planner.num_poses, 3)
    assert output["confidences"].shape == (BATCH_SIZE, expected_k)
    assert torch.isfinite(output["trajectories"]).all()
    assert torch.isfinite(output["confidences"]).all()
    assert torch.all(output["confidences"] >= 0)
    torch.testing.assert_close(
        output["confidences"].sum(dim=1),
        torch.ones(BATCH_SIZE),
        rtol=1e-6,
        atol=1e-6,
    )


def test_interleaved_finalize_uses_latest_context_for_final_positive_time_data_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _build_planner(DiffusionPlanner, {"use_last_frame_only": True})
    status, z_ar, _ = _interleaved_inputs(planner)
    output_modes = planner.num_samples
    inference_noise = torch.full(
        (BATCH_SIZE, output_modes, planner.num_poses, planner.traj_dim),
        2.0,
    )
    with torch.no_grad():
        stale_context = planner._prepare_context(z_ar[:, :2], None, None, None)
        latest_context = planner._prepare_context(z_ar, None, None, None)
    stale_context = stale_context.repeat_interleave(output_modes, dim=0)
    latest_context = latest_context.repeat_interleave(output_modes, dim=0)
    assert not torch.allclose(stale_context, latest_context)

    clean_data_value = -0.75
    solver_inputs: list[torch.Tensor] = []
    final_predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def identity_solver_step(
        x_t: torch.Tensor,
        context_k: torch.Tensor,
        status_k: torch.Tensor,
        noise_schedule: object,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del context_k, status_k, noise_schedule, s, t
        solver_inputs.append(x_t.detach().clone())
        return x_t

    def final_data_prediction(
        x_t: torch.Tensor,
        t: torch.Tensor,
        cross_c: torch.Tensor,
        status_emb: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        del status_emb
        final_predictions.append((x_t.detach().clone(), t.detach().clone(), cross_c.detach().clone()))
        return None, torch.full_like(x_t, clean_data_value)

    monkeypatch.setattr(planner, "_run_interleaved_solver_step", identity_solver_step)
    monkeypatch.setattr(planner.dit, "forward", final_data_prediction)

    with torch.no_grad():
        state = planner.init_interleaved_inference_state(
            status_feature=status,
            total_condition_updates=2,
            inference_noise=inference_noise,
        )
        state = planner.advance_interleaved_inference(state, z_ar[:, :2])
        state = planner.advance_interleaved_inference(state, z_ar)
        output = planner.finalize_interleaved_inference(state, z_ar)

    assert solver_inputs
    assert len(final_predictions) == 1
    final_x_t, final_t, final_context = final_predictions[0]
    torch.testing.assert_close(final_x_t, torch.full_like(final_x_t, 2.0), rtol=0.0, atol=0.0)
    assert torch.isfinite(final_t).all()
    assert torch.all(final_t > 0)
    torch.testing.assert_close(final_context, latest_context, rtol=0.0, atol=0.0)
    assert not torch.allclose(final_context, stale_context)

    expected_nd = torch.full_like(inference_noise, clean_data_value)
    expected_3d = planner._convert_nd_to_3d(expected_nd)
    positive_time_3d = planner._convert_nd_to_3d(inference_noise)
    torch.testing.assert_close(output["trajectories"], expected_3d, rtol=0.0, atol=0.0)
    assert not torch.allclose(output["trajectories"], positive_time_3d)
