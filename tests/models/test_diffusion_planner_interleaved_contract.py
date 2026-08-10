"""Numerical finalize contract for base-planner interleaved diffusion inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/diffusion_planner_contract_v1.json"
BATCH_SIZE = 2


def _legacy_profile() -> dict[str, Any]:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    return contract["profiles"]["legacy_single_6d"]


def _materialize_kwargs(serialized_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name, value in serialized_kwargs.items():
        if isinstance(value, dict) and set(value) == {"shape", "dtype"}:
            dtype = getattr(torch, value["dtype"].removeprefix("torch."))
            kwargs[name] = torch.ones(value["shape"], dtype=dtype)
        else:
            kwargs[name] = value
    return kwargs


def test_interleaved_finalize_performs_final_data_prediction_and_returns_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _legacy_profile()
    kwargs = _materialize_kwargs(profile["kwargs"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(809)
        planner = DiffusionPlanner(**kwargs)
    planner.eval()

    z_ar = torch.linspace(
        -1.0,
        1.0,
        BATCH_SIZE * kwargs["num_poses"] * kwargs["tokens_per_frame"] * kwargs["encoder_dim"],
    ).reshape(BATCH_SIZE, kwargs["num_poses"] * kwargs["tokens_per_frame"], kwargs["encoder_dim"])
    status_feature = torch.zeros(BATCH_SIZE, kwargs["status_dim"])
    z_observed = torch.ones(
        BATCH_SIZE,
        profile["attributes"]["num_observed_frames"] * kwargs["tokens_per_frame"],
        kwargs["encoder_dim"],
    )
    output_modes = kwargs["num_samples"]
    inference_noise = torch.full(
        (BATCH_SIZE, output_modes, kwargs["num_poses"], kwargs["traj_dim"]),
        2.0,
    )
    clean_data_value = -0.75
    with torch.no_grad():
        stale_context = planner._prepare_context(
            z_ar[:, : kwargs["tokens_per_frame"]],
            None,
            z_observed,
            None,
        )
        latest_context = planner._prepare_context(z_ar, None, z_observed, None)
    if profile["attributes"]["_uses_batch_expansion"]:
        batch_expansion = profile["attributes"]["_batch_K"]
        stale_context = stale_context.repeat_interleave(batch_expansion, dim=0)
        latest_context = latest_context.repeat_interleave(batch_expansion, dim=0)
    assert not torch.allclose(stale_context, latest_context)
    solver_step_inputs: list[torch.Tensor] = []
    data_prediction_inputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def identity_solver_step(
        x_t: torch.Tensor,
        context_k: torch.Tensor,
        status_k: torch.Tensor,
        noise_schedule: object,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        del context_k, status_k, noise_schedule, s, t
        solver_step_inputs.append(x_t.detach().clone())
        return x_t

    def final_data_prediction_probe(
        x_t: torch.Tensor,
        t: torch.Tensor,
        cross_c: torch.Tensor,
        status_emb: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        del status_emb
        data_prediction_inputs.append((x_t.detach().clone(), t.detach().clone(), cross_c.detach().clone()))
        return None, torch.full_like(x_t, clean_data_value)

    monkeypatch.setattr(planner, "_run_interleaved_solver_step", identity_solver_step)
    monkeypatch.setattr(planner.dit, "forward", final_data_prediction_probe)

    with torch.no_grad():
        state = planner.init_interleaved_inference_state(
            status_feature,
            total_condition_updates=2,
            inference_noise=inference_noise,
        )
        state = planner.advance_interleaved_inference(
            state,
            z_ar[:, : kwargs["tokens_per_frame"]],
            z_observed=z_observed,
        )
        state = planner.advance_interleaved_inference(state, z_ar, z_observed=z_observed)
        output = planner.finalize_interleaved_inference(state, z_ar, z_observed=z_observed)

    assert solver_step_inputs
    assert len(data_prediction_inputs) == 1
    final_x_t, final_t, final_context = data_prediction_inputs[0]
    torch.testing.assert_close(final_x_t, torch.full_like(final_x_t, 2.0), rtol=0.0, atol=0.0)
    assert torch.isfinite(final_t).all()
    assert torch.all(final_t > 0)
    torch.testing.assert_close(final_context, latest_context, rtol=0.0, atol=0.0)
    assert not torch.allclose(final_context, stale_context)

    expected_nd = torch.full_like(inference_noise, clean_data_value)
    expected_3d = planner._convert_6d_to_3d(
        expected_nd.reshape(BATCH_SIZE * output_modes, kwargs["num_poses"], kwargs["traj_dim"])
    ).reshape(BATCH_SIZE, output_modes, kwargs["num_poses"], 3)
    expected_eval_keys = {field["name"] for field in profile["forward_contracts"]["eval"]["fields"]}
    assert set(output) == expected_eval_keys
    torch.testing.assert_close(output["trajectories"], expected_3d, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_interleaved_rejects_low_precision_before_retained_solver_updates(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    profile = _legacy_profile()
    kwargs = _materialize_kwargs(profile["kwargs"])
    planner = DiffusionPlanner(**kwargs).to(dtype=dtype).eval()
    z_ar = torch.zeros(
        BATCH_SIZE,
        kwargs["num_poses"] * kwargs["tokens_per_frame"],
        kwargs["encoder_dim"],
        dtype=dtype,
    )
    z_observed = torch.zeros(
        BATCH_SIZE,
        profile["attributes"]["num_observed_frames"] * kwargs["tokens_per_frame"],
        kwargs["encoder_dim"],
        dtype=dtype,
    )
    status_feature = torch.zeros(BATCH_SIZE, kwargs["status_dim"], dtype=dtype)
    inference_noise = torch.zeros(
        BATCH_SIZE,
        kwargs["num_samples"],
        kwargs["num_poses"],
        kwargs["traj_dim"],
        dtype=dtype,
    )

    def solver_update_bomb(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("retained solver update was reached before dtype validation")

    monkeypatch.setattr(planner, "_run_interleaved_solver_step", solver_update_bomb)
    with pytest.raises(TypeError, match="dtype|precision|float16|bfloat16|supported"):
        state = planner.init_interleaved_inference_state(
            status_feature,
            total_condition_updates=1,
            inference_noise=inference_noise,
        )
        planner.advance_interleaved_inference(state, z_ar, z_observed=z_observed)
