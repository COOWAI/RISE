"""Integration contracts for retained diffusion-planner variants."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner
from app.vjepa_cowa_world_model.models.flow_matching_diffusion_planner import FlowMatchingDiffusionPlanner
from app.vjepa_cowa_world_model.models.prefix_conditioned_diffusion_planner import PrefixConditionedDiffusionPlanner
from app.vjepa_cowa_world_model.models.seeded_diffusion_planner import (
    PrefixConditionedSeededDiffusionPlanner,
    SeededDiffusionPlanner,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/diffusion_planner_contract_v1.json"
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
}
BASE_LOSS_KEYS = {"loss", "reg_loss", "conf_loss", "cover_loss", "vel_loss", "yaw_loss"}


def _legacy_contract() -> dict[str, Any]:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    return contract["profiles"]["legacy_single_6d"]


def _build_planner(planner_cls: type[DiffusionPlanner], **overrides: Any) -> DiffusionPlanner:
    kwargs = {**BASE, **overrides}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3101)
        return planner_cls(**kwargs)


def _synthetic_inputs(
    planner: DiffusionPlanner,
    *,
    context_source: str = "future",
) -> dict[str, torch.Tensor | None]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3102)
        z_observed = None
        action_history = None
        if context_source == "observed":
            z_observed = torch.randn(
                BATCH_SIZE,
                planner.num_observed_frames * planner.tokens_per_frame,
                planner.encoder_dim,
            )
        elif context_source == "action":
            action_history = torch.randn(
                BATCH_SIZE,
                planner.num_observed_frames,
                planner.action_history_dim,
            )
        elif context_source != "future":
            raise ValueError(f"unsupported test context source: {context_source}")

        output_modes = planner.num_modes if planner.num_modes > 1 else planner.num_samples
        return {
            "z_ar": torch.randn(
                BATCH_SIZE,
                planner.num_poses * planner.tokens_per_frame,
                planner.encoder_dim,
            ),
            "status_feature": torch.randn(BATCH_SIZE, 4),
            "z_observed": z_observed,
            "action_history": action_history,
            "gt_trajectory": torch.randn(BATCH_SIZE, planner.num_poses, planner.traj_dim),
            "inference_noise": torch.randn(
                BATCH_SIZE,
                output_modes,
                planner.num_poses,
                planner.traj_dim,
            ),
        }


def _train(
    planner: DiffusionPlanner,
    inputs: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    planner.train()
    return planner(
        inputs["z_ar"],
        inputs["status_feature"],
        z_observed=inputs["z_observed"],
        action_history=inputs["action_history"],
        gt_trajectory=inputs["gt_trajectory"],
    )


def _eval(
    planner: DiffusionPlanner,
    inputs: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    planner.eval()
    with torch.no_grad():
        return planner(
            inputs["z_ar"],
            inputs["status_feature"],
            z_observed=inputs["z_observed"],
            action_history=inputs["action_history"],
            inference_noise=inputs["inference_noise"],
        )


def _assert_base_contract(output: dict[str, torch.Tensor], phase: str) -> None:
    fields = _legacy_contract()["forward_contracts"][phase]["fields"]
    assert set(output) == {field["name"] for field in fields}
    for field in fields:
        value = output[field["name"]]
        assert list(value.shape) == field["shape"]
        assert str(value.dtype) == field["dtype"]
        assert (value.ndim == 0) is field["is_scalar"]
        assert torch.isfinite(value).all()


def _assert_deterministic_supplied_noise(
    planner: DiffusionPlanner,
    inputs: dict[str, torch.Tensor | None],
) -> dict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3103)
        first = _eval(planner, copy.copy(inputs))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9991)
        second = _eval(planner, copy.copy(inputs))

    assert set(first) == set(second) == {"trajectories", "confidences"}
    torch.testing.assert_close(first["trajectories"], second["trajectories"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first["confidences"], second["confidences"], rtol=0.0, atol=0.0)
    assert torch.isfinite(first["trajectories"]).all()
    assert torch.isfinite(first["confidences"]).all()
    torch.testing.assert_close(first["confidences"].sum(dim=1), torch.ones(BATCH_SIZE), rtol=1e-6, atol=1e-6)
    return first


def test_seeded_gaussian_variant_matches_base_training_and_inference_contracts() -> None:
    planner = _build_planner(SeededDiffusionPlanner, init_traj_strategy="gaussian", observed_token_mode="none")
    inputs = _synthetic_inputs(planner)

    _assert_base_contract(_train(planner, inputs), "train")
    inference = _assert_deterministic_supplied_noise(planner, inputs)
    _assert_base_contract(inference, "eval")


@pytest.mark.parametrize(
    ("planner_cls", "context_source", "variant_kwargs"),
    (
        (
            PrefixConditionedDiffusionPlanner,
            "observed",
            {"observed_token_mode": "concat", "use_action_history": False},
        ),
        (
            PrefixConditionedSeededDiffusionPlanner,
            "action",
            {
                "observed_token_mode": "none",
                "use_action_history": True,
                "init_traj_strategy": "gaussian",
            },
        ),
    ),
)
def test_prefix_variants_execute_a_legal_zero_prefix_with_an_external_context_source(
    planner_cls: type[DiffusionPlanner],
    context_source: str,
    variant_kwargs: dict[str, Any],
) -> None:
    planner = _build_planner(
        planner_cls,
        train_min_prefix_frames=0,
        train_full_prefix_prob=0.0,
        train_max_non_full_prefix_frames=0,
        **variant_kwargs,
    )
    inputs = _synthetic_inputs(planner, context_source=context_source)

    train_output = _train(planner, inputs)

    prefix_sample = planner.last_train_prefix_sample
    assert prefix_sample is not None
    assert prefix_sample.prefix_steps == 0
    _assert_base_contract(train_output, "train")
    inference = _assert_deterministic_supplied_noise(planner, inputs)
    _assert_base_contract(inference, "eval")


@pytest.mark.parametrize(
    ("planner_cls", "variant_kwargs"),
    (
        (PrefixConditionedDiffusionPlanner, {}),
        (PrefixConditionedSeededDiffusionPlanner, {"init_traj_strategy": "gaussian"}),
    ),
)
def test_prefix_variants_execute_a_fixed_one_frame_prefix_profile(
    planner_cls: type[DiffusionPlanner],
    variant_kwargs: dict[str, Any],
) -> None:
    planner = _build_planner(
        planner_cls,
        train_min_prefix_frames=1,
        train_full_prefix_prob=0.0,
        train_max_non_full_prefix_frames=1,
        observed_token_mode="none",
        **variant_kwargs,
    )
    inputs = _synthetic_inputs(planner)

    train_output = _train(planner, inputs)

    prefix_sample = planner.last_train_prefix_sample
    assert prefix_sample is not None
    assert prefix_sample.prefix_steps == 1
    _assert_base_contract(train_output, "train")
    inference = _assert_deterministic_supplied_noise(planner, inputs)
    _assert_base_contract(inference, "eval")


def test_flow_matching_variant_retains_flow_losses_and_deterministic_inference() -> None:
    planner = _build_planner(
        FlowMatchingDiffusionPlanner,
        flow_sampler="euler",
        observed_token_mode="none",
    )
    inputs = _synthetic_inputs(planner)

    training = _train(planner, inputs)

    assert BASE_LOSS_KEYS <= set(training)
    assert {"flow_loss", "endpoint_reg_loss"} <= set(training)
    for name in BASE_LOSS_KEYS | {"flow_loss", "endpoint_reg_loss"}:
        assert training[name].ndim == 0
        assert torch.isfinite(training[name])
    assert training["winner_traj_3d"].shape == (BATCH_SIZE, planner.num_poses, 3)
    assert torch.isfinite(training["winner_traj_3d"]).all()
    inference = _assert_deterministic_supplied_noise(planner, inputs)
    _assert_base_contract(inference, "eval")

    with pytest.raises(NotImplementedError, match="interleaved"):
        planner.init_interleaved_inference_state(
            inputs["status_feature"],
            total_condition_updates=2,
            inference_noise=inputs["inference_noise"],
        )


@pytest.mark.parametrize(
    ("planner_cls", "variant_kwargs"),
    (
        (SeededDiffusionPlanner, {"init_traj_strategy": "gaussian"}),
        (FlowMatchingDiffusionPlanner, {"flow_sampler": "euler"}),
    ),
)
def test_single_mode_multisample_inference_preserves_double_dtype_with_supplied_noise(
    planner_cls: type[DiffusionPlanner],
    variant_kwargs: dict[str, Any],
) -> None:
    planner = _build_planner(planner_cls, observed_token_mode="none", **variant_kwargs).double().eval()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3104)
        z_ar = torch.randn(
            BATCH_SIZE,
            planner.num_poses * planner.tokens_per_frame,
            planner.encoder_dim,
            dtype=torch.float64,
        )
        status_feature = torch.randn(BATCH_SIZE, BASE["status_dim"], dtype=torch.float64)
        inference_noise = torch.randn(
            BATCH_SIZE,
            planner.num_samples,
            planner.num_poses,
            planner.traj_dim,
            dtype=torch.float64,
        )

    with torch.no_grad():
        output = planner(
            z_ar,
            status_feature,
            z_observed=None,
            action_history=None,
            inference_noise=inference_noise,
        )

    assert planner.num_modes == 1
    assert planner.num_samples > 1
    assert output["trajectories"].dtype == torch.float64
    assert output["confidences"].dtype == torch.float64
    assert torch.isfinite(output["trajectories"]).all()
    assert torch.isfinite(output["confidences"]).all()
    torch.testing.assert_close(
        output["confidences"].sum(dim=1),
        torch.ones(BATCH_SIZE, dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("planner_cls", "variant_kwargs"),
    (
        (SeededDiffusionPlanner, {"init_traj_strategy": "gaussian"}),
        (FlowMatchingDiffusionPlanner, {"flow_sampler": "euler"}),
    ),
)
def test_joint_single_token_multimode_never_routes_as_independent_modes(
    monkeypatch: pytest.MonkeyPatch,
    planner_cls: type[DiffusionPlanner],
    variant_kwargs: dict[str, Any],
) -> None:
    planner = _build_planner(
        planner_cls,
        trajectory_token_mode="single_token",
        num_modes=2,
        independent_modes=False,
        observed_token_mode="none",
        **variant_kwargs,
    )
    inputs = _synthetic_inputs(planner)

    def independent_route_bomb(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("joint single-token multimode execution was routed through an independent-mode path")

    monkeypatch.setattr(planner, "_training_forward_independent", independent_route_bomb)
    monkeypatch.setattr(planner, "_inference_forward_independent", independent_route_bomb)

    training = _train(planner, inputs)
    inference = _assert_deterministic_supplied_noise(planner, inputs)

    assert planner.independent_modes is False
    assert BASE_LOSS_KEYS <= set(training)
    assert all(torch.isfinite(training[name]) for name in BASE_LOSS_KEYS)
    assert training["winner_traj_3d"].shape == (BATCH_SIZE, planner.num_poses, 3)
    assert inference["trajectories"].shape == (BATCH_SIZE, 2, planner.num_poses, 3)
    assert inference["confidences"].shape == (BATCH_SIZE, 2)
