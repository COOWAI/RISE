"""Public loss, inference, buffer, and constructor contract for DiffusionPlanner."""

from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/diffusion_planner_contract_v1.json"
BATCH_SIZE = 2
FINITE_LOSS_NAMES = ("loss", "reg_loss", "conf_loss", "cover_loss", "vel_loss", "yaw_loss")
INVALID_CASE_NAMES = (
    "invalid_trajectory_token_mode",
    "invalid_adaln_version",
    "invalid_observed_token_mode",
    "single_token_mode_expansion",
    "independent_mode_expansion",
)


def _load_contract() -> dict[str, Any]:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    return contract


def _materialize_kwargs(serialized_kwargs: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name, value in serialized_kwargs.items():
        if isinstance(value, dict) and set(value) == {"shape", "dtype"}:
            dtype = getattr(torch, value["dtype"].removeprefix("torch."))
            kwargs[name] = torch.ones(value["shape"], dtype=dtype)
        else:
            kwargs[name] = copy.deepcopy(value)
    return kwargs


def _build_profile(
    profile_name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> tuple[DiffusionPlanner, dict[str, Any], dict[str, Any]]:
    profile = _load_contract()["profiles"][profile_name]
    kwargs = _materialize_kwargs(profile["kwargs"])
    if overrides:
        kwargs.update(overrides)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        planner = DiffusionPlanner(**kwargs)
    return planner, profile, kwargs


def _nonzero_randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32) + 0.375


def _input_config(profile: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    input_config = copy.deepcopy({**profile["attributes"], **kwargs})
    if "observed_token_mode" in kwargs and kwargs["observed_token_mode"] is None:
        input_config["observed_token_mode"] = copy.deepcopy(profile["attributes"]["observed_token_mode"])
    return input_config


def _synthetic_inputs(profile: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, torch.Tensor | None]:
    input_config = _input_config(profile, kwargs)
    num_modes = input_config["num_modes"]
    output_modes = num_modes if num_modes > 1 else input_config["num_samples"]
    observed_token_mode = input_config["observed_token_mode"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(202)
        z_observed = None
        if observed_token_mode in {"concat", "concat_type_embed"}:
            z_observed = _nonzero_randn(
                BATCH_SIZE,
                input_config["num_observed_frames"] * input_config["tokens_per_frame"],
                input_config["encoder_dim"],
            )
        action_history = None
        if input_config["use_action_history"]:
            action_history = _nonzero_randn(
                BATCH_SIZE,
                input_config["num_observed_frames"],
                input_config["action_history_dim"],
            )
        anchor_state = None
        if input_config["use_anchor_frame"]:
            anchor_state = _nonzero_randn(BATCH_SIZE, input_config["traj_dim"])
        return {
            "z_ar": _nonzero_randn(
                BATCH_SIZE,
                input_config["num_poses"] * input_config["tokens_per_frame"],
                input_config["encoder_dim"],
            ),
            "status_feature": _nonzero_randn(BATCH_SIZE, input_config["status_dim"]),
            "z_observed": z_observed,
            "action_history": action_history,
            "anchor_state": anchor_state,
            "gt_trajectory": _nonzero_randn(
                BATCH_SIZE,
                input_config["num_poses"],
                input_config["traj_dim"],
            ),
            # Noise covers future poses only. The optional anchor is never supplied as a noise frame.
            "inference_noise": _nonzero_randn(
                BATCH_SIZE,
                output_modes,
                input_config["num_poses"],
                input_config["traj_dim"],
            ),
        }


def _expected_keys(profile: dict[str, Any], phase: str) -> set[str]:
    return {field["name"] for field in profile["forward_contracts"][phase]["fields"]}


def _field_descriptors(profile: dict[str, Any], phase: str) -> dict[str, dict[str, Any]]:
    return {field["name"]: field for field in profile["forward_contracts"][phase]["fields"]}


def _recorded_regression_weights(*, requires_grad: bool = False) -> torch.Tensor:
    profile = _load_contract()["profiles"]["weighted_regression"]
    num_poses = profile["kwargs"]["num_poses"]
    return torch.arange(1, num_poses + 1, dtype=torch.float32, requires_grad=requires_grad)


class _UnitStdTrainingSDEProbe:
    def __init__(self) -> None:
        self.clean_inputs: list[torch.Tensor] = []

    def marginal_prob(self, x_start: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        del t
        self.clean_inputs.append(x_start.detach().clone())
        std = torch.ones(
            (x_start.shape[0],) + (1,) * (x_start.ndim - 1),
            dtype=x_start.dtype,
            device=x_start.device,
        )
        return torch.zeros_like(x_start), std


def _train(planner: DiffusionPlanner, inputs: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor]:
    planner.train()
    return planner(
        inputs["z_ar"],
        inputs["status_feature"],
        z_observed=inputs["z_observed"],
        action_history=inputs["action_history"],
        gt_trajectory=inputs["gt_trajectory"],
        anchor_state=inputs["anchor_state"],
    )


def _eval(planner: DiffusionPlanner, inputs: dict[str, torch.Tensor | None]) -> dict[str, torch.Tensor]:
    planner.eval()
    with torch.no_grad():
        return planner(
            inputs["z_ar"],
            inputs["status_feature"],
            z_observed=inputs["z_observed"],
            action_history=inputs["action_history"],
            anchor_state=inputs["anchor_state"],
            inference_noise=inputs["inference_noise"],
        )


def test_formal_observed_action_training_has_finite_losses_and_gradients() -> None:
    planner, profile, kwargs = _build_profile("formal_observed_action_v2")
    inputs = _synthetic_inputs(profile, kwargs)

    output = _train(planner, inputs)

    assert set(output) == _expected_keys(profile, "train")
    for name in FINITE_LOSS_NAMES:
        assert output[name].ndim == 0
        assert torch.isfinite(output[name])
    train_fields = _field_descriptors(profile, "train")
    for name in ("winner_idx", "winner_traj_3d", "awta_temperature", "cls_sample_valid_ratio"):
        descriptor = train_fields[name]
        value = output[name]
        assert list(value.shape) == descriptor["shape"]
        assert str(value.dtype) == descriptor["dtype"]
        assert (value.ndim == 0) == descriptor["is_scalar"]
    for name in ("awta_temperature", "cls_sample_valid_ratio"):
        assert torch.isfinite(output[name])

    output["loss"].backward()
    trainable_grads = [parameter.grad for parameter in planner.parameters() if parameter.requires_grad]
    assert any(
        gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0.0
        for gradient in trainable_grads
    )


def test_training_regression_target_is_clean_x_start_and_not_sampled_epsilon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner, profile, kwargs = _build_profile("legacy_single_6d")
    inputs = _synthetic_inputs(profile, kwargs)
    clean_gt = torch.zeros_like(inputs["gt_trajectory"])
    inputs = {**inputs, "gt_trajectory": clean_gt}
    sde_probe = _UnitStdTrainingSDEProbe()
    denoiser_inputs: list[torch.Tensor] = []

    def clean_data_probe(
        x_t: torch.Tensor,
        t: torch.Tensor,
        cross_c: torch.Tensor,
        status_emb: torch.Tensor,
    ) -> tuple[None, torch.Tensor]:
        del t, cross_c, status_emb
        denoiser_inputs.append(x_t.detach().clone())
        return None, torch.zeros_like(x_t)

    monkeypatch.setattr(planner, "sde", sde_probe)
    monkeypatch.setattr(planner.dit, "forward", clean_data_probe)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(303)
        output = _train(planner, inputs)

    assert sde_probe.clean_inputs
    assert any(torch.equal(clean_input, torch.zeros_like(clean_input)) for clean_input in sde_probe.clean_inputs)
    assert len(denoiser_inputs) == 1
    sampled_epsilon = denoiser_inputs[0]
    epsilon_target_loss = sampled_epsilon.square().mean()
    assert epsilon_target_loss > 0
    torch.testing.assert_close(output["reg_loss"], torch.zeros_like(output["reg_loss"]), rtol=0.0, atol=0.0)
    assert not torch.isclose(output["reg_loss"], epsilon_target_loss)


def test_supplied_future_noise_makes_formal_inference_deterministic() -> None:
    planner, profile, kwargs = _build_profile("formal_observed_action_v2")
    inputs = _synthetic_inputs(profile, kwargs)
    noise = inputs["inference_noise"]
    assert isinstance(noise, torch.Tensor)
    output_modes = kwargs["num_modes"] if kwargs["num_modes"] > 1 else kwargs["num_samples"]
    assert noise.shape == (BATCH_SIZE, output_modes, kwargs["num_poses"], kwargs["traj_dim"])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(811)
        first = _eval(planner, {**inputs, "inference_noise": noise.clone()})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(977)
        second = _eval(planner, {**inputs, "inference_noise": noise.clone()})

    assert set(first) == set(second) == _expected_keys(profile, "eval")
    torch.testing.assert_close(first["trajectories"], second["trajectories"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first["confidences"], second["confidences"], rtol=0.0, atol=0.0)
    eval_fields = _field_descriptors(profile, "eval")
    assert list(first["trajectories"].shape) == eval_fields["trajectories"]["shape"]
    assert list(first["confidences"].shape) == eval_fields["confidences"]["shape"]
    assert torch.isfinite(first["trajectories"]).all()
    assert torch.isfinite(first["confidences"]).all()
    assert torch.all(first["confidences"] >= 0)
    torch.testing.assert_close(first["confidences"].sum(dim=1), torch.ones(BATCH_SIZE))

    different_noise = torch.linspace(-1.25, 1.25, noise.numel(), dtype=noise.dtype).reshape(noise.shape)
    assert not torch.equal(noise, different_noise)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(811)
        different = _eval(planner, {**inputs, "inference_noise": different_noise})
    assert not torch.equal(first["trajectories"], different["trajectories"])


def test_legacy_six_dimensional_training_and_eval_contract() -> None:
    planner, profile, kwargs = _build_profile("legacy_single_6d")
    inputs = _synthetic_inputs(profile, kwargs)
    gt = inputs["gt_trajectory"]
    assert isinstance(gt, torch.Tensor)
    assert gt.shape == (BATCH_SIZE, kwargs["num_poses"], kwargs["traj_dim"])

    train_output = _train(planner, inputs)
    recorded_train_keys = _expected_keys(profile, "train")
    assert len(recorded_train_keys) == 6
    assert set(train_output) == recorded_train_keys
    assert all(value.ndim == 0 and torch.isfinite(value) for value in train_output.values())

    with pytest.raises(ValueError, match="gt_trajectory|shape|traj_dim"):
        _train(planner, {**inputs, "gt_trajectory": gt[..., :3]})

    eval_output = _eval(planner, inputs)
    assert set(eval_output) == _expected_keys(profile, "eval")
    eval_fields = _field_descriptors(profile, "eval")
    assert list(eval_output["trajectories"].shape) == eval_fields["trajectories"]["shape"]
    assert list(eval_output["confidences"].shape) == eval_fields["confidences"]["shape"]


def test_set_awta_temperature_updates_public_training_value() -> None:
    planner, profile, kwargs = _build_profile("formal_joint_anchor_v2")
    inputs = _synthetic_inputs(profile, kwargs)

    assert planner.set_awta_temperature(2.25) is None
    torch.testing.assert_close(planner.awta_temperature, torch.tensor(2.25))
    output = _train(planner, inputs)
    assert set(output) == _expected_keys(profile, "train")
    torch.testing.assert_close(output["awta_temperature"], torch.tensor(2.25))


@pytest.mark.parametrize("requires_grad", (False, True))
def test_weighted_regression_buffer_preserves_tensor_contract(requires_grad: bool) -> None:
    weights = _recorded_regression_weights(requires_grad=requires_grad)
    planner, profile, _ = _build_profile("weighted_regression", overrides={"reg_timestep_weights": weights})
    buffers = dict(planner.named_buffers())

    assert "reg_timestep_weights" in buffers
    assert buffers["reg_timestep_weights"].dtype == torch.float32
    assert buffers["reg_timestep_weights"].shape == (profile["kwargs"]["num_poses"],)
    assert buffers["reg_timestep_weights"].requires_grad is requires_grad
    assert "reg_timestep_weights" not in planner.state_dict()


def test_all_zero_regression_timestep_weights_fail_fast() -> None:
    profile = _load_contract()["profiles"]["weighted_regression"]
    weights = torch.zeros(profile["kwargs"]["num_poses"], dtype=torch.float32)

    with pytest.raises(ValueError, match="weight|sum|positive"):
        _build_profile("weighted_regression", overrides={"reg_timestep_weights": weights})


def test_partial_zero_regression_timestep_weights_are_valid_and_finite() -> None:
    recorded_profile = _load_contract()["profiles"]["weighted_regression"]
    num_poses = recorded_profile["kwargs"]["num_poses"]
    num_modes = recorded_profile["kwargs"]["num_modes"]
    weights = torch.arange(num_poses, dtype=torch.float32)
    assert torch.any(weights == 0)
    assert torch.all(weights >= 0)
    assert weights.sum() > 0

    planner, profile, kwargs = _build_profile(
        "weighted_regression",
        overrides={"reg_timestep_weights": weights},
    )
    torch.testing.assert_close(planner.reg_timestep_weights, weights)
    pred_xy = torch.arange(BATCH_SIZE * num_modes * num_poses * 2, dtype=torch.float32).reshape(
        BATCH_SIZE,
        num_modes,
        num_poses,
        2,
    )
    gt_xy = torch.zeros(BATCH_SIZE, num_poses, 2, dtype=torch.float32)
    helper_loss = planner._xy_regression_loss_per_mode(pred_xy, gt_xy)
    output = _train(planner, _synthetic_inputs(profile, kwargs))

    assert torch.isfinite(helper_loss).all()
    assert torch.isfinite(output["reg_loss"])
    assert torch.isfinite(output["loss"])


def test_weighted_xy_regression_uses_relative_timestep_weights() -> None:
    profile = _load_contract()["profiles"]["weighted_regression"]
    num_poses = profile["kwargs"]["num_poses"]
    num_modes = profile["kwargs"]["num_modes"]
    weights = _recorded_regression_weights()
    scaled_weights = weights * 9.0
    uniform_weights = torch.ones_like(weights)
    planner, _, _ = _build_profile("weighted_regression", overrides={"reg_timestep_weights": weights})
    scaled_planner, _, _ = _build_profile(
        "weighted_regression",
        overrides={"reg_timestep_weights": scaled_weights},
    )
    uniform_planner, _, _ = _build_profile(
        "weighted_regression",
        overrides={"reg_timestep_weights": uniform_weights},
    )
    batch_scale = torch.arange(1, BATCH_SIZE + 1, dtype=torch.float32).reshape(BATCH_SIZE, 1, 1, 1)
    mode_scale = torch.arange(1, num_modes + 1, dtype=torch.float32).reshape(1, num_modes, 1, 1)
    timestep_scale = torch.arange(1, num_poses + 1, dtype=torch.float32).reshape(1, 1, num_poses, 1)
    error = batch_scale * mode_scale * timestep_scale
    pred_xy = torch.cat((error, error * 0.5), dim=-1)
    gt_xy = torch.zeros(BATCH_SIZE, num_poses, 2, dtype=torch.float32)

    loss = planner._xy_regression_loss_per_mode(pred_xy, gt_xy)
    scaled_loss = scaled_planner._xy_regression_loss_per_mode(pred_xy, gt_xy)
    uniform_loss = uniform_planner._xy_regression_loss_per_mode(pred_xy, gt_xy)

    assert loss.shape == (BATCH_SIZE, num_modes)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss, scaled_loss)
    assert not torch.allclose(loss, uniform_loss)


@pytest.mark.parametrize("profile_name", ("legacy_single_6d", "weighted_regression"))
def test_relative_timestep_weights_change_real_training_losses_and_reg_gradient(profile_name: str) -> None:
    profile = _load_contract()["profiles"][profile_name]
    relative_weights = torch.tensor([1.0, 2.0, 8.0])
    scaled_weights = relative_weights * 7.0
    reversed_weights = relative_weights.flip(0)
    planners = [
        _build_profile(profile_name, overrides={"reg_timestep_weights": weights})[0]
        for weights in (relative_weights, scaled_weights, reversed_weights)
    ]
    for planner in planners:
        with torch.no_grad():
            for parameter in planner.parameters():
                parameter.zero_()

    inputs = _synthetic_inputs(profile, _materialize_kwargs(profile["kwargs"]))
    recorded_gt = inputs["gt_trajectory"]
    assert isinstance(recorded_gt, torch.Tensor)
    gt = torch.zeros_like(recorded_gt)
    gt[:, :, 0] = torch.tensor([0.25, 2.0, 8.0])
    gt[:, :, 1] = torch.tensor([0.5, 3.0, 12.0])
    anchor_state = inputs["anchor_state"]
    if isinstance(anchor_state, torch.Tensor):
        anchor_state = torch.zeros_like(anchor_state)
    inputs = {**inputs, "gt_trajectory": gt, "anchor_state": anchor_state}
    assert not torch.equal(gt[:, 0, :2].abs().sum(), gt[:, -1, :2].abs().sum())

    def real_losses_and_reg_gradient(planner: DiffusionPlanner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planner.zero_grad(set_to_none=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(707)
            output = _train(planner, inputs)
        total_loss = output["loss"]
        reg_loss = output["reg_loss"]
        reg_loss.backward()
        gradient = torch.cat(
            [
                (
                    torch.zeros_like(parameter).flatten()
                    if parameter.grad is None
                    else parameter.grad.detach().flatten().clone()
                )
                for parameter in planner.parameters()
            ]
        )
        return total_loss.detach(), reg_loss.detach(), gradient

    weighted_total, weighted_reg, weighted_gradient = real_losses_and_reg_gradient(planners[0])
    scaled_total, scaled_reg, scaled_gradient = real_losses_and_reg_gradient(planners[1])
    reversed_total, reversed_reg, reversed_gradient = real_losses_and_reg_gradient(planners[2])

    assert torch.isfinite(
        torch.stack((weighted_total, scaled_total, reversed_total, weighted_reg, scaled_reg, reversed_reg))
    ).all()
    assert all(
        torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        for gradient in (weighted_gradient, scaled_gradient, reversed_gradient)
    )
    torch.testing.assert_close(weighted_total, scaled_total)
    torch.testing.assert_close(weighted_reg, scaled_reg)
    torch.testing.assert_close(weighted_gradient, scaled_gradient)
    assert not torch.allclose(weighted_total, reversed_total)
    assert not torch.allclose(weighted_reg, reversed_reg)
    assert not torch.allclose(weighted_gradient, reversed_gradient)


def test_generated_inference_noise_matches_half_precision_planner_input() -> None:
    planner, profile, kwargs = _build_profile("formal_joint_anchor_v2")
    planner.to(dtype=torch.float16)
    inputs = _synthetic_inputs(profile, kwargs)
    reference = inputs["z_ar"].to(dtype=torch.float16)
    output_modes = kwargs["num_modes"] if kwargs["num_modes"] > 1 else kwargs["num_samples"]

    noise = planner._resolve_inference_noise(
        None,
        B=reference.shape[0],
        K=output_modes,
        device=reference.device,
    )

    assert noise.is_floating_point()
    assert noise.dtype == reference.dtype == next(planner.parameters()).dtype
    assert noise.device == reference.device == next(planner.parameters()).device


@pytest.mark.parametrize(
    ("profile_name", "expected_adaln_version"),
    (("legacy_single_6d", "legacy"), ("formal_joint_anchor_v2", "v2"), ("adaln_v3", "v3")),
)
def test_fresh_adaln_blocks_are_zero_gated_while_output_heads_remain_nonzero(
    profile_name: str,
    expected_adaln_version: str,
) -> None:
    planner, profile, _ = _build_profile(profile_name)
    assert profile["attributes"]["adaln_version"] == expected_adaln_version

    blocks = tuple(planner.dit.blocks)
    assert blocks
    for block in blocks:
        modulation_linears = tuple(
            module for module in block.adaLN_modulation.modules() if isinstance(module, torch.nn.Linear)
        )
        assert modulation_linears
        output_linear = modulation_linears[-1]
        torch.testing.assert_close(output_linear.weight, torch.zeros_like(output_linear.weight), rtol=0.0, atol=0.0)
        assert output_linear.bias is not None
        torch.testing.assert_close(output_linear.bias, torch.zeros_like(output_linear.bias), rtol=0.0, atol=0.0)

    output_head_linears: dict[str, torch.nn.Linear] = {}
    for head_name in ("reg", "proj", "cls"):
        head = getattr(planner.dit.final_layer, head_name, None)
        if head is None:
            continue
        linears = tuple(module for module in head.modules() if isinstance(module, torch.nn.Linear))
        if linears:
            output_head_linears[head_name] = linears[-1]
    assert output_head_linears
    for output_linear in output_head_linears.values():
        assert torch.count_nonzero(output_linear.weight) > 0


@pytest.mark.parametrize("case_name", INVALID_CASE_NAMES)
def test_fixture_invalid_constructor_cases_raise_exact_recorded_exception(case_name: str) -> None:
    contract = _load_contract()
    assert tuple(contract["invalid_cases"]) == INVALID_CASE_NAMES
    case = contract["invalid_cases"][case_name]
    exception_type = getattr(builtins, case["exception_class"])

    with pytest.raises(exception_type) as exc_info:
        DiffusionPlanner(**_materialize_kwargs(case["kwargs"]))
    assert type(exc_info.value) is exception_type


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"num_modes": 0}, "num_modes"),
        ({"num_modes": -1}, "num_modes"),
        ({"num_modes": 1, "independent_modes": True}, "independent_modes|num_modes"),
        ({"command_dim": -1}, "command_dim"),
        ({"command_dim": 5}, "command_dim"),
    ),
)
def test_new_topology_validation_fails_fast(overrides: dict[str, Any], message: str) -> None:
    profile = _load_contract()["profiles"]["legacy_single_6d"]
    kwargs = _materialize_kwargs(profile["kwargs"])
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        DiffusionPlanner(**kwargs)
