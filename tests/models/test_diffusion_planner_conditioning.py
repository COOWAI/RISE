"""Conditioning and topology coverage for the public DiffusionPlanner forward API."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from app.vjepa_cowa_world_model.models import diffusion_planner as diffusion_planner_module
from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/diffusion_planner_contract_v1.json"
BATCH_SIZE = 2


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
        torch.manual_seed(401)
        planner = DiffusionPlanner(**kwargs)
    return planner, profile, kwargs


def _nonzero_randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32) + 0.625


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
        torch.manual_seed(402)
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
            # The anchor frame stays separate from the supplied future-horizon noise.
            "inference_noise": _nonzero_randn(
                BATCH_SIZE,
                output_modes,
                input_config["num_poses"],
                input_config["traj_dim"],
            ),
        }


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


def _assert_recorded_output(output: dict[str, torch.Tensor], profile: dict[str, Any], phase: str) -> None:
    fields = profile["forward_contracts"][phase]["fields"]
    assert set(output) == {field["name"] for field in fields}
    for field in fields:
        value = output[field["name"]]
        assert list(value.shape) == field["shape"]
        assert str(value.dtype) == field["dtype"]
        assert (value.ndim == 0) is field["is_scalar"]
        if value.is_floating_point():
            assert torch.isfinite(value).all()


@pytest.mark.parametrize(
    ("profile_name", "expected_mode"),
    (("legacy_single_6d", "single_token"), ("formal_joint_anchor_v2", "per_pose_token")),
)
def test_both_trajectory_token_modes_train_and_evaluate(profile_name: str, expected_mode: str) -> None:
    planner, profile, kwargs = _build_profile(profile_name)
    inputs = _synthetic_inputs(profile, kwargs)

    assert planner.trajectory_token_mode == expected_mode
    _assert_recorded_output(_train(planner, inputs), profile, "train")
    _assert_recorded_output(_eval(planner, inputs), profile, "eval")


@pytest.mark.parametrize(
    ("constructor_mode", "expected_mode"),
    ((None, "concat"), ("none", "none"), ("concat", "concat"), ("concat_type_embed", "concat_type_embed")),
)
def test_observed_token_modes_normalize_and_accept_their_public_inputs(
    constructor_mode: str | None,
    expected_mode: str,
) -> None:
    planner, profile, kwargs = _build_profile(
        "legacy_single_6d",
        overrides={"observed_token_mode": constructor_mode},
    )
    inputs = _synthetic_inputs(profile, kwargs)
    input_config = _input_config(profile, kwargs)

    assert planner.observed_token_mode == expected_mode
    if constructor_mode is None:
        assert expected_mode == profile["attributes"]["observed_token_mode"]
    if expected_mode in {"concat", "concat_type_embed"}:
        z_observed = inputs["z_observed"]
        assert isinstance(z_observed, torch.Tensor)
        assert z_observed.shape == (
            BATCH_SIZE,
            input_config["num_observed_frames"] * input_config["tokens_per_frame"],
            input_config["encoder_dim"],
        )
    else:
        assert inputs["z_observed"] is None
    _assert_recorded_output(_train(planner, inputs), profile, "train")


@pytest.mark.parametrize("constructor_mode", (None, "concat", "concat_type_embed"))
def test_required_observed_tokens_reject_missing_and_malformed_inputs(constructor_mode: str | None) -> None:
    planner, profile, kwargs = _build_profile(
        "legacy_single_6d",
        overrides={"observed_token_mode": constructor_mode},
    )
    inputs = _synthetic_inputs(profile, kwargs)
    input_config = _input_config(profile, kwargs)
    expected_tokens = input_config["num_observed_frames"] * input_config["tokens_per_frame"]

    with pytest.raises(ValueError, match="z_observed"):
        _train(planner, {**inputs, "z_observed": None})
    with pytest.raises(ValueError, match="z_observed"):
        _train(
            planner,
            {
                **inputs,
                "z_observed": _nonzero_randn(BATCH_SIZE, expected_tokens, input_config["encoder_dim"] + 1),
            },
        )


@pytest.mark.parametrize("constructor_mode", ("concat", "concat_type_embed"))
@pytest.mark.parametrize("z_ar_tokens", (5, 6))
def test_required_observed_tokens_never_fall_back_based_on_z_ar_divisibility(
    constructor_mode: str,
    z_ar_tokens: int,
) -> None:
    planner, profile, kwargs = _build_profile(
        "legacy_single_6d",
        overrides={"observed_token_mode": constructor_mode},
    )
    inputs = _synthetic_inputs(profile, kwargs)
    z_ar = _nonzero_randn(BATCH_SIZE, z_ar_tokens, kwargs["encoder_dim"])

    with pytest.raises(ValueError, match="z_observed"):
        _train(planner, {**inputs, "z_ar": z_ar, "z_observed": None})


def test_action_history_conditioning_accepts_valid_and_rejects_invalid_inputs() -> None:
    planner, profile, kwargs = _build_profile("formal_observed_action_v2")
    inputs = _synthetic_inputs(profile, kwargs)
    input_config = _input_config(profile, kwargs)
    history = inputs["action_history"]
    assert isinstance(history, torch.Tensor)
    assert history.shape == (
        BATCH_SIZE,
        input_config["num_observed_frames"],
        input_config["action_history_dim"],
    )
    _assert_recorded_output(_train(planner, inputs), profile, "train")

    with pytest.raises(ValueError, match="action_history"):
        _train(planner, {**inputs, "action_history": None})
    with pytest.raises(ValueError, match="action_history"):
        _train(
            planner,
            {
                **inputs,
                "action_history": _nonzero_randn(
                    BATCH_SIZE,
                    input_config["num_observed_frames"],
                    input_config["action_history_dim"] + 1,
                ),
            },
        )


@pytest.mark.parametrize("context_source", ("observed", "action_history"))
def test_empty_last_frame_history_accepts_other_valid_context_sources(context_source: str) -> None:
    overrides: dict[str, Any]
    if context_source == "observed":
        overrides = {"observed_token_mode": "concat", "use_action_history": False}
    else:
        overrides = {"observed_token_mode": "none", "use_action_history": True}
    planner, profile, kwargs = _build_profile("legacy_single_6d", overrides=overrides)
    input_config = _input_config(profile, kwargs)
    empty_z_ar = torch.empty(BATCH_SIZE, 0, input_config["encoder_dim"])
    z_observed = None
    action_history = None
    if context_source == "observed":
        z_observed = _nonzero_randn(
            BATCH_SIZE,
            input_config["num_observed_frames"] * input_config["tokens_per_frame"],
            input_config["encoder_dim"],
        )
    else:
        action_history = _nonzero_randn(
            BATCH_SIZE,
            input_config["num_observed_frames"],
            input_config["action_history_dim"],
        )

    context = planner._prepare_context(empty_z_ar, None, z_observed, action_history)

    assert planner.use_last_frame_only is True
    assert context.ndim == 3
    assert context.shape[0] == BATCH_SIZE
    assert context.shape[1] > 0
    assert torch.isfinite(context).all()


def test_empty_last_frame_history_rejects_when_no_context_source_exists() -> None:
    planner, _, kwargs = _build_profile(
        "legacy_single_6d",
        overrides={"observed_token_mode": "none", "use_action_history": False},
    )
    empty_z_ar = torch.empty(BATCH_SIZE, 0, kwargs["encoder_dim"])

    with pytest.raises(ValueError, match="context|z_ar|empty|token"):
        planner._prepare_context(empty_z_ar, None, None, None)


def test_command_split_status_accepts_exact_width_and_rejects_malformed_status() -> None:
    planner, profile, kwargs = _build_profile("command_split_v2")
    inputs = _synthetic_inputs(profile, kwargs)
    status = inputs["status_feature"]
    assert isinstance(status, torch.Tensor)
    assert status.shape == (BATCH_SIZE, kwargs["status_dim"])
    assert 0 < planner.command_dim < kwargs["status_dim"]
    _assert_recorded_output(_train(planner, inputs), profile, "train")

    with pytest.raises(ValueError, match="status_feature|status"):
        _train(planner, {**inputs, "status_feature": status[:, :-1]})


def test_anchor_conditioning_is_required_and_excluded_from_returned_horizon() -> None:
    planner, profile, kwargs = _build_profile("formal_joint_anchor_v2")
    inputs = _synthetic_inputs(profile, kwargs)
    anchor = inputs["anchor_state"]
    assert isinstance(anchor, torch.Tensor)
    assert anchor.shape == (BATCH_SIZE, kwargs["traj_dim"])

    output = _eval(planner, inputs)
    _assert_recorded_output(output, profile, "eval")
    eval_fields = {field["name"]: field for field in profile["forward_contracts"]["eval"]["fields"]}
    assert output["trajectories"].shape[2] == eval_fields["trajectories"]["shape"][2]
    assert inputs["inference_noise"].shape[2] == kwargs["num_poses"]

    with pytest.raises(ValueError, match="anchor_state|anchor"):
        _eval(planner, {**inputs, "anchor_state": None})
    with pytest.raises(ValueError, match="anchor_state|anchor"):
        _eval(planner, {**inputs, "anchor_state": anchor[:, :-1]})


@pytest.mark.parametrize("dtype", (torch.float64, torch.float16))
def test_default_anchor_uses_active_planner_dtype_and_device(dtype: torch.dtype) -> None:
    planner, _, _ = _build_profile("formal_joint_anchor_v2")
    planner.to(dtype=dtype)
    active_parameter = next(planner.parameters())

    anchor = planner._get_anchor(None, BATCH_SIZE, active_parameter.device)

    assert isinstance(anchor, torch.Tensor)
    assert anchor.dtype == active_parameter.dtype
    assert anchor.device == active_parameter.device


@pytest.mark.parametrize("malformation", ("dtype", "device"))
def test_explicit_anchor_rejects_dtype_and_device_mismatch(malformation: str) -> None:
    planner, _, kwargs = _build_profile("formal_joint_anchor_v2")
    active_parameter = next(planner.parameters())
    if malformation == "dtype":
        anchor = torch.zeros(BATCH_SIZE, kwargs["traj_dim"], dtype=torch.float64, device=active_parameter.device)
    else:
        anchor = torch.empty(BATCH_SIZE, kwargs["traj_dim"], dtype=active_parameter.dtype, device="meta")

    with pytest.raises((TypeError, ValueError), match=f"anchor|{malformation}"):
        planner._get_anchor(anchor, BATCH_SIZE, active_parameter.device)


@pytest.mark.parametrize(
    "profile_name",
    ("formal_joint_anchor_v2", "independent_modes", "mode_token_expansion", "adaln_v3"),
)
def test_multimodal_topologies_match_recorded_train_and_eval_contracts(profile_name: str) -> None:
    planner, profile, kwargs = _build_profile(profile_name)
    inputs = _synthetic_inputs(profile, kwargs)

    train_output = _train(planner, inputs)
    eval_output = _eval(planner, inputs)

    _assert_recorded_output(train_output, profile, "train")
    _assert_recorded_output(eval_output, profile, "eval")
    confidences = eval_output["confidences"]
    assert torch.isfinite(confidences).all()
    assert torch.all(confidences >= 0)
    torch.testing.assert_close(confidences.sum(dim=1), torch.ones(BATCH_SIZE))


def test_nonempty_regression_timestep_weights_train_and_evaluate() -> None:
    planner, profile, kwargs = _build_profile("weighted_regression")
    inputs = _synthetic_inputs(profile, kwargs)

    assert planner.reg_timestep_weights.numel() == profile["kwargs"]["num_poses"] > 0
    assert torch.all(planner.reg_timestep_weights != 0)
    _assert_recorded_output(_train(planner, inputs), profile, "train")
    _assert_recorded_output(_eval(planner, inputs), profile, "eval")


@pytest.mark.parametrize("malformation", ("rank", "shape"))
def test_inference_noise_rejects_wrong_rank_and_shape(malformation: str) -> None:
    planner, profile, kwargs = _build_profile("formal_joint_anchor_v2")
    inputs = _synthetic_inputs(profile, kwargs)
    noise = inputs["inference_noise"]
    assert isinstance(noise, torch.Tensor)
    if malformation == "rank":
        invalid_noise = noise.flatten(start_dim=2)
    else:
        invalid_noise = torch.cat((noise, noise[:, :, :1]), dim=2)

    with pytest.raises(ValueError, match="inference_noise|noise"):
        _eval(planner, {**inputs, "inference_noise": invalid_noise})


@pytest.mark.parametrize("malformation", ("dtype", "device"))
def test_inference_noise_rejects_reference_mismatch_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    planner, profile, kwargs = _build_profile("formal_joint_anchor_v2")
    inputs = _synthetic_inputs(profile, kwargs)
    noise = inputs["inference_noise"]
    assert isinstance(noise, torch.Tensor)
    if malformation == "dtype":
        invalid_noise = noise.to(torch.float64)
    else:
        invalid_noise = torch.empty(noise.shape, dtype=noise.dtype, device="meta")

    def sampler_bomb(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("sampling was reached before inference_noise validation")

    monkeypatch.setattr(diffusion_planner_module, "dpm_sampler", sampler_bomb)
    with pytest.raises(ValueError, match=f"inference_noise|noise|{malformation}"):
        _eval(planner, {**inputs, "inference_noise": invalid_noise})


@pytest.mark.parametrize("malformation", ("rank", "shape"))
def test_ground_truth_rejects_wrong_rank_and_shape(malformation: str) -> None:
    planner, profile, kwargs = _build_profile("formal_joint_anchor_v2")
    inputs = _synthetic_inputs(profile, kwargs)
    gt = inputs["gt_trajectory"]
    assert isinstance(gt, torch.Tensor)
    if malformation == "rank":
        invalid_gt = gt.flatten(start_dim=1)
    else:
        invalid_gt = torch.cat((gt, gt[:, :1]), dim=1)

    with pytest.raises(ValueError, match="gt_trajectory|ground.truth|shape"):
        _train(planner, {**inputs, "gt_trajectory": invalid_gt})
