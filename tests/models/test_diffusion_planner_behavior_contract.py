"""Exercise the recorded black-box behavior contract on synthetic CPU tensors."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from app.vjepa_cowa_world_model.diffusion_utils.sampling import dpm_sampler
from app.vjepa_cowa_world_model.diffusion_utils.sde import VPSDE_linear
from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/diffusion_planner_contract_v1.json"
PROFILE_NAMES = (
    "legacy_single_6d",
    "formal_joint_anchor_v2",
    "formal_observed_action_v2",
    "command_split_v2",
    "independent_modes",
    "mode_token_expansion",
    "adaln_v3",
    "weighted_regression",
)
INVALID_CASE_NAMES = (
    "invalid_trajectory_token_mode",
    "invalid_adaln_version",
    "invalid_observed_token_mode",
    "single_token_mode_expansion",
    "independent_mode_expansion",
)


def _load_contract() -> dict:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    return contract


def _materialize_kwargs(serialized_kwargs: dict) -> dict:
    kwargs = {}
    for name, value in serialized_kwargs.items():
        if isinstance(value, dict) and set(value) == {"shape", "dtype"}:
            dtype = getattr(torch, value["dtype"].removeprefix("torch."))
            kwargs[name] = torch.ones(value["shape"], dtype=dtype)
        else:
            kwargs[name] = value
    return kwargs


def _normalize_attribute(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if isinstance(value, tuple):
        return [_normalize_attribute(item) for item in value]
    raise TypeError(f"Unsupported compatibility attribute type: {type(value).__name__}")


def _attribute_schema(planner: DiffusionPlanner) -> dict:
    attributes = {
        name: _normalize_attribute(value)
        for name, value in vars(planner).items()
        if not name.startswith("_") and isinstance(value, (bool, int, float, str))
    }
    for name in ("_has_velocity", "_yaw_slice", "_uses_batch_expansion", "_batch_K"):
        attributes[name] = _normalize_attribute(getattr(planner, name))
    attributes["dit.model_type"] = _normalize_attribute(planner.dit.model_type)
    return attributes


def _named_buffer_schema(module: torch.nn.Module) -> list[dict]:
    state_names = set(module.state_dict())
    return [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "in_state_dict": name in state_names,
        }
        for name, tensor in module.named_buffers()
    ]


def _output_fields(output: dict[str, torch.Tensor]) -> list[dict]:
    return [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "is_scalar": tensor.ndim == 0,
        }
        for name, tensor in output.items()
    ]


def _optional_tensor_schema(name: str, value, *, include_device: bool = False) -> dict:
    if value is None:
        schema = {
            "name": name,
            "shape": None,
            "dtype": None,
            "is_scalar": False,
            "is_none": True,
        }
    else:
        assert isinstance(value, torch.Tensor)
        schema = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "is_scalar": value.ndim == 0,
            "is_none": False,
        }
    if include_device:
        schema["device"] = None if value is None else str(value.device)
    return schema


def _synthetic_inputs(planner: DiffusionPlanner, serialized_kwargs: dict) -> dict:
    batch = 2
    z_observed = None
    if planner.observed_token_mode in {"concat", "concat_type_embed"}:
        z_observed = torch.zeros(
            batch,
            planner.num_observed_frames * planner.tokens_per_frame,
            planner.encoder_dim,
        )
    action_history = None
    if planner.use_action_history:
        action_history = torch.zeros(batch, planner.num_observed_frames, planner.action_history_dim)
    anchor_state = None
    if planner.use_anchor_frame:
        anchor_state = torch.zeros(batch, planner.traj_dim)
    output_modes = planner.num_modes if planner.num_modes > 1 else planner.num_samples
    return {
        "z_ar": torch.zeros(batch, 6, planner.encoder_dim),
        "status_feature": torch.zeros(batch, serialized_kwargs["status_dim"]),
        "z_context": None,
        "z_observed": z_observed,
        "action_history": action_history,
        "anchor_state": anchor_state,
        "gt_trajectory": torch.zeros(batch, planner.num_poses, planner.traj_dim),
        "inference_noise": torch.zeros(batch, output_modes, planner.num_poses, planner.traj_dim),
    }


def _dit_forward_contract(planner: DiffusionPlanner, inputs: dict) -> dict:
    context_tokens = planner._prepare_context(
        inputs["z_ar"],
        inputs["z_context"],
        inputs["z_observed"],
        inputs["action_history"],
    )
    status_emb = planner._prepare_status(inputs["status_feature"])
    expanded_batch = inputs["z_ar"].shape[0]
    if planner._uses_batch_expansion:
        expanded_batch *= planner._batch_K
        context_tokens = context_tokens.repeat_interleave(planner._batch_K, dim=0)
        status_emb = status_emb.repeat_interleave(planner._batch_K, dim=0)

    if planner.trajectory_token_mode == "single_token":
        x = torch.zeros(expanded_batch, planner.total_frames * planner.traj_dim)
    elif planner.num_modes > 1 and not planner._uses_batch_expansion:
        x = torch.zeros(expanded_batch, planner.num_modes, planner.total_frames, planner.traj_dim)
    else:
        x = torch.zeros(expanded_batch, planner.total_frames, planner.traj_dim)
    result = planner.dit(x, torch.zeros(expanded_batch), context_tokens, status_emb)
    assert isinstance(result, tuple)
    assert len(result) == 2
    classification, prediction = result
    return {
        "signature": str(inspect.signature(type(planner.dit).forward)),
        "fields": [
            _optional_tensor_schema("classification", classification),
            _optional_tensor_schema("prediction", prediction),
        ],
    }


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_profile_attributes_buffers_and_nested_dit_contract(profile_name: str) -> None:
    contract = _load_contract()
    assert tuple(contract["profiles"]) == PROFILE_NAMES
    profile = contract["profiles"][profile_name]
    planner = DiffusionPlanner(**_materialize_kwargs(profile["kwargs"]))
    inputs = _synthetic_inputs(planner, profile["kwargs"])

    assert _attribute_schema(planner) == profile["attributes"]
    assert _named_buffer_schema(planner) == profile["named_buffers"]
    dit_contract = _dit_forward_contract(planner, inputs)
    assert dit_contract == profile["dit_forward_contract"]
    assert [field["name"] for field in dit_contract["fields"]] == ["classification", "prediction"]

    if profile_name == "weighted_regression":
        buffers = {buffer["name"]: buffer for buffer in profile["named_buffers"]}
        assert buffers["reg_timestep_weights"] == {
            "name": "reg_timestep_weights",
            "shape": [3],
            "dtype": "torch.float32",
            "in_state_dict": False,
        }


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_profile_train_and_eval_output_contracts(profile_name: str) -> None:
    contract = _load_contract()
    profile = contract["profiles"][profile_name]
    kwargs = profile["kwargs"]
    planner = DiffusionPlanner(**_materialize_kwargs(kwargs))
    inputs = _synthetic_inputs(planner, kwargs)

    assert list(inputs["z_ar"].shape) == [2, 6, 8]
    assert list(inputs["gt_trajectory"].shape) == [2, 3, kwargs["traj_dim"]]
    assert inputs["inference_noise"].shape[2] == kwargs["num_poses"]
    if profile_name == "legacy_single_6d":
        assert list(inputs["gt_trajectory"].shape) == [2, 3, 6]

    planner.train()
    train_output = planner(
        inputs["z_ar"],
        inputs["status_feature"],
        z_context=inputs["z_context"],
        z_observed=inputs["z_observed"],
        action_history=inputs["action_history"],
        gt_trajectory=inputs["gt_trajectory"],
        anchor_state=inputs["anchor_state"],
    )
    train_fields = _output_fields(train_output)
    assert train_fields == profile["forward_contracts"]["train"]["fields"]

    planner.eval()
    with torch.no_grad():
        eval_output = planner(
            inputs["z_ar"],
            inputs["status_feature"],
            z_context=inputs["z_context"],
            z_observed=inputs["z_observed"],
            action_history=inputs["action_history"],
            anchor_state=inputs["anchor_state"],
            inference_noise=inputs["inference_noise"],
        )
    eval_fields = _output_fields(eval_output)
    assert eval_fields == profile["forward_contracts"]["eval"]["fields"]

    train_by_name = {field["name"]: field for field in train_fields}
    eval_by_name = {field["name"]: field for field in eval_fields}
    output_modes = kwargs.get("num_modes", 1)
    if output_modes > 1:
        assert train_by_name["winner_idx"] == {
            "name": "winner_idx",
            "shape": [2],
            "dtype": "torch.int64",
            "is_scalar": False,
        }
        assert train_by_name["winner_traj_3d"] == {
            "name": "winner_traj_3d",
            "shape": [2, 3, 3],
            "dtype": "torch.float32",
            "is_scalar": False,
        }
        for scalar_name in ("awta_temperature", "cls_sample_valid_ratio"):
            field = train_by_name[scalar_name]
            assert field["shape"] == []
            assert field["is_scalar"] is True
            dtype = getattr(torch, field["dtype"].removeprefix("torch."))
            assert torch.empty((), dtype=dtype).is_floating_point()
        assert eval_by_name["trajectories"]["shape"] == [2, output_modes, 3, 3]
        assert eval_by_name["confidences"]["shape"] == [2, output_modes]
        assert inputs["inference_noise"].shape[1] == output_modes
    else:
        expected_samples = kwargs["num_samples"]
        assert eval_by_name["trajectories"]["shape"] == [2, expected_samples, 3, 3]
        assert eval_by_name["confidences"]["shape"] == [2, expected_samples]
        assert inputs["inference_noise"].shape[1] == expected_samples


def test_invalid_constructor_cases_raise_recorded_exception_classes() -> None:
    contract = _load_contract()
    assert tuple(contract["invalid_cases"]) == INVALID_CASE_NAMES
    for case in contract["invalid_cases"].values():
        exception_class = None
        try:
            DiffusionPlanner(**_materialize_kwargs(case["kwargs"]))
        except Exception as exc:  # noqa: BLE001 - compare only the black-box exception class
            exception_class = type(exc).__name__
        assert exception_class == case["exception_class"]


class _SamplerModel(torch.nn.Module):
    model_type = "noise"

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs):
        del t, kwargs
        return None, torch.zeros_like(x)


def test_sampler_two_item_return_contract() -> None:
    contract = _load_contract()["sampler_contract"]
    result = dpm_sampler(_SamplerModel(), torch.zeros(2, 3, 4), diffusion_steps=3)
    assert isinstance(result, tuple)
    assert len(result) == 2
    classification, sampled = result
    fields = [
        _optional_tensor_schema("classification", classification, include_device=True),
        _optional_tensor_schema("sampled", sampled, include_device=True),
    ]
    assert fields == contract["fields"]
    assert fields[0]["is_none"] is True
    assert fields[1]["shape"] == [2, 3, 4]
    assert fields[1]["dtype"] == "torch.float32"
    assert fields[1]["device"] == "cpu"


def _method_output_schema(names: tuple[str, ...], values: tuple) -> list[dict]:
    return [_optional_tensor_schema(name, value) for name, value in zip(names, values)]


def test_vpsde_public_method_shape_dtype_contract() -> None:
    contract = _load_contract()["sde_contract"]
    sde = VPSDE_linear(beta_min=0.1, beta_max=20.0)
    x = torch.zeros(2, 3, 4)
    t = torch.zeros(2)
    drift, diffusion = sde.sde(x, t)
    mean, std = sde.marginal_prob(x, t)

    actual = {
        "T": sde.T,
        "has__beta_min": hasattr(sde, "_beta_min"),
        "has__beta_max": hasattr(sde, "_beta_max"),
        "is_torch_nn_module": isinstance(sde, torch.nn.Module),
        "has_state_dict": hasattr(sde, "state_dict"),
        "method_outputs": {
            "sde": _method_output_schema(("drift", "diffusion"), (drift, diffusion)),
            "marginal_prob": _method_output_schema(("mean", "std"), (mean, std)),
            "marginal_prob_std": _method_output_schema(("std",), (sde.marginal_prob_std(t),)),
            "diffusion_coeff": _method_output_schema(("diffusion",), (sde.diffusion_coeff(t),)),
        },
    }
    assert actual == contract
    assert sde.T == 1.0
    assert hasattr(sde, "_beta_min")
    assert hasattr(sde, "_beta_max")
    assert not isinstance(sde, torch.nn.Module)
    assert not hasattr(sde, "state_dict")
