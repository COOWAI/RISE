"""Contract tests for the thin adapter around the retained MIT DPM-Solver."""

from __future__ import annotations

import copy
import pickle
import re
from typing import Any

import pytest
import torch

from app.vjepa_cowa_world_model.diffusion_utils import dpm_solver_pytorch as dpm
from app.vjepa_cowa_world_model.diffusion_utils import sampling
from app.vjepa_cowa_world_model.diffusion_utils.sampling import dpm_sampler

OPTION_KEYS = {
    "noise_schedule_params": {"continuous_beta_0", "continuous_beta_1", "dtype"},
    "model_wrapper_params": {
        "guidance_type",
        "condition",
        "unconditional_condition",
        "guidance_scale",
        "classifier_fn",
        "classifier_kwargs",
    },
    "dpm_solver_params": {
        "correcting_x0_fn",
        "correcting_xt_fn",
        "thresholding_max_val",
        "dynamic_thresholding_ratio",
    },
    "sample_params": {
        "order",
        "skip_type",
        "method",
        "denoise_to_zero",
        "t_start",
        "t_end",
        "lower_order_final",
        "solver_type",
        "atol",
        "rtol",
        "return_intermediate",
    },
}
RESERVED_KEYS = {"schedule", "model_type", "model_kwargs", "algorithm_type", "steps"}
_UNSET = object()


class _NoiseModel(torch.nn.Module):
    model_type = "noise"

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del t, kwargs
        return torch.zeros_like(x)


class _TrackedNoiseModel(torch.nn.Module):
    model_type = "noise"

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        del kwargs
        self.calls.append(
            {
                "x_shape": tuple(x.shape),
                "t_shape": tuple(t.shape),
                "t_dtype": t.dtype,
                "t_device": t.device,
                "t_is_floating": t.is_floating_point(),
            }
        )
        classification = torch.stack((x.flatten(start_dim=1).mean(dim=1), t.to(dtype=x.dtype)), dim=1)
        prediction = torch.zeros_like(x)
        return classification, prediction


class _MissingModelType(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        del t
        return torch.zeros_like(x)


class _UnsupportedModelType(_NoiseModel):
    model_type = "definitely_not_a_diffusion_parameterization"


def _classifier_fn(*args: Any, **kwargs: Any) -> None:
    del args, kwargs


def _correcting_xt_fn(x: torch.Tensor, t: torch.Tensor, step: int) -> torch.Tensor:
    del t, step
    return x


def _install_solver_spies(
    monkeypatch: pytest.MonkeyPatch,
    sampled: torch.Tensor,
    classification: object,
) -> dict[str, Any]:
    calls: dict[str, Any] = {}
    noise_schedule = object()
    wrapped_model = object()

    def noise_schedule_spy(
        schedule: Any = _UNSET,
        betas: Any = _UNSET,
        alphas_cumprod: Any = _UNSET,
        continuous_beta_0: Any = _UNSET,
        continuous_beta_1: Any = _UNSET,
        dtype: Any = _UNSET,
    ) -> object:
        del betas, alphas_cumprod
        calls["noise_schedule"] = {
            "schedule": schedule,
            "continuous_beta_0": continuous_beta_0,
            "continuous_beta_1": continuous_beta_1,
            "dtype": dtype,
        }
        return noise_schedule

    def model_wrapper_spy(
        model: torch.nn.Module,
        schedule: object,
        model_type: Any = _UNSET,
        model_kwargs: Any = _UNSET,
        guidance_type: Any = _UNSET,
        condition: Any = _UNSET,
        unconditional_condition: Any = _UNSET,
        guidance_scale: Any = _UNSET,
        classifier_fn: Any = _UNSET,
        classifier_kwargs: Any = _UNSET,
    ) -> object:
        calls["model_wrapper"] = {
            "model": model,
            "noise_schedule": schedule,
            "model_type": model_type,
            "model_kwargs": model_kwargs,
            "guidance_type": guidance_type,
            "condition": condition,
            "unconditional_condition": unconditional_condition,
            "guidance_scale": guidance_scale,
            "classifier_fn": classifier_fn,
            "classifier_kwargs": classifier_kwargs,
        }
        return wrapped_model

    class SolverSpy:
        def __init__(
            self,
            model_fn: object,
            schedule: object,
            algorithm_type: Any = _UNSET,
            correcting_x0_fn: Any = _UNSET,
            correcting_xt_fn: Any = _UNSET,
            thresholding_max_val: Any = _UNSET,
            dynamic_thresholding_ratio: Any = _UNSET,
        ) -> None:
            calls["solver"] = {
                "model_fn": model_fn,
                "noise_schedule": schedule,
                "algorithm_type": algorithm_type,
                "correcting_x0_fn": correcting_x0_fn,
                "correcting_xt_fn": correcting_xt_fn,
                "thresholding_max_val": thresholding_max_val,
                "dynamic_thresholding_ratio": dynamic_thresholding_ratio,
            }

        def sample(
            self,
            x: torch.Tensor,
            steps: Any = _UNSET,
            t_start: Any = _UNSET,
            t_end: Any = _UNSET,
            order: Any = _UNSET,
            skip_type: Any = _UNSET,
            method: Any = _UNSET,
            lower_order_final: Any = _UNSET,
            denoise_to_zero: Any = _UNSET,
            solver_type: Any = _UNSET,
            atol: Any = _UNSET,
            rtol: Any = _UNSET,
            return_intermediate: Any = _UNSET,
        ) -> tuple[object, torch.Tensor]:
            calls["sample"] = {
                "x": x,
                "steps": steps,
                "order": order,
                "skip_type": skip_type,
                "method": method,
                "denoise_to_zero": denoise_to_zero,
                "t_start": t_start,
                "t_end": t_end,
                "lower_order_final": lower_order_final,
                "solver_type": solver_type,
                "atol": atol,
                "rtol": rtol,
                "return_intermediate": return_intermediate,
            }
            return classification, sampled

    monkeypatch.setattr(dpm, "NoiseScheduleVP", noise_schedule_spy)
    monkeypatch.setattr(dpm, "model_wrapper", model_wrapper_spy)
    monkeypatch.setattr(dpm, "DPM_Solver", SolverSpy)
    return calls


def _install_solver_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    def noise_schedule_spy(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        return object()

    def model_wrapper_spy(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        return object()

    def solver_bomb(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("retained DPM solver was reached before adapter validation")

    monkeypatch.setattr(dpm, "NoiseScheduleVP", noise_schedule_spy)
    monkeypatch.setattr(dpm, "model_wrapper", model_wrapper_spy)
    monkeypatch.setattr(dpm, "DPM_Solver", solver_bomb)


def _install_schedule_bomb(monkeypatch: pytest.MonkeyPatch) -> None:
    def schedule_bomb(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("noise schedule construction was reached before adapter validation")

    monkeypatch.setattr(dpm, "NoiseScheduleVP", schedule_bomb)


def _call_with_mapping(mapping_name: str, mapping: dict[str, Any]) -> None:
    mappings = {name: {} for name in OPTION_KEYS}
    mappings[mapping_name] = mapping
    sampling.dpm_sampler(
        _NoiseModel(),
        torch.ones(2, 3),
        noise_schedule_params=mappings["noise_schedule_params"],
        model_wrapper_params=mappings["model_wrapper_params"],
        dpm_solver_params=mappings["dpm_solver_params"],
        sample_params=mappings["sample_params"],
    )


def test_sampler_passes_owned_defaults_and_preserves_solver_return_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_t = torch.linspace(-1.0, 1.0, 24, dtype=torch.float32).reshape(2, 3, 4)
    sampled = x_t.clone()
    classification = object()
    calls = _install_solver_spies(monkeypatch, sampled, classification)
    model = _NoiseModel()
    other_model_params = {"context": "kept", "nested": {"items": [1, 2]}}
    snapshot = copy.deepcopy(other_model_params)

    returned_classification, returned_sampled = dpm_sampler(model, x_t, other_model_params=other_model_params)

    assert other_model_params == snapshot
    assert calls["noise_schedule"]["schedule"] == "linear"
    assert calls["noise_schedule"]["continuous_beta_0"] == 0.1
    assert calls["noise_schedule"]["continuous_beta_1"] == 20.0
    assert calls["model_wrapper"]["model"] is model
    assert calls["model_wrapper"]["model_type"] == model.model_type == "noise"
    assert calls["model_wrapper"]["model_kwargs"] == other_model_params
    assert calls["model_wrapper"]["model_kwargs"] is not other_model_params
    assert calls["model_wrapper"]["model_kwargs"]["nested"] is other_model_params["nested"]
    assert calls["solver"]["algorithm_type"] == "dpmsolver++"
    assert calls["sample"]["x"] is x_t
    assert calls["sample"]["steps"] == 2
    assert calls["sample"]["order"] == 2
    assert calls["sample"]["skip_type"] == "logSNR"
    assert calls["sample"]["method"] == "multistep"
    assert calls["sample"]["denoise_to_zero"] is True
    assert returned_classification is classification
    assert returned_sampled is sampled
    assert returned_sampled.shape == x_t.shape
    assert returned_sampled.dtype == x_t.dtype
    assert returned_sampled.device == x_t.device


def test_sampler_default_multistep_executes_the_real_retained_solver_deterministically() -> None:
    model = _TrackedNoiseModel()
    x_t = torch.tensor([[-0.8, -0.2], [0.3, 0.9]], dtype=torch.float32)
    original_x_t = x_t.clone()
    first_input = x_t.clone()
    second_input = x_t.clone()

    first_classification, first_sampled = dpm_sampler(model, first_input, diffusion_steps=2)
    first_calls = tuple(model.calls)
    model.calls.clear()
    second_classification, second_sampled = dpm_sampler(model, second_input, diffusion_steps=2)
    second_calls = tuple(model.calls)

    assert first_calls
    assert first_calls == second_calls
    for call in first_calls:
        assert call["x_shape"] == tuple(x_t.shape)
        assert call["t_shape"] == (x_t.shape[0],)
        assert call["t_is_floating"] is True
        assert torch.empty((), dtype=call["t_dtype"]).is_floating_point()
        assert call["t_device"] == x_t.device

    expected_classification_schema = ((x_t.shape[0], 2), x_t.dtype, x_t.device)
    first_classification_schema = (
        tuple(first_classification.shape),
        first_classification.dtype,
        first_classification.device,
    )
    second_classification_schema = (
        tuple(second_classification.shape),
        second_classification.dtype,
        second_classification.device,
    )
    assert first_classification_schema == second_classification_schema == expected_classification_schema
    torch.testing.assert_close(first_classification, second_classification, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_sampled, second_sampled, rtol=0.0, atol=0.0)
    for sampled in (first_sampled, second_sampled):
        assert sampled.shape == x_t.shape
        assert sampled.dtype == x_t.dtype
        assert sampled.device == x_t.device
        assert torch.isfinite(sampled).all()
    torch.testing.assert_close(x_t, original_x_t, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first_input, original_x_t, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second_input, original_x_t, rtol=0.0, atol=0.0)

    different_x_t = torch.tensor([[1.2, -0.7], [0.1, -1.4]], dtype=x_t.dtype)
    _, different_sampled = dpm_sampler(model, different_x_t.clone(), diffusion_steps=2)
    assert different_sampled.shape == x_t.shape
    assert torch.isfinite(different_sampled).all()
    assert not torch.equal(first_sampled, different_sampled)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_sampler_rejects_low_precision_before_schedule_construction(
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
) -> None:
    _install_schedule_bomb(monkeypatch)
    x_t = torch.tensor([[-0.8, -0.2], [0.3, 0.9]], dtype=dtype, device="cpu")

    with pytest.raises(TypeError, match="dtype|precision|float16|bfloat16|supported"):
        dpm_sampler(_TrackedNoiseModel(), x_t, diffusion_steps=2)


def test_sampler_real_multistep_preserves_float64_dtype_and_device() -> None:
    model = _TrackedNoiseModel()
    x_t = torch.tensor([[-0.8, -0.2], [0.3, 0.9]], dtype=torch.float64, device="cpu")

    classification, sampled = dpm_sampler(model, x_t, diffusion_steps=2)

    assert sampled.shape == x_t.shape
    assert sampled.dtype == x_t.dtype
    assert sampled.device == x_t.device
    assert torch.isfinite(sampled).all()
    assert classification.dtype == x_t.dtype
    assert classification.device == x_t.device
    assert torch.isfinite(classification).all()


def test_sampler_routes_every_override_without_mutating_caller_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_t = torch.linspace(-2.0, 2.0, 24, dtype=torch.float32).reshape(2, 3, 4)
    sampled = x_t.clone()
    calls = _install_solver_spies(monkeypatch, sampled, object())
    other_model_params = {"context": "value", "nested": {"items": [3, 4]}}
    noise_schedule_params = {
        "continuous_beta_0": 0.2,
        "continuous_beta_1": 12.0,
        "dtype": torch.float64,
    }
    model_wrapper_params = {
        "guidance_type": "classifier",
        "condition": ("conditional",),
        "unconditional_condition": ("unconditional",),
        "guidance_scale": 2.5,
        "classifier_fn": _classifier_fn,
        "classifier_kwargs": {"label": 7},
    }
    dpm_solver_params = {
        "correcting_x0_fn": "dynamic_thresholding",
        "correcting_xt_fn": _correcting_xt_fn,
        "thresholding_max_val": 1.25,
        "dynamic_thresholding_ratio": 0.9,
    }
    sample_params = {
        "order": 3,
        "skip_type": "time_uniform",
        "method": "multistep",
        "denoise_to_zero": False,
        "t_start": 0.9,
        "t_end": 0.02,
        "lower_order_final": False,
        "solver_type": "taylor",
        "atol": 0.1,
        "rtol": 0.2,
        "return_intermediate": False,
    }
    mappings = (
        other_model_params,
        noise_schedule_params,
        model_wrapper_params,
        dpm_solver_params,
        sample_params,
    )
    equality_snapshots = copy.deepcopy(mappings)
    byte_snapshot = pickle.dumps(mappings)

    dpm_sampler(
        _NoiseModel(),
        x_t,
        other_model_params=other_model_params,
        diffusion_steps=5,
        noise_schedule_params=noise_schedule_params,
        model_wrapper_params=model_wrapper_params,
        dpm_solver_params=dpm_solver_params,
        sample_params=sample_params,
    )

    assert mappings == equality_snapshots
    assert pickle.dumps(mappings) == byte_snapshot
    assert calls["noise_schedule"] == {"schedule": "linear", **noise_schedule_params}
    assert {key: calls["model_wrapper"][key] for key in model_wrapper_params} == model_wrapper_params
    assert calls["model_wrapper"]["model_kwargs"] == other_model_params
    assert calls["model_wrapper"]["model_kwargs"] is not other_model_params
    assert calls["model_wrapper"]["model_kwargs"]["nested"] is other_model_params["nested"]
    assert {key: calls["solver"][key] for key in dpm_solver_params} == dpm_solver_params
    assert calls["solver"]["algorithm_type"] == "dpmsolver++"
    assert {key: calls["sample"][key] for key in sample_params} == sample_params
    assert calls["sample"]["steps"] == 5


@pytest.mark.parametrize("model", (_MissingModelType(), _UnsupportedModelType()))
def test_sampler_requires_a_supported_model_type_before_solver_use(
    monkeypatch: pytest.MonkeyPatch,
    model: torch.nn.Module,
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(ValueError, match="model_type"):
        dpm_sampler(model, torch.ones(2, 3))


@pytest.mark.parametrize(
    ("x_t", "error_type", "message"),
    (
        (torch.ones(2, 3, dtype=torch.int64), TypeError, "floating"),
        (torch.ones(5, dtype=torch.float32), ValueError, "rank|ndim"),
    ),
)
def test_sampler_rejects_invalid_initial_tensor_before_solver_use(
    monkeypatch: pytest.MonkeyPatch,
    x_t: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(error_type, match=message):
        dpm_sampler(_NoiseModel(), x_t)


@pytest.mark.parametrize("diffusion_steps", (-1, 0, 1))
def test_sampler_requires_at_least_two_diffusion_steps(
    monkeypatch: pytest.MonkeyPatch,
    diffusion_steps: int,
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(ValueError, match="diffusion_steps"):
        dpm_sampler(_NoiseModel(), torch.ones(2, 3), diffusion_steps=diffusion_steps)


@pytest.mark.parametrize("method", ("singlestep", "singlestep_fixed", "adaptive"))
def test_sampler_rejects_tuple_incompatible_methods_before_solver_use(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(ValueError, match="method|multistep"):
        dpm_sampler(_NoiseModel(), torch.ones(2, 3), sample_params={"method": method})


def test_sampler_rejects_intermediate_return_protocol_before_solver_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(ValueError, match="return_intermediate"):
        dpm_sampler(_NoiseModel(), torch.ones(2, 3), sample_params={"return_intermediate": True})


@pytest.mark.parametrize(
    ("order", "diffusion_steps", "error_type"),
    (
        (True, 2, TypeError),
        (2.0, 2, TypeError),
        ("2", 2, TypeError),
        (0, 2, ValueError),
        (-1, 2, ValueError),
        (3, 2, ValueError),
        (4, 4, ValueError),
    ),
)
def test_sampler_rejects_invalid_or_unachievable_order_before_solver_use(
    monkeypatch: pytest.MonkeyPatch,
    order: object,
    diffusion_steps: int,
    error_type: type[Exception],
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(error_type, match="order|integer|diffusion_steps"):
        dpm_sampler(
            _NoiseModel(),
            torch.ones(2, 3),
            diffusion_steps=diffusion_steps,
            sample_params={"order": order},
        )


@pytest.mark.parametrize(
    "noise_schedule_params",
    (
        {"continuous_beta_0": 0.0},
        {"continuous_beta_0": -0.1},
        {"continuous_beta_0": 20.0},
        {"continuous_beta_1": 0.0},
        {"continuous_beta_1": 0.05},
        {"continuous_beta_0": float("nan")},
        {"continuous_beta_0": float("inf")},
        {"continuous_beta_1": float("nan")},
        {"continuous_beta_1": float("inf")},
    ),
)
def test_sampler_rejects_invalid_merged_linear_beta_range_before_solver_use(
    monkeypatch: pytest.MonkeyPatch,
    noise_schedule_params: dict[str, float],
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(ValueError, match="beta|finite|positive"):
        dpm_sampler(
            _NoiseModel(),
            torch.ones(2, 3),
            noise_schedule_params=noise_schedule_params,
        )


@pytest.mark.parametrize("beta_value", (True, "0.1"))
def test_sampler_rejects_malformed_beta_types_before_schedule_construction(
    monkeypatch: pytest.MonkeyPatch,
    beta_value: object,
) -> None:
    _install_schedule_bomb(monkeypatch)

    with pytest.raises(TypeError, match="beta|type|real|numeric"):
        dpm_sampler(
            _NoiseModel(),
            torch.ones(2, 3),
            noise_schedule_params={"continuous_beta_0": beta_value},
        )


@pytest.mark.parametrize("mapping_name", tuple(OPTION_KEYS))
def test_each_sampler_mapping_rejects_unknown_keys(
    monkeypatch: pytest.MonkeyPatch,
    mapping_name: str,
) -> None:
    _install_solver_bomb(monkeypatch)
    key = "not_an_adapter_option"

    with pytest.raises(ValueError, match=key):
        _call_with_mapping(mapping_name, {key: object()})


@pytest.mark.parametrize(
    ("mapping_name", "key"),
    tuple((mapping_name, key) for mapping_name in OPTION_KEYS for key in sorted(RESERVED_KEYS)),
)
def test_each_sampler_mapping_rejects_every_reserved_key(
    monkeypatch: pytest.MonkeyPatch,
    mapping_name: str,
    key: str,
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(ValueError, match=re.escape(key)):
        _call_with_mapping(mapping_name, {key: object()})


@pytest.mark.parametrize(
    ("mapping_name", "key"),
    tuple(
        (mapping_name, key)
        for mapping_name in OPTION_KEYS
        for owner, owned_keys in OPTION_KEYS.items()
        if owner != mapping_name
        for key in sorted(owned_keys)
    ),
)
def test_each_sampler_mapping_rejects_keys_owned_by_another_mapping(
    monkeypatch: pytest.MonkeyPatch,
    mapping_name: str,
    key: str,
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(ValueError, match=re.escape(key)):
        _call_with_mapping(mapping_name, {key: object()})


def test_sampler_rejects_cross_mapping_duplicates_before_ownership_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_solver_bomb(monkeypatch)

    with pytest.raises(ValueError, match="duplicate|multiple"):
        dpm_sampler(
            _NoiseModel(),
            torch.ones(2, 3),
            noise_schedule_params={"steps": 2},
            sample_params={"steps": 2},
        )
