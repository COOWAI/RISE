# Copyright (c) 2026 RISE Contributors
# RISE provenance: independent-diffusion-v1
"""Validated adapter for the retained MIT DPM-Solver implementation."""

import math
from collections.abc import Mapping
from numbers import Real
from typing import Dict

import torch

from . import dpm_solver_pytorch as dpm

_OPTION_KEYS = {
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
_RESERVED_KEYS = {"schedule", "model_type", "model_kwargs", "algorithm_type", "steps"}
_SUPPORTED_MODEL_TYPES = {"noise", "x_start", "v", "score"}
_SUPPORTED_SAMPLE_DTYPES = {torch.float32, torch.float64}


def _copy_mapping(name, mapping):
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(mapping).__name__}")
    return dict(mapping)


def _validate_options(option_mappings):
    owners = {}
    for mapping_name, mapping in option_mappings.items():
        for key in mapping:
            if key in owners:
                raise ValueError(f"option {key!r} appears in multiple mappings: {owners[key]} and {mapping_name}")
            owners[key] = mapping_name

    for mapping_name, mapping in option_mappings.items():
        allowed = _OPTION_KEYS[mapping_name]
        for key in mapping:
            if key in _RESERVED_KEYS:
                raise ValueError(f"{mapping_name} cannot override reserved option {key!r}")
            if key not in allowed:
                owner = next((name for name, keys in _OPTION_KEYS.items() if key in keys), None)
                if owner is None:
                    raise ValueError(f"unknown option {key!r} in {mapping_name}")
                raise ValueError(f"option {key!r} belongs to {owner}, not {mapping_name}")


def _validate_schedule_options(schedule_options):
    beta_values = {}
    for key in ("continuous_beta_0", "continuous_beta_1"):
        value = schedule_options[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{key} must be a real numeric value, got {type(value).__name__}")
        beta_values[key] = float(value)
    beta_min = beta_values["continuous_beta_0"]
    beta_max = beta_values["continuous_beta_1"]
    if not math.isfinite(beta_min) or not math.isfinite(beta_max):
        raise ValueError("continuous beta values must be finite")
    if beta_min <= 0.0 or beta_max <= 0.0:
        raise ValueError("continuous beta values must be positive")
    if beta_max <= beta_min:
        raise ValueError("continuous_beta_1 must be greater than continuous_beta_0")
    schedule_options.update(beta_values)


def _validate_sample_options(sample_options, diffusion_steps):
    if sample_options["method"] != "multistep":
        raise ValueError("sample method must be 'multistep'")
    return_intermediate = sample_options.get("return_intermediate", False)
    if not isinstance(return_intermediate, bool):
        raise TypeError("return_intermediate must be a bool")
    if return_intermediate:
        raise ValueError("return_intermediate=True is incompatible with the adapter return protocol")
    order = sample_options["order"]
    if not isinstance(order, int) or isinstance(order, bool):
        raise TypeError("order must be an integer")
    if order not in {1, 2, 3}:
        raise ValueError("order must be one of 1, 2, or 3")
    if order > diffusion_steps:
        raise ValueError("order must not exceed diffusion_steps")


def dpm_sampler(
    model: torch.nn.Module,
    x_T,
    other_model_params: Dict = {},
    diffusion_steps=2,
    noise_schedule_params: Dict = {},
    model_wrapper_params: Dict = {},
    dpm_solver_params: Dict = {},
    sample_params: Dict = {},
):
    """Sample ``x_T`` with DPM-Solver++ while keeping caller options isolated."""
    if not isinstance(x_T, torch.Tensor):
        raise TypeError(f"x_T must be a torch.Tensor, got {type(x_T).__name__}")
    if not x_T.is_floating_point():
        raise TypeError(f"x_T must have a floating dtype, got {x_T.dtype}")
    if x_T.dtype not in _SUPPORTED_SAMPLE_DTYPES:
        raise TypeError(f"x_T dtype must be float32 or float64 for the retained solver, got {x_T.dtype}")
    if x_T.ndim < 2:
        raise ValueError(f"x_T rank/ndim must be at least 2, got {x_T.ndim}")
    if not isinstance(diffusion_steps, int) or isinstance(diffusion_steps, bool):
        raise TypeError(f"diffusion_steps must be an integer, got {type(diffusion_steps).__name__}")
    if diffusion_steps < 2:
        raise ValueError(f"diffusion_steps must be at least 2, got {diffusion_steps}")

    model_type = getattr(model, "model_type", None)
    if model_type not in _SUPPORTED_MODEL_TYPES:
        raise ValueError(f"model.model_type must be one of {sorted(_SUPPORTED_MODEL_TYPES)}, got {model_type!r}")

    copied_other = _copy_mapping("other_model_params", other_model_params)
    copied_options = {
        "noise_schedule_params": _copy_mapping("noise_schedule_params", noise_schedule_params),
        "model_wrapper_params": _copy_mapping("model_wrapper_params", model_wrapper_params),
        "dpm_solver_params": _copy_mapping("dpm_solver_params", dpm_solver_params),
        "sample_params": _copy_mapping("sample_params", sample_params),
    }
    _validate_options(copied_options)

    schedule_options = {"continuous_beta_0": 0.1, "continuous_beta_1": 20.0}
    schedule_options.update(copied_options["noise_schedule_params"])

    effective_sample_options = {
        "order": 2,
        "skip_type": "logSNR",
        "method": "multistep",
        "denoise_to_zero": True,
    }
    effective_sample_options.update(copied_options["sample_params"])
    _validate_schedule_options(schedule_options)
    _validate_sample_options(effective_sample_options, diffusion_steps)

    noise_schedule = dpm.NoiseScheduleVP(schedule="linear", **schedule_options)
    model_fn = dpm.model_wrapper(
        model,
        noise_schedule,
        model_type=model_type,
        model_kwargs=copied_other,
        **copied_options["model_wrapper_params"],
    )
    solver = dpm.DPM_Solver(
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        **copied_options["dpm_solver_params"],
    )
    return solver.sample(x_T, steps=diffusion_steps, **effective_sample_options)
