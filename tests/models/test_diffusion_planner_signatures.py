"""Lock the callable signatures and abstract SDE surface of diffusion planning."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch

from app.vjepa_cowa_world_model.diffusion_utils.sampling import dpm_sampler
from app.vjepa_cowa_world_model.diffusion_utils.sde import SDE, VPSDE_linear
from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/diffusion_planner_contract_v1.json"
PLANNER_SIGNATURE_NAMES = (
    "__init__",
    "forward",
    "set_awta_temperature",
    "convert_3d_to_6d",
    "convert_3d_to_nd",
    "init_interleaved_inference_state",
    "advance_interleaved_inference",
    "finalize_interleaved_inference",
    "_prepare_context",
    "_prepare_status",
    "_get_anchor",
    "_resolve_inference_noise",
    "_correct_anchor_xt",
    "_convert_6d_to_3d",
    "_convert_nd_to_3d",
    "_xy_regression_loss_per_mode",
    "_training_forward",
    "_training_forward_independent",
    "_training_forward_multimodal",
    "_inference_forward",
    "_inference_forward_independent",
    "_inference_forward_multimodal",
    "_run_interleaved_solver_step",
    "_interleaved_target_sampling_steps",
)
SDE_SIGNATURE_NAMES = (
    "__init__",
    "sde",
    "marginal_prob",
    "marginal_prob_std",
    "diffusion_coeff",
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


def test_all_recorded_public_and_protected_signatures_match() -> None:
    contract = _load_contract()
    expected_names = {
        *(f"DiffusionPlanner.{name}" for name in PLANNER_SIGNATURE_NAMES),
        *(f"SDE.{name}" for name in SDE_SIGNATURE_NAMES),
        "VPSDE_linear.__init__",
        "dpm_sampler",
    }
    assert set(contract["signatures"]) == expected_names

    live_callables = {
        **{f"DiffusionPlanner.{name}": getattr(DiffusionPlanner, name) for name in PLANNER_SIGNATURE_NAMES},
        **{f"SDE.{name}": getattr(SDE, name) for name in SDE_SIGNATURE_NAMES},
        "VPSDE_linear.__init__": VPSDE_linear.__init__,
        "dpm_sampler": dpm_sampler,
    }
    for name, expected in contract["signatures"].items():
        assert str(inspect.signature(live_callables[name])) == expected


def test_nested_dit_forward_signature_matches_for_every_profile() -> None:
    contract = _load_contract()
    for profile in contract["profiles"].values():
        planner = DiffusionPlanner(**_materialize_kwargs(profile["kwargs"]))
        assert str(inspect.signature(type(planner.dit).forward)) == profile["dit_forward_contract"]["signature"]


def test_sde_abstract_surface_is_exact_and_base_is_not_instantiable() -> None:
    contract = _load_contract()
    assert contract["abstract_methods"] == [
        "T",
        "diffusion_coeff",
        "marginal_prob",
        "marginal_prob_std",
        "sde",
    ]
    assert sorted(SDE.__abstractmethods__) == contract["abstract_methods"]
    with pytest.raises(TypeError):
        SDE()
