"""Lock the ordered checkpoint schema of the characterized diffusion planners."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from app.vjepa_cowa_world_model.models.backbones.vjepa21.vision_transformer import VIT_EMBED_DIMS
from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner
from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.configs.resolution import load_resolved_training_params
from app.vjepa_cowa_world_model.training.model_factories.planner import init_planner

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


def _state_dict_schema(module: torch.nn.Module) -> list[dict]:
    return [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in module.state_dict().items()
    ]


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_profile_ordered_checkpoint_schema_and_strict_load(profile_name: str) -> None:
    contract = _load_contract()
    assert tuple(contract["profiles"]) == PROFILE_NAMES
    profile = contract["profiles"][profile_name]
    kwargs = _materialize_kwargs(profile["kwargs"])

    planner = DiffusionPlanner(**kwargs)
    assert _state_dict_schema(planner) == profile["state_dict"]

    clone = DiffusionPlanner(**_materialize_kwargs(profile["kwargs"]))
    clone.load_state_dict(planner.state_dict(), strict=True)
    assert _state_dict_schema(clone) == profile["state_dict"]


def test_formal_factory_ordered_checkpoint_schema_and_strict_load() -> None:
    contract = _load_contract()
    assert tuple(contract["factory_profiles"]) == ("formal_v2_e120_factory",)
    profile = contract["factory_profiles"]["formal_v2_e120_factory"]

    params = load_resolved_training_params(ROOT / profile["config_path"])
    config = parse_training_config(params)
    assert VIT_EMBED_DIMS[config.model.model_name] == profile["encoder_dim"] == 1024

    planner = init_planner(
        config,
        encoder_dim=profile["encoder_dim"],
        device=torch.device("cpu"),
    )
    assert planner is not None
    assert _state_dict_schema(planner) == profile["state_dict"]

    clone = init_planner(
        config,
        encoder_dim=profile["encoder_dim"],
        device=torch.device("cpu"),
    )
    assert clone is not None
    clone.load_state_dict(planner.state_dict(), strict=True)
    assert _state_dict_schema(clone) == profile["state_dict"]
