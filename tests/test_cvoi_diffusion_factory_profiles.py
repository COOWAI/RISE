"""Seven-stage CVoI integration contract for the real diffusion factory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from app.vjepa_cowa_world_model.models.diffusion_planner import DiffusionPlanner
from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.configs.resolution import load_resolved_training_params
from app.vjepa_cowa_world_model.training.model_factories.planner import init_planner

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/train/navsim/cvoi_manual_full"
FIXTURE = ROOT / "tests/fixtures/diffusion_planner_contract_v1.json"
NAVSIM_HORIZON = 4
CONFIG_NAMES = (
    "01_predictor_lewm_pure.yaml",
    "02_p0_uniform.yaml",
    "03_field_full.yaml",
    "04_calibration_full.yaml",
    "05_p1_full.yaml",
    "06_stop_full.yaml",
    "07_gate_full.yaml",
)
PLANNER_CONFIG_NAMES = CONFIG_NAMES[1:]
PLANNER_SCHEMA_KEYS = {
    "use_planner",
    "planner_type",
    "diff_hidden_dim",
    "diff_num_layers",
    "diff_num_heads",
    "diff_dropout",
    "diff_mlp_ratio",
    "diff_inference_steps",
    "diff_trajectory_token_mode",
    "diff_adaln_version",
    "diff_use_anchor_frame",
}


def _factory_profile() -> dict[str, object]:
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert contract["schema_version"] == 1
    assert tuple(contract["factory_profiles"]) == ("formal_v2_e120_factory",)
    return contract["factory_profiles"]["formal_v2_e120_factory"]


def _tiny_factory_state_keys() -> tuple[str, ...]:
    names = [entry["name"] for entry in _factory_profile()["state_dict"]]
    return tuple(
        name for name in names if not name.startswith("dit.blocks.") or int(name.split(".", maxsplit=3)[2]) == 0
    )


def _apply_tiny_profile(config: object) -> None:
    config.planner.diff_hidden_dim = 16
    config.planner.diff_num_layers = 1
    config.planner.diff_num_heads = 4
    config.planner.diff_dropout = 0.0
    config.planner.diff_mlp_ratio = 2.0
    config.planner.diff_inference_steps = 3


def test_cvoi_manual_full_yaml_inventory_is_exactly_the_seven_flat_profiles() -> None:
    assert tuple(path.name for path in sorted(CONFIG_ROOT.glob("*.yaml"))) == CONFIG_NAMES


def test_predictor_stage_retains_diffusion_schema_but_factory_stays_disabled() -> None:
    config_path = CONFIG_ROOT / CONFIG_NAMES[0]
    params = load_resolved_training_params(config_path)
    assert PLANNER_SCHEMA_KEYS <= set(params["planner"])
    config = parse_training_config(params)

    assert config.planner.use_planner is False
    assert config.planner.planner_type == "diffusion"
    assert config.planner.diff_trajectory_token_mode == "per_pose_token"
    assert config.planner.diff_adaln_version == "v2"
    assert config.planner.diff_use_anchor_frame is True
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3401)
        planner = init_planner(
            config,
            encoder_dim=8,
            device=torch.device("cpu"),
            num_poses=NAVSIM_HORIZON,
            tokens_per_frame_override=2,
        )
    assert planner is None


@pytest.mark.parametrize("config_name", PLANNER_CONFIG_NAMES)
def test_each_cvoi_planner_stage_builds_the_formal_v2_tiny_factory_schema(config_name: str) -> None:
    config_path = CONFIG_ROOT / config_name
    params = load_resolved_training_params(config_path)
    config = parse_training_config(params)
    assert config.cvoi.max_horizon == NAVSIM_HORIZON
    assert config.cvoi.rollout_horizons == [0, 1, 2, 3, 4]
    _apply_tiny_profile(config)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3402)
        planner = init_planner(
            config,
            encoder_dim=8,
            device=torch.device("cpu"),
            num_poses=NAVSIM_HORIZON,
            tokens_per_frame_override=2,
        )

    assert isinstance(planner, DiffusionPlanner)
    assert planner.trajectory_token_mode == "per_pose_token"
    assert planner.adaln_version == "v2"
    assert planner.use_anchor_frame is True
    assert planner.observed_token_mode == "concat_type_embed"
    assert tuple(planner.state_dict()) == _tiny_factory_state_keys()
