"""Unknown-key drift gate for the seven retained Full training configs."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from app.vjepa_cowa_world_model.training.configs.parse import parse_training_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs/train/navsim/cvoi_manual_full"
EXPECTED_CONFIGS = {
    "01_predictor_lewm_pure.yaml",
    "02_p0_uniform.yaml",
    "03_field_full.yaml",
    "04_calibration_full.yaml",
    "05_p1_full.yaml",
    "06_stop_full.yaml",
    "07_gate_full.yaml",
}


def test_all_retained_full_configs_pass_the_fail_loud_parser() -> None:
    paths = tuple(sorted(CONFIG_ROOT.glob("*.yaml")))
    assert {path.name for path in paths} == EXPECTED_CONFIGS

    for path in paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), f"{path} must contain one mapping"
        parse_training_config(copy.deepcopy(payload))
