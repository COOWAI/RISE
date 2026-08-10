"""Tests for the NavSim training constants retained by the manual CVoI chain."""

import pytest

from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_HORIZONS,
    FORMAL_V2_NAVSIM_MAX_AGENTS,
    FORMAL_V2_NAVSIM_MAX_HORIZON,
    FORMAL_V2_NAVSIM_METRIC_PROTOCOL_IDS,
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P0_POLICIES,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
    get_formal_v2_navsim_metric_protocol,
)


def test_navsim_protocol_retains_training_semantics_without_a_dag() -> None:
    assert FORMAL_V2_NAVSIM_MAX_AGENTS == 1024
    assert FORMAL_V2_NAVSIM_MAX_HORIZON == 4
    assert FORMAL_V2_NAVSIM_HORIZONS == (0, 1, 2, 3, 4)
    assert FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS == (10, 20, 30, 35, 40, 45, 50)
    assert FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS == tuple(range(5, 81, 5))
    assert dict(FORMAL_V2_NAVSIM_P0_POLICIES) == {
        "uniform": (0.2, 0.2, 0.2, 0.2, 0.2),
        "extremes": (0.5, 0.0, 0.0, 0.0, 0.5),
        "short_heavy": (0.225, 0.225, 0.225, 0.225, 0.1),
        "no_full": (0.25, 0.25, 0.25, 0.25, 0.0),
    }


def test_navsim_protocol_retains_only_the_direct_v2_metric_authority() -> None:
    assert FORMAL_V2_NAVSIM_METRIC_PROTOCOL_IDS == ("epdms_v2_one_stage_navtest",)
    assert get_formal_v2_navsim_metric_protocol("epdms_v2_one_stage_navtest") == {
        "protocol_id": "epdms_v2_one_stage_navtest",
        "score_family": "epdms",
        "aggregate_key": "average_all_frames",
        "authority_script": "scripts/eval_navsim/eval_navsim_v2_pdms.sh",
        "authority_runner": "run_pdm_score_one_stage.py",
        "scorer_relative_path": "navsim/planning/script/run_pdm_score_one_stage.py",
        "devkit_checkout": "/path/to/navsim-devkit",
        "devkit_repository_root": "/path/to/navsim-devkit",
        "devkit_revision": "937cefc1b116f930990abea1c54185308a96029f",
    }
    with pytest.raises(ValueError, match="unknown Formal-v2 NavSim metric protocol"):
        get_formal_v2_navsim_metric_protocol("pdms_v1_navtest")
    with pytest.raises(ValueError, match="unknown Formal-v2 NavSim metric protocol"):
        get_formal_v2_navsim_metric_protocol("unknown")
