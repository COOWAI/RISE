"""Tests for the retained direct NavSim V2 metric authority."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.vjepa_cowa_world_model.training import cvoi_navsim_protocols as protocols_module
from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import V2_PROTOCOL_ID, get_cvoi_navsim_metric_protocol


def test_only_direct_v2_protocol_authority_is_public() -> None:
    assert V2_PROTOCOL_ID == "epdms_v2_one_stage_navtest"
    for removed_name in (
        "V1_PROTOCOL_ID",
        "CVOI_NAVSIM_METRIC_PROTOCOL_IDS",
        "CvoiNavSimOfficialConfig",
        "CvoiNavSimProtocolRuntimeConfig",
        "CvoiNavSimVerifiedDevkit",
        "build_cvoi_navsim_explicit_devkit_git_commands",
        "verify_configured_devkit",
    ):
        assert removed_name not in protocols_module.__dict__


def test_v2_protocol_has_the_exact_direct_epdms_authority() -> None:
    protocol = get_cvoi_navsim_metric_protocol(V2_PROTOCOL_ID)

    assert protocol.protocol_id == V2_PROTOCOL_ID
    assert protocol.authority_script == Path("scripts/eval_navsim/eval_navsim_v2_pdms.sh")
    assert protocol.scorer_entrypoint == "navsim/planning/script/run_pdm_score_one_stage.py"
    assert protocol.split == "navtest"
    assert protocol.summary_token == "average_all_frames"
    assert protocol.cache_family == "navsim_v2"
    assert protocol.required_components == frozenset(
        {
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "driving_direction_compliance",
            "traffic_light_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "lane_keeping",
            "history_comfort",
            "two_frame_extended_comfort",
        }
    )
    assert protocol.nullable_components == frozenset({"two_frame_extended_comfort"})

    with pytest.raises(FrozenInstanceError):
        protocol.split = "navtrain"


@pytest.mark.parametrize("protocol_id", ["pdms_v1_navtest", "navhard_two_stage", "", None, 2])
def test_non_v2_protocol_ids_are_rejected(protocol_id: object) -> None:
    with pytest.raises(ValueError, match="V2|protocol"):
        get_cvoi_navsim_metric_protocol(protocol_id)  # type: ignore[arg-type]
