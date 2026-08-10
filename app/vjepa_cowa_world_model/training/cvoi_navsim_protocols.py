"""Retained NavSim V2 metric authority for direct CVoI evaluation."""

from dataclasses import dataclass
from pathlib import Path

V2_PROTOCOL_ID = "epdms_v2_one_stage_navtest"


@dataclass(frozen=True)
class CvoiNavSimMetricProtocol:
    """The single NavSim authority retained by the manual CVoI workflow."""

    protocol_id: str
    authority_script: Path
    scorer_entrypoint: str
    split: str
    summary_token: str
    cache_family: str
    required_components: frozenset[str]
    nullable_components: frozenset[str]


_V2_PROTOCOL = CvoiNavSimMetricProtocol(
    protocol_id=V2_PROTOCOL_ID,
    authority_script=Path("scripts/eval_navsim/eval_navsim_v2_pdms.sh"),
    scorer_entrypoint="navsim/planning/script/run_pdm_score_one_stage.py",
    split="navtest",
    summary_token="average_all_frames",
    cache_family="navsim_v2",
    required_components=frozenset(
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
    ),
    nullable_components=frozenset({"two_frame_extended_comfort"}),
)


def get_cvoi_navsim_metric_protocol(protocol_id: str) -> CvoiNavSimMetricProtocol:
    """Return the retained V2 protocol or fail instead of substituting one."""

    if type(protocol_id) is not str or protocol_id != V2_PROTOCOL_ID:
        raise ValueError(f"unknown CVoI NavSim V2 metric protocol {protocol_id!r}; expected {V2_PROTOCOL_ID!r}")
    return _V2_PROTOCOL


__all__ = ("CvoiNavSimMetricProtocol", "V2_PROTOCOL_ID", "get_cvoi_navsim_metric_protocol")
