"""Counterfactual supervision protocol configuration."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CounterfactualHazardNegativePairingConfig:
    """Exact factual-real negative pairing contract for retained CF hazards."""

    enabled: bool = False
    hazard_domain: str = "counterfactual"
    hazard_accident_types: List[str] = field(default_factory=lambda: ["自车行为引起", "非自车行为引起"])
    safe_negative_domain: str = "real"
    safe_negative_semantics: str = "factual_real_sample"
    pairing_key: List[str] = field(default_factory=lambda: ["base_scene_id", "window_start_pos"])
    same_source_identity_required: bool = True
    cross_scene_pairing_forbidden: bool = True
    unmatched_pair_is_failure: bool = True
    fallback_pairing_forbidden: bool = True
    relabel_non_ego_hazard_as_safe_forbidden: bool = True


@dataclass
class CounterfactualSupervisionConfig:
    """Independent policy contract for counterfactual supervision v2."""

    enabled: bool = False
    protocol_version: str = "cf_supervision_v2"
    imitation_policy: str = "real_and_cf_safe"
    world_model_policy: str = "all_valid"
    value_policy: str = "all_valid"
    ego_hazard_types: List[str] = field(default_factory=lambda: ["自车行为引起", "非自车行为引起"])
    cf_sample_accident_type_allowlist: List[str] = field(default_factory=lambda: ["自车行为引起", "非自车行为引起"])
    cf_sample_filter_mode: str = "strict_allowlist"
    retained_accident_types: List[str] = field(default_factory=lambda: ["自车行为引起", "非自车行为引起"])
    retained_accident_types_are_all_hazards: bool = True
    hazard_negative_pairing: CounterfactualHazardNegativePairingConfig = field(
        default_factory=CounterfactualHazardNegativePairingConfig
    )
    hazard_target_mode: str = "episode_ranking"
