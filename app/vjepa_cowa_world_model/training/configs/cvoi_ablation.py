"""Strict identities and registered modes for the manual NavSim e120 CVoI chain."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Mapping

from app.vjepa_cowa_world_model.training.prefix_schedule import PrefixDistribution, resolve_prefix_distribution

CVOI_FORMAL_V2_NAVSIM_E120_ABLATION_SCHEMA = "cvoi_formal_v2_navsim_e120_ablation_v1"
CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL = "formal_v2_navsim_e120_h4_v3"
CVOI_EVALUATION_SEED = 239
CVOI_CF_FIELD_WEIGHTS = {
    "none": (0.0, 0.0),
    "hazard_only": (1.0, 0.0),
    "quality_only": (0.0, 1.0),
    "hazard_quality": (1.0, 1.0),
}
CVOI_PREFIX_POLICY_PARAMETERS = {
    "uniform": {"full_prefix_prob": 0.2, "min_prefix_steps": 0, "max_non_full_prefix_steps": None},
    "extremes": {"full_prefix_prob": 0.5, "min_prefix_steps": 0, "max_non_full_prefix_steps": 0},
    "short_heavy": {"full_prefix_prob": 0.1, "min_prefix_steps": 0, "max_non_full_prefix_steps": None},
    "no_full": {"full_prefix_prob": 0.0, "min_prefix_steps": 0, "max_non_full_prefix_steps": None},
}

# Retained by the generic feature-mask utility that the e120-specific mask wraps.
CVOI_GATE_FEATURE_MODES = frozenset({"full", "gate_no_stop_value", "gate_no_explicit_value_summary"})
CVOI_EXPERIMENT_ROLES = frozenset({"main", "ablation"})
CVOI_FORMAL_V2_NAVSIM_E120_INITIALIZATION_MODE = "full_state_warmstart"
CVOI_FORMAL_V2_NAVSIM_E120_PREFIX_MODES = frozenset(CVOI_PREFIX_POLICY_PARAMETERS)
CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES = frozenset(
    {"full", "without_field", "without_stop", "without_value_summary"}
)
CVOI_FORMAL_V2_NAVSIM_E120_VALUE_MECHANISMS = frozenset(
    {
        ("hazard_quality", "local_geometry"),
        ("none", "local_geometry"),
        ("hazard_only", "local_geometry"),
        ("quality_only", "local_geometry"),
        ("hazard_quality", "local_geometry_no_order"),
        ("hazard_quality", "factual_only"),
    }
)

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,95}")


@dataclass(frozen=True)
class CvoiFormalV2NavSimE120AblationSignature:
    """Identity for one retained NavSim-only e120 lineage."""

    schema: str
    protocol_version: str
    experiment_role: str
    branch_id: str
    shared_cohort_id: str
    initialization_mode: str
    cf_field_supervision: str
    field_calibration_mode: str
    p0_prefix_mode: str
    gate_feature_mode: str
    train_seed: int
    evaluation_seed: int
    training_stride: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_cvoi_ablation_signature(raw: object) -> CvoiFormalV2NavSimE120AblationSignature:
    """Parse the sole retained e120 ablation identity without accepting extra fields."""

    if not isinstance(raw, Mapping):
        raise ValueError("cvoi.ablation_signature must be a mapping")
    schema = raw.get("schema")
    if schema != CVOI_FORMAL_V2_NAVSIM_E120_ABLATION_SCHEMA:
        raise ValueError(
            "cvoi.ablation_signature.schema must be exactly "
            f"{CVOI_FORMAL_V2_NAVSIM_E120_ABLATION_SCHEMA!r}, got {schema!r}"
        )

    expected = frozenset(CvoiFormalV2NavSimE120AblationSignature.__dataclass_fields__)
    actual = frozenset(raw)
    if actual != expected:
        raise ValueError(
            "cvoi.ablation_signature fields mismatch: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )
    value = CvoiFormalV2NavSimE120AblationSignature(**dict(raw))
    fixed_strings = {
        "schema": CVOI_FORMAL_V2_NAVSIM_E120_ABLATION_SCHEMA,
        "protocol_version": CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL,
        "initialization_mode": CVOI_FORMAL_V2_NAVSIM_E120_INITIALIZATION_MODE,
    }
    for field_name, expected_value in fixed_strings.items():
        candidate = getattr(value, field_name)
        if type(candidate) is not str or candidate != expected_value:
            raise ValueError(
                f"cvoi.ablation_signature.{field_name} must be exactly {expected_value!r}, got {candidate!r}"
            )
    for field_name in ("experiment_role", "branch_id", "shared_cohort_id"):
        candidate = getattr(value, field_name)
        if not isinstance(candidate, str) or _IDENTIFIER.fullmatch(candidate) is None:
            raise ValueError(f"cvoi.ablation_signature.{field_name} must be lowercase snake case")
    if value.experiment_role not in CVOI_EXPERIMENT_ROLES:
        raise ValueError(
            "cvoi.ablation_signature.experiment_role must be one of "
            f"{sorted(CVOI_EXPERIMENT_ROLES)}, got {value.experiment_role!r}"
        )
    mechanism = (value.cf_field_supervision, value.field_calibration_mode)
    if mechanism not in CVOI_FORMAL_V2_NAVSIM_E120_VALUE_MECHANISMS:
        raise ValueError(
            "cvoi.ablation_signature cf_field_supervision/field_calibration_mode must identify one of the "
            f"six NavSim mechanisms, got {mechanism!r}"
        )
    if value.p0_prefix_mode not in CVOI_FORMAL_V2_NAVSIM_E120_PREFIX_MODES:
        raise ValueError(
            "cvoi.ablation_signature.p0_prefix_mode must be one of "
            f"{sorted(CVOI_FORMAL_V2_NAVSIM_E120_PREFIX_MODES)}, got {value.p0_prefix_mode!r}"
        )
    if value.gate_feature_mode not in CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES:
        raise ValueError(
            "cvoi.ablation_signature.gate_feature_mode must be one of "
            f"{sorted(CVOI_FORMAL_V2_NAVSIM_E120_GATE_FEATURE_MODES)}, got {value.gate_feature_mode!r}"
        )
    if type(value.train_seed) is not int or value.train_seed != CVOI_EVALUATION_SEED:
        raise ValueError(f"cvoi.ablation_signature.train_seed must be exactly 239, got {value.train_seed!r}")
    if type(value.evaluation_seed) is not int or value.evaluation_seed != CVOI_EVALUATION_SEED:
        raise ValueError(f"cvoi.ablation_signature.evaluation_seed must be exactly 239, got {value.evaluation_seed!r}")
    if type(value.training_stride) is not int or value.training_stride != 4:
        raise ValueError(f"cvoi.ablation_signature.training_stride must be exactly 4, got {value.training_stride!r}")
    return value


def is_cvoi_gate_ablation_shared_parent(stored: object, expected: object) -> bool:
    """Return whether one registered e120 Gate mask may consume the full parent."""

    try:
        parent = parse_cvoi_ablation_signature(stored).to_dict()
        child = parse_cvoi_ablation_signature(expected).to_dict()
    except (TypeError, ValueError):
        return False
    if child["gate_feature_mode"] not in {"without_field", "without_stop", "without_value_summary"}:
        return False
    if parent["experiment_role"] != "main" or parent["branch_id"] != "full":
        return False
    if parent["gate_feature_mode"] != "full" or child["experiment_role"] != "ablation":
        return False
    ignored = {"branch_id", "experiment_role", "gate_feature_mode"}
    return all(parent[key] == child[key] for key in child if key not in ignored)


def is_cvoi_p0_controller_shared_parent(stored: object, expected: object) -> bool:
    """Allow only the e120 strict Real-only Field trunk to seed the matched P0 Controller."""

    try:
        parent = parse_cvoi_ablation_signature(stored).to_dict()
        child = parse_cvoi_ablation_signature(expected).to_dict()
    except (TypeError, ValueError):
        return False
    if parent["branch_id"] != "strict_real_only" or child["branch_id"] != "p0":
        return False
    if parent["cf_field_supervision"] != "none" or child["cf_field_supervision"] != "none":
        return False
    ignored = {"branch_id"}
    return all(parent[key] == child[key] for key in child if key not in ignored)


def is_cvoi_navsim_e120_p1_value_parent(stored: object, expected: object) -> bool:
    """Allow an e120 P1 branch to consume only its signed Calibration parent."""

    try:
        parent = parse_cvoi_ablation_signature(stored).to_dict()
        child = parse_cvoi_ablation_signature(expected).to_dict()
    except (TypeError, ValueError):
        return False
    if child["branch_id"] != f"p1_{parent['branch_id']}":
        return False
    ignored = {"branch_id"}
    return all(parent[key] == child[key] for key in child if key not in ignored)


def resolve_cvoi_prefix_distribution(mode: str, *, horizon_steps: int) -> PrefixDistribution:
    """Resolve one registered e120 P0 training policy."""

    if mode not in CVOI_PREFIX_POLICY_PARAMETERS:
        raise ValueError(f"unknown CVoI prefix mode {mode!r}")
    distribution = resolve_prefix_distribution(
        enabled=True,
        horizon_steps=horizon_steps,
        **CVOI_PREFIX_POLICY_PARAMETERS[mode],
    )
    support = tuple(
        (prefix_steps, probability)
        for prefix_steps, probability in zip(distribution.prefix_steps, distribution.probabilities)
        if probability > 0.0
    )
    return PrefixDistribution(
        horizon_steps=distribution.horizon_steps,
        prefix_steps=tuple(value[0] for value in support),
        probabilities=tuple(value[1] for value in support),
    )
