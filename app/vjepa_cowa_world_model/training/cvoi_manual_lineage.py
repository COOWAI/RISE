"""Exact lineage authority for the retained manual NavSim CVoI chain."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CVOI_MANUAL_FULL_RESULTS_ROOT = Path("/path/to/rise/results/cvoi_manual_full")
CVOI_MANUAL_ABLATION_RESULTS_ROOT = Path("/path/to/rise/results/cvoi_manual_ablation")

_FULL_HANDOFF_SUFFIXES = {
    "p0_handoff": Path("handoff/p0_selected.pt"),
    "field_handoff": Path("handoff/field.pt"),
    "calibration_handoff": Path("handoff/calibration.pt"),
    "p1_handoff": Path("handoff/p1_selected.pt"),
    "stop_handoff": Path("handoff/stop.pt"),
    "oracle_handoff": Path("handoff/oracle_full.sqlite3"),
    "gate_handoff": Path("handoff/gate.pt"),
}

_FULL_STAGE_ARTIFACT_SUFFIXES = {
    "unguided_planner": {
        "output_checkpoint": Path("p0/p0_planner_checkpoint.pt"),
    },
    "field_warmup": {
        "unguided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p0_handoff"],
        "output_checkpoint": _FULL_HANDOFF_SUFFIXES["field_handoff"],
    },
    "field_calibrated": {
        "unguided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p0_handoff"],
        "field_checkpoint": _FULL_HANDOFF_SUFFIXES["field_handoff"],
        "output_checkpoint": _FULL_HANDOFF_SUFFIXES["calibration_handoff"],
    },
    "guided_planner": {
        "unguided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p0_handoff"],
        "field_checkpoint": _FULL_HANDOFF_SUFFIXES["calibration_handoff"],
        "output_checkpoint": Path("p1/p1_planner_checkpoint.pt"),
    },
    "stop_calibrated": {
        "unguided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p0_handoff"],
        "field_checkpoint": _FULL_HANDOFF_SUFFIXES["calibration_handoff"],
        "guided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p1_handoff"],
        "output_checkpoint": _FULL_HANDOFF_SUFFIXES["stop_handoff"],
    },
    "gate_distillation": {
        "oracle_path": _FULL_HANDOFF_SUFFIXES["oracle_handoff"],
        "output_checkpoint": _FULL_HANDOFF_SUFFIXES["gate_handoff"],
    },
    "evaluation": {
        "unguided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p0_handoff"],
        "field_checkpoint": _FULL_HANDOFF_SUFFIXES["calibration_handoff"],
        "guided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p1_handoff"],
        "dual_value_checkpoint": _FULL_HANDOFF_SUFFIXES["stop_handoff"],
    },
}

_ABLATION_STAGE_ARTIFACT_SUFFIXES = {
    "field_warmup": {
        "output_checkpoint": _FULL_HANDOFF_SUFFIXES["field_handoff"],
    },
    "field_calibrated": {
        "field_checkpoint": _FULL_HANDOFF_SUFFIXES["field_handoff"],
        "output_checkpoint": _FULL_HANDOFF_SUFFIXES["calibration_handoff"],
    },
    "guided_planner": {
        "field_checkpoint": _FULL_HANDOFF_SUFFIXES["calibration_handoff"],
        "output_checkpoint": Path("p1/p1_planner_checkpoint.pt"),
    },
    "stop_calibrated": {
        "field_checkpoint": _FULL_HANDOFF_SUFFIXES["calibration_handoff"],
        "guided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p1_handoff"],
        "output_checkpoint": _FULL_HANDOFF_SUFFIXES["stop_handoff"],
    },
    "gate_distillation": {
        "output_checkpoint": _FULL_HANDOFF_SUFFIXES["gate_handoff"],
    },
    "evaluation": {
        "field_checkpoint": _FULL_HANDOFF_SUFFIXES["calibration_handoff"],
        "guided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p1_handoff"],
        "dual_value_checkpoint": _FULL_HANDOFF_SUFFIXES["stop_handoff"],
    },
}

_VALUE_NAME_BY_MECHANISM = {
    ("hazard_quality", "local_geometry"): "full",
    ("none", "local_geometry"): "no_cf",
    ("hazard_only", "local_geometry"): "hazard_only",
    ("quality_only", "local_geometry"): "quality_only",
}
_VALUE_MECHANISM_BY_NAME = {name: mechanism for mechanism, name in _VALUE_NAME_BY_MECHANISM.items()}
_VALUE_STAGES = {
    "field_warmup": "field",
    "field_calibrated": "calibration",
    "guided_planner": "p1",
    "stop_calibrated": "stop",
}
_GATE_CONTRACTS = {
    "full": ("main", "hazard_quality", "full", "full"),
    "no_cf": ("ablation", "none", "full", "no_cf"),
    "without_field": ("ablation", "hazard_quality", "without_field", "full"),
    "without_stop": ("ablation", "hazard_quality", "without_stop", "full"),
    "without_value_summary": (
        "ablation",
        "hazard_quality",
        "without_value_summary",
        "full",
    ),
}


def _validate_lexical_absolute_path(path: str | Path, *, field: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise TypeError(f"{field} must be a string or pathlib.Path")
    raw = str(path)
    normalized = Path(raw)
    if not normalized.is_absolute():
        raise ValueError(f"{field} must be absolute: {normalized}")
    if ".." in normalized.parts:
        raise ValueError(f"{field} must not contain '..' traversal: {normalized}")
    if raw != str(normalized):
        raise ValueError(f"{field} must use canonical lexical spelling: {raw}")
    return normalized


def derive_cvoi_manual_full_handoffs(results_root: str | Path) -> dict[str, Path]:
    """Derive the seven fixed Full-chain handoffs from one canonical root."""

    root = _validate_lexical_absolute_path(results_root, field="manual CVoI Full results root")
    return {name: root / suffix for name, suffix in _FULL_HANDOFF_SUFFIXES.items()}


def resolve_cvoi_manual_full_results_root(
    handoffs: Mapping[str, str | Path],
    *,
    expected_results_root: str | Path | None = None,
) -> Path:
    """Resolve and structurally validate one root shared by named handoffs."""

    if not isinstance(handoffs, Mapping) or not handoffs:
        raise ValueError("manual CVoI Full handoffs must be a non-empty mapping")
    unknown = set(handoffs) - set(_FULL_HANDOFF_SUFFIXES)
    if unknown:
        raise ValueError(f"unsupported manual CVoI Full handoff names: {sorted(unknown)!r}")

    resolved_root: Path | None = None
    normalized_handoffs: dict[str, Path] = {}
    for name, raw_path in handoffs.items():
        path = _validate_lexical_absolute_path(raw_path, field=f"manual CVoI Full {name} path")
        suffix = _FULL_HANDOFF_SUFFIXES[name]
        if path.parts[-len(suffix.parts) :] != suffix.parts:
            raise ValueError(f"manual CVoI Full {name} fixed handoff path must end with exact suffix {suffix}")
        candidate_root = Path(*path.parts[: -len(suffix.parts)])
        candidate_root = _validate_lexical_absolute_path(
            candidate_root,
            field="manual CVoI Full results root",
        )
        if resolved_root is not None and candidate_root != resolved_root:
            raise ValueError("manual CVoI Full handoffs must share the same results root")
        resolved_root = candidate_root
        normalized_handoffs[name] = path

    if resolved_root is None:
        raise RuntimeError("manual CVoI Full handoff resolution lost its results root")
    if expected_results_root is None:
        expected_results_root = resolved_root
    expected = derive_cvoi_manual_full_handoffs(expected_results_root)
    for name, path in normalized_handoffs.items():
        if path != expected[name]:
            raise ValueError(
                f"manual CVoI Full {name} must use the shared results root; "
                f"path must be exactly {expected[name]} "
                f"(suffix {_FULL_HANDOFF_SUFFIXES[name]})"
            )
    return Path(expected_results_root)


def _configuration_value(config: object, field_name: str) -> object:
    if isinstance(config, Mapping):
        return config.get(field_name)
    return getattr(config, field_name, None)


def _signature_value(config: object, field_name: str) -> object:
    signature = _configuration_value(config, "ablation_signature")
    return _configuration_value(signature, field_name)


def _results_root_from_suffixed_path(
    value: object,
    *,
    field_name: str,
    suffix: Path,
    root_field: str = "manual CVoI Full results root",
) -> Path:
    role = {
        "unguided_planner_checkpoint": "P0 checkpoint",
        "field_checkpoint": "Calibration/Field checkpoint",
        "guided_planner_checkpoint": "P1 checkpoint",
        "dual_value_checkpoint": "Stop checkpoint",
        "oracle_path": "Oracle path",
        "output_checkpoint": "stage output path",
    }.get(field_name, "artifact path")
    try:
        path = _validate_lexical_absolute_path(value, field=f"cvoi.{field_name}")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"cvoi.{field_name} {role} must use fixed handoff path ending exactly with {suffix}: {error}"
        ) from error
    if path.parts[-len(suffix.parts) :] != suffix.parts:
        raise ValueError(f"cvoi.{field_name} {role} must use fixed handoff path ending exactly with {suffix}")
    return _validate_lexical_absolute_path(
        Path(*path.parts[: -len(suffix.parts)]),
        field=root_field,
    )


def resolve_cvoi_manual_full_results_root_from_config(config: object) -> Path:
    """Resolve the configured Full root from one stage's explicit artifact paths.

    The neutral module constant documents the public example only. Runtime callers
    use this function so a user-edited YAML remains the path authority.
    """

    stage = _configuration_value(config, "stage")
    if type(stage) is not str or stage not in _FULL_STAGE_ARTIFACT_SUFFIXES:
        raise ValueError(f"unsupported manual CVoI stage for results-root resolution: {stage!r}")
    suffixes = _FULL_STAGE_ARTIFACT_SUFFIXES[stage]
    experiment_role = _signature_value(config, "experiment_role")
    if experiment_role == "ablation" and stage not in {"unguided_planner", "gate_distillation"}:
        suffixes = {
            "unguided_planner_checkpoint": _FULL_HANDOFF_SUFFIXES["p0_handoff"],
        }
    roots = {}
    for field_name, suffix in suffixes.items():
        value = _configuration_value(config, field_name)
        if value is None:
            continue
        roots[field_name] = _results_root_from_suffixed_path(
            value,
            field_name=field_name,
            suffix=suffix,
        )
    if not roots:
        raise ValueError(f"manual CVoI stage={stage!r} does not configure a results-root artifact path")
    distinct_roots = set(roots.values())
    if len(distinct_roots) != 1:
        rendered = {field_name: str(root) for field_name, root in sorted(roots.items())}
        raise ValueError(f"manual CVoI Full stage artifacts must share one results root: {rendered}")
    return next(iter(distinct_roots))


def resolve_cvoi_manual_ablation_results_root_from_config(
    config: object,
    *,
    artifact_fields: tuple[str, ...] | None = None,
) -> Path:
    """Resolve the branch-parent ablation root from stage-local configured artifacts."""

    stage = _configuration_value(config, "stage")
    suffixes = _ABLATION_STAGE_ARTIFACT_SUFFIXES.get(stage)
    if suffixes is None:
        raise ValueError(f"unsupported manual CVoI ablation stage for results-root resolution: {stage!r}")
    if _signature_value(config, "experiment_role") != "ablation":
        raise ValueError("manual CVoI ablation results-root resolution requires experiment_role='ablation'")

    if stage == "gate_distillation":
        branch_name = _signature_value(config, "branch_id")
        if branch_name not in _GATE_CONTRACTS or branch_name == "full":
            raise ValueError(f"unsupported manual CVoI ablation Gate branch: {branch_name!r}")
    else:
        mechanism = (
            _signature_value(config, "cf_field_supervision"),
            _signature_value(config, "field_calibration_mode"),
        )
        branch_name = _VALUE_NAME_BY_MECHANISM.get(mechanism)
        if branch_name is None or branch_name == "full":
            raise ValueError(f"unsupported manual CVoI ablation Value mechanism: {mechanism!r}")

    if artifact_fields is not None:
        unknown_fields = set(artifact_fields) - set(suffixes)
        if unknown_fields:
            raise ValueError(
                f"manual CVoI ablation stage={stage!r} cannot derive its root from fields "
                f"{sorted(unknown_fields)!r}"
            )
        suffixes = {field_name: suffixes[field_name] for field_name in artifact_fields}

    configured_suffixes = {
        field_name: suffix
        for field_name, suffix in suffixes.items()
        if _configuration_value(config, field_name) is not None
    }
    if artifact_fields is None and "output_checkpoint" in configured_suffixes:
        configured_suffixes = {"output_checkpoint": configured_suffixes["output_checkpoint"]}

    branch_roots = {}
    for field_name, suffix in configured_suffixes.items():
        value = _configuration_value(config, field_name)
        branch_roots[field_name] = _results_root_from_suffixed_path(
            value,
            field_name=field_name,
            suffix=suffix,
            root_field="manual CVoI ablation branch result root",
        )
    if not branch_roots:
        raise ValueError(f"manual CVoI ablation stage={stage!r} does not configure a branch-local artifact path")
    distinct_roots = set(branch_roots.values())
    if len(distinct_roots) != 1:
        rendered = {field_name: str(root) for field_name, root in sorted(branch_roots.items())}
        raise ValueError(f"manual CVoI ablation stage artifacts must share one branch result root: {rendered}")
    branch_root = next(iter(distinct_roots))
    if branch_root.name != branch_name:
        role = "Calibration" if "field_checkpoint" in branch_roots else "stage output"
        raise ValueError(
            f"manual CVoI ablation {role} fixed handoff path must use branch directory "
            f"{branch_name!r}, got {branch_root}"
        )
    return _validate_lexical_absolute_path(
        branch_root.parent,
        field="manual CVoI ablation results root",
    )


def reject_unedited_cvoi_public_placeholders(config: object, *, boundary: str) -> None:
    """Reject visible public path examples only when a production boundary starts."""

    if type(boundary) is not str or not boundary.strip():
        raise ValueError("placeholder validation boundary must be a non-empty string")
    placeholders: list[str] = []
    visited: set[int] = set()

    def _walk(value: object, path: str) -> None:
        if isinstance(value, str):
            if value.startswith("/path/to/"):
                placeholders.append(f"{path}={value}")
            return
        if value is None or isinstance(value, (bool, int, float, Path)):
            return
        identity = id(value)
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, Mapping):
            for key, child in value.items():
                _walk(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(child, f"{path}[{index}]")
            return
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for key, child in attributes.items():
                _walk(child, f"{path}.{key}" if path else key)

    _walk(config, "config")
    if placeholders:
        raise ValueError(
            f"{boundary} production preflight requires you to replace unedited /path/to/ placeholders: "
            + ", ".join(sorted(placeholders))
        )


@dataclass(frozen=True)
class CvoiManualValueLineage:
    """One retained Value-training lineage and its fixed handoff paths."""

    name: str
    cf_field_supervision: str
    result_root: Path
    p0_result_root: Path

    def __post_init__(self) -> None:
        result_root = _validate_lexical_absolute_path(
            self.result_root,
            field="CvoiManualValueLineage result_root",
        )
        p0_result_root = _validate_lexical_absolute_path(
            self.p0_result_root,
            field="CvoiManualValueLineage p0_result_root",
        )
        if self.name == "full" and p0_result_root != result_root:
            raise ValueError("full Value lineage requires p0_result_root to equal result_root")
        if self.name != "full" and p0_result_root == result_root:
            raise ValueError("ablation Value lineage requires p0_result_root to remain separate from result_root")
        object.__setattr__(self, "result_root", result_root)
        object.__setattr__(self, "p0_result_root", p0_result_root)

    @property
    def p0_branch_id(self) -> str:
        return "p0_uniform"

    @property
    def p0_handoff(self) -> Path:
        return self.p0_result_root / "handoff/p0_selected.pt"

    @property
    def field_handoff(self) -> Path:
        return self.result_root / "handoff/field.pt"

    @property
    def calibration_handoff(self) -> Path:
        return self.result_root / "handoff/calibration.pt"

    @property
    def p1_handoff(self) -> Path:
        return self.result_root / "handoff/p1_selected.pt"

    @property
    def stop_handoff(self) -> Path:
        return self.result_root / "handoff/stop.pt"

    @property
    def oracle_handoff(self) -> Path:
        return self.result_root / "handoff/oracle_full.sqlite3"

    @property
    def gate_handoff(self) -> Path:
        return self.result_root / "handoff/gate.pt"

    def checkpoint_branch_id(self, stage: str) -> str:
        prefix = _VALUE_STAGES.get(stage)
        if prefix is None:
            raise ValueError(f"unsupported manual Value stage: {stage!r}")
        if stage == "stop_calibrated" and self.name not in {"full", "no_cf"}:
            raise ValueError(f"{self.name!r} has no retained Stop stage")
        return f"{prefix}_{self.name}"


@dataclass(frozen=True)
class CvoiManualGateBranch:
    """One retained Gate branch and the Value lineage that owns its Oracle."""

    name: str
    feature_mode: str
    result_root: Path
    oracle_value_lineage: str


@dataclass(frozen=True)
class CvoiManualRuntimeValueInput:
    """Exact Value artifact consumed by one configured runtime stage."""

    lineage: CvoiManualValueLineage
    required_phase: str
    path_field: str
    checkpoint_path: Path

    @property
    def required_branch_id(self) -> str:
        return self.lineage.checkpoint_branch_id(self.required_phase)


def _signature_field(signature: object, field_name: str) -> str:
    value = getattr(signature, field_name, None)
    if type(value) is not str or not value:
        raise ValueError(f"manual CVoI signature requires non-empty {field_name}")
    return value


def _value_lineage_by_name(
    name: str,
    *,
    required_stage: str,
    full_results_root: str | Path | None = None,
    ablation_results_root: str | Path | None = None,
) -> CvoiManualValueLineage:
    mechanism = _VALUE_MECHANISM_BY_NAME.get(name)
    if mechanism is None:
        raise ValueError(f"unsupported manual Value lineage: {name!r}")
    if required_stage not in _VALUE_STAGES:
        raise ValueError(f"unsupported manual Value stage: {required_stage!r}")
    if required_stage == "stop_calibrated" and name not in {"full", "no_cf"}:
        raise ValueError(f"{name!r} has no retained Stop stage")
    supervision, _ = mechanism
    p0_result_root = _validate_lexical_absolute_path(
        CVOI_MANUAL_FULL_RESULTS_ROOT if full_results_root is None else full_results_root,
        field="manual CVoI Full results root",
    )
    if name == "full":
        result_root = p0_result_root
    else:
        ablation_root = _validate_lexical_absolute_path(
            CVOI_MANUAL_ABLATION_RESULTS_ROOT if ablation_results_root is None else ablation_results_root,
            field="manual CVoI ablation results root",
        )
        result_root = ablation_root / name
    return CvoiManualValueLineage(
        name=name,
        cf_field_supervision=supervision,
        result_root=result_root,
        p0_result_root=p0_result_root,
    )


def resolve_cvoi_manual_value_lineage(
    signature: object,
    *,
    stage: str,
    full_results_root: str | Path | None = None,
    ablation_results_root: str | Path | None = None,
) -> CvoiManualValueLineage:
    """Resolve one exact Value lineage from its typed-signature fields.

    Parameters
    ----------
    signature:
        Object exposing the fixed manual-ablation signature fields.
    stage:
        Exact Value-training stage whose branch identity must be validated.

    Returns
    -------
    CvoiManualValueLineage
        Immutable lineage and handoff authority for the requested stage.
    """

    supervision = _signature_field(signature, "cf_field_supervision")
    calibration_mode = _signature_field(signature, "field_calibration_mode")
    name = _VALUE_NAME_BY_MECHANISM.get((supervision, calibration_mode))
    if name is None:
        raise ValueError(
            "unsupported manual Value mechanism: "
            f"cf_field_supervision={supervision!r}, field_calibration_mode={calibration_mode!r}"
        )
    if _signature_field(signature, "p0_prefix_mode") != "uniform":
        raise ValueError("manual Value lineage requires p0_prefix_mode='uniform'")
    if _signature_field(signature, "gate_feature_mode") != "full":
        raise ValueError("manual Value lineage requires gate_feature_mode='full'")

    expected_role = "main" if name == "full" else "ablation"
    role = _signature_field(signature, "experiment_role")
    if role != expected_role:
        raise ValueError(f"manual Value lineage {name!r} requires experiment_role={expected_role!r}, got {role!r}")

    lineage = _value_lineage_by_name(
        name,
        required_stage=stage,
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )
    expected_branch_id = f"p1_{name}" if stage == "guided_planner" else name
    branch_id = _signature_field(signature, "branch_id")
    if branch_id != expected_branch_id:
        raise ValueError(f"manual Value stage {stage!r} requires branch_id={expected_branch_id!r}, got {branch_id!r}")
    return lineage


def resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
    *,
    phase: str,
    branch_id: str,
    full_results_root: str | Path | None = None,
    ablation_results_root: str | Path | None = None,
) -> CvoiManualValueLineage:
    """Reverse one exact direct-checkpoint branch into its manual lineage.

    Parameters
    ----------
    phase:
        Exact direct Value checkpoint phase.
    branch_id:
        Exact branch identifier stored in checkpoint metadata.

    Returns
    -------
    CvoiManualValueLineage
        Immutable lineage matched by both phase and branch identifier.
    """

    prefix_by_phase = {
        "field_warmup": "field",
        "field_calibrated": "calibration",
        "guided_planner": "p1",
        "stop_calibrated": "stop",
    }
    prefix = prefix_by_phase.get(phase)
    if prefix is None:
        raise ValueError(f"unsupported manual Value phase: {phase!r}")
    if type(branch_id) is not str or not branch_id.startswith(f"{prefix}_"):
        raise ValueError(f"{phase} branch_id must start with {prefix!r}")
    name = branch_id.removeprefix(f"{prefix}_")
    lineage = _value_lineage_by_name(
        name,
        required_stage=phase,
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )
    if lineage.checkpoint_branch_id(phase) != branch_id:
        raise ValueError(f"invalid manual Value checkpoint branch_id: {branch_id!r}")
    return lineage


def resolve_cvoi_manual_value_lineage_from_artifacts(
    artifacts: Mapping[str, str | Path],
    *,
    phase: str,
    branch_id: str,
) -> CvoiManualValueLineage:
    """Resolve one Value lineage from its explicit P0 and branch-local handoffs."""

    required_roles = {
        "p0_planner_checkpoint",
        "field_checkpoint",
        "calibration_checkpoint",
        "p1_planner_checkpoint",
        "stop_checkpoint",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_roles:
        raise ValueError("manual Value artifacts must contain exactly the five fixed handoff roles")

    full_results_root = resolve_cvoi_manual_full_results_root({"p0_handoff": artifacts["p0_planner_checkpoint"]})
    value_results_root = resolve_cvoi_manual_full_results_root(
        {
            "field_handoff": artifacts["field_checkpoint"],
            "calibration_handoff": artifacts["calibration_checkpoint"],
            "p1_handoff": artifacts["p1_planner_checkpoint"],
            "stop_handoff": artifacts["stop_checkpoint"],
        }
    )

    prefix = _VALUE_STAGES.get(phase)
    if prefix is None:
        raise ValueError(f"unsupported manual Value phase: {phase!r}")
    expected_prefix = f"{prefix}_"
    if type(branch_id) is not str or not branch_id.startswith(expected_prefix):
        raise ValueError(f"{phase} branch_id must start with {prefix!r}")
    lineage_name = branch_id.removeprefix(expected_prefix)
    if lineage_name == "full":
        if value_results_root != full_results_root:
            raise ValueError("full manual Value artifacts must share the P0 results root")
        ablation_results_root = value_results_root.parent
    else:
        if value_results_root.name != lineage_name:
            raise ValueError(
                "manual Value branch-local artifacts must use a directory named "
                f"{lineage_name!r}, got {value_results_root}"
            )
        ablation_results_root = value_results_root.parent

    return resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase=phase,
        branch_id=branch_id,
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )


def build_cvoi_manual_value_parents(
    lineage: CvoiManualValueLineage,
    phase: str,
) -> dict[str, object]:
    """Build the existing direct-Value checkpoint parent payload exactly.

    Parameters
    ----------
    lineage:
        Validated manual Value lineage.
    phase:
        Direct Value phase whose parent schema is requested.

    Returns
    -------
    dict[str, object]
        Exact nested parent mapping stored in the direct checkpoint.
    """

    if not isinstance(lineage, CvoiManualValueLineage):
        raise TypeError("lineage must be a CvoiManualValueLineage")
    p0 = {"stage": "p0", "branch_id": "p0_uniform"}
    if phase == "field_warmup":
        return {"unguided_planner": p0}
    if phase == "field_calibrated":
        return {
            "unguided_planner": p0,
            "field": {
                "phase": "field_warmup",
                "branch_id": lineage.checkpoint_branch_id("field_warmup"),
            },
        }
    if phase == "stop_calibrated":
        lineage.checkpoint_branch_id(phase)
        return {
            "unguided_planner": p0,
            "calibration": {
                "phase": "field_calibrated",
                "branch_id": lineage.checkpoint_branch_id("field_calibrated"),
            },
            "guided_planner": {
                "stage": "p1",
                "branch_id": lineage.checkpoint_branch_id("guided_planner"),
            },
        }
    raise ValueError(f"unsupported direct Value phase: {phase!r}")


def resolve_cvoi_manual_gate_branch(
    signature: object,
    *,
    full_results_root: str | Path | None = None,
    ablation_results_root: str | Path | None = None,
) -> CvoiManualGateBranch:
    """Resolve one exact retained Gate experiment and its Oracle source.

    Parameters
    ----------
    signature:
        Object exposing the fixed manual-ablation signature fields.

    Returns
    -------
    CvoiManualGateBranch
        Immutable Gate branch, result root, and Oracle lineage authority.
    """

    branch_id = _signature_field(signature, "branch_id")
    contract = _GATE_CONTRACTS.get(branch_id)
    if contract is None:
        raise ValueError(f"unsupported manual Gate branch_id: {branch_id!r}")
    expected_role, expected_supervision, expected_feature_mode, oracle_lineage = contract
    if _signature_field(signature, "field_calibration_mode") != "local_geometry":
        raise ValueError("manual Gate branch requires field_calibration_mode='local_geometry'")
    if _signature_field(signature, "p0_prefix_mode") != "uniform":
        raise ValueError("manual Gate branch requires p0_prefix_mode='uniform'")

    expected_fields = {
        "experiment_role": expected_role,
        "cf_field_supervision": expected_supervision,
        "gate_feature_mode": expected_feature_mode,
    }
    for field_name, expected in expected_fields.items():
        actual = _signature_field(signature, field_name)
        if actual != expected:
            raise ValueError(f"manual Gate branch {branch_id!r} requires {field_name}={expected!r}, got {actual!r}")
    if branch_id == "full":
        result_root = _validate_lexical_absolute_path(
            CVOI_MANUAL_FULL_RESULTS_ROOT if full_results_root is None else full_results_root,
            field="manual CVoI Full results root",
        )
    else:
        ablation_root = _validate_lexical_absolute_path(
            CVOI_MANUAL_ABLATION_RESULTS_ROOT if ablation_results_root is None else ablation_results_root,
            field="manual CVoI ablation results root",
        )
        result_root = ablation_root / branch_id
    return CvoiManualGateBranch(
        name=branch_id,
        feature_mode=expected_feature_mode,
        result_root=result_root,
        oracle_value_lineage=oracle_lineage,
    )


def derive_cvoi_manual_value_oracle_handoff(
    value_lineage: str,
    *,
    full_results_root: str | Path | None = None,
    ablation_results_root: str | Path | None = None,
) -> Path:
    """Derive a retained Oracle path without requiring an unrelated P0 root."""

    if value_lineage == "full":
        result_root = _validate_lexical_absolute_path(
            CVOI_MANUAL_FULL_RESULTS_ROOT if full_results_root is None else full_results_root,
            field="manual CVoI Full results root",
        )
    elif value_lineage == "no_cf":
        ablation_root = _validate_lexical_absolute_path(
            CVOI_MANUAL_ABLATION_RESULTS_ROOT if ablation_results_root is None else ablation_results_root,
            field="manual CVoI ablation results root",
        )
        result_root = ablation_root / value_lineage
    else:
        raise ValueError(f"unsupported manual Oracle Value lineage: {value_lineage!r}")
    return result_root / _FULL_HANDOFF_SUFFIXES["oracle_handoff"]


def resolve_cvoi_manual_runtime_value_input(
    signature: object,
    *,
    configured_stage: str,
    evaluation_mode: str = "controller",
    full_results_root: str | Path | None = None,
    ablation_results_root: str | Path | None = None,
) -> CvoiManualRuntimeValueInput | None:
    """Resolve the exact Calibration or Stop artifact consumed at runtime.

    Parameters
    ----------
    signature:
        Object exposing the fixed manual-ablation signature fields.
    configured_stage:
        Training or evaluation stage requesting a Value artifact.
    evaluation_mode:
        Existing controller, P0-forced, or P1-Field-forced evaluation mode.

    Returns
    -------
    CvoiManualRuntimeValueInput | None
        Exact Value input, or ``None`` for stages that consume no Value model.
    """

    if configured_stage == "evaluation":
        if evaluation_mode == "p0_forced":
            return None
        evaluation_inputs = {
            "p1_field_forced": ("field_checkpoint", "field_calibrated"),
            "controller": ("dual_value_checkpoint", "stop_calibrated"),
        }
        selected = evaluation_inputs.get(evaluation_mode)
        if selected is None:
            raise ValueError(f"unsupported manual evaluation mode: {evaluation_mode!r}")
        path_field, required_phase = selected
        signature_stage = "guided_planner"
    else:
        runtime_inputs = {
            "guided_planner": ("guided_planner", "field_checkpoint", "field_calibrated"),
            "stop_calibrated": ("stop_calibrated", "field_checkpoint", "field_calibrated"),
        }
        selected_runtime = runtime_inputs.get(configured_stage)
        if selected_runtime is None:
            return None
        if evaluation_mode != "controller":
            raise ValueError(f"manual stage {configured_stage!r} requires evaluation_mode='controller'")
        signature_stage, path_field, required_phase = selected_runtime

    lineage = resolve_cvoi_manual_value_lineage(
        signature,
        stage=signature_stage,
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )
    checkpoint_path = {
        "field_calibrated": lineage.calibration_handoff,
        "stop_calibrated": lineage.stop_handoff,
    }[required_phase]
    return CvoiManualRuntimeValueInput(
        lineage=lineage,
        required_phase=required_phase,
        path_field=path_field,
        checkpoint_path=checkpoint_path,
    )
