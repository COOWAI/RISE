"""Strict runtime primitives for CVoI Formal-v2 NavSim e120 Planner training."""

from __future__ import annotations

import copy
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from app.vjepa_cowa_world_model.training import cvoi_formal_v2_full_state_warmstart as _warmstart
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage as _manual_lineage
from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_HORIZONS,
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)
from app.vjepa_cowa_world_model.training.cvoi_gate_pipeline import CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import NAVTRAIN_GATE_PROTOCOL_ID
from app.vjepa_cowa_world_model.training.sequential_budget_control import (
    CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
    SequentialRolloutGate,
)
from app.vjepa_cowa_world_model.training.sequential_gate_training import (
    SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_NAVSIM_E120,
    validate_formal_v2_lambda_grid,
)

FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION = CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL
FORMAL_V2_NAVSIM_E120_DISTRIBUTED_STATUS_SCHEMA = "cvoi_formal_v2_navsim_e120_distributed_status_v1"
FORMAL_V2_NAVSIM_E120_GUIDANCE_SCHEMA = "cvoi_formal_v2_navsim_fixed_guidance_v1"
FORMAL_V2_NAVSIM_E120_DIRECT_LINEAGE_SCHEMA = "cvoi_formal_v2_navsim_e120_direct_lineage_v2"
FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_SCHEMA = "cvoi_formal_v2_navsim_e120_direct_checkpoint_v2"
FORMAL_V2_NAVSIM_E120_DIRECT_RESUME_SCHEMA = "cvoi_formal_v2_navsim_e120_direct_resume_v2"
FORMAL_V2_NAVSIM_E120_DIRECT_RESUME_CONTRACT = FORMAL_V2_NAVSIM_E120_DIRECT_RESUME_SCHEMA

MODEL_ROLES = ("encoder", "predictor", "planner")
FORMAL_V2_NAVSIM_E120_DIRECT_LINEAGE_FIELDS = frozenset(
    {
        "schema",
        "protocol_version",
        "stage",
        "branch_id",
    }
)
FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_FIELDS = frozenset(
    {
        "schema",
        "protocol_version",
        "resume_contract",
        "run_id",
        "stage",
        "encoder",
        "predictor",
        "planner",
        "optimizer",
        "scaler",
        "scheduler",
        "wd_scheduler",
        "epoch",
        "training_stop_epoch",
        "schedule_epochs",
        "selection_checkpoint_epochs",
        "cumulative_horizon_histogram",
        "role_state_shapes",
        "lineage",
    }
)

_DISTRIBUTED_STATUS_FIELDS = frozenset({"schema", "ok", "result", "error_type", "error_message"})
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,127}")
_DIRECT_BRANCHES_BY_STAGE = {
    "p0": frozenset({"p0_uniform"}),
    "p1": frozenset(
        {
            "p1_full",
            "p1_hazard_only",
            "p1_no_cf",
            "p1_quality_only",
        }
    ),
}
_DIRECT_FORBIDDEN_KEY_MARKERS = ("receipt", "audit", "provenance", "source_commit")
_DIRECT_CALIBRATION_METADATA_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "protocol_version",
        "branch_id",
        "epoch",
        "roles",
        "parents",
    }
)
_DIRECT_CALIBRATION_ROLE_FIELDS = frozenset({"keys", "shapes"})
_DIRECT_CALIBRATION_PARENT_FIELDS = frozenset({"unguided_planner", "field"})
_DIRECT_EPDMS_GATE_FIELDS = frozenset(
    {
        "schema",
        "feature_schema",
        "latent_dim",
        "hidden_dim",
        "feature_dim",
        "lambda_grid",
        "provenance",
        "state_dict",
    }
)
_DIRECT_EPDMS_IDENTITIES = {
    "full": ("p1_full", "calibration_full", "stop_full", "p1_full", "full"),
    "no_cf": ("p1_no_cf", "calibration_no_cf", "stop_no_cf", "p1_no_cf", "full"),
    "hazard_only": (
        "p1_hazard_only",
        "calibration_hazard_only",
        None,
        None,
        None,
    ),
    "quality_only": (
        "p1_quality_only",
        "calibration_quality_only",
        None,
        None,
        None,
    ),
    "without_field": ("p1_full", "calibration_full", "stop_full", "p1_full", "without_field"),
    "without_stop": ("p1_full", "calibration_full", "stop_full", "p1_full", "without_stop"),
    "without_value_summary": (
        "p1_full",
        "calibration_full",
        "stop_full",
        "p1_full",
        "without_value_summary",
    ),
}


@dataclass(frozen=True)
class CvoiDirectEpdmsArtifactIdentity:
    """Expected structural identities for one retained direct EPDMS branch."""

    p0_branch_id: str
    p1_branch_id: str
    calibration_branch_id: str
    stop_branch_id: str | None
    oracle_lineage: str | None
    gate_feature_mode: str | None


def resolve_cvoi_direct_epdms_artifact_identity(
    branch: str,
    *,
    evaluation_mode: str,
) -> CvoiDirectEpdmsArtifactIdentity:
    """Resolve exact checkpoint metadata without inferring identity from paths."""

    if type(branch) is not str or branch not in _DIRECT_EPDMS_IDENTITIES:
        raise ValueError(f"direct EPDMS branch must be one of {sorted(_DIRECT_EPDMS_IDENTITIES)!r}, got {branch!r}")
    if evaluation_mode not in {"controller", "p0_forced", "p1_field_forced"} or type(evaluation_mode) is not str:
        raise ValueError(f"unsupported direct EPDMS evaluation_mode: {evaluation_mode!r}")
    p1_branch_id, calibration_branch_id, stop_branch_id, oracle_lineage, gate_feature_mode = _DIRECT_EPDMS_IDENTITIES[
        branch
    ]
    controller_branch = stop_branch_id is not None
    if evaluation_mode == "controller" and not controller_branch:
        raise ValueError(f"direct EPDMS branch {branch!r} does not define a controller")
    if evaluation_mode != "controller" and controller_branch:
        raise ValueError(f"direct EPDMS branch {branch!r} does not define a forced-horizon run")
    return CvoiDirectEpdmsArtifactIdentity(
        p0_branch_id="p0_uniform",
        p1_branch_id=p1_branch_id,
        calibration_branch_id=calibration_branch_id,
        stop_branch_id=stop_branch_id if evaluation_mode == "controller" else None,
        oracle_lineage=oracle_lineage if evaluation_mode == "controller" else None,
        gate_feature_mode=gate_feature_mode if evaluation_mode == "controller" else None,
    )


@dataclass
class FormalV2NavSimE120HorizonExposureState:
    """Cumulative successful-optimizer exposure for one resumable Planner run."""

    prior: Mapping[int, int] = field(default_factory=lambda: {horizon: 0 for horizon in FORMAL_V2_NAVSIM_HORIZONS})
    local: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.prior = _validate_histogram_allow_empty(self.prior, name="prior horizon histogram")
        self.local = {horizon: 0 for horizon in FORMAL_V2_NAVSIM_HORIZONS}

    def record(self, *, horizon: int, batch_size: int) -> None:
        """Record samples only after the caller has completed an optimizer step."""

        if type(horizon) is not int or horizon not in self.local:
            raise ValueError(
                f"NavSim e120 optimized horizon must be one of {list(FORMAL_V2_NAVSIM_HORIZONS)}, got {horizon!r}"
            )
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError(f"NavSim e120 optimized batch_size must be a positive integer, got {batch_size!r}")
        self.local[horizon] += batch_size

    def reset_local(self) -> None:
        """Clear current-process counts without changing the resume baseline."""

        self.local = {horizon: 0 for horizon in FORMAL_V2_NAVSIM_HORIZONS}

    def snapshot(self, *, device: torch.device) -> dict[int, int]:
        """Return prior plus the all-reduced successful exposure on every rank."""

        counts = torch.tensor(
            [self.local[horizon] for horizon in FORMAL_V2_NAVSIM_HORIZONS], dtype=torch.long, device=device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
        return {horizon: self.prior[horizon] + int(counts[horizon].item()) for horizon in FORMAL_V2_NAVSIM_HORIZONS}


def _require_exact_fields(value: Mapping[str, object], fields: frozenset[str], *, name: str) -> None:
    missing = sorted(fields - set(value), key=repr)
    unknown = sorted(set(value) - fields, key=repr)
    if missing or unknown:
        raise ValueError(f"invalid {name} fields: missing={missing}, unknown={unknown}")


def _require_identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase snake-case identifier")
    return value


def _require_stage(value: object) -> str:
    if type(value) is not str or value not in {"p0", "p1"}:
        raise ValueError(f"stage must be exactly 'p0' or 'p1', got {value!r}")
    return value


def _reject_direct_proof_keys(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                lowered = key.lower()
                is_sha_key = (
                    lowered == "sha" or "sha256" in lowered or lowered.startswith("sha_") or lowered.endswith("_sha")
                )
                if is_sha_key or any(marker in lowered for marker in _DIRECT_FORBIDDEN_KEY_MARKERS):
                    raise ValueError(f"direct structural payload contains forbidden proof key {path}.{key}")
            _reject_direct_proof_keys(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_direct_proof_keys(nested, path=f"{path}[{index}]")


def build_formal_v2_navsim_e120_direct_lineage(
    *,
    stage: str,
    branch_id: str,
) -> dict[str, object]:
    """Build proof-free structural lineage for one manually operated Planner stage."""

    return validate_formal_v2_navsim_e120_direct_lineage(
        {
            "schema": FORMAL_V2_NAVSIM_E120_DIRECT_LINEAGE_SCHEMA,
            "protocol_version": FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
            "stage": stage,
            "branch_id": branch_id,
        }
    )


def validate_formal_v2_navsim_e120_direct_lineage(
    value: object,
    *,
    expected_stage: str | None = None,
) -> dict[str, object]:
    """Validate the exact stage/branch identity used by direct checkpoints."""

    if not isinstance(value, Mapping):
        raise ValueError("direct NavSim e120 lineage must be a mapping")
    _require_exact_fields(
        value,
        FORMAL_V2_NAVSIM_E120_DIRECT_LINEAGE_FIELDS,
        name="direct NavSim e120 lineage",
    )
    _reject_direct_proof_keys(value, path="lineage")
    if value["schema"] != FORMAL_V2_NAVSIM_E120_DIRECT_LINEAGE_SCHEMA:
        raise ValueError(f"direct lineage.schema must be {FORMAL_V2_NAVSIM_E120_DIRECT_LINEAGE_SCHEMA!r}")
    if value["protocol_version"] != FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION:
        raise ValueError(f"direct lineage.protocol_version must be {FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION!r}")
    stage = _require_stage(value["stage"])
    if expected_stage is not None and stage != _require_stage(expected_stage):
        raise ValueError(f"direct lineage stage mismatch: expected={expected_stage!r}, actual={stage!r}")
    branch_id = _require_identifier(value["branch_id"], name="lineage.branch_id")
    allowed_branches = _DIRECT_BRANCHES_BY_STAGE[stage]
    if branch_id not in allowed_branches:
        raise ValueError(
            f"direct {stage.upper()} lineage.branch_id must be one of "
            f"{sorted(allowed_branches)!r}, got {branch_id!r}"
        )
    return {
        "schema": FORMAL_V2_NAVSIM_E120_DIRECT_LINEAGE_SCHEMA,
        "protocol_version": FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
        "stage": stage,
        "branch_id": branch_id,
    }


def resolve_formal_v2_navsim_e120_selected_checkpoint(
    path: str | Path,
    *,
    results_root: str | Path,
    stage: str,
) -> Path:
    """Resolve one fixed P0/P1 handoff copy or in-stage symlink target."""

    normalized_stage = _require_stage(stage)
    root = Path(results_root)
    if not root.is_absolute():
        raise ValueError(f"results_root path must be absolute: {root}")
    configured = Path(path)
    if not configured.is_absolute():
        raise ValueError(f"selected checkpoint path must be absolute: {configured}")
    expected = root / "handoff" / f"{normalized_stage}_selected.pt"
    if str(configured) != str(expected):
        raise ValueError(
            f"selected {normalized_stage.upper()} checkpoint path must be exactly {expected}, got {configured}"
        )

    if configured.is_symlink():
        try:
            target = configured.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"selected {normalized_stage.upper()} checkpoint symlink is broken: {configured}"
            ) from error
        except RuntimeError as error:
            raise ValueError(
                f"selected {normalized_stage.upper()} checkpoint symlink is cyclic: {configured}"
            ) from error
        if not target.is_file():
            raise ValueError(
                f"selected {normalized_stage.upper()} checkpoint symlink target must be a regular file: {target}"
            )
        stage_root = (root / normalized_stage).resolve(strict=False)
        try:
            target.relative_to(stage_root)
        except ValueError as error:
            raise ValueError(
                f"selected {normalized_stage.upper()} checkpoint symlink target is outside {stage_root}: {target}"
            ) from error
        return target

    if not configured.exists():
        raise FileNotFoundError(f"selected {normalized_stage.upper()} checkpoint does not exist: {configured}")
    if not configured.is_file():
        raise ValueError(f"selected {normalized_stage.upper()} checkpoint must be a regular file: {configured}")
    return configured


def _normalize_state(value: object, *, name: str, clone: bool) -> dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty tensor mapping")
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(tensor):
            raise ValueError(f"{name} must map string keys to tensors")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not key or key in normalized:
            raise ValueError(f"{name} contains duplicate or empty normalized key {key!r}")
        normalized[key] = (
            tensor.detach().cpu().clone(memory_format=torch.preserve_format) if clone else tensor.detach()
        )
    return normalized


def _module_core(module: object, *, role: str) -> torch.nn.Module:
    if not isinstance(module, torch.nn.Module):
        raise ValueError(f"modules[{role!r}] must be a torch.nn.Module")
    core = getattr(module, "module", module)
    if not isinstance(core, torch.nn.Module):
        raise ValueError(f"modules[{role!r}].module must be a torch.nn.Module")
    return core


def _require_modules(modules: object) -> dict[str, torch.nn.Module]:
    if not isinstance(modules, Mapping):
        raise ValueError("modules must be a mapping")
    if set(modules) != set(MODEL_ROLES) or any(not isinstance(key, str) for key in modules):
        raise ValueError(f"modules must contain exactly {list(MODEL_ROLES)!r}")
    return {role: _module_core(modules[role], role=role) for role in MODEL_ROLES}


def _normalize_shape_mapping(value: object, *, name: str) -> dict[str, list[int]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized: dict[str, list[int]] = {}
    for key, shape in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if type(shape) is not list or any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError(f"{name}.{key} must be a list of non-negative integer dimensions")
        normalized[key] = list(shape)
    return normalized


def _normalize_training_state(value: object, *, name: str, require_nonempty: bool) -> dict[str, object]:
    if not isinstance(value, Mapping) or (require_nonempty and not value):
        qualifier = "non-empty " if require_nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}mapping")
    return copy.deepcopy(dict(value))


def _validate_training_state_mapping(value: object, *, name: str, require_nonempty: bool) -> None:
    if not isinstance(value, Mapping) or (require_nonempty and not value):
        qualifier = "non-empty " if require_nonempty else ""
        raise ValueError(f"{name} must be a {qualifier}mapping")


def _validate_histogram_allow_empty(value: object, *, name: str) -> dict[int, int]:
    if not isinstance(value, Mapping) or set(value) != set(FORMAL_V2_NAVSIM_HORIZONS):
        raise ValueError(f"{name} must contain exactly integer H0-H4")
    normalized: dict[int, int] = {}
    for horizon in FORMAL_V2_NAVSIM_HORIZONS:
        count = value[horizon]
        if type(count) is not int or count < 0:
            raise ValueError(f"{name}[{horizon}] must be a non-negative integer")
        normalized[horizon] = count
    return normalized


def _validate_histogram(value: object) -> dict[int, int]:
    normalized = _validate_histogram_allow_empty(value, name="cumulative_horizon_histogram")
    if sum(normalized.values()) <= 0:
        raise ValueError("cumulative_horizon_histogram must record at least one sample")
    return normalized


def _direct_role_state_shapes_from_states(states: Mapping[str, object]) -> dict[str, dict[str, list[int]]]:
    return {
        role: {
            key: list(tensor.shape)
            for key, tensor in sorted(_normalize_state(states[role], name=f"{role} state", clone=False).items())
        }
        for role in MODEL_ROLES
    }


def _validate_direct_role_state_shapes(
    value: object,
    *,
    states: Mapping[str, object],
) -> dict[str, dict[str, list[int]]]:
    if not isinstance(value, Mapping) or set(value) != set(MODEL_ROLES):
        raise ValueError(f"direct role_state_shapes roles must contain exactly {list(MODEL_ROLES)!r}")
    normalized = {
        role: _normalize_shape_mapping(value[role], name=f"role_state_shapes.{role}") for role in MODEL_ROLES
    }
    expected = _direct_role_state_shapes_from_states(states)
    for role in MODEL_ROLES:
        if set(normalized[role]) != set(expected[role]):
            raise ValueError(f"direct role_state_shapes {role} state keys mismatch")
        if normalized[role] != expected[role]:
            raise ValueError(f"direct role_state_shapes {role} state shapes mismatch")
    return normalized


def _expected_direct_schedule(
    stage: str,
    *,
    training_stop_epoch: object,
) -> tuple[int, int, tuple[int, ...]]:
    normalized_stage = _require_stage(stage)
    if normalized_stage == "p1":
        if type(training_stop_epoch) is not int or training_stop_epoch != 80:
            raise ValueError("P1 training_stop_epoch must be exactly 80")
        return 80, 80, FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS
    if type(training_stop_epoch) is not int or training_stop_epoch != 50:
        raise ValueError("Uniform P0 training_stop_epoch must be exactly 50")
    return 50, 50, FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS


def build_formal_v2_navsim_e120_direct_checkpoint(
    *,
    modules: object,
    optimizer: object,
    scaler: object,
    scheduler: object,
    wd_scheduler: object,
    run_id: str,
    stage: str,
    epoch: int,
    training_stop_epoch: int,
    schedule_epochs: int,
    selection_checkpoint_epochs: tuple[int, ...],
    cumulative_horizon_histogram: object,
    lineage: object,
) -> dict[str, object]:
    """Build a proof-free full-state checkpoint for manual same-run resume."""

    normalized_modules = _require_modules(modules)
    states = {
        role: _normalize_state(normalized_modules[role].state_dict(), name=f"{role} state", clone=True)
        for role in MODEL_ROLES
    }
    payload: dict[str, object] = {
        "schema": FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_SCHEMA,
        "protocol_version": FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
        "resume_contract": FORMAL_V2_NAVSIM_E120_DIRECT_RESUME_CONTRACT,
        "run_id": run_id,
        "stage": stage,
        **states,
        "optimizer": optimizer,
        "scaler": scaler,
        "scheduler": scheduler,
        "wd_scheduler": wd_scheduler,
        "epoch": epoch,
        "training_stop_epoch": training_stop_epoch,
        "schedule_epochs": schedule_epochs,
        "selection_checkpoint_epochs": selection_checkpoint_epochs,
        "cumulative_horizon_histogram": cumulative_horizon_histogram,
        "role_state_shapes": _direct_role_state_shapes_from_states(states),
        "lineage": lineage,
    }
    return validate_formal_v2_navsim_e120_direct_checkpoint(payload)


def _validate_formal_v2_navsim_e120_direct_checkpoint(
    payload: object,
    *,
    clone_model_states: bool,
    include_training_states: bool,
) -> dict[str, object]:
    """Validate the exact envelope with selectable resume/deployment retention."""

    if not isinstance(payload, Mapping):
        raise ValueError("direct NavSim e120 checkpoint must be a mapping")
    _require_exact_fields(
        payload,
        FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_FIELDS,
        name="direct NavSim e120 checkpoint",
    )
    _reject_direct_proof_keys(payload, path="checkpoint")
    if payload["schema"] != FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_SCHEMA:
        raise ValueError(f"direct checkpoint schema must be {FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_SCHEMA!r}")
    if payload["protocol_version"] != FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION:
        raise ValueError(f"direct checkpoint protocol_version must be {FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION!r}")
    if payload["resume_contract"] != FORMAL_V2_NAVSIM_E120_DIRECT_RESUME_CONTRACT:
        raise ValueError(f"direct checkpoint resume_contract must be {FORMAL_V2_NAVSIM_E120_DIRECT_RESUME_CONTRACT!r}")
    stage = _require_stage(payload["stage"])
    lineage = validate_formal_v2_navsim_e120_direct_lineage(payload["lineage"], expected_stage=stage)
    run_id = _require_identifier(payload["run_id"], name="run_id")
    states = {
        role: _normalize_state(
            payload[role],
            name=f"direct checkpoint {role}",
            clone=clone_model_states,
        )
        for role in MODEL_ROLES
    }
    role_shapes = _validate_direct_role_state_shapes(payload["role_state_shapes"], states=states)
    expected_schedule, stop_epoch, expected_candidates = _expected_direct_schedule(
        stage,
        training_stop_epoch=payload["training_stop_epoch"],
    )
    epoch = payload["epoch"]
    if type(epoch) is not int or epoch <= 0 or epoch > stop_epoch:
        raise ValueError(
            f"direct checkpoint epoch must be positive and no greater than training_stop_epoch={stop_epoch}"
        )
    if type(payload["schedule_epochs"]) is not int or payload["schedule_epochs"] != expected_schedule:
        raise ValueError(f"direct {stage.upper()} schedule_epochs must be exactly {expected_schedule}")
    candidates = payload["selection_checkpoint_epochs"]
    if type(candidates) is not tuple or candidates != expected_candidates:
        raise ValueError(f"direct {stage.upper()} selection_checkpoint_epochs must be exactly {expected_candidates!r}")
    histogram = _validate_histogram(payload["cumulative_horizon_histogram"])
    training_state_requirements = {
        "optimizer": True,
        "scaler": False,
        "scheduler": True,
        "wd_scheduler": True,
    }
    for name, require_nonempty in training_state_requirements.items():
        _validate_training_state_mapping(
            payload[name],
            name=name,
            require_nonempty=require_nonempty,
        )
    normalized = {
        "schema": FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_SCHEMA,
        "protocol_version": FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
        "resume_contract": FORMAL_V2_NAVSIM_E120_DIRECT_RESUME_CONTRACT,
        "run_id": run_id,
        "stage": stage,
        **states,
        "epoch": epoch,
        "training_stop_epoch": stop_epoch,
        "schedule_epochs": expected_schedule,
        "selection_checkpoint_epochs": candidates,
        "cumulative_horizon_histogram": histogram,
        "role_state_shapes": role_shapes,
        "lineage": lineage,
    }
    if include_training_states:
        normalized.update(
            {
                name: _normalize_training_state(
                    payload[name],
                    name=name,
                    require_nonempty=require_nonempty,
                )
                for name, require_nonempty in training_state_requirements.items()
            }
        )
    return normalized


def validate_formal_v2_navsim_e120_direct_checkpoint(payload: object) -> dict[str, object]:
    """Validate and clone an exact structural checkpoint for training/resume."""

    return _validate_formal_v2_navsim_e120_direct_checkpoint(
        payload,
        clone_model_states=True,
        include_training_states=True,
    )


def validate_cvoi_direct_epdms_planner_checkpoint(
    payload: object,
    *,
    expected_stage: str,
    expected_branch_id: str,
) -> dict[str, object]:
    """Validate one exact deployment envelope without cloning tensor/training state."""

    normalized = _validate_formal_v2_navsim_e120_direct_checkpoint(
        payload,
        clone_model_states=False,
        include_training_states=False,
    )
    stage = _require_stage(expected_stage)
    branch_id = _require_identifier(expected_branch_id, name="expected_branch_id")
    if normalized["stage"] != stage:
        raise ValueError(f"direct EPDMS Planner stage mismatch: expected={stage!r}, actual={normalized['stage']!r}")
    lineage = normalized["lineage"]
    if lineage["stage"] != stage or lineage["branch_id"] != branch_id:
        raise ValueError(
            "direct EPDMS Planner lineage mismatch: "
            f"expected={{'stage': {stage!r}, 'branch_id': {branch_id!r}}}, "
            f"actual={{'stage': {lineage['stage']!r}, 'branch_id': {lineage['branch_id']!r}}}"
        )
    return {
        "stage": normalized["stage"],
        "lineage": normalized["lineage"],
        "protocol_version": normalized["protocol_version"],
        "role_state_shapes": normalized["role_state_shapes"],
        **{role: normalized[role] for role in MODEL_ROLES},
    }


def _require_state_loader(value: object, *, name: str) -> Callable[[object], object]:
    loader = getattr(value, "load_state_dict", None)
    if not callable(loader):
        raise ValueError(f"{name} must provide load_state_dict() for a full same-run resume")
    return loader


def prepare_formal_v2_navsim_e120_direct_same_run_resume(
    payload: object,
    *,
    expected_run_id: str,
    expected_stage: str,
    expected_lineage: object,
    expected_training_stop_epoch: int,
    warmstart_requested: bool,
    model_only: bool,
) -> dict[str, object]:
    """Prevalidate a proof-free same-run resume and forbid initialization replay."""

    if type(warmstart_requested) is not bool:
        raise ValueError("warmstart_requested must be boolean")
    if warmstart_requested:
        raise ValueError("direct same-run resume forbids re-warmstart")
    if type(model_only) is not bool:
        raise ValueError("model_only must be boolean")
    if model_only:
        raise ValueError("direct same-run resume forbids model-only restore")
    normalized = validate_formal_v2_navsim_e120_direct_checkpoint(payload)
    expected_run = _require_identifier(expected_run_id, name="expected_run_id")
    if normalized["run_id"] != expected_run:
        raise ValueError(
            f"direct cross-run resume rejected: expected={expected_run!r}, actual={normalized['run_id']!r}"
        )
    stage = _require_stage(expected_stage)
    if normalized["stage"] != stage:
        raise ValueError(f"direct cross-stage resume rejected: expected={stage!r}, actual={normalized['stage']!r}")
    if (
        type(expected_training_stop_epoch) is not int
        or normalized["training_stop_epoch"] != expected_training_stop_epoch
    ):
        raise ValueError(
            "direct same-run resume training_stop_epoch mismatch: "
            f"expected={expected_training_stop_epoch!r}, actual={normalized['training_stop_epoch']!r}"
        )
    lineage = validate_formal_v2_navsim_e120_direct_lineage(expected_lineage, expected_stage=stage)
    if normalized["lineage"] != lineage:
        raise ValueError("direct same-run resume lineage does not exactly match configured lineage")
    return normalized


def restore_formal_v2_navsim_e120_direct_same_run_resume(
    payload: object,
    *,
    modules: object,
    optimizer: object,
    scaler: object,
    scheduler: object,
    wd_scheduler: object,
    expected_run_id: str,
    expected_stage: str,
    expected_lineage: object,
    expected_training_stop_epoch: int,
    warmstart_requested: bool,
    model_only: bool,
) -> dict[str, object]:
    """Strictly restore all model/training roles from one direct same-run checkpoint."""

    normalized = prepare_formal_v2_navsim_e120_direct_same_run_resume(
        payload,
        expected_run_id=expected_run_id,
        expected_stage=expected_stage,
        expected_lineage=expected_lineage,
        expected_training_stop_epoch=expected_training_stop_epoch,
        warmstart_requested=warmstart_requested,
        model_only=model_only,
    )
    normalized_modules = _require_modules(modules)
    state_loaders = {
        "optimizer": _require_state_loader(optimizer, name="optimizer"),
        "scaler": _require_state_loader(scaler, name="scaler"),
        "scheduler": _require_state_loader(scheduler, name="scheduler"),
        "wd_scheduler": _require_state_loader(wd_scheduler, name="wd_scheduler"),
    }
    target_states = {role: normalized_modules[role].state_dict() for role in MODEL_ROLES}
    _validate_direct_role_state_shapes(normalized["role_state_shapes"], states=target_states)
    for role in MODEL_ROLES:
        _load_prepared_role(normalized_modules[role], normalized[role], role=role)
    restored_shapes = _direct_role_state_shapes_from_states(
        {role: normalized_modules[role].state_dict() for role in MODEL_ROLES}
    )
    if restored_shapes != normalized["role_state_shapes"]:
        raise RuntimeError("direct same-run resume role structure changed during strict load")
    for name, loader in state_loaders.items():
        loader(copy.deepcopy(normalized[name]))
    return {
        "start_epoch": normalized["epoch"],
        "run_id": normalized["run_id"],
        "stage": normalized["stage"],
        "training_stop_epoch": normalized["training_stop_epoch"],
        "schedule_epochs": normalized["schedule_epochs"],
        "selection_checkpoint_epochs": normalized["selection_checkpoint_epochs"],
        "cumulative_horizon_histogram": normalized["cumulative_horizon_histogram"],
        "role_state_shapes": restored_shapes,
        "lineage": normalized["lineage"],
    }


def read_formal_v2_navsim_e120_direct_checkpoint(path: str | Path) -> dict[str, object]:
    """Load one ordinary absolute checkpoint path and validate its structural envelope."""

    artifact = Path(path)
    if not artifact.is_absolute():
        raise ValueError(f"direct NavSim e120 checkpoint path must be absolute: {artifact}")
    if not artifact.exists():
        raise FileNotFoundError(f"direct NavSim e120 checkpoint does not exist: {artifact}")
    if not artifact.is_file():
        raise ValueError(f"direct NavSim e120 checkpoint must be a regular file: {artifact}")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    return validate_formal_v2_navsim_e120_direct_checkpoint(payload)


def read_cvoi_direct_epdms_planner_checkpoint(
    path: str | Path,
    *,
    expected_stage: str,
    expected_branch_id: str,
) -> dict[str, object]:
    """Read and validate one direct Planner envelope exactly once."""

    artifact = Path(path)
    if not artifact.is_absolute():
        raise ValueError(f"direct EPDMS Planner checkpoint path must be absolute: {artifact}")
    if artifact.is_symlink():
        raise ValueError(f"direct EPDMS Planner checkpoint must be a non-symlink regular file: {artifact}")
    if not artifact.exists():
        raise FileNotFoundError(f"direct EPDMS Planner checkpoint does not exist: {artifact}")
    if not artifact.is_file():
        raise ValueError(f"direct EPDMS Planner checkpoint must be a non-symlink regular file: {artifact}")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    return validate_cvoi_direct_epdms_planner_checkpoint(
        payload,
        expected_stage=expected_stage,
        expected_branch_id=expected_branch_id,
    )


def validate_cvoi_direct_epdms_gate_checkpoint(
    payload: object,
    *,
    branch: str,
    oracle_sha256: str,
    gate_feature_mode: str,
) -> dict[str, object]:
    """Validate Gate structure against a projected Oracle identity only."""

    identity = resolve_cvoi_direct_epdms_artifact_identity(branch, evaluation_mode="controller")
    if type(oracle_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", oracle_sha256) is None:
        raise ValueError("direct EPDMS projected Oracle identity must be one lowercase SHA-256")
    if gate_feature_mode != identity.gate_feature_mode:
        raise ValueError(
            f"direct EPDMS branch {branch!r} requires gate_feature_mode={identity.gate_feature_mode!r}, "
            f"got {gate_feature_mode!r}"
        )
    if not isinstance(payload, Mapping):
        raise ValueError("direct EPDMS Gate checkpoint must be a mapping")
    _require_exact_fields(payload, _DIRECT_EPDMS_GATE_FIELDS, name="direct EPDMS Gate checkpoint")
    if payload["schema"] != SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_NAVSIM_E120:
        raise ValueError(
            "direct EPDMS Gate schema must be "
            f"{SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_NAVSIM_E120!r}, got {payload['schema']!r}"
        )
    if payload["feature_schema"] != CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA:
        raise ValueError("direct EPDMS Gate feature schema differs from the e120 runtime")
    latent_dim = payload["latent_dim"]
    hidden_dim = payload["hidden_dim"]
    feature_dim = payload["feature_dim"]
    if type(latent_dim) is not int or latent_dim <= 0:
        raise ValueError("direct EPDMS Gate latent_dim must be a positive integer")
    if type(hidden_dim) is not int or hidden_dim <= 0:
        raise ValueError("direct EPDMS Gate hidden_dim must be a positive integer")
    if type(feature_dim) is not int or feature_dim != 2 * latent_dim + 7:
        raise ValueError("direct EPDMS Gate feature_dim is inconsistent with its architecture")
    lambda_grid = validate_formal_v2_lambda_grid(payload["lambda_grid"])

    provenance = payload["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or not provenance
        or any(
            type(key) is not str or not key or type(value) is not str or not value for key, value in provenance.items()
        )
    ):
        raise ValueError("direct EPDMS Gate provenance must be a non-empty string mapping")
    expected_provenance = {
        "gate_pipeline": CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION,
        "oracle_protocol": NAVTRAIN_GATE_PROTOCOL_ID,
        "oracle_sha256": oracle_sha256,
        "oracle_lineage": identity.oracle_lineage,
        "gate_feature_schema": CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
        "gate_feature_mode": gate_feature_mode,
    }
    if set(provenance) != set(expected_provenance):
        raise ValueError(
            "direct EPDMS Gate provenance fields mismatch: "
            f"missing={sorted(set(expected_provenance) - set(provenance))}, "
            f"unexpected={sorted(set(provenance) - set(expected_provenance))}"
        )
    drifted = {
        field: (provenance.get(field), expected)
        for field, expected in expected_provenance.items()
        if provenance.get(field) != expected
    }
    if drifted:
        raise ValueError(f"direct EPDMS Gate projected provenance mismatch: {drifted}")

    state = payload["state_dict"]
    if (
        not isinstance(state, Mapping)
        or not state
        or any(type(key) is not str or not key or not torch.is_tensor(value) for key, value in state.items())
    ):
        raise ValueError("direct EPDMS Gate state_dict must map non-empty string keys to tensors")
    with torch.random.fork_rng(devices=[]):
        probe = SequentialRolloutGate(latent_dim=latent_dim, hidden_dim=hidden_dim)
        probe.load_state_dict(dict(state), strict=True)
    return {
        "schema": SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_NAVSIM_E120,
        "feature_schema": CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "feature_dim": feature_dim,
        "lambda_grid": lambda_grid,
        "provenance": dict(provenance),
        "state_dict": dict(state),
    }


def read_cvoi_direct_epdms_gate_checkpoint(
    path: str | Path,
    *,
    branch: str,
    oracle_sha256: str,
    gate_feature_mode: str,
) -> dict[str, object]:
    """Read one ordinary Gate handoff without receiving an Oracle path."""

    artifact = Path(path)
    if not artifact.is_absolute():
        raise ValueError(f"direct EPDMS Gate checkpoint path must be absolute: {artifact}")
    if artifact.is_symlink():
        raise ValueError(f"direct EPDMS Gate checkpoint must be a non-symlink regular file: {artifact}")
    if not artifact.exists():
        raise FileNotFoundError(f"direct EPDMS Gate checkpoint does not exist: {artifact}")
    if not artifact.is_file():
        raise ValueError(f"direct EPDMS Gate checkpoint must be a non-symlink regular file: {artifact}")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    return validate_cvoi_direct_epdms_gate_checkpoint(
        payload,
        branch=branch,
        oracle_sha256=oracle_sha256,
        gate_feature_mode=gate_feature_mode,
    )


def build_cvoi_direct_epdms_gate(
    payload: object,
    *,
    branch: str,
    oracle_sha256: str,
    gate_feature_mode: str,
    expected_latent_dim: int,
    device: torch.device,
) -> SequentialRolloutGate:
    """Instantiate and freeze one already preflighted direct Gate payload."""

    normalized = validate_cvoi_direct_epdms_gate_checkpoint(
        payload,
        branch=branch,
        oracle_sha256=oracle_sha256,
        gate_feature_mode=gate_feature_mode,
    )
    if type(expected_latent_dim) is not int or expected_latent_dim <= 0:
        raise ValueError("direct EPDMS expected Gate latent_dim must be a positive integer")
    if normalized["latent_dim"] != expected_latent_dim:
        raise ValueError(
            "direct EPDMS Gate latent_dim differs from the encoder: "
            f"expected={expected_latent_dim}, actual={normalized['latent_dim']}"
        )
    gate = SequentialRolloutGate(
        latent_dim=int(normalized["latent_dim"]),
        hidden_dim=int(normalized["hidden_dim"]),
    )
    gate.load_state_dict(normalized["state_dict"], strict=True)
    gate.to(device=device)
    gate.eval()
    gate.requires_grad_(False)
    if gate.training or any(parameter.requires_grad for parameter in gate.parameters()):
        raise RuntimeError("direct EPDMS Gate must be frozen in eval mode")
    return gate


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"checkpoint parent must be a directory: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_checkpoint_output(path: str | Path) -> Path:
    output = Path(path)
    if not output.is_absolute() or ".." in output.parts:
        raise ValueError(f"NavSim e120 checkpoint output must be a normalized absolute path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.resolve(strict=True) != output.parent or output.parent.is_symlink():
        raise ValueError(f"NavSim e120 checkpoint parent must be a canonical non-symlink directory: {output.parent}")
    try:
        metadata = os.lstat(output)
    except FileNotFoundError:
        return output
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"NavSim e120 checkpoint output must not be a symlink: {output}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"NavSim e120 checkpoint output must be a regular file: {output}")
    return output


def _write_validated_checkpoint(
    path: str | Path,
    normalized: Mapping[str, object],
    *,
    replace: bool,
) -> Path:
    output = _prepare_checkpoint_output(path)
    if output.exists() and not replace:
        raise FileExistsError(f"immutable NavSim e120 checkpoint already exists: {output}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(normalized, handle)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, output)
        else:
            try:
                os.link(temporary, output)
            except FileExistsError as error:
                raise FileExistsError(f"immutable NavSim e120 checkpoint already exists: {output}") from error
        _fsync_directory(output.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_formal_v2_navsim_e120_direct_checkpoint(
    path: str | Path,
    payload: object,
    *,
    replace: bool,
) -> Path:
    """Atomically publish one validated proof-free milestone or same-run latest file."""

    if type(replace) is not bool:
        raise ValueError("replace must be boolean")
    normalized = validate_formal_v2_navsim_e120_direct_checkpoint(payload)
    return _write_validated_checkpoint(path, normalized, replace=replace)


def _load_prepared_role(module: torch.nn.Module, state: Mapping[str, torch.Tensor], *, role: str) -> None:
    raw_state = module.state_dict()
    raw_by_normalized: dict[str, str] = {}
    for raw_key in raw_state:
        normalized_key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if normalized_key in raw_by_normalized:
            raise ValueError(f"target {role} contains duplicate normalized key {normalized_key!r}")
        raw_by_normalized[normalized_key] = raw_key
    module.load_state_dict({raw_by_normalized[key]: tensor for key, tensor in state.items()}, strict=True)


def _require_absolute_regular_direct_artifact(path: str | Path, *, name: str) -> Path:
    artifact = Path(path)
    if not artifact.is_absolute():
        raise ValueError(f"{name} path must be absolute: {artifact}")
    if not artifact.exists():
        raise FileNotFoundError(f"{name} does not exist: {artifact}")
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError(f"{name} must be a regular file: {artifact}")
    return artifact


def _validate_direct_calibration_checkpoint_metadata(
    value: object,
    *,
    required_branch_id: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("direct Calibration checkpoint validator must return a metadata mapping")
    _require_exact_fields(
        value,
        _DIRECT_CALIBRATION_METADATA_FIELDS,
        name="direct Calibration checkpoint metadata",
    )
    if value["schema"] != "cvoi_dual_value_navsim_e120_v1":
        raise ValueError("direct Calibration metadata schema must be 'cvoi_dual_value_navsim_e120_v1'")
    if value["protocol_version"] != FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION:
        raise ValueError(
            f"direct Calibration metadata protocol_version must be " f"{FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION!r}"
        )
    if value["phase"] != "field_calibrated":
        raise ValueError("direct Calibration metadata phase must be 'field_calibrated'")
    calibration_lineage = _manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase="field_calibrated",
        branch_id=required_branch_id,
    )
    expected_parents = _manual_lineage.build_cvoi_manual_value_parents(
        calibration_lineage,
        "field_calibrated",
    )
    if value["branch_id"] != required_branch_id:
        raise ValueError(f"direct Calibration metadata branch_id must be {required_branch_id!r}")
    epoch = value["epoch"]
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("direct Calibration metadata epoch must be a positive integer")

    roles = value["roles"]
    if not isinstance(roles, Mapping) or set(roles) != {"value_model"}:
        raise ValueError("direct Calibration metadata roles must contain exactly 'value_model'")
    value_model = roles["value_model"]
    if not isinstance(value_model, Mapping):
        raise ValueError("direct Calibration metadata roles.value_model must be a mapping")
    _require_exact_fields(
        value_model,
        _DIRECT_CALIBRATION_ROLE_FIELDS,
        name="direct Calibration metadata roles.value_model",
    )
    keys = value_model["keys"]
    if type(keys) is not list or not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("direct Calibration value_model keys must be a non-empty list of strings")
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("direct Calibration value_model keys must be sorted and unique")
    shapes = value_model["shapes"]
    if not isinstance(shapes, Mapping) or set(shapes) != set(keys):
        raise ValueError("direct Calibration value_model shapes keys must exactly match keys")
    normalized_shapes: dict[str, list[int]] = {}
    for key in keys:
        shape = shapes[key]
        if type(shape) is not list or any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError(
                f"direct Calibration value_model shapes.{key} must contain non-negative integer dimensions"
            )
        normalized_shapes[key] = list(shape)

    parents = value["parents"]
    if not isinstance(parents, Mapping):
        raise ValueError("direct Calibration metadata parents must be a mapping")
    _require_exact_fields(
        parents,
        _DIRECT_CALIBRATION_PARENT_FIELDS,
        name="direct Calibration metadata parents",
    )
    unguided = parents["unguided_planner"]
    if not isinstance(unguided, Mapping):
        raise ValueError("direct Calibration unguided_planner parent must be a mapping")
    _require_exact_fields(
        unguided,
        frozenset({"stage", "branch_id"}),
        name="direct Calibration unguided_planner parent",
    )
    if dict(unguided) != expected_parents["unguided_planner"]:
        raise ValueError(
            "direct Calibration unguided_planner parent must be exactly " f"{expected_parents['unguided_planner']!r}"
        )
    field_parent = parents["field"]
    if not isinstance(field_parent, Mapping):
        raise ValueError("direct Calibration Field parent must be a mapping")
    _require_exact_fields(
        field_parent,
        frozenset({"phase", "branch_id"}),
        name="direct Calibration Field parent",
    )
    if dict(field_parent) != expected_parents["field"]:
        raise ValueError(f"direct Calibration Field parent must be exactly {expected_parents['field']!r}")
    return {
        "schema": "cvoi_dual_value_navsim_e120_v1",
        "phase": "field_calibrated",
        "protocol_version": FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
        "branch_id": required_branch_id,
        "epoch": epoch,
        "roles": {
            "value_model": {
                "keys": list(keys),
                "shapes": normalized_shapes,
            }
        },
        "parents": copy.deepcopy(expected_parents),
    }


def initialize_fresh_p0_direct_rank0(
    *,
    modules: object,
    checkpoint_path: str | Path,
    params_pretrain_path: str | Path,
    lineage: object,
    training_stop_epoch: int,
) -> dict[str, object]:
    """Initialize manual Uniform-P0 directly from e120 without issuing a receipt."""

    normalized_modules = _require_modules(modules)
    normalized_lineage = validate_formal_v2_navsim_e120_direct_lineage(lineage, expected_stage="p0")
    _, stop_epoch, _ = _expected_direct_schedule(
        "p0",
        training_stop_epoch=training_stop_epoch,
    )
    _warmstart.apply_formal_v2_full_state_warmstart_direct(
        checkpoint_path,
        params_pretrain_path,
        normalized_modules,
    )
    role_shapes = _direct_role_state_shapes_from_states(
        {role: normalized_modules[role].state_dict() for role in MODEL_ROLES}
    )
    return {
        "stage": "p0",
        "training_stop_epoch": stop_epoch,
        "role_state_shapes": role_shapes,
        "lineage": normalized_lineage,
    }


def initialize_fresh_p1_direct_rank0(
    *,
    modules: object,
    checkpoint_path: str | Path,
    params_pretrain_path: str | Path,
    lineage: object,
    parent_checkpoint_path: str | Path,
    calibration_checkpoint_path: str | Path,
    calibration_checkpoint_validator: Callable[[Path], Mapping[str, object]],
) -> dict[str, object]:
    """Warm-start e120, then overlay only a structural selected Uniform-P0 parent."""

    normalized_modules = _require_modules(modules)
    normalized_lineage = validate_formal_v2_navsim_e120_direct_lineage(lineage, expected_stage="p1")
    value_lineage = _manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase="guided_planner",
        branch_id=normalized_lineage["branch_id"],
    )
    required_calibration_branch_id = value_lineage.checkpoint_branch_id("field_calibrated")
    calibration_path = _require_absolute_regular_direct_artifact(
        calibration_checkpoint_path,
        name="Calibration checkpoint",
    )
    if not callable(calibration_checkpoint_validator):
        raise ValueError("calibration_checkpoint_validator must be callable")
    calibration_metadata = calibration_checkpoint_validator(calibration_path)
    _validate_direct_calibration_checkpoint_metadata(
        calibration_metadata,
        required_branch_id=required_calibration_branch_id,
    )
    _warmstart.apply_formal_v2_full_state_warmstart_direct(
        checkpoint_path,
        params_pretrain_path,
        normalized_modules,
    )
    warmstart_states = {role: normalized_modules[role].state_dict() for role in MODEL_ROLES}
    warmstart_shapes = _direct_role_state_shapes_from_states(warmstart_states)
    parent = read_formal_v2_navsim_e120_direct_checkpoint(parent_checkpoint_path)
    if parent["stage"] != "p0" or parent["lineage"]["branch_id"] != "p0_uniform":
        raise ValueError("P1 parent must be a selected P0 checkpoint from branch p0_uniform")
    if parent["epoch"] not in FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS:
        raise ValueError(f"P1 parent epoch must be a P0 candidate in {FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS!r}")
    if parent["role_state_shapes"] != warmstart_shapes:
        raise ValueError("selected P0 role state shapes do not match the e120 warmstart roles")
    warmstart_encoder = _normalize_state(
        warmstart_states["encoder"],
        name="warmstart encoder state",
        clone=False,
    )
    if any(not torch.equal(parent["encoder"][key], warmstart_encoder[key]) for key in sorted(warmstart_encoder)):
        raise ValueError("selected P0 encoder tensors do not match the e120 warmstart encoder")
    for role in ("predictor", "planner"):
        _load_prepared_role(normalized_modules[role], parent[role], role=role)
    final_shapes = _direct_role_state_shapes_from_states(
        {role: normalized_modules[role].state_dict() for role in MODEL_ROLES}
    )
    if final_shapes != parent["role_state_shapes"]:
        raise RuntimeError("P1 direct overlay changed selected P0 role structure")
    return {
        "stage": "p1",
        "training_stop_epoch": 80,
        "parent_epoch": parent["epoch"],
        "role_state_shapes": final_shapes,
        "lineage": normalized_lineage,
    }


def _validate_distributed_status(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("distributed initialization status must be a mapping")
    _require_exact_fields(value, _DISTRIBUTED_STATUS_FIELDS, name="distributed initialization status")
    if value["schema"] != FORMAL_V2_NAVSIM_E120_DISTRIBUTED_STATUS_SCHEMA:
        raise RuntimeError("distributed initialization status schema mismatch")
    if type(value["ok"]) is not bool:
        raise RuntimeError("distributed initialization status ok must be boolean")
    if value["ok"]:
        if value["error_type"] is not None or value["error_message"] is not None:
            raise RuntimeError("successful distributed initialization status must not carry an error")
    else:
        if value["result"] is not None:
            raise RuntimeError("failed distributed initialization status must not carry a result")
        if not isinstance(value["error_type"], str) or not value["error_type"]:
            raise RuntimeError("failed distributed initialization status requires error_type")
        if not isinstance(value["error_message"], str) or not value["error_message"]:
            raise RuntimeError("failed distributed initialization status requires error_message")
    return dict(value)


def run_rank0_initialization_and_broadcast(
    *,
    rank: int,
    modules: object,
    distributed: object,
    rank0_initializer: Callable[[], object],
) -> object:
    """Broadcast rank-zero status first, then every role parameter and persistent buffer."""

    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    normalized_modules = _require_modules(modules)
    broadcast_object = getattr(distributed, "broadcast_object", None)
    broadcast_tensor = getattr(distributed, "broadcast_tensor", None)
    if not callable(broadcast_object) or not callable(broadcast_tensor):
        raise ValueError("distributed adapter must provide broadcast_object and broadcast_tensor")
    if not callable(rank0_initializer):
        raise ValueError("rank0_initializer must be callable")
    status: object = None
    if rank == 0:
        try:
            result = rank0_initializer()
        except Exception as error:
            status = {
                "schema": FORMAL_V2_NAVSIM_E120_DISTRIBUTED_STATUS_SCHEMA,
                "ok": False,
                "result": None,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        else:
            status = {
                "schema": FORMAL_V2_NAVSIM_E120_DISTRIBUTED_STATUS_SCHEMA,
                "ok": True,
                "result": result,
                "error_type": None,
                "error_message": None,
            }
    received = _validate_distributed_status(broadcast_object(status, src=0))
    if not received["ok"]:
        raise RuntimeError(
            "rank-zero NavSim e120 initialization failed: " f"{received['error_type']}: {received['error_message']}"
        )
    for role in MODEL_ROLES:
        module = normalized_modules[role]
        for name, parameter in sorted(module.named_parameters(remove_duplicate=False), key=lambda item: item[0]):
            broadcast_tensor(parameter.data, src=0, name=f"{role}.parameter.{name}")
        state_keys = set(module.state_dict())
        for name, buffer in sorted(module.named_buffers(remove_duplicate=False), key=lambda item: item[0]):
            if name in state_keys:
                broadcast_tensor(buffer.data, src=0, name=f"{role}.buffer.{name}")
    return received["result"]
