"""Planner-line bridge for the proof-free manual Formal-v2 NavSim e120 chain."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Protocol

import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage, cvoi_value
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_full_state_warmstart import (
    validate_formal_v2_full_state_warmstart_paths,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_runtime import (
    FORMAL_V2_NAVSIM_E120_GUIDANCE_SCHEMA,
    FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
    FormalV2NavSimE120HorizonExposureState,
    build_formal_v2_navsim_e120_direct_checkpoint,
    build_formal_v2_navsim_e120_direct_lineage,
    initialize_fresh_p0_direct_rank0,
    initialize_fresh_p1_direct_rank0,
    read_formal_v2_navsim_e120_direct_checkpoint,
    resolve_formal_v2_navsim_e120_selected_checkpoint,
    restore_formal_v2_navsim_e120_direct_same_run_resume,
    run_rank0_initialization_and_broadcast,
    validate_formal_v2_navsim_e120_direct_lineage,
    write_formal_v2_navsim_e120_direct_checkpoint,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_HORIZONS,
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)

_IDENTIFIER = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")
_PLAN_FIELDS = frozenset(
    {
        "run_id",
        "stage",
        "training_stop_epoch",
        "schedule_epochs",
        "selection_checkpoint_epochs",
        "warmstart_checkpoint_path",
        "warmstart_params_path",
        "lineage",
        "parent_checkpoint_path",
        "calibration_checkpoint_path",
        "guidance_signature",
        "resume_checkpoint_path",
    }
)
_FIELD_STATUS_FIELDS = frozenset(
    {
        "ok",
        "present",
        "architecture",
        "state_keys",
        "state_shapes",
        "error_type",
        "error_message",
    }
)
_FIXED_GUIDANCE_SIGNATURE = {
    "schema": FORMAL_V2_NAVSIM_E120_GUIDANCE_SCHEMA,
    "steps": 2,
    "objective": "last",
    "step_size": 0.05,
    "max_delta_norm": 0.25,
    "detach_output": True,
}


@dataclass(frozen=True)
class FormalV2NavSimE120PlannerPlan:
    """Complete structural initialization contract for one manual Planner run."""

    run_id: str
    stage: str
    training_stop_epoch: int
    schedule_epochs: int
    selection_checkpoint_epochs: tuple[int, ...]
    warmstart_checkpoint_path: Path
    warmstart_params_path: Path
    lineage: Mapping[str, object]
    parent_checkpoint_path: Path | None
    calibration_checkpoint_path: Path | None
    guidance_signature: Mapping[str, object] | None
    resume_checkpoint_path: Path | None = None


@dataclass(frozen=True)
class FormalV2NavSimE120PlannerRuntimeState:
    """Mutable training state boundary returned to the Planner line."""

    start_epoch: int
    initialization: str
    exposure: FormalV2NavSimE120HorizonExposureState
    initialization_result: object


class NavSimE120HorizonExposureRecorder(Protocol):
    """Minimal committed-update interface for the manual H0--H4 run."""

    def record(self, *, horizon: int, batch_size: int) -> None:
        """Commit one successfully optimized batch to the horizon histogram."""


def formal_v2_navsim_e120_periodic_checkpoint_path(folder: str | Path, *, epoch: int) -> Path:
    """Return the manual Planner milestone path using the existing ``e{epoch}.pt`` contract."""

    if type(epoch) is not int or epoch <= 0:
        raise ValueError("NavSim e120 periodic checkpoint epoch must be a positive integer")
    return Path(folder) / f"e{epoch}.pt"


def record_formal_v2_navsim_e120_optimizer_exposure(
    recorder: NavSimE120HorizonExposureRecorder | None,
    *,
    optimizer_step_successful: bool,
    horizon: int | None,
    batch_size: int | None,
) -> None:
    """Record one committed manual Planner update in exactly one H0--H4 bin."""

    if recorder is None:
        return
    if optimizer_step_successful is not True:
        raise RuntimeError("NavSim e120 rejected optimizer step; equal-update accounting cannot continue")
    if type(horizon) is not int or horizon not in FORMAL_V2_NAVSIM_HORIZONS:
        raise ValueError(
            f"NavSim e120 optimized horizon must be one of {list(FORMAL_V2_NAVSIM_HORIZONS)}, got {horizon!r}"
        )
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError(f"NavSim e120 optimized batch_size must be a positive integer, got {batch_size!r}")
    recorder.record(horizon=horizon, batch_size=batch_size)


@dataclass(frozen=True)
class TorchDistributedAdapter:
    """Small explicit adapter used by the strict rank-zero initializer."""

    rank: int
    world_size: int

    def __post_init__(self) -> None:
        if type(self.rank) is not int or type(self.world_size) is not int:
            raise ValueError("distributed rank/world_size must be integers")
        if self.world_size <= 0 or self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("distributed rank/world_size are inconsistent")
        initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.world_size > 1 and not initialized:
            raise RuntimeError("multi-rank NavSim e120 initialization requires torch.distributed")

    def broadcast_object(self, value: object, *, src: int) -> object:
        if self.world_size == 1:
            if self.rank != src or value is None:
                raise RuntimeError("single-rank object broadcast requires a rank-zero value")
            return value
        values = [value]
        torch.distributed.broadcast_object_list(values, src=src)
        return values[0]

    def broadcast_tensor(self, tensor: torch.Tensor, *, src: int, name: str) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("broadcast tensor name must be non-empty")
        if self.world_size > 1:
            torch.distributed.broadcast(tensor, src=src)


@dataclass(frozen=True)
class _BroadcastTensorLeaf:
    """Picklable tensor descriptor for distributed training-state transfer."""

    index: int
    shape: tuple[int, ...]
    dtype: torch.dtype


def planner_uses_navsim_e120_runtime(cvoi: object) -> bool:
    """Return whether the Planner must use the isolated e120 lifecycle."""

    return getattr(cvoi, "protocol_version", None) == FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION


def planner_uses_legacy_open_loop_selection(cvoi: object) -> bool:
    """Disable L2/collision selection and summaries for the NavSim-only profile."""

    return not planner_uses_navsim_e120_runtime(cvoi)


def formal_v2_navsim_e120_milestone_due(
    epoch: int,
    *,
    save_every_freq: int,
    selection_checkpoint_epochs: tuple[int, ...],
) -> bool:
    """Union ordinary milestones with mandatory selection candidates."""

    if type(epoch) is not int or epoch <= 0:
        raise ValueError("epoch must be a positive one-based integer")
    if type(save_every_freq) is not int or save_every_freq <= 0:
        raise ValueError("save_every_freq must be a positive integer")
    if type(selection_checkpoint_epochs) is not tuple or any(
        type(candidate) is not int or candidate <= 0 for candidate in selection_checkpoint_epochs
    ):
        raise ValueError("selection_checkpoint_epochs must be a tuple of positive integers")
    if tuple(sorted(set(selection_checkpoint_epochs))) != selection_checkpoint_epochs:
        raise ValueError("selection_checkpoint_epochs must be unique and strictly increasing")
    return epoch % save_every_freq == 0 or epoch in selection_checkpoint_epochs


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase snake-case identifier")
    return value


def _absolute_path(value: object, *, field: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a normalized absolute path: {path}")
    return path


def _optional_absolute_path(value: str | Path | None, *, field: str) -> Path | None:
    if value is None:
        return None
    return _absolute_path(value, field=field)


def _ablation_mapping(value: object) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        normalized = asdict(value)
    elif isinstance(value, Mapping):
        normalized = dict(value)
    elif hasattr(value, "__dict__"):
        normalized = vars(value)
    else:
        raise ValueError("cvoi.ablation_signature must expose an immutable typed mapping")
    if not isinstance(normalized, Mapping):
        raise ValueError("cvoi.ablation_signature must resolve to a mapping")
    return copy.deepcopy(dict(normalized))


def _resolve_manual_value_lineage(
    signature: object,
    *,
    stage: str,
    full_results_root: Path | None = None,
    ablation_results_root: Path | None = None,
) -> cvoi_manual_lineage.CvoiManualValueLineage:
    """Normalize every supported signature carrier before authority resolution."""

    normalized = _ablation_mapping(signature)
    try:
        signature_view = SimpleNamespace(**normalized)
    except TypeError as error:
        raise ValueError("cvoi.ablation_signature keys must be valid string field names") from error
    return cvoi_manual_lineage.resolve_cvoi_manual_value_lineage(
        signature_view,
        stage=stage,
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )


def _fixed_guidance_signature(config: object) -> dict[str, object]:
    guidance = getattr(config, "value_guidance", None)
    if guidance is None:
        raise ValueError("P1 requires value_guidance configuration")
    signature = {
        "schema": FORMAL_V2_NAVSIM_E120_GUIDANCE_SCHEMA,
        "steps": getattr(guidance, "steps", None),
        "objective": getattr(guidance, "objective", None),
        "step_size": getattr(guidance, "step_size", None),
        "max_delta_norm": getattr(guidance, "max_delta_norm", None),
        "detach_output": getattr(guidance, "detach_output", None),
    }
    if signature != _FIXED_GUIDANCE_SIGNATURE:
        raise ValueError(f"P1 value_guidance must equal the fixed e120 signature: {_FIXED_GUIDANCE_SIGNATURE!r}")
    return signature


def _locked_warmstart_paths(cvoi: object) -> tuple[Path, Path]:
    warmstart = getattr(cvoi, "full_state_warmstart", None)
    if warmstart is None:
        raise ValueError("NavSim e120 Planner requires cvoi.full_state_warmstart")
    checkpoint = getattr(warmstart, "source_checkpoint", None)
    params = getattr(warmstart, "source_params_pretrain", None)
    return validate_formal_v2_full_state_warmstart_paths(
        getattr(checkpoint, "path", None),
        getattr(params, "path", None),
    )


def _require_regular_calibration_path(
    value: object,
    *,
    expected: Path,
) -> Path:
    path = _absolute_path(value, field="cvoi.field_checkpoint")
    expected = _absolute_path(expected, field="expected Calibration checkpoint")
    if path != expected:
        raise ValueError(f"Calibration checkpoint path must be exactly {expected}, got {path}")
    if not path.exists():
        raise FileNotFoundError(f"Calibration checkpoint does not exist: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Calibration checkpoint must be a non-symlink regular file: {path}")
    return path


def load_formal_v2_navsim_e120_calibration_model_direct(
    config: object,
    *,
    embed_dim: int,
    device: torch.device,
) -> PrefixDualValueModel | None:
    """Strict-load the manual Calibration artifact without legacy audit machinery."""

    cvoi = getattr(config, "cvoi", None)
    if not planner_uses_navsim_e120_runtime(cvoi):
        raise ValueError("direct Calibration loading requires the NavSim e120 protocol")
    if type(embed_dim) is not int or embed_dim <= 0:
        raise ValueError("direct Calibration embed_dim must be a positive integer")
    if not isinstance(device, torch.device):
        raise ValueError("direct Calibration device must be a torch.device")
    stage = getattr(cvoi, "stage", None)
    if stage == "unguided_planner":
        if getattr(cvoi, "field_checkpoint", None) is not None:
            raise ValueError("direct P0 must not configure a Calibration checkpoint")
        return None
    if stage != "guided_planner":
        raise ValueError("direct Calibration loading supports only unguided_planner or guided_planner")

    full_results_root = cvoi_manual_lineage.resolve_cvoi_manual_full_results_root_from_config(cvoi)
    signature = getattr(cvoi, "ablation_signature", None)
    signature_mapping = _ablation_mapping(signature)
    ablation_results_root = (
        cvoi_manual_lineage.resolve_cvoi_manual_ablation_results_root_from_config(cvoi)
        if signature_mapping.get("experiment_role") == "ablation"
        else None
    )
    value_lineage = _resolve_manual_value_lineage(
        signature,
        stage="guided_planner",
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )
    calibration_path = _require_regular_calibration_path(
        getattr(cvoi, "field_checkpoint", None),
        expected=value_lineage.calibration_handoff,
    )
    payload = cvoi_value.read_cvoi_navsim_e120_direct_value_checkpoint(
        calibration_path,
        required_phase="field_calibrated",
        required_branch_id=value_lineage.checkpoint_branch_id("field_calibrated"),
        map_location="cpu",
    )
    architecture = payload["architecture"]
    if architecture["embed_dim"] != embed_dim:
        raise ValueError(
            f"direct Calibration embed_dim must match encoder embed_dim={embed_dim}, "
            f"got {architecture['embed_dim']!r}"
        )
    model = PrefixDualValueModel(**architecture)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device=device)
    model.eval()
    model.requires_grad_(False)
    return model


def build_formal_v2_navsim_e120_planner_plan(config: object) -> FormalV2NavSimE120PlannerPlan:
    """Build one direct P0/P1 plan without applying weights or reading proof artifacts."""

    cvoi = getattr(config, "cvoi", None)
    if not planner_uses_navsim_e120_runtime(cvoi):
        raise ValueError("Planner config does not select formal_v2_navsim_e120_h4_v3")
    cvoi_manual_lineage.reject_unedited_cvoi_public_placeholders(cvoi, boundary="CVoI Planner")
    outer_stage = getattr(cvoi, "stage", None)
    signature = getattr(cvoi, "ablation_signature", None)
    signature_mapping = _ablation_mapping(signature)
    value_lineage: cvoi_manual_lineage.CvoiManualValueLineage | None = None
    if outer_stage == "unguided_planner":
        stage = "p0"
        branch_id = "p0_uniform"
    elif outer_stage == "guided_planner":
        stage = "p1"
        full_results_root = cvoi_manual_lineage.resolve_cvoi_manual_full_results_root_from_config(cvoi)
        ablation_results_root = (
            cvoi_manual_lineage.resolve_cvoi_manual_ablation_results_root_from_config(cvoi)
            if signature_mapping.get("experiment_role") == "ablation"
            else None
        )
        value_lineage = _resolve_manual_value_lineage(
            signature,
            stage="guided_planner",
            full_results_root=full_results_root,
            ablation_results_root=ablation_results_root,
        )
        branch_id = value_lineage.checkpoint_branch_id("guided_planner")
    else:
        raise ValueError("NavSim e120 Planner supports only unguided_planner or guided_planner")

    ablation = signature_mapping
    if ablation.get("branch_id") != branch_id:
        raise ValueError(f"direct {stage.upper()} ablation branch_id must be exactly {branch_id!r}")
    seed = ablation.get("train_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("cvoi.ablation_signature.train_seed must be a non-negative integer")
    run_id = _require_identifier(f"{branch_id}_s{seed}", field="Planner run_id")
    checkpoint_path, params_path = _locked_warmstart_paths(cvoi)

    optimization = getattr(config, "optimization", None)
    meta = getattr(config, "meta", None)
    stop_epoch = getattr(optimization, "epochs", None)
    schedule_epochs = getattr(optimization, "schedule_epochs", None)
    selection_epochs = getattr(meta, "selection_checkpoint_epochs", None)
    if isinstance(selection_epochs, list):
        selection_epochs = tuple(selection_epochs)

    parent_path: Path | None = None
    calibration_path: Path | None = None
    guidance: Mapping[str, object] | None = None
    if stage == "p0":
        if getattr(cvoi, "unguided_planner_checkpoint", None) is not None:
            raise ValueError("P0 must not configure an unguided Planner parent")
        if getattr(cvoi, "field_checkpoint", None) is not None:
            raise ValueError("P0 must not configure a Calibration checkpoint")
    else:
        if value_lineage is None:
            raise RuntimeError("guided Planner lineage resolution did not produce a Value lineage")
        configured_parent = _absolute_path(
            getattr(cvoi, "unguided_planner_checkpoint", None),
            field="cvoi.unguided_planner_checkpoint",
        )
        parent_path = resolve_formal_v2_navsim_e120_selected_checkpoint(
            configured_parent,
            results_root=full_results_root,
            stage="p0",
        )
        parent = read_formal_v2_navsim_e120_direct_checkpoint(parent_path)
        if parent["stage"] != "p0":
            raise ValueError("selected parent checkpoint stage must be P0")
        if parent["lineage"]["branch_id"] != "p0_uniform":
            raise ValueError("selected parent checkpoint branch must be p0_uniform")
        if parent["epoch"] not in FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS:
            raise ValueError(f"selected parent epoch must be a candidate in {FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS!r}")
        calibration_path = _require_regular_calibration_path(
            getattr(cvoi, "field_checkpoint", None),
            expected=value_lineage.calibration_handoff,
        )
        guidance = _fixed_guidance_signature(config)

    plan = FormalV2NavSimE120PlannerPlan(
        run_id=run_id,
        stage=stage,
        training_stop_epoch=stop_epoch,
        schedule_epochs=schedule_epochs,
        selection_checkpoint_epochs=selection_epochs,
        warmstart_checkpoint_path=checkpoint_path,
        warmstart_params_path=params_path,
        lineage=build_formal_v2_navsim_e120_direct_lineage(stage=stage, branch_id=branch_id),
        parent_checkpoint_path=parent_path,
        calibration_checkpoint_path=calibration_path,
        guidance_signature=guidance,
    )
    return validate_formal_v2_navsim_e120_planner_plan(plan)


def validate_formal_v2_navsim_e120_planner_plan(
    plan: FormalV2NavSimE120PlannerPlan,
) -> FormalV2NavSimE120PlannerPlan:
    """Revalidate the exact direct stage schedule and structural bindings."""

    if not isinstance(plan, FormalV2NavSimE120PlannerPlan):
        raise TypeError("plan must be FormalV2NavSimE120PlannerPlan")
    if set(vars(plan)) != _PLAN_FIELDS:
        raise ValueError("Planner plan fields do not match the direct contract")
    run_id = _require_identifier(plan.run_id, field="Planner run_id")
    lineage = validate_formal_v2_navsim_e120_direct_lineage(plan.lineage, expected_stage=plan.stage)
    expected_run_prefix = f"{lineage['branch_id']}_s"
    if not run_id.startswith(expected_run_prefix) or not run_id.removeprefix(expected_run_prefix).isdigit():
        raise ValueError(f"Planner run_id must use the stable {expected_run_prefix}<train_seed> form")
    validate_formal_v2_full_state_warmstart_paths(
        plan.warmstart_checkpoint_path,
        plan.warmstart_params_path,
    )
    _optional_absolute_path(plan.resume_checkpoint_path, field="plan.resume_checkpoint_path")

    if plan.stage == "p0":
        if plan.training_stop_epoch != 50 or plan.schedule_epochs != 50:
            raise ValueError("direct P0 training_stop_epoch and schedule_epochs must both be exactly 50")
        if plan.selection_checkpoint_epochs != FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS:
            raise ValueError("direct P0 selection checkpoint epochs are not canonical")
        if (
            plan.parent_checkpoint_path is not None
            or plan.calibration_checkpoint_path is not None
            or plan.guidance_signature is not None
        ):
            raise ValueError("direct P0 must not bind parent, Calibration, or guidance inputs")
    else:
        if plan.parent_checkpoint_path is None:
            raise ValueError("direct P1 requires the selected P0 parent")
        parent = _absolute_path(plan.parent_checkpoint_path, field="plan.parent_checkpoint_path")
        branch_id = lineage["branch_id"]
        if branch_id == "p1_full":
            full_results_root = cvoi_manual_lineage.resolve_cvoi_manual_full_results_root(
                {"calibration_handoff": plan.calibration_checkpoint_path}
            )
            ablation_results_root = None
        else:
            parent_parts = parent.parts
            if parent.parts[-2:] == ("handoff", "p0_selected.pt"):
                full_results_root = cvoi_manual_lineage.resolve_cvoi_manual_full_results_root({"p0_handoff": parent})
            elif "p0" in parent_parts:
                p0_index = len(parent_parts) - 1 - tuple(reversed(parent_parts)).index("p0")
                full_results_root = Path(*parent_parts[:p0_index])
            else:
                raise ValueError("direct P1 parent must identify its configured Full results root")
            calibration = _absolute_path(
                plan.calibration_checkpoint_path,
                field="plan.calibration_checkpoint_path",
            )
            if calibration.parts[-2:] != ("handoff", "calibration.pt"):
                raise ValueError("direct P1 Calibration path must use the fixed handoff/calibration.pt suffix")
            ablation_branch_root = calibration.parent.parent
            expected_branch_name = branch_id.removeprefix("p1_")
            if ablation_branch_root.name != expected_branch_name:
                raise ValueError(
                    "direct P1 Calibration path must use the matching ablation branch directory "
                    f"{expected_branch_name!r}"
                )
            ablation_results_root = ablation_branch_root.parent
        value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
            phase="guided_planner",
            branch_id=branch_id,
            full_results_root=full_results_root,
            ablation_results_root=ablation_results_root,
        )
        if plan.training_stop_epoch != 80 or plan.schedule_epochs != 80:
            raise ValueError("direct P1 training_stop_epoch and schedule_epochs must both be exactly 80")
        if plan.selection_checkpoint_epochs != FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS:
            raise ValueError("direct P1 selection checkpoint epochs are not canonical")
        handoff = value_lineage.p0_handoff
        p0_root = full_results_root / "p0"
        if parent != handoff:
            try:
                parent.relative_to(p0_root)
            except ValueError as error:
                raise ValueError(
                    "direct P1 parent must be the fixed handoff copy or an in-P0 symlink target"
                ) from error
        if plan.calibration_checkpoint_path != value_lineage.calibration_handoff:
            raise ValueError(
                "direct P1 Calibration path must be the matching lineage handoff "
                f"{value_lineage.calibration_handoff}"
            )
        if (
            not isinstance(plan.guidance_signature, Mapping)
            or dict(plan.guidance_signature) != _FIXED_GUIDANCE_SIGNATURE
        ):
            raise ValueError("direct P1 guidance signature is not canonical")
    return plan


def build_formal_v2_navsim_e120_planner_plan_on_rank0(
    config: object,
    *,
    rank: int,
    distributed: object,
    resume_path: str | Path | None,
) -> FormalV2NavSimE120PlannerPlan:
    """Build the plan only on rank zero, then broadcast one authoritative result."""

    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    broadcast_object = getattr(distributed, "broadcast_object", None)
    if not callable(broadcast_object):
        raise ValueError("distributed adapter must provide broadcast_object")
    status: object = None
    if rank == 0:
        try:
            plan = replace(
                build_formal_v2_navsim_e120_planner_plan(config),
                resume_checkpoint_path=_optional_absolute_path(resume_path, field="resume checkpoint"),
            )
        except Exception as error:
            status = {
                "ok": False,
                "plan": None,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        else:
            status = {
                "ok": True,
                "plan": plan,
                "error_type": None,
                "error_message": None,
            }
    received = broadcast_object(status, src=0)
    fields = {"ok", "plan", "error_type", "error_message"}
    if not isinstance(received, Mapping) or set(received) != fields or type(received.get("ok")) is not bool:
        raise RuntimeError("rank-zero NavSim e120 Planner plan broadcast returned an invalid status")
    if not received["ok"]:
        raise RuntimeError(
            "rank-zero NavSim e120 Planner plan failed: "
            f"{received.get('error_type')}: {received.get('error_message')}"
        )
    if received["error_type"] is not None or received["error_message"] is not None:
        raise RuntimeError("successful NavSim e120 Planner plan status must not carry error details")
    return validate_formal_v2_navsim_e120_planner_plan(received["plan"])


def _pack_training_state_tree(
    value: object,
    *,
    tensors: list[torch.Tensor],
    path: str,
) -> object:
    if torch.is_tensor(value):
        if value.layout != torch.strided:
            raise ValueError(f"{path} tensor must use strided layout")
        descriptor = _BroadcastTensorLeaf(
            index=len(tensors),
            shape=tuple(value.shape),
            dtype=value.dtype,
        )
        tensors.append(value.detach())
        return descriptor
    if isinstance(value, Mapping):
        return {
            key: _pack_training_state_tree(item, tensors=tensors, path=f"{path}.{key}") for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _pack_training_state_tree(item, tensors=tensors, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _pack_training_state_tree(item, tensors=tensors, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    return copy.deepcopy(value)


def _unpack_training_state_tree(value: object, *, tensors: list[torch.Tensor], path: str) -> object:
    if isinstance(value, _BroadcastTensorLeaf):
        if value.index < 0 or value.index >= len(tensors):
            raise RuntimeError(f"{path} tensor descriptor index is out of range")
        tensor = tensors[value.index]
        if tuple(tensor.shape) != value.shape or tensor.dtype != value.dtype:
            raise RuntimeError(f"{path} tensor descriptor does not match the broadcast tensor")
        return tensor
    if isinstance(value, Mapping):
        return {
            key: _unpack_training_state_tree(item, tensors=tensors, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _unpack_training_state_tree(item, tensors=tensors, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _unpack_training_state_tree(item, tensors=tensors, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    return value


def _walk_tree(value: object) -> Iterator[object]:
    yield value
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_tree(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_tree(item)


def _require_device(*, device: object) -> torch.device:
    if isinstance(device, torch.device):
        return device
    raise ValueError("distributed state transfer requires a torch device")


def _module_device(modules: object) -> torch.device:
    if not isinstance(modules, Mapping) or set(modules) != {"encoder", "predictor", "planner"}:
        raise ValueError("distributed Planner roles must contain encoder, predictor, and planner")
    for role in ("encoder", "predictor", "planner"):
        module = getattr(modules[role], "module", modules[role])
        if not isinstance(module, torch.nn.Module):
            raise ValueError(f"distributed Planner role {role!r} must be a torch module")
        try:
            return next(module.parameters()).device
        except StopIteration:
            continue
    raise ValueError("distributed Planner roles must contain at least one parameter")


def _broadcast_training_state_from_rank0(
    rank0_state: Mapping[str, object] | None,
    *,
    field: str,
    rank: int,
    distributed: object,
    device: torch.device,
) -> Mapping[str, object]:
    device = _require_device(device=device)
    broadcast_object = getattr(distributed, "broadcast_object", None)
    broadcast_tensor = getattr(distributed, "broadcast_tensor", None)
    if not callable(broadcast_object) or not callable(broadcast_tensor):
        raise ValueError("distributed adapter must provide object and tensor broadcast")
    tensors: list[torch.Tensor] = []
    packed: object = None
    if rank == 0:
        if not isinstance(rank0_state, Mapping):
            raise RuntimeError(f"rank-zero {field} state is unavailable")
        packed = _pack_training_state_tree(rank0_state, tensors=tensors, path=field)
    received = broadcast_object(packed, src=0)
    leaves = [leaf for leaf in _walk_tree(received) if isinstance(leaf, _BroadcastTensorLeaf)]
    if sorted(leaf.index for leaf in leaves) != list(range(len(leaves))):
        raise RuntimeError(f"distributed {field} tensor descriptors are not contiguous")
    ordered_leaves = sorted(leaves, key=lambda leaf: leaf.index)
    if rank == 0:
        tensors = [tensor.to(device=device).contiguous() for tensor in tensors]
    else:
        tensors = [torch.empty(leaf.shape, dtype=leaf.dtype, device=device) for leaf in ordered_leaves]
    for leaf in leaves:
        broadcast_tensor(tensors[leaf.index], src=0, name=f"{field}.tensor.{leaf.index}")
    if rank == 0:
        if rank0_state is None:
            raise RuntimeError(f"rank-zero {field} state disappeared during broadcast")
        return rank0_state
    cpu_tensors = [tensor.detach().cpu() for tensor in tensors]
    unpacked = _unpack_training_state_tree(received, tensors=cpu_tensors, path=field)
    if not isinstance(unpacked, Mapping):
        raise RuntimeError(f"distributed {field} state must reconstruct to a mapping")
    return unpacked


def _state_dict(value: object, *, field: str) -> Mapping[str, object]:
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise ValueError(f"{field} must provide state_dict()")
    state = state_dict()
    if not isinstance(state, Mapping):
        raise ValueError(f"{field}.state_dict() must return a mapping")
    return state


def _load_state_dict(value: object, state: Mapping[str, object], *, field: str) -> None:
    load_state_dict = getattr(value, "load_state_dict", None)
    if not callable(load_state_dict):
        raise ValueError(f"{field} must provide load_state_dict()")
    load_state_dict(state)


def initialize_formal_v2_navsim_e120_planner_runtime(
    plan: FormalV2NavSimE120PlannerPlan,
    *,
    modules: object,
    optimizer: object,
    scaler: object,
    scheduler: object,
    wd_scheduler: object,
    rank: int,
    distributed: object,
    resume_path: str | Path | None,
    resume_model_only: bool,
) -> FormalV2NavSimE120PlannerRuntimeState:
    """Fresh-initialize once, or restore one exact direct same-run checkpoint."""

    plan = validate_formal_v2_navsim_e120_planner_plan(plan)
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if type(resume_model_only) is not bool:
        raise ValueError("resume_model_only must be boolean")
    requested_resume_path = _optional_absolute_path(resume_path, field="NavSim e120 resume path")
    if requested_resume_path != plan.resume_checkpoint_path:
        raise ValueError(
            "NavSim e120 resume path must equal the authoritative rank-zero Planner plan: "
            f"requested={requested_resume_path!s}, authoritative={plan.resume_checkpoint_path!s}"
        )

    if plan.resume_checkpoint_path is not None:
        if resume_model_only:
            raise ValueError("NavSim e120 same-run resume forbids model-only restore")
        rank0_training_states: dict[str, Mapping[str, object]] = {}

        def rank0_resume() -> dict[str, object]:
            payload = read_formal_v2_navsim_e120_direct_checkpoint(plan.resume_checkpoint_path)
            restored = restore_formal_v2_navsim_e120_direct_same_run_resume(
                payload,
                modules=modules,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                expected_run_id=plan.run_id,
                expected_stage=plan.stage,
                expected_lineage=plan.lineage,
                expected_training_stop_epoch=plan.training_stop_epoch,
                warmstart_requested=False,
                model_only=False,
            )
            for name, value in (
                ("optimizer", optimizer),
                ("scaler", scaler),
                ("scheduler", scheduler),
                ("wd_scheduler", wd_scheduler),
            ):
                state = _state_dict(value, field=name)
                _pack_training_state_tree(state, tensors=[], path=name)
                rank0_training_states[name] = state
            return {"restored": restored}

        result = run_rank0_initialization_and_broadcast(
            rank=rank,
            modules=modules,
            distributed=distributed,
            rank0_initializer=rank0_resume,
        )
        if not isinstance(result, Mapping) or set(result) != {"restored"}:
            raise RuntimeError("rank-zero direct resume result is incomplete")
        restored = result["restored"]
        if not isinstance(restored, Mapping):
            raise RuntimeError("rank-zero direct resume payload is invalid")
        for name, value in (
            ("optimizer", optimizer),
            ("scaler", scaler),
            ("scheduler", scheduler),
            ("wd_scheduler", wd_scheduler),
        ):
            state = _broadcast_training_state_from_rank0(
                rank0_training_states.get(name),
                field=name,
                rank=rank,
                distributed=distributed,
                device=_module_device(modules),
            )
            if rank != 0:
                _load_state_dict(value, state, field=name)
        return FormalV2NavSimE120PlannerRuntimeState(
            start_epoch=restored["start_epoch"],
            initialization="same_run_full_state_resume",
            exposure=FormalV2NavSimE120HorizonExposureState(prior=restored["cumulative_horizon_histogram"]),
            initialization_result=restored,
        )

    if resume_model_only:
        raise ValueError("fresh NavSim e120 initialization forbids resume_model_only=true")

    def rank0_initializer() -> object:
        if plan.stage == "p0":
            return initialize_fresh_p0_direct_rank0(
                modules=modules,
                checkpoint_path=plan.warmstart_checkpoint_path,
                params_pretrain_path=plan.warmstart_params_path,
                lineage=plan.lineage,
                training_stop_epoch=plan.training_stop_epoch,
            )
        value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
            phase="guided_planner",
            branch_id=plan.lineage["branch_id"],
        )
        return initialize_fresh_p1_direct_rank0(
            modules=modules,
            checkpoint_path=plan.warmstart_checkpoint_path,
            params_pretrain_path=plan.warmstart_params_path,
            lineage=plan.lineage,
            parent_checkpoint_path=plan.parent_checkpoint_path,
            calibration_checkpoint_path=plan.calibration_checkpoint_path,
            calibration_checkpoint_validator=lambda path: (
                cvoi_value.read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata(
                    path,
                    required_branch_id=value_lineage.checkpoint_branch_id("field_calibrated"),
                )
            ),
        )

    result = run_rank0_initialization_and_broadcast(
        rank=rank,
        modules=modules,
        distributed=distributed,
        rank0_initializer=rank0_initializer,
    )
    return FormalV2NavSimE120PlannerRuntimeState(
        start_epoch=0,
        initialization="fresh_full_state_warmstart",
        exposure=FormalV2NavSimE120HorizonExposureState(),
        initialization_result=result,
    )


def load_formal_v2_navsim_e120_field_model_on_rank0(
    *,
    loader: Callable[[], PrefixDualValueModel | None],
    rank: int,
    distributed: object,
    device: torch.device,
) -> PrefixDualValueModel | None:
    """Load Field on rank zero, then broadcast architecture and exact structure."""

    if not callable(loader):
        raise ValueError("Field loader must be callable")
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    device = _require_device(device=device)
    broadcast_object = getattr(distributed, "broadcast_object", None)
    broadcast_tensor = getattr(distributed, "broadcast_tensor", None)
    if not callable(broadcast_object) or not callable(broadcast_tensor):
        raise ValueError("distributed adapter must provide broadcast_object and broadcast_tensor")
    model: PrefixDualValueModel | None = None
    status: object = None
    if rank == 0:
        try:
            model = loader()
            if model is None:
                status = {
                    "ok": True,
                    "present": False,
                    "architecture": None,
                    "state_keys": None,
                    "state_shapes": None,
                    "error_type": None,
                    "error_message": None,
                }
            else:
                if not isinstance(model, PrefixDualValueModel):
                    raise ValueError("NavSim e120 Field loader must return PrefixDualValueModel or None")
                model.to(device=device)
                model.eval()
                model.requires_grad_(False)
                state = model.state_dict()
                status = {
                    "ok": True,
                    "present": True,
                    "architecture": {
                        "embed_dim": model.embed_dim,
                        "hidden_dim": model.hidden_dim,
                        "num_layers": model.num_layers,
                        "dropout": model.dropout,
                    },
                    "state_keys": sorted(state),
                    "state_shapes": {key: list(value.shape) for key, value in sorted(state.items())},
                    "error_type": None,
                    "error_message": None,
                }
        except Exception as error:
            status = {
                "ok": False,
                "present": False,
                "architecture": None,
                "state_keys": None,
                "state_shapes": None,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
    received = broadcast_object(status, src=0)
    if not isinstance(received, Mapping) or set(received) != _FIELD_STATUS_FIELDS:
        raise RuntimeError("rank-zero Field broadcast returned an invalid status")
    if type(received["ok"]) is not bool or type(received["present"]) is not bool:
        raise RuntimeError("rank-zero Field status booleans are invalid")
    if not received["ok"]:
        raise RuntimeError(
            "rank-zero NavSim e120 Field load failed: "
            f"{received.get('error_type')}: {received.get('error_message')}"
        )
    if received["error_type"] is not None or received["error_message"] is not None:
        raise RuntimeError("successful rank-zero Field status must not carry error details")
    if not received["present"]:
        if any(received[field] is not None for field in ("architecture", "state_keys", "state_shapes")):
            raise RuntimeError("absent rank-zero Field status must not carry model metadata")
        return None

    architecture = received["architecture"]
    if not isinstance(architecture, Mapping) or set(architecture) != {
        "embed_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
    }:
        raise RuntimeError("rank-zero Field status has invalid architecture")
    if rank != 0:
        model = PrefixDualValueModel(**architecture).to(device=device)
    if model is None:
        raise RuntimeError("present rank-zero Field status did not produce a model")
    for name, parameter in sorted(model.named_parameters(remove_duplicate=False), key=lambda item: item[0]):
        broadcast_tensor(parameter.data, src=0, name=f"field.parameter.{name}")
    state_keys = set(model.state_dict())
    for name, buffer in sorted(model.named_buffers(remove_duplicate=False), key=lambda item: item[0]):
        if name in state_keys:
            broadcast_tensor(buffer.data, src=0, name=f"field.buffer.{name}")
    actual_state = model.state_dict()
    actual_keys = sorted(actual_state)
    actual_shapes = {key: list(value.shape) for key, value in sorted(actual_state.items())}
    if received["state_keys"] != actual_keys:
        raise RuntimeError("broadcast Field state keys differ from the rank-zero structure")
    if received["state_shapes"] != actual_shapes:
        raise RuntimeError("broadcast Field state shapes differ from the rank-zero structure")
    model.eval()
    model.requires_grad_(False)
    return model


def save_formal_v2_navsim_e120_planner_checkpoint(
    *,
    plan: FormalV2NavSimE120PlannerPlan,
    modules: object,
    optimizer: object,
    scaler: object,
    scheduler: object,
    wd_scheduler: object,
    epoch: int,
    exposure: FormalV2NavSimE120HorizonExposureState,
    device: torch.device,
    rank: int,
    distributed: object,
    path: str | Path,
    replace: bool,
) -> Path | None:
    """All-reduce exposure and atomically publish a direct checkpoint on rank zero."""

    plan = validate_formal_v2_navsim_e120_planner_plan(plan)
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    if not hasattr(exposure, "snapshot") or not callable(exposure.snapshot):
        raise ValueError("exposure must provide snapshot(device=...)")
    histogram = exposure.snapshot(device=device)
    broadcast_object = getattr(distributed, "broadcast_object", None)
    if not callable(broadcast_object):
        raise ValueError("distributed adapter must provide broadcast_object")
    status: object = None
    if rank == 0:
        try:
            payload = build_formal_v2_navsim_e120_direct_checkpoint(
                modules=modules,
                optimizer=_state_dict(optimizer, field="optimizer"),
                scaler=_state_dict(scaler, field="scaler"),
                scheduler=_state_dict(scheduler, field="scheduler"),
                wd_scheduler=_state_dict(wd_scheduler, field="wd_scheduler"),
                run_id=plan.run_id,
                stage=plan.stage,
                epoch=epoch,
                training_stop_epoch=plan.training_stop_epoch,
                schedule_epochs=plan.schedule_epochs,
                selection_checkpoint_epochs=plan.selection_checkpoint_epochs,
                cumulative_horizon_histogram=histogram,
                lineage=plan.lineage,
            )
            output = write_formal_v2_navsim_e120_direct_checkpoint(path, payload, replace=replace)
        except Exception as error:
            status = {
                "ok": False,
                "output": None,
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        else:
            status = {
                "ok": True,
                "output": str(output),
                "error_type": None,
                "error_message": None,
            }
    received = broadcast_object(status, src=0)
    if not isinstance(received, Mapping) or set(received) != {
        "ok",
        "output",
        "error_type",
        "error_message",
    }:
        raise RuntimeError("rank-zero checkpoint publication returned an invalid status")
    if received["ok"] is not True:
        raise RuntimeError(
            "rank-zero NavSim e120 checkpoint publication failed: "
            f"{received.get('error_type')}: {received.get('error_message')}"
        )
    output = received["output"]
    if not isinstance(output, str) or not Path(output).is_absolute():
        raise RuntimeError("successful rank-zero checkpoint publication requires an absolute output path")
    return Path(output) if rank == 0 else None


def reconcile_formal_v2_navsim_e120_resume_milestone(
    *,
    plan: FormalV2NavSimE120PlannerPlan,
    runtime_state: FormalV2NavSimE120PlannerRuntimeState,
    save_every_freq: int,
    rank: int,
    distributed: object,
    publish_checkpoint: Callable[[int, str | Path, bool], Path | None],
) -> Path | None:
    """Recover a due direct milestone after preemption of the latest file."""

    plan = validate_formal_v2_navsim_e120_planner_plan(plan)
    if not isinstance(runtime_state, FormalV2NavSimE120PlannerRuntimeState):
        raise TypeError("runtime_state must be FormalV2NavSimE120PlannerRuntimeState")
    if runtime_state.initialization != "same_run_full_state_resume":
        return None
    completed_epoch = runtime_state.start_epoch
    if not formal_v2_navsim_e120_milestone_due(
        completed_epoch,
        save_every_freq=save_every_freq,
        selection_checkpoint_epochs=plan.selection_checkpoint_epochs,
    ):
        return None
    if plan.resume_checkpoint_path is None:
        raise ValueError("same-run milestone recovery requires plan.resume_checkpoint_path")
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative integer")
    broadcast_object = getattr(distributed, "broadcast_object", None)
    if not callable(broadcast_object):
        raise ValueError("distributed adapter must provide broadcast_object")
    if not callable(publish_checkpoint):
        raise ValueError("publish_checkpoint must be callable")

    milestone_path = formal_v2_navsim_e120_periodic_checkpoint_path(
        plan.resume_checkpoint_path.parent,
        epoch=completed_epoch,
    )
    restored = runtime_state.initialization_result
    if not isinstance(restored, Mapping):
        raise RuntimeError("same-run resume result must be a mapping for milestone recovery")
    expected = {
        "run_id": plan.run_id,
        "stage": plan.stage,
        "epoch": completed_epoch,
        "training_stop_epoch": plan.training_stop_epoch,
        "schedule_epochs": plan.schedule_epochs,
        "selection_checkpoint_epochs": plan.selection_checkpoint_epochs,
        "cumulative_horizon_histogram": restored.get("cumulative_horizon_histogram"),
        "role_state_shapes": restored.get("role_state_shapes"),
        "lineage": plan.lineage,
    }
    status: object = None
    if rank == 0:
        try:
            if milestone_path.exists():
                existing = read_formal_v2_navsim_e120_direct_checkpoint(milestone_path)
                for field, expected_value in expected.items():
                    if existing.get(field) != expected_value:
                        raise ValueError(
                            f"existing resume milestone {field} mismatch: "
                            f"expected={expected_value!r}, actual={existing.get(field)!r}"
                        )
                action = "existing"
            else:
                action = "publish"
            status = {
                "ok": True,
                "action": action,
                "path": str(milestone_path),
                "error_type": None,
                "error_message": None,
            }
        except Exception as error:
            status = {
                "ok": False,
                "action": None,
                "path": str(milestone_path),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
    received = broadcast_object(status, src=0)
    fields = {"ok", "action", "path", "error_type", "error_message"}
    if not isinstance(received, Mapping) or set(received) != fields:
        raise RuntimeError("rank-zero milestone recovery returned an invalid status")
    if received["ok"] is not True:
        raise RuntimeError(
            "rank-zero NavSim e120 milestone recovery failed: "
            f"{received.get('error_type')}: {received.get('error_message')}"
        )
    if received["path"] != str(milestone_path):
        raise RuntimeError("rank-zero milestone recovery path differs from the canonical epoch path")
    if received["action"] == "existing":
        return milestone_path
    if received["action"] != "publish":
        raise RuntimeError("rank-zero milestone recovery action is invalid")
    publish_checkpoint(completed_epoch, milestone_path, False)
    return milestone_path
