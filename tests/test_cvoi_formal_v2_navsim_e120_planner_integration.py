"""Direct Planner-line integration contracts for the manual NavSim e120 chain."""

from __future__ import annotations

import ast
import copy
import inspect
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training import cvoi_formal_v2_full_state_warmstart as warmstart
from app.vjepa_cowa_world_model.training import cvoi_formal_v2_navsim_e120_planner_integration as integration
from app.vjepa_cowa_world_model.training import cvoi_formal_v2_navsim_e120_runtime as runtime
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage
from app.vjepa_cowa_world_model.training import cvoi_runtime as legacy_cvoi_runtime
from app.vjepa_cowa_world_model.training import cvoi_value
from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import CvoiFormalV2NavSimE120AblationSignature
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)

PLANNER_LINE = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "vjepa_cowa_world_model"
    / "training"
    / "lines"
    / "planner_world_model.py"
)

_P1_LINEAGE_CASES = (
    ("full", "hazard_quality", "p1_full", "calibration_full"),
    ("no_cf", "none", "p1_no_cf", "calibration_no_cf"),
    ("hazard_only", "hazard_only", "p1_hazard_only", "calibration_hazard_only"),
    ("quality_only", "quality_only", "p1_quality_only", "calibration_quality_only"),
)


class _State:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class _TensorState(_State):
    def __init__(self, value: int, tensor: torch.Tensor) -> None:
        super().__init__(value)
        self.tensor = tensor

    def state_dict(self) -> dict[str, object]:
        return {"value": self.value, "nested": {"tensor": self.tensor}}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.value = state["value"]
        self.tensor = state["nested"]["tensor"]


class _Module(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value]))


class _Exposure:
    def __init__(self, value: dict[int, int]) -> None:
        self.value = value
        self.devices: list[torch.device] = []

    def snapshot(self, *, device: torch.device) -> dict[int, int]:
        self.devices.append(device)
        return dict(self.value)


class _ObjectBroadcast:
    def __init__(self, received: object | None = None) -> None:
        self.received = received
        self.sent: list[object] = []

    def broadcast_object(self, value: object, *, src: int) -> object:
        assert src == 0
        self.sent.append(value)
        return value if self.received is None else self.received


class _BroadcastBus:
    def __init__(self) -> None:
        self.objects: list[object] = []
        self.tensors: list[torch.Tensor] = []


class _BusEndpoint:
    def __init__(self, bus: _BroadcastBus, *, rank: int) -> None:
        self.bus = bus
        self.rank = rank
        self.object_index = 0
        self.tensor_index = 0

    def broadcast_object(self, value: object, *, src: int) -> object:
        assert src == 0
        if self.rank == 0:
            self.bus.objects.append(value)
            return value
        received = self.bus.objects[self.object_index]
        self.object_index += 1
        return received

    def broadcast_tensor(self, tensor: torch.Tensor, *, src: int, name: str) -> None:
        assert src == 0
        assert name
        if self.rank == 0:
            self.bus.tensors.append(tensor.detach().cpu().clone())
            return
        tensor.copy_(self.bus.tensors[self.tensor_index].to(device=tensor.device, dtype=tensor.dtype))
        self.tensor_index += 1


def _modules(value: float = 1.0) -> dict[str, _Module]:
    return {role: _Module(value + index) for index, role in enumerate(("encoder", "predictor", "planner"))}


def _ablation(*, stage: str, seed: int = 239) -> SimpleNamespace:
    return SimpleNamespace(
        schema="cvoi_formal_v2_navsim_e120_ablation_v1",
        protocol_version=runtime.FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
        experiment_role="main",
        branch_id="p0_uniform" if stage == "unguided_planner" else "p1_full",
        shared_cohort_id="navsim_e120_s239_stride4",
        initialization_mode="full_state_warmstart",
        cf_field_supervision="hazard_quality",
        field_calibration_mode="local_geometry",
        p0_prefix_mode="uniform",
        gate_feature_mode="full",
        train_seed=seed,
        evaluation_seed=239,
        training_stride=4,
    )


def _config(results_root: Path, *, stage: str, seed: int = 239) -> SimpleNamespace:
    p0 = stage == "unguided_planner"
    warmstart_root = results_root.parent / "portable-warmstart"
    return SimpleNamespace(
        cvoi=SimpleNamespace(
            protocol_version=runtime.FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
            stage=stage,
            ablation_signature=_ablation(stage=stage, seed=seed),
            full_state_warmstart=SimpleNamespace(
                source_checkpoint=SimpleNamespace(
                    path=str(warmstart_root / "e120.pt"),
                    sha256="poisoned-and-ignored",
                ),
                source_params_pretrain=SimpleNamespace(
                    path=str(warmstart_root / "params-pretrain.yaml"),
                    receipt_path="/poisoned-and-ignored",
                ),
                receipt_path="/poisoned-and-ignored",
            ),
            navsim_selection=SimpleNamespace(p0_receipt_path="/poisoned-and-ignored"),
            unguided_planner_checkpoint=None if p0 else str(results_root / "handoff" / "p0_selected.pt"),
            field_checkpoint=None if p0 else str(results_root / "handoff" / "calibration.pt"),
        ),
        optimization=SimpleNamespace(epochs=50 if p0 else 80, schedule_epochs=50 if p0 else 80),
        meta=SimpleNamespace(
            selection_checkpoint_epochs=(
                FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS if p0 else FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS
            ),
            resume_model_only=False,
        ),
        value_guidance=SimpleNamespace(
            enabled=not p0,
            steps=2,
            objective="last",
            step_size=0.05,
            max_delta_norm=0.25,
            detach_output=True,
        ),
    )


def _set_manual_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    root = tmp_path / "cvoi_manual_full"
    (root / "handoff").mkdir(parents=True)
    ablation_root = tmp_path / "cvoi_manual_ablation"
    ablation_root.mkdir()
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_FULL_RESULTS_ROOT", root)
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_ABLATION_RESULTS_ROOT", ablation_root)
    return root


def _lineage_calibration_path(
    full_root: Path,
    *,
    lineage_name: str,
) -> Path:
    result_root = (
        full_root if lineage_name == "full" else cvoi_manual_lineage.CVOI_MANUAL_ABLATION_RESULTS_ROOT / lineage_name
    )
    return result_root / "handoff" / "calibration.pt"


def _direct_p1_config(
    full_root: Path,
    *,
    lineage_name: str,
    supervision: str,
) -> SimpleNamespace:
    config = _config(full_root, stage="guided_planner")
    config.cvoi.ablation_signature.experiment_role = "main" if lineage_name == "full" else "ablation"
    config.cvoi.ablation_signature.branch_id = f"p1_{lineage_name}"
    config.cvoi.ablation_signature.cf_field_supervision = supervision
    config.cvoi.unguided_planner_checkpoint = str(full_root / "handoff" / "p0_selected.pt")
    config.cvoi.field_checkpoint = str(_lineage_calibration_path(full_root, lineage_name=lineage_name))
    return config


def _direct_calibration_payload(
    *,
    embed_dim: int,
    branch_id: str,
) -> dict[str, object]:
    lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase="field_calibrated",
        branch_id=branch_id,
    )
    return cvoi_value.build_cvoi_navsim_e120_direct_value_checkpoint(
        PrefixDualValueModel(embed_dim=embed_dim, hidden_dim=3),
        phase="field_calibrated",
        branch_id=branch_id,
        epoch=7,
        parents=cvoi_manual_lineage.build_cvoi_manual_value_parents(lineage, "field_calibrated"),
    )


def _write_lineage_calibration(
    full_root: Path,
    *,
    lineage_name: str,
    branch_id: str,
) -> Path:
    path = _lineage_calibration_path(full_root, lineage_name=lineage_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_direct_calibration_payload(embed_dim=2, branch_id=branch_id), path)
    return path


def _direct_checkpoint_payload(
    *,
    stage: str = "p0",
    branch_id: str = "p0_uniform",
    run_id: str = "p0_uniform_s239",
    epoch: int = 35,
    modules: dict[str, _Module] | None = None,
) -> dict[str, object]:
    return runtime.build_formal_v2_navsim_e120_direct_checkpoint(
        modules=_modules() if modules is None else modules,
        optimizer={"value": 11},
        scaler={"value": 12},
        scheduler={"value": 13},
        wd_scheduler={"value": 14},
        run_id=run_id,
        stage=stage,
        epoch=epoch,
        training_stop_epoch=50 if stage == "p0" else 80,
        schedule_epochs=50 if stage == "p0" else 80,
        selection_checkpoint_epochs=(
            FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS if stage == "p0" else FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS
        ),
        cumulative_horizon_histogram={0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        lineage=runtime.build_formal_v2_navsim_e120_direct_lineage(stage=stage, branch_id=branch_id),
    )


def _write_parent(
    root: Path,
    *,
    symlink: bool = False,
    payload: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    handoff = root / "handoff" / "p0_selected.pt"
    payload = _direct_checkpoint_payload() if payload is None else payload
    if symlink:
        target = root / "p0" / "checkpoints" / "e35.pt"
        target.parent.mkdir(parents=True)
        torch.save(payload, target)
        handoff.symlink_to(target)
        return handoff, target
    torch.save(payload, handoff)
    return handoff, handoff


def _calibration_parents() -> dict[str, object]:
    return {
        "unguided_planner": {"stage": "p0", "branch_id": "p0_uniform"},
        "field": {"phase": "field_warmup", "branch_id": "field_full"},
    }


def _write_calibration(root: Path) -> Path:
    path = root / "handoff" / "calibration.pt"
    model = PrefixDualValueModel(embed_dim=2, hidden_dim=3)
    payload = cvoi_value.build_cvoi_navsim_e120_direct_value_checkpoint(
        model,
        phase="field_calibrated",
        branch_id="calibration_full",
        epoch=7,
        parents=_calibration_parents(),
    )
    torch.save(payload, path)
    return path


def _poison_legacy_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "apply_formal_v2_full_state_warmstart",
        "read_formal_v2_full_state_warmstart_receipt",
        "validate_formal_v2_full_state_warmstart_receipt",
    ):
        assert not hasattr(warmstart, name)
    for name in (
        "build_formal_v2_navsim_e120_checkpoint",
        "build_formal_v2_navsim_e120_lineage",
        "read_formal_v2_navsim_e120_checkpoint",
        "restore_same_run_resume",
        "write_formal_v2_navsim_e120_checkpoint",
    ):
        assert not hasattr(runtime, name)


def test_path_only_uniform_p0_plan_has_exact_direct_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    _poison_legacy_helpers(monkeypatch)

    plan = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="unguided_planner"))

    assert plan.run_id == "p0_uniform_s239"
    assert plan.stage == "p0"
    assert plan.training_stop_epoch == plan.schedule_epochs == 50
    assert plan.selection_checkpoint_epochs == FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS
    assert plan.warmstart_checkpoint_path == root.parent / "portable-warmstart/e120.pt"
    assert plan.warmstart_params_path == root.parent / "portable-warmstart/params-pretrain.yaml"
    assert plan.lineage == runtime.build_formal_v2_navsim_e120_direct_lineage(
        stage="p0",
        branch_id="p0_uniform",
    )
    assert plan.parent_checkpoint_path is None
    assert plan.calibration_checkpoint_path is None
    assert plan.guidance_signature is None


def test_path_only_uniform_p0_plan_accepts_a_portable_warmstart_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    config = _config(root, stage="unguided_planner")
    checkpoint_path = "/opt/rise-user/checkpoints/e120.pt"
    params_path = "/opt/rise-user/checkpoints/params-pretrain.yaml"
    config.cvoi.full_state_warmstart.source_checkpoint.path = checkpoint_path
    config.cvoi.full_state_warmstart.source_params_pretrain.path = params_path

    plan = integration.build_formal_v2_navsim_e120_planner_plan(config)

    assert plan.warmstart_checkpoint_path == Path(checkpoint_path)
    assert plan.warmstart_params_path == Path(params_path)


@pytest.mark.parametrize("symlink", [False, True])
def test_p1_plan_accepts_fixed_parent_copy_or_in_tree_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink: bool,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    configured, resolved = _write_parent(root, symlink=symlink)
    calibration = _write_calibration(root)
    reads: list[Path] = []
    real_read = runtime.read_formal_v2_navsim_e120_direct_checkpoint

    def read(path: str | Path) -> dict[str, object]:
        reads.append(Path(path))
        return real_read(path)

    monkeypatch.setattr(integration, "read_formal_v2_navsim_e120_direct_checkpoint", read)
    plan = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="guided_planner"))

    assert configured == root / "handoff" / "p0_selected.pt"
    assert plan.run_id == "p1_full_s239"
    assert plan.stage == "p1"
    assert plan.parent_checkpoint_path == resolved
    assert plan.calibration_checkpoint_path == calibration
    assert plan.lineage == runtime.build_formal_v2_navsim_e120_direct_lineage(
        stage="p1",
        branch_id="p1_full",
    )
    assert reads == [resolved]
    assert plan.guidance_signature == {
        "schema": runtime.FORMAL_V2_NAVSIM_E120_GUIDANCE_SCHEMA,
        "steps": 2,
        "objective": "last",
        "step_size": 0.05,
        "max_delta_norm": 0.25,
        "detach_output": True,
    }


@pytest.mark.parametrize(
    ("lineage_name", "supervision", "p1_branch", "calibration_branch"),
    _P1_LINEAGE_CASES,
)
def test_p1_plan_matrix_uses_full_p0_and_branch_local_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage_name: str,
    supervision: str,
    p1_branch: str,
    calibration_branch: str,
) -> None:
    full_root = _set_manual_root(monkeypatch, tmp_path)
    parent, resolved_parent = _write_parent(full_root)
    calibration = _write_lineage_calibration(
        full_root,
        lineage_name=lineage_name,
        branch_id=calibration_branch,
    )
    config = _direct_p1_config(
        full_root,
        lineage_name=lineage_name,
        supervision=supervision,
    )

    plan = integration.build_formal_v2_navsim_e120_planner_plan(config)

    assert parent == full_root / "handoff" / "p0_selected.pt"
    assert plan.run_id == f"{p1_branch}_s239"
    assert plan.lineage["branch_id"] == p1_branch
    assert plan.parent_checkpoint_path == resolved_parent
    assert plan.calibration_checkpoint_path == calibration


@pytest.mark.parametrize("carrier", ["namespace", "mapping", "typed_dataclass"])
def test_p1_plan_normalizes_each_supported_signature_carrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    carrier: str,
) -> None:
    full_root = _set_manual_root(monkeypatch, tmp_path)
    _write_parent(full_root)
    calibration = _write_lineage_calibration(
        full_root,
        lineage_name="no_cf",
        branch_id="calibration_no_cf",
    )
    config = _direct_p1_config(
        full_root,
        lineage_name="no_cf",
        supervision="none",
    )
    signature = vars(config.cvoi.ablation_signature).copy()
    if carrier == "mapping":
        config.cvoi.ablation_signature = signature
    elif carrier == "typed_dataclass":
        config.cvoi.ablation_signature = CvoiFormalV2NavSimE120AblationSignature(**signature)

    plan = integration.build_formal_v2_navsim_e120_planner_plan(config)

    assert plan.lineage["branch_id"] == "p1_no_cf"
    assert plan.calibration_checkpoint_path == calibration


@pytest.mark.parametrize("signature", [None, {"branch_id": "p1_full"}])
def test_p1_plan_rejects_missing_or_incomplete_signature_without_defaulting_full(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    signature: object,
) -> None:
    full_root = _set_manual_root(monkeypatch, tmp_path)
    config = _config(full_root, stage="guided_planner")
    config.cvoi.ablation_signature = signature

    with pytest.raises(ValueError, match="ablation_signature|manual CVoI signature"):
        integration.build_formal_v2_navsim_e120_planner_plan(config)


def test_no_cf_plan_validation_rejects_full_calibration_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_root = _set_manual_root(monkeypatch, tmp_path)
    _write_parent(full_root)
    full_calibration = _write_lineage_calibration(
        full_root,
        lineage_name="full",
        branch_id="calibration_full",
    )
    _write_lineage_calibration(
        full_root,
        lineage_name="no_cf",
        branch_id="calibration_no_cf",
    )
    plan = integration.build_formal_v2_navsim_e120_planner_plan(
        _direct_p1_config(
            full_root,
            lineage_name="no_cf",
            supervision="none",
        )
    )

    with pytest.raises(ValueError, match="Calibration"):
        integration.validate_formal_v2_navsim_e120_planner_plan(
            replace(plan, calibration_checkpoint_path=full_calibration)
        )


def test_plan_fields_and_values_contain_no_proof_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    plan = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="unguided_planner"))
    expected_fields = {
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

    assert {field.name for field in fields(type(plan))} == expected_fields
    assert {field.name for field in fields(integration.FormalV2NavSimE120PlannerRuntimeState)} == {
        "start_epoch",
        "initialization",
        "exposure",
        "initialization_result",
    }
    serialized = repr(plan).lower()
    assert all(marker not in serialized for marker in ("receipt", "sha256", "audit", "provenance"))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload.update(stage="p1"), "stage"),
        (lambda payload: payload["lineage"].update(branch_id="p1_full"), "branch"),
        (lambda payload: payload.update(epoch=34), "candidate"),
        (lambda payload: payload.pop("planner"), "fields"),
        (lambda payload: payload["role_state_shapes"]["planner"].update(extra=[1]), "keys"),
        (lambda payload: payload["role_state_shapes"]["planner"].update(weight=[2]), "shapes"),
    ],
)
def test_p1_plan_rejects_parent_stage_branch_epoch_role_key_or_shape_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: object,
    message: str,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    payload = _direct_checkpoint_payload()
    mutator(payload)
    handoff = root / "handoff" / "p0_selected.pt"
    torch.save(payload, handoff)
    _write_calibration(root)

    with pytest.raises((ValueError, KeyError), match=message):
        integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="guided_planner"))


def test_p1_plan_rejects_noncanonical_parent_and_calibration_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    _write_parent(root)
    _write_calibration(root)
    config = _config(root, stage="guided_planner")
    config.cvoi.unguided_planner_checkpoint = str(root / "p0" / "other.pt")
    with pytest.raises(ValueError, match="exactly"):
        integration.build_formal_v2_navsim_e120_planner_plan(config)

    config = _config(root, stage="guided_planner")
    config.cvoi.field_checkpoint = str(root / "handoff" / "field.pt")
    with pytest.raises(ValueError, match="Calibration.*exactly"):
        integration.build_formal_v2_navsim_e120_planner_plan(config)

    (root / "handoff" / "calibration.pt").unlink()
    target = root / "calibration" / "e7.pt"
    target.parent.mkdir()
    target.write_bytes(b"not-inspected-by-plan")
    (root / "handoff" / "calibration.pt").symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink regular file"):
        integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="guided_planner"))


@pytest.mark.parametrize("contents", [{"field_head.weight": torch.ones(1)}, b"not-a-checkpoint"])
def test_fresh_p1_rejects_raw_field_or_corrupt_calibration_before_warmstart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: object,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    _write_parent(root)
    calibration = root / "handoff" / "calibration.pt"
    if isinstance(contents, bytes):
        calibration.write_bytes(contents)
    else:
        torch.save(contents, calibration)
    plan = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="guided_planner"))
    monkeypatch.setattr(
        warmstart,
        "apply_formal_v2_full_state_warmstart_direct",
        lambda *_: pytest.fail("invalid Calibration must fail before warmstart"),
    )

    with pytest.raises(RuntimeError, match="rank-zero NavSim e120 initialization failed"):
        integration.initialize_formal_v2_navsim_e120_planner_runtime(
            plan,
            modules=_modules(),
            optimizer=_State(1),
            scaler=_State(2),
            scheduler=_State(3),
            wd_scheduler=_State(4),
            rank=0,
            distributed=_BusEndpoint(_BroadcastBus(), rank=0),
            resume_path=None,
            resume_model_only=False,
        )


def test_rank0_plan_builder_never_changes_modules_or_calls_warmstart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    _poison_legacy_helpers(monkeypatch)
    monkeypatch.setattr(
        warmstart,
        "apply_formal_v2_full_state_warmstart_direct",
        lambda *_: pytest.fail("plan construction must not warmstart"),
    )
    before = _modules()
    snapshots = {role: copy.deepcopy(module.state_dict()) for role, module in before.items()}

    plan = integration.build_formal_v2_navsim_e120_planner_plan_on_rank0(
        _config(root, stage="unguided_planner"),
        rank=0,
        distributed=_ObjectBroadcast(),
        resume_path=tmp_path / "latest.pt",
    )

    assert plan.resume_checkpoint_path == tmp_path / "latest.pt"
    assert "modules" not in inspect.signature(integration.build_formal_v2_navsim_e120_planner_plan_on_rank0).parameters
    assert all(
        torch.equal(before[role].state_dict()[key], value)
        for role, state in snapshots.items()
        for key, value in state.items()
    )


def test_fresh_initialization_warmstarts_exactly_once_and_uses_official_calibration_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    _write_parent(root)
    _write_calibration(root)
    p0 = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="unguided_planner"))
    p1 = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="guided_planner"))
    calls: list[tuple[str, object]] = []

    def initialize_p0(**kwargs: object) -> dict[str, object]:
        calls.append(("p0", kwargs))
        return {
            "stage": "p0",
            "training_stop_epoch": 50,
            "role_state_shapes": {role: {"weight": [1]} for role in ("encoder", "predictor", "planner")},
            "lineage": p0.lineage,
        }

    def initialize_p1(**kwargs: object) -> dict[str, object]:
        calls.append(("p1", kwargs))
        calibration_checkpoint_validator = kwargs["calibration_checkpoint_validator"]
        assert callable(calibration_checkpoint_validator)
        assert p1.calibration_checkpoint_path is not None
        calibration_metadata = calibration_checkpoint_validator(p1.calibration_checkpoint_path)
        assert calibration_metadata["branch_id"] == "calibration_full"
        return {
            "stage": "p1",
            "training_stop_epoch": 80,
            "parent_epoch": 35,
            "role_state_shapes": {role: {"weight": [1]} for role in ("encoder", "predictor", "planner")},
            "lineage": p1.lineage,
        }

    monkeypatch.setattr(integration, "initialize_fresh_p0_direct_rank0", initialize_p0)
    monkeypatch.setattr(integration, "initialize_fresh_p1_direct_rank0", initialize_p1)
    for plan in (p0, p1):
        state = integration.initialize_formal_v2_navsim_e120_planner_runtime(
            plan,
            modules=_modules(),
            optimizer=_State(1),
            scaler=_State(2),
            scheduler=_State(3),
            wd_scheduler=_State(4),
            rank=0,
            distributed=_BusEndpoint(_BroadcastBus(), rank=0),
            resume_path=None,
            resume_model_only=False,
        )
        assert state.start_epoch == 0
        assert state.initialization == "fresh_full_state_warmstart"
    assert [stage for stage, _ in calls] == ["p0", "p1"]


@pytest.mark.parametrize(
    ("lineage_name", "supervision", "p1_branch", "calibration_branch"),
    _P1_LINEAGE_CASES,
)
def test_fresh_p1_runtime_callback_binds_matching_calibration_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage_name: str,
    supervision: str,
    p1_branch: str,
    calibration_branch: str,
) -> None:
    full_root = _set_manual_root(monkeypatch, tmp_path)
    _write_parent(full_root)
    calibration_path = _write_lineage_calibration(
        full_root,
        lineage_name=lineage_name,
        branch_id=calibration_branch,
    )
    plan = integration.build_formal_v2_navsim_e120_planner_plan(
        _direct_p1_config(
            full_root,
            lineage_name=lineage_name,
            supervision=supervision,
        )
    )
    metadata_reads: list[tuple[Path, str]] = []

    def read_metadata(path: str | Path, *, required_branch_id: str) -> dict[str, object]:
        metadata_reads.append((Path(path), required_branch_id))
        payload = _direct_calibration_payload(embed_dim=2, branch_id=calibration_branch)
        return {
            key: payload[key]
            for key in ("schema", "phase", "protocol_version", "branch_id", "epoch", "roles", "parents")
        }

    def initialize_p1(**kwargs: object) -> dict[str, object]:
        validator = kwargs["calibration_checkpoint_validator"]
        assert callable(validator)
        validator(kwargs["calibration_checkpoint_path"])
        return {
            "stage": "p1",
            "training_stop_epoch": 80,
            "parent_epoch": 35,
            "role_state_shapes": {role: {"weight": [1]} for role in ("encoder", "predictor", "planner")},
            "lineage": plan.lineage,
        }

    monkeypatch.setattr(
        cvoi_value,
        "read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata",
        read_metadata,
    )
    monkeypatch.setattr(integration, "initialize_fresh_p1_direct_rank0", initialize_p1)
    state = integration.initialize_formal_v2_navsim_e120_planner_runtime(
        plan,
        modules=_modules(),
        optimizer=_State(1),
        scaler=_State(2),
        scheduler=_State(3),
        wd_scheduler=_State(4),
        rank=0,
        distributed=_BusEndpoint(_BroadcastBus(), rank=0),
        resume_path=None,
        resume_model_only=False,
    )

    assert state.start_epoch == 0
    assert plan.lineage["branch_id"] == p1_branch
    assert metadata_reads == [(calibration_path, calibration_branch)]


def test_no_cf_calibration_metadata_rejects_full_field_parent_before_warmstart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_root = _set_manual_root(monkeypatch, tmp_path)
    _write_parent(full_root)
    calibration_path = _lineage_calibration_path(full_root, lineage_name="no_cf")
    calibration_path.parent.mkdir(parents=True)
    payload = _direct_calibration_payload(embed_dim=2, branch_id="calibration_no_cf")
    payload["parents"]["field"]["branch_id"] = "field_full"
    torch.save(payload, calibration_path)
    plan = integration.build_formal_v2_navsim_e120_planner_plan(
        _direct_p1_config(
            full_root,
            lineage_name="no_cf",
            supervision="none",
        )
    )
    real_read = cvoi_value.read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata

    def read_metadata(path: str | Path, *, required_branch_id: str) -> dict[str, object]:
        assert required_branch_id == "calibration_no_cf"
        return real_read(path, required_branch_id=required_branch_id)

    def initialize_p1(**kwargs: object) -> dict[str, object]:
        validator = kwargs["calibration_checkpoint_validator"]
        assert callable(validator)
        validator(kwargs["calibration_checkpoint_path"])
        pytest.fail("invalid no-CF Calibration metadata must fail before warmstart")

    monkeypatch.setattr(
        cvoi_value,
        "read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata",
        read_metadata,
    )
    monkeypatch.setattr(integration, "initialize_fresh_p1_direct_rank0", initialize_p1)

    with pytest.raises(RuntimeError, match="parent"):
        integration.initialize_formal_v2_navsim_e120_planner_runtime(
            plan,
            modules=_modules(),
            optimizer=_State(1),
            scaler=_State(2),
            scheduler=_State(3),
            wd_scheduler=_State(4),
            rank=0,
            distributed=_BusEndpoint(_BroadcastBus(), rank=0),
            resume_path=None,
            resume_model_only=False,
        )


def test_same_run_resume_restores_all_states_without_rewarm_or_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    plan = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="unguided_planner"))
    source_modules = _modules(10.0)
    payload = _direct_checkpoint_payload(epoch=20, modules=source_modules)
    payload["optimizer"] = {"value": 21, "nested": {"tensor": torch.tensor([1.5, 2.5])}}
    payload["scaler"] = {"value": 22}
    payload["scheduler"] = {"value": 23}
    payload["wd_scheduler"] = {"value": 24}
    resume_path = tmp_path / "latest.pt"
    torch.save(payload, resume_path)
    plan = replace(plan, resume_checkpoint_path=resume_path)
    target_modules = _modules(-10.0)
    states = [
        _TensorState(-1, torch.zeros(2)),
        _State(-2),
        _State(-3),
        _State(-4),
    ]
    monkeypatch.setattr(
        warmstart,
        "apply_formal_v2_full_state_warmstart_direct",
        lambda *_: pytest.fail("same-run resume must not rewarm"),
    )

    state = integration.initialize_formal_v2_navsim_e120_planner_runtime(
        plan,
        modules=target_modules,
        optimizer=states[0],
        scaler=states[1],
        scheduler=states[2],
        wd_scheduler=states[3],
        rank=0,
        distributed=_BusEndpoint(_BroadcastBus(), rank=0),
        resume_path=resume_path,
        resume_model_only=False,
    )

    assert state.start_epoch == 20
    assert state.initialization == "same_run_full_state_resume"
    assert [item.value for item in states] == [21, 22, 23, 24]
    assert torch.equal(states[0].tensor, torch.tensor([1.5, 2.5]))
    assert all(
        torch.equal(target_modules[role].weight, source_modules[role].weight)
        for role in ("encoder", "predictor", "planner")
    )


def test_same_branch_different_run_id_is_rejected_on_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    plan = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="unguided_planner"))
    resume_path = tmp_path / "latest.pt"
    torch.save(_direct_checkpoint_payload(run_id="p0_uniform_s777"), resume_path)
    plan = replace(plan, resume_checkpoint_path=resume_path)

    with pytest.raises(RuntimeError, match="cross-run"):
        integration.initialize_formal_v2_navsim_e120_planner_runtime(
            plan,
            modules=_modules(),
            optimizer=_State(1),
            scaler=_State(2),
            scheduler=_State(3),
            wd_scheduler=_State(4),
            rank=0,
            distributed=_BusEndpoint(_BroadcastBus(), rank=0),
            resume_path=resume_path,
            resume_model_only=False,
        )


def test_resume_reads_once_on_rank0_and_broadcasts_model_and_training_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    plan = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="unguided_planner"))
    resume_path = tmp_path / "latest.pt"
    plan = replace(plan, resume_checkpoint_path=resume_path)
    bus = _BroadcastBus()
    reads: list[Path] = []

    def read(path: str | Path) -> dict[str, object]:
        reads.append(Path(path))
        return {"direct": True}

    def restore(payload: object, **kwargs: object) -> dict[str, object]:
        assert payload == {"direct": True}
        for index, role in enumerate(("encoder", "predictor", "planner"), start=11):
            kwargs["modules"][role].load_state_dict({"weight": torch.tensor([float(index)])})
        kwargs["optimizer"].load_state_dict(
            {"value": 21, "nested": {"tensor": torch.tensor([1.5, 2.5], dtype=torch.float64)}}
        )
        for index, name in enumerate(("scaler", "scheduler", "wd_scheduler"), start=22):
            kwargs[name].load_state_dict({"value": index})
        return {
            "start_epoch": 20,
            "cumulative_horizon_histogram": {0: 10, 1: 20, 2: 30, 3: 40, 4: 50},
            "role_state_shapes": {role: {"weight": [1]} for role in ("encoder", "predictor", "planner")},
        }

    monkeypatch.setattr(integration, "read_formal_v2_navsim_e120_direct_checkpoint", read)
    monkeypatch.setattr(integration, "restore_formal_v2_navsim_e120_direct_same_run_resume", restore)
    rank0_modules = _modules(0.0)
    rank0_states = [_TensorState(1, torch.zeros(2, dtype=torch.float64)), *[_State(v) for v in (2, 3, 4)]]
    rank0 = integration.initialize_formal_v2_navsim_e120_planner_runtime(
        plan,
        modules=rank0_modules,
        optimizer=rank0_states[0],
        scaler=rank0_states[1],
        scheduler=rank0_states[2],
        wd_scheduler=rank0_states[3],
        rank=0,
        distributed=_BusEndpoint(bus, rank=0),
        resume_path=resume_path,
        resume_model_only=False,
    )
    monkeypatch.setattr(
        integration,
        "read_formal_v2_navsim_e120_direct_checkpoint",
        lambda _: pytest.fail("non-rank-zero process must not read resume"),
    )
    rank1_modules = _modules(-10.0)
    rank1_states = [
        _TensorState(-1, torch.zeros(2, dtype=torch.float64)),
        *[_State(v) for v in (-2, -3, -4)],
    ]
    rank1 = integration.initialize_formal_v2_navsim_e120_planner_runtime(
        plan,
        modules=rank1_modules,
        optimizer=rank1_states[0],
        scaler=rank1_states[1],
        scheduler=rank1_states[2],
        wd_scheduler=rank1_states[3],
        rank=1,
        distributed=_BusEndpoint(bus, rank=1),
        resume_path=resume_path,
        resume_model_only=False,
    )

    assert reads == [resume_path]
    assert rank0.start_epoch == rank1.start_epoch == 20
    assert [state.value for state in rank1_states] == [21, 22, 23, 24]
    assert [rank1_modules[role].weight.item() for role in ("encoder", "predictor", "planner")] == [11, 12, 13]


def test_rank0_training_state_broadcast_moves_noncontiguous_cpu_tensor_to_collective_device() -> None:
    class _Capture:
        def __init__(self) -> None:
            self.tensors: list[torch.Tensor] = []

        def broadcast_object(self, value: object, *, src: int) -> object:
            assert src == 0
            return value

        def broadcast_tensor(self, tensor: torch.Tensor, *, src: int, name: str) -> None:
            assert src == 0
            assert name
            self.tensors.append(tensor)

    source_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)
    assert not source_tensor.is_contiguous()
    source_state = {"nested": {"tensor": source_tensor}}
    distributed = _Capture()

    returned = integration._broadcast_training_state_from_rank0(
        source_state,
        field="optimizer",
        rank=0,
        distributed=distributed,
        device=torch.device("meta"),
    )

    assert returned is source_state
    assert len(distributed.tensors) == 1
    assert distributed.tensors[0].device.type == "meta"
    assert distributed.tensors[0].is_contiguous()
    assert source_tensor.device.type == "cpu"
    assert not source_tensor.is_contiguous()


def test_nonrank_training_state_broadcast_unpacks_cpu_tensor_after_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = integration._BroadcastTensorLeaf(
        index=0,
        shape=(2, 3),
        dtype=torch.float32,
    )

    class _Receive:
        def broadcast_object(self, value: object, *, src: int) -> object:
            assert value is None
            assert src == 0
            return {"nested": {"tensor": descriptor}}

        def broadcast_tensor(self, tensor: torch.Tensor, *, src: int, name: str) -> None:
            assert src == 0
            assert name == "optimizer.tensor.0"
            tensor.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))

    original_cpu = torch.Tensor.cpu
    cpu_calls: list[torch.device] = []

    def record_cpu(tensor: torch.Tensor) -> torch.Tensor:
        cpu_calls.append(tensor.device)
        return original_cpu(tensor)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    restored = integration._broadcast_training_state_from_rank0(
        None,
        field="optimizer",
        rank=1,
        distributed=_Receive(),
        device=torch.device("cpu"),
    )

    tensor = restored["nested"]["tensor"]
    assert cpu_calls == [torch.device("cpu")]
    assert tensor.device.type == "cpu"
    assert tensor.is_contiguous()
    assert torch.equal(tensor, torch.arange(6, dtype=torch.float32).reshape(2, 3))


def test_checkpoint_save_uses_direct_payload_role_shapes_and_rank0_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    plan = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="unguided_planner"))
    exposure = _Exposure({0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
    built: list[dict[str, object]] = []
    writes: list[tuple[Path, bool]] = []

    def build(**kwargs: object) -> dict[str, object]:
        built.append(dict(kwargs))
        return {"role_state_shapes": {"encoder": {}, "predictor": {}, "planner": {}}}

    monkeypatch.setattr(integration, "build_formal_v2_navsim_e120_direct_checkpoint", build)
    monkeypatch.setattr(
        integration,
        "write_formal_v2_navsim_e120_direct_checkpoint",
        lambda path, payload, replace: writes.append((Path(path), replace)) or Path(path),
    )
    common = {
        "plan": plan,
        "modules": _modules(),
        "optimizer": _State(1),
        "scaler": _State(2),
        "scheduler": _State(3),
        "wd_scheduler": _State(4),
        "epoch": 10,
        "exposure": exposure,
        "device": torch.device("cpu"),
        "path": tmp_path / "e10.pt",
        "replace": False,
    }
    bus = _BroadcastBus()
    output = integration.save_formal_v2_navsim_e120_planner_checkpoint(
        rank=0,
        distributed=_BusEndpoint(bus, rank=0),
        **common,
    )
    assert (
        integration.save_formal_v2_navsim_e120_planner_checkpoint(
            rank=1,
            distributed=_BusEndpoint(bus, rank=1),
            **common,
        )
        is None
    )

    assert output == tmp_path / "e10.pt"
    assert built[0]["cumulative_horizon_histogram"] == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    assert built[0]["lineage"] == plan.lineage
    assert writes == [(tmp_path / "e10.pt", False)]


def test_reconcile_uses_direct_role_shapes_for_existing_milestone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    base = integration.build_formal_v2_navsim_e120_planner_plan(_config(root, stage="unguided_planner"))
    resume_path = tmp_path / "latest.pt"
    plan = replace(base, resume_checkpoint_path=resume_path)
    shapes = {role: {"weight": [1]} for role in ("encoder", "predictor", "planner")}
    restored = {
        "run_id": plan.run_id,
        "stage": "p0",
        "epoch": 20,
        "training_stop_epoch": 50,
        "schedule_epochs": 50,
        "selection_checkpoint_epochs": FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
        "cumulative_horizon_histogram": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        "role_state_shapes": shapes,
        "lineage": plan.lineage,
    }
    state = integration.FormalV2NavSimE120PlannerRuntimeState(
        start_epoch=20,
        initialization="same_run_full_state_resume",
        exposure=runtime.FormalV2NavSimE120HorizonExposureState(prior=restored["cumulative_horizon_histogram"]),
        initialization_result=restored,
    )
    milestone = tmp_path / "epoch-20.pt"
    milestone.touch()
    monkeypatch.setattr(
        integration,
        "formal_v2_navsim_e120_periodic_checkpoint_path",
        lambda *_args, **_kwargs: milestone,
    )
    monkeypatch.setattr(
        integration,
        "read_formal_v2_navsim_e120_direct_checkpoint",
        lambda _: {
            **restored,
            "role_state_shapes": shapes,
        },
    )

    output = integration.reconcile_formal_v2_navsim_e120_resume_milestone(
        plan=plan,
        runtime_state=state,
        save_every_freq=10,
        rank=0,
        distributed=_ObjectBroadcast(),
        publish_checkpoint=lambda *_: pytest.fail("matching milestone must not be overwritten"),
    )

    assert output == milestone


def test_frozen_field_model_broadcast_status_is_structural_and_exact() -> None:
    bus = _BroadcastBus()

    def load() -> PrefixDualValueModel:
        model = PrefixDualValueModel(embed_dim=2, hidden_dim=3)
        for parameter in model.parameters():
            parameter.data.fill_(0.25)
        return model

    rank0 = integration.load_formal_v2_navsim_e120_field_model_on_rank0(
        loader=load,
        rank=0,
        distributed=_BusEndpoint(bus, rank=0),
        device=torch.device("cpu"),
    )
    rank1 = integration.load_formal_v2_navsim_e120_field_model_on_rank0(
        loader=lambda: pytest.fail("rank one must not load Field"),
        rank=1,
        distributed=_BusEndpoint(bus, rank=1),
        device=torch.device("cpu"),
    )

    status = bus.objects[0]
    assert rank0 is not None and rank1 is not None
    assert set(status) == {
        "ok",
        "present",
        "architecture",
        "state_keys",
        "state_shapes",
        "error_type",
        "error_message",
    }
    assert all("sha256" not in key.lower() and key.lower() != "sha" for key in status)
    assert status["state_keys"] == sorted(rank0.state_dict())
    assert status["state_shapes"] == {key: list(value.shape) for key, value in sorted(rank0.state_dict().items())}
    assert all(torch.equal(rank0.state_dict()[key], rank1.state_dict()[key]) for key in rank0.state_dict())


def test_direct_calibration_model_loader_strict_loads_without_legacy_audit_or_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)
    calibration_path = _write_calibration(root)
    config = _config(root, stage="guided_planner")
    direct_reads: list[Path] = []
    direct_reader = cvoi_value.read_cvoi_navsim_e120_direct_value_checkpoint

    assert {"_load_configured_cvoi_audit", "_sha256_file"}.isdisjoint(vars(legacy_cvoi_runtime))
    assert "load_prefix_dual_value_checkpoint" not in vars(cvoi_value)

    def read_direct(path: str | Path, **kwargs: object) -> dict[str, object]:
        direct_reads.append(Path(path))
        return direct_reader(path, **kwargs)

    monkeypatch.setattr(cvoi_value, "read_cvoi_navsim_e120_direct_value_checkpoint", read_direct)
    model = integration.load_formal_v2_navsim_e120_calibration_model_direct(
        config,
        embed_dim=2,
        device=torch.device("cpu"),
    )

    assert model is not None
    assert direct_reads == [calibration_path]
    assert model.embed_dim == 2
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    payload = direct_reader(
        calibration_path,
        required_phase="field_calibrated",
        required_branch_id="calibration_full",
    )
    assert all(torch.equal(model.state_dict()[key], value) for key, value in payload["state_dict"].items())


@pytest.mark.parametrize(
    ("lineage_name", "supervision", "p1_branch", "calibration_branch"),
    _P1_LINEAGE_CASES,
)
def test_direct_calibration_loader_uses_matching_p1_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage_name: str,
    supervision: str,
    p1_branch: str,
    calibration_branch: str,
) -> None:
    full_root = _set_manual_root(monkeypatch, tmp_path)
    calibration_path = _write_lineage_calibration(
        full_root,
        lineage_name=lineage_name,
        branch_id=calibration_branch,
    )
    config = _direct_p1_config(
        full_root,
        lineage_name=lineage_name,
        supervision=supervision,
    )
    assert config.cvoi.ablation_signature.branch_id == p1_branch
    payload = _direct_calibration_payload(embed_dim=2, branch_id=calibration_branch)
    calls: list[tuple[Path, str, str]] = []

    def fake_read(
        path: str | Path,
        *,
        required_phase: str,
        required_branch_id: str,
        **_: object,
    ) -> dict[str, object]:
        calls.append((Path(path), required_phase, required_branch_id))
        return payload

    monkeypatch.setattr(cvoi_value, "read_cvoi_navsim_e120_direct_value_checkpoint", fake_read)
    model = integration.load_formal_v2_navsim_e120_calibration_model_direct(
        config,
        embed_dim=2,
        device=torch.device("cpu"),
    )

    assert model is not None
    assert calls == [(calibration_path, "field_calibrated", calibration_branch)]


def test_no_cf_calibration_loader_rejects_full_path_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_root = _set_manual_root(monkeypatch, tmp_path)
    full_calibration = _write_lineage_calibration(
        full_root,
        lineage_name="full",
        branch_id="calibration_full",
    )
    config = _direct_p1_config(
        full_root,
        lineage_name="no_cf",
        supervision="none",
    )
    config.cvoi.field_checkpoint = str(full_calibration)
    monkeypatch.setattr(
        cvoi_value,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("path drift must fail before reading Calibration"),
    )

    with pytest.raises(ValueError, match="Calibration"):
        integration.load_formal_v2_navsim_e120_calibration_model_direct(
            config,
            embed_dim=2,
            device=torch.device("cpu"),
        )


def test_direct_p0_calibration_model_loader_returns_none_without_reading_value_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _set_manual_root(monkeypatch, tmp_path)

    def lineage_resolver(*_args: object, **_kwargs: object) -> object:
        pytest.fail("P0 must not parse a Value lineage")

    monkeypatch.setattr(
        integration,
        "resolve_cvoi_manual_value_lineage",
        lineage_resolver,
        raising=False,
    )
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "resolve_cvoi_manual_value_lineage",
        lineage_resolver,
    )
    monkeypatch.setattr(
        cvoi_value,
        "read_cvoi_navsim_e120_direct_value_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("P0 must not read a Calibration artifact"),
    )

    assert (
        integration.load_formal_v2_navsim_e120_calibration_model_direct(
            _config(root, stage="unguided_planner"),
            embed_dim=2,
            device=torch.device("cpu"),
        )
        is None
    )


def test_planner_line_uses_direct_plan_signature_and_strict_lifecycle() -> None:
    source = PLANNER_LINE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_formal_v2_navsim_e120_planner_plan_on_rank0"
    ]

    assert len(calls) == 1
    assert {keyword.arg for keyword in calls[0].keywords} == {
        "rank",
        "distributed",
        "resume_path",
    }
    assert "initialize_formal_v2_navsim_e120_planner_runtime" in source
    assert "save_formal_v2_navsim_e120_planner_checkpoint" in source
    assert "reconcile_formal_v2_navsim_e120_resume_milestone" in source


def test_planner_line_e120_loader_lambda_calls_only_the_direct_calibration_helper() -> None:
    tree = ast.parse(PLANNER_LINE.read_text(encoding="utf-8"))
    matching: list[ast.If] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        calls = [
            child
            for statement in node.body
            for child in ast.walk(statement)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        ]
        if any(call.func.id == "load_formal_v2_navsim_e120_field_model_on_rank0" for call in calls):
            matching.append(node)

    assert len(matching) == 1
    direct_calls = [
        node.func.id
        for statement in matching[0].body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert "load_formal_v2_navsim_e120_calibration_model_direct" in direct_calls
    assert "load_cvoi_dual_value_model" not in direct_calls


def test_integration_source_has_no_legacy_proof_or_selection_symbols() -> None:
    source = Path(integration.__file__).read_text(encoding="utf-8")
    forbidden = {
        "strict_artifact_sha256",
        "canonical_json_document_sha256",
        "read_formal_v2_full_state_warmstart_receipt",
        "validate_formal_v2_full_state_warmstart_receipt",
        "read_formal_v2_navsim_epdms_selection_receipt",
        "formal_v2_navsim_epdms_selection_receipt_sha256",
        "role_state_sha256",
        "_open_regular_nofollow",
        "_selection_validator",
        "cvoi_planner_runtime",
    }

    assert all(symbol not in source for symbol in forbidden)


def test_manual_periodic_checkpoint_path_uses_training_line_e_epoch_contract(tmp_path: Path) -> None:
    assert integration.formal_v2_navsim_e120_periodic_checkpoint_path(tmp_path, epoch=35) == tmp_path / "e35.pt"

    with pytest.raises(ValueError, match="positive"):
        integration.formal_v2_navsim_e120_periodic_checkpoint_path(tmp_path, epoch=0)


def test_navsim_profile_selection_and_milestone_helpers_remain_exact() -> None:
    assert integration.planner_uses_navsim_e120_runtime(
        SimpleNamespace(protocol_version=runtime.FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION)
    )
    assert not integration.planner_uses_legacy_open_loop_selection(
        SimpleNamespace(protocol_version=runtime.FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION)
    )
    expected = set(range(10, 51, 10)) | set(FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS)
    actual = {
        epoch
        for epoch in range(1, 51)
        if integration.formal_v2_navsim_e120_milestone_due(
            epoch,
            save_every_freq=10,
            selection_checkpoint_epochs=FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
        )
    }
    assert actual == expected


def test_torch_distributed_adapter_is_explicit_for_single_and_multi_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = integration.TorchDistributedAdapter(rank=0, world_size=1)
    assert adapter.broadcast_object({"ok": True}, src=0) == {"ok": True}
    adapter.broadcast_tensor(torch.ones(1), src=0, name="encoder.weight")
    with pytest.raises(RuntimeError, match="rank-zero value"):
        adapter.broadcast_object(None, src=0)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="requires torch.distributed"):
        integration.TorchDistributedAdapter(rank=0, world_size=2)
