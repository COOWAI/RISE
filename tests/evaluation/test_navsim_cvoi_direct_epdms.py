"""Direct NavTest CVoI EPDMS bindings for the NavSim Agent."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np
import pytest
import torch
import yaml

from app.vjepa_cowa_world_model.evaluation.cvoi_direct_epdms import CvoiDirectEpdmsProjection
from app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity import observation_key_tensor, unsigned_seed_tensor
from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training import cvoi_formal_v2_navsim_e120_runtime as e120_runtime
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage
from app.vjepa_cowa_world_model.training.cvoi_execution import cvoi_sample_seed
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_E120_LAMBDA_GRID
from app.vjepa_cowa_world_model.training.cvoi_gate_pipeline import CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import NAVTRAIN_GATE_PROTOCOL_ID
from app.vjepa_cowa_world_model.training.cvoi_value import build_cvoi_navsim_e120_direct_value_checkpoint
from app.vjepa_cowa_world_model.training.sequential_budget_control import (
    CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
    SequentialRolloutGate,
)
from app.vjepa_cowa_world_model.training.sequential_gate_training import (
    SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3,
    save_sequential_gate_checkpoint,
)
from tests.evaluation.test_navsim_stage3_alignment import _load_navsim_agent_module

_DIRECT_EVALUATION_SEED = 239
_DIRECT_OBSERVATION_KEY = "c" * 64
_DIRECT_SCENARIO_TOKEN = "navtest-direct-a"


def _agent_module():
    return _load_navsim_agent_module()


def _agent_class():
    return _agent_module().VJEPAWorldModelAgent


def _module_globals() -> dict[str, object]:
    return _agent_class().initialize.__globals__


def _frozen(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def _tiny_roles(*, encoder_weight: float = 1.0, planner_weight: float = 3.0) -> dict[str, torch.nn.Module]:
    roles = {
        "encoder": torch.nn.Linear(1, 1, bias=False),
        "predictor": torch.nn.Linear(1, 1, bias=False),
        "planner": torch.nn.Linear(1, 1, bias=False),
    }
    with torch.no_grad():
        roles["encoder"].weight.fill_(encoder_weight)
        roles["predictor"].weight.fill_(2.0)
        roles["planner"].weight.fill_(planner_weight)
    return roles


def _planner_payload(
    *,
    stage: str,
    branch_id: str,
    encoder_weight: float = 1.0,
    planner_weight: float = 3.0,
) -> dict[str, object]:
    return e120_runtime.build_formal_v2_navsim_e120_direct_checkpoint(
        modules=_tiny_roles(encoder_weight=encoder_weight, planner_weight=planner_weight),
        optimizer={"state": {}, "param_groups": [{"lr": 1e-4}]},
        scaler={},
        scheduler={"last_epoch": 1},
        wd_scheduler={"last_epoch": 1},
        run_id=f"{branch_id}_direct_eval",
        stage=stage,
        epoch=35 if stage == "p0" else 5,
        training_stop_epoch=50 if stage == "p0" else 80,
        schedule_epochs=50 if stage == "p0" else 80,
        selection_checkpoint_epochs=(
            e120_runtime.FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS
            if stage == "p0"
            else e120_runtime.FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS
        ),
        cumulative_horizon_histogram={0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
        lineage=e120_runtime.build_formal_v2_navsim_e120_direct_lineage(
            stage=stage,
            branch_id=branch_id,
        ),
    )


def _value_payload(*, phase: str, branch_id: str) -> dict[str, object]:
    lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase=phase,
        branch_id=branch_id,
    )
    return build_cvoi_navsim_e120_direct_value_checkpoint(
        PrefixDualValueModel(embed_dim=1, hidden_dim=2),
        phase=phase,
        branch_id=branch_id,
        epoch=3,
        parents=cvoi_manual_lineage.build_cvoi_manual_value_parents(lineage, phase),
    )


def _write_selected_checkpoint(
    root: Path,
    *,
    stage: str,
    payload: Mapping[str, object],
) -> tuple[Path, Path]:
    stage_dir = root / stage
    handoff_dir = root / "handoff"
    stage_dir.mkdir(parents=True, exist_ok=True)
    handoff_dir.mkdir(parents=True, exist_ok=True)
    target = stage_dir / "candidate.pt"
    torch.save(dict(payload), target)
    selected = handoff_dir / f"{stage}_selected.pt"
    selected.symlink_to(target)
    return selected, target


def _write_gate(
    path: Path,
    *,
    oracle_sha256: str,
    oracle_lineage: str,
    feature_mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_sequential_gate_checkpoint(
        path,
        SequentialRolloutGate(latent_dim=1, hidden_dim=2),
        lambda_grid=FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
        provenance={
            "gate_pipeline": CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION,
            "oracle_protocol": NAVTRAIN_GATE_PROTOCOL_ID,
            "oracle_sha256": oracle_sha256,
            "oracle_lineage": oracle_lineage,
            "gate_feature_schema": CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
            "gate_feature_mode": feature_mode,
        },
        protocol_version=SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3,
    )


def _artifact_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    branch: str,
    evaluation_mode: str,
    horizon: int | None = None,
) -> tuple[CvoiDirectEpdmsProjection, dict[str, Path]]:
    full_root = (tmp_path / "cvoi_manual_full").resolve()
    ablation_root = (tmp_path / "cvoi_manual_ablation").resolve()
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        (tmp_path / "unrelated-module-authority").resolve(),
    )
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_ABLATION_RESULTS_ROOT", ablation_root)

    p0_selected, p0_target = _write_selected_checkpoint(
        full_root,
        stage="p0",
        payload=_planner_payload(stage="p0", branch_id="p0_uniform"),
    )
    paths: dict[str, Path] = {"p0_selected": p0_selected, "p0_target": p0_target}
    common = {
        "branch": branch,
        "split": "navtest",
        "protocol": "epdms_v2_one_stage_navtest",
        "evaluation_mode": evaluation_mode,
        "horizon": horizon,
        "guidance_steps": 0 if evaluation_mode == "p0_forced" else 2,
        "training_config_path": (tmp_path / "training.yaml").resolve(),
        "encoder_checkpoint_path": (tmp_path / "encoder.pt").resolve(),
        "scenario_manifest_path": (tmp_path / "navtest.jsonl").resolve(),
        "output_directory": (tmp_path / "output" / ("controller" if horizon is None else f"h{horizon}")).resolve(),
    }
    if evaluation_mode == "p0_forced":
        return (
            CvoiDirectEpdmsProjection(
                **common,
                p0_planner_checkpoint_path=p0_selected,
            ),
            paths,
        )

    value_name = branch if branch in {"no_cf", "hazard_only", "quality_only"} else "full"
    value_root = full_root if value_name == "full" else ablation_root / value_name
    p1_branch = f"p1_{value_name}"
    p1_selected, p1_target = _write_selected_checkpoint(
        value_root,
        stage="p1",
        payload=_planner_payload(stage="p1", branch_id=p1_branch, planner_weight=4.0),
    )
    calibration_path = value_root / "handoff/calibration.pt"
    torch.save(_value_payload(phase="field_calibrated", branch_id=f"calibration_{value_name}"), calibration_path)
    paths.update(
        {
            "p1_selected": p1_selected,
            "p1_target": p1_target,
            "calibration": calibration_path,
        }
    )
    if evaluation_mode == "p1_field_forced":
        return (
            CvoiDirectEpdmsProjection(
                **common,
                calibration_checkpoint_path=calibration_path,
                p1_planner_checkpoint_path=p1_selected,
            ),
            paths,
        )

    stop_path = value_root / "handoff/stop.pt"
    torch.save(_value_payload(phase="stop_calibrated", branch_id=f"stop_{value_name}"), stop_path)
    feature_mode = branch if branch in {"without_field", "without_stop", "without_value_summary"} else "full"
    gate_root = full_root if branch == "full" else ablation_root / branch
    gate_path = gate_root / "handoff/gate.pt"
    oracle_sha256 = "a" * 64
    _write_gate(
        gate_path,
        oracle_sha256=oracle_sha256,
        oracle_lineage="p1_no_cf" if branch == "no_cf" else "p1_full",
        feature_mode=feature_mode,
    )
    paths.update({"stop": stop_path, "gate": gate_path})
    return (
        CvoiDirectEpdmsProjection(
            **common,
            p0_planner_checkpoint_path=p0_selected,
            calibration_checkpoint_path=calibration_path,
            p1_planner_checkpoint_path=p1_selected,
            stop_checkpoint_path=stop_path,
            gate_checkpoint_path=gate_path,
            gate_feature_mode=feature_mode,
            oracle_sha256=oracle_sha256,
        ),
        paths,
    )


def _artifact_presence(bundle: object) -> set[str]:
    return {
        field
        for field in ("p0_checkpoint", "calibration_checkpoint", "p1_checkpoint", "stop_checkpoint", "gate_checkpoint")
        if getattr(bundle, field) is not None
    }


def test_direct_constructor_exposes_only_one_effective_config_parameter() -> None:
    parameters = inspect.signature(_agent_class().__init__).parameters

    assert {name for name in parameters if name.startswith("cvoi_direct_")} == {"cvoi_direct_epdms_config_path"}
    assert not any(name.startswith("cvoi_official_") for name in parameters)
    assert {
        "CvoiOfficialEffectivePolicy",
        "resolve_cvoi_official_effective_policy",
        "open_cvoi_navsim_artifact_handles",
    }.isdisjoint(_module_globals())


def test_direct_constructor_is_mutually_exclusive_with_manual_request() -> None:
    with pytest.raises(ValueError, match="direct.*manual|mutually exclusive"):
        _agent_class()(
            device="cpu",
            cvoi_direct_epdms_config_path="/direct/effective.json",
            cvoi_manual_navtrain_gate_config_path="/manual/scorer.json",
        )


def test_direct_constructor_preserves_only_the_effective_path() -> None:
    agent = _agent_class()(device="cpu", cvoi_direct_epdms_config_path="/direct/effective.json")

    assert agent._cvoi_direct_epdms_config_path == "/direct/effective.json"
    assert agent._cvoi_direct_runtime is None
    assert agent._cvoi_manual_runtime is None
    assert not hasattr(agent, "_cvoi_official_request")
    assert not hasattr(agent, "_cvoi_official_policy")


def test_direct_initialize_loads_effective_projection_and_never_projects_public_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    projection = CvoiDirectEpdmsProjection(
        branch="full",
        split="navtest",
        protocol="epdms_v2_one_stage_navtest",
        evaluation_mode="controller",
        horizon=None,
        guidance_steps=2,
        training_config_path=(tmp_path / "training.yaml").resolve(),
        encoder_checkpoint_path=(tmp_path / "encoder.pt").resolve(),
        scenario_manifest_path=(tmp_path / "navtest.jsonl").resolve(),
        output_directory=(tmp_path / "output").resolve(),
        p0_planner_checkpoint_path=(tmp_path / "p0.pt").resolve(),
        calibration_checkpoint_path=(tmp_path / "calibration.pt").resolve(),
        p1_planner_checkpoint_path=(tmp_path / "p1.pt").resolve(),
        stop_checkpoint_path=(tmp_path / "stop.pt").resolve(),
        gate_checkpoint_path=(tmp_path / "gate.pt").resolve(),
        gate_feature_mode="full",
        oracle_sha256="a" * 64,
    )
    agent_class = _agent_class()
    globals_dict = agent_class.initialize.__globals__
    events: list[object] = []

    def load_effective(path: Path) -> CvoiDirectEpdmsProjection:
        events.append(("load_effective", path))
        return projection

    def stop_at_preflight(value: CvoiDirectEpdmsProjection) -> object:
        events.append(("preflight", value))
        raise ValueError("direct artifact preflight sentinel")

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("public projection/model construction must not run")

    monkeypatch.setitem(globals_dict, "load_cvoi_direct_epdms_projection", load_effective)
    monkeypatch.setitem(globals_dict, "_read_cvoi_direct_epdms_artifacts", stop_at_preflight)
    monkeypatch.setitem(globals_dict, "project_cvoi_direct_epdms_run", forbidden)
    monkeypatch.setitem(globals_dict, "init_encoder", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    effective_path = (tmp_path / "effective.json").resolve()
    agent = agent_class(device="cuda", cvoi_direct_epdms_config_path=str(effective_path))

    with pytest.raises(ValueError, match="artifact preflight sentinel"):
        agent.initialize()

    assert events == [("load_effective", effective_path), ("preflight", projection)]
    assert not hasattr(projection, "oracle_path")


def test_direct_agent_imports_only_the_effective_loader_not_the_public_projector() -> None:
    globals_dict = _module_globals()

    assert "load_cvoi_direct_epdms_projection" in globals_dict
    assert "project_cvoi_direct_epdms_run" not in globals_dict
    context_type = globals_dict["_DirectEpdmsRuntimeContext"]
    fields = set(getattr(context_type, "_fields", ()))
    if not fields:
        fields = set(getattr(context_type, "__annotations__", {}))
    assert "oracle_path" not in fields


@pytest.mark.parametrize(
    ("device", "cuda_available", "message"),
    [
        ("cpu", True, "requires a CUDA device"),
        ("cuda", False, "requires available CUDA|CUDA.*unavailable"),
    ],
)
def test_direct_initialize_requires_available_cuda_before_reading_any_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device: str,
    cuda_available: bool,
    message: str,
) -> None:
    agent_class = _agent_class()
    globals_dict = agent_class.initialize.__globals__
    projection_reads: list[Path] = []

    def forbidden_projection_read(path: Path) -> object:
        projection_reads.append(path)
        raise AssertionError("direct device validation must run before projection I/O")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setitem(globals_dict, "load_cvoi_direct_epdms_projection", forbidden_projection_read)
    effective_path = (tmp_path / "effective.json").resolve()
    agent = agent_class(device=device, cvoi_direct_epdms_config_path=str(effective_path))

    with pytest.raises((ValueError, RuntimeError), match=message):
        agent.initialize()

    assert projection_reads == []


def test_direct_initialize_uses_only_the_context_full_state_encoder_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_config_path = Path(__file__).resolve().parents[2] / "configs/train/navsim/cvoi_manual_full/05_p1_full.yaml"
    projection = replace(
        _compute_projection(
            tmp_path,
            evaluation_mode="controller",
            horizon=None,
            branch="full",
        ),
        training_config_path=source_config_path,
    )
    agent_class = _agent_class()
    globals_dict = agent_class.initialize.__globals__
    artifacts_type = globals_dict["_DirectEpdmsArtifacts"]
    artifacts = artifacts_type(None, None, None, None, None)
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(globals_dict, "load_cvoi_direct_epdms_projection", lambda path: projection)
    monkeypatch.setitem(globals_dict, "_read_cvoi_direct_epdms_artifacts", lambda value: artifacts)
    monkeypatch.setitem(
        globals_dict,
        "read_cvoi_direct_epdms_scenario_manifest",
        lambda path: {_DIRECT_OBSERVATION_KEY: _DIRECT_SCENARIO_TOKEN},
    )
    monkeypatch.setitem(globals_dict, "require_cvoi_stage_for_entry", lambda *args, **kwargs: None)
    monkeypatch.setitem(globals_dict, "validate_cvoi_sequential_runtime_config", lambda config: None)
    monkeypatch.setitem(globals_dict, "cvoi_enabled", lambda config: True)
    monkeypatch.setitem(globals_dict, "value_planning_enabled", lambda config: False)
    monkeypatch.setitem(globals_dict, "get_encoder_embed_dim", lambda encoder: 1)
    monkeypatch.setattr(agent_class, "_parse_inference_params", lambda self: None)

    def context_factory(config: object, device: torch.device) -> torch.nn.Module:
        calls.append(("context_full_state", device))
        return torch.nn.Linear(1, 1)

    def forbidden_generic_factory(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls.append(("generic", None))
        raise AssertionError("direct initialize must not construct a target encoder")

    def stop_after_encoder(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("direct encoder factory sentinel")

    monkeypatch.setitem(
        globals_dict,
        "init_context_encoder_for_full_state_warmstart",
        context_factory,
    )
    monkeypatch.setitem(globals_dict, "init_encoder", forbidden_generic_factory)
    monkeypatch.setattr(agent_class, "_initialize_stage12", stop_after_encoder)
    agent = agent_class(
        device="cuda",
        cvoi_direct_epdms_config_path=str((tmp_path / "effective.json").resolve()),
    )

    with pytest.raises(RuntimeError, match="factory sentinel"):
        agent.initialize()

    assert calls == [("context_full_state", torch.device("cuda"))]


def test_non_direct_stage12_initialize_keeps_the_generic_encoder_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = (tmp_path / "ordinary.yaml").resolve()
    config_path.write_text("{}\n", encoding="utf-8")
    agent_class = _agent_class()
    globals_dict = agent_class.initialize.__globals__
    parsed_config = SimpleNamespace(multiview=SimpleNamespace(enabled=False))
    calls: list[str] = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(globals_dict, "parse_training_config", lambda raw: parsed_config)
    monkeypatch.setitem(globals_dict, "require_cvoi_stage_for_entry", lambda *args, **kwargs: None)
    monkeypatch.setitem(globals_dict, "validate_cvoi_sequential_runtime_config", lambda config: None)
    monkeypatch.setitem(globals_dict, "cvoi_enabled", lambda config: False)
    monkeypatch.setitem(globals_dict, "value_planning_enabled", lambda config: False)
    monkeypatch.setitem(globals_dict, "get_encoder_embed_dim", lambda encoder: 1)
    monkeypatch.setattr(agent_class, "_parse_inference_params", lambda self: None)

    def generic_factory(config: object, device: torch.device) -> tuple[torch.nn.Module, None]:
        calls.append("init_encoder")
        return torch.nn.Linear(1, 1), None

    def forbidden_context_factory(*args: object, **kwargs: object) -> object:
        del args, kwargs
        calls.append("context_full_state")
        raise AssertionError("ordinary initialize must retain the generic encoder factory")

    def stop_after_encoder(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("ordinary encoder factory sentinel")

    monkeypatch.setitem(globals_dict, "init_encoder", generic_factory)
    monkeypatch.setitem(
        globals_dict,
        "init_context_encoder_for_full_state_warmstart",
        forbidden_context_factory,
    )
    monkeypatch.setattr(agent_class, "_initialize_stage12", stop_after_encoder)
    agent = agent_class(device="cpu", training_config_path=str(config_path))

    with pytest.raises(RuntimeError, match="factory sentinel"):
        agent.initialize()

    assert calls == ["init_encoder"]


@pytest.mark.parametrize(
    ("config_relative_path", "branch", "mode", "horizon", "expected_controller_lineage"),
    [
        (
            "configs/train/navsim/cvoi_manual_full/05_p1_full.yaml",
            "full",
            "controller",
            None,
            "value_guided",
        ),
    ],
)
def test_direct_runtime_config_is_a_private_clone_of_the_original_guided_planner_yaml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config_relative_path: str,
    branch: str,
    mode: str,
    horizon: int | None,
    expected_controller_lineage: str,
) -> None:
    source_path = Path(__file__).resolve().parents[2] / config_relative_path
    raw_config = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_config = _module_globals()["parse_training_config"](raw_config)
    source_identity = {
        "stage": source_config.cvoi.stage,
        "evaluation_mode": source_config.cvoi.evaluation_mode,
        "controller_lineage": source_config.cvoi.controller_lineage,
        "guidance_steps": source_config.cvoi.guidance_steps,
        "world_model_checkpoint": source_config.cvoi.world_model_checkpoint,
        "output_checkpoint": source_config.cvoi.output_checkpoint,
    }
    projection, _ = _artifact_projection(
        tmp_path,
        monkeypatch,
        branch=branch,
        evaluation_mode=mode,
        horizon=horizon,
    )
    projection = replace(projection, training_config_path=source_path)

    runtime_config = _module_globals()["_build_cvoi_direct_epdms_training_config"](
        source_config,
        projection,
    )

    assert raw_config["cvoi"]["stage"] == "guided_planner"
    assert source_identity["stage"] == "guided_planner"
    assert {
        "stage": source_config.cvoi.stage,
        "evaluation_mode": source_config.cvoi.evaluation_mode,
        "controller_lineage": source_config.cvoi.controller_lineage,
        "guidance_steps": source_config.cvoi.guidance_steps,
        "world_model_checkpoint": source_config.cvoi.world_model_checkpoint,
        "output_checkpoint": source_config.cvoi.output_checkpoint,
    } == source_identity
    assert runtime_config is not source_config
    assert runtime_config.cvoi is not source_config.cvoi
    assert runtime_config.meta is not source_config.meta
    assert runtime_config.cvoi.stage == "evaluation"
    assert runtime_config.cvoi.evaluation_mode == mode
    assert runtime_config.cvoi.controller_lineage == expected_controller_lineage
    assert runtime_config.cvoi.guidance_steps == projection.guidance_steps
    assert runtime_config.cvoi.oracle_path is None


@pytest.mark.parametrize(
    ("branch", "expected_fields"),
    [
        ("full", {"p0_checkpoint", "calibration_checkpoint", "p1_checkpoint", "stop_checkpoint", "gate_checkpoint"}),
    ],
)
def test_direct_controller_preflight_reads_the_exact_branch_artifact_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    branch: str,
    expected_fields: set[str],
) -> None:
    projection, _ = _artifact_projection(
        tmp_path,
        monkeypatch,
        branch=branch,
        evaluation_mode="controller",
    )

    bundle = _module_globals()["_read_cvoi_direct_epdms_artifacts"](projection)

    assert _artifact_presence(bundle) == expected_fields
    assert not hasattr(bundle, "oracle_path")


def _nested_mapping_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(key for key in value if isinstance(key, str)),
            *(nested for child in value.values() for nested in _nested_mapping_keys(child)),
        }
    if isinstance(value, (tuple, list)):
        return {nested for child in value for nested in _nested_mapping_keys(child)}
    return set()


@pytest.mark.parametrize(
    ("branch", "mode", "horizon"),
    [
        ("full", "controller", None),
    ],
)
def test_direct_preflight_retains_deployment_state_without_any_resume_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    branch: str,
    mode: str,
    horizon: int | None,
) -> None:
    projection, _ = _artifact_projection(
        tmp_path,
        monkeypatch,
        branch=branch,
        evaluation_mode=mode,
        horizon=horizon,
    )

    bundle = _module_globals()["_read_cvoi_direct_epdms_artifacts"](projection)

    forbidden_resume_fields = {"optimizer", "scaler", "scheduler", "wd_scheduler"}
    for field in (
        "p0_checkpoint",
        "calibration_checkpoint",
        "p1_checkpoint",
        "stop_checkpoint",
        "gate_checkpoint",
    ):
        payload = getattr(bundle, field)
        if payload is not None:
            assert _nested_mapping_keys(payload).isdisjoint(forbidden_resume_fields)


def test_direct_controller_p0_preflight_retains_only_crosscheck_and_planner_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    projection, _ = _artifact_projection(
        tmp_path,
        monkeypatch,
        branch="full",
        evaluation_mode="controller",
    )

    bundle = _module_globals()["_read_cvoi_direct_epdms_artifacts"](projection)
    p0 = bundle.p0_checkpoint

    assert p0 is not None
    assert set(p0) == {
        "stage",
        "lineage",
        "protocol_version",
        "role_state_shapes",
        "encoder_sha256",
        "planner",
    }
    assert set(p0["role_state_shapes"]) == {"encoder", "predictor", "planner"}
    assert "encoder" not in p0
    assert "predictor" not in p0
    assert isinstance(p0["encoder_sha256"], str) and len(p0["encoder_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_p1",
        "aliased_planners",
        "wrong_p1_stage",
        "wrong_p1_branch",
        "wrong_p1_protocol",
        "wrong_calibration_parent",
        "wrong_stop_parent",
        "wrong_gate_digest",
        "wrong_gate_lineage",
        "wrong_gate_feature_mode",
        "encoder_state_drift",
    ],
)
def test_direct_controller_preflight_fails_closed_on_artifact_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    projection, paths = _artifact_projection(
        tmp_path,
        monkeypatch,
        branch="full",
        evaluation_mode="controller",
    )
    if mutation == "missing_p1":
        paths["p1_target"].unlink()
    elif mutation == "aliased_planners":
        projection = replace(
            projection,
            p1_planner_checkpoint_path=projection.p0_planner_checkpoint_path,
        )
    elif mutation in {"wrong_p1_stage", "wrong_p1_branch", "encoder_state_drift"}:
        if mutation == "wrong_p1_stage":
            payload = _planner_payload(stage="p0", branch_id="p0_uniform")
        else:
            payload = _planner_payload(
                stage="p1",
                branch_id="p1_no_cf" if mutation == "wrong_p1_branch" else "p1_full",
                encoder_weight=9.0 if mutation == "encoder_state_drift" else 1.0,
            )
        torch.save(payload, paths["p1_target"])
    elif mutation == "wrong_p1_protocol":
        payload = torch.load(paths["p1_target"], map_location="cpu", weights_only=False)
        payload["protocol_version"] = "legacy_v1"
        torch.save(payload, paths["p1_target"])
    elif mutation in {"wrong_calibration_parent", "wrong_stop_parent"}:
        role = "calibration" if mutation == "wrong_calibration_parent" else "stop"
        payload = torch.load(paths[role], map_location="cpu", weights_only=False)
        if role == "calibration":
            payload["parents"]["field"]["branch_id"] = "field_no_cf"
        else:
            payload["parents"]["guided_planner"]["branch_id"] = "p1_no_cf"
        torch.save(payload, paths[role])
    else:
        payload = torch.load(paths["gate"], map_location="cpu", weights_only=False)
        provenance_key, drifted = {
            "wrong_gate_digest": ("oracle_sha256", "b" * 64),
            "wrong_gate_lineage": ("oracle_lineage", "p1_no_cf"),
            "wrong_gate_feature_mode": ("gate_feature_mode", "without_stop"),
        }[mutation]
        payload["provenance"][provenance_key] = drifted
        torch.save(payload, paths["gate"])

    with pytest.raises(
        (FileNotFoundError, ValueError),
        match="P1|p1|alias|stage|branch|protocol|parent|Gate|gate|Oracle|oracle|encoder",
    ):
        _module_globals()["_read_cvoi_direct_epdms_artifacts"](projection)


@pytest.mark.parametrize(
    ("branch", "mode", "horizon", "expected_slots", "expected_planner_factory_calls"),
    [
        (
            "full",
            "controller",
            None,
            {
                "_cvoi_direct_p0_planner",
                "_cvoi_direct_p1_planner",
                "_cvoi_direct_field_value",
                "_cvoi_direct_stop_value",
                "_cvoi_direct_gate",
            },
            2,
        ),
    ],
)
def test_direct_stage12_initialization_bypasses_generic_loaders_and_populates_only_exact_mode_slots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    branch: str,
    mode: str,
    horizon: int | None,
    expected_slots: set[str],
    expected_planner_factory_calls: int,
) -> None:
    projection, _ = _artifact_projection(
        tmp_path,
        monkeypatch,
        branch=branch,
        evaluation_mode=mode,
        horizon=horizon,
    )
    agent_class = _agent_class()
    globals_dict = agent_class._initialize_stage12.__globals__
    artifacts = globals_dict["_read_cvoi_direct_epdms_artifacts"](projection)
    context_type = globals_dict["_DirectEpdmsRuntimeContext"]
    agent = agent_class(device="cpu")
    agent._config = SimpleNamespace(data=SimpleNamespace(num_target_frames=5))
    agent._cvoi_direct_runtime = context_type(projection=projection, artifacts=artifacts)

    def forbidden_generic_loader(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("direct stage12 initialization called a generic artifact loader")

    for loader_name in (
        "load_cvoi_dual_value_model",
        "load_cvoi_gate_for_evaluation",
        "load_pretrained_checkpoint",
    ):
        monkeypatch.setitem(globals_dict, loader_name, forbidden_generic_loader)

    monkeypatch.setitem(
        globals_dict,
        "resolve_main_predictor_runtime_overrides",
        lambda *args, **kwargs: (None, None),
    )
    monkeypatch.setitem(
        globals_dict,
        "resolve_main_timeline",
        lambda *args, **kwargs: SimpleNamespace(num_future_steps=4),
    )

    def tiny_predictor_factory(*args: object, **kwargs: object):
        del args, kwargs
        return torch.nn.Linear(1, 1, bias=False), None, 1, False

    planner_instances: list[torch.nn.Module] = []

    def tiny_planner_factory(*args: object, **kwargs: object) -> torch.nn.Module:
        del args, kwargs
        planner = torch.nn.Linear(1, 1, bias=False)
        planner_instances.append(planner)
        return planner

    monkeypatch.setitem(globals_dict, "init_predictor_runtime_with_token_ae", tiny_predictor_factory)
    monkeypatch.setitem(globals_dict, "init_planner", tiny_planner_factory)
    monkeypatch.setitem(globals_dict, "maybe_register_parallel_predictor_tokens", lambda *args, **kwargs: None)
    monkeypatch.setattr(agent, "_build_multiview_fusion", lambda *args, **kwargs: None)
    encoder = torch.nn.Linear(1, 1, bias=False)

    agent._initialize_stage12(torch.device("cpu"), encoder, encoder_embed_dim=1)

    direct_slots = {
        "_cvoi_direct_p0_planner",
        "_cvoi_direct_p1_planner",
        "_cvoi_direct_field_value",
        "_cvoi_direct_stop_value",
        "_cvoi_direct_gate",
    }
    assert {name for name in direct_slots if getattr(agent, name) is not None} == expected_slots
    assert len(planner_instances) == expected_planner_factory_calls
    active_planner = agent._cvoi_direct_p0_planner if mode == "p0_forced" else agent._cvoi_direct_p1_planner
    assert agent._planner is active_planner
    assert torch.equal(encoder.weight.detach(), torch.tensor([[1.0]]))
    assert torch.equal(agent._predictor.weight.detach(), torch.tensor([[2.0]]))
    if agent._cvoi_direct_p0_planner is not None:
        assert torch.equal(agent._cvoi_direct_p0_planner.weight.detach(), torch.tensor([[3.0]]))
    if agent._cvoi_direct_p1_planner is not None:
        assert torch.equal(agent._cvoi_direct_p1_planner.weight.detach(), torch.tensor([[4.0]]))
    if agent._cvoi_direct_field_value is not None:
        assert isinstance(agent._cvoi_direct_field_value, PrefixDualValueModel)
    if agent._cvoi_direct_stop_value is not None:
        assert isinstance(agent._cvoi_direct_stop_value, PrefixDualValueModel)
    if agent._cvoi_direct_gate is not None:
        assert isinstance(agent._cvoi_direct_gate, SequentialRolloutGate)
    assert agent._cvoi_dual_value is None
    assert agent._cvoi_gate is None
    for module in [encoder, agent._predictor, *(getattr(agent, name) for name in expected_slots)]:
        assert not module.training
        assert all(not parameter.requires_grad for parameter in module.parameters())
    assert all(
        getattr(agent._cvoi_direct_runtime.artifacts, field) is None
        for field in (
            "p0_checkpoint",
            "calibration_checkpoint",
            "p1_checkpoint",
            "stop_checkpoint",
            "gate_checkpoint",
        )
    )


def _direct_module_agent(projection: CvoiDirectEpdmsProjection) -> object:
    agent = object.__new__(_agent_class())
    agent._config = SimpleNamespace(
        cvoi=SimpleNamespace(
            stage="evaluation",
            protocol_version="formal_v2_navsim_e120_h4_v3",
        )
    )
    agent._cvoi_direct_runtime = SimpleNamespace(
        projection=projection,
        evaluation_seed=_DIRECT_EVALUATION_SEED,
    )
    agent._cvoi_evaluation_gate_feature_mode = None
    agent._cvoi_direct_p0_planner = None
    agent._cvoi_direct_p1_planner = None
    agent._cvoi_direct_field_value = None
    agent._cvoi_direct_stop_value = None
    agent._cvoi_direct_gate = None
    return agent


def _install_expected_direct_modules(agent: object, mode: str) -> None:
    if mode in {"controller", "p0_forced"}:
        agent._cvoi_direct_p0_planner = _frozen(torch.nn.Linear(1, 1))
    if mode in {"controller", "p1_field_forced"}:
        agent._cvoi_direct_p1_planner = _frozen(torch.nn.Linear(1, 1))
    if mode in {"controller", "p1_field_forced"}:
        agent._cvoi_direct_field_value = _frozen(torch.nn.Linear(1, 1))
    if mode == "controller":
        agent._cvoi_direct_stop_value = _frozen(torch.nn.Linear(1, 1))
        agent._cvoi_direct_gate = _frozen(SequentialRolloutGate(latent_dim=1, hidden_dim=2))


@pytest.mark.parametrize(
    ("branch", "mode", "horizon", "present"),
    [
        (
            "full",
            "controller",
            None,
            {
                "_cvoi_direct_p0_planner",
                "_cvoi_direct_p1_planner",
                "_cvoi_direct_field_value",
                "_cvoi_direct_stop_value",
                "_cvoi_direct_gate",
            },
        ),
    ],
)
def test_direct_runtime_binds_exactly_the_mode_specific_frozen_modules(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    branch: str,
    mode: str,
    horizon: int | None,
    present: set[str],
) -> None:
    projection, _ = _artifact_projection(
        tmp_path,
        monkeypatch,
        branch=branch,
        evaluation_mode=mode,
        horizon=horizon,
    )
    agent = _direct_module_agent(projection)
    _install_expected_direct_modules(agent, mode)

    agent._bind_cvoi_direct_runtime()

    slots = {
        "_cvoi_direct_p0_planner",
        "_cvoi_direct_p1_planner",
        "_cvoi_direct_field_value",
        "_cvoi_direct_stop_value",
        "_cvoi_direct_gate",
    }
    assert {name for name in slots if getattr(agent, name) is not None} == present


@pytest.mark.parametrize("mutation", ["missing_p1", "missing_stop", "training", "requires_grad"])
def test_direct_runtime_rejects_incomplete_extra_or_unfrozen_modules_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
) -> None:
    projection, _ = _artifact_projection(
        tmp_path,
        monkeypatch,
        branch="full",
        evaluation_mode="controller",
        horizon=None,
    )
    agent = _direct_module_agent(projection)
    _install_expected_direct_modules(agent, "controller")
    if mutation == "missing_p1":
        agent._cvoi_direct_p1_planner = None
    elif mutation == "missing_stop":
        agent._cvoi_direct_stop_value = None
    elif mutation == "training":
        agent._cvoi_direct_p0_planner.train()
    else:
        next(agent._cvoi_direct_p0_planner.parameters()).requires_grad_(True)

    with pytest.raises((RuntimeError, ValueError), match="direct|module|planner|P1|Stop|frozen|eval"):
        agent._bind_cvoi_direct_runtime()


@pytest.mark.parametrize("horizon", range(5))
def test_direct_controller_selects_p0_only_at_h0_and_p1_at_h1_through_h4(horizon: int) -> None:
    agent = object.__new__(_agent_class())
    p0 = _frozen(torch.nn.Linear(1, 1))
    p1 = _frozen(torch.nn.Linear(1, 1))
    agent._cvoi_direct_runtime = SimpleNamespace(
        projection=SimpleNamespace(evaluation_mode="controller", horizon=None)
    )
    agent._cvoi_direct_p0_planner = p0
    agent._cvoi_direct_p1_planner = p1

    selected = agent._select_cvoi_direct_planner(horizon)

    assert selected is (p0 if horizon == 0 else p1)


def test_stage12_final_planner_forward_uses_the_direct_horizon_selector() -> None:
    source = inspect.getsource(_agent_class()._forward_stage12)

    assert "self._select_cvoi_direct_planner(" in source


class _DirectFeatureBuilder:
    def __init__(self, features: dict[str, torch.Tensor]) -> None:
        self.features = features

    def compute_features(self, agent_input: object) -> dict[str, torch.Tensor]:
        del agent_input
        return {name: value.clone() for name, value in self.features.items()}


class _DirectTrajectory:
    def __init__(self, poses: np.ndarray, trajectory_sampling: object) -> None:
        self.poses = poses
        self.trajectory_sampling = trajectory_sampling


def _write_direct_scenario_manifest(path: Path) -> None:
    row = {
        "schema": "cvoi_navsim_scenario_v1",
        "protocol_id": "epdms_v2_one_stage_navtest",
        "scenario_token": _DIRECT_SCENARIO_TOKEN,
        "observation_key": _DIRECT_OBSERVATION_KEY,
        "log_name": "navtest-log-a",
        "current_camera_data_path": "sensors/navtest-direct-a.jpg",
    }
    path.write_text(
        json.dumps(row, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="ascii",
    )


def _compute_projection(
    tmp_path: Path,
    *,
    evaluation_mode: str,
    horizon: int | None,
    branch: str,
) -> CvoiDirectEpdmsProjection:
    output_directory = (tmp_path / "direct-output" / ("controller" if horizon is None else f"h{horizon}")).resolve()
    handoffs = cvoi_manual_lineage.derive_cvoi_manual_full_handoffs((tmp_path / "cvoi_manual_full").resolve())
    (output_directory / "policy_traces").mkdir(parents=True)
    scenario_manifest_path = (tmp_path / "navtest-scenarios.jsonl").resolve()
    _write_direct_scenario_manifest(scenario_manifest_path)
    common = {
        "branch": branch,
        "split": "navtest",
        "protocol": "epdms_v2_one_stage_navtest",
        "evaluation_mode": evaluation_mode,
        "horizon": horizon,
        "guidance_steps": 0 if evaluation_mode == "p0_forced" else 2,
        "training_config_path": (tmp_path / "training.yaml").resolve(),
        "encoder_checkpoint_path": (tmp_path / "encoder.pt").resolve(),
        "scenario_manifest_path": scenario_manifest_path,
        "output_directory": output_directory,
    }
    if evaluation_mode == "p0_forced":
        return CvoiDirectEpdmsProjection(
            **common,
            p0_planner_checkpoint_path=handoffs["p0_handoff"],
        )
    if evaluation_mode == "p1_field_forced":
        return CvoiDirectEpdmsProjection(
            **common,
            calibration_checkpoint_path=handoffs["calibration_handoff"],
            p1_planner_checkpoint_path=handoffs["p1_handoff"],
        )
    return CvoiDirectEpdmsProjection(
        **common,
        p0_planner_checkpoint_path=handoffs["p0_handoff"],
        calibration_checkpoint_path=handoffs["calibration_handoff"],
        p1_planner_checkpoint_path=handoffs["p1_handoff"],
        stop_checkpoint_path=handoffs["stop_handoff"],
        gate_checkpoint_path=handoffs["gate_handoff"],
        gate_feature_mode="full",
        oracle_sha256="a" * 64,
    )


def _direct_compute_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    evaluation_mode: str = "controller",
    horizon: int | None = None,
    branch: str = "full",
    executed_horizon: int = 3,
    trajectory: torch.Tensor | None = None,
) -> tuple[object, CvoiDirectEpdmsProjection, int, list[dict[str, object]]]:
    projection = _compute_projection(
        tmp_path,
        evaluation_mode=evaluation_mode,
        horizon=horizon,
        branch=branch,
    )
    trace_output_dir = projection.output_directory / "policy_traces"
    trace_stat = trace_output_dir.stat(follow_symlinks=False)
    sample_seed = cvoi_sample_seed(_DIRECT_EVALUATION_SEED, _DIRECT_OBSERVATION_KEY)
    features = {
        "video_clip": torch.ones(3, 2, 4, 4),
        "states": torch.zeros(2, 7),
        "actions": torch.zeros(1, 7),
        "extrinsics": torch.zeros(2, 7),
        "cvoi_observation_key": observation_key_tensor(_DIRECT_OBSERVATION_KEY),
        "cvoi_rng_seed_bytes": unsigned_seed_tensor(sample_seed),
    }
    agent_class = _agent_class()
    monkeypatch.setitem(agent_class.compute_trajectory.__globals__, "Trajectory", _DirectTrajectory)
    agent = agent_class(device="cpu")
    agent._cvoi_manual_runtime = None
    agent._cvoi_direct_runtime = SimpleNamespace(
        projection=projection,
        artifacts=SimpleNamespace(),
        scenario_tokens_by_observation_key={_DIRECT_OBSERVATION_KEY: _DIRECT_SCENARIO_TOKEN},
        evaluation_seed=_DIRECT_EVALUATION_SEED,
        trace_output_dir=trace_output_dir,
        trace_output_dir_identity=(trace_stat.st_dev, trace_stat.st_ino),
    )
    agent._config = SimpleNamespace(
        cvoi=SimpleNamespace(
            max_horizon=4,
            ablation_signature=SimpleNamespace(evaluation_seed=_DIRECT_EVALUATION_SEED),
        ),
        meta=SimpleNamespace(seed=_DIRECT_EVALUATION_SEED),
    )
    agent._encoder = torch.nn.Linear(1, 1)
    agent._feature_builder = _DirectFeatureBuilder(features)
    agent._cvoi_trajectory_sampling = SimpleNamespace(num_poses=8)
    agent._cvoi_evaluation_guidance_steps = None
    agent._cvoi_evaluation_forced_horizon = None
    agent._cvoi_evaluation_gate_feature_mode = projection.gate_feature_mode
    agent._cvoi_latency_mode = False
    agent.eval = lambda: agent
    agent.get_feature_builders = lambda: [agent._feature_builder]
    prepared_calls: list[dict[str, object]] = []

    def prepare_cvoi_features(values: dict[str, object]) -> dict[str, object]:
        prepared_calls.append(dict(values))
        return values

    agent.prepare_cvoi_features = prepare_cvoi_features
    agent.set_cvoi_evaluation_guidance_steps = lambda value: setattr(
        agent,
        "_cvoi_evaluation_guidance_steps",
        value,
    )
    agent.set_cvoi_evaluation_forced_horizon = lambda value: setattr(
        agent,
        "_cvoi_evaluation_forced_horizon",
        value,
    )
    agent.set_cvoi_latency_mode = lambda value: setattr(agent, "_cvoi_latency_mode", value)
    output_trajectory = torch.zeros(1, 8, 3) if trajectory is None else trajectory

    def forward(model_features: dict[str, object]) -> dict[str, torch.Tensor]:
        assert model_features["cvoi_rng_seed"] == sample_seed
        assert "cvoi_observation_key" not in model_features
        assert "cvoi_rng_seed_bytes" not in model_features
        agent._last_cvoi_trace = {
            "stop_horizon": executed_horizon,
            "decisions": [],
            "predicted_deltas": [],
            "rollout_latency_ms": 0.0,
            "guidance": {
                "guidance_steps": 0.0 if executed_horizon == 0 else float(projection.guidance_steps),
            },
        }
        agent._last_cvoi_latency_components = {
            "encoder": 1.0,
            "adaptive_rollout_value_gate_guidance": 2.0,
            "planner_and_output": 3.0,
        }
        return {"trajectory": output_trajectory.clone()}

    agent.forward = forward
    return agent, projection, sample_seed, prepared_calls


def test_direct_runtime_context_carries_only_prevalidated_online_scenario_authority() -> None:
    context_type = _module_globals()["_DirectEpdmsRuntimeContext"]
    fields = set(getattr(context_type, "_fields", ()))
    if not fields:
        fields = set(getattr(context_type, "__annotations__", {}))

    assert {
        "projection",
        "artifacts",
        "scenario_tokens_by_observation_key",
        "evaluation_seed",
        "trace_output_dir",
        "trace_output_dir_identity",
    } <= fields
    assert "scenario_manifest" not in fields
    assert "oracle_path" not in fields


def test_direct_feature_builder_enables_sample_identity_from_the_parsed_evaluation_seed(tmp_path: Path) -> None:
    projection = _compute_projection(
        tmp_path,
        evaluation_mode="controller",
        horizon=None,
        branch="full",
    )
    agent = _agent_class()(device="cpu")
    agent._config = SimpleNamespace(
        cvoi=SimpleNamespace(
            ablation_signature=SimpleNamespace(evaluation_seed=_DIRECT_EVALUATION_SEED),
        )
    )
    agent._cvoi_direct_runtime = SimpleNamespace(
        projection=projection,
        evaluation_seed=_DIRECT_EVALUATION_SEED,
    )

    builder = agent._build_feature_builder()

    assert builder._official_cvoi_identity is True
    assert builder._cvoi_evaluation_seed == _DIRECT_EVALUATION_SEED


def test_generic_feature_builder_keeps_identity_disabled_without_any_cvoi_runtime() -> None:
    agent = _agent_class()(device="cpu")
    agent._config = SimpleNamespace(meta=SimpleNamespace(seed=_DIRECT_EVALUATION_SEED))

    builder = agent._build_feature_builder()

    assert builder._official_cvoi_identity is False
    assert builder._cvoi_evaluation_seed is None


@pytest.mark.parametrize(
    ("evaluation_mode", "horizon", "branch", "executed_horizon", "expected_forced_horizon"),
    [
        ("controller", None, "full", 3, None),
    ],
)
def test_direct_compute_decodes_identity_runs_once_and_writes_the_exact_exclusive_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluation_mode: str,
    horizon: int | None,
    branch: str,
    executed_horizon: int,
    expected_forced_horizon: int | None,
) -> None:
    agent, projection, sample_seed, prepared_calls = _direct_compute_agent(
        tmp_path,
        monkeypatch,
        evaluation_mode=evaluation_mode,
        horizon=horizon,
        branch=branch,
        executed_horizon=executed_horizon,
    )

    result = agent.compute_trajectory(object(), object(), object(), 0.5)

    assert isinstance(result, _DirectTrajectory)
    assert result.trajectory_sampling is agent._cvoi_trajectory_sampling
    assert result.poses.shape == (8, 3)
    assert np.isfinite(result.poses).all()
    assert len(prepared_calls) == 1
    assert prepared_calls[0]["cvoi_rng_seed"] == sample_seed
    assert agent._cvoi_evaluation_forced_horizon == expected_forced_horizon
    assert agent._cvoi_evaluation_guidance_steps == projection.guidance_steps
    assert agent._cvoi_latency_mode is True
    trace_path = projection.output_directory / "policy_traces" / f"{_DIRECT_SCENARIO_TOKEN}.json"
    assert list((projection.output_directory / "policy_traces").glob("*.json")) == [trace_path]
    actual_trace = json.loads(trace_path.read_text(encoding="ascii"))
    assert actual_trace["latency_ms"] == pytest.approx(6.0)
    expected_trace = {
        "schema": "cvoi_direct_epdms_trace",
        "version": 1,
        "split": "navtest",
        "protocol": "epdms_v2_one_stage_navtest",
        "branch": branch,
        "scenario_token": _DIRECT_SCENARIO_TOKEN,
        "evaluation_seed": _DIRECT_EVALUATION_SEED,
        "final_horizon": executed_horizon,
        "latency_ms": pytest.approx(6.0),
    }
    assert actual_trace == expected_trace

    with pytest.raises(FileExistsError, match="trace|exists"):
        agent.compute_trajectory(object())
    assert json.loads(trace_path.read_text(encoding="ascii")) == expected_trace


@pytest.mark.parametrize(
    "unsafe_token",
    ["../escape", "navtest/escape", "/absolute", "..", ""],
)
def test_direct_compute_rejects_path_unsafe_scenario_tokens_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_token: str,
) -> None:
    agent, projection, _sample_seed, _prepared_calls = _direct_compute_agent(tmp_path, monkeypatch)
    agent._cvoi_direct_runtime.scenario_tokens_by_observation_key = {
        _DIRECT_OBSERVATION_KEY: unsafe_token,
    }

    with pytest.raises(ValueError, match="scenario.*token|token.*path|filename|safe"):
        agent.compute_trajectory(object())

    trace_directory = projection.output_directory / "policy_traces"
    assert list(trace_directory.iterdir()) == []
    assert not (projection.output_directory / "escape.json").exists()


@pytest.mark.parametrize(
    ("trajectory", "message"),
    [
        (torch.zeros(1, 7, 3), "num_poses|pose count|8"),
        (torch.zeros(1, 8, 2), r"shape|\[1, 8, 3\]"),
        (torch.full((1, 8, 3), float("nan")), "finite|NaN|Inf"),
    ],
)
def test_direct_compute_rejects_invalid_trajectory_before_publishing_a_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trajectory: torch.Tensor,
    message: str,
) -> None:
    agent, projection, _sample_seed, _prepared_calls = _direct_compute_agent(
        tmp_path,
        monkeypatch,
        trajectory=trajectory,
    )

    with pytest.raises(ValueError, match=message):
        agent.compute_trajectory(object())

    assert list((projection.output_directory / "policy_traces").glob("*.json")) == []


@pytest.mark.parametrize("identity_drift", ["unknown_observation", "sample_seed"])
def test_direct_compute_rejects_identity_drift_before_prepare_or_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_drift: str,
) -> None:
    agent, projection, _sample_seed, prepared_calls = _direct_compute_agent(tmp_path, monkeypatch)
    if identity_drift == "unknown_observation":
        agent._cvoi_direct_runtime.scenario_tokens_by_observation_key = {}
    else:
        agent._feature_builder.features["cvoi_rng_seed_bytes"] = unsigned_seed_tensor(0)
    agent.forward = lambda features: (_ for _ in ()).throw(AssertionError("forward must not run"))

    with pytest.raises(ValueError, match="observation|manifest|seed"):
        agent.compute_trajectory(object())

    assert prepared_calls == []
    assert list((projection.output_directory / "policy_traces").glob("*.json")) == []


def test_non_direct_compute_trajectory_keeps_the_generic_tensor_only_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_class = _agent_class()
    monkeypatch.setitem(agent_class.compute_trajectory.__globals__, "Trajectory", _DirectTrajectory)
    agent = agent_class(device="cpu")
    agent._cvoi_manual_runtime = None
    agent._cvoi_direct_runtime = None
    agent._encoder = torch.nn.Linear(1, 1)
    agent._feature_builder = _DirectFeatureBuilder({"video_clip": torch.ones(3, 2, 4, 4)})
    agent._cvoi_trajectory_sampling = SimpleNamespace(num_poses=8)
    agent.eval = lambda: agent
    agent.get_feature_builders = lambda: [agent._feature_builder]
    forward_calls: list[dict[str, object]] = []

    def forward(features: dict[str, object]) -> dict[str, torch.Tensor]:
        forward_calls.append(features)
        return {"trajectory": torch.zeros(1, 8, 3)}

    agent.forward = forward

    result = agent.compute_trajectory(object())

    assert isinstance(result, _DirectTrajectory)
    assert result.poses.shape == (8, 3)
    assert len(forward_calls) == 1
    assert set(forward_calls[0]) == {"video_clip"}
    assert not (tmp_path / "direct-output").exists()
