"""CPU contracts for the CVoI Formal-v2 NavSim e120 runtime boundary."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from app.vjepa_cowa_world_model.training import cvoi_formal_v2_full_state_warmstart as warmstart
from app.vjepa_cowa_world_model.training import cvoi_formal_v2_navsim_e120_runtime as runtime
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage as manual_lineage
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_E120_LAMBDA_GRID


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Linear(3, 2)
        self.register_buffer("encoder_scale", torch.tensor(91.0))


class _Predictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 2, bias=False)
        self.register_buffer("predictor_scale", torch.tensor(92.0))


class _Planner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.z_head = torch.nn.Linear(4, 3, bias=False)
        self.observed_source_embedding = torch.nn.Embedding(2, 4)
        self.register_buffer("planner_scale", torch.tensor(93.0))


def _modules(*, fill: float = 99.0) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {
        "encoder": _Encoder(),
        "predictor": _Predictor(),
        "planner": _Planner(),
    }
    with torch.no_grad():
        for module in modules.values():
            for parameter in module.parameters():
                parameter.fill_(fill)
            for buffer in module.buffers():
                buffer.fill_(fill)
    return modules


def _source_payload() -> dict[str, object]:
    return {
        "encoder": {
            "stem.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "stem.bias": torch.tensor([1.0, 2.0]),
            "encoder_scale": torch.tensor(2.5),
        },
        "predictor": {
            "projection.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "predictor_scale": torch.tensor(3.5),
        },
        "planner": {
            "z_head.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "observed_source_embedding.weight": torch.tensor([[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]]),
            "planner_scale": torch.tensor(4.5),
        },
        "opt": {"state": "must never be restored"},
        "scheduler": {"last_epoch": 120},
        "epoch": 120,
    }


def _verified_e120_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    checkpoint = tmp_path / "e120.pt"
    params = tmp_path / "params-pretrain.yaml"
    torch.save(_source_payload(), checkpoint)
    params.write_text("meta:\n  seed: 239\n", encoding="utf-8")
    monkeypatch.setattr(warmstart, "FORMAL_V2_E120_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setattr(warmstart, "FORMAL_V2_E120_PARAMS_PRETRAIN_PATH", str(params))
    return checkpoint, params


def _training_states() -> dict[str, object]:
    return {
        "optimizer": {"state": {}, "param_groups": [{"lr": 2e-4}]},
        "scaler": {},
        "scheduler": {"last_epoch": 9},
        "wd_scheduler": {"last_epoch": 9},
    }


def _direct_lineage(*, stage: str = "p0", branch_id: str | None = None) -> dict[str, object]:
    return runtime.build_formal_v2_navsim_e120_direct_lineage(
        stage=stage,
        branch_id=branch_id or ("p0_uniform" if stage == "p0" else "p1_full"),
    )


def _direct_checkpoint_payload(
    modules: dict[str, torch.nn.Module],
    *,
    stage: str = "p0",
    branch_id: str | None = None,
    run_id: str | None = None,
    epoch: int | None = None,
) -> dict[str, object]:
    normalized_branch = branch_id or ("p0_uniform" if stage == "p0" else "p1_full")
    return runtime.build_formal_v2_navsim_e120_direct_checkpoint(
        modules=modules,
        run_id=run_id or f"{normalized_branch}_run_001",
        stage=stage,
        epoch=epoch or (35 if stage == "p0" else 5),
        training_stop_epoch=50 if stage == "p0" else 80,
        schedule_epochs=50 if stage == "p0" else 80,
        selection_checkpoint_epochs=(
            runtime.FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS
            if stage == "p0"
            else runtime.FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS
        ),
        cumulative_horizon_histogram={0: 10, 1: 20, 2: 30, 3: 40, 4: 50},
        lineage=_direct_lineage(stage=stage, branch_id=normalized_branch),
        **_training_states(),
    )


def _assert_no_proof_keys(value: object) -> None:
    forbidden = ("receipt", "audit", "provenance", "source_commit")
    if isinstance(value, dict):
        for key in value:
            lowered = str(key).lower()
            assert "sha256" not in lowered
            assert lowered != "sha" and not lowered.startswith("sha_") and not lowered.endswith("_sha")
            assert all(marker not in lowered for marker in forbidden)
        for nested in value.values():
            _assert_no_proof_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_proof_keys(nested)


def _direct_calibration_metadata(*, branch_id: str = "calibration_full") -> dict[str, object]:
    lineage = manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase="field_calibrated",
        branch_id=branch_id,
    )
    return {
        "schema": "cvoi_dual_value_navsim_e120_v1",
        "phase": "field_calibrated",
        "protocol_version": runtime.FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
        "branch_id": branch_id,
        "epoch": 1,
        "roles": {
            "value_model": {
                "keys": ["head.bias", "head.weight"],
                "shapes": {
                    "head.bias": [1],
                    "head.weight": [1, 4],
                },
            }
        },
        "parents": manual_lineage.build_cvoi_manual_value_parents(lineage, "field_calibrated"),
    }


def _write_direct_calibration_checkpoint(path: Path, metadata: object = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"metadata": _direct_calibration_metadata() if metadata is None else metadata},
        path,
    )
    return path


def _strict_fake_direct_calibration_reader(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("strict fake Calibration checkpoint is unreadable") from error
    if type(payload) is not dict or set(payload) != {"metadata"}:
        raise ValueError("strict fake Calibration checkpoint envelope mismatch")
    metadata = payload["metadata"]
    if type(metadata) is not dict:
        raise ValueError("strict fake Calibration metadata must be a mapping")
    return copy.deepcopy(metadata)


def test_signed_artifact_proof_runtime_surface_is_removed() -> None:
    retired_names = {
        "FORMAL_V2_NAVSIM_E120_RUNTIME_SCHEMA",
        "FORMAL_V2_NAVSIM_E120_LINEAGE_SCHEMA",
        "FORMAL_V2_NAVSIM_E120_CHECKPOINT_SCHEMA",
        "FORMAL_V2_NAVSIM_E120_RESUME_CONTRACT",
        "canonical_json_sha256",
        "canonical_json_document_sha256",
        "role_state_sha256",
        "build_formal_v2_navsim_e120_runtime_signature",
        "validate_formal_v2_navsim_e120_runtime_signature",
        "build_formal_v2_navsim_e120_lineage",
        "validate_formal_v2_navsim_e120_lineage",
        "build_formal_v2_navsim_e120_checkpoint",
        "validate_formal_v2_navsim_e120_checkpoint",
        "prepare_same_run_resume",
        "restore_same_run_resume",
        "initialize_fresh_p0_rank0",
        "initialize_fresh_p1_rank0",
        "read_formal_v2_navsim_e120_checkpoint",
        "write_formal_v2_navsim_e120_checkpoint",
    }

    assert retired_names.isdisjoint(vars(runtime))


def test_direct_lineage_is_exact_structural_and_rejects_wrong_branch_or_proof_fields() -> None:
    p0 = _direct_lineage()
    p1 = _direct_lineage(stage="p1")

    assert p0 == {
        "schema": runtime.FORMAL_V2_NAVSIM_E120_DIRECT_LINEAGE_SCHEMA,
        "protocol_version": runtime.FORMAL_V2_NAVSIM_E120_PROTOCOL_VERSION,
        "stage": "p0",
        "branch_id": "p0_uniform",
    }
    assert p1["branch_id"] == "p1_full"
    _assert_no_proof_keys(p0)
    _assert_no_proof_keys(p1)

    with pytest.raises(ValueError, match="branch_id"):
        runtime.build_formal_v2_navsim_e120_direct_lineage(stage="p0", branch_id="p0_extremes")
    with pytest.raises(ValueError, match="branch_id"):
        runtime.build_formal_v2_navsim_e120_direct_lineage(stage="p1", branch_id="p0_uniform")
    with pytest.raises(ValueError, match="unknown=.*sha"):
        runtime.validate_formal_v2_navsim_e120_direct_lineage(dict(p0, checkpoint_sha256="0" * 64))


@pytest.mark.parametrize(
    "branch_id",
    ["p1_full", "p1_no_cf", "p1_hazard_only", "p1_quality_only"],
)
def test_direct_p1_lineage_accepts_each_retained_manual_value_branch(branch_id: str) -> None:
    lineage = runtime.build_formal_v2_navsim_e120_direct_lineage(
        stage="p1",
        branch_id=branch_id,
    )

    assert lineage["branch_id"] == branch_id
    assert (
        runtime.validate_formal_v2_navsim_e120_direct_lineage(
            lineage,
            expected_stage="p1",
        )
        == lineage
    )


@pytest.mark.parametrize(
    ("stage", "branch_id"),
    [
        ("p0", "p0_extremes"),
        ("p1", "p1_without_field"),
        ("p1", "p1_factual_only"),
        ("p0", "p1_full"),
        ("p0", "p1_no_cf"),
        ("p0", "p1_hazard_only"),
        ("p0", "p1_quality_only"),
    ],
)
def test_direct_lineage_rejects_unretained_or_cross_stage_branch(stage: str, branch_id: str) -> None:
    with pytest.raises(ValueError, match="branch_id"):
        runtime.build_formal_v2_navsim_e120_direct_lineage(
            stage=stage,
            branch_id=branch_id,
        )


def test_direct_checkpoint_is_exact_structural_and_rejects_state_schedule_and_proof_drift() -> None:
    modules = _modules()
    payload = _direct_checkpoint_payload(modules)
    normalized = runtime.validate_formal_v2_navsim_e120_direct_checkpoint(payload)

    assert set(normalized) == runtime.FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_FIELDS
    assert normalized["schema"] == runtime.FORMAL_V2_NAVSIM_E120_DIRECT_CHECKPOINT_SCHEMA
    assert normalized["resume_contract"] == runtime.FORMAL_V2_NAVSIM_E120_DIRECT_RESUME_CONTRACT
    assert normalized["role_state_shapes"] == {
        role: {key: list(tensor.shape) for key, tensor in sorted(module.state_dict().items())}
        for role, module in modules.items()
    }
    _assert_no_proof_keys(normalized)
    payload["planner"]["z_head.weight"].add_(1000.0)
    assert not torch.equal(payload["planner"]["z_head.weight"], normalized["planner"]["z_head.weight"])

    drifts: list[tuple[str, dict[str, object], str]] = []
    forbidden = copy.deepcopy(normalized)
    forbidden["optimizer"]["audit_receipt"] = "forbidden"
    drifts.append(("forbidden", forbidden, "forbidden proof key.*audit_receipt"))
    missing_role = copy.deepcopy(normalized)
    del missing_role["role_state_shapes"]["planner"]
    drifts.append(("role", missing_role, "roles"))
    missing_key = copy.deepcopy(normalized)
    del missing_key["role_state_shapes"]["planner"]["z_head.weight"]
    drifts.append(("key", missing_key, "state keys"))
    wrong_shape = copy.deepcopy(normalized)
    wrong_shape["role_state_shapes"]["planner"]["z_head.weight"] = [99]
    drifts.append(("shape", wrong_shape, "state shapes"))
    wrong_schedule = copy.deepcopy(normalized)
    wrong_schedule["selection_checkpoint_epochs"] = ()
    drifts.append(("schedule", wrong_schedule, "selection_checkpoint_epochs"))
    empty_histogram = copy.deepcopy(normalized)
    empty_histogram["cumulative_horizon_histogram"] = {horizon: 0 for horizon in range(5)}
    drifts.append(("histogram", empty_histogram, "at least one"))
    missing_optimizer = copy.deepcopy(normalized)
    missing_optimizer["optimizer"] = {}
    drifts.append(("training state", missing_optimizer, "optimizer"))
    invalid_run = copy.deepcopy(normalized)
    invalid_run["run_id"] = "INVALID-RUN"
    drifts.append(("run", invalid_run, "run_id"))
    for _, drifted, message in drifts:
        with pytest.raises(ValueError, match=message):
            runtime.validate_formal_v2_navsim_e120_direct_checkpoint(drifted)


def test_direct_checkpoint_reader_uses_plain_path_load_without_proof_helpers(tmp_path: Path) -> None:
    payload = _direct_checkpoint_payload(_modules())
    output = tmp_path / "direct.pt"
    torch.save(payload, output)

    assert not hasattr(runtime, "_sha256_handle")
    assert not hasattr(runtime, "_open_regular_nofollow")
    loaded = runtime.read_formal_v2_navsim_e120_direct_checkpoint(output)

    assert loaded["lineage"] == payload["lineage"]
    assert loaded["role_state_shapes"] == payload["role_state_shapes"]
    with pytest.raises(ValueError, match="absolute"):
        runtime.read_formal_v2_navsim_e120_direct_checkpoint(Path("relative.pt"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        runtime.read_formal_v2_navsim_e120_direct_checkpoint(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="regular file"):
        runtime.read_formal_v2_navsim_e120_direct_checkpoint(tmp_path)


def test_direct_epdms_planner_validator_returns_only_zero_copy_deployment_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _direct_checkpoint_payload(
        _modules(),
        stage="p1",
        branch_id="p1_full",
    )

    def forbid_deepcopy(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("direct EPDMS deployment validation must not call copy.deepcopy")

    monkeypatch.setattr(runtime.copy, "deepcopy", forbid_deepcopy)
    deployment = runtime.validate_cvoi_direct_epdms_planner_checkpoint(
        payload,
        expected_stage="p1",
        expected_branch_id="p1_full",
    )

    assert set(deployment) == {
        "stage",
        "lineage",
        "protocol_version",
        "role_state_shapes",
        *runtime.MODEL_ROLES,
    }
    for role in runtime.MODEL_ROLES:
        assert set(deployment[role]) == set(payload[role])
        for key, source_tensor in payload[role].items():
            deployed_tensor = deployment[role][key]
            assert deployed_tensor.data_ptr() == source_tensor.data_ptr()
            assert deployed_tensor.untyped_storage().data_ptr() == source_tensor.untyped_storage().data_ptr()


def test_direct_epdms_planner_reader_preserves_loaded_tensor_storage_without_deepcopy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _direct_checkpoint_payload(_modules())
    checkpoint = (tmp_path / "direct-epdms-p0.pt").resolve()
    checkpoint.write_bytes(b"reader preflight sentinel")

    def load_complete_checkpoint(
        path: Path,
        *,
        map_location: str,
        weights_only: bool,
    ) -> dict[str, object]:
        assert path == checkpoint
        assert map_location == "cpu"
        assert weights_only is True
        return payload

    def forbid_deepcopy(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("direct EPDMS Planner reader must not call copy.deepcopy")

    monkeypatch.setattr(runtime.torch, "load", load_complete_checkpoint)
    monkeypatch.setattr(runtime.copy, "deepcopy", forbid_deepcopy)
    deployment = runtime.read_cvoi_direct_epdms_planner_checkpoint(
        checkpoint,
        expected_stage="p0",
        expected_branch_id="p0_uniform",
    )

    assert set(deployment).isdisjoint({"optimizer", "scaler", "scheduler", "wd_scheduler"})
    for role in runtime.MODEL_ROLES:
        for key, source_tensor in payload[role].items():
            deployed_tensor = deployment[role][key]
            assert deployed_tensor.data_ptr() == source_tensor.data_ptr()
            assert deployed_tensor.untyped_storage().data_ptr() == source_tensor.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    ("mutation", "expected_stage", "expected_branch_id", "message"),
    [
        ("training_state", "p0", "p0_uniform", "optimizer"),
        ("stage", "p1", "p1_full", "stage mismatch"),
        ("lineage", "p0", "p1_full", "lineage mismatch"),
        ("role_shape", "p0", "p0_uniform", "state shapes"),
    ],
)
def test_direct_epdms_planner_validator_rejects_invalid_full_envelope_without_copying(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_stage: str,
    expected_branch_id: str,
    message: str,
) -> None:
    payload = _direct_checkpoint_payload(_modules())
    if mutation == "training_state":
        payload["optimizer"] = []
    elif mutation == "role_shape":
        payload["role_state_shapes"]["planner"]["z_head.weight"] = [99]

    def forbid_deepcopy(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("invalid direct EPDMS deployment envelopes must not be deep-copied")

    monkeypatch.setattr(runtime.copy, "deepcopy", forbid_deepcopy)
    with pytest.raises(ValueError, match=message):
        runtime.validate_cvoi_direct_epdms_planner_checkpoint(
            payload,
            expected_stage=expected_stage,
            expected_branch_id=expected_branch_id,
        )


def test_direct_epdms_gate_validator_rejects_extra_provenance_field() -> None:
    branch = "full"
    oracle_sha256 = "a" * 64
    identity = runtime.resolve_cvoi_direct_epdms_artifact_identity(branch, evaluation_mode="controller")
    gate = runtime.SequentialRolloutGate(latent_dim=2, hidden_dim=4)
    payload = {
        "schema": runtime.SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_NAVSIM_E120,
        "feature_schema": runtime.CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
        "latent_dim": gate.latent_dim,
        "hidden_dim": gate.hidden_dim,
        "feature_dim": gate.feature_dim,
        "lambda_grid": list(FORMAL_V2_NAVSIM_E120_LAMBDA_GRID),
        "provenance": {
            "gate_pipeline": runtime.CVOI_NAVSIM_E120_OFFICIAL_GATE_PIPELINE_VERSION,
            "oracle_protocol": runtime.NAVTRAIN_GATE_PROTOCOL_ID,
            "oracle_sha256": oracle_sha256,
            "oracle_lineage": identity.oracle_lineage,
            "gate_feature_schema": runtime.CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
            "gate_feature_mode": identity.gate_feature_mode,
        },
        "state_dict": gate.state_dict(),
    }

    normalized = runtime.validate_cvoi_direct_epdms_gate_checkpoint(
        payload,
        branch=branch,
        oracle_sha256=oracle_sha256,
        gate_feature_mode="full",
    )
    assert normalized["provenance"] == payload["provenance"]

    payload["provenance"]["source_checkpoint"] = "/unexpected/gate.pt"
    with pytest.raises(ValueError, match="provenance fields mismatch.*unexpected"):
        runtime.validate_cvoi_direct_epdms_gate_checkpoint(
            payload,
            branch=branch,
            oracle_sha256=oracle_sha256,
            gate_feature_mode="full",
        )


def test_direct_checkpoint_writer_supports_immutable_and_atomic_replace(tmp_path: Path) -> None:
    modules = _modules()
    payload = _direct_checkpoint_payload(modules)
    output = tmp_path / "run" / "latest.pt"

    assert runtime.write_formal_v2_navsim_e120_direct_checkpoint(output, payload, replace=False) == output
    with pytest.raises(FileExistsError, match="immutable"):
        runtime.write_formal_v2_navsim_e120_direct_checkpoint(output, payload, replace=False)

    with torch.no_grad():
        modules["planner"].z_head.weight.add_(5.0)
    replacement = _direct_checkpoint_payload(modules, epoch=40)
    runtime.write_formal_v2_navsim_e120_direct_checkpoint(output, replacement, replace=True)
    assert runtime.read_formal_v2_navsim_e120_direct_checkpoint(output)["epoch"] == 40

    target = tmp_path / "target.pt"
    target.write_bytes(b"target")
    symlink = tmp_path / "symlink.pt"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        runtime.write_formal_v2_navsim_e120_direct_checkpoint(symlink, replacement, replace=True)


def test_direct_same_run_resume_restores_all_states_and_rejects_reinitialization() -> None:
    source_modules = _modules()
    payload = _direct_checkpoint_payload(source_modules)
    target_modules = _modules(fill=-1.0)
    optimizer = _StateRecorder()
    scaler = _StateRecorder()
    scheduler = _StateRecorder()
    wd_scheduler = _StateRecorder()

    result = runtime.restore_formal_v2_navsim_e120_direct_same_run_resume(
        payload,
        modules=target_modules,
        optimizer=optimizer,
        scaler=scaler,
        scheduler=scheduler,
        wd_scheduler=wd_scheduler,
        expected_run_id="p0_uniform_run_001",
        expected_stage="p0",
        expected_lineage=_direct_lineage(),
        expected_training_stop_epoch=50,
        warmstart_requested=False,
        model_only=False,
    )

    assert result == {
        "start_epoch": 35,
        "run_id": "p0_uniform_run_001",
        "stage": "p0",
        "training_stop_epoch": 50,
        "schedule_epochs": 50,
        "selection_checkpoint_epochs": runtime.FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
        "cumulative_horizon_histogram": {0: 10, 1: 20, 2: 30, 3: 40, 4: 50},
        "role_state_shapes": payload["role_state_shapes"],
        "lineage": payload["lineage"],
    }
    for role in runtime.MODEL_ROLES:
        assert all(
            torch.equal(tensor, source_modules[role].state_dict()[key])
            for key, tensor in target_modules[role].state_dict().items()
        )
    assert optimizer.loaded == payload["optimizer"]
    assert scaler.loaded == payload["scaler"]
    assert scheduler.loaded == payload["scheduler"]
    assert wd_scheduler.loaded == payload["wd_scheduler"]

    kwargs = {
        "expected_run_id": "p0_uniform_run_001",
        "expected_stage": "p0",
        "expected_lineage": _direct_lineage(),
        "expected_training_stop_epoch": 50,
        "warmstart_requested": False,
        "model_only": False,
    }
    with pytest.raises(ValueError, match="re-warmstart"):
        runtime.prepare_formal_v2_navsim_e120_direct_same_run_resume(
            payload,
            **dict(kwargs, warmstart_requested=True),
        )
    with pytest.raises(ValueError, match="model-only"):
        runtime.prepare_formal_v2_navsim_e120_direct_same_run_resume(payload, **dict(kwargs, model_only=True))
    with pytest.raises(ValueError, match="cross-run"):
        runtime.prepare_formal_v2_navsim_e120_direct_same_run_resume(
            payload,
            **dict(kwargs, expected_run_id="different_run"),
        )
    another_run = _direct_checkpoint_payload(_modules(), run_id="p0_uniform_run_002")
    assert runtime.validate_formal_v2_navsim_e120_direct_checkpoint(another_run)["lineage"] == payload["lineage"]
    with pytest.raises(ValueError, match="cross-run"):
        runtime.prepare_formal_v2_navsim_e120_direct_same_run_resume(
            another_run,
            **kwargs,
        )
    with pytest.raises(ValueError, match="cross-stage"):
        runtime.prepare_formal_v2_navsim_e120_direct_same_run_resume(
            _direct_checkpoint_payload(_modules(), stage="p1"),
            **dict(
                kwargs,
                expected_run_id="p1_full_run_001",
                expected_stage="p0",
                expected_lineage=_direct_lineage(),
                expected_training_stop_epoch=80,
            ),
        )
    with pytest.raises(ValueError, match="training_stop_epoch mismatch"):
        runtime.prepare_formal_v2_navsim_e120_direct_same_run_resume(
            payload,
            **dict(kwargs, expected_training_stop_epoch=80),
        )
    with pytest.raises(ValueError, match="lineage"):
        runtime.prepare_formal_v2_navsim_e120_direct_same_run_resume(
            payload,
            **dict(kwargs, expected_lineage=_direct_lineage(stage="p1")),
        )


def test_fresh_p0_direct_uses_path_warmstart_and_returns_only_structural_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, params = _verified_e120_sources(tmp_path, monkeypatch)
    modules = _modules(fill=-99.0)

    assert not hasattr(warmstart, "apply_formal_v2_full_state_warmstart")
    result = runtime.initialize_fresh_p0_direct_rank0(
        modules=modules,
        checkpoint_path=checkpoint,
        params_pretrain_path=params,
        lineage=_direct_lineage(),
        training_stop_epoch=50,
    )

    assert result == {
        "stage": "p0",
        "training_stop_epoch": 50,
        "role_state_shapes": {
            role: {key: list(tensor.shape) for key, tensor in sorted(module.state_dict().items())}
            for role, module in modules.items()
        },
        "lineage": _direct_lineage(),
    }
    _assert_no_proof_keys(result)
    assert torch.equal(modules["encoder"].stem.weight, _source_payload()["encoder"]["stem.weight"])
    with pytest.raises(ValueError, match="stage mismatch"):
        runtime.initialize_fresh_p0_direct_rank0(
            modules=_modules(),
            checkpoint_path=checkpoint,
            params_pretrain_path=params,
            lineage=_direct_lineage(stage="p1"),
            training_stop_epoch=50,
        )
    with pytest.raises(ValueError, match="exactly 50"):
        runtime.initialize_fresh_p0_direct_rank0(
            modules=_modules(),
            checkpoint_path=checkpoint,
            params_pretrain_path=params,
            lineage=_direct_lineage(),
            training_stop_epoch=45,
        )


def test_fresh_p1_direct_overlays_selected_p0_predictor_planner_and_rejects_structural_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, params = _verified_e120_sources(tmp_path, monkeypatch)
    parent_modules = _modules()
    warmstart.apply_formal_v2_full_state_warmstart_direct(checkpoint, params, parent_modules)
    with torch.no_grad():
        parent_modules["predictor"].projection.weight.add_(100.0)
        parent_modules["planner"].z_head.weight.add_(200.0)
    parent_payload = _direct_checkpoint_payload(parent_modules)
    parent_path = tmp_path / "p0-selected.pt"
    torch.save(parent_payload, parent_path)
    calibration_path = _write_direct_calibration_checkpoint(tmp_path / "calibration.pt")
    child_modules = _modules(fill=-99.0)

    result = runtime.initialize_fresh_p1_direct_rank0(
        modules=child_modules,
        checkpoint_path=checkpoint,
        params_pretrain_path=params,
        lineage=_direct_lineage(stage="p1"),
        parent_checkpoint_path=parent_path,
        calibration_checkpoint_path=calibration_path,
        calibration_checkpoint_validator=_strict_fake_direct_calibration_reader,
    )

    assert result["stage"] == "p1"
    assert result["training_stop_epoch"] == 80
    assert result["parent_epoch"] == 35
    _assert_no_proof_keys(result)
    assert torch.equal(child_modules["encoder"].stem.weight, _source_payload()["encoder"]["stem.weight"])
    assert torch.equal(child_modules["predictor"].projection.weight, parent_modules["predictor"].projection.weight)
    assert torch.equal(child_modules["planner"].z_head.weight, parent_modules["planner"].z_head.weight)

    with pytest.raises(FileNotFoundError, match="Calibration"):
        runtime.initialize_fresh_p1_direct_rank0(
            modules=_modules(),
            checkpoint_path=checkpoint,
            params_pretrain_path=params,
            lineage=_direct_lineage(stage="p1"),
            parent_checkpoint_path=parent_path,
            calibration_checkpoint_path=tmp_path / "missing-calibration.pt",
            calibration_checkpoint_validator=_strict_fake_direct_calibration_reader,
        )

    drift_cases: list[tuple[str, dict[str, object], str]] = []
    wrong_stage = _direct_checkpoint_payload(_modules(), stage="p1")
    drift_cases.append(("stage", wrong_stage, "selected P0"))
    wrong_branch = copy.deepcopy(parent_payload)
    wrong_branch["lineage"]["branch_id"] = "p1_full"
    drift_cases.append(("branch", wrong_branch, "branch_id"))
    wrong_epoch = copy.deepcopy(parent_payload)
    wrong_epoch["epoch"] = 11
    drift_cases.append(("epoch", wrong_epoch, "candidate"))
    missing_role = copy.deepcopy(parent_payload)
    del missing_role["role_state_shapes"]["planner"]
    drift_cases.append(("role", missing_role, "roles"))
    missing_key = copy.deepcopy(parent_payload)
    del missing_key["predictor"]["projection.weight"]
    drift_cases.append(("key", missing_key, "state keys"))
    wrong_shape = copy.deepcopy(parent_payload)
    wrong_shape["planner"]["z_head.weight"] = torch.zeros(1)
    drift_cases.append(("shape", wrong_shape, "state shapes"))
    encoder_drift = copy.deepcopy(parent_payload)
    encoder_drift["encoder"]["stem.weight"].add_(1.0)
    drift_cases.append(("encoder", encoder_drift, "encoder.*warmstart"))
    for name, drifted, message in drift_cases:
        drift_path = tmp_path / f"{name}.pt"
        torch.save(drifted, drift_path)
        with pytest.raises(ValueError, match=message):
            runtime.initialize_fresh_p1_direct_rank0(
                modules=_modules(),
                checkpoint_path=checkpoint,
                params_pretrain_path=params,
                lineage=_direct_lineage(stage="p1"),
                parent_checkpoint_path=drift_path,
                calibration_checkpoint_path=calibration_path,
                calibration_checkpoint_validator=_strict_fake_direct_calibration_reader,
            )


@pytest.mark.parametrize(
    ("p1_branch", "calibration_branch"),
    [
        ("p1_full", "calibration_full"),
        ("p1_no_cf", "calibration_no_cf"),
        ("p1_hazard_only", "calibration_hazard_only"),
        ("p1_quality_only", "calibration_quality_only"),
    ],
)
def test_fresh_p1_direct_accepts_matching_manual_calibration_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    p1_branch: str,
    calibration_branch: str,
) -> None:
    checkpoint, params = _verified_e120_sources(tmp_path, monkeypatch)
    parent_modules = _modules()
    warmstart.apply_formal_v2_full_state_warmstart_direct(checkpoint, params, parent_modules)
    parent_path = tmp_path / "p0-selected.pt"
    torch.save(_direct_checkpoint_payload(parent_modules), parent_path)
    calibration_path = _write_direct_calibration_checkpoint(
        tmp_path / f"{calibration_branch}.pt",
        _direct_calibration_metadata(branch_id=calibration_branch),
    )

    result = runtime.initialize_fresh_p1_direct_rank0(
        modules=_modules(fill=-99.0),
        checkpoint_path=checkpoint,
        params_pretrain_path=params,
        lineage={
            **_direct_lineage(stage="p1"),
            "branch_id": p1_branch,
        },
        parent_checkpoint_path=parent_path,
        calibration_checkpoint_path=calibration_path,
        calibration_checkpoint_validator=_strict_fake_direct_calibration_reader,
    )

    assert result["stage"] == "p1"
    assert result["lineage"]["branch_id"] == p1_branch
    assert result["parent_epoch"] == 35


@pytest.mark.parametrize(
    ("calibration_branch", "field_branch", "message"),
    [
        ("calibration_full", "field_full", "Calibration.*branch_id"),
        ("calibration_no_cf", "field_full", "Calibration.*[Pp]arent"),
    ],
)
def test_fresh_p1_no_cf_rejects_mismatched_calibration_before_warmstart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_branch: str,
    field_branch: str,
    message: str,
) -> None:
    checkpoint, params = _verified_e120_sources(tmp_path, monkeypatch)
    metadata = _direct_calibration_metadata(branch_id=calibration_branch)
    metadata["parents"]["field"]["branch_id"] = field_branch
    calibration_path = _write_direct_calibration_checkpoint(
        tmp_path / f"{calibration_branch}-{field_branch}.pt",
        metadata,
    )

    def poison_warmstart(*_: object, **__: object) -> None:
        raise AssertionError("warmstart must not run before Calibration lineage validation succeeds")

    monkeypatch.setattr(warmstart, "apply_formal_v2_full_state_warmstart_direct", poison_warmstart)

    with pytest.raises(ValueError, match=message):
        runtime.initialize_fresh_p1_direct_rank0(
            modules=_modules(fill=-99.0),
            checkpoint_path=checkpoint,
            params_pretrain_path=params,
            lineage={
                **_direct_lineage(stage="p1"),
                "branch_id": "p1_no_cf",
            },
            parent_checkpoint_path=tmp_path / "unused-parent.pt",
            calibration_checkpoint_path=calibration_path,
            calibration_checkpoint_validator=_strict_fake_direct_calibration_reader,
        )


def test_fresh_p1_direct_calibration_validation_is_exact_and_precedes_warmstart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, params = _verified_e120_sources(tmp_path, monkeypatch)
    calibration_path = _write_direct_calibration_checkpoint(tmp_path / "calibration.pt")
    corrupt_path = tmp_path / "corrupt-calibration.pt"
    corrupt_path.write_bytes(b"not a torch checkpoint")

    def poison_warmstart(*_: object, **__: object) -> None:
        raise AssertionError("warmstart must not run before Calibration validation succeeds")

    monkeypatch.setattr(warmstart, "apply_formal_v2_full_state_warmstart_direct", poison_warmstart)

    invalid_metadata: list[tuple[str, object]] = []
    missing = _direct_calibration_metadata()
    del missing["parents"]
    invalid_metadata.append(("missing field", missing))
    extra = _direct_calibration_metadata()
    extra["unexpected"] = None
    invalid_metadata.append(("extra field", extra))
    raw_field = _direct_calibration_metadata()
    raw_field["phase"] = "field_warmup"
    invalid_metadata.append(("raw Field phase", raw_field))
    wrong_protocol = _direct_calibration_metadata()
    wrong_protocol["protocol_version"] = "formal_v2"
    invalid_metadata.append(("protocol", wrong_protocol))
    wrong_branch = _direct_calibration_metadata()
    wrong_branch["branch_id"] = "field_full"
    invalid_metadata.append(("branch", wrong_branch))
    for epoch in (0, -1, True, 1.0):
        wrong_epoch = _direct_calibration_metadata()
        wrong_epoch["epoch"] = epoch
        invalid_metadata.append(("epoch", wrong_epoch))
    extra_role = _direct_calibration_metadata()
    extra_role["roles"]["other"] = {"keys": ["x"], "shapes": {"x": []}}
    invalid_metadata.append(("role", extra_role))
    missing_role_field = _direct_calibration_metadata()
    del missing_role_field["roles"]["value_model"]["shapes"]
    invalid_metadata.append(("role field", missing_role_field))
    extra_role_field = _direct_calibration_metadata()
    extra_role_field["roles"]["value_model"]["digest"] = "forbidden"
    invalid_metadata.append(("role field", extra_role_field))
    for keys in ([], ["head.weight", "head.bias"], ["head.bias", "head.bias"], ["head.bias", 1]):
        wrong_keys = _direct_calibration_metadata()
        wrong_keys["roles"]["value_model"]["keys"] = keys
        invalid_metadata.append(("role keys", wrong_keys))
    missing_shape = _direct_calibration_metadata()
    del missing_shape["roles"]["value_model"]["shapes"]["head.bias"]
    invalid_metadata.append(("shape keys", missing_shape))
    extra_shape = _direct_calibration_metadata()
    extra_shape["roles"]["value_model"]["shapes"]["extra"] = []
    invalid_metadata.append(("shape keys", extra_shape))
    invalid_shape = _direct_calibration_metadata()
    invalid_shape["roles"]["value_model"]["shapes"]["head.bias"] = [-1]
    invalid_metadata.append(("shape", invalid_shape))
    missing_parent = _direct_calibration_metadata()
    del missing_parent["parents"]["field"]
    invalid_metadata.append(("parent", missing_parent))
    extra_parent = _direct_calibration_metadata()
    extra_parent["parents"]["other"] = {}
    invalid_metadata.append(("parent", extra_parent))
    wrong_planner_parent = _direct_calibration_metadata()
    wrong_planner_parent["parents"]["unguided_planner"]["branch_id"] = "p1_full"
    invalid_metadata.append(("planner parent", wrong_planner_parent))
    wrong_field_parent = _direct_calibration_metadata()
    wrong_field_parent["parents"]["field"]["phase"] = "field_calibrated"
    invalid_metadata.append(("Field parent", wrong_field_parent))

    validators: list[tuple[str, object]] = [
        ("non-callable", None),
        ("None result", lambda _: None),
        ("non-mapping result", lambda _: []),
        *[(name, lambda _, metadata=metadata: copy.deepcopy(metadata)) for name, metadata in invalid_metadata],
        (
            "validator failure",
            lambda _: (_ for _ in ()).throw(ValueError("strict validator rejected Calibration")),
        ),
        ("corrupt payload", _strict_fake_direct_calibration_reader),
    ]
    for name, validator in validators:
        modules = _modules(fill=-99.0)
        before = {role: copy.deepcopy(module.state_dict()) for role, module in modules.items()}
        selected_calibration_path = corrupt_path if name == "corrupt payload" else calibration_path
        with pytest.raises(ValueError):
            runtime.initialize_fresh_p1_direct_rank0(
                modules=modules,
                checkpoint_path=checkpoint,
                params_pretrain_path=params,
                lineage=_direct_lineage(stage="p1"),
                parent_checkpoint_path=tmp_path / "unused-parent.pt",
                calibration_checkpoint_path=selected_calibration_path,
                calibration_checkpoint_validator=validator,
            )
        for role, module in modules.items():
            assert all(torch.equal(tensor, before[role][key]) for key, tensor in module.state_dict().items())

    callback_calls: list[Path] = []
    for invalid_path in (Path("relative-calibration.pt"), tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            runtime.initialize_fresh_p1_direct_rank0(
                modules=_modules(),
                checkpoint_path=checkpoint,
                params_pretrain_path=params,
                lineage=_direct_lineage(stage="p1"),
                parent_checkpoint_path=tmp_path / "unused-parent.pt",
                calibration_checkpoint_path=invalid_path,
                calibration_checkpoint_validator=lambda path: callback_calls.append(path),
            )
    assert callback_calls == []


@pytest.mark.parametrize("stage", ["p0", "p1"])
def test_selected_checkpoint_resolver_accepts_exact_regular_handoff_copy(tmp_path: Path, stage: str) -> None:
    results_root = tmp_path / "results"
    handoff = results_root / "handoff"
    handoff.mkdir(parents=True)
    selected = handoff / f"{stage}_selected.pt"
    selected.write_bytes(b"checkpoint")

    resolved = runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
        selected,
        results_root=results_root,
        stage=stage,
    )

    assert resolved == selected


@pytest.mark.parametrize("stage", ["p0", "p1"])
def test_selected_checkpoint_resolver_accepts_symlink_only_into_its_stage_tree(tmp_path: Path, stage: str) -> None:
    results_root = tmp_path / "results"
    handoff = results_root / "handoff"
    stage_directory = results_root / stage / "checkpoints"
    handoff.mkdir(parents=True)
    stage_directory.mkdir(parents=True)
    target = stage_directory / "epoch-35.pt"
    target.write_bytes(b"checkpoint")
    selected = handoff / f"{stage}_selected.pt"
    selected.symlink_to(target)

    resolved = runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
        selected,
        results_root=results_root,
        stage=stage,
    )

    assert resolved == target.resolve()


@pytest.mark.parametrize("stage", ["p0", "p1"])
def test_selected_checkpoint_resolver_rejects_missing_broken_directory_and_escape(
    tmp_path: Path,
    stage: str,
) -> None:
    results_root = tmp_path / "results"
    handoff = results_root / "handoff"
    handoff.mkdir(parents=True)
    selected = handoff / f"{stage}_selected.pt"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            selected,
            results_root=results_root,
            stage=stage,
        )

    selected.symlink_to(results_root / stage / "missing.pt")
    with pytest.raises(FileNotFoundError, match="broken"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            selected,
            results_root=results_root,
            stage=stage,
        )

    selected.unlink()
    selected.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            selected,
            results_root=results_root,
            stage=stage,
        )

    selected.rmdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"checkpoint")
    selected.symlink_to(outside)
    with pytest.raises(ValueError, match="outside"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            selected,
            results_root=results_root,
            stage=stage,
        )


@pytest.mark.parametrize("stage", ["", "P0", "p2", None, 0])
def test_selected_checkpoint_resolver_rejects_invalid_stage(tmp_path: Path, stage: object) -> None:
    results_root = tmp_path / "results"
    selected = results_root / "handoff" / "p0_selected.pt"

    with pytest.raises(ValueError, match="stage"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            selected,
            results_root=results_root,
            stage=stage,
        )


def test_selected_checkpoint_resolver_requires_absolute_root_and_exact_stage_path(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    handoff = results_root / "handoff"
    handoff.mkdir(parents=True)
    p0_selected = handoff / "p0_selected.pt"
    p1_selected = handoff / "p1_selected.pt"
    p0_selected.write_bytes(b"p0")
    p1_selected.write_bytes(b"p1")

    with pytest.raises(ValueError, match="results_root.*absolute"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            Path("results/handoff/p0_selected.pt"),
            results_root=Path("results"),
            stage="p0",
        )
    with pytest.raises(ValueError, match="path.*absolute"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            Path("handoff/p0_selected.pt"),
            results_root=results_root,
            stage="p0",
        )
    with pytest.raises(ValueError, match="exactly"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            results_root / "p0" / "epoch-35.pt",
            results_root=results_root,
            stage="p0",
        )
    with pytest.raises(ValueError, match="exactly"):
        runtime.resolve_formal_v2_navsim_e120_selected_checkpoint(
            p0_selected,
            results_root=results_root,
            stage="p1",
        )


class _FakeDistributed:
    def __init__(self, received_status: object | None = None) -> None:
        self.received_status = received_status
        self.events: list[tuple[str, object]] = []

    def broadcast_object(self, value: object, *, src: int) -> object:
        self.events.append(("object", value))
        return value if self.received_status is None else self.received_status

    def broadcast_tensor(self, tensor: torch.Tensor, *, src: int, name: str) -> None:
        self.events.append(("tensor", name))


class _StateRecorder:
    def __init__(self) -> None:
        self.loaded: object = None

    def load_state_dict(self, state: object) -> None:
        self.loaded = copy.deepcopy(state)


def test_horizon_exposure_state_tracks_only_successful_optimizer_samples_and_resumes_cumulatively() -> None:
    exposure = runtime.FormalV2NavSimE120HorizonExposureState(prior={0: 2, 1: 3, 2: 5, 3: 7, 4: 11})

    exposure.record(horizon=0, batch_size=11)
    exposure.record(horizon=3, batch_size=13)

    assert exposure.snapshot(device=torch.device("cpu")) == {0: 13, 1: 3, 2: 5, 3: 20, 4: 11}
    exposure.reset_local()
    assert exposure.snapshot(device=torch.device("cpu")) == {0: 2, 1: 3, 2: 5, 3: 7, 4: 11}
    with pytest.raises(ValueError, match="horizon"):
        exposure.record(horizon=5, batch_size=1)
    with pytest.raises(ValueError, match="batch_size"):
        exposure.record(horizon=0, batch_size=0)


def test_distributed_initialization_broadcasts_rank0_error_status_before_any_tensor_collective() -> None:
    distributed = _FakeDistributed()

    def fail() -> object:
        raise ValueError("parent digest mismatch")

    with pytest.raises(RuntimeError, match="ValueError.*parent digest mismatch"):
        runtime.run_rank0_initialization_and_broadcast(
            rank=0,
            modules=_modules(),
            distributed=distributed,
            rank0_initializer=fail,
        )

    assert [kind for kind, _ in distributed.events] == ["object"]


def test_distributed_initialization_broadcasts_all_role_parameters_and_buffers_in_stable_order() -> None:
    distributed = _FakeDistributed()
    marker = {"initialized": True}

    result = runtime.run_rank0_initialization_and_broadcast(
        rank=0,
        modules=_modules(),
        distributed=distributed,
        rank0_initializer=lambda: marker,
    )

    assert result == marker
    assert distributed.events[0][0] == "object"
    assert [name for kind, name in distributed.events if kind == "tensor"] == [
        "encoder.parameter.stem.bias",
        "encoder.parameter.stem.weight",
        "encoder.buffer.encoder_scale",
        "predictor.parameter.projection.weight",
        "predictor.buffer.predictor_scale",
        "planner.parameter.observed_source_embedding.weight",
        "planner.parameter.z_head.weight",
        "planner.buffer.planner_scale",
    ]


def test_nonzero_rank_never_runs_rank0_initializer_and_obeys_broadcast_failure() -> None:
    called = False

    def forbidden() -> object:
        nonlocal called
        called = True
        raise AssertionError("nonzero rank must not initialize")

    distributed = _FakeDistributed(
        received_status={
            "schema": runtime.FORMAL_V2_NAVSIM_E120_DISTRIBUTED_STATUS_SCHEMA,
            "ok": False,
            "result": None,
            "error_type": "ValueError",
            "error_message": "strict load failed",
        }
    )
    with pytest.raises(RuntimeError, match="strict load failed"):
        runtime.run_rank0_initialization_and_broadcast(
            rank=1,
            modules=_modules(),
            distributed=distributed,
            rank0_initializer=forbidden,
        )

    assert called is False
    assert [kind for kind, _ in distributed.events] == ["object"]
