"""Independent Value artifact contract for the NavSim-e120 protocol."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage, cvoi_runtime
from app.vjepa_cowa_world_model.training import cvoi_value as cvoi_value_module
from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL
from app.vjepa_cowa_world_model.training.cvoi_value import build_cvoi_navsim_e120_direct_value_checkpoint


def test_generic_value_checkpoint_api_surface_is_absent() -> None:
    retired_names = {
        "CVOI_VALUE_CHECKPOINT_SCHEMA",
        "CVOI_VALUE_PROTOCOL_LEGACY",
        "CVoIValueCheckpointMetadata",
        "load_prefix_dual_value_checkpoint",
        "save_prefix_dual_value_checkpoint",
        "validate_cvoi_value_checkpoint_metadata",
    }

    assert retired_names.isdisjoint(vars(cvoi_value_module))


def test_retired_e120_signed_header_api_surface_is_absent() -> None:
    retired_names = {
        "CVOI_NAVSIM_E120_ARTIFACT_HEADER_KEY",
        "CVOI_NAVSIM_E120_ARTIFACT_HEADER_SCHEMA",
        "build_cvoi_navsim_e120_artifact_header",
        "validate_cvoi_navsim_e120_value_artifact_header",
    }

    assert not (retired_names & set(vars(cvoi_value_module)))


def test_retained_value_owner_has_no_generic_formal_artifact_dependency() -> None:
    source = Path(cvoi_value_module.__file__).read_text(encoding="utf-8")

    assert "cvoi_formal_v2_artifacts" not in source
    assert "CvoiFormalV2ArtifactHeader" not in source
    assert "require_cvoi_formal_v2_artifact" not in source


def test_runtime_navsim_e120_field_loader_never_calls_world4drive_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff_root = tmp_path / "handoff"
    handoff_root.mkdir()
    calibration_path = handoff_root / "calibration.pt"
    source_model = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
    payload = build_cvoi_navsim_e120_direct_value_checkpoint(
        source_model,
        phase="field_calibrated",
        branch_id="calibration_full",
        epoch=7,
        parents={
            "unguided_planner": {"stage": "p0", "branch_id": "p0_uniform"},
            "field": {"phase": "field_warmup", "branch_id": "field_full"},
        },
    )
    torch.save(payload, calibration_path)
    direct_calls: list[tuple[Path, str, str]] = []
    config = SimpleNamespace(
        cvoi=SimpleNamespace(
            enabled=True,
            protocol_version=CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL,
            stage="guided_planner",
            field_checkpoint=str(calibration_path),
            ablation_signature=SimpleNamespace(
                experiment_role="main",
                branch_id="p1_full",
                cf_field_supervision="hazard_quality",
                field_calibration_mode="local_geometry",
                p0_prefix_mode="uniform",
                gate_feature_mode="full",
            ),
        ),
    )
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_FULL_RESULTS_ROOT", tmp_path)

    assert {
        "_read_value_checkpoint_payload",
        "_load_configured_cvoi_audit",
        "_sha256_file",
        "load_prefix_dual_value_checkpoint",
        "validate_cvoi_world4drive_value_payload",
    }.isdisjoint(vars(cvoi_runtime))
    direct_reader = cvoi_value_module.read_cvoi_navsim_e120_direct_value_checkpoint

    def read_direct(path: str | Path, **kwargs: object) -> dict[str, object]:
        direct_calls.append(
            (
                Path(path),
                str(kwargs["required_phase"]),
                str(kwargs["required_branch_id"]),
            )
        )
        return direct_reader(path, **kwargs)

    monkeypatch.setattr(cvoi_value_module, "read_cvoi_navsim_e120_direct_value_checkpoint", read_direct)
    model = cvoi_runtime.load_cvoi_dual_value_model(config, embed_dim=4, device=torch.device("cpu"))

    assert isinstance(model, PrefixDualValueModel)
    assert direct_calls == [(calibration_path, "field_calibrated", "calibration_full")]
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(torch.equal(model.state_dict()[key], value) for key, value in payload["state_dict"].items())


def test_runtime_gate_provenance_uses_only_the_embedded_manual_oracle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = object()
    config = SimpleNamespace(
        cvoi=SimpleNamespace(
            enabled=True,
            stage="evaluation",
            protocol_version=CVOI_FORMAL_V2_NAVSIM_E120_PROTOCOL,
            oracle_path="/oracle.sqlite3",
            ablation_signature=SimpleNamespace(gate_feature_mode="full"),
        )
    )
    expected = {
        "oracle_protocol": "epdms_v2_one_stage_navtrain_gate_label_v1",
        "gate_pipeline": "offline_navsim_e120_official_epdms_gate_distillation_v1",
        "gate_feature_schema": "cvoi_gate_online_features_v2",
        "gate_feature_mode": "full",
    }
    monkeypatch.setattr(
        cvoi_runtime,
        "open_embedded_oracle_store_v2",
        lambda path: nullcontext(oracle),
    )

    def build_expected(path: Path, artifact: object, *, gate_feature_mode: str) -> dict[str, str]:
        assert path == Path("/oracle.sqlite3")
        assert artifact is oracle
        assert gate_feature_mode == "full"
        return expected

    monkeypatch.setattr(cvoi_runtime, "build_navtrain_gate_checkpoint_provenance", build_expected)

    assert not hasattr(cvoi_runtime, "build_cvoi_oracle_provenance")
    assert not hasattr(cvoi_runtime, "open_oracle_store")
    assert cvoi_runtime.build_cvoi_gate_provenance(config) == expected
