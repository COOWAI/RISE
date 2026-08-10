"""Static contract for the seven manually launched CVOI training configs."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (
    FORMAL_V2_NAVSIM_DIRECT_PREFLIGHT_ROOT,
    build_formal_v2_navsim_direct_task_projection,
    build_formal_v2_navsim_root_catalog,
)

ROOT = Path("configs/train/navsim/cvoi_manual_full")
RESULTS = "/path/to/rise/results/cvoi_manual_full"
WARMSTART = "/path/to/checkpoints/rise/e120.pt"
WARMSTART_PARAMS = "/path/to/checkpoints/rise/params-pretrain.yaml"
FILES = {
    "01_predictor_lewm_pure.yaml",
    "02_p0_uniform.yaml",
    "03_field_full.yaml",
    "04_calibration_full.yaml",
    "05_p1_full.yaml",
    "06_stop_full.yaml",
    "07_gate_full.yaml",
}
STAGES = {
    "02_p0_uniform.yaml": ("unguided_planner", "p0_uniform", "p0"),
    "03_field_full.yaml": ("field_warmup", "full", "field"),
    "04_calibration_full.yaml": ("field_calibrated", "full", "calibration"),
    "05_p1_full.yaml": ("guided_planner", "p1_full", "p1"),
    "06_stop_full.yaml": ("stop_calibrated", "full", "stop"),
    "07_gate_full.yaml": ("gate_distillation", "full", "gate"),
}
OPTIMIZATION = {
    "02_p0_uniform.yaml": (50, 2e-4, 15, 15, [10, 20, 30, 35, 40, 45, 50]),
    "03_field_full.yaml": (1, 1e-4, 0, 0, []),
    "04_calibration_full.yaml": (1, 5e-5, 0, 0, []),
    "05_p1_full.yaml": (80, 2e-4, 15, 15, list(range(5, 81, 5))),
    "06_stop_full.yaml": (1, 5e-5, 0, 0, []),
    "07_gate_full.yaml": (50, 1e-3, 0, 0, []),
}
ARTIFACT_FIELDS = {
    "world_model_checkpoint",
    "seed_planner_checkpoint",
    "unguided_planner_checkpoint",
    "field_checkpoint",
    "guided_planner_checkpoint",
    "dual_value_checkpoint",
    "oracle_path",
    "gate_checkpoint",
    "output_checkpoint",
    "token_ae_checkpoint",
}
REMOVED_GENERIC_CVOI_FIELDS = {
    "artifact_only",
    "audit_manifest_path",
    "audit_path_mode",
    "audit_verification_mode",
}
ARTIFACTS = {
    "02_p0_uniform.yaml": {
        "world_model_checkpoint": WARMSTART,
        "seed_planner_checkpoint": WARMSTART,
        "output_checkpoint": f"{RESULTS}/p0/p0_planner_checkpoint.pt",
    },
    "03_field_full.yaml": {
        "world_model_checkpoint": WARMSTART,
        "seed_planner_checkpoint": WARMSTART,
        "unguided_planner_checkpoint": f"{RESULTS}/handoff/p0_selected.pt",
        "output_checkpoint": f"{RESULTS}/handoff/field.pt",
    },
    "04_calibration_full.yaml": {
        "world_model_checkpoint": WARMSTART,
        "seed_planner_checkpoint": WARMSTART,
        "unguided_planner_checkpoint": f"{RESULTS}/handoff/p0_selected.pt",
        "field_checkpoint": f"{RESULTS}/handoff/field.pt",
        "output_checkpoint": f"{RESULTS}/handoff/calibration.pt",
    },
    "05_p1_full.yaml": {
        "world_model_checkpoint": WARMSTART,
        "seed_planner_checkpoint": WARMSTART,
        "unguided_planner_checkpoint": f"{RESULTS}/handoff/p0_selected.pt",
        "field_checkpoint": f"{RESULTS}/handoff/calibration.pt",
        "output_checkpoint": f"{RESULTS}/p1/p1_planner_checkpoint.pt",
    },
    "06_stop_full.yaml": {
        "world_model_checkpoint": WARMSTART,
        "seed_planner_checkpoint": WARMSTART,
        "unguided_planner_checkpoint": f"{RESULTS}/handoff/p0_selected.pt",
        "field_checkpoint": f"{RESULTS}/handoff/calibration.pt",
        "guided_planner_checkpoint": f"{RESULTS}/handoff/p1_selected.pt",
        "output_checkpoint": f"{RESULTS}/handoff/stop.pt",
    },
    "07_gate_full.yaml": {
        "oracle_path": f"{RESULTS}/handoff/oracle_full.sqlite3",
        "output_checkpoint": f"{RESULTS}/handoff/gate.pt",
    },
}
PROJECTION_BY_FILE = {
    "02_p0_uniform.yaml": ("p0", "uniform"),
    "03_field_full.yaml": ("field", "full"),
    "04_calibration_full.yaml": ("calibration", "full"),
    "05_p1_full.yaml": ("p1", "full"),
    "06_stop_full.yaml": ("stop", "full"),
}
FORBIDDEN_MARKERS = (
    "extends",
    "artifact_only",
    "$deferred",
    "$deferred_artifact_path",
    "$deferred_source_path",
    "$delete",
    "{source_commit}",
    "audit",
    "receipt",
    "sha256",
    "provenance",
)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} must contain a YAML mapping"
    return payload


def _configs() -> dict[str, dict[str, Any]]:
    assert ROOT.is_dir(), f"missing manual config directory: {ROOT}"
    return {name: _load(ROOT / name) for name in sorted(FILES)}


def test_manual_config_inventory_is_exactly_the_seven_agreed_files() -> None:
    assert ROOT.is_dir(), f"missing manual config directory: {ROOT}"
    assert {path.name for path in ROOT.glob("*.yaml")} == FILES


def test_all_manual_configs_are_flat_mappings_without_control_plane_proofs_and_parse() -> None:
    for name, payload in _configs().items():
        parse_training_config(copy.deepcopy(payload))
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
        for marker in FORBIDDEN_MARKERS:
            assert marker not in serialized, f"{name} contains forbidden marker {marker!r}"


def test_predictor_config_keeps_the_approved_lewm_pure_semantics() -> None:
    predictor = _configs()["01_predictor_lewm_pure.yaml"]
    assert predictor["method"] == "lewm"
    assert predictor["planner"]["use_planner"] is False
    assert predictor["train"]["encoder_train"] is True
    assert predictor["train"]["predictor_train"] is True
    assert predictor["train"]["predictor_type"] == "ac_transformer"
    assert predictor["data"]["navsim"]["image_require_policy"] == "all_frames"
    assert predictor["optimization"]["epochs"] == 80
    assert predictor["optimization"]["lr"] == 2e-4
    assert predictor["optimization"]["warmup"] == 8
    assert predictor["optimization"]["anneal"] == 15


def test_cvoi_stage_identity_resources_schedule_and_guidance_are_preserved() -> None:
    for name, payload in _configs().items():
        if name == "01_predictor_lewm_pure.yaml":
            continue
        expected_stage, expected_branch, folder = STAGES[name]
        epochs, learning_rate, warmup, anneal, candidates = OPTIMIZATION[name]

        assert payload["folder"] == f"{RESULTS}/{folder}"
        assert payload["nodes"] == 1
        assert payload["tasks_per_node"] == (8 if name in {"02_p0_uniform.yaml", "05_p1_full.yaml"} else 1)
        assert payload["cvoi"]["stage"] == expected_stage
        assert payload["cvoi"]["ablation_signature"]["branch_id"] == expected_branch
        assert payload["optimization"]["epochs"] == epochs
        assert payload["optimization"]["lr"] == learning_rate
        assert payload["optimization"]["warmup"] == warmup
        assert payload["optimization"]["anneal"] == anneal
        assert payload["meta"]["selection_checkpoint_epochs"] == candidates

    p0 = _configs()["02_p0_uniform.yaml"]
    assert p0["predictor_dynamic_rollout"]["horizon_probabilities"] == [0.2] * 5
    assert p0["value_guidance"]["enabled"] is False
    assert p0["train"]["predictor_planner_finetune"] is True
    p1 = _configs()["05_p1_full.yaml"]
    assert p1["value_guidance"]["enabled"] is True
    assert p1["train"]["predictor_planner_finetune"] is True


def test_cvoi_artifact_handoffs_are_stage_exact_and_warmstarts_are_path_only() -> None:
    for name, payload in _configs().items():
        if name == "01_predictor_lewm_pure.yaml":
            continue
        cvoi = payload["cvoi"]
        configured = {key: cvoi[key] for key in ARTIFACT_FIELDS if key in cvoi}
        assert configured == ARTIFACTS[name]
        if name == "07_gate_full.yaml":
            assert "full_state_warmstart" not in cvoi
            assert "navsim_selection" not in cvoi
            continue
        assert cvoi["full_state_warmstart"] == {
            "schema": "cvoi_full_state_warmstart_config_v1",
            "import_mode": "full_state_warmstart",
            "source_checkpoint": {"path": WARMSTART},
            "source_params_pretrain": {"path": WARMSTART_PARAMS},
        }
        assert "navsim_selection" not in cvoi

    for name in ("02_p0_uniform.yaml", "05_p1_full.yaml"):
        payload = _configs()[name]
        assert payload["meta"]["predictor_checkpoint"] is None
        assert payload["meta"]["pretrain_checkpoint_full"] is None
        assert f"{RESULTS}/predictor" not in json.dumps(payload, sort_keys=True)


def test_e120_parser_rejects_every_removed_proof_binding() -> None:
    baseline = _configs()["03_field_full.yaml"]
    parsed = parse_training_config(copy.deepcopy(baseline))
    warmstart = parsed.cvoi.full_state_warmstart

    assert warmstart is not None
    assert not hasattr(warmstart.source_checkpoint, "sha256")
    assert not hasattr(warmstart.source_params_pretrain, "sha256")
    assert not hasattr(warmstart, "receipt_path")
    assert not hasattr(parsed.cvoi, "navsim_selection")
    assert not hasattr(parsed.cvoi, "field_validation_receipt_path")
    assert not hasattr(parsed.cvoi, "field_validation_source_config_path")
    for field_name in REMOVED_GENERIC_CVOI_FIELDS:
        assert not hasattr(parsed.cvoi, field_name)

    mutations = (
        lambda payload: payload["cvoi"]["full_state_warmstart"]["source_checkpoint"].update(sha256="0" * 64),
        lambda payload: payload["cvoi"]["full_state_warmstart"].update(receipt_path="/results/receipt.json"),
        lambda payload: payload["cvoi"].update(navsim_selection={}),
        lambda payload: payload["cvoi"].update(field_validation_receipt_path="/results/field.json"),
        lambda payload: payload["cvoi"].update(field_validation_source_config_path="/results/field.yaml"),
        lambda payload: payload["cvoi"].update(artifact_only=False),
        lambda payload: payload["cvoi"].update(audit_manifest_path="/results/audit.json"),
        lambda payload: payload["cvoi"].update(audit_path_mode="exact"),
        lambda payload: payload["cvoi"].update(audit_verification_mode="live"),
    )
    for mutate in mutations:
        payload = copy.deepcopy(baseline)
        mutate(payload)
        with pytest.raises(ValueError):
            parse_training_config(payload)


def test_non_gate_data_roots_equal_the_direct_authority_projection() -> None:
    configs = _configs()
    catalog = build_formal_v2_navsim_root_catalog()
    for name, (stage, branch) in PROJECTION_BY_FILE.items():
        navsim = configs[name]["data"]["navsim"]
        expected = build_formal_v2_navsim_direct_task_projection(
            stage,
            branch,
            catalog,
            FORMAL_V2_NAVSIM_DIRECT_PREFLIGHT_ROOT,
        )
        assert {
            "train_roots": navsim["train_roots"],
            "val_roots": navsim["val_roots"],
            "balance_train_roots": navsim["balance_train_roots"],
        } == expected

    field = configs["03_field_full.yaml"]["data"]["navsim"]
    assert [root["domain"] for root in field["train_roots"]] == ["real", "counterfactual"]
    assert field["balance_train_roots"] is True
    for name in (
        "02_p0_uniform.yaml",
        "04_calibration_full.yaml",
        "05_p1_full.yaml",
        "06_stop_full.yaml",
    ):
        navsim = configs[name]["data"]["navsim"]
        assert [root["domain"] for root in navsim["train_roots"]] == ["real"]
        assert navsim["balance_train_roots"] is False


def test_gate_is_oracle_only_and_contains_no_navsim_or_model_initialization() -> None:
    gate = _configs()["07_gate_full.yaml"]
    assert "navsim" not in gate["data"]
    assert gate["cvoi"]["gate_training_batch_size"] == 4096
    assert gate["cvoi"]["oracle_path"] == f"{RESULTS}/handoff/oracle_full.sqlite3"
    assert gate["cvoi"]["output_checkpoint"] == f"{RESULTS}/handoff/gate.pt"
    assert ARTIFACT_FIELDS.intersection(gate["cvoi"]) == {"oracle_path", "output_checkpoint"}


def test_runbook_generates_both_exact_field_quality_sidecars_manually() -> None:
    runbook = (ROOT / "README.md").read_text(encoding="utf-8")
    field_roots = _configs()["03_field_full.yaml"]["data"]["navsim"]
    train_root = next(root for root in field_roots["train_roots"] if root["domain"] == "counterfactual")
    val_root = next(root for root in field_roots["val_roots"] if root["domain"] == "counterfactual")

    assert runbook.count("tools/generate_navsim_cf_trajectory_quality.py") == 2
    assert runbook.count("--formal-v2-annotations") == 2
    assert '"$CVOI_RESULTS_ROOT/preflight/trajectory_quality/navsim_cf_train.json"' in runbook
    assert '"$CVOI_RESULTS_ROOT/preflight/trajectory_quality/navsim_cf_val.json"' in runbook
    for root in (train_root, val_root):
        assert f"--pkl-root {root['data_path']}" in runbook
        assert f"--pose-overlay-root {root['pose_overlay_path']}" in runbook
        assert f"--formal-v2-annotations {root['annotations_path']}" in runbook
