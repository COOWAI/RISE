"""Tests for the Full manual NavSim CVoI lineage authority."""

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.vjepa_cowa_world_model.training import cvoi_manual_lineage as lineage_module
from app.vjepa_cowa_world_model.training.cvoi_manual_lineage import (
    CVOI_MANUAL_ABLATION_RESULTS_ROOT,
    CVOI_MANUAL_FULL_RESULTS_ROOT,
    CvoiManualRuntimeValueInput,
    build_cvoi_manual_value_parents,
    resolve_cvoi_manual_gate_branch,
    resolve_cvoi_manual_runtime_value_input,
    resolve_cvoi_manual_value_lineage,
    resolve_cvoi_manual_value_lineage_by_checkpoint_branch,
)

EXPECTED_FULL_RESULTS_ROOT = Path("/path/to/rise/results/cvoi_manual_full")
EXPECTED_ABLATION_RESULTS_ROOT = Path("/path/to/rise/results/cvoi_manual_ablation")
EXPECTED_FULL_HANDOFF_SUFFIXES = {
    "p0_handoff": Path("handoff/p0_selected.pt"),
    "field_handoff": Path("handoff/field.pt"),
    "calibration_handoff": Path("handoff/calibration.pt"),
    "p1_handoff": Path("handoff/p1_selected.pt"),
    "stop_handoff": Path("handoff/stop.pt"),
    "oracle_handoff": Path("handoff/oracle_full.sqlite3"),
    "gate_handoff": Path("handoff/gate.pt"),
}


def _signature(*, stage: str, **updates: str) -> SimpleNamespace:
    values = {
        "experiment_role": "main",
        "branch_id": "p1_full" if stage == "guided_planner" else "full",
        "cf_field_supervision": "hazard_quality",
        "field_calibration_mode": "local_geometry",
        "p0_prefix_mode": "uniform",
        "gate_feature_mode": "full",
        **updates,
    }
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("stage", "signature_branch", "checkpoint_branch"),
    [
        ("field_warmup", "full", "field_full"),
        ("field_calibrated", "full", "calibration_full"),
        ("guided_planner", "p1_full", "p1_full"),
        ("stop_calibrated", "full", "stop_full"),
    ],
)
def test_full_value_lineage_resolves_exact_stage_identity(
    stage: str,
    signature_branch: str,
    checkpoint_branch: str,
) -> None:
    lineage = resolve_cvoi_manual_value_lineage(
        _signature(stage=stage, branch_id=signature_branch),
        stage=stage,
    )

    assert lineage.name == "full"
    assert lineage.cf_field_supervision == "hazard_quality"
    assert lineage.checkpoint_branch_id(stage) == checkpoint_branch
    assert lineage.p0_branch_id == "p0_uniform"
    assert lineage.result_root == EXPECTED_FULL_RESULTS_ROOT
    with pytest.raises(FrozenInstanceError):
        lineage.name = "changed"


def test_full_value_lineage_uses_configured_canonical_results_root() -> None:
    results_root = Path("/opt/rise-user/results/full")

    lineage = resolve_cvoi_manual_value_lineage(
        _signature(stage="field_warmup"),
        stage="field_warmup",
        full_results_root=results_root,
    )

    assert lineage.result_root == results_root
    assert lineage.p0_handoff == results_root / "handoff/p0_selected.pt"


def test_full_value_lineage_does_not_consult_the_ablation_default(monkeypatch: pytest.MonkeyPatch) -> None:
    results_root = Path("/opt/rise-user/results/full")
    monkeypatch.setattr(
        lineage_module,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        Path("relative-default-must-not-be-read"),
    )

    lineage = resolve_cvoi_manual_value_lineage(
        _signature(stage="field_warmup"),
        stage="field_warmup",
        full_results_root=results_root,
    )

    assert lineage.result_root == results_root


def test_value_lineage_construction_requires_explicit_p0_result_root() -> None:
    with pytest.raises(TypeError, match="p0_result_root"):
        lineage_module.CvoiManualValueLineage(
            name="full",
            cf_field_supervision="hazard_quality",
            result_root=Path("/opt/rise-user/results/full"),
        )


@pytest.mark.parametrize(
    ("name", "result_root", "p0_result_root", "message"),
    [
        (
            "full",
            Path("/opt/rise-user/results/full"),
            Path("results/full"),
            "p0_result_root.*absolute",
        ),
        (
            "full",
            Path("/opt/rise-user/results/full"),
            Path("/opt/rise-user/results/other"),
            "full.*p0_result_root.*result_root",
        ),
        (
            "no_cf",
            Path("/opt/rise-user/results/ablation/no_cf"),
            Path("/opt/rise-user/results/ablation/no_cf"),
            "ablation.*p0_result_root.*separate",
        ),
    ],
)
def test_value_lineage_construction_rejects_invalid_or_inconsistent_roots(
    name: str,
    result_root: Path,
    p0_result_root: Path,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        lineage_module.CvoiManualValueLineage(
            name=name,
            cf_field_supervision="hazard_quality",
            result_root=result_root,
            p0_result_root=p0_result_root,
        )


@pytest.mark.parametrize(
    "results_root",
    [EXPECTED_FULL_RESULTS_ROOT, Path("/opt/rise-user/results/full")],
)
def test_full_handoffs_derive_from_any_canonical_absolute_results_root(results_root: Path) -> None:
    handoffs = lineage_module.derive_cvoi_manual_full_handoffs(results_root)

    assert handoffs == {name: results_root / suffix for name, suffix in EXPECTED_FULL_HANDOFF_SUFFIXES.items()}
    assert lineage_module.resolve_cvoi_manual_full_results_root(handoffs) == results_root


def test_full_handoff_derivation_accepts_absolute_symlink_root(tmp_path: Path) -> None:
    physical_root = tmp_path / "physical-results"
    physical_root.mkdir()
    symlink_root = tmp_path / "linked-results"
    symlink_root.symlink_to(physical_root, target_is_directory=True)

    handoffs = lineage_module.derive_cvoi_manual_full_handoffs(symlink_root)

    assert handoffs == {name: symlink_root / suffix for name, suffix in EXPECTED_FULL_HANDOFF_SUFFIXES.items()}
    assert lineage_module.resolve_cvoi_manual_full_results_root(handoffs) == symlink_root


@pytest.mark.parametrize(
    "results_root",
    [
        Path("results/full"),
        Path("/opt/rise-user/results/full/../full"),
    ],
)
def test_full_handoff_derivation_rejects_relative_or_traversing_root(results_root: Path) -> None:
    with pytest.raises(ValueError, match="results root.*(absolute|traversal)"):
        lineage_module.derive_cvoi_manual_full_handoffs(results_root)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"gate_handoff": Path("/opt/rise-user/results/other/handoff/gate.pt")},
            "same results root",
        ),
        (
            {"p0_handoff": Path("/opt/rise-user/results/full/handoff/p0.pt")},
            "p0_selected.pt",
        ),
        (
            {"p0_handoff": Path("results/full/handoff/p0_selected.pt")},
            "absolute",
        ),
        (
            {"p0_handoff": Path("/opt/rise-user/results/full/../full/handoff/p0_selected.pt")},
            "traversal",
        ),
    ],
)
def test_full_handoff_root_resolution_rejects_structural_path_drift(
    updates: dict[str, Path],
    message: str,
) -> None:
    results_root = Path("/opt/rise-user/results/full")
    handoffs = lineage_module.derive_cvoi_manual_full_handoffs(results_root)
    handoffs.update(updates)

    with pytest.raises(ValueError, match=message):
        lineage_module.resolve_cvoi_manual_full_results_root(handoffs)


def test_full_handoff_validation_rejects_confusing_prefix_sibling() -> None:
    results_root = Path("/opt/rise-user/results/full")
    sibling_path = Path("/opt/rise-user/results/full-other/handoff/p0_selected.pt")

    with pytest.raises(ValueError, match="shared results root.*path must be exactly"):
        lineage_module.resolve_cvoi_manual_full_results_root(
            {"p0_handoff": sibling_path},
            expected_results_root=results_root,
        )


def test_custom_full_root_does_not_change_ablation_root_contract() -> None:
    full_results_root = Path("/opt/rise-user/results/full")
    lineage = resolve_cvoi_manual_value_lineage(
        _signature(
            stage="field_warmup",
            experiment_role="ablation",
            branch_id="no_cf",
            cf_field_supervision="none",
        ),
        stage="field_warmup",
        full_results_root=full_results_root,
    )

    assert CVOI_MANUAL_ABLATION_RESULTS_ROOT == EXPECTED_ABLATION_RESULTS_ROOT
    assert lineage.result_root == EXPECTED_ABLATION_RESULTS_ROOT / "no_cf"
    assert lineage.p0_handoff == full_results_root / "handoff/p0_selected.pt"


def test_ablation_value_lineage_uses_independent_configured_root_and_shared_full_p0() -> None:
    full_results_root = Path("/opt/rise-user/results/full")
    ablation_results_root = Path("/srv/rise-user/results/ablation")

    lineage = resolve_cvoi_manual_value_lineage(
        _signature(
            stage="field_warmup",
            experiment_role="ablation",
            branch_id="no_cf",
            cf_field_supervision="none",
        ),
        stage="field_warmup",
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )

    assert lineage.result_root == ablation_results_root / "no_cf"
    assert lineage.p0_handoff == full_results_root / "handoff/p0_selected.pt"


def test_ablation_shared_p0_uses_neutral_default_full_root() -> None:
    lineage = resolve_cvoi_manual_value_lineage(
        _signature(
            stage="field_warmup",
            experiment_role="ablation",
            branch_id="no_cf",
            cf_field_supervision="none",
        ),
        stage="field_warmup",
    )

    assert lineage.result_root == EXPECTED_ABLATION_RESULTS_ROOT / "no_cf"
    assert lineage.p0_handoff == EXPECTED_FULL_RESULTS_ROOT / "handoff/p0_selected.pt"


@pytest.mark.parametrize(
    "results_root",
    [
        "/opt/rise-user/results/./full",
        "/opt/rise-user//results/full",
        "/opt/rise-user/results/full/",
    ],
)
def test_full_handoff_derivation_rejects_noncanonical_lexical_spelling(results_root: str) -> None:
    with pytest.raises(ValueError, match="canonical.*lexical"):
        lineage_module.derive_cvoi_manual_full_handoffs(results_root)


def test_full_gate_branch_uses_configured_canonical_results_root() -> None:
    results_root = Path("/opt/rise-user/results/full")

    branch = resolve_cvoi_manual_gate_branch(
        _signature(stage="gate_distillation"),
        full_results_root=results_root,
    )

    assert branch.result_root == results_root


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"experiment_role": "other"}, "experiment_role"),
        ({"experiment_role": ""}, "experiment_role"),
        ({"branch_id": "other"}, "branch_id"),
        ({"branch_id": ""}, "branch_id"),
        ({"p0_prefix_mode": "extremes"}, "p0_prefix_mode"),
        ({"gate_feature_mode": "other"}, "gate_feature_mode"),
        ({"field_calibration_mode": "factual_only"}, "mechanism"),
    ],
)
def test_full_value_lineage_rejects_mismatched_signature(
    updates: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_cvoi_manual_value_lineage(
            _signature(stage="field_warmup", **updates),
            stage="field_warmup",
        )


@pytest.mark.parametrize(
    ("evaluation_mode", "phase", "path_field"),
    [
        ("p1_field_forced", "field_calibrated", "field_checkpoint"),
        ("controller", "stop_calibrated", "dual_value_checkpoint"),
    ],
)
def test_full_evaluation_resolves_exact_value_input(
    evaluation_mode: str,
    phase: str,
    path_field: str,
) -> None:
    value_input = resolve_cvoi_manual_runtime_value_input(
        _signature(stage="guided_planner"),
        configured_stage="evaluation",
        evaluation_mode=evaluation_mode,
    )

    assert isinstance(value_input, CvoiManualRuntimeValueInput)
    assert value_input.lineage.name == "full"
    assert value_input.required_phase == phase
    assert value_input.path_field == path_field
    assert value_input.required_branch_id == value_input.lineage.checkpoint_branch_id(phase)


def test_full_evaluation_p0_forced_requires_no_value_input() -> None:
    assert (
        resolve_cvoi_manual_runtime_value_input(
            _signature(stage="guided_planner"),
            configured_stage="evaluation",
            evaluation_mode="p0_forced",
        )
        is None
    )


@pytest.mark.parametrize(
    ("configured_stage", "signature_stage"),
    [
        ("guided_planner", "guided_planner"),
        ("stop_calibrated", "stop_calibrated"),
    ],
)
def test_full_training_runtime_resolves_calibrated_field_input(
    configured_stage: str,
    signature_stage: str,
) -> None:
    value_input = resolve_cvoi_manual_runtime_value_input(
        _signature(stage=signature_stage),
        configured_stage=configured_stage,
    )

    assert isinstance(value_input, CvoiManualRuntimeValueInput)
    assert value_input.required_phase == "field_calibrated"
    assert value_input.path_field == "field_checkpoint"


def test_runtime_value_input_rejects_unsupported_modes() -> None:
    signature = _signature(stage="guided_planner")
    with pytest.raises(ValueError, match="evaluation mode"):
        resolve_cvoi_manual_runtime_value_input(
            signature,
            configured_stage="evaluation",
            evaluation_mode="forced_horizon",
        )
    with pytest.raises(ValueError, match="evaluation_mode"):
        resolve_cvoi_manual_runtime_value_input(
            signature,
            configured_stage="guided_planner",
            evaluation_mode="p1_field_forced",
        )
    assert resolve_cvoi_manual_runtime_value_input(signature, configured_stage="field_warmup") is None


@pytest.mark.parametrize(
    ("phase", "branch_id"),
    [
        ("field_warmup", "field_full"),
        ("field_calibrated", "calibration_full"),
        ("guided_planner", "p1_full"),
        ("stop_calibrated", "stop_full"),
    ],
)
def test_reverse_full_checkpoint_branch_resolution(phase: str, branch_id: str) -> None:
    lineage = resolve_cvoi_manual_value_lineage_by_checkpoint_branch(phase=phase, branch_id=branch_id)
    assert lineage.name == "full"
    assert lineage.checkpoint_branch_id(phase) == branch_id


def test_reverse_checkpoint_branch_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError):
        resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
            phase="guided_planner",
            branch_id="p1_unknown",
        )


def test_full_value_parent_maps_preserve_checkpoint_schema() -> None:
    lineage = resolve_cvoi_manual_value_lineage(
        _signature(stage="stop_calibrated"),
        stage="stop_calibrated",
    )

    assert build_cvoi_manual_value_parents(lineage, "field_warmup") == {
        "unguided_planner": {"stage": "p0", "branch_id": "p0_uniform"}
    }
    assert build_cvoi_manual_value_parents(lineage, "field_calibrated") == {
        "unguided_planner": {"stage": "p0", "branch_id": "p0_uniform"},
        "field": {"phase": "field_warmup", "branch_id": "field_full"},
    }
    assert build_cvoi_manual_value_parents(lineage, "stop_calibrated") == {
        "unguided_planner": {"stage": "p0", "branch_id": "p0_uniform"},
        "calibration": {"phase": "field_calibrated", "branch_id": "calibration_full"},
        "guided_planner": {"stage": "p1", "branch_id": "p1_full"},
    }
    with pytest.raises(ValueError):
        build_cvoi_manual_value_parents(lineage, "guided_planner")


def test_full_gate_branch_resolves_exact_contract() -> None:
    branch = resolve_cvoi_manual_gate_branch(_signature(stage="gate_distillation"))

    assert branch.name == "full"
    assert branch.feature_mode == "full"
    assert branch.oracle_value_lineage == "full"
    assert branch.result_root == EXPECTED_FULL_RESULTS_ROOT


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"experiment_role": "other"}, "experiment_role"),
        ({"p0_prefix_mode": "extremes"}, "p0_prefix_mode"),
        ({"branch_id": "other"}, "branch_id"),
        ({"cf_field_supervision": "none"}, "cf_field_supervision"),
        ({"gate_feature_mode": "other"}, "gate_feature_mode"),
    ],
)
def test_full_gate_branch_rejects_mismatched_contract(
    updates: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_cvoi_manual_gate_branch(_signature(stage="gate_distillation", **updates))


def test_manual_full_result_root_is_absolute() -> None:
    assert CVOI_MANUAL_FULL_RESULTS_ROOT == EXPECTED_FULL_RESULTS_ROOT
    assert isinstance(CVOI_MANUAL_FULL_RESULTS_ROOT, Path)
    assert CVOI_MANUAL_FULL_RESULTS_ROOT.is_absolute()
