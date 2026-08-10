"""End-to-end path-flow contract for the seven manual NavSim CVoI configs."""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import yaml

from app.vjepa_cowa_world_model.training import cvoi_formal_v2_navsim_e120_planner_integration as planner_integration
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage, cvoi_offline
from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.configs.data import MongoRawConfig
from app.vjepa_cowa_world_model.training.seg_data import AutonomousDrivingDatasetOnlySeg, init_data_only_seg

CONFIG_ROOT = Path("configs/train/navsim/cvoi_manual_full")
PUBLIC_FULL_ROOT = "/path/to/rise/results/cvoi_manual_full"

_PORTABLE_PREFIXES = {
    PUBLIC_FULL_ROOT: None,
    "/path/to/checkpoints/rise": "/opt/rise-assets/checkpoints",
    "/path/to/checkpoints/vitl_merge_3dataset_e50.pt": "/opt/rise-assets/backbones/vitl_merge_3dataset_e50.pt",
    "/path/to/navsim/dataset": "/opt/rise-assets/navsim/dataset",
    "/path/to/counterfactual": "/opt/rise-assets/counterfactual",
}


def _load_configs() -> dict[str, dict[str, object]]:
    return {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in sorted(CONFIG_ROOT.glob("*.yaml"))}


def _rewrite_path_prefixes(value: object, replacements: dict[str, str]) -> object:
    if isinstance(value, dict):
        return {key: _rewrite_path_prefixes(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite_path_prefixes(child, replacements) for child in value]
    if isinstance(value, str):
        for source, target in replacements.items():
            if value == source or value.startswith(f"{source}/"):
                return f"{target}{value.removeprefix(source)}"
    return value


def _custom_graph(full_root: Path) -> dict[str, dict[str, object]]:
    replacements = {
        source: str(full_root) if target is None else target for source, target in _PORTABLE_PREFIXES.items()
    }
    return {
        name: _rewrite_path_prefixes(copy.deepcopy(payload), replacements) for name, payload in _load_configs().items()
    }


def test_custom_absolute_graph_parses_and_keeps_yaml_paths_as_runtime_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_root = (tmp_path / "second-install" / "results" / "full").resolve()
    graph = _custom_graph(full_root)
    poisoned_default = Path("/must/not/select/module-default")
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_FULL_RESULTS_ROOT", poisoned_default)

    parsed = {name: parse_training_config(payload) for name, payload in graph.items()}
    handoffs = cvoi_manual_lineage.derive_cvoi_manual_full_handoffs(full_root)

    for name, config in parsed.items():
        if name == "01_predictor_lewm_pure.yaml":
            continue
        assert cvoi_manual_lineage.resolve_cvoi_manual_full_results_root_from_config(config.cvoi) == full_root

    assert parsed["03_field_full.yaml"].cvoi.unguided_planner_checkpoint == str(handoffs["p0_handoff"])
    assert parsed["04_calibration_full.yaml"].cvoi.field_checkpoint == str(handoffs["field_handoff"])
    assert parsed["05_p1_full.yaml"].cvoi.field_checkpoint == str(handoffs["calibration_handoff"])
    assert parsed["06_stop_full.yaml"].cvoi.guided_planner_checkpoint == str(handoffs["p1_handoff"])
    assert parsed["07_gate_full.yaml"].cvoi.oracle_path == str(handoffs["oracle_handoff"])
    assert parsed["07_gate_full.yaml"].cvoi.output_checkpoint == str(handoffs["gate_handoff"])

    field_roots = parsed["03_field_full.yaml"].data.navsim
    assert field_roots is not None
    cf_train = next(root for root in field_roots.train_roots if root["domain"] == "counterfactual")
    cf_val = next(root for root in field_roots.val_roots if root["domain"] == "counterfactual")
    assert cf_train["trajectory_quality_path"] == str(full_root / "preflight/trajectory_quality/navsim_cf_train.json")
    assert cf_val["trajectory_quality_path"] == str(full_root / "preflight/trajectory_quality/navsim_cf_val.json")


def test_custom_p1_plan_uses_the_parsed_full_root_without_module_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_root = (tmp_path / "runtime" / "full").resolve()
    graph = _custom_graph(full_root)
    config = parse_training_config(graph["05_p1_full.yaml"])
    handoff_dir = full_root / "handoff"
    handoff_dir.mkdir(parents=True)
    Path(config.cvoi.unguided_planner_checkpoint).write_bytes(b"selected P0 placeholder")
    Path(config.cvoi.field_checkpoint).write_bytes(b"Calibration placeholder")

    observed_results_roots: list[Path] = []

    def _resolve_parent(path: Path, *, results_root: Path, stage: str) -> Path:
        observed_results_roots.append(Path(results_root))
        assert stage == "p0"
        return Path(path)

    monkeypatch.setattr(
        planner_integration,
        "resolve_formal_v2_navsim_e120_selected_checkpoint",
        _resolve_parent,
    )
    monkeypatch.setattr(
        planner_integration,
        "read_formal_v2_navsim_e120_direct_checkpoint",
        lambda _path: {"stage": "p0", "lineage": {"branch_id": "p0_uniform"}, "epoch": 35},
    )
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        Path("/must/not/select/module-default"),
    )

    plan = planner_integration.build_formal_v2_navsim_e120_planner_plan(config)

    assert observed_results_roots == [full_root]
    assert plan.parent_checkpoint_path == handoffs_path(full_root, "p0_selected.pt")
    assert plan.calibration_checkpoint_path == handoffs_path(full_root, "calibration.pt")


def handoffs_path(full_root: Path, filename: str) -> Path:
    return full_root / "handoff" / filename


def test_unrelated_data_defaults_use_neutral_public_examples() -> None:
    parsed = parse_training_config({"data": {"mongo_raw": {"enabled": False}}})

    assert parsed.segmentation.seg_data_root == "/path/to/segmentation/annotations"
    assert parsed.data.mongo_raw is not None
    assert parsed.data.mongo_raw.default_storage_root == "/path/to/mongo/default-storage"
    assert parsed.data.mongo_raw.e2e_storage_root == "/path/to/mongo/e2e-storage"
    assert parsed.data.mongo_raw.clipdata_storage_root == "/path/to/mongo/clipdata-storage"
    assert MongoRawConfig().default_storage_root == "/path/to/mongo/default-storage"

    dataset_default = inspect.signature(AutonomousDrivingDatasetOnlySeg).parameters["seg_data_root"].default
    loader_default = inspect.signature(init_data_only_seg).parameters["seg_data_root"].default
    assert dataset_default == "/path/to/segmentation/annotations"
    assert loader_default == "/path/to/segmentation/annotations"


def test_public_placeholders_parse_but_fail_the_offline_production_preflight() -> None:
    field_payload = _load_configs()["03_field_full.yaml"]
    config = parse_training_config(field_payload)

    with pytest.raises(ValueError, match=r"production preflight.*replace.*?/path/to/"):
        cvoi_offline._preflight_direct_navsim_e120_value_handoffs(config)


@pytest.mark.parametrize("config_name", ["02_p0_uniform.yaml", "05_p1_full.yaml"])
def test_public_planner_placeholders_parse_but_fail_the_production_build(config_name: str) -> None:
    config = parse_training_config(_load_configs()[config_name])

    with pytest.raises(ValueError, match=r"Planner production preflight.*replace.*?/path/to/"):
        planner_integration.build_formal_v2_navsim_e120_planner_plan(config)
