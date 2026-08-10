import csv
import hashlib
import inspect
import json
import os
import shutil
import sqlite3
import stat
from argparse import Namespace
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from app.vjepa_cowa_world_model.training import cvoi_manual_lineage
from app.vjepa_cowa_world_model.training import cvoi_manual_navtrain_oracle as manual_oracle
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_E120_LAMBDA_GRID
from app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle import (
    MANUAL_NAVTRAIN_POLICY_TRACE_SCHEMA,
    MANUAL_NAVTRAIN_SCORER_SCHEMA,
    ManualNavTrainAuthorityBundle,
    ManualNavTrainScorerConfig,
    ManualOraclePaths,
    aggregate_manual_navtrain_oracle,
    build_manual_navtrain_manifest,
    load_manual_oracle_environment,
    read_manual_navtrain_authority_bundle,
    read_manual_navtrain_scorer_config,
    score_manual_navtrain_horizon,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import (
    NAVTRAIN_GATE_PROTOCOL_ID,
    open_embedded_oracle_store_v2,
    open_feature_store,
    open_score_store,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import V2_PROTOCOL_ID
from tools import build_cvoi_navsim_scenario_manifest as authority_builder

_DIRECTORY_ENV_NAMES = (
    "CVOI_NAVSIM_DATA_ROOT",
    "CVOI_NAVSIM_EXP_ROOT",
    "CVOI_NUPLAN_MAPS_ROOT",
    "CVOI_NAVSIM_METRIC_CACHE_ROOT",
    "CVOI_NAVSIM_DEVKIT_ROOT",
)
_ENV_NAMES = (*_DIRECTORY_ENV_NAMES, "CVOI_NAVSIM_PYTHON_BIN")
_HANDOFF_NAMES = {
    "p0_planner_checkpoint": "p0_selected.pt",
    "field_checkpoint": "field.pt",
    "calibration_checkpoint": "calibration.pt",
    "p1_planner_checkpoint": "p1_selected.pt",
    "stop_checkpoint": "stop.pt",
}
_OFFICIAL_SCORE_COMPONENTS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_CONFIG_BY_LINEAGE = {
    "full": _REPOSITORY_ROOT / "configs/train/navsim/cvoi_manual_full/05_p1_full.yaml",
    "no_cf": _REPOSITORY_ROOT / "configs/train/navsim/cvoi_manual_full/05_p1_full.yaml",
}
_EXPECTED_LAMBDA_GRID = FORMAL_V2_NAVSIM_E120_LAMBDA_GRID


@dataclass(frozen=True)
class _PreparedScoreInputs:
    results_root: Path
    full_root: Path
    ablation_root: Path
    environment: object
    source_config_path: Path
    lineage: str

    @property
    def artifacts(self) -> dict[str, Path]:
        return {
            "p0_planner_checkpoint": self.full_root / "handoff/p0_selected.pt",
            "field_checkpoint": self.results_root / "handoff/field.pt",
            "calibration_checkpoint": self.results_root / "handoff/calibration.pt",
            "p1_planner_checkpoint": self.results_root / "handoff/p1_selected.pt",
            "stop_checkpoint": self.results_root / "handoff/stop.pt",
        }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _environment_values(tmp_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in _DIRECTORY_ENV_NAMES:
        path = (tmp_path / name.lower()).resolve()
        path.mkdir()
        values[name] = str(path)
    python_bin = (tmp_path / "python-bin").resolve()
    python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_bin.chmod(0o700)
    values["CVOI_NAVSIM_PYTHON_BIN"] = str(python_bin)
    return values


def _write_manual_raw_bundle(output_dir: Path, environment: object, *, split: str = "navtrain") -> None:
    output_dir.mkdir()
    data_root = environment.data_root
    navsim_exp_root = environment.navsim_exp_root
    maps_root = environment.maps_root
    metric_cache_root = environment.metric_cache_root
    devkit_root = environment.devkit_root

    split_file = devkit_root / f"navsim/planning/script/config/common/train_test_split/{split}.yaml"
    split_file.parent.mkdir(parents=True)
    split_file.write_text(f"split: {split}\n", encoding="utf-8")
    scene_filter = split_file.parent / f"scene_filter/{split}.yaml"
    scene_filter.parent.mkdir()
    scene_filter.write_text(f"filter: {split}\n", encoding="utf-8")

    token = "token-a"
    camera_path = data_root / "camera.jpg"
    camera_path.write_bytes(b"camera")
    cache_path = metric_cache_root / token / "metric_cache.pkl"
    cache_path.parent.mkdir()
    cache_bytes = b"cache"
    cache_path.write_bytes(cache_bytes)
    metadata_path = metric_cache_root / "metadata/cache.csv"
    metadata_path.parent.mkdir()
    metadata_bytes = f"file_path\n{cache_path}\n".encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)

    scenario_row = {
        "schema": "cvoi_navsim_scenario_v1",
        "protocol_id": V2_PROTOCOL_ID,
        "scenario_token": token,
        "observation_key": "a" * 64,
        "log_name": "log-a",
        "current_camera_data_path": str(camera_path),
    }
    scenario_bytes = _canonical_json(scenario_row) + b"\n"
    inventory = {
        "schema": "cvoi_navsim_metric_cache_inventory_v1",
        "protocol_id": V2_PROTOCOL_ID,
        "metric_cache_root": str(metric_cache_root),
        "metadata": {
            "path": str(metadata_path),
            "sha256": _sha256(metadata_bytes),
            "header": ["file_path"],
            "row_count": 1,
        },
        "tokens": [token],
        "entries": [
            {
                "scenario_token": token,
                "path": str(cache_path),
                "sha256": _sha256(cache_bytes),
                "size_bytes": len(cache_bytes),
            }
        ],
    }
    inventory_bytes = _canonical_json(inventory)
    manifest = {
        "schema": "cvoi_navsim_raw_v2_authority_v1",
        "protocol_id": V2_PROTOCOL_ID,
        "split": split,
        "roots": {
            "data_root": str(data_root),
            "navsim_exp_root": str(navsim_exp_root),
            "maps_root": str(maps_root),
            "metric_cache_root": str(metric_cache_root),
            "devkit_root": str(devkit_root),
        },
        "sensor_contract": {
            "num_history_frames": 4,
            "cam_f0": [3],
            "cam_l0": False,
            "cam_l1": False,
            "cam_l2": False,
            "cam_r0": False,
            "cam_r1": False,
            "cam_r2": False,
            "cam_b0": False,
            "lidar_pc": False,
        },
        "token_selection": {
            "mode": f"official_{split}",
            "subset_path": None,
            "subset_sha256": None,
        },
        "token_inventory": {"count": 1, "tokens": [token]},
        "artifacts": {
            "scenario_manifest": {
                "path": "scenario_manifest.jsonl",
                "sha256": _sha256(scenario_bytes),
                "row_count": 1,
            },
            "metric_cache_inventory": {
                "path": "metric_cache_inventory.json",
                "sha256": _sha256(inventory_bytes),
            },
        },
        "split_authority": {
            "train_test_split": {"path": str(split_file), "sha256": _sha256(split_file.read_bytes())},
            "scene_filter": {"path": str(scene_filter), "sha256": _sha256(scene_filter.read_bytes())},
        },
    }
    (output_dir / "scenario_manifest.jsonl").write_bytes(scenario_bytes)
    (output_dir / "metric_cache_inventory.json").write_bytes(inventory_bytes)
    (output_dir / "manifest.json").write_bytes(_canonical_json(manifest))


def _patch_manual_lineage_roots(
    monkeypatch: pytest.MonkeyPatch,
    *,
    full_root: Path,
    ablation_root: Path,
) -> None:
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_FULL_RESULTS_ROOT", full_root)
    monkeypatch.setattr(cvoi_manual_lineage, "CVOI_MANUAL_ABLATION_RESULTS_ROOT", ablation_root)
    # Keep tests compatible with an implementation that imports either constant
    # directly while still requiring the fixed lineage authority.
    monkeypatch.setattr(manual_oracle, "CVOI_MANUAL_FULL_RESULTS_ROOT", full_root, raising=False)
    monkeypatch.setattr(manual_oracle, "CVOI_MANUAL_ABLATION_RESULTS_ROOT", ablation_root, raising=False)


def _write_lineage_source_config(
    path: Path,
    *,
    lineage: str,
    full_root: Path,
    results_root: Path,
) -> Path:
    source = yaml.safe_load(_SOURCE_CONFIG_BY_LINEAGE[lineage].read_bytes())
    source["folder"] = str(results_root / "p1")
    cvoi = source["cvoi"]
    cvoi["unguided_planner_checkpoint"] = str(full_root / "handoff/p0_selected.pt")
    cvoi["field_checkpoint"] = str(results_root / "handoff/calibration.pt")
    cvoi["output_checkpoint"] = str(results_root / "p1/p1_planner_checkpoint.pt")
    if lineage == "no_cf":
        signature = cvoi["ablation_signature"]
        signature["experiment_role"] = "ablation"
        signature["branch_id"] = "p1_no_cf"
        signature["cf_field_supervision"] = "none"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    return path.resolve()


def _prepare_score_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    lineage: str = "full",
) -> _PreparedScoreInputs:
    tmp_path.mkdir(parents=True, exist_ok=True)
    environment_values = _environment_values(tmp_path)
    environment_values["CUDA_VISIBLE_DEVICES"] = "0"
    environment = load_manual_oracle_environment(environment_values, require_cuda=True)
    full_root = (tmp_path / "cvoi_manual_full").resolve()
    ablation_root = (tmp_path / "cvoi_manual_ablation").resolve()
    results_root = full_root if lineage == "full" else ablation_root / lineage
    full_root.mkdir()
    ablation_root.mkdir()
    if results_root != full_root:
        results_root.mkdir()
    _patch_manual_lineage_roots(
        monkeypatch,
        full_root=full_root,
        ablation_root=ablation_root,
    )
    (results_root / "oracle/work").mkdir(parents=True)
    _write_manual_raw_bundle(results_root / "oracle/work/manifest", environment)
    source_config_path = _write_lineage_source_config(
        tmp_path / f"{lineage}_05_p1.yaml",
        lineage=lineage,
        full_root=full_root,
        results_root=results_root,
    )
    artifacts = {
        "p0_planner_checkpoint": full_root / "handoff/p0_selected.pt",
        "field_checkpoint": results_root / "handoff/field.pt",
        "calibration_checkpoint": results_root / "handoff/calibration.pt",
        "p1_planner_checkpoint": results_root / "handoff/p1_selected.pt",
        "stop_checkpoint": results_root / "handoff/stop.pt",
    }
    for role, path in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{role}\n".encode("ascii"))
    return _PreparedScoreInputs(
        results_root=results_root,
        full_root=full_root,
        ablation_root=ablation_root,
        environment=environment,
        source_config_path=source_config_path,
        lineage=f"p1_{lineage}",
    )


def _replace_selected_handoffs_with_in_stage_symlinks(prepared: _PreparedScoreInputs) -> None:
    for stage, stage_root in (("p0", prepared.full_root), ("p1", prepared.results_root)):
        candidate_dir = stage_root / stage / "checkpoints"
        candidate_dir.mkdir(parents=True)
        candidate = candidate_dir / "selected.pt"
        candidate.write_bytes(f"{stage}-candidate\n".encode("ascii"))
        selected = stage_root / "handoff" / f"{stage}_selected.pt"
        selected.unlink()
        selected.symlink_to(candidate)


def _write_official_csv(path: Path, *, horizon: int) -> None:
    fields = ["token", "valid", "score", *_OFFICIAL_SCORE_COMPONENTS]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for token, score in (
            ("token-a", 0.4 + horizon * 0.05),
            ("average_all_frames", 0.4 + horizon * 0.05),
        ):
            row = {field: "1.0" for field in _OFFICIAL_SCORE_COMPONENTS}
            row.update(token=token, valid="true", score=str(score))
            writer.writerow(row)


def _fake_score_runner(calls: list[tuple[list[str], dict[str, object]]]):
    def run(arguments: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((arguments, kwargs))
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        config_path = Path(environment["CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH"])
        config = read_manual_navtrain_scorer_config(config_path)
        _write_official_csv(config.output_dir / "scores.csv", horizon=config.forced_horizon)
        for scenario in config.authority.scenario_manifest.scenarios:
            trace = {
                "schema": MANUAL_NAVTRAIN_POLICY_TRACE_SCHEMA,
                "protocol_id": NAVTRAIN_GATE_PROTOCOL_ID,
                "scenario_token": scenario.scenario_token,
                "observation_key": scenario.observation_key,
                "policy_id": config.policy_id,
                "lineage": config.lineage,
                "horizon": config.forced_horizon,
                "gate_features": [float(config.forced_horizon), 1.0, 2.0, 3.0],
                "observed_feature_sha256": "f" * 64,
            }
            (config.trace_output_dir / f"{scenario.observation_key}.json").write_bytes(_canonical_json(trace))
        return SimpleNamespace(returncode=0)

    return run


@pytest.mark.parametrize("lineage", ["full", "no_cf"])
def test_load_manual_oracle_source_resolves_exact_full_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage=lineage)
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        (tmp_path / "poisoned-default-full").resolve(),
    )
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        (tmp_path / "poisoned-default-ablation").resolve(),
    )

    source = manual_oracle.load_manual_oracle_source(
        prepared.source_config_path,
        results_root=prepared.results_root,
    )

    assert isinstance(source, manual_oracle.ManualOracleSource)
    assert source.source_config_path == prepared.source_config_path
    assert source.results_root == prepared.results_root
    assert source.lineage == prepared.lineage
    assert source.value_lineage.name == lineage
    assert source.p0_planner_checkpoint == prepared.full_root / "handoff/p0_selected.pt"
    assert source.field_checkpoint == prepared.results_root / "handoff/field.pt"
    assert source.calibration_checkpoint == prepared.results_root / "handoff/calibration.pt"
    assert source.p1_planner_checkpoint == prepared.results_root / "handoff/p1_selected.pt"
    assert source.stop_checkpoint == prepared.results_root / "handoff/stop.pt"
    assert source.oracle_path == prepared.results_root / "handoff/oracle_full.sqlite3"
    assert source.lambda_grid == _EXPECTED_LAMBDA_GRID
    assert dict(source.artifacts) == prepared.artifacts


def test_load_manual_oracle_source_resolves_runbook_relative_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage="full")
    monkeypatch.setattr(manual_oracle, "_REPOSITORY_ROOT", tmp_path.resolve())

    source = manual_oracle.load_manual_oracle_source(
        prepared.source_config_path.relative_to(tmp_path),
        results_root=prepared.results_root,
    )

    assert source.source_config_path == prepared.source_config_path
    assert source.lineage == "p1_full"


def test_manual_oracle_actions_require_keyword_only_source_config() -> None:
    for action in (score_manual_navtrain_horizon, aggregate_manual_navtrain_oracle):
        parameter = inspect.signature(action).parameters["source_config_path"]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty


def test_manual_oracle_source_rejects_forged_lineage_or_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage="full")
    source = manual_oracle.load_manual_oracle_source(
        prepared.source_config_path,
        results_root=prepared.results_root,
    )
    forged_root = tmp_path / "forged"
    forged_lineage = cvoi_manual_lineage.CvoiManualValueLineage(
        name="full",
        cf_field_supervision="hazard_quality",
        result_root=forged_root,
        p0_result_root=forged_root,
    )
    with pytest.raises(ValueError, match="lineage authority"):
        replace(
            source,
            value_lineage=forged_lineage,
            results_root=forged_root,
            field_checkpoint=forged_lineage.field_handoff,
            calibration_checkpoint=forged_lineage.calibration_handoff,
            p1_planner_checkpoint=forged_lineage.p1_handoff,
            stop_checkpoint=forged_lineage.stop_handoff,
            oracle_path=forged_lineage.oracle_handoff,
        )

    source_link = tmp_path / "source-link.yaml"
    source_link.symlink_to(source.source_config_path)
    with pytest.raises(ValueError, match="source_config_path|non-symlink"):
        replace(source, source_config_path=source_link)


def test_load_manual_oracle_source_rejects_cross_root_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage="full")
    foreign_root = tmp_path / "foreign-results"
    foreign_root.mkdir()

    with pytest.raises(ValueError, match="results_root|lineage|root"):
        manual_oracle.load_manual_oracle_source(
            prepared.source_config_path,
            results_root=foreign_root,
        )


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("cvoi", "unguided_planner_checkpoint"), "/wrong/p0_selected.pt"),
        (("cvoi", "field_checkpoint"), "/wrong/calibration.pt"),
        (("cvoi", "output_checkpoint"), "/wrong/p1_planner_checkpoint.pt"),
        (("cvoi", "stage"), "evaluation"),
        (("cvoi", "ablation_signature", "branch_id"), "p1_other"),
        (("cvoi", "lambda_grid"), [0.0, 0.005]),
    ],
)
def test_load_manual_oracle_source_rejects_path_branch_or_lambda_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    replacement: object,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage="full")
    raw = yaml.safe_load(prepared.source_config_path.read_bytes())
    target = raw
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = replacement
    prepared.source_config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="source|P0|Calibration|P1|branch|lambda|path|stage|guided_planner"):
        manual_oracle.load_manual_oracle_source(
            prepared.source_config_path,
            results_root=prepared.results_root,
        )


@pytest.mark.parametrize("invalid_kind", ["missing", "symlink", "duplicate", "nonmapping"])
def test_load_manual_oracle_source_rejects_invalid_yaml_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage="full")
    source_path = prepared.source_config_path
    if invalid_kind == "missing":
        source_path = tmp_path / "missing.yaml"
    elif invalid_kind == "symlink":
        link = tmp_path / "source-link.yaml"
        link.symlink_to(source_path)
        source_path = link
    elif invalid_kind == "duplicate":
        source_path.write_bytes(source_path.read_bytes() + b"\nstage: guided_planner\n")
    else:
        source_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError), match="source|YAML|mapping|duplicate|exist|symlink"):
        manual_oracle.load_manual_oracle_source(
            source_path,
            results_root=prepared.results_root,
        )


def test_load_manual_oracle_environment_validates_exact_machine_inputs(tmp_path: Path) -> None:
    values = _environment_values(tmp_path)

    without_cuda = load_manual_oracle_environment(values, require_cuda=False)
    with_cuda = load_manual_oracle_environment(
        {**values, "CUDA_VISIBLE_DEVICES": "0,1"},
        require_cuda=True,
    )

    assert without_cuda.data_root == Path(values["CVOI_NAVSIM_DATA_ROOT"])
    assert without_cuda.navsim_exp_root == Path(values["CVOI_NAVSIM_EXP_ROOT"])
    assert without_cuda.maps_root == Path(values["CVOI_NUPLAN_MAPS_ROOT"])
    assert without_cuda.metric_cache_root == Path(values["CVOI_NAVSIM_METRIC_CACHE_ROOT"])
    assert without_cuda.devkit_root == Path(values["CVOI_NAVSIM_DEVKIT_ROOT"])
    assert without_cuda.python_bin == Path(values["CVOI_NAVSIM_PYTHON_BIN"])
    assert without_cuda.cuda_visible_devices is None
    assert with_cuda.cuda_visible_devices == "0,1"


@pytest.mark.parametrize("missing_name", _ENV_NAMES)
def test_load_manual_oracle_environment_rejects_each_missing_input(
    tmp_path: Path,
    missing_name: str,
) -> None:
    values = _environment_values(tmp_path)
    del values[missing_name]

    with pytest.raises(ValueError, match=missing_name):
        load_manual_oracle_environment(values, require_cuda=False)


@pytest.mark.parametrize("invalid_name", _ENV_NAMES)
def test_load_manual_oracle_environment_rejects_relative_inputs(
    tmp_path: Path,
    invalid_name: str,
) -> None:
    values = _environment_values(tmp_path)
    values[invalid_name] = "relative/path"

    with pytest.raises(ValueError, match=invalid_name):
        load_manual_oracle_environment(values, require_cuda=False)


@pytest.mark.parametrize("invalid_name", _ENV_NAMES)
def test_load_manual_oracle_environment_rejects_missing_inputs(
    tmp_path: Path,
    invalid_name: str,
) -> None:
    values = _environment_values(tmp_path)
    values[invalid_name] = str((tmp_path / f"missing-{invalid_name}").resolve())

    with pytest.raises(ValueError, match=invalid_name):
        load_manual_oracle_environment(values, require_cuda=False)


@pytest.mark.parametrize("invalid_name", _ENV_NAMES)
def test_load_manual_oracle_environment_rejects_symlink_inputs(
    tmp_path: Path,
    invalid_name: str,
) -> None:
    values = _environment_values(tmp_path)
    link = tmp_path / f"link-{invalid_name}"
    link.symlink_to(Path(values[invalid_name]), target_is_directory=invalid_name in _DIRECTORY_ENV_NAMES)
    values[invalid_name] = str(link)

    with pytest.raises(ValueError, match=invalid_name):
        load_manual_oracle_environment(values, require_cuda=False)


def test_load_manual_oracle_environment_rejects_non_directory_root(tmp_path: Path) -> None:
    values = _environment_values(tmp_path)
    regular_file = (tmp_path / "not-directory").resolve()
    regular_file.write_bytes(b"x")
    values["CVOI_NAVSIM_DATA_ROOT"] = str(regular_file)

    with pytest.raises(ValueError, match="CVOI_NAVSIM_DATA_ROOT"):
        load_manual_oracle_environment(values, require_cuda=False)


def test_load_manual_oracle_environment_requires_executable_python(tmp_path: Path) -> None:
    values = _environment_values(tmp_path)
    python_bin = Path(values["CVOI_NAVSIM_PYTHON_BIN"])
    python_bin.chmod(0o600)

    with pytest.raises(ValueError, match="CVOI_NAVSIM_PYTHON_BIN.*executable"):
        load_manual_oracle_environment(values, require_cuda=False)


@pytest.mark.parametrize("cuda_value", [None, "", " ", " 0", "0 "])
def test_load_manual_oracle_environment_requires_trimmed_cuda_only_when_requested(
    tmp_path: Path,
    cuda_value: str | None,
) -> None:
    values = _environment_values(tmp_path)
    if cuda_value is not None:
        values["CUDA_VISIBLE_DEVICES"] = cuda_value

    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        load_manual_oracle_environment(values, require_cuda=True)

    without_cuda = load_manual_oracle_environment(values, require_cuda=False)
    assert without_cuda.cuda_visible_devices is None


def test_manual_oracle_paths_use_one_fixed_layout(tmp_path: Path) -> None:
    results_root = tmp_path.resolve()

    paths = ManualOraclePaths.from_results_root(results_root)

    assert paths.results_root == results_root
    assert paths.manifest_dir == tmp_path / "oracle/work/manifest"
    assert paths.horizon_dir(3) == tmp_path / "oracle/work/h3"
    assert paths.score_store(3) == tmp_path / "oracle/work/h3/score_store.sqlite3"
    assert paths.feature_store(3) == tmp_path / "oracle/work/h3/feature_store.sqlite3"
    assert paths.oracle_handoff == tmp_path / "handoff/oracle_full.sqlite3"


@pytest.mark.parametrize("horizon", [-1, 5, True, "2"])
def test_manual_oracle_paths_reject_invalid_horizons(tmp_path: Path, horizon: Any) -> None:
    paths = ManualOraclePaths.from_results_root(tmp_path.resolve())

    with pytest.raises(ValueError, match="horizon"):
        paths.horizon_dir(horizon)


def test_manual_oracle_paths_reject_symlink_results_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="results_root"):
        ManualOraclePaths.from_results_root(linked_root)


def test_read_manual_navtrain_authority_bundle_reads_only_raw_three_file_authority(tmp_path: Path) -> None:
    environment = load_manual_oracle_environment(_environment_values(tmp_path), require_cuda=False)
    bundle_root = (tmp_path / "raw-bundle").resolve()
    _write_manual_raw_bundle(bundle_root, environment)

    bundle = read_manual_navtrain_authority_bundle(bundle_root)

    assert isinstance(bundle, ManualNavTrainAuthorityBundle)
    assert bundle.root == bundle_root
    assert bundle.protocol_id == V2_PROTOCOL_ID
    assert bundle.split == "navtrain"
    assert (
        bundle.data_root,
        bundle.navsim_exp_root,
        bundle.maps_root,
        bundle.metric_cache_root,
        bundle.devkit_root,
    ) == (
        environment.data_root,
        environment.navsim_exp_root,
        environment.maps_root,
        environment.metric_cache_root,
        environment.devkit_root,
    )
    assert bundle.scenario_manifest.tokens == ("token-a",)
    assert bundle.scenario_manifest.scenario_for_token("token-a").log_name == "log-a"
    assert bundle.metric_cache_paths["token-a"] == environment.metric_cache_root / "token-a/metric_cache.pkl"
    assert set(bundle_root.iterdir()) == {
        bundle_root / "manifest.json",
        bundle_root / "scenario_manifest.jsonl",
        bundle_root / "metric_cache_inventory.json",
    }


def test_read_manual_navtrain_authority_bundle_rejects_navtest_cross_split(tmp_path: Path) -> None:
    environment = load_manual_oracle_environment(_environment_values(tmp_path), require_cuda=False)
    bundle_root = (tmp_path / "raw-bundle").resolve()
    _write_manual_raw_bundle(bundle_root, environment, split="navtest")

    with pytest.raises(ValueError, match="split.*expected_split"):
        read_manual_navtrain_authority_bundle(bundle_root)


def test_read_manual_navtrain_authority_bundle_rejects_dag_identity(tmp_path: Path) -> None:
    environment = load_manual_oracle_environment(_environment_values(tmp_path), require_cuda=False)
    bundle_root = (tmp_path / "raw-bundle").resolve()
    _write_manual_raw_bundle(bundle_root, environment)
    manifest_path = bundle_root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["task_identity"] = {
        "task_id": "forbidden",
        "task_signature": "a" * 64,
        "manifest_sha256": "b" * 64,
        "source_commit": "c" * 40,
    }
    manifest_path.write_bytes(_canonical_json(manifest))

    with pytest.raises(ValueError, match="manifest.json.*fields"):
        read_manual_navtrain_authority_bundle(bundle_root)


def test_read_manual_navtrain_authority_bundle_rejects_extra_or_symlink_artifact(tmp_path: Path) -> None:
    environment = load_manual_oracle_environment(_environment_values(tmp_path), require_cuda=False)
    bundle_root = (tmp_path / "raw-bundle").resolve()
    _write_manual_raw_bundle(bundle_root, environment)
    (bundle_root / "receipt.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly"):
        read_manual_navtrain_authority_bundle(bundle_root)

    (bundle_root / "receipt.json").unlink()
    inventory = bundle_root / "metric_cache_inventory.json"
    victim = tmp_path / "victim.json"
    victim.write_bytes(inventory.read_bytes())
    inventory.unlink()
    inventory.symlink_to(victim)

    with pytest.raises(ValueError, match="non-symlink regular"):
        read_manual_navtrain_authority_bundle(bundle_root)


def test_build_manual_navtrain_manifest_calls_raw_builder_once_with_retained_arguments(tmp_path: Path) -> None:
    environment = load_manual_oracle_environment(_environment_values(tmp_path), require_cuda=False)
    results_root = (tmp_path / "results").resolve()
    results_root.mkdir()
    calls: list[Namespace] = []

    def fake_build_bundle(args: Namespace) -> Path:
        calls.append(args)
        assert stat.S_ISDIR(os.fstat(args.output_parent_fd).st_mode)
        _write_manual_raw_bundle(args.output_dir, environment)
        return args.output_dir

    bundle = build_manual_navtrain_manifest(
        results_root,
        environment,
        bundle_builder=fake_build_bundle,
    )

    assert isinstance(bundle, ManualNavTrainAuthorityBundle)
    assert len(calls) == 1
    args = calls[0]
    assert args.protocol_id == V2_PROTOCOL_ID
    assert args.split == "navtrain"
    assert args.data_root == environment.data_root
    assert args.navsim_exp_root == environment.navsim_exp_root
    assert args.maps_root == environment.maps_root
    assert args.metric_cache_root == environment.metric_cache_root
    assert args.devkit_root == environment.devkit_root
    assert args.output_dir == results_root / "oracle/work/manifest"
    assert args.token_subset_path is None
    assert set(vars(args)) == {
        "protocol_id",
        "split",
        "devkit_root",
        "data_root",
        "navsim_exp_root",
        "maps_root",
        "metric_cache_root",
        "output_dir",
        "output_parent_fd",
        "token_subset_path",
    }
    with pytest.raises(OSError):
        os.fstat(args.output_parent_fd)
    assert not (results_root / "handoff").exists()


def test_real_raw_v2_builder_hands_navtrain_authority_to_manual_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = load_manual_oracle_environment(_environment_values(tmp_path), require_cuda=False)
    results_root = (tmp_path / "results").resolve()
    results_root.mkdir()
    token = "token-a"
    cache_path = environment.metric_cache_root / token / "metric_cache.pkl"
    cache_path.parent.mkdir()
    cache_path.write_bytes(b"cache:token-a")
    metadata_path = environment.metric_cache_root / "metadata/cache.csv"
    metadata_path.parent.mkdir()
    metadata_path.write_text(f"file_path\n{cache_path}\n", encoding="utf-8")
    split_path = environment.devkit_root / "navtrain.yaml"
    split_path.write_text("split: navtrain\n", encoding="utf-8")
    filter_path = environment.devkit_root / "navtrain-filter.yaml"
    filter_path.write_text("filter: navtrain\n", encoding="utf-8")

    class SceneFilter:
        tokens = [token]
        num_history_frames = 4

    class SensorConfig:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class SceneLoader:
        def __init__(self, **_kwargs: object) -> None:
            self.tokens = (token,)
            self.scene_frames_dicts = {
                token: [{}, {}, {}, {"log_name": "log-a", "cams": {"cam_f0": {"data_path": "raw/a.jpg"}}}]
            }

        def get_agent_input_from_token(self, scenario_token: str) -> object:
            assert scenario_token == token
            camera = SimpleNamespace(image=np.array([1, 2, 3], dtype=np.uint8), camera_path="raw/a.jpg")
            return SimpleNamespace(cameras=[SimpleNamespace(cam_f0=camera)])

    class MetricCacheLoader:
        def __init__(self, root: Path) -> None:
            assert root == environment.metric_cache_root
            self.tokens = (token,)
            self.metric_cache_paths = {token: cache_path}

    monkeypatch.setattr(
        authority_builder,
        "_load_split_config",
        lambda root, *, split: ("trainval", SceneFilter(), (token,), 4, split_path, filter_path),
    )
    monkeypatch.setattr(
        authority_builder,
        "_import_navsim_loaders",
        lambda: (SensorConfig, SceneLoader, MetricCacheLoader),
    )

    bundle = build_manual_navtrain_manifest(results_root, environment)

    assert isinstance(bundle, ManualNavTrainAuthorityBundle)
    assert bundle.root == results_root / "oracle/work/manifest"
    assert bundle.split == "navtrain"
    assert bundle.scenario_manifest.tokens == (token,)
    assert dict(bundle.metric_cache_paths) == {token: cache_path}
    assert (
        bundle.data_root,
        bundle.navsim_exp_root,
        bundle.maps_root,
        bundle.metric_cache_root,
        bundle.devkit_root,
    ) == (
        environment.data_root,
        environment.navsim_exp_root,
        environment.maps_root,
        environment.metric_cache_root,
        environment.devkit_root,
    )


def test_build_manual_navtrain_manifest_rejects_symlink_parent_before_builder(tmp_path: Path) -> None:
    environment = load_manual_oracle_environment(_environment_values(tmp_path), require_cuda=False)
    results_root = (tmp_path / "results").resolve()
    results_root.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (results_root / "oracle").symlink_to(foreign, target_is_directory=True)
    called = False

    def forbidden_builder(_args: Namespace) -> Path:
        nonlocal called
        called = True
        raise AssertionError("builder must not run")

    with pytest.raises(ValueError, match="oracle"):
        build_manual_navtrain_manifest(
            results_root,
            environment,
            bundle_builder=forbidden_builder,
        )

    assert not called


@pytest.mark.parametrize("lineage", ["full", "no_cf"])
def test_score_manual_navtrain_horizon_is_one_isolated_action_with_canonical_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage=lineage)
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        (tmp_path / "poisoned-default-full").resolve(),
    )
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        (tmp_path / "poisoned-default-ablation").resolve(),
    )
    paths = ManualOraclePaths.from_results_root(prepared.results_root)
    source_bytes = prepared.source_config_path.read_bytes()
    calls: list[tuple[list[str], dict[str, object]]] = []

    result = score_manual_navtrain_horizon(
        prepared.results_root,
        prepared.environment,
        2,
        source_config_path=prepared.source_config_path,
        runner=_fake_score_runner(calls),
    )

    assert len(calls) == 1
    arguments, kwargs = calls[0]
    assert arguments == [
        str(_REPOSITORY_ROOT / "scripts/eval_navsim/eval_navsim_v2_pdms.sh"),
        str(prepared.results_root / "handoff/p1_selected.pt"),
        str(paths.horizon_dir(2) / "effective.yaml"),
        "cvoi-manual-oracle-h2",
    ]
    assert kwargs["cwd"] == _REPOSITORY_ROOT
    assert kwargs["check"] is False
    subprocess_environment = kwargs["env"]
    assert isinstance(subprocess_environment, dict)
    assert subprocess_environment["CVOI_MANUAL_NAVTRAIN_GATE"] == "1"
    assert subprocess_environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert subprocess_environment["MAX_WORKERS"] == "1"
    assert subprocess_environment["USE_PROCESS_POOL"] == "false"
    assert subprocess_environment["FORWARD_MODE"] == "stage12"
    assert subprocess_environment["NAVSIM_OUTPUT_DIR"] == str(paths.horizon_dir(2) / "scorer_output")
    assert {path.name for path in (prepared.results_root / "oracle/work").iterdir() if path.name.startswith("h")} == {
        "h2"
    }
    assert result.score_store_path == paths.score_store(2)
    assert result.feature_store_path == paths.feature_store(2)
    assert prepared.source_config_path.read_bytes() == source_bytes

    scorer_bytes = result.scorer_config_path.read_bytes()
    assert scorer_bytes == _canonical_json(json.loads(scorer_bytes))
    assert all(
        forbidden not in scorer_bytes.decode("ascii")
        for forbidden in ("receipt", "task_id", "source_commit", "sha256", "provenance")
    )
    config = read_manual_navtrain_scorer_config(result.scorer_config_path)
    assert isinstance(config, ManualNavTrainScorerConfig)
    assert config.schema == MANUAL_NAVTRAIN_SCORER_SCHEMA
    assert config.protocol_id == NAVTRAIN_GATE_PROTOCOL_ID
    assert config.policy_id == f"{prepared.lineage}__fixed_h2_k2"
    assert config.lineage == prepared.lineage
    assert config.forced_horizon == 2
    assert config.guidance_steps == 2
    assert config.source_config_path == prepared.source_config_path
    assert dict(config.artifacts) == prepared.artifacts

    projection = yaml.safe_load(result.effective_config_path.read_bytes())
    source = yaml.safe_load(source_bytes)
    assert projection["stage"] == source["stage"]
    assert projection["cvoi"]["stage"] == "evaluation"
    assert projection["cvoi"]["evaluation_mode"] == "p1_field_forced"
    assert projection["cvoi"]["controller_lineage"] == "value_guided"
    assert projection["cvoi"]["world_model_checkpoint"] == source["cvoi"]["world_model_checkpoint"]
    assert projection["cvoi"]["seed_planner_checkpoint"] == source["cvoi"]["seed_planner_checkpoint"]
    assert projection["cvoi"]["unguided_planner_checkpoint"] == str(prepared.full_root / "handoff/p0_selected.pt")
    assert projection["cvoi"]["field_checkpoint"] == str(prepared.results_root / "handoff/calibration.pt")
    assert projection["cvoi"]["guided_planner_checkpoint"] == str(prepared.results_root / "handoff/p1_selected.pt")
    assert projection["cvoi"]["dual_value_checkpoint"] == str(prepared.results_root / "handoff/stop.pt")
    assert projection["cvoi"]["output_checkpoint"] is None
    assert "forced_horizon" not in projection["cvoi"]

    with open_score_store(paths.score_store(2)) as score_store:
        assert score_store.metadata.policy_id == f"{prepared.lineage}__fixed_h2_k2"
        assert score_store.metadata.lineage == prepared.lineage
        assert score_store.metadata.horizon == 2
        assert score_store.row_count == 1
    with open_feature_store(paths.feature_store(2)) as feature_store:
        assert feature_store.metadata.policy_id == f"{prepared.lineage}__fixed_h2_k2"
        assert feature_store.metadata.lineage == prepared.lineage
        assert feature_store.metadata.horizon == 2
        assert tuple(feature_store.iter_rows())[0].features == pytest.approx((2.0, 1.0, 2.0, 3.0))
    trace_path = paths.horizon_dir(2) / "policy_traces" / f"{'a' * 64}.json"
    assert json.loads(trace_path.read_bytes())["lineage"] == prepared.lineage
    assert not any(
        "receipt" in path.name or "completion" in path.name or "lease" in path.name
        for path in paths.horizon_dir(2).rglob("*")
    )


@pytest.mark.parametrize("lineage", ["full"])
def test_score_manual_navtrain_horizon_accepts_selected_symlinks_and_preserves_fixed_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage=lineage)
    _replace_selected_handoffs_with_in_stage_symlinks(prepared)

    result = score_manual_navtrain_horizon(
        prepared.results_root,
        prepared.environment,
        2,
        source_config_path=prepared.source_config_path,
        runner=_fake_score_runner([]),
    )

    config = read_manual_navtrain_scorer_config(result.scorer_config_path)
    assert config.artifacts["p0_planner_checkpoint"] == prepared.full_root / "handoff/p0_selected.pt"
    assert config.artifacts["p1_planner_checkpoint"] == prepared.results_root / "handoff/p1_selected.pt"
    raw = json.loads(result.scorer_config_path.read_bytes())
    assert raw["artifacts"]["p0_planner_checkpoint"] == str(prepared.full_root / "handoff/p0_selected.pt")
    assert raw["artifacts"]["p1_planner_checkpoint"] == str(prepared.results_root / "handoff/p1_selected.pt")


@pytest.mark.parametrize("failure", ["broken", "escape", "wrong_stage"])
def test_score_manual_navtrain_horizon_rejects_invalid_selected_symlink_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch)
    selected = prepared.full_root / "handoff/p0_selected.pt"
    selected.unlink()
    if failure == "broken":
        selected.symlink_to(prepared.full_root / "p0/missing.pt")
    elif failure == "escape":
        outside = tmp_path / "outside.pt"
        outside.write_bytes(b"outside")
        selected.symlink_to(outside)
    else:
        wrong_stage = prepared.full_root / "p1/checkpoints/selected.pt"
        wrong_stage.parent.mkdir(parents=True)
        wrong_stage.write_bytes(b"wrong-stage")
        selected.symlink_to(wrong_stage)
    calls: list[tuple[list[str], dict[str, object]]] = []

    with pytest.raises((FileNotFoundError, ValueError), match="broken|outside"):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            2,
            source_config_path=prepared.source_config_path,
            runner=_fake_score_runner(calls),
        )

    assert calls == []
    assert not (prepared.results_root / "oracle/work/h2").exists()


@pytest.mark.parametrize("horizon", [-1, 5, True, "2"])
def test_score_manual_navtrain_horizon_rejects_invalid_horizon_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    horizon: Any,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch)
    calls: list[tuple[list[str], dict[str, object]]] = []

    with pytest.raises(ValueError, match="horizon"):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            horizon,
            source_config_path=prepared.source_config_path,
            runner=_fake_score_runner(calls),
        )

    assert calls == []


def test_score_manual_navtrain_horizon_rejects_missing_manifest_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch)
    shutil.rmtree(prepared.results_root / "oracle/work/manifest")
    calls: list[tuple[list[str], dict[str, object]]] = []

    with pytest.raises(ValueError, match="manifest"):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            2,
            source_config_path=prepared.source_config_path,
            runner=_fake_score_runner(calls),
        )

    assert calls == []
    assert not (prepared.results_root / "oracle/work/h2").exists()


@pytest.mark.parametrize("missing_name", _HANDOFF_NAMES.values())
def test_score_manual_navtrain_horizon_rejects_each_missing_handoff_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch)
    next(path for path in prepared.artifacts.values() if path.name == missing_name).unlink()
    calls: list[tuple[list[str], dict[str, object]]] = []

    with pytest.raises((FileNotFoundError, ValueError), match="handoff|does not exist"):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            2,
            source_config_path=prepared.source_config_path,
            runner=_fake_score_runner(calls),
        )

    assert calls == []
    assert not (prepared.results_root / "oracle/work/h2").exists()


def test_score_manual_navtrain_horizon_rejects_existing_work_before_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch)
    (prepared.results_root / "oracle/work/h2").mkdir()
    calls: list[tuple[list[str], dict[str, object]]] = []

    with pytest.raises(FileExistsError, match="h2"):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            2,
            source_config_path=prepared.source_config_path,
            runner=_fake_score_runner(calls),
        )

    assert calls == []


def test_score_manual_navtrain_horizon_requires_zero_exit_and_exact_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="exit"):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            2,
            source_config_path=prepared.source_config_path,
            runner=lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
        )

    prepared = _prepare_score_inputs(tmp_path / "second", monkeypatch)

    def no_outputs(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0)

    with pytest.raises(ValueError, match="exactly one official CSV"):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            2,
            source_config_path=prepared.source_config_path,
            runner=no_outputs,
        )


@pytest.mark.parametrize("lineage", ["full"])
def test_aggregate_manual_navtrain_oracle_is_self_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lineage: str,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage=lineage)
    paths = ManualOraclePaths.from_results_root(prepared.results_root)
    for horizon in range(5):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            horizon,
            source_config_path=prepared.source_config_path,
            runner=_fake_score_runner([]),
        )

    receipt = aggregate_manual_navtrain_oracle(
        prepared.results_root,
        source_config_path=prepared.source_config_path,
    )
    assert receipt.path == prepared.results_root / "handoff/oracle_full.sqlite3"
    with open_embedded_oracle_store_v2(receipt.path, expected_sha256=receipt.sha256) as store:
        expected = store.read_training_batch(
            record_ids=[0, 0],
            horizons=[0, 3],
            lambda_computes=[0.0, 0.005],
        )
        assert store.metadata.lineage == prepared.lineage
        assert store.policy_ids_by_horizon == tuple(
            f"{prepared.lineage}__fixed_h{horizon}_k{0 if horizon == 0 else 2}" for horizon in range(5)
        )
        assert store.feature_payload_policy == "embedded_h0_h4_float32_le_v1"

    for horizon in range(5):
        paths.score_store(horizon).unlink()
        paths.feature_store(horizon).unlink()
    with open_embedded_oracle_store_v2(receipt.path, expected_sha256=receipt.sha256) as store:
        actual = store.read_training_batch(
            record_ids=[0, 0],
            horizons=[0, 3],
            lambda_computes=[0.0, 0.005],
        )
    assert actual == expected


@pytest.mark.parametrize("store_kind", ["score", "feature"])
def test_aggregate_manual_navtrain_oracle_rejects_full_h3_lineage_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    store_kind: str,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch, lineage="full")
    paths = ManualOraclePaths.from_results_root(prepared.results_root)
    for horizon in range(5):
        score_manual_navtrain_horizon(
            prepared.results_root,
            prepared.environment,
            horizon,
            source_config_path=prepared.source_config_path,
            runner=_fake_score_runner([]),
        )
    store_path = paths.score_store(3) if store_kind == "score" else paths.feature_store(3)
    with sqlite3.connect(store_path) as connection:
        connection.execute(
            "UPDATE metadata SET value=? WHERE key='lineage'",
            (json.dumps("p1_other", separators=(",", ":")),),
        )

    with pytest.raises(ValueError, match="H3.*identity|H3.*lineage|policy identity"):
        aggregate_manual_navtrain_oracle(
            prepared.results_root,
            source_config_path=prepared.source_config_path,
        )


def test_aggregate_manual_navtrain_oracle_rejects_missing_intermediate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepare_score_inputs(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="H0.*score"):
        aggregate_manual_navtrain_oracle(
            prepared.results_root,
            source_config_path=prepared.source_config_path,
        )
