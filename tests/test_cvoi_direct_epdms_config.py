"""Strict Full-controller contracts for direct NavSim CVoI EPDMS."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from app.vjepa_cowa_world_model.evaluation.cvoi_direct_epdms import (
    CVOI_DIRECT_EPDMS_EFFECTIVE_SCHEMA,
    CVOI_DIRECT_EPDMS_EFFECTIVE_VERSION,
    load_cvoi_direct_epdms_config,
    load_cvoi_direct_epdms_projection,
    project_cvoi_direct_epdms_run,
    run_cvoi_direct_epdms,
    write_cvoi_direct_epdms_projection,
)
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage

PUBLIC_KEYS = {
    "schema",
    "version",
    "branch",
    "run_kind",
    "split",
    "protocol",
    "max_horizon",
    "guidance_steps",
    "training_config_path",
    "encoder_checkpoint_path",
    "scenario_manifest_path",
    "output_root",
    "artifacts",
}
EFFECTIVE_COMMON_KEYS = {
    "schema",
    "version",
    "branch",
    "split",
    "protocol",
    "evaluation_mode",
    "horizon",
    "guidance_steps",
    "training_config_path",
    "encoder_checkpoint_path",
    "scenario_manifest_path",
    "output_directory",
}
CONTROLLER_ARTIFACT_KEYS = {
    "p0_planner_checkpoint_path",
    "calibration_checkpoint_path",
    "p1_planner_checkpoint_path",
    "stop_checkpoint_path",
    "gate_checkpoint_path",
    "oracle_path",
    "gate_feature_mode",
}
EFFECTIVE_CONTROLLER_KEYS = CONTROLLER_ARTIFACT_KEYS - {"oracle_path"} | {"oracle_sha256"}
FORBIDDEN_REPOSITORY_CONTROL_KEY_TOKENS = {
    "audit",
    "bundle",
    "dag",
    "extends",
    "loop",
    "receipt",
    "registry",
    "retry",
    "scheduler",
    "selection",
    "task",
    "tasks",
}

REPOSITORY_EPDMS_CONFIG_ROOT = Path("configs/eval/navsim/cvoi_manual_epdms")
REPOSITORY_EPDMS_FILES = {"full_controller.yaml": ("full", "controller")}
REPOSITORY_EPDMS_ENTRIES = {*REPOSITORY_EPDMS_FILES, "README.md"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_TRAINING_CONFIG = "configs/train/navsim/cvoi_manual_full/05_p1_full.yaml"
PORTABLE_FULL_RESULTS_ROOT = Path("/path/to/rise/results/cvoi_manual_full")
PORTABLE_EPDMS_RESULTS_ROOT = Path("/path/to/rise/results/cvoi_manual_epdms")
PORTABLE_E120_ENCODER_CHECKPOINT = Path("/path/to/checkpoints/rise/e120.pt")
PORTABLE_NAVTEST_SCENARIO_MANIFEST = PORTABLE_FULL_RESULTS_ROOT / "preflight/scenario_manifest.jsonl"


@pytest.fixture(autouse=True)
def _isolate_manual_lineage_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_root = (tmp_path / "manual-authority").resolve()
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_FULL_RESULTS_ROOT",
        authority_root / "cvoi_manual_full",
    )
    monkeypatch.setattr(
        cvoi_manual_lineage,
        "CVOI_MANUAL_ABLATION_RESULTS_ROOT",
        authority_root / "retired",
    )


def _write_public_config(
    tmp_path: Path,
    *,
    mutate: Any = None,
) -> tuple[Path, dict[str, object]]:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    common_paths = {
        name: (inputs / name).resolve() for name in ("training.yaml", "encoder.pt", "scenario_manifest.jsonl")
    }
    for role, path in common_paths.items():
        path.write_bytes(role.encode("ascii"))

    full_root = (tmp_path / "portable-results/cvoi_manual_full").resolve()
    artifact_paths = {
        "p0_planner_checkpoint_path": full_root / "handoff/p0_selected.pt",
        "calibration_checkpoint_path": full_root / "handoff/calibration.pt",
        "p1_planner_checkpoint_path": full_root / "handoff/p1_selected.pt",
        "stop_checkpoint_path": full_root / "handoff/stop.pt",
        "gate_checkpoint_path": full_root / "handoff/gate.pt",
        "oracle_path": full_root / "handoff/oracle_full.sqlite3",
    }
    for artifact_key, artifact_path in artifact_paths.items():
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_key not in {"gate_checkpoint_path", "oracle_path"}:
            artifact_path.write_bytes(artifact_key.encode("ascii"))

    oracle_bytes = b"direct-epdms-test-oracle"
    artifact_paths["oracle_path"].write_bytes(oracle_bytes)
    oracle_sha256 = hashlib.sha256(oracle_bytes).hexdigest()
    torch.save(
        {
            "schema": "sequential_cvoi_gate_navsim_e120_v1",
            "provenance": {
                "oracle_sha256": oracle_sha256,
                "oracle_lineage": "p1_full",
                "gate_feature_mode": "full",
            },
        },
        artifact_paths["gate_checkpoint_path"],
    )
    artifacts: dict[str, object] = {key: str(path) for key, path in artifact_paths.items()}
    artifacts["gate_feature_mode"] = "full"
    payload: dict[str, object] = {
        "schema": "cvoi_direct_epdms",
        "version": 1,
        "branch": "full",
        "run_kind": "controller",
        "split": "navtest",
        "protocol": "epdms_v2_one_stage_navtest",
        "max_horizon": 4,
        "guidance_steps": 2,
        "training_config_path": str(common_paths["training.yaml"]),
        "encoder_checkpoint_path": str(common_paths["encoder.pt"]),
        "scenario_manifest_path": str(common_paths["scenario_manifest.jsonl"]),
        "output_root": str((tmp_path / "outputs/full").resolve()),
        "artifacts": artifacts,
    }
    if mutate is not None:
        mutate(payload)
    config_path = (tmp_path / "full_controller.yaml").resolve()
    config_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return config_path, payload


def _rewrite_gate_checkpoint(raw: dict[str, object], mutate: Any) -> None:
    artifacts = raw["artifacts"]
    if not isinstance(artifacts, dict):
        raise TypeError("test fixture artifacts changed type")
    path = Path(artifacts["gate_checkpoint_path"])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, path)


def _project_full(tmp_path: Path):
    public_path, raw = _write_public_config(tmp_path)
    projection = project_cvoi_direct_epdms_run(
        load_cvoi_direct_epdms_config(public_path),
        forced_horizon=None,
    )
    return public_path, raw, projection


def test_controller_projection_contains_only_online_controller_inputs(tmp_path: Path) -> None:
    _, raw, projection = _project_full(tmp_path)
    artifacts = raw["artifacts"]
    assert isinstance(artifacts, dict)

    assert projection.branch == "full"
    assert projection.split == "navtest"
    assert projection.protocol == "epdms_v2_one_stage_navtest"
    assert projection.evaluation_mode == "controller"
    assert projection.horizon is None
    assert projection.guidance_steps == 2
    assert projection.p0_planner_checkpoint_path == Path(artifacts["p0_planner_checkpoint_path"])
    assert projection.calibration_checkpoint_path == Path(artifacts["calibration_checkpoint_path"])
    assert projection.p1_planner_checkpoint_path == Path(artifacts["p1_planner_checkpoint_path"])
    assert projection.stop_checkpoint_path == Path(artifacts["stop_checkpoint_path"])
    assert projection.gate_checkpoint_path == Path(artifacts["gate_checkpoint_path"])
    assert projection.gate_feature_mode == "full"
    assert projection.oracle_sha256 == hashlib.sha256(Path(artifacts["oracle_path"]).read_bytes()).hexdigest()
    assert not hasattr(projection, "oracle_path")


def test_controller_configuration_forbids_a_forced_horizon(tmp_path: Path) -> None:
    path, _ = _write_public_config(tmp_path)
    config = load_cvoi_direct_epdms_config(path)

    with pytest.raises(ValueError, match="horizon"):
        project_cvoi_direct_epdms_run(config, forced_horizon=0)


def test_full_public_config_rejects_wrong_branch_or_run_kind(tmp_path: Path) -> None:
    for index, mutation in enumerate(
        (
            lambda payload: payload.__setitem__("branch", "other"),
            lambda payload: payload.__setitem__("run_kind", "forced"),
            lambda payload: payload.__setitem__("run_kind", "other"),
        )
    ):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        path, _ = _write_public_config(case_root, mutate=mutation)
        with pytest.raises(ValueError, match="branch|run_kind"):
            load_cvoi_direct_epdms_config(path)


def test_full_public_config_rejects_noncanonical_handoff_authority(tmp_path: Path) -> None:
    def mutate(payload: dict[str, object]) -> None:
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, dict)
        artifacts["calibration_checkpoint_path"] = str(
            (tmp_path / "foreign-results/cvoi_manual_full/handoff/calibration.pt").resolve()
        )

    path, _ = _write_public_config(tmp_path, mutate=mutate)

    with pytest.raises(ValueError, match="authority|handoff|branch|artifact"):
        load_cvoi_direct_epdms_config(path)


@pytest.mark.parametrize(
    "artifact_key",
    [
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
        "stop_checkpoint_path",
        "gate_checkpoint_path",
        "oracle_path",
    ],
)
def test_full_public_config_rejects_every_wrong_handoff_suffix(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    def mutate(payload: dict[str, object]) -> None:
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, dict)
        original = Path(artifacts[artifact_key])
        artifacts[artifact_key] = str(original.with_name(f"wrong-{original.name}"))

    path, _ = _write_public_config(tmp_path, mutate=mutate)

    with pytest.raises(ValueError, match="suffix|handoff|root"):
        load_cvoi_direct_epdms_config(path)


def test_controller_projection_rejects_gate_oracle_digest_drift(tmp_path: Path) -> None:
    path, raw = _write_public_config(tmp_path)
    config = load_cvoi_direct_epdms_config(path)
    artifacts = raw["artifacts"]
    assert isinstance(artifacts, dict)
    Path(artifacts["oracle_path"]).write_bytes(b"changed-after-gate-training")

    with pytest.raises(ValueError, match="Oracle|oracle|SHA"):
        project_cvoi_direct_epdms_run(config, forced_horizon=None)


@pytest.mark.parametrize("nested", [False, True])
def test_public_yaml_rejects_duplicate_keys(tmp_path: Path, nested: bool) -> None:
    path, _ = _write_public_config(tmp_path)
    text = path.read_text(encoding="utf-8")
    if nested:
        text = text.replace(
            "  gate_feature_mode: full\n",
            "  gate_feature_mode: full\n  gate_feature_mode: full\n",
            1,
        )
    else:
        text = text.replace("version: 1\n", "version: 1\nversion: 1\n", 1)
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        load_cvoi_direct_epdms_config(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("schema", "legacy_gate"),
        lambda payload: payload.__setitem__("provenance", []),
        lambda payload: payload.__setitem__("provenance", {}),
        lambda payload: payload["provenance"].pop("oracle_sha256"),
        lambda payload: payload["provenance"].__setitem__("oracle_sha256", b"0" * 64),
        lambda payload: payload["provenance"].__setitem__("oracle_sha256", "0" * 63),
        lambda payload: payload["provenance"].__setitem__("oracle_sha256", "g" * 64),
        lambda payload: payload["provenance"].__setitem__("oracle_sha256", "A" * 64),
        lambda payload: payload["provenance"].__setitem__("oracle_lineage", "p1_other"),
        lambda payload: payload["provenance"].__setitem__("gate_feature_mode", "other"),
    ],
)
def test_controller_projection_rejects_invalid_gate_authority(
    tmp_path: Path,
    mutate: Any,
) -> None:
    path, raw = _write_public_config(tmp_path)
    _rewrite_gate_checkpoint(raw, mutate)

    with pytest.raises(ValueError, match="Gate|gate|Oracle|oracle|SHA|schema|provenance"):
        project_cvoi_direct_epdms_run(
            load_cvoi_direct_epdms_config(path),
            forced_horizon=None,
        )


def test_public_loader_is_lexical_only_and_allows_authoritative_paths_to_be_absent(tmp_path: Path) -> None:
    path, raw = _write_public_config(tmp_path)
    for key in ("training_config_path", "encoder_checkpoint_path", "scenario_manifest_path"):
        Path(raw[key]).unlink()
    artifacts = raw["artifacts"]
    assert isinstance(artifacts, dict)
    for key, value in artifacts.items():
        if key != "gate_feature_mode":
            Path(value).unlink()

    assert load_cvoi_direct_epdms_config(path).branch == "full"


@pytest.mark.parametrize(
    "artifact_key",
    [
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
        "stop_checkpoint_path",
    ],
)
def test_public_loader_allows_selected_handoff_symlinks(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    path, raw = _write_public_config(tmp_path)
    artifacts = raw["artifacts"]
    assert isinstance(artifacts, dict)
    handoff = Path(artifacts[artifact_key])
    target = handoff.with_name(f"{handoff.stem}-target{handoff.suffix}")
    handoff.rename(target)
    handoff.symlink_to(target)

    assert load_cvoi_direct_epdms_config(path).branch == "full"


@pytest.mark.parametrize("artifact_key", ["oracle_path", "gate_checkpoint_path"])
@pytest.mark.parametrize("replacement", ["symlink", "directory"])
def test_controller_projection_requires_regular_non_symlink_gate_and_oracle(
    tmp_path: Path,
    artifact_key: str,
    replacement: str,
) -> None:
    path, raw = _write_public_config(tmp_path)
    artifacts = raw["artifacts"]
    assert isinstance(artifacts, dict)
    artifact = Path(artifacts[artifact_key])
    if replacement == "symlink":
        target = artifact.with_name(f"{artifact.stem}-target{artifact.suffix}")
        artifact.rename(target)
        artifact.symlink_to(target)
    else:
        artifact.unlink()
        artifact.mkdir()

    with pytest.raises(ValueError, match="Gate|gate|Oracle|oracle|regular|symlink"):
        project_cvoi_direct_epdms_run(
            load_cvoi_direct_epdms_config(path),
            forced_horizon=None,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("unknown", 1),
        lambda payload: payload.__setitem__("protocol", "pdms_v1_navtest"),
        lambda payload: payload.__setitem__("split", "navtrain"),
        lambda payload: payload.__setitem__("max_horizon", 3),
        lambda payload: payload.__setitem__("guidance_steps", 3),
        lambda payload: payload.__setitem__("training_config_path", "relative.yaml"),
        lambda payload: payload["artifacts"].__setitem__(
            "calibration_checkpoint_path",
            payload["artifacts"]["p1_planner_checkpoint_path"],
        ),
        lambda payload: payload["artifacts"].__setitem__("unexpected", "/absolute/unexpected.pt"),
    ],
)
def test_public_config_rejects_schema_path_and_artifact_drift(
    tmp_path: Path,
    mutate: Any,
) -> None:
    path, _ = _write_public_config(tmp_path, mutate=mutate)

    with pytest.raises(ValueError):
        load_cvoi_direct_epdms_config(path)


def test_public_config_resolves_the_repository_relative_p1_training_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _ = _write_public_config(
        tmp_path,
        mutate=lambda payload: payload.__setitem__("training_config_path", PORTABLE_TRAINING_CONFIG),
    )
    monkeypatch.chdir(tmp_path)

    parsed = load_cvoi_direct_epdms_config(path)

    assert parsed.training_config_path == (REPOSITORY_ROOT / PORTABLE_TRAINING_CONFIG).resolve()


def test_unedited_portable_placeholders_fail_before_projection_or_scorer(tmp_path: Path) -> None:
    output_root = (tmp_path / "must-not-be-created").resolve()

    def use_placeholders(payload: dict[str, object]) -> None:
        payload["training_config_path"] = PORTABLE_TRAINING_CONFIG
        payload["encoder_checkpoint_path"] = str(PORTABLE_E120_ENCODER_CHECKPOINT)
        payload["scenario_manifest_path"] = str(PORTABLE_NAVTEST_SCENARIO_MANIFEST)
        payload["output_root"] = str(output_root)
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, dict)
        artifacts.update(
            {
                "p0_planner_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/p0_selected.pt"),
                "calibration_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/calibration.pt"),
                "p1_planner_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/p1_selected.pt"),
                "stop_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/stop.pt"),
                "gate_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/gate.pt"),
                "oracle_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/oracle_full.sqlite3"),
            }
        )

    path, _ = _write_public_config(tmp_path, mutate=use_placeholders)
    scorer_called = False

    def forbidden_scorer(*args: object, **kwargs: object) -> object:
        nonlocal scorer_called
        scorer_called = True
        raise AssertionError("placeholder config reached the scorer")

    with pytest.raises(ValueError, match=r"unedited.*?/path/to|/path/to.*edit"):
        run_cvoi_direct_epdms(
            path,
            forced_horizon=None,
            subprocess_run=forbidden_scorer,
            environ={},
        )

    assert not scorer_called
    assert not output_root.exists()


def test_effective_projection_round_trips_exact_controller_schema(tmp_path: Path) -> None:
    _, _, projection = _project_full(tmp_path)
    output = (tmp_path / "effective.json").resolve()

    assert write_cvoi_direct_epdms_projection(projection, output) == output
    assert load_cvoi_direct_epdms_projection(output) == projection
    raw = json.loads(output.read_text(encoding="utf-8"))
    assert raw["schema"] == CVOI_DIRECT_EPDMS_EFFECTIVE_SCHEMA == "cvoi_direct_epdms_effective"
    assert raw["version"] == CVOI_DIRECT_EPDMS_EFFECTIVE_VERSION == 1
    assert set(raw) == EFFECTIVE_COMMON_KEYS | EFFECTIVE_CONTROLLER_KEYS


def test_effective_loader_rejects_public_unknown_and_mode_incompatible_payloads(tmp_path: Path) -> None:
    public_path, _, projection = _project_full(tmp_path)
    with pytest.raises(ValueError):
        load_cvoi_direct_epdms_projection(public_path)

    effective_path = (tmp_path / "effective.json").resolve()
    write_cvoi_direct_epdms_projection(projection, effective_path)
    raw = json.loads(effective_path.read_text(encoding="utf-8"))
    for mutation in ("unknown", "forced_horizon"):
        drifted = dict(raw)
        drifted[mutation] = 0
        path = (tmp_path / f"drifted-{mutation}.json").resolve()
        path.write_text(json.dumps(drifted), encoding="utf-8")
        with pytest.raises(ValueError):
            load_cvoi_direct_epdms_projection(path)


@pytest.mark.parametrize(
    "field_name",
    [
        "training_config_path",
        "encoder_checkpoint_path",
        "scenario_manifest_path",
        "output_directory",
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
        "stop_checkpoint_path",
        "gate_checkpoint_path",
    ],
)
def test_effective_loader_rejects_every_relative_controller_path(
    tmp_path: Path,
    field_name: str,
) -> None:
    _, _, projection = _project_full(tmp_path)
    effective_path = (tmp_path / "effective.json").resolve()
    write_cvoi_direct_epdms_projection(projection, effective_path)
    raw = json.loads(effective_path.read_text(encoding="utf-8"))
    raw[field_name] = f"relative/{field_name}"
    drifted_path = (tmp_path / "relative.json").resolve()
    drifted_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="absolute|path"):
        load_cvoi_direct_epdms_projection(drifted_path)


@pytest.mark.parametrize(
    "mutation",
    [
        {"branch": "other"},
        {"horizon": 0},
        {"guidance_steps": 0},
        {"oracle_sha256": "A" * 64},
        {"gate_feature_mode": "other"},
    ],
)
def test_effective_loader_rejects_controller_identity_drift(
    tmp_path: Path,
    mutation: dict[str, object],
) -> None:
    _, _, projection = _project_full(tmp_path)
    effective_path = (tmp_path / "effective.json").resolve()
    write_cvoi_direct_epdms_projection(projection, effective_path)
    raw = json.loads(effective_path.read_text(encoding="utf-8"))
    raw.update(mutation)
    drifted_path = (tmp_path / "drifted.json").resolve()
    drifted_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValueError):
        load_cvoi_direct_epdms_projection(drifted_path)


def test_effective_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    _, _, projection = _project_full(tmp_path)
    effective_path = (tmp_path / "effective.json").resolve()
    write_cvoi_direct_epdms_projection(projection, effective_path)
    text = effective_path.read_text(encoding="utf-8")
    duplicate_path = (tmp_path / "duplicate.json").resolve()
    duplicate_path.write_text(text.replace("{", '{"version":1,', 1), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        load_cvoi_direct_epdms_projection(duplicate_path)


def test_public_and_effective_loaders_read_through_pinned_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public_path, _, projection = _project_full(tmp_path)
    effective_path = (tmp_path / "effective.json").resolve()
    write_cvoi_direct_epdms_projection(projection, effective_path)

    def forbidden_path_read(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("config loader reopened a validated path")

    monkeypatch.setattr(Path, "read_text", forbidden_path_read)

    assert load_cvoi_direct_epdms_config(public_path).branch == "full"
    assert load_cvoi_direct_epdms_projection(effective_path) == projection


def test_effective_writer_requires_absolute_new_target(tmp_path: Path) -> None:
    _, _, projection = _project_full(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        write_cvoi_direct_epdms_projection(projection, Path("relative.json"))

    output = (tmp_path / "effective.json").resolve()
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        write_cvoi_direct_epdms_projection(projection, output)

    missing_parent = (tmp_path / "missing/effective.json").resolve()
    with pytest.raises((ValueError, FileNotFoundError), match="parent|directory|exist"):
        write_cvoi_direct_epdms_projection(projection, missing_parent)


def _assert_repository_epdms_inventory() -> None:
    assert REPOSITORY_EPDMS_CONFIG_ROOT.is_dir()
    entries = tuple(REPOSITORY_EPDMS_CONFIG_ROOT.iterdir())
    assert {entry.name for entry in entries} == REPOSITORY_EPDMS_ENTRIES
    assert all(entry.is_file() for entry in entries)


def _repository_payload() -> dict[str, object]:
    _assert_repository_epdms_inventory()
    path = (REPOSITORY_EPDMS_CONFIG_ROOT / "full_controller.yaml").resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    parsed = load_cvoi_direct_epdms_config(path)
    assert parsed.branch == "full"
    assert parsed.run_kind == "controller"
    return payload


def _nested_mapping_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            keys.append(key)
            keys.extend(_nested_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_nested_mapping_keys(nested))
    return tuple(keys)


def test_repository_direct_epdms_directory_is_one_yaml_and_one_readme() -> None:
    _assert_repository_epdms_inventory()


def test_repository_full_epdms_yaml_is_flat_and_has_no_control_plane_surface() -> None:
    payload = _repository_payload()

    assert set(payload) == PUBLIC_KEYS
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    assert set(artifacts) == CONTROLLER_ARTIFACT_KEYS
    for key in _nested_mapping_keys(payload):
        key_tokens = set(key.lower().replace("-", "_").split("_"))
        assert not key_tokens & FORBIDDEN_REPOSITORY_CONTROL_KEY_TOKENS

    text = (REPOSITORY_EPDMS_CONFIG_ROOT / "full_controller.yaml").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "<<:" not in text
    assert not re.search(r"(?:^|[\s:\[\]{},])[&*][A-Za-z_][A-Za-z0-9_.-]*", text)
    assert not re.search(r"\$\{|!env\b|\bos\.environ\b|\benv\s*\(", lowered)
    assert "generated" not in lowered
    assert "world4drive" not in lowered
    assert "nuscenes" not in lowered


def test_repository_full_epdms_yaml_uses_portable_navsim_e120_paths() -> None:
    payload = _repository_payload()

    assert payload["schema"] == "cvoi_direct_epdms"
    assert payload["version"] == 1
    assert payload["branch"] == "full"
    assert payload["run_kind"] == "controller"
    assert payload["split"] == "navtest"
    assert payload["protocol"] == "epdms_v2_one_stage_navtest"
    assert payload["max_horizon"] == 4
    assert payload["guidance_steps"] == 2
    assert payload["training_config_path"] == PORTABLE_TRAINING_CONFIG
    assert payload["encoder_checkpoint_path"] == str(PORTABLE_E120_ENCODER_CHECKPOINT)
    assert payload["scenario_manifest_path"] == str(PORTABLE_NAVTEST_SCENARIO_MANIFEST)
    assert payload["output_root"] == str(PORTABLE_EPDMS_RESULTS_ROOT / "full")
    assert payload["artifacts"] == {
        "p0_planner_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/p0_selected.pt"),
        "calibration_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/calibration.pt"),
        "p1_planner_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/p1_selected.pt"),
        "stop_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/stop.pt"),
        "gate_checkpoint_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/gate.pt"),
        "oracle_path": str(PORTABLE_FULL_RESULTS_ROOT / "handoff/oracle_full.sqlite3"),
        "gate_feature_mode": "full",
    }


def test_repository_direct_epdms_readme_lists_only_the_full_controller() -> None:
    _assert_repository_epdms_inventory()
    text = (REPOSITORY_EPDMS_CONFIG_ROOT / "README.md").read_text(encoding="utf-8")

    assert "tools/run_cvoi_direct_epdms.py" in text
    assert "full_controller.yaml" in text
    assert not re.search(r"run_cvoi_direct_epdms\.py[^\n]*--horizon", text)
