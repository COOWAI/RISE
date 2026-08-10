"""One-run CLI contract for direct NavTest CVoI EPDMS evaluation."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from app.vjepa_cowa_world_model.evaluation import cvoi_direct_epdms as direct
from app.vjepa_cowa_world_model.evaluation.cvoi_direct_epdms import (
    CvoiDirectEpdmsArtifacts,
    CvoiDirectEpdmsConfig,
    CvoiDirectEpdmsProjection,
)
from tools import run_cvoi_direct_epdms as cli


def _poison(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("forbidden side effect")


def _config(tmp_path: Path, *, run_kind: str) -> CvoiDirectEpdmsConfig:
    branch = "full" if run_kind == "controller" else "hazard_only"
    inputs = (tmp_path / "inputs").resolve()
    return CvoiDirectEpdmsConfig(
        schema="cvoi_direct_epdms",
        version=1,
        branch=branch,
        run_kind=run_kind,
        split="navtest",
        protocol="epdms_v2_one_stage_navtest",
        max_horizon=4,
        guidance_steps=2,
        training_config_path=inputs / "training.yaml",
        encoder_checkpoint_path=inputs / "encoder.pt",
        scenario_manifest_path=inputs / "scenario_manifest.jsonl",
        output_root=(tmp_path / "output" / branch).resolve(),
        artifacts=CvoiDirectEpdmsArtifacts(
            p0_planner_checkpoint_path=inputs / "p0_selected.pt",
            calibration_checkpoint_path=inputs / "calibration.pt",
            p1_planner_checkpoint_path=inputs / "p1_selected.pt",
            stop_checkpoint_path=inputs / "stop.pt" if run_kind == "controller" else None,
            gate_checkpoint_path=inputs / "gate.pt" if run_kind == "controller" else None,
            oracle_path=inputs / "oracle.sqlite3" if run_kind == "controller" else None,
            gate_feature_mode="full" if run_kind == "controller" else None,
        ),
    )


def _projection(
    config: CvoiDirectEpdmsConfig,
    *,
    forced_horizon: int | None,
    output_directory: Path | None = None,
    oracle_sha256: str | None = None,
) -> CvoiDirectEpdmsProjection:
    if config.run_kind == "controller":
        return CvoiDirectEpdmsProjection(
            branch=config.branch,
            split=config.split,
            protocol=config.protocol,
            evaluation_mode="controller",
            horizon=None,
            guidance_steps=2,
            training_config_path=config.training_config_path,
            encoder_checkpoint_path=config.encoder_checkpoint_path,
            scenario_manifest_path=config.scenario_manifest_path,
            output_directory=output_directory or config.output_root,
            p0_planner_checkpoint_path=config.artifacts.p0_planner_checkpoint_path,
            calibration_checkpoint_path=config.artifacts.calibration_checkpoint_path,
            p1_planner_checkpoint_path=config.artifacts.p1_planner_checkpoint_path,
            stop_checkpoint_path=config.artifacts.stop_checkpoint_path,
            gate_checkpoint_path=config.artifacts.gate_checkpoint_path,
            gate_feature_mode=config.artifacts.gate_feature_mode,
            oracle_sha256=oracle_sha256 or "a" * 64,
        )
    if forced_horizon == 0:
        return CvoiDirectEpdmsProjection(
            branch=config.branch,
            split=config.split,
            protocol=config.protocol,
            evaluation_mode="p0_forced",
            horizon=0,
            guidance_steps=0,
            training_config_path=config.training_config_path,
            encoder_checkpoint_path=config.encoder_checkpoint_path,
            scenario_manifest_path=config.scenario_manifest_path,
            output_directory=output_directory or config.output_root / "h0",
            p0_planner_checkpoint_path=config.artifacts.p0_planner_checkpoint_path,
        )
    return CvoiDirectEpdmsProjection(
        branch=config.branch,
        split=config.split,
        protocol=config.protocol,
        evaluation_mode="p1_field_forced",
        horizon=forced_horizon,
        guidance_steps=2,
        training_config_path=config.training_config_path,
        encoder_checkpoint_path=config.encoder_checkpoint_path,
        scenario_manifest_path=config.scenario_manifest_path,
        output_directory=output_directory or config.output_root / f"h{forced_horizon}",
        calibration_checkpoint_path=config.artifacts.calibration_checkpoint_path,
        p1_planner_checkpoint_path=config.artifacts.p1_planner_checkpoint_path,
    )


def _write_fake_effective(projection: CvoiDirectEpdmsProjection, path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "cvoi_direct_epdms_effective",
                "version": 1,
                "branch": projection.branch,
            }
        ),
        encoding="utf-8",
    )
    return path


def _install_lightweight_preflight_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mismatch: str | None = None,
    expected_encoder_embed_dim: int = 4,
) -> list[str]:
    events: list[str] = []
    paths = {
        "p0_planner_checkpoint_path": Path("/artifacts/p0.pt"),
        "calibration_checkpoint_path": Path("/artifacts/calibration.pt"),
        "p1_planner_checkpoint_path": Path("/artifacts/p1.pt"),
        "stop_checkpoint_path": Path("/artifacts/stop.pt"),
        "gate_checkpoint_path": Path("/artifacts/gate.pt"),
    }
    role_shapes = {
        "encoder": {"weight": [1]},
        "predictor": {"weight": [1]},
        "planner": {"weight": [1]},
    }
    p1_role_shapes = role_shapes
    if mismatch == "role_shapes":
        p1_role_shapes = {
            **role_shapes,
            "planner": {"weight": [2]},
        }

    class EncoderHandle:
        def __enter__(self) -> "EncoderHandle":
            events.append("encoder-enter")
            return self

        def read(self, size: int) -> bytes:
            assert size == 1
            events.append("encoder-read")
            return b"x"

        def __exit__(self, *_args: object) -> None:
            events.append("encoder-close")

    def open_encoder(path: Path, *, context: str) -> EncoderHandle:
        assert context == "direct EPDMS encoder checkpoint"
        events.append(f"encoder-open:{path}")
        return EncoderHandle()

    def resolve_planner(path: Path, *, results_root: Path, stage: str) -> Path:
        assert results_root.is_absolute()
        events.append(f"resolve-{stage}:{path}")
        return path

    def read_planner(path: Path, *, expected_stage: str, expected_branch_id: str) -> dict[str, object]:
        events.append(f"planner-{expected_stage}:{expected_branch_id}")
        return {
            "role_state_shapes": role_shapes if expected_stage == "p0" else p1_role_shapes,
            "encoder": expected_stage,
        }

    def state_sha256(state: object, *, role: str) -> str:
        events.append(f"hash:{role}")
        if mismatch == "encoder" and role == "P1 encoder":
            return "b" * 64
        return "a" * 64

    def read_value(
        path: Path,
        *,
        required_phase: str,
        required_branch_id: str,
        map_location: str,
    ) -> dict[str, object]:
        assert map_location == "cpu"
        events.append(f"value-{required_phase}:{required_branch_id}:{path}")
        return {"architecture": {"embed_dim": 4}}

    def read_gate(
        path: Path,
        *,
        branch: str,
        oracle_sha256: str,
        gate_feature_mode: str,
    ) -> dict[str, object]:
        assert oracle_sha256 == "a" * 64
        events.append(f"gate:{branch}:{gate_feature_mode}:{path}")
        return {"latent_dim": 4}

    def artifact_paths(projection: CvoiDirectEpdmsProjection) -> dict[str, Path]:
        events.append("artifact-paths")
        return {name: path for name, path in paths.items() if getattr(projection, name) is not None}

    monkeypatch.setattr(
        direct,
        "_preflight_direct_training_config",
        lambda projection, *, expected_p1_branch_id: (
            events.append(f"training:{expected_p1_branch_id}") or expected_encoder_embed_dim
        ),
    )
    monkeypatch.setattr(direct, "_open_regular_nonsymlink_binary", open_encoder)
    monkeypatch.setattr(
        direct,
        "read_cvoi_direct_epdms_scenario_manifest",
        lambda path: events.append(f"manifest:{path}") or {},
    )
    monkeypatch.setattr(
        direct,
        "_direct_projection_artifact_paths",
        artifact_paths,
    )
    monkeypatch.setattr(direct, "resolve_formal_v2_navsim_e120_selected_checkpoint", resolve_planner)
    monkeypatch.setattr(direct, "read_cvoi_direct_epdms_planner_checkpoint", read_planner)
    monkeypatch.setattr(direct, "_direct_state_sha256", state_sha256)
    monkeypatch.setattr(direct, "read_cvoi_navsim_e120_direct_value_checkpoint", read_value)
    monkeypatch.setattr(direct, "read_cvoi_direct_epdms_gate_checkpoint", read_gate)
    return events


def test_parser_exposes_only_the_full_controller_config() -> None:
    parser = cli.build_parser()
    actions = {action.dest: action for action in parser._actions}

    assert set(actions) == {"help", "config"}
    assert actions["config"].required is True
    assert {option for action in parser._actions for option in action.option_strings} == {
        "-h",
        "--help",
        "--config",
    }


def test_help_has_no_run_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_cvoi_direct_epdms", _poison)

    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])

    assert raised.value.code == 0


def test_cli_delegates_exactly_one_absolute_full_controller_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = (tmp_path / "direct.yaml").resolve()
    calls: list[tuple[Path, int | None]] = []

    def fake_run(path: Path, *, forced_horizon: int | None) -> int:
        calls.append((path, forced_horizon))
        return 0

    monkeypatch.setattr(cli, "run_cvoi_direct_epdms", fake_run)
    arguments = ["--config", str(config_path)]

    assert cli.main(arguments) == 0
    assert calls == [(config_path, None)]


@pytest.mark.parametrize(
    "arguments",
    [
        ["--config", "/absolute/config.yaml", "--horizon", "0"],
        ["--config", "/absolute/config.yaml", "--horizon", "-1"],
        ["--config", "/absolute/config.yaml", "--horizon", "5"],
        ["--config", "/absolute/config.yaml", "--horizons", "0,1,2,3,4"],
        ["--config", "/absolute/config.yaml", "--retry", "3"],
        ["--config", "/absolute/config.yaml", "--registry", "/registry.json"],
        ["--config", "/absolute/config.yaml", "--artifact-bundle", "/bundle.json"],
        ["--config", "/absolute/config.yaml", "--cwd", "/repo"],
    ],
)
def test_cli_rejects_horizon_loop_retry_registry_bundle_and_cwd_surfaces(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    monkeypatch.setattr(cli, "run_cvoi_direct_epdms", _poison)

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)

    assert raised.value.code == 2


def test_runner_api_has_no_loop_retry_registry_bundle_or_cwd_controls() -> None:
    parameters = inspect.signature(direct.run_cvoi_direct_epdms).parameters

    assert set(parameters) == {"config_path", "forced_horizon", "subprocess_run", "environ"}
    assert parameters["forced_horizon"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["subprocess_run"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["environ"].kind is inspect.Parameter.KEYWORD_ONLY


def test_runner_exposes_one_shared_projection_preflight_boundary() -> None:
    preflight = direct.preflight_cvoi_direct_epdms_projection
    parameters = inspect.signature(preflight).parameters

    assert set(parameters) == {"projection"}


def test_shared_projection_preflight_covers_every_controller_input_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_kind="controller")
    projection = _projection(config, forced_horizon=None, oracle_sha256="a" * 64)
    events = _install_lightweight_preflight_fakes(monkeypatch)

    direct.preflight_cvoi_direct_epdms_projection(projection)

    assert events == [
        "training:p1_full",
        f"encoder-open:{projection.encoder_checkpoint_path}",
        "encoder-enter",
        "encoder-read",
        "encoder-close",
        f"manifest:{projection.scenario_manifest_path}",
        "artifact-paths",
        "resolve-p0:/artifacts/p0.pt",
        "planner-p0:p0_uniform",
        "hash:P0 encoder",
        "resolve-p1:/artifacts/p1.pt",
        "planner-p1:p1_full",
        "hash:P1 encoder",
        "value-field_calibrated:calibration_full:/artifacts/calibration.pt",
        "value-stop_calibrated:stop_full:/artifacts/stop.pt",
        "gate:full:full:/artifacts/gate.pt",
    ]


def test_training_preflight_resolves_the_canonical_encoder_embed_dim(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, run_kind="controller")
    projection = replace(
        _projection(config, forced_horizon=None, oracle_sha256="a" * 64),
        training_config_path=(
            Path(direct.__file__).resolve().parents[3] / "configs/train/navsim/cvoi_manual_full/05_p1_full.yaml"
        ).resolve(),
    )

    assert (
        direct._preflight_direct_training_config(
            projection,
            expected_p1_branch_id="p1_full",
        )
        == 1024
    )


@pytest.mark.parametrize(
    ("run_kind", "horizon"),
    [
        ("controller", None),
    ],
)
def test_shared_projection_preflight_binds_value_embed_dim_to_the_canonical_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_kind: str,
    horizon: int | None,
) -> None:
    config = _config(tmp_path, run_kind=run_kind)
    projection = _projection(config, forced_horizon=horizon, oracle_sha256="a" * 64)
    events = _install_lightweight_preflight_fakes(monkeypatch, expected_encoder_embed_dim=1024)

    with pytest.raises(ValueError, match=r"embed_dim.*encoder|encoder.*embed_dim"):
        direct.preflight_cvoi_direct_epdms_projection(projection)

    assert any(event.startswith("value-field_calibrated:") for event in events)
    assert not any(event.startswith("gate:") for event in events)


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("role_shapes", "architectures"),
        ("encoder", "encoder states"),
    ],
)
def test_shared_projection_preflight_rejects_p0_p1_model_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
    message: str,
) -> None:
    config = _config(tmp_path, run_kind="controller")
    projection = _projection(config, forced_horizon=None, oracle_sha256="a" * 64)
    events = _install_lightweight_preflight_fakes(monkeypatch, mismatch=mismatch)

    with pytest.raises(ValueError, match=message):
        direct.preflight_cvoi_direct_epdms_projection(projection)

    assert "planner-p0:p0_uniform" in events
    assert "planner-p1:p1_full" in events
    assert not any(event.startswith("value-") for event in events)
    assert not any(event.startswith("gate:") for event in events)


@pytest.mark.parametrize("horizon", range(5))
def test_cli_rejects_every_forced_horizon_before_loading_the_config(
    monkeypatch: pytest.MonkeyPatch,
    horizon: int,
) -> None:
    monkeypatch.setattr(cli, "run_cvoi_direct_epdms", _poison)

    with pytest.raises(SystemExit) as raised:
        cli.main(["--config", "/absolute/config.yaml", "--horizon", str(horizon)])

    assert raised.value.code == 2


def test_one_run_creates_output_once_and_invokes_one_clean_explicit_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_kind="controller")
    output_directory = (tmp_path / "selected-controller-output").resolve()
    projection = _projection(config, forced_horizon=None, output_directory=output_directory)
    config_path = (tmp_path / "controller.yaml").resolve()
    effective_path = output_directory / "cvoi_direct_epdms_effective.json"
    scorer_output = output_directory / "scorer_output"
    policy_traces = output_directory / "policy_traces"
    events: list[str] = []
    projection_calls: list[int | None] = []
    subprocess_calls: list[tuple[list[str], dict[str, Any]]] = []
    aggregate_calls: list[Path] = []

    def fake_load(path: Path) -> CvoiDirectEpdmsConfig:
        assert path == config_path
        events.append("load")
        return config

    def fake_project(
        config_value: CvoiDirectEpdmsConfig,
        *,
        forced_horizon: int | None,
    ) -> CvoiDirectEpdmsProjection:
        assert config_value is config
        projection_calls.append(forced_horizon)
        events.append("project")
        return projection

    def fake_preflight(projection_value: CvoiDirectEpdmsProjection) -> None:
        assert projection_value is projection
        assert not output_directory.exists()
        events.append("preflight")

    def fake_write(projection_value: CvoiDirectEpdmsProjection, path: Path) -> Path:
        assert projection_value is projection
        assert path == effective_path
        assert path.is_absolute()
        assert output_directory.is_dir()
        events.append("write")
        return _write_fake_effective(projection_value, path)

    def fake_subprocess_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert output_directory.is_dir()
        assert effective_path.is_file()
        assert scorer_output.is_dir()
        assert policy_traces.is_dir()
        events.append("subprocess")
        subprocess_calls.append((list(command), dict(kwargs)))
        return subprocess.CompletedProcess(command, 0)

    def fake_aggregate(path: Path) -> tuple[()]:
        events.append("aggregate")
        aggregate_calls.append(path)
        return ()

    monkeypatch.setattr(direct, "load_cvoi_direct_epdms_config", fake_load)
    monkeypatch.setattr(direct, "project_cvoi_direct_epdms_run", fake_project)
    monkeypatch.setattr(
        direct,
        "preflight_cvoi_direct_epdms_projection",
        fake_preflight,
        raising=False,
    )
    monkeypatch.setattr(direct, "write_cvoi_direct_epdms_projection", fake_write)
    monkeypatch.setattr(direct, "aggregate_direct_epdms_results", fake_aggregate)

    real_os_mkdir = direct.os.mkdir
    real_path_mkdir = Path.mkdir
    root_creation_transitions: list[Path] = []
    tracking_depth = 0

    def tracked_os_mkdir(
        path: os.PathLike[str] | str,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal tracking_depth
        candidate = Path(path)
        outermost = tracking_depth == 0
        absent_before = outermost and not os.path.lexists(candidate)
        tracking_depth += 1
        try:
            if dir_fd is None:
                real_os_mkdir(path, mode)
            else:
                real_os_mkdir(path, mode, dir_fd=dir_fd)
        finally:
            tracking_depth -= 1
        if outermost and candidate == output_directory and absent_before and candidate.is_dir():
            root_creation_transitions.append(candidate)
            events.append("mkdir-root")

    def tracked_path_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        nonlocal tracking_depth
        candidate = Path(path)
        outermost = tracking_depth == 0
        absent_before = outermost and not os.path.lexists(candidate)
        tracking_depth += 1
        try:
            real_path_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
        finally:
            tracking_depth -= 1
        if outermost and candidate == output_directory and absent_before and candidate.is_dir():
            root_creation_transitions.append(candidate)
            events.append("mkdir-root")

    monkeypatch.setattr(direct.os, "mkdir", tracked_os_mkdir)
    monkeypatch.setattr(Path, "mkdir", tracked_path_mkdir)
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    monkeypatch.chdir(foreign_cwd)
    supplied_environment = {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/explicit/project",
        "CUDA_VISIBLE_DEVICES": "7",
        "LANG": "C.UTF-8",
        "NAVSIM_OUTPUT_DIR": "/stale/output",
        "CVOI_DIRECT_EPDMS": "0",
        "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH": "/stale/effective.json",
        "CVOI_OFFICIAL_POLICY_ID": "legacy-policy",
        "CVOI_OFFICIAL_ARTIFACT_BUNDLE_PATH": "/legacy/bundle.json",
        "CVOI_POLICY_REGISTRY_PATH": "/legacy/registry.json",
        "CVOI_RETRY_COUNT": "9",
        "CVOI_MANUAL_NAVTRAIN_GATE": "1",
    }

    returncode = direct.run_cvoi_direct_epdms(
        config_path,
        forced_horizon=None,
        subprocess_run=fake_subprocess_run,
        environ=supplied_environment,
    )

    assert returncode == 0
    assert projection_calls == [None]
    assert root_creation_transitions == [output_directory]
    assert events[:4] == ["load", "project", "preflight", "mkdir-root"]
    assert events.index("mkdir-root") < events.index("subprocess")
    assert events.index("write") < events.index("subprocess") < events.index("aggregate")
    assert scorer_output.is_dir()
    assert policy_traces.is_dir()
    assert aggregate_calls == [output_directory]
    assert len(subprocess_calls) == 1
    command, options = subprocess_calls[0]
    expected_script = Path(direct.__file__).resolve().parents[3] / "scripts/eval_navsim/eval_navsim_v2_pdms.sh"
    script_arguments = [Path(value) for value in command if value.endswith("eval_navsim_v2_pdms.sh")]
    assert script_arguments == [expected_script]
    assert expected_script.is_absolute()
    assert command.count(str(projection.encoder_checkpoint_path)) == 1
    assert command.count(str(projection.training_config_path)) == 1
    assert str(config_path) not in command
    assert "cwd" not in options
    assert options["check"] is False
    assert options["env"] == {
        "PATH": "/usr/bin",
        "PYTHONPATH": "/explicit/project",
        "CUDA_VISIBLE_DEVICES": "7",
        "LANG": "C.UTF-8",
        "NAVSIM_OUTPUT_DIR": str(scorer_output),
        "CVOI_DIRECT_EPDMS": "1",
        "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH": str(effective_path),
    }
    direct_payload = {key: value for key, value in options["env"].items() if key.startswith("CVOI_DIRECT_EPDMS_")}
    assert direct_payload == {"CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH": str(effective_path)}
    forbidden_text = " ".join([*command, *options["env"].keys()]).lower()
    assert "registry" not in forbidden_text
    assert "bundle" not in forbidden_text
    assert "retry" not in forbidden_text


def test_preexisting_selected_output_fails_before_projection_write_or_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_kind="controller")
    output_directory = (tmp_path / "already-exists").resolve()
    output_directory.mkdir()
    marker = output_directory / "operator-owned.txt"
    marker.write_text("keep", encoding="utf-8")
    projection = _projection(config, forced_horizon=None, output_directory=output_directory)
    config_path = (tmp_path / "controller.yaml").resolve()
    events: list[str] = []
    monkeypatch.setattr(
        direct,
        "load_cvoi_direct_epdms_config",
        lambda path: events.append("load") or config,
    )
    monkeypatch.setattr(
        direct,
        "project_cvoi_direct_epdms_run",
        lambda config_value, *, forced_horizon: events.append("project") or projection,
    )
    monkeypatch.setattr(
        direct,
        "preflight_cvoi_direct_epdms_projection",
        lambda projection_value: None,
        raising=False,
    )
    monkeypatch.setattr(direct, "write_cvoi_direct_epdms_projection", _poison)
    monkeypatch.setattr(direct, "aggregate_direct_epdms_results", _poison)

    with pytest.raises(FileExistsError):
        direct.run_cvoi_direct_epdms(
            config_path,
            forced_horizon=None,
            subprocess_run=_poison,
            environ={"PATH": "/usr/bin"},
        )

    assert events == ["load", "project"]
    assert marker.read_text(encoding="utf-8") == "keep"
    assert set(output_directory.iterdir()) == {marker}


def test_projection_preflight_failure_happens_before_any_output_or_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_kind="controller")
    output_directory = (tmp_path / "preflight-must-not-create-output").resolve()
    projection = _projection(config, forced_horizon=None, output_directory=output_directory)
    config_path = (tmp_path / "controller.yaml").resolve()
    events: list[str] = []

    monkeypatch.setattr(
        direct,
        "load_cvoi_direct_epdms_config",
        lambda path: events.append("load") or config,
    )
    monkeypatch.setattr(
        direct,
        "project_cvoi_direct_epdms_run",
        lambda config_value, *, forced_horizon: events.append("project") or projection,
    )

    def reject_projection(projection_value: CvoiDirectEpdmsProjection) -> None:
        assert projection_value is projection
        events.append("preflight")
        raise ValueError("invalid direct EPDMS artifact identity")

    monkeypatch.setattr(
        direct,
        "preflight_cvoi_direct_epdms_projection",
        reject_projection,
        raising=False,
    )
    monkeypatch.setattr(direct, "write_cvoi_direct_epdms_projection", _poison)
    monkeypatch.setattr(direct, "aggregate_direct_epdms_results", _poison)

    with pytest.raises(ValueError, match="artifact identity"):
        direct.run_cvoi_direct_epdms(
            config_path,
            forced_horizon=None,
            subprocess_run=_poison,
            environ={"PATH": "/usr/bin"},
        )

    assert events == ["load", "project", "preflight"]
    assert not output_directory.exists()


def test_scorer_failure_code_is_propagated_once_without_partial_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_kind="controller")
    output_directory = (tmp_path / "failed-controller").resolve()
    scorer_output = output_directory / "scorer_output"
    policy_traces = output_directory / "policy_traces"
    projection = _projection(config, forced_horizon=None, output_directory=output_directory)
    config_path = (tmp_path / "controller.yaml").resolve()
    subprocess_calls: list[list[str]] = []
    monkeypatch.setattr(direct, "load_cvoi_direct_epdms_config", lambda path: config)
    monkeypatch.setattr(
        direct,
        "project_cvoi_direct_epdms_run",
        lambda config_value, *, forced_horizon: projection,
    )
    monkeypatch.setattr(
        direct,
        "preflight_cvoi_direct_epdms_projection",
        lambda projection_value: None,
        raising=False,
    )
    monkeypatch.setattr(direct, "write_cvoi_direct_epdms_projection", _write_fake_effective)
    monkeypatch.setattr(direct, "aggregate_direct_epdms_results", _poison)

    def failing_subprocess(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert scorer_output.is_dir()
        assert policy_traces.is_dir()
        subprocess_calls.append(list(command))
        return subprocess.CompletedProcess(command, 23)

    returncode = direct.run_cvoi_direct_epdms(
        config_path,
        forced_horizon=None,
        subprocess_run=failing_subprocess,
        environ={"PATH": "/usr/bin"},
    )

    assert returncode == 23
    assert len(subprocess_calls) == 1
    assert (output_directory / "cvoi_direct_epdms_effective.json").is_file()
    assert scorer_output.is_dir()
    assert policy_traces.is_dir()
    assert not (output_directory / "records.jsonl").exists()
    assert not (output_directory / "summary.json").exists()


def test_controller_oracle_is_hashed_and_closed_before_projection_write_and_agent_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, run_kind="controller")
    output_directory = (tmp_path / "controller-output").resolve()
    scorer_output = output_directory / "scorer_output"
    policy_traces = output_directory / "policy_traces"
    config_path = (tmp_path / "controller.yaml").resolve()
    oracle_path = (tmp_path / "oracle.sqlite3").resolve()
    oracle_path.write_bytes(b"controller-only-offline-oracle")
    events: list[str] = []
    state: dict[str, Any] = {}

    def fake_project(
        config_value: CvoiDirectEpdmsConfig,
        *,
        forced_horizon: int | None,
    ) -> CvoiDirectEpdmsProjection:
        assert config_value is config
        assert forced_horizon is None
        handle = oracle_path.open("rb")
        state["oracle_handle"] = handle
        events.append("oracle-open")
        try:
            digest = hashlib.sha256(handle.read()).hexdigest()
            events.append("oracle-hashed")
        finally:
            handle.close()
            events.append("oracle-closed")
        state["oracle_sha256"] = digest
        return _projection(
            config,
            forced_horizon=None,
            output_directory=output_directory,
            oracle_sha256=digest,
        )

    def fake_write(projection_value: CvoiDirectEpdmsProjection, path: Path) -> Path:
        assert state["oracle_handle"].closed
        assert projection_value.oracle_sha256 == state["oracle_sha256"]
        assert not hasattr(projection_value, "oracle_path")
        events.append("projection-write")
        return _write_fake_effective(projection_value, path)

    def fake_preflight(projection_value: CvoiDirectEpdmsProjection) -> None:
        assert state["oracle_handle"].closed
        assert projection_value.oracle_sha256 == state["oracle_sha256"]
        events.append("preflight")

    def fake_subprocess(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert state["oracle_handle"].closed
        assert scorer_output.is_dir()
        assert policy_traces.is_dir()
        events.append("agent-subprocess")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(direct, "load_cvoi_direct_epdms_config", lambda path: config)
    monkeypatch.setattr(direct, "project_cvoi_direct_epdms_run", fake_project)
    monkeypatch.setattr(
        direct,
        "preflight_cvoi_direct_epdms_projection",
        fake_preflight,
        raising=False,
    )
    monkeypatch.setattr(direct, "write_cvoi_direct_epdms_projection", fake_write)
    monkeypatch.setattr(
        direct,
        "aggregate_direct_epdms_results",
        lambda path: events.append("aggregate") or (),
    )

    assert (
        direct.run_cvoi_direct_epdms(
            config_path,
            forced_horizon=None,
            subprocess_run=fake_subprocess,
            environ={"PATH": "/usr/bin"},
        )
        == 0
    )
    assert events == [
        "oracle-open",
        "oracle-hashed",
        "oracle-closed",
        "preflight",
        "projection-write",
        "agent-subprocess",
        "aggregate",
    ]


def test_relative_config_is_not_resolved_from_the_current_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    foreign_cwd = tmp_path / "foreign-cwd"
    foreign_cwd.mkdir()
    (foreign_cwd / "relative.yaml").write_text("schema: cvoi_direct_epdms\n", encoding="utf-8")
    monkeypatch.chdir(foreign_cwd)
    monkeypatch.setattr(direct, "project_cvoi_direct_epdms_run", _poison)
    monkeypatch.setattr(direct, "write_cvoi_direct_epdms_projection", _poison)
    monkeypatch.setattr(direct, "aggregate_direct_epdms_results", _poison)

    with pytest.raises(ValueError, match="absolute"):
        direct.run_cvoi_direct_epdms(
            Path("relative.yaml"),
            forced_horizon=None,
            subprocess_run=_poison,
            environ={"PATH": "/usr/bin"},
        )
