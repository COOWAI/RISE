"""Executable shell-boundary tests for one-run direct NavSim EPDMS."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/eval_navsim/eval_navsim_v2_pdms.sh"

MANUAL_VARIABLES = (
    "CVOI_MANUAL_NAVTRAIN_GATE",
    "CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH",
)
REQUIRED_NAVSIM_ROOTS = (
    "OPENSCENE_DATA_ROOT",
    "NAVSIM_EXP_ROOT",
    "NUPLAN_MAPS_ROOT",
    "NAVSIM_DEVKIT_ROOT",
)


def _install_python_recorder(tmp_path: Path) -> tuple[Path, Path]:
    recorder = tmp_path / "python-recorder.sh"
    call_log = tmp_path / "python-calls.tsv"
    recorder.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "{ printf 'CALL'; printf '\\t%s' \"$@\"; printf '\\n'; } >> \"${FAKE_PYTHON_CALL_LOG:?}\"\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)
    return recorder, call_log


def _base_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    recorder, call_log = _install_python_recorder(tmp_path)
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(
            (
                "CVOI_DIRECT_EPDMS",
                "CVOI_MANUAL_NAVTRAIN_GATE",
                "CVOI_FORMAL_V2_NAVSIM_E120",
            )
        ):
            environment.pop(name)
    environment.update(
        {
            "PYTHON_BIN": str(recorder),
            "FAKE_PYTHON_CALL_LOG": str(call_log),
            "OPENSCENE_DATA_ROOT": str(tmp_path / "openscene"),
            "NAVSIM_EXP_ROOT": str(tmp_path / "navsim-exp"),
            "NUPLAN_MAPS_ROOT": str(tmp_path / "maps"),
            "NAVSIM_DEVKIT_ROOT": str(tmp_path / "navsim-v2"),
            "PYTHONPATH": "",
            "METRIC_CACHE_PATH": str(tmp_path / "metric-cache"),
            "MAX_WORKERS": "37",
            "USE_PROCESS_POOL": "true",
            "FORWARD_MODE": "stage3",
            "PROPOSAL_CHECKPOINT": "/must/not/leak.pt",
        }
    )
    return environment, call_log


def _direct_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    environment, call_log = _base_environment(tmp_path)
    effective_directory = tmp_path / "effective configs"
    effective_directory.mkdir()
    effective_path = effective_directory / "direct.json"
    effective_path.write_text("{}\n", encoding="utf-8")
    environment.update(
        {
            "CVOI_DIRECT_EPDMS": "1",
            "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH": str(effective_path),
            "NAVSIM_OUTPUT_DIR": str(tmp_path / "output"),
        }
    )
    return environment, call_log, effective_path


def _run(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "/checkpoint.pt", "/training.yaml", "direct-contract"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _calls(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [line.split("\t")[1:] for line in path.read_text(encoding="utf-8").splitlines()]


def test_direct_mode_makes_one_isolated_navtest_scorer_call(tmp_path: Path) -> None:
    environment, call_log, effective_path = _direct_environment(tmp_path)

    completed = _run(environment)

    assert completed.returncode == 0, completed.stderr
    calls = _calls(call_log)
    assert len(calls) == 1
    arguments = calls[0]
    assert any(value.endswith("run_pdm_score_one_stage.py") for value in arguments)
    assert not any(value.endswith("repair_navsim_metric_cache_metadata.py") for value in arguments)
    assert "train_test_split=navtest" in arguments
    assert not any(value.startswith("train_test_split.scene_filter.tokens=") for value in arguments)
    assert "worker.max_workers=1" in arguments
    assert "worker.use_process_pool=false" in arguments
    assert "++agent.forward_mode=stage12" in arguments
    assert "++agent.proposal_checkpoint_path=" in arguments
    assert f"output_dir={environment['NAVSIM_OUTPUT_DIR']}" in arguments

    cvoi_arguments = [value for value in arguments if "agent.cvoi_" in value]
    assert cvoi_arguments == [f"++agent.cvoi_direct_epdms_config_path={effective_path}"]
    assert arguments.count(f"++agent.cvoi_direct_epdms_config_path={effective_path}") == 1
    assert not any("agent.cvoi_manual_navtrain_gate" in value for value in arguments)

    # The legacy positional arguments remain part of the shell ABI, but a direct
    # effective projection is the sole model/config authority passed to the Agent.
    # Empty compatibility overrides are allowed; the positional values must never
    # become a second checkpoint or training-config source.
    checkpoint_arguments = [value for value in arguments if value.startswith("agent.checkpoint_path=")]
    training_config_arguments = [value for value in arguments if value.startswith("agent.training_config_path=")]
    assert checkpoint_arguments in ([], ["agent.checkpoint_path="])
    assert training_config_arguments in ([], ["agent.training_config_path="])
    nonempty_agent_authorities = [
        value
        for value in arguments
        if value.startswith(
            (
                "agent.checkpoint_path=",
                "agent.training_config_path=",
                "++agent.cvoi_direct_epdms_config_path=",
            )
        )
        and value.rsplit("=", 1)[1]
    ]
    assert nonempty_agent_authorities == [f"++agent.cvoi_direct_epdms_config_path={effective_path}"]
    assert "/checkpoint.pt" not in arguments
    assert "/training.yaml" not in arguments


@pytest.mark.parametrize("variable_name", REQUIRED_NAVSIM_ROOTS)
def test_direct_mode_requires_each_navsim_root_before_python(
    tmp_path: Path,
    variable_name: str,
) -> None:
    environment, call_log, _ = _direct_environment(tmp_path)
    environment.pop(variable_name)

    completed = _run(environment)

    assert completed.returncode != 0
    assert _calls(call_log) == []
    assert variable_name in completed.stderr
    assert "Set" in completed.stderr


def test_shell_has_no_navsim_root_defaults_internal_paths_or_horizon_override() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for variable_name in REQUIRED_NAVSIM_ROOTS:
        assert f"${{{variable_name}:?Set " in source
        assert f"${{{variable_name}:-" not in source
    assert "/" + "disk/" not in source
    assert "--horizon" not in source


@pytest.mark.parametrize("mode", ["", "2", "true", "-1"])
def test_direct_mode_must_be_exactly_zero_or_one_before_python(tmp_path: Path, mode: str) -> None:
    environment, call_log, _ = _direct_environment(tmp_path)
    environment["CVOI_DIRECT_EPDMS"] = mode

    completed = _run(environment)

    assert completed.returncode != 0
    assert _calls(call_log) == []
    assert "CVOI_DIRECT_EPDMS" in completed.stderr


@pytest.mark.parametrize("path_value", [None, "", "relative/effective.json"])
def test_direct_mode_requires_one_nonempty_absolute_effective_path_before_python(
    tmp_path: Path,
    path_value: str | None,
) -> None:
    environment, call_log, _ = _direct_environment(tmp_path)
    if path_value is None:
        environment.pop("CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH")
    else:
        environment["CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH"] = path_value

    completed = _run(environment)

    assert completed.returncode != 0
    assert _calls(call_log) == []
    assert "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH" in completed.stderr


def test_direct_mode_rejects_an_alias_config_payload_before_python(tmp_path: Path) -> None:
    environment, call_log, _ = _direct_environment(tmp_path)
    environment["CVOI_DIRECT_EPDMS_CONFIG_PATH"] = "/forbidden/alias.json"

    completed = _run(environment)

    assert completed.returncode != 0
    assert _calls(call_log) == []
    assert "CVOI_DIRECT_EPDMS_CONFIG_PATH" in completed.stderr


def test_direct_effective_path_cannot_leak_into_manual_mode(tmp_path: Path) -> None:
    environment, call_log, _ = _direct_environment(tmp_path)
    environment["CVOI_DIRECT_EPDMS"] = "0"
    environment["CVOI_MANUAL_NAVTRAIN_GATE"] = "1"
    environment["CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH"] = "/manual.json"

    completed = _run(environment)

    assert completed.returncode != 0
    assert _calls(call_log) == []
    assert "CVOI_DIRECT_EPDMS_EFFECTIVE_CONFIG_PATH" in completed.stderr


def test_legacy_official_shell_surface_is_removed() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "CVOI_OFFICIAL_" not in source
    assert "cvoi_official_" not in source


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CVOI_MANUAL_NAVTRAIN_GATE", "0"),
        ("CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH", "/forbidden/manual.json"),
    ],
)
def test_direct_mode_rejects_every_manual_gate_variable_before_python(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    environment, call_log, _ = _direct_environment(tmp_path)
    environment[name] = value

    completed = _run(environment)

    assert completed.returncode != 0
    assert _calls(call_log) == []
    assert "mutually exclusive" in completed.stderr
    assert name in completed.stderr


def test_shell_rejects_a_run_without_one_retained_mode(tmp_path: Path) -> None:
    environment, call_log = _base_environment(tmp_path)

    completed = _run(environment)

    assert completed.returncode == 2
    assert _calls(call_log) == []
    assert "exactly one" in completed.stderr


def test_shell_rejects_both_retained_modes_before_python(tmp_path: Path) -> None:
    environment, call_log, _ = _direct_environment(tmp_path)
    environment["CVOI_MANUAL_NAVTRAIN_GATE"] = "1"
    environment["CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH"] = "/manual.json"

    completed = _run(environment)

    assert completed.returncode == 2
    assert _calls(call_log) == []
    assert "exactly one" in completed.stderr


def test_shell_syntax_is_valid() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
