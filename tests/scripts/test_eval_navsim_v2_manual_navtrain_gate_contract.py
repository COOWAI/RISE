"""Executable shell-boundary tests for the manual NavTrain Oracle scorer."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/eval_navsim/eval_navsim_v2_pdms.sh"


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    recorder = tmp_path / "python-recorder.sh"
    call_log = tmp_path / "calls.tsv"
    recorder.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "{ printf 'CALL'; printf '\\t%s' \"$@\"; printf '\\n'; } >> \"${FAKE_PYTHON_CALL_LOG:?}\"\n",
        encoding="utf-8",
    )
    recorder.chmod(0o755)
    devkit = tmp_path / "navsim-v2"
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(
            (
                "CVOI_DIRECT_EPDMS",
                "CVOI_MANUAL_NAVTRAIN_GATE",
                "CVOI_FORMAL_V2_NAVSIM_E120",
            )
        ):
            environment.pop(key)
    environment.update(
        {
            "PYTHON_BIN": str(recorder),
            "FAKE_PYTHON_CALL_LOG": str(call_log),
            "OPENSCENE_DATA_ROOT": str(tmp_path / "openscene"),
            "NAVSIM_EXP_ROOT": str(tmp_path / "navsim-exp"),
            "NUPLAN_MAPS_ROOT": str(tmp_path / "maps"),
            "NAVSIM_DEVKIT_ROOT": str(devkit),
            "PYTHONPATH": "",
            "METRIC_CACHE_PATH": str(tmp_path / "metric-cache"),
            "NAVSIM_OUTPUT_DIR": str(tmp_path / "output"),
            "CVOI_MANUAL_NAVTRAIN_GATE": "1",
            "CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH": str(tmp_path / "manual-config.json"),
            "MAX_WORKERS": "37",
            "USE_PROCESS_POOL": "true",
            "FORWARD_MODE": "stage3",
            "PROPOSAL_CHECKPOINT": "/must/not/leak.pt",
        }
    )
    return environment, call_log


def _run(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), "/missing-checkpoint.pt", "/missing-training.yaml", "manual-navtrain-gate"],
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


def test_manual_navtrain_gate_forces_one_isolated_navtrain_scorer_call(tmp_path: Path) -> None:
    environment, call_log = _environment(tmp_path)

    completed = _run(environment)

    assert completed.returncode == 0, completed.stderr
    calls = _calls(call_log)
    assert len(calls) == 1
    arguments = calls[0]
    assert any(value.endswith("run_pdm_score_one_stage.py") for value in arguments)
    assert "--config-dir" in arguments
    assert str(REPO_ROOT / "configs/navsim/cvoi_manual") in arguments
    assert "train_test_split=navtrain" in arguments
    assert "worker.max_workers=1" in arguments
    assert "worker.use_process_pool=false" in arguments
    assert "++agent.forward_mode=stage12" in arguments
    assert "++agent.proposal_checkpoint_path=" in arguments
    assert "agent=cvoi_manual_vjepa_world_model_agent" in arguments
    assert (
        f"++agent.cvoi_manual_navtrain_gate_config_path=" f"{environment['CVOI_MANUAL_NAVTRAIN_GATE_CONFIG_PATH']}"
    ) in arguments
    assert not any("cvoi_formal_v2_navsim_e120_" in value for value in arguments)
    assert not any(value.endswith("repair_navsim_metric_cache_metadata.py") for value in arguments)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CVOI_FORMAL_V2_NAVSIM_E120", "1"),
        ("CVOI_FORMAL_V2_NAVSIM_E120_SELECTION", "0"),
        ("CVOI_FORMAL_V2_NAVSIM_E120_CONFIG_PATH", "/formal.json"),
        ("CVOI_FORMAL_V2_NAVSIM_E120_SELECTION_CONFIG_PATH", "/selection.json"),
        ("CVOI_FORMAL_V2_NAVSIM_E120_NAVTRAIN_GATE", "0"),
        ("CVOI_FORMAL_V2_NAVSIM_E120_NAVTRAIN_GATE_CONFIG_PATH", "/gate.json"),
        ("CVOI_FORMAL_V2_NAVSIM_E120_HYDRA_AGENT_CONFIG_PATH", "/agent.yaml"),
        ("CVOI_FORMAL_V2_NAVSIM_E120_HYDRA_AGENT_CONFIG_SHA256", "0" * 64),
    ],
)
def test_manual_navtrain_gate_rejects_retired_formal_variables_before_python(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    environment, call_log = _environment(tmp_path)
    environment[name] = value

    completed = _run(environment)

    assert completed.returncode != 0
    assert _calls(call_log) == []
    assert "mutually exclusive" in completed.stderr


@pytest.mark.parametrize("mode", ["", "2", "true"])
def test_manual_navtrain_gate_rejects_invalid_mode_before_python(tmp_path: Path, mode: str) -> None:
    environment, call_log = _environment(tmp_path)
    environment["CVOI_MANUAL_NAVTRAIN_GATE"] = mode

    completed = _run(environment)

    assert completed.returncode != 0
    assert _calls(call_log) == []
    assert "CVOI_MANUAL_NAVTRAIN_GATE" in completed.stderr
