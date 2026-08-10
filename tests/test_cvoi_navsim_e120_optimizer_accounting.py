"""Formal-v2 optimizer accounting must count only committed updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_planner_integration import (
    record_formal_v2_navsim_e120_optimizer_exposure,
)


@dataclass
class _Recorder:
    calls: list[tuple[int, int]] = field(default_factory=list)

    def record(self, *, horizon: int, batch_size: int) -> None:
        self.calls.append((horizon, batch_size))


def test_successful_optimizer_step_commits_exactly_one_horizon_exposure() -> None:
    recorder = _Recorder()

    record_formal_v2_navsim_e120_optimizer_exposure(
        recorder,
        optimizer_step_successful=True,
        horizon=2,
        batch_size=8,
    )

    assert recorder.calls == [(2, 8)]


def test_rejected_optimizer_step_fails_fast_without_changing_exposure() -> None:
    recorder = _Recorder()

    with pytest.raises(RuntimeError, match="rejected optimizer step"):
        record_formal_v2_navsim_e120_optimizer_exposure(
            recorder,
            optimizer_step_successful=False,
            horizon=1,
            batch_size=8,
        )

    assert recorder.calls == []


def test_non_formal_training_does_not_require_formal_accounting_metadata() -> None:
    record_formal_v2_navsim_e120_optimizer_exposure(
        None,
        optimizer_step_successful=False,
        horizon=None,
        batch_size=None,
    )


def test_h4_optimizer_step_is_part_of_the_manual_navsim_exposure_contract() -> None:
    recorder = _Recorder()

    record_formal_v2_navsim_e120_optimizer_exposure(
        recorder,
        optimizer_step_successful=True,
        horizon=4,
        batch_size=3,
    )

    assert recorder.calls == [(4, 3)]


def test_planner_training_line_never_records_exposure_before_optimizer_commit() -> None:
    source_path = (
        Path(__file__).resolve().parents[1] / "app/vjepa_cowa_world_model/training/lines/planner_world_model.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "formal_v2_planner_epoch_state.record" not in source
    assert "cvoi_formal_v2_optimizer_accounting" not in source
    successful_commit = source.index("optimizer_step_successful=_value_optimizer_step_successful")
    assert source.rfind("optimizer.step()", 0, successful_commit) >= 0
