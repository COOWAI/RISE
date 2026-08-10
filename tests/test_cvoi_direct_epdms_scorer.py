"""Strict deterministic result contracts for one direct NavTest EPDMS run."""

from __future__ import annotations

import csv
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Callable

import pytest

from app.vjepa_cowa_world_model.evaluation.cvoi_direct_epdms import (
    DirectEpdmsSceneRecord,
    aggregate_direct_epdms_results,
    read_cvoi_direct_epdms_scenario_manifest,
    read_direct_epdms_records,
)

SCENE_SCHEMA = "cvoi_direct_epdms_scene_v1"
TRACE_SCHEMA = "cvoi_direct_epdms_trace"
TRACE_VERSION = 1
SUMMARY_SCHEMA = "cvoi_direct_epdms_summary_v1"
PROTOCOL = "epdms_v2_one_stage_navtest"
RETAINED_V2_SCORE_FIXTURE = Path(__file__).parent / "fixtures/cvoi_navsim/epdms_v2_one_stage.csv"
V2_COMPONENTS = (
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
RECORD_FIELDS = {
    "schema",
    "branch",
    "scenario_token",
    "evaluation_seed",
    "epdms",
    "final_horizon",
    "latency_ms",
}
TRACE_FIELDS = {
    "schema",
    "version",
    "split",
    "protocol",
    "branch",
    "scenario_token",
    "evaluation_seed",
    "final_horizon",
    "latency_ms",
}


def _record(
    token: str,
    *,
    branch: str = "full",
    seed: int = 239,
    epdms: float = 0.75,
    horizon: int = 2,
    latency_ms: float = 11.5,
) -> dict[str, object]:
    return {
        "schema": SCENE_SCHEMA,
        "branch": branch,
        "scenario_token": token,
        "evaluation_seed": seed,
        "epdms": epdms,
        "final_horizon": horizon,
        "latency_ms": latency_ms,
    }


def _write_records(root: Path, rows: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "records.jsonl"
    path.write_text(
        "".join(json.dumps(row, allow_nan=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _trace(
    token: str,
    *,
    branch: str = "full",
    seed: int = 239,
    horizon: int = 2,
    latency_ms: float = 11.5,
) -> dict[str, object]:
    return {
        "schema": TRACE_SCHEMA,
        "version": TRACE_VERSION,
        "split": "navtest",
        "protocol": PROTOCOL,
        "branch": branch,
        "scenario_token": token,
        "evaluation_seed": seed,
        "final_horizon": horizon,
        "latency_ms": latency_ms,
    }


def _write_trace(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _write_score_csv(
    path: Path,
    rows: list[tuple[str, object]],
    *,
    summary_score: object | None = None,
    extra_field: str | None = None,
    omit_field: str | None = None,
) -> None:
    fields = ["token", "valid", "score", *V2_COMPONENTS]
    if omit_field is not None:
        fields.remove(omit_field)
    if extra_field is not None:
        fields.append(extra_field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for token, score in rows:
            row: dict[str, object] = {component: "1.0" for component in V2_COMPONENTS}
            row.update(token=token, valid="true", score=score)
            if omit_field is not None:
                row.pop(omit_field)
            if extra_field is not None:
                row[extra_field] = "unexpected"
            writer.writerow(row)
        aggregate_score = summary_score
        if aggregate_score is None:
            aggregate_score = sum(float(score) for _, score in rows) / len(rows)
        summary: dict[str, object] = {component: "1.0" for component in V2_COMPONENTS}
        summary.update(token="average_all_frames", valid="true", score=aggregate_score)
        if omit_field is not None:
            summary.pop(omit_field)
        if extra_field is not None:
            summary[extra_field] = "unexpected"
        writer.writerow(summary)


def _write_raw_result_root(
    tmp_path: Path,
    *,
    tokens: tuple[str, ...] = ("navtest-a", "navtest-b"),
    branch: str = "full",
    seed: int = 239,
) -> Path:
    root = (tmp_path / "direct-result").resolve()
    scorer_output = root / "scorer_output"
    traces = root / "policy_traces"
    scorer_output.mkdir(parents=True)
    traces.mkdir()
    _write_score_csv(
        scorer_output / "scores.csv",
        [(token, 0.70 + index * 0.10) for index, token in enumerate(tokens)],
    )
    for index, token in enumerate(tokens):
        _write_trace(
            traces / f"{token}.json",
            _trace(
                token,
                branch=branch,
                seed=seed,
                horizon=index,
                latency_ms=10.0 + index * 2.0,
            ),
        )
    return root


def test_direct_scene_record_is_frozen_and_exposes_only_the_retained_fields() -> None:
    record = DirectEpdmsSceneRecord(
        branch="full",
        scenario_token="navtest-a",
        evaluation_seed=239,
        epdms=0.75,
        final_horizon=2,
        latency_ms=11.5,
    )

    assert record.branch == "full"
    assert record.scenario_token == "navtest-a"
    assert record.evaluation_seed == 239
    assert record.epdms == pytest.approx(0.75)
    assert record.final_horizon == 2
    assert record.latency_ms == pytest.approx(11.5)
    with pytest.raises(FrozenInstanceError):
        record.epdms = 0.0  # type: ignore[misc]


def test_read_direct_records_accepts_one_sorted_navtest_run(tmp_path: Path) -> None:
    root = (tmp_path / "stored").resolve()
    _write_records(
        root,
        [
            _record("navtest-a", epdms=0.7, horizon=0, latency_ms=10.0),
            _record("navtest-b", epdms=0.8, horizon=4, latency_ms=12.0),
        ],
    )

    records = read_direct_epdms_records(root)

    assert isinstance(records, tuple)
    assert [record.scenario_token for record in records] == ["navtest-a", "navtest-b"]
    assert [record.final_horizon for record in records] == [0, 4]
    assert [record.epdms for record in records] == pytest.approx([0.7, 0.8])
    assert {record.branch for record in records} == {"full"}
    assert {record.evaluation_seed for record in records} == {239}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(schema="wrong"), "schema"),
        (lambda row: row.update(version=1), "unknown|field|version"),
        (lambda row: row.pop("epdms"), "missing|field|epdms"),
        (lambda row: row.update(unexpected=True), "unknown|field|unexpected"),
        (lambda row: row.update(branch=""), "branch"),
        (lambda row: row.update(scenario_token=""), "scenario_token"),
        (lambda row: row.update(evaluation_seed=True), "evaluation_seed"),
        (lambda row: row.update(epdms=float("nan")), "finite|NaN|epdms"),
        (lambda row: row.update(epdms=float("inf")), "finite|Infinity|epdms"),
        (lambda row: row.update(epdms=-0.1), r"epdms|\[0, 1\]"),
        (lambda row: row.update(epdms=1.1), r"epdms|\[0, 1\]"),
        (lambda row: row.update(final_horizon=True), "final_horizon"),
        (lambda row: row.update(final_horizon=-1), "final_horizon|H0|0"),
        (lambda row: row.update(final_horizon=5), "final_horizon|H4|4"),
        (lambda row: row.update(latency_ms=float("nan")), "finite|NaN|latency"),
        (lambda row: row.update(latency_ms=float("inf")), "finite|Infinity|latency"),
        (lambda row: row.update(latency_ms=-0.1), "non-negative|latency"),
    ],
)
def test_read_direct_records_rejects_invalid_or_partial_rows(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    message: str,
) -> None:
    row = _record("navtest-a")
    mutate(row)
    root = (tmp_path / "stored").resolve()
    _write_records(root, [row])

    with pytest.raises((TypeError, ValueError), match=message):
        read_direct_epdms_records(root)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_record("navtest-a"), _record("navtest-a")], "duplicate|unique"),
        ([_record("navtest-b"), _record("navtest-a")], "sorted"),
        ([_record("navtrain-a")], "NavTest|navtest"),
        ([_record("navtest-a"), _record("navtest-b", branch="no_cf")], "branch"),
        ([_record("navtest-a"), _record("navtest-b", seed=3407)], "seed"),
    ],
)
def test_read_direct_records_rejects_cross_row_inventory_drift(
    tmp_path: Path,
    rows: list[dict[str, object]],
    message: str,
) -> None:
    root = (tmp_path / "stored").resolve()
    _write_records(root, rows)

    with pytest.raises(ValueError, match=message):
        read_direct_epdms_records(root)


def test_read_direct_records_requires_one_absolute_existing_records_file(tmp_path: Path) -> None:
    relative = Path("relative-result")
    with pytest.raises((TypeError, ValueError), match="absolute"):
        read_direct_epdms_records(relative)

    missing = (tmp_path / "missing").resolve()
    with pytest.raises(ValueError, match="records.jsonl|existing"):
        read_direct_epdms_records(missing)

    empty = (tmp_path / "empty").resolve()
    _write_records(empty, [])
    with pytest.raises(ValueError, match="empty|record"):
        read_direct_epdms_records(empty)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("log_name", ""),
        ("log_name", None),
        ("current_camera_data_path", ""),
        ("current_camera_data_path", ["not", "a", "path"]),
    ],
)
def test_direct_scenario_manifest_rejects_invalid_retained_identity_fields(
    tmp_path: Path,
    field: str,
    invalid_value: object,
) -> None:
    row: dict[str, object] = {
        "schema": "cvoi_navsim_scenario_v1",
        "protocol_id": PROTOCOL,
        "scenario_token": "navtest-a",
        "observation_key": "a" * 64,
        "log_name": "log-a",
        "current_camera_data_path": "/dataset/log-a/cam_f0.jpg",
    }
    row[field] = invalid_value
    path = (tmp_path / "scenario-manifest.jsonl").resolve()
    path.write_text(
        json.dumps(row, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=field):
        read_cvoi_direct_epdms_scenario_manifest(path)


def test_aggregate_joins_exact_score_and_trace_inventories_and_exclusively_writes_outputs(tmp_path: Path) -> None:
    root = _write_raw_result_root(tmp_path)
    assert {path.name for path in root.iterdir()} == {"scorer_output", "policy_traces"}

    aggregate_direct_epdms_results(root)

    assert {path.name for path in root.iterdir()} == {
        "scorer_output",
        "policy_traces",
        "records.jsonl",
        "summary.json",
    }
    records = read_direct_epdms_records(root)
    assert [record.scenario_token for record in records] == ["navtest-a", "navtest-b"]
    assert [record.epdms for record in records] == pytest.approx([0.7, 0.8])
    assert [record.final_horizon for record in records] == [0, 1]
    assert [record.latency_ms for record in records] == pytest.approx([10.0, 12.0])
    summary_bytes = (root / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    assert summary == {
        "schema": SUMMARY_SCHEMA,
        "branch": "full",
        "evaluation_seed": 239,
        "scenario_count": 2,
        "mean_epdms": pytest.approx(0.75),
        "mean_latency_ms": pytest.approx(11.0),
    }


def test_aggregate_accepts_the_retained_pandas_epdms_v2_csv(tmp_path: Path) -> None:
    root = _write_raw_result_root(tmp_path, tokens=("token-a", "token-b"))
    (root / "scorer_output/scores.csv").write_bytes(RETAINED_V2_SCORE_FIXTURE.read_bytes())

    records = aggregate_direct_epdms_results(root)

    assert [record.scenario_token for record in records] == ["token-a", "token-b"]
    assert [record.epdms for record in records] == pytest.approx([0.66, 0.68])


def test_aggregate_accepts_the_real_timestamped_epdms_v2_csv_name(tmp_path: Path) -> None:
    root = _write_raw_result_root(tmp_path)
    score_path = root / "scorer_output/scores.csv"
    timestamped_path = score_path.with_name("2026.07.31.12.00.00.csv")
    score_path.rename(timestamped_path)

    records = aggregate_direct_epdms_results(root)

    assert [record.scenario_token for record in records] == ["navtest-a", "navtest-b"]
    assert timestamped_path.is_file()


@pytest.mark.parametrize("csv_count", [0, 2])
def test_aggregate_requires_exactly_one_epdms_v2_csv(tmp_path: Path, csv_count: int) -> None:
    root = _write_raw_result_root(tmp_path)
    score_path = root / "scorer_output/scores.csv"
    if csv_count == 0:
        score_path.unlink()
    else:
        payload = score_path.read_bytes()
        score_path.rename(score_path.with_name("2026.07.31.12.00.00.csv"))
        score_path.with_name("2026.07.31.12.00.01.csv").write_bytes(payload)

    with pytest.raises(ValueError, match=r"exactly one|found [02]|score CSV"):
        aggregate_direct_epdms_results(root)

    assert not (root / "records.jsonl").exists()
    assert not (root / "summary.json").exists()


def test_aggregate_rejects_a_symlinked_epdms_v2_csv(tmp_path: Path) -> None:
    root = _write_raw_result_root(tmp_path)
    score_path = root / "scorer_output/scores.csv"
    real_payload = score_path.with_name("score-payload.bin")
    score_path.rename(real_payload)
    score_path.with_name("2026.07.31.12.00.00.csv").symlink_to(real_payload.name)

    with pytest.raises(ValueError, match=r"symlink|non-symlink"):
        aggregate_direct_epdms_results(root)

    assert not (root / "records.jsonl").exists()
    assert not (root / "summary.json").exists()


def test_aggregate_rejects_an_epdms_v2_csv_resolving_outside_scorer_output(tmp_path: Path) -> None:
    root = _write_raw_result_root(tmp_path)
    score_path = root / "scorer_output/scores.csv"
    outside_path = (tmp_path / "outside.csv").resolve()
    score_path.rename(outside_path)
    score_path.with_name("2026.07.31.12.00.00.csv").symlink_to(outside_path)

    with pytest.raises(ValueError, match=r"contained|outside|scorer_output"):
        aggregate_direct_epdms_results(root)

    assert not (root / "records.jsonl").exists()
    assert not (root / "summary.json").exists()


@pytest.mark.parametrize("mutation", ["nonconsecutive", "second_index"])
def test_aggregate_rejects_invalid_pandas_epdms_v2_index(tmp_path: Path, mutation: str) -> None:
    root = _write_raw_result_root(tmp_path, tokens=("token-a", "token-b"))
    score_path = root / "scorer_output/scores.csv"
    with RETAINED_V2_SCORE_FIXTURE.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    if mutation == "nonconsecutive":
        rows[2][0] = "7"
    else:
        rows[0].insert(1, "")
        for index, row in enumerate(rows[1:]):
            row.insert(1, str(index))
    with score_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)

    with pytest.raises(ValueError, match="index"):
        aggregate_direct_epdms_results(root)

    assert not (root / "records.jsonl").exists()
    assert not (root / "summary.json").exists()


@pytest.mark.parametrize("existing_name", ["records.jsonl", "summary.json"])
def test_aggregate_preflights_both_exclusive_outputs_before_writing_either(
    tmp_path: Path,
    existing_name: str,
) -> None:
    root = _write_raw_result_root(tmp_path)
    existing = root / existing_name
    original = b"operator-owned\n"
    existing.write_bytes(original)

    with pytest.raises(FileExistsError, match=existing_name):
        aggregate_direct_epdms_results(root)

    assert existing.read_bytes() == original
    other_name = "summary.json" if existing_name == "records.jsonl" else "records.jsonl"
    assert not (root / other_name).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_trace", "trace|missing|coverage|inventory"),
        ("extra_trace", "trace|extra|coverage|inventory"),
        ("trace_token", "token|coverage|inventory"),
        ("duplicate_trace_token", "duplicate|token"),
        ("mixed_branch", "branch"),
        ("mixed_seed", "seed"),
        ("navtrain_split", "NavTest|navtest|split"),
        ("navtrain_token", "NavTest|navtest"),
        ("wrong_protocol", "protocol"),
        ("wrong_schema", "schema"),
        ("wrong_version", "version"),
        ("missing_field", "missing|field|latency"),
        ("unknown_field", "unknown|field|unexpected"),
        ("bad_horizon", "horizon|H4|4"),
        ("nan_latency", "finite|NaN|latency"),
        ("inf_latency", "finite|Infinity|latency"),
    ],
)
def test_aggregate_rejects_invalid_trace_or_score_trace_inventory_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    root = _write_raw_result_root(tmp_path)
    traces = root / "policy_traces"
    first = traces / "navtest-a.json"
    second = traces / "navtest-b.json"
    if mutation == "missing_trace":
        first.unlink()
    elif mutation == "extra_trace":
        _write_trace(traces / "navtest-extra.json", _trace("navtest-extra"))
    else:
        target = second if mutation in {"duplicate_trace_token", "mixed_branch", "mixed_seed"} else first
        payload = json.loads(target.read_text(encoding="utf-8"))
        if mutation == "trace_token":
            payload["scenario_token"] = "navtest-other"
        elif mutation == "duplicate_trace_token":
            payload["scenario_token"] = "navtest-a"
        elif mutation == "mixed_branch":
            payload["branch"] = "no_cf"
        elif mutation == "mixed_seed":
            payload["evaluation_seed"] = 3407
        elif mutation == "navtrain_split":
            payload["split"] = "navtrain"
        elif mutation == "navtrain_token":
            payload["scenario_token"] = "navtrain-a"
        elif mutation == "wrong_protocol":
            payload["protocol"] = "pdms_v1_navtest"
        elif mutation == "wrong_schema":
            payload["schema"] = "wrong"
        elif mutation == "wrong_version":
            payload["version"] = 2
        elif mutation == "missing_field":
            payload.pop("latency_ms")
        elif mutation == "unknown_field":
            payload["unexpected"] = True
        elif mutation == "bad_horizon":
            payload["final_horizon"] = 5
        elif mutation == "nan_latency":
            payload["latency_ms"] = float("nan")
        elif mutation == "inf_latency":
            payload["latency_ms"] = float("inf")
        else:
            raise AssertionError(f"unhandled mutation {mutation!r}")
        _write_trace(target, payload)

    with pytest.raises(ValueError, match=message):
        aggregate_direct_epdms_results(root)

    assert not (root / "records.jsonl").exists()
    assert not (root / "summary.json").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_score", "score|missing|coverage|inventory"),
        ("extra_score", "score|extra|coverage|inventory"),
        ("duplicate_score", "duplicate|token"),
        ("nan_score", "finite|NaN|score"),
        ("inf_score", "finite|Infinity|score"),
        ("partial_csv", "field|missing"),
        ("unknown_csv", "field|extra|unknown"),
        ("missing_summary", "average_all_frames|aggregate"),
    ],
)
def test_aggregate_reuses_strict_v2_score_csv_validation(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    root = _write_raw_result_root(tmp_path)
    path = root / "scorer_output/scores.csv"
    if mutation == "missing_score":
        _write_score_csv(path, [("navtest-a", 0.7)])
    elif mutation == "extra_score":
        _write_score_csv(path, [("navtest-a", 0.7), ("navtest-b", 0.8), ("navtest-extra", 0.9)])
    elif mutation == "duplicate_score":
        _write_score_csv(path, [("navtest-a", 0.7), ("navtest-a", 0.8)])
    elif mutation == "nan_score":
        _write_score_csv(path, [("navtest-a", "nan"), ("navtest-b", 0.8)], summary_score=0.75)
    elif mutation == "inf_score":
        _write_score_csv(path, [("navtest-a", "inf"), ("navtest-b", 0.8)], summary_score=0.75)
    elif mutation == "partial_csv":
        _write_score_csv(path, [("navtest-a", 0.7), ("navtest-b", 0.8)], omit_field="lane_keeping")
    elif mutation == "unknown_csv":
        _write_score_csv(path, [("navtest-a", 0.7), ("navtest-b", 0.8)], extra_field="unexpected")
    elif mutation == "missing_summary":
        raw = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(raw[:-1]), encoding="utf-8")
    else:
        raise AssertionError(f"unhandled mutation {mutation!r}")

    with pytest.raises(ValueError, match=message):
        aggregate_direct_epdms_results(root)

    assert not (root / "records.jsonl").exists()
    assert not (root / "summary.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_scorer_dir",
        "missing_trace_dir",
        "extra_root_entry",
        "relative_root",
    ],
)
def test_aggregate_requires_the_exact_precreated_result_root_layout(tmp_path: Path, mutation: str) -> None:
    root = _write_raw_result_root(tmp_path)
    requested = root
    if mutation == "missing_scorer_dir":
        (root / "scorer_output/scores.csv").unlink()
        (root / "scorer_output").rmdir()
    elif mutation == "missing_trace_dir":
        for path in (root / "policy_traces").iterdir():
            path.unlink()
        (root / "policy_traces").rmdir()
    elif mutation == "extra_root_entry":
        (root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    elif mutation == "relative_root":
        requested = Path("relative-result")
    else:
        raise AssertionError(f"unhandled mutation {mutation!r}")

    with pytest.raises((TypeError, ValueError), match="root|absolute|scorer_output|policy_traces|layout"):
        aggregate_direct_epdms_results(requested)
