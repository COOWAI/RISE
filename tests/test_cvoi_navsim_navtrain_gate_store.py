"""Bounded-memory SQLite stores for the official-navtrain Gate pipeline."""

from __future__ import annotations

import csv
import hashlib
import os
import sqlite3
import struct
from pathlib import Path

import pytest

from app.vjepa_cowa_world_model.training import cvoi_navsim_navtrain_gate_store as store_module
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import (
    FEATURE_STORE_SCHEMA,
    MANUAL_ORACLE_STORE_SCHEMA_V2,
    FeatureRow,
    FeatureStoreMetadata,
    OracleStoreMetadata,
    ScoreIdentity,
    ScoreRow,
    ScoreStoreMetadata,
    create_embedded_oracle_store_v2,
    create_feature_store,
    create_score_store,
    create_score_store_from_official_csv,
    open_embedded_oracle_store_v2,
    open_feature_store,
    open_score_store,
    sha256_file,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
COMPONENTS = (
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


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(31), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split(log_name: str) -> str:
    bucket = int(hashlib.sha256(log_name.encode("utf-8")).hexdigest(), 16) % 10
    return "dev" if bucket == 0 else "train"


def _logs_for_both_splits() -> tuple[str, str]:
    train = next(name for index in range(100) if _split(name := f"log-{index}") == "train")
    dev = next(name for index in range(100) if _split(name := f"log-{index}") == "dev")
    return train, dev


def _feature_metadata(horizon: int) -> FeatureStoreMetadata:
    return FeatureStoreMetadata(
        protocol_id="epdms_v2_one_stage_navtrain_gate_label_v1",
        policy_id=f"forced-h{horizon}",
        lineage="fixture-lineage",
        horizon=horizon,
        scenario_manifest_sha256=SHA_A,
        metric_cache_inventory_sha256=SHA_B,
        feature_schema="sequential_cvoi_gate_features_lambda_independent_h4_v1",
        feature_sources=("pooled_observed", "pooled_prefix", "field_value"),
        common_random_seed=17,
    )


def _feature_rows(horizon: int) -> list[FeatureRow]:
    return [
        FeatureRow(0, "token-a", "1" * 64, "c" * 64, (horizon + 0.5, 1.0, 2.0)),
        FeatureRow(1, "token-b", "2" * 64, "d" * 64, (horizon + 1.5, 3.0, 4.0)),
    ]


def _score_metadata(horizon: int, source: Path) -> ScoreStoreMetadata:
    return ScoreStoreMetadata(
        protocol_id="epdms_v2_one_stage_navtrain_gate_label_v1",
        policy_id=f"forced-h{horizon}",
        lineage="fixture-lineage",
        horizon=horizon,
        scenario_manifest_sha256=SHA_A,
        metric_cache_inventory_sha256=SHA_B,
        source_path=source,
        source_sha256=_sha(source),
        score_semantics="official_v2_one_stage_ordinary_row_score",
    )


def _score_rows(horizon: int, logs: tuple[str, str], *, score_shift: float = 0.0) -> list[ScoreRow]:
    return [
        ScoreRow(0, "token-a", "1" * 64, logs[0], 0.10 + horizon * 0.05 + score_shift),
        ScoreRow(1, "token-b", "2" * 64, logs[1], 0.20 + horizon * 0.05 + score_shift),
    ]


def _oracle_metadata() -> OracleStoreMetadata:
    return OracleStoreMetadata(
        protocol_id="epdms_v2_one_stage_navtrain_gate_label_v1",
        lineage="fixture-lineage",
        scenario_manifest_sha256=SHA_A,
        metric_cache_inventory_sha256=SHA_B,
        lambda_grid=(0.0, 0.1),
    )


def _create_oracle_intermediates(
    tmp_path: Path,
    *,
    score_shift: float = 0.0,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, str], dict[int, str]]:
    logs = _logs_for_both_splits()
    score_paths: dict[int, Path] = {}
    feature_paths: dict[int, Path] = {}
    score_hashes: dict[int, str] = {}
    feature_hashes: dict[int, str] = {}
    for horizon in range(5):
        source = tmp_path / f"score-source-h{horizon}.txt"
        source.write_text(f"score h{horizon}", encoding="utf-8")
        score_path = tmp_path / f"score-h{horizon}.sqlite3"
        feature_path = tmp_path / f"feature-h{horizon}.sqlite3"
        score_receipt = create_score_store(
            score_path,
            _score_metadata(horizon, source),
            _score_rows(horizon, logs, score_shift=score_shift),
            aggregate_score=0.4 + 0.01 * horizon,
        )
        feature_receipt = create_feature_store(
            feature_path,
            _feature_metadata(horizon),
            _feature_rows(horizon),
        )
        score_paths[horizon] = score_path
        feature_paths[horizon] = feature_path
        score_hashes[horizon] = score_receipt.sha256
        feature_hashes[horizon] = feature_receipt.sha256
    return score_paths, feature_paths, score_hashes, feature_hashes


def test_feature_store_streaming_round_trip_and_create_once(tmp_path: Path) -> None:
    path = tmp_path / "features.sqlite3"
    consumed: list[int] = []

    def rows():
        for row in _feature_rows(2):
            consumed.append(row.row_index)
            yield row

    receipt = create_feature_store(path, _feature_metadata(2), rows())
    assert consumed == [0, 1]
    assert receipt.sha256 == sha256_file(path)
    assert receipt.row_count == 2
    assert receipt.feature_dim == 3
    assert not Path(f"{path}-wal").exists()
    assert not Path(f"{path}-shm").exists()

    with open_feature_store(path, expected_sha256=receipt.sha256) as store:
        assert store.metadata == _feature_metadata(2)
        assert [(row.row_index, row.token, row.features) for row in store.iter_rows()] == [
            (0, "token-a", (2.5, 1.0, 2.0)),
            (1, "token-b", (3.5, 3.0, 4.0)),
        ]
        assert store.read_feature_batch([1, 0, 1]) == (
            (3.5, 3.0, 4.0),
            (2.5, 1.0, 2.0),
            (3.5, 3.0, 4.0),
        )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_feature_store(path, _feature_metadata(2), _feature_rows(2))


def test_feature_store_rejects_nan_symlink_sidecars_schema_count_and_dimension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite"):
        create_feature_store(
            tmp_path / "nan.sqlite3",
            _feature_metadata(0),
            [FeatureRow(0, "token-a", "1" * 64, "c" * 64, (float("nan"),))],
        )
    assert not (tmp_path / "nan.sqlite3").exists()

    path = tmp_path / "features.sqlite3"
    receipt = create_feature_store(path, _feature_metadata(0), _feature_rows(0))
    link = tmp_path / "features-link.sqlite3"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        open_feature_store(link)

    sidecar = Path(f"{path}-wal")
    sidecar.touch()
    with pytest.raises(ValueError, match="sidecar"):
        open_feature_store(path)
    sidecar.unlink()

    variants = {
        "schema": "UPDATE metadata SET value='\"wrong\"' WHERE key='schema'",
        "count": "DELETE FROM rows WHERE row_index=1",
        "dimension": "UPDATE rows SET feature=x'00000000' WHERE row_index=0",
        "rogue-column": "ALTER TABLE rows ADD COLUMN rogue TEXT",
        "fractional-horizon": "UPDATE metadata SET value='0.5' WHERE key='horizon'",
    }
    for name, statement in variants.items():
        tampered = tmp_path / f"tampered-{name}.sqlite3"
        tampered.write_bytes(path.read_bytes())
        connection = sqlite3.connect(tampered)
        connection.execute(statement)
        connection.commit()
        connection.close()
        with pytest.raises(ValueError):
            open_feature_store(tampered)

    assert sha256_file(path) == receipt.sha256


def test_score_store_iterable_and_official_csv_are_streamed(tmp_path: Path) -> None:
    logs = _logs_for_both_splits()
    source = tmp_path / "source.txt"
    source.write_text("bound source", encoding="utf-8")
    path = tmp_path / "scores.sqlite3"
    receipt = create_score_store(
        path,
        _score_metadata(1, source),
        _score_rows(1, logs),
        aggregate_score=0.625,
    )
    with open_score_store(path, expected_sha256=receipt.sha256) as store:
        assert store.aggregate_score == 0.625
        assert [(row.token, row.log_name, row.score) for row in store.iter_rows()] == [
            ("token-a", logs[0], pytest.approx(0.15)),
            ("token-b", logs[1], pytest.approx(0.25)),
        ]

    csv_path = tmp_path / "official.csv"
    fields = ["token", "valid", "score", *COMPONENTS]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for token, score in (("token-a", "0.3"), ("token-b", "0.4"), ("average_all_frames", "0.35")):
            row = {field: "1.0" for field in COMPONENTS}
            row.update(token=token, valid="true", score=score)
            writer.writerow(row)
    csv_metadata = ScoreStoreMetadata(
        protocol_id="epdms_v2_one_stage_navtrain_gate_label_v1",
        policy_id="forced-h3",
        lineage="fixture-lineage",
        horizon=3,
        scenario_manifest_sha256=SHA_A,
        metric_cache_inventory_sha256=SHA_B,
        source_path=csv_path,
        source_sha256=_sha(csv_path),
        score_semantics="official_v2_one_stage_ordinary_row_score",
    )
    csv_store = tmp_path / "csv-scores.sqlite3"
    csv_receipt = create_score_store_from_official_csv(
        csv_store,
        csv_metadata,
        identities=(
            ScoreIdentity(0, "token-a", "1" * 64, logs[0]),
            ScoreIdentity(1, "token-b", "2" * 64, logs[1]),
        ),
    )
    with open_score_store(csv_store, expected_sha256=csv_receipt.sha256) as store:
        assert store.aggregate_score == 0.35
        assert [row.score for row in store.iter_rows()] == [0.3, 0.4]


def test_embedded_oracle_store_v2_is_self_contained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_paths, feature_paths, _, _ = _create_oracle_intermediates(tmp_path)
    oracle_path = tmp_path / "oracle-v2.sqlite3"
    receipt = create_embedded_oracle_store_v2(
        oracle_path,
        _oracle_metadata(),
        score_store_paths=score_paths,
        feature_store_paths=feature_paths,
    )

    connection = sqlite3.connect(oracle_path)
    assert connection.execute("PRAGMA user_version").fetchone() == (2,)
    assert connection.execute(
        "SELECT type,name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall() == [
        ("table", "features"),
        ("table", "metadata"),
        ("table", "rows"),
    ]
    assert connection.execute(
        "SELECT horizon,COUNT(*) FROM features GROUP BY horizon ORDER BY horizon"
    ).fetchall() == [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    assert metadata["schema"] == f'"{MANUAL_ORACLE_STORE_SCHEMA_V2}"'
    assert metadata["feature_payload_policy"] == '"embedded_h0_h4_float32_le_v1"'
    assert not {
        "official_csv_sha256s",
        "score_store_sha256s",
        "feature_store_sha256s",
    } & set(metadata)
    connection.close()

    for path in (*score_paths.values(), *feature_paths.values()):
        path.unlink()
    monkeypatch.setattr(
        store_module,
        "open_feature_store",
        lambda *_args, **_kwargs: pytest.fail("v2 open must not access an external feature store"),
    )
    monkeypatch.setattr(
        store_module,
        "open_score_store",
        lambda *_args, **_kwargs: pytest.fail("v2 open must not access an external score store"),
    )
    with open_embedded_oracle_store_v2(oracle_path, expected_sha256=receipt.sha256) as store:
        assert [row.record_id for row in store.iter_rows()] == [0, 1]
        assert list(store.iter_split_record_ids("train")) == [0]
        assert list(store.iter_split_record_ids("dev")) == [1]
        assert [row.features for row in store.read_feature_batch(record_ids=[1, 0, 1], horizons=[3, 0, 4])] == [
            (4.5, 3.0, 4.0),
            (0.5, 1.0, 2.0),
            (5.5, 3.0, 4.0),
        ]
        batch = store.read_training_batch(
            record_ids=[0, 1],
            horizons=[0, 3],
            lambda_computes=[0.1, 0.0],
        )
    assert batch[0].features == (0.5, 1.0, 2.0, 0.1)
    assert batch[1].features == (4.5, 3.0, 4.0, 0.0)
    assert batch[0].continue_target is False
    assert batch[1].continue_target is True


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_h4",
        "short_blob",
        "nan_blob",
        "positive_infinity_blob",
        "negative_infinity_blob",
        "text_feature",
        "null_feature",
        "wrong_record_id",
        "rogue_feature_refs",
    ),
)
def test_embedded_oracle_store_v2_rejects_corrupt_features_on_open(
    tmp_path: Path,
    mutation: str,
) -> None:
    score_paths, feature_paths, _, _ = _create_oracle_intermediates(tmp_path)
    valid = tmp_path / "oracle-v2.sqlite3"
    create_embedded_oracle_store_v2(
        valid,
        _oracle_metadata(),
        score_store_paths=score_paths,
        feature_store_paths=feature_paths,
    )
    tampered = tmp_path / f"oracle-v2-{mutation}.sqlite3"
    tampered.write_bytes(valid.read_bytes())
    connection = sqlite3.connect(tampered)
    if mutation == "missing_h4":
        connection.execute("DELETE FROM features WHERE horizon=4 AND record_id=1")
    elif mutation == "short_blob":
        connection.execute("UPDATE features SET feature=? WHERE record_id=1 AND horizon=4", (b"\x00" * 8,))
    elif mutation in {"nan_blob", "positive_infinity_blob", "negative_infinity_blob"}:
        non_finite = {
            "nan_blob": float("nan"),
            "positive_infinity_blob": float("inf"),
            "negative_infinity_blob": float("-inf"),
        }[mutation]
        connection.execute(
            "UPDATE features SET feature=? WHERE record_id=1 AND horizon=4",
            (struct.pack("<3f", 1.0, non_finite, 2.0),),
        )
    elif mutation == "text_feature":
        connection.execute("UPDATE features SET feature='not-a-blob' WHERE record_id=1 AND horizon=4")
    elif mutation == "null_feature":
        connection.execute("ALTER TABLE features RENAME TO old_features")
        connection.execute(
            "CREATE TABLE features ("
            "record_id INTEGER NOT NULL, horizon INTEGER NOT NULL CHECK(horizon BETWEEN 0 AND 4), "
            "feature BLOB, PRIMARY KEY(record_id,horizon)) WITHOUT ROWID"
        )
        connection.execute("INSERT INTO features SELECT * FROM old_features")
        connection.execute("UPDATE features SET feature=NULL WHERE record_id=1 AND horizon=4")
        connection.execute("DROP TABLE old_features")
    elif mutation == "wrong_record_id":
        connection.execute("UPDATE features SET record_id=9 WHERE record_id=1 AND horizon=4")
    elif mutation == "rogue_feature_refs":
        connection.execute("CREATE TABLE feature_refs(horizon INTEGER PRIMARY KEY,path TEXT,sha256 TEXT)")
    else:
        raise AssertionError(f"unhandled mutation {mutation}")
    connection.commit()
    connection.close()

    with pytest.raises(ValueError):
        open_embedded_oracle_store_v2(tampered)


def test_embedded_oracle_store_v2_rejects_symlink_input_and_output(tmp_path: Path) -> None:
    score_paths, feature_paths, _, _ = _create_oracle_intermediates(tmp_path)
    feature_link = tmp_path / "feature-link.sqlite3"
    feature_link.symlink_to(feature_paths[0])
    feature_paths[0] = feature_link
    with pytest.raises(ValueError, match="symlink"):
        create_embedded_oracle_store_v2(
            tmp_path / "oracle-v2.sqlite3",
            _oracle_metadata(),
            score_store_paths=score_paths,
            feature_store_paths=feature_paths,
        )

    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    output_link = tmp_path / "oracle-link.sqlite3"
    output_link.symlink_to(victim)
    feature_paths[0] = feature_link.resolve()
    with pytest.raises(ValueError, match="symlink"):
        create_embedded_oracle_store_v2(
            output_link,
            _oracle_metadata(),
            score_store_paths=score_paths,
            feature_store_paths=feature_paths,
        )
    assert victim.read_text(encoding="utf-8") == "unchanged"


def test_embedded_oracle_store_v2_atomically_replaces_a_regular_target(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    first_dir.mkdir()
    first_scores, first_features, _, _ = _create_oracle_intermediates(first_dir)
    oracle_path = tmp_path / "oracle-v2.sqlite3"
    first_receipt = create_embedded_oracle_store_v2(
        oracle_path,
        _oracle_metadata(),
        score_store_paths=first_scores,
        feature_store_paths=first_features,
    )

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second_scores, second_features, _, _ = _create_oracle_intermediates(second_dir, score_shift=0.1)
    second_receipt = create_embedded_oracle_store_v2(
        oracle_path,
        _oracle_metadata(),
        score_store_paths=second_scores,
        feature_store_paths=second_features,
    )

    assert second_receipt.sha256 != first_receipt.sha256
    with open_embedded_oracle_store_v2(oracle_path) as store:
        assert next(store.iter_rows()).scores[0] == pytest.approx(0.2)


def test_embedded_oracle_store_v2_replace_failure_preserves_previous_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score_paths, feature_paths, _, _ = _create_oracle_intermediates(tmp_path)
    oracle_path = tmp_path / "oracle-v2.sqlite3"
    create_embedded_oracle_store_v2(
        oracle_path,
        _oracle_metadata(),
        score_store_paths=score_paths,
        feature_store_paths=feature_paths,
    )
    old_bytes = oracle_path.read_bytes()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(store_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        create_embedded_oracle_store_v2(
            oracle_path,
            _oracle_metadata(),
            score_store_paths=score_paths,
            feature_store_paths=feature_paths,
        )

    assert oracle_path.read_bytes() == old_bytes
    assert not list(tmp_path.glob(".oracle-v2.sqlite3.staging-*"))


def test_feature_store_detects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "features.sqlite3"
    receipt = create_feature_store(path, _feature_metadata(0), _feature_rows(0))
    connection = sqlite3.connect(path)
    connection.execute("UPDATE rows SET feature=x'0000807f0000807f0000807f' WHERE row_index=0")
    connection.commit()
    connection.close()
    assert sha256_file(path) != receipt.sha256
    with pytest.raises(ValueError):
        open_feature_store(path, expected_sha256=receipt.sha256)


def test_store_target_rejects_existing_symlink_without_following_it(tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("unchanged", encoding="utf-8")
    target = tmp_path / "target.sqlite3"
    target.symlink_to(victim)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        create_feature_store(target, _feature_metadata(0), _feature_rows(0))
    assert victim.read_text(encoding="utf-8") == "unchanged"
    assert os.path.islink(target)
    assert FEATURE_STORE_SCHEMA.endswith("sqlite_v1")
