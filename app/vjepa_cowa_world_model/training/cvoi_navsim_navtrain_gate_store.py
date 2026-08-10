"""Bounded-memory SQLite artifacts for manual NavTrain Gate supervision.

The manual H4 pipeline writes intermediate feature and score stores per forced
horizon, then embeds all H0--H4 supervision in one self-contained Oracle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sqlite3
import stat
import struct
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

FEATURE_STORE_SCHEMA = "cvoi_navsim_v2_navtrain_feature_sqlite_v1"
SCORE_STORE_SCHEMA = "cvoi_navsim_v2_navtrain_ordinary_score_sqlite_v1"
MANUAL_ORACLE_STORE_SCHEMA_V2 = "cvoi_navsim_v2_navtrain_oracle_sqlite_v2"
NAVTRAIN_GATE_PROTOCOL_ID = "epdms_v2_one_stage_navtrain_gate_label_v1"
NAVTRAIN_GATE_TRAINING_BATCH_SIZE = 4096

_APPLICATION_ID = 0x43564F49
_INTERMEDIATE_USER_VERSION = 1
_MANUAL_ORACLE_USER_VERSION = 2
_USER_VERSION = _INTERMEDIATE_USER_VERSION
_EMBEDDED_FEATURE_POLICY = "embedded_h0_h4_float32_le_v1"
_HORIZONS = tuple(range(5))
_SHA256_LENGTH = 64
_SCORE_COMPONENTS = frozenset(
    {
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "driving_direction_compliance",
        "traffic_light_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "lane_keeping",
        "history_comfort",
        "two_frame_extended_comfort",
    }
)
_NULLABLE_COMPONENTS = frozenset({"two_frame_extended_comfort"})


@dataclass(frozen=True)
class FeatureRow:
    row_index: int
    token: str
    observation_key: str
    observed_feature_sha256: str
    features: tuple[float, ...]


@dataclass(frozen=True)
class FeatureStoreMetadata:
    protocol_id: str
    policy_id: str
    lineage: str
    horizon: int
    scenario_manifest_sha256: str
    metric_cache_inventory_sha256: str
    feature_schema: str
    feature_sources: tuple[str, ...]
    common_random_seed: int


@dataclass(frozen=True)
class ScoreIdentity:
    row_index: int
    token: str
    observation_key: str
    log_name: str


@dataclass(frozen=True)
class ScoreRow:
    row_index: int
    token: str
    observation_key: str
    log_name: str
    score: float


@dataclass(frozen=True)
class ScoreStoreMetadata:
    protocol_id: str
    policy_id: str
    lineage: str
    horizon: int
    scenario_manifest_sha256: str
    metric_cache_inventory_sha256: str
    source_path: Path
    source_sha256: str
    score_semantics: str


@dataclass(frozen=True)
class OracleStoreMetadata:
    protocol_id: str
    lineage: str
    scenario_manifest_sha256: str
    metric_cache_inventory_sha256: str
    lambda_grid: tuple[float, ...]


@dataclass(frozen=True)
class StoreReceipt:
    path: Path
    sha256: str
    row_count: int
    feature_dim: Optional[int] = None


@dataclass(frozen=True)
class OracleRow:
    record_id: int
    token: str
    observation_key: str
    log_name: str
    split: str
    split_index: int
    scores: tuple[float, ...]


@dataclass(frozen=True)
class FeatureBatchRow:
    record_id: int
    horizon: int
    features: tuple[float, ...]


@dataclass(frozen=True)
class TrainingBatchRow:
    record_id: int
    token: str
    log_name: str
    horizon: int
    lambda_compute: float
    features: tuple[float, ...]
    target_delta: float
    continue_target: bool


def sha256_file(path: Path) -> str:
    """Hash a regular file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_horizon(value: object, *, field: str = "horizon") -> int:
    if type(value) is not int or value not in _HORIZONS:
        raise ValueError(f"{field} must be an integer in [0, 4]")
    return value


def _finite(value: object, *, field: str, unit: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if unit and not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result


def _validate_feature_metadata(metadata: FeatureStoreMetadata) -> None:
    if type(metadata) is not FeatureStoreMetadata:
        raise TypeError("metadata must be FeatureStoreMetadata")
    for field in ("protocol_id", "policy_id", "lineage", "feature_schema"):
        _require_string(getattr(metadata, field), field=f"feature metadata {field}")
    _require_horizon(metadata.horizon)
    _require_sha256(metadata.scenario_manifest_sha256, field="scenario manifest SHA-256")
    _require_sha256(metadata.metric_cache_inventory_sha256, field="metric cache inventory SHA-256")
    if type(metadata.feature_sources) is not tuple or not metadata.feature_sources:
        raise ValueError("feature_sources must be a non-empty tuple")
    if len(set(metadata.feature_sources)) != len(metadata.feature_sources):
        raise ValueError("feature_sources must be unique")
    for source in metadata.feature_sources:
        _require_string(source, field="feature source")
    if type(metadata.common_random_seed) is not int or metadata.common_random_seed < 0:
        raise ValueError("common_random_seed must be a non-negative integer")


def _validate_score_metadata(metadata: ScoreStoreMetadata, *, verify_source: bool = True) -> None:
    if type(metadata) is not ScoreStoreMetadata:
        raise TypeError("metadata must be ScoreStoreMetadata")
    for field in ("protocol_id", "policy_id", "lineage", "score_semantics"):
        _require_string(getattr(metadata, field), field=f"score metadata {field}")
    _require_horizon(metadata.horizon)
    _require_sha256(metadata.scenario_manifest_sha256, field="scenario manifest SHA-256")
    _require_sha256(metadata.metric_cache_inventory_sha256, field="metric cache inventory SHA-256")
    source = _strict_regular_file(metadata.source_path, field="score source")
    expected = _require_sha256(metadata.source_sha256, field="score source SHA-256")
    if verify_source:
        descriptor, actual = _open_verified_descriptor(
            source,
            expected_sha256=expected,
            field="score source",
        )
        os.close(descriptor)
        if actual != expected:
            raise RuntimeError("verified score source digest differs")


def _validate_oracle_metadata(metadata: OracleStoreMetadata) -> None:
    if type(metadata) is not OracleStoreMetadata:
        raise TypeError("metadata must be OracleStoreMetadata")
    _require_string(metadata.protocol_id, field="Oracle protocol_id")
    _require_string(metadata.lineage, field="Oracle lineage")
    _require_sha256(metadata.scenario_manifest_sha256, field="scenario manifest SHA-256")
    _require_sha256(metadata.metric_cache_inventory_sha256, field="metric cache inventory SHA-256")
    if type(metadata.lambda_grid) is not tuple or not metadata.lambda_grid:
        raise ValueError("Oracle lambda_grid must be a non-empty tuple")
    values = tuple(_finite(value, field="Oracle lambda") for value in metadata.lambda_grid)
    if values != tuple(sorted(set(values))) or values[0] != 0.0:
        raise ValueError("Oracle lambda_grid must be unique, sorted, and start at zero")


def _strict_regular_file(path: Path, *, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError(f"{field} must be an absolute Path")
    if path.is_symlink():
        raise ValueError(f"{field} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise ValueError(f"{field} must exist") from error
    if resolved != path:
        raise ValueError(f"{field} must be canonical")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field} must be a regular file")
    return path


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        chunk = os.pread(descriptor, 1024 * 1024, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk)
        offset += len(chunk)


def _open_verified_descriptor(
    path: Path,
    *,
    expected_sha256: Optional[str],
    field: str,
) -> tuple[int, str]:
    path = _strict_regular_file(path, field=field)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{field} could not be opened without following symlinks") from error
    try:
        descriptor_stat = os.fstat(descriptor)
        path_stat = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(descriptor_stat.st_mode) or (
            descriptor_stat.st_dev,
            descriptor_stat.st_ino,
        ) != (path_stat.st_dev, path_stat.st_ino):
            raise ValueError(f"{field} changed while it was being opened")
        digest = _sha256_descriptor(descriptor)
        if expected_sha256 is not None and digest != _require_sha256(
            expected_sha256,
            field=f"{field} SHA-256",
        ):
            raise ValueError(f"{field} SHA-256 mismatch")
        return descriptor, digest
    except Exception:
        os.close(descriptor)
        raise


def _prepare_target(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or path.suffix != ".sqlite3":
        raise ValueError("SQLite store path must be an absolute .sqlite3 Path")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing SQLite store: {path}")
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise ValueError("SQLite store parent must be canonical and non-symlink")
    return path.parent / f".{path.name}.staging-{uuid.uuid4().hex}"


def _prepare_replace_target(path: Path) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or path.suffix != ".sqlite3":
        raise ValueError("SQLite store path must be an absolute .sqlite3 Path")
    if path.is_symlink():
        raise ValueError("replaceable SQLite store target must not be a symlink")
    if path.exists() and not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("replaceable SQLite store target must be absent or a regular file")
    path.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
    if path.parent.is_symlink() or path.parent.resolve(strict=True) != path.parent:
        raise ValueError("SQLite store parent must be canonical and non-symlink")
    return path.parent / f".{path.name}.staging-{uuid.uuid4().hex}"


def _connect_writer(path: Path, *, user_version: int) -> sqlite3.Connection:
    if type(user_version) is not int or user_version not in {
        _INTERMEDIATE_USER_VERSION,
        _MANUAL_ORACLE_USER_VERSION,
    }:
        raise ValueError("SQLite writer user_version is unsupported")
    connection = sqlite3.connect(path, isolation_level=None)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(f"PRAGMA application_id={_APPLICATION_ID}")
    connection.execute(f"PRAGMA user_version={user_version}")
    connection.execute("BEGIN IMMEDIATE")
    return connection


def _publish_store(staging: Path, target: Path) -> StoreReceipt:
    os.chmod(staging, 0o640, follow_symlinks=False)
    descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(staging, target, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing SQLite store: {target}") from error
    finally:
        staging.unlink(missing_ok=True)
    directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return StoreReceipt(target, sha256_file(target), 0, None)


def _publish_store_replace(staging: Path, target: Path) -> StoreReceipt:
    os.chmod(staging, 0o640, follow_symlinks=False)
    descriptor = os.open(staging, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(staging, target)
    directory = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return StoreReceipt(target, sha256_file(target), 0, None)


def _metadata_value(value: object) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, tuple):
        value = list(value)
    return json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _write_metadata(connection: sqlite3.Connection, values: Mapping[str, object]) -> None:
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID")
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES (?,?)",
        ((key, _metadata_value(value)) for key, value in sorted(values.items())),
    )


def _read_metadata(connection: sqlite3.Connection) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in connection.execute("SELECT key,value FROM metadata ORDER BY key"):
        if key in result:
            raise ValueError("SQLite metadata contains duplicate keys")
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("SQLite metadata contains invalid JSON") from error
        if _metadata_value(decoded) != value:
            raise ValueError("SQLite metadata must use canonical JSON")
        result[key] = decoded
    return result


def _open_reader(
    path: Path,
    *,
    expected_sha256: Optional[str],
    expected_user_version: int,
    field: str,
) -> tuple[Path, str, sqlite3.Connection]:
    path = _strict_regular_file(path, field=field)
    if path.suffix != ".sqlite3":
        raise ValueError(f"{field} must be a SQLite .sqlite3 store")
    for suffix in ("-wal", "-shm", "-journal"):
        if Path(f"{path}{suffix}").exists() or Path(f"{path}{suffix}").is_symlink():
            raise ValueError(f"{field} has a forbidden SQLite sidecar {suffix}")
    descriptor, digest = _open_verified_descriptor(
        path,
        expected_sha256=expected_sha256,
        field=field,
    )
    try:
        if os.pread(descriptor, 16, 0) != b"SQLite format 3\x00":
            raise ValueError(f"{field} must be a SQLite database")
        pinned_uri = Path(f"/proc/self/fd/{descriptor}").as_uri()
        connection = sqlite3.connect(f"{pinned_uri}?mode=ro&immutable=1", uri=True)
    finally:
        os.close(descriptor)
    connection.execute("PRAGMA query_only=ON")
    if connection.execute("PRAGMA application_id").fetchone() != (_APPLICATION_ID,):
        connection.close()
        raise ValueError(f"{field} application_id differs")
    if connection.execute("PRAGMA user_version").fetchone() != (expected_user_version,):
        connection.close()
        raise ValueError(f"{field} user_version differs")
    if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
        connection.close()
        raise ValueError(f"{field} failed SQLite quick_check")
    return path, digest, connection


def _require_exact_tables(connection: sqlite3.Connection, expected: set[str], *, field: str) -> None:
    rows = connection.execute(
        "SELECT type,name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name"
    ).fetchall()
    if rows != [("table", name) for name in sorted(expected)]:
        raise ValueError(f"{field} SQLite schema objects differ: {rows!r}")


def _require_exact_columns(
    connection: sqlite3.Connection,
    table: str,
    expected: Sequence[tuple[str, str, int, object, int]],
    *,
    field: str,
) -> None:
    actual = [tuple(row[1:]) for row in connection.execute(f"PRAGMA table_info({table})")]
    if actual != list(expected):
        raise ValueError(f"{field} SQLite {table} columns differ: {actual!r}")


def _require_unique_indexes(connection: sqlite3.Connection, table: str, count: int, *, field: str) -> None:
    actual = sorted((row[2], row[3], row[4]) for row in connection.execute(f"PRAGMA index_list({table})"))
    expected = [(1, "u", 0)] * count
    if actual != expected:
        raise ValueError(f"{field} SQLite {table} unique indexes differ: {actual!r}")


_METADATA_COLUMNS = (
    ("key", "TEXT", 1, None, 1),
    ("value", "TEXT", 1, None, 0),
)
_FEATURE_COLUMNS = (
    ("row_index", "INTEGER", 0, None, 1),
    ("token", "TEXT", 1, None, 0),
    ("observation_key", "TEXT", 1, None, 0),
    ("observed_feature_sha256", "TEXT", 1, None, 0),
    ("feature", "BLOB", 1, None, 0),
)
_SCORE_COLUMNS = (
    ("row_index", "INTEGER", 0, None, 1),
    ("token", "TEXT", 1, None, 0),
    ("observation_key", "TEXT", 1, None, 0),
    ("log_name", "TEXT", 1, None, 0),
    ("score", "REAL", 1, None, 0),
)
_EMBEDDED_FEATURE_COLUMNS = (
    ("record_id", "INTEGER", 1, None, 1),
    ("horizon", "INTEGER", 1, None, 2),
    ("feature", "BLOB", 1, None, 0),
)
_ORACLE_COLUMNS = (
    ("record_id", "INTEGER", 0, None, 1),
    ("token", "TEXT", 1, None, 0),
    ("observation_key", "TEXT", 1, None, 0),
    ("log_name", "TEXT", 1, None, 0),
    ("split", "TEXT", 1, None, 0),
    ("split_index", "INTEGER", 1, None, 0),
    ("score_h0", "REAL", 1, None, 0),
    ("score_h1", "REAL", 1, None, 0),
    ("score_h2", "REAL", 1, None, 0),
    ("score_h3", "REAL", 1, None, 0),
    ("score_h4", "REAL", 1, None, 0),
)


def _feature_blob(features: Sequence[float]) -> bytes:
    if isinstance(features, (str, bytes)) or not features:
        raise ValueError("features must be a non-empty numeric sequence")
    values = tuple(_finite(value, field="feature value") for value in features)
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_feature(blob: bytes, dimension: int) -> tuple[float, ...]:
    if not isinstance(blob, bytes) or len(blob) != dimension * 4:
        raise ValueError("feature BLOB dimension differs")
    values = struct.unpack(f"<{dimension}f", blob)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("feature BLOB contains a non-finite value")
    return tuple(float(value) for value in values)


def create_feature_store(
    path: Path,
    metadata: FeatureStoreMetadata,
    rows: Iterable[FeatureRow],
) -> StoreReceipt:
    """Stream one forced-horizon feature artifact to a create-once store."""

    _validate_feature_metadata(metadata)
    staging = _prepare_target(path)
    connection: Optional[sqlite3.Connection] = None
    row_count = 0
    feature_dim: Optional[int] = None
    try:
        connection = _connect_writer(staging, user_version=_INTERMEDIATE_USER_VERSION)
        connection.execute(
            "CREATE TABLE rows ("
            "row_index INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, observation_key TEXT NOT NULL UNIQUE, "
            "observed_feature_sha256 TEXT NOT NULL, feature BLOB NOT NULL)"
        )
        for row in rows:
            if type(row) is not FeatureRow or row.row_index != row_count:
                raise ValueError("feature row indices must be consecutive and zero-based")
            _require_string(row.token, field="feature token")
            _require_sha256(row.observation_key, field="feature observation_key")
            _require_sha256(row.observed_feature_sha256, field="observed feature SHA-256")
            blob = _feature_blob(row.features)
            dimension = len(blob) // 4
            if feature_dim is None:
                feature_dim = dimension
            elif feature_dim != dimension:
                raise ValueError("feature rows must share one dimension")
            connection.execute(
                "INSERT INTO rows(row_index,token,observation_key,observed_feature_sha256,feature) "
                "VALUES (?,?,?,?,?)",
                (row.row_index, row.token, row.observation_key, row.observed_feature_sha256, blob),
            )
            row_count += 1
        if row_count == 0 or feature_dim is None:
            raise ValueError("feature store requires at least one row")
        _write_metadata(
            connection,
            {
                "schema": FEATURE_STORE_SCHEMA,
                **asdict(metadata),
                "row_count": row_count,
                "feature_dim": feature_dim,
                "feature_encoding": "little_endian_float32_blob",
            },
        )
        connection.execute("COMMIT")
        connection.close()
        connection = None
        published = _publish_store(staging, path)
        return StoreReceipt(published.path, published.sha256, row_count, feature_dim)
    except Exception:
        if connection is not None:
            connection.close()
        staging.unlink(missing_ok=True)
        raise


class FeatureStore:
    def __init__(self, path: Path, sha256: str, connection: sqlite3.Connection, raw: Mapping[str, object]) -> None:
        string_fields = (
            "protocol_id",
            "policy_id",
            "lineage",
            "scenario_manifest_sha256",
            "metric_cache_inventory_sha256",
            "feature_schema",
        )
        if any(type(raw[field]) is not str for field in string_fields):
            raise ValueError("feature store string metadata types differ")
        if type(raw["horizon"]) is not int or type(raw["common_random_seed"]) is not int:
            raise ValueError("feature store integer metadata types differ")
        if not isinstance(raw["feature_sources"], list) or any(
            type(value) is not str for value in raw["feature_sources"]
        ):
            raise ValueError("feature store feature_sources metadata type differs")
        self.path = path
        self.sha256 = sha256
        self._connection = connection
        self.metadata = FeatureStoreMetadata(
            protocol_id=raw["protocol_id"],
            policy_id=raw["policy_id"],
            lineage=raw["lineage"],
            horizon=raw["horizon"],
            scenario_manifest_sha256=raw["scenario_manifest_sha256"],
            metric_cache_inventory_sha256=raw["metric_cache_inventory_sha256"],
            feature_schema=raw["feature_schema"],
            feature_sources=tuple(raw["feature_sources"]),
            common_random_seed=raw["common_random_seed"],
        )
        _validate_feature_metadata(self.metadata)
        self.row_count = int(raw["row_count"])
        self.feature_dim = int(raw["feature_dim"])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> FeatureStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def iter_identities(self) -> Iterator[tuple[int, str, str, str]]:
        yield from self._connection.execute(
            "SELECT row_index,token,observation_key,observed_feature_sha256 FROM rows ORDER BY row_index"
        )

    def iter_rows(self) -> Iterator[FeatureRow]:
        for index, token, observation_key, observed_sha, blob in self._connection.execute(
            "SELECT row_index,token,observation_key,observed_feature_sha256,feature FROM rows ORDER BY row_index"
        ):
            yield FeatureRow(index, token, observation_key, observed_sha, _unpack_feature(blob, self.feature_dim))

    def read_feature_batch(self, row_indices: Sequence[int]) -> tuple[tuple[float, ...], ...]:
        if isinstance(row_indices, (str, bytes)) or not row_indices:
            raise ValueError("feature batch indices must be non-empty")
        requested = tuple(row_indices)
        if any(type(index) is not int or index < 0 or index >= self.row_count for index in requested):
            raise IndexError("feature batch index is out of range")
        blobs: dict[int, bytes] = {}
        unique = tuple(dict.fromkeys(requested))
        for offset in range(0, len(unique), 900):
            chunk = unique[offset : offset + 900]
            placeholders = ",".join("?" for _ in chunk)
            for index, blob in self._connection.execute(
                f"SELECT row_index,feature FROM rows WHERE row_index IN ({placeholders})", chunk
            ):
                blobs[index] = blob
        if set(blobs) != set(unique):
            raise ValueError("feature batch coverage differs")
        return tuple(_unpack_feature(blobs[index], self.feature_dim) for index in requested)


def open_feature_store(path: Path, *, expected_sha256: Optional[str] = None) -> FeatureStore:
    path, digest, connection = _open_reader(
        path,
        expected_sha256=expected_sha256,
        expected_user_version=_INTERMEDIATE_USER_VERSION,
        field="feature store",
    )
    try:
        _require_exact_tables(connection, {"metadata", "rows"}, field="feature store")
        _require_exact_columns(connection, "metadata", _METADATA_COLUMNS, field="feature store")
        _require_exact_columns(connection, "rows", _FEATURE_COLUMNS, field="feature store")
        _require_unique_indexes(connection, "rows", 2, field="feature store")
        raw = _read_metadata(connection)
        expected_keys = {
            "schema",
            *FeatureStoreMetadata.__dataclass_fields__,
            "row_count",
            "feature_dim",
            "feature_encoding",
        }
        if set(raw) != expected_keys or raw["schema"] != FEATURE_STORE_SCHEMA:
            raise ValueError("feature store metadata schema differs")
        row_count = raw["row_count"]
        dimension = raw["feature_dim"]
        if type(row_count) is not int or row_count <= 0 or type(dimension) is not int or dimension <= 0:
            raise ValueError("feature store count/dimension metadata differs")
        stats = connection.execute(
            "SELECT COUNT(*),MIN(row_index),MAX(row_index),MIN(length(feature)),MAX(length(feature)),"
            "COUNT(DISTINCT token),COUNT(DISTINCT observation_key) FROM rows"
        ).fetchone()
        if stats != (row_count, 0, row_count - 1, dimension * 4, dimension * 4, row_count, row_count):
            raise ValueError("feature store row count or dimension differs")
        if raw["feature_encoding"] != "little_endian_float32_blob":
            raise ValueError("feature store encoding differs")
        return FeatureStore(path, digest, connection, raw)
    except Exception:
        connection.close()
        raise


def _score_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE rows ("
        "row_index INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, observation_key TEXT NOT NULL UNIQUE, "
        "log_name TEXT NOT NULL, score REAL NOT NULL CHECK(score>=0.0 AND score<=1.0))"
    )


def _score_metadata_values(
    metadata: ScoreStoreMetadata, *, row_count: int, aggregate_score: float
) -> dict[str, object]:
    return {
        "schema": SCORE_STORE_SCHEMA,
        **asdict(metadata),
        "source_path": str(metadata.source_path),
        "row_count": row_count,
        "aggregate_score": aggregate_score,
    }


def create_score_store(
    path: Path,
    metadata: ScoreStoreMetadata,
    rows: Iterable[ScoreRow],
    *,
    aggregate_score: float,
) -> StoreReceipt:
    """Stream already-validated ordinary scores into one store."""

    _validate_score_metadata(metadata)
    aggregate = _finite(aggregate_score, field="official aggregate score", unit=True)
    staging = _prepare_target(path)
    connection: Optional[sqlite3.Connection] = None
    row_count = 0
    try:
        connection = _connect_writer(staging, user_version=_INTERMEDIATE_USER_VERSION)
        _score_schema(connection)
        for row in rows:
            if type(row) is not ScoreRow or row.row_index != row_count:
                raise ValueError("score row indices must be consecutive and zero-based")
            _require_string(row.token, field="score token")
            _require_sha256(row.observation_key, field="score observation_key")
            _require_string(row.log_name, field="score log_name")
            score = _finite(row.score, field="ordinary score", unit=True)
            connection.execute(
                "INSERT INTO rows(row_index,token,observation_key,log_name,score) VALUES (?,?,?,?,?)",
                (row.row_index, row.token, row.observation_key, row.log_name, score),
            )
            row_count += 1
        if row_count == 0:
            raise ValueError("score store requires at least one row")
        _write_metadata(connection, _score_metadata_values(metadata, row_count=row_count, aggregate_score=aggregate))
        connection.execute("COMMIT")
        connection.close()
        connection = None
        published = _publish_store(staging, path)
        return StoreReceipt(published.path, published.sha256, row_count, None)
    except Exception:
        if connection is not None:
            connection.close()
        staging.unlink(missing_ok=True)
        raise


def _csv_number(value: str, *, field: str, unit: bool = False) -> float:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must not be blank")
    try:
        number = float(value.strip())
    except ValueError as error:
        raise ValueError(f"{field} must be numeric") from error
    return _finite(number, field=field, unit=unit)


def create_score_store_from_official_csv(
    path: Path,
    metadata: ScoreStoreMetadata,
    *,
    identities: Iterable[ScoreIdentity],
) -> StoreReceipt:
    """Validate and stream the scorer-owned CSV without a Python row map."""

    _validate_score_metadata(metadata, verify_source=False)
    staging = _prepare_target(path)
    connection: Optional[sqlite3.Connection] = None
    try:
        connection = _connect_writer(staging, user_version=_INTERMEDIATE_USER_VERSION)
        connection.execute(
            "CREATE TABLE expected ("
            "row_index INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, observation_key TEXT NOT NULL UNIQUE, "
            "log_name TEXT NOT NULL)"
        )
        identity_count = 0
        for identity in identities:
            if type(identity) is not ScoreIdentity or identity.row_index != identity_count:
                raise ValueError("score identities must be consecutive and zero-based")
            _require_string(identity.token, field="score identity token")
            _require_sha256(identity.observation_key, field="score identity observation_key")
            _require_string(identity.log_name, field="score identity log_name")
            connection.execute(
                "INSERT INTO expected(row_index,token,observation_key,log_name) VALUES (?,?,?,?)",
                (identity.row_index, identity.token, identity.observation_key, identity.log_name),
            )
            identity_count += 1
        if identity_count == 0:
            raise ValueError("score CSV import requires at least one identity")
        connection.execute("CREATE TABLE incoming (token TEXT PRIMARY KEY, score REAL NOT NULL)")
        aggregate: Optional[float] = None
        source_descriptor, _ = _open_verified_descriptor(
            metadata.source_path,
            expected_sha256=metadata.source_sha256,
            field="official score CSV",
        )
        with os.fdopen(source_descriptor, "r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            fieldnames = reader.fieldnames
            if not fieldnames or len(fieldnames) != len(set(fieldnames)):
                raise ValueError("official CSV requires a unique non-empty header")
            index_fields = [field for field in fieldnames if field in {"", "Unnamed: 0"}]
            if len(index_fields) > 1:
                raise ValueError("official CSV may contain at most one pandas index column")
            actual = set(fieldnames) - set(index_fields)
            expected_fields = {"token", "valid", "score", *_SCORE_COMPONENTS}
            if actual != expected_fields:
                raise ValueError("official CSV fields mismatch")
            index_field = index_fields[0] if index_fields else None
            for row_index, raw in enumerate(reader):
                if None in raw or any(value is None for value in raw.values()):
                    raise ValueError("official CSV has a malformed column count")
                if index_field is not None and raw[index_field] != str(row_index):
                    raise ValueError("official CSV pandas index must be consecutive and zero-based")
                token = _require_string(raw["token"], field="official CSV token")
                if raw["valid"].strip().lower() != "true":
                    raise ValueError(f"official CSV token {token!r} must have valid=true")
                score = _csv_number(raw["score"], field=f"official CSV token {token!r} score", unit=True)
                for component in _SCORE_COMPONENTS:
                    value = raw[component].strip()
                    if not value or value.lower() == "nan":
                        if component not in _NULLABLE_COMPONENTS:
                            raise ValueError(f"official CSV token {token!r} has null non-nullable component")
                    else:
                        _csv_number(value, field=f"official CSV token {token!r} component {component!r}")
                if token == "average_all_frames":
                    if aggregate is not None:
                        raise ValueError("official CSV contains duplicate aggregate rows")
                    aggregate = score
                else:
                    try:
                        connection.execute("INSERT INTO incoming(token,score) VALUES (?,?)", (token, score))
                    except sqlite3.IntegrityError as error:
                        raise ValueError(f"official CSV contains duplicate token {token!r}") from error
        if aggregate is None:
            raise ValueError("official CSV is missing average_all_frames")
        missing = connection.execute("SELECT token FROM expected EXCEPT SELECT token FROM incoming LIMIT 1").fetchone()
        extra = connection.execute("SELECT token FROM incoming EXCEPT SELECT token FROM expected LIMIT 1").fetchone()
        if missing is not None or extra is not None:
            raise ValueError(f"official CSV token coverage mismatch: missing={missing}, extra={extra}")
        _score_schema(connection)
        connection.execute(
            "INSERT INTO rows(row_index,token,observation_key,log_name,score) "
            "SELECT e.row_index,e.token,e.observation_key,e.log_name,i.score "
            "FROM expected e JOIN incoming i USING(token) ORDER BY e.row_index"
        )
        connection.execute("DROP TABLE incoming")
        connection.execute("DROP TABLE expected")
        _write_metadata(
            connection,
            _score_metadata_values(metadata, row_count=identity_count, aggregate_score=aggregate),
        )
        connection.execute("COMMIT")
        connection.close()
        connection = None
        published = _publish_store(staging, path)
        return StoreReceipt(published.path, published.sha256, identity_count, None)
    except (csv.Error, UnicodeError) as error:
        if connection is not None:
            connection.close()
        staging.unlink(missing_ok=True)
        raise ValueError(f"official score CSV is malformed: {error}") from error
    except Exception:
        if connection is not None:
            connection.close()
        staging.unlink(missing_ok=True)
        raise


class ScoreStore:
    def __init__(self, path: Path, sha256: str, connection: sqlite3.Connection, raw: Mapping[str, object]) -> None:
        string_fields = (
            "protocol_id",
            "policy_id",
            "lineage",
            "scenario_manifest_sha256",
            "metric_cache_inventory_sha256",
            "source_path",
            "source_sha256",
            "score_semantics",
        )
        if any(type(raw[field]) is not str for field in string_fields):
            raise ValueError("score store string metadata types differ")
        if type(raw["horizon"]) is not int:
            raise ValueError("score store horizon metadata type differs")
        self.path = path
        self.sha256 = sha256
        self._connection = connection
        self.metadata = ScoreStoreMetadata(
            protocol_id=raw["protocol_id"],
            policy_id=raw["policy_id"],
            lineage=raw["lineage"],
            horizon=raw["horizon"],
            scenario_manifest_sha256=raw["scenario_manifest_sha256"],
            metric_cache_inventory_sha256=raw["metric_cache_inventory_sha256"],
            source_path=Path(raw["source_path"]),
            source_sha256=raw["source_sha256"],
            score_semantics=raw["score_semantics"],
        )
        _validate_score_metadata(self.metadata)
        self.row_count = int(raw["row_count"])
        self.aggregate_score = float(raw["aggregate_score"])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ScoreStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def iter_rows(self) -> Iterator[ScoreRow]:
        for row in self._connection.execute(
            "SELECT row_index,token,observation_key,log_name,score FROM rows ORDER BY row_index"
        ):
            yield ScoreRow(row[0], row[1], row[2], row[3], float(row[4]))


def open_score_store(path: Path, *, expected_sha256: Optional[str] = None) -> ScoreStore:
    path, digest, connection = _open_reader(
        path,
        expected_sha256=expected_sha256,
        expected_user_version=_INTERMEDIATE_USER_VERSION,
        field="score store",
    )
    try:
        _require_exact_tables(connection, {"metadata", "rows"}, field="score store")
        _require_exact_columns(connection, "metadata", _METADATA_COLUMNS, field="score store")
        _require_exact_columns(connection, "rows", _SCORE_COLUMNS, field="score store")
        _require_unique_indexes(connection, "rows", 2, field="score store")
        raw = _read_metadata(connection)
        expected_keys = {"schema", *ScoreStoreMetadata.__dataclass_fields__, "row_count", "aggregate_score"}
        if set(raw) != expected_keys or raw["schema"] != SCORE_STORE_SCHEMA:
            raise ValueError("score store metadata schema differs")
        row_count = raw["row_count"]
        if type(row_count) is not int or row_count <= 0:
            raise ValueError("score store row_count differs")
        stats = connection.execute(
            "SELECT COUNT(*),MIN(row_index),MAX(row_index),COUNT(DISTINCT token),"
            "COUNT(DISTINCT observation_key) FROM rows"
        ).fetchone()
        if stats != (row_count, 0, row_count - 1, row_count, row_count):
            raise ValueError("score store row coverage differs")
        aggregate = _finite(raw["aggregate_score"], field="score store aggregate", unit=True)
        invalid = connection.execute(
            "SELECT 1 FROM rows WHERE score<0.0 OR score>1.0 OR score!=score LIMIT 1"
        ).fetchone()
        if invalid is not None:
            raise ValueError("score store contains an invalid score")
        raw = dict(raw)
        raw["aggregate_score"] = aggregate
        return ScoreStore(path, digest, connection, raw)
    except Exception:
        connection.close()
        raise


def _split(log_name: str) -> str:
    bucket = int(hashlib.sha256(log_name.encode("utf-8")).hexdigest(), 16) % 10
    return "dev" if bucket == 0 else "train"


def create_embedded_oracle_store_v2(
    path: Path,
    metadata: OracleStoreMetadata,
    *,
    score_store_paths: Mapping[int, Path],
    feature_store_paths: Mapping[int, Path],
) -> StoreReceipt:
    """Stream-join H0--H4 scores and features into a self-contained Oracle."""

    _validate_oracle_metadata(metadata)
    for field, value in (
        ("score_store_paths", score_store_paths),
        ("feature_store_paths", feature_store_paths),
    ):
        if not isinstance(value, Mapping) or set(value) != set(_HORIZONS):
            raise ValueError(f"{field} must contain exactly horizons 0--4")
    score_stores: list[ScoreStore] = []
    feature_stores: list[FeatureStore] = []
    staging: Optional[Path] = None
    connection: Optional[sqlite3.Connection] = None
    try:
        for horizon in _HORIZONS:
            score_stores.append(open_score_store(score_store_paths[horizon], expected_sha256=None))
        for horizon in _HORIZONS:
            feature_stores.append(open_feature_store(feature_store_paths[horizon], expected_sha256=None))
        row_count = score_stores[0].row_count
        feature_dim = feature_stores[0].feature_dim
        score_semantics = score_stores[0].metadata.score_semantics
        feature_contract = (
            feature_stores[0].metadata.feature_schema,
            feature_stores[0].metadata.feature_sources,
            feature_stores[0].metadata.common_random_seed,
        )
        for horizon in _HORIZONS:
            score = score_stores[horizon]
            feature = feature_stores[horizon]
            if score.metadata.horizon != horizon or feature.metadata.horizon != horizon:
                raise ValueError("store horizon differs from its map key")
            for common in (score.metadata, feature.metadata):
                if (
                    common.protocol_id != metadata.protocol_id
                    or common.lineage != metadata.lineage
                    or common.scenario_manifest_sha256 != metadata.scenario_manifest_sha256
                    or common.metric_cache_inventory_sha256 != metadata.metric_cache_inventory_sha256
                ):
                    raise ValueError("forced-horizon store metadata differs from Oracle metadata")
            if score.metadata.policy_id != feature.metadata.policy_id:
                raise ValueError("forced-horizon score and feature policy IDs differ")
            if score.metadata.score_semantics != score_semantics:
                raise ValueError("forced-horizon score semantics differ")
            if (
                feature.metadata.feature_schema,
                feature.metadata.feature_sources,
                feature.metadata.common_random_seed,
            ) != feature_contract:
                raise ValueError("forced-horizon feature contracts differ")
            if score.row_count != row_count or feature.row_count != row_count:
                raise ValueError("forced-horizon stores have different row counts")
            if feature.feature_dim != feature_dim:
                raise ValueError("forced-horizon feature dimensions differ")
        staging = _prepare_replace_target(path)
        connection = _connect_writer(staging, user_version=_MANUAL_ORACLE_USER_VERSION)
        connection.execute(
            "CREATE TABLE rows ("
            "record_id INTEGER PRIMARY KEY, token TEXT NOT NULL UNIQUE, observation_key TEXT NOT NULL UNIQUE, "
            "log_name TEXT NOT NULL, split TEXT NOT NULL CHECK(split IN ('train','dev')), "
            "split_index INTEGER NOT NULL, "
            "score_h0 REAL NOT NULL, score_h1 REAL NOT NULL, score_h2 REAL NOT NULL, score_h3 REAL NOT NULL, "
            "score_h4 REAL NOT NULL, UNIQUE(split,split_index))"
        )
        connection.execute(
            "CREATE TABLE features ("
            "record_id INTEGER NOT NULL, horizon INTEGER NOT NULL CHECK(horizon BETWEEN 0 AND 4), "
            "feature BLOB NOT NULL, PRIMARY KEY(record_id,horizon)) WITHOUT ROWID"
        )
        score_iterators = [store.iter_rows() for store in score_stores]
        feature_iterators = [store.iter_rows() for store in feature_stores]
        split_counts = {"train": 0, "dev": 0}
        for record_id in range(row_count):
            scores = [next(iterator) for iterator in score_iterators]
            features = [next(iterator) for iterator in feature_iterators]
            baseline = scores[0]
            score_identity = {(row.row_index, row.token, row.observation_key, row.log_name) for row in scores}
            feature_identity = {(row.row_index, row.token, row.observation_key) for row in features}
            observed_digests = {row.observed_feature_sha256 for row in features}
            if (
                score_identity != {(record_id, baseline.token, baseline.observation_key, baseline.log_name)}
                or feature_identity != {(record_id, baseline.token, baseline.observation_key)}
                or len(observed_digests) != 1
            ):
                raise ValueError("forced-horizon token/observation/feature identity differs")
            split = _split(baseline.log_name)
            split_index = split_counts[split]
            split_counts[split] += 1
            connection.execute(
                "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    baseline.token,
                    baseline.observation_key,
                    baseline.log_name,
                    split,
                    split_index,
                    *(row.score for row in scores),
                ),
            )
            connection.executemany(
                "INSERT INTO features(record_id,horizon,feature) VALUES (?,?,?)",
                ((record_id, horizon, _feature_blob(features[horizon].features)) for horizon in _HORIZONS),
            )
        sentinel = object()
        for iterator in (*score_iterators, *feature_iterators):
            if next(iterator, sentinel) is not sentinel:
                raise ValueError("forced-horizon store contains rows beyond its declared row count")
        _write_metadata(
            connection,
            {
                "schema": MANUAL_ORACLE_STORE_SCHEMA_V2,
                **asdict(metadata),
                "row_count": row_count,
                "feature_dim": feature_dim,
                "policy_ids_by_horizon": [store.metadata.policy_id for store in score_stores],
                "official_aggregate_scores": [store.aggregate_score for store in score_stores],
                "compute_costs": list(_HORIZONS),
                "feature_payload_policy": _EMBEDDED_FEATURE_POLICY,
            },
        )
        connection.execute("COMMIT")
        connection.close()
        connection = None
        published = _publish_store_replace(staging, path)
        staging = None
        return StoreReceipt(published.path, published.sha256, row_count, feature_dim)
    finally:
        if connection is not None:
            connection.close()
        if staging is not None:
            staging.unlink(missing_ok=True)
        for store in score_stores:
            store.close()
        for store in feature_stores:
            store.close()


class _OracleStoreTrainingMixin:
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()

    def __enter__(self) -> _OracleStoreTrainingMixin:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def iter_rows(self) -> Iterator[OracleRow]:
        for row in self._connection.execute(
            "SELECT record_id,token,observation_key,log_name,split,split_index,"
            "score_h0,score_h1,score_h2,score_h3,score_h4 FROM rows ORDER BY record_id"
        ):
            yield OracleRow(row[0], row[1], row[2], row[3], row[4], row[5], tuple(float(v) for v in row[6:]))

    def iter_split_record_ids(self, split: str) -> Iterator[int]:
        if split not in {"train", "dev"}:
            raise ValueError("Oracle split must be train or dev")
        for (record_id,) in self._connection.execute(
            "SELECT record_id FROM rows WHERE split=? ORDER BY split_index", (split,)
        ):
            yield record_id

    def _rows_for_ids(self, record_ids: Sequence[int]) -> dict[int, OracleRow]:
        unique = tuple(dict.fromkeys(record_ids))
        result: dict[int, OracleRow] = {}
        for offset in range(0, len(unique), 900):
            chunk = unique[offset : offset + 900]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT record_id,token,observation_key,log_name,split,split_index,"
                "score_h0,score_h1,score_h2,score_h3,score_h4 FROM rows "
                f"WHERE record_id IN ({placeholders})"
            )
            for row in self._connection.execute(query, chunk):
                result[row[0]] = OracleRow(
                    row[0], row[1], row[2], row[3], row[4], row[5], tuple(float(v) for v in row[6:])
                )
        if set(result) != set(unique):
            raise ValueError("Oracle batch row coverage differs")
        return result

    def read_training_batch(
        self,
        *,
        record_ids: Sequence[int],
        horizons: Sequence[int],
        lambda_computes: Sequence[float],
    ) -> tuple[TrainingBatchRow, ...]:
        if len(record_ids) != len(horizons) or len(record_ids) != len(lambda_computes) or not record_ids:
            raise ValueError("training batch inputs must be equally sized and non-empty")
        rows = self._rows_for_ids(record_ids)
        features = self.read_feature_batch(record_ids=record_ids, horizons=horizons)
        result = []
        for record_id, horizon, lambda_compute, feature_row in zip(record_ids, horizons, lambda_computes, features):
            horizon = _require_horizon(horizon)
            if horizon == 4:
                raise ValueError("Gate training examples require decision horizons H0--H3")
            lambda_compute = _finite(lambda_compute, field="lambda_compute")
            row = rows[record_id]
            utilities = tuple(score - lambda_compute * cost for score, cost in zip(row.scores, self.compute_costs))
            target_delta = max(utilities[horizon + 1 :]) - utilities[horizon]
            result.append(
                TrainingBatchRow(
                    record_id=record_id,
                    token=row.token,
                    log_name=row.log_name,
                    horizon=horizon,
                    lambda_compute=lambda_compute,
                    features=(*feature_row.features, lambda_compute),
                    target_delta=target_delta,
                    continue_target=target_delta > 0.0,
                )
            )
        return tuple(result)


class EmbeddedOracleStoreV2(_OracleStoreTrainingMixin):
    def __init__(
        self,
        path: Path,
        sha256: str,
        connection: sqlite3.Connection,
        raw: Mapping[str, object],
    ) -> None:
        string_fields = (
            "protocol_id",
            "lineage",
            "scenario_manifest_sha256",
            "metric_cache_inventory_sha256",
        )
        if any(type(raw[field]) is not str for field in string_fields):
            raise ValueError("embedded Oracle store string metadata types differ")
        if not isinstance(raw["lambda_grid"], list) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw["lambda_grid"]
        ):
            raise ValueError("embedded Oracle store lambda_grid metadata type differs")
        self.path = path
        self.sha256 = sha256
        self._connection = connection
        self._closed = False
        self.metadata = OracleStoreMetadata(
            protocol_id=raw["protocol_id"],
            lineage=raw["lineage"],
            scenario_manifest_sha256=raw["scenario_manifest_sha256"],
            metric_cache_inventory_sha256=raw["metric_cache_inventory_sha256"],
            lambda_grid=tuple(float(value) for value in raw["lambda_grid"]),
        )
        _validate_oracle_metadata(self.metadata)
        self.row_count = int(raw["row_count"])
        self.feature_dim = int(raw["feature_dim"])
        self.policy_ids_by_horizon = tuple(raw["policy_ids_by_horizon"])
        self.official_aggregate_scores = tuple(float(value) for value in raw["official_aggregate_scores"])
        self.aggregate_scores = self.official_aggregate_scores
        self.compute_costs = tuple(float(value) for value in raw["compute_costs"])
        self.feature_payload_policy = raw["feature_payload_policy"]

    def read_feature_batch(
        self,
        *,
        record_ids: Sequence[int],
        horizons: Sequence[int],
    ) -> tuple[FeatureBatchRow, ...]:
        if len(record_ids) != len(horizons) or not record_ids:
            raise ValueError("feature batch record_ids/horizons must be equally sized and non-empty")
        requested: list[tuple[int, int]] = []
        for record_id, horizon in zip(record_ids, horizons):
            if type(record_id) is not int or record_id < 0 or record_id >= self.row_count:
                raise IndexError("Oracle feature record_id is out of range")
            requested.append((record_id, _require_horizon(horizon)))
        unique = tuple(dict.fromkeys(requested))
        blobs: dict[tuple[int, int], bytes] = {}
        for offset in range(0, len(unique), 450):
            chunk = unique[offset : offset + 450]
            placeholders = ",".join("(?,?)" for _ in chunk)
            parameters = tuple(value for pair in chunk for value in pair)
            query = "SELECT record_id,horizon,feature FROM features " f"WHERE (record_id,horizon) IN ({placeholders})"
            for record_id, horizon, blob in self._connection.execute(query, parameters):
                blobs[(record_id, horizon)] = blob
        if set(blobs) != set(unique):
            raise ValueError("embedded Oracle feature batch coverage differs")
        decoded = {key: _unpack_feature(blob, self.feature_dim) for key, blob in blobs.items()}
        return tuple(
            FeatureBatchRow(record_id, horizon, decoded[(record_id, horizon)]) for record_id, horizon in requested
        )


def _validate_embedded_oracle_metadata(raw: Mapping[str, object]) -> tuple[dict[str, object], int, int]:
    expected_keys = {
        "schema",
        *OracleStoreMetadata.__dataclass_fields__,
        "row_count",
        "feature_dim",
        "policy_ids_by_horizon",
        "official_aggregate_scores",
        "compute_costs",
        "feature_payload_policy",
    }
    if set(raw) != expected_keys or raw["schema"] != MANUAL_ORACLE_STORE_SCHEMA_V2:
        raise ValueError("embedded Oracle store metadata schema differs")
    row_count = raw["row_count"]
    feature_dim = raw["feature_dim"]
    if type(row_count) is not int or row_count <= 0 or type(feature_dim) is not int or feature_dim <= 0:
        raise ValueError("embedded Oracle store row_count/feature_dim differs")
    string_fields = (
        "protocol_id",
        "lineage",
        "scenario_manifest_sha256",
        "metric_cache_inventory_sha256",
    )
    if any(type(raw[field]) is not str for field in string_fields):
        raise ValueError("embedded Oracle store string metadata types differ")
    lambda_grid = raw["lambda_grid"]
    if not isinstance(lambda_grid, list) or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in lambda_grid
    ):
        raise ValueError("embedded Oracle store lambda_grid metadata type differs")
    oracle_metadata = OracleStoreMetadata(
        protocol_id=raw["protocol_id"],
        lineage=raw["lineage"],
        scenario_manifest_sha256=raw["scenario_manifest_sha256"],
        metric_cache_inventory_sha256=raw["metric_cache_inventory_sha256"],
        lambda_grid=tuple(float(value) for value in lambda_grid),
    )
    _validate_oracle_metadata(oracle_metadata)
    for key in ("policy_ids_by_horizon", "official_aggregate_scores", "compute_costs"):
        if not isinstance(raw[key], list) or len(raw[key]) != len(_HORIZONS):
            raise ValueError(f"embedded Oracle store {key} must contain H0--H4")
    policy_ids = tuple(
        _require_string(value, field="embedded Oracle policy ID") for value in raw["policy_ids_by_horizon"]
    )
    aggregates = tuple(
        _finite(value, field="embedded Oracle official aggregate score", unit=True)
        for value in raw["official_aggregate_scores"]
    )
    costs = tuple(_finite(value, field="embedded Oracle compute cost") for value in raw["compute_costs"])
    if costs != tuple(float(horizon) for horizon in _HORIZONS):
        raise ValueError("embedded Oracle compute costs must be rollout counts H0--H4")
    if raw["feature_payload_policy"] != _EMBEDDED_FEATURE_POLICY:
        raise ValueError("embedded Oracle feature payload policy differs")
    normalized = dict(raw)
    normalized["policy_ids_by_horizon"] = list(policy_ids)
    normalized["official_aggregate_scores"] = list(aggregates)
    normalized["compute_costs"] = list(costs)
    return normalized, row_count, feature_dim


def _validate_embedded_oracle_rows(connection: sqlite3.Connection, *, row_count: int) -> None:
    stats = connection.execute(
        "SELECT COUNT(*),MIN(record_id),MAX(record_id),COUNT(DISTINCT token),"
        "COUNT(DISTINCT observation_key),COUNT(DISTINCT split || ':' || split_index) FROM rows"
    ).fetchone()
    if stats != (row_count, 0, row_count - 1, row_count, row_count, row_count):
        raise ValueError("embedded Oracle store row coverage differs")
    split_counts = {"train": 0, "dev": 0}
    for expected_record_id, row in enumerate(
        connection.execute(
            "SELECT record_id,token,observation_key,log_name,split,split_index,"
            "score_h0,score_h1,score_h2,score_h3,score_h4 FROM rows ORDER BY record_id"
        )
    ):
        record_id, token, observation_key, log_name, split, split_index, *scores = row
        if type(record_id) is not int or record_id != expected_record_id:
            raise ValueError("embedded Oracle record IDs must be consecutive and zero-based")
        _require_string(token, field="embedded Oracle token")
        _require_sha256(observation_key, field="embedded Oracle observation_key")
        _require_string(log_name, field="embedded Oracle log_name")
        if type(split) is not str or split not in split_counts or split != _split(log_name):
            raise ValueError("embedded Oracle split differs")
        if type(split_index) is not int or split_index != split_counts[split]:
            raise ValueError("embedded Oracle split indices must be consecutive and zero-based")
        split_counts[split] += 1
        for horizon, score in enumerate(scores):
            _finite(score, field=f"embedded Oracle H{horizon} score", unit=True)


def _validate_embedded_oracle_features(
    connection: sqlite3.Connection,
    *,
    row_count: int,
    feature_dim: int,
) -> None:
    stats = connection.execute("SELECT COUNT(*),MIN(record_id),MAX(record_id) FROM features").fetchone()
    if stats != (row_count * len(_HORIZONS), 0, row_count - 1):
        raise ValueError("embedded Oracle feature row coverage differs")
    grouped = connection.execute(
        "SELECT horizon,COUNT(*),MIN(record_id),MAX(record_id) " "FROM features GROUP BY horizon ORDER BY horizon"
    ).fetchall()
    expected_grouped = [(horizon, row_count, 0, row_count - 1) for horizon in _HORIZONS]
    if grouped != expected_grouped:
        raise ValueError("embedded Oracle feature horizon coverage differs")
    if connection.execute("SELECT 1 FROM features WHERE typeof(feature)!='blob' LIMIT 1").fetchone() is not None:
        raise ValueError("embedded Oracle feature payload must be a BLOB")
    if (
        connection.execute(
            "SELECT 1 FROM features WHERE length(feature)!=? LIMIT 1",
            (feature_dim * 4,),
        ).fetchone()
        is not None
    ):
        raise ValueError("embedded Oracle feature BLOB dimension differs")
    feature_count = 0
    for record_id, horizon, blob in connection.execute(
        "SELECT record_id,horizon,feature FROM features ORDER BY record_id,horizon"
    ):
        expected_record_id, expected_horizon = divmod(feature_count, len(_HORIZONS))
        if (
            type(record_id) is not int
            or type(horizon) is not int
            or (record_id, horizon) != (expected_record_id, expected_horizon)
        ):
            raise ValueError("embedded Oracle feature identities differ")
        _unpack_feature(blob, feature_dim)
        feature_count += 1
    if feature_count != row_count * len(_HORIZONS):
        raise ValueError("embedded Oracle feature scan coverage differs")


def open_embedded_oracle_store_v2(
    path: Path,
    *,
    expected_sha256: Optional[str] = None,
) -> EmbeddedOracleStoreV2:
    """Open and fully validate a self-contained manual Oracle v2 store."""

    path, digest, connection = _open_reader(
        path,
        expected_sha256=expected_sha256,
        expected_user_version=_MANUAL_ORACLE_USER_VERSION,
        field="embedded Oracle store",
    )
    try:
        _require_exact_tables(connection, {"features", "metadata", "rows"}, field="embedded Oracle store")
        _require_exact_columns(connection, "metadata", _METADATA_COLUMNS, field="embedded Oracle store")
        _require_exact_columns(connection, "rows", _ORACLE_COLUMNS, field="embedded Oracle store")
        _require_exact_columns(
            connection,
            "features",
            _EMBEDDED_FEATURE_COLUMNS,
            field="embedded Oracle store",
        )
        _require_unique_indexes(connection, "rows", 3, field="embedded Oracle store")
        feature_indexes = [(row[2], row[3], row[4]) for row in connection.execute("PRAGMA index_list(features)")]
        if feature_indexes != [(1, "pk", 0)]:
            raise ValueError(f"embedded Oracle SQLite features primary key differs: {feature_indexes!r}")
        feature_sql_row = connection.execute(
            "SELECT sql FROM sqlite_schema WHERE type='table' AND name='features'"
        ).fetchone()
        if (
            feature_sql_row is None
            or type(feature_sql_row[0]) is not str
            or "WITHOUTROWID" not in "".join(feature_sql_row[0].upper().split())
        ):
            raise ValueError("embedded Oracle features table must use WITHOUT ROWID")
        raw, row_count, feature_dim = _validate_embedded_oracle_metadata(_read_metadata(connection))
        _validate_embedded_oracle_rows(connection, row_count=row_count)
        _validate_embedded_oracle_features(
            connection,
            row_count=row_count,
            feature_dim=feature_dim,
        )
        return EmbeddedOracleStoreV2(path, digest, connection, raw)
    except Exception:
        connection.close()
        raise


__all__ = [
    "FEATURE_STORE_SCHEMA",
    "MANUAL_ORACLE_STORE_SCHEMA_V2",
    "NAVTRAIN_GATE_PROTOCOL_ID",
    "NAVTRAIN_GATE_TRAINING_BATCH_SIZE",
    "SCORE_STORE_SCHEMA",
    "EmbeddedOracleStoreV2",
    "FeatureBatchRow",
    "FeatureRow",
    "FeatureStoreMetadata",
    "OracleRow",
    "OracleStoreMetadata",
    "ScoreIdentity",
    "ScoreRow",
    "ScoreStoreMetadata",
    "StoreReceipt",
    "TrainingBatchRow",
    "create_embedded_oracle_store_v2",
    "create_feature_store",
    "create_score_store",
    "create_score_store_from_official_csv",
    "open_embedded_oracle_store_v2",
    "open_feature_store",
    "open_score_store",
    "sha256_file",
]
