"""NavSim-free records and reader for raw V2 scenario authorities."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType

from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import V2_PROTOCOL_ID, get_cvoi_navsim_metric_protocol

_AUTHORITY_SCHEMA = "cvoi_navsim_raw_v2_authority_v1"
_SCENARIO_SCHEMA = "cvoi_navsim_scenario_v1"
_INVENTORY_SCHEMA = "cvoi_navsim_metric_cache_inventory_v1"
_TOKEN_SUBSET_SCHEMA = "cvoi_navsim_token_subset_v1"
_AUTHORITY_FILES = frozenset({"manifest.json", "scenario_manifest.jsonl", "metric_cache_inventory.json"})
_SPLITS = frozenset({"navtrain", "navtest"})
_LOWER_HEX = frozenset("0123456789abcdef")
_SENSOR_FIELDS = frozenset(
    {
        "num_history_frames",
        "cam_f0",
        "cam_l0",
        "cam_l1",
        "cam_l2",
        "cam_r0",
        "cam_r1",
        "cam_r2",
        "cam_b0",
        "lidar_pc",
    }
)


def _require_nonempty_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without surrounding whitespace")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(character not in _LOWER_HEX for character in value):
        raise ValueError(f"{field} must be exactly 64 lowercase hexadecimal characters")
    return value


def validate_cvoi_navsim_scenario_token(value: object) -> str:
    """Return one token safe to use as a metric-cache directory entry."""

    token = _require_nonempty_string(value, field="scenario token")
    if token in {".", ".."} or "/" in token or "\\" in token or Path(token).name != token:
        raise ValueError(f"scenario token must be one safe directory entry, got {token!r}")
    return token


@dataclass(frozen=True)
class CvoiNavSimScenario:
    """One V2 scenario joined to a decoded current-camera observation."""

    scenario_token: str
    observation_key: str
    log_name: str
    current_camera_data_path: str

    def __post_init__(self) -> None:
        validate_cvoi_navsim_scenario_token(self.scenario_token)
        _require_sha256(self.observation_key, field="observation_key")
        _require_nonempty_string(self.log_name, field="log_name")
        _require_nonempty_string(self.current_camera_data_path, field="current_camera_data_path")


@dataclass(frozen=True)
class CvoiNavSimScenarioManifest:
    """A validated V2 token/observation-key bijection."""

    protocol_id: str
    scenarios: tuple[CvoiNavSimScenario, ...]
    _scenarios_by_token: Mapping[str, CvoiNavSimScenario] = dataclass_field(init=False, repr=False, compare=False)
    _tokens_by_observation_key: Mapping[str, str] = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        get_cvoi_navsim_metric_protocol(self.protocol_id)
        if type(self.scenarios) is not tuple or not self.scenarios:
            raise ValueError("scenarios must be a non-empty tuple")
        if any(not isinstance(row, CvoiNavSimScenario) for row in self.scenarios):
            raise ValueError("scenarios must contain only CvoiNavSimScenario records")
        tokens = tuple(row.scenario_token for row in self.scenarios)
        keys = tuple(row.observation_key for row in self.scenarios)
        if tokens != tuple(sorted(tokens)) or len(set(tokens)) != len(tokens):
            raise ValueError("scenario tokens must be unique and sorted")
        if len(set(keys)) != len(keys):
            raise ValueError("observation keys must be unique across scenario rows")
        object.__setattr__(
            self,
            "_scenarios_by_token",
            MappingProxyType({row.scenario_token: row for row in self.scenarios}),
        )
        object.__setattr__(
            self,
            "_tokens_by_observation_key",
            MappingProxyType({row.observation_key: row.scenario_token for row in self.scenarios}),
        )

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(row.scenario_token for row in self.scenarios)

    @property
    def scenarios_by_token(self) -> Mapping[str, CvoiNavSimScenario]:
        return self._scenarios_by_token

    @property
    def tokens_by_observation_key(self) -> Mapping[str, str]:
        return self._tokens_by_observation_key

    def scenario_for_token(self, scenario_token: str) -> CvoiNavSimScenario:
        try:
            return self._scenarios_by_token[scenario_token]
        except KeyError as exc:
            raise KeyError(f"unknown CVoI NavSim scenario token {scenario_token!r}") from exc

    def token_for_observation_key(self, key: str) -> str:
        try:
            return self._tokens_by_observation_key[key]
        except KeyError as exc:
            raise KeyError(f"unknown CVoI NavSim observation key {key!r}") from exc


@dataclass(frozen=True)
class CvoiNavSimMetricCacheEntry:
    """One raw V2 metric-cache file bound to its content digest."""

    scenario_token: str
    path: Path
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        validate_cvoi_navsim_scenario_token(self.scenario_token)
        if not isinstance(self.path, Path) or not self.path.is_absolute():
            raise ValueError("metric-cache entry path must be an absolute Path")
        _require_sha256(self.sha256, field="metric-cache entry sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("metric-cache entry size_bytes must be a non-negative integer")


@dataclass(frozen=True)
class CvoiNavSimMetricCacheInventory:
    """Strict raw V2 metadata and per-token metric-cache inventory."""

    protocol_id: str
    metric_cache_root: Path
    metadata_path: Path
    metadata_sha256: str
    entries: tuple[CvoiNavSimMetricCacheEntry, ...]

    def __post_init__(self) -> None:
        get_cvoi_navsim_metric_protocol(self.protocol_id)
        if not isinstance(self.metric_cache_root, Path) or not self.metric_cache_root.is_absolute():
            raise ValueError("metric_cache_root must be an absolute Path")
        if not isinstance(self.metadata_path, Path) or not self.metadata_path.is_absolute():
            raise ValueError("metadata_path must be an absolute Path")
        _require_sha256(self.metadata_sha256, field="metadata_sha256")
        if type(self.entries) is not tuple or not self.entries:
            raise ValueError("metric-cache entries must be a non-empty tuple")
        if any(not isinstance(entry, CvoiNavSimMetricCacheEntry) for entry in self.entries):
            raise ValueError("metric-cache entries must contain typed records")
        tokens = tuple(entry.scenario_token for entry in self.entries)
        if tokens != tuple(sorted(tokens)) or len(set(tokens)) != len(tokens):
            raise ValueError("metric-cache entry tokens must be unique and sorted")

    @property
    def tokens(self) -> tuple[str, ...]:
        return tuple(entry.scenario_token for entry in self.entries)


@dataclass(frozen=True)
class CvoiNavSimRawV2Authority:
    """Typed view of one receipt-free raw V2 three-file authority."""

    root: Path
    protocol_id: str
    split: str
    data_root: Path
    navsim_exp_root: Path
    maps_root: Path
    metric_cache_root: Path
    devkit_root: Path
    scenario_manifest: CvoiNavSimScenarioManifest
    metric_cache_inventory: CvoiNavSimMetricCacheInventory

    def __post_init__(self) -> None:
        get_cvoi_navsim_metric_protocol(self.protocol_id)
        _require_split(self.split, field="split")
        if self.scenario_manifest.protocol_id != self.protocol_id:
            raise ValueError("scenario manifest protocol must match raw authority")
        if self.metric_cache_inventory.protocol_id != self.protocol_id:
            raise ValueError("metric-cache inventory protocol must match raw authority")
        if self.metric_cache_inventory.metric_cache_root != self.metric_cache_root:
            raise ValueError("metric-cache inventory root must match raw authority")
        if self.scenario_manifest.tokens != self.metric_cache_inventory.tokens:
            raise ValueError("scenario and metric-cache inventory tokens must match exactly")

    @property
    def metric_cache_paths(self) -> Mapping[str, Path]:
        return MappingProxyType({entry.scenario_token: entry.path for entry in self.metric_cache_inventory.entries})


def _unique_tokens(values: Iterable[object], *, label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a non-empty iterable of scenario tokens")
    tokens: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = validate_cvoi_navsim_scenario_token(value)
        if token in seen:
            raise ValueError(f"{label} contains duplicate token {token!r}")
        seen.add(token)
        tokens.append(token)
    if not tokens:
        raise ValueError(f"{label} must not be empty")
    return tuple(sorted(tokens))


def build_cvoi_navsim_scenario_manifest(
    *,
    protocol_id: str,
    rows: Iterable[CvoiNavSimScenario],
    expected_metric_cache_tokens: Iterable[str],
) -> CvoiNavSimScenarioManifest:
    """Validate a raw V2 token/key bijection without importing NavSim."""

    get_cvoi_navsim_metric_protocol(protocol_id)
    if isinstance(rows, (str, bytes)):
        raise ValueError("rows must be a non-empty iterable of CvoiNavSimScenario records")
    ordered_rows = tuple(
        sorted(rows, key=lambda row: row.scenario_token if isinstance(row, CvoiNavSimScenario) else "")
    )
    if not ordered_rows or any(not isinstance(row, CvoiNavSimScenario) for row in ordered_rows):
        raise ValueError("rows must contain only CvoiNavSimScenario records")
    manifest = CvoiNavSimScenarioManifest(protocol_id=protocol_id, scenarios=ordered_rows)
    cache_tokens = _unique_tokens(expected_metric_cache_tokens, label="metric-cache tokens")
    if manifest.tokens != cache_tokens:
        raise ValueError(
            "metric-cache tokens must match scenario tokens exactly; "
            f"scenarios={manifest.tokens}, metric_cache={cache_tokens}"
        )
    return manifest


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"JSON contains forbidden non-finite value {value!r}")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _parse_canonical_json(raw: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be valid duplicate-free UTF-8 JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must contain a JSON object")
    if raw != _canonical_json_bytes(value):
        raise ValueError(f"{label} must use canonical compact JSON without a trailing newline")
    return value


def _exact_mapping(value: object, *, fields: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    actual = set(value)
    if any(type(key) is not str for key in actual):
        raise ValueError(f"{label} keys must be strings")
    missing = fields - actual
    unknown = actual - fields
    if missing or unknown:
        raise ValueError(f"{label} fields mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")
    return value  # type: ignore[return-value]


def _require_split(value: object, *, field: str) -> str:
    if type(value) is not str or value not in _SPLITS:
        raise ValueError(f"{field} must be exactly 'navtrain' or 'navtest'")
    return value


def _strict_directory(value: object, *, field: str) -> Path:
    if not isinstance(value, Path) or not value.is_absolute() or value.is_symlink():
        raise ValueError(f"{field} must be an absolute canonical non-symlink directory")
    try:
        resolved = value.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{field} must be an existing directory: {value}") from exc
    if resolved != value or not resolved.is_dir():
        raise ValueError(f"{field} must be an absolute canonical non-symlink directory")
    return resolved


def _manifest_directory(value: object, *, field: str) -> Path:
    if type(value) is not str:
        raise ValueError(f"{field} must record an absolute canonical directory")
    return _strict_directory(Path(value), field=field)


def _strict_file(value: object, *, field: str) -> Path:
    path = Path(value) if isinstance(value, str) else value
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{field} must be an absolute canonical non-symlink regular file")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{field} must be an existing regular file: {path}") from exc
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{field} must be an absolute canonical non-symlink regular file")
    return resolved


def _source_record(value: object, *, label: str) -> Path:
    record = _exact_mapping(value, fields=frozenset({"path", "sha256"}), label=label)
    path = _strict_file(record["path"], field=f"{label}.path")
    expected = _require_sha256(record["sha256"], field=f"{label}.sha256")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} sha256 digest mismatch: expected {expected}, got {actual}")
    return path


def _parse_tokens(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a canonical sorted JSON list")
    tokens = _unique_tokens(value, label=label)
    if tuple(value) != tokens:
        raise ValueError(f"{label} must be a canonical sorted JSON list")
    return tokens


def _parse_scenario_manifest(raw: bytes, *, expected_tokens: tuple[str, ...]) -> CvoiNavSimScenarioManifest:
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("scenario_manifest.jsonl must be non-empty and end with a newline")
    rows: list[CvoiNavSimScenario] = []
    fields = frozenset(
        {
            "schema",
            "protocol_id",
            "scenario_token",
            "observation_key",
            "log_name",
            "current_camera_data_path",
        }
    )
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if line == b"\n" or not line.endswith(b"\n"):
            raise ValueError(f"scenario_manifest.jsonl row {index} is blank or lacks its newline")
        row = _exact_mapping(
            _parse_canonical_json(line[:-1], label=f"scenario_manifest.jsonl row {index}"),
            fields=fields,
            label=f"scenario_manifest.jsonl row {index}",
        )
        if row["schema"] != _SCENARIO_SCHEMA or row["protocol_id"] != V2_PROTOCOL_ID:
            raise ValueError(f"scenario_manifest.jsonl row {index} must use the raw V2 schema")
        rows.append(
            CvoiNavSimScenario(
                scenario_token=row["scenario_token"],  # type: ignore[arg-type]
                observation_key=row["observation_key"],  # type: ignore[arg-type]
                log_name=row["log_name"],  # type: ignore[arg-type]
                current_camera_data_path=row["current_camera_data_path"],  # type: ignore[arg-type]
            )
        )
    manifest = build_cvoi_navsim_scenario_manifest(
        protocol_id=V2_PROTOCOL_ID,
        rows=rows,
        expected_metric_cache_tokens=expected_tokens,
    )
    if tuple(row.scenario_token for row in rows) != expected_tokens:
        raise ValueError("scenario_manifest.jsonl rows must exactly follow token_inventory")
    return manifest


def _parse_metric_cache_inventory(
    raw: bytes,
    *,
    metric_cache_root: Path,
    expected_tokens: tuple[str, ...],
) -> CvoiNavSimMetricCacheInventory:
    inventory = _exact_mapping(
        _parse_canonical_json(raw, label="metric_cache_inventory.json"),
        fields=frozenset({"schema", "protocol_id", "metric_cache_root", "metadata", "tokens", "entries"}),
        label="metric_cache_inventory.json",
    )
    if inventory["schema"] != _INVENTORY_SCHEMA or inventory["protocol_id"] != V2_PROTOCOL_ID:
        raise ValueError("metric_cache_inventory.json must use the raw V2 inventory schema")
    if inventory["metric_cache_root"] != str(metric_cache_root):
        raise ValueError("metric_cache_inventory.json root differs from manifest.json")
    if _parse_tokens(inventory["tokens"], label="metric-cache inventory tokens") != expected_tokens:
        raise ValueError("metric-cache inventory tokens differ from manifest token_inventory")

    metadata = _exact_mapping(
        inventory["metadata"],
        fields=frozenset({"path", "sha256", "header", "row_count"}),
        label="metric_cache_inventory.json metadata",
    )
    metadata_path = _strict_file(metadata["path"], field="metric-cache metadata path")
    if metadata_path.parent != metric_cache_root / "metadata" or metadata_path.suffix != ".csv":
        raise ValueError("metric-cache metadata must be one CSV directly below metric_cache_root/metadata")
    visible_metadata = tuple(entry for entry in metadata_path.parent.iterdir() if ".csv" in entry.name)
    if visible_metadata != (metadata_path,):
        raise ValueError("metric-cache metadata must contain exactly one scorer-visible CSV")
    metadata_bytes = metadata_path.read_bytes()
    metadata_sha256 = _require_sha256(metadata["sha256"], field="metric-cache metadata sha256")
    if hashlib.sha256(metadata_bytes).hexdigest() != metadata_sha256:
        raise ValueError("metric-cache metadata sha256 digest mismatch")
    if metadata["header"] != ["file_path"]:
        raise ValueError("metric-cache metadata header must equal ['file_path']")
    if type(metadata["row_count"]) is not int or metadata["row_count"] != len(expected_tokens):
        raise ValueError("metric-cache metadata row_count differs from token count")
    try:
        csv_rows = list(csv.reader(io.StringIO(metadata_bytes.decode("utf-8"), newline="")))
    except UnicodeDecodeError as exc:
        raise ValueError("metric-cache metadata must be UTF-8") from exc
    if not csv_rows or csv_rows[0] != ["file_path"] or len(csv_rows) != len(expected_tokens) + 1:
        raise ValueError("metric-cache metadata CSV rows differ from token inventory")
    csv_mapping: dict[str, Path] = {}
    for row_index, row in enumerate(csv_rows[1:], start=2):
        if len(row) != 1 or not row[0]:
            raise ValueError(f"metric-cache metadata CSV row {row_index} must contain one path")
        path = _strict_file(row[0], field=f"metric-cache metadata CSV row {row_index}")
        if not path.is_relative_to(metric_cache_root) or path.name != "metric_cache.pkl":
            raise ValueError("metric-cache metadata paths must be contained metric_cache.pkl files")
        token = validate_cvoi_navsim_scenario_token(path.parent.name)
        if token in csv_mapping:
            raise ValueError(f"metric-cache metadata contains duplicate token {token!r}")
        csv_mapping[token] = path

    raw_entries = inventory["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) != len(expected_tokens):
        raise ValueError("metric-cache entries must exactly cover token_inventory")
    entries: list[CvoiNavSimMetricCacheEntry] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _exact_mapping(
            raw_entry,
            fields=frozenset({"scenario_token", "path", "sha256", "size_bytes"}),
            label=f"metric-cache entry {index}",
        )
        token = validate_cvoi_navsim_scenario_token(entry["scenario_token"])
        path = _strict_file(entry["path"], field=f"metric-cache entry {index}.path")
        if not path.is_relative_to(metric_cache_root) or path.name != "metric_cache.pkl" or path.parent.name != token:
            raise ValueError("metric-cache entries must be contained <token>/metric_cache.pkl files")
        cache_bytes = path.read_bytes()
        sha256 = _require_sha256(entry["sha256"], field=f"metric-cache entry {index}.sha256")
        if hashlib.sha256(cache_bytes).hexdigest() != sha256:
            raise ValueError(f"metric-cache entry {index} sha256 digest mismatch")
        size_bytes = entry["size_bytes"]
        if type(size_bytes) is not int or size_bytes != len(cache_bytes):
            raise ValueError(f"metric-cache entry {index} size differs from the live cache file")
        entries.append(CvoiNavSimMetricCacheEntry(token, path, sha256, size_bytes))
    entries_tuple = tuple(entries)
    if tuple(entry.scenario_token for entry in entries_tuple) != expected_tokens:
        raise ValueError("metric-cache entries must exactly follow token_inventory")
    if csv_mapping != {entry.scenario_token: entry.path for entry in entries_tuple}:
        raise ValueError("metric-cache metadata CSV mapping differs from inventory entries")
    return CvoiNavSimMetricCacheInventory(
        protocol_id=V2_PROTOCOL_ID,
        metric_cache_root=metric_cache_root,
        metadata_path=metadata_path,
        metadata_sha256=metadata_sha256,
        entries=entries_tuple,
    )


def _validate_sensor_contract(value: object) -> None:
    sensor = _exact_mapping(value, fields=_SENSOR_FIELDS, label="sensor_contract")
    num_history_frames = sensor["num_history_frames"]
    if type(num_history_frames) is not int or num_history_frames < 1:
        raise ValueError("sensor_contract.num_history_frames must be a positive integer")
    if sensor["cam_f0"] != [num_history_frames - 1]:
        raise ValueError("sensor_contract.cam_f0 must select the final history frame")
    for field in _SENSOR_FIELDS - {"num_history_frames", "cam_f0"}:
        if sensor[field] is not False:
            raise ValueError(f"sensor_contract.{field} must be false")


def _validate_token_selection(
    value: object,
    *,
    split: str,
    expected_tokens: tuple[str, ...],
) -> None:
    selection = _exact_mapping(
        value,
        fields=frozenset({"mode", "subset_path", "subset_sha256"}),
        label="token_selection",
    )
    official = {"mode": f"official_{split}", "subset_path": None, "subset_sha256": None}
    if selection == official:
        return
    if split != "navtest" or selection["mode"] != "explicit_subset":
        raise ValueError(f"token_selection must identify the explicit {split} cohort")
    subset_path = _strict_file(selection["subset_path"], field="token_selection.subset_path")
    subset_bytes = subset_path.read_bytes()
    subset_sha256 = _require_sha256(selection["subset_sha256"], field="token_selection.subset_sha256")
    if hashlib.sha256(subset_bytes).hexdigest() != subset_sha256:
        raise ValueError("token_selection subset sha256 digest mismatch")
    subset = _exact_mapping(
        _parse_canonical_json(subset_bytes, label="token subset JSON"),
        fields=frozenset({"schema", "split", "tokens"}),
        label="token subset JSON",
    )
    if subset["schema"] != _TOKEN_SUBSET_SCHEMA or subset["split"] != "navtest":
        raise ValueError("token subset JSON must use the NavTest subset schema")
    if _parse_tokens(subset["tokens"], label="token subset tokens") != expected_tokens:
        raise ValueError("token subset tokens differ from raw authority token_inventory")


def read_cvoi_navsim_raw_v2_authority(
    root: Path,
    *,
    expected_split: str,
) -> CvoiNavSimRawV2Authority:
    """Read one exact raw V2 authority for an explicit NavTrain or NavTest split."""

    split = _require_split(expected_split, field="expected_split")
    strict_root = _strict_directory(root, field="raw V2 authority root")
    entries = tuple(strict_root.iterdir())
    if {entry.name for entry in entries} != _AUTHORITY_FILES:
        raise ValueError(f"raw V2 authority root must contain exactly {sorted(_AUTHORITY_FILES)}")
    artifacts = {entry.name: _strict_file(entry, field=f"raw V2 authority artifact {entry.name}") for entry in entries}
    manifest_bytes = artifacts["manifest.json"].read_bytes()
    scenario_bytes = artifacts["scenario_manifest.jsonl"].read_bytes()
    inventory_bytes = artifacts["metric_cache_inventory.json"].read_bytes()
    manifest = _exact_mapping(
        _parse_canonical_json(manifest_bytes, label="manifest.json"),
        fields=frozenset(
            {
                "schema",
                "protocol_id",
                "split",
                "roots",
                "sensor_contract",
                "token_selection",
                "token_inventory",
                "split_authority",
                "artifacts",
            }
        ),
        label="manifest.json",
    )
    if manifest["schema"] != _AUTHORITY_SCHEMA or manifest["protocol_id"] != V2_PROTOCOL_ID:
        raise ValueError("manifest.json must be a raw V2 authority")
    if manifest["split"] != split:
        raise ValueError(f"manifest.json split {manifest['split']!r} differs from expected_split {split!r}")

    roots = _exact_mapping(
        manifest["roots"],
        fields=frozenset({"data_root", "navsim_exp_root", "maps_root", "metric_cache_root", "devkit_root"}),
        label="manifest.json roots",
    )
    parsed_roots = {field: _manifest_directory(roots[field], field=f"manifest.json roots.{field}") for field in roots}
    _validate_sensor_contract(manifest["sensor_contract"])
    token_inventory = _exact_mapping(
        manifest["token_inventory"],
        fields=frozenset({"count", "tokens"}),
        label="manifest.json token_inventory",
    )
    tokens = _parse_tokens(token_inventory["tokens"], label="manifest.json token_inventory.tokens")
    if type(token_inventory["count"]) is not int or token_inventory["count"] != len(tokens):
        raise ValueError("manifest.json token_inventory.count differs from tokens")
    _validate_token_selection(manifest["token_selection"], split=split, expected_tokens=tokens)
    split_authority = _exact_mapping(
        manifest["split_authority"],
        fields=frozenset({"train_test_split", "scene_filter"}),
        label="manifest.json split_authority",
    )
    _source_record(split_authority["train_test_split"], label="manifest.json split_authority.train_test_split")
    _source_record(split_authority["scene_filter"], label="manifest.json split_authority.scene_filter")

    artifact_records = _exact_mapping(
        manifest["artifacts"],
        fields=frozenset({"scenario_manifest", "metric_cache_inventory"}),
        label="manifest.json artifacts",
    )
    scenario_artifact = _exact_mapping(
        artifact_records["scenario_manifest"],
        fields=frozenset({"path", "sha256", "row_count"}),
        label="manifest.json artifacts.scenario_manifest",
    )
    if (
        scenario_artifact["path"] != "scenario_manifest.jsonl"
        or type(scenario_artifact["row_count"]) is not int
        or scenario_artifact["row_count"] != len(tokens)
        or _require_sha256(scenario_artifact["sha256"], field="scenario artifact sha256")
        != hashlib.sha256(scenario_bytes).hexdigest()
    ):
        raise ValueError("manifest.json scenario artifact does not bind scenario_manifest.jsonl")
    inventory_artifact = _exact_mapping(
        artifact_records["metric_cache_inventory"],
        fields=frozenset({"path", "sha256"}),
        label="manifest.json artifacts.metric_cache_inventory",
    )
    if (
        inventory_artifact["path"] != "metric_cache_inventory.json"
        or _require_sha256(inventory_artifact["sha256"], field="inventory artifact sha256")
        != hashlib.sha256(inventory_bytes).hexdigest()
    ):
        raise ValueError("manifest.json inventory artifact does not bind metric_cache_inventory.json")

    scenario_manifest = _parse_scenario_manifest(scenario_bytes, expected_tokens=tokens)
    metric_cache_inventory = _parse_metric_cache_inventory(
        inventory_bytes,
        metric_cache_root=parsed_roots["metric_cache_root"],
        expected_tokens=tokens,
    )
    return CvoiNavSimRawV2Authority(
        root=strict_root,
        protocol_id=V2_PROTOCOL_ID,
        split=split,
        data_root=parsed_roots["data_root"],
        navsim_exp_root=parsed_roots["navsim_exp_root"],
        maps_root=parsed_roots["maps_root"],
        metric_cache_root=parsed_roots["metric_cache_root"],
        devkit_root=parsed_roots["devkit_root"],
        scenario_manifest=scenario_manifest,
        metric_cache_inventory=metric_cache_inventory,
    )


__all__ = (
    "CvoiNavSimMetricCacheEntry",
    "CvoiNavSimMetricCacheInventory",
    "CvoiNavSimRawV2Authority",
    "CvoiNavSimScenario",
    "CvoiNavSimScenarioManifest",
    "build_cvoi_navsim_scenario_manifest",
    "read_cvoi_navsim_raw_v2_authority",
    "validate_cvoi_navsim_scenario_token",
)
