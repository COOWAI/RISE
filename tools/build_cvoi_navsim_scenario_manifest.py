#!/usr/bin/env python3
"""Build one receipt-free raw NavSim V2 scenario authority."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import io
import json
import os
import stat
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import (
    V2_PROTOCOL_ID,
    get_cvoi_navsim_metric_protocol,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

_RENAME_NOREPLACE = 1
_AUTHORITY_FILES = frozenset({"scenario_manifest.jsonl", "metric_cache_inventory.json", "manifest.json"})
_TOKEN_SUBSET_SCHEMA = "cvoi_navsim_token_subset_v1"
_MAX_TOKEN_SUBSET_BYTES = 64 * 1024
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _canonical_json(value: object) -> bytes:
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


def _parse_canonical_json_object(raw: bytes, *, label: str) -> Mapping[str, object]:
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
    if raw != _canonical_json(value):
        raise ValueError(f"{label} must use canonical compact JSON without a trailing newline")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _require_private_directory(value: os.stat_result, *, field: str) -> None:
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError(f"{field} must identify a directory")
    if value.st_uid != os.geteuid() or value.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"{field} must be owned by the effective uid and writable by no other principal")


def _require_pinned_output_parent(output_dir: Path, parent_fd: int) -> None:
    if not isinstance(output_dir, Path) or not output_dir.is_absolute() or output_dir.name in {"", ".", ".."}:
        raise ValueError("output_dir must be an absolute child path")
    if type(parent_fd) is not int or parent_fd < 0:
        raise ValueError("output_parent_fd must be an open directory descriptor")
    try:
        descriptor_stat = os.fstat(parent_fd)
        path_stat = os.stat(output_dir.parent, follow_symlinks=False)
    except OSError as exc:
        raise ValueError("output parent could not be verified") from exc
    _require_private_directory(descriptor_stat, field="output parent descriptor")
    _require_private_directory(path_stat, field="output parent path")
    if output_dir.parent.is_symlink() or output_dir.parent.resolve(strict=True) != output_dir.parent:
        raise ValueError("output_dir parent must be a canonical non-symlink directory")
    if not _same_file_identity(descriptor_stat, path_stat):
        raise ValueError("output_parent_fd must identify output_dir.parent")


def _rename_noreplace(parent_fd: int, source_name: str, target_name: str) -> None:
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError) as exc:
        raise RuntimeError("atomic raw-authority publication requires renameat2(RENAME_NOREPLACE)") from exc
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(target_name),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == 17:
            raise FileExistsError(f"output_dir already exists or is a symlink: {target_name}")
        raise OSError(error, os.strerror(error), target_name)


def _write_all(fd: int, data: bytes) -> None:
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written <= 0:
            raise OSError("raw-authority file write made no progress")
        offset += written


def _cleanup_staging(parent_fd: int, staging_name: str) -> None:
    try:
        staging_fd = os.open(staging_name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    try:
        for name in _AUTHORITY_FILES:
            try:
                os.unlink(name, dir_fd=staging_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(staging_fd)
    try:
        os.rmdir(staging_name, dir_fd=parent_fd)
    except FileNotFoundError:
        pass


def _publish_authority(
    *,
    output_dir: Path,
    output_parent_fd: int,
    artifacts: Mapping[str, bytes],
) -> Path:
    if set(artifacts) != _AUTHORITY_FILES:
        raise ValueError(f"raw V2 authority must contain exactly {sorted(_AUTHORITY_FILES)}")
    _require_pinned_output_parent(output_dir, output_parent_fd)
    try:
        os.stat(output_dir.name, dir_fd=output_parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ValueError("output_dir entry could not be inspected") from exc
    else:
        raise FileExistsError(f"output_dir already exists or is a symlink: {output_dir}")

    staging_name = f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging_fd: int | None = None
    try:
        os.mkdir(staging_name, mode=0o700, dir_fd=output_parent_fd)
        staging_fd = os.open(staging_name, _DIRECTORY_FLAGS, dir_fd=output_parent_fd)
        for name in sorted(_AUTHORITY_FILES):
            fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                0o600,
                dir_fd=staging_fd,
            )
            try:
                _write_all(fd, artifacts[name])
                os.fsync(fd)
            finally:
                os.close(fd)
        os.fsync(staging_fd)
        os.close(staging_fd)
        staging_fd = None
        _rename_noreplace(output_parent_fd, staging_name, output_dir.name)
        os.fsync(output_parent_fd)
    except Exception:
        if staging_fd is not None:
            os.close(staging_fd)
        _cleanup_staging(output_parent_fd, staging_name)
        raise
    return output_dir


def _existing_directory(path: Path, *, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{field} must be an absolute canonical non-symlink directory")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{field} must be an existing directory: {path}") from exc
    if resolved != path or not resolved.is_dir():
        raise ValueError(f"{field} must be an absolute canonical non-symlink directory")
    return resolved


def _contained_file(path: Path, *, root: Path, field: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{field} must be a non-symlink file contained in {root}")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{field} must be an existing regular file: {path}") from exc
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise ValueError(f"{field} must be a regular file contained in {root}")
    return resolved


def _existing_regular_file(path: Path, *, field: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{field} must be an absolute canonical non-symlink regular file")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{field} must be an existing regular file: {path}") from exc
    if resolved != path or not resolved.is_file():
        raise ValueError(f"{field} must be an absolute canonical non-symlink regular file")
    return resolved


def _nonempty_string(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty string without surrounding whitespace")
    return value


def _unique_sorted_tokens(value: object, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{field} must be a non-empty sequence of unique tokens")
    tokens: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = _nonempty_string(item, field=field)
        if token in {".", ".."} or "/" in token or "\\" in token:
            raise ValueError(f"{field} contains unsafe token {token!r}")
        if token in seen:
            raise ValueError(f"{field} contains duplicate token {token!r}")
        seen.add(token)
        tokens.append(token)
    return tuple(sorted(tokens))


def _load_token_selection(
    token_subset_path: Path | None,
    *,
    official_tokens: tuple[str, ...],
    split: str,
) -> tuple[tuple[str, ...], dict[str, object]]:
    if split == "navtrain":
        if token_subset_path is not None:
            raise ValueError("official navtrain cohort forbids a token subset")
        return official_tokens, {"mode": "official_navtrain", "subset_path": None, "subset_sha256": None}
    if split != "navtest":
        raise ValueError("scenario split must be exactly 'navtrain' or 'navtest'")
    if token_subset_path is None:
        return official_tokens, {"mode": "official_navtest", "subset_path": None, "subset_sha256": None}

    subset_path = _existing_regular_file(token_subset_path, field="token_subset_path")
    raw = subset_path.read_bytes()
    if len(raw) > _MAX_TOKEN_SUBSET_BYTES:
        raise ValueError(f"token subset JSON exceeds the maximum size of {_MAX_TOKEN_SUBSET_BYTES} bytes")
    subset = _parse_canonical_json_object(raw, label="token subset JSON")
    expected_fields = {"schema", "split", "tokens"}
    if set(subset) != expected_fields or any(type(key) is not str for key in subset):
        raise ValueError("token subset JSON fields must be exactly schema, split, and tokens")
    if subset["schema"] != _TOKEN_SUBSET_SCHEMA or subset["split"] != "navtest":
        raise ValueError("token subset JSON must use the NavTest subset schema")
    selected_tokens = _unique_sorted_tokens(subset["tokens"], field="token subset tokens")
    if not isinstance(subset["tokens"], list) or tuple(subset["tokens"]) != selected_tokens:
        raise ValueError("token subset tokens must be a canonical sorted JSON list")
    if not set(selected_tokens) < set(official_tokens):
        raise ValueError("token subset tokens must be a proper nonempty subset of official navtest tokens")
    return selected_tokens, {
        "mode": "explicit_subset",
        "subset_path": str(subset_path),
        "subset_sha256": _sha256_bytes(raw),
    }


def _require_data_split(value: object, *, split: str) -> str:
    data_split = _nonempty_string(value, field=f"{split} data_split")
    expected_data_split = "trainval" if split == "navtrain" else "test"
    if data_split != expected_data_split:
        raise ValueError(f"{split} data_split must be exactly {expected_data_split!r}")
    return data_split


def _load_split_config(
    devkit_root: Path,
    *,
    split: str,
) -> tuple[str, object, tuple[str, ...], int, Path, Path]:
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if split not in {"navtrain", "navtest"}:
        raise ValueError("scenario split must be exactly 'navtrain' or 'navtest'")
    split_root = devkit_root / "navsim/planning/script/config/common/train_test_split"
    split_path = _contained_file(split_root / f"{split}.yaml", root=devkit_root, field=f"{split} split config")
    split_config = OmegaConf.to_container(OmegaConf.load(split_path), resolve=True)
    if not isinstance(split_config, Mapping):
        raise ValueError(f"{split} split config must resolve to a mapping")
    defaults = split_config.get("defaults")
    if isinstance(defaults, (str, bytes)) or not isinstance(defaults, Sequence):
        raise ValueError(f"{split} defaults must contain exactly one scene_filter mapping")
    scene_filter_defaults = [item for item in defaults if isinstance(item, Mapping) and "scene_filter" in item]
    if len(scene_filter_defaults) != 1:
        raise ValueError(f"{split} defaults must contain exactly one scene_filter mapping")
    scene_filter_name = _nonempty_string(
        scene_filter_defaults[0]["scene_filter"],
        field=f"{split} scene_filter name",
    )
    if scene_filter_name != split:
        raise ValueError(f"{split} split must select exactly the {split} scene filter")
    data_split = _require_data_split(split_config.get("data_split"), split=split)
    scene_filter_path = _contained_file(
        split_root / "scene_filter" / f"{scene_filter_name}.yaml",
        root=devkit_root,
        field=f"{split} scene_filter config",
    )
    scene_filter = instantiate(OmegaConf.load(scene_filter_path))
    tokens = _unique_sorted_tokens(scene_filter.tokens, field=f"configured {split} tokens")
    num_history_frames = scene_filter.num_history_frames
    if type(num_history_frames) is not int or num_history_frames < 1:
        raise ValueError("scene_filter.num_history_frames must be a positive integer")
    return data_split, scene_filter, tokens, num_history_frames, split_path, scene_filter_path


def _import_navsim_loaders() -> tuple[type[object], type[object], type[object]]:
    from navsim.common.dataclasses import SensorConfig
    from navsim.common.dataloader import MetricCacheLoader, SceneLoader

    return SensorConfig, SceneLoader, MetricCacheLoader


def _sensor_config(sensor_config_type: type[object], num_history_frames: int) -> tuple[object, dict[str, object]]:
    contract: dict[str, object] = {
        "num_history_frames": num_history_frames,
        "cam_f0": [num_history_frames - 1],
        "cam_l0": False,
        "cam_l1": False,
        "cam_l2": False,
        "cam_r0": False,
        "cam_r1": False,
        "cam_r2": False,
        "cam_b0": False,
        "lidar_pc": False,
    }
    return (
        sensor_config_type(
            cam_f0=contract["cam_f0"],
            cam_l0=False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        ),
        contract,
    )


def _build_scene_loader(
    scene_loader_type: type[object],
    *,
    data_root: Path,
    data_split: str,
    scene_filter: object,
    sensor_config: object,
) -> object:
    return scene_loader_type(
        data_path=data_root / "navsim_logs" / data_split,
        original_sensor_path=data_root / "sensor_blobs" / data_split,
        scene_filter=scene_filter,
        synthetic_sensor_path=None,
        synthetic_scenes_path=None,
        sensor_config=sensor_config,
    )


def _scenario_rows(
    scene_loader: object,
    tokens: tuple[str, ...],
    num_history_frames: int,
) -> list[dict[str, object]]:
    from app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity import observation_key

    rows: list[dict[str, object]] = []
    observation_keys: set[str] = set()
    for token in tokens:
        agent_input = scene_loader.get_agent_input_from_token(token)
        current_camera = agent_input.cameras[-1].cam_f0
        key = observation_key(current_camera.image)
        if key in observation_keys:
            raise ValueError(f"decoded current-camera observation key is not unique for token {token!r}")
        observation_keys.add(key)
        frame = scene_loader.scene_frames_dicts[token][num_history_frames - 1]
        if not isinstance(frame, Mapping):
            raise ValueError(f"raw current frame for token {token!r} must be a mapping")
        log_name = _nonempty_string(frame.get("log_name"), field=f"raw log_name for token {token!r}")
        cameras = frame.get("cams")
        if not isinstance(cameras, Mapping):
            raise ValueError(f"raw cams for token {token!r} must be a mapping")
        front_entries = [value for name, value in cameras.items() if type(name) is str and name.lower() == "cam_f0"]
        if len(front_entries) != 1 or not isinstance(front_entries[0], Mapping):
            raise ValueError(f"raw cams for token {token!r} must contain one mapping cam_f0")
        data_path = _nonempty_string(
            front_entries[0].get("data_path"),
            field=f"raw cam_f0 data_path for token {token!r}",
        )
        camera_path = current_camera.camera_path
        if camera_path is not None:
            camera_path_string = str(camera_path)
            if not camera_path_string or camera_path_string != data_path:
                raise ValueError(f"V2 Camera.camera_path disagrees with raw cam_f0 data_path for token {token!r}")
        rows.append(
            {
                "schema": "cvoi_navsim_scenario_v1",
                "protocol_id": V2_PROTOCOL_ID,
                "scenario_token": token,
                "observation_key": key,
                "log_name": log_name,
                "current_camera_data_path": data_path,
            }
        )
    return rows


def _strict_metadata_csv_entry(metric_cache_root: Path) -> Path:
    metadata_dir = metric_cache_root / "metadata"
    if metadata_dir.is_symlink() or not metadata_dir.is_dir():
        raise ValueError(f"metric-cache metadata must be an existing directory: {metadata_dir}")
    visible_entries = [entry for entry in metadata_dir.iterdir() if ".csv" in entry.name]
    if len(visible_entries) != 1:
        raise ValueError("metric-cache metadata must contain exactly one scorer-visible CSV")
    entry = visible_entries[0]
    if entry.is_symlink() or entry.suffix != ".csv" or not entry.is_file():
        raise ValueError("the scorer-visible metadata entry must be a non-symlink regular CSV")
    return entry.resolve(strict=True)


def _metric_cache_inventory(
    metric_cache_root: Path,
    metric_cache_loader: object,
    expected_tokens: tuple[str, ...],
) -> dict[str, object]:
    metadata_path = _strict_metadata_csv_entry(metric_cache_root)
    metadata_bytes = metadata_path.read_bytes()
    try:
        csv_rows = list(csv.reader(io.StringIO(metadata_bytes.decode("utf-8"), newline="")))
    except UnicodeDecodeError as exc:
        raise ValueError("metric-cache metadata CSV must be UTF-8") from exc
    if not csv_rows or csv_rows[0] != ["file_path"] or len(csv_rows) == 1:
        raise ValueError("metric-cache metadata CSV must have a file_path header and data rows")
    csv_mapping: dict[str, Path] = {}
    entries: list[dict[str, object]] = []
    for row_number, row in enumerate(csv_rows[1:], start=2):
        if len(row) != 1 or not row[0]:
            raise ValueError(f"metric-cache metadata row {row_number} must contain one path")
        raw_path = Path(row[0])
        if not raw_path.is_absolute():
            raise ValueError(f"metric-cache metadata path must be absolute: {row[0]!r}")
        try:
            resolved = raw_path.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise ValueError(f"metric-cache file must exist: {raw_path}") from exc
        if (
            not resolved.is_file()
            or resolved.name != "metric_cache.pkl"
            or not resolved.is_relative_to(metric_cache_root)
        ):
            raise ValueError(f"metric-cache path must be a contained metric_cache.pkl file: {raw_path}")
        token = _nonempty_string(resolved.parent.name, field=f"metric-cache token for row {row_number}")
        if token in csv_mapping:
            raise ValueError(f"metric-cache metadata contains duplicate token {token!r}")
        csv_mapping[token] = resolved
        data = resolved.read_bytes()
        entries.append(
            {
                "scenario_token": token,
                "path": str(resolved),
                "sha256": _sha256_bytes(data),
                "size_bytes": len(data),
            }
        )
    if _unique_sorted_tokens(tuple(csv_mapping), field="metric-cache metadata tokens") != expected_tokens:
        raise ValueError("metric-cache metadata tokens must exactly match selected scenario tokens")
    loader_mapping = metric_cache_loader.metric_cache_paths
    if not isinstance(loader_mapping, Mapping):
        raise ValueError("MetricCacheLoader.metric_cache_paths must be a mapping")
    normalized_loader = {str(token): Path(path).resolve(strict=True) for token, path in loader_mapping.items()}
    if normalized_loader != csv_mapping:
        raise ValueError("MetricCacheLoader.metric_cache_paths must exactly match the metadata CSV")
    entries.sort(key=lambda entry: str(entry["scenario_token"]))
    return {
        "schema": "cvoi_navsim_metric_cache_inventory_v1",
        "protocol_id": V2_PROTOCOL_ID,
        "metric_cache_root": str(metric_cache_root),
        "metadata": {
            "path": str(metadata_path),
            "sha256": _sha256_bytes(metadata_bytes),
            "header": ["file_path"],
            "row_count": len(csv_rows) - 1,
        },
        "tokens": list(expected_tokens),
        "entries": entries,
    }


def build_bundle(args: argparse.Namespace) -> Path:
    """Build one raw V2 authority from explicit environment paths."""

    protocol = get_cvoi_navsim_metric_protocol(args.protocol_id)
    if protocol.protocol_id != V2_PROTOCOL_ID:
        raise ValueError("raw authority builder supports only the retained V2 protocol")
    split = args.split
    if split not in {"navtrain", "navtest"}:
        raise ValueError("split must be exactly 'navtrain' or 'navtest'")
    output_dir = args.output_dir
    _require_pinned_output_parent(output_dir, args.output_parent_fd)
    try:
        os.stat(output_dir.name, dir_fd=args.output_parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ValueError("output_dir entry could not be inspected") from exc
    else:
        raise FileExistsError(f"output_dir already exists or is a symlink: {output_dir}")

    devkit_root = _existing_directory(args.devkit_root, field="devkit_root")
    data_root = _existing_directory(args.data_root, field="data_root")
    navsim_exp_root = _existing_directory(args.navsim_exp_root, field="navsim_exp_root")
    maps_root = _existing_directory(args.maps_root, field="maps_root")
    metric_cache_root = _existing_directory(args.metric_cache_root, field="metric_cache_root")
    data_split, scene_filter, official_tokens, num_history_frames, split_path, filter_path = _load_split_config(
        devkit_root,
        split=split,
    )
    selected_tokens, token_selection = _load_token_selection(
        args.token_subset_path,
        official_tokens=official_tokens,
        split=split,
    )
    if token_selection["mode"] == "explicit_subset":
        scene_filter.tokens = list(selected_tokens)
    sensor_type, scene_loader_type, cache_loader_type = _import_navsim_loaders()
    sensor_config, sensor_contract = _sensor_config(sensor_type, num_history_frames)
    scene_loader = _build_scene_loader(
        scene_loader_type,
        data_root=data_root,
        data_split=data_split,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    scene_tokens = _unique_sorted_tokens(scene_loader.tokens, field="SceneLoader.tokens")
    cache_loader = cache_loader_type(metric_cache_root)
    cache_tokens = _unique_sorted_tokens(cache_loader.tokens, field="MetricCacheLoader.tokens")
    if selected_tokens != scene_tokens or selected_tokens != cache_tokens:
        raise ValueError("configured split, SceneLoader, and MetricCacheLoader tokens must match exactly")

    rows = _scenario_rows(scene_loader, selected_tokens, num_history_frames)
    inventory = _metric_cache_inventory(metric_cache_root, cache_loader, selected_tokens)
    scenario_bytes = b"".join(_canonical_json(row) + b"\n" for row in rows)
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
        "sensor_contract": sensor_contract,
        "token_selection": token_selection,
        "token_inventory": {"count": len(selected_tokens), "tokens": list(selected_tokens)},
        "split_authority": {
            "train_test_split": {"path": str(split_path), "sha256": _sha256_file(split_path)},
            "scene_filter": {"path": str(filter_path), "sha256": _sha256_file(filter_path)},
        },
        "artifacts": {
            "scenario_manifest": {
                "path": "scenario_manifest.jsonl",
                "sha256": _sha256_bytes(scenario_bytes),
                "row_count": len(rows),
            },
            "metric_cache_inventory": {
                "path": "metric_cache_inventory.json",
                "sha256": _sha256_bytes(inventory_bytes),
            },
        },
    }
    return _publish_authority(
        output_dir=output_dir,
        output_parent_fd=args.output_parent_fd,
        artifacts={
            "scenario_manifest.jsonl": scenario_bytes,
            "metric_cache_inventory.json": inventory_bytes,
            "manifest.json": _canonical_json(manifest),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-id", choices=(V2_PROTOCOL_ID,), required=True)
    parser.add_argument("--split", choices=("navtrain", "navtest"), required=True)
    parser.add_argument("--devkit-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--navsim-exp-root", type=Path, required=True)
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--metric-cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-parent-fd", type=int, required=True)
    parser.add_argument("--token-subset-path", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    output_dir = build_bundle(parse_args(argv))
    logger.info("Built raw NavSim V2 scenario authority: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
