"""Tests for raw V2 NavTrain/NavTest scenario authorities."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import pytest

from app.vjepa_cowa_world_model.training import cvoi_navsim_scenarios as scenarios_module
from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import V2_PROTOCOL_ID
from app.vjepa_cowa_world_model.training.cvoi_navsim_scenarios import (
    CvoiNavSimScenario,
    build_cvoi_navsim_scenario_manifest,
    read_cvoi_navsim_raw_v2_authority,
    validate_cvoi_navsim_scenario_token,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_raw_authority(tmp_path: Path, *, split: str) -> Path:
    root = (tmp_path / f"authority-{split}").resolve()
    root.mkdir()
    data_root = (tmp_path / f"data-{split}").resolve()
    navsim_exp_root = (tmp_path / f"exp-{split}").resolve()
    maps_root = (tmp_path / f"maps-{split}").resolve()
    metric_cache_root = (tmp_path / f"cache-{split}").resolve()
    devkit_root = (tmp_path / f"devkit-{split}").resolve()
    for path in (data_root, navsim_exp_root, maps_root, metric_cache_root, devkit_root):
        path.mkdir()

    tokens = (f"{split}-a", f"{split}-b")
    cache_paths: list[Path] = []
    entries: list[dict[str, object]] = []
    for token in tokens:
        cache_path = metric_cache_root / token / "metric_cache.pkl"
        cache_path.parent.mkdir()
        cache_bytes = f"cache:{token}".encode("utf-8")
        cache_path.write_bytes(cache_bytes)
        cache_paths.append(cache_path)
        entries.append(
            {
                "scenario_token": token,
                "path": str(cache_path),
                "sha256": _sha256(cache_bytes),
                "size_bytes": len(cache_bytes),
            }
        )
    metadata_path = metric_cache_root / "metadata" / "cache.csv"
    metadata_path.parent.mkdir()
    metadata_bytes = ("file_path\n" + "".join(f"{path}\n" for path in cache_paths)).encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)

    scenario_bytes = b"".join(
        _canonical_json(
            {
                "schema": "cvoi_navsim_scenario_v1",
                "protocol_id": V2_PROTOCOL_ID,
                "scenario_token": token,
                "observation_key": f"{index + 1:064x}",
                "log_name": f"log-{index}",
                "current_camera_data_path": f"raw/{token}.jpg",
            }
        )
        + b"\n"
        for index, token in enumerate(tokens)
    )
    inventory_bytes = _canonical_json(
        {
            "schema": "cvoi_navsim_metric_cache_inventory_v1",
            "protocol_id": V2_PROTOCOL_ID,
            "metric_cache_root": str(metric_cache_root),
            "metadata": {
                "path": str(metadata_path),
                "sha256": _sha256(metadata_bytes),
                "header": ["file_path"],
                "row_count": len(tokens),
            },
            "tokens": list(tokens),
            "entries": entries,
        }
    )
    split_path = devkit_root / f"{split}.yaml"
    scene_filter_path = devkit_root / f"{split}-filter.yaml"
    split_path.write_bytes(f"split: {split}\n".encode("utf-8"))
    scene_filter_path.write_bytes(f"filter: {split}\n".encode("utf-8"))
    manifest_bytes = _canonical_json(
        {
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
            "sensor_contract": {
                "num_history_frames": 4,
                "cam_f0": [3],
                "cam_l0": False,
                "cam_l1": False,
                "cam_l2": False,
                "cam_r0": False,
                "cam_r1": False,
                "cam_r2": False,
                "cam_b0": False,
                "lidar_pc": False,
            },
            "token_selection": {
                "mode": f"official_{split}",
                "subset_path": None,
                "subset_sha256": None,
            },
            "token_inventory": {"count": len(tokens), "tokens": list(tokens)},
            "split_authority": {
                "train_test_split": {"path": str(split_path), "sha256": _sha256(split_path.read_bytes())},
                "scene_filter": {
                    "path": str(scene_filter_path),
                    "sha256": _sha256(scene_filter_path.read_bytes()),
                },
            },
            "artifacts": {
                "scenario_manifest": {
                    "path": "scenario_manifest.jsonl",
                    "sha256": _sha256(scenario_bytes),
                    "row_count": len(tokens),
                },
                "metric_cache_inventory": {
                    "path": "metric_cache_inventory.json",
                    "sha256": _sha256(inventory_bytes),
                },
            },
        }
    )
    (root / "scenario_manifest.jsonl").write_bytes(scenario_bytes)
    (root / "metric_cache_inventory.json").write_bytes(inventory_bytes)
    (root / "manifest.json").write_bytes(manifest_bytes)
    return root


def test_pure_v2_manifest_is_a_frozen_token_observation_bijection() -> None:
    rows = (
        CvoiNavSimScenario("token-a", "1" * 64, "log-a", "raw/a.jpg"),
        CvoiNavSimScenario("token-b", "2" * 64, "log-b", "raw/b.jpg"),
    )

    manifest = build_cvoi_navsim_scenario_manifest(
        protocol_id=V2_PROTOCOL_ID,
        rows=reversed(rows),
        expected_metric_cache_tokens=("token-b", "token-a"),
    )

    assert manifest.tokens == ("token-a", "token-b")
    assert isinstance(manifest.scenarios_by_token, MappingProxyType)
    assert isinstance(manifest.tokens_by_observation_key, MappingProxyType)
    assert manifest.scenario_for_token("token-a") == rows[0]
    assert manifest.token_for_observation_key("2" * 64) == "token-b"
    with pytest.raises(FrozenInstanceError):
        manifest.protocol_id = "other"


def test_manifest_requires_exact_v2_cache_tokens_and_unique_observation_keys() -> None:
    row = CvoiNavSimScenario("token-a", "1" * 64, "log-a", "raw/a.jpg")
    with pytest.raises(ValueError, match="metric-cache tokens.*exactly"):
        build_cvoi_navsim_scenario_manifest(
            protocol_id=V2_PROTOCOL_ID,
            rows=(row,),
            expected_metric_cache_tokens=("token-a", "token-extra"),
        )
    with pytest.raises(ValueError, match="observation key"):
        build_cvoi_navsim_scenario_manifest(
            protocol_id=V2_PROTOCOL_ID,
            rows=(row, replace(row, scenario_token="token-b")),
            expected_metric_cache_tokens=("token-a", "token-b"),
        )
    with pytest.raises(ValueError, match="V2|protocol"):
        build_cvoi_navsim_scenario_manifest(
            protocol_id="pdms_v1_navtest",
            rows=(row,),
            expected_metric_cache_tokens=("token-a",),
        )


@pytest.mark.parametrize("value", ["", " token", "token ", ".", "..", "a/b", "a\\b", None, 3])
def test_scenario_tokens_must_be_nonempty_safe_directory_entries(value: object) -> None:
    with pytest.raises(ValueError, match="scenario token"):
        validate_cvoi_navsim_scenario_token(value)


@pytest.mark.parametrize("split", ["navtrain", "navtest"])
def test_raw_v2_authority_reader_accepts_only_the_explicit_requested_split(tmp_path: Path, split: str) -> None:
    root = _write_raw_authority(tmp_path, split=split)

    authority = read_cvoi_navsim_raw_v2_authority(root, expected_split=split)

    assert authority.root == root
    assert authority.protocol_id == V2_PROTOCOL_ID
    assert authority.split == split
    assert authority.scenario_manifest.tokens == (f"{split}-a", f"{split}-b")
    assert authority.metric_cache_inventory.tokens == authority.scenario_manifest.tokens
    assert tuple(authority.metric_cache_paths) == authority.scenario_manifest.tokens
    assert isinstance(authority.metric_cache_paths, MappingProxyType)
    opposite = "navtest" if split == "navtrain" else "navtrain"
    with pytest.raises(ValueError, match=f"expected.*{opposite}|split.*{opposite}"):
        read_cvoi_navsim_raw_v2_authority(root, expected_split=opposite)


@pytest.mark.parametrize("expected_split", ["", "train", "NAVTEST", None, 2])
def test_raw_v2_authority_reader_requires_an_explicit_supported_split(
    tmp_path: Path,
    expected_split: object,
) -> None:
    root = _write_raw_authority(tmp_path, split="navtest")
    with pytest.raises(ValueError, match="expected_split"):
        read_cvoi_navsim_raw_v2_authority(root, expected_split=expected_split)  # type: ignore[arg-type]


@pytest.mark.parametrize("mutation", ["extra", "scenario", "inventory", "cache", "metadata", "symlink"])
def test_raw_v2_authority_reader_rejects_non_exact_or_tampered_inputs(tmp_path: Path, mutation: str) -> None:
    root = _write_raw_authority(tmp_path, split="navtest")
    if mutation == "extra":
        (root / "receipt.json").write_bytes(b"{}")
    elif mutation == "scenario":
        (root / "scenario_manifest.jsonl").write_bytes(b"tampered\n")
    elif mutation == "inventory":
        (root / "metric_cache_inventory.json").write_bytes(b"{}")
    elif mutation == "cache":
        authority = read_cvoi_navsim_raw_v2_authority(root, expected_split="navtest")
        authority.metric_cache_inventory.entries[0].path.write_bytes(b"tampered")
    elif mutation == "metadata":
        authority = read_cvoi_navsim_raw_v2_authority(root, expected_split="navtest")
        authority.metric_cache_inventory.metadata_path.write_bytes(b"file_path\n")
    else:
        manifest = root / "manifest.json"
        contents = manifest.read_bytes()
        manifest.unlink()
        target = tmp_path / "manifest-target.json"
        target.write_bytes(contents)
        manifest.symlink_to(target)

    with pytest.raises(ValueError, match="exactly|artifact|digest|sha256|cache|metadata|symlink|regular"):
        read_cvoi_navsim_raw_v2_authority(root, expected_split="navtest")


def test_signed_suite_and_execution_surfaces_are_absent() -> None:
    for removed_name in (
        "CVOI_NAVSIM_FORMAL_AGENT_CONFIG_RELATIVE_PATH",
        "CVOI_NAVSIM_MAX_TOKEN_OVERRIDE_JSON_BYTES",
        "CvoiNavSimScenarioExecutionBinding",
        "CvoiNavSimSourceDigest",
        "CvoiNavSimDevkitApiDigest",
        "CvoiNavSimArtifactDigest",
        "CvoiNavSimScenarioManifestBundle",
        "read_cvoi_navsim_scenario_execution_binding",
        "read_cvoi_navsim_scenario_manifest_bundle",
        "run_cvoi_navsim_scenario_manifest_builder",
    ):
        assert removed_name not in scenarios_module.__dict__


def test_scenario_module_import_remains_navsim_free() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    code = (
        "import importlib,sys;"
        "importlib.import_module('app.vjepa_cowa_world_model.training.cvoi_navsim_scenarios');"
        "raise SystemExit(int(any(n == 'navsim' or n.startswith('navsim.') for n in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
