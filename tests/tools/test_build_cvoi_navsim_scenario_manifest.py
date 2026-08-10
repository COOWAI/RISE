"""Tests for the receipt-free raw NavSim V2 authority builder."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.vjepa_cowa_world_model.training.cvoi_navsim_protocols import V2_PROTOCOL_ID
from app.vjepa_cowa_world_model.training.cvoi_navsim_scenarios import read_cvoi_navsim_raw_v2_authority
from tools import build_cvoi_navsim_scenario_manifest as builder

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "tools/build_cvoi_navsim_scenario_manifest.py"


class _SceneFilter:
    def __init__(self, tokens: tuple[str, ...]) -> None:
        self.tokens = list(tokens)
        self.num_history_frames = 4


class _SensorConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _SceneLoader:
    def __init__(
        self,
        *,
        data_path: Path,
        original_sensor_path: Path,
        scene_filter: _SceneFilter,
        synthetic_sensor_path: None,
        synthetic_scenes_path: None,
        sensor_config: _SensorConfig,
    ) -> None:
        del data_path, original_sensor_path, synthetic_sensor_path, synthetic_scenes_path, sensor_config
        self.tokens = tuple(scene_filter.tokens)
        self.scene_frames_dicts = {
            token: [
                {},
                {},
                {},
                {"log_name": f"log-{token}", "cams": {"cam_f0": {"data_path": f"raw/{token}.jpg"}}},
            ]
            for token in self.tokens
        }

    def get_agent_input_from_token(self, token: str) -> object:
        image = np.frombuffer(token.encode("utf-8"), dtype=np.uint8)
        camera = SimpleNamespace(image=image, camera_path=f"raw/{token}.jpg")
        return SimpleNamespace(cameras=[SimpleNamespace(cam_f0=camera)])


class _MetricCacheLoader:
    root: Path
    tokens_fixture: tuple[str, ...]

    def __init__(self, root: Path) -> None:
        assert root == self.root
        self.tokens = self.tokens_fixture
        self.metric_cache_paths = {token: root / token / "metric_cache.pkl" for token in self.tokens}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    split: str,
    tokens: tuple[str, ...] = ("token-a", "token-b"),
    token_subset_path: Path | None = None,
) -> tuple[argparse.Namespace, int]:
    data_root = (tmp_path / "data").resolve()
    navsim_exp_root = (tmp_path / "exp").resolve()
    maps_root = (tmp_path / "maps").resolve()
    metric_cache_root = (tmp_path / "cache").resolve()
    devkit_root = (tmp_path / "devkit").resolve()
    for path in (data_root, navsim_exp_root, maps_root, metric_cache_root, devkit_root):
        path.mkdir()
    (data_root / "navsim_logs" / ("trainval" if split == "navtrain" else "test")).mkdir(parents=True)
    (data_root / "sensor_blobs" / ("trainval" if split == "navtrain" else "test")).mkdir(parents=True)

    cache_tokens = tokens
    if token_subset_path is not None and split == "navtest":
        cache_tokens = tuple(json.loads(token_subset_path.read_bytes())["tokens"])
    cache_paths: list[Path] = []
    for token in cache_tokens:
        cache_path = metric_cache_root / token / "metric_cache.pkl"
        cache_path.parent.mkdir()
        cache_path.write_bytes(f"cache:{token}".encode("utf-8"))
        cache_paths.append(cache_path)
    metadata_path = metric_cache_root / "metadata" / "cache.csv"
    metadata_path.parent.mkdir()
    metadata_path.write_text("file_path\n" + "".join(f"{path}\n" for path in cache_paths), encoding="utf-8")

    split_path = devkit_root / f"{split}.yaml"
    filter_path = devkit_root / f"{split}-filter.yaml"
    split_path.write_text(f"split: {split}\n", encoding="utf-8")
    filter_path.write_text(f"filter: {split}\n", encoding="utf-8")
    scene_filter = _SceneFilter(tokens)
    monkeypatch.setattr(
        builder,
        "_load_split_config",
        lambda root, *, split: (
            "trainval" if split == "navtrain" else "test",
            scene_filter,
            tokens,
            4,
            split_path,
            filter_path,
        ),
    )
    monkeypatch.setattr(builder, "_import_navsim_loaders", lambda: (_SensorConfig, _SceneLoader, _MetricCacheLoader))
    _MetricCacheLoader.root = metric_cache_root
    _MetricCacheLoader.tokens_fixture = cache_tokens

    output_parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    args = argparse.Namespace(
        protocol_id=V2_PROTOCOL_ID,
        split=split,
        devkit_root=devkit_root,
        data_root=data_root,
        navsim_exp_root=navsim_exp_root,
        maps_root=maps_root,
        metric_cache_root=metric_cache_root,
        output_dir=(tmp_path / "authority").resolve(),
        output_parent_fd=output_parent_fd,
        token_subset_path=token_subset_path,
    )
    return args, output_parent_fd


@pytest.mark.parametrize("split", ["navtrain", "navtest"])
def test_builder_creates_exact_raw_v2_authority_for_explicit_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    split: str,
) -> None:
    args, output_parent_fd = _fixture(tmp_path, monkeypatch, split=split)
    try:
        output = builder.build_bundle(args)
    finally:
        os.close(output_parent_fd)

    assert {entry.name for entry in output.iterdir()} == {
        "manifest.json",
        "scenario_manifest.jsonl",
        "metric_cache_inventory.json",
    }
    authority = read_cvoi_navsim_raw_v2_authority(output, expected_split=split)
    assert authority.split == split
    assert authority.scenario_manifest.tokens == ("token-a", "token-b")
    manifest_raw = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert manifest_raw == _canonical_json(manifest)
    assert manifest["schema"] == "cvoi_navsim_raw_v2_authority_v1"
    assert manifest["protocol_id"] == V2_PROTOCOL_ID
    assert manifest["token_selection"] == {
        "mode": f"official_{split}",
        "subset_path": None,
        "subset_sha256": None,
    }
    assert "devkit" not in manifest
    assert "authority" not in manifest
    assert "task_identity" not in manifest


def test_navtest_builder_retains_explicit_token_subset_without_mutating_source_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subset_path = (tmp_path / "subset.json").resolve()
    subset_path.write_bytes(
        _canonical_json(
            {
                "schema": "cvoi_navsim_token_subset_v1",
                "split": "navtest",
                "tokens": ["token-a"],
            }
        )
    )
    args, output_parent_fd = _fixture(
        tmp_path,
        monkeypatch,
        split="navtest",
        token_subset_path=subset_path,
    )
    try:
        output = builder.build_bundle(args)
    finally:
        os.close(output_parent_fd)

    authority = read_cvoi_navsim_raw_v2_authority(output, expected_split="navtest")
    assert authority.scenario_manifest.tokens == ("token-a",)
    assert json.loads((output / "manifest.json").read_bytes())["token_selection"]["mode"] == "explicit_subset"


def test_navtrain_builder_forbids_token_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    subset_path = (tmp_path / "subset.json").resolve()
    subset_path.write_bytes(b"{}")
    args, output_parent_fd = _fixture(
        tmp_path,
        monkeypatch,
        split="navtrain",
        token_subset_path=subset_path,
    )
    try:
        with pytest.raises(ValueError, match="navtrain.*forbids"):
            builder.build_bundle(args)
    finally:
        os.close(output_parent_fd)
    assert not args.output_dir.exists()


def test_navtest_split_config_rejects_trainval_data_split() -> None:
    with pytest.raises(ValueError, match="navtest data_split must be exactly 'test'"):
        builder._require_data_split("trainval", split="navtest")


def test_builder_rejects_v1_before_importing_navsim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, output_parent_fd = _fixture(tmp_path, monkeypatch, split="navtest")
    args.protocol_id = "pdms_v1_navtest"
    monkeypatch.setattr(builder, "_import_navsim_loaders", lambda: pytest.fail("must reject before import"))
    try:
        with pytest.raises(ValueError, match="V2|protocol"):
            builder.build_bundle(args)
    finally:
        os.close(output_parent_fd)
    assert not args.output_dir.exists()


def test_builder_never_overwrites_existing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    args, output_parent_fd = _fixture(tmp_path, monkeypatch, split="navtest")
    args.output_dir.mkdir()
    marker = args.output_dir / "keep.txt"
    marker.write_text("keep\n", encoding="utf-8")
    try:
        with pytest.raises(FileExistsError, match="already exists"):
            builder.build_bundle(args)
    finally:
        os.close(output_parent_fd)
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_help_exposes_only_raw_v2_builder_arguments() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-B", "-s", str(CLI_PATH), "--help"],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--split {navtrain,navtest}" in result.stdout
    assert "--protocol-id" in result.stdout
    for removed_option in (
        "--expected-devkit-revision",
        "--task-id",
        "--task-signature",
        "--task-manifest-sha256",
        "--source-commit",
        "--hydra-agent-config-path",
        "--hydra-agent-config-sha256",
    ):
        assert removed_option not in result.stdout


def test_builder_source_has_no_v1_checkout_or_signed_suite_surface() -> None:
    source = CLI_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "V1_PROTOCOL_ID",
        "verify_configured_devkit",
        "expected_devkit_revision",
        "task_identity",
        "hydra_agent_config",
        "scenario_manifest_bundle_v4",
        "scenario_manifest_bundle_v2",
    ):
        assert forbidden not in source
