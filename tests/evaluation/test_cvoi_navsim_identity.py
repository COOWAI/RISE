"""Tests for NavSim-free CVoI observation and seed identities."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity import (
    decode_observation_key,
    decode_unsigned_seed,
    encode_observation_key,
    encode_unsigned_seed,
    observation_key,
    observation_key_tensor,
    unsigned_seed_tensor,
)


def test_observation_key_is_content_based_and_layout_independent() -> None:
    image = np.arange(60, dtype=np.uint8).reshape(4, 5, 3)

    expected = observation_key(image)
    assert expected == observation_key(image.copy(order="C"))
    assert expected == observation_key(np.asfortranarray(image))
    assert len(expected) == 64
    assert expected == expected.lower()
    assert expected != observation_key(image.astype(np.float32))
    assert expected != observation_key(image.reshape(5, 4, 3))
    changed = image.copy()
    changed[0, 0, 0] += 1
    assert expected != observation_key(changed)


@pytest.mark.parametrize("value", [None, [[1]], np.array([object()], dtype=object)])
def test_observation_key_rejects_non_arrays_and_object_dtype(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="ndarray|object"):
        observation_key(value)


def test_observation_key_tensor_codec_is_strict() -> None:
    key = observation_key(np.arange(18, dtype=np.uint8).reshape(2, 3, 3))
    encoded = encode_observation_key(key)

    assert encoded.dtype is torch.uint8
    assert encoded.shape == (32,)
    assert decode_observation_key(encoded) == key
    assert torch.equal(observation_key_tensor(key), encoded)

    for bad_key in ("0" * 63, "A" * 64, "g" * 64, 1):
        with pytest.raises((TypeError, ValueError), match="64 lowercase|observation"):
            encode_observation_key(bad_key)
    for bad_tensor in (torch.zeros(31, dtype=torch.uint8), torch.zeros(32), np.zeros(32, dtype=np.uint8)):
        with pytest.raises((TypeError, ValueError), match="uint8|shape|tensor"):
            decode_observation_key(bad_tensor)


@pytest.mark.parametrize("value", [0, 1, 2**63, 2**64 - 1])
def test_unsigned_seed_codec_preserves_all_64_bits(value: int) -> None:
    encoded = encode_unsigned_seed(value)

    assert encoded.dtype is torch.uint8
    assert encoded.shape == (8,)
    assert decode_unsigned_seed(encoded) == value
    assert torch.equal(unsigned_seed_tensor(value), encoded)


@pytest.mark.parametrize("value", [True, False, -1, 2**64, 1.0, "1"])
def test_unsigned_seed_codec_rejects_non_unsigned_64_bit_ints(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="unsigned|integer|seed"):
        encode_unsigned_seed(value)


def test_identity_module_import_does_not_import_navsim() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib
import sys

assert not any(name == "navsim" or name.startswith("navsim.") for name in sys.modules)
evaluation = importlib.import_module("app.vjepa_cowa_world_model.evaluation")
identity = importlib.import_module("app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity")
assert evaluation.__all__ == ("VJEPAWorldModelAgent", "VJEPAFeatureBuilder")
assert callable(identity.observation_key)
assert not any(name == "navsim" or name.startswith("navsim.") for name in sys.modules)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_protocol_and_project_logger_import_without_user_site_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib
import sys

protocols = importlib.import_module("app.vjepa_cowa_world_model.training.cvoi_navsim_protocols")
project_logging = importlib.import_module("src.utils.logging")
assert protocols.V2_PROTOCOL_ID == "epdms_v2_one_stage_navtest"
assert "V1_PROTOCOL_ID" not in protocols.__dict__
assert callable(project_logging.get_logger)
assert "torch" not in sys.modules
assert "prettytable" not in sys.modules
assert "app.vjepa_cowa_world_model.losses" not in sys.modules
assert "app.vjepa_cowa_world_model.utils" not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [sys.executable, "-B", "-s", "-c", script],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_manifest_tool_help_runs_without_user_site_dependencies() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-s",
            str(repo_root / "tools/build_cvoi_navsim_scenario_manifest.py"),
            "--help",
        ],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout
    assert "--split {navtrain,navtest}" in result.stdout
    assert "--expected-devkit-revision" not in result.stdout
    assert "--task-id" not in result.stdout


def test_identity_imports_torch_only_when_a_tensor_codec_is_used() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib
import sys

import numpy as np

identity = importlib.import_module("app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity")
assert "torch" not in sys.modules
key = identity.observation_key(np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
assert "torch" not in sys.modules
encoded = identity.encode_observation_key(key)
assert "torch" in sys.modules
assert identity.decode_observation_key(encoded) == key
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)

    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_package_public_exports_are_cached_lazy_attributes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib
import sys

package = importlib.import_module("app.vjepa_cowa_world_model")
assert "app.vjepa_cowa_world_model.losses" not in sys.modules
assert "app.vjepa_cowa_world_model.utils" not in sys.modules
assert "app.vjepa_cowa_world_model.models" not in sys.modules

for name, owner in (
    ("wta_loss", "app.vjepa_cowa_world_model.losses"),
    ("get_status_dim", "app.vjepa_cowa_world_model.utils"),
    ("MultiModalTemporalPlanner", "app.vjepa_cowa_world_model.models"),
):
    assert name not in package.__dict__
    exported = getattr(package, name)
    assert owner in sys.modules
    assert package.__dict__[name] is exported
    assert getattr(package, name) is exported
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment.setdefault("MPLCONFIGDIR", "/tmp")

    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_training_public_exports_have_complete_cached_lazy_routes() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib

training = importlib.import_module("app.vjepa_cowa_world_model.training")
assert set(training._EXPORT_MODULES) == set(training.__all__) | {"MongoRawConfig", "ValuePlanningConfig"}

for name, owner in (
    ("TrainingConfig", "app.vjepa_cowa_world_model.training.config"),
    ("create_train_dataloader", "app.vjepa_cowa_world_model.training.data"),
    ("setup_distributed", "app.vjepa_cowa_world_model.training.distributed"),
    ("EMAUpdater", "app.vjepa_cowa_world_model.training.ema"),
    ("setup_logging", "app.vjepa_cowa_world_model.training.logging"),
    ("TrainingTimer", "app.vjepa_cowa_world_model.training.loop"),
    ("init_encoder", "app.vjepa_cowa_world_model.training.models"),
    ("freeze_parameters", "app.vjepa_cowa_world_model.training.optimizer"),
    ("load_checkpoint", "app.vjepa_cowa_world_model.training.checkpoint"),
    ("RolloutBuffer", "app.vjepa_cowa_world_model.training.rl_buffer"),
    ("require_info_field", "app.vjepa_cowa_world_model.training.rl_hugsim_env"),
    ("compute_policy_mean", "app.vjepa_cowa_world_model.training.rl_policy"),
    ("MongoRawConfig", "app.vjepa_cowa_world_model.training.config"),
    ("ValuePlanningConfig", "app.vjepa_cowa_world_model.training.config"),
):
    exported = getattr(training, name)
    assert training._EXPORT_MODULES[name] == owner
    assert training.__dict__[name] is exported
    assert getattr(training, name) is exported
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    environment.setdefault("MPLCONFIGDIR", "/tmp")

    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=repo_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
