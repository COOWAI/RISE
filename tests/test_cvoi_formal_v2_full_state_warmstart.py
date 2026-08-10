"""Proof-free e120 full-state warm-start tests."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest
import torch

EXPECTED_CHECKPOINT_PATH = "/path/to/checkpoints/rise/e120.pt"
EXPECTED_PARAMS_PRETRAIN_PATH = str(Path(EXPECTED_CHECKPOINT_PATH).with_name("params-pretrain.yaml"))


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor(99.0))


class _Predictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(4, 2, bias=False)


class _Planner(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 3, bias=False)
        self.observed_source_embedding = torch.nn.Embedding(2, 4)


def _warmstart_module() -> ModuleType:
    return importlib.import_module("app.vjepa_cowa_world_model.training.cvoi_formal_v2_full_state_warmstart")


def _source_states(*, module_prefix: bool = False) -> dict[str, dict[str, torch.Tensor]]:
    prefix = "module." if module_prefix else ""
    return {
        "encoder": {
            f"{prefix}stem.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            f"{prefix}stem.bias": torch.tensor([1.0, 2.0]),
            f"{prefix}scale": torch.tensor(2.5),
        },
        "predictor": {
            f"{prefix}projection.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        },
        "planner": {
            f"{prefix}head.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            f"{prefix}observed_source_embedding.weight": torch.tensor(
                [[1.0, -2.0, 3.0, -4.0], [5.0, -6.0, 7.0, -8.0]]
            ),
        },
    }


def _target_states() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "encoder": {
            "stem.weight": torch.zeros(2, 3),
            "stem.bias": torch.zeros(2),
            "scale": torch.tensor(99.0),
        },
        "predictor": {"projection.weight": torch.zeros(2, 4)},
        "planner": {
            "head.weight": torch.zeros(3, 4),
            "observed_source_embedding.weight": torch.full((2, 4), 99.0),
        },
    }


def _modules() -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {
        "encoder": _Encoder(),
        "predictor": _Predictor(),
        "planner": _Planner(),
    }
    with torch.no_grad():
        for module in modules.values():
            for parameter in module.parameters():
                parameter.fill_(99.0)
    return modules


def _checkpoint_payload(*, module_prefix: bool = False) -> dict[str, object]:
    return {
        **_source_states(module_prefix=module_prefix),
        "opt": {"state": "must not be restored"},
        "scheduler": {"last_epoch": 120},
        "epoch": 120,
    }


def _direct_sources(
    tmp_path: Path,
    *,
    payload: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    checkpoint_path = tmp_path / "e120.pt"
    params_path = tmp_path / "params-pretrain.yaml"
    torch.save(_checkpoint_payload() if payload is None else payload, checkpoint_path)
    params_path.write_text("meta:\n  seed: 239\n", encoding="utf-8")
    return checkpoint_path, params_path


def test_public_e120_paths_are_exactly_locked() -> None:
    warmstart = _warmstart_module()

    assert warmstart.FORMAL_V2_E120_CHECKPOINT_PATH == EXPECTED_CHECKPOINT_PATH
    assert warmstart.FORMAL_V2_E120_PARAMS_PRETRAIN_PATH == EXPECTED_PARAMS_PRETRAIN_PATH


def test_retired_verified_and_receipt_api_surface_is_absent() -> None:
    warmstart = _warmstart_module()
    retired_names = {
        "FORMAL_V2_E120_CHECKPOINT_SHA256",
        "FORMAL_V2_E120_PARAMS_PRETRAIN_SHA256",
        "FORMAL_V2_FULL_STATE_WARMSTART_SCHEMA",
        "FORMAL_V2_FULL_STATE_WARMSTART_MODE",
        "FORMAL_V2_FULL_STATE_WARMSTART_NOT_RESTORED",
        "verify_formal_v2_e120_source_artifacts",
        "load_formal_v2_full_state_warmstart",
        "apply_formal_v2_full_state_warmstart",
        "validate_formal_v2_full_state_warmstart_receipt",
        "write_formal_v2_full_state_warmstart_receipt",
        "read_formal_v2_full_state_warmstart_receipt",
    }

    assert not (retired_names & set(vars(warmstart)))


def test_prepare_returns_independent_normalized_model_states_only() -> None:
    warmstart = _warmstart_module()
    payload = _checkpoint_payload(module_prefix=True)
    original_embedding = payload["planner"]["module.observed_source_embedding.weight"]
    original_values = original_embedding.clone()

    prepared = warmstart.prepare_formal_v2_full_state_warmstart(payload, target_states=_target_states())

    assert list(prepared) == ["encoder", "predictor", "planner"]
    assert prepared["planner"]["observed_source_embedding.weight"] is not original_embedding
    assert torch.equal(prepared["planner"]["observed_source_embedding.weight"], original_values)
    original_embedding.zero_()
    assert torch.equal(prepared["planner"]["observed_source_embedding.weight"], original_values)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_role", "missing required roles"),
        ("missing_key", "missing"),
        ("extra_key", "unexpected"),
        ("shape", "shape mismatch"),
        ("zero_embedding", "must not be all zeros"),
    ],
)
def test_prepare_rejects_role_key_shape_and_embedding_drift(mutation: str, message: str) -> None:
    warmstart = _warmstart_module()
    payload = _checkpoint_payload()
    if mutation == "missing_role":
        del payload["predictor"]
    elif mutation == "missing_key":
        del payload["encoder"]["stem.bias"]
    elif mutation == "extra_key":
        payload["predictor"]["unexpected"] = torch.ones(1)
    elif mutation == "shape":
        payload["planner"]["head.weight"] = torch.ones(2, 4)
    else:
        payload["planner"]["observed_source_embedding.weight"] = torch.zeros(2, 4)

    with pytest.raises(ValueError, match=message):
        warmstart.prepare_formal_v2_full_state_warmstart(payload, target_states=_target_states())


def test_direct_apply_restores_all_roles_from_a_portable_absolute_pair(tmp_path: Path) -> None:
    warmstart = _warmstart_module()
    checkpoint_path, params_path = _direct_sources(tmp_path)
    modules = _modules()

    result = warmstart.apply_formal_v2_full_state_warmstart_direct(checkpoint_path, params_path, modules)

    assert result is None
    expected = _source_states()
    for role, module in modules.items():
        actual = module.state_dict()
        assert set(actual) == set(expected[role])
        for key, value in expected[role].items():
            assert torch.equal(actual[key], value)
    assert not any(
        "optimizer" in key or "scheduler" in key for module in modules.values() for key in module.state_dict()
    )


@pytest.mark.parametrize(
    ("source", "kind", "message"),
    [
        ("checkpoint", "missing", "does not exist"),
        ("checkpoint", "directory", "regular file"),
        ("params", "missing", "does not exist"),
        ("params", "directory", "regular file"),
    ],
)
def test_direct_apply_requires_both_locked_sources_to_be_regular_files(
    tmp_path: Path,
    source: str,
    kind: str,
    message: str,
) -> None:
    warmstart = _warmstart_module()
    checkpoint_path, params_path = _direct_sources(tmp_path)
    selected = checkpoint_path if source == "checkpoint" else params_path
    selected.unlink()
    if kind == "directory":
        selected.mkdir()

    with pytest.raises((FileNotFoundError, ValueError), match=message):
        warmstart.apply_formal_v2_full_state_warmstart_direct(checkpoint_path, params_path, _modules())


@pytest.mark.parametrize(
    ("checkpoint_value", "params_value", "message"),
    [
        ("e120.pt", "/opt/rise-user/checkpoints/params-pretrain.yaml", "absolute"),
        (
            "/opt/rise-user/checkpoints/../e120.pt",
            "/opt/rise-user/checkpoints/params-pretrain.yaml",
            r"traversal|\.\.",
        ),
        (
            "/opt/rise-user/checkpoints/e120.pth",
            "/opt/rise-user/checkpoints/params-pretrain.yaml",
            r"\.pt",
        ),
        (
            "/opt/rise-user/checkpoints/e120.pt",
            "/opt/rise-user/checkpoints/pretrain.yaml",
            "params-pretrain.yaml",
        ),
        (
            "/opt/rise-user/checkpoints/e120.pt",
            "/opt/rise-user/config/params-pretrain.yaml",
            "same parent",
        ),
    ],
)
def test_direct_apply_rejects_structurally_invalid_warmstart_pairs(
    checkpoint_value: str,
    params_value: str,
    message: str,
) -> None:
    warmstart = _warmstart_module()

    with pytest.raises(ValueError, match=message):
        warmstart.apply_formal_v2_full_state_warmstart_direct(checkpoint_value, params_value, _modules())
