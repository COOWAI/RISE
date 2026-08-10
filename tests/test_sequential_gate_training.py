from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from app.vjepa_cowa_world_model.training import sequential_gate_training as gate_training
from app.vjepa_cowa_world_model.training.artifact_publish import atomic_torch_save_no_overwrite
from app.vjepa_cowa_world_model.training.sequential_budget_control import SequentialRolloutGate, sequential_gate_loss
from app.vjepa_cowa_world_model.training.sequential_gate_training import (
    load_sequential_gate_checkpoint,
    save_sequential_gate_checkpoint,
    train_sequential_gate_epoch,
)

E120_PROTOCOL = gate_training.SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3
E120_LAMBDA_GRID = [0.0, 0.001, 0.005, 0.01, 0.05]


def _write_legacy_world4drive_gate(
    path: Path,
    gate: SequentialRolloutGate,
    *,
    provenance: dict[str, str],
) -> None:
    """Write a legacy-v1 fixture without exposing a production writer."""

    torch.save(
        {
            "schema": gate_training.SEQUENTIAL_GATE_CHECKPOINT_SCHEMA,
            "feature_schema": gate_training.SEQUENTIAL_GATE_FEATURE_SCHEMA,
            "latent_dim": gate.latent_dim,
            "hidden_dim": gate.hidden_dim,
            "feature_dim": gate.feature_dim,
            "lambda_grid": [0.0],
            "provenance": provenance,
            "state_dict": gate.state_dict(),
        },
        path,
    )


def test_retired_signed_e120_gate_protocol_is_absent() -> None:
    assert not hasattr(gate_training, "SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120")
    assert not hasattr(gate_training, "SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2")
    assert not hasattr(gate_training, "SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_V2")


def test_manual_h4_gate_checkpoint_is_direct_and_round_trips(tmp_path: Path) -> None:
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    provenance = {"oracle": "a"}
    path = tmp_path / "h4.pt"

    save_sequential_gate_checkpoint(
        path,
        gate,
        lambda_grid=E120_LAMBDA_GRID,
        provenance=provenance,
        protocol_version=E120_PROTOCOL,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)

    assert set(payload) == {
        "schema",
        "feature_schema",
        "latent_dim",
        "hidden_dim",
        "feature_dim",
        "lambda_grid",
        "provenance",
        "state_dict",
    }
    assert not any("sha256" in key or "receipt" in key for key in payload)
    loaded = load_sequential_gate_checkpoint(
        path,
        device=torch.device("cpu"),
        expected_provenance=provenance,
        expected_protocol_version=E120_PROTOCOL,
    )
    assert loaded.training is False


def test_legacy_world4drive_gate_protocol_is_read_only(tmp_path: Path) -> None:
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)

    with pytest.raises(ValueError, match="creation supports only"):
        save_sequential_gate_checkpoint(
            tmp_path / "legacy.pt",
            gate,
            lambda_grid=[0.0],
            provenance={"oracle": "legacy"},
            protocol_version=gate_training.SEQUENTIAL_GATE_PROTOCOL_LEGACY,
        )


def test_train_sequential_gate_epoch_updates_parameters():
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-2)
    before = [parameter.detach().clone() for parameter in gate.parameters()]
    batch = {
        "features": torch.randn(6, gate.feature_dim),
        "target_delta": torch.tensor([0.2, 0.1, -0.1, -0.2, 0.3, -0.3]),
        "continue_target": torch.tensor([True, True, False, False, True, False]),
    }

    metrics = train_sequential_gate_epoch(gate, [batch], optimizer=optimizer, device=torch.device("cpu"))

    assert metrics["loss"] > 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, gate.parameters()))


def test_train_sequential_gate_epoch_reports_sample_weighted_metrics() -> None:
    torch.manual_seed(7)
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=0.0)
    batches = [
        {
            "features": torch.randn(size, gate.feature_dim),
            "target_delta": torch.linspace(-0.2, 0.3, size),
            "continue_target": torch.arange(size) % 2 == 0,
        }
        for size in (1, 3)
    ]
    expected = {"loss": 0.0, "classification": 0.0, "regression": 0.0}
    with torch.no_grad():
        for batch in batches:
            output = sequential_gate_loss(
                gate(batch["features"]),
                target_delta=batch["target_delta"],
                continue_target=batch["continue_target"],
            )
            for name in expected:
                expected[name] += float(getattr(output, name)) * len(batch["features"])
    expected = {name: value / 4 for name, value in expected.items()}

    actual = train_sequential_gate_epoch(gate, batches, optimizer=optimizer, device=torch.device("cpu"))

    assert actual == pytest.approx(expected)


def test_sequential_gate_checkpoint_round_trip_and_provenance(tmp_path: Path):
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    path = tmp_path / "gate.pt"
    provenance = {"oracle": "abc", "value": "v", "planner": "p"}
    _write_legacy_world4drive_gate(path, gate, provenance=provenance)

    loaded = load_sequential_gate_checkpoint(
        path,
        device=torch.device("cpu"),
        expected_provenance=provenance,
        expected_protocol_version=gate_training.SEQUENTIAL_GATE_PROTOCOL_LEGACY,
    )

    assert loaded.feature_dim == gate.feature_dim
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_sequential_gate_checkpoint_refuses_overwrite_and_can_retry_after_staging_failure(tmp_path: Path):
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    failed_path = tmp_path / "failed-gate.pt"
    with patch(
        "app.vjepa_cowa_world_model.training.artifact_publish.torch.save",
        side_effect=RuntimeError("injected serialization failure"),
    ):
        with pytest.raises(RuntimeError, match="injected serialization failure"):
            save_sequential_gate_checkpoint(
                failed_path,
                gate,
                lambda_grid=E120_LAMBDA_GRID,
                provenance={"oracle": "a"},
                protocol_version=E120_PROTOCOL,
            )
    assert not failed_path.exists()

    save_sequential_gate_checkpoint(
        failed_path,
        gate,
        lambda_grid=E120_LAMBDA_GRID,
        provenance={"oracle": "a"},
        protocol_version=E120_PROTOCOL,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_sequential_gate_checkpoint(
            failed_path,
            gate,
            lambda_grid=E120_LAMBDA_GRID,
            provenance={"oracle": "a"},
            protocol_version=E120_PROTOCOL,
        )


def test_atomic_checkpoint_publish_cannot_overwrite_a_concurrent_winner(tmp_path: Path) -> None:
    target = tmp_path / "artifact.pt"

    def publish_concurrent_winner(_source, destination):
        Path(destination).write_bytes(b"concurrent-winner")
        raise FileExistsError("simulated atomic EEXIST")

    with patch(
        "app.vjepa_cowa_world_model.training.artifact_publish.os.link",
        side_effect=publish_concurrent_winner,
    ):
        with pytest.raises(FileExistsError, match="appeared while publishing"):
            atomic_torch_save_no_overwrite({"value": torch.tensor(1)}, target)

    assert target.read_bytes() == b"concurrent-winner"
    assert len(list(tmp_path.glob(".artifact.pt.staging-*"))) == 1


def test_sequential_gate_checkpoint_rejects_provenance_drift(tmp_path: Path):
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    path = tmp_path / "gate.pt"
    _write_legacy_world4drive_gate(path, gate, provenance={"oracle": "a"})

    with pytest.raises(ValueError, match="provenance"):
        load_sequential_gate_checkpoint(
            path,
            device=torch.device("cpu"),
            expected_provenance={"oracle": "b"},
            expected_protocol_version=gate_training.SEQUENTIAL_GATE_PROTOCOL_LEGACY,
        )


@pytest.mark.parametrize("invalid_value", [None, 7, "", "   "])
def test_sequential_gate_checkpoint_rejects_non_string_or_blank_raw_provenance(
    tmp_path: Path,
    invalid_value: object,
):
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)

    with pytest.raises(ValueError, match="provenance"):
        save_sequential_gate_checkpoint(
            tmp_path / "gate.pt",
            gate,
            lambda_grid=E120_LAMBDA_GRID,
            provenance={"oracle": invalid_value},
            protocol_version=E120_PROTOCOL,
        )


@pytest.mark.parametrize("invalid_value", [None, 7, "", "   "])
def test_sequential_gate_loader_rejects_invalid_raw_checkpoint_provenance(
    tmp_path: Path,
    invalid_value: object,
):
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    path = tmp_path / "gate.pt"
    _write_legacy_world4drive_gate(path, gate, provenance={"oracle": "valid"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["provenance"] = {"oracle": invalid_value}
    torch.save(payload, path)

    with pytest.raises(ValueError, match="provenance"):
        load_sequential_gate_checkpoint(
            path,
            device=torch.device("cpu"),
            expected_protocol_version=gate_training.SEQUENTIAL_GATE_PROTOCOL_LEGACY,
        )


def test_sequential_gate_loader_preserves_legacy_weights_only_false_default(tmp_path: Path) -> None:
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    path = tmp_path / "gate.pt"
    _write_legacy_world4drive_gate(path, gate, provenance={"oracle": "valid"})
    original_load = torch.load

    with patch.object(gate_training.torch, "load", wraps=original_load) as loader:
        load_sequential_gate_checkpoint(
            path,
            device=torch.device("cpu"),
            expected_protocol_version=gate_training.SEQUENTIAL_GATE_PROTOCOL_LEGACY,
        )

    assert loader.call_args.kwargs["weights_only"] is False


@pytest.mark.parametrize("failure", ["keys", "shape", "dtype"])
def test_sequential_gate_legacy_prevalidation_rejects_before_state_mutation(
    tmp_path: Path,
    failure: str,
) -> None:
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    path = tmp_path / "gate.pt"
    _write_legacy_world4drive_gate(path, gate, provenance={"oracle": "valid"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    first_key = next(iter(state))
    if failure == "keys":
        state["unexpected"] = state.pop(first_key)
    elif failure == "shape":
        shaped_key = next(key for key, value in state.items() if value.ndim > 1)
        state[shaped_key] = state[shaped_key].reshape(-1)
    else:
        state[first_key] = state[first_key].to(dtype=torch.float64)
    torch.save(payload, path)

    with (
        patch.object(
            SequentialRolloutGate,
            "load_state_dict",
            side_effect=AssertionError("state mutation must not start"),
        ),
        pytest.raises(ValueError, match=failure),
    ):
        load_sequential_gate_checkpoint(
            path,
            device=torch.device("cpu"),
            expected_protocol_version=gate_training.SEQUENTIAL_GATE_PROTOCOL_LEGACY,
            _checkpoint_weights_only=True,
            _prevalidate_state_dict=True,
        )


def test_sequential_gate_legacy_validate_only_never_loads_module_state(tmp_path: Path) -> None:
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    path = tmp_path / "gate.pt"
    _write_legacy_world4drive_gate(path, gate, provenance={"oracle": "valid"})

    with patch.object(
        SequentialRolloutGate,
        "load_state_dict",
        side_effect=AssertionError("validate-only must not mutate Gate state"),
    ):
        loaded = load_sequential_gate_checkpoint(
            path,
            device=torch.device("cpu"),
            expected_provenance={"oracle": "valid"},
            expected_protocol_version=gate_training.SEQUENTIAL_GATE_PROTOCOL_LEGACY,
            _checkpoint_weights_only=True,
            _prevalidate_state_dict=True,
            _validate_only=True,
        )

    assert loaded is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("unexpected", "forbidden", "fields"),
        ("latent_dim", True, "latent_dim"),
        ("hidden_dim", "8", "hidden_dim"),
        ("feature_dim", 14.0, "feature_dim"),
        ("lambda_grid", [True], "lambda_grid"),
        ("lambda_grid", ["0.0"], "lambda_grid"),
        ("lambda_grid", [0], "lambda_grid"),
    ],
)
def test_sequential_gate_legacy_prevalidation_rejects_metadata_type_or_field_drift(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    gate = SequentialRolloutGate(latent_dim=2, hidden_dim=8)
    path = tmp_path / "gate.pt"
    _write_legacy_world4drive_gate(path, gate, provenance={"oracle": "valid"})
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, path)

    with (
        patch.object(
            SequentialRolloutGate,
            "load_state_dict",
            side_effect=AssertionError("metadata validation must precede state mutation"),
        ) as load_state,
        pytest.raises(ValueError, match=message),
    ):
        load_sequential_gate_checkpoint(
            path,
            device=torch.device("cpu"),
            expected_protocol_version=gate_training.SEQUENTIAL_GATE_PROTOCOL_LEGACY,
            _checkpoint_weights_only=True,
            _prevalidate_state_dict=True,
        )

    load_state.assert_not_called()
