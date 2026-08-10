"""Training and checkpoint protocol for the sequential CVoI rollout gate."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch

from app.vjepa_cowa_world_model.training.artifact_publish import (
    atomic_torch_save_no_overwrite,
    atomic_torch_save_replace,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import FORMAL_V2_NAVSIM_E120_LAMBDA_GRID
from app.vjepa_cowa_world_model.training.sequential_budget_control import (
    CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
    SequentialRolloutGate,
    sequential_gate_loss,
)

SEQUENTIAL_GATE_CHECKPOINT_SCHEMA = "sequential_cvoi_gate_v1"
SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_NAVSIM_E120 = "sequential_cvoi_gate_navsim_e120_v1"
SEQUENTIAL_GATE_FEATURE_SCHEMA = "obs,prefix,field,stop,slope,horizon,current_cost,next_cost,lambda"
SEQUENTIAL_GATE_PROTOCOL_LEGACY = "legacy_v1"
SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3 = "formal_v2_navsim_e120_h4_v3"


def _validate_provenance(provenance: Mapping[str, str]) -> Dict[str, str]:
    if (
        not isinstance(provenance, Mapping)
        or not provenance
        or not all(
            isinstance(key, str) and bool(key.strip()) and isinstance(value, str) and bool(value.strip())
            for key, value in provenance.items()
        )
    ):
        raise ValueError("sequential Gate provenance must be a non-empty string mapping")
    return dict(provenance)


def _validate_lambda_grid(lambda_grid: Sequence[float]) -> list[float]:
    values = [float(value) for value in lambda_grid]
    if not values or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("lambda_grid must contain finite, non-negative values")
    return values


def validate_formal_v2_lambda_grid(lambda_grid: Sequence[float]) -> list[float]:
    """Require the immutable five-point NavSim-e120 Gate grid."""

    values = _validate_lambda_grid(lambda_grid)
    expected = list(FORMAL_V2_NAVSIM_E120_LAMBDA_GRID)
    if values != expected:
        raise ValueError(f"NavSim-e120 Gate requires the fixed lambda grid {expected}, got {values}")
    return values


def train_sequential_gate_epoch(
    gate: SequentialRolloutGate,
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 0.05,
    regression_weight: float = 0.5,
) -> Dict[str, float]:
    """Train one epoch from distilled marginal-utility examples."""

    gate.train()
    totals = {"loss": 0.0, "classification": 0.0, "regression": 0.0}
    num_examples = 0
    for batch in batches:
        missing = sorted({"features", "target_delta", "continue_target"} - set(batch))
        if missing:
            raise ValueError(f"sequential Gate batch is missing required keys: {missing}")
        features = batch["features"].to(device=device, dtype=torch.float32)
        target_delta = batch["target_delta"].to(device=device, dtype=torch.float32)
        continue_target = batch["continue_target"].to(device=device, dtype=torch.bool)
        if features.ndim != 2 or int(features.shape[0]) < 1:
            raise ValueError("sequential Gate training features must be a non-empty [B,F] tensor")
        batch_size = int(features.shape[0])

        optimizer.zero_grad(set_to_none=True)
        predicted_delta = gate(features)
        loss_output = sequential_gate_loss(
            predicted_delta,
            target_delta=target_delta,
            continue_target=continue_target,
            temperature=temperature,
            regression_weight=regression_weight,
        )
        if not torch.isfinite(loss_output.loss):
            raise ValueError("sequential Gate loss is not finite")
        loss_output.loss.backward()
        optimizer.step()

        totals["loss"] += float(loss_output.loss.detach().cpu()) * batch_size
        totals["classification"] += float(loss_output.classification.detach().cpu()) * batch_size
        totals["regression"] += float(loss_output.regression.detach().cpu()) * batch_size
        num_examples += batch_size
    if num_examples == 0:
        raise ValueError("sequential Gate training requires at least one batch")
    return {name: value / num_examples for name, value in totals.items()}


def _sequential_gate_checkpoint_payload(
    gate: SequentialRolloutGate,
    *,
    lambda_grid: Sequence[float],
    provenance: Mapping[str, str],
    protocol_version: str,
) -> Dict[str, object]:
    """Build one direct NavSim-e120 Gate checkpoint payload."""

    if protocol_version != SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3:
        raise ValueError(
            "Gate checkpoint creation supports only "
            f"{SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3!r}, got {protocol_version!r}"
        )
    return {
        "schema": SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_NAVSIM_E120,
        "feature_schema": CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA,
        "latent_dim": gate.latent_dim,
        "hidden_dim": gate.hidden_dim,
        "feature_dim": gate.feature_dim,
        "lambda_grid": validate_formal_v2_lambda_grid(lambda_grid),
        "provenance": _validate_provenance(provenance),
        "state_dict": gate.state_dict(),
    }


def save_sequential_gate_checkpoint(
    path: str | Path,
    gate: SequentialRolloutGate,
    *,
    lambda_grid: Sequence[float],
    provenance: Mapping[str, str],
    protocol_version: str,
) -> None:
    """Create one direct NavSim-e120 Gate checkpoint without overwriting."""

    payload = _sequential_gate_checkpoint_payload(
        gate,
        lambda_grid=lambda_grid,
        provenance=provenance,
        protocol_version=protocol_version,
    )
    atomic_torch_save_no_overwrite(payload, path)


def save_sequential_gate_checkpoint_replace(
    path: str | Path,
    gate: SequentialRolloutGate,
    *,
    lambda_grid: Sequence[float],
    provenance: Mapping[str, str],
    protocol_version: str,
) -> None:
    """Failure-atomically replace one direct NavSim-e120 Gate checkpoint."""

    payload = _sequential_gate_checkpoint_payload(
        gate,
        lambda_grid=lambda_grid,
        provenance=provenance,
        protocol_version=protocol_version,
    )
    atomic_torch_save_replace(payload, path)


def load_sequential_gate_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    expected_protocol_version: str,
    expected_provenance: Mapping[str, str] | None = None,
    _checkpoint_weights_only: Optional[bool] = None,
    _prevalidate_state_dict: bool = False,
    _validate_only: bool = False,
) -> Optional[SequentialRolloutGate]:
    """Load and freeze a Gate only when schema and provenance match exactly."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"sequential Gate checkpoint does not exist: {path}")
    weights_only = False if _checkpoint_weights_only is None else _checkpoint_weights_only
    if type(weights_only) is not bool:
        raise TypeError("_checkpoint_weights_only must be bool or None")
    if type(_prevalidate_state_dict) is not bool:
        raise TypeError("_prevalidate_state_dict must be bool")
    if type(_validate_only) is not bool:
        raise TypeError("_validate_only must be bool")
    if _validate_only and not _prevalidate_state_dict:
        raise ValueError("_validate_only requires _prevalidate_state_dict=True")
    payload = torch.load(path, map_location=device, weights_only=weights_only)
    if not isinstance(payload, Mapping):
        raise ValueError("sequential Gate checkpoint must contain a mapping")
    if expected_protocol_version == SEQUENTIAL_GATE_PROTOCOL_LEGACY:
        expected_schema = SEQUENTIAL_GATE_CHECKPOINT_SCHEMA
        expected_feature_schema = SEQUENTIAL_GATE_FEATURE_SCHEMA
        if _prevalidate_state_dict:
            expected_fields = {
                "schema",
                "feature_schema",
                "latent_dim",
                "hidden_dim",
                "feature_dim",
                "lambda_grid",
                "provenance",
                "state_dict",
            }
            missing = expected_fields - set(payload)
            unexpected = set(payload) - expected_fields
            if missing or unexpected:
                raise ValueError(
                    "invalid direct legacy sequential Gate checkpoint fields: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected, key=str)}"
                )
            for field in ("latent_dim", "hidden_dim", "feature_dim"):
                value = payload[field]
                if type(value) is not int or value <= 0:
                    raise ValueError(f"direct legacy sequential Gate {field} must be a positive int")
            lambda_grid = payload["lambda_grid"]
            if (
                type(lambda_grid) is not list
                or not lambda_grid
                or any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in lambda_grid)
            ):
                raise ValueError("direct legacy sequential Gate lambda_grid must be a non-empty list of finite floats")
    elif expected_protocol_version == SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3:
        expected_schema = SEQUENTIAL_GATE_CHECKPOINT_SCHEMA_NAVSIM_E120
        expected_feature_schema = CVOI_FORMAL_V2_GATE_FEATURE_SCHEMA
        expected_fields = {
            "schema",
            "feature_schema",
            "latent_dim",
            "hidden_dim",
            "feature_dim",
            "lambda_grid",
            "provenance",
            "state_dict",
        }
        missing = expected_fields - set(payload)
        unexpected = set(payload) - expected_fields
        if missing or unexpected:
            raise ValueError(
                "invalid NavSim-e120 sequential Gate checkpoint fields: "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected, key=str)}"
            )
    else:
        raise ValueError(
            "sequential Gate loading supports only read-only legacy_v1 and direct NavSim-e120 checkpoints, "
            f"got {expected_protocol_version!r}"
        )
    if payload.get("schema") != expected_schema:
        raise ValueError(f"unexpected sequential Gate checkpoint schema: {payload.get('schema')!r}")
    if payload.get("feature_schema") != expected_feature_schema:
        raise ValueError("sequential Gate feature schema does not match the runtime schema")
    provenance = _validate_provenance(payload.get("provenance", {}))
    if expected_provenance is not None and provenance != _validate_provenance(expected_provenance):
        raise ValueError("sequential Gate checkpoint provenance does not match expected provenance")
    if expected_protocol_version == SEQUENTIAL_GATE_PROTOCOL_FORMAL_V2_NAVSIM_E120_H4_V3:
        validate_formal_v2_lambda_grid(payload.get("lambda_grid", []))
    else:
        _validate_lambda_grid(payload.get("lambda_grid", []))

    gate = SequentialRolloutGate(latent_dim=int(payload["latent_dim"]), hidden_dim=int(payload["hidden_dim"]))
    if int(payload.get("feature_dim", -1)) != gate.feature_dim:
        raise ValueError("sequential Gate checkpoint feature_dim is inconsistent with its architecture")
    if _prevalidate_state_dict:
        state = payload.get("state_dict")
        if not isinstance(state, Mapping) or any(
            not isinstance(key, str) or not key or not torch.is_tensor(value) for key, value in state.items()
        ):
            raise ValueError("sequential Gate state_dict must map non-empty string keys to tensors")
        expected_state = gate.state_dict()
        if set(state) != set(expected_state):
            raise ValueError(
                "sequential Gate state keys mismatch: "
                f"missing={sorted(set(expected_state) - set(state))}, "
                f"unexpected={sorted(set(state) - set(expected_state))}"
            )
        shape_mismatch = {
            key: (tuple(state[key].shape), tuple(expected_state[key].shape))
            for key in expected_state
            if state[key].shape != expected_state[key].shape
        }
        if shape_mismatch:
            raise ValueError(f"sequential Gate state shape mismatch: {shape_mismatch}")
        dtype_mismatch = {
            key: (state[key].dtype, expected_state[key].dtype)
            for key in expected_state
            if state[key].dtype != expected_state[key].dtype
        }
        if dtype_mismatch:
            raise ValueError(f"sequential Gate state dtype mismatch: {dtype_mismatch}")
    if _validate_only:
        return None
    gate.load_state_dict(payload["state_dict"], strict=True)
    gate.to(device=device)
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    return gate
