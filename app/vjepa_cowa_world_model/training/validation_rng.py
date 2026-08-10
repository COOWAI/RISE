"""Sample-keyed random streams for deterministic validation protocols.

Validation randomness is derived only from a stable dataset identity and the
semantic protocol key.  Execution-layout details such as rank, world size,
batch position, and epoch are intentionally not accepted by this API.
"""

from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Any, Mapping, Optional, Sequence

import torch

_MAX_TORCH_SEED = 2**63 - 1
VALIDATION_RNG_CONTRACT_VERSION = "sample_keyed_common_random_v2"


def _nonempty_key(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def validation_seed(
    *,
    base_seed: int,
    sample_id: str,
    protocol: str,
    horizon: Optional[int],
    stream: str,
) -> int:
    """Derive a stable torch seed from a semantic validation key.

    Python's process-randomized ``hash()`` is deliberately never used.
    ``horizon=None`` is the explicit full-horizon protocol key.
    """

    if not isinstance(base_seed, Integral) or isinstance(base_seed, bool):
        raise TypeError(f"base_seed must be an integer, got {type(base_seed).__name__}")
    sample_id = _nonempty_key(sample_id, name="sample_id")
    protocol = _nonempty_key(protocol, name="protocol")
    stream = _nonempty_key(stream, name="stream")
    if horizon is not None and (not isinstance(horizon, Integral) or isinstance(horizon, bool)):
        raise TypeError(f"horizon must be an integer or None, got {type(horizon).__name__}")
    if horizon is not None and int(horizon) < 0:
        raise ValueError(f"horizon must be non-negative or None, got {horizon}")

    payload = json.dumps(
        {
            "base_seed": int(base_seed),
            "horizon": None if horizon is None else int(horizon),
            "protocol": protocol,
            "sample_id": sample_id,
            "stream": stream,
            "version": 1,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=b"vjepa-val-rng").digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % _MAX_TORCH_SEED


def validation_batch_seeds(
    *,
    base_seed: int,
    sample_ids: Sequence[str],
    protocol: str,
    horizon: Optional[int],
    stream: str,
) -> tuple[int, ...]:
    """Return one semantic RNG seed per sample, preserving input order."""

    if isinstance(sample_ids, (str, bytes)):
        raise TypeError("sample_ids must be a sequence of sample identity strings")
    return tuple(
        validation_seed(
            base_seed=base_seed,
            sample_id=sample_id,
            protocol=protocol,
            horizon=horizon,
            stream=stream,
        )
        for sample_id in sample_ids
    )


def make_validation_generator(
    *,
    base_seed: int,
    sample_id: str,
    protocol: str,
    horizon: Optional[int],
    stream: str,
    device: torch.device,
) -> torch.Generator:
    """Construct a device-local generator without reading global RNG state."""

    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(
        validation_seed(
            base_seed=base_seed,
            sample_id=sample_id,
            protocol=protocol,
            horizon=horizon,
            stream=stream,
        )
    )
    return generator


def _validation_random_tensor(
    shape: Sequence[int],
    *,
    distribution: str,
    base_seed: int,
    sample_ids: Sequence[str],
    protocol: str,
    horizon: Optional[int],
    stream: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    normalized_shape = tuple(int(value) for value in shape)
    if not normalized_shape or any(value < 0 for value in normalized_shape):
        raise ValueError(f"shape must contain non-negative dimensions including a batch axis, got {shape}")
    if normalized_shape[0] != len(sample_ids):
        raise ValueError(
            f"shape batch dimension {normalized_shape[0]} does not match sample_ids length {len(sample_ids)}"
        )
    if not dtype.is_floating_point:
        raise TypeError(f"validation random tensors require a floating dtype, got {dtype}")
    if distribution not in {"rand", "randn"}:
        raise ValueError(f"unsupported validation random distribution: {distribution!r}")

    tail_shape = normalized_shape[1:]
    samples = []
    for sample_id in sample_ids:
        generator = make_validation_generator(
            base_seed=base_seed,
            sample_id=sample_id,
            protocol=protocol,
            horizon=horizon,
            stream=stream,
            device=device,
        )
        random_fn = torch.rand if distribution == "rand" else torch.randn
        samples.append(random_fn(tail_shape, generator=generator, device=device, dtype=dtype))
    if not samples:
        return torch.empty(normalized_shape, device=device, dtype=dtype)
    return torch.stack(samples, dim=0)


def validation_randn(
    shape: Sequence[int],
    *,
    base_seed: int,
    sample_ids: Sequence[str],
    protocol: str,
    horizon: Optional[int],
    stream: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw standard-normal values independently from each sample's stream."""

    return _validation_random_tensor(
        shape,
        distribution="randn",
        base_seed=base_seed,
        sample_ids=sample_ids,
        protocol=protocol,
        horizon=horizon,
        stream=stream,
        device=device,
        dtype=dtype,
    )


def validation_rand(
    shape: Sequence[int],
    *,
    base_seed: int,
    sample_ids: Sequence[str],
    protocol: str,
    horizon: Optional[int],
    stream: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw uniform ``[0, 1)`` values independently from each sample's stream."""

    return _validation_random_tensor(
        shape,
        distribution="rand",
        base_seed=base_seed,
        sample_ids=sample_ids,
        protocol=protocol,
        horizon=horizon,
        stream=stream,
        device=device,
        dtype=dtype,
    )


def _normalize_sample_id(value: Any) -> str:
    if torch.is_tensor(value):
        if value.ndim != 0:
            raise ValueError(f"stable_sample_id entries must be scalar, got tensor shape {tuple(value.shape)}")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise TypeError("stable_sample_id entries must be strings or integers, " f"got {type(value).__name__}")
    return _nonempty_key(str(value), name="stable_sample_id")


def resolve_stable_sample_ids(metadata: Mapping[str, Any], *, batch_size: int) -> tuple[str, ...]:
    """Extract the mandatory, collated per-sample dataset identity.

    There is intentionally no path/metadata/rank/batch fallback: such a fallback
    would make validation noise depend on execution layout or mutable storage.
    """

    if not isinstance(metadata, Mapping):
        raise TypeError("validation metadata must be a mapping containing stable_sample_id")
    if "stable_sample_id" not in metadata:
        raise KeyError("validation metadata is missing required stable_sample_id")
    if not isinstance(batch_size, Integral) or isinstance(batch_size, bool) or int(batch_size) < 0:
        raise ValueError(f"batch_size must be a non-negative integer, got {batch_size!r}")

    raw = metadata["stable_sample_id"]
    if torch.is_tensor(raw):
        if raw.ndim == 0:
            values = [raw]
        elif raw.ndim == 1:
            values = list(raw.unbind(0))
        else:
            raise ValueError(f"metadata.stable_sample_id must be rank 0 or 1, got shape {tuple(raw.shape)}")
    elif isinstance(raw, (str, Integral)) and not isinstance(raw, bool):
        values = [raw]
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        raise TypeError(
            "metadata.stable_sample_id must be a scalar or sequence of strings/integers, " f"got {type(raw).__name__}"
        )
    if len(values) != int(batch_size):
        raise ValueError(
            f"metadata.stable_sample_id must contain batch_size={int(batch_size)} values, got {len(values)}"
        )
    return tuple(_normalize_sample_id(value) for value in values)
