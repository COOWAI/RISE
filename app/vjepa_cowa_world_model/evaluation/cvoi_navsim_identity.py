"""Path-independent identities shared by isolated Native NavSim evaluation."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

_OBSERVATION_KEY_NAMESPACE = b"cvoi-navsim-decoded-observation-v1\x00"
_OBSERVATION_KEY_PATTERN = re.compile(r"[0-9a-f]{64}")
_UINT64_LIMIT = 1 << 64


def observation_key(image: np.ndarray) -> str:
    """Hash a decoded image's exact dtype, shape, and C-order contents.

    Parameters
    ----------
    image : numpy.ndarray
        Raw decoded final-history camera image.

    Returns
    -------
    str
        A 64-character lowercase SHA256 hexadecimal digest.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy ndarray, got {type(image).__name__}")
    if image.dtype.hasobject:
        raise ValueError("image object dtype is not supported")

    dtype_bytes = image.dtype.str.encode("utf-8")
    digest = hashlib.sha256()
    digest.update(_OBSERVATION_KEY_NAMESPACE)
    digest.update(len(dtype_bytes).to_bytes(4, byteorder="big", signed=False))
    digest.update(dtype_bytes)
    digest.update(image.ndim.to_bytes(4, byteorder="big", signed=False))
    for dimension in image.shape:
        digest.update(dimension.to_bytes(8, byteorder="big", signed=False))
    digest.update(np.ascontiguousarray(image).tobytes(order="C"))
    return digest.hexdigest()


def _validated_observation_key(value: object) -> str:
    if type(value) is not str or _OBSERVATION_KEY_PATTERN.fullmatch(value) is None:
        raise ValueError("observation key must be exactly 64 lowercase hexadecimal characters")
    return value


def encode_observation_key(value: str) -> torch.Tensor:
    """Encode a hexadecimal observation key as an exact ``torch.uint8[32]`` tensor."""

    import torch

    key = _validated_observation_key(value)
    return torch.tensor(list(bytes.fromhex(key)), dtype=torch.uint8)


def observation_key_tensor(value: str) -> torch.Tensor:
    """Alias used by the scorer-facing feature builder."""

    return encode_observation_key(value)


def decode_observation_key(value: torch.Tensor) -> str:
    """Decode an exact ``torch.uint8[32]`` tensor to lowercase hexadecimal."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"observation key value must be a torch tensor, got {type(value).__name__}")
    if value.dtype is not torch.uint8:
        raise ValueError(f"observation key tensor dtype must be torch.uint8, got {value.dtype}")
    if tuple(value.shape) != (32,):
        raise ValueError(f"observation key tensor shape must be [32], got {tuple(value.shape)}")
    return bytes(value.detach().cpu().tolist()).hex()


def _validated_unsigned_seed(value: object) -> int:
    if type(value) is not int:
        raise TypeError(f"seed must be an unsigned 64-bit integer, got {type(value).__name__}")
    if value < 0 or value >= _UINT64_LIMIT:
        raise ValueError(f"seed must be an unsigned 64-bit integer in [0, 2**64), got {value}")
    return value


def encode_unsigned_seed(value: int) -> torch.Tensor:
    """Encode the full unsigned 64-bit seed as ``torch.uint8[8]`` big-endian bytes."""

    import torch

    seed = _validated_unsigned_seed(value)
    return torch.tensor(list(seed.to_bytes(8, byteorder="big", signed=False)), dtype=torch.uint8)


def unsigned_seed_tensor(value: int) -> torch.Tensor:
    """Alias used by the scorer-facing feature builder."""

    return encode_unsigned_seed(value)


def decode_unsigned_seed(value: torch.Tensor) -> int:
    """Decode an exact ``torch.uint8[8]`` tensor without signed truncation."""

    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"seed value must be a torch tensor, got {type(value).__name__}")
    if value.dtype is not torch.uint8:
        raise ValueError(f"seed tensor dtype must be torch.uint8, got {value.dtype}")
    if tuple(value.shape) != (8,):
        raise ValueError(f"seed tensor shape must be [8], got {tuple(value.shape)}")
    return int.from_bytes(bytes(value.detach().cpu().tolist()), byteorder="big", signed=False)
