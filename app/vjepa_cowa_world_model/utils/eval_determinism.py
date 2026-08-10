"""Deterministic RNG helpers for validation/evaluation loops."""

from __future__ import annotations

import contextlib
import hashlib
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def extract_batch_metadata(sample) -> Optional[Dict[str, Any]]:
    if isinstance(sample, (list, tuple)) and sample and isinstance(sample[-1], dict):
        return sample[-1]
    return None


def _get_nested_value(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if current is None:
            return default
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    return current if current is not None else default


def normalize_seed_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): normalize_seed_value(value[key]) for key in sorted(value.keys())}
    if isinstance(value, (list, tuple)):
        return [normalize_seed_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def make_validation_rng_seed(
    config: Any,
    *,
    stage: str,
    epoch: int,
    batch_idx: int,
    metadata: Optional[Dict[str, Any]],
    rank: int,
) -> int:
    base_seed = int(_get_nested_value(config, "meta", "seed", default=0))
    # meta.val_stable_noise (default True): hold per-sample sampling noise FIXED across epochs by
    # dropping `epoch` from the seed key. This makes the validation curve vary only with model
    # weights instead of with fresh per-epoch sampling draws, eliminating the epoch-to-epoch jitter
    # that turns single-epoch "best" / best_open_loop.pt into a lucky-draw artifact. The seed stays
    # per-sample (keyed on metadata) so different samples still get decorrelated noise, just stable
    # across epochs. Set meta.val_stable_noise=False to restore the legacy per-epoch-varying noise.
    stable_noise = bool(_get_nested_value(config, "meta", "val_stable_noise", default=True))
    epoch_key = 0 if stable_noise else int(epoch)
    metadata_identity = normalize_seed_value(metadata) if metadata else None
    fallback_identity = {"rank": int(rank), "batch_idx": int(batch_idx)}
    payload = (base_seed, str(stage), epoch_key, metadata_identity or fallback_identity)
    digest = hashlib.blake2b(repr(payload).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def _cuda_devices_for(device: torch.device) -> list[int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


@contextlib.contextmanager
def preserve_eval_rng_state(device: torch.device):
    """Preserve Python, NumPy, Torch CPU, and the selected local CUDA RNG."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = _cuda_devices_for(device)
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _seed_local_eval_rng(seed: int, device: torch.device) -> None:
    """Seed only CPU and the selected local CUDA generator."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.default_generator.manual_seed(seed)
    cuda_devices = _cuda_devices_for(device)
    if cuda_devices:
        with torch.cuda.device(cuda_devices[0]):
            torch.cuda.manual_seed(seed)


@contextlib.contextmanager
def deterministic_eval_rng(seed: int, device: torch.device):
    """Temporarily seed evaluation randomness without changing training RNG state."""

    with preserve_eval_rng_state(device):
        _seed_local_eval_rng(seed, device)
        yield
