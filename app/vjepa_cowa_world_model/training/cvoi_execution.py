"""Shared numerical and random execution contract for offline and deployed CVoI."""

from __future__ import annotations

import hashlib
import random
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Iterator, Sequence

import numpy as np
import torch

CVOI_EXECUTION_DTYPE_SCHEMA = "cvoi_execution_dtype_v1"
CVOI_INFERENCE_RNG_SCHEMA = "cvoi_sample_rng_v1"
_RNG_NAMESPACE = "cvoi-navsim-v1"
_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@contextmanager
def common_random_numbers(seed: int) -> Iterator[None]:
    """Reset and restore global RNGs around one fixed-policy horizon run."""

    if type(seed) is not int or seed < 0:
        raise ValueError(f"CVoI common-random seed must be a non-negative integer, got {seed!r}")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng():
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _configured_dtype_name(config: Any) -> str:
    meta_dtype = getattr(getattr(config, "meta", None), "dtype", None)
    if isinstance(meta_dtype, str):
        name = meta_dtype.strip().lower()
    else:
        direct_dtype = getattr(config, "dtype", None)
        names = {value: key for key, value in _DTYPE_BY_NAME.items()}
        name = names.get(direct_dtype, "")
    if name not in _DTYPE_BY_NAME:
        raise ValueError(f"CVoI execution dtype must be one of {sorted(_DTYPE_BY_NAME)}, got {meta_dtype!r}")
    return name


def cvoi_execution_dtype_signature(config: Any) -> dict[str, object]:
    dtype_name = _configured_dtype_name(config)
    deterministic = bool(getattr(getattr(config, "meta", None), "deterministic", False))
    return {
        "schema": CVOI_EXECUTION_DTYPE_SCHEMA,
        "required_device": "cuda",
        "input_storage_dtype": "float32",
        "cuda_autocast_enabled": dtype_name in {"bfloat16", "float16"},
        "cuda_autocast_dtype": dtype_name,
        "non_cuda_autocast_enabled": False,
        "value_gate_dtype": "float32",
        "backend_policy": {
            "deterministic_algorithms": deterministic,
            "deterministic_warn_only": deterministic,
            "cudnn_deterministic": deterministic,
            "cudnn_benchmark": not deterministic,
        },
    }


def seed_cvoi_process(config: Any) -> None:
    """Seed single-process data/model initialization before creating loaders or modules."""

    seed = getattr(getattr(config, "meta", None), "seed", None)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"CVoI process seed must be a non-negative integer, got {seed!r}")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    deterministic = bool(getattr(getattr(config, "meta", None), "deterministic", False))
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    try:
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)
    except TypeError:
        torch.use_deterministic_algorithms(deterministic)


def cvoi_execution_autocast(config: Any, device: torch.device | str) -> AbstractContextManager:
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"CVoI execution supports only cpu/cuda devices, got {device.type!r}")
    dtype_name = _configured_dtype_name(config)
    return torch.autocast(
        device_type=device.type,
        dtype=_DTYPE_BY_NAME[dtype_name],
        enabled=device.type == "cuda" and dtype_name in {"bfloat16", "float16"},
    )


def cvoi_inference_rng_signature(config: Any) -> dict[str, object]:
    seed = resolve_cvoi_evaluation_seed(config)
    return {
        "schema": CVOI_INFERENCE_RNG_SCHEMA,
        "base_seed": seed,
        "namespace": _RNG_NAMESPACE,
        "sample_identity": "stable_sample_id",
        "digest": "sha256_first_64_big_endian",
        "common_across_horizons_and_lambda": True,
        "planner_noise": "explicit_per_sample_torch_randn_float32_v1",
    }


def resolve_cvoi_evaluation_seed(config: Any) -> int:
    """Return the formal common evaluation seed, or the legacy training seed."""

    signature = getattr(getattr(config, "cvoi", None), "ablation_signature", None)
    seed = (
        getattr(signature, "evaluation_seed", None)
        if signature is not None
        else getattr(getattr(config, "meta", None), "seed", None)
    )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"CVoI inference requires a non-negative integer evaluation seed, got {seed!r}")
    return seed


def cvoi_sample_seed(base_seed: int, stable_sample_id: str) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError(f"CVoI base seed must be a non-negative integer, got {base_seed!r}")
    if not isinstance(stable_sample_id, str) or not stable_sample_id.strip():
        raise ValueError("CVoI sample seed requires a non-empty stable_sample_id")
    payload = f"{_RNG_NAMESPACE}:{base_seed}:{stable_sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def cvoi_planner_inference_noise(
    planner: object,
    *,
    seeds: Sequence[int],
    device: torch.device | str,
) -> torch.Tensor:
    """Generate the exact per-sample diffusion noise shared by val, Oracle, and deployment."""

    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence) or not seeds:
        raise ValueError("CVoI planner noise requires a non-empty seed sequence")
    normalized_seeds = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"CVoI planner noise seed must be a non-negative integer, got {seed!r}")
        normalized_seeds.append(seed)
    core = planner.module if hasattr(planner, "module") else planner
    required = ("num_modes", "num_samples", "num_poses", "traj_dim")
    missing = [name for name in required if not hasattr(core, name)]
    if missing:
        raise ValueError(f"CVoI diffusion planner is missing noise-shape attributes: {missing}")
    num_modes = int(core.num_modes) if int(core.num_modes) > 1 else int(core.num_samples)
    num_poses = int(core.num_poses)
    traj_dim = int(core.traj_dim)
    if num_modes <= 0 or num_poses <= 0 or traj_dim <= 0:
        raise ValueError("CVoI diffusion planner noise dimensions must be positive")
    device = torch.device(device)
    rows = []
    for seed in normalized_seeds:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        rows.append(
            torch.randn(
                (1, num_modes, num_poses, traj_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        )
    return torch.cat(rows, dim=0)


class CvoiValueDtypeAdapter(torch.nn.Module):
    """Run the Value model in FP32 while preserving gradients to lower-precision latents."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("CVoI Value dtype adapter requires a torch.nn.Module")
        dtypes = {value.dtype for value in (*model.parameters(), *model.buffers()) if value.is_floating_point()}
        if dtypes != {torch.float32}:
            raise ValueError(f"CVoI Value model must have exactly FP32 floating state, got {sorted(map(str, dtypes))}")
        self.model = model

    def forward(
        self,
        z_observed: torch.Tensor,
        z_future: torch.Tensor,
        *,
        tokens_per_frame: int,
    ) -> object:
        if z_observed.device != z_future.device:
            raise ValueError("CVoI Value latent inputs must share one device")
        with torch.autocast(device_type=z_observed.device.type, enabled=False):
            return self.model(
                z_observed.to(dtype=torch.float32),
                z_future.to(dtype=torch.float32),
                tokens_per_frame=tokens_per_frame,
            )
