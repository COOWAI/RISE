"""Strict sample-level supervision masks for counterfactual protocol v2."""

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch

REAL_DOMAIN = "real"
COUNTERFACTUAL_DOMAIN = "counterfactual"
VALID_DATASET_DOMAINS = {REAL_DOMAIN, COUNTERFACTUAL_DOMAIN}
EGO_HAZARD_TYPES = {"自车行为引起", "非自车行为引起"}
OTHER_ACCIDENT_TYPES = {"有事故但与自车无关"}
KNOWN_HAZARD_TYPES = EGO_HAZARD_TYPES | OTHER_ACCIDENT_TYPES


@dataclass(frozen=True)
class CounterfactualSampleMasks:
    """Normalized boolean masks for one real/counterfactual batch."""

    real: torch.Tensor
    cf_safe: torch.Tensor
    cf_hazard: torch.Tensor
    cf_ego_hazard: torch.Tensor
    cf_other_accident: torch.Tensor
    imitation: torch.Tensor
    world_model: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class DistributedMaskNormalization:
    """DDP scaling for a local mean over a globally masked batch."""

    local_count: torch.Tensor
    global_count: torch.Tensor
    mean_scale: torch.Tensor


def _metadata_list(metadata: Dict[str, Any], name: str) -> Sequence[Any]:
    value = metadata.get(name)
    if value is None:
        raise ValueError(f"counterfactual supervision requires metadata[{name!r}]")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"metadata[{name!r}] must be a batch sequence, got {type(value).__name__}")
    return value


def _metadata_bool_tensor(
    metadata: Dict[str, Any],
    name: str,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    value = metadata.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"metadata[{name!r}] must be a bool tensor")
    if value.dtype != torch.bool or value.ndim != 1 or int(value.shape[0]) != batch_size:
        raise ValueError(
            f"metadata[{name!r}] must have dtype=bool and shape [{batch_size}], "
            f"got dtype={value.dtype}, shape={tuple(value.shape)}"
        )
    return value.to(device=device)


def build_counterfactual_sample_masks(
    metadata: Dict[str, Any],
    device: torch.device,
) -> CounterfactualSampleMasks:
    """Build strict v2 world/value/imitation masks from normalized metadata."""

    if not isinstance(metadata, dict):
        raise ValueError("counterfactual supervision requires batch metadata")
    domains = [str(value) for value in _metadata_list(metadata, "dataset_domain")]
    batch_size = len(domains)
    if batch_size == 0:
        raise ValueError("counterfactual supervision does not accept an empty batch")
    unknown_domains = sorted(set(domains) - VALID_DATASET_DOMAINS)
    if unknown_domains:
        raise ValueError(f"metadata['dataset_domain'] contains unknown domains: {unknown_domains}")

    annotation_valid = _metadata_bool_tensor(
        metadata,
        "cf_annotation_valid",
        batch_size=batch_size,
        device=device,
    )
    is_hazard = _metadata_bool_tensor(metadata, "cf_is_hazard", batch_size=batch_size, device=device)
    hazard_types = [str(value) for value in _metadata_list(metadata, "cf_hazard_type")]
    if len(hazard_types) != batch_size:
        raise ValueError(
            f"metadata['cf_hazard_type'] length {len(hazard_types)} does not match batch size {batch_size}"
        )

    real = torch.as_tensor([domain == REAL_DOMAIN for domain in domains], dtype=torch.bool, device=device)
    counterfactual = ~real
    if bool((real & (annotation_valid | is_hazard)).any().item()):
        raise ValueError("real samples cannot carry counterfactual annotations or hazards")
    if bool((counterfactual & ~annotation_valid).any().item()):
        raise ValueError("counterfactual samples require valid annotations under cf_supervision_v2")

    unknown_hazards = sorted(
        {
            hazard_type
            for hazard, hazard_type in zip(is_hazard.tolist(), hazard_types)
            if hazard and hazard_type not in KNOWN_HAZARD_TYPES
        }
    )
    if unknown_hazards:
        raise ValueError(f"metadata['cf_hazard_type'] contains unknown hazard types: {unknown_hazards}")
    invalid_non_hazard_types = [
        idx
        for idx, (hazard, hazard_type) in enumerate(zip(is_hazard.tolist(), hazard_types))
        if not hazard and hazard_type
    ]
    if invalid_non_hazard_types:
        raise ValueError(f"non-hazard samples must have empty cf_hazard_type: indices={invalid_non_hazard_types}")

    cf_hazard = counterfactual & annotation_valid & is_hazard
    cf_safe = counterfactual & annotation_valid & ~is_hazard
    cf_ego_hazard = cf_hazard & torch.as_tensor(
        [value in EGO_HAZARD_TYPES for value in hazard_types], dtype=torch.bool, device=device
    )
    cf_other_accident = cf_hazard & torch.as_tensor(
        [value in OTHER_ACCIDENT_TYPES for value in hazard_types], dtype=torch.bool, device=device
    )
    imitation = real | cf_safe
    all_valid = real | (counterfactual & annotation_valid)
    return CounterfactualSampleMasks(
        real=real,
        cf_safe=cf_safe,
        cf_hazard=cf_hazard,
        cf_ego_hazard=cf_ego_hazard,
        cf_other_accident=cf_other_accident,
        imitation=imitation,
        world_model=all_valid,
        value=all_valid,
    )


def distributed_mask_normalization(
    mask: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype = torch.float32,
) -> DistributedMaskNormalization:
    """Return the DDP scale that turns local means into a global masked mean."""

    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.ndim != 1:
        raise ValueError(
            f"{name} mask must be a bool tensor with shape [B], "
            f"got {type(mask).__name__}, dtype={getattr(mask, 'dtype', None)}, "
            f"shape={getattr(mask, 'shape', None)}"
        )
    local_count = mask.sum().to(dtype=dtype)
    global_count = local_count.detach().clone()
    world_size = 1
    dist = getattr(torch, "distributed", None)
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = int(dist.get_world_size())
        if world_size > 1:
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if float(global_count.item()) <= 0.0:
        raise ValueError(f"{name} has zero global eligible samples")
    mean_scale = local_count * (float(world_size) / global_count)
    return DistributedMaskNormalization(
        local_count=local_count,
        global_count=global_count,
        mean_scale=mean_scale,
    )


def distributed_masked_mean(
    per_sample_loss: torch.Tensor,
    mask: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Globally normalize masked per-sample losses while preserving an empty local graph."""

    if per_sample_loss.ndim != 1:
        raise ValueError(f"{name} per_sample_loss must have shape [B], got {tuple(per_sample_loss.shape)}")
    if mask.ndim != 1 or int(mask.shape[0]) != int(per_sample_loss.shape[0]):
        raise ValueError(
            f"{name} mask shape {tuple(mask.shape)} does not match per_sample_loss "
            f"shape {tuple(per_sample_loss.shape)}"
        )
    mask = mask.to(device=per_sample_loss.device, dtype=torch.bool)
    normalization = distributed_mask_normalization(mask, name=name, dtype=per_sample_loss.dtype)
    local_sum = (per_sample_loss * mask.to(dtype=per_sample_loss.dtype)).sum()
    return local_sum * (normalization.mean_scale / normalization.local_count.clamp_min(1.0))
