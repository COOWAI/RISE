"""Strict NavSim mixed-batch adapter for CVoI Field supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence

import torch

from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import CVOI_CF_FIELD_WEIGHTS
from app.vjepa_cowa_world_model.training.counterfactual_supervision import build_counterfactual_sample_masks
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
from app.vjepa_cowa_world_model.training.navsim_data import (
    CF_QUALITY_SCHEMA,
    NAVSIM_DEFAULT_MAX_AGENTS,
    REAL_AGENT_COORDINATE_FRAME,
    REAL_AGENT_GEOMETRY_SOURCE,
)

NAVSIM_CVOI_FIELD_ADAPTER_SCHEMA = "navsim_cvoi_field_batch_v1"
NAVSIM_E120_QUALITY_FIELD_ADAPTER_SCHEMA = "navsim_e120_quality_field_batch_v1"


@dataclass(frozen=True)
class RealGeometryTargetRequest:
    """Real-only inputs supplied to an external Planner/evaluator callback.

    The adapter deliberately does not construct geometry utilities. The
    callback must run the planner candidates and the independent real-geometry
    evaluator, then return one score per real sample and future prefix.
    """

    batch_indices: torch.Tensor
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    z_observed: torch.Tensor
    z_future: torch.Tensor
    states: torch.Tensor
    agent_boxes: torch.Tensor
    agent_mask: torch.Tensor
    raw_agent_count: torch.Tensor
    geometry_source: tuple[str, ...]
    geometry_coordinate_frame: tuple[str, ...]


RealGeometryTargetProvider = Callable[[RealGeometryTargetRequest], torch.Tensor]


@dataclass(frozen=True)
class RealQualityTargetRequest:
    """Identity and latent inputs for route-free real trajectory quality."""

    batch_indices: torch.Tensor
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    z_observed: torch.Tensor
    z_future: torch.Tensor


RealQualityTargetProvider = Callable[[RealQualityTargetRequest], torch.Tensor]


def _require_batch_tensor(value: object, *, name: str, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape[0] != batch_size:
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
        raise ValueError(f"NavSim CVoI {name} must be a tensor with batch size {batch_size}, got {shape}")
    return value


def _require_metadata_bool(metadata: Mapping[str, object], name: str, *, batch_size: int) -> torch.Tensor:
    value = metadata.get(name)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or tuple(value.shape) != (batch_size,):
        raise ValueError(f"NavSim CVoI metadata[{name!r}] must be bool [{batch_size}]")
    return value.cpu()


def _require_metadata_strings(
    metadata: Mapping[str, object],
    name: str,
    *,
    batch_size: int,
) -> tuple[str, ...]:
    value = metadata.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != batch_size:
        raise ValueError(f"NavSim CVoI metadata[{name!r}] must contain {batch_size} entries")
    strings = tuple(str(entry) for entry in value)
    if any(not entry for entry in strings):
        raise ValueError(f"NavSim CVoI metadata[{name!r}] must contain non-empty strings")
    return strings


def require_navsim_e120_metadata_strings(
    metadata: Mapping[str, object],
    name: str,
    *,
    batch_size: int,
) -> tuple[str, ...]:
    """Require native string metadata without legacy string coercion."""

    value = metadata.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != batch_size:
        raise ValueError(f"NavSim-e120 metadata[{name!r}] must contain {batch_size} non-empty strings")
    if any(not isinstance(entry, str) or not entry for entry in value):
        raise ValueError(f"NavSim-e120 metadata[{name!r}] must contain {batch_size} non-empty strings")
    return tuple(value)


def _require_navsim_e120_window_start_positions(
    metadata: Mapping[str, object],
    *,
    batch_size: int,
) -> tuple[int, ...]:
    value = metadata.get("window_start_pos")
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.shape[0] != batch_size:
        raise ValueError(f"NavSim-e120 metadata['window_start_pos'] must be integer [{batch_size}]")
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"NavSim-e120 metadata['window_start_pos'] must be integer [{batch_size}]")
    positions = tuple(int(entry) for entry in value.cpu().tolist())
    if any(entry < 0 for entry in positions):
        raise ValueError("NavSim-e120 metadata['window_start_pos'] must be non-negative")
    return positions


def _build_navsim_e120_hazard_pairs(
    metadata: Mapping[str, object],
    *,
    domains: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, int]]]:
    batch_size = len(domains)
    base_scene_ids = require_navsim_e120_metadata_strings(
        metadata,
        "base_scene_id",
        batch_size=batch_size,
    )
    window_start_positions = _require_navsim_e120_window_start_positions(metadata, batch_size=batch_size)
    hazards = _require_metadata_bool(metadata, "cf_is_hazard", batch_size=batch_size)
    raw_hazard_types = metadata.get("cf_hazard_type")
    if (
        isinstance(raw_hazard_types, (str, bytes))
        or not isinstance(raw_hazard_types, Sequence)
        or len(raw_hazard_types) != batch_size
        or any(not isinstance(entry, str) for entry in raw_hazard_types)
    ):
        raise ValueError(f"NavSim-e120 metadata['cf_hazard_type'] must contain {batch_size} strings")
    hazard_types = tuple(raw_hazard_types)

    factual_by_key: dict[tuple[str, int], int] = {}
    for index, domain in enumerate(domains):
        if domain != "real":
            continue
        key = (base_scene_ids[index], window_start_positions[index])
        if key in factual_by_key:
            raise ValueError(f"duplicate matched real factual sample for CVoI pair key {key!r}")
        factual_by_key[key] = index

    real_indices = []
    counterfactual_indices = []
    pair_keys = []
    seen_counterfactual: set[tuple[str, int]] = set()
    for index, domain in enumerate(domains):
        if domain != "counterfactual":
            continue
        key = (base_scene_ids[index], window_start_positions[index])
        if key in seen_counterfactual:
            raise ValueError(f"duplicate counterfactual CVoI hazard pair key {key!r}")
        seen_counterfactual.add(key)
        if hazards[index].item() is not True or hazard_types[index] not in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST:
            raise ValueError(
                "NavSim-e120 counterfactual rows must remain hazards in the exact accident_type allowlist; "
                f"index={index}, hazard={bool(hazards[index])}, type={hazard_types[index]!r}"
            )
        real_index = factual_by_key.get(key)
        if real_index is None:
            raise ValueError(f"missing matched real factual sample for CVoI pair key {key!r}")
        real_indices.append(real_index)
        counterfactual_indices.append(index)
        pair_keys.append(key)
    if counterfactual_indices and len(real_indices) != len(counterfactual_indices):
        raise RuntimeError("NavSim-e120 internal hazard pairing count mismatch")
    return (
        torch.as_tensor(real_indices, dtype=torch.long),
        torch.as_tensor(counterfactual_indices, dtype=torch.long),
        pair_keys,
    )


def _future_prefix_count(z_future: torch.Tensor, *, tokens_per_frame: Optional[int]) -> int:
    if z_future.ndim == 4:
        if z_future.shape[2] < 1:
            raise ValueError("z_future must contain at least one spatial token per frame")
        count = int(z_future.shape[1])
    elif z_future.ndim == 3:
        if type(tokens_per_frame) is not int or tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame is required and positive for flat z_future")
        if int(z_future.shape[1]) % tokens_per_frame:
            raise ValueError("tokens_per_frame must divide the flat z_future token length")
        count = int(z_future.shape[1]) // tokens_per_frame
    else:
        raise ValueError(f"z_future must be [B, N, D] or [B, F, T, D], got {tuple(z_future.shape)}")
    if count < 1:
        raise ValueError("z_future must contain at least one future prefix")
    return count


def _validate_latents(
    z_observed: object,
    z_future: object,
    *,
    batch_size: int,
    tokens_per_frame: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, int]:
    observed = _require_batch_tensor(z_observed, name="z_observed", batch_size=batch_size)
    future = _require_batch_tensor(z_future, name="z_future", batch_size=batch_size)
    for name, value in (("z_observed", observed), ("z_future", future)):
        if value.ndim not in (3, 4) or not value.is_floating_point():
            raise ValueError(f"NavSim CVoI {name} must be floating [B, N, D] or [B, F, T, D]")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"NavSim CVoI {name} contains NaN/Inf")
    if observed.shape[-1] != future.shape[-1]:
        raise ValueError("z_observed and z_future embedding dimensions must match")
    if observed.ndim == 3:
        if type(tokens_per_frame) is not int or tokens_per_frame <= 0:
            raise ValueError("z_observed requires a positive tokens_per_frame for flat tokens")
        if int(observed.shape[1]) % tokens_per_frame:
            raise ValueError("z_observed token length must be divisible by tokens_per_frame")
        observed_frame_count = int(observed.shape[1]) // tokens_per_frame
    else:
        if observed.shape[2] < 1:
            raise ValueError("z_observed must contain at least one spatial token per frame")
        observed_frame_count = int(observed.shape[1])
    if observed_frame_count < 1:
        raise ValueError("z_observed must contain at least one observed frame")
    return observed, future, _future_prefix_count(future, tokens_per_frame=tokens_per_frame)


def _validate_geometry_contract(
    metadata: Mapping[str, object],
    *,
    real_mask: torch.Tensor,
    agent_boxes: torch.Tensor,
    agent_mask: torch.Tensor,
) -> None:
    batch_size = int(real_mask.shape[0])
    geometry_present = _require_metadata_bool(metadata, "geometry_present", batch_size=batch_size)
    future_valid = _require_metadata_bool(metadata, "future_agent_geometry_valid", batch_size=batch_size)
    if not torch.equal(geometry_present, real_mask):
        raise ValueError("geometry_present must agree exactly with the real/counterfactual domain mask")
    if not torch.equal(future_valid, real_mask):
        raise ValueError("future_agent_geometry_valid must be true exactly for real CVoI Field rows")

    if agent_boxes.ndim != 4 or agent_boxes.shape[-2:] != (NAVSIM_DEFAULT_MAX_AGENTS, 7):
        raise ValueError(f"NavSim CVoI agent_boxes must be [B, T, {NAVSIM_DEFAULT_MAX_AGENTS}, 7]")
    if not agent_boxes.is_floating_point():
        raise ValueError("NavSim CVoI agent_boxes must be floating point")
    if agent_mask.dtype != torch.bool or tuple(agent_mask.shape) != tuple(agent_boxes.shape[:-1]):
        raise ValueError("NavSim CVoI agent_mask must be bool [B, T, 256] aligned with agent_boxes")
    if agent_mask.device != agent_boxes.device:
        raise ValueError("NavSim CVoI agent_boxes and agent_mask must be on the same device")

    device_real_mask = real_mask.to(device=agent_boxes.device)
    cf_mask = ~device_real_mask
    if bool(agent_mask[cf_mask].any().item()) or bool((agent_boxes[cf_mask] != 0).any().item()):
        raise ValueError("counterfactual zero geometry transport tensors must not carry geometry")
    selected_boxes = agent_boxes[device_real_mask][agent_mask[device_real_mask]]
    if selected_boxes.numel() and (
        not bool(torch.isfinite(selected_boxes).all().item()) or bool((selected_boxes[:, 3:6] <= 0.0).any().item())
    ):
        raise ValueError("real agent boxes must contain finite positive dimensions")

    truncated = metadata.get("agent_geometry_truncated")
    sources = metadata.get("geometry_source")
    frames = metadata.get("geometry_coordinate_frame")
    raw_agent_counts = metadata.get("raw_agent_count")
    for name, value in (
        ("agent_geometry_truncated", truncated),
        ("geometry_source", sources),
        ("geometry_coordinate_frame", frames),
        ("raw_agent_count", raw_agent_counts),
    ):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != batch_size:
            raise ValueError(f"NavSim CVoI metadata[{name!r}] must contain {batch_size} entries")
    for index, is_real in enumerate(real_mask.tolist()):
        if is_real:
            if truncated[index] is not False:
                raise ValueError(f"real sample {index} must have agent_geometry_truncated=false")
            if sources[index] != REAL_AGENT_GEOMETRY_SOURCE or frames[index] != REAL_AGENT_COORDINATE_FRAME:
                raise ValueError(f"real sample {index} has invalid geometry provenance")
            raw_count = raw_agent_counts[index]
            if (
                not isinstance(raw_count, torch.Tensor)
                or raw_count.ndim != 1
                or raw_count.shape[0] != agent_mask.shape[1]
                or raw_count.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            ):
                raise ValueError(f"real sample {index} raw_agent_count must be integer [T]")
            mask_count = agent_mask[index].sum(dim=-1).to(device="cpu", dtype=torch.long)
            if not torch.equal(raw_count.to(device="cpu", dtype=torch.long), mask_count):
                raise ValueError(f"real sample {index} raw_agent_count must match agent_mask counts")
        elif (
            truncated[index] is not None
            or sources[index] is not None
            or frames[index] is not None
            or raw_agent_counts[index] is not None
        ):
            raise ValueError(f"counterfactual sample {index} cannot claim geometry provenance or agent counts")


def _validate_cf_quality(
    metadata: Mapping[str, object],
    *,
    counterfactual_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(counterfactual_mask.shape[0])
    present = _require_metadata_bool(metadata, "cf_quality_present", batch_size=batch_size)
    quality = metadata.get("cf_quality")
    if (
        not isinstance(quality, torch.Tensor)
        or not quality.is_floating_point()
        or tuple(quality.shape) != (batch_size,)
    ):
        raise ValueError(f"NavSim CVoI metadata['cf_quality'] must be floating [{batch_size}]")
    quality = quality.cpu()
    if not torch.equal(present, counterfactual_mask):
        missing = torch.nonzero(counterfactual_mask & ~present, as_tuple=False).flatten().tolist()
        raise ValueError(f"counterfactual rows require trajectory quality sidecar labels; missing indices={missing}")
    selected = quality[counterfactual_mask]
    if not bool(torch.isfinite(selected).all().item()) or bool(((selected < 0.0) | (selected > 1.0)).any().item()):
        raise ValueError("counterfactual trajectory quality must be finite and in [0, 1]")
    if bool(torch.isfinite(quality[~counterfactual_mask]).any().item()):
        raise ValueError("real rows cannot carry counterfactual trajectory quality")
    schemas = metadata.get("cf_quality_schema")
    sources = metadata.get("cf_quality_source")
    for name, values in (("cf_quality_schema", schemas), ("cf_quality_source", sources)):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != batch_size:
            raise ValueError(f"NavSim CVoI metadata[{name!r}] must contain {batch_size} entries")
    for index, is_counterfactual in enumerate(counterfactual_mask.tolist()):
        if is_counterfactual:
            if schemas[index] != CF_QUALITY_SCHEMA:
                raise ValueError(f"counterfactual trajectory quality schema must be {CF_QUALITY_SCHEMA}")
            if sources[index] != "trajectory_quality_sidecar":
                raise ValueError("counterfactual trajectory quality source must be trajectory_quality_sidecar")
        elif schemas[index] is not None or sources[index] is not None:
            raise ValueError(f"real sample {index} cannot carry counterfactual trajectory quality provenance")
    return quality


def _validate_real_targets(
    targets: object,
    *,
    real_count: int,
    num_prefixes: int,
    provider_name: str,
) -> torch.Tensor:
    if not isinstance(targets, torch.Tensor) or tuple(targets.shape) != (real_count, num_prefixes):
        shape = tuple(targets.shape) if isinstance(targets, torch.Tensor) else type(targets).__name__
        raise ValueError(f"{provider_name} must return [{real_count}, {num_prefixes}], got {shape}")
    if not targets.is_floating_point() or not bool(torch.isfinite(targets).all().item()):
        raise ValueError(f"{provider_name} must return finite floating scores")
    if bool(((targets < 0.0) | (targets > 1.0)).any().item()):
        raise ValueError(f"{provider_name} scores must be in [0, 1]")
    return targets.detach().clone()


def adapt_navsim_cvoi_field_batch(
    navsim_batch: Sequence[object],
    *,
    z_observed: torch.Tensor,
    z_future: torch.Tensor,
    real_geometry_target_provider: Optional[RealGeometryTargetProvider] = None,
    tokens_per_frame: Optional[int] = None,
    cf_field_supervision: str = "hazard_quality",
) -> Dict[str, object]:
    """Convert one standard NavSim batch into strict CVoI Field inputs.

    No geometry score is synthesized here. Whenever the batch contains real
    rows, ``real_geometry_target_provider`` is mandatory and receives only
    those rows. Counterfactual rows receive hazard/quality ranking labels and
    their transport-only zero boxes are validated but never passed to the
    callback.
    """

    if isinstance(navsim_batch, (str, bytes)) or not isinstance(navsim_batch, Sequence) or len(navsim_batch) != 12:
        raise ValueError("NavSim CVoI adapter requires the standard 12-element NavSim batch tuple")
    metadata = navsim_batch[11]
    if not isinstance(metadata, Mapping):
        raise ValueError("NavSim CVoI batch element 11 must be metadata mapping")
    domains_value = metadata.get("dataset_domain")
    if isinstance(domains_value, (str, bytes)) or not isinstance(domains_value, Sequence):
        raise ValueError("NavSim CVoI metadata['dataset_domain'] must be a sequence")
    domains = [str(domain) for domain in domains_value]
    batch_size = len(domains)
    if batch_size < 1:
        raise ValueError("NavSim CVoI batch cannot be empty")
    if cf_field_supervision not in CVOI_CF_FIELD_WEIGHTS:
        raise ValueError(f"unknown cf_field_supervision {cf_field_supervision!r}")
    hazard_weight, quality_weight = CVOI_CF_FIELD_WEIGHTS[cf_field_supervision]

    observed, future, num_prefixes = _validate_latents(
        z_observed,
        z_future,
        batch_size=batch_size,
        tokens_per_frame=tokens_per_frame,
    )
    real_mask = torch.as_tensor([domain == "real" for domain in domains], dtype=torch.bool)
    if any(domain not in {"real", "counterfactual"} for domain in domains):
        raise ValueError(f"NavSim CVoI metadata contains unknown dataset domains: {domains}")
    if hazard_weight > 0.0:
        hazard_metadata = {
            "dataset_domain": domains,
            "cf_annotation_valid": metadata["cf_annotation_valid"],
            "cf_is_hazard": metadata["cf_is_hazard"],
            "cf_hazard_type": metadata["cf_hazard_type"],
        }
        masks = build_counterfactual_sample_masks(hazard_metadata, device=torch.device("cpu"))
        if not torch.equal(masks.real.cpu(), real_mask):
            raise ValueError("NavSim CVoI hazard metadata domain mask mismatch")
    counterfactual_mask = ~real_mask

    states = _require_batch_tensor(navsim_batch[2], name="states", batch_size=batch_size)
    agent_boxes = _require_batch_tensor(navsim_batch[7], name="agent_boxes", batch_size=batch_size)
    agent_mask = _require_batch_tensor(navsim_batch[8], name="agent_mask", batch_size=batch_size)
    if (
        states.ndim != 3
        or states.shape[-1] != 7
        or not states.is_floating_point()
        or not bool(torch.isfinite(states).all().item())
    ):
        raise ValueError("NavSim CVoI states must be finite floating [B, T, 7]")
    if agent_boxes.ndim == 4 and states.shape[1] != agent_boxes.shape[1]:
        raise ValueError("NavSim CVoI states and agent_boxes must share the same time dimension")
    _validate_geometry_contract(
        metadata,
        real_mask=real_mask,
        agent_boxes=agent_boxes,
        agent_mask=agent_mask,
    )
    quality = _validate_cf_quality(metadata, counterfactual_mask=counterfactual_mask) if quality_weight > 0.0 else None
    sample_ids = _require_metadata_strings(metadata, "stable_sample_id", batch_size=batch_size)

    output: Dict[str, object] = {
        "adapter_schema": NAVSIM_CVOI_FIELD_ADAPTER_SCHEMA,
        "z_observed": observed,
        "z_future": future,
        "dataset_domains": domains,
        "real_mask": real_mask,
        "counterfactual_mask": counterfactual_mask,
        "stable_sample_ids": list(sample_ids),
    }
    if bool(counterfactual_mask.any().item()) and hazard_weight > 0.0:
        output["cf_hazard"] = metadata["cf_is_hazard"].cpu()
        output["cf_hazard_types"] = [str(value) for value in metadata["cf_hazard_type"]]
    if bool(counterfactual_mask.any().item()) and quality_weight > 0.0:
        output["cf_quality"] = quality

    real_count = int(real_mask.sum().item())
    if real_count:
        if real_geometry_target_provider is None or not callable(real_geometry_target_provider):
            raise ValueError(
                "real_geometry_target_provider is required for real rows; "
                "the adapter never fabricates real_geometry_targets"
            )
        group_ids = _require_metadata_strings(metadata, "base_scene_id", batch_size=batch_size)
        indices = torch.nonzero(real_mask, as_tuple=False).flatten()
        request = RealGeometryTargetRequest(
            batch_indices=indices,
            sample_ids=tuple(sample_ids[index] for index in indices.tolist()),
            group_ids=tuple(group_ids[index] for index in indices.tolist()),
            z_observed=observed[real_mask.to(device=observed.device)].detach(),
            z_future=future[real_mask.to(device=future.device)].detach(),
            states=states[real_mask.to(device=states.device)].detach(),
            agent_boxes=agent_boxes[real_mask.to(device=agent_boxes.device)].detach(),
            agent_mask=agent_mask[real_mask.to(device=agent_mask.device)].detach(),
            raw_agent_count=torch.stack([metadata["raw_agent_count"][index] for index in indices.tolist()]),
            geometry_source=tuple(str(metadata["geometry_source"][index]) for index in indices.tolist()),
            geometry_coordinate_frame=tuple(
                str(metadata["geometry_coordinate_frame"][index]) for index in indices.tolist()
            ),
        )
        with torch.no_grad():
            raw_targets = real_geometry_target_provider(request)
        output["real_geometry_targets"] = _validate_real_targets(
            raw_targets,
            real_count=real_count,
            num_prefixes=num_prefixes,
            provider_name="real geometry target provider",
        )
        output["real_group_ids"] = list(request.group_ids)
    return output


def adapt_navsim_e120_quality_field_batch(
    navsim_batch: Sequence[object],
    *,
    z_observed: torch.Tensor,
    z_future: torch.Tensor,
    real_quality_target_provider: Optional[RealQualityTargetProvider] = None,
    tokens_per_frame: Optional[int] = None,
    cf_field_supervision: str = "hazard_quality",
) -> Dict[str, object]:
    """Build the NavSim-e120 Field batch without reading future geometry.

    Counterfactual hazard labels come only from the normalized annotation
    contract and counterfactual quality comes only from its signed sidecar.
    Real rows are scored by the supplied route-free planner-output callback.
    """

    if isinstance(navsim_batch, (str, bytes)) or not isinstance(navsim_batch, Sequence) or len(navsim_batch) != 12:
        raise ValueError("NavSim-e120 quality adapter requires the standard 12-element batch tuple")
    metadata = navsim_batch[11]
    if not isinstance(metadata, Mapping):
        raise ValueError("NavSim-e120 quality batch element 11 must be metadata mapping")
    domains_value = metadata.get("dataset_domain")
    if isinstance(domains_value, (str, bytes)) or not isinstance(domains_value, Sequence):
        raise ValueError("NavSim-e120 quality metadata['dataset_domain'] must be a sequence")
    if len(domains_value) == 0:
        raise ValueError("NavSim-e120 quality batch cannot be empty")
    domains = list(
        require_navsim_e120_metadata_strings(
            metadata,
            "dataset_domain",
            batch_size=len(domains_value),
        )
    )
    if any(domain not in {"real", "counterfactual"} for domain in domains):
        raise ValueError(f"NavSim-e120 quality batch contains unknown domains: {domains}")
    if cf_field_supervision not in CVOI_CF_FIELD_WEIGHTS:
        raise ValueError(f"unknown cf_field_supervision {cf_field_supervision!r}")
    hazard_weight, quality_weight = CVOI_CF_FIELD_WEIGHTS[cf_field_supervision]
    observed, future, num_prefixes = _validate_latents(
        z_observed,
        z_future,
        batch_size=len(domains),
        tokens_per_frame=tokens_per_frame,
    )
    real_mask = torch.as_tensor([domain == "real" for domain in domains], dtype=torch.bool)
    counterfactual_mask = ~real_mask
    if hazard_weight > 0.0:
        hazard_metadata = {
            "dataset_domain": domains,
            "cf_annotation_valid": metadata["cf_annotation_valid"],
            "cf_is_hazard": metadata["cf_is_hazard"],
            "cf_hazard_type": metadata["cf_hazard_type"],
        }
        masks = build_counterfactual_sample_masks(hazard_metadata, device=torch.device("cpu"))
        if not torch.equal(masks.real.cpu(), real_mask):
            raise ValueError("NavSim-e120 quality annotation domain mask mismatch")
    quality = _validate_cf_quality(metadata, counterfactual_mask=counterfactual_mask) if quality_weight > 0.0 else None
    sample_ids = require_navsim_e120_metadata_strings(metadata, "stable_sample_id", batch_size=len(domains))
    group_ids = require_navsim_e120_metadata_strings(metadata, "base_scene_id", batch_size=len(domains))
    output: Dict[str, object] = {
        "adapter_schema": NAVSIM_E120_QUALITY_FIELD_ADAPTER_SCHEMA,
        "z_observed": observed,
        "z_future": future,
        "dataset_domains": domains,
        "real_mask": real_mask,
        "counterfactual_mask": counterfactual_mask,
        "stable_sample_ids": list(sample_ids),
    }
    if bool(counterfactual_mask.any().item()) and hazard_weight > 0.0:
        output["cf_hazard"] = metadata["cf_is_hazard"].cpu()
        output["cf_hazard_types"] = [str(value) for value in metadata["cf_hazard_type"]]
        real_pair_indices, counterfactual_pair_indices, pair_keys = _build_navsim_e120_hazard_pairs(
            metadata,
            domains=domains,
        )
        output["cf_hazard_pair_real_indices"] = real_pair_indices
        output["cf_hazard_pair_counterfactual_indices"] = counterfactual_pair_indices
        output["cf_hazard_pair_keys"] = pair_keys
    if bool(counterfactual_mask.any().item()) and quality_weight > 0.0:
        output["cf_quality"] = quality
    real_count = int(real_mask.sum().item())
    if real_count:
        if real_quality_target_provider is None or not callable(real_quality_target_provider):
            raise ValueError("real_quality_target_provider is required for real NavSim-e120 rows")
        indices = torch.nonzero(real_mask, as_tuple=False).flatten()
        request = RealQualityTargetRequest(
            batch_indices=indices,
            sample_ids=tuple(sample_ids[index] for index in indices.tolist()),
            group_ids=tuple(group_ids[index] for index in indices.tolist()),
            z_observed=observed[real_mask.to(device=observed.device)].detach(),
            z_future=future[real_mask.to(device=future.device)].detach(),
        )
        with torch.no_grad():
            raw_targets = real_quality_target_provider(request)
        output["real_quality_targets"] = _validate_real_targets(
            raw_targets,
            real_count=real_count,
            num_prefixes=num_prefixes,
            provider_name="real quality target provider",
        )
        output["real_group_ids"] = list(request.group_ids)
    return output
