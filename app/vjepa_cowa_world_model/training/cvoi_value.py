"""Strict domain routing for CVoI dual-value supervision and metadata."""

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training.counterfactual_supervision import KNOWN_HAZARD_TYPES
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
from app.vjepa_cowa_world_model.training.cvoi_manual_lineage import (
    build_cvoi_manual_value_parents,
    resolve_cvoi_manual_value_lineage_by_checkpoint_branch,
)

CVOI_NAVSIM_E120_VALUE_CHECKPOINT_SCHEMA = "cvoi_dual_value_navsim_e120_v1"
CVOI_VALUE_PROTOCOL_FORMAL_V2_NAVSIM_E120 = "formal_v2_navsim_e120_h4_v3"
CVOI_FIELD_WARMUP_DATA_SCHEMA = "cvoi_field_warmup_data_v1"
CVOI_VALUE_CHECKPOINT_PHASES = (
    "field_warmup",
    "field_calibrated",
    "stop_calibrated",
)
_VALID_DOMAINS = {"real", "counterfactual"}
_PREFIX_DUAL_VALUE_ARCHITECTURE_FIELDS = frozenset(
    {
        "embed_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
    }
)
_NAVSIM_E120_DIRECT_VALUE_CHECKPOINT_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "protocol_version",
        "branch_id",
        "epoch",
        "architecture",
        "roles",
        "parents",
        "state_dict",
    }
)
_NAVSIM_E120_DIRECT_VALUE_METADATA_FIELDS = frozenset(
    {
        "schema",
        "phase",
        "protocol_version",
        "branch_id",
        "epoch",
        "roles",
        "parents",
    }
)
_NAVSIM_E120_DIRECT_VALUE_ROLE_FIELDS = frozenset({"keys", "shapes"})
_NAVSIM_E120_DIRECT_VALUE_PHASES = frozenset(CVOI_VALUE_CHECKPOINT_PHASES)
_NAVSIM_E120_DIRECT_VALUE_FORBIDDEN_KEY_MARKERS = (
    "receipt",
    "audit",
    "provenance",
    "source_commit",
)


@dataclass(frozen=True)
class CVoIValueLossOutput:
    """One differentiable loss with detached scalar diagnostics."""

    loss: torch.Tensor
    diagnostics: Dict[str, float]


def validate_field_warmup_data_provenance(value: object) -> Dict[str, object]:
    """Validate the exact dataset-domain identity of a Field warm-up artifact."""

    if not isinstance(value, Mapping):
        raise ValueError("field_warmup_data_provenance must be a mapping")
    expected_fields = {"schema", "domain", "roots", "real_sample_count", "cf_sample_count"}
    if set(value) != expected_fields:
        raise ValueError(
            "field_warmup_data_provenance fields mismatch: "
            f"missing={sorted(expected_fields - set(value))}, unexpected={sorted(set(value) - expected_fields)}"
        )
    if value["schema"] != CVOI_FIELD_WARMUP_DATA_SCHEMA:
        raise ValueError(f"field_warmup_data_provenance.schema must be {CVOI_FIELD_WARMUP_DATA_SCHEMA!r}")
    domain = value["domain"]
    if domain not in {"real", "real_cf"}:
        raise ValueError("field_warmup_data_provenance.domain must be 'real' or 'real_cf'")
    raw_roots = value["roots"]
    if isinstance(raw_roots, (str, bytes)) or not isinstance(raw_roots, Sequence) or not raw_roots:
        raise ValueError("field_warmup_data_provenance.roots must be a non-empty sequence")
    roots = []
    for index, root in enumerate(raw_roots):
        if not isinstance(root, Mapping) or set(root) != {"name", "domain"}:
            raise ValueError(f"field_warmup_data_provenance.roots[{index}] must contain only name and domain")
        name = root["name"]
        root_domain = root["domain"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"field_warmup_data_provenance.roots[{index}].name must be non-empty")
        if root_domain not in {"real", "counterfactual"}:
            raise ValueError(f"field_warmup_data_provenance.roots[{index}].domain is invalid")
        roots.append({"name": name, "domain": root_domain})
    root_names = [root["name"] for root in roots]
    if len(set(root_names)) != len(root_names):
        raise ValueError("field_warmup_data_provenance root names must be unique")
    counts = {}
    for name in ("real_sample_count", "cf_sample_count"):
        count = value[name]
        if type(count) is not int or count < 0:
            raise ValueError(f"field_warmup_data_provenance.{name} must be a non-negative integer")
        counts[name] = count
    root_domains = {root["domain"] for root in roots}
    if domain == "real":
        if "counterfactual" in root_domains:
            raise ValueError("Real-only Field warm-up provenance cannot contain a counterfactual root")
        if root_domains != {"real"}:
            raise ValueError("Real-only Field warm-up provenance requires at least one real root")
        if counts["cf_sample_count"] != 0:
            raise ValueError("Real-only Field warm-up provenance requires cf_sample_count=0")
        if counts["real_sample_count"] <= 0:
            raise ValueError("Real-only Field warm-up provenance requires positive real_sample_count")
    else:
        if root_domains != {"real", "counterfactual"}:
            raise ValueError("Real+CF Field warm-up provenance requires both real and counterfactual roots")
        if counts["real_sample_count"] <= 0:
            raise ValueError("Real+CF Field warm-up provenance requires positive real_sample_count")
        if counts["cf_sample_count"] <= 0:
            raise ValueError("Real+CF Field warm-up provenance requires positive cf_sample_count")
    return {
        "schema": CVOI_FIELD_WARMUP_DATA_SCHEMA,
        "domain": domain,
        "roots": roots,
        **counts,
    }


def _validate_domains(dataset_domains: Sequence[str], *, batch_size: int) -> Tuple[str, ...]:
    if isinstance(dataset_domains, (str, bytes)) or not isinstance(dataset_domains, Sequence):
        raise ValueError("dataset_domains must be a sequence with one domain per sample")
    domains = tuple(str(domain) for domain in dataset_domains)
    if len(domains) != batch_size:
        raise ValueError(f"dataset_domains length {len(domains)} does not match batch size {batch_size}")
    unknown = sorted(set(domains) - _VALID_DOMAINS)
    if unknown:
        raise ValueError(f"dataset_domains contains unknown domains: {unknown}")
    return domains


def _validate_nonnegative_finite(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    return value


def _zero_loss(values: torch.Tensor) -> torch.Tensor:
    return values.sum() * 0.0


def _local_order_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    *,
    group_ids: Sequence[str],
    margin: float,
) -> Tuple[torch.Tensor, int]:
    if len(group_ids) != predicted.shape[0] or any(not str(group_id) for group_id in group_ids):
        raise ValueError("real_group_ids must contain one non-empty group id per selected real sample")
    same_group = torch.as_tensor(
        [[left == right for right in group_ids] for left in group_ids],
        dtype=torch.bool,
        device=predicted.device,
    )
    target_difference = targets.unsqueeze(1) - targets.unsqueeze(0)
    ordered = (target_difference > 0) & same_group.unsqueeze(-1)
    pair_count = int(ordered.sum().item())
    if pair_count == 0:
        return _zero_loss(predicted), 0
    predicted_difference = predicted.unsqueeze(1) - predicted.unsqueeze(0)
    losses = torch.relu(float(margin) - predicted_difference)
    return losses[ordered].mean(), pair_count


def _hazard_ranking_loss(
    values: torch.Tensor,
    hazard: torch.Tensor,
    hazard_types: Sequence[str],
    *,
    margin: float,
) -> Tuple[torch.Tensor, int, int]:
    safe_values = values[~hazard]
    unique_types = sorted({hazard_type for is_hazard, hazard_type in zip(hazard.tolist(), hazard_types) if is_hazard})
    losses = []
    pair_count = 0
    for hazard_type in unique_types:
        type_mask = torch.as_tensor(
            [
                is_hazard and current_type == hazard_type
                for is_hazard, current_type in zip(hazard.tolist(), hazard_types)
            ],
            dtype=torch.bool,
            device=values.device,
        )
        hazard_values = values[type_mask]
        type_pair_count = int(safe_values.shape[0] * hazard_values.shape[0])
        if type_pair_count:
            differences = safe_values[:, None, :] - hazard_values[None, :, :]
            losses.append(torch.relu(float(margin) - differences).mean())
            pair_count += type_pair_count
    if not losses:
        return _zero_loss(values), 0, len(unique_types)
    return torch.stack(losses).mean(), pair_count, len(unique_types)


def _matched_real_counterfactual_hazard_ranking_loss(
    values: torch.Tensor,
    domains: Sequence[str],
    hazard: object,
    hazard_types: object,
    real_indices: object,
    counterfactual_indices: object,
    *,
    margin: float,
) -> Tuple[torch.Tensor, int, int]:
    batch_size = int(values.shape[0])
    if not isinstance(hazard, torch.Tensor) or hazard.dtype != torch.bool or tuple(hazard.shape) != (batch_size,):
        raise ValueError(f"cf_hazard must be bool [{batch_size}] for matched NavSim-e120 ranking")
    if (
        isinstance(hazard_types, (str, bytes))
        or not isinstance(hazard_types, Sequence)
        or len(hazard_types) != batch_size
        or any(not isinstance(value, str) for value in hazard_types)
    ):
        raise ValueError("cf_hazard_types must contain one string per NavSim-e120 batch row")
    for name, indices in (
        ("cf_hazard_pair_real_indices", real_indices),
        ("cf_hazard_pair_counterfactual_indices", counterfactual_indices),
    ):
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.long
            or indices.ndim != 1
            or indices.numel() == 0
        ):
            raise ValueError(f"{name} must be a non-empty int64 [num_pairs] tensor")
        if bool(((indices < 0) | (indices >= batch_size)).any().item()):
            raise ValueError(f"{name} contains an out-of-range batch index")
        if len(set(indices.tolist())) != indices.numel():
            raise ValueError(f"{name} must not contain duplicate indices")
    if real_indices.shape != counterfactual_indices.shape:
        raise ValueError("NavSim-e120 matched hazard pair index tensors must have equal shape")
    real_list = real_indices.tolist()
    counterfactual_list = counterfactual_indices.tolist()
    expected_counterfactual = {index for index, domain in enumerate(domains) if domain == "counterfactual"}
    if set(counterfactual_list) != expected_counterfactual:
        raise ValueError("NavSim-e120 every counterfactual hazard must have exactly one matched real factual sample")
    normalized_types = list(hazard_types)
    hazard_cpu = hazard.cpu()
    for real_index, counterfactual_index in zip(real_list, counterfactual_list):
        if domains[real_index] != "real" or hazard_cpu[real_index].item() or normalized_types[real_index]:
            raise ValueError(f"NavSim-e120 matched factual index {real_index} is not an unlabeled real row")
        if domains[counterfactual_index] != "counterfactual" or not hazard_cpu[counterfactual_index].item():
            raise ValueError(f"NavSim-e120 matched counterfactual index {counterfactual_index} is not a hazard")
        if normalized_types[counterfactual_index] not in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST:
            raise ValueError("NavSim-e120 counterfactual hazard type must remain in the exact accident_type allowlist")
    real_device_indices = real_indices.to(device=values.device)
    counterfactual_device_indices = counterfactual_indices.to(device=values.device)
    differences = values[real_device_indices] - values[counterfactual_device_indices]
    loss = torch.relu(float(margin) - differences).mean()
    type_count = len({normalized_types[index] for index in counterfactual_list})
    return loss, len(real_list), type_count


def _quality_ranking_loss(
    values: torch.Tensor,
    quality: torch.Tensor,
    *,
    margin: float,
) -> Tuple[torch.Tensor, int]:
    ordered = quality[:, None] > quality[None, :]
    pair_count = int(ordered.sum().item())
    if pair_count == 0:
        return _zero_loss(values), 0
    differences = values[:, None, :] - values[None, :, :]
    losses = torch.relu(float(margin) - differences)
    expanded_order = ordered.unsqueeze(-1).expand_as(losses)
    return losses[expanded_order].mean(), pair_count


def _select_real_geometry_targets(
    targets: torch.Tensor,
    *,
    real_mask: torch.Tensor,
    expected_shape: torch.Size,
) -> torch.Tensor:
    if not isinstance(targets, torch.Tensor):
        raise ValueError("real_geometry_targets must be a tensor when real samples are present")
    real_count = int(real_mask.sum().item())
    real_shape = (real_count, int(expected_shape[1]))
    if tuple(targets.shape) == tuple(expected_shape):
        selected = targets.to(device=real_mask.device)[real_mask]
    elif tuple(targets.shape) == real_shape:
        selected = targets.to(device=real_mask.device)
    else:
        raise ValueError(
            "real_geometry_targets must be [B, F] or [num_real, F], got "
            f"{tuple(targets.shape)} for field_values {tuple(expected_shape)}"
        )
    if not selected.is_floating_point():
        raise ValueError("real_geometry_targets must be floating point")
    if not bool(torch.isfinite(selected).all().item()):
        raise ValueError("real_geometry_targets contains NaN/Inf in real rows")
    return selected.detach()


def compute_domain_routed_field_loss(
    field_values: torch.Tensor,
    dataset_domains: Sequence[str],
    *,
    real_geometry_targets: Optional[torch.Tensor] = None,
    real_group_ids: Optional[Sequence[str]] = None,
    cf_hazard: Optional[torch.Tensor] = None,
    cf_hazard_types: Optional[Sequence[str]] = None,
    cf_quality: Optional[torch.Tensor] = None,
    huber_delta: float = 1.0,
    real_order_weight: float = 0.0,
    real_order_margin: float = 0.0,
    cf_hazard_weight: float = 1.0,
    cf_quality_weight: float = 1.0,
    cf_ranking_margin: float = 1.0,
) -> CVoIValueLossOutput:
    """Route real field calibration and CF ranking without target leakage.

    Real samples consume caller-supplied, protocol-specific absolute targets.
    Counterfactual samples never index or validate real targets; they receive
    only hazard and/or quality pairwise ranking supervision.
    """

    if not isinstance(field_values, torch.Tensor) or field_values.ndim != 2 or field_values.shape[1] < 1:
        raise ValueError(f"field_values must be non-empty [B, F], got {getattr(field_values, 'shape', None)}")
    if not field_values.is_floating_point():
        raise ValueError("field_values must be floating point")
    if not bool(torch.isfinite(field_values).all().item()):
        raise ValueError("field_values contains NaN/Inf")
    domains = _validate_domains(dataset_domains, batch_size=int(field_values.shape[0]))
    huber_delta = float(huber_delta)
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError(f"huber_delta must be finite and positive, got {huber_delta}")
    real_order_weight = _validate_nonnegative_finite(real_order_weight, name="real_order_weight")
    real_order_margin = _validate_nonnegative_finite(real_order_margin, name="real_order_margin")
    cf_hazard_weight = _validate_nonnegative_finite(cf_hazard_weight, name="cf_hazard_weight")
    cf_quality_weight = _validate_nonnegative_finite(cf_quality_weight, name="cf_quality_weight")
    cf_ranking_margin = _validate_nonnegative_finite(cf_ranking_margin, name="cf_ranking_margin")

    real_mask = torch.as_tensor([domain == "real" for domain in domains], device=field_values.device)
    cf_mask = ~real_mask
    real_count = int(real_mask.sum().item())
    cf_count = int(cf_mask.sum().item())
    zero = _zero_loss(field_values)

    real_huber = zero
    real_order = zero
    real_order_pairs = 0
    if real_count:
        if real_geometry_targets is None:
            raise ValueError("real_geometry_targets is required when real samples are present")
        selected_targets = _select_real_geometry_targets(
            real_geometry_targets,
            real_mask=real_mask,
            expected_shape=field_values.shape,
        ).to(dtype=field_values.dtype)
        if bool(((selected_targets < 0.0) | (selected_targets > 1.0)).any().item()):
            raise ValueError("real_geometry_targets must be in [0, 1]")
        selected_values = field_values[real_mask]
        real_huber = F.huber_loss(selected_values, selected_targets, reduction="mean", delta=huber_delta)
        if real_order_weight > 0.0:
            if real_group_ids is None:
                raise ValueError("real_group_ids is required when real_order_weight > 0")
            if isinstance(real_group_ids, (str, bytes)) or not isinstance(real_group_ids, Sequence):
                raise ValueError("real_group_ids must be a sequence")
            if len(real_group_ids) == field_values.shape[0]:
                selected_group_ids = [
                    str(group_id) for group_id, keep in zip(real_group_ids, real_mask.tolist()) if keep
                ]
            elif len(real_group_ids) == real_count:
                selected_group_ids = [str(group_id) for group_id in real_group_ids]
            else:
                raise ValueError("real_group_ids must have length B or num_real")
            real_order, real_order_pairs = _local_order_loss(
                selected_values,
                selected_targets,
                group_ids=selected_group_ids,
                margin=real_order_margin,
            )

    cf_hazard_loss = zero
    cf_quality_loss = zero
    cf_hazard_pairs = 0
    cf_hazard_type_count = 0
    cf_quality_pairs = 0
    if cf_count:
        if cf_hazard is None and cf_quality is None and (cf_hazard_weight > 0.0 or cf_quality_weight > 0.0):
            raise ValueError("counterfactual field supervision requires hazard or quality ranking labels")
        cf_values = field_values[cf_mask]
        if cf_hazard_weight > 0.0 and cf_hazard is not None:
            if (
                not isinstance(cf_hazard, torch.Tensor)
                or cf_hazard.dtype != torch.bool
                or cf_hazard.ndim != 1
                or cf_hazard.shape[0] != field_values.shape[0]
            ):
                raise ValueError(f"cf_hazard must be bool [B], got {getattr(cf_hazard, 'shape', None)}")
            selected_hazard = cf_hazard.to(device=field_values.device)[cf_mask]
            if isinstance(cf_hazard_types, (str, bytes)) or not isinstance(cf_hazard_types, Sequence):
                raise ValueError("cf_hazard_types is required with one type per batch sample")
            if len(cf_hazard_types) != field_values.shape[0]:
                raise ValueError("cf_hazard_types length must match batch size")
            normalized_hazard_types = [str(value) for value in cf_hazard_types]
            for index, (domain, is_hazard, hazard_type) in enumerate(
                zip(domains, cf_hazard.tolist(), normalized_hazard_types)
            ):
                if domain == "real" and (is_hazard or hazard_type):
                    raise ValueError(f"real sample {index} cannot carry CF hazard labels")
                if domain == "counterfactual" and is_hazard and hazard_type not in KNOWN_HAZARD_TYPES:
                    raise ValueError(
                        f"counterfactual hazard sample {index} has unknown cf_hazard_type {hazard_type!r}"
                    )
                if domain == "counterfactual" and not is_hazard and hazard_type:
                    raise ValueError(f"counterfactual safe sample {index} must have empty cf_hazard_type")
            selected_hazard_types = [
                hazard_type for hazard_type, keep in zip(normalized_hazard_types, cf_mask.tolist()) if keep
            ]
            cf_hazard_loss, cf_hazard_pairs, cf_hazard_type_count = _hazard_ranking_loss(
                cf_values,
                selected_hazard,
                selected_hazard_types,
                margin=cf_ranking_margin,
            )
        if cf_quality_weight > 0.0 and cf_quality is not None:
            if (
                not isinstance(cf_quality, torch.Tensor)
                or cf_quality.ndim != 1
                or cf_quality.shape[0] != field_values.shape[0]
            ):
                raise ValueError(f"cf_quality must be [B], got {getattr(cf_quality, 'shape', None)}")
            selected_quality = cf_quality.to(device=field_values.device)[cf_mask]
            if not selected_quality.is_floating_point():
                raise ValueError("cf_quality must be floating point")
            if not bool(torch.isfinite(selected_quality).all().item()):
                raise ValueError("cf_quality contains NaN/Inf in counterfactual rows")
            if bool(((selected_quality < 0.0) | (selected_quality > 1.0)).any().item()):
                raise ValueError("cf_quality must be in [0, 1] for counterfactual rows")
            cf_quality_loss, cf_quality_pairs = _quality_ranking_loss(
                cf_values,
                selected_quality.detach(),
                margin=cf_ranking_margin,
            )

    loss = (
        real_huber
        + real_order_weight * real_order
        + cf_hazard_weight * cf_hazard_loss
        + cf_quality_weight * cf_quality_loss
    )
    diagnostics = {
        "field_loss": float(loss.detach()),
        "real_huber_loss": float(real_huber.detach()),
        "real_order_loss": float(real_order.detach()),
        "cf_hazard_ranking_loss": float(cf_hazard_loss.detach()),
        "cf_quality_ranking_loss": float(cf_quality_loss.detach()),
        "real_count": float(real_count),
        "cf_count": float(cf_count),
        "real_order_pairs": float(real_order_pairs),
        "cf_hazard_pairs": float(cf_hazard_pairs),
        "cf_hazard_type_count": float(cf_hazard_type_count),
        "cf_quality_pairs": float(cf_quality_pairs),
    }
    return CVoIValueLossOutput(loss=loss, diagnostics=diagnostics)


def compute_real_stop_value_loss(
    stop_values: torch.Tensor,
    stop_targets: torch.Tensor,
    dataset_domains: Sequence[str],
    *,
    huber_delta: float = 1.0,
) -> CVoIValueLossOutput:
    """Calibrate stop values on an exclusively real batch.

    Target provenance is selected and validated by the protocol-specific
    caller. Counterfactual samples fail before targets are inspected.
    """

    if not isinstance(stop_values, torch.Tensor) or stop_values.ndim != 2 or stop_values.shape[1] < 1:
        raise ValueError(f"stop_values must be non-empty [B, F+1], got {getattr(stop_values, 'shape', None)}")
    domains = _validate_domains(dataset_domains, batch_size=int(stop_values.shape[0]))
    if any(domain != "real" for domain in domains):
        raise ValueError("stop value calibration is real-only and rejects counterfactual samples")
    if not isinstance(stop_targets, torch.Tensor) or stop_targets.shape != stop_values.shape:
        raise ValueError(
            f"stop_targets shape {getattr(stop_targets, 'shape', None)} does not match {tuple(stop_values.shape)}"
        )
    if not stop_values.is_floating_point() or not stop_targets.is_floating_point():
        raise ValueError("stop_values and stop_targets must be floating point")
    if not bool(torch.isfinite(stop_values).all().item()) or not bool(torch.isfinite(stop_targets).all().item()):
        raise ValueError("stop_values and stop_targets must be finite")
    if bool(((stop_targets < 0.0) | (stop_targets > 1.0)).any().item()):
        raise ValueError("stop_targets must be in [0, 1]")
    huber_delta = float(huber_delta)
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError(f"huber_delta must be finite and positive, got {huber_delta}")

    loss = F.huber_loss(
        stop_values,
        stop_targets.detach().to(device=stop_values.device, dtype=stop_values.dtype),
        reduction="mean",
        delta=huber_delta,
    )
    return CVoIValueLossOutput(
        loss=loss,
        diagnostics={
            "stop_loss": float(loss.detach()),
            "real_count": float(stop_values.shape[0]),
            "stop_prefix_count": float(stop_values.numel()),
        },
    )


def compute_navsim_e120_quality_field_loss(
    field_values: torch.Tensor,
    dataset_domains: Sequence[str],
    *,
    real_quality_targets: Optional[torch.Tensor] = None,
    real_group_ids: Optional[Sequence[str]] = None,
    cf_hazard: Optional[torch.Tensor] = None,
    cf_hazard_types: Optional[Sequence[str]] = None,
    cf_hazard_pair_real_indices: Optional[torch.Tensor] = None,
    cf_hazard_pair_counterfactual_indices: Optional[torch.Tensor] = None,
    cf_quality: Optional[torch.Tensor] = None,
    **loss_kwargs: object,
) -> CVoIValueLossOutput:
    """Train the NavSim-e120 Field on explicit quality and signed CF labels."""

    normalized_loss_kwargs = dict(loss_kwargs)
    cf_hazard_weight = _validate_nonnegative_finite(
        normalized_loss_kwargs.pop("cf_hazard_weight", 1.0),
        name="cf_hazard_weight",
    )
    cf_ranking_margin = _validate_nonnegative_finite(
        normalized_loss_kwargs.pop("cf_ranking_margin", 1.0),
        name="cf_ranking_margin",
    )
    base = compute_domain_routed_field_loss(
        field_values,
        dataset_domains,
        real_geometry_targets=real_quality_targets,
        real_group_ids=real_group_ids,
        cf_hazard=cf_hazard,
        cf_hazard_types=cf_hazard_types,
        cf_quality=cf_quality,
        cf_hazard_weight=0.0,
        cf_ranking_margin=cf_ranking_margin,
        **normalized_loss_kwargs,
    )
    if cf_hazard_weight == 0.0 or "counterfactual" not in dataset_domains:
        return base
    hazard_loss, pair_count, type_count = _matched_real_counterfactual_hazard_ranking_loss(
        field_values,
        dataset_domains,
        cf_hazard,
        cf_hazard_types,
        cf_hazard_pair_real_indices,
        cf_hazard_pair_counterfactual_indices,
        margin=cf_ranking_margin,
    )
    loss = base.loss + cf_hazard_weight * hazard_loss
    diagnostics = dict(base.diagnostics)
    diagnostics.update(
        {
            "field_loss": float(loss.detach()),
            "cf_hazard_ranking_loss": float(hazard_loss.detach()),
            "cf_hazard_pairs": float(pair_count),
            "cf_hazard_type_count": float(type_count),
        }
    )
    return CVoIValueLossOutput(loss=loss, diagnostics=diagnostics)


def compute_navsim_e120_stop_quality_loss(
    stop_values: torch.Tensor,
    quality_targets: torch.Tensor,
    dataset_domains: Sequence[str],
    *,
    huber_delta: float = 1.0,
) -> CVoIValueLossOutput:
    """Calibrate NavSim-e120 Stop values against route-free quality."""

    return compute_real_stop_value_loss(
        stop_values,
        quality_targets,
        dataset_domains,
        huber_delta=huber_delta,
    )


def _prefix_dual_value_architecture(model: PrefixDualValueModel) -> Dict[str, object]:
    if not isinstance(model, PrefixDualValueModel):
        raise TypeError(f"model must be PrefixDualValueModel, got {type(model).__name__}")
    return {
        "embed_dim": model.embed_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "dropout": model.dropout,
    }


def _validate_prefix_dual_value_architecture(architecture: object) -> Dict[str, object]:
    if not isinstance(architecture, Mapping):
        raise ValueError("PrefixDualValueModel checkpoint architecture must be a mapping")
    fields = frozenset(architecture.keys())
    missing = _PREFIX_DUAL_VALUE_ARCHITECTURE_FIELDS - fields
    unexpected = fields - _PREFIX_DUAL_VALUE_ARCHITECTURE_FIELDS
    if missing or unexpected:
        raise ValueError(
            "PrefixDualValueModel checkpoint architecture fields mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    normalized: Dict[str, object] = {}
    for field in ("embed_dim", "hidden_dim", "num_layers"):
        value = architecture[field]
        if type(value) is not int or value <= 0:
            raise ValueError(f"PrefixDualValueModel checkpoint architecture {field} must be a positive integer")
        normalized[field] = value
    dropout = architecture["dropout"]
    if type(dropout) is not float or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("PrefixDualValueModel checkpoint architecture dropout must be a float in [0, 1)")
    normalized["dropout"] = dropout
    return normalized


def _require_direct_value_exact_fields(
    value: Mapping[object, object],
    fields: frozenset[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    missing = sorted(fields - actual, key=repr)
    unexpected = sorted(actual - fields, key=repr)
    if missing or unexpected:
        raise ValueError(f"{name} fields mismatch: missing={missing}, unexpected={unexpected}")


def _reject_direct_value_proof_keys(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if isinstance(key, str):
                lowered = key.lower()
                is_sha_key = (
                    lowered == "sha" or "sha256" in lowered or lowered.startswith("sha_") or lowered.endswith("_sha")
                )
                if is_sha_key or any(marker in lowered for marker in _NAVSIM_E120_DIRECT_VALUE_FORBIDDEN_KEY_MARKERS):
                    raise ValueError(f"direct Value payload contains forbidden proof key {path}.{key}")
            _reject_direct_value_proof_keys(nested, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            _reject_direct_value_proof_keys(nested, path=f"{path}[{index}]")


def _normalize_direct_value_state(value: object, *, name: str) -> Dict[str, torch.Tensor]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    normalized: Dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if not isinstance(raw_key, str) or not raw_key or not torch.is_tensor(tensor):
            raise ValueError(f"{name} must map non-empty string keys to tensors")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not key or key in normalized:
            raise ValueError(f"{name} contains duplicate or empty normalized key {key!r}")
        normalized[key] = tensor.detach().cpu().clone(memory_format=torch.preserve_format)
    return {key: normalized[key] for key in sorted(normalized)}


def _direct_value_role_from_state(state: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    keys = sorted(state)
    return {
        "keys": keys,
        "shapes": {key: list(state[key].shape) for key in keys},
    }


def _validate_direct_value_roles(
    value: object,
    *,
    state: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"value_model"}:
        raise ValueError("direct Value checkpoint roles must contain exactly 'value_model'")
    role = value["value_model"]
    if not isinstance(role, Mapping):
        raise ValueError("direct Value checkpoint roles.value_model must be a mapping")
    _require_direct_value_exact_fields(
        role,
        _NAVSIM_E120_DIRECT_VALUE_ROLE_FIELDS,
        name="direct Value checkpoint roles.value_model",
    )
    keys = role["keys"]
    if type(keys) is not list or not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("direct Value checkpoint value_model keys must be a non-empty list of strings")
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise ValueError("direct Value checkpoint value_model keys must be sorted and unique")
    shapes = role["shapes"]
    if not isinstance(shapes, Mapping) or set(shapes) != set(keys):
        raise ValueError("direct Value checkpoint value_model shapes keys must exactly match keys")
    normalized_shapes: Dict[str, list[int]] = {}
    for key in keys:
        shape = shapes[key]
        if type(shape) is not list or any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError(
                f"direct Value checkpoint value_model shapes.{key} must contain non-negative integer dimensions"
            )
        normalized_shapes[key] = list(shape)
    state_keys = set(state)
    if set(keys) != state_keys:
        raise ValueError("direct Value checkpoint value_model state keys do not match roles")
    expected_shapes = {key: list(state[key].shape) for key in keys}
    if normalized_shapes != expected_shapes:
        raise ValueError("direct Value checkpoint value_model state shapes do not match roles")
    return {
        "value_model": {
            "keys": list(keys),
            "shapes": normalized_shapes,
        }
    }


def _strict_load_direct_value_state(
    model: PrefixDualValueModel,
    state: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> None:
    target_state = _normalize_direct_value_state(
        model.state_dict(),
        name=f"direct Value {name} state",
    )
    if set(target_state) != set(state):
        raise ValueError(f"direct Value checkpoint {name} state keys mismatch")
    target_shapes = {key: list(target_state[key].shape) for key in sorted(target_state)}
    checkpoint_shapes = {key: list(state[key].shape) for key in sorted(state)}
    if target_shapes != checkpoint_shapes:
        raise ValueError(f"direct Value checkpoint {name} state shapes mismatch")
    model.load_state_dict(state, strict=True)


def _validate_direct_value_parents(
    value: object,
    *,
    expected: Mapping[str, object],
    phase: str,
) -> Dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("direct Value checkpoint parents must be a mapping")
    _require_direct_value_exact_fields(
        value,
        frozenset(expected),
        name=f"direct Value checkpoint {phase} parents",
    )
    for parent_name, expected_parent in expected.items():
        parent = value[parent_name]
        if not isinstance(parent, Mapping):
            raise ValueError(f"direct Value checkpoint parent {parent_name!r} must be a mapping")
        _require_direct_value_exact_fields(
            parent,
            frozenset(expected_parent),
            name=f"direct Value checkpoint parent {parent_name!r}",
        )
        if dict(parent) != expected_parent:
            raise ValueError(f"direct Value checkpoint parent {parent_name!r} must be exactly {expected_parent!r}")
    return copy.deepcopy(expected)


def build_cvoi_navsim_e120_direct_value_checkpoint(
    model: PrefixDualValueModel,
    *,
    phase: str,
    branch_id: str,
    epoch: int,
    parents: Mapping[str, object],
) -> Dict[str, object]:
    """Build one proof-free structural Value checkpoint."""

    state = _normalize_direct_value_state(model.state_dict(), name="direct Value model state")
    payload = {
        "schema": CVOI_NAVSIM_E120_VALUE_CHECKPOINT_SCHEMA,
        "phase": phase,
        "protocol_version": CVOI_VALUE_PROTOCOL_FORMAL_V2_NAVSIM_E120,
        "branch_id": branch_id,
        "epoch": epoch,
        "architecture": _prefix_dual_value_architecture(model),
        "roles": {"value_model": _direct_value_role_from_state(state)},
        "parents": parents,
        "state_dict": state,
    }
    return validate_cvoi_navsim_e120_direct_value_checkpoint(payload)


def validate_cvoi_navsim_e120_direct_value_checkpoint(
    payload: object,
    *,
    required_phase: Optional[str] = None,
    required_branch_id: Optional[str] = None,
    target_model: Optional[PrefixDualValueModel] = None,
) -> Dict[str, object]:
    """Validate and optionally strict-load one proof-free e120 Value checkpoint."""

    if not isinstance(payload, Mapping):
        raise ValueError("direct Value checkpoint must be a mapping")
    _reject_direct_value_proof_keys(payload, path="checkpoint")
    _require_direct_value_exact_fields(
        payload,
        _NAVSIM_E120_DIRECT_VALUE_CHECKPOINT_FIELDS,
        name="direct Value checkpoint",
    )
    if payload["schema"] != CVOI_NAVSIM_E120_VALUE_CHECKPOINT_SCHEMA:
        raise ValueError(f"direct Value checkpoint schema must be {CVOI_NAVSIM_E120_VALUE_CHECKPOINT_SCHEMA!r}")
    if payload["protocol_version"] != CVOI_VALUE_PROTOCOL_FORMAL_V2_NAVSIM_E120:
        raise ValueError(
            "direct Value checkpoint protocol_version must be " f"{CVOI_VALUE_PROTOCOL_FORMAL_V2_NAVSIM_E120!r}"
        )
    phase = payload["phase"]
    if type(phase) is not str or phase not in _NAVSIM_E120_DIRECT_VALUE_PHASES:
        raise ValueError(
            "direct Value checkpoint phase must be one of " f"{CVOI_VALUE_CHECKPOINT_PHASES!r}, got {phase!r}"
        )
    branch_id = payload["branch_id"]
    lineage = resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase=phase,
        branch_id=branch_id,
    )
    expected_parents = build_cvoi_manual_value_parents(lineage, phase)
    parents = _validate_direct_value_parents(
        payload["parents"],
        expected=expected_parents,
        phase=phase,
    )
    epoch = payload["epoch"]
    if type(epoch) is not int or epoch <= 0:
        raise ValueError("direct Value checkpoint epoch must be a positive integer")
    if required_phase is not None:
        if type(required_phase) is not str or required_phase not in _NAVSIM_E120_DIRECT_VALUE_PHASES:
            raise ValueError(f"required phase is unsupported: {required_phase!r}")
        if phase != required_phase:
            raise ValueError(f"direct Value checkpoint required phase {required_phase!r}, got {phase!r}")
    if required_branch_id is not None:
        if not isinstance(required_branch_id, str) or not required_branch_id:
            raise ValueError("required branch must be a non-empty string")
        if branch_id != required_branch_id:
            raise ValueError(f"direct Value checkpoint required branch {required_branch_id!r}, got {branch_id!r}")

    architecture = _validate_prefix_dual_value_architecture(payload["architecture"])
    state = _normalize_direct_value_state(payload["state_dict"], name="direct Value checkpoint state_dict")
    roles = _validate_direct_value_roles(payload["roles"], state=state)
    with torch.random.fork_rng(devices=[]):
        architecture_probe = PrefixDualValueModel(**architecture)
        _strict_load_direct_value_state(
            architecture_probe,
            state,
            name="declared architecture",
        )
    if target_model is not None:
        target_architecture = _prefix_dual_value_architecture(target_model)
        if architecture != target_architecture:
            raise ValueError(
                "direct Value checkpoint architecture mismatch: "
                f"expected {target_architecture!r}, got {architecture!r}"
            )
        _strict_load_direct_value_state(target_model, state, name="target model")

    return {
        "schema": CVOI_NAVSIM_E120_VALUE_CHECKPOINT_SCHEMA,
        "phase": phase,
        "protocol_version": CVOI_VALUE_PROTOCOL_FORMAL_V2_NAVSIM_E120,
        "branch_id": branch_id,
        "epoch": epoch,
        "architecture": architecture,
        "roles": roles,
        "parents": parents,
        "state_dict": state,
    }


def read_cvoi_navsim_e120_direct_value_checkpoint(
    path: str | Path,
    *,
    required_phase: str,
    required_branch_id: str,
    target_model: Optional[PrefixDualValueModel] = None,
    map_location: Any = "cpu",
) -> Dict[str, object]:
    """Read a normal absolute file and fully validate its structural payload."""

    artifact = Path(path)
    if not artifact.is_absolute():
        raise ValueError(f"direct Value checkpoint path must be absolute: {artifact}")
    if artifact.is_symlink():
        raise ValueError(f"direct Value checkpoint must be a non-symlink regular file: {artifact}")
    if not artifact.exists():
        raise FileNotFoundError(f"direct Value checkpoint does not exist: {artifact}")
    if not artifact.is_file():
        raise ValueError(f"direct Value checkpoint must be a non-symlink regular file: {artifact}")
    payload = torch.load(artifact, map_location=map_location, weights_only=True)
    return validate_cvoi_navsim_e120_direct_value_checkpoint(
        payload,
        required_phase=required_phase,
        required_branch_id=required_branch_id,
        target_model=target_model,
    )


def read_cvoi_navsim_e120_direct_calibration_checkpoint_metadata(
    path: str | Path,
    *,
    required_branch_id: str,
) -> Dict[str, object]:
    """Read and fully validate Calibration, returning its seven-field callback metadata."""

    payload = read_cvoi_navsim_e120_direct_value_checkpoint(
        path,
        required_phase="field_calibrated",
        required_branch_id=required_branch_id,
    )
    metadata = {key: payload[key] for key in _NAVSIM_E120_DIRECT_VALUE_METADATA_FIELDS}
    return copy.deepcopy(metadata)
