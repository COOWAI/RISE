"""Stage-aware training epochs for the CVoI dual-value model."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import CVOI_CF_FIELD_WEIGHTS
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
from app.vjepa_cowa_world_model.training.cvoi_value import (
    CVOI_FIELD_WARMUP_DATA_SCHEMA,
    CVOI_VALUE_CHECKPOINT_PHASES,
    compute_navsim_e120_quality_field_loss,
    compute_navsim_e120_stop_quality_loss,
    validate_field_warmup_data_provenance,
)

_FIELD_PHASES = frozenset({"field_warmup", "field_calibrated"})
_BASE_BATCH_KEYS = frozenset({"z_observed", "z_future", "dataset_domains"})
_VALID_DOMAINS = frozenset({"real", "counterfactual"})
_FIELD_WARMUP_DOMAINS = frozenset({"real", "real_cf"})
_NAVSIM_E120_PROTOCOL = "formal_v2_navsim_e120_h4_v3"


def _validate_navsim_e120_protocol(protocol_version: object) -> str:
    if protocol_version != _NAVSIM_E120_PROTOCOL:
        raise ValueError(
            "NavSim-e120 Value training requires "
            f"protocol_version={_NAVSIM_E120_PROTOCOL!r}, got {protocol_version!r}"
        )
    return _NAVSIM_E120_PROTOCOL


def build_field_warmup_data_provenance(
    *,
    field_warmup_domain: str,
    train_roots: Sequence[Mapping[str, object]],
    epoch_metrics: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    """Bind a warm-up checkpoint to configured roots and observed domain counts."""

    if field_warmup_domain not in _FIELD_WARMUP_DOMAINS:
        raise ValueError(
            "field_warmup_domain must be one of " f"{sorted(_FIELD_WARMUP_DOMAINS)}, got {field_warmup_domain!r}"
        )
    if isinstance(train_roots, (str, bytes)) or not isinstance(train_roots, Sequence) or not train_roots:
        raise ValueError("Field warm-up train_roots must be a non-empty sequence")
    roots = []
    for index, root in enumerate(train_roots):
        if not isinstance(root, Mapping):
            raise ValueError(f"Field warm-up train_roots[{index}] must be a mapping")
        name = root.get("name")
        domain = root.get("domain")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Field warm-up train_roots[{index}] requires a non-empty name")
        roots.append({"name": name, "domain": domain})
    if isinstance(epoch_metrics, (str, bytes)) or not isinstance(epoch_metrics, Sequence) or not epoch_metrics:
        raise ValueError("Field warm-up epoch_metrics must be a non-empty sequence")
    counts = {"real_sample_count": 0, "cf_sample_count": 0}
    for epoch, metrics in enumerate(epoch_metrics):
        if not isinstance(metrics, Mapping):
            raise ValueError(f"Field warm-up epoch_metrics[{epoch}] must be a mapping")
        for name in counts:
            value = metrics.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Field warm-up epoch_metrics[{epoch}].{name} must be a finite integer count")
            normalized = int(value)
            if normalized < 0 or float(value) != float(normalized):
                raise ValueError(f"Field warm-up epoch_metrics[{epoch}].{name} must be a non-negative integer count")
            counts[name] += normalized
    return validate_field_warmup_data_provenance(
        {
            "schema": CVOI_FIELD_WARMUP_DATA_SCHEMA,
            "domain": field_warmup_domain,
            "roots": roots,
            **counts,
        }
    )


def _configure_phase_parameters(model: PrefixDualValueModel, *, phase: str) -> None:
    """Apply the lifecycle's strict parameter ownership policy."""

    if phase in _FIELD_PHASES:
        model.requires_grad_(True)
        model.stop_head.requires_grad_(False)
        model.train()
        return

    model.requires_grad_(False)
    model.stop_head.requires_grad_(True)
    # The frozen GRU may contain inter-layer dropout. Keep its representation
    # deterministic while the linear stop head is calibrated.
    model.eval()
    model.stop_head.train()


def _validate_optimizer_coverage(
    model: PrefixDualValueModel,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer_parameter_ids = {
        id(parameter) for parameter_group in optimizer.param_groups for parameter in parameter_group.get("params", ())
    }
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
    ]
    if missing:
        raise ValueError(f"optimizer does not contain trainable CVoI value parameters: {missing}")


def _value_model_dtype(model: PrefixDualValueModel) -> torch.dtype:
    dtypes = {parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()}
    if len(dtypes) != 1:
        raise ValueError(f"PrefixDualValueModel must use exactly one floating dtype, got {sorted(map(str, dtypes))}")
    return next(iter(dtypes))


def _require_tensor(batch: Mapping[str, object], key: str) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"CVoI value batch field {key!r} must be a tensor")
    return value


def _validate_latents(batch: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor, int]:
    z_observed = _require_tensor(batch, "z_observed")
    z_future = _require_tensor(batch, "z_future")
    for name, latent in (("z_observed", z_observed), ("z_future", z_future)):
        if latent.ndim not in (3, 4):
            raise ValueError(f"{name} must be [B, N, D] or [B, F, T, D], got {tuple(latent.shape)}")
        if not latent.is_floating_point():
            raise ValueError(f"{name} must be floating point")
        if not bool(torch.isfinite(latent).all().item()):
            raise ValueError(f"{name} contains NaN/Inf")
    if z_observed.shape[0] < 1:
        raise ValueError("CVoI value batch must contain at least one sample")
    if z_future.shape[0] != z_observed.shape[0]:
        raise ValueError(
            f"z_future batch size {z_future.shape[0]} does not match z_observed batch size {z_observed.shape[0]}"
        )
    return z_observed, z_future, int(z_observed.shape[0])


def _validate_domains(dataset_domains: object, *, batch_size: int) -> tuple[str, ...]:
    if isinstance(dataset_domains, (str, bytes)) or not isinstance(dataset_domains, Sequence):
        raise ValueError("dataset_domains must be a sequence with one explicit domain per sample")
    domains = tuple(str(domain) for domain in dataset_domains)
    if len(domains) != batch_size:
        raise ValueError(f"dataset_domains length {len(domains)} does not match batch size {batch_size}")
    unknown = sorted(set(domains) - _VALID_DOMAINS)
    if unknown:
        raise ValueError(f"dataset_domains contains unknown domains: {unknown}")
    return domains


def _validate_batch_keys(
    batch: object,
    *,
    phase: str,
) -> Mapping[str, object]:
    if not isinstance(batch, Mapping):
        raise ValueError("each CVoI value batch must be a mapping")
    missing = sorted(_BASE_BATCH_KEYS - frozenset(batch.keys()))
    if missing:
        raise ValueError(f"CVoI value batch is missing required keys: {missing}")
    if phase == "stop_calibrated" and "stop_quality_targets" not in batch:
        raise ValueError("CVoI stop_calibrated batch is missing required key: stop_quality_targets")
    return batch


def _prepare_field_targets(
    batch: Mapping[str, object],
    *,
    domains: Sequence[str],
    device: torch.device,
    cf_field_supervision: str,
) -> Dict[str, object]:
    real_target_key = "real_quality_targets"
    required = []
    if "real" in domains:
        required.append(real_target_key)
    if "counterfactual" in domains:
        if cf_field_supervision in {"hazard_only", "hazard_quality"}:
            required.extend(("cf_hazard", "cf_hazard_types"))
            required.extend(
                (
                    "cf_hazard_pair_real_indices",
                    "cf_hazard_pair_counterfactual_indices",
                    "cf_hazard_pair_keys",
                )
            )
        if cf_field_supervision in {"quality_only", "hazard_quality"}:
            required.append("cf_quality")
    missing = [key for key in required if key not in batch]
    if missing:
        raise ValueError(f"CVoI field batch is missing domain-specific targets: {missing}")

    targets: Dict[str, object] = {
        "real_targets": None,
        "real_group_ids": None,
        "cf_hazard": None,
        "cf_hazard_types": None,
        "cf_hazard_pair_real_indices": None,
        "cf_hazard_pair_counterfactual_indices": None,
        "cf_hazard_pair_keys": None,
        "cf_quality": None,
    }
    tensor_keys = [real_target_key]
    sequence_keys = ["real_group_ids"]
    if cf_field_supervision in {"hazard_only", "hazard_quality"}:
        tensor_keys.append("cf_hazard")
        sequence_keys.append("cf_hazard_types")
        tensor_keys.extend(
            (
                "cf_hazard_pair_real_indices",
                "cf_hazard_pair_counterfactual_indices",
            )
        )
        sequence_keys.append("cf_hazard_pair_keys")
    if cf_field_supervision in {"quality_only", "hazard_quality"}:
        tensor_keys.append("cf_quality")
    for key in tensor_keys:
        if key in batch:
            output_key = "real_targets" if key == real_target_key else key
            targets[output_key] = _require_tensor(batch, key).to(device=device)
    for key in sequence_keys:
        if key in batch:
            value = batch[key]
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ValueError(f"CVoI value batch field {key!r} must be a sequence")
            targets[key] = list(value)
    return targets


def _loss_kwargs(value: Optional[Mapping[str, object]], *, name: str) -> Dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _is_additive_diagnostic(name: str) -> bool:
    return name.endswith("_count") or name.endswith("_pairs")


def _ranking_accuracy(differences: torch.Tensor) -> tuple[float, float]:
    if differences.numel() == 0:
        return 0.0, 0.0
    accuracy = float((differences > 0.0).to(dtype=torch.float32).mean().item())
    return accuracy, 1.0 - accuracy


def _matched_hazard_validation_diagnostics(
    values: torch.Tensor,
    *,
    real_indices: torch.Tensor,
    counterfactual_indices: torch.Tensor,
    pair_keys: Sequence[tuple[str, int]],
    cf_hazard_count: int,
    ranking_loss: float,
    valid_pair_count: int,
) -> Dict[str, object]:
    """Describe exact factual-vs-CF hazard pairs without inventing safe CF rows."""

    expected_count = cf_hazard_count
    factual_anchor_count = int(real_indices.numel())
    if expected_count < 1 or factual_anchor_count != expected_count:
        raise ValueError("matched Real/CF validation requires one factual anchor for every CF hazard")
    if int(counterfactual_indices.numel()) != expected_count or valid_pair_count != expected_count:
        raise ValueError("matched Real/CF validation requires complete exact-pair coverage")
    differences = (
        values[real_indices.to(device=values.device)] - values[counterfactual_indices.to(device=values.device)]
    )
    accuracy, violation_rate = _ranking_accuracy(differences)
    canonical_keys = json.dumps(
        sorted([list(key) for key in pair_keys]),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")
    return {
        "enabled": True,
        "cf_hazard_count": cf_hazard_count,
        "factual_anchor_count": factual_anchor_count,
        "expected_matched_pair_count": expected_count,
        "valid_matched_pair_count": valid_pair_count,
        "pair_coverage": valid_pair_count / expected_count,
        "matched_pair_key_set_sha256": hashlib.sha256(canonical_keys).hexdigest(),
        "ranking_loss": ranking_loss,
        "ranking_accuracy": accuracy,
        "violation_rate": violation_rate,
    }


def _validated_matched_pair_batch(
    *,
    domains: Sequence[str],
    targets: Mapping[str, object],
    seen_pair_keys: set[tuple[str, int]],
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[str, int]]]:
    """Validate one atomic mixed batch and return its exact local pair map."""

    batch_size = len(domains)
    real_indices = targets["cf_hazard_pair_real_indices"]
    counterfactual_indices = targets["cf_hazard_pair_counterfactual_indices"]
    hazard = targets["cf_hazard"]
    hazard_types = targets["cf_hazard_types"]
    raw_pair_keys = targets["cf_hazard_pair_keys"]
    for name, indices in (
        ("cf_hazard_pair_real_indices", real_indices),
        ("cf_hazard_pair_counterfactual_indices", counterfactual_indices),
    ):
        if not isinstance(indices, torch.Tensor) or indices.dtype != torch.long or indices.ndim != 1:
            raise ValueError(f"{name} must be an int64 [num_pairs] tensor")
        if indices.numel() == 0 or bool(((indices < 0) | (indices >= batch_size)).any().item()):
            raise ValueError(f"{name} must contain non-empty in-range batch indices")
        if len(set(indices.tolist())) != indices.numel():
            raise ValueError(f"{name} must not contain duplicate indices")
    if real_indices.shape != counterfactual_indices.shape:
        raise ValueError("matched Real/CF validation pair index tensors must have equal shape")
    expected_real = {index for index, domain in enumerate(domains) if domain == "real"}
    expected_cf = {index for index, domain in enumerate(domains) if domain == "counterfactual"}
    if (
        not expected_cf
        or set(real_indices.tolist()) != expected_real
        or set(counterfactual_indices.tolist()) != expected_cf
    ):
        raise ValueError("matched Real/CF validation requires one unique factual anchor for every CF hazard")
    if not isinstance(hazard, torch.Tensor) or hazard.dtype != torch.bool or tuple(hazard.shape) != (batch_size,):
        raise ValueError(f"cf_hazard must be bool [{batch_size}] for matched Real/CF validation")
    if (
        isinstance(hazard_types, (str, bytes))
        or not isinstance(hazard_types, Sequence)
        or len(hazard_types) != batch_size
    ):
        raise ValueError("cf_hazard_types must contain one string per matched validation row")
    for index, domain in enumerate(domains):
        hazard_type = hazard_types[index]
        if not isinstance(hazard_type, str):
            raise ValueError("cf_hazard_types must contain strings")
        if domain == "real" and (bool(hazard[index].item()) or hazard_type):
            raise ValueError("matched factual anchors must be unlabeled real rows")
        if domain == "counterfactual" and (
            not bool(hazard[index].item()) or hazard_type not in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
        ):
            raise ValueError("matched counterfactual rows must be hazards in the exact accident-type allowlist")
    if isinstance(raw_pair_keys, (str, bytes)) or not isinstance(raw_pair_keys, Sequence):
        raise ValueError("cf_hazard_pair_keys must be a sequence")
    pair_keys: list[tuple[str, int]] = []
    for raw_key in raw_pair_keys:
        if (
            not isinstance(raw_key, (tuple, list))
            or len(raw_key) != 2
            or type(raw_key[0]) is not str
            or not raw_key[0]
            or type(raw_key[1]) is not int
        ):
            raise ValueError(f"invalid matched Real/CF pair key {raw_key!r}")
        key = (raw_key[0], raw_key[1])
        if key in seen_pair_keys or key in pair_keys:
            raise ValueError(f"duplicate matched Real/CF pair key {key!r}")
        pair_keys.append(key)
    if len(pair_keys) != len(expected_cf):
        raise ValueError("cf_hazard_pair_keys must contain one key per matched CF hazard")
    seen_pair_keys.update(pair_keys)
    return real_indices.cpu(), counterfactual_indices.cpu(), pair_keys


def _quality_validation_diagnostics(
    values: torch.Tensor,
    quality: Optional[torch.Tensor],
    *,
    enabled: bool,
    ranking_loss: float,
    valid_pair_count: int,
) -> Dict[str, object]:
    if not enabled:
        return {"enabled": False}
    if not isinstance(quality, torch.Tensor):
        raise RuntimeError("enabled quality validation requires collected quality labels")
    quality = quality.to(device=values.device)
    ordered = quality[:, None] > quality[None, :]
    differences = values[:, None, :] - values[None, :, :]
    selected = differences[ordered.unsqueeze(-1).expand_as(differences)]
    accuracy, violation_rate = _ranking_accuracy(selected)
    label_count = int(quality.shape[0])
    candidate_pair_count = label_count * (label_count - 1) // 2
    return {
        "enabled": True,
        "label_count": label_count,
        "candidate_pair_count": candidate_pair_count,
        "valid_pair_count": valid_pair_count,
        "pair_coverage": 0.0 if candidate_pair_count == 0 else valid_pair_count / candidate_pair_count,
        "ranking_loss": ranking_loss,
        "ranking_accuracy": accuracy,
        "violation_rate": violation_rate,
    }


def _collect_field_validation_domain(
    model: PrefixDualValueModel,
    batches: Iterable[Mapping[str, object]],
    *,
    expected_scope: str,
    device: torch.device,
    model_dtype: torch.dtype,
    tokens_per_frame: Optional[int],
    cf_field_supervision: str,
    field_loss_kwargs: Mapping[str, object],
) -> Dict[str, object]:
    """Evaluate one explicit validation scope and compute global pair diagnostics."""

    if expected_scope not in {"real", "counterfactual", "matched_real_counterfactual"}:
        raise ValueError(f"unknown CVoI Field validation scope {expected_scope!r}")

    field_values: list[torch.Tensor] = []
    real_targets: list[torch.Tensor] = []
    real_group_ids: list[str] = []
    cf_hazard: list[torch.Tensor] = []
    cf_hazard_types: list[str] = []
    cf_quality: list[torch.Tensor] = []
    all_domains: list[str] = []
    matched_real_indices: list[torch.Tensor] = []
    matched_counterfactual_indices: list[torch.Tensor] = []
    matched_pair_keys: list[tuple[str, int]] = []
    seen_pair_keys: set[tuple[str, int]] = set()
    num_batches = 0
    sample_count = 0
    row_count = 0
    future_width: Optional[int] = None

    for raw_batch in batches:
        batch = _validate_batch_keys(
            raw_batch,
            phase="field_warmup",
        )
        z_observed, z_future, batch_size = _validate_latents(batch)
        domains = _validate_domains(batch["dataset_domains"], batch_size=batch_size)
        if expected_scope in {"real", "counterfactual"} and any(domain != expected_scope for domain in domains):
            raise ValueError(f"{expected_scope} validation loader must contain only {expected_scope} samples")
        if expected_scope == "matched_real_counterfactual" and set(domains) != {"real", "counterfactual"}:
            raise ValueError("matched_real_counterfactual validation loader requires exact mixed Real/CF pairs")
        targets = _prepare_field_targets(
            batch,
            domains=domains,
            device=device,
            cf_field_supervision=cf_field_supervision,
        )
        output = model(
            z_observed.to(device=device, dtype=model_dtype),
            z_future.to(device=device, dtype=model_dtype),
            tokens_per_frame=tokens_per_frame,
        )
        values = output.field_values.detach()
        if values.ndim != 2 or values.shape[0] != batch_size or values.shape[1] < 1:
            raise ValueError(f"CVoI Field validation model output must be non-empty [B, F], got {tuple(values.shape)}")
        if not bool(torch.isfinite(values).all().item()):
            raise ValueError("CVoI Field validation model output contains NaN/Inf")
        if future_width is None:
            future_width = int(values.shape[1])
        elif int(values.shape[1]) != future_width:
            raise ValueError(
                "CVoI Field validation requires a fixed future width across the full loader, "
                f"expected {future_width}, got {int(values.shape[1])}"
            )
        field_values.append(values)
        all_domains.extend(domains)
        if "real" in domains:
            target = targets["real_targets"]
            if not isinstance(target, torch.Tensor):
                raise RuntimeError("real validation targets were not prepared")
            real_targets.append(target.detach())
            groups = targets["real_group_ids"]
            if groups is not None:
                real_group_ids.extend(str(value) for value in groups)
        if expected_scope != "real":
            hazard = targets["cf_hazard"]
            if isinstance(hazard, torch.Tensor):
                cf_hazard.append(hazard.detach())
            hazard_types = targets["cf_hazard_types"]
            if hazard_types is not None:
                cf_hazard_types.extend(str(value) for value in hazard_types)
            quality = targets["cf_quality"]
            if isinstance(quality, torch.Tensor):
                cf_quality.append(quality.detach())
            if expected_scope == "matched_real_counterfactual":
                local_real, local_cf, local_keys = _validated_matched_pair_batch(
                    domains=domains,
                    targets=targets,
                    seen_pair_keys=seen_pair_keys,
                )
                matched_real_indices.append(local_real + row_count)
                matched_counterfactual_indices.append(local_cf + row_count)
                matched_pair_keys.extend(local_keys)
        num_batches += 1
        sample_count += sum(domain == ("real" if expected_scope == "real" else "counterfactual") for domain in domains)
        row_count += batch_size

    if num_batches == 0:
        raise ValueError(f"CVoI Field {expected_scope} validation requires at least one batch")

    all_values = torch.cat(field_values, dim=0)
    all_real_pair_indices = torch.cat(matched_real_indices) if matched_real_indices else None
    all_counterfactual_pair_indices = (
        torch.cat(matched_counterfactual_indices) if matched_counterfactual_indices else None
    )
    hazard_weight, quality_weight = CVOI_CF_FIELD_WEIGHTS[cf_field_supervision]
    common_loss_inputs = {
        "real_group_ids": real_group_ids or None,
        "cf_hazard": torch.cat(cf_hazard, dim=0) if cf_hazard else None,
        "cf_hazard_types": cf_hazard_types or None,
        "cf_quality": torch.cat(cf_quality, dim=0) if cf_quality else None,
        "cf_hazard_weight": hazard_weight,
        "cf_quality_weight": quality_weight,
        **field_loss_kwargs,
    }
    loss_output = compute_navsim_e120_quality_field_loss(
        all_values,
        all_domains,
        real_quality_targets=torch.cat(real_targets, dim=0) if real_targets else None,
        cf_hazard_pair_real_indices=all_real_pair_indices,
        cf_hazard_pair_counterfactual_indices=all_counterfactual_pair_indices,
        **common_loss_inputs,
    )
    if not bool(torch.isfinite(loss_output.loss).item()):
        raise ValueError(f"CVoI Field {expected_scope} validation loss is not finite")
    diagnostics = {name: float(value) for name, value in loss_output.diagnostics.items()}
    if any(not math.isfinite(value) for value in diagnostics.values()):
        raise ValueError(f"CVoI Field {expected_scope} validation diagnostics contain NaN/Inf")
    hazard_enabled = expected_scope != "real" and hazard_weight > 0.0
    quality_enabled = expected_scope != "real" and quality_weight > 0.0
    all_quality = torch.cat(cf_quality, dim=0) if cf_quality else None
    if expected_scope != "real":
        diagnostics["field_loss"] = (
            hazard_weight * diagnostics["cf_hazard_ranking_loss"]
            + quality_weight * diagnostics["cf_quality_ranking_loss"]
        )
    cf_mask = torch.as_tensor([domain == "counterfactual" for domain in all_domains], dtype=torch.bool)
    if hazard_enabled:
        if all_real_pair_indices is None or all_counterfactual_pair_indices is None:
            raise ValueError("NavSim-e120 hazard validation requires exact matched Real/CF pair indices")
        hazard_ranking = _matched_hazard_validation_diagnostics(
            all_values,
            real_indices=all_real_pair_indices,
            counterfactual_indices=all_counterfactual_pair_indices,
            pair_keys=matched_pair_keys,
            cf_hazard_count=int(cf_mask.sum().item()),
            ranking_loss=diagnostics["cf_hazard_ranking_loss"],
            valid_pair_count=int(diagnostics["cf_hazard_pairs"]),
        )
    else:
        hazard_ranking = {"enabled": False}
    quality_values = all_values[cf_mask] if expected_scope != "real" else all_values
    quality_labels = all_quality[cf_mask] if all_quality is not None and expected_scope != "real" else all_quality
    diagnostics.update(
        {
            "domain": "real" if expected_scope == "real" else "counterfactual",
            "num_batches": num_batches,
            "sample_count": sample_count,
            "hazard_ranking": hazard_ranking,
            "quality_ranking": _quality_validation_diagnostics(
                quality_values,
                quality_labels,
                enabled=quality_enabled,
                ranking_loss=diagnostics["cf_quality_ranking_loss"],
                valid_pair_count=int(diagnostics["cf_quality_pairs"]),
            ),
        }
    )
    return diagnostics


@torch.no_grad()
def validate_cvoi_field_epoch(
    model: PrefixDualValueModel,
    *,
    real_batches: Iterable[Mapping[str, object]],
    counterfactual_batches: Optional[Iterable[Mapping[str, object]]],
    device: torch.device,
    tokens_per_frame: Optional[int] = None,
    field_loss_kwargs: Optional[Mapping[str, object]] = None,
    cf_field_supervision: str = "hazard_quality",
    protocol_version: str = _NAVSIM_E120_PROTOCOL,
) -> Dict[str, object]:
    """Run NavSim-e120 diagnostic-only real/CF Field validation.

    The two loaders remain separate so a strict real-only branch cannot accidentally consume
    counterfactual validation samples. Counterfactual predictions and labels are concatenated
    before the ranking loss is computed, allowing valid pairs to span batch boundaries.
    """

    if not isinstance(model, PrefixDualValueModel):
        raise TypeError(f"model must be PrefixDualValueModel, got {type(model).__name__}")
    _validate_navsim_e120_protocol(protocol_version)
    if cf_field_supervision not in CVOI_CF_FIELD_WEIGHTS:
        raise ValueError(
            "cf_field_supervision must be one of " f"{sorted(CVOI_CF_FIELD_WEIGHTS)}, got {cf_field_supervision!r}"
        )
    if cf_field_supervision == "none" and counterfactual_batches is not None:
        raise ValueError("counterfactual_batches must be None when cf_field_supervision='none'")
    if cf_field_supervision != "none" and counterfactual_batches is None:
        raise ValueError(f"cf_field_supervision={cf_field_supervision!r} requires counterfactual_batches")
    field_kwargs = _loss_kwargs(field_loss_kwargs, name="field_loss_kwargs")
    owned_weight_keys = {"cf_hazard_weight", "cf_quality_weight"}
    if owned_weight_keys.intersection(field_kwargs):
        raise ValueError("CF field weights are owned by cf_field_supervision and cannot be overridden")

    device = torch.device(device)
    model.to(device=device)
    model_dtype = _value_model_dtype(model)
    training_modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        real_metrics = _collect_field_validation_domain(
            model,
            real_batches,
            expected_scope="real",
            device=device,
            model_dtype=model_dtype,
            tokens_per_frame=tokens_per_frame,
            cf_field_supervision=cf_field_supervision,
            field_loss_kwargs=field_kwargs,
        )
        cf_metrics = None
        if counterfactual_batches is not None:
            counterfactual_scope = (
                "matched_real_counterfactual"
                if cf_field_supervision in {"hazard_only", "hazard_quality"}
                else "counterfactual"
            )
            cf_metrics = _collect_field_validation_domain(
                model,
                counterfactual_batches,
                expected_scope=counterfactual_scope,
                device=device,
                model_dtype=model_dtype,
                tokens_per_frame=tokens_per_frame,
                cf_field_supervision=cf_field_supervision,
                field_loss_kwargs=field_kwargs,
            )
            hazard_weight, quality_weight = CVOI_CF_FIELD_WEIGHTS[cf_field_supervision]
            if hazard_weight > 0.0 and cf_metrics["cf_hazard_pairs"] <= 0.0:
                raise ValueError("CVoI Field counterfactual validation requires cf_hazard_pairs>0")
            if quality_weight > 0.0 and cf_metrics["cf_quality_pairs"] <= 0.0:
                raise ValueError("CVoI Field counterfactual validation requires cf_quality_pairs>0")
        return {"real": real_metrics, "counterfactual": cf_metrics}
    finally:
        for module, training in training_modes.items():
            module.training = training


def train_cvoi_value_epoch(
    model: PrefixDualValueModel,
    batches: Iterable[Mapping[str, object]],
    *,
    optimizer: torch.optim.Optimizer,
    phase: str,
    device: torch.device,
    tokens_per_frame: Optional[int] = None,
    field_loss_kwargs: Optional[Mapping[str, object]] = None,
    stop_loss_kwargs: Optional[Mapping[str, object]] = None,
    cf_field_supervision: str = "hazard_quality",
    field_warmup_domain: str = "real_cf",
    value_updates_per_epoch: Optional[int] = None,
    protocol_version: str = _NAVSIM_E120_PROTOCOL,
) -> Dict[str, object]:
    """Train one NavSim-e120 lifecycle phase of the CVoI dual-value model.

    ``field_warmup`` routes its configured real or Real+CF supervision by domain.
    ``stop_calibrated`` and ``field_calibrated`` are always real-only. All
    absolute targets use the route-free NavSim-e120 quality contract.

    Loss diagnostics are averaged over optimizer steps. Counts and pair counts
    are accumulated over the epoch.
    """

    if not isinstance(model, PrefixDualValueModel):
        raise TypeError(f"model must be PrefixDualValueModel, got {type(model).__name__}")
    _validate_navsim_e120_protocol(protocol_version)
    if phase not in CVOI_VALUE_CHECKPOINT_PHASES:
        raise ValueError(f"phase must be one of {CVOI_VALUE_CHECKPOINT_PHASES}, got {phase!r}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(f"optimizer must be torch.optim.Optimizer, got {type(optimizer).__name__}")
    device = torch.device(device)
    field_kwargs = _loss_kwargs(field_loss_kwargs, name="field_loss_kwargs")
    stop_kwargs = _loss_kwargs(stop_loss_kwargs, name="stop_loss_kwargs")
    if phase in _FIELD_PHASES and stop_kwargs:
        raise ValueError("stop_loss_kwargs are not valid during a field phase")
    if phase == "stop_calibrated" and field_kwargs:
        raise ValueError("field_loss_kwargs are not valid during stop calibration")
    if cf_field_supervision not in CVOI_CF_FIELD_WEIGHTS:
        raise ValueError(
            "cf_field_supervision must be one of " f"{sorted(CVOI_CF_FIELD_WEIGHTS)}, got {cf_field_supervision!r}"
        )
    if field_warmup_domain not in _FIELD_WARMUP_DOMAINS:
        raise ValueError(
            "field_warmup_domain must be one of " f"{sorted(_FIELD_WARMUP_DOMAINS)}, got {field_warmup_domain!r}"
        )
    owned_weight_keys = {"cf_hazard_weight", "cf_quality_weight"}
    if owned_weight_keys.intersection(field_kwargs):
        raise ValueError("CF field weights are owned by cf_field_supervision and cannot be overridden")
    cf_hazard_weight, cf_quality_weight = CVOI_CF_FIELD_WEIGHTS[cf_field_supervision]
    if value_updates_per_epoch is not None and (
        type(value_updates_per_epoch) is not int or value_updates_per_epoch <= 0
    ):
        raise ValueError(
            "value_updates_per_epoch must be a positive integer when set, " f"got {value_updates_per_epoch!r}"
        )

    model.to(device=device)
    _configure_phase_parameters(model, phase=phase)
    _validate_optimizer_coverage(model, optimizer)
    model_dtype = _value_model_dtype(model)

    diagnostic_totals: Dict[str, float] = {}
    num_batches = 0
    sample_count = 0
    real_sample_count = 0
    cf_sample_count = 0
    schedule_rows: list[dict[str, object]] = []
    schedule_ids_missing = False
    batch_iterator = iter(batches)

    def _bounded_batches():
        if value_updates_per_epoch is None:
            yield from batch_iterator
            return
        for update_index in range(value_updates_per_epoch):
            try:
                yield next(batch_iterator)
            except StopIteration as exc:
                raise ValueError(
                    f"CVoI value epoch required {value_updates_per_epoch} updates but provided only "
                    f"{update_index} batches"
                ) from exc

    for raw_batch in _bounded_batches():
        batch = _validate_batch_keys(
            raw_batch,
            phase=phase,
        )
        z_observed, z_future, batch_size = _validate_latents(batch)
        domains = _validate_domains(batch["dataset_domains"], batch_size=batch_size)
        real_only = phase in {"field_calibrated", "stop_calibrated"}
        if real_only and any(domain != "real" for domain in domains):
            raise ValueError(f"CVoI {phase} is real-only and rejects counterfactual samples before model forward")
        if phase == "field_warmup" and field_warmup_domain == "real" and "counterfactual" in domains:
            raise ValueError("strict Real-only Field warm-up rejects counterfactual samples before model forward")
        if "stable_sample_ids" in batch:
            sample_ids = batch["stable_sample_ids"]
            if isinstance(sample_ids, (str, bytes)) or not isinstance(sample_ids, Sequence):
                raise ValueError("stable_sample_ids must be a sequence")
            normalized_sample_ids = [str(value) for value in sample_ids]
            if len(normalized_sample_ids) != batch_size or any(not value for value in normalized_sample_ids):
                raise ValueError("stable_sample_ids must contain one non-empty value per batch sample")
            schedule_rows.append({"sample_ids": normalized_sample_ids, "dataset_domains": list(domains)})
        else:
            schedule_ids_missing = True
        if phase in _FIELD_PHASES:
            field_targets = _prepare_field_targets(
                batch,
                domains=domains,
                device=device,
                cf_field_supervision=cf_field_supervision,
            )
            stop_quality_targets = None
        else:
            field_targets = None
            stop_quality_targets = _require_tensor(batch, "stop_quality_targets").to(device=device)

        z_observed = z_observed.to(device=device, dtype=model_dtype)
        z_future = z_future.to(device=device, dtype=model_dtype)
        optimizer.zero_grad(set_to_none=True)
        output = model(z_observed, z_future, tokens_per_frame=tokens_per_frame)

        if phase in _FIELD_PHASES:
            if field_targets is None:
                raise RuntimeError("field targets were not prepared for a field phase")
            common_loss_inputs = {
                "real_group_ids": field_targets["real_group_ids"],
                "cf_hazard": field_targets["cf_hazard"],
                "cf_hazard_types": field_targets["cf_hazard_types"],
                "cf_quality": field_targets["cf_quality"],
                "cf_hazard_weight": cf_hazard_weight,
                "cf_quality_weight": cf_quality_weight,
                **field_kwargs,
            }
            common_loss_inputs.update(
                {
                    "cf_hazard_pair_real_indices": field_targets["cf_hazard_pair_real_indices"],
                    "cf_hazard_pair_counterfactual_indices": field_targets["cf_hazard_pair_counterfactual_indices"],
                }
            )
            loss_output = compute_navsim_e120_quality_field_loss(
                output.field_values,
                domains,
                real_quality_targets=field_targets["real_targets"],
                **common_loss_inputs,
            )
        else:
            if stop_quality_targets is None:
                raise RuntimeError("stop targets were not prepared for stop calibration")
            loss_output = compute_navsim_e120_stop_quality_loss(
                output.stop_values,
                stop_quality_targets,
                domains,
                **stop_kwargs,
            )

        if not bool(torch.isfinite(loss_output.loss).item()):
            raise ValueError(f"CVoI value {phase} loss is not finite")
        loss_output.loss.backward()
        optimizer.step()

        diagnostics = dict(loss_output.diagnostics)
        diagnostics["loss"] = float(loss_output.loss.detach().cpu())
        for name, value in diagnostics.items():
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"CVoI value diagnostic {name!r} is not finite")
            diagnostic_totals[name] = diagnostic_totals.get(name, 0.0) + value
        num_batches += 1
        sample_count += batch_size
        real_sample_count += sum(domain == "real" for domain in domains)
        cf_sample_count += sum(domain == "counterfactual" for domain in domains)
        diagnostic_totals["cf_forward_count"] = diagnostic_totals.get("cf_forward_count", 0.0) + float(
            "counterfactual" in domains
        )

    if num_batches == 0:
        raise ValueError("CVoI value training requires at least one batch")

    diagnostics = {
        name: total if _is_additive_diagnostic(name) else total / num_batches
        for name, total in diagnostic_totals.items()
    }
    diagnostics.update(
        {
            "num_batches": float(num_batches),
            "optimizer_steps": float(num_batches),
            "optimizer_step_count": float(num_batches),
            "cf_batch_count": float(diagnostic_totals.get("cf_forward_count", 0.0)),
            "sample_count": float(sample_count),
            "real_sample_count": float(real_sample_count),
            "cf_sample_count": float(cf_sample_count),
        }
    )
    if value_updates_per_epoch is not None:
        diagnostics["value_updates_per_epoch"] = float(value_updates_per_epoch)
    if schedule_rows and not schedule_ids_missing:
        canonical = json.dumps(schedule_rows, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        diagnostics["eligibility_schedule_signature"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return diagnostics
