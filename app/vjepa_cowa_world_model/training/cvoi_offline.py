"""Manual NavSim-e120 Value and SQLite Gate training stages."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training.artifact_publish import atomic_torch_save_replace
from app.vjepa_cowa_world_model.training.configs.cvoi_ablation import CVOI_CF_FIELD_WEIGHTS
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
from app.vjepa_cowa_world_model.training.cvoi_gate_pipeline import (
    CvoiGateTrainingReport,
    _train_cvoi_navsim_e120_official_gate_from_open_store,
)
from app.vjepa_cowa_world_model.training.cvoi_manual_lineage import (
    CvoiManualGateBranch,
    CvoiManualValueLineage,
    build_cvoi_manual_value_parents,
    derive_cvoi_manual_value_oracle_handoff,
    reject_unedited_cvoi_public_placeholders,
    resolve_cvoi_manual_ablation_results_root_from_config,
    resolve_cvoi_manual_full_results_root,
    resolve_cvoi_manual_full_results_root_from_config,
    resolve_cvoi_manual_gate_branch,
    resolve_cvoi_manual_value_lineage,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import open_embedded_oracle_store_v2
from app.vjepa_cowa_world_model.training.cvoi_value import (
    build_cvoi_navsim_e120_direct_value_checkpoint,
    read_cvoi_navsim_e120_direct_value_checkpoint,
)
from app.vjepa_cowa_world_model.training.cvoi_value_training import train_cvoi_value_epoch, validate_cvoi_field_epoch


class CvoiOfflineAdapterRequiredError(RuntimeError):
    """Raised before data access when an offline model/data adapter is absent."""


@dataclass(frozen=True)
class CvoiValueTrainingReport:
    """Metrics and artifact produced by one Value lifecycle stage."""

    phase: str
    checkpoint_path: Path
    epoch_metrics: tuple[dict[str, object], ...]
    field_validation_metrics: Optional[dict[str, object]] = None


CvoiOfflineReport = CvoiGateTrainingReport | CvoiValueTrainingReport

_NAVSIM_E120_ADAPTER_INPUT_SCHEMAS = {
    "field_warmup": "navsim_e120_quality_field_batch_v1",
    "field_calibrated": "navsim_e120_local_quality_calibration_v1",
    "stop_calibrated": "navsim_e120_stop_quality_v1",
}
_PLANNER_STAGES = frozenset({"unguided_planner", "guided_planner"})
_OFFLINE_GATE_HIDDEN_DIM = 128
_OFFLINE_GATE_TEMPERATURE = 0.05
_OFFLINE_GATE_REGRESSION_WEIGHT = 0.5
_FORMAL_V2_NAVSIM_E120_PROTOCOL = "formal_v2_navsim_e120_h4_v3"
_DIRECT_VALUE_STAGES = frozenset({"field_warmup", "field_calibrated", "stop_calibrated"})


def _require_config_section(config: Any, name: str) -> Any:
    section = getattr(config, name, None)
    if section is None:
        raise ValueError(f"CVoI offline training requires config.{name}")
    return section


def _adapter_required(
    stage: str,
    *,
    expected_schema: Optional[str] = None,
) -> CvoiOfflineAdapterRequiredError:
    schema = _NAVSIM_E120_ADAPTER_INPUT_SCHEMAS[stage] if expected_schema is None else expected_schema
    return CvoiOfflineAdapterRequiredError(
        f"cvoi.stage={stage!r} requires a production dataset/model adapter emitting {schema}; "
        "train_cvoi_offline refuses to infer targets from raw batches or fabricate supervision"
    )


def _direct_navsim_e120_value_handoff_paths(
    lineage: CvoiManualValueLineage,
    *,
    stage: str,
) -> dict[str, Path]:
    if stage == "field_warmup":
        return {
            "unguided_planner_checkpoint": lineage.p0_handoff,
            "output_checkpoint": lineage.field_handoff,
        }
    if stage == "field_calibrated":
        return {
            "unguided_planner_checkpoint": lineage.p0_handoff,
            "field_checkpoint": lineage.field_handoff,
            "output_checkpoint": lineage.calibration_handoff,
        }
    if stage == "stop_calibrated":
        return {
            "unguided_planner_checkpoint": lineage.p0_handoff,
            "field_checkpoint": lineage.calibration_handoff,
            "guided_planner_checkpoint": lineage.p1_handoff,
            "output_checkpoint": lineage.stop_handoff,
        }
    raise ValueError(f"direct NavSim-e120 Value stage is unsupported: {stage!r}")


def _preflight_direct_navsim_e120_value_handoffs(config: Any) -> CvoiManualValueLineage:
    reject_unedited_cvoi_public_placeholders(config, boundary="CVoI Value")
    stage = getattr(config.cvoi, "stage", None)
    if stage not in _DIRECT_VALUE_STAGES:
        raise ValueError(f"direct NavSim-e120 Value stage is unsupported: {stage!r}")
    signature = getattr(config.cvoi, "ablation_signature", None)
    if signature is None:
        raise ValueError("direct NavSim-e120 Value stages require cvoi.ablation_signature")
    experiment_role = getattr(signature, "experiment_role", None)
    full_results_root = resolve_cvoi_manual_full_results_root_from_config(config.cvoi)
    ablation_results_root = (
        resolve_cvoi_manual_ablation_results_root_from_config(config.cvoi) if experiment_role == "ablation" else None
    )
    lineage = resolve_cvoi_manual_value_lineage(
        signature,
        stage=stage,
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )
    for field_name, path in _direct_navsim_e120_value_handoff_paths(lineage, stage=stage).items():
        expected_path = str(path)
        actual_path = getattr(config.cvoi, field_name, None)
        if type(actual_path) is not str or actual_path != expected_path:
            raise ValueError(f"cvoi.{field_name} must use fixed handoff path {expected_path!r}, got {actual_path!r}")

    output_path = Path(config.cvoi.output_checkpoint)
    output_parent = output_path.parent
    if output_parent.is_symlink():
        raise ValueError(f"direct NavSim-e120 output parent directory must not be a symlink: {output_parent}")
    if not output_parent.exists():
        raise FileNotFoundError(f"direct NavSim-e120 output parent directory does not exist: {output_parent}")
    if not output_parent.is_dir():
        raise ValueError(f"direct NavSim-e120 output parent must be a directory: {output_parent}")
    if output_path.is_symlink() or (output_path.exists() and not output_path.is_file()):
        raise ValueError(
            "direct NavSim-e120 output target must be absent or a regular non-symlink file: " f"{output_path}"
        )
    return lineage


def _uses_direct_navsim_e120_value_handoffs(config: Any) -> bool:
    return (
        getattr(config.cvoi, "protocol_version", None) == _FORMAL_V2_NAVSIM_E120_PROTOCOL
        and getattr(config.cvoi, "stage", None) in _DIRECT_VALUE_STAGES
    )


def _preflight_direct_navsim_e120_gate_handoffs(cvoi: Any) -> tuple[str, str, CvoiManualGateBranch]:
    reject_unedited_cvoi_public_placeholders(cvoi, boundary="CVoI Gate")
    signature = getattr(cvoi, "ablation_signature", None)
    if signature is None:
        raise ValueError("direct NavSim-e120 Gate requires cvoi.ablation_signature")
    experiment_role = getattr(signature, "experiment_role", None)
    if experiment_role == "main":
        full_results_root = resolve_cvoi_manual_full_results_root_from_config(cvoi)
        ablation_results_root = None
        gate_branch = resolve_cvoi_manual_gate_branch(
            signature,
            full_results_root=full_results_root,
        )
    else:
        full_results_root = None
        ablation_results_root = resolve_cvoi_manual_ablation_results_root_from_config(cvoi)
        gate_branch = resolve_cvoi_manual_gate_branch(
            signature,
            ablation_results_root=ablation_results_root,
        )
    configured_oracle_path = getattr(cvoi, "oracle_path", None)
    if gate_branch.oracle_value_lineage == "full" and experiment_role == "ablation":
        full_results_root = resolve_cvoi_manual_full_results_root({"oracle_handoff": configured_oracle_path})
    expected_oracle_path = derive_cvoi_manual_value_oracle_handoff(
        gate_branch.oracle_value_lineage,
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )
    expected_paths = {
        "oracle_path": expected_oracle_path,
        "output_checkpoint": gate_branch.result_root / "handoff/gate.pt",
    }
    validated_paths = {}
    for field_name, path in expected_paths.items():
        expected_path = str(path)
        actual_path = getattr(cvoi, field_name, None)
        if type(actual_path) is not str or actual_path != expected_path:
            raise ValueError(f"cvoi.{field_name} must use fixed handoff path {expected_path!r}, got {actual_path!r}")
        validated_paths[field_name] = actual_path
    return validated_paths["oracle_path"], validated_paths["output_checkpoint"], gate_branch


def load_cvoi_offline_adapter(config: Any, *, device: torch.device) -> Any:
    """Instantiate an explicit ``module:factory`` adapter from configuration."""

    if not _uses_direct_navsim_e120_value_handoffs(config):
        raise ValueError(
            "offline adapters are supported only for the manual NavSim-e120 " "Field/Calibration/Stop Value stages"
        )
    _preflight_direct_navsim_e120_value_handoffs(config)
    factory_path = getattr(config.cvoi, "offline_adapter_factory", None)
    if not isinstance(factory_path, str) or not factory_path.strip() or ":" not in factory_path:
        raise ValueError("offline CVoI Value stages require cvoi.offline_adapter_factory='module:factory'")
    module_name, attribute_name = factory_path.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("cvoi.offline_adapter_factory must use the exact 'module:factory' form")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ValueError(f"cvoi.offline_adapter_factory is not callable: {factory_path!r}")
    adapter = factory(config=config, device=torch.device(device))
    if adapter is None:
        raise ValueError(f"cvoi.offline_adapter_factory returned None: {factory_path!r}")
    return adapter


def _new_value_model(config: Any, adapter: Any, *, device: torch.device) -> PrefixDualValueModel:
    embed_dim = getattr(adapter, "embed_dim", None)
    if type(embed_dim) is not int or embed_dim <= 0:
        raise ValueError("CVoI Value adapter must declare a positive integer embed_dim")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.meta.seed))
        model = PrefixDualValueModel(
            embed_dim=embed_dim,
            hidden_dim=int(config.cvoi.value_hidden_dim),
            num_layers=int(config.cvoi.value_num_layers),
            dropout=float(config.cvoi.value_dropout),
        )
    return model.to(device=device)


def stream_validated_value_batches(
    adapter: Any,
    *,
    stage: str,
    epoch: int,
    field_warmup_domain: str,
    expected_adapter_schema: Optional[str] = None,
) -> Iterator[Mapping[str, object]]:
    """Yield validated adapter batches without retaining GPU latents across an epoch."""

    callback = getattr(adapter, "value_batches", None)
    if not callable(callback):
        raise _adapter_required(stage, expected_schema=expected_adapter_schema)
    seen_domains = set()
    batch_count = 0
    for batch in callback(stage, epoch, provenance=None):
        if not isinstance(batch, Mapping):
            raise ValueError("CVoI adapter batches must be mappings")
        if "cvoi_provenance" in batch:
            raise ValueError("direct CVoI adapter batches must omit cvoi_provenance")
        expected_schema = (
            _NAVSIM_E120_ADAPTER_INPUT_SCHEMAS[stage] if expected_adapter_schema is None else expected_adapter_schema
        )
        if not isinstance(expected_schema, str) or not expected_schema:
            raise ValueError("expected_adapter_schema must be a non-empty string")
        if batch.get("adapter_schema") != expected_schema:
            raise ValueError(
                f"CVoI adapter batch schema must be {expected_schema!r} for {stage}, "
                f"got {batch.get('adapter_schema')!r}"
            )
        domains = batch.get("dataset_domains")
        if isinstance(domains, (str, bytes)) or not isinstance(domains, Sequence):
            raise ValueError("CVoI adapter batch requires explicit dataset_domains")
        normalized_domains = tuple(str(domain) for domain in domains)
        if stage in {"field_calibrated", "stop_calibrated"} and any(domain != "real" for domain in normalized_domains):
            raise ValueError(f"{stage} requires real-only adapter samples")
        if (
            stage == "field_warmup"
            and field_warmup_domain == "real"
            and any(domain != "real" for domain in normalized_domains)
        ):
            raise ValueError("field_warmup domain='real' requires real-only adapter samples")
        seen_domains.update(normalized_domains)
        batch_count += 1
        yield batch
    if batch_count == 0:
        raise ValueError(f"CVoI adapter returned no batches for {stage} epoch {epoch}")
    if stage == "field_warmup":
        if field_warmup_domain == "real":
            expected_domains = {"real"}
        elif field_warmup_domain == "real_cf":
            expected_domains = {"real", "counterfactual"}
        else:
            raise ValueError("field_warmup_domain must be 'real' or 'real_cf', " f"got {field_warmup_domain!r}")
        if seen_domains != expected_domains:
            raise ValueError(
                f"field_warmup domain={field_warmup_domain!r} requires adapter domains "
                f"{sorted(expected_domains)}, got {sorted(seen_domains)}"
            )
    if stage in {"field_calibrated", "stop_calibrated"} and seen_domains != {"real"}:
        raise ValueError(f"{stage} requires real-only adapter samples")


def stream_validated_field_validation_batches(
    adapter: Any,
    *,
    domain: str,
    expected_adapter_schema: Optional[str] = None,
) -> Iterator[Mapping[str, object]]:
    """Yield one explicit validation scope from the production adapter."""

    if domain not in {"real", "counterfactual", "matched_real_counterfactual"}:
        raise ValueError(
            "Field validation scope must be real, counterfactual, or matched_real_counterfactual, " f"got {domain!r}"
        )
    callback = getattr(adapter, "field_validation_batches", None)
    if not callable(callback):
        raise CvoiOfflineAdapterRequiredError(
            "formal_v2_navsim_e120_h4_v3 Field production requires adapter.field_validation_batches"
            "(*, domain, provenance)"
        )
    batch_count = 0
    for batch in callback(domain=domain, provenance=None):
        if not isinstance(batch, Mapping):
            raise ValueError("CVoI Field validation adapter batches must be mappings")
        if "cvoi_provenance" in batch:
            raise ValueError("direct CVoI Field validation batches must omit cvoi_provenance")
        expected_schema = (
            _NAVSIM_E120_ADAPTER_INPUT_SCHEMAS["field_warmup"]
            if expected_adapter_schema is None
            else expected_adapter_schema
        )
        if not isinstance(expected_schema, str) or not expected_schema:
            raise ValueError("expected_adapter_schema must be a non-empty string")
        if batch.get("adapter_schema") != expected_schema:
            raise ValueError(f"CVoI Field validation adapter batch schema must be {expected_schema!r}")
        domains = batch.get("dataset_domains")
        if isinstance(domains, (str, bytes)) or not isinstance(domains, Sequence) or not domains:
            raise ValueError("CVoI Field validation batch requires explicit non-empty dataset_domains")
        if domain in {"real", "counterfactual"} and any(value != domain for value in domains):
            raise ValueError(f"CVoI Field validation callback domain={domain!r} returned another domain")
        if domain == "matched_real_counterfactual":
            batch_size = len(domains)
            expected_real = {index for index, value in enumerate(domains) if value == "real"}
            expected_cf = {index for index, value in enumerate(domains) if value == "counterfactual"}
            if not expected_cf or len(expected_real) != len(expected_cf) or set(domains) != {"real", "counterfactual"}:
                raise ValueError("matched_real_counterfactual validation requires non-empty exact 1:1 mixed pairs")
            real_indices = batch.get("cf_hazard_pair_real_indices")
            cf_indices = batch.get("cf_hazard_pair_counterfactual_indices")
            for name, indices, expected in (
                ("cf_hazard_pair_real_indices", real_indices, expected_real),
                ("cf_hazard_pair_counterfactual_indices", cf_indices, expected_cf),
            ):
                if (
                    not isinstance(indices, torch.Tensor)
                    or indices.dtype != torch.long
                    or indices.ndim != 1
                    or set(indices.tolist()) != expected
                ):
                    raise ValueError(f"matched_real_counterfactual {name} must exactly cover its domain rows")
            pair_keys = batch.get("cf_hazard_pair_keys")
            if isinstance(pair_keys, (str, bytes)) or not isinstance(pair_keys, Sequence):
                raise ValueError("matched_real_counterfactual requires explicit cf_hazard_pair_keys")
            normalized_keys = []
            for raw_key in pair_keys:
                if (
                    not isinstance(raw_key, (tuple, list))
                    or len(raw_key) != 2
                    or type(raw_key[0]) is not str
                    or not raw_key[0]
                    or type(raw_key[1]) is not int
                ):
                    raise ValueError(f"invalid matched_real_counterfactual pair key {raw_key!r}")
                normalized_keys.append((raw_key[0], raw_key[1]))
            if len(normalized_keys) != len(expected_cf) or len(set(normalized_keys)) != len(normalized_keys):
                raise ValueError("matched_real_counterfactual pair keys must be unique and cover every CF row")
            hazard = batch.get("cf_hazard")
            hazard_types = batch.get("cf_hazard_types")
            if (
                not isinstance(hazard, torch.Tensor)
                or hazard.dtype != torch.bool
                or tuple(hazard.shape) != (batch_size,)
                or isinstance(hazard_types, (str, bytes))
                or not isinstance(hazard_types, Sequence)
                or len(hazard_types) != batch_size
            ):
                raise ValueError("matched_real_counterfactual requires one exact hazard label per row")
            for index, value in enumerate(domains):
                hazard_type = hazard_types[index]
                if value == "real" and (bool(hazard[index].item()) or hazard_type):
                    raise ValueError("matched_real_counterfactual factual anchors must be unlabeled")
                if value == "counterfactual" and (
                    not bool(hazard[index].item()) or hazard_type not in FORMAL_V2_NAVSIM_CF_ACCIDENT_TYPE_ALLOWLIST
                ):
                    raise ValueError(
                        "matched_real_counterfactual CF rows must be hazards in the exact accident-type allowlist"
                    )
        batch_count += 1
        yield batch
    if batch_count == 0:
        raise ValueError(f"CVoI Field validation adapter returned no {domain} batches")


def _build_value_optimizer(model: torch.nn.Module, optimization: Any) -> torch.optim.AdamW:
    optimizer_name = str(getattr(optimization, "optimizer", "adamw")).lower()
    if optimizer_name != "adamw":
        raise ValueError(f"CVoI Value training requires optimization.optimizer='adamw', got {optimizer_name!r}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("CVoI Value optimizer requires at least one trainable parameter")
    return torch.optim.AdamW(
        parameters,
        lr=float(optimization.lr),
        weight_decay=float(optimization.weight_decay),
        betas=tuple(float(value) for value in optimization.betas),
        eps=float(optimization.eps),
    )


def _validate_field_warmup_pair_coverage(
    metrics: Mapping[str, object],
    *,
    epoch: int,
    cf_field_supervision: str = "hazard_quality",
) -> None:
    if cf_field_supervision not in CVOI_CF_FIELD_WEIGHTS:
        raise ValueError(f"unknown cf_field_supervision {cf_field_supervision!r}")
    hazard_weight, quality_weight = CVOI_CF_FIELD_WEIGHTS[cf_field_supervision]
    required = []
    if hazard_weight > 0.0:
        required.append("cf_hazard_pairs")
    if quality_weight > 0.0:
        required.append("cf_quality_pairs")
    for name in required:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"field_warmup epoch {epoch} requires finite diagnostic {name}")
        if float(value) <= 0.0:
            raise ValueError(f"field_warmup epoch {epoch} requires {name}>0; CF ranking supervision was ineffective")


def _load_direct_parent_value_model(
    config: Any,
    *,
    adapter: Any,
    phase: str,
    lineage: CvoiManualValueLineage,
    device: torch.device,
) -> PrefixDualValueModel:
    parent_phase = "field_warmup" if phase == "field_calibrated" else "field_calibrated"
    parent_branch = lineage.checkpoint_branch_id(parent_phase)
    target_model = _new_value_model(config, adapter, device=device)
    read_cvoi_navsim_e120_direct_value_checkpoint(
        config.cvoi.field_checkpoint,
        required_phase=parent_phase,
        required_branch_id=parent_branch,
        target_model=target_model,
        map_location="cpu",
    )
    return target_model


def _train_direct_navsim_e120_value_stage(
    config: Any,
    *,
    adapter: Any,
    device: torch.device,
) -> CvoiValueTrainingReport:
    phase = str(config.cvoi.stage)
    lineage = _preflight_direct_navsim_e120_value_handoffs(config)
    if phase == "stop_calibrated" and getattr(config.cvoi, "controller_lineage", None) != "value_guided":
        raise ValueError("direct NavSim-e120 Stop requires cvoi.controller_lineage='value_guided'")
    if phase == "field_warmup":
        model = _new_value_model(config, adapter, device=device)
    else:
        model = _load_direct_parent_value_model(
            config,
            adapter=adapter,
            phase=phase,
            lineage=lineage,
            device=device,
        )

    if phase in {"field_warmup", "field_calibrated"}:
        model.requires_grad_(True)
        model.stop_head.requires_grad_(False)
    else:
        model.requires_grad_(False)
        model.stop_head.requires_grad_(True)
    optimizer = _build_value_optimizer(model, config.optimization)
    cf_mode = str(lineage.cf_field_supervision)
    field_warmup_domain = getattr(config.cvoi, "field_warmup_domain", None)
    expected_warmup_domain = "real" if cf_mode == "none" else "real_cf"
    if field_warmup_domain != expected_warmup_domain or type(field_warmup_domain) is not str:
        raise ValueError(
            f"direct {lineage.name} NavSim-e120 Value requires "
            f"cvoi.field_warmup_domain={expected_warmup_domain!r}, got {field_warmup_domain!r}"
        )

    epoch_metrics = []
    for epoch in range(int(config.optimization.epochs)):
        batches = stream_validated_value_batches(
            adapter,
            stage=phase,
            epoch=epoch,
            field_warmup_domain=field_warmup_domain,
            expected_adapter_schema=_NAVSIM_E120_ADAPTER_INPUT_SCHEMAS[phase],
        )
        field_kwargs = None
        if phase == "field_calibrated":
            field_kwargs = {
                "real_order_weight": 1.0,
                "real_order_margin": float(getattr(config.cvoi, "field_calibration_order_margin", 0.1)),
            }
        metrics = train_cvoi_value_epoch(
            model,
            batches,
            optimizer=optimizer,
            phase=phase,
            device=device,
            tokens_per_frame=getattr(config.cvoi, "tokens_per_frame", None),
            field_loss_kwargs=field_kwargs,
            cf_field_supervision=cf_mode,
            field_warmup_domain=field_warmup_domain,
            value_updates_per_epoch=getattr(config.cvoi, "value_updates_per_epoch", None),
            protocol_version=_FORMAL_V2_NAVSIM_E120_PROTOCOL,
        )
        if phase == "field_warmup":
            _validate_field_warmup_pair_coverage(
                metrics,
                epoch=epoch,
                cf_field_supervision=cf_mode,
            )
        if phase in {"field_warmup", "field_calibrated"}:
            schedule_signature = metrics.get("eligibility_schedule_signature")
            if not isinstance(schedule_signature, str) or len(schedule_signature) != 64:
                raise ValueError(f"direct NavSim-e120 {phase} requires eligibility_schedule_signature")
        epoch_metrics.append(metrics)
    if not epoch_metrics:
        raise ValueError(f"direct NavSim-e120 {phase} requires at least one training epoch")

    field_validation_metrics = None
    if phase == "field_warmup":
        real_batches = stream_validated_field_validation_batches(
            adapter,
            domain="real",
            expected_adapter_schema=_NAVSIM_E120_ADAPTER_INPUT_SCHEMAS[phase],
        )
        counterfactual_batches = (
            None
            if cf_mode == "none"
            else stream_validated_field_validation_batches(
                adapter,
                domain="matched_real_counterfactual",
                expected_adapter_schema=_NAVSIM_E120_ADAPTER_INPUT_SCHEMAS[phase],
            )
        )
        field_validation_metrics = validate_cvoi_field_epoch(
            model,
            real_batches=real_batches,
            counterfactual_batches=counterfactual_batches,
            device=device,
            tokens_per_frame=getattr(config.cvoi, "tokens_per_frame", None),
            cf_field_supervision=cf_mode,
            protocol_version=_FORMAL_V2_NAVSIM_E120_PROTOCOL,
        )

    payload = build_cvoi_navsim_e120_direct_value_checkpoint(
        model,
        phase=phase,
        branch_id=lineage.checkpoint_branch_id(phase),
        epoch=len(epoch_metrics),
        parents=build_cvoi_manual_value_parents(lineage, phase),
    )
    checkpoint_path = atomic_torch_save_replace(payload, config.cvoi.output_checkpoint)
    return CvoiValueTrainingReport(
        phase=phase,
        checkpoint_path=checkpoint_path,
        epoch_metrics=tuple(epoch_metrics),
        field_validation_metrics=field_validation_metrics,
    )


def run_cvoi_offline_stage(
    config: Any,
    *,
    device: torch.device,
    adapter: Any = None,
    _allow_cpu_for_tests: bool = False,
) -> CvoiOfflineReport:
    """Execute one manual NavSim-e120 Value or SQLite Gate stage."""

    cvoi = _require_config_section(config, "cvoi")
    device = torch.device(device)
    if device.type != "cuda" and not _allow_cpu_for_tests:
        raise RuntimeError("CVoI artifact production requires a CUDA device")
    if getattr(cvoi, "enabled", None) is not True:
        raise ValueError("train_cvoi_offline requires cvoi.enabled=true")
    if getattr(cvoi, "protocol_version", None) != _FORMAL_V2_NAVSIM_E120_PROTOCOL:
        raise ValueError("train_cvoi_offline supports only the manual formal_v2_navsim_e120_h4_v3 protocol")

    stage = getattr(cvoi, "stage", None)
    if not isinstance(stage, str) or not stage:
        raise ValueError(f"cvoi.stage must be a non-empty string, got {stage!r}")
    if stage in _DIRECT_VALUE_STAGES:
        if adapter is None:
            raise _adapter_required(stage)
        return _train_direct_navsim_e120_value_stage(config, adapter=adapter, device=device)
    if stage in _PLANNER_STAGES:
        raise ValueError(f"cvoi.stage={stage!r} must run with --train-script train_predictor_rollout_planner")
    if stage == "evaluation":
        raise ValueError('cvoi.stage="evaluation" must run through the direct NavSim evaluation entry')
    if stage != "gate_distillation":
        raise ValueError(f"train_cvoi_offline does not recognize cvoi.stage={stage!r}")

    oracle_path, output_checkpoint, gate_branch = _preflight_direct_navsim_e120_gate_handoffs(cvoi)
    optimization = _require_config_section(config, "optimization")
    meta = _require_config_section(config, "meta")
    train_kwargs = {
        "lambda_grid": getattr(cvoi, "lambda_grid", None),
        "epochs": getattr(optimization, "epochs", None),
        "learning_rate": getattr(optimization, "lr", None),
        "weight_decay": getattr(optimization, "weight_decay", None),
        "betas": getattr(optimization, "betas", None),
        "eps": getattr(optimization, "eps", None),
        "batch_size": getattr(cvoi, "gate_training_batch_size", None),
        "hidden_dim": _OFFLINE_GATE_HIDDEN_DIM,
        "temperature": _OFFLINE_GATE_TEMPERATURE,
        "regression_weight": _OFFLINE_GATE_REGRESSION_WEIGHT,
        "seed": getattr(meta, "seed", None),
        "device": device,
        "gate_feature_mode": gate_branch.feature_mode,
    }
    expected_oracle_lineage = f"p1_{gate_branch.oracle_value_lineage}"
    with open_embedded_oracle_store_v2(Path(oracle_path)) as oracle_store:
        if oracle_store.metadata.lineage != expected_oracle_lineage:
            raise ValueError(
                "manual Gate Oracle lineage mismatch: "
                f"expected {expected_oracle_lineage!r}, got {oracle_store.metadata.lineage!r}"
            )
        return _train_cvoi_navsim_e120_official_gate_from_open_store(
            Path(oracle_path),
            oracle_store,
            output_checkpoint,
            **train_kwargs,
        )
