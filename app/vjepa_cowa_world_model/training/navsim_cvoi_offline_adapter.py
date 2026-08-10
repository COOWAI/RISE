"""Direct NavSim-e120 adapter for the retained manual CVoI training chain.

The adapter owns loader iteration and route-free quality-target construction.
Future logged geometry, JSONL Oracle jobs, audit receipts, and the legacy
two-pass evaluator are intentionally outside this retained boundary.
"""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Protocol, Sequence, runtime_checkable

import torch

from app.vjepa_cowa_world_model.training.cvoi_execution import cvoi_sample_seed, resolve_cvoi_evaluation_seed
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_quality import (
    CVOI_NAVSIM_E120_MAX_PROGRESS_M,
    NavSimE120QualitySample,
    collect_navsim_e120_local_quality_targets_direct,
    collect_navsim_e120_stop_quality_target_direct,
    score_navsim_e120_trajectory_quality,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (
    resolve_formal_v2_navsim_scene_filter_path,
    validate_formal_v2_navsim_cf_annotation_source,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (  # noqa: F401
    CvoiPlannerEvaluation,
    NavSimCvoiEncodedBatch,
    NavSimCvoiModelBatch,
    build_navsim_cvoi_model_batch,
    navsim_cvoi_raw_prefix,
    validate_navsim_cvoi_encoded_batch,
)
from app.vjepa_cowa_world_model.training.data import (
    create_train_dataloader,
    create_val_dataloader,
    create_validation_transforms,
)
from app.vjepa_cowa_world_model.training.navsim_cvoi_batch import (
    RealQualityTargetRequest,
    adapt_navsim_e120_quality_field_batch,
    require_navsim_e120_metadata_strings,
)

CVOI_NAVSIM_OFFLINE_ADAPTER_FACTORY = (
    "app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter:create_navsim_cvoi_offline_adapter"
)
CVOI_NAVSIM_E120_FIELD_CALIBRATION_SCHEMA = "navsim_e120_local_quality_calibration_v1"
CVOI_NAVSIM_E120_STOP_CALIBRATION_SCHEMA = "navsim_e120_stop_quality_v1"

_DIRECT_PROTOCOL = "formal_v2_navsim_e120_h4_v3"
_RUNTIME_METHODS = (
    "encode_batch",
    "evaluate_unguided_prefix",
    "evaluate_guided_horizon",
)


@runtime_checkable
class NavSimCvoiModelRuntime(Protocol):
    """Frozen model operations required by direct e120 Value stages."""

    embed_dim: int

    def encode_batch(self, batch: NavSimCvoiModelBatch) -> NavSimCvoiEncodedBatch:
        raise NotImplementedError

    def evaluate_unguided_prefix(
        self,
        *,
        context: object,
        z_observed: torch.Tensor,
        prefix: torch.Tensor,
        horizon: int,
        seed: int,
    ) -> CvoiPlannerEvaluation:
        raise NotImplementedError

    def evaluate_guided_horizon(
        self,
        *,
        context: object,
        z_observed: torch.Tensor,
        raw_prefix: torch.Tensor,
        horizon: int,
        apply_guidance: bool,
        seed: int,
        guidance_steps: Optional[int] = None,
    ) -> CvoiPlannerEvaluation:
        raise NotImplementedError


def _prefix_count(z_future: torch.Tensor, *, tokens_per_frame: Optional[int]) -> int:
    if z_future.ndim == 4:
        return int(z_future.shape[1])
    if z_future.ndim != 3:
        raise ValueError(f"CVoI runtime z_future must be [B,N,D] or [B,F,T,D], got {tuple(z_future.shape)}")
    if type(tokens_per_frame) is not int or tokens_per_frame <= 0:
        raise ValueError("CVoI flat runtime z_future requires positive cvoi.tokens_per_frame")
    if int(z_future.shape[1]) % tokens_per_frame:
        raise ValueError("cvoi.tokens_per_frame must divide runtime z_future token count")
    return int(z_future.shape[1]) // tokens_per_frame


def _future_as_frames(z_future: torch.Tensor, *, tokens_per_frame: Optional[int]) -> torch.Tensor:
    if z_future.ndim == 4:
        return z_future
    prefix_count = _prefix_count(z_future, tokens_per_frame=tokens_per_frame)
    return z_future.reshape(z_future.shape[0], prefix_count, int(tokens_per_frame), z_future.shape[-1])


def _build_navsim_e120_quality_samples(
    navsim_batch: Sequence[object],
    *,
    config: Any,
) -> dict[int, NavSimE120QualitySample]:
    metadata = navsim_batch[11]
    if not isinstance(metadata, Mapping):
        raise ValueError("NavSim-e120 quality sample construction requires metadata")
    domains_value = metadata.get("dataset_domain")
    if isinstance(domains_value, (str, bytes)) or not isinstance(domains_value, Sequence):
        raise ValueError("NavSim-e120 quality dataset_domain must be a sequence")
    batch_size = len(domains_value)
    if batch_size < 1:
        raise ValueError("NavSim-e120 quality batch cannot be empty")
    domains = require_navsim_e120_metadata_strings(metadata, "dataset_domain", batch_size=batch_size)
    sample_ids = require_navsim_e120_metadata_strings(metadata, "stable_sample_id", batch_size=batch_size)
    scene_ids = require_navsim_e120_metadata_strings(metadata, "base_scene_id", batch_size=batch_size)
    evaluation_seed = resolve_cvoi_evaluation_seed(config)
    samples = {}
    for index, domain in enumerate(domains):
        if domain == "counterfactual":
            continue
        if domain != "real":
            raise ValueError(f"NavSim-e120 quality batch contains unknown domain {domain!r}")
        sample_id = sample_ids[index]
        samples[index] = NavSimE120QualitySample(
            sample_id=sample_id,
            source_scene_id=scene_ids[index],
            seed=cvoi_sample_seed(evaluation_seed, sample_id),
        )
    return samples


def _validate_planner_evaluation(
    result: object,
    *,
    expected_guidance_steps: int,
) -> CvoiPlannerEvaluation:
    if not isinstance(result, CvoiPlannerEvaluation):
        raise TypeError("NavSim CVoI planner runtime must return CvoiPlannerEvaluation")
    if type(result.guidance_steps) is not int or result.guidance_steps != expected_guidance_steps:
        raise ValueError(
            f"NavSim CVoI planner runtime must report guidance_steps={expected_guidance_steps}, "
            f"got {result.guidance_steps!r}"
        )
    latency_ms = float(result.latency_ms)
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise ValueError("NavSim CVoI planner runtime latency_ms must be finite and non-negative")
    return result


def _validate_controller_lineage(value: object) -> str:
    if value not in {"value_guided", "p0_controller"}:
        raise ValueError("cvoi.controller_lineage must be 'value_guided' or 'p0_controller'")
    return str(value)


class NavSimCvoiOfflineAdapter:
    """Execute the retained direct e120 Value stages over NavSim loaders."""

    def __init__(
        self,
        *,
        config: Any,
        device: torch.device,
        loader: Sequence[Sequence[object]],
        field_validation_loaders: Optional[Mapping[str, Sequence[Sequence[object]]]] = None,
        runtime: NavSimCvoiModelRuntime,
    ) -> None:
        if getattr(config.cvoi, "protocol_version", None) != _DIRECT_PROTOCOL:
            raise ValueError(f"NavSim CVoI offline adapter supports only {_DIRECT_PROTOCOL}")
        self.config = config
        self.device = torch.device(device)
        self.loader = loader
        if field_validation_loaders is None:
            normalized_validation_loaders: dict[str, Sequence[Sequence[object]]] = {}
        else:
            if not isinstance(field_validation_loaders, Mapping):
                raise TypeError("field_validation_loaders must be a mapping")
            unknown_domains = set(field_validation_loaders) - {
                "real",
                "counterfactual",
                "matched_real_counterfactual",
            }
            if unknown_domains:
                raise ValueError(f"unknown Field validation loader domains: {sorted(unknown_domains)}")
            normalized_validation_loaders = dict(field_validation_loaders)
        self.field_validation_loaders = normalized_validation_loaders
        self.runtime = runtime
        self.controller_lineage = _validate_controller_lineage(
            getattr(config.cvoi, "controller_lineage", "value_guided")
        )
        if str(config.cvoi.stage) == "stop_calibrated" and self.controller_lineage != "value_guided":
            raise ValueError("direct NavSim-e120 Stop requires controller_lineage='value_guided'")
        self.embed_dim = getattr(runtime, "embed_dim", None)
        if type(self.embed_dim) is not int or self.embed_dim <= 0:
            raise ValueError("NavSim CVoI runtime must declare positive integer embed_dim")
        missing_methods = [name for name in _RUNTIME_METHODS if not callable(getattr(runtime, name, None))]
        if missing_methods:
            raise ValueError(f"NavSim CVoI runtime is missing required methods: {missing_methods}")
        if (
            str(config.cvoi.stage) == "stop_calibrated"
            and self.controller_lineage == "value_guided"
            and not callable(getattr(runtime, "encode_p1_batch", None))
        ):
            raise ValueError("NavSim-e120 P1 Stop stage requires runtime.encode_p1_batch for the P1 policy pair")

    @staticmethod
    def _attach_batch_provenance(
        batch: Mapping[str, object],
        *,
        provenance: Optional[Mapping[str, str]],
    ) -> dict[str, object]:
        if provenance is not None:
            raise ValueError("direct NavSim-e120 adapter requires provenance=None")
        normalized = dict(batch)
        if "cvoi_provenance" in normalized:
            raise ValueError("direct NavSim-e120 adapter batches must omit cvoi_provenance")
        return normalized

    def _encode(self, navsim_batch: Sequence[object]) -> NavSimCvoiEncodedBatch:
        states = navsim_batch[2]
        if not isinstance(states, torch.Tensor) or states.ndim < 1:
            raise ValueError("NavSim-e120 batch states must be a tensor")
        batch_size = int(states.shape[0])
        uses_p1_pair = str(self.config.cvoi.stage) == "stop_calibrated" and self.controller_lineage == "value_guided"
        encode = self.runtime.encode_p1_batch if uses_p1_pair else self.runtime.encode_batch
        with torch.no_grad():
            encoded = encode(
                build_navsim_cvoi_model_batch(
                    navsim_batch,
                    config=self.config,
                    device=self.device,
                )
            )
        return validate_navsim_cvoi_encoded_batch(
            encoded,
            batch_size=batch_size,
            embed_dim=self.embed_dim,
            max_horizon=int(self.config.cvoi.max_horizon),
            tokens_per_frame=getattr(self.config.cvoi, "tokens_per_frame", None),
        )

    def _navsim_e120_warmup_targets(
        self,
        request: RealQualityTargetRequest,
        *,
        samples: Mapping[int, NavSimE120QualitySample],
        encoded: NavSimCvoiEncodedBatch,
    ) -> torch.Tensor:
        future = _future_as_frames(
            request.z_future,
            tokens_per_frame=getattr(self.config.cvoi, "tokens_per_frame", None),
        )
        targets = torch.empty(
            (len(request.sample_ids), int(self.config.cvoi.max_horizon)),
            dtype=torch.float32,
            device=request.z_future.device,
        )
        timestep_sec = 1.0 / float(self.config.data.fps)
        for local_index, batch_index in enumerate(request.batch_indices.tolist()):
            sample = samples[batch_index]
            for horizon in range(1, int(self.config.cvoi.max_horizon) + 1):
                with torch.no_grad():
                    result = self.runtime.evaluate_unguided_prefix(
                        context=encoded.model_contexts[batch_index],
                        z_observed=request.z_observed[local_index : local_index + 1],
                        prefix=future[local_index : local_index + 1, :horizon],
                        horizon=horizon,
                        seed=sample.seed,
                    )
                    result = _validate_planner_evaluation(result, expected_guidance_steps=0)
                    quality = score_navsim_e120_trajectory_quality(
                        result.pred_trajs,
                        result.confidences,
                        timestep_sec=timestep_sec,
                        max_progress_m=CVOI_NAVSIM_E120_MAX_PROGRESS_M,
                    ).quality
                targets[local_index, horizon - 1] = quality[0].to(device=targets.device, dtype=targets.dtype)
        return targets

    def _field_warmup_batch(
        self,
        navsim_batch: Sequence[object],
        *,
        encoded: NavSimCvoiEncodedBatch,
        provenance: Optional[Mapping[str, str]],
    ) -> Mapping[str, object]:
        ablation_signature = getattr(self.config.cvoi, "ablation_signature", None)
        cf_field_supervision = (
            "hazard_quality" if ablation_signature is None else str(ablation_signature.cf_field_supervision)
        )
        quality_samples = _build_navsim_e120_quality_samples(navsim_batch, config=self.config)
        adapted = adapt_navsim_e120_quality_field_batch(
            navsim_batch,
            z_observed=encoded.z_observed,
            z_future=encoded.z_future,
            real_quality_target_provider=lambda request: self._navsim_e120_warmup_targets(
                request,
                samples=quality_samples,
                encoded=encoded,
            ),
            tokens_per_frame=getattr(self.config.cvoi, "tokens_per_frame", None),
            cf_field_supervision=cf_field_supervision,
        )
        return self._attach_batch_provenance(adapted, provenance=provenance)

    def _navsim_e120_field_calibration_batches(
        self,
        navsim_batch: Sequence[object],
        *,
        encoded: NavSimCvoiEncodedBatch,
        provenance: Optional[Mapping[str, str]],
    ) -> Iterator[Mapping[str, object]]:
        samples = _build_navsim_e120_quality_samples(navsim_batch, config=self.config)
        future = _future_as_frames(
            encoded.z_future,
            tokens_per_frame=getattr(self.config.cvoi, "tokens_per_frame", None),
        )
        timestep_sec = 1.0 / float(self.config.data.fps)
        for batch_index, sample in samples.items():
            z_observed = encoded.z_observed[batch_index : batch_index + 1]

            def evaluate_prefix(prefix: torch.Tensor, horizon: int, seed: int) -> CvoiPlannerEvaluation:
                result = self.runtime.evaluate_unguided_prefix(
                    context=encoded.model_contexts[batch_index],
                    z_observed=z_observed,
                    prefix=prefix,
                    horizon=horizon,
                    seed=seed,
                )
                return _validate_planner_evaluation(result, expected_guidance_steps=0)

            ablation_signature = getattr(self.config.cvoi, "ablation_signature", None)
            calibration_mode = (
                "local_geometry" if ablation_signature is None else str(ablation_signature.field_calibration_mode)
            )
            targets = collect_navsim_e120_local_quality_targets_direct(
                sample,
                base_future_latent=future[batch_index : batch_index + 1],
                num_perturbations=int(self.config.cvoi.field_calibration_num_perturbations),
                perturbation_scale=float(self.config.cvoi.field_calibration_perturbation_scale),
                max_delta_norm=float(self.config.cvoi.field_calibration_max_delta_norm),
                evaluate_prefix=evaluate_prefix,
                timestep_sec=timestep_sec,
                max_progress_m=CVOI_NAVSIM_E120_MAX_PROGRESS_M,
                calibration_mode=calibration_mode,
            )
            candidate_count = int(targets.candidate_latents.shape[0])
            yield self._attach_batch_provenance(
                {
                    "adapter_schema": CVOI_NAVSIM_E120_FIELD_CALIBRATION_SCHEMA,
                    "z_observed": z_observed.expand(candidate_count, *z_observed.shape[1:]).clone(),
                    "z_future": targets.candidate_latents,
                    "dataset_domains": ["real"] * candidate_count,
                    "real_quality_targets": targets.quality_targets,
                    "real_group_ids": list(targets.group_ids),
                    "stable_sample_ids": [sample.sample_id] * candidate_count,
                },
                provenance=provenance,
            )

    def _navsim_e120_stop_calibration_batches(
        self,
        navsim_batch: Sequence[object],
        *,
        encoded: NavSimCvoiEncodedBatch,
        provenance: Optional[Mapping[str, str]],
    ) -> Iterator[Mapping[str, object]]:
        samples = _build_navsim_e120_quality_samples(navsim_batch, config=self.config)
        timestep_sec = 1.0 / float(self.config.data.fps)
        for batch_index, sample in samples.items():
            z_observed = encoded.z_observed[batch_index : batch_index + 1]
            z_future = encoded.z_future[batch_index : batch_index + 1]

            def evaluate_horizon(horizon: int, apply_guidance: bool, seed: int) -> CvoiPlannerEvaluation:
                raw_prefix = navsim_cvoi_raw_prefix(
                    z_future,
                    horizon,
                    tokens_per_frame=getattr(self.config.cvoi, "tokens_per_frame", None),
                )
                if self.controller_lineage == "p0_controller":
                    if apply_guidance:
                        raise ValueError("P0 Controller NavSim-e120 Stop calibration forbids Guidance")
                    result = self.runtime.evaluate_unguided_prefix(
                        context=encoded.model_contexts[batch_index],
                        z_observed=z_observed,
                        prefix=raw_prefix,
                        horizon=horizon,
                        seed=seed,
                    )
                else:
                    result = self.runtime.evaluate_guided_horizon(
                        context=encoded.model_contexts[batch_index],
                        z_observed=z_observed,
                        raw_prefix=raw_prefix,
                        horizon=horizon,
                        apply_guidance=apply_guidance,
                        seed=seed,
                    )
                return _validate_planner_evaluation(
                    result,
                    expected_guidance_steps=2 if apply_guidance else 0,
                )

            targets = collect_navsim_e120_stop_quality_target_direct(
                sample,
                max_horizon=int(self.config.cvoi.max_horizon),
                evaluate_horizon=evaluate_horizon,
                timestep_sec=timestep_sec,
                max_progress_m=CVOI_NAVSIM_E120_MAX_PROGRESS_M,
                controller_lineage=self.controller_lineage,
            )
            yield self._attach_batch_provenance(
                {
                    "adapter_schema": CVOI_NAVSIM_E120_STOP_CALIBRATION_SCHEMA,
                    "z_observed": z_observed,
                    "z_future": z_future,
                    "dataset_domains": ["real"],
                    "stop_quality_targets": targets.quality_targets,
                },
                provenance=provenance,
            )

    def value_batches(
        self,
        stage: str,
        epoch: int,
        *,
        provenance: Optional[Mapping[str, str]],
    ) -> Iterator[Mapping[str, object]]:
        """Yield one direct e120 Value lifecycle epoch."""

        del epoch
        if stage not in {"field_warmup", "field_calibrated", "stop_calibrated"}:
            raise ValueError(f"NavSim CVoI adapter does not produce Value batches for stage={stage!r}")
        configured_stage = getattr(self.config.cvoi, "stage", None)
        if type(configured_stage) is not str or stage != configured_stage:
            raise ValueError(
                f"direct NavSim-e120 Value batch stage={stage!r} must match configured "
                f"cvoi.stage={configured_stage!r}"
            )
        for navsim_batch in self.loader:
            encoded = self._encode(navsim_batch)
            if stage == "field_warmup":
                yield self._field_warmup_batch(navsim_batch, encoded=encoded, provenance=provenance)
            elif stage == "field_calibrated":
                yield from self._navsim_e120_field_calibration_batches(
                    navsim_batch,
                    encoded=encoded,
                    provenance=provenance,
                )
            else:
                yield from self._navsim_e120_stop_calibration_batches(
                    navsim_batch,
                    encoded=encoded,
                    provenance=provenance,
                )

    def field_validation_batches(
        self,
        *,
        domain: str,
        provenance: Optional[Mapping[str, str]],
    ) -> Iterator[Mapping[str, object]]:
        """Yield one explicit direct e120 Field validation scope."""

        if domain not in {"real", "counterfactual", "matched_real_counterfactual"}:
            raise ValueError(
                "Field validation scope must be real, counterfactual, or matched_real_counterfactual, "
                f"got {domain!r}"
            )
        loader = self.field_validation_loaders.get(domain)
        if loader is None:
            raise ValueError(f"{domain} Field validation loader is not configured")
        for navsim_batch in loader:
            encoded = self._encode(navsim_batch)
            yield self._field_warmup_batch(navsim_batch, encoded=encoded, provenance=provenance)


def load_navsim_cvoi_runtime(config: Any, *, device: torch.device) -> NavSimCvoiModelRuntime:
    """Load the explicit frozen-model runtime used by the built-in adapter."""

    factory_path = getattr(config.cvoi, "offline_runtime_factory", None)
    if not isinstance(factory_path, str) or not factory_path.strip() or ":" not in factory_path:
        raise ValueError("built-in NavSim CVoI adapter requires cvoi.offline_runtime_factory='module:factory'")
    module_name, attribute_name = factory_path.rsplit(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("cvoi.offline_runtime_factory must use the exact 'module:factory' form")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise ValueError(f"cvoi.offline_runtime_factory is not callable: {factory_path!r}")
    runtime = factory(config=config, device=torch.device(device))
    if runtime is None:
        raise ValueError(f"cvoi.offline_runtime_factory returned None: {factory_path!r}")
    return runtime


def _root_value(root: object, field_name: str) -> object:
    if isinstance(root, Mapping):
        return root.get(field_name)
    return getattr(root, field_name, None)


def _require_direct_root_path(
    root: object,
    *,
    field_name: str,
    root_name: str,
    kind: str,
) -> Path:
    value = _root_value(root, field_name)
    if type(value) is not str or not value.strip():
        raise ValueError(f"direct NavSim root {root_name!r} requires non-empty {field_name}")
    path = resolve_formal_v2_navsim_scene_filter_path(value) if field_name == "scene_filter_yaml" else Path(value)
    if not path.is_absolute():
        raise ValueError(f"direct NavSim root {root_name!r} {field_name} must be absolute: {path}")
    if path.is_symlink():
        raise ValueError(f"direct NavSim root {root_name!r} {field_name} must not be a symlink: {path}")
    if kind == "directory":
        if not path.is_dir():
            raise FileNotFoundError(f"direct NavSim root {root_name!r} {field_name} directory does not exist: {path}")
    elif kind == "file":
        if not path.is_file():
            raise FileNotFoundError(f"direct NavSim root {root_name!r} {field_name} file does not exist: {path}")
    else:
        raise ValueError(f"unsupported direct NavSim root path kind: {kind!r}")
    return path


def _validate_direct_navsim_roots(config: Any) -> None:
    roots = (*tuple(config.data.navsim.train_roots), *tuple(config.data.navsim.val_roots))
    if not roots:
        raise ValueError("direct NavSim adapter requires explicit train_roots or val_roots")
    for index, root in enumerate(roots):
        raw_name = _root_value(root, "name")
        root_name = raw_name if isinstance(raw_name, str) and raw_name else f"root[{index}]"
        for field_name in ("data_path", "sensor_blobs_path"):
            _require_direct_root_path(
                root,
                field_name=field_name,
                root_name=root_name,
                kind="directory",
            )
        if _root_value(root, "pose_overlay_path") is not None:
            _require_direct_root_path(
                root,
                field_name="pose_overlay_path",
                root_name=root_name,
                kind="directory",
            )
        for field_name in ("scene_filter_yaml", "annotations_path", "trajectory_quality_path"):
            if _root_value(root, field_name) is not None:
                _require_direct_root_path(
                    root,
                    field_name=field_name,
                    root_name=root_name,
                    kind="file",
                )
        if _root_value(root, "domain") == "counterfactual":
            annotations_path = _require_direct_root_path(
                root,
                field_name="annotations_path",
                root_name=root_name,
                kind="file",
            )
            validate_formal_v2_navsim_cf_annotation_source(
                annotations_path,
                root_name=root_name,
            )


def _create_navsim_cvoi_adapter_inputs(
    config: Any,
    *,
    device: torch.device,
) -> tuple[
    NavSimCvoiModelRuntime,
    Sequence[Sequence[object]],
    dict[str, Sequence[Sequence[object]]],
]:
    runtime = load_navsim_cvoi_runtime(config, device=device)
    validation_transform = create_validation_transforms(config)
    loader, _ = create_train_dataloader(config, rank=0, world_size=1, transform=validation_transform)
    field_validation_loaders: dict[str, Sequence[Sequence[object]]] = {}
    if getattr(config.cvoi, "stage", None) == "field_warmup":
        field_warmup_domain = getattr(config.cvoi, "field_warmup_domain", None)
        if field_warmup_domain == "real":
            validation_domains = ("real",)
        elif field_warmup_domain == "real_cf":
            ablation_signature = getattr(config.cvoi, "ablation_signature", None)
            if ablation_signature is None:
                raise ValueError("NavSim-e120 Field requires cvoi.ablation_signature")
            cf_field_supervision = str(ablation_signature.cf_field_supervision)
            if cf_field_supervision in {"hazard_only", "hazard_quality"}:
                validation_domains = ("real", "matched_real_counterfactual")
            elif cf_field_supervision == "quality_only":
                validation_domains = ("real", "counterfactual")
            else:
                raise ValueError(
                    "NavSim-e120 real_cf Field requires hazard_only, quality_only, or hazard_quality supervision"
                )
        else:
            raise ValueError("NavSim-e120 Field requires cvoi.field_warmup_domain='real' or 'real_cf'")
        for validation_domain in validation_domains:
            validation_loader, _ = create_val_dataloader(
                config,
                rank=0,
                world_size=1,
                transform=validation_transform,
                validation_domain=validation_domain,
            )
            if validation_loader is None:
                raise ValueError(f"NavSim-e120 Field requires a non-empty {validation_domain} validation loader")
            field_validation_loaders[validation_domain] = validation_loader
    return runtime, loader, field_validation_loaders


def create_navsim_cvoi_offline_adapter(*, config: Any, device: torch.device) -> NavSimCvoiOfflineAdapter:
    """Build the direct e120 adapter configured by ``cvoi.offline_adapter_factory``."""

    if getattr(config.cvoi, "protocol_version", None) != _DIRECT_PROTOCOL:
        raise ValueError(f"built-in NavSim CVoI offline adapter supports only {_DIRECT_PROTOCOL}")
    device = torch.device(device)
    _validate_direct_navsim_roots(config)
    runtime, loader, field_validation_loaders = _create_navsim_cvoi_adapter_inputs(config, device=device)
    return NavSimCvoiOfflineAdapter(
        config=config,
        device=device,
        loader=loader,
        field_validation_loaders=field_validation_loaders,
        runtime=runtime,
    )
