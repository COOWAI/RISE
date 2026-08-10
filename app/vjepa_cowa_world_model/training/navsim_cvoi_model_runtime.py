"""Frozen NavSim model runtime for offline CVoI target collection.

This module is the deployment-model side of navsim_cvoi_offline_adapter.
It never receives geometry, ground-truth trajectories, or counterfactual
labels. Predictor side inputs beyond the observed prefix are explicit zeros;
they exist only to preserve the checkpoint's training-time tensor shapes.
"""

from __future__ import annotations

import copy
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch

from app.vjepa_cowa_world_model.models.multiview_fusion import PETRMultiViewFusion
from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output
from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training.config import (
    resolve_main_encoder_raw_tokens_per_frame,
    resolve_planner_use_observed_tokens,
)
from app.vjepa_cowa_world_model.training.configs.common import resolve_predictor_runtime_normalize_reps
from app.vjepa_cowa_world_model.training.cvoi_audit import load_cvoi_audit_manifest
from app.vjepa_cowa_world_model.training.cvoi_execution import (
    CvoiValueDtypeAdapter,
    common_random_numbers,
    cvoi_execution_autocast,
    cvoi_execution_dtype_signature,
    cvoi_inference_rng_signature,
    cvoi_planner_inference_noise,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_full_state_warmstart import (
    apply_formal_v2_full_state_warmstart_direct,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_runtime import (
    read_formal_v2_navsim_e120_direct_checkpoint,
    resolve_formal_v2_navsim_e120_selected_checkpoint,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_MAX_HORIZON,
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)
from app.vjepa_cowa_world_model.training.cvoi_manual_lineage import resolve_cvoi_manual_full_results_root
from app.vjepa_cowa_world_model.training.cvoi_runtime import (
    _cvoi_multiview_signature,
    _cvoi_planner_signature,
    _cvoi_world_execution_signature,
    apply_cvoi_planner_guidance,
    load_cvoi_dual_value_model,
    validate_cvoi_planner_lineage,
)
from app.vjepa_cowa_world_model.training.cvoi_world4drive_runtime import (
    WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS,
    CvoiPlannerEvaluation,
    CvoiWorld4DriveRuntimeBinding,
    load_cvoi_world4drive_value_model,
    read_world4drive_value_checkpoint,
    validate_cvoi_world4drive_gate,
    validate_world4drive_value_lineage,
)
from app.vjepa_cowa_world_model.training.model_factories.encoder import (
    get_encoder_embed_dim,
    init_encoder,
    init_encoder_for_full_state_warmstart,
)
from app.vjepa_cowa_world_model.training.model_factories.planner import init_planner
from app.vjepa_cowa_world_model.training.model_factories.predictor import (
    init_predictor_runtime_with_token_ae,
    resolve_main_predictor_runtime_overrides,
)
from app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter import (
    NavSimCvoiEncodedBatch,
    NavSimCvoiModelBatch,
)
from app.vjepa_cowa_world_model.training.planner_anchor import build_ego_relative_diffusion_anchor
from app.vjepa_cowa_world_model.training.predictor_stepping import (
    make_predictor_step_fn,
    rollout_latent_predictions,
    validate_empty_future_planner_conditions,
)
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (
    build_predictor_timeline_inputs,
    enforce_cvoi_zero_future_aux,
    forward_main_context,
)
from app.vjepa_cowa_world_model.training.sequential_budget_control import extract_prefix_gate_values
from app.vjepa_cowa_world_model.utils import (
    build_observed_action_trajectory_history,
    prepare_inference_consistent_status_vector,
    resolve_planner_status_dim,
    resolve_planner_use_drive_command,
)

CVOI_NAVSIM_MODEL_RUNTIME_FACTORY = (
    "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime:create_navsim_cvoi_model_runtime"
)

_ONLINE_METADATA_KEYS = frozenset(
    {
        "camera_names",
        "camera_intrinsics",
        "camera2ego",
        "metadata_valid_mask",
        "observed_metadata_valid_mask",
    }
)
_P1_STAGES = frozenset({"stop_calibrated", "evaluation"})
_NAVSIM_E120_PROTOCOL = "formal_v2_navsim_e120_h4_v3"
_NAVSIM_E120_DIRECT_BRANCH_BY_STAGE = {
    "p0": "p0_uniform",
    "p1": "p1_full",
}
_NAVSIM_E120_DIRECT_CANDIDATE_EPOCHS = {
    "p0": FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    "p1": FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
}


@dataclass(frozen=True)
class _World4DriveAblationIdentity:
    """Minimal private identity required by retained inference/provenance helpers."""

    schema: str
    experiment_role: str
    branch_id: str
    shared_cohort_id: str
    cf_field_supervision: str
    field_calibration_mode: str
    p0_prefix_mode: str
    gate_feature_mode: str
    train_seed: int
    evaluation_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "experiment_role": self.experiment_role,
            "branch_id": self.branch_id,
            "shared_cohort_id": self.shared_cohort_id,
            "cf_field_supervision": self.cf_field_supervision,
            "field_calibration_mode": self.field_calibration_mode,
            "p0_prefix_mode": self.p0_prefix_mode,
            "gate_feature_mode": self.gate_feature_mode,
            "train_seed": self.train_seed,
            "evaluation_seed": self.evaluation_seed,
        }


@dataclass(frozen=True)
class NavSimCvoiPlannerContext:
    """Per-sample, online-only conditions shared by P0 and P1."""

    status_feature: torch.Tensor
    z_first_frame: Optional[torch.Tensor]
    z_observed_for_planner: Optional[torch.Tensor]
    action_history: Optional[torch.Tensor]
    anchor_state: Optional[torch.Tensor]
    rollout_latency_ms_by_horizon: tuple[float, ...]


@dataclass(frozen=True)
class _NavSimCvoiOnlineModelContext:
    """Static planner conditions for one incremental batch-one call."""

    status_feature: torch.Tensor
    z_first_frame: Optional[torch.Tensor]
    z_observed_for_planner: Optional[torch.Tensor]
    action_history: Optional[torch.Tensor]
    anchor_state: Optional[torch.Tensor]


@dataclass(frozen=True)
class NavSimCvoiOnlineSession:
    """Observed encoding and step Predictor inputs for one online call."""

    z_observed: torch.Tensor
    model_context: object
    predictor_inputs: object
    step_predictor: Callable[..., torch.Tensor]
    policy: str


def _freeze_module(module: Optional[torch.nn.Module]) -> None:
    if module is None:
        return
    module.requires_grad_(False)
    module.eval()


def _time_tensor_operation(reference: torch.Tensor, operation: Callable[[], object]) -> tuple[object, float]:
    """Measure one adaptive-compute operation with correct CUDA synchronization."""

    if reference.is_cuda:
        torch.cuda.synchronize(reference.device)
    start = time.perf_counter()
    result = operation()
    if reference.is_cuda:
        torch.cuda.synchronize(reference.device)
    latency_ms = (time.perf_counter() - start) * 1000.0
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise RuntimeError(f"CVoI adaptive-compute latency must be finite and non-negative, got {latency_ms}")
    return result, latency_ms


def _normalize_state_dict(state: object, *, checkpoint_name: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{checkpoint_name} must contain a non-empty planner state_dict")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(f"{checkpoint_name} planner state_dict must map string keys to tensors")
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def _direct_policy_module_core(module: torch.nn.Module, *, role: str) -> torch.nn.Module:
    core = module.module if hasattr(module, "module") else module
    if not isinstance(core, torch.nn.Module):
        raise ValueError(f"direct NavSim-e120 {role} target must be a torch.nn.Module")
    return core


def _normalize_direct_policy_state(state: object, *, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{name} must be a non-empty tensor mapping")
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, tensor in state.items():
        if not isinstance(raw_key, str) or not raw_key or not torch.is_tensor(tensor):
            raise ValueError(f"{name} must map non-empty string keys to tensors")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not key or key in normalized:
            raise ValueError(f"{name} contains duplicate or empty normalized key {key!r}")
        normalized[key] = tensor
    return {key: normalized[key] for key in sorted(normalized)}


def _prepare_direct_policy_role_state(
    payload: Mapping[str, object],
    *,
    module: torch.nn.Module,
    role: str,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    core = _direct_policy_module_core(module, role=role)
    checkpoint_state = _normalize_direct_policy_state(
        payload.get(role),
        name=f"direct NavSim-e120 checkpoint {role} state",
    )
    target_state = _normalize_direct_policy_state(
        core.state_dict(),
        name=f"direct NavSim-e120 target {role} state",
    )
    checkpoint_keys = set(checkpoint_state)
    target_keys = set(target_state)
    if checkpoint_keys != target_keys:
        raise ValueError(
            f"direct NavSim-e120 {role} state keys mismatch: "
            f"missing={sorted(target_keys - checkpoint_keys)}, "
            f"unexpected={sorted(checkpoint_keys - target_keys)}"
        )
    checkpoint_shapes = {key: list(checkpoint_state[key].shape) for key in sorted(checkpoint_state)}
    target_shapes = {key: list(target_state[key].shape) for key in sorted(target_state)}
    if checkpoint_shapes != target_shapes:
        raise ValueError(
            f"direct NavSim-e120 {role} state shapes mismatch: "
            f"expected={target_shapes!r}, got={checkpoint_shapes!r}"
        )
    role_state_shapes = payload.get("role_state_shapes")
    if not isinstance(role_state_shapes, Mapping) or role_state_shapes.get(role) != target_shapes:
        raise ValueError(f"direct NavSim-e120 {role} role_state_shapes do not match the target module")
    return core, checkpoint_state, target_state


def _restore_navsim_e120_direct_policy_checkpoint(
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    planner: torch.nn.Module,
    path: str | Path,
    expected_stage: str,
    results_root: str | Path,
) -> dict[str, object]:
    """Restore one proof-free selected Predictor+Planner pair."""

    if expected_stage not in _NAVSIM_E120_DIRECT_BRANCH_BY_STAGE:
        raise ValueError(f"unsupported direct NavSim-e120 policy stage {expected_stage!r}")
    with torch.random.fork_rng(devices=[]):
        resolved_path = resolve_formal_v2_navsim_e120_selected_checkpoint(
            path,
            results_root=results_root,
            stage=expected_stage,
        )
        payload = read_formal_v2_navsim_e120_direct_checkpoint(resolved_path)
        if payload["stage"] != expected_stage:
            raise ValueError(
                f"direct NavSim-e120 policy stage mismatch: "
                f"expected={expected_stage!r}, actual={payload['stage']!r}"
            )
        expected_branch = _NAVSIM_E120_DIRECT_BRANCH_BY_STAGE[expected_stage]
        lineage = payload["lineage"]
        if not isinstance(lineage, Mapping) or lineage.get("branch_id") != expected_branch:
            raise ValueError(f"direct NavSim-e120 {expected_stage.upper()} branch must be {expected_branch!r}")
        candidates = _NAVSIM_E120_DIRECT_CANDIDATE_EPOCHS[expected_stage]
        if payload["epoch"] not in candidates:
            raise ValueError(
                f"direct NavSim-e120 {expected_stage.upper()} selected epoch must be one of {candidates!r}, "
                f"got {payload['epoch']!r}"
            )

        encoder_core, checkpoint_encoder, warmstart_encoder = _prepare_direct_policy_role_state(
            payload,
            module=encoder,
            role="encoder",
        )
        predictor_core, checkpoint_predictor, _ = _prepare_direct_policy_role_state(
            payload,
            module=predictor,
            role="predictor",
        )
        planner_core, checkpoint_planner, _ = _prepare_direct_policy_role_state(
            payload,
            module=planner,
            role="planner",
        )
        del encoder_core
        for key in sorted(checkpoint_encoder):
            checkpoint_tensor = checkpoint_encoder[key].detach().cpu()
            warmstart_tensor = warmstart_encoder[key].detach().cpu()
            if checkpoint_tensor.dtype != warmstart_tensor.dtype or not torch.equal(
                checkpoint_tensor,
                warmstart_tensor,
            ):
                raise ValueError(
                    f"direct NavSim-e120 {expected_stage.upper()} encoder tensor {key!r} "
                    "does not match the strict e120 warm-start"
                )
        predictor_core.load_state_dict(checkpoint_predictor, strict=True)
        planner_core.load_state_dict(checkpoint_planner, strict=True)

    return {
        "checkpoint_path": str(Path(path)),
        "resolved_checkpoint_path": str(resolved_path),
        "stage": expected_stage,
        "branch_id": expected_branch,
        "epoch": payload["epoch"],
    }


def _time_length(value: torch.Tensor, *, video: bool = False) -> int:
    if video:
        if value.ndim == 5:
            return int(value.shape[2])
        if value.ndim == 6:
            return int(value.shape[3])
        raise ValueError("NavSim CVoI context_frames must be [B,C,T,H,W] or [B,V,C,T,H,W]")
    if value.ndim < 2:
        raise ValueError("NavSim CVoI timeline tensors must include batch and time dimensions")
    return int(value.shape[1])


def _require_finite_tensor(
    value: object,
    *,
    name: str,
    batch_size: int,
    time_length: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"NavSim CVoI {name} must be a tensor")
    if value.ndim < 2 or int(value.shape[0]) != batch_size or int(value.shape[1]) != time_length:
        raise ValueError(
            f"NavSim CVoI {name} must have batch={batch_size}, time={time_length}; got {tuple(value.shape)}"
        )
    if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"NavSim CVoI {name} must be finite floating point")
    return value


def _zero_extend(value: torch.Tensor, *, total_length: int, name: str) -> torch.Tensor:
    current = int(value.shape[1])
    if current > total_length:
        raise ValueError(f"NavSim CVoI {name} observed length {current} exceeds target timeline {total_length}")
    if current == total_length:
        return value
    padding = value.new_zeros((value.shape[0], total_length - current, *value.shape[2:]))
    return torch.cat([value, padding], dim=1)


def _zero_extend_mask(
    value: object,
    *,
    batch_size: int,
    observed: int,
    total: int,
    name: str,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or value.shape[:2] != (batch_size, observed):
        raise ValueError(f"NavSim CVoI metadata[{name!r}] must start with [B={batch_size}, T={observed}]")
    if value.dtype != torch.bool:
        raise ValueError(f"NavSim CVoI metadata[{name!r}] must be boolean")
    return _zero_extend(value, total_length=total, name=f"metadata[{name!r}]")


def _flatten_latent(value: torch.Tensor, *, tokens_per_frame: int, name: str) -> torch.Tensor:
    if value.ndim == 3:
        if int(value.shape[1]) % tokens_per_frame:
            raise ValueError(f"NavSim CVoI {name} token length must be divisible by {tokens_per_frame}")
        return value
    if value.ndim != 4 or int(value.shape[2]) != tokens_per_frame:
        raise ValueError(f"NavSim CVoI {name} must be [B,N,D] or [B,F,{tokens_per_frame},D], got {tuple(value.shape)}")
    return value.flatten(1, 2)


def _prefix_frames(value: torch.Tensor, *, tokens_per_frame: int, name: str) -> int:
    flat = _flatten_latent(value, tokens_per_frame=tokens_per_frame, name=name)
    return int(flat.shape[1]) // tokens_per_frame


class NavSimCvoiProductionModelRuntime:
    """Strict frozen-model implementation of NavSimCvoiModelRuntime."""

    def __init__(
        self,
        *,
        config: Any,
        device: torch.device,
        encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        predictor_p1: Optional[torch.nn.Module],
        token_ae: Optional[torch.nn.Module],
        planner_p0: torch.nn.Module,
        planner_p1: Optional[torch.nn.Module],
        dual_value_model: Optional[torch.nn.Module],
        multiview_fusion: Optional[torch.nn.Module],
        embed_dim: int,
        tokens_per_frame: int,
        runtime_normalize_reps: bool,
        num_planner_poses: int,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.encoder = encoder
        self.predictor_p0 = predictor
        self.predictor = predictor
        self.predictor_p1 = predictor_p1
        self.token_ae = token_ae
        self.planner_p0 = planner_p0
        self.planner_p1 = planner_p1
        self.dual_value_model = dual_value_model
        self._dual_value_adapter = None if dual_value_model is None else CvoiValueDtypeAdapter(dual_value_model)
        self.multiview_fusion = multiview_fusion
        self.embed_dim = int(embed_dim)
        self.tokens_per_frame = int(tokens_per_frame)
        self.max_horizon = int(config.cvoi.max_horizon)
        self.runtime_normalize_reps = bool(runtime_normalize_reps)
        self.num_planner_poses = int(num_planner_poses)
        if self.embed_dim <= 0 or self.tokens_per_frame <= 0 or self.max_horizon <= 0:
            raise ValueError("CVoI runtime dimensions and horizon must be positive")
        if self.num_planner_poses <= 0:
            raise ValueError("CVoI planner must predict at least one raw future pose")
        if (self.predictor_p1 is None) != (self.planner_p1 is None):
            raise ValueError("CVoI P1 Predictor and Planner must be present or absent as one policy pair")
        if (
            str(getattr(config.cvoi, "protocol_version", "legacy_v1")) == _NAVSIM_E120_PROTOCOL
            and self.predictor_p1 is not None
            and self.predictor_p1 is self.predictor_p0
        ):
            raise ValueError("NavSim-e120 P0 and P1 policy bundles must not share a Predictor instance")
        for module in (
            self.encoder,
            self.predictor_p0,
            self.predictor_p1,
            self.token_ae,
            self.planner_p0,
            self.planner_p1,
            self.dual_value_model,
            self._dual_value_adapter,
            self.multiview_fusion,
        ):
            _freeze_module(module)

    def _validate_model_batch(self, batch: NavSimCvoiModelBatch) -> int:
        if not isinstance(batch, NavSimCvoiModelBatch):
            raise TypeError("NavSim CVoI production runtime requires NavSimCvoiModelBatch")
        if not isinstance(batch.metadata, Mapping):
            raise TypeError("NavSim CVoI model metadata must be a mapping")
        unknown_metadata = sorted(set(batch.metadata) - _ONLINE_METADATA_KEYS)
        if unknown_metadata:
            raise ValueError(f"NavSim CVoI model boundary rejects metadata key(s): {unknown_metadata}")
        if not isinstance(batch.context_frames, torch.Tensor) or not batch.context_frames.is_floating_point():
            raise ValueError("NavSim CVoI context_frames must be a floating tensor")
        batch_size = int(batch.context_frames.shape[0])
        observed = int(self.config.train.num_observed_frames)
        if batch_size < 1 or _time_length(batch.context_frames, video=True) != observed:
            raise ValueError(f"NavSim CVoI context_frames must contain exactly {observed} observed frames")
        if not bool(torch.isfinite(batch.context_frames).all().item()):
            raise ValueError("NavSim CVoI context_frames must be finite")
        for name, value, length in (
            ("actions", batch.actions, observed - 1),
            ("states", batch.states, observed),
            ("extrinsics", batch.extrinsics, observed),
            ("driving_command", batch.driving_command, observed),
            ("ego_dynamics", batch.ego_dynamics, observed),
        ):
            _require_finite_tensor(value, name=name, batch_size=batch_size, time_length=length)
        if batch.proposal_context_frames is not None:
            proposal = batch.proposal_context_frames
            if (
                not isinstance(proposal, torch.Tensor)
                or not proposal.is_floating_point()
                or int(proposal.shape[0]) != batch_size
                or _time_length(proposal, video=True) != observed
                or not bool(torch.isfinite(proposal).all().item())
            ):
                raise ValueError("NavSim CVoI proposal_context_frames must contain only observed finite video")
        for name, matrix_size in (("camera_intrinsics", 3), ("camera2ego", 4)):
            value = batch.metadata.get(name)
            if value is None:
                continue
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 5
                or int(value.shape[0]) != batch_size
                or int(value.shape[2]) != observed
                or value.shape[-2:] != (matrix_size, matrix_size)
                or not value.is_floating_point()
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(
                    f"NavSim CVoI {name} must contain exactly the observed [B,V,T,{matrix_size},{matrix_size}]"
                )
        metadata_valid_mask = batch.metadata.get("metadata_valid_mask")
        if metadata_valid_mask is not None and (
            not isinstance(metadata_valid_mask, torch.Tensor)
            or metadata_valid_mask.dtype != torch.bool
            or metadata_valid_mask.shape != (batch_size, observed)
        ):
            raise ValueError("NavSim CVoI metadata_valid_mask must be observed-only bool [B,T]")
        observed_metadata_valid_mask = batch.metadata.get("observed_metadata_valid_mask")
        if observed_metadata_valid_mask is not None and (
            not isinstance(observed_metadata_valid_mask, torch.Tensor)
            or observed_metadata_valid_mask.dtype != torch.bool
            or observed_metadata_valid_mask.shape != (batch_size,)
        ):
            raise ValueError("NavSim CVoI observed_metadata_valid_mask must be bool [B]")
        return batch_size

    def _encode_batch_with_policy(
        self,
        batch: NavSimCvoiModelBatch,
        *,
        predictor: torch.nn.Module,
        planner: torch.nn.Module,
    ) -> NavSimCvoiEncodedBatch:
        """Encode observed video and roll exactly H steps with one inseparable policy pair."""

        batch_size = self._validate_model_batch(batch)
        observed_raw = int(self.config.train.num_observed_frames)
        total_raw = int(self.config.data.num_target_frames)
        if total_raw <= observed_raw:
            raise ValueError("CVoI checkpoint timeline must include raw future poses")

        actions = _zero_extend(batch.actions, total_length=total_raw - 1, name="actions")
        states = _zero_extend(batch.states, total_length=total_raw, name="states")
        extrinsics = _zero_extend(batch.extrinsics, total_length=total_raw, name="extrinsics")
        driving_command = _zero_extend(
            batch.driving_command,
            total_length=total_raw,
            name="driving_command",
        )
        ego_dynamics = _zero_extend(batch.ego_dynamics, total_length=total_raw, name="ego_dynamics")
        metadata_valid_mask = _zero_extend_mask(
            batch.metadata.get("metadata_valid_mask"),
            batch_size=batch_size,
            observed=observed_raw,
            total=total_raw,
            name="metadata_valid_mask",
        )
        observed_metadata_valid_mask = batch.metadata.get("observed_metadata_valid_mask")
        if observed_metadata_valid_mask is not None and (
            not isinstance(observed_metadata_valid_mask, torch.Tensor)
            or observed_metadata_valid_mask.dtype != torch.bool
            or int(observed_metadata_valid_mask.shape[0]) != batch_size
        ):
            raise ValueError("NavSim CVoI observed_metadata_valid_mask must be a boolean tensor with batch B")

        predictor_inputs = enforce_cvoi_zero_future_aux(
            build_predictor_timeline_inputs(
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
                config=self.config,
                encoder=self.encoder,
                dt=1.0 / float(self.config.data.fps),
                metadata_valid_mask=metadata_valid_mask,
                observed_metadata_valid_mask=observed_metadata_valid_mask,
            )
        )
        if int(predictor_inputs.tokens_per_frame) != self.tokens_per_frame:
            raise ValueError(
                "CVoI predictor timeline tokens_per_frame mismatch: "
                f"runtime={self.tokens_per_frame}, timeline={predictor_inputs.tokens_per_frame}"
            )
        predictor_observed = int(predictor_inputs.num_observed_steps)
        predictor_total = int(predictor_inputs.num_time_steps)
        if predictor_total - predictor_observed < self.max_horizon:
            raise ValueError(
                f"CVoI predictor exposes only {predictor_total - predictor_observed} future steps; "
                f"requires H={self.max_horizon}"
            )

        camera_metadata = {
            key: value
            for key, value in batch.metadata.items()
            if key in {"camera_names", "camera_intrinsics", "camera2ego"}
        }
        if bool(getattr(self.config.multiview, "enabled", False)):
            missing = [key for key in ("camera_intrinsics", "camera2ego") if key not in camera_metadata]
            if missing:
                raise ValueError(f"CVoI multiview runtime requires observed camera metadata: missing={missing}")

        with torch.no_grad(), cvoi_execution_autocast(self.config, self.device):
            z_context = forward_main_context(
                self.encoder,
                batch.context_frames,
                config=self.config,
                runtime_normalize_reps=self.runtime_normalize_reps,
                token_ae=self.token_ae,
                multiview_fusion=self.multiview_fusion,
                camera_metadata=camera_metadata,
            )
            expected_observed_tokens = predictor_observed * self.tokens_per_frame
            if (
                z_context.ndim != 3
                or z_context.shape[0] != batch_size
                or z_context.shape[-1] != self.embed_dim
                or z_context.shape[1] != expected_observed_tokens
                or not bool(torch.isfinite(z_context).all().item())
            ):
                raise ValueError(
                    "CVoI encoder must return the exact observed latent prefix "
                    f"[B,{expected_observed_tokens},{self.embed_dim}], got {tuple(z_context.shape)}"
                )
            step_predictor = make_predictor_step_fn(
                predictor,
                self.config,
                predictor_observed,
                driving_command=predictor_inputs.driving_command,
                ego_dynamics=predictor_inputs.ego_dynamics,
                predictor_no_aux_input=bool(self.config.train.predictor_no_aux_input),
                normalize_reps=self.runtime_normalize_reps,
                no_grad=True,
            )
            rollout_latency_ms_by_horizon = [0.0]
            cumulative_rollout_latency_ms = 0.0

            def timed_step_predictor(
                z_prefix: torch.Tensor,
                actions_step: torch.Tensor,
                states_step: torch.Tensor,
                extrinsics_step: torch.Tensor,
            ) -> torch.Tensor:
                nonlocal cumulative_rollout_latency_ms
                output, step_latency_ms = _time_tensor_operation(
                    z_prefix,
                    lambda: step_predictor(z_prefix, actions_step, states_step, extrinsics_step),
                )
                if not isinstance(output, torch.Tensor):
                    raise TypeError("CVoI predictor step must return a tensor")
                cumulative_rollout_latency_ms += step_latency_ms
                rollout_latency_ms_by_horizon.append(cumulative_rollout_latency_ms)
                return output

            _, z_future = rollout_latent_predictions(
                timed_step_predictor,
                config=self.config,
                z_context=z_context,
                actions=predictor_inputs.actions,
                states=predictor_inputs.states,
                extrinsics=predictor_inputs.extrinsics,
                num_obs=predictor_observed,
                tokens_per_frame=self.tokens_per_frame,
                num_total=predictor_total,
                compute_tf=False,
                needs_ar_rollout=True,
                planner_only_error_context=True,
                validate_ic_prefix=True,
                rollout_end_step=predictor_observed + self.max_horizon,
            )
        if len(rollout_latency_ms_by_horizon) != self.max_horizon + 1:
            raise RuntimeError(
                "CVoI predictor timing must record exactly one cumulative value per rollout horizon; "
                f"got {len(rollout_latency_ms_by_horizon)} values for H={self.max_horizon}"
            )
        if z_future is None or z_future.shape != (
            batch_size,
            self.max_horizon * self.tokens_per_frame,
            self.embed_dim,
        ):
            raise ValueError(
                "CVoI predictor must return exactly H future latent frames, got "
                f"{None if z_future is None else tuple(z_future.shape)}"
            )

        status_feature = prepare_inference_consistent_status_vector(
            batch.states,
            num_observed=observed_raw,
            driving_command=batch.driving_command,
            ego_dynamics=batch.ego_dynamics,
            state_dim=resolve_planner_status_dim(self.config),
            use_drive_command=resolve_planner_use_drive_command(self.config),
        )
        z_first_frame = z_context[:, : self.tokens_per_frame] if bool(self.config.planner.use_z_context) else None
        z_observed_for_planner = z_context if resolve_planner_use_observed_tokens(self.config) else None
        action_history = None
        if bool(self.config.planner.use_action_history_for_planner):
            action_history = build_observed_action_trajectory_history(
                predictor_inputs.actions,
                num_observed_frames=predictor_observed,
                action_history_dim=int(self.config.planner.action_history_dim),
                dt=float(self.config.planner.diff_dt) * max(int(predictor_inputs.frame_stride), 1),
            )
        anchor_state = build_ego_relative_diffusion_anchor(
            planner,
            ego_dynamics=batch.ego_dynamics,
            observed_frames=observed_raw,
            reference=batch.states,
        )
        contexts = tuple(
            NavSimCvoiPlannerContext(
                status_feature=status_feature[index : index + 1],
                z_first_frame=None if z_first_frame is None else z_first_frame[index : index + 1],
                z_observed_for_planner=(
                    None if z_observed_for_planner is None else z_observed_for_planner[index : index + 1]
                ),
                action_history=None if action_history is None else action_history[index : index + 1],
                anchor_state=None if anchor_state is None else anchor_state[index : index + 1],
                rollout_latency_ms_by_horizon=tuple(rollout_latency_ms_by_horizon),
            )
            for index in range(batch_size)
        )
        return NavSimCvoiEncodedBatch(
            z_observed=z_context.reshape(batch_size, predictor_observed, self.tokens_per_frame, self.embed_dim),
            z_future=z_future.reshape(batch_size, self.max_horizon, self.tokens_per_frame, self.embed_dim),
            model_contexts=contexts,
        )

    def encode_batch(self, batch: NavSimCvoiModelBatch) -> NavSimCvoiEncodedBatch:
        """Encode and roll with the complete P0 Predictor+Planner policy pair."""

        return self._encode_batch_with_policy(
            batch,
            predictor=self.predictor_p0,
            planner=self.planner_p0,
        )

    def encode_p1_batch(self, batch: NavSimCvoiModelBatch) -> NavSimCvoiEncodedBatch:
        """Encode and roll with the complete P1 Predictor+Planner policy pair."""

        if self.predictor_p1 is None or self.planner_p1 is None:
            raise RuntimeError("P1 encoding requires a loaded P1 Predictor+Planner policy pair")
        return self._encode_batch_with_policy(
            batch,
            predictor=self.predictor_p1,
            planner=self.planner_p1,
        )

    def start_online_session(
        self,
        batch: NavSimCvoiModelBatch,
        *,
        policy: str,
    ) -> NavSimCvoiOnlineSession:
        """Encode observed inputs once without executing a future Predictor step."""

        if policy not in {"p0", "p1"}:
            raise ValueError("online CVoI policy must be exactly 'p0' or 'p1'")
        if policy == "p1" and (self.predictor_p1 is None or self.planner_p1 is None):
            raise RuntimeError("online P1 session requires a loaded P1 Predictor+Planner pair")
        predictor = self.predictor_p0 if policy == "p0" else self.predictor_p1
        planner = self.planner_p0 if policy == "p0" else self.planner_p1
        if predictor is None or planner is None:
            raise RuntimeError(f"online {policy.upper()} session is missing its policy pair")

        batch_size = self._validate_model_batch(batch)
        if batch_size != 1:
            raise ValueError("online CVoI session requires batch size 1")
        observed_raw = int(self.config.train.num_observed_frames)
        total_raw = int(self.config.data.num_target_frames)
        if total_raw <= observed_raw:
            raise ValueError("CVoI checkpoint timeline must include raw future poses")

        actions = _zero_extend(batch.actions, total_length=total_raw - 1, name="actions")
        states = _zero_extend(batch.states, total_length=total_raw, name="states")
        extrinsics = _zero_extend(batch.extrinsics, total_length=total_raw, name="extrinsics")
        driving_command = _zero_extend(batch.driving_command, total_length=total_raw, name="driving_command")
        ego_dynamics = _zero_extend(batch.ego_dynamics, total_length=total_raw, name="ego_dynamics")
        metadata_valid_mask = _zero_extend_mask(
            batch.metadata.get("metadata_valid_mask"),
            batch_size=batch_size,
            observed=observed_raw,
            total=total_raw,
            name="metadata_valid_mask",
        )
        observed_metadata_valid_mask = batch.metadata.get("observed_metadata_valid_mask")
        predictor_inputs = enforce_cvoi_zero_future_aux(
            build_predictor_timeline_inputs(
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
                config=self.config,
                encoder=self.encoder,
                dt=1.0 / float(self.config.data.fps),
                metadata_valid_mask=metadata_valid_mask,
                observed_metadata_valid_mask=observed_metadata_valid_mask,
            )
        )
        if int(predictor_inputs.tokens_per_frame) != self.tokens_per_frame:
            raise ValueError("online CVoI predictor timeline tokens_per_frame mismatch")
        predictor_observed = int(predictor_inputs.num_observed_steps)
        predictor_total = int(predictor_inputs.num_time_steps)
        if predictor_total - predictor_observed < self.max_horizon:
            raise ValueError(
                f"online CVoI predictor exposes only {predictor_total - predictor_observed} future steps; "
                f"requires H={self.max_horizon}"
            )

        camera_metadata = {
            key: value
            for key, value in batch.metadata.items()
            if key in {"camera_names", "camera_intrinsics", "camera2ego"}
        }
        if bool(getattr(self.config.multiview, "enabled", False)):
            missing = [key for key in ("camera_intrinsics", "camera2ego") if key not in camera_metadata]
            if missing:
                raise ValueError(f"online CVoI multiview runtime requires observed camera metadata: missing={missing}")

        with torch.no_grad(), cvoi_execution_autocast(self.config, self.device):
            z_context = forward_main_context(
                self.encoder,
                batch.context_frames,
                config=self.config,
                runtime_normalize_reps=self.runtime_normalize_reps,
                token_ae=self.token_ae,
                multiview_fusion=self.multiview_fusion,
                camera_metadata=camera_metadata,
            )
        expected_observed_tokens = predictor_observed * self.tokens_per_frame
        if (
            not isinstance(z_context, torch.Tensor)
            or z_context.shape != (1, expected_observed_tokens, self.embed_dim)
            or not z_context.is_floating_point()
            or not bool(torch.isfinite(z_context).all().item())
        ):
            raise ValueError(
                "online CVoI encoder must return the exact observed latent prefix "
                f"[1,{expected_observed_tokens},{self.embed_dim}]"
            )
        step_predictor = make_predictor_step_fn(
            predictor,
            self.config,
            predictor_observed,
            driving_command=predictor_inputs.driving_command,
            ego_dynamics=predictor_inputs.ego_dynamics,
            predictor_no_aux_input=bool(self.config.train.predictor_no_aux_input),
            normalize_reps=self.runtime_normalize_reps,
            no_grad=True,
        )

        status_feature = prepare_inference_consistent_status_vector(
            batch.states,
            num_observed=observed_raw,
            driving_command=batch.driving_command,
            ego_dynamics=batch.ego_dynamics,
            state_dim=resolve_planner_status_dim(self.config),
            use_drive_command=resolve_planner_use_drive_command(self.config),
        )
        z_first_frame = z_context[:, : self.tokens_per_frame] if bool(self.config.planner.use_z_context) else None
        z_observed_for_planner = z_context if resolve_planner_use_observed_tokens(self.config) else None
        action_history = None
        if bool(self.config.planner.use_action_history_for_planner):
            action_history = build_observed_action_trajectory_history(
                predictor_inputs.actions,
                num_observed_frames=predictor_observed,
                action_history_dim=int(self.config.planner.action_history_dim),
                dt=float(self.config.planner.diff_dt) * max(int(predictor_inputs.frame_stride), 1),
            )
        anchor_state = build_ego_relative_diffusion_anchor(
            planner,
            ego_dynamics=batch.ego_dynamics,
            observed_frames=observed_raw,
            reference=batch.states,
        )
        context = _NavSimCvoiOnlineModelContext(
            status_feature=status_feature,
            z_first_frame=z_first_frame,
            z_observed_for_planner=z_observed_for_planner,
            action_history=action_history,
            anchor_state=anchor_state,
        )
        return NavSimCvoiOnlineSession(
            z_observed=z_context.detach(),
            model_context=context,
            predictor_inputs=predictor_inputs,
            step_predictor=step_predictor,
            policy=policy,
        )

    def _validate_online_prefix(
        self,
        session: NavSimCvoiOnlineSession,
        raw_prefix: torch.Tensor,
    ) -> tuple[_NavSimCvoiOnlineModelContext, torch.Tensor, int]:
        if not isinstance(session, NavSimCvoiOnlineSession):
            raise TypeError("online CVoI operation requires NavSimCvoiOnlineSession")
        if not isinstance(session.model_context, _NavSimCvoiOnlineModelContext):
            raise TypeError("online CVoI session has an invalid model context")
        if (
            not isinstance(raw_prefix, torch.Tensor)
            or raw_prefix.ndim != 3
            or raw_prefix.shape[0] != 1
            or raw_prefix.shape[2] != self.embed_dim
            or raw_prefix.shape[1] % self.tokens_per_frame
            or not raw_prefix.is_floating_point()
        ):
            raise ValueError("online CVoI raw_prefix must be floating [1,h*tokens_per_frame,embed_dim]")
        horizon = int(raw_prefix.shape[1]) // self.tokens_per_frame
        if horizon < 0 or horizon > self.max_horizon:
            raise ValueError(f"online CVoI raw prefix horizon must be in [0,{self.max_horizon}]")
        return session.model_context, raw_prefix, horizon

    def rollout_online_step(
        self,
        session: NavSimCvoiOnlineSession,
        raw_prefix: torch.Tensor,
        *,
        next_horizon: int,
    ) -> torch.Tensor:
        """Execute exactly one Predictor invocation without nested timing."""

        _, raw_prefix, horizon = self._validate_online_prefix(session, raw_prefix)
        if type(next_horizon) is not int or next_horizon != horizon + 1 or next_horizon > self.max_horizon:
            raise ValueError("online CVoI next_horizon must equal current prefix horizon + 1")
        inputs = session.predictor_inputs
        num_observed = int(inputs.num_observed_steps)
        num_total = int(inputs.num_time_steps)
        timeline_step = num_observed + next_horizon - 1
        if timeline_step == num_total - 1:
            actions_step = inputs.actions
            states_step = inputs.states[:, :-1]
            extrinsics_step = inputs.extrinsics[:, :-1]
        else:
            actions_step = inputs.actions[:, :timeline_step]
            states_step = inputs.states[:, :timeline_step]
            extrinsics_step = inputs.extrinsics[:, :timeline_step]
        rolled = torch.cat([session.z_observed, raw_prefix], dim=1)
        with torch.no_grad(), cvoi_execution_autocast(self.config, self.device):
            output = session.step_predictor(rolled, actions_step, states_step, extrinsics_step)
        if (
            not isinstance(output, torch.Tensor)
            or output.ndim != 3
            or output.shape[0] != 1
            or output.shape[1] < self.tokens_per_frame
            or output.shape[2] != self.embed_dim
            or not output.is_floating_point()
        ):
            raise ValueError("online CVoI Predictor step must return floating [1,N>=tokens_per_frame,embed_dim]")
        return output[:, -self.tokens_per_frame :]

    def online_value_features(
        self,
        session: NavSimCvoiOnlineSession,
        raw_prefix: torch.Tensor,
        *,
        horizon: int,
        controller_lineage: str,
    ) -> dict[str, torch.Tensor]:
        """Evaluate the Stop/Field heads for one currently visited prefix."""

        _, raw_prefix, actual_horizon = self._validate_online_prefix(session, raw_prefix)
        if type(horizon) is not int or horizon != actual_horizon:
            raise ValueError("online CVoI Value horizon must match the raw prefix")
        expected_lineage = "p0_controller" if session.policy == "p0" else "value_guided"
        if controller_lineage != expected_lineage:
            raise ValueError("online CVoI controller lineage does not match the policy session")
        if self._dual_value_adapter is None:
            raise RuntimeError("online CVoI Value evaluation requires a loaded dual-value model")
        with torch.no_grad():
            output = self._dual_value_adapter(
                session.z_observed,
                raw_prefix,
                tokens_per_frame=self.tokens_per_frame,
            )
        values = extract_prefix_gate_values(output)
        field_value = values["field_value"]
        if controller_lineage == "p0_controller":
            field_value = torch.zeros_like(values["stop_value"])
        return {"field_value": field_value, "stop_value": values["stop_value"]}

    def prepare_online_terminal_prefix(
        self,
        session: NavSimCvoiOnlineSession,
        raw_prefix: torch.Tensor,
        *,
        horizon: int,
        controller_lineage: str,
        guidance_steps: Optional[int],
    ) -> tuple[torch.Tensor, Mapping[str, float]]:
        """Apply terminal Guidance only; Planner inference is deliberately separate."""

        _, raw_prefix, actual_horizon = self._validate_online_prefix(session, raw_prefix)
        if type(horizon) is not int or horizon != actual_horizon:
            raise ValueError("online CVoI terminal horizon must match the raw prefix")
        expected_lineage = "p0_controller" if session.policy == "p0" else "value_guided"
        if controller_lineage != expected_lineage:
            raise ValueError("online CVoI controller lineage does not match the policy session")
        if guidance_steps is not None and (type(guidance_steps) is not int or guidance_steps not in {0, 1, 2, 3, 4}):
            raise ValueError("online CVoI guidance_steps must be one of {0,1,2,3,4} or None")
        if horizon == 0 or controller_lineage == "p0_controller" or guidance_steps == 0:
            return raw_prefix.detach(), {
                "guidance_steps": 0.0,
                "guidance_skipped_h0": float(horizon == 0),
                "delta_norm": 0.0,
                "field_value_before": 0.0,
                "field_value_after": 0.0,
            }
        if self._dual_value_adapter is None:
            raise RuntimeError("online CVoI Guidance requires a loaded dual-value model")
        applied_steps = 2 if guidance_steps is None else guidance_steps
        guided, diagnostics = apply_cvoi_planner_guidance(
            session.z_observed,
            raw_prefix,
            self._dual_value_adapter,
            tokens_per_frame=self.tokens_per_frame,
            config=self.config,
            evaluation_guidance_steps=applied_steps,
        )
        actual_steps = diagnostics.get("guidance_steps")
        if (
            isinstance(actual_steps, bool)
            or not isinstance(actual_steps, (int, float))
            or not math.isfinite(float(actual_steps))
            or float(actual_steps) != float(applied_steps)
        ):
            raise RuntimeError(f"online CVoI runtime expected Guidance K={applied_steps}")
        return guided.to(dtype=raw_prefix.dtype), dict(diagnostics)

    def plan_online_terminal_prefix(
        self,
        session: NavSimCvoiOnlineSession,
        planner_prefix: torch.Tensor,
        *,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run only the policy-matched Planner with explicit common-random seed."""

        context, planner_prefix, _ = self._validate_online_prefix(session, planner_prefix)
        planner = self.planner_p0 if session.policy == "p0" else self.planner_p1
        if planner is None:
            raise RuntimeError(f"online {session.policy.upper()} Planner is not loaded")
        return self._planner_forward(planner, context=context, prefix_flat=planner_prefix, seed=seed)

    def _validate_evaluation_inputs(
        self,
        *,
        context: object,
        z_observed: torch.Tensor,
        prefix: torch.Tensor,
        horizon: int,
        apply_guidance: Optional[bool],
    ) -> tuple[NavSimCvoiPlannerContext, torch.Tensor, torch.Tensor]:
        required = (
            "status_feature",
            "z_first_frame",
            "z_observed_for_planner",
            "action_history",
            "anchor_state",
            "rollout_latency_ms_by_horizon",
        )
        if not all(hasattr(context, field) for field in required):
            raise TypeError("CVoI planner context is missing online planner conditions")
        horizon = int(horizon)
        if horizon < 0 or horizon > self.max_horizon:
            raise ValueError(f"CVoI horizon must be in [0, {self.max_horizon}], got {horizon}")
        if apply_guidance is not None and bool(apply_guidance) != (horizon > 0):
            raise ValueError("CVoI Guidance must run exactly when horizon > 0")
        rollout_latencies = context.rollout_latency_ms_by_horizon
        if (
            not isinstance(rollout_latencies, tuple)
            or len(rollout_latencies) != self.max_horizon + 1
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or value < 0.0
                for value in rollout_latencies
            )
            or rollout_latencies[0] != 0.0
            or any(right < left for left, right in zip(rollout_latencies, rollout_latencies[1:]))
        ):
            raise ValueError("CVoI planner context requires non-decreasing rollout latency for h=0..H")
        observed_flat = _flatten_latent(
            z_observed,
            tokens_per_frame=self.tokens_per_frame,
            name="z_observed",
        )
        prefix_flat = _flatten_latent(prefix, tokens_per_frame=self.tokens_per_frame, name="prefix")
        if (
            observed_flat.shape[0] != 1
            or prefix_flat.shape[0] != 1
            or observed_flat.shape[-1] != self.embed_dim
            or prefix_flat.shape[-1] != self.embed_dim
        ):
            raise ValueError("CVoI horizon evaluation requires batch size 1 with the runtime embed_dim")
        if _prefix_frames(prefix, tokens_per_frame=self.tokens_per_frame, name="prefix") != horizon:
            raise ValueError("CVoI raw prefix frame count must equal horizon")
        if not bool(torch.isfinite(observed_flat).all().item()) or not bool(torch.isfinite(prefix_flat).all().item()):
            raise ValueError("CVoI horizon latents must be finite")
        return context, observed_flat, prefix_flat

    def _planner_forward(
        self,
        planner: torch.nn.Module,
        *,
        context: NavSimCvoiPlannerContext,
        prefix_flat: torch.Tensor,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        validate_empty_future_planner_conditions(
            prefix_flat,
            z_context=context.z_first_frame,
            z_observed=context.z_observed_for_planner,
            action_history=context.action_history,
        )
        kwargs = {
            "z_context": context.z_first_frame,
            "z_observed": context.z_observed_for_planner,
            "action_history": context.action_history,
        }
        if str(self.config.planner.planner_type) == "diffusion":
            kwargs["anchor_state"] = context.anchor_state
            kwargs["inference_noise"] = cvoi_planner_inference_noise(
                planner,
                seeds=[int(seed)],
                device=prefix_flat.device,
            )
        with (
            common_random_numbers(int(seed)),
            torch.no_grad(),
            cvoi_execution_autocast(self.config, self.device),
        ):
            output = planner(prefix_flat, context.status_feature, **kwargs)
        output = validate_planner_output(output, mode="inference", num_poses=self.num_planner_poses)
        return output["trajectories"], output["confidences"]

    def _guided_prefix(
        self,
        z_observed: torch.Tensor,
        prefix_flat: torch.Tensor,
        *,
        horizon: int,
        guidance_steps: Optional[int] = None,
    ) -> tuple[torch.Tensor, int, float]:
        if horizon == 0:
            return prefix_flat.detach(), 0, 0.0
        if self.dual_value_model is None:
            raise RuntimeError("guided CVoI runtime requires a loaded PrefixDualValueModel")
        if self._dual_value_adapter is None:
            raise RuntimeError("guided CVoI runtime is missing its Value dtype adapter")
        result, guidance_latency_ms = _time_tensor_operation(
            prefix_flat,
            lambda: apply_cvoi_planner_guidance(
                z_observed,
                prefix_flat,
                self._dual_value_adapter,
                tokens_per_frame=self.tokens_per_frame,
                config=self.config,
                evaluation_guidance_steps=guidance_steps,
            ),
        )
        guided, diagnostics = result
        applied_steps = int(diagnostics["guidance_steps"])
        expected_steps = 2 if guidance_steps is None else guidance_steps
        if applied_steps != expected_steps:
            raise RuntimeError(f"CVoI runtime expected Guidance K={expected_steps}, got {applied_steps}")
        return guided.to(dtype=prefix_flat.dtype), applied_steps, guidance_latency_ms

    def evaluate_unguided_prefix(
        self,
        *,
        context: object,
        z_observed: torch.Tensor,
        prefix: torch.Tensor,
        horizon: int,
        seed: int,
    ) -> CvoiPlannerEvaluation:
        context, _, prefix_flat = self._validate_evaluation_inputs(
            context=context,
            z_observed=z_observed,
            prefix=prefix,
            horizon=horizon,
            apply_guidance=None,
        )
        pred_trajs, confidences = self._planner_forward(
            self.planner_p0,
            context=context,
            prefix_flat=prefix_flat,
            seed=seed,
        )
        adaptive_latency_ms = float(context.rollout_latency_ms_by_horizon[int(horizon)])
        return CvoiPlannerEvaluation(pred_trajs, confidences, adaptive_latency_ms, 0)

    def evaluate_p1_unguided_prefix(
        self,
        *,
        context: object,
        z_observed: torch.Tensor,
        prefix: torch.Tensor,
        horizon: int,
        seed: int,
    ) -> CvoiPlannerEvaluation:
        """Run P1 on the raw prefix, isolating Guidance from Planner training."""

        if self.planner_p1 is None:
            raise RuntimeError("P1 unguided evaluation requires a loaded P1 planner")
        context, _, prefix_flat = self._validate_evaluation_inputs(
            context=context,
            z_observed=z_observed,
            prefix=prefix,
            horizon=horizon,
            apply_guidance=None,
        )
        pred_trajs, confidences = self._planner_forward(
            self.planner_p1,
            context=context,
            prefix_flat=prefix_flat,
            seed=seed,
        )
        replay_latency_ms = float(context.rollout_latency_ms_by_horizon[int(horizon)])
        return CvoiPlannerEvaluation(pred_trajs, confidences, replay_latency_ms, 0)

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
        if self.planner_p1 is None:
            raise RuntimeError("guided CVoI runtime requires a loaded P1 planner")
        context, observed_flat, prefix_flat = self._validate_evaluation_inputs(
            context=context,
            z_observed=z_observed,
            prefix=raw_prefix,
            horizon=horizon,
            apply_guidance=apply_guidance,
        )
        guided, guidance_steps, guidance_latency_ms = self._guided_prefix(
            observed_flat,
            prefix_flat,
            horizon=int(horizon),
            guidance_steps=guidance_steps,
        )
        pred_trajs, confidences = self._planner_forward(
            self.planner_p1,
            context=context,
            prefix_flat=guided,
            seed=seed,
        )
        adaptive_latency_ms = float(context.rollout_latency_ms_by_horizon[int(horizon)]) + guidance_latency_ms
        return CvoiPlannerEvaluation(pred_trajs, confidences, adaptive_latency_ms, guidance_steps)


def _init_multiview_fusion(
    config: Any,
    *,
    embed_dim: int,
    raw_tokens_per_frame: int,
    device: torch.device,
) -> Optional[torch.nn.Module]:
    if not bool(getattr(config.multiview, "enabled", False)):
        return None
    if str(config.multiview.fusion_type) != "petr_cross_attn":
        raise ValueError(f"unsupported CVoI multiview fusion_type={config.multiview.fusion_type!r}")
    return PETRMultiViewFusion(
        embed_dim=int(embed_dim),
        tokens_per_frame=int(raw_tokens_per_frame),
        hidden_dim=int(config.multiview.hidden_dim),
        num_heads=int(config.multiview.num_heads),
        dropout=float(config.multiview.dropout),
        output_mode=str(config.multiview.output_mode),
    ).to(device)


def _world4drive_sha256(path: str | Path) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"World4Drive runtime artifact does not exist: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _world4drive_module_state(
    state: object,
    *,
    module: torch.nn.Module,
    name: str,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    core = module.module if hasattr(module, "module") else module
    if not isinstance(core, torch.nn.Module):
        raise TypeError(f"World4Drive {name} target must be a torch.nn.Module")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"World4Drive {name} checkpoint state must be a non-empty mapping")
    checkpoint: dict[str, torch.Tensor] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or not raw_key or not torch.is_tensor(value):
            raise ValueError(f"World4Drive {name} checkpoint state must map non-empty string keys to tensors")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if not key or key in checkpoint:
            raise ValueError(f"World4Drive {name} checkpoint has duplicate normalized state key {key!r}")
        checkpoint[key] = value
    target = core.state_dict()
    if set(checkpoint) != set(target):
        raise ValueError(
            f"World4Drive {name} state keys mismatch: "
            f"missing={sorted(set(target) - set(checkpoint))}, "
            f"unexpected={sorted(set(checkpoint) - set(target))}"
        )
    shape_mismatch = {
        key: (tuple(checkpoint[key].shape), tuple(target[key].shape))
        for key in target
        if checkpoint[key].shape != target[key].shape
    }
    if shape_mismatch:
        raise ValueError(f"World4Drive {name} state shape mismatch: {shape_mismatch}")
    dtype_mismatch = {
        key: (checkpoint[key].dtype, target[key].dtype) for key in target if checkpoint[key].dtype != target[key].dtype
    }
    if dtype_mismatch:
        raise ValueError(f"World4Drive {name} state dtype mismatch: {dtype_mismatch}")
    return core, checkpoint


def _read_world4drive_checkpoint(path: str | Path, *, name: str) -> Mapping[str, object]:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"World4Drive {name} checkpoint does not exist: {artifact}")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"World4Drive {name} checkpoint must contain a mapping")
    return payload


def _require_world4drive_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"World4Drive {name} must be a lowercase SHA-256 digest")
    return value


def _world4drive_exactly_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(_world4drive_exactly_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _world4drive_exactly_equal(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    return left == right


def _expected_world4drive_planner_ablation(binding: CvoiWorld4DriveRuntimeBinding, *, p0: bool) -> dict[str, object]:
    if p0:
        return {
            "schema": "cvoi_ablation_v1",
            "experiment_role": "main",
            "branch_id": "uniform_s239",
            "shared_cohort_id": "seed_s239",
            "cf_field_supervision": "hazard_quality",
            "field_calibration_mode": "local_geometry",
            "p0_prefix_mode": "uniform",
            "gate_feature_mode": "full",
            "train_seed": 239,
            "evaluation_seed": 239,
        }
    return {
        "schema": "cvoi_ablation_v1",
        "experiment_role": "ablation",
        "branch_id": f"{binding.lineage}_s239",
        "shared_cohort_id": "seed_s239",
        "cf_field_supervision": "hazard_quality" if binding.lineage == "real_cf_value" else "none",
        "field_calibration_mode": "local_geometry",
        "p0_prefix_mode": "uniform",
        "gate_feature_mode": "full",
        "train_seed": 239,
        "evaluation_seed": 239,
    }


def _validate_world4drive_planner_payload(
    payload: Mapping[str, object],
    *,
    binding: CvoiWorld4DriveRuntimeBinding,
    p0: bool,
    config: Any,
) -> object:
    expected_stage = "unguided_planner" if p0 else "guided_planner"
    signature = validate_cvoi_planner_lineage(payload, expected_stage=expected_stage)
    if not isinstance(signature, Mapping) or set(signature) != WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS:
        actual = set(signature) if isinstance(signature, Mapping) else set()
        raise ValueError(
            "World4Drive Planner runtime signature fields mismatch: "
            f"missing={sorted(WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS - actual)}, "
            f"unexpected={sorted(actual - WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS)}"
        )
    audit = load_cvoi_audit_manifest(
        binding.checkpoint_audit_manifest_path,
        verification_mode="receipt_only",
    ).to_dict()
    expected_runtime = {
        "audit_signature": audit,
        "guidance_steps": binding.guidance_steps,
        "guidance_objective": binding.guidance_objective,
        "guidance_step_size": float(config.value_guidance.step_size),
        "guidance_max_delta_norm": float(config.value_guidance.max_delta_norm),
        "guidance_detach_output": bool(config.value_guidance.detach_output),
        "predictor_type": str(config.train.predictor_type).lower(),
        "runtime_normalize_reps": resolve_predictor_runtime_normalize_reps(config),
        "tokens_per_frame": binding.tokens_per_frame,
        "num_observed_frames": int(config.train.num_observed_frames),
        "num_target_frames": int(config.data.num_target_frames),
        "timestep_sec": binding.timestep_sec,
        "multiview_signature": _cvoi_multiview_signature(config),
        "planner_signature": _cvoi_planner_signature(config),
        "world_execution_signature": _cvoi_world_execution_signature(config),
        "execution_dtype_signature": cvoi_execution_dtype_signature(config),
        "inference_rng_signature": cvoi_inference_rng_signature(config),
        "world_model_sha256": _world4drive_sha256(binding.world_model_checkpoint),
        "token_ae_sha256": _world4drive_sha256(binding.token_ae_checkpoint),
        "gate_sha256": None,
    }
    mismatch = {
        key: (signature.get(key), value)
        for key, value in expected_runtime.items()
        if not _world4drive_exactly_equal(signature.get(key), value)
    }
    if mismatch:
        raise ValueError(f"World4Drive Planner runtime lineage mismatch: {mismatch}")
    expected_ablation = _expected_world4drive_planner_ablation(binding, p0=p0)
    if not _world4drive_exactly_equal(signature.get("ablation_signature"), expected_ablation):
        raise ValueError("World4Drive Planner ablation lineage mismatch")
    if signature.get("p0_protocol") != "fixed_final_epoch_v1":
        raise ValueError("World4Drive Planner must use p0_protocol='fixed_final_epoch_v1'")
    expected_distribution = {str(horizon): 0.25 for horizon in range(4)}
    if not _world4drive_exactly_equal(signature.get("p0_prefix_distribution"), expected_distribution):
        raise ValueError("World4Drive Planner P0 prefix distribution lineage mismatch")
    if p0:
        _require_world4drive_sha256(signature.get("parent_planner_sha256"), name="P0 parent Planner SHA-256")
        if signature.get("dual_value_sha256") is not None:
            raise ValueError("World4Drive P0 Planner must not reference a Field checkpoint")
        if payload.get("epoch") != 20:
            raise ValueError("World4Drive P0 trust anchor must be the fixed final epoch=20 checkpoint")
    else:
        if binding.field_checkpoint is None or binding.guided_planner_checkpoint is None:
            raise ValueError("World4Drive P1 Planner requires Field/P1 binding paths")
        expected_lineage = {
            "parent_planner_sha256": _world4drive_sha256(binding.unguided_planner_checkpoint),
            "dual_value_sha256": _world4drive_sha256(binding.field_checkpoint),
        }
        lineage_mismatch = {
            key: (signature.get(key), value) for key, value in expected_lineage.items() if signature.get(key) != value
        }
        if lineage_mismatch:
            raise ValueError(f"World4Drive P1 direct-parent lineage mismatch: {lineage_mismatch}")
    return payload.get("planner")


def _restore_world4drive_read_only_runtime(
    *,
    config: Any,
    binding: CvoiWorld4DriveRuntimeBinding,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    token_ae: Optional[torch.nn.Module],
    planner_p0: torch.nn.Module,
    planner_p1: Optional[torch.nn.Module],
    multiview_fusion: Optional[torch.nn.Module],
    embed_dim: int,
    device: torch.device,
) -> PrefixDualValueModel:
    """Prevalidate every read-only artifact, then mutate the frozen modules once."""

    world = _read_world4drive_checkpoint(binding.world_model_checkpoint, name="world model")
    encoder_state = None
    for key in (getattr(config.meta, "context_encoder_key", "encoder"), "encoder", "target_encoder"):
        if key in world:
            encoder_state = world[key]
            break
    if encoder_state is None or "predictor" not in world:
        raise ValueError("World4Drive world-model checkpoint requires encoder and predictor states")
    encoder_core, encoder_state = _world4drive_module_state(encoder_state, module=encoder, name="encoder")
    predictor_core, predictor_state = _world4drive_module_state(
        world["predictor"],
        module=predictor,
        name="predictor",
    )
    fusion_core = None
    fusion_state = None
    if bool(getattr(config.multiview, "enabled", False)):
        if multiview_fusion is None or "multiview_fusion" not in world:
            raise ValueError("World4Drive multiview world checkpoint requires fusion weights")
        fusion_core, fusion_state = _world4drive_module_state(
            world["multiview_fusion"],
            module=multiview_fusion,
            name="multiview fusion",
        )
    elif multiview_fusion is not None:
        raise ValueError("World4Drive received multiview fusion while multiview.enabled=false")

    if token_ae is None:
        raise ValueError("World4Drive direct runtime requires the bound TokenAE-128 module")
    token_ae_payload = _read_world4drive_checkpoint(binding.token_ae_checkpoint, name="TokenAE")
    token_ae_state = token_ae_payload.get("token_ae", token_ae_payload)
    token_ae_core, token_ae_state = _world4drive_module_state(
        token_ae_state,
        module=token_ae,
        name="TokenAE",
    )

    p0_payload = _read_world4drive_checkpoint(binding.unguided_planner_checkpoint, name="P0 Planner")
    p0_state = _validate_world4drive_planner_payload(p0_payload, binding=binding, p0=True, config=config)
    p0_core, p0_state = _world4drive_module_state(p0_state, module=planner_p0, name="P0 Planner")
    p1_core = None
    p1_state = None
    if binding.controller_lineage == "value_guided":
        if planner_p1 is None or binding.guided_planner_checkpoint is None:
            raise ValueError("Value-guided World4Drive runtime requires a P1 Planner")
        p1_payload = _read_world4drive_checkpoint(binding.guided_planner_checkpoint, name="P1 Planner")
        p1_state = _validate_world4drive_planner_payload(p1_payload, binding=binding, p0=False, config=config)
        p1_core, p1_state = _world4drive_module_state(p1_state, module=planner_p1, name="P1 Planner")
    elif planner_p1 is not None:
        raise ValueError("P0 World4Drive runtime must not initialize a P1 Planner")

    value_payload = read_world4drive_value_checkpoint(binding.dual_value_checkpoint)
    validate_world4drive_value_lineage(value_payload, binding=binding, embed_dim=embed_dim)
    value_model = load_cvoi_world4drive_value_model(binding, embed_dim=embed_dim, device=torch.device("cpu"))

    encoder_core.load_state_dict(encoder_state, strict=True)
    predictor_core.load_state_dict(predictor_state, strict=True)
    token_ae_core.load_state_dict(token_ae_state, strict=True)
    p0_core.load_state_dict(p0_state, strict=True)
    if p1_core is not None and p1_state is not None:
        p1_core.load_state_dict(p1_state, strict=True)
    if fusion_core is not None and fusion_state is not None:
        fusion_core.load_state_dict(fusion_state, strict=True)
    for module in (encoder, predictor, token_ae, planner_p0, planner_p1, multiview_fusion, value_model):
        _freeze_module(module)
    return value_model.to(device=device)


def _validate_world4drive_runtime_binding(
    config: Any,
    binding: CvoiWorld4DriveRuntimeBinding,
) -> None:
    """Validate the public read-only binding against the ordinary model config."""

    if not isinstance(binding, CvoiWorld4DriveRuntimeBinding):
        raise TypeError("binding must be a CvoiWorld4DriveRuntimeBinding")
    if bool(getattr(getattr(config, "cvoi", None), "enabled", False)):
        raise ValueError("World4Drive direct base config must not enable the training CVoI protocol")
    expected_binding = {
        "protocol_version": "world4drive_evaluation_v1",
        "stage": "evaluation",
        "max_horizon": 3,
        "tokens_per_frame": 128,
        "compute_costs": (0.0, 1.0, 2.0, 3.0),
        "controller_batch_size": 1,
        "guidance_steps": 2,
        "guidance_objective": "last",
        "timestep_sec": 0.5,
    }
    mismatched_binding = {
        field: (getattr(binding, field), expected)
        for field, expected in expected_binding.items()
        if getattr(binding, field) != expected or type(getattr(binding, field)) is not type(expected)
    }
    if mismatched_binding:
        raise ValueError(f"World4Drive runtime binding semantics mismatch: {mismatched_binding}")
    expected_controller = "p0_controller" if binding.lineage == "p0_controller" else "value_guided"
    if binding.lineage not in {"p0_controller", "real_only_value", "real_cf_value"}:
        raise ValueError(f"unsupported World4Drive runtime lineage: {binding.lineage!r}")
    if binding.controller_lineage != expected_controller:
        raise ValueError("World4Drive runtime binding controller lineage mismatch")
    if expected_controller == "p0_controller":
        if binding.field_checkpoint is not None or binding.guided_planner_checkpoint is not None:
            raise ValueError("P0 World4Drive runtime binding forbids Field/P1 checkpoints")
    elif not binding.field_checkpoint or not binding.guided_planner_checkpoint:
        raise ValueError("Value-guided World4Drive runtime binding requires Field/P1 checkpoints")

    runtime_ae = getattr(getattr(config, "meta", None), "ae_checkpoint", None)
    if not isinstance(runtime_ae, str) or not runtime_ae.strip():
        raise ValueError("World4Drive direct base config requires meta.ae_checkpoint")
    if Path(runtime_ae).expanduser().resolve(strict=False) != Path(binding.token_ae_checkpoint).expanduser().resolve(
        strict=False
    ):
        raise ValueError("World4Drive meta.ae_checkpoint must match the bound TokenAE artifact")
    if str(getattr(getattr(config, "model", None), "backbone", "")) != "vjepa_img_encoder":
        raise ValueError("World4Drive direct runtime requires model.backbone='vjepa_img_encoder'")
    expected_load_policy = {
        "load_encoder": True,
        "load_predictor": False,
        "load_planner": False,
        "load_seg": False,
    }
    load_policy_mismatch = {
        field: (getattr(config.meta, field, None), expected)
        for field, expected in expected_load_policy.items()
        if getattr(config.meta, field, None) is not expected
    }
    if load_policy_mismatch:
        raise ValueError(f"World4Drive direct runtime model load policy mismatch: {load_policy_mismatch}")
    token_ae = getattr(config, "token_ae", None)
    if not bool(getattr(token_ae, "enabled", False)) or int(getattr(token_ae, "num_latent_tokens", -1)) != 128:
        raise ValueError("World4Drive direct runtime requires enabled TokenAE-128")
    if str(getattr(getattr(config, "train", None), "predictor_type", "")).lower() != "ac_transformer":
        raise ValueError("World4Drive direct runtime requires train.predictor_type='ac_transformer'")
    if not bool(getattr(config.train, "predictor_inference_consistent", False)) or bool(
        getattr(config.train, "predictor_no_aux_input", False)
    ):
        raise ValueError("World4Drive direct runtime requires the inference-consistent Predictor auxiliary contract")
    if str(getattr(getattr(config, "planner", None), "z_ar_mode", "")) != "full":
        raise ValueError("World4Drive direct runtime requires planner.z_ar_mode='full'")
    fps = getattr(getattr(config, "data", None), "fps", None)
    if type(fps) not in (int, float) or isinstance(fps, bool) or 1.0 / float(fps) != binding.timestep_sec:
        raise ValueError("World4Drive direct runtime data.fps must match binding.timestep_sec=0.5")
    future_steps = int(config.data.num_target_frames) - int(config.train.num_observed_frames)
    if future_steps < binding.max_horizon:
        raise ValueError(
            f"World4Drive direct runtime requires at least H={binding.max_horizon} future steps, got {future_steps}"
        )
    if float(getattr(config.planner, "diff_dt", float("nan"))) != binding.timestep_sec:
        raise ValueError("World4Drive direct runtime planner.diff_dt must match binding.timestep_sec=0.5")
    guidance = getattr(config, "value_guidance", None)
    guidance_expected = {
        "enabled": True,
        "steps": binding.guidance_steps,
        "objective": binding.guidance_objective,
        "step_size": 0.05,
        "max_delta_norm": 0.25,
        "detach_output": True,
    }
    guidance_mismatch = {
        field: (getattr(guidance, field, None), expected)
        for field, expected in guidance_expected.items()
        if getattr(guidance, field, None) != expected or type(getattr(guidance, field, None)) is not type(expected)
    }
    if guidance_mismatch:
        raise ValueError(f"World4Drive direct runtime Guidance semantics mismatch: {guidance_mismatch}")


def _build_world4drive_private_runtime_config(
    config: Any,
    binding: CvoiWorld4DriveRuntimeBinding,
) -> Any:
    """Create the private legacy-shaped adapter consumed by retained inference code."""

    _validate_world4drive_runtime_binding(config, binding)
    runtime_config = copy.deepcopy(config)
    runtime_config.meta.pretrain_checkpoint = None
    runtime_config.meta.pretrain_checkpoint_full = None
    runtime_config.meta.predictor_checkpoint = None
    cvoi = runtime_config.cvoi
    cvoi.enabled = True
    cvoi.protocol_version = "legacy_v1"
    cvoi.schema = "cvoi_dual_value_v1"
    cvoi.stage = "evaluation"
    cvoi.evaluation_mode = "controller"
    cvoi.field_warmup_domain = "real_cf" if binding.lineage == "real_cf_value" else "real"
    cvoi.guidance_steps = binding.guidance_steps
    cvoi.guidance_objective = binding.guidance_objective
    cvoi.max_horizon = binding.max_horizon
    cvoi.rollout_horizons = list(range(binding.max_horizon + 1))
    cvoi.controller_batch_size = binding.controller_batch_size
    cvoi.controller_lineage = binding.controller_lineage
    cvoi.compute_costs = list(binding.compute_costs)
    cvoi.seed_planner_checkpoint = binding.unguided_planner_checkpoint
    cvoi.unguided_planner_checkpoint = binding.unguided_planner_checkpoint
    cvoi.field_checkpoint = binding.field_checkpoint
    cvoi.guided_planner_checkpoint = binding.guided_planner_checkpoint
    cvoi.dual_value_checkpoint = binding.dual_value_checkpoint
    cvoi.oracle_path = binding.oracle_path
    cvoi.gate_checkpoint = binding.gate_checkpoint
    cvoi.world_model_checkpoint = binding.world_model_checkpoint
    cvoi.token_ae_checkpoint = binding.token_ae_checkpoint
    cvoi.tokens_per_frame = binding.tokens_per_frame
    cvoi.ablation_signature = _World4DriveAblationIdentity(**_expected_world4drive_planner_ablation(binding, p0=False))
    runtime_config._world4drive_runtime_binding = binding
    return runtime_config


def _create_navsim_cvoi_runtime(
    *,
    config: Any,
    binding: Optional[CvoiWorld4DriveRuntimeBinding],
    device: torch.device,
    restore_policy: str,
    _allow_cpu_for_tests: bool = False,
    _navsim_e120_rng_isolated: bool = False,
) -> NavSimCvoiProductionModelRuntime:
    """Build and strictly restore the deployment-equivalent frozen runtime."""

    if restore_policy not in {"navsim_e120_manual", "world4drive_read_only"}:
        raise ValueError(f"unsupported NavSim CVoI restore policy: {restore_policy!r}")
    world4drive_read_only = restore_policy == "world4drive_read_only"
    if world4drive_read_only != (binding is not None):
        raise ValueError("World4Drive restore policy and runtime binding must be provided together")
    device = torch.device(device)
    if device.type != "cuda" and not _allow_cpu_for_tests:
        raise RuntimeError("NavSim CVoI production model runtime requires CUDA")
    if not bool(getattr(config.cvoi, "enabled", False)):
        raise ValueError("NavSim CVoI model runtime requires cvoi.enabled=true")
    stage = str(config.cvoi.stage)
    if stage == "gate_distillation":
        raise ValueError("NavSim CVoI Gate consumes only its Oracle artifact and must not load a model policy pair")
    protocol_version = str(getattr(config.cvoi, "protocol_version", "legacy_v1"))
    if not world4drive_read_only and protocol_version != _NAVSIM_E120_PROTOCOL:
        raise ValueError(
            "retained NavSim CVoI model runtime supports only direct NavSim-e120 checkpoints; "
            f"got protocol={protocol_version!r}"
        )
    controller_lineage = str(getattr(config.cvoi, "controller_lineage", "value_guided"))
    if (
        protocol_version == _NAVSIM_E120_PROTOCOL
        and stage == "stop_calibrated"
        and controller_lineage != "value_guided"
    ):
        raise ValueError("direct NavSim-e120 Stop requires cvoi.controller_lineage='value_guided'")
    if not bool(getattr(config.train, "predictor_inference_consistent", False)):
        raise ValueError("NavSim CVoI runtime requires predictor_inference_consistent=true")
    if bool(getattr(config.train, "predictor_no_aux_input", False)):
        raise ValueError(
            "NavSim CVoI runtime requires train.predictor_no_aux_input=false "
            "when predictor_inference_consistent=true"
        )
    if str(getattr(config.planner, "z_ar_mode", "")) != "full":
        raise ValueError("NavSim CVoI runtime requires planner.z_ar_mode='full'")
    max_horizon = int(config.cvoi.max_horizon)
    expected_max_horizon = FORMAL_V2_NAVSIM_MAX_HORIZON if protocol_version == _NAVSIM_E120_PROTOCOL else 3
    if max_horizon != expected_max_horizon:
        raise ValueError(
            f"NavSim CVoI runtime protocol={protocol_version!r} requires controller "
            f"H={expected_max_horizon}, got {max_horizon}"
        )
    num_planner_poses = int(config.data.num_target_frames) - int(config.train.num_observed_frames)
    if num_planner_poses < max_horizon:
        raise ValueError(f"CVoI planner raw horizon {num_planner_poses} must cover controller H={max_horizon}")

    if protocol_version == _NAVSIM_E120_PROTOCOL and not _navsim_e120_rng_isolated:
        rng_devices: list[int] = []
        if device.type == "cuda":
            rng_devices.append(device.index if device.index is not None else torch.cuda.current_device())
        with torch.random.fork_rng(devices=rng_devices):
            return _create_navsim_cvoi_runtime(
                config=config,
                binding=binding,
                device=device,
                restore_policy=restore_policy,
                _allow_cpu_for_tests=_allow_cpu_for_tests,
                _navsim_e120_rng_isolated=True,
            )

    if protocol_version == _NAVSIM_E120_PROTOCOL:
        encoder, target_encoder = init_encoder_for_full_state_warmstart(config, device)
    else:
        encoder, target_encoder = init_encoder(config, device)
    del target_encoder
    embed_dim = get_encoder_embed_dim(encoder)
    raw_tokens_override, predictor_img_size_override = resolve_main_predictor_runtime_overrides(config, encoder)
    predictor_init_kwargs = {}
    if world4drive_read_only:
        predictor_init_kwargs["_checkpoint_weights_only"] = True
        predictor_init_kwargs["_defer_token_ae_state_load"] = True
    predictor, token_ae, tokens_per_frame, runtime_normalize_reps = init_predictor_runtime_with_token_ae(
        config,
        device=device,
        encoder_embed_dim=embed_dim,
        raw_tokens_per_frame_override=raw_tokens_override,
        predictor_img_size_override=predictor_img_size_override,
        **predictor_init_kwargs,
    )
    if bool(getattr(config.token_ae, "enabled", False)):
        configured_tokens = int(config.token_ae.num_latent_tokens)
        if token_ae is None or int(tokens_per_frame) != configured_tokens:
            raise ValueError(
                "CVoI TokenAE runtime mismatch: "
                f"configured={configured_tokens}, loaded_tokens_per_frame={tokens_per_frame}, "
                f"module_present={token_ae is not None}"
            )

    raw_tokens_per_frame = resolve_main_encoder_raw_tokens_per_frame(config, encoder)
    multiview_fusion = _init_multiview_fusion(
        config,
        embed_dim=embed_dim,
        raw_tokens_per_frame=raw_tokens_per_frame,
        device=device,
    )
    planner_p0 = init_planner(
        config,
        embed_dim,
        device,
        num_poses=num_planner_poses,
        tokens_per_frame_override=int(tokens_per_frame),
    )
    if planner_p0 is None:
        raise ValueError("NavSim CVoI runtime requires learned planner P0")
    uses_p1 = stage in _P1_STAGES and controller_lineage == "value_guided"
    planner_p1 = None
    predictor_p1 = None
    if uses_p1:
        planner_p1 = init_planner(
            config,
            embed_dim,
            device,
            num_poses=num_planner_poses,
            tokens_per_frame_override=int(tokens_per_frame),
        )
        if planner_p1 is None:
            raise ValueError("NavSim CVoI guided stages require learned planner P1")
        if protocol_version == _NAVSIM_E120_PROTOCOL:
            predictor_p1, token_ae_p1, p1_tokens_per_frame, p1_runtime_normalize_reps = (
                init_predictor_runtime_with_token_ae(
                    config,
                    device=device,
                    encoder_embed_dim=embed_dim,
                    raw_tokens_per_frame_override=raw_tokens_override,
                    predictor_img_size_override=predictor_img_size_override,
                )
            )
            if token_ae_p1 is not None:
                raise ValueError("NavSim-e120 P1 policy pair forbids a separate TokenAE")
            if int(p1_tokens_per_frame) != int(tokens_per_frame) or bool(p1_runtime_normalize_reps) != bool(
                runtime_normalize_reps
            ):
                raise ValueError("NavSim-e120 P0/P1 Predictor runtime interfaces must match exactly")
            if predictor_p1 is predictor:
                raise ValueError("NavSim-e120 P0 and P1 policy bundles require distinct Predictor instances")
        else:
            predictor_p1 = predictor

    if world4drive_read_only:
        if binding is None:
            raise RuntimeError("World4Drive read-only runtime lost its immutable binding")
        dual_value_model = _restore_world4drive_read_only_runtime(
            config=config,
            binding=binding,
            encoder=encoder,
            predictor=predictor,
            token_ae=token_ae,
            planner_p0=planner_p0,
            planner_p1=planner_p1,
            multiview_fusion=multiview_fusion,
            embed_dim=embed_dim,
            device=device,
        )
        return NavSimCvoiProductionModelRuntime(
            config=config,
            device=device,
            encoder=encoder,
            predictor=predictor,
            predictor_p1=predictor_p1,
            token_ae=token_ae,
            planner_p0=planner_p0,
            planner_p1=planner_p1,
            dual_value_model=dual_value_model,
            multiview_fusion=multiview_fusion,
            embed_dim=embed_dim,
            tokens_per_frame=int(tokens_per_frame),
            runtime_normalize_reps=bool(runtime_normalize_reps),
            num_planner_poses=num_planner_poses,
        )

    warmstart = getattr(config.cvoi, "full_state_warmstart", None)
    if warmstart is None:
        raise ValueError("NavSim-e120 offline runtime requires cvoi.full_state_warmstart")
    apply_formal_v2_full_state_warmstart_direct(
        warmstart.source_checkpoint.path,
        warmstart.source_params_pretrain.path,
        {"encoder": encoder, "predictor": predictor, "planner": planner_p0},
    )
    results_root = resolve_cvoi_manual_full_results_root({"p0_handoff": config.cvoi.unguided_planner_checkpoint})
    _restore_navsim_e120_direct_policy_checkpoint(
        encoder=encoder,
        predictor=predictor,
        planner=planner_p0,
        path=config.cvoi.unguided_planner_checkpoint,
        expected_stage="p0",
        results_root=results_root,
    )
    if planner_p1 is not None and predictor_p1 is not None:
        guided_path = getattr(config.cvoi, "guided_planner_checkpoint", None)
        if not isinstance(guided_path, str) or not guided_path.strip():
            raise ValueError(f"cvoi.stage={stage!r} requires cvoi.guided_planner_checkpoint")
        resolve_cvoi_manual_full_results_root(
            {"p1_handoff": guided_path},
            expected_results_root=results_root,
        )
        _restore_navsim_e120_direct_policy_checkpoint(
            encoder=encoder,
            predictor=predictor_p1,
            planner=planner_p1,
            path=guided_path,
            expected_stage="p1",
            results_root=results_root,
        )
    dual_value_model = load_cvoi_dual_value_model(config, embed_dim=embed_dim, device=device)
    evaluation_mode = str(getattr(config.cvoi, "evaluation_mode", "controller"))
    requires_value = stage in _P1_STAGES and evaluation_mode != "p0_forced"
    if requires_value and dual_value_model is None:
        raise ValueError(f"cvoi.stage={stage!r} requires a lifecycle-valid dual-value model")

    return NavSimCvoiProductionModelRuntime(
        config=config,
        device=device,
        encoder=encoder,
        predictor=predictor,
        predictor_p1=predictor_p1,
        token_ae=token_ae,
        planner_p0=planner_p0,
        planner_p1=planner_p1,
        dual_value_model=dual_value_model,
        multiview_fusion=multiview_fusion,
        embed_dim=embed_dim,
        tokens_per_frame=int(tokens_per_frame),
        runtime_normalize_reps=bool(runtime_normalize_reps),
        num_planner_poses=num_planner_poses,
    )


def create_navsim_cvoi_model_runtime(
    *,
    config: Any,
    device: torch.device,
    _allow_cpu_for_tests: bool = False,
) -> NavSimCvoiProductionModelRuntime:
    """Build the retained manual NavSim-e120 runtime."""

    return _create_navsim_cvoi_runtime(
        config=config,
        binding=None,
        device=device,
        restore_policy="navsim_e120_manual",
        _allow_cpu_for_tests=_allow_cpu_for_tests,
    )


def create_navsim_cvoi_world4drive_runtime(
    *,
    config: Any,
    binding: CvoiWorld4DriveRuntimeBinding,
    device: torch.device,
    _allow_cpu_for_tests: bool = False,
) -> NavSimCvoiProductionModelRuntime:
    """Build one read-only World4Drive runtime without enabling public training CVoI."""

    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" and not _allow_cpu_for_tests:
        raise RuntimeError("NavSim CVoI production model runtime requires CUDA")
    runtime_config = _build_world4drive_private_runtime_config(config, binding)
    validate_cvoi_world4drive_gate(binding)
    return _create_navsim_cvoi_runtime(
        config=runtime_config,
        binding=binding,
        device=resolved_device,
        restore_policy="world4drive_read_only",
        _allow_cpu_for_tests=_allow_cpu_for_tests,
    )
