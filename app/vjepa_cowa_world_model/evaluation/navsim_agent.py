"""
VJEPAWorldModelAgent — 将 VJEPA 世界模型 (encoder + predictor + planner)
包装为 NavSim AbstractAgent，用于 PDMS / EPDMS 评估。

推理流程:
  AgentInput → VJEPAFeatureBuilder → video_clip + states + actions + extrinsics
    → Encoder → z [B, T*P, D]
    → Predictor (autoregressive rollout) → z_ar [B, pred*P, D]
    → Planner → K trajectories + confidences → 取最高置信度轨迹
    → 截断至 8 个 pose (4s, 0.5s interval) → Trajectory

使用方法:
  在 Hydra 配置中设置 checkpoint_path 和 training_config_path，
  初始化时自动加载 checkpoint 并构建模型。
"""

import copy
import hashlib
import inspect
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, List, Mapping, NamedTuple, Optional

import torch
import torch.nn as nn
import yaml
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from app.vjepa_cowa_world_model.evaluation.cvoi_direct_epdms import (
    CVOI_DIRECT_EPDMS_TRACE_SCHEMA,
    CVOI_DIRECT_EPDMS_TRACE_VERSION,
    CvoiDirectEpdmsProjection,
    load_cvoi_direct_epdms_projection,
    read_cvoi_direct_epdms_scenario_manifest,
    validate_cvoi_direct_epdms_scenario_token,
)
from app.vjepa_cowa_world_model.evaluation.cvoi_navsim_identity import decode_observation_key, decode_unsigned_seed
from app.vjepa_cowa_world_model.evaluation.navsim_feature_builder import VJEPAFeatureBuilder
from app.vjepa_cowa_world_model.models import PETRMultiViewFusion
from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.models.trajectory_value import TemporalTrajectoryValueHead
from app.vjepa_cowa_world_model.training import cvoi_manual_lineage
from app.vjepa_cowa_world_model.training.budget_control import (
    load_budget_controller_from_checkpoint,
    resolve_controller_budget_profile,
)
from app.vjepa_cowa_world_model.training.checkpoint import load_pretrained_checkpoint, load_state_dict_helper
from app.vjepa_cowa_world_model.training.config import (
    TrainingConfig,
    parse_training_config,
    resolve_main_encoder_raw_tokens_per_frame,
    resolve_proposal_num_time_steps,
    resolve_proposal_runtime_normalize_reps,
    resolve_proposal_tokens_per_frame,
)
from app.vjepa_cowa_world_model.training.cvoi_execution import (
    CvoiValueDtypeAdapter,
    common_random_numbers,
    cvoi_execution_autocast,
    cvoi_planner_inference_noise,
    cvoi_sample_seed,
    resolve_cvoi_evaluation_seed,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_runtime import (
    build_cvoi_direct_epdms_gate,
    read_cvoi_direct_epdms_gate_checkpoint,
    read_cvoi_direct_epdms_planner_checkpoint,
    read_formal_v2_navsim_e120_direct_checkpoint,
    resolve_cvoi_direct_epdms_artifact_identity,
    resolve_formal_v2_navsim_e120_selected_checkpoint,
    validate_formal_v2_navsim_e120_direct_checkpoint,
)
from app.vjepa_cowa_world_model.training.cvoi_manual_navtrain_oracle import (
    MANUAL_NAVTRAIN_POLICY_TRACE_SCHEMA,
    MANUAL_NAVTRAIN_SCORER_SCHEMA,
    ManualNavTrainScorerConfig,
    ManualOracleSource,
    read_manual_navtrain_scorer_config,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import NAVTRAIN_GATE_PROTOCOL_ID
from app.vjepa_cowa_world_model.training.cvoi_runtime import (
    apply_cvoi_planner_guidance,
    cvoi_enabled,
    load_cvoi_dual_value_model,
    load_cvoi_gate_for_evaluation,
    require_cvoi_stage_for_entry,
    validate_cvoi_sequential_runtime_config,
)
from app.vjepa_cowa_world_model.training.cvoi_value import read_cvoi_navsim_e120_direct_value_checkpoint
from app.vjepa_cowa_world_model.training.encoder_direct_validation import (
    adapt_encoder_direct_action_history_from_checkpoint,
    resolve_encoder_direct_action_history_frames,
)
from app.vjepa_cowa_world_model.training.latent_value_guidance import (
    apply_latent_value_guidance,
    cvoi_evaluation_guidance_steps,
)
from app.vjepa_cowa_world_model.training.models import (
    configure_vjepa_encoder_trainability,
    get_encoder_embed_dim,
    init_context_encoder_for_full_state_warmstart,
    init_encoder,
    init_planner,
    init_predictor_runtime_with_token_ae,
    init_proposal_encoder,
    resolve_main_predictor_runtime_overrides,
)
from app.vjepa_cowa_world_model.training.planner_anchor import build_ego_relative_diffusion_anchor
from app.vjepa_cowa_world_model.training.predictor_parallel import (
    forward_parallel_predictor,
    maybe_register_parallel_predictor_tokens,
    use_parallel_predictor,
)
from app.vjepa_cowa_world_model.training.predictor_stepping import (
    make_predictor_step_fn,
    predictor_autoregressive_rollout,
    validate_empty_future_planner_conditions,
)
from app.vjepa_cowa_world_model.training.runtimes.encoder_token_runtime import (
    init_encoder_direct_encoder,
    init_encoder_direct_planner,
    is_vjepa_img_encoder,
    resolve_action_history_dt,
    resolve_encoder_direct_tokens_per_frame,
)
from app.vjepa_cowa_world_model.training.runtimes.forward_runtime import ForwardRuntime
from app.vjepa_cowa_world_model.training.runtimes.latent_diffusion_runtime import (
    resolve_latent_dit_sampler_params,
    sample_latent_dit_predictor,
    use_latent_dit_predictor,
)
from app.vjepa_cowa_world_model.training.runtimes.refinement_runtime import (
    apply_stage3_refinement_input_gates,
    build_proposal_history,
    build_stage_predictor_rollout_fn,
    build_status_feature,
    call_planner_method,
    forward_frozen_proposal,
    freeze_module_eval,
    is_vjepa_proposal_encoder,
    load_frozen_proposal_encoder,
    load_frozen_proposal_provider,
    maybe_expand_manual_proposal,
    resolve_proposal_token_ae_module,
    select_observed_context_clips,
)
from app.vjepa_cowa_world_model.training.runtimes.sequential_rollout_runtime import run_sequential_rollout
from app.vjepa_cowa_world_model.training.runtimes.world_model_runtime import (
    build_parallel_predictor_timeline_inputs,
    build_predictor_timeline_inputs,
    enforce_cvoi_zero_future_aux,
    forward_main_context,
    forward_main_context_dual,
    resolve_main_timeline,
)
from app.vjepa_cowa_world_model.training.sequential_budget_control import (
    build_lambda_independent_sequential_gate_features,
    extract_prefix_gate_values,
)
from app.vjepa_cowa_world_model.training.value_planning import (
    score_trajectories_method1,
    value_planning_enabled,
    value_planning_method1_enabled,
)
from app.vjepa_cowa_world_model.utils import (
    build_observed_action_trajectory_history,
    prepare_inference_consistent_status_vector,
    prepare_status_feature,
    resolve_effective_planner_status_dim,
    resolve_planner_status_dim,
    resolve_planner_use_drive_command,
    select_best_trajectory,
)
from app.vjepa_cowa_world_model.utils.eval_determinism import deterministic_eval_rng


def _load_cvoi_formal_v2_navsim_e120_exact_state(
    module: nn.Module,
    state: Mapping[str, object],
    *,
    role: str,
) -> None:
    """Load one normalized e120 role with an exact runtime key/shape contract."""

    core = getattr(module, "module", module)
    if not isinstance(core, nn.Module):
        raise ValueError(f"NavSim-e120 {role} runtime must be a torch.nn.Module")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"NavSim-e120 {role} state must be a non-empty mapping")
    if any(type(key) is not str or not key or not torch.is_tensor(value) for key, value in state.items()):
        raise ValueError(f"NavSim-e120 {role} state must map non-empty string keys to tensors")
    expected = core.state_dict()
    actual_keys = set(state)
    expected_keys = set(expected)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches = sorted(
        key for key in actual_keys & expected_keys if tuple(state[key].shape) != tuple(expected[key].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            f"NavSim-e120 {role} state does not exactly match its runtime envelope: "
            f"missing={missing}, unexpected={unexpected}, shape_mismatches={shape_mismatches}"
        )
    incompatible = core.load_state_dict(dict(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"NavSim-e120 {role} strict load returned incompatible keys: {incompatible}")


logger = logging.getLogger(__name__)

# NavSim 评估: 4s horizon, 0.5s interval = 8 poses
NAVSIM_TRAJECTORY_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)
NAVSIM_NUM_POSES = 8
_FORWARD_MODES = {"stage12", "stage2", "stage3", "encoder_direct"}
_CVOI_MANUAL_ARTIFACT_ROLES = frozenset(
    {
        "p0_planner_checkpoint",
        "field_checkpoint",
        "calibration_checkpoint",
        "p1_planner_checkpoint",
        "stop_checkpoint",
    }
)
_CVOI_MANUAL_MODEL_ROLES = ("encoder", "predictor", "planner")


class _ManualNavTrainArtifacts(NamedTuple):
    """Structurally validated, proof-free handoff payloads."""

    p0_checkpoint: Mapping[str, object]
    p1_checkpoint: Mapping[str, object]
    field_architecture: Mapping[str, object]
    calibration_architecture: Mapping[str, object]
    stop_architecture: Mapping[str, object]


class _ManualNavTrainRuntimeContext(NamedTuple):
    """The complete immutable runtime authority needed by one manual scorer call."""

    policy_id: str
    lineage: str
    planner_stage: str
    scenario_manifest: object
    forced_horizon: int
    guidance_steps: int
    common_random_seed: int
    trace_output_dir: Path
    trace_output_dir_identity: tuple[int, int]
    p0_planner_checkpoint: Path
    field_checkpoint: Path
    calibration_checkpoint: Path
    p1_planner_checkpoint: Path
    stop_checkpoint: Path


class _DirectEpdmsArtifacts(NamedTuple):
    """Mode-minimal, structurally validated direct deployment payloads."""

    p0_checkpoint: Optional[Mapping[str, object]]
    calibration_checkpoint: Optional[Mapping[str, object]]
    p1_checkpoint: Optional[Mapping[str, object]]
    stop_checkpoint: Optional[Mapping[str, object]]
    gate_checkpoint: Optional[Mapping[str, object]]


class _DirectEpdmsRuntimeContext(NamedTuple):
    """Agent-local direct authority; no public YAML or Oracle path is retained."""

    projection: CvoiDirectEpdmsProjection
    artifacts: _DirectEpdmsArtifacts
    scenario_tokens_by_observation_key: Optional[Mapping[str, str]] = None
    evaluation_seed: Optional[int] = None
    trace_output_dir: Optional[Path] = None
    trace_output_dir_identity: Optional[tuple[int, int]] = None


def _cvoi_direct_state_sha256(state: Mapping[str, object], *, role: str) -> str:
    """Hash one CPU state without materializing a second full byte copy."""

    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"direct EPDMS {role} state must be a non-empty mapping")
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        if type(key) is not str or not key or not torch.is_tensor(tensor):
            raise ValueError(f"direct EPDMS {role} state must map non-empty string keys to tensors")
        metadata = json.dumps(
            {
                "key": key,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        byte_view = tensor.detach().cpu().reshape(-1).contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))
    return digest.hexdigest()


def _retain_cvoi_direct_planner_roles(
    payload: Mapping[str, object],
    *,
    retain_world_model: bool,
) -> Mapping[str, object]:
    """Drop optimizer/resume state before another ViT-G envelope is opened."""

    retained: Dict[str, object] = {
        "stage": payload["stage"],
        "lineage": dict(payload["lineage"]),
        "protocol_version": payload["protocol_version"],
        "role_state_shapes": {role: dict(shapes) for role, shapes in payload["role_state_shapes"].items()},
        "encoder_sha256": _cvoi_direct_state_sha256(payload["encoder"], role="encoder"),
        "planner": dict(payload["planner"]),
    }
    if retain_world_model:
        retained["encoder"] = dict(payload["encoder"])
        retained["predictor"] = dict(payload["predictor"])
    return MappingProxyType(retained)


_DIRECT_FULL_HANDOFF_NAME_BY_ARTIFACT_FIELD = {
    "p0_planner_checkpoint_path": "p0_handoff",
    "calibration_checkpoint_path": "calibration_handoff",
    "p1_planner_checkpoint_path": "p1_handoff",
    "stop_checkpoint_path": "stop_handoff",
    "gate_checkpoint_path": "gate_handoff",
}


def _direct_full_results_root(projection: CvoiDirectEpdmsProjection) -> Path:
    handoffs = {
        handoff_name: path
        for field_name, handoff_name in _DIRECT_FULL_HANDOFF_NAME_BY_ARTIFACT_FIELD.items()
        if (path := getattr(projection, field_name)) is not None
    }
    return cvoi_manual_lineage.resolve_cvoi_manual_full_results_root(handoffs)


def _direct_epdms_artifact_paths(projection: CvoiDirectEpdmsProjection) -> Dict[str, Path]:
    """Validate the exact handoff paths exposed by one effective mode."""

    if not isinstance(projection, CvoiDirectEpdmsProjection):
        raise TypeError("direct EPDMS projection must be CvoiDirectEpdmsProjection")
    mode = projection.evaluation_mode
    identity = resolve_cvoi_direct_epdms_artifact_identity(
        projection.branch,
        evaluation_mode=mode,
    )
    full_results_root = _direct_full_results_root(projection) if projection.branch == "full" else None
    value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
        phase="guided_planner",
        branch_id=identity.p1_branch_id,
        full_results_root=full_results_root,
    )
    expected: Dict[str, Path] = {}
    if mode in {"controller", "p0_forced"}:
        p0_root = (
            full_results_root if full_results_root is not None else cvoi_manual_lineage.CVOI_MANUAL_FULL_RESULTS_ROOT
        )
        expected["p0_planner_checkpoint_path"] = cvoi_manual_lineage.derive_cvoi_manual_full_handoffs(p0_root)[
            "p0_handoff"
        ]
    if mode in {"controller", "p1_field_forced"}:
        expected.update(
            {
                "calibration_checkpoint_path": value_lineage.calibration_handoff,
                "p1_planner_checkpoint_path": value_lineage.p1_handoff,
            }
        )
    if mode == "controller":
        gate_root = (
            full_results_root
            if projection.branch == "full"
            else cvoi_manual_lineage.CVOI_MANUAL_ABLATION_RESULTS_ROOT / projection.branch
        )
        if gate_root is None:
            raise RuntimeError("direct EPDMS Full results root was not resolved")
        expected.update(
            {
                "stop_checkpoint_path": value_lineage.stop_handoff,
                "gate_checkpoint_path": gate_root / "handoff/gate.pt",
            }
        )

    artifact_fields = {
        "p0_planner_checkpoint_path",
        "calibration_checkpoint_path",
        "p1_planner_checkpoint_path",
        "stop_checkpoint_path",
        "gate_checkpoint_path",
    }
    actual_present = {field_name for field_name in artifact_fields if getattr(projection, field_name) is not None}
    if actual_present != set(expected):
        raise ValueError(
            f"direct EPDMS {mode} artifact fields mismatch: "
            f"missing={sorted(set(expected) - actual_present)}, "
            f"unexpected={sorted(actual_present - set(expected))}"
        )
    drifted = {
        field_name: (getattr(projection, field_name), expected_path)
        for field_name, expected_path in expected.items()
        if getattr(projection, field_name) != expected_path
    }
    if drifted:
        raise ValueError(f"direct EPDMS artifact paths differ from the manual handoff authority: {drifted}")

    if mode == "controller":
        if projection.horizon is not None or projection.guidance_steps != 2:
            raise ValueError("direct EPDMS controller requires horizon=None and guidance_steps=2")
        if projection.gate_feature_mode != identity.gate_feature_mode:
            raise ValueError("direct EPDMS controller Gate feature mode differs from its branch identity")
        if (
            type(projection.oracle_sha256) is not str
            or re.fullmatch(r"[0-9a-f]{64}", projection.oracle_sha256) is None
        ):
            raise ValueError("direct EPDMS controller requires one projected Oracle SHA-256")
    elif mode == "p0_forced":
        if projection.horizon != 0 or projection.guidance_steps != 0:
            raise ValueError("direct EPDMS p0_forced requires H0/K0")
        if projection.gate_feature_mode is not None or projection.oracle_sha256 is not None:
            raise ValueError("direct EPDMS p0_forced forbids Gate and Oracle identity")
    elif mode == "p1_field_forced":
        if type(projection.horizon) is not int or not 1 <= projection.horizon <= 4:
            raise ValueError("direct EPDMS p1_field_forced requires one horizon in H1--H4")
        if projection.guidance_steps != 2:
            raise ValueError("direct EPDMS p1_field_forced requires K2")
        if projection.gate_feature_mode is not None or projection.oracle_sha256 is not None:
            raise ValueError("direct EPDMS p1_field_forced forbids Gate and Oracle identity")
    else:
        raise ValueError(f"unsupported direct EPDMS evaluation mode: {mode!r}")

    resolved_identities: Dict[Path, str] = {}
    for field_name, path in expected.items():
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(f"direct EPDMS artifact does not exist: {path}") from error
        previous = resolved_identities.get(resolved)
        if previous is not None:
            raise ValueError(f"direct EPDMS artifacts {previous!r} and {field_name!r} alias {resolved}")
        resolved_identities[resolved] = field_name
    return expected


def _read_cvoi_direct_epdms_artifacts(
    projection: CvoiDirectEpdmsProjection,
) -> _DirectEpdmsArtifacts:
    """Read and cross-check only the artifacts visible in one projection."""

    paths = _direct_epdms_artifact_paths(projection)
    full_results_root = _direct_full_results_root(projection) if projection.branch == "full" else None
    identity = resolve_cvoi_direct_epdms_artifact_identity(
        projection.branch,
        evaluation_mode=projection.evaluation_mode,
    )
    p0_checkpoint = None
    p1_checkpoint = None
    calibration_checkpoint = None
    stop_checkpoint = None
    gate_checkpoint = None

    p0_path = paths.get("p0_planner_checkpoint_path")
    if p0_path is not None:
        resolved = resolve_formal_v2_navsim_e120_selected_checkpoint(
            p0_path,
            results_root=(
                full_results_root
                if full_results_root is not None
                else cvoi_manual_lineage.CVOI_MANUAL_FULL_RESULTS_ROOT
            ),
            stage="p0",
        )
        p0_payload = read_cvoi_direct_epdms_planner_checkpoint(
            resolved,
            expected_stage="p0",
            expected_branch_id=identity.p0_branch_id,
        )
        p0_checkpoint = _retain_cvoi_direct_planner_roles(
            p0_payload,
            retain_world_model=projection.evaluation_mode == "p0_forced",
        )
        del p0_payload

    p1_path = paths.get("p1_planner_checkpoint_path")
    if p1_path is not None:
        value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_by_checkpoint_branch(
            phase="guided_planner",
            branch_id=identity.p1_branch_id,
            full_results_root=full_results_root,
        )
        resolved = resolve_formal_v2_navsim_e120_selected_checkpoint(
            p1_path,
            results_root=value_lineage.result_root,
            stage="p1",
        )
        p1_payload = read_cvoi_direct_epdms_planner_checkpoint(
            resolved,
            expected_stage="p1",
            expected_branch_id=identity.p1_branch_id,
        )
        p1_checkpoint = _retain_cvoi_direct_planner_roles(
            p1_payload,
            retain_world_model=True,
        )
        del p1_payload

    calibration_path = paths.get("calibration_checkpoint_path")
    if calibration_path is not None:
        calibration_checkpoint = read_cvoi_navsim_e120_direct_value_checkpoint(
            calibration_path,
            required_phase="field_calibrated",
            required_branch_id=identity.calibration_branch_id,
            map_location="cpu",
        )

    stop_path = paths.get("stop_checkpoint_path")
    if stop_path is not None:
        if identity.stop_branch_id is None:
            raise RuntimeError("direct EPDMS controller Stop identity was not resolved")
        stop_checkpoint = read_cvoi_navsim_e120_direct_value_checkpoint(
            stop_path,
            required_phase="stop_calibrated",
            required_branch_id=identity.stop_branch_id,
            map_location="cpu",
        )

    gate_path = paths.get("gate_checkpoint_path")
    if gate_path is not None:
        if projection.oracle_sha256 is None or projection.gate_feature_mode is None:
            raise RuntimeError("direct EPDMS controller Gate identity was not projected")
        gate_checkpoint = read_cvoi_direct_epdms_gate_checkpoint(
            gate_path,
            branch=projection.branch,
            oracle_sha256=projection.oracle_sha256,
            gate_feature_mode=projection.gate_feature_mode,
        )

    if p0_checkpoint is not None and p1_checkpoint is not None:
        if p0_checkpoint["role_state_shapes"] != p1_checkpoint["role_state_shapes"]:
            raise ValueError("direct EPDMS P0/P1 role architectures differ")
        if p0_checkpoint["encoder_sha256"] != p1_checkpoint["encoder_sha256"]:
            raise ValueError("direct EPDMS P0/P1 encoder states differ")
    if calibration_checkpoint is not None and stop_checkpoint is not None:
        if calibration_checkpoint["architecture"] != stop_checkpoint["architecture"]:
            raise ValueError("direct EPDMS Calibration/Stop Value architectures differ")

    return _DirectEpdmsArtifacts(
        p0_checkpoint=p0_checkpoint,
        calibration_checkpoint=calibration_checkpoint,
        p1_checkpoint=p1_checkpoint,
        stop_checkpoint=stop_checkpoint,
        gate_checkpoint=gate_checkpoint,
    )


def _build_cvoi_direct_epdms_training_config(
    source_config: TrainingConfig,
    projection: CvoiDirectEpdmsProjection,
) -> TrainingConfig:
    """Build a private runtime view after the training YAML passed its own contract."""

    if not isinstance(projection, CvoiDirectEpdmsProjection):
        raise TypeError("direct EPDMS projection must be CvoiDirectEpdmsProjection")
    source_cvoi = getattr(source_config, "cvoi", None)
    if (
        source_cvoi is None
        or source_cvoi.protocol_version != "formal_v2_navsim_e120_h4_v3"
        or source_cvoi.stage != "guided_planner"
    ):
        raise ValueError("direct EPDMS training_config_path must contain one parsed NavSim-e120 guided_planner config")
    identity = resolve_cvoi_direct_epdms_artifact_identity(
        projection.branch,
        evaluation_mode=projection.evaluation_mode,
    )
    signature = source_cvoi.ablation_signature
    full_results_root = _direct_full_results_root(projection) if projection.branch == "full" else None
    source_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage(
        signature,
        stage="guided_planner",
        full_results_root=full_results_root,
    )
    if source_lineage.checkpoint_branch_id("guided_planner") != identity.p1_branch_id:
        raise ValueError(
            "direct EPDMS source training config branch differs from the projected P1 lineage: "
            f"expected={identity.p1_branch_id!r}, "
            f"actual={source_lineage.checkpoint_branch_id('guided_planner')!r}"
        )

    runtime_config = copy.deepcopy(source_config)
    cvoi = runtime_config.cvoi
    artifact_fields = (
        "seed_planner_checkpoint",
        "unguided_planner_checkpoint",
        "field_checkpoint",
        "guided_planner_checkpoint",
        "dual_value_checkpoint",
        "oracle_path",
        "gate_checkpoint",
        "output_checkpoint",
        "world_model_checkpoint",
        "token_ae_checkpoint",
    )
    for field_name in artifact_fields:
        setattr(cvoi, field_name, None)
    cvoi.stage = "evaluation"
    cvoi.evaluation_mode = projection.evaluation_mode
    cvoi.controller_lineage = "p0_controller" if projection.evaluation_mode == "p0_forced" else "value_guided"
    cvoi.guidance_steps = projection.guidance_steps
    if projection.p0_planner_checkpoint_path is not None:
        cvoi.unguided_planner_checkpoint = str(projection.p0_planner_checkpoint_path)
    if projection.calibration_checkpoint_path is not None:
        cvoi.field_checkpoint = str(projection.calibration_checkpoint_path)
    if projection.p1_planner_checkpoint_path is not None:
        cvoi.guided_planner_checkpoint = str(projection.p1_planner_checkpoint_path)
    if projection.stop_checkpoint_path is not None:
        cvoi.dual_value_checkpoint = str(projection.stop_checkpoint_path)
    if projection.gate_checkpoint_path is not None:
        cvoi.gate_checkpoint = str(projection.gate_checkpoint_path)
    validate_cvoi_sequential_runtime_config(runtime_config)
    return runtime_config


def _validate_manual_planner_payload(
    payload: object,
    *,
    expected_stage: str,
    expected_branch_id: str,
) -> Mapping[str, object]:
    """Recheck the direct reader's role/key/shape contract at the Agent boundary."""

    if not isinstance(payload, Mapping):
        raise ValueError("manual NavTrain Planner checkpoint must be a mapping")
    lineage = payload.get("lineage")
    if payload.get("stage") != expected_stage or not isinstance(lineage, Mapping):
        raise ValueError(f"manual NavTrain Planner checkpoint must have stage {expected_stage!r}")
    if lineage.get("stage") != expected_stage or lineage.get("branch_id") != expected_branch_id:
        raise ValueError(
            "manual NavTrain Planner lineage must be " f"stage={expected_stage!r}, branch_id={expected_branch_id!r}"
        )
    role_shapes = payload.get("role_state_shapes")
    if not isinstance(role_shapes, Mapping) or set(role_shapes) != set(_CVOI_MANUAL_MODEL_ROLES):
        raise ValueError(f"manual NavTrain Planner roles must be exactly {_CVOI_MANUAL_MODEL_ROLES!r}")
    for role in _CVOI_MANUAL_MODEL_ROLES:
        state = payload.get(role)
        declared = role_shapes[role]
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"manual NavTrain Planner roles require a non-empty {role} mapping")
        if not isinstance(declared, Mapping) or set(declared) != set(state):
            raise ValueError(f"manual NavTrain Planner {role} role keys differ from role_state_shapes")
        for key, tensor in state.items():
            if type(key) is not str or not key or not torch.is_tensor(tensor):
                raise ValueError(f"manual NavTrain Planner {role} role must map string keys to tensors")
            shape = declared[key]
            if not isinstance(shape, list) or tuple(shape) != tuple(tensor.shape):
                raise ValueError(f"manual NavTrain Planner {role}.{key} shape differs from role_state_shapes")
    return payload


def _read_manual_navtrain_artifacts(
    artifacts: Mapping[str, Path],
    *,
    expected_lineage: str,
) -> _ManualNavTrainArtifacts:
    """Read and cross-check the five ordinary handoff files before model construction."""

    if not isinstance(artifacts, Mapping) or set(artifacts) != _CVOI_MANUAL_ARTIFACT_ROLES:
        raise ValueError("manual NavTrain artifacts must contain exactly five direct handoff roles")
    if any(not isinstance(path, Path) or not path.is_absolute() for path in artifacts.values()):
        raise ValueError("manual NavTrain artifact paths must be absolute Path values")

    if type(expected_lineage) is not str or expected_lineage not in {"p1_full", "p1_no_cf"}:
        raise ValueError(
            "manual NavTrain artifacts support only the retained p1_full and p1_no_cf Stop/Oracle lineages"
        )
    value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_from_artifacts(
        artifacts,
        phase="guided_planner",
        branch_id=expected_lineage,
    )
    expected_artifacts = {
        "p0_planner_checkpoint": value_lineage.p0_handoff,
        "field_checkpoint": value_lineage.field_handoff,
        "calibration_checkpoint": value_lineage.calibration_handoff,
        "p1_planner_checkpoint": value_lineage.p1_handoff,
        "stop_checkpoint": value_lineage.stop_handoff,
    }
    if dict(artifacts) != expected_artifacts:
        drifted = {
            role: (artifacts.get(role), expected_path)
            for role, expected_path in expected_artifacts.items()
            if artifacts.get(role) != expected_path
        }
        raise ValueError(f"manual NavTrain artifacts differ from the fixed {expected_lineage} authority: {drifted}")

    p0_results_root = value_lineage.p0_handoff.parent.parent
    resolved_p0 = resolve_formal_v2_navsim_e120_selected_checkpoint(
        artifacts["p0_planner_checkpoint"],
        results_root=p0_results_root,
        stage="p0",
    )
    resolved_p1 = resolve_formal_v2_navsim_e120_selected_checkpoint(
        artifacts["p1_planner_checkpoint"],
        results_root=value_lineage.result_root,
        stage="p1",
    )
    p0_checkpoint = _validate_manual_planner_payload(
        read_formal_v2_navsim_e120_direct_checkpoint(resolved_p0),
        expected_stage="p0",
        expected_branch_id="p0_uniform",
    )
    p1_checkpoint = _validate_manual_planner_payload(
        read_formal_v2_navsim_e120_direct_checkpoint(resolved_p1),
        expected_stage="p1",
        expected_branch_id=expected_lineage,
    )
    if p0_checkpoint["role_state_shapes"] != p1_checkpoint["role_state_shapes"]:
        raise ValueError("manual NavTrain P0/P1 Planner role key/shape architectures differ")
    p0_encoder = p0_checkpoint["encoder"]
    p1_encoder = p1_checkpoint["encoder"]
    if set(p0_encoder) != set(p1_encoder) or any(
        not torch.equal(p0_encoder[key], p1_encoder[key]) for key in p0_encoder
    ):
        raise ValueError("manual NavTrain P0/P1 encoder states must be tensorwise equal")

    value_specs = (
        (
            "field_checkpoint",
            "field_warmup",
            value_lineage.checkpoint_branch_id("field_warmup"),
        ),
        (
            "calibration_checkpoint",
            "field_calibrated",
            value_lineage.checkpoint_branch_id("field_calibrated"),
        ),
        (
            "stop_checkpoint",
            "stop_calibrated",
            value_lineage.checkpoint_branch_id("stop_calibrated"),
        ),
    )
    value_payloads: list[Mapping[str, object]] = []
    for role, phase, branch_id in value_specs:
        payload = read_cvoi_navsim_e120_direct_value_checkpoint(
            artifacts[role],
            required_phase=phase,
            required_branch_id=branch_id,
            map_location="cpu",
        )
        if (
            not isinstance(payload, Mapping)
            or payload.get("phase") != phase
            or payload.get("branch_id") != branch_id
            or not isinstance(payload.get("architecture"), Mapping)
            or not isinstance(payload.get("state_dict"), Mapping)
            or not payload["state_dict"]
        ):
            raise ValueError(f"manual NavTrain {role} direct Value payload is structurally invalid")
        value_payloads.append(payload)
    architectures = tuple(dict(payload["architecture"]) for payload in value_payloads)
    if architectures[1:] != architectures[:1] * 2:
        raise ValueError("manual NavTrain Field/Calibration/Stop Value architectures differ")
    return _ManualNavTrainArtifacts(
        p0_checkpoint=p0_checkpoint,
        p1_checkpoint=p1_checkpoint,
        field_architecture=MappingProxyType(architectures[0]),
        calibration_architecture=MappingProxyType(architectures[1]),
        stop_architecture=MappingProxyType(architectures[2]),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _open_trace_directory_fd(trace_output_dir: Path) -> int:
    """Open an absolute directory without following any path component symlink."""

    if not isinstance(trace_output_dir, Path) or not trace_output_dir.is_absolute():
        raise ValueError("CVoI trace_output_dir must be an absolute Path")
    if trace_output_dir.anchor != os.sep or any(component == os.pardir for component in trace_output_dir.parts[1:]):
        raise ValueError("CVoI trace_output_dir must use a canonical absolute path")

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor: Optional[int] = None
    try:
        descriptor = os.open(os.sep, directory_flags)
        for component in trace_output_dir.parts[1:]:
            next_descriptor = os.open(component, directory_flags, dir_fd=descriptor)
            try:
                os.close(descriptor)
            except OSError:
                descriptor = None
                try:
                    os.close(next_descriptor)
                except OSError as cleanup_exc:
                    logger.error(
                        "Failed to close newly opened CVoI trace directory descriptor",
                        exc_info=(type(cleanup_exc), cleanup_exc, cleanup_exc.__traceback__),
                    )
                raise
            descriptor = next_descriptor
    except OSError as exc:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as cleanup_exc:
                logger.error(
                    "Failed to close CVoI trace directory descriptor after open failure",
                    exc_info=(type(cleanup_exc), cleanup_exc, cleanup_exc.__traceback__),
                )
        raise ValueError(
            f"CVoI trace_output_dir must be an existing non-symlink directory: {trace_output_dir}"
        ) from exc
    if descriptor is None:
        raise RuntimeError("CVoI trace directory traversal did not produce a descriptor")
    return descriptor


def _write_exclusive_trace(
    trace_output_dir: Path,
    observation_key_value: str,
    trace: Mapping[str, object],
    *,
    expected_directory_identity: tuple[int, int],
    destination_stem: Optional[str] = None,
) -> Path:
    """Atomically publish one canonical trace without overwrite semantics."""

    if type(observation_key_value) is not str or re.fullmatch(r"[0-9a-f]{64}", observation_key_value) is None:
        raise ValueError("CVoI observation key must be a 64-character lowercase hexadecimal string")
    if destination_stem is None:
        destination_stem = observation_key_value
    else:
        destination_stem = validate_cvoi_direct_epdms_scenario_token(destination_stem)
    if (
        type(expected_directory_identity) is not tuple
        or len(expected_directory_identity) != 2
        or any(type(value) is not int or value < 0 for value in expected_directory_identity)
    ):
        raise ValueError("CVoI expected trace directory identity must be a (device, inode) integer tuple")

    payload = _canonical_json_bytes(dict(trace))
    destination_name = f"{destination_stem}.json"
    staging_name = f".{observation_key_value}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    directory_descriptor = _open_trace_directory_fd(trace_output_dir)
    destination = trace_output_dir / destination_name
    descriptor: Optional[int] = None
    staging_created = False
    primary_exception: Optional[BaseException] = None
    primary_traceback = None
    try:
        directory_stat = os.fstat(directory_descriptor)
        actual_directory_identity = (directory_stat.st_dev, directory_stat.st_ino)
        if actual_directory_identity != expected_directory_identity:
            raise RuntimeError(
                "CVoI trace directory identity drifted after policy resolution: "
                f"expected={expected_directory_identity}, actual={actual_directory_identity}"
            )
        descriptor = os.open(
            staging_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o644,
            dir_fd=directory_descriptor,
        )
        staging_created = True
        remaining = memoryview(payload)
        while remaining:
            bytes_written = os.write(descriptor, remaining)
            if bytes_written <= 0:
                raise OSError("CVoI trace staging write made no progress")
            remaining = remaining[bytes_written:]
        os.fsync(descriptor)
        descriptor_to_close = descriptor
        descriptor = None
        os.close(descriptor_to_close)
        try:
            os.link(
                staging_name,
                destination_name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(f"CVoI trace already exists: {destination}") from exc
        except OSError as link_exc:
            try:
                staging_stat = os.stat(staging_name, dir_fd=directory_descriptor, follow_symlinks=False)
                destination_stat = os.stat(destination_name, dir_fd=directory_descriptor, follow_symlinks=False)
            except OSError:
                raise link_exc
            if (staging_stat.st_dev, staging_stat.st_ino) != (destination_stat.st_dev, destination_stat.st_ino):
                raise link_exc
        os.fsync(directory_descriptor)
    except BaseException as exc:
        primary_exception = exc
        primary_traceback = exc.__traceback__

    cleanup_errors: List[BaseException] = []
    if descriptor is not None:
        descriptor_to_close = descriptor
        descriptor = None
        try:
            os.close(descriptor_to_close)
        except BaseException as exc:
            cleanup_errors.append(exc)
    if staging_created:
        try:
            os.unlink(staging_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        except BaseException as exc:
            cleanup_errors.append(exc)
        else:
            try:
                os.fsync(directory_descriptor)
            except BaseException as exc:
                cleanup_errors.append(exc)
    try:
        os.close(directory_descriptor)
    except BaseException as exc:
        cleanup_errors.append(exc)

    if primary_exception is not None:
        for cleanup_error in cleanup_errors:
            logger.error(
                "CVoI trace cleanup failed while preserving the primary publish exception",
                exc_info=(type(cleanup_error), cleanup_error, cleanup_error.__traceback__),
            )
        raise primary_exception.with_traceback(primary_traceback)
    if cleanup_errors:
        for secondary_cleanup_error in cleanup_errors[1:]:
            logger.error(
                "Additional CVoI trace cleanup failure",
                exc_info=(
                    type(secondary_cleanup_error),
                    secondary_cleanup_error,
                    secondary_cleanup_error.__traceback__,
                ),
            )
        cleanup_error = cleanup_errors[0]
        raise cleanup_error.with_traceback(cleanup_error.__traceback__)
    return destination


def _seed_tensor_payload(tensor: Optional[torch.Tensor]) -> Optional[Dict[str, Any]]:
    if tensor is None:
        return None
    cpu_tensor = tensor.detach().cpu().to(torch.float32)
    return {"shape": list(cpu_tensor.shape), "values": cpu_tensor.flatten().tolist()}


def _make_navsim_eval_rng_seed(
    config: TrainingConfig,
    *,
    stage: str,
    states: torch.Tensor,
    actions: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
) -> int:
    meta = getattr(config, "meta", None)
    base_seed = int(getattr(meta, "seed", 0))
    payload = {
        "base_seed": base_seed,
        "stage": f"navsim_{stage}",
        "states": _seed_tensor_payload(states),
        "actions": _seed_tensor_payload(actions),
        "driving_command": _seed_tensor_payload(driving_command),
        "ego_dynamics": _seed_tensor_payload(ego_dynamics),
    }
    digest = hashlib.blake2b(repr(payload).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**63 - 1)


def _pad_or_trim_temporal_tensor(tensor: Optional[torch.Tensor], length: int) -> Optional[torch.Tensor]:
    """Pad or trim [B, T, ...] temporal tensors to ``length``."""
    if tensor is None or tensor.ndim < 3:
        return tensor
    if tensor.shape[1] == length:
        return tensor
    if tensor.shape[1] > length:
        return tensor[:, :length]
    pad_shape = list(tensor.shape)
    pad_shape[1] = length - tensor.shape[1]
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=1)


_deterministic_navsim_eval_rng = deterministic_eval_rng


def _validate_eval_rollout_future_steps(rollout_future_steps: Optional[int]) -> Optional[int]:
    if rollout_future_steps is None:
        return None
    if isinstance(rollout_future_steps, bool) or not isinstance(rollout_future_steps, int):
        raise TypeError("rollout_future_steps must be a non-negative integer or None")
    if rollout_future_steps < 0:
        raise ValueError("rollout_future_steps must be a non-negative integer or None")
    return rollout_future_steps


def _resolve_eval_rollout_end_step(
    rollout_future_steps: Optional[int],
    *,
    num_observed_steps: int,
    num_future_steps: int,
) -> Optional[int]:
    rollout_future_steps = _validate_eval_rollout_future_steps(rollout_future_steps)
    if rollout_future_steps is None:
        return None
    if rollout_future_steps > num_future_steps:
        raise ValueError(
            "rollout_future_steps exceeds available predictor future steps: "
            f"requested {rollout_future_steps}, available {num_future_steps}"
        )
    return num_observed_steps + rollout_future_steps


def _validate_fixed_rollout_capability(config: TrainingConfig, *, forward_mode: str) -> None:
    """Validate that H0-Hk changes only the variable-length predictor prefix seen by the planner."""
    if forward_mode != "stage12":
        raise ValueError("fixed rollout is only supported with forward_mode='stage12'")
    if not bool(config.train.predictor_inference_consistent):
        raise ValueError("fixed rollout requires train.predictor_inference_consistent=true")
    if bool(config.train.predictor_no_aux_input):
        raise ValueError("fixed rollout requires train.predictor_no_aux_input=false")
    if use_latent_dit_predictor(config):
        raise ValueError("fixed rollout does not support train.predictor_type='latent_dit'")
    if use_parallel_predictor(config):
        raise ValueError("fixed rollout does not support train.use_parallel_predictor=true")
    if str(getattr(config.planner, "z_ar_mode", "full")) != "full":
        raise ValueError("fixed rollout requires planner.z_ar_mode='full'")
    if str(config.planner.planner_type) != "diffusion":
        raise ValueError(
            "fixed rollout requires a diffusion planner that consumes variable-length z_ar; "
            f"got planner.planner_type={config.planner.planner_type!r}"
        )
    if str(getattr(config.planner, "policy_output_source", "planner")) != "planner":
        raise ValueError("fixed rollout requires planner.policy_output_source='planner'")
    if bool(config.planner.use_z_context):
        raise ValueError("fixed rollout requires planner.use_z_context=false so the planner consumes z_ar")
    if bool(getattr(getattr(config, "value_guidance", None), "enabled", False)):
        raise ValueError("fixed rollout requires value_guidance.enabled=false")
    if value_planning_enabled(config):
        raise ValueError("fixed rollout requires value_planning.enabled=false")
    if cvoi_enabled(config):
        raise ValueError("fixed rollout cannot be combined with CVoI")
    if bool(getattr(getattr(config, "budget_controller", None), "enabled", False)):
        raise ValueError("fixed rollout cannot be combined with the budget controller")


class VJEPAWorldModelAgent(AbstractAgent):
    """
    VJEPA 世界模型 Agent，用于 NavSim PDMS/EPDMS 评估。

    Parameters
    ----------
    checkpoint_path : str
        训练好的 checkpoint 路径 (.pth)
    training_config_path : str
        训练使用的 YAML 配置文件路径
    crop_size : int
        输入图像裁剪尺寸 (默认 256)
    camera_name : str
        使用的摄像头名称 (默认 "cam_f0")
    num_history_frames : int
        AgentInput 中使用的历史帧数 (默认 4，NavSim V2 标准)
    device : str
        推理设备 (默认 "cuda")
    """

    _rollout_future_steps: Optional[int] = None

    def __init__(
        self,
        checkpoint_path: str = "",
        training_config_path: str = "",
        encoder_checkpoint_path: str = "",
        proposal_checkpoint_path: str = "",
        crop_size: int = 256,
        camera_name: str = "cam_f0",
        camera_names: Optional[List[str]] = None,
        num_history_frames: int = 4,
        device: str = "cuda",
        forward_mode: str = "stage12",
        rollout_future_steps: Optional[int] = None,
        trajectory_sampling: Optional[TrajectorySampling] = None,
        cvoi_manual_navtrain_gate_config_path: Optional[str] = None,
        cvoi_direct_epdms_config_path: Optional[str] = None,
    ) -> None:
        if trajectory_sampling is None:
            trajectory_sampling = NAVSIM_TRAJECTORY_SAMPLING
        # 兼容 V1/V2 的 AbstractAgent 签名差异
        if "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters:
            super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=False)
        else:
            super().__init__(requires_scene=False)
        self._cvoi_trajectory_sampling = trajectory_sampling
        forward_mode = str(forward_mode).lower()
        if forward_mode not in _FORWARD_MODES:
            raise ValueError(f"forward_mode must be one of {sorted(_FORWARD_MODES)}, got {forward_mode!r}")
        rollout_future_steps = _validate_eval_rollout_future_steps(rollout_future_steps)
        if rollout_future_steps is not None and forward_mode != "stage12":
            raise ValueError("rollout_future_steps is only supported with forward_mode='stage12'")
        self.checkpoint_path = checkpoint_path
        self.training_config_path = training_config_path
        self.encoder_checkpoint_path = encoder_checkpoint_path
        self.crop_size = crop_size
        self.camera_name = camera_name
        self.camera_names = list(camera_names) if camera_names is not None else [camera_name]
        self.num_history_frames = num_history_frames
        self._forward_mode = forward_mode
        self._rollout_future_steps = rollout_future_steps
        self._proposal_checkpoint_path = proposal_checkpoint_path
        self._device_str = device
        if cvoi_manual_navtrain_gate_config_path is not None and (
            type(cvoi_manual_navtrain_gate_config_path) is not str or not cvoi_manual_navtrain_gate_config_path.strip()
        ):
            raise ValueError("cvoi_manual_navtrain_gate_config_path must be a non-empty string")
        if cvoi_direct_epdms_config_path is not None and (
            type(cvoi_direct_epdms_config_path) is not str or not cvoi_direct_epdms_config_path.strip()
        ):
            raise ValueError("cvoi_direct_epdms_config_path must be a non-empty string")
        if cvoi_direct_epdms_config_path is not None and not Path(cvoi_direct_epdms_config_path).is_absolute():
            raise ValueError("cvoi_direct_epdms_config_path must be absolute")
        request_kinds = (
            cvoi_manual_navtrain_gate_config_path is not None,
            cvoi_direct_epdms_config_path is not None,
        )
        if sum(request_kinds) > 1:
            raise ValueError("manual NavTrain and direct EPDMS requests are mutually exclusive")
        self._cvoi_manual_navtrain_gate_config_path = cvoi_manual_navtrain_gate_config_path
        self._cvoi_direct_epdms_config_path = cvoi_direct_epdms_config_path

        # 模型组件 (在 initialize() 中构建)
        self._encoder: Optional[nn.Module] = None
        self._predictor: Optional[nn.Module] = None
        self._planner: Optional[nn.Module] = None
        self._cvoi_direct_p0_planner: Optional[nn.Module] = None
        self._cvoi_direct_p1_planner: Optional[nn.Module] = None
        self._value_head: Optional[nn.Module] = None
        self._cvoi_dual_value: Optional[nn.Module] = None
        self._cvoi_dual_value_adapter: Optional[nn.Module] = None
        self._cvoi_direct_field_value: Optional[nn.Module] = None
        self._cvoi_direct_field_value_adapter: Optional[nn.Module] = None
        self._cvoi_direct_stop_value: Optional[nn.Module] = None
        self._cvoi_direct_stop_value_adapter: Optional[nn.Module] = None
        self._cvoi_direct_gate: Optional[nn.Module] = None
        self._cvoi_navtrain_stop_value: Optional[nn.Module] = None
        self._cvoi_navtrain_stop_value_adapter: Optional[nn.Module] = None
        self._cvoi_gate: Optional[nn.Module] = None
        self._last_cvoi_trace: Optional[Dict[str, Any]] = None
        self._last_cvoi_planner_output: Optional[Dict[str, torch.Tensor]] = None
        self._last_cvoi_latency_components: Optional[Dict[str, float]] = None
        self._last_cvoi_navtrain_gate_features: Optional[Dict[str, Any]] = None
        self._cvoi_latency_mode: bool = False
        self._cvoi_evaluation_guidance_steps: Optional[int] = None
        self._cvoi_evaluation_forced_horizon: Optional[int] = None
        self._cvoi_evaluation_gate_feature_mode: Optional[str] = None
        self._budget_controller: Optional[nn.Module] = None
        self._multiview_fusion: Optional[nn.Module] = None
        self._config: Optional[TrainingConfig] = None
        self._cvoi_manual_runtime: Optional[_ManualNavTrainRuntimeContext] = None
        self._cvoi_manual_artifacts: Optional[_ManualNavTrainArtifacts] = None
        self._cvoi_direct_projection: Optional[CvoiDirectEpdmsProjection] = None
        self._cvoi_direct_runtime: Optional[_DirectEpdmsRuntimeContext] = None

        # 推理参数 (在 initialize() 中从 config 解析)
        self._tokens_per_frame: int = 0
        self._num_poses: int = 0
        self._normalize_reps: bool = True
        self._status_mode: str = "first"
        self._z_ar_mode: str = "full"
        self._use_z_context: bool = False
        self._use_observed_tokens: bool = False
        self._stage12_planner_uses_full_context: bool = False
        self._use_states_for_planner: bool = False
        self._predictor_inference_consistent: bool = False
        self._predictor_no_aux_input: bool = False
        self._use_states_for_predictor: bool = True
        self._num_observed_frames: int = 2
        self._action_dim: int = 7

        # FeatureBuilder
        self._feature_builder: Optional[VJEPAFeatureBuilder] = None

        # Staged refinement slots (initialized in _initialize_stage2/_initialize_stage3)
        self._proposal_planner: Optional[nn.Module] = None
        self._proposal_encoder: Optional[nn.Module] = None
        self._token_ae: Optional[nn.Module] = None
        self._proposal_token_ae: Optional[nn.Module] = None
        self._main_context_timeline = None
        self._runtime_normalize_reps: bool = True
        self._proposal_runtime_normalize_reps: Optional[bool] = None

    # ------------------------------------------------------------------
    #  AbstractAgent 接口实现
    # ------------------------------------------------------------------

    def set_rollout_future_steps(self, rollout_future_steps: Optional[int]) -> None:
        """Select a fixed Stage-12 predictor future prefix for a subsequent policy call."""
        validated = _validate_eval_rollout_future_steps(rollout_future_steps)
        if validated is not None and self._forward_mode != "stage12":
            raise ValueError("rollout_future_steps is only supported with forward_mode='stage12'")
        if validated is not None and self._config is not None:
            self._validate_fixed_rollout_capabilities()
        self._rollout_future_steps = validated

    def _validate_fixed_rollout_capabilities(self) -> None:
        """Fail before inference when the configured Stage-12 path cannot represent H0-Hk."""
        if self._config is None:
            raise RuntimeError("fixed rollout capabilities cannot be validated before loading the config")
        _validate_fixed_rollout_capability(self._config, forward_mode=self._forward_mode)

    def name(self) -> str:
        return "vjepa_world_model_agent"

    def get_sensor_config(self) -> SensorConfig:
        """
        默认只需要前视摄像头；多视角配置会请求 L0/F0/R0。
        根据 num_history_frames 决定哪些帧加载相机图像。
        """
        frame_indices = list(range(self.num_history_frames))
        requested = {name.lower() for name in self.camera_names}
        return SensorConfig(
            cam_f0=frame_indices if "cam_f0" in requested else False,
            cam_l0=frame_indices if "cam_l0" in requested else False,
            cam_l1=False,
            cam_l2=False,
            cam_r0=frame_indices if "cam_r0" in requested else False,
            cam_r1=False,
            cam_r2=False,
            cam_b0=False,
            lidar_pc=False,
        )

    @staticmethod
    def _normalize_navsim_trajectory_length(
        trajectory: torch.Tensor,
        num_poses: int = NAVSIM_NUM_POSES,
    ) -> torch.Tensor:
        """Trim or pad [B, T, 3] trajectories to NavSim sampling length."""
        if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
            raise ValueError(f"Expected trajectory shape [B, T, 3], got {tuple(trajectory.shape)}")
        if trajectory.shape[1] == num_poses:
            return trajectory
        if trajectory.shape[1] > num_poses:
            return trajectory[:, :num_poses, :]
        if trajectory.shape[1] == 0:
            pad = trajectory.new_zeros((trajectory.shape[0], num_poses, trajectory.shape[2]))
        else:
            pad = trajectory[:, -1:, :].expand(-1, num_poses - trajectory.shape[1], -1)
        return torch.cat([trajectory, pad], dim=1)

    @staticmethod
    def _validate_fixed_rollout_trajectory(
        trajectory: torch.Tensor,
        num_poses: int,
    ) -> torch.Tensor:
        """Require the planner's exact NavSim trajectory contract without legacy repair."""
        if trajectory.ndim != 3 or tuple(trajectory.shape[1:]) != (num_poses, 3):
            raise ValueError(
                f"fixed rollout planner trajectory must contain exactly {num_poses} poses "
                f"with shape [B, {num_poses}, 3], got {tuple(trajectory.shape)}"
            )
        if not torch.isfinite(trajectory).all():
            raise ValueError("fixed rollout planner trajectory must contain only finite values")
        return trajectory

    @staticmethod
    def _maybe_set_legacy_stage3_refinement_core_type(
        config: TrainingConfig,
        checkpoint: Dict[str, Any],
    ) -> bool:
        """Detect legacy transformer Stage-3 checkpoints before planner construction."""
        planner_state = checkpoint.get("planner") if isinstance(checkpoint, dict) else None
        if not isinstance(planner_state, dict):
            return False
        if getattr(config.planner, "refinement_core_type", None):
            return False

        has_transformer_core = any(key.startswith("refine_core.transformer.") for key in planner_state)
        has_diffusion_core = any(key.startswith("refine_core.dit.") for key in planner_state)
        if not has_transformer_core or has_diffusion_core:
            return False

        config.planner.refinement_core_type = "transformer"
        logger.info(
            "Stage3 checkpoint uses legacy transformer refinement core; "
            "setting planner.refinement_core_type=transformer for evaluation compatibility"
        )
        return True

    @staticmethod
    def _extract_proposal_core_planner_state(
        planner_state: Dict[str, Any],
        config: TrainingConfig,
    ) -> Optional[Dict[str, Any]]:
        """Extract ``TransformerProposalProvider.core`` weights for stage12 planner eval."""
        proposal_cfg = getattr(config, "proposal", None)
        if (
            proposal_cfg is None
            or not bool(getattr(proposal_cfg, "enabled", False))
            or getattr(proposal_cfg, "provider_type", None) != "transformer"
        ):
            return None
        if not isinstance(planner_state, dict):
            return None

        normalized_state = {
            key[7:] if isinstance(key, str) and key.startswith("module.") else key: value
            for key, value in planner_state.items()
        }
        core_prefix = "core."
        core_state = {
            key[len(core_prefix) :]: value
            for key, value in normalized_state.items()
            if isinstance(key, str) and key.startswith(core_prefix)
        }
        if not core_state:
            return None

        non_core_keys = [key for key in normalized_state if not (isinstance(key, str) and key.startswith(core_prefix))]
        if non_core_keys:
            raise RuntimeError(
                "proposal.enabled=true with proposal.provider_type='transformer' and checkpoint['planner'] "
                "contains proposal core weights, but also has non-core planner keys. "
                f"Refusing ambiguous stage12 planner load. non_core_keys={non_core_keys[:10]}"
            )
        return core_state

    def _build_stage12_proposal_core_planner(
        self,
        *,
        device: torch.device,
        encoder_embed_dim: int,
        tokens_per_frame: int,
    ) -> nn.Module:
        """Build a train-time transformer proposal core for stage12 PDMS compatibility."""
        from app.vjepa_cowa_world_model.models.proposal_providers import build_proposal_provider

        cfg = self._config
        if cfg is None:
            raise RuntimeError("Training config must be loaded before building a proposal-core planner")
        if cfg.proposal.use_separate_encoder:
            raise RuntimeError(
                "Stage12 cannot evaluate a proposal-core planner checkpoint with "
                "proposal.use_separate_encoder=true because the stage12 forward path only has main encoder tokens. "
                "Use a staged proposal/refinement forward mode for this checkpoint."
            )

        proposal_tokens_per_frame = resolve_proposal_tokens_per_frame(cfg, None)
        if int(proposal_tokens_per_frame) != int(tokens_per_frame):
            raise RuntimeError(
                "Stage12 proposal-core planner token mismatch: "
                f"proposal_tokens_per_frame={proposal_tokens_per_frame}, "
                f"main_runtime_tokens_per_frame={tokens_per_frame}. "
                "The proposal core cannot be fed by the stage12 main predictor timeline."
            )

        proposal_num_context_frames = resolve_proposal_num_time_steps(cfg, None)
        num_poses = int(cfg.data.num_target_frames) - int(cfg.train.num_observed_frames)
        status_dim = resolve_effective_planner_status_dim(cfg)
        use_cmd = resolve_planner_use_drive_command(cfg)
        command_dim = (
            4 if (use_cmd and cfg.planner.split_status_embedding and cfg.train.predictor_inference_consistent) else 0
        )
        proposal_planner = build_proposal_provider(
            config=cfg,
            encoder_dim=encoder_embed_dim,
            tokens_per_frame=proposal_tokens_per_frame,
            num_poses=num_poses,
            status_dim=status_dim,
            command_dim=command_dim,
            num_context_frames=proposal_num_context_frames,
            num_observed_frames=cfg.train.num_observed_frames,
        ).to(device)
        proposal_core = getattr(proposal_planner, "core", None)
        if proposal_core is None:
            raise RuntimeError("Expected transformer proposal provider to expose a 'core' planner module")
        logger.info(
            "Built stage12 proposal-core planner: tokens_per_frame=%d, context_steps=%d, num_poses=%d",
            int(proposal_tokens_per_frame),
            int(proposal_num_context_frames),
            int(num_poses),
        )
        return proposal_core

    def _configured_cvoi_planner_checkpoint(self) -> str:
        lineage = self._config.cvoi.controller_lineage
        if lineage == "p0_controller":
            checkpoint = self._config.cvoi.unguided_planner_checkpoint
        elif lineage == "value_guided":
            checkpoint = self._config.cvoi.guided_planner_checkpoint
        else:
            raise ValueError(f"unsupported CVoI controller_lineage: {lineage!r}")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"CVoI {lineage} Planner checkpoint must be a non-empty path")
        return checkpoint

    def _validate_configured_cvoi_planner_checkpoint(self, checkpoint: Mapping[str, object]) -> Mapping[str, object]:
        """Validate the retained manual NavSim-e120 Planner envelope."""

        manual_runtime = getattr(self, "_cvoi_manual_runtime", None)
        if manual_runtime is None:
            raise ValueError(
                "generic CVoI Planner checkpoint validation is retired; "
                "use the manual NavSim-e120 scorer or direct EPDMS runtime"
            )
        normalized = validate_formal_v2_navsim_e120_direct_checkpoint(checkpoint)
        return _validate_manual_planner_payload(
            normalized,
            expected_stage="p1",
            expected_branch_id=manual_runtime.lineage,
        )

    def _bind_cvoi_manual_runtime(self) -> None:
        """Bind the proof-free fixed-H teacher to its two direct Value models."""

        if self._cvoi_manual_runtime is None:
            return
        if self._cvoi_gate is not None:
            raise RuntimeError("manual NavTrain fixed policy must not load or execute a Gate")
        if self._cvoi_dual_value is None or self._cvoi_dual_value_adapter is None:
            raise RuntimeError("manual NavTrain P1 teacher requires its Calibration field-Value model")
        if self._cvoi_navtrain_stop_value is None or self._cvoi_navtrain_stop_value_adapter is None:
            raise RuntimeError("manual NavTrain P1 teacher requires its Stop value model")
        self._cvoi_evaluation_gate_feature_mode = None

    def _bind_cvoi_direct_runtime(self) -> None:
        """Require the exact frozen module set for one direct projection."""

        runtime = getattr(self, "_cvoi_direct_runtime", None)
        if runtime is None:
            return
        projection = runtime.projection
        if (
            self._config is None
            or str(self._config.cvoi.stage) != "evaluation"
            or str(self._config.cvoi.protocol_version) != "formal_v2_navsim_e120_h4_v3"
        ):
            raise ValueError("direct EPDMS modules require the NavSim-e120 evaluation runtime")
        slots = {
            "p0_planner": getattr(self, "_cvoi_direct_p0_planner", None),
            "p1_planner": getattr(self, "_cvoi_direct_p1_planner", None),
            "field_value": getattr(self, "_cvoi_direct_field_value", None),
            "stop_value": getattr(self, "_cvoi_direct_stop_value", None),
            "gate": getattr(self, "_cvoi_direct_gate", None),
        }
        expected_by_mode = {
            "controller": {"p0_planner", "p1_planner", "field_value", "stop_value", "gate"},
            "p0_forced": {"p0_planner"},
            "p1_field_forced": {"p1_planner", "field_value"},
        }
        expected = expected_by_mode.get(projection.evaluation_mode)
        if expected is None:
            raise ValueError(f"unsupported direct EPDMS evaluation mode: {projection.evaluation_mode!r}")
        present = {name for name, module in slots.items() if module is not None}
        if present != expected:
            raise RuntimeError(
                f"direct EPDMS module set mismatch: missing={sorted(expected - present)}, "
                f"unexpected={sorted(present - expected)}"
            )
        for role, module in slots.items():
            if module is None:
                continue
            if not isinstance(module, nn.Module):
                raise TypeError(f"direct EPDMS {role} must be a torch.nn.Module")
            if module.training or any(parameter.requires_grad for parameter in module.parameters()):
                raise RuntimeError(f"direct EPDMS {role} must be frozen in eval mode")

        field_value = slots["field_value"]
        stop_value = slots["stop_value"]
        gate = slots["gate"]
        self._cvoi_direct_field_value_adapter = (
            None if field_value is None else CvoiValueDtypeAdapter(field_value).eval()
        )
        self._cvoi_direct_stop_value_adapter = None if stop_value is None else CvoiValueDtypeAdapter(stop_value).eval()
        self._cvoi_gate = gate
        self._cvoi_evaluation_forced_horizon = projection.horizon
        self._cvoi_evaluation_guidance_steps = projection.guidance_steps
        self._cvoi_evaluation_gate_feature_mode = (
            projection.gate_feature_mode if projection.evaluation_mode == "controller" else None
        )

    def _select_cvoi_direct_planner(self, stop_horizon: int) -> nn.Module:
        """Choose P0 at H0 and P1 at H1--H4 without running both planners."""

        runtime = getattr(self, "_cvoi_direct_runtime", None)
        if runtime is None:
            if self._planner is None:
                raise RuntimeError("NavSim Planner is not initialized")
            return self._planner
        if type(stop_horizon) is not int or stop_horizon not in range(5):
            raise ValueError(f"direct EPDMS stop_horizon must be an integer in H0--H4, got {stop_horizon!r}")
        projection = runtime.projection
        if projection.evaluation_mode == "controller":
            planner = self._cvoi_direct_p0_planner if stop_horizon == 0 else self._cvoi_direct_p1_planner
        elif projection.evaluation_mode == "p0_forced":
            if stop_horizon != 0:
                raise RuntimeError("direct EPDMS p0_forced runtime produced a nonzero horizon")
            planner = self._cvoi_direct_p0_planner
        elif projection.evaluation_mode == "p1_field_forced":
            if stop_horizon != projection.horizon:
                raise RuntimeError(
                    "direct EPDMS p1_field_forced runtime horizon differs from its effective projection"
                )
            planner = self._cvoi_direct_p1_planner
        else:
            raise ValueError(f"unsupported direct EPDMS evaluation mode: {projection.evaluation_mode!r}")
        if planner is None:
            raise RuntimeError("direct EPDMS selected Planner is not loaded")
        return planner

    def initialize(self) -> None:
        """加载训练配置和 checkpoint，构建模型。"""
        direct_effective_path = getattr(self, "_cvoi_direct_epdms_config_path", None)
        direct_projection: Optional[CvoiDirectEpdmsProjection] = None
        direct_device: Optional[torch.device] = None
        if direct_effective_path is not None:
            try:
                direct_device = torch.device(self._device_str)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(f"direct EPDMS received an invalid device: {self._device_str!r}") from exc
            if direct_device.type != "cuda":
                raise ValueError("direct EPDMS requires a CUDA device because latency is a required result")
            if not torch.cuda.is_available():
                raise RuntimeError("direct EPDMS requires available CUDA; CPU fallback is forbidden")
            if direct_device.index is not None:
                device_count = torch.cuda.device_count()
                if direct_device.index >= device_count:
                    raise ValueError(
                        "direct EPDMS CUDA device index "
                        f"{direct_device.index} is out of range for device_count={device_count}"
                    )
            if self._forward_mode != "stage12":
                raise ValueError("direct EPDMS requires forward_mode='stage12'")
            if self._rollout_future_steps is not None:
                raise ValueError("direct EPDMS forbids rollout_future_steps")
            if self._proposal_checkpoint_path:
                raise ValueError("direct EPDMS stage12 execution forbids proposal_checkpoint_path")
            if self.training_config_path or self.checkpoint_path or self.encoder_checkpoint_path:
                raise ValueError(
                    "direct EPDMS effective projection is the sole training/checkpoint/encoder path authority"
                )
            direct_projection = load_cvoi_direct_epdms_projection(Path(direct_effective_path))
            direct_artifacts = _read_cvoi_direct_epdms_artifacts(direct_projection)
            scenario_tokens = read_cvoi_direct_epdms_scenario_manifest(
                direct_projection.scenario_manifest_path,
            )
            trace_output_dir = direct_projection.output_directory / "policy_traces"
            try:
                trace_stat = trace_output_dir.stat(follow_symlinks=False)
            except (FileNotFoundError, OSError) as error:
                raise ValueError(
                    f"direct EPDMS policy trace directory must already exist: {trace_output_dir}"
                ) from error
            if not trace_output_dir.is_dir() or trace_output_dir.is_symlink():
                raise ValueError(
                    f"direct EPDMS policy trace directory must be a non-symlink directory: {trace_output_dir}"
                )
            self._cvoi_direct_projection = direct_projection
            self._cvoi_direct_runtime = _DirectEpdmsRuntimeContext(
                projection=direct_projection,
                artifacts=direct_artifacts,
                scenario_tokens_by_observation_key=scenario_tokens,
                trace_output_dir=trace_output_dir,
                trace_output_dir_identity=(trace_stat.st_dev, trace_stat.st_ino),
            )
            self.training_config_path = str(direct_projection.training_config_path)
            self.encoder_checkpoint_path = str(direct_projection.encoder_checkpoint_path)
            active_planner_path = (
                direct_projection.p0_planner_checkpoint_path
                if direct_projection.evaluation_mode == "p0_forced"
                else direct_projection.p1_planner_checkpoint_path
            )
            if active_planner_path is None:
                raise RuntimeError("direct EPDMS projection does not contain its active Planner")
            self.checkpoint_path = str(active_planner_path)

        manual_scorer_path = getattr(self, "_cvoi_manual_navtrain_gate_config_path", None)
        manual_scorer_config: Optional[ManualNavTrainScorerConfig] = None
        manual_value_lineage = None
        if manual_scorer_path is not None:
            try:
                device = torch.device(self._device_str)
            except (RuntimeError, TypeError) as exc:
                raise ValueError(f"manual NavTrain scorer received an invalid device: {self._device_str!r}") from exc
            if device.type != "cuda":
                raise ValueError("manual NavTrain scorer requires a CUDA device")
            if not torch.cuda.is_available():
                raise RuntimeError("manual NavTrain scorer requires available CUDA; CPU fallback is forbidden")
            if device.index is not None:
                device_count = torch.cuda.device_count()
                if device.index >= device_count:
                    raise ValueError(
                        "manual NavTrain scorer CUDA device index "
                        f"{device.index} is out of range for device_count={device_count}"
                    )
            manual_scorer_config = read_manual_navtrain_scorer_config(Path(manual_scorer_path))
            if not isinstance(manual_scorer_config, ManualNavTrainScorerConfig):
                raise TypeError("manual NavTrain scorer reader must return ManualNavTrainScorerConfig")
            if (
                manual_scorer_config.schema != MANUAL_NAVTRAIN_SCORER_SCHEMA
                or manual_scorer_config.protocol_id != NAVTRAIN_GATE_PROTOCOL_ID
            ):
                raise ValueError("manual NavTrain scorer schema or protocol differs")
            source = manual_scorer_config.source
            if not isinstance(source, ManualOracleSource):
                raise ValueError("manual NavTrain scorer source must be a ManualOracleSource")
            if type(source.lineage) is not str or source.lineage not in {"p1_full", "p1_no_cf"}:
                raise ValueError("manual NavTrain scorer supports only p1_full and p1_no_cf source lineages")
            manual_value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage_from_artifacts(
                source.artifacts,
                phase="guided_planner",
                branch_id=source.lineage,
            )
            authoritative_artifacts = {
                "p0_planner_checkpoint": manual_value_lineage.p0_handoff,
                "field_checkpoint": manual_value_lineage.field_handoff,
                "calibration_checkpoint": manual_value_lineage.calibration_handoff,
                "p1_planner_checkpoint": manual_value_lineage.p1_handoff,
                "stop_checkpoint": manual_value_lineage.stop_handoff,
            }
            if (
                source.value_lineage != manual_value_lineage
                or source.results_root != manual_value_lineage.result_root
                or dict(source.artifacts) != authoritative_artifacts
            ):
                raise ValueError("manual NavTrain scorer source differs from the fixed Value-lineage authority")
            if (
                manual_scorer_config.lineage != source.lineage
                or dict(manual_scorer_config.artifacts) != authoritative_artifacts
            ):
                drifted = {
                    role: (manual_scorer_config.artifacts.get(role), expected_path)
                    for role, expected_path in authoritative_artifacts.items()
                    if manual_scorer_config.artifacts.get(role) != expected_path
                }
                raise ValueError(
                    "manual NavTrain scorer lineage or artifacts differ from its source authority: "
                    f"lineage={(manual_scorer_config.lineage, source.lineage)!r}, artifacts={drifted}"
                )
            horizon_dir = manual_scorer_config.effective_config_path.parent
            try:
                horizon_results_root = horizon_dir.parents[2]
            except IndexError as exc:
                raise ValueError("manual NavTrain scorer path is outside the fixed results layout") from exc
            if (
                Path(manual_scorer_path) != horizon_dir / "scorer_config.json"
                or horizon_results_root != source.results_root
                or manual_scorer_config.output_dir != horizon_dir / "scorer_output"
                or manual_scorer_config.trace_output_dir != horizon_dir / "policy_traces"
            ):
                raise ValueError("manual NavTrain scorer paths differ from the fixed horizon-local layout")
            if self._forward_mode != "stage12":
                raise ValueError("manual NavTrain scorer requires forward_mode='stage12'")
            if self._proposal_checkpoint_path:
                raise ValueError("manual NavTrain stage12 execution forbids proposal_checkpoint_path")
            if self.encoder_checkpoint_path:
                raise ValueError("manual NavTrain stage12 execution forbids encoder_checkpoint_path")
            if Path(self.training_config_path) != manual_scorer_config.effective_config_path:
                raise ValueError("training_config_path must equal the manual NavTrain effective evaluation config")
            expected_checkpoint = manual_scorer_config.artifacts["p1_planner_checkpoint"]
            if Path(self.checkpoint_path) != expected_checkpoint:
                raise ValueError("checkpoint_path must equal the manual NavTrain P1 handoff")
        elif direct_projection is not None:
            if direct_device is None:
                raise RuntimeError("direct EPDMS CUDA device preflight was not retained")
            device = direct_device
        else:
            device = torch.device(self._device_str if torch.cuda.is_available() else "cpu")

        # 1. 加载训练配置
        if not self.training_config_path:
            raise ValueError("training_config_path is required for VJEPAWorldModelAgent")
        logger.info(f"Loading training config from: {self.training_config_path}")
        effective_config_bytes = Path(self.training_config_path).read_bytes()
        raw_config = yaml.safe_load(effective_config_bytes)
        parsed_config = parse_training_config(raw_config)
        self._config = (
            _build_cvoi_direct_epdms_training_config(parsed_config, direct_projection)
            if direct_projection is not None
            else parsed_config
        )
        if direct_projection is not None:
            if self._cvoi_direct_runtime is None:
                raise RuntimeError("direct EPDMS runtime context disappeared during config parsing")
            self._cvoi_direct_runtime = self._cvoi_direct_runtime._replace(
                evaluation_seed=resolve_cvoi_evaluation_seed(self._config),
            )
        if manual_scorer_config is not None:
            if manual_value_lineage is None:
                raise RuntimeError("manual NavTrain Value lineage was not resolved before config parsing")
            raw_cvoi = raw_config.get("cvoi") if isinstance(raw_config, Mapping) else None
            expected_artifacts = manual_scorer_config.artifacts
            if not isinstance(raw_cvoi, Mapping) or "forced_horizon" in raw_cvoi:
                raise ValueError("manual NavTrain effective config must contain CVoI without forced_horizon")
            effective_signature = getattr(self._config.cvoi, "ablation_signature", None)
            effective_value_lineage = cvoi_manual_lineage.resolve_cvoi_manual_value_lineage(
                effective_signature,
                stage="guided_planner",
                full_results_root=manual_value_lineage.p0_result_root,
                ablation_results_root=manual_value_lineage.result_root.parent,
            )
            if (
                effective_value_lineage != manual_value_lineage
                or effective_value_lineage != manual_scorer_config.source.value_lineage
                or effective_value_lineage.checkpoint_branch_id("guided_planner") != manual_scorer_config.lineage
            ):
                raise ValueError("manual NavTrain effective CVoI signature differs from its source Value lineage")
            expected_fields = {
                "stage": "evaluation",
                "evaluation_mode": "p1_field_forced",
                "controller_lineage": "value_guided",
                "unguided_planner_checkpoint": str(expected_artifacts["p0_planner_checkpoint"]),
                "field_checkpoint": str(expected_artifacts["calibration_checkpoint"]),
                "guided_planner_checkpoint": str(expected_artifacts["p1_planner_checkpoint"]),
                "dual_value_checkpoint": str(expected_artifacts["stop_checkpoint"]),
                "output_checkpoint": None,
            }
            drifted = {
                field: (raw_cvoi.get(field), expected)
                for field, expected in expected_fields.items()
                if raw_cvoi.get(field) != expected
            }
            if drifted:
                raise ValueError(f"manual NavTrain effective CVoI config differs: {drifted}")
            self._cvoi_manual_artifacts = _read_manual_navtrain_artifacts(
                expected_artifacts,
                expected_lineage=manual_scorer_config.lineage,
            )
            trace_stat = manual_scorer_config.trace_output_dir.stat(follow_symlinks=False)
            self._cvoi_manual_runtime = _ManualNavTrainRuntimeContext(
                policy_id=manual_scorer_config.policy_id,
                lineage=manual_scorer_config.lineage,
                planner_stage=manual_scorer_config.planner_stage,
                scenario_manifest=manual_scorer_config.authority.scenario_manifest,
                forced_horizon=manual_scorer_config.forced_horizon,
                guidance_steps=manual_scorer_config.guidance_steps,
                common_random_seed=manual_scorer_config.common_random_seed,
                trace_output_dir=manual_scorer_config.trace_output_dir,
                trace_output_dir_identity=(trace_stat.st_dev, trace_stat.st_ino),
                p0_planner_checkpoint=expected_artifacts["p0_planner_checkpoint"],
                field_checkpoint=expected_artifacts["field_checkpoint"],
                calibration_checkpoint=expected_artifacts["calibration_checkpoint"],
                p1_planner_checkpoint=expected_artifacts["p1_planner_checkpoint"],
                stop_checkpoint=expected_artifacts["stop_checkpoint"],
            )
        require_cvoi_stage_for_entry(
            self._config,
            entry="VJEPAWorldModelAgent",
            allowed_stages=("evaluation",),
        )
        validate_cvoi_sequential_runtime_config(self._config)
        if cvoi_enabled(self._config):
            if self._forward_mode != "stage12":
                raise ValueError("CVoI sequential evaluation is supported only for forward_mode='stage12'")
            if direct_projection is None:
                configured_planner = os.path.realpath(self._configured_cvoi_planner_checkpoint())
                requested_planner = os.path.realpath(self.checkpoint_path)
                if configured_planner != requested_planner:
                    raise ValueError(
                        "NavSim checkpoint_path must equal the configured CVoI lineage Planner during evaluation; "
                        f"got {requested_planner!r} and {configured_planner!r}"
                    )
        if value_planning_enabled(self._config) and self._forward_mode != "stage12":
            raise ValueError(
                "value_planning.enabled=true is currently supported only for forward_mode='stage12' "
                "because value-guided stage12 inference needs predictor rollout plus a loaded value_head."
            )

        # 2. 解析推理参数
        self._parse_inference_params()
        if self._rollout_future_steps is not None:
            self._validate_fixed_rollout_capabilities()
        if getattr(self._config.multiview, "enabled", False):
            navsim_cfg = getattr(self._config.data, "navsim", None)
            if navsim_cfg is not None:
                self.camera_names = [name.lower() for name in navsim_cfg.camera_names]

        # 3. 构建模型 — 按 forward_mode 分支
        logger.info("Initializing encoder...")
        if direct_projection is not None:
            encoder = init_context_encoder_for_full_state_warmstart(self._config, device)
            target_encoder = None
        elif self._forward_mode == "encoder_direct":
            encoder, target_encoder = init_encoder_direct_encoder(self._config, device)
        else:
            encoder, target_encoder = init_encoder(self._config, device)
        self._encoder = encoder
        encoder_embed_dim = get_encoder_embed_dim(encoder)
        logger.info(f"Encoder embed_dim: {encoder_embed_dim}")

        if self._forward_mode == "stage3":
            self._initialize_stage3(device, encoder, target_encoder, encoder_embed_dim)
        elif self._forward_mode == "stage2":
            self._initialize_stage2(device, encoder, target_encoder, encoder_embed_dim)
        elif self._forward_mode == "encoder_direct":
            self._initialize_encoder_direct(device, encoder, encoder_embed_dim)
        else:
            self._initialize_stage12(device, encoder, encoder_embed_dim)

        self._bind_cvoi_manual_runtime()
        self._bind_cvoi_direct_runtime()

        # 6. 构建 feature builder
        self._feature_builder = self._build_feature_builder()

        logger.info(f"VJEPAWorldModelAgent initialized successfully! forward_mode={self._forward_mode}")

    def _build_cvoi_direct_value_model(
        self,
        payload: Mapping[str, object],
        *,
        encoder_embed_dim: int,
        device: torch.device,
        role: str,
    ) -> nn.Module:
        """Instantiate one preflighted Calibration/Stop model exactly once."""

        architecture = payload.get("architecture")
        state = payload.get("state_dict")
        if not isinstance(architecture, Mapping) or not isinstance(state, Mapping):
            raise ValueError(f"direct EPDMS {role} Value payload is incomplete")
        if architecture.get("embed_dim") != int(encoder_embed_dim):
            raise ValueError(f"direct EPDMS {role} Value embed_dim differs from encoder_embed_dim={encoder_embed_dim}")
        with torch.random.fork_rng(devices=[]):
            model = PrefixDualValueModel(**dict(architecture))
            model.load_state_dict(dict(state), strict=True)
        model.to(device=device)
        model.eval()
        model.requires_grad_(False)
        if model.training or any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError(f"direct EPDMS {role} Value failed to freeze")
        return model

    def _initialize_cvoi_direct_stage12(
        self,
        *,
        device: torch.device,
        encoder: nn.Module,
        encoder_embed_dim: int,
        tokens_per_frame: int,
        main_full_timeline: Any,
    ) -> None:
        """Load the exact mode-minimal direct artifacts without generic CVoI loaders."""

        runtime = self._cvoi_direct_runtime
        if runtime is None:
            raise RuntimeError("direct EPDMS stage12 initialization requires a runtime context")
        projection = runtime.projection
        artifacts = runtime.artifacts
        if self._planner is None:
            raise RuntimeError("direct EPDMS requires an initialized Planner envelope")

        if projection.evaluation_mode == "controller":
            self._cvoi_direct_p1_planner = self._planner
            self._cvoi_direct_p0_planner = init_planner(
                self._config,
                encoder_embed_dim,
                device,
                tokens_per_frame_override=tokens_per_frame,
            )
        elif projection.evaluation_mode == "p0_forced":
            self._cvoi_direct_p0_planner = self._planner
        elif projection.evaluation_mode == "p1_field_forced":
            self._cvoi_direct_p1_planner = self._planner
        else:
            raise ValueError(f"unsupported direct EPDMS evaluation mode: {projection.evaluation_mode!r}")

        maybe_register_parallel_predictor_tokens(
            predictor=self._predictor,
            config=self._config,
            embed_dim=encoder_embed_dim,
            future_steps=main_full_timeline.num_future_steps,
            tokens_per_frame=tokens_per_frame,
            device=device,
        )

        active_checkpoint = (
            artifacts.p0_checkpoint if projection.evaluation_mode == "p0_forced" else artifacts.p1_checkpoint
        )
        if active_checkpoint is None:
            raise RuntimeError("direct EPDMS active Planner checkpoint was not preflighted")
        _load_cvoi_formal_v2_navsim_e120_exact_state(
            encoder,
            active_checkpoint["encoder"],
            role="encoder",
        )
        _load_cvoi_formal_v2_navsim_e120_exact_state(
            self._predictor,
            active_checkpoint["predictor"],
            role="predictor",
        )
        if artifacts.p0_checkpoint is not None:
            if self._cvoi_direct_p0_planner is None:
                raise RuntimeError("direct EPDMS P0 checkpoint has no P0 Planner envelope")
            _load_cvoi_formal_v2_navsim_e120_exact_state(
                self._cvoi_direct_p0_planner,
                artifacts.p0_checkpoint["planner"],
                role="P0 planner",
            )
        if artifacts.p1_checkpoint is not None:
            if self._cvoi_direct_p1_planner is None:
                raise RuntimeError("direct EPDMS P1 checkpoint has no P1 Planner envelope")
            _load_cvoi_formal_v2_navsim_e120_exact_state(
                self._cvoi_direct_p1_planner,
                artifacts.p1_checkpoint["planner"],
                role="P1 planner",
            )

        if artifacts.calibration_checkpoint is not None:
            self._cvoi_direct_field_value = self._build_cvoi_direct_value_model(
                artifacts.calibration_checkpoint,
                encoder_embed_dim=encoder_embed_dim,
                device=device,
                role="Calibration",
            )
        if artifacts.stop_checkpoint is not None:
            self._cvoi_direct_stop_value = self._build_cvoi_direct_value_model(
                artifacts.stop_checkpoint,
                encoder_embed_dim=encoder_embed_dim,
                device=device,
                role="Stop",
            )
        if artifacts.gate_checkpoint is not None:
            if projection.oracle_sha256 is None or projection.gate_feature_mode is None:
                raise RuntimeError("direct EPDMS Gate projection identity is incomplete")
            self._cvoi_direct_gate = build_cvoi_direct_epdms_gate(
                artifacts.gate_checkpoint,
                branch=projection.branch,
                oracle_sha256=projection.oracle_sha256,
                gate_feature_mode=projection.gate_feature_mode,
                expected_latent_dim=encoder_embed_dim,
                device=device,
            )

        self._multiview_fusion = self._build_multiview_fusion(
            device,
            encoder_embed_dim,
        )
        if self._multiview_fusion is not None:
            raise ValueError("direct EPDMS manual checkpoints do not contain a multiview_fusion role")

        modules = [
            encoder,
            self._predictor,
            self._cvoi_direct_p0_planner,
            self._cvoi_direct_p1_planner,
            self._cvoi_direct_field_value,
            self._cvoi_direct_stop_value,
            self._cvoi_direct_gate,
        ]
        for module in modules:
            if module is None:
                continue
            module.to(device).eval()
            module.requires_grad_(False)
            if module.training or any(parameter.requires_grad for parameter in module.parameters()):
                raise RuntimeError("direct EPDMS model failed to freeze before scoring")
        self._cvoi_direct_runtime = runtime._replace(
            artifacts=_DirectEpdmsArtifacts(
                p0_checkpoint=None,
                calibration_checkpoint=None,
                p1_checkpoint=None,
                stop_checkpoint=None,
                gate_checkpoint=None,
            )
        )

    def _initialize_stage12(self, device: torch.device, encoder: nn.Module, encoder_embed_dim: int) -> None:
        """Stage-1/2 初始化：MultiModalTemporalPlanner + predictor rollout。"""
        manual_runtime = getattr(self, "_cvoi_manual_runtime", None)
        main_tokens_override, predictor_img_size_override = resolve_main_predictor_runtime_overrides(
            self._config, encoder
        )
        main_full_timeline = resolve_main_timeline(
            self._config,
            encoder=encoder,
            num_raw_frames=self._config.data.num_target_frames,
        )
        self._main_context_timeline = main_full_timeline

        logger.info("Initializing predictor runtime (stage12)...")
        predictor, token_ae, tokens_per_frame, runtime_normalize_reps = init_predictor_runtime_with_token_ae(
            self._config,
            device=device,
            encoder_embed_dim=encoder_embed_dim,
            raw_tokens_per_frame_override=main_tokens_override,
            predictor_img_size_override=predictor_img_size_override,
        )
        self._predictor = predictor
        self._token_ae = token_ae
        self._tokens_per_frame = tokens_per_frame
        self._runtime_normalize_reps = runtime_normalize_reps

        logger.info("Initializing planner (stage12)...")
        self._planner = init_planner(
            self._config,
            encoder_embed_dim,
            device,
            tokens_per_frame_override=tokens_per_frame,
        )
        if self._cvoi_direct_runtime is not None:
            self._initialize_cvoi_direct_stage12(
                device=device,
                encoder=encoder,
                encoder_embed_dim=encoder_embed_dim,
                tokens_per_frame=tokens_per_frame,
                main_full_timeline=main_full_timeline,
            )
            return
        if value_planning_enabled(self._config):
            self._value_head = TemporalTrajectoryValueHead(
                embed_dim=int(encoder_embed_dim),
                hidden_dim=int(getattr(self._config.value_planning, "hidden_dim", 512)),
                dropout=0.0,
            ).to(device)
        self._cvoi_dual_value = load_cvoi_dual_value_model(
            self._config,
            embed_dim=int(encoder_embed_dim),
            device=device,
        )
        self._cvoi_dual_value_adapter = (
            None if self._cvoi_dual_value is None else CvoiValueDtypeAdapter(self._cvoi_dual_value)
        )
        if manual_runtime is not None:
            original_evaluation_mode = self._config.cvoi.evaluation_mode
            self._config.cvoi.evaluation_mode = "controller"
            try:
                self._cvoi_navtrain_stop_value = load_cvoi_dual_value_model(
                    self._config,
                    embed_dim=int(encoder_embed_dim),
                    device=device,
                )
            finally:
                self._config.cvoi.evaluation_mode = original_evaluation_mode
            if self._cvoi_navtrain_stop_value is None:
                raise RuntimeError("manual NavTrain scorer failed to load its direct Stop value model")
            self._cvoi_navtrain_stop_value_adapter = CvoiValueDtypeAdapter(self._cvoi_navtrain_stop_value)
        self._cvoi_gate = load_cvoi_gate_for_evaluation(self._config, device=device)
        self._multiview_fusion = self._build_multiview_fusion(
            device,
            encoder_embed_dim,
            tokens_per_frame_override=main_tokens_override,
        )

        if self._planner is None:
            raise ValueError("Planner must be enabled (planner.use_planner=true) for PDMS evaluation")

        maybe_register_parallel_predictor_tokens(
            predictor=self._predictor,
            config=self._config,
            embed_dim=encoder_embed_dim,
            future_steps=main_full_timeline.num_future_steps,
            tokens_per_frame=self._tokens_per_frame,
            device=device,
        )

        # Manual selected checkpoints are exact full-state envelopes; loading a second
        # pretrained source would violate the single-checkpoint encoder/predictor/planner contract.
        manual_e120 = manual_runtime is not None
        if manual_e120:
            pretrained_modules = {"predictor": False}
        else:
            pretrained_modules = load_pretrained_checkpoint(
                self._config.meta.pretrain_checkpoint_full,
                self._encoder,
                None,
                self._predictor,
                None,
                None,
                self._planner,
                load_encoder=False,
                load_predictor=self._config.meta.load_predictor,
                load_seg=False,
                load_planner=False,
                context_encoder_key=self._config.meta.context_encoder_key,
                target_encoder_key=self._config.meta.target_encoder_key,
                rank=0,
                world_size=1,
                predictor_checkpoint=self._config.meta.predictor_checkpoint,
            )
        predictor_loaded_from_pretrained = pretrained_modules.get("predictor", False)

        # 加载主 checkpoint
        if self.checkpoint_path and (manual_e120 or os.path.exists(self.checkpoint_path)):
            logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
            if manual_e120:
                if self._cvoi_manual_artifacts is None:
                    raise RuntimeError("manual NavTrain P1 artifact was not preflighted")
                checkpoint = self._cvoi_manual_artifacts.p1_checkpoint
            else:
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
            if cvoi_enabled(self._config):
                checkpoint = self._validate_configured_cvoi_planner_checkpoint(checkpoint)

            # 加载 encoder
            encoder_loaded = False
            if not encoder_loaded:
                for key in [
                    self._config.meta.context_encoder_key,
                    "encoder",
                    "target_encoder",
                    "ema_encoder",
                ]:
                    if key in checkpoint:
                        if manual_e120:
                            _load_cvoi_formal_v2_navsim_e120_exact_state(
                                self._encoder,
                                checkpoint[key],
                                role="encoder",
                            )
                        else:
                            load_state_dict_helper(self._encoder, checkpoint[key], f"encoder (from '{key}')")
                        encoder_loaded = True
                        break
            if not encoder_loaded:
                if self.encoder_checkpoint_path and os.path.exists(self.encoder_checkpoint_path):
                    logger.info(f"Loading encoder from separate checkpoint: {self.encoder_checkpoint_path}")
                    encoder_ckpt = torch.load(self.encoder_checkpoint_path, map_location=device)
                    for key in [
                        self._config.meta.context_encoder_key,
                        "encoder",
                        "target_encoder",
                        "ema_encoder",
                    ]:
                        if key in encoder_ckpt:
                            load_state_dict_helper(
                                self._encoder, encoder_ckpt[key], f"encoder (from separate file, key='{key}')"
                            )
                            encoder_loaded = True
                            break
                if not encoder_loaded:
                    raise RuntimeError(
                        "Eval agent found no encoder weights in the main checkpoint or the separate encoder "
                        f"file (checkpoint={self.checkpoint_path!r}, encoder_ckpt={self.encoder_checkpoint_path!r}); "
                        "refusing to evaluate a randomly-initialized encoder."
                    )

            # 加载 predictor
            if "predictor" in checkpoint:
                if manual_e120:
                    _load_cvoi_formal_v2_navsim_e120_exact_state(
                        self._predictor,
                        checkpoint["predictor"],
                        role="predictor",
                    )
                else:
                    load_state_dict_helper(self._predictor, checkpoint["predictor"], "predictor")
                predictor_loaded_from_pretrained = True
            elif predictor_loaded_from_pretrained:
                logger.info("Stage12 main checkpoint has no predictor; using pretrained predictor")
            else:
                raise RuntimeError(
                    f"Eval agent found no predictor weights (checkpoint {self.checkpoint_path!r} has no "
                    "'predictor' key and none were loaded from a pretrained source); refusing to evaluate a "
                    "randomly-initialized predictor."
                )

            # 加载 planner
            if "planner" in checkpoint:
                planner_state = checkpoint["planner"]
                proposal_core_state = self._extract_proposal_core_planner_state(planner_state, self._config)
                if proposal_core_state is not None:
                    self._planner = self._build_stage12_proposal_core_planner(
                        device=device,
                        encoder_embed_dim=encoder_embed_dim,
                        tokens_per_frame=tokens_per_frame,
                    )
                    self._stage12_planner_uses_full_context = True
                    logger.info(
                        "Loading stage12 planner from TransformerProposalProvider core.* checkpoint state "
                        "(proposal.enabled=true)"
                    )
                    if manual_e120:
                        _load_cvoi_formal_v2_navsim_e120_exact_state(
                            self._planner,
                            proposal_core_state,
                            role="planner",
                        )
                    else:
                        load_state_dict_helper(self._planner, proposal_core_state, "planner (proposal core)")
                else:
                    if manual_e120:
                        _load_cvoi_formal_v2_navsim_e120_exact_state(
                            self._planner,
                            planner_state,
                            role="planner",
                        )
                    else:
                        load_state_dict_helper(self._planner, planner_state, "planner")
            else:
                raise RuntimeError(
                    f"Eval agent found no planner weights ('planner' key missing from checkpoint "
                    f"{self.checkpoint_path!r}); refusing to evaluate a randomly-initialized planner."
                )
            if self._value_head is not None:
                if "value_head" not in checkpoint:
                    raise RuntimeError(
                        "value_planning.enabled=true but checkpoint has no 'value_head'; "
                        "value-guided inference requires a trained TemporalTrajectoryValueHead."
                    )
                load_state_dict_helper(self._value_head, checkpoint["value_head"], "value_head")
            if self._multiview_fusion is not None and "multiview_fusion" in checkpoint:
                load_state_dict_helper(self._multiview_fusion, checkpoint["multiview_fusion"], "multiview_fusion")
            budget_controller_active = (
                bool(getattr(getattr(self._config, "budget_controller", None), "enabled", False))
                and getattr(self._config.budget_controller, "mode", None) == "eval"
            )
            if budget_controller_active:
                budget_controller_checkpoint = getattr(self._config.budget_controller, "controller_checkpoint", None)
                if not budget_controller_checkpoint:
                    raise ValueError(
                        "budget_controller.enabled=true with mode='eval' requires "
                        "budget_controller.controller_checkpoint"
                    )
                self._budget_controller = load_budget_controller_from_checkpoint(
                    budget_controller_checkpoint,
                    device=device,
                )
                logger.info("budget_controller: loaded eval controller from %s", budget_controller_checkpoint)

            logger.info("Stage12 checkpoint loaded successfully!")
        else:
            raise FileNotFoundError(
                f"Eval agent checkpoint not found (path={self.checkpoint_path!r}); refusing to produce eval "
                "scores from randomly-initialized weights."
            )

        self._encoder.to(device).eval()
        self._predictor.to(device).eval()
        self._planner.to(device).eval()
        if self._value_head is not None:
            self._value_head.to(device).eval()
        if self._cvoi_dual_value is not None:
            self._cvoi_dual_value.to(device).eval()
        if self._cvoi_dual_value_adapter is not None:
            self._cvoi_dual_value_adapter.to(device).eval()
        if self._cvoi_navtrain_stop_value is not None:
            self._cvoi_navtrain_stop_value.to(device).eval()
        if self._cvoi_navtrain_stop_value_adapter is not None:
            self._cvoi_navtrain_stop_value_adapter.to(device).eval()
        if self._cvoi_gate is not None:
            self._cvoi_gate.to(device).eval()
        if self._budget_controller is not None:
            self._budget_controller.to(device).eval()
        if self._multiview_fusion is not None:
            self._multiview_fusion.to(device).eval()

    def _build_multiview_fusion(
        self,
        device: torch.device,
        encoder_embed_dim: int,
        tokens_per_frame_override: Optional[int] = None,
    ) -> Optional[nn.Module]:
        cfg = self._config
        if cfg is None or not getattr(cfg.multiview, "enabled", False):
            return None
        if cfg.multiview.fusion_type != "petr_cross_attn":
            raise ValueError(f"Unsupported multiview.fusion_type={cfg.multiview.fusion_type!r}")
        del tokens_per_frame_override
        raw_tokens_per_frame = resolve_main_encoder_raw_tokens_per_frame(cfg, self._encoder)
        return PETRMultiViewFusion(
            embed_dim=int(encoder_embed_dim),
            tokens_per_frame=raw_tokens_per_frame,
            hidden_dim=int(cfg.multiview.hidden_dim),
            num_heads=int(cfg.multiview.num_heads),
            dropout=float(cfg.multiview.dropout),
            output_mode=str(getattr(cfg.multiview, "output_mode", "fused")),
        ).to(device)

    def _initialize_encoder_direct(self, device: torch.device, encoder: nn.Module, encoder_embed_dim: int) -> None:
        """Encoder-direct 初始化：encoder 观测 token 直接输入 planner，不构建 predictor。"""
        cfg = self._config

        logger.info("Initializing planner (encoder_direct)...")
        planner = init_encoder_direct_planner(cfg, encoder_embed_dim, device, encoder=encoder)
        if planner is None:
            raise ValueError("Planner must be enabled (planner.use_planner=true) for encoder_direct PDMS evaluation")

        self._planner = planner
        self._predictor = None
        self._tokens_per_frame = resolve_encoder_direct_tokens_per_frame(cfg, encoder)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info(f"Loading encoder-direct checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=device)

            encoder_loaded = False
            for key in [cfg.meta.context_encoder_key, "encoder", "target_encoder", "ema_encoder"]:
                if key in checkpoint:
                    load_state_dict_helper(encoder, checkpoint[key], f"encoder_direct encoder (from '{key}')")
                    encoder_loaded = True
                    break
            if not encoder_loaded and self.encoder_checkpoint_path and os.path.exists(self.encoder_checkpoint_path):
                logger.info(f"Loading encoder from separate checkpoint: {self.encoder_checkpoint_path}")
                encoder_ckpt = torch.load(self.encoder_checkpoint_path, map_location=device)
                for key in [cfg.meta.context_encoder_key, "encoder", "target_encoder", "ema_encoder"]:
                    if key in encoder_ckpt:
                        load_state_dict_helper(
                            encoder,
                            encoder_ckpt[key],
                            f"encoder_direct encoder (from separate file, key='{key}')",
                        )
                        encoder_loaded = True
                        break
            if not encoder_loaded:
                raise RuntimeError(
                    "Eval agent (encoder-direct) found no encoder weights in the checkpoint or the separate "
                    f"encoder file (checkpoint={self.checkpoint_path!r}, "
                    f"encoder_ckpt={self.encoder_checkpoint_path!r}); "
                    "refusing to evaluate a randomly-initialized encoder."
                )

            if "planner" in checkpoint:
                adapt_encoder_direct_action_history_from_checkpoint(
                    SimpleNamespace(planner=planner),
                    checkpoint["planner"],
                )
                load_state_dict_helper(planner, checkpoint["planner"], "encoder_direct planner")
            else:
                raise RuntimeError(
                    f"Eval agent (encoder-direct) found no planner weights ('planner' key missing from "
                    f"checkpoint {self.checkpoint_path!r}); refusing to evaluate a randomly-initialized planner."
                )

            logger.info("Encoder-direct checkpoint loaded successfully!")
        else:
            raise FileNotFoundError(
                f"Eval agent checkpoint not found (path={self.checkpoint_path!r}); refusing to produce eval "
                "scores from randomly-initialized weights."
            )

        encoder.to(device).eval()
        planner.to(device).eval()

    def _initialize_stage2(
        self,
        device: torch.device,
        encoder: nn.Module,
        target_encoder: nn.Module,
        encoder_embed_dim: int,
    ) -> None:
        """Stage-2 初始化：frozen proposal provider + RefinementDecoder。"""
        from app.vjepa_cowa_world_model.models import RefinementDecoder
        from app.vjepa_cowa_world_model.models.proposal_providers import build_proposal_provider

        cfg = self._config

        if not getattr(cfg, "proposal", None) or not cfg.proposal.enabled:
            raise ValueError("Stage-2 refinement requires proposal.enabled=true")
        if not cfg.proposal.freeze:
            raise ValueError("Stage-2 refinement expects proposal.freeze=true")
        if cfg.proposal.provider_type != "history_kinematic" and not cfg.proposal.checkpoint:
            if not self._proposal_checkpoint_path:
                raise ValueError(
                    "Stage-2 frozen transformer/diffusion proposal requires "
                    "proposal.checkpoint in config or proposal_checkpoint_path parameter"
                )

        cfg.planner.use_planner = True
        cfg.planner.use_z_context = True
        cfg.refinement.num_modes = cfg.proposal.num_modes
        cfg.planner.num_modes = cfg.proposal.num_modes
        cfg.segmentation.use_segmentation = False

        main_tokens_override, predictor_img_size_override = resolve_main_predictor_runtime_overrides(cfg, encoder)
        predictor, token_ae, tokens_per_frame, runtime_normalize_reps = init_predictor_runtime_with_token_ae(
            cfg,
            device=device,
            encoder_embed_dim=encoder_embed_dim,
            raw_tokens_per_frame_override=main_tokens_override,
            predictor_img_size_override=predictor_img_size_override,
        )
        self._predictor = predictor
        self._token_ae = token_ae
        self._tokens_per_frame = tokens_per_frame
        self._runtime_normalize_reps = runtime_normalize_reps

        self._main_context_timeline = resolve_main_timeline(
            cfg, encoder=encoder, num_raw_frames=cfg.train.num_observed_frames
        )
        logger.info(
            "Main encoder timeline: raw_observed=%d stride=%d observed_steps=%d tokens_per_step=%d",
            self._main_context_timeline.raw_num_frames,
            self._main_context_timeline.frame_stride,
            self._main_context_timeline.num_observed_steps,
            self._main_context_timeline.tokens_per_frame,
        )

        raw_future_frames = cfg.data.num_target_frames - cfg.train.num_observed_frames
        if raw_future_frames % self._main_context_timeline.frame_stride != 0:
            raise ValueError(
                f"Stage-2 future horizon must be divisible by frame stride: "
                f"future_frames={raw_future_frames}, stride={self._main_context_timeline.frame_stride}"
            )
        maybe_register_parallel_predictor_tokens(
            predictor=predictor,
            config=cfg,
            embed_dim=encoder_embed_dim,
            future_steps=raw_future_frames // self._main_context_timeline.frame_stride,
            tokens_per_frame=tokens_per_frame,
            device=device,
        )

        num_poses = cfg.data.num_target_frames - cfg.train.num_observed_frames
        status_dim = resolve_effective_planner_status_dim(cfg)
        use_cmd = resolve_planner_use_drive_command(cfg)
        command_dim = (
            4 if (use_cmd and cfg.planner.split_status_embedding and cfg.train.predictor_inference_consistent) else 0
        )
        planner = RefinementDecoder(
            encoder_dim=encoder_embed_dim,
            tf_d_model=cfg.planner.tf_d_model,
            tf_d_ffn=cfg.planner.tf_d_ffn,
            tf_num_layers=cfg.planner.tf_num_layers,
            tf_num_head=cfg.planner.tf_num_head,
            tf_dropout=cfg.planner.tf_dropout,
            tokens_per_frame=tokens_per_frame,
            num_poses=num_poses,
            num_time_steps=num_poses,
            num_context_frames=self._main_context_timeline.num_observed_steps,
            status_dim=status_dim,
            use_spatial_tokens=cfg.planner.use_spatial_tokens,
            num_modes=cfg.refinement.num_modes,
            use_temporal=True,
            use_time_aligned_bias=cfg.planner.temporal_alignment,
            use_status_for_planner=cfg.planner.use_status_for_planner,
            command_dim=command_dim,
            max_rounds=max(2, cfg.refinement.inference_num_rounds, cfg.refinement_gated.num_rounds),
        ).to(device)
        self._planner = planner

        proposal_encoder = None
        proposal_encoder_embed_dim = encoder_embed_dim
        if cfg.proposal.use_separate_encoder and cfg.proposal.provider_type != "history_kinematic":
            proposal_encoder = init_proposal_encoder(cfg, device)
            proposal_encoder_embed_dim = get_encoder_embed_dim(proposal_encoder)
            freeze_module_eval(proposal_encoder)
        elif cfg.proposal.use_separate_encoder:
            logger.info("Skipping separate proposal_encoder for history_kinematic proposal provider.")
        self._proposal_encoder = proposal_encoder

        proposal_tokens_per_frame = resolve_proposal_tokens_per_frame(cfg, proposal_encoder)
        proposal_num_context_frames = resolve_proposal_num_time_steps(cfg, proposal_encoder)
        proposal_planner = build_proposal_provider(
            config=cfg,
            encoder_dim=proposal_encoder_embed_dim,
            tokens_per_frame=proposal_tokens_per_frame,
            num_poses=num_poses,
            status_dim=status_dim,
            command_dim=command_dim,
            num_context_frames=proposal_num_context_frames,
            num_observed_frames=cfg.train.num_observed_frames,
        ).to(device)
        freeze_module_eval(proposal_planner)
        self._proposal_planner = proposal_planner

        self._proposal_runtime_normalize_reps = resolve_proposal_runtime_normalize_reps(cfg)
        self._proposal_token_ae = resolve_proposal_token_ae_module(cfg, token_ae)

        configure_vjepa_encoder_trainability(encoder, cfg, trainable=False)
        configure_vjepa_encoder_trainability(target_encoder, cfg, trainable=False)

        pretrained_modules = load_pretrained_checkpoint(
            cfg.meta.pretrain_checkpoint_full,
            encoder,
            target_encoder,
            predictor,
            None,
            None,
            planner,
            load_encoder=cfg.meta.load_encoder,
            load_predictor=cfg.meta.load_predictor,
            load_seg=False,
            load_planner=False,
            context_encoder_key=cfg.meta.context_encoder_key,
            target_encoder_key=cfg.meta.target_encoder_key,
            rank=0,
            world_size=1,
            predictor_checkpoint=cfg.meta.predictor_checkpoint,
        )
        predictor_loaded_from_pretrained = pretrained_modules.get("predictor", False)

        proposal_ckpt_path = self._proposal_checkpoint_path or getattr(cfg.proposal, "checkpoint", "")
        if proposal_ckpt_path:
            load_frozen_proposal_provider(proposal_planner, proposal_ckpt_path, config=cfg)
            logger.info(f"Proposal planner loaded from: {proposal_ckpt_path}")
        load_frozen_proposal_encoder(proposal_encoder, cfg)

        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info(f"Loading main checkpoint from: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=device)

            encoder_loaded = False
            for key in [cfg.meta.context_encoder_key, "encoder", "target_encoder", "ema_encoder"]:
                if key in checkpoint:
                    load_state_dict_helper(encoder, checkpoint[key], f"encoder (from '{key}')")
                    encoder_loaded = True
                    break
            if not encoder_loaded:
                if self.encoder_checkpoint_path and os.path.exists(self.encoder_checkpoint_path):
                    logger.info(f"Loading encoder from separate checkpoint: {self.encoder_checkpoint_path}")
                    encoder_ckpt = torch.load(self.encoder_checkpoint_path, map_location=device)
                    for key in [cfg.meta.context_encoder_key, "encoder", "target_encoder", "ema_encoder"]:
                        if key in encoder_ckpt:
                            load_state_dict_helper(
                                encoder, encoder_ckpt[key], f"encoder (from separate file, key='{key}')"
                            )
                            encoder_loaded = True
                            break
                if not encoder_loaded:
                    raise RuntimeError(
                        "Eval agent found no encoder weights in the main checkpoint or the separate encoder "
                        f"file (checkpoint={self.checkpoint_path!r}, encoder_ckpt={self.encoder_checkpoint_path!r}); "
                        "refusing to evaluate a randomly-initialized encoder."
                    )

            if "predictor" in checkpoint:
                load_state_dict_helper(predictor, checkpoint["predictor"], "predictor")
                predictor_loaded_from_pretrained = True
            elif predictor_loaded_from_pretrained:
                logger.info("Stage2 main checkpoint has no predictor; using pretrained predictor")
            else:
                raise RuntimeError(
                    f"Eval agent found no predictor weights (checkpoint {self.checkpoint_path!r} has no "
                    "'predictor' key and none were loaded from a pretrained source); refusing to evaluate a "
                    "randomly-initialized predictor."
                )

            if "planner" in checkpoint:
                load_state_dict_helper(planner, checkpoint["planner"], "planner")
            else:
                raise RuntimeError(
                    f"Eval agent found no planner weights ('planner' key missing from checkpoint "
                    f"{self.checkpoint_path!r}); refusing to evaluate a randomly-initialized planner."
                )

            logger.info("Stage2 main checkpoint loaded successfully!")
        else:
            raise FileNotFoundError(
                f"Eval agent checkpoint not found (path={self.checkpoint_path!r}); refusing to produce eval "
                "scores from randomly-initialized weights."
            )

        for m in [encoder, predictor, planner, proposal_planner, proposal_encoder, token_ae, self._proposal_token_ae]:
            if m is not None:
                m.to(device).eval()

    def _initialize_stage3(
        self,
        device: torch.device,
        encoder: nn.Module,
        target_encoder: nn.Module,
        encoder_embed_dim: int,
    ) -> None:
        """Stage-3 初始化：RefinementDecoder + frozen proposal provider + conditional predictor rollout。"""
        from app.vjepa_cowa_world_model.models import build_refinement_decoder
        from app.vjepa_cowa_world_model.models.proposal_providers import build_proposal_provider

        cfg = self._config

        # 1. Config 校验与覆盖 (与 train_refinement_gated.py:149-158 一致)
        if not getattr(cfg, "proposal", None) or not cfg.proposal.enabled:
            raise ValueError("Stage-3 requires proposal.enabled=true in config")
        if not cfg.proposal.freeze:
            raise ValueError("Stage-3 expects proposal.freeze=true in config")
        if cfg.proposal.provider_type != "history_kinematic" and not cfg.proposal.checkpoint:
            if not self._proposal_checkpoint_path:
                raise ValueError(
                    "Stage-3 frozen transformer/diffusion proposal requires "
                    "proposal.checkpoint in config or proposal_checkpoint_path parameter"
                )

        cfg.planner.use_planner = True
        cfg.planner.use_z_context = True
        cfg.planner.num_modes = cfg.proposal.num_modes
        cfg.segmentation.use_segmentation = False

        # 2. Init predictor with TokenAE
        main_tokens_override, predictor_img_size_override = resolve_main_predictor_runtime_overrides(cfg, encoder)
        predictor, token_ae, tokens_per_frame, runtime_normalize_reps = init_predictor_runtime_with_token_ae(
            cfg,
            device=device,
            encoder_embed_dim=encoder_embed_dim,
            raw_tokens_per_frame_override=main_tokens_override,
            predictor_img_size_override=predictor_img_size_override,
        )
        self._predictor = predictor
        self._token_ae = token_ae
        self._tokens_per_frame = tokens_per_frame
        self._runtime_normalize_reps = runtime_normalize_reps

        # 3. Resolve main timeline
        self._main_context_timeline = resolve_main_timeline(
            cfg, encoder=encoder, num_raw_frames=cfg.train.num_observed_frames
        )
        logger.info(
            "Main encoder timeline: raw_observed=%d stride=%d observed_steps=%d tokens_per_step=%d",
            self._main_context_timeline.raw_num_frames,
            self._main_context_timeline.frame_stride,
            self._main_context_timeline.num_observed_steps,
            self._main_context_timeline.tokens_per_frame,
        )

        # 4. Register parallel predictor tokens
        raw_future_frames = cfg.data.num_target_frames - cfg.train.num_observed_frames
        if raw_future_frames % self._main_context_timeline.frame_stride != 0:
            raise ValueError(
                f"Stage-3 future horizon must be divisible by frame stride: "
                f"future_frames={raw_future_frames}, stride={self._main_context_timeline.frame_stride}"
            )
        maybe_register_parallel_predictor_tokens(
            predictor=predictor,
            config=cfg,
            embed_dim=encoder_embed_dim,
            future_steps=raw_future_frames // self._main_context_timeline.frame_stride,
            tokens_per_frame=tokens_per_frame,
            device=device,
        )

        stage3_checkpoint = None
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info(f"Inspecting Stage3 checkpoint architecture from: {self.checkpoint_path}")
            stage3_checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            self._maybe_set_legacy_stage3_refinement_core_type(cfg, stage3_checkpoint)

        # 5. Build RefinementDecoder (planner)
        num_poses = cfg.data.num_target_frames - cfg.train.num_observed_frames
        status_dim = resolve_effective_planner_status_dim(cfg)
        use_cmd = resolve_planner_use_drive_command(cfg)
        command_dim = (
            4 if (use_cmd and cfg.planner.split_status_embedding and cfg.train.predictor_inference_consistent) else 0
        )
        planner = build_refinement_decoder(
            config=cfg,
            encoder_dim=encoder_embed_dim,
            tokens_per_frame=tokens_per_frame,
            num_poses=num_poses,
            status_dim=status_dim,
            command_dim=command_dim,
            main_num_context_frames=self._main_context_timeline.num_observed_steps,
        ).to(device)
        self._planner = planner

        # 6. Build proposal_planner and freeze
        proposal_encoder = None
        proposal_encoder_embed_dim = encoder_embed_dim
        if cfg.proposal.use_separate_encoder and cfg.proposal.provider_type != "history_kinematic":
            proposal_encoder = init_proposal_encoder(cfg, device)
            proposal_encoder_embed_dim = get_encoder_embed_dim(proposal_encoder)
            freeze_module_eval(proposal_encoder)
        self._proposal_encoder = proposal_encoder

        proposal_tokens_per_frame = resolve_proposal_tokens_per_frame(cfg, proposal_encoder)
        proposal_num_context_frames = resolve_proposal_num_time_steps(cfg, proposal_encoder)
        proposal_planner = build_proposal_provider(
            config=cfg,
            encoder_dim=proposal_encoder_embed_dim,
            tokens_per_frame=proposal_tokens_per_frame,
            num_poses=num_poses,
            status_dim=status_dim,
            command_dim=command_dim,
            num_context_frames=proposal_num_context_frames,
            num_observed_frames=cfg.train.num_observed_frames,
        ).to(device)
        freeze_module_eval(proposal_planner)
        self._proposal_planner = proposal_planner

        # 7. Resolve proposal runtime params
        self._proposal_runtime_normalize_reps = resolve_proposal_runtime_normalize_reps(cfg)
        self._proposal_token_ae = resolve_proposal_token_ae_module(cfg, token_ae)

        # 8. Configure V-JEPA encoder trainability
        configure_vjepa_encoder_trainability(encoder, cfg, trainable=False)

        # 9. Load checkpoints
        # 9a. Load pretrained encoder/predictor fallback, then stage checkpoint overrides trainable modules.
        pretrained_modules = load_pretrained_checkpoint(
            cfg.meta.pretrain_checkpoint_full,
            encoder,
            target_encoder,
            predictor,
            None,
            None,
            planner,
            load_encoder=cfg.meta.load_encoder,
            load_predictor=cfg.meta.load_predictor,
            load_seg=False,
            load_planner=False,
            context_encoder_key=cfg.meta.context_encoder_key,
            target_encoder_key=cfg.meta.target_encoder_key,
            rank=0,
            world_size=1,
            predictor_checkpoint=cfg.meta.predictor_checkpoint,
        )
        predictor_loaded_from_pretrained = pretrained_modules.get("predictor", False)

        # 9b. Load proposal_planner from proposal checkpoint
        proposal_ckpt_path = self._proposal_checkpoint_path or getattr(cfg.proposal, "checkpoint", "")
        if proposal_ckpt_path:
            load_frozen_proposal_provider(proposal_planner, proposal_ckpt_path, config=cfg)
            logger.info(f"Proposal planner loaded from: {proposal_ckpt_path}")

        # 9c. Load proposal_encoder
        load_frozen_proposal_encoder(proposal_encoder, cfg)

        # 9d. Load main checkpoint (encoder/predictor/planner)
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            logger.info(f"Loading main checkpoint from: {self.checkpoint_path}")
            checkpoint = stage3_checkpoint
            if checkpoint is None:
                checkpoint = torch.load(self.checkpoint_path, map_location="cpu")

            encoder_loaded = False
            for key in [cfg.meta.context_encoder_key, "encoder", "target_encoder", "ema_encoder"]:
                if key in checkpoint:
                    load_state_dict_helper(encoder, checkpoint[key], f"encoder (from '{key}')")
                    encoder_loaded = True
                    break
            if not encoder_loaded:
                if self.encoder_checkpoint_path and os.path.exists(self.encoder_checkpoint_path):
                    logger.info(f"Loading encoder from separate checkpoint: {self.encoder_checkpoint_path}")
                    encoder_ckpt = torch.load(self.encoder_checkpoint_path, map_location=device)
                    for key in [cfg.meta.context_encoder_key, "encoder", "target_encoder", "ema_encoder"]:
                        if key in encoder_ckpt:
                            load_state_dict_helper(
                                encoder, encoder_ckpt[key], f"encoder (from separate file, key='{key}')"
                            )
                            encoder_loaded = True
                            break
                if not encoder_loaded:
                    raise RuntimeError(
                        "Eval agent found no encoder weights in the main checkpoint or the separate encoder "
                        f"file (checkpoint={self.checkpoint_path!r}, encoder_ckpt={self.encoder_checkpoint_path!r}); "
                        "refusing to evaluate a randomly-initialized encoder."
                    )

            if "predictor" in checkpoint:
                load_state_dict_helper(predictor, checkpoint["predictor"], "predictor")
                predictor_loaded_from_pretrained = True
            elif predictor_loaded_from_pretrained:
                logger.info("Stage3 main checkpoint has no predictor; using pretrained predictor")
            else:
                raise RuntimeError(
                    f"Eval agent found no predictor weights (checkpoint {self.checkpoint_path!r} has no "
                    "'predictor' key and none were loaded from a pretrained source); refusing to evaluate a "
                    "randomly-initialized predictor."
                )

            if "planner" in checkpoint:
                load_state_dict_helper(planner, checkpoint["planner"], "planner")
            else:
                raise RuntimeError(
                    f"Eval agent found no planner weights ('planner' key missing from checkpoint "
                    f"{self.checkpoint_path!r}); refusing to evaluate a randomly-initialized planner."
                )

            logger.info("Stage3 main checkpoint loaded successfully!")
        else:
            raise FileNotFoundError(
                f"Eval agent checkpoint not found (path={self.checkpoint_path!r}); refusing to produce eval "
                "scores from randomly-initialized weights."
            )

        # 10. Set all modules to eval
        for m in [encoder, predictor, planner, proposal_planner, proposal_encoder, token_ae, self._proposal_token_ae]:
            if m is not None:
                m.to(device).eval()

    def _requires_proposal_video_clip(self) -> bool:
        cfg = self._config
        return (
            self._forward_mode in {"stage2", "stage3"}
            and cfg is not None
            and self._proposal_encoder is not None
            and cfg.proposal.provider_type != "history_kinematic"
        )

    def _build_feature_builder(self) -> VJEPAFeatureBuilder:
        crop_size = self.crop_size
        crop_top_bottom = None
        if self._config is not None and is_vjepa_img_encoder(self._config):
            crop_size = self._config.model.vjepa_resolution
            crop_top_bottom = self._config.model.vjepa_crop_top_bottom
            logger.info(
                "Enabling main video_clip V-JEPA transform with resolution=%s, crop_top_bottom=%s",
                crop_size,
                crop_top_bottom,
            )

        proposal_crop_size = None
        proposal_crop_top_bottom = None
        if self._requires_proposal_video_clip():
            proposal_crop_size = self._config.proposal.vjepa_resolution
            if is_vjepa_proposal_encoder(self._proposal_encoder):
                proposal_crop_top_bottom = self._config.proposal.vjepa_crop_top_bottom
            logger.info(
                "Enabling proposal_video_clip with resolution=%s, crop_top_bottom=%s",
                proposal_crop_size,
                proposal_crop_top_bottom,
            )
        manual_runtime = getattr(self, "_cvoi_manual_runtime", None)
        direct_runtime = getattr(self, "_cvoi_direct_runtime", None)
        if direct_runtime is not None and direct_runtime.evaluation_seed is None:
            raise RuntimeError("direct EPDMS feature identity requires a resolved evaluation seed")
        return VJEPAFeatureBuilder(
            crop_size=crop_size,
            camera_name=self.camera_name,
            camera_names=self.camera_names if len(self.camera_names) > 1 else None,
            action_dim=self._action_dim,
            crop_top_bottom=crop_top_bottom,
            proposal_crop_size=proposal_crop_size,
            proposal_crop_top_bottom=proposal_crop_top_bottom,
            official_cvoi_identity=manual_runtime is not None or direct_runtime is not None,
            cvoi_evaluation_seed=(
                manual_runtime.common_random_seed
                if manual_runtime is not None
                else (direct_runtime.evaluation_seed if direct_runtime is not None else None)
            ),
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        if self._feature_builder is None:
            # 在 initialize() 之前被调用时，使用默认参数创建
            self._feature_builder = self._build_feature_builder()
        return [self._feature_builder]

    def compute_trajectory(
        self,
        agent_input: Any,
        past_ego_simulated_states: Any = None,
        metric_cache: Any = None,
        observation_interval: Any = None,
    ) -> Trajectory:
        """Execute ordinary, manual NavTrain, or direct NavTest evaluation."""

        manual_runtime = getattr(self, "_cvoi_manual_runtime", None)
        direct_runtime = getattr(self, "_cvoi_direct_runtime", None)
        if manual_runtime is not None:
            return self._compute_manual_navtrain_trajectory(agent_input, manual_runtime)
        if direct_runtime is not None:
            return self._compute_direct_epdms_trajectory(agent_input, direct_runtime)
        if getattr(self, "_encoder", None) is None:
            base_compute = super().compute_trajectory
            if len(inspect.signature(base_compute).parameters) == 1:
                return base_compute(agent_input)
            return base_compute(
                agent_input,
                past_ego_simulated_states,
                metric_cache,
                observation_interval,
            )
        self.eval()
        device = next(self._encoder.parameters()).device
        features: Dict[str, Any] = {}
        for builder in self.get_feature_builders():
            built = builder.compute_features(agent_input)
            if not isinstance(built, dict):
                raise TypeError("NavSim feature builder must return a dictionary")
            duplicate_names = set(features).intersection(built)
            if duplicate_names:
                raise ValueError(f"NavSim feature builders produced duplicate keys: {sorted(duplicate_names)}")
            features.update(built)
        batched_features = {
            name: value.unsqueeze(0).to(device) for name, value in features.items() if isinstance(value, torch.Tensor)
        }
        non_tensor_features = sorted(name for name, value in features.items() if not isinstance(value, torch.Tensor))
        if non_tensor_features:
            raise TypeError(f"NavSim model features must be tensors, got non-tensor keys {non_tensor_features}")
        if past_ego_simulated_states is not None:
            batched_features["past_ego_simulated_states"] = past_ego_simulated_states
        if metric_cache is not None:
            batched_features["metric_cache"] = metric_cache
        if observation_interval is not None:
            batched_features["observation_interval"] = observation_interval
        with torch.no_grad():
            predictions = self.forward(batched_features)
        if not isinstance(predictions, Mapping) or "trajectory" not in predictions:
            raise TypeError("NavSim forward must return a mapping containing 'trajectory'")
        trajectory = predictions["trajectory"]
        if not isinstance(trajectory, torch.Tensor) or trajectory.ndim != 3 or trajectory.shape[0] != 1:
            shape = None if not isinstance(trajectory, torch.Tensor) else tuple(trajectory.shape)
            raise ValueError(f"NavSim forward trajectory must have shape [1, num_poses, 3], got {shape}")
        poses = trajectory.squeeze(0).detach().cpu().numpy()
        return Trajectory(poses=poses, trajectory_sampling=self._cvoi_trajectory_sampling)

    def _compute_direct_epdms_trajectory(
        self,
        agent_input: Any,
        runtime: _DirectEpdmsRuntimeContext,
    ) -> Trajectory:
        """Execute one projected direct policy and publish its minimal NavTest trace."""

        if (
            runtime.scenario_tokens_by_observation_key is None
            or runtime.evaluation_seed is None
            or runtime.trace_output_dir is None
            or runtime.trace_output_dir_identity is None
        ):
            raise RuntimeError("direct EPDMS runtime identity is incomplete")
        self.eval()
        features: Dict[str, Any] = {}
        for builder in self.get_feature_builders():
            built = builder.compute_features(agent_input)
            if not isinstance(built, dict):
                raise TypeError("direct EPDMS feature builder must return a dictionary")
            duplicate_names = set(features).intersection(built)
            if duplicate_names:
                raise ValueError(f"direct EPDMS feature builders produced duplicate keys: {sorted(duplicate_names)}")
            features.update(built)
        try:
            observation_key_value = decode_observation_key(features.pop("cvoi_observation_key"))
            sample_seed = decode_unsigned_seed(features.pop("cvoi_rng_seed_bytes"))
        except KeyError as error:
            raise ValueError(f"direct EPDMS feature schema is missing {error.args[0]!r}") from error
        try:
            scenario_token = runtime.scenario_tokens_by_observation_key[observation_key_value]
        except KeyError as error:
            raise ValueError(
                f"observation key {observation_key_value!r} is absent from the direct NavTest scenario manifest"
            ) from error
        expected_seed = cvoi_sample_seed(runtime.evaluation_seed, observation_key_value)
        if sample_seed != expected_seed:
            raise ValueError(f"direct EPDMS sample seed mismatch: expected {expected_seed}, got {sample_seed}")

        batched_features: Dict[str, Any] = {}
        for name, value in features.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"direct EPDMS model feature {name!r} must be a tensor")
            batched_features[name] = value.unsqueeze(0)
        batched_features["cvoi_rng_seed"] = sample_seed
        prepared_features = self.prepare_cvoi_features(batched_features)

        projection = runtime.projection
        expected_gate_mode = projection.gate_feature_mode if projection.evaluation_mode == "controller" else None
        if self._cvoi_evaluation_gate_feature_mode != expected_gate_mode:
            raise RuntimeError("direct EPDMS runtime Gate mode differs from its effective projection")
        self.set_cvoi_evaluation_guidance_steps(projection.guidance_steps)
        self.set_cvoi_evaluation_forced_horizon(projection.horizon)
        self.set_cvoi_latency_mode(True)
        with torch.no_grad():
            predictions = self.forward(prepared_features)
        if not isinstance(predictions, Mapping) or "trajectory" not in predictions:
            raise TypeError("direct EPDMS forward must return a mapping containing 'trajectory'")
        trajectory = predictions["trajectory"]
        expected_num_poses = self._cvoi_trajectory_sampling.num_poses
        if (
            not isinstance(trajectory, torch.Tensor)
            or trajectory.ndim != 3
            or trajectory.shape != (1, expected_num_poses, 3)
        ):
            raise ValueError(
                "direct EPDMS trajectory must have exact shape "
                f"[1, {expected_num_poses}, 3], got {getattr(trajectory, 'shape', None)}"
            )
        if not bool(torch.isfinite(trajectory).all().item()):
            raise ValueError("direct EPDMS trajectory must contain only finite values")

        runtime_trace = self.get_last_cvoi_trace()
        final_horizon = runtime_trace.get("stop_horizon")
        if type(final_horizon) is not int or final_horizon not in range(5):
            raise RuntimeError(f"direct EPDMS runtime produced invalid final horizon {final_horizon!r}")
        if projection.evaluation_mode != "controller" and final_horizon != projection.horizon:
            raise RuntimeError(
                "direct EPDMS forced runtime horizon differs from its effective projection: "
                f"actual={final_horizon}, expected={projection.horizon}"
            )
        latency_components = self.get_last_cvoi_latency_components()
        if any(not math.isfinite(value) or value < 0.0 for value in latency_components.values()):
            raise RuntimeError("direct EPDMS latency components must be finite and non-negative")
        latency_ms = sum(latency_components.values())
        trace = {
            "schema": CVOI_DIRECT_EPDMS_TRACE_SCHEMA,
            "version": CVOI_DIRECT_EPDMS_TRACE_VERSION,
            "split": projection.split,
            "protocol": projection.protocol,
            "branch": projection.branch,
            "scenario_token": scenario_token,
            "evaluation_seed": runtime.evaluation_seed,
            "final_horizon": final_horizon,
            "latency_ms": latency_ms,
        }
        _write_exclusive_trace(
            runtime.trace_output_dir,
            observation_key_value,
            trace,
            expected_directory_identity=runtime.trace_output_dir_identity,
            destination_stem=scenario_token,
        )
        return Trajectory(
            poses=trajectory.squeeze(0).detach().cpu().numpy(),
            trajectory_sampling=self._cvoi_trajectory_sampling,
        )

    def _compute_manual_navtrain_trajectory(
        self,
        agent_input: Any,
        runtime: _ManualNavTrainRuntimeContext,
    ) -> Trajectory:
        """Execute one proof-free fixed-H P1 teacher call and publish its nine-field trace."""

        self._last_cvoi_navtrain_gate_features = None
        self.eval()
        features: Dict[str, Any] = {}
        for builder in self.get_feature_builders():
            built = builder.compute_features(agent_input)
            if not isinstance(built, dict):
                raise TypeError("manual NavTrain feature builder must return a dictionary")
            duplicate_names = set(features).intersection(built)
            if duplicate_names:
                raise ValueError(
                    f"manual NavTrain feature builders produced duplicate keys: {sorted(duplicate_names)}"
                )
            features.update(built)
        try:
            observation_key_value = decode_observation_key(features.pop("cvoi_observation_key"))
            sample_seed = decode_unsigned_seed(features.pop("cvoi_rng_seed_bytes"))
        except KeyError as exc:
            raise ValueError(f"manual NavTrain feature schema is missing {exc.args[0]!r}") from exc
        try:
            scenario_token = runtime.scenario_manifest.token_for_observation_key(observation_key_value)
        except KeyError as exc:
            raise ValueError(
                f"observation key {observation_key_value!r} is absent from the raw NavTrain scenario manifest"
            ) from exc
        expected_seed = cvoi_sample_seed(runtime.common_random_seed, observation_key_value)
        if sample_seed != expected_seed:
            raise ValueError(f"manual NavTrain sample seed mismatch: expected {expected_seed}, got {sample_seed}")

        batched_features: Dict[str, Any] = {}
        for name, value in features.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"manual NavTrain model feature {name!r} must be a tensor")
            batched_features[name] = value.unsqueeze(0)
        batched_features["cvoi_rng_seed"] = sample_seed
        prepared_features = self.prepare_cvoi_features(batched_features)

        if self._cvoi_gate is not None or self._cvoi_evaluation_gate_feature_mode is not None:
            raise RuntimeError("manual NavTrain fixed policy must not execute a Gate")
        self.set_cvoi_evaluation_guidance_steps(runtime.guidance_steps)
        self.set_cvoi_evaluation_forced_horizon(runtime.forced_horizon)
        self.set_cvoi_latency_mode(False)
        with torch.no_grad():
            predictions = self.forward(prepared_features)
        if not isinstance(predictions, Mapping) or "trajectory" not in predictions:
            raise TypeError("manual NavTrain forward must return a mapping containing 'trajectory'")
        trajectory = predictions["trajectory"]
        expected_num_poses = self._cvoi_trajectory_sampling.num_poses
        if (
            not isinstance(trajectory, torch.Tensor)
            or trajectory.ndim != 3
            or trajectory.shape != (1, expected_num_poses, 3)
        ):
            raise ValueError(
                "manual NavTrain forward trajectory must have shape "
                f"[1, {expected_num_poses}, 3], got {getattr(trajectory, 'shape', None)}"
            )
        if not bool(torch.isfinite(trajectory).all().item()):
            raise ValueError("manual NavTrain forward trajectory must contain only finite values")
        poses = trajectory.squeeze(0).detach().cpu().numpy()

        runtime_trace = self.get_last_cvoi_trace()
        executed_horizon = runtime_trace.get("stop_horizon")
        if executed_horizon != runtime.forced_horizon:
            raise RuntimeError(
                "manual NavTrain runtime horizon differs from its fixed policy: "
                f"actual={executed_horizon!r}, expected={runtime.forced_horizon}"
            )
        guidance = runtime_trace.get("guidance")
        expected_guidance_steps = 0 if runtime.forced_horizon == 0 else runtime.guidance_steps
        if (
            not isinstance(guidance, Mapping)
            or isinstance(guidance.get("guidance_steps"), bool)
            or not isinstance(guidance.get("guidance_steps"), (int, float))
            or float(guidance["guidance_steps"]) != float(expected_guidance_steps)
        ):
            raise RuntimeError("manual NavTrain runtime Guidance steps differ from its fixed policy")
        captured = self._last_cvoi_navtrain_gate_features
        if not isinstance(captured, Mapping) or set(captured) != {
            "gate_features",
            "observed_feature_sha256",
            "horizon",
        }:
            raise RuntimeError("manual NavTrain policy did not emit its online feature trace")
        if captured["horizon"] != runtime.forced_horizon:
            raise RuntimeError("manual NavTrain feature horizon differs from the executed forced horizon")

        trace = {
            "schema": MANUAL_NAVTRAIN_POLICY_TRACE_SCHEMA,
            "protocol_id": NAVTRAIN_GATE_PROTOCOL_ID,
            "scenario_token": scenario_token,
            "observation_key": observation_key_value,
            "policy_id": runtime.policy_id,
            "lineage": runtime.lineage,
            "horizon": runtime.forced_horizon,
            "gate_features": captured["gate_features"],
            "observed_feature_sha256": captured["observed_feature_sha256"],
        }
        navsim_trajectory = Trajectory(poses=poses, trajectory_sampling=self._cvoi_trajectory_sampling)
        _write_exclusive_trace(
            runtime.trace_output_dir,
            observation_key_value,
            trace,
            expected_directory_identity=runtime.trace_output_dir_identity,
        )
        return navsim_trajectory

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self._forward_mode == "encoder_direct":
            return self._forward_encoder_direct(features)
        if self._forward_mode == "stage3":
            return self._forward_stage3(features)
        if self._forward_mode == "stage2":
            return self._forward_stage2(features)
        return self._forward_stage12(features)

    @torch.no_grad()
    def _forward_encoder_direct(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encoder-direct 前向通路：encoder 观测 token → planner。"""
        device = next(self._encoder.parameters()).device
        dtype = torch.float32
        cfg = self._config

        video_clip = features["video_clip"].to(device, dtype=dtype)
        states = features["states"].to(device, dtype=dtype)
        actions = features["actions"].to(device, dtype=dtype)

        driving_command = features.get("driving_command")
        if driving_command is not None:
            driving_command = driving_command.to(device, dtype=dtype)
        ego_dynamics = features.get("ego_dynamics")
        if ego_dynamics is not None:
            ego_dynamics = ego_dynamics.to(device, dtype=dtype)

        num_observed = cfg.train.num_observed_frames
        observed_context_clips = select_observed_context_clips(video_clip, num_observed)

        # Single-source forward contract: build the SAME ForwardRuntime training does (Phase 1)
        # so encode / status / action-history / planner inputs are identical to the train path.
        runtime = ForwardRuntime.encoder_direct_from_config(cfg, encoder=self._encoder, planner=self._planner)
        z_encoder_obs = runtime.observed_tokens(runtime.encode_context(observed_context_clips))

        status_feature = runtime.status_feature(states, driving_command=driving_command, ego_dynamics=ego_dynamics)
        planner_action_history = None
        if runtime.spec.use_action_history:
            planner_action_history = runtime.action_history(
                actions, num_observed_frames=resolve_encoder_direct_action_history_frames(cfg, self._planner)
            )

        planner_out = runtime.forward_planner(z_encoder_obs, status_feature, planner_action_history)
        if "trajectories" in planner_out:
            trajectory = select_best_trajectory(
                planner_out["trajectories"].float(),
                planner_out["confidences"].float(),
            )
        else:
            trajectory = planner_out["trajectory"].float()

        trajectory = self._normalize_navsim_trajectory_length(trajectory)
        return {"trajectory": trajectory.cpu()}

    def _forward_stage12(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        世界模型前向推理。

        Parameters
        ----------
        features : dict
            来自 VJEPAFeatureBuilder 的特征字典，batch 维度已由 compute_trajectory 添加:
            - "video_clip"       : [B, C, T, H, W]
            - "states"           : [B, T, 7]
            - "actions"          : [B, T-1, action_dim]
            - "extrinsics"       : [B, T, 7]
            - "driving_command"  : [B, T, 4]  (optional)
            - "ego_dynamics"     : [B, T, 4]  (optional)

        Returns
        -------
        dict
            "trajectory" : [B, num_poses, 3]  (x, y, heading)
        """
        device = next(self._encoder.parameters()).device
        dtype = torch.float32
        if cvoi_enabled(self._config):
            self._last_cvoi_trace = None
            self._last_cvoi_planner_output = None
            self._last_cvoi_latency_components = None

        video_clip = features["video_clip"].to(device, dtype=dtype)  # [B, C, T, H, W]
        states = features["states"].to(device, dtype=dtype)  # [B, T, 7]
        actions = features["actions"].to(device, dtype=dtype)  # [B, T-1, action_dim]
        extrinsics = features["extrinsics"].to(device, dtype=dtype)  # [B, T, 7]

        # driving_command/ego_dynamics (可选，用于 IC 函数)
        driving_command = features.get("driving_command")
        if driving_command is not None:
            driving_command = driving_command.to(device, dtype=dtype)  # [B, T, 4]
        ego_dynamics = features.get("ego_dynamics")
        if ego_dynamics is not None:
            ego_dynamics = ego_dynamics.to(device, dtype=dtype)  # [B, T, 4]

        # Per-scene deterministic RNG seed (mirrors _forward_stage2/_forward_stage3). The latent-DiT
        # predictor sample() and a diffusion planner both draw random noise; without this, stage12
        # PDMS open-loop eval is non-deterministic run-to-run and "best checkpoint" selection becomes
        # a lucky draw. Seeded from scene content so it is reproducible and order-independent.
        if cvoi_enabled(self._config):
            eval_rng_seed = features.get("cvoi_rng_seed")
            if isinstance(eval_rng_seed, bool) or not isinstance(eval_rng_seed, int) or eval_rng_seed < 0:
                raise ValueError("CVoI Stage12 requires a non-negative integer cvoi_rng_seed")
        else:
            eval_rng_seed = _make_navsim_eval_rng_seed(
                self._config,
                stage="stage12",
                states=states,
                actions=actions,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
            )

        if video_clip.ndim == 6:
            _, _, _, T, _, _ = video_clip.shape
        else:
            _, _, T, _, _ = video_clip.shape
        tokens_per_frame = self._tokens_per_frame

        # ============ 1. Encoder ============
        camera_metadata = {}
        for key in ("camera_intrinsics", "camera2ego"):
            value = features.get(key)
            if torch.is_tensor(value):
                camera_metadata[key] = value.to(device, dtype=dtype)
        encoder_autocast = (
            cvoi_execution_autocast(self._config, device) if cvoi_enabled(self._config) else nullcontext()
        )
        latency_events = None
        if self._cvoi_latency_mode:
            if not cvoi_enabled(self._config) or str(self._config.cvoi.stage) != "evaluation":
                raise ValueError("CVoI latency mode requires an enabled evaluation config")
            if device.type != "cuda":
                raise RuntimeError("CVoI Formal-v2 latency mode requires CUDA")
            latency_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            latency_events[0].record()
        with encoder_autocast:
            z = forward_main_context(
                self._encoder,
                video_clip,
                config=self._config,
                runtime_normalize_reps=self._runtime_normalize_reps,
                token_ae=self._token_ae,
                multiview_fusion=self._multiview_fusion,
                camera_metadata=camera_metadata,
            )
        if latency_events is not None:
            latency_events[1].record()

        # ============ 1.5 Align side inputs to the main-encoder predictor timeline ============
        # NavSim 只提供观测帧图像；actions/states/extrinsics 需要先补到训练的原始时间长度，
        # 再按 V-JEPA/多视角主 encoder 的时间步长下采样，保持 predictor 的 T 与 z 的 T 一致。
        raw_target_frames = max(int(getattr(self._config.data, "num_target_frames", T)), T)
        actions = _pad_or_trim_temporal_tensor(actions, raw_target_frames - 1)
        states = _pad_or_trim_temporal_tensor(states, raw_target_frames)
        extrinsics = _pad_or_trim_temporal_tensor(extrinsics, raw_target_frames)
        driving_command = _pad_or_trim_temporal_tensor(driving_command, raw_target_frames)
        ego_dynamics = _pad_or_trim_temporal_tensor(ego_dynamics, raw_target_frames)

        dt = 1.0 / float(max(getattr(self._config.data, "fps", 2), 1))

        # ============ 2. Predictor ============
        if use_latent_dit_predictor(self._config):
            if self._rollout_future_steps is not None:
                raise ValueError("fixed rollout evaluation does not support latent-DiT predictors")
            # latent_dit is trained with the autoregressive timeline builder unless
            # use_parallel_predictor is set; eval must use the same builder so the
            # side-condition action mean keeps the same denominator (bug #4 skew).
            _timeline_builder = (
                build_parallel_predictor_timeline_inputs
                if use_parallel_predictor(self._config)
                else build_predictor_timeline_inputs
            )
            predictor_inputs = _timeline_builder(
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
                config=self._config,
                encoder=self._encoder,
                dt=dt,
            )
            tokens_per_frame = predictor_inputs.tokens_per_frame
            if self._budget_controller is not None:
                raise ValueError(
                    "rollout budget controller currently supports only the non-parallel "
                    "ac_transformer autoregressive predictor path"
                )
            with _deterministic_navsim_eval_rng(eval_rng_seed, device):
                z_ar = sample_latent_dit_predictor(
                    predictor=self._predictor,
                    z_context=z,
                    predictor_inputs=predictor_inputs,
                    tokens_per_frame=tokens_per_frame,
                    num_observed_steps=predictor_inputs.num_observed_steps,
                    runtime_normalize_reps=self._runtime_normalize_reps,
                    config=self._config,
                    **resolve_latent_dit_sampler_params(self._config).as_kwargs(),
                )
        elif use_parallel_predictor(self._config):
            if self._rollout_future_steps is not None:
                raise ValueError("fixed rollout evaluation does not support parallel predictors")
            predictor_inputs = build_parallel_predictor_timeline_inputs(
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
                config=self._config,
                encoder=self._encoder,
                dt=dt,
            )
            tokens_per_frame = predictor_inputs.tokens_per_frame
            if self._budget_controller is not None:
                raise ValueError(
                    "rollout budget controller currently supports only the non-parallel "
                    "ac_transformer autoregressive predictor path"
                )
            parallel_output = forward_parallel_predictor(
                predictor=self._predictor,
                observed_tokens=z,
                actions=predictor_inputs.actions,
                states=predictor_inputs.states,
                extrinsics=predictor_inputs.extrinsics,
                config=self._config,
                tokens_per_frame=tokens_per_frame,
                runtime_normalize_reps=self._runtime_normalize_reps,
                num_observed_steps=predictor_inputs.num_observed_steps,
                driving_command=predictor_inputs.driving_command,
                ego_dynamics=predictor_inputs.ego_dynamics,
                predictor_no_aux_input=self._predictor_no_aux_input,
            )
            z_ar = parallel_output.z_future
        else:
            predictor_inputs = build_predictor_timeline_inputs(
                actions=actions,
                states=states,
                extrinsics=extrinsics,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
                config=self._config,
                encoder=self._encoder,
                dt=dt,
            )
            if cvoi_enabled(self._config):
                if self._rollout_future_steps is not None:
                    raise ValueError("fixed rollout evaluation cannot be combined with CVoI")
                predictor_inputs = enforce_cvoi_zero_future_aux(predictor_inputs)
            tokens_per_frame = predictor_inputs.tokens_per_frame
            fixed_rollout_end_step = _resolve_eval_rollout_end_step(
                self._rollout_future_steps,
                num_observed_steps=predictor_inputs.num_observed_steps,
                num_future_steps=predictor_inputs.num_future_steps,
            )
            budget_rollout_end_step = None
            if self._budget_controller is not None:
                if fixed_rollout_end_step is not None:
                    raise ValueError("fixed rollout evaluation cannot be combined with the budget controller")
                z_budget_obs = z[:, : predictor_inputs.num_observed_steps * tokens_per_frame]
                controller_budget, budget_profile = resolve_controller_budget_profile(
                    self._budget_controller,
                    z_budget_obs,
                    config=self._config,
                    deterministic=True,
                    max_future_steps=predictor_inputs.num_future_steps,
                )
                budget_rollout_end_step = predictor_inputs.num_observed_steps + budget_profile.rollout_future_steps
                logger.debug(
                    "[budget_controller][navsim] budget=%.4f profile=%s rollout_end_step=%d",
                    float(controller_budget[0].detach().cpu()),
                    budget_profile,
                    budget_rollout_end_step,
                )
            resolved_rollout_end_step = (
                fixed_rollout_end_step if fixed_rollout_end_step is not None else budget_rollout_end_step
            )
            if not self._should_run_cvoi_sequential_prefix():
                z_ar = self._run_predictor_ar(
                    z,
                    predictor_inputs.actions,
                    predictor_inputs.states,
                    predictor_inputs.extrinsics,
                    tokens_per_frame,
                    driving_command=predictor_inputs.driving_command,
                    ego_dynamics=predictor_inputs.ego_dynamics,
                    num_observed_steps=predictor_inputs.num_observed_steps,
                    num_future_steps=predictor_inputs.num_future_steps,
                    rollout_end_step=resolved_rollout_end_step,
                )
            else:
                if budget_rollout_end_step is not None:
                    raise ValueError("CVoI Gate cannot be combined with the legacy budget controller")
                z_ar = self._run_cvoi_sequential_prefix(z, predictor_inputs, tokens_per_frame)

        # ============ 3. Planner ============
        z_ar_planner = z_ar if self._z_ar_mode == "full" else z_ar[:, :tokens_per_frame]
        if cvoi_enabled(self._config):
            if self._cvoi_gate is None and self._cvoi_evaluation_forced_horizon is None:
                raise RuntimeError("CVoI evaluation requires either a forced horizon or the sequential Gate path")
        elif bool(getattr(getattr(self._config, "value_guidance", None), "enabled", False)):
            if self._value_head is None:
                raise ValueError("value_guidance.enabled=true requires a loaded value_head")
            z_ar_planner, _value_guidance_diag = apply_latent_value_guidance(
                z_ar_planner,
                self._value_head,
                tokens_per_frame=tokens_per_frame,
                config=self._config,
            )
            logger.debug(
                "[value_guidance][navsim] value_before=%.5f value_after=%.5f delta_norm=%.5f steps=%.0f",
                _value_guidance_diag["value_before"],
                _value_guidance_diag["value_after"],
                _value_guidance_diag["delta_norm"],
                _value_guidance_diag["guidance_steps"],
            )
        if latency_events is not None:
            latency_events[2].record()

        # 构建 status_feature
        status_feature = self._build_status_feature(
            states, actions, driving_command=driving_command, ego_dynamics=ego_dynamics
        )

        active_planner = self._planner
        if self._cvoi_direct_runtime is not None:
            if not isinstance(self._last_cvoi_trace, Mapping):
                raise RuntimeError("direct EPDMS sequential runtime did not expose its selected horizon")
            active_planner = self._select_cvoi_direct_planner(self._last_cvoi_trace.get("stop_horizon"))
        if active_planner is None:
            raise RuntimeError("stage12 Planner is not initialized")

        # z_context (可选)
        if self._use_z_context:
            if self._stage12_planner_uses_full_context:
                planner_context_steps = int(getattr(active_planner, "num_context_frames", 1))
                z_first_frame = z[:, : planner_context_steps * tokens_per_frame]
            else:
                z_first_frame = z[:, :tokens_per_frame]
        else:
            z_first_frame = None

        # z_observed (可选)
        if self._use_observed_tokens:
            z_observed = z[:, : predictor_inputs.num_observed_steps * tokens_per_frame]
        else:
            z_observed = None

        planner_action_history = None
        if getattr(self._config.planner, "use_action_history_for_planner", False):
            planner_action_history = build_observed_action_trajectory_history(
                predictor_inputs.actions,
                num_observed_frames=predictor_inputs.num_observed_steps,
                action_history_dim=int(getattr(self._config.planner, "action_history_dim", 3)),
                dt=resolve_action_history_dt(self._config),
            )

        validate_empty_future_planner_conditions(
            z_ar_planner,
            z_context=z_first_frame,
            z_observed=z_observed,
            action_history=planner_action_history,
        )

        planner_kwargs = {
            "z_context": z_first_frame,
            "z_observed": z_observed,
            "action_history": planner_action_history,
        }
        if str(self._config.planner.planner_type) == "diffusion":
            planner_anchor_state = build_ego_relative_diffusion_anchor(
                active_planner,
                ego_dynamics=ego_dynamics,
                observed_frames=self._config.train.num_observed_frames,
                reference=states,
            )
            planner_kwargs["anchor_state"] = planner_anchor_state
            if cvoi_enabled(self._config):
                planner_kwargs["inference_noise"] = cvoi_planner_inference_noise(
                    active_planner,
                    seeds=[eval_rng_seed],
                    device=device,
                )
        planner_autocast = (
            cvoi_execution_autocast(self._config, device) if cvoi_enabled(self._config) else nullcontext()
        )
        planner_rng = (
            common_random_numbers(eval_rng_seed)
            if cvoi_enabled(self._config)
            else _deterministic_navsim_eval_rng(eval_rng_seed, device)
        )
        with planner_rng, planner_autocast:
            planner_output = active_planner(z_ar_planner, status_feature, **planner_kwargs)

        # 提取最高置信度轨迹
        if "trajectories" in planner_output:
            from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output

            # Fail-loud contract check before indexing (clear error instead of a bare KeyError /
            # silent shape drift) — same inference contract the training lines already enforce.
            strict_num_poses = (
                self._cvoi_trajectory_sampling.num_poses
                if (self._cvoi_direct_runtime is not None or self._rollout_future_steps is not None)
                else None
            )
            validate_planner_output(
                planner_output,
                mode="inference",
                num_poses=strict_num_poses,
            )
            pred_trajs = planner_output["trajectories"]  # [B, K, num_poses, 3]
            pred_conf = planner_output["confidences"]  # [B, K]
            if cvoi_enabled(self._config):
                self._store_cvoi_planner_output(pred_trajs, pred_conf)
            if value_planning_method1_enabled(self._config):
                value_result = score_trajectories_method1(
                    predictor=self._predictor,
                    value_head=self._value_head,
                    z_context=z,
                    trajs=pred_trajs.float(),
                    actions=actions,
                    states=states,
                    driving_command=driving_command,
                    ego_dynamics=ego_dynamics,
                    config=self._config,
                    tokens_per_frame=tokens_per_frame,
                    runtime_normalize_reps=bool(self._runtime_normalize_reps),
                    dt=float(dt),
                    predictor_observed_steps=predictor_inputs.num_observed_steps,
                    predictor_frame_stride=predictor_inputs.frame_stride,
                    confidences=pred_conf.float(),
                )
                trajectory = value_result["value_selected_trajectory"]
            else:
                best_idx = pred_conf.argmax(dim=1)  # [B]
                best_idx_exp = best_idx.view(-1, 1, 1, 1).expand(-1, 1, pred_trajs.shape[2], pred_trajs.shape[3])
                trajectory = pred_trajs.gather(1, best_idx_exp).squeeze(1)  # [B, num_poses, 3]
        else:
            if self._cvoi_latency_mode:
                raise ValueError("CVoI Formal-v2 latency requires planner trajectories and confidences")
            trajectory = planner_output["trajectory"]  # [B, num_poses, 3]

        if self._rollout_future_steps is not None:
            trajectory = self._validate_fixed_rollout_trajectory(
                trajectory,
                num_poses=self._cvoi_trajectory_sampling.num_poses,
            )
        elif self._cvoi_direct_runtime is not None:
            expected_num_poses = self._cvoi_trajectory_sampling.num_poses
            if trajectory.ndim != 3 or trajectory.shape[1] != expected_num_poses:
                raise ValueError(
                    "strict CVoI planner trajectory pose count must equal trajectory_sampling.num_poses: "
                    f"expected {expected_num_poses}, got shape {tuple(trajectory.shape)}"
                )
        else:
            # Ordinary NavSim evaluation retains its historical pad/trim behavior.
            trajectory = self._normalize_navsim_trajectory_length(trajectory)

        if latency_events is not None:
            latency_events[3].record()
            latency_events[3].synchronize()
            self._last_cvoi_latency_components = {
                "encoder": float(latency_events[0].elapsed_time(latency_events[1])),
                "adaptive_rollout_value_gate_guidance": float(latency_events[1].elapsed_time(latency_events[2])),
                "planner_and_output": float(latency_events[2].elapsed_time(latency_events[3])),
            }

        return {"trajectory": trajectory if self._cvoi_latency_mode else trajectory.cpu()}

    @torch.no_grad()
    def _forward_stage2(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage-2 前向通路：frozen proposal → conditional predictor rollout → refinement。

        对齐 val_lewm_staged.py 的 stage2 验证逻辑。
        """
        from app.vjepa_cowa_world_model.val_lewm_staged import run_stage2_refinement_for_validation

        device = next(self._encoder.parameters()).device
        dtype = torch.float32
        cfg = self._config

        video_clip = features["video_clip"].to(device, dtype=dtype)
        proposal_video_clip = features.get("proposal_video_clip")
        if proposal_video_clip is not None:
            proposal_video_clip = proposal_video_clip.to(device, dtype=dtype)
        states = features["states"].to(device, dtype=dtype)
        actions = features["actions"].to(device, dtype=dtype)

        driving_command = features.get("driving_command")
        if driving_command is not None:
            driving_command = driving_command.to(device, dtype=dtype)
        ego_dynamics = features.get("ego_dynamics")
        if ego_dynamics is not None:
            ego_dynamics = ego_dynamics.to(device, dtype=dtype)

        dt = 1.0 / float(max(getattr(cfg.data, "fps", 2), 1))

        observed_context_clips = select_observed_context_clips(video_clip, cfg.train.num_observed_frames)
        if self._requires_proposal_video_clip():
            if proposal_video_clip is None:
                raise ValueError(
                    "Stage-2 separate proposal encoder requires 'proposal_video_clip' from VJEPAFeatureBuilder"
                )
            proposal_context_clips = select_observed_context_clips(
                proposal_video_clip,
                cfg.train.num_observed_frames,
            )
        else:
            proposal_context_clips = observed_context_clips

        use_shared_proposal_context = (
            self._proposal_encoder is None and cfg.proposal.provider_type != "history_kinematic"
        )
        if use_shared_proposal_context:
            z_context, z_context_proposal = forward_main_context_dual(
                self._encoder,
                observed_context_clips,
                config=cfg,
                predictor_normalize_reps=self._runtime_normalize_reps,
                proposal_normalize_reps=self._proposal_runtime_normalize_reps,
                predictor_token_ae=self._token_ae,
                proposal_token_ae=self._proposal_token_ae,
            )
        else:
            z_context = forward_main_context(
                self._encoder,
                observed_context_clips,
                config=cfg,
                runtime_normalize_reps=self._runtime_normalize_reps,
                token_ae=self._token_ae,
            )
            z_context_proposal = None

        status_feature = build_status_feature(cfg, states, actions, driving_command, ego_dynamics)
        history_traj = build_proposal_history(cfg, actions, dt)

        eval_rng_seed = _make_navsim_eval_rng_seed(
            cfg,
            stage="stage2",
            states=states,
            actions=actions,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
        )
        with _deterministic_navsim_eval_rng(eval_rng_seed, device):
            if use_shared_proposal_context:
                proposal_out = self._proposal_planner(
                    z_context=z_context_proposal,
                    status_feature=status_feature,
                    history_traj=history_traj,
                )
            else:
                proposal_out = forward_frozen_proposal(
                    proposal_encoder=self._proposal_encoder,
                    proposal_planner=self._proposal_planner,
                    context_clips=proposal_context_clips,
                    use_tubelet_repeat=cfg.data.use_tubelet_repeat,
                    proposal_normalize_reps=self._proposal_runtime_normalize_reps,
                    proposal_token_ae=self._proposal_token_ae,
                    status_feature=status_feature,
                    history_traj=history_traj,
                    num_observed_frames=cfg.train.num_observed_frames,
                )

            proposal_out = maybe_expand_manual_proposal(cfg, proposal_out)
            from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output

            # Fail-loud proposal contract before indexing (verified: trajectories [B, K, num_poses, 3] +
            # confidences [B, K]; maybe_expand_manual_proposal preserves it).
            validate_planner_output(proposal_out, mode="inference")
            proposal_trajs = proposal_out["trajectories"].float()
            proposal_conf = proposal_out["confidences"].float()

            proposal_features = proposal_out.get("proposal_features")
            planner_module = self._planner.module if hasattr(self._planner, "module") else self._planner
            if proposal_features is None or proposal_features.shape[-1] != planner_module.tf_d_model:
                proposal_features = planner_module.encode_proposal_features(proposal_trajs)

            _predictor_rollout_fn = build_stage_predictor_rollout_fn(
                stage="stage2",
                predictor=self._predictor,
                z_context=z_context,
                actions=actions,
                states=states,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
                config=cfg,
                tokens_per_frame=self._tokens_per_frame,
                runtime_normalize_reps=self._runtime_normalize_reps,
                dt=dt,
                predictor_observed_steps=self._main_context_timeline.num_observed_steps,
                predictor_frame_stride=self._main_context_timeline.frame_stride,
            )

            trajectory = run_stage2_refinement_for_validation(
                planner=self._planner,
                config=cfg,
                z_context=z_context,
                status_feature=status_feature,
                proposal_trajs=proposal_trajs,
                proposal_conf=proposal_conf,
                proposal_features=proposal_features,
                predictor_rollout_fn=_predictor_rollout_fn,
                call_planner_method_fn=call_planner_method,
            ).float()

        trajectory = self._normalize_navsim_trajectory_length(trajectory)
        return {"trajectory": trajectory.cpu()}

    @torch.no_grad()
    def _forward_stage3(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Stage-3 前向通路：frozen proposal → conditional predictor rollout → iterative refinement。

        对齐 val_lewm_staged.py:1026-1191 的 stage3 验证逻辑。
        """
        from app.vjepa_cowa_world_model.val_lewm_staged import select_stage3_refined_predictions

        device = next(self._encoder.parameters()).device
        dtype = torch.float32
        cfg = self._config

        video_clip = features["video_clip"].to(device, dtype=dtype)  # [B, C, T, H, W]
        proposal_video_clip = features.get("proposal_video_clip")
        if proposal_video_clip is not None:
            proposal_video_clip = proposal_video_clip.to(device, dtype=dtype)
        states = features["states"].to(device, dtype=dtype)  # [B, T, 7]
        actions = features["actions"].to(device, dtype=dtype)  # [B, T-1, action_dim]

        driving_command = features.get("driving_command")
        if driving_command is not None:
            driving_command = driving_command.to(device, dtype=dtype)
        ego_dynamics = features.get("ego_dynamics")
        if ego_dynamics is not None:
            ego_dynamics = ego_dynamics.to(device, dtype=dtype)

        dt = 1.0 / float(max(getattr(cfg.data, "fps", 2), 1))

        # 1. Select observed context clips
        observed_context_clips = select_observed_context_clips(video_clip, cfg.train.num_observed_frames)
        if self._requires_proposal_video_clip():
            if proposal_video_clip is None:
                raise ValueError(
                    "Stage-3 separate proposal encoder requires 'proposal_video_clip' from VJEPAFeatureBuilder"
                )
            proposal_context_clips = select_observed_context_clips(
                proposal_video_clip,
                cfg.train.num_observed_frames,
            )
        else:
            proposal_context_clips = observed_context_clips

        # 2. Forward main context (encoder → z_context)
        use_shared_proposal_context = (
            self._proposal_encoder is None and cfg.proposal.provider_type != "history_kinematic"
        )

        if use_shared_proposal_context:
            z_context, z_context_proposal = forward_main_context_dual(
                self._encoder,
                observed_context_clips,
                config=cfg,
                predictor_normalize_reps=self._runtime_normalize_reps,
                proposal_normalize_reps=self._proposal_runtime_normalize_reps,
                predictor_token_ae=self._token_ae,
                proposal_token_ae=self._proposal_token_ae,
            )
        else:
            z_context = forward_main_context(
                self._encoder,
                observed_context_clips,
                config=cfg,
                runtime_normalize_reps=self._runtime_normalize_reps,
                token_ae=self._token_ae,
            )
            z_context_proposal = None

        # 3. Build status feature and proposal history
        status_feature = build_status_feature(cfg, states, actions, driving_command, ego_dynamics)
        history_traj = build_proposal_history(cfg, actions, dt)

        eval_rng_seed = _make_navsim_eval_rng_seed(
            cfg,
            stage="stage3",
            states=states,
            actions=actions,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
        )
        with _deterministic_navsim_eval_rng(eval_rng_seed, device):
            # 4. Forward frozen proposal
            if use_shared_proposal_context:
                proposal_out = self._proposal_planner(
                    z_context=z_context_proposal,
                    status_feature=status_feature,
                    history_traj=history_traj,
                )
            else:
                proposal_out = forward_frozen_proposal(
                    proposal_encoder=self._proposal_encoder,
                    proposal_planner=self._proposal_planner,
                    context_clips=proposal_context_clips,
                    use_tubelet_repeat=cfg.data.use_tubelet_repeat,
                    proposal_normalize_reps=self._proposal_runtime_normalize_reps,
                    proposal_token_ae=self._proposal_token_ae,
                    status_feature=status_feature,
                    history_traj=history_traj,
                    num_observed_frames=cfg.train.num_observed_frames,
                )

            # 5. Maybe expand manual proposal
            proposal_out = maybe_expand_manual_proposal(cfg, proposal_out)
            from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output

            # Fail-loud proposal contract before indexing (verified: trajectories [B, K, num_poses, 3] +
            # confidences [B, K]; maybe_expand_manual_proposal preserves it).
            validate_planner_output(proposal_out, mode="inference")
            proposal_trajs = proposal_out["trajectories"].float()
            proposal_conf = proposal_out["confidences"].float()

            # 6. Encode proposal features
            proposal_features = proposal_out.get("proposal_features")
            planner_module = self._planner.module if hasattr(self._planner, "module") else self._planner
            if not getattr(cfg.refinement_gated, "refine_use_proposal_features", True):
                proposal_features = None
            elif proposal_features is None or proposal_features.shape[-1] != planner_module.tf_d_model:
                proposal_features = planner_module.encode_proposal_features(proposal_trajs)

            # 7. Define predictor rollout closure
            _predictor_rollout_fn = build_stage_predictor_rollout_fn(
                stage="stage3",
                predictor=self._predictor,
                z_context=z_context,
                actions=actions,
                states=states,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
                config=cfg,
                tokens_per_frame=self._tokens_per_frame,
                runtime_normalize_reps=self._runtime_normalize_reps,
                dt=dt,
                predictor_observed_steps=self._main_context_timeline.num_observed_steps,
                predictor_frame_stride=self._main_context_timeline.frame_stride,
            )

            # 8. Apply stage-3 refinement input gates
            refinement_inputs = apply_stage3_refinement_input_gates(
                config=cfg,
                z_context=z_context,
                status_feature=status_feature,
                proposal_traj=proposal_trajs,
                proposal_logits=proposal_conf,
                proposal_features=proposal_features,
                predictor_rollout_fn=_predictor_rollout_fn,
            )

            # 9. Call forward_iterative
            _traj_rounds, refined_display = call_planner_method(
                self._planner,
                "forward_iterative",
                refinement_inputs.z_context,
                refinement_inputs.status_feature,
                proposal_traj=refinement_inputs.proposal_traj,
                proposal_logits=refinement_inputs.proposal_logits,
                proposal_features=refinement_inputs.proposal_features,
                predictor_rollout_fn=refinement_inputs.predictor_rollout_fn,
                num_rounds=cfg.refinement_gated.num_rounds,
                grad_checkpoint=False,
                detach_future=True,
                use_initial_proposal_features=refinement_inputs.use_initial_proposal_features,
                return_single_final=not getattr(cfg.refinement_gated, "use_multimodal_final", False),
            )

            # 10. Select best trajectory
            refined_display, _refined_pred_trajs = select_stage3_refined_predictions(
                cfg,
                traj_rounds=_traj_rounds,
                traj_final=refined_display,
            )
            trajectory = refined_display.float()

        # 对齐 NAVSIM 的 8 个 pose 采样长度
        trajectory = self._normalize_navsim_trajectory_length(trajectory)

        return {"trajectory": trajectory.cpu()}

    def _parse_inference_params(self) -> None:
        """从 TrainingConfig 中解析推理参数。"""
        cfg = self._config

        self._tokens_per_frame = cfg.data.tokens_per_frame
        self._normalize_reps = cfg.loss.normalize_reps
        self._action_dim = cfg.train.action_dim

        # Planner 参数
        self._status_mode = cfg.planner.states_mode
        self._z_ar_mode = getattr(cfg.planner, "z_ar_mode", "full")
        self._use_z_context = cfg.planner.use_z_context
        self._use_observed_tokens = getattr(cfg.planner, "use_observed_tokens", False)
        self._use_states_for_planner = cfg.planner.use_states_for_planner

        # Predictor 参数
        self._predictor_inference_consistent = cfg.train.predictor_inference_consistent
        self._predictor_no_aux_input = cfg.train.predictor_no_aux_input
        self._use_states_for_predictor = cfg.train.use_states_for_predictor
        self._num_observed_frames = cfg.train.num_observed_frames

        predictor_free_mode = self._forward_mode in {"stage2", "stage3", "encoder_direct"}

        # Staged refinement / encoder-direct 不走 _run_predictor_ar，跳过互斥检查
        if not predictor_free_mode:
            if self._predictor_no_aux_input:
                self._use_states_for_predictor = False
            if self._predictor_no_aux_input and self._predictor_inference_consistent:
                logger.warning(
                    "predictor_no_aux_input=True and predictor_inference_consistent=True "
                    "are mutually exclusive; disabling predictor_inference_consistent"
                )
                self._predictor_inference_consistent = False

        # num_poses: 世界模型可能预测更多 pose，但我们只取 NAVSIM 的 8 个
        self._num_poses = NAVSIM_NUM_POSES

        # Staged refinement / encoder-direct 不使用 _run_predictor_ar，_num_future_to_predict 不需要
        if predictor_free_mode:
            self._num_future_to_predict = 0
        elif self._predictor_inference_consistent:
            self._num_future_to_predict = cfg.data.num_target_frames - self._num_observed_frames
        else:
            self._num_future_to_predict = cfg.data.num_target_frames - 1

        logger.info(
            f"Inference params: tokens_per_frame={self._tokens_per_frame}, "
            f"num_poses={self._num_poses}, num_future_to_predict={self._num_future_to_predict}, "
            f"normalize_reps={self._normalize_reps}, "
            f"status_mode={self._status_mode}, z_ar_mode={self._z_ar_mode}, "
            f"predictor_inference_consistent={self._predictor_inference_consistent}, "
            f"num_observed_frames={self._num_observed_frames}, action_dim={self._action_dim}, "
            f"forward_mode={self._forward_mode}"
        )

    def get_last_cvoi_trace(self) -> Dict[str, Any]:
        """Return the most recent online-only Gate trace for compute reporting."""

        if self._last_cvoi_trace is None:
            raise RuntimeError("no CVoI inference trace is available before the first trajectory computation")
        return {
            "stop_horizon": int(self._last_cvoi_trace["stop_horizon"]),
            "decisions": list(self._last_cvoi_trace["decisions"]),
            "predicted_deltas": list(self._last_cvoi_trace["predicted_deltas"]),
            "rollout_latency_ms": float(self._last_cvoi_trace["rollout_latency_ms"]),
            "guidance": dict(self._last_cvoi_trace["guidance"]),
        }

    def set_cvoi_evaluation_guidance_steps(self, steps: Optional[int]) -> None:
        """Set an evaluation-only K override without changing the trained K=2 contract."""

        if self._config is None or str(self._config.cvoi.stage) != "evaluation":
            raise ValueError("CVoI guidance-step override is permitted only after evaluation config initialization")
        allowed_steps = cvoi_evaluation_guidance_steps(self._config, include_disabled=True)
        if steps is not None and (type(steps) is not int or steps not in allowed_steps):
            raise ValueError(f"CVoI evaluation guidance steps must be one of {allowed_steps} or None")
        self._cvoi_evaluation_guidance_steps = steps

    def set_cvoi_evaluation_forced_horizon(self, horizon: Optional[int]) -> None:
        """Set a known evaluation stop horizon while bypassing Value and Gate calls."""

        if self._config is None or str(self._config.cvoi.stage) != "evaluation":
            raise ValueError("CVoI forced horizon is permitted only after evaluation config initialization")
        max_horizon = int(self._config.cvoi.max_horizon)
        if horizon is not None and (type(horizon) is not int or horizon not in range(max_horizon + 1)):
            raise ValueError(f"CVoI evaluation forced horizon must be one of {set(range(max_horizon + 1))} or None")
        self._cvoi_evaluation_forced_horizon = horizon

    def set_cvoi_latency_mode(self, enabled: bool) -> None:
        """Keep Formal-v2 timing outputs on device and expose CUDA-event phase timings."""

        if self._config is None or str(self._config.cvoi.stage) != "evaluation":
            raise ValueError("CVoI latency mode is permitted only after evaluation config initialization")
        if type(enabled) is not bool:
            raise TypeError("CVoI latency mode enabled must be a bool")
        self._cvoi_latency_mode = enabled
        self._last_cvoi_latency_components = None

    def set_cvoi_evaluation_gate(self, gate: nn.Module, *, feature_mode: str) -> None:
        """Switch a strictly loaded NavSim-e120 Gate and its matching online feature mask."""

        if self._config is None or str(self._config.cvoi.stage) != "evaluation":
            raise ValueError("CVoI Gate override is permitted only after evaluation config initialization")
        if getattr(self._config.cvoi, "protocol_version", None) != "formal_v2_navsim_e120_h4_v3":
            raise ValueError("CVoI Gate override is available only for NavSim-e120 evaluation")
        if not isinstance(gate, nn.Module):
            raise TypeError("CVoI Gate override requires a torch.nn.Module")
        allowed_feature_modes = {"full", "without_field", "without_stop", "without_value_summary"}
        if type(feature_mode) is not str or feature_mode not in allowed_feature_modes:
            raise ValueError("CVoI Gate feature_mode is incompatible with the NavSim-e120 protocol")
        if any(parameter.requires_grad for parameter in gate.parameters()) or gate.training:
            raise ValueError("CVoI evaluation Gate override must be frozen and in eval mode")
        self._cvoi_gate = gate
        self._cvoi_evaluation_gate_feature_mode = feature_mode

    def _store_cvoi_planner_output(self, pred_trajs: torch.Tensor, confidences: torch.Tensor) -> None:
        """Store immutable planner outputs, preserving CUDA residency only for Formal-v2 latency."""

        if not isinstance(pred_trajs, torch.Tensor) or not isinstance(confidences, torch.Tensor):
            raise TypeError("CVoI planner outputs must be tensors")
        if self._cvoi_latency_mode:
            stored_trajs = pred_trajs.detach().clone()
            stored_confidences = confidences.detach().clone()
        else:
            stored_trajs = pred_trajs.detach().cpu().clone()
            stored_confidences = confidences.detach().cpu().clone()
        self._last_cvoi_planner_output = {
            "pred_trajs": stored_trajs,
            "confidences": stored_confidences,
        }

    def prepare_cvoi_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Move observed-only evaluation features to the model device before timing."""

        if self._config is None or str(self._config.cvoi.stage) != "evaluation":
            raise ValueError("CVoI feature preparation is permitted only after evaluation config initialization")
        if not isinstance(features, dict) or not features:
            raise ValueError("CVoI feature preparation requires a non-empty feature dictionary")
        if self._encoder is None:
            raise RuntimeError("CVoI feature preparation requires an initialized encoder")
        device = next(self._encoder.parameters()).device
        prepared: Dict[str, Any] = {}
        for name, value in features.items():
            if name == "cvoi_rng_seed":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError("cvoi_rng_seed must be an integer")
                prepared[name] = value
            elif isinstance(value, torch.Tensor):
                prepared[name] = value.to(device=device, dtype=torch.float32)
            else:
                raise TypeError(f"unsupported CVoI feature {name!r}: expected a tensor")
        return prepared

    def get_last_cvoi_planner_output(self) -> Dict[str, torch.Tensor]:
        """Return cloned raw planner candidates and deployment confidences."""

        if self._last_cvoi_planner_output is None:
            raise RuntimeError("no CVoI planner output is available before the first trajectory computation")
        return {
            "pred_trajs": self._last_cvoi_planner_output["pred_trajs"].clone(),
            "confidences": self._last_cvoi_planner_output["confidences"].clone(),
        }

    def get_last_cvoi_latency_components(self) -> Dict[str, float]:
        """Return CUDA-event timings for the exact most recent Formal-v2 forward boundary."""

        if self._last_cvoi_latency_components is None:
            raise RuntimeError("no CVoI latency trace is available before the first latency-mode forward")
        expected = {
            "encoder",
            "adaptive_rollout_value_gate_guidance",
            "planner_and_output",
        }
        if set(self._last_cvoi_latency_components) != expected:
            raise RuntimeError("CVoI latency trace contains an invalid component schema")
        return {name: float(value) for name, value in self._last_cvoi_latency_components.items()}

    def _should_run_cvoi_sequential_prefix(self) -> bool:
        """Select the sequential path before any full autoregressive rollout can execute."""

        if self._cvoi_evaluation_forced_horizon is not None or self._cvoi_gate is not None:
            return True
        if cvoi_enabled(self._config):
            raise RuntimeError("CVoI evaluation requires either a forced horizon or the sequential Gate path")
        return False

    def _capture_navtrain_gate_teacher_features(
        self,
        observed: torch.Tensor,
        raw_prefix: torch.Tensor,
        *,
        horizon: int,
        tokens_per_frame: int,
    ) -> None:
        """Capture online-only, lambda-independent Gate features for one forced H."""

        manual_runtime = getattr(self, "_cvoi_manual_runtime", None)
        if manual_runtime is None:
            return
        planner_stage = manual_runtime.planner_stage
        if self._cvoi_navtrain_stop_value_adapter is None:
            raise RuntimeError("manual NavTrain feature capture requires the direct Stop value model")
        values = self._cvoi_navtrain_stop_value_adapter(
            observed,
            raw_prefix,
            tokens_per_frame=tokens_per_frame,
        )
        stop_values = getattr(values, "stop_values", None)
        field_values = getattr(values, "field_values", None)
        if not isinstance(stop_values, torch.Tensor) or stop_values.shape != (1, horizon + 1):
            raise ValueError("navtrain Gate stop_values must have shape [1,h+1]")
        if not isinstance(field_values, torch.Tensor) or field_values.shape != (1, horizon):
            raise ValueError("navtrain Gate field_values must have shape [1,h]")
        if not bool(torch.isfinite(stop_values).all().item()) or not bool(torch.isfinite(field_values).all().item()):
            raise ValueError("navtrain Gate Value outputs must be finite")
        stop_value = stop_values[:, -1].to(torch.float32)
        previous_stop_value = stop_value if horizon == 0 else stop_values[:, -2].to(torch.float32)
        if planner_stage == "p0" or horizon == 0:
            field_value = torch.zeros_like(stop_value)
        else:
            field_value = field_values[:, -1].to(torch.float32)
        pooled_observed = observed.float().mean(dim=1)
        pooled_prefix = torch.zeros_like(pooled_observed) if horizon == 0 else raw_prefix.float().mean(dim=1)
        costs = tuple(float(value) for value in self._config.cvoi.compute_costs)
        max_horizon = int(self._config.cvoi.max_horizon)
        if len(costs) != max_horizon + 1 or horizon not in range(max_horizon + 1):
            raise ValueError("navtrain Gate compute-cost schedule does not cover H0--H4")
        current_cost = stop_value.new_full((1,), costs[horizon])
        next_cost = stop_value.new_full((1,), costs[min(horizon + 1, max_horizon)])
        features = build_lambda_independent_sequential_gate_features(
            pooled_observed=pooled_observed,
            pooled_prefix=pooled_prefix,
            field_value=field_value,
            stop_value=stop_value,
            previous_stop_value=previous_stop_value,
            horizon=torch.tensor([horizon], dtype=torch.long, device=observed.device),
            max_horizon=max_horizon,
            current_cost=current_cost,
            next_cost=next_cost,
        )
        if features.ndim != 2 or features.shape[0] != 1 or not bool(torch.isfinite(features).all().item()):
            raise ValueError("navtrain Gate feature vector must be one finite row")
        observed_bytes = pooled_observed.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
        self._last_cvoi_navtrain_gate_features = {
            "gate_features": [float(value) for value in features[0].detach().cpu().tolist()],
            "observed_feature_sha256": hashlib.sha256(observed_bytes).hexdigest(),
            "horizon": horizon,
        }

    def _run_cvoi_sequential_prefix(
        self, z: torch.Tensor, predictor_inputs: Any, tokens_per_frame: int
    ) -> torch.Tensor:
        """Incrementally roll raw predictor tokens until the CVoI Gate stops."""

        if z.ndim != 3 or z.shape[0] != 1:
            raise ValueError(f"CVoI NavSim runtime requires batch size 1 encoder tokens, got {tuple(z.shape)}")
        num_observed = int(predictor_inputs.num_observed_steps)
        num_future = int(predictor_inputs.num_future_steps)
        max_horizon = int(self._config.cvoi.max_horizon)
        if num_future < max_horizon:
            raise ValueError(f"CVoI runtime requires at least {max_horizon} future steps, got {num_future}")
        observed = z[:, : num_observed * tokens_per_frame].detach()
        if observed.shape[1] != num_observed * tokens_per_frame:
            raise ValueError("encoder tokens do not cover the configured observed prefix")
        forced_horizon = self._cvoi_evaluation_forced_horizon
        if forced_horizon == 0:
            empty_prefix = observed.new_empty(observed.shape[0], 0, observed.shape[2])
            self._capture_navtrain_gate_teacher_features(
                observed,
                empty_prefix,
                horizon=0,
                tokens_per_frame=tokens_per_frame,
            )
            guidance_diagnostics = {
                "guidance_steps": 0.0,
                "guidance_skipped_h0": 1.0,
                "delta_norm": 0.0,
                "field_value_before": 0.0,
                "field_value_after": 0.0,
            }
            self._last_cvoi_trace = {
                "stop_horizon": 0,
                "decisions": [],
                "predicted_deltas": [],
                "rollout_latency_ms": 0.0,
                "guidance": guidance_diagnostics,
            }
            return empty_prefix

        direct_runtime = getattr(self, "_cvoi_direct_runtime", None)
        stop_value_adapter = (
            getattr(self, "_cvoi_direct_stop_value_adapter", None)
            if direct_runtime is not None
            else self._cvoi_dual_value_adapter
        )
        field_value_adapter = (
            getattr(self, "_cvoi_direct_field_value_adapter", None)
            if direct_runtime is not None
            else self._cvoi_dual_value_adapter
        )
        if forced_horizon is None and (self._cvoi_gate is None or stop_value_adapter is None):
            raise RuntimeError("CVoI online sequential runtime requires both a Gate and a stop-Value model")
        num_total = num_observed + num_future
        step_predictor = make_predictor_step_fn(
            self._predictor,
            self._config,
            num_observed,
            driving_command=predictor_inputs.driving_command,
            ego_dynamics=predictor_inputs.ego_dynamics,
            normalize_reps=self._runtime_normalize_reps,
        )

        def rollout_step(raw_prefix: torch.Tensor, next_horizon: int) -> torch.Tensor:
            rolled = torch.cat([observed, raw_prefix], dim=1)
            timeline_step = num_observed + int(next_horizon) - 1
            if timeline_step == num_total - 1:
                actions_step = predictor_inputs.actions
                states_step = predictor_inputs.states[:, :-1]
                extrinsics_step = predictor_inputs.extrinsics[:, :-1]
            else:
                actions_step = predictor_inputs.actions[:, :timeline_step]
                states_step = predictor_inputs.states[:, :timeline_step]
                extrinsics_step = predictor_inputs.extrinsics[:, :timeline_step]
            with cvoi_execution_autocast(self._config, rolled.device):
                return step_predictor(rolled, actions_step, states_step, extrinsics_step)[:, -tokens_per_frame:]

        def value_features(
            observed_latent: torch.Tensor,
            raw_prefix: torch.Tensor,
            horizon: int,
        ) -> Dict[str, torch.Tensor]:
            del horizon
            if stop_value_adapter is None:
                raise RuntimeError("CVoI online sequential runtime requires a stop-Value model")
            output = stop_value_adapter(
                observed_latent,
                raw_prefix,
                tokens_per_frame=tokens_per_frame,
            )
            values = extract_prefix_gate_values(output)
            if getattr(self._config.cvoi, "controller_lineage", "value_guided") == "p0_controller":
                values["field_value"] = torch.zeros_like(values["stop_value"])
            return {"field_value": values["field_value"], "stop_value": values["stop_value"]}

        guidance_diagnostics: Dict[str, float] = {}

        def stop_and_prepare(raw_prefix: torch.Tensor, horizon: int, apply_guidance: bool) -> torch.Tensor:
            if apply_guidance != (horizon > 0):
                raise RuntimeError("CVoI sequential runtime produced an inconsistent Guidance decision")
            guidance_disabled = (
                horizon == 0
                or self._cvoi_evaluation_guidance_steps == 0
                or getattr(self._config.cvoi, "controller_lineage", "value_guided") == "p0_controller"
            )
            if guidance_disabled:
                guidance_diagnostics.update(
                    {
                        "guidance_steps": 0.0,
                        "guidance_skipped_h0": float(horizon == 0),
                        "delta_norm": 0.0,
                        "field_value_before": 0.0,
                        "field_value_after": 0.0,
                    }
                )
                return raw_prefix.detach()
            if field_value_adapter is None:
                raise RuntimeError("CVoI P1 forced Guidance requires a loaded field-Value model")
            guided, diagnostics = apply_cvoi_planner_guidance(
                observed,
                raw_prefix,
                field_value_adapter,
                tokens_per_frame=tokens_per_frame,
                config=self._config,
                evaluation_guidance_steps=self._cvoi_evaluation_guidance_steps,
            )
            guidance_diagnostics.update(diagnostics)
            return guided

        if forced_horizon is not None:
            raw_prefix = observed.new_empty(observed.shape[0], 0, observed.shape[2])
            if z.is_cuda:
                torch.cuda.synchronize(z.device)
            start_time = time.perf_counter()
            for next_horizon in range(1, forced_horizon + 1):
                next_tokens = rollout_step(raw_prefix, next_horizon)
                raw_prefix = torch.cat([raw_prefix, next_tokens], dim=1)
            self._capture_navtrain_gate_teacher_features(
                observed,
                raw_prefix,
                horizon=forced_horizon,
                tokens_per_frame=tokens_per_frame,
            )
            planner_input = stop_and_prepare(raw_prefix, forced_horizon, forced_horizon > 0)
            if z.is_cuda:
                torch.cuda.synchronize(z.device)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_cvoi_trace = {
                "stop_horizon": forced_horizon,
                "decisions": [],
                "predicted_deltas": [],
                "rollout_latency_ms": latency_ms,
                "guidance": dict(guidance_diagnostics),
            }
            return planner_input

        if z.is_cuda:
            torch.cuda.synchronize(z.device)
        start_time = time.perf_counter()
        lambda_compute = float(self._config.cvoi.lambda_compute)
        protocol_version = str(getattr(self._config.cvoi, "protocol_version", "legacy_v1"))
        if protocol_version != "formal_v2_navsim_e120_h4_v3":
            raise ValueError("NavSim CVoI sequential rollout requires protocol_version='formal_v2_navsim_e120_h4_v3'")
        result = run_sequential_rollout(
            observed_latent=observed,
            gate=self._cvoi_gate,
            max_horizon=max_horizon,
            lambda_compute=lambda_compute,
            compute_costs=list(self._config.cvoi.compute_costs),
            rollout_step=rollout_step,
            value_features=value_features,
            stop_and_plan=stop_and_prepare,
            gate_feature_mode=(
                getattr(self, "_cvoi_evaluation_gate_feature_mode", None)
                if getattr(self, "_cvoi_evaluation_gate_feature_mode", None) is not None
                else (
                    "full"
                    if getattr(self._config.cvoi, "ablation_signature", None) is None
                    else str(self._config.cvoi.ablation_signature.gate_feature_mode)
                )
            ),
            gate_feature_protocol="formal_v2_navsim_e120_h4_v3",
        )
        if z.is_cuda:
            torch.cuda.synchronize(z.device)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        result.require_finite_rollout_tokens()
        self._last_cvoi_trace = {
            "stop_horizon": result.stop_horizon,
            "decisions": list(result.decisions),
            "predicted_deltas": list(result.predicted_deltas),
            "rollout_latency_ms": latency_ms,
            "guidance": dict(guidance_diagnostics),
        }
        if not torch.is_tensor(result.planner_output):
            raise TypeError("CVoI stop callback must return the terminal guided latent tensor")
        return result.planner_output

    def _run_predictor_ar(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        states: torch.Tensor,
        extrinsics: torch.Tensor,
        tokens_per_frame: int,
        driving_command: Optional[torch.Tensor] = None,
        ego_dynamics: Optional[torch.Tensor] = None,
        num_observed_steps: Optional[int] = None,
        num_future_steps: Optional[int] = None,
        rollout_end_step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive predictor rollout，与 val_command.py 中的逻辑完全对齐。

        Parameters
        ----------
        z : [B, T*P, D]
        actions : [B, T-1, action_dim]
        states : [B, T, 7]
        extrinsics : [B, T, 7]
        tokens_per_frame : int

        Returns
        -------
        z_ar : [B, pred_frames*P, D]
        """
        predictor_observed_steps = (
            int(num_observed_steps) if num_observed_steps is not None else self._num_observed_frames
        )
        predictor_future_steps = int(num_future_steps) if num_future_steps is not None else self._num_future_to_predict
        num_obs = predictor_observed_steps

        _step_predictor = make_predictor_step_fn(
            self._predictor,
            self._config,
            num_obs,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            normalize_reps=self._runtime_normalize_reps,
        )

        z_tf = None
        if not self._predictor_inference_consistent:
            # Teacher forcing (仅基于 encoder z 的可用帧)
            T_enc = z.size(1) // tokens_per_frame  # encoder 实际编码的帧数
            _z_enc = z[:, :-tokens_per_frame]
            _s = states[:, : T_enc - 1]
            _e = extrinsics[:, : T_enc - 1]
            z_tf = _step_predictor(_z_enc, actions[:, : T_enc - 1], _s, _e)

        # Autoregressive rollout
        # 使用 predictor timeline 的 future_steps 扩展 AR loop，而不是按 z 的帧数（仅 T_obs）
        # 训练时 num_total = num_target_frames (e.g. 20)，这里保持一致
        num_total = predictor_observed_steps + predictor_future_steps

        return predictor_autoregressive_rollout(
            _step_predictor,
            z,
            actions,
            states,
            extrinsics,
            num_obs=num_obs,
            tokens_per_frame=tokens_per_frame,
            num_total=num_total,
            predictor_inference_consistent=self._predictor_inference_consistent,
            z_tf=z_tf,
            rollout_end_step=rollout_end_step,
        )

    def _build_status_feature(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        driving_command: Optional[torch.Tensor] = None,
        ego_dynamics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """构建 planner 所需的 status_feature。"""
        if self._predictor_inference_consistent:
            return prepare_inference_consistent_status_vector(
                states,
                num_observed=self._num_observed_frames,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
                state_dim=resolve_planner_status_dim(self._config),
                use_drive_command=resolve_planner_use_drive_command(self._config),
            )
        else:
            return prepare_status_feature(
                states,
                actions,
                status_mode=self._status_mode,
                use_states_for_planner=self._use_states_for_planner,
                action_dim=self._action_dim,
                driving_command=driving_command,
                ego_dynamics=ego_dynamics,
            )
