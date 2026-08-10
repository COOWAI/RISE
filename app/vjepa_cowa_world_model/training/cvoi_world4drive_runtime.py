"""Retained World4Drive evaluation DTOs and NavSim model boundary.

This module owns the geometry-free 12-element NavSim batch adapter and the
immutable values exchanged by World4Drive collection.  Legacy training owners
temporarily re-export these exact objects so callers cannot fork class identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training.cf_trajectory_quality import counterfactual_quality_schema
from app.vjepa_cowa_world_model.training.configs.cvoi_world4drive import (
    CVOI_WORLD4DRIVE_LINEAGE_ORDER,
    CvoiWorld4DriveDirectConfig,
)
from app.vjepa_cowa_world_model.training.cvoi_audit import load_cvoi_audit_manifest, validate_cvoi_audit_signature
from app.vjepa_cowa_world_model.training.cvoi_execution import cvoi_sample_seed, resolve_cvoi_evaluation_seed
from app.vjepa_cowa_world_model.training.cvoi_value import validate_field_warmup_data_provenance
from app.vjepa_cowa_world_model.training.geometry_outcome import PlanningOutcomeEvaluator
from app.vjepa_cowa_world_model.training.runtimes.sequential_rollout_runtime import run_sequential_rollout
from app.vjepa_cowa_world_model.training.sequential_gate_training import load_sequential_gate_checkpoint

_MAX_HORIZON = 3
_WORLD4DRIVE_GATE_PIPELINE_VERSION = "offline_cvoi_gate_distillation_v1"
_WORLD4DRIVE_GATE_BINDING_ATTR = "_cvoi_world4drive_runtime_binding"
_WORLD4DRIVE_ORACLE_PROTOCOL = "real_geometry_cvoi_oracle_v1"
_WORLD4DRIVE_P0_FIELD_SENTINEL_SHA256 = hashlib.sha256(b"cvoi_p0_no_field_v1").hexdigest()
WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "guidance_steps",
        "guidance_objective",
        "guidance_step_size",
        "guidance_max_delta_norm",
        "guidance_detach_output",
        "audit_signature",
        "predictor_type",
        "runtime_normalize_reps",
        "tokens_per_frame",
        "num_observed_frames",
        "num_target_frames",
        "timestep_sec",
        "multiview_signature",
        "planner_signature",
        "world_execution_signature",
        "execution_dtype_signature",
        "inference_rng_signature",
        "world_model_sha256",
        "token_ae_sha256",
        "parent_planner_sha256",
        "dual_value_sha256",
        "gate_sha256",
        "ablation_signature",
        "p0_protocol",
        "p0_prefix_distribution",
    }
)
_WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS = WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS
_MODEL_METADATA_KEYS = frozenset(
    {
        "camera_names",
        "camera_intrinsics",
        "camera2ego",
        "metadata_valid_mask",
        "observed_metadata_valid_mask",
    }
)
_TRACE_FIELDS = frozenset(
    {
        "stop_horizon",
        "decisions",
        "predicted_deltas",
        "rollout_latency_ms",
        "guidance",
    }
)
_GUIDANCE_FIELDS = frozenset(
    {
        "guidance_steps",
        "guidance_skipped_h0",
        "delta_norm",
        "field_value_before",
        "field_value_after",
    }
)
_FORBIDDEN_ONLINE_FIELD_FRAGMENTS = (
    "agent_box",
    "agent_mask",
    "collision",
    "future_geometry",
    "future_agent",
    "ground_truth",
    "gt_trajectory",
    "near_miss",
    "offroad",
    "safety",
    "ttc",
)


@dataclass(frozen=True)
class CvoiWorld4DriveRuntimeBinding:
    """Read-only checkpoint binding for one retained World4Drive lineage."""

    protocol_version: str
    lineage: str
    stage: str
    max_horizon: int
    tokens_per_frame: int
    compute_costs: tuple[float, ...]
    controller_batch_size: int
    controller_lineage: str
    guidance_steps: int
    guidance_objective: str
    timestep_sec: float
    world_model_checkpoint: str
    token_ae_checkpoint: str
    unguided_planner_checkpoint: str
    field_checkpoint: Optional[str]
    guided_planner_checkpoint: Optional[str]
    dual_value_checkpoint: str
    oracle_path: str
    gate_checkpoint: str
    checkpoint_audit_manifest_path: str


def build_cvoi_world4drive_runtime_binding(
    direct: CvoiWorld4DriveDirectConfig,
    *,
    lineage: str,
) -> CvoiWorld4DriveRuntimeBinding:
    """Project one exact direct-config lineage onto the frozen runtime contract."""

    if not isinstance(direct, CvoiWorld4DriveDirectConfig):
        raise TypeError("direct must be a parsed CvoiWorld4DriveDirectConfig")
    if lineage not in CVOI_WORLD4DRIVE_LINEAGE_ORDER:
        raise ValueError(f"unsupported World4Drive lineage: {lineage!r}")
    artifacts = direct.lineages[lineage]
    p0 = lineage == "p0_controller"
    if p0:
        if artifacts.field_checkpoint is not None:
            raise ValueError("P0 World4Drive runtime binding forbids a Field checkpoint")
        if artifacts.planner_checkpoint != direct.common_artifacts.unguided_planner_checkpoint:
            raise ValueError("P0 World4Drive Planner must be the common unguided Planner")
    elif artifacts.field_checkpoint is None:
        raise ValueError(f"World4Drive lineage {lineage!r} requires a Field checkpoint")
    return CvoiWorld4DriveRuntimeBinding(
        protocol_version="world4drive_evaluation_v1",
        lineage=lineage,
        stage="evaluation",
        max_horizon=3,
        tokens_per_frame=128,
        compute_costs=(0.0, 1.0, 2.0, 3.0),
        controller_batch_size=1,
        controller_lineage="p0_controller" if p0 else "value_guided",
        guidance_steps=2,
        guidance_objective="last",
        timestep_sec=0.5,
        world_model_checkpoint=direct.common_artifacts.world_model_checkpoint,
        token_ae_checkpoint=direct.common_artifacts.token_ae_checkpoint,
        unguided_planner_checkpoint=direct.common_artifacts.unguided_planner_checkpoint,
        field_checkpoint=None if p0 else artifacts.field_checkpoint,
        guided_planner_checkpoint=None if p0 else artifacts.planner_checkpoint,
        dual_value_checkpoint=artifacts.stop_checkpoint,
        oracle_path=artifacts.oracle_path,
        gate_checkpoint=artifacts.gate_checkpoint,
        checkpoint_audit_manifest_path=direct.checkpoint_audit_manifest_path,
    )


def validate_world4drive_direct_inputs(
    direct: CvoiWorld4DriveDirectConfig,
    *,
    require_model_artifacts: bool,
) -> None:
    """Require every direct input before a job is allowed to create outputs."""

    paths = [
        direct.checkpoint_audit_manifest_path,
        direct.dataset_audit_manifest_path,
    ]
    if require_model_artifacts:
        paths.extend(
            (
                direct.common_artifacts.world_model_checkpoint,
                direct.common_artifacts.token_ae_checkpoint,
                direct.common_artifacts.unguided_planner_checkpoint,
            )
        )
        for _lineage, artifacts in direct.lineage_items():
            paths.extend(
                path
                for path in (
                    artifacts.planner_checkpoint,
                    artifacts.stop_checkpoint,
                    artifacts.oracle_path,
                    artifacts.gate_checkpoint,
                    artifacts.field_checkpoint,
                )
                if path is not None
            )
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"World4Drive direct input file(s) do not exist: {missing}")


def validate_world4drive_base_data_root(
    base_config: object,
    direct_config: CvoiWorld4DriveDirectConfig,
) -> None:
    """Require the duplicated base cohort and all direct evaluation seeds to remain identical."""

    navsim = getattr(getattr(base_config, "data", None), "navsim", None)
    roots = getattr(navsim, "val_roots", None)
    if isinstance(roots, (str, bytes)) or not isinstance(roots, Sequence):
        raise ValueError("World4Drive data.navsim.val_roots must be an exact one-root sequence")
    roots = tuple(roots)
    if len(roots) != 1 or not isinstance(roots[0], Mapping) or dict(roots[0]) != dict(direct_config.real_val_root):
        raise ValueError("World4Drive data.navsim.val_roots must exactly equal cvoi_world4drive.real_val_root")
    seed = getattr(getattr(base_config, "meta", None), "seed", None)
    if type(seed) is not int or seed != direct_config.random_stop_seed:
        raise ValueError("World4Drive base meta.seed must exactly equal cvoi_world4drive.random_stop_seed")
    if direct_config.random_stop_seed != 239:
        raise ValueError("World4Drive direct evaluation requires the retained seed 239")


def _load_world4drive_dataset_audit(*args: object, **kwargs: object) -> object:
    from app.vjepa_cowa_world_model.training.cvoi_world4drive_identity import load_world4drive_audit_manifest

    return load_world4drive_audit_manifest(*args, **kwargs)


def preflight_world4drive_direct_semantics(direct_config: CvoiWorld4DriveDirectConfig) -> None:
    """CPU-only live validation for the dataset audit and all three checkpoint bindings."""

    _load_world4drive_dataset_audit(
        direct_config.dataset_audit_manifest_path,
        expected_real_root=direct_config.real_val_root,
    )
    for lineage in CVOI_WORLD4DRIVE_LINEAGE_ORDER:
        binding = build_cvoi_world4drive_runtime_binding(direct_config, lineage=lineage)
        build_world4drive_evaluation_runtime_signature_payload(binding)
        validate_cvoi_world4drive_gate(binding)


def require_world4drive_outputs_absent(paths: Iterable[str | Path]) -> None:
    """Reject an output set if any target already exists, including dangling links."""

    existing = [str(Path(path)) for path in paths if Path(path).exists() or Path(path).is_symlink()]
    if existing:
        raise FileExistsError(f"World4Drive direct output(s) already exist: {existing}")


def require_world4drive_inputs_present(paths: Iterable[str | Path]) -> None:
    """Require every assembled report input to be an existing directory."""

    missing = [str(Path(path)) for path in paths if not Path(path).is_dir()]
    if missing:
        raise FileNotFoundError(f"World4Drive direct input directories do not exist: {missing}")


def _sha256_file(path: str | Path) -> str:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"World4Drive artifact does not exist: {artifact}")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    try:
        canonical = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("World4Drive provenance must be JSON serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _valid_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exactly_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(_exactly_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_exactly_equal(a, b) for a, b in zip(left, right))
    return bool(left == right)


def _world4drive_checkpoint_audit(binding: CvoiWorld4DriveRuntimeBinding):
    return load_cvoi_audit_manifest(
        binding.checkpoint_audit_manifest_path,
        verification_mode="receipt_only",
    )


def _validate_value_architecture(value: object, *, embed_dim: int) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("PrefixDualValueModel checkpoint architecture must be a mapping")
    expected_fields = {"embed_dim", "hidden_dim", "num_layers", "dropout"}
    if set(value) != expected_fields:
        raise ValueError("PrefixDualValueModel checkpoint architecture fields mismatch")
    architecture = dict(value)
    for name in ("embed_dim", "hidden_dim", "num_layers"):
        field = architecture[name]
        if type(field) is not int or field <= 0:
            raise ValueError(f"PrefixDualValueModel checkpoint architecture {name} must be a positive integer")
    dropout = architecture["dropout"]
    if type(dropout) is not float or not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("PrefixDualValueModel checkpoint architecture dropout must be a float in [0, 1)")
    if architecture["embed_dim"] != int(embed_dim):
        raise ValueError(
            f"World4Drive Value embed_dim must match encoder embed_dim={int(embed_dim)}, "
            f"got {architecture['embed_dim']!r}"
        )
    return architecture


def _expected_world4drive_ablation(lineage: str) -> dict[str, object]:
    return {
        "schema": "cvoi_ablation_v1",
        "experiment_role": "ablation",
        "branch_id": f"{lineage}_s239",
        "shared_cohort_id": "seed_s239",
        "cf_field_supervision": "hazard_quality" if lineage == "real_cf_value" else "none",
        "field_calibration_mode": "local_geometry",
        "p0_prefix_mode": "uniform",
        "gate_feature_mode": "full",
        "train_seed": 239,
        "evaluation_seed": 239,
    }


def _expected_world4drive_planner_ablation(
    binding: CvoiWorld4DriveRuntimeBinding,
    *,
    p0: bool,
) -> dict[str, object]:
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
    return _expected_world4drive_ablation(binding.lineage)


def read_world4drive_planner_runtime_signature(
    binding: CvoiWorld4DriveRuntimeBinding,
    *,
    p0: bool,
) -> Mapping[str, object]:
    """Safely read and validate one retained Planner's self-described lineage."""

    path = binding.unguided_planner_checkpoint if p0 else binding.guided_planner_checkpoint
    if not isinstance(path, str) or not path.strip():
        raise ValueError("World4Drive Planner binding path is missing")
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"World4Drive Planner checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("World4Drive Planner checkpoint must contain a mapping")
    signature = payload.get("cvoi_runtime_signature")
    if not isinstance(signature, Mapping) or set(signature) != _WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS:
        actual = set(signature) if isinstance(signature, Mapping) else set()
        raise ValueError(
            "World4Drive Planner runtime signature fields mismatch: "
            f"missing={sorted(_WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS - actual)}, "
            f"unexpected={sorted(actual - _WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS)}"
        )
    expected_stage = "unguided_planner" if p0 else "guided_planner"
    expected = {
        "schema": "cvoi_dual_value_v1",
        "stage": expected_stage,
        "guidance_steps": binding.guidance_steps,
        "guidance_objective": binding.guidance_objective,
        "audit_signature": _world4drive_checkpoint_audit(binding).to_dict(),
        "tokens_per_frame": binding.tokens_per_frame,
        "timestep_sec": binding.timestep_sec,
        "world_model_sha256": _sha256_file(binding.world_model_checkpoint),
        "token_ae_sha256": _sha256_file(binding.token_ae_checkpoint),
        "gate_sha256": None,
        "ablation_signature": _expected_world4drive_planner_ablation(binding, p0=p0),
        "p0_protocol": "fixed_final_epoch_v1",
        "p0_prefix_distribution": {str(horizon): 0.25 for horizon in range(_MAX_HORIZON + 1)},
    }
    mismatch = {
        key: (signature.get(key), value)
        for key, value in expected.items()
        if not _exactly_equal(signature.get(key), value)
    }
    if mismatch:
        raise ValueError(f"World4Drive Planner retained lineage mismatch: {mismatch}")
    if p0:
        _valid_sha256(signature.get("parent_planner_sha256"), name="P0 parent_planner_sha256")
        if signature.get("dual_value_sha256") is not None:
            raise ValueError("World4Drive P0 Planner must not reference a Field checkpoint")
        if payload.get("epoch") != 20:
            raise ValueError("World4Drive P0 trust anchor must be fixed final epoch=20")
    else:
        if binding.field_checkpoint is None:
            raise ValueError("World4Drive P1 Planner binding requires a Field checkpoint")
        parent_expected = {
            "parent_planner_sha256": _sha256_file(binding.unguided_planner_checkpoint),
            "dual_value_sha256": _sha256_file(binding.field_checkpoint),
        }
        parent_mismatch = {
            key: (signature.get(key), value) for key, value in parent_expected.items() if signature.get(key) != value
        }
        if parent_mismatch:
            raise ValueError(f"World4Drive P1 direct-parent lineage mismatch: {parent_mismatch}")
    return MappingProxyType(dict(signature))


def build_world4drive_evaluation_runtime_signature_payload(
    binding: CvoiWorld4DriveRuntimeBinding,
) -> Mapping[str, object]:
    """Rebuild the legacy evaluation signature from retained validated artifacts."""

    p0_signature = read_world4drive_planner_runtime_signature(binding, p0=True)
    selected_signature = p0_signature
    selected_planner = binding.unguided_planner_checkpoint
    if binding.controller_lineage == "value_guided":
        p1_signature = read_world4drive_planner_runtime_signature(binding, p0=False)
        shared_fields = _WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS - {
            "stage",
            "parent_planner_sha256",
            "dual_value_sha256",
            "ablation_signature",
        }
        shared_mismatch = {
            field: (p0_signature[field], p1_signature[field])
            for field in shared_fields
            if not _exactly_equal(p0_signature[field], p1_signature[field])
        }
        if shared_mismatch:
            raise ValueError(f"World4Drive P0/P1 common runtime lineage mismatch: {shared_mismatch}")
        selected_signature = p1_signature
        if binding.guided_planner_checkpoint is None:
            raise ValueError("World4Drive value-guided binding requires a P1 Planner")
        selected_planner = binding.guided_planner_checkpoint

    value_payload = read_world4drive_value_checkpoint(binding.dual_value_checkpoint)
    architecture = value_payload.get("architecture")
    if not isinstance(architecture, Mapping) or type(architecture.get("embed_dim")) is not int:
        raise ValueError("World4Drive Stop checkpoint architecture is missing embed_dim")
    validate_world4drive_value_lineage(
        value_payload,
        binding=binding,
        embed_dim=architecture["embed_dim"],
    )
    runtime_payload = dict(selected_signature)
    runtime_payload.update(
        {
            "stage": "evaluation",
            "parent_planner_sha256": _sha256_file(selected_planner),
            "dual_value_sha256": _sha256_file(binding.dual_value_checkpoint),
            "gate_sha256": _sha256_file(binding.gate_checkpoint),
            "ablation_signature": _expected_world4drive_ablation(binding.lineage),
        }
    )
    return MappingProxyType(runtime_payload)


def _validate_world4drive_value_state(
    state: object,
    *,
    model: PrefixDualValueModel,
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("World4Drive Value state_dict must be a non-empty mapping")
    if any(not isinstance(key, str) or not key or not torch.is_tensor(value) for key, value in state.items()):
        raise ValueError("World4Drive Value state_dict must map non-empty string keys to tensors")
    checkpoint = dict(state)
    target = model.state_dict()
    if set(checkpoint) != set(target):
        raise ValueError(
            "World4Drive Value state keys mismatch: "
            f"missing={sorted(set(target) - set(checkpoint))}, "
            f"unexpected={sorted(set(checkpoint) - set(target))}"
        )
    shape_mismatches = {
        key: (tuple(checkpoint[key].shape), tuple(target[key].shape))
        for key in target
        if checkpoint[key].shape != target[key].shape
    }
    if shape_mismatches:
        raise ValueError(f"World4Drive Value state shape mismatch: {shape_mismatches}")
    dtype_mismatches = {
        key: (checkpoint[key].dtype, target[key].dtype) for key in target if checkpoint[key].dtype != target[key].dtype
    }
    if dtype_mismatches:
        raise ValueError(f"World4Drive Value state dtype mismatch: {dtype_mismatches}")
    return checkpoint


def read_world4drive_value_checkpoint(path: str | Path) -> Mapping[str, object]:
    """Read a legacy-shaped retained Value artifact through the safe tensor-only loader."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"World4Drive Value checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("World4Drive Value checkpoint must contain a mapping")
    return payload


def validate_world4drive_value_lineage(
    payload: Mapping[str, object],
    *,
    binding: CvoiWorld4DriveRuntimeBinding,
    embed_dim: int,
) -> None:
    """Validate the complete immutable Value lineage before loading any tensor."""

    if not isinstance(payload, Mapping):
        raise ValueError("World4Drive Value checkpoint must contain a mapping")
    required = {
        "schema",
        "phase",
        "architecture",
        "real_evaluator_signature",
        "cf_quality_schema",
        "audit_signature",
        "lineage",
        "controller_lineage",
        "ablation_signature",
        "field_warmup_data_provenance",
        "state_dict",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"World4Drive Value checkpoint is missing required fields: {sorted(missing)}")
    if payload["schema"] != "cvoi_dual_value_v1" or payload["phase"] != "stop_calibrated":
        raise ValueError("World4Drive Value checkpoint must be legacy stop_calibrated cvoi_dual_value_v1")
    if payload["controller_lineage"] != binding.controller_lineage:
        raise ValueError("World4Drive Value controller lineage does not match the runtime binding")
    architecture = _validate_value_architecture(payload["architecture"], embed_dim=embed_dim)

    expected_evaluator = PlanningOutcomeEvaluator(timestep_sec=binding.timestep_sec).signature
    if not _exactly_equal(payload["real_evaluator_signature"], expected_evaluator):
        raise ValueError("World4Drive Value real evaluator signature mismatch")
    expected_cf = counterfactual_quality_schema(timestep_sec=binding.timestep_sec, max_progress_m=20.0)
    if not _exactly_equal(payload["cf_quality_schema"], expected_cf):
        raise ValueError("World4Drive Value CF quality schema mismatch")

    stored_audit = validate_cvoi_audit_signature(payload["audit_signature"]).to_dict()
    expected_audit = _world4drive_checkpoint_audit(binding).to_dict()
    if stored_audit != expected_audit:
        raise ValueError("World4Drive Value checkpoint audit signature mismatch")

    expected_ablation = _expected_world4drive_ablation(binding.lineage)
    if not _exactly_equal(payload["ablation_signature"], expected_ablation):
        raise ValueError("World4Drive Value checkpoint ablation lineage mismatch")
    warmup = validate_field_warmup_data_provenance(payload["field_warmup_data_provenance"])
    expected_domain = "real_cf" if binding.lineage == "real_cf_value" else "real"
    if warmup["domain"] != expected_domain:
        raise ValueError("World4Drive Value warm-up domain does not match the runtime lineage")
    root_domains = {root["domain"] for root in warmup["roots"]}
    if expected_domain == "real":
        if root_domains != {"real"} or warmup["real_sample_count"] <= 0 or warmup["cf_sample_count"] != 0:
            raise ValueError("strict Real-only World4Drive Value contains CF warm-up provenance")
    elif (
        root_domains != {"real", "counterfactual"}
        or warmup["real_sample_count"] <= 0
        or warmup["cf_sample_count"] <= 0
    ):
        raise ValueError("World4Drive Real+CF Value requires positive samples and roots from both domains")

    lineage = payload["lineage"]
    if not isinstance(lineage, Mapping):
        raise ValueError("World4Drive Value checkpoint lineage must be a mapping")
    lineage_fields = {
        "world_model_sha256",
        "token_ae_sha256",
        "unguided_planner_sha256",
        "parent_value_sha256",
        "guided_planner_sha256",
        "guidance_signature",
    }
    if set(lineage) != lineage_fields:
        raise ValueError("World4Drive Value checkpoint lineage fields mismatch")
    expected_common = {
        "world_model_sha256": _sha256_file(binding.world_model_checkpoint),
        "token_ae_sha256": _sha256_file(binding.token_ae_checkpoint),
        "unguided_planner_sha256": _sha256_file(binding.unguided_planner_checkpoint),
    }
    common_mismatch = {
        key: (lineage.get(key), value) for key, value in expected_common.items() if lineage.get(key) != value
    }
    if common_mismatch:
        raise ValueError(f"World4Drive Value common artifact lineage mismatch: {common_mismatch}")
    parent_sha = _valid_sha256(lineage["parent_value_sha256"], name="parent_value_sha256")
    del parent_sha
    if binding.controller_lineage == "p0_controller":
        if binding.field_checkpoint is not None or binding.guided_planner_checkpoint is not None:
            raise ValueError("P0 World4Drive runtime forbids Field/P1 artifacts")
        if lineage["guided_planner_sha256"] != expected_common["unguided_planner_sha256"]:
            raise ValueError("P0 World4Drive Value must deploy the P0 planner")
        if lineage["guidance_signature"] is not None:
            raise ValueError("P0 World4Drive Value must not declare Guidance")
    else:
        if binding.field_checkpoint is None or binding.guided_planner_checkpoint is None:
            raise ValueError("Value-guided World4Drive runtime requires Field/P1 artifacts")
        expected_parent = _sha256_file(binding.field_checkpoint)
        expected_planner = _sha256_file(binding.guided_planner_checkpoint)
        if lineage["parent_value_sha256"] != expected_parent:
            raise ValueError("World4Drive Value parent Field lineage mismatch")
        if lineage["guided_planner_sha256"] != expected_planner:
            raise ValueError("World4Drive Value guided Planner lineage mismatch")
        expected_guidance = {
            "schema": "cvoi_fixed_guidance_v1",
            "steps": binding.guidance_steps,
            "objective": binding.guidance_objective,
            "step_size": 0.05,
            "max_delta_norm": 0.25,
            "detach_output": True,
        }
        if not _exactly_equal(lineage["guidance_signature"], expected_guidance):
            raise ValueError("World4Drive Value Guidance lineage mismatch")

    with torch.random.fork_rng(devices=[]):
        probe = PrefixDualValueModel(**architecture)
    _validate_world4drive_value_state(payload["state_dict"], model=probe)


def load_cvoi_world4drive_value_model(
    binding: CvoiWorld4DriveRuntimeBinding,
    *,
    embed_dim: int,
    device: torch.device,
) -> PrefixDualValueModel:
    """Load and freeze the single Stop/Field dual-head model used by evaluation."""

    payload = read_world4drive_value_checkpoint(binding.dual_value_checkpoint)
    validate_world4drive_value_lineage(payload, binding=binding, embed_dim=embed_dim)
    with torch.random.fork_rng(devices=[]):
        model = PrefixDualValueModel(**payload["architecture"])
    state = _validate_world4drive_value_state(payload["state_dict"], model=model)
    model.load_state_dict(state, strict=True)
    model.to(device=device).eval().requires_grad_(False)
    return model


def build_world4drive_gate_provenance(binding: CvoiWorld4DriveRuntimeBinding) -> Mapping[str, str]:
    """Rebuild the exact legacy Gate provenance from explicit read-only paths."""

    audit = _world4drive_checkpoint_audit(binding)
    provenance = {
        "world_model": _sha256_file(binding.world_model_checkpoint),
        "unguided_planner": _sha256_file(binding.unguided_planner_checkpoint),
        "value": _sha256_file(binding.dual_value_checkpoint),
        "evaluator": _canonical_sha256(PlanningOutcomeEvaluator(timestep_sec=binding.timestep_sec).signature),
        **audit.to_provenance(),
    }
    if binding.controller_lineage == "p0_controller":
        provenance.update(
            {
                "field": _WORLD4DRIVE_P0_FIELD_SENTINEL_SHA256,
                "planner": provenance["unguided_planner"],
                "latent_policy": "raw_prefix",
                "field_guidance": "false",
            }
        )
    else:
        if binding.field_checkpoint is None or binding.guided_planner_checkpoint is None:
            raise ValueError("Value-guided World4Drive Gate provenance requires Field/P1 artifacts")
        provenance.update(
            {
                "field": _sha256_file(binding.field_checkpoint),
                "planner": _sha256_file(binding.guided_planner_checkpoint),
            }
        )
    provenance.update(
        {
            "oracle_protocol": _WORLD4DRIVE_ORACLE_PROTOCOL,
            "oracle_sha256": _sha256_file(binding.oracle_path),
            "gate_pipeline": _WORLD4DRIVE_GATE_PIPELINE_VERSION,
        }
    )
    return provenance


def load_cvoi_world4drive_gate(
    binding: CvoiWorld4DriveRuntimeBinding,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Load one existing legacy-shaped Gate without exposing a training protocol."""

    expected = build_world4drive_gate_provenance(binding)
    loaded = load_sequential_gate_checkpoint(
        binding.gate_checkpoint,
        device=device,
        expected_provenance=expected,
        expected_protocol_version="legacy_v1",
        _checkpoint_weights_only=True,
        _prevalidate_state_dict=True,
    )
    if loaded is None:
        raise RuntimeError("World4Drive Gate runtime loader did not return a module")
    existing_binding = getattr(loaded, _WORLD4DRIVE_GATE_BINDING_ATTR, None)
    if existing_binding is not None and existing_binding != binding:
        raise ValueError("loaded World4Drive Gate already carries a different runtime binding")
    setattr(loaded, _WORLD4DRIVE_GATE_BINDING_ATTR, binding)
    return loaded


def validate_cvoi_world4drive_gate(binding: CvoiWorld4DriveRuntimeBinding) -> None:
    """Validate the bound Gate without loading its state into a runtime module."""

    expected = build_world4drive_gate_provenance(binding)
    loaded = load_sequential_gate_checkpoint(
        binding.gate_checkpoint,
        device=torch.device("cpu"),
        expected_provenance=expected,
        expected_protocol_version="legacy_v1",
        _checkpoint_weights_only=True,
        _prevalidate_state_dict=True,
        _validate_only=True,
    )
    if loaded is not None:
        raise RuntimeError("World4Drive Gate validate-only loader unexpectedly returned a module")


def _finite_number(name: str, value: object, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result}")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _exact_fields(name: str, value: Mapping[str, object], expected: frozenset[str]) -> None:
    fields = set(value)
    if any(not isinstance(field, str) for field in fields):
        raise TypeError(f"{name} field names must be strings")
    missing = sorted(expected - fields)
    unexpected = sorted(fields - expected)
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")
    if unexpected:
        raise ValueError(f"{name} contains unexpected fields: {unexpected}")


def _reject_online_future_fields(value: Mapping[str, object], *, prefix: str = "") -> None:
    visited: set[int] = set()

    def visit(mapping: Mapping[object, object], path_prefix: str) -> None:
        mapping_id = id(mapping)
        if mapping_id in visited:
            return
        visited.add(mapping_id)
        for field, nested in mapping.items():
            if not isinstance(field, str):
                continue
            path = f"{path_prefix}.{field}" if path_prefix else field
            normalized = field.strip().lower().replace("-", "_").replace(" ", "_")
            if any(fragment in normalized for fragment in _FORBIDDEN_ONLINE_FIELD_FRAGMENTS):
                raise ValueError(f"Controller trace contains forbidden future/evaluation field: {path}")
            if isinstance(nested, Mapping):
                visit(nested, path)

    visit(value, prefix)


@dataclass(frozen=True)
class CvoiPlannerEvaluation:
    """Deployment planner output at one fixed raw rollout horizon."""

    pred_trajs: torch.Tensor
    confidences: torch.Tensor
    latency_ms: float
    guidance_steps: int


@dataclass(frozen=True)
class NavSimCvoiModelBatch:
    """Geometry-free inputs allowed to cross into the frozen model runtime."""

    context_frames: torch.Tensor
    actions: torch.Tensor
    states: torch.Tensor
    extrinsics: torch.Tensor
    driving_command: torch.Tensor
    ego_dynamics: torch.Tensor
    proposal_context_frames: Optional[torch.Tensor]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class NavSimCvoiEncodedBatch:
    """Predictor rollout latents and opaque online-only planner contexts."""

    z_observed: torch.Tensor
    z_future: torch.Tensor
    model_contexts: tuple[object, ...]


@dataclass(frozen=True)
class PreparedNavSimCvoiDeployment:
    """Observed-only NavSim features prepared outside the timed online region."""

    features: Mapping[str, object]
    raw_poses: int

    def __post_init__(self) -> None:
        if not isinstance(self.features, Mapping) or not self.features:
            raise ValueError("prepared NavSim CVoI features must be a non-empty mapping")
        if isinstance(self.raw_poses, bool) or not isinstance(self.raw_poses, int) or self.raw_poses < 1:
            raise ValueError("prepared NavSim CVoI raw_poses must be a positive integer")
        object.__setattr__(self, "features", dict(self.features))


@dataclass(frozen=True)
class CvoiControllerTrace:
    """Validated online-only Controller trace for one deployed trajectory."""

    stop_horizon: int
    decisions: tuple[str, ...]
    predicted_deltas: tuple[float, ...]
    rollout_latency_ms: float
    guidance: Mapping[str, float]

    def __post_init__(self) -> None:
        if isinstance(self.stop_horizon, bool) or not isinstance(self.stop_horizon, int):
            raise TypeError("Controller trace stop_horizon must be an integer")
        if self.stop_horizon < 0 or self.stop_horizon > _MAX_HORIZON:
            raise ValueError(f"Controller trace stop_horizon must be in [0, {_MAX_HORIZON}]")

        if isinstance(self.decisions, (str, bytes)) or not isinstance(self.decisions, Sequence):
            raise TypeError("Controller trace decisions must be a sequence")
        decisions = tuple(self.decisions)
        expected_decisions = ("ROLL",) * self.stop_horizon + ("STOP",)
        if decisions != expected_decisions:
            raise ValueError(
                "Controller trace decisions must contain one ROLL per executed step followed by STOP; "
                f"expected {expected_decisions}, got {decisions}"
            )

        if isinstance(self.predicted_deltas, (str, bytes)) or not isinstance(self.predicted_deltas, Sequence):
            raise TypeError("Controller trace predicted_deltas must be a sequence")
        expected_delta_count = min(self.stop_horizon + 1, _MAX_HORIZON)
        if len(self.predicted_deltas) != expected_delta_count:
            raise ValueError(
                "Controller trace predicted_deltas must contain every evaluated Gate output; "
                f"expected {expected_delta_count}, got {len(self.predicted_deltas)}"
            )
        predicted_deltas = tuple(
            _finite_number(f"Controller trace predicted_deltas[{index}]", delta)
            for index, delta in enumerate(self.predicted_deltas)
        )
        latency_ms = _finite_number("Controller trace rollout_latency_ms", self.rollout_latency_ms, nonnegative=True)

        if not isinstance(self.guidance, Mapping):
            raise TypeError("Controller trace guidance must be a mapping")
        _reject_online_future_fields({"guidance": self.guidance})
        _exact_fields("Controller trace guidance", self.guidance, _GUIDANCE_FIELDS)
        guidance = {
            name: _finite_number(
                f"Controller trace guidance.{name}",
                self.guidance[name],
                nonnegative=name in {"guidance_steps", "guidance_skipped_h0", "delta_norm"},
            )
            for name in _GUIDANCE_FIELDS
        }
        expected_steps = 0.0 if self.stop_horizon == 0 else 2.0
        expected_skip = 1.0 if self.stop_horizon == 0 else 0.0
        if guidance["guidance_steps"] != expected_steps:
            raise ValueError(f"Controller trace Guidance must use K={int(expected_steps)} at h={self.stop_horizon}")
        if guidance["guidance_skipped_h0"] != expected_skip:
            raise ValueError("Controller trace guidance_skipped_h0 is inconsistent with stop_horizon")
        if self.stop_horizon == 0 and guidance["delta_norm"] != 0.0:
            raise ValueError("Controller trace h=0 Guidance must have zero delta_norm")

        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "predicted_deltas", predicted_deltas)
        object.__setattr__(self, "rollout_latency_ms", latency_ms)
        object.__setattr__(self, "guidance", MappingProxyType(guidance))

    @classmethod
    def from_mapping(cls, trace: Mapping[str, object]) -> "CvoiControllerTrace":
        """Parse the exact dictionary returned by ``get_last_cvoi_trace``."""

        if not isinstance(trace, Mapping):
            raise TypeError(f"Controller trace must be a mapping, got {type(trace).__name__}")
        _reject_online_future_fields(trace)
        _exact_fields("Controller trace", trace, _TRACE_FIELDS)
        return cls(
            stop_horizon=trace["stop_horizon"],
            decisions=trace["decisions"],
            predicted_deltas=trace["predicted_deltas"],
            rollout_latency_ms=trace["rollout_latency_ms"],
            guidance=trace["guidance"],
        )


@dataclass(frozen=True)
class CvoiFixedComputeTrace:
    """Minimal compute accounting for a forced horizon, without Gate outputs."""

    horizon: int
    latency_ms: float
    guidance_steps: int

    def __post_init__(self) -> None:
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("fixed compute trace horizon must be an integer")
        if self.horizon < 0 or self.horizon > _MAX_HORIZON:
            raise ValueError(f"fixed compute trace horizon must be in [0, {_MAX_HORIZON}]")
        object.__setattr__(
            self,
            "latency_ms",
            _finite_number("fixed compute trace latency_ms", self.latency_ms, nonnegative=True),
        )
        if isinstance(self.guidance_steps, bool) or not isinstance(self.guidance_steps, int):
            raise TypeError("fixed compute trace guidance_steps must be an integer")
        if self.guidance_steps not in {0, 1, 2, 3, 4}:
            raise ValueError("fixed compute trace guidance_steps must be one of {0,1,2,3,4}")
        if self.horizon == 0 and self.guidance_steps != 0:
            raise ValueError("fixed compute trace h=0 must skip Guidance")


@dataclass(frozen=True)
class CvoiDeploymentOutput:
    """Raw planner candidates, deployment confidence, and policy compute traces."""

    pred_trajs: torch.Tensor
    confidences: torch.Tensor
    controller_traces: tuple[Mapping[str, object] | CvoiControllerTrace | CvoiFixedComputeTrace, ...]

    def __post_init__(self) -> None:
        trajectories = self.pred_trajs
        if (
            not isinstance(trajectories, torch.Tensor)
            or trajectories.ndim != 4
            or trajectories.shape[0] < 1
            or trajectories.shape[1] < 1
            or trajectories.shape[2] < 1
            or trajectories.shape[3] != 3
            or not trajectories.dtype.is_floating_point
            or not bool(torch.isfinite(trajectories).all().item())
        ):
            shape = (
                tuple(trajectories.shape) if isinstance(trajectories, torch.Tensor) else type(trajectories).__name__
            )
            raise ValueError(f"pred_trajs must be finite floating [B,K>0,P>0,3], got {shape}")
        confidences = self.confidences
        if (
            not isinstance(confidences, torch.Tensor)
            or confidences.shape != trajectories.shape[:2]
            or not confidences.dtype.is_floating_point
            or confidences.device != trajectories.device
            or not bool(torch.isfinite(confidences).all().item())
        ):
            raise ValueError("confidences must be finite floating [B,K] on the pred_trajs device")
        if not isinstance(self.controller_traces, tuple) or len(self.controller_traces) != trajectories.shape[0]:
            raise ValueError("controller_traces must be a tuple with one online trace per trajectory")

    @property
    def selected_trajectory(self) -> torch.Tensor:
        """Return the deployment confidence argmax without GT hindsight."""

        indices = self.confidences.argmax(dim=1)
        batch_indices = torch.arange(self.pred_trajs.shape[0], device=self.pred_trajs.device)
        return self.pred_trajs[batch_indices, indices]


def _move_tensor(value: object, *, device: torch.device, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"NavSim CVoI model input {name} must be a tensor")
    return value.to(device=device, non_blocking=True)


def _slice_video_observed(value: torch.Tensor, *, num_observed_frames: int, name: str) -> torch.Tensor:
    if value.ndim == 5:
        time_dim = 2
    elif value.ndim == 6:
        time_dim = 3
    else:
        raise ValueError(f"NavSim CVoI {name} must be [B,C,T,H,W] or [B,V,C,T,H,W]")
    if int(value.shape[time_dim]) < num_observed_frames:
        raise ValueError(f"NavSim CVoI {name} does not contain {num_observed_frames} observed frames")
    index = [slice(None)] * value.ndim
    index[time_dim] = slice(0, num_observed_frames)
    return value[tuple(index)]


def _slice_timeline_observed(value: torch.Tensor, *, length: int, name: str) -> torch.Tensor:
    if value.ndim < 2 or int(value.shape[1]) < length:
        raise ValueError(f"NavSim CVoI {name} does not cover the deployment-visible timeline length={length}")
    return value[:, :length]


def _slice_camera_metadata(value: torch.Tensor, *, num_observed_frames: int, name: str) -> torch.Tensor:
    if value.ndim != 5 or int(value.shape[2]) < num_observed_frames:
        raise ValueError(f"NavSim CVoI metadata[{name!r}] must be [B,V,T,M,M] with T >= observed frames")
    return value[:, :, :num_observed_frames]


def _model_batch(
    navsim_batch: Sequence[object],
    *,
    config: Any,
    device: torch.device,
) -> NavSimCvoiModelBatch:
    if isinstance(navsim_batch, (str, bytes)) or not isinstance(navsim_batch, Sequence) or len(navsim_batch) != 12:
        raise ValueError("NavSim CVoI runtime requires the standard 12-element batch tuple")
    metadata = navsim_batch[11]
    if not isinstance(metadata, Mapping):
        raise ValueError("NavSim CVoI runtime requires metadata at batch element 11")
    num_observed_frames = int(config.train.num_observed_frames)
    if num_observed_frames < 1:
        raise ValueError("NavSim CVoI model boundary requires train.num_observed_frames >= 1")
    safe_metadata = {}
    for key, value in metadata.items():
        if key not in _MODEL_METADATA_KEYS:
            continue
        if key in {"camera_intrinsics", "camera2ego"}:
            value = _slice_camera_metadata(
                _move_tensor(value, device=device, name=key),
                num_observed_frames=num_observed_frames,
                name=key,
            )
        elif key == "metadata_valid_mask":
            value = _slice_timeline_observed(
                _move_tensor(value, device=device, name=key),
                length=num_observed_frames,
                name=key,
            )
        elif isinstance(value, torch.Tensor):
            value = value.to(device=device, non_blocking=True)
        safe_metadata[key] = value
    context_frames = _slice_video_observed(
        _move_tensor(navsim_batch[0], device=device, name="context_frames"),
        num_observed_frames=num_observed_frames,
        name="context_frames",
    )
    proposal_context_frames = None
    if navsim_batch[10] is not None:
        proposal_context_frames = _slice_video_observed(
            _move_tensor(navsim_batch[10], device=device, name="proposal_context_frames"),
            num_observed_frames=num_observed_frames,
            name="proposal_context_frames",
        )
    return NavSimCvoiModelBatch(
        context_frames=context_frames,
        actions=_slice_timeline_observed(
            _move_tensor(navsim_batch[1], device=device, name="actions"),
            length=max(num_observed_frames - 1, 0),
            name="actions",
        ),
        states=_slice_timeline_observed(
            _move_tensor(navsim_batch[2], device=device, name="states"),
            length=num_observed_frames,
            name="states",
        ),
        extrinsics=_slice_timeline_observed(
            _move_tensor(navsim_batch[3], device=device, name="extrinsics"),
            length=num_observed_frames,
            name="extrinsics",
        ),
        driving_command=_slice_timeline_observed(
            _move_tensor(navsim_batch[5], device=device, name="driving_command"),
            length=num_observed_frames,
            name="driving_command",
        ),
        ego_dynamics=_slice_timeline_observed(
            _move_tensor(navsim_batch[6], device=device, name="ego_dynamics"),
            length=num_observed_frames,
            name="ego_dynamics",
        ),
        proposal_context_frames=proposal_context_frames,
        metadata=safe_metadata,
    )


def build_navsim_cvoi_model_batch(
    navsim_batch: Sequence[object],
    *,
    config: Any,
    device: torch.device,
) -> NavSimCvoiModelBatch:
    """Expose the geometry-free, observed-only model boundary to evaluation."""

    return _model_batch(navsim_batch, config=config, device=device)


def _single_metadata_entry(metadata: Mapping[str, object], name: str) -> object:
    value = metadata.get(name)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 1:
        raise ValueError(f"World4Drive metadata[{name!r}] must contain exactly one entry")
    return value[0]


def prepare_world4drive_deployment_input(
    navsim_batch: Sequence[object],
    *,
    runtime: object,
) -> PreparedNavSimCvoiDeployment:
    """Validate one Real cohort sample and prepare only observed model inputs plus its Planner seed."""

    if isinstance(navsim_batch, (str, bytes)) or not isinstance(navsim_batch, Sequence) or len(navsim_batch) != 12:
        raise ValueError("World4Drive latency requires the standard 12-element NavSim batch")
    states = navsim_batch[2]
    if not isinstance(states, torch.Tensor) or states.ndim != 3 or int(states.shape[0]) != 1:
        raise ValueError("World4Drive latency requires NavSim batch size 1")
    config = getattr(runtime, "config", None)
    device = getattr(runtime, "device", None)
    raw_poses = getattr(runtime, "num_planner_poses", None)
    if config is None or device is None or type(raw_poses) is not int or raw_poses <= 0:
        raise TypeError("World4Drive runtime must expose config, device, and positive num_planner_poses")
    if int(getattr(getattr(config, "cvoi", None), "controller_batch_size", -1)) != 1:
        raise ValueError("World4Drive latency requires cvoi.controller_batch_size=1")

    metadata = navsim_batch[11]
    if not isinstance(metadata, Mapping):
        raise ValueError("World4Drive latency requires metadata at batch element 11")
    if _single_metadata_entry(metadata, "dataset_domain") != "real":
        raise ValueError("World4Drive latency is Real-only")
    stable_sample_id = _single_metadata_entry(metadata, "stable_sample_id")
    if not isinstance(stable_sample_id, str) or not stable_sample_id.strip():
        raise ValueError("World4Drive stable_sample_id must contain one non-empty string")
    geometry_present = metadata.get("geometry_present")
    future_valid = metadata.get("future_agent_geometry_valid")
    if (
        not isinstance(geometry_present, torch.Tensor)
        or geometry_present.dtype != torch.bool
        or geometry_present.shape != (1,)
        or not bool(geometry_present[0].item())
        or not isinstance(future_valid, torch.Tensor)
        or future_valid.dtype != torch.bool
        or future_valid.shape != (1,)
        or not bool(future_valid[0].item())
    ):
        raise ValueError("World4Drive latency requires valid future agent geometry")
    if _single_metadata_entry(metadata, "agent_geometry_truncated") is not False:
        raise ValueError("World4Drive latency rejects truncated agent geometry")

    model_batch = build_navsim_cvoi_model_batch(
        navsim_batch,
        config=config,
        device=torch.device(device),
    )
    planner_seed = cvoi_sample_seed(resolve_cvoi_evaluation_seed(config), stable_sample_id)
    return PreparedNavSimCvoiDeployment(
        features={"model_batch": model_batch, "planner_seed": planner_seed},
        raw_poses=raw_poses,
    )


def _validate_online_planner_output(
    trajectories: object,
    confidences: object,
    *,
    runtime: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    poses = getattr(runtime, "num_planner_poses", None)
    if type(poses) is not int or poses <= 0:
        raise TypeError("World4Drive runtime must expose positive num_planner_poses")
    if (
        not isinstance(trajectories, torch.Tensor)
        or trajectories.ndim != 4
        or trajectories.shape[0] != 1
        or trajectories.shape[1] < 1
        or trajectories.shape[2:] != (poses, 3)
        or not trajectories.is_floating_point()
        or not bool(torch.isfinite(trajectories).all().item())
    ):
        raise ValueError(f"World4Drive Planner candidates must be finite [1,K,{poses},3]")
    if (
        not isinstance(confidences, torch.Tensor)
        or confidences.shape != trajectories.shape[:2]
        or confidences.device != trajectories.device
        or not confidences.is_floating_point()
        or not bool(torch.isfinite(confidences).all().item())
    ):
        raise ValueError("World4Drive Planner confidences must be finite [1,K] on the candidate device")
    return trajectories, confidences


def _elapsed_ms(start: float, end: float, *, name: str) -> float:
    latency_ms = (float(end) - float(start)) * 1000.0
    if not math.isfinite(latency_ms) or latency_ms < 0.0:
        raise RuntimeError(f"World4Drive {name} latency must be finite and non-negative")
    return latency_ms


def run_world4drive_sequential_deployment(
    prepared: PreparedNavSimCvoiDeployment,
    *,
    runtime: object,
    gate: torch.nn.Module,
    max_horizon: int,
    compute_costs: tuple[float, ...],
    controller_lineage: str,
    lambda_compute: float,
    guidance_steps: Optional[int],
    forced_horizon: Optional[int],
    synchronize: Optional[Callable[[], None]] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CvoiDeploymentOutput:
    """Run one true incremental policy call with Planner outside the adaptive clock."""

    if not isinstance(prepared, PreparedNavSimCvoiDeployment):
        raise TypeError("prepared must be a PreparedNavSimCvoiDeployment")
    if set(prepared.features) != {"model_batch", "planner_seed"}:
        raise ValueError("World4Drive prepared features must contain exactly model_batch and planner_seed")
    planner_seed = prepared.features["planner_seed"]
    if isinstance(planner_seed, bool) or not isinstance(planner_seed, int) or planner_seed < 0:
        raise ValueError("World4Drive prepared planner_seed must be a non-negative integer")
    if type(max_horizon) is not int or max_horizon != 3:
        raise ValueError("World4Drive incremental deployment requires max_horizon=3")
    if compute_costs != (0.0, 1.0, 2.0, 3.0):
        raise ValueError("World4Drive incremental deployment requires compute_costs=(0,1,2,3)")
    if controller_lineage not in {"p0_controller", "value_guided"}:
        raise ValueError("World4Drive controller_lineage must be p0_controller or value_guided")
    if isinstance(lambda_compute, bool) or not isinstance(lambda_compute, (int, float)):
        raise ValueError("World4Drive lambda_compute must be numeric")
    lambda_compute = float(lambda_compute)
    if not math.isfinite(lambda_compute) or lambda_compute != 0.05:
        raise ValueError("World4Drive incremental deployment requires lambda_compute=0.05")
    if guidance_steps is not None and (type(guidance_steps) is not int or guidance_steps not in {1, 2, 3, 4}):
        raise ValueError("World4Drive guidance_steps must be one of {1,2,3,4} or None")
    if forced_horizon is not None and (type(forced_horizon) is not int or forced_horizon not in {0, 1, 2, 3}):
        raise ValueError("World4Drive forced_horizon must be one of {0,1,2,3} or None")
    if controller_lineage == "p0_controller" and forced_horizon not in {None, 0}:
        raise ValueError("World4Drive P0 forced horizons above H0 are forbidden")
    if synchronize is None:
        device = torch.device(getattr(runtime, "device", "cpu"))

        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    if not callable(synchronize) or not callable(clock):
        raise TypeError("World4Drive synchronize and clock must be callable")
    if not isinstance(gate, torch.nn.Module):
        raise TypeError("World4Drive deployment gate must be a torch.nn.Module")

    policy = "p0" if controller_lineage == "p0_controller" else "p1"
    session = runtime.start_online_session(prepared.features["model_batch"], policy=policy)
    z_observed = getattr(session, "z_observed", None)
    if not isinstance(z_observed, torch.Tensor) or z_observed.ndim != 3 or z_observed.shape[0] != 1:
        raise ValueError("World4Drive online session must expose batch-one z_observed")
    raw_prefix = z_observed.new_empty((1, 0, z_observed.shape[-1]))

    if forced_horizon == 0:
        planner_prefix = raw_prefix
        trace: Mapping[str, object] | CvoiFixedComputeTrace = CvoiFixedComputeTrace(
            horizon=0,
            latency_ms=0.0,
            guidance_steps=0,
        )
    elif forced_horizon is not None:
        synchronize()
        adaptive_start = float(clock())
        for next_horizon in range(1, forced_horizon + 1):
            next_tokens = runtime.rollout_online_step(session, raw_prefix, next_horizon=next_horizon)
            raw_prefix = torch.cat([raw_prefix, next_tokens], dim=1)
        planner_prefix, diagnostics = runtime.prepare_online_terminal_prefix(
            session,
            raw_prefix,
            horizon=forced_horizon,
            controller_lineage=controller_lineage,
            guidance_steps=guidance_steps,
        )
        synchronize()
        adaptive_ms = _elapsed_ms(adaptive_start, float(clock()), name="adaptive region")
        actual_steps = diagnostics.get("guidance_steps") if isinstance(diagnostics, Mapping) else None
        expected_steps = 2 if guidance_steps is None else guidance_steps
        if (
            isinstance(actual_steps, bool)
            or not isinstance(actual_steps, (int, float))
            or not math.isfinite(float(actual_steps))
            or float(actual_steps) != float(expected_steps)
        ):
            raise ValueError("World4Drive forced Guidance diagnostics must report guidance_steps")
        trace = CvoiFixedComputeTrace(
            horizon=forced_horizon,
            latency_ms=adaptive_ms,
            guidance_steps=expected_steps,
        )
    else:
        guidance_diagnostics: dict[str, float] = {}

        def rollout_step(prefix: torch.Tensor, next_horizon: int) -> torch.Tensor:
            return runtime.rollout_online_step(session, prefix, next_horizon=next_horizon)

        def value_features(_observed: torch.Tensor, prefix: torch.Tensor, horizon: int) -> dict[str, torch.Tensor]:
            return runtime.online_value_features(
                session,
                prefix,
                horizon=horizon,
                controller_lineage=controller_lineage,
            )

        def stop_and_prepare(prefix: torch.Tensor, horizon: int, apply_guidance: bool) -> torch.Tensor:
            if apply_guidance != (horizon > 0):
                raise RuntimeError("World4Drive sequential rollout produced an inconsistent Guidance decision")
            guided, diagnostics = runtime.prepare_online_terminal_prefix(
                session,
                prefix,
                horizon=horizon,
                controller_lineage=controller_lineage,
                guidance_steps=guidance_steps,
            )
            if not isinstance(diagnostics, Mapping):
                raise TypeError("World4Drive Guidance diagnostics must be a mapping")
            guidance_diagnostics.update({name: float(value) for name, value in diagnostics.items()})
            return guided

        synchronize()
        adaptive_start = float(clock())
        rollout = run_sequential_rollout(
            observed_latent=z_observed,
            gate=gate,
            max_horizon=max_horizon,
            lambda_compute=lambda_compute,
            compute_costs=list(compute_costs),
            rollout_step=rollout_step,
            value_features=value_features,
            stop_and_plan=stop_and_prepare,
            gate_feature_mode="full",
            gate_feature_protocol="legacy_v1",
        )
        synchronize()
        adaptive_ms = _elapsed_ms(adaptive_start, float(clock()), name="adaptive region")
        rollout.require_finite_rollout_tokens()
        planner_prefix = rollout.planner_output
        if not isinstance(planner_prefix, torch.Tensor):
            raise TypeError("World4Drive sequential stop callback must return the terminal latent prefix")
        trace = {
            "stop_horizon": rollout.stop_horizon,
            "decisions": list(rollout.decisions),
            "predicted_deltas": list(rollout.predicted_deltas),
            "rollout_latency_ms": adaptive_ms,
            "guidance": dict(guidance_diagnostics),
        }

    trajectories, confidences = runtime.plan_online_terminal_prefix(
        session,
        planner_prefix,
        seed=planner_seed,
    )
    trajectories, confidences = _validate_online_planner_output(trajectories, confidences, runtime=runtime)
    if prepared.raw_poses != int(trajectories.shape[2]):
        raise ValueError("World4Drive prepared raw_poses disagrees with Planner output")
    return CvoiDeploymentOutput(
        pred_trajs=trajectories,
        confidences=confidences,
        controller_traces=(trace,),
    )


def _validate_world4drive_deployment_components(
    runtime: object,
    *,
    binding: CvoiWorld4DriveRuntimeBinding,
    gate: torch.nn.Module,
    synchronize: Optional[Callable[[], None]],
    clock: Callable[[], float],
) -> None:
    if not isinstance(binding, CvoiWorld4DriveRuntimeBinding):
        raise TypeError("World4Drive deployment binding must be CvoiWorld4DriveRuntimeBinding")
    if binding.protocol_version != "world4drive_evaluation_v1" or binding.stage != "evaluation":
        raise ValueError("World4Drive deployment binding must use the retained evaluation protocol")
    if binding.max_horizon != 3:
        raise ValueError("World4Drive deployment binding requires H=3")
    if binding.tokens_per_frame != 128:
        raise ValueError("World4Drive deployment binding requires tokens_per_frame=128")
    if binding.compute_costs != (0.0, 1.0, 2.0, 3.0):
        raise ValueError("World4Drive deployment binding requires compute_costs=(0,1,2,3)")
    if binding.controller_batch_size != 1:
        raise ValueError("World4Drive deployment binding requires controller_batch_size=1")
    if binding.lineage not in CVOI_WORLD4DRIVE_LINEAGE_ORDER:
        raise ValueError("World4Drive deployment binding has an unsupported lineage")
    expected_controller = "p0_controller" if binding.lineage == "p0_controller" else "value_guided"
    if binding.controller_lineage != expected_controller:
        raise ValueError("World4Drive deployment binding has an inconsistent Controller policy pair")
    if binding.guidance_steps != 2 or binding.guidance_objective != "last" or binding.timestep_sec != 0.5:
        raise ValueError("World4Drive deployment binding has an inconsistent Guidance or timestep contract")
    if binding.controller_lineage == "p0_controller":
        if binding.field_checkpoint is not None or binding.guided_planner_checkpoint is not None:
            raise ValueError("P0 World4Drive deployment binding forbids Field/P1 artifacts")
    elif binding.field_checkpoint is None or binding.guided_planner_checkpoint is None:
        raise ValueError("Value-guided World4Drive deployment binding requires Field/P1 artifacts")

    required = (
        "start_online_session",
        "rollout_online_step",
        "online_value_features",
        "prepare_online_terminal_prefix",
        "plan_online_terminal_prefix",
    )
    missing = [name for name in required if not callable(getattr(runtime, name, None))]
    if missing:
        raise TypeError(f"World4Drive online runtime is missing callable methods: {missing}")
    runtime_binding = getattr(getattr(runtime, "config", None), "_world4drive_runtime_binding", None)
    if runtime_binding != binding:
        raise ValueError("World4Drive runtime and deployment binding must match exactly")
    if getattr(runtime, "predictor_p0", None) is None or getattr(runtime, "planner_p0", None) is None:
        raise ValueError("World4Drive runtime requires the loaded P0 Predictor/Planner policy pair")
    predictor_p1 = getattr(runtime, "predictor_p1", None)
    planner_p1 = getattr(runtime, "planner_p1", None)
    if binding.controller_lineage == "p0_controller":
        if predictor_p1 is not None or planner_p1 is not None:
            raise ValueError("P0 World4Drive runtime must not expose a loaded P1 policy pair")
    elif predictor_p1 is None or planner_p1 is None:
        raise ValueError("Value-guided World4Drive runtime requires the loaded P1 Predictor/Planner policy pair")
    if getattr(runtime, "max_horizon", None) != binding.max_horizon:
        raise ValueError("World4Drive runtime max_horizon disagrees with binding")
    if getattr(runtime, "tokens_per_frame", None) != binding.tokens_per_frame:
        raise ValueError("World4Drive runtime tokens_per_frame disagrees with binding")
    runtime_embed_dim = getattr(runtime, "embed_dim", None)
    if type(runtime_embed_dim) is not int or runtime_embed_dim <= 0:
        raise ValueError("World4Drive runtime embed_dim must be a positive integer")

    if not isinstance(gate, torch.nn.Module):
        raise TypeError("World4Drive deployment gate must be a torch.nn.Module")
    if getattr(gate, _WORLD4DRIVE_GATE_BINDING_ATTR, None) != binding:
        raise ValueError("World4Drive Gate and deployment binding must match exactly")
    gate_latent_dim = getattr(gate, "latent_dim", None)
    if type(gate_latent_dim) is not int or gate_latent_dim != runtime_embed_dim:
        raise ValueError("World4Drive Gate latent_dim disagrees with runtime embed_dim")
    expected_feature_dim = 2 * runtime_embed_dim + 7
    gate_feature_dim = getattr(gate, "feature_dim", None)
    if type(gate_feature_dim) is not int or gate_feature_dim != expected_feature_dim:
        raise ValueError("World4Drive Gate feature_dim disagrees with the legacy runtime schema")
    if gate.training:
        raise ValueError("World4Drive deployment Gate must be in eval mode")
    if any(parameter.requires_grad for parameter in gate.parameters()):
        raise ValueError("World4Drive deployment Gate must be frozen")
    raw_runtime_device = getattr(runtime, "device", None)
    if raw_runtime_device is None:
        raise ValueError("World4Drive runtime must expose its device")
    runtime_device = torch.device(raw_runtime_device)
    if runtime_device.type == "cuda" and runtime_device.index is None:
        runtime_device = torch.device("cuda", torch.cuda.current_device())
    gate_tensors = tuple(gate.parameters()) + tuple(gate.buffers())
    if not gate_tensors:
        raise ValueError("World4Drive deployment Gate must expose device-bound state")
    if any(tensor.device != runtime_device for tensor in gate_tensors):
        raise ValueError("World4Drive deployment Gate must be on the runtime device")
    if synchronize is not None and not callable(synchronize):
        raise TypeError("World4Drive synchronize must be callable or None")
    if not callable(clock):
        raise TypeError("World4Drive clock must be callable")


def evaluate_world4drive_prepared(
    prepared: PreparedNavSimCvoiDeployment,
    *,
    runtime: object,
    binding: CvoiWorld4DriveRuntimeBinding,
    gate: torch.nn.Module,
    lambda_compute: float,
    guidance_steps: Optional[int] = None,
    forced_horizon: Optional[int] = None,
    synchronize: Optional[Callable[[], None]] = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CvoiDeploymentOutput:
    """Project an immutable binding onto one prepared incremental deployment call."""

    _validate_world4drive_deployment_components(
        runtime,
        binding=binding,
        gate=gate,
        synchronize=synchronize,
        clock=clock,
    )
    return run_world4drive_sequential_deployment(
        prepared,
        runtime=runtime,
        gate=gate,
        max_horizon=binding.max_horizon,
        compute_costs=binding.compute_costs,
        controller_lineage=binding.controller_lineage,
        lambda_compute=lambda_compute,
        guidance_steps=guidance_steps,
        forced_horizon=forced_horizon,
        synchronize=synchronize,
        clock=clock,
    )


class World4DriveRuntimeDeployment:
    """Immutable adapter from one read-only lineage runtime to online latency calls."""

    def __init__(
        self,
        runtime: object,
        *,
        binding: CvoiWorld4DriveRuntimeBinding,
        gate: torch.nn.Module,
        synchronize: Optional[Callable[[], None]] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        _validate_world4drive_deployment_components(
            runtime,
            binding=binding,
            gate=gate,
            synchronize=synchronize,
            clock=clock,
        )
        self.runtime = runtime
        self.binding = binding
        self.controller_lineage = binding.controller_lineage
        self.gate = gate
        self.synchronize = synchronize
        self.clock = clock

    def prepare(self, navsim_batch: Sequence[object]) -> PreparedNavSimCvoiDeployment:
        return prepare_world4drive_deployment_input(navsim_batch, runtime=self.runtime)

    def evaluate_prepared(
        self,
        prepared: PreparedNavSimCvoiDeployment,
        *,
        lambda_compute: float,
        guidance_steps: Optional[int] = None,
        forced_horizon: Optional[int] = None,
    ) -> CvoiDeploymentOutput:
        return run_world4drive_sequential_deployment(
            prepared,
            runtime=self.runtime,
            gate=self.gate,
            max_horizon=self.binding.max_horizon,
            compute_costs=self.binding.compute_costs,
            controller_lineage=self.binding.controller_lineage,
            lambda_compute=lambda_compute,
            guidance_steps=guidance_steps,
            forced_horizon=forced_horizon,
            synchronize=self.synchronize,
            clock=self.clock,
        )


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


def _raw_prefix(z_future: torch.Tensor, horizon: int, *, tokens_per_frame: Optional[int]) -> torch.Tensor:
    horizon = int(horizon)
    prefix_count = _prefix_count(z_future, tokens_per_frame=tokens_per_frame)
    if horizon < 0 or horizon > prefix_count:
        raise ValueError(f"CVoI raw prefix horizon must be in [0, {prefix_count}], got {horizon}")
    if z_future.ndim == 4:
        return z_future[:, :horizon]
    return z_future[:, : horizon * int(tokens_per_frame)]


def navsim_cvoi_raw_prefix(
    z_future: torch.Tensor,
    horizon: int,
    *,
    tokens_per_frame: Optional[int],
) -> torch.Tensor:
    """Return the raw predictor prefix corresponding to one forced horizon."""

    return _raw_prefix(z_future, horizon, tokens_per_frame=tokens_per_frame)


def _validate_encoded_batch(
    encoded: object,
    *,
    batch_size: int,
    embed_dim: int,
    max_horizon: int,
    tokens_per_frame: Optional[int],
) -> NavSimCvoiEncodedBatch:
    if not isinstance(encoded, NavSimCvoiEncodedBatch):
        raise TypeError("NavSim CVoI runtime encode_batch must return NavSimCvoiEncodedBatch")
    for name, value in (("z_observed", encoded.z_observed), ("z_future", encoded.z_future)):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim not in (3, 4)
            or value.shape[0] != batch_size
            or value.shape[-1] != embed_dim
            or not value.is_floating_point()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(
                f"NavSim CVoI runtime {name} must be finite floating [B,N,{embed_dim}] or [B,F,T,{embed_dim}]"
            )
    prefix_count = _prefix_count(encoded.z_future, tokens_per_frame=tokens_per_frame)
    if prefix_count != max_horizon:
        raise ValueError(
            f"NavSim CVoI runtime must produce exactly H={max_horizon} future prefixes, got {prefix_count}"
        )
    if len(encoded.model_contexts) != batch_size:
        raise ValueError("NavSim CVoI runtime model_contexts must contain one entry per sample")
    return encoded


def validate_navsim_cvoi_encoded_batch(
    encoded: object,
    *,
    batch_size: int,
    embed_dim: int,
    max_horizon: int,
    tokens_per_frame: Optional[int],
) -> NavSimCvoiEncodedBatch:
    """Validate the frozen runtime output before any policy evaluation."""

    return _validate_encoded_batch(
        encoded,
        batch_size=batch_size,
        embed_dim=embed_dim,
        max_horizon=max_horizon,
        tokens_per_frame=tokens_per_frame,
    )


__all__ = (
    "WORLD4DRIVE_RUNTIME_SIGNATURE_FIELDS",
    "CvoiControllerTrace",
    "CvoiDeploymentOutput",
    "CvoiFixedComputeTrace",
    "CvoiPlannerEvaluation",
    "CvoiWorld4DriveRuntimeBinding",
    "NavSimCvoiEncodedBatch",
    "NavSimCvoiModelBatch",
    "PreparedNavSimCvoiDeployment",
    "World4DriveRuntimeDeployment",
    "build_cvoi_world4drive_runtime_binding",
    "build_navsim_cvoi_model_batch",
    "build_world4drive_evaluation_runtime_signature_payload",
    "build_world4drive_gate_provenance",
    "evaluate_world4drive_prepared",
    "load_cvoi_world4drive_gate",
    "load_cvoi_world4drive_value_model",
    "navsim_cvoi_raw_prefix",
    "preflight_world4drive_direct_semantics",
    "prepare_world4drive_deployment_input",
    "read_world4drive_value_checkpoint",
    "read_world4drive_planner_runtime_signature",
    "require_world4drive_inputs_present",
    "require_world4drive_outputs_absent",
    "run_world4drive_sequential_deployment",
    "validate_cvoi_world4drive_gate",
    "validate_world4drive_base_data_root",
    "validate_world4drive_direct_inputs",
    "validate_world4drive_value_lineage",
    "validate_navsim_cvoi_encoded_batch",
)
