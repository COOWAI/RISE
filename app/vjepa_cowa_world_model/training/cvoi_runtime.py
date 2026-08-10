"""Manual NavSim-e120 and direct World4Drive runtime helpers."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training import cvoi_value as cvoi_value_module
from app.vjepa_cowa_world_model.training.configs.common import (
    resolve_main_encoder_frame_stride,
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
)
from app.vjepa_cowa_world_model.training.configs.parse import TrainingConfig, parse_training_config
from app.vjepa_cowa_world_model.training.cvoi_gate_pipeline import build_navtrain_gate_checkpoint_provenance
from app.vjepa_cowa_world_model.training.cvoi_manual_lineage import (
    resolve_cvoi_manual_ablation_results_root_from_config,
    resolve_cvoi_manual_full_results_root,
    resolve_cvoi_manual_runtime_value_input,
)
from app.vjepa_cowa_world_model.training.cvoi_navsim_navtrain_gate_store import open_embedded_oracle_store_v2
from app.vjepa_cowa_world_model.training.latent_value_guidance import apply_cvoi_latent_value_guidance
from app.vjepa_cowa_world_model.training.planner_anchor import CVOI_EGO_RELATIVE_ANCHOR_PROTOCOL
from app.vjepa_cowa_world_model.training.sequential_gate_training import load_sequential_gate_checkpoint

_PLANNER_STAGES = frozenset({"unguided_planner", "guided_planner"})
_CVOI_DIFFUSION_PLANNER_POLICY_SCHEMA = "cvoi_diffusion_planner_policy_v1"
_GUIDANCE_STAGES = frozenset({"guided_planner", "stop_calibrated", "evaluation"})
_NAVSIM_E120_H4V3_NO_VALUE_STAGES = frozenset(
    {
        "unguided_planner",
        "field_warmup",
        "field_calibrated",
        "gate_distillation",
    }
)
_NAVSIM_E120_H4V3_VALUE_STAGE_INPUT = {
    "guided_planner": ("field_checkpoint", "field_calibrated"),
    "stop_calibrated": ("field_checkpoint", "field_calibrated"),
}
_NAVSIM_E120_H4V3_EVALUATION_INPUT = {
    "controller": ("dual_value_checkpoint", "stop_calibrated"),
    "p1_field_forced": ("field_checkpoint", "field_calibrated"),
    "p0_forced": None,
}
_NAVSIM_E120_H4V3_STAGES = (
    _NAVSIM_E120_H4V3_NO_VALUE_STAGES | frozenset(_NAVSIM_E120_H4V3_VALUE_STAGE_INPUT) | frozenset({"evaluation"})
)
_WORLD4DRIVE_STRIPPED_TOP_LEVEL = frozenset({"cvoi_world4drive"})
_WORLD4DRIVE_FORBIDDEN_TOP_LEVEL = frozenset(
    {
        "cvoi",
        "dag",
        "tasks",
        "placement",
    }
)
_WORLD4DRIVE_FORBIDDEN_TOP_LEVEL_PREFIXES = ("cvoi_formal_v2_",)


def parse_world4drive_base_model_config(args: Mapping[str, object]) -> TrainingConfig:
    """Parse only the ordinary model/data sections of a direct World4Drive YAML."""

    if not isinstance(args, Mapping):
        raise TypeError("World4Drive direct config must be a mapping")
    forbidden = sorted(
        str(key)
        for key in args
        if key in _WORLD4DRIVE_FORBIDDEN_TOP_LEVEL
        or (
            isinstance(key, str)
            and any(key.startswith(prefix) for prefix in _WORLD4DRIVE_FORBIDDEN_TOP_LEVEL_PREFIXES)
        )
    )
    if forbidden:
        raise ValueError(f"World4Drive direct config contains forbidden training/governance sections: {forbidden}")
    model_args = {
        key: copy.deepcopy(value) for key, value in args.items() if key not in _WORLD4DRIVE_STRIPPED_TOP_LEVEL
    }
    config = parse_training_config(model_args, _allow_evaluation_value_guidance=True)
    if config.cvoi.enabled:
        raise ValueError("World4Drive direct base config must not enable the training CVoI protocol")
    return config


def cvoi_enabled(config: Any) -> bool:
    return bool(getattr(getattr(config, "cvoi", None), "enabled", False))


def cvoi_guidance_enabled(config: Any) -> bool:
    if not cvoi_enabled(config):
        return False
    return (
        str(config.cvoi.stage) in _GUIDANCE_STAGES
        and getattr(config.cvoi, "controller_lineage", "value_guided") == "value_guided"
    )


def resolve_cvoi_training_rollout_horizon(config: Any, *, total_future_steps: int) -> int:
    """Cap Planner-training imagination at Controller H without shortening its trajectory head."""

    total = int(total_future_steps)
    if total < 1:
        raise ValueError(f"training rollout requires at least one future step, got {total_future_steps}")
    if not cvoi_enabled(config):
        return total
    horizon = int(config.cvoi.max_horizon)
    if total < horizon:
        raise ValueError(f"CVoI training rollout requires at least H={horizon} future steps, got {total}")
    return horizon


def resolve_cvoi_validation_rollout_horizon(config: Any) -> Optional[int]:
    """Select Planner checkpoints at the deployed controller-full horizon h=H."""

    if not cvoi_enabled(config) or str(config.cvoi.stage) not in _PLANNER_STAGES:
        return None
    horizon = int(config.cvoi.max_horizon)
    if horizon <= 0:
        raise ValueError(f"CVoI validation max_horizon must be positive, got {horizon}")
    return horizon


def require_cvoi_stage_for_entry(config: Any, *, entry: str, allowed_stages: Sequence[str]) -> None:
    """Prevent an enabled CVoI stage from silently running through the wrong entry."""

    if not cvoi_enabled(config):
        return
    allowed = frozenset(str(stage) for stage in allowed_stages)
    stage = str(config.cvoi.stage)
    if stage not in allowed:
        raise ValueError(f"{entry} cannot execute cvoi.stage={stage!r}; allowed stages are {sorted(allowed)}")


def require_cvoi_planner_stage(config: Any) -> None:
    require_cvoi_stage_for_entry(
        config,
        entry="train_predictor_rollout_planner",
        allowed_stages=_PLANNER_STAGES,
    )


def validate_cvoi_sequential_runtime_config(config: Any) -> None:
    """Require the first-version incremental AR runtime used by the Gate."""

    if not cvoi_enabled(config) or str(config.cvoi.stage) != "evaluation":
        return
    if str(getattr(config.train, "predictor_type", "")).lower() != "ac_transformer":
        raise ValueError("CVoI sequential evaluation requires train.predictor_type='ac_transformer'")
    if bool(getattr(config.train, "use_parallel_predictor", False)):
        raise ValueError("CVoI sequential evaluation requires train.use_parallel_predictor=false")
    if not bool(getattr(config.train, "predictor_inference_consistent", False)):
        raise ValueError("CVoI sequential evaluation requires train.predictor_inference_consistent=true")
    if str(getattr(config.planner, "z_ar_mode", "full")) != "full":
        raise ValueError("CVoI sequential evaluation requires planner.z_ar_mode='full'")
    if not bool(getattr(config.planner, "use_planner", False)):
        raise ValueError("CVoI sequential evaluation requires planner.use_planner=true")
    if str(getattr(config.planner, "planner_type", "")).lower() != "diffusion":
        raise ValueError("CVoI sequential evaluation requires planner.planner_type='diffusion'")
    if bool(getattr(config.planner, "use_z_context", False)):
        raise ValueError("CVoI sequential evaluation requires planner.use_z_context=false so rollout affects policy")
    if not bool(getattr(config.planner, "use_observed_tokens", False)) and not bool(
        getattr(config.planner, "use_action_history_for_planner", False)
    ):
        raise ValueError("CVoI sequential evaluation h=0 requires observed tokens or action history")
    future_steps = int(config.data.num_target_frames) - int(config.train.num_observed_frames)
    if future_steps < int(config.cvoi.max_horizon):
        raise ValueError(
            f"CVoI sequential evaluation requires at least H={config.cvoi.max_horizon} future steps, "
            f"got {future_steps}"
        )


def _sha256_canonical_json(value: object) -> str:
    try:
        canonical = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("CVoI signature must be JSON serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cvoi_multiview_signature(config: Any) -> dict[str, object]:
    multiview = getattr(config, "multiview", None)
    enabled = bool(getattr(multiview, "enabled", False))
    if not enabled:
        return {"enabled": False}
    navsim = getattr(getattr(config, "data", None), "navsim", None)
    if navsim is None:
        raise ValueError("CVoI multiview signature requires data.navsim")
    global_camera_names = [str(value) for value in getattr(navsim, "camera_names", [])]
    if len(global_camera_names) < 2 or len(set(global_camera_names)) != len(global_camera_names):
        raise ValueError("CVoI multiview requires at least two unique ordered camera_names")
    roots = []
    seen_names = set()
    for root in [*getattr(navsim, "train_roots", []), *getattr(navsim, "val_roots", [])]:
        name = str(root.get("name", ""))
        if name in seen_names:
            continue
        seen_names.add(name)
        camera_names = [str(value) for value in root.get("camera_names", global_camera_names)]
        sensor_path = root.get("sensor_blobs_path")
        roots.append(
            {
                "name": name,
                "domain": str(root.get("domain", "")),
                "camera_names": camera_names,
                "sensor_blobs_path": (
                    None if sensor_path is None else str(Path(str(sensor_path)).expanduser().absolute())
                ),
            }
        )
    return {
        "enabled": True,
        "fusion_type": str(multiview.fusion_type),
        "output_mode": str(multiview.output_mode),
        "hidden_dim": int(multiview.hidden_dim),
        "num_heads": int(multiview.num_heads),
        "dropout": float(multiview.dropout),
        "camera_names": global_camera_names,
        "roots": roots,
    }


def _cvoi_planner_signature(config: Any) -> dict[str, object]:
    """Bind every diffusion setting that can change candidates or confidence selection."""

    planner = getattr(config, "planner", None)
    if planner is None:
        raise ValueError("CVoI runtime signature requires planner configuration")
    if str(getattr(planner, "planner_type", "")).lower() != "diffusion":
        raise ValueError("CVoI runtime signature requires planner.planner_type='diffusion'")
    fields = (
        "use_spatial_tokens",
        "use_temporal",
        "temporal_alignment",
        "z_ar_mode",
        "planner_input_source",
        "num_modes",
        "num_context_frames",
        "states_mode",
        "use_status_for_planner",
        "use_states_for_planner",
        "use_z_context",
        "observed_token_mode",
        "use_observed_tokens",
        "use_action_history_for_planner",
        "action_history_dim",
        "latent_dit_action_source",
        "policy_output_source",
        "planner_type",
        "diff_hidden_dim",
        "diff_num_layers",
        "diff_num_heads",
        "diff_dropout",
        "diff_mlp_ratio",
        "diff_sde_beta_min",
        "diff_sde_beta_max",
        "diff_inference_steps",
        "diff_num_samples",
        "diff_traj_dim",
        "diff_dt",
        "diff_trajectory_token_mode",
        "diff_adaln_version",
        "diff_use_last_frame_only",
        "diff_interleave_predictor_sampling",
        "diff_num_modes",
        "diff_independent_modes",
        "diff_mode_token_expansion",
        "diff_use_anchor_frame",
        "diff_init_traj_strategy",
        "diff_init_traj_noise_scale",
        "diff_init_traj_yaw_span_deg",
        "diff_init_traj_speed_scale_span",
        "diff_generation_framework",
        "diff_flow_matching_variant",
        "diff_flow_shift",
        "diff_flow_sampler",
        "diff_conf_temperature",
        "split_status_embedding",
        "use_drive_command",
        "status_dim",
    )
    missing = [name for name in fields if not hasattr(planner, name)]
    if missing:
        raise ValueError(f"CVoI planner configuration is missing signed policy fields: {missing}")
    signature = {
        "schema": _CVOI_DIFFUSION_PLANNER_POLICY_SCHEMA,
        "anchor_protocol": CVOI_EGO_RELATIVE_ANCHOR_PROTOCOL,
        "checkpoint_selection_rollout_horizon": int(config.cvoi.max_horizon),
    }
    signature.update({name: getattr(planner, name) for name in fields})
    _sha256_canonical_json(signature)
    return signature


def _signed_section(section: Any, *, section_name: str, fields: Sequence[str]) -> dict[str, object]:
    if section is None:
        raise ValueError(f"CVoI world execution signature requires {section_name} configuration")
    missing = [name for name in fields if not hasattr(section, name)]
    if missing:
        raise ValueError(f"CVoI {section_name} configuration is missing signed fields: {missing}")
    return {name: getattr(section, name) for name in fields}


def _cvoi_world_execution_signature(config: Any) -> dict[str, object]:
    """Bind config-only switches that alter frozen encoder/predictor forward semantics."""

    model = _signed_section(
        getattr(config, "model", None),
        section_name="model",
        fields=(
            "backbone",
            "vjepa_resolution",
            "vjepa_crop_top_bottom",
            "vjepa_num_frames",
            "vjepa_checkpoint_key",
            "vjepa_use_grid_mask",
            "vjepa_use_causal_attention",
            "patch_size",
            "pred_depth",
            "pred_num_heads",
            "pred_embed_dim",
            "pred_is_frame_causal",
            "uniform_power",
            "use_rope",
            "use_silu",
            "use_pred_silu",
            "wide_silu",
            "use_extrinsics",
            "use_mask_tokens",
            "zero_init_mask_tokens",
        ),
    )
    train = _signed_section(
        getattr(config, "train", None),
        section_name="train",
        fields=(
            "predictor_type",
            "use_parallel_predictor",
            "predictor_inference_consistent",
            "predictor_aux_policy",
            "predictor_no_aux_input",
            "use_states_for_predictor",
            "action_dim",
            "state_dim",
            "command_dim",
            "use_drive_command",
            "num_encoder_frames",
            "num_observed_frames",
        ),
    )
    data = _signed_section(
        getattr(config, "data", None),
        section_name="data",
        fields=(
            "fps",
            "num_target_frames",
            "crop_size",
            "patch_size",
            "tubelet_size",
            "use_tubelet_repeat",
        ),
    )
    meta = _signed_section(
        getattr(config, "meta", None),
        section_name="meta",
        fields=("use_sdpa", "deterministic"),
    )
    token_ae_config = getattr(config, "token_ae", None)
    if token_ae_config is None or not hasattr(token_ae_config, "enabled"):
        raise ValueError("CVoI world execution signature requires token_ae.enabled")
    token_ae: dict[str, object] = {"enabled": bool(token_ae_config.enabled)}
    if token_ae["enabled"]:
        token_ae.update(
            _signed_section(
                token_ae_config,
                section_name="token_ae",
                fields=(
                    "num_latent_tokens",
                    "num_heads",
                    "encoder_depth",
                    "decoder_depth",
                    "mlp_ratio",
                    "dropout",
                    "encoder_mode",
                    "pos_embed_type",
                    "input_grid_size",
                    "latent_grid_size",
                    "temporal_depth",
                    "temporal_num_heads",
                    "temporal_mlp_ratio",
                    "temporal_causal",
                    "temporal_mode",
                    "temporal_pos_embed_type",
                    "input_frame_mode",
                ),
            )
        )
    signature = {
        "schema": "cvoi_world_execution_v1",
        "meta": meta,
        "model": model,
        "train": train,
        "data": data,
        "token_ae": token_ae,
        "timeline": {
            "frame_stride": resolve_main_encoder_frame_stride(config),
            "observed_steps": resolve_main_encoder_num_observed_steps(config),
            "total_steps": resolve_main_encoder_num_time_steps(
                config,
                num_raw_frames=int(config.data.num_target_frames),
            ),
        },
        "future_aux_policy": "zero_after_observed_v1",
    }
    _sha256_canonical_json(signature)
    return signature


def resolve_cvoi_planner_checkpoint_paths(
    config: Any,
    *,
    legacy_latest_path: str,
    legacy_resume_path: Optional[str],
) -> tuple[str, Optional[str]]:
    """Make ``cvoi.output_checkpoint`` authoritative for planner-stage saves."""

    if not cvoi_enabled(config):
        return legacy_latest_path, legacy_resume_path
    require_cvoi_planner_stage(config)
    output_path = getattr(config.cvoi, "output_checkpoint", None)
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError(f"cvoi.stage={config.cvoi.stage!r} requires cvoi.output_checkpoint")
    if getattr(config.meta, "resume_checkpoint", None) is not None:
        if legacy_resume_path is None:
            raise RuntimeError("explicit meta.resume_checkpoint did not resolve to a checkpoint")
        if Path(legacy_resume_path).resolve() != Path(output_path).resolve():
            raise ValueError(
                "CVoI planner resume must point to cvoi.output_checkpoint; "
                f"got resume={legacy_resume_path!r}, output={output_path!r}"
            )
        resume_path = legacy_resume_path
    else:
        resume_path = output_path if Path(output_path).is_file() else None
    return output_path, resume_path


def validate_cvoi_planner_lineage(
    checkpoint: Mapping[str, object],
    *,
    expected_stage: str,
) -> Mapping[str, object]:
    signature = checkpoint.get("cvoi_runtime_signature")
    if not isinstance(signature, Mapping):
        raise ValueError("CVoI planner checkpoint is missing cvoi_runtime_signature")
    expected = {
        "schema": "cvoi_dual_value_v1",
        "stage": expected_stage,
        "guidance_steps": 2,
        "guidance_objective": "last",
    }
    mismatched = {key: (signature.get(key), value) for key, value in expected.items() if signature.get(key) != value}
    if mismatched:
        raise ValueError(f"CVoI planner checkpoint lineage mismatch: {mismatched}")
    if expected_stage == "unguided_planner" and signature.get("dual_value_sha256") is not None:
        raise ValueError("unguided CVoI planner checkpoint must not reference a dual-value artifact")
    if expected_stage == "guided_planner":
        value_hash = signature.get("dual_value_sha256")
        if not isinstance(value_hash, str) or len(value_hash) != 64:
            raise ValueError("guided CVoI planner checkpoint must record the field checkpoint SHA-256")
    return signature


def _load_navsim_e120_h4v3_direct_value_model(
    config: Any,
    *,
    embed_dim: int,
    device: torch.device,
) -> Optional[PrefixDualValueModel]:
    """Restore the exact branch-local handoff required by one h4v3 stage."""

    stage = getattr(config.cvoi, "stage", None)
    if type(stage) is not str or stage not in _NAVSIM_E120_H4V3_STAGES:
        raise ValueError(f"h4v3 cvoi.stage must be one of {sorted(_NAVSIM_E120_H4V3_STAGES)!r}, got {stage!r}")
    evaluation_mode = getattr(config.cvoi, "evaluation_mode", "controller")
    if stage == "evaluation" and (
        type(evaluation_mode) is not str or evaluation_mode not in _NAVSIM_E120_H4V3_EVALUATION_INPUT
    ):
        raise ValueError(
            "h4v3 evaluation cvoi.evaluation_mode must be one of "
            f"{sorted(_NAVSIM_E120_H4V3_EVALUATION_INPUT)!r}, got {evaluation_mode!r}"
        )
    if stage != "evaluation" and (evaluation_mode != "controller" or type(evaluation_mode) is not str):
        raise ValueError(
            f"h4v3 cvoi.stage={stage!r} requires cvoi.evaluation_mode='controller', " f"got {evaluation_mode!r}"
        )
    if stage in _NAVSIM_E120_H4V3_NO_VALUE_STAGES:
        return None
    if stage == "evaluation" and evaluation_mode == "p0_forced":
        return None
    signature = getattr(config.cvoi, "ablation_signature", None)
    if signature is None:
        raise ValueError("h4v3 Value loading requires cvoi.ablation_signature")
    experiment_role = getattr(signature, "experiment_role", None)
    value_path_field = (
        _NAVSIM_E120_H4V3_EVALUATION_INPUT[evaluation_mode][0]
        if stage == "evaluation"
        else _NAVSIM_E120_H4V3_VALUE_STAGE_INPUT[stage][0]
    )
    value_handoff_name = {
        "field_checkpoint": "calibration_handoff",
        "dual_value_checkpoint": "stop_handoff",
    }[value_path_field]
    if experiment_role == "main":
        full_results_root = resolve_cvoi_manual_full_results_root(
            {value_handoff_name: getattr(config.cvoi, value_path_field, None)}
        )
    elif experiment_role == "ablation":
        full_results_root = resolve_cvoi_manual_full_results_root(
            {"p0_handoff": getattr(config.cvoi, "unguided_planner_checkpoint", None)}
        )
    else:
        raise ValueError("h4v3 Value loading requires experiment_role to be 'main' or 'ablation'")
    ablation_results_root = (
        resolve_cvoi_manual_ablation_results_root_from_config(
            config.cvoi,
            artifact_fields=(value_path_field,),
        )
        if experiment_role == "ablation"
        else None
    )
    runtime_input = resolve_cvoi_manual_runtime_value_input(
        signature,
        configured_stage=stage,
        evaluation_mode=evaluation_mode,
        full_results_root=full_results_root,
        ablation_results_root=ablation_results_root,
    )
    if runtime_input is None:
        return None

    checkpoint_path = getattr(config.cvoi, runtime_input.path_field, None)
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError(f"cvoi.stage={stage!r} requires cvoi.{runtime_input.path_field}")
    expected_checkpoint_path = runtime_input.checkpoint_path
    if checkpoint_path != str(expected_checkpoint_path):
        raise ValueError(
            f"cvoi.stage={stage!r} must use fixed handoff path {str(expected_checkpoint_path)!r}, "
            f"got {checkpoint_path!r}"
        )

    with torch.random.fork_rng(devices=[]):
        payload = cvoi_value_module.read_cvoi_navsim_e120_direct_value_checkpoint(
            checkpoint_path,
            required_phase=runtime_input.required_phase,
            required_branch_id=runtime_input.required_branch_id,
            map_location="cpu",
        )
        architecture = payload["architecture"]
        stored_embed_dim = architecture["embed_dim"]
        if stored_embed_dim != int(embed_dim):
            raise ValueError(
                f"CVoI dual-value embed_dim must match encoder embed_dim={int(embed_dim)}, got {stored_embed_dim!r}"
            )
        model = PrefixDualValueModel(**architecture)
        model.load_state_dict(payload["state_dict"], strict=True)

    model.to(device=device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_cvoi_dual_value_model(
    config: Any,
    *,
    embed_dim: int,
    device: torch.device,
) -> Optional[PrefixDualValueModel]:
    """Load the exact manual NavSim-e120 lifecycle checkpoint for this stage."""

    if not cvoi_enabled(config):
        return None
    protocol_version = getattr(config.cvoi, "protocol_version", None)
    if protocol_version != "formal_v2_navsim_e120_h4_v3":
        raise ValueError(f"unsupported CVoI Value runtime protocol: {protocol_version!r}")
    return _load_navsim_e120_h4v3_direct_value_model(
        config,
        embed_dim=embed_dim,
        device=device,
    )


def build_cvoi_gate_provenance(config: Any) -> dict[str, str]:
    """Build provenance from the manual embedded NavTrain SQLite Oracle."""

    if not cvoi_enabled(config) or str(config.cvoi.stage) != "evaluation":
        raise ValueError('CVoI Gate provenance is defined only for cvoi.stage="evaluation"')
    if getattr(config.cvoi, "protocol_version", None) != "formal_v2_navsim_e120_h4_v3":
        raise ValueError("CVoI Gate loading supports only formal_v2_navsim_e120_h4_v3")
    oracle_path = getattr(config.cvoi, "oracle_path", None)
    if not isinstance(oracle_path, str) or not oracle_path.strip():
        raise ValueError('cvoi.stage="evaluation" requires cvoi.oracle_path')
    ablation_signature = getattr(config.cvoi, "ablation_signature", None)
    if ablation_signature is None:
        raise ValueError("manual NavSim-e120 Gate loading requires cvoi.ablation_signature")
    oracle_file = Path(oracle_path)
    with open_embedded_oracle_store_v2(oracle_file) as oracle:
        return build_navtrain_gate_checkpoint_provenance(
            oracle_file,
            oracle,
            gate_feature_mode=str(ablation_signature.gate_feature_mode),
        )


def load_cvoi_gate_for_evaluation(config: Any, *, device: torch.device) -> Optional[torch.nn.Module]:
    if not cvoi_enabled(config):
        return None
    if str(config.cvoi.stage) != "evaluation":
        return None
    evaluation_mode = str(getattr(config.cvoi, "evaluation_mode", "controller"))
    if evaluation_mode != "controller":
        return None
    checkpoint_path = getattr(config.cvoi, "gate_checkpoint", None)
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError("cvoi.stage='evaluation' requires cvoi.gate_checkpoint")
    return load_sequential_gate_checkpoint(
        checkpoint_path,
        device=device,
        expected_provenance=build_cvoi_gate_provenance(config),
        expected_protocol_version=getattr(config.cvoi, "protocol_version", "legacy_v1"),
    )


def apply_cvoi_planner_guidance(
    z_observed: torch.Tensor,
    z_future: torch.Tensor,
    dual_value_model: Optional[torch.nn.Module],
    *,
    tokens_per_frame: int,
    config: Any,
    evaluation_guidance_steps: Optional[int] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply fixed K=2 Guidance or preserve the unguided tensor exactly."""

    if evaluation_guidance_steps is not None and str(getattr(config.cvoi, "stage", "")) != "evaluation":
        raise ValueError("evaluation_guidance_steps is permitted only in the CVoI evaluation stage")

    if not cvoi_guidance_enabled(config):
        return z_future, {
            "guidance_steps": 0.0,
            "guidance_skipped_h0": float(z_future.shape[1] == 0),
            "delta_norm": 0.0,
            "field_value_before": 0.0,
            "field_value_after": 0.0,
        }
    if z_future.ndim != 3 or int(z_future.shape[1]) % int(tokens_per_frame) != 0:
        raise ValueError("CVoI Guidance future latents must be [B,F*tokens_per_frame,D]")
    max_future_tokens = int(config.cvoi.max_horizon) * int(tokens_per_frame)
    if int(z_future.shape[1]) > max_future_tokens:
        z_future = z_future[:, :max_future_tokens]
    if dual_value_model is None:
        raise ValueError("enabled CVoI Guidance requires a loaded PrefixDualValueModel")
    return apply_cvoi_latent_value_guidance(
        z_observed,
        z_future,
        dual_value_model,
        tokens_per_frame=tokens_per_frame,
        config=config,
        evaluation_guidance_steps=evaluation_guidance_steps,
    )
