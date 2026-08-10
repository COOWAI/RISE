"""Strict config contract for the NavSim-only Formal-v2 e120 Planner profile."""

from __future__ import annotations

import importlib
from copy import deepcopy
from dataclasses import fields

import pytest

from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_full_state_warmstart import (
    FORMAL_V2_E120_CHECKPOINT_PATH,
    FORMAL_V2_E120_PARAMS_PRETRAIN_PATH,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_config import (
    build_formal_v2_navsim_e120_public_config,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_E120_DEFAULT_LAMBDA,
    FORMAL_V2_NAVSIM_E120_LAMBDA_GRID,
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P0_POLICIES,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_roots import (
    FORMAL_V2_NAVSIM_DIRECT_PREFLIGHT_ROOT,
    build_formal_v2_navsim_direct_task_projection,
    build_formal_v2_navsim_root_catalog,
)

PROFILE = "formal_v2_navsim_e120_h4_v3"
ABLATION_SCHEMA = "cvoi_formal_v2_navsim_e120_ablation_v1"
CVOI_SCHEMA = "cvoi_dual_value_navsim_e120_v1"
PREFLIGHT_ROOT = str(FORMAL_V2_NAVSIM_DIRECT_PREFLIGHT_ROOT)
FORMAL_MAX_AGENTS = 1024
FULL_RESULTS_ROOT = "/path/to/rise/results/cvoi_manual_full"


def _ablation_signature(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": ABLATION_SCHEMA,
        "protocol_version": PROFILE,
        "experiment_role": "main",
        "branch_id": "p0_uniform",
        "shared_cohort_id": "navsim_e120_s239_stride4",
        "initialization_mode": "full_state_warmstart",
        "cf_field_supervision": "hazard_quality",
        "field_calibration_mode": "local_geometry",
        "p0_prefix_mode": "uniform",
        "gate_feature_mode": "full",
        "train_seed": 239,
        "evaluation_seed": 239,
        "training_stride": 4,
    }
    value.update(updates)
    return value


def _warmstart_binding(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema": "cvoi_full_state_warmstart_config_v1",
        "import_mode": "full_state_warmstart",
        "source_checkpoint": {
            "path": FORMAL_V2_E120_CHECKPOINT_PATH,
        },
        "source_params_pretrain": {
            "path": FORMAL_V2_E120_PARAMS_PRETRAIN_PATH,
        },
    }
    value.update(updates)
    return value


def _runtime_roots_by_role() -> dict[str, dict[str, object]]:
    projection = build_formal_v2_navsim_direct_task_projection(
        "field",
        "full",
        build_formal_v2_navsim_root_catalog(),
        PREFLIGHT_ROOT,
    )
    roots = [*projection["train_roots"], *projection["val_roots"]]
    roles_by_name = {
        "navsim_real_train": "real_train",
        "navsim_cf_train": "cf_train",
        "navsim_real_navtest": "real_navtest",
        "navsim_cf_val": "cf_val",
    }
    by_role: dict[str, dict[str, object]] = {}
    for source in roots:
        root = deepcopy(source)
        role = roles_by_name[root["name"]]
        root.update(
            {
                "annotation_selection": (
                    "trajectory_match_and_accident_type_allowlist"
                    if root["domain"] == "counterfactual"
                    else "all_valid"
                ),
                "load_agent_annotations": root["domain"] == "real",
                "max_agents": FORMAL_MAX_AGENTS,
                "max_scenes": None,
                "window_stride": 4,
            }
        )
        if root["domain"] == "counterfactual":
            root["window_start_policy"] = "counterfactual_scene_start"
        by_role[role] = root
    return by_role


def _real_root(role: str) -> dict[str, object]:
    return deepcopy(_runtime_roots_by_role()[role])


def _counterfactual_root(role: str = "cf_train") -> dict[str, object]:
    return deepcopy(_runtime_roots_by_role()[role])


def _apply_signed_public_compatibility(args: dict[str, object]) -> None:
    public = build_formal_v2_navsim_e120_public_config()
    compatibility = public["compatibility"]
    parser_defaults = public["parser_defaults"]
    preserved = public["preserved"]

    args["method"] = compatibility["method"]
    args["model"] = deepcopy(compatibility["model"])
    args["train"] = {
        **deepcopy(parser_defaults["train"]),
        **deepcopy(compatibility["predictor_inputs"]),
        **deepcopy(compatibility["training_state"]),
        "num_encoder_frames": 4,
    }
    args["planner"] = {
        **deepcopy(parser_defaults["planner"]),
        **deepcopy(compatibility["planner"]),
        "tf_d_model": 256,
        "tf_d_ffn": 1024,
        "tf_num_layers": 3,
        "tf_num_head": 8,
        "tf_dropout": 0.0,
        "enable_rl_actor_critic": False,
        "rl_action_dim": 2,
    }

    selection_epochs = args["meta"]["selection_checkpoint_epochs"]
    args["meta"].update(deepcopy(parser_defaults["meta"]))
    args["meta"].update(
        {
            "seed": compatibility["runtime"]["seed"],
            "dtype": compatibility["runtime"]["dtype"],
            "use_sdpa": compatibility["runtime"]["use_sdpa"],
            "context_encoder_key": compatibility["runtime"]["context_encoder_key"],
            "target_encoder_key": compatibility["runtime"]["target_encoder_key"],
            "save_every_freq": preserved["checkpoint_cadence"]["save_every_freq"],
            "val_freq": preserved["checkpoint_cadence"]["val_freq"],
            "selection_checkpoint_epochs": selection_epochs,
        }
    )
    data_compatibility = compatibility["data"]
    args["data"].update(
        {key: deepcopy(value) for key, value in data_compatibility.items() if not key.startswith("navsim_")}
    )
    args["data"]["navsim"].update(
        {
            "camera_name": data_compatibility["navsim_camera_name"],
            "image_require_policy": data_compatibility["navsim_image_require_policy"],
            "max_frame_gap": data_compatibility["navsim_max_frame_gap"],
        }
    )
    args["segmentation"] = {
        "use_segmentation": compatibility["runtime"]["segmentation_enabled"],
        "seg_loss_weight": compatibility["runtime"]["segmentation_loss_weight"],
    }
    args["token_ae"] = {"enabled": compatibility["runtime"]["token_ae_enabled"]}
    args["ema"] = deepcopy(preserved["ema"])
    args["loss"] = deepcopy(preserved["loss"])
    args["loss"]["auto_steps"] = None
    args["data_aug"] = deepcopy(preserved["data_aug"])

    stage_epochs = {
        "epochs": args["optimization"]["epochs"],
        "schedule_epochs": args["optimization"]["schedule_epochs"],
    }
    args["optimization"] = {
        **deepcopy(parser_defaults["optimization"]),
        **deepcopy(public["optimization"]["shared_adamw"]),
        **stage_epochs,
    }


def _planner_args(*, stage: str = "p0", prefix_mode: str = "uniform") -> dict[str, object]:
    if stage not in {"p0", "p1"}:
        raise ValueError(stage)
    outer_stage = "unguided_planner" if stage == "p0" else "guided_planner"
    epochs = 50 if stage == "p0" else 80
    selection_epochs = (
        FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS
        if stage == "p0" and prefix_mode == "uniform"
        else () if stage == "p0" else FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS
    )
    if stage == "p0":
        experiment_role = "main" if prefix_mode == "uniform" else "ablation"
        branch_id = f"p0_{prefix_mode}"
    else:
        experiment_role = "main"
        branch_id = "p1_full"
    signature = _ablation_signature(
        experiment_role=experiment_role,
        branch_id=branch_id,
        p0_prefix_mode=prefix_mode,
    )
    args = {
        "method": "lewm",
        "meta": {
            "seed": 239,
            "dtype": "bfloat16",
            "resume_checkpoint": None,
            "pretrain_repo": None,
            "pretrain_checkpoint": None,
            "pretrain_checkpoint_full": None,
            "predictor_checkpoint": None,
            "value_checkpoint": None,
            "planner_value_checkpoint": None,
            "ae_checkpoint": None,
            "load_encoder": False,
            "load_predictor": False,
            "load_planner": False,
            "load_seg": False,
            "context_encoder_key": "encoder",
            "target_encoder_key": "target_encoder",
            "save_every_freq": 10,
            "selection_checkpoint_epochs": list(selection_epochs),
            "use_sdpa": True,
            "val_freq": 5,
            "resume_model_only": False,
            "auto_resume_latest": False,
        },
        "data": {
            "batch_size": 16,
            "crop_size": 256,
            "dataset_fpcs": [12],
            "fps": 2,
            "num_target_frames": 12,
            "num_workers": 4,
            "patch_size": 16,
            "persistent_workers": True,
            "pin_mem": True,
            "tubelet_size": 2,
            "use_tubelet_repeat": True,
            "navsim": {
                "enabled": True,
                "camera_name": "CAM_F0",
                "image_require_policy": "observed_only",
                "max_frame_gap": 1,
                "max_agents": FORMAL_MAX_AGENTS,
                "max_scenes": None,
                "max_val_scenes": None,
                "window_stride": 4,
                "val_window_stride": 4,
                "balance_train_roots": False,
                "train_roots": [_real_root("real_train")],
                "val_roots": [_real_root("real_navtest")],
                "data_path": "",
                "sensor_blobs_path": "",
                "camera_names": ["CAM_F0"],
                "num_history_frames": None,
                "index_cache": True,
                "tail_seconds": None,
                "counterfactual_tail_seconds": None,
                "load_agent_annotations": True,
                "scene_filter_yaml": None,
                "pose_overlay_path": None,
                "pose_overlay_coord_frame": "opencv_first_frame",
                "pose_overlay_required": False,
            },
        },
        "train": {
            "encoder_train": False,
            "seg_head": False,
            "encoder_ema": False,
            "perceiver_ema": False,
            "predictor_train": False,
            "predictor_planner_finetune": True,
            "use_states_for_predictor": False,
            "action_dim": 3,
            "state_dim": 8,
            "use_drive_command": False,
            "predictor_inference_consistent": True,
            "predictor_aux_policy": "inference_consistent",
            "use_parallel_predictor": False,
            "predictor_supervision_mode": "tf_ar",
            "predictor_loss_scope": "future_only",
            "predictor_use_z_ar_supervision": True,
            "reuse_context_as_target_when_frozen": False,
            "predictor_no_aux_input": False,
            "num_encoder_frames": 4,
            "num_observed_frames": 4,
            "predictor_type": "ac_transformer",
        },
        "predictor_dynamic_rollout": {
            "enabled": True,
            "full_prefix_prob": FORMAL_V2_NAVSIM_P0_POLICIES[prefix_mode][-1],
            "min_prefix_steps": 0,
            "max_non_full_prefix_steps": 3,
            "max_horizon": 4,
            "horizon_probabilities": list(FORMAL_V2_NAVSIM_P0_POLICIES[prefix_mode]),
        },
        "planner": {
            "use_planner": True,
            "planner_loss_weight": 1,
            "z_ar_mode": "full",
            "planner_input_source": "z_ar",
            "num_modes": 6,
            "use_status_for_planner": True,
            "use_states_for_planner": True,
            "use_z_context": False,
            "observed_token_mode": "concat_type_embed",
            "use_action_history_for_planner": True,
            "action_history_dim": 3,
            "policy_output_source": "planner",
            "planner_type": "diffusion",
            "diff_hidden_dim": 384,
            "diff_num_layers": 12,
            "diff_num_heads": 12,
            "diff_inference_steps": 20,
            "diff_num_samples": 6,
            "diff_num_modes": 6,
            "diff_traj_dim": 4,
            "diff_dt": 0.5,
            "diff_trajectory_token_mode": "per_pose_token",
            "diff_adaln_version": "v2",
            "diff_use_last_frame_only": False,
            "diff_train_prefix_conditioning": False,
            "diff_use_anchor_frame": True,
            "split_status_embedding": False,
            "use_drive_command": False,
            "status_dim": 8,
            "enable_rl_actor_critic": False,
            "rl_action_dim": 2,
        },
        "segmentation": {"use_segmentation": False, "seg_loss_weight": 0.0},
        "token_ae": {"enabled": False},
        "value_guidance": {
            "enabled": stage == "p1",
            "steps": 2,
            "objective": "last",
            "step_size": 0.05,
            "max_delta_norm": 0.25,
            "detach_output": True,
        },
        "optimization": {
            "ipe": None,
            "optimizer": "adamw",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "lr": 2e-4,
            "start_lr": 2e-5,
            "final_lr": 0.0,
            "weight_decay": 0.04,
            "final_weight_decay": 0.04,
            "enc_lr_scale": 0.001,
            "predictor_lr_scale": 0.1,
            "warmup": 15,
            "anneal": 15,
            "epochs": epochs,
            "schedule_epochs": epochs,
        },
        "world_model": {
            "enabled": True,
            "sigreg_weight": 0.09,
            "sigreg_knots": 17,
            "sigreg_num_proj": 1024,
            "projector_hidden_dim": 2048,
            "embed_dim": 192,
            "num_subspaces": 1,
            "subspace_dim": None,
            "init_mode": "orthogonal_frozen",
            "theta": 0.0,
        },
        "rl": {"enabled": False},
        "cvoi": {
            "enabled": True,
            "max_agents": FORMAL_MAX_AGENTS,
            "protocol_version": PROFILE,
            "schema": CVOI_SCHEMA,
            "stage": outer_stage,
            "ablation_mode": "manual_ablation",
            "ablation_signature": signature,
            "full_state_warmstart": _warmstart_binding(),
            "guidance_steps": 2,
            "guidance_objective": "last",
            "controller_batch_size": 1,
            "max_horizon": 4,
            "rollout_horizons": [0, 1, 2, 3, 4],
            "compute_costs": [0.0, 1.0, 2.0, 3.0, 4.0],
            "lambda_compute": FORMAL_V2_NAVSIM_E120_DEFAULT_LAMBDA,
            "lambda_grid": list(FORMAL_V2_NAVSIM_E120_LAMBDA_GRID),
            "world_model_checkpoint": FORMAL_V2_E120_CHECKPOINT_PATH,
            "seed_planner_checkpoint": FORMAL_V2_E120_CHECKPOINT_PATH,
            "unguided_planner_checkpoint": (None if stage == "p0" else f"{FULL_RESULTS_ROOT}/handoff/p0_selected.pt"),
            "field_checkpoint": (None if stage == "p0" else f"{FULL_RESULTS_ROOT}/handoff/calibration.pt"),
            "output_checkpoint": (
                f"{FULL_RESULTS_ROOT}/p0/p0_planner_checkpoint.pt"
                if stage == "p0"
                else f"{FULL_RESULTS_ROOT}/p1/p1_planner_checkpoint.pt"
            ),
        },
    }
    _apply_signed_public_compatibility(args)
    return args


def _stage_args(
    stage: str,
    *,
    supervision: str = "hazard_quality",
    calibration: str = "local_geometry",
) -> dict[str, object]:
    if stage in {"unguided_planner", "guided_planner"}:
        return _planner_args(stage="p0" if stage == "unguided_planner" else "p1")
    if stage not in {
        "field_warmup",
        "field_calibrated",
        "stop_calibrated",
        "gate_distillation",
        "evaluation",
    }:
        raise ValueError(stage)

    args = _planner_args()
    cvoi = args["cvoi"]
    cvoi["stage"] = stage
    lineage_by_mechanism = {
        ("hazard_quality", "local_geometry"): "full",
        ("none", "local_geometry"): "no_cf",
        ("hazard_only", "local_geometry"): "hazard_only",
        ("quality_only", "local_geometry"): "quality_only",
    }
    branch_id = lineage_by_mechanism.get(
        (supervision, calibration),
        f"{stage}_{supervision}_{calibration}",
    )
    experiment_role = "main" if branch_id == "full" else "ablation"
    cvoi["ablation_signature"] = _ablation_signature(
        experiment_role=experiment_role,
        branch_id=branch_id,
        cf_field_supervision=supervision,
        field_calibration_mode=calibration,
    )
    for field_name in (
        "unguided_planner_checkpoint",
        "field_checkpoint",
        "guided_planner_checkpoint",
        "dual_value_checkpoint",
        "oracle_path",
        "gate_checkpoint",
        "output_checkpoint",
    ):
        cvoi[field_name] = None
    cvoi["gate_training_batch_size"] = 4096 if stage == "gate_distillation" else None

    args["meta"]["selection_checkpoint_epochs"] = []
    args["train"]["predictor_planner_finetune"] = False
    args["train"]["predictor_validation_enabled"] = False
    args["optimization"].update(
        {
            "ipe": 7,
            "optimizer": "adamw",
            "lr": 7.3e-4,
            "start_lr": 7.1e-5,
            "final_lr": 1e-6,
            "weight_decay": 0.031,
            "final_weight_decay": 0.029,
            "enc_lr_scale": 0.7,
            "predictor_lr_scale": 0.6,
            "warmup": 3,
            "anneal": 4,
            "epochs": 7,
            "schedule_epochs": 11,
        }
    )
    args["value_guidance"]["enabled"] = stage in {"stop_calibrated", "evaluation"}

    navsim = args["data"]["navsim"]
    navsim["balance_train_roots"] = False
    navsim["train_roots"] = [_real_root("real_train")]
    navsim["val_roots"] = [_real_root("real_navtest")]

    if stage == "field_warmup":
        cvoi.update(
            {
                "unguided_planner_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/p0_selected.pt",
                "output_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/field.pt",
                "offline_adapter_factory": (
                    "app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter:"
                    "create_navsim_cvoi_offline_adapter"
                ),
                "offline_runtime_factory": (
                    "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime:" "create_navsim_cvoi_model_runtime"
                ),
            }
        )
        strict_real_only = supervision == "none" and calibration == "local_geometry"
        cvoi["field_warmup_domain"] = "real" if strict_real_only else "real_cf"
        if not strict_real_only:
            navsim["balance_train_roots"] = True
            navsim["train_roots"].append(_counterfactual_root())
            navsim["val_roots"].append(_counterfactual_root("cf_val"))
        if supervision in {"hazard_quality", "hazard_only", "quality_only"}:
            args["validation_suite"] = {
                "enabled": True,
                "protocol_version": "dynamic_rollout_validation_navsim_h4_v3",
                "horizons": [0, 1, 2, 3, 4],
                "expected_weights": list(FORMAL_V2_NAVSIM_P0_POLICIES["uniform"]),
                "primary_domain": "real",
                "primary_cohort": "all",
                "primary_protocol": "full",
            }
    elif stage == "field_calibrated":
        cvoi.update(
            {
                "unguided_planner_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/p0_selected.pt",
                "field_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/field.pt",
                "output_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/calibration.pt",
                "offline_adapter_factory": (
                    "app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter:"
                    "create_navsim_cvoi_offline_adapter"
                ),
                "offline_runtime_factory": (
                    "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime:" "create_navsim_cvoi_model_runtime"
                ),
            }
        )
    elif stage == "stop_calibrated":
        cvoi.update(
            {
                "unguided_planner_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/p0_selected.pt",
                "field_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/calibration.pt",
                "guided_planner_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/p1_selected.pt",
                "output_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/stop.pt",
                "offline_adapter_factory": (
                    "app.vjepa_cowa_world_model.training.navsim_cvoi_offline_adapter:"
                    "create_navsim_cvoi_offline_adapter"
                ),
                "offline_runtime_factory": (
                    "app.vjepa_cowa_world_model.training.navsim_cvoi_model_runtime:" "create_navsim_cvoi_model_runtime"
                ),
            }
        )
    elif stage == "gate_distillation":
        for field_name in ("world_model_checkpoint", "seed_planner_checkpoint"):
            cvoi[field_name] = None
        cvoi.pop("full_state_warmstart")
        cvoi.update(
            {
                "oracle_path": f"{FULL_RESULTS_ROOT}/handoff/oracle_full.sqlite3",
                "output_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/gate.pt",
            }
        )
        args["data"].pop("navsim")
        args["value_guidance"]["enabled"] = False
    else:
        cvoi.update(
            {
                "unguided_planner_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/p0_selected.pt",
                "field_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/calibration.pt",
                "guided_planner_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/p1_selected.pt",
                "dual_value_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/stop.pt",
                "oracle_path": f"{FULL_RESULTS_ROOT}/handoff/oracle_full.sqlite3",
                "gate_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/gate.pt",
            }
        )
        navsim["train_roots"] = []

    return args


def _direct_stage_args(stage: str) -> dict[str, object]:
    """Return one proof-free stage config for the manual Full training chain."""

    return _stage_args(stage)


def _direct_evaluation_args() -> dict[str, object]:
    """Return the proof-free P1-forced evaluation config."""

    args = _direct_stage_args("evaluation")
    cvoi = args["cvoi"]
    cvoi.update(
        {
            "evaluation_mode": "p1_field_forced",
            "controller_lineage": "value_guided",
            "world_model_checkpoint": FORMAL_V2_E120_CHECKPOINT_PATH,
            "seed_planner_checkpoint": FORMAL_V2_E120_CHECKPOINT_PATH,
            "unguided_planner_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/p0_selected.pt",
            "field_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/calibration.pt",
            "guided_planner_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/p1_selected.pt",
            "dual_value_checkpoint": f"{FULL_RESULTS_ROOT}/handoff/stop.pt",
            "oracle_path": None,
            "gate_checkpoint": None,
            "output_checkpoint": None,
            "token_ae_checkpoint": None,
        }
    )
    args["data"]["navsim"]["train_roots"] = [_real_root("real_train")]
    return args


def _different_signed_value(value: object) -> object:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        return value + 0.125
    if type(value) is str:
        return f"{value}_drift"
    if value is None:
        return "drift"
    if type(value) is list:
        return [*deepcopy(value), "drift"]
    raise TypeError(type(value).__name__)


def _signed_public_mutation_cases() -> list[tuple[str, str, object]]:
    public = build_formal_v2_navsim_e120_public_config()
    compatibility = public["compatibility"]
    parser_defaults = public["parser_defaults"]
    preserved = public["preserved"]
    cases: list[tuple[str, str, object]] = [
        ("top", "method", _different_signed_value(compatibility["method"])),
    ]
    for source in (compatibility["model"], parser_defaults["model"]):
        cases.extend(("model", key, _different_signed_value(value)) for key, value in source.items())
    for source in (
        compatibility["predictor_inputs"],
        compatibility["training_state"],
        parser_defaults["train"],
    ):
        cases.extend(("train", key, _different_signed_value(value)) for key, value in source.items())
    for source in (compatibility["planner"], parser_defaults["planner"]):
        cases.extend(("planner", key, _different_signed_value(value)) for key, value in source.items())
    for key, value in compatibility["data"].items():
        if key.startswith("navsim_"):
            cases.append(("navsim", key.removeprefix("navsim_"), _different_signed_value(value)))
        else:
            cases.append(("data", key, _different_signed_value(value)))
    runtime_meta_fields = {
        key: compatibility["runtime"][key]
        for key in ("dtype", "seed", "use_sdpa", "context_encoder_key", "target_encoder_key")
    }
    for source in (runtime_meta_fields, parser_defaults["meta"], preserved["checkpoint_cadence"]):
        cases.extend(("meta", key, _different_signed_value(value)) for key, value in source.items())
    cases.extend(
        [
            (
                "segmentation",
                "use_segmentation",
                _different_signed_value(compatibility["runtime"]["segmentation_enabled"]),
            ),
            (
                "segmentation",
                "seg_loss_weight",
                _different_signed_value(compatibility["runtime"]["segmentation_loss_weight"]),
            ),
            (
                "token_ae",
                "enabled",
                _different_signed_value(compatibility["runtime"]["token_ae_enabled"]),
            ),
        ]
    )
    for section, source in (
        ("ema", preserved["ema"]),
        ("loss", preserved["loss"]),
        ("data_aug", preserved["data_aug"]),
        ("optimization", parser_defaults["optimization"]),
        ("optimization", public["optimization"]["shared_adamw"]),
    ):
        cases.extend((section, key, _different_signed_value(value)) for key, value in source.items())
    return cases


def _ablation_module():
    return importlib.import_module("app.vjepa_cowa_world_model.training.configs.cvoi_ablation")


def _cvoi_module():
    return importlib.import_module("app.vjepa_cowa_world_model.training.configs.cvoi")


def test_navsim_ablation_signature_is_independent_exact_and_metric_free() -> None:
    module = _ablation_module()
    parsed = module.parse_cvoi_ablation_signature(_ablation_signature())

    assert isinstance(parsed, module.CvoiFormalV2NavSimE120AblationSignature)
    assert parsed.to_dict() == _ablation_signature()
    assert tuple(field.name for field in fields(type(parsed))) == tuple(_ablation_signature())
    assert not any("metric" in name or "task_score" in name for name in parsed.to_dict())

    for forbidden in ("task_score_schema", "metric", "initialization"):
        drifted = _ablation_signature(**{forbidden: "forbidden"})
        with pytest.raises(ValueError, match="fields mismatch"):
            module.parse_cvoi_ablation_signature(drifted)


@pytest.mark.parametrize(
    ("supervision", "calibration"),
    [
        ("hazard_quality", "local_geometry"),
        ("none", "local_geometry"),
        ("hazard_only", "local_geometry"),
        ("quality_only", "local_geometry"),
        ("hazard_quality", "local_geometry_no_order"),
        ("hazard_quality", "factual_only"),
    ],
)
def test_navsim_ablation_registers_exactly_six_value_mechanisms(supervision: str, calibration: str) -> None:
    parsed = _ablation_module().parse_cvoi_ablation_signature(
        _ablation_signature(cf_field_supervision=supervision, field_calibration_mode=calibration)
    )
    assert (parsed.cf_field_supervision, parsed.field_calibration_mode) == (supervision, calibration)


@pytest.mark.parametrize("prefix_mode", ["uniform", "extremes", "short_heavy", "no_full"])
def test_navsim_ablation_registers_exactly_four_prefix_modes(prefix_mode: str) -> None:
    parsed = _ablation_module().parse_cvoi_ablation_signature(_ablation_signature(p0_prefix_mode=prefix_mode))
    assert parsed.p0_prefix_mode == prefix_mode


@pytest.mark.parametrize("gate_mode", ["full", "without_field", "without_stop", "without_value_summary"])
def test_navsim_ablation_registers_exactly_four_gate_modes(gate_mode: str) -> None:
    parsed = _ablation_module().parse_cvoi_ablation_signature(_ablation_signature(gate_feature_mode=gate_mode))
    assert parsed.gate_feature_mode == gate_mode


def test_navsim_gate_ablations_share_only_the_matching_full_parent() -> None:
    module = _ablation_module()
    parent = _ablation_signature(branch_id="full")
    without_field = _ablation_signature(
        experiment_role="ablation",
        branch_id="without_field",
        gate_feature_mode="without_field",
    )
    without_stop = _ablation_signature(
        experiment_role="ablation",
        branch_id="without_stop",
        gate_feature_mode="without_stop",
    )
    without_value_summary = _ablation_signature(
        experiment_role="ablation",
        branch_id="without_value_summary",
        gate_feature_mode="without_value_summary",
    )

    assert module.is_cvoi_gate_ablation_shared_parent(parent, without_field)
    assert module.is_cvoi_gate_ablation_shared_parent(parent, without_stop)
    assert module.is_cvoi_gate_ablation_shared_parent(parent, without_value_summary)
    assert not module.is_cvoi_gate_ablation_shared_parent(
        _ablation_signature(shared_cohort_id="other_cohort"),
        without_stop,
    )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"protocol_version": "formal_v2"}, "protocol_version"),
        ({"initialization_mode": "migrated"}, "full_state_warmstart"),
        ({"train_seed": True}, "train_seed"),
        ({"evaluation_seed": 3407}, "evaluation_seed"),
        ({"training_stride": 1}, "training_stride"),
        ({"p0_prefix_mode": "full_only"}, "p0_prefix_mode"),
        ({"gate_feature_mode": "gate_no_stop_value"}, "gate_feature_mode"),
        ({"cf_field_supervision": "none", "field_calibration_mode": "factual_only"}, "six"),
    ],
)
def test_navsim_ablation_rejects_protocol_drift(updates: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _ablation_module().parse_cvoi_ablation_signature(_ablation_signature(**updates))


def test_nested_warmstart_binding_is_strict_path_only_and_typed() -> None:
    config = parse_training_config(_planner_args())

    assert config.cvoi.full_state_warmstart.source_checkpoint.path == FORMAL_V2_E120_CHECKPOINT_PATH
    assert config.cvoi.full_state_warmstart.source_params_pretrain.path == FORMAL_V2_E120_PARAMS_PRETRAIN_PATH
    assert not hasattr(config.cvoi.full_state_warmstart.source_checkpoint, "sha256")
    assert not hasattr(config.cvoi.full_state_warmstart, "receipt_path")
    assert not hasattr(config.cvoi, "navsim_selection")
    assert _cvoi_module().is_cvoi_formal_v2_navsim_e120_profile(config.cvoi)
    assert _cvoi_module().cvoi_uses_managed_predictor_initialization(config.cvoi)


def test_nested_warmstart_accepts_a_portable_absolute_pair_and_matching_e120_aliases() -> None:
    checkpoint_path = "/opt/rise-user/checkpoints/e120.pt"
    params_path = "/opt/rise-user/checkpoints/params-pretrain.yaml"
    args = _planner_args()
    args["cvoi"]["full_state_warmstart"] = _warmstart_binding(
        source_checkpoint={"path": checkpoint_path},
        source_params_pretrain={"path": params_path},
    )
    args["cvoi"]["world_model_checkpoint"] = checkpoint_path
    args["cvoi"]["seed_planner_checkpoint"] = checkpoint_path

    config = parse_training_config(args)

    assert config.cvoi.full_state_warmstart.source_checkpoint.path == checkpoint_path
    assert config.cvoi.full_state_warmstart.source_params_pretrain.path == params_path
    assert config.cvoi.world_model_checkpoint == checkpoint_path
    assert config.cvoi.seed_planner_checkpoint == checkpoint_path


@pytest.mark.parametrize(
    ("checkpoint_path", "params_path", "message"),
    [
        ("e120.pt", "/opt/rise-user/checkpoints/params-pretrain.yaml", "absolute"),
        (
            "/opt/rise-user/checkpoints/e120.pt",
            "params-pretrain.yaml",
            "absolute",
        ),
        (
            "/opt/rise-user/checkpoints/../e120.pt",
            "/opt/rise-user/checkpoints/params-pretrain.yaml",
            r"traversal|\.\.",
        ),
        (
            "/opt/rise-user/checkpoints/e120.pt",
            "/opt/rise-user/checkpoints/../params-pretrain.yaml",
            r"traversal|\.\.",
        ),
        (
            "/opt/rise-user/checkpoints/./e120.pt",
            "/opt/rise-user/checkpoints/params-pretrain.yaml",
            "canonical",
        ),
        (
            "/opt/rise-user/checkpoints/e120.pth",
            "/opt/rise-user/checkpoints/params-pretrain.yaml",
            r"\.pt",
        ),
        (
            "/opt/rise-user/checkpoints/e120.pt",
            "/opt/rise-user/checkpoints/params-pretrain.yml",
            "params-pretrain.yaml",
        ),
        (
            "/opt/rise-user/checkpoints/e120.pt",
            "/opt/rise-user/config/params-pretrain.yaml",
            "same parent",
        ),
    ],
)
def test_nested_warmstart_rejects_structurally_invalid_pairs(
    checkpoint_path: str,
    params_path: str,
    message: str,
) -> None:
    args = _planner_args()
    args["cvoi"]["full_state_warmstart"] = _warmstart_binding(
        source_checkpoint={"path": checkpoint_path},
        source_params_pretrain={"path": params_path},
    )
    args["cvoi"]["world_model_checkpoint"] = checkpoint_path
    args["cvoi"]["seed_planner_checkpoint"] = checkpoint_path

    with pytest.raises(ValueError, match=message):
        parse_training_config(args)


@pytest.mark.parametrize("alias", ["world_model_checkpoint", "seed_planner_checkpoint"])
def test_nested_warmstart_requires_each_e120_checkpoint_alias_to_match_the_configured_source(alias: str) -> None:
    checkpoint_path = "/opt/rise-user/checkpoints/e120.pt"
    args = _planner_args()
    args["cvoi"]["full_state_warmstart"] = _warmstart_binding(
        source_checkpoint={"path": checkpoint_path},
        source_params_pretrain={"path": "/opt/rise-user/checkpoints/params-pretrain.yaml"},
    )
    args["cvoi"]["world_model_checkpoint"] = checkpoint_path
    args["cvoi"]["seed_planner_checkpoint"] = checkpoint_path
    args["cvoi"][alias] = "/opt/rise-user/checkpoints/copied-e120.pt"

    with pytest.raises(ValueError, match=alias):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("section", "field_name", "drifted_value"),
    _signed_public_mutation_cases(),
    ids=lambda value: str(value),
)
def test_planner_profile_rejects_every_signed_public_config_field_drift(
    section: str,
    field_name: str,
    drifted_value: object,
) -> None:
    args = _planner_args()
    if section == "top":
        args[field_name] = drifted_value
    elif section == "navsim":
        args["data"]["navsim"][field_name] = drifted_value
    else:
        args[section][field_name] = drifted_value

    with pytest.raises((TypeError, ValueError)):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda args: args["cvoi"]["full_state_warmstart"].update(extra=True), "Unknown config key|fields"),
        (
            lambda args: args["cvoi"]["full_state_warmstart"]["source_checkpoint"].update(sha256="A" * 64),
            "Unknown config key|fields",
        ),
    ],
)
def test_nested_warmstart_rejects_removed_proof_fields(mutation, message: str) -> None:
    args = _planner_args()
    mutation(args)
    with pytest.raises(ValueError, match=message):
        parse_training_config(args)


def test_p0_and_p1_planners_accept_only_real_navsim_roots_and_managed_predictor_loading() -> None:
    p0 = parse_training_config(_planner_args(stage="p0"))
    p1 = parse_training_config(_planner_args(stage="p1"))

    assert p0.train.predictor_planner_finetune is True
    assert p1.train.predictor_planner_finetune is True
    assert {root["domain"] for root in p0.data.navsim.train_roots + p0.data.navsim.val_roots} == {"real"}
    assert {root["domain"] for root in p1.data.navsim.train_roots + p1.data.navsim.val_roots} == {"real"}

    for section in ("train_roots", "val_roots"):
        drifted = _planner_args()
        drifted["data"]["navsim"][section].append(_counterfactual_root())
        with pytest.raises(ValueError, match="real-only"):
            parse_training_config(drifted)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("resume_checkpoint", "/checkpoints/resume.pt"),
        ("pretrain_repo", "facebookresearch/vjepa2"),
        ("pretrain_checkpoint", "/checkpoints/pretrain.pt"),
        ("pretrain_checkpoint_full", "/checkpoints/full.pt"),
        ("predictor_checkpoint", "/checkpoints/predictor.pt"),
        ("value_checkpoint", "/checkpoints/value.pt"),
        ("planner_value_checkpoint", "/checkpoints/planner-value.pt"),
        ("ae_checkpoint", "/checkpoints/ae.pt"),
        ("load_encoder", True),
        ("load_predictor", True),
        ("load_planner", True),
        ("load_seg", True),
        ("resume_model_only", True),
        ("auto_resume_latest", True),
    ],
)
def test_profile_rejects_every_generic_model_loader(field_name: str, bad_value: object) -> None:
    args = _planner_args()
    args["meta"][field_name] = bad_value
    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


@pytest.mark.parametrize(
    "field_name",
    ["initialization", "migration_receipt_path", "e_star_selection_path", "native_joint_config_sha256"],
)
def test_profile_forbids_legacy_initialization_and_selection_fields_even_when_null(field_name: str) -> None:
    args = _planner_args()
    args["cvoi"][field_name] = None
    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


def test_profile_requires_no_task_score_schema_and_only_it_can_alias_locked_e120_roles() -> None:
    args = _planner_args()
    args["cvoi"]["task_score_schema"] = "cvoi_world4drive_l2_collision_task_score_v2"
    with pytest.raises(ValueError, match="task_score_schema"):
        parse_training_config(args)

    wrong = _planner_args()
    wrong["cvoi"]["seed_planner_checkpoint"] = "/checkpoints/copied-e120.pt"
    with pytest.raises(ValueError, match="seed_planner_checkpoint"):
        parse_training_config(wrong)

    cvoi_module = _cvoi_module()
    legacy = cvoi_module.CVoIConfig(
        enabled=True,
        protocol_version="legacy_v1",
        stage="unguided_planner",
        world_model_checkpoint=FORMAL_V2_E120_CHECKPOINT_PATH,
        seed_planner_checkpoint=FORMAL_V2_E120_CHECKPOINT_PATH,
    )
    with pytest.raises(ValueError, match="must be exactly.*formal_v2_navsim_e120_h4_v3"):
        cvoi_module.validate_cvoi_config(legacy, None)


@pytest.mark.parametrize(
    "field_name",
    ["artifact_only", "audit_manifest_path", "audit_path_mode", "audit_verification_mode"],
)
def test_navsim_profile_rejects_historical_audit_controls(field_name: str) -> None:
    args = _planner_args()
    args["cvoi"][field_name] = "legacy" if field_name != "artifact_only" else True

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("lambda_compute", -1.0),
        ("lambda_grid", []),
        ("lambda_grid", [0.0, 0.1, 0.1]),
        ("compute_costs", [0.0, 2.0, 1.0, 3.0]),
        ("compute_costs", [0.0, 1.0]),
        ("value_hidden_dim", 0),
        ("value_num_layers", 0),
        ("value_dropout", 1.0),
        ("tokens_per_frame", 0),
        ("value_updates_per_epoch", 0),
        ("field_calibration_num_perturbations", 0),
        ("field_calibration_perturbation_scale", 0.0),
        ("field_calibration_max_delta_norm", 0.0),
        ("field_calibration_order_margin", -0.1),
    ],
)
def test_navsim_profile_preserves_common_cvoi_safety_validation(
    field_name: str,
    bad_value: object,
) -> None:
    args = _planner_args()
    args["cvoi"][field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("lambda_compute", 0.01),
        ("lambda_grid", [0.0, 0.01, 0.05]),
        ("value_updates_per_epoch", 1),
        ("field_calibration_num_perturbations", 5),
        ("field_calibration_perturbation_scale", 0.1),
        ("field_calibration_order_margin", 0.2),
    ],
)
def test_navsim_profile_preserves_exact_formal_v2_mechanism(field_name: str, bad_value: object) -> None:
    args = _planner_args()
    args["cvoi"][field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("section", "field_name", "bad_value"),
    [
        ("value_guidance", "step_size", 0.1),
        ("value_guidance", "max_delta_norm", 0.5),
        ("cvoi", "guidance_steps", 99),
        ("cvoi", "guidance_objective", "sum"),
        ("cvoi", "controller_batch_size", 2),
        ("planner", "enable_rl_actor_critic", True),
        ("planner", "rl_action_dim", 3),
        ("loss", "auto_steps", 4),
        ("data", "datasets", ["nuScenes"]),
        ("data", "camera_frame", True),
    ],
)
def test_navsim_profile_rejects_unsigned_forward_and_parser_default_drift(
    section: str,
    field_name: str,
    bad_value: object,
) -> None:
    args = _planner_args()
    args[section][field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("enabled", False),
        ("sigreg_weight", 0.1),
        ("sigreg_knots", 19),
        ("sigreg_num_proj", 512),
        ("projector_hidden_dim", 1024),
        ("embed_dim", 256),
        ("num_subspaces", 2),
        ("subspace_dim", 96),
        ("init_mode", "random"),
        ("theta", 0.5),
    ],
)
def test_navsim_profile_rejects_world_model_drift(field_name: str, bad_value: object) -> None:
    args = _planner_args()
    args["world_model"][field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


def test_navsim_profile_forbids_rl_even_when_other_planner_fields_match() -> None:
    args = _planner_args()
    args["rl"]["enabled"] = True

    with pytest.raises(ValueError, match="rl.enabled"):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("stage", "mutation"),
    [
        ("p0", lambda args: args["cvoi"].update(field_checkpoint="/results/unexpected-field.pt")),
        ("p1", lambda args: args["cvoi"].update(output_checkpoint=FORMAL_V2_E120_CHECKPOINT_PATH)),
        (
            "p1",
            lambda args: args["cvoi"].update(
                unguided_planner_checkpoint="/results/shared/../same.pt",
                field_checkpoint="/results/same.pt",
            ),
        ),
    ],
)
def test_navsim_profile_rejects_forbidden_or_normalized_artifact_aliases(stage: str, mutation) -> None:
    args = _planner_args(stage=stage)
    mutation(args)

    with pytest.raises(ValueError, match="forbids artifact|distinct|alias|locked e120"):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("annotations_require_trajectory_match", False),
        ("annotations_require_trajectory_match", None),
        ("annotations_path", None),
        ("annotations_drop_distorted", False),
        ("pose_overlay_required", False),
        ("pose_overlay_path", None),
        ("pose_overlay_coord_frame", "ego"),
        ("pose_overlay_txt_start_seconds", 0.5),
        ("trajectory_quality_path", None),
        ("window_start_policy", "sliding"),
        ("tail_seconds", 5.0),
        ("load_agent_annotations", True),
        ("max_agents", 255),
        ("window_stride", 1),
    ],
)
def test_counterfactual_root_contract_is_complete_and_strict(field_name: str, bad_value: object) -> None:
    args = _stage_args("field_warmup")
    args["data"]["navsim"]["train_roots"][1][field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("section", "index", "field_name", "bad_value"),
    [
        ("train_roots", 0, "name", "real_train"),
        ("train_roots", 0, "data_path", "/totally/wrong/logs"),
        ("train_roots", 0, "sensor_blobs_path", "/totally/wrong/blobs"),
        ("train_roots", 0, "unexpected", "drift"),
        ("val_roots", 0, "dataset_id", "wrong-navtest"),
        ("val_roots", 0, "scene_filter_yaml", "/totally/wrong/navtest.yaml"),
    ],
)
def test_real_roots_must_equal_the_committed_authority_projection(
    section: str,
    index: int,
    field_name: str,
    bad_value: object,
) -> None:
    args = _planner_args()
    args["data"]["navsim"][section][index][field_name] = bad_value

    with pytest.raises(ValueError, match="authority|template|root"):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("section", "index", "field_name", "bad_value"),
    [
        ("train_roots", 1, "annotations_path", "/results/other/annotations/navsim_cf_train.json"),
        ("train_roots", 1, "trajectory_quality_path", "/results/other/quality/navsim_cf_train.json"),
        ("val_roots", 1, "annotations_path", "/results/preflight/annotations/wrong.json"),
        ("val_roots", 1, "pose_overlay_path", "/totally/wrong/poses"),
        ("val_roots", 1, "unexpected", "drift"),
    ],
)
def test_counterfactual_roots_bind_one_preflight_and_committed_authority(
    section: str,
    index: int,
    field_name: str,
    bad_value: object,
) -> None:
    args = _stage_args("field_warmup")
    args["data"]["navsim"][section][index][field_name] = bad_value

    with pytest.raises(ValueError, match="authority|template|preflight|root"):
        parse_training_config(args)


def test_real_root_forbids_trajectory_match_key_even_when_false_or_null() -> None:
    for value in (False, None):
        args = _planner_args()
        args["data"]["navsim"]["train_roots"][0]["annotations_require_trajectory_match"] = value
        with pytest.raises(ValueError, match="annotations_require_trajectory_match"):
            parse_training_config(args)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("data_path", "/fallback/navsim/logs"),
        ("sensor_blobs_path", "/fallback/navsim/blobs"),
        ("camera_names", ["CAM_F0", "CAM_L0"]),
        ("num_history_frames", 4),
        ("index_cache", False),
        ("tail_seconds", 5.0),
        ("counterfactual_tail_seconds", 5.0),
        ("load_agent_annotations", False),
        ("pose_overlay_path", "/fallback/poses"),
        ("pose_overlay_required", True),
    ],
)
def test_navsim_global_historical_fallback_fields_are_signed(field_name: str, bad_value: object) -> None:
    args = _planner_args()
    args["data"]["navsim"][field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


def test_uniform_nonuniform_and_p1_schedule_contracts_are_independent() -> None:
    uniform = parse_training_config(_planner_args(stage="p0", prefix_mode="uniform"))
    assert uniform.optimization.epochs == 50
    assert uniform.optimization.schedule_epochs == 50
    assert uniform.meta.selection_checkpoint_epochs == FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS

    nonuniform_args = _planner_args(stage="p0", prefix_mode="extremes")
    nonuniform_args["optimization"]["epochs"] = 35
    nonuniform_args["optimization"]["schedule_epochs"] = 50
    nonuniform = parse_training_config(nonuniform_args)
    assert nonuniform.optimization.epochs == 35
    assert nonuniform.optimization.schedule_epochs == 50
    assert nonuniform.meta.selection_checkpoint_epochs == ()

    p1 = parse_training_config(_planner_args(stage="p1"))
    assert p1.optimization.epochs == 80
    assert p1.optimization.schedule_epochs == 80
    assert p1.meta.selection_checkpoint_epochs == FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda args: args["optimization"].update(epochs=49), "selection_checkpoint_epochs|Uniform P0.*epochs"),
        (lambda args: args["optimization"].update(schedule_epochs=49), "P0.*schedule_epochs"),
        (lambda args: args["meta"].update(selection_checkpoint_epochs=[10, 20]), "selection_checkpoint_epochs"),
    ],
)
def test_uniform_p0_rejects_schedule_or_selection_drift(mutation, message: str) -> None:
    args = _planner_args(stage="p0", prefix_mode="uniform")
    mutation(args)
    with pytest.raises(ValueError, match=message):
        parse_training_config(args)


def test_nonuniform_p0_still_requires_a_selected_candidate_epoch() -> None:
    args = _planner_args(stage="p0", prefix_mode="extremes")
    args["optimization"].update(epochs=35, schedule_epochs=50)
    args["optimization"]["epochs"] = 25
    with pytest.raises(ValueError, match="candidate"):
        parse_training_config(args)


def test_p1_requires_exact_schedule() -> None:
    args = _planner_args(stage="p1")
    args["optimization"]["schedule_epochs"] = 50
    with pytest.raises(ValueError, match="P1.*schedule_epochs"):
        parse_training_config(args)


@pytest.mark.parametrize(
    "stage",
    [
        "unguided_planner",
        "guided_planner",
        "field_warmup",
        "field_calibrated",
        "stop_calibrated",
        "gate_distillation",
    ],
)
def test_navsim_profile_dispatches_exactly_six_manual_cvoi_stages(stage: str) -> None:
    config = parse_training_config(_stage_args(stage))

    assert config.cvoi.stage == stage
    if stage not in {"unguided_planner", "guided_planner"}:
        assert config.optimization.lr == 7.3e-4
        assert config.optimization.epochs == 7
        assert config.optimization.schedule_epochs == 11


def test_e120_public_stage_sets_match_the_six_stage_direct_contract() -> None:
    module = _cvoi_module()
    expected = frozenset(
        {
            "unguided_planner",
            "field_warmup",
            "field_calibrated",
            "guided_planner",
            "stop_calibrated",
            "gate_distillation",
        }
    )

    assert module.CVOI_FORMAL_V2_NAVSIM_E120_STAGES == expected
    assert module.CVOI_FORMAL_V2_NAVSIM_E120_MODEL_STAGES == expected - {"gate_distillation"}


@pytest.mark.parametrize(
    "stage",
    [
        "unguided_planner",
        "guided_planner",
        "field_warmup",
        "field_calibrated",
        "stop_calibrated",
        "gate_distillation",
    ],
)
def test_non_evaluation_stages_forbid_forced_evaluation_modes(stage: str) -> None:
    args = _stage_args(stage)
    args["cvoi"]["evaluation_mode"] = "p0_forced"

    with pytest.raises(ValueError, match="evaluation_mode"):
        parse_training_config(args)


@pytest.mark.parametrize(
    "stage",
    ["unguided_planner", "guided_planner", "field_warmup", "field_calibrated"],
)
def test_pre_controller_stages_require_value_guided_lineage(stage: str) -> None:
    args = _stage_args(stage)
    args["cvoi"]["controller_lineage"] = "p0_controller"

    with pytest.raises(ValueError, match="controller_lineage"):
        parse_training_config(args)


def test_field_warmup_domain_must_match_the_projected_root_domains() -> None:
    mixed = _stage_args("field_warmup")
    mixed["cvoi"]["field_warmup_domain"] = "real"
    with pytest.raises(ValueError, match="field_warmup_domain"):
        parse_training_config(mixed)

    strict = _stage_args("field_warmup", supervision="none", calibration="local_geometry")
    strict["cvoi"]["field_warmup_domain"] = "real_cf"
    with pytest.raises(ValueError, match="field_warmup_domain"):
        parse_training_config(strict)


def test_field_root_projection_is_stage_exact() -> None:
    full = parse_training_config(_stage_args("field_warmup"))
    strict = parse_training_config(_stage_args("field_warmup", supervision="none", calibration="local_geometry"))

    assert [root["domain"] for root in full.data.navsim.train_roots] == ["real", "counterfactual"]
    assert [root["domain"] for root in full.data.navsim.val_roots] == ["real", "counterfactual"]
    assert full.data.navsim.balance_train_roots is True
    assert [root["domain"] for root in strict.data.navsim.train_roots] == ["real"]
    assert [root["domain"] for root in strict.data.navsim.val_roots] == ["real"]
    assert strict.data.navsim.balance_train_roots is False


def test_gate_consumes_only_oracle_artifact_and_no_navsim_roots() -> None:
    args = _stage_args("gate_distillation")
    args["cvoi"]["gate_training_batch_size"] = 4096
    config = parse_training_config(args)

    assert config.data.navsim is None
    assert config.cvoi.oracle_path == f"{FULL_RESULTS_ROOT}/handoff/oracle_full.sqlite3"
    assert config.cvoi.output_checkpoint == f"{FULL_RESULTS_ROOT}/handoff/gate.pt"
    assert config.cvoi.full_state_warmstart is None
    assert not hasattr(config.cvoi, "navsim_selection")
    assert config.cvoi.controller_batch_size == 1
    assert config.cvoi.gate_training_batch_size == 4096


def test_gate_training_batch_size_is_explicit_and_stage_scoped() -> None:
    missing = _stage_args("gate_distillation")
    missing["cvoi"].pop("gate_training_batch_size")
    with pytest.raises(ValueError, match="gate_training_batch_size"):
        parse_training_config(missing)

    drifted = _stage_args("gate_distillation")
    drifted["cvoi"]["gate_training_batch_size"] = 2048
    with pytest.raises(ValueError, match="gate_training_batch_size"):
        parse_training_config(drifted)

    non_gate = _stage_args("field_warmup")
    non_gate["cvoi"]["gate_training_batch_size"] = 4096
    with pytest.raises(ValueError, match="gate_training_batch_size"):
        parse_training_config(non_gate)


def test_disabled_default_is_not_the_navsim_e120_training_profile() -> None:
    assert not _cvoi_module().is_cvoi_formal_v2_navsim_e120_profile(parse_training_config({}).cvoi)


def test_profile_nested_sections_are_copied_not_shared() -> None:
    first = parse_training_config(_planner_args()).cvoi
    second = parse_training_config(_planner_args()).cvoi

    assert first.full_state_warmstart is not second.full_state_warmstart
    assert first.full_state_warmstart.source_checkpoint is not second.full_state_warmstart.source_checkpoint


@pytest.mark.parametrize(
    "stage",
    [
        "unguided_planner",
        "field_warmup",
        "field_calibrated",
        "guided_planner",
        "stop_calibrated",
        "gate_distillation",
    ],
)
def test_manual_full_stages_parse_without_proof_fields(stage: str) -> None:
    config = parse_training_config(_direct_stage_args(stage))

    assert config.cvoi.stage == stage
    assert not hasattr(config.cvoi, "navsim_selection")
    assert not hasattr(config.cvoi, "field_validation_receipt_path")
    assert not hasattr(config.cvoi, "field_validation_source_config_path")
    if stage == "gate_distillation":
        assert config.data.navsim is None
        assert config.cvoi.full_state_warmstart is None
    else:
        assert not hasattr(config.cvoi.full_state_warmstart.source_checkpoint, "sha256")
        assert not hasattr(config.cvoi.full_state_warmstart.source_params_pretrain, "sha256")
        assert not hasattr(config.cvoi.full_state_warmstart, "receipt_path")


def test_direct_p1_forced_evaluation_skips_legacy_root_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _cvoi_module()

    def _poison(*_args, **_kwargs):
        raise AssertionError("legacy root authority was invoked")

    monkeypatch.setattr(module, "build_formal_v2_navsim_root_catalog", _poison)
    monkeypatch.setattr(module, "validate_formal_v2_navsim_direct_task_projection", _poison)

    config = parse_training_config(_direct_evaluation_args())

    assert config.cvoi.stage == "evaluation"
    assert config.cvoi.evaluation_mode == "p1_field_forced"
    assert config.cvoi.controller_lineage == "value_guided"
    assert config.cvoi.rollout_horizons == [0, 1, 2, 3, 4]
    assert config.cvoi.output_checkpoint is None


@pytest.mark.parametrize(
    "field_name",
    [
        "world_model_checkpoint",
        "seed_planner_checkpoint",
        "unguided_planner_checkpoint",
        "field_checkpoint",
        "guided_planner_checkpoint",
        "dual_value_checkpoint",
    ],
)
def test_direct_p1_forced_evaluation_requires_all_six_artifacts(field_name: str) -> None:
    args = _direct_evaluation_args()
    args["cvoi"][field_name] = None

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("oracle_path", "/results/oracle/oracle.jsonl"),
        ("gate_checkpoint", "/results/gate/gate.pt"),
        ("output_checkpoint", "/results/evaluation/output.pt"),
        ("token_ae_checkpoint", "/results/token_ae/token_ae.pt"),
    ],
)
def test_direct_p1_forced_evaluation_forbids_non_input_artifacts(field_name: str, value: str) -> None:
    args = _direct_evaluation_args()
    args["cvoi"][field_name] = value

    with pytest.raises(ValueError, match="forbids artifact"):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("window_stride", 1),
        ("max_agents", FORMAL_MAX_AGENTS - 1),
    ],
)
def test_direct_p1_forced_evaluation_keeps_local_root_validation(
    field_name: str,
    bad_value: object,
) -> None:
    args = _direct_evaluation_args()
    args["data"]["navsim"]["train_roots"][0][field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("section", "index", "field_name", "bad_value"),
    [
        ("train_roots", 0, "name", "wrong_real_train"),
        ("train_roots", 0, "domain", "counterfactual"),
        ("train_roots", 0, "data_path", "/wrong/navsim/logs"),
        ("train_roots", 0, "sensor_blobs_path", "/wrong/navsim/blobs"),
        ("train_roots", 0, "scene_filter_yaml", "/wrong/navtrain.yaml"),
        ("train_roots", 0, "image_require_policy", "all_frames"),
        ("train_roots", 0, "num_observed_frames", 3),
        ("train_roots", 0, "num_target_frames", 11),
        ("train_roots", 0, "window_stride", 1),
        ("train_roots", 0, "repeat", 2),
    ],
)
def test_direct_roots_reject_effective_semantic_drift(
    section: str,
    index: int,
    field_name: str,
    bad_value: object,
) -> None:
    args = _direct_stage_args("unguided_planner")
    args["data"]["navsim"][section][index][field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        parse_training_config(args)


def test_direct_roots_reject_balancing_drift() -> None:
    args = _direct_stage_args("field_warmup")
    args["data"]["navsim"]["balance_train_roots"] = False

    with pytest.raises(ValueError, match="balance_train_roots"):
        parse_training_config(args)


def test_direct_counterfactual_quality_root_cannot_self_authorize() -> None:
    args = _direct_stage_args("field_warmup")
    for collection, role in (("train_roots", "cf_train"), ("val_roots", "cf_val")):
        counterfactual = next(
            root for root in args["data"]["navsim"][collection] if root["domain"] == "counterfactual"
        )
        counterfactual["trajectory_quality_path"] = (
            f"/results/operator-substituted-preflight/trajectory_quality/navsim_{role}.json"
        )

    with pytest.raises(ValueError, match="trajectory_quality_path|preflight|authority"):
        parse_training_config(args)


@pytest.mark.parametrize(
    ("stage", "extra_field", "extra_value"),
    [
        ("unguided_planner", "field_checkpoint", "/results/unrelated/field.pt"),
        ("field_warmup", "field_checkpoint", "/results/unrelated/field.pt"),
        ("field_calibrated", "guided_planner_checkpoint", "/results/unrelated/p1.pt"),
        ("guided_planner", "guided_planner_checkpoint", "/results/unrelated/p1.pt"),
        ("stop_calibrated", "oracle_path", "/results/unrelated/oracle.sqlite3"),
        ("gate_distillation", "world_model_checkpoint", "/results/unrelated/world.pt"),
    ],
)
def test_manual_full_stages_reject_unrelated_artifacts(
    stage: str,
    extra_field: str,
    extra_value: str,
) -> None:
    args = _direct_stage_args(stage)
    args["cvoi"][extra_field] = extra_value

    with pytest.raises(ValueError, match="forbids artifact"):
        parse_training_config(args)
