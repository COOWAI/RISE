"""Explicit e120 training-semantics contract for CVoI Formal-v2 NavSim."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_MAX_AGENTS,
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)

_CANONICAL_PUBLIC_CONFIG: dict[str, object] = {
    "schema": "cvoi_formal_v2_navsim_e120_public_config_v1",
    "initialization": {
        "mode": "full_state_warmstart",
        "roles": ["encoder", "predictor", "planner"],
        "preserve_parameters": ["observed_source_embedding.weight"],
        "restore_optimizer": False,
        "restore_scheduler": False,
        "restore_epoch": False,
        "allow_native_initialization": False,
        "allow_legacy_missing_embedding_migration": False,
        "allow_fallback": False,
    },
    "compatibility": {
        "method": "lewm",
        "forward_semantics": {
            "training_line": "train_predictor_rollout_planner",
            "encoder_output": "observed_video_patch_tokens",
            "predictor_rollout": "autoregressive_future_latent_tokens",
            "planner_conditioning": "observed_concat_type_embed_plus_predicted_future_tokens",
            "planner_output": "multimodal_diffusion_trajectories_and_confidences",
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
            "navsim_camera_name": "CAM_F0",
            "navsim_image_require_policy": "observed_only",
            "navsim_max_agents": FORMAL_V2_NAVSIM_MAX_AGENTS,
            "navsim_max_frame_gap": 1,
        },
        "runtime": {
            "dtype": "bfloat16",
            "seed": 239,
            "use_sdpa": True,
            "context_encoder_key": "encoder",
            "target_encoder_key": "target_encoder",
            "token_ae_enabled": False,
            "segmentation_enabled": False,
            "segmentation_loss_weight": 0.0,
        },
        "model": {
            "model_name": "vit_large",
            "backbone": "vjepa_img_encoder",
            "vjepa_resolution": [256, 512],
            "vjepa_crop_top_bottom": 28,
            "vjepa_num_frames": 2,
            "vjepa_checkpoint_key": "encoder",
            "vjepa_use_grid_mask": False,
            "vjepa_use_causal_attention": True,
            "patch_size": 16,
            "pred_depth": 12,
            "pred_num_heads": 12,
            "pred_embed_dim": 384,
            "pred_is_frame_causal": True,
            "uniform_power": True,
            "use_rope": True,
            "use_extrinsics": False,
            "use_activation_checkpointing": True,
        },
        "predictor_inputs": {
            "predictor_type": "ac_transformer",
            "num_observed_frames": 4,
            "action_dim": 3,
            "state_dim": 8,
            "use_states_for_predictor": False,
            "predictor_aux_policy": "inference_consistent",
            "predictor_inference_consistent": True,
            "predictor_loss_scope": "future_only",
            "predictor_supervision_mode": "tf_ar",
            "predictor_use_z_ar_supervision": True,
            "use_parallel_predictor": False,
        },
        "training_state": {
            "encoder_train": False,
            "seg_head": False,
            "encoder_ema": False,
            "perceiver_ema": False,
            "predictor_train": False,
            "predictor_planner_finetune": True,
            "use_drive_command": False,
        },
        "planner": {
            "use_planner": True,
            "planner_type": "diffusion",
            "planner_loss_weight": 1,
            "num_modes": 6,
            "observed_token_mode": "concat_type_embed",
            "use_observed_tokens": True,
            "use_action_history_for_planner": True,
            "action_history_dim": 3,
            "use_status_for_planner": True,
            "use_states_for_planner": True,
            "status_dim": 8,
            "split_status_embedding": False,
            "use_drive_command": False,
            "diff_hidden_dim": 384,
            "diff_num_layers": 12,
            "diff_num_heads": 12,
            "diff_dropout": 0.0,
            "diff_mlp_ratio": 4.0,
            "diff_sde_beta_min": 0.1,
            "diff_sde_beta_max": 20.0,
            "diff_inference_steps": 20,
            "diff_num_samples": 6,
            "diff_num_modes": 6,
            "diff_traj_dim": 4,
            "diff_dt": 0.5,
            "diff_trajectory_token_mode": "per_pose_token",
            "diff_adaln_version": "v2",
            "diff_use_last_frame_only": False,
            "diff_interleave_predictor_sampling": False,
            "diff_train_prefix_conditioning": False,
            "diff_train_min_prefix_frames": 1,
            "diff_train_full_prefix_prob": 0.25,
            "diff_train_max_non_full_prefix_frames": None,
            "diff_independent_modes": False,
            "diff_mode_token_expansion": False,
            "diff_use_anchor_frame": True,
            "diff_init_traj_strategy": "gaussian",
            "diff_init_traj_noise_scale": 1.0,
            "diff_init_traj_yaw_span_deg": 30.0,
            "diff_init_traj_speed_scale_span": 0.2,
            "diff_cls_loss_weight": 1.0,
            "diff_reg_loss_weight": 1.0,
            "diff_vel_loss_weight": 0.5,
            "diff_yaw_loss_weight": 0.5,
            "diff_generation_framework": "vp_diffusion",
            "diff_conf_temperature": 1.5,
            "diff_cls_th": 2.0,
            "diff_cls_ignore": 0.2,
            "awta_init_temperature": 8.0,
            "awta_exp_base": 0.984,
            "awta_min_temperature": 0.1,
        },
    },
    "optimization": {
        "shared_adamw": {
            "optimizer": "adamw",
            "ipe": None,
            "betas": [0.9, 0.999],
            "eps": 1e-08,
            "lr": 0.0002,
            "start_lr": 2e-05,
            "final_lr": 0.0,
            "weight_decay": 0.04,
            "final_weight_decay": 0.04,
            "enc_lr_scale": 0.001,
            "predictor_lr_scale": 0.1,
            "warmup": 15,
            "anneal": 15,
        },
        "p0": {
            "epochs": 50,
            "schedule_epochs": 50,
            "selection_checkpoint_epochs": list(FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS),
        },
        "p1": {
            "epochs": 80,
            "schedule_epochs": 80,
            "selection_checkpoint_epochs": list(FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS),
        },
    },
    "preserved": {
        "ema": {"ema_start": 0.996, "ema_end": 0.999},
        "loss": {"loss_exp": 1.0, "normalize_reps": False},
        "data_aug": {
            "horizontal_flip": False,
            "random_resize_aspect_ratio": [0.75, 1.35],
            "random_resize_scale": [1.777, 1.777],
            "motion_shift": False,
            "reprob": 0.0,
            "auto_augment": False,
        },
        "checkpoint_cadence": {"save_every_freq": 10, "val_freq": 5},
    },
    "parser_defaults": {
        "meta": {
            "deterministic": False,
            "val_stable_noise": True,
            "predictor_runtime_normalize_reps": None,
            "selection_checkpoint_epochs": [],
            "save_from_epoch": 0,
            "sync_gc": False,
            "resume_broadcast": False,
            "resume_model_only": False,
            "auto_resume_latest": False,
        },
        "model": {
            "use_silu": False,
            "use_pred_silu": False,
            "wide_silu": True,
            "use_mask_tokens": False,
            "zero_init_mask_tokens": True,
            "compile_model": False,
        },
        "train": {
            "command_dim": 0,
            "predictor_validation_enabled": True,
            "predictor_static_graph": False,
            "reuse_context_as_target_when_frozen": False,
            "predictor_no_aux_input": False,
            "latent_dit_planner_input": "train_helper",
        },
        "planner": {
            "use_spatial_tokens": False,
            "use_temporal": False,
            "temporal_alignment": True,
            "z_ar_mode": "full",
            "planner_input_source": "z_ar",
            "num_modes": 6,
            "num_context_frames": 1,
            "conf_loss_weight": 1.0,
            "reg_loss_weight": 1.0,
            "horizon_reg_loss_seconds": [],
            "horizon_reg_loss_weights": [],
            "horizon_reg_loss_normalize": True,
            "states_mode": "first",
            "use_z_context": False,
            "latent_dit_action_source": "planner",
            "policy_output_source": "planner",
            "wta_loss_version": "v1",
            "wta_temperature": 1.0,
            "wta_alpha": 5.0,
            "wta_global_batch_norm": True,
            "cover_loss_weight": 0.1,
            "refinement_core_type": None,
            "diff_flow_matching_variant": "rectified",
            "diff_flow_shift": 1.0,
            "diff_flow_sampler": "euler",
            "diff_flow_timestep_sampling": "logit_normal",
        },
        "optimization": {
            "grad_clip_norm": 1.0,
            "is_anneal": False,
            "anneal_ckpt": None,
            "resume_anneal": False,
            "ipe_scale": 1.0,
        },
    },
}


def _validate_exact(value: object, expected: object, *, path: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping, got {type(value).__name__}")
        expected_keys = set(expected)
        actual_keys = set(value)
        missing = expected_keys - actual_keys
        unknown = actual_keys - expected_keys
        if missing:
            raise ValueError(f"{path} is missing required fields: {sorted(missing, key=str)}")
        if unknown:
            raise ValueError(f"{path} has unknown fields: {sorted(unknown, key=str)}")
        for key, expected_item in expected.items():
            _validate_exact(value[key], expected_item, path=f"{path}.{key}")
        return

    if isinstance(expected, list):
        if type(value) is not list:
            raise ValueError(f"{path} must be a list, got {type(value).__name__}")
        if len(value) != len(expected):
            raise ValueError(f"{path} must contain exactly {len(expected)} items, got {len(value)}")
        for index, expected_item in enumerate(expected):
            _validate_exact(value[index], expected_item, path=f"{path}[{index}]")
        return

    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{path} must be exactly {expected!r}, got {value!r}")


def validate_formal_v2_navsim_e120_public_config(config: Mapping[str, object]) -> dict[str, object]:
    """Fail fast unless ``config`` exactly matches the e120 training contract."""

    _validate_exact(config, _CANONICAL_PUBLIC_CONFIG, path="public_config")
    return copy.deepcopy(dict(config))


def build_formal_v2_navsim_e120_public_config() -> dict[str, object]:
    """Return a fresh, validated copy of the explicit training contract."""

    config = copy.deepcopy(_CANONICAL_PUBLIC_CONFIG)
    return validate_formal_v2_navsim_e120_public_config(config)
