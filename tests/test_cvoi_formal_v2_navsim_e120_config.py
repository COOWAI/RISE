"""Canonical e120 public compatibility and selection-checkpoint configuration."""

from __future__ import annotations

import importlib

import pytest

from app.vjepa_cowa_world_model.training.config import parse_training_config
from app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_protocol import (
    FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS,
    FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS,
)


def _authority():
    return importlib.import_module("app.vjepa_cowa_world_model.training.cvoi_formal_v2_navsim_e120_config")


def test_public_baseline_preserves_e120_compatibility_and_training_contract() -> None:
    authority = _authority()

    baseline = authority.build_formal_v2_navsim_e120_public_config()

    assert baseline["initialization"] == {
        "mode": "full_state_warmstart",
        "roles": ["encoder", "predictor", "planner"],
        "preserve_parameters": ["observed_source_embedding.weight"],
        "restore_optimizer": False,
        "restore_scheduler": False,
        "restore_epoch": False,
        "allow_native_initialization": False,
        "allow_legacy_missing_embedding_migration": False,
        "allow_fallback": False,
    }
    compatibility = baseline["compatibility"]
    assert compatibility["method"] == "lewm"
    assert compatibility["forward_semantics"] == {
        "training_line": "train_predictor_rollout_planner",
        "encoder_output": "observed_video_patch_tokens",
        "predictor_rollout": "autoregressive_future_latent_tokens",
        "planner_conditioning": "observed_concat_type_embed_plus_predicted_future_tokens",
        "planner_output": "multimodal_diffusion_trajectories_and_confidences",
    }
    assert compatibility["data"] == {
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
        "navsim_max_agents": 1024,
        "navsim_max_frame_gap": 1,
    }
    assert compatibility["runtime"] == {
        "dtype": "bfloat16",
        "seed": 239,
        "use_sdpa": True,
        "context_encoder_key": "encoder",
        "target_encoder_key": "target_encoder",
        "token_ae_enabled": False,
        "segmentation_enabled": False,
        "segmentation_loss_weight": 0.0,
    }
    model = compatibility["model"]
    assert model["model_name"] == "vit_large"
    assert model["backbone"] == "vjepa_img_encoder"
    assert model["vjepa_resolution"] == [256, 512]
    assert model["vjepa_checkpoint_key"] == "encoder"
    assert model["pred_num_heads"] == 12
    retired_prefix = "_".join(("drive", "jepa"))
    assert not any(retired_prefix in key.lower() for key in model)
    assert compatibility["predictor_inputs"] == {
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
    }
    assert compatibility["training_state"]["perceiver_ema"] is False
    planner = compatibility["planner"]
    assert planner["planner_type"] == "diffusion"
    assert type(planner["planner_loss_weight"]) is int and planner["planner_loss_weight"] == 1
    assert planner["observed_token_mode"] == "concat_type_embed"
    assert planner["use_action_history_for_planner"] is True
    assert planner["action_history_dim"] == 3
    assert planner["status_dim"] == 8
    assert planner["diff_hidden_dim"] == 384
    assert planner["diff_num_layers"] == 12
    assert planner["diff_num_heads"] == 12
    assert planner["diff_inference_steps"] == 20
    assert planner["diff_num_samples"] == 6
    assert planner["diff_num_modes"] == 6
    assert planner["diff_traj_dim"] == 4
    assert planner["diff_trajectory_token_mode"] == "per_pose_token"

    optimizer = baseline["optimization"]["shared_adamw"]
    assert optimizer == {
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
    }
    assert baseline["optimization"]["p0"] == {
        "epochs": 50,
        "schedule_epochs": 50,
        "selection_checkpoint_epochs": list(FORMAL_V2_NAVSIM_P0_CANDIDATE_EPOCHS),
    }
    assert baseline["optimization"]["p1"] == {
        "epochs": 80,
        "schedule_epochs": 80,
        "selection_checkpoint_epochs": list(FORMAL_V2_NAVSIM_P1_CANDIDATE_EPOCHS),
    }
    assert baseline["preserved"]["ema"] == {"ema_start": 0.996, "ema_end": 0.999}
    assert baseline["preserved"]["loss"] == {"loss_exp": 1.0, "normalize_reps": False}
    assert baseline["preserved"]["checkpoint_cadence"] == {"save_every_freq": 10, "val_freq": 5}
    assert baseline["parser_defaults"]
    assert authority.validate_formal_v2_navsim_e120_public_config(baseline) == baseline


def test_public_training_contract_has_no_self_signature_or_artifact_digests() -> None:
    authority = _authority()
    baseline = authority.build_formal_v2_navsim_e120_public_config()

    assert "artifacts" not in baseline
    assert not hasattr(authority, "FORMAL_V2_NAVSIM_E120_PUBLIC_CONFIG_SHA256")
    assert not hasattr(authority, "compute_formal_v2_navsim_e120_public_config_sha256")
    assert not hasattr(authority, "verify_formal_v2_navsim_e120_artifacts")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("parser_defaults"),
        lambda value: value.update(extra=True),
        lambda value: value["compatibility"]["planner"].update(diff_hidden_dim=256),
        lambda value: value["optimization"]["shared_adamw"].update(optimizer="sgd"),
    ],
)
def test_public_baseline_rejects_missing_unknown_and_drift(mutation) -> None:
    authority = _authority()
    baseline = authority.build_formal_v2_navsim_e120_public_config()
    mutation(baseline)

    with pytest.raises(ValueError):
        authority.validate_formal_v2_navsim_e120_public_config(baseline)


def test_public_baseline_is_fresh_and_mapping_order_independent() -> None:
    authority = _authority()
    first = authority.build_formal_v2_navsim_e120_public_config()
    second = authority.build_formal_v2_navsim_e120_public_config()
    first["compatibility"]["planner"]["diff_num_layers"] = 1

    assert second["compatibility"]["planner"]["diff_num_layers"] == 12
    reordered = dict(reversed(list(second.items())))
    assert authority.validate_formal_v2_navsim_e120_public_config(reordered) == second


def test_meta_selection_checkpoint_epochs_parses_to_strict_tuple() -> None:
    config = parse_training_config(
        {
            "meta": {"selection_checkpoint_epochs": [10, 20, 35, 50]},
            "optimization": {"epochs": 50},
        }
    )
    assert config.meta.selection_checkpoint_epochs == (10, 20, 35, 50)
    assert parse_training_config({}).meta.selection_checkpoint_epochs == ()


@pytest.mark.parametrize(
    "value",
    [
        [True],
        [1.0],
        [0],
        [-1],
        [10, 10],
        [20, 10],
        "10,20",
    ],
)
def test_meta_selection_checkpoint_epochs_rejects_invalid_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="selection_checkpoint_epochs"):
        parse_training_config(
            {
                "meta": {"selection_checkpoint_epochs": value},
                "optimization": {"epochs": 50},
            }
        )


def test_meta_selection_checkpoint_epochs_rejects_out_of_range_and_unknown_key() -> None:
    with pytest.raises(ValueError, match="selection_checkpoint_epochs.*optimization.epochs"):
        parse_training_config(
            {
                "meta": {"selection_checkpoint_epochs": [10, 51]},
                "optimization": {"epochs": 50},
            }
        )
    with pytest.raises(ValueError, match="Unknown config key 'meta.selection_checkpoint_epoch'"):
        parse_training_config({"meta": {"selection_checkpoint_epoch": [10]}})
