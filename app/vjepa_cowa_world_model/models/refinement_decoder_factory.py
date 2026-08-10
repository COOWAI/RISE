"""Factory helpers for Stage-3 refinement decoders."""

from __future__ import annotations

from .diffusion_refinement_decoder import DiffusionRefinementDecoder


def resolve_refinement_core_type(config) -> str:
    explicit_core = getattr(config.planner, "refinement_core_type", None)
    if explicit_core:
        return str(explicit_core).lower()
    return str(getattr(config.planner, "planner_type", "transformer")).lower()


def build_refinement_decoder(
    config,
    encoder_dim: int,
    tokens_per_frame: int,
    num_poses: int,
    status_dim: int,
    command_dim: int,
    main_num_context_frames: int | None = None,
):
    core_type = resolve_refinement_core_type(config)
    context_frames = main_num_context_frames or config.train.num_observed_frames
    max_rounds = max(config.refinement_gated.num_rounds, 2)

    if core_type in {"transformer", "multimodal", "multimodal_temporal", "refinement_decoder"}:
        from .refinement_decoder import RefinementDecoder

        return RefinementDecoder(
            encoder_dim=encoder_dim,
            tf_d_model=config.planner.tf_d_model,
            tf_d_ffn=config.planner.tf_d_ffn,
            tf_num_layers=config.planner.tf_num_layers,
            tf_num_head=config.planner.tf_num_head,
            tf_dropout=config.planner.tf_dropout,
            tokens_per_frame=tokens_per_frame,
            num_poses=num_poses,
            num_time_steps=num_poses,
            num_context_frames=context_frames,
            status_dim=status_dim,
            use_spatial_tokens=config.planner.use_spatial_tokens,
            num_modes=config.planner.num_modes,
            use_temporal=True,
            use_time_aligned_bias=config.planner.temporal_alignment,
            use_status_for_planner=config.planner.use_status_for_planner,
            command_dim=command_dim,
            max_rounds=max_rounds,
        )

    if core_type == "diffusion":
        return DiffusionRefinementDecoder(
            encoder_dim=encoder_dim,
            hidden_dim=config.planner.diff_hidden_dim,
            depth=config.planner.diff_num_layers,
            heads=config.planner.diff_num_heads,
            dropout=config.planner.diff_dropout,
            mlp_ratio=config.planner.diff_mlp_ratio,
            tokens_per_frame=tokens_per_frame,
            num_poses=num_poses,
            status_dim=status_dim,
            num_modes=config.planner.num_modes,
            traj_dim=config.planner.diff_traj_dim,
            sde_beta_min=config.planner.diff_sde_beta_min,
            sde_beta_max=config.planner.diff_sde_beta_max,
            num_samples=config.planner.diff_num_samples,
            inference_steps=config.planner.diff_inference_steps,
            trajectory_token_mode=config.planner.diff_trajectory_token_mode,
            use_anchor_frame=config.planner.diff_use_anchor_frame,
            independent_modes=config.planner.diff_independent_modes,
            cls_loss_weight=config.planner.diff_cls_loss_weight,
            reg_loss_weight=config.planner.diff_reg_loss_weight,
            vel_loss_weight=config.planner.diff_vel_loss_weight,
            yaw_loss_weight=config.planner.diff_yaw_loss_weight,
            awta_init_temperature=config.planner.awta_init_temperature,
            awta_min_temperature=config.planner.awta_min_temperature,
            conf_temperature=config.planner.diff_conf_temperature,
            cls_th=config.planner.diff_cls_th,
            cls_ignore=config.planner.diff_cls_ignore,
            command_dim=command_dim,
            adaln_version=config.planner.diff_adaln_version,
            mode_token_expansion=config.planner.diff_mode_token_expansion,
            max_rounds=max_rounds,
            dt=config.planner.diff_dt,
        )

    raise ValueError(f"Unsupported refinement_core_type/planner_type for Stage-3 refinement: {core_type}")
