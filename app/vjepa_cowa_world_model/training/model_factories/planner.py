"""Split from training/models.py (verbatim node moves). Part: planner."""

from typing import Optional

import torch
import torch.nn as nn

from app.vjepa_cowa_world_model.training.config import (
    TrainingConfig,
    is_vjepa_main_encoder_config,
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
    resolve_planner_observed_token_mode,
)
from app.vjepa_cowa_world_model.training.configs.planner import PLANNER_TYPES
from app.vjepa_cowa_world_model.training.model_factories.common import logger
from app.vjepa_cowa_world_model.utils import get_status_dim, resolve_planner_use_drive_command


def init_planner(
    config: TrainingConfig,
    encoder_dim: int,
    device: torch.device,
    num_poses: Optional[int] = None,
    tokens_per_frame_override: Optional[int] = None,
) -> Optional[nn.Module]:
    """
    初始化 planner

    Args:
        config: 训练配置
        encoder_dim: encoder 的嵌入维度
        device: 设备
        num_poses: 可选，自定义 num_poses 值。如果为 None，则根据模式自动计算：
                   - predictor_inference_consistent=True: total_frames - num_encoder_frames
                   - 否则: total_frames - 1
        tokens_per_frame_override: 可选，覆盖 planner 使用的每帧 token 数

    Returns:
        Optional[nn.Module]: planner 模型，如果未启用则返回 None
    """
    if config.predictor_dynamic_rollout.enabled and config.planner.diff_train_prefix_conditioning:
        raise ValueError(
            "predictor_dynamic_rollout.enabled and planner.diff_train_prefix_conditioning cannot both be true: "
            "outer and planner-internal prefix sampling would sample twice"
        )
    if not config.planner.use_planner:
        logger.info("use_planner=False, planner is disabled")
        return None

    policy_output_source = str(getattr(config.planner, "policy_output_source", "planner")).lower()
    if policy_output_source not in {"planner", "joint_action"}:
        raise ValueError(
            "planner.policy_output_source must be one of ['planner', 'joint_action'], " f"got {policy_output_source!r}"
        )
    if policy_output_source == "joint_action":
        logger.info("planner.policy_output_source='joint_action', learned planner module is disabled")
        return None

    # Fail-loud on an unknown planner_type instead of silently falling through to the transformer
    # default below — a typo like "diffsion" must not quietly build the wrong planner. Only
    # "diffusion" takes the dedicated branch; every other value historically defaulted to the
    # transformer planner, so the valid set is exactly these two.
    if config.planner.planner_type not in PLANNER_TYPES:
        raise ValueError(
            f"Unknown planner.planner_type={config.planner.planner_type!r}; " f"expected one of {PLANNER_TYPES}."
        )

    # 计算 num_poses
    # total_frames = config.data.num_target_frames // config.data.tubelet_size
    total_frames = (
        config.data.num_target_frames
    )  # should be num_target_frames, not divided by tubelet_size, because planner operates at frame level now
    if num_poses is None:
        # 默认计算逻辑
        if config.train.predictor_inference_consistent:
            num_poses = total_frames - config.train.num_encoder_frames
        else:
            num_poses = total_frames - 1

    # Planner trajectory heads still predict raw future poses (num_poses), but the temporal
    # memory may be produced at a coarser main-encoder step.  V-JEPA main encodes one
    # non-overlapping 2-frame chunk as one predictor/planner token step.
    num_time_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_poses)
    num_context_frames = max(1, int(config.planner.num_context_frames))
    tokens_per_frame = (
        int(tokens_per_frame_override) if tokens_per_frame_override is not None else config.data.tokens_per_frame
    )
    enable_rl_actor_critic = config.planner.enable_rl_actor_critic or config.rl.enabled
    use_action_history = bool(config.planner.use_action_history_for_planner)
    action_history_dim = int(config.planner.action_history_dim)
    if is_vjepa_main_encoder_config(config):
        num_observed_frames = resolve_main_encoder_num_observed_steps(config)
    else:
        num_observed_frames = int(config.train.num_encoder_frames)
    observed_token_mode = resolve_planner_observed_token_mode(config)

    # 根据 predictor_inference_consistent 和 use_states_for_planner 决定 status_dim
    # IC 格式: 7(drive_command_7) 或 8(drive_command_8)
    # 根据 predictor_inference_consistent、use_drive_command_for_predictor 和 use_states_for_planner 决定 status_dim
    _use_cmd = resolve_planner_use_drive_command(config)
    if config.rl.enabled:
        planner_status_dim = get_status_dim(
            config.rl.status_mode,
            num_context_frames=num_context_frames,
        )
    elif config.train.predictor_inference_consistent:
        # planner.status_dim > 0 时优先使用（解耦 predictor.state_dim 与 planner.status_dim）
        planner_status_dim = config.planner.status_dim if config.planner.status_dim > 0 else config.train.state_dim
    elif config.planner.use_states_for_planner:
        planner_status_dim = 7  # 原始 states 维度
    else:
        planner_status_dim = 8  # 提取的特征维度

    # use_drive_command=False 时：planner 维度减 4，command_dim 强制 0
    if not _use_cmd:
        planner_status_dim = planner_status_dim - 4
        logger.info(f"use_drive_command=False: planner status_dim -> {planner_status_dim}, command_dim forced to 0")

    # 防止未知维度静默走到 Linear 层产生难以定位的 shape error
    _valid_dims = (3, 4, 7, 8, 12) if not _use_cmd else (7, 8, 12)
    if not (planner_status_dim in _valid_dims):
        raise AssertionError(
            f"Unsupported planner_status_dim={planner_status_dim}; expected one of {_valid_dims}. "
            f"Check train.state_dim and planner.status_dim in your config."
        )

    # 计算 command_dim：IC 模式下 status = [cmd(4) | kinematics(N)]，拆分嵌入
    if _use_cmd and config.planner.split_status_embedding and config.train.predictor_inference_consistent:
        planner_command_dim = 4  # drive_command one-hot 维度
    else:
        planner_command_dim = 0  # 旧行为，不拆分（或 use_drive_command=False）

    diff_adaln_version = config.planner.diff_adaln_version
    diff_init_traj_strategy = str(config.planner.diff_init_traj_strategy).lower()
    diff_init_traj_noise_scale = float(config.planner.diff_init_traj_noise_scale)
    diff_init_traj_yaw_span_deg = float(config.planner.diff_init_traj_yaw_span_deg)
    diff_init_traj_speed_scale_span = float(config.planner.diff_init_traj_speed_scale_span)
    diff_dt = float(config.planner.diff_dt)
    diff_generation_framework = str(config.planner.diff_generation_framework).lower()
    use_seeded_diff_init = not (
        diff_init_traj_strategy == "gaussian"
        and diff_init_traj_noise_scale == 1.0
        and diff_init_traj_yaw_span_deg == 30.0
        and diff_init_traj_speed_scale_span == 0.2
    )

    # ── Status dimension summary（所有 planner 类型共享）──
    _status_layouts = {
        (3, False): "[velocity, acceleration, yaw_rate]",
        (4, False): "[vx, vy, ax, ay]",
        (7, True): "[cmd(4), velocity, acceleration, yaw_rate]",
        (8, True): "[cmd(4), vx, vy, ax, ay]",
        (8, False): "[vx, vy, ax, ay, x_local, y_local, sin_yaw, cos_yaw]",
        (12, True): "[cmd(4), vx, vy, ax, ay, x_local, y_local, sin_yaw, cos_yaw]",
    }
    _layout = _status_layouts.get(
        (planner_status_dim, _use_cmd),
        f"custom({planner_status_dim}d)",
    )
    logger.info(
        f"[Status Summary] planner_status_dim={planner_status_dim}, "
        f"command_dim={planner_command_dim}, "
        f"use_drive_command={_use_cmd}, "
        f"split_status_embedding={config.planner.split_status_embedding} | layout: {_layout}"
    )

    def _diffusion_base_kwargs():
        """Kwargs shared verbatim by every diffusion-family variant (the 50+-key common core).

        Returned identical across variants; each branch then layers on its variant-specific extras
        (flow / prefix / seeded-init). Closures over the already-resolved locals keep this DRY without
        changing any key or value (so model construction stays byte-identical to the prior 3× copies).
        """
        return dict(
            encoder_dim=encoder_dim,
            num_poses=num_poses,
            status_dim=planner_status_dim,
            hidden_dim=config.planner.diff_hidden_dim,
            depth=config.planner.diff_num_layers,
            heads=config.planner.diff_num_heads,
            dropout=config.planner.diff_dropout,
            mlp_ratio=config.planner.diff_mlp_ratio,
            traj_dim=config.planner.diff_traj_dim,
            sde_beta_min=config.planner.diff_sde_beta_min,
            sde_beta_max=config.planner.diff_sde_beta_max,
            num_samples=config.planner.diff_num_samples,
            inference_steps=config.planner.diff_inference_steps,
            use_z_context=config.planner.use_z_context,
            tokens_per_frame=tokens_per_frame,
            trajectory_token_mode=config.planner.diff_trajectory_token_mode,
            use_last_frame_only=config.planner.diff_use_last_frame_only,
            use_action_history=use_action_history,
            action_history_dim=action_history_dim,
            num_observed_frames=num_observed_frames,
            observed_token_mode=observed_token_mode,
            num_modes=config.planner.diff_num_modes,
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
            command_dim=planner_command_dim,
            adaln_version=diff_adaln_version,
            mode_token_expansion=config.planner.diff_mode_token_expansion,
        )

    def _seed_init_kwargs():
        """Seeded trajectory-init kwargs (flow-matching always; other variants when use_seeded_diff_init)."""
        return dict(
            init_traj_strategy=diff_init_traj_strategy,
            init_traj_noise_scale=diff_init_traj_noise_scale,
            init_traj_yaw_span_deg=diff_init_traj_yaw_span_deg,
            init_traj_speed_scale_span=diff_init_traj_speed_scale_span,
            dt=diff_dt,
        )

    # 根据 planner_type 选择 planner 实现
    if config.planner.planner_type == "diffusion":
        if diff_generation_framework in {"flow_matching", "flow", "fm"}:
            if config.planner.diff_interleave_predictor_sampling:
                raise ValueError(
                    "diff_interleave_predictor_sampling=True is not supported with flow_matching "
                    "(FlowMatchingDiffusionPlanner.init_interleaved_inference_state raises at inference) — "
                    "disable interleave or use vp_diffusion."
                )
            from app.vjepa_cowa_world_model.models import FlowMatchingDiffusionPlanner as PlannerCls

            planner_kwargs = _diffusion_base_kwargs()
            planner_kwargs.update(_seed_init_kwargs())  # flow-matching always uses seeded init
            planner_kwargs.update(
                flow_matching_variant=config.planner.diff_flow_matching_variant,
                flow_shift=config.planner.diff_flow_shift,
                flow_sampler=config.planner.diff_flow_sampler,
                flow_timestep_sampling=config.planner.diff_flow_timestep_sampling,
                train_prefix_conditioning=config.planner.diff_train_prefix_conditioning,
                train_min_prefix_frames=config.planner.diff_train_min_prefix_frames,
                train_full_prefix_prob=config.planner.diff_train_full_prefix_prob,
                train_max_non_full_prefix_frames=config.planner.diff_train_max_non_full_prefix_frames,
            )
            planner = PlannerCls(**planner_kwargs).to(device)
            planner_impl_name = PlannerCls.__name__
        elif diff_generation_framework not in {"vp_diffusion", "diffusion", "vp", "vp_sde"}:
            raise ValueError(
                f"Unsupported diff_generation_framework={diff_generation_framework!r}; "
                "expected 'vp_diffusion' or 'flow_matching'"
            )
        elif config.planner.diff_train_prefix_conditioning:
            if use_seeded_diff_init:
                from app.vjepa_cowa_world_model.models import PrefixConditionedSeededDiffusionPlanner as PlannerCls
            else:
                from app.vjepa_cowa_world_model.models import PrefixConditionedDiffusionPlanner as PlannerCls

            planner_kwargs = _diffusion_base_kwargs()
            planner_kwargs.update(
                train_min_prefix_frames=config.planner.diff_train_min_prefix_frames,
                train_full_prefix_prob=config.planner.diff_train_full_prefix_prob,
                train_max_non_full_prefix_frames=config.planner.diff_train_max_non_full_prefix_frames,
            )
            if use_seeded_diff_init:
                planner_kwargs.update(_seed_init_kwargs())

            planner = PlannerCls(**planner_kwargs).to(device)
            planner_impl_name = PlannerCls.__name__
        else:
            if use_seeded_diff_init:
                from app.vjepa_cowa_world_model.models import SeededDiffusionPlanner as PlannerCls
            else:
                from app.vjepa_cowa_world_model.models import DiffusionPlanner as PlannerCls

            planner_kwargs = _diffusion_base_kwargs()
            if use_seeded_diff_init:
                planner_kwargs.update(_seed_init_kwargs())

            planner = PlannerCls(**planner_kwargs).to(device)
            planner_impl_name = PlannerCls.__name__

        planner_params = sum(p.numel() for p in planner.parameters())
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"({planner_impl_name}, type=DiT, hidden_dim={config.planner.diff_hidden_dim}, "
            f"depth={config.planner.diff_num_layers}, heads={config.planner.diff_num_heads}, "
            f"traj_dim={config.planner.diff_traj_dim}, num_poses={num_poses}, "
            f"trajectory_token_mode={config.planner.diff_trajectory_token_mode}, "
            f"interleave_predictor_sampling={config.planner.diff_interleave_predictor_sampling}, "
            f"generation_framework={diff_generation_framework}, "
            f"flow_variant={config.planner.diff_flow_matching_variant}, "
            f"sde_beta=[{config.planner.diff_sde_beta_min}, {config.planner.diff_sde_beta_max}], "
            f"inference_steps={config.planner.diff_inference_steps}, "
            f"num_samples={config.planner.diff_num_samples}, "
            f"num_modes={config.planner.diff_num_modes}, "
            f"seed_init={diff_init_traj_strategy}, "
            f"status_dim={planner_status_dim}, command_dim={planner_command_dim})"
        )
        return planner

    # 默认: transformer planner (原始实现)
    from app.vjepa_cowa_world_model.models import MultiModalTemporalPlanner

    planner = MultiModalTemporalPlanner(
        encoder_dim=encoder_dim,
        tf_d_model=config.planner.tf_d_model,
        tf_d_ffn=config.planner.tf_d_ffn,
        tf_num_layers=config.planner.tf_num_layers,
        tf_num_head=config.planner.tf_num_head,
        tf_dropout=config.planner.tf_dropout,
        tokens_per_frame=tokens_per_frame,
        num_poses=num_poses,
        num_time_steps=num_time_steps,
        num_context_frames=num_context_frames,
        status_dim=planner_status_dim,
        use_spatial_tokens=config.planner.use_spatial_tokens,
        num_modes=config.planner.num_modes,
        use_temporal=config.planner.use_temporal,
        use_time_aligned_bias=config.planner.temporal_alignment,
        use_z_context=config.planner.use_z_context,
        use_status_for_planner=config.planner.use_status_for_planner,
        use_observed_tokens=config.planner.use_observed_tokens,
        observed_token_mode=observed_token_mode,
        use_action_history=use_action_history,
        action_history_dim=action_history_dim,
        enable_rl_actor_critic=enable_rl_actor_critic,
        rl_action_dim=config.planner.rl_action_dim,
        num_observed_frames=num_observed_frames,
        command_dim=planner_command_dim,
    ).to(device)

    # 打印参数量和配置信息
    planner_params = sum(p.numel() for p in planner.parameters())

    if config.planner.use_z_context:
        input_src = "z_context (first-frame encoder output)"
    elif config.planner.use_observed_tokens:
        input_src = (
            f"z_observed+z_ar (mode={observed_token_mode}, "
            f"observed {config.train.num_observed_frames} frames + predictor output)"
        )
    else:
        input_src = "z_ar (predictor output)"

    if config.train.predictor_inference_consistent:
        status_info = f"status_dim={planner_status_dim} (inference_consistent, command_dim={planner_command_dim})"
    elif config.rl.enabled:
        status_info = f"status_dim={planner_status_dim} (rl:{config.rl.status_mode})"
    elif config.planner.use_states_for_planner:
        status_info = f"status_dim={planner_status_dim} (raw_states)"
    else:
        status_info = f"status_dim={planner_status_dim} (extracted)"

    rl_info = (
        f", actor_critic={enable_rl_actor_critic}, rl_action_dim={config.planner.rl_action_dim}"
        if enable_rl_actor_critic
        else ""
    )

    if config.planner.use_temporal and config.planner.use_z_context:
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"(TemporalPlanner, input={input_src}, num_context_frames={num_context_frames}, "
            f"use_spatial_tokens={config.planner.use_spatial_tokens}, {status_info}{rl_info})"
        )
    elif config.planner.use_temporal:
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"(TemporalPlanner, input={input_src}, num_time_steps={num_time_steps}, "
            f"use_spatial_tokens={config.planner.use_spatial_tokens}, "
            f"temporal_alignment={config.planner.temporal_alignment}, {status_info}{rl_info})"
        )
    else:
        logger.info(
            f"planner_params: {planner_params / 1e6:.2f}M "
            f"(SingleFramePlanner, input={input_src}, "
            f"use_spatial_tokens={config.planner.use_spatial_tokens}, {status_info}{rl_info})"
        )

    return planner
