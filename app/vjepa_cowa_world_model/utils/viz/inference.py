# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Encoder + predictor rollout for trajectory visualization.

Uses the shared predictor auxiliary-input policy helper so visualizations match training.
"""

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.predictor_stepping import (
    make_predictor_step_fn,
    predictor_autoregressive_rollout,
)
from app.vjepa_cowa_world_model.utils import (
    prepare_inference_consistent_status_vector,
    prepare_status_feature,
    resolve_planner_status_dim,
    resolve_planner_use_drive_command,
)
from app.vjepa_cowa_world_model.utils.viz.data import prepare_encoder_input


def run_inference(
    encoder,
    predictor,
    planner,
    frames,
    states,
    actions,
    extrinsics,
    device,
    dtype,
    cfg,
    states_full=None,
    actions_full=None,
    extrinsics_full=None,
):
    """
    运行推理获取预测轨迹.
    支持 train_command.py / train_giant_first.py 的推理配置:
    - predictor_inference_consistent: 推理一致模式
    - predictor_no_aux_input: predictor 不使用辅助输入
    - num_observed_frames: 观测帧数

    Args:
        cfg: TrainingConfig 结构化配置对象
    """
    # 从 TrainingConfig 读取配置
    tokens_per_frame = cfg.data.tokens_per_frame
    num_target_frames = cfg.data.num_target_frames
    normalize_reps = cfg.loss.normalize_reps
    status_mode = cfg.planner.states_mode
    # Predictor 配置
    predictor_inference_consistent = cfg.train.predictor_inference_consistent
    num_observed_frames = cfg.train.num_observed_frames
    num_poses = num_target_frames - num_observed_frames
    action_dim = cfg.train.action_dim
    # Planner 配置
    use_z_context = cfg.planner.use_z_context
    use_states_for_planner = cfg.planner.use_states_for_planner
    use_observed_tokens = cfg.planner.use_observed_tokens
    # 移动数据到设备
    frames = frames.to(device, dtype=dtype)
    states = states.to(device, dtype=torch.float)
    if actions is not None:
        actions = actions.to(device, dtype=torch.float)
    if extrinsics is not None:
        extrinsics = extrinsics.to(device, dtype=torch.float)
    else:
        # 当 extrinsics 为 None 时，创建全零 tensor（predictor 在 use_extrinsics=False 时不使用它）
        extrinsics = torch.zeros(states.shape[0], states.shape[1], 7, device=device, dtype=torch.float)
    if states_full is not None:
        states_full = states_full.to(device, dtype=torch.float)
    if actions_full is not None:
        actions_full = actions_full.to(device, dtype=torch.float)
    if extrinsics_full is not None:
        extrinsics_full = extrinsics_full.to(device, dtype=torch.float)
    B, C, T, H, W = frames.shape
    with torch.cuda.amp.autocast(dtype=dtype, enabled=True):
        # ==================== Encoder Forward ====================
        encoder_input = prepare_encoder_input(frames)
        z_context_out = encoder([encoder_input])
        z = z_context_out[0]
        z = z.view(B, T, -1, z.size(-1)).flatten(1, 2)
        if normalize_reps:
            z = F.layer_norm(z, (z.size(-1),))
        if not (z.shape[1] == frames.shape[2] * tokens_per_frame):
            raise AssertionError(
                f"z shape mismatch: {z.shape}, expected second dim to be " f"{frames.shape[2] * tokens_per_frame}"
            )
        if not (actions.shape[1] == frames.shape[2] - 1):
            raise AssertionError(
                f"actions shape mismatch: {actions.shape}, expected second dim to be " f"{frames.shape[2] - 1}"
            )
        if not (states.shape[1] == frames.shape[2]):
            raise AssertionError(
                f"states shape mismatch: {states.shape}, expected second dim to be " f"{frames.shape[2]}"
            )
        if not (extrinsics.shape[1] == frames.shape[2]):
            raise AssertionError(
                f"extrinsics shape mismatch: {extrinsics.shape}, expected second dim to be " f"{frames.shape[2]}"
            )
        # ==================== Predictor Forward ====================
        num_obs = num_observed_frames

        _step_predictor = make_predictor_step_fn(
            predictor,
            cfg,
            num_obs,
            normalize_reps=normalize_reps,
        )

        # Teacher forcing
        _z_enc = z[:, :-tokens_per_frame]
        _s, _e = states[:, :-1], extrinsics[:, :-1]
        z_tf = _step_predictor(_z_enc, actions, _s, _e)
        # Autoregressive rollout (shared with val_command via predictor_autoregressive_rollout).
        num_total = z.size(1) // tokens_per_frame
        z_ar = predictor_autoregressive_rollout(
            _step_predictor,
            z,
            actions,
            states,
            extrinsics,
            num_obs=num_obs,
            tokens_per_frame=tokens_per_frame,
            num_total=num_total,
            predictor_inference_consistent=predictor_inference_consistent,
            z_tf=z_tf,
        )
        # ==================== Planner Forward ====================
        if predictor_inference_consistent:
            status_feature = prepare_inference_consistent_status_vector(
                states,
                num_observed=num_observed_frames,
                state_dim=resolve_planner_status_dim(cfg),
                use_drive_command=resolve_planner_use_drive_command(cfg),
            )
        else:
            status_feature = prepare_status_feature(
                states,
                actions,
                mode=status_mode,
                use_states_for_planner=use_states_for_planner,
                action_dim=action_dim,
            )
        z_first_frame = z[:, :tokens_per_frame] if use_z_context else None
        if use_observed_tokens:
            z_observed = z[:, : num_observed_frames * tokens_per_frame]
        else:
            z_observed = None
        planner_output = planner(
            z_ar,
            status_feature,
            z_context=z_first_frame,
            z_observed=z_observed,
        )

        pred_trajs = planner_output["trajectories"]  # [B, K, num_poses, 3]
        confidences = planner_output["confidences"]  # [B, K]
        # ==================== GT 轨迹转换 ====================
        StateSE2_indices = [0, 1, 5]
        states_se2 = states[:, :, StateSE2_indices]
        origin_x = states_se2[:, 0, 0]
        origin_y = states_se2[:, 0, 1]
        origin_yaw = states_se2[:, 0, 2]
        future_start_idx = num_observed_frames if predictor_inference_consistent else 1
        dx = states_se2[:, future_start_idx:, 0] - origin_x[:, None]
        dy = states_se2[:, future_start_idx:, 1] - origin_y[:, None]
        dyaw = states_se2[:, future_start_idx:, 2] - origin_yaw[:, None]
        cos_h = torch.cos(-origin_yaw)
        sin_h = torch.sin(-origin_yaw)
        ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
        ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
        ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
        gt_traj = torch.stack([ego_x, ego_y, ego_yaw], dim=-1)  # [B, T_ds-1, 3]
        gt_traj = gt_traj[:, :num_poses]  # [B, num_poses, 3]
    return pred_trajs, confidences, gt_traj
