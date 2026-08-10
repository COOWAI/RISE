# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Clip loading, preprocessing, and action math for trajectory visualization."""

import json
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from scipy.spatial.transform import Rotation

from app.vjepa_cowa_world_model.training.encoder_inputs import build_tubelet_encoder_input
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_actions_3d(states):
    """Build 3D actions: [dx_ego, dy_ego, d_yaw] from states [T, 7]."""
    T = states.shape[0]
    actions = np.zeros((T - 1, 3), dtype=np.float32)
    for t in range(T - 1):
        dx_global = states[t + 1, 0] - states[t, 0]
        dy_global = states[t + 1, 1] - states[t, 1]
        yaw = states[t, 5]
        cos_h = np.cos(-yaw)
        sin_h = np.sin(-yaw)
        dx_ego = cos_h * dx_global - sin_h * dy_global
        dy_ego = sin_h * dx_global + cos_h * dy_global
        d_yaw = states[t + 1, 5] - states[t, 5]
        d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
        actions[t] = np.asarray([dx_ego, dy_ego, d_yaw], dtype=np.float32)
    return actions


def prepare_encoder_input(context_clips: torch.Tensor) -> torch.Tensor:
    """
    与 train_command_v2.forward_context() 保持一致的 encoder 输入构造。

    输入:  [B, C, T, H, W]
    输出:  [B*T, C, tubelet_size, H, W]
    """
    return build_tubelet_encoder_input(context_clips)


def load_sample_data(data_dir, camera="CAM_FRONT", frame_indices=None):
    """
    从单个轨迹目录加载数据
    Args:
        data_dir: 轨迹目录路径
        camera: 相机名称
        frame_indices: 指定加载的帧索引， None则加载全部
    Returns:
        dict: 包含图像、states、extrinsics, 相机参数等
    """
    # 加载 metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    # 加载 HDF5 数据
    h5_path = os.path.join(data_dir, "trajectory.h5")
    with h5py.File(h5_path, "r") as f:
        # cartesian_position: [T, 6] = [x, y, z, roll, pitch, yaw]
        # gripper_position:   [T] 或 [T, 1] = velocity
        cartesian_pos = f["observation/robot_state/cartesian_position"][:]
        velocity = f["observation/robot_state/gripper_position"][:]
        # 确保 velocity 是 [T, 1]
        if velocity.ndim == 1:
            velocity = velocity[:, None]
        states = np.concatenate([cartesian_pos, velocity], axis=1)  # [T, 7]
        if not (states.shape[1] == 7):
            raise AssertionError(f"Expected states dim 7, got {states.shape[1]}")
        # 加载相机外参
        extrinsic_key = f"observation/camera_extrinsics/{camera}_left"
        if extrinsic_key in f:
            extrinsics = f[extrinsic_key][:]
        else:
            # 如果没有该相机的外参， 使用默认值
            logger.warning(f"No extrinsics found for {camera}, using zeros")
            extrinsics = np.zeros((len(states), 7))
    # 加载视频
    video_key = f"{camera.lower()}_mp4_path"
    if video_key not in metadata:
        raise ValueError(f"No video path found for camera {camera}")
    video_path = os.path.join(data_dir, metadata[video_key])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
    vlen = len(vr)
    # 选择帧索引
    if frame_indices is None:
        frame_indices = np.arange(vlen)
    # 读取视频帧
    vr.seek(0)
    frames = vr.get_batch(frame_indices.tolist()).asnumpy()  # [T, H, W, C]
    # 确保 states 和 frames 对齐
    min_len = min(len(frames), len(states))
    if len(frames) != len(states):
        logger.warning(
            f"Frames ({len(frames)}) and states ({len(states)}) length mismatch, " f"truncating to {min_len}"
        )
        frames = frames[:min_len]
        states = states[:min_len]
        extrinsics = extrinsics[:min_len]
    # 获取相机参数
    camera_params = metadata.get("camera_params", {}).get(camera, {})
    if not camera_params:
        raise ValueError(f"No camera params found for {camera}")
    return {
        "frames": frames,  # [T, H, W, C]
        "states": states,  # [T, 7]
        "extrinsics": extrinsics,  # [T, 7]
        "camera_params": camera_params,
        "metadata": metadata,
        "frame_indices": frame_indices,
        "video_fps": vr.get_avg_fps(),
    }


def get_future_pose_start_index(window_start, current_frame, future_start_offset, num_poses):
    """
    计算当前帧对应的未来轨迹起始索引。

    预测轨迹的第 i 个点对应原始视频中的
    window_start + future_start_offset + i 帧。
    当当前帧已经走过这些时刻时，应从可视化与指标中剔除。
    """
    rel_frame = max(0, current_frame - window_start)
    pose_offsets = future_start_offset + np.arange(num_poses)
    pose_start_idx = int(np.searchsorted(pose_offsets, rel_frame, side="right"))
    return min(pose_start_idx, num_poses)


def slice_future_horizon(pred_trajs, gt_traj, pose_start_idx):
    """
    仅保留相对于当前帧仍然在未来的轨迹段。
    """
    if pred_trajs is None or gt_traj is None:
        return pred_trajs, gt_traj

    if pred_trajs.ndim == 3:
        pred_trajs = pred_trajs[:, pose_start_idx:, :]
    else:
        pred_trajs = pred_trajs[pose_start_idx:, :]
    gt_traj = gt_traj[pose_start_idx:, :]
    return pred_trajs, gt_traj


def preprocess_clip(frames_np, crop_size):
    """
    将原始视频帧预处理为模型输入 tensor
    """
    frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2).contiguous().float()
    C_ch, T_fr, H_orig, W_orig = frames_t.shape
    frames_t = frames_t.reshape(C_ch * T_fr, H_orig, W_orig).unsqueeze(0)
    frames_t = F.interpolate(frames_t, size=(crop_size, crop_size), mode="bilinear", align_corners=False)
    frames_t = frames_t.reshape(C_ch, T_fr, crop_size, crop_size)
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(3, 1, 1, 1)
    frames_t = (frames_t - mean) / std
    return frames_t.unsqueeze(0)


def load_trajectory_list(data_path):
    """
    从 txt 文件加载轨迹目录列表
    """
    with open(data_path, "r") as f:
        trajectories = [line.strip() for line in f.readlines() if line.strip()]
    logger.info(f"Loaded {len(trajectories)} trajectories from {data_path}")
    return trajectories


def compute_actions_4d(states):
    """
    计算动作序列 - 4维版本（自动驾驶领域）

    Args:
        states: [T, 7] - [x, y, z, roll, pitch, yaw, velocity]

    Returns:
        actions: [T-1, 4] - [dx, dy, dyaw, velocity]
    """
    T = len(states)
    actions = np.zeros((T - 1, 4))

    for t in range(T - 1):
        dx = states[t + 1, 0] - states[t, 0]
        dy = states[t + 1, 1] - states[t, 1]
        dyaw = states[t + 1, 5] - states[t, 5]
        dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
        velocity = states[t, 6]
        actions[t] = np.array([dx, dy, dyaw, velocity])

    return actions


def compute_actions_7d(states):
    """
    计算动作序列 - 7维版本（机器人领域）

    Args:
        states: [T, 7] - [x, y, z, roll, pitch, yaw, velocity]

    Returns:
        actions: [T-1, 7] - [dx, dy, dz, d_roll, d_pitch, d_yaw, velocity]
    """
    T = len(states)
    actions = np.zeros((T - 1, 7))

    for t in range(T - 1):
        xyz_diff = states[t + 1, :3] - states[t, :3]

        R1 = Rotation.from_euler("xyz", states[t, 3:6]).as_matrix()
        R2 = Rotation.from_euler("xyz", states[t + 1, 3:6]).as_matrix()
        R_diff = R2 @ R1.T
        angle_diff = Rotation.from_matrix(R_diff).as_euler("xyz")

        velocity = states[t, 6]
        actions[t] = np.concatenate([xyz_diff, angle_diff, [velocity]])

    return actions
