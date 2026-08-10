# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Pure-numpy geometry primitives for trajectory visualization.

Ego<->camera<->image projection and ego-frame SE2 re-referencing, extracted
verbatim from the (now deprecated) ``visualize_trajectory_on_image.py`` so that
the active visualization CLIs no longer reach into ``app.vjepa_cowa_planner``.

No torch / cv2 / decord dependency — numpy only.
"""

import numpy as np


def get_camera_transform_matrices(camera_params):
    """
    从相机参数提取变换矩阵

    Args:
        camera_params: dict, 包含 intrinsics, T_cam2ego, R_cam2ego

    Returns:
        dict: intrinsic [3,3], R_ego2cam [3,3], T_ego2cam [3]
    """
    intrinsic = np.array(camera_params["intrinsics"])  # [3, 3]
    T_cam2ego = np.array(camera_params["T_cam2ego"])  # [3]
    R_cam2ego = np.array(camera_params["R_cam2ego"])  # [3, 3]

    # Ego -> Camera 变换
    R_ego2cam = R_cam2ego.T
    T_ego2cam = -R_ego2cam @ T_cam2ego

    return {
        "intrinsic": intrinsic,
        "R_ego2cam": R_ego2cam,
        "T_ego2cam": T_ego2cam,
        "R_cam2ego": R_cam2ego,
        "T_cam2ego": T_cam2ego,
    }


def transform_ego_to_camera(points_ego, R_ego2cam, T_ego2cam):
    """
    将 ego-centric 坐标转换到相机坐标系

    Args:
        points_ego: [N, 3] ego坐标系下的点 (x, y, z)
        R_ego2cam: [3, 3] 旋转矩阵
        T_ego2cam: [3] 平移向量

    Returns:
        points_cam: [N, 3] 相机坐标系下的点
    """
    points_cam = (R_ego2cam @ points_ego.T).T + T_ego2cam
    return points_cam


def transform_camera_to_image(points_cam, intrinsic, image_shape=None, eps=1e-3):
    """
    将相机坐标系下的点投影到图像平面

    Args:
        points_cam: [N, 3] 相机坐标系下的点
        intrinsic: [3, 3] 相机内参矩阵
        image_shape: (H, W) 图像尺寸，用于计算有效掩码
        eps: 深度阈值

    Returns:
        pixels: [N, 2] 图像坐标 (u, v)
        valid_mask: [N] bool, 是否在视野内
    """
    # 透视投影: pixel = intrinsic @ point_cam, pixel_2d = pixel[:2] / pixel[2]
    pixels_homo = (intrinsic @ points_cam.T).T  # [N, 3]

    # 深度有效性检查
    depths = pixels_homo[:, 2]
    valid_depth = depths > eps

    # 透视除法
    pixels = pixels_homo[:, :2] / np.maximum(depths[:, None], eps)

    # 图像边界检查
    valid_mask = valid_depth.copy()
    if image_shape is not None:
        img_h, img_w = image_shape
        valid_mask = (
            valid_mask & (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)
        )

    return pixels, valid_mask


def transform_trajectory_to_image(traj_ego, camera_params, image_shape=None):
    """
    将 ego-centric 轨迹投影到图像平面

    Args:
        traj_ego: [num_poses, 3] ego-centric (x, y, yaw)
        camera_params: dict with intrinsics, T_cam2ego, R_cam2ego
        image_shape: (H, W) 图像尺寸

    Returns:
        pixels: [num_poses, 2] 图像坐标
        valid_mask: [num_poses] 是否在视野内
    """
    # 获取变换矩阵
    transforms = get_camera_transform_matrices(camera_params)
    intrinsic = transforms["intrinsic"]
    R_ego2cam = transforms["R_ego2cam"]
    T_ego2cam = transforms["T_ego2cam"]

    # 将轨迹点转换为 3D 点 (z=0, 地面平面)
    num_poses = traj_ego.shape[0]
    points_ego = np.zeros((num_poses, 3))
    points_ego[:, :2] = traj_ego[:, :2]  # x, y
    points_ego[:, 2] = 0.0  # z = 0 (地面)

    # Ego -> Camera
    points_cam = transform_ego_to_camera(points_ego, R_ego2cam, T_ego2cam)

    # Camera -> Image
    pixels, valid_mask = transform_camera_to_image(points_cam, intrinsic, image_shape)

    return pixels, valid_mask


def reref_trajectory_ego(traj_ego, origin_state, current_state):
    """
    将 ego-centric 轨迹从 origin_state 参考帧重新参考到 current_state 帧。

    推理时轨迹是相对于滑动窗口起始帧的 ego 坐标系，但渲染第 t 帧时需要转换到
    第 t 帧的 ego 坐标系，才能正确投影到当前帧的相机图像上。

    变换过程: Ego(origin) -> World -> Ego(current)

    Args:
        traj_ego: [num_poses, 3] 或 [K, num_poses, 3]，ego-centric (x, y, yaw)，相对于 origin_state
        origin_state: [7] 原始参考帧状态 [x, y, z, roll, pitch, yaw, velocity]
        current_state: [7] 当前帧状态

    Returns:
        traj_new: 同形状，相对于 current_state 的 ego-centric 轨迹
    """
    ox, oy, oyaw = origin_state[0], origin_state[1], origin_state[5]
    cx, cy, cyaw = current_state[0], current_state[1], current_state[5]

    # 支持 [num_poses, 3] 和 [K, num_poses, 3] 两种输入
    orig_shape = traj_ego.shape
    if traj_ego.ndim == 3:
        traj_flat = traj_ego.reshape(-1, 3)
    else:
        traj_flat = traj_ego

    # Step 1: Ego(origin) -> World
    cos_o = np.cos(oyaw)
    sin_o = np.sin(oyaw)
    world_x = cos_o * traj_flat[:, 0] - sin_o * traj_flat[:, 1] + ox
    world_y = sin_o * traj_flat[:, 0] + cos_o * traj_flat[:, 1] + oy
    world_yaw = traj_flat[:, 2] + oyaw

    # Step 2: World -> Ego(current)
    dx = world_x - cx
    dy = world_y - cy
    cos_c = np.cos(-cyaw)
    sin_c = np.sin(-cyaw)
    new_x = cos_c * dx - sin_c * dy
    new_y = sin_c * dx + cos_c * dy
    new_yaw = np.arctan2(np.sin(world_yaw - cyaw), np.cos(world_yaw - cyaw))

    traj_new = np.stack([new_x, new_y, new_yaw], axis=-1)
    return traj_new.reshape(orig_shape)
