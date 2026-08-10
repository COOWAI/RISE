# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
状态特征提取工具

提供统一的接口从 states 和 actions 提取 planner 状态特征。
支持多种 status_mode 以适应不同的训练配置。
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401

logger = logging.getLogger(__name__)

# Threshold for detecting un-centred UTM coordinates leaking into status features.
# NavSim UTM values are O(10^5 ~ 10^6); centred values are O(10^2) at most.
_UTM_LEAK_WARN_THRESHOLD = 10_000.0


def _warn_if_utm_leak(xy: torch.Tensor, mode_name: str) -> None:
    """Emit a one-time warning if *xy* looks like raw UTM coordinates."""
    with torch.no_grad():
        max_abs = xy.abs().max().item()
    if max_abs > _UTM_LEAK_WARN_THRESHOLD:
        logger.warning(
            "[status_features] mode='%s': detected |x/y|=%.0f > %.0f — "
            "this looks like raw UTM global coordinates leaking into the planner. "
            "Coordinates should be scene-centred (O(100)) before use. "
            "Check navsim_data._build_states() or apply normalisation.",
            mode_name,
            max_abs,
            _UTM_LEAK_WARN_THRESHOLD,
        )


def get_status_dim(status_mode: str, num_context_frames: int = 1) -> int:
    """
    返回 prepare_status_feature 在给定 status_mode 下的输出维度。

    Parameters
    ----------
    status_mode        : 状态特征模式
    num_context_frames : 上下文帧数

    Returns
    -------
    int: 输出特征维度
    """
    if status_mode == "ego_history_sequence":
        return num_context_frames * 3  # [vel, acc, yaw_rate] * T
    elif status_mode == "current_only":
        return 5  # [vel, acc, yaw, x, y]
    elif status_mode == "current_plus_command":
        return 9  # [vel, acc, yaw, x, y] + [4-dim one-hot command]
    elif status_mode == "history_trajectory":
        return num_context_frames * 3 + 2  # ego-centric [dx, dy, dyaw] * T + [vel, acc]
    elif status_mode == "raw_states":
        return 7  # [x, y, z, roll, pitch, yaw, velocity]
    elif status_mode == "first":
        # train_command.py 中 use_states_for_planner=False 时的模式
        return 8  # [velocity, acceleration, yaw, xy(2), action_feat(3)]
    elif status_mode == "last":
        return 8  # 同 first
    elif status_mode == "history_pool":
        return 8  # 同 first
    elif status_mode == "drive_command":
        return 4  # [GO_STRAIGHT, TURN_LEFT, TURN_RIGHT, U_TURN]
    elif status_mode == "drive_command_7":
        return 7  # [drive_cmd(4) + velocity(1) + acceleration(1) + yaw_rate(1)]
    elif status_mode == "drive_command_8":
        return 8  # [drive_cmd(4) + vx + vy + ax + ay]
    else:
        raise ValueError(f"Unknown status_mode: {status_mode}")


def prepare_status_feature(
    states: torch.Tensor,
    actions: torch.Tensor,
    status_mode: str = "current_only",
    num_context_frames: int = 1,
    straight_thresh: float = 0.3,
    uturn_thresh: float = 2.5,
    # 兼容旧版参数
    mode: str = None,
    use_states_for_planner: bool = False,
    action_dim: int = 3,
    # NavSim 原始字段（可选）
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    从 states 提取 planner 状态特征，统一输出 [B, status_dim]。

    Parameters
    ----------
    states             : [B, T, 7]  每帧 [x, y, z, roll, pitch, yaw, velocity]
    actions            : [B, T-1, action_dim]  （部分模式未使用，保留接口兼容）
    status_mode        : 状态特征模式
        - "ego_history_sequence": 多帧 [velocity, acceleration, yaw_rate]，对齐 num_context_frames
        - "current_only":         当前帧 [velocity, acceleration, yaw, x, y]
        - "current_plus_command": current_only + 基于历史帧 yaw 趋势的 drive_command (4-dim)
        - "history_trajectory":   ego-centric 历史轨迹 [dx, dy, dyaw] * T + [velocity, acceleration]
        - "raw_states":           当前帧原始 states [x, y, z, roll, pitch, yaw, velocity]
        - "first":                起始时刻特征 [velocity, acceleration, yaw, x, y, action_feat]
        - "last":                 最后时刻特征
        - "history_pool":         时序均值池化
        - "drive_command":        纯 one-hot drive command [4]
        - "drive_command_7":      drive command + velocity + acceleration + yaw_rate [7]
    num_context_frames : 上下文帧数
    straight_thresh    : 直行判定阈值
    uturn_thresh       : U-turn 判定阈值

    兼容旧版参数 (deprecated):
    mode               : status_mode 的别名，优先级低于 status_mode
    use_states_for_planner : 是否使用 states 作为 planner 输入
    action_dim         : action 维度

    NavSim 原始字段（可选，传入时优先使用）:
    driving_command    : [B, T, 4] 或 [B, 4]  原始导航指令 one-hot
    ego_dynamics       : [B, T, 4]  原始 [vx, vy, ax, ay]

    Returns
    -------
    torch.Tensor: [B, status_dim]
    """
    # 兼容旧版参数：mode -> status_mode
    if mode is not None and status_mode == "current_only":
        status_mode = mode

    # 兼容旧版逻辑：use_states_for_planner
    if not use_states_for_planner and status_mode in ["first", "last", "history_pool"]:
        # 旧版行为：use_states_for_planner=False 时返回 8 维特征
        pass  # _prepare_status_feature_legacy 已经处理这种情况
    B = states.shape[0]
    T = states.shape[1]
    ncf = min(num_context_frames, T)
    cur_idx = ncf - 1  # "当前帧" = context 窗口的最后一帧

    if status_mode == "ego_history_sequence":
        # 每帧: [velocity, acceleration, yaw_rate]
        feats = []
        for t in range(ncf):
            vel = states[:, t, 6:7]  # [B, 1]
            if t > 0:
                acc = states[:, t, 6:7] - states[:, t - 1, 6:7]
                dyaw = torch.atan2(
                    torch.sin(states[:, t, 5:6] - states[:, t - 1, 5:6]),
                    torch.cos(states[:, t, 5:6] - states[:, t - 1, 5:6]),
                )
            else:
                acc = torch.zeros_like(vel)
                dyaw = torch.zeros_like(vel)
            feats.append(torch.cat([vel, acc, dyaw], dim=-1))  # [B, 3]
        return torch.cat(feats, dim=-1)  # [B, ncf * 3]

    elif status_mode == "current_only":
        vel = states[:, cur_idx, 6:7]
        acc = (states[:, cur_idx, 6:7] - states[:, cur_idx - 1, 6:7]) if cur_idx > 0 else torch.zeros_like(vel)
        yaw = states[:, cur_idx, 5:6]
        xy = states[:, cur_idx, 0:2]
        _warn_if_utm_leak(xy, "current_only")
        return torch.cat([vel, acc, yaw, xy], dim=-1)  # [B, 5]

    elif status_mode == "current_plus_command":
        # current_only 部分
        vel = states[:, cur_idx, 6:7]
        acc = (states[:, cur_idx, 6:7] - states[:, cur_idx - 1, 6:7]) if cur_idx > 0 else torch.zeros_like(vel)
        yaw = states[:, cur_idx, 5:6]
        xy = states[:, cur_idx, 0:2]
        _warn_if_utm_leak(xy, "current_plus_command")
        ego_feat = torch.cat([vel, acc, yaw, xy], dim=-1)  # [B, 5]

        # 基于历史帧 yaw 趋势估计 drive_command（不依赖未来帧）
        yaw_start = states[:, 0, 5]
        yaw_cur = states[:, cur_idx, 5]
        delta_yaw = torch.atan2(torch.sin(yaw_cur - yaw_start), torch.cos(yaw_cur - yaw_start))
        abs_delta = torch.abs(delta_yaw)

        cmd = torch.zeros(B, 4, device=states.device, dtype=states.dtype)
        cmd[:, 0] = (abs_delta < straight_thresh).float()
        cmd[:, 1] = ((delta_yaw > straight_thresh) & (abs_delta < uturn_thresh)).float()
        cmd[:, 2] = ((delta_yaw < -straight_thresh) & (abs_delta < uturn_thresh)).float()
        cmd[:, 3] = (abs_delta >= uturn_thresh).float()

        return torch.cat([ego_feat, cmd], dim=-1)  # [B, 9]

    elif status_mode == "history_trajectory":
        # 以当前帧为原点，转换历史帧 pose 到 ego-centric 坐标系
        # Use float64 for position diffs to guard against large UTM values.
        cur_x = states[:, cur_idx, 0].double()  # [B]
        cur_y = states[:, cur_idx, 1].double()  # [B]
        cur_yaw = states[:, cur_idx, 5].double()  # [B]
        cos_h = torch.cos(-cur_yaw)
        sin_h = torch.sin(-cur_yaw)

        pos_x = states[:, :, 0].double()  # [B, T]
        pos_y = states[:, :, 1].double()  # [B, T]

        traj_feats = []
        for t in range(ncf):
            dx_world = pos_x[:, t] - cur_x
            dy_world = pos_y[:, t] - cur_y
            # 旋转到 ego-centric
            dx_ego = cos_h * dx_world - sin_h * dy_world  # [B]
            dy_ego = sin_h * dx_world + cos_h * dy_world  # [B]
            dyaw = torch.atan2(
                torch.sin(states[:, t, 5] - cur_yaw.float()),
                torch.cos(states[:, t, 5] - cur_yaw.float()),
            )  # [B]
            traj_feats.append(torch.stack([dx_ego.float(), dy_ego.float(), dyaw], dim=-1))  # [B, 3]
        traj_flat = torch.cat(traj_feats, dim=-1)  # [B, ncf * 3]

        vel = states[:, cur_idx, 6:7]
        acc = (states[:, cur_idx, 6:7] - states[:, cur_idx - 1, 6:7]) if cur_idx > 0 else torch.zeros_like(vel)
        return torch.cat([traj_flat, vel, acc], dim=-1)  # [B, ncf * 3 + 2]

    elif status_mode == "raw_states":
        _warn_if_utm_leak(states[:, cur_idx, 0:2], "raw_states")
        return states[:, cur_idx, :]  # [B, 7]

    elif status_mode in ["first", "last", "history_pool"]:
        # 兼容 train_command.py 中的 prepare_status_feature 实现
        return _prepare_status_feature_legacy(states, actions, status_mode)

    elif status_mode == "drive_command":
        return _prepare_drive_command(
            states,
            num_observed=num_context_frames,
            action_dim=4,
            straight_thresh=straight_thresh,
            uturn_thresh=uturn_thresh,
            driving_command=driving_command,
        )

    elif status_mode == "drive_command_7":
        return _prepare_drive_command_7(
            states,
            num_observed=num_context_frames,
            straight_thresh=straight_thresh,
            uturn_thresh=uturn_thresh,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
        )

    elif status_mode == "drive_command_8":
        return _prepare_drive_command_8(
            states,
            num_observed=num_context_frames,
            straight_thresh=straight_thresh,
            uturn_thresh=uturn_thresh,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
        )

    else:
        raise ValueError(f"Unknown status_mode: {status_mode}")


def _prepare_status_feature_legacy(
    states: torch.Tensor,
    actions: torch.Tensor,
    mode: str = "first",
) -> torch.Tensor:
    """
    兼容 train_command.py 中的 prepare_status_feature 实现。

    mode:
        - "last": 使用最后一个时刻
        - "first": 使用起始时刻
        - "history_pool": 对时序做均值池化
    """
    B = states.shape[0]

    if mode == "first":
        idx = 0
        velocity = states[:, idx, 6:7]
        if states.shape[1] >= 2:
            acceleration = states[:, 1, 6:7] - states[:, 0, 6:7]
        else:
            acceleration = torch.zeros_like(velocity)
        yaw = states[:, idx, 5:6]
        xy = states[:, idx, 0:2]
        if actions is not None and actions.shape[1] > 0:
            action_feat = actions[:, 0, :3]
        else:
            action_feat = torch.zeros(B, 3, device=states.device, dtype=states.dtype)
    elif mode == "history_pool":
        velocity = states[:, :, 6:7].mean(dim=1)
        if states.shape[1] >= 2:
            dv = states[:, 1:, 6:7] - states[:, :-1, 6:7]
            acceleration = dv.mean(dim=1)
        else:
            acceleration = torch.zeros_like(velocity)
        yaw = states[:, :, 5:6].mean(dim=1)
        xy = states[:, :, 0:2].mean(dim=1)
        if actions is not None and actions.shape[1] > 0:
            action_feat = actions[:, :, :3].mean(dim=1)
        else:
            action_feat = torch.zeros(B, 3, device=states.device, dtype=states.dtype)
    else:  # "last"
        velocity = states[:, -1, 6:7]  # [B, 1]
        if states.shape[1] >= 2:
            acceleration = states[:, -1, 6:7] - states[:, -2, 6:7]
        else:
            acceleration = torch.zeros_like(velocity)
        yaw = states[:, -1, 5:6]  # [B, 1]
        xy = states[:, -1, 0:2]  # [B, 2]
        if actions is not None and actions.shape[1] > 0:
            action_feat = actions[:, -1, :3]
        else:
            action_feat = torch.zeros(B, 3, device=states.device, dtype=states.dtype)

    _warn_if_utm_leak(xy, f"legacy/{mode}")
    return torch.cat([velocity, acceleration, yaw, xy, action_feat], dim=-1)  # [B, 8]


def _prepare_drive_command(
    states: torch.Tensor,
    *,
    num_observed: int,
    action_dim: int = 4,
    straight_thresh: float = 0.3,
    uturn_thresh: float = 2.5,
    driving_command: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    计算 drive_command（基于 delta_yaw 首尾差）。

    当 driving_command 不为 None 时，直接使用原始导航指令，跳过 yaw 差分分类。

    Parameters
    ----------
    states          : [B, T, 7]
    action_dim      : 输出维度 (4 或 7)
    straight_thresh : 直行判定阈值
    uturn_thresh    : U-turn 判定阈值
    driving_command : [B, T, 4] 或 [B, 4] 原始导航指令 one-hot（可选）

    Returns
    -------
    torch.Tensor: [B, action_dim]
    """
    B = states.shape[0]

    if driving_command is not None:
        # 使用原始导航指令
        if driving_command.ndim == 3:
            # 取当前帧（最后观测帧）的导航指令；取首帧会让 cmd 滞后 (num_observed-1) 帧，
            # 在命令切换（路口前后）时与 GT 未来轨迹矛盾。
            cmd_idx = min(num_observed - 1, driving_command.shape[1] - 1)
            cmd = driving_command[:, cmd_idx, :4]  # [B, 4]
        else:
            cmd = driving_command[:, :4]  # [B, 4]
        cmd = cmd.to(device=states.device, dtype=states.dtype)
        if action_dim > 4:
            pad = torch.zeros(B, action_dim - 4, device=states.device, dtype=states.dtype)
            cmd = torch.cat([cmd, pad], dim=-1)
        return cmd  # [B, action_dim]

    yaw = states[:, :, 5]  # [B, T]
    yaw_start = yaw[:, 0]  # [B]
    yaw_end = yaw[:, -1]  # [B]
    delta_yaw = torch.atan2(torch.sin(yaw_end - yaw_start), torch.cos(yaw_end - yaw_start))  # [B]

    abs_delta = torch.abs(delta_yaw)
    drive_cmd = torch.zeros(B, action_dim, device=states.device, dtype=states.dtype)
    drive_cmd[:, 0] = (abs_delta < straight_thresh).float()  # GO_STRAIGHT
    drive_cmd[:, 1] = ((delta_yaw > straight_thresh) & (abs_delta < uturn_thresh)).float()  # TURN_LEFT
    drive_cmd[:, 2] = ((delta_yaw < -straight_thresh) & (abs_delta < uturn_thresh)).float()  # TURN_RIGHT
    drive_cmd[:, 3] = (abs_delta >= uturn_thresh).float()  # U_TURN

    return drive_cmd  # [B, action_dim]


def _prepare_drive_command_7(
    states: torch.Tensor,
    num_observed: int = 2,
    straight_thresh: float = 0.3,
    uturn_thresh: float = 2.5,
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
    use_drive_command: bool = True,
) -> torch.Tensor:
    """
    构建推理一致的 status 向量（单帧版本，用于 planner）。

    当 driving_command 不为 None 时，直接使用原始导航指令替代 yaw 差分分类。
    当 ego_dynamics 不为 None 时，使用原始 vx 和 ax 替代离散差分。

    Parameters
    ----------
    states          : [B, T, 7]
    num_observed    : 已观测的帧数
    straight_thresh : 直行判定阈值
    uturn_thresh    : U-turn 判定阈值
    driving_command : [B, T, 4] 或 [B, 4] 原始导航指令 one-hot（可选）
    ego_dynamics    : [B, T, 4] 原始 [vx, vy, ax, ay]（可选）
    use_drive_command : False 时去掉前 4 维 cmd，返回 [B, 3]

    返回: [B, 7] - [go_straight, turn_left, turn_right, u_turn, velocity, acceleration, yaw_rate]
          或 [B, 3] - [velocity, acceleration, yaw_rate]（use_drive_command=False）
    """
    B = states.shape[0]

    # --- Drive command ---
    if use_drive_command:
        # fail-loud: 禁止用 yaw 差分伪造导航指令（yaw[:, -1] 会泄露未来帧），缺失即报错。
        if driving_command is None:
            raise ValueError(
                "_prepare_drive_command_7: use_drive_command=True 但 driving_command 缺失。"
                "请确保 dataloader/metadata 提供 driving_command；禁止用 yaw 差分伪造（且会泄露未来帧 yaw）。"
            )
        if driving_command.ndim == 3:
            # 取当前帧（最后观测帧）的导航指令，与下方 velocity/yaw_rate 的取帧一致；
            # 取首帧会让 cmd 滞后 (num_observed-1) 帧，在命令切换（路口前后）时与 GT 未来轨迹矛盾。
            cmd_idx = min(num_observed - 1, driving_command.shape[1] - 1)
            cmd = driving_command[:, cmd_idx, :4]  # [B, 4]
        else:
            cmd = driving_command[:, :4]  # [B, 4]
        cmd = cmd.to(device=states.device, dtype=states.dtype)
        go_straight = cmd[:, 0]
        turn_left = cmd[:, 1]
        turn_right = cmd[:, 2]
        u_turn = cmd[:, 3]

    # --- Velocity & Acceleration ---
    if ego_dynamics is not None:
        ego_dyn = ego_dynamics.to(device=states.device, dtype=states.dtype)
        obs_idx = min(num_observed - 1, ego_dyn.shape[1] - 1)
        velocity = ego_dyn[:, obs_idx, 0]  # vx
        acceleration = ego_dyn[:, obs_idx, 2]  # ax
    else:
        # 7 维 status 的 velocity 直接来自 states 第 7 维（真实速度，非伪造），accel 用观测帧内差分。
        velocity = states[:, num_observed - 1, 6]
        if num_observed >= 2:
            acceleration = states[:, num_observed - 1, 6] - states[:, num_observed - 2, 6]
        else:
            acceleration = torch.zeros(B, device=states.device, dtype=states.dtype)

    # --- Yaw rate (始终从 yaw 差分计算，ego_dynamic_state 中无此字段) ---
    if num_observed >= 2:
        yaw_rate = torch.atan2(
            torch.sin(states[:, num_observed - 1, 5] - states[:, num_observed - 2, 5]),
            torch.cos(states[:, num_observed - 1, 5] - states[:, num_observed - 2, 5]),
        )
    else:
        yaw_rate = torch.zeros(B, device=states.device, dtype=states.dtype)

    if use_drive_command:
        return torch.stack(
            [go_straight, turn_left, turn_right, u_turn, velocity, acceleration, yaw_rate], dim=-1
        )  # [B, 7]
    return torch.stack([velocity, acceleration, yaw_rate], dim=-1)  # [B, 3]


def _prepare_drive_command_8(
    states: torch.Tensor,
    num_observed: int = 2,
    straight_thresh: float = 0.3,
    uturn_thresh: float = 2.5,
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
    use_drive_command: bool = True,
) -> torch.Tensor:
    """
    构建 8 维推理一致的 status 向量。

    相比 7 维版本新增: vy, ay（来自 ego_dynamics）。
    移除了 14 维版本中的 yaw_rate, roll, pitch, δx_ego, δy_ego, δyaw。
    所有特征仅使用 observed 范围内的状态，不触及未来帧的 GT 信息。

    Parameters
    ----------
    states          : [B, T, 7]  每帧 [x, y, z, roll, pitch, yaw, velocity]
    num_observed    : 已观测的帧数
    straight_thresh : 直行判定阈值
    uturn_thresh    : U-turn 判定阈值
    driving_command : [B, T, 4] 或 [B, 4] 原始导航指令 one-hot（可选）
    ego_dynamics    : [B, T, 4] 原始 [vx, vy, ax, ay]（可选）
    use_drive_command : False 时去掉前 4 维 cmd，返回 [B, 4]

    Returns
    -------
    torch.Tensor
        [B, 8] = [go_straight, turn_left, turn_right, u_turn, vx, vy, ax, ay]
        或 [B, 4] = [vx, vy, ax, ay]（use_drive_command=False）
    """
    obs_idx = min(num_observed - 1, states.shape[1] - 1)

    # --- Drive command (same as 7-dim version) ---
    if use_drive_command:
        if driving_command is None:
            raise ValueError(
                "_prepare_drive_command_8: use_drive_command=True 但 driving_command 缺失。"
                "请确保 dataloader/metadata 提供 driving_command；禁止用 yaw 差分伪造（且会泄露未来帧 yaw）。"
            )
        if driving_command.ndim == 3:
            # 取当前帧（最后观测帧）的导航指令，与 obs_idx 取帧约定一致（见 _prepare_drive_command_7）。
            cmd_idx = min(obs_idx, driving_command.shape[1] - 1)
            cmd = driving_command[:, cmd_idx, :4]
        else:
            cmd = driving_command[:, :4]
        cmd = cmd.to(device=states.device, dtype=states.dtype)
        go_straight = cmd[:, 0]
        turn_left = cmd[:, 1]
        turn_right = cmd[:, 2]
        u_turn = cmd[:, 3]

    # --- vx, vy, ax, ay ---
    # fail-loud: 8 维 status 需要真实 vx/vy/ax/ay，缺 ego_dynamics 不得用 speed/差分伪造（vy/ay 只能伪造成 0）。
    if ego_dynamics is None:
        raise ValueError(
            "_prepare_drive_command_8: ego_dynamics 缺失。8 维 status 需要真实 [vx, vy, ax, ay]，"
            "禁止用 states 速度/差分伪造（会把 vy/ay 置零）。请确保 dataloader 提供 ego_dynamics [B, T, 4]。"
        )
    ego_dyn = ego_dynamics.to(device=states.device, dtype=states.dtype)
    dyn_idx = min(obs_idx, ego_dyn.shape[1] - 1)
    vx = ego_dyn[:, dyn_idx, 0]
    vy = ego_dyn[:, dyn_idx, 1]
    ax = ego_dyn[:, dyn_idx, 2]
    ay = ego_dyn[:, dyn_idx, 3]

    if use_drive_command:
        return torch.stack(
            [go_straight, turn_left, turn_right, u_turn, vx, vy, ax, ay],
            dim=-1,
        )  # [B, 8]
    return torch.stack(
        [vx, vy, ax, ay],
        dim=-1,
    )  # [B, 4]


def _prepare_drive_command_12(
    states: torch.Tensor,
    num_observed: int = 2,
    straight_thresh: float = 0.3,
    uturn_thresh: float = 2.5,
    driving_command: Optional[torch.Tensor] = None,
    ego_dynamics: Optional[torch.Tensor] = None,
    use_drive_command: bool = True,
) -> torch.Tensor:
    """
    构建 12 维推理一致的 status 向量（planner 专用）。

    相比 8 维版本新增 4 维相对 pose：
        [x_local, y_local, sin(yaw_local), cos(yaw_local)]

    其中 x_local/y_local 是 ego 坐标系下相对 **frame 0（场景起点）** 的位移；
    sin/cos(yaw_local) 是航向变化的三角编码（避免 ±π 跳变）。
    states 已场景居中（navsim_data.py 减去了 frame 0 原点），
    故 states[:, obs_idx, 0:2] 直接就是相对 frame 0 的世界坐标位移。

    与 GT trajectory ego 变换（origin=当前观测帧）在时间方向上正交互补：
    输入 pose 代表"过去→现在"的累积历史；GT 代表"现在→未来"的预测。

    Parameters
    ----------
    states          : [B, T, 7]  每帧 [x, y, z, roll, pitch, yaw, velocity]
    num_observed    : 已观测帧数
    straight_thresh : 直行判定阈值
    uturn_thresh    : U-turn 判定阈值
    driving_command : [B, T, 4] 或 [B, 4] 原始导航指令 one-hot（可选）
    ego_dynamics    : [B, T, 4] 原始 [vx, vy, ax, ay]（可选）

    Returns
    -------
    Returns 8 when use_drive_command=False
    torch.Tensor
        [B, 12] = [cmd(4), vx, vy, ax, ay, x_local, y_local, sin_yaw_local, cos_yaw_local]
    """
    base = _prepare_drive_command_8(
        states,
        num_observed=num_observed,
        straight_thresh=straight_thresh,
        uturn_thresh=uturn_thresh,
        driving_command=driving_command,
        ego_dynamics=ego_dynamics,
        use_drive_command=use_drive_command,
    )  # [B, 8] or [B, 4]

    obs_idx = min(num_observed - 1, states.shape[1] - 1)
    cur_x = states[:, obs_idx, 0]
    cur_y = states[:, obs_idx, 1]
    cur_yaw = states[:, obs_idx, 5]
    first_yaw = states[:, 0, 5]

    cos_h = torch.cos(-cur_yaw)
    sin_h = torch.sin(-cur_yaw)
    x_local = cos_h * cur_x - sin_h * cur_y
    y_local = sin_h * cur_x + cos_h * cur_y
    yaw_delta = cur_yaw - first_yaw
    sin_yaw_local = torch.sin(yaw_delta)
    cos_yaw_local = torch.cos(yaw_delta)

    return torch.cat(
        [
            base,
            x_local.unsqueeze(-1),
            y_local.unsqueeze(-1),
            sin_yaw_local.unsqueeze(-1),
            cos_yaw_local.unsqueeze(-1),
        ],
        dim=-1,
    )  # [B, 12]


def resolve_planner_status_dim(config) -> int:
    """解析 planner status 维度：planner.status_dim > 0 优先，否则回落 train.state_dim。

    用于解耦 predictor.state_dim（固定 8 维保持 checkpoint 兼容）与
    planner.status_dim（可选扩展到 12 维以增加几何上下文）。
    """
    # planner 始终是 PlannerConfig dataclass（status_dim 默认 0），直接索引而非 getattr 兜底。
    # 语义保留：status_dim>0 时优先，否则回落 train.state_dim（解耦 predictor.state_dim 与 planner.status_dim）。
    sd = config.planner.status_dim
    return sd if sd and sd > 0 else config.train.state_dim


def resolve_planner_use_drive_command(config) -> bool:
    planner_flag = getattr(getattr(config, "planner", None), "use_drive_command", None)
    if planner_flag is None:
        return bool(getattr(getattr(config, "train", None), "use_drive_command", True))
    return bool(planner_flag)


def resolve_effective_planner_status_dim(config) -> int:
    raw_dim = resolve_planner_status_dim(config)
    use_drive_command = resolve_planner_use_drive_command(config)
    return effective_status_dim(raw_dim, use_drive_command=use_drive_command)


def prepare_inference_consistent_states(
    states,
    num_observed=2,
    straight_thresh=0.3,
    uturn_thresh=2.5,
    driving_command=None,
    ego_dynamics=None,
    state_dim=7,
    use_drive_command=True,
):
    """
    构建推理一致的 state 输入，委托给对应的 _prepare_drive_command_* 函数。

    Args:
        states: [B, T, 7] - [x, y, z, roll, pitch, yaw, velocity]
        num_observed: 已观测的 frame 数量
        straight_thresh: 直行判定阈值
        uturn_thresh: U-turn 判定阈值
        driving_command: [B, T, 4] 或 [B, 4] 原始导航指令 one-hot（可选）
        ego_dynamics: [B, T, 4] 原始 [vx, vy, ax, ay]（可选）
        state_dim: 原始维度 (7, 8 或 12)
        use_drive_command: False 时去掉 drive_command 4 维

    Returns:
        [B, T, effective_dim] - 在所有 temporal 位置 replicate 相同的向量
    """
    B, T, _ = states.shape

    if state_dim == 7:
        state_vec = _prepare_drive_command_7(
            states,
            num_observed=num_observed,
            straight_thresh=straight_thresh,
            uturn_thresh=uturn_thresh,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            use_drive_command=use_drive_command,
        )
    elif state_dim == 8:
        state_vec = _prepare_drive_command_8(
            states,
            num_observed=num_observed,
            straight_thresh=straight_thresh,
            uturn_thresh=uturn_thresh,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            use_drive_command=use_drive_command,
        )
    elif state_dim == 12:
        state_vec = _prepare_drive_command_12(
            states,
            num_observed=num_observed,
            straight_thresh=straight_thresh,
            uturn_thresh=uturn_thresh,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            use_drive_command=use_drive_command,
        )
    else:
        # fail-loud: 不再把任意 state_dim 静默当作 7 维处理。
        raise ValueError(
            f"prepare_inference_consistent_states: unsupported state_dim={state_dim}, must be one of 7/8/12"
        )

    D = state_vec.shape[-1]
    return state_vec.unsqueeze(1).expand(B, T, D).contiguous()


def mask_future_actions(actions, num_known_actions):
    """
    将未来 actions 置零，只保留已知的 historical actions。

    Parameters
    ----------
    actions           : [B, T, action_dim]  完整 action 序列
    num_known_actions : 已知 action 数（= num_observed_frames - 1）

    Returns
    -------
    tuple[Tensor, Tensor]:
        masked_actions : [B, T, action_dim]  前 num_known_actions 个保持不变，其余为 0
        action_mask    : [B, T] bool         True = 未知/被 mask 的位置，False = 已知位置
    """
    B, T = actions.shape[:2]
    masked = torch.zeros_like(actions)
    action_mask = torch.ones(B, T, dtype=torch.bool, device=actions.device)
    if num_known_actions > 0:
        n = min(num_known_actions, T)
        masked[:, :n] = actions[:, :n]
        action_mask[:, :n] = False
    return masked, action_mask


def effective_status_dim(state_dim: int, use_drive_command: bool = True) -> int:
    """计算实际 status 维度：use_drive_command=False 时去掉 4 维 cmd。"""
    return state_dim if use_drive_command else state_dim - 4


def build_future_gt_trajectory_from_states(
    states: torch.Tensor,
    num_observed_frames: int,
    num_poses: Optional[int] = None,
) -> torch.Tensor:
    """根据 states 构建以最后观测帧为原点的未来 GT 轨迹。"""
    if states.ndim != 3:
        raise ValueError(f"Expected states shape [B, T, D], got {tuple(states.shape)}")
    if states.shape[-1] < 6:
        raise ValueError(f"Expected states last dim >= 6, got {states.shape[-1]}")
    if num_observed_frames < 1:
        raise ValueError(f"num_observed_frames must be >= 1, got {num_observed_frames}")

    batch_size, total_frames, _ = states.shape
    future_start_idx = min(num_observed_frames, total_frames)
    origin_idx = future_start_idx - 1

    if future_start_idx >= total_frames:
        empty = torch.zeros(batch_size, 0, 3, device=states.device, dtype=states.dtype)
        return empty[:, :num_poses] if num_poses is not None else empty

    states_se2 = states[:, :, [0, 1, 5]].double()
    origin_x = states_se2[:, origin_idx, 0]
    origin_y = states_se2[:, origin_idx, 1]
    origin_yaw = states_se2[:, origin_idx, 2]

    dx = states_se2[:, future_start_idx:, 0] - origin_x[:, None]
    dy = states_se2[:, future_start_idx:, 1] - origin_y[:, None]
    dyaw = states_se2[:, future_start_idx:, 2] - origin_yaw[:, None]

    cos_h = torch.cos(-origin_yaw)
    sin_h = torch.sin(-origin_yaw)
    ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
    ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
    ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

    gt_trajectory = torch.stack([ego_x, ego_y, ego_yaw], dim=-1).to(dtype=states.dtype)
    if num_poses is not None:
        gt_trajectory = gt_trajectory[:, :num_poses]
    return gt_trajectory


def build_observed_action_trajectory_history(
    actions: torch.Tensor,
    num_observed_frames: int,
    action_history_dim: int = 3,
    dt: float = 1.0,
) -> torch.Tensor:
    """将观测 action 前缀累计为以最后观测帧为原点的历史轨迹特征。"""
    if actions.ndim != 3:
        raise ValueError(f"Expected actions shape [B, T, A], got {tuple(actions.shape)}")
    if num_observed_frames < 1:
        raise ValueError(f"num_observed_frames must be >= 1, got {num_observed_frames}")
    if action_history_dim not in (3, 4, 6):
        raise ValueError(f"Unsupported action_history_dim={action_history_dim}; expected one of (3, 4, 6)")
    if action_history_dim == 6 and dt <= 0:
        raise ValueError(f"dt must be > 0 when action_history_dim=6, got {dt}")

    batch_size, total_steps, action_dim = actions.shape
    observed_action_steps = min(max(num_observed_frames - 1, 0), total_steps)
    available_frames = observed_action_steps + 1
    poses = torch.zeros(batch_size, available_frames, 3, device=actions.device, dtype=actions.dtype)

    if observed_action_steps > 0:
        if action_dim >= 6:
            # 当前调用传入的是 [dx, dy, dyaw] action，不会进入 action_dim >= 6 分支；
            # 该分支仅保留给未来 state-layout action（yaw 在第 5 列）的兼容路径。
            dx = actions[:, :observed_action_steps, 0]
            dy = actions[:, :observed_action_steps, 1]
            dyaw = actions[:, :observed_action_steps, 5]
        elif action_dim >= 3:
            dx = actions[:, :observed_action_steps, 0]
            dy = actions[:, :observed_action_steps, 1]
            dyaw = actions[:, :observed_action_steps, 2]
        else:
            raise ValueError(f"Unsupported action_dim={action_dim}; expected >= 3")

        for step_idx in range(observed_action_steps):
            prev_pose = poses[:, step_idx]
            prev_yaw = prev_pose[:, 2]
            cos_yaw = torch.cos(prev_yaw)
            sin_yaw = torch.sin(prev_yaw)

            poses[:, step_idx + 1, 0] = prev_pose[:, 0] + cos_yaw * dx[:, step_idx] - sin_yaw * dy[:, step_idx]
            poses[:, step_idx + 1, 1] = prev_pose[:, 1] + sin_yaw * dx[:, step_idx] + cos_yaw * dy[:, step_idx]
            next_yaw = prev_yaw + dyaw[:, step_idx]
            poses[:, step_idx + 1, 2] = torch.atan2(torch.sin(next_yaw), torch.cos(next_yaw))

    anchor_pose = poses[:, -1]
    anchor_x = anchor_pose[:, 0]
    anchor_y = anchor_pose[:, 1]
    anchor_yaw = anchor_pose[:, 2]
    cos_anchor = torch.cos(-anchor_yaw)
    sin_anchor = torch.sin(-anchor_yaw)

    dx_world = poses[:, :, 0] - anchor_x[:, None]
    dy_world = poses[:, :, 1] - anchor_y[:, None]
    history = torch.zeros_like(poses)
    history[:, :, 0] = cos_anchor[:, None] * dx_world - sin_anchor[:, None] * dy_world
    history[:, :, 1] = sin_anchor[:, None] * dx_world + cos_anchor[:, None] * dy_world
    rel_yaw = poses[:, :, 2] - anchor_yaw[:, None]
    history[:, :, 2] = torch.atan2(torch.sin(rel_yaw), torch.cos(rel_yaw))

    if available_frames == num_observed_frames:
        base_history = history
    else:
        base_history = torch.zeros(batch_size, num_observed_frames, 3, device=actions.device, dtype=actions.dtype)
        pad_count = num_observed_frames - available_frames
        base_history[:, :pad_count] = history[:, :1]
        base_history[:, pad_count:] = history

    if action_history_dim == 3:
        return base_history

    x = base_history[:, :, 0]
    y = base_history[:, :, 1]
    yaw = base_history[:, :, 2]
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    if action_history_dim == 4:
        return torch.stack([x, y, cos_yaw, sin_yaw], dim=-1)

    vx = torch.zeros_like(x)
    vy = torch.zeros_like(y)
    if base_history.shape[1] > 1:
        vx[:, 1:] = (x[:, 1:] - x[:, :-1]) / dt
        vy[:, 1:] = (y[:, 1:] - y[:, :-1]) / dt
        vx[:, 0] = vx[:, 1]
        vy[:, 0] = vy[:, 1]

    return torch.stack([x, y, vx, vy, cos_yaw, sin_yaw], dim=-1)


def prepare_inference_consistent_status_vector(
    states,
    num_observed=2,
    straight_thresh=0.3,
    uturn_thresh=2.5,
    driving_command=None,
    ego_dynamics=None,
    state_dim=7,
    use_drive_command=True,
):
    """
    构建推理一致的 status 向量（单帧版本，用于 planner）。

    state_dim=7 时等价于 _prepare_drive_command_7；
    state_dim=8 时委托 _prepare_drive_command_8；
    state_dim=12 时委托 _prepare_drive_command_12（含 x_local/y_local/sin_yaw/cos_yaw）。

    Args:
        states: [B, T, 7] - [x, y, z, roll, pitch, yaw, velocity]
        num_observed: 已观测的帧数
        straight_thresh: 直行判定阈值
        uturn_thresh: U-turn 判定阈值
        driving_command: [B, T, 4] 或 [B, 4] 原始导航指令 one-hot（可选）
        ego_dynamics: [B, T, 4] 原始 [vx, vy, ax, ay]（可选）
        state_dim: 输出维度 (7, 8 或 12)

    Returns:
        [B, state_dim]
    """
    if state_dim == 8:
        return _prepare_drive_command_8(
            states,
            num_observed=num_observed,
            straight_thresh=straight_thresh,
            uturn_thresh=uturn_thresh,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            use_drive_command=use_drive_command,
        )
    if state_dim == 12:
        return _prepare_drive_command_12(
            states,
            num_observed=num_observed,
            straight_thresh=straight_thresh,
            uturn_thresh=uturn_thresh,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            use_drive_command=use_drive_command,
        )
    return _prepare_drive_command_7(
        states,
        num_observed=num_observed,
        straight_thresh=straight_thresh,
        uturn_thresh=uturn_thresh,
        driving_command=driving_command,
        ego_dynamics=ego_dynamics,
        use_drive_command=use_drive_command,
    )
