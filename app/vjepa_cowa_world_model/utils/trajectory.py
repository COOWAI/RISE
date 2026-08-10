# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
轨迹工具函数
"""

import torch


def select_best_trajectory(
    pred_trajs: torch.Tensor,
    conf_logits: torch.Tensor,
) -> torch.Tensor:
    """
    推理时按最高置信度选出最优轨迹。

    Parameters
    ----------
    pred_trajs  : [B, K, num_poses, 3]
    conf_logits : [B, K]

    Returns
    -------
    best_traj   : [B, num_poses, 3]
    """
    best_idx = conf_logits.argmax(dim=1)  # [B]
    best_idx_exp = best_idx.view(-1, 1, 1, 1).expand(-1, 1, pred_trajs.shape[2], pred_trajs.shape[3])
    return pred_trajs.gather(1, best_idx_exp).squeeze(1)  # [B, num_poses, 3]


def _normalize_confidence_weights(conf_values: torch.Tensor) -> torch.Tensor:
    row_sums = conf_values.sum(dim=1, keepdim=True)
    looks_like_prob = (
        bool(torch.all(conf_values >= 0).item())
        and bool(torch.all(conf_values <= 1).item())
        and torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    if looks_like_prob:
        return conf_values / row_sums.clamp_min(1e-8)
    return conf_values.softmax(dim=1)


def confidence_weighted_trajectory(
    pred_trajs: torch.Tensor,
    conf_logits: torch.Tensor,
) -> torch.Tensor:
    """按置信度 softmax 对多模态轨迹做加权平均。"""
    weights = _normalize_confidence_weights(conf_logits).unsqueeze(-1).unsqueeze(-1)
    return (pred_trajs * weights).sum(dim=1)


def trajectory_to_control_action(
    traj: torch.Tensor,
    current_speed: torch.Tensor,
    current_steer: torch.Tensor,
    dt: float,
    wheel_base: float,
    action_low: torch.Tensor = None,
    action_high: torch.Tensor = None,
) -> torch.Tensor:
    """
    将 ego-centric 轨迹近似映射为 HUGSIM 控制量 `[steer_rate, acc]`。

    假设轨迹坐标系为 `x=forward, y=left`，使用第一步目标点做纯跟踪近似。
    """
    if traj.ndim != 3 or traj.shape[-1] < 2:
        raise ValueError(f"Expected traj shape [B, T, >=2], got {tuple(traj.shape)}")

    target = traj[:, 0, :2]
    forward = target[:, 0]
    left = target[:, 1]

    distance = torch.sqrt(forward.square() + left.square()).clamp_min(1e-4)
    desired_speed = distance / max(float(dt), 1e-4)
    acc = (desired_speed - current_speed) / max(float(dt), 1e-4)

    curvature = 2.0 * left / distance.square().clamp_min(1e-4)
    desired_steer = torch.atan(wheel_base * curvature)
    steer_rate = (desired_steer - current_steer) / max(float(dt), 1e-4)

    action = torch.stack([steer_rate, acc], dim=-1)

    if action_low is not None:
        action = torch.maximum(action, action_low.view(1, -1))
    if action_high is not None:
        action = torch.minimum(action, action_high.view(1, -1))

    return action
