# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
单模型轨迹损失函数（无置信度预测）
"""

from typing import Optional

import torch

from app.vjepa_cowa_world_model.losses.wta_loss import _mean_l1_with_optional_timestep_weights


def single_model_loss(
    pred_trajs: torch.Tensor,
    gt_traj: torch.Tensor,
    reg_loss_weight: float = 1.0,
    alpha: float = 5.0,
    eps: float = 1e-6,
    timestep_weights: Optional[torch.Tensor] = None,
) -> dict:
    """
    单模型轨迹损失 (无置信度预测)

    Parameters
    ----------
    pred_trajs      : [B, 1, num_poses, 3]  预测轨迹（1 条）
    gt_traj         : [B, num_poses, 3]     GT 轨迹
    reg_loss_weight : 回归损失权重
    alpha           : 长度归一化系数
    eps             : 数值稳定
    timestep_weights: [num_poses] 可选时间步回归权重；仅影响回归项

    Returns
    -------
    dict:
        "loss"      : 总损失（scalar）
        "reg_loss"  : 回归损失（scalar）
        "conf_loss" : 0 (scalar, 兼容多模态接口)
        "cover_loss": 0 (scalar, 兼容多模态接口)
        "winner_idx": None (兼容多模态接口)
    """
    B, K, num_poses, _ = pred_trajs.shape
    if not (K == 1):
        raise AssertionError(f"Single model loss requires K=1, got K={K}")

    # 只取第一条轨迹
    pred_traj = pred_trajs.squeeze(1)  # [B, num_poses, 3]

    # 长度归一化 L1 损失
    per_sample_l1 = _mean_l1_with_optional_timestep_weights(
        (pred_traj - gt_traj).abs(),
        timestep_weights=timestep_weights,
    )  # [B]
    dxy = gt_traj[:, 1:, :2] - gt_traj[:, :-1, :2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1)  # [B]
    w = 1.0 / (alpha + arc_len)
    w = w * (w.numel() / (w.sum() + eps))
    reg_loss = (w * per_sample_l1).mean()

    total = reg_loss_weight * reg_loss

    return {
        "loss": total,
        "reg_loss": reg_loss,
        "conf_loss": torch.tensor(0.0, device=pred_trajs.device),
        "cover_loss": torch.tensor(0.0, device=pred_trajs.device),
        "winner_idx": None,
    }


def l1_length_normalized_loss(pred, gt, alpha=5.0, eps=1e-6):
    """
    长度归一化的 L1 损失 (参考 V-JEPA)
    避免短轨迹和长轨迹的权重不平衡

    Parameters
    ----------
    pred   : [B, num_poses, 3]  预测轨迹
    gt     : [B, num_poses, 3]  GT 轨迹
    alpha  : 长度归一化系数
    eps    : 数值稳定

    Returns
    -------
    scalar: 长度归一化后的 L1 损失
    """
    per_sample_l1 = (pred - gt).abs().mean(dim=[1, 2])  # [B]
    dxy = gt[:, 1:, :2] - gt[:, :-1, :2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1)  # [B]
    w = 1.0 / (alpha + arc_len)
    w = w * (w.numel() / (w.sum() + eps))
    return (w * per_sample_l1).mean()
