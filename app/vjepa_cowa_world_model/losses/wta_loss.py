# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Winner-Takes-All 多模态轨迹损失函数

包含三个版本的 WTA 损失:
- wta_loss (v1): 原版硬标签
- wta_loss_v2: 软标签 + Cover 损失
- wta_loss_v3: Annealed WTA (所有轨迹参与回归)
"""

from typing import Optional

import torch
import torch.nn.functional as F


def _resolve_rank_signal(
    dist_xy: torch.Tensor,
    mode_scores: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return the per-mode ranking signal (higher = better mode).

    With ``mode_scores=None`` this is exactly ``-dist_xy`` (so winner selection
    and soft targets are byte-identical to the legacy trajectory-L2 behaviour).
    When ``mode_scores`` (a composite reward, ``[B, K]``, higher=better) is
    supplied — e.g. from ``losses.reward_selector.compute_mode_rewards`` — the
    winner becomes ``mode_scores.argmax`` and the v2/v3 soft targets follow the
    reward instead of the negative distance.
    """
    if mode_scores is None:
        return -dist_xy
    if mode_scores.shape != dist_xy.shape:
        raise ValueError(
            f"mode_scores must match per-mode distance shape {tuple(dist_xy.shape)}, "
            f"got {tuple(mode_scores.shape)}"
        )
    return mode_scores.to(device=dist_xy.device, dtype=dist_xy.dtype)


def _mean_l1_with_optional_timestep_weights(
    abs_error: torch.Tensor,
    timestep_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reduce absolute trajectory error over time/state dims with optional time weights."""
    if timestep_weights is None:
        return abs_error.mean(dim=(-2, -1))

    num_poses = abs_error.shape[-2]
    if isinstance(timestep_weights, torch.Tensor):
        weights = timestep_weights.to(device=abs_error.device, dtype=abs_error.dtype)
    else:
        weights = torch.tensor(timestep_weights, device=abs_error.device, dtype=abs_error.dtype)
    if weights.ndim != 1 or weights.shape[0] != num_poses:
        raise ValueError(f"timestep_weights must have shape [{num_poses}], got {tuple(weights.shape)}")

    view_shape = [1] * abs_error.ndim
    view_shape[-2] = num_poses
    return (abs_error * weights.view(*view_shape)).mean(dim=(-2, -1))


def _normalize_arc_length_weights(w: torch.Tensor, eps: float, global_batch: bool) -> torch.Tensor:
    """Normalize arc-length weights locally or across all DDP ranks."""

    if not global_batch:
        return w * (w.numel() / (w.sum() + eps))

    dist = getattr(torch, "distributed", None)
    if dist is None or not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return w * (w.numel() / (w.sum() + eps))

    local_sum = w.sum()
    local_count = w.new_tensor(float(w.numel()))
    if local_sum.requires_grad or local_count.requires_grad:
        raise RuntimeError("Arc-length weight all_reduce stats must not require grad; gt_traj should be detached.")
    stats = torch.stack([local_sum, local_count])
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return w * (stats[1] / (stats[0] + eps))


def wta_loss(
    pred_trajs: torch.Tensor,
    pred_conf_logits: torch.Tensor,
    gt_traj: torch.Tensor,
    reg_loss_weight: float = 1.0,
    conf_loss_weight: float = 1.0,
    alpha: float = 5.0,
    eps: float = 1e-6,
    timestep_weights: Optional[torch.Tensor] = None,
    mode_scores: Optional[torch.Tensor] = None,
    global_batch: bool = False,
) -> dict:
    """
    Winner-Takes-All 多模态轨迹损失 (原版 - 硬标签)

    Parameters
    ----------
    pred_trajs      : [B, K, num_poses, 3]  预测轨迹（K 条）
    pred_conf_logits: [B, K]                置信度 logit（未经 softmax）
    gt_traj         : [B, num_poses, 3]     GT 轨迹
    reg_loss_weight : 回归损失权重
    conf_loss_weight: 置信度损失权重
    alpha           : 长度归一化系数
    eps             : 数值稳定
    timestep_weights: [num_poses] 可选时间步回归权重；仅影响回归项
    mode_scores     : [B, K] 可选 reward-based mode 选择信号（higher=better）。
                      None 时退化为按 -ADE 选 winner（与原行为完全一致）；
                      提供时 winner=argmax(mode_scores)（方向 B：reward selector）。

    Returns
    -------
    dict:
        "loss"      : 总损失（scalar）
        "reg_loss"  : WTA 回归损失（scalar）
        "conf_loss" : 置信度 CE 损失（scalar）
        "cover_loss": Cover损失（原版为0）
        "winner_idx": [B] 每个样本的 winner mode 下标（用于 logging）
    """
    B, K, num_poses, _ = pred_trajs.shape

    # ── Step 1: 找 winner ─────────────────────────────────────────────
    # 计算每条预测轨迹与 GT 的 ADE（平均位移误差，只用 xy）
    gt_expanded = gt_traj.unsqueeze(1).expand_as(pred_trajs)  # [B, K, num_poses, 3]

    # ADE: 只用 xy 两维
    dist_xy = torch.norm(
        pred_trajs[..., :2] - gt_expanded[..., :2],
        dim=-1,
    ).mean(
        dim=-1
    )  # [B, K]

    # winner 由 ranking 信号决定：默认 -ADE（argmax(-ADE)==argmin(ADE)），
    # 提供 mode_scores 时改为 reward 最大的 mode。
    rank_signal = _resolve_rank_signal(dist_xy, mode_scores)
    winner_idx = rank_signal.argmax(dim=1)  # [B]

    # ── Step 2: WTA 回归损失（只对 winner 计算）─────────────────────
    # 取出 winner 轨迹
    winner_idx_exp = winner_idx.view(B, 1, 1, 1).expand(B, 1, num_poses, 3)
    winner_traj = pred_trajs.gather(dim=1, index=winner_idx_exp).squeeze(1)
    # winner_traj: [B, num_poses, 3]

    # 长度归一化 L1 损失
    per_sample_l1 = _mean_l1_with_optional_timestep_weights(
        (winner_traj - gt_traj).abs(),
        timestep_weights=timestep_weights,
    )  # [B]
    dxy = gt_traj[:, 1:, :2] - gt_traj[:, :-1, :2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1)  # [B]
    w = 1.0 / (alpha + arc_len)
    w = _normalize_arc_length_weights(w, eps=eps, global_batch=global_batch)
    reg_loss = (w * per_sample_l1).mean()

    # ── Step 3: 置信度损失（CE，winner 类别监督）──────────────────────
    conf_loss = F.cross_entropy(
        pred_conf_logits,
        winner_idx,
        reduction="mean",
    )

    # ── Step 4: 合并 ──────────────────────────────────────────────────
    total = reg_loss_weight * reg_loss + conf_loss_weight * conf_loss

    return {
        "loss": total,
        "reg_loss": reg_loss,
        "conf_loss": conf_loss,
        "cover_loss": torch.tensor(0.0, device=pred_trajs.device),  # 原版无cover损失
        "winner_idx": winner_idx,
    }


def wta_loss_v2(
    pred_trajs: torch.Tensor,
    pred_conf_logits: torch.Tensor,
    gt_traj: torch.Tensor,
    reg_loss_weight: float = 1.0,
    conf_loss_weight: float = 1.0,
    cover_loss_weight: float = 0.1,
    alpha: float = 5.0,
    temperature: float = 1.0,
    eps: float = 1e-6,
    timestep_weights: Optional[torch.Tensor] = None,
    mode_scores: Optional[torch.Tensor] = None,
    global_batch: bool = False,
) -> dict:
    """
    Winner-Takes-All 多模态轨迹损失 (改进版 - 软标签 + Cover损失)

    mode_scores : [B, K] 可选 reward 选择信号（higher=better）。None 时按 -ADE
                  选 winner 且软标签用 softmax(-ADE/T)（原行为）；提供时 winner=
                  argmax(reward) 且软标签用 softmax(reward/T)（方向 B reward selector）。

    改进点:
    1. 使用软标签代替硬标签，提高泛化性
    2. 添加Cover损失，鼓励不同mode覆盖不同的轨迹空间

    Parameters
    ----------
    pred_trajs      : [B, K, num_poses, 3]  预测轨迹（K 条）
    pred_conf_logits: [B, K]                置信度 logit（未经 softmax）
    gt_traj         : [B, num_poses, 3]     GT 轨迹
    reg_loss_weight : 回归损失权重
    conf_loss_weight: 置信度损失权重
    cover_loss_weight: Cover损失权重（鼓励轨迹多样性）
    alpha           : 长度归一化系数
    temperature     : 软标签温度参数，越大越平滑
    eps             : 数值稳定
    timestep_weights: [num_poses] 可选时间步回归权重；仅影响回归项

    Returns
    -------
    dict:
        "loss"      : 总损失（scalar）
        "reg_loss"  : WTA 回归损失（scalar）
        "conf_loss" : 置信度损失（scalar）
        "cover_loss": Cover损失（scalar）
        "winner_idx": [B] 每个样本的 winner mode 下标
    """
    B, K, num_poses, _ = pred_trajs.shape

    # ── Step 1: 计算所有轨迹与GT的距离 ───────────────────────────────
    gt_expanded = gt_traj.unsqueeze(1).expand_as(pred_trajs)  # [B, K, num_poses, 3]

    # ADE: 只用 xy 两维
    dist_xy = torch.norm(
        pred_trajs[..., :2] - gt_expanded[..., :2],
        dim=-1,
    ).mean(
        dim=-1
    )  # [B, K]

    # ── Step 2: Winner选择 ───────────────────────────────────────────
    rank_signal = _resolve_rank_signal(dist_xy, mode_scores)
    winner_idx = rank_signal.argmax(dim=1)  # [B]

    # ── Step 3: 回归损失（只对winner计算）────────────────────────────
    winner_idx_exp = winner_idx.view(B, 1, 1, 1).expand(B, 1, num_poses, 3)
    winner_traj = pred_trajs.gather(dim=1, index=winner_idx_exp).squeeze(1)

    # 长度归一化 L1 损失
    per_sample_l1 = _mean_l1_with_optional_timestep_weights(
        (winner_traj - gt_traj).abs(),
        timestep_weights=timestep_weights,
    )  # [B]
    dxy = gt_traj[:, 1:, :2] - gt_traj[:, :-1, :2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1)  # [B]
    w = 1.0 / (alpha + arc_len)
    w = _normalize_arc_length_weights(w, eps=eps, global_batch=global_batch)
    reg_loss = (w * per_sample_l1).mean()

    # ── Step 4: 软标签置信度损失 ─────────────────────────────────────
    # 使用 ranking 信号的 softmax 作为软标签：默认 -ADE（距离越近权重越高），
    # 提供 mode_scores 时用 reward（reward 越高权重越高）。
    _LOGIT_CLAMP = 50.0
    conf_logits_v2 = (rank_signal / temperature).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
    soft_target = F.softmax(conf_logits_v2, dim=1).detach()  # [B, K] — CE teacher target; detach so the
    # confidence loss only trains pred_conf_logits, not the trajectories/rewards rank_signal is built from

    # 使用交叉熵损失（pred_conf_logits是logits，soft_target是概率分布）
    log_probs = F.log_softmax(pred_conf_logits, dim=1)  # [B, K]
    conf_loss = -(soft_target * log_probs).sum(dim=1).mean()  # scalar

    # ── Step 5: Cover损失 - 鼓励轨迹多样性 ─────────────────────────────
    if K > 1:
        # 计算不同mode之间的轨迹相似度
        # [B, K, num_poses, 3] -> [B, K, num_poses*3]
        traj_flat = pred_trajs.flatten(2)  # [B, K, num_poses*3]

        # L2归一化
        traj_norm = F.normalize(traj_flat, p=2, dim=-1)  # [B, K, num_poses*3]

        # 计算余弦相似度矩阵 [B, K, K]
        sim_matrix = torch.bmm(traj_norm, traj_norm.transpose(1, 2))

        # 移除对角线（自相似度），只考虑不同mode之间的相似度
        mask = 1.0 - torch.eye(K, device=pred_trajs.device).unsqueeze(0)  # [1, K, K]
        off_diag_sim = sim_matrix * mask  # [B, K, K]

        # Cover损失：惩罚过高的相似度
        # 平方操作放大高相似度的惩罚
        cover_loss = (off_diag_sim**2).sum(dim=[1, 2]) / (K * (K - 1))  # [B]
        cover_loss = cover_loss.mean()
    else:
        cover_loss = torch.tensor(0.0, device=pred_trajs.device)

    # ── Step 6: 合并总损失 ───────────────────────────────────────────
    total = reg_loss_weight * reg_loss + conf_loss_weight * conf_loss + cover_loss_weight * cover_loss

    return {
        "loss": total,
        "reg_loss": reg_loss,
        "conf_loss": conf_loss,
        "cover_loss": cover_loss,
        "winner_idx": winner_idx,
    }


def wta_loss_v3(
    pred_trajs: torch.Tensor,
    pred_conf_logits: torch.Tensor,
    gt_traj: torch.Tensor,
    reg_loss_weight: float = 1.0,
    conf_loss_weight: float = 1.0,
    cover_loss_weight: float = 0.1,
    alpha: float = 5.0,
    conf_temperature: float = 1.5,
    awta_temperature: float = 8.0,
    eps: float = 1e-6,
    timestep_weights: Optional[torch.Tensor] = None,
    mode_scores: Optional[torch.Tensor] = None,
    global_batch: bool = False,
) -> dict:
    """
    Annealed Winner-Takes-All 多模态轨迹损失
    (基于 ICRA 2025: "Annealed Winner-Takes-All for Motion Forecasting")

    核心改进：所有 K 条轨迹都参与回归，按距离加权；温度随训练退火。

    与 v1/v2 的关键区别:
    - v1/v2: 只有 winner 收到回归梯度，其余 K-1 条完全无回归信号
    - v3:    所有 K 条轨迹都按 softmax(-dist/T) 加权参与回归
             T 随 epoch 退火：初期均匀训练 → 后期逐渐聚焦 winner

    Parameters
    ----------
    pred_trajs       : [B, K, num_poses, 3]  预测轨迹（K 条）
    pred_conf_logits : [B, K]                置信度 logit（未经 softmax）
    gt_traj          : [B, num_poses, 3]     GT 轨迹
    reg_loss_weight  : 回归损失权重
    conf_loss_weight : 置信度损失权重
    cover_loss_weight: Cover损失权重
    alpha            : 长度归一化系数
    conf_temperature : 置信度软标签温度（固定）
    awta_temperature : 当前退火温度（由外部调度器控制，逐epoch衰减）
    eps              : 数值稳定
    timestep_weights : [num_poses] 可选时间步回归权重；仅影响回归项
    mode_scores      : [B, K] 可选 reward 选择信号（higher=better）。None 时
                       winner/aWTA 权重/软标签均按 -ADE（与原行为一致）；提供时
                       改用 reward（aWTA 回归权重与软标签变为 softmax(reward/T)）。

    Returns
    -------
    dict:
        "loss"      : 总损失（scalar）
        "reg_loss"  : 加权回归损失（scalar）
        "conf_loss" : 置信度损失（scalar）
        "cover_loss": Cover损失（scalar）
        "winner_idx": [B] 每个样本的 winner mode 下标（用于 logging）
    """
    B, K, num_poses, _ = pred_trajs.shape

    # ── Step 1: 计算所有轨迹与GT的距离 ───────────────────────────────
    gt_expanded = gt_traj.unsqueeze(1).expand_as(pred_trajs)  # [B, K, num_poses, 3]

    # TODO(hch): 当前 mode 分配/置信度监督仅基于 xy 距离，而回归项使用 x/y/yaw 三维 L1。
    # 这会导致 mode ranking 与回归目标不完全一致；后续可评估是否将 yaw 显式纳入
    # winner/conf target，或将 yaw 单独加权后并入距离度量。
    # 每条轨迹每个pose的L1误差 → per-mode ADE (xy only)
    dist_xy = torch.norm(
        pred_trajs[..., :2] - gt_expanded[..., :2],
        dim=-1,
    ).mean(
        dim=-1
    )  # [B, K]

    # ranking 信号：默认 -ADE，提供 mode_scores 时用 reward（higher=better）
    rank_signal = _resolve_rank_signal(dist_xy, mode_scores)
    winner_idx = rank_signal.argmax(dim=1)  # [B] for logging

    # ── Step 2: aWTA 加权回归损失（所有mode参与）─────────────────────
    # 核心：softmax(rank_signal / T) 让所有mode按 ranking 获得回归权重
    # T大 → 权重均匀（所有mode平等训练）；T小 → 聚焦winner（接近标准WTA）
    _LOGIT_CLAMP = 50.0
    awta_logits = (rank_signal / awta_temperature).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
    awta_weights = F.softmax(awta_logits, dim=1).detach()  # [B, K] (stop-gradient on weights)

    # 每条轨迹的per-sample L1损失（含长度归一化）
    per_mode_l1 = _mean_l1_with_optional_timestep_weights(
        (pred_trajs - gt_expanded).abs(),
        timestep_weights=timestep_weights,
    )  # [B, K]

    # 长度归一化权重（与v1/v2相同）
    dxy = gt_traj[:, 1:, :2] - gt_traj[:, :-1, :2]
    arc_len = torch.linalg.norm(dxy, dim=-1).sum(dim=1)  # [B]
    w = 1.0 / (alpha + arc_len)
    w = _normalize_arc_length_weights(w, eps=eps, global_batch=global_batch)  # [B]

    # 加权回归损失：每个mode按aWTA权重贡献
    weighted_l1 = (awta_weights * per_mode_l1).sum(dim=1)  # [B]
    reg_loss = (w * weighted_l1).mean()

    # ── Step 3: 软标签置信度损失（与v2相同）──────────────────────────
    conf_logits = (rank_signal / conf_temperature).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
    soft_target = F.softmax(conf_logits, dim=1).detach()  # [B, K] — detached CE teacher target (same as v2)
    log_probs = F.log_softmax(pred_conf_logits, dim=1)  # [B, K]
    conf_loss = -(soft_target * log_probs).sum(dim=1).mean()

    # ── Step 4: Cover损失（与v2相同）─────────────────────────────────
    if K > 1:
        traj_flat = pred_trajs.flatten(2)  # [B, K, num_poses*3]
        traj_norm = F.normalize(traj_flat, p=2, dim=-1)
        sim_matrix = torch.bmm(traj_norm, traj_norm.transpose(1, 2))
        mask = 1.0 - torch.eye(K, device=pred_trajs.device).unsqueeze(0)
        off_diag_sim = sim_matrix * mask
        cover_loss = (off_diag_sim**2).sum(dim=[1, 2]) / (K * (K - 1))
        cover_loss = cover_loss.mean()
    else:
        cover_loss = torch.tensor(0.0, device=pred_trajs.device)

    # ── Step 5: 合并总损失 ───────────────────────────────────────────
    total = reg_loss_weight * reg_loss + conf_loss_weight * conf_loss + cover_loss_weight * cover_loss

    return {
        "loss": total,
        "reg_loss": reg_loss,
        "conf_loss": conf_loss,
        "cover_loss": cover_loss,
        "winner_idx": winner_idx,
    }


def awta_temperature_schedule(
    init_temperature: float,
    epoch: int,
    exp_base: float,
    min_temperature: float = 0.1,
) -> float:
    """
    aWTA 退火温度调度器（指数衰减 + 温度下限）

    Parameters
    ----------
    init_temperature : 初始温度（推荐 8.0~10.0）
    epoch            : 当前 epoch（从 0 开始）
    exp_base         : 衰减底数
                       - 短训练（<50 epochs）: 0.85~0.90
                       - 长训练（300+ epochs）: 0.98~0.99
                       公式: base = (target_final_T / init_T) ^ (1 / total_epochs)
    min_temperature  : 温度下限，防止完全退化为hard WTA（推荐 0.1）

    Returns
    -------
    float: 当前温度 = max(init_temperature * exp_base^epoch, min_temperature)

    温度变化示例（init=8.0, base=0.984, min_T=0.1, 315 epochs）:
        epoch   0 →  T = 8.00  (近似均匀权重，所有mode平等训练)
        epoch  50 →  T = 3.59  (轻微分化)
        epoch 100 →  T = 1.61  (开始明显分化)
        epoch 150 →  T = 0.72  (聚焦好的mode)
        epoch 200 →  T = 0.32  (接近WTA但仍保留多样性)
        epoch 250 →  T = 0.15  (强聚焦winner)
        epoch 300 →  T = 0.10  (下限保护)
    """
    return max(init_temperature * (exp_base**epoch), min_temperature)


def get_loss_function(version: str = "v1"):
    """
    工厂方法：根据版本返回对应的损失函数

    Parameters
    ----------
    version : 损失函数版本 ("v1", "v2", "v3")

    Returns
    -------
    对应的损失函数
    """
    loss_fns = {
        "v1": wta_loss,
        "v2": wta_loss_v2,
        "v3": wta_loss_v3,
    }
    if version not in loss_fns:
        raise ValueError(f"Unknown loss version: {version}. Available: {list(loss_fns.keys())}")
    return loss_fns[version]
