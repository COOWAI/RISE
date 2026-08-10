"""Phase 3: world-model 导向的候选轨迹优化器(轻量,无参数,非 RefinementDecoder)。

机制:候选轨迹的增量 ``delta`` 为叶子变量 → 可微 rollout(冻结 world model)
→ 冻结 reward head 打分(Phase 1 联合训练的 ``PredictorRewardHead``,**不依赖
h_target → 推理时可用**)+ comfort 罚项 → ``∇_delta objective`` 梯度上升 N 步,
L∞ 信赖域约束(xy/yaw 分别限幅)防止脱离候选邻域。

为可测试性与解耦,rollout 与 reward 以 **callable 注入**:
- ``rollout_fn(trajs [B,K,P,3]) -> z_hat [B,K,N,D]``(须可微,内部勿 detach)
- ``reward_fn(z_hat [B,K,N,D]) -> reward [B,K]``(higher = better)

训练接入注意(v1 刻意不做):若直接把优化后的候选替换进 WTA,优化产物对
planner 参数**无梯度**(delta 路径不经过 planner),会静默切断 planner 训练。
v1 仅在推理/验证期使用(对比优化前后 ADE/FDE);train-time 接入需
straight-through(``traj + (opt - traj.detach())``)蒸馏,留作后续。

Fail-loud:objective 对 delta 无梯度(rollout 内部 detach / 图断裂)或梯度
非有限 → raise,绝不静默跳过优化。
"""

from typing import Callable, Dict, Optional, Tuple

import torch


def wm_guided_optimize_trajectories(
    trajs: torch.Tensor,
    rollout_fn: Callable[[torch.Tensor], torch.Tensor],
    reward_fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    steps: int,
    lr: float,
    trust_radius_xy: float,
    trust_radius_yaw: float,
    comfort_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    comfort_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """对候选轨迹做 N 步 reward 梯度上升(信赖域内)。

    Parameters
    ----------
    trajs           : [B, K, P, 3] 初始候选(内部 detach,不回传给生成器)
    rollout_fn      : 可微 world-model rollout(candidate → future latent)
    reward_fn       : latent → per-mode reward [B, K](higher=better)
    steps / lr      : 梯度上升步数与步长
    trust_radius_xy : xy 增量的 L∞ 上限(米)
    trust_radius_yaw: yaw 增量的 L∞ 上限(弧度)
    comfort_fn      : 可选 [B,K,P,3] → [B,K] 风险(higher=worse),罚项
    comfort_weight  : comfort 罚项权重

    Returns
    -------
    (optimized_trajs [B,K,P,3] detached, diagnostics)
    diagnostics: reward_before / reward_after / delta_xy_max / delta_yaw_max
    """
    if trajs.ndim != 4 or trajs.shape[-1] != 3:
        raise ValueError(f"trajs must be [B, K, P, 3], got {tuple(trajs.shape)}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if lr <= 0:
        raise ValueError(f"lr must be > 0, got {lr}")
    if trust_radius_xy <= 0 or trust_radius_yaw <= 0:
        raise ValueError(f"trust radii must be > 0, got xy={trust_radius_xy}, yaw={trust_radius_yaw}")
    if comfort_weight != 0.0 and comfort_fn is None:
        raise ValueError("comfort_weight != 0 requires comfort_fn (fail-loud)。")

    base = trajs.detach()
    delta = torch.zeros_like(base, requires_grad=True)

    def _objective(candidate: torch.Tensor) -> torch.Tensor:
        z_hat = rollout_fn(candidate)
        reward = reward_fn(z_hat)
        if reward.shape != candidate.shape[:2]:
            raise ValueError(f"reward_fn must return [B, K]={tuple(candidate.shape[:2])}, got {tuple(reward.shape)}")
        objective = reward.mean()
        if comfort_weight != 0.0:
            objective = objective - float(comfort_weight) * comfort_fn(candidate).mean()
        return objective

    # 验证/推理环境常在 no_grad 下,显式启用局部 grad。
    with torch.enable_grad():
        reward_before = float(_objective(base).detach())
        for _ in range(int(steps)):
            candidate = base + delta
            objective = _objective(candidate)
            if not objective.requires_grad:
                # rollout/reward 整条路径被 detach,objective 已无 grad_fn——
                # torch.autograd.grad 会抛通用错误,这里先给出可定位的诊断。
                raise RuntimeError(
                    "wm_guided_traj_optimizer: objective 对 delta 无梯度——rollout/reward "
                    "路径被 detach 或图断裂(fail-loud,禁止静默跳过优化)。"
                )
            (grad,) = torch.autograd.grad(objective, delta, allow_unused=True)
            if grad is None:
                raise RuntimeError(
                    "wm_guided_traj_optimizer: objective 对 delta 无梯度——rollout/reward "
                    "路径被 detach 或图断裂(fail-loud,禁止静默跳过优化)。"
                )
            if not torch.isfinite(grad).all():
                raise RuntimeError("wm_guided_traj_optimizer: 梯度含 NaN/Inf(fail-loud)。")
            with torch.no_grad():
                new_delta = delta + float(lr) * grad
                new_delta[..., :2].clamp_(-float(trust_radius_xy), float(trust_radius_xy))
                new_delta[..., 2].clamp_(-float(trust_radius_yaw), float(trust_radius_yaw))
            delta = new_delta.requires_grad_(True)
        optimized = (base + delta).detach()
        reward_after = float(_objective(optimized).detach())

    diagnostics = {
        "reward_before": reward_before,
        "reward_after": reward_after,
        "delta_xy_max": float(delta.detach()[..., :2].abs().max()),
        "delta_yaw_max": float(delta.detach()[..., 2].abs().max()),
    }
    return optimized, diagnostics
