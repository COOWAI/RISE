# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
EMA (指数移动平均) 模块

提供 EMA 更新相关的功能。
"""

from typing import Callable, Generator

import torch
import torch.nn as nn


def create_momentum_scheduler(
    ema_start: float, ema_end: float, ipe: int, num_epochs: int
) -> Generator[float, None, None]:
    """
    创建 momentum 调度器生成器

    Args:
        ema_start: EMA 起始 momentum
        ema_end: EMA 结束 momentum
        ipe: 每个 epoch 的迭代次数
        num_epochs: 总 epoch 数

    Yields:
        float: 当前 step 的 momentum 值
    """
    total_steps = int(ipe * num_epochs) + 1
    for i in range(total_steps):
        m = ema_start + i * (ema_end - ema_start) / (ipe * num_epochs)
        yield m


def create_ema_update_fn(encoder: nn.Module, target_encoder: nn.Module) -> Callable[[float], None]:
    """
    创建 EMA 更新函数

    Args:
        encoder: encoder 模型 (student)
        target_encoder: target_encoder 模型 (teacher)

    Returns:
        Callable[[float], None]: EMA 更新函数，接收 momentum 参数
    """

    @torch.no_grad()
    def update_ema(m: float) -> None:
        """
        使用动态 momentum 更新 target encoder 的参数
        采用高效的原地操作方式

        Args:
            m: 当前的 momentum 值 (从 momentum_scheduler 获取)
        """
        # 收集所有参数
        params_k = []
        params_q = []
        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
            params_k.append(param_k)
            params_q.append(param_q)

        # 高效的原地操作
        torch._foreach_mul_(params_k, m)
        torch._foreach_add_(params_k, params_q, alpha=1 - m)

    return update_ema


class EMAUpdater:
    """
    EMA 更新器类，封装 EMA 更新逻辑

    使用示例:
        ema_updater = EMAUpdater(encoder, target_encoder, ema_start=0.996, ema_end=0.999)
        for epoch in range(num_epochs):
            for batch in dataloader:
                # ... 训练代码 ...
                ema_updater.step()  # 更新 EMA
    """

    def __init__(
        self,
        encoder: nn.Module,
        target_encoder: nn.Module,
        ema_start: float = 0.996,
        ema_end: float = 0.999,
        ipe: int = 100,
        num_epochs: int = 100,
    ):
        """
        初始化 EMA 更新器

        Args:
            encoder: encoder 模型 (student)
            target_encoder: target_encoder 模型 (teacher)
            ema_start: EMA 起始 momentum
            ema_end: EMA 结束 momentum
            ipe: 每个 epoch 的迭代次数
            num_epochs: 总 epoch 数
        """
        self.encoder = encoder
        self.target_encoder = target_encoder
        self.momentum_scheduler = create_momentum_scheduler(ema_start, ema_end, ipe, num_epochs)
        self._update_fn = create_ema_update_fn(encoder, target_encoder)

    def step(self) -> float:
        """
        执行一步 EMA 更新

        Returns:
            float: 当前使用的 momentum 值
        """
        m = next(self.momentum_scheduler)
        self._update_fn(m)
        return m

    def reset(self) -> None:
        """重置 target_encoder 为 encoder 的状态"""
        self.target_encoder.load_state_dict(self.encoder.state_dict())
