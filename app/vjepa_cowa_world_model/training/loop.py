# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
训练循环工具模块

提供训练循环中的通用工具函数。
"""

from __future__ import annotations

import gc
import hashlib
import os
import random
import time
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from src.utils.logging import AverageMeter, get_logger

logger = get_logger(__name__)


def make_dataloader_generator(*, rank: int, stream: str) -> torch.Generator:
    """Create a deterministic private RNG stream for a DataLoader.

    DataLoader iterator construction consumes one random number to derive worker
    seeds even when ``num_workers=0``.  Supplying a private generator prevents
    validation iterator creation from advancing the training-side global RNG.
    """
    if not isinstance(stream, str) or not stream.strip():
        raise ValueError("dataloader generator stream must be a non-empty string")

    payload = f"{torch.initial_seed()}:{int(rank)}:{stream}".encode("utf-8")
    seed = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)
    return torch.Generator().manual_seed(seed)


def seed_dataloader_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn：每个 worker 的 numpy/random 种子派生自 torch 的 per-worker 种子，
    使数据增强在重启后可复现，且各 worker 互不相同（否则多 worker 的 numpy/random 会产生相同的增强序列）。"""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def is_epoch_validation_due(val_loader, epoch: int, val_freq) -> bool:
    """Whether validation should run this epoch.

    Single source shared by all training lines. Safe for val_freq=0/None (returns False)
    instead of crashing on ``(epoch + 1) % val_freq`` (ZeroDivisionError / TypeError).
    """
    if val_loader is None or val_freq is None:
        return False
    val_freq = int(val_freq)
    return val_freq > 0 and (int(epoch) + 1) % val_freq == 0


def create_loss_meters() -> Dict[str, AverageMeter]:
    """
    创建所有损失记录器

    Returns:
        Dict[str, AverageMeter]: 损失记录器字典
    """
    return {
        "loss": AverageMeter(),
        "jloss": AverageMeter(),
        "sloss": AverageMeter(),
        "seg_loss": AverageMeter(),
        "mask_loss": AverageMeter(),
        "dice_loss": AverageMeter(),
        "traj_loss": AverageMeter(),
        "reg_loss": AverageMeter(),
        "conf_loss": AverageMeter(),
        "cover_loss": AverageMeter(),
        "vel_loss": AverageMeter(),
        "yaw_loss": AverageMeter(),
        "cls_valid_ratio": AverageMeter(),
        "sigreg_loss": AverageMeter(),
        "ortho_loss": AverageMeter(),
        "iter_time": AverageMeter(),
        "gpu_time": AverageMeter(),
        "data_load_time": AverageMeter(),
    }


def init_training_loop(
    dataloader: DataLoader,
    sampler: DistributedSampler,
    start_epoch: int,
    skip_batches: int = -1,
    sync_gc: bool = False,
) -> Tuple[Iterator, Dict[str, AverageMeter]]:
    """
    初始化训练循环

    Args:
        dataloader: 数据加载器
        sampler: 分布式采样器
        start_epoch: 开始的 epoch
        skip_batches: 跳过的批次数
        sync_gc: 是否同步垃圾回收

    Returns:
        Tuple[Iterator, Dict[str, AverageMeter]]: (数据迭代器, 损失记录器)
    """
    sampler.set_epoch(start_epoch)
    loader = iter(dataloader)

    # 跳过指定批次
    if skip_batches > 0:
        logger.info(f"主人，跳过 {skip_batches} 个批次")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"跳过 {itr}/{skip_batches} 批次")
            try:
                _ = next(loader)
            except StopIteration:
                loader = iter(dataloader)
                _ = next(loader)

    # 垃圾回收设置
    if sync_gc:
        gc.disable()
        gc.collect()

    loss_meters = create_loss_meters()
    return loader, loss_meters


def get_next_batch(
    loader: Iterator, dataloader: DataLoader, sampler: DistributedSampler, epoch: int, max_retries: int = 5
) -> Tuple[Iterator, Any, bool]:
    """
    获取下一个批次，处理 StopIteration 和异常

    当迭代器耗尽时，会刷新 sampler 的 epoch 并创建新的迭代器。
    调用方必须使用返回的 loader 更新自己的引用，因为耗尽后
    函数内部创建的新迭代器不会自动反映到调用方的变量上。

    Args:
        loader: 数据迭代器
        dataloader: 数据加载器
        sampler: 分布式采样器
        epoch: 当前 epoch
        max_retries: 最大重试次数

    Returns:
        Tuple[Iterator, Any, bool]: (loader, sample, success)
            loader   - 当前（可能已刷新的）迭代器，调用方应更新自身引用
            sample   - 数据样本
            success  - 是否成功获取
    """
    iter_retries = 0
    while True:
        try:
            sample = next(loader)
            return loader, sample, True
        except StopIteration:
            logger.info("主人，数据加载器已耗尽，正在刷新...")
            sampler.set_epoch(epoch)
            loader = iter(dataloader)
        except Exception as e:
            if iter_retries < max_retries:
                logger.warning(f"主人，加载数据时遇到异常 (重试次数 {iter_retries}):\n{e}")
                iter_retries += 1
                time.sleep(5)
            else:
                logger.error(f"主人，数据加载超过最大重试次数 ({max_retries})，抛出异常终止训练。")
                raise e


def load_clips(
    sample: Any, device: torch.device, use_segmentation: bool = True, dtype: torch.dtype = torch.float
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[Any],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """
    加载剪辑数据

    Args:
        sample: 数据样本（tuple from collate_fn）
        device: 设备
        use_segmentation: 是否使用分割
        dtype: 数据类型

    Returns:
        Tuple: (context_frames, actions, states, extrinsics, seg_targets,
                driving_command, ego_dynamics)
            driving_command: [B, T, 4] 原始导航指令 one-hot，NavSim 数据集返回；其他数据集为 None
            ego_dynamics:    [B, T, 4] 原始 [vx, vy, ax, ay]，NavSim 数据集返回；其他数据集为 None
    """
    context_frames = sample[0].to(device, non_blocking=True)  # [B, C, 2, H, W]
    actions = sample[1].to(device, dtype=dtype, non_blocking=True)  # [B, 15, 7]
    states = sample[2].to(device, dtype=dtype, non_blocking=True)  # [B, 16, 7]
    extrinsics = sample[3].to(device, dtype=dtype, non_blocking=True)  # [B, 16, 7]

    # 分割标注
    seg_targets = None
    if use_segmentation and len(sample) > 4:
        seg_targets = sample[4]
        if isinstance(seg_targets, list):
            for i in range(len(seg_targets)):
                if "labels" in seg_targets[i]:
                    seg_targets[i]["labels"] = seg_targets[i]["labels"].to(device, non_blocking=True)
                if "masks" in seg_targets[i]:
                    seg_targets[i]["masks"] = seg_targets[i]["masks"].to(device, non_blocking=True)

    # NavSim 原始字段（collate tuple 末尾追加，其他数据集不返回）
    driving_command = None
    if len(sample) > 5 and sample[5] is not None:
        driving_command = sample[5].to(device, dtype=dtype, non_blocking=True)  # [B, T, 4]

    ego_dynamics = None
    if len(sample) > 6 and sample[6] is not None:
        ego_dynamics = sample[6].to(device, dtype=dtype, non_blocking=True)  # [B, T, 4]

    return context_frames, actions, states, extrinsics, seg_targets, driving_command, ego_dynamics


def maybe_run_gc(itr: int, freq: int = 50, sync_gc: bool = False) -> None:
    """
    根据频率运行垃圾回收

    Args:
        itr: 当前迭代
        freq: 垃圾回收频率
        sync_gc: 是否启用同步垃圾回收
    """
    if sync_gc and (itr + 1) % freq == 0:
        logger.info("主人，正在运行垃圾回收...")
        gc.collect()


class TrainingTimer:
    """
    训练计时器

    使用示例:
        timer = TrainingTimer()
        timer.start_iteration()
        # ... 数据加载 ...
        timer.record_data_load()
        # ... 训练步骤 ...
        iter_time, data_load_time = timer.stop_iteration()
    """

    def __init__(self):
        self.iter_start_time = 0.0
        self.data_load_time = 0.0

    def start_iteration(self) -> None:
        """开始新迭代的计时"""
        self.iter_start_time = time.time()

    def record_data_load(self) -> float:
        """记录数据加载时间"""
        self.data_load_time = (time.time() - self.iter_start_time) * 1000.0
        return self.data_load_time

    def stop_iteration(self) -> Tuple[float, float]:
        """
        停止迭代计时

        Returns:
            Tuple[float, float]: (iter_time_ms, data_load_time_ms)
        """
        iter_elapsed_time_ms = (time.time() - self.iter_start_time) * 1000.0
        return iter_elapsed_time_ms, self.data_load_time


def resolve_timing_warmup_iters(env_var: str = "LEWM_TIMING_WARMUP_ITERS", default: int = 3) -> int:
    """Resolve how many initial training iterations to exclude from timing summaries."""
    raw_value = os.environ.get(env_var, str(default))
    try:
        return max(0, int(raw_value))
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %d", env_var, raw_value, default)
        return max(0, int(default))


def create_stage_timing_meters() -> Dict[str, AverageMeter]:
    """Create meters for internal Stage-2/Stage-3 training-loop timing."""
    return {
        "iter_ms": AverageMeter(),
        "train_ms": AverageMeter(),
        "gpu_ms": AverageMeter(),
        "data_ms": AverageMeter(),
    }


def start_cuda_timing(enabled: bool = True) -> Tuple[Optional[torch.cuda.Event], Optional[torch.cuda.Event]]:
    """Create and record CUDA events for the train-step portion of an iteration.

    ``enabled=False`` returns ``(None, None)`` so ``stop_cuda_timing`` skips its ``torch.cuda.synchronize()``
    — callers should enable it only on log iterations, otherwise the per-iter sync serializes CPU/GPU every
    step for diagnostics that are only summarized periodically.
    """
    if not enabled or not torch.cuda.is_available():
        return None, None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    return start_event, end_event


def stop_cuda_timing(start_event: Optional[torch.cuda.Event], end_event: Optional[torch.cuda.Event]) -> float:
    """Stop CUDA timing and return elapsed milliseconds, or -1 when unavailable."""
    if start_event is None or end_event is None:
        return -1.0
    end_event.record()
    torch.cuda.synchronize()
    return float(start_event.elapsed_time(end_event))


def record_stage_timing(
    timing_meters: Dict[str, AverageMeter],
    itr: int,
    warmup_iters: int,
    iter_ms: float,
    train_ms: float,
    gpu_ms: float,
    data_ms: float,
) -> bool:
    """Record timing after warmup; returns True if the iteration was included."""
    if itr < warmup_iters:
        return False
    timing_meters["iter_ms"].update(float(iter_ms))
    timing_meters["train_ms"].update(float(train_ms))
    timing_meters["data_ms"].update(float(data_ms))
    if gpu_ms >= 0:
        timing_meters["gpu_ms"].update(float(gpu_ms))
    return True


def log_stage_timing_summary(
    stage: str,
    stage_logger: Any,
    epoch: int,
    timing_meters: Dict[str, AverageMeter],
    warmup_iters: int,
) -> None:
    """Log an epoch-level internal timing summary that excludes setup/checkpoint overhead."""
    measured_iters = int(timing_meters["iter_ms"].count)
    gpu_ms_avg = timing_meters["gpu_ms"].avg if timing_meters["gpu_ms"].count > 0 else -1.0
    stage_logger.info(
        "[%s-timing-summary] epoch=%d measured_iters=%d warmup_iters=%d "
        "iter_ms_avg=%.3f train_ms_avg=%.3f gpu_ms_avg=%.3f data_ms_avg=%.3f",
        stage,
        epoch + 1,
        measured_iters,
        warmup_iters,
        timing_meters["iter_ms"].avg,
        timing_meters["train_ms"].avg,
        gpu_ms_avg,
        timing_meters["data_ms"].avg,
    )
