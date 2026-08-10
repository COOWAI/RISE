# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
日志记录模块

提供训练日志记录功能。
"""

import logging
import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from src.utils.logging import AverageMeter, TableLogger, get_logger

logger = get_logger(__name__)

_RUNTIME_LOG_HANDLER_FLAG = "_vjepa_world_model_runtime_log_handler"
_LOG_RUN_ID_ENV = "VJEPA_LOG_RUN_ID"


def _predictor_camera_metric_tag(metric_key: str, namespace: str) -> Optional[str]:
    prefix = "predictor_per_camera_"
    if not metric_key.startswith(prefix) or "/" not in metric_key:
        return None
    metric_name, camera_name = metric_key[len(prefix) :].split("/", 1)
    if metric_name == "num_tokens":
        return None
    return f"{namespace}/per_camera_{metric_name}/{camera_name}"


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


OPEN_LOOP_DIAGNOSTIC_METRICS = ("l2_avg", "collision_rate", "ade", "fde", "minfde_k")
OPEN_LOOP_SELECTION_RULE = "L2_avg -> Collision Rate -> ADE -> FDE -> minFDE@K"
OPEN_LOOP_REPORTED_SECONDS = (1, 2, 3, 4)


# 默认日志列配置
DEFAULT_LOG_COLUMNS = [
    ("%s", "type"),
    ("%d", "epoch"),
    ("%.5f", "loss"),
    ("%.5f", "seg_loss"),
    ("%.5f", "mask_loss"),
    ("%.5f", "dice_loss"),
    ("%.5f", "traj_loss"),
    ("%.5f", "reg_loss"),
    ("%.5f", "conf_loss"),
    ("%.1f", "avg_iter_time(ms)"),
    ("%.1f", "avg_gpu_time(ms)"),
    ("%.1f", "avg_dataload_time(ms)"),
    ("%.5f", "val_ade"),
    ("%.5f", "val_fde"),
    ("%.5f", "val_minade_k"),
    ("%.5f", "val_minfde_k"),
    ("%.5f", "val_l2_avg"),
    ("%.5f", "val_collision_rate"),
]

DEFAULT_LOG_COMMENTS = [
    "Training Log - Each epoch records average metrics",
    "Columns: [type, epoch] | [loss, seg_loss*, traj_loss*] | [time metrics] | [val metrics]",
    "  - type: 'train' = epoch avg losses, 'val' = validation metrics",
    "  - seg_loss*: mask_loss + dice_loss",
    "  - traj_loss*: wta_loss + reg_loss + conf_loss",
    "  - val metrics: ADE, FDE, minADE@K, minFDE@K, L2_avg, Collision_rate (only in val rows)",
]


def _metric_or_inf(metrics: Dict[str, float], key: str) -> float:
    value = metrics.get(key, float("inf"))
    if value is None:
        return float("inf")
    value = float(value)
    # NaN (e.g. a degenerate collision avg) must map to +inf, else once a NaN becomes the recorded
    # "best" no finite candidate ever compares better (all NaN comparisons are False) and it sticks.
    if value != value:
        return float("inf")
    return value


def build_validation_record(epoch: int, val_metrics: Dict[str, float]) -> Dict[str, float]:
    """构造用于排序和诊断的验证记录。"""
    record = {"epoch": int(epoch)}
    for key in ("ade", "fde", "minade_k", "minfde_k", "l2_avg", "collision_rate"):
        record[key] = _metric_or_inf(val_metrics, key)
    return record


def get_open_loop_sort_key(val_metrics: Dict[str, float]) -> tuple:
    """开环 checkpoint 排序键，越小越好。"""
    return tuple(_metric_or_inf(val_metrics, key) for key in OPEN_LOOP_DIAGNOSTIC_METRICS)


def is_better_open_loop_candidate(
    candidate_metrics: Dict[str, float], current_best_metrics: Optional[Dict[str, float]]
) -> bool:
    """按开环优先级比较两个 checkpoint 候选。"""
    if current_best_metrics is None:
        return True
    return get_open_loop_sort_key(candidate_metrics) < get_open_loop_sort_key(current_best_metrics)


def build_open_loop_diagnostic_rows(validation_history: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """为验证历史生成开环诊断排序表。"""
    if not validation_history:
        return []

    rows = [build_validation_record(record.get("epoch", 0), record) for record in validation_history]

    for metric_name in OPEN_LOOP_DIAGNOSTIC_METRICS:
        ordered_rows = sorted(rows, key=lambda row: (row[metric_name], row["epoch"]))
        prev_value = None
        prev_rank = 0
        for position, row in enumerate(ordered_rows, start=1):
            if prev_value is None or row[metric_name] != prev_value:
                prev_rank = position
                prev_value = row[metric_name]
            row[f"rank_{metric_name}"] = prev_rank

    for row in rows:
        row["open_loop_score"] = sum(row[f"rank_{metric_name}"] for metric_name in OPEN_LOOP_DIAGNOSTIC_METRICS)

    return sorted(
        rows,
        key=lambda row: (
            row["open_loop_score"],
            get_open_loop_sort_key(row),
            row["epoch"],
        ),
    )


def format_open_loop_diagnostic_table(rows: List[Dict[str, float]], top_k: int = 5) -> List[str]:
    """格式化开环 checkpoint 诊断表。"""
    if not rows:
        return []

    lines = [
        "Open-loop checkpoint ranking (lower is better; score=sum of metric ranks)",
        "epoch | score | l2_avg | collision | ade | fde | minfde | ranks(l2/collision/ade/fde/minfde)",
    ]

    for row in rows[:top_k]:
        rank_summary = "/".join(str(int(row[f"rank_{metric_name}"])) for metric_name in OPEN_LOOP_DIAGNOSTIC_METRICS)
        lines.append(
            "{epoch:>5} | {score:>5} | {l2_avg:.5f} | {collision_rate:.5f} | {ade:.5f} | {fde:.5f} | "
            "{minfde_k:.5f} | {rank_summary}".format(
                epoch=int(row["epoch"]),
                score=int(row["open_loop_score"]),
                l2_avg=row["l2_avg"],
                collision_rate=row["collision_rate"],
                ade=row["ade"],
                fde=row["fde"],
                minfde_k=row["minfde_k"],
                rank_summary=rank_summary,
            )
        )

    return lines


def log_open_loop_diagnostic_table(
    validation_history: Optional[List[Dict[str, float]]], rank: int, top_k: int = 5
) -> None:
    """将开环 checkpoint 诊断表写入日志。"""
    if rank != 0 or not validation_history:
        return

    rows = build_open_loop_diagnostic_rows(validation_history)
    for line in format_open_loop_diagnostic_table(rows, top_k=top_k):
        logger.info(line)


def _sanitize_log_token(token: str) -> str:
    """将 run id 转成适合文件名的 token。"""
    sanitized = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in token)
    return sanitized.strip("_") or "run"


def _build_log_run_id() -> str:
    """生成每次 setup_logging 调用唯一的日志 run id。"""
    configured_run_id = os.environ.get(_LOG_RUN_ID_ENV)
    if configured_run_id:
        return _sanitize_log_token(configured_run_id)

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    time_ns = time.time_ns() % 1_000_000_000
    return f"{timestamp}_{time_ns:09d}_pid{os.getpid()}"


def _reserve_rank_log_file(folder: str, rank: int) -> str:
    """为当前训练实例预留新的 rank 日志文件，避免复用历史 log_r*.txt。"""
    os.makedirs(folder, exist_ok=True)
    run_id = _build_log_run_id()
    stem = f"log_r{rank}_{run_id}"
    attempt = 0

    while True:
        suffix = "" if attempt == 0 else f"_{attempt}"
        log_file = os.path.join(folder, f"{stem}{suffix}.txt")
        try:
            with open(log_file, "x", encoding="utf-8"):
                pass
            return log_file
        except FileExistsError:
            attempt += 1


def _detach_runtime_log_handlers() -> None:
    """移除本模块安装到 root logger 的旧 file handlers。"""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if not getattr(handler, _RUNTIME_LOG_HANDLER_FLAG, False):
            continue
        root_logger.removeHandler(handler)
        handler.close()


def _attach_runtime_log_handler(log_file: str) -> None:
    """Attach a root file handler so child loggers also write into the per-rank log file."""
    runtime_log_path = os.path.abspath(log_file)
    _detach_runtime_log_handlers()

    root_logger = logging.getLogger()
    existing_handler = any(
        isinstance(handler, logging.FileHandler)
        and os.path.abspath(getattr(handler, "baseFilename", "")) == runtime_log_path
        for handler in root_logger.handlers
    )
    if existing_handler:
        return

    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    runtime_handler = logging.FileHandler(runtime_log_path, mode="a", encoding="utf-8")
    runtime_handler.setLevel(logging.INFO)

    formatter = None
    for handler in root_logger.handlers:
        if handler.formatter is not None:
            formatter = handler.formatter
            break
    if formatter is None:
        formatter = logging.Formatter(
            "[%(levelname)-8s][%(asctime)s][%(name)-20s][%(funcName)-25s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    runtime_handler.setFormatter(formatter)
    setattr(runtime_handler, _RUNTIME_LOG_HANDLER_FLAG, True)
    root_logger.addHandler(runtime_handler)


def setup_logging(
    folder: str, rank: int, log_columns: Optional[List] = None, log_comments: Optional[List[str]] = None
) -> tuple:
    """
    设置日志记录器

    Args:
        folder: 保存文件夹
        rank: 当前进程 rank
        log_columns: 日志列配置
        log_comments: 日志注释

    Returns:
        tuple: (csv_logger, tb_writer)
    """
    if log_columns is None:
        log_columns = DEFAULT_LOG_COLUMNS
    if log_comments is None:
        log_comments = DEFAULT_LOG_COMMENTS

    # CSV 日志
    log_file = _reserve_rank_log_file(folder, rank)
    csv_logger = TableLogger(
        log_file,
        *log_columns,
        mode="+a",
        comments=log_comments,
    )
    _attach_runtime_log_handler(log_file)
    logger.info(f"Training log will be saved to: {log_file}")

    # TensorBoard
    if rank == 0:
        tensorboard_dir = os.path.join(folder, "tensorboard_logs")
        tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        logger.info(f"TensorBoard logs will be saved to: {tensorboard_dir}")
    else:
        tb_writer = None

    return csv_logger, tb_writer


def log_training_metrics(
    tb_writer: Optional[SummaryWriter],
    loss_meter: AverageMeter,
    jloss_meter: AverageMeter,
    sloss_meter: AverageMeter,
    seg_loss_meter: AverageMeter,
    mask_loss_meter: AverageMeter,
    dice_loss_meter: AverageMeter,
    traj_loss_meter: AverageMeter,
    reg_loss_meter: AverageMeter,
    conf_loss_meter: AverageMeter,
    cover_loss_meter: AverageMeter,
    iter_time_meter: AverageMeter,
    gpu_time_meter: AverageMeter,
    data_elapsed_time_meter: AverageMeter,
    epoch: int,
    itr: int,
    ipe: int,
    rank: int,
    _new_lr: float,
    _new_wd: float,
    train_start_time: Optional[float] = None,
    start_epoch: int = 0,
    total_epochs: Optional[int] = None,
    log_freq: int = 10,
    wta_loss_version: str = "v1",
    awta_init_temperature: float = 8.0,
    awta_exp_base: float = 0.984,
    awta_min_temperature: float = 0.1,
    num_modes: int = 6,
    planner_type: str = "transformer",
    vel_loss_meter: Optional[AverageMeter] = None,
    yaw_loss_meter: Optional[AverageMeter] = None,
    cls_valid_ratio_meter: Optional[AverageMeter] = None,
    sigreg_loss_meter: Optional[AverageMeter] = None,
    ortho_loss_meter: Optional[AverageMeter] = None,
) -> None:
    """
    记录训练指标

    Args:
        tb_writer: TensorBoard writer
        *_meter: 各种 AverageMeter
        epoch: 当前 epoch
        itr: 当前迭代
        ipe: 每个 epoch 的迭代次数
        rank: 当前进程 rank
        _new_lr: 当前学习率
        _new_wd: 当前权重衰减
        log_freq: 日志频率
        wta_loss_version: WTA 损失版本
        awta_*: aWTA 参数
        num_modes: 模态数
        cls_valid_ratio_meter: diffusion planner cls gate 有效样本比例
        sigreg_loss_meter: le-wm SIGReg 正则化损失
    """
    from app.vjepa_cowa_world_model.losses import awta_temperature_schedule

    if isinstance(start_epoch, str):
        legacy_log_freq = train_start_time
        legacy_wta_loss_version = start_epoch
        legacy_awta_init_temperature = total_epochs
        legacy_awta_exp_base = log_freq
        legacy_awta_min_temperature = wta_loss_version
        legacy_num_modes = awta_init_temperature

        train_start_time = None
        start_epoch = 0
        total_epochs = None
        log_freq = int(legacy_log_freq) if legacy_log_freq is not None else 10
        wta_loss_version = legacy_wta_loss_version
        awta_init_temperature = float(legacy_awta_init_temperature)
        awta_exp_base = float(legacy_awta_exp_base)
        awta_min_temperature = float(legacy_awta_min_temperature)
        num_modes = int(legacy_num_modes)

    if train_start_time is None:
        train_start_time = time.time()
    if total_epochs is None:
        total_epochs = max(epoch + 1, start_epoch + 1)

    completed_iters = max((epoch - start_epoch) * ipe + (itr + 1), 1)
    total_iters = max((total_epochs - start_epoch) * ipe, 1)
    elapsed_sec = max(time.time() - train_start_time, 0.0)
    avg_iter_sec = max(elapsed_sec / completed_iters, 1.0e-6)
    remaining_iters = max(total_iters - completed_iters, 0)
    eta_sec = remaining_iters * avg_iter_sec
    time_stats = {
        "elapsed_sec": elapsed_sec,
        "eta_sec": eta_sec,
        "expected_total_sec": elapsed_sec + eta_sec,
    }

    # TensorBoard logging — 加 log_freq 门控，与 console 对齐
    if rank == 0 and tb_writer is not None and ((itr % log_freq == 0) or (itr == ipe - 1)):
        global_step = epoch * ipe + itr

        # 损失指标
        tb_writer.add_scalar("Loss/total", loss_meter.avg, global_step)
        tb_writer.add_scalar("Loss/joint_embedding", jloss_meter.avg, global_step)
        tb_writer.add_scalar("Loss/autoregressive", sloss_meter.avg, global_step)
        tb_writer.add_scalar("Loss/segmentation", seg_loss_meter.avg, global_step)
        tb_writer.add_scalar("Loss/segmentation_mask", mask_loss_meter.avg, global_step)
        tb_writer.add_scalar("Loss/segmentation_dice", dice_loss_meter.avg, global_step)
        tb_writer.add_scalar("Loss/trajectory", traj_loss_meter.avg, global_step)

        # SIGReg loss (le-wm)
        if sigreg_loss_meter is not None:
            tb_writer.add_scalar("Loss/sigreg", sigreg_loss_meter.avg, global_step)
        if ortho_loss_meter is not None:
            tb_writer.add_scalar("Loss/ortho", ortho_loss_meter.avg, global_step)

        # 优化器参数
        tb_writer.add_scalar("Optimization/learning_rate", _new_lr, global_step)
        tb_writer.add_scalar("Optimization/weight_decay", _new_wd, global_step)

        # 性能指标
        tb_writer.add_scalar("Performance/iter_time_ms", iter_time_meter.avg, global_step)
        tb_writer.add_scalar("Performance/gpu_time_ms", gpu_time_meter.avg, global_step)
        tb_writer.add_scalar("Performance/data_load_time_ms", data_elapsed_time_meter.avg, global_step)

        # 内存使用
        tb_writer.add_scalar(
            "System/cuda_memory_allocated_MB", torch.cuda.max_memory_allocated() / 1024.0**2, global_step
        )

        # Planner 详细损失
        if wta_loss_version in ("v2", "v3"):
            tb_writer.add_scalar("Planner/cover_loss", cover_loss_meter.avg, global_step)
        if wta_loss_version == "v3":
            cur_awta_temp_log = awta_temperature_schedule(
                awta_init_temperature, epoch, awta_exp_base, awta_min_temperature
            )
            tb_writer.add_scalar("Planner/awta_temperature", cur_awta_temp_log, global_step)
        # Diffusion planner 详细损失
        if planner_type == "diffusion":
            tb_writer.add_scalar("Planner/reg_loss", reg_loss_meter.avg, global_step)
            tb_writer.add_scalar("Planner/focal_cls_loss", conf_loss_meter.avg, global_step)
            if vel_loss_meter is not None:
                tb_writer.add_scalar("Planner/vel_loss", vel_loss_meter.avg, global_step)
            if yaw_loss_meter is not None:
                tb_writer.add_scalar("Planner/yaw_loss", yaw_loss_meter.avg, global_step)
            if cls_valid_ratio_meter is not None:
                tb_writer.add_scalar("Planner/cls_sample_valid_ratio", cls_valid_ratio_meter.avg, global_step)
    # 控制台日志 - 只有 rank 0 打印
    if rank == 0 and ((itr % log_freq == 0) or (itr == ipe - 1)):
        _log_to_console(
            loss_meter,
            jloss_meter,
            sloss_meter,
            seg_loss_meter,
            mask_loss_meter,
            dice_loss_meter,
            traj_loss_meter,
            reg_loss_meter,
            conf_loss_meter,
            cover_loss_meter,
            iter_time_meter,
            gpu_time_meter,
            data_elapsed_time_meter,
            epoch,
            itr,
            _new_lr,
            _new_wd,
            time_stats,
            wta_loss_version,
            awta_init_temperature,
            awta_exp_base,
            awta_min_temperature,
            num_modes,
            planner_type,
            vel_loss_meter,
            yaw_loss_meter,
            sigreg_loss_meter,
            ortho_loss_meter,
        )


def log_predictor_camera_training_metrics(
    tb_writer: Optional[SummaryWriter],
    predictor_camera_loss_meters: Dict[str, AverageMeter],
    epoch: int,
    itr: int,
    ipe: int,
    rank: int,
    log_freq: int = 10,
) -> None:
    """Log per-camera predictor training diagnostics."""
    if rank != 0 or tb_writer is None or not predictor_camera_loss_meters:
        return
    if not ((itr % log_freq == 0) or (itr == ipe - 1)):
        return

    global_step = epoch * ipe + itr
    for metric_key, meter in sorted(predictor_camera_loss_meters.items()):
        tag = _predictor_camera_metric_tag(metric_key, "PredictorTrain")
        if tag is not None:
            tb_writer.add_scalar(tag, meter.avg, global_step)


def _log_to_console(
    loss_meter,
    jloss_meter,
    sloss_meter,
    seg_loss_meter,
    mask_loss_meter,
    dice_loss_meter,
    traj_loss_meter,
    reg_loss_meter,
    conf_loss_meter,
    cover_loss_meter,
    iter_time_meter,
    gpu_time_meter,
    data_elapsed_time_meter,
    epoch,
    itr,
    _new_lr,
    _new_wd,
    time_stats,
    wta_loss_version,
    awta_init_temperature,
    awta_exp_base,
    awta_min_temperature,
    num_modes,
    planner_type="transformer",
    vel_loss_meter=None,
    yaw_loss_meter=None,
    sigreg_loss_meter=None,
    ortho_loss_meter=None,
) -> None:
    """打印日志到控制台"""
    from app.vjepa_cowa_world_model.losses import awta_temperature_schedule

    # SIGReg: append to jepa section if available
    if sigreg_loss_meter is not None:
        base_msg = (
            "[%d, %5d] loss: %.3f "
            "[jepa: %.2f+%.2f, sigreg: %.4f, ortho: %.4f, seg: %.3f (mask: %.3f, dice: %.3f)] "
            "[wd: %.2e] [lr: %.2e] "
            "[mem: %.2e] "
            "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms] "
            "[elapsed: %s] [eta: %s] [total: %s]"
        )
    else:
        base_msg = (
            "[%d, %5d] loss: %.3f "
            "[jepa: %.2f+%.2f, seg: %.3f (mask: %.3f, dice: %.3f)] "
            "[wd: %.2e] [lr: %.2e] "
            "[mem: %.2e] "
            "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms] "
            "[elapsed: %s] [eta: %s] [total: %s]"
        )
    base_args = (
        epoch + 1,
        itr,
        loss_meter.avg,
        jloss_meter.avg,
        sloss_meter.avg,
    )
    if sigreg_loss_meter is not None:
        base_args += (sigreg_loss_meter.avg,)
        base_args += (ortho_loss_meter.avg if ortho_loss_meter is not None else 0.0,)
    base_args += (
        seg_loss_meter.avg,
        mask_loss_meter.avg,
        dice_loss_meter.avg,
        _new_wd,
        _new_lr,
        torch.cuda.max_memory_allocated() / 1024.0**2,
        iter_time_meter.avg,
        gpu_time_meter.avg,
        data_elapsed_time_meter.avg,
        _format_duration(time_stats["elapsed_sec"]),
        _format_duration(time_stats["eta_sec"]),
        _format_duration(time_stats["expected_total_sec"]),
    )

    # Split point: prefix (before [wd:]) vs suffix (from [wd:] onward)
    _prefix_len = len(base_args) - 9  # suffix: wd, lr, mem, iter, gpu, data, elapsed, eta, total

    if planner_type == "diffusion":
        # Diffusion planner 日志: loss, reg, cls(focal), vel, yaw
        _vel = vel_loss_meter.avg if vel_loss_meter is not None else 0.0
        _yaw = yaw_loss_meter.avg if yaw_loss_meter is not None else 0.0
        logger.info(
            base_msg.replace("] [wd:", "[traj=[diff:%.3f reg:%.3f cls:%.3f vel:%.3f yaw:%.3f]] [wd:"),
            *base_args[:_prefix_len],
            traj_loss_meter.avg,
            reg_loss_meter.avg,
            conf_loss_meter.avg,
            _vel,
            _yaw,
            *base_args[_prefix_len:],
        )
    elif num_modes == 1:
        # 单模型日志
        logger.info(
            base_msg.replace("] [wd:", "[traj=[single:%.3f reg:%.3f]] [wd:"),
            *base_args[:_prefix_len],
            traj_loss_meter.avg,
            reg_loss_meter.avg,
            *base_args[_prefix_len:],
        )
    elif wta_loss_version == "v2":
        # 多模型 v2 日志
        logger.info(
            base_msg.replace("] [wd:", "[traj=[wta:%.3f reg:%.3f conf:%.3f cover:%.3f]] [wd:"),
            *base_args[:_prefix_len],
            traj_loss_meter.avg,
            reg_loss_meter.avg,
            conf_loss_meter.avg,
            cover_loss_meter.avg,
            *base_args[_prefix_len:],
        )
    elif wta_loss_version == "v3":
        # 多模型 v3 (aWTA) 日志
        cur_awta_temp_log = awta_temperature_schedule(
            awta_init_temperature, epoch, awta_exp_base, awta_min_temperature
        )
        logger.info(
            base_msg.replace("] [wd:", "[traj=[aWTA:%.3f reg:%.3f conf:%.3f cover:%.3f T:%.2f]] [wd:"),
            *base_args[:_prefix_len],
            traj_loss_meter.avg,
            reg_loss_meter.avg,
            conf_loss_meter.avg,
            cover_loss_meter.avg,
            cur_awta_temp_log,
            *base_args[_prefix_len:],
        )
    else:  # v1
        # 多模型 v1 日志
        logger.info(
            base_msg.replace("] [wd:", "[traj=[wta:%.3f reg:%.3f conf:%.3f]] [wd:"),
            *base_args[:_prefix_len],
            traj_loss_meter.avg,
            reg_loss_meter.avg,
            conf_loss_meter.avg,
            *base_args[_prefix_len:],
        )


def log_epoch_summary(
    tb_writer: Optional[SummaryWriter],
    csv_logger: TableLogger,
    loss_meter: AverageMeter,
    jloss_meter: AverageMeter,
    sloss_meter: AverageMeter,
    seg_loss_meter: AverageMeter,
    mask_loss_meter: AverageMeter,
    dice_loss_meter: AverageMeter,
    traj_loss_meter: AverageMeter,
    reg_loss_meter: AverageMeter,
    conf_loss_meter: AverageMeter,
    iter_time_meter: AverageMeter,
    gpu_time_meter: AverageMeter,
    data_elapsed_time_meter: AverageMeter,
    epoch: int,
    rank: int,
    has_validation: bool = False,
) -> None:
    """
    记录 epoch 结束时的摘要

    Args:
        tb_writer: TensorBoard writer
        csv_logger: CSV logger
        *_meter: 各种 AverageMeter
        epoch: 当前 epoch
        rank: 当前进程 rank
        has_validation: 是否有验证
    """
    logger.info(
        f"主人，Epoch {epoch + 1} 平均损失: %.3f (JEPA: %.3f, Seg: %.3f)"
        % (loss_meter.avg, jloss_meter.avg + sloss_meter.avg, seg_loss_meter.avg)
    )

    if rank == 0 and tb_writer is not None:
        tb_writer.add_scalar("Epoch/avg_loss", loss_meter.avg, epoch + 1)
        tb_writer.add_scalar("Epoch/avg_jloss", jloss_meter.avg, epoch + 1)
        tb_writer.add_scalar("Epoch/avg_sloss", sloss_meter.avg, epoch + 1)
        tb_writer.add_scalar("Epoch/avg_seg_loss", seg_loss_meter.avg, epoch + 1)
        tb_writer.flush()

    # 记录到 CSV
    if rank == 0:
        csv_logger.log(
            "train",
            epoch + 1,
            loss_meter.avg,
            seg_loss_meter.avg,
            mask_loss_meter.avg,
            dice_loss_meter.avg,
            traj_loss_meter.avg,
            reg_loss_meter.avg,
            conf_loss_meter.avg,
            iter_time_meter.avg,
            gpu_time_meter.avg,
            data_elapsed_time_meter.avg,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            add_separator=not has_validation,
        )


def log_validation_metrics(
    tb_writer: Optional[SummaryWriter],
    csv_logger: TableLogger,
    val_metrics: Dict[str, float],
    epoch: int,
    rank: int,
    validation_history: Optional[List[Dict[str, float]]] = None,
) -> None:
    """
    记录验证指标

    Args:
        tb_writer: TensorBoard writer
        csv_logger: CSV logger
        val_metrics: 验证指标字典
        epoch: 当前 epoch
        rank: 当前进程 rank
        validation_history: 验证历史，用于输出开环诊断表
    """
    # TensorBoard
    if rank == 0 and tb_writer is not None:
        tb_writer.add_scalar("Validation/ADE", val_metrics["ade"], epoch + 1)
        tb_writer.add_scalar("Validation/FDE", val_metrics["fde"], epoch + 1)
        if "minade_k" in val_metrics:
            tb_writer.add_scalar("Validation/minADE@K", val_metrics["minade_k"], epoch + 1)
        if "minfde_k" in val_metrics:
            tb_writer.add_scalar("Validation/minFDE@K", val_metrics["minfde_k"], epoch + 1)
        # NuScenes L2 per timestep
        if "l2_avg" in val_metrics:
            tb_writer.add_scalar("Validation/L2_avg", val_metrics["l2_avg"], epoch + 1)
        if "l2_point_avg" in val_metrics:
            tb_writer.add_scalar("Validation/PointL2_avg", val_metrics["l2_point_avg"], epoch + 1)
        for sec in OPEN_LOOP_REPORTED_SECONDS:
            key = f"l2_at_{sec}s"
            if key in val_metrics:
                tb_writer.add_scalar(f"Validation/L2_at_{sec}s", val_metrics[key], epoch + 1)
            point_key = f"l2_point_at_{sec}s"
            if point_key in val_metrics:
                tb_writer.add_scalar(f"Validation/PointL2_at_{sec}s", val_metrics[point_key], epoch + 1)
        # NuScenes Collision rate
        if "collision_rate" in val_metrics:
            tb_writer.add_scalar("Validation/Collision_rate", val_metrics["collision_rate"], epoch + 1)
        for sec in OPEN_LOOP_REPORTED_SECONDS:
            key = f"collision_at_{sec}s"
            if key in val_metrics:
                tb_writer.add_scalar(f"Validation/Collision_at_{sec}s", val_metrics[key], epoch + 1)
        tb_writer.flush()

    if rank == 0:
        msg = f"Validation metrics logged to TensorBoard: ADE={val_metrics['ade']:.4f}, FDE={val_metrics['fde']:.4f}"
        if "minade_k" in val_metrics and "minfde_k" in val_metrics:
            msg += f", minADE@K={val_metrics['minade_k']:.4f}, minFDE@K={val_metrics['minfde_k']:.4f}"
        if "l2_avg" in val_metrics:
            msg += f", L2_avg={val_metrics['l2_avg']:.4f}"
        if "l2_point_avg" in val_metrics:
            msg += f", PointL2_avg={val_metrics['l2_point_avg']:.4f}"
        if "collision_rate" in val_metrics:
            msg += f", Collision_rate={val_metrics['collision_rate']:.4f}"
        logger.info(msg)
        log_open_loop_diagnostic_table(validation_history, rank, top_k=5)

    # CSV
    if rank == 0:
        csv_logger.log(
            "val",
            epoch + 1,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            val_metrics.get("ade", float("nan")),
            val_metrics.get("fde", float("nan")),
            val_metrics.get("minade_k", float("nan")),
            val_metrics.get("minfde_k", float("nan")),
            val_metrics.get("l2_avg", float("nan")),
            val_metrics.get("collision_rate", float("nan")),
            add_separator=True,
        )


def log_predictor_validation_metrics(
    tb_writer: Optional[SummaryWriter],
    predictor_metrics: Dict[str, float],
    epoch: int,
    rank: int,
) -> None:
    """记录 predictor-only validation 指标。"""

    metric_tags = {
        "predictor_loss": "loss",
        "predictor_objective_loss": "objective_loss",
        "predictor_world_loss": "world_loss",
        "predictor_action_loss": "action_loss",
        "predictor_joint_loss": "joint_loss",
        "predictor_jloss": "jloss",
        "predictor_sloss": "sloss",
        "predictor_flow_loss": "flow_loss",
        "predictor_x0_loss": "x0_loss",
        "predictor_num_tokens": "num_tokens",
        "predictor_num_samples": "num_samples",
        "predictor_batches": "batches",
        "predictor_failed_batches": "failed_batches",
    }

    if rank == 0 and tb_writer is not None:
        for key, tag_name in metric_tags.items():
            if key in predictor_metrics:
                tb_writer.add_scalar(f"PredictorValidation/{tag_name}", predictor_metrics[key], epoch + 1)
        for key, value in sorted(predictor_metrics.items()):
            tag = _predictor_camera_metric_tag(key, "PredictorValidation")
            if tag is not None:
                tb_writer.add_scalar(tag, value, epoch + 1)
        tb_writer.flush()

    if rank == 0:
        msg = f"Predictor validation metrics logged: loss={predictor_metrics['predictor_loss']:.5f}"
        if "predictor_objective_loss" in predictor_metrics:
            msg += f", objective={predictor_metrics['predictor_objective_loss']:.5f}"
        if "predictor_jloss" in predictor_metrics:
            msg += f", jloss={predictor_metrics['predictor_jloss']:.5f}"
        if "predictor_sloss" in predictor_metrics:
            msg += f", sloss={predictor_metrics['predictor_sloss']:.5f}"
        if "predictor_flow_loss" in predictor_metrics:
            msg += f", flow={predictor_metrics['predictor_flow_loss']:.5f}"
        if "predictor_x0_loss" in predictor_metrics:
            msg += f", x0={predictor_metrics['predictor_x0_loss']:.5f}"
        if "predictor_action_loss" in predictor_metrics:
            msg += f", action={predictor_metrics['predictor_action_loss']:.5f}"
        if "predictor_joint_loss" in predictor_metrics:
            msg += f", joint={predictor_metrics['predictor_joint_loss']:.5f}"
        camera_loss_keys = sorted(key for key in predictor_metrics if key.startswith("predictor_per_camera_loss/"))
        for key in camera_loss_keys:
            camera_name = key.rsplit("/", 1)[1]
            msg += f", {camera_name}={predictor_metrics[key]:.5f}"
        msg += f", tokens={predictor_metrics.get('predictor_num_tokens', 0.0):.0f}"
        logger.info(msg)


def log_training_summary(
    best_epoch: int,
    best_ade: float,
    best_fde: float,
    best_minade_k: float,
    best_minfde_k: float,
    best_path: str,
    tb_writer: Optional[SummaryWriter],
    rank: int,
    best_l2_avg: float = float("inf"),
    best_collision_rate: float = float("inf"),
    validation_history: Optional[List[Dict[str, float]]] = None,
    best_selector_label: str = OPEN_LOOP_SELECTION_RULE,
    best_predictor_epoch: int = 0,
    best_predictor_loss: float = float("inf"),
    best_predictor_path: Optional[str] = None,
) -> None:
    """
    打印训练结束统计

    Args:
        best_epoch: 最佳 epoch
        best_ade: 最佳 ADE
        best_fde: 最佳 FDE
        best_minade_k: 最佳 minADE@K
        best_minfde_k: 最佳 minFDE@K
        best_path: 最佳 checkpoint 路径
        tb_writer: TensorBoard writer
        rank: 当前进程 rank
        best_l2_avg: 最佳 L2 avg
        best_collision_rate: 最佳碰撞率
        validation_history: 验证历史，用于输出最终开环诊断表
        best_selector_label: 最佳 checkpoint 选择规则说明
        best_predictor_epoch: 最佳 predictor checkpoint 的 epoch（按 predictor val loss 选择）
        best_predictor_loss: 最佳 predictor 验证 loss
        best_predictor_path: 最佳 predictor checkpoint 路径
    """
    if rank == 0:
        logger.info("=" * 60)
        logger.info("*** 训练完成! 最佳模型统计 ***")
        # Open-loop (planner) metrics are only meaningful when the planner ran
        # validation. For predictor-only runs the selector never fires, leaving
        # best_epoch=0 / inf — print a clear note instead of misleading inf values.
        planner_selected = best_epoch > 0 and best_ade < float("inf")
        if planner_selected:
            logger.info(f"最佳 checkpoint 选择规则: {best_selector_label}")
            logger.info(f"最佳 Epoch: {best_epoch}")
            logger.info(f"最佳 ADE: {best_ade:.5f}")
            logger.info(f"最佳 FDE: {best_fde:.5f}")
            logger.info(f"最佳 minADE@K: {best_minade_k:.5f}")
            logger.info(f"最佳 minFDE@K: {best_minfde_k:.5f}")
            if best_l2_avg < float("inf"):
                logger.info(f"最佳 L2 avg: {best_l2_avg:.5f}")
            if best_collision_rate < float("inf"):
                logger.info(f"最佳 Collision Rate: {best_collision_rate:.5f}")
            logger.info(f"最佳checkpoint保存于: {best_path}")
        else:
            logger.info("开环(planner)最佳 checkpoint 未记录（planner 验证未启用或无有效指标）。")

        if best_predictor_epoch > 0:
            logger.info("-" * 60)
            logger.info("最佳 predictor checkpoint (规则: predictor 验证 loss 最低):")
            logger.info(f"  最佳 Epoch: {best_predictor_epoch}")
            logger.info(f"  最佳 predictor loss: {best_predictor_loss:.5f}")
            if best_predictor_path is not None:
                logger.info(f"  保存于: {best_predictor_path}")

        log_open_loop_diagnostic_table(validation_history, rank, top_k=5)
        logger.info("=" * 60)

    if rank == 0 and tb_writer is not None:
        tb_writer.close()
        logger.info("主人，TensorBoard writer已关闭。")

    logger.info("主人，训练完成！")
