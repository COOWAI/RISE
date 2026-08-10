"""Shared training-run bootstrap for every line under training/lines/.

每条训练线的 main() 以同一序列启动：分布式 -> 设备 -> checkpoint 路径 -> 日志。
setup_run() 把这四步收为一处；线专属的 config 校验/改写发生在调用之前，
线专属的 best-checkpoint 追踪变量保持在各 line 模块中。
"""

import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict

from app.vjepa_cowa_world_model.training.checkpoint import setup_checkpoint_paths
from app.vjepa_cowa_world_model.training.distributed import setup_device, setup_distributed
from app.vjepa_cowa_world_model.training.logging import setup_logging
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RunContext:
    config: Any
    world_size: int
    rank: int
    device: Any
    ckpt_paths: Dict[str, str]
    csv_logger: Any
    tb_writer: Any


def _log_effective_config(config: Any, rank: int) -> None:
    """Dump the fully-resolved config once (rank 0) so every run records exactly what it executed
    with — including the eager-resolved ``auto`` fields. Pure logging; never alters behavior."""
    if rank != 0:
        return
    payload = asdict(config) if is_dataclass(config) else config
    logger.info(
        "Effective resolved config (rank0):\n%s",
        json.dumps(payload, indent=2, default=str, sort_keys=True),
    )


def setup_run(config) -> RunContext:
    """统一执行 distributed/device/ckpt-paths/logging 四步启动。"""
    world_size, rank = setup_distributed(config.meta.seed, config.meta.deterministic)
    device = setup_device(rank)
    ckpt_paths = setup_checkpoint_paths(config.meta.folder, config.meta.resume_checkpoint)
    csv_logger, tb_writer = setup_logging(config.meta.folder, rank)
    _log_effective_config(config, rank)
    return RunContext(
        config=config,
        world_size=world_size,
        rank=rank,
        device=device,
        ckpt_paths=ckpt_paths,
        csv_logger=csv_logger,
        tb_writer=tb_writer,
    )
