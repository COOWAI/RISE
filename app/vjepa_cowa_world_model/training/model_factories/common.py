"""Split from training/models.py (verbatim node moves). Part: common."""

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def compile_models(
    encoder: nn.Module,
    target_encoder: nn.Module,
    predictor: nn.Module,
    seg_head: Optional[nn.Module] = None,
    compile_model: bool = False,
) -> None:
    """
    编译模型 (torch.compile)

    Args:
        encoder: encoder 模型
        target_encoder: target_encoder 模型
        predictor: predictor 模型
        seg_head: seg_head 模型 (可选)
        compile_model: 是否编译模型
    """
    if not compile_model:
        return

    logger.info("Compiling encoder, target_encoder, and predictor.")
    torch._dynamo.config.optimize_ddp = False
    encoder.compile()
    target_encoder.compile()
    predictor.compile()
    if seg_head is not None:
        seg_head.compile()
