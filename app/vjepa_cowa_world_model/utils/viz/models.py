# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Checkpoint loading for visualization (encoder EMA->fallback + predictor + planner)."""

import torch

from app.vjepa_cowa_world_model.training import (
    get_encoder_embed_dim,
    init_encoder,
    init_planner,
    init_predictor,
    load_state_dict_helper,
    parse_training_config,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_models(checkpoint_path, config, device):
    """
    加载 encoder + predictor + planner 模型（使用 training 模块的统一初始化）。

    Args:
        checkpoint_path: checkpoint 文件路径
        config: 配置字典 (从 yaml 加载)
        device: torch 设备
    Returns:
        encoder, predictor, planner, cfg (TrainingConfig)
    """
    # 将 dict config 转为 TrainingConfig dataclass
    cfg = parse_training_config(config)

    # ==================== 初始化模型 ====================
    encoder, _ = init_encoder(cfg, device)
    encoder_embed_dim = get_encoder_embed_dim(encoder)
    logger.info(f"encoder_embed_dim: {encoder_embed_dim}")

    predictor = init_predictor(cfg, device, encoder_embed_dim)

    num_poses = cfg.data.num_target_frames - cfg.train.num_observed_frames
    planner = init_planner(cfg, encoder_embed_dim, device, num_poses=num_poses)

    # ==================== 加载 checkpoint ====================
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 加载 encoder (优先使用 target_encoder，因为 EMA 版本通常更好)
    if "target_encoder" in checkpoint:
        load_state_dict_helper(encoder, checkpoint["target_encoder"], "target_encoder")
    elif "encoder" in checkpoint:
        load_state_dict_helper(encoder, checkpoint["encoder"], "encoder")

    # 加载 predictor
    if "predictor" in checkpoint:
        load_state_dict_helper(predictor, checkpoint["predictor"], "predictor")
    else:
        logger.warning("No predictor weights found in checkpoint!")

    # 加载 planner
    if planner is not None and "planner" in checkpoint:
        load_state_dict_helper(planner, checkpoint["planner"], "planner")
    elif planner is not None:
        logger.warning("No planner weights found in checkpoint!")

    # 设置为评估模式
    encoder.eval()
    predictor.eval()
    if planner is not None:
        planner.eval()
    logger.info("Models loaded successfully!")
    return encoder, predictor, planner, cfg
