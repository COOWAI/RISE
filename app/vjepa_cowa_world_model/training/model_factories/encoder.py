"""Split from training/models.py (verbatim node moves). Part: encoder."""

import copy
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from app.vjepa_cowa_world_model.models.backbones.vjepa_video_model import init_video_model as init_video_model_vjepa
from app.vjepa_cowa_world_model.training.config import (
    TrainingConfig,
    is_factory_pretrained_main_encoder_config,
    resolve_proposal_encoder_backbone,
)
from app.vjepa_cowa_world_model.training.configs.core import BACKBONE_TYPES
from app.vjepa_cowa_world_model.training.model_factories.common import _is_main_process, logger


def _init_encoder_vjepa2_1(
    config: TrainingConfig,
    device: torch.device,
) -> nn.Module:
    """
    初始化 V-JEPA 2.1 encoder (inference mode only)

    使用 models/backbones/vjepa21 的 VisionTransformer 和 MultiSeqWrapper。
    当 training=False 时，输出形状为 [B, N, embed_dim]，与 V-JEPA 2 一致。

    Args:
        config: 训练配置
        device: 设备

    Returns:
        nn.Module: encoder (MultiSeqWrapper wrapped)
    """
    import app.vjepa_cowa_world_model.models.backbones.vjepa21.vision_transformer as vjepa21_vit
    from app.vjepa_cowa_world_model.models.backbones.vjepa21.wrappers import MultiSeqWrapper as MultiSeqWrapper21

    if config.model.model_name not in vjepa21_vit.__dict__:
        _valid = sorted(n for n in vjepa21_vit.__dict__ if n.startswith("vit_"))
        raise ValueError(f"Unknown model.model_name={config.model.model_name!r}; valid: {_valid}")
    encoder = vjepa21_vit.__dict__[config.model.model_name](
        img_size=config.data.crop_size,
        patch_size=config.data.patch_size,
        num_frames=512,
        tubelet_size=config.data.tubelet_size,
        uniform_power=config.model.uniform_power,
        use_sdpa=config.meta.use_sdpa,
        use_silu=config.model.use_silu,
        wide_silu=config.model.wide_silu,
        use_activation_checkpointing=config.model.use_activation_checkpointing,
        use_rope=config.model.use_rope,
        # V-JEPA 2.1 专用参数
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    encoder = MultiSeqWrapper21(encoder)
    encoder.to(device)

    return encoder


def init_encoder(
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """
    初始化 encoder 和 target_encoder

    根据 config.model.backbone 选择 V-JEPA 2 或 V-JEPA 2.1 编码器。

    Args:
        config: 训练配置
        device: 设备

    Returns:
        Tuple[nn.Module, nn.Module]: (encoder, target_encoder)
    """
    encoder = init_context_encoder(config, device)

    target_encoder = copy.deepcopy(encoder)
    logger.info("end init encoder")

    # 打印参数量（仅主进程）
    encoder_params = sum(p.numel() for p in encoder.parameters())
    target_encoder_params = sum(p.numel() for p in target_encoder.parameters())
    if _is_main_process():
        logger.info(f"init encoder_params: {encoder_params / 1e6:>8.2f}M")
        logger.info(f"init target_encoder_params: {target_encoder_params / 1e6:>8.2f}M")

    return encoder, target_encoder


def init_context_encoder_for_full_state_warmstart(
    config: TrainingConfig,
    device: torch.device,
) -> nn.Module:
    """Construct the Formal-v2 V-JEPA context encoder without loading weights.

    This entry point exists only for ``formal_v2_navsim_e120_h4_v3``.  It
    creates the exact module structure required by the checkpoint and leaves
    all state restoration to the subsequent three-role full-state warm-start.
    """

    if config.model.backbone != "vjepa_img_encoder":
        raise ValueError("full-state warm-start encoder construction requires model.backbone='vjepa_img_encoder'")
    if config.meta.load_encoder is not False:
        raise ValueError("full-state warm-start encoder construction requires meta.load_encoder=false")
    if config.meta.pretrain_checkpoint_full is not None:
        raise ValueError("full-state warm-start encoder construction requires meta.pretrain_checkpoint_full=null")
    cvoi = getattr(config, "cvoi", None)
    if getattr(cvoi, "enabled", None) is not True:
        raise ValueError("full-state warm-start encoder construction requires cvoi.enabled=true")
    if getattr(cvoi, "protocol_version", None) != "formal_v2_navsim_e120_h4_v3":
        raise ValueError("full-state warm-start encoder construction requires protocol formal_v2_navsim_e120_h4_v3")
    warmstart = getattr(cvoi, "full_state_warmstart", None)
    if getattr(warmstart, "import_mode", None) != "full_state_warmstart":
        raise ValueError("full-state warm-start encoder construction requires import_mode='full_state_warmstart'")

    logger.info("begin init context encoder (backbone=vjepa_img_encoder, checkpoint=deferred_full_state):")
    encoder = _init_vjepa_img_encoder(config, device, checkpoint_path=None)
    logger.info("end init context encoder")

    encoder_params = sum(p.numel() for p in encoder.parameters())
    if _is_main_process():
        logger.info(f"init encoder_params: {encoder_params / 1e6:>8.2f}M")
    return encoder


def init_encoder_for_full_state_warmstart(
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """Construct context and target encoders for a training full-state warm-start."""

    encoder = init_context_encoder_for_full_state_warmstart(config, device)
    target_encoder = copy.deepcopy(encoder)
    logger.info("end init encoder")

    target_encoder_params = sum(p.numel() for p in target_encoder.parameters())
    if _is_main_process():
        logger.info(f"init target_encoder_params: {target_encoder_params / 1e6:>8.2f}M")
    return encoder, target_encoder


def _init_vjepa_img_encoder(
    config: TrainingConfig,
    device: torch.device,
    *,
    checkpoint_path: Optional[str],
) -> nn.Module:
    from app.vjepa_cowa_world_model.models.vjepa_img_encoder import VJEPAImgEncoderAdapter

    return VJEPAImgEncoderAdapter(
        checkpoint_path=checkpoint_path,
        resolution=config.model.vjepa_resolution,
        num_frames=config.model.vjepa_num_frames,
        max_num_observed_frames=config.data.num_target_frames,
        checkpoint_key=config.model.vjepa_checkpoint_key,
        model_name=config.model.model_name,
        patch_size=config.data.patch_size,
        tubelet_size=config.data.tubelet_size,
        uniform_power=config.model.uniform_power,
        use_rope=config.model.use_rope,
        use_sdpa=config.meta.use_sdpa,
        use_activation_checkpointing=config.model.use_activation_checkpointing,
        use_grid_mask=config.model.vjepa_use_grid_mask,
        use_causal_attention=config.model.vjepa_use_causal_attention,
    ).to(device)


def _init_dinov2_img_encoder(config: TrainingConfig, device: torch.device) -> nn.Module:
    from app.vjepa_cowa_world_model.models.dinov2_img_encoder import Dinov2ImageEncoderAdapter

    validate_factory_pretrained_main_encoder_load_plan(config)
    return Dinov2ImageEncoderAdapter(
        checkpoint_path=config.meta.pretrain_checkpoint_full,
        resolution=config.model.dinov2_resolution,
        patch_size=config.data.patch_size,
        frame_stride=config.model.dinov2_frame_stride,
        forward_chunk_size=config.model.dinov2_forward_chunk_size,
        model_name=config.model.dinov2_model_name,
    ).to(device)


def validate_dinov2_checkpoint_source_separation(
    pretrain_checkpoint_full: str | None,
    predictor_checkpoint: str | None,
) -> None:
    """Require official DINOv2 and predictor checkpoints to identify distinct sources."""
    if not pretrain_checkpoint_full or not predictor_checkpoint:
        return

    official_path = Path(pretrain_checkpoint_full).expanduser()
    predictor_path = Path(predictor_checkpoint).expanduser()
    same_source = official_path.resolve(strict=False) == predictor_path.resolve(strict=False)
    if not same_source and official_path.exists() and predictor_path.exists():
        same_source = official_path.samefile(predictor_path)
    if same_source:
        raise ValueError(
            "DINOv2 checkpoint source separation requires meta.pretrain_checkpoint_full and "
            "meta.predictor_checkpoint to identify independent files; both refer to the same source: "
            f"meta.pretrain_checkpoint_full={pretrain_checkpoint_full!r}, "
            f"meta.predictor_checkpoint={predictor_checkpoint!r}"
        )


def validate_factory_pretrained_main_encoder_load_plan(config: TrainingConfig) -> None:
    """Validate strict source separation for a factory-loaded DINOv2 encoder."""
    if config.model.backbone != "dinov2_img_encoder":
        return
    if not config.meta.load_encoder:
        raise ValueError("DINOv2 official encoder requires meta.load_encoder=true")
    if not config.meta.pretrain_checkpoint_full:
        raise ValueError("DINOv2 official encoder requires non-empty meta.pretrain_checkpoint_full")
    conflicts = [
        name
        for name, enabled in (
            ("load_seg", config.meta.load_seg),
            ("load_planner", config.meta.load_planner),
        )
        if enabled
    ]
    if conflicts:
        raise ValueError("DINOv2 official flat checkpoint conflicts with: " + ", ".join(conflicts))
    if config.meta.load_predictor:
        if not config.meta.predictor_checkpoint:
            raise ValueError(
                "DINOv2 meta.load_predictor=true requires an independent meta.predictor_checkpoint; "
                "the official flat encoder checkpoint cannot supply predictor weights"
            )
        validate_dinov2_checkpoint_source_separation(
            config.meta.pretrain_checkpoint_full,
            config.meta.predictor_checkpoint,
        )


def init_context_encoder(config: TrainingConfig, device: torch.device) -> nn.Module:
    """初始化单个上下文 encoder，不额外创建 target_encoder。"""
    backbone = config.model.backbone
    logger.info(f"begin init context encoder (backbone={backbone}):")

    if backbone == "dinov2_img_encoder":
        if not config.meta.load_encoder:
            raise ValueError("DINOv2 img encoder requires meta.load_encoder=true")
        if not config.meta.pretrain_checkpoint_full:
            raise ValueError("DINOv2 img encoder requires non-empty meta.pretrain_checkpoint_full")
        encoder = _init_dinov2_img_encoder(config, device)
    elif backbone == "vjepa_img_encoder":
        # fail-loud (point 22): V-JEPA encoder 必须加载预训练 backbone 权重，
        # 禁止 load_encoder=false 静默得到随机初始化的 encoder。
        if not config.meta.load_encoder:
            raise ValueError(
                "V-JEPA img encoder requires meta.load_encoder=true (must load pretrained backbone weights); "
                "load_encoder=false would silently random-initialize the encoder."
            )
        checkpoint_path = config.meta.pretrain_checkpoint_full
        encoder = _init_vjepa_img_encoder(config, device, checkpoint_path=checkpoint_path)
    elif backbone == "vjepa2.1":
        encoder = _init_encoder_vjepa2_1(config, device)
    elif backbone == "vjepa2":
        encoder, _ = init_video_model_vjepa(
            uniform_power=config.model.uniform_power,
            use_mask_tokens=config.model.use_mask_tokens,
            num_mask_tokens=10,
            zero_init_mask_tokens=config.model.zero_init_mask_tokens,
            device=device,
            patch_size=config.data.patch_size,
            max_num_frames=512,
            tubelet_size=config.data.tubelet_size,
            model_name=config.model.model_name,
            crop_size=config.data.crop_size,
            pred_depth=config.model.pred_depth,
            pred_num_heads=config.model.pred_num_heads,
            pred_embed_dim=config.model.pred_embed_dim,
            use_sdpa=config.meta.use_sdpa,
            use_silu=config.model.use_silu,
            use_pred_silu=config.model.use_pred_silu,
            wide_silu=config.model.wide_silu,
            use_rope=config.model.use_rope,
            use_activation_checkpointing=config.model.use_activation_checkpointing,
        )
    else:
        raise ValueError(f"Unknown model.backbone={backbone!r}; expected one of {BACKBONE_TYPES}.")

    encoder_params = sum(p.numel() for p in encoder.parameters())
    if _is_main_process():
        logger.info(f"init context_encoder_params: {encoder_params / 1e6:>8.2f}M")
    return encoder


def init_proposal_encoder(config: TrainingConfig, device: torch.device) -> nn.Module:
    """初始化独立 proposal encoder，允许与主 encoder 使用不同 backbone。"""
    backbone = resolve_proposal_encoder_backbone(config)
    logger.info(f"begin init proposal encoder (backbone={backbone}):")

    if backbone == "vjepa_img_encoder":
        from app.vjepa_cowa_world_model.models.vjepa_img_encoder import VJEPAImgEncoderAdapter

        checkpoint_key = config.proposal.vjepa_checkpoint_key or config.proposal.encoder_checkpoint_key
        encoder = VJEPAImgEncoderAdapter(
            checkpoint_path=None,
            resolution=config.proposal.vjepa_resolution,
            num_frames=config.proposal.vjepa_num_frames,
            max_num_observed_frames=config.train.num_observed_frames,
            checkpoint_key=checkpoint_key,
            model_name=config.proposal.encoder_model_name or config.model.model_name,
            patch_size=config.data.patch_size,
            tubelet_size=config.data.tubelet_size,
            uniform_power=config.model.uniform_power,
            use_rope=config.model.use_rope,
            use_sdpa=config.meta.use_sdpa,
            use_activation_checkpointing=config.model.use_activation_checkpointing,
            use_grid_mask=config.proposal.vjepa_use_grid_mask,
            use_causal_attention=config.proposal.vjepa_use_causal_attention,
        ).to(device)
    elif backbone == config.model.backbone:
        encoder = init_context_encoder(config, device)
    else:
        raise ValueError(
            "Unsupported proposal.encoder_backbone=%r with model.backbone=%r. "
            "Currently heterogeneous proposal encoders support only 'vjepa_img_encoder'."
            % (backbone, config.model.backbone)
        )

    encoder_params = sum(p.numel() for p in encoder.parameters())
    if _is_main_process():
        logger.info(f"init proposal_encoder_params: {encoder_params / 1e6:>8.2f}M")
    return encoder


def is_vjepa_encoder(module: Optional[nn.Module]) -> bool:
    if module is None:
        return False
    core = module.module if hasattr(module, "module") else module
    return bool(getattr(core, "is_vjepa_img_encoder_adapter", False)) or (
        core.__class__.__name__ == "VJEPAImgEncoderAdapter"
    )


def is_dinov2_encoder(module: Optional[nn.Module]) -> bool:
    if module is None:
        return False
    core = module.module if hasattr(module, "module") else module
    return bool(getattr(core, "is_dinov2_img_encoder_adapter", False)) or (
        core.__class__.__name__ == "Dinov2ImageEncoderAdapter"
    )


def is_pretrained_image_encoder(module: Optional[nn.Module]) -> bool:
    return is_vjepa_encoder(module) or is_dinov2_encoder(module)


def configure_pretrained_image_encoder_trainability(
    encoder: Optional[nn.Module],
    config: TrainingConfig,
    trainable: Optional[bool] = None,
) -> None:
    """Apply the freeze/eval policy for factory-pretrained image encoders."""
    if not is_pretrained_image_encoder(encoder):
        return
    core = encoder.module if hasattr(encoder, "module") else encoder
    should_train = bool(config.train.encoder_train if trainable is None else trainable)
    if should_train:
        core.train()
        for parameter in core.parameters():
            parameter.requires_grad = True
        return
    core.eval()
    for parameter in core.parameters():
        parameter.requires_grad = False


def configure_vjepa_encoder_trainability(
    encoder: Optional[nn.Module],
    config: TrainingConfig,
    trainable: Optional[bool] = None,
) -> None:
    """Apply the main V-JEPA freeze/eval policy after init or DDP wrapping."""
    if is_vjepa_encoder(encoder):
        configure_pretrained_image_encoder_trainability(encoder, config, trainable=trainable)


def should_save_main_encoder(config: TrainingConfig) -> bool:
    """Return whether checkpoints should include the main encoder state."""
    return bool(config.train.encoder_train or is_factory_pretrained_main_encoder_config(config))


def get_encoder_embed_dim(encoder: nn.Module) -> int:
    """
    获取 encoder 的嵌入维度

    Args:
        encoder: encoder 模型

    Returns:
        int: 嵌入维度
    """
    core = encoder.module if hasattr(encoder, "module") else encoder
    if hasattr(core, "embed_dim"):
        return int(core.embed_dim)
    return int(core.backbone.embed_dim)
