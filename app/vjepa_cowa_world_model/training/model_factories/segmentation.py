"""Split from training/models.py (verbatim node moves). Part: segmentation."""

from typing import Optional, Tuple

import torch
import torch.nn as nn


def init_segmentation_modules(
    device: torch.device,
    use_segmentation: bool = True,
    encoder_embed_dim: Optional[int] = None,
    num_classes: int = 2,
    loss_seg_weight: float = 2.0,
    loss_dice_weight: float = 5.0,
) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
    """
    初始化 seg_neck 和 seg_head

    Args:
        device: 设备
        use_segmentation: 是否使用分割模块
        encoder_embed_dim: encoder 嵌入维度；seg neck 的输入通道必须与之匹配（fail-loud：
            use_segmentation=True 时必须提供，否则换 ViT 尺寸会静默 shape 不匹配）

    Returns:
        Tuple[Optional[nn.Module], Optional[nn.Module]]: (seg_neck, seg_head)
    """
    if not use_segmentation:
        return None, None

    # fail-loud: seg neck 输入宽度必须等于 encoder embed dim（旧代码写死 1408 = ViT-giant，
    # 换 ViT-Large/Small 会静默 shape 崩）。
    if encoder_embed_dim is None:
        raise ValueError(
            "init_segmentation_modules requires encoder_embed_dim when use_segmentation=True "
            "(seg neck input width must match the encoder embedding dim)."
        )

    # Lazy import to avoid requiring torchcv when segmentation is disabled
    from app.vjepa_cowa_world_model.models.seg.co_detr_decoder import CoDetrDecoder
    from app.vjepa_cowa_world_model.models.seg.seg_head2 import SimpleSemanticSegHead
    from app.vjepa_cowa_world_model.models.seg.seg_neck2 import SFP

    seg_neck = SFP(input_channels=[int(encoder_embed_dim)], out_channels=256, use_p2=True, use_act_checkpoint=False)

    seg_head = SimpleSemanticSegHead(
        input_strides=[4, 8, 16, 32, 64],
        num_classes=num_classes,
        decoder=CoDetrDecoder(
            num_proposals=1500,
            embed_dims=256,
            num_heads=8,
            num_levels=5,
            dropout=0.0,
            feedforward_channels=2048,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            two_stage=False,
            num_co_heads=0,
            with_coord_feat=False,
            with_pos_coord=False,
        ),
        embed_dims=256,
        out_mask_dim=256,  # 占位
        loss_weights={
            "loss_seg": loss_seg_weight,
            "loss_dice": loss_dice_weight,
        },
        subcat_num=0,
    )

    seg_neck = seg_neck.to(device)
    seg_head = seg_head.to(device)

    return seg_neck, seg_head
