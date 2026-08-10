"""Predictor initialization vendored verbatim from app/vjepa_droid/utils.py."""

import logging

import torch

import src.models.ac_predictor as vit_ac_pred

logger = logging.getLogger(__name__)


def _build_grad_scaler(mixed_precision=False):
    if not mixed_precision:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler()
    return torch.cuda.amp.GradScaler()


def init_predictor_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    pred_depth=6,
    pred_num_heads=None,
    pred_embed_dim=384,
    embed_dim=1024,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=False,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    return_all_tokens=False,
    action_embed_dim=7,
    state_embed_dim=None,
    use_extrinsics=False,
    command_dim=0,
    old_pred=False,
    use_perceiver_ema=False,
    target_shape=None,
):

    predictor = vit_ac_pred.__dict__["vit_ac_predictor"](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=pred_embed_dim,
        action_embed_dim=action_embed_dim,
        state_embed_dim=state_embed_dim,
        depth=pred_depth,
        is_frame_causal=pred_is_frame_causal,
        num_heads=pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_sdpa=use_sdpa,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_extrinsics=use_extrinsics,
        use_activation_checkpointing=use_activation_checkpointing,
        use_perceiver_ema=use_perceiver_ema,
        target_shape=target_shape,
        command_dim=command_dim,
    )
    predictor.to(device)
    logger.info(predictor)

    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    # logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return predictor
