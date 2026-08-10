"""Split from training/models.py (verbatim node moves). Part: predictor."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.config import (
    TrainingConfig,
    _get_nested_value,
    is_vjepa_main_encoder_config,
    resolve_effective_tokens_per_frame,
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
    resolve_main_encoder_predictor_img_size,
    resolve_main_encoder_raw_tokens_per_frame,
    resolve_main_encoder_tokens_per_frame,
    resolve_predictor_runtime_normalize_reps,
)
from app.vjepa_cowa_world_model.training.configs.core import PREDICTOR_TYPES
from app.vjepa_cowa_world_model.training.model_factories.common import _is_main_process, logger


def _ac_predictor_kwargs(config, device, *, crop_size, embed_dim, state_embed_dim, command_dim):
    """Shared `init_predictor_model` kwargs for the AC-transformer predictor.

    Both the standard path (`init_predictor`) and the Token-AE path (`init_predictor_for_ae`) build the
    same predictor; only crop_size / embed_dim / state_embed_dim / command_dim differ per call site.
    Returned identical otherwise (byte-identical to the prior two copies).
    """
    return dict(
        uniform_power=config.model.uniform_power,
        device=device,
        patch_size=config.data.patch_size,
        max_num_frames=512,
        tubelet_size=config.data.tubelet_size,
        model_name=config.model.model_name,
        crop_size=crop_size,
        pred_depth=config.model.pred_depth,
        pred_num_heads=config.model.pred_num_heads,
        pred_embed_dim=config.model.pred_embed_dim,
        embed_dim=embed_dim,
        action_embed_dim=config.train.action_dim,
        state_embed_dim=state_embed_dim,
        command_dim=command_dim,
        pred_is_frame_causal=config.model.pred_is_frame_causal,
        use_extrinsics=config.model.use_extrinsics,
        use_sdpa=config.meta.use_sdpa,
        use_silu=config.model.use_silu,
        use_pred_silu=config.model.use_pred_silu,
        wide_silu=config.model.wide_silu,
        use_rope=config.model.use_rope,
        use_activation_checkpointing=config.model.use_activation_checkpointing,
        use_perceiver_ema=config.train.perceiver_ema,
        target_shape=None,
    )


def init_predictor(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    predictor_img_size_override=None,
    tokens_per_frame_override: Optional[int] = None,
) -> nn.Module:
    """
    初始化 action-conditioned predictor

    Args:
        config: 训练配置
        device: 设备
        encoder_embed_dim: encoder 的嵌入维度

    Returns:
        nn.Module: predictor 模型
    """
    predictor_type = str(config.train.predictor_type).lower()
    if predictor_type == "latent_dit":
        tokens_per_frame = (
            int(tokens_per_frame_override)
            if tokens_per_frame_override is not None
            else resolve_effective_tokens_per_frame(config)
        )
        return _init_latent_dit_predictor(
            config=config,
            device=device,
            encoder_embed_dim=encoder_embed_dim,
            tokens_per_frame=tokens_per_frame,
        )
    if predictor_type != "ac_transformer":
        raise ValueError(f"Unknown train.predictor_type={predictor_type!r}; expected one of {PREDICTOR_TYPES}.")

    logger.info("begin init action-conditioned predictor:")

    # use_drive_command=False 时：去掉 4 维 cmd，command_dim 强制 0
    _use_cmd = config.train.use_drive_command
    _state_dim = config.train.state_dim
    _command_dim = config.train.command_dim
    if not _use_cmd:
        _state_dim = _state_dim - 4
        _command_dim = 0
        logger.info(
            f"use_drive_command=False: predictor state_dim {_state_dim + 4} -> {_state_dim}, command_dim forced to 0"
        )

    # 断言：predictor command_dim > 0 要求 split_status_embedding 开启
    if (
        _command_dim > 0
        and not config.planner.split_status_embedding
        and not config.train.predictor_inference_consistent
    ):
        raise ValueError(
            f"predictor command_dim={_command_dim} > 0 but planner.split_status_embedding=False. "
            "Enable split_status_embedding or set command_dim=0."
        )

    from app.vjepa_cowa_world_model.models.backbones.droid_predictor import init_predictor_model

    # 计算 state_embed_dim: 当 state_dim != action_dim 时使用独立维度
    _state_embed_dim = None
    if _state_dim != config.train.action_dim:
        _state_embed_dim = _state_dim

    predictor_img_size = (
        predictor_img_size_override if predictor_img_size_override is not None else config.data.crop_size
    )

    predictor = init_predictor_model(
        **_ac_predictor_kwargs(
            config,
            device,
            crop_size=predictor_img_size,
            embed_dim=encoder_embed_dim,
            state_embed_dim=_state_embed_dim,
            command_dim=_command_dim,
        )
    )

    logger.info(
        f"end init predictor (action_embed_dim={config.train.action_dim}, "
        f"state_embed_dim={_state_embed_dim}, command_dim={_command_dim}, "
        f"use_drive_command={_use_cmd}, predictor_img_size={predictor_img_size})"
    )

    # 打印参数量（仅主进程）
    predictor_params = sum(p.numel() for p in predictor.parameters())
    if _is_main_process():
        logger.info(f"init predictor_params: {predictor_params / 1e6:>8.2f}M")

    return predictor


def _init_latent_dit_predictor(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    tokens_per_frame: int,
) -> nn.Module:
    """Initialize the parallel latent-token DiT predictor."""
    from app.vjepa_cowa_world_model.models import LatentDiTPredictor

    objective = str(config.predictor_dit.objective).lower()
    valid_objectives = {"flow_matching", "x0_prediction"}
    if objective not in valid_objectives:
        raise ValueError("predictor_dit.objective must be one of " f"{sorted(valid_objectives)}, got {objective!r}")
    if bool(getattr(config.predictor_dit, "joint_action_enabled", False)):
        joint_action_noise_mode = str(getattr(config.predictor_dit, "joint_action_noise_mode", "shared")).lower()
        if joint_action_noise_mode not in {"shared", "decoupled"}:
            raise ValueError("predictor_dit.joint_action_noise_mode must be one of ['shared', 'decoupled']")
        if str(getattr(config.predictor_dit, "joint_action_state_mode", "last_observed")).lower() != "last_observed":
            raise ValueError("predictor_dit.joint_action_state_mode currently supports only 'last_observed'")
        if str(getattr(config.predictor_dit, "joint_action_guidance_mode", "cond_only")).lower() != "cond_only":
            raise ValueError("predictor_dit.joint_action_guidance_mode currently supports only 'cond_only'")
    joint_action_inference_noise_mode = str(
        getattr(config.predictor_dit, "joint_action_inference_noise_mode", "shared")
    ).lower()
    if joint_action_inference_noise_mode not in {"shared", "decoupled"}:
        raise ValueError("predictor_dit.joint_action_inference_noise_mode must be 'shared' or 'decoupled'")
    joint_video_final_noise = float(getattr(config.predictor_dit, "joint_video_final_noise", 0.0))
    if not 0.0 <= joint_video_final_noise < 1.0:
        raise ValueError(f"predictor_dit.joint_video_final_noise must be in [0.0, 1.0), got {joint_video_final_noise}")
    if joint_video_final_noise > 0.0 and joint_action_inference_noise_mode != "decoupled":
        raise ValueError(
            "predictor_dit.joint_video_final_noise > 0 requires "
            "predictor_dit.joint_action_inference_noise_mode='decoupled'"
        )

    num_total_steps = resolve_main_encoder_num_time_steps(
        config,
        num_raw_frames=int(config.data.num_target_frames),
    )
    num_observed_steps = resolve_main_encoder_num_observed_steps(config)
    num_future_steps = num_total_steps - num_observed_steps
    if num_future_steps <= 0:
        raise ValueError(
            "LatentDiTPredictor requires at least one future step: "
            f"total_steps={num_total_steps}, observed_steps={num_observed_steps}"
        )

    # use_drive_command=False 时去掉 4 维 drive-command（与其它 predictor init 一致；main 在 models.py
    # 把这一逻辑也补到了 LatentDiT，这里用直接索引保持 fail-loud）。
    _use_drive_command = config.train.use_drive_command
    _state_dim = int(config.train.state_dim)
    if not _use_drive_command:
        _state_dim = _state_dim - 4
        if _state_dim <= 0:
            raise ValueError(
                "train.use_drive_command=false removes the 4 command dimensions, "
                f"but train.state_dim={int(config.train.state_dim)} leaves effective state_dim={_state_dim}."
            )

    predictor = LatentDiTPredictor(
        embed_dim=int(encoder_embed_dim),
        tokens_per_frame=int(tokens_per_frame),
        num_future_steps=int(num_future_steps),
        action_dim=int(config.train.action_dim),
        state_dim=_state_dim,
        extrinsics_dim=7,
        hidden_dim=int(config.predictor_dit.hidden_dim),
        depth=int(config.predictor_dit.depth),
        num_heads=int(config.predictor_dit.num_heads),
        dropout=float(config.predictor_dit.dropout),
        x0_loss_weight=float(config.predictor_dit.x0_loss_weight),
        bottleneck_dim=config.predictor_dit.bottleneck_dim,
        max_steps=int(config.predictor_dit.max_condition_steps),
        conditioning_mode=str(config.predictor_dit.conditioning_mode),
        use_anchor_frame=bool(config.predictor_dit.use_anchor_frame),
        objective=objective,
        joint_action_enabled=bool(getattr(config.predictor_dit, "joint_action_enabled", False)),
        joint_action_dim=int(getattr(config.predictor_dit, "joint_action_dim", int(config.train.action_dim))),
        joint_action_state_dim=int(getattr(config.predictor_dit, "joint_action_state_dim", _state_dim)),
        joint_action_inference_noise_mode=joint_action_inference_noise_mode,
        joint_video_final_noise=joint_video_final_noise,
    ).to(device)
    logger.info(
        "Initialized LatentDiTPredictor: tokens_per_frame=%d future_steps=%d hidden_dim=%d depth=%d heads=%d "
        "state_dim=%d raw_state_dim=%d use_drive_command=%s bottleneck_dim=%s conditioning_mode=%s "
        "use_anchor_frame=%s objective=%s",
        int(tokens_per_frame),
        int(num_future_steps),
        int(config.predictor_dit.hidden_dim),
        int(config.predictor_dit.depth),
        int(config.predictor_dit.num_heads),
        _state_dim,
        int(config.train.state_dim),
        _use_drive_command,
        str(config.predictor_dit.bottleneck_dim),
        str(config.predictor_dit.conditioning_mode),
        str(config.predictor_dit.use_anchor_frame),
        objective,
    )
    return predictor


def init_predictor_for_ae(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    num_latent_tokens: int,
    embed_dim_override: Optional[int] = None,
    latent_grid_size: Optional[Tuple[int, int]] = None,
) -> nn.Module:
    """
    初始化适配 Token AE 压缩 token 的 predictor。

    Token AE 将 encoder 的 256 tokens/frame 压缩到 num_latent_tokens (如 64)。
    原始 predictor 假设 tokens_per_frame = (crop_size / patch_size)²，
    内部用 grid_height × grid_width 做帧级 reshape 和 frame-causal attention mask。

    适配方法：用 virtual crop_size 使 (crop_height / patch_size) × (crop_width / patch_size)
    = num_latent_tokens，例如 num_latent_tokens=32 且 latent_grid_size=(4, 8)
    → virtual crop_size = (64, 128)。

    注意：
    - predictor 的所有 Linear 权重与 grid 尺寸无关，预训练权重可直接加载
    - 仅 attention mask 和 RoPE 根据 grid 尺寸动态计算（非学习参数）
    - 由于空间假设变了，predictor 需要 fine-tune (建议 predictor_train=True)

    Args:
        config: 训练配置
        device: 设备
        encoder_embed_dim: encoder 嵌入维度 (如 1408)
        num_latent_tokens: AE 压缩后每帧 token 数 (如 64)
        embed_dim_override: predictor I/O 维度覆盖值；默认沿用 encoder_embed_dim
        latent_grid_size: AE latent token 的空间网格 (H, W)；为空时兼容旧的方形推导

    Returns:
        nn.Module: 适配后的 predictor
    """
    patch_size = config.data.patch_size  # typically 16
    if latent_grid_size is None and getattr(config, "token_ae", None) is not None:
        latent_grid_size = config.token_ae.latent_grid_size
    if latent_grid_size is None:
        import math

        virtual_grid = int(math.sqrt(num_latent_tokens))
        if virtual_grid * virtual_grid != num_latent_tokens:
            raise ValueError(
                f"num_latent_tokens={num_latent_tokens} 必须是完全平方数，或显式配置 token_ae.latent_grid_size；"
                f"sqrt={math.sqrt(num_latent_tokens):.2f}"
            )
        virtual_grid_height = virtual_grid
        virtual_grid_width = virtual_grid
    else:
        if not isinstance(latent_grid_size, (list, tuple)) or len(latent_grid_size) != 2:
            raise ValueError(f"token_ae.latent_grid_size must be a 2-element sequence, got {latent_grid_size!r}")
        virtual_grid_height = int(latent_grid_size[0])
        virtual_grid_width = int(latent_grid_size[1])
        if virtual_grid_height <= 0 or virtual_grid_width <= 0:
            raise ValueError(f"token_ae.latent_grid_size must be positive, got {latent_grid_size!r}")
        if virtual_grid_height * virtual_grid_width != int(num_latent_tokens):
            raise ValueError(
                "token_ae.latent_grid_size does not match num_latent_tokens: "
                f"latent_grid_size={latent_grid_size!r}, num_latent_tokens={num_latent_tokens}"
            )
    virtual_crop_size = (
        virtual_grid_height * patch_size
        if virtual_grid_height == virtual_grid_width
        else (virtual_grid_height * patch_size, virtual_grid_width * patch_size)
    )

    logger.info(
        "init predictor for Token AE: num_latent_tokens=%d → virtual_grid=%d×%d → "
        "virtual_crop_size=%s (original crop_size=%s)",
        num_latent_tokens,
        virtual_grid_height,
        virtual_grid_width,
        virtual_crop_size,
        config.data.crop_size,
    )

    # use_drive_command=False 时：去掉 4 维 cmd，command_dim 强制 0
    _use_cmd = config.train.use_drive_command
    _state_dim = config.train.state_dim
    _command_dim = config.train.command_dim
    if not _use_cmd:
        _state_dim = _state_dim - 4
        _command_dim = 0
        logger.info(
            f"use_drive_command=False: predictor state_dim {_state_dim + 4} -> {_state_dim}, command_dim forced to 0"
        )

    # 计算 state_embed_dim: 当 state_dim != action_dim 时使用独立维度
    _state_embed_dim = None
    if _state_dim != config.train.action_dim:
        _state_embed_dim = _state_dim

    predictor_io_dim = int(embed_dim_override) if embed_dim_override is not None else int(encoder_embed_dim)

    from app.vjepa_cowa_world_model.models.backbones.droid_predictor import init_predictor_model

    predictor = init_predictor_model(
        **_ac_predictor_kwargs(
            config,
            device,
            crop_size=virtual_crop_size,  # 虚拟 crop_size 使 grid 匹配 AE token 数
            embed_dim=predictor_io_dim,
            state_embed_dim=_state_embed_dim,
            command_dim=_command_dim,
        )
    )

    predictor_params = sum(p.numel() for p in predictor.parameters())
    logger.info(
        "predictor for AE: params=%.2fM, virtual_grid=%d×%d, tokens_per_frame=%d, io_dim=%d, "
        "action_embed_dim=%d, state_embed_dim=%s, command_dim=%d, use_drive_command=%s",
        predictor_params / 1e6,
        virtual_grid_height,
        virtual_grid_width,
        num_latent_tokens,
        predictor_io_dim,
        config.train.action_dim,
        _state_embed_dim,
        _command_dim,
        _use_cmd,
    )

    # Enable gradient checkpointing to reduce memory usage during training
    # This is especially beneficial with LoRA adapters: reduces activation memory ~30-40%
    if hasattr(predictor, "gradient_checkpointing_enable"):
        try:
            predictor.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for predictor (memory ~30-40% reduction)")
        except Exception as e:
            logger.warning("Could not enable gradient checkpointing for predictor: %s", e)

    return predictor


def load_frozen_token_ae(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    tokens_per_frame: int,
    normalize_reps: bool,
    dtype: Optional[torch.dtype] = None,
    _checkpoint_weights_only: Optional[bool] = None,
    _defer_state_dict_load: bool = False,
):
    """Load a frozen TokenAE from checkpoint for training or validation runtime use."""
    if not getattr(config, "token_ae", None) or not config.token_ae.enabled:
        return None, normalize_reps
    if type(_defer_state_dict_load) is not bool:
        raise TypeError("_defer_state_dict_load must be bool")

    ae_checkpoint = getattr(config.meta, "ae_checkpoint", None)
    if not ae_checkpoint:
        raise ValueError("meta.ae_checkpoint must be provided when token_ae.enabled=True")

    from app.vjepa_cowa_world_model.models.token_ae import TokenAE

    load_kwargs = {}
    if _checkpoint_weights_only is not None:
        if type(_checkpoint_weights_only) is not bool:
            raise TypeError("_checkpoint_weights_only must be bool or None")
        load_kwargs["weights_only"] = _checkpoint_weights_only
    token_ae_ckpt = torch.load(ae_checkpoint, map_location="cpu", **load_kwargs)
    token_ae_cfg = token_ae_ckpt.get("config", {})
    token_ae_embed_dim = int(token_ae_cfg.get("embed_dim", encoder_embed_dim))
    token_ae_tokens_per_frame = int(token_ae_cfg.get("tokens_per_frame", tokens_per_frame))
    token_ae_normalize_reps = bool(token_ae_cfg.get("normalize_reps", normalize_reps))

    if token_ae_embed_dim != encoder_embed_dim:
        raise ValueError(
            f"Token AE embed_dim={token_ae_embed_dim} does not match encoder embed_dim={encoder_embed_dim}"
        )
    if token_ae_tokens_per_frame != tokens_per_frame:
        raise ValueError(
            "Token AE tokens_per_frame="
            f"{token_ae_tokens_per_frame} does not match current data setting {tokens_per_frame}"
        )

    configured_num_latent_tokens = int(config.token_ae.num_latent_tokens)
    checkpoint_num_latent_tokens = int(token_ae_cfg.get("num_latent_tokens", configured_num_latent_tokens))
    if checkpoint_num_latent_tokens != configured_num_latent_tokens:
        raise ValueError(
            "Token AE num_latent_tokens mismatch: "
            f"checkpoint={checkpoint_num_latent_tokens}, config={configured_num_latent_tokens}"
        )

    token_ae = TokenAE(
        embed_dim=token_ae_embed_dim,
        tokens_per_frame=token_ae_tokens_per_frame,
        num_latent_tokens=checkpoint_num_latent_tokens,
        num_heads=int(token_ae_cfg.get("num_heads", config.token_ae.num_heads)),
        encoder_depth=int(token_ae_cfg.get("encoder_depth", config.token_ae.encoder_depth)),
        decoder_depth=int(token_ae_cfg.get("decoder_depth", config.token_ae.decoder_depth)),
        mlp_ratio=float(token_ae_cfg.get("mlp_ratio", config.token_ae.mlp_ratio)),
        dropout=float(token_ae_cfg.get("dropout", config.token_ae.dropout)),
        encoder_mode=token_ae_cfg.get("encoder_mode", config.token_ae.encoder_mode),
        loss_type=token_ae_cfg.get("loss_type", config.token_ae.loss_type),
        cos_loss_weight=float(token_ae_cfg.get("cos_loss_weight", config.token_ae.cos_loss_weight)),
        latent_reg_weight=float(token_ae_cfg.get("latent_reg_weight", config.token_ae.latent_reg_weight)),
        pos_embed_type=token_ae_cfg.get("pos_embed_type", config.token_ae.pos_embed_type),
        input_grid_size=token_ae_cfg.get("input_grid_size", config.token_ae.input_grid_size),
        latent_grid_size=token_ae_cfg.get("latent_grid_size", config.token_ae.latent_grid_size),
        temporal_depth=int(token_ae_cfg.get("temporal_depth", config.token_ae.temporal_depth)),
        temporal_num_heads=token_ae_cfg.get("temporal_num_heads", config.token_ae.temporal_num_heads),
        temporal_mlp_ratio=token_ae_cfg.get("temporal_mlp_ratio", config.token_ae.temporal_mlp_ratio),
        temporal_causal=bool(token_ae_cfg.get("temporal_causal", config.token_ae.temporal_causal)),
        temporal_mode=token_ae_cfg.get("temporal_mode", config.token_ae.temporal_mode),
        temporal_pos_embed_type=token_ae_cfg.get("temporal_pos_embed_type", config.token_ae.temporal_pos_embed_type),
        temporal_loss_weight=float(token_ae_cfg.get("temporal_loss_weight", config.token_ae.temporal_loss_weight)),
    )
    if dtype is None:
        token_ae = token_ae.to(device=device)
    else:
        token_ae = token_ae.to(device=device, dtype=dtype)

    token_ae_state = token_ae_ckpt["token_ae"] if "token_ae" in token_ae_ckpt else token_ae_ckpt
    if not _defer_state_dict_load:
        token_ae.load_state_dict(token_ae_state, strict=True)

    for parameter in token_ae.parameters():
        parameter.requires_grad = False
    token_ae.eval()

    # Stamp the resolved normalization flag onto the module so downstream consumers
    # (e.g. validation) can reuse the exact training-time value without re-reading
    # the checkpoint. Plain attribute -> not part of state_dict.
    token_ae.runtime_normalize_reps = bool(token_ae_normalize_reps)

    return token_ae, token_ae_normalize_reps


def resolve_runtime_normalize_reps(config, token_ae: Optional[nn.Module] = None) -> bool:
    """Resolve the effective representation-normalization flag used at runtime.

    Mirrors ``init_predictor_runtime_with_token_ae``: the predictor checkpoint's
    adjacent pretrain config provides the base value, and an enabled TokenAE
    checkpoint's embedded ``config.normalize_reps`` overrides it. Validation/eval
    must use this (not the raw experiment config) so the planner input is
    normalized exactly as during training.
    """
    normalize_reps = resolve_predictor_runtime_normalize_reps(config)

    token_ae_enabled = bool(_get_nested_value(config, "token_ae", "enabled", default=False))
    if not token_ae_enabled:
        return normalize_reps

    # Fast path: reuse the value already resolved when the TokenAE was loaded.
    if token_ae is not None:
        stamped = getattr(token_ae, "runtime_normalize_reps", None)
        if stamped is not None:
            return bool(stamped)

    # 读取 TokenAE checkpoint 内嵌 config 的 normalize_reps（point 21）。
    ae_checkpoint = _get_nested_value(config, "meta", "ae_checkpoint", default=None)
    if ae_checkpoint:
        # fail-loud (point 21): 读取失败直接报错，禁止 except 吞掉后回退到实验配置 flag——
        # 那会让 planner 输入归一化方式与训练时不一致。
        token_ae_ckpt = torch.load(ae_checkpoint, map_location="cpu")
        token_ae_cfg = token_ae_ckpt.get("config", {}) if isinstance(token_ae_ckpt, dict) else {}
        override = token_ae_cfg.get("normalize_reps", None)
        if override is not None:
            normalize_reps = bool(override)
    return normalize_reps


def prepare_runtime_tokens(
    tokens: torch.Tensor,
    num_frames: int,
    normalize_reps: bool,
    token_ae: Optional[nn.Module] = None,
) -> torch.Tensor:
    """Compress per-frame tokens with TokenAE if present, then apply runtime normalization."""
    input_dtype = tokens.dtype
    if token_ae is not None:
        ae_tokens_per_frame = int(getattr(token_ae, "tokens_per_frame"))
        expected_tokens = int(num_frames) * ae_tokens_per_frame
        if tokens.size(1) != expected_tokens:
            if tokens.size(1) % ae_tokens_per_frame != 0:
                raise ValueError(
                    "Cannot infer TokenAE frame count: "
                    f"tokens={tokens.size(1)}, num_frames={num_frames}, "
                    f"ae_tokens_per_frame={ae_tokens_per_frame}"
                )
            num_frames = tokens.size(1) // ae_tokens_per_frame
        token_ae_parameters = getattr(token_ae, "parameters", None)
        ae_param = next(token_ae_parameters(), None) if callable(token_ae_parameters) else None
        if ae_param is not None and tokens.is_floating_point() and tokens.dtype != ae_param.dtype:
            tokens = tokens.to(dtype=ae_param.dtype)
        tokens = token_ae.encode(tokens, num_frames=num_frames)
        if tokens.is_floating_point() and tokens.dtype != input_dtype:
            tokens = tokens.to(dtype=input_dtype)
    if normalize_reps:
        tokens = F.layer_norm(tokens, (tokens.size(-1),))
    return tokens


def register_predictor_future_query_tokens(
    predictor: nn.Module,
    embed_dim: int,
    future_tubelets: int,
    tokens_per_frame: int,
    device: torch.device,
    init_std: float = 0.02,
) -> None:
    """Register learnable predictor query tokens for future tubelets."""
    if future_tubelets <= 0:
        return

    predictor_core = predictor.module if hasattr(predictor, "module") else predictor
    num_future_tokens = future_tubelets * tokens_per_frame
    expected_shape = (1, num_future_tokens, embed_dim)
    existing = getattr(predictor_core, "future_query_tokens", None)
    if existing is not None:
        if tuple(existing.shape) != expected_shape:
            raise ValueError(
                "Existing predictor.future_query_tokens shape mismatch: "
                f"got {tuple(existing.shape)}, expected {expected_shape}"
            )
        return

    future_query_tokens = nn.Parameter(torch.empty(expected_shape, device=device))
    nn.init.trunc_normal_(future_query_tokens, std=init_std)
    predictor_core.register_parameter("future_query_tokens", future_query_tokens)
    logger.info(
        "Registered predictor future query tokens: future_tubelets=%d, tokens=%d, embed_dim=%d",
        future_tubelets,
        num_future_tokens,
        embed_dim,
    )


def build_predictor_input_with_future_queries(predictor: nn.Module, observed_tokens: torch.Tensor) -> torch.Tensor:
    """Append learnable future query tokens after observed predictor input tokens."""
    predictor_core = predictor.module if hasattr(predictor, "module") else predictor
    future_query_tokens = getattr(predictor_core, "future_query_tokens", None)
    if future_query_tokens is None:
        return observed_tokens
    return torch.cat([observed_tokens, future_query_tokens.expand(observed_tokens.size(0), -1, -1)], dim=1)


def init_predictor_runtime_with_token_ae(
    config: TrainingConfig,
    device: torch.device,
    encoder_embed_dim: int,
    raw_tokens_per_frame_override: Optional[int] = None,
    predictor_img_size_override=None,
    _checkpoint_weights_only: Optional[bool] = None,
    _defer_token_ae_state_load: bool = False,
):
    """Initialize predictor plus optional frozen TokenAE runtime state for training scripts."""
    raw_tokens_per_frame = (
        int(raw_tokens_per_frame_override)
        if raw_tokens_per_frame_override is not None
        else config.data.tokens_per_frame
    )
    token_ae_enabled = bool(getattr(config, "token_ae", None) and config.token_ae.enabled)
    if token_ae_enabled:
        effective_tokens_per_frame = resolve_effective_tokens_per_frame(config)
    else:
        effective_tokens_per_frame = (
            int(raw_tokens_per_frame_override)
            if raw_tokens_per_frame_override is not None
            else resolve_effective_tokens_per_frame(config)
        )
    runtime_normalize_reps = resolve_predictor_runtime_normalize_reps(config)
    token_ae = None

    if token_ae_enabled:
        if type(_defer_token_ae_state_load) is not bool:
            raise TypeError("_defer_token_ae_state_load must be bool")
        token_ae_load_kwargs = {}
        if _checkpoint_weights_only is not None:
            if type(_checkpoint_weights_only) is not bool:
                raise TypeError("_checkpoint_weights_only must be bool or None")
            token_ae_load_kwargs["_checkpoint_weights_only"] = _checkpoint_weights_only
        if _defer_token_ae_state_load:
            token_ae_load_kwargs["_defer_state_dict_load"] = True
        token_ae, runtime_normalize_reps = load_frozen_token_ae(
            config,
            device=device,
            encoder_embed_dim=encoder_embed_dim,
            tokens_per_frame=raw_tokens_per_frame,
            normalize_reps=runtime_normalize_reps,
            dtype=config.dtype,
            **token_ae_load_kwargs,
        )
        if str(config.train.predictor_type).lower() == "latent_dit":
            predictor = _init_latent_dit_predictor(
                config=config,
                device=device,
                encoder_embed_dim=encoder_embed_dim,
                tokens_per_frame=effective_tokens_per_frame,
            )
        else:
            predictor = init_predictor_for_ae(
                config,
                device=device,
                encoder_embed_dim=encoder_embed_dim,
                num_latent_tokens=effective_tokens_per_frame,
                latent_grid_size=getattr(token_ae, "latent_grid_size", config.token_ae.latent_grid_size),
            )
    else:
        predictor = init_predictor(
            config,
            device,
            encoder_embed_dim,
            predictor_img_size_override=predictor_img_size_override,
            tokens_per_frame_override=effective_tokens_per_frame,
        )

    return predictor, token_ae, effective_tokens_per_frame, runtime_normalize_reps


def resolve_main_predictor_runtime_overrides(config: TrainingConfig, encoder: Optional[nn.Module] = None):
    """Return raw-token/grid overrides needed by non-default main encoders."""
    if not is_vjepa_main_encoder_config(config):
        if bool(config.multiview.enabled):
            return resolve_main_encoder_tokens_per_frame(config, encoder), resolve_main_encoder_predictor_img_size(
                config, encoder
            )
        return None, None
    if bool(_get_nested_value(config, "token_ae", "enabled", default=False)):
        return resolve_main_encoder_raw_tokens_per_frame(config, encoder), resolve_main_encoder_predictor_img_size(
            config, encoder
        )
    return resolve_main_encoder_tokens_per_frame(config, encoder), resolve_main_encoder_predictor_img_size(
        config, encoder
    )
