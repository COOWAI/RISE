"""Shared runtime helpers for encoder-direct NavSim training and evaluation."""

import copy
from typing import Any, Optional

import torch
import torch.nn.functional as F  # noqa: F401 - kept for downstream parity imports

from app.vjepa_cowa_world_model.training.configs.planner import PLANNER_TYPES
from app.vjepa_cowa_world_model.training.encoder_inputs import build_tubelet_encoder_input
from app.vjepa_cowa_world_model.utils import resolve_effective_planner_status_dim, resolve_planner_use_drive_command
from app.vjepa_cowa_world_model.utils.planner_training import (
    build_horizon_regression_timestep_weights,
    resolve_validation_timestep_sec,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_action_history_dt(config: Any) -> float:
    """Resolve timestep for 6D action-history velocity features."""
    fps = getattr(config.data, "fps", None)
    if fps is not None and float(fps) > 0:
        return 1.0 / float(fps)

    diff_dt = getattr(config.planner, "diff_dt", None)
    if diff_dt is not None and float(diff_dt) > 0:
        return float(diff_dt)

    return 1.0


def is_vjepa_img_encoder(config: Any) -> bool:
    """Whether encoder-direct runtime uses the V-JEPA image encoder adapter."""
    return getattr(getattr(config, "model", None), "backbone", "") == "vjepa_img_encoder"


def get_encoder_core(encoder: Optional[Any]) -> Optional[Any]:
    """Return the wrapped encoder module only for reading static attributes."""
    if encoder is None:
        return None
    return encoder.module if hasattr(encoder, "module") else encoder


def should_load_generic_pretrained_checkpoint(
    load_encoder: bool,
    load_predictor: bool,
    load_seg: bool,
    load_planner: bool,
) -> bool:
    """Return whether any generic pretrained checkpoint component is requested."""
    return bool(load_encoder or load_predictor or load_seg or load_planner)


def configure_vjepa_adapter_trainability(encoder: Any, config: Any) -> None:
    """Freeze V-JEPA backbone in encoder-direct mode unless encoder_train is true."""
    if not is_vjepa_img_encoder(config):
        return
    if getattr(config.train, "encoder_train", False):
        return

    core = get_encoder_core(encoder)
    if core is None:
        raise AttributeError("V-JEPA encoder must be an adapter with a backbone module")

    backbone = getattr(core, "backbone", None)
    if backbone is None:
        raise AttributeError("V-JEPA adapter must expose a backbone module")

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()


def add_vjepa_projector_param_groups(optimizer: Any, encoder: Any, config: Any) -> None:
    """No-op kept for backward compatibility with older encoder-direct training code."""
    del optimizer, encoder, config
    return


def resolve_encoder_direct_tokens_per_frame(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve the effective planner token count per temporal step."""
    if is_vjepa_img_encoder(config):
        core = get_encoder_core(encoder)
        tokens_per_frame = getattr(core, "tokens_per_frame", None) if core is not None else None
        if tokens_per_frame is not None:
            return int(tokens_per_frame)

        height, width = getattr(config.model, "vjepa_resolution")
        patch_size = int(getattr(config.data, "patch_size"))
        return int((int(height) // patch_size) * (int(width) // patch_size))

    return int(config.data.tokens_per_frame)


def resolve_encoder_direct_num_time_steps(config: Any, encoder: Optional[Any] = None) -> int:
    """Resolve the effective planner temporal length for encoder-direct tokens."""
    if is_vjepa_img_encoder(config):
        core = get_encoder_core(encoder)
        num_time_steps = getattr(core, "num_time_steps", None) if core is not None else None
        if num_time_steps is not None:
            return int(num_time_steps)
        return 1

    return int(config.train.num_observed_frames)


def forward_encoder_direct_tokens(encoder: Any, context_clips: torch.Tensor, config: Any) -> torch.Tensor:
    """Forward encoder-direct context clips into planner token features."""
    if is_vjepa_img_encoder(config):
        return encoder(
            context_clips,
            num_observed_frames=config.train.num_observed_frames,
            use_causal_attention=getattr(config.model, "vjepa_use_causal_attention", True),
        )

    batch_size, _, max_num_frames, _, _ = context_clips.shape
    encoder_input = build_tubelet_encoder_input(context_clips, config.data.tubelet_size)
    z_context_out = encoder([encoder_input])
    z = z_context_out[0]
    return z.view(batch_size, max_num_frames, -1, z.size(-1)).flatten(1, 2)


def init_encoder_direct_encoder(config: Any, device: torch.device):
    """Initialize the encoder pair for encoder-direct runtime."""
    if is_vjepa_img_encoder(config):
        from app.vjepa_cowa_world_model.models.vjepa_img_encoder import VJEPAImgEncoderAdapter

        encoder = VJEPAImgEncoderAdapter(
            checkpoint_path=config.meta.pretrain_checkpoint_full,
            resolution=config.model.vjepa_resolution,
            num_frames=config.model.vjepa_num_frames,
            max_num_observed_frames=config.train.num_observed_frames,
            checkpoint_key=config.model.vjepa_checkpoint_key,
            model_name=config.model.model_name,
            patch_size=config.data.patch_size,
            tubelet_size=config.data.tubelet_size,
            uniform_power=config.model.uniform_power,
            use_rope=config.model.use_rope,
            use_sdpa=config.meta.use_sdpa,
            use_activation_checkpointing=config.model.use_activation_checkpointing,
            use_grid_mask=config.model.vjepa_use_grid_mask,
            use_causal_attention=getattr(config.model, "vjepa_use_causal_attention", True),
        ).to(device)
        target_encoder = copy.deepcopy(encoder)
        return encoder, target_encoder

    from app.vjepa_cowa_world_model.training.models import init_encoder

    return init_encoder(config, device)


def init_encoder_direct_planner(config: Any, encoder_dim: int, device: torch.device, encoder: Optional[Any] = None):
    """Initialize the planner exactly as encoder-direct training expects."""
    if not config.planner.use_planner:
        logger.info("use_planner=False, planner is disabled")
        return None

    num_observed = config.train.num_observed_frames
    total_frames = config.data.num_target_frames
    num_poses = total_frames - num_observed
    tokens_per_frame = resolve_encoder_direct_tokens_per_frame(config, encoder)
    encoder_direct_num_time_steps = resolve_encoder_direct_num_time_steps(config, encoder)
    use_action_history = bool(getattr(config.planner, "use_action_history_for_planner", False))
    action_history_dim = int(getattr(config.planner, "action_history_dim", 3))

    if not 1 <= num_observed < total_frames:
        raise ValueError(
            f"train.num_observed_frames must satisfy 1 <= num_observed_frames < num_target_frames, "
            f"got {num_observed} and {total_frames}"
        )

    use_command = resolve_planner_use_drive_command(config)
    planner_status_dim = resolve_effective_planner_status_dim(config)
    if use_command and config.planner.split_status_embedding and config.train.predictor_inference_consistent:
        planner_command_dim = 4
    else:
        planner_command_dim = 0

    status_layouts = {
        3: "[velocity, acceleration, yaw_rate]",
        4: "[vx, vy, ax, ay]",
        7: "[cmd(4), velocity, acceleration, yaw_rate]",
        8: "[cmd(4), vx, vy, ax, ay]" if use_command else "[vx, vy, ax, ay, x_local, y_local, sin_yaw, cos_yaw]",
        12: "[cmd(4), vx, vy, ax, ay, x_local, y_local, sin_yaw, cos_yaw]",
    }
    layout = status_layouts.get(planner_status_dim, f"custom({planner_status_dim}d)")
    logger.info(
        "[Status Summary] planner_status_dim=%d, command_dim=%d, use_drive_command=%s | layout: %s",
        planner_status_dim,
        planner_command_dim,
        use_command,
        layout,
    )

    planner_type = config.planner.planner_type
    if planner_type not in PLANNER_TYPES:
        raise ValueError(f"Unknown planner.planner_type={planner_type!r}; expected one of {PLANNER_TYPES}.")
    if planner_type == "diffusion":
        from app.vjepa_cowa_world_model.models import DiffusionPlanner

        planner_timestep_sec = resolve_validation_timestep_sec(
            fps=getattr(config.data, "fps", None),
            diff_dt=getattr(config.planner, "diff_dt", None),
            default=0.5,
        )
        diff_reg_timestep_weights = build_horizon_regression_timestep_weights(
            num_poses=num_poses,
            timestep_sec=planner_timestep_sec,
            horizon_seconds=config.planner.horizon_reg_loss_seconds,
            horizon_weights=config.planner.horizon_reg_loss_weights,
            normalize=config.planner.horizon_reg_loss_normalize,
            device=device,
            dtype=torch.float32,
        )
        if config.planner.horizon_reg_loss_seconds and diff_reg_timestep_weights is None:
            logger.warning(
                "[HorizonReg] configured horizons are outside num_poses=%d (timestep_sec=%.4f); disabled",
                num_poses,
                planner_timestep_sec,
            )

        planner = DiffusionPlanner(
            encoder_dim=encoder_dim,
            num_poses=num_poses,
            status_dim=planner_status_dim,
            hidden_dim=config.planner.diff_hidden_dim,
            depth=config.planner.diff_num_layers,
            heads=config.planner.diff_num_heads,
            dropout=config.planner.diff_dropout,
            mlp_ratio=config.planner.diff_mlp_ratio,
            traj_dim=config.planner.diff_traj_dim,
            sde_beta_min=config.planner.diff_sde_beta_min,
            sde_beta_max=config.planner.diff_sde_beta_max,
            num_samples=config.planner.diff_num_samples,
            inference_steps=config.planner.diff_inference_steps,
            use_z_context=False,
            tokens_per_frame=tokens_per_frame,
            use_action_history=use_action_history,
            action_history_dim=action_history_dim,
            num_observed_frames=num_observed,
            command_dim=planner_command_dim,
            num_modes=config.planner.diff_num_modes,
            use_anchor_frame=config.planner.diff_use_anchor_frame,
            trajectory_token_mode=config.planner.diff_trajectory_token_mode,
            use_last_frame_only=config.planner.diff_use_last_frame_only,
            independent_modes=getattr(config.planner, "diff_independent_modes", False),
            adaln_version=config.planner.diff_adaln_version,
            cls_loss_weight=config.planner.diff_cls_loss_weight,
            reg_loss_weight=config.planner.diff_reg_loss_weight,
            vel_loss_weight=config.planner.diff_vel_loss_weight,
            yaw_loss_weight=config.planner.diff_yaw_loss_weight,
            reg_timestep_weights=diff_reg_timestep_weights,
            awta_init_temperature=config.planner.awta_init_temperature,
            awta_min_temperature=config.planner.awta_min_temperature,
            conf_temperature=config.planner.diff_conf_temperature,
            cls_th=config.planner.diff_cls_th,
            cls_ignore=config.planner.diff_cls_ignore,
            mode_token_expansion=config.planner.diff_mode_token_expansion,
        ).to(device)
        planner_params = sum(p.numel() for p in planner.parameters())
        logger.info(
            "[EncoderDirect] DiffusionPlanner: %.2fM params, num_observed=%d, num_poses=%d, "
            "tokens_per_frame=%d, status_dim=%d, command_dim=%d",
            planner_params / 1e6,
            num_observed,
            num_poses,
            tokens_per_frame,
            planner_status_dim,
            planner_command_dim,
        )
        return planner

    from app.vjepa_cowa_world_model.models import MultiModalTemporalPlanner

    planner = MultiModalTemporalPlanner(
        encoder_dim=encoder_dim,
        tf_d_model=config.planner.tf_d_model,
        tf_d_ffn=config.planner.tf_d_ffn,
        tf_num_layers=config.planner.tf_num_layers,
        tf_num_head=config.planner.tf_num_head,
        tf_dropout=config.planner.tf_dropout,
        tokens_per_frame=tokens_per_frame,
        num_poses=num_poses,
        num_time_steps=encoder_direct_num_time_steps,
        status_dim=planner_status_dim,
        use_spatial_tokens=config.planner.use_spatial_tokens,
        num_modes=config.planner.num_modes,
        use_temporal=True,
        use_time_aligned_bias=config.planner.temporal_alignment,
        use_z_context=False,
        use_status_for_planner=config.planner.use_status_for_planner,
        use_observed_tokens=False,
        use_action_history=use_action_history,
        action_history_dim=action_history_dim,
        num_observed_frames=num_observed,
        command_dim=planner_command_dim,
    ).to(device)
    planner_params = sum(p.numel() for p in planner.parameters())
    logger.info(
        "[EncoderDirect] MultiModalTemporalPlanner: %.2fM params, num_observed=%d, num_poses=%d, "
        "tokens_per_frame=%d, num_time_steps(temporal)=%d, use_spatial_tokens=%s, "
        "temporal_alignment=%s, status_dim=%d, command_dim=%d",
        planner_params / 1e6,
        num_observed,
        num_poses,
        tokens_per_frame,
        encoder_direct_num_time_steps,
        config.planner.use_spatial_tokens,
        config.planner.temporal_alignment,
        planner_status_dim,
        planner_command_dim,
    )
    return planner
