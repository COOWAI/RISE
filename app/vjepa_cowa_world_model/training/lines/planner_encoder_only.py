# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
Navsim Encoder-Direct Planner 训练脚本

基于 train_navsim_v2.py，跳过 predictor 的自回归推理，
直接将 encoder 的全部观测帧输出 (num_observed_frames 帧) 作为 planner 的输入。
Planner 走时序路径 (_build_memory_temporal)，充分利用多帧时序信息。

核心区别:
- Predictor 不参与训练循环 (JEPA loss = 0)
- Planner 的 num_time_steps = num_observed_frames (而非 num_poses)
- Planner 输入: z_context[:, :num_observed_frames * tokens_per_frame]
- use_z_context=False, use_temporal=True → 走时序路径

用法:
    python -m app.main --fname configs/train/navsim/vitl16/encoder_direct.yaml \\
        --train-script train_planner_encoder_only
"""

import copy
import gc  # noqa: F401
import math  # noqa: F401
import os
import random
import sys
import time  # noqa: F401

import numpy as np
import torch
import torch.nn as nn  # noqa: F401
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

_REPO_ROOT = os.environ.get(
    "VJEPA_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
)
_TORCHCV_PATH = os.path.join(_REPO_ROOT, "ddddetection_torchcv")
if _TORCHCV_PATH not in sys.path:
    sys.path.insert(0, _TORCHCV_PATH)

from app.vjepa_cowa_world_model.losses import l1_length_normalized_loss  # noqa: F401, E402
from app.vjepa_cowa_world_model.losses import awta_temperature_schedule, convert_trajectory_3d_to_nd  # noqa: E402
from app.vjepa_cowa_world_model.models import MultiModalTemporalPlanner  # noqa: E402
from app.vjepa_cowa_world_model.models.planner_contracts import validate_planner_output  # noqa: E402
from app.vjepa_cowa_world_model.training.configs.planner import PLANNER_TYPES  # noqa: E402

try:
    from app.vjepa_cowa_world_model.models.vjepa_img_encoder import VJEPAImgEncoderAdapter  # noqa: E402,F401
except ImportError:  # pragma: no cover - supports lightweight import stubs in source tests
    VJEPAImgEncoderAdapter = None
from app.vjepa_cowa_world_model.training import (  # noqa: E402
    OPEN_LOOP_SELECTION_RULE,
    TrainingTimer,
    add_encoder_param_groups,
    add_planner_param_groups,
    build_validation_record,
    calculate_iterations_per_epoch,
    compile_models,
    create_ema_update_fn,
    create_loss_meters,
    create_momentum_scheduler,
    create_optimizer_and_scheduler,
    create_train_dataloader,
    create_transforms,
    create_val_dataloader,
    create_validation_transforms,
    freeze_parameters,
    get_encoder_embed_dim,
    get_next_batch,
    init_encoder,
    init_predictor,
    init_segmentation_modules,
    init_training_loop,
    is_epoch_validation_due,
    load_checkpoint,
    load_clips,
    load_pretrained_checkpoint,
    log_epoch_summary,
    log_trainable_parameters,
    log_training_metrics,
    log_training_summary,
    log_validation_metrics,
    maybe_run_gc,
    parse_training_config,
    resume_from_checkpoint,
    save_training_checkpoint,
    wait_for_checkpoint_save,
    wrap_ddp_models,
)
from app.vjepa_cowa_world_model.training.pipeline import setup_run  # noqa: E402
from app.vjepa_cowa_world_model.training.runtimes.encoder_token_runtime import (  # noqa: E402
    forward_encoder_direct_tokens,
)
from app.vjepa_cowa_world_model.training.runtimes.forward_runtime import ForwardRuntime  # noqa: E402
from app.vjepa_cowa_world_model.training.runtimes.loop_runner import (  # noqa: E402
    TrainingLoopRunner,
    resolve_periodic_checkpoint_epoch,
)
from app.vjepa_cowa_world_model.training.services import BestOpenLoopTracker  # noqa: E402
from app.vjepa_cowa_world_model.training.validation_capabilities import (  # noqa: E402
    validate_validation_suite_execution_contract,
)
from app.vjepa_cowa_world_model.training.validation_distributed import (  # noqa: E402
    raise_if_validation_failed,
    reduce_open_loop_validation_totals,
    wrap_validation_batch_error,
)
from app.vjepa_cowa_world_model.training.validation_rng import (  # noqa: E402
    resolve_stable_sample_ids,
    validation_randn,
)
from app.vjepa_cowa_world_model.utils import (  # noqa: E402
    prepare_seg_features,
    resolve_effective_planner_status_dim,
    resolve_planner_use_drive_command,
    save_training_visualization,
    select_best_trajectory,
    visualize_multimodal_trajectory,
    visualize_trajectory,
)
from app.vjepa_cowa_world_model.utils.eval_determinism import extract_batch_metadata  # noqa: E402
from app.vjepa_cowa_world_model.utils.metrics import (  # noqa: E402
    WORLD4DRIVE_REPORTED_SECONDS,
    compute_collision_rate,
    compute_world4drive_l2_metrics,
    populate_point_l2_horizons,
    populate_world4drive_collision_horizons,
    populate_world4drive_l2_horizons,
)
from app.vjepa_cowa_world_model.utils.planner_training import (  # noqa: E402
    _resolve_action_history_dt as _shared_resolve_action_history_dt,
)
from app.vjepa_cowa_world_model.utils.planner_training import (  # noqa: E402
    build_horizon_regression_timestep_weights,
    compute_planner_wta_loss,
    horizon_seconds_to_step_index,
    resolve_validation_timestep_sec,
)
from src.utils.logging import AverageMeter, get_logger, gpu_timer  # noqa: F401, E402

log_freq = 10
CHECKPOINT_FREQ = 1  # navsim-specific: save every epoch
GARBAGE_COLLECT_ITR_FREQ = 50

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def _resolve_action_history_dt(config) -> float:
    """Keep the maintained-line compatibility helper backed by the shared resolver."""
    return _shared_resolve_action_history_dt(config)


def _is_vjepa_img_encoder(config) -> bool:
    """Whether encoder-direct training uses the V-JEPA image encoder adapter."""
    return getattr(getattr(config, "model", None), "backbone", "") == "vjepa_img_encoder"


def _get_encoder_core(encoder):
    """Return the wrapped encoder module only for reading static attributes."""
    if encoder is None:
        return None
    return encoder.module if hasattr(encoder, "module") else encoder


def _should_load_generic_pretrained_checkpoint(load_encoder, load_predictor, load_seg, load_planner) -> bool:
    """Return whether any generic pretrained checkpoint component is requested."""
    return bool(load_encoder or load_predictor or load_seg or load_planner)


def _configure_vjepa_adapter_trainability(encoder, config) -> None:
    """Freeze V-JEPA backbone. The adapter has no projector — the planner owns
    the encoder_dim → hidden_dim projection, so backbone is the only encoder module."""
    if not _is_vjepa_img_encoder(config):
        return
    if getattr(config.train, "encoder_train", False):
        return

    core = _get_encoder_core(encoder)
    if core is None:
        raise AttributeError("V-JEPA encoder must be an adapter with a backbone module")

    backbone = getattr(core, "backbone", None)
    if backbone is None:
        raise AttributeError("V-JEPA adapter must expose a backbone module")

    for parameter in backbone.parameters():
        parameter.requires_grad = False
    backbone.eval()


def _optimizer_contains_parameter(optimizer, parameter) -> bool:
    """Return whether an optimizer already owns a parameter object."""
    return any(parameter is existing for group in optimizer.param_groups for existing in group.get("params", []))


def _add_vjepa_projector_param_groups(optimizer, encoder, config) -> None:
    """No-op: V-JEPA adapter no longer has an internal projector. The
    encoder_dim → hidden_dim projection is owned by the planner (e.g.
    ``DiffusionPlanner.context_proj``) and is registered through the planner's
    own param group, so the optimizer does not need an extra encoder group."""
    del optimizer, encoder, config
    return


def _resolve_encoder_direct_tokens_per_frame(config, encoder=None) -> int:
    """Resolve the effective planner token count per temporal step."""
    if _is_vjepa_img_encoder(config):
        core = _get_encoder_core(encoder)
        tokens_per_frame = getattr(core, "tokens_per_frame", None) if core is not None else None
        if tokens_per_frame is not None:
            return int(tokens_per_frame)

        height, width = getattr(config.model, "vjepa_resolution")
        patch_size = int(getattr(config.data, "patch_size"))
        return int((int(height) // patch_size) * (int(width) // patch_size))

    return int(config.data.tokens_per_frame)


def _resolve_encoder_direct_num_time_steps(config, encoder=None) -> int:
    """Resolve the effective planner temporal length for encoder-direct tokens."""
    if _is_vjepa_img_encoder(config):
        core = _get_encoder_core(encoder)
        num_time_steps = getattr(core, "num_time_steps", None) if core is not None else None
        if num_time_steps is not None:
            return int(num_time_steps)
        return 1

    return int(config.train.num_observed_frames)


def _forward_encoder_direct_tokens(encoder, context_clips, config):
    """Forward encoder-direct context clips into planner token features.

    Single source of truth: delegates to the shared runtime implementation
    (``encoder_direct_runtime.forward_encoder_direct_tokens``). Training (this script),
    open-loop NuScenes validation and PDMS closed-loop eval (``navsim_agent`` calls the
    same runtime function) therefore encode observed clips identically — editing the
    encoding logic in one place can no longer silently desync train vs infer. Guarded by
    ``test_forward_encoder_direct_tokens_delegates_to_shared_runtime``.
    """
    return forward_encoder_direct_tokens(encoder, context_clips, config)


def _init_encoder_direct_encoder(config, device):
    """Initialize the encoder pair for encoder-direct training."""
    if _is_vjepa_img_encoder(config):
        if VJEPAImgEncoderAdapter is None:
            raise ImportError("VJEPAImgEncoderAdapter is required for backbone='vjepa_img_encoder'")

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

    return init_encoder(config, device)


# ──────────────────────────────────────────────────────────────────────
# Planner 初始化: encoder-direct 专用
# ──────────────────────────────────────────────────────────────────────
def _init_encoder_direct_planner(config, encoder_dim, device, encoder=None):
    """
    初始化 Encoder-Direct 模式的 Planner。

    关键区别与 init_planner():
    - use_z_context = False:  直接通过 z_ar 位置参数传入 encoder token
    - use_temporal = True:    强制时序模式，走 _build_memory_temporal
    - num_time_steps = num_observed_frames: temporal_embedding 按观测帧数初始化
    - use_observed_tokens = False: 观测帧已经是主输入，不需要额外拼接

    Parameters
    ----------
    config       : TrainingConfig
    encoder_dim  : int, encoder 的嵌入维度
    device       : torch.device
    encoder      : Optional[nn.Module], 用于读取 V-JEPA adapter 的有效 token/time-step 属性

    Returns
    -------
    Optional[nn.Module]: planner 模型，如果 use_planner=False 则返回 None
    """
    if not config.planner.use_planner:
        logger.info("use_planner=False, planner is disabled")
        return None

    num_observed = config.train.num_observed_frames
    total_frames = config.data.num_target_frames
    num_poses = total_frames - num_observed
    tokens_per_frame = _resolve_encoder_direct_tokens_per_frame(config, encoder)
    encoder_direct_num_time_steps = _resolve_encoder_direct_num_time_steps(config, encoder)
    use_action_history = bool(getattr(config.planner, "use_action_history_for_planner", False))
    action_history_dim = int(getattr(config.planner, "action_history_dim", 3))

    if not 1 <= num_observed < total_frames:
        raise ValueError(
            f"train.num_observed_frames must satisfy 1 <= num_observed_frames < num_target_frames, "
            f"got {num_observed} and {total_frames}"
        )

    # status_dim: 与 runtime status_feature 构造保持一致
    _use_cmd = resolve_planner_use_drive_command(config)
    planner_status_dim = resolve_effective_planner_status_dim(config)

    # command_dim: split_status_embedding 时拆分导航指令和运动学特征
    if _use_cmd and config.planner.split_status_embedding and config.train.predictor_inference_consistent:
        planner_command_dim = 4  # drive_command one-hot 维度
    else:
        planner_command_dim = 0  # 旧行为，不拆分（或 use_drive_command=False）

    # 根据 planner_type 选择实现
    planner_type = config.planner.planner_type
    if planner_type not in PLANNER_TYPES:
        raise ValueError(f"Unknown planner.planner_type={planner_type!r}; expected one of {PLANNER_TYPES}.")

    # ── Status dimension summary（encoder-direct 所有 planner 类型共享）──
    _status_layouts = {
        3: "[velocity, acceleration, yaw_rate]",
        4: "[vx, vy, ax, ay]",
        7: "[cmd(4), velocity, acceleration, yaw_rate]",
        8: "[cmd(4), vx, vy, ax, ay]" if _use_cmd else "[vx, vy, ax, ay, x_local, y_local, sin_yaw, cos_yaw]",
        12: "[cmd(4), vx, vy, ax, ay, x_local, y_local, sin_yaw, cos_yaw]",
    }
    _layout = _status_layouts.get(planner_status_dim, f"custom({planner_status_dim}d)")
    logger.info(
        f"[Status Summary] planner_status_dim={planner_status_dim}, "
        f"command_dim={planner_command_dim}, "
        f"use_drive_command={_use_cmd} | layout: {_layout}"
    )

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

        # Encoder-only supports ONLY the base VP DiffusionPlanner. The main factory (model_factories/planner.py)
        # branches generation_framework / prefix-conditioning / seeded-init into FlowMatching / PrefixConditioned
        # / Seeded variants; this path instantiates base DiffusionPlanner unconditionally, so fail loud rather
        # than silently train a base VP planner when a different variant was configured.
        _gen_fw = str(config.planner.diff_generation_framework).lower()
        _seeded_init = not (
            str(config.planner.diff_init_traj_strategy) == "gaussian"
            and float(config.planner.diff_init_traj_noise_scale) == 1.0
            and float(config.planner.diff_init_traj_yaw_span_deg) == 30.0
            and float(config.planner.diff_init_traj_speed_scale_span) == 0.2
        )
        if (
            _gen_fw not in {"vp_diffusion", "diffusion", "vp", "vp_sde"}
            or config.planner.diff_train_prefix_conditioning
            or _seeded_init
        ):
            raise ValueError(
                "planner_encoder_only supports only the base VP DiffusionPlanner, but the config requests a "
                f"different diffusion variant (generation_framework={_gen_fw!r}, "
                f"prefix_conditioning={config.planner.diff_train_prefix_conditioning}, seeded_init={_seeded_init}). "
                "Use train_planner_world_model for flow-matching / prefix-conditioned / seeded-init variants."
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
            f"[EncoderDirect] DiffusionPlanner: {planner_params / 1e6:.2f}M params, "
            f"num_observed={num_observed}, num_poses={num_poses}, "
            f"tokens_per_frame={tokens_per_frame}, "
            f"status_dim={planner_status_dim}, command_dim={planner_command_dim}"
        )
        if config.planner.horizon_reg_loss_seconds and diff_reg_timestep_weights is not None:
            logger.info(
                "[HorizonReg] DiffusionPlanner internal XY reg weights enabled: timestep_sec=%.4f num_poses=%d",
                planner_timestep_sec,
                num_poses,
            )
        return planner

    # 默认: Transformer planner
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
        use_temporal=True,  # 强制时序模式
        use_time_aligned_bias=config.planner.temporal_alignment,
        use_z_context=False,  # 直接通过 z_ar 传入 encoder token
        use_status_for_planner=config.planner.use_status_for_planner,
        use_observed_tokens=False,  # 观测帧已是主输入
        use_action_history=use_action_history,
        action_history_dim=action_history_dim,
        num_observed_frames=num_observed,
        command_dim=planner_command_dim,
    ).to(device)

    planner_params = sum(p.numel() for p in planner.parameters())

    logger.info(
        f"[EncoderDirect] MultiModalTemporalPlanner: {planner_params / 1e6:.2f}M params, "
        f"num_observed={num_observed}, num_poses={num_poses}, "
        f"tokens_per_frame={tokens_per_frame}, "
        f"num_time_steps(temporal)={encoder_direct_num_time_steps}, "
        f"use_spatial_tokens={config.planner.use_spatial_tokens}, "
        f"temporal_alignment={config.planner.temporal_alignment}, "
        f"status_dim={planner_status_dim}, command_dim={planner_command_dim}"
    )

    return planner


# ──────────────────────────────────────────────────────────────────────
# Encoder-Direct 验证函数
# ──────────────────────────────────────────────────────────────────────
def _validate_encoder_direct(
    encoder,
    planner,
    val_loader,
    val_sampler,
    config,
    epoch,
    rank,
    world_size,
    vis_output_dir=None,
    vis_every_n_batches: int = 50,
    vis_samples_per_batch: int = 20,
):
    """
    Encoder-Direct 模式的验证:
    直接使用 encoder 观测帧输出作为 planner 输入，跳过 predictor。

    Returns
    -------
    dict: 包含 ade, fde, minade_k, minfde_k, l2_avg, collision_rate 的指标字典
    """
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    which_dtype = config.meta.dtype
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    num_observed = config.train.num_observed_frames
    num_poses = config.data.num_target_frames - num_observed
    timestep_sec = resolve_validation_timestep_sec(
        fps=getattr(config.data, "fps", None),
        diff_dt=getattr(config.planner, "diff_dt", None),
        default=0.5,
    )
    encoder_unwrapped = encoder.module if hasattr(encoder, "module") else encoder
    planner_unwrapped = planner.module if hasattr(planner, "module") else planner
    encoder_was_training = encoder_unwrapped.training
    planner_was_training = planner_unwrapped.training

    def _restore_validation_training_states():
        encoder_unwrapped.train(encoder_was_training)
        planner_unwrapped.train(planner_was_training)
        _configure_vjepa_adapter_trainability(encoder_unwrapped, config)

    encoder_unwrapped.eval()
    planner_unwrapped.eval()

    total_ade = 0.0
    total_fde = 0.0
    total_minade_k = 0.0
    total_minfde_k = 0.0
    total_samples = 0
    failed_batches = 0

    # L2 per-timestep accumulator (weighted by batch size)
    l2_per_step_acc = None  # will be np.array of shape [T]
    l2_total_samples = 0

    # Collision accumulators (raw counts)
    box_collision_counts_acc = None  # np.array [T_future]
    point_collision_counts_acc = None  # np.array [T_future]
    gt_collision_counts_acc = None  # np.array [T_future]
    collision_total_samples = 0
    missing_bev_segmentation_warned = False
    local_error = None

    val_sampler.set_epoch(epoch)

    batch_idx = -1
    try:
        val_iterator = iter(val_loader)
    except Exception as error:
        local_error = RuntimeError("Encoder-direct validation dataloader initialization failed")
        local_error.__cause__ = error
        val_iterator = iter(())
    while local_error is None:
        try:
            sample = next(val_iterator)
        except StopIteration:
            break
        except Exception as error:
            local_error = RuntimeError("Encoder-direct validation dataloader iteration failed")
            local_error.__cause__ = error
            break
        batch_idx += 1
        try:
            metadata = extract_batch_metadata(sample)
            context_frames = sample[0].to(device, non_blocking=True)
            actions = sample[1].to(device, dtype=torch.float, non_blocking=True)
            states = sample[2].to(device, dtype=torch.float, non_blocking=True)
            driving_command = (
                sample[5].to(device, non_blocking=True) if len(sample) > 5 and sample[5] is not None else None
            )
            ego_dynamics = (
                sample[6].to(device, non_blocking=True) if len(sample) > 6 and sample[6] is not None else None
            )
            # BEV segmentation for collision (index 9)
            bev_segmentation = sample[9].numpy() if len(sample) > 9 and sample[9] is not None else None
            if bev_segmentation is None and not missing_bev_segmentation_warned:
                logger.warning(
                    "Encoder-direct validation batches do not include BEV segmentation; "
                    "collision metrics will be reported as inf."
                )
                missing_bev_segmentation_warned = True

            B = context_frames.shape[0]
            planner_noise = None
            if str(config.planner.planner_type).lower() == "diffusion":
                sample_ids = resolve_stable_sample_ids(metadata, batch_size=B)
                planner_modes = (
                    int(planner_unwrapped.num_modes)
                    if int(planner_unwrapped.num_modes) > 1
                    else int(planner_unwrapped.num_samples)
                )
                planner_noise = validation_randn(
                    (
                        B,
                        planner_modes,
                        int(planner_unwrapped.num_poses),
                        int(planner_unwrapped.traj_dim),
                    ),
                    base_seed=int(config.meta.seed),
                    sample_ids=sample_ids,
                    protocol="encoder-direct/planner",
                    horizon=None,
                    stream="trajectory_initial_noise",
                    device=device,
                    dtype=torch.float32,
                )

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                # Single-source forward contract (Phase 1): identical to the NavSim eval agent's
                # encoder-direct forward, via ForwardRuntime built from the same config.
                runtime = ForwardRuntime.encoder_direct_from_config(
                    config, encoder=encoder_unwrapped, planner=planner_unwrapped
                )
                z_encoder_obs = runtime.observed_tokens(runtime.encode_context(context_frames))

                status_feature = runtime.status_feature(
                    states, driving_command=driving_command, ego_dynamics=ego_dynamics
                )
                planner_action_history = None
                if runtime.spec.use_action_history:
                    action_history_frames = int(getattr(planner_unwrapped, "num_observed_frames", num_observed))
                    planner_action_history = runtime.action_history(actions, num_observed_frames=action_history_frames)

                # Planner forward
                planner_out = runtime.forward_planner(
                    z_encoder_obs,
                    status_feature,
                    planner_action_history,
                    inference_noise=planner_noise,
                )
                validate_planner_output(planner_out, mode="inference")
                pred_trajs = planner_out["trajectories"].float()  # [B, K, num_poses, 3]
                pred_conf = planner_out["confidences"].float()  # [B, K]

                # GT trajectory
                # Use float64 for position diff to guard against large UTM values.
                StateSE2_indices = [0, 1, 5]
                states_se2 = states[:, :, StateSE2_indices].double()
                origin_idx = num_observed - 1
                origin_x = states_se2[:, origin_idx, 0]
                origin_y = states_se2[:, origin_idx, 1]
                origin_yaw = states_se2[:, origin_idx, 2]

                dx = states_se2[:, num_observed:, 0] - origin_x[:, None]
                dy = states_se2[:, num_observed:, 1] - origin_y[:, None]
                dyaw = states_se2[:, num_observed:, 2] - origin_yaw[:, None]

                cos_h = torch.cos(-origin_yaw)
                sin_h = torch.sin(-origin_yaw)
                ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
                ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
                ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

                gt_trajectory = torch.stack([ego_x, ego_y, ego_yaw], dim=-1).float()
                gt_trajectory = gt_trajectory[:, :num_poses]

                # 选择最佳模态
                best_traj = select_best_trajectory(pred_trajs, pred_conf)  # [B, num_poses, 3]

                # 计算指标 (ADE, FDE)
                pred_xy = best_traj[:, :, :2]
                gt_xy = gt_trajectory[:, :, :2]
                displacement = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, num_poses]
                ade = displacement.mean(dim=-1)  # [B]
                fde = displacement[:, -1]  # [B]

                # minADE@K, minFDE@K
                all_pred_xy = pred_trajs[:, :, :, :2]  # [B, K, num_poses, 2]
                gt_xy_expand = gt_xy.unsqueeze(1)  # [B, 1, num_poses, 2]
                all_disp = torch.norm(all_pred_xy - gt_xy_expand, dim=-1)  # [B, K, num_poses]
                all_ade = all_disp.mean(dim=-1)  # [B, K]
                all_fde = all_disp[:, :, -1]  # [B, K]
                minade_k = all_ade.min(dim=-1).values  # [B]
                minfde_k = all_fde.min(dim=-1).values  # [B]

                total_ade += ade.sum().item()
                total_fde += fde.sum().item()
                total_minade_k += minade_k.sum().item()
                total_minfde_k += minfde_k.sum().item()
                total_samples += B

                if vis_output_dir and rank == 0 and batch_idx % vis_every_n_batches == 0:
                    visualize_multimodal_trajectory(
                        pred_trajs=pred_trajs,
                        pred_conf=pred_conf,
                        gt_traj=gt_trajectory,
                        output_dir=vis_output_dir,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        limit=vis_samples_per_batch,
                    )

                # L2 per-timestep accumulation
                l2_result = compute_world4drive_l2_metrics(best_traj, gt_trajectory, timestep_sec=timestep_sec)
                l2_steps = np.array(l2_result["l2_per_step"])  # [T]
                if len(l2_steps) not in (0, num_poses):
                    raise ValueError(f"Encoder-direct L2 horizon must be {num_poses}, got {len(l2_steps)}")
                if len(l2_steps):
                    l2_per_step_acc = l2_per_step_acc + l2_steps * B if l2_per_step_acc is not None else l2_steps * B
                    l2_total_samples += B

                # Collision rate accumulation
                if bev_segmentation is not None:
                    bev_seg_future = bev_segmentation[:, num_observed:, :, :]
                    bev_seg_future = bev_seg_future[:, :num_poses]
                    T_future_bev = bev_seg_future.shape[1]
                    pred_np = best_traj[:, :T_future_bev].cpu().numpy()
                    gt_np = gt_trajectory[:, :T_future_bev].cpu().numpy()
                    ego_poses_np = states.cpu().numpy()  # [B, T_total, 7]
                    coll_result = compute_collision_rate(
                        pred_traj=pred_np,
                        gt_traj=gt_np,
                        segmentation=bev_seg_future,
                        ego_poses=ego_poses_np,
                        future_start_idx=num_observed,
                        timestep_sec=timestep_sec,
                        reference_frame_idx=num_observed - 1,
                    )
                    T_future = pred_np.shape[1]
                    bc = np.array(coll_result["collision_counts"][:T_future], dtype=np.float64)
                    pc = np.array(coll_result["point_collision_counts"][:T_future], dtype=np.float64)
                    gt_collisions = np.array(coll_result["gt_collision_counts"][:T_future], dtype=np.float64)
                    collision_lengths = (len(bc), len(pc), len(gt_collisions))
                    if collision_lengths != (num_poses, num_poses, num_poses):
                        raise ValueError(
                            f"Encoder-direct collision horizons must each be {num_poses}, got {collision_lengths}"
                        )
                    box_collision_counts_acc = (
                        box_collision_counts_acc + bc if box_collision_counts_acc is not None else bc
                    )
                    point_collision_counts_acc = (
                        point_collision_counts_acc + pc if point_collision_counts_acc is not None else pc
                    )
                    gt_collision_counts_acc = (
                        gt_collision_counts_acc + gt_collisions
                        if gt_collision_counts_acc is not None
                        else gt_collisions
                    )
                    collision_total_samples += B

        except Exception as e:
            local_error = wrap_validation_batch_error("Encoder-direct validation", batch_idx, e)
            break

    _restore_validation_training_states()
    raise_if_validation_failed(
        local_error,
        validation_name="Encoder-direct validation",
        device=device,
        world_size=world_size,
    )

    collision_sums = None
    if box_collision_counts_acc is not None:
        collision_sums = (
            box_collision_counts_acc,
            point_collision_counts_acc,
            gt_collision_counts_acc,
        )
    reduced = reduce_open_loop_validation_totals(
        metric_sums=[total_ade, total_fde, total_minade_k, total_minfde_k],
        total_samples=total_samples,
        l2_sums=l2_per_step_acc,
        l2_samples=l2_total_samples,
        collision_sums=collision_sums,
        collision_samples=collision_total_samples,
        num_steps=num_poses,
        device=device,
        world_size=world_size,
    )
    total_ade, total_fde, total_minade_k, total_minfde_k = reduced["metric_sums"]
    total_samples = int(reduced["total_samples"])
    l2_total_samples = int(reduced["l2_samples"])
    l2_per_step_acc = np.asarray(reduced["l2_sums"], dtype=np.float64)
    box_collision_counts_acc = np.asarray(reduced["box_collision_sums"], dtype=np.float64)
    point_collision_counts_acc = np.asarray(reduced["point_collision_sums"], dtype=np.float64)
    gt_collision_counts_acc = np.asarray(reduced["gt_collision_sums"], dtype=np.float64)
    collision_total_samples = int(reduced["collision_samples"])

    if total_samples == 0:
        raise RuntimeError("Encoder-direct validation produced zero global successful samples")

    metrics = {
        "ade": total_ade / total_samples,
        "fde": total_fde / total_samples,
        "minade_k": total_minade_k / total_samples,
        "minfde_k": total_minfde_k / total_samples,
    }

    # Compute l2_avg from accumulated per-step values
    if l2_total_samples > 0:
        l2_avg_steps = l2_per_step_acc / l2_total_samples
        metrics["l2_per_step"] = l2_avg_steps.tolist()
        populate_world4drive_l2_horizons(metrics, l2_avg_steps, timestep_sec)
        populate_point_l2_horizons(metrics, l2_avg_steps, timestep_sec)
    else:
        metrics["l2_avg"] = float("inf")

    # Compute collision_rate from accumulated counts
    if collision_total_samples > 0:
        box_collision_per_step = box_collision_counts_acc / max(float(collision_total_samples), 1.0)
        metrics["collision_per_step"] = box_collision_per_step.tolist()
        metrics["collision_counts"] = box_collision_counts_acc.astype(np.int64).tolist()
        metrics["gt_collision_counts"] = gt_collision_counts_acc.astype(np.int64).tolist()
        populate_world4drive_collision_horizons(
            metrics,
            box_collision_counts_acc,
            total_samples=collision_total_samples,
            timestep_sec=timestep_sec,
            metric_prefix="collision",
            avg_key="collision_rate",
        )
    else:
        metrics["collision_rate"] = float("inf")

    if rank == 0:
        logger.info("=" * 50)
        logger.info(f"[EncoderDirect] Validation Results - Epoch {epoch}:")
        logger.info(f"  ADE (Average Displacement Error): {metrics['ade']:.4f} m")
        logger.info(f"  FDE (Final Displacement Error):    {metrics['fde']:.4f} m")
        logger.info(f"  minADE@K:                         {metrics['minade_k']:.4f} m")
        logger.info(f"  minFDE@K:                         {metrics['minfde_k']:.4f} m")
        l2_line = f"  L2_avg:                           {metrics.get('l2_avg', float('inf')):.4f} m"
        for sec in WORLD4DRIVE_REPORTED_SECONDS:
            key = f"l2_at_{sec}s"
            if key in metrics:
                l2_line += f" | L2@{sec}s={metrics[key]:.4f}"
        logger.info(l2_line)
        if "l2_point_avg" in metrics:
            point_l2_line = f"  PointL2_avg:                      {metrics['l2_point_avg']:.4f} m"
            for sec in WORLD4DRIVE_REPORTED_SECONDS:
                key = f"l2_point_at_{sec}s"
                if key in metrics:
                    point_l2_line += f" | PointL2@{sec}s={metrics[key]:.4f}"
            logger.info(point_l2_line)
        col_line = f"  Collision Rate:                   {metrics.get('collision_rate', float('inf')):.4f}"
        for sec in WORLD4DRIVE_REPORTED_SECONDS:
            key = f"collision_at_{sec}s"
            if key in metrics:
                col_line += f" | Col@{sec}s={metrics[key]:.4f}"
        logger.info(col_line)
        logger.info(f"  Total samples: {total_samples}, failed batches: {failed_batches}")
        logger.info("=" * 50)

    return metrics


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main(args, resume_preempt=False):
    """Navsim Encoder-Direct Planner 主训练函数"""
    config = parse_training_config(args)
    logger.info(f"[EncoderDirect] {config.meta.dtype=}")

    validate_validation_suite_execution_contract(
        config,
        line_name="planner_encoder_only",
        declared_executors=set(),
        active_consumers={"planner"} if bool(config.planner.use_planner) else set(),
    )

    # -- Anneal (cooldown) validation
    is_anneal = config.optimization.is_anneal
    if is_anneal and not config.optimization.anneal_ckpt:
        raise ValueError("is_anneal=True requires optimization.anneal_ckpt to be set")
    if is_anneal and resume_preempt:
        config.optimization.resume_anneal = True
        logger.info("[Anneal] resume_preempt=True → switching to resume_anneal mode")

    ctx = setup_run(config)
    world_size, rank, device, ckpt_paths = ctx.world_size, ctx.rank, ctx.device, ctx.ckpt_paths
    csv_logger, tb_writer = ctx.csv_logger, ctx.tb_writer

    latest_path, best_path, resume_path = ckpt_paths["latest"], ckpt_paths["best"], ckpt_paths["resume"]

    best_tracker = BestOpenLoopTracker(best_path)
    validation_history = []

    encoder, target_encoder = _init_encoder_direct_encoder(config, device)
    encoder_embed_dim = get_encoder_embed_dim(encoder)
    logger.info(f"encoder_embed_dim: {encoder_embed_dim}")

    # Predictor: 仍然加载（用于 checkpoint 兼容），但不参与训练循环
    predictor = init_predictor(config, device, encoder_embed_dim)
    seg_neck, seg_head = init_segmentation_modules(
        device,
        config.segmentation.use_segmentation,
        encoder_embed_dim=encoder_embed_dim,
        num_classes=config.segmentation.num_classes,
        loss_seg_weight=config.segmentation.loss_seg_weight,
        loss_dice_weight=config.segmentation.loss_dice_weight,
    )

    # Encoder-Direct Planner: num_time_steps = num_observed_frames
    num_observed = config.train.num_observed_frames
    total_frames = config.data.num_target_frames
    num_poses_init = total_frames - num_observed
    planner = _init_encoder_direct_planner(config, encoder_embed_dim, device, encoder=encoder)

    logger.info(
        f"[EncoderDirect] Mode: encoder output → planner (skip predictor), "
        f"num_observed_frames={num_observed}, num_target_frames={total_frames}, "
        f"num_poses={num_poses_init}"
    )

    compile_models(encoder, target_encoder, predictor, seg_head, config.model.compile_model)

    transform = create_transforms(config)
    validation_transform = create_validation_transforms(config)
    train_loader, train_sampler = create_train_dataloader(config, rank, world_size, transform)
    val_loader, val_sampler = create_val_dataloader(config, rank, world_size, validation_transform)
    ipe = calculate_iterations_per_epoch(config, train_loader)

    # Configure V-JEPA frozen-backbone/projector-trainable state before optimizer/DDP construction.
    _configure_vjepa_adapter_trainability(encoder, config)

    optimizer, scaler, scheduler, wd_scheduler = create_optimizer_and_scheduler(
        config, encoder, predictor, seg_neck, seg_head, planner, ipe
    )
    if config.planner.use_planner and planner is not None:
        add_planner_param_groups(optimizer, planner)

    models = wrap_ddp_models(
        encoder,
        target_encoder,
        predictor,
        seg_neck,
        seg_head,
        planner,
        encoder_train=config.train.encoder_train,
        use_planner=config.planner.use_planner,
        use_status_for_planner=config.planner.use_status_for_planner,
        use_temporal=True,  # encoder-direct 强制时序模式
        use_z_context=False,  # 直接通过 z_ar 传入
    )
    encoder, target_encoder, predictor, seg_neck, seg_head, planner = (
        models["encoder"],
        models["target_encoder"],
        models["predictor"],
        models["seg_neck"],
        models["seg_head"],
        models["planner"],
    )

    freeze_parameters(
        encoder,
        target_encoder,
        predictor,
        seg_neck,
        seg_head,
        planner,
        encoder_train=config.train.encoder_train,
        predictor_train=False,  # Predictor 始终冻结 (不参与训练)
        seg_head_train=config.train.seg_head,
    )
    _configure_vjepa_adapter_trainability(encoder, config)
    _add_vjepa_projector_param_groups(optimizer, encoder, config)
    if config.train.encoder_train:
        add_encoder_param_groups(optimizer, encoder, config.optimization.enc_lr_scale)

    predictor_state_mode = "[EncoderDirect] predictor skipped (encoder output → planner directly)"
    log_trainable_parameters(
        encoder, predictor, seg_neck, seg_head, planner, optimizer, config.planner.use_planner, predictor_state_mode
    )

    momentum_scheduler = create_momentum_scheduler(
        config.ema.ema_start, config.ema.ema_end, ipe, config.optimization.epochs
    )
    update_ema = create_ema_update_fn(encoder, target_encoder)
    generic_load_encoder = config.meta.load_encoder and not _is_vjepa_img_encoder(config)

    # -- Checkpoint loading (anneal-aware)
    if is_anneal and not config.optimization.resume_anneal:
        logger.info(f"[Anneal] Loading anneal checkpoint from: {config.optimization.anneal_ckpt}")
        load_pretrained_checkpoint(
            config.optimization.anneal_ckpt,
            encoder,
            target_encoder,
            predictor,
            seg_neck,
            seg_head,
            planner,
            load_encoder=generic_load_encoder,
            load_predictor=True,
            load_seg=config.meta.load_seg,
            load_planner=config.meta.load_planner,
            context_encoder_key=config.meta.context_encoder_key,
            target_encoder_key=config.meta.target_encoder_key,
            rank=rank,
            world_size=world_size,
            predictor_checkpoint=config.meta.predictor_checkpoint,
        )
        start_epoch = 0
        logger.info("[Anneal] Fresh anneal start: epoch=0, scheduler not restored")
    else:
        should_load_pretrained = _should_load_generic_pretrained_checkpoint(
            generic_load_encoder,
            config.meta.load_predictor,
            config.meta.load_seg,
            config.meta.load_planner,
        )
        if should_load_pretrained:
            load_pretrained_checkpoint(
                config.meta.pretrain_checkpoint_full,
                encoder,
                target_encoder,
                predictor,
                seg_neck,
                seg_head,
                planner,
                load_encoder=generic_load_encoder,
                load_predictor=config.meta.load_predictor,
                load_seg=config.meta.load_seg,
                load_planner=config.meta.load_planner,
                context_encoder_key=config.meta.context_encoder_key,
                target_encoder_key=config.meta.target_encoder_key,
                rank=rank,
                world_size=world_size,
                predictor_checkpoint=config.meta.predictor_checkpoint,
            )
        else:
            logger.info(
                "[EncoderDirect] Skipping generic pretrained checkpoint load because "
                "load_encoder/load_predictor/load_seg/load_planner are all false"
            )

        start_epoch = resume_from_checkpoint(
            resume_path=resume_path,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            seg_head=seg_head,
            seg_neck=seg_neck,
            planner=planner,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            use_planner=config.planner.use_planner,
            load_planner=(not config.meta.resume_model_only) or config.meta.load_planner,
            rank=rank,
            world_size=world_size,
            use_broadcast=config.meta.resume_broadcast,
            model_only=config.meta.resume_model_only,
        )

        # Restore best-metric tracker from best_path so a resumed run does not
        # overwrite a real best checkpoint on its first validation. best_path embeds
        # best_open_loop_record in extra_state (top level); resumed latest.pt does not.
        if not config.meta.resume_model_only:
            best_tracker.restore(load_checkpoint)

    # -- Fast-forward momentum scheduler to match start_epoch
    for _ in range(start_epoch * ipe):
        next(momentum_scheduler)

    encoder_checkpoint_train = config.train.encoder_train or _is_vjepa_img_encoder(config)

    def save_checkpoint_fn(epoch, path, extra_state=None):
        save_training_checkpoint(
            path=path,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            seg_neck=seg_neck,
            seg_head=seg_head,
            planner=planner,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            epoch=epoch,
            loss=loss_meter.avg,
            batch_size=config.data.batch_size,
            world_size=world_size,
            lr=config.optimization.lr,
            rank=rank,
            use_planner=config.planner.use_planner,
            encoder_train=encoder_checkpoint_train,
            encoder_ema=config.train.encoder_ema,
            predictor_train=False,
            seg_head_train=config.train.seg_head,
            extra_state=extra_state,
        )

    def log_optimizer_group_lrs(epoch, itr):
        if rank != 0:
            return
        group_logs = []
        for group_idx, group in enumerate(optimizer.param_groups):
            params = group.get("params", [])
            num_tensors = len(params)
            num_params = sum(p.numel() for p in params if p is not None)
            group_logs.append(
                f"g{group_idx}: lr={group.get('lr', 0.0):.8e}, "
                f"lr_scale={group.get('lr_scale', 1.0):.4g}, "
                f"wd={group.get('weight_decay', 0.0):.8e}, "
                f"tensors={num_tensors}, params={num_params / 1e6:.2f}M"
            )
        logger.info(f"Epoch {epoch + 1} Iter {itr + 1} optimizer param group LRs:\n" + "\n".join(group_logs))

    logger.info("初始化数据加载器...")
    loader, _ = init_training_loop(
        train_loader, train_sampler, start_epoch, config.meta.skip_batches, config.meta.sync_gc
    )

    tokens_per_frame = _resolve_encoder_direct_tokens_per_frame(config, encoder)
    encoder_direct_num_time_steps = _resolve_encoder_direct_num_time_steps(config, encoder)
    num_poses = config.data.num_target_frames - config.train.num_observed_frames
    planner_timestep_sec = resolve_validation_timestep_sec(
        fps=getattr(config.data, "fps", None),
        diff_dt=getattr(config.planner, "diff_dt", None),
        default=0.5,
    )
    horizon_reg_timestep_weights = None
    if config.planner.planner_type == "diffusion":
        if rank == 0 and config.planner.horizon_reg_loss_seconds:
            logger.info("[HorizonReg] handled by DiffusionPlanner internal XY regression loss")
    else:
        horizon_reg_timestep_weights = build_horizon_regression_timestep_weights(
            num_poses=num_poses,
            timestep_sec=planner_timestep_sec,
            horizon_seconds=config.planner.horizon_reg_loss_seconds,
            horizon_weights=config.planner.horizon_reg_loss_weights,
            normalize=config.planner.horizon_reg_loss_normalize,
            device=device,
            dtype=torch.float32,
        )
        if rank == 0 and config.planner.horizon_reg_loss_seconds and horizon_reg_timestep_weights is None:
            logger.warning(
                "[HorizonReg] configured horizons are outside num_poses=%d (timestep_sec=%.4f); disabled",
                num_poses,
                planner_timestep_sec,
            )
        elif rank == 0 and config.planner.horizon_reg_loss_seconds:
            weights_cpu = horizon_reg_timestep_weights.detach().cpu()
            parts = []
            for seconds, raw_weight in zip(
                config.planner.horizon_reg_loss_seconds,
                config.planner.horizon_reg_loss_weights,
            ):
                step_idx = horizon_seconds_to_step_index(float(seconds), planner_timestep_sec)
                if 0 <= step_idx < num_poses:
                    parts.append(
                        f"{float(seconds):g}s->idx{step_idx}:raw={float(raw_weight):g},"
                        f"norm={weights_cpu[step_idx].item():.4g}"
                    )
                else:
                    parts.append(f"{float(seconds):g}s->idx{step_idx}:out_of_range")
            logger.info(
                "[HorizonReg] timestep_sec=%.4f num_poses=%d weights=[%s]",
                planner_timestep_sec,
                num_poses,
                "; ".join(parts),
            )

    train_start_time = time.time()

    # Kept in main() scope so save_checkpoint_fn (a closure over loss_meter.avg) still sees the latest
    # epoch's meter after run_epoch returns — preserving the leak the old `for epoch` loop provided.
    loss_meter = None

    def run_epoch(epoch):
        nonlocal loss_meter
        logger.info(f"[EncoderDirect] 开始训练 Epoch {epoch + 1}")
        train_sampler.set_epoch(epoch)
        loader = iter(train_loader)

        # 防御性保障：确保 planner 在训练模式（validation 结束后可能处于 eval 模式）
        if config.planner.use_planner and planner is not None:
            planner.train()

        # Diffusion planner (hybrid aWTA loss) 每 epoch 刷新温度
        if getattr(config.planner, "use_planner", False) and config.planner.planner_type == "diffusion":
            cur_awta_temp = awta_temperature_schedule(
                init_temperature=config.planner.awta_init_temperature,
                epoch=epoch,
                exp_base=config.planner.awta_exp_base,
                min_temperature=config.planner.awta_min_temperature,
            )
            _planner_core = planner.module if hasattr(planner, "module") else planner
            _planner_core.set_awta_temperature(cur_awta_temp)
            logger.info(f"[aWTA] epoch={epoch} T={cur_awta_temp:.4f}")

        loss_meters = create_loss_meters()
        loss_meter, jloss_meter, sloss_meter = loss_meters["loss"], loss_meters["jloss"], loss_meters["sloss"]
        cls_valid_ratio_meter = loss_meters["cls_valid_ratio"]
        seg_loss_meter, mask_loss_meter, dice_loss_meter = (
            loss_meters["seg_loss"],
            loss_meters["mask_loss"],
            loss_meters["dice_loss"],
        )
        traj_loss_meter, reg_loss_meter, conf_loss_meter, cover_loss_meter = (
            loss_meters["traj_loss"],
            loss_meters["reg_loss"],
            loss_meters["conf_loss"],
            loss_meters["cover_loss"],
        )
        vel_loss_meter, yaw_loss_meter = loss_meters["vel_loss"], loss_meters["yaw_loss"]
        iter_time_meter, gpu_time_meter, data_elapsed_time_meter = (
            loss_meters["iter_time"],
            loss_meters["gpu_time"],
            loss_meters["data_load_time"],
        )

        for itr in range(ipe):
            timer = TrainingTimer()
            timer.start_iteration()
            loader, sample, success = get_next_batch(loader, train_loader, train_sampler, epoch)
            if not success:
                continue
            context_clips, actions, states, extrinsics, seg_targets, driving_command, ego_dynamics = load_clips(
                sample, device, config.segmentation.use_segmentation, torch.float
            )
            data_elapsed_time_ms = timer.record_data_load()
            maybe_run_gc(itr, GARBAGE_COLLECT_ITR_FREQ, config.meta.sync_gc)

            should_visualize = (rank == 0) and (itr % 200 == 0)
            vis_output_dir = os.path.join(config.meta.folder, "train_vis_debug")

            def train_step():
                _new_lr, _new_wd = scheduler.step(), wd_scheduler.step()
                # Single-source forward contract (Phase 1): same spec-driven observed-tokens / status /
                # action-history as the eval agent + in-train validation. Encode stays in forward_context
                # below to preserve its grad-context handling (still consistent via the shared forward).
                ed_runtime = ForwardRuntime.encoder_direct_from_config(config, encoder=encoder, planner=planner)

                # ============ Encoder Forward ============
                # (与 train_navsim_v2 相同的 per-frame encoding)
                def forward_context(c):
                    needs_encoder_forward_grad = config.train.encoder_train
                    grad_ctx = torch.enable_grad() if needs_encoder_forward_grad else torch.no_grad()
                    with grad_ctx:
                        z = _forward_encoder_direct_tokens(encoder, c, config)
                    if config.loss.normalize_reps:
                        z = F.layer_norm(z, (z.size(-1),))
                    return z

                # Forward pass
                with torch.cuda.amp.autocast(dtype=config.dtype, enabled=config.mixed_precision):
                    z_context = forward_context(context_clips)

                    # ============ 跳过 Predictor，直接使用 encoder 观测帧输出 ============
                    num_obs = config.train.num_observed_frames
                    z_encoder_obs = ed_runtime.observed_tokens(z_context)

                    # Shape assertions
                    expected_encoder_direct_tokens = encoder_direct_num_time_steps * tokens_per_frame
                    if not (z_encoder_obs.shape[1] == expected_encoder_direct_tokens):
                        raise AssertionError(
                            f"z_encoder_obs shape mismatch: {z_encoder_obs.shape}, "
                            f"expected second dim to be {expected_encoder_direct_tokens}"
                        )

                    # JEPA loss = 0 (predictor 不参与)
                    jepa_loss = torch.tensor(0.0, device=device)
                    jloss = torch.tensor(0.0, device=device)
                    sloss = torch.tensor(0.0, device=device)

                    # ============ Planner ============
                    traj_loss = torch.tensor(0.0, device=device)
                    _reg_loss = torch.tensor(0.0, device=device)
                    _conf_loss = torch.tensor(0.0, device=device)
                    _cover_loss = torch.tensor(0.0, device=device)
                    _vel_loss = torch.tensor(0.0, device=device)
                    _yaw_loss = torch.tensor(0.0, device=device)
                    _cls_valid_ratio = torch.tensor(0.0, device=device)

                    if config.planner.use_planner and planner is not None:
                        # Status feature: 只使用观测帧信息 (inference-consistent 格式)
                        status_feature = ed_runtime.status_feature(
                            states, driving_command=driving_command, ego_dynamics=ego_dynamics
                        )
                        planner_action_history = None
                        if ed_runtime.spec.use_action_history:
                            _planner_for_history = planner.module if hasattr(planner, "module") else planner
                            action_history_frames = int(getattr(_planner_for_history, "num_observed_frames", num_obs))
                            planner_action_history = ed_runtime.action_history(
                                actions, num_observed_frames=action_history_frames
                            )

                        # GT trajectory: 从 num_observed_frames 开始 (第一个"未来"帧)
                        StateSE2_indices = [0, 1, 5]
                        states_se2 = states[:, :, StateSE2_indices].double()

                        future_start_idx = num_obs
                        # 使用最后观测帧作为原点，使 GT 轨迹从 (0,0,0) 附近开始，
                        # 降低 planner 需要隐式推断累积位移的负担。
                        origin_idx = future_start_idx - 1
                        origin_x = states_se2[:, origin_idx, 0]
                        origin_y = states_se2[:, origin_idx, 1]
                        origin_yaw = states_se2[:, origin_idx, 2]

                        dx = states_se2[:, future_start_idx:, 0] - origin_x[:, None]
                        dy = states_se2[:, future_start_idx:, 1] - origin_y[:, None]
                        dyaw = states_se2[:, future_start_idx:, 2] - origin_yaw[:, None]

                        cos_h = torch.cos(-origin_yaw)
                        sin_h = torch.sin(-origin_yaw)
                        ego_x = cos_h[:, None] * dx - sin_h[:, None] * dy
                        ego_y = sin_h[:, None] * dx + cos_h[:, None] * dy
                        ego_yaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))

                        gt_trajectory = torch.stack([ego_x, ego_y, ego_yaw], dim=-1).float()
                        gt_trajectory = gt_trajectory[:, :num_poses]

                        # Safety: detect anomalous GT trajectory jumps
                        # caused by residual frame-index gaps in the data.
                        _MAX_STEP_DISP = 20.0  # metres per time-step
                        with torch.no_grad():
                            _gt_xy = gt_trajectory[..., :2]
                            if _gt_xy.shape[1] > 1:
                                _step_disp = torch.norm(_gt_xy[:, 1:, :] - _gt_xy[:, :-1, :], dim=-1)
                                _max_disp_per_sample = _step_disp.max(dim=1).values
                                _anomaly_mask = _max_disp_per_sample > _MAX_STEP_DISP
                                if _anomaly_mask.any():
                                    _n_bad = int(_anomaly_mask.sum().item())
                                    logger.warning(
                                        "GT trajectory anomaly: %d/%d samples exceed %.1f m/step "
                                        "(max=%.1f m). Zeroing anomalous GT to prevent loss corruption.",
                                        _n_bad,
                                        _anomaly_mask.shape[0],
                                        _MAX_STEP_DISP,
                                        _max_disp_per_sample.max().item(),
                                    )
                                    gt_trajectory[_anomaly_mask] = 0.0

                        # Diffusion planner support
                        if config.planner.planner_type == "diffusion":
                            gt_traj_nd = convert_trajectory_3d_to_nd(
                                gt_trajectory,
                                dt=config.planner.diff_dt,
                                traj_dim=config.planner.diff_traj_dim,
                            )
                            # 构造 anchor_state: 观测帧最后一帧的 ego-relative 状态
                            if config.planner.diff_use_anchor_frame:
                                last_obs_idx = future_start_idx - 1
                                _dx = states_se2[:, last_obs_idx, 0] - origin_x
                                _dy = states_se2[:, last_obs_idx, 1] - origin_y
                                _dyaw = states_se2[:, last_obs_idx, 2] - origin_yaw
                                _ego_x = cos_h * _dx - sin_h * _dy
                                _ego_y = sin_h * _dx + cos_h * _dy
                                _ego_yaw = torch.atan2(torch.sin(_dyaw), torch.cos(_dyaw))
                                if config.planner.diff_traj_dim == 4:
                                    action_states = torch.stack(
                                        [
                                            _ego_x,
                                            _ego_y,
                                            torch.cos(_ego_yaw),
                                            torch.sin(_ego_yaw),
                                        ],
                                        dim=-1,
                                    ).float()  # [B, 4]
                                else:
                                    # fail-loud (point 32): 缺 ego_dynamics 时不得把 6D anchor 速度静默置零。
                                    if ego_dynamics is None:
                                        raise ValueError(
                                            "6D anchor needs real ego_dynamics for vx/vy; got None (禁止静默置零)"
                                        )
                                    _anchor_vx = ego_dynamics[:, last_obs_idx, 0]
                                    _anchor_vy = ego_dynamics[:, last_obs_idx, 1]
                                    action_states = torch.stack(
                                        [
                                            _ego_x,
                                            _ego_y,
                                            _anchor_vx,
                                            _anchor_vy,
                                            torch.cos(_ego_yaw),
                                            torch.sin(_ego_yaw),
                                        ],
                                        dim=-1,
                                    ).float()  # [B, 6]
                            else:
                                action_states = None

                            diff_result = planner(
                                z_encoder_obs,
                                status_feature,
                                action_history=planner_action_history,
                                gt_trajectory=gt_traj_nd,
                                anchor_state=action_states,
                            )
                            validate_planner_output(
                                diff_result,
                                mode="training",
                                required_training_keys=("reg_loss", "conf_loss", "cover_loss"),
                            )
                            traj_loss = diff_result["loss"]
                            _reg_loss = diff_result["reg_loss"]
                            _conf_loss = diff_result["conf_loss"]
                            _cover_loss = diff_result["cover_loss"]
                            _vel_loss = diff_result.get("vel_loss", torch.tensor(0.0, device=device))
                            _yaw_loss = diff_result.get("yaw_loss", torch.tensor(0.0, device=device))
                            _cls_valid_ratio = diff_result.get(
                                "cls_sample_valid_ratio",
                                torch.tensor(0.0, device=device),
                            )

                            if should_visualize:
                                # Bypass DDP wrapper: DDP forward fires _sync_buffers (BROADCAST),
                                # but other ranks are doing backward ALLREDUCE — collective ordering desync.
                                planner_core = planner.module if hasattr(planner, "module") else planner
                                planner_core.eval()
                                with torch.no_grad():
                                    infer_out = planner_core(
                                        z_encoder_obs,
                                        status_feature,
                                        action_history=planner_action_history,
                                    )
                                planner_core.train()
                                pred_trajs = infer_out["trajectories"]
                                pred_conf = infer_out["confidences"]
                                best_traj = select_best_trajectory(pred_trajs, pred_conf)
                                visualize_trajectory(
                                    pred_traj=best_traj,
                                    gt_traj=gt_trajectory,
                                    output_dir=vis_output_dir,
                                    epoch=epoch,
                                    itr=itr,
                                    limit=5,
                                )
                        else:
                            # Transformer planner: 直接传入 encoder 观测帧 token
                            planner_out = planner(
                                z_encoder_obs,  # 作为 z_ar 参数，实际是 encoder 观测帧 token
                                status_feature,
                                action_history=planner_action_history,
                            )
                            validate_planner_output(planner_out, mode="inference")
                            pred_trajs = planner_out["trajectories"]
                            pred_conf = planner_out["confidences"]

                            wta_result = compute_planner_wta_loss(
                                config,
                                pred_trajs=pred_trajs,
                                pred_conf=pred_conf,
                                gt_traj=gt_trajectory,
                                epoch=epoch,
                                timestep_weights=horizon_reg_timestep_weights,
                                global_batch=bool(config.planner.wta_global_batch_norm),
                            )

                            traj_loss = wta_result["loss"]
                            _reg_loss = wta_result["reg_loss"]
                            _conf_loss = wta_result["conf_loss"]
                            _cover_loss = wta_result["cover_loss"]

                            if should_visualize:
                                best_traj = select_best_trajectory(pred_trajs, pred_conf)
                                visualize_trajectory(
                                    pred_traj=best_traj,
                                    gt_traj=gt_trajectory,
                                    output_dir=vis_output_dir,
                                    epoch=epoch,
                                    itr=itr,
                                    limit=5,
                                )

                    # Total loss: JEPA(=0) + planner
                    loss = jepa_loss + config.planner.planner_loss_weight * traj_loss

                    # Segmentation loss
                    seg_loss_value = 0.0
                    mask_loss_value = 0.0
                    dice_loss_value = 0.0
                    valid_samples = 0
                    neck_out = None
                    vis_meta = None

                    if config.segmentation.use_segmentation and seg_head is not None:
                        neck_out, batched_targets, valid_samples, vis_meta = prepare_seg_features(
                            context_clips=context_clips,
                            seg_targets=seg_targets,
                            z_perceiver=z_context,
                            seg_neck=seg_neck,
                            tubelet_size=config.data.tubelet_size,
                            tokens_per_frame=tokens_per_frame,
                            device=device,
                            mixed_precision=config.mixed_precision,
                            dtype=config.dtype,
                            normalize_reps=config.loss.normalize_reps,
                        )

                    if config.segmentation.use_segmentation and seg_head is not None and valid_samples > 0:
                        loss_dict = seg_head.module.get_loss(
                            inputs=neck_out, targets=batched_targets, input_query=None
                        )
                        total_seg_loss = sum(v for k, v in loss_dict.items() if "loss" in k)
                        seg_loss = total_seg_loss / valid_samples
                        loss = loss + (config.segmentation.seg_loss_weight * seg_loss)

                        with torch.no_grad():
                            last_layer_idx = 6
                            seg_loss_value = (
                                loss_dict.get(f"loss_seg_{last_layer_idx}", torch.tensor(0.0)).detach().item()
                            )
                            dice_loss_value = (
                                loss_dict.get(f"loss_dice_{last_layer_idx}", torch.tensor(0.0)).detach().item()
                            )
                            mask_loss_value = seg_loss_value

                # Always call backward to keep DDP gradient sync across all ranks
                _nan_detected = torch.isnan(loss) or torch.isinf(loss)

                if config.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if _nan_detected:
                    logger.warning(
                        f"[epoch {epoch + 1}, iter {itr}] NaN/Inf loss detected "
                        f"(loss={loss.item():.4g}), skipping optimizer step"
                    )
                    optimizer.zero_grad()
                    if config.mixed_precision:
                        scaler.update()
                else:
                    if config.planner.use_planner and planner is not None:
                        torch.nn.utils.clip_grad_norm_(
                            planner.parameters(), max_norm=config.optimization.grad_clip_norm
                        )
                    if seg_head is not None:
                        torch.nn.utils.clip_grad_norm_(
                            seg_head.parameters(), max_norm=config.optimization.grad_clip_norm
                        )

                    if config.mixed_precision:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                # EMA 更新
                if config.train.encoder_ema:
                    m = next(momentum_scheduler)
                    update_ema(m)

                # 可视化 (segmentation)
                if should_visualize and valid_samples > 0:
                    with torch.no_grad():
                        real_head = seg_head.module if isinstance(seg_head, DistributedDataParallel) else seg_head
                        real_head.eval()
                        pred_result = real_head(inputs=[x.float() for x in neck_out], input_query=None)
                        save_training_visualization(pred_result, vis_meta, vis_output_dir, epoch, itr)
                        real_head.train()

                return (
                    float(loss),
                    float(jloss),
                    float(sloss),
                    float(seg_loss_value),
                    float(mask_loss_value),
                    float(traj_loss),
                    float(dice_loss_value),
                    float(_reg_loss),
                    float(_conf_loss),
                    float(_cover_loss),
                    float(_vel_loss),
                    float(_yaw_loss),
                    float(_cls_valid_ratio),
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                jloss,
                sloss,
                seg_loss_value,
                mask_loss_value,
                traj_loss,
                dice_loss_value,
                reg_loss_value,
                conf_loss_value,
                cover_loss_value,
                vel_loss_value,
                yaw_loss_value,
                cls_valid_ratio_value,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)

            if itr == 0:
                log_optimizer_group_lrs(epoch, itr)

            iter_elapsed_time_ms, _ = timer.stop_iteration()

            for m, v in [
                (loss_meter, loss),
                (jloss_meter, jloss),
                (sloss_meter, sloss),
                (seg_loss_meter, seg_loss_value),
                (mask_loss_meter, mask_loss_value),
                (dice_loss_meter, dice_loss_value),
                (traj_loss_meter, traj_loss),
                (reg_loss_meter, reg_loss_value),
                (conf_loss_meter, conf_loss_value),
                (cover_loss_meter, cover_loss_value),
                (vel_loss_meter, vel_loss_value),
                (yaw_loss_meter, yaw_loss_value),
                (cls_valid_ratio_meter, cls_valid_ratio_value),
                (iter_time_meter, iter_elapsed_time_ms),
                (gpu_time_meter, gpu_etime_ms),
                (data_elapsed_time_meter, data_elapsed_time_ms),
            ]:
                m.update(v)
            if (
                rank == 0
                and config.planner.planner_type == "diffusion"
                and ((itr % log_freq == 0) or (itr == ipe - 1))
            ):
                logger.info(
                    "[diffusion gate] cls_sample_valid_ratio=%.2f%%",
                    100.0 * cls_valid_ratio_meter.avg,
                )

            log_training_metrics(
                tb_writer=tb_writer,
                loss_meter=loss_meter,
                jloss_meter=jloss_meter,
                sloss_meter=sloss_meter,
                seg_loss_meter=seg_loss_meter,
                mask_loss_meter=mask_loss_meter,
                dice_loss_meter=dice_loss_meter,
                traj_loss_meter=traj_loss_meter,
                reg_loss_meter=reg_loss_meter,
                conf_loss_meter=conf_loss_meter,
                cover_loss_meter=cover_loss_meter,
                iter_time_meter=iter_time_meter,
                gpu_time_meter=gpu_time_meter,
                data_elapsed_time_meter=data_elapsed_time_meter,
                epoch=epoch,
                itr=itr,
                ipe=ipe,
                rank=rank,
                _new_lr=_new_lr,
                _new_wd=_new_wd,
                train_start_time=train_start_time,
                start_epoch=start_epoch,
                total_epochs=config.optimization.epochs,
                log_freq=log_freq,
                wta_loss_version=config.planner.wta_loss_version,
                awta_init_temperature=config.planner.awta_init_temperature,
                awta_exp_base=config.planner.awta_exp_base,
                awta_min_temperature=config.planner.awta_min_temperature,
                num_modes=config.planner.num_modes,
                planner_type=config.planner.planner_type,
                vel_loss_meter=vel_loss_meter,
                yaw_loss_meter=yaw_loss_meter,
                cls_valid_ratio_meter=cls_valid_ratio_meter,
            )

        has_validation = config.planner.use_planner and is_epoch_validation_due(
            val_loader, epoch, config.meta.val_freq
        )
        log_epoch_summary(
            tb_writer,
            csv_logger,
            loss_meter,
            jloss_meter,
            sloss_meter,
            seg_loss_meter,
            mask_loss_meter,
            dice_loss_meter,
            traj_loss_meter,
            reg_loss_meter,
            conf_loss_meter,
            iter_time_meter,
            gpu_time_meter,
            data_elapsed_time_meter,
            epoch,
            rank,
            has_validation,
        )

        return has_validation

    def _save_latest(epoch):
        save_checkpoint_fn(epoch + 1, latest_path)

    def _save_periodic(epoch):
        periodic_epoch = resolve_periodic_checkpoint_epoch(
            epoch,
            periodic_one_based=bool(config.meta.selection_checkpoint_epochs),
            selection_checkpoint_epochs=config.meta.selection_checkpoint_epochs,
        )
        save_checkpoint_fn(epoch + 1, os.path.join(config.meta.folder, f"e{periodic_epoch}.pt"))

    def _run_validation_isolated(epoch):
        logger.info(f"[EncoderDirect] Running validation at epoch {epoch + 1}...")
        val_metrics = _validate_encoder_direct(
            encoder=encoder,
            planner=planner,
            val_loader=val_loader,
            val_sampler=val_sampler,
            config=config,
            epoch=epoch + 1,
            rank=rank,
            world_size=world_size,
            vis_output_dir=os.path.join(config.meta.folder, "val_vis"),
        )
        log_validation_metrics(
            tb_writer,
            csv_logger,
            val_metrics,
            epoch,
            rank,
            validation_history=validation_history,
        )

        current_record = build_validation_record(epoch + 1, val_metrics)
        validation_history.append(current_record)
        best_tracker.consider(current_record, val_metrics, save_fn=save_checkpoint_fn)

    def _run_validation(epoch):
        from app.vjepa_cowa_world_model.utils.eval_determinism import preserve_eval_rng_state

        with preserve_eval_rng_state(device):
            return _run_validation_isolated(epoch)

    # Phase 5: outer epoch iteration + checkpoint cadence + barrier + validation dispatch are owned by
    # TrainingLoopRunner (composition); run_epoch keeps the full per-epoch body verbatim. Byte-identical.
    TrainingLoopRunner(
        start_epoch=start_epoch,
        epochs=config.optimization.epochs,
        ipe=ipe,
        rank=rank,
        checkpoint_freq=CHECKPOINT_FREQ,
        save_every_freq=config.meta.save_every_freq,
        save_from_epoch=config.meta.save_from_epoch,
        periodic_one_based=bool(config.meta.selection_checkpoint_epochs),
        selection_checkpoint_epochs=config.meta.selection_checkpoint_epochs,
    ).run(
        run_epoch=run_epoch,
        save_latest=_save_latest,
        save_periodic=_save_periodic,
        run_validation=_run_validation,
    )

    wait_for_checkpoint_save()
    log_training_summary(
        best_tracker.epoch,
        best_tracker.ade,
        best_tracker.fde,
        best_tracker.minade_k,
        best_tracker.minfde_k,
        best_path,
        tb_writer,
        rank,
        best_l2_avg=best_tracker.l2_avg,
        best_collision_rate=best_tracker.collision_rate,
        validation_history=validation_history,
        best_selector_label=OPEN_LOOP_SELECTION_RULE,
    )
