"""Shared helpers for staged LeWM predictor/planner training."""

from __future__ import annotations

import importlib.util
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F
import yaml

from app.vjepa_cowa_world_model.planning import states_from_traj
from app.vjepa_cowa_world_model.training.checkpoint import load_checkpoint, load_state_dict_helper
from app.vjepa_cowa_world_model.training.config import (
    normalize_image_size,
    resolve_proposal_encoder_backbone,
    resolve_proposal_num_time_steps,
    resolve_proposal_runtime_normalize_reps,
    resolve_proposal_use_token_ae,
)
from app.vjepa_cowa_world_model.training.encoder_inputs import build_tubelet_encoder_input
from app.vjepa_cowa_world_model.training.models import (
    build_predictor_input_with_future_queries,
    prepare_runtime_tokens,
)
from app.vjepa_cowa_world_model.training.validation_rng import validation_randn
from app.vjepa_cowa_world_model.utils import (
    build_observed_action_trajectory_history,
    prepare_inference_consistent_states,
    prepare_inference_consistent_status_vector,
    prepare_status_feature,
    resolve_planner_status_dim,
    resolve_planner_use_drive_command,
)
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module
from app.vjepa_cowa_world_model.utils.planner_training import compute_planner_wta_loss

_proposal_logger = logging.getLogger(__name__)

_TRAIN_LEWM_PATH = Path(__file__).resolve().parents[2] / "train_le-wm.py"
_PROPOSAL_CONTEXT_CLIPS_INDEX = 10
_PROPOSAL_MODE_EXPANSION_PATH = Path(__file__).resolve().parents[2] / "models" / "proposal_mode_expansion.py"

# The staged-pipeline stage label ("stage2"/"stage3") used to double as the config section name. The config
# sections were renamed (stage2->refinement, stage3->refinement_gated); map the label to the new section so
# callers can keep passing the stage label while the config exposes the new names.
_STAGE_CONFIG_SECTION = {"stage2": "refinement", "stage3": "refinement_gated"}


def _stage_section(config, stage):
    """Resolve the (renamed) config section for a staged-pipeline stage label."""
    return getattr(config, _STAGE_CONFIG_SECTION.get(stage, stage), None)


@lru_cache(maxsize=1)
def _load_proposal_mode_expansion_module():
    spec = importlib.util.spec_from_file_location(
        "app.vjepa_cowa_world_model.models.proposal_mode_expansion",
        _PROPOSAL_MODE_EXPANSION_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Could not load proposal mode expansion module from {_PROPOSAL_MODE_EXPANSION_PATH}")
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def load_train_lewm_module():
    spec = importlib.util.spec_from_file_location("app.vjepa_cowa_world_model.train_lewm_base", _TRAIN_LEWM_PATH)
    module = importlib.util.module_from_spec(spec)
    if not (spec.loader is not None):
        raise AssertionError("assertion failed: spec.loader is not None")
    spec.loader.exec_module(module)
    return module


def is_vjepa_proposal_encoder(module: Optional[torch.nn.Module]) -> bool:
    if module is None:
        return False
    core = unwrap_module(module)
    return bool(getattr(core, "is_vjepa_img_encoder_adapter", False)) or (
        core.__class__.__name__ == "VJEPAImgEncoderAdapter"
    )


def requires_proposal_context_clips(module: Optional[torch.nn.Module]) -> bool:
    """Return whether the proposal branch must use dataloader-provided proposal clips."""
    return is_vjepa_proposal_encoder(module)


def encode_context_tokens(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    use_tubelet_repeat: bool,
) -> tuple[torch.Tensor, int]:
    """Encode context clips once and return raw per-frame encoder tokens."""
    batch_size, _, num_frames, _, _ = context_clips.shape
    encoder_input = context_clips
    if use_tubelet_repeat:
        encoder_input = build_tubelet_encoder_input(context_clips)
    z_context = encoder([encoder_input])[0]
    if use_tubelet_repeat:
        z_context = z_context.view(batch_size, num_frames, -1, z_context.size(-1)).flatten(1, 2)
    return z_context, num_frames


def forward_context(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    use_tubelet_repeat: bool,
    runtime_normalize_reps: bool,
    token_ae: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    z_context, num_frames = encode_context_tokens(encoder, context_clips, use_tubelet_repeat)
    return prepare_runtime_tokens(
        z_context,
        num_frames=num_frames,
        normalize_reps=runtime_normalize_reps,
        token_ae=token_ae,
    )


def forward_context_dual(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    use_tubelet_repeat: bool,
    predictor_normalize_reps: bool,
    proposal_normalize_reps: bool,
    predictor_token_ae: Optional[torch.nn.Module] = None,
    proposal_token_ae: Optional[torch.nn.Module] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode once, then prepare predictor/refiner and frozen-proposal streams separately."""
    raw_tokens, num_frames = encode_context_tokens(encoder, context_clips, use_tubelet_repeat)
    predictor_context = prepare_runtime_tokens(
        raw_tokens,
        num_frames=num_frames,
        normalize_reps=predictor_normalize_reps,
        token_ae=predictor_token_ae,
    )
    proposal_context = prepare_runtime_tokens(
        raw_tokens,
        num_frames=num_frames,
        normalize_reps=proposal_normalize_reps,
        token_ae=proposal_token_ae,
    )
    return predictor_context, proposal_context


def forward_frozen_proposal(
    proposal_encoder: Optional[torch.nn.Module],
    proposal_planner: torch.nn.Module,
    context_clips: torch.Tensor,
    use_tubelet_repeat: bool,
    proposal_normalize_reps: bool,
    status_feature: torch.Tensor,
    history_traj: Optional[torch.Tensor] = None,
    proposal_token_ae: Optional[torch.nn.Module] = None,
    num_observed_frames: Optional[int] = None,
    inference_noise: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Run the frozen proposal branch and return proposal trajectories only.

    Parameters
    ----------
    proposal_encoder : optional module
        Independent frozen encoder used by transformer/diffusion proposal providers.
        History-kinematic providers ignore z_context and may pass ``None``.
    proposal_planner : module
        Frozen proposal provider returning trajectories/confidences/features.

    Returns
    -------
    dict:
        Provider output with at least ``trajectories`` and ``confidences``.
    """
    if proposal_encoder is None:
        z_context = status_feature.new_zeros(status_feature.shape[0], 1, 1)
    elif is_vjepa_proposal_encoder(proposal_encoder):
        observed_frames = int(num_observed_frames or context_clips.shape[2])
        raw_tokens = proposal_encoder(context_clips, num_observed_frames=observed_frames)
        core = unwrap_module(proposal_encoder)
        encoder_num_frames = int(getattr(core, "num_frames", 2))
        runtime_config = type(
            "_VJEPAProposalRuntimeConfig",
            (),
            {
                "proposal": type(
                    "_ProposalConfig",
                    (),
                    {
                        "encoder_backbone": "vjepa_img_encoder",
                        "vjepa_num_frames": encoder_num_frames,
                    },
                )(),
                "train": type("_TrainConfig", (), {"num_observed_frames": observed_frames})(),
            },
        )()
        # 抓手：契约对账。这里手搓的临时 runtime_config 只填了 resolver 当前需要的两个字段。
        # 如果 resolve_proposal_num_time_steps 之后被改成读取更多 config 字段（model.*、data.* 等），
        # 这个临时类会缺字段并悄悄回退到默认值，导致 token 形状错配。下面的对账 assert 会立刻报错。
        resolved_num_time_steps = resolve_proposal_num_time_steps(runtime_config, proposal_encoder=proposal_encoder)
        encoder_num_time_steps = getattr(core, "num_time_steps", None)
        if encoder_num_time_steps is not None:
            expected_num_time_steps = int(encoder_num_time_steps)
        else:
            if not (encoder_num_frames > 0 and observed_frames % encoder_num_frames == 0):
                raise AssertionError(
                    f"V-JEPA proposal encoder requires observed_frames ({observed_frames}) "
                    f"to be divisible by encoder.num_frames ({encoder_num_frames})."
                )
            expected_num_time_steps = observed_frames // encoder_num_frames
        if not (resolved_num_time_steps == expected_num_time_steps):
            raise AssertionError(
                f"V-JEPA proposal num_time_steps contract drift: "
                f"resolve_proposal_num_time_steps returned {resolved_num_time_steps}, "
                f"but the in-place computation expects {expected_num_time_steps} "
                f"(observed_frames={observed_frames}, encoder_num_frames={encoder_num_frames}, "
                f"encoder.num_time_steps={encoder_num_time_steps}). "
                f"This typically means resolve_proposal_num_time_steps was extended to read "
                f"additional config fields not provided by the temporary runtime_config above; "
                f"either extend the temp class to satisfy the new contract, or replace this block "
                f"with the direct computation."
            )
        z_context = prepare_runtime_tokens(
            raw_tokens,
            num_frames=resolved_num_time_steps,
            normalize_reps=proposal_normalize_reps,
            token_ae=proposal_token_ae,
        )
    else:
        z_context = forward_context(
            proposal_encoder,
            context_clips,
            use_tubelet_repeat=use_tubelet_repeat,
            runtime_normalize_reps=proposal_normalize_reps,
            token_ae=proposal_token_ae,
        )
    proposal_kwargs = {
        "z_context": z_context,
        "status_feature": status_feature,
        "history_traj": history_traj,
    }
    if inference_noise is not None:
        proposal_kwargs["inference_noise"] = inference_noise
    with torch.no_grad():
        return proposal_planner(**proposal_kwargs)


def resolve_proposal_token_ae_module(config: Any, token_ae: Optional[torch.nn.Module]) -> Optional[torch.nn.Module]:
    """Return the TokenAE module that should be used by the proposal stream."""
    if not resolve_proposal_use_token_ae(config):
        return None
    if token_ae is None:
        raise ValueError("proposal.use_token_ae=true requires token_ae.enabled=true and a loaded meta.ae_checkpoint")
    return token_ae


def call_planner_method(planner: torch.nn.Module, method_name: str, *args, **kwargs):
    planner_module = unwrap_module(planner)
    method = getattr(planner_module, method_name, None)
    if method is None:
        raise AttributeError(f"Planner module does not define method '{method_name}'")
    return method(*args, **kwargs)


@dataclass(frozen=True)
class Stage3RefinementInputs:
    """Inputs passed into the trainable Stage-3 refinement planner."""

    z_context: torch.Tensor
    status_feature: torch.Tensor
    proposal_traj: torch.Tensor
    proposal_logits: torch.Tensor
    proposal_features: Optional[torch.Tensor]
    predictor_rollout_fn: Optional[Callable[[torch.Tensor], torch.Tensor]]
    use_initial_proposal_features: bool


def _stage3_refine_flag(config: Any, name: str) -> bool:
    return bool(getattr(getattr(config, "refinement_gated", None), name, True))


def apply_stage3_refinement_input_gates(
    config: Any,
    z_context: torch.Tensor,
    status_feature: torch.Tensor,
    proposal_traj: torch.Tensor,
    proposal_logits: torch.Tensor,
    proposal_features: Optional[torch.Tensor],
    predictor_rollout_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
) -> Stage3RefinementInputs:
    """Mask optional inputs before calling the trainable Stage-3 refinement planner."""
    use_proposal_features = _stage3_refine_flag(config, "refine_use_proposal_features")
    return Stage3RefinementInputs(
        z_context=z_context if _stage3_refine_flag(config, "refine_use_z_context") else torch.zeros_like(z_context),
        status_feature=(
            status_feature
            if _stage3_refine_flag(config, "refine_use_status_feature")
            else torch.zeros_like(status_feature)
        ),
        proposal_traj=(
            proposal_traj
            if _stage3_refine_flag(config, "refine_use_proposal_traj")
            else torch.zeros_like(proposal_traj)
        ),
        proposal_logits=(
            proposal_logits
            if _stage3_refine_flag(config, "refine_use_proposal_logits")
            else torch.zeros_like(proposal_logits)
        ),
        proposal_features=proposal_features if use_proposal_features else None,
        predictor_rollout_fn=(
            predictor_rollout_fn if _stage3_refine_flag(config, "refine_use_predictor_rollout") else None
        ),
        use_initial_proposal_features=use_proposal_features,
    )


def select_observed_context_clips(context_clips: torch.Tensor, num_observed_frames: int) -> torch.Tensor:
    if context_clips.ndim != 5:
        raise ValueError(f"Expected context_clips shape [B, C, T, H, W], got ndim={context_clips.ndim}")
    if num_observed_frames <= 0:
        raise ValueError(f"num_observed_frames must be positive, got {num_observed_frames}")
    if context_clips.shape[2] < num_observed_frames:
        raise ValueError(
            f"Observed-context slice mismatch: got T={context_clips.shape[2]}, expected >= {num_observed_frames}"
        )
    return context_clips[:, :, :num_observed_frames]


def load_proposal_context_clips(
    sample: Any,
    fallback_context_clips: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float,
    require_proposal_context: bool = False,
) -> torch.Tensor:
    """Load proposal-specific context clips when a dataloader provides dual transforms.

    When ``require_proposal_context`` is ``True`` (for example because the caller uses
    a V-JEPA proposal encoder), the sample tuple MUST carry a tensor at the fixed
    positional slot ``_PROPOSAL_CONTEXT_CLIPS_INDEX``; otherwise the proposal branch
    would silently fall back to the predictor-side context and feed a wrong-resolution
    clip into the frozen V-JEPA adapter. Separate V-JEPA proposal encoders can reuse
    ``fallback_context_clips`` because they share the main branch input layout. The assert is the magic-index 抓手:
    any future reorder of ``navsim_world_model_collate_fn`` that breaks this contract
    will fail loudly here instead of silently producing wrong proposals.
    """
    has_proposal_slot = (
        len(sample) > _PROPOSAL_CONTEXT_CLIPS_INDEX and sample[_PROPOSAL_CONTEXT_CLIPS_INDEX] is not None
    )
    if require_proposal_context:
        if not (has_proposal_slot):
            raise AssertionError(
                f"V-JEPA proposal_encoder requires the dataloader to provide "
                f"proposal-specific context clips at sample[{_PROPOSAL_CONTEXT_CLIPS_INDEX}], "
                f"but sample has length {len(sample)} or contains None at that index. "
                f"Verify that navsim_world_model_collate_fn still appends proposal_context_frames "
                f"at index {_PROPOSAL_CONTEXT_CLIPS_INDEX} and that proposal_transform was wired "
                f"into the dataloader (see create_proposal_transforms in training/data.py)."
            )
        proposal_clips = sample[_PROPOSAL_CONTEXT_CLIPS_INDEX]
        if not (torch.is_tensor(proposal_clips)):
            raise AssertionError(
                f"sample[{_PROPOSAL_CONTEXT_CLIPS_INDEX}] is expected to be a torch.Tensor of "
                f"proposal context clips, got type={type(proposal_clips).__name__}. "
                f"Did navsim_world_model_collate_fn change its tuple layout?"
            )
    if has_proposal_slot:
        return sample[_PROPOSAL_CONTEXT_CLIPS_INDEX].to(device, dtype=dtype, non_blocking=True)
    return fallback_context_clips


def build_status_feature(
    config,
    states: torch.Tensor,
    actions: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
) -> torch.Tensor:
    if config.train.predictor_inference_consistent:
        return prepare_inference_consistent_status_vector(
            states,
            num_observed=config.train.num_observed_frames,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            state_dim=resolve_planner_status_dim(config),
            use_drive_command=resolve_planner_use_drive_command(config),
        )
    return prepare_status_feature(
        states,
        actions,
        mode=config.planner.states_mode,
        use_states_for_planner=config.planner.use_states_for_planner,
        action_dim=config.train.action_dim,
        driving_command=driving_command,
        ego_dynamics=ego_dynamics,
    )


def build_gt_trajectory(config, states: torch.Tensor, num_poses: int) -> torch.Tensor:
    states_se2 = states[:, :, [0, 1, 5]].double()
    future_start_idx = config.train.num_observed_frames if config.train.predictor_inference_consistent else 1
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
    return gt_trajectory[:, :num_poses]


def gather_mode_trajectory(pred_trajs: torch.Tensor, winner_idx: torch.Tensor) -> torch.Tensor:
    num_poses = pred_trajs.shape[2]
    winner_idx_exp = winner_idx.view(pred_trajs.shape[0], 1, 1, 1).expand(pred_trajs.shape[0], 1, num_poses, 3)
    return pred_trajs.gather(dim=1, index=winner_idx_exp).squeeze(1)


def compute_multimodal_proposal_loss(
    config,
    pred_trajs: torch.Tensor,
    pred_conf: torch.Tensor,
    gt_trajectory: torch.Tensor,
    epoch: int,
    mode_scores: Optional[torch.Tensor] = None,
) -> dict:
    """Multi-modal proposal WTA loss.

    ``mode_scores`` ([B, K], higher=better, e.g. the world-model latent
    selector's per-mode reward) optionally overrides the winner/ranking signal
    inside ``wta_loss*``; ``None`` keeps the legacy trajectory-ADE behaviour
    byte-identical (stage2/3 callers pass nothing).
    """
    return compute_planner_wta_loss(
        config,
        pred_trajs=pred_trajs,
        pred_conf=pred_conf,
        gt_traj=gt_trajectory,
        epoch=epoch,
        mode_scores=mode_scores,
        global_batch=bool(config.planner.wta_global_batch_norm),
    )


def pairwise_diversity_hinge_loss(pred_trajs: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    if pred_trajs.shape[1] < 2:
        return pred_trajs.new_zeros(())
    traj_flat = pred_trajs[..., :2].flatten(2)
    pairwise_dist = torch.cdist(traj_flat, traj_flat, p=2)
    eye = torch.eye(pairwise_dist.shape[1], device=pairwise_dist.device, dtype=torch.bool).unsqueeze(0)
    penalties = torch.clamp(margin - pairwise_dist, min=0.0)
    penalties = penalties.masked_fill(eye, 0.0)
    normalizer = pred_trajs.shape[0] * pred_trajs.shape[1] * max(pred_trajs.shape[1] - 1, 1)
    return penalties.sum() / float(normalizer)


def freeze_module_eval(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def build_proposal_history(config, actions: torch.Tensor, dt: float) -> Optional[torch.Tensor]:
    if not getattr(config, "proposal", None) or not config.proposal.enabled:
        return None
    needs_history = config.proposal.provider_type == "history_kinematic" or bool(
        getattr(config.planner, "use_action_history_for_planner", False)
    )
    if not needs_history:
        return None
    return build_observed_action_trajectory_history(
        actions=actions,
        num_observed_frames=config.train.num_observed_frames,
        action_history_dim=int(getattr(config.planner, "action_history_dim", 3)),
        dt=dt,
    )


def maybe_expand_manual_proposal(config: Any, proposal_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Optionally expand a single frozen proposal trajectory into K manual modes."""
    proposal_config = getattr(config, "proposal", None)
    if proposal_config is None or not bool(getattr(proposal_config, "manual_mode_expansion", False)):
        return proposal_out

    expander_module = _load_proposal_mode_expansion_module()
    return expander_module.expand_single_mode_proposal(proposal_out, proposal_config)


def _maybe_strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> Optional[dict[str, torch.Tensor]]:
    matched = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    return matched or None


def extract_proposal_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("proposal checkpoint must be a dict-like state container")

    state = checkpoint
    for key in ("proposal_planner", "state_dict", "planner"):
        value = state.get(key)
        if isinstance(value, dict):
            state = value
            if key == "proposal_planner":
                break

    module_stripped = _maybe_strip_prefix(state, "module.")
    if module_stripped is not None:
        state = module_stripped

    for prefix in ("proposal_planner.", "proposal_core.", "core."):
        stripped = _maybe_strip_prefix(state, prefix)
        if stripped is not None:
            return stripped

    return state


_PROPOSAL_PRETRAIN_YAML_NAMES = ("params-pretrain.yaml", "params-pretrain.yml")

# Fields that must align between proposal pretraining and current usage. Any drift
# silently distorts the input distribution fed into the frozen proposal core.
# 抓手：input domain (encoder reps / token_ae / status / history) 必须 bit-level 对齐。
_PROPOSAL_CRITICAL_FIELDS = (
    ("model.backbone", ("model", "backbone"), None),
    ("model.model_name", ("model", "model_name"), None),
    ("model.vjepa_resolution", ("model", "vjepa_resolution"), None),
    ("model.vjepa_num_frames", ("model", "vjepa_num_frames"), None),
    ("loss.normalize_reps", ("loss", "normalize_reps"), False),
    ("data.crop_size", ("data", "crop_size"), None),
    ("data.patch_size", ("data", "patch_size"), None),
    ("data.tubelet_size", ("data", "tubelet_size"), None),
    ("data.num_target_frames", ("data", "num_target_frames"), None),
    ("data.use_tubelet_repeat", ("data", "use_tubelet_repeat"), None),
    ("train.num_observed_frames", ("train", "num_observed_frames"), None),
    ("train.predictor_inference_consistent", ("train", "predictor_inference_consistent"), False),
    ("train.action_dim", ("train", "action_dim"), None),
    ("train.state_dim", ("train", "state_dim"), None),
    ("planner.action_history_dim", ("planner", "action_history_dim"), None),
    ("planner.use_action_history_for_planner", ("planner", "use_action_history_for_planner"), False),
    ("proposal.provider_type", ("proposal", "provider_type"), None),
    ("proposal.use_z_context", ("proposal", "use_z_context"), None),
    # Frozen diffusion proposal checkpoints are architecture-bound.  ``strict=False``
    # does not ignore same-name shape mismatches, so catch these at the config
    # boundary and report the real drift before ``load_state_dict`` fails deep in
    # DiffusionPlanner.
    ("planner.diff_hidden_dim", ("planner", "diff_hidden_dim"), None),
    ("planner.diff_num_layers", ("planner", "diff_num_layers"), None),
    ("planner.diff_num_heads", ("planner", "diff_num_heads"), None),
    ("planner.diff_mlp_ratio", ("planner", "diff_mlp_ratio"), None),
    ("planner.diff_sde_beta_min", ("planner", "diff_sde_beta_min"), None),
    ("planner.diff_sde_beta_max", ("planner", "diff_sde_beta_max"), None),
    ("planner.diff_inference_steps", ("planner", "diff_inference_steps"), None),
    ("planner.diff_num_samples", ("planner", "diff_num_samples"), None),
    ("planner.diff_traj_dim", ("planner", "diff_traj_dim"), None),
    ("planner.diff_trajectory_token_mode", ("planner", "diff_trajectory_token_mode"), None),
    ("planner.diff_use_anchor_frame", ("planner", "diff_use_anchor_frame"), None),
    ("planner.diff_use_last_frame_only", ("planner", "diff_use_last_frame_only"), None),
    ("planner.diff_train_prefix_conditioning", ("planner", "diff_train_prefix_conditioning"), None),
    ("planner.diff_adaln_version", ("planner", "diff_adaln_version"), None),
    ("planner.diff_independent_modes", ("planner", "diff_independent_modes"), None),
    ("planner.diff_mode_token_expansion", ("planner", "diff_mode_token_expansion"), None),
)


def _yaml_get(data: Any, *keys: str, default: Any = None) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _config_get(config: Any, *keys: str, default: Any = None) -> Any:
    cur = config
    for key in keys:
        if cur is None:
            return default
        cur = getattr(cur, key, None)
    return default if cur is None else cur


def _proposal_alignment_current_value(config: Any, key_path: tuple[str, ...], default: Any) -> Any:
    if key_path == ("model", "backbone"):
        return resolve_proposal_encoder_backbone(config)
    if key_path == ("model", "model_name"):
        proposal_model_name = _config_get(config, "proposal", "encoder_model_name")
        return proposal_model_name or _config_get(config, "model", "model_name", default=default)
    if key_path == ("model", "vjepa_resolution"):
        resolution = _config_get(config, "proposal", "vjepa_resolution", default=default)
        return list(resolution) if isinstance(resolution, tuple) else resolution
    if key_path == ("model", "vjepa_num_frames"):
        return _config_get(config, "proposal", "vjepa_num_frames", default=default)
    if key_path == ("loss", "normalize_reps"):
        return resolve_proposal_runtime_normalize_reps(config)
    if key_path == ("token_ae", "enabled"):
        return resolve_proposal_use_token_ae(config)
    return _config_get(config, *key_path, default=default)


def _skip_proposal_alignment_field(config: Any, key_path: tuple[str, ...]) -> bool:
    if not bool(_config_get(config, "proposal", "use_separate_encoder", default=False)):
        return False
    return key_path in {
        ("data", "crop_size"),
        ("data", "patch_size"),
        ("data", "tubelet_size"),
    }


def _normalize_proposal_alignment_value(label: str, value: Any) -> Any:
    if value is None:
        return None
    if label == "data.crop_size":
        return normalize_image_size(value)
    return value


def _resolve_drive_command_from_yaml(pretrain: dict) -> bool:
    planner_flag = _yaml_get(pretrain, "planner", "use_drive_command")
    if planner_flag is not None:
        return bool(planner_flag)
    return bool(_yaml_get(pretrain, "train", "use_drive_command", default=True))


def _resolve_proposal_num_modes_from_yaml(pretrain: dict) -> Optional[int]:
    for path in (("proposal", "num_modes"), ("planner", "diff_num_modes"), ("planner", "num_modes")):
        value = _yaml_get(pretrain, *path)
        if value is not None:
            return int(value)
    return None


def _resolve_current_provider_num_modes(config: Any) -> Optional[int]:
    provider_modes = _config_get(config, "proposal", "provider_num_modes", default=None)
    if provider_modes is not None:
        return int(provider_modes)
    current_modes = _config_get(config, "proposal", "num_modes", default=None)
    if current_modes is not None:
        return int(current_modes)
    return None


def _check_proposal_config_alignment(config: Any, checkpoint_path: str) -> None:
    """Verify the frozen proposal checkpoint was trained under an aligned config.

    底层逻辑：proposal 是冻结模块，其指标的可信度依赖输入分布与训练时一致。
    任何 normalize_reps/token_ae/帧数/状态维度 等颗粒度漂移，都会让指标静默失真。
    与其下游肉眼对比再排查，不如在加载时直接 raise，闭环留在最早的现场。
    """
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    candidates = [ckpt_path.parent / name for name in _PROPOSAL_PRETRAIN_YAML_NAMES]
    pretrain_yaml = next((path for path in candidates if path.is_file()), None)
    if pretrain_yaml is None:
        raise RuntimeError(
            "Frozen proposal alignment check failed: no params-pretrain.yaml beside "
            f"{ckpt_path}. 冻结 proposal 必须携带 pretrain config，才能验证 "
            "proposal.runtime_normalize_reps / proposal.use_token_ae / 帧数 / 维度 是否对齐。"
        )

    try:
        with open(pretrain_yaml, "r", encoding="utf-8") as handle:
            pretrain = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(
            f"Frozen proposal alignment check failed: could not read/parse the pretrain config "
            f"{pretrain_yaml} ({exc}). The frozen-proposal drift guard must not be silently disabled on an "
            "unreadable sidecar."
        ) from exc

    mismatches: list[str] = []

    for label, key_path, default in _PROPOSAL_CRITICAL_FIELDS:
        if _skip_proposal_alignment_field(config, key_path):
            continue
        raw_pretrain = _yaml_get(pretrain, *key_path, default=None)
        if raw_pretrain is None and default is None:
            # Field absent in pretrain yaml and no boolean fallback — skip rather than
            # forge a phantom mismatch (e.g. pretrain trained via a non-proposal app).
            continue
        pretrain_val = raw_pretrain if raw_pretrain is not None else default
        current_val = _proposal_alignment_current_value(config, key_path, default)
        pretrain_val = _normalize_proposal_alignment_value(label, pretrain_val)
        current_val = _normalize_proposal_alignment_value(label, current_val)
        if pretrain_val is None and current_val is None:
            continue
        if pretrain_val != current_val:
            mismatches.append(f"{label}: pretrain={pretrain_val!r} vs current={current_val!r}")

    current_token_ae_enabled = resolve_proposal_use_token_ae(config)
    pretrain_proposal_token_ae = _yaml_get(pretrain, "proposal", "use_token_ae", default=None)
    if pretrain_proposal_token_ae is None:
        pretrain_proposal_token_ae = False
    if bool(pretrain_proposal_token_ae) != current_token_ae_enabled:
        mismatches.append(
            "proposal.use_token_ae: "
            f"pretrain={bool(pretrain_proposal_token_ae)} vs current={current_token_ae_enabled}"
        )

    pretrain_token_ae_enabled = bool(_yaml_get(pretrain, "token_ae", "enabled", default=False))
    if current_token_ae_enabled and not pretrain_token_ae_enabled:
        mismatches.append("proposal.use_token_ae: pretrain token_ae.enabled=False vs current=True")
    elif pretrain_token_ae_enabled and current_token_ae_enabled:
        for sub_key in ("num_latent_tokens", "encoder_mode", "encoder_depth", "decoder_depth"):
            pretrain_sub = _yaml_get(pretrain, "token_ae", sub_key)
            current_sub = _config_get(config, "token_ae", sub_key)
            if pretrain_sub is not None and current_sub is not None and pretrain_sub != current_sub:
                mismatches.append(f"token_ae.{sub_key}: pretrain={pretrain_sub!r} vs current={current_sub!r}")

    # 局部 import 仅为避免循环依赖；resolve_planner_use_drive_command 是
    # use_drive_command 的唯一解析入口，禁止 except 后换一套手工解析兜底
    # （两套解析可能得出不同值，让一致性检查本身失真）。
    from app.vjepa_cowa_world_model.utils import resolve_planner_use_drive_command

    current_drive = bool(resolve_planner_use_drive_command(config))
    pretrain_drive = _resolve_drive_command_from_yaml(pretrain)
    if pretrain_drive != current_drive:
        mismatches.append(f"resolved use_drive_command: pretrain={pretrain_drive} vs current={current_drive}")

    pretrain_modes = _resolve_proposal_num_modes_from_yaml(pretrain)
    current_provider_modes = _resolve_current_provider_num_modes(config)
    if pretrain_modes is not None and current_provider_modes is not None and pretrain_modes != current_provider_modes:
        mismatches.append(
            f"proposal.provider_num_modes: pretrain={pretrain_modes} vs current={current_provider_modes}"
        )

    if not mismatches:
        _proposal_logger.info("Frozen proposal alignment check passed for %s (vs %s).", ckpt_path, pretrain_yaml)
        return

    detail = "\n    - ".join(mismatches)
    raise RuntimeError(
        "Frozen proposal checkpoint is incompatible with the current config — proposal "
        "metrics would silently regress.\n"
        f"  ckpt:           {ckpt_path}\n"
        f"  pretrain yaml:  {pretrain_yaml}\n"
        f"  mismatched fields:\n    - {detail}\n"
        "  抓手：要么把当前 yaml 与 proposal pretrain 拉齐 "
        "(proposal.runtime_normalize_reps / proposal.use_token_ae / 帧数 / 维度),\n"
        "  要么改用一份在当前配置下训练的 proposal checkpoint。"
    )


def load_frozen_proposal_provider(
    module: torch.nn.Module,
    checkpoint_path: Optional[str],
    *,
    config: Optional[Any] = None,
    freeze: bool = True,
) -> None:
    """Load proposal-provider weights from a checkpoint.

    ``freeze=True``(默认,stage2/3 语义)加载后冻结并 eval;``freeze=False``
    仅加载权重(train_reward warm-start:provider 是被训练对象,保持可训练)。
    """
    if not checkpoint_path:
        return

    if config is not None:
        _check_proposal_config_alignment(config, checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(f"Proposal checkpoint not found: {checkpoint_path}")

    target = getattr(module, "core", module)
    state_dict = extract_proposal_state_dict(checkpoint)
    load_state_dict_helper(target, state_dict, "proposal_provider")
    if freeze:
        freeze_module_eval(module)


def _proposal_encoder_checkpoint_path(config: Any) -> Optional[str]:
    proposal_cfg = getattr(config, "proposal", None)
    if proposal_cfg is None:
        return None
    return getattr(proposal_cfg, "encoder_checkpoint", None) or getattr(proposal_cfg, "checkpoint", None)


def _extract_encoder_state_dict_from_checkpoint(checkpoint: dict, preferred_key: str) -> dict[str, torch.Tensor]:
    candidate_keys = []
    for key in (preferred_key, "encoder", "ema_encoder", "target_encoder"):
        if key and key not in candidate_keys:
            candidate_keys.append(key)

    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    for prefix in (f"{preferred_key}.", "encoder.", "ema_encoder.", "target_encoder."):
        if not prefix or prefix == ".":
            continue
        stripped = _maybe_strip_prefix(checkpoint, prefix)
        if stripped is not None:
            return stripped

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    available = sorted(str(key) for key in checkpoint.keys())[:20]
    raise RuntimeError(
        "Proposal encoder checkpoint does not contain compatible encoder weights. "
        f"preferred_key={preferred_key!r}, available_keys={available}"
    )


def load_frozen_proposal_encoder(module: Optional[torch.nn.Module], config: Any) -> None:
    """Load and freeze the independent proposal encoder when configured."""
    proposal_cfg = getattr(config, "proposal", None)
    if module is None or proposal_cfg is None or not getattr(proposal_cfg, "use_separate_encoder", False):
        return
    if not getattr(proposal_cfg, "encoder_freeze", True):
        raise ValueError("proposal.encoder_freeze=false is not supported for staged frozen proposal branches")

    checkpoint_path = _proposal_encoder_checkpoint_path(config)
    if not checkpoint_path:
        raise ValueError(
            "proposal.use_separate_encoder=true requires proposal.encoder_checkpoint or proposal.checkpoint "
            "with encoder weights"
        )

    preferred_key = getattr(proposal_cfg, "encoder_checkpoint_key", "encoder")
    if is_vjepa_proposal_encoder(module):
        vjepa_key = getattr(proposal_cfg, "vjepa_checkpoint_key", None) or preferred_key
        unwrap_module(module).load_backbone_checkpoint(checkpoint_path, vjepa_key)
        freeze_module_eval(module)
        return

    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        raise FileNotFoundError(f"Proposal encoder checkpoint not found: {checkpoint_path}")

    state_dict = _extract_encoder_state_dict_from_checkpoint(checkpoint, preferred_key)
    load_state_dict_helper(module, state_dict, "proposal_encoder")
    freeze_module_eval(module)


def _expand_driving_command(driving_command: Optional[torch.Tensor], batch_size: int, num_modes: int):
    if driving_command is None:
        return None
    if driving_command.ndim == 2:
        return driving_command[:, None, :].expand(batch_size, num_modes, -1).reshape(batch_size * num_modes, -1)
    if driving_command.ndim == 3:
        return (
            driving_command[:, None, :, :]
            .expand(batch_size, num_modes, -1, -1)
            .reshape(
                batch_size * num_modes,
                driving_command.shape[1],
                driving_command.shape[2],
            )
        )
    raise ValueError(f"Unsupported driving_command shape: {tuple(driving_command.shape)}")


def _build_ego_actions_between_states(states: torch.Tensor, action_dim: int) -> torch.Tensor:
    if states.shape[1] < 2:
        return states.new_zeros(states.shape[0], 0, action_dim)
    current = states[:, :-1]
    nxt = states[:, 1:]
    dx_global = nxt[..., 0] - current[..., 0]
    dy_global = nxt[..., 1] - current[..., 1]
    yaw = current[..., 5]
    cos_h = torch.cos(-yaw)
    sin_h = torch.sin(-yaw)
    dx_ego = cos_h * dx_global - sin_h * dy_global
    dy_ego = sin_h * dx_global + cos_h * dy_global
    d_yaw = nxt[..., 5] - current[..., 5]
    d_yaw = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))
    actions = states.new_zeros(states.shape[0], states.shape[1] - 1, action_dim)
    if action_dim > 0:
        actions[..., 0] = dx_ego
    if action_dim > 1:
        actions[..., 1] = dy_ego
    if action_dim > 2:
        actions[..., 2] = d_yaw
    return actions


def _build_ego_actions_between_traj_points(traj_xy_yaw: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Build predictor actions as ego-frame displacement deltas, matching NavSim dataset actions."""
    if traj_xy_yaw.ndim != 4 or traj_xy_yaw.shape[-1] != 3:
        raise ValueError(f"traj_xy_yaw must be [B, K, T, 3], got {tuple(traj_xy_yaw.shape)}")
    if traj_xy_yaw.shape[2] < 2:
        return traj_xy_yaw.new_zeros(traj_xy_yaw.shape[0], traj_xy_yaw.shape[1], 0, action_dim)

    current = traj_xy_yaw[:, :, :-1]
    nxt = traj_xy_yaw[:, :, 1:]
    dx_global = nxt[..., 0] - current[..., 0]
    dy_global = nxt[..., 1] - current[..., 1]
    yaw = current[..., 2]
    cos_h = torch.cos(-yaw)
    sin_h = torch.sin(-yaw)
    dx_ego = cos_h * dx_global - sin_h * dy_global
    dy_ego = sin_h * dx_global + cos_h * dy_global
    d_yaw = nxt[..., 2] - current[..., 2]
    d_yaw = torch.atan2(torch.sin(d_yaw), torch.cos(d_yaw))

    actions = traj_xy_yaw.new_zeros(traj_xy_yaw.shape[0], traj_xy_yaw.shape[1], traj_xy_yaw.shape[2] - 1, action_dim)
    if action_dim > 0:
        actions[..., 0] = dx_ego
    if action_dim > 1:
        actions[..., 1] = dy_ego
    if action_dim > 2:
        actions[..., 2] = d_yaw
    return actions


def _future_traj_to_predictor_actions(traj_xy_yaw: torch.Tensor, dt: float, action_dim: int) -> torch.Tensor:
    """Convert a future trajectory prefix to the action convention consumed by the predictor.

    ``dt`` is intentionally ignored: NavSim actions in this codebase are per-step
    ego displacement deltas, not velocities.
    """
    del dt
    return _build_ego_actions_between_traj_points(traj_xy_yaw, action_dim)


def _downsample_observed_indices(raw_num_obs: int, stride: int, device: torch.device) -> torch.Tensor:
    if stride <= 1:
        return torch.arange(raw_num_obs, device=device, dtype=torch.long)
    if raw_num_obs % stride != 0:
        raise ValueError(f"num_observed_frames ({raw_num_obs}) must be divisible by predictor_frame_stride ({stride})")
    return torch.arange(stride - 1, raw_num_obs, stride, device=device, dtype=torch.long)


def resolve_stage_predictor_rollout_seconds(config: Any, stage: str) -> Optional[float]:
    stage_config = _stage_section(config, stage)
    rollout_seconds = getattr(stage_config, "predictor_rollout_seconds", None)
    if rollout_seconds is None:
        return None
    rollout_seconds = float(rollout_seconds)
    return rollout_seconds if rollout_seconds > 0 else None


def resolve_predictor_rollout_num_steps(
    rollout_seconds: Optional[float],
    total_steps: int,
    fps: float,
    predictor_frame_stride: int = 1,
) -> Optional[int]:
    if rollout_seconds is None:
        return None
    rollout_seconds = float(rollout_seconds)
    if rollout_seconds <= 0:
        return None
    total_steps = int(total_steps)
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    predictor_frame_stride = max(1, int(predictor_frame_stride))

    seconds_per_step = predictor_frame_stride / fps
    num_steps = max(1, int(math.ceil((rollout_seconds / seconds_per_step) - 1e-9)))
    if num_steps > total_steps:
        available_seconds = total_steps * seconds_per_step
        raise ValueError(
            f"predictor_rollout_seconds={rollout_seconds:g} requires predictor steps={num_steps}, "
            f"but only {total_steps} steps ({available_seconds:g}s) are available"
        )
    return num_steps


def select_predictor_rollout_tokens(
    z_fut_m: torch.Tensor,
    rollout_seconds: Optional[float],
    tokens_per_frame: int,
    fps: float,
    predictor_frame_stride: int = 1,
) -> torch.Tensor:
    if rollout_seconds is None:
        return z_fut_m
    rollout_seconds = float(rollout_seconds)
    if rollout_seconds <= 0:
        return z_fut_m
    if z_fut_m.ndim != 4:
        raise ValueError(f"Expected predictor rollout tokens [B, K, N, D], got ndim={z_fut_m.ndim}")
    tokens_per_frame = int(tokens_per_frame)
    if tokens_per_frame <= 0:
        raise ValueError(f"tokens_per_frame must be positive, got {tokens_per_frame}")
    fps = float(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    predictor_frame_stride = max(1, int(predictor_frame_stride))

    total_tokens = int(z_fut_m.shape[2])
    if total_tokens % tokens_per_frame != 0:
        raise ValueError(
            f"predictor rollout token count ({total_tokens}) must be divisible by "
            f"tokens_per_frame ({tokens_per_frame})"
        )
    total_steps = total_tokens // tokens_per_frame
    num_steps = resolve_predictor_rollout_num_steps(
        rollout_seconds,
        total_steps=total_steps,
        fps=fps,
        predictor_frame_stride=predictor_frame_stride,
    )
    if num_steps is None:
        return z_fut_m
    if num_steps == total_steps:
        return z_fut_m
    return z_fut_m[:, :, : num_steps * tokens_per_frame, :]


def build_stage_predictor_rollout_fn(
    *,
    stage: str,
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
    config: Any,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    dt: float,
    predictor_observed_steps: Optional[int] = None,
    predictor_frame_stride: int = 1,
    validation_sample_ids: Optional[Sequence[str]] = None,
    validation_base_seed: Optional[int] = None,
    validation_protocol: Optional[str] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if stage not in {"stage2", "stage3"}:
        raise ValueError(f"stage must be 'stage2' or 'stage3', got {stage!r}")

    stage_config = _stage_section(config, stage)
    use_random_latent = bool(getattr(stage_config, "refine_use_random_predictor_latent", False))
    keep_initial_actions = bool(getattr(stage_config, "refine_keep_initial_actions", False))
    initial_action_trajs: Optional[torch.Tensor] = None
    rollout_seconds = resolve_stage_predictor_rollout_seconds(config, stage)
    validation_call_index = 0

    def _predictor_rollout_fn(traj_modes: torch.Tensor) -> torch.Tensor:
        nonlocal initial_action_trajs, validation_call_index
        future_action_trajs = None
        if keep_initial_actions:
            if initial_action_trajs is None:
                initial_action_trajs = traj_modes.detach()
            else:
                future_action_trajs = initial_action_trajs.to(device=traj_modes.device, dtype=traj_modes.dtype)

        output = rollout_predictor_modes(
            predictor=predictor,
            z_context=z_context,
            future_trajs=traj_modes,
            future_action_trajs=future_action_trajs,
            actions=actions,
            states=states,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            config=config,
            tokens_per_frame=tokens_per_frame,
            runtime_normalize_reps=runtime_normalize_reps,
            dt=dt,
            predictor_observed_steps=predictor_observed_steps,
            predictor_frame_stride=predictor_frame_stride,
            use_random_latent=use_random_latent,
            predictor_rollout_seconds=rollout_seconds,
            validation_sample_ids=validation_sample_ids,
            validation_base_seed=validation_base_seed,
            validation_protocol=validation_protocol,
            validation_stream=f"predictor_initial_noise/round-{validation_call_index}",
        )
        validation_call_index += 1
        return output

    return _predictor_rollout_fn


def resolve_stage2_refinement_num_rounds(config: Any) -> int:
    stage2_config = getattr(config, "refinement", None)
    num_rounds = int(getattr(stage2_config, "inference_num_rounds", 1))
    if num_rounds < 1:
        raise ValueError(f"stage2.inference_num_rounds must be >= 1, got {num_rounds}")
    return num_rounds


def run_stage2_refinement_for_training(
    *,
    planner: Any,
    config: Any,
    z_context: torch.Tensor,
    status_feature: torch.Tensor,
    proposal_trajs: torch.Tensor,
    proposal_logits: torch.Tensor,
    proposal_features: Optional[torch.Tensor],
    predictor_rollout_fn: Callable[[torch.Tensor], torch.Tensor],
    call_planner_method_fn: Callable[..., Any],
) -> torch.Tensor:
    num_rounds = resolve_stage2_refinement_num_rounds(config)
    detach_future = bool(getattr(getattr(config, "refinement", None), "detach_zfut", True))
    if num_rounds > 1:
        _traj_rounds, traj_final = call_planner_method_fn(
            planner,
            "forward_iterative",
            z_context,
            status_feature,
            proposal_traj=proposal_trajs,
            proposal_logits=proposal_logits,
            proposal_features=proposal_features,
            predictor_rollout_fn=predictor_rollout_fn,
            num_rounds=num_rounds,
            grad_checkpoint=False,
            detach_future=detach_future,
            use_initial_proposal_features=True,
        )
        if traj_final is None:
            raise ValueError("Stage-2 iterative training requires forward_iterative to return traj_final")
        return traj_final

    z_fut_m = predictor_rollout_fn(proposal_trajs)
    if z_fut_m is not None and detach_future:
        z_fut_m = z_fut_m.detach()
    return call_planner_method_fn(
        planner,
        "forward_refine",
        z_context,
        status_feature,
        proposal_trajs,
        z_fut_m,
        proposal_features,
    )


def _expand_or_build_rollout_command(
    driving_command: Optional[torch.Tensor],
    batch_size: int,
    num_modes: int,
    observed_indices: torch.Tensor,
    num_future: int,
):
    if driving_command is None:
        return None
    if driving_command.ndim != 3:
        return _expand_driving_command(driving_command, batch_size, num_modes)
    obs_cmd = driving_command.index_select(1, observed_indices.to(driving_command.device))
    future_cmd = obs_cmd[:, -1:, :].expand(batch_size, num_future, obs_cmd.shape[-1])
    full_cmd = torch.cat([obs_cmd, future_cmd], dim=1)
    return (
        full_cmd[:, None, :, :]
        .expand(batch_size, num_modes, -1, -1)
        .reshape(
            batch_size * num_modes,
            full_cmd.shape[1],
            full_cmd.shape[2],
        )
    )


def rollout_predictor_modes(
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    future_trajs: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
    config,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    dt: float,
    predictor_observed_steps: Optional[int] = None,
    predictor_frame_stride: int = 1,
    future_action_trajs: Optional[torch.Tensor] = None,
    use_random_latent: bool = False,
    predictor_rollout_seconds: Optional[float] = None,
    validation_sample_ids: Optional[Sequence[str]] = None,
    validation_base_seed: Optional[int] = None,
    validation_protocol: Optional[str] = None,
    validation_stream: str = "predictor_initial_noise",
) -> torch.Tensor:
    batch_size, num_modes, raw_num_future, _ = future_trajs.shape
    if future_action_trajs is not None and tuple(future_action_trajs.shape) != tuple(future_trajs.shape):
        raise ValueError(
            "future_action_trajs must match future_trajs shape before predictor_frame_stride downsampling: "
            f"future_action_trajs={tuple(future_action_trajs.shape)}, future_trajs={tuple(future_trajs.shape)}"
        )
    raw_num_obs = config.train.num_observed_frames if config.train.predictor_inference_consistent else 1
    frame_stride = max(1, int(predictor_frame_stride))
    effective_dt = float(dt) * frame_stride
    if frame_stride > 1:
        if raw_num_future % frame_stride != 0:
            raise ValueError(
                f"future_trajs length ({raw_num_future}) must be divisible by predictor_frame_stride ({frame_stride})"
            )
        future_trajs = future_trajs[:, :, frame_stride - 1 :: frame_stride]
        if future_action_trajs is not None:
            future_action_trajs = future_action_trajs[:, :, frame_stride - 1 :: frame_stride]
    num_future = future_trajs.shape[2]
    max_future_steps = resolve_predictor_rollout_num_steps(
        predictor_rollout_seconds,
        total_steps=num_future,
        fps=getattr(getattr(config, "data", None), "fps", 1.0),
        predictor_frame_stride=frame_stride,
    )
    if max_future_steps is not None and max_future_steps < num_future:
        future_trajs = future_trajs[:, :, :max_future_steps]
        if future_action_trajs is not None:
            future_action_trajs = future_action_trajs[:, :, :max_future_steps]
        num_future = max_future_steps
    num_obs = int(predictor_observed_steps) if predictor_observed_steps is not None else raw_num_obs

    validation_key_values = (validation_sample_ids, validation_base_seed, validation_protocol)
    if any(value is not None for value in validation_key_values) and not all(
        value is not None for value in validation_key_values
    ):
        raise ValueError(
            "validation_sample_ids, validation_base_seed, and validation_protocol must be provided together"
        )
    explicit_initial_noise = None
    if validation_sample_ids is not None:
        if len(validation_sample_ids) != batch_size:
            raise ValueError(
                f"validation_sample_ids length {len(validation_sample_ids)} != rollout batch_size {batch_size}"
            )
        explicit_initial_noise = validation_randn(
            (batch_size, num_modes, num_future * int(tokens_per_frame), z_context.size(-1)),
            base_seed=int(validation_base_seed),
            sample_ids=validation_sample_ids,
            protocol=str(validation_protocol),
            horizon=int(num_future),
            stream=validation_stream,
            device=z_context.device,
            dtype=z_context.dtype,
        )

    if use_random_latent:
        if explicit_initial_noise is not None:
            return explicit_initial_noise
        return z_context.new_empty(batch_size, num_modes, num_future * tokens_per_frame, z_context.size(-1)).normal_()

    origin = future_trajs.new_zeros(batch_size, num_modes, 1, future_trajs.shape[-1])
    traj_with_origin = torch.cat([origin, future_trajs], dim=2)
    action_trajs = future_trajs if future_action_trajs is None else future_action_trajs
    action_origin = action_trajs.new_zeros(batch_size, num_modes, 1, action_trajs.shape[-1])
    action_traj_with_origin = torch.cat([action_origin, action_trajs], dim=2)
    future_actions = _future_traj_to_predictor_actions(
        action_traj_with_origin,
        dt=effective_dt,
        action_dim=int(config.train.action_dim),
    )
    future_states_raw = states_from_traj(traj_with_origin, dt=effective_dt, state_dim=7, inference_consistent=False)[
        :, :, 1:, :
    ]

    observed_indices = _downsample_observed_indices(raw_num_obs, frame_stride, states.device)
    if frame_stride > 1:
        observed_states_source = states.index_select(1, observed_indices)
        observed_actions_source = _build_ego_actions_between_states(observed_states_source, config.train.action_dim)
    else:
        observed_states_source = states[:, :raw_num_obs, :7]
        observed_actions_source = actions[:, : max(0, raw_num_obs - 1)]

    observed_actions = (
        observed_actions_source[:, : max(0, num_obs - 1)].unsqueeze(1).expand(batch_size, num_modes, -1, -1)
    )
    full_actions = torch.cat([observed_actions, future_actions], dim=2).reshape(
        batch_size * num_modes, -1, config.train.action_dim
    )

    observed_states_raw = observed_states_source[:, :num_obs, :7].unsqueeze(1).expand(batch_size, num_modes, -1, -1)
    full_states_raw = torch.cat([observed_states_raw, future_states_raw], dim=2).reshape(batch_size * num_modes, -1, 7)

    rollout_driving_command = _expand_or_build_rollout_command(
        driving_command,
        batch_size,
        num_modes,
        observed_indices,
        num_future,
    )
    rollout_ego_dynamics = None
    if config.train.predictor_inference_consistent:
        if ego_dynamics is not None:
            if frame_stride > 1:
                obs_dyn_source = ego_dynamics.index_select(1, observed_indices.to(ego_dynamics.device))
            else:
                obs_dyn_source = ego_dynamics[:, :raw_num_obs]
            obs_dyn = obs_dyn_source[:, :num_obs].unsqueeze(1).expand(batch_size, num_modes, -1, -1)

            future_vx = future_actions[..., 0]
            future_vy = future_actions[..., 1]
            future_ax = (
                torch.cat([future_vx[..., 1:] - future_vx[..., :-1], torch.zeros_like(future_vx[..., :1])], dim=-1)
                / effective_dt
            )
            future_ay = (
                torch.cat([future_vy[..., 1:] - future_vy[..., :-1], torch.zeros_like(future_vy[..., :1])], dim=-1)
                / effective_dt
            )
            future_dyn = torch.stack([future_vx, future_vy, future_ax, future_ay], dim=-1)
            full_dyn = torch.cat([obs_dyn, future_dyn], dim=2).reshape(batch_size * num_modes, -1, 4)
            rollout_ego_dynamics = full_dyn
        elif int(config.train.state_dim) >= 8:
            # fail-loud: state_dim>=8 的 IC rollout 需要观测帧真实 [vx,vy,ax,ay]。旧代码在此用
            # 零速度拼出非 None 的 full_dyn，会绕过 _prepare_drive_command_8 的 None 守卫。
            raise ValueError(
                "rollout_predictor_modes: config.train.state_dim>=8 但 ego_dynamics 缺失；"
                "禁止用零速度伪造观测帧动力学（请提供 ego_dynamics 或改用 state_dim=7）。"
            )
        else:
            # state_dim=7：传 None，让 _prepare_drive_command_7 回退到 states[:, :, 6] 真实速度。
            full_dyn = None
        full_states = prepare_inference_consistent_states(
            full_states_raw,
            num_observed=num_obs,
            driving_command=rollout_driving_command,
            ego_dynamics=full_dyn,
            state_dim=config.train.state_dim,
            use_drive_command=config.train.use_drive_command,
        )
    elif config.train.use_states_for_predictor:
        if config.train.state_dim <= full_states_raw.shape[-1]:
            full_states = full_states_raw[..., : config.train.state_dim]
        else:
            full_states = full_states_raw.new_zeros(
                full_states_raw.shape[0], full_states_raw.shape[1], config.train.state_dim
            )
            full_states[..., : full_states_raw.shape[-1]] = full_states_raw
    else:
        full_states = full_states_raw.new_zeros(
            full_states_raw.shape[0], full_states_raw.shape[1], config.train.state_dim
        )

    action_steps = full_actions.shape[1]
    extrinsics = full_actions.new_zeros(full_actions.shape[0], action_steps, max(config.train.action_dim - 1, 1))
    rollout = z_context[:, : num_obs * tokens_per_frame].repeat_interleave(num_modes, dim=0)

    if str(getattr(config.train, "predictor_type", "")).lower() == "latent_dit":
        from app.vjepa_cowa_world_model.training.runtimes.latent_diffusion_runtime import (
            latent_dit_masked_inpainting_enabled,
            resolve_latent_dit_sampler_params,
            sample_latent_dit_predictor,
        )

        predictor_core = unwrap_module(predictor)
        latent_extrinsics_dim = getattr(predictor_core, "extrinsics_dim", None)
        if latent_extrinsics_dim is None:
            raise ValueError("latent_dit predictor must expose extrinsics_dim for candidate rollout sampling")
        latent_extrinsics = full_actions.new_zeros(
            full_actions.shape[0],
            full_states_raw.shape[1],
            int(latent_extrinsics_dim),
        )
        predictor_inputs = SimpleNamespace(
            actions=full_actions,
            states=full_states_raw,
            extrinsics=latent_extrinsics,
            driving_command=rollout_driving_command,
            ego_dynamics=rollout_ego_dynamics,
            metadata_valid_mask=None,
            observed_metadata_valid_mask=None,
        )
        sample_kwargs = resolve_latent_dit_sampler_params(config).as_kwargs()
        if latent_dit_masked_inpainting_enabled(config) or predictor_rollout_seconds is not None:
            sample_kwargs["future_steps"] = int(num_future)
        z_pred = sample_latent_dit_predictor(
            predictor=predictor,
            z_context=rollout,
            predictor_inputs=predictor_inputs,
            tokens_per_frame=int(tokens_per_frame),
            num_observed_steps=num_obs,
            runtime_normalize_reps=bool(runtime_normalize_reps),
            config=config,
            initial_noise=(
                None
                if explicit_initial_noise is None
                else explicit_initial_noise.reshape(
                    batch_size * num_modes,
                    explicit_initial_noise.shape[2],
                    explicit_initial_noise.shape[3],
                )
            ),
            **sample_kwargs,
        )
        return z_pred.reshape(batch_size, num_modes, z_pred.shape[1], z_pred.shape[2])

    if bool(getattr(config.train, "use_parallel_predictor", False)):
        predictor_core = unwrap_module(predictor)
        if getattr(predictor_core, "future_query_tokens", None) is not None:
            parallel_actions = torch.cat(
                [full_actions, full_actions.new_zeros(full_actions.shape[0], 1, full_actions.shape[-1])], dim=1
            )
            parallel_extrinsics = parallel_actions.new_zeros(
                parallel_actions.shape[0],
                parallel_actions.shape[1],
                max(config.train.action_dim - 1, 1),
            )
            z_input = build_predictor_input_with_future_queries(predictor, rollout)
            num_parallel_steps = z_input.shape[1] // tokens_per_frame
            if parallel_actions.shape[1] != num_parallel_steps:
                raise ValueError(
                    "Parallel staged rollout action steps mismatch: "
                    f"actions={parallel_actions.shape[1]}, expected={num_parallel_steps}"
                )
            if full_states_raw.shape[1] != num_parallel_steps:
                raise ValueError(
                    "Parallel staged rollout state steps mismatch: "
                    f"states={full_states_raw.shape[1]}, expected={num_parallel_steps}"
                )
            z_pred = predictor(z_input, parallel_actions, full_states, parallel_extrinsics)
            if runtime_normalize_reps:
                z_pred = F.layer_norm(z_pred, (z_pred.size(-1),))
            return z_pred[:, num_obs * tokens_per_frame :].reshape(
                batch_size, num_modes, num_future * tokens_per_frame, z_pred.size(-1)
            )

    for step in range(num_obs, num_obs + num_future):
        step_actions = full_actions[:, :step]
        step_states = full_states[:, :step]
        step_extrinsics = extrinsics[:, :step]
        next_tokens = predictor(rollout, step_actions, step_states, step_extrinsics)[:, -tokens_per_frame:]
        if runtime_normalize_reps:
            next_tokens = F.layer_norm(next_tokens, (next_tokens.size(-1),))
        rollout = torch.cat([rollout, next_tokens], dim=1)

    return rollout[:, num_obs * tokens_per_frame :].reshape(
        batch_size, num_modes, num_future * tokens_per_frame, rollout.size(-1)
    )


def save_stage_checkpoint(
    path: str,
    encoder: torch.nn.Module,
    target_encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    planner: Optional[torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scaler,
    scheduler,
    wd_scheduler,
    epoch: int,
    loss: float,
    config,
    rank: int,
    world_size: int,
    extra_state: Optional[dict[str, Any]] = None,
) -> None:
    if rank != 0:
        return

    def _state_dict(module: Optional[torch.nn.Module]):
        if module is None:
            return None
        inner = unwrap_module(module)
        return inner.state_dict()

    save_dict = {
        "opt": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "batch_size": config.data.batch_size,
        "world_size": world_size,
        "lr": config.optimization.lr,
        "encoder": _state_dict(encoder),
        "target_encoder": _state_dict(target_encoder),
        "predictor": _state_dict(predictor),
        "saved_modules": ["encoder", "target_encoder", "predictor"],
    }
    if getattr(config, "proposal", None) is not None and config.proposal.enabled:
        save_dict["proposal"] = {
            "provider_type": config.proposal.provider_type,
            "checkpoint": config.proposal.checkpoint,
            "use_separate_encoder": getattr(config.proposal, "use_separate_encoder", False),
            "encoder_backbone": getattr(config.proposal, "encoder_backbone", None),
            "encoder_model_name": getattr(config.proposal, "encoder_model_name", None),
            "encoder_checkpoint": getattr(config.proposal, "encoder_checkpoint", None),
            "encoder_checkpoint_key": getattr(config.proposal, "encoder_checkpoint_key", "encoder"),
            "encoder_freeze": getattr(config.proposal, "encoder_freeze", True),
            "vjepa_resolution": getattr(config.proposal, "vjepa_resolution", None),
            "vjepa_crop_top_bottom": getattr(config.proposal, "vjepa_crop_top_bottom", None),
            "vjepa_num_frames": getattr(config.proposal, "vjepa_num_frames", None),
            "vjepa_checkpoint_key": getattr(config.proposal, "vjepa_checkpoint_key", None),
            "vjepa_use_grid_mask": getattr(config.proposal, "vjepa_use_grid_mask", None),
            "freeze": config.proposal.freeze,
            "num_modes": config.proposal.num_modes,
            "log_metrics_only": config.proposal.log_metrics_only,
        }
    if planner is not None:
        save_dict["planner"] = _state_dict(planner)
        save_dict["saved_modules"].append("planner")
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    if wd_scheduler is not None:
        save_dict["wd_scheduler"] = wd_scheduler.state_dict()
    if extra_state is not None:
        save_dict.update(extra_state)
    torch.save(save_dict, path)


# shared by train_refinement / train_refinement_gated (moved verbatim)
def _planner_command_dim(config) -> int:
    use_cmd = resolve_planner_use_drive_command(config)
    if use_cmd and config.planner.split_status_embedding and config.train.predictor_inference_consistent:
        return 4
    return 0
