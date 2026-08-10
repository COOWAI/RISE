"""Main encoder runtime helpers for V-JEPA and V-JEPA backbones."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: F401  # kept for callers that mirror predictor runtime imports

from app.vjepa_cowa_world_model.training.config import (
    is_dinov2_main_encoder_config,
    is_vjepa_main_encoder_config,
    resolve_main_encoder_frame_stride,
    resolve_main_encoder_num_observed_steps,
    resolve_main_encoder_num_time_steps,
    resolve_main_encoder_raw_tokens_per_frame,
    resolve_main_encoder_tokens_per_frame,
)
from app.vjepa_cowa_world_model.training.encoder_inputs import build_tubelet_encoder_input
from app.vjepa_cowa_world_model.training.models import prepare_runtime_tokens
from app.vjepa_cowa_world_model.utils.module_utils import unwrap_module


@dataclass(frozen=True)
class MainEncoderTimeline:
    """Resolved main-encoder temporal contract."""

    raw_num_frames: int  # 原始输入有多少帧
    frame_stride: int  # 每个 predictor 时间步跨多少原始帧
    num_time_steps: int  # predictor 实际看到多少个时间步
    num_observed_steps: int  # observed 部分有多少个时间步
    num_future_steps: int  # future 部分有多少个时间步
    tokens_per_frame: int  # 每个时间步对应多少 latent tokens


@dataclass(frozen=True)
class PredictorTimelineInputs:
    """Predictor inputs after aligning raw frame tensors to main-encoder steps."""

    raw_num_frames: int
    frame_stride: int
    num_time_steps: int
    num_observed_steps: int
    num_future_steps: int
    tokens_per_frame: int
    actions: torch.Tensor
    states: torch.Tensor
    extrinsics: torch.Tensor
    driving_command: Optional[torch.Tensor]
    ego_dynamics: Optional[torch.Tensor]
    metadata_valid_mask: Optional[torch.Tensor] = None
    observed_metadata_valid_mask: Optional[torch.Tensor] = None


def _zero_cvoi_future_timeline(
    value: Optional[torch.Tensor],
    *,
    start: int,
    name: str,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor) or value.ndim < 2 or start < 0 or start > int(value.shape[1]):
        raise ValueError(f"CVoI aligned predictor {name} cannot be zeroed from index {start}")
    result = value.clone()
    result[:, start:] = 0
    return result


def enforce_cvoi_zero_future_aux(inputs: PredictorTimelineInputs) -> PredictorTimelineInputs:
    """Erase future conditions reconstructed while aligning an observed-only CVoI input."""

    if not isinstance(inputs, PredictorTimelineInputs):
        raise TypeError("CVoI timeline sanitizer requires PredictorTimelineInputs")
    num_observed = int(inputs.num_observed_steps)
    return replace(
        inputs,
        actions=_zero_cvoi_future_timeline(
            inputs.actions,
            start=max(num_observed - 1, 0),
            name="actions",
        ),
        states=_zero_cvoi_future_timeline(inputs.states, start=num_observed, name="states"),
        extrinsics=_zero_cvoi_future_timeline(inputs.extrinsics, start=num_observed, name="extrinsics"),
        driving_command=_zero_cvoi_future_timeline(
            inputs.driving_command,
            start=num_observed,
            name="driving_command",
        ),
        ego_dynamics=_zero_cvoi_future_timeline(
            inputs.ego_dynamics,
            start=num_observed,
            name="ego_dynamics",
        ),
        metadata_valid_mask=_zero_cvoi_future_timeline(
            inputs.metadata_valid_mask,
            start=num_observed,
            name="metadata_valid_mask",
        ),
    )


def is_vjepa_main_encoder(module: Optional[torch.nn.Module]) -> bool:
    if module is None:
        return False
    core = unwrap_module(module)
    return bool(getattr(core, "is_vjepa_img_encoder_adapter", False)) or (
        core.__class__.__name__ == "VJEPAImgEncoderAdapter"
    )


def is_dinov2_main_encoder(module: Optional[torch.nn.Module]) -> bool:
    if module is None:
        return False
    core = unwrap_module(module)
    return bool(getattr(core, "is_dinov2_img_encoder_adapter", False)) or (
        core.__class__.__name__ == "Dinov2ImageEncoderAdapter"
    )


def _vjepa_use_causal_attention(config) -> bool:
    return bool(config.model.vjepa_use_causal_attention)


def _vjepa_raw_observed_frames(config) -> int:
    return int(config.train.num_observed_frames)


def resolve_main_timeline(config, encoder: Optional[torch.nn.Module], num_raw_frames: int) -> MainEncoderTimeline:
    frame_stride = resolve_main_encoder_frame_stride(config, encoder)
    num_time_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_raw_frames, encoder=encoder)
    num_observed_steps = resolve_main_encoder_num_observed_steps(config, encoder)
    if num_observed_steps > num_time_steps:
        raise ValueError(
            f"Observed predictor steps ({num_observed_steps}) exceed total predictor steps ({num_time_steps})"
        )
    return MainEncoderTimeline(
        raw_num_frames=int(num_raw_frames),
        frame_stride=frame_stride,
        num_time_steps=num_time_steps,
        num_observed_steps=num_observed_steps,
        num_future_steps=num_time_steps - num_observed_steps,
        tokens_per_frame=resolve_main_encoder_tokens_per_frame(config, encoder),
    )


def _encode_single_view_main_context_tokens(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    config,
) -> tuple[torch.Tensor, int]:
    """Encode a raw clip window and return flattened per-step tokens plus step count."""
    if context_clips.ndim != 5:
        raise ValueError(f"Expected context_clips shape [B, C, T, H, W], got ndim={context_clips.ndim}")

    batch_size, _, num_raw_frames, _, _ = context_clips.shape
    dinov2_config = is_dinov2_main_encoder_config(config)
    dinov2_encoder = is_dinov2_main_encoder(encoder)
    if dinov2_config != dinov2_encoder:
        raise ValueError(
            "DINOv2 main encoder config/marker mismatch: "
            f"config backbone={config.model.backbone!r}, encoder marker/type="
            f"{dinov2_encoder}/{unwrap_module(encoder).__class__.__name__}"
        )
    if dinov2_config:
        raw_tokens = encoder(context_clips, num_observed_frames=num_raw_frames)
        num_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_raw_frames, encoder=encoder)
        expected_tokens = num_steps * resolve_main_encoder_raw_tokens_per_frame(config, encoder=encoder)
        if not isinstance(raw_tokens, torch.Tensor) or raw_tokens.ndim != 3:
            actual = None if not isinstance(raw_tokens, torch.Tensor) else tuple(raw_tokens.shape)
            raise ValueError(f"DINOv2 encoder tokens must have shape [B, N, D]; expected rank=3, actual={actual}")
        if raw_tokens.shape[0] != batch_size or raw_tokens.shape[1] != expected_tokens:
            raise ValueError(
                "DINOv2 encoder token shape mismatch: "
                f"expected batch/tokens=({batch_size}, {expected_tokens}), "
                f"actual={tuple(raw_tokens.shape)}"
            )
        return raw_tokens, num_steps

    if is_vjepa_main_encoder_config(config) or is_vjepa_main_encoder(encoder):
        use_causal_attention = _vjepa_use_causal_attention(config)
        raw_tokens = encoder(
            context_clips,
            num_observed_frames=num_raw_frames,
            use_causal_attention=use_causal_attention,
        )
        raw_observed_frames = _vjepa_raw_observed_frames(config)
        if not use_causal_attention and raw_observed_frames < num_raw_frames:
            observed_tokens = encoder(
                context_clips,
                num_observed_frames=raw_observed_frames,
                use_causal_attention=use_causal_attention,
            )
            if observed_tokens.shape[1] > raw_tokens.shape[1]:
                raise ValueError(
                    f"V-JEPA observed prefix tokens {observed_tokens.shape[1]} exceed full tokens "
                    f"{raw_tokens.shape[1]}"
                )
            raw_tokens = torch.cat([observed_tokens, raw_tokens[:, observed_tokens.shape[1] :]], dim=1)
        num_steps = resolve_main_encoder_num_time_steps(config, num_raw_frames=num_raw_frames, encoder=encoder)
        return raw_tokens, num_steps

    encoder_input = context_clips
    if bool(config.data.use_tubelet_repeat):
        encoder_input = build_tubelet_encoder_input(context_clips, config.data.tubelet_size)
    encoded = encoder([encoder_input])[0]
    if bool(config.data.use_tubelet_repeat):
        encoded = encoded.view(batch_size, num_raw_frames, -1, encoded.size(-1)).flatten(1, 2)
    return encoded, num_raw_frames


def encode_main_context_tokens(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    config,
    multiview_fusion: Optional[torch.nn.Module] = None,
    camera_metadata: Optional[dict] = None,
) -> tuple[torch.Tensor, int]:
    """Encode a raw clip window and optionally fuse multiple camera views."""
    if context_clips.ndim == 5:
        return _encode_single_view_main_context_tokens(encoder, context_clips, config)
    if context_clips.ndim != 6:
        raise ValueError(
            f"Expected context_clips shape [B, C, T, H, W] or [B, V, C, T, H, W], " f"got ndim={context_clips.ndim}"
        )
    if multiview_fusion is None:
        raise ValueError("Multi-view context clips require a multiview_fusion module")

    batch_size, num_views, channels, num_raw_frames, height, width = context_clips.shape
    flat_clips = context_clips.reshape(batch_size * num_views, channels, num_raw_frames, height, width)
    raw_tokens, num_steps = _encode_single_view_main_context_tokens(encoder, flat_clips, config)
    raw_tokens = raw_tokens.view(batch_size, num_views, raw_tokens.shape[1], raw_tokens.shape[2])

    camera_metadata = camera_metadata or {}
    camera_intrinsics = camera_metadata.get("camera_intrinsics")
    camera2ego = camera_metadata.get("camera2ego")
    # Multi-view PETR fusion is geometry-driven: the per-view position embedding is built entirely
    # from camera_intrinsics + camera2ego. If either is missing the fusion silently falls back to
    # identity geometry (every view becomes geometrically identical), which corrupts the fused
    # tokens at train/infer without any error. Fail loudly so a dataloader/feature-builder that
    # forgot to provide per-view camera geometry is caught immediately.
    missing_geometry = [
        name
        for name, value in (("camera_intrinsics", camera_intrinsics), ("camera2ego", camera2ego))
        if not torch.is_tensor(value)
    ]
    if missing_geometry:
        raise ValueError(
            "Multi-view fusion requires camera_metadata with tensor "
            f"camera_intrinsics and camera2ego, but missing/invalid: {missing_geometry}. "
            "Ensure the dataloader / NavSim feature builder provides per-view camera geometry "
            "(camera_metadata must not be empty when multiview is enabled)."
        )
    return (
        multiview_fusion(
            raw_tokens,
            camera_intrinsics=camera_intrinsics,
            camera2ego=camera2ego,
            image_shape=(height, width),
        ),
        num_steps,
    )


def forward_main_context(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    config,
    runtime_normalize_reps: bool,
    token_ae: Optional[torch.nn.Module] = None,
    multiview_fusion: Optional[torch.nn.Module] = None,
    camera_metadata: Optional[dict] = None,
) -> torch.Tensor:
    raw_tokens, num_steps = encode_main_context_tokens(
        encoder,
        context_clips,
        config,
        multiview_fusion=multiview_fusion,
        camera_metadata=camera_metadata,
    )
    return prepare_runtime_tokens(
        raw_tokens,
        num_frames=num_steps,
        normalize_reps=runtime_normalize_reps,
        token_ae=token_ae,
    )


def forward_main_context_dual(
    encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    config,
    predictor_normalize_reps: bool,
    proposal_normalize_reps: bool,
    predictor_token_ae: Optional[torch.nn.Module] = None,
    proposal_token_ae: Optional[torch.nn.Module] = None,
    multiview_fusion: Optional[torch.nn.Module] = None,
    camera_metadata: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_tokens, num_steps = encode_main_context_tokens(
        encoder,
        context_clips,
        config,
        multiview_fusion=multiview_fusion,
        camera_metadata=camera_metadata,
    )
    predictor_context = prepare_runtime_tokens(
        raw_tokens,
        num_frames=num_steps,
        normalize_reps=predictor_normalize_reps,
        token_ae=predictor_token_ae,
    )
    proposal_context = prepare_runtime_tokens(
        raw_tokens,
        num_frames=num_steps,
        normalize_reps=proposal_normalize_reps,
        token_ae=proposal_token_ae,
    )
    return predictor_context, proposal_context


def forward_main_target(
    target_encoder: torch.nn.Module,
    target_clips: torch.Tensor,
    config,
    runtime_normalize_reps: bool,
    token_ae: Optional[torch.nn.Module] = None,
    multiview_fusion: Optional[torch.nn.Module] = None,
    camera_metadata: Optional[dict] = None,
) -> torch.Tensor:
    with torch.no_grad():
        return forward_main_context(
            encoder=target_encoder,
            context_clips=target_clips,
            config=config,
            runtime_normalize_reps=runtime_normalize_reps,
            token_ae=token_ae,
            multiview_fusion=multiview_fusion,
            camera_metadata=camera_metadata,
        )


def resolve_main_target_tokens(
    *,
    reuse_context_as_target: bool,
    z_context: torch.Tensor,
    target_encoder: torch.nn.Module,
    context_clips: torch.Tensor,
    config,
    runtime_normalize_reps: bool,
    token_ae: Optional[torch.nn.Module] = None,
    multiview_fusion: Optional[torch.nn.Module] = None,
    camera_metadata: Optional[dict] = None,
) -> torch.Tensor:
    """Return detached context tokens or run the target encoder for predictor supervision."""
    if reuse_context_as_target:
        return z_context.detach()
    return forward_main_target(
        target_encoder,
        context_clips,
        config=config,
        runtime_normalize_reps=runtime_normalize_reps,
        token_ae=token_ae,
        multiview_fusion=multiview_fusion,
        camera_metadata=camera_metadata,
    )


def should_reuse_context_as_target(config, encoder: Optional[torch.nn.Module] = None) -> bool:
    """Return whether frozen main-encoder tokens can be reused as predictor targets."""
    train_cfg = config.train
    if not bool(train_cfg.reuse_context_as_target_when_frozen):
        return False
    if bool(train_cfg.encoder_train) or bool(train_cfg.encoder_ema):
        return False
    return bool(
        is_vjepa_main_encoder_config(config)
        or is_vjepa_main_encoder(encoder)
        or is_dinov2_main_encoder_config(config)
        or is_dinov2_main_encoder(encoder)
    )


def _anchor_indices(num_raw_frames: int, frame_stride: int, device: torch.device) -> torch.Tensor:
    if frame_stride <= 1:
        return torch.arange(num_raw_frames, device=device, dtype=torch.long)
    return torch.arange(frame_stride - 1, num_raw_frames, frame_stride, device=device, dtype=torch.long)


def _index_temporal(tensor: Optional[torch.Tensor], indices: torch.Tensor) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if tensor.ndim < 3:
        return tensor
    if tensor.shape[1] < int(indices[-1].item()) + 1:
        raise ValueError(
            f"Temporal tensor length {tensor.shape[1]} is too short for anchor index {int(indices[-1].item())}"
        )
    return tensor.index_select(1, indices.to(tensor.device))


def _index_metadata_valid_mask(mask: Optional[torch.Tensor], indices: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.ndim != 2:
        raise ValueError(f"metadata_valid_mask must have shape [B, T], got {tuple(mask.shape)}")
    if mask.shape[1] < int(indices[-1].item()) + 1:
        raise ValueError(
            f"metadata_valid_mask length {mask.shape[1]} is too short for anchor index {int(indices[-1].item())}"
        )
    return mask.index_select(1, indices.to(mask.device))


def _first_frame_indices(num_raw_frames: int, frame_stride: int, device: torch.device) -> torch.Tensor:
    if frame_stride <= 1:
        return torch.arange(num_raw_frames, device=device, dtype=torch.long)
    return torch.arange(0, num_raw_frames, frame_stride, device=device, dtype=torch.long)


def _pad_or_trim_temporal(tensor: torch.Tensor, length: int) -> torch.Tensor:
    if tensor.shape[1] == length:
        return tensor
    if tensor.shape[1] > length:
        return tensor[:, :length]
    pad_shape = list(tensor.shape)
    pad_shape[1] = length - tensor.shape[1]
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=1)


def build_ego_actions_between_states(states: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Build ego-frame actions between consecutive SE(2) states."""
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


def build_predictor_timeline_inputs(
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
    config,
    encoder: Optional[torch.nn.Module],
    dt: float,
    metadata_valid_mask: Optional[torch.Tensor] = None,
    observed_metadata_valid_mask: Optional[torch.Tensor] = None,
) -> PredictorTimelineInputs:
    """Align predictor side inputs to the main encoder's temporal steps."""
    del dt  # State deltas already encode displacement in the sampled timeline.
    num_raw_frames = int(states.shape[1])
    timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=num_raw_frames)

    if timeline.frame_stride == 1:
        return PredictorTimelineInputs(
            raw_num_frames=timeline.raw_num_frames,
            frame_stride=timeline.frame_stride,
            num_time_steps=timeline.num_time_steps,
            num_observed_steps=timeline.num_observed_steps,
            num_future_steps=timeline.num_future_steps,
            tokens_per_frame=timeline.tokens_per_frame,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            metadata_valid_mask=metadata_valid_mask,
            observed_metadata_valid_mask=observed_metadata_valid_mask,
        )

    indices = _anchor_indices(num_raw_frames, timeline.frame_stride, states.device)
    chunk_states = states.index_select(1, indices)
    chunk_actions = build_ego_actions_between_states(chunk_states, int(config.train.action_dim))
    chunk_extrinsics = _index_temporal(extrinsics, indices)
    if chunk_extrinsics is None:
        chunk_extrinsics = chunk_actions.new_zeros(
            chunk_actions.shape[0],
            chunk_states.shape[1],
            max(int(config.train.action_dim) - 1, 1),
        )
    chunk_driving_command = _index_temporal(driving_command, indices)
    chunk_ego_dynamics = _index_temporal(ego_dynamics, indices)
    chunk_metadata_valid_mask = _index_metadata_valid_mask(metadata_valid_mask, indices)

    return PredictorTimelineInputs(
        raw_num_frames=timeline.raw_num_frames,
        frame_stride=timeline.frame_stride,
        num_time_steps=timeline.num_time_steps,
        num_observed_steps=timeline.num_observed_steps,
        num_future_steps=timeline.num_future_steps,
        tokens_per_frame=timeline.tokens_per_frame,
        actions=chunk_actions,
        states=chunk_states,
        extrinsics=chunk_extrinsics,
        driving_command=chunk_driving_command,
        ego_dynamics=chunk_ego_dynamics,
        metadata_valid_mask=chunk_metadata_valid_mask,
        observed_metadata_valid_mask=observed_metadata_valid_mask,
    )


def build_parallel_predictor_timeline_inputs(
    actions: torch.Tensor,
    states: torch.Tensor,
    extrinsics: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
    config,
    encoder: Optional[torch.nn.Module],
    dt: float,
    metadata_valid_mask: Optional[torch.Tensor] = None,
    observed_metadata_valid_mask: Optional[torch.Tensor] = None,
) -> PredictorTimelineInputs:
    """Align side inputs for full-step future-query predictor forwarding."""
    del dt
    num_raw_frames = int(states.shape[1])
    timeline = resolve_main_timeline(config, encoder=encoder, num_raw_frames=num_raw_frames)

    if timeline.frame_stride == 1:
        full_actions = _pad_or_trim_temporal(actions, timeline.num_time_steps)
        return PredictorTimelineInputs(
            raw_num_frames=timeline.raw_num_frames,
            frame_stride=timeline.frame_stride,
            num_time_steps=timeline.num_time_steps,
            num_observed_steps=timeline.num_observed_steps,
            num_future_steps=timeline.num_future_steps,
            tokens_per_frame=timeline.tokens_per_frame,
            actions=full_actions,
            states=_pad_or_trim_temporal(states, timeline.num_time_steps),
            extrinsics=_pad_or_trim_temporal(extrinsics, timeline.num_time_steps),
            driving_command=(
                _pad_or_trim_temporal(driving_command, timeline.num_time_steps)
                if driving_command is not None and driving_command.ndim >= 3
                else driving_command
            ),
            ego_dynamics=(
                _pad_or_trim_temporal(ego_dynamics, timeline.num_time_steps)
                if ego_dynamics is not None and ego_dynamics.ndim >= 3
                else ego_dynamics
            ),
            metadata_valid_mask=(
                _pad_or_trim_temporal(metadata_valid_mask, timeline.num_time_steps)
                if metadata_valid_mask is not None and metadata_valid_mask.ndim >= 2
                else metadata_valid_mask
            ),
            observed_metadata_valid_mask=observed_metadata_valid_mask,
        )

    # Use end-of-chunk anchors (1,3,5,…) and rebuild each token-step's ego motion as the 2-frame chunk-to-chunk
    # displacement via build_ego_actions_between_states — identical to the sequential path. The previous
    # first-frame (0,2,4,…) subsampling of RAW per-frame actions captured only 1 of the 2 raw frames per step,
    # under-counting the ego motion the predictor conditions on (V-JEPA emits one token step per 2 frames).
    # NB: build ego actions from the *unpadded* selected states, then pad — padding first would synthesize
    # spurious motion from pad frames. (Independent of main's main_encoder_runtime fix for the same bug.)
    indices = _anchor_indices(num_raw_frames, timeline.frame_stride, states.device)[: timeline.num_time_steps]
    chunk_states_sel = states.index_select(1, indices)
    chunk_actions = build_ego_actions_between_states(chunk_states_sel, int(config.train.action_dim))
    chunk_actions = _pad_or_trim_temporal(chunk_actions, timeline.num_time_steps)
    chunk_states = _pad_or_trim_temporal(chunk_states_sel, timeline.num_time_steps)
    chunk_extrinsics = _index_temporal(extrinsics, indices)
    if chunk_extrinsics is None:
        chunk_extrinsics = chunk_actions.new_zeros(
            chunk_actions.shape[0],
            timeline.num_time_steps,
            max(int(config.train.action_dim) - 1, 1),
        )
    chunk_extrinsics = _pad_or_trim_temporal(chunk_extrinsics, timeline.num_time_steps)
    chunk_driving_command = _index_temporal(driving_command, indices)
    chunk_ego_dynamics = _index_temporal(ego_dynamics, indices)
    chunk_metadata_valid_mask = _index_metadata_valid_mask(metadata_valid_mask, indices)

    return PredictorTimelineInputs(
        raw_num_frames=timeline.raw_num_frames,
        frame_stride=timeline.frame_stride,
        num_time_steps=timeline.num_time_steps,
        num_observed_steps=timeline.num_observed_steps,
        num_future_steps=timeline.num_future_steps,
        tokens_per_frame=timeline.tokens_per_frame,
        actions=chunk_actions,
        states=chunk_states,
        extrinsics=chunk_extrinsics,
        driving_command=chunk_driving_command,
        ego_dynamics=chunk_ego_dynamics,
        metadata_valid_mask=(
            _pad_or_trim_temporal(chunk_metadata_valid_mask, timeline.num_time_steps)
            if chunk_metadata_valid_mask is not None and chunk_metadata_valid_mask.ndim >= 2
            else chunk_metadata_valid_mask
        ),
        observed_metadata_valid_mask=observed_metadata_valid_mask,
    )
