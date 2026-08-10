"""Runtime helpers shared by reward-model training and inference."""

from typing import Any, Mapping, Optional, Tuple

import torch

from app.vjepa_cowa_world_model.training.config import TrainingConfig
from app.vjepa_cowa_world_model.training.reward_labels import RewardLabelConfig

# Index of agent boxes / mask in the world-model collate tuple (navsim + b2d).
# See navsim_world_model_collate_fn / b2d_world_model_collate_fn return order.
_COLLATE_AGENT_BOXES_IDX = 7
_COLLATE_AGENT_MASK_IDX = 8


def _validate_normalized_real_geometry_metadata(sample: Any, *, batch_size: int) -> None:
    """Reject normalized NavSim placeholder geometry before tensor access."""

    metadata = sample[-1] if sample and isinstance(sample[-1], Mapping) else None
    if metadata is None or "geometry_present" not in metadata:
        return
    domains = metadata.get("dataset_domain")
    if not isinstance(domains, (list, tuple)) or len(domains) != batch_size:
        raise ValueError("normalized agent geometry requires dataset_domain with one entry per sample")
    if any(domain != "real" for domain in domains):
        raise ValueError("agent-box safety losses require real geometry and reject counterfactual placeholders")
    for name in ("geometry_present", "future_agent_geometry_valid"):
        value = metadata.get(name)
        if not torch.is_tensor(value) or value.dtype != torch.bool or value.shape != (batch_size,):
            raise ValueError(f"normalized agent geometry requires metadata[{name!r}] bool [B]")
        if not bool(value.all().item()):
            raise ValueError(f"agent-box safety losses require metadata[{name!r}]=true for every sample")
    truncated = metadata.get("agent_geometry_truncated")
    if torch.is_tensor(truncated):
        valid_truncation = truncated.dtype == torch.bool and truncated.shape == (batch_size,)
        any_truncated = valid_truncation and bool(truncated.any().item())
    elif isinstance(truncated, (list, tuple)):
        valid_truncation = len(truncated) == batch_size and all(type(value) is bool for value in truncated)
        any_truncated = valid_truncation and any(truncated)
    else:
        valid_truncation = False
        any_truncated = False
    if not valid_truncation or any_truncated:
        raise ValueError("agent-box safety losses require explicit agent_geometry_truncated=false")
    expected_lists = {
        "geometry_source": "logged_nuscenes_gt",
        "geometry_coordinate_frame": "per_frame_ego",
    }
    for name, expected in expected_lists.items():
        value = metadata.get(name)
        if not isinstance(value, (list, tuple)) or len(value) != batch_size or any(item != expected for item in value):
            raise ValueError(f"agent-box safety losses require metadata[{name!r}]={expected!r}")


def extract_future_agent_boxes(
    sample: Any,
    *,
    future_start_idx: int,
    num_poses: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice future agent boxes/mask from a world-model collate ``sample``.

    The navsim/b2d collate tuple carries ``agent_boxes`` at index 7 and
    ``agent_mask`` at index 8 — per-frame ego-coordinate boxes
    ``[x, y, z, length, width, height, heading]`` at the **same reduced-frame
    cadence as ``states``**, so they are frame-aligned with ``gt_trajectory``.
    Returns ``(agent_boxes_future [B, num_poses, A, 7], agent_mask_future
    [B, num_poses, A])`` on ``device``.

    Fail-loud: raises if the sample lacks agent boxes, the tensors have the wrong
    shape, or the future window does not cover ``num_poses`` frames — never a
    silent skip or fabricated boxes (per repo convention). Note that real boxes
    require the dataset to be built with ``load_agent_annotations=True`` (the
    default); otherwise the collate emits all-zero boxes and collision risk would
    be silently zero.
    """
    if not isinstance(sample, (tuple, list)) or len(sample) <= _COLLATE_AGENT_MASK_IDX:
        raise ValueError(
            "reward_selector collision needs agent boxes at "
            f"sample[{_COLLATE_AGENT_BOXES_IDX}]/sample[{_COLLATE_AGENT_MASK_IDX}]; got sample of "
            f"type {type(sample).__name__} length {len(sample) if hasattr(sample, '__len__') else 'n/a'} "
            "(this data source does not provide agent annotations — fail-loud)."
        )
    agent_boxes = sample[_COLLATE_AGENT_BOXES_IDX]
    agent_mask = sample[_COLLATE_AGENT_MASK_IDX]
    if not torch.is_tensor(agent_boxes) or agent_boxes.ndim != 4 or agent_boxes.shape[-1] != 7:
        raise ValueError(
            f"agent_boxes (sample[{_COLLATE_AGENT_BOXES_IDX}]) must be [B, T, A, 7]; got "
            f"{tuple(agent_boxes.shape) if torch.is_tensor(agent_boxes) else type(agent_boxes).__name__}"
        )
    if not torch.is_tensor(agent_mask) or tuple(agent_mask.shape) != tuple(agent_boxes.shape[:3]):
        raise ValueError(
            f"agent_mask (sample[{_COLLATE_AGENT_MASK_IDX}]) must be {tuple(agent_boxes.shape[:3])}; got "
            f"{tuple(agent_mask.shape) if torch.is_tensor(agent_mask) else type(agent_mask).__name__}"
        )
    _validate_normalized_real_geometry_metadata(sample, batch_size=int(agent_boxes.shape[0]))
    metadata = sample[-1] if sample and isinstance(sample[-1], Mapping) else None
    if metadata is not None and "geometry_present" in metadata:
        if agent_boxes.shape[2] > 256:
            raise ValueError(f"normalized NavSim agent capacity exceeds 256: {agent_boxes.shape[2]}")
        valid_boxes = agent_boxes[agent_mask.bool()]
        if not torch.isfinite(valid_boxes).all() or (valid_boxes[:, 3:6] <= 0.0).any():
            raise ValueError("normalized real agent boxes must be finite with positive dimensions")

    start = int(future_start_idx)
    end = start + int(num_poses)
    if start < 0 or agent_boxes.shape[1] < end:
        raise ValueError(
            f"agent boxes temporal dim {agent_boxes.shape[1]} does not cover the planner future "
            f"window [{start}:{end}] (num_poses={num_poses}); cannot align to planner poses (fail-loud)."
        )

    boxes = agent_boxes[:, start:end].to(device=device, dtype=torch.float32, non_blocking=True)
    mask = agent_mask[:, start:end].to(device=device, non_blocking=True).bool()
    return boxes, mask


def maybe_compute_planner_mode_scores(
    config: TrainingConfig,
    pred_trajs: torch.Tensor,
    gt_traj: torch.Tensor,
    *,
    sample: Any = None,
    future_start_idx: Optional[int] = None,
    num_poses: Optional[int] = None,
    agent_boxes: Optional[torch.Tensor] = None,
    agent_mask: Optional[torch.Tensor] = None,
    z_hat_modes: Optional[torch.Tensor] = None,
    h_target_future: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Reward-based per-mode scores for the planner WTA selector (W4D world-model selector).

    Returns ``mode_reward`` ``[B, K]`` (higher = better) when ``reward_selector``
    is enabled and there is more than one mode, otherwise ``None`` (the caller
    then falls back to the legacy trajectory-L2 winner).

    Primary signal is the **world-model latent match**: the caller supplies
    ``z_hat_modes`` (per-mode predictor rollout, see ``rollout_selector_latents``)
    and ``h_target_future`` (real future latent), required when
    ``world_model_weight != 0``. ``collision_weight != 0`` extracts future agent
    boxes from the collate ``sample`` (needs ``sample``/``future_start_idx``/
    ``num_poses``). Comfort uses ``1 / fps`` (planner raw-frame cadence).
    Fail-loud: a non-zero component weight whose inputs are missing raises.
    """
    rs = config.reward_selector
    if not rs.enabled or pred_trajs.shape[1] <= 1:
        return None

    if rs.world_model_weight != 0.0 and (z_hat_modes is None or h_target_future is None):
        raise ValueError(
            "reward_selector.world_model_weight != 0 requires z_hat_modes + h_target_future "
            "(caller computes the predictor rollout + sliced real future latent via "
            "rollout_selector_latents); none were provided (fail-loud)."
        )

    if rs.collision_weight != 0.0 and agent_boxes is None:
        if sample is None or future_start_idx is None or num_poses is None:
            raise ValueError(
                "reward_selector.collision_weight != 0 requires the collate `sample` plus "
                "`future_start_idx`/`num_poses` to extract future agent boxes (fail-loud)."
            )
        agent_boxes, agent_mask = extract_future_agent_boxes(
            sample, future_start_idx=int(future_start_idx), num_poses=int(num_poses), device=pred_trajs.device
        )

    # Lazy import to keep this lightweight module free of the heavy losses package.
    from app.vjepa_cowa_world_model.losses.reward_selector import compute_mode_rewards

    timestep_sec = 1.0 / max(float(config.data.fps), 1.0)
    label_config = make_reward_label_config(config, timestep_sec=timestep_sec)
    return compute_mode_rewards(
        pred_trajs,
        gt_traj,
        world_model_weight=rs.world_model_weight,
        trajectory_error_weight=rs.trajectory_error_weight,
        comfort_weight=rs.comfort_weight,
        collision_weight=rs.collision_weight,
        offroad_weight=rs.offroad_weight,
        timestep_sec=timestep_sec,
        label_config=label_config,
        z_hat_modes=z_hat_modes,
        h_target_future=h_target_future,
        normalize_reps=bool(config.loss.normalize_reps),
        agent_boxes=agent_boxes,
        agent_mask=agent_mask,
    )["mode_reward"]


def rollout_selector_latents(
    *,
    predictor: torch.nn.Module,
    z_context: torch.Tensor,
    pred_trajs: torch.Tensor,
    h_target: torch.Tensor,
    config: TrainingConfig,
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    dt: float,
    num_observed_steps: int,
    predictor_frame_stride: int,
    actions: torch.Tensor,
    states: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll out the world model per candidate trajectory and slice the real future
    latent to match — the inputs to the world-model latent selector.

    Returns ``(z_hat_modes [B, K, num_future*tpf, D], h_target_future [B, num_future*tpf, D])``.
    Runs under ``no_grad`` and detaches both (the selector is a *selection* signal,
    not a training signal for the world model — doc §12).

    ``h_target`` is the full real future latent ``[B, num_time_steps*tpf, D]`` from
    the (frozen) target encoder; the matching future tokens start at
    ``num_observed_steps * tokens_per_frame``.
    """
    # predictor_frame_stride is handled inside rollout_predictor_modes; the
    # h_target offset below (num_obs * tokens_per_frame) is in encoder-timeline
    # token space, so it is frame-stride-agnostic — the exact slice the production
    # JEPA loss uses to align predictor output vs target latent.
    stride = int(predictor_frame_stride)

    # Lazy import to avoid a heavy import cycle at module load.
    from app.vjepa_cowa_world_model.training.runtimes.refinement_runtime import rollout_predictor_modes

    with torch.no_grad():
        z_hat = rollout_predictor_modes(
            predictor=predictor,
            z_context=z_context,
            future_trajs=pred_trajs,
            actions=actions,
            states=states,
            driving_command=driving_command,
            ego_dynamics=ego_dynamics,
            config=config,
            tokens_per_frame=int(tokens_per_frame),
            runtime_normalize_reps=bool(runtime_normalize_reps),
            dt=float(dt),
            predictor_observed_steps=int(num_observed_steps),
            predictor_frame_stride=stride,
        )  # [B, K, num_future*tpf, D]
    z_hat = z_hat.detach()

    offset = int(num_observed_steps) * int(tokens_per_frame)
    n = int(z_hat.shape[2])
    if h_target.ndim != 3 or h_target.shape[0] != z_hat.shape[0] or h_target.shape[-1] != z_hat.shape[-1]:
        raise ValueError(f"h_target {tuple(h_target.shape)} must be [B={z_hat.shape[0]}, T*tpf, D={z_hat.shape[-1]}]")
    if h_target.shape[1] < offset + n:
        raise ValueError(
            f"h_target tokens {h_target.shape[1]} < future window offset {offset} + {n}; "
            "cannot align rollout to the real future latent (fail-loud)."
        )
    h_future = h_target[:, offset : offset + n].detach()
    return z_hat, h_future


def make_reward_label_config(config: TrainingConfig, *, timestep_sec: float) -> RewardLabelConfig:
    """Create label config from ``TrainingConfig.reward`` and the active predictor timestep."""
    reward = config.reward
    return RewardLabelConfig(
        timestep_sec=float(timestep_sec),
        horizon_seconds=tuple(int(sec) for sec in reward.horizon_seconds),
        near_miss_distance=float(reward.near_miss_distance),
        near_miss_weight=float(reward.near_miss_weight),
        comfort_weight=float(reward.comfort_weight),
    )


def align_temporal_to_timeline(
    tensor: torch.Tensor,
    *,
    frame_stride: int,
    num_time_steps: int,
) -> torch.Tensor:
    """Align a raw temporal tensor to predictor timeline steps.

    For strided main encoders this mirrors ``build_predictor_timeline_inputs``:
    timeline step ``k`` is anchored at raw frame ``frame_stride - 1 + k * frame_stride``.
    If the raw clip is short, the last available anchored frame is repeated.
    """
    if tensor.ndim < 2:
        raise ValueError(f"Expected temporal tensor with shape [B, T, ...], got {tuple(tensor.shape)}")
    frame_stride = int(frame_stride)
    num_time_steps = int(num_time_steps)
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive")

    raw_steps = int(tensor.shape[1])
    if raw_steps <= 0:
        raise ValueError("Cannot align an empty temporal tensor")

    if frame_stride == 1:
        aligned = tensor[:, :num_time_steps]
    else:
        indices = torch.arange(frame_stride - 1, raw_steps, frame_stride, device=tensor.device, dtype=torch.long)
        if indices.numel() == 0:
            indices = torch.tensor([raw_steps - 1], device=tensor.device, dtype=torch.long)
        aligned = tensor.index_select(1, indices[:num_time_steps])

    if aligned.shape[1] == num_time_steps:
        return aligned
    if aligned.shape[1] > num_time_steps:
        return aligned[:, :num_time_steps]

    pad_count = num_time_steps - aligned.shape[1]
    pad_frame = aligned[:, -1:].expand(-1, pad_count, *aligned.shape[2:])
    return torch.cat([aligned, pad_frame], dim=1)
