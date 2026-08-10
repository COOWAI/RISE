"""Safety reward labels for predictor-token reward model training."""

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch

from app.vjepa_cowa_world_model.utils.metrics import EGO_LENGTH, EGO_LENGTH_OFFSET, EGO_WIDTH


@dataclass
class RewardLabelConfig:
    """Configuration for offline safety reward labels.

    Labels are in ``[0, 1]`` and use the convention selected for this task:
    larger values mean safer futures.
    """

    timestep_sec: float = 0.5
    horizon_seconds: Tuple[int, ...] = (1, 2, 3, 4)
    near_miss_distance: float = 2.0
    near_miss_weight: float = 0.75
    comfort_weight: float = 0.20
    accel_threshold: float = 6.0
    accel_margin: float = 4.0
    yaw_rate_threshold: float = 1.0
    yaw_rate_margin: float = 1.0
    jerk_threshold: float = 8.0
    jerk_margin: float = 8.0


@dataclass
class SafetyRewardLabels:
    """Reward labels and diagnostics for one batch."""

    scalar: torch.Tensor
    horizons: torch.Tensor
    step_rewards: torch.Tensor
    step_risk: torch.Tensor
    components: Dict[str, torch.Tensor] = field(default_factory=dict)


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _safe_margin_risk(value: torch.Tensor, threshold: float, margin: float) -> torch.Tensor:
    if margin <= 0:
        raise ValueError("margin must be positive")
    return torch.clamp((value - float(threshold)) / float(margin), min=0.0, max=1.0)


def _agent_clearance_to_ego(agent_boxes: torch.Tensor, agent_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Approximate OBB clearance between each agent and the ego footprint.

    Parameters
    ----------
    agent_boxes : [B, T, A, 7]
        Per-frame ego-coordinate boxes ``[x, y, z, length, width, height, heading]``.
    agent_mask : [B, T, A]
        True for valid agents.

    Returns
    -------
    min_clearance : [B, T]
        Distance in metres. Zero means overlap under the inflated-rectangle test.
    collision : [B, T]
        True if any valid agent overlaps the ego footprint.
    """
    if agent_boxes.ndim != 4 or agent_boxes.shape[-1] != 7:
        raise ValueError(f"agent_boxes must have shape [B, T, A, 7], got {tuple(agent_boxes.shape)}")
    if agent_mask.shape != agent_boxes.shape[:3]:
        raise ValueError(f"agent_mask must have shape {tuple(agent_boxes.shape[:3])}, got {tuple(agent_mask.shape)}")

    x = agent_boxes[..., 0]
    y = agent_boxes[..., 1]
    length = torch.clamp(agent_boxes[..., 3], min=0.0)
    width = torch.clamp(agent_boxes[..., 4], min=0.0)
    heading = agent_boxes[..., 6]

    # Compare the ego footprint center, including ST-P3/VAD's forward offset,
    # against each agent OBB inflated by half the ego footprint.
    rel_x = float(EGO_LENGTH_OFFSET) - x
    rel_y = -y
    cos_h = torch.cos(-heading)
    sin_h = torch.sin(-heading)
    local_x = cos_h * rel_x - sin_h * rel_y
    local_y = sin_h * rel_x + cos_h * rel_y

    half_x = 0.5 * length + 0.5 * float(EGO_LENGTH)
    half_y = 0.5 * width + 0.5 * float(EGO_WIDTH)
    dx = torch.abs(local_x) - half_x
    dy = torch.abs(local_y) - half_y

    collision_per_agent = (dx <= 0.0) & (dy <= 0.0) & agent_mask
    outside_dx = torch.clamp(dx, min=0.0)
    outside_dy = torch.clamp(dy, min=0.0)
    clearance_per_agent = torch.sqrt(outside_dx.square() + outside_dy.square())
    clearance_per_agent = torch.where(
        agent_mask,
        clearance_per_agent,
        torch.full_like(clearance_per_agent, float("inf")),
    )

    min_clearance = torch.amin(clearance_per_agent, dim=-1)
    min_clearance = torch.where(torch.isfinite(min_clearance), min_clearance, torch.full_like(min_clearance, 1e6))
    collision = torch.any(collision_per_agent, dim=-1)
    return min_clearance, collision


def _comfort_risk(states: torch.Tensor, num_observed_frames: int, future_steps: int, timestep_sec: float, config):
    speed = states[..., 6]
    yaw = states[..., 5]
    dt = float(timestep_sec)
    if dt <= 0:
        raise ValueError("timestep_sec must be positive")

    accel = torch.diff(speed, dim=1) / dt
    yaw_rate = _wrap_angle(torch.diff(yaw, dim=1)) / dt
    start = max(0, int(num_observed_frames) - 1)
    accel_future = accel[:, start : start + future_steps]
    yaw_rate_future = yaw_rate[:, start : start + future_steps]

    if accel_future.shape[1] < future_steps:
        pad = future_steps - accel_future.shape[1]
        accel_future = torch.nn.functional.pad(accel_future, (0, pad))
        yaw_rate_future = torch.nn.functional.pad(yaw_rate_future, (0, pad))

    if accel_future.shape[1] > 1:
        jerk = torch.diff(accel_future, dim=1, prepend=accel_future[:, :1]) / dt
    else:
        jerk = torch.zeros_like(accel_future)

    accel_risk = _safe_margin_risk(torch.abs(accel_future), config.accel_threshold, config.accel_margin)
    yaw_risk = _safe_margin_risk(torch.abs(yaw_rate_future), config.yaw_rate_threshold, config.yaw_rate_margin)
    jerk_risk = _safe_margin_risk(torch.abs(jerk), config.jerk_threshold, config.jerk_margin)
    return torch.maximum(torch.maximum(accel_risk, yaw_risk), jerk_risk)


def compute_safety_reward_labels(
    states: torch.Tensor,
    agent_boxes: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    num_observed_frames: int,
    config: RewardLabelConfig = None,
) -> SafetyRewardLabels:
    """Compute scalar and horizon safety rewards from NavSim-style batches.

    Parameters
    ----------
    states : [B, T, 7]
        Ego states.
    agent_boxes : [B, T, A, 7]
        Dynamic-agent boxes in each frame's ego coordinate system.
    agent_mask : [B, T, A]
        Valid-agent mask.
    num_observed_frames : int
        Number of context frames. Future labels start at this index.
    config : RewardLabelConfig
        Label hyperparameters.

    Returns
    -------
    SafetyRewardLabels
        ``scalar`` is ``[B]`` and ``horizons`` is ``[B, num_horizons]``.
    """
    cfg = config or RewardLabelConfig()
    if states.ndim != 3 or states.shape[-1] != 7:
        raise ValueError(f"states must have shape [B, T, 7], got {tuple(states.shape)}")
    if not 1 <= int(num_observed_frames) < states.shape[1]:
        raise ValueError(f"num_observed_frames must be in [1, T-1], got {num_observed_frames} for T={states.shape[1]}")

    future_start = int(num_observed_frames)
    future_steps = states.shape[1] - future_start
    future_boxes = agent_boxes[:, future_start:]
    future_mask = agent_mask[:, future_start:]

    min_clearance, collision = _agent_clearance_to_ego(future_boxes, future_mask)
    near_risk = torch.clamp((float(cfg.near_miss_distance) - min_clearance) / float(cfg.near_miss_distance), 0.0, 1.0)
    near_risk = torch.where(collision, torch.zeros_like(near_risk), near_risk)

    comfort_risk = _comfort_risk(states, future_start, future_steps, cfg.timestep_sec, cfg)
    step_risk = torch.maximum(
        collision.to(states.dtype),
        torch.maximum(float(cfg.near_miss_weight) * near_risk, float(cfg.comfort_weight) * comfort_risk),
    )
    step_risk = torch.clamp(step_risk, 0.0, 1.0)
    step_rewards = 1.0 - step_risk

    horizons = []
    for seconds in cfg.horizon_seconds:
        horizon_steps = int(round(float(seconds) / float(cfg.timestep_sec)))
        horizon_steps = min(max(horizon_steps, 1), future_steps)
        horizon_risk = torch.amax(step_risk[:, :horizon_steps], dim=1)
        horizons.append(1.0 - horizon_risk)
    horizon_rewards = torch.stack(horizons, dim=1)
    scalar = horizon_rewards.mean(dim=1)

    return SafetyRewardLabels(
        scalar=scalar,
        horizons=horizon_rewards,
        step_rewards=step_rewards,
        step_risk=step_risk,
        components={
            "collision": collision,
            "min_clearance": min_clearance,
            "near_miss": near_risk,
            "comfort": comfort_risk,
        },
    )
