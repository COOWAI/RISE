# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""Reward-based mode selector for multi-modal trajectory planning.

This implements the W4D **world-model selector** (doc §1): the winner is NOT the
trajectory closest to GT, but the candidate whose **world-model-imagined next-frame
latent** best matches the **real** next-frame latent. The *primary* signal is the
per-mode latent match (predictor rollout ``ẑ^k`` vs frozen target-encoder
``h_target``); trajectory-error / comfort / collision are weighted *shaping* terms
(doc §6, à la ``reward = -world_model_loss - λ·traj_err - λ·collision - λ·jerk``).

Composite reward (convention: **higher = better**)::

    reward_k =   w_world   * world_model_reward_k   (cosine - recon vs real future latent; PRIMARY)
               - w_traj    * trajectory_error_k     (imitation accuracy vs GT; shaping)
               - w_comfort * comfort_risk_k         (accel / yaw-rate / jerk of the candidate; shaping)
               - w_coll    * collision_risk_k       (counterfactual, non-reactive; shaping)
               - w_offroad * offroad_risk_k         (requires a drivable map; unwired → raises)

The returned ``mode_reward`` is passed to ``wta_loss*`` as the ``mode_scores``
argument so the winner becomes ``mode_reward.argmax(dim=1)`` and (for v2/v3) the
soft confidence / aWTA weights follow the reward instead of ``-dist_xy``. At
inference ``select_best_trajectory`` keeps using the confidence head, which the
WTA confidence loss has trained to predict the reward-winner — so the selection
propagates to inference *through* the confidence head (mirrors W4D ``wm_cls_head``).

The latent rollout is a **selection** signal, so callers run it under ``no_grad``
/ detach it (doc §12: do not train the world model via the selector reward).

Fail-loud (per repo convention): a component whose weight is non-zero but whose
inputs are missing raises rather than being silently dropped. world_model needs
``z_hat_modes`` + ``h_target_future``; collision needs future agent boxes; offroad
needs a drivable map (unwired → raises).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.training.reward_labels import RewardLabelConfig, _safe_margin_risk, _wrap_angle
from app.vjepa_cowa_world_model.utils.metrics import EGO_LENGTH, EGO_LENGTH_OFFSET, EGO_WIDTH


def _validate_traj_shapes(pred_trajs: torch.Tensor, gt_traj: torch.Tensor) -> None:
    if pred_trajs.ndim != 4 or pred_trajs.shape[-1] != 3:
        raise ValueError(f"pred_trajs must be [B, K, P, 3], got {tuple(pred_trajs.shape)}")
    if gt_traj.ndim != 3 or gt_traj.shape[-1] != 3:
        raise ValueError(f"gt_traj must be [B, P, 3], got {tuple(gt_traj.shape)}")
    if gt_traj.shape[0] != pred_trajs.shape[0] or gt_traj.shape[1] != pred_trajs.shape[2]:
        raise ValueError(
            "pred_trajs and gt_traj must share batch and pose dims; "
            f"got pred_trajs={tuple(pred_trajs.shape)}, gt_traj={tuple(gt_traj.shape)}"
        )


def compute_trajectory_error(pred_trajs: torch.Tensor, gt_traj: torch.Tensor) -> torch.Tensor:
    """Per-mode average displacement error (xy only) to GT.

    Parameters
    ----------
    pred_trajs : [B, K, P, 3]
    gt_traj    : [B, P, 3]

    Returns
    -------
    ade : [B, K]  (metres, higher = worse)
    """
    _validate_traj_shapes(pred_trajs, gt_traj)
    gt_expanded = gt_traj.unsqueeze(1)  # [B, 1, P, 3]
    return torch.norm(pred_trajs[..., :2] - gt_expanded[..., :2], dim=-1).mean(dim=-1)  # [B, K]


def compute_comfort_risk(
    pred_trajs: torch.Tensor,
    *,
    timestep_sec: float,
    config: RewardLabelConfig,
) -> torch.Tensor:
    """Per-mode comfort risk in ``[0, 1]`` derived from the candidate kinematics.

    Speed/accel/jerk and yaw-rate are computed from the candidate poses
    themselves (``x=forward, y=left`` ego frame, cumulative positions starting
    from the last-observed origin at ``(0, 0, 0)``). Risk is the per-mode maximum
    over future steps of ``max(accel_risk, yaw_rate_risk, jerk_risk)`` using the
    same safe-margin thresholds as the offline safety labels.

    Parameters
    ----------
    pred_trajs   : [B, K, P, 3]  (x, y, yaw)
    timestep_sec : seconds between consecutive poses (e.g. frame_stride / fps)
    config       : threshold/margin hyperparameters

    Returns
    -------
    comfort_risk : [B, K]
    """
    if pred_trajs.ndim != 4 or pred_trajs.shape[-1] != 3:
        raise ValueError(f"pred_trajs must be [B, K, P, 3], got {tuple(pred_trajs.shape)}")
    dt = float(timestep_sec)
    if dt <= 0:
        raise ValueError(f"timestep_sec must be positive, got {timestep_sec}")

    B, K, P, _ = pred_trajs.shape
    # Prepend the last-observed origin (ego at (0,0) heading 0) so the first
    # future step has a well-defined speed / yaw-rate.
    origin = torch.zeros(B, K, 1, 3, device=pred_trajs.device, dtype=pred_trajs.dtype)
    full = torch.cat([origin, pred_trajs], dim=2)  # [B, K, P+1, 3]

    xy = full[..., :2]
    yaw = full[..., 2]

    step_disp = torch.norm(xy[:, :, 1:, :] - xy[:, :, :-1, :], dim=-1)  # [B, K, P]
    speed = step_disp / dt  # [B, K, P]
    accel = torch.diff(speed, dim=2, prepend=speed[:, :, :1]) / dt  # [B, K, P]
    jerk = torch.diff(accel, dim=2, prepend=accel[:, :, :1]) / dt  # [B, K, P]
    yaw_rate = _wrap_angle(yaw[:, :, 1:] - yaw[:, :, :-1]) / dt  # [B, K, P]

    accel_risk = _safe_margin_risk(torch.abs(accel), config.accel_threshold, config.accel_margin)
    yaw_risk = _safe_margin_risk(torch.abs(yaw_rate), config.yaw_rate_threshold, config.yaw_rate_margin)
    jerk_risk = _safe_margin_risk(torch.abs(jerk), config.jerk_threshold, config.jerk_margin)

    step_risk = torch.maximum(torch.maximum(accel_risk, yaw_risk), jerk_risk)  # [B, K, P]
    return step_risk.amax(dim=2)  # [B, K]


def _obb_collision_per_agent(agent_boxes: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
    """Exact four-axis SAT overlap of each agent OBB with the candidate ego OBB."""

    x = agent_boxes[..., 0]
    y = agent_boxes[..., 1]
    agent_half_length = 0.5 * torch.clamp(agent_boxes[..., 3], min=0.0)
    agent_half_width = 0.5 * torch.clamp(agent_boxes[..., 4], min=0.0)
    heading = agent_boxes[..., 6]

    ego_half_length = 0.5 * float(EGO_LENGTH)
    ego_half_width = 0.5 * float(EGO_WIDTH)
    tx = x - float(EGO_LENGTH_OFFSET)
    ty = y
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    abs_cos = torch.abs(cos_heading)
    abs_sin = torch.abs(sin_heading)

    # Separating axes are ego forward/left and agent forward/left. The ego is
    # axis-aligned in the candidate frame, while agent heading is relative to it.
    overlap_ego_x = torch.abs(tx) <= (ego_half_length + agent_half_length * abs_cos + agent_half_width * abs_sin)
    overlap_ego_y = torch.abs(ty) <= (ego_half_width + agent_half_length * abs_sin + agent_half_width * abs_cos)
    translation_agent_x = tx * cos_heading + ty * sin_heading
    translation_agent_y = -tx * sin_heading + ty * cos_heading
    overlap_agent_x = torch.abs(translation_agent_x) <= (
        agent_half_length + ego_half_length * abs_cos + ego_half_width * abs_sin
    )
    overlap_agent_y = torch.abs(translation_agent_y) <= (
        agent_half_width + ego_half_length * abs_sin + ego_half_width * abs_cos
    )
    return overlap_ego_x & overlap_ego_y & overlap_agent_x & overlap_agent_y & agent_mask


def _point_to_segments_distance(
    points: torch.Tensor,
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
) -> torch.Tensor:
    """Pairwise point-to-segment distances ``[..., num_points, num_segments]``."""

    segment_delta = segment_ends - segment_starts
    point_delta = points.unsqueeze(-2) - segment_starts.unsqueeze(-3)
    denominator = segment_delta.square().sum(dim=-1).clamp_min(1e-12).unsqueeze(-2)
    projection = (point_delta * segment_delta.unsqueeze(-3)).sum(dim=-1) / denominator
    projection = torch.clamp(projection, min=0.0, max=1.0)
    closest = segment_starts.unsqueeze(-3) + projection.unsqueeze(-1) * segment_delta.unsqueeze(-3)
    return torch.linalg.vector_norm(points.unsqueeze(-2) - closest, dim=-1)


def _obb_clearance_to_ego(
    agent_boxes: torch.Tensor,
    agent_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact OBB collision and Euclidean clearance to the candidate ego OBB."""

    half_length = 0.5 * torch.clamp(agent_boxes[..., 3], min=0.0)
    half_width = 0.5 * torch.clamp(agent_boxes[..., 4], min=0.0)
    heading = agent_boxes[..., 6]
    cos_heading = torch.cos(heading)
    sin_heading = torch.sin(heading)
    forward = torch.stack([cos_heading, sin_heading], dim=-1)
    left = torch.stack([-sin_heading, cos_heading], dim=-1)
    signs = agent_boxes.new_tensor(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    centers = agent_boxes[..., :2].unsqueeze(-2)
    sign_shape = (*([1] * half_length.ndim), 4, 1)
    forward_sign = signs[:, 0].reshape(sign_shape)
    left_sign = signs[:, 1].reshape(sign_shape)
    agent_corners = (
        centers
        + forward_sign * half_length.unsqueeze(-1).unsqueeze(-1) * forward.unsqueeze(-2)
        + left_sign * half_width.unsqueeze(-1).unsqueeze(-1) * left.unsqueeze(-2)
    )

    ego_half_extents = agent_boxes.new_tensor([0.5 * float(EGO_LENGTH), 0.5 * float(EGO_WIDTH)])
    ego_center = agent_boxes.new_tensor([float(EGO_LENGTH_OFFSET), 0.0])
    ego_corners = ego_center + signs * ego_half_extents
    ego_corners = ego_corners.view(*([1] * (agent_corners.ndim - 2)), 4, 2).expand_as(agent_corners)

    agent_ends = torch.roll(agent_corners, shifts=-1, dims=-2)
    ego_ends = torch.roll(ego_corners, shifts=-1, dims=-2)
    agent_to_ego = _point_to_segments_distance(agent_corners, ego_corners, ego_ends)
    ego_to_agent = _point_to_segments_distance(ego_corners, agent_corners, agent_ends)
    clearance_per_agent = torch.minimum(
        agent_to_ego.amin(dim=(-2, -1)),
        ego_to_agent.amin(dim=(-2, -1)),
    )
    collision_per_agent = _obb_collision_per_agent(agent_boxes, agent_mask)
    clearance_per_agent = torch.where(
        collision_per_agent,
        torch.zeros_like(clearance_per_agent),
        clearance_per_agent,
    )
    clearance_per_agent = torch.where(
        agent_mask,
        clearance_per_agent,
        torch.full_like(clearance_per_agent, float("inf")),
    )
    min_clearance = clearance_per_agent.amin(dim=-1)
    min_clearance = torch.where(
        torch.isfinite(min_clearance),
        min_clearance,
        torch.full_like(min_clearance, 1e6),
    )
    return min_clearance, collision_per_agent.any(dim=-1)


def compute_collision_components(
    pred_trajs: torch.Tensor,
    gt_traj: torch.Tensor,
    agent_boxes: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    config: Optional[RewardLabelConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-mode collision and near-miss components from all agents.

    Agent boxes are logged in *each future frame's real-ego coordinate system*.
    For a counterfactual candidate trajectory the ego would be elsewhere, so each
    box is re-expressed in the candidate's ego frame at step ``f`` by composing
    two rigid transforms that only use **real, logged data**:

    1. real-ego-frame-``f`` → current-ego frame, using the GT future pose at ``f``
       (``gt_traj[:, f]`` — where the ego actually was);
    2. current-ego frame → candidate-ego-frame-``f``, using the candidate pose at
       ``f`` (``pred_trajs[:, k, f]``).

    Agents keep their logged motion (non-reactive) — the standard open-loop /
    NavSim-PDM convention. Only the ego is counterfactual; nothing is fabricated.

    Parameters
    ----------
    pred_trajs  : [B, K, P, 3]  candidate poses (x, y, yaw), current-ego frame
    gt_traj     : [B, P, 3]     GT future poses (x, y, yaw), current-ego frame
    agent_boxes : [B, P, A, 7]  future-sliced per-frame-ego boxes [x,y,z,l,w,h,heading]
    agent_mask  : [B, P, A]     valid-agent mask
    config      : near-miss hyperparameters

    Returns
    -------
    collision_any  : [B, K]  true when any future pose overlaps an agent
    near_miss_risk : [B, K]  maximum unweighted non-collision clearance risk
    """
    _validate_traj_shapes(pred_trajs, gt_traj)
    if agent_boxes.ndim != 4 or agent_boxes.shape[-1] != 7:
        raise ValueError(f"agent_boxes must be [B, P, A, 7], got {tuple(agent_boxes.shape)}")
    B, K, P, _ = pred_trajs.shape
    if agent_boxes.shape[0] != B or agent_boxes.shape[1] != P:
        raise ValueError(
            "agent_boxes must be future-sliced and aligned to the predicted poses "
            f"([B={B}, P={P}, A, 7]); got {tuple(agent_boxes.shape)}"
        )
    if agent_mask.shape != agent_boxes.shape[:3]:
        raise ValueError(f"agent_mask must be {tuple(agent_boxes.shape[:3])}, got {tuple(agent_mask.shape)}")
    if agent_boxes.shape[2] < 1:
        raise ValueError("agent_boxes agent dimension must be non-empty")
    if pred_trajs.device != gt_traj.device or pred_trajs.device != agent_boxes.device:
        raise ValueError("pred_trajs, gt_traj, and agent_boxes must be on the same device")
    if agent_mask.device != pred_trajs.device:
        raise ValueError("agent_mask must be on the same device as pred_trajs")

    cfg = config or RewardLabelConfig()
    near_dist = float(cfg.near_miss_distance)
    if not math.isfinite(near_dist) or near_dist <= 0.0:
        raise ValueError(f"near_miss_distance must be finite and positive, got {cfg.near_miss_distance}")

    boxes = agent_boxes.to(dtype=pred_trajs.dtype)
    A = boxes.shape[2]

    # Logged agent state in each future frame's real-ego coordinates.
    xa = boxes[..., 0].unsqueeze(1)  # [B, 1, P, A]
    ya = boxes[..., 1].unsqueeze(1)
    za = boxes[..., 2].unsqueeze(1)
    length = boxes[..., 3].unsqueeze(1)
    width = boxes[..., 4].unsqueeze(1)
    height = boxes[..., 5].unsqueeze(1)
    ha = boxes[..., 6].unsqueeze(1)  # [B, 1, P, A]

    # GT pose (real-ego-frame-f → current-ego frame).
    gx = gt_traj[..., 0].unsqueeze(1).unsqueeze(-1)  # [B, 1, P, 1]
    gy = gt_traj[..., 1].unsqueeze(1).unsqueeze(-1)
    gyaw = gt_traj[..., 2].unsqueeze(1).unsqueeze(-1)
    cos_g = torch.cos(gyaw)
    sin_g = torch.sin(gyaw)
    p_cur_x = cos_g * xa - sin_g * ya + gx  # [B, 1, P, A]
    p_cur_y = sin_g * xa + cos_g * ya + gy
    h_cur = ha + gyaw

    # Candidate pose (current-ego frame → candidate-ego-frame-f).
    cx = pred_trajs[..., 0].unsqueeze(-1)  # [B, K, P, 1]
    cy = pred_trajs[..., 1].unsqueeze(-1)
    cyaw = pred_trajs[..., 2].unsqueeze(-1)
    cos_c = torch.cos(cyaw)
    sin_c = torch.sin(cyaw)
    rel_x = p_cur_x - cx  # broadcast [B, 1, P, A] - [B, K, P, 1] -> [B, K, P, A]
    rel_y = p_cur_y - cy
    p_cand_x = cos_c * rel_x + sin_c * rel_y  # R(-cyaw)
    p_cand_y = -sin_c * rel_x + cos_c * rel_y
    h_cand = h_cur - cyaw  # [B, K, P, A]

    za_e = za.expand(B, K, P, A)
    length_e = length.expand(B, K, P, A)
    width_e = width.expand(B, K, P, A)
    height_e = height.expand(B, K, P, A)
    cand_boxes = torch.stack(
        [p_cand_x, p_cand_y, za_e, length_e, width_e, height_e, h_cand], dim=-1
    )  # [B, K, P, A, 7]

    mask_e = agent_mask.unsqueeze(1).expand(B, K, P, A).reshape(B * K, P, A)
    flat_boxes = cand_boxes.reshape(B * K, P, A, 7)
    min_clearance, collision = _obb_clearance_to_ego(flat_boxes, mask_e)  # [B*K, P]

    near_risk = torch.clamp((near_dist - min_clearance) / near_dist, 0.0, 1.0)
    near_risk = torch.where(collision, torch.zeros_like(near_risk), near_risk)
    collision_any = collision.reshape(B, K, P).any(dim=2)
    near_miss_risk = near_risk.reshape(B, K, P).amax(dim=2)
    return collision_any, near_miss_risk


def compute_collision_risk(
    pred_trajs: torch.Tensor,
    gt_traj: torch.Tensor,
    agent_boxes: torch.Tensor,
    agent_mask: torch.Tensor,
    *,
    config: RewardLabelConfig,
) -> torch.Tensor:
    """Per-mode counterfactual collision/near-miss risk in ``[0, 1]``.

    This compatibility wrapper preserves the established weighted risk while
    exposing the binary collision and unweighted near-miss terms separately via
    :func:`compute_collision_components`.
    """
    collision_any, near_miss_risk = compute_collision_components(
        pred_trajs,
        gt_traj,
        agent_boxes,
        agent_mask,
        config=config,
    )
    risk = torch.maximum(
        collision_any.to(dtype=near_miss_risk.dtype),
        float(config.near_miss_weight) * near_miss_risk,
    )
    return torch.clamp(risk, 0.0, 1.0)


def compute_world_model_latent_reward(
    z_hat_modes: torch.Tensor,
    h_target_future: torch.Tensor,
    *,
    normalize_reps: bool = True,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Per-mode **world-model latent-matching** reward (higher = better).

    This is the *core* W4D selector signal (doc §1): for each candidate
    trajectory, the world model (predictor) imagines the next-frame latent
    ``ẑ^k``; the best mode is the one whose imagined future latent is closest to
    the **real** future latent ``h_target`` (from the frozen target encoder).

    Parameters
    ----------
    z_hat_modes     : [B, K, N, D]  per-mode predicted future latents (predictor rollout)
    h_target_future : [B, N, D]     real future latent, sliced to the same future tokens
    normalize_reps  : LayerNorm both sides over D before recon/cosine (match the
                      JEPA-loss convention so we compare like with like)

    Returns
    -------
    dict with ``recon`` ``[B,K]`` (MSE, lower=better), ``cosine`` ``[B,K]``
    (mean per-token cosine, higher=better) and ``reward`` ``[B,K] = cosine - recon``.
    """
    if z_hat_modes.ndim != 4:
        raise ValueError(f"z_hat_modes must be [B, K, N, D], got {tuple(z_hat_modes.shape)}")
    if h_target_future.ndim != 3:
        raise ValueError(f"h_target_future must be [B, N, D], got {tuple(h_target_future.shape)}")
    B, K, N, D = z_hat_modes.shape
    if tuple(h_target_future.shape) != (B, N, D):
        raise ValueError(
            f"h_target_future {tuple(h_target_future.shape)} must align with z_hat_modes "
            f"[B={B}, N={N}, D={D}]; slice h_target to the future tokens the rollout covers."
        )
    z = z_hat_modes
    h = h_target_future.unsqueeze(1)  # [B, 1, N, D]
    if normalize_reps:
        z = F.layer_norm(z, (D,))
        h = F.layer_norm(h, (D,))
    recon = ((z - h) ** 2).mean(dim=(2, 3))  # [B, K]
    cosine = F.cosine_similarity(z, h, dim=-1, eps=eps).mean(dim=2)  # [B, K]
    return {"recon": recon, "cosine": cosine, "reward": cosine - recon}


def compute_mode_rewards(
    pred_trajs: torch.Tensor,
    gt_traj: torch.Tensor,
    *,
    world_model_weight: float,
    trajectory_error_weight: float,
    comfort_weight: float,
    collision_weight: float,
    offroad_weight: float,
    timestep_sec: float,
    label_config: RewardLabelConfig,
    z_hat_modes: Optional[torch.Tensor] = None,
    h_target_future: Optional[torch.Tensor] = None,
    normalize_reps: bool = True,
    agent_boxes: Optional[torch.Tensor] = None,
    agent_mask: Optional[torch.Tensor] = None,
    drivable_map: Optional[object] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the per-mode composite reward (higher = better) and the winner.

    Primary signal is the **world-model latent match** (``world_model_weight`` on
    ``compute_world_model_latent_reward``); trajectory-error / comfort / collision
    are weighted *shaping* terms (doc §6). Each weighted component is fail-loud: a
    non-zero weight with missing inputs raises rather than silently dropping the
    term. Returns ``mode_reward`` ``[B, K]``, ``winner_idx`` ``[B]`` and
    per-component tensors.
    """
    _validate_traj_shapes(pred_trajs, gt_traj)
    B, K = pred_trajs.shape[0], pred_trajs.shape[1]
    components: Dict[str, torch.Tensor] = {}

    reward = torch.zeros(B, K, device=pred_trajs.device, dtype=pred_trajs.dtype)

    if float(world_model_weight) != 0.0:
        if z_hat_modes is None or h_target_future is None:
            raise ValueError(
                "world_model_weight != 0 requires z_hat_modes [B,K,N,D] (predictor rollout per mode) "
                "and h_target_future [B,N,D] (real future latent); none were provided (fail-loud)."
            )
        wm = compute_world_model_latent_reward(z_hat_modes, h_target_future, normalize_reps=normalize_reps)
        components["world_model_recon"] = wm["recon"]
        components["world_model_cosine"] = wm["cosine"]
        components["world_model_reward"] = wm["reward"]
        reward = reward + float(world_model_weight) * wm["reward"].to(device=reward.device, dtype=reward.dtype)

    if float(trajectory_error_weight) != 0.0:
        traj_err = compute_trajectory_error(pred_trajs, gt_traj)
        components["trajectory_error"] = traj_err
        reward = reward - float(trajectory_error_weight) * traj_err

    if float(comfort_weight) != 0.0:
        comfort_risk = compute_comfort_risk(pred_trajs, timestep_sec=timestep_sec, config=label_config)
        components["comfort_risk"] = comfort_risk
        reward = reward - float(comfort_weight) * comfort_risk

    if float(collision_weight) != 0.0:
        if agent_boxes is None or agent_mask is None:
            raise ValueError(
                "collision_weight != 0 requires agent_boxes and agent_mask "
                "(future-sliced, per-frame-ego); none were provided (fail-loud)."
            )
        collision_risk = compute_collision_risk(pred_trajs, gt_traj, agent_boxes, agent_mask, config=label_config)
        components["collision_risk"] = collision_risk
        reward = reward - float(collision_weight) * collision_risk

    if float(offroad_weight) != 0.0:
        # No drivable-area / map input is plumbed in this repo. Fabricating an
        # offroad label would violate the fail-loud convention, so refuse.
        raise NotImplementedError(
            "offroad_weight != 0 requires a drivable-area map, which is not "
            "available in the current data pipeline. Plumb a drivable map and "
            "implement compute_offroad_risk before enabling this term."
        )

    if not components:
        raise ValueError("compute_mode_rewards: all component weights are zero — nothing to select on.")

    winner_idx = reward.argmax(dim=1)
    return {"mode_reward": reward, "winner_idx": winner_idx, **components}
