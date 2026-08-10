"""Planner-aware trajectory selection utilities for visualization."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _pairwise_trajectory_distance(pred_trajs: np.ndarray) -> np.ndarray:
    if pred_trajs.ndim != 3:
        raise ValueError(f"Expected pred_trajs shape [K, T, 3], got {list(pred_trajs.shape)}")

    xy = pred_trajs[..., :2]
    diff = xy[:, None, :, :] - xy[None, :, :, :]
    pairwise = np.linalg.norm(diff, axis=-1).mean(axis=-1)
    return pairwise


def select_display_trajectory_index(
    pred_trajs: np.ndarray,
    confidences: Optional[np.ndarray],
    planner_type: str = "transformer",
) -> int:
    if pred_trajs.ndim != 3:
        raise ValueError(f"Expected pred_trajs shape [K, T, 3], got {list(pred_trajs.shape)}")
    if pred_trajs.shape[0] == 0:
        raise ValueError("pred_trajs must contain at least one trajectory")

    if planner_type == "diffusion":
        pairwise = _pairwise_trajectory_distance(pred_trajs)
        return int(np.argmin(pairwise.sum(axis=1)))

    if confidences is None:
        raise ValueError("confidences are required for transformer planner visualization")
    if confidences.ndim != 1:
        raise ValueError(f"Expected confidences shape [K], got {list(confidences.shape)}")
    return int(np.argmax(confidences))


def _compute_ade_np(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    displacement = np.linalg.norm(pred_traj[..., :2] - gt_traj[..., :2], axis=-1)
    return float(displacement.mean())


def _compute_fde_np(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    return float(np.linalg.norm(pred_traj[-1, :2] - gt_traj[-1, :2]))


def _compute_minade_minfde_k_np(pred_trajs: np.ndarray, gt_traj: np.ndarray) -> Dict[str, float]:
    displacement = np.linalg.norm(pred_trajs[..., :2] - gt_traj[None, :, :2], axis=-1)
    ade_k = displacement.mean(axis=-1)
    fde_k = displacement[:, -1]
    return {
        "minade_k": float(ade_k.min()),
        "minfde_k": float(fde_k.min()),
    }


def compute_visualization_metrics_np(
    pred_trajs: Optional[np.ndarray],
    confidences: Optional[np.ndarray],
    gt_traj: Optional[np.ndarray],
    planner_type: str = "transformer",
) -> Dict[str, Optional[float]]:
    if pred_trajs is None or gt_traj is None:
        return {
            "ade": None,
            "fde": None,
            "minade_k": None,
            "minfde_k": None,
            "display_idx": None,
        }

    if gt_traj.shape[0] == 0 or pred_trajs.shape[-2] == 0:
        return {
            "ade": None,
            "fde": None,
            "minade_k": None,
            "minfde_k": None,
            "display_idx": None,
        }

    display_idx = select_display_trajectory_index(pred_trajs, confidences, planner_type=planner_type)
    display_traj = pred_trajs[display_idx]
    min_metrics = _compute_minade_minfde_k_np(pred_trajs, gt_traj)

    return {
        "ade": _compute_ade_np(display_traj, gt_traj),
        "fde": _compute_fde_np(display_traj, gt_traj),
        "minade_k": min_metrics["minade_k"],
        "minfde_k": min_metrics["minfde_k"],
        "display_idx": display_idx,
    }


def get_display_mode_info(
    planner_type: str,
    confidences: np.ndarray,
    display_idx: int,
) -> Dict[str, Optional[str]]:
    if planner_type == "diffusion":
        return {
            "title": "Pred repr (green), GT (blue), samples (light green)",
            "confidence_text": f"Samples: {len(confidences)}, display: medoid",
        }

    conf_probs = np.exp(confidences - confidences.max())
    conf_probs = conf_probs / max(float(conf_probs.sum()), 1e-8)
    return {
        "title": "Pred (green), GT (blue)",
        "confidence_text": f"Best mode confidence: {conf_probs[display_idx]:.2%}",
    }
