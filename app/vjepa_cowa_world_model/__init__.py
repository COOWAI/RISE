# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
vjepa_cowa_world_model - 重构后的 Planner 训练模块

目录结构:
├── models/
│   └── multimodal_planner.py    # 统一的 MultiModalTemporalPlanner
├── losses/
│   ├── wta_loss.py              # wta_loss, wta_loss_v2, wta_loss_v3
│   └── single_model_loss.py     # single_model_loss, l1_length_normalized_loss
├── utils/
│   ├── status_features.py       # prepare_status_feature, get_status_dim
│   ├── seg_features.py          # prepare_seg_features
│   ├── trajectory.py            # select_best_trajectory
│   └── visualization.py         # visualize_trajectory, save_training_visualization
├── training/
│   └── __init__.py              # (预留) 训练器基类
"""

from importlib import import_module

__all__ = [
    # Models
    "MultiModalTemporalPlanner",
    # Losses
    "wta_loss",
    "wta_loss_v2",
    "wta_loss_v3",
    "awta_temperature_schedule",
    "get_loss_function",
    "single_model_loss",
    "l1_length_normalized_loss",
    "gaussian_log_prob",
    "gaussian_entropy",
    "ppo_loss",
    # Utils
    "prepare_status_feature",
    "get_status_dim",
    "prepare_seg_features",
    "select_best_trajectory",
    "confidence_weighted_trajectory",
    "trajectory_to_control_action",
    "visualize_trajectory",
    "save_training_visualization",
]


_LOSS_EXPORTS = frozenset(
    {
        "awta_temperature_schedule",
        "gaussian_entropy",
        "gaussian_log_prob",
        "get_loss_function",
        "l1_length_normalized_loss",
        "ppo_loss",
        "single_model_loss",
        "wta_loss",
        "wta_loss_v2",
        "wta_loss_v3",
    }
)
_UTIL_EXPORTS = frozenset(
    {
        "confidence_weighted_trajectory",
        "get_status_dim",
        "prepare_seg_features",
        "prepare_status_feature",
        "save_training_visualization",
        "select_best_trajectory",
        "trajectory_to_control_action",
        "visualize_trajectory",
    }
)


def __getattr__(name: str) -> object:
    """Resolve and cache public exports without eagerly importing PyTorch modules."""

    if name in _LOSS_EXPORTS:
        module = import_module(".losses", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _UTIL_EXPORTS:
        module = import_module(".utils", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name == "MultiModalTemporalPlanner":
        module = import_module(".models", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
