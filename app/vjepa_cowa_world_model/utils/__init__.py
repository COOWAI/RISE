# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
工具函数模块

包含:
- prepare_status_feature, get_status_dim: 状态特征提取
- prepare_seg_features: 分割特征提取
- select_best_trajectory: 轨迹选择
- visualize_trajectory, save_training_visualization: 可视化
- compute_world4drive_l2_metrics, compute_collision_rate: 开环评估指标
- compute_collision_rate_legacy: 向后兼容的碰撞率计算
"""

from .metrics import (  # noqa: F401
    WORLD4DRIVE_REPORTED_SECONDS,
    compute_collision_rate,
    compute_collision_rate_legacy,
    compute_l2_per_timestep,
    compute_world4drive_l2_metrics,
    populate_point_l2_horizons,
    populate_world4drive_collision_horizons,
    populate_world4drive_l2_horizons,
)
from .seg_features import prepare_seg_features
from .status_features import (
    build_future_gt_trajectory_from_states,
    build_observed_action_trajectory_history,
    get_status_dim,
    mask_future_actions,
    prepare_inference_consistent_states,
    prepare_inference_consistent_status_vector,
    prepare_status_feature,
    resolve_effective_planner_status_dim,
    resolve_planner_status_dim,
    resolve_planner_use_drive_command,
)
from .trajectory import confidence_weighted_trajectory, select_best_trajectory, trajectory_to_control_action
from .visualization import save_training_visualization, visualize_multimodal_trajectory, visualize_trajectory

# 为 prepare_inference_consistent_status_vector 创建别名（实际上调用 _prepare_drive_command_7）
# from .status_features import _prepare_drive_command_7 as prepare_inference_consistent_status_vector

__all__ = [
    "prepare_status_feature",
    "get_status_dim",
    "build_future_gt_trajectory_from_states",
    "build_observed_action_trajectory_history",
    "prepare_inference_consistent_states",
    "prepare_inference_consistent_status_vector",
    "resolve_effective_planner_status_dim",
    "resolve_planner_use_drive_command",
    "resolve_planner_status_dim",
    "mask_future_actions",
    "prepare_seg_features",
    "select_best_trajectory",
    "confidence_weighted_trajectory",
    "trajectory_to_control_action",
    "visualize_trajectory",
    "visualize_multimodal_trajectory",
    "save_training_visualization",
    "compute_l2_per_timestep",
    "compute_world4drive_l2_metrics",
    "populate_world4drive_collision_horizons",
    "populate_world4drive_l2_horizons",
    "populate_point_l2_horizons",
    "WORLD4DRIVE_REPORTED_SECONDS",
    "compute_collision_rate",
    "compute_collision_rate_legacy",
]
