# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
可视化工具函数

包含轨迹可视化和分割训练可视化。
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_trajectory(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    output_dir: str,
    epoch: int,
    itr: int,
    limit: int = 5,
    file_prefix: Optional[str] = None,
):
    """
    可视化预测轨迹和真实轨迹

    Args:
        pred_traj: 预测轨迹 [B, num_poses, 3] (x, y, yaw)
        gt_traj: 真实轨迹 [B, num_poses, 3] (x, y, yaw)
        output_dir: 输出目录
        epoch: 当前 epoch
        itr: 当前 iteration
        limit: 最多可视化多少个样本
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_traj = pred_traj.detach().cpu().float().numpy()
    gt_traj = gt_traj.detach().cpu().float().numpy()

    batch_size = min(pred_traj.shape[0], limit)
    num_poses = pred_traj.shape[1]
    saved_paths = []

    for i in range(batch_size):
        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制起点 (0, 0) - 车辆当前位置
        ax.scatter(0, 0, c="red", s=100, marker="*", label="Ego (Start)", zorder=5)

        # 绘制真实轨迹
        gt_x = gt_traj[i, :, 0]
        gt_y = gt_traj[i, :, 1]
        gt_yaw = gt_traj[i, :, 2]

        # 绘制轨迹线
        ax.plot(gt_x, gt_y, "g-", linewidth=2, label=f"Ground Truth ({num_poses} pts)", alpha=0.8)
        ax.scatter(gt_x, gt_y, c="green", s=30, alpha=0.6)

        # 绘制真实轨迹的方向箭头
        arrow_step = max(1, len(gt_x) // 4)  # 最多绘制4个箭头
        for j in range(0, len(gt_x), arrow_step):
            dx = 0.3 * np.cos(gt_yaw[j])
            dy = 0.3 * np.sin(gt_yaw[j])
            ax.arrow(gt_x[j], gt_y[j], dx, dy, head_width=0.15, head_length=0.1, fc="green", ec="green", alpha=0.7)

        # 绘制预测轨迹
        pred_x = pred_traj[i, :, 0]
        pred_y = pred_traj[i, :, 1]
        pred_yaw = pred_traj[i, :, 2]

        # 检查预测值是否有异常
        has_nan = np.any(np.isnan(pred_x)) or np.any(np.isnan(pred_y))
        has_inf = np.any(np.isinf(pred_x)) or np.any(np.isinf(pred_y))
        max_abs = max(np.abs(pred_x).max(), np.abs(pred_y).max())

        ax.plot(pred_x, pred_y, "b-", linewidth=2, label=f"Prediction ({num_poses} pts)", alpha=0.8)
        ax.scatter(pred_x, pred_y, c="blue", s=30, alpha=0.6)

        # 绘制预测轨迹的方向箭头
        for j in range(0, len(pred_x), arrow_step):
            dx = 0.3 * np.cos(pred_yaw[j])
            dy = 0.3 * np.sin(pred_yaw[j])
            ax.arrow(pred_x[j], pred_y[j], dx, dy, head_width=0.15, head_length=0.1, fc="blue", ec="blue", alpha=0.7)

        # 计算 L2 误差
        l2_error = np.sqrt(((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2).mean())
        yaw_error = np.abs(pred_yaw - gt_yaw).mean()

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        # 标题包含调试信息
        debug_info = f"NaN: {has_nan}, Inf: {has_inf}, MaxAbs: {max_abs:.2f}"
        ax.set_title(
            f"Trajectory Visualization\nEpoch {epoch}, Iter {itr}, Sample {i}\n"
            f"L2 Error: {l2_error:.3f}m, Yaw Error: {yaw_error:.3f}rad\n"
            f"[{debug_info}]"
        )
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        # 动态调整坐标轴范围（基于GT轨迹，避免异常预测值影响）
        all_x = np.concatenate([gt_x, [0]])
        all_y = np.concatenate([gt_y, [0]])

        # 如果预测值正常，也纳入范围计算
        if max_abs < 100:  # 阈值：100米
            all_x = np.concatenate([all_x, pred_x])
            all_y = np.concatenate([all_y, pred_y])

        margin = max(1.0, (all_x.max() - all_x.min()) * 0.1)
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        filename = f"{file_prefix}_s{i}.png" if file_prefix else f"traj_E{epoch}_I{itr}_s{i}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths


def visualize_multimodal_trajectory(
    pred_trajs: torch.Tensor,
    pred_conf: torch.Tensor,
    gt_traj: torch.Tensor,
    output_dir: str,
    epoch: int,
    batch_idx: int,
    limit: int = 4,
    file_prefix: Optional[str] = None,
):
    """
    可视化多模态预测轨迹和真实轨迹

    所有 K 条模态轨迹用不同颜色绘制（半透明），best trajectory 加粗高亮，
    GT 用绿色实线。标题显示 best 模态的 ADE/FDE 和各模态置信度。

    Args:
        pred_trajs: 预测轨迹 [B, K, num_poses, 3] (x, y, yaw)
        pred_conf: 置信度 [B, K]
        gt_traj: 真实轨迹 [B, num_poses, 3] (x, y, yaw)
        output_dir: 输出目录
        epoch: 当前 epoch
        batch_idx: 当前 batch index
        limit: 最多可视化多少个样本
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_trajs = pred_trajs.detach().cpu().float().numpy()
    pred_conf = pred_conf.detach().cpu().float().numpy()
    gt_traj = gt_traj.detach().cpu().float().numpy()

    batch_size = min(pred_trajs.shape[0], limit)
    K = pred_trajs.shape[1]
    num_poses = pred_trajs.shape[2]
    cmap = plt.cm.get_cmap("tab10", max(K, 10))
    saved_paths = []

    for i in range(batch_size):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Ego start
        ax.scatter(0, 0, c="red", s=100, marker="*", label="Ego (Start)", zorder=5)

        # GT trajectory
        gt_x = gt_traj[i, :, 0]
        gt_y = gt_traj[i, :, 1]
        gt_yaw = gt_traj[i, :, 2]
        ax.plot(gt_x, gt_y, "g-", linewidth=2.5, label=f"GT ({num_poses} pts)", alpha=0.9, zorder=4)
        ax.scatter(gt_x, gt_y, c="green", s=30, alpha=0.6, zorder=4)
        arrow_step = max(1, len(gt_x) // 4)
        for j in range(0, len(gt_x), arrow_step):
            dx = 0.3 * np.cos(gt_yaw[j])
            dy = 0.3 * np.sin(gt_yaw[j])
            ax.arrow(gt_x[j], gt_y[j], dx, dy, head_width=0.15, head_length=0.1, fc="green", ec="green", alpha=0.7)

        # Best mode index
        best_k = int(np.argmax(pred_conf[i]))

        # Draw all K modes
        for k in range(K):
            px = pred_trajs[i, k, :, 0]
            py = pred_trajs[i, k, :, 1]
            conf_val = pred_conf[i, k]
            color = cmap(k)

            if k == best_k:
                # Best mode: bold highlight
                ax.plot(
                    px,
                    py,
                    color=color,
                    linewidth=3.0,
                    alpha=1.0,
                    label=f"Best M{k} (conf={conf_val:.3f})",
                    zorder=3,
                )
                ax.scatter(px, py, color=color, s=40, alpha=0.8, zorder=3)
            else:
                ax.plot(
                    px,
                    py,
                    color=color,
                    linewidth=1.5,
                    alpha=0.4,
                    label=f"M{k} (conf={conf_val:.3f})",
                    zorder=2,
                )
                ax.scatter(px, py, color=color, s=15, alpha=0.3, zorder=2)

        # Compute metrics for best mode
        best_px = pred_trajs[i, best_k, :, 0]
        best_py = pred_trajs[i, best_k, :, 1]
        ade = np.sqrt(((best_px - gt_x) ** 2 + (best_py - gt_y) ** 2)).mean()
        fde = np.sqrt((best_px[-1] - gt_x[-1]) ** 2 + (best_py[-1] - gt_y[-1]) ** 2)

        # minADE across all modes
        all_ade = []
        for k in range(K):
            kx = pred_trajs[i, k, :, 0]
            ky = pred_trajs[i, k, :, 1]
            all_ade.append(np.sqrt(((kx - gt_x) ** 2 + (ky - gt_y) ** 2)).mean())
        min_ade = min(all_ade)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(
            f"Val Multimodal Trajectory — Epoch {epoch}, Batch {batch_idx}, Sample {i}\n"
            f"Best ADE: {ade:.3f}m, FDE: {fde:.3f}m, minADE@{K}: {min_ade:.3f}m"
        )
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        # Axis range based on GT + best mode
        all_x = np.concatenate([gt_x, best_px, [0]])
        all_y = np.concatenate([gt_y, best_py, [0]])
        margin = max(1.0, (all_x.max() - all_x.min()) * 0.1)
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        filename = f"{file_prefix}_s{i}.png" if file_prefix else f"val_traj_E{epoch}_B{batch_idx}_s{i}.png"
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths


def visualize_history_vs_gt(
    history_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    output_dir: str,
    file_prefix: str,
    limit: int = 4,
    show_arrows: bool = False,
):
    """可视化观测历史轨迹与未来 GT 轨迹。"""
    os.makedirs(output_dir, exist_ok=True)

    history_traj = history_traj.detach().cpu().float().numpy()
    gt_traj = gt_traj.detach().cpu().float().numpy()

    batch_size = min(history_traj.shape[0], gt_traj.shape[0], limit)
    saved_paths = []

    for i in range(batch_size):
        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(0, 0, c="red", s=100, marker="*", label="Ego (Current)", zorder=5)

        history_x = history_traj[i, :, 0]
        history_y = history_traj[i, :, 1]
        ax.plot(history_x, history_y, color="tab:orange", linewidth=2.0, linestyle="--", label="Observed History")
        ax.scatter(history_x, history_y, color="tab:orange", s=35, alpha=0.8)

        future_x = gt_traj[i, :, 0]
        future_y = gt_traj[i, :, 1]
        ax.plot(future_x, future_y, color="green", linewidth=2.2, label="Future GT")
        ax.scatter(future_x, future_y, color="green", s=35, alpha=0.8)

        for hist_idx, (x_val, y_val) in enumerate(zip(history_x, history_y)):
            ax.text(x_val, y_val, f"H{hist_idx}")
        for fut_idx, (x_val, y_val) in enumerate(zip(future_x, future_y)):
            ax.text(x_val, y_val, f"F{fut_idx}")

        if show_arrows and history_traj.shape[-1] >= 3:
            for x_val, y_val, yaw_val in history_traj[i, :, :3]:
                dx = 0.3 * np.cos(yaw_val)
                dy = 0.3 * np.sin(yaw_val)
                ax.arrow(x_val, y_val, dx, dy, head_width=0.12, head_length=0.08, fc="tab:orange", ec="tab:orange")
        if show_arrows and gt_traj.shape[-1] >= 3:
            for x_val, y_val, yaw_val in gt_traj[i, :, :3]:
                dx = 0.3 * np.cos(yaw_val)
                dy = 0.3 * np.sin(yaw_val)
                ax.arrow(x_val, y_val, dx, dy, head_width=0.12, head_length=0.08, fc="green", ec="green")

        all_x = np.concatenate([history_x, future_x, np.array([0.0])])
        all_y = np.concatenate([history_y, future_y, np.array([0.0])])
        margin = max(1.0, max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) * 0.1)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Observed History vs Future GT")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
        ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
        ax.set_ylim(all_y.min() - margin, all_y.max() + margin)

        save_path = os.path.join(output_dir, f"{file_prefix}_s{i}.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        saved_paths.append(save_path)

    return saved_paths


def save_training_visualization(pred_results, vis_meta, output_dir, epoch, itr, limit=5):
    """
    统一的可视化绘图函数（用于分割任务）

    Args:
        pred_results: 预测结果列表
        vis_meta: 可视化元数据
        output_dir: 输出目录
        epoch: 当前 epoch
        itr: 当前 iteration
        limit: 最多可视化多少个样本
    """
    os.makedirs(output_dir, exist_ok=True)
    # 取最后一层输出，通常是 [N, num_classes, H, W]
    pred_raw_batch = pred_results[-1]

    count = 0
    for i, meta in enumerate(vis_meta):
        if count >= limit:
            break

        # 1. 处理预测 Mask
        pred_raw = pred_raw_batch[i]  # [num_classes, H, W]

        # 简单处理：Sigmoid + Argmax (假设你的逻辑是这样)
        mask_cat = torch.argmax(pred_raw.sigmoid(), dim=0)  # 注意 dim 可能是 0 或 1，取决于你的 shape

        mask_cat = mask_cat.cpu().numpy().astype(np.uint8)
        mask_cat[mask_cat == 0] = 0  # 背景
        mask_cat[mask_cat == 1] = 255  # 前景 (假设二分类)

        # 2. 处理 GT
        gt_tensor = meta["gt_mask"]
        if gt_tensor.dim() == 3:
            gt_merged_mask, _ = torch.max(gt_tensor, dim=0)
        else:
            gt_merged_mask = gt_tensor
        gt_vis = gt_merged_mask.numpy()

        # 3. 处理原图
        img_tensor = meta["img_tensor"]
        img_vis = img_tensor.permute(1, 2, 0).numpy()
        img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-6)

        # 4. 绘图
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_vis)
        axes[0].set_title(f"E{epoch}_I{itr} Frame {meta['t_frame']}")
        axes[0].axis("off")

        axes[1].imshow(gt_vis, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(mask_cat, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")

        save_path = os.path.join(output_dir, f"vis_E{epoch}_I{itr}_s{i}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        count += 1
