# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

"""
轨迹可视化脚本 - 将预测轨迹投影到前视相机图像上

支持 train_command.py / train_giant_first.py 的推理配置:
- predictor_inference_consistent: 推理一致模式
- predictor_no_aux_input: predictor 不使用辅助输入
- num_observed_frames: 观测帧数

"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

# 添加项目路径
VJEPA_ROOT = os.environ.get("VJEPA_ROOT", os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, VJEPA_ROOT)
# torchcv 本地包路径 (seg_head2.py 通过 training/models.py 间接依赖)
sys.path.insert(0, os.path.join(VJEPA_ROOT, "..", "ddddetection_torchcv"))

from app.vjepa_cowa_world_model.utils.viz.config_resolution import load_visualization_config  # noqa: E402
from app.vjepa_cowa_world_model.utils.viz.data import (  # noqa: E402
    compute_actions_3d,
    get_future_pose_start_index,
    load_sample_data,
    load_trajectory_list,
    preprocess_clip,
    slice_future_horizon,
)
from app.vjepa_cowa_world_model.utils.viz.draw import (  # noqa: E402
    draw_all_trajectories_side_by_side,
    save_color_space_check_image,
)
from app.vjepa_cowa_world_model.utils.viz.geometry import reref_trajectory_ego  # noqa: E402
from app.vjepa_cowa_world_model.utils.viz.inference import run_inference  # noqa: E402
from app.vjepa_cowa_world_model.utils.viz.models import load_models  # noqa: E402
from app.vjepa_cowa_world_model.utils.viz.selection import compute_visualization_metrics_np  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def compute_visualization_metrics(pred_trajs, confidences, gt_traj, planner_type="transformer"):
    """
    基于当前帧剩余的 future horizon 重新计算展示用指标。
    """
    return compute_visualization_metrics_np(
        pred_trajs,
        confidences,
        gt_traj,
        planner_type=planner_type,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trajectory predictions on images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to txt file containing trajectory directories"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file (yaml)")
    parser.add_argument("--output_dir", type=str, default="./vis_results", help="Output directory for visualizations")
    parser.add_argument("--camera", type=str, default="CAM_FRONT", help="Camera view to use")
    parser.add_argument("--num_trajectories", type=int, default=5, help="Number of trajectories to visualize")
    parser.add_argument("--stride", type=int, default=20, help="Sliding window stride (frames)")
    parser.add_argument("--mode", type=str, default="both", choices=["image", "video", "both"], help="Output mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    # Fail-loud: the config must come from the checkpoint sidecar (params-pretrain.yaml) or an
    # explicit --config, never a fabricated default — visualization has to use the exact config
    # the checkpoint was trained with (see load_visualization_config).
    config = load_visualization_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 加载模型
    logger.info("Loading models...")
    encoder, predictor, planner, cfg = load_models(args.checkpoint, config, device)
    dtype = cfg.dtype
    logger.info(f"Using device: {device}, dtype: {dtype}")
    logger.info(f"Visualization config source: {config.get('__config_source__', 'unknown')}")
    # 从 txt 文件加载轨迹列表
    logger.info(f"Loading trajectory list from {args.data_path}...")
    trajectory_list = load_trajectory_list(args.data_path)
    # 限制轨迹数量
    num_trajectories = min(args.num_trajectories, len(trajectory_list))
    trajectory_list = trajectory_list[:num_trajectories]
    logger.info(f"Processing {num_trajectories} trajectories...")
    # 模型窗口配置 (从 cfg 读取)
    num_target = cfg.data.num_target_frames
    crop_size = cfg.data.crop_size
    output_fps = cfg.data.fps
    stride = args.stride
    # 推理相关配置
    predictor_inference_consistent = cfg.train.predictor_inference_consistent
    num_observed_frames = cfg.train.num_observed_frames
    use_extrinsics = cfg.model.use_extrinsics
    planner_type = cfg.planner.planner_type
    logger.info(
        f"  Config: "
        f"  planner_type={planner_type}, "
        f"  predictor_inference_consistent={predictor_inference_consistent}, "
        f"  num_observed_frames={num_observed_frames}, "
        f"  predictor_no_aux_input={cfg.train.predictor_no_aux_input}, "
        f"  use_z_context={cfg.planner.use_z_context}, "
        f"  use_temporal={cfg.planner.use_temporal}"
    )
    # ==================== 滑动窗口遍历每条完整轨迹 ====================
    # 全局指标累积（跨所有轨迹）
    global_metrics = {"ade": [], "fde": [], "minade_k": [], "minfde_k": []}
    for traj_idx, traj_dir in enumerate(trajectory_list):
        logger.info(f"Processing trajectory {traj_idx + 1}/{num_trajectories}: {traj_dir}")
        try:
            # 加载完整轨迹数据
            sample_data = load_sample_data(traj_dir, camera=args.camera)
            all_frames = sample_data["frames"]  # RGB 格式 (decord 输出)
            all_states = sample_data["states"]
            camera_params = sample_data["camera_params"]
            T_total = all_frames.shape[0]
            H, W = all_frames.shape[1], all_frames.shape[2]
            traj_name = os.path.basename(traj_dir)
            if T_total < num_target:
                logger.warning(f"Not enough frames ({T_total} < {num_target}) in {traj_dir}, skipping")
                continue
            # 计算所有滑动窗口起始位置
            window_starts = list(range(0, T_total - num_target + 1, stride))
            if not window_starts or window_starts[-1] != T_total - num_target:
                window_starts.append(T_total - num_target)
            logger.info(
                f"  Total frames: {T_total}, window size: {num_target}, "
                f"stride: {stride}, num windows: {len(window_starts)}"
            )
            # 为每帧缓存最近一次的预测结果
            # 当多个滑动窗口覆盖同一帧时，保留“最新覆盖该帧”的窗口预测，
            # 这样可视化更贴近当前时刻的观测上下文。
            frame_predictions = [None] * T_total
            for wi, start in enumerate(window_starts):
                end = start + num_target
                window_frames = all_frames[start:end]
                window_states = all_states[start:end]
                window_actions = compute_actions_3d(window_states)
                # 计算 extrinsics (如果使用)
                if use_extrinsics:
                    window_extrinsics = sample_data["extrinsics"][start:end]
                else:
                    window_extrinsics = None
                # 预处理 + 推理
                frames_tensor = preprocess_clip(window_frames, crop_size)
                states_tensor = torch.from_numpy(window_states).float().unsqueeze(0)
                actions_tensor = torch.from_numpy(window_actions).float().unsqueeze(0)
                if use_extrinsics and window_extrinsics is not None:
                    extrinsics_tensor = torch.from_numpy(window_extrinsics).float().unsqueeze(0)
                else:
                    extrinsics_tensor = None
                pred_trajs, confidences, gt_traj = run_inference(
                    encoder,
                    predictor,
                    planner,
                    frames_tensor,
                    states_tensor,
                    actions_tensor,
                    extrinsics_tensor,
                    device,
                    dtype,
                    cfg,
                )
                pred_np = pred_trajs[0].cpu().float().detach().numpy()
                conf_np = confidences[0].cpu().float().detach().numpy()
                gt_np = gt_traj[0].cpu().float().detach().numpy()
                # 将该窗口的预测结果分配给窗口覆盖的所有帧。
                # 由于窗口按时间顺序遍历，这里直接覆盖即可保留“最新覆盖该帧”的预测。
                for t in range(start, end):
                    frame_predictions[t] = (pred_np, conf_np, gt_np, start)
                if (wi + 1) % 20 == 0 or wi == len(window_starts) - 1:
                    logger.info(f"    Window {wi + 1}/{len(window_starts)} done")
            # ==================== 整个 clip 的平均评测指标 ====================
            future_start_offset = num_observed_frames if predictor_inference_consistent else 1
            clip_metrics = {"ade": [], "fde": [], "minade_k": [], "minfde_k": []}
            for t in range(T_total):
                if frame_predictions[t] is not None:
                    pred_np_m, conf_np_m, gt_np_m, win_start_m = frame_predictions[t]
                    pose_start_idx_m = get_future_pose_start_index(
                        win_start_m, t, future_start_offset, gt_np_m.shape[0]
                    )
                    pred_f_m, gt_f_m = slice_future_horizon(pred_np_m, gt_np_m, pose_start_idx_m)
                    m = compute_visualization_metrics(
                        pred_f_m,
                        conf_np_m,
                        gt_f_m,
                        planner_type=planner_type,
                    )
                    for k in clip_metrics:
                        if m[k] is not None:
                            clip_metrics[k].append(m[k])

            clip_avg = {}
            for k in clip_metrics:
                vals = clip_metrics[k]
                clip_avg[k] = float(np.mean(vals)) if vals else None

            def _fmt(name, val):
                return f"{name}={val:.4f}" if val is not None else f"{name}=N/A"

            logger.info(
                f"  Clip avg metrics: "
                f"{_fmt('ADE', clip_avg['ade'])}, "
                f"{_fmt('FDE', clip_avg['fde'])}, "
                f"{_fmt('minADE@K', clip_avg['minade_k'])}, "
                f"{_fmt('minFDE@K', clip_avg['minfde_k'])}"
            )

            # 累积到全局指标
            for k in global_metrics:
                if clip_avg[k] is not None:
                    global_metrics[k].append(clip_avg[k])

            # 创建该轨迹的输出子目录
            traj_output_dir = os.path.join(args.output_dir, traj_name)
            os.makedirs(traj_output_dir, exist_ok=True)

            # 保存 clip 平均指标到文件
            metrics_path = os.path.join(traj_output_dir, "clip_metrics.json")
            with open(metrics_path, "w") as mf:
                json.dump(clip_avg, mf, indent=2)

            # 保存颜色空间对照图，方便肉眼确认输入帧是否为 RGB
            color_check_path = os.path.join(traj_output_dir, "color_space_check.png")
            save_color_space_check_image(all_frames[0], color_check_path)
            if args.mode in ["video", "both"]:
                # 生成完整轨迹视频
                video_path = os.path.join(traj_output_dir, "trajectory.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                # 实际并排帧的尺寸是 W + H (相机图宽 + BEV宽)
                combined_w = W + H
                # FFmpeg requires even dimensions
                vw = combined_w if combined_w % 2 == 0 else combined_w + 1
                vh = H if H % 2 == 0 else H + 1
                video_writer = cv2.VideoWriter(video_path, fourcc, output_fps, (vw, vh))
                resize_needed = vw != combined_w or vh != H

                for t in range(T_total):
                    bg_image = cv2.cvtColor(all_frames[t], cv2.COLOR_RGB2BGR)  # 显示时转 BGR
                    if frame_predictions[t] is not None:
                        pred_np, conf_np, gt_np, win_start = frame_predictions[t]
                        pose_start_idx = get_future_pose_start_index(win_start, t, future_start_offset, gt_np.shape[0])
                        pred_future, gt_future = slice_future_horizon(pred_np, gt_np, pose_start_idx)
                        metrics = compute_visualization_metrics(
                            pred_future,
                            conf_np,
                            gt_future,
                            planner_type=planner_type,
                        )
                        origin_state = all_states[win_start]
                        current_state = all_states[t]
                        pred_reref = reref_trajectory_ego(pred_future, origin_state, current_state)
                        gt_reref = reref_trajectory_ego(gt_future, origin_state, current_state)
                        vis_frame = draw_all_trajectories_side_by_side(
                            bg_image,
                            pred_reref,
                            conf_np,
                            gt_reref,
                            camera_params,
                            ade=metrics["ade"],
                            fde=metrics["fde"],
                            minade_k=metrics["minade_k"],
                            minfde_k=metrics["minfde_k"],
                            planner_type=planner_type,
                        )
                    else:
                        # 对于无预测的帧，仍然创建并排空视图
                        vis_frame = np.hstack([bg_image, np.zeros((H, H, 3), dtype=np.uint8)])
                    # 确保帧尺寸匹配VideoWriter配置
                    if resize_needed:
                        vis_frame = cv2.resize(vis_frame, (vw, vh))
                    # 帧信息（放在左侧相机视图上）
                    cv2.putText(
                        vis_frame, f"{traj_name}", (10, vh - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                    )
                    cv2.putText(
                        vis_frame,
                        f"Frame {t}/{T_total}",
                        (10, vh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    video_writer.write(vis_frame)  # vis_frame 已经是 BGR 格式
                video_writer.release()
                logger.info(f"  Saved: {video_path} ({T_total} frames)")
            # 保存若干关键帧图像
            if args.mode in ["image", "both"]:
                key_frame_indices = np.linspace(0, T_total - 1, min(5, T_total), dtype=int)
                for ki, t in enumerate(key_frame_indices):
                    bg_image = cv2.cvtColor(all_frames[t], cv2.COLOR_RGB2BGR)  # 显示时转 BGR
                    if frame_predictions[t] is not None:
                        pred_np, conf_np, gt_np, win_start = frame_predictions[t]
                        pose_start_idx = get_future_pose_start_index(win_start, t, future_start_offset, gt_np.shape[0])
                        pred_future, gt_future = slice_future_horizon(pred_np, gt_np, pose_start_idx)
                        metrics = compute_visualization_metrics(
                            pred_future,
                            conf_np,
                            gt_future,
                            planner_type=planner_type,
                        )
                        origin_state = all_states[win_start]
                        current_state = all_states[t]
                        pred_reref = reref_trajectory_ego(pred_future, origin_state, current_state)
                        gt_reref = reref_trajectory_ego(gt_future, origin_state, current_state)
                        vis_frame = draw_all_trajectories_side_by_side(
                            bg_image,
                            pred_reref,
                            conf_np,
                            gt_reref,
                            camera_params,
                            ade=metrics["ade"],
                            fde=metrics["fde"],
                            minade_k=metrics["minade_k"],
                            minfde_k=metrics["minfde_k"],
                            planner_type=planner_type,
                        )
                    else:
                        vis_frame = np.hstack([bg_image, np.zeros((H, H, 3), dtype=np.uint8)])
                    img_path = os.path.join(traj_output_dir, f"frame_{t:04d}.png")
                    cv2.imwrite(img_path, vis_frame)  # vis_frame 已经是 BGR 格式
        except Exception as e:
            logger.warning(f"Failed to process {traj_dir}: {e}")
            import traceback

            traceback.print_exc()
            continue
    # ==================== 全局平均指标汇总 ====================
    global_avg = {}
    for k in global_metrics:
        vals = global_metrics[k]
        global_avg[k] = float(np.mean(vals)) if vals else None

    def _fmt(name, val):
        return f"{name}={val:.4f}" if val is not None else f"{name}=N/A"

    logger.info(
        f"Global avg metrics across {num_trajectories} trajectories: "
        f"{_fmt('ADE', global_avg['ade'])}, "
        f"{_fmt('FDE', global_avg['fde'])}, "
        f"{_fmt('minADE@K', global_avg['minade_k'])}, "
        f"{_fmt('minFDE@K', global_avg['minfde_k'])}"
    )
    # 保存全局指标到文件
    global_metrics_path = os.path.join(args.output_dir, "global_metrics.json")
    with open(global_metrics_path, "w") as mf:
        json.dump(global_avg, mf, indent=2)
    logger.info(f"Global metrics saved to {global_metrics_path}")
    logger.info(f"Visualization complete! Processed {num_trajectories} trajectories, saved to {args.output_dir}")


if __name__ == "__main__":
    main()
