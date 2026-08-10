# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""OpenCV (BGR) camera + BEV trajectory overlay rendering."""

import cv2
import numpy as np

from app.vjepa_cowa_world_model.utils.viz.geometry import transform_trajectory_to_image
from app.vjepa_cowa_world_model.utils.viz.selection import get_display_mode_info, select_display_trajectory_index


def draw_trajectory_on_image(image, pixels, valid_mask, color, thickness=2, draw_points=True, alpha=1.0):
    """
    在图像上绘制单条轨迹。

    注意:
        `image` 必须是 OpenCV 使用的 BGR 图像；`color` 也应按 BGR 传入。
    """
    image = image.copy()
    valid_pixels = pixels[valid_mask].astype(np.int32)
    if len(valid_pixels) < 2:
        return image
    cv2.polylines(image, [valid_pixels], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    if draw_points:
        for i, pt in enumerate(valid_pixels):
            radius = max(2, 5 - i // 2)
            cv2.circle(image, tuple(pt), radius, color, -1, lineType=cv2.LINE_AA)
    return image


def _draw_sample_trajectories_on_image(image, pred_trajs, camera_params, color=(120, 200, 120)):
    vis_image = image.copy()
    H, W = image.shape[:2]
    for sample_traj in pred_trajs:
        pred_pixels, pred_valid = transform_trajectory_to_image(sample_traj, camera_params, (H, W))
        vis_image = draw_trajectory_on_image(
            vis_image,
            pred_pixels,
            pred_valid,
            color,
            thickness=1,
            draw_points=False,
        )
    return cv2.addWeighted(image, 0.55, vis_image, 0.45, 0)


def _draw_sample_trajectories_on_bev(bev_canvas, pred_trajs, ego_to_pix, in_bounds, color=(120, 200, 120)):
    for sample_traj in pred_trajs:
        pred_pix = ego_to_pix(sample_traj[:, :2])
        pred_mask = in_bounds(pred_pix)
        if pred_mask.sum() >= 2:
            cv2.polylines(bev_canvas, [pred_pix[pred_mask]], False, color, 1, cv2.LINE_AA)


def _draw_visualization_overlay_text(vis_image, planner_type, confidences, display_idx, ade, fde, minade_k, minfde_k):
    display_info = get_display_mode_info(planner_type, confidences, display_idx)
    cv2.putText(
        vis_image,
        display_info["title"],
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    if display_info["confidence_text"] is not None:
        cv2.putText(
            vis_image,
            display_info["confidence_text"],
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        y_offset = 70
    else:
        y_offset = 50

    if ade is not None:
        cv2.putText(vis_image, f"ADE: {ade:.3f}m", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
    if fde is not None:
        cv2.putText(vis_image, f"FDE: {fde:.3f}m", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
    if minade_k is not None:
        cv2.putText(
            vis_image, f"minADE@K: {minade_k:.3f}m", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
        y_offset += 20
    if minfde_k is not None:
        cv2.putText(
            vis_image, f"minFDE@K: {minfde_k:.3f}m", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )


def get_bev_max_range(*traj_arrays, min_range=0.1):
    """
    计算 BEV 画布的显示范围，兼容空轨迹。
    """
    xy_list = [traj[:, :2] for traj in traj_arrays if traj is not None and len(traj) > 0]
    if not xy_list:
        return min_range
    all_xy = np.vstack(xy_list)
    return max(float(np.max(np.abs(all_xy))), min_range)


def draw_bev_minimap(image, pred_traj, gt_traj, map_size=150, margin=10):
    """
    在图像右上角绘制 BEV 轨迹小地图。

    注意:
        `image` 必须是 OpenCV 使用的 BGR 图像，返回值也保持 BGR。
    """
    H, W = image.shape[:2]
    minimap = np.zeros((map_size, map_size, 3), dtype=np.uint8)
    minimap[:] = (40, 40, 40)
    max_range = get_bev_max_range(pred_traj, gt_traj, min_range=0.05)
    ppm = (map_size * 0.4) / max_range
    center = map_size // 2

    def ego_to_pix(xy):
        u = center - (xy[:, 1] * ppm).astype(int)
        v = center - (xy[:, 0] * ppm).astype(int)
        return np.stack([u, v], axis=-1)

    def in_bounds(pix):
        return (pix[:, 0] >= 0) & (pix[:, 0] < map_size) & (pix[:, 1] >= 0) & (pix[:, 1] < map_size)

    gt_pix = ego_to_pix(gt_traj[:, :2])
    gt_mask = in_bounds(gt_pix)
    if gt_mask.sum() >= 2:
        cv2.polylines(minimap, [gt_pix[gt_mask]], False, (255, 0, 0), 2, cv2.LINE_AA)
        for pt in gt_pix[gt_mask]:
            cv2.circle(minimap, tuple(pt), 2, (255, 0, 0), -1)
    pred_pix = ego_to_pix(pred_traj[:, :2])
    pred_mask = in_bounds(pred_pix)
    if pred_mask.sum() >= 2:
        cv2.polylines(minimap, [pred_pix[pred_mask]], False, (0, 255, 0), 2, cv2.LINE_AA)
        for pt in pred_pix[pred_mask]:
            cv2.circle(minimap, tuple(pt), 2, (0, 255, 0), -1)
    cv2.circle(minimap, (center, center), 4, (0, 0, 255), -1)
    cv2.arrowedLine(minimap, (center, center), (center, center - 20), (0, 0, 255), 1, cv2.LINE_AA, tipLength=0.4)
    cv2.rectangle(minimap, (0, 0), (map_size - 1, map_size - 1), (180, 180, 180), 1)
    cv2.putText(minimap, "BEV", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    scale_text = f"{max_range:.2f}m" if max_range < 1.0 else f"{max_range:.1f}m"
    cv2.putText(minimap, scale_text, (map_size - 55, map_size - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
    x0 = max(0, W - map_size - margin)
    y0 = margin
    x1 = min(W, x0 + map_size)
    y1 = min(H, y0 + map_size)
    mw, mh = x1 - x0, y1 - y0
    roi = image[y0:y1, x0:x1]
    blended = cv2.addWeighted(roi, 0.3, minimap[:mh, :mw], 0.7, 0)
    image[y0:y1, x0:x1] = blended
    return image


def draw_error_labels(bev_canvas, pred_traj, gt_traj, ppm, center):
    """
    在BEV画布上绘制误差数值标签
    bev_canvas: BEV图像画布
    pred_traj: [num_poses, 3] 预测轨迹
    gt_traj: [num_poses, 3] GT轨迹
    ppm: pixels per meter
    center: BEV中心点坐标
    """
    for i in range(len(pred_traj)):
        gt_xy = gt_traj[i, :2]
        pred_xy = pred_traj[i, :2]

        # 转换到像素坐标 (ego_to_pix logic)
        u_pred = center - (pred_xy[1] * ppm)
        v_pred = center - (pred_xy[0] * ppm)

        # 计算误差
        error = np.linalg.norm(pred_xy - gt_xy)

        # 像素坐标
        pred_pix = (int(u_pred), int(v_pred))

        # 画误差数值标签 (在预测点上方)
        label = f"{error:.2f}m"
        cv2.putText(
            bev_canvas, label, (pred_pix[0] - 20, pred_pix[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1
        )


def draw_all_trajectories_side_by_side(
    image,
    pred_trajs,
    confidences,
    gt_traj,
    camera_params,
    ade=None,
    fde=None,
    minade_k=None,
    minfde_k=None,
    planner_type="transformer",
):
    """
    并排显示相机视图和BEV视图
    - 左侧：相机图像 + 预测/GT轨迹叠加
    - 右侧：大尺寸BEV视图 + 误差向量 + 数值标签

    注意:
        `image` 必须是 OpenCV 使用的 BGR 图像，返回值也保持 BGR。
    """
    H, W = image.shape[:2]

    # 左侧：相机视图（与原来相同）
    vis_image = image.copy()
    display_idx = select_display_trajectory_index(pred_trajs, confidences, planner_type=planner_type)
    best_traj = pred_trajs[display_idx]
    if planner_type == "diffusion":
        vis_image = _draw_sample_trajectories_on_image(vis_image, pred_trajs, camera_params)
    pred_pixels, pred_valid = transform_trajectory_to_image(best_traj, camera_params, (H, W))
    vis_image = draw_trajectory_on_image(
        vis_image, pred_pixels, pred_valid, (0, 255, 0), thickness=3, draw_points=True
    )
    gt_pixels, gt_valid = transform_trajectory_to_image(gt_traj, camera_params, (H, W))
    vis_image = draw_trajectory_on_image(vis_image, gt_pixels, gt_valid, (255, 0, 0), thickness=3, draw_points=True)
    ego_point_ego = np.array([[0, 0, 0]])
    ego_pixels, ego_valid = transform_trajectory_to_image(ego_point_ego, camera_params, (H, W))
    if ego_valid[0]:
        cv2.circle(vis_image, tuple(ego_pixels[0].astype(int)), 8, (0, 0, 255), -1)
    _draw_visualization_overlay_text(
        vis_image,
        planner_type,
        confidences,
        display_idx,
        ade,
        fde,
        minade_k,
        minfde_k,
    )

    # 右侧：BEV大图
    bev_size = H  # 与相机图像同高
    bev_canvas = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    bev_canvas[:] = (30, 30, 30)

    # 计算BEV缩放
    max_range = get_bev_max_range(best_traj, gt_traj, min_range=0.1)
    ppm = (bev_size * 0.45) / max_range
    center = bev_size // 2

    def ego_to_pix(xy):
        u = center - (xy[:, 1] * ppm)
        v = center - (xy[:, 0] * ppm)
        return np.stack([u, v], axis=-1).astype(int)

    def in_bounds(pix):
        return (pix[:, 0] >= 0) & (pix[:, 0] < bev_size) & (pix[:, 1] >= 0) & (pix[:, 1] < bev_size)

    # 绘制GT轨迹 (蓝色)
    gt_pix = ego_to_pix(gt_traj[:, :2])
    gt_mask = in_bounds(gt_pix)
    if gt_mask.sum() >= 2:
        cv2.polylines(bev_canvas, [gt_pix[gt_mask]], False, (255, 0, 0), 2, cv2.LINE_AA)
        for pt in gt_pix[gt_mask]:
            cv2.circle(bev_canvas, tuple(pt), 3, (255, 0, 0), -1)

    if planner_type == "diffusion":
        _draw_sample_trajectories_on_bev(bev_canvas, pred_trajs, ego_to_pix, in_bounds)

    # 绘制预测轨迹 (绿色)
    pred_pix = ego_to_pix(best_traj[:, :2])
    pred_mask = in_bounds(pred_pix)
    if pred_mask.sum() >= 2:
        cv2.polylines(bev_canvas, [pred_pix[pred_mask]], False, (0, 255, 0), 2, cv2.LINE_AA)
        for pt in pred_pix[pred_mask]:
            cv2.circle(bev_canvas, tuple(pt), 3, (0, 255, 0), -1)

    # 绘制误差数值标签
    draw_error_labels(bev_canvas, best_traj, gt_traj, ppm, center)

    # 绘制坐标系指示
    cv2.circle(bev_canvas, (center, center), 5, (0, 0, 255), -1)
    cv2.arrowedLine(bev_canvas, (center, center), (center, center - 40), (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(bev_canvas, "BEV (top-down)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    scale_text = f"Scale: {max_range:.2f}m"
    cv2.putText(bev_canvas, scale_text, (10, bev_size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # 并排拼接
    combined = np.hstack([vis_image, bev_canvas])

    return combined


def draw_all_trajectories(
    image,
    pred_trajs,
    confidences,
    gt_traj,
    camera_params,
    ade=None,
    fde=None,
    minade_k=None,
    minfde_k=None,
    planner_type="transformer",
):
    """
    在图像上绘制最佳预测轨迹和 GT 轨迹, 并附带 BEV 小地图。

    注意:
        `image` 必须是 OpenCV 使用的 BGR 图像，返回值也保持 BGR。
    """
    H, W = image.shape[:2]
    vis_image = image.copy()
    display_idx = select_display_trajectory_index(pred_trajs, confidences, planner_type=planner_type)
    best_traj = pred_trajs[display_idx]
    if planner_type == "diffusion":
        vis_image = _draw_sample_trajectories_on_image(vis_image, pred_trajs, camera_params)
    pred_pixels, pred_valid = transform_trajectory_to_image(best_traj, camera_params, (H, W))
    vis_image = draw_trajectory_on_image(
        vis_image, pred_pixels, pred_valid, (0, 255, 0), thickness=3, draw_points=True
    )
    gt_pixels, gt_valid = transform_trajectory_to_image(gt_traj, camera_params, (H, W))
    vis_image = draw_trajectory_on_image(vis_image, gt_pixels, gt_valid, (255, 0, 0), thickness=3, draw_points=True)
    ego_point_ego = np.array([[0, 0, 0]])
    ego_pixels, ego_valid = transform_trajectory_to_image(ego_point_ego, camera_params, (H, W))
    if ego_valid[0]:
        cv2.circle(vis_image, tuple(ego_pixels[0].astype(int)), 8, (0, 0, 255), -1)
    vis_image = draw_bev_minimap(vis_image, best_traj, gt_traj)
    _draw_visualization_overlay_text(
        vis_image,
        planner_type,
        confidences,
        display_idx,
        ade,
        fde,
        minade_k,
        minfde_k,
    )
    return vis_image


def save_color_space_check_image(frame_rgb, output_path):
    """
    保存一张颜色空间对照图，用于肉眼确认 decord 输出是否为 RGB。

    左侧：直接按原数组写入（OpenCV 会按 BGR 理解，若原图是 RGB 会出现颜色颠倒）
    右侧：先做 RGB->BGR 转换后再写入（这是 OpenCV 正常显示方式）
    """
    if frame_rgb is None:
        return

    raw_panel = frame_rgb.copy()
    converted_panel = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    H, W = raw_panel.shape[:2]
    canvas = np.zeros((H + 40, W * 2, 3), dtype=np.uint8)
    canvas[:H, :W] = raw_panel
    canvas[:H, W:] = converted_panel

    cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (0, 255, 255), 2)
    cv2.rectangle(canvas, (W, 0), (W * 2 - 1, H - 1), (0, 255, 0), 2)
    cv2.putText(canvas, "Left: raw frame array", (10, H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(
        canvas, "Right: RGB->BGR converted", (W + 10, H + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )
    cv2.putText(
        canvas,
        "If left looks color-swapped, source is RGB",
        (10, H + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    cv2.imwrite(output_path, canvas)
