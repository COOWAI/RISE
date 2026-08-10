"""NuScenes 开环评估指标：分时间步 L2 Error 和碰撞率

实现 ST-P3/VAD 标准的 BEV 栅格化碰撞检测和逐时间步 L2 位移误差。

Collision rate 对齐 World4Drive/ST-P3/VAD：
- BEV seg map 使用 per-frame ego 坐标系
- 同时计算 point collision (obj_col) 和 box collision (obj_box_col)
- GT 碰撞时间步从碰撞计数中排除
- horizon 指标为截至该秒的累计均值（与 World4Drive plan_obj_box_col_* 一致）
"""

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)
WORLD4DRIVE_REPORTED_SECONDS = (1, 2, 3, 4)

# NuScenes ST-P3/VAD 碰撞检测标准参数
EGO_WIDTH = 1.85  # Lincoln MKZ 宽度 (m)
EGO_LENGTH = 4.084  # Lincoln MKZ 长度 (m)
EGO_LENGTH_OFFSET = 0.5  # 前向偏移 (m)
BEV_RANGE = 50.0  # BEV 覆盖范围 (m), [-50, 50]
BEV_RESOLUTION = 0.5  # 每像素对应距离 (m)
BEV_SIZE = int(2 * BEV_RANGE / BEV_RESOLUTION)  # 200


# =====================================================================
#  L2 per timestep
# =====================================================================


def compute_l2_per_timestep(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    timestep_sec: float = 0.5,
) -> Dict[str, float]:
    """计算分时间步 L2 位移误差（NuScenes 标准）。

    Parameters
    ----------
    pred_traj    : [B, T, 3]  预测轨迹 (x, y, yaw)，自车坐标系
    gt_traj      : [B, T, 3]  GT 轨迹 (x, y, yaw)，自车坐标系
    timestep_sec : float      每步时间间隔（秒），默认 0.5s (2Hz)

    Returns
    -------
    dict:
        "l2_avg"      : 所有报告时间点的平均 L2
        "l2_at_Xs"    : 各整秒时间点的 L2 (X = 1, 2, 3, 4)
        "l2_per_step" : 每步 L2 值列表
    """
    pred_xy = pred_traj[..., :2]  # [B, T, 2]
    gt_xy = gt_traj[..., :2]  # [B, T, 2]
    l2 = torch.norm(pred_xy - gt_xy, dim=-1)  # [B, T]
    l2_per_step = l2.mean(dim=0)  # [T]

    T = l2_per_step.shape[0]
    result: Dict[str, float] = {}
    result["l2_per_step"] = l2_per_step.tolist()

    # 报告整秒时间点 (1s, 2s, 3s, 4s)
    reported_horizons: List[float] = []
    for sec in [1, 2, 3, 4]:
        # round() 防浮点截断：如 timestep_sec=0.2 时 3/0.2=14.999...，int() 会错位到 2.8s。
        step_idx = int(round(sec / timestep_sec)) - 1  # 0-indexed
        if 0 <= step_idx < T:
            result[f"l2_at_{sec}s"] = l2_per_step[step_idx].item()
            reported_horizons.append(l2_per_step[step_idx].item())

    if not reported_horizons:
        # fail-loud: 轨迹连 1s horizon 都覆盖不到时禁止回退成 0.0（会被当成完美指标）。
        raise ValueError(
            f"compute_l2_per_timestep: no reported horizon fits trajectory "
            f"(T={T}, timestep_sec={timestep_sec}); l2_avg 禁止回退为 0.0。"
        )
    result["l2_avg"] = float(np.mean(reported_horizons))

    return result


def populate_world4drive_l2_horizons(
    metrics: Dict[str, float],
    l2_per_step: np.ndarray,
    timestep_sec: float,
    reported_seconds: Tuple[int, ...] = WORLD4DRIVE_REPORTED_SECONDS,
) -> None:
    """Populate cumulative ADE@horizon fields used by World4Drive/ST-P3."""
    reported_horizons = []
    for sec in reported_seconds:
        horizon_steps = int(round(sec / timestep_sec))
        if 0 < horizon_steps <= len(l2_per_step):
            l2_value = float(np.mean(l2_per_step[:horizon_steps]))
            metrics[f"l2_at_{sec}s"] = l2_value
            reported_horizons.append(l2_value)
    # fail-loud: 无可报告 horizon 时不得回退成 l2_avg=0.0（对"越小越好"的指标 0.0 是满分，
    # 会让退化/损坏的验证在 checkpoint 选择中胜出）。空 horizon 说明轨迹长度不足以覆盖任一报告秒，是真实错误。
    if not reported_horizons:
        raise ValueError(
            "populate_world4drive_l2_horizons: no reportable horizon within the trajectory "
            f"(timestep_sec={timestep_sec}, reported_seconds={reported_seconds}, "
            f"len(l2_per_step)={len(l2_per_step)}); refusing to report l2_avg=0.0 for a degenerate trajectory."
        )
    metrics["l2_avg"] = float(np.mean(reported_horizons))


def populate_point_l2_horizons(
    metrics: Dict[str, float],
    l2_per_step: np.ndarray,
    timestep_sec: float,
    reported_seconds: Tuple[int, ...] = WORLD4DRIVE_REPORTED_SECONDS,
) -> None:
    """Populate original single-point L2@horizon fields for compatibility."""
    reported_horizons = []
    for sec in reported_seconds:
        step_idx = int(round(sec / timestep_sec)) - 1
        if 0 <= step_idx < len(l2_per_step):
            l2_value = float(l2_per_step[step_idx])
            metrics[f"l2_point_at_{sec}s"] = l2_value
            reported_horizons.append(l2_value)
    # fail-loud: 同 l2_avg，空 horizon 不得回退成 0.0（满分），应报错暴露退化轨迹。
    if not reported_horizons:
        raise ValueError(
            "populate_point_l2_horizons: no reportable horizon within the trajectory "
            f"(timestep_sec={timestep_sec}, reported_seconds={reported_seconds}, "
            f"len(l2_per_step)={len(l2_per_step)}); refusing to report l2_point_avg=0.0 for a degenerate trajectory."
        )
    metrics["l2_point_avg"] = float(np.mean(reported_horizons))


def populate_world4drive_collision_horizons(
    metrics: Dict[str, float],
    collision_counts: np.ndarray,
    total_samples: float,
    timestep_sec: float,
    metric_prefix: str = "collision",
    avg_key: str = "collision_rate",
    reported_seconds: Tuple[int, ...] = WORLD4DRIVE_REPORTED_SECONDS,
) -> None:
    """Populate cumulative collision@horizon fields used by World4Drive.

    World4Drive masks GT-colliding timesteps from the numerator, then reports
    the mean collision over all timesteps up to each horizon.
    """
    per_step = np.asarray(collision_counts, dtype=np.float64) / max(float(total_samples), 1.0)
    reported_horizons = []
    for sec in reported_seconds:
        horizon_steps = int(round(sec / timestep_sec))
        if 0 < horizon_steps <= len(per_step):
            value = float(np.mean(per_step[:horizon_steps]))
            metrics[f"{metric_prefix}_at_{sec}s"] = value
            reported_horizons.append(value)
    if not reported_horizons:
        # No reported-second horizon fits this trajectory length (e.g. a short eval). Report NaN, not a
        # misleading 0.0 (which reads as "no collisions") — convention allows NaN+log over a raise here.
        logger.warning(
            "No collision horizon fits reported_seconds=%s at timestep_sec=%s (per_step len=%d); "
            "reporting %s=NaN instead of 0.0.",
            reported_seconds,
            timestep_sec,
            len(per_step),
            avg_key,
        )
        metrics[avg_key] = float("nan")
    else:
        metrics[avg_key] = float(np.mean(reported_horizons))


def compute_world4drive_l2_metrics(
    pred_traj: torch.Tensor,
    gt_traj: torch.Tensor,
    timestep_sec: float = 0.5,
) -> Dict[str, float]:
    """Compute World4Drive cumulative L2 metrics plus original point L2 metrics.

    ``l2_at_*`` uses cumulative ADE up to the horizon. ``l2_point_at_*`` keeps
    the old single-horizon displacement for side-by-side reporting.
    """
    pred_xy = pred_traj[..., :2]
    gt_xy = gt_traj[..., :2]
    l2 = torch.norm(pred_xy - gt_xy, dim=-1)
    l2_per_step = l2.mean(dim=0).detach().cpu().double().numpy()

    result: Dict[str, float] = {"l2_per_step": l2_per_step.tolist()}
    populate_world4drive_l2_horizons(result, l2_per_step, timestep_sec)
    populate_point_l2_horizons(result, l2_per_step, timestep_sec)
    return result


# =====================================================================
#  Collision rate helpers
# =====================================================================


def _create_ego_footprint_pixels() -> Tuple[np.ndarray, np.ndarray]:
    """创建固定的自车轴对齐矩形像素足迹（相对于自车中心的像素偏移）。

    遵循 ST-P3/VAD 标准：ego 为轴对齐矩形，带 0.5m 前向偏移。
    使用 cv2.fillPoly 来创建像素足迹。

    Returns
    -------
    (row_offsets, col_offsets) : 自车中心位于 (0,0) 时的像素坐标偏移
    """
    half_w = EGO_WIDTH / 2.0
    half_l = EGO_LENGTH / 2.0

    # 自车矩形四角（米，自车中心为原点）
    # BEV 坐标系 x=前方, y=左方
    # 四角 (前方为+x)，+0.5m 前向偏移
    corners_m = np.array(
        [
            [-half_l + EGO_LENGTH_OFFSET, half_w],  # 后左
            [half_l + EGO_LENGTH_OFFSET, half_w],  # 前左
            [half_l + EGO_LENGTH_OFFSET, -half_w],  # 前右
            [-half_l + EGO_LENGTH_OFFSET, -half_w],  # 后右
        ]
    )

    # 米 → 像素偏移（以 canvas 中心为原点）
    canvas_half = 20  # 20 像素半径足以覆盖 ~4m 的车
    canvas_size = 2 * canvas_half + 1
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # corners → 像素坐标 (col, row) 以 canvas 中心为原点
    pts_col = (corners_m[:, 0] / BEV_RESOLUTION + canvas_half).astype(np.int32)
    pts_row = (-corners_m[:, 1] / BEV_RESOLUTION + canvas_half).astype(np.int32)

    pts = np.stack([pts_col, pts_row], axis=1).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts.reshape((-1, 1, 2))], color=1)

    rr, cc = np.where(canvas > 0)
    # 转换为相对于中心的偏移
    rr = rr - canvas_half
    cc = cc - canvas_half

    return rr, cc


def _rasterize_agents_to_bev(
    agent_boxes: np.ndarray,
    agent_mask: np.ndarray,
) -> np.ndarray:
    """将单帧的 agent OBB 栅格化到 BEV 分割图上。

    agent_boxes 应在该帧的 ego 坐标系下（per-frame ego coords）。

    Parameters
    ----------
    agent_boxes : [max_agents, 7]  = [x, y, z, length, width, height, heading]
    agent_mask  : [max_agents]     bool

    Returns
    -------
    segmentation : [BEV_SIZE, BEV_SIZE]  uint8, 1=occupied, 0=free
    """
    seg = np.zeros((BEV_SIZE, BEV_SIZE), dtype=np.uint8)

    for i in range(agent_boxes.shape[0]):
        if not agent_mask[i]:
            continue

        x, y, _, length, width, _, heading = agent_boxes[i]

        # 检查 agent 是否在 BEV 范围内
        if abs(x) > BEV_RANGE or abs(y) > BEV_RANGE:
            continue

        half_l = length / 2.0
        half_w = width / 2.0

        # OBB 四角（agent 局部坐标系）
        corners_local = np.array(
            [
                [half_l, half_w],
                [half_l, -half_w],
                [-half_l, -half_w],
                [-half_l, half_w],
            ]
        )

        # 旋转到全局坐标系
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners_global = corners_local @ rot.T + np.array([x, y])

        # 米 → 像素
        # BEV 图像中心对应坐标原点
        # col = (x + BEV_RANGE) / BEV_RESOLUTION
        # row = (BEV_RANGE - y) / BEV_RESOLUTION  (y 轴反向)
        corners_col = ((corners_global[:, 0] + BEV_RANGE) / BEV_RESOLUTION).astype(np.int32)
        corners_row = ((BEV_RANGE - corners_global[:, 1]) / BEV_RESOLUTION).astype(np.int32)

        # cv2.fillPoly expects (x, y) = (col, row)
        pts = np.stack([corners_col, corners_row], axis=1).reshape((-1, 1, 2))
        cv2.fillPoly(seg, [pts], color=1)

    return seg


def _transform_traj_point_to_frame_t(
    point_ego0: np.ndarray,
    ego_pose_0: np.ndarray,
    ego_pose_t: np.ndarray,
) -> np.ndarray:
    """将轨迹点从 frame-0 ego 坐标系转换到 frame-t ego 坐标系。

    Parameters
    ----------
    point_ego0 : [2]  (x, y) in frame-0 ego coords
    ego_pose_0 : [7]  frame-0 ego state (x, y, z, roll, pitch, yaw, speed)
    ego_pose_t : [7]  frame-t ego state (x, y, z, roll, pitch, yaw, speed)

    Returns
    -------
    point_t : [2]  (x, y) in frame-t ego coords
    """
    px, py = float(point_ego0[0]), float(point_ego0[1])

    # Step 1: frame-0 ego coords → scene (global) coords
    ex_0 = float(ego_pose_0[0])
    ey_0 = float(ego_pose_0[1])
    eyaw_0 = float(ego_pose_0[5])
    cos_0 = np.cos(eyaw_0)
    sin_0 = np.sin(eyaw_0)
    gx = ex_0 + cos_0 * px - sin_0 * py
    gy = ey_0 + sin_0 * px + cos_0 * py

    # Step 2: scene coords → frame-t ego coords
    ex_t = float(ego_pose_t[0])
    ey_t = float(ego_pose_t[1])
    eyaw_t = float(ego_pose_t[5])
    cos_t_inv = np.cos(-eyaw_t)
    sin_t_inv = np.sin(-eyaw_t)
    dx = gx - ex_t
    dy = gy - ey_t
    tx = cos_t_inv * dx - sin_t_inv * dy
    ty = sin_t_inv * dx + cos_t_inv * dy

    return np.array([tx, ty], dtype=np.float64)


def _point_to_bev_pixel(x: float, y: float) -> Tuple[int, int]:
    """将 ego 坐标系中的 (x, y) 转换为 BEV 像素 (row, col)。

    Parameters
    ----------
    x : forward direction (meters)
    y : left direction (meters)

    Returns
    -------
    (row, col) : BEV 像素坐标
    """
    # floor, not int(): int() truncates toward zero, so a point just past the negative edge
    # (e.g. x < -BEV_RANGE) would round to 0 and read as falsely in-bounds.
    col = math.floor((x + BEV_RANGE) / BEV_RESOLUTION)
    row = math.floor((BEV_RANGE - y) / BEV_RESOLUTION)
    return row, col


# =====================================================================
#  Collision rate (ST-P3/VAD aligned)
# =====================================================================


def compute_collision_rate(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    segmentation: np.ndarray,
    ego_poses: np.ndarray,
    future_start_idx: int = 1,
    timestep_sec: float = 0.5,
    reference_frame_idx: int = 0,
) -> Dict[str, float]:
    """计算时间步级别碰撞率（严格对齐 ST-P3/VAD 标准）。

    BEV seg map 在 per-frame ego 坐标系，预测轨迹在 reference_frame ego 坐标系。
    函数内部将每个轨迹点转换到对应帧的 ego 坐标系后进行碰撞检测。

    同时计算两类碰撞：
    - point collision (obj_col): 单像素中心点检查
    - box collision (obj_box_col): ego 足迹多边形检查 (~32 像素)

    GT 碰撞时间步从 numerator 中排除；报告的 horizon 指标是截至该秒的累计均值。

    Parameters
    ----------
    pred_traj      : [B, T_future, 3]  预测轨迹 (x, y, yaw)，reference_frame ego 坐标系
    gt_traj        : [B, T_future, 3]  GT 轨迹，同坐标系
    segmentation   : [B, T_future, BEV_SIZE, BEV_SIZE]  预计算 BEV seg map
                     每帧在该帧 ego 坐标系下, uint8/bool, 1=occupied
    ego_poses      : [B, T_total, 7]  ego states (x, y, z, roll, pitch, yaw, speed)
                     用于坐标转换。ego_poses[:, reference_frame_idx] 为轨迹参考帧。
    future_start_idx : int  ego_poses 中未来帧的起始索引
    timestep_sec   : float  每步时间间隔
    reference_frame_idx : int  轨迹所在的参考帧索引（默认 0，IC 模式下为 num_observed-1）

    Returns
    -------
    dict:
        "collision_rate"      : box collision 所有报告时间点的平均碰撞率
        "collision_at_Xs"     : box collision 各整秒时间点的碰撞率
        "collision_per_step"  : box collision 每步碰撞率列表
        "collision_counts"    : box collision 每步碰撞计数
        "point_collision_rate"      : point collision 平均碰撞率
        "point_collision_at_Xs"     : point collision 各整秒碰撞率
        "point_collision_per_step"  : point collision 每步碰撞率列表
        "point_collision_counts"    : point collision 每步碰撞计数
        "total_samples"       : 总样本数 B
        "gt_collision_counts" : 每步 GT 碰撞排除数（仅供调试）
    """
    # 缓存 ego footprint 像素偏移
    ego_rr, ego_cc = _create_ego_footprint_pixels()

    B = pred_traj.shape[0]
    T_future = pred_traj.shape[1]

    # 统计每步碰撞数
    box_collision_counts = np.zeros(T_future, dtype=np.int64)
    point_collision_counts = np.zeros(T_future, dtype=np.int64)
    gt_collision_counts = np.zeros(T_future, dtype=np.int64)

    for b in range(B):
        for t in range(T_future):
            frame_t_idx = future_start_idx + t  # ego_poses 中对应的帧索引

            # 安全检查
            if frame_t_idx >= ego_poses.shape[1]:
                continue

            seg = segmentation[b, t]  # [BEV_SIZE, BEV_SIZE]

            # --- 将 GT 轨迹点转换到 frame-t ego 坐标系 ---
            gt_point_ego0 = gt_traj[b, t, :2]
            gt_point_t = _transform_traj_point_to_frame_t(
                gt_point_ego0, ego_poses[b, reference_frame_idx], ego_poses[b, frame_t_idx]
            )

            # --- 检查 GT box collision（用于过滤） ---
            gt_row, gt_col = _point_to_bev_pixel(gt_point_t[0], gt_point_t[1])
            gt_box_rr = ego_rr + gt_row
            gt_box_cc = ego_cc + gt_col
            gt_valid_mask = (gt_box_rr >= 0) & (gt_box_rr < BEV_SIZE) & (gt_box_cc >= 0) & (gt_box_cc < BEV_SIZE)
            gt_in_collision = False
            if gt_valid_mask.any():
                gt_in_collision = bool(seg[gt_box_rr[gt_valid_mask], gt_box_cc[gt_valid_mask]].any())

            if gt_in_collision:
                # GT 碰撞时间步排除
                gt_collision_counts[t] += 1
                continue

            # --- 将预测轨迹点转换到 frame-t ego 坐标系 ---
            pred_point_ego0 = pred_traj[b, t, :2]
            pred_point_t = _transform_traj_point_to_frame_t(
                pred_point_ego0, ego_poses[b, reference_frame_idx], ego_poses[b, frame_t_idx]
            )

            # --- Point collision (obj_col): 单像素检查 ---
            p_row, p_col = _point_to_bev_pixel(pred_point_t[0], pred_point_t[1])
            if 0 <= p_row < BEV_SIZE and 0 <= p_col < BEV_SIZE:
                if seg[p_row, p_col]:
                    point_collision_counts[t] += 1

            # --- Box collision (obj_box_col): ego 足迹检查 ---
            p_box_rr = ego_rr + p_row
            p_box_cc = ego_cc + p_col
            p_valid_mask = (p_box_rr >= 0) & (p_box_rr < BEV_SIZE) & (p_box_cc >= 0) & (p_box_cc < BEV_SIZE)
            if p_valid_mask.any() and seg[p_box_rr[p_valid_mask], p_box_cc[p_valid_mask]].any():
                box_collision_counts[t] += 1

    # World4Drive/ST-P3: GT-colliding steps are masked from numerator;
    # rates are normalized by the total sample count and averaged cumulatively.
    box_collision_per_step = box_collision_counts / max(float(B), 1.0)
    point_collision_per_step = point_collision_counts / max(float(B), 1.0)

    result: Dict[str, float] = {}
    result["collision_per_step"] = box_collision_per_step.tolist()
    result["collision_counts"] = box_collision_counts.tolist()
    result["point_collision_per_step"] = point_collision_per_step.tolist()
    result["point_collision_counts"] = point_collision_counts.tolist()
    result["total_samples"] = int(B)
    result["gt_collision_counts"] = gt_collision_counts.tolist()

    populate_world4drive_collision_horizons(
        result,
        box_collision_counts,
        total_samples=B,
        timestep_sec=timestep_sec,
        metric_prefix="collision",
        avg_key="collision_rate",
    )
    populate_world4drive_collision_horizons(
        result,
        point_collision_counts,
        total_samples=B,
        timestep_sec=timestep_sec,
        metric_prefix="point_collision",
        avg_key="point_collision_rate",
    )

    return result


# =====================================================================
#  Legacy interface (backward compatible)
# =====================================================================


def compute_collision_rate_legacy(
    pred_traj: np.ndarray,
    gt_traj: np.ndarray,
    agent_boxes_all: np.ndarray,
    agent_mask_all: np.ndarray,
    states: np.ndarray,
    future_start_idx: int = 1,
    timestep_sec: float = 0.5,
) -> Dict[str, float]:
    """Legacy 碰撞率计算（保留向后兼容）。

    内部将 agent boxes 从各帧 ego 坐标系就地栅格化到 per-frame BEV seg map，
    然后调用新的 compute_collision_rate。

    Parameters
    ----------
    pred_traj       : [B, T_future, 3]  预测轨迹 (x, y, yaw)，frame-0 ego 坐标系
    gt_traj         : [B, T_future, 3]  GT 轨迹，同坐标系
    agent_boxes_all : [B, T_total, max_agents, 7]  原始 agent boxes（各帧 ego 坐标系）
    agent_mask_all  : [B, T_total, max_agents]     bool
    states          : [B, T_total, 7]  ego states
    future_start_idx: int  states 中未来帧的起始索引
    timestep_sec    : float  每步时间间隔

    Returns
    -------
    dict: 同 compute_collision_rate
    """
    B = pred_traj.shape[0]
    T_future = pred_traj.shape[1]

    # 预计算 per-frame BEV seg maps
    segmentation = np.zeros((B, T_future, BEV_SIZE, BEV_SIZE), dtype=np.uint8)
    for b in range(B):
        for t in range(T_future):
            frame_t_idx = future_start_idx + t
            if frame_t_idx < agent_boxes_all.shape[1]:
                segmentation[b, t] = _rasterize_agents_to_bev(
                    agent_boxes_all[b, frame_t_idx],
                    agent_mask_all[b, frame_t_idx],
                )

    return compute_collision_rate(
        pred_traj=pred_traj,
        gt_traj=gt_traj,
        segmentation=segmentation,
        ego_poses=states,
        future_start_idx=future_start_idx,
        timestep_sec=timestep_sec,
    )
