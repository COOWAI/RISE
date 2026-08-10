"""World-model 预测准度辅助监督 (路线 Phase 1 / world-model-selector notes 方向 D).

为 predictor 训练提供两个 config 门控的辅助 loss(第三项 λ^k 折扣实现在
``training/predictor_loss.py``,因为它直接改写 sloss):

1. **reward/risk 辅助头联合训练** (doc 9.3, ``wm_aux.reward_head_weight``):
   ``PredictorRewardHead`` 以**带梯度的 z_ar**(predictor AR rollout 输出)为输入,
   BCE 监督 vs ``compute_safety_reward_labels`` 离线安全标签。与
   ``train_reward_model``(冻结 predictor 上训头)的本质区别:梯度流入 predictor,
   迫使 world model 学到与驾驶决策相关的环境结构。

2. **contrastive ranking** (doc 9.4, ``wm_aux.contrastive_weight``):
   GT 轨迹条件下 rollout 出的未来 latent,应当比反事实轨迹(无参数
   ``HistoryKinematicProposalProvider`` 运动学网格生成)条件下的更接近真实未来
   latent ``h_target``。正样本(GT 轨迹)与 K 条负样本**走同一次
   ``rollout_predictor_modes`` 调用**(stack 成 K+1 modes),只有轨迹不同——
   避免两条 conditioning 路径的系统性偏差。margin ranking loss,梯度流经正负
   两支(predictor 同时学"按真实 action 预测更准"与"不同 action → 不同未来")。

Fail-loud(仓库约定):权重非零但输入缺失(无 agent boxes / 无 h_target /
监督 scope 不符)一律 raise,不静默跳过。
"""

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from app.vjepa_cowa_world_model.models.predictor_reward import PredictorRewardHead, compute_reward_head_loss
from app.vjepa_cowa_world_model.models.proposal_providers import HistoryKinematicProposalProvider
from app.vjepa_cowa_world_model.training.config import TrainingConfig
from app.vjepa_cowa_world_model.training.predictor_loss import predictor_uses_future_only_loss_scope
from app.vjepa_cowa_world_model.training.reward_labels import compute_safety_reward_labels
from app.vjepa_cowa_world_model.training.runtimes.reward_runtime import (
    align_temporal_to_timeline,
    extract_future_agent_boxes,
    make_reward_label_config,
)
from app.vjepa_cowa_world_model.utils.status_features import (
    build_future_gt_trajectory_from_states,
    build_observed_action_trajectory_history,
)


def wm_aux_enabled(config: TrainingConfig) -> bool:
    """Whether any wm_aux component that needs runtime modules is on."""
    wm = config.wm_aux
    return float(wm.reward_head_weight) > 0.0 or float(wm.contrastive_weight) > 0.0


def init_wm_aux_modules(
    config: TrainingConfig,
    *,
    encoder_dim: int,
    device: torch.device,
) -> Optional[Dict[str, torch.nn.Module]]:
    """Instantiate the trainable reward head and the parameter-free negative
    generator. Returns ``None`` when both components are off."""
    wm = config.wm_aux
    if not wm_aux_enabled(config):
        return None

    modules: Dict[str, torch.nn.Module] = {}
    if float(wm.reward_head_weight) > 0.0:
        modules["reward_head"] = PredictorRewardHead(
            embed_dim=int(encoder_dim),
            hidden_dim=int(wm.reward_head_hidden_dim),
            num_horizons=len(config.reward.horizon_seconds),
        ).to(device)

    if float(wm.contrastive_weight) > 0.0:
        num_negatives = int(wm.contrastive_num_negatives)
        num_poses = int(config.data.num_target_frames) - int(config.train.num_observed_frames)
        if num_poses < 1:
            raise ValueError(
                "wm_aux contrastive requires at least 1 future frame; got "
                f"num_target_frames={config.data.num_target_frames}, "
                f"num_observed_frames={config.train.num_observed_frames}"
            )
        # 无参数运动学网格 → 免费反事实负样本(轨迹与 GT 同 ego 系/同 raw 帧率)。
        modules["negative_provider"] = HistoryKinematicProposalProvider(
            num_modes=num_negatives,
            num_poses=num_poses,
            hidden_dim=8,  # features 不使用,仅满足构造;最小化开销
        ).to(device)

    return modules


def compute_wm_reward_head_loss(
    *,
    config: TrainingConfig,
    reward_head: torch.nn.Module,
    z_ar: torch.Tensor,
    sample: Any,
    timeline_states: torch.Tensor,
    num_observed_steps: int,
    num_time_steps: int,
    frame_stride: int,
    tokens_per_frame: int,
) -> Dict[str, torch.Tensor]:
    """Joint reward/risk head loss on grad-carrying ``z_ar`` (doc 9.3).

    ``z_ar`` must be the future-only AR rollout tokens ``[B, F*tpf, D]``
    (predictor_loss future_only scope). Labels mirror train_reward_model:
    agent boxes from the collate sample, aligned to the predictor timeline.
    """
    if not predictor_uses_future_only_loss_scope(config):
        raise ValueError(
            "wm_aux.reward_head_weight > 0 requires predictor_loss_scope == 'future_only' "
            "(z_ar must contain exactly the future tokens); got next_step scope (fail-loud)."
        )
    if timeline_states.ndim != 3 or timeline_states.shape[-1] < 7:
        raise ValueError(f"timeline_states must be [B, T, >=7], got {tuple(timeline_states.shape)}")

    # 复用 collision 接线的 collate 索引/形状守卫;此处取全时间线再按 stride 对齐。
    raw_boxes, raw_mask = extract_future_agent_boxes(
        sample,
        future_start_idx=0,
        num_poses=int(sample[7].shape[1]),
        device=z_ar.device,
    )
    tl_boxes = align_temporal_to_timeline(raw_boxes, frame_stride=frame_stride, num_time_steps=num_time_steps)
    tl_mask = align_temporal_to_timeline(raw_mask, frame_stride=frame_stride, num_time_steps=num_time_steps).bool()

    timestep_sec = float(frame_stride) / max(float(config.data.fps), 1.0)
    label_config = make_reward_label_config(config, timestep_sec=timestep_sec)
    labels = compute_safety_reward_labels(
        timeline_states[..., :7].float(),
        tl_boxes,
        tl_mask,
        num_observed_frames=int(num_observed_steps),
        config=label_config,
    )

    outputs = reward_head(z_ar, tokens_per_frame=int(tokens_per_frame))
    result = compute_reward_head_loss(outputs, {"scalar": labels.scalar, "horizons": labels.horizons})
    return {
        "loss": result["loss"],
        "scalar_loss": result["scalar_loss"],
        "horizon_loss": result["horizon_loss"],
    }


def compute_wm_contrastive_loss(
    *,
    config: TrainingConfig,
    predictor: torch.nn.Module,
    negative_provider: torch.nn.Module,
    z_context: torch.Tensor,
    h_target: torch.Tensor,
    actions: torch.Tensor,
    states: torch.Tensor,
    driving_command: Optional[torch.Tensor],
    ego_dynamics: Optional[torch.Tensor],
    tokens_per_frame: int,
    runtime_normalize_reps: bool,
    num_observed_steps: int,
    frame_stride: int,
) -> Dict[str, torch.Tensor]:
    """Contrastive ranking loss (doc 9.4): rollout under the GT trajectory must
    match the real future latent better than rollouts under counterfactual
    trajectories, by at least ``contrastive_margin``.

    Positive and negatives share ONE ``rollout_predictor_modes`` call (stacked
    as K+1 modes) so only the conditioning trajectory differs. Gradients flow
    through both branches into the predictor.
    """
    if h_target is None:
        raise ValueError("wm_aux.contrastive_weight > 0 requires h_target (real future latent); got None.")

    wm = config.wm_aux
    num_negatives = int(wm.contrastive_num_negatives)
    num_obs_raw = int(config.train.num_observed_frames)
    num_poses = int(config.data.num_target_frames) - num_obs_raw
    dt = 1.0 / max(float(config.data.fps), 1.0)

    # 正样本 = 真实发生的未来轨迹(ego 系,最后观测帧为原点)。
    gt_traj = build_future_gt_trajectory_from_states(states, num_obs_raw, num_poses=num_poses)
    if gt_traj.shape[1] != num_poses:
        raise ValueError(
            f"GT future trajectory covers {gt_traj.shape[1]} poses, expected {num_poses}; "
            "states tensor does not span the full clip (fail-loud)."
        )

    # 负样本 = 运动学网格反事实轨迹(同帧率/同坐标系)。
    history = build_observed_action_trajectory_history(
        actions=actions,
        num_observed_frames=num_obs_raw,
        action_history_dim=3,
        dt=dt,
    )
    negatives = negative_provider(None, None, history)["trajectories"]  # [B, K, P, 3]
    if negatives.shape[1] != num_negatives or negatives.shape[2] != num_poses:
        raise ValueError(
            f"negative trajectories must be [B, {num_negatives}, {num_poses}, 3], got {tuple(negatives.shape)}"
        )

    # 一次 rollout:mode 0 = GT(正),mode 1..K = 反事实(负)。
    candidate_trajs = torch.cat([gt_traj.unsqueeze(1), negatives], dim=1).to(z_context.dtype)

    # Lazy import to avoid module-level cycle with lewm_stage_runtime.
    from app.vjepa_cowa_world_model.training.runtimes.refinement_runtime import rollout_predictor_modes

    z_hat = rollout_predictor_modes(
        predictor=predictor,
        z_context=z_context,
        future_trajs=candidate_trajs,
        actions=actions,
        states=states,
        driving_command=driving_command,
        ego_dynamics=ego_dynamics,
        config=config,
        tokens_per_frame=int(tokens_per_frame),
        runtime_normalize_reps=bool(runtime_normalize_reps),
        dt=dt,
        predictor_observed_steps=int(num_observed_steps),
        predictor_frame_stride=int(frame_stride),
    )  # [B, K+1, N, D],带梯度

    offset = int(num_observed_steps) * int(tokens_per_frame)
    n_tokens = int(z_hat.shape[2])
    if h_target.shape[1] < offset + n_tokens:
        raise ValueError(
            f"h_target tokens {h_target.shape[1]} < future window offset {offset} + {n_tokens}; "
            "cannot align contrastive rollout to the real future latent (fail-loud)."
        )
    h_future = h_target[:, offset : offset + n_tokens].detach()

    # Lazy import(losses 包较重)。score = cosine − recon,higher = better。
    from app.vjepa_cowa_world_model.losses.reward_selector import compute_world_model_latent_reward

    scores = compute_world_model_latent_reward(
        z_hat,
        h_future,
        normalize_reps=bool(config.loss.normalize_reps),
    )[
        "reward"
    ]  # [B, K+1]

    score_pos = scores[:, :1]  # [B, 1]
    score_neg = scores[:, 1:]  # [B, K]
    margin = float(wm.contrastive_margin)
    ranking_loss = F.relu(margin - score_pos + score_neg).mean()
    with torch.no_grad():
        ranking_acc = (score_pos > score_neg.amax(dim=1, keepdim=True)).float().mean()
        pos_neg_gap = (score_pos - score_neg.mean(dim=1, keepdim=True)).mean()

    return {"loss": ranking_loss, "ranking_acc": ranking_acc, "pos_neg_gap": pos_neg_gap}
