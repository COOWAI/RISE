# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
分割特征提取工具
"""

import math

import torch


def prepare_seg_features(
    context_clips,
    seg_targets,
    z_perceiver,
    seg_neck,
    tubelet_size,
    tokens_per_frame,
    device,
    mixed_precision,
    dtype,
    normalize_reps=False,
):
    """
    统一的前向特征提取函数。
    负责：Encoder -> Perceiver -> Reshape -> 提取 Query/Memory -> Neck -> 输出 Head 的输入特征

    Parameters
    ----------
    context_clips    : 输入视频片段
    seg_targets      : 分割目标
    z_perceiver      : Perceiver 输出
    seg_neck         : 分割 neck 模块
    tubelet_size     : Tubelet 大小
    tokens_per_frame : 每帧 token 数
    device           : 设备
    mixed_precision  : 是否使用混合精度
    dtype            : 数据类型

    Returns
    -------
    neck_out        : Neck 输出
    batched_targets : 批量目标
    valid_samples   : 有效样本数
    vis_meta        : 可视化元数据
    """
    # 记录用于可视化的元数据 (保留原始图片引用和对应索引)
    vis_meta = []

    with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
        # Reshape 逻辑 (统一在此处管理)
        B, Total_Latents, D = z_perceiver.shape
        num_frames_latent = Total_Latents // tokens_per_frame

        # Encoder Output Reshape
        enc_tokens = z_perceiver.shape[1]
        enc_dim = z_perceiver.shape[-1]
        spatial_tokens = enc_tokens // num_frames_latent
        r_size_enc = int(math.sqrt(spatial_tokens))

        # [B, T_latent, H_feat, W_feat, D]
        z_perceiver_reshaped = z_perceiver.view(B, num_frames_latent, r_size_enc, r_size_enc, enc_dim)

        # 收集 Batch 数据
        batch_queries_list = []
        batch_targets_list = []

        for b in range(B):
            if seg_targets[b] is None:
                continue
            masks_k, indices_k = seg_targets[b]

            for k in range(len(indices_k)):
                t_frame = indices_k[k].item()
                t_latent = t_frame // tubelet_size

                if t_latent >= num_frames_latent:
                    continue

                # A. 提取 Perceiver Latent -> 作为 Query
                query_slice = z_perceiver_reshaped[b, t_latent].permute(2, 0, 1)

                # B. 收集
                batch_queries_list.append(query_slice)

                # C. Target 准备
                target_k = {
                    "labels": torch.ones(masks_k[k].shape[0], dtype=torch.long, device=device),
                    "masks": masks_k[k].float().to(device),
                }
                batch_targets_list.append(target_k)

                # D. 收集可视化元数据
                vis_meta.append(
                    {
                        "batch_idx": b,
                        "t_frame": t_frame,
                        "gt_mask": masks_k[k].cpu(),  # 此时先转 CPU 省显存
                        "img_tensor": context_clips[b, :, t_frame, :, :].cpu(),  # 原图
                    }
                )

        # 堆叠与过 Neck
        valid_samples = len(batch_queries_list)
        neck_out = None
        batched_targets = {}

        if valid_samples > 0:
            batched_queries = torch.stack(batch_queries_list)  # [N, D, H, W]

            # 堆叠 Targets
            example_target = batch_targets_list[0]
            for key in example_target.keys():
                values = [d[key] for d in batch_targets_list]
                if isinstance(values[0], torch.Tensor):
                    stacked_val = torch.stack(values)
                    if stacked_val.dim() == 4 and stacked_val.shape[1] == 1:
                        stacked_val = stacked_val.squeeze(1)
                    batched_targets[key] = stacked_val
                else:
                    batched_targets[key] = values

            # 进入 Neck
            neck_out = seg_neck(batched_queries)

    return neck_out, batched_targets, valid_samples, vis_meta
