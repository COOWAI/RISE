"""Shared encoder-input construction — single source for train == val == eval == viz.

The V-JEPA encoder consumes per-frame tubelet clips. Seven call sites build this input from a
``[B, C, T, H, W]`` clip in exactly the same way: the encoder-direct / main-encoder / lewm-stage
/ forward runtimes (training), ``val_command`` (validation), ``rl_policy`` (RL rollout), and the
visualization data path. Centralising it here means the encoder preprocessing cannot silently
diverge between training and inference/visualization — the consistency this module exists to
guarantee.
"""

import torch


def build_tubelet_encoder_input(context_clips: torch.Tensor, tubelet_size: int = 2) -> torch.Tensor:
    """``[B, C, T, H, W] -> [B*T, C, tubelet_size, H, W]`` (each frame repeated ``tubelet_size`` times).

    Mirrors the per-frame tubelet input the encoder expects. ``tubelet_size`` defaults to 2 — the
    value every training/val/viz call site historically hardcoded; pass the configured
    ``tubelet_size`` explicitly where it is available (e.g. the RL rollout already does).
    """
    return context_clips.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1)
