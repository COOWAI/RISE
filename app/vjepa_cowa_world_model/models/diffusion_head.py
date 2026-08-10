# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
"""
DiT-based Diffusion Head with Flow Matching for validating the predict module.

Architecture:
- Input: noisy image patches (patchified via conv encoder)
- Condition: predictor output tokens [B, N, cond_dim]
- Output: velocity field prediction for flow matching
- Model: DiT (Diffusion Transformer) with cross-attention to condition

Flow Matching (Conditional Optimal Transport):
- Forward: x_t = (1-t) * x_0 + t * x_1, where x_0 ~ N(0,I), x_1 = data
- Velocity: v = x_1 - x_0
- Loss: ||v_theta(x_t, t, cond) - v||^2
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ======================== Modulation (adaLN-Zero) ========================
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """Embed scalar timestep into vector representation."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ======================== DiT Block ========================
class DiTBlock(nn.Module):
    """
    DiT block with adaLN-Zero conditioning and cross-attention to predictor tokens.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, cond_dim=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.0)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # Cross-attention to condition (predictor tokens)
        self.has_cross_attn = cond_dim is not None
        if self.has_cross_attn:
            self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.0)
            self.cond_proj = nn.Linear(cond_dim, hidden_size) if cond_dim != hidden_size else nn.Identity()
            self.norm_cond = nn.LayerNorm(hidden_size, eps=1e-6)

        # adaLN-Zero modulation: 6 params for self-attn and mlp, +3 for cross-attn
        num_ada_params = 9 if self.has_cross_attn else 6
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, num_ada_params * hidden_size, bias=True),
        )

    def forward(self, x, c, cond_tokens=None):
        """
        x: [B, N, D] - noisy image tokens
        c: [B, D] - timestep embedding
        cond_tokens: [B, M, cond_dim] - predictor output tokens (condition)
        """
        ada_params = self.adaLN_modulation(c).chunk(9 if self.has_cross_attn else 6, dim=1)

        if self.has_cross_attn:
            shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = ada_params
        else:
            shift_sa, scale_sa, gate_sa, shift_mlp, scale_mlp, gate_mlp = ada_params

        # Self-attention
        x_norm = modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_sa.unsqueeze(1) * attn_out

        # Cross-attention to condition
        if self.has_cross_attn and cond_tokens is not None:
            x_norm_ca = modulate(self.norm_cross(x), shift_ca, scale_ca)
            cond_kv = self.norm_cond(self.cond_proj(cond_tokens))
            ca_out, _ = self.cross_attn(x_norm_ca, cond_kv, cond_kv)
            x = x + gate_ca.unsqueeze(1) * ca_out

        # MLP
        x_norm_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm_mlp)

        return x


# ======================== Final Layer ========================
class FinalLayer(nn.Module):
    """Final layer of DiT with adaLN modulation."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ======================== DiT Flow Matching Model ========================
class DiTFlowMatching(nn.Module):
    """
    DiT-based flow matching model.

    Uses predictor output as conditioning to generate target images.
    Operates on single frames (selected from the target sequence).

    Args:
        img_size: input image size (256)
        patch_size: patch size for patchifying images (16 for consistency with ViT encoder)
        in_channels: input image channels (3 for RGB)
        hidden_size: DiT hidden dimension
        depth: number of DiT blocks
        num_heads: number of attention heads
        mlp_ratio: MLP hidden dim ratio
        cond_dim: dimension of conditioning tokens (predictor output dim, 1408 for vitg)
        cond_tokens_per_frame: number of conditioning tokens per frame (256 for 256px/16patch)
        use_activation_checkpointing: gradient checkpointing to save memory
    """

    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        cond_dim=1408,
        cond_tokens_per_frame=256,
        use_activation_checkpointing=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (img_size // patch_size) ** 2
        self.cond_tokens_per_frame = cond_tokens_per_frame
        self.use_activation_checkpointing = use_activation_checkpointing

        # Patch embedding for noisy images
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))

        # Timestep embedding
        self.t_embedder = TimestepEmbedder(hidden_size)

        # Frame index embedding (which target frame we're generating)
        self.frame_embedder = nn.Embedding(32, hidden_size)  # up to 32 frames

        # DiT blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio, cond_dim=cond_dim) for _ in range(depth)]
        )

        # Output projection: predict velocity field
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize pos_embed with sincos
        pos_embed = get_2d_sincos_pos_embed(self.hidden_size, int(self.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (xavier uniform)
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-init the final layer and adaLN modulation
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-init adaLN modulation in each block
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: [B, N, patch_size^2 * C]
        returns: [B, C, H, W]
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        c = self.in_channels
        x = x.reshape(-1, h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(-1, c, h * p, w * p)
        return x

    def forward(self, x_noisy, t, cond_tokens, frame_idx=None):
        """
        Forward pass: predict velocity field v(x_t, t, cond).

        Args:
            x_noisy: [B, C, H, W] noisy images at time t
            t: [B] timestep values in [0, 1]
            cond_tokens: [B, M, cond_dim] conditioning tokens from predictor
            frame_idx: [B] integer frame indices (which frame in the target seq)

        Returns:
            v_pred: [B, C, H, W] predicted velocity field
        """
        # Patchify noisy input
        x = self.x_embedder(x_noisy)  # [B, D, H/p, W/p]
        x = rearrange(x, "b d h w -> b (h w) d")  # [B, N, D]

        # Add positional embedding
        x = x + self.pos_embed

        # Timestep + frame conditioning
        t_emb = self.t_embedder(t)  # [B, D]
        if frame_idx is not None:
            t_emb = t_emb + self.frame_embedder(frame_idx)

        # DiT blocks with cross-attention to predictor tokens
        for block in self.blocks:
            if self.use_activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, t_emb, cond_tokens, use_reentrant=False)
            else:
                x = block(x, t_emb, cond_tokens)

        # Final layer: predict velocity
        x = self.final_layer(x, t_emb)  # [B, N, p*p*C]

        # Unpatchify
        v_pred = self.unpatchify(x)  # [B, C, H, W]

        return v_pred


# ======================== Flow Matching Utilities ========================
class FlowMatchingScheduler:
    """
    Conditional Optimal Transport flow matching scheduler.

    Forward process: x_t = (1 - t) * noise + t * data
    Target velocity: v = data - noise
    """

    @staticmethod
    def sample_timestep(batch_size, device, logit_normal=True):
        """
        Sample timestep t in [0, 1].
        Uses logit-normal distribution for better training (per Stable Diffusion 3).
        """
        if logit_normal:
            # Logit-normal: sample from N(0,1), apply sigmoid
            u = torch.randn(batch_size, device=device)
            t = torch.sigmoid(u)
        else:
            t = torch.rand(batch_size, device=device)
        # Clamp to avoid singularities at t=0 and t=1
        t = t.clamp(1e-5, 1.0 - 1e-5)
        return t

    @staticmethod
    def forward_process(x_1, t):
        """
        Interpolate between noise x_0 and data x_1.

        x_t = (1 - t) * x_0 + t * x_1

        Args:
            x_1: [B, C, H, W] clean data
            t: [B] timestep

        Returns:
            x_t: [B, C, H, W] noisy data
            x_0: [B, C, H, W] noise
            velocity: [B, C, H, W] target velocity (x_1 - x_0)
        """
        x_0 = torch.randn_like(x_1)
        t_expand = t[:, None, None, None]  # [B, 1, 1, 1]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        velocity = x_1 - x_0
        return x_t, x_0, velocity

    @staticmethod
    @torch.no_grad()
    def sample(model, cond_tokens, shape, device, num_steps=50, frame_idx=None, clamp_output=False):
        """
        Euler ODE solver for sampling.

        dx/dt = v_theta(x_t, t, cond)
        Integrate from t=0 to t=1.

        Args:
            model: DiTFlowMatching model
            cond_tokens: [B, M, D] conditioning tokens
            shape: (B, C, H, W) output shape
            device: device
            num_steps: number of integration steps
            frame_idx: [B] frame indices
            clamp_output: if True, clamp output to [-1, 1]. Set to False when
                the training data range exceeds [-1, 1] (e.g. ImageNet-normalized
                images with additional scaling).

        Returns:
            x_1: [B, C, H, W] generated images
        """
        model.eval()
        dt = 1.0 / num_steps
        x = torch.randn(shape, device=device)

        for i in range(num_steps):
            t_val = i / num_steps
            t = torch.full((shape[0],), t_val, device=device)
            v = model(x, t, cond_tokens, frame_idx=frame_idx)
            x = x + v * dt

        if clamp_output:
            x = x.clamp(-1, 1)
        return x


# ======================== Loss Function ========================
def flow_matching_loss(v_pred, v_target):
    """
    Simple MSE loss between predicted and target velocity fields.

    Args:
        v_pred: [B, C, H, W] predicted velocity
        v_target: [B, C, H, W] target velocity (x_1 - x_0)

    Returns:
        loss: scalar
    """
    return F.mse_loss(v_pred, v_target)


# ======================== Positional Embedding Utility ========================
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generate 2D sinusoidal positional embeddings.
    """
    import numpy as np

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # [2, H, W]
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])

    pos_embed = _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def _get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    import numpy as np

    if not (embed_dim % 2 == 0):
        raise AssertionError("assertion failed: embed_dim % 2 == 0")
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def _get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    import numpy as np

    if not (embed_dim % 2 == 0):
        raise AssertionError("assertion failed: embed_dim % 2 == 0")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
