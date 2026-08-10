"""
Token Autoencoder: Deterministic autoencoder for compressing per-frame patch tokens.

Compresses the spatial tokens output by a frozen ViT encoder to a smaller
set of latent tokens per frame (e.g., 256 -> 144), reducing sequence length
for downstream modules.

Architecture (Perceiver / Q-Former style):
    Encoder: learnable queries + 2D sincos pos embed cross-attend to input -> latent tokens
    Decoder: reconstruction queries + 2D sincos pos embed cross-attend to latent -> reconstructed tokens

Operates per-frame by default, with an optional causal temporal latent mixer.

Key design choices:
    - Deterministic (no KL / reparameterization) -- avoids information destruction
    - 2D sincos positional embeddings on all queries and inputs -- provides spatial awareness
      so that each latent token "knows" which spatial region it should attend to

Dimensions:
    input tokens   : [B, T * tokens_per_frame, D]   (D = 1408 for ViT-Giant)
    latent tokens  : [B, T * num_latent_tokens, D]
    output tokens  : [B, T * tokens_per_frame, D]
"""

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.pos_embs import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed


def _normalize_grid_size(grid_size) -> Tuple[int, int]:
    if isinstance(grid_size, int):
        size = int(grid_size)
        return size, size
    if isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
        return int(grid_size[0]), int(grid_size[1])
    raise ValueError(f"grid_size must be an int or a 2-element sequence, got {grid_size!r}")


def _infer_grid_size(num_tokens: int, grid_size=None, name: str = "tokens") -> Tuple[int, int]:
    if grid_size is not None:
        height, width = _normalize_grid_size(grid_size)
        if height * width != num_tokens:
            raise ValueError(f"{name} grid_size={height}x{width} does not match token count {num_tokens}")
        return height, width

    height = int(math.isqrt(num_tokens))
    while height > 1 and num_tokens % height != 0:
        height -= 1
    width = num_tokens // height
    return height, width


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention + FFN block.

    query attends to key/value, with pre-norm (LayerNorm).
    Uses F.scaled_dot_product_attention for Flash/Memory-efficient attention.

    Supports optional K/V caching for parallel mode efficiency:
    - Pre-compute K/V from input tokens once
    - Reuse across multiple cross-attention layers

    Parameters
    ----------
    embed_dim   : int   hidden dimension
    num_heads   : int   number of attention heads
    mlp_ratio   : float FFN expansion ratio
    dropout     : float dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # Separate projections for Q, K, V (more efficient than nn.MultiheadAttention)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def compute_kv(self, kv: torch.Tensor) -> tuple:
        """Pre-compute K and V tensors for caching.

        Parameters
        ----------
        kv : [B, N_kv, D]

        Returns
        -------
        tuple: (k, v) each [B, num_heads, N_kv, head_dim]
        """
        B, N_kv, _ = kv.shape
        kv_normed = self.norm_kv(kv)
        k = self.k_proj(kv_normed).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_normed).view(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        cached_kv: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query     : [B, N_q, D]
        kv        : [B, N_kv, D]  ignored if cached_kv is provided
        cached_kv : tuple of (k, v) pre-computed tensors, optional

        Returns
        -------
        [B, N_q, D]
        """
        B, N_q, _ = query.shape

        # Pre-norm query and project
        q_normed = self.norm_q(query)
        q = self.q_proj(q_normed).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)

        # Use cached K/V or compute fresh
        if cached_kv is not None:
            k, v = cached_kv
        else:
            k, v = self.compute_kv(kv)

        # SDPA (Flash Attention / Memory Efficient Attention)
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(B, N_q, self.embed_dim)
        attn_out = self.out_proj(attn_out)

        # Residual
        query = query + attn_out

        # FFN with pre-norm
        query = query + self.ffn(self.norm_ffn(query))
        return query


class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention + FFN block with pre-norm.

    Uses F.scaled_dot_product_attention for Flash/Memory-efficient attention.
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.norm = nn.LayerNorm(embed_dim)

        # Fused QKV projection for self-attention efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, N, D]

        Returns
        -------
        [B, N, D]
        """
        B, N, _ = x.shape

        # Pre-norm
        normed = self.norm(x)

        # Fused QKV projection
        qkv = self.qkv_proj(normed).view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SDPA (Flash Attention / Memory Efficient Attention)
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        # Reshape and project output
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.embed_dim)
        attn_out = self.out_proj(attn_out)

        # Residual + FFN
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class TemporalSelfAttentionBlock(nn.Module):
    """Causal temporal self-attention over latent tokens.

    The block keeps the latent-token index fixed and mixes information only
    across time: [B, T, L, D] -> [B, T, L, D].  With causal=True, frame t can
    attend only to frames <= t, so compressed tokens do not leak future context.
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal

        self.norm = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def _causal_mask(self, num_frames: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.zeros(num_frames, num_frames, device=device, dtype=dtype)
        future_mask = torch.triu(
            torch.ones(num_frames, num_frames, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask.masked_fill(future_mask, float("-inf"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T, L, D]

        Returns
        -------
        [B, T, L, D]
        """
        B, T, L, D = x.shape
        x_time = x.permute(0, 2, 1, 3).reshape(B * L, T, D)

        normed = self.norm(x_time)
        qkv = self.qkv_proj(normed).view(B * L, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if self.causal and T > 1:
            attn_mask = self._causal_mask(T, x.device, q.dtype)
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(B * L, T, D)
        attn_out = self.out_proj(attn_out)

        x_time = x_time + attn_out
        x_time = x_time + self.ffn(self.norm_ffn(x_time))
        return x_time.view(B, L, T, D).permute(0, 2, 1, 3)


class BlockCausalTemporalSelfAttentionBlock(nn.Module):
    """Block-causal temporal self-attention over all latent tokens."""

    def __init__(
        self,
        embed_dim: int = 1408,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.causal = causal

        self.norm = nn.LayerNorm(embed_dim)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm_ffn = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def _block_causal_mask(
        self,
        num_frames: int,
        num_latents: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        frame_idx = torch.arange(num_frames, device=device).repeat_interleave(num_latents)
        future_mask = frame_idx[:, None] < frame_idx[None, :]
        mask = torch.zeros(num_frames * num_latents, num_frames * num_latents, device=device, dtype=dtype)
        return mask.masked_fill(future_mask, float("-inf"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply block-causal temporal mixing.

        Parameters
        ----------
        x : [B, T, L, D]

        Returns
        -------
        [B, T, L, D]
        """
        batch_size, num_frames, num_latents, dim = x.shape
        x_seq = x.reshape(batch_size, num_frames * num_latents, dim)

        normed = self.norm(x_seq)
        qkv = self.qkv_proj(normed).view(batch_size, num_frames * num_latents, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = None
        if self.causal and num_frames > 1:
            attn_mask = self._block_causal_mask(num_frames, num_latents, x.device, q.dtype)
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_frames * num_latents, dim)
        attn_out = self.out_proj(attn_out)

        x_seq = x_seq + attn_out
        x_seq = x_seq + self.ffn(self.norm_ffn(x_seq))
        return x_seq.view(batch_size, num_frames, num_latents, dim)


class SpatialPoolingModule(nn.Module):
    """Swin-style spatial patch merging for token compression.

    Groups adjacent spatial patches (e.g., 2×2) and projects the concatenated
    features down to the original embedding dimension. Preserves local spatial
    structure unlike global cross-attention.

    Example: 16×16=256 tokens → 8×8=64 tokens with h_factor=w_factor=2.

    Parameters
    ----------
    embed_dim        : int   input/output embedding dimension
    input_grid_size  : int or tuple   spatial grid size (H, W)
    target_grid_size : int or tuple   target spatial grid size (H, W)
    dropout          : float dropout rate
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        input_grid_size: Any = 16,
        target_grid_size: Any = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        input_grid_height, input_grid_width = _normalize_grid_size(input_grid_size)
        target_grid_height, target_grid_width = _normalize_grid_size(target_grid_size)
        if not (input_grid_height % target_grid_height == 0):
            raise AssertionError(
                f"input_grid_height={input_grid_height} must be divisible by "
                f"target_grid_height={target_grid_height}"
            )
        if not (input_grid_width % target_grid_width == 0):
            raise AssertionError(
                f"input_grid_width={input_grid_width} must be divisible by " f"target_grid_width={target_grid_width}"
            )
        self.input_grid_size = (input_grid_height, input_grid_width)
        self.target_grid_size = (target_grid_height, target_grid_width)
        self.embed_dim = embed_dim

        self.h_factor = input_grid_height // target_grid_height
        self.w_factor = input_grid_width // target_grid_width
        self.merge_factor = self.h_factor * self.w_factor  # e.g. 2*2=4

        merge_dim = self.merge_factor * embed_dim
        hidden_dim = merge_dim // 2

        # Swin-style merging: concat neighbors → project down
        self.norm = nn.LayerNorm(merge_dim)
        self.pre_proj = nn.Linear(merge_dim, hidden_dim, bias=False)
        self.pre_proj_norm = nn.LayerNorm(hidden_dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_proj = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.residual_proj = nn.Linear(merge_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool spatial tokens via patch merging.

        Parameters
        ----------
        x : [B, H_in * W_in, D]
            Input tokens on a square spatial grid.

        Returns
        -------
        [B, H_out * W_out, D]  pooled tokens (same embed_dim)
        """
        B, N, D = x.shape
        H_in, W_in = self.input_grid_size
        H_out, W_out = self.target_grid_size

        if not (N == H_in * W_in):
            raise AssertionError(f"Token count {N} != expected {H_in * W_in}")

        # [B, H_out, h_f, W_out, w_f, D] → group neighbors
        x = x.view(B, H_out, self.h_factor, W_out, self.w_factor, D)
        # [B, H_out, W_out, h_f, w_f, D] → [B, H_out*W_out, merge_factor*D]
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H_out * W_out, self.merge_factor * D)

        # Residual path
        residual = self.residual_proj(x)

        # Main path: norm → project → channel mix → project
        x = self.norm(x)
        x = self.pre_proj(x)
        x = x + self.channel_mixer(self.pre_proj_norm(x))
        x = self.output_proj(x)

        return x + residual


class PruMergePooling(nn.Module):
    """PruMerge-style adaptive token reduction via importance-based pruning and merging.

    Reference:
        LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models
        Shang et al., ICCV 2025
        https://arxiv.org/abs/2403.15388

    This module implements a two-stage token compression strategy:
        1. Prune: Select top-k important tokens based on learned importance scores
        2. Merge: Aggregate pruned tokens into their most similar retained tokens

    Since V-JEPA 2 encoder does not have a CLS token, we use a learnable importance
    predictor instead of CLS attention scores as in the original paper.

    Key adaptations for V-JEPA 2:
        - Learnable importance head replaces CLS attention (no CLS token in V-JEPA 2)
        - Soft merging preserves gradient flow during training
        - Spatial-aware importance using 2D positional information

    Parameters
    ----------
    embed_dim        : int   token embedding dimension (1408 for ViT-Giant)
    input_tokens     : int   number of input tokens (e.g., 256 = 16x16)
    output_tokens    : int   number of output tokens (e.g., 64 = 8x8)
    num_heads        : int   attention heads for importance estimation (default: 8)
    temperature      : float softmax temperature for soft assignment (default: 0.1)
    dropout          : float dropout rate (default: 0.0)
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        input_tokens: int = 256,
        output_tokens: int = 64,
        num_heads: int = 8,
        temperature: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.temperature = temperature

        # Learnable importance predictor
        # Replaces CLS attention in original PruMerge since V-JEPA 2 has no CLS token
        self.importance_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 4, 1),
        )

        # Feature refinement after merging
        self.merge_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Output projection with residual
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply PruMerge compression to input tokens.

        Parameters
        ----------
        x : [B, N, D]  input tokens where N = input_tokens

        Returns
        -------
        [B, K, D]  compressed tokens where K = output_tokens
        """
        B, N, D = x.shape
        K = self.output_tokens

        if not (N == self.input_tokens):
            raise AssertionError(f"Expected {self.input_tokens} tokens, got {N}")
        if not (K <= N):
            raise AssertionError(f"output_tokens {K} must be <= input_tokens {N}")

        # ============================================================
        # Stage 1: PRUNE - Select top-K important tokens
        # ============================================================

        # Compute importance scores for each token
        # [B, N, 1] -> [B, N]
        importance = self.importance_head(x).squeeze(-1)

        # Select top-K tokens by importance
        topk_indices = torch.topk(importance, K, dim=-1).indices  # [B, K]
        topk_indices = torch.sort(topk_indices, dim=-1).values
        topk_values = torch.gather(importance, dim=1, index=topk_indices)  # [B, K]

        # Gather retained tokens
        # [B, K, D]
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        retained = torch.gather(x, dim=1, index=topk_indices_expanded)
        # Keep importance prediction on the differentiable path. If routing only
        # depends on top-k indices, the scorer never receives gradients under DDP.
        retained = retained * (1.0 + 0.1 * torch.tanh(topk_values).unsqueeze(-1))

        # ============================================================
        # Stage 2: MERGE - Aggregate pruned tokens into retained ones
        # ============================================================

        # Create mask for pruned tokens
        # [B, N]
        # Use scatter to create mask: 1 for retained, 0 for pruned
        retained_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        retained_mask.scatter_(1, topk_indices, True)
        pruned_mask = ~retained_mask

        # Get pruned tokens
        # We need to handle variable number of pruned tokens per batch
        # For efficiency, we compute soft assignment for ALL tokens to retained tokens
        # then mask out the retained tokens' self-contribution

        # Compute similarity between all tokens and retained tokens
        # [B, N, D] @ [B, D, K] -> [B, N, K]
        x_norm = F.normalize(x, dim=-1)
        retained_norm = F.normalize(retained, dim=-1)
        similarity = torch.bmm(x_norm, retained_norm.transpose(1, 2))

        # Soft assignment with temperature
        # [B, N, K]
        soft_assign = F.softmax(similarity / self.temperature, dim=-1)

        # Mask out retained tokens (they don't contribute to merging)
        # [B, N, 1]
        pruned_weight = pruned_mask.float().unsqueeze(-1)

        # Weighted aggregation: each retained token aggregates from pruned tokens
        # [B, K, N] @ [B, N, D] -> [B, K, D]
        # Weight by soft_assign (transposed) and pruned_weight
        weighted_assign = soft_assign.transpose(1, 2) * pruned_weight.transpose(1, 2)  # [B, K, N]

        # Normalize by sum of weights (avoid division by zero)
        assign_sum = weighted_assign.sum(dim=-1, keepdim=True).clamp(min=1e-6)  # [B, K, 1]
        weighted_assign = weighted_assign / assign_sum

        # Aggregate pruned tokens
        merged_contribution = torch.bmm(weighted_assign, x)  # [B, K, D]

        # ============================================================
        # Stage 3: Combine retained and merged contributions
        # ============================================================

        # Residual combination with projection
        merged = retained + self.merge_proj(merged_contribution)
        output = self.output_norm(merged)

        return output


class TokenAEEncoder(nn.Module):
    """Compress per-frame tokens via learnable queries + cross-attention -> latent tokens.

    Positional embeddings are added to both input tokens and latent queries so
    that each latent query is spatially aware of which region it should attend to.

    Position encoding note:
        The upstream V-JEPA 2 encoder uses **3D RoPE** (rotary positional embedding
        inside attention), NOT additive sincos.  This means encoder output tokens
        do not carry explicit additive positional information — spatial awareness
        is conveyed implicitly through attention patterns.  Therefore this module
        adds its own positional signal via ``pos_embed_type``:

        - ``"sincos"`` (default): Fixed 2D sincos.  Robust, zero extra parameters.
        - ``"learnable"``: Learnable positional embeddings that can adapt to the
          encoder's RoPE feature distribution.  May improve alignment at the cost
          of (num_input_tokens + num_latent_tokens) × embed_dim extra parameters.

    Supports four encoder modes via ``encoder_mode``:

    - ``"query"`` (default): Learnable queries attend to input via cross-attention.
    - ``"serial"``: Spatial pooling first → use pooled tokens as initial queries
      → cross-attention + self-attention refinement. (串联)
    - ``"parallel"``: Spatial pooling branch runs in parallel with the learnable
      query branch → outputs are summed (residual / 并联).
    - ``"prumerge"``: PruMerge-style two-stage compression (Prune + Merge).
      Reference: LLaVA-PruMerge (Shang et al., ICCV 2025)

    Parameters
    ----------
    embed_dim         : int   token embedding dimension (1408 for ViT-Giant)
    tokens_per_frame  : int   number of input tokens per frame (e.g., 256 = 16x16)
    num_latent_tokens : int   number of compressed tokens per frame (e.g., 64 = 8x8)
    num_heads         : int   attention heads
    depth             : int   number of cross-attention layers
    mlp_ratio         : float FFN hidden dim ratio
    dropout           : float dropout rate
    encoder_mode      : str   one of "query", "serial", "parallel", "prumerge"
    pos_embed_type    : str   "sincos" (fixed 2D sincos) or "learnable" (trainable)
    input_grid_size   : tuple optional explicit input grid (H, W)
    latent_grid_size  : tuple optional explicit latent grid (H, W)
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        tokens_per_frame: int = 256,
        num_latent_tokens: int = 144,
        num_heads: int = 16,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        encoder_mode: str = "query",
        pos_embed_type: str = "sincos",
        input_grid_size=None,
        latent_grid_size=None,
    ):
        super().__init__()
        self.num_latent_tokens = num_latent_tokens
        self.tokens_per_frame = tokens_per_frame
        self.embed_dim = embed_dim
        self.encoder_mode = encoder_mode
        self.pos_embed_type = pos_embed_type

        if not (
            encoder_mode
            in (
                "query",
                "serial",
                "parallel",
                "prumerge",
            )
        ):
            raise AssertionError(
                f"encoder_mode must be 'query', 'serial', 'parallel', or 'prumerge', got '{encoder_mode}'"
            )

        input_grid_size = _infer_grid_size(tokens_per_frame, input_grid_size, "tokens_per_frame")
        latent_grid_size = _infer_grid_size(num_latent_tokens, latent_grid_size, "num_latent_tokens")
        self.input_grid_size = input_grid_size
        self.latent_grid_size = latent_grid_size

        # -- Spatial pooling module (used in serial and parallel modes) --
        if encoder_mode in ("serial", "parallel"):
            self.spatial_pooling = SpatialPoolingModule(
                embed_dim=embed_dim,
                input_grid_size=input_grid_size,
                target_grid_size=latent_grid_size,
                dropout=dropout,
            )
        else:
            self.spatial_pooling = None

        # -- PruMerge pooling module (used in prumerge mode) --
        # Reference: LLaVA-PruMerge (Shang et al., ICCV 2025)
        # Two-stage compression: Prune top-K by importance + Merge rest to similar
        if encoder_mode == "prumerge":
            self.prumerge_pooling = PruMergePooling(
                embed_dim=embed_dim,
                input_tokens=tokens_per_frame,
                output_tokens=num_latent_tokens,
                num_heads=num_heads,
                temperature=0.1,
                dropout=dropout,
            )
        else:
            self.prumerge_pooling = None

        # -- Learnable query tokens (used in query and parallel modes) --
        if encoder_mode in ("query", "parallel"):
            self.latent_queries = nn.Parameter(torch.randn(1, num_latent_tokens, embed_dim) * 0.02)
        else:
            self.latent_queries = None

        # Cross-attention layers: queries attend to input frame tokens
        # (not used in prumerge mode, but kept for consistency)
        if encoder_mode != "prumerge":
            self.cross_attn_layers = nn.ModuleList(
                [CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
            )
            # Self-attention layers for query refinement
            self.self_attn_layers = nn.ModuleList(
                [SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
            )
        else:
            self.cross_attn_layers = None
            self.self_attn_layers = None

        # Output normalization
        self.norm_out = nn.LayerNorm(embed_dim)

        # Parallel mode: additional norm for pooling branch before summation
        if encoder_mode == "parallel":
            self.norm_pool = nn.LayerNorm(embed_dim)
        else:
            self.norm_pool = None

        # -- Positional embeddings --
        # V-JEPA 2 encoder uses 3D RoPE (non-additive, rotary).  Output tokens have
        # implicit spatial structure but no explicit additive position signal.
        # "sincos": fixed 2D sincos — robust baseline, zero trainable params.
        # "learnable": trainable embeddings — can adapt to RoPE feature distribution.
        if not (
            pos_embed_type
            in (
                "sincos",
                "learnable",
            )
        ):
            raise AssertionError(f"pos_embed_type must be 'sincos' or 'learnable', got '{pos_embed_type}'")
        if pos_embed_type == "sincos":
            input_pos = get_2d_sincos_pos_embed(embed_dim, input_grid_size)  # [P, D] numpy
            self.register_buffer("input_pos_embed", torch.from_numpy(input_pos).float().unsqueeze(0))
            if encoder_mode == "prumerge":
                self.register_buffer("latent_pos_embed", torch.zeros(1, num_latent_tokens, embed_dim))
            else:
                latent_pos = get_2d_sincos_pos_embed(embed_dim, latent_grid_size)  # [L, D] numpy
                self.register_buffer("latent_pos_embed", torch.from_numpy(latent_pos).float().unsqueeze(0))
        else:  # learnable
            self.input_pos_embed = nn.Parameter(torch.randn(1, tokens_per_frame, embed_dim) * 0.02)
            if encoder_mode == "prumerge":
                self.register_buffer("latent_pos_embed", torch.zeros(1, num_latent_tokens, embed_dim))
            else:
                self.latent_pos_embed = nn.Parameter(torch.randn(1, num_latent_tokens, embed_dim) * 0.02)

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        frame_tokens : [B_frames, tokens_per_frame, D]
            Input tokens for a batch of individual frames.

        Returns
        -------
        z : [B_frames, num_latent_tokens, D]  compressed latent tokens
        """
        Bf = frame_tokens.shape[0]

        # Add 2D positional embedding to input tokens
        frame_tokens_pos = frame_tokens + self.input_pos_embed.to(frame_tokens.dtype)

        if self.encoder_mode == "serial":
            # --- Serial (串联): pooling → use as queries → cross-attn + self-attn ---
            if not (self.spatial_pooling is not None):
                raise AssertionError("assertion failed: self.spatial_pooling is not None")
            if not (self.cross_attn_layers is not None and self.self_attn_layers is not None):
                raise AssertionError(
                    "assertion failed: self.cross_attn_layers is not None and self.self_attn_layers is not None"
                )
            # Step 1: spatial pooling on raw tokens (before pos embed, to preserve grid)
            pooled = self.spatial_pooling(frame_tokens)  # [Bf, L, D]
            # Step 2: use pooled tokens as initial queries (+ latent pos embed)
            queries = pooled + self.latent_pos_embed.to(pooled.dtype)
            # Step 3: cross-attention refinement against pos-embedded input
            for cross_layer, self_layer in zip(self.cross_attn_layers, self.self_attn_layers):
                queries = cross_layer(queries, frame_tokens_pos)
                queries = self_layer(queries)
            z = self.norm_out(queries)

        elif self.encoder_mode == "parallel":
            # --- Parallel (并联): pooling branch ∥ query branch → sum ---
            # --- With K/V caching: pre-compute K/V once, reuse across layers ---
            if not (self.spatial_pooling is not None):
                raise AssertionError("assertion failed: self.spatial_pooling is not None")
            if not (self.norm_pool is not None):
                raise AssertionError("assertion failed: self.norm_pool is not None")
            if not (self.latent_queries is not None):
                raise AssertionError("assertion failed: self.latent_queries is not None")
            if not (self.cross_attn_layers is not None and self.self_attn_layers is not None):
                raise AssertionError(
                    "assertion failed: self.cross_attn_layers is not None and self.self_attn_layers is not None"
                )

            # Branch A: spatial pooling (no attention, fast)
            pooled = self.spatial_pooling(frame_tokens)  # [Bf, L, D]
            pooled = self.norm_pool(pooled)

            # Branch B: learnable queries + cross-attn with K/V caching
            queries = self.latent_queries.expand(Bf, -1, -1)  # [Bf, L, D]
            queries = queries + self.latent_pos_embed.to(queries.dtype)

            # Pre-compute K/V for each layer (each layer has its own norm_kv and projections)
            cached_kvs = []
            for layer in self.cross_attn_layers:
                if not (isinstance(layer, CrossAttentionBlock)):
                    raise AssertionError("assertion failed: isinstance(layer, CrossAttentionBlock)")
                cached_kvs.append(layer.compute_kv(frame_tokens_pos))

            # Cross-attention with cached K/V + self-attention
            for cross_layer, self_layer, cached_kv in zip(self.cross_attn_layers, self.self_attn_layers, cached_kvs):
                queries = cross_layer(queries, frame_tokens_pos, cached_kv=cached_kv)
                queries = self_layer(queries)

            queries = self.norm_out(queries)

            # Residual combination
            z = queries + pooled

        elif self.encoder_mode == "prumerge":
            # --- PruMerge mode (ICCV 2025) ---
            if not (self.prumerge_pooling is not None):
                raise AssertionError("assertion failed: self.prumerge_pooling is not None")
            # Reference: LLaVA-PruMerge (Shang et al., ICCV 2025)
            # Two-stage compression: Prune top-K by importance + Merge rest to similar
            #
            # Unlike parallel mode which uses fixed spatial pooling, PruMerge
            # dynamically selects important tokens and merges the rest.
            # This is more adaptive to content and preserves important details.
            #
            # NOTE: We feed pos-embedded tokens so the importance head has spatial
            # context. Selected tokens inherit their original spatial positions.
            # We do NOT add latent_pos_embed here because PruMerge output tokens
            # are ordered by importance, not in a regular spatial grid — a fixed
            # 8×8 sincos would assign wrong positions.
            z = self.prumerge_pooling(frame_tokens_pos)  # [Bf, L, D]
            z = self.norm_out(z)

        else:
            # --- Query mode (default, original behavior) ---
            if not (self.latent_queries is not None):
                raise AssertionError("assertion failed: self.latent_queries is not None")
            if not (self.cross_attn_layers is not None and self.self_attn_layers is not None):
                raise AssertionError(
                    "assertion failed: self.cross_attn_layers is not None and self.self_attn_layers is not None"
                )
            queries = self.latent_queries.expand(Bf, -1, -1)  # [Bf, L, D]
            queries = queries + self.latent_pos_embed.to(queries.dtype)
            for cross_layer, self_layer in zip(self.cross_attn_layers, self.self_attn_layers):
                queries = cross_layer(queries, frame_tokens_pos)
                queries = self_layer(queries)
            z = self.norm_out(queries)

        return z


class TokenAEDecoder(nn.Module):
    """Reconstruct per-frame tokens from compressed latent tokens via cross-attention.

    Positional embeddings are added to both latent input and reconstruction
    queries to maintain spatial correspondence.

    Parameters
    ----------
    embed_dim         : int   token embedding dimension
    tokens_per_frame  : int   number of original tokens per frame to reconstruct (256)
    num_latent_tokens : int   number of latent tokens per frame (144)
    num_heads         : int   attention heads
    depth             : int   number of cross-attention layers
    mlp_ratio         : float FFN hidden dim ratio
    dropout           : float dropout rate
    pos_embed_type    : str   "sincos" (fixed 2D sincos) or "learnable" (trainable)
    input_grid_size   : tuple optional explicit reconstruction grid (H, W)
    latent_grid_size  : tuple optional explicit latent grid (H, W)
    latent_pos_embed_type : str "spatial" or "none"
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        tokens_per_frame: int = 256,
        num_latent_tokens: int = 144,
        num_heads: int = 16,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pos_embed_type: str = "sincos",
        input_grid_size=None,
        latent_grid_size=None,
        latent_pos_embed_type: str = "spatial",
    ):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.num_latent_tokens = num_latent_tokens
        self.embed_dim = embed_dim
        self.latent_pos_embed_type = latent_pos_embed_type

        # Learnable reconstruction queries (expand latent -> original token count)
        self.recon_queries = nn.Parameter(torch.randn(1, tokens_per_frame, embed_dim) * 0.02)

        # Cross-attention layers: recon queries attend to latent tokens
        self.cross_attn_layers = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )

        # Self-attention layers for reconstruction refinement
        self.self_attn_layers = nn.ModuleList(
            [SelfAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )

        # -- Positional embeddings (consistent with encoder's pos_embed_type) --
        recon_grid_size = _infer_grid_size(tokens_per_frame, input_grid_size, "tokens_per_frame")
        latent_grid_size = _infer_grid_size(num_latent_tokens, latent_grid_size, "num_latent_tokens")
        if not (
            latent_pos_embed_type
            in (
                "spatial",
                "none",
            )
        ):
            raise AssertionError(f"latent_pos_embed_type must be 'spatial' or 'none', got '{latent_pos_embed_type}'")

        if pos_embed_type == "sincos":
            recon_pos = get_2d_sincos_pos_embed(embed_dim, recon_grid_size)
            self.register_buffer("recon_pos_embed", torch.from_numpy(recon_pos).float().unsqueeze(0))
            if latent_pos_embed_type == "none":
                self.register_buffer("latent_pos_embed", torch.zeros(1, num_latent_tokens, embed_dim))
            else:
                latent_pos = get_2d_sincos_pos_embed(embed_dim, latent_grid_size)
                self.register_buffer("latent_pos_embed", torch.from_numpy(latent_pos).float().unsqueeze(0))
        else:  # learnable
            self.recon_pos_embed = nn.Parameter(torch.randn(1, tokens_per_frame, embed_dim) * 0.02)
            if latent_pos_embed_type == "none":
                self.register_buffer("latent_pos_embed", torch.zeros(1, num_latent_tokens, embed_dim))
            else:
                self.latent_pos_embed = nn.Parameter(torch.randn(1, num_latent_tokens, embed_dim) * 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : [B_frames, num_latent_tokens, D]  latent tokens from encoder

        Returns
        -------
        recon : [B_frames, tokens_per_frame, D]  reconstructed frame tokens
        """
        Bf = z.shape[0]

        # Add 2D positional embedding to latent input
        z = z + self.latent_pos_embed.to(z.dtype)

        # Expand learnable reconstruction queries + add 2D positional embedding
        queries = self.recon_queries.expand(Bf, -1, -1)  # [Bf, P, D]
        queries = queries + self.recon_pos_embed.to(queries.dtype)

        # Cross-attend to latent tokens + self-attend
        for cross_layer, self_layer in zip(self.cross_attn_layers, self.self_attn_layers):
            queries = cross_layer(queries, z)  # attend to latent
            queries = self_layer(queries)  # refine

        recon = queries  # [Bf, P, D]
        return recon


class TokenAE(nn.Module):
    """Deterministic token autoencoder for per-frame token compression.

    Compresses encoder tokens from ``tokens_per_frame`` (256) to
    ``num_latent_tokens`` (e.g., 144) per frame, then reconstructs them.

    Designed to sit between the frozen ViT encoder and downstream modules.

    Key features:
        - Deterministic (no VAE sampling/KL) -- preserves information faithfully
        - Configurable positional embeddings (sincos or learnable)
        - Perceiver-style cross-attention architecture

    Parameters
    ----------
    embed_dim         : int   encoder embedding dimension (1408 for ViT-Giant)
    tokens_per_frame  : int   number of spatial tokens per frame from encoder (256)
    num_latent_tokens : int   compressed token count per frame (144)
    num_heads         : int   attention heads for all attention layers
    encoder_depth     : int   number of layers in the autoencoder encoder
    decoder_depth     : int   number of layers in the autoencoder decoder
    mlp_ratio         : float FFN expansion ratio
    dropout           : float dropout rate
    loss_type         : str   "mse" or "smooth_l1" (huber loss, more robust to outliers)
    cos_loss_weight   : float weight for cosine similarity loss (default 0.25)
    latent_reg_weight : float weight for latent norm regularization (0 = disabled)
    pos_embed_type    : str   "sincos" (fixed 2D) or "learnable" (trainable, better for RoPE encoder)
    temporal_depth    : int   number of causal temporal latent self-attention layers
    temporal_mode     : str   "index" keeps latent indices isolated; "block_causal"
                            mixes all latent tokens with a block-causal mask
    temporal_pos_embed_type : str   "none" or "sincos" 1D temporal position encoding
    temporal_loss_weight : float weight for frame-delta reconstruction loss
    """

    def __init__(
        self,
        embed_dim: int = 1408,
        tokens_per_frame: int = 256,
        num_latent_tokens: int = 144,
        num_heads: int = 16,
        encoder_depth: int = 4,
        decoder_depth: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        encoder_mode: str = "query",
        loss_type: str = "smooth_l1",
        cos_loss_weight: float = 0.25,
        latent_reg_weight: float = 0.0,
        pos_embed_type: str = "sincos",
        input_grid_size=None,
        latent_grid_size=None,
        temporal_depth: int = 0,
        temporal_num_heads: Optional[int] = None,
        temporal_mlp_ratio: Optional[float] = None,
        temporal_causal: bool = True,
        temporal_mode: str = "index",
        temporal_pos_embed_type: str = "none",
        temporal_loss_weight: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.tokens_per_frame = tokens_per_frame
        self.num_latent_tokens = num_latent_tokens
        self.encoder_mode = encoder_mode
        self.loss_type = loss_type
        self.cos_loss_weight = cos_loss_weight
        self.latent_reg_weight = latent_reg_weight
        self.temporal_depth = int(temporal_depth)
        self.temporal_causal = bool(temporal_causal)
        self.temporal_mode = temporal_mode
        self.temporal_pos_embed_type = temporal_pos_embed_type
        self.temporal_loss_weight = float(temporal_loss_weight)
        if not (
            self.temporal_pos_embed_type
            in (
                "none",
                "sincos",
            )
        ):
            raise AssertionError(
                f"temporal_pos_embed_type must be 'none' or 'sincos', got '{self.temporal_pos_embed_type}'"
            )

        self.encoder = TokenAEEncoder(
            embed_dim=embed_dim,
            tokens_per_frame=tokens_per_frame,
            num_latent_tokens=num_latent_tokens,
            num_heads=num_heads,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            encoder_mode=encoder_mode,
            pos_embed_type=pos_embed_type,
            input_grid_size=input_grid_size,
            latent_grid_size=latent_grid_size,
        )

        self.decoder = TokenAEDecoder(
            embed_dim=embed_dim,
            tokens_per_frame=tokens_per_frame,
            num_latent_tokens=num_latent_tokens,
            num_heads=num_heads,
            depth=decoder_depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            pos_embed_type=pos_embed_type,
            input_grid_size=input_grid_size,
            latent_grid_size=latent_grid_size,
            latent_pos_embed_type="none" if encoder_mode == "prumerge" else "spatial",
        )

        temporal_heads = int(temporal_num_heads) if temporal_num_heads is not None else num_heads
        temporal_ratio = float(temporal_mlp_ratio) if temporal_mlp_ratio is not None else mlp_ratio
        if self.temporal_depth > 0:
            if self.temporal_mode in ("index", "fixed_index"):
                temporal_block_cls = TemporalSelfAttentionBlock
            elif self.temporal_mode in ("block", "block_causal"):
                temporal_block_cls = BlockCausalTemporalSelfAttentionBlock
            else:
                raise ValueError(
                    f"Unsupported temporal_mode={self.temporal_mode!r}; expected 'index' or 'block_causal'"
                )
            self.temporal_layers = nn.ModuleList(
                [
                    temporal_block_cls(
                        embed_dim=embed_dim,
                        num_heads=temporal_heads,
                        mlp_ratio=temporal_ratio,
                        dropout=dropout,
                        causal=self.temporal_causal,
                    )
                    for _ in range(self.temporal_depth)
                ]
            )
        else:
            self.temporal_layers = None

    def _apply_temporal(self, z_frames: torch.Tensor) -> torch.Tensor:
        if self.temporal_layers is None:
            return z_frames
        if self.temporal_pos_embed_type == "sincos":
            _, num_frames, _, _ = z_frames.shape
            temporal_pos = get_1d_sincos_pos_embed(self.embed_dim, num_frames)
            temporal_pos = torch.from_numpy(temporal_pos).to(device=z_frames.device, dtype=z_frames.dtype)
            z_frames = z_frames + temporal_pos.unsqueeze(0).unsqueeze(2)
        for layer in self.temporal_layers:
            z_frames = layer(z_frames)
        return z_frames

    def encode(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Encode multi-frame tokens to compressed latent tokens.

        Parameters
        ----------
        x          : [B, T * tokens_per_frame, D]  encoder output (all frames concatenated)
        num_frames : int  number of frames (T)

        Returns
        -------
        z : [B, T * num_latent_tokens, D]  compressed latent tokens
        """
        B = x.shape[0]
        P = self.tokens_per_frame
        L = self.num_latent_tokens
        D = self.embed_dim

        encoder_dtype = next(self.encoder.parameters()).dtype
        if x.dtype != encoder_dtype:
            x = x.to(dtype=encoder_dtype)

        # Reshape to per-frame: [B*T, P, D]
        x_frames = x.view(B, num_frames, P, D).reshape(B * num_frames, P, D)

        # Encode each frame independently
        z_frames = self.encoder(x_frames)
        # z_frames: [B*T, L, D]

        z_frames = z_frames.view(B, num_frames, L, D)
        z_frames = self._apply_temporal(z_frames)

        # Reshape back to sequence: [B, T*L, D]
        z = z_frames.flatten(1, 2)

        return z

    def decode(self, z: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Decode compressed latent tokens back to full frame tokens.

        Parameters
        ----------
        z          : [B, T * num_latent_tokens, D]  latent tokens
        num_frames : int  number of frames (T)

        Returns
        -------
        recon : [B, T * tokens_per_frame, D]  reconstructed tokens
        """
        decoder_dtype = next(self.decoder.parameters()).dtype
        if z.dtype != decoder_dtype:
            z = z.to(dtype=decoder_dtype)
        B = z.shape[0]
        L = self.num_latent_tokens
        P = self.tokens_per_frame
        D = self.embed_dim

        # Per-frame decoding: [B*T, L, D]
        z_frames = z.view(B, num_frames, L, D).reshape(B * num_frames, L, D)
        recon_frames = self.decoder(z_frames)  # [B*T, P, D]

        # Reshape back: [B, T*P, D]
        recon = recon_frames.view(B, num_frames, P, D).flatten(1, 2)
        return recon

    def forward(self, x: torch.Tensor, num_frames: int) -> Dict[str, torch.Tensor]:
        """Full forward pass: encode -> decode + compute losses.

        Parameters
        ----------
        x          : [B, T * tokens_per_frame, D]  encoder output
        num_frames : int  number of frames (T)

        Returns
        -------
        dict with keys:
            "z"           : [B, T * num_latent_tokens, D]   compressed tokens
            "recon"       : [B, T * tokens_per_frame, D]    reconstructed tokens
            "loss"        : scalar  total loss
            "recon_loss"  : scalar  reconstruction loss (smooth_l1 or mse)
            "cos_loss"    : scalar  cosine reconstruction loss
            "latent_reg"  : scalar  latent norm regularization (if enabled)
        """
        z = self.encode(x, num_frames)
        recon = self.decode(z, num_frames)

        # Compute losses in float32 for numerical stability (bfloat16-safe).
        recon_f = recon.float()
        x_f = x.float()
        z_f = z.float()

        # Reconstruction loss: smooth_l1 is more robust to outliers than MSE
        if self.loss_type == "smooth_l1":
            # Smooth L1 (Huber loss): linear for |error| > 1, quadratic for |error| < 1
            recon_loss = F.smooth_l1_loss(recon_f, x_f, beta=1.0)
        else:
            # MSE fallback
            recon_loss = F.mse_loss(recon_f, x_f)

        # Cosine similarity loss: preserves feature directions
        cos_loss = 1.0 - F.cosine_similarity(recon_f, x_f, dim=-1).mean()

        # Latent regularization: prevent z magnitude from growing unbounded
        # This is optional but useful for downstream modules that expect normalized features
        if self.latent_reg_weight > 0:
            # L2 norm regularization on z (encourage compact representation)
            latent_reg = (z_f.norm(dim=-1) ** 2).mean() * self.latent_reg_weight
        else:
            latent_reg = torch.zeros(1, device=z.device, dtype=torch.float32).squeeze()

        if self.temporal_loss_weight > 0 and num_frames > 1:
            B = x.shape[0]
            P = self.tokens_per_frame
            D = self.embed_dim
            recon_seq = recon_f.view(B, num_frames, P, D)
            target_seq = x_f.view(B, num_frames, P, D)
            temporal_loss = F.smooth_l1_loss(
                recon_seq[:, 1:] - recon_seq[:, :-1],
                target_seq[:, 1:] - target_seq[:, :-1],
                beta=1.0,
            )
        else:
            temporal_loss = torch.zeros(1, device=z.device, dtype=torch.float32).squeeze()

        # Total loss with configurable weights
        total_loss = recon_loss + self.cos_loss_weight * cos_loss + latent_reg
        total_loss = total_loss + self.temporal_loss_weight * temporal_loss

        return {
            "z": z,
            "recon": recon,
            "loss": total_loss,
            "recon_loss": recon_loss,
            "cos_loss": cos_loss,
            "latent_reg": latent_reg,
            "temporal_loss": temporal_loss,
        }

    def compress(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Compress tokens for inference (no reconstruction, no loss).

        Parameters
        ----------
        x          : [B, T * tokens_per_frame, D]
        num_frames : int

        Returns
        -------
        z : [B, T * num_latent_tokens, D]
        """
        return self.encode(x, num_frames)

    def extra_repr(self) -> str:
        return (
            f"tokens_per_frame={self.tokens_per_frame}, "
            f"num_latent_tokens={self.num_latent_tokens}, "
            f"embed_dim={self.embed_dim}, "
            f"encoder_mode={self.encoder_mode}, "
            f"compression_ratio={self.tokens_per_frame / self.num_latent_tokens:.1f}x, "
            f"loss_type={self.loss_type}, "
            f"cos_loss_weight={self.cos_loss_weight}, "
            f"latent_reg_weight={self.latent_reg_weight}, "
            f"temporal_depth={self.temporal_depth}, "
            f"temporal_mode={self.temporal_mode}, "
            f"temporal_pos_embed_type={self.temporal_pos_embed_type}, "
            f"temporal_causal={self.temporal_causal}, "
            f"temporal_loss_weight={self.temporal_loss_weight}"
        )
