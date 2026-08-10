"""PETR-style multi-view token fusion for NavSim cameras."""

from __future__ import annotations

import math
import numbers
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn


class PETRMultiViewFusion(nn.Module):
    """Fuse shared-encoder view tokens into the existing single-stream token API.

    Parameters
    ----------
    embed_dim : int
        V-JEPA token dimension.
    tokens_per_frame : int
        Number of spatial tokens per predictor time step.
    hidden_dim : int
        MLP hidden dimension for PETR position encoding and feed-forward block.
    num_heads : int
        Number of cross-attention heads.
    dropout : float
        Dropout used inside cross-attention and the feed-forward block.
    output_mode : str
        ``"fused"`` keeps the legacy single token stream. ``"per_view"`` keeps one query stream per camera and
        concatenates them within each frame, matching LAW-style per-camera supervision.

    Inputs
    ------
    view_tokens : [B, V, T, P, D] or [B, V, T*P, D]
        Per-camera tokens from the shared encoder. The 5D ViT-style shape is preferred; the flattened 4D shape is
        accepted for existing call sites.
    camera_intrinsics : [B, V, T, 3, 3] or [B, V, 3, 3], optional
        Resized/cropped intrinsics for each view.
    camera2ego : [B, V, T, 4, 4] or [B, V, 4, 4], optional
        Camera-to-ego transforms for each view.

    Returns
    -------
    torch.Tensor
        Fused tokens with shape ``[B, T*P, D]``.
    """

    def __init__(
        self,
        embed_dim: int,
        tokens_per_frame: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        depth_num: int = 64,
        depth_start: float = 1.0,
        position_range: Sequence[float] = (-65.0, -65.0, -8.0, 65.0, 65.0, 8.0),
        lid: bool = False,
        output_mode: str = "fused",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.tokens_per_frame = int(tokens_per_frame)
        if self.tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive")
        self.output_mode = str(output_mode).lower()
        if self.output_mode not in ("fused", "per_view"):
            raise ValueError(f"output_mode must be 'fused' or 'per_view', got {output_mode!r}")

        self.query_embed = nn.Parameter(torch.zeros(1, self.tokens_per_frame, self.embed_dim))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        if isinstance(depth_num, bool) or not isinstance(depth_num, numbers.Integral):
            raise ValueError("depth_num must be an integer")
        self.depth_num = int(depth_num)
        if self.depth_num <= 0:
            raise ValueError("depth_num must be positive")
        try:
            self.depth_start = float(depth_start)
        except (TypeError, ValueError) as exc:
            raise ValueError("depth_start must be numeric") from exc
        if not math.isfinite(self.depth_start):
            raise ValueError("depth_start must be finite")
        if self.depth_start <= 0.0:
            raise ValueError("depth_start must be positive")
        self.lid = bool(lid)

        try:
            position_range_values = tuple(float(value) for value in position_range)
        except (TypeError, ValueError) as exc:
            raise ValueError("position_range must contain six numeric values") from exc
        if len(position_range_values) != 6:
            raise ValueError("position_range must contain six values")
        if not all(math.isfinite(value) for value in position_range_values):
            raise ValueError("position_range values must be finite")
        for axis, (lower, upper) in enumerate(zip(position_range_values[:3], position_range_values[3:])):
            if upper <= lower:
                raise ValueError(f"position_range max bound must exceed min bound for axis {axis}")
        if self.depth_start >= position_range_values[3]:
            raise ValueError("depth_start must be less than position_range x max bound")
        self.register_buffer(
            "position_range",
            torch.tensor(position_range_values, dtype=torch.float32),
            persistent=False,
        )

        self.position_mlp = nn.Sequential(
            nn.Linear(3 * self.depth_num, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.embed_dim),
        )
        self.cross_attn = nn.MultiheadAttention(
            self.embed_dim,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(self.embed_dim)
        self.kv_norm = nn.LayerNorm(self.embed_dim)
        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.embed_dim),
        )

    def forward(
        self,
        view_tokens: torch.Tensor,
        camera_intrinsics: Optional[torch.Tensor] = None,
        camera2ego: Optional[torch.Tensor] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if view_tokens.ndim == 5:
            batch_size, num_views, num_steps, tokens_per_frame, embed_dim = view_tokens.shape
            if tokens_per_frame != self.tokens_per_frame:
                raise ValueError(
                    f"view token patches per frame {tokens_per_frame} must match tokens_per_frame "
                    f"{self.tokens_per_frame}"
                )
            tokens = view_tokens
        elif view_tokens.ndim == 4:
            batch_size, num_views, num_tokens, embed_dim = view_tokens.shape
            if num_tokens % self.tokens_per_frame != 0:
                raise ValueError(
                    f"Token stream length {num_tokens} must be divisible by tokens_per_frame {self.tokens_per_frame}"
                )
            num_steps = num_tokens // self.tokens_per_frame
            tokens = view_tokens.reshape(batch_size, num_views, num_steps, self.tokens_per_frame, embed_dim)
        else:
            raise ValueError(
                f"Expected view_tokens [B, V, T, P, D] or [B, V, T*P, D], got shape {tuple(view_tokens.shape)}"
            )
        if embed_dim != self.embed_dim:
            raise ValueError(f"view token dim {embed_dim} does not match fusion embed_dim {self.embed_dim}")

        position = self._build_position_embedding(
            batch_size=batch_size,
            num_views=num_views,
            num_steps=num_steps,
            device=view_tokens.device,
            dtype=view_tokens.dtype,
            camera_intrinsics=camera_intrinsics,
            camera2ego=camera2ego,
            image_shape=image_shape,
        )
        if self.output_mode == "per_view":
            return self._forward_per_view(tokens, position)
        return self._forward_fused(tokens, position)

    def _forward_fused(self, tokens: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, num_steps, _, embed_dim = tokens.shape
        kv = self.kv_norm(tokens + position)
        kv = kv.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_steps,
            num_views * self.tokens_per_frame,
            embed_dim,
        )

        query = tokens.mean(dim=1) + self.query_embed.to(dtype=tokens.dtype)
        query = self.query_norm(query).reshape(batch_size * num_steps, self.tokens_per_frame, embed_dim)

        fused, _ = self.cross_attn(query, kv, kv, need_weights=False)
        fused = self.out_norm(fused + query)
        fused = self.out_norm(fused + self.ffn(fused))
        return fused.view(batch_size, num_steps * self.tokens_per_frame, embed_dim)

    def _forward_per_view(self, tokens: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, num_steps, _, embed_dim = tokens.shape
        kv = self.kv_norm(tokens + position)
        kv = kv.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_steps * num_views,
            self.tokens_per_frame,
            embed_dim,
        )

        query = tokens + self.query_embed.to(dtype=tokens.dtype)
        query = self.query_norm(query)
        query = query.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_steps * num_views,
            self.tokens_per_frame,
            embed_dim,
        )

        per_view, _ = self.cross_attn(query, kv, kv, need_weights=False)
        per_view = self.out_norm(per_view + query)
        per_view = self.out_norm(per_view + self.ffn(per_view))
        return per_view.view(batch_size, num_steps, num_views, self.tokens_per_frame, embed_dim).reshape(
            batch_size,
            num_steps * num_views * self.tokens_per_frame,
            embed_dim,
        )

    def _build_position_embedding(
        self,
        *,
        batch_size: int,
        num_views: int,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        camera_intrinsics: Optional[torch.Tensor],
        camera2ego: Optional[torch.Tensor],
        image_shape: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        features = self._build_frustum_features(
            batch_size=batch_size,
            num_views=num_views,
            num_steps=num_steps,
            device=device,
            dtype=dtype,
            camera_intrinsics=camera_intrinsics,
            camera2ego=camera2ego,
            image_shape=image_shape,
        )
        mlp_dtype = self.position_mlp[0].weight.dtype
        return self.position_mlp(features.to(dtype=mlp_dtype)).to(dtype=dtype)

    def _build_frustum_features(
        self,
        *,
        batch_size: int,
        num_views: int,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        camera_intrinsics: Optional[torch.Tensor],
        camera2ego: Optional[torch.Tensor],
        image_shape: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        geometry_dtype = torch.float32
        grid = self._token_grid(device=device, dtype=geometry_dtype, image_shape=image_shape)
        token_centers = grid[:, :2].view(1, 1, 1, self.tokens_per_frame, 1, 2)
        token_centers = token_centers.expand(
            batch_size, num_views, num_steps, self.tokens_per_frame, self.depth_num, 2
        )

        depths = self._depth_bins(device=device, dtype=geometry_dtype).view(1, 1, 1, 1, self.depth_num, 1)
        depths = depths.expand(batch_size, num_views, num_steps, self.tokens_per_frame, self.depth_num, 1)
        image_depth = torch.cat([token_centers * depths, depths], dim=-1)

        if camera_intrinsics is None:
            raise ValueError(
                "PETRMultiViewFusion requires camera_intrinsics: the geometry-driven position "
                "embedding must not silently fall back to identity intrinsics (that would make every "
                "view geometrically identical and corrupt the fused tokens)."
            )
        intrinsics = self._expand_intrinsics(camera_intrinsics, batch_size, num_views, num_steps).to(
            device=device,
            dtype=geometry_dtype,
        )
        inv_intrinsics = torch.linalg.inv(intrinsics)
        camera_xyz = torch.matmul(
            inv_intrinsics.unsqueeze(-3).unsqueeze(-3),
            image_depth.unsqueeze(-1),
        ).squeeze(-1)

        ones = torch.ones_like(camera_xyz[..., :1])
        camera_homogeneous = torch.cat([camera_xyz, ones], dim=-1)
        if camera2ego is None:
            raise ValueError(
                "PETRMultiViewFusion requires camera2ego: the geometry-driven position embedding "
                "must not silently fall back to identity extrinsics (that would make every view "
                "geometrically identical and corrupt the fused tokens)."
            )
        transform = self._expand_camera2ego(camera2ego, batch_size, num_views, num_steps).to(
            device=device,
            dtype=geometry_dtype,
        )
        ego_homogeneous = torch.matmul(
            transform.unsqueeze(-3).unsqueeze(-3),
            camera_homogeneous.unsqueeze(-1),
        ).squeeze(-1)
        ego_xyz = ego_homogeneous[..., :3]

        position_range = self.position_range.to(device=device, dtype=geometry_dtype)
        range_min = position_range[:3].view(1, 1, 1, 1, 1, 3)
        range_max = position_range[3:].view(1, 1, 1, 1, 1, 3)
        normalized = (ego_xyz - range_min) / (range_max - range_min)
        encoded = self._inverse_sigmoid(normalized)

        # TODO: Add StreamPETR-style top-k token selection and temporal memory after dense frustum PE is validated.
        features = encoded.reshape(batch_size, num_views, num_steps, self.tokens_per_frame, 3 * self.depth_num)
        return features.to(dtype=dtype)

    def _depth_bins(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        index = torch.arange(start=0, end=self.depth_num, step=1, device=device, dtype=dtype)
        max_depth = self.position_range[3].to(device=device, dtype=dtype)
        depth_start = torch.as_tensor(self.depth_start, device=device, dtype=dtype)
        if self.lid:
            index_1 = index + 1.0
            bin_size = (max_depth - depth_start) / (self.depth_num * (1 + self.depth_num))
            return depth_start + bin_size * index * index_1

        bin_size = (max_depth - depth_start) / self.depth_num
        return depth_start + bin_size * index

    @staticmethod
    def _inverse_sigmoid(values: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        values = values.clamp(min=eps, max=1.0 - eps)
        return torch.log(values / (1.0 - values))

    def _token_grid(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        image_shape: Optional[Tuple[int, int]],
    ) -> torch.Tensor:
        grid_h, grid_w = self._infer_grid_shape(self.tokens_per_frame, image_shape=image_shape)
        if image_shape is None:
            height = float(grid_h)
            width = float(grid_w)
        else:
            height = float(image_shape[0])
            width = float(image_shape[1])

        y = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5) * (height / grid_h)
        x = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5) * (width / grid_w)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        ones = torch.ones_like(xx)
        return torch.stack([xx, yy, ones], dim=-1).reshape(self.tokens_per_frame, 3)

    @staticmethod
    def _infer_grid_shape(
        tokens_per_frame: int,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        if image_shape is not None:
            height = float(image_shape[0])
            width = float(image_shape[1])
            if height <= 0.0 or width <= 0.0:
                raise ValueError(f"image_shape dimensions must be positive, got {image_shape}")
            target_ratio = width / height
            best_shape = None
            best_score = (float("inf"), 0)
            max_factor = int(math.sqrt(tokens_per_frame))
            for factor in range(1, max_factor + 1):
                if tokens_per_frame % factor != 0:
                    continue
                paired = tokens_per_frame // factor
                for grid_h, grid_w in ((factor, paired), (paired, factor)):
                    ratio = float(grid_w) / float(grid_h)
                    score = (abs(math.log(ratio / target_ratio)), -min(grid_h, grid_w))
                    if score < best_score:
                        best_shape = (grid_h, grid_w)
                        best_score = score
            if best_shape is not None:
                return best_shape

        side = int(math.sqrt(tokens_per_frame))
        if side * side == tokens_per_frame:
            return side, side
        return 1, tokens_per_frame

    @staticmethod
    def _expand_intrinsics(
        intrinsics: torch.Tensor,
        batch_size: int,
        num_views: int,
        num_steps: int,
    ) -> torch.Tensor:
        return PETRMultiViewFusion._expand_temporal_matrix(
            intrinsics,
            batch_size=batch_size,
            num_views=num_views,
            num_steps=num_steps,
            matrix_size=3,
            name="camera_intrinsics",
        )

    @staticmethod
    def _expand_camera2ego(
        camera2ego: torch.Tensor,
        batch_size: int,
        num_views: int,
        num_steps: int,
    ) -> torch.Tensor:
        return PETRMultiViewFusion._expand_temporal_matrix(
            camera2ego,
            batch_size=batch_size,
            num_views=num_views,
            num_steps=num_steps,
            matrix_size=4,
            name="camera2ego",
        )

    @staticmethod
    def _expand_temporal_matrix(
        matrix: torch.Tensor,
        *,
        batch_size: int,
        num_views: int,
        num_steps: int,
        matrix_size: int,
        name: str,
    ) -> torch.Tensor:
        if matrix.ndim == 4:
            matrix = matrix.unsqueeze(2)
        elif matrix.ndim != 5:
            raise ValueError(
                f"{name} must have shape [B,V,T,{matrix_size},{matrix_size}] or "
                f"[B,V,{matrix_size},{matrix_size}], got {tuple(matrix.shape)}"
            )
        if matrix.shape[-2:] != (matrix_size, matrix_size):
            raise ValueError(f"{name} must end with [{matrix_size},{matrix_size}], got {tuple(matrix.shape[-2:])}")

        metadata_steps = int(matrix.shape[2])
        if metadata_steps not in (1, num_steps):
            if metadata_steps > num_steps and metadata_steps % num_steps == 0:
                stride = metadata_steps // num_steps
                indices = torch.arange(stride - 1, metadata_steps, stride, device=matrix.device, dtype=torch.long)
                matrix = matrix.index_select(2, indices)
            else:
                raise ValueError(f"{name} temporal length {metadata_steps} cannot align to token steps {num_steps}")

        if matrix.shape[0] not in (1, batch_size):
            raise ValueError(f"{name} batch dimension {matrix.shape[0]} cannot broadcast to batch_size {batch_size}")
        if matrix.shape[1] not in (1, num_views):
            raise ValueError(f"{name} view dimension {matrix.shape[1]} cannot broadcast to num_views {num_views}")
        return matrix.expand(batch_size, num_views, num_steps, matrix_size, matrix_size)
