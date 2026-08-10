"""V-JEPA image encoder adapter for LEWM training.

This module keeps the V-JEPA image backbone isolated from the main V-JEPA
video encoder path.  It consumes normalized clips in ``[B, C, T, H, W]`` format,
selects the latest observed frames required by the V-JEPA backbone, and
projects backbone tokens into the caller-requested embedding dimension.
"""

import inspect
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import src.models.vision_transformer as vit
from src.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["VJEPAGridMask", "VJEPAImgEncoderAdapter"]

_TORCH_LOAD_PARAMS = inspect.signature(torch.load).parameters


def _load_checkpoint_dict(checkpoint_path: str) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in _TORCH_LOAD_PARAMS:
        kwargs["weights_only"] = True
    if "mmap" in _TORCH_LOAD_PARAMS:
        kwargs["mmap"] = True
    return torch.load(checkpoint_path, **kwargs)


class VJEPAGridMask(nn.Module):
    """V-JEPA-compatible GridMask augmentation for image batches."""

    def __init__(
        self,
        use_h: bool = True,
        use_w: bool = True,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 1,
        prob: float = 0.7,
    ) -> None:
        super().__init__()
        if rotate < 1:
            raise ValueError("rotate must be >= 1")
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        if not 0.0 < ratio <= 1.0:
            raise ValueError("ratio must be in (0, 1]")
        if mode not in (0, 1):
            raise ValueError("mode must be 0 or 1")

        self.use_h = bool(use_h)
        self.use_w = bool(use_w)
        self.rotate = int(rotate)
        self.offset = bool(offset)
        self.prob = float(prob)
        self.ratio = float(ratio)
        self.mode = int(mode)
        self.st_prob = float(prob)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Apply V-JEPA GridMask to an image batch in training mode.

        Parameters
        ----------
        image : torch.Tensor
            Image batch with shape ``[N, C, H, W]``.

        Returns
        -------
        torch.Tensor
            Masked image batch with the same shape as input.
        """
        if not self.training:
            return image
        if image.ndim != 4:
            raise ValueError("VJEPAGridMask expects image tensors with shape [N, C, H, W]")
        if np.random.rand() > self.prob:
            return image

        batch_size, channels, height, width = image.size()
        if height <= 2:
            return image

        flat_image = image.reshape(-1, height, width)
        canvas_height = int(1.5 * height)
        canvas_width = int(1.5 * width)
        grid_size = int(np.random.randint(2, height))
        cut_size = min(max(int(grid_size * self.ratio + 0.5), 1), grid_size - 1)
        mask = np.ones((canvas_height, canvas_width), np.float32)
        start_h = int(np.random.randint(grid_size))
        start_w = int(np.random.randint(grid_size))

        if self.use_h:
            for idx in range(canvas_height // grid_size):
                start = grid_size * idx + start_h
                end = min(start + cut_size, canvas_height)
                mask[start:end, :] *= 0
        if self.use_w:
            for idx in range(canvas_width // grid_size):
                start = grid_size * idx + start_w
                end = min(start + cut_size, canvas_width)
                mask[:, start:end] *= 0

        if self.rotate > 1:
            rotate_degree = int(np.random.randint(self.rotate))
            if rotate_degree != 0:
                from PIL import Image

                mask = np.asarray(Image.fromarray(np.uint8(mask)).rotate(rotate_degree))

        crop_top = (canvas_height - height) // 2
        crop_left = (canvas_width - width) // 2
        mask = mask[crop_top : crop_top + height, crop_left : crop_left + width]
        mask = torch.from_numpy(mask).to(dtype=flat_image.dtype, device=flat_image.device)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(flat_image)

        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(height, width) - 0.5)).to(
                dtype=flat_image.dtype,
                device=flat_image.device,
            )
            flat_image = flat_image * mask + offset * (1 - mask)
        else:
            flat_image = flat_image * mask

        return flat_image.reshape(batch_size, channels, height, width)


class VJEPAImgEncoderAdapter(nn.Module):
    """Adapter that exposes a V-JEPA image encoder as token features.

    The adapter returns raw backbone tokens (no internal projector) — the
    downstream planner already owns an ``encoder_dim → hidden_dim`` projection
    (e.g. ``DiffusionPlanner.context_proj`` or ``MultiModalTemporalPlanner.temporal_fc``),
    so adding a second projection here would be redundant.

    Parameters
    ----------
    embed_dim : Optional[int]
        Deprecated. Kept for backward compatibility — ignored at runtime.
        Adapter exposes ``backbone.embed_dim`` (e.g. 1024 for ViT-Large)
        regardless of what is passed.
    checkpoint_path : Optional[str]
        Optional backbone checkpoint path.  Only shape-compatible backbone keys
        under ``checkpoint_key`` are loaded.
    resolution : Tuple[int, int]
        Expected image resolution as ``(height, width)``.
    backbone : Optional[nn.Module]
        Prebuilt backbone, mainly for tests and dependency injection.
    use_causal_attention : bool
        Whether to apply the block-causal temporal attention mask by default.
        Set to ``False`` for parallel/full attention over observed encoder frames.
    """

    def __init__(
        self,
        embed_dim: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        resolution: Tuple[int, int] = (256, 512),
        backbone: Optional[nn.Module] = None,
        model_name: str = "vit_large",
        patch_size: int = 16,
        num_frames: int = 2,
        max_num_observed_frames: int = 2,
        tubelet_size: int = 2,
        uniform_power: bool = False,
        use_rope: bool = False,
        use_sdpa: bool = False,
        use_activation_checkpointing: bool = False,
        checkpoint_key: str = "target_encoder",
        use_grid_mask: bool = True,
        grid_mask_prob: float = 0.7,
        use_causal_attention: bool = True,
    ) -> None:
        super().__init__()
        self.is_vjepa_img_encoder_adapter = True
        self.resolution = self._validate_resolution(resolution)
        self.patch_size = int(patch_size)
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.use_causal_attention = bool(use_causal_attention)

        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.num_frames != 2:
            raise ValueError("V-JEPA ImgEncoder requires num_frames=2")
        if self.tubelet_size != 2:
            raise ValueError("V-JEPA ImgEncoder requires tubelet_size=2")
        if self.resolution[0] % self.patch_size != 0 or self.resolution[1] % self.patch_size != 0:
            raise ValueError("resolution must be divisible by patch_size")

        self.max_num_observed_frames = int(max_num_observed_frames)
        if self.max_num_observed_frames < self.num_frames:
            raise ValueError(
                f"max_num_observed_frames ({self.max_num_observed_frames}) must be >= num_frames ({self.num_frames})"
            )
        if self.max_num_observed_frames % self.num_frames != 0:
            raise ValueError(
                f"max_num_observed_frames ({self.max_num_observed_frames}) must be divisible by "
                f"num_frames ({self.num_frames})"
            )
        self.num_time_steps = self.max_num_observed_frames // self.num_frames
        spatial_tokens = (self.resolution[0] // self.patch_size) * (self.resolution[1] // self.patch_size)
        self.tokens_per_frame = spatial_tokens

        # V-JEPA pretrained checkpoints were trained with RoPE; the absolute
        # sincos pos_embed path now supports non-square via the why_nips fix in
        # src/models/utils/pos_embs.py, but we still require use_rope=True for
        # this adapter to keep checkpoint compatibility and avoid surprising
        # users with a representation distribution shift.
        if backbone is None and self.resolution[0] != self.resolution[1] and not bool(use_rope):
            raise ValueError(
                f"VJEPAImgEncoderAdapter with non-square resolution "
                f"{self.resolution} requires use_rope=True (V-JEPA pretrained "
                f"convention). Set model.use_rope: true in your YAML, or pass a "
                f"prebuilt backbone if you have a sincos checkpoint compatible "
                f"with rectangular pos_embed."
            )

        self.backbone = (
            backbone
            if backbone is not None
            else self._build_backbone(
                model_name=model_name,
                patch_size=self.patch_size,
                num_frames=self.num_frames,
                tubelet_size=self.tubelet_size,
                uniform_power=uniform_power,
                use_rope=use_rope,
                use_sdpa=use_sdpa,
                use_activation_checkpointing=use_activation_checkpointing,
            )
        )

        self.embed_dim = self._resolve_backbone_embed_dim(self.backbone)
        if embed_dim is not None and int(embed_dim) != self.embed_dim:
            logger.warning(
                "VJEPAImgEncoderAdapter: embed_dim=%s is ignored; adapter exposes "
                "backbone.embed_dim=%s and lets the planner own the projection.",
                embed_dim,
                self.embed_dim,
            )
        self.grid_mask = VJEPAGridMask(prob=grid_mask_prob) if use_grid_mask else nn.Identity()

        if checkpoint_path:
            self.load_backbone_checkpoint(checkpoint_path, checkpoint_key)

    def _build_backbone(
        self,
        model_name: str,
        patch_size: int,
        num_frames: int,
        tubelet_size: int,
        uniform_power: bool,
        use_rope: bool,
        use_sdpa: bool,
        use_activation_checkpointing: bool,
    ) -> nn.Module:
        if model_name not in vit.__dict__:
            raise KeyError(f"Unknown V-JEPA image backbone model_name: {model_name}")
        return vit.__dict__[model_name](
            img_size=self.resolution,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_rope=use_rope,
            use_sdpa=use_sdpa,
            use_activation_checkpointing=use_activation_checkpointing,
        )

    @staticmethod
    def _resolve_backbone_embed_dim(backbone: nn.Module) -> int:
        embed_dim = getattr(backbone, "embed_dim", None)
        if embed_dim is None:
            embed_dim = getattr(backbone, "num_features", None)
        if embed_dim is None:
            raise ValueError("V-JEPA backbone must expose embed_dim or num_features")
        return int(embed_dim)

    @staticmethod
    def _validate_resolution(resolution: Tuple[int, int]) -> Tuple[int, int]:
        try:
            height, width = resolution
        except (TypeError, ValueError) as exc:
            raise ValueError("resolution must contain (height, width)") from exc
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0:
            raise ValueError("resolution values must be positive")
        return height, width

    @staticmethod
    def _strip_checkpoint_prefix(key: str) -> str:
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix) :]
                    changed = True
        return clean_key

    def load_backbone_checkpoint(self, checkpoint_path: str, checkpoint_key: str) -> None:
        """Load shape-compatible backbone weights from a checkpoint."""
        checkpoint = _load_checkpoint_dict(checkpoint_path)
        if not isinstance(checkpoint, dict):
            raise KeyError(
                f"Checkpoint must be a dict containing key {checkpoint_key!r}; "
                f"got {type(checkpoint).__name__}; available keys: []"
            )
        if checkpoint_key not in checkpoint:
            available_keys = list(checkpoint.keys())[:20]
            raise KeyError(
                f"Checkpoint does not contain key {checkpoint_key!r}; "
                f"available keys (first {len(available_keys)}): {available_keys}"
            )

        raw_state_dict = checkpoint[checkpoint_key]
        if not isinstance(raw_state_dict, dict):
            raise ValueError(f"Checkpoint key {checkpoint_key!r} must contain a state dict")

        backbone_state = self.backbone.state_dict()
        loadable_state: Dict[str, Any] = {}
        skipped = 0
        for key, value in raw_state_dict.items():
            clean_key = self._strip_checkpoint_prefix(key)
            value_shape = getattr(value, "shape", None)
            if (
                clean_key in backbone_state
                and value_shape is not None
                and tuple(backbone_state[clean_key].shape) == tuple(value_shape)
            ):
                loadable_state[clean_key] = value
            else:
                skipped += 1

        if not loadable_state:
            raw_keys = list(raw_state_dict.keys())[:20]
            backbone_keys = list(backbone_state.keys())[:20]
            raise RuntimeError(
                "No shape-compatible V-JEPA backbone weights found in "
                f"checkpoint {checkpoint_path!r} under key {checkpoint_key!r}; "
                "refusing to keep a randomly initialized backbone. "
                f"checkpoint keys (first {len(raw_keys)}): {raw_keys}; "
                f"backbone keys (first {len(backbone_keys)}): {backbone_keys}; skipped={skipped}"
            )

        # strict=False tolerates the checkpoint carrying MORE than the backbone (predictor / target_encoder
        # keys are filtered out into `skipped`/unexpected); but a backbone param that did NOT load is left
        # randomly initialized — fail loud on that (consumed weights must all load).
        incompatible = self.backbone.load_state_dict(loadable_state, strict=False)
        logger.info(
            "Loaded V-JEPA backbone checkpoint %s key=%s matched=%d skipped=%d missing=%d unexpected=%d",
            checkpoint_path,
            checkpoint_key,
            len(loadable_state),
            skipped,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if incompatible.missing_keys:
            raise RuntimeError(
                f"V-JEPA backbone has {len(incompatible.missing_keys)} parameter(s) not loaded from "
                f"{checkpoint_path!r} (key={checkpoint_key!r}) — they would stay randomly initialized; "
                f"refusing a silent partial-random backbone. missing (first 20): {incompatible.missing_keys[:20]}"
            )

    def forward(
        self,
        context_clips: torch.Tensor,
        num_observed_frames: int,
        use_causal_attention: Optional[bool] = None,
    ) -> torch.Tensor:
        """Encode observed V-JEPA frames in a single backbone forward.

        All ``num_observed_frames`` frames are processed in one call. By
        default, the adapter applies a block-causal attention mask that
        enforces temporal causality across ``num_observed_frames // num_frames``
        time steps. When ``use_causal_attention=False``, the mask is omitted and
        the backbone uses parallel/full attention over the observed window. Each
        step still spans ``num_frames`` (=2) raw frames via the backbone's
        tubelet embedding, so the per-step token count remains
        ``tokens_per_frame``.

        Parameters
        ----------
        context_clips : torch.Tensor
            Clip tensor with shape ``[B, C, T, H, W]``.
        num_observed_frames : int
            Number of observed frames available at the front of ``context_clips``.
            Must be divisible by ``num_frames`` (=2) and ``<= max_num_observed_frames``.
        use_causal_attention : Optional[bool]
            Optional per-call override for the adapter default attention mode.

        Returns
        -------
        torch.Tensor
            Raw backbone tokens with shape
            ``[B, (num_observed_frames // num_frames) * tokens_per_frame, embed_dim]``.
            Token order is row-major over (time_step, spatial_position) so the
            output is layout-compatible with the previous chunk-by-chunk path.
        """
        if context_clips.ndim != 5:
            raise ValueError("context_clips must have shape [B, C, T, H, W]")

        batch_size, channels, clip_frames, height, width = context_clips.shape
        if num_observed_frames < self.num_frames:
            raise ValueError(f"V-JEPA ImgEncoder requires at least {self.num_frames} observed frames")
        if num_observed_frames > clip_frames:
            raise ValueError("num_observed_frames cannot exceed context_clips temporal dimension")
        if (height, width) != self.resolution:
            expected_height, expected_width = self.resolution
            raise ValueError(
                f"V-JEPA ImgEncoder expects spatial size {expected_height}x{expected_width}, got {height}x{width}"
            )
        if num_observed_frames % self.num_frames != 0:
            raise ValueError(
                f"num_observed_frames ({num_observed_frames}) must be divisible by num_frames ({self.num_frames})"
            )
        if num_observed_frames > self.max_num_observed_frames:
            raise ValueError(
                f"num_observed_frames ({num_observed_frames}) exceeds adapter capacity "
                f"max_num_observed_frames={self.max_num_observed_frames}"
            )

        video = context_clips[:, :, :num_observed_frames]
        T_obs = video.shape[2]

        flat_video = video.permute(0, 2, 1, 3, 4).reshape(batch_size * T_obs, channels, height, width)
        flat_video = self.grid_mask(flat_video)
        video = flat_video.reshape(batch_size, T_obs, channels, height, width).permute(0, 2, 1, 3, 4)

        num_steps = T_obs // self.num_frames
        H_p = height // self.patch_size
        W_p = width // self.patch_size
        resolved_use_causal_attention = (
            self.use_causal_attention if use_causal_attention is None else bool(use_causal_attention)
        )
        attn_mask = None
        if resolved_use_causal_attention:
            from src.models.utils.modules import build_action_block_causal_attention_mask

            attn_mask = build_action_block_causal_attention_mask(
                num_steps,
                H_p,
                W_p,
                add_tokens=0,
            ).to(video.device, non_blocking=True)

        tokens = self.backbone(video, attn_mask=attn_mask)
        if isinstance(tokens, (list, tuple)):
            if not tokens:
                raise ValueError("V-JEPA backbone returned an empty token sequence")
            tokens = tokens[-1]
        if tokens.ndim != 3:
            raise ValueError("V-JEPA backbone must return tokens with shape [B, N, D]")

        expected = num_steps * self.tokens_per_frame
        if tokens.shape[1] != expected:
            raise ValueError(
                f"V-JEPA backbone returned {tokens.shape[1]} tokens, expected {expected} "
                f"(num_steps={num_steps}, tokens_per_frame={self.tokens_per_frame})"
            )
        return tokens
