"""Strict DINOv2 image-encoder adapter for temporal token extraction."""

import inspect
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_DINOV2_MODEL = "vit_large_patch14_reg4_dinov2"
_TORCH_LOAD_PARAMS = inspect.signature(torch.load).parameters

__all__ = ["Dinov2ImageEncoderAdapter", "SUPPORTED_DINOV2_MODEL"]


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _validate_resolution(resolution: Any) -> Tuple[int, int]:
    if not isinstance(resolution, tuple) or len(resolution) != 2:
        raise ValueError("resolution must be a (height, width) tuple of positive integers")
    height = _require_positive_int(resolution[0], "resolution[0]")
    width = _require_positive_int(resolution[1], "resolution[1]")
    return height, width


def _load_flat_tensor_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    kwargs: Dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in _TORCH_LOAD_PARAMS:
        kwargs["weights_only"] = True
    if "mmap" in _TORCH_LOAD_PARAMS:
        kwargs["mmap"] = True
    checkpoint = torch.load(checkpoint_path, **kwargs)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("DINOv2 checkpoint must be a non-empty flat tensor state dict")
    if not all(type(key) is str and isinstance(value, torch.Tensor) for key, value in checkpoint.items()):
        raise ValueError("DINOv2 checkpoint must be a flat tensor state dict; nested checkpoints are unsupported")
    return checkpoint


def _convert_official_dinov2_reg4_state_dict(source: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert the official DINOv2 ViT-L/14-reg4 keys into timm's exact schema."""
    required_keys = ("mask_token", "register_tokens", "cls_token", "pos_embed")
    for key in required_keys:
        if key not in source:
            raise ValueError(f"Official DINOv2 checkpoint is missing required key {key!r}")
    if "reg_token" in source:
        raise ValueError("Official DINOv2 checkpoint has conflicting keys 'register_tokens' and 'reg_token'")

    mask_token = source["mask_token"]
    register_tokens = source["register_tokens"]
    cls_token = source["cls_token"]
    pos_embed = source["pos_embed"]
    for key, value in (
        ("mask_token", mask_token),
        ("register_tokens", register_tokens),
        ("cls_token", cls_token),
        ("pos_embed", pos_embed),
    ):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Official DINOv2 checkpoint key {key!r} must be a Tensor")
    expected_shapes = {
        "mask_token": (1, 1024),
        "register_tokens": (1, 4, 1024),
        "cls_token": (1, 1, 1024),
        "pos_embed": (1, 1370, 1024),
    }
    for key, value in (
        ("mask_token", mask_token),
        ("register_tokens", register_tokens),
        ("cls_token", cls_token),
        ("pos_embed", pos_embed),
    ):
        if tuple(value.shape) != expected_shapes[key]:
            raise ValueError(
                f"Official DINOv2 checkpoint key {key!r} must have shape {expected_shapes[key]}, "
                f"got {tuple(value.shape)}"
            )

    converted: Dict[str, torch.Tensor] = {}
    for source_key, value in source.items():
        if source_key == "mask_token":
            continue
        if source_key == "register_tokens":
            target_key = "reg_token"
            target_value = value
        elif source_key == "cls_token":
            target_key = source_key
            target_value = value + pos_embed[:, :1]
        elif source_key == "pos_embed":
            target_key = source_key
            target_value = value[:, 1:]
        else:
            target_key = re.sub(r"^(blocks\.\d+\.mlp)\.w12\.(weight|bias)$", r"\1.fc1.\2", source_key)
            target_key = re.sub(r"^(blocks\.\d+\.mlp)\.w3\.(weight|bias)$", r"\1.fc2.\2", target_key)
            target_value = value
        if target_key in converted:
            raise ValueError(f"Official DINOv2 checkpoint key conversion collision at {target_key!r}")
        converted[target_key] = target_value
    return converted


class Dinov2ImageEncoderAdapter(nn.Module):
    """Select stride-aligned image frames and return DINOv2 patch tokens.

    Parameters
    ----------
    context_clips : [B, C, T, H, W]
        Video clips containing the observed timeline.
    """

    def __init__(
        self,
        model_name: str = SUPPORTED_DINOV2_MODEL,
        checkpoint_path: Optional[str] = None,
        resolution: Tuple[int, int] = (224, 448),
        patch_size: int = 14,
        frame_stride: int = 2,
        forward_chunk_size: int = 16,
        backbone: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.resolution = _validate_resolution(resolution)
        self.patch_size = _require_positive_int(patch_size, "patch_size")
        self.num_frames = _require_positive_int(frame_stride, "frame_stride")
        self.forward_chunk_size = _require_positive_int(forward_chunk_size, "forward_chunk_size")
        if self.resolution[0] % self.patch_size or self.resolution[1] % self.patch_size:
            raise ValueError("resolution must be divisible by patch_size")
        self.tokens_per_frame = (self.resolution[0] // self.patch_size) * (self.resolution[1] // self.patch_size)
        self.is_dinov2_img_encoder_adapter = True

        state_dict: Optional[Dict[str, torch.Tensor]] = None
        is_production_backbone = backbone is None
        if is_production_backbone:
            if model_name != SUPPORTED_DINOV2_MODEL:
                raise ValueError(f"Unsupported DINOv2 model_name {model_name!r}; expected {SUPPORTED_DINOV2_MODEL!r}")
            if self.patch_size != 14:
                raise ValueError("Supported DINOv2 model requires patch_size=14")
            if not isinstance(checkpoint_path, str) or not checkpoint_path:
                raise ValueError("checkpoint_path must be a non-empty path when no DINOv2 backbone is injected")
            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(f"DINOv2 checkpoint does not exist or is not a file: {path}")
            state_dict = _convert_official_dinov2_reg4_state_dict(_load_flat_tensor_state_dict(path))
            import timm

            backbone = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)

        self.backbone = backbone
        self.embed_dim = _require_positive_int(getattr(backbone, "embed_dim", None), "backbone.embed_dim")
        self.num_prefix_tokens = _require_positive_int(
            getattr(backbone, "num_prefix_tokens", None), "backbone.num_prefix_tokens"
        )
        self._validate_backbone_patch_metadata(backbone, require_patch_metadata=is_production_backbone)
        if is_production_backbone and (self.embed_dim != 1024 or self.num_prefix_tokens != 5):
            raise ValueError("Production DINOv2 backbone metadata must expose embed_dim=1024 and num_prefix_tokens=5")

        if not is_production_backbone and checkpoint_path is not None:
            if not isinstance(checkpoint_path, str) or not checkpoint_path:
                raise ValueError("checkpoint_path must be a non-empty path when provided")
            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(f"DINOv2 checkpoint does not exist or is not a file: {path}")
            state_dict = _load_flat_tensor_state_dict(path)
        if state_dict is not None:
            self.backbone.load_state_dict(state_dict, strict=True)
            logger.info("Loaded strict DINOv2 checkpoint from %s", path)

    def _validate_backbone_patch_metadata(self, backbone: nn.Module, require_patch_metadata: bool) -> None:
        patch_embed = getattr(backbone, "patch_embed", None)
        if patch_embed is None:
            if require_patch_metadata:
                raise ValueError("Production DINOv2 backbone must expose backbone.patch_embed.patch_size")
            return
        patch_shape = getattr(patch_embed, "patch_size", None)
        if (
            not isinstance(patch_shape, tuple)
            or len(patch_shape) != 2
            or type(patch_shape[0]) is not int
            or type(patch_shape[1]) is not int
            or patch_shape != (self.patch_size, self.patch_size)
        ):
            raise ValueError("backbone.patch_embed.patch_size must be an exact integer pair matching patch_size")
        if require_patch_metadata and patch_shape != (14, 14):
            raise ValueError("Production DINOv2 backbone must expose patch_embed.patch_size=(14, 14)")

    def forward(self, context_clips: torch.Tensor, num_observed_frames: int) -> torch.Tensor:
        """Return selected DINOv2 patch tokens with shape ``[B, S*P, D]``."""
        if not isinstance(context_clips, torch.Tensor) or context_clips.ndim != 5:
            raise ValueError("context_clips must be a Tensor with shape [B, C, T, H, W]")
        batch_size, channels, total_frames, height, width = context_clips.shape
        if batch_size <= 0:
            raise ValueError("context_clips batch dimension must be positive")
        if channels != 3:
            raise ValueError(f"context_clips must have 3 channels, got {channels}")
        if (height, width) != self.resolution:
            raise ValueError(
                f"context_clips spatial size must match resolution {self.resolution}, got {(height, width)}"
            )
        if type(num_observed_frames) is not int or num_observed_frames <= 0:
            raise ValueError("num_observed_frames must be a positive integer")
        if num_observed_frames > total_frames:
            raise ValueError("num_observed_frames must not exceed clip T")
        if num_observed_frames % self.num_frames:
            raise ValueError("num_observed_frames must be divisible by frame_stride")

        selected = context_clips[:, :, self.num_frames - 1 : num_observed_frames : self.num_frames]
        steps = selected.shape[2]
        images = selected.permute(0, 2, 1, 3, 4).reshape(batch_size * steps, channels, height, width)
        patch_outputs = []
        expected_tokens = self.num_prefix_tokens + self.tokens_per_frame
        for start in range(0, images.shape[0], self.forward_chunk_size):
            image_chunk = images[start : start + self.forward_chunk_size]
            tokens = self.backbone.forward_features(image_chunk)
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise ValueError("backbone.forward_features must return a Tensor with shape [N, tokens, embed_dim]")
            if tokens.shape[0] != image_chunk.shape[0]:
                raise ValueError("backbone.forward_features batch dimension must match its image chunk")
            if tokens.shape[1] != expected_tokens:
                raise ValueError(
                    "backbone.forward_features token count must equal num_prefix_tokens + tokens_per_frame"
                )
            if tokens.shape[2] != self.embed_dim:
                raise ValueError("backbone.forward_features embedding dimension must match backbone.embed_dim")
            patch_outputs.append(tokens[:, self.num_prefix_tokens :])
        if not patch_outputs:
            raise ValueError("DINOv2 forward selected no image frames; check num_observed_frames and frame_stride")
        patches = torch.cat(patch_outputs, dim=0)
        return patches.reshape(batch_size, steps * self.tokens_per_frame, self.embed_dim)
