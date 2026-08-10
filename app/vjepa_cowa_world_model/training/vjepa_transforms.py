"""V-JEPA image encoder input transforms."""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


class VJEPAImageTransform:
    """Transform raw image clips for the V-JEPA image encoder.

    Parameters
    ----------
    resolution : Tuple[int, int]
        Target ``(height, width)`` for bilinear interpolation.
    crop_top_bottom : int
        Number of pixels cropped from both top and bottom before resizing.
    """

    def __init__(self, resolution: Tuple[int, int] = (256, 512), crop_top_bottom: int = 28) -> None:
        self.is_validation_transform = True
        if len(resolution) != 2:
            raise ValueError("resolution must contain (height, width)")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.crop_top_bottom = int(crop_top_bottom)
        if self.crop_top_bottom < 0:
            raise ValueError("crop_top_bottom must be non-negative")

        self._mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1, 1)
        self._std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1, 1)

    def __call__(self, clip: np.ndarray) -> torch.Tensor:
        """Convert a raw ``[T, H, W, 3]`` clip to normalized ``[C, T, H, W]``.

        Parameters
        ----------
        clip : np.ndarray
            Raw RGB clip with shape ``[T, H, W, 3]`` and uint8-like values.

        Returns
        -------
        torch.Tensor
            ImageNet-normalized float32 tensor with shape ``[3, T, H, W]``.
        """
        if not isinstance(clip, np.ndarray):
            raise ValueError("VJEPAImageTransform expects a numpy array with shape [T, H, W, 3]")
        if clip.ndim != 4 or clip.shape[-1] != 3:
            raise ValueError("VJEPAImageTransform expects input shape [T, H, W, 3]")

        tensor = torch.as_tensor(clip, dtype=torch.float32)
        if self.crop_top_bottom > 0:
            input_height = tensor.shape[1]
            cropped_height = input_height - 2 * self.crop_top_bottom
            if cropped_height <= 0:
                raise ValueError(
                    f"Input height {input_height} is too small for crop_top_bottom={self.crop_top_bottom}"
                )
            tensor = tensor[:, self.crop_top_bottom : -self.crop_top_bottom, :, :]

        tensor = tensor.permute(0, 3, 1, 2).contiguous().div_(255.0)
        tensor = F.interpolate(tensor, size=self.resolution, mode="bilinear", align_corners=False)
        tensor = tensor.permute(1, 0, 2, 3).contiguous()
        tensor = (tensor - self._mean.to(device=tensor.device)) / self._std.to(device=tensor.device)
        return tensor.contiguous()
