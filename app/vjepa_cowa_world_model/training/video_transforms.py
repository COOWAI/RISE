# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torchvision.transforms as transforms

import src.datasets.utils.video.transforms as video_transforms
from src.datasets.utils.video.randerase import RandomErasing


def _normalize_crop_size(crop_size):
    if isinstance(crop_size, int):
        size = int(crop_size)
        return size, size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        return int(crop_size[0]), int(crop_size[1])
    raise ValueError(f"crop_size must be an int or a 2-element sequence, got {crop_size!r}")


def make_transforms(
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3 / 4, 4 / 3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    deterministic=False,
):

    _frames_augmentation = VideoTransform(
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
        normalize=normalize,
        deterministic=deterministic,
    )
    return _frames_augmentation


class VideoTransform(object):

    def __init__(
        self,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3 / 4, 4 / 3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        deterministic=False,
    ):
        self.is_validation_transform = bool(deterministic)
        self.random_horizontal_flip = False if deterministic else random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = False if deterministic else auto_augment
        self.motion_shift = False if deterministic else motion_shift
        self.crop_size = crop_size
        self.crop_height, self.crop_width = _normalize_crop_size(crop_size)
        self.mean = torch.tensor(normalize[0], dtype=torch.float32)
        self.std = torch.tensor(normalize[1], dtype=torch.float32)
        if not self.auto_augment:
            # Without auto-augment, PIL and tensor conversions simply scale uint8 space by 255.
            self.mean *= 255.0
            self.std *= 255.0

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_height, self.crop_width),
            # auto_augment="rand-m4-n4-w1-mstd0.5-inc1",
            auto_augment="rand-m7-n4-mstd0.5-inc1",
            interpolation="bicubic",
        )

        if deterministic:
            self.spatial_transform = deterministic_resize_center_crop
        else:
            self.spatial_transform = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )

        self.reprob = 0.0 if deterministic else reprob
        self.erase_transform = RandomErasing(
            self.reprob,
            mode="pixel",
            max_count=1,
            num_splits=1,
            device="cpu",
        )

    def __call__(self, buffer):

        if self.auto_augment:
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = self.autoaug_transform(buffer)
            buffer = [transforms.ToTensor()(img) for img in buffer]
            buffer = torch.stack(buffer)  # T C H W
            buffer = buffer.permute(0, 2, 3, 1)  # T H W C
        elif torch.is_tensor(buffer):
            # TODO: ensure input is always a tensor?
            buffer = buffer.to(torch.float32)
        else:
            buffer = torch.tensor(buffer, dtype=torch.float32)

        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_height,
            target_width=self.crop_width,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        buffer = _tensor_normalize_inplace(buffer, self.mean, self.std)
        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer


def deterministic_resize_center_crop(
    images,
    target_height,
    target_width,
    scale=None,
    ratio=None,
):
    """Resize to cover the target rectangle, then take a deterministic center crop."""
    del scale, ratio
    if images.ndim != 4:
        raise ValueError(f"Expected [C, T, H, W] images, got shape {tuple(images.shape)}")
    source_height, source_width = int(images.shape[-2]), int(images.shape[-1])
    if source_height <= 0 or source_width <= 0:
        raise ValueError(f"Source image dimensions must be positive, got {(source_height, source_width)}")
    target_height = int(target_height)
    target_width = int(target_width)
    if target_height <= 0 or target_width <= 0:
        raise ValueError(f"Target image dimensions must be positive, got {(target_height, target_width)}")

    resize_scale = max(target_height / source_height, target_width / source_width)
    resized_height = max(target_height, int(round(source_height * resize_scale)))
    resized_width = max(target_width, int(round(source_width * resize_scale)))
    frames = images.permute(1, 0, 2, 3)
    frames = torch.nn.functional.interpolate(
        frames,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )
    top = (resized_height - target_height) // 2
    left = (resized_width - target_width) // 2
    frames = frames[:, :, top : top + target_height, left : left + target_width]
    return frames.permute(1, 0, 2, 3).contiguous()


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) is list:
        mean = torch.tensor(mean)
    if type(std) is list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


def _tensor_normalize_inplace(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize (with dimensions C, T, H, W).
        mean (tensor): mean value to subtract (in 0 to 255 floats).
        std (tensor): std to divide (in 0 to 255 floats).
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()

    C, T, H, W = tensor.shape
    tensor = tensor.view(C, -1).permute(1, 0)  # Make C the last dimension
    tensor.sub_(mean).div_(std)
    tensor = tensor.permute(1, 0).view(C, T, H, W)  # Put C back in front
    return tensor
