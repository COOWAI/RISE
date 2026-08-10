from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairscale.nn.checkpoint import checkpoint_wrapper


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        else:
            dim = x.shape.index(self.normalized_shape[0])
            u = x.mean(dim, keepdim=True)
            v = (x - u).pow(2).mean(dim, keepdim=True)
            x = (x - u) / torch.sqrt(v + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SFP(nn.Module):
    r"""
    Our custom implementation of Simple Feature Pyramid (SFP).
    """

    def __init__(
        self,
        input_channels,
        out_channels: int,
        output_stage_keys=None,
        use_p2=False,
        use_act_checkpoint=False,
    ):
        super().__init__()
        self.input_channels = input_channels

        self.out_channels = out_channels
        self.num_ins = len(self.input_channels)
        self.use_p2 = use_p2
        if self.use_p2:
            self.p2 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.input_channels[0], self.input_channels[0] // 2, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.input_channels[0] // 2),
                nn.GELU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(
                    self.input_channels[0] // 2, self.input_channels[0] // 4, kernel_size=3, padding=1, bias=False
                ),
                LayerNorm(self.input_channels[0] // 4),
                nn.GELU(),
                nn.Conv2d(self.input_channels[0] // 4, self.out_channels, kernel_size=1, bias=False),
                LayerNorm(self.out_channels),
                nn.GELU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.out_channels),
            )

        self.p3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.input_channels[0], self.input_channels[0] // 2, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.input_channels[0] // 2),
            nn.GELU(),
            nn.Conv2d(self.input_channels[0] // 2, self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels),
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(self.input_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels),
        )
        self.p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.input_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels),
        )
        self.p6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.input_channels[0], self.input_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm(self.input_channels[0]),
            nn.GELU(),
            nn.Conv2d(self.input_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels),
        )

        if use_act_checkpoint:
            self.p3 = checkpoint_wrapper(self.p3)
            self.p4 = checkpoint_wrapper(self.p4)
            self.p5 = checkpoint_wrapper(self.p5)
            self.p6 = checkpoint_wrapper(self.p6)
            if self.use_p2:
                self.p2 = checkpoint_wrapper(self.p2)

        self.output_num = 5 if self.use_p2 else 4
        if output_stage_keys is None:
            self.out_indices = [i for i in range(self.output_num)]
        else:
            self.out_indices = [int(i.split("_")[-1]) - 1 for i in output_stage_keys]

    def forward(self, x: torch.Tensor, **kwargs) -> Dict:
        """Forward function."""
        # inputs = x
        # x = inputs[0]
        p4 = self.p4(x)
        p3 = self.p3(x)
        p5 = self.p5(x)
        p6 = self.p6(x)
        outs = [p3, p4, p5, p6]
        if self.use_p2:
            outs = [self.p2(x)] + outs
        # # out_layers
        if not (len(outs) == self.output_num):
            raise AssertionError("assertion failed: len(outs) == self.output_num")
        # results = {}
        # for idx in range(self.output_num):
        #     if idx in self.out_indices:
        #         results[f"neck_{idx+1}"] = outs[idx]
        return outs
