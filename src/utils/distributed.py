# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os

import torch
import torch.distributed as dist

from src.utils.logging import get_logger

logger = get_logger()


def init_distributed(port=37129):
    # try to set all environment variables to avoid triggering a segfault
    # environment variables can be reallocated during execution of torch.distributed.init_process_group
    # idea is a race condition may trigger if init_progress_group is modifying an environment variable at
    # same time as Python, so we try to set all environs before initializing distributed

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    # Read rank, world_size and local_rank from torchrun environment variables
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    # local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(port)

    # Use gloo backend for single GPU, nccl for multi-GPU
    # Use tcp init_method to avoid IPv6 DNS issues
    backend = "gloo" if world_size == 1 else "nccl"
    torch.distributed.init_process_group(
        backend=backend, world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=1800)
    )

    return world_size, rank


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
