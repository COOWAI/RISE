"""Distributed sampler policies for training and exact validation."""

from __future__ import annotations

from typing import Iterator, Sized

from torch.utils.data import Sampler


class ExactDistributedEvalSampler(Sampler[int]):
    """Partition validation indices exactly once without padding or dropping.

    The quotient/remainder split is contiguous and deterministic. Ranks below
    ``len(dataset) % num_replicas`` receive one additional sample; when there
    are more ranks than samples, the remaining ranks receive an empty slice.
    """

    def __init__(self, dataset: Sized, *, num_replicas: int, rank: int) -> None:
        if int(num_replicas) <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if int(rank) < 0 or int(rank) >= int(num_replicas):
            raise ValueError(f"rank must be in [0, {int(num_replicas)}), got {rank}")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def _bounds(self) -> tuple[int, int]:
        dataset_size = len(self.dataset)
        quotient, remainder = divmod(dataset_size, self.num_replicas)
        local_size = quotient + int(self.rank < remainder)
        start = self.rank * quotient + min(self.rank, remainder)
        return start, start + local_size

    def __iter__(self) -> Iterator[int]:
        start, stop = self._bounds()
        return iter(range(start, stop))

    def __len__(self) -> int:
        start, stop = self._bounds()
        return stop - start

    def set_epoch(self, epoch: int) -> None:
        """Keep a ``DistributedSampler``-compatible no-op interface."""
        del epoch
