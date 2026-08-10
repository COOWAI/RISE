"""End-of-validation distributed consensus helpers.

Validation shards may contain different numbers of batches, including zero.
Consequently these helpers must only be called after every rank has exhausted
its local loader; no validation batch path may invoke them.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


def distributed_validation_active(world_size: int) -> bool:
    """Return whether validation collectives are required, validating setup."""

    requested_world_size = int(world_size)
    if requested_world_size < 1:
        raise ValueError(f"validation world_size must be positive, got {world_size}")
    if dist.is_available() and dist.is_initialized():
        actual_world_size = int(dist.get_world_size())
        if actual_world_size != requested_world_size:
            raise RuntimeError(
                "validation world_size does not match the initialized process group: "
                f"requested={requested_world_size}, actual={actual_world_size}"
            )
        return actual_world_size > 1
    if requested_world_size > 1:
        raise RuntimeError(
            f"validation world_size={requested_world_size} requires an initialized torch.distributed group"
        )
    return False


def raise_if_validation_failed(
    local_error: Optional[BaseException],
    *,
    validation_name: str,
    device: torch.device,
    world_size: int,
) -> None:
    """Make a rank-local validation error terminate every rank consistently.

    This is a single fixed-shape consensus performed after all local loaders
    are exhausted.  Detailed tracebacks remain rank-local in the logs, while
    the raised summary is byte-identical on every rank.
    """

    if not validation_name:
        raise ValueError("validation_name must be non-empty")
    active = distributed_validation_active(world_size)
    failed_ranks = 1 if local_error is not None else 0
    if active:
        status = torch.tensor([failed_ranks], dtype=torch.int64, device=device)
        dist.all_reduce(status, op=dist.ReduceOp.SUM)
        failed_ranks = int(status.item())
    if failed_ranks == 0:
        return

    message = (
        f"{validation_name} failed on {failed_ranks}/{int(world_size)} ranks; "
        "all ranks aborted before metric reduction"
    )
    raise RuntimeError(message) from local_error


def wrap_validation_batch_error(validation_name: str, batch_idx: int, error: BaseException) -> RuntimeError:
    """Attach batch context to an error retained until final consensus."""

    wrapped = RuntimeError(f"{validation_name} batch {int(batch_idx)} failed")
    wrapped.__cause__ = error
    return wrapped


def reduce_open_loop_validation_totals(
    *,
    metric_sums: Sequence[float],
    total_samples: int,
    l2_sums: Optional[Sequence[float]],
    collision_sums: Optional[Tuple[Sequence[float], Sequence[float], Sequence[float]]],
    collision_samples: int,
    num_steps: int,
    device: torch.device,
    world_size: int,
    l2_samples: Optional[int] = None,
) -> Dict[str, object]:
    """Reduce planner validation totals in one fixed-shape collective.

    Every rank contributes all fields, using zero vectors for an empty local
    shard or for locally unavailable collision annotations.  This avoids the
    conditional collectives that deadlock when ``world_size > dataset_size``.
    """

    if len(metric_sums) != 4:
        raise ValueError(f"open-loop metric_sums must contain ADE/FDE/minADE/minFDE, got {len(metric_sums)}")
    num_steps = int(num_steps)
    if num_steps < 1:
        raise ValueError(f"open-loop num_steps must be positive, got {num_steps}")
    total_samples = int(total_samples)
    if l2_samples is None:
        l2_samples = total_samples if l2_sums is not None else 0
    l2_samples = int(l2_samples)
    collision_samples = int(collision_samples)
    if (
        total_samples < 0
        or l2_samples < 0
        or l2_samples > total_samples
        or collision_samples < 0
        or collision_samples > total_samples
    ):
        raise ValueError(
            "open-loop sample counts must satisfy 0 <= metric_samples <= total_samples, got "
            f"l2_samples={l2_samples}, collision_samples={collision_samples}, total_samples={total_samples}"
        )

    if l2_sums is None:
        if l2_samples:
            raise ValueError("l2_samples is nonzero but L2 totals are missing")
        local_l2 = [0.0] * num_steps
    else:
        local_l2 = [float(value) for value in l2_sums]
        if len(local_l2) != num_steps:
            raise ValueError(f"open-loop L2 totals must have {num_steps} steps, got {len(local_l2)}")

    if collision_sums is None:
        if collision_samples:
            raise ValueError("collision_samples is nonzero but collision totals are missing")
        box_sums = [0.0] * num_steps
        point_sums = [0.0] * num_steps
        gt_sums = [0.0] * num_steps
    else:
        box_sums, point_sums, gt_sums = ([float(value) for value in values] for values in collision_sums)
        lengths = (len(box_sums), len(point_sums), len(gt_sums))
        if lengths != (num_steps, num_steps, num_steps):
            raise ValueError(f"open-loop collision totals must each have {num_steps} steps, got {lengths}")

    payload = torch.tensor(
        [
            *[float(value) for value in metric_sums],
            float(total_samples),
            float(l2_samples),
            *local_l2,
            *box_sums,
            *point_sums,
            *gt_sums,
            float(collision_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    if distributed_validation_active(world_size):
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    reduced = payload.tolist()
    cursor = 0
    reduced_metric_sums = reduced[cursor : cursor + 4]
    cursor += 4
    reduced_total_samples = int(reduced[cursor])
    cursor += 1
    reduced_l2_samples = int(reduced[cursor])
    cursor += 1
    reduced_l2 = reduced[cursor : cursor + num_steps]
    cursor += num_steps
    reduced_box = reduced[cursor : cursor + num_steps]
    cursor += num_steps
    reduced_point = reduced[cursor : cursor + num_steps]
    cursor += num_steps
    reduced_gt = reduced[cursor : cursor + num_steps]
    cursor += num_steps
    reduced_collision_samples = int(reduced[cursor])
    return {
        "metric_sums": reduced_metric_sums,
        "total_samples": reduced_total_samples,
        "l2_samples": reduced_l2_samples,
        "l2_sums": reduced_l2,
        "box_collision_sums": reduced_box,
        "point_collision_sums": reduced_point,
        "gt_collision_sums": reduced_gt,
        "collision_samples": reduced_collision_samples,
    }
