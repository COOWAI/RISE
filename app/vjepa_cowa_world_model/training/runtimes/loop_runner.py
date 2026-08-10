"""Composition training-loop runner shared by the training lines (Phase 5 slice 1).

The five line ``main()``s re-implement the *same* outer epoch/iteration scaffolding verbatim — epoch
iteration, the per-iteration loop, the validation **cadence**, the checkpoint **cadence**, the
``dist.barrier`` placement, and the best-checkpoint hook. Only the math (``train_step``), data
unpacking, and per-line setup (e.g. aWTA temperature) differ.

``TrainingLoopRunner`` owns that orchestration; each line passes callbacks for its line-specific
parts. This is **composition, not a base class** (Codex/consensus): the math-bearing ``train_step``
stays in the line. The orchestration here mirrors the existing line cadence *exactly* so a migrated
line stays GPU byte-identical (validation gate computed before the epoch summary; checkpoint saved
before the barrier; validation+best run after the barrier, only when the gate is true).

Adoption is staged: migrate one line at a time (start with ``navsim_v2_encoder_direct``, which already
uses ``ForwardRuntime`` + has parity tests), proving fixed-seed byte-parity before the next line.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


def _default_barrier() -> None:
    # Imported lazily so importing this module does not require a distributed context.
    import torch.distributed as dist

    dist.barrier()


def resolve_periodic_checkpoint_epoch(
    epoch: int,
    *,
    periodic_one_based: bool,
    selection_checkpoint_epochs: Tuple[int, ...],
) -> int:
    """Resolve the checkpoint label/cadence epoch while preserving legacy defaults."""

    if selection_checkpoint_epochs and not periodic_one_based:
        raise ValueError("selection_checkpoint_epochs requires periodic_one_based=True")
    if periodic_one_based:
        return epoch + 1
    return epoch


@dataclass
class TrainingLoopRunner:
    """Owns the epoch/iteration scaffolding + validation/checkpoint cadence; lines supply the math.

    Args mirror the locals every line already computes (``start_epoch``, ``epochs``, ``ipe``,
    ``rank``) plus the two checkpoint-cadence knobs. ``barrier`` is injectable for testing.
    """

    start_epoch: int
    epochs: int
    ipe: int
    rank: int
    checkpoint_freq: int
    save_every_freq: int
    save_from_epoch: int = 0
    periodic_one_based: bool = False
    selection_checkpoint_epochs: Tuple[int, ...] = ()
    barrier: Optional[Callable[[], None]] = None

    def __post_init__(self) -> None:
        selection_epochs = self.selection_checkpoint_epochs
        if type(selection_epochs) is not tuple:
            raise TypeError("selection_checkpoint_epochs must be a tuple of positive 1-based integers")
        if any(type(epoch) is not int or epoch <= 0 for epoch in selection_epochs):
            raise ValueError("selection_checkpoint_epochs must contain only positive 1-based integers")
        if any(current >= following for current, following in zip(selection_epochs, selection_epochs[1:])):
            raise ValueError("selection_checkpoint_epochs must be unique and strictly increasing")
        if selection_epochs and not self.periodic_one_based:
            raise ValueError("selection_checkpoint_epochs requires periodic_one_based=True")
        if selection_epochs and selection_epochs[-1] > self.epochs:
            raise ValueError(
                "selection_checkpoint_epochs must not exceed epochs: "
                f"last={selection_epochs[-1]}, epochs={self.epochs}"
            )

    def run(
        self,
        *,
        run_epoch: Callable[[int], bool],
        save_latest: Callable[[int], None],
        save_periodic: Optional[Callable[[int], None]] = None,
        run_validation: Optional[Callable[[int], None]] = None,
    ) -> None:
        """Drive the training loop with the exact per-epoch cadence the lines use.

        A coarse, byte-parity-friendly contract: the line keeps its *entire* per-epoch body verbatim
        inside ``run_epoch`` (per-epoch setup + the iteration loop + the in-epoch summary logging, with
        the loss meters as plain locals), returning ``has_validation``. The runner owns only the
        cross-line-duplicated trailing cadence, which is cleanly separable from the meters:

        Per epoch (``start_epoch`` .. ``epochs``-1):
          1. ``has_validation = run_epoch(epoch)`` — line: full epoch body (math stays here)
          2. checkpoint: ``save_latest`` when ``epoch % checkpoint_freq == 0`` or last epoch;
             ``save_periodic`` once for the union of the ordinary ``save_every_freq`` cadence and
             the required 1-based ``selection_checkpoint_epochs`` (latest.pt is unaffected)
          3. ``barrier()``
          4. ``run_validation(epoch)`` — only when ``has_validation`` — line: validate / log / best.consider
        """
        if self.selection_checkpoint_epochs and save_periodic is None:
            raise ValueError("save_periodic callback is required when selection_checkpoint_epochs is non-empty")
        barrier = self.barrier if self.barrier is not None else _default_barrier
        for epoch in range(self.start_epoch, self.epochs):
            has_validation = bool(run_epoch(epoch))

            if epoch % self.checkpoint_freq == 0 or epoch == (self.epochs - 1):
                save_latest(epoch)
            periodic_epoch = resolve_periodic_checkpoint_epoch(
                epoch,
                periodic_one_based=self.periodic_one_based,
                selection_checkpoint_epochs=self.selection_checkpoint_epochs,
            )
            ordinary_periodic_due = (
                self.save_every_freq > 0
                and periodic_epoch >= self.save_from_epoch
                and periodic_epoch % self.save_every_freq == 0
            )
            selection_periodic_due = (epoch + 1) in self.selection_checkpoint_epochs
            if ordinary_periodic_due or selection_periodic_due:
                if save_periodic is not None:
                    save_periodic(epoch)

            barrier()

            if has_validation and run_validation is not None:
                run_validation(epoch)
