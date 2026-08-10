"""Best open-loop checkpoint tracking service.

Several training lines select their "best" checkpoint by the same open-loop rule
(``L2_avg -> Collision -> ADE -> FDE -> minFDE@K``) and duplicated the identical
bookkeeping verbatim: a bundle of ``best_*`` scalars plus a ``best_open_loop_record``,
restored from ``best_open_loop.pt`` on resume and updated/saved after every validation.
This service owns that state so the lines stop re-implementing it; behaviour is
identical to the previous inlined blocks.

Adopted by ``navsim_v2`` and ``navsim_v2_encoder_direct`` (which both save via a
``save_checkpoint_fn(epoch, path, extra_state=...)`` closure and log the same message).
``stage2_b`` / ``stage3_c`` share the selection rule but save via
``save_stage_checkpoint(...)`` (different signature) and log a different message; they
can adopt this later by passing a ``save_fn`` adapter — left as a follow-up since they
are not GPU-verifiable on the current server.
"""

import copy
from typing import Any, Callable, Dict, Optional

from app.vjepa_cowa_world_model.training.logging import OPEN_LOOP_SELECTION_RULE, is_better_open_loop_candidate
from src.utils.logging import get_logger

logger = get_logger(__name__)

# extra_state schema tag embedded in best_open_loop.pt. Stable interface (checkpoint
# format) - do not change without a migration.
OPEN_LOOP_METRIC_SCHEMA = "w4d_l2_primary_point_l2_aux_v1"
FORMAL_V2_OPEN_LOOP_METRIC_SCHEMA = "cvoi_formal_v2_macro_l2_collision_v2"

_SELECTION_RULES = frozenset({"legacy", "formal_v2_macro"})

_BEST_METRIC_KEYS = ("ade", "fde", "minade_k", "minfde_k", "l2_avg", "collision_rate")


class BestOpenLoopTracker:
    """Tracks the best open-loop checkpoint for one training line.

    Holds the same ``best_*`` scalar state the lines exposed inline, so the final
    ``log_training_summary`` call reads ``tracker.epoch`` / ``tracker.ade`` / ... .
    """

    def __init__(
        self,
        best_path: str,
        *,
        validation_signature: Optional[Dict[str, Any]] = None,
        selection_rule: str = "legacy",
    ) -> None:
        self.best_path = best_path
        if selection_rule not in _SELECTION_RULES:
            raise ValueError(f"selection_rule must be one of {sorted(_SELECTION_RULES)}, got {selection_rule!r}")
        self.selection_rule = selection_rule
        if validation_signature is not None:
            if not isinstance(validation_signature, dict) or not validation_signature:
                raise ValueError("validation_signature must be a non-empty dict when provided")
            if validation_signature.get("version") != "dynamic_rollout_validation_v2":
                raise ValueError(
                    "validation_signature.version must be 'dynamic_rollout_validation_v2', "
                    f"got {validation_signature.get('version')!r}"
                )
        self.validation_signature = copy.deepcopy(validation_signature)
        self.record: Optional[Dict[str, float]] = None
        self.ade = float("inf")
        self.fde = float("inf")
        self.minade_k = float("inf")
        self.minfde_k = float("inf")
        self.l2_avg = float("inf")
        self.collision_rate = float("inf")
        self.epoch = 0

    def _adopt(self, record: Dict[str, float]) -> None:
        self.record = record
        self.ade = record.get("ade", float("inf"))
        self.fde = record.get("fde", float("inf"))
        self.minade_k = record.get("minade_k", float("inf"))
        self.minfde_k = record.get("minfde_k", float("inf"))
        self.l2_avg = record.get("l2_avg", float("inf"))
        self.collision_rate = record.get("collision_rate", float("inf"))
        self.epoch = record.get("epoch", 0)

    def restore(self, load_checkpoint_fn: Callable[[str], Optional[dict]]) -> bool:
        """Restore the best-metric state from ``best_open_loop.pt`` on resume.

        ``best_open_loop.pt`` embeds ``best_open_loop_record`` at top level (via
        extra_state); a resumed ``latest.pt`` does not, so without this a resumed run
        would overwrite a real best checkpoint on its first validation. Returns True
        if a prior best record was found and adopted.
        """
        ckpt = load_checkpoint_fn(self.best_path)
        if ckpt is not None and ckpt.get("best_open_loop_record") is not None:
            stored_rule = ckpt.get("best_open_loop_selection_rule", "legacy")
            if stored_rule != self.selection_rule:
                raise RuntimeError(
                    "best open-loop selection rule mismatch; "
                    f"checkpoint={stored_rule!r}, requested={self.selection_rule!r}: {self.best_path}"
                )
            if self.validation_signature is not None:
                stored_signature = ckpt.get("validation_suite_signature")
                if stored_signature is None:
                    raise RuntimeError(
                        f"best open-loop checkpoint is missing validation_suite_signature: {self.best_path}"
                    )
                if stored_signature != self.validation_signature:
                    raise RuntimeError(
                        "best open-loop validation suite signature mismatch; checkpoint selection semantics "
                        f"changed for {self.best_path}"
                    )
            self._adopt(ckpt["best_open_loop_record"])
            logger.info(
                "Restored best open-loop tracker from %s (epoch=%s, L2_avg=%.5f)",
                self.best_path,
                self.epoch,
                self.l2_avg,
            )
            return True
        return False

    def consider(
        self,
        record: Dict[str, float],
        val_metrics: Dict[str, float],
        *,
        save_fn: Callable[..., None],
    ) -> bool:
        """If ``record`` beats the current best by the open-loop rule, adopt + save it.

        ``record`` must be the output of ``build_validation_record(epoch, val_metrics)``
        (the same object the caller appends to its ``validation_history``); ``epoch`` is
        read from ``record["epoch"]`` for the save. Returns True if a new best was saved.
        """
        if self.selection_rule == "formal_v2_macro":
            candidate_key = (float(record["l2_avg"]), float(record["collision_rate"]), int(record["epoch"]))
            current_key = (
                None
                if self.record is None
                else (
                    float(self.record["l2_avg"]),
                    float(self.record["collision_rate"]),
                    int(self.record["epoch"]),
                )
            )
            is_better = current_key is None or candidate_key < current_key
        else:
            is_better = is_better_open_loop_candidate(record, self.record)
        if not is_better:
            return False
        self._adopt(record)
        metric_schema = (
            FORMAL_V2_OPEN_LOOP_METRIC_SCHEMA if self.selection_rule == "formal_v2_macro" else OPEN_LOOP_METRIC_SCHEMA
        )
        extra_state = {
            "best_open_loop_metrics": val_metrics,
            "best_open_loop_record": record,
            "best_open_loop_metric_schema": metric_schema,
            "best_open_loop_selection_rule": self.selection_rule,
        }
        if self.validation_signature is not None:
            extra_state["validation_suite_signature"] = copy.deepcopy(self.validation_signature)
        save_fn(
            record["epoch"],
            self.best_path,
            extra_state=extra_state,
        )
        selector_label = (
            "Macro-L2 -> Collision -> Earlier Epoch"
            if self.selection_rule == "formal_v2_macro"
            else OPEN_LOOP_SELECTION_RULE
        )
        logger.info(
            f"*** 新最佳开环 checkpoint! Epoch {self.epoch} | 规则: {selector_label} | "
            f"L2_avg: {self.l2_avg:.5f}, Collision: {self.collision_rate:.5f}, "
            f"ADE: {self.ade:.5f}, FDE: {self.fde:.5f}, minFDE@K: {self.minfde_k:.5f} ***"
        )
        return True


class BestPredictorTracker:
    """Tracks the best predictor checkpoint by validation ``predictor_loss`` (lower is better).

    Shared by ``lewm_predictor`` and ``navsim_v2``: identical init + resume-restore (from
    ``best_predictor.pt``'s ``best_predictor_metrics``) + the min-loss comparison. Each line
    keeps its own save call and log message (their checkpoint save signatures and log text
    differ), so this owns only the state and the selection decision. Behaviour is identical
    to the previous inlined blocks.
    """

    def __init__(self, best_path: str, *, validation_signature: Optional[Dict[str, Any]] = None) -> None:
        self.best_path = best_path
        if validation_signature is not None and (
            not isinstance(validation_signature, dict) or not validation_signature
        ):
            raise ValueError("predictor validation_signature must be a non-empty dict when provided")
        self.validation_signature = copy.deepcopy(validation_signature)
        self.loss = float("inf")
        self.epoch = 0

    def restore(self, load_checkpoint_fn: Callable[[str], Optional[dict]]) -> bool:
        """Restore the best predictor loss/epoch from ``best_predictor.pt`` on resume.

        ``best_predictor.pt`` embeds ``best_predictor_metrics`` at top level (via
        extra_state); a resumed ``latest.pt`` does not, so without this a resumed run
        would overwrite a real best_predictor checkpoint on its first validation.
        """
        ckpt = load_checkpoint_fn(self.best_path)
        if ckpt is not None and ckpt.get("best_predictor_metrics") is not None:
            if self.validation_signature is not None:
                stored_signature = ckpt.get("predictor_validation_signature")
                if stored_signature is None:
                    raise RuntimeError(
                        f"best predictor checkpoint is missing predictor_validation_signature: {self.best_path}"
                    )
                if stored_signature != self.validation_signature:
                    raise RuntimeError(
                        "best predictor validation signature mismatch; checkpoint selection semantics "
                        f"changed for {self.best_path}"
                    )
            self.loss = float(ckpt["best_predictor_metrics"].get("predictor_loss", float("inf")))
            self.epoch = int(ckpt.get("epoch", 0))
            logger.info(
                "Restored best predictor tracker from %s (epoch=%d, loss=%.5f)",
                self.best_path,
                self.epoch,
                self.loss,
            )
            return True
        return False

    def consider(self, current_loss: float, epoch: int) -> bool:
        """If ``current_loss`` beats the best so far, adopt it and return True.

        The caller is responsible for saving the checkpoint and logging on a True return
        (those steps differ per line); state is updated here so ``self.epoch`` / ``self.loss``
        read correctly in the subsequent save/log/summary.
        """
        if current_loss < self.loss:
            self.loss = current_loss
            self.epoch = epoch
            return True
        return False

    def consider_suite_result(self, suite_result: Dict[str, Any], epoch: int) -> bool:
        """Select only the factual/all/full predictor loss from a structured suite."""

        try:
            factual_full = suite_result["metrics"]["real"]["all"]["full"]
            primary = suite_result["primary"]
            current_loss = float(factual_full["predictor_loss"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("predictor checkpoint selection requires metrics/real/all/full/predictor_loss") from exc
        if dict(primary) != dict(factual_full):
            raise ValueError("predictor suite primary must exactly match metrics/real/all/full")
        return self.consider(current_loss, epoch)
