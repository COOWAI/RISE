"""Lightweight, fail-loud contracts for planner modules and their forward outputs.

Single source of truth for:
  * the observed-token-mode vocabulary shared by every planner, and
  * the planner forward-output schema (:class:`PlannerOutput`) plus a validator.

This is **documentation + runtime checks, NOT a forced base class.** Planners stay
free-standing ``nn.Module``\\ s — the diffusion family keeps its deliberate inheritance
tree (CLAUDE.md invariant #4) and the multimodal planner stays a parallel tree. An agent
adding or editing a planner reads :class:`PlannerOutput` / :func:`validate_planner_output`
to know exactly what ``forward()`` must return, and :func:`normalize_observed_token_mode`
to know the legal observed-token modes.

Every planner ``forward`` returns one of two shapes:
  * **inference**: ``{"trajectories": [B, K, num_poses, 3], "confidences": [B, K], ...}``
  * **training**  (diffusion family, gt provided): ``{"loss", "reg_loss", "conf_loss", "cover_loss"}``
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, TypedDict

# ---------------------------------------------------------------------------
# Observed-token-mode vocabulary (canonical home; re-exported by the planners
# and by training/configs/common.py so there is exactly one source of truth).
# ---------------------------------------------------------------------------
PLANNER_OBSERVED_TOKEN_NONE = "none"
PLANNER_OBSERVED_TOKEN_CONCAT = "concat"
PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED = "concat_type_embed"
PLANNER_OBSERVED_TOKEN_MODES = {
    PLANNER_OBSERVED_TOKEN_NONE,
    PLANNER_OBSERVED_TOKEN_CONCAT,
    PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
}

_OBSERVED_TOKEN_ALIASES = {
    "off": PLANNER_OBSERVED_TOKEN_NONE,
    "false": PLANNER_OBSERVED_TOKEN_NONE,
    "simple": PLANNER_OBSERVED_TOKEN_CONCAT,
    "true": PLANNER_OBSERVED_TOKEN_CONCAT,
    "source_embed": PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    "type_embed": PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
    "type_embedding": PLANNER_OBSERVED_TOKEN_CONCAT_TYPE_EMBED,
}


def normalize_observed_token_mode(observed_token_mode: Any, use_observed_tokens: bool) -> str:
    """Resolve a raw ``observed_token_mode`` value to one of ``PLANNER_OBSERVED_TOKEN_MODES``.

    ``None`` falls back to ``concat`` when ``use_observed_tokens`` else ``none`` (preserving the
    pre-existing default). Unknown values fail loud. This is the single canonical implementation;
    ``diffusion_planner`` / ``multimodal_planner`` / ``configs.common`` import it.
    """
    if observed_token_mode is None:
        return PLANNER_OBSERVED_TOKEN_CONCAT if use_observed_tokens else PLANNER_OBSERVED_TOKEN_NONE
    mode = str(observed_token_mode).strip().lower().replace("-", "_")
    mode = _OBSERVED_TOKEN_ALIASES.get(mode, mode)
    if mode not in PLANNER_OBSERVED_TOKEN_MODES:
        raise ValueError(
            "observed_token_mode must be one of "
            f"{sorted(PLANNER_OBSERVED_TOKEN_MODES)}, got {observed_token_mode!r}"
        )
    return mode


# ---------------------------------------------------------------------------
# Planner forward-output schema + validator.
# ---------------------------------------------------------------------------
class PlannerOutput(TypedDict, total=False):
    """Schema of a planner ``forward`` return dict (mode-dependent; all keys optional).

    Inference keys:
        trajectories : ``[B, K, num_poses, 3]`` (x, y, yaw)
        confidences  : ``[B, K]`` unnormalized logits
    Training keys (diffusion family when ``gt_trajectory`` is provided):
        loss, reg_loss, conf_loss, cover_loss
    """

    trajectories: Any
    confidences: Any
    loss: Any
    reg_loss: Any
    conf_loss: Any
    cover_loss: Any


INFERENCE_KEYS = ("trajectories", "confidences")
TRAINING_LOSS_KEY = "loss"


def _shape(value: Any) -> Optional[tuple]:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else None


def validate_planner_output(
    out: Any,
    *,
    mode: Optional[str] = None,
    num_poses: Optional[int] = None,
    pose_dim: int = 3,
    required_training_keys: tuple = (),
) -> "PlannerOutput":
    """Fail-loud check that ``out`` matches the planner forward contract; returns ``out``.

    Args:
        out: the dict returned by a planner ``forward``.
        mode: ``"inference"`` | ``"training"`` | ``None``. ``None`` infers the mode — if a
            ``"loss"`` key is present it is treated as training, else inference.
        num_poses: if given, assert ``trajectories.shape[2] == num_poses``.
        pose_dim: required last-axis size of ``trajectories`` (default 3 = x, y, yaw).

    Raises:
        ValueError / TypeError on any violation (no silent pass).
    """
    if not isinstance(out, Mapping):
        raise TypeError(f"planner output must be a mapping, got {type(out).__name__}")
    if mode not in (None, "inference", "training"):
        raise ValueError(f"mode must be None|'inference'|'training', got {mode!r}")

    if mode is None:
        mode = "training" if TRAINING_LOSS_KEY in out else "inference"

    if mode == "training":
        if TRAINING_LOSS_KEY not in out:
            raise ValueError(f"training planner output missing required key {TRAINING_LOSS_KEY!r}; got {sorted(out)}")
        missing_extra = [k for k in required_training_keys if k not in out]
        if missing_extra:
            raise ValueError(f"training planner output missing required keys {missing_extra}; got {sorted(out)}")
        return out  # type: ignore[return-value]

    # inference mode
    missing = [k for k in INFERENCE_KEYS if k not in out]
    if missing:
        raise ValueError(f"inference planner output missing keys {missing}; got {sorted(out)}")
    traj_shape = _shape(out["trajectories"])
    conf_shape = _shape(out["confidences"])
    if traj_shape is None or len(traj_shape) != 4:
        raise ValueError(f"'trajectories' must be a 4-D tensor [B, K, num_poses, {pose_dim}], got shape {traj_shape}")
    if traj_shape[-1] != pose_dim:
        raise ValueError(f"'trajectories' last axis must be {pose_dim} (x, y, yaw), got {traj_shape[-1]}")
    if conf_shape is None or len(conf_shape) != 2:
        raise ValueError(f"'confidences' must be a 2-D tensor [B, K], got shape {conf_shape}")
    if traj_shape[:2] != conf_shape[:2]:
        raise ValueError(
            f"'trajectories' and 'confidences' must agree on [B, K]; " f"got {traj_shape[:2]} vs {conf_shape[:2]}"
        )
    if num_poses is not None and traj_shape[2] != num_poses:
        raise ValueError(f"'trajectories' num_poses axis must be {num_poses}, got {traj_shape[2]}")
    return out  # type: ignore[return-value]
