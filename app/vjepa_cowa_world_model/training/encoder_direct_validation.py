"""Shared encoder-direct validation helpers.

This module keeps encoder-direct evaluation out of ``val_command.py`` because the
runtime is predictor-free: observed encoder tokens are fed directly to the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from app.vjepa_cowa_world_model.training.checkpoint import load_state_dict_helper
from app.vjepa_cowa_world_model.training.models import get_encoder_embed_dim
from app.vjepa_cowa_world_model.training.runtimes.encoder_token_runtime import (
    init_encoder_direct_encoder,
    init_encoder_direct_planner,
    resolve_encoder_direct_tokens_per_frame,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

EVAL_MODES = {"auto", "predictor", "encoder_direct"}


@dataclass
class EncoderDirectEvalRuntime:
    """Runtime modules needed by encoder-direct validation."""

    encoder: Any
    planner: Any
    tokens_per_frame: int
    encoder_embed_dim: int


def normalize_eval_mode(eval_mode: str | None) -> str:
    """Normalize and validate an eval-mode selector."""
    mode = str(eval_mode or "auto").strip().lower()
    if mode not in EVAL_MODES:
        raise ValueError(f"eval_mode must be one of {sorted(EVAL_MODES)}, got {eval_mode!r}")
    return mode


def should_use_encoder_direct_eval(config: Any, eval_mode: str | None = "auto") -> bool:
    """Return whether a config should use the predictor-free encoder-direct eval path."""
    mode = normalize_eval_mode(eval_mode)
    if mode == "encoder_direct":
        return True
    if mode == "predictor":
        return False

    meta = getattr(config, "meta", None)
    planner = getattr(config, "planner", None)
    load_predictor = bool(getattr(meta, "load_predictor", True))
    use_planner = bool(getattr(planner, "use_planner", False))
    return use_planner and not load_predictor


def explain_eval_mode_decision(config: Any, eval_mode: str | None = "auto") -> tuple[bool, str, str | None]:
    """Resolve the eval path together with a human-readable reason and an optional warning.

    The ``auto`` heuristic (``use_planner and not load_predictor``) silently decides between the
    predictor and predictor-free forward. Returning the reason lets the caller LOG which path was
    chosen and why, so a mis-set ``meta.load_predictor`` (or a forced ``eval_mode`` that contradicts
    the config) can no longer silently route a model through the wrong forward without a trace.

    Returns ``(use_encoder_direct, reason, warning_or_None)``.
    """
    mode = normalize_eval_mode(eval_mode)
    meta = getattr(config, "meta", None)
    planner = getattr(config, "planner", None)
    load_predictor = bool(getattr(meta, "load_predictor", True))
    use_planner = bool(getattr(planner, "use_planner", False))
    use_encoder_direct = should_use_encoder_direct_eval(config, mode)
    path = "encoder_direct (predictor-free)" if use_encoder_direct else "predictor"
    reason = f"eval_mode={mode} | use_planner={use_planner} | load_predictor={load_predictor} -> path={path}"

    warning: str | None = None
    if mode == "encoder_direct" and load_predictor:
        warning = (
            "eval_mode=encoder_direct forces the predictor-free path, but meta.load_predictor=True "
            "(this config declares a predictor); the predictor will be IGNORED at eval."
        )
    elif mode == "predictor" and not load_predictor:
        warning = (
            "eval_mode=predictor forces the predictor path, but meta.load_predictor=False "
            "(no predictor weights will load); the predictor forward will likely fail."
        )
    elif mode == "auto" and use_encoder_direct:
        reason += " [auto chose encoder_direct because load_predictor is False]"
    return use_encoder_direct, reason, warning


def init_encoder_direct_eval_runtime(config: Any, device: torch.device) -> EncoderDirectEvalRuntime:
    """Initialize encoder-direct encoder and planner modules for validation."""
    encoder, target_encoder = init_encoder_direct_encoder(config, device)
    del target_encoder
    encoder_embed_dim = get_encoder_embed_dim(encoder)
    planner = init_encoder_direct_planner(config, encoder_embed_dim, device, encoder=encoder)
    if planner is None:
        raise ValueError("planner.use_planner must be enabled for encoder-direct eval")
    tokens_per_frame = resolve_encoder_direct_tokens_per_frame(config, encoder)
    return EncoderDirectEvalRuntime(
        encoder=encoder,
        planner=planner,
        tokens_per_frame=tokens_per_frame,
        encoder_embed_dim=encoder_embed_dim,
    )


def _find_state_tensor(state_dict: dict[str, Any], key: str) -> torch.Tensor | None:
    value = state_dict.get(key)
    if torch.is_tensor(value):
        return value
    value = state_dict.get(f"module.{key}")
    if torch.is_tensor(value):
        return value
    return None


def resolve_encoder_direct_action_history_frames(config: Any, planner: Any) -> int:
    """Resolve the action-history token length expected by an encoder-direct planner."""
    planner_unwrapped = planner.module if hasattr(planner, "module") else planner
    if getattr(planner_unwrapped, "action_history_pos_embedding", None) is not None:
        num_frames = getattr(planner_unwrapped, "num_observed_frames", None)
        if num_frames is not None:
            return int(num_frames)
    if bool(getattr(getattr(config, "planner", None), "use_action_history_for_planner", False)):
        num_frames = getattr(planner_unwrapped, "num_observed_frames", None)
        if num_frames is not None:
            return int(num_frames)
    return int(getattr(getattr(config, "train", None), "num_observed_frames", 1))


def adapt_encoder_direct_action_history_from_checkpoint(
    runtime: EncoderDirectEvalRuntime,
    planner_state_dict: dict[str, Any],
) -> None:
    """Match action-history embedding length to legacy encoder-direct checkpoints."""
    source = _find_state_tensor(planner_state_dict, "action_history_pos_embedding.weight")
    if source is None:
        return

    planner = runtime.planner.module if hasattr(runtime.planner, "module") else runtime.planner
    target = getattr(planner, "action_history_pos_embedding", None)
    if target is None:
        return
    if tuple(target.weight.shape) == tuple(source.shape):
        return

    num_embeddings, embedding_dim = int(source.shape[0]), int(source.shape[1])
    config_num_observed = int(target.weight.shape[0])
    planner.action_history_pos_embedding = nn.Embedding(num_embeddings, embedding_dim).to(
        device=target.weight.device,
        dtype=target.weight.dtype,
    )
    planner.num_observed_frames = num_embeddings
    # Loud, not silent: resizing only the action-history embedding lets the checkpoint LOAD, but the
    # status vector, observed-token slicing and GT-origin all still use the eval config's
    # num_observed_frames. If the two disagree, action-history conditioning is misaligned with the
    # rest of the pipeline — a genuine config<->checkpoint mismatch, not a benign adaptation.
    logger.warning(
        "[encoder-direct eval] action-history length MISMATCH: checkpoint trained with "
        "num_observed_frames=%d, but this eval config built the planner with num_observed_frames=%d. "
        "Resized the action-history embedding to %d so the checkpoint loads, BUT status vector / "
        "observed-token slicing / GT-origin still use the config value (%d). If these disagree the "
        "action-history conditioning is misaligned with the rest of the pipeline; set "
        "train.num_observed_frames=%d to match the checkpoint and keep train==infer.",
        num_embeddings,
        config_num_observed,
        num_embeddings,
        config_num_observed,
        num_embeddings,
    )


def load_encoder_direct_checkpoint(
    checkpoint_path: str | Path, runtime: EncoderDirectEvalRuntime, device: torch.device
) -> None:
    """Load encoder/planner weights from an encoder-direct checkpoint.

    Predictor weights are intentionally ignored: encoder-direct evaluation does not
    instantiate a predictor.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"expected checkpoint dict: {checkpoint_path}")

    encoder_loaded = False
    for key in ("encoder", "target_encoder", "ema_encoder"):
        if key in checkpoint:
            load_state_dict_helper(runtime.encoder, checkpoint[key], f"encoder_direct encoder ({key})")
            encoder_loaded = True
            break
    if not encoder_loaded:
        # fail-loud (point 20): encoder 权重缺失直接报错，禁止用 init/pretrain 权重跑出"看似正常"的 ADE/FDE。
        raise KeyError(
            "No encoder weights (encoder/target_encoder/ema_encoder) found in encoder-direct checkpoint: "
            f"{checkpoint_path}"
        )

    if "planner" not in checkpoint:
        raise ValueError(f"checkpoint missing planner key: {checkpoint_path}")
    adapt_encoder_direct_action_history_from_checkpoint(runtime, checkpoint["planner"])
    load_state_dict_helper(runtime.planner, checkpoint["planner"], "encoder_direct planner")


def run_encoder_direct_validation(
    runtime: EncoderDirectEvalRuntime,
    val_loader: Any,
    val_sampler: Any,
    config: Any,
    epoch: int = 0,
    rank: int = 0,
    world_size: int = 1,
    vis_output_dir: str | None = None,
) -> dict[str, Any]:
    """Run the training-aligned encoder-direct validation loop."""
    from app.vjepa_cowa_world_model.training.lines.planner_encoder_only import _validate_encoder_direct

    runtime.encoder.eval()
    runtime.planner.eval()
    return _validate_encoder_direct(
        encoder=runtime.encoder,
        planner=runtime.planner,
        val_loader=val_loader,
        val_sampler=val_sampler,
        config=config,
        epoch=epoch,
        rank=rank,
        world_size=world_size,
        vis_output_dir=vis_output_dir,
    )
