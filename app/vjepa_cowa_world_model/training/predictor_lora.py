"""Predictor LoRA helpers shared by training entries (navsim_v2 superset variant kept)."""

import torch

from src.utils.logging import get_logger

logger = get_logger(__name__)


_KNOWN_PREDICTOR_LORA_KEYS = frozenset({"enabled", "r", "alpha", "dropout", "target_modules", "train_bias"})


def apply_predictor_lora(predictor: torch.nn.Module, lora_cfg: dict) -> torch.nn.Module:
    """Wrap predictor with PEFT LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    # `predictor_lora` is a top-level section the parse-time guard skips, so validate keys here (fail loud on
    # typos like `dropuot` that would otherwise be silently dropped to a default).
    unknown = set(lora_cfg) - _KNOWN_PREDICTOR_LORA_KEYS
    if unknown:
        raise ValueError(
            f"Unknown predictor_lora key(s) {sorted(unknown)}; valid: {sorted(_KNOWN_PREDICTOR_LORA_KEYS)}."
        )

    # structural LoRA knobs are direct-indexed (fail loud if missing) — they change the adapted model;
    # dropout is the one optional regularizer with a standard default.
    r = int(lora_cfg["r"])
    lora_alpha = int(lora_cfg.get("alpha", r))  # convention: alpha defaults to r
    lora_dropout = float(lora_cfg.get("dropout", 0.0))
    target_modules = lora_cfg["target_modules"]

    logger.info(
        "Applying predictor LoRA: r=%d alpha=%d dropout=%.3f targets=%s",
        r,
        lora_alpha,
        lora_dropout,
        target_modules,
    )

    predictor = get_peft_model(
        predictor,
        LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        ),
    )

    lora_params = sum(p.numel() for n, p in predictor.named_parameters() if p.requires_grad and "lora_" in n)
    total_params = sum(p.numel() for p in predictor.parameters())
    logger.info(
        "Predictor LoRA params: %.2fM / total predictor params: %.2fM",
        lora_params / 1e6,
        total_params / 1e6,
    )
    return predictor


def set_predictor_lora_trainable(predictor: torch.nn.Module, train_bias: bool = False) -> None:
    """Freeze all predictor params except LoRA adapters (and optional bias)."""
    for n, p in predictor.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        elif train_bias and n.endswith("bias"):
            p.requires_grad = True
        else:
            p.requires_grad = False
