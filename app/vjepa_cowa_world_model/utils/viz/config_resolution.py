"""Helpers for resolving visualization configs from checkpoints."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def get_checkpoint_sidecar_config_path(checkpoint_path: Optional[str]) -> Optional[str]:
    if not checkpoint_path:
        return None
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    sidecar_path = os.path.join(checkpoint_dir, "params-pretrain.yaml")
    if os.path.isfile(sidecar_path):
        return sidecar_path
    return None


def load_visualization_config(
    config_path: Optional[str],
    checkpoint_path: Optional[str],
) -> Dict[str, Any]:
    """
    Resolve the config used for visualization, fail-loud.

    Priority:
    1. checkpoint-sidecar params-pretrain.yaml (the exact config the checkpoint was trained with)
    2. explicitly provided config_path

    Raises if neither is available. Visualization must use the *same* config as training so the
    model architecture / preprocessing / forward match exactly; falling back to a fabricated
    default would silently build a mismatched model (the inconsistency this whole path exists to
    prevent), so we refuse rather than guess.
    """
    sidecar_path = get_checkpoint_sidecar_config_path(checkpoint_path)
    if sidecar_path is not None:
        resolved = _read_yaml(sidecar_path)
        resolved["__config_source__"] = sidecar_path
        return resolved

    if config_path is not None:
        resolved = _read_yaml(config_path)
        resolved["__config_source__"] = config_path
        return resolved

    raise ValueError(
        "Cannot resolve a visualization config: no checkpoint sidecar "
        "(params-pretrain.yaml next to the checkpoint) was found and no explicit --config was "
        "given. Visualization must use the same config as training; refusing to fall back to a "
        "fabricated default that would silently build a mismatched model. "
        f"(checkpoint_path={checkpoint_path!r}, config_path={config_path!r})"
    )
