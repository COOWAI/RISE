"""Resolve `extends` in training YAMLs at load time (deep-merge).

This lets a runnable config compose a dataset-agnostic recipe (model/planner/loss/optimization) with a thin
per-dataset data profile (the `data:` block + dataset-specific tuning), instead of forking the whole config
tree per dataset. A flat yaml with no `extends` resolves to itself unchanged (idempotent). Fail-loud: missing
base, non-mapping yaml, dict-vs-scalar merge conflict, and `extends` cycles all raise.

Merge semantics: dicts deep-merge; lists and scalars replace wholesale; a file's own keys override its bases;
within an `extends` list, later entries override earlier ones. The `extends` key is stripped from the result.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) != isinstance(val, dict):
            raise ValueError(
                f"extends merge conflict at key '{key}': cannot merge "
                f"{'dict' if isinstance(out[key], dict) else type(out[key]).__name__} with "
                f"{'dict' if isinstance(val, dict) else type(val).__name__}"
            )
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val  # scalar or list: replace wholesale
    return out


def _resolve(path: Path, stack: List[str]) -> Dict[str, Any]:
    real = str(path.resolve())
    if real in stack:
        raise ValueError(f"extends cycle detected: {' -> '.join(stack + [real])}")
    if not path.is_file():
        raise FileNotFoundError(f"extends base config not found: {path}")
    with open(path) as handle:
        raw = yaml.load(handle, Loader=yaml.FullLoader)
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a YAML mapping, got {type(raw).__name__}: {path}")
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    bases = [extends] if isinstance(extends, str) else list(extends)
    merged: Dict[str, Any] = {}
    for base in bases:
        merged = _deep_merge(merged, _resolve(path.parent / base, stack + [real]))
    return _deep_merge(merged, raw)


def load_resolved_training_params(fname: Union[str, Path]) -> Dict[str, Any]:
    """Load a training YAML, resolving any `extends` chain into a single fully-merged params dict."""
    return _resolve(Path(fname), [])
