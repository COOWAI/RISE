"""Small shared module helpers (single source for previously duplicated one-liners)."""

from typing import Any


def unwrap_module(module: Any) -> Any:
    """剥离 DDP 等 .module 包装。"""
    return module.module if hasattr(module, "module") else module


def set_frozen_eval(module, name):
    """冻结模块并置 eval（原 train_reward_model/_validate 的 _set_frozen_eval）。"""
    if module is None:
        raise ValueError(f"{name} is required but None")
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def get_nested_config(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if current is None:
            return default
        if hasattr(current, key):
            current = getattr(current, key)
        elif isinstance(current, dict):
            current = current.get(key)
        else:
            return default
    return current if current is not None else default
