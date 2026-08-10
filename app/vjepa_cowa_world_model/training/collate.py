"""Shared batch-stacking helpers for the world-model collate functions.

The 3 dataset loaders (navsim / b2d / mongo_raw) stay separate (CLAUDE.md invariant #3), but their collate
functions share most boilerplate. These helpers carry the shared stacking; each loader keeps its own collate
entry point + tuple order, calling these. The helpers do NOT fabricate placeholders — missing required fields
raise (fail-loud); optional-field handling stays in each loader (b2d's `_stack_optional_field`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch


def stack_core(batch: Sequence[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack the world-model core: (context_frames, actions, states, extrinsics)."""
    context_frames = torch.stack([item["buffer"] for item in batch])
    actions = torch.stack([torch.from_numpy(item["actions"]) for item in batch])
    states = torch.stack([torch.from_numpy(item["states"]) for item in batch])
    extrinsics = torch.stack([torch.from_numpy(item["extrinsics"]) for item in batch])
    return context_frames, actions, states, extrinsics


def stack_required(batch: Sequence[Dict[str, Any]], key: str) -> torch.Tensor:
    """Stack a required per-item numpy field (raises if any item lacks it)."""
    return torch.stack([torch.from_numpy(item[key]) for item in batch])


def stack_agent_mask(batch: Sequence[Dict[str, Any]], key: str = "agent_mask") -> torch.Tensor:
    """Stack the agent validity mask as a bool tensor."""
    return torch.stack([torch.from_numpy(item[key].astype(np.uint8)).bool() for item in batch])


def stack_proposal_buffer(batch: Sequence[Dict[str, Any]]) -> Optional[torch.Tensor]:
    """Stack the optional proposal-encoder frames (None unless the first item carries `proposal_buffer`)."""
    if batch and batch[0].get("proposal_buffer") is not None:
        return torch.stack([item["proposal_buffer"] for item in batch])
    return None


def make_stable_sample_id(namespace: str, *identity_parts: Any) -> str:
    """Build a canonical sample identity without machine-specific absolute paths."""
    values = [str(namespace).strip(), *(str(part).strip() for part in identity_parts)]
    if any(not value for value in values):
        raise ValueError(f"stable_sample_id components must be non-empty, got {values!r}")
    absolute_parts = [value for value in values if os.path.isabs(value)]
    if absolute_parts:
        raise ValueError(f"stable_sample_id components cannot be absolute paths: {absolute_parts!r}")
    return ":".join(values)


def build_stable_sample_metadata(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate and collate the canonical per-sample identity contract."""
    stable_ids = []
    for index, item in enumerate(batch):
        stable_id = item.get("stable_sample_id")
        if not isinstance(stable_id, str) or not stable_id.strip():
            raise ValueError(f"Sample {index} requires a non-empty string stable_sample_id")
        alias = item.get("sample_id", stable_id)
        if alias != stable_id:
            raise ValueError(f"Sample {index} sample_id alias {alias!r} does not match stable_sample_id {stable_id!r}")
        stable_ids.append(stable_id)
    return {"stable_sample_id": stable_ids, "sample_id": list(stable_ids)}


def build_window_metadata(batch: Sequence[Dict[str, Any]], path_key: str) -> Dict[str, Any]:
    """Build the shared window metadata dict. `path_key` is the loader's source-path field (pkl_path/ann_file)."""
    metadata = build_stable_sample_metadata(batch)
    metadata.update(
        {
            "scene_name": [str(item.get("scene_name", "")) for item in batch],
            path_key: [str(item.get(path_key, "")) for item in batch],
            "window_start_pos": torch.as_tensor(
                [int(item.get("window_start_pos", -1)) for item in batch], dtype=torch.long
            ),
            "sampled_frame_indices": torch.stack(
                [
                    torch.as_tensor(item.get("sampled_frame_indices", item.get("indices", [])), dtype=torch.long)
                    for item in batch
                ]
            ),
            "image_frame_indices": torch.stack(
                [
                    torch.as_tensor(
                        item.get("image_frame_indices", item.get("sampled_frame_indices", item.get("indices", []))),
                        dtype=torch.long,
                    )
                    for item in batch
                ]
            ),
        }
    )
    sample_tokens: list[Optional[str]] = []
    sample_token_valid: list[bool] = []
    for index, item in enumerate(batch):
        if "sample_token" not in item:
            sample_tokens.append(None)
            sample_token_valid.append(False)
            continue
        token = item["sample_token"]
        if not isinstance(token, str) or not token:
            raise ValueError(f"Sample {index} sample_token must be a non-empty string, got {token!r}")
        sample_tokens.append(token)
        sample_token_valid.append(True)
    if any(sample_token_valid):
        metadata["sample_token"] = sample_tokens
        metadata["sample_token_valid_mask"] = torch.as_tensor(sample_token_valid, dtype=torch.bool)
    return metadata
