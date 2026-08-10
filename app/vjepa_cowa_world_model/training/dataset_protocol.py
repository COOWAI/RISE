"""The world-model dataset *builder protocol* — documentation + structural contract.

CLAUDE.md invariant #3: the three world-model loaders (``navsim_data`` / ``b2d_data`` /
``mongo_raw_data``) deliberately stay separate; their bodies are intentionally format-specific
(coordinate frames, optional-vs-mandatory metadata, PIL vs libclip vs Decord I/O). They are **not**
merged and do **not** inherit a common base class.

What they *do* share is a convention: each implements the same set of per-clip ``_build_*`` methods.
This module captures that convention as a ``typing.Protocol`` so an agent (or human) adding a new
world-model dataset has one place to read *what to implement*, with the expected tensor shapes. It is
a **documentation + static-typing aid only** — importing it has no runtime effect and the loaders are
not required to inherit it (structural typing).

To add a dataset, mirror an existing loader (``navsim_data.py`` is the reference), implement the
methods below with format-specific bodies, expose an ``init_<name>_data(...)`` factory, and register
it in ``training/data.py`` (see ``EXTENDING.md``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class WorldModelDatasetBuilder(Protocol):
    """Per-clip feature builders shared by every world-model loader (by convention, not inheritance).

    Shapes use ``T`` = frames-per-clip. Methods marked *conditional* return ``None`` when the source
    metadata is absent; the consumer (status_features / predictor_aux) then fails loud only when the
    active config actually requires the feature (never fabricates values — CLAUDE.md fail-loud).
    """

    # --- mandatory ---------------------------------------------------------------------------
    def _build_states(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Ego states ``[T, state_dim]`` in the loader's canonical frame (e.g. ego-centered)."""
        ...

    def _build_actions(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Per-step actions ``[T-1, action_dim]`` (e.g. ego-frame xyz delta + yaw)."""
        ...

    def _load_clip_images(self, *args: Any, **kwargs: Any) -> Any:
        """Decode/load the clip's frames (PIL list / tensor), format-specific I/O."""
        ...

    # --- conditional (return None when source metadata is missing) ---------------------------
    def _build_driving_commands(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        """One-hot navigation command ``[T, command_dim]`` or ``None`` if unavailable."""
        ...

    def _build_ego_dynamics(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        """Ego dynamics ``[T, 4]`` = (vx, vy, ax, ay) or ``None`` if unavailable."""
        ...

    def _build_agent_annotations(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """Dynamic-agent boxes in the ego frame, or ``None`` when not loaded."""
        ...

    def _build_bev_segmentation(self, *args: Any, **kwargs: Any) -> Optional[np.ndarray]:
        """Rasterized BEV agent occupancy, or ``None`` when not loaded."""
        ...


# The canonical builder-method names, so tooling/tests can assert a new loader implements the
# convention without importing the heavy loader modules.
WORLD_MODEL_BUILDER_METHODS: Sequence[str] = (
    "_build_states",
    "_build_actions",
    "_load_clip_images",
)
WORLD_MODEL_OPTIONAL_BUILDER_METHODS: Sequence[str] = (
    "_build_driving_commands",
    "_build_ego_dynamics",
    "_build_agent_annotations",
    "_build_bev_segmentation",
)
