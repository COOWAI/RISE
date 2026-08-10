"""Require independent provenance on the public diffusion implementation."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_MARKER = "# RISE provenance: independent-diffusion-v1"
TARGETS = (
    "app/vjepa_cowa_world_model/models/diffusion_planner.py",
    "app/vjepa_cowa_world_model/diffusion_utils/__init__.py",
    "app/vjepa_cowa_world_model/diffusion_utils/sde.py",
    "app/vjepa_cowa_world_model/diffusion_utils/sampling.py",
)


@pytest.mark.parametrize("relative_path", TARGETS)
def test_public_diffusion_source_has_independent_provenance(relative_path: str) -> None:
    text = (ROOT / relative_path).read_text(encoding="utf-8")
    folded = text.casefold()

    assert PROVENANCE_MARKER in text
    assert "adapted from xtr" not in folded
    assert "xtr/" not in folded
