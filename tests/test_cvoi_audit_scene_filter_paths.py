"""Focused path-resolution tests for Formal-v2 CVoI audit scene filters."""

from pathlib import Path

import pytest

from app.vjepa_cowa_world_model.training import cvoi_audit


@pytest.mark.parametrize("split", ("navtrain", "navtest"))
def test_audit_resolves_configured_formal_scene_filter_for_exact_and_portable_modes(
    split: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured = f"configs/navsim/scene_filters/{split}.yaml"
    repository_root = Path(cvoi_audit.__file__).resolve().parents[3]
    receipt_path = repository_root / configured
    monkeypatch.chdir(tmp_path)

    resolved = cvoi_audit._resolve_cvoi_configured_scene_filter_path(
        configured,
        name=f"real_{split}.scene_filter_yaml",
    )

    assert resolved == receipt_path
    assert resolved.is_file()
    assert (
        cvoi_audit._select_cvoi_static_scene_filter_live_path(
            receipt_path,
            configured_path=configured,
            path_mode="exact",
            name=f"real_{split}",
        )
        == receipt_path
    )
    assert (
        cvoi_audit._select_cvoi_static_scene_filter_live_path(
            Path("/receipt/from/another/machine/scene-filter.yaml"),
            configured_path=configured,
            path_mode="portable_content",
            name=f"real_{split}",
        )
        == receipt_path
    )
    with pytest.raises(ValueError, match="exact repository-relative path"):
        cvoi_audit._resolve_cvoi_configured_scene_filter_path(
            f"scene_filter/{split}.yaml",
            name="invalid.relative",
        )
    with pytest.raises(ValueError, match="exact repository-relative path"):
        cvoi_audit._resolve_cvoi_configured_scene_filter_path(
            str(receipt_path),
            name="invalid.absolute",
        )
