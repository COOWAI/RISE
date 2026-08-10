# V-JEPA Neutralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the vendored legacy directory and all retired project-name variants while preserving the V-JEPA encoder, NavSim scene filters, planner outputs, and packaging behavior.

**Architecture:** Retain only the two runtime scene-filter manifests under the repository's canonical `configs/navsim/` tree, use the existing canonical Apache-2.0 license, and rename the image-adapter surface to `vjepa`. Replace the planner's hidden external trajectory-head import with a small first-party module that preserves its parameter and output contracts. Enforce the result with a tracked-source zero-residue test and package-level verification.

**Tech Stack:** Python 3.11, PyTorch, pytest, Hydra/OmegaConf YAML, setuptools, Bash, Git.

---

## File Map

- Create `app/vjepa_cowa_world_model/models/trajectory_head.py`: self-contained trajectory regression head.
- Rename `app/vjepa_cowa_world_model/models/vjepa_img_encoder.py`: neutral V-JEPA image adapter and GridMask implementation.
- Rename `app/vjepa_cowa_world_model/training/vjepa_transforms.py`: neutral V-JEPA image preprocessing.
- Move `configs/navsim/scene_filters/navtrain.yaml` and `configs/navsim/scene_filters/navtest.yaml`: canonical retained NavSim split manifests.
- Modify `app/vjepa_cowa_world_model/models/multimodal_planner.py` and `app/vjepa_cowa_world_model/models/refinement_decoder.py`: consume the first-party trajectory head.
- Modify `app/vjepa_cowa_world_model/training/configs/{common,core,cvoi,parse,planner}.py`: expose only `vjepa` configuration fields and validators.
- Modify training/evaluation modules under `app/vjepa_cowa_world_model/{training,evaluation}/`: rename imports, helpers, runtime markers, and dispatch values without changing computation.
- Modify the seven YAML files under `configs/train/navsim/cvoi_manual_full/`: use `vjepa_img_encoder`, `vjepa_*` fields, and canonical scene-filter paths where applicable.
- Modify `MANIFEST.in`, `THIRD_PARTY_NOTICES.md`, `tools/check_public_surface.py`, and packaging/licensing tests: package and attribute the new paths only.
- Modify affected tests under `tests/`: assert the new public contract and remove external namespace mocks.

### Task 1: Internalize the trajectory regression head

**Files:**
- Create: `app/vjepa_cowa_world_model/models/trajectory_head.py`
- Modify: `app/vjepa_cowa_world_model/models/multimodal_planner.py`
- Modify: `app/vjepa_cowa_world_model/models/refinement_decoder.py`
- Modify: `tests/models/test_planner_output_contract.py`
- Modify: `tests/test_planner_observed_token_mode.py`
- Modify: `tests/test_status_features.py`

- [ ] **Step 1: Write the failing trajectory-head contract test**

Add the import and test below to `tests/models/test_planner_output_contract.py`:

```python
import math

from app.vjepa_cowa_world_model.models.trajectory_head import TrajectoryHead


class TestTrajectoryHead(unittest.TestCase):
    def test_preserves_parameter_and_output_contract(self):
        head = TrajectoryHead(num_poses=3, d_ffn=16, d_model=8)

        self.assertEqual(
            tuple(head.state_dict()),
            ("_mlp.0.weight", "_mlp.0.bias", "_mlp.2.weight", "_mlp.2.bias"),
        )
        result = head(torch.randn(2, 3, 8))

        self.assertEqual(set(result), {"trajectory"})
        self.assertEqual(result["trajectory"].shape, (2, 3, 3))
        self.assertTrue(torch.all(result["trajectory"][..., 2].abs() <= math.pi))
```

- [ ] **Step 2: Run the test and confirm the module is missing**

Run:

```bash
pytest tests/models/test_planner_output_contract.py::TestTrajectoryHead -q
```

Expected: collection fails with `ModuleNotFoundError` for `models.trajectory_head`.

- [ ] **Step 3: Add the minimal first-party implementation**

Create `app/vjepa_cowa_world_model/models/trajectory_head.py`:

```python
"""Small trajectory regression head shared by RISE planners."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

TRAJECTORY_STATE_SIZE = 3
TRAJECTORY_HEADING_INDEX = 2


class TrajectoryHead(nn.Module):
    """Map per-pose query tokens to bounded planar poses."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int) -> None:
        super().__init__()
        self._num_poses = int(num_poses)
        self._d_model = int(d_model)
        self._d_ffn = int(d_ffn)
        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, TRAJECTORY_STATE_SIZE),
        )

    def forward(self, object_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, TRAJECTORY_STATE_SIZE)
        poses[..., TRAJECTORY_HEADING_INDEX] = (
            poses[..., TRAJECTORY_HEADING_INDEX].tanh() * math.pi
        )
        return {"trajectory": poses}
```

- [ ] **Step 4: Remove the lazy external import**

In `multimodal_planner.py`, import `TrajectoryHead` from the new module, delete `_load_trajectory_head()` and `__getattr__()`, and replace the constructor lookup with:

```python
trajectory_head_cls = TrajectoryHead
```

In `refinement_decoder.py`, use focused imports:

```python
from .multimodal_planner import MultiModalTemporalPlanner
from .trajectory_head import TrajectoryHead
```

Remove the temporary trajectory-head classes, old package entries in `sys.modules`, and related comments from `tests/test_planner_observed_token_mode.py` and `tests/test_status_features.py`. Remove the `try/except ImportError` skip from the real multimodal planner test because construction is now self-contained.

- [ ] **Step 5: Run focused planner tests**

Run:

```bash
pytest tests/models/test_planner_output_contract.py tests/test_planner_observed_token_mode.py tests/test_status_features.py -q
```

Expected: all selected tests pass with no NavSim package installed.

- [ ] **Step 6: Commit the self-contained planner boundary**

```bash
git add app/vjepa_cowa_world_model/models/trajectory_head.py app/vjepa_cowa_world_model/models/multimodal_planner.py app/vjepa_cowa_world_model/models/refinement_decoder.py tests/models/test_planner_output_contract.py tests/test_planner_observed_token_mode.py tests/test_status_features.py
git commit -m "refactor: internalize trajectory head"
```

### Task 2: Move the retained scene filters and canonicalize licensing

**Files:**
- Move: legacy `navtrain.yaml` to `configs/navsim/scene_filters/navtrain.yaml`
- Move: legacy `navtest.yaml` to `configs/navsim/scene_filters/navtest.yaml`
- Delete: redundant vendored Apache license and its now-empty parent tree
- Modify: `app/vjepa_cowa_world_model/training/cvoi_formal_v2_navsim_roots.py`
- Modify: `configs/train/navsim/cvoi_manual_full/{01_predictor_lewm_pure,02_p0_uniform,03_field_full,04_calibration_full,05_p1_full,06_stop_full}.yaml`
- Modify: `MANIFEST.in`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `tools/check_public_surface.py`
- Modify: `tests/test_cvoi_audit_scene_filter_paths.py`
- Modify: `tests/test_cvoi_formal_v2_navsim_roots.py`
- Modify: `tests/test_navsim_cvoi_offline_adapter.py`
- Modify: `tests/test_minimal_repository_surface.py`
- Modify: `tests/test_public_licensing.py`
- Modify: `tests/test_public_packaging.py`
- Modify: `tests/test_public_repository_surface.py`

- [ ] **Step 1: Change path and license tests to the intended layout**

Use these path constants in path-resolution tests and expected catalog values:

```python
TRAIN_SCENE_FILTER = "configs/navsim/scene_filters/navtrain.yaml"
TEST_SCENE_FILTER = "configs/navsim/scene_filters/navtest.yaml"
```

Change the minimal-surface assertion to:

```python
scene_filter_files = {
    path.relative_to(REPO_ROOT).as_posix()
    for path in (REPO_ROOT / "configs/navsim/scene_filters").glob("*.yaml")
}
assert scene_filter_files == {
    "configs/navsim/scene_filters/navtrain.yaml",
    "configs/navsim/scene_filters/navtest.yaml",
}
```

Change the Apache license test to assert the single canonical file directly:

```python
apache_text = _read("licenses/Apache-2.0.txt")
assert "Apache License\n                           Version 2.0, January 2004" in apache_text
```

Require both new YAML paths in the public-surface fixture and require `configs/navsim/scene_filters` rather than a vendored root in the sdist manifest test.

- [ ] **Step 2: Run the changed tests and confirm they fail on missing paths**

Run:

```bash
pytest tests/test_cvoi_audit_scene_filter_paths.py tests/test_cvoi_formal_v2_navsim_roots.py tests/test_minimal_repository_surface.py tests/test_public_licensing.py tests/test_public_packaging.py tests/test_public_repository_surface.py -q
```

Expected: failures identify missing `configs/navsim/scene_filters/*.yaml` and stale packaging/license expectations.

- [ ] **Step 3: Move the manifests without modifying their bytes**

Create `configs/navsim/scene_filters/`, move both YAML files with `git mv`, and remove the redundant vendored license with `git rm`. Confirm the retained hashes:

```bash
legacy_root='Drive_'JEPA
mkdir -p configs/navsim/scene_filters
git mv "$legacy_root/navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml" configs/navsim/scene_filters/navtrain.yaml
git mv "$legacy_root/navsim_v2/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml" configs/navsim/scene_filters/navtest.yaml
git rm "$legacy_root/LICENSE"
```

```text
c37fea567a0cfdbc29076cca893d4f5dd32db59baec18ae214527206d6b64e6f  navtrain.yaml
61284edf5003c0291f843ce9817c822ba306609a62d54544223adae3fc7fc9cd  navtest.yaml
```

Run:

```bash
shasum -a 256 configs/navsim/scene_filters/navtrain.yaml configs/navsim/scene_filters/navtest.yaml
```

Expected: the two hashes exactly match the values above.

- [ ] **Step 4: Update every scene-filter consumer**

Set the root catalog constants to:

```python
_FORMAL_V2_NAVSIM_TRAIN_SCENE_FILTER = "configs/navsim/scene_filters/navtrain.yaml"
_FORMAL_V2_NAVSIM_TEST_SCENE_FILTER = "configs/navsim/scene_filters/navtest.yaml"
```

Replace the two repository-relative values in each of the six listed training YAML files and update the audit/offline-adapter tests to the same paths.

- [ ] **Step 5: Update packaging, public-surface, and notices**

Remove the vendored recursive include from `MANIFEST.in`; the existing `recursive-include configs *.md *.yaml *.yml` includes both manifests. In `tools/check_public_surface.py`, drop the redundant vendored license requirement and require the two new YAML paths. In `THIRD_PARTY_NOTICES.md`, replace the vendored-tree bullet with a path-specific statement that the two NavSim scene-filter manifests are Apache-2.0 components covered by `licenses/Apache-2.0.txt`.

Remove the retired root alternative from `DOCUMENTED_PATH_PATTERN` in `tests/test_minimal_repository_surface.py` so only `app|configs|docs|scripts|src|tests|tools` remain.

- [ ] **Step 6: Run focused path, license, and package-surface tests**

Run:

```bash
pytest tests/test_cvoi_audit_scene_filter_paths.py tests/test_cvoi_formal_v2_navsim_roots.py tests/test_navsim_cvoi_offline_adapter.py tests/test_minimal_repository_surface.py tests/test_public_licensing.py tests/test_public_packaging.py tests/test_public_repository_surface.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit the directory extraction**

```bash
git add MANIFEST.in THIRD_PARTY_NOTICES.md app configs tests tools
git commit -m "refactor: extract NavSim scene filters"
```

### Task 3: Rename the V-JEPA adapter contract atomically

**Files:**
- Rename: `app/vjepa_cowa_world_model/models/vjepa_img_encoder.py`
- Rename: `app/vjepa_cowa_world_model/training/vjepa_transforms.py`
- Modify: all matching Python/YAML/test files under `app/`, `configs/`, `tests/`, and `tools/`

- [ ] **Step 1: Add the zero-residue tracked-source test**

Add this contract to `tests/test_public_repository_surface.py`:

```python
_RETIRED_PROJECT_PATTERN = re.compile("drive" + r"[-_]?" + "jepa", flags=re.IGNORECASE)


def test_tracked_sources_contain_no_retired_project_identifiers() -> None:
    tracked_paths = _run_git(REPO_ROOT, "ls-files").splitlines()
    violations: list[str] = []
    for relative_path in tracked_paths:
        if _RETIRED_PROJECT_PATTERN.search(relative_path):
            violations.append(relative_path)
            continue
        path = REPO_ROOT / relative_path
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if _RETIRED_PROJECT_PATTERN.search(content):
            violations.append(relative_path)
    assert not violations, f"tracked paths or text retain retired identifiers: {sorted(violations)}"
```

- [ ] **Step 2: Run the contract and confirm it exposes the full rename surface**

Run:

```bash
pytest tests/test_public_repository_surface.py::test_tracked_sources_contain_no_retired_project_identifiers -q
```

Expected: failure lists the two legacy module paths and every tracked text file that still uses a retired spelling.

- [ ] **Step 3: Rename the two modules**

Use `git mv` so history is preserved:

```bash
legacy_stem='drive_'jepa
git mv "app/vjepa_cowa_world_model/models/${legacy_stem}_img_encoder.py" app/vjepa_cowa_world_model/models/vjepa_img_encoder.py
git mv "app/vjepa_cowa_world_model/training/${legacy_stem}_transforms.py" app/vjepa_cowa_world_model/training/vjepa_transforms.py
```

- [ ] **Step 4: Apply the mechanical identifier mapping to every matching tracked text file**

Apply this exact mapping, including imports, dataclass fields, helper names, backbone strings, capability markers, error messages, comments, and tests:

```text
retired snake_case prefix -> vjepa
retired CamelCase prefix  -> VJEPA
retired hyphenated name   -> V-JEPA
```

The key resulting interfaces are:

```python
BACKBONE_TYPES = ("vjepa2", "vjepa2.1", "vjepa_img_encoder", "dinov2_img_encoder")

class ModelConfig:
    vjepa_resolution: tuple[int, int] = (256, 512)
    vjepa_crop_top_bottom: int = 28
    vjepa_num_frames: int = 2
    vjepa_checkpoint_key: str = "target_encoder"
    vjepa_use_grid_mask: bool = True
    vjepa_use_causal_attention: bool = True
```

Rename the exported adapter surface to:

```python
__all__ = ["VJEPAGridMask", "VJEPAImgEncoderAdapter"]
```

Rename the existing class declarations to `VJEPAGridMask` and `VJEPAImgEncoderAdapter` without changing their method bodies. Set the adapter capability marker to `is_vjepa_img_encoder_adapter = True`. Apply the equivalent `vjepa_*` field set to `ProposalConfig` and all parsing/default inheritance logic. Do not read old keys or expose aliases.

- [ ] **Step 5: Update a focused configuration assertion**

In `tests/test_cvoi_formal_v2_navsim_e120_config.py`, assert:

```python
model = compatibility["model"]
assert model["backbone"] == "vjepa_img_encoder"
assert model["vjepa_resolution"] == [256, 512]
retired_prefix = "drive" + "_" + "jepa"
assert all(retired_prefix not in key.casefold() for key in model)
```

- [ ] **Step 6: Format and run the affected configuration, encoder, runtime, and evaluation tests**

Run:

```bash
python3 -m isort app tests --check
python3 -m black app tests --check
pytest tests/test_cvoi_formal_v2_navsim_e120_config.py tests/test_cvoi_formal_v2_navsim_e120_encoder_initialization.py tests/test_cvoi_runtime.py tests/test_navsim_cvoi_model_runtime.py tests/evaluation/test_navsim_stage3_alignment.py tests/test_status_features.py tests/test_public_repository_surface.py -q
```

Expected: format checks and all selected tests pass, including the zero-residue contract.

- [ ] **Step 7: Commit the neutral naming surface**

```bash
git add app configs tests tools
git commit -m "refactor: rename V-JEPA adapter surface"
```

### Task 4: Verify the complete repository and release artifacts

**Files:**
- Modify only files required by verification failures; no compatibility shims.

- [ ] **Step 1: Scan paths and text independently of pytest**

Construct the search expression without embedding the retired identifier as contiguous text:

```bash
retired_pattern='drive[-_]?'jepa
rg -n -i "$retired_pattern" . --hidden -g '!.git/**'
```

Expected: exit code `1` and no output. Also run:

```bash
git ls-files | rg -i "$retired_pattern"
```

Expected: exit code `1` and no output.

- [ ] **Step 2: Confirm the vendored directory is absent and retained manifests are unchanged**

Run a top-level directory listing and the SHA-256 command from Task 2.

Expected: no vendored legacy directory; both hashes match the recorded values.

- [ ] **Step 3: Run the complete test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass with no collection errors, failures, or unexpected skips caused by the removed namespace.

- [ ] **Step 4: Run repository formatting and release checks**

Run:

```bash
make lint
python3 tools/check_package.py
```

Expected: lint succeeds; package output contains `wheel: ok`, `sdist: ok`, and `installed artifact smoke checks: ok`.

- [ ] **Step 5: Review the final diff for semantic and rename integrity**

Run:

```bash
git diff --check
git diff --stat 25e16d4..HEAD
git status --short
```

Expected: no whitespace errors; the scene-filter files appear as byte-preserving renames; the worktree is clean unless verification corrections are pending.

- [ ] **Step 6: Commit any verification-only corrections**

If verification required corrections, stage only those files and commit:

```bash
git commit -m "test: enforce neutral V-JEPA repository surface"
```

If no corrections were needed, do not create an empty commit.
