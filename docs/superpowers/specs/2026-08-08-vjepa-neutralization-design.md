# V-JEPA Naming and Scene-Filter Migration Design

## Goal

Remove the vendored legacy directory and every spelling variant of its retired project identifier from the repository while preserving the V-JEPA image-encoder behavior and the two NavSim scene-filter manifests used by training and evaluation.

## Scope

The migration is intentionally breaking. It will not provide compatibility aliases for old Python imports, class names, configuration keys, backbone values, runtime markers, or repository-relative scene-filter paths. Checkpoint tensor keys and model computation are not changed unless they currently contain one of the retired identifiers.

## Repository Layout

Move the two retained manifests without changing their bytes:

- `navtrain.yaml` to `configs/navsim/scene_filters/navtrain.yaml`
- `navtest.yaml` to `configs/navsim/scene_filters/navtest.yaml`

Delete the remaining vendored directory. Its `LICENSE` file is byte-identical to `licenses/Apache-2.0.txt`, so the canonical Apache-2.0 license remains at the existing root-level license path. Update `THIRD_PARTY_NOTICES.md` to attribute the retained NavSim scene-filter manifests at their new paths and reference the canonical license.

## Naming Contract

Use `vjepa` as the neutral feature name throughout the repository:

- The legacy image-encoder module becomes `vjepa_img_encoder.py`.
- The legacy transform module becomes `vjepa_transforms.py`.
- The backbone value becomes `vjepa_img_encoder`.
- Configuration fields and Python identifiers use the `vjepa_` prefix.
- The adapter, image-transform, and grid-mask classes become `VJEPAImgEncoderAdapter`, `VJEPAImageTransform`, and `VJEPAGridMask`.
- Runtime capability markers use `is_vjepa_img_encoder_adapter`.

Existing `vjepa2` and `vjepa2.1` backbone values remain distinct. The renamed `vjepa_img_encoder` adapter keeps its current preprocessing, checkpoint-loading, and non-square-resolution behavior.

## Dependency Updates

Update all consumers atomically:

- The retained training configurations and formal NavSim root catalog point to the new scene-filter paths and use the new backbone/configuration names.
- Training, evaluation, model factories, runtime lines, validation, and feature-building modules import and call the renamed interfaces.
- The planner's small trajectory-regression head becomes a first-party module, preserving its constructor, parameter hierarchy, output mapping, and bounded-heading behavior while removing the hidden legacy namespace import.
- Packaging and public-surface declarations include the new scene-filter files and no longer require or package the removed directory.
- Tests, fixtures, comments, error messages, and mock module names use the new terminology.

No old-name parser shim or migration fallback will be added. A user supplying an old configuration receives the existing unknown-backbone or missing-field validation error.

## Verification

Add a repository-surface contract that constructs the retired search terms from noncontiguous fragments and fails if a tracked path or tracked text file contains any retired identifier, case-insensitively. Verify the change with:

1. A whole-repository path and content scan for zero retired identifiers.
2. Scene-filter path, configuration, licensing, packaging, encoder, evaluation, and runtime tests.
3. The complete test suite.
4. Source-distribution and wheel checks, including installed-artifact smoke tests.
5. A final Git diff review confirming that the large YAML files are byte-preserving renames.

## Risks and Controls

The principal risk is an incomplete atomic rename across configuration-driven branches. Targeted tests and the zero-residue contract cover that boundary. The second risk is accidental modification of the large scene-filter manifests; hash comparison before and after migration confirms byte identity. The third risk is loss of third-party licensing information; the canonical Apache-2.0 text and path-specific notice remain in the repository and package.
