# Contributing to RISE

Thank you for helping improve RISE. This repository is a research codebase, so focused changes with reproducible
local checks are the easiest to review.

## Before opening a change

1. Search the issue tracker for an existing report or proposal.
2. For a bug, include a minimal reproduction and the exact error. For a larger feature, open a feature request first
   so its scope and compatibility with the manual seven-stage workflow can be discussed.
3. Do not include datasets, model weights, generated results, credentials, private endpoints, or machine-specific
   paths in a contribution.

## Development setup

RISE requires Python 3.11 or newer. From a clean checkout, install the project and its test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run the relevant focused tests while developing, then run the full local checks before opening a pull request:

```bash
python -m isort app src tests --check
python -m black --check app src tests
python -m flake8 --config .flake8 --show-source --statistics app src tests
python -m pytest -q tests
```

GPU training, Oracle construction, and EPDMS scoring are not required for a documentation-only or unit-tested
change. Never claim those checks unless they were actually run in a correctly provisioned environment.

## Change guidelines

- Keep changes narrowly scoped and explain their user-visible effect.
- Add or update tests for behavior changes and preserve fail-fast validation.
- Preserve existing third-party copyright and license headers.
- Keep the seven CVoI stages as independent flat YAML files with manual stage handoff; do not introduce automatic
  scheduling or candidate selection.
- Use neutral `/path/to/` examples in tracked configuration and documentation.
- Update the changelog when a change affects users.

By submitting a contribution, you agree that your contribution may be distributed under the repository's MIT
license, while existing third-party components remain under their stated licenses.
