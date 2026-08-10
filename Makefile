.PHONY: lint test package-check public-surface-check clean-tree-check release-check

PYTHON ?= python3

lint:
	$(PYTHON) -m isort app src tests --check
	$(PYTHON) -m black --check app src tests
	$(PYTHON) -m flake8 --config .flake8 --show-source --statistics app src tests

test:
	$(PYTHON) -m pytest -q tests

package-check:
	$(PYTHON) tools/check_package.py

public-surface-check:
	$(PYTHON) tools/check_public_surface.py .

clean-tree-check:
	@status="$$(git status --porcelain --untracked-files=all)" && test -z "$$status"

release-check:
	$(MAKE) lint
	$(MAKE) test
	$(MAKE) package-check
	$(MAKE) public-surface-check
	$(MAKE) clean-tree-check
