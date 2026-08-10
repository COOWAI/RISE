"""Pytest collection-time isolation of stubbed ``sys.modules`` entries.

Several unit tests stub ``torch`` (and a couple of siblings) in ``sys.modules`` at
import time to exercise pure-logic helpers without loading the real, heavy module.
If such a test forgets to restore the real module — or fails mid-import after
installing its stub — every test module collected *afterwards* sees the stub and
errors with ``module 'torch' has no attribute 'cuda'``. That made the combined
``pytest tests/ app/.../tests/`` run abort during collection even though each file
passes in isolation.

This conftest makes the combined suite robust to that (current and future): it
captures the real ``torch`` family once, then restores it after each test module is
imported during collection. A test may still stub torch for its own body; it just
can no longer leak that stub into its siblings.
"""

import importlib
import sys

import pytest

# Import the real modules so they are loaded and captured before any test stubs them.
# These are the modules unit tests replace in sys.modules to run import-light: the
# torch family, the logging helpers, and the app utils facade.
_GUARDED = (
    "torch",
    "torchvision",
    "torch.utils",
    "torch.utils.data",
    "src.utils.logging",
    "app.vjepa_cowa_world_model.utils",
)
_REAL = {name: importlib.import_module(name) for name in _GUARDED}

# Some tests inject a stub planner via `_ensure_package("...models").MultiModalTemporalPlanner =
# _StubPlanner`, which mutates the REAL models facade *in place* (identity unchanged). sys.modules
# restoration cannot undo an in-place attribute mutation, so a later test doing
# `from app...models import MultiModalTemporalPlanner` would get the leaked stub. Snapshot the
# facade's __dict__ and restore it (drop added attrs, reset changed ones) alongside the sys.modules
# guards. The facade lazy-loads heavy classes via __getattr__, so the snapshot is light and any
# dropped attr just re-lazy-loads on next access.
_MODELS_FACADE_NAME = "app.vjepa_cowa_world_model.models"
try:
    _MODELS_FACADE = importlib.import_module(_MODELS_FACADE_NAME)
    _MODELS_FACADE_DICT = dict(_MODELS_FACADE.__dict__)
    _MODELS_FACADE_KEYS = set(_MODELS_FACADE_DICT)
except Exception:  # pragma: no cover - facade import is best-effort
    _MODELS_FACADE = None
    _MODELS_FACADE_DICT = {}
    _MODELS_FACADE_KEYS = set()


def _restore_guarded_modules() -> None:
    for name, real in _REAL.items():
        if sys.modules.get(name) is not real:
            sys.modules[name] = real
    if _MODELS_FACADE is not None and sys.modules.get(_MODELS_FACADE_NAME) is _MODELS_FACADE:
        cur = _MODELS_FACADE.__dict__
        if set(cur) != _MODELS_FACADE_KEYS or any(cur.get(k) is not v for k, v in _MODELS_FACADE_DICT.items()):
            cur.clear()
            cur.update(_MODELS_FACADE_DICT)


class _TorchIsolatingModule(pytest.Module):
    """A pytest Module that restores the real guarded modules after importing the test file."""

    def _getobj(self):
        try:
            return super()._getobj()
        finally:
            _restore_guarded_modules()


@pytest.hookimpl
def pytest_pycollect_makemodule(module_path, parent):
    return _TorchIsolatingModule.from_parent(parent, path=module_path)


@pytest.fixture(autouse=True)
def _isolate_guarded_modules():
    """Restore the guarded modules after every test.

    Collection-time restoration (above) stops stubs leaking between *files*; this stops
    them leaking between *tests* within a run when a test installs a stub at runtime
    (e.g. in setUp / a monkeypatch) without tearing it down. Tests that legitimately
    install a stub for their own body are unaffected — restoration happens only after
    the test finishes.
    """
    snapshot = dict(sys.modules)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name not in snapshot:
                del sys.modules[name]
        sys.modules.update(snapshot)
        _restore_guarded_modules()
