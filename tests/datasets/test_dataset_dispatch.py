"""Phase 0 safety net: lock the dataset-dispatch selection logic.

The dataset loader is selected from config in ``training/data.py`` via an if/elif chain
(navsim / bench2drive / mongo_raw / legacy-seg). Phase 3 replaces that chain with a static
registry; these tests lock the externally-observable behavior so the registry refactor cannot
silently change which loader a given config selects.

Selection is exercised by monkeypatching each ``init_*`` factory with a sentinel, so no real
data is touched.
"""

import pytest

from app.vjepa_cowa_world_model.training import data as data_mod
from app.vjepa_cowa_world_model.training.configs.parse import parse_training_config


class _Loader:
    def __init__(self, name):
        self.name = name

    def __len__(self):  # create_*_dataloader logs len(loader)
        return 0


def _stub(name):
    def _factory(*args, **kwargs):
        return _Loader(name), None

    return _factory


def _config(**sections):
    cfg = {
        "data": {
            "num_target_frames": 8,
            "crop_size": 32,
            "patch_size": 16,
            "fps": 2,
            "batch_size": 2,
            "num_workers": 0,
        },
        "train": {"num_observed_frames": 4},
        "segmentation": {"use_segmentation": False},
    }
    for section, values in sections.items():
        cfg.setdefault(section, {}).update(values)
    return parse_training_config(cfg)


# --- predicate characterization (pure) -------------------------------------------------------
def test_enabled_predicates_reflect_config():
    navsim_cfg = _config(data={"navsim": {"enabled": True, "data_path": "/a", "sensor_blobs_path": "/b"}})
    assert data_mod._is_navsim_enabled(navsim_cfg)
    assert not data_mod._is_bench2drive_enabled(navsim_cfg)
    assert not data_mod._is_mongo_raw_enabled(navsim_cfg)

    none_cfg = _config(data={"datasets": ["/seg"]})
    assert not data_mod._is_navsim_enabled(none_cfg)
    assert not data_mod._is_bench2drive_enabled(none_cfg)
    assert not data_mod._is_mongo_raw_enabled(none_cfg)


def test_select_returns_single_enabled_or_none():
    navsim_cfg = _config(data={"navsim": {"enabled": True, "data_path": "/a", "sensor_blobs_path": "/b"}})
    assert data_mod._select_world_model_dataset(navsim_cfg) == "navsim"
    mongo_cfg = _config(data={"mongo_raw": {"enabled": True}})
    assert data_mod._select_world_model_dataset(mongo_cfg) == "mongo_raw"
    none_cfg = _config(data={"datasets": ["/seg"]})
    assert data_mod._select_world_model_dataset(none_cfg) is None


def test_multiple_enabled_fails_loud():
    cfg = _config(
        data={
            "navsim": {"enabled": True, "data_path": "/a", "sensor_blobs_path": "/b"},
            "mongo_raw": {"enabled": True},
        }
    )
    with pytest.raises(ValueError, match="Multiple datasets enabled"):
        data_mod._select_world_model_dataset(cfg)


# --- single-enabled selection (the contract Phase 3 must preserve) ---------------------------
@pytest.fixture
def stub_factories(monkeypatch):
    # navsim is imported into data_mod's namespace at top; the others are imported lazily inside
    # create_*_dataloader. Inject fake modules into sys.modules so those lazy imports resolve to
    # stubs without importing the real heavy modules (mongo_raw_data needs libclip_container, etc.).
    import sys
    import types

    monkeypatch.setattr(data_mod, "init_navsim_data", _stub("navsim"))
    for modname, attr, name in [
        ("app.vjepa_cowa_world_model.training.mongo_raw_data", "init_mongo_raw_data", "mongo_raw"),
        ("app.vjepa_cowa_world_model.training.b2d_data", "init_bench2drive_data", "bench2drive"),
        ("app.vjepa_cowa_world_model.training.seg_data", "init_data_only_seg", "seg"),
    ]:
        fake = types.ModuleType(modname)
        setattr(fake, attr, _stub(name))
        monkeypatch.setitem(sys.modules, modname, fake)


def test_navsim_selected(stub_factories):
    cfg = _config(data={"navsim": {"enabled": True, "data_path": "/a", "sensor_blobs_path": "/b"}})
    loader, _ = data_mod.create_train_dataloader(cfg, rank=0, world_size=1)
    assert loader.name == "navsim"


def test_bench2drive_selected(stub_factories):
    cfg = _config(data={"bench2drive": {"enabled": True, "ann_file": "/a.pkl", "data_root": "/r"}})
    loader, _ = data_mod.create_train_dataloader(cfg, rank=0, world_size=1)
    assert loader.name == "bench2drive"


def test_mongo_raw_selected(stub_factories):
    cfg = _config(data={"mongo_raw": {"enabled": True}})
    loader, _ = data_mod.create_train_dataloader(cfg, rank=0, world_size=1)
    assert loader.name == "mongo_raw"


def test_legacy_seg_selected_when_none_enabled(stub_factories):
    cfg = _config(data={"datasets": ["/seg"]})
    loader, _ = data_mod.create_train_dataloader(cfg, rank=0, world_size=1)
    assert loader.name == "seg"
