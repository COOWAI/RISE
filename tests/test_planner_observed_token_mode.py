import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_package(module_name: str, package_path: Path):
    module = sys.modules.get(module_name)
    if module is None:
        module = types.ModuleType(module_name)
        module.__path__ = [str(package_path)]
        sys.modules[module_name] = module
    return module


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_training_config_module():
    _ensure_package("app", REPO_ROOT / "app")
    _ensure_package("app.vjepa_cowa_world_model", REPO_ROOT / "app" / "vjepa_cowa_world_model")
    _ensure_package("app.vjepa_cowa_world_model.training", REPO_ROOT / "app" / "vjepa_cowa_world_model" / "training")
    return _load_module_from_path(
        "app.vjepa_cowa_world_model.training.config",
        REPO_ROOT / "app" / "vjepa_cowa_world_model" / "training" / "config.py",
    )


def _load_multimodal_planner_module():
    return _load_module_from_path(
        "app.vjepa_cowa_world_model.models.multimodal_planner",
        REPO_ROOT / "app" / "vjepa_cowa_world_model" / "models" / "multimodal_planner.py",
    )


def _load_diffusion_planner_module():
    _ensure_package("app", REPO_ROOT / "app")
    _ensure_package("app.vjepa_cowa_world_model", REPO_ROOT / "app" / "vjepa_cowa_world_model")
    _ensure_package("app.vjepa_cowa_world_model.models", REPO_ROOT / "app" / "vjepa_cowa_world_model" / "models")
    _ensure_package(
        "app.vjepa_cowa_world_model.diffusion_utils",
        REPO_ROOT / "app" / "vjepa_cowa_world_model" / "diffusion_utils",
    )
    return _load_module_from_path(
        "app.vjepa_cowa_world_model.models.diffusion_planner",
        REPO_ROOT / "app" / "vjepa_cowa_world_model" / "models" / "diffusion_planner.py",
    )


# These module-level loads exec fresh copies of training/models modules into sys.modules at
# collection time. Keep the objects via the variables below, then restore sys.modules so later tests
# do not observe the custom-loaded submodules. Per-test restoration cannot undo a collection-time leak.
_PRE_MODULE_LEVEL_LOAD = dict(sys.modules)

config_module = _load_training_config_module()
multimodal_planner_module = _load_multimodal_planner_module()
diffusion_planner_module = _load_diffusion_planner_module()

for _leaked in list(sys.modules):
    if _leaked not in _PRE_MODULE_LEVEL_LOAD:
        del sys.modules[_leaked]
sys.modules.update(_PRE_MODULE_LEVEL_LOAD)


class TestPlannerObservedTokenModeConfig(unittest.TestCase):
    def test_parse_observed_token_mode_enum(self):
        config = config_module.parse_training_config(
            {
                "planner": {
                    "observed_token_mode": "concat_type_embed",
                }
            }
        )

        self.assertEqual(config.planner.observed_token_mode, "concat_type_embed")
        self.assertTrue(config.planner.use_observed_tokens)

    def test_legacy_use_observed_tokens_maps_to_concat(self):
        config = config_module.parse_training_config(
            {
                "planner": {
                    "use_observed_tokens": True,
                }
            }
        )

        self.assertEqual(config.planner.observed_token_mode, "concat")
        self.assertTrue(config.planner.use_observed_tokens)

    def test_invalid_observed_token_mode_raises(self):
        with self.assertRaises(ValueError):
            config_module.parse_training_config(
                {
                    "planner": {
                        "observed_token_mode": "source_magic",
                    }
                }
            )

    def test_runtime_resolver_keeps_legacy_namespace_compatible(self):
        config = types.SimpleNamespace(
            planner=types.SimpleNamespace(
                use_observed_tokens=True,
            )
        )

        self.assertEqual(config_module.resolve_planner_observed_token_mode(config), "concat")
        self.assertTrue(config_module.resolve_planner_use_observed_tokens(config))


class TestPlannerObservedTokenModeModels(unittest.TestCase):
    def test_transformer_planner_adds_source_embeddings(self):
        planner = multimodal_planner_module.MultiModalTemporalPlanner(
            encoder_dim=8,
            tf_d_model=4,
            tf_d_ffn=8,
            tf_num_layers=1,
            tf_num_head=2,
            tokens_per_frame=2,
            num_poses=2,
            num_time_steps=2,
            num_observed_frames=1,
            status_dim=3,
            use_temporal=True,
            use_time_aligned_bias=False,
            use_status_for_planner=False,
            use_observed_tokens=True,
            observed_token_mode="concat_type_embed",
        )
        self.assertTrue(hasattr(planner, "observed_source_embedding"))

        with torch.no_grad():
            planner.observed_source_embedding.weight[0].fill_(1.0)
            planner.observed_source_embedding.weight[1].fill_(2.0)

        z_ar = torch.zeros(1, 4, 8)
        z_observed = torch.zeros(1, 2, 8)
        prepared, num_steps = planner._prepare_planner_input(z_ar, z_context=None, z_observed=z_observed)

        self.assertEqual(num_steps, 3)
        self.assertTrue(torch.equal(prepared[:, :2], torch.ones_like(prepared[:, :2])))
        self.assertTrue(torch.equal(prepared[:, 2:], torch.full_like(prepared[:, 2:], 2.0)))

    def test_transformer_none_mode_ignores_passed_observed_tokens(self):
        planner = multimodal_planner_module.MultiModalTemporalPlanner(
            encoder_dim=8,
            tf_d_model=4,
            tf_d_ffn=8,
            tf_num_layers=1,
            tf_num_head=2,
            tokens_per_frame=2,
            num_poses=2,
            num_time_steps=2,
            num_observed_frames=1,
            status_dim=3,
            use_temporal=True,
            use_time_aligned_bias=False,
            use_status_for_planner=False,
            observed_token_mode="none",
        )

        z_ar = torch.zeros(1, 4, 8)
        z_observed = torch.ones(1, 2, 8)
        prepared, num_steps = planner._prepare_planner_input(z_ar, z_context=None, z_observed=z_observed)

        self.assertEqual(num_steps, 2)
        self.assertEqual(prepared.shape, z_ar.shape)
        self.assertTrue(torch.equal(prepared, z_ar))

    def test_diffusion_planner_adds_source_embeddings(self):
        planner = diffusion_planner_module.DiffusionPlanner(
            encoder_dim=8,
            num_poses=2,
            status_dim=3,
            hidden_dim=8,
            depth=1,
            heads=2,
            mlp_ratio=2.0,
            tokens_per_frame=2,
            use_last_frame_only=False,
            observed_token_mode="concat_type_embed",
        )
        self.assertTrue(hasattr(planner, "observed_source_embedding"))

        with torch.no_grad():
            planner.context_proj = nn.Identity()
            planner.observed_source_embedding.weight[0].fill_(1.0)
            planner.observed_source_embedding.weight[1].fill_(2.0)

        z_ar = torch.zeros(1, 4, 8)
        z_observed = torch.zeros(1, 2, 8)
        context = planner._prepare_context(z_ar, z_observed=z_observed)

        self.assertEqual(context.shape, (1, 6, 8))
        self.assertTrue(torch.equal(context[:, :2], torch.ones_like(context[:, :2])))
        self.assertTrue(torch.equal(context[:, 2:], torch.full_like(context[:, 2:], 2.0)))


if __name__ == "__main__":
    unittest.main()
