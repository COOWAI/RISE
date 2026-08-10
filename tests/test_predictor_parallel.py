import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTOR_PARALLEL_PATH = REPO_ROOT / "app" / "vjepa_cowa_world_model" / "training" / "predictor_parallel.py"


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
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_predictor_parallel_module():
    _ensure_package("app", REPO_ROOT / "app")
    _ensure_package("app.vjepa_cowa_world_model", REPO_ROOT / "app" / "vjepa_cowa_world_model")
    _ensure_package("app.vjepa_cowa_world_model.training", REPO_ROOT / "app" / "vjepa_cowa_world_model" / "training")

    models_module = types.ModuleType("app.vjepa_cowa_world_model.training.models")

    def _build_predictor_input_with_future_queries(predictor, observed_tokens):
        core = predictor.module if hasattr(predictor, "module") else predictor
        future_query_tokens = getattr(core, "future_query_tokens", None)
        if future_query_tokens is None:
            return observed_tokens
        return torch.cat([observed_tokens, future_query_tokens.expand(observed_tokens.size(0), -1, -1)], dim=1)

    def _register_predictor_future_query_tokens(predictor, embed_dim, future_tubelets, tokens_per_frame, device):
        core = predictor.module if hasattr(predictor, "module") else predictor
        core.future_query_tokens = torch.nn.Parameter(
            torch.zeros(1, future_tubelets * tokens_per_frame, embed_dim, device=device)
        )

    models_module.build_predictor_input_with_future_queries = _build_predictor_input_with_future_queries
    models_module.register_predictor_future_query_tokens = _register_predictor_future_query_tokens
    sys.modules[models_module.__name__] = models_module

    utils_module = types.ModuleType("app.vjepa_cowa_world_model.utils")
    utils_module.mask_future_actions = lambda actions, num_known: (
        actions,
        torch.arange(actions.shape[1], device=actions.device).unsqueeze(0).expand(actions.shape[0], -1) >= num_known,
    )
    utils_module.prepare_inference_consistent_states = lambda states, **kwargs: states
    sys.modules[utils_module.__name__] = utils_module

    return _load_module_from_path(
        "app.vjepa_cowa_world_model.training.predictor_parallel",
        PREDICTOR_PARALLEL_PATH,
    )


class TestPredictorParallelHelpers(unittest.TestCase):
    def test_registers_future_queries_from_config_steps(self):
        module = _load_predictor_parallel_module()
        predictor = torch.nn.Linear(4, 4)
        config = types.SimpleNamespace(train=types.SimpleNamespace(use_parallel_predictor=True))

        module.maybe_register_parallel_predictor_tokens(
            predictor=predictor,
            config=config,
            embed_dim=4,
            future_steps=3,
            tokens_per_frame=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(predictor.future_query_tokens.shape), (1, 6, 4))

    def test_parallel_forward_returns_future_tokens_and_optional_z_ar(self):
        module = _load_predictor_parallel_module()
        calls = []

        class PredictorStub(torch.nn.Module):
            def forward(self, z, actions, states, extrinsics, action_mask=None, state_mask=None):
                calls.append(
                    {
                        "z_shape": tuple(z.shape),
                        "actions_shape": tuple(actions.shape),
                        "states_shape": tuple(states.shape),
                        "mask_shape": None if action_mask is None else tuple(action_mask.shape),
                        "state_mask_shape": None if state_mask is None else tuple(state_mask.shape),
                        "mask_values": None if action_mask is None else action_mask.clone(),
                        "state_mask_values": None if state_mask is None else state_mask.clone(),
                    }
                )
                del extrinsics
                return z + actions[:, :1, :1].mean() * 0.0

        predictor = PredictorStub()
        predictor.future_query_tokens = torch.nn.Parameter(torch.zeros(1, 4, 3))
        config = types.SimpleNamespace(
            train=types.SimpleNamespace(
                use_parallel_predictor=True,
                predictor_inference_consistent=True,
                state_dim=7,
                use_drive_command=False,
                use_states_for_predictor=True,
                predictor_no_aux_input=False,
                predictor_use_z_ar_supervision=True,
            )
        )
        z_context = torch.randn(2, 2, 3)
        actions = torch.randn(2, 3, 2)
        states = torch.randn(2, 3, 7)
        extrinsics = torch.randn(2, 3, 1)

        output = module.forward_parallel_predictor(
            predictor=predictor,
            observed_tokens=z_context,
            actions=actions,
            states=states,
            extrinsics=extrinsics,
            config=config,
            tokens_per_frame=2,
            runtime_normalize_reps=False,
            num_observed_steps=1,
            driving_command=None,
            ego_dynamics=None,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["z_shape"], (2, 6, 3))
        self.assertEqual(calls[0]["actions_shape"], (2, 3, 2))
        self.assertEqual(calls[0]["states_shape"], (2, 3, 3))
        self.assertEqual(calls[0]["mask_shape"], (2, 3))
        self.assertIsNone(calls[0]["state_mask_shape"])
        self.assertEqual(tuple(output.z_future.shape), (2, 4, 3))
        self.assertIs(output.z_ar, output.z_future)


if __name__ == "__main__":
    unittest.main()
