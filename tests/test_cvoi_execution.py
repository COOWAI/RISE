from types import SimpleNamespace

import pytest
import torch

from app.vjepa_cowa_world_model.models.prefix_dual_value import PrefixDualValueModel
from app.vjepa_cowa_world_model.training.cvoi_execution import (
    CvoiValueDtypeAdapter,
    cvoi_execution_autocast,
    cvoi_execution_dtype_signature,
    cvoi_inference_rng_signature,
    cvoi_planner_inference_noise,
    cvoi_sample_seed,
    resolve_cvoi_evaluation_seed,
)


def _config(dtype: str = "bfloat16") -> SimpleNamespace:
    return SimpleNamespace(meta=SimpleNamespace(dtype=dtype))


def test_execution_dtype_signature_is_explicit_about_cuda_and_value_precision() -> None:
    signature = cvoi_execution_dtype_signature(_config())

    assert signature == {
        "schema": "cvoi_execution_dtype_v1",
        "required_device": "cuda",
        "input_storage_dtype": "float32",
        "cuda_autocast_enabled": True,
        "cuda_autocast_dtype": "bfloat16",
        "non_cuda_autocast_enabled": False,
        "value_gate_dtype": "float32",
        "backend_policy": {
            "deterministic_algorithms": False,
            "deterministic_warn_only": False,
            "cudnn_deterministic": False,
            "cudnn_benchmark": True,
        },
    }


def test_value_dtype_adapter_keeps_value_and_guidance_in_fp32_with_bf16_latents() -> None:
    model = PrefixDualValueModel(embed_dim=4, hidden_dim=6)
    adapter = CvoiValueDtypeAdapter(model)
    observed = torch.randn(1, 8, 4, dtype=torch.bfloat16)
    future = torch.randn(1, 4, 4, dtype=torch.bfloat16, requires_grad=True)

    output = adapter(observed, future, tokens_per_frame=2)
    output.field_values.sum().backward()

    assert output.field_values.dtype == torch.float32
    assert output.stop_values.dtype == torch.float32
    assert future.grad is not None
    assert torch.isfinite(future.grad).all()


def test_sample_rng_contract_binds_seed_and_stable_sample_identity() -> None:
    config = _config()
    config.meta.seed = 239

    signature = cvoi_inference_rng_signature(config)
    first = cvoi_sample_seed(239, "navsim:real:scene-0001:0")
    second = cvoi_sample_seed(239, "navsim:real:scene-0001:0")

    assert signature["base_seed"] == 239
    assert signature["common_across_horizons_and_lambda"] is True
    assert first == second
    assert first != cvoi_sample_seed(240, "navsim:real:scene-0001:0")


def test_formal_ablation_uses_evaluation_seed_without_changing_training_seed() -> None:
    config = SimpleNamespace(
        meta=SimpleNamespace(seed=3407),
        cvoi=SimpleNamespace(
            ablation_signature=SimpleNamespace(train_seed=3407, evaluation_seed=239),
        ),
    )

    assert resolve_cvoi_evaluation_seed(config) == 239
    assert config.meta.seed == 3407
    assert cvoi_inference_rng_signature(config)["base_seed"] == 239


def test_planner_noise_is_explicit_and_sample_seeded() -> None:
    planner = SimpleNamespace(num_modes=6, num_samples=6, num_poses=8, traj_dim=4)
    seeds = [cvoi_sample_seed(239, "sample-a"), cvoi_sample_seed(239, "sample-b")]

    first = cvoi_planner_inference_noise(planner, seeds=seeds, device=torch.device("cpu"))
    second = cvoi_planner_inference_noise(planner, seeds=seeds, device=torch.device("cpu"))

    assert first.shape == (2, 6, 8, 4)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert not torch.equal(first[0], first[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution-dtype equivalence")
def test_cuda_autocast_contract_is_identical_for_offline_and_online_calls() -> None:
    config = _config()
    model = torch.nn.Linear(8, 4).cuda().eval()
    inputs = torch.randn(2, 8, device="cuda")

    with torch.no_grad(), cvoi_execution_autocast(config, inputs.device):
        offline = model(inputs)
    with torch.no_grad(), cvoi_execution_autocast(config, inputs.device):
        online = model(inputs)

    assert offline.dtype == torch.bfloat16
    assert torch.equal(offline, online)
