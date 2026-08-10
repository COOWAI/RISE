"""Public mathematical and validation contract for the linear VP-SDE."""

from __future__ import annotations

import math

import pytest
import torch

from app.vjepa_cowa_world_model.diffusion_utils.sde import VPSDE_linear

BETA_MIN = 0.1
BETA_MAX = 20.0
RANK_SHAPES = {
    1: (2,),
    2: (2, 5),
    4: (2, 3, 4, 5),
    5: (2, 2, 3, 4, 5),
}
PUBLIC_TIME_ENTRY_NAMES = ("sde", "marginal_prob", "diffusion_coeff", "marginal_prob_std")


def _batch_view(values: torch.Tensor, ndim: int) -> torch.Tensor:
    if values.ndim == 0:
        return values
    return values.reshape((values.shape[0],) + (1,) * (ndim - 1))


def _closed_forms(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    beta_t = BETA_MIN + t * (BETA_MAX - BETA_MIN)
    log_alpha = -0.25 * t.square() * (BETA_MAX - BETA_MIN) - 0.5 * t * BETA_MIN
    std = torch.sqrt(1.0 - torch.exp(2.0 * log_alpha))
    return beta_t, log_alpha, std


def _call_time_entry(sde: VPSDE_linear, method_name: str, t: torch.Tensor) -> object:
    if method_name in {"sde", "marginal_prob"}:
        return getattr(sde, method_name)(torch.ones(2, 3, dtype=t.dtype), t)
    return getattr(sde, method_name)(t)


def _stable_reference_std(t: float) -> float:
    log_alpha = -0.25 * t * t * (BETA_MAX - BETA_MIN) - 0.5 * t * BETA_MIN
    return math.sqrt(-math.expm1(2.0 * log_alpha))


def test_vpsde_linear_matches_public_closed_forms_for_reference_tensor() -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
    t = torch.tensor([0.25, 0.75], dtype=torch.float64)
    beta_t, log_alpha, expected_std = _closed_forms(t)
    beta_view = _batch_view(beta_t, x.ndim)
    alpha_view = _batch_view(torch.exp(log_alpha), x.ndim)
    std_view = _batch_view(expected_std, x.ndim)

    drift, diffusion = sde.sde(x, t)
    mean, std = sde.marginal_prob(x, t)

    assert sde.T == 1.0
    torch.testing.assert_close(drift, -0.5 * beta_view * x)
    torch.testing.assert_close(diffusion, torch.sqrt(beta_view))
    torch.testing.assert_close(mean, x * alpha_view)
    torch.testing.assert_close(std, std_view)
    torch.testing.assert_close(sde.diffusion_coeff(t), torch.sqrt(beta_t))
    torch.testing.assert_close(sde.marginal_prob_std(t), expected_std)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rank", tuple(RANK_SHAPES))
@pytest.mark.parametrize("scalar_time", (False, True))
def test_vpsde_linear_broadcasts_batch_first_for_arbitrary_rank(
    dtype: torch.dtype,
    rank: int,
    scalar_time: bool,
) -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)
    shape = RANK_SHAPES[rank]
    x = torch.arange(math.prod(shape), dtype=dtype).reshape(shape) + 0.25
    t = torch.tensor(0.4, dtype=dtype) if scalar_time else torch.tensor([0.2, 0.8], dtype=dtype)
    beta_t, log_alpha, expected_std = _closed_forms(t)

    drift, diffusion = sde.sde(x, t)
    mean, std = sde.marginal_prob(x, t)
    beta_view = _batch_view(beta_t, x.ndim)
    alpha_view = _batch_view(torch.exp(log_alpha), x.ndim)
    std_view = _batch_view(expected_std, x.ndim)

    assert drift.shape == x.shape
    assert mean.shape == x.shape
    assert drift.dtype == mean.dtype == x.dtype
    assert diffusion.dtype == std.dtype == x.dtype
    torch.testing.assert_close(drift, -0.5 * beta_view * x)
    torch.testing.assert_close(diffusion, torch.sqrt(beta_view))
    torch.testing.assert_close(mean, alpha_view * x)
    torch.testing.assert_close(std, std_view)
    torch.testing.assert_close(sde.diffusion_coeff(t), torch.sqrt(beta_t))
    torch.testing.assert_close(sde.marginal_prob_std(t), expected_std)


@pytest.mark.parametrize(
    ("x", "t", "error_type", "message"),
    (
        (torch.ones(2, 3, dtype=torch.int64), torch.ones(2), TypeError, "floating"),
        (torch.ones(2, 3), torch.ones(2, dtype=torch.int64), TypeError, "floating"),
        (torch.ones(2, 3, dtype=torch.float32), torch.ones(2, dtype=torch.float64), TypeError, "dtype"),
        (torch.ones(2, 3), torch.ones(2, 1), ValueError, "shape|scalar|batch"),
        (torch.ones(2, 3), torch.ones(3), ValueError, "batch"),
        (torch.tensor(1.0), torch.ones(2), ValueError, "batch"),
    ),
)
def test_vpsde_linear_rejects_invalid_public_tensor_inputs(
    x: torch.Tensor,
    t: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)

    with pytest.raises(error_type, match=message):
        sde.sde(x, t)
    with pytest.raises(error_type, match=message):
        sde.marginal_prob(x, t)


def test_vpsde_linear_rejects_scalar_x_even_with_scalar_time() -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)
    x = torch.tensor(1.0, dtype=torch.float32)
    t = torch.tensor(0.5, dtype=torch.float32)

    with pytest.raises(ValueError, match="batch"):
        sde.sde(x, t)
    with pytest.raises(ValueError, match="batch"):
        sde.marginal_prob(x, t)


def test_vpsde_linear_rejects_device_mismatch_before_computation() -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)
    x = torch.ones(2, 3, dtype=torch.float32, device="cpu")
    t = torch.empty(2, dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="device"):
        sde.sde(x, t)
    with pytest.raises(ValueError, match="device"):
        sde.marginal_prob(x, t)


@pytest.mark.parametrize("method_name", ("diffusion_coeff", "marginal_prob_std"))
@pytest.mark.parametrize(
    ("t", "error_type", "message"),
    (
        (torch.ones(2, dtype=torch.int64), TypeError, "floating"),
        (torch.ones(2, 1, dtype=torch.float32), ValueError, "shape|scalar|batch"),
    ),
)
def test_vpsde_linear_time_only_helpers_reject_invalid_time_tensors(
    method_name: str,
    t: torch.Tensor,
    error_type: type[Exception],
    message: str,
) -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)

    with pytest.raises(error_type, match=message):
        getattr(sde, method_name)(t)


@pytest.mark.parametrize("method_name", PUBLIC_TIME_ENTRY_NAMES)
@pytest.mark.parametrize("invalid_time", (-1.0e-6, 1.000001, float("-inf"), float("inf"), float("nan")))
def test_every_public_time_entry_rejects_values_outside_closed_sde_interval(
    method_name: str,
    invalid_time: float,
) -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)
    t = torch.tensor([0.25, invalid_time], dtype=torch.float32)

    with pytest.raises(ValueError, match="time|finite|range|0|T"):
        _call_time_entry(sde, method_name, t)


@pytest.mark.parametrize("method_name", PUBLIC_TIME_ENTRY_NAMES)
@pytest.mark.parametrize("endpoint_name", ("zero", "T"))
def test_every_public_time_entry_accepts_closed_interval_endpoints(
    method_name: str,
    endpoint_name: str,
) -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)
    endpoint = 0.0 if endpoint_name == "zero" else sde.T
    t = torch.full((2,), endpoint, dtype=torch.float32)

    result = _call_time_entry(sde, method_name, t)
    values = result if isinstance(result, tuple) else (result,)
    assert all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in values)
    if method_name == "sde":
        assert values[0].shape == (2, 3)
        assert values[1].shape == _batch_view(t, 2).shape
    elif method_name == "marginal_prob":
        assert values[0].shape == (2, 3)
        assert values[1].shape == (2, 1)
    else:
        assert values[0].shape == t.shape

    if endpoint_name == "zero" and method_name in {"marginal_prob", "marginal_prob_std"}:
        std = values[1] if method_name == "marginal_prob" else values[0]
        torch.testing.assert_close(std, torch.zeros_like(std), rtol=0.0, atol=0.0)


@pytest.mark.parametrize("method_name", ("marginal_prob", "marginal_prob_std"))
@pytest.mark.parametrize(
    ("dtype", "small_time", "rtol"),
    (
        (torch.float32, 1.0e-8, 5.0e-4),
        (torch.float16, 1.0e-3, 3.0e-2),
        (torch.bfloat16, 1.0e-3, 6.0e-2),
    ),
)
def test_small_positive_time_std_is_finite_nonzero_and_numerically_stable_on_cpu(
    method_name: str,
    dtype: torch.dtype,
    small_time: float,
    rtol: float,
) -> None:
    sde = VPSDE_linear(beta_min=BETA_MIN, beta_max=BETA_MAX)
    t = torch.tensor(small_time, dtype=dtype, device="cpu")

    result = _call_time_entry(sde, method_name, t)
    std = result[1] if method_name == "marginal_prob" else result
    assert isinstance(std, torch.Tensor)
    assert std.dtype == dtype
    assert std.device == t.device
    assert torch.isfinite(std).all()
    assert torch.all(std > 0)

    actual = std.to(torch.float64)
    expected = torch.full_like(actual, _stable_reference_std(small_time))
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=0.0)


@pytest.mark.parametrize(
    ("beta_min", "beta_max", "message"),
    (
        (0.0, 20.0, "beta_min"),
        (-0.1, 20.0, "beta_min"),
        (0.1, 0.1, "beta_max"),
        (0.2, 0.1, "beta_max"),
        (float("nan"), 20.0, "finite"),
        (float("inf"), 20.0, "finite"),
        (0.1, float("nan"), "finite"),
        (0.1, float("inf"), "finite"),
    ),
)
def test_vpsde_linear_rejects_invalid_beta_ranges(beta_min: float, beta_max: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VPSDE_linear(beta_min=beta_min, beta_max=beta_max)
