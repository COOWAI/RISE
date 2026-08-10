# Copyright (c) 2026 RISE Contributors
# RISE provenance: independent-diffusion-v1
"""Variance-preserving stochastic differential equations for trajectory diffusion."""

import math
from abc import ABC, abstractmethod

import torch


class SDE(ABC):
    """Abstract public surface for a forward stochastic differential equation."""

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def T(self):
        """Terminal diffusion time."""

    @abstractmethod
    def sde(self, x, t):
        """Return drift and diffusion coefficient at ``(x, t)``."""

    @abstractmethod
    def marginal_prob(self, x, t):
        """Return the conditional marginal mean and standard deviation."""

    @abstractmethod
    def marginal_prob_std(self, t):
        """Return the conditional marginal standard deviation."""

    @abstractmethod
    def diffusion_coeff(self, t):
        """Return the scalar diffusion coefficient at time ``t``."""


class VPSDE_linear(SDE):
    """Continuous linear-beta variance-preserving SDE."""

    def __init__(self, beta_max=20.0, beta_min=0.1):
        super().__init__()
        try:
            beta_min_value = float(beta_min)
            beta_max_value = float(beta_max)
        except (TypeError, ValueError) as exc:
            raise ValueError("beta_min and beta_max must be finite real numbers") from exc
        if not math.isfinite(beta_min_value) or not math.isfinite(beta_max_value):
            raise ValueError("beta_min and beta_max must be finite")
        if beta_min_value <= 0.0:
            raise ValueError(f"beta_min must be positive, got {beta_min!r}")
        if beta_max_value <= beta_min_value:
            raise ValueError(
                f"beta_max must be greater than beta_min; got beta_max={beta_max!r}, beta_min={beta_min!r}"
            )
        self._beta_min = beta_min_value
        self._beta_max = beta_max_value

    @property
    def T(self):
        return 1.0

    @staticmethod
    def _validate_time(t):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"t must be a torch.Tensor, got {type(t).__name__}")
        if not t.is_floating_point():
            raise TypeError(f"t must have a floating dtype, got {t.dtype}")
        if t.ndim not in (0, 1):
            raise ValueError(f"t must be scalar or a one-dimensional batch tensor, got shape {tuple(t.shape)}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError("t must contain only finite values")
        if not bool(((t >= 0.0) & (t <= 1.0)).all()):
            raise ValueError("time values must be within the closed interval [0, T]")

    @classmethod
    def _validate_pair(cls, x, t):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if not x.is_floating_point():
            raise TypeError(f"x must have a floating dtype, got {x.dtype}")
        if x.ndim < 1:
            raise ValueError("x must have a batch dimension")
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"t must be a torch.Tensor, got {type(t).__name__}")
        if x.device != t.device:
            raise ValueError(f"x and t must be on the same device, got {x.device} and {t.device}")
        cls._validate_time(t)
        if x.dtype != t.dtype:
            raise TypeError(f"x and t must have the same dtype, got {x.dtype} and {t.dtype}")
        if t.ndim == 1 and t.shape[0] != x.shape[0]:
            raise ValueError(f"t batch size must match x batch size, got {t.shape[0]} and {x.shape[0]}")

    @staticmethod
    def _batch_view(values, x_ndim):
        if values.ndim == 0:
            return values
        return values.reshape((values.shape[0],) + (1,) * (x_ndim - 1))

    def _beta(self, t):
        return self._beta_min + t * (self._beta_max - self._beta_min)

    def _log_alpha(self, t):
        return -0.25 * (self._beta_max - self._beta_min) * t.square() - 0.5 * self._beta_min * t

    def sde(self, x, t):
        self._validate_pair(x, t)
        beta_t = self._batch_view(self._beta(t), x.ndim)
        return -0.5 * beta_t * x, torch.sqrt(beta_t)

    def marginal_prob(self, x, t):
        self._validate_pair(x, t)
        log_alpha = self._log_alpha(t)
        alpha = self._batch_view(torch.exp(log_alpha), x.ndim)
        std = self._batch_view(torch.sqrt(-torch.expm1(2.0 * log_alpha)), x.ndim)
        return alpha * x, std

    def marginal_prob_std(self, t):
        self._validate_time(t)
        log_alpha = self._log_alpha(t)
        return torch.sqrt(-torch.expm1(2.0 * log_alpha))

    def diffusion_coeff(self, t):
        self._validate_time(t)
        return torch.sqrt(self._beta(t))
