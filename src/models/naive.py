"""Trivial naive baselines (Group A — no fitting required).

* ``Naive``: predict the last observed value, repeated for the horizon.
* ``SeasonalNaive``: predict the value from K steps ago (K = configurable
  seasonal period — default 24 for daily seasonality at hourly resolution).

These are intentionally parameter-free; they exist to give a floor that
every more sophisticated model should comfortably beat.
"""
from __future__ import annotations

import numpy as np

from .base import WindowedForecaster


class NaiveForecaster(WindowedForecaster):
    name = "naive"
    is_stochastic = False
    supports_multi_horizon = True

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        # nothing to fit
        return

    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        # repeat last observation along horizon
        last = X[:, -1:]                         # (N, 1)
        return np.repeat(last, self.horizon, axis=1)


class SeasonalNaiveForecaster(WindowedForecaster):
    name = "seasonal_naive"
    is_stochastic = False
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cpu"):
        super().__init__(horizon, hparams, seed, device)
        # default seasonal period: 24 (daily) at 1-h resolution
        self.K = int(self.hparams.get("seasonal_period", 24))
        if self.K < 1:
            raise ValueError("seasonal_period must be >= 1")

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        return

    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        """For each window, the t-th forecast step copies the value from
        ``K - (t % K)`` positions back in the lookback (so t=0 uses the
        K-th most recent value, t=1 the (K-1)-th, etc.). This is exactly
        the standard *seasonal naive* forecast.
        """
        N, L = X.shape
        K = self.K
        if L < K:
            raise ValueError(f"lookback {L} smaller than seasonal period {K}")
        out = np.empty((N, self.horizon), dtype=X.dtype)
        for t in range(self.horizon):
            # offset back from the end: -K + (t mod K)
            back_idx = -K + (t % K)
            out[:, t] = X[:, back_idx]
        return out
