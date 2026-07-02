"""Classical local statistical baselines (Group A).

For each test window we fit a small model on the 168-hour lookback and
forecast ``horizon`` steps ahead. This mirrors what a real online
forecasting system would do — refit on the most recent context window
each step. We deliberately do NOT use auto-order search per window: the
order is selected ONCE on a sample of training-set lookbacks, then held
fixed across the val/test grid (so the fit-per-window stays cheap and
the comparison stays fair).

Models implemented
------------------
* ARIMA — fixed (p,d,q) from ``pmdarima.auto_arima`` on the training set;
  per-window fit with statsmodels ``SARIMAX``.
* Holt-Winters Exponential Smoothing — additive trend + seasonal (period
  selectable, default 24 = daily at hourly resolution).
* Theta — statsforecast Theta.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np

from .base import WindowedForecaster

logger = logging.getLogger(__name__)

# silence over-eager statsmodels / pmdarima warnings during the per-window loop
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# ARIMA — fixed-order local refit per window
# ---------------------------------------------------------------------------


class ArimaForecaster(WindowedForecaster):
    name = "arima"
    is_stochastic = False

    def __init__(self, horizon, hparams=None, seed=42, device="cpu"):
        super().__init__(horizon, hparams, seed, device)
        # Order can either be explicitly provided in hparams or set to "auto"
        # (in which case we pick it once in fit() using auto_arima).
        self._order: tuple[int, int, int] | None = self.hparams.get("order")
        if self._order is not None:
            self._order = tuple(int(x) for x in self._order)
        self._max_order_search = int(self.hparams.get("max_order_search", 3))
        self._n_train_samples_for_order = int(self.hparams.get("n_train_samples_for_order", 6))

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        if self._order is None:
            self._order = self._pick_order(X)
            logger.info("[arima] picked order=%s on %d training lookbacks",
                        self._order, self._n_train_samples_for_order)

    def _pick_order(self, X: np.ndarray) -> tuple[int, int, int]:
        import pmdarima as pm
        n = X.shape[0]
        # take a deterministic, evenly-spaced sample of training lookbacks
        idxs = np.linspace(0, n - 1, num=min(self._n_train_samples_for_order, n)).astype(int)
        votes: dict[tuple[int, int, int], int] = {}
        for i in idxs:
            try:
                fit = pm.auto_arima(
                    X[i],
                    seasonal=False,
                    max_p=self._max_order_search,
                    max_q=self._max_order_search,
                    max_d=2,
                    suppress_warnings=True,
                    error_action="ignore",
                    stepwise=True,
                )
                votes[fit.order] = votes.get(fit.order, 0) + 1
            except Exception as e:
                logger.debug("[arima] auto_arima failed on sample %d: %s", i, e)
        if not votes:
            return (1, 1, 1)
        # majority vote
        order, _ = max(votes.items(), key=lambda kv: kv[1])
        return order

    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA
        N = X.shape[0]
        out = np.zeros((N, self.horizon), dtype=np.float32)
        p, d, q = self._order
        for i in range(N):
            try:
                model = _ARIMA(X[i], order=(p, d, q),
                               enforce_stationarity=False,
                               enforce_invertibility=False)
                res = model.fit(method_kwargs={"warn_convergence": False})
                fc = res.forecast(steps=self.horizon)
                out[i] = np.asarray(fc, dtype=np.float32)
            except Exception:
                # robust fallback: last-value persistence
                out[i] = X[i, -1]
        # final NaN/inf guard
        bad = ~np.isfinite(out)
        if bad.any():
            # any pathological prediction → fall back to lookback mean of that row
            row_means = np.nanmean(X, axis=1).astype(np.float32)
            out[bad] = np.broadcast_to(row_means[:, None], out.shape)[bad]
        return out


# ---------------------------------------------------------------------------
# Holt-Winters Exponential Smoothing
# ---------------------------------------------------------------------------


class HoltWintersForecaster(WindowedForecaster):
    name = "holt_winters"
    is_stochastic = False

    def __init__(self, horizon, hparams=None, seed=42, device="cpu"):
        super().__init__(horizon, hparams, seed, device)
        self.seasonal_period = int(self.hparams.get("seasonal_period", 24))
        self.trend = self.hparams.get("trend", "add")           # 'add' | 'mul' | None
        self.seasonal = self.hparams.get("seasonal", "add")     # 'add' | 'mul' | None

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        # no global fit needed; per-window fit happens in predict
        return

    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        N = X.shape[0]
        out = np.zeros((N, self.horizon), dtype=np.float32)
        for i in range(N):
            try:
                model = ExponentialSmoothing(
                    X[i],
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_period,
                    initialization_method="estimated",
                )
                res = model.fit(disp=False, optimized=True)
                fc = res.forecast(self.horizon)
                out[i] = np.asarray(fc, dtype=np.float32)
            except Exception:
                out[i] = X[i, -1]
        bad = ~np.isfinite(out)
        if bad.any():
            row_means = np.nanmean(X, axis=1).astype(np.float32)
            out[bad] = np.broadcast_to(row_means[:, None], out.shape)[bad]
        return out


# ---------------------------------------------------------------------------
# Theta method
# ---------------------------------------------------------------------------


class ThetaForecaster(WindowedForecaster):
    name = "theta"
    is_stochastic = False

    def __init__(self, horizon, hparams=None, seed=42, device="cpu"):
        super().__init__(horizon, hparams, seed, device)
        self.seasonal_period = int(self.hparams.get("seasonal_period", 24))

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        return

    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        # Use statsforecast's vectorised Theta on a small history per window
        from statsforecast.models import Theta as SF_Theta
        N = X.shape[0]
        out = np.zeros((N, self.horizon), dtype=np.float32)
        for i in range(N):
            try:
                m = SF_Theta(season_length=self.seasonal_period)
                m.fit(X[i].astype(np.float64))
                fc = m.predict(h=self.horizon)["mean"]
                out[i] = np.asarray(fc, dtype=np.float32)
            except Exception:
                out[i] = X[i, -1]
        bad = ~np.isfinite(out)
        if bad.any():
            row_means = np.nanmean(X, axis=1).astype(np.float32)
            out[bad] = np.broadcast_to(row_means[:, None], out.shape)[bad]
        return out
