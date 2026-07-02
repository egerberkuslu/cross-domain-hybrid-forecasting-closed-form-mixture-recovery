"""ARFIMA / Fractional ARIMA — long-range-dependent baseline canonical for
network-traffic forecasting (Beran 1994; Karagiannis et al. 2004
IEEE Trans. Networking; Park & Willinger book 2000).

Standard ARIMA restricts the differencing order to integers; ARFIMA allows
d ∈ (−0.5, 0.5) so the auto-covariance decays as k^{2d−1} (long memory iff
d > 0).  This file implements a lightweight ARFIMA(p, d, q) baseline:

  1. Estimate Hurst exponent H once on the training series (R/S analysis),
     then ``d_hat = clip(H − 0.5, −0.49, 0.49)``.
  2. Pick the ARMA(p, q) order using ``pmdarima.auto_arima`` on a few
     fractionally differenced training windows.
  3. Per test window: fractional-difference → ARMA fit/forecast → fractional
     integrate back.

Conforms to the WindowedForecaster API (X has shape (N, lookback), y has
shape (N, horizon); predict returns (N, horizon)).
"""
from __future__ import annotations

import logging
import math
import warnings

import numpy as np

from .base import WindowedForecaster

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------- pure-numpy helpers ----------


def hurst_rs(
    x: np.ndarray, min_window: int = 16, max_window: int | None = None
) -> float:
    """Hurst exponent via classic R/S (rescaled range) analysis."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 64:
        return 0.5
    if max_window is None:
        max_window = n // 2
    ws = np.unique(
        np.logspace(math.log10(min_window), math.log10(max_window), 20).astype(int)
    )
    rs_vals = []
    for w in ws:
        if w < 4 or w >= n:
            continue
        n_chunks = n // w
        chunks = x[: n_chunks * w].reshape(n_chunks, w)
        rs_chunk = []
        for c in chunks:
            mu = c.mean()
            dev = c - mu
            z = np.cumsum(dev)
            R = z.max() - z.min()
            S = c.std(ddof=0)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            rs_vals.append((w, float(np.mean(rs_chunk))))
    if len(rs_vals) < 3:
        return 0.5
    ws_arr = np.array([w for w, _ in rs_vals], dtype=float)
    rs_arr = np.array([r for _, r in rs_vals], dtype=float)
    slope, _ = np.polyfit(np.log(ws_arr), np.log(np.maximum(rs_arr, 1e-12)), 1)
    return float(np.clip(slope, 0.05, 0.95))


def _frac_diff_weights(d: float, K: int) -> np.ndarray:
    w = np.zeros(K + 1, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, K + 1):
        w[k] = w[k - 1] * (-(d - k + 1) / k)
    return w


def frac_diff(x: np.ndarray, d: float, K: int) -> np.ndarray:
    """Truncated fractional differencing (1 − L)^d applied to x."""
    x = np.asarray(x, dtype=np.float64).ravel()
    K = min(K, len(x) - 1)
    if K <= 0:
        return x.copy()
    w = _frac_diff_weights(d, K)
    out = np.zeros_like(x)
    for t in range(len(x)):
        kmax = min(K, t)
        seg = x[t - kmax : t + 1][::-1]
        out[t] = float(np.dot(w[: kmax + 1], seg))
    return out


def frac_int(diffed: np.ndarray, d: float, K: int) -> np.ndarray:
    """Inverse: (1 − L)^{-d} applied to diffed."""
    return frac_diff(diffed, -d, K)


# ---------- forecaster ----------


class FARIMAForecaster(WindowedForecaster):
    """ARFIMA baseline with auto-estimated fractional d."""

    name = "farima"
    is_stochastic = False

    def __init__(self, horizon, hparams=None, seed=42, device="cpu"):
        super().__init__(horizon, hparams, seed, device)
        self._p: int = int(self.hparams.get("p", 1))
        self._q: int = int(self.hparams.get("q", 1))
        self._max_order_search: int = int(self.hparams.get("max_order_search", 2))
        self._trunc_K: int = int(self.hparams.get("trunc_K", 100))
        self._d_hat: float = 0.0
        self._order: tuple[int, int] | None = None

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        # Concatenate a few training windows for Hurst estimation
        n_take = min(4, X.shape[0])
        idxs = np.linspace(0, X.shape[0] - 1, n_take).astype(int)
        cat = np.concatenate([X[i] for i in idxs])
        H = hurst_rs(cat)
        self._d_hat = float(np.clip(H - 0.5, -0.49, 0.49))
        logger.info(
            "[farima] estimated Hurst=%.3f → d_hat=%.3f (long-mem=%s)",
            H,
            self._d_hat,
            self._d_hat > 0,
        )
        # ARMA order picked on a fractionally-differenced sample
        try:
            import pmdarima as pm

            xd = frac_diff(cat, self._d_hat, self._trunc_K)
            fit = pm.auto_arima(
                xd,
                seasonal=False,
                max_p=self._max_order_search,
                max_q=self._max_order_search,
                d=0,
                max_d=0,
                suppress_warnings=True,
                error_action="ignore",
                stepwise=True,
            )
            p, _, q = fit.order
            self._order = (int(p), int(q))
            logger.info("[farima] ARMA order=%s", self._order)
        except Exception as e:
            self._order = (self._p, self._q)
            logger.warning("[farima] auto_arima failed (%s) — fallback to (1,1)", e)

    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA

        N = X.shape[0]
        out = np.zeros((N, self.horizon), dtype=np.float32)
        p, q = self._order or (1, 1)
        d = self._d_hat
        K = min(self._trunc_K, X.shape[1] - 1)
        for i in range(N):
            try:
                xd = frac_diff(X[i].astype(np.float64), d, K)
                m = _ARIMA(
                    xd,
                    order=(p, 0, q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = m.fit(method_kwargs={"warn_convergence": False})
                fc = np.asarray(res.forecast(steps=self.horizon), dtype=np.float64)
                # Inverse-difference the combined (xd + fc) series, take last h
                full = np.concatenate([xd, fc])
                back = frac_int(full, d, min(self._trunc_K, len(full) - 1))
                out[i] = back[-self.horizon :].astype(np.float32)
            except Exception:
                out[i] = X[i, -1]
        bad = ~np.isfinite(out)
        if bad.any():
            row_means = np.nanmean(X, axis=1).astype(np.float32)
            out[bad] = np.broadcast_to(row_means[:, None], out.shape)[bad]
        return out
