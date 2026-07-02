"""CHA-Hybrid — Component-wise, Horizon-adaptive Hybrid (proposed model).

Architecture
------------
Two forecast paths are combined per-horizon by a learned scalar weight
``alpha_h`` (one per forecast horizon, tuned on the validation set):

    final_h = alpha_h * decomp_h + (1 - alpha_h) * global_h

* **decomp_h** — for each test window we STL-decompose the 168-step
  lookback into ``trend + seasonal + residual``, then forecast each
  component independently for ``horizon`` steps ahead and sum them:

      decomp_h(t) = trend_hat(t) + seasonal_hat(t) + residual_hat(t)

  - trend_hat: degree-1 polynomial extrapolation on the lookback's trend
    (cheap and faithful for the smooth STL trend); optionally an ARIMA
    refit per window when ``trend_model='arima'``.
  - seasonal_hat: seasonal-naive (period = ``stl_period``).
  - residual_hat: a darts BlockRNN(GRU) trained once on the full
    training-set STL residual; per-window we feed it the lookback's
    residual as context and read off the GRU's h-step forecast.

* **global_h** — a darts BlockRNN(LSTM) trained once on the raw
  scaled training series, producing the h-step forecast directly from
  the (unmodified) lookback.

The α grid is searched on the validation set (cheap because both
component-forecast and global-forecast vectors are pre-computed; the
search is a single linear-combination loop).

This mirrors the spec in the project brief:

  STL decomposition → trend/seasonal/residual; forecast trend with
  classical, seasonal with seasonal classical, residual with GRU;
  sum = decomposition forecast. Plus a global LSTM. Final = α·decomp +
  (1−α)·global, α_h tuned per horizon on val.
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import STL

from .base import BaseForecaster, FitReport
from .deep_darts import GRUForecaster, LSTMForecaster

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _stl_decompose(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply STL and return (trend, seasonal, residual) arrays."""
    if values.size < 2 * period:
        # too short for STL — fall back to constant trend + zero seasonal
        return values.copy(), np.zeros_like(values), np.zeros_like(values)
    try:
        # statsmodels STL needs robust period; we set robust=False for speed
        res = STL(values.astype(np.float64), period=period, robust=False).fit()
        return (np.asarray(res.trend, dtype=np.float32),
                np.asarray(res.seasonal, dtype=np.float32),
                np.asarray(res.resid, dtype=np.float32))
    except Exception:
        return values.astype(np.float32), np.zeros_like(values, dtype=np.float32), np.zeros_like(values, dtype=np.float32)


def _trend_forecast_linear(trend: np.ndarray, h: int, tail: int = 24) -> np.ndarray:
    """Degree-1 polynomial extrapolation on the last `tail` points of the trend."""
    n = trend.size
    tail = min(tail, n)
    x = np.arange(n - tail, n, dtype=np.float64)
    y = trend[-tail:].astype(np.float64)
    if np.allclose(y, y[0]):
        return np.full(h, float(y[-1]), dtype=np.float32)
    slope, intercept = np.polyfit(x, y, 1)
    x_fut = np.arange(n, n + h, dtype=np.float64)
    return (slope * x_fut + intercept).astype(np.float32)


def _trend_forecast_arima(trend: np.ndarray, h: int, order=(1, 1, 0)) -> np.ndarray:
    """ARIMA forecast on the trend component (used when trend_model='arima')."""
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    try:
        m = _ARIMA(trend, order=order,
                   enforce_stationarity=False, enforce_invertibility=False)
        return np.asarray(m.fit(method_kwargs={"warn_convergence": False})
                          .forecast(steps=h), dtype=np.float32)
    except Exception:
        return _trend_forecast_linear(trend, h)


def _seasonal_forecast(seasonal: np.ndarray, h: int, period: int) -> np.ndarray:
    """Seasonal-naive forecast: copy seasonal[t - period + (t mod period)]."""
    L = seasonal.size
    if L < period:
        return np.full(h, 0.0, dtype=np.float32)
    out = np.empty(h, dtype=np.float32)
    for t in range(h):
        out[t] = seasonal[-period + (t % period)]
    return out


# ---------------------------------------------------------------------------
# main class
# ---------------------------------------------------------------------------


class CHAHybridForecaster(BaseForecaster):
    name = "cha_hybrid"
    is_stochastic = True
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.stl_period = int(hp.pop("stl_period", 24))
        self.trend_model_name = str(hp.pop("trend_model", "linear"))
        self.trend_arima_order = tuple(hp.pop("trend_arima_order", (1, 1, 0)))
        self.seasonal_model_name = str(hp.pop("seasonal_model", "seasonal_naive"))
        self.residual_model_name = str(hp.pop("residual_model", "gru"))
        self.global_model_name = str(hp.pop("global_model", "lstm"))
        self.alpha_search = list(hp.pop("alpha_search",
                                        [round(0.1 * i, 2) for i in range(11)]))
        # alpha can either be a single value (used for all horizons) or a
        # per-horizon dict {h: alpha}; we store both forms.
        self.alpha_h: float | None = hp.pop("alpha", None)

        # subordinate model hparams (per the CHA-Hybrid config block)
        self.gru_hparams = dict(hp.pop("gru", {}))
        self.lstm_hparams = dict(hp.pop("lstm", {}))
        # absorb anything else as either training-loop control or ignore
        self._extra = hp

        self.input_chunk_length = int(self.gru_hparams.pop("input_chunk_length", 168))
        self.lstm_hparams.setdefault("input_chunk_length", self.input_chunk_length)
        self.gru_hparams["input_chunk_length"] = self.input_chunk_length

        # populated in fit()
        self.gru_model: GRUForecaster | None = None
        self.lstm_model: LSTMForecaster | None = None
        self._train_recon_diag: dict | None = None    # STL reconstruction diagnostic
        self._val_alpha_search_diag: list[dict] | None = None
        self._chosen_alpha: float | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train, val=None, *, train_series=None, val_series=None) -> "CHAHybridForecaster":
        if train_series is None:
            raise ValueError("cha_hybrid requires train_series (scaled 1-D array)")
        t0 = time.perf_counter()

        # --- 1) STL on the training series + reconstruction diagnostic ---
        # We compute STL on the longest contiguous non-NaN chunk of train_series
        # (Abilene has long NaN gaps in train) so the GRU sees a clean signal.
        train_arr = np.asarray(train_series, dtype=np.float32)
        chunks = _largest_finite_chunk(train_arr, min_len=2 * self.stl_period + 8)
        if chunks is None:
            raise RuntimeError("cha_hybrid: no contiguous train chunk long enough for STL")
        chunk_start, chunk_end = chunks
        clean_train = train_arr[chunk_start:chunk_end]
        trend, seasonal, residual = _stl_decompose(clean_train, period=self.stl_period)
        # reconstruction check (this is part of the Phase-4 verification)
        recon = trend + seasonal + residual
        recon_err = float(np.max(np.abs(recon - clean_train)))
        recon_rel = float(recon_err / (np.std(clean_train) + 1e-12))
        self._train_recon_diag = {
            "max_abs_err": recon_err,
            "rel_err_vs_std": recon_rel,
            "n": int(clean_train.size),
        }
        logger.info(
            "[cha_hybrid] STL reconstruction: max|y - (trend+seas+resid)|=%.3e "
            "(rel-to-std=%.3e, n=%d) — within tol=1e-4: %s",
            recon_err, recon_rel, clean_train.size, recon_err < 1e-4,
        )

        # --- 2) Train residual GRU on the training residual ---
        # Build a "residual series" with NaNs preserved in the non-contiguous
        # regions so darts treats them as chunks (just like the raw series).
        residual_full = np.full_like(train_arr, np.nan, dtype=np.float32)
        residual_full[chunk_start:chunk_end] = residual

        # val residuals (best-effort — they're only used for early stopping)
        if val_series is not None:
            val_arr = np.asarray(val_series, dtype=np.float32)
            val_chunk = _largest_finite_chunk(val_arr, min_len=2 * self.stl_period + 8)
            if val_chunk is not None:
                _, vs, vr = _stl_decompose(val_arr[val_chunk[0]:val_chunk[1]],
                                          period=self.stl_period)
                val_residual_full = np.full_like(val_arr, np.nan, dtype=np.float32)
                val_residual_full[val_chunk[0]:val_chunk[1]] = vr
            else:
                val_residual_full = None
        else:
            val_residual_full = None

        logger.info("[cha_hybrid] training residual-GRU on STL residual …")
        # the residual WindowSet is built lazily from the residual series; we
        # only need the GRU's fit(train_series=residual_full) path, so pass a
        # dummy WindowSet derived from the shape of `train`.
        self.gru_model = GRUForecaster(
            horizon=self.horizon, hparams=dict(self.gru_hparams),
            seed=self.seed, device=self.device,
        )
        self.gru_model.fit(train, val,
                           train_series=residual_full,
                           val_series=val_residual_full)

        # --- 3) Train global LSTM on raw scaled training series ---
        logger.info("[cha_hybrid] training global-LSTM on raw scaled series …")
        self.lstm_model = LSTMForecaster(
            horizon=self.horizon, hparams=dict(self.lstm_hparams),
            seed=self.seed, device=self.device,
        )
        self.lstm_model.fit(train, val,
                            train_series=train_arr,
                            val_series=val_series)

        # --- 4) Tune alpha_h on validation set ---
        if val is not None:
            logger.info("[cha_hybrid] tuning alpha_h on validation …")
            decomp_val = self._predict_decomposition(val.X)   # (N, h)
            global_val = self.lstm_model.predict(val)         # (N, h)
            best_alpha = None
            best_rmse = float("inf")
            search_log = []
            for a in self.alpha_search:
                combined = a * decomp_val + (1.0 - a) * global_val
                err = float(np.sqrt(np.mean((combined - val.y) ** 2)))
                search_log.append({"alpha": float(a), "val_rmse": err})
                if err < best_rmse:
                    best_rmse = err; best_alpha = float(a)
            self._chosen_alpha = best_alpha
            self.alpha_h = best_alpha
            self._val_alpha_search_diag = search_log
            logger.info("[cha_hybrid] h=%d chose alpha=%.2f (val_rmse=%.4f). "
                        "search: %s",
                        self.horizon, best_alpha, best_rmse,
                        " ".join(f"{r['alpha']:.1f}={r['val_rmse']:.4f}" for r in search_log))
        else:
            # no val provided — default to 0.5
            self.alpha_h = float(self.alpha_h) if self.alpha_h is not None else 0.5
            logger.info("[cha_hybrid] no val provided; alpha=%.2f", self.alpha_h)

        elapsed = time.perf_counter() - t0
        self.fit_report = FitReport(
            train_seconds=float(elapsed),
            n_train_samples=int(train.X.shape[0]),
            n_parameters=self._count_params(),
            extra={
                "stl_recon": self._train_recon_diag,
                "chosen_alpha": float(self.alpha_h),
                "alpha_search": self._val_alpha_search_diag,
            },
        )
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, windows) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("cha_hybrid: predict before fit")
        decomp = self._predict_decomposition(windows.X)
        global_pred = self.lstm_model.predict(windows)
        alpha = float(self.alpha_h)
        combined = alpha * decomp + (1.0 - alpha) * global_pred
        return self._check_pred(combined, n_expected=windows.X.shape[0])

    def predict_decomposition_only(self, windows) -> np.ndarray:
        """Ablation hook: return decomposition forecast without combination."""
        return self._predict_decomposition(windows.X)

    def predict_global_only(self, windows) -> np.ndarray:
        """Ablation hook: return global LSTM forecast without combination."""
        return self.lstm_model.predict(windows)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _predict_decomposition(self, X: np.ndarray) -> np.ndarray:
        """Per-window: STL on lookback, forecast trend+seasonal+residual, sum.

        For the residual component we hand the GRU a list of
        residual-series contexts (one per window) and read off the GRU's
        h-step forecast — that mirrors how the model was trained.
        """
        from darts import TimeSeries
        import pandas as pd

        N, L = X.shape
        h = self.horizon
        # 1) STL on every lookback
        trends = np.empty((N, L), dtype=np.float32)
        seasonals = np.empty((N, L), dtype=np.float32)
        residuals = np.empty((N, L), dtype=np.float32)
        for i in range(N):
            t_i, s_i, r_i = _stl_decompose(X[i], period=self.stl_period)
            trends[i], seasonals[i], residuals[i] = t_i, s_i, r_i

        # 2) trend forecasts
        trend_fc = np.zeros((N, h), dtype=np.float32)
        if self.trend_model_name == "arima":
            for i in range(N):
                trend_fc[i] = _trend_forecast_arima(trends[i], h, self.trend_arima_order)
        else:  # 'linear' (default — fast, faithful)
            for i in range(N):
                trend_fc[i] = _trend_forecast_linear(trends[i], h)

        # 3) seasonal forecasts (seasonal_naive)
        seasonal_fc = np.zeros((N, h), dtype=np.float32)
        for i in range(N):
            seasonal_fc[i] = _seasonal_forecast(seasonals[i], h, self.stl_period)

        # 4) residual forecasts via the trained GRU.  We feed it the
        # lookback residuals as a batched list of TimeSeries contexts.
        ctx_list = []
        base_idx = pd.date_range("2000-01-01", periods=L, freq="h")
        for i in range(N):
            ctx_list.append(TimeSeries.from_series(
                pd.Series(residuals[i].astype(np.float32), index=base_idx)
            ))
        preds = self.gru_model._darts_model.predict(
            n=h, series=ctx_list, verbose=False,
        )
        if isinstance(preds, TimeSeries):
            preds = [preds]
        residual_fc = np.stack(
            [np.asarray(p.values(), dtype=np.float32).ravel()[:h] for p in preds],
            axis=0,
        )

        decomp = trend_fc + seasonal_fc + residual_fc
        return decomp.astype(np.float32)

    def _count_params(self) -> int | None:
        n = 0
        for m in (self.gru_model, self.lstm_model):
            try:
                inner = m._darts_model.model
                n += int(sum(p.numel() for p in inner.parameters()))
            except Exception:
                pass
        return n or None


# ---------------------------------------------------------------------------
# small utility: find the largest contiguous non-NaN run in a 1-D array
# ---------------------------------------------------------------------------


def _largest_finite_chunk(arr: np.ndarray, min_len: int = 1) -> tuple[int, int] | None:
    """Return (start, end_exclusive) of the longest run of finite values."""
    finite = np.isfinite(arr)
    if not finite.any():
        return None
    # find runs of True
    pad = np.concatenate([[False], finite, [False]])
    diff = np.diff(pad.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    if starts.size == 0:
        return None
    lens = ends - starts
    i = int(np.argmax(lens))
    if int(lens[i]) < min_len:
        return None
    return int(starts[i]), int(ends[i])
