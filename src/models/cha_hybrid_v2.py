"""CHA-Hybrid v2 — Adaptive Mixture of Classical Decomposition and Foundation Model.

Differences vs v1:

* **Trend** forecast uses Theta (Assimakopoulos & Nikolopoulos 2000) instead
  of linear extrapolation — Theta dominates the GEANT baseline by a wide
  margin, suggesting it captures the trend behaviour better when the
  series shows a level shift over the lookback.
* **Residual** forecast uses an LSTM (the v1 ablation showed LSTM-residual
  beat GRU-residual 8 / 5 / 2 across the (dataset × horizon) cells).
* **Global** forecast replaces the in-house BlockRNN(LSTM) with a
  pre-trained foundation model (TimesFM 2.5 zero-shot). This is the
  conceptual novelty: the per-horizon α weight now arbitrates between a
  classical-decomposition expert and a foundation-model expert, picking
  whichever generalises better on the validation set.
* The per-horizon α grid is fine-tuned to step 0.05.

The same ``predict_decomposition_only`` / ``predict_global_only`` hooks
remain so the same Phase-6 ablation pipeline works.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from .base import BaseForecaster, FitReport
from .deep_darts import LSTMForecaster, _lookbacks_to_series_list
from .timesfm_model import TimesFMForecaster
from .cha_hybrid import (
    _stl_decompose,
    _trend_forecast_linear,
    _seasonal_forecast,
    _largest_finite_chunk,
)

logger = logging.getLogger(__name__)


def _trend_forecast_theta(
    trend: np.ndarray, h: int, seasonality: int = 1
) -> np.ndarray:
    """Forecast the STL trend component with the Theta method (statsforecast)."""
    from statsforecast.models import Theta as SFTheta

    try:
        m = SFTheta(season_length=seasonality)
        m.fit(trend.astype(np.float64))
        return np.asarray(m.predict(h=h)["mean"], dtype=np.float32)
    except Exception:
        return _trend_forecast_linear(trend, h)


class CHAHybridV2Forecaster(BaseForecaster):
    name = "cha_hybrid_v2"
    is_stochastic = True
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.stl_period = int(hp.pop("stl_period", 24))
        self.alpha_search = list(
            hp.pop("alpha_search", [round(0.05 * i, 2) for i in range(21)])
        )
        self.alpha_h: float | None = hp.pop("alpha", None)

        # residual-LSTM hyperparameters
        self.lstm_hparams = dict(hp.pop("lstm", {}))
        self.lstm_hparams.setdefault("input_chunk_length", 168)
        self.lstm_hparams.setdefault("n_epochs", 30)
        self.lstm_hparams.setdefault("patience", 6)
        self.lstm_hparams.setdefault("hidden_dim", 64)

        # global TimesFM hyperparameters
        self.timesfm_hparams = dict(hp.pop("timesfm", {}))
        self.timesfm_hparams.setdefault("input_chunk_length", 512)
        self.timesfm_hparams.setdefault("batch_size_predict", 16)

        self._extra = hp
        self.residual_model: LSTMForecaster | None = None
        self.global_model: TimesFMForecaster | None = None
        self._train_recon_diag: dict | None = None
        self._val_alpha_search_diag: list[dict] | None = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, train, val=None, *, train_series=None, val_series=None):
        if train_series is None:
            raise ValueError("cha_hybrid_v2 requires train_series (scaled 1-D array)")
        t0 = time.perf_counter()

        # ---- 1) STL on the training series ----
        train_arr = np.asarray(train_series, dtype=np.float32)
        chunks = _largest_finite_chunk(train_arr, min_len=2 * self.stl_period + 8)
        if chunks is None:
            raise RuntimeError(
                "cha_hybrid_v2: no contiguous train chunk long enough for STL"
            )
        cs, ce = chunks
        clean = train_arr[cs:ce]
        trend, seas, resid = _stl_decompose(clean, period=self.stl_period)
        recon_err = float(np.max(np.abs((trend + seas + resid) - clean)))
        self._train_recon_diag = {
            "max_abs_err": recon_err,
            "rel_err_vs_std": float(recon_err / (np.std(clean) + 1e-12)),
            "n": int(clean.size),
        }
        residual_full = np.full_like(train_arr, np.nan, dtype=np.float32)
        residual_full[cs:ce] = resid

        if val_series is not None:
            val_arr = np.asarray(val_series, dtype=np.float32)
            vchunks = _largest_finite_chunk(val_arr, min_len=2 * self.stl_period + 8)
            if vchunks is not None:
                _, vs, vr = _stl_decompose(
                    val_arr[vchunks[0] : vchunks[1]], period=self.stl_period
                )
                val_residual_full = np.full_like(val_arr, np.nan, dtype=np.float32)
                val_residual_full[vchunks[0] : vchunks[1]] = vr
            else:
                val_residual_full = None
        else:
            val_residual_full = None

        # ---- 2) Residual LSTM ----
        logger.info("[cha_v2] training residual-LSTM …")
        self.residual_model = LSTMForecaster(
            horizon=self.horizon,
            hparams=dict(self.lstm_hparams),
            seed=self.seed,
            device=self.device,
        )
        self.residual_model.fit(
            train, val, train_series=residual_full, val_series=val_residual_full
        )

        # ---- 3) Global TimesFM (zero-shot, no fit) ----
        logger.info("[cha_v2] loading global-TimesFM (zero-shot) …")
        self.global_model = TimesFMForecaster(
            horizon=self.horizon,
            hparams=dict(self.timesfm_hparams),
            seed=self.seed,
            device=self.device,
        )
        self.global_model.fit(train, val, train_series=train_arr, val_series=val_series)

        # ---- 4) α-tuning on validation set ----
        if val is not None:
            logger.info("[cha_v2] tuning alpha_h on validation …")
            decomp_val = self._predict_decomposition(val.X)
            global_val = self.global_model.predict(val)
            best_alpha = None
            best_rmse = float("inf")
            search_log = []
            for a in self.alpha_search:
                combined = a * decomp_val + (1.0 - a) * global_val
                err = float(np.sqrt(np.mean((combined - val.y) ** 2)))
                search_log.append({"alpha": float(a), "val_rmse": err})
                if err < best_rmse:
                    best_rmse = err
                    best_alpha = float(a)
            self.alpha_h = best_alpha
            self._val_alpha_search_diag = search_log
            logger.info(
                "[cha_v2] h=%d chose alpha=%.2f (val_rmse=%.4f). search: %s",
                self.horizon,
                best_alpha,
                best_rmse,
                " ".join(f"{r['alpha']:.2f}={r['val_rmse']:.4f}" for r in search_log),
            )
        else:
            self.alpha_h = float(self.alpha_h) if self.alpha_h is not None else 0.5

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
            raise RuntimeError("cha_hybrid_v2: predict before fit")
        decomp = self._predict_decomposition(windows.X)
        global_pred = self.global_model.predict(windows)
        a = float(self.alpha_h)
        combined = a * decomp + (1.0 - a) * global_pred
        return self._check_pred(combined, n_expected=windows.X.shape[0])

    def predict_decomposition_only(self, windows) -> np.ndarray:
        return self._predict_decomposition(windows.X)

    def predict_global_only(self, windows) -> np.ndarray:
        return self.global_model.predict(windows)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _predict_decomposition(self, X: np.ndarray) -> np.ndarray:
        from darts import TimeSeries

        N, L = X.shape
        h = self.horizon

        # STL on every lookback
        trends = np.empty((N, L), dtype=np.float32)
        seasonals = np.empty((N, L), dtype=np.float32)
        residuals = np.empty((N, L), dtype=np.float32)
        for i in range(N):
            t_i, s_i, r_i = _stl_decompose(X[i], period=self.stl_period)
            trends[i], seasonals[i], residuals[i] = t_i, s_i, r_i

        # trend with Theta
        trend_fc = np.zeros((N, h), dtype=np.float32)
        for i in range(N):
            trend_fc[i] = _trend_forecast_theta(trends[i], h, seasonality=1)

        # seasonal with seasonal-naive
        seasonal_fc = np.zeros((N, h), dtype=np.float32)
        for i in range(N):
            seasonal_fc[i] = _seasonal_forecast(seasonals[i], h, self.stl_period)

        # residual with LSTM (batched darts predict)
        base_idx = pd.date_range("2000-01-01", periods=L, freq="h")
        ctx_list = [
            TimeSeries.from_series(
                pd.Series(residuals[i].astype(np.float32), index=base_idx)
            )
            for i in range(N)
        ]
        preds = self.residual_model._darts_model.predict(
            n=h, series=ctx_list, verbose=False
        )
        if isinstance(preds, TimeSeries):
            preds = [preds]
        residual_fc = np.stack(
            [np.asarray(p.values(), dtype=np.float32).ravel()[:h] for p in preds],
            axis=0,
        )

        return (trend_fc + seasonal_fc + residual_fc).astype(np.float32)

    def _count_params(self) -> int | None:
        n = 0
        try:
            n += int(
                sum(
                    p.numel()
                    for p in self.residual_model._darts_model.model.parameters()
                )
            )
        except Exception:
            pass
        # TimesFM 2.5 = 200M (treat as a fixed constant since it's pretrained)
        n += 200_000_000
        return n
