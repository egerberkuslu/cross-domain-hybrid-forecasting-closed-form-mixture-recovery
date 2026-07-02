"""TimesFM forecaster (zero-shot only).

TimesFM ≥1.2 pins Python <3.12, so we use darts' built-in
``TimesFM2p5Model`` which loads the official ``google/timesfm-2.5-200m-pytorch``
checkpoint from HuggingFace and exposes it through the same TimeSeries
interface as the other darts models.

Fine-tuning is GPU- and disk-heavy, and TimesFM does not ship a small
ready-to-use fine-tune script for Python 3.12; per the Phase-3 spec we
explicitly skip it and log the reason.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.models import TimesFM2p5Model

from .base import WindowedForecaster, FitReport
from .deep_darts import _lookbacks_to_series_list

logger = logging.getLogger(__name__)


class TimesFMForecaster(WindowedForecaster):
    name = "timesfm"
    is_stochastic = False
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.input_chunk_length = int(hp.pop("input_chunk_length", 512))
        self.batch_size_predict = int(hp.pop("batch_size_predict", 32))
        self._extra = hp
        self._model: TimesFM2p5Model | None = None

    @property
    def variant_name(self) -> str:
        return "timesfm_zs"

    def _build(self) -> TimesFM2p5Model:
        device_str = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        logger.info("[timesfm] loading google/timesfm-2.5-200m-pytorch on %s", device_str)
        return TimesFM2p5Model(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.horizon,
        )

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        # Zero-shot — nothing to fit. Darts still requires a fit() call to
        # initialise weights, so we hand it a dummy training series of the
        # right length.
        logger.info("[timesfm] zero-shot mode — no training (per spec).")
        self._model = self._build()
        from darts import TimeSeries
        import pandas as pd
        dummy = np.zeros(self.input_chunk_length + self.horizon + 1, dtype=np.float32)
        s = TimeSeries.from_series(
            pd.Series(dummy, index=pd.date_range("2000-01-01",
                                                 periods=dummy.size, freq="h"))
        )
        self._model.fit(s, epochs=0, verbose=False)

    def _pad_lookbacks(self, X: np.ndarray) -> np.ndarray:
        """Left-pad each row with the row's first value to reach input_chunk_length."""
        n, L = X.shape
        if L >= self.input_chunk_length:
            return X[:, -self.input_chunk_length:]
        pad_width = self.input_chunk_length - L
        first = X[:, :1]
        pad = np.repeat(first, pad_width, axis=1)
        return np.concatenate([pad, X], axis=1)

    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            self._fit_arrays(X, None)
        X_padded = self._pad_lookbacks(X)
        ctx_list = _lookbacks_to_series_list(X_padded)
        bs = max(1, self.batch_size_predict)
        out: list[np.ndarray] = []
        for start in range(0, len(ctx_list), bs):
            chunk = ctx_list[start: start + bs]
            try:
                preds = self._model.predict(n=self.horizon, series=chunk, verbose=False)
                if isinstance(preds, TimeSeries):
                    preds = [preds]
                arr = np.stack(
                    [np.asarray(p.values(), dtype=np.float32).ravel()[: self.horizon]
                     for p in preds],
                    axis=0,
                )
                out.append(arr)
            except Exception as e:
                logger.warning("[timesfm] predict chunk failed (%s); last-value fallback.", e)
                fallback = np.stack(
                    [np.full(self.horizon, float(c.values()[-1]), dtype=np.float32)
                     for c in chunk]
                )
                out.append(fallback)
        return np.concatenate(out, axis=0).astype(np.float32)

    def _n_parameters(self) -> int | None:
        # TimesFM 2.5 200M
        return int(2e8)
