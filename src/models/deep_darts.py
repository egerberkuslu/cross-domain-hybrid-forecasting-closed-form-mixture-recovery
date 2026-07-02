"""Modern deep forecasting baselines via the ``darts`` library.

Six models share one wrapper:

  ============  ==================================  =========================
  Name           darts class                         Notes
  ============  ==================================  =========================
  lstm           BlockRNNModel(model="LSTM")        block / multi-step output
  gru            BlockRNNModel(model="GRU")         block / multi-step output
  tcn            TCNModel                           dilated causal conv
  nbeats         NBEATSModel                        generic stack
  dlinear        DLinearModel                       simple decomp + linear
  patchtst       (custom) — HuggingFace             PatchTST (Nie et al. 2023)
  ============  ==================================  =========================

PatchTST is not in darts so we implement it separately in
``patchtst_hf.py``; this file covers the five darts-native models.

Training data is the **scaled training series** (a 1-D numpy array; NaN-
containing long-gap regions are split into contiguous chunks so each
chunk is a valid darts ``TimeSeries``). Prediction is batched: each test
window's lookback is wrapped as a TimeSeries and passed in a list to
``model.predict(n=horizon, series=list)``.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import torch

# silence overly noisy library warnings
import warnings

warnings.filterwarnings("ignore", message=".*'pytorch_lightning'.*")
warnings.filterwarnings("ignore", category=UserWarning)

from darts import TimeSeries
from darts.models import (
    BlockRNNModel,
    DLinearModel,
    NBEATSModel,
    TCNModel,
    NHiTSModel,
    TFTModel,
    TiDEModel,
    TSMixerModel,
)

from .base import BaseForecaster, FitReport

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _series_to_contiguous_chunks(arr: np.ndarray, freq: str = "h") -> list[TimeSeries]:
    """Split a 1-D series with NaNs into a list of NaN-free TimeSeries chunks.

    Darts ``BlockRNNModel`` and friends accept a list of multiple training
    series; we exploit that to skip Abilene's inter-week gaps cleanly.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("expected 1-D series")
    # synthetic time index so darts doesn't complain
    idx = pd.date_range("2000-01-01", periods=arr.size, freq=freq)
    s = pd.Series(arr, index=idx)
    chunks: list[TimeSeries] = []
    # group by contiguous non-NaN runs
    isna = s.isna()
    group_id = isna.cumsum()  # increments at each NaN boundary
    for _, sub in s.dropna().groupby(group_id):
        if len(sub) < 8:  # require a few points
            continue
        # rebuild an evenly-spaced index for darts
        sub2 = pd.Series(
            sub.values,
            index=pd.date_range("2000-01-01", periods=len(sub), freq=freq),
        )
        chunks.append(TimeSeries.from_series(sub2))
    return chunks


def _lookbacks_to_series_list(X: np.ndarray, freq: str = "h") -> list[TimeSeries]:
    """Convert (N, L) lookbacks into N independent TimeSeries (for batched predict)."""
    out: list[TimeSeries] = []
    base = pd.date_range("2000-01-01", periods=X.shape[1], freq=freq)
    for i in range(X.shape[0]):
        s = pd.Series(X[i].astype(np.float32), index=base)
        out.append(TimeSeries.from_series(s))
    return out


# ----------------------------------------------------------------------
# wrapper base
# ----------------------------------------------------------------------


class _DartsBlockBase(BaseForecaster):
    """Common scaffolding for darts deep models with explicit output_chunk_length."""

    DARTS_CLASS: type = None  # subclass sets

    is_stochastic = True  # neural nets benefit from multi-seed runs
    supports_multi_horizon = True

    # default training hyperparameters (overridable via hparams)
    DEFAULT_BATCH_SIZE = 64
    DEFAULT_EPOCHS = 50
    DEFAULT_LR = 1e-3
    DEFAULT_PATIENCE = 8

    # Subclasses list the model-specific darts kwargs they accept (so we
    # can silently drop unrelated hparams from other models without darts
    # raising "Invalid model creation parameters").
    ALLOWED_MODEL_KWARGS: tuple[str, ...] = ()

    def __init__(self, horizon, hparams=None, seed=42, device="cpu"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.input_chunk_length = int(hp.pop("input_chunk_length", 168))
        self.batch_size = int(hp.pop("batch_size", self.DEFAULT_BATCH_SIZE))
        self.n_epochs = int(hp.pop("n_epochs", self.DEFAULT_EPOCHS))
        self.lr = float(hp.pop("lr", self.DEFAULT_LR))
        self.patience = int(hp.pop("patience", self.DEFAULT_PATIENCE))
        self._extra = hp  # may contain hparams from other models
        self._darts_model: Any = None

    # ---- subclass hook ----

    def _make_kwargs(self) -> dict:
        """Compose constructor kwargs for the darts class."""
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            mode="min",
        )
        accelerator = (
            "gpu" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        trainer_kwargs = {
            "accelerator": accelerator,
            "devices": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
            "callbacks": [early_stop],
        }
        # only keep _extra keys that this concrete model understands
        filtered_extra = {
            k: v for k, v in self._extra.items() if k in self.ALLOWED_MODEL_KWARGS
        }
        return dict(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.horizon,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            optimizer_kwargs={"lr": self.lr},
            random_state=self.seed,
            pl_trainer_kwargs=trainer_kwargs,
            **filtered_extra,
        )

    # ---- fit / predict ----

    def fit(
        self,
        train,
        val=None,
        *,
        train_series=None,
        val_series=None,
    ):
        import time

        if train_series is None:
            raise ValueError(f"{self.name}: requires train_series (1-D scaled array)")
        train_ts = _series_to_contiguous_chunks(train_series)
        val_ts = (
            _series_to_contiguous_chunks(val_series) if val_series is not None else None
        )
        kwargs = self._make_kwargs()
        self._darts_model = self.DARTS_CLASS(**kwargs)
        t0 = time.perf_counter()
        # darts' BlockRNN can train on a list of multiple series; if any chunk
        # is shorter than input+output we drop it
        keep_train = [
            s for s in train_ts if len(s) >= self.input_chunk_length + self.horizon
        ]
        if not keep_train:
            raise RuntimeError(f"{self.name}: no training chunk long enough")
        keep_val = None
        if val_ts is not None:
            keep_val = [
                s for s in val_ts if len(s) >= self.input_chunk_length + self.horizon
            ]
            if not keep_val:
                keep_val = None
        self._darts_model.fit(series=keep_train, val_series=keep_val, verbose=False)
        elapsed = time.perf_counter() - t0
        self.fit_report = FitReport(
            train_seconds=float(elapsed),
            n_train_samples=int(train.X.shape[0]),
            n_parameters=self._count_params(),
        )
        self._fitted = True
        return self

    def predict(self, windows):
        if not self._fitted:
            raise RuntimeError(f"{self.name}: predict before fit")
        ctx_list = _lookbacks_to_series_list(windows.X)
        # batched predict over a list of series
        preds = self._darts_model.predict(
            n=self.horizon, series=ctx_list, verbose=False
        )
        # darts returns a list of TimeSeries when input is a list
        if isinstance(preds, TimeSeries):
            preds = [preds]
        out = np.stack(
            [
                np.asarray(p.values(), dtype=np.float32).ravel()[: self.horizon]
                for p in preds
            ],
            axis=0,
        )
        return self._check_pred(out, n_expected=windows.X.shape[0])

    # _DartsBlockBase overrides fit/predict directly; the WindowedForecaster
    # abstract hooks are unused here, but satisfy BaseForecaster's contract.
    pass

    def _count_params(self) -> int | None:
        try:
            mod = self._darts_model.model
            return int(sum(p.numel() for p in mod.parameters()))
        except Exception:
            return None


# ----------------------------------------------------------------------
# concrete model classes
# ----------------------------------------------------------------------


class LSTMForecaster(_DartsBlockBase):
    name = "lstm"
    DARTS_CLASS = BlockRNNModel
    ALLOWED_MODEL_KWARGS = ("hidden_dim", "n_rnn_layers", "dropout")

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw["model"] = "LSTM"
        kw.setdefault("hidden_dim", 64)
        kw.setdefault("n_rnn_layers", 1)
        kw.setdefault("dropout", 0.1)
        return kw


class GRUForecaster(_DartsBlockBase):
    name = "gru"
    DARTS_CLASS = BlockRNNModel
    ALLOWED_MODEL_KWARGS = ("hidden_dim", "n_rnn_layers", "dropout")

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw["model"] = "GRU"
        kw.setdefault("hidden_dim", 64)
        kw.setdefault("n_rnn_layers", 1)
        kw.setdefault("dropout", 0.1)
        return kw


class TCNForecaster(_DartsBlockBase):
    name = "tcn"
    DARTS_CLASS = TCNModel
    ALLOWED_MODEL_KWARGS = (
        "num_filters",
        "kernel_size",
        "dropout",
        "weight_norm",
        "dilation_base",
    )

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw.setdefault("num_filters", 16)
        kw.setdefault("kernel_size", 3)
        kw.setdefault("dropout", 0.1)
        return kw


class NBEATSForecaster(_DartsBlockBase):
    name = "nbeats"
    DARTS_CLASS = NBEATSModel
    ALLOWED_MODEL_KWARGS = (
        "num_stacks",
        "num_blocks",
        "num_layers",
        "layer_widths",
        "expansion_coefficient_dim",
        "trend_polynomial_degree",
    )

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw.setdefault("num_stacks", 2)
        kw.setdefault("num_blocks", 2)
        kw.setdefault("num_layers", 2)
        kw.setdefault("layer_widths", 64)
        return kw


class DLinearForecaster(_DartsBlockBase):
    name = "dlinear"
    DARTS_CLASS = DLinearModel
    ALLOWED_MODEL_KWARGS = ("kernel_size", "const_init", "use_static_covariates")

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw.setdefault("kernel_size", 25)
        return kw


# ----------------------------------------------------------------------
# Modern 2023-2024 SOTA additions (per Aouedi et al. 2025 survey,
# Liu et al. 2026 cloud-workload paper, Koumar et al. 2025 TNSM benchmark)
# ----------------------------------------------------------------------


class NHiTSForecaster(_DartsBlockBase):
    """N-HiTS (Challu et al., AAAI 2023) — N-BEATS's modern successor with
    multi-rate sampling and hierarchical interpolation. Strong on long horizons.
    """

    name = "nhits"
    DARTS_CLASS = NHiTSModel
    ALLOWED_MODEL_KWARGS = (
        "num_stacks",
        "num_blocks",
        "num_layers",
        "layer_widths",
        "pooling_kernel_sizes",
        "n_freq_downsample",
        "dropout",
        "activation",
    )

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw.setdefault("num_stacks", 3)
        kw.setdefault("num_blocks", 1)
        kw.setdefault("num_layers", 2)
        kw.setdefault("layer_widths", 64)
        kw.setdefault("dropout", 0.1)
        return kw


class TFTForecaster(_DartsBlockBase):
    """Temporal Fusion Transformer (Lim et al., 2021) — interpretable
    attention + LSTM hybrid. Still SOTA-class per most 2024-2025 surveys.
    """

    name = "tft"
    DARTS_CLASS = TFTModel
    ALLOWED_MODEL_KWARGS = (
        "hidden_size",
        "lstm_layers",
        "num_attention_heads",
        "dropout",
        "hidden_continuous_size",
        "categorical_embedding_sizes",
        "add_relative_index",
        "add_encoders",
    )

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw.setdefault("hidden_size", 32)
        kw.setdefault("lstm_layers", 1)
        kw.setdefault("num_attention_heads", 4)
        kw.setdefault("dropout", 0.1)
        # TFT in darts requires either future_covariates or this flag
        kw["add_relative_index"] = True
        return kw


class TiDEForecaster(_DartsBlockBase):
    """TiDE (Das et al., TMLR 2024) — Google's long-term forecasting model,
    pure-MLP encoder-decoder. Strong baseline per multiple 2024-2025 surveys.
    """

    name = "tide"
    DARTS_CLASS = TiDEModel
    ALLOWED_MODEL_KWARGS = (
        "num_encoder_layers",
        "num_decoder_layers",
        "decoder_output_dim",
        "hidden_size",
        "temporal_width_past",
        "temporal_width_future",
        "temporal_decoder_hidden",
        "use_layer_norm",
        "dropout",
    )

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw.setdefault("num_encoder_layers", 2)
        kw.setdefault("num_decoder_layers", 2)
        kw.setdefault("decoder_output_dim", 8)
        kw.setdefault("hidden_size", 64)
        kw.setdefault("dropout", 0.1)
        return kw


class TSMixerForecaster(_DartsBlockBase):
    """TSMixer (Chen et al., TMLR 2023) — Google's all-MLP architecture for
    time-series. Cited as backbone of MSCAF cloud-workload model (Liu 2026).
    """

    name = "tsmixer"
    DARTS_CLASS = TSMixerModel
    ALLOWED_MODEL_KWARGS = (
        "hidden_size",
        "ff_size",
        "num_blocks",
        "activation",
        "dropout",
        "norm_type",
    )

    def _make_kwargs(self):
        kw = super()._make_kwargs()
        kw.setdefault("hidden_size", 64)
        kw.setdefault("ff_size", 64)
        kw.setdefault("num_blocks", 2)
        kw.setdefault("dropout", 0.1)
        return kw
