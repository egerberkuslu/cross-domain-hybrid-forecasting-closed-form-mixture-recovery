"""Modern 2024-2025 SOTA foundation-model baselines.

Wrappers around three recent zero-shot foundation models that are widely
benchmarked in the 2024-2025 TS-forecasting literature:

  * **Chronos-Bolt** (Amazon, 2024-2025) — distilled, encoder-only version
    of Chronos-T5; much faster than the original chronos-T5 family.
    HuggingFace IDs: ``amazon/chronos-bolt-{tiny,mini,small,base}``.
  * **MOIRAI** (Woo et al., ICML **2024**) — Salesforce's "universal" TS
    forecasting transformer (decoder-only, mixed-frequency pretraining).
    HF IDs: ``Salesforce/moirai-1.1-R-{small,base,large}``.
  * **TTM** (Tiny Time Mixers, Ekambaram et al., NeurIPS **2024**) — IBM
    Granite's lightweight (1-5M params) foundation model. HF ID:
    ``ibm-granite/granite-timeseries-ttm-r2``.

All three are *zero-shot*: ``fit`` only loads the pretrained weights;
``predict`` does the forecast.  This matches our existing
``timesfm_zs`` / ``chronos_zs`` conventions.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .base import BaseForecaster, FitReport
from .chronos_model import ChronosForecaster

logger = logging.getLogger(__name__)


# ===========================================================================
# Chronos-Bolt — same code path as Chronos but with bolt pretrained id.
# We expose a thin subclass so the registry lists it as a distinct variant.
# ===========================================================================


class ChronosBoltForecaster(BaseForecaster):
    """Chronos-Bolt zero-shot wrapper.

    Bolt is a distilled, encoder-only Chronos variant. The chronos-forecasting
    package exposes it via a separate ``ChronosBoltPipeline`` (different
    interface from the T5 ``ChronosPipeline``).
    """

    name = "chronos_bolt"
    is_stochastic = False
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.pretrained = hp.pop("pretrained", "amazon/chronos-bolt-small")
        self.batch_size_predict = int(hp.pop("batch_size_predict", 16))
        self._extra = hp
        self._device = (
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.pipeline = None

    @property
    def variant_name(self) -> str:
        return "chronos_bolt_zs"

    def _load(self):
        from chronos import ChronosBoltPipeline

        if self.pipeline is None:
            logger.info(
                "[chronos_bolt] loading %s on %s", self.pretrained, self._device
            )
            self.pipeline = ChronosBoltPipeline.from_pretrained(
                self.pretrained,
                device_map=self._device,
                dtype=torch.float32 if self._device == "cpu" else torch.bfloat16,
            )

    def fit(self, train, val=None, *, train_series=None, val_series=None):
        import time

        t0 = time.perf_counter()
        self._load()
        n_params = None
        try:
            n_params = int(sum(p.numel() for p in self.pipeline.model.parameters()))
        except Exception:
            pass
        self.fit_report = FitReport(
            train_seconds=float(time.perf_counter() - t0),
            n_train_samples=int(train.X.shape[0]),
            n_parameters=n_params,
        )
        self._fitted = True
        return self

    @torch.no_grad()
    def predict(self, windows):
        if self.pipeline is None:
            self._load()
        contexts = [torch.tensor(x, dtype=torch.float32) for x in windows.X]
        preds = np.zeros((windows.X.shape[0], self.horizon), dtype=np.float32)
        bs = max(1, self.batch_size_predict)
        for start in range(0, len(contexts), bs):
            chunk = contexts[start : start + bs]
            try:
                # Bolt returns a (B, num_quantiles, h) tensor; mean over quantiles
                quants = self.pipeline.predict(
                    chunk,
                    prediction_length=self.horizon,
                    limit_prediction_length=False,
                )
                arr = quants.detach().float().cpu().numpy()
                # take the median quantile (middle index) — typically [.1, .2 ... .9]
                if arr.ndim == 3:
                    mid = arr.shape[1] // 2
                    out = arr[:, mid, :]
                else:
                    out = arr
                preds[start : start + out.shape[0]] = out.astype(np.float32)
            except Exception as e:
                logger.warning("[chronos_bolt] predict chunk failed (%s); fallback", e)
                for j, ctx in enumerate(chunk):
                    preds[start + j] = float(ctx[-1])
        return self._check_pred(preds, n_expected=windows.X.shape[0])


# ===========================================================================
# MOIRAI — Salesforce ICML 2024
# ===========================================================================


class MoiraiForecaster(BaseForecaster):
    name = "moirai"
    is_stochastic = False
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.pretrained = hp.pop("pretrained", "Salesforce/moirai-1.1-R-small")
        self.input_chunk_length = int(hp.pop("input_chunk_length", 512))
        self.patch_size: str | int = hp.pop("patch_size", "auto")
        self.num_samples = int(hp.pop("num_samples", 20))
        self.batch_size_predict = int(hp.pop("batch_size_predict", 16))
        self._extra = hp
        self._device = (
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self._module = None
        self._n_params = None

    @property
    def variant_name(self) -> str:
        return "moirai_zs"

    def _load(self):
        from uni2ts.model.moirai import MoiraiModule

        if self._module is None:
            logger.info("[moirai] loading %s on %s", self.pretrained, self._device)
            self._module = MoiraiModule.from_pretrained(self.pretrained)
            self._module = self._module.to(self._device)
            self._module.eval()
            self._n_params = int(sum(p.numel() for p in self._module.parameters()))

    def fit(self, train, val=None, *, train_series=None, val_series=None):
        import time

        t0 = time.perf_counter()
        self._load()
        self.fit_report = FitReport(
            train_seconds=float(time.perf_counter() - t0),
            n_train_samples=int(train.X.shape[0]),
            n_parameters=self._n_params,
        )
        self._fitted = True
        return self

    @torch.no_grad()
    def predict(self, windows):
        from uni2ts.model.moirai import MoiraiForecast

        if not self._fitted:
            self.fit(windows)
        X = np.asarray(windows.X, dtype=np.float32)
        N, L = X.shape
        h = self.horizon
        # Pad each lookback to input_chunk_length (left-pad with first value)
        if L < self.input_chunk_length:
            pad = np.repeat(X[:, :1], self.input_chunk_length - L, axis=1)
            X_pad = np.concatenate([pad, X], axis=1)
        else:
            X_pad = X[:, -self.input_chunk_length :]
        ctx_len = X_pad.shape[1]

        # Build the MOIRAI forecast wrapper once per call
        mfcast = (
            MoiraiForecast(
                module=self._module,
                prediction_length=h,
                context_length=ctx_len,
                patch_size=self.patch_size,
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            )
            .to(self._device)
            .eval()
        )

        preds = np.zeros((N, h), dtype=np.float32)
        bs = max(1, self.batch_size_predict)
        for start in range(0, N, bs):
            chunk = X_pad[start : start + bs]
            try:
                past = (
                    torch.from_numpy(chunk).float().unsqueeze(-1).to(self._device)
                )  # (B, L, 1)
                past_observed = torch.ones_like(past, dtype=torch.bool)
                past_is_pad = torch.zeros(
                    past.shape[0], past.shape[1], dtype=torch.bool, device=self._device
                )
                samples = mfcast(
                    past_target=past,
                    past_observed_target=past_observed,
                    past_is_pad=past_is_pad,
                )
                # samples shape: (B, num_samples, h, 1)
                arr = samples.detach().float().cpu().numpy()
                if arr.ndim == 4:
                    arr = arr[..., 0]  # (B, num_samples, h)
                median = np.median(arr, axis=1).astype(np.float32)
                preds[start : start + median.shape[0]] = median
            except Exception as e:
                logger.warning(
                    "[moirai] predict chunk failed (%s) — last-value fallback", e
                )
                for j in range(chunk.shape[0]):
                    preds[start + j] = float(chunk[j, -1])
        return self._check_pred(preds, n_expected=N)


# ===========================================================================
# TTM — IBM Granite Tiny Time Mixers (NeurIPS 2024)
# ===========================================================================


class TTMForecaster(BaseForecaster):
    name = "ttm"
    is_stochastic = False
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.pretrained = hp.pop("pretrained", "ibm-granite/granite-timeseries-ttm-r2")
        self.context_length = int(hp.pop("context_length", 512))
        self.batch_size_predict = int(hp.pop("batch_size_predict", 16))
        self._extra = hp
        self._device = (
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self._model = None
        self._n_params = None

    @property
    def variant_name(self) -> str:
        return "ttm_zs"

    def _load(self):
        if self._model is not None:
            return
        # TTM has its own architecture (TinyTimeMixer) shipped by IBM in the
        # ``tsfm_public`` package; transformers' AutoModel cannot resolve it
        # by default, so we use the dedicated class.
        from tsfm_public import TinyTimeMixerForPrediction

        logger.info("[ttm] loading %s on %s", self.pretrained, self._device)
        self._model = (
            TinyTimeMixerForPrediction.from_pretrained(
                self.pretrained,
                # TTM-r2 expects context_length=512 by default; override allowed
            )
            .to(self._device)
            .eval()
        )
        # Try to read the model's native prediction & context length
        cfg = getattr(self._model, "config", None)
        if cfg is not None:
            self._native_pred_len = int(getattr(cfg, "prediction_length", self.horizon))
            self._native_ctx_len = int(
                getattr(cfg, "context_length", self.context_length)
            )
        else:
            self._native_pred_len = self.horizon
            self._native_ctx_len = self.context_length
        self._n_params = int(sum(p.numel() for p in self._model.parameters()))

    def fit(self, train, val=None, *, train_series=None, val_series=None):
        import time

        t0 = time.perf_counter()
        self._load()
        self.fit_report = FitReport(
            train_seconds=float(time.perf_counter() - t0),
            n_train_samples=int(train.X.shape[0]),
            n_parameters=self._n_params,
        )
        self._fitted = True
        return self

    @torch.no_grad()
    def predict(self, windows):
        if not self._fitted:
            self.fit(windows)
        X = np.asarray(windows.X, dtype=np.float32)
        N, L = X.shape
        h = self.horizon

        if L < self.context_length:
            pad = np.repeat(X[:, :1], self.context_length - L, axis=1)
            X_pad = np.concatenate([pad, X], axis=1)
        else:
            X_pad = X[:, -self.context_length :]

        preds = np.zeros((N, h), dtype=np.float32)
        bs = max(1, self.batch_size_predict)
        for start in range(0, N, bs):
            chunk = X_pad[start : start + bs]
            try:
                past = (
                    torch.from_numpy(chunk).float().unsqueeze(-1).to(self._device)
                )  # (B, L, 1)
                # TTM forward returns predictions in 'prediction_outputs'
                out = self._model(past_values=past)
                # Accept several output schemas
                if hasattr(out, "prediction_outputs"):
                    pred = out.prediction_outputs
                elif hasattr(out, "logits"):
                    pred = out.logits
                else:
                    pred = out
                # Shape: (B, forecast_length, n_channels)
                arr = pred.detach().float().cpu().numpy()
                if arr.ndim == 3:
                    arr = arr[..., 0]  # (B, h_native)
                # truncate or pad to requested horizon
                if arr.shape[1] >= h:
                    out_h = arr[:, :h]
                else:
                    last = arr[:, -1:]
                    extra = np.repeat(last, h - arr.shape[1], axis=1)
                    out_h = np.concatenate([arr, extra], axis=1)
                preds[start : start + out_h.shape[0]] = out_h.astype(np.float32)
            except Exception as e:
                logger.warning(
                    "[ttm] predict chunk failed (%s) — last-value fallback", e
                )
                for j in range(chunk.shape[0]):
                    preds[start + j] = float(chunk[j, -1])
        return self._check_pred(preds, n_expected=N)
