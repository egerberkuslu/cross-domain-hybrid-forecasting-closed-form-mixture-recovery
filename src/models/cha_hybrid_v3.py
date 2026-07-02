"""CHA-Hybrid v3 — same architecture as v2 but **global expert = Chronos-Bolt**.

Motivation
----------
Phase-5 results showed Chronos-Bolt (Amazon, 2024-2025) dominates the
cesnet leaderboard and beats TimesFM on abilene short horizons by
12-14 % RMSE.  Since CHA-Hybrid v2 used TimesFM as its global-foundation
expert, v2 inherited TimesFM's ceiling on cesnet — losing ~12 % to a
pure Chronos-Bolt baseline.

v3 simply swaps the global expert:

    decomp expert  =  STL(trend=Theta) + SeasonalNaive + LSTM-residual
    global expert  =  Chronos-Bolt (amazon/chronos-bolt-small)
    combined       =  α * decomp + (1 - α) * global,  α tuned per-horizon

Everything else (STL period, alpha grid, residual sub-model, val-set
α-tuning) is inherited from v2.  This is the minimal architectural
change that should let the mixture match or exceed every foundation
model we've seen so far.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .base import FitReport
from .cha_hybrid_v2 import CHAHybridV2Forecaster
from .deep_darts import LSTMForecaster
from .sota_2025 import ChronosBoltForecaster

logger = logging.getLogger(__name__)


class CHAHybridV3Forecaster(CHAHybridV2Forecaster):
    """v2 with Chronos-Bolt as the global expert (instead of TimesFM)."""

    name = "cha_hybrid_v3"
    is_stochastic = True
    supports_multi_horizon = True

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        hp = dict(hparams or {})
        # Allow caller to override which Chronos-Bolt size; default to small
        bolt_hp = dict(hp.pop("chronos_bolt", {}))
        bolt_hp.setdefault("pretrained", "amazon/chronos-bolt-small")
        bolt_hp.setdefault("batch_size_predict", 16)
        # store on hparams under a key v2 will ignore
        hp["_chronos_bolt_hparams"] = bolt_hp
        super().__init__(horizon=horizon, hparams=hp, seed=seed, device=device)

    def fit(self, train, val=None, *, train_series=None, val_series=None):
        """Same as v2 but builds the global expert as Chronos-Bolt."""
        if train_series is None:
            raise ValueError("cha_hybrid_v3 requires train_series (1-D scaled array)")
        t0 = time.perf_counter()

        # ---- 1) STL on the training series ----
        from .cha_hybrid import _stl_decompose, _largest_finite_chunk

        train_arr = np.asarray(train_series, dtype=np.float32)
        chunks = _largest_finite_chunk(train_arr, min_len=2 * self.stl_period + 8)
        if chunks is None:
            raise RuntimeError(
                "cha_hybrid_v3: no contiguous train chunk long enough for STL"
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
        logger.info("[cha_v3] training residual-LSTM …")
        self.residual_model = LSTMForecaster(
            horizon=self.horizon,
            hparams=dict(self.lstm_hparams),
            seed=self.seed,
            device=self.device,
        )
        self.residual_model.fit(
            train, val, train_series=residual_full, val_series=val_residual_full
        )

        # ---- 3) Global Chronos-Bolt (zero-shot, no fit) ----
        logger.info("[cha_v3] loading global-Chronos-Bolt (zero-shot) …")
        bolt_hp = dict(self.hparams.get("_chronos_bolt_hparams", {}))
        self.global_model = ChronosBoltForecaster(
            horizon=self.horizon,
            hparams=bolt_hp,
            seed=self.seed,
            device=self.device,
        )
        self.global_model.fit(train, val, train_series=train_arr, val_series=val_series)

        # ---- 4) α-tuning on validation set ----
        if val is not None:
            logger.info("[cha_v3] tuning alpha_h on validation …")
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
                "[cha_v3] h=%d chose alpha=%.2f (val_rmse=%.4f)",
                self.horizon,
                best_alpha,
                best_rmse,
            )
        else:
            self.alpha_h = float(self.alpha_h) if self.alpha_h is not None else 0.5

        elapsed = time.perf_counter() - t0
        # n_parameters: LSTM residual + Chronos-Bolt-small (~48M)
        bolt_params = 48_000_000
        try:
            lstm_params = int(
                sum(
                    p.numel()
                    for p in self.residual_model._darts_model.model.parameters()
                )
            )
        except Exception:
            lstm_params = 0
        self.fit_report = FitReport(
            train_seconds=float(elapsed),
            n_train_samples=int(train.X.shape[0]),
            n_parameters=lstm_params + bolt_params,
            extra={
                "stl_recon": self._train_recon_diag,
                "chosen_alpha": float(self.alpha_h),
                "alpha_search": self._val_alpha_search_diag,
            },
        )
        self._fitted = True
        return self

    def save_checkpoint(self, path):
        """Save trainable parts: LSTM-residual state_dict + α + base hparams.

        The global expert (Chronos-Bolt) is loaded from HuggingFace at
        ``amazon/chronos-bolt-small`` and is *not* fine-tuned, so we only
        record its identifier rather than re-saving 48M weights.
        """
        from pathlib import Path
        import json, torch as _torch

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # LSTM-residual sub-model state_dict
        try:
            lstm_state = self.residual_model._darts_model.model.state_dict()
        except Exception:
            lstm_state = None
        ckpt = {
            "model_name": self.name,
            "horizon": self.horizon,
            "seed": self.seed,
            "alpha_h": float(self.alpha_h) if self.alpha_h is not None else None,
            "stl_period": self.stl_period,
            "alpha_search_grid": list(self.alpha_search),
            "alpha_search_diag": self._val_alpha_search_diag,
            "stl_recon_diag": self._train_recon_diag,
            "lstm_hparams": dict(self.lstm_hparams),
            "global_expert": {
                "type": "chronos_bolt",
                "pretrained": self.hparams.get("_chronos_bolt_hparams", {}).get(
                    "pretrained", "amazon/chronos-bolt-small"
                ),
                "weights_source": "huggingface_hub (not stored — fully reproducible from pretrained id)",
            },
            "lstm_residual_state_dict": lstm_state,
        }
        _torch.save(ckpt, p)
        # marker JSON for human readability
        meta = {k: v for k, v in ckpt.items() if k != "lstm_residual_state_dict"}
        meta_path = p.with_suffix(p.suffix + ".json")
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2, default=str)
        return p
