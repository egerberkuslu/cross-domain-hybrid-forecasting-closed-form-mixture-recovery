"""CHA-Hybrid v4-fix: like v4 but with proper held-out early stopping.

The original v4 trained the α(x) MLP on the full validation set and
early-stopped on the same set's MSE -- which is just "train to
convergence on val", not real early stopping.  On datasets with
val/test distribution drift (Abilene at h=1 / h=3) this caused the
MLP to overfit val and regress on test by up to +19 %.

This fix introduces a chronologically-internal split of the
validation set:

    val_train (first 80 %)  ->  optimiser sees this
    val_holdout (last 20 %) ->  early-stopping criterion

We also shrink the MLP (16 -> 8 hidden, 2 -> 1 layer = ~75 params
instead of ~400) and bump weight_decay 1e-4 -> 5e-3.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
from torch import nn

from .base import FitReport
from .cha_hybrid_v3 import CHAHybridV3Forecaster
from .cha_hybrid_v4 import _AlphaMLP, _context_features

logger = logging.getLogger(__name__)


class _AlphaMLPSmall(nn.Module):
    """Single hidden layer, 8 units."""

    def __init__(self, in_dim: int, hidden: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CHAHybridV4FixForecaster(CHAHybridV3Forecaster):
    """v3 + properly regularised, held-out-early-stopped α(x) MLP."""

    name = "cha_hybrid_v4_fix"

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon=horizon, hparams=hparams, seed=seed, device=device)
        amlp = dict(self.hparams.get("alpha_mlp", {}))
        self._alpha_hidden = int(amlp.get("hidden", 8))
        self._alpha_epochs = int(amlp.get("epochs", 500))
        self._alpha_lr = float(amlp.get("lr", 1e-3))
        self._alpha_wd = float(amlp.get("weight_decay", 5e-3))
        self._alpha_patience = int(amlp.get("patience", 30))
        self._val_holdout_frac = float(amlp.get("val_holdout_frac", 0.2))
        self.alpha_mlp: nn.Module | None = None
        self._alpha_mlp_diag: dict | None = None

    def fit(self, train, val=None, *, train_series=None, val_series=None):
        super().fit(train, val, train_series=train_series, val_series=val_series)
        if val is None:
            logger.warning("[v4fix] no val → scalar α_h fallback")
            return self
        t0 = time.perf_counter()
        decomp_val = self._predict_decomposition(val.X)
        global_val = self.global_model.predict(val)
        feat = _context_features(val.X, stl_period=self.stl_period)

        n = feat.shape[0]
        cut = max(int(n * (1 - self._val_holdout_frac)), 8)
        # CHRONOLOGICAL split — first cut goes to train, rest to holdout
        feat_tr = torch.tensor(feat[:cut], dtype=torch.float32, device=self.device)
        feat_ho = torch.tensor(feat[cut:], dtype=torch.float32, device=self.device)
        dec_tr = torch.tensor(decomp_val[:cut], dtype=torch.float32, device=self.device)
        dec_ho = torch.tensor(decomp_val[cut:], dtype=torch.float32, device=self.device)
        glb_tr = torch.tensor(global_val[:cut], dtype=torch.float32, device=self.device)
        glb_ho = torch.tensor(global_val[cut:], dtype=torch.float32, device=self.device)
        y_tr = torch.tensor(val.y[:cut], dtype=torch.float32, device=self.device)
        y_ho = torch.tensor(val.y[cut:], dtype=torch.float32, device=self.device)

        mlp = _AlphaMLPSmall(feat.shape[1], self._alpha_hidden).to(self.device)
        opt = torch.optim.Adam(
            mlp.parameters(), lr=self._alpha_lr, weight_decay=self._alpha_wd
        )
        best_ho = float("inf")
        best_state = None
        n_bad = 0
        history = []
        for ep in range(self._alpha_epochs):
            mlp.train()
            opt.zero_grad()
            a = mlp(feat_tr).unsqueeze(-1)
            mix = a * dec_tr + (1.0 - a) * glb_tr
            loss = ((mix - y_tr) ** 2).mean()
            loss.backward()
            opt.step()
            mlp.eval()
            with torch.no_grad():
                a_ho = mlp(feat_ho).unsqueeze(-1)
                mix_ho = a_ho * dec_ho + (1.0 - a_ho) * glb_ho
                ho_mse = float(((mix_ho - y_ho) ** 2).mean().item())
            history.append(
                {"epoch": ep, "tr_mse": float(loss.item()), "ho_mse": ho_mse}
            )
            if ho_mse < best_ho - 1e-6:
                best_ho = ho_mse
                best_state = {
                    k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()
                }
                n_bad = 0
            else:
                n_bad += 1
                if n_bad >= self._alpha_patience:
                    break
        if best_state is not None:
            mlp.load_state_dict(best_state)
        mlp.eval()
        with torch.no_grad():
            a_full = (
                mlp(torch.tensor(feat, dtype=torch.float32, device=self.device))
                .cpu()
                .numpy()
            )
        self.alpha_mlp = mlp
        self._alpha_mlp_diag = {
            "best_ho_mse": best_ho,
            "epochs_trained": int(len(history)),
            "epochs_no_improve_at_stop": int(n_bad),
            "alpha_min": float(np.min(a_full)),
            "alpha_max": float(np.max(a_full)),
            "alpha_mean": float(np.mean(a_full)),
            "alpha_std": float(np.std(a_full)),
            "val_holdout_frac": self._val_holdout_frac,
            "val_split_cut": int(cut),
            "n_val_train": int(cut),
            "n_val_holdout": int(n - cut),
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
        logger.info(
            "[v4fix] α(x) MLP: best_ho_mse=%.4f, α∈[%.2f,%.2f] mean=%.2f, "
            "%d epochs (val_tr=%d, val_ho=%d)",
            best_ho,
            self._alpha_mlp_diag["alpha_min"],
            self._alpha_mlp_diag["alpha_max"],
            self._alpha_mlp_diag["alpha_mean"],
            self._alpha_mlp_diag["epochs_trained"],
            cut,
            n - cut,
        )
        return self

    def predict(self, windows):
        if not self._fitted:
            raise RuntimeError(f"{self.name}: predict() before fit()")
        decomp_pred = self._predict_decomposition(windows.X)
        global_pred = self.global_model.predict(windows)
        if self.alpha_mlp is None:
            a = float(self.alpha_h)
            mix = a * decomp_pred + (1.0 - a) * global_pred
            return self._check_pred(mix, n_expected=windows.X.shape[0])
        feat = _context_features(windows.X, stl_period=self.stl_period)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device)
        self.alpha_mlp.eval()
        with torch.no_grad():
            a = self.alpha_mlp(feat_t).cpu().numpy().reshape(-1, 1)
        mix = a * decomp_pred + (1.0 - a) * global_pred
        return self._check_pred(mix.astype(np.float32), n_expected=windows.X.shape[0])

    def save_checkpoint(self, path):
        from pathlib import Path
        import json
        import torch as _torch

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            lstm_state = self.residual_model._darts_model.model.state_dict()
        except Exception:
            lstm_state = None
        mlp_state = self.alpha_mlp.state_dict() if self.alpha_mlp is not None else None
        ckpt = {
            "model_name": self.name,
            "horizon": self.horizon,
            "seed": self.seed,
            "alpha_h_v3_fallback": float(self.alpha_h)
            if self.alpha_h is not None
            else None,
            "alpha_mlp_state_dict": mlp_state,
            "alpha_mlp_diag": self._alpha_mlp_diag,
            "alpha_mlp_hparams": {
                "hidden": self._alpha_hidden,
                "epochs": self._alpha_epochs,
                "lr": self._alpha_lr,
                "weight_decay": self._alpha_wd,
                "patience": self._alpha_patience,
                "val_holdout_frac": self._val_holdout_frac,
            },
            "stl_period": self.stl_period,
            "lstm_residual_state_dict": lstm_state,
        }
        _torch.save(ckpt, p)
        meta = {
            k: v
            for k, v in ckpt.items()
            if k not in ("lstm_residual_state_dict", "alpha_mlp_state_dict")
        }
        with p.with_suffix(p.suffix + ".json").open("w") as f:
            json.dump(meta, f, indent=2, default=str)
        return p
