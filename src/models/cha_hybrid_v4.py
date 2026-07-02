"""CHA-Hybrid v4 — same experts as v3, but the mixture weight is a *learned*
per-sample function ``α(x)`` instead of a single per-horizon scalar.

Motivation
----------
v3 uses one scalar ``α_h`` tuned on the validation set; that scalar must
serve every sample (every context window).  Some context windows are
clearly structural (strong daily seasonality, stable residuals) — the
decomposition path should dominate.  Other windows show regime shifts
or anomalies — the foundation model should dominate.  A single ``α_h``
splits the difference.

v4 replaces ``α_h`` with a small MLP

    α(x) = σ(MLP(φ(x)))

where ``φ(x)`` is a hand-crafted, normalisation-invariant feature vector
extracted from the context window (mean, std, slope, residual variance,
seasonal-phase indicator, last-value gap, etc.).  The MLP is trained
*after* both experts are fitted, by minimising

    L = mean_i [(α(x_i) · y_dec_i + (1−α(x_i)) · y_glob_i − y_true_i)^2]

on the validation set.  Both experts stay frozen.

This is the methodological-novelty piece that turns "we picked a good
combination weight" into "we *learned* the combination function" — the
move from a single number to a learned policy.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import torch
from torch import nn

from .base import FitReport
from .cha_hybrid_v3 import CHAHybridV3Forecaster

logger = logging.getLogger(__name__)


# ---------- feature extraction (deterministic, scale-aware) ----------


def _context_features(X: np.ndarray, stl_period: int = 24) -> np.ndarray:
    """Map a batch of lookbacks (N, L) to features (N, F).

    All features are normalisation-invariant w.r.t. the StandardScaler
    used at preprocessing (mean ≈ 0, std ≈ 1 globally), so the MLP can
    transfer across datasets without re-training.
    """
    X = np.asarray(X, dtype=np.float64)
    N, L = X.shape
    out = np.zeros((N, 8), dtype=np.float32)
    out[:, 0] = X.mean(axis=1)
    out[:, 1] = X.std(axis=1)
    # slope (least-squares on the lookback)
    t = np.arange(L, dtype=np.float64)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum() + 1e-12
    out[:, 2] = ((X - X.mean(axis=1, keepdims=True)) * (t - t_mean)).sum(axis=1) / t_var
    # last-value vs trailing window mean
    last_n = min(stl_period, L)
    out[:, 3] = X[:, -1] - X[:, -last_n:].mean(axis=1)
    # residual variance: variance of detrended (linear) signal
    pred_line = X.mean(axis=1, keepdims=True) + (t - t_mean) * out[:, 2:3]
    out[:, 4] = (X - pred_line).var(axis=1)
    # seasonal phase: position of latest sample within the seasonal cycle
    out[:, 5] = float((L - 1) % stl_period) / float(max(stl_period - 1, 1))
    # last-window max - min (range)
    out[:, 6] = X[:, -last_n:].max(axis=1) - X[:, -last_n:].min(axis=1)
    # autocorrelation at lag = period (if available)
    if L > stl_period + 4:
        a = X[:, stl_period:]
        b = X[:, :-stl_period]
        am = a.mean(axis=1, keepdims=True)
        bm = b.mean(axis=1, keepdims=True)
        num = ((a - am) * (b - bm)).mean(axis=1)
        den = a.std(axis=1) * b.std(axis=1) + 1e-12
        out[:, 7] = num / den
    return out


class _AlphaMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class CHAHybridV4Forecaster(CHAHybridV3Forecaster):
    """v3 + learned per-sample α(x) via a small MLP head.

    All v3 fitting + checkpoint plumbing is inherited; we only override
    the α-selection step (fit) and the prediction blend (predict)."""

    name = "cha_hybrid_v4"

    def __init__(self, horizon, hparams=None, seed=42, device="cuda"):
        super().__init__(horizon=horizon, hparams=hparams, seed=seed, device=device)
        amlp = dict(self.hparams.get("alpha_mlp", {}))
        self._alpha_hidden = int(amlp.get("hidden", 16))
        self._alpha_epochs = int(amlp.get("epochs", 300))
        self._alpha_lr = float(amlp.get("lr", 1e-3))
        self._alpha_wd = float(amlp.get("weight_decay", 1e-4))
        self._alpha_patience = int(amlp.get("patience", 30))
        self.alpha_mlp: _AlphaMLP | None = None
        self._alpha_mlp_diag: dict | None = None

    # --- override α-tuning step ---

    def fit(self, train, val=None, *, train_series=None, val_series=None):
        # Run the v3 fit but disable its α-search; we'll do α(x) instead.
        # The cleanest path is to call v3.fit (it picks scalar α_h on val),
        # then replace α_h with a learned MLP head fitted on val.
        super().fit(
            train,
            val,
            train_series=train_series,
            val_series=val_series,
        )
        if val is None:
            logger.warning(
                "[cha_v4] no val set — learned α(x) skipped, using v3 scalar"
            )
            return self
        logger.info("[cha_v4] training α(x) MLP …")
        t0 = time.perf_counter()
        # Validation residuals from each expert
        decomp_val = self._predict_decomposition(val.X)
        global_val = self.global_model.predict(val)
        # Context features on val.X
        feat = _context_features(val.X, stl_period=self.stl_period)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=self.device)
        dec_t = torch.tensor(decomp_val, dtype=torch.float32, device=self.device)
        glb_t = torch.tensor(global_val, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(val.y, dtype=torch.float32, device=self.device)
        mlp = _AlphaMLP(feat.shape[1], self._alpha_hidden).to(self.device)
        opt = torch.optim.Adam(
            mlp.parameters(), lr=self._alpha_lr, weight_decay=self._alpha_wd
        )
        best_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        history = []
        for ep in range(self._alpha_epochs):
            mlp.train()
            opt.zero_grad()
            a = mlp(feat_t).unsqueeze(-1)
            mix = a * dec_t + (1.0 - a) * glb_t
            loss = ((mix - y_t) ** 2).mean()
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
            if loss.item() < best_loss - 1e-6:
                best_loss = float(loss.item())
                best_state = {
                    k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self._alpha_patience:
                    break
        if best_state is not None:
            mlp.load_state_dict(best_state)
        mlp.eval()
        with torch.no_grad():
            a_val = mlp(feat_t).cpu().numpy()
        self.alpha_mlp = mlp
        self._alpha_mlp_diag = {
            "best_val_mse": best_loss,
            "epochs_trained": int(len(history)),
            "alpha_min": float(np.min(a_val)),
            "alpha_max": float(np.max(a_val)),
            "alpha_mean": float(np.mean(a_val)),
            "alpha_std": float(np.std(a_val)),
            "alpha_hist_first": float(history[0]) if history else None,
            "alpha_hist_last": float(history[-1]) if history else None,
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
        logger.info(
            "[cha_v4] α(x) MLP: best_val_mse=%.4f, α∈[%.2f,%.2f] mean=%.2f, %d epochs",
            best_loss,
            self._alpha_mlp_diag["alpha_min"],
            self._alpha_mlp_diag["alpha_max"],
            self._alpha_mlp_diag["alpha_mean"],
            self._alpha_mlp_diag["epochs_trained"],
        )
        return self

    # --- override prediction blend ---

    def predict(self, windows):
        if not self._fitted:
            raise RuntimeError(f"{self.name}: predict() before fit()")
        decomp_pred = self._predict_decomposition(windows.X)
        global_pred = self.global_model.predict(windows)
        if self.alpha_mlp is None:
            # Fallback to v3 scalar α_h
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

    # --- checkpoint ---

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
            },
            "stl_period": self.stl_period,
            "stl_recon_diag": self._train_recon_diag,
            "lstm_hparams": dict(self.lstm_hparams),
            "global_expert": {
                "type": "chronos_bolt",
                "pretrained": self.hparams.get("_chronos_bolt_hparams", {}).get(
                    "pretrained", "amazon/chronos-bolt-small"
                ),
            },
            "lstm_residual_state_dict": lstm_state,
        }
        _torch.save(ckpt, p)
        meta = {
            k: v
            for k, v in ckpt.items()
            if k not in ("lstm_residual_state_dict", "alpha_mlp_state_dict")
        }
        meta_path = p.with_suffix(p.suffix + ".json")
        with meta_path.open("w") as f:
            json.dump(meta, f, indent=2, default=str)
        return p
