"""Forecasting metrics: RMSE, MAE, MAPE, sMAPE, R^2.

All metrics work on flattened arrays. `mape` and `smape` automatically
mask out tiny / zero true values to avoid division blow-up — this is the
standard Hyndman convention for sMAPE and lets us safely apply them to
scaled or unscaled predictions.
"""
from __future__ import annotations

import numpy as np


def _flat(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def rmse(y_true, y_pred) -> float:
    yt, yp = _flat(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _flat(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true, y_pred, eps_frac: float = 1e-3) -> float:
    """Mean Absolute Percentage Error.

    Mask out observations whose |y_true| is below ``eps_frac * median(|y_true|)``
    to avoid pathological blow-ups near zero.
    """
    yt, yp = _flat(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    eps = eps_frac * np.median(np.abs(yt))
    mask = np.abs(yt) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE (Hyndman 2006).  Returns 0..200 percentage."""
    yt, yp = _flat(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100.0)


def r2(y_true, y_pred) -> float:
    """Coefficient of determination."""
    yt, yp = _flat(y_true, y_pred)
    if yt.size < 2:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


_METRICS = {
    "rmse":  rmse,
    "mae":   mae,
    "mape":  mape,
    "smape": smape,
    "r2":    r2,
}


def compute_all(y_true, y_pred) -> dict[str, float]:
    """Compute every standard metric. Returns a dict {name: value}."""
    return {name: fn(y_true, y_pred) for name, fn in _METRICS.items()}
