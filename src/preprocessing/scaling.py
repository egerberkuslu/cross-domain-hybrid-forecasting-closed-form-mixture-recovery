"""Per-dataset scaler that is fit on the TRAINING split only.

The same fitted scaler is then applied to train / val / test, so no
information from the future leaks into the model's input distribution.
The scaler is serialised with joblib so the inverse transform can be
recovered when reporting metrics in original units.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


_SCALERS: dict[str, type] = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
}


def _build_scaler(name: str):
    if name not in _SCALERS:
        raise KeyError(f"Unknown scaler '{name}'. Choices: {sorted(_SCALERS)}")
    return _SCALERS[name]()


def fit_scaler_on_train(
    train: pd.DataFrame,
    scaler_name: str = "standard",
    save_to: str | Path | None = None,
):
    """Fit a 1-D scaler on train['value'] (NaNs dropped); optionally save.

    Returns the fitted scaler (a scikit-learn estimator).
    """
    train_values = train["value"].dropna().to_numpy(dtype=np.float64).reshape(-1, 1)
    if train_values.size == 0:
        raise ValueError("Training split is empty after dropping NaNs.")
    sc = _build_scaler(scaler_name)
    sc.fit(train_values)
    logger.info(
        "[scaler] fit %s on TRAIN only: n=%d, mean=%.3e, std=%.3e, min=%.3e, max=%.3e",
        scaler_name, train_values.size,
        float(train_values.mean()), float(train_values.std()),
        float(train_values.min()), float(train_values.max()),
    )
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(sc, save_to)
        logger.info("[scaler] saved to %s", save_to)
    return sc


def transform(sc, df: pd.DataFrame) -> pd.DataFrame:
    """Apply a fitted scaler to a frame with a 'value' column.

    NaNs are preserved (we do not impute here — Phase 2 already interpolated
    short gaps). Returns a new frame with the scaled column.
    """
    arr = df["value"].to_numpy(dtype=np.float64)
    mask = np.isnan(arr)
    out = np.empty_like(arr)
    out[mask] = np.nan
    if (~mask).any():
        valid = arr[~mask].reshape(-1, 1)
        out[~mask] = sc.transform(valid).ravel()
    return pd.DataFrame({"value": out}, index=df.index)
