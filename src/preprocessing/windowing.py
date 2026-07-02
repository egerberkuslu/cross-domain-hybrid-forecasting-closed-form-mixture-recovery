"""Sliding-window supervised sample construction.

For a series ``x`` of length ``N`` with lookback ``L`` and horizon ``h``,
a window anchored at index ``i`` (0-indexed) gives:

    X[i] = x[i : i + L]                  shape (L,)
    y[i] = x[i + L : i + L + h]          shape (h,)
    target_ts[i] = timestamps[i + L : i + L + h]

A window is "assigned" to a split based on the **last target timestamp**
``timestamps[i + L + h - 1]``. This means a test-set window may legally
use lookback values that extend into the val/train period — that mirrors
real online-forecasting deployment where past data is always available.

Windows whose lookback OR target contains any NaN are skipped, so the
returned arrays are guaranteed NaN-/inf-free.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WindowSet:
    """One (X, y, target-time, split-name) bundle for a single horizon."""

    X: np.ndarray                     # (n, L)
    y: np.ndarray                     # (n, h)
    target_times: pd.DatetimeIndex    # last-target timestamps (len n)
    split: str                        # 'train' | 'val' | 'test'
    horizon: int
    lookback: int

    def __post_init__(self) -> None:
        assert self.X.ndim == 2 and self.y.ndim == 2
        assert self.X.shape[0] == self.y.shape[0] == len(self.target_times)
        assert self.X.shape[1] == self.lookback
        assert self.y.shape[1] == self.horizon
        assert not np.isnan(self.X).any(), f"{self.split}/h={self.horizon}: NaN in X"
        assert not np.isnan(self.y).any(), f"{self.split}/h={self.horizon}: NaN in y"
        assert not np.isinf(self.X).any(), f"{self.split}/h={self.horizon}: inf in X"
        assert not np.isinf(self.y).any(), f"{self.split}/h={self.horizon}: inf in y"


def _sliding(arr: np.ndarray, win: int) -> np.ndarray:
    """Stride-based sliding view: shape (N - win + 1, win)."""
    if arr.size < win:
        return np.empty((0, win), dtype=arr.dtype)
    return np.lib.stride_tricks.sliding_window_view(arr, win)


def make_sliding_windows(
    series: pd.Series,
    split_ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    lookback: int,
    horizon: int,
) -> dict[str, WindowSet]:
    """Build per-split window sets for one (lookback, horizon) configuration.

    ``series`` MUST already be scaled and aligned on a complete grid (NaNs
    allowed but those windows are dropped).
    """
    if series.index.has_duplicates:
        raise ValueError("series index has duplicates")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series index must be monotonic increasing")

    arr = series.to_numpy(dtype=np.float64)
    times = series.index
    n = arr.size
    win = lookback + horizon
    if n < win:
        raise ValueError(f"series too short ({n}) for lookback+horizon={win}")

    # Whole-series sliding window of length L+h.
    full = _sliding(arr, win)                # (n - win + 1, L + h)
    last_target_ts = times[win - 1: n]       # shape (n - win + 1,)

    # Drop any windows that contain NaN anywhere
    finite_mask = np.isfinite(full).all(axis=1)

    # Split assignment based on last-target timestamp
    out: dict[str, WindowSet] = {}
    for name, (lo, hi) in split_ranges.items():
        in_range = (last_target_ts >= lo) & (last_target_ts <= hi)
        sel = finite_mask & in_range
        sel_idx = np.where(sel)[0]
        windows = full[sel_idx]
        X = windows[:, :lookback].astype(np.float32, copy=False)
        y = windows[:, lookback:].astype(np.float32, copy=False)
        ws = WindowSet(
            X=X,
            y=y,
            target_times=last_target_ts[sel_idx],
            split=name,
            horizon=horizon,
            lookback=lookback,
        )
        out[name] = ws
        logger.info(
            "[window] split=%s h=%d L=%d -> X=%s y=%s "
            "(target range %s .. %s)",
            name, horizon, lookback, ws.X.shape, ws.y.shape,
            ws.target_times.min() if len(ws.target_times) else "n/a",
            ws.target_times.max() if len(ws.target_times) else "n/a",
        )
    return out
