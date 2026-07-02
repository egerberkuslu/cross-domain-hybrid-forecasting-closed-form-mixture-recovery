"""Missing-value handling for the aggregated series.

Strategy:
  * Reindex to a complete grid at the dataset's target frequency so that
    *every* expected timestamp is present (the loader may have skipped
    buckets that had no source rows).
  * Time-based linear interpolation across short gaps (``max_gap_steps``,
    default 24 = one day at hourly resolution).
  * Longer gaps are left as NaN — interpolating across multi-week silence
    (e.g. Abilene's 25-day gaps between weekly capture files) would
    fabricate data. The windowing step later skips windows that contain
    NaN, so the model arrays are guaranteed NaN-free.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class InterpReport:
    n_total: int
    n_native_nans: int        # NaNs inherited from the loader (incl. outliers)
    n_filled: int             # NaNs filled by interpolation
    n_remaining: int          # NaNs left after interpolation (long gaps)
    max_gap_steps: int        # the cap used


def handle_missing(
    df: pd.DataFrame,
    freq: str,
    max_gap_steps: int = 24,
    method: str = "time",
) -> tuple[pd.DataFrame, InterpReport]:
    """Reindex onto a complete grid + interpolate gaps up to ``max_gap_steps``.

    Returns the cleaned frame plus an :class:`InterpReport` for logging.
    """
    if "value" not in df.columns:
        raise ValueError("Expected a 'value' column on the input frame.")

    # 1) build the complete grid
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    reindexed = df.reindex(full_idx)
    reindexed.index.name = df.index.name or "timestamp"

    n_total = len(reindexed)
    n_native_nans = int(reindexed["value"].isna().sum())

    # 2) interpolate only across short gaps
    series = reindexed["value"]
    if method == "time":
        interp_full = series.interpolate(method="time", limit_direction="both")
    else:
        interp_full = series.interpolate(method=method, limit_direction="both")

    # blend: replace NaNs whose CONTIGUOUS gap length is <= max_gap_steps
    isna = series.isna()
    if isna.any():
        # gap-id groups: each contiguous NaN run gets an integer id
        group_id = (~isna).cumsum()
        gap_sizes = isna.groupby(group_id).transform("sum")
        small_gap = isna & (gap_sizes <= max_gap_steps)
        out = series.where(~small_gap, interp_full)
    else:
        out = series

    n_remaining = int(out.isna().sum())
    n_filled = n_native_nans - n_remaining

    cleaned = pd.DataFrame({"value": out}, index=reindexed.index)
    rep = InterpReport(
        n_total=n_total,
        n_native_nans=n_native_nans,
        n_filled=n_filled,
        n_remaining=n_remaining,
        max_gap_steps=max_gap_steps,
    )
    logger.info(
        "[interp] grid=%d, native_nans=%d, filled=%d (gaps<=%d), remaining=%d",
        n_total, n_native_nans, n_filled, max_gap_steps, n_remaining,
    )
    return cleaned, rep
