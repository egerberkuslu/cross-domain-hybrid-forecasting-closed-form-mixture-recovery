"""Base dataset loader.

Every concrete loader returns a single pandas Series-style DataFrame:

    index:   DatetimeIndex, monotonic-increasing, tz-naive
    column:  'value'  (float)  — aggregate total network traffic per bucket

All loaders cache the cleaned aggregate to ``data/processed/<name>.parquet``
so subsequent phases re-use a stable, common format.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DatasetSpec:
    name: str
    raw_dir: Path
    processed_path: Path
    native_freq: str          # e.g. '5min', '10min', '15min'
    resample_to: str          # e.g. '1h'
    description: str
    # Mark values whose magnitude exceeds `outlier_clip_factor * median` as
    # NaN — these are physically implausible measurement glitches well above
    # the backbone's known peak capacity (e.g. a single 5-min Abilene row
    # at 5.3 PB ⇒ 14 Tbps on a 10 Gbps link). Phase 2 interpolation fills
    # the resulting NaNs.  Set to None to disable.
    outlier_clip_factor: float | None = 100.0


class BaseLoader(ABC):
    """Skeleton that defines the common (download → parse → save) workflow."""

    def __init__(self, spec: DatasetSpec):
        self.spec = spec
        self.spec.raw_dir.mkdir(parents=True, exist_ok=True)
        self.spec.processed_path.parent.mkdir(parents=True, exist_ok=True)

    # --- concrete loaders implement these two ---

    @abstractmethod
    def download(self) -> None:
        """Make sure raw files are present in ``self.spec.raw_dir``."""

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """Return the aggregate-total series at native frequency.

        Returned frame must have a DatetimeIndex and a single column 'value'
        (float). The index must be monotonic-increasing but may have gaps.
        """

    # --- common pipeline ---

    def aggregate(self, df_native: pd.DataFrame) -> pd.DataFrame:
        """Resample the native-frequency series to ``self.spec.resample_to``."""
        if df_native.index.tz is not None:
            df_native = df_native.tz_convert(None)
        # Drop duplicates by averaging — keeps the series strictly monotonic.
        if df_native.index.has_duplicates:
            df_native = df_native.groupby(level=0).mean()
        agg = df_native.resample(self.spec.resample_to).sum(min_count=1)
        agg = agg.sort_index()
        return agg

    def flag_extreme_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Replace physically-implausible outliers with NaN.

        We use a robust criterion (median × factor) rather than mean/std so a
        single outlier cannot mask itself. Returns (df, n_flagged).
        """
        factor = self.spec.outlier_clip_factor
        if factor is None or factor <= 0:
            return df, 0
        med = float(df["value"].median())
        if not np.isfinite(med) or med <= 0:
            return df, 0
        threshold = med * factor
        mask = df["value"] > threshold
        n = int(mask.sum())
        if n > 0:
            df = df.copy()
            df.loc[mask, "value"] = np.nan
            logger.warning(
                "[%s] flagged %d extreme outliers (> %g × median = %.3e) as NaN; "
                "Phase 2 interpolation will fill them.",
                self.spec.name, n, factor, threshold,
            )
        return df, n

    def run(self, force: bool = False) -> pd.DataFrame:
        """Download (if needed), parse, aggregate, validate, cache."""
        if self.spec.processed_path.exists() and not force:
            logger.info("[%s] cached processed file: %s", self.spec.name,
                        self.spec.processed_path)
            return pd.read_parquet(self.spec.processed_path)

        self.download()
        native = self.parse()
        self._validate_native(native)

        agg = self.aggregate(native)
        agg, n_outliers = self.flag_extreme_outliers(agg)
        # store the outlier count as a parquet metadata attr for the report
        agg.attrs["n_extreme_outliers_flagged"] = n_outliers
        self._validate_aggregate(agg)

        agg.to_parquet(self.spec.processed_path)
        logger.info("[%s] wrote %s (%d rows, %s — %s)",
                    self.spec.name,
                    self.spec.processed_path,
                    len(agg),
                    agg.index.min(),
                    agg.index.max())
        return agg

    # --- validation ---

    @staticmethod
    def _validate_native(df: pd.DataFrame) -> None:
        if "value" not in df.columns:
            raise ValueError(f"Native parse must have a 'value' column, got {df.columns}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Native parse must use a DatetimeIndex")
        if df["value"].dropna().empty:
            raise ValueError("Native parse returned an empty series")

    @staticmethod
    def _validate_aggregate(df: pd.DataFrame) -> None:
        if not df.index.is_monotonic_increasing:
            raise ValueError("Aggregate index must be monotonic increasing")
        if df["value"].dropna().empty:
            raise ValueError("Aggregate series is fully empty")
