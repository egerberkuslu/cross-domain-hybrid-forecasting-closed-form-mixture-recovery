"""BurstGPT — Azure OpenAI ChatGPT / GPT-4 inference request trace.

Reference: Wang et al. "BurstGPT: A Real-world Workload Dataset to Optimize
LLM Serving Systems", https://github.com/HPMLL/BurstGPT.
Mirror: https://huggingface.co/datasets/lzzmm/BurstGPT.

The published dataset is ~213 days of Azure OpenAI conversation/API logs
(~10.31M requests). Each row carries:

    Timestamp, Model, Request tokens, Response tokens, Total tokens, Log Type

The ``Timestamp`` is **seconds since trace start** (not an absolute clock).
We anchor the trace to an arbitrary but fixed start date so the resulting
DatetimeIndex is monotonic-increasing and hourly resampling produces a
clean ``requests/hour`` aggregate.

We bundle the two public CSVs (``BurstGPT_1.csv`` covering days 0..~61 and
``BurstGPT_without_fails_2.csv`` covering days ~61..~121) into a single
contiguous stream because their timestamps are issued from the same monotonic
clock (the second file's timestamps continue from the first's).

Primary metric: request COUNT per hour. Token-rate variants can be derived
from the raw CSV by re-running with a different aggregator.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, DatasetSpec
from ._utils import stream_download

logger = logging.getLogger(__name__)


# HuggingFace mirror — the GitHub repo points at the same files.
_HF_BASE = "https://huggingface.co/datasets/lzzmm/BurstGPT/resolve/main/data"

# Files we attempt to assemble. Order matters — timestamps are continuous.
_FILES: list[tuple[str, str]] = [
    ("BurstGPT_1.csv", f"{_HF_BASE}/BurstGPT_1.csv"),
    ("BurstGPT_without_fails_2.csv", f"{_HF_BASE}/BurstGPT_without_fails_2.csv"),
]

# Arbitrary anchor — paper uses no absolute calendar, but a fixed start lets
# weekly seasonality (lag-168) line up with day-of-week if anyone correlates
# against a calendar. We pick a Monday so weekday=0 == start of trace.
_TRACE_START = pd.Timestamp("2023-01-02 00:00:00")  # Monday


class BurstGPTLoader(BaseLoader):
    """Aggregate request count per hour across BurstGPT_{1,2}.csv."""

    def __init__(
        self, raw_dir: str | Path, processed_path: str | Path, resample_to: str = "1h"
    ):
        super().__init__(
            DatasetSpec(
                name="burstgpt",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="1s",
                resample_to=resample_to,
                description=(
                    "BurstGPT Azure-OpenAI ChatGPT/GPT-4 request trace — "
                    "hourly request count over ~120 days."
                ),
                # LLM request rate has occasional bursts; disable the median*100
                # filter — bursts are exactly what we want to forecast.
                outlier_clip_factor=None,
            )
        )

    def aggregate(self, df_native: pd.DataFrame) -> pd.DataFrame:
        """Event-count semantics: hours with no requests are 0, not NaN."""
        if df_native.index.tz is not None:
            df_native = df_native.tz_convert(None)
        if df_native.index.has_duplicates:
            df_native = df_native.groupby(level=0).sum()
        agg = df_native.resample(self.spec.resample_to).sum(min_count=0)
        agg = agg.sort_index()
        return agg

    def download(self) -> None:
        for fname, url in _FILES:
            dest = self.spec.raw_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                logger.info(
                    "[burstgpt] cached %s (%.1f MB)", dest, dest.stat().st_size / 1e6
                )
                continue
            stream_download(url, dest)

    def parse(self) -> pd.DataFrame:
        frames = []
        for fname, _ in _FILES:
            path = self.spec.raw_dir / fname
            if not path.exists():
                logger.warning("[burstgpt] missing %s, skipping", path)
                continue
            df = pd.read_csv(path, usecols=["Timestamp"])
            # Timestamps come in either int or float seconds-since-start.
            ts_sec = pd.to_numeric(df["Timestamp"], errors="coerce")
            ts_sec = ts_sec.dropna()
            frames.append(ts_sec)
            logger.info(
                "[burstgpt] %s: %d rows, %.0f..%.0f s",
                fname,
                len(ts_sec),
                ts_sec.min(),
                ts_sec.max(),
            )
        if not frames:
            raise RuntimeError("[burstgpt] no input files found")
        all_seconds = pd.concat(frames, ignore_index=True).sort_values()

        # Convert seconds-since-start to absolute timestamps.
        ts = _TRACE_START + pd.to_timedelta(all_seconds.values, unit="s")
        # One row per request — aggregate to per-second counts so the BaseLoader
        # resample-by-sum step yields requests/hour exactly.
        per_second = pd.Series(
            np.ones(len(ts), dtype=np.int64), index=pd.DatetimeIndex(ts)
        )
        # Many requests share the same second — collapse via sum.
        per_second = per_second.groupby(level=0).sum()
        out = per_second.to_frame(name="value").astype({"value": "float64"})
        out = out.sort_index()
        logger.info(
            "[burstgpt] %d unique seconds, range %s -> %s, total requests=%d",
            len(out),
            out.index.min(),
            out.index.max(),
            int(out["value"].sum()),
        )
        return out
