"""Abilene 5-min traffic-matrix loader (Y. Zhang's Internet2 archive).

Each weekly file ``X??.gz`` holds 12*24*7 = 2016 5-minute traffic matrices,
720 floats per row organised as 144 OD pairs × 5 estimator variants. The
*real* OD volume sits at positions 0, 5, 10, ..., 715. Summing those 144
real ODs gives the total inbound+outbound traffic for that 5-minute bin.

Unit in the source file: ``100 bytes / 5 min`` (the 100 is the packet
sampling rate). We multiply by 100 to recover ``bytes / 5 min`` and then
let the base class resample to the target frequency by summation, yielding
``bytes / target_bin``.

Per the readme weekly start dates are *not* contiguous — there are gaps
between some weeks. We honour those gaps so the resampled hourly series
contains NaNs in those gaps (Phase 2 will interpolate).
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, DatasetSpec
from ._utils import stream_download

logger = logging.getLogger(__name__)


_BASE = "http://www.cs.utexas.edu/~yzhang/research/AbileneTM"

# (filename, first-day-of-week) tuples copied verbatim from the readme.
_WEEK_STARTS: list[tuple[str, str]] = [
    ("X01.gz", "2004-03-01"),
    ("X02.gz", "2004-03-08"),
    ("X03.gz", "2004-04-02"),
    ("X04.gz", "2004-04-09"),
    ("X05.gz", "2004-04-22"),
    ("X06.gz", "2004-05-01"),
    ("X07.gz", "2004-05-08"),
    ("X08.gz", "2004-05-15"),
    ("X09.gz", "2004-05-22"),
    ("X10.gz", "2004-05-29"),
    ("X11.gz", "2004-06-05"),
    ("X12.gz", "2004-06-12"),
    ("X13.gz", "2004-06-19"),
    ("X14.gz", "2004-06-26"),
    ("X15.gz", "2004-07-03"),
    ("X16.gz", "2004-07-10"),
    ("X17.gz", "2004-07-17"),
    ("X18.gz", "2004-07-24"),
    ("X19.gz", "2004-07-31"),
    ("X20.gz", "2004-08-07"),
    ("X21.gz", "2004-08-13"),
    ("X22.gz", "2004-08-21"),
    ("X23.gz", "2004-08-28"),
    ("X24.gz", "2004-09-04"),
]

_ROWS_PER_WEEK = 12 * 24 * 7         # 2016
_VALUES_PER_ROW = 720                # 144 OD × 5 variants
_OD_PAIRS = 144
_VARIANTS_PER_OD = 5
_SAMPLING_RATE = 100                 # value unit is "100 bytes / 5 min"


class AbileneLoader(BaseLoader):
    def __init__(self, raw_dir: str | Path, processed_path: str | Path, resample_to: str = "1h"):
        super().__init__(
            DatasetSpec(
                name="abilene",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="5min",
                resample_to=resample_to,
                description="Abilene 5-min OD bytes summed across 144 OD pairs (real-OD column).",
            )
        )

    def download(self) -> None:
        for fname, _ in _WEEK_STARTS:
            dest = self.spec.raw_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            stream_download(f"{_BASE}/{fname}", dest)

    def parse(self) -> pd.DataFrame:
        # real-OD columns: positions 0, 5, 10, ..., 715 within each row
        real_cols = np.arange(_OD_PAIRS) * _VARIANTS_PER_OD  # shape (144,)

        all_idx = []
        all_vals = []

        for fname, week_start in _WEEK_STARTS:
            path = self.spec.raw_dir / fname
            with gzip.open(path, "rt") as f:
                # parse the whole week as a 2D matrix
                arr = np.loadtxt(f, dtype=np.float64)
            if arr.shape != (_ROWS_PER_WEEK, _VALUES_PER_ROW):
                raise ValueError(
                    f"[abilene] unexpected shape for {fname}: {arr.shape}"
                )
            real_od = arr[:, real_cols]               # (2016, 144)
            totals_per_5min = real_od.sum(axis=1)     # (2016,) — sum across OD pairs
            # convert "100 bytes / 5 min" to "bytes / 5 min"
            totals_per_5min *= _SAMPLING_RATE

            idx = pd.date_range(start=week_start, periods=_ROWS_PER_WEEK, freq="5min")
            all_idx.append(idx)
            all_vals.append(totals_per_5min)
            logger.info("[abilene] parsed %s (%s, mean=%.3e)",
                        fname, week_start, totals_per_5min.mean())

        idx = pd.DatetimeIndex(np.concatenate(all_idx), name="timestamp")
        vals = np.concatenate(all_vals)
        out = pd.DataFrame({"value": vals}, index=idx).sort_index()
        # Just in case any duplicate index from contiguous weeks — average.
        if out.index.has_duplicates:
            out = out.groupby(level=0).mean()
        logger.info("[abilene] %d 5-min rows, %s -> %s",
                    len(out), out.index.min(), out.index.max())
        return out
