"""Alibaba PAI 2020 GPU cluster trace — job submission rate.

Reference: Weng et al. "MLaaS in the Wild: Workload Analysis and Scheduling
in Large-Scale Heterogeneous GPU Clusters", NSDI 2022.
Repo: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020
Mirror: https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces/

Two months of job-submission events from a production Alibaba PAI cluster
(~6500 GPUs, ~1.05M jobs, July-August 2020). The ``pai_job_table.csv``
schema (no header in the file — provided separately as ``.header``):

    job_name, inst_id, user, status, start_time, end_time

Timestamps are seconds since the start of the trace (relative). We anchor
to a fixed Monday so weekly seasonality lines up with calendar weekdays.

Primary metric: job submission COUNT per hour (driven by ``start_time``).
This is a strong proxy for AI/ML workload arrival rate.
"""
from __future__ import annotations

import logging
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, DatasetSpec
from ._utils import stream_download

logger = logging.getLogger(__name__)


_OSS_BASE = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2020GPUTraces"

# Job table is the primary source — one row per submitted job.
_ARCHIVE = "pai_job_table.tar.gz"
_CSV = "pai_job_table.csv"
_HEADER = "job_name,inst_id,user,status,start_time,end_time".split(",")

# Anchor: a Monday so weekday=0 maps to start of trace. Trace year is 2020.
_TRACE_START = pd.Timestamp("2020-07-06 00:00:00")  # Monday


class AlibabaPAILoader(BaseLoader):
    """Job-submission rate per hour for the 2020 Alibaba PAI GPU trace."""

    def __init__(
        self, raw_dir: str | Path, processed_path: str | Path, resample_to: str = "1h"
    ):
        super().__init__(
            DatasetSpec(
                name="alibaba_pai",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="1s",
                resample_to=resample_to,
                description=(
                    "Alibaba PAI 2020 GPU trace — hourly ML job submission "
                    "rate (~6.5k GPUs, ~1.05M jobs, 2 months)."
                ),
                outlier_clip_factor=None,  # job bursts are real
            )
        )

    def aggregate(self, df_native: pd.DataFrame) -> pd.DataFrame:
        """Event-count semantics: hours with no job submissions are 0, not NaN."""
        if df_native.index.tz is not None:
            df_native = df_native.tz_convert(None)
        if df_native.index.has_duplicates:
            df_native = df_native.groupby(level=0).sum()
        agg = df_native.resample(self.spec.resample_to).sum(min_count=0)
        agg = agg.sort_index()
        return agg

    def download(self) -> None:
        archive = self.spec.raw_dir / _ARCHIVE
        csv = self.spec.raw_dir / _CSV
        if csv.exists() and csv.stat().st_size > 10_000_000:
            logger.info(
                "[alibaba_pai] cached %s (%.1f MB)", csv, csv.stat().st_size / 1e6
            )
            return
        if not archive.exists() or archive.stat().st_size < 10_000_000:
            stream_download(f"{_OSS_BASE}/{_ARCHIVE}", archive)
        # Extract CSV from the tar.gz.
        logger.info("[alibaba_pai] extracting %s", archive)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(self.spec.raw_dir)

    def parse(self) -> pd.DataFrame:
        path = self.spec.raw_dir / _CSV
        if not path.exists():
            raise FileNotFoundError(f"[alibaba_pai] missing {path}")

        # File has no header line — provide the names explicitly.
        df = pd.read_csv(
            path,
            header=None,
            names=_HEADER,
            usecols=["start_time", "status"],
            dtype={"status": "string"},
        )
        # start_time is seconds-since-trace-start; some rows have NaN.
        start_sec = pd.to_numeric(df["start_time"], errors="coerce")
        start_sec = start_sec.dropna()
        # Drop the rare negative or absurd values, if any.
        start_sec = start_sec[(start_sec >= 0) & (start_sec < 365 * 86400)]
        logger.info("[alibaba_pai] %d valid job-submission events", len(start_sec))

        ts = _TRACE_START + pd.to_timedelta(start_sec.values, unit="s")
        per_second = pd.Series(
            np.ones(len(ts), dtype=np.int64), index=pd.DatetimeIndex(ts)
        )
        per_second = per_second.groupby(level=0).sum()
        out = per_second.to_frame(name="value").astype({"value": "float64"})
        out = out.sort_index()
        logger.info(
            "[alibaba_pai] %d unique seconds, range %s -> %s",
            len(out),
            out.index.min(),
            out.index.max(),
        )
        return out
