"""NAB (Numenta Anomaly Benchmark) traffic-relevant loaders.

Two series we use:

  * ``nab_ec2_network_in`` — real AWS CloudWatch EC2 network-in bytes,
    5-minute resolution, ~14 days. Modern (2014) cloud-network workload.
  * ``nab_twitter_volume`` — Twitter @AAPL mention rate, 5-minute
    resolution, ~55 days. Proxy for web-service event-rate load.

Both are public GitHub-raw downloads. Single (timestamp, value) CSV;
mapped through our shared ``BaseLoader`` so the same outlier filter,
aggregation, and parquet cache apply.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, DatasetSpec
from ._utils import stream_download

logger = logging.getLogger(__name__)


_BASE_RAW = "https://raw.githubusercontent.com/numenta/NAB/master/data"


class _NABSingleSeriesLoader(BaseLoader):
    """Common scaffold for NAB CSV series."""

    SRC_PATH: str = ""  # relative path inside NAB/data/
    LOCAL_NAME: str = ""

    def download(self) -> None:
        dest = self.spec.raw_dir / self.LOCAL_NAME
        if not dest.exists():
            stream_download(f"{_BASE_RAW}/{self.SRC_PATH}", dest)
        else:
            logger.info(
                "[%s] cached %s (%.1f KB)",
                self.spec.name,
                dest,
                dest.stat().st_size / 1e3,
            )

    def parse(self) -> pd.DataFrame:
        df = pd.read_csv(self.spec.raw_dir / self.LOCAL_NAME)
        if "timestamp" not in df or "value" not in df:
            raise ValueError(f"unexpected NAB columns: {list(df.columns)}")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp")
        df["value"] = df["value"].astype("float64")
        df = df.sort_index()
        if df.index.has_duplicates:
            df = df.groupby(level=0).mean()
        logger.info(
            "[%s] parsed %d rows, %s -> %s, mean=%.3e",
            self.spec.name,
            len(df),
            df.index.min(),
            df.index.max(),
            float(df["value"].mean()),
        )
        return df


class NABAWSCpuLoader(_NABSingleSeriesLoader):
    """AWS CloudWatch EC2 auto-scaling-group CPU utilisation — 63 days, 5-min.

    Modern (2014) public cloud-infrastructure monitoring trace. Used in
    NAB as a real-anomaly benchmark; for our forecasting study we use
    only its raw value series (CPU %) which is a strong proxy for
    aggregate cloud-workload demand on a virtualised compute fleet.
    """

    SRC_PATH = "realKnownCause/cpu_utilization_asg_misconfiguration.csv"
    LOCAL_NAME = "cpu_utilization_asg_misconfiguration.csv"

    def __init__(self, raw_dir, processed_path, resample_to: str = "1h"):
        super().__init__(
            DatasetSpec(
                name="nab_aws_cpu",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="5min",
                resample_to=resample_to,
                description="NAB AWS CloudWatch EC2 ASG CPU utilisation (5-min, ~63d).",
            )
        )


class NABTwitterAAPLLoader(_NABSingleSeriesLoader):
    SRC_PATH = "realTweets/Twitter_volume_AAPL.csv"
    LOCAL_NAME = "Twitter_volume_AAPL.csv"

    def __init__(self, raw_dir, processed_path, resample_to: str = "1h"):
        super().__init__(
            DatasetSpec(
                name="nab_twitter",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="5min",
                resample_to=resample_to,
                description="NAB Twitter @AAPL mention volume, 5-min, ~55d.",
            )
        )
