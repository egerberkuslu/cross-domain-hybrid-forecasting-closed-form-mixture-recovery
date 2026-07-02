"""CESNET-TimeSeries24 loader.

We aggregate the *institutions* level at the *1-hour* aggregation (already
provided in the Zenodo bundle) by summing the bytes-transmitted column
(`n_bytes`) across all institutions per timestamp.

The resulting series therefore represents *total CESNET3 backbone bytes
transmitted per hour* — the natural "aggregate total traffic" of the network.

Source: Koumar et al. 2025 (Scientific Data 12:338),
DOI 10.5281/zenodo.13382427.
"""
from __future__ import annotations

import io
import logging
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, DatasetSpec
from ._utils import stream_download

logger = logging.getLogger(__name__)


ZENODO_BASE = "https://zenodo.org/api/records/13382427/files"
INSTITUTIONS_URL = f"{ZENODO_BASE}/institutions.tar.gz/content"
TIMES_URL = f"{ZENODO_BASE}/times.tar.gz/content"

INSTITUTIONS_FILENAME = "institutions.tar.gz"
TIMES_FILENAME = "times.tar.gz"


class CesnetLoader(BaseLoader):
    """Loader for the CESNET-TimeSeries24 dataset.

    Aggregation strategy:
      * use the pre-computed 1-hour aggregation
      * sum `n_bytes` across all institutions per timestamp
      * timestamps come from `times/times_1_hour.csv`
    """

    def __init__(self, raw_dir: str | Path, processed_path: str | Path, resample_to: str = "1h"):
        super().__init__(
            DatasetSpec(
                name="cesnet",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="1h",     # we are already using the 1-hour aggregation
                resample_to=resample_to,
                description="CESNET-TimeSeries24 institution-level bytes summed network-wide.",
            )
        )

    def download(self) -> None:
        inst = self.spec.raw_dir / INSTITUTIONS_FILENAME
        times = self.spec.raw_dir / TIMES_FILENAME
        if not inst.exists():
            stream_download(INSTITUTIONS_URL, inst)
        else:
            logger.info("[cesnet] cached %s (%.1f MB)", inst, inst.stat().st_size / 1e6)
        if not times.exists():
            stream_download(TIMES_URL, times)
        else:
            logger.info("[cesnet] cached %s (%.1f KB)", times, times.stat().st_size / 1e3)

    def parse(self) -> pd.DataFrame:
        inst_tar = self.spec.raw_dir / INSTITUTIONS_FILENAME
        times_tar = self.spec.raw_dir / TIMES_FILENAME

        # -- timestamps --
        with tarfile.open(times_tar, "r:gz") as tar:
            f = tar.extractfile("times/times_1_hour.csv")
            times_df = pd.read_csv(io.BytesIO(f.read()))
        times_df["time"] = pd.to_datetime(times_df["time"], utc=True).dt.tz_convert(None)
        times_df = times_df.set_index("id_time")["time"]
        logger.info("[cesnet] %d hourly timestamps loaded (%s -> %s)",
                    len(times_df), times_df.iloc[0], times_df.iloc[-1])

        # -- per-institution n_bytes summed --
        with tarfile.open(inst_tar, "r:gz") as tar:
            members = [
                m for m in tar.getmembers()
                if m.name.startswith("institutions/agg_1_hour/") and m.name.endswith(".csv")
            ]
            logger.info("[cesnet] %d institution CSVs found in agg_1_hour", len(members))
            running = None
            n_seen = 0
            for m in members:
                f = tar.extractfile(m)
                df = pd.read_csv(f, usecols=["id_time", "n_bytes"])
                # use float to allow summation across many institutions safely
                df["n_bytes"] = df["n_bytes"].astype("float64")
                df = df.set_index("id_time")["n_bytes"]
                if running is None:
                    running = df.copy()
                else:
                    running = running.add(df, fill_value=0.0)
                n_seen += 1
                if n_seen % 50 == 0:
                    logger.info("[cesnet] aggregated %d/%d institutions",
                                n_seen, len(members))
            logger.info("[cesnet] aggregated all %d institutions", n_seen)

        # -- join with timestamps --
        out = pd.DataFrame({"value": running}).reset_index()
        out = out.merge(times_df.rename("timestamp"), left_on="id_time", right_index=True, how="inner")
        out = out.set_index("timestamp").sort_index()[["value"]]
        # make tz-naive (already converted) and float
        out.index = pd.DatetimeIndex(out.index, name="timestamp")
        out["value"] = out["value"].astype("float64")
        logger.info("[cesnet] parsed series: %d rows, %s -> %s, sum_n_bytes mean=%.3e",
                    len(out), out.index.min(), out.index.max(), float(out["value"].mean()))
        return out
