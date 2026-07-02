"""Azure LLM Inference Trace 2024 (DynamoLLM, HPCA 2025).

Reference: Stojkovic, Zhang, Goiri, Torrellas, Choukse. "DynamoLLM:
Designing LLM Inference Clusters for Performance and Energy Efficiency",
HPCA 2025. https://arxiv.org/abs/2408.00741
Data: https://github.com/Azure/AzurePublicDataset (AzureLLMInferenceDataset2024)

The 2024 release ships two one-week traces collected May 10-19 2024 from
production Azure OpenAI clusters:

  * AzureLLMInferenceTrace_conv_1week.csv  — conversation (chat) requests
  * AzureLLMInferenceTrace_code_1week.csv  — code-completion requests

Schema (both files):

    TIMESTAMP             # absolute wall-clock, microsecond resolution
    ContextTokens         # prompt tokens
    GeneratedTokens       # response tokens

Primary metric here: request COUNT per hour, summed across conv+code
(both are the same Azure OpenAI inference workload, separated only by
endpoint type). A secondary loader could emit token-rate variants.

The repo also publishes 1-hour ``AzureLLMInferenceTrace_{conv,code}.csv``
samples in ``/data/`` — these are kept as a fallback so the loader still
works when the multi-hundred-MB 1-week files are absent.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseLoader, DatasetSpec
from ._utils import stream_download

logger = logging.getLogger(__name__)


_BLOB_BASE = (
    "https://azurepublicdatasettraces.blob.core.windows.net/" "azurellminfererencetrace"
)
_GH_RAW = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data"

# (local-name, url) — order: prefer 1-week files, fall back to 1-hour samples.
_PREFERRED_FILES: list[tuple[str, str]] = [
    (
        "AzureLLMInferenceTrace_conv_1week.csv",
        f"{_BLOB_BASE}/AzureLLMInferenceTrace_conv_1week.csv",
    ),
    (
        "AzureLLMInferenceTrace_code_1week.csv",
        f"{_BLOB_BASE}/AzureLLMInferenceTrace_code_1week.csv",
    ),
]
_FALLBACK_FILES: list[tuple[str, str]] = [
    ("AzureLLMInferenceTrace_conv.csv", f"{_GH_RAW}/AzureLLMInferenceTrace_conv.csv"),
    ("AzureLLMInferenceTrace_code.csv", f"{_GH_RAW}/AzureLLMInferenceTrace_code.csv"),
]


class AzureLLM2024Loader(BaseLoader):
    """Aggregate Azure OpenAI conv+code request counts to hourly."""

    def __init__(
        self, raw_dir: str | Path, processed_path: str | Path, resample_to: str = "1h"
    ):
        super().__init__(
            DatasetSpec(
                name="azure_llm_2024",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="1s",
                resample_to=resample_to,
                description=(
                    "Azure OpenAI conv+code inference traces (May 2024), "
                    "hourly request count."
                ),
                outlier_clip_factor=None,  # keep production bursts intact
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
        # Attempt the 1-week files first. If a download fails or the file
        # is absent, the parse step will fall back to whatever is on disk.
        for fname, url in _PREFERRED_FILES:
            dest = self.spec.raw_dir / fname
            if dest.exists() and dest.stat().st_size > 1_000_000:
                logger.info(
                    "[azure_llm_2024] cached %s (%.1f MB)",
                    dest,
                    dest.stat().st_size / 1e6,
                )
                continue
            try:
                stream_download(url, dest)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[azure_llm_2024] could not fetch %s (%s); "
                    "loader will fall back to /data sample if available.",
                    url,
                    e,
                )
        # Always grab the small fallback samples too — they are tiny.
        for fname, url in _FALLBACK_FILES:
            dest = self.spec.raw_dir / fname
            if dest.exists() and dest.stat().st_size > 0:
                continue
            try:
                stream_download(url, dest)
            except Exception as e:  # noqa: BLE001
                logger.warning("[azure_llm_2024] fallback %s failed: %s", url, e)

    def parse(self) -> pd.DataFrame:
        # Prefer the long 1-week files; if both are missing, use the 1-hour
        # GitHub samples so the pipeline still runs on a fresh checkout.
        files = []
        for fname, _ in _PREFERRED_FILES:
            p = self.spec.raw_dir / fname
            if p.exists() and p.stat().st_size > 1_000_000:
                files.append(p)
        if not files:
            for fname, _ in _FALLBACK_FILES:
                p = self.spec.raw_dir / fname
                if p.exists() and p.stat().st_size > 0:
                    files.append(p)
            logger.warning(
                "[azure_llm_2024] 1-week files missing — using 1-hour samples; "
                "the resulting series will only span ~1 hour."
            )
        if not files:
            raise RuntimeError(
                "[azure_llm_2024] no input files in %s" % self.spec.raw_dir
            )

        frames = []
        for path in files:
            df = pd.read_csv(path, usecols=["TIMESTAMP"])
            ts = pd.to_datetime(df["TIMESTAMP"], errors="coerce").dropna()
            frames.append(ts)
            logger.info(
                "[azure_llm_2024] %s: %d rows, %s..%s",
                path.name,
                len(ts),
                ts.min(),
                ts.max(),
            )

        all_ts = pd.concat(frames, ignore_index=True).sort_values()
        # Floor to seconds so coincident requests collapse cleanly.
        idx = pd.DatetimeIndex(all_ts.dt.floor("s").values)
        per_second = pd.Series(np.ones(len(idx), dtype=np.int64), index=idx)
        per_second = per_second.groupby(level=0).sum()
        out = per_second.to_frame(name="value").astype({"value": "float64"})
        out = out.sort_index()
        logger.info(
            "[azure_llm_2024] aggregated to %d unique seconds, range %s -> %s",
            len(out),
            out.index.min(),
            out.index.max(),
        )
        return out
