"""Azure LLM Inference Trace 2024, aggregated at 5-minute granularity.

Companion to :mod:`azure_llm_2024` (which aggregates at 1-hour). The
hourly aggregation yields only $\\sim$216 samples, below the LSTM
residual sub-module's training-data floor. This 5-minute variant yields
$\\sim$2592 samples (9 days $\\times$ 24h $\\times$ 12), enough to train
the decomposition path's LSTM residual head.

Both loaders consume the same raw files in ``data/raw/azure_llm_2024/``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .base import BaseLoader, DatasetSpec

logger = logging.getLogger(__name__)


class AzureLLM2024_5mLoader(BaseLoader):
    """Aggregate Azure OpenAI conv+code request counts to 5-minute bins."""

    def __init__(
        self, raw_dir: str | Path, processed_path: str | Path, resample_to: str = "5min"
    ):
        super().__init__(
            DatasetSpec(
                name="azure_llm_2024_5m",
                raw_dir=Path(raw_dir),
                processed_path=Path(processed_path),
                native_freq="1s",
                resample_to=resample_to,
                description=(
                    "Azure OpenAI conv+code inference traces (May 2024), "
                    "aggregated at 5-minute granularity. 2592 samples."
                ),
                outlier_clip_factor=None,  # arrival counts; no clipping
            )
        )

    def download(self) -> None:
        # Reuses files downloaded by AzureLLM2024Loader.
        from .azure_llm_2024 import _FALLBACK_FILES, _PREFERRED_FILES, stream_download

        for fname, url in _PREFERRED_FILES + _FALLBACK_FILES:
            target = self.spec.raw_dir / fname
            if target.exists():
                continue
            try:
                stream_download(url, target)
            except Exception as e:
                logger.warning("[%s] download %s failed: %s", self.spec.name, fname, e)

    def parse(self) -> pd.DataFrame:
        files = [
            self.spec.raw_dir / "AzureLLMInferenceTrace_conv_1week.csv",
            self.spec.raw_dir / "AzureLLMInferenceTrace_code_1week.csv",
        ]
        files = [f for f in files if f.exists()]
        if not files:
            raise FileNotFoundError(
                f"No raw Azure LLM 2024 traces in {self.spec.raw_dir}"
            )
        counts = []
        for f in files:
            df = pd.read_csv(f, usecols=["TIMESTAMP"])
            df["TIMESTAMP"] = pd.to_datetime(
                df["TIMESTAMP"], format="ISO8601", utc=True
            ).dt.tz_localize(None)
            s = df.groupby(pd.Grouper(key="TIMESTAMP", freq="5min")).size()
            counts.append(s)
        combined = pd.concat(counts, axis=1).fillna(0).sum(axis=1).astype("float32")
        combined.name = "value"
        return combined.to_frame()
