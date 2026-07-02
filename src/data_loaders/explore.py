"""Per-dataset exploration / quality-report generator (Phase 1 deliverable).

Outputs:
    figures/<name>_overview.png    — full series with a zoomed inset
    results/data_summary.csv       — one row per dataset
    results/data_summary.json      — same content, machine-readable
    logs/data_explore_<ts>.log     — full log (handled by setup_logging)
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataReport:
    name: str
    n_rows: int
    start: str
    end: str
    inferred_freq: str
    expected_freq: str
    span_days: float
    missing_pct: float
    n_duplicates: int
    monotonic: bool
    n_unique: int
    n_extreme_outliers_flagged: int
    n_missing_value_nan: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    p01: float
    p99: float
    skew: float
    kurt: float

    def to_row(self) -> dict:
        return asdict(self)


def report_one(name: str, df: pd.DataFrame, expected_freq: str) -> DataReport:
    series = df["value"]
    n = len(series)
    start, end = series.index.min(), series.index.max()
    span = (end - start).total_seconds() / 86400.0

    # missing % over full expected grid
    full_idx = pd.date_range(start=start, end=end, freq=expected_freq)
    aligned = series.reindex(full_idx)
    missing = int(aligned.isna().sum())
    missing_pct = 100.0 * missing / len(full_idx)

    inferred = pd.infer_freq(series.index) or "?"

    rep = DataReport(
        name=name,
        n_rows=int(n),
        start=str(start),
        end=str(end),
        inferred_freq=str(inferred),
        expected_freq=str(expected_freq),
        span_days=float(round(span, 3)),
        missing_pct=float(round(missing_pct, 4)),
        n_duplicates=int(series.index.duplicated().sum()),
        monotonic=bool(series.index.is_monotonic_increasing),
        n_unique=int(series.nunique()),
        n_extreme_outliers_flagged=int(df.attrs.get("n_extreme_outliers_flagged", 0)),
        n_missing_value_nan=int(series.isna().sum()),
        min=float(series.min()),
        max=float(series.max()),
        mean=float(series.mean()),
        median=float(series.median()),
        std=float(series.std()),
        p01=float(series.quantile(0.01)),
        p99=float(series.quantile(0.99)),
        skew=float(series.skew()),
        kurt=float(series.kurt()),
    )
    return rep


def plot_overview(name: str, df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [2, 1]})
    ax_full, ax_zoom = axes

    df["value"].plot(ax=ax_full, lw=0.6, color="#1f77b4")
    ax_full.set_title(f"{name} — full aggregate series ({len(df)} hourly bins)")
    ax_full.set_ylabel("value")
    ax_full.grid(True, alpha=0.3)
    ax_full.xaxis.set_major_locator(mdates.AutoDateLocator())

    # zoom: first 14 days of data (or whatever exists)
    zoom_end = df.index.min() + pd.Timedelta("14D")
    zoom = df.loc[df.index <= zoom_end, "value"]
    zoom.plot(ax=ax_zoom, lw=0.9, color="#ff7f0e")
    ax_zoom.set_title("Zoom: first 14 days")
    ax_zoom.set_ylabel("value")
    ax_zoom.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def write_summaries(reports: list[DataReport], csv_path: Path, json_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_row() for r in reports]).set_index("name")
    df.to_csv(csv_path)
    with json_path.open("w") as f:
        json.dump([r.to_row() for r in reports], f, indent=2, default=str)
