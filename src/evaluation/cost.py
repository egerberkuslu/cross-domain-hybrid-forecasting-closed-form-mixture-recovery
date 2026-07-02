"""Computational cost table: train time, inference time, parameter count.

Per the Phase-6 spec: "Measure computational cost (training time, inference
time, parameter count) for every model."  We aggregate over seeds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cost_table(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Return a per-variant cost summary aggregated across dataset/horizon/seed."""
    grp = df_runs.groupby("variant").agg(
        fit_seconds_mean=("fit_seconds", "mean"),
        fit_seconds_total=("fit_seconds", "sum"),
        predict_seconds_mean=("predict_seconds", "mean"),
        predict_seconds_total=("predict_seconds", "sum"),
        n_parameters=("n_parameters", "first"),
        n_runs=("rmse_native", "count"),
    )
    grp = grp.sort_values("fit_seconds_total", ascending=False)
    return grp


def cost_table_per_dataset(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Same as cost_table but split per (variant, dataset)."""
    grp = df_runs.groupby(["variant", "dataset"]).agg(
        fit_seconds_mean=("fit_seconds", "mean"),
        predict_seconds_mean=("predict_seconds", "mean"),
        n_parameters=("n_parameters", "first"),
        n_runs=("rmse_native", "count"),
    )
    return grp.reset_index()


def write_cost_tables(df_runs: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    p1 = out_dir / "cost_table.csv"
    cost_table(df_runs).to_csv(p1)
    written["overall"] = p1
    p2 = out_dir / "cost_table_per_dataset.csv"
    cost_table_per_dataset(df_runs).to_csv(p2, index=False)
    written["per_dataset"] = p2
    return written
