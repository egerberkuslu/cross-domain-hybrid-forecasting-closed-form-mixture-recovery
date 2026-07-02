"""Refresh the table_{rmse,mae,mape,smape,r2}_{scaled,native}.csv
leaderboard files from outputs/eval_v3/tables/metrics_aggregated.csv.

These were originally written by phase6_eval_v3.py at the early
Stage-A run (before v4_fix and the size/STL sensitivity variants
existed) and have been stale ever since.  This script re-derives
them straight from the fresh metrics_aggregated.csv without
re-running any model fits.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.runlog import PhaseTimer

OUT = Path("outputs/eval_v3/tables")


def main():
    src = OUT / "metrics_aggregated.csv"
    if not src.exists():
        raise SystemExit(
            "metrics_aggregated.csv missing — run refresh_metrics_aggregated.py first"
        )
    df = pd.read_csv(src)
    for metric in ("rmse", "mae", "mape", "smape", "r2"):
        for scope in ("scaled", "native"):
            col = f"{metric}_{scope}_mean"
            if col not in df.columns:
                continue
            piv = df.pivot_table(
                index="variant",
                columns=["dataset", "horizon"],
                values=col,
            ).reset_index()
            out = OUT / f"table_{metric}_{scope}.csv"
            piv.to_csv(out, index=False)
            print(f"wrote {out}  ({piv.shape[0]} variants × {piv.shape[1]-1} cells)")


if __name__ == "__main__":
    with PhaseTimer(
        "refresh_native_leaderboards",
        notes="re-derive table_{rmse,mae,mape,smape,r2}_{scaled,native}.csv",
    ) as t:
        main()
        t.add_output("dir", str(OUT))
