"""Quick rebuild of the master metrics CSVs from outputs/metrics/*.json.

phase6_eval_v3 re-runs ablations which is expensive.  When we already
have all the per-cell metrics in outputs/metrics/, we just need to
re-aggregate them into the long-form and aggregated CSVs that
phase7_figures.py and other downstream consumers read.

Produces (refreshing in-place):
  outputs/eval_v3/tables/metrics_long_form.csv
  outputs/eval_v3/tables/metrics_aggregated.csv
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.runlog import PhaseTimer

OUT = Path("outputs/eval_v3/tables")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in glob.glob("outputs/metrics/*.json"):
        try:
            m = json.load(open(f))
            if m.get("status") != "ok":
                continue
            rec = {
                "dataset": m["dataset"],
                "variant": m["variant"],
                "model": m.get("model", ""),
                "horizon": int(m["horizon"]),
                "seed": int(m["seed"]),
                "fit_seconds": float(m.get("fit_seconds", 0)),
                "predict_seconds": float(m.get("predict_seconds", 0)),
                "n_parameters": int(m.get("n_parameters", 0) or 0),
            }
            for scope in ("metrics_scaled", "metrics_native"):
                scope_tag = scope.replace("metrics_", "")
                d = m.get(scope, {}) or {}
                for k in ("rmse", "mae", "mape", "smape", "r2"):
                    rec[f"{k}_{scope_tag}"] = float(d.get(k, np.nan))
            rows.append(rec)
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        print("(no metrics found)")
        return
    df = df.sort_values(["dataset", "variant", "horizon", "seed"])
    out_long = OUT / "metrics_long_form.csv"
    df.to_csv(out_long, index=False)
    print(f"wrote {out_long}  ({len(df)} rows, {df['variant'].nunique()} variants)")

    # Aggregate over seeds
    grp = df.groupby(["dataset", "variant", "horizon"], as_index=False).agg(
        rmse_scaled_mean=("rmse_scaled", "mean"),
        rmse_scaled_std=("rmse_scaled", "std"),
        mae_scaled_mean=("mae_scaled", "mean"),
        mae_scaled_std=("mae_scaled", "std"),
        mape_scaled_mean=("mape_scaled", "mean"),
        smape_scaled_mean=("smape_scaled", "mean"),
        r2_scaled_mean=("r2_scaled", "mean"),
        rmse_native_mean=("rmse_native", "mean"),
        mae_native_mean=("mae_native", "mean"),
        mape_native_mean=("mape_native", "mean"),
        smape_native_mean=("smape_native", "mean"),
        r2_native_mean=("r2_native", "mean"),
        n_parameters=("n_parameters", "first"),
        fit_seconds_mean=("fit_seconds", "mean"),
        predict_seconds_mean=("predict_seconds", "mean"),
        n_seeds=("seed", "nunique"),
    )
    out_agg = OUT / "metrics_aggregated.csv"
    grp.to_csv(out_agg, index=False)
    print(f"wrote {out_agg}  ({len(grp)} aggregated rows)")

    # Diagnostics
    print(f"\nVariant family counts:")
    fam_counts = df["variant"].value_counts()
    for v, c in fam_counts.items():
        if "cha_hybrid" in v:
            print(f"  {v:<32} {c} cells")


if __name__ == "__main__":
    with PhaseTimer(
        "refresh_metrics_aggregated",
        notes="re-aggregate outputs/metrics/*.json → master CSVs",
    ) as t:
        main()
        t.add_output("long", str(OUT / "metrics_long_form.csv"))
        t.add_output("agg", str(OUT / "metrics_aggregated.csv"))
