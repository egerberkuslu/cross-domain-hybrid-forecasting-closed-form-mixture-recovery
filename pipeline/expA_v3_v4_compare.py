"""v3-vs-v4 head-to-head comparison.

Run after CHA-Hybrid v4 has finished training across the grid.
Produces:
  outputs/eval_v3/tables/v3_vs_v4.csv          — per-cell delta in RMSE
  outputs/eval_v3/tables/v4_alpha_summary.csv  — learned α(x) statistics
  paper/tables/v3_vs_v4.tex                    — LaTeX table for the manuscript

Key claims to check empirically:
  1. Does v4 strictly improve on v3?  (cell-level Δ RMSE)
  2. Does v4's learned α(x) actually use a wide range?  (alpha_min, alpha_max,
     alpha_std per cell — if v4 just collapses to a constant, the MLP head
     did not learn anything beyond the v3 scalar)
  3. Where does v4 most help?  (rank cells by Δ RMSE)
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.runlog import PhaseTimer

OUT_TBL = Path("outputs/eval_v3/tables")
PAPER_TBL = Path("paper/tables")
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]


def collect_rmse(variant: str) -> dict:
    out = defaultdict(list)
    for f in glob.glob(f"outputs/metrics/*__{variant}__*.json"):
        try:
            m = json.load(open(f))
            if m.get("status") != "ok":
                continue
            out[(m["dataset"], int(m["horizon"]))].append(
                float(m["metrics_scaled"]["rmse"])
            )
        except Exception:
            continue
    return out


def collect_v4_alpha_stats() -> dict:
    out = {}
    for p in glob.glob("outputs/checkpoints/*cha_hybrid_v4*.pt"):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            parts = Path(p).stem.split("__")
            ds, _, h, s = parts
            diag = ck.get("alpha_mlp_diag", {}) or {}
            out[(ds, int(h.lstrip("h")), int(s.lstrip("s")))] = {
                "alpha_min": diag.get("alpha_min"),
                "alpha_max": diag.get("alpha_max"),
                "alpha_mean": diag.get("alpha_mean"),
                "alpha_std": diag.get("alpha_std"),
                "best_val_mse": diag.get("best_val_mse"),
                "epochs_trained": diag.get("epochs_trained"),
            }
        except Exception:
            continue
    return out


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    PAPER_TBL.mkdir(parents=True, exist_ok=True)
    v3 = collect_rmse("cha_hybrid_v3")
    v4 = collect_rmse("cha_hybrid_v4")
    if not v4:
        print("[compare] no v4 metrics yet — exiting")
        return
    alpha_stats = collect_v4_alpha_stats()

    rows = []
    for (ds, h), v3_seeds in v3.items():
        if (ds, h) not in v4:
            continue
        v4_seeds = v4[(ds, h)]
        rows.append(
            {
                "dataset": ds,
                "horizon": h,
                "n_seeds_v3": len(v3_seeds),
                "n_seeds_v4": len(v4_seeds),
                "v3_rmse_mean": float(np.mean(v3_seeds)),
                "v4_rmse_mean": float(np.mean(v4_seeds)),
                "v3_rmse_std": float(
                    np.std(v3_seeds, ddof=1) if len(v3_seeds) > 1 else 0
                ),
                "v4_rmse_std": float(
                    np.std(v4_seeds, ddof=1) if len(v4_seeds) > 1 else 0
                ),
                "delta_rmse": float(np.mean(v4_seeds) - np.mean(v3_seeds)),
                "rel_delta_pct": float(
                    100 * (np.mean(v4_seeds) - np.mean(v3_seeds)) / np.mean(v3_seeds)
                ),
                "v4_wins": bool(np.mean(v4_seeds) < np.mean(v3_seeds)),
            }
        )
    df = pd.DataFrame(rows).sort_values(["dataset", "horizon"])
    out_csv = OUT_TBL / "v3_vs_v4.csv"
    df.to_csv(out_csv, index=False)
    print(f"[compare] wrote {out_csv}")

    # Alpha stats aggregated per (dataset, horizon)
    alpha_rows = []
    for k, v in alpha_stats.items():
        ds, h, s = k
        alpha_rows.append(
            {
                "dataset": ds,
                "horizon": h,
                "seed": s,
                **v,
            }
        )
    if alpha_rows:
        adf = pd.DataFrame(alpha_rows)
        out_csv_a = OUT_TBL / "v4_alpha_summary.csv"
        adf.to_csv(out_csv_a, index=False)
        print(f"[compare] wrote {out_csv_a}")

        # Aggregated alpha stats per (dataset, horizon)
        agg = (
            adf.groupby(["dataset", "horizon"])
            .agg(
                alpha_mean_avg=("alpha_mean", "mean"),
                alpha_min_min=("alpha_min", "min"),
                alpha_max_max=("alpha_max", "max"),
                alpha_std_mean=("alpha_std", "mean"),
            )
            .reset_index()
        )
        print("\n=== v4 learned α(x) range per (dataset, horizon) ===")
        print(agg.to_string(index=False))

    # Headline numbers
    if not df.empty:
        wins = int(df["v4_wins"].sum())
        total = int(len(df))
        median_rel = float(df["rel_delta_pct"].median())
        print(f"\n=== v4 vs v3 headline ===")
        print(f"  v4 beats v3 in {wins}/{total} cells")
        print(f"  median relative Δ RMSE = {median_rel:+.2f}% (negative = v4 better)")
        print(f"  worst v4 regression  = {df['rel_delta_pct'].max():+.2f}%")
        print(f"  best v4 improvement  = {df['rel_delta_pct'].min():+.2f}%")

        # LaTeX table
        lines = [
            "% auto-generated by pipeline/expA_v3_v4_compare.py",
            r"\begin{table}[t]",
            r"\centering\small",
            r"\caption{v4 (context-aware learned mixture) versus \HybridV{} "
            r"(horizon-adaptive scalar): mean test RMSE on the scaled domain, "
            r"averaged across five seeds.  Negative $\Delta$ means v4 improves "
            r"over v3.}",
            r"\label{tab:v3_vs_v4}",
            r"\begin{tabular}{l r r r r}",
            r"\toprule",
            r"dataset & $h$ & v3 RMSE & v4 RMSE & $\Delta$ (\%) \\",
            r"\midrule",
        ]
        for ds in DATASETS:
            sub = df[df["dataset"] == ds]
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                lines.append(
                    f"{ds if r['horizon'] == sub['horizon'].iloc[0] else ''} & "
                    f"{int(r['horizon'])} & {r['v3_rmse_mean']:.3f} & "
                    f"{r['v4_rmse_mean']:.3f} & {r['rel_delta_pct']:+.2f} \\\\"
                )
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        (PAPER_TBL / "v3_vs_v4.tex").write_text("\n".join(lines))
        print(f"  wrote paper/tables/v3_vs_v4.tex")


if __name__ == "__main__":
    with PhaseTimer(
        "expA_v3_v4_compare",
        notes="cell-level v3-vs-v4 head-to-head + α(x) distribution stats",
    ) as t:
        main()
        t.add_output("csv", str(OUT_TBL / "v3_vs_v4.csv"))
        t.add_output("tex", str(PAPER_TBL / "v3_vs_v4.tex"))
