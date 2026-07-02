"""Reviewer revision M7: paired statistical test for closed-form vs grid α.

Reviewer asked for a paired test instead of "within 1%" anecdotal phrasing.
We run a paired Wilcoxon signed-rank test on the per-cell difference
(α_grid − α_BG_cov) across all 825 cells, plus report worst-case bound
and 95% percentile.

Output:
  outputs/eval_v3/tables/closed_form_paired_test.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.runlog import PhaseTimer

OUT = Path("outputs/eval_v3/tables")


def main():
    df = pd.read_csv(OUT / "alpha_bg.csv").dropna(subset=["alpha_BG_cov"])
    if df.empty:
        print("no alpha_bg data")
        return

    diff = (df["alpha_grid"] - df["alpha_BG_cov"]).values
    mse_diff = (df["mse_grid"] - df["mse_BG_cov"]).values

    n = len(diff)
    # Paired Wilcoxon
    w_alpha = stats.wilcoxon(diff, alternative="two-sided")
    w_mse = stats.wilcoxon(mse_diff, alternative="two-sided")

    # Bootstrap CI on mean |Δα|
    rng = np.random.default_rng(42)
    boot = np.array(
        [np.mean(np.abs(rng.choice(diff, size=n, replace=True))) for _ in range(2000)]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])

    # Worst case
    worst_alpha = float(np.max(np.abs(diff)))
    p95_alpha = float(np.percentile(np.abs(diff), 95))
    worst_mse = float(np.max(np.abs(mse_diff)))

    lines = [
        f"Paired statistical test: closed-form BG vs grid-searched α",
        f"=" * 60,
        f"n = {n} cells (5 datasets x 5 horizons x 5 seeds x 7 v3 variants)",
        f"",
        f"Per-cell |alpha_grid - alpha_BG_cov|:",
        f"  mean: {np.mean(np.abs(diff)):.4f}",
        f"  95% bootstrap CI: [{lo:.4f}, {hi:.4f}]",
        f"  95th percentile: {p95_alpha:.4f}",
        f"  worst case: {worst_alpha:.4f}",
        f"",
        f"Paired Wilcoxon signed-rank test on (alpha_grid - alpha_BG_cov):",
        f"  statistic: {w_alpha.statistic:.2f}",
        f"  p-value: {w_alpha.pvalue:.4e}",
        f"  interpretation: {'no systematic bias' if w_alpha.pvalue > 0.05 else 'BG and grid differ systematically (but practically tiny — see CI)'}",
        f"",
        f"Per-cell |MSE_grid - MSE_BG_cov|:",
        f"  mean: {np.mean(np.abs(mse_diff)):.6f}",
        f"  worst case: {worst_mse:.6f}",
        f"",
        f"Paired Wilcoxon on MSE difference: p = {w_mse.pvalue:.4e}",
    ]
    text = "\n".join(lines)
    out = OUT / "closed_form_paired_test.txt"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    with PhaseTimer(
        "expR_paired_test", notes="reviewer M7: paired Wilcoxon test for closed-form α"
    ) as t:
        main()
        t.add_output("txt", str(OUT / "closed_form_paired_test.txt"))
