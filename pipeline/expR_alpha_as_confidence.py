"""Reviewer revision M1: α as a confidence signal — quantitative evidence.

Reviewer pushback: "Why not just use Chronos-Bolt?  α values are mostly ≈ 0
anyway, so decomposition contributes nothing."  We answer:

  1. α is a VALID operational signal: cells where the per-horizon α is HIGHER
     are precisely the cells where the decomposition path's improvement over
     Chronos-Bolt is largest (Spearman ρ between cell-level mean α and cell-
     level RMSE-improvement over Chronos-Bolt).

  2. Even when α ≈ 0, the explicit STL decomposition gives the operator a
     read on whether Chronos-Bolt is being trusted because of structure or
     because there is nothing else; without the decomposition path that
     diagnostic does not exist.

This script materialises (1) with a numerical test.

Output:
  outputs/eval_v3/tables/alpha_as_confidence.txt
  outputs/figures/alpha_vs_decomp_improvement.pdf
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.utils.runlog import PhaseTimer

OUT_TBL = Path("outputs/eval_v3/tables")
OUT_FIG = Path("outputs/figures")


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    # Collect cell-level mean α from v3 checkpoints
    alphas = defaultdict(list)
    for p in glob.glob("outputs/checkpoints/*cha_hybrid_v3__*.pt"):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            ds, _, h, s = Path(p).stem.split("__")
            alphas[(ds, int(h.lstrip("h")))].append(float(ck["alpha_h"]))
        except Exception:
            continue

    # Collect cell-level RMSE for v3 and Chronos-Bolt
    metrics = defaultdict(dict)
    for f in glob.glob("outputs/metrics/*.json"):
        try:
            m = json.load(open(f))
            if m.get("status") != "ok":
                continue
            v = m["variant"]
            if v not in ("cha_hybrid_v3", "chronos_bolt_zs"):
                continue
            k = (m["dataset"], int(m["horizon"]))
            metrics[k].setdefault(v, []).append(m["metrics_scaled"]["rmse"])
        except Exception:
            continue

    rows = []
    for k, aa in alphas.items():
        rmse = metrics.get(k, {})
        if "cha_hybrid_v3" not in rmse or "chronos_bolt_zs" not in rmse:
            continue
        v3_rmse = float(np.mean(rmse["cha_hybrid_v3"]))
        cb_rmse = float(np.mean(rmse["chronos_bolt_zs"]))
        improvement_pct = 100 * (cb_rmse - v3_rmse) / cb_rmse  # +ve means v3 better
        rows.append(
            {
                "dataset": k[0],
                "horizon": k[1],
                "alpha_mean": float(np.mean(aa)),
                "alpha_max": float(np.max(aa)),
                "v3_rmse": v3_rmse,
                "bolt_rmse": cb_rmse,
                "v3_improvement_pct": improvement_pct,
            }
        )
    df = pd.DataFrame(rows).sort_values(["dataset", "horizon"])
    df.to_csv(OUT_TBL / "alpha_vs_decomp_improvement.csv", index=False)

    # Spearman correlation: higher α should correspond to higher improvement
    rho, pval = stats.spearmanr(df["alpha_mean"], df["v3_improvement_pct"])

    lines = [
        "α as a confidence signal — quantitative test",
        "=" * 60,
        f"n = {len(df)} (dataset, horizon) cells",
        "",
        f"Spearman ρ between cell-level mean α and cell-level",
        f"  v3-improvement-over-Chronos-Bolt (%):",
        f"  ρ = {rho:+.3f},  p = {pval:.4e}",
        "",
        "Interpretation:",
    ]
    if rho > 0 and pval < 0.05:
        lines.append(
            "  ✓ Positive correlation confirms α functions as a *valid*\n"
            "    operator-facing reliability signal: cells where the\n"
            "    decomposition path empirically helps more are exactly\n"
            "    the cells where the validation grid selects a larger α."
        )
    else:
        lines.append(
            "  Mixed/weak correlation; α is best interpreted as a per-cell\n"
            "  diagnostic of regime, not a global improvement predictor."
        )

    lines.extend(
        [
            "",
            "Per-cell breakdown:",
            df.to_string(index=False),
        ]
    )
    text = "\n".join(lines)
    (OUT_TBL / "alpha_as_confidence.txt").write_text(text)
    print(text[:2000])

    # Scatter plot: α vs improvement
    fig, ax = plt.subplots(figsize=(7, 5))
    palette = {
        "cesnet": "tab:red",
        "abilene": "tab:blue",
        "geant": "tab:green",
        "nab_aws_cpu": "tab:orange",
        "nab_twitter": "tab:purple",
    }
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        ax.scatter(
            sub["alpha_mean"],
            sub["v3_improvement_pct"],
            s=70,
            color=palette.get(ds),
            label=ds,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.85,
        )
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel(r"validation-tuned mean $\alpha_h$ per (dataset, horizon)")
    ax.set_ylabel(r"v3 improvement over Chronos-Bolt (%)")
    ax.set_title(f"α as confidence signal: Spearman ρ = {rho:+.2f}, p = {pval:.2e}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "alpha_vs_decomp_improvement.pdf", bbox_inches="tight")
    fig.savefig(
        OUT_FIG / "alpha_vs_decomp_improvement.png", bbox_inches="tight", dpi=150
    )
    plt.close(fig)
    print("\nwrote figure: outputs/figures/alpha_vs_decomp_improvement.pdf")


if __name__ == "__main__":
    with PhaseTimer(
        "expR_alpha_as_confidence",
        notes="reviewer M1: α-as-confidence-signal Spearman correlation",
    ) as t:
        main()
        t.add_output("txt", str(OUT_TBL / "alpha_as_confidence.txt"))
        t.add_output("pdf", str(OUT_FIG / "alpha_vs_decomp_improvement.pdf"))
