"""Stage A — Experiment 1.3: Per-sample interpretability visualisation.

For each dataset and a couple of representative horizons, picks one test
window and overlays:
  * ground truth y_true
  * decomposition-expert-only prediction (STL trend + seasonal + LSTM-residual)
  * global-expert-only prediction (Chronos-Bolt zero-shot)
  * the proposed hybrid prediction with the tuned α weight

Produces a 5×3 grid PDF.  Output:
  outputs/figures/per_sample_decomp.pdf

Requires Phase 6 v3 ablation outputs (decomp_only / chronos_bolt /
cha_hybrid_v3 prediction .npz files).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.runlog import PhaseTimer

PRED_DIR = Path("outputs/predictions")
OUT_DIR = Path("outputs/figures")
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
HORIZONS = [1, 6, 24]
SEED = 42


def _load(ds: str, variant: str, h: int, s: int) -> np.ndarray | None:
    p = PRED_DIR / f"{ds}__{variant}__h{h}__s{s}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=False)
    return d["y_true_scaled"], d["y_pred_scaled"]


def _pick_window(y_true: np.ndarray, n: int = 1, mode: str = "median") -> int:
    """Pick a representative window — by default the one whose ground-truth
    range is at the median (avoids extreme outliers and dead-flat regions)."""
    rng = y_true.max(axis=1) - y_true.min(axis=1)
    if mode == "median":
        idx = int(np.argsort(rng)[len(rng) // 2])
    elif mode == "max":
        idx = int(np.argmax(rng))
    else:
        idx = int(np.argmin(rng))
    return idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nrows, ncols = len(DATASETS), len(HORIZONS)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 2.4 * nrows), squeeze=False
    )
    for i, ds in enumerate(DATASETS):
        for j, h in enumerate(HORIZONS):
            ax = axes[i][j]
            dec = _load(ds, "cha_hybrid_v3_decomp_only", h, SEED)
            glb = _load(ds, "chronos_bolt_zs", h, SEED)
            hyb = _load(ds, "cha_hybrid_v3", h, SEED)
            if hyb is None:
                ax.text(
                    0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            y_true, y_hyb = hyb
            idx = _pick_window(y_true)
            t = np.arange(y_true.shape[1])
            ax.plot(t, y_true[idx], "k-", lw=2.0, label="truth", alpha=0.9)
            if dec is not None:
                _, ydec = dec
                if ydec.shape[0] > idx:
                    ax.plot(
                        t,
                        ydec[idx],
                        "tab:blue",
                        ls="--",
                        lw=1.2,
                        label="decomp only",
                        alpha=0.8,
                    )
            if glb is not None:
                _, yglb = glb
                if yglb.shape[0] > idx:
                    ax.plot(
                        t,
                        yglb[idx],
                        "tab:orange",
                        ls=":",
                        lw=1.2,
                        label="global only",
                        alpha=0.8,
                    )
            ax.plot(t, y_hyb[idx], "tab:green", lw=1.6, label="hybrid", alpha=0.95)
            ax.set_title(f"{ds} · h={h}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            if i == 0 and j == ncols - 1:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.suptitle(
        "Per-sample forecast: decomposition vs. global vs. hybrid", fontsize=11, y=1.005
    )
    fig.tight_layout()
    out_pdf = OUT_DIR / "per_sample_decomp.pdf"
    out_png = OUT_DIR / "per_sample_decomp.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"[1.3] wrote {out_pdf}")
    print(f"[1.3] wrote {out_png}")


if __name__ == "__main__":
    with PhaseTimer(
        "expA_1.3_per_sample_viz",
        notes=f"5 datasets × {len(HORIZONS)} horizons; representative window per cell",
    ) as t:
        main()
        t.add_output("pdf", str(OUT_DIR / "per_sample_decomp.pdf"))
        t.add_output("png", str(OUT_DIR / "per_sample_decomp.png"))
