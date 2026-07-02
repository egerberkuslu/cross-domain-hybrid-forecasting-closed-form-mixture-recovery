"""Generate the CHA-Hybrid v3 / v4 architecture flowchart for the paper.

Two-panel figure:
  (a) Data path: lookback → STL → (trend, season, residual) + foundation
                 → α-mixture → forecast
  (b) v3 vs v4 mixture head detail

Output:
  outputs/figures/fig_architecture.pdf
  outputs/figures/fig_architecture.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils.runlog import PhaseTimer

OUT = Path("outputs/figures")


def _box(
    ax,
    xy,
    w,
    h,
    text,
    facecolor="#ECEFF4",
    edgecolor="#2E3440",
    fontsize=8,
    weight="normal",
):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        weight=weight,
    )


def _arrow(ax, xy1, xy2, lw=1.2, color="#2E3440"):
    arr = FancyArrowPatch(
        xy1,
        xy2,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12, 4.0), gridspec_kw={"width_ratios": [3, 1.8]}
    )

    # ---- Panel (a): full architecture ----
    ax = ax_a
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    _box(ax, (0.0, 2.2), 1.4, 0.9, "lookback\n$\\mathbf{x}_t$", facecolor="#FFE4B5")

    # Decomposition path
    _box(ax, (1.8, 4.2), 1.6, 0.9, "STL\nperiod $P$", facecolor="#A3BE8C")
    _box(ax, (3.7, 5.0), 1.6, 0.5, "Theta\ntrend", facecolor="#D8DEE9", fontsize=7)
    _box(ax, (3.7, 4.3), 1.6, 0.5, "SeasonalNaive", facecolor="#D8DEE9", fontsize=7)
    _box(
        ax,
        (3.7, 3.6),
        1.6,
        0.5,
        "LSTM\nresidual",
        facecolor="#88C0D0",
        fontsize=7,
        weight="bold",
    )
    _box(ax, (5.6, 4.2), 1.4, 0.9, "$\\hat{y}^{\\mathrm{dec}}$", facecolor="#EBCB8B")

    # Foundation path
    _box(ax, (1.8, 0.6), 1.6, 0.9, "Chronos-Bolt\n(frozen)", facecolor="#B48EAD")
    _box(ax, (5.6, 0.6), 1.4, 0.9, "$\\hat{y}^{\\mathrm{glob}}$", facecolor="#EBCB8B")

    # Mixture
    _box(
        ax,
        (7.3, 2.2),
        1.4,
        0.9,
        "$\\alpha$-mixture",
        facecolor="#5E81AC",
        fontsize=9,
        weight="bold",
    )
    _box(
        ax,
        (8.9, 2.2),
        1.0,
        0.9,
        "$\\hat{y}_t$",
        facecolor="#FFE4B5",
        fontsize=10,
        weight="bold",
    )

    # Arrows
    _arrow(ax, (1.4, 2.7), (1.8, 4.6))  # lookback → STL
    _arrow(ax, (1.4, 2.6), (1.8, 1.05))  # lookback → foundation
    _arrow(ax, (3.4, 5.25), (3.7, 5.25))
    _arrow(ax, (3.4, 4.55), (3.7, 4.55))
    _arrow(ax, (3.4, 3.85), (3.7, 3.85))
    _arrow(ax, (5.3, 4.7), (5.6, 4.7))
    _arrow(ax, (5.3, 4.0), (5.6, 4.55))
    _arrow(ax, (5.3, 5.25), (5.6, 4.85))
    _arrow(ax, (7.0, 4.55), (7.3, 3.0))  # decomp → mixture
    _arrow(ax, (7.0, 1.05), (7.3, 2.4))  # global → mixture
    _arrow(ax, (8.7, 2.65), (8.9, 2.65))  # mixture → forecast

    ax.text(
        5.0,
        5.55,
        "decomposition expert",
        fontsize=9,
        ha="center",
        style="italic",
        color="#5E81AC",
    )
    ax.text(
        5.0,
        0.25,
        "global expert",
        fontsize=9,
        ha="center",
        style="italic",
        color="#5E81AC",
    )
    ax.text(
        5.0,
        0.0,
        "(a) two-tier hybrid architecture",
        fontsize=10,
        ha="center",
        weight="bold",
    )

    # ---- Panel (b): mixture head detail ----
    ax = ax_b
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # CHA-S head (selected variant)
    _box(
        ax,
        (0.4, 4.0),
        2.4,
        0.7,
        "CHA-S: scalar $\\alpha_h$\ngrid-tuned on val",
        facecolor="#A3BE8C",
        fontsize=8,
        weight="bold",
    )
    _box(
        ax,
        (0.4, 3.0),
        2.4,
        0.7,
        "closed-form\nBates--Granger recovery",
        facecolor="#A3BE8C",
        fontsize=7,
    )

    # CHA-L head (negative result)
    _box(
        ax,
        (0.4, 1.6),
        2.4,
        1.2,
        "CHA-L (negative):\n$\\alpha(\\mathbf{x}) = $ MLP$_{\\theta}(\\phi(\\mathbf{x}))$\nper-window learned",
        facecolor="#5E81AC",
        fontsize=8,
        weight="bold",
    )
    _box(
        ax,
        (3.0, 1.6),
        2.6,
        1.2,
        "$\\phi(\\mathbf{x})$: 8 features\n"
        "mean, std, slope, range,\nresidvar, last-gap,\n"
        "seasonal-phase, ACF",
        facecolor="#D8DEE9",
        fontsize=7,
    )

    # CHA-LR (regularised negative result)
    _box(
        ax,
        (0.4, 0.2),
        5.2,
        1.1,
        "CHA-LR (negative):\nheld-out early stopping (val→val_tr 80% / val_ho 20%);\n"
        "8-hidden 1-layer MLP, $\\lambda{=}5{\\times}10^{-3}$",
        facecolor="#88C0D0",
        fontsize=7,
        weight="bold",
    )

    ax.text(
        3.0, 5.05, "(b) three mixture heads", fontsize=10, ha="center", weight="bold"
    )

    fig.tight_layout()
    out_pdf = OUT / "fig_architecture.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(OUT / "fig_architecture.png", bbox_inches="tight", dpi=150)
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    with PhaseTimer(
        "build_architecture_figure",
        notes="paper Fig 1: two-tier architecture + mixture heads",
    ) as t:
        main()
        t.add_output("pdf", str(OUT / "fig_architecture.pdf"))
