"""Build the cross-domain forecasting figure for §experimental-eval.

Compares per-dataset RMSE (mean across 3 seeds) for a representative
slice of forecasters at h=6 across the 8 datasets, with a visual
separator between the network-traffic and AI-inference subgroups.

Successful AI-inference runs are scarce on Azure and AlibabaPAI
(foundation-model jobs OOMed on the available GPU); we therefore
restrict the figure to the model set that succeeded on every
dataset: naive, seasonal_naive, arima, theta, holt_winters.
This keeps the cross-domain claim defensible.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "outputs" / "metrics"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NETWORK_DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
INFERENCE_DATASETS = ["burstgpt", "azure_llm_2024", "alibaba_pai"]
DATASETS = NETWORK_DATASETS + INFERENCE_DATASETS

MODELS = [
    "naive",
    "seasonal_naive",
    "theta",
    "arima",
    "chronos_bolt_zs",
    "cha_hybrid_v3",
]
HORIZON = 6


def load_rmse(dataset: str, model: str, horizon: int) -> float | None:
    """Return mean RMSE across seeds, or None if no successful run."""
    paths = list(METRICS_DIR.glob(f"{dataset}__{model}__h{horizon}__s*.json"))
    if not paths:
        return None
    vals = []
    for p in paths:
        try:
            with p.open() as fh:
                d = json.load(fh)
        except Exception:
            continue
        # successful runs have metrics under 'test' / 'rmse' (sometimes nested)
        rmse = _extract_rmse(d)
        if rmse is not None:
            vals.append(rmse)
    return mean(vals) if vals else None


def _extract_rmse(d: dict) -> float | None:
    """Return scaled-domain RMSE for a successful run, else None."""
    if d.get("status") != "ok":
        return None
    scaled = d.get("metrics_scaled")
    if isinstance(scaled, dict) and isinstance(scaled.get("rmse"), (int, float)):
        return float(scaled["rmse"])
    return None


def build_matrix() -> dict[tuple, float | None]:
    out: dict[tuple, float | None] = {}
    for d in DATASETS:
        for m in MODELS:
            out[(d, m)] = load_rmse(d, m, HORIZON)
    return out


def make_figure(matrix: dict, path: Path) -> None:
    n_models = len(MODELS)
    n_datasets = len(DATASETS)
    x = np.arange(n_datasets)
    bar_w = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    pretty_model = {
        "naive": "Naive",
        "seasonal_naive": "SeasonalNaive",
        "theta": "Theta",
        "arima": "ARIMA",
        "chronos_bolt_zs": "Chronos-Bolt",
        "cha_hybrid_v3": "CHA-S",
    }
    pretty_dataset = {
        "cesnet": "CESNET",
        "abilene": "Abilene",
        "geant": "GEANT",
        "nab_aws_cpu": "NAB-CPU",
        "nab_twitter": "NAB-Twitter",
        "burstgpt": "BurstGPT",
        "azure_llm_2024": "AzureLLM",
        "alibaba_pai": "AlibabaPAI",
    }

    fig, ax = plt.subplots(figsize=(9, 4.0))
    for j, m in enumerate(MODELS):
        ys = []
        for d in DATASETS:
            v = matrix.get((d, m))
            ys.append(v if v is not None else 0.0)
        ax.bar(
            x + (j - n_models / 2 + 0.5) * bar_w,
            ys,
            bar_w,
            label=pretty_model.get(m, m),
            color=colors[j],
            edgecolor="black",
            linewidth=0.3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [pretty_dataset.get(d, d) for d in DATASETS],
        rotation=25,
        ha="right",
        fontsize=11,
    )
    ax.set_ylabel(f"Test RMSE ($h{{=}}{HORIZON}$)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    # Visual separator between domains
    boundary = len(NETWORK_DATASETS) - 0.5
    ax.axvline(boundary, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.text(
        boundary / 2,
        ax.get_ylim()[1] * 0.95,
        "Network traffic",
        ha="center",
        fontsize=11,
        color="gray",
    )
    ax.text(
        boundary + (len(INFERENCE_DATASETS)) / 2 + 0.5,
        ax.get_ylim()[1] * 0.95,
        "Edge AI inference",
        ha="center",
        fontsize=11,
        color="gray",
    )
    ax.legend(
        frameon=False,
        fontsize=10,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
    )
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    m = build_matrix()
    out = FIG_DIR / "cross_domain_rmse.pdf"
    make_figure(m, out)
    print(f"wrote {out}")
    print("\n=== matrix (mean RMSE across seeds, '—' = no run) ===")
    print(f"{'dataset':18s}  " + "  ".join(f"{x:>10s}" for x in MODELS))
    for d in DATASETS:
        row = []
        for mod in MODELS:
            v = m.get((d, mod))
            row.append(f"{v:>10.3f}" if v is not None else f"{'—':>10s}")
        print(f"{d:18s}  " + "  ".join(row))


if __name__ == "__main__":
    main()
