"""Build the systems-evaluation figures for the §swarm-eval section.

Produces two PDFs in `paper/figures/`:
  - `swarm_p99_variance.pdf`  — grouped bar of P99 mean ± std on the
                                 saturation workload (Azure), reactive vs
                                 FD-DSP-Auto, for the three fleet sizes.
                                 Tells the variance-reduction story.
  - `swarm_slo_attainment.pdf` — grouped bar of SLO attainment across
                                 (workload × fleet) cells, reactive vs auto.
                                 Shows sub-saturation parity at a glance.

Reads from `outputs/swarm_results/` (baseline) and
`outputs/swarm_results_auto/` (auto-tune). Idempotent; safe to re-run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "swarm_results"
AUTO_DIR = ROOT / "outputs" / "swarm_results_auto"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_runs(directory: Path, policy_filter: str | None = None) -> list[dict]:
    runs = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        with path.open() as fh:
            r = json.load(fh)
        if policy_filter is not None and not r["policy_name"].startswith(policy_filter):
            continue
        runs.append(r)
    return runs


def group(runs: list[dict]) -> dict[tuple, dict]:
    g: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        g[(r["dataset_name"], r["fleet_size"])].append(r)
    out: dict[tuple, dict] = {}
    for k, items in g.items():
        out[k] = {
            "p99_mean": mean(x["p99_latency_ms"] for x in items),
            "p99_std": pstdev(x["p99_latency_ms"] for x in items)
            if len(items) > 1
            else 0.0,
            "slo_mean": mean(x["slo_attainment"] for x in items),
            "slo_std": pstdev(x["slo_attainment"] for x in items)
            if len(items) > 1
            else 0.0,
            "rej_mean": mean(x["rejection_rate"] for x in items),
        }
    return out


def fig_p99_variance(reactive: dict, autotune: dict, path: Path) -> None:
    fleet_sizes = [10, 25, 50]
    react_mean = [reactive[("azure_llm_2024", n)]["p99_mean"] for n in fleet_sizes]
    react_std = [reactive[("azure_llm_2024", n)]["p99_std"] for n in fleet_sizes]
    auto_mean = [autotune[("azure_llm_2024", n)]["p99_mean"] for n in fleet_sizes]
    auto_std = [autotune[("azure_llm_2024", n)]["p99_std"] for n in fleet_sizes]

    x = np.arange(len(fleet_sizes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(
        x - w / 2,
        react_mean,
        w,
        yerr=react_std,
        capsize=4,
        color="#4878D0",
        label="Reactive baseline",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        auto_mean,
        w,
        yerr=auto_std,
        capsize=4,
        color="#EE854A",
        label="FD-DSP-Auto",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in fleet_sizes])
    ax.set_ylabel("P99 end-to-end latency (ms)")
    ax.set_xlabel("Fleet size (AzureLLMInferenceTrace-2024)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["right", "top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_slo_grid(reactive: dict, autotune: dict, path: Path) -> None:
    workloads = ["alibaba_pai", "burstgpt", "azure_llm_2024"]
    fleets = [10, 25, 50]
    cells = [(w, n) for w in workloads for n in fleets]
    labels = [f"{w.replace('_', '\\_')}\nn={n}" for w, n in cells]
    react = [reactive[c]["slo_mean"] * 100 for c in cells]
    auto = [autotune[c]["slo_mean"] * 100 for c in cells]

    x = np.arange(len(cells))
    w = 0.4
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.bar(
        x - w / 2,
        react,
        w,
        color="#4878D0",
        label="Reactive baseline",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        auto,
        w,
        color="#EE854A",
        label="FD-DSP-Auto",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=0)
    ax.set_ylabel("SLO attainment (\\%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["right", "top"]].set_visible(False)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    reactive = group(load_runs(BASE_DIR, policy_filter="reactive"))
    autotune = group(load_runs(AUTO_DIR))

    p1 = FIG_DIR / "swarm_p99_variance.pdf"
    fig_p99_variance(reactive, autotune, p1)
    print(f"wrote {p1}")

    p2 = FIG_DIR / "swarm_slo_attainment.pdf"
    fig_slo_grid(reactive, autotune, p2)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
