"""Phase-7 figures + paper-ready tables.

Produces under ``figures/`` and ``results/eval_v3/tables/``:

  Figures (PNG @ 130 dpi, also .pdf for paper):
    fig01_rank_heatmap.png      — per (dataset, horizon) rank of cha_hybrid_v3
                                  vs the best baseline; coloured by Δ-RMSE.
    fig02_per_horizon_rmse.png  — line plot of RMSE vs horizon for every
                                  dataset; CHA-S highlighted.
    fig03_ablation.png          — bar chart: v3 vs (decomp_only, global_only,
                                  fixed_alpha_0.5, altres_gru) per dataset.
    fig04_alpha_distribution.png— learned α_h distribution per dataset+horizon.
    fig05_cost_vs_accuracy.png  — log-log scatter of train+predict time vs RMSE;
                                  colour-coded by model family.
    fig06_dm_test_grid.png      — heatmap of CHA-S DM p-values across
                                  (model, dataset, horizon).
    fig07_pred_vs_actual.png    — sample prediction-vs-actual line plot for
                                  geant test set (one horizon).
    fig08_stl_decomposition.png — STL decomposition illustration on cesnet.

  Tables (CSV — paper-ready):
    table_main_results.csv      — wide pivot of mean ± std RMSE_native across all
                                  models (rows) × (dataset, horizon) (cols).
    table_proposed_wins.csv     — CHA-S vs each rival: W/L/T across 25 cells.
    table_dm_summary.csv        — DM-test significance summary per rival.
    table_cost.csv              — train_s, predict_s, n_params per model.
    table_ablation.csv          — RMSE per ablation variant per (dataset, horizon).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation import (
    load_all_runs,
    aggregate,
    pairwise_dm_against_proposed,
    cost_table,
    proposed_rank,
)
from src.preprocessing import load_preprocessed


PROPOSED = "cha_hybrid_v3"
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
HORIZONS = [1, 3, 6, 12, 24]

FIG_DIR = Path("outputs/figures")  # consolidated (was Path("figures") pre-reorg)
TABLE_DIR = Path("outputs/eval_v3/tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# --- Display names used in figure titles / legends / axis labels.
# Internal release-code names → public paper names (CHA-S / CHA-L / CHA-LR).
DISPLAY_NAME = {
    "cha_hybrid_v3": "CHA-S",
    "cha_hybrid_v4": "CHA-L",
    "cha_hybrid_v4_fix": "CHA-LR",
    "cha_hybrid": "CHA-Hybrid v1",
    "cha_hybrid_v2": "CHA-Hybrid v2",
    "cha_hybrid_v3_tiny": "CHA-S (bolt-tiny)",
    "cha_hybrid_v3_mini": "CHA-S (bolt-mini)",
    "cha_hybrid_v3_base": "CHA-S (bolt-base)",
    "cha_hybrid_v3_stl12": "CHA-S (P=12)",
    "cha_hybrid_v3_stl48": "CHA-S (P=48)",
    "cha_hybrid_v3_stl168": "CHA-S (P=168)",
    "cha_hybrid_v3_decomp_only": "CHA-S decomp-only",
    "cha_hybrid_v3_global_only": "Chronos-Bolt only",
    "cha_hybrid_v3_fixed_alpha_0.5": "CHA-S fixed-$\\alpha$=0.5",
    "cha_hybrid_v3_altres_gru": "CHA-S altres-GRU",
    "chronos_zs": "Chronos-T5 (ZS)",
    "chronos_ft": "Chronos-T5 (FT)",
    "chronos_bolt_zs": "Chronos-Bolt",
    "timesfm_zs": "TimesFM",
    "moirai_zs": "MOIRAI",
    "ttm_zs": "TTM",
    "seasonal_naive": "Seasonal-Naive",
    "holt_winters": "Holt-Winters",
    "cha_hybrid_decomp_only": "v1 decomp-only",
    "cha_hybrid_global_only": "v1 global-only",
    "cha_hybrid_fixed_alpha_0.5": "v1 fixed-$\\alpha$=0.5",
    "cha_hybrid_altres_lstm": "v1 altres-LSTM",
    "lstm": "LSTM",
    "gru": "GRU",
    "tcn": "TCN",
    "nbeats": "N-BEATS",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "nhits": "N-HiTS",
    "tft": "TFT",
    "tide": "TiDE",
    "tsmixer": "TSMixer",
    "naive": "Naive",
    "arima": "ARIMA",
    "theta": "Theta",
    "farima": "FARIMA",
    "xgboost": "XGBoost",
}


def display_name(variant: str) -> str:
    return DISPLAY_NAME.get(variant, variant)


# ---------------------------------------------------------------- helpers
def _model_family(variant: str) -> str:
    if variant == "cha_hybrid_v4_fix":
        return "Proposed (CHA-LR)"
    if variant == "cha_hybrid_v4":
        return "Proposed (CHA-L)"
    if variant.startswith("cha_hybrid_v3"):
        return "Proposed (CHA-S family)"
    if variant.startswith("cha_hybrid_v2"):
        return "CHA-Hybrid v2 (legacy)"
    if variant.startswith("cha_hybrid"):
        return "CHA-Hybrid v1 (legacy)"
    if variant in {"naive", "seasonal_naive", "holt_winters", "theta", "arima"}:
        return "Statistical"
    if variant in {"xgboost"}:
        return "Classical ML"
    if variant in {
        "lstm",
        "gru",
        "tcn",
        "nbeats",
        "dlinear",
        "patchtst",
        "nhits",
        "tft",
        "tide",
        "tsmixer",
    }:
        return "Deep neural"
    if variant in {
        "chronos_zs",
        "chronos_ft",
        "chronos_bolt_zs",
        "timesfm_zs",
        "moirai_zs",
        "ttm_zs",
    }:
        return "Foundation"
    return "Other"


# =========================================================================
# Figure 1 — rank heatmap
# =========================================================================
def fig01_rank_heatmap(df_runs):
    """3-panel side-by-side rank heatmap for CHA-S, CHA-L, CHA-LR."""
    family = [
        ("cha_hybrid_v3", "CHA-S"),
        ("cha_hybrid_v4", "CHA-L"),
        ("cha_hybrid_v4_fix", "CHA-LR"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 13))
    for ax, (variant, label) in zip(axes, family):
        df_rank = proposed_rank(df_runs, variant, "rmse_native")
        if df_rank.empty:
            ax.text(
                0.5,
                0.5,
                f"no data for {label}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        pivot = df_rank.pivot(index="dataset", columns="horizon", values="rank")
        pivot = pivot.reindex(index=DATASETS, columns=HORIZONS)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn_r",
            cbar=True,
            cbar_kws={"label": "Rank (1 = best)"},
            ax=ax,
            vmin=1,
            vmax=15,
        )
        ax.set_title(f"{label} — Rank per (dataset, horizon)", fontsize=13)
        ax.set_xlabel("Forecast horizon (hours)")
        ax.set_ylabel("Dataset")
    fig.tight_layout()
    out = FIG_DIR / "fig01_rank_heatmap.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def _old_fig01_rank_heatmap_unused(df_runs):
    df_rank = proposed_rank(df_runs, PROPOSED, "rmse_native")
    pivot = df_rank.pivot(index="dataset", columns="horizon", values="rank")
    pivot = pivot.reindex(index=DATASETS, columns=HORIZONS)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="RdYlGn_r",
        cbar_kws={"label": "Rank of CHA-S (1 = best)"},
        vmin=1,
        vmax=15,
        ax=ax,
    )
    ax.set_title("CHA-S — Rank per (dataset, horizon)")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Dataset")
    fig.tight_layout()
    out = FIG_DIR / "fig01_rank_heatmap.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# =========================================================================
# Figure 2 — RMSE vs horizon line plot
# =========================================================================
def fig02_per_horizon_rmse(df_runs):
    df_agg = aggregate(df_runs)
    # The proposed family plus the strongest external competitors.
    proposed_family = ["cha_hybrid_v3", "cha_hybrid_v4", "cha_hybrid_v4_fix"]
    competitors = ["chronos_bolt_zs", "timesfm_zs", "patchtst", "nhits"]
    ds_pretty = {
        "cesnet": "CESNET",
        "abilene": "Abilene",
        "geant": "GEANT",
        "nab_aws_cpu": "NAB-CPU",
        "nab_twitter": "NAB-Twitter",
    }
    # 2 rows x 3 cols layout (leaves one panel for the legend) — wider per-panel
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=False)
    axes_flat = axes.flatten()
    colours = {
        "cha_hybrid_v3": "tab:red",
        "cha_hybrid_v4": "tab:purple",
        "cha_hybrid_v4_fix": "tab:brown",
        "chronos_bolt_zs": "tab:blue",
        "timesfm_zs": "tab:cyan",
        "patchtst": "tab:olive",
        "nhits": "tab:gray",
    }
    handles_seen = {}
    for ax, ds in zip(axes_flat, DATASETS):
        sub = df_agg[df_agg["dataset"] == ds]
        for v in proposed_family + competitors:
            cell = sub[sub["variant"] == v].sort_values("horizon")
            if cell.empty:
                continue
            is_prop = v in proposed_family
            (line,) = ax.plot(
                cell["horizon"],
                cell["rmse_scaled_mean"],
                marker="o" if is_prop else "s",
                markersize=7 if is_prop else 5,
                linestyle="-" if is_prop else "--",
                linewidth=2.4 if is_prop else 1.4,
                color=colours.get(v),
                label=display_name(v),
            )
            handles_seen[display_name(v)] = line
        ax.set_title(ds_pretty.get(ds, ds), fontsize=14, fontweight="bold")
        ax.set_xlabel("Horizon $h$", fontsize=13)
        ax.set_ylabel("Test RMSE (scaled)", fontsize=13)
        ax.set_xticks(HORIZONS)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(True, alpha=0.3)
    # Hide the 6th axes; use it for the legend
    axes_flat[5].axis("off")
    axes_flat[5].legend(
        handles_seen.values(),
        handles_seen.keys(),
        loc="center",
        fontsize=13,
        frameon=True,
        title="Method",
        title_fontsize=13,
    )
    fig.suptitle(
        "RMSE vs forecast horizon — proposed CHA-S/CHA-L/CHA-LR vs top competitors",
        fontsize=15,
    )
    fig.tight_layout()
    out = FIG_DIR / "fig02_per_horizon_rmse.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# =========================================================================
# Figure 3 — ablation bar chart
# =========================================================================
def fig03_ablation(df_runs):
    abl = [
        "cha_hybrid_v3",
        "cha_hybrid_v3_decomp_only",
        "cha_hybrid_v3_global_only",
        "cha_hybrid_v3_fixed_alpha_0.5",
        "cha_hybrid_v3_altres_gru",
    ]
    df_agg = aggregate(df_runs)
    sub = df_agg[df_agg["variant"].isin(abl)]
    if sub.empty:
        print("[fig03] no ablation data yet — skipping")
        return None
    # average across horizons per (dataset, variant)
    pivot = (
        sub.groupby(["dataset", "variant"])["rmse_scaled_mean"]
        .mean()
        .reset_index()
        .pivot(index="dataset", columns="variant", values="rmse_scaled_mean")
        .reindex(index=DATASETS, columns=abl)
    )
    # Rename columns to display names so the legend reads cleanly.
    pivot = pivot.rename(columns={c: display_name(c) for c in pivot.columns})
    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", linewidth=0.4)
    ax.set_title(
        "Ablation — CHA-S vs its 4 ablation variants (mean RMSE across horizons)"
    )
    ax.set_ylabel("RMSE (scaled, mean over horizons)")
    ax.set_xlabel("Dataset")
    ax.legend(title="Variant", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig03_ablation.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# =========================================================================
# Figure 4 — learned α distribution
# =========================================================================
def fig04_alpha_distribution():
    """Read α_h from per-run metrics' chosen_hparams (when present) or from
    the checkpoint marker JSON for cha_hybrid_v3."""
    alphas = defaultdict(list)  # (dataset, horizon) -> list of alpha values
    # Strict match: ONLY the proposed CHA-S (cha_hybrid_v3) variant —
    # the *cha_hybrid_v3* glob would also pick up _tiny, _mini, _base,
    # _stl12/48/168 sensitivity variants, polluting the heatmap.
    for fn in Path("outputs/checkpoints").glob("*__cha_hybrid_v3__h*__s*.pt.json"):
        try:
            d = json.loads(fn.read_text())
            ds = fn.stem.split("__")[0]
            h = int(d.get("horizon", 0))
            a = d.get("alpha_h")
            if a is not None:
                alphas[(ds, h)].append(float(a))
        except Exception:
            continue
    if not alphas:
        print("[fig04] no alpha data found — skipping")
        return None
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot = pd.DataFrame(
        {
            ds: [
                np.mean(alphas[(ds, h)]) if (ds, h) in alphas else np.nan
                for h in HORIZONS
            ]
            for ds in DATASETS
        },
        index=HORIZONS,
    ).T  # rows = dataset, cols = horizon
    sns.heatmap(
        pivot.astype(float),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "α (0 = pure global, 1 = pure decomp)"},
    )
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Dataset")
    ax.set_title("Validation-tuned scalar α_h per (dataset, horizon) — CHA-S")
    fig.tight_layout()
    out = FIG_DIR / "fig04_alpha_distribution.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# =========================================================================
# Figure 5 — cost vs accuracy scatter
# =========================================================================
def fig05_cost_vs_accuracy(df_runs):
    df_agg = aggregate(df_runs)
    # Exclude internal legacy variants (CHA-Hybrid v1 / v2) and their
    # ablation children from the published figure; they are not part of
    # the proposed family and would clutter the plot with unexplained
    # legend entries.
    legacy = df_agg["variant"].str.startswith("cha_hybrid_v2") | (
        df_agg["variant"].str.startswith("cha_hybrid")
        & ~df_agg["variant"].str.startswith("cha_hybrid_v")
    )
    df_agg = df_agg[~legacy].copy()
    df_agg["family"] = df_agg["variant"].map(_model_family)
    # average per variant across all datasets/horizons
    summary = (
        df_agg.groupby(["variant", "family"])
        .agg(
            rmse=("rmse_scaled_mean", "mean"),
            fit_s=("fit_seconds_mean", "mean"),
            pred_s=("predict_seconds_mean", "mean"),
        )
        .reset_index()
    )
    summary["total_s"] = summary["fit_s"] + summary["pred_s"]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    families = summary["family"].unique()
    palette = sns.color_palette("tab10", n_colors=len(families))
    # Single labeling block (used by all families) -- restricted to key
    # methods to avoid label collisions in the dense cluster near 1 sec.
    labeled = {
        "cha_hybrid_v3",
        "chronos_bolt_zs",
        "timesfm_zs",
        "chronos_ft",
        "arima",
        "naive",
    }
    # Hand-tuned offsets to avoid overlap on the cost-accuracy plane.
    offsets = {
        "cha_hybrid_v3": (12, -22),
        "chronos_bolt_zs": (-95, 18),
        "chronos_ft": (-110, -4),
        "timesfm_zs": (-90, -22),
        "arima": (12, -4),
        "naive": (12, -2),
    }
    for fam, color in zip(families, palette):
        sub = summary[summary["family"] == fam]
        ax.scatter(
            sub["total_s"],
            sub["rmse"],
            s=110,
            label=fam,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.85,
        )
        for _, r in sub.iterrows():
            if r["variant"] not in labeled:
                continue
            off = offsets.get(r["variant"], (6, 4))
            ax.annotate(
                display_name(r["variant"]),
                (r["total_s"], r["rmse"]),
                fontsize=11,
                fontweight="bold",
                alpha=0.95,
                xytext=off,
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.25, 5.0)
    ax.set_xlabel(
        "Total time per run -- train + predict (sec, log scale)",
        fontsize=13,
    )
    ax.set_ylabel(
        "Mean RMSE (scaled, log; averaged over 25 cells)",
        fontsize=13,
    )
    ax.set_title(
        "Computational cost vs accuracy (outliers clipped)",
        fontsize=14,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.95)
    fig.tight_layout()
    out = FIG_DIR / "fig05_cost_vs_accuracy.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# =========================================================================
# Figure 6 — DM-test heatmap
# =========================================================================
def fig06_dm_test_grid():
    # Prefer the curated, complete saved DM result (all 5 network datasets x 5
    # horizons against the 17 paper rivals, no missing cells). Fall back to the
    # live recompute only if the saved file is absent.
    _dm_full = TABLE_DIR.parent / "dm_test" / "dm_test_pairwise_full.csv"
    if _dm_full.exists():
        df_dm = pd.read_csv(_dm_full)
        df_dm = df_dm[df_dm["variant_a"] == PROPOSED].copy()
    else:
        df_dm = pairwise_dm_against_proposed(proposed=PROPOSED)
    if df_dm.empty:
        print("[fig06] no DM data — skipping")
        return None

    # Three-way encoding: +1 = CHA-S significantly better,
    # 0 = statistical tie, -1 = CHA-S significantly worse.
    def _verdict(row):
        if not row["significant_at_005"]:
            return 0.0
        return 1.0 if row["statistic"] < 0 else -1.0

    df_dm["verdict"] = df_dm.apply(_verdict, axis=1)
    # Exclude legacy CHA-Hybrid v1 / v2 ablation children from the
    # published heatmap; they are not the rival comparisons of interest
    # and pollute the y-axis with unexplained legend entries.
    legacy_b = df_dm["variant_b"].str.startswith("cha_hybrid_v2") | (
        df_dm["variant_b"].str.startswith("cha_hybrid")
        & ~df_dm["variant_b"].str.startswith("cha_hybrid_v")
    )
    df_dm = df_dm[~legacy_b].copy()
    pivot = df_dm.pivot_table(
        index="variant_b",
        columns=["dataset", "horizon"],
        values="verdict",
        aggfunc="mean",
    )
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([DATASETS, HORIZONS]))
    pivot.index = [display_name(v) for v in pivot.index]
    fig, ax = plt.subplots(figsize=(20, 14))
    sns.heatmap(
        pivot.astype(float),
        annot=False,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={
            "label": "−1 = CHA-S significantly worse · 0 = tie · +1 = CHA-S significantly better (p<0.05)",
            "shrink": 0.6,
        },
        ax=ax,
        linewidths=0.3,
        linecolor="white",
    )
    ax.set_title(
        "Diebold–Mariano test (CHA-S vs each rival, HLN small-sample correction, p<0.05)",
        fontsize=17,
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="x", labelsize=12, rotation=45)
    ax.set_xlabel("(dataset, horizon)", fontsize=14)
    ax.set_ylabel("rival baseline", fontsize=14)
    fig.tight_layout()
    out = FIG_DIR / "fig06_dm_test_grid.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# =========================================================================
# Figure 7 — prediction vs actual
# =========================================================================
def fig07_pred_vs_actual():
    """Sample prediction vs ground truth for the GEANT trace at h=6."""
    try:
        d_v3 = np.load("outputs/predictions/geant__cha_hybrid_v3__h6__s42.npz")
        d_bolt = np.load("outputs/predictions/geant__chronos_bolt_zs__h6__s42.npz")
    except FileNotFoundError as e:
        print(f"[fig07] prediction file missing: {e}")
        return None
    yt = d_v3["y_true_native"]  # (N, h)
    yp_v3 = d_v3["y_pred_native"]
    yp_bolt = d_bolt["y_pred_native"]
    # take the 1-step-ahead prediction at each test origin
    yt1, v3_1, b_1 = yt[:, 0], yp_v3[:, 0], yp_bolt[:, 0]
    n_show = min(200, yt1.size)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(n_show)
    ax.plot(x, yt1[:n_show], color="black", lw=1.3, label="Ground truth")
    ax.plot(x, v3_1[:n_show], color="C2", lw=1.0, label="CHA-S (proposed)")
    ax.plot(
        x,
        b_1[:n_show],
        color="C1",
        lw=0.7,
        alpha=0.7,
        label="Chronos-Bolt (best baseline)",
    )
    ax.set_title("Prediction vs actual — geant, h=6 (first 200 test windows)")
    ax.set_xlabel("Test window index")
    ax.set_ylabel("Traffic (kbps)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig07_pred_vs_actual.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# =========================================================================
# Figure 8 — STL decomposition illustration
# =========================================================================
def fig08_stl_decomposition():
    from statsmodels.tsa.seasonal import STL

    pp = load_preprocessed("cesnet")
    train = pp.split_native.train["value"].dropna().values[: 24 * 14]  # 2 weeks
    res = STL(train.astype(np.float64), period=24, robust=False).fit()
    fig, axes = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(train, color="black", lw=0.8)
    axes[0].set_title(
        "(a) Original cesnet aggregate bytes/h — first 14 days of training"
    )
    axes[1].plot(res.trend, color="C0", lw=1.2)
    axes[1].set_title("(b) Trend component (smooth)")
    axes[2].plot(res.seasonal, color="C2", lw=0.8)
    axes[2].set_title("(c) Seasonal component (daily, period=24)")
    axes[3].plot(res.resid, color="C3", lw=0.6)
    axes[3].set_title("(d) Residual component")
    axes[3].set_xlabel("Hour")
    for a in axes:
        a.grid(True, alpha=0.3)
    fig.suptitle(
        "STL decomposition used in CHA-Hybrid (cesnet illustration)", fontsize=12
    )
    fig.tight_layout()
    out = FIG_DIR / "fig08_stl_decomposition.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# =========================================================================
# Tables
# =========================================================================
def tables(df_runs):
    df_agg = aggregate(df_runs)
    # T1 — main results pivot
    rows = []
    for _, r in df_agg.iterrows():
        m = r["rmse_native_mean"]
        s = r["rmse_native_std"]
        n = r["rmse_native_count"]
        cell = f"{m:.3e}" if (pd.isna(s) or n <= 1) else f"{m:.3e} ± {s:.2e}"
        rows.append((r["variant"], r["dataset"], int(r["horizon"]), cell))
    t1 = pd.DataFrame(rows, columns=["variant", "dataset", "horizon", "rmse_native"])
    pivot = t1.pivot_table(
        index="variant",
        columns=["dataset", "horizon"],
        values="rmse_native",
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=pd.MultiIndex.from_product([DATASETS, HORIZONS]))
    pivot.to_csv(TABLE_DIR / "table_main_results.csv")

    # T2 — proposed wins
    df_dm = pairwise_dm_against_proposed(proposed=PROPOSED)
    if not df_dm.empty:
        out = []
        for rival in sorted(df_dm["variant_b"].unique()):
            sub = df_dm[df_dm["variant_b"] == rival]
            w = ((sub["statistic"] < 0) & sub["significant_at_005"]).sum()
            l = ((sub["statistic"] > 0) & sub["significant_at_005"]).sum()
            t = (~sub["significant_at_005"]).sum()
            out.append((rival, int(w), int(l), int(t)))
        t2 = pd.DataFrame(out, columns=["rival", "v3_wins", "v3_losses", "ties"])
        t2.to_csv(TABLE_DIR / "table_proposed_wins.csv", index=False)

    # T3 — cost
    t3 = cost_table(df_runs)
    t3.to_csv(TABLE_DIR / "table_cost.csv")

    # T4 — ablation
    abl = [
        "cha_hybrid_v3",
        "cha_hybrid_v3_decomp_only",
        "cha_hybrid_v3_global_only",
        "cha_hybrid_v3_fixed_alpha_0.5",
        "cha_hybrid_v3_altres_gru",
    ]
    sub = df_agg[df_agg["variant"].isin(abl)]
    if not sub.empty:
        t4 = sub.pivot_table(
            index="variant", columns=["dataset", "horizon"], values="rmse_native_mean"
        )
        t4 = t4.reindex(columns=pd.MultiIndex.from_product([DATASETS, HORIZONS]))
        t4.to_csv(TABLE_DIR / "table_ablation.csv")
    return TABLE_DIR


# =========================================================================
def main() -> int:
    df_runs = load_all_runs()
    print(f"loaded {len(df_runs)} runs")
    print()
    print("Generating figures …")
    for fn in [
        fig01_rank_heatmap,
        fig02_per_horizon_rmse,
        fig03_ablation,
        fig04_alpha_distribution,
        fig05_cost_vs_accuracy,
        fig06_dm_test_grid,
        fig07_pred_vs_actual,
        fig08_stl_decomposition,
    ]:
        try:
            out = fn(df_runs) if "df_runs" in fn.__code__.co_varnames else fn()
            print(f"  {fn.__name__:<32s} → {out}")
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  {fn.__name__:<32s} FAILED: {e}")
    print()
    print("Generating tables …")
    out_dir = tables(df_runs)
    print(f"  tables → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
