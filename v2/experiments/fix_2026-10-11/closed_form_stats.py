"""audit item 4 (+ item 1 floor count, Abilene ratio, NAB AWS-CPU estimators).

Inputs (no retraining, CPU only):
  outputs/checkpoints/<ds>__<variant>__h<h>__s<seed>.pt.json  (validation grid)
  outputs/predictions/*cha_hybrid_v3*, *decomp_only*, *chronos_bolt_zs*  (test)

What it computes
  A. Quadratic identity. The mixture validation MSE is exactly quadratic:
       M(a) = (1-a)(1-2a) M(0) + 4a(1-a) M(1/2) + a(2a-1) M(1).
     The 21-point grid trace is checked against this identity (max relative
     error, expected ~1e-6 from float32 RMSE storage). Consequence: the closed
     form is the EXACT minimiser of M on [0,1] and the 0.05 grid is the
     approximation; |a_clip - a_grid| <= 0.025 (half the grid step) holds by
     construction.
  B. Unclipped vs clipped (Algorithm 2) recovery on the 825 v3-family cells:
     legacy numbers (unclipped: mean |da| 0.029, CI, worst 0.179, Wilcoxon)
     are reproduced for traceability; the Algorithm-2 numbers (clipped mean,
     max, count of negative a*, max validation dMSE) are the ones to report.
  C. Same for the generation sweep (6 experts, seed 42).
  D. Closed-form CHA-S TEST accuracy on all 25 cells (5 seeds): the weight is
     the clipped closed form from validation, the forecast is
     a*dec + (1-a)*bolt on the saved test forecasts. The grid-alpha version
     is recomputed the same way and checked against the saved CHA-S file.
  E. Floor bound out of sample: per cell, is seed-mean CHA-S test RMSE <=
     min(seed-mean decomp-only RMSE, Bolt RMSE)? Count, excess per cell,
     largest excess, count within 0.5 %. Grid alpha and closed-form alpha.
  F. Abilene (all datasets) decomposition/foundation error ratio on
     validation (MSE and RMSE) and on test.
  G. Alpha estimators side by side (explains NAB AWS-CPU 0 vs 0.66): grid
     alpha (validation, per horizon, flattened), closed form on validation,
     in-sample closed form on the full test split, and the monitor's rolling
     estimator (W=60 horizon-averaged test windows, mean of the first 40).

Outputs (this folder, never overwriting):
  closed_form_cells.csv, closed_form_generation.csv, closed_form_test_accuracy.csv,
  floor_bound_test.csv, decomp_ratio.csv, alpha_estimators.csv,
  closed_form_summary.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BOLT,
    CHAS,
    DATASETS,
    DEC,
    GRID_STEP,
    HORIZONS,
    SEEDS,
    SWAP_VARIANTS,
    V3_VARIANTS,
    align,
    bg_from_errors,
    bg_from_three,
    ckpt_json,
    load_bolt,
    load_pred,
    quad_M,
    rmse,
    three_points,
    val_grid,
    write_csv,
    write_json,
)

W_MON, WARMUP_MON = 60, 40  # expR6 monitor constants, for section G


def cell_row(ds, variant, h, seed):
    ck = ckpt_json(ds, variant, h, seed)
    if ck is None:
        return None
    g = val_grid(ck)
    if g is None:
        return None
    alphas, mses = g
    m0, mh, m1 = three_points(alphas, mses)
    a_star, a_clip, A = bg_from_three(m0, mh, m1)
    pred = quad_M(alphas, m0, mh, m1)
    rel = np.abs(pred - mses) / np.maximum(mses, 1e-12)
    a_grid = float(ck.get("alpha_h"))
    m_grid = float(mses[int(np.argmin(np.abs(alphas - a_grid)))])
    return {
        "dataset": ds,
        "variant": variant,
        "horizon": h,
        "seed": seed,
        "alpha_grid": a_grid,
        "alpha_bg_unclipped": a_star,
        "alpha_bg_clipped": a_clip,
        "gap_unclipped": abs(a_grid - a_star) if np.isfinite(a_star) else np.nan,
        "gap_clipped": abs(a_grid - a_clip),
        "alpha_iv": m0 / (m0 + m1),
        "gap_iv": abs(a_grid - m0 / (m0 + m1)),
        "val_mse_grid": m_grid,
        "val_mse_closed_form": float(quad_M(a_clip, m0, mh, m1)),
        "val_dmse_grid_minus_cf": m_grid - float(quad_M(a_clip, m0, mh, m1)),
        "quad_identity_max_rel_err": float(rel.max()),
        "leading_coef_A": A,
        "M0": m0,
        "Mhalf": mh,
        "M1": m1,
    }


def summarize_gaps(df: pd.DataFrame, label: str) -> dict:
    d = df.dropna(subset=["alpha_bg_unclipped"])
    diff_unc = (d["alpha_grid"] - d["alpha_bg_unclipped"]).to_numpy()
    rng = np.random.default_rng(42)
    absd = np.abs(diff_unc)
    boot = np.array([np.mean(rng.choice(absd, size=absd.size)) for _ in range(2000)])
    w = stats.wilcoxon(diff_unc, alternative="two-sided") if absd.size > 10 else None
    return {
        "label": label,
        "n_cells": int(len(df)),
        "n_degenerate_A_le_0": int(df["alpha_bg_unclipped"].isna().sum()),
        "quad_identity_max_rel_err": float(df["quad_identity_max_rel_err"].max()),
        "legacy_unclipped": {
            "mean_abs_gap": float(absd.mean()),
            "ci95_boot": [
                float(np.percentile(boot, 2.5)),
                float(np.percentile(boot, 97.5)),
            ],
            "max_abs_gap": float(absd.max()),
            "n_negative_alpha_star": int((d["alpha_bg_unclipped"] < 0).sum()),
            "n_alpha_star_gt_1": int((d["alpha_bg_unclipped"] > 1).sum()),
            "wilcoxon_p": float(w.pvalue) if w is not None else None,
            "median_signed_grid_minus_bg": float(np.median(diff_unc)),
            "note": "compares the UNCLIPPED BG value with the grid; not Algorithm 2",
        },
        "algorithm2_clipped": {
            "mean_abs_gap": float(df["gap_clipped"].mean()),
            "max_abs_gap": float(df["gap_clipped"].max()),
            "n_gap_gt_half_step": int((df["gap_clipped"] > GRID_STEP / 2 + 1e-6).sum()),
            "max_val_dmse_grid_minus_cf": float(df["val_dmse_grid_minus_cf"].max()),
            "min_val_dmse_grid_minus_cf": float(df["val_dmse_grid_minus_cf"].min()),
            "bound_by_construction": GRID_STEP / 2,
        },
        "share_alpha_grid_le_0.05": float((df["alpha_grid"] <= 0.05 + 1e-9).mean()),
        "inverse_variance_mean_abs_gap": float(df["gap_iv"].mean()),
    }


def rolling_alpha_mean(y, fd, ff, W=W_MON, warm=WARMUP_MON):
    """expR6 estimator: horizon-averaged series, rolling W windows, clipped."""
    ed, ef = fd - y, ff - y
    diff = ed - ef
    a = []
    for t in range(W, len(y)):
        sl = slice(t - W, t)
        den = float(np.dot(diff[sl], diff[sl]))
        a.append(
            0.0
            if den < 1e-12
            else float(np.clip(-np.dot(ef[sl], diff[sl]) / den, 0, 1))
        )
    a = np.array(a)
    return float(a[:warm].mean()), float(a.mean())


def main():
    # ---------------- A/B: 825 cells
    rows = []
    for v in V3_VARIANTS:
        for ds in DATASETS:
            for h in HORIZONS:
                for s in SEEDS:
                    r = cell_row(ds, v, h, s)
                    if r is not None:
                        rows.append(r)
    cells = pd.DataFrame(rows)
    write_csv(cells, "closed_form_cells.csv")
    summary = {"v3_family": summarize_gaps(cells, "7 v3 variants")}
    per_h = (
        cells.groupby("horizon")
        .agg(
            alpha_grid=("alpha_grid", "mean"),
            alpha_bg_unclipped=("alpha_bg_unclipped", "mean"),
            alpha_bg_clipped=("alpha_bg_clipped", "mean"),
            gap_unclipped=("gap_unclipped", "mean"),
            gap_clipped=("gap_clipped", "mean"),
            alpha_iv=("alpha_iv", "mean"),
            gap_iv=("gap_iv", "mean"),
        )
        .reset_index()
    )
    summary["v3_family_per_horizon"] = per_h.to_dict(orient="records")

    # ---------------- C: generation sweep
    grows = []
    for expert, v in SWAP_VARIANTS.items():
        for ds in DATASETS:
            for h in HORIZONS:
                r = cell_row(ds, v, h, 42)
                if r is not None:
                    r["expert"] = expert
                    grows.append(r)
    gen = pd.DataFrame(grows)
    if not gen.empty:
        write_csv(gen, "closed_form_generation.csv")
        summary["generation"] = {
            e: {
                "n": int(len(g)),
                "mean_gap_clipped": float(g["gap_clipped"].mean()),
                "max_gap_clipped": float(g["gap_clipped"].max()),
                "share_alpha_grid_le_0.05": float(
                    (g["alpha_grid"] <= 0.05 + 1e-9).mean()
                ),
            }
            for e, g in gen.groupby("expert")
        }
        summary["generation_all"] = {
            "mean_gap_clipped": float(gen["gap_clipped"].mean()),
            "max_gap_clipped": float(gen["gap_clipped"].max()),
            "note": "<= 0.025 by construction (exact quadratic, 0.05 grid incl. endpoints)",
        }

    # ---------------- D/E: test accuracy and floor bound
    trows, recon = [], []
    for ds in DATASETS:
        for h in HORIZONS:
            b = load_bolt(ds, h)
            for s in SEEDS:
                c = cells[
                    (cells.dataset == ds)
                    & (cells.variant == CHAS)
                    & (cells.horizon == h)
                    & (cells.seed == s)
                ]
                dd = load_pred(ds, DEC, h, s)
                v3 = load_pred(ds, CHAS, h, s)
                if c.empty or dd is None or b is None or v3 is None:
                    continue
                y, fdec, fb, fv3 = align(dd[0], dd[1], b[1], v3[1])
                a_g = float(c.alpha_grid.iloc[0])
                a_c = float(c.alpha_bg_clipped.iloc[0])
                p_grid = a_g * fdec + (1 - a_g) * fb
                recon.append(float(np.max(np.abs(p_grid - fv3))))
                trows.append(
                    {
                        "dataset": ds,
                        "horizon": h,
                        "seed": s,
                        "alpha_grid": a_g,
                        "alpha_cf": a_c,
                        "rmse_chas_saved": rmse(y, fv3),
                        "rmse_chas_grid_recomputed": rmse(y, p_grid),
                        "rmse_chas_closed_form": rmse(y, a_c * fdec + (1 - a_c) * fb),
                        "rmse_decomp_only": rmse(y, fdec),
                        "rmse_bolt": rmse(y, fb),
                        "alpha_test_insample": bg_from_errors(y, fdec, fb),
                    }
                )
    acc = pd.DataFrame(trows)
    write_csv(acc, "closed_form_test_accuracy.csv")
    summary["grid_reconstruction_max_abs_diff"] = float(max(recon)) if recon else None

    cell = acc.groupby(["dataset", "horizon"]).mean(numeric_only=True).reset_index()
    frows = []
    for _, r in cell.iterrows():
        best = min(r.rmse_decomp_only, r.rmse_bolt)
        frows.append(
            {
                "dataset": r.dataset,
                "horizon": int(r.horizon),
                "rmse_chas_grid": r.rmse_chas_saved,
                "rmse_chas_cf": r.rmse_chas_closed_form,
                "rmse_decomp": r.rmse_decomp_only,
                "rmse_bolt": r.rmse_bolt,
                "better_endpoint": "decomp"
                if r.rmse_decomp_only < r.rmse_bolt
                else "bolt",
                "excess_pct_grid": 100 * (r.rmse_chas_saved / best - 1),
                "excess_pct_cf": 100 * (r.rmse_chas_closed_form / best - 1),
                "floor_holds_grid": bool(r.rmse_chas_saved <= best),
                "floor_holds_cf": bool(r.rmse_chas_closed_form <= best),
                "cf_minus_grid_pct": 100
                * (r.rmse_chas_closed_form / r.rmse_chas_saved - 1),
            }
        )
    floor = pd.DataFrame(frows)
    write_csv(floor, "floor_bound_test.csv")
    for tag in ("grid", "cf"):
        ex = floor[f"excess_pct_{tag}"]
        worst = floor.loc[ex.idxmax()]
        summary[f"floor_test_{tag}"] = {
            "cells": int(len(floor)),
            "holds": int(floor[f"floor_holds_{tag}"].sum()),
            "violations": [
                f"{r.dataset} h={int(r.horizon)}"
                for r in floor.loc[~floor[f"floor_holds_{tag}"]].itertuples()
            ],
            "largest_excess_pct": float(ex.max()),
            "largest_excess_cell": f"{worst.dataset} h={int(worst.horizon)}",
            "within_0.5pct": int((ex <= 0.5).sum()),
        }
    summary["closed_form_vs_grid_test"] = {
        "mean_cf_minus_grid_pct": float(floor["cf_minus_grid_pct"].mean()),
        "max_abs_cf_minus_grid_pct": float(floor["cf_minus_grid_pct"].abs().max()),
    }

    # ---------------- F: decomposition / foundation ratio
    rrows = []
    for ds in DATASETS:
        for h in HORIZONS:
            c = cells[
                (cells.dataset == ds) & (cells.variant == CHAS) & (cells.horizon == h)
            ]
            a = acc[(acc.dataset == ds) & (acc.horizon == h)]
            if c.empty:
                continue
            ratio_mse = (c.M1 / c.M0).to_numpy()
            rrows.append(
                {
                    "dataset": ds,
                    "horizon": h,
                    "val_mse_ratio_dec_over_bolt_mean": float(ratio_mse.mean()),
                    "val_mse_ratio_min": float(ratio_mse.min()),
                    "val_mse_ratio_max": float(ratio_mse.max()),
                    "val_rmse_ratio_mean": float(np.sqrt(ratio_mse).mean()),
                    "test_rmse_ratio_dec_over_bolt": float(
                        (a.rmse_decomp_only / a.rmse_bolt).mean()
                    )
                    if not a.empty
                    else np.nan,
                }
            )
    ratio = pd.DataFrame(rrows)
    write_csv(ratio, "decomp_ratio.csv")
    ab = ratio[ratio.dataset == "abilene"]
    summary["abilene_decomp_ratio"] = {
        "val_mse_ratio_range": [
            float(ab.val_mse_ratio_min.min()),
            float(ab.val_mse_ratio_max.max()),
        ],
        "val_rmse_ratio_range": [
            float(ab.val_rmse_ratio_mean.min()),
            float(ab.val_rmse_ratio_mean.max()),
        ],
        "test_rmse_ratio_range": [
            float(ab.test_rmse_ratio_dec_over_bolt.min()),
            float(ab.test_rmse_ratio_dec_over_bolt.max()),
        ],
    }

    # ---------------- G: alpha estimators
    erows = []
    for ds in DATASETS:
        for h in HORIZONS:
            c = cells[
                (cells.dataset == ds) & (cells.variant == CHAS) & (cells.horizon == h)
            ]
            a = acc[(acc.dataset == ds) & (acc.horizon == h)]
            row = {
                "dataset": ds,
                "horizon": h,
                "alpha_grid_val_mean": float(c.alpha_grid.mean())
                if not c.empty
                else np.nan,
                "alpha_cf_val_mean": float(c.alpha_bg_clipped.mean())
                if not c.empty
                else np.nan,
                "alpha_cf_test_insample_mean": float(a.alpha_test_insample.mean())
                if not a.empty
                else np.nan,
            }
            dd, b = load_pred(ds, DEC, h, 42), load_bolt(ds, h)
            if dd is not None and b is not None:
                y, fdec, fb = align(dd[0], dd[1], b[1])
                warm, full = rolling_alpha_mean(
                    y.mean(axis=1), fdec.mean(axis=1), fb.mean(axis=1)
                )
                row["alpha_rolling_monitor_first40_s42"] = warm
                row["alpha_rolling_monitor_all_s42"] = full
            erows.append(row)
    est = pd.DataFrame(erows)
    write_csv(est, "alpha_estimators.csv")
    nab = est[(est.dataset == "nab_aws_cpu") & (est.horizon == 6)]
    if not nab.empty:
        summary["nab_aws_cpu_h6_estimators"] = nab.iloc[0].to_dict()

    write_json(summary, "closed_form_summary.json")


if __name__ == "__main__":
    main()
