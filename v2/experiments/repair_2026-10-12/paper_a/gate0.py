"""A-A1 Gate 0 (headroom). VALIDATION DATA ONLY. Definition frozen in REPAIR_PREREG.md section A-A1/Gate 0.

Gate 0a  cross-fitted (leave-block-out) headroom per cell; in-sample oracle reported as an upper bound only
Gate 0b  third-expert (persistence) stacking headroom, CAL -> SEL
Drift    one-parameter transfer loss between the two validation halves (reported, not a rule)

Usage: python3 gate0.py [--dry-run]
Outputs: gate0_cells.csv, gate0_summary.json (this folder).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import rcommon as R

L_PRIMARY, L_SENS = 48, (24, 96)
THRESH = 0.005  # 0.5 % of RMSE
MIN_CELLS = 6  # gate passes if at least 6/25 cells have headroom (i.e. NOT <0.5 % in >= 20/25)


def block_edges(n, L):
    nb = max(1, n // L)
    e = [i * L for i in range(nb)] + [n]
    return e


def oracle_block_mse(see, sed, sdd, h, L):
    """In-sample per-block oracle (upper bound only, optimistic by construction)."""
    n = len(see)
    e = block_edges(n, L)
    tot = 0.0
    for a, b in zip(e[:-1], e[1:]):
        al = R.alpha_bg(sed[a:b], sdd[a:b])
        tot += see[a:b].sum() + 2 * al * sed[a:b].sum() + al * al * sdd[a:b].sum()
    return tot / (n * h)


def crossfit_block_mse(see, sed, sdd, h, L, mode):
    """Block b is scored with alpha estimated WITHOUT block b.

    mode 'nb'    : alpha from the two adjacent blocks (b-1, b+1), h windows purged next to b
    mode 'prev'  : alpha from block b-1 only (causal; block 0 uses the global alpha of the other blocks)
    """
    n = len(see)
    e = block_edges(n, L)
    nb = len(e) - 1
    tot = 0.0
    for b in range(nb):
        a0, b0 = e[b], e[b + 1]
        idx = []
        if mode == "nb":
            if b > 0:
                idx.append(np.arange(e[b - 1], max(e[b - 1], a0 - h)))
            if b < nb - 1:
                idx.append(np.arange(min(e[b + 2], b0 + h), e[b + 2]))
        else:
            if b > 0:
                idx.append(np.arange(e[b - 1], max(e[b - 1], a0 - h)))
        idx = np.concatenate(idx) if idx else np.arange(0)
        if idx.size == 0:  # fall back to every other block
            idx = np.concatenate([np.arange(e[j], e[j + 1]) for j in range(nb) if j != b] or [np.arange(0)])
        al = R.alpha_bg(sed[idx], sdd[idx]) if idx.size else 0.0
        tot += see[a0:b0].sum() + 2 * al * sed[a0:b0].sum() + al * al * sdd[a0:b0].sum()
    return tot / (n * h)


def nnls3(Xd, yd, iters=500):
    """min ||yd - Xd w||^2, w >= 0 by projected gradient (3 columns, tiny)."""
    w = np.zeros(Xd.shape[1])
    G = Xd.T @ Xd
    c = Xd.T @ yd
    lip = np.linalg.eigvalsh(G).max() + 1e-12
    for _ in range(iters):
        w = np.maximum(0.0, w - (G @ w - c) / lip)
    return w


def cell_gate(ds, h, rng_seed=0):
    rows = []
    v0 = R.load_val(ds, h, R.SEEDS[0])
    n = v0["y"].shape[0]
    per_seed = []
    for s in R.SEEDS:
        v = R.load_val(ds, h, s)
        y, dec, glob = v["y"], v["dec"], v["glob"]
        see, sed, sdd = R.window_stats(y, dec, glob)
        a_dep = R.deployed_alpha(ds, h, s)
        mse_const = R.mse_from_stats(a_dep, see, sed, sdd, h)
        mse_bolt = see.sum() / (n * h)
        rec = {"seed": s, "alpha_dep": a_dep, "rmse_const": np.sqrt(mse_const), "rmse_bolt": np.sqrt(mse_bolt)}
        for L in (L_PRIMARY,) + L_SENS:
            ob = oracle_block_mse(see, sed, sdd, h, L)
            rec[f"insample_L{L}"] = 1.0 - np.sqrt(ob / mse_const)
            rec[f"xfit_nb_L{L}"] = 1.0 - np.sqrt(crossfit_block_mse(see, sed, sdd, h, L, "nb") / mse_const)
            rec[f"xfit_prev_L{L}"] = 1.0 - np.sqrt(crossfit_block_mse(see, sed, sdd, h, L, "prev") / mse_const)
        # drift transfer between halves (one parameter, no noise correction needed)
        m = n // 2
        a1 = R.alpha_bg(sed[:m], sdd[:m])
        a2 = R.alpha_bg(sed[m:], sdd[m:])
        def part(a, sl):
            return R.mse_from_stats(a, see[sl], sed[sl], sdd[sl], h)
        loss12 = np.sqrt(part(a1, slice(m, n)) / part(a2, slice(m, n))) - 1.0
        loss21 = np.sqrt(part(a2, slice(0, m)) / part(a1, slice(0, m))) - 1.0
        rec["drift_transfer_loss"] = 0.5 * (loss12 + loss21)
        rec["alpha_half1"], rec["alpha_half2"] = a1, a2
        # Gate 0b: persistence as third expert, CAL (first 60 %) -> SEL (last 40 % after purge h)
        gap = h
        ntrain = int(0.6 * n)
        sel = np.arange(ntrain + gap, n)
        cal = np.arange(1, ntrain)  # window 0 has no persistence value
        per = np.repeat(y[:-1, :1], h, axis=1)  # persistence of window t = y[t-1,0]
        per = np.vstack([per[:1], per])  # window 0 filled with its successor (never used in cal/sel)
        def fit_pred(idx_fit, idx_pred):
            D = np.stack([(dec - glob)[idx_fit].ravel(), (per - glob)[idx_fit].ravel()], axis=1)
            w = nnls3(D, (y - glob)[idx_fit].ravel())
            if w.sum() > 1:
                w = w / w.sum()
            P = glob[idx_pred] + w[0] * (dec - glob)[idx_pred] + w[1] * (per - glob)[idx_pred]
            return P, w
        P3, w3 = fit_pred(cal, sel)
        a_cal = R.alpha_bg(sed[cal], sdd[cal])
        P2 = R.mix(dec[sel], glob[sel], a_cal)
        rec["sel_rmse_3exp"], rec["sel_rmse_2exp"] = R.rmse(y[sel], P3), R.rmse(y[sel], P2)
        rec["sel_rmse_bolt"] = R.rmse(y[sel], glob[sel])
        rec["gate0b_gain"] = 1.0 - rec["sel_rmse_3exp"] / rec["sel_rmse_2exp"]
        rec["w_persist"] = w3[1]
        per_seed.append(rec)
    out = {"dataset": ds, "horizon": h, "n_val": n}
    for L in (L_PRIMARY,) + L_SENS:
        for k in ("insample", "xfit_nb", "xfit_prev"):
            out[f"{k}_L{L}"] = float(np.mean([r[f"{k}_L{L}"] for r in per_seed]))
    out["const_vs_bolt"] = 1.0 - np.mean([r["rmse_const"] / r["rmse_bolt"] for r in per_seed])
    out["xfit_vs_bolt"] = 1.0 - (1.0 - out["const_vs_bolt"]) * (1.0 - out[f"xfit_nb_L{L_PRIMARY}"])
    out["drift_transfer_loss"] = np.mean([r["drift_transfer_loss"] for r in per_seed])
    out["alpha_dep_mean"] = np.mean([r["alpha_dep"] for r in per_seed])
    out["gate0b_gain"] = np.mean([r["gate0b_gain"] for r in per_seed])
    out["gate0b_gain_sd"] = np.std([r["gate0b_gain"] for r in per_seed])
    out["w_persist"] = np.mean([r["w_persist"] for r in per_seed])
    out["has_headroom"] = bool(out[f"xfit_nb_L{L_PRIMARY}"] >= THRESH)
    out["has_3exp_headroom"] = bool(out["gate0b_gain"] >= THRESH)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    log = R.install_guard()
    cells = R.CELLS[:2] if a.dry_run else R.CELLS
    rows = []
    for i, (ds, h) in enumerate(cells):
        rows.append(cell_gate(ds, h))
        R.progress(f"gate0 {i + 1}/{len(cells)} {ds} h{h}")
        print(ds, h, round(rows[-1][f"xfit_nb_L{L_PRIMARY}"], 4), rows[-1]["has_headroom"], flush=True)
    df = pd.DataFrame(rows)
    assert not any("outputs/predictions" in p for p in log), "test data touched"
    if a.dry_run:
        print("dry-run ok; test files read:", 0)
        return
    df.to_csv(R.HERE / "gate0_cells.csv", index=False)
    n_head = int(df.has_headroom.sum())
    summ = {
        "n_cells": len(df),
        "cells_with_xfit_headroom_ge_0.5pct": n_head,
        "gate0a_pass_run_V1_V4": bool(n_head >= MIN_CELLS),
        "mean_xfit_nb_L48": float(df["xfit_nb_L48"].mean()),
        "median_xfit_nb_L48": float(df["xfit_nb_L48"].median()),
        "max_xfit_nb_L48": float(df["xfit_nb_L48"].max()),
        "mean_xfit_prev_L48": float(df["xfit_prev_L48"].mean()),
        "mean_insample_oracle_L48": float(df["insample_L48"].mean()),
        "cells_xfit_nb_ge_0.5pct_L24": int((df["xfit_nb_L24"] >= THRESH).sum()),
        "cells_xfit_nb_ge_0.5pct_L96": int((df["xfit_nb_L96"] >= THRESH).sum()),
        "cells_xfit_prev_ge_0.5pct_L48": int((df["xfit_prev_L48"] >= THRESH).sum()),
        "mean_drift_transfer_loss": float(df.drift_transfer_loss.mean()),
        "cells_drift_loss_ge_0.5pct": int((df.drift_transfer_loss >= THRESH).sum()),
        "mean_const_vs_bolt_gain_val": float(df.const_vs_bolt.mean()),
        "mean_xfit_gain_vs_bolt_val": float(df.xfit_vs_bolt.mean()),
        "gate0b_cells_ge_0.5pct": int(df.has_3exp_headroom.sum()),
        "gate0b_pass_run_V5": bool(df.has_3exp_headroom.sum() >= MIN_CELLS),
        "gate0b_mean_gain": float(df.gate0b_gain.mean()),
        "per_dataset_xfit_nb_L48": df.groupby("dataset")["xfit_nb_L48"].mean().to_dict(),
    }
    (R.HERE / "gate0_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
