"""Round 2 Gate E0: validation-only gate for a candidate complementary expert E in the mixture with Chronos-Bolt.
Definition frozen in REPAIR_PREREG_2.md section 3. Inputs: val_experts/*.npz (validation forecasts of E, trained on TRAIN windows only)
and the stored validation forecasts of Bolt and of the decomposition path (5 seeds). No test file is read (guard installed).

Per cell (25), per expert:
  r      = MSE_E / MSE_Bolt on CAL;  rho = correlation of the two error series on CAL (pooled over steps)
  g_in   = in-sample gain of the best [0,1] mixture of Bolt and E over Bolt on CAL (oracle ceiling)
  g_oos  = gain on SEL of the mixture with alpha fitted on CAL (grid 0.05) over Bolt
  g_3    = gain on SEL of NNLS stacking [Bolt, dec_seed, E] fitted on CAL, over Bolt, mean over 5 seeds
  xfit   = cross-fitted block headroom (gate 0a, L = 48) of the [Bolt, E] pair on the whole validation slice
Gate E0 passes iff  (a) r <= 1.2 in >= 15 cells,  (b) g_in >= 1.0 % in >= 8 cells,  (c) g_oos >= 0.5 % in >= 6 cells.
Reported, not required: rho, g_3 (cells >= 0.5 %), xfit (cells >= 0.5 %).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rcommon as R  # noqa: E402
import gate0 as G0  # noqa: E402

EXPERTS = ["EQ", "ER", "EB", "EX"]
NULLS = ["EQ0", "EB0"]
MINC = {"a": 15, "b": 8, "c": 6}


def stats_pair(y, a, b):
    """see, sed, sdd for the mixture f = b + alpha (a - b) with b the anchor (Bolt)."""
    return R.window_stats(y, a, b)


def nnls(Xd, yd, iters=400):
    w = np.zeros(Xd.shape[1])
    Gm, c = Xd.T @ Xd, Xd.T @ yd
    lip = np.linalg.eigvalsh(Gm).max() + 1e-12
    for _ in range(iters):
        w = np.maximum(0.0, w - (Gm @ w - c) / lip)
    return w


def cell(ds, h, ex, dry=False):

    d = np.load(HERE / "val_experts" / f"{ds}__h{h}.npz")
    v0 = R._raw_val(ds, h, R.SEEDS[0])
    n = min(len(d["y_val"]), len(v0["y"]))
    y, B, E = v0["y"][:n].astype(np.float64), v0["glob"][:n].astype(np.float64), d[ex][:n].astype(np.float64)
    nc, s0 = R.split_val(n, h)
    cal, sel = np.arange(nc), np.arange(s0, n)
    see, sed, sdd = stats_pair(y, E, B)
    mseB_c, mseE_c = np.mean((B[cal] - y[cal]) ** 2), np.mean((E[cal] - y[cal]) ** 2)
    eB, eE = (B[cal] - y[cal]).ravel(), (E[cal] - y[cal]).ravel()
    rho = float(np.corrcoef(eB, eE)[0, 1])
    a_cal = R.alpha_bg(sed[cal], sdd[cal])
    g_in = 1 - np.sqrt(R.mse_from_stats(a_cal, see[cal], sed[cal], sdd[cal], h) / mseB_c)
    a_s = R.snap(a_cal)
    mix_sel = R.mix(E[sel], B[sel], a_s)
    g_oos = 1 - R.rmse(y[sel], mix_sel) / R.rmse(y[sel], B[sel])
    # three-expert stacking per decomposition seed
    g3 = []
    for s in R.SEEDS:
        v = R._raw_val(ds, h, s)
        D = v["dec"][:n].astype(np.float64)
        X = np.stack([(D - B)[cal].ravel(), (E - B)[cal].ravel()], 1)
        w = nnls(X, (y - B)[cal].ravel())
        if w.sum() > 1:
            w = w / w.sum()
        P = B[sel] + w[0] * (D - B)[sel] + w[1] * (E - B)[sel]
        g3.append(1 - R.rmse(y[sel], P) / R.rmse(y[sel], B[sel]))
    # cross-fitted block headroom of the pair on the whole validation slice (const = alpha on all windows, snapped)
    a_all = R.snap(R.alpha_bg(sed, sdd))
    mse_const = R.mse_from_stats(a_all, see, sed, sdd, h)
    xfit = 1 - np.sqrt(G0.crossfit_block_mse(see, sed, sdd, h, G0.L_PRIMARY, "nb") / mse_const)
    return {"dataset": ds, "horizon": h, "expert": ex, "n": n, "r": float(mseE_c / mseB_c), "rho": rho, "alpha_cal": a_s, "g_in": float(g_in),
            "g_oos": float(g_oos), "g_3": float(np.mean(g3)), "xfit": float(xfit), "rmse_E_over_B_sel": float(R.rmse(y[sel], E[sel]) / R.rmse(y[sel], B[sel]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    log = R.install_guard()
    R.guard_selftest()
    cells = [c for c in R.CELLS if (HERE / "val_experts" / f"{c[0]}__h{c[1]}.npz").exists()]
    if a.dry_run:
        cells = cells[:2]
    rows = []
    for ex in EXPERTS + NULLS:
        for ds, h in cells:
            rows.append(cell(ds, h, ex, a.dry_run))
    assert not any("predictions" in p for p in log), "test data touched"
    df = pd.DataFrame(rows)
    if a.dry_run:
        print(df.round(4).to_string())
        print("dry-run ok; test files read: 0")
        return
    df.to_csv(HERE / "gate_e0_cells.csv", index=False)
    summ = {}
    for ex in EXPERTS + NULLS:
        d = df[df.expert == ex]
        a_ = int((d.r <= 1.2).sum()); b_ = int((d.g_in >= 0.01).sum()); c_ = int((d.g_oos >= 0.005).sum())
        summ[ex] = {"n_cells": len(d), "a_cells_r_le_1.2": a_, "b_cells_gin_ge_1pct": b_, "c_cells_goos_ge_0.5pct": c_,
                    "pass": bool(len(d) == 25 and a_ >= MINC["a"] and b_ >= MINC["b"] and c_ >= MINC["c"]),
                    "median_r": float(d.r.median()), "median_rho": float(d.rho.median()), "cells_r_le_1.2_and_rho_le_0.8": int(((d.r <= 1.2) & (d.rho <= 0.8)).sum()),
                    "mean_g_in": float(d.g_in.mean()), "max_g_in": float(d.g_in.max()), "mean_g_oos": float(d.g_oos.mean()), "max_g_oos": float(d.g_oos.max()),
                    "cells_g3_ge_0.5pct": int((d.g_3 >= 0.005).sum()), "mean_g3": float(d.g_3.mean()), "cells_xfit_ge_0.5pct": int((d.xfit >= 0.005).sum()), "mean_xfit": float(d.xfit.mean())}
    passing = [e for e in EXPERTS if e != "EX" and summ[e]["pass"]]
    best = max(passing, key=lambda e: summ[e]["mean_g_oos"]) if passing else None
    # test candidates: the passing experts and, by explicit decision documented in REPAIR_PREREG_2.md section 4, EQ (misses condition (a) only)
    cands = list(passing)
    if "EQ" not in cands:
        cands.append("EQ")
    out = {"minimum_cells": MINC, "experts": summ, "passing_headline_candidates": passing, "best": best, "test_candidates": cands,
           "note": "EX is a reference and is not eligible; null experts are reported for scale; EQ is a secondary candidate that misses (a) only"}
    (HERE / "gate_e0_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
