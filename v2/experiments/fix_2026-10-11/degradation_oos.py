"""audit item 5: degradation sweep with alpha fitted on VALIDATION, scored on TEST.

pipeline/expR5_alpha_stretch.py degraded the foundation forecast on the test
split, fitted the closed-form alpha on those same degraded test forecasts and
scored the hybrid on them: an in-sample oracle. "Hybrid strictly below the
degraded foundation" is then Proposition 1 applied in sample, not evidence.

Out-of-sample protocol here (h = 6 by default, all horizons optional):
  scenario: the foundation expert is degraded in the deployment regime, i.e.
  on validation AND on test, with independent noise draws
      f_glob~ = f_glob + gamma * s * z,  z ~ N(0, 1)
  fit:   alpha(gamma) = clipped Bates-Granger on the degraded VALIDATION split
  score: hybrid = alpha * f_dec + (1 - alpha) * f_glob~ on the degraded TEST
         split; compare with the degraded foundation, the decomposition path
         and the better of the two (out-of-sample floor check).

Two fitting paths:
  A "val_forecasts": uses experiments/fix_2026-10-11/val_forecasts/ (step 0);
    s = std of the foundation residual on validation; seeded z on validation.
  B "val_moments" (always available, no model run): from the validation grid
    in the checkpoint JSON. With independent zero-mean noise of variance
    v = gamma^2 s^2, the expected validation curve is
        M~(0) = M(0) + v,  M~(1/2) = M(1/2) + v/4,  M~(1) = M(1),
    and alpha(gamma) is the clipped closed form of that curve. s^2 = M(0)
    (foundation validation MSE; equals the residual variance up to the
    squared bias). Also used for the Chronos-2 / TiRex panel (swap
    checkpoints, seed 42 only).
Test-side noise: s_test = std of the foundation residual on test (as legacy),
R replicates per (dataset, seed, gamma).

Legacy replica: expR5 is re-run bit-for-bit (RNG 2024, same loop order) to
check alpha_stretch.csv and to show the in-sample alpha next to the OOS one.

Outputs: degradation_oos_runs.csv, degradation_oos_cells.csv,
         degradation_legacy_check.csv, degradation_oos_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BOLT,
    DATASETS,
    DEC,
    EVAL,
    SEEDS,
    align,
    bg_from_errors,
    bg_from_three,
    ckpt_json,
    load_bolt,
    load_pred,
    load_val_forecast,
    rmse,
    three_points,
    val_grid,
    write_csv,
    write_json,
)

GAMMAS = [0.0, 0.25, 0.5, 1.0, 2.0]
EXPERT_CKPT = {
    "chronos_bolt_zs": "cha_hybrid_v3",
    "chronos2_zs": "cha_hybrid_swap_chronos2",
    "tirex_zs": "cha_hybrid_swap_tirex",
}


def legacy_replica(h=6):
    """expR5, exact: RNG(2024) consumed in order ds -> gamma -> seed."""
    rng = np.random.default_rng(2024)
    rows = []
    for ds in DATASETS:
        for g in GAMMAS:
            al, rh, rf = [], [], []
            for s in [42, 123, 2024, 31337, 7]:
                rd = load_pred(ds, DEC, h, s)
                rgl = load_pred(ds, "cha_hybrid_v3_global_only", h, s)
                if rd is None or rgl is None:
                    continue
                # float32 as in expR5 so the replica is bit-identical
                y = rd[0].astype(np.float32).ravel()
                fd = rd[1].astype(np.float32).ravel()
                ff = rgl[1].astype(np.float32).ravel()
                n = min(len(y), len(fd), len(ff))
                y, fd, ff = y[:n], fd[:n], ff[:n]
                sigma = float(np.std(ff - y)) + 1e-9
                z = rng.standard_normal(n)
                ffd = ff + g * sigma * z
                a = bg_from_errors(y, fd, ffd)
                al.append(a)
                rh.append(rmse(y, a * fd + (1 - a) * ffd))
                rf.append(rmse(y, ffd))
            if al:
                rows.append(
                    {
                        "dataset": ds,
                        "gamma": g,
                        "alpha_insample": float(np.mean(al)),
                        "rmse_hybrid_insample": float(np.mean(rh)),
                        "rmse_fnd_degraded": float(np.mean(rf)),
                    }
                )
    return pd.DataFrame(rows)


def alpha_from_moments(ck, gamma):
    g = val_grid(ck)
    if g is None:
        return None
    m0, mh, m1 = three_points(*g)
    v = (gamma**2) * m0
    return bg_from_three(m0 + v, mh + v / 4.0, m1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", nargs="*", type=int, default=[6])
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--experts", nargs="*", default=list(EXPERT_CKPT))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    summary = {}

    leg = legacy_replica(6)
    ref_p = EVAL / "tables" / "alpha_stretch.csv"
    if ref_p.exists():
        ref = pd.read_csv(ref_p)
        leg = leg.merge(
            ref[["dataset", "gamma", "alpha_mean"]], on=["dataset", "gamma"], how="left"
        )
        leg["match"] = np.isclose(leg.alpha_insample, leg.alpha_mean, atol=1e-9)
        summary["legacy_alpha_all_match"] = bool(leg["match"].all())
    write_csv(leg, "degradation_legacy_check.csv")

    rows = []
    for expert in args.experts:
        ck_variant = EXPERT_CKPT[expert]
        seeds = SEEDS if expert == BOLT else [42]
        for h in args.horizons:
            for ds in DATASETS:
                b = load_bolt(ds, h, expert=expert)
                if b is None:
                    continue
                for s in seeds:
                    d = load_pred(ds, DEC, h, s)
                    ck = ckpt_json(ds, ck_variant, h, s)
                    if d is None:
                        continue
                    y, fd, ff = align(d[0], d[1], b[1])
                    s_test = float(np.std(ff - y))
                    vf = load_val_forecast(ds, h, s) if expert == BOLT else None
                    for g in GAMMAS:
                        fits = {}
                        if ck is not None:
                            a_m = alpha_from_moments(ck, g)
                            if a_m is not None:
                                fits["val_moments"] = a_m
                        for r in range(args.replicates):
                            if vf is not None:
                                yv, dv, gv = align(vf["y"], vf["dec"], vf["glob"])
                                s_val = float(np.std(gv - yv))
                                gvd = gv + g * s_val * rng.standard_normal(gv.shape)
                                fits["val_forecasts"] = bg_from_errors(yv, dv, gvd)
                            ffd = ff + g * s_test * rng.standard_normal(ff.shape)
                            r_f, r_d = rmse(y, ffd), rmse(y, fd)
                            a_in = bg_from_errors(y, fd, ffd)
                            for path, a in fits.items():
                                r_h = rmse(y, a * fd + (1 - a) * ffd)
                                rows.append(
                                    {
                                        "expert": expert,
                                        "fit_path": path,
                                        "dataset": ds,
                                        "horizon": h,
                                        "seed": s,
                                        "gamma": g,
                                        "replicate": r,
                                        "alpha_val": a,
                                        "alpha_test_insample": a_in,
                                        "rmse_hybrid_oos": r_h,
                                        "rmse_hybrid_insample": rmse(
                                            y, a_in * fd + (1 - a_in) * ffd
                                        ),
                                        "rmse_fnd_degraded": r_f,
                                        "rmse_decomp": r_d,
                                        "hyb_below_fnd": bool(r_h < r_f),
                                        "floor_holds_oos": bool(r_h <= min(r_f, r_d)),
                                        "excess_over_better_pct": 100
                                        * (r_h / min(r_f, r_d) - 1),
                                    }
                                )
    runs = pd.DataFrame(rows)
    write_csv(runs, "degradation_oos_runs.csv")
    if runs.empty:
        write_json(summary, "degradation_oos_summary.json")
        return
    cells = (
        runs.groupby(["expert", "fit_path", "dataset", "horizon", "gamma"])
        .agg(
            alpha_val_mean=("alpha_val", "mean"),
            alpha_val_sd=("alpha_val", "std"),
            alpha_insample_mean=("alpha_test_insample", "mean"),
            rmse_hybrid_oos=("rmse_hybrid_oos", "mean"),
            rmse_hybrid_insample=("rmse_hybrid_insample", "mean"),
            rmse_fnd_degraded=("rmse_fnd_degraded", "mean"),
            rmse_decomp=("rmse_decomp", "mean"),
            share_hyb_below_fnd=("hyb_below_fnd", "mean"),
            share_floor_holds=("floor_holds_oos", "mean"),
            max_excess_pct=("excess_over_better_pct", "max"),
        )
        .reset_index()
    )
    write_csv(cells, "degradation_oos_cells.csv")
    for (expert, path), g in cells.groupby(["expert", "fit_path"]):
        g2 = g[g.gamma == 2.0]
        gp = g[g.gamma > 0]
        nab = g2[g2.dataset == "nab_aws_cpu"]
        summary[f"{expert}/{path}"] = {
            "alpha_gamma0_range": [
                float(g[g.gamma == 0].alpha_val_mean.min()),
                float(g[g.gamma == 0].alpha_val_mean.max()),
            ],
            "alpha_gamma2_range": [
                float(g2.alpha_val_mean.min()),
                float(g2.alpha_val_mean.max()),
            ],
            "monotone_in_gamma_all_datasets": bool(
                all(
                    np.all(
                        np.diff(x.sort_values("gamma").alpha_val_mean.to_numpy())
                        >= -1e-9
                    )
                    for _, x in g.groupby("dataset")
                )
            ),
            "min_share_hyb_below_fnd_gamma_gt0": float(gp.share_hyb_below_fnd.min()),
            "cells_gamma_gt0_hyb_not_always_below_fnd": [
                f"{r.dataset} g={r.gamma}"
                for r in gp[gp.share_hyb_below_fnd < 1].itertuples()
            ],
            "min_share_floor_holds_gamma_gt0": float(gp.share_floor_holds.min()),
            "max_excess_over_better_pct": float(g.max_excess_pct.max()),
            "nab_aws_cpu_gamma2": nab.iloc[0][
                [
                    "rmse_hybrid_oos",
                    "rmse_fnd_degraded",
                    "rmse_decomp",
                    "alpha_val_mean",
                ]
            ].to_dict()
            if not nab.empty
            else None,
        }
    write_json(summary, "degradation_oos_summary.json")


if __name__ == "__main__":
    main()
