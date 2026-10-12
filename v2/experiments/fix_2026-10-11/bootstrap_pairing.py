"""audit item: bootstrap SDs with proper per-seed pairing (tab:multimetric).

pipeline/expR4_multi_metric.py concatenated the five CHA-S seeds (5 x N
windows) and bootstrapped that pool, while Chronos-Bolt is one deterministic
run (N windows). The CHA-S SD is therefore ~sqrt(5) too small (NAB AWS-CPU,
identical forecasts: 0.150 vs 0.335).

This script
  1. reproduces the legacy pooled numbers exactly (same RNG, same loop over
     all nine models) so the replacement can be traced;
  2. recomputes, per seed s, a PAIRED bootstrap over the N test windows (the
     same resampled indices for CHA-S seed s and Bolt): SD of each model's
     metric and SD of the paired difference; i.i.d. windows (as legacy) and
     a moving-block variant (block = h);
  3. reports the per-model SD as the mean over seeds of the per-seed SDs, the
     point estimate as the mean over seeds, and the NAB Twitter sMAPE gap
     expressed in Bolt SDs and in paired-difference SDs.

Outputs: bootstrap_legacy_check.csv, bootstrap_paired.csv,
         bootstrap_paired_h6_table.csv, bootstrap_pairing_summary.json
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
    CHAS,
    DATASETS,
    EVAL,
    HORIZONS,
    SEEDS,
    align,
    load_bolt,
    load_pred,
    write_csv,
    write_json,
)

LEGACY_MODELS = [
    "cha_hybrid_v3",
    "chronos_bolt_zs",
    "arima",
    "lstm",
    "xgboost",
    "patchtst",
    "nhits",
    "timesfm_zs",
    "moirai_zs",
]
LEGACY_LABEL = {"cha_hybrid_v3": "CHA-S", "chronos_bolt_zs": "Chronos-Bolt"}


def smape(yt, yp):
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    mask = denom > 1e-8
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100.0)


def metrics(yt, yp):
    return (
        float(np.sqrt(np.mean((yt - yp) ** 2))),
        float(np.mean(np.abs(yt - yp))),
        smape(yt, yp),
    )


def legacy(B=1000):
    """Exact replica of expR4 (RNG 7, loop ds -> h -> 9 models)."""
    rng = np.random.default_rng(7)
    seeds = [42, 123, 2024, 31337, 7]
    rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            for m in LEGACY_MODELS:
                yts, yps = [], []
                for s in seeds:
                    r = load_pred(ds, m, h, s)
                    if r is not None:
                        yts.append(r[0].astype(np.float32))
                        yps.append(r[1].astype(np.float32))
                if not yts:
                    continue
                yt, yp = np.concatenate(yts), np.concatenate(yps)
                n = len(yt)
                rb, mb, sb = [], [], []
                for _ in range(B):
                    idx = rng.integers(0, n, n)
                    rb.append(np.sqrt(np.mean((yt[idx] - yp[idx]) ** 2)))
                    mb.append(np.mean(np.abs(yt[idx] - yp[idx])))
                    sb.append(smape(yt[idx], yp[idx]))
                if m in LEGACY_LABEL:
                    rmse_, mae_, sm_ = metrics(yt, yp)
                    rows.append(
                        {
                            "dataset": ds,
                            "horizon": h,
                            "model": LEGACY_LABEL[m],
                            "n_pooled_windows": n,
                            "rmse": rmse_,
                            "rmse_sd": float(np.std(rb)),
                            "mae": mae_,
                            "mae_sd": float(np.std(mb)),
                            "smape": sm_,
                            "smape_sd": float(np.std(sb)),
                        }
                    )
    return pd.DataFrame(rows)


def mbb(n, block, B, rng):
    nb = int(np.ceil(n / block))
    st = rng.integers(0, max(n - block + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return np.minimum(idx, n - 1)


def paired(B, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            b = load_bolt(ds, h)
            if b is None:
                continue
            for s in SEEDS:
                c = load_pred(ds, CHAS, h, s)
                if c is None:
                    continue
                yt, yc, yb = align(c[0], c[1], b[1])
                n = yt.shape[0]
                for scheme, idx in (
                    ("iid", rng.integers(0, n, size=(B, n))),
                    (f"mbb{h}", mbb(n, h, B, rng)),
                ):
                    mc = np.array([metrics(yt[i], yc[i]) for i in idx])
                    mbolt = np.array([metrics(yt[i], yb[i]) for i in idx])
                    pc, pb = metrics(yt, yc), metrics(yt, yb)
                    for j, name in enumerate(("rmse", "mae", "smape")):
                        rows.append(
                            {
                                "dataset": ds,
                                "horizon": h,
                                "seed": s,
                                "scheme": scheme,
                                "metric": name,
                                "chas": pc[j],
                                "bolt": pb[j],
                                "chas_sd": float(mc[:, j].std()),
                                "bolt_sd": float(mbolt[:, j].std()),
                                "diff": pc[j] - pb[j],
                                "diff_sd": float((mc[:, j] - mbolt[:, j]).std()),
                            }
                        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--skip-legacy", action="store_true")
    args = ap.parse_args()
    summary = {}

    if not args.skip_legacy:
        leg = legacy(args.B)
        ref_p = EVAL / "tables" / "multi_metric_all.csv"
        if ref_p.exists():
            ref = pd.read_csv(ref_p)
            leg = leg.merge(
                ref[["dataset", "horizon", "model", "rmse_sd", "mae_sd", "smape_sd"]],
                on=["dataset", "horizon", "model"],
                how="left",
                suffixes=("", "_ref"),
            )
            for k in ("rmse_sd", "mae_sd", "smape_sd"):
                leg[f"{k}_match"] = np.isclose(
                    leg[k], leg[f"{k}_ref"], rtol=1e-6, atol=1e-9
                )
            summary["legacy_sd_all_match"] = bool(
                leg[[c for c in leg if c.endswith("_match")]].all().all()
            )
        write_csv(leg, "bootstrap_legacy_check.csv")

    pr = paired(args.B, args.seed)
    write_csv(pr, "bootstrap_paired.csv")
    agg = (
        pr.groupby(["dataset", "horizon", "scheme", "metric"])
        .agg(
            chas=("chas", "mean"),
            chas_seed_sd=("chas", "std"),
            chas_boot_sd=("chas_sd", "mean"),
            bolt=("bolt", "mean"),
            bolt_boot_sd=("bolt_sd", "mean"),
            diff=("diff", "mean"),
            diff_boot_sd=("diff_sd", "mean"),
        )
        .reset_index()
    )
    agg["gap_in_bolt_sd"] = agg["diff"] / agg["bolt_boot_sd"]
    agg["gap_in_diff_sd"] = agg["diff"] / agg["diff_boot_sd"]
    agg["within_one_bolt_sd"] = agg["gap_in_bolt_sd"].abs() <= 1.0
    write_csv(agg[(agg.horizon == 6)], "bootstrap_paired_h6_table.csv")

    for scheme in ("iid", "mbb6"):
        sub = agg[(agg.horizon == 6) & (agg.scheme == scheme)]
        tw = sub[(sub.dataset == "nab_twitter") & (sub.metric == "smape")]
        cpu = sub[(sub.dataset == "nab_aws_cpu") & (sub.metric == "rmse")]
        summary[f"h6_{scheme}"] = {
            "nab_twitter_smape": tw.iloc[0][
                [
                    "chas",
                    "bolt",
                    "diff",
                    "chas_boot_sd",
                    "bolt_boot_sd",
                    "diff_boot_sd",
                    "gap_in_bolt_sd",
                    "gap_in_diff_sd",
                ]
            ].to_dict()
            if not tw.empty
            else None,
            "nab_aws_cpu_rmse_sds": cpu.iloc[0][
                ["chas_boot_sd", "bolt_boot_sd"]
            ].to_dict()
            if not cpu.empty
            else None,
            "cells_metric_outside_one_bolt_sd": [
                f"{r.dataset}/{r.metric}"
                for r in sub[~sub.within_one_bolt_sd].itertuples()
            ],
        }
    write_json(summary, "bootstrap_pairing_summary.json")


if __name__ == "__main__":
    main()
