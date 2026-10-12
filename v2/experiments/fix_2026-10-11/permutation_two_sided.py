"""audit item 2: two-sided, per-seed-paired permutation test CHA-S vs Chronos-Bolt.

What was wrong in pipeline/expR2_permutation_test.py
  * one-sided p = P(T_null >= T_obs): a cell where Bolt wins more windows can
    only come out as p ~ 1, never as a significant loss;
  * five CHA-S seeds were concatenated against ONE deterministic Bolt run, so
    every Bolt error appears five times (n = 5 x N, pseudo-replication);
  * sign flips were per window although h-step windows with stride 1 overlap.

What this script does (all on the saved test forecasts, scaled domain)
  * per window w the paired statistic uses all available hybrid seeds
    without replication: d_w = mean_s [ I(hybrid_s wins w) - I(rival wins w) ]
    with a symmetric 5 % band (hybrid wins if se_c < 0.95 se_r, rival wins if
    se_r < 0.95 se_c). T = mean_w d_w = (hybrid win rate) - (rival win rate).
    A second statistic is the mean squared-error difference
    d_w = mean_s se_c,s(w) - se_r(w).
  * null: sign flips in contiguous blocks of length h (overlap-aware; use
    --block 1 for per-window flips), two-sided p = P(|T_null| >= |T_obs|).
  * per seed: the same test for each hybrid seed separately.
  * multiple testing across the 25 cells of each comparison: Holm and BH.
  * --generation adds the generation-sweep DM / permutation pairs: CHA-S
    (Bolt) vs each new-generation zero-shot rival, and each swap hybrid
    (seed 42) vs its own frozen expert. Identity pairs (bit-identical
    forecasts, alpha = 0) are flagged and get no p-value.
  * legacy reproduction: the original pooled one-sided test is recomputed so
    observed win rates and p-values can be matched to per_window_permutation.csv.

Outputs: permutation_two_sided_cells.csv, permutation_two_sided_seeds.csv,
         permutation_legacy_check.csv, permutation_two_sided_summary.json
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
    bh,
    holm,
    load_bolt,
    load_pred,
    write_csv,
    write_json,
)

BAND = 0.05
GEN_RIVALS = [
    "chronos2_zs",
    "chronos2_small_zs",
    "timesfm3_zs",
    "flowstate_zs",
    "tirex_zs",
]
GEN_SWAP = {
    "cha_hybrid_swap_chronos2": "chronos2_zs",
    "cha_hybrid_swap_chronos2_small": "chronos2_small_zs",
    "cha_hybrid_swap_timesfm3": "timesfm3_zs",
    "cha_hybrid_swap_flowstate": "flowstate_zs",
    "cha_hybrid_swap_tirex": "tirex_zs",
}


def se_per_window(y, p):
    return np.mean((y - p) ** 2, axis=1)


def win_indicators(se_c, se_r):
    c_win = se_c < (1.0 - BAND) * se_r
    r_win = se_r < (1.0 - BAND) * se_c
    return c_win.astype(float), r_win.astype(float)


def block_signflip_p(d, block, n_perm, rng, chunk=500):
    """Two-sided p-value of mean(d) under block sign flips."""
    d = np.asarray(d, dtype=float)
    n = d.size
    nb = int(np.ceil(n / block))
    obs = float(d.mean())
    exceed = 0
    nulls = []
    done = 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        s = rng.choice(np.array([-1.0, 1.0]), size=(k, nb))
        s = np.repeat(s, block, axis=1)[:, :n]
        null = (s * d).mean(axis=1)
        exceed += int((np.abs(null) >= abs(obs) - 1e-15).sum())
        nulls.append(null)
        done += k
    return obs, (1 + exceed) / (1 + n_perm), float(np.concatenate(nulls).std())


def legacy_one_sided(ds, h, rng, n_perm):
    """Exact re-implementation of expR2 (pooled, per-window, one-sided)."""
    se_c_all, se_b_all = [], []
    b = load_bolt(ds, h)
    for s in [42, 123, 2024, 7, 31337]:  # expR2 seed order
        c = load_pred(ds, CHAS, h, s)
        if c is None or b is None:
            continue
        yt, yc, yb = align(c[0], c[1], b[1])
        se_c_all.append(se_per_window(yt, yc))
        se_b_all.append(se_per_window(yt, yb))
    if not se_c_all:
        return None
    sc, sb = np.concatenate(se_c_all), np.concatenate(se_b_all)
    n = sc.size
    rel = (sb - sc) / np.maximum(sb, 1e-12)
    obs = float((rel > BAND).mean())
    null = np.empty(n_perm)
    for k in range(n_perm):
        flip = rng.random(n) > 0.5
        pv = np.where(flip, sb, sc)
        pb = np.where(flip, sc, sb)
        null[k] = ((pb - pv) / np.maximum(pb, 1e-12) > BAND).mean()
    return {
        "n_pooled": n,
        "observed_win_rate": obs,
        "p_one_sided": float((null >= obs).mean()),
    }


def run_pair(comparison, hyb, seeds, rival, args, rng):
    cell_rows, seed_rows = [], []
    for ds in DATASETS:
        for h in HORIZONS:
            r = load_pred(ds, rival, h, 42)
            if r is None:
                continue
            block = h if args.block == "h" else int(args.block)
            per_seed = []
            for s in seeds:
                c = load_pred(ds, hyb, h, s)
                if c is None:
                    continue
                yt, yc, yr = align(c[0], c[1], r[1])
                per_seed.append(
                    (
                        s,
                        se_per_window(yt, yc),
                        se_per_window(yt, yr),
                        float(np.max(np.abs(yc - yr))),
                    )
                )
            if not per_seed:
                continue
            n = min(x[1].size for x in per_seed)
            se_r = per_seed[0][2][:n]
            identity = all(x[3] == 0.0 for x in per_seed)
            Cw = np.stack([win_indicators(x[1][:n], se_r)[0] for x in per_seed])
            Rw = np.stack([win_indicators(x[1][:n], se_r)[1] for x in per_seed])
            SEc = np.stack([x[1][:n] for x in per_seed])
            row = {
                "comparison": comparison,
                "hybrid": hyb,
                "rival": rival,
                "dataset": ds,
                "horizon": h,
                "n_windows": n,
                "n_seeds": len(per_seed),
                "block": block,
                "identity": identity,
                "hyb_win_rate": float(Cw.mean()),
                "rival_win_rate": float(Rw.mean()),
                "tie_rate": float(1 - Cw.mean() - Rw.mean()),
                "rel_mse_diff_pct": 100
                * float(SEc.mean() - se_r.mean())
                / max(float(se_r.mean()), 1e-12),
            }
            if identity:
                row.update(
                    {
                        "T_winrate_diff": 0.0,
                        "p_two_sided_winrate": np.nan,
                        "T_mse_diff": 0.0,
                        "p_two_sided_mse": np.nan,
                        "null_sd_winrate": np.nan,
                    }
                )
            else:
                T_win, p_win, sd_win = block_signflip_p(
                    (Cw - Rw).mean(axis=0), block, args.n_perm, rng
                )
                T_se, p_se, _ = block_signflip_p(
                    SEc.mean(axis=0) - se_r, block, args.n_perm, rng
                )
                row.update(
                    {
                        "T_winrate_diff": T_win,
                        "p_two_sided_winrate": p_win,
                        "T_mse_diff": T_se,
                        "p_two_sided_mse": p_se,
                        "null_sd_winrate": sd_win,
                    }
                )
            row["rival_wins_more_windows"] = bool(row["T_winrate_diff"] < 0)
            cell_rows.append(row)
            if len(per_seed) > 1 and not identity:
                for s, sc, sr, _ in per_seed:
                    cw, rw = win_indicators(sc[:n], sr[:n])
                    Ts, ps, _ = block_signflip_p(cw - rw, block, args.n_perm, rng)
                    Tm, pm, _ = block_signflip_p(
                        sc[:n] - sr[:n], block, args.n_perm, rng
                    )
                    seed_rows.append(
                        {
                            "comparison": comparison,
                            "dataset": ds,
                            "horizon": h,
                            "seed": s,
                            "hyb_win_rate": float(cw.mean()),
                            "rival_win_rate": float(rw.mean()),
                            "T_winrate_diff": Ts,
                            "p_two_sided_winrate": ps,
                            "T_mse_diff": Tm,
                            "p_two_sided_mse": pm,
                        }
                    )
            print(
                f"{comparison:34s} {ds:12s} h={h:2d} T_win={row['T_winrate_diff']:+.3f} p={row['p_two_sided_winrate']}"
            )
    return cell_rows, seed_rows


def verdicts(cells, summary):
    out = []
    for comp, g in cells.groupby("comparison", sort=False):
        g = g.copy()
        ok = ~g["identity"]
        summary[comp] = {
            "cells": int(len(g)),
            "identity_cells": [
                f"{r.dataset} h={r.horizon}" for r in g[~ok].itertuples()
            ],
            "rival_wins_more_windows": int((g.rival_wins_more_windows & ok).sum()),
        }
        for stat in ("winrate", "mse"):
            T = g["T_winrate_diff" if stat == "winrate" else "T_mse_diff"].to_numpy()
            good = T > 0 if stat == "winrate" else T < 0
            p = g[f"p_two_sided_{stat}"].to_numpy()
            for corr in ("raw", "holm", "bh"):
                padj = np.full(p.size, np.nan)
                if ok.any():
                    pv = p[ok.to_numpy()]
                    padj[ok.to_numpy()] = (
                        pv
                        if corr == "raw"
                        else (holm(pv) if corr == "holm" else bh(pv))
                    )
                sig = np.nan_to_num(padj, nan=1.0) < 0.05
                g[f"p_{stat}_{corr}"] = padj
                g[f"verdict_{stat}_{corr}"] = np.where(
                    ~ok,
                    "identity",
                    np.where(
                        sig & good,
                        "hyb_better",
                        np.where(sig & ~good, "hyb_worse", "tie"),
                    ),
                )
                summary[comp][f"{stat}_{corr}"] = {
                    "hyb_better": int((sig & good).sum()),
                    "tie": int((~sig & ok.to_numpy()).sum()),
                    "hyb_worse": int((sig & ~good).sum()),
                    "identity": int((~ok).sum()),
                    "worse_cells": [
                        f"{r.dataset} h={r.horizon}"
                        for r in g[sig & ~good].itertuples()
                    ],
                    "better_cells": [
                        f"{r.dataset} h={r.horizon}" for r in g[sig & good].itertuples()
                    ],
                }
        out.append(g)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--block", default="h", help="'h' (default) or an integer")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--skip-legacy", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pairs = [("chas_vs_bolt", CHAS, SEEDS, BOLT)]
    if args.generation:
        pairs += [(f"chas_vs_{r}", CHAS, SEEDS, r) for r in GEN_RIVALS]
        pairs += [(f"{hyb}_vs_own_expert", hyb, [42], r) for hyb, r in GEN_SWAP.items()]
    cell_rows, seed_rows = [], []
    for comp, hyb, seeds, rival in pairs:
        c, s = run_pair(comp, hyb, seeds, rival, args, rng)
        cell_rows += c
        seed_rows += s

    summary = {"n_perm": args.n_perm, "block": args.block, "band": BAND}
    cells = verdicts(pd.DataFrame(cell_rows), summary)
    write_csv(cells, "permutation_two_sided_cells.csv")

    seeds = pd.DataFrame(seed_rows)
    if not seeds.empty:
        for stat in ("winrate", "mse"):
            T = seeds[
                "T_winrate_diff" if stat == "winrate" else "T_mse_diff"
            ].to_numpy()
            good = T > 0 if stat == "winrate" else T < 0
            p = seeds[f"p_two_sided_{stat}"].to_numpy()
            seeds[f"sig_better_{stat}"] = (p < 0.05) & good
            seeds[f"sig_worse_{stat}"] = (p < 0.05) & ~good
        b = seeds[seeds.comparison == "chas_vs_bolt"]
        agg = b.groupby(["dataset", "horizon"]).agg(
            worse=("sig_worse_mse", "sum"), better=("sig_better_mse", "sum")
        )
        summary["chas_vs_bolt_per_seed"] = {
            "cells_any_seed_sig_worse_mse": int((agg.worse > 0).sum()),
            "cells_majority_seeds_sig_worse_mse": int((agg.worse >= 3).sum()),
            "cells_majority_seeds_sig_better_mse": int((agg.better >= 3).sum()),
        }
        write_csv(seeds, "permutation_two_sided_seeds.csv")

    if not args.skip_legacy:
        ref_p = EVAL / "tables" / "per_window_permutation.csv"
        ref = pd.read_csv(ref_p) if ref_p.exists() else None
        lrng = np.random.default_rng(1234567)  # expR2 RNG seed, same loop order
        lrows = []
        for ds in DATASETS:
            for h in HORIZONS:
                r = legacy_one_sided(ds, h, lrng, 2000)
                if r is None:
                    continue
                r.update({"dataset": ds, "horizon": h})
                if ref is not None:
                    m = ref[(ref.dataset == ds) & (ref.horizon == h)]
                    if not m.empty:
                        r["ref_observed_win_rate"] = float(m.observed_win_rate.iloc[0])
                        r["ref_p_value"] = float(m.p_value.iloc[0])
                        r["obs_match"] = bool(
                            abs(r["ref_observed_win_rate"] - r["observed_win_rate"])
                            < 1e-12
                        )
                        r["p_match"] = bool(
                            abs(r["ref_p_value"] - r["p_one_sided"]) < 1e-12
                        )
                lrows.append(r)
        leg = pd.DataFrame(lrows)
        write_csv(leg, "permutation_legacy_check.csv")
        if "obs_match" in leg:
            summary["legacy_observed_win_rate_all_match"] = bool(leg.obs_match.all())
            summary["legacy_p_all_match"] = bool(leg.p_match.all())
            summary["legacy_cells_p_ge_0.999"] = [
                f"{r.dataset} h={r.horizon}"
                for r in leg[leg.p_one_sided >= 0.999].itertuples()
            ]
    write_json(summary, "permutation_two_sided_summary.json")


if __name__ == "__main__":
    main()
