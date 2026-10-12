"""audit item: DM with bit-identical cells excluded; TOST identities flagged.

DM (reproduces pipeline/phase6_dm_test_full.py first, then corrects it)
  * legacy: native domain, seed-mean forecasts, errors flattened over the
    (window x step) grid, HAC Bartlett lag h-1, HLN correction, p < 0.05;
    17 baselines x 25 cells = 425 decisions. Expected 277/134/14,
    BH 269/145/11, Holm 225/194/6 (family = all 425 decisions).
  * identity flag: a (cell, baseline) pair whose two forecasts are equal up to
    numerical residue (max |f_a - f_b| <= tol * std(y), tol = 1e-6) is not a
    comparison of two forecasters. GEANT h=24 vs Bolt (p = 0.049 on a
    3.7e-9 residue) is the known case. Identity pairs are reported as
    "identity", and the counts are given with them excluded and with them
    counted as ties.
  * per-origin variant: the loss differential is averaged over the h steps
    of each forecast origin and the HAC lag h-1 is applied to that origin
    series (the textbook h-step DM). The flattened legacy variant applies
    lag h-1 to a series whose overlap dependence spans far more than h-1
    flattened positions, so its p-values are optimistic.

TOST (reproduces pipeline/expR3_tost_equivalence.py first)
  * legacy: B = 2000 i.i.d. window bootstrap, 90 % CI of RMSE_c - RMSE_b,
    margin 5 % of RMSE_b. NOTE: Bolt predictions exist for seed 42 only, so
    the legacy loop actually used ONE seed pair per cell, not five pooled
    seeds as the paper says.
  * identity flag: delta == 0 and CI == [0, 0] (forecasts identical); these
    cells are "identity", not evidence of equivalence.
  * per-seed TOST (each seed vs Bolt) and a seed-mean-forecast TOST, both
    with a moving-block bootstrap (block = h).

Outputs: dm_cells.csv, dm_summary.csv, tost_cells.csv, dm_tost_summary.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BOLT,
    CHAS,
    DATASETS,
    HORIZONS,
    PRED,
    SEEDS,
    align,
    bh,
    holm,
    load_bolt,
    load_pred,
    write_csv,
    write_json,
)

STRONG_BASELINES = [
    "chronos_bolt_zs",
    "chronos_zs",
    "moirai_zs",
    "timesfm_zs",
    "ttm_zs",
    "patchtst",
    "nhits",
    "nbeats",
    "tft",
    "dlinear",
    "tide",
    "tsmixer",
    "xgboost",
    "arima",
    "farima",
    "theta",
    "seasonal_naive",
]
ID_TOL = 1e-6


def seed_mean_native(ds, variant, h):
    files = sorted(PRED.glob(f"{ds}__{variant}__h{h}__s*.npz"))
    if not files:
        return None, None
    preds, yt = [], None
    for f in files:
        d = np.load(f)
        if yt is None:
            yt = d["y_true_native"].astype(np.float64)
        preds.append(d["y_pred_native"].astype(np.float64))
    return yt, np.mean(np.stack(preds), axis=0)


def _autocov(x, k):
    n = x.size
    if k >= n:
        return 0.0
    xm = x.mean()
    return float(np.sum((x[k:] - xm) * (x[: n - k] - xm)) / n)


def dm_hln(d, h):
    """Same arithmetic as src/evaluation/dm_test.py, applied to a given d."""
    d = d[np.isfinite(d)]
    T = d.size
    if T < 8:
        return np.nan, np.nan
    dbar = float(d.mean())
    var = _autocov(d, 0)
    for k in range(1, h):
        var += 2.0 * (1.0 - k / h) * _autocov(d, k)
    if var <= 0:
        return np.nan, np.nan
    stat = dbar / np.sqrt(var / T)
    if T > h:
        stat *= np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
        p = 2.0 * (1.0 - student_t.cdf(abs(stat), df=T - 1))
    else:
        p = 2.0 * (1.0 - norm.cdf(abs(stat)))
    return float(stat), float(p)


def counts(df, pcol, stat_col="stat", mask=None):
    d = df if mask is None else df[mask]
    sig = d[pcol] < 0.05
    return {
        "better": int(((d[stat_col] < 0) & sig).sum()),
        "tie": int((~sig).sum()),
        "worse": int(((d[stat_col] > 0) & sig).sum()),
        "n": int(len(d)),
    }


def run_dm():
    rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            ya, pa = seed_mean_native(ds, CHAS, h)
            if ya is None:
                continue
            for vb in STRONG_BASELINES:
                yb, pb = seed_mean_native(ds, vb, h)
                if yb is None or not np.allclose(ya, yb):
                    continue
                ea, eb = ya - pa, yb - pb
                d_flat = (ea**2 - eb**2).ravel()
                s_flat, p_flat = dm_hln(d_flat, h)
                d_orig = (ea**2 - eb**2).mean(axis=1)
                s_orig, p_orig = dm_hln(d_orig, h)
                resid = float(np.max(np.abs(pa - pb)))
                scale = float(np.std(ya)) + 1e-300
                rows.append(
                    {
                        "dataset": ds,
                        "horizon": h,
                        "baseline": vb,
                        "stat": s_flat,
                        "p": p_flat,
                        "stat_per_origin": s_orig,
                        "p_per_origin": p_orig,
                        "max_abs_forecast_diff": resid,
                        "rel_forecast_diff": resid / scale,
                        "identity": bool(resid <= ID_TOL * scale),
                    }
                )
    df = pd.DataFrame(rows)
    df["p_bh"] = bh(df["p"].to_numpy())
    df["p_holm"] = holm(df["p"].to_numpy())
    ni = ~df["identity"]
    # corrections recomputed on the non-identity family
    df["p_bh_noid"] = np.nan
    df["p_holm_noid"] = np.nan
    df.loc[ni, "p_bh_noid"] = bh(df.loc[ni, "p"].to_numpy())
    df.loc[ni, "p_holm_noid"] = holm(df.loc[ni, "p"].to_numpy())
    df["p_per_origin_bh"] = bh(df["p_per_origin"].fillna(1.0).to_numpy())
    df["p_per_origin_holm"] = holm(df["p_per_origin"].fillna(1.0).to_numpy())
    return df


def summarize_dm(df):
    s = {
        "legacy_raw": counts(df, "p"),
        "legacy_bh": counts(df, "p_bh"),
        "legacy_holm": counts(df, "p_holm"),
        "identity_pairs": [
            f"{r.dataset} h={r.horizon} vs {r.baseline} (p={r.p:.3g}, resid={r.max_abs_forecast_diff:.2g})"
            for r in df[df.identity].itertuples()
        ],
    }
    ni = ~df["identity"]
    s["corrected_raw_identity_excluded"] = counts(df, "p", mask=ni)
    s["corrected_bh_identity_excluded"] = counts(df, "p_bh_noid", mask=ni)
    s["corrected_holm_identity_excluded"] = counts(df, "p_holm_noid", mask=ni)
    tmp = df.copy()
    tmp.loc[tmp.identity, "p"] = 1.0
    s["corrected_raw_identity_as_tie"] = counts(tmp, "p")
    s["per_origin_raw"] = counts(
        df.assign(stat=df.stat_per_origin), "p_per_origin", mask=ni
    )
    s["per_origin_bh"] = counts(
        df.assign(stat=df.stat_per_origin), "p_per_origin_bh", mask=ni
    )
    s["per_origin_holm"] = counts(
        df.assign(stat=df.stat_per_origin), "p_per_origin_holm", mask=ni
    )
    bolt = df[df.baseline == BOLT]
    s["bolt_legacy"] = counts(bolt, "p")
    s["bolt_identity_as_tie"] = counts(
        bolt.assign(p=np.where(bolt.identity, 1.0, bolt.p)), "p"
    )
    s["bolt_identity_excluded"] = counts(bolt, "p", mask=~bolt.identity)
    s["bolt_per_origin"] = counts(
        bolt.assign(stat=bolt.stat_per_origin), "p_per_origin", mask=~bolt.identity
    )
    s["bolt_losses_legacy"] = [
        f"{r.dataset} h={r.horizon}"
        for r in bolt[(bolt.p < 0.05) & (bolt.stat > 0)].itertuples()
    ]
    s["bolt_wins_legacy"] = [
        f"{r.dataset} h={r.horizon} p={r.p:.3g}"
        for r in bolt[(bolt.p < 0.05) & (bolt.stat < 0)].itertuples()
    ]
    per_b = []
    for b, g in df.groupby("baseline"):
        c1, c2 = counts(g, "p"), counts(g, "p", mask=~g.identity)
        per_b.append(
            {
                "baseline": b,
                "legacy_better": c1["better"],
                "legacy_tie": c1["tie"],
                "legacy_worse": c1["worse"],
                "n_identity": int(g.identity.sum()),
                "noid_better": c2["better"],
                "noid_tie": c2["tie"],
                "noid_worse": c2["worse"],
            }
        )
    return s, pd.DataFrame(per_b)


def mbb_indices(n, block, B, rng):
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return np.minimum(idx, n - 1)


def tost_verdict(lo, hi, eps, delta):
    if delta == 0.0 and lo == 0.0 and hi == 0.0:
        return "IDENTITY"
    if hi < -eps:
        return "BETTER"
    if lo > eps:
        return "WORSE"
    if lo >= -eps and hi <= eps:
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def run_tost(B, block_mode, seed):
    rows = []
    # ---- legacy reproduction (expR3: RNG 12345, i.i.d. window bootstrap)
    lrng = np.random.default_rng(12345)
    for ds in DATASETS:
        for h in HORIZONS:
            vparts, bparts, used = [], [], []
            for s in SEEDS:  # expR3 order [42,123,2024,31337,7]
                c = load_pred(ds, CHAS, h, s)
                b = load_pred(ds, BOLT, h, s)  # exists for s=42 only
                if c is None or b is None:
                    continue
                n = min(c[0].shape[0], b[0].shape[0])
                vparts.append(np.mean((c[0][:n] - c[1][:n]) ** 2, axis=1))
                bparts.append(np.mean((b[0][:n] - b[1][:n]) ** 2, axis=1))
                used.append(s)
            if not vparts:
                continue
            ev, eb = np.concatenate(vparts), np.concatenate(bparts)
            n = ev.size
            eps = 0.05 * float(np.sqrt(eb.mean()))
            deltas = np.empty(B)
            for k in range(B):
                idx = lrng.integers(0, n, n)
                deltas[k] = np.sqrt(ev[idx].mean()) - np.sqrt(eb[idx].mean())
            lo, hi = np.percentile(deltas, [5, 95])
            delta = float(np.sqrt(ev.mean()) - np.sqrt(eb.mean()))
            rows.append(
                {
                    "mode": "legacy_iid",
                    "dataset": ds,
                    "horizon": h,
                    "seeds_used": ",".join(map(str, used)),
                    "n": n,
                    "delta": delta,
                    "ci90_lo": float(lo),
                    "ci90_hi": float(hi),
                    "eps": eps,
                    "verdict": tost_verdict(float(lo), float(hi), eps, delta),
                }
            )
    # ---- corrected: per seed (paired with the deterministic Bolt) + seed-mean
    rng = np.random.default_rng(seed)
    for ds in DATASETS:
        for h in HORIZONS:
            b = load_bolt(ds, h)
            if b is None:
                continue
            block = h if block_mode == "h" else int(block_mode)
            preds = {}
            for s in SEEDS:
                c = load_pred(ds, CHAS, h, s)
                if c is not None:
                    preds[s] = c
            if not preds:
                continue
            n = min(min(c[0].shape[0] for c in preds.values()), b[0].shape[0])
            y = b[0][:n]
            eb = np.mean((y - b[1][:n]) ** 2, axis=1)
            series = {
                f"seed{s}": np.mean((y - c[1][:n]) ** 2, axis=1)
                for s, c in preds.items()
            }
            mean_fc = np.mean(np.stack([c[1][:n] for c in preds.values()]), axis=0)
            series["seedmean_forecast"] = np.mean((y - mean_fc) ** 2, axis=1)
            idx = mbb_indices(n, block, B, rng)
            for name, ev in series.items():
                deltas = np.sqrt(ev[idx].mean(axis=1)) - np.sqrt(eb[idx].mean(axis=1))
                lo, hi = np.percentile(deltas, [5, 95])
                delta = float(np.sqrt(ev.mean()) - np.sqrt(eb.mean()))
                eps = 0.05 * float(np.sqrt(eb.mean()))
                identical = bool(np.max(np.abs(ev - eb)) == 0.0)
                rows.append(
                    {
                        "mode": f"mbb_{name}",
                        "block": block,
                        "dataset": ds,
                        "horizon": h,
                        "seeds_used": name,
                        "n": n,
                        "delta": delta,
                        "ci90_lo": float(lo),
                        "ci90_hi": float(hi),
                        "eps": eps,
                        "verdict": "IDENTITY"
                        if identical
                        else tost_verdict(float(lo), float(hi), eps, delta),
                    }
                )
    return pd.DataFrame(rows)


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


def run_generation(B, seed):
    """Generation-sweep DM and TOST tables with identity pairs flagged.

    DM: CHA-S (Bolt, seed-mean) vs each new-generation rival, and each swap
    hybrid (seed 42) vs its own frozen expert, native domain, flattened
    (legacy) and per-origin. TOST (scaled, moving-block bootstrap, block = h)
    for the same pairs. A swap hybrid with alpha = 0 reproduces its expert
    exactly; such cells are "identity", not equivalence or ties.
    """
    rng = np.random.default_rng(seed)
    pairs = [(CHAS, r, f"chas_vs_{r}") for r in GEN_RIVALS]
    pairs += [(h_, r, f"{h_}_vs_own_expert") for h_, r in GEN_SWAP.items()]
    dm_rows, tost_rows = [], []
    for hyb, rival, comp in pairs:
        for ds in DATASETS:
            for h in HORIZONS:
                ya, pa = seed_mean_native(ds, hyb, h)
                yb, pb = seed_mean_native(ds, rival, h)
                if (
                    ya is None
                    or yb is None
                    or ya.shape != yb.shape
                    or not np.allclose(ya, yb)
                ):
                    continue
                ea, eb = ya - pa, yb - pb
                s_f, p_f = dm_hln((ea**2 - eb**2).ravel(), h)
                s_o, p_o = dm_hln((ea**2 - eb**2).mean(axis=1), h)
                resid = float(np.max(np.abs(pa - pb)))
                ident = bool(resid <= ID_TOL * (float(np.std(ya)) + 1e-300))
                dm_rows.append(
                    {
                        "comparison": comp,
                        "dataset": ds,
                        "horizon": h,
                        "stat": s_f,
                        "p": p_f,
                        "stat_per_origin": s_o,
                        "p_per_origin": p_o,
                        "identity": ident,
                        "max_abs_forecast_diff": resid,
                    }
                )
                # TOST on scaled, seed-mean hybrid forecast
                hs = [load_pred(ds, hyb, h, s) for s in SEEDS]
                hs = [x for x in hs if x is not None]
                rv = load_pred(ds, rival, h, 42)
                if not hs or rv is None:
                    continue
                n = min(min(x[0].shape[0] for x in hs), rv[0].shape[0])
                y = rv[0][:n]
                fc = np.mean(np.stack([x[1][:n] for x in hs]), axis=0)
                ev = np.mean((y - fc) ** 2, axis=1)
                er = np.mean((y - rv[1][:n]) ** 2, axis=1)
                idx = mbb_indices(n, h, B, rng)
                deltas = np.sqrt(ev[idx].mean(axis=1)) - np.sqrt(er[idx].mean(axis=1))
                lo, hi = np.percentile(deltas, [5, 95])
                delta = float(np.sqrt(ev.mean()) - np.sqrt(er.mean()))
                eps = 0.05 * float(np.sqrt(er.mean()))
                v = (
                    "IDENTITY"
                    if float(np.max(np.abs(fc - rv[1][:n]))) == 0.0
                    else tost_verdict(float(lo), float(hi), eps, delta)
                )
                tost_rows.append(
                    {
                        "comparison": comp,
                        "dataset": ds,
                        "horizon": h,
                        "delta": delta,
                        "ci90_lo": float(lo),
                        "ci90_hi": float(hi),
                        "eps": eps,
                        "verdict": v,
                    }
                )
    dm = pd.DataFrame(dm_rows)
    tost = pd.DataFrame(tost_rows)
    s = {}
    for comp, g in dm.groupby("comparison", sort=False):
        ni = ~g.identity
        s[comp] = {
            "dm_legacy": counts(g, "p"),
            "dm_identity_excluded": counts(g, "p", mask=ni),
            "dm_per_origin_identity_excluded": counts(
                g.assign(stat=g.stat_per_origin), "p_per_origin", mask=ni
            ),
            "n_identity": int(g.identity.sum()),
        }
        t = tost[tost.comparison == comp]
        if not t.empty:
            s[comp]["tost"] = {k: int(v) for k, v in t.verdict.value_counts().items()}
    return dm, tost, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--block", default="h")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--generation", action="store_true")
    args = ap.parse_args()

    dm = run_dm()
    write_csv(dm, "dm_cells.csv")
    s_dm, per_b = summarize_dm(dm)
    write_csv(per_b, "dm_summary.csv")

    tost = run_tost(args.B, args.block, args.seed)
    write_csv(tost, "tost_cells.csv")
    s_tost = {}
    for mode, g in tost.groupby("mode"):
        vc = g.verdict.value_counts().to_dict()
        s_tost[mode] = {k: int(v) for k, v in vc.items()}
        s_tost[mode]["identity_cells"] = [
            f"{r.dataset} h={r.horizon}"
            for r in g[g.verdict == "IDENTITY"].itertuples()
        ]
        s_tost[mode]["non_equivalent_cells"] = [
            f"{r.dataset} h={r.horizon}:{r.verdict}"
            for r in g[~g.verdict.isin(["EQUIVALENT", "IDENTITY"])].itertuples()
        ]
    seedwise = tost[
        tost["mode"].str.contains("_seed") & ~tost["mode"].str.contains("seedmean")
    ]
    if not seedwise.empty:
        per_cell = seedwise.groupby(["dataset", "horizon"]).verdict.agg(
            lambda v: (v == "EQUIVALENT").sum()
        )
        s_tost["per_seed_cells_equivalent_all_5_seeds"] = int((per_cell == 5).sum())
    out = {"dm": s_dm, "tost": s_tost, "identity_tol_rel_std": ID_TOL}
    if args.generation:
        gdm, gtost, s_gen = run_generation(args.B, args.seed + 1)
        write_csv(gdm, "dm_generation_cells.csv")
        write_csv(gtost, "tost_generation_cells.csv")
        out["generation"] = s_gen
    write_json(out, "dm_tost_summary.json")


if __name__ == "__main__":
    main()
