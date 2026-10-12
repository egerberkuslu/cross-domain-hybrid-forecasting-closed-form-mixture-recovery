"""DM analysis for Paper A v2: per-forecast-origin Diebold-Mariano test.

For every dataset-by-horizon cell and each of the 17 baselines the loss
differential is the squared-error difference averaged over the h steps of each
forecast origin, so the series has one value per origin, not one per step.  The
variance is Newey-West (Bartlett weights 1 - k/h, lags 1..h-1) with the
Harvey-Leybourne-Newbold small-sample correction and a Student t reference,
the arithmetic of src/evaluation/dm_test.py applied to the origin series.
Forecasts are the seed-mean native-scale forecasts, as in the published test.
A pair whose forecasts coincide up to numerical residue (max |difference| <= 1e-6 sd(y))
is an identity pair: it is counted as a tie and receives no p-value, and it is
left out of the BH and Holm families.  For reconciliation the flattened
(one value per step) test of the submitted version is recomputed as well.

Outputs in ../results_v2: dm_cells.csv, dm_per_baseline.csv, dm_summary.json
Run: PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 dm_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIX = Path(__file__).resolve().parents[2] / "experiments" / "fix_2026-10-11"
sys.path.insert(0, str(FIX))
from _common import CHAS, DATASETS, HORIZONS, bh, holm  # noqa: E402
from dm_tost_identity import ID_TOL, STRONG_BASELINES, dm_hln, seed_mean_native  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results_v2"
BOLT = "chronos_bolt_zs"


def verdict(p, stat, ok):
    sig = np.nan_to_num(p, nan=1.0) < 0.05
    return np.where(~ok, "identical", np.where(sig & (stat < 0), "better", np.where(sig & (stat > 0), "worse", "tie")))


def main():
    rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            ya, pa = seed_mean_native(ds, CHAS, h)
            for vb in STRONG_BASELINES:
                yb, pb = seed_mean_native(ds, vb, h)
                if yb is None or not np.allclose(ya, yb):
                    continue
                n = min(len(ya), len(yb))
                ea, eb = (ya - pa)[:n], (yb - pb)[:n]
                dd = ea**2 - eb**2
                s_o, p_o = dm_hln(dd.mean(axis=1), h)
                s_f, p_f = dm_hln(dd.ravel(), h)
                resid = float(np.max(np.abs(pa[:n] - pb[:n])))
                rows.append(dict(dataset=ds, horizon=h, baseline=vb, n_origins=n, stat_origin=s_o, p_origin=p_o, stat_flat=s_f, p_flat=p_f, resid=resid, identity=bool(resid <= ID_TOL * (float(np.std(ya)) + 1e-300))))
    df = pd.DataFrame(rows)
    ok = ~df.identity.to_numpy()
    for tag in ("origin", "flat"):
        p = df[f"p_{tag}"].to_numpy()
        for corr, fn in (("bh", bh), ("holm", holm)):
            adj = np.full(len(df), np.nan)
            adj[ok] = fn(np.nan_to_num(p[ok], nan=1.0))
            df[f"p_{tag}_{corr}"] = adj
        df[f"verdict_{tag}_raw"] = verdict(p, df[f"stat_{tag}"].to_numpy(), ok)
        df[f"verdict_{tag}_bh"] = verdict(df[f"p_{tag}_bh"].to_numpy(), df[f"stat_{tag}"].to_numpy(), ok)
        df[f"verdict_{tag}_holm"] = verdict(df[f"p_{tag}_holm"].to_numpy(), df[f"stat_{tag}"].to_numpy(), ok)
    df.to_csv(OUT / "dm_cells.csv", index=False)

    def cnt(d, col):
        v = d[col]
        return {k: int((v == k).sum()) for k in ("better", "tie", "worse", "identical")}

    # identical pairs are ties in the reported counts
    def cnt_tie(d, col):
        c = cnt(d, col)
        c["tie"] += c.pop("identical")
        return c

    s = {"n_pairs": int(len(df)), "n_identical": int(df.identity.sum()), "identical_pairs": [f"{r.dataset} h={r.horizon} vs {r.baseline}" for r in df[df.identity].itertuples()], "family_size_bh_holm": int(ok.sum())}
    for tag in ("origin", "flat"):
        for corr in ("raw", "bh", "holm"):
            s[f"{tag}_{corr}"] = cnt_tie(df, f"verdict_{tag}_{corr}")
            s[f"{tag}_{corr}_identity_excluded"] = cnt(df, f"verdict_{tag}_{corr}")
    b = df[df.baseline == BOLT]
    for tag in ("origin", "flat"):
        for corr in ("raw", "bh", "holm"):
            s[f"bolt_{tag}_{corr}"] = cnt_tie(b, f"verdict_{tag}_{corr}")
    for tag in ("origin", "flat"):
        for corr in ("raw", "holm"):
            col = f"verdict_{tag}_{corr}"
            s[f"bolt_{tag}_{corr}_worse_cells"] = [f"{r.dataset} h={r.horizon}" for r in b[b[col] == "worse"].itertuples()]
            s[f"bolt_{tag}_{corr}_better_cells"] = [f"{r.dataset} h={r.horizon} p={getattr(r, 'p_' + tag):.3g}" for r in b[b[col] == "better"].itertuples()]
    s["abilene_bolt_p_origin"] = {int(r.horizon): float(r.p_origin) for r in b[b.dataset == "abilene"].itertuples()}
    s["nab_twitter_h24_bolt"] = {"p_flat": float(b[(b.dataset == "nab_twitter") & (b.horizon == 24)].p_flat.iloc[0]), "p_origin": float(b[(b.dataset == "nab_twitter") & (b.horizon == 24)].p_origin.iloc[0])}
    s["geant_h24_bolt_identity"] = bool(b[(b.dataset == "geant") & (b.horizon == 24)].identity.iloc[0])
    # per baseline table (raw origin test, identity as tie) plus BH/Holm worse
    per = []
    for bl in STRONG_BASELINES:
        g = df[df.baseline == bl]
        c = cnt_tie(g, "verdict_origin_raw")
        cb = cnt_tie(g, "verdict_origin_bh")
        ch = cnt_tie(g, "verdict_origin_holm")
        cf = cnt_tie(g, "verdict_flat_raw")
        per.append(dict(baseline=bl, better=c["better"], tie=c["tie"], worse=c["worse"], bh_better=cb["better"], bh_worse=cb["worse"], holm_better=ch["better"], holm_worse=ch["worse"], flat_better=cf["better"], flat_tie=cf["tie"], flat_worse=cf["worse"], n_identical=int(g.identity.sum())))
    pd.DataFrame(per).to_csv(OUT / "dm_per_baseline.csv", index=False)
    (OUT / "dm_summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("identical_pairs")}, indent=1))
    print(pd.DataFrame(per).to_string())


if __name__ == "__main__":
    main()
