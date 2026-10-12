"""PERM analysis for Paper A v2: seed-paired, two-sided block permutation test.

For every dataset-by-horizon cell the five hybrid seeds are paired with the single
seed-42 Chronos-Bolt run, and the expert is counted once.  Per window w
    d_w = mean over seeds s of [ I(hybrid_s wins w) - I(Bolt wins w) ],
a win being a squared error at least 5 % below the rival's.  T = mean_w d_w is the
hybrid win rate minus the Bolt win rate.  Null: signs of d_w flipped in contiguous
blocks of length b (b = h respects the overlap of h-step windows).  Two-sided
p = (1 + #{|T_null| >= |T_obs|}) / (1 + B).  Holm over the cells whose forecasts
are not bit-identical for every seed (identical cells get no p-value).  Block lengths
1, h, 2h and 24.  A second statistic, the mean squared-error difference, is stored
for reference.

Outputs in ../results_v2: perm_cells_block{1,h,2h,24}.csv, perm_summary.json
Run: PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 perm_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIX = Path(__file__).resolve().parents[2] / "experiments" / "fix_2026-10-11"
sys.path.insert(0, str(FIX))
from _common import BOLT, CHAS, DATASETS, HORIZONS, SEEDS, align, bh, holm, load_bolt, load_pred  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results_v2"
BAND = 0.05
B = 20000
BLOCKS = ["1", "h", "2h", "24"]


def block_len(tag, h):
    return {"1": 1, "h": h, "2h": 2 * h, "24": 24}[tag]


def block_p(d, block, rng, chunk=1000):
    d = np.asarray(d, float)
    n = d.size
    nb = int(np.ceil(n / block))
    obs = float(d.mean())
    exceed = 0
    done = 0
    while done < B:
        k = min(chunk, B - done)
        s = rng.choice(np.array([-1.0, 1.0]), size=(k, nb))
        s = np.repeat(s, block, axis=1)[:, :n]
        exceed += int((np.abs((s * d).mean(axis=1)) >= abs(obs) - 1e-15).sum())
        done += k
    return obs, (1 + exceed) / (1 + B)


def cell_inputs():
    cells = {}
    for ds in DATASETS:
        for h in HORIZONS:
            b = load_bolt(ds, h)
            per = []
            for s in SEEDS:
                c = load_pred(ds, CHAS, h, s)
                if c is None:
                    continue
                yt, yc, yb = align(c[0], c[1], b[1])
                per.append((np.mean((yt - yc) ** 2, 1), np.mean((yt - yb) ** 2, 1), float(np.max(np.abs(yc - yb)))))
            n = min(x[0].size for x in per)
            se_b = per[0][1][:n]
            SEc = np.stack([x[0][:n] for x in per])
            cw = (SEc < (1 - BAND) * se_b).astype(float)
            rw = (se_b[None] < (1 - BAND) * SEc).astype(float)
            cells[(ds, h)] = dict(n=n, n_seeds=len(per), identity=all(x[2] == 0.0 for x in per), d_win=(cw - rw).mean(0), d_mse=SEc.mean(0) - se_b, hyb_win=float(cw.mean()), bolt_win=float(rw.mean()), rel_mse=100 * float(SEc.mean() - se_b.mean()) / float(se_b.mean()))
    return cells


def main():
    cells = cell_inputs()
    summary = {"B": B, "band": BAND, "seeds": SEEDS, "family": "cells whose forecasts differ from Bolt for at least one seed"}
    for tag in BLOCKS:
        rng = np.random.default_rng(20261012)
        rows = []
        for (ds, h), c in cells.items():
            blk = block_len(tag, h)
            row = dict(dataset=ds, horizon=h, block=blk, n_windows=c["n"], n_seeds=c["n_seeds"], identity=c["identity"], hyb_win_rate=c["hyb_win"], bolt_win_rate=c["bolt_win"], rel_mse_diff_pct=c["rel_mse"])
            if c["identity"]:
                row.update(T_winrate_diff=0.0, p_raw=np.nan, T_mse_diff=0.0, p_mse_raw=np.nan)
            else:
                T, p = block_p(c["d_win"], blk, rng)
                Tm, pm = block_p(c["d_mse"], blk, rng)
                row.update(T_winrate_diff=T, p_raw=p, T_mse_diff=Tm, p_mse_raw=pm)
            rows.append(row)
        df = pd.DataFrame(rows)
        ok = ~df.identity.to_numpy()
        for col, src in (("holm", "p_raw"), ("bh", "p_raw")):
            adj = np.full(len(df), np.nan)
            adj[ok] = (holm if col == "holm" else bh)(df.loc[ok, src].to_numpy())
            df[f"p_{col}"] = adj
        for col in ("raw", "holm"):
            p = df["p_raw" if col == "raw" else "p_holm"].to_numpy()
            sig = np.nan_to_num(p, nan=1.0) < 0.05
            better = df.T_winrate_diff > 0
            df[f"verdict_{col}"] = np.where(~ok, "identical", np.where(sig & better, "better", np.where(sig & ~better, "worse", "tie")))
        df.to_csv(OUT / f"perm_cells_block{tag}.csv", index=False)
        s = {}
        for col in ("raw", "holm"):
            v = df[f"verdict_{col}"]
            s[col] = {k: int((v == k).sum()) for k in ("better", "tie", "worse", "identical")}
            s[f"{col}_worse_cells"] = [f"{r.dataset} h={r.horizon}" for r in df[v == "worse"].itertuples()]
            s[f"{col}_better_cells"] = [f"{r.dataset} h={r.horizon}" for r in df[v == "better"].itertuples()]
        s["n_family"] = int(ok.sum())
        s["bolt_wins_more_windows_cells"] = int(((df.T_winrate_diff < 0) & ok).sum())
        summary[f"block_{tag}"] = s
        print(tag, s["holm"], s["holm_worse_cells"], "raw", s["raw"], flush=True)
    (OUT / "perm_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
