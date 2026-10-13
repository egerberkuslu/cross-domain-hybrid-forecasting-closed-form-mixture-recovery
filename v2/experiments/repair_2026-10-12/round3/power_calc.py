"""Power of the round 3 confirmation rule, from round 2 effect sizes only (no fresh series involved).

Template: the 25 test cells of E-Q (three-expert mixture) of round 2 (`../paper_a/round2/a5_test_cells.csv`): relative RMSE change rel and its
moving-block bootstrap 95 % interval, from which se = (ci_hi - ci_lo) / 3.92. Fresh pool: 100 cells = 20 series x 5 horizons, each fresh series takes the
round 2 cells of its source (cesnet institution -> CESNET aggregate cells, Abilene OD -> Abilene aggregate cells, GEANT OD -> GEANT aggregate cells) at the
same horizon. True effect of a fresh cell = shrink * (-rel_template) * u_series with u_series lognormal(0, 0.5) (series-to-series heterogeneity); the
estimate is N(true effect, se_template). Cell p value two-sided normal, Holm over the 100 cells, win = Holm p < 0.05 and estimated gain >= 0.5 %,
loss = Holm p < 0.05 and estimated gain < 0. Pooled gain = mean over the 100 cells; CI by bootstrap of cells and by two-stage bootstrap of series then horizons.
Rule = wins >= K, no loss, pooled gain >= 0.5 %, cell-level and series-clustered CI lower bounds above 0.  Output: power_calc.json.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE.parent / "paper_a" / "round2" / "a5_test_cells.csv")
d = d[d.expert == "EQ"].copy()
d["se"] = (d.ci_hi - d.ci_lo) / 3.92
tmpl = {ds: g.sort_values("horizon") for ds, g in d.groupby("dataset")}
SRC = {"cesnet": "cesnet", "abilene": "abilene", "geant": "geant"}
POOL = [("cesnet", 8), ("abilene", 6), ("geant", 6)]
HS = [1, 3, 6, 12, 24]


def holm(p):
    n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k]); adj[k] = min(1.0, run)
    return adj


from scipy.stats import norm  # noqa: E402


def simulate(shrink, K, B=1500, tau=0.5, seed=1, boot=200):
    rng = np.random.default_rng(seed)
    wins_all, ok = [], 0
    series = [(s, i) for s, n in POOL for i in range(n)]
    for _ in range(B):
        est, se, sid, hz = [], [], [], []
        for si, (s, i) in enumerate(series):
            u = np.exp(rng.normal(0, tau))
            t = tmpl[SRC[s]]
            for j, h in enumerate(HS):
                th = shrink * (-t.rel_vs_bolt.values[j]) * u
                s_ = t.se.values[j]
                est.append(rng.normal(th, s_)); se.append(s_); sid.append(si); hz.append(j)
        est, se, sid = np.array(est), np.array(se), np.array(sid)
        p = 2 * (1 - norm.cdf(np.abs(est) / se))
        ph = holm(p)
        wins = int(((ph < 0.05) & (est >= 0.005)).sum()); loss = int(((ph < 0.05) & (est < 0)).sum())
        wins_all.append(wins)
        pooled = est.mean()
        if wins >= K and loss == 0 and pooled >= 0.005:
            bi = rng.integers(0, len(est), (boot, len(est)))
            lo_cell = np.quantile(est[bi].mean(1), 0.025)
            m = np.empty(boot)
            for b in range(boot):
                pick = rng.integers(0, len(series), len(series))
                m[b] = np.mean(np.concatenate([est[sid == k][rng.integers(0, 5, 5)] for k in pick]))
            lo_ser = np.quantile(m, 0.025)
            ok += int(lo_cell > 0 and lo_ser > 0)
    return {"mean_wins": float(np.mean(wins_all)), "p_wins_ge_K": float(np.mean(np.array(wins_all) >= K)), "power_rule": ok / B, "wins_quantiles": np.quantile(wins_all, [0.1, 0.5, 0.9]).tolist()}


out = {"template_cells": len(d), "pool_cells": 100, "K_grid": {}}
for K in (5, 10, 15, 20):
    out["K_grid"][str(K)] = {str(sh): simulate(sh, K) for sh in (1.0, 0.75, 0.5, 0.25)}
out["null"] = {str(K): simulate(0.0, K) for K in (5, 10)}
(HERE / "power_calc.json").write_text(json.dumps(out, indent=1))
for K, v in out["K_grid"].items():
    print("K", K, {sh: round(r["power_rule"], 2) for sh, r in v.items()})
print("null P(wins>=K):", {K: r["p_wins_ge_K"] for K, r in out["null"].items()})
print("mean wins by shrink (K=10):", {sh: round(r["mean_wins"], 1) for sh, r in out["K_grid"]["10"].items()})
