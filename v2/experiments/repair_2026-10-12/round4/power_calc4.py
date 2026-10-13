"""Round 4 power: effect distribution of ROUND 3 (the 100 fresh cells of the first fresh pool), no pool-2 data involved.

Round 3 per-cell test results (`../round3/r3_test_cells.csv`): estimated gain over Chronos-Bolt and its bootstrap se = (ci_hi - ci_lo) / 3.92. Pool 2 is simulated from
the round 3 SERIES of the same source: a pool-2 series copies the five (gain, se) cells of a round 3 series drawn with replacement (series-level heterogeneity is thus
empirical). Scenarios for the true gain theta of a cell:
  MEDIAN  theta = 1.02 % x u_series, u lognormal(0, 0.5): the round 3 median cell gain, with series-to-series heterogeneity (the planning scenario)
  EMP     theta = the round 3 estimate itself (mean 0.50 %, heavy negative tail from one Abilene series)
  TAIL    theta = max(round 3 estimate, 0): the same pool with the loss tail removed
estimate ~ N(theta, se); cell p two-sided normal; Holm over all cells; win = Holm p < 0.05 and gain >= 0.5 %; loss = Holm p < 0.05 and gain < 0.
Rule (REPAIR_PREREG_4.md section 4) = wins >= ceil(0.08 x cells), no loss, pooled gain >= 0.5 %, cell-level AND series-clustered 95 % lower bounds > 0, median cell gain > 0.
Output power_calc4.json.
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE.parent / "round3" / "r3_test_cells.csv")
d["src"] = d.series.str.replace(r"\d.*$", "", regex=True).str.rstrip("_")
d["se"] = (d.ci_hi - d.ci_lo) / 3.92
d["gain"] = d.gain_vs_bolt
ser = {s.split("_")[0]: [g.sort_values("horizon")[["gain", "se"]].values for _, g in x.groupby("series")] for s, x in d.groupby("src")}
MEDGAIN = float(d.gain.median())


def holm(p):
    n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def simulate(comp, scenario, B=400, boot=150, seed=3):
    rng = np.random.default_rng(seed)
    ok = 0
    wins_all = []
    K = int(np.ceil(0.08 * 5 * sum(comp.values())))
    for _ in range(B):
        est, se, sid = [], [], []
        k = 0
        for src, n in comp.items():
            for _i in range(n):
                tmpl = ser[src][rng.integers(0, len(ser[src]))]
                u = np.exp(rng.normal(0, 0.5))
                for g, s in tmpl:
                    th = {"MEDIAN": MEDGAIN * u, "EMP": g, "TAIL": max(g, 0.0)}[scenario]
                    est.append(rng.normal(th, s)); se.append(s); sid.append(k)
                k += 1
        est, se, sid = np.array(est), np.array(se), np.array(sid)
        p = 2 * (1 - norm.cdf(np.abs(est) / se))
        ph = holm(p)
        wins = int(((ph < 0.05) & (est >= 0.005)).sum()); loss = int(((ph < 0.05) & (est < 0)).sum())
        wins_all.append(wins)
        if wins >= K and loss == 0 and est.mean() >= 0.005 and np.median(est) > 0:
            bi = rng.integers(0, len(est), (boot, len(est)))
            lo_c = np.quantile(est[bi].mean(1), 0.025)
            m = np.empty(boot)
            for b in range(boot):
                pick = rng.integers(0, k, k)
                m[b] = np.mean(np.concatenate([est[sid == q][rng.integers(0, 5, 5)] for q in pick]))
            ok += int(lo_c > 0 and np.quantile(m, 0.025) > 0)
    return {"K": K, "power": ok / B, "mean_wins": float(np.mean(wins_all))}


COMPS = {"40": {"cesnet": 20, "abilene": 8, "geant": 12}, "60": {"cesnet": 30, "abilene": 10, "geant": 20}, "75": {"cesnet": 40, "abilene": 10, "geant": 25}}  # all within the A half (CESNET 45, Abilene 10, GEANT 111)
out = {"median_gain_round3": MEDGAIN, "comps": COMPS, "res": {}}
for name, comp in COMPS.items():
    out["res"][name] = {sc: simulate(comp, sc) for sc in ("MEDIAN", "EMP", "TAIL")}
    print(name, {sc: (r["K"], round(r["power"], 2), round(r["mean_wins"], 1)) for sc, r in out["res"][name].items()}, flush=True)
(HERE / "power_calc4.json").write_text(json.dumps(out, indent=1))
