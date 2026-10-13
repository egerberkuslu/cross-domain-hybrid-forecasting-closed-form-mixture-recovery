"""Round 5 power (accuracy part): effect distribution of ROUND 4 (300 cells, 60 series), no round 5 data involved.

Round 4 per-cell test results (`../round4/r4_test_cells.csv`): gain over Chronos-Bolt, gain over the permuted-rows control (a) and over the null expert (b),
raw paired-permutation p over Chronos-Bolt. A round 5 pool is simulated from the round 4 SERIES of the same source (series drawn with replacement, its five cells copied, so the
series-to-series heterogeneity is the empirical one and the observed gains already carry their sampling noise). Sources absent from round 4 are not simulated.
Scenarios:
  EMP    round 4 gains as observed (mean +0.60 % over Bolt, +0.64 % over control a, +0.43 % over control b)
  HALF   all three gain families shifted down by half of their round 4 pooled mean (a decayed effect; p values and win flags kept)
  NET    the round 4 per-cell-minimum comparison (gain over the better of the two controls, mean +0.14 %), for comparison with the round 4 rule
Power is computed for the conditions that can bind: pooled gain over Bolt >= m with the
series-clustered 95 % lower bound > 0; each control: series-clustered one-sided bootstrap p, Holm over the two controls at 0.05, and the pooled gain over each control at least m; median cell gain over Bolt > 0. The K-wins and no-loss conditions are reported as simulated counts only.
m = 0.219 % primary, 0.5 % secondary. The series-clustered interval is approximated here by a single-stage bootstrap of the series means (B = 400).
Output power_calc5.json.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE.parent / "round4" / "r4_test_cells.csv")
d["src"] = d.series.str.replace(r"\d.*$", "", regex=True).str.rstrip("_").str.replace("_od", "").str.replace("_inst", "")
d["gb"] = d.gain_vs_bolt
d["ga"] = 1 - d.rmse_var / d.rmse_ctrl_a
d["gc"] = 1 - d.rmse_var / d.rmse_ctrl_b
d["gnet"] = d.gain_vs_best_ctrl
d["praw"] = d.p_vs_bolt
SER = {s: [g.sort_values("horizon")[["gb", "ga", "gc", "gnet", "praw"]].values for _, g in x.groupby("series")] for s, x in d.groupby("src")}
MEAN = {k: float(d[k].mean()) for k in ("gb", "ga", "gc", "gnet")}


def holm(p):
    n = len(p); o = np.argsort(p); adj = np.empty(n); run = 0.0
    for i, k in enumerate(o):
        run = max(run, (n - i) * p[k]); adj[k] = min(1.0, run)
    return adj


def one_sided_p(means, rng, B=400):
    bi = rng.integers(0, len(means), (B, len(means)))
    return (1 + (means[bi].mean(1) <= 0).sum()) / (B + 1)


def simulate(comp, scenario, minimum, n_sim=400, seed=5):
    rng = np.random.default_rng(seed)
    ok = np.zeros(n_sim, bool); ok_old = np.zeros(n_sim, bool); wins_all = []; loss_all = []
    K = int(np.ceil(0.08 * 5 * sum(comp.values())))
    for t in range(n_sim):
        rows = []; sid = []; k = 0
        for src, n in comp.items():
            for _ in range(n):
                tm = SER[src][rng.integers(0, len(SER[src]))]
                rows.append(tm); sid += [k] * len(tm); k += 1
        X = np.vstack(rows); sid = np.array(sid)
        gb, ga, gc, gn, p = (X[:, i].copy() for i in range(5))
        if scenario == "HALF":
            gb -= 0.5 * MEAN["gb"]; ga -= 0.5 * MEAN["ga"]; gc -= 0.5 * MEAN["gc"]; gn -= 0.5 * MEAN["gnet"]
        ph = holm(p)
        win = int(((ph < 0.05) & (gb >= minimum)).sum()); loss = int(((ph < 0.05) & (gb < 0)).sum())
        wins_all.append(win); loss_all.append(loss)
        sm = lambda g: np.array([g[sid == q].mean() for q in range(k)])
        smb = sm(gb)
        bi = rng.integers(0, k, (400, k))
        lo_b = np.quantile(smb[bi].mean(1), 0.025)
        base = gb.mean() >= minimum and lo_b > 0 and np.median(gb) > 0  # K wins and no-loss are reported as expected counts: they are not binding (round 4: 66 wins for K = 24, no loss) and the permutation p-value resolution caps the simulated Holm wins on large pools
        if scenario == "NET":
            sc = sm(gn); ctl = ctl_old = one_sided_p(sc, rng) < 0.05
        else:
            pa, pc = one_sided_p(sm(ga), rng), one_sided_p(sm(gc), rng)
            ctl_old = bool(holm(np.array([pa, pc])).max() < 0.05)
            ctl = ctl_old and ga.mean() >= minimum and gc.mean() >= minimum
        ok[t] = base and ctl
        ok_old[t] = base and ctl_old
    return {"K": K, "power": float(ok.mean()), "power_without_control_mean_condition": float(ok_old.mean()), "mean_wins": float(np.mean(wins_all)), "mean_losses": float(np.mean(loss_all))}


# A-half unused accuracy series: CESNET 13, GEANT 91 (cesnet_inst262 and 275 were monitor-only in round 4 and are excluded); F stratum: 100 per source
COMPS = {
    "A104": {"cesnet": 13, "geant": 91},
    "A104+Fgeant100": {"cesnet": 13, "geant": 191},
    "A104+F300": {"cesnet": 113, "abilene": 100, "geant": 191},
    "round4_60": {"cesnet": 30, "abilene": 10, "geant": 20},
}
out = {"mean_gain_round4": MEAN, "comps": COMPS, "res": {}}
for name, comp in COMPS.items():
    out["res"][name] = {}
    for sc in ("EMP", "HALF", "NET"):
        for mn, m in (("primary", 0.00219), ("secondary", 0.005)):
            r = simulate(comp, sc, m)
            out["res"][name][f"{sc}_{mn}"] = r
    print(name, {k: (r["K"], round(r["power"], 2)) for k, r in out["res"][name].items()}, flush=True)
(HERE / "power_calc5.json").write_text(json.dumps(out, indent=1))
