"""Round 5 power (monitor part): per-trace pass pattern of ROUND 4 (RF detector, 12 traces), no round 5 data involved.

A trace passes condition 1 (false alarm <= 5 %), condition 2 (shift-specific, Newcombe lower bound > 0), condition 3 (above the placebo clock and shift-specific), as coded in
round4/r4_monitor.summarize. The pattern of a round 5 trace is drawn from the round 4 patterns of the same source (CESNET 6, Abilene 6) with a Jeffreys-type Dirichlet(0.5)
posterior over the eight patterns, so the unknown pass probabilities are integrated out. Traces are independent in this calculation (an optimistic assumption for the 53 Abilene OD
traces of one network over the same 699 windows). Rule: c1 >= need, c2 >= need, c3 >= min(c2, need), need = ceil(0.8 n).
Scenarios: POST (the Dirichlet posterior above), P90 (every condition passes with probability 0.90 independently), P95 (0.95).
Output power_monitor5.json.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
t = pd.DataFrame(json.loads((HERE.parent / "round4" / "r4_monitor_summary.json").read_text())["table"])
t = t[t.detector == "RF"].copy()
t["src"] = np.where(t.series.str.startswith("abilene"), "abilene", "cesnet")
t["c1"] = (t.fa <= 0.05).astype(int)
t["c2"] = (t.ss_lo > 0).astype(int)
t["c3"] = ((t.above_placebo_lo > 0) & (t.ss_lo > 0)).astype(int)
PAT = {s: np.array([[r.c1, r.c2, r.c3] for r in g.itertuples()]) for s, g in t.groupby("src")}
ALL = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)])


def sim(comp, scen, n_sim=4000, seed=7):
    rng = np.random.default_rng(seed)
    n = sum(comp.values())
    need = int(np.ceil(0.8 * n))
    ok = 0
    for _ in range(n_sim):
        rows = []
        for s, k in comp.items():
            if scen == "POST":
                cnt = np.array([(PAT[s] == p).all(1).sum() for p in ALL]) + 0.5
                w = rng.dirichlet(cnt)
                rows.append(ALL[rng.choice(8, size=k, p=w)])
            else:
                q = {"P90": 0.90, "P95": 0.95}[scen]
                rows.append((rng.random((k, 3)) < q).astype(int))
        X = np.vstack(rows)
        c1, c2, c3 = X[:, 0].sum(), X[:, 1].sum(), X[:, 2].sum()
        ok += int(c1 >= need and c2 >= need and c3 >= min(c2, need))
    return {"n": n, "need": need, "power": ok / n_sim}


COMPS = {"12_round4_design": {"cesnet": 6, "abilene": 6}, "30": {"cesnet": 11, "abilene": 19}, "64_all_F": {"cesnet": 11, "abilene": 53}}
out = {"round4_patterns_by_source": {s: p.sum(0).tolist() for s, p in PAT.items()}, "n_traces_by_source": {s: len(p) for s, p in PAT.items()}, "res": {}}
for name, comp in COMPS.items():
    out["res"][name] = {sc: sim(comp, sc) for sc in ("POST", "P90", "P95")}
    print(name, {k: (v["need"], round(v["power"], 2)) for k, v in out["res"][name].items()}, flush=True)
(HERE / "power_monitor5.json").write_text(json.dumps(out, indent=1))
