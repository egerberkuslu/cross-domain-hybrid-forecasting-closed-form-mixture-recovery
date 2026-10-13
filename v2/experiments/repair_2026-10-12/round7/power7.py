"""Round 7 power (DATA7.md section c, REPAIR_PREREG_7_DRAFT.md section 6). Exact binomial, traces independent (optimistic, see the design-effect rows).

Rules (m = 3 variants, Holm; the first Holm step alpha / 3 is used for every variant, which is conservative for the 'at least one variant' event):
  R80   X >= ceil(0.8 n) AND P(Bin(n, 0.5) >= X) < alpha / 3          (round 4 to 6 rule)
  R50   P(Bin(n, 0.5) >= X) < alpha / 3                                 (binomial part alone)
  R60   P(Bin(n, 0.6) >= X) < alpha / 3                                 (H0: works on at most 60 % of the traces)
  R70   P(Bin(n, 0.7) >= X) < alpha / 3
Power = P(Bin(n, pi) >= k_rule). Design effect DE: n replaced by round(n / DE) (crude allowance for traces from one network and calendar).
Writes power7.json. Block "feasibility_adjusted" (needs frozen_r7.json): the revealed design pass rates (val7_rows.csv, val7_w4_rows.csv, pooled revealed r5 and r6, 84 traces) are split by validation feasibility
(feasible level found or not) per variant and re-weighted to the feasibility shares of the round 7 pool, pi_v = f_v p_feasible + (1 - f_v) p_infeasible; power is computed at pi_v. Block "actual_pool": the same rules on the ACTUAL primary pool of round 7 (n from degen7.json when complete, otherwise from pool7_manifest.json),
for the design inputs pi of REPAIR_PREREG_7.md section 6, with the design effect rows (n_eff = round(n / 1.5)).
"""
import json
from math import ceil, comb
from pathlib import Path

ALPHA, M = 0.05, 3
NS = [20, 30, 40, 50, 60, 80, 100, 120]
PIS = [0.70, 0.75, 0.80, 0.85, 0.90]
PIS_ACTUAL = [0.70, 0.75, 0.80, 0.821, 0.845, 0.857, 0.875, 0.881, 0.90]


def sf(k, n, p):
    return sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1))


def kmin(n, p0, a=ALPHA / M):
    for k in range(n + 1):
        if sf(k, n, p0) < a:
            return k
    return n + 1


def kr(rule, n):
    if rule == "R80":
        return max(ceil(0.8 * n), kmin(n, 0.5))
    return kmin(n, {"R50": 0.5, "R60": 0.6, "R70": 0.7}[rule])


def main():
    out = {"rules": {}, "min_n_for_power_0.8": {}, "design_effect_1.5": {}}
    for rule in ("R80", "R50", "R60", "R70"):
        out["rules"][rule] = {
            str(n): {
                "k": kr(rule, n),
                **{f"pi={p}": round(sf(kr(rule, n), n, p), 3) for p in PIS},
            }
            for n in NS
        }
        out["min_n_for_power_0.8"][rule] = {}
        for p in PIS:
            hit = next(
                (n for n in range(10, 401) if sf(kr(rule, n), n, p) >= 0.8), None
            )
            out["min_n_for_power_0.8"][rule][f"pi={p}"] = hit
        out["design_effect_1.5"][rule] = {
            str(n): {
                f"pi={p}": round(sf(kr(rule, round(n / 1.5)), round(n / 1.5), p), 3)
                for p in PIS
            }
            for n in NS
        }
    here = Path(__file__).resolve().parent
    dg = here / "degen7.json"
    if dg.exists() and json.loads(dg.read_text()).get("complete"):
        d = json.loads(dg.read_text())
        n, src = len(d["selected"]["sub"]) + len(d["selected"]["ip"]), "degen7.json (after the degenerate-expert rule)"
    else:
        pm = json.loads((here / "pool7_manifest.json").read_text())
        n, src = len(pm["monitor_primary"]["sub"]) + len(pm["monitor_primary"]["ip"]), "pool7_manifest.json (before the degenerate-expert rule)"
    ne = round(n / 1.5)
    act = {"n": n, "n_source": src, "n_eff_design_effect_1.5": ne, "alpha": ALPHA, "holm_first_step": ALPHA / M, "rules": {}}
    for rule in ("R80", "R50", "R60", "R70"):
        act["rules"][rule] = {
            "k": kr(rule, n),
            "power": {f"pi={p}": round(sf(kr(rule, n), n, p), 3) for p in PIS_ACTUAL},
            "k_design_effect_1.5": kr(rule, ne),
            "power_design_effect_1.5": {f"pi={p}": round(sf(kr(rule, ne), ne, p), 3) for p in PIS_ACTUAL},
        }
    act["min_pi_for_power_0.8_R80"] = next((round(p / 1000, 3) for p in range(500, 1001) if sf(kr("R80", n), n, p / 1000) >= 0.8), None)
    out["actual_pool"] = act
    fz = here / "frozen_r7.json"
    if fz.exists():
        import pandas as pd

        fr = json.loads(fz.read_text())
        prim = [t for t in fr["traces"] if fr["strata"][t] != "G7"]
        d = pd.concat([pd.read_csv(here / "val7_rows.csv"), pd.read_csv(here / "val7_w4_rows.csv")])
        adj = {"n": len(prim), "note": "design pass rates re-weighted to the round 7 feasibility shares; information for the power statement, not an input of any parameter"}
        for arm in ("W1", "W3", "W4"):
            g = d[d.arm == arm]
            pf, pi_ = g[g.feasible].joint.mean(), g[~g.feasible].joint.mean()
            nf, ni = int(g.feasible.sum()), int((~g.feasible).sum())
            f = sum(bool(fr["params"][t][arm].get("feasible", True)) for t in prim) / len(prim)
            pi_adj = f * pf + (1 - f) * (pi_ if ni else pf)
            adj[arm] = {"design_feasible_n": nf, "design_infeasible_n": ni, "design_pass_feasible": round(float(pf), 4),
                        "design_pass_infeasible": round(float(pi_), 4) if ni else None, "design_pass_all": round(float(g.joint.mean()), 4),
                        "round7_feasible_share": round(f, 4), "pi_adjusted": round(float(pi_adj), 4),
                        "power_R80": round(sf(kr("R80", len(prim)), len(prim), pi_adj), 3),
                        "power_R80_design_effect_1.5": round(sf(kr("R80", round(len(prim) / 1.5)), round(len(prim) / 1.5), pi_adj), 3)}
        out["feasibility_adjusted"] = adj
    (here / "power7.json").write_text(
        json.dumps(out, indent=1)
    )
    for rule, d in out["rules"].items():
        print(rule)
        for n, r in d.items():
            print("  n", n, r)
    print(json.dumps(out["min_n_for_power_0.8"], indent=1))
    print(json.dumps(out["actual_pool"], indent=1))
    print(json.dumps(out.get("feasibility_adjusted"), indent=1))


if __name__ == "__main__":
    main()
