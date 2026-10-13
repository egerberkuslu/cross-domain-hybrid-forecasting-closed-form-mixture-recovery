"""Round 6 power of the trace-count rules (closed form, binomial; traces treated as independent, which the Abilene OD components are not).
  rule A  X >= ceil(0.8 n)                                          (the round 4/5 fraction)
  rule B  one-sided exact binomial test of H0: pi <= 0.70, Holm over 3 variants (worst case alpha/3 = 0.0167 for the first rejection; alpha = 0.05 single)
Output: power6.json, power6_combined.json and (after the select stage) power6_scenarios.json. Run: PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python power6.py
"""
import json
from math import ceil
from pathlib import Path
import numpy as np
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
NS = [11, 18, 20, 22, 31, 40, 49, 64]
PIS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98]


def k_rule_b(n, alpha, pi0=0.70):
    """smallest k with P(Binom(n, pi0) >= k) <= alpha"""
    for k in range(n + 1):
        if binom.sf(k - 1, n, pi0) <= alpha:
            return k
    return n + 1


def need_pi(n, k, target=0.8):
    grid = np.arange(0.5, 1.0001, 0.001)
    ok = [p for p in grid if binom.sf(k - 1, n, p) >= target]
    return float(round(ok[0], 3)) if ok else None


out = {}
for n in NS:
    kA = ceil(0.8 * n)
    kB1, kB3 = k_rule_b(n, 0.05), k_rule_b(n, 0.05 / 3)
    out[n] = {"k_ruleA": kA, "k_ruleB_alpha05": kB1, "k_ruleB_holm_first_alpha0167": kB3,
              "pi_for_power80_ruleA": need_pi(n, kA), "pi_for_power80_ruleB_holm": need_pi(n, kB3) if kB3 <= n else None,
              "power_ruleA": {p: round(float(binom.sf(kA - 1, n, p)), 3) for p in PIS},
              "power_ruleB_holm": {p: (round(float(binom.sf(kB3 - 1, n, p)), 3) if kB3 <= n else 0.0) for p in PIS}}
(HERE / "power6.json").write_text(json.dumps(out, indent=1))
for n, v in out.items():
    print(n, v)

# combined rule of REPAIR_PREREG_6_DRAFT.md section 6: k = max(ceil(0.8 n), Holm-first count against pi0 = 0.5 at alpha / 3)
comb_out = {}
for n in [11, 18, 20, 22, 31, 40, 64]:
    kA = ceil(0.8 * n)
    kH = min(k for k in range(n + 2) if binom.sf(k - 1, n, 0.5) <= 0.05 / 3)
    k = max(kA, kH)
    comb_out[n] = {"k": k, "kA": kA, "k_holm_vs_0.5": kH, "pi_for_power80": need_pi(n, k), **{f"power_pi{p}": round(float(binom.sf(k - 1, n, p)), 3) for p in [0.84, 0.86, 0.875, 0.9, 0.95, 0.98]}}
(HERE / "power6_combined.json").write_text(json.dumps(comb_out, indent=1))
print(json.dumps(comb_out))


# ---- scenarios of REPAIR_PREREG_6.md section 6 (run after the select stage; reads frozen_r6.json feasibility and the revealed round 5 pass rates of val6_rows.csv)
FROZEN = HERE / "frozen_r6.json"
if FROZEN.exists():
    import pandas as pd

    def pwr(ps, k):
        d = np.array([1.0])
        for p in ps:
            d = np.convolve(d, [1 - p, p])
        return float(d[k:].sum())

    def k_comb(n):
        return max(ceil(0.8 * n), min(k for k in range(n + 2) if binom.sf(k - 1, n, 0.5) <= 0.05 / 3))

    fr = json.loads(FROZEN.read_text())
    p2 = [n for n in fr["traces"] if fr["strata"][n].startswith("P2")]
    rev = pd.read_csv(HERE / "val6_rows.csv")
    rev = rev[rev.split == "revealed_r5_test"]
    FAIL = ["cesnet_inst159", "cesnet_inst217"]
    sc = {}
    for arm in ("V1", "V2", "V3"):
        g = rev[rev.arm == arm]
        pf, pi = float(g[g.feasible].joint.mean()), float(g[~g.feasible].joint.mean())
        feas = {n: bool(fr["params"][n][arm].get("feasible", True)) for n in p2}
        n, k = len(p2), k_comb(len(p2))
        nf = sum(feas.values())
        sc[arm] = {
            "n": n, "k": k, "feasible_P2": nf, "revealed_pass_rate_feasible": round(pf, 4), "revealed_pass_rate_infeasible": round(pi, 4),
            "revealed_pass_rate_all64": round(float(g.joint.mean()), 4),
            "power_homogeneous_revealed_rate": round(pwr([float(g.joint.mean())] * n, k), 3),
            "power_feasible_infeasible_split": round(pwr([pf if feas[x] else pi for x in p2], k), 3),
            "power_split_and_two_traces_fail": round(pwr([0.0 if x in FAIL else (pf if feas[x] else pi) for x in p2], k), 3),
            "k_if_infeasible_traces_dropped": k_comb(nf),
            "power_infeasible_traces_dropped": round(pwr([pf] * nf, k_comb(nf)), 3),
            "infeasible_traces": [x for x in p2 if not feas[x]],
        }
    sc["note"] = {"fail_traces": FAIL, "rates": "revealed round 5 test, validation-selected variants, by validation feasibility (val6_rows.csv)"}
    (HERE / "power6_scenarios.json").write_text(json.dumps(sc, indent=1))
    print(json.dumps(sc, indent=1))
