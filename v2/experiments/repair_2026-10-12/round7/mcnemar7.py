"""Registered secondary claim of REPAIR_PREREG_7.md section 6: one-sided exact McNemar
of each succeeding variant against STB7 on the 40 P7 traces, Holm across the succeeding
variants. Also re-derives X per arm from r7_monitor_rows.csv and compares with the summary."""
import csv, json, sys
from math import comb

H = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
rows = list(csv.DictReader(open(f"{H}/r7_monitor_rows.csv")))
summ = json.load(open(f"{H}/r7_monitor_summary.json"))


def joint(r):
    return r["joint"].strip().lower() == "true"


def table(stratum_prefix):
    t = {}
    for r in rows:
        if r["stratum"].startswith(stratum_prefix):
            t.setdefault(r["arm"], {})[r["series"]] = joint(r)
    return t


P = table("P7")
need = summ["P7"]["need"]
out = {"X_rederived": {}, "X_summary": {}, "mismatch": []}
for arm, d in P.items():
    out["X_rederived"][arm] = sum(d.values())
    out["X_summary"][arm] = summ["P7"]["arms"][arm]["X"]
    if len(d) != 40 or out["X_rederived"][arm] != out["X_summary"][arm]:
        out["mismatch"].append(arm)
if out["mismatch"]:
    print("MISMATCH", out["mismatch"], out["X_rederived"], out["X_summary"])
    json.dump(out, open(f"{H}/mcnemar7.json", "w"), indent=1)
    sys.exit(1)

# succeeding variants by the registered rule, recomputed from the rows
n = 40
succ = []
raw = {}
for v in ("W1", "W3", "W4"):
    x = out["X_rederived"][v]
    p = sum(comb(n, k) for k in range(x, n + 1)) / 2**n
    raw[v] = p
# Holm across the three variants for the success rule
order = sorted(raw, key=raw.get)
holm3, run = {}, 0.0
for i, v in enumerate(order):
    run = max(run, min(1.0, raw[v] * (3 - i)))
    holm3[v] = run
rand_ok = {v: summ["P7"]["arms"][v]["rand_ctrl_c3_share_mean"] <= 0.05 for v in raw}
for v in raw:
    if out["X_rederived"][v] >= need and holm3[v] < 0.05 and rand_ok[v]:
        succ.append(v)
out["success_rule"] = {"p_binom": raw, "p_holm3": holm3, "random_control_ok": rand_ok, "succeeding": succ}
hl = max(succ, key=lambda v: (out["X_rederived"][v], -["W1", "W3", "W4"].index(v)))
out["headline"] = hl

base = P["STB7"]
mc = {}
for v in succ:
    a = P[v]
    keys = sorted(a)
    b = sum(1 for k in keys if a[k] and not base[k])   # variant only
    c = sum(1 for k in keys if base[k] and not a[k])   # STB7 only
    m = b + c
    p = sum(comb(m, k) for k in range(b, m + 1)) / 2**m if m else 1.0
    mc[v] = {"v_only": b, "stb7_only": c, "both": sum(1 for k in keys if a[k] and base[k]),
             "neither": sum(1 for k in keys if not a[k] and not base[k]), "p_one_sided": p}
order = sorted(mc, key=lambda v: mc[v]["p_one_sided"])
run = 0.0
for i, v in enumerate(order):
    run = max(run, min(1.0, mc[v]["p_one_sided"] * (len(order) - i)))
    mc[v]["p_holm"] = run
    mc[v]["claim"] = run < 0.05
out["mcnemar_vs_STB7"] = mc
out["claim_holds_for_all_succeeding"] = all(mc[v]["claim"] for v in mc)
json.dump(out, open(f"{H}/mcnemar7.json", "w"), indent=1)
print(json.dumps(out, indent=1))
