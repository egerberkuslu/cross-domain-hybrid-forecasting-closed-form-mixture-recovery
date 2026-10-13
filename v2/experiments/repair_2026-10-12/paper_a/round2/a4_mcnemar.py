"""Pre-registered reporting item of REPAIR_PREREG_2.md section 5: exact McNemar per (trace, placement), headline DA against M0 and M1, Holm over 15. Reads a4_test_runs.csv."""
import json, sys
import pandas as pd
sys.path.insert(0, "..")
import stats_tools as T
from rcommon import holm
r = pd.read_csv("a4_test_runs.csv")
r = r[r.kind == "inj"]
out = {}
for other in ("M0", "M1"):
    ps, lab = [], []
    for (ds, pos), g in r.groupby(["dataset", "pos"]):
        w = g.pivot_table(index=["seed", "inj"], columns="detector", values="detected", aggfunc="first")
        b = int(((w["DA"] == 1) & (w[other] == 0)).sum()); c = int(((w["DA"] == 0) & (w[other] == 1)).sum())
        ps.append(T.mcnemar_exact(b, c)); lab.append((ds, pos, b, c))
    ph = holm(__import__("numpy").array(ps))
    out[other] = [{"dataset": l[0], "pos": l[1], "da_only": l[2], "other_only": l[3], "p": float(p), "p_holm": float(q)} for l, p, q in zip(lab, ps, ph)]
json.dump(out, open("a4_mcnemar.json", "w"), indent=1)
for k, v in out.items():
    print(k, "significant after Holm:", [(x["dataset"], x["pos"], x["da_only"], x["other_only"]) for x in v if x["p_holm"] < 0.05])
