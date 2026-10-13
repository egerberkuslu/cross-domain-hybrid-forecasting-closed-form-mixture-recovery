"""EXPLORATORY (not pre-registered as a headline): floor counts of the non-headline A-A1b variants on the test split.
Run after the pre-registered A-A1b test. Uses the best config per variant from frozen_a1b.json (selected on validation).
Never enters a headline; used only to say whether any mixture-retaining rule keeps the floor.
"""
import json
import numpy as np
import pandas as pd
import rcommon as R
import a1

R.install_guard()
R.allow_test()
fr = json.loads((R.HERE / "frozen_a1b.json").read_text())
best = fr["best_per_variant"]
rows = []
for ds, h in R.CELLS:
    mse = {k: [] for k in ("bolt", "dec", "V0", "V1", "V2", "V3", "V4")}
    alpha_mean = {k: [] for k in ("V1", "V2", "V3", "V4")}
    for s in R.SEEDS:
        v = R.load_val(ds, h, s)
        t = R.load_test(ds, h, s)
        y = np.vstack([v["y"], t["y"]]); dec = np.vstack([v["dec"], t["dec"]]); glob = np.vstack([v["glob"], t["glob"]])
        nh = len(v["y"]); a0 = R.deployed_alpha(ds, h, s); yt = t["y"]
        mse["bolt"].append(((t["glob"] - yt) ** 2).mean(1)); mse["dec"].append(((t["dec"] - yt) ** 2).mean(1))
        for var in ("V0", "V1", "V2", "V3", "V4"):
            cfg = {} if var == "V0" else best[var]["cfg"]
            f, a = a1.forecast(var, cfg, y, dec, glob, nh, h, a0, np.arange(nh))
            mse[var].append(((f - yt) ** 2).mean(1))
            if var != "V0":
                alpha_mean[var].append(float(np.mean(a)))
    m = {k: np.mean(v, axis=0) for k, v in mse.items()}
    ref = min(np.sqrt(m["bolt"].mean()), np.sqrt(m["dec"].mean()))
    row = {"dataset": ds, "horizon": h, "rmse_bolt": float(np.sqrt(m["bolt"].mean()))}
    for var in ("V0", "V1", "V2", "V3", "V4"):
        r = float(np.sqrt(m[var].mean()))
        row[f"rel_ref_{var}"] = r / ref - 1.0
        row[f"rel_bolt_{var}"] = r / row["rmse_bolt"] - 1.0
        if var != "V0":
            row[f"alpha_{var}"] = float(np.mean(alpha_mean[var]))
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(R.HERE / "a1b_explore_cells.csv", index=False)
out = {}
for var in ("V0", "V1", "V2", "V3", "V4"):
    out[var] = {"floor_holds_cells": int((df[f"rel_ref_{var}"] <= 0.001).sum()), "max_excess_pct": float(100 * df[f"rel_ref_{var}"].max()),
                "mean_rel_vs_bolt_pct": float(100 * df[f"rel_bolt_{var}"].mean()), "mean_alpha": float(df[f"alpha_{var}"].mean()) if var != "V0" else None}
(R.HERE / "a1b_explore_summary.json").write_text(json.dumps({"EXPLORATORY": True, "cfg": {k: best[k]["cfg"] for k in ("V1", "V2", "V3", "V4")}, "results": out}, indent=1))
print(json.dumps(out, indent=1))
