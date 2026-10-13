"""EXPLORATORY diagnostic for A-A2 (post-test, never a headline): validation versus test alpha-stream statistics per trace."""
import json
import numpy as np
import pandas as pd
import rcommon as R
import a2
import monitor_far as mf

R.install_guard(); R.allow_test()
fr = json.loads((R.HERE / "frozen_a2.json").read_text())
rows = []
for ds in R.DATASETS:
    for s in R.SEEDS:
        P = fr["params"][f"{ds}|{s}"]
        yv, dv, gv = a2.val_stream(ds, s)
        yt, dt, gt = a2.test_stream(ds, s, False)
        _, av, rv, qv, ev = a2.streams(yv, dv, gv)
        _, at, rt, qt, et = a2.streams(yt, dt, gt)
        sv, st = mf.smooth(av, "ema"), mf.smooth(at, "ema")
        rows.append({"dataset": ds, "seed": s, "a_mean_val": av.mean(), "a_mean_test": at.mean(), "a_sd_val": av.std(), "a_sd_test": at.std(),
                     "sm_q95_val": np.quantile(sv, .95), "sm_q95_test": np.quantile(st, .95), "tau_M1": P["M1"]["tau"], "h_M1": P["M1"]["h"],
                     "share_test_above_tau": float((st >= P["M1"]["tau"]).mean()), "share_val_above_tau": float((sv >= P["M1"]["tau"]).mean()),
                     "cusum_h_feasible": P["M1"]["feasible"], "n_val": len(av), "n_test": len(at)})
df = pd.DataFrame(rows)
df.to_csv(R.HERE / "a2_diag.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(df.groupby("dataset").mean(numeric_only=True).round(3).T)
