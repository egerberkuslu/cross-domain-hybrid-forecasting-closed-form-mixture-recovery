"""Gate 0c (informational): test-side constant-weight oracle headroom from EXISTING published numbers only.

The mixture MSE is exactly quadratic in alpha, so three stored test RMSEs fix it:
alpha = 0 (Chronos-Bolt), alpha = 1 (decomposition only) and alpha = alpha_grid (shipped CHA-S).
Inputs: experiments/fix_2026-10-11/closed_form_test_accuracy.csv (already used in the paper). No forecast file
under outputs/predictions is opened by this script, so the sealed test split is not read again.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import rcommon as R

d = pd.read_csv(R.FIX / "closed_form_test_accuracy.csv")
rows = []
for r in d.itertuples():
    m0, m1, ma = r.rmse_bolt ** 2, r.rmse_decomp_only ** 2, r.rmse_chas_grid_recomputed ** 2
    a = r.alpha_grid
    if a in (0.0, 1.0):
        rows.append((r.dataset, r.horizon, r.seed, np.nan, 0.0, a, 0.0 if a == 0.0 else 1 - np.sqrt(ma / m0)))
        continue
    # M(x) = m0 + b x + c x^2 with M(1) = m1: b + c = m1 - m0 ; M(a) = ma
    c = ((ma - m0) - a * (m1 - m0)) / (a * a - a)
    b = (m1 - m0) - c
    x = np.clip(-b / (2 * c), 0, 1) if c > 1e-12 else (0.0 if b > 0 else 1.0)
    mo = m0 + b * x + c * x * x
    rows.append((r.dataset, r.horizon, r.seed, x, 1 - np.sqrt(mo / ma), a, 1 - np.sqrt(mo / m0)))
o = pd.DataFrame(rows, columns=["dataset", "horizon", "seed", "alpha_test_oracle", "headroom_const_oracle", "alpha_grid", "oracle_gain_vs_bolt"])
c = o.groupby(["dataset", "horizon"]).agg(alpha_test_oracle=("alpha_test_oracle", "mean"), headroom=("headroom_const_oracle", "mean"), alpha_grid=("alpha_grid", "mean"), oracle_gain_vs_bolt=("oracle_gain_vs_bolt", "mean")).reset_index()
fb = pd.read_csv(R.FIX / "floor_bound_test.csv")[["dataset", "horizon", "excess_pct_grid", "floor_holds_grid"]]
c = c.merge(fb, on=["dataset", "horizon"])
c.to_csv(R.HERE / "gate0c_cells.csv", index=False)
s = {
    "cells_headroom_ge_0.5pct": int((c.headroom >= 0.005).sum()),
    "mean_headroom": float(c.headroom.mean()),
    "max_headroom": float(c.headroom.max()),
    "cells_floor_fails": int((~c.floor_holds_grid.astype(bool)).sum()),
    "mean_excess_pct_where_floor_fails": float(c[~c.floor_holds_grid.astype(bool)].excess_pct_grid.mean()),
    "max_excess_pct": float(c.excess_pct_grid.max()),
    "max_test_oracle_constant_gain_vs_bolt": float(c.oracle_gain_vs_bolt.max()),
    "mean_test_oracle_constant_gain_vs_bolt": float(c.oracle_gain_vs_bolt.mean()),
    "cells_test_oracle_gain_vs_bolt_ge_0.5pct": int((c.oracle_gain_vs_bolt >= 0.005).sum()),
}
(R.HERE / "gate0c_summary.json").write_text(json.dumps(s, indent=2))
print(json.dumps(s, indent=2))
print(c.round(4).to_string())
