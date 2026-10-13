"""Validation-only: error-variance ratio r and error correlation rho of decomposition vs Chronos-Bolt per cell and the implied
oracle constant-mixture gain. Inputs: experiments/fix_2026-10-11/closed_form_cells.csv (M0, M1, leading coefficient A on VALIDATION)."""
import numpy as np, pandas as pd
import rcommon as R
c = pd.read_csv(R.FIX / "closed_form_cells.csv")
c = c[c.variant == "cha_hybrid_v3"].copy()
c["r"] = c.M1 / c.M0
c["rho"] = (c.M0 + c.M1 - c.leading_coef_A) / 2 / np.sqrt(c.M0 * c.M1)
g = c.groupby(["dataset", "horizon"])[["r", "rho"]].mean().reset_index()
g["oracle_gain_rmse"] = 1 - np.sqrt(g.r * (1 - g.rho ** 2) / (1 + g.r - 2 * g.rho * np.sqrt(g.r)))
g.to_csv(R.HERE / "ceiling_r_rho_cells.csv", index=False)
print(g[["r", "rho", "oracle_gain_rmse"]].describe().round(4))
