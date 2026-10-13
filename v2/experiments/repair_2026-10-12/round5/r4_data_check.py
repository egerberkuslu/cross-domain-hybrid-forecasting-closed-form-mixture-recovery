"""Sanity check of the round 5 statistics on the ROUND 4 cells (already revealed; not a test). Writes r4_data_check.txt.
Command (from /home/ege/Desktop/zaman-serisi-yayin/experiments/repair_2026-10-12/round5):
  PYTHONDONTWRITEBYTECODE=1 ../../../.chas-venv/bin/python r4_data_check.py > r4_data_check.txt 2>/dev/null"""
import sys

import numpy as np
import pandas as pd

sys.argv = ["x"]
sys.path[:0] = [".", "../round4", "../round3", "../paper_a", "../fresh_cells"]
import r5_eval as E  # noqa: E402
import stats_tools as T  # noqa: E402

d = pd.read_csv("../round4/r4_test_cells.csv")
ser = d.series.values
cl = list(zip(d.series, d.horizon))
print("Sanity check of the registered statistics on the ROUND 4 cells (already revealed; not a test)")
for nm, g in [("bolt", d.gain_vs_bolt), ("ctrl_a", 1 - d.rmse_var / d.rmse_ctrl_a), ("ctrl_b", 1 - d.rmse_var / d.rmse_ctrl_b), ("best_of_two", d.gain_vs_best_ctrl)]:
    r = E.ci_p(g.values, ser)
    print(nm, {k: round(v, 6) for k, v in r.items()})
    print("  T.cell_cluster_boot_mean two_stage", [round(x, 6) for x in T.cell_cluster_boot_mean(g.values, cl, two_stage=True)])
pa = E.ci_p((1 - d.rmse_var / d.rmse_ctrl_a).values, ser, seed=1)
pb = E.ci_p((1 - d.rmse_var / d.rmse_ctrl_b).values, ser, seed=2)
print("holm adj", E.R.holm(np.array([pa["p_one_sided"], pb["p_one_sided"]])))
print("pooled gain over each control vs the primary minimum 0.219 %%: a %.3f %%, b %.3f %%; vs the secondary 0.5 %%: b would FAIL (%.3f %%)" % (100 * pa["mean"], 100 * pb["mean"], 100 * pb["mean"]))
