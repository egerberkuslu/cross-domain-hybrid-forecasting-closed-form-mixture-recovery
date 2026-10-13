"""Round 6 diagnosis, part 2: per-trace cause of the RF condition-3 misses of round 5 (revealed rows only; no new read of any test file).
Inputs: round5/r5_monitor_summary.json (table), diag6_rows.csv (diag6.py). Output: diag6_classes.csv, diag6_classes.json.
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python classify6.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paper_a"))
import stats_tools as T

s = json.loads((HERE.parent / "round5" / "r5_monitor_summary.json").read_text())
t = pd.DataFrame(s["table"])
d = pd.read_csv(HERE / "diag6_rows.csv")
d = d[(d.split == "test_revealed") & (d.detector == "RF")].set_index("series")
h = t[t.detector == "RF"].set_index("series").join(d[["placebo_pre_active", "alarm_at_burn_end", "first_onset_window", "onsets_prefix_per100", "onsets_rest_per100", "q_shift_prefix_sd", "S_at_burn_end", "level"]])
h["c2"] = h.ss_lo > 0
h["c3"] = (h.above_placebo_lo > 0) & h.c2
f = h[~h.c3].copy()
f["cause_c2_fail"] = ~f.c2
f["cause_placebo_ge_0.6"] = f.placebo_rate >= 0.6
f["cause_slow_cap100"] = (f.det_cap100 <= 0.7) & (f.placebo_rate < 0.6)
f["cause_resolution_only"] = ~f.cause_c2_fail & ~f["cause_placebo_ge_0.6"] & ~f.cause_slow_cap100
f["burn_release"] = f.alarm_at_burn_end
f.to_csv(HERE / "diag6_classes.csv")
# Newcombe resolution: largest placebo count (of 5) that still passes, for each injected count at cap 100 (of 30)
res = {x: max([k for k in range(6) if T.newcombe_diff(x, 30, k, 5)[1] > 0], default=-1) for x in range(18, 31)}
# counterfactuals on revealed rows (illustrative, disclosed, not used for any choice)
cf_same_cap = int(((h.det_rate - h.placebo_rate) > 0).sum())
out = {"rf_traces": int(len(h)), "c3_fail": int(len(f)),
       "fail_c2": int(f.cause_c2_fail.sum()), "fail_placebo_ge_0.6": int(f["cause_placebo_ge_0.6"].sum()), "fail_slow_detection_cap100_with_low_placebo": int(f.cause_slow_cap100.sum()),
       "fail_resolution_only": int(f.cause_resolution_only.sum()),
       "placebo_ge_0.6_and_alarm_on_at_burn_end": int((f["cause_placebo_ge_0.6"] & f.alarm_at_burn_end).sum()),
       "placebo_ge_0.6_traces_total_all64": int((h.placebo_rate >= 0.6).sum()),
       "max_placebo_hits_of5_passing_given_inj_hits_of30_cap100": res,
       "rf_det150_mean": float(h.det_rate.mean()), "rf_det100_mean": float(h.det_cap100.mean()),
       "traces_det100_le_0.7": int((h.det_cap100 <= 0.7).sum()), "traces_det150_eq1": int((h.det_rate == 1).sum()),
       "names": {c: sorted(f.index[f[c]].tolist()) for c in ("cause_c2_fail", "cause_placebo_ge_0.6", "cause_slow_cap100", "cause_resolution_only")}}
(HERE / "diag6_classes.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
pd.set_option("display.width", 250)
print(f[["fa", "det_rate", "det_cap100", "placebo_rate", "ss_lo", "above_placebo_lo", "placebo_pre_active", "alarm_at_burn_end", "first_onset_window", "onsets_prefix_per100", "onsets_rest_per100"]].round(3).sort_values("placebo_rate").to_string())
