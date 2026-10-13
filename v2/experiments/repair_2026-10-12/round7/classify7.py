"""Round 7 cause classes of the failed traces (REPAIR_PREREG_7.md sections 4 and 9, registered rule; same rule as round6/classify6_r6.py with class (2) split by the control).
Reads only the per-trace rows of the test stage; reads no forecast file.
For every (arm, primary trace) with J = false the first matching class applies, in this order:
  c1_false_alarm_excess     c1 false (no-fault false-alarm onset share above 5 %)
  c2_insensitive            c1 true, c2 false, matched no-injection control silent (ctl = 0): the detector did not see the injected shift
  c2_control_alarms         c1 true, c2 false, ctl > 0: the no-injection control also alarmed at the change points
  c3_null_exceeded          c1 and c2 true, c3' false, p0 = max(p_dense, p_rand) above 0.5 (det - p0 >= 0.5 is unreachable even at det = 1)
  c3_slow_detection         c1 and c2 true, c3' false, p0 at most 0.5
Output: r7_causes.json (counts per arm and stratum group, trace names per class). Run after the test stage:
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python classify7.py [rows.csv]
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CLASSES = ("c1_false_alarm_excess", "c2_insensitive", "c2_control_alarms", "c3_null_exceeded", "c3_slow_detection")


def cause(r):
    if r["joint"]:
        return None
    if not r["c1"]:
        return CLASSES[0]
    if not r["c2"]:
        return CLASSES[1] if r["ctl"] == 0 else CLASSES[2]
    return CLASSES[3] if r["p0"] > 0.5 else CLASSES[4]


def main(rows_csv):
    d = pd.read_csv(rows_csv)
    d["cause"] = d.apply(cause, axis=1)
    out = {}
    for arm, g in d.groupby("arm"):
        for scope, gg in (("P7", g[g.stratum != "G7"]), ("G7", g[g.stratum == "G7"])):
            if len(gg) == 0:
                continue
            f = gg[gg.cause.notna()]
            out[f"{arm}_{scope}"] = {"traces": int(len(gg)), "failed": int(len(f)), **{c: int((f.cause == c).sum()) for c in CLASSES},
                                     "names": {c: sorted(f.series[f.cause == c].tolist()) for c in CLASSES}}
    (HERE / "r7_causes.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "names"} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "r7_monitor_rows.csv")
