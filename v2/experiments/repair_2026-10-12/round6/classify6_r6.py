"""Round 6 cause classes of the failed traces (REPAIR_PREREG_6.md section 9, registered rule). Reads only the per-trace rows of the test stage; reads no forecast file.
For every (arm, P2 trace) with J = false the first matching class applies, in this order:
  c1_false_alarm_excess     c1 false (no-fault false-alarm onset share above 5 %)
  c2_not_shift_specific     c1 true, c2 false (Newcombe lower bound of det - ctl not above 0)
  c3_null_exceeded          c1 and c2 true, c3' false, p0 = max(p_dense, p_rand) above 0.5 (det - p0 >= 0.5 is unreachable even at det = 1)
  c3_slow_detection         c1 and c2 true, c3' false, p0 at most 0.5 (the injected-run detection share within the cap of 150 stays below p0 + 0.5)
Output: r6_causes.json (counts per arm and per stratum, trace names per class). Run after the test stage:
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python classify6_r6.py [rows.csv]
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CLASSES = ("c1_false_alarm_excess", "c2_not_shift_specific", "c3_null_exceeded", "c3_slow_detection")


def cause(r):
    if r["joint"]:
        return None
    if not r["c1"]:
        return CLASSES[0]
    if not r["c2"]:
        return CLASSES[1]
    return CLASSES[2] if r["p0"] > 0.5 else CLASSES[3]


def main(rows_csv):
    d = pd.read_csv(rows_csv)
    d["cause"] = d.apply(cause, axis=1)
    out = {}
    for arm, g in d.groupby("arm"):
        for scope, gg in (("P2", g[g.stratum != "P3"]), ("P3", g[g.stratum == "P3"])):
            if len(gg) == 0:
                continue
            f = gg[gg.cause.notna()]
            out[f"{arm}_{scope}"] = {"traces": int(len(gg)), "failed": int(len(f)), **{c: int((f.cause == c).sum()) for c in CLASSES},
                                     "names": {c: sorted(f.series[f.cause == c].tolist()) for c in CLASSES}}
    (HERE / "r6_causes.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "names"} for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "r6_monitor_rows.csv")
