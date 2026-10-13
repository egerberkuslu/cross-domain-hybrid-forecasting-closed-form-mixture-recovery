"""Round 7 pool-size alternatives (INFORMATION for the coordinator; not used by any frozen stage). Counts only: how many drifting IP-address and subnet series would be admitted
under (a) the round 6 CESNET missing limit 0.05 (the registered rule), (b) the DATA7.md rule M7a (train missing <= 20 %, longest train gap <= 168 h, zero <= 5 %, median > 0,
>= 690 complete test windows with <= 1 break), and with the family cap 3, 4, 5 or none. Values are read for train and validation hours only (shift); for test only the positions of
missing hours are counted. Writes alt7_counts.json. The shift used here is data7_inventory.profile's tv_shift (interpolated series), which differs marginally from build_pool4.shift used by the registered
screen (58 against 59 drifting subnets under the registered rule after the round 3 parent exclusion); the counts are information, not the registered counts (pool7_manifest.json).
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python alt7_counts.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_pool7 as B  # noqa: E402
import data7_inventory as D  # noqa: E402

F = B.F


def main():
    ip2inst, sub2inst, sub2insts, nsub = B.relationship()
    excl, _ = B.revealed_monitor_institutions()
    out = {}
    for level, parent_of in (("sub", sub2inst), ("ip", ip2inst)):
        df = B.hourly(level)
        full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
        rows = []
        for c in df.columns:
            inst = parent_of[int(c)]
            if level == "sub":
                if any(x in excl for x in sub2insts[int(c)]):
                    continue
            elif inst in excl:
                continue
            r = D.profile("cesnet", c, df[c].reindex(full))
            r["institution"] = inst
            rows.append(r)
        d = pd.DataFrame(rows)
        d["drift"] = d.tv_shift >= 0.3
        d["rule_registered"] = (d.nan_share <= 0.05) & (d.zero_share <= 0.05) & d.median_pos & (d.test_windows_h6 >= 690)
        d["M7a"] = (d.nan_train <= 0.20) & (d.longest_gap_train_h <= 168) & (d.zero_share <= 0.05) & d.median_pos & (d.test_windows_h6 >= 690) & (d.test_window_breaks <= 1)
        res = {}
        for rule in ("rule_registered", "M7a"):
            g = d[d[rule] & d.drift].sort_values("id")
            res[rule] = {"drifting": int(len(g)), "institutions": int(g.institution.nunique())}
            for cap in (3, 4, 5, 10**6):
                cnt, kept = {}, 0
                for r in g.itertuples():
                    if cnt.get(r.institution, 0) < cap:
                        cnt[r.institution] = cnt.get(r.institution, 0) + 1
                        kept += 1
                res[rule][f"cap_{cap if cap < 10**6 else 'none'}"] = kept
        out[level] = res
    out["note"] = "tv_shift of data7_inventory.profile, not build_pool4.shift; excluded parents are those of rounds 3 to 6 (build_pool7.revealed_monitor_institutions); counts are information only"
    (HERE / "alt7_counts.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
