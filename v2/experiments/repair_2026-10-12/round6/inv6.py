"""Round 6 data inventory: which CESNET / Abilene / GEANT series were never used by any monitor or model test, and which of them drift.

Reads only the cached raw hourly matrices (fresh_cells/cache/*.parquet), reserved_series.json, partition_F.json and the pool manifests.
The drift screen uses TRAIN and VALIDATION hours only (first 85 % of the hourly index, fractions 0.70 / 0.15 as fresh_cells.FRACS); no test hour is read
for the screen except to count the length of the series. It is an approximation of build_pool4.shift (which uses the preprocessed split boundaries);
its agreement with the manifest value is checked on the built series and reported.
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python inv6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FC = HERE.parent / "fresh_cells"
sys.path.insert(0, str(FC))
import fresh_cells as F  # noqa: E402

LOOK, H = F.LOOKBACK, 6


def approx_shift(s: pd.Series):
    s = F._flag_outliers(s).interpolate(method="time", limit_direction="both")
    n = len(s)
    ntr, nva = int(n * F.FRACS[0]), int(n * F.FRACS[1])
    tr, va = s.iloc[:ntr], s.iloc[ntr : ntr + nva]
    return float(abs(va.mean() - tr.mean()) / tr.std()), n - ntr - nva


def main():
    R = json.loads((FC / "reserved_series.json").read_text())
    P = json.loads((FC / "partition_F.json").read_text())
    m4 = json.loads((FC / "pool4_manifest.json").read_text())
    m5 = json.loads((FC / "pool5_manifest.json").read_text())
    known = {e["name"]: e["train_val_shift_sd"] for e in m5["series"]}
    known.update(
        {
            e["name"]: e.get("train_val_shift_sd")
            for e in m4["series"]
            if e.get("train_val_shift_sd") is not None
        }
    )
    out, check = {}, []
    for src in ("cesnet", "abilene", "geant"):
        t = F.admissibility_table(src)
        ids = [str(x) for x in t.id]
        adm = set(str(x) for x in t[t.admissible].id)
        used = {
            "round3_reserved": set(R["reserved"][src]),
            "pilot": set(P["pilot"][src]),
            "F_registered (scored by F+G arms; 64 drifting CESNET/Abilene = round 5 monitor)": set(
                P["F_registered"][src]
            ),
            "round4/5 A_half (accuracy and round 4 monitor)": set(
                R["round4_A"]["A_half"][src]
            ),
            "round5_FG half (Paper B replication, Paper A must not use)": set(
                R["round5_FG"]["FG_half"][src]
            ),
        }
        alloc = set().union(*used.values())
        df = F.load_raw_hourly(src)
        full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
        rows = []
        for c in df.columns:
            sid = str(c)
            sh, ntest_h = approx_shift(df[c].reindex(full))
            rows.append(
                {
                    "id": sid,
                    "admissible": sid in adm,
                    "allocated": sid in alloc,
                    "tv_shift": sh,
                    "test_hours": ntest_h,
                    "test_windows_h6_est": ntest_h - LOOK - H + 1 + LOOK,
                }
            )  # windows start inside the test hours (lookback reaches back into val)
            name = F.series_name(src, c)
            if name in known and known[name] is not None:
                check.append({"name": name, "manifest": known[name], "approx": sh})
        d = pd.DataFrame(rows)
        un = d[~d.allocated]
        out[src] = {
            "raw_series": len(d),
            "admissible": int(d.admissible.sum()),
            "allocated_by_block": {k: len(v) for k, v in used.items()},
            "allocated_total": len(alloc),
            "unallocated_total": int(len(un)),
            "unallocated_admissible": int(un.admissible.sum()),
            "unallocated_inadmissible": int((~un.admissible).sum()),
            "unallocated_inadmissible_drifting_ge0.3": int(
                ((~un.admissible) & (un.tv_shift >= 0.3)).sum()
            ),
            "test_hours": int(d.test_hours.iloc[0]),
            "FG_half_drifting_ge0.3": int(
                d[
                    d.id.isin(
                        used[
                            "round5_FG half (Paper B replication, Paper A must not use)"
                        ]
                    )
                ]
                .tv_shift.ge(0.3)
                .sum()
            ),
            "F_registered_drifting_ge0.3_approx": int(
                d[
                    d.id.isin(
                        used[
                            "F_registered (scored by F+G arms; 64 drifting CESNET/Abilene = round 5 monitor)"
                        ]
                    )
                ]
                .tv_shift.ge(0.3)
                .sum()
            ),
            "inadmissible_reasons": {
                k: int((~t[k]).sum()) for k in ("ok_missing", "ok_zero", "ok_median")
            },
        }
        un.to_csv(HERE / f"inv6_unallocated_{src}.csv", index=False)
    c = pd.DataFrame(check)
    if len(c):
        c["agree_0.3"] = (c.manifest >= 0.3) == (c.approx >= 0.3)
        out["screen_check_vs_manifest"] = {
            "n": int(len(c)),
            "agree_on_0.3_rule": int(c["agree_0.3"].sum()),
            "max_abs_diff": float((c.manifest - c.approx).abs().max()),
        }
    (HERE / "inv6_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
