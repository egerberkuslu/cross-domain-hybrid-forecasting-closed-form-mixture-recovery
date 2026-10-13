"""Round 7 data inventory (DATA7.md section b): the never-allocated drifting leftovers of CESNET, Abilene and GEANT under candidate missing-data rules.

Reads only the cached raw hourly matrices (fresh_cells/cache/*.parquet), reserved_series.json (all blocks including round6_M), partition_F.json and
round6/pool6_manifest.json. No forecast, no model, no q stream. Values are read for TRAIN and VALIDATION hours only (shift screen, gap profile of train);
for the test split only the POSITIONS of missing hours are counted (stream length and contiguity), never a value.
Writes data7_leftovers.csv and data7_leftovers_summary.json.
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python data7_inventory.py
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

LOOK, H, MAXGAP_SHIPPED = F.LOOKBACK, 6, 24
MIN_TEST_WIN = {"cesnet": 690, "abilene": 690, "geant": 420}
R6_MISSING_MAX = {"cesnet": 0.05, "abilene": 0.16, "geant": 0.07}


def runs(mask):
    """lengths of consecutive True runs"""
    m = np.asarray(mask, bool)
    if not m.any():
        return np.array([], int)
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def profile(src, sid, s_raw):
    s = F._flag_outliers(s_raw)
    n = len(s)
    ntr, nva = int(n * F.FRACS[0]), int(n * F.FRACS[1])
    isna = s.isna().to_numpy()
    gl = runs(isna)
    # shipped filling: gaps <= 24 h interpolated (time), longer gaps stay missing and their windows are dropped
    grp = np.cumsum(~isna)
    gsize = pd.Series(isna).groupby(grp).transform("sum").to_numpy()
    long_na = isna & (gsize > MAXGAP_SHIPPED)
    filled = s.interpolate(method="time", limit_direction="both")
    tr, va = filled.iloc[:ntr], filled.iloc[ntr : ntr + nva]
    shift = (
        float(abs(va.mean() - tr.mean()) / tr.std()) if tr.std() > 0 else float("nan")
    )
    # complete h = 6 test windows: target hours inside test, lookback 168 + horizon 6 free of long-gap hours (positions only)
    bad = np.concatenate([[0], np.cumsum(long_na)])
    starts = np.arange(ntr + nva - LOOK, n - LOOK - H + 1)
    ok = (bad[starts + LOOK + H] - bad[starts]) == 0
    test_win = int(ok.sum())
    breaks_test = int(np.sum(np.diff(ok.astype(int)) == -1))
    sl = lambda a, b: isna[a:b]
    return {
        "source": src,
        "id": str(sid),
        "nan_share": float(isna.mean()),
        "nan_train": float(sl(0, ntr).mean()),
        "nan_val": float(sl(ntr, ntr + nva).mean()),
        "nan_test": float(sl(ntr + nva, n).mean()),
        "longest_gap_h": int(gl.max()) if gl.size else 0,
        "longest_gap_train_h": int(runs(sl(0, ntr)).max())
        if runs(sl(0, ntr)).size
        else 0,
        "n_gaps_gt24": int((gl > MAXGAP_SHIPPED).sum()),
        "interp_share_train": float((isna[:ntr] & ~long_na[:ntr]).mean()),
        "zero_share": float((s == 0).mean()),
        "median_pos": bool(float(s.median()) > 0),
        "tv_shift": shift,
        "test_windows_h6": test_win,
        "test_window_breaks": breaks_test,
    }


def main():
    R = json.loads((FC / "reserved_series.json").read_text())
    P = json.loads((FC / "partition_F.json").read_text())
    used = {}
    for src in ("cesnet", "abilene", "geant"):
        u = (
            set(map(str, R["reserved"][src]))
            | set(map(str, P["pilot"][src]))
            | set(map(str, P["F_registered"][src]))
        )
        u |= set(map(str, R["round4_A"]["A_half"][src])) | set(
            map(str, R["round5_FG"]["FG_half"][src])
        )
        for k in ("P2_new", "P2_FG", "P3"):
            blk = R["round6_M"].get(k, {})
            if isinstance(blk, dict):
                u |= set(map(str, blk.get(src, [])))
            elif isinstance(blk, list):
                pre = F.series_name(src, "")
                u |= {
                    x[len(pre) :]
                    for x in blk
                    if isinstance(x, str) and x.startswith(pre)
                }
        used[src] = u
    m6 = json.loads((HERE.parent / "round6" / "pool6_manifest.json").read_text())
    for e in m6["series"]:
        used[e["source"]].add(str(e["id"]))
    rows = []
    for src in ("cesnet", "abilene", "geant"):
        df = F.load_raw_hourly(src)
        full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
        adm = F.admissibility_table(src).set_index("id")
        for c in df.columns:
            if str(c) in used[src]:
                continue
            r = profile(src, c, df[c].reindex(full))
            r["admissible_original"] = bool(adm.loc[c, "admissible"])
            rows.append(r)
        print(src, "leftovers", sum(1 for r in rows if r["source"] == src), flush=True)
    d = pd.DataFrame(rows)
    d["drifting"] = d.tv_shift >= 0.3
    d["len_ok"] = d.apply(lambda r: r.test_windows_h6 >= MIN_TEST_WIN[r.source], axis=1)
    d["r6_rule"] = (
        d.apply(lambda r: r.nan_share <= R6_MISSING_MAX[r.source], axis=1)
        & (d.zero_share <= 0.05)
        & d.median_pos
    )
    # candidate round 7 missing-data rules (decided on train missingness and on positions only)
    d["M7a"] = (
        (d.nan_train <= 0.20)
        & (d.longest_gap_train_h <= 168)
        & (d.zero_share <= 0.05)
        & d.median_pos
        & d.len_ok
        & (d.test_window_breaks <= 1)
    )
    d["M7b"] = (
        (d.nan_train <= 0.30)
        & (d.zero_share <= 0.05)
        & d.median_pos
        & d.len_ok
        & (d.test_window_breaks <= 3)
    )
    d["M7c"] = (d.zero_share <= 0.05) & d.median_pos & d.len_ok
    d.to_csv(HERE / "data7_leftovers.csv", index=False)
    summ = {}
    for src, g in d.groupby("source"):
        gd = g[g.drifting]
        summ[src] = {
            "leftovers": int(len(g)),
            "drifting_ge0.3": int(len(gd)),
            "drifting_len_ok": int(gd.len_ok.sum()),
            "drifting_r6_rule": int(gd.r6_rule.sum()),
            "drifting_M7a": int(gd.M7a.sum()),
            "drifting_M7b": int(gd.M7b.sum()),
            "drifting_M7c": int(gd.M7c.sum()),
            "drifting_M7a_ids": gd[gd.M7a].id.tolist(),
            "drifting_M7b_ids": gd[gd.M7b].id.tolist(),
            "drifting_nan_share_quantiles": [
                float(x) for x in gd.nan_share.quantile([0, 0.25, 0.5, 0.75, 1]).values
            ]
            if len(gd)
            else [],
            "drifting_nan_train_quantiles": [
                float(x) for x in gd.nan_train.quantile([0, 0.25, 0.5, 0.75, 1]).values
            ]
            if len(gd)
            else [],
            "drifting_longest_gap_quantiles": [
                float(x)
                for x in gd.longest_gap_h.quantile([0, 0.25, 0.5, 0.75, 1]).values
            ]
            if len(gd)
            else [],
            "drifting_test_windows_quantiles": [
                float(x)
                for x in gd.test_windows_h6.quantile([0, 0.25, 0.5, 0.75, 1]).values
            ]
            if len(gd)
            else [],
            "drifting_zero_share_gt5pct": int((gd.zero_share > 0.05).sum()),
        }
    (HERE / "data7_leftovers_summary.json").write_text(json.dumps(summ, indent=1))
    print(
        json.dumps(
            {
                k: {kk: vv for kk, vv in v.items() if not kk.endswith("_ids")}
                for k, v in summ.items()
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
