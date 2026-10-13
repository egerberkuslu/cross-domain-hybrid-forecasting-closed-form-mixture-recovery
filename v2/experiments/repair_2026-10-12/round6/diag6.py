"""Round 6 diagnosis (no new test read): why condition 3 of the round 5 monitor failed.

Reads ONLY (a) the round 5 validation streams r5_val/*__h6.npz of the 64 stratum-F traces and (b) the round 5 TEST streams r5_test/*__h6.npz of the same 64
traces, which were already read and scored by r5_monitor.py --stage test (results revealed in r5_monitor_summary.json). Nothing here is used to choose a
round 6 parameter: every tuning number of REPAIR_PREREG_6_DRAFT.md is computed on validation streams by the round 6 select stage.
Frozen round 5 monitor parameters (frozen_r5_monitor.json, hash re-checked) are used unchanged; no detector is re-calibrated here.

Output: diag6_rows.csv (one row per trace x detector x split), diag6_summary.json.
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python diag6.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R5 = HERE.parent / "round5"
for p in (
    R5,
    HERE.parent / "round4",
    HERE.parent / "round3",
    HERE.parent / "paper_a",
    HERE.parent / "paper_a" / "round2",
    HERE.parent / "fresh_cells",
):
    sys.path.insert(0, str(p))
import rcommon as R  # noqa: E402
import r3_monitor as M3  # noqa: E402
import r4_monitor as M4  # noqa: E402

A2 = M4.A2
PERS = {"ST": 3, "AD": 1, "RF": 3}
OFF = A2.W + A2.DELAY  # decision k <-> window t = k + OFF


def run_start(hit, k):
    """Index where the run of consecutive hits containing decision k began."""
    j = k
    while j > 0 and hit[j - 1]:
        j -= 1
    return j


def trace_rows(name, P, split_dir, split):
    y, fE, fB = M3.streams_from(M3.load_h6(split_dir, name))
    t, a, r, q, e = A2.streams(y, fE, fB)
    n = len(y)
    hits = M3.all_hits(P, q, e)
    out = []
    for d in M4.DETS:
        h = hits[d]
        on = A2.onsets_k(h, PERS[d])
        plc = []
        pre_active = 0
        for pj in M4.PLACEBO:
            lag = A2.eval_run(h, t, pj, pj + A2.DELAY + 1, PERS[d])[2]
            det = lag is not None and lag <= M4.CAP_PLC
            plc.append(det)
            if det:
                k_alarm = int(np.where(t == pj + lag)[0][0])
                pre_active += int(
                    t[run_start(h, k_alarm)] < pj
                )  # the alarm run had started before the placebo clock
        # windows: post burn-in prefix [166, 417] vs late segment [n-110, n)
        burn_end = M3.BURN + OFF
        pre = (t >= burn_end) & (t < M4.CP_LO)
        late = t >= n - 110
        first_on = int(t[np.argmax(on)]) if on.any() else -1
        # natural drift of the clean prefix: q level shift between [66,190) and [190,417) in units of the validation-q sd
        qa, qb = q[t < M4.PLACEBO[0]], q[(t >= M4.PLACEBO[0]) & (t < M4.CP_LO)]
        sdq = float(np.std(q)) + 1e-12
        ya, yb = y[: M4.PLACEBO[0]], y[M4.PLACEBO[0] : M4.CP_LO]
        row = {
            "series": name,
            "split": split,
            "detector": d,
            "n": n,
            "placebo_rate": float(np.mean(plc)),
            "placebo_pre_active": pre_active,
            "hit_share_prefix": float(h[pre].mean()),
            "hit_share_late": float(h[late].mean()),
            "hit_share_all": float(h.mean()),
            "onsets_prefix_per100": float(on[pre].sum() / max(1, pre.sum()) * 100),
            "onsets_rest_per100": float(
                on[~pre & (t >= burn_end)].sum()
                / max(1, (~pre & (t >= burn_end)).sum())
                * 100
            ),
            "first_onset_window": first_on,
            "alarm_at_burn_end": bool(h[M3.BURN]) if len(h) > M3.BURN else False,
            "q_shift_prefix_sd": float((qb.mean() - qa.mean()) / sdq),
            "y_shift_prefix_sd": float(
                (yb.mean() - ya.mean()) / (np.std(y[: M4.CP_LO]) + 1e-12)
            ),
            "fa_onset_share": float(on.sum() / len(q)),
        }
        if d == "RF":
            z = M3.rf_z(q, P["RF"]["G"], P["RF"]["floor"])
            S_ = M4.T.cusum_S(z[None], P["RF"]["K"])[0]
            row.update(
                {
                    "S_at_burn_end": float(S_[M3.BURN - 1]),
                    "level": P["RF"]["level"],
                    "z_mean_prefix": float(z[pre].mean()),
                    "z_mean_rest": float(z[~pre & (t >= burn_end)].mean()),
                    "absz_gt3_prefix": float((np.abs(z[pre]) > 3).mean()),
                    "absz_gt3_rest": float(
                        (np.abs(z[~pre & (t >= burn_end)]) > 3).mean()
                    ),
                }
            )
        out.append(row)
    return out


def main():
    p = R5 / "frozen_r5_monitor.json"
    assert (
        R.sha256_file(p) == (R5 / "frozen_r5_monitor.sha256").read_text().strip()
    ), "frozen_r5_monitor.json changed"
    fr = json.loads(p.read_text())
    names = json.loads(
        (HERE.parent / "fresh_cells" / "pool5_manifest.json").read_text()
    )["monitor_traces"]
    rows = []
    for name in names:
        rows += trace_rows(name, fr["params"][name], R5 / "r5_val", "val")
        rows += trace_rows(name, fr["params"][name], R5 / "r5_test", "test_revealed")
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "diag6_rows.csv", index=False)
    # injection lags (already revealed rows of r5_monitor_runs.csv)
    runs = pd.read_csv(R5 / "r5_monitor_runs.csv")
    inj = runs[runs.kind == "inj"]
    lag_tab = {}
    for d in M4.DETS:
        g = inj[inj.detector == d]
        lg = g.lag.dropna()
        lag_tab[d] = {
            "n_runs": int(len(g)),
            "missed_all": int(g.lag.isna().sum()),
            "lag_le100": int((lg <= 100).sum()),
            "lag_100_150": int(((lg > 100) & (lg <= 150)).sum()),
            "lag_gt150": int((lg > 150).sum()),
            "lag_median": float(lg.median()),
            "lag_q90": float(lg.quantile(0.9)),
            "lag_by_pos_median": {
                int(k): float(v) for k, v in g.groupby("pos").lag.median().items()
            },
        }
    plc = runs[runs.kind == "plc"]
    plc_lag = {
        d: plc[(plc.detector == d) & plc.detected].lag.describe().to_dict()
        for d in M4.DETS
    }
    summ = {"lags_injected_revealed": lag_tab, "placebo_detected_lag_revealed": plc_lag}
    for split in ("val", "test_revealed"):
        for d in M4.DETS:
            g = df[(df.split == split) & (df.detector == d)]
            summ[f"{split}|{d}"] = {
                "traces": int(len(g)),
                "mean_placebo_rate": float(g.placebo_rate.mean()),
                "traces_placebo_ge_0.6": int((g.placebo_rate >= 0.6).sum()),
                "traces_placebo_0": int((g.placebo_rate == 0).sum()),
                "placebo_hits_total": int(round((g.placebo_rate * 5).sum())),
                "placebo_hits_from_alarm_started_before_clock": int(
                    g.placebo_pre_active.sum()
                ),
                "mean_hit_share_prefix": float(g.hit_share_prefix.mean()),
                "mean_hit_share_late": float(g.hit_share_late.mean()),
                "mean_onsets_prefix_per100": float(g.onsets_prefix_per100.mean()),
                "mean_onsets_rest_per100": float(g.onsets_rest_per100.mean()),
                "alarm_at_burn_end": int(g.alarm_at_burn_end.sum()),
                "fa_le_5pct": int((g.fa_onset_share <= 0.05).sum()),
            }
            if d == "RF":
                summ[f"{split}|{d}"].update(
                    {
                        "S_at_burn_end_gt_level": int(
                            (g.S_at_burn_end > g.level).sum()
                        ),
                        "mean_absz_gt3_prefix": float(g.absz_gt3_prefix.mean()),
                        "mean_absz_gt3_rest": float(g.absz_gt3_rest.mean()),
                        "corr_placebo_vs_abs_q_shift": float(
                            np.corrcoef(g.placebo_rate, g.q_shift_prefix_sd.abs())[0, 1]
                        ),
                        "corr_placebo_vs_abs_y_shift": float(
                            np.corrcoef(g.placebo_rate, g.y_shift_prefix_sd.abs())[0, 1]
                        ),
                    }
                )
    (HERE / "diag6_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
