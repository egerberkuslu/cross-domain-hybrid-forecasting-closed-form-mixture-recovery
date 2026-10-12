"""MON analysis for Paper A v2 (rule frozen in V2_PLAN.md, 'Monitor threshold rule').

Thresholds are designed on the no-fault validation stream only (target 5 % false
alarms), the smoother is a causal EMA with lambda = 9/11, and the label delay is h.
On the test stream we report, per trace, the no-fault false-alarm rate, the
detection rate and median lag under injection, the pre-information alarm share,
and the no-injection control (same change points, same detection rule, clean
stream).  Helper functions are imported from the fix-wave monitor_far.py so the
stream construction is identical to the recompute.

Outputs in ../results_v2: mon_runs.csv, mon_control.csv, mon_nofault.csv,
mon_thresholds.csv, mon_cells.csv, mon_summary.json, mon_example_streams.csv
Run: PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 mon_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIX = Path(__file__).resolve().parents[2] / "experiments" / "fix_2026-10-11"
sys.path.insert(0, str(FIX))
import monitor_far as mf  # noqa: E402
from _common import BOLT, DATASETS, DEC, SEEDS, align, load_bolt, load_pred, load_val_forecast  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results_v2"
H = 6
DELAY = H
SMK = "ema"
POSITIONS = [0.25, 0.5, 0.75]
INJ_SEEDS = 10
PREFIX = 0.3
DETS = mf.DETECTORS  # abs, cusum, comp, rescusum


def main():
    thr_rows, nf_rows, run_rows, ctl_rows, ex_rows = [], [], [], [], []
    for ds in DATASETS:
        b = load_bolt(ds, H)
        for s in SEEDS:
            d = load_pred(ds, DEC, H, s)
            vf = load_val_forecast(ds, H, s)
            assert b is not None and d is not None and vf is not None, (ds, s)
            y, fd, ff = align(d[0].mean(1), d[1].mean(1), b[1].mean(1))
            n = len(y)
            sigma = float(np.std(ff - y)) + 1e-9
            t_arr, a0, r0 = mf.alpha_and_rmse_stream(y, fd, ff, DELAY)
            yv, dv, gv = align(vf["y"].mean(1), vf["dec"].mean(1), vf["glob"].mean(1))
            _, av, rv = mf.alpha_and_rmse_stream(yv, dv, gv, DELAY)
            smv = mf.smooth(av, SMK)
            p = mf.design_params(av, smv, rv)
            # validation false-alarm rate actually reached by the designed thresholds
            cv = mf.conditions(av, smv, rv, p, start=0)
            val_fa = {det: float(mf.alarm_onsets(cv[det]).mean()) for det in DETS}
            thr_rows.append({"dataset": ds, "seed": s, "n_val_decisions": int(av.size), **{k: v for k, v in p.items()}, **{f"val_fa_{k}": v for k, v in val_fa.items()}})
            # change points, identical to monitor_far (design independent)
            k0 = int(PREFIX * a0.size)
            first = int(t_arr[min(max(mf.WARMUP, k0), t_arr.size - 1)])
            lo_cp, hi_cp = first + 20 + DELAY, n - 40
            eval_from = int(t_arr[0])  # start = 0, designs use no test data
            sm0 = mf.smooth(a0, SMK)
            c0 = mf.conditions(a0, sm0, r0, p, start=0)
            for det in DETS:
                hit = c0[det]
                nf_rows.append({"dataset": ds, "seed": s, "detector": det, "n_eval": int(hit.size), "fa_rate": float(mf.alarm_onsets(hit).mean()), "any_alarm": bool(mf.alarm_onsets(hit).any())})
            for pi, pos in enumerate(POSITIONS):
                cp = int(round(lo_cp + pos * (hi_cp - lo_cp)))
                t_info = cp + DELAY + 1
                # no-injection control: same cp, same rule, clean stream
                for det in DETS:
                    fa, pia, lag = mf.eval_fault(c0[det], t_arr, cp, t_info, eval_from)
                    ctl_rows.append({"dataset": ds, "seed": s, "detector": det, "position": pos, "cp": cp, "t_info": t_info, "pre_fa": fa, "pre_info_alarm": pia, "detected": lag is not None, "lag": lag})
                for r in range(INJ_SEEDS):
                    rng = np.random.default_rng(1_000_003 * (pi + 1) + 7919 * r + s)
                    ffs = mf.inject(ff, y, cp, rng, sigma)
                    _, a1, r1 = mf.alpha_and_rmse_stream(y, fd, ffs, DELAY)
                    sm1 = mf.smooth(a1, SMK)
                    c1 = mf.conditions(a1, sm1, r1, p, start=0)
                    for det in DETS:
                        fa, pia, lag = mf.eval_fault(c1[det], t_arr, cp, t_info, eval_from)
                        run_rows.append({"dataset": ds, "seed": s, "detector": det, "position": pos, "cp": cp, "t_info": t_info, "inj_seed": r, "pre_fa": fa, "pre_info_alarm": pia, "detected": lag is not None, "lag": lag})
                    if s == 42 and pos == 0.5 and r == 0:
                        for j, t in enumerate(t_arr):
                            ex_rows.append({"dataset": ds, "t": int(t), "alpha_smoothed_nofault": float(sm0[j]), "alpha_smoothed_fault": float(sm1[j]), "d3_hit_nofault": bool(c0["comp"][j]), "d3_hit_fault": bool(c1["comp"][j]), "cp": cp, "t_info": t_info, "tau": p["tau"]})
        print("done", ds, flush=True)

    thr, nf, runs, ctl, ex = map(pd.DataFrame, (thr_rows, nf_rows, run_rows, ctl_rows, ex_rows))
    for name, df in (("mon_thresholds", thr), ("mon_nofault", nf), ("mon_runs", runs), ("mon_control", ctl), ("mon_example_streams", ex)):
        df.to_csv(OUT / f"{name}.csv", index=False)

    def agg_runs(df, tag):
        g = df.groupby(["dataset", "detector"])
        return g.agg(**{f"det_rate_{tag}": ("detected", "mean"), f"lag_median_{tag}": ("lag", "median"), f"lag_p25_{tag}": ("lag", lambda x: np.nanpercentile(x.dropna(), 25) if x.notna().any() else np.nan), f"lag_p75_{tag}": ("lag", lambda x: np.nanpercentile(x.dropna(), 75) if x.notna().any() else np.nan), f"pre_info_{tag}": ("pre_info_alarm", "mean"), f"n_{tag}": ("detected", "size")}).reset_index()

    cells = nf.groupby(["dataset", "detector"]).agg(nofault_fa_mean=("fa_rate", "mean"), nofault_fa_min=("fa_rate", "min"), nofault_fa_max=("fa_rate", "max"), nofault_any_alarm_share=("any_alarm", "mean")).reset_index()
    cells = cells.merge(agg_runs(runs, "inj"), on=["dataset", "detector"]).merge(agg_runs(ctl, "ctl"), on=["dataset", "detector"])
    # share of injected detections whose control run (same seed, same cp) also alarms at or before the injected lag
    key = ["dataset", "seed", "detector", "position"]
    c1 = ctl[key + ["detected", "lag"]].rename(columns={"detected": "c_det", "lag": "c_lag"})
    m = runs.merge(c1, on=key)
    m = m[m.detected]
    m["ctrl_alarm_by_then"] = m.c_det & (m.c_lag <= m.lag)
    sh = m.groupby(["dataset", "detector"]).ctrl_alarm_by_then.mean().rename("share_inj_detections_also_in_control").reset_index()
    cells = cells.merge(sh, on=["dataset", "detector"], how="left")
    cells.to_csv(OUT / "mon_cells.csv", index=False)
    t3 = thr.groupby("dataset").agg(h_feasible_share=("h_feasible", "mean"), h_mean=("h", "mean"), tau_mean=("tau", "mean"), val_fa_d3_mean=("val_fa_comp", "mean"), n_val=("n_val_decisions", "first"))
    summary = {"horizon": H, "delay": DELAY, "smoother": SMK, "ema_lambda": 9 / 11, "traces_d3_nofault_le_5pct": int((cells[cells.detector == "comp"].nofault_fa_mean <= 0.05).sum()), "thresholds": t3.reset_index().to_dict(orient="records")}
    (OUT / "mon_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(cells[cells.detector == "comp"].round(3).T)
    print(t3.round(3))


if __name__ == "__main__":
    main()
