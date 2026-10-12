"""Change-point placement variant of the MON analysis for Paper A v2.

The main analysis (mon_analysis.py) places the three change points at 0.25,
0.5 and 0.75 of the range that starts after the warm-up prefix.  This variant
places them at 0.2, 0.45 and 0.7 of the post-warm-up stream, from the first
decision plus 20 to 110 steps before the end, with every other rule of
mon_analysis.py unchanged (thresholds designed on the no-fault validation
stream, EMA smoother, label delay h, same injection, same detectors).
It backs the sentence of the monitor section that the NAB AWS-CPU control
result depends on where the change points fall.

Output: ../results_v2/mon_placement_variant.json
Run: PYTHONDONTWRITEBYTECODE=1 nice -n 19 python3 mon_placement_variant.py
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
from _common import DATASETS, DEC, SEEDS, align, load_bolt, load_pred, load_val_forecast  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results_v2"
H = 6
DELAY = H
SMK = "ema"
POSITIONS = [0.2, 0.45, 0.7]
INJ_SEEDS = 10
DETS = mf.DETECTORS
NAMES = {"abs": "D1", "cusum": "D2", "comp": "D3", "rescusum": "D4"}


def main():
    run_rows, ctl_rows, nf_rows, feas = [], [], [], {}
    for ds in DATASETS:
        b = load_bolt(ds, H)
        feas[ds] = []
        for s in SEEDS:
            d = load_pred(ds, DEC, H, s)
            vf = load_val_forecast(ds, H, s)
            y, fd, ff = align(d[0].mean(1), d[1].mean(1), b[1].mean(1))
            n = len(y)
            sigma = float(np.std(ff - y)) + 1e-9
            t_arr, a0, r0 = mf.alpha_and_rmse_stream(y, fd, ff, DELAY)
            yv, dv, gv = align(vf["y"].mean(1), vf["dec"].mean(1), vf["glob"].mean(1))
            _, av, rv = mf.alpha_and_rmse_stream(yv, dv, gv, DELAY)
            p = mf.design_params(av, mf.smooth(av, SMK), rv)
            feas[ds].append(bool(p["h_feasible"]))
            lo_cp, hi_cp = int(t_arr[0]) + 20, n - 110
            eval_from = int(t_arr[0])
            sm0 = mf.smooth(a0, SMK)
            c0 = mf.conditions(a0, sm0, r0, p, start=0)
            for det in DETS:
                nf_rows.append({"dataset": ds, "seed": s, "detector": det, "fa_rate": float(mf.alarm_onsets(c0[det]).mean())})
            for pi, pos in enumerate(POSITIONS):
                cp = int(round(lo_cp + pos * (hi_cp - lo_cp)))
                t_info = cp + DELAY + 1
                for det in DETS:
                    fa, pia, lag = mf.eval_fault(c0[det], t_arr, cp, t_info, eval_from)
                    ctl_rows.append({"dataset": ds, "seed": s, "detector": det, "position": pos, "detected": lag is not None, "lag": lag})
                for r in range(INJ_SEEDS):
                    rng = np.random.default_rng(1_000_003 * (pi + 1) + 7919 * r + s)
                    ffs = mf.inject(ff, y, cp, rng, sigma)
                    _, a1, r1 = mf.alpha_and_rmse_stream(y, fd, ffs, DELAY)
                    c1 = mf.conditions(a1, mf.smooth(a1, SMK), r1, p, start=0)
                    for det in DETS:
                        fa, pia, lag = mf.eval_fault(c1[det], t_arr, cp, t_info, eval_from)
                        run_rows.append({"dataset": ds, "seed": s, "detector": det, "position": pos, "inj_seed": r, "pre_info_alarm": pia, "detected": lag is not None, "lag": lag})
        print("done", ds, flush=True)

    runs, ctl, nf = map(pd.DataFrame, (run_rows, ctl_rows, nf_rows))
    key = ["dataset", "seed", "detector", "position"]
    m = runs.merge(ctl[key + ["detected", "lag"]].rename(columns={"detected": "c_det", "lag": "c_lag"}), on=key)
    m = m[m.detected]
    m["both"] = m.c_det & (m.c_lag <= m.lag)
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for det in DETS:
            r = runs[(runs.dataset == ds) & (runs.detector == det)]
            c = ctl[(ctl.dataset == ds) & (ctl.detector == det)]
            f = nf[(nf.dataset == ds) & (nf.detector == det)]
            mm = m[(m.dataset == ds) & (m.detector == det)]
            out[ds][NAMES[det]] = {
                "fa": round(float(f.fa_rate.mean()), 3),
                "fa_min": round(float(f.fa_rate.min()), 3),
                "fa_max": round(float(f.fa_rate.max()), 3),
                "det": round(float(r.detected.mean()), 3),
                "lag": float(np.nanmedian(r.lag.astype(float))) if r.detected.any() else None,
                "pre_info": round(float(r.pre_info_alarm.mean()), 3),
                "ctrl": round(float(c.detected.mean()), 3),
                "ctrl_lag": float(np.nanmedian(c.lag.astype(float))) if c.detected.any() else None,
                "share_det_also_ctrl": round(float(mm.both.sum() / max(1, len(mm))), 3),
            }
        out[ds]["D2_feasible_share"] = float(np.mean(feas[ds]))
    (OUT / "mon_placement_variant.json").write_text(json.dumps({"positions": POSITIONS, "cp_range": "t_arr[0]+20 .. n-110", "horizon": H, "results": out}, indent=1, default=float))
    print(json.dumps(out["nab_aws_cpu"]))


if __name__ == "__main__":
    main()
