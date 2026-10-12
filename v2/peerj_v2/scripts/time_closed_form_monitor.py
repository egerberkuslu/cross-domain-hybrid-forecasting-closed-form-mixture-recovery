"""Timing of the closed-form weight fit and of one monitor step (V2_VERIFY fix K4).

Closed-form fit: three mixture evaluations on the validation forecasts of one
cell (M(0), M(1/2), M(1)) plus the Bates-Granger solution and the clip.  For
comparison the 21-point grid search on the same forecasts is timed.
Monitor step: one decision of detector D3 on the horizon-averaged stream, i.e.
the rolling clipped Bates-Granger weight over W = 60 labelled windows, one EMA
update (lambda = 9/11), one CUSUM update, the absolute gate and the
persistence counter.
Setting: single pinned core, single-threaded math libraries, nice 19.
Run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 taskset -c 0 python3 time_closed_form_monitor.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

FIX = Path(__file__).resolve().parents[2] / "experiments" / "fix_2026-10-11"
sys.path.insert(0, str(FIX))
from _common import BOLT, DEC, align, load_bolt, load_pred, load_val_forecast, bg_from_three  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results_v2"
REPS = 200
W, LAM, K = 60, 9 / 11, 0.5


def med_us(fn, reps=REPS, inner=1):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        ts.append((time.perf_counter_ns() - t0) / inner / 1e3)
    ts = np.array(ts)
    return float(np.median(ts)), float(np.percentile(ts, 5)), float(np.percentile(ts, 95))


def cpu_model():
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main():
    res = {"cpu": cpu_model(), "reps": REPS, "numpy": np.__version__, "python": platform.python_version()}
    cells = {}
    for ds, h, s in [("cesnet", 6, 42), ("abilene", 6, 42), ("geant", 6, 42)]:
        pass
    from _common import DATASETS
    for ds in DATASETS:
        vf = load_val_forecast(ds, 6, 42)
        if vf is None:
            continue
        y, fd, fg = vf["y"], vf["dec"], vf["glob"]

        def mse(a):
            return float(np.mean((a * fd + (1 - a) * fg - y) ** 2))

        def closed():
            m0, mh, m1 = mse(0.0), mse(0.5), mse(1.0)
            return bg_from_three(m0, mh, m1)[1]

        grid = np.linspace(0, 1, 21)

        def gridsearch():
            return grid[int(np.argmin([mse(a) for a in grid]))]

        c = med_us(closed)
        g = med_us(gridsearch)
        cells[ds] = {"n_val_windows": int(y.shape[0]), "closed_form_us": c, "grid21_us": g}
    res["fit"] = cells
    res["fit_median_over_traces_us"] = float(np.median([v["closed_form_us"][0] for v in cells.values()]))
    res["grid_median_over_traces_us"] = float(np.median([v["grid21_us"][0] for v in cells.values()]))

    # monitor step on a test stream
    mon = {}
    for ds in DATASETS:
        b = load_bolt(ds, 6)
        d = load_pred(ds, DEC, 6, 42)
        if b is None or d is None:
            continue
        y, fd, ff = align(d[0].mean(1), d[1].mean(1), b[1].mean(1))
        ed, ef = fd - y, ff - y
        diff = ed - ef
        delay = 6
        t = min(len(y) - 1, W + delay + 50)
        state = {"s": 0.3, "S": 0.0, "run": 0}
        mu, sd, hthr, tau = 0.05, 0.05, 5.0, 0.05

        def step():
            sl = slice(t - delay - W, t - delay)
            dd = diff[sl]
            den = float(np.dot(dd, dd))
            a = 0.0 if den < 1e-12 else float(np.clip(-np.dot(ef[sl], dd) / den, 0, 1))
            s = LAM * state["s"] + (1 - LAM) * a
            S = max(0.0, state["S"] + ((a - mu) / sd - K))
            hit = (s >= tau) and (S > hthr)
            run = state["run"] + 1 if hit else 0
            return a, s, S, run >= 3

        mon[ds] = {"monitor_step_us": med_us(step, inner=20)}
    res["monitor"] = mon
    res["monitor_median_over_traces_us"] = float(np.median([v["monitor_step_us"][0] for v in mon.values()]))
    (OUT / "timing_closed_form_monitor.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
