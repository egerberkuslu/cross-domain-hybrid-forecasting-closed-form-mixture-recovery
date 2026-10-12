"""audit item 3: alpha-monitor with validation-designed thresholds, no-fault
false-alarm rates, several injection seeds and positions, detection counted
only after the change point has entered the estimator.

What was wrong in pipeline/expR6_monitor_master.py
  * thresholds (tau = 0.20, k = 0.5, h = 5) and the CUSUM baseline (mean/sd
    of the first 40 alpha values) were set on the SAME test streams;
  * one decomposition seed (42) and one injection seed per dataset;
  * the change point sits at the stream midpoint; at decision t = cp the
    rolling window [t-60, t) holds only pre-shift windows, so an alarm at
    lag 0 cannot be a detection. With NO fault injected, D3 already alarms at
    "lag" 0 on CESNET and Abilene (+69 GEANT, +20 NAB Twitter);
  * the smoother is np.convolve(..., mode="same"): a CENTRED, non-causal
    10-window moving average that reads up to 5 future alpha values, while
    Algorithm 3 in an earlier draft describes an exponential smoother;
  * the rolling window uses windows up to t-1, whose h-step targets are not
    all observed at t (label delay ignored).

Lag-0 artefact exclusion used here. The estimate at decision t uses windows
[t - delay - W, t - delay). It can contain window cp only if
t >= t_info = cp + delay + 1. Alarms at cp <= t < t_info are counted as
"pre-information alarms" (false alarms), never as detections; the
persistence counter is reset at t_info, so a detection needs PERSIST
consecutive hits all at t >= t_info. delay = h (strict: only fully labelled
windows) is the recommended setting; delay = 0 is kept for comparison.

Designs
  legacy      : constants above, baseline from the first 40 test alphas,
                centred smoother, delay 0 (reproduces the published table).
  val         : baseline (mu, sd), absolute threshold tau and CUSUM
                thresholds designed on the no-fault VALIDATION stream of the
                same decomposition seed (needs step 0): tau = max(q_{1-target}
                of smoothed val alpha, TAU_FLOOR), CUSUM h = smallest value on
                a grid whose val alarm rate <= target (same for D4 on the
                rolling RMSE). sd floors avoid degenerate z-scores.
  test_prefix : fallback when validation streams are missing: the first 30 %
                of the no-fault test stream is the design set and is
                excluded from every evaluation.
Evaluation per (dataset, decomposition seed): no-fault FA rate on the clean
test stream; then faults injected at 3 positions x R injection seeds
(same ramp as expR6: +3 sd ramp over 100 windows with 50 % multiplicative
noise), detection rate, lag, pre-shift FA, pre-information alarms.

Outputs: monitor_legacy_check.csv, monitor_thresholds.csv,
         monitor_nofault.csv, monitor_injection_runs.csv,
         monitor_far_cells.csv, monitor_far_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BOLT,
    DATASETS,
    DEC,
    EVAL,
    SEEDS,
    align,
    load_bolt,
    load_pred,
    load_val_forecast,
    npz_path,
    write_csv,
    write_json,
)

W, WARMUP, SMOOTH = 60, 40, 10
TAU_ABS, K, H_CUSUM, PERSIST = 0.20, 0.5, 5.0, 3
TARGET = 0.05
TAU_FLOOR = 0.05  # one alpha-grid step
SD_FLOOR_ALPHA = 0.01
H_GRID = np.arange(1.0, 30.01, 0.5)
DETECTORS = ("abs", "cusum", "comp", "rescusum")


# ------------------------------------------------------------ streams
def horizon_avg(arr):
    return arr.mean(axis=1)


def alpha_and_rmse_stream(y, fd, ff, delay):
    """Rolling clipped BG alpha and foundation RMSE at decisions t."""
    ed, ef = fd - y, ff - y
    diff = ed - ef
    n = len(y)
    t_arr = np.arange(W + delay, n)
    a = np.empty(t_arr.size)
    r = np.empty(t_arr.size)
    for j, t in enumerate(t_arr):
        sl = slice(t - delay - W, t - delay)
        den = float(np.dot(diff[sl], diff[sl]))
        a[j] = (
            0.0
            if den < 1e-12
            else float(np.clip(-np.dot(ef[sl], diff[sl]) / den, 0, 1))
        )
        r[j] = float(np.sqrt(np.mean((ff[sl] - y[sl]) ** 2)))
    return t_arr, a, r


def smooth(a, kind):
    if kind == "centred":  # legacy, non-causal
        return np.convolve(a, np.ones(SMOOTH) / SMOOTH, mode="same")
    if kind == "trailing":
        c = np.concatenate([[0.0], np.cumsum(a)])
        k = np.arange(1, a.size + 1)
        lo = np.maximum(0, k - SMOOTH)
        return (c[k] - c[lo]) / (k - lo)
    if kind == "ema":
        lam = 1.0 - 2.0 / (SMOOTH + 1.0)
        out = np.empty_like(a)
        s = a[0]
        for i, x in enumerate(a):
            s = lam * s + (1 - lam) * x if i else x
            out[i] = s
        return out
    raise ValueError(kind)


def cusum_hits(x, mu, sd, h, start=0):
    hits = np.zeros(x.size, bool)
    S = 0.0
    for k in range(start, x.size):
        S = max(0.0, S + ((x[k] - mu) / sd - K))
        hits[k] = S > h
    return hits


def conditions(a, sm, r, p, start=0):
    abs_hit = np.zeros(a.size, bool)
    abs_hit[start:] = sm[start:] >= p["tau"]
    cus = cusum_hits(a, p["mu"], p["sd"], p["h"], start)
    res = cusum_hits(r, p["mu_r"], p["sd_r"], p["h_r"], start)
    return {"abs": abs_hit, "cusum": cus, "comp": abs_hit & cus, "rescusum": res}


def alarm_onsets(hit):
    """Legacy counting: alarm when PERSIST consecutive hits, then reset."""
    out = np.zeros(hit.size, bool)
    run = 0
    for k, x in enumerate(hit):
        run = run + 1 if x else 0
        if run >= PERSIST:
            out[k] = True
            run = 0
    return out


def eval_fault(hit, t_arr, cp, t_info, eval_from):
    """Pre-cp FA rate, pre-information alarm, detection lag (t >= t_info)."""
    m = t_arr >= eval_from
    hit, t = hit[m], t_arr[m]
    pre = t < cp
    on = alarm_onsets(hit[pre])
    fa = float(on.mean()) if pre.any() else np.nan
    mid = (t >= cp) & (t < t_info)
    pre_info_alarm = bool(alarm_onsets(hit[mid]).any()) if mid.any() else False
    post = t >= t_info
    lag = None
    run = 0
    for tk, x in zip(t[post], hit[post]):
        run = run + 1 if x else 0
        if run >= PERSIST:
            lag = int(tk - cp)
            break
    return fa, pre_info_alarm, lag


def legacy_eval(hit, idx, cp):
    """Exact copy of expR6 _eval_detector semantics."""
    run_len, fired_at, fa, pre = 0, None, 0, 0
    for k in range(len(idx)):
        run_len = run_len + 1 if hit[k] else 0
        alarm = run_len >= PERSIST
        if idx[k] < cp:
            if idx[k] >= 0:
                pre += 1
                if alarm:
                    fa += 1
                    run_len = 0
        elif alarm and fired_at is None:
            fired_at = int(idx[k] - cp)
    return fa / max(1, pre), fired_at


def inject(ff, y, cp, rng, sigma):
    out = ff.copy()
    for t in range(cp, len(ff)):
        ramp = min(1.0, (t - cp) / 100.0)
        out[t] = ff[t] + ramp * 3.0 * sigma * (1.0 + 0.5 * rng.standard_normal())
    return out


# ------------------------------------------------------------ designs
def legacy_params(a, r):
    return {
        "tau": TAU_ABS,
        "mu": float(a[:WARMUP].mean()),
        "sd": float(a[:WARMUP].std()) + 1e-6,
        "h": H_CUSUM,
        "mu_r": float(r[:WARMUP].mean()),
        "sd_r": float(r[:WARMUP].std()) + 1e-6,
        "h_r": H_CUSUM,
    }


def design_params(a, sm, r):
    """Thresholds from a no-fault design stream (validation or test prefix)."""
    p = {
        "mu": float(a.mean()),
        "sd": max(float(a.std()), SD_FLOOR_ALPHA),
        "mu_r": float(r.mean()),
        "sd_r": max(float(r.std()), 1e-3 * max(float(r.mean()), 1e-12)),
        "tau": max(float(np.quantile(sm, 1 - TARGET)) + 1e-9, TAU_FLOOR),
    }
    p["h"] = H_GRID[-1]
    p["h_feasible"] = False
    for hh in H_GRID:
        if alarm_onsets(cusum_hits(a, p["mu"], p["sd"], hh)).mean() <= TARGET:
            p["h"], p["h_feasible"] = float(hh), True
            break
    p["h_r"] = H_GRID[-1]
    p["h_r_feasible"] = False
    for hh in H_GRID:
        if alarm_onsets(cusum_hits(r, p["mu_r"], p["sd_r"], hh)).mean() <= TARGET:
            p["h_r"], p["h_r_feasible"] = float(hh), True
            break
    return p


# ------------------------------------------------------------ main
def legacy_check(h):
    rows = []
    ref_p = EVAL / "tables" / "alpha_monitor.json"
    ref = json.loads(ref_p.read_text()) if ref_p.exists() else {}
    for ds in DATASETS:
        pd_, pb_ = npz_path(ds, DEC, h, 42), npz_path(
            ds, "cha_hybrid_v3_global_only", h, 42
        )
        if not (pd_.exists() and pb_.exists()):
            continue
        # raw float32 arrays averaged in float32, exactly as expR6's load()
        d, b = np.load(pd_), np.load(pb_)
        y, fd, ff = align(
            d["y_true_scaled"].mean(axis=1),
            d["y_pred_scaled"].mean(axis=1),
            b["y_pred_scaled"].mean(axis=1),
        )
        n = len(y)
        cp = max(W + WARMUP + 20, n // 2)
        for fault in (True, False):
            rng = np.random.default_rng(20260529)
            sg = float(np.std(ff - y)) + 1e-9
            ffs = inject(ff, y, cp, rng, sg) if fault else ff.copy()
            idx, a, r = alpha_and_rmse_stream(y, fd, ffs, 0)
            sm = smooth(a, "centred")
            p = legacy_params(a, r)
            c = conditions(a, sm, r, p, start=WARMUP)
            mask = idx >= idx[WARMUP]
            row = {
                "dataset": ds,
                "fault_injected": fault,
                "cp": cp,
                "baseline_alpha": round(p["mu"], 3),
            }
            for det in DETECTORS:
                fa, lag = legacy_eval(c[det][mask], idx[mask], cp)
                row[f"{det}_fa"] = round(fa, 4)
                row[f"{det}_lag"] = lag
                if fault and ds in ref:
                    row[f"{det}_fa_ref"] = ref[ds].get(f"{det}_fa")
                    row[f"{det}_lag_ref"] = ref[ds].get(f"{det}_lag")
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--experts", nargs="*", default=[BOLT, "chronos2_zs", "tirex_zs"])
    ap.add_argument("--inj-seeds", type=int, default=10)
    ap.add_argument("--positions", nargs="*", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--smoothers", nargs="*", default=["centred", "trailing", "ema"])
    ap.add_argument("--delays", nargs="*", default=["0", "h"])
    ap.add_argument("--prefix", type=float, default=0.3)
    args = ap.parse_args()
    H = args.horizon

    leg = legacy_check(H)
    write_csv(leg, "monitor_legacy_check.csv")
    summary = {"horizon": H, "target": TARGET}
    lf = leg[leg.fault_injected]
    if "comp_lag_ref" in lf:
        summary["legacy_reproduced"] = bool(
            all(
                (
                    lf[f"{d}_lag"].astype(object).fillna(-1)
                    == lf[f"{d}_lag_ref"].astype(object).fillna(-1)
                ).all()
                and np.allclose(lf[f"{d}_fa"], lf[f"{d}_fa_ref"], atol=1e-4)
                for d in DETECTORS
            )
        )
    nf = leg[~leg.fault_injected]
    summary["legacy_no_fault_comp_lag"] = {
        r.dataset: r.comp_lag for r in nf.itertuples()
    }

    thr_rows, nf_rows, run_rows, example_rows = [], [], [], []
    for expert in args.experts:
        for ds in DATASETS:
            b = load_bolt(ds, H, expert=expert)
            if b is None:
                continue
            for s in SEEDS:
                d = load_pred(ds, DEC, H, s)
                if d is None:
                    continue
                y, fd, ff = align(
                    horizon_avg(d[0]), horizon_avg(d[1]), horizon_avg(b[1])
                )
                n = len(y)
                sigma = float(np.std(ff - y)) + 1e-9
                vf = load_val_forecast(ds, H, s) if expert == BOLT else None
                for dl in args.delays:
                    delay = H if dl == "h" else int(dl)
                    t_arr, a0, r0 = alpha_and_rmse_stream(y, fd, ff, delay)
                    if vf is not None:
                        yv, dv, gv = align(
                            horizon_avg(vf["y"]),
                            horizon_avg(vf["dec"]),
                            horizon_avg(vf["glob"]),
                        )
                        _, av, rv = alpha_and_rmse_stream(yv, dv, gv, delay)
                    # change points are design-independent: placed after the
                    # latest design region (legacy warm-up or test prefix), so
                    # every design is scored on the same faulted streams
                    k0 = int(args.prefix * a0.size)
                    first = int(t_arr[min(max(WARMUP, k0), t_arr.size - 1)])
                    lo_cp, hi_cp = first + 20 + delay, n - 40
                    faults = []
                    if hi_cp > lo_cp:
                        for pi, pos in enumerate(args.positions):
                            cp = int(round(lo_cp + pos * (hi_cp - lo_cp)))
                            for r in range(args.inj_seeds):
                                rng = np.random.default_rng(
                                    1_000_003 * (pi + 1) + 7919 * r + s
                                )
                                ffs = inject(ff, y, cp, rng, sigma)
                                _, a1, r1 = alpha_and_rmse_stream(y, fd, ffs, delay)
                                faults.append((pos, r, cp, cp + delay + 1, a1, r1))
                    for smk in args.smoothers:
                        designs = (
                            ["legacy"]
                            + (["val"] if vf is not None else [])
                            + ["test_prefix"]
                        )
                        for design in designs:
                            if design == "legacy":
                                p = legacy_params(a0, r0)
                                start = WARMUP
                            elif design == "val":
                                p = design_params(av, smooth(av, smk), rv)
                                start = 0
                            else:
                                p = design_params(
                                    a0[:k0], smooth(a0[:k0], smk), r0[:k0]
                                )
                                start = k0
                            eval_from = int(t_arr[min(start, t_arr.size - 1)])
                            key = {
                                "expert": expert,
                                "dataset": ds,
                                "seed": s,
                                "delay": dl,
                                "smoother": smk,
                                "design": design,
                            }
                            thr_rows.append({**key, **{k: v for k, v in p.items()}})
                            # ---- no-fault stream
                            c0 = conditions(a0, smooth(a0, smk), r0, p, start=start)
                            m = t_arr >= eval_from
                            for det in DETECTORS:
                                hit = c0[det][m]
                                nf_rows.append(
                                    {
                                        **key,
                                        "detector": det,
                                        "n_eval": int(m.sum()),
                                        "fa_rate": float(alarm_onsets(hit).mean()),
                                        "alarm_window_share": float(hit.mean()),
                                        "any_alarm": bool(alarm_onsets(hit).any()),
                                    }
                                )
                            # ---- one example stream for the headline figure
                            if (
                                expert == BOLT
                                and ds == "cesnet"
                                and s == 42
                                and dl == "h"
                                and smk == "ema"
                                and design in ("val", "test_prefix")
                                and faults
                            ):
                                ex = next(
                                    (f for f in faults if f[0] == 0.5 and f[1] == 0),
                                    faults[0],
                                )
                                sm0 = smooth(a0, smk)
                                sm1 = smooth(ex[4], smk)
                                c1 = conditions(ex[4], sm1, ex[5], p, start=start)
                                for j, t in enumerate(t_arr):
                                    example_rows.append(
                                        {
                                            "design": design,
                                            "t": int(t),
                                            "alpha_smoothed_nofault": float(sm0[j]),
                                            "alpha_smoothed_fault": float(sm1[j]),
                                            "d3_hit_nofault": bool(c0["comp"][j]),
                                            "d3_hit_fault": bool(c1["comp"][j]),
                                            "cp": ex[2],
                                            "t_info": ex[3],
                                            "tau": p["tau"],
                                            "eval_from": eval_from,
                                        }
                                    )
                            # ---- faults
                            for pos, r, cp, t_info, a1, r1 in faults:
                                c1 = conditions(a1, smooth(a1, smk), r1, p, start=start)
                                for det in DETECTORS:
                                    fa, pia, lag = eval_fault(
                                        c1[det], t_arr, cp, t_info, eval_from
                                    )
                                    run_rows.append(
                                        {
                                            **key,
                                            "detector": det,
                                            "position": pos,
                                            "cp": cp,
                                            "t_info": t_info,
                                            "inj_seed": r,
                                            "pre_cp_windows": int(
                                                (
                                                    (t_arr >= eval_from) & (t_arr < cp)
                                                ).sum()
                                            ),
                                            "pre_fa": fa,
                                            "pre_info_alarm": pia,
                                            "detected": lag is not None,
                                            "lag": lag,
                                        }
                                    )
            print(f"done {expert} {ds}")

    thr = pd.DataFrame(thr_rows)
    nfd = pd.DataFrame(nf_rows)
    runs = pd.DataFrame(run_rows)
    write_csv(thr, "monitor_thresholds.csv")
    write_csv(nfd, "monitor_nofault.csv")
    write_csv(runs, "monitor_injection_runs.csv")
    if example_rows:
        write_csv(pd.DataFrame(example_rows), "monitor_stream_example.csv")

    keys = ["expert", "design", "delay", "smoother", "dataset", "detector"]
    a = (
        nfd.groupby(keys)
        .agg(
            nofault_fa_mean=("fa_rate", "mean"),
            nofault_fa_max=("fa_rate", "max"),
            nofault_any_alarm_share=("any_alarm", "mean"),
        )
        .reset_index()
    )
    bcol = (
        runs.groupby(keys)
        .agg(
            detection_rate=("detected", "mean"),
            lag_median=("lag", "median"),
            lag_p25=(
                "lag",
                lambda x: np.nanpercentile(x.dropna(), 25)
                if x.notna().any()
                else np.nan,
            ),
            lag_p75=(
                "lag",
                lambda x: np.nanpercentile(x.dropna(), 75)
                if x.notna().any()
                else np.nan,
            ),
            pre_fa_mean=("pre_fa", "mean"),
            pre_info_alarm_share=("pre_info_alarm", "mean"),
            n_runs=("detected", "size"),
        )
        .reset_index()
    )
    cells = a.merge(bcol, on=keys, how="left")
    write_csv(cells, "monitor_far_cells.csv")

    for expert in cells.expert.unique():
        for design in ("val", "test_prefix", "legacy"):
            g = cells[
                (cells.expert == expert)
                & (cells.design == design)
                & (cells.delay == "h")
                & (cells.smoother == "ema")
                & (cells.detector == "comp")
            ]
            if g.empty:
                continue
            summary[f"{expert}/{design}/delay=h/ema/D3"] = {
                "per_dataset": g.set_index("dataset")[
                    [
                        "nofault_fa_mean",
                        "detection_rate",
                        "lag_median",
                        "pre_info_alarm_share",
                    ]
                ].to_dict(orient="index"),
                "datasets_nofault_fa_le_target": int(
                    (g.nofault_fa_mean <= TARGET).sum()
                ),
                "datasets_detection_rate_ge_0.9": int((g.detection_rate >= 0.9).sum()),
            }
            g4 = cells[
                (cells.expert == expert)
                & (cells.design == design)
                & (cells.delay == "h")
                & (cells.smoother == "ema")
                & (cells.detector == "rescusum")
            ]
            if not g4.empty:
                summary[f"{expert}/{design}/delay=h/ema/D4"] = g4.set_index("dataset")[
                    ["nofault_fa_mean", "detection_rate", "lag_median"]
                ].to_dict(orient="index")
    write_json(summary, "monitor_far_summary.json")


if __name__ == "__main__":
    main()
