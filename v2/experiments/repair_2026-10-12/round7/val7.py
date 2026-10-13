"""Round 7 design input (REPAIR_PREREG_7_DRAFT.md section 5): candidate repair variants of the c2 failure, selected on validation and scored on REVEALED
test streams whose results are public (round 6: 20 P2 traces, r6_monitor_summary.json; round 5: 64 stratum-F traces, r5_monitor_summary.json).
Never a round 7 trace, never a re-score of rounds 5 or 6 (no success rule is evaluated; rows are labelled revealed_r5_test / revealed_r6_test).

Arms (all re-armed, burn-in held, onset-localised events, the round 6 corrected per-trace conditions c1, c2, c3' with L = 150 and MARGIN = 0.5):
  V3    the round 6 V3 (r6_lib.select_rf, unchanged), for reference
  W1    SRC: segment-residual calibration. The validation q is split by binary segmentation (pcfit: at most 4 change points, segments >= 120,
             effect |dm| / pooled within-sd >= 1.0); the level, K and G are calibrated on the residual q - fit (block bootstrap of the residual), the
             floor is 0.5 sd(residual), the EW variance is Huber-clipped at 3 sd, revised fallback (no feasible level -> smallest level meeting the
             validation-stream null and FA conditions, flagged infeasible). On test the RF statistic runs on raw q, warm-started as V3.
  W2    SRC-TC: W1 as stage 1 at a relaxed null target (0.30), then a confirmation: an event at k is confirmed at k + C (C = 30) iff
             mean(q[k+1 .. k+C]) - m0 >= c * s0, m0 and s0 the mean and robust sd (floored) of q over [k - G - 60, k - G); c on the grid 0.5..6
             (step 0.25) is the smallest value meeting the three null conditions on the residual rows. The confirmed event is the alarm.
  W3    RS-LMB (bootstrap-only repair, for attribution): floor 0.5 * 1.4826 * MAD(q_val), Huber EW variance, level calibrated on raw q with the
             level-matched block bootstrap (each block shifted so its first 20 values match the last 20 of the previous one), revised fallback.
  W4    RS-LMB-RF (added after the first val7 run, disclosed): W3 with the floor of W1 (0.5 sd of the validation residual q - pcfit(q)) instead
             of the MAD floor; level calibrated on raw q with the level-matched bootstrap, Huber EW variance, revised fallback.
             Run with ARMS7=W4 (rows appended to val7_rows.csv as a separate file val7_w4_rows.csv).
  STB7  fair baseline: static CUSUM, reference = mean of the last 200 validation decisions (the warm segment of W1 to W3), scale = sd of the
             validation residual, level and K calibrated on the residual exactly as W1 (same grid, same null rule, same fallback).
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python val7.py
Writes val7_rows.csv, val7_summary.json. Memory < 1 GB per worker (3 workers).
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R6 = HERE.parent / "round6"
R5 = HERE.parent / "round5"
sys.path.insert(0, str(R6))
import r6_lib as L6  # noqa: E402

A2, M3, mf, T = L6.A2, L6.M3, L6.mf, L6.T
import os
ARMS = tuple(os.environ.get("ARMS7", "V3,W1,W2,W3,STB7").split(","))
OUT_TAG = os.environ.get("OUT7", "")
C_CONF, NULL_STAGE1 = 30, 0.30
C_GRID = np.arange(0.5, 6.01, 0.25)
seed_r6 = lambda pi, r_: A2.test_seed(pi, r_, 6)
seed_r5 = lambda pi, r_: A2.test_seed(pi, r_, 0)


# ------------------------------------------------------------------ statistics
def rf_z_rs(q, G, floor, q_warm=None, huber=3.0):
    qq = q if q_warm is None or len(q_warm) == 0 else np.concatenate([q_warm, q])
    off = len(qq) - len(q)
    n = len(qq)
    z = np.zeros(n)
    m = float(np.mean(qq[:20]))
    v = float(np.var(qq[:20])) + 1e-12
    for t in range(n):
        if t - G >= 0:
            x = qq[t - G]
            m = M3.DECAY * m + (1 - M3.DECAY) * x
            d = x - m
            v = M3.DECAY * v + (1 - M3.DECAY) * min(d * d, huber * huber * v)
        z[t] = (qq[t] - m) / max(np.sqrt(v), floor)
    return z[off:]


def robust_floor(q):
    return max(0.5 * 1.4826 * float(np.median(np.abs(q - np.median(q)))), 0.05)


def lmb_rows(x, G, B=L6.B_BOOT, seed=51, k=20):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    n = len(x)
    blk = int(min(2 * (G + 60), max(60, n // 2)))
    nb = int(np.ceil(n / blk))
    st = rng.integers(0, max(1, n - blk + 1), size=(B, nb))
    out = np.empty((B, nb * blk))
    for b in range(B):
        cur = x[st[b, 0] : st[b, 0] + blk].copy()
        parts = [cur]
        for j in range(1, nb):
            nxt = x[st[b, j] : st[b, j] + blk]
            nxt = nxt + (parts[-1][-k:].mean() - nxt[:k].mean())
            parts.append(nxt)
        out[b] = np.concatenate(parts)
    return out[:, :n]


def calibrate(Zrows, K, t, target=L6.NULL_TARGET):
    """Round 6 null-calibrated level with the revised fallback. Returns dict (level, feasible, val_null)."""
    ev, fa = L6.cusum_events(Zrows, K, L6.LEVELS, True, True)
    nr = L6.null_rate(ev, t)
    ok_self = nr[0] <= target
    ok = (
        ok_self
        & (np.quantile(nr[1:], 0.9, axis=0) <= target)
        & (np.quantile(fa[1:], 0.9, axis=0) <= L6.FA_TARGET)
    )
    if ok.any():
        i = int(np.argmax(ok))
        return {
            "level": float(L6.LEVELS[i]),
            "feasible": True,
            "val_null": float(nr[0, i]),
        }
    ok2 = ok_self & (fa[0] <= L6.FA_TARGET)
    if ok2.any():
        i = int(np.argmax(ok2))
        return {
            "level": float(L6.LEVELS[i]),
            "feasible": False,
            "val_null": float(nr[0, i]),
            "fallback": "self",
        }
    return {
        "level": float(L6.LEVELS[-1]),
        "feasible": False,
        "val_null": float(nr[0, -1]),
        "fallback": "max",
    }


def confirm(ev1, q, G, floor, c):
    """Two-stage confirmation. ev1 (n,) stage-1 events; returns confirmed events (n,)."""
    n = len(q)
    out = np.zeros(n, bool)
    busy = -1
    for k in np.flatnonzero(ev1):
        if k <= busy or k + C_CONF >= n:
            continue
        a, b = max(0, k - G - 60), max(1, k - G)
        ref = q[a:b] if b - a >= 10 else q[max(0, k - 70) : max(1, k - 10)]
        m0 = float(np.mean(ref))
        s0 = max(1.4826 * float(np.median(np.abs(ref - np.median(ref)))), floor)
        if np.mean(q[k + 1 : k + C_CONF + 1]) - m0 >= c * s0:
            out[k + C_CONF] = True
        busy = k + C_CONF
    return out


def events(arm, P, q, q_warm):
    if arm == "V3":
        return L6.arm_events("V3", P, q, None, q_warm)
    if arm == "STB7":
        z = (q - P["mu"]) / P["sd"]
        ev, fa = L6.cusum_events(z[None], P["K"], [P["level"]], True, True)
        return ev[0, 0], float(fa[0, 0])
    z = rf_z_rs(q, P["G"], P["floor"], q_warm)
    ev, fa = L6.cusum_events(z[None], P["K"], [P["level"]], True, True)
    if arm in ("W1", "W3", "W4"):
        return ev[0, 0], float(fa[0, 0])
    evc = confirm(ev[0, 0], q, P["G"], P["floor"], P["c"])
    return evc, float(evc.sum() / len(q))


# ------------------------------------------------------------------ selection on validation
def pcfit(q, min_seg=120, max_cp=4, thr=1.0):
    """Piecewise-constant fit of q by binary segmentation: split where |mean difference| / pooled within-sd is largest, accept iff >= thr,
    segments >= min_seg, at most max_cp change points. Fixed before any round 7 data."""
    segs = [(0, len(q))]
    cps = 0
    while cps < max_cp:
        best = None
        for si, (a, b) in enumerate(segs):
            x = q[a:b]
            if len(x) < 2 * min_seg:
                continue
            cs, cs2 = np.cumsum(x), np.cumsum(x * x)
            for k in range(min_seg, len(x) - min_seg + 1, 5):
                n1, n2 = k, len(x) - k
                m1, m2 = cs[k - 1] / n1, (cs[-1] - cs[k - 1]) / n2
                v1 = cs2[k - 1] / n1 - m1 * m1
                v2 = (cs2[-1] - cs2[k - 1]) / n2 - m2 * m2
                eff = abs(m2 - m1) / np.sqrt(max((n1 * v1 + n2 * v2) / len(x), 1e-12))
                if best is None or eff > best[0]:
                    best = (eff, si, a + k)
        if best is None or best[0] < thr:
            break
        _, si, k = best
        a, b = segs.pop(si)
        segs += [(a, k), (k, b)]
        cps += 1
    fit = np.empty(len(q))
    for a, b in segs:
        fit[a:b] = q[a:b].mean()
    return fit, cps


def det_val(arm, P, vruns, warm):
    return sum(int(L6.detect(events(arm, P, x["q"], warm)[0], x["t"], x["cp"])) for x in vruns) / len(vruns)


def calib_tc(rows, Z, G, K, floor, t):
    c1 = calibrate(Z, K, t, target=NULL_STAGE1)
    ev1, _ = L6.cusum_events(Z, K, [c1["level"]], True, True)
    for cc in C_GRID:
        evc = np.stack([confirm(ev1[i, 0], rows[i], G, floor, cc) for i in range(len(rows))])
        nr = L6.null_rate(evc, t)
        fa = evc.mean(1)
        if nr[0] <= L6.NULL_TARGET and np.quantile(nr[1:], 0.9) <= L6.NULL_TARGET and np.quantile(fa[1:], 0.9) <= L6.FA_TARGET:
            return {"G": G, "K": K, "floor": floor, "level": c1["level"], "c": float(cc), "feasible": bool(c1["feasible"]), "val_null": float(nr[0])}
    return {"G": G, "K": K, "floor": floor, "level": c1["level"], "c": float(C_GRID[-1]), "feasible": False, "val_null": float("nan")}


def select(arm, q, t, vruns):
    warm = q[: L6.WARM]
    if arm == "V3":
        return L6.select_rf("V3", q, t, None, vruns)
    fit, ncp = pcfit(q)
    res = q - fit
    if arm == "STB7":
        mu, sd = float(q[-L6.WARM :].mean()), max(float(res.std()), 1e-3)
        z = res / sd
        rows = np.vstack([z[None], L6.boot_rows(z, 60)])
        best = None
        for K in (0.5, 1.0, 2.0):
            c = calibrate(rows, K, t)
            P = {"mu": mu, "sd": sd, "K": K, "n_cp_val": ncp, **c}
            det = det_val("STB7", {**P, "mu": float(q.mean())}, vruns, None)  # validation injections are scored around the validation mean
            key = (c["feasible"], det, -c["val_null"])
            if best is None or key > best[0]:
                best = (key, {**P, "val_det": det})
        return best[1]
    best = None
    for G, K in L6.RF_GRID:
        if arm == "W3":
            floor = robust_floor(q)
            rows = np.vstack([q[None], lmb_rows(q, G)])
        elif arm == "W4":
            floor = max(0.5 * float(res.std()), 0.05)
            rows = np.vstack([q[None], lmb_rows(q, G)])
        else:
            floor = max(0.5 * float(res.std()), 0.05)
            rows = np.vstack([res[None], L6.boot_rows(res, G)])
        Z = np.stack([rf_z_rs(r, G, floor, r[: L6.WARM]) for r in rows])
        if arm == "W2":
            P = calib_tc(rows, Z, G, K, floor, t)
        else:
            P = {"G": G, "K": K, "floor": floor, **calibrate(Z, K, t)}
        P["n_cp_val"] = ncp
        det = det_val(arm, P, vruns, warm)
        vn = P["val_null"] if np.isfinite(P["val_null"]) else 1.0
        key = (P["feasible"], det, -vn, G)
        if best is None or key > best[0]:
            best = (key, {**P, "val_det": det})
    return best[1]


# ------------------------------------------------------------------ scoring (round 6 corrected criterion, copied from r6_lib.evaluate_trace)
def score(arm, P, y, fE, fB, seed_fn, q_warm, rc_seed, inj=A2.INJ_TEST, cp_lo=420):
    t, a, r, q, e = A2.streams(y, fE, fB)
    n = len(y)
    ev0, fa = events(arm, P, q, q_warm)
    cps = L6.injection_cps(t, n, cp_lo=cp_lo)
    ctl = [bool(L6.detect(ev0, t, cp)) for cp in cps]
    nul = float(L6.null_rate(ev0, t))
    m = int((t >= t[0] + L6.BURN).sum())
    n_ev = int(ev0[L6.BURN :].sum())
    p_rand = L6.random_rate(n_ev, m, L6.L_CAP - A2.DELAY)
    sigma = float(np.std(fB - y)) + 1e-9
    det = []
    for pi, cp in enumerate(cps):
        for r_ in range(inj):
            ffs = mf.inject(fB, y, cp, np.random.default_rng(seed_fn(pi, r_)), sigma)
            t1, _, _, q1, _ = A2.streams(y, fE, ffs)
            det.append(bool(L6.detect(events(arm, P, q1, q_warm)[0], t1, cp)))
    x1, n1, x2, n2 = int(sum(det)), len(det), int(sum(ctl)), len(ctl)
    ss_lo = T.newcombe_diff(x1, n1, x2, n2)[1]
    p0 = max(p_rand, nul)
    rr = np.random.default_rng(rc_seed)
    rc = []
    for _ in range(200):
        evr = np.zeros(len(q), bool)
        if n_ev:
            evr[L6.BURN + rr.choice(m, size=min(n_ev, m), replace=False)] = True
        xr = sum(bool(L6.detect(evr, t, cp)) for cp in cps) * inj
        rc.append(xr / n1 - p0 >= L6.MARGIN)
    return {
        "fa": fa,
        "c1": fa <= L6.FA_TARGET,
        "det": x1 / n1,
        "ctl": x2 / n2,
        "c2": ss_lo > 0,
        "null_dense": nul,
        "p_rand": p_rand,
        "p0": p0,
        "c3": (x1 / n1 - p0 >= L6.MARGIN) and (ss_lo > 0),
        "joint": (fa <= L6.FA_TARGET) and (ss_lo > 0) and (x1 / n1 - p0 >= L6.MARGIN),
        "n_events_nofault": n_ev,
        "rand_ctrl_c3only_share": float(np.mean(rc)),
        "det_by_cp": json.dumps(
            [float(np.mean(det[i * inj : (i + 1) * inj])) for i in range(3)]
        ),
    }


def one(job):
    name, split, vdir, tdir, idx = job
    y, fE, fB = M3.streams_from(M3.load_h6(Path(vdir), name))
    t, a, r, q, e = A2.streams(y, fE, fB)
    vruns = A2.inject_streams(y, fE, fB, A2.POSITIONS_VAL, A2.INJ_VAL, A2.val_seed, 0)
    yt, fEt, fBt = M3.streams_from(
        M3.load_h6(Path(tdir), name)
    )  # REVEALED test stream (public result), design input only
    nt = len(yt)
    cp_lo = 420 if nt >= 690 else 200
    out = []
    for arm in ARMS:
        P = select(arm, q, t, vruns)
        qw = q[-L6.WARM :] if arm != "STB7" else None
        s = score(
            arm,
            P,
            yt,
            fEt,
            fBt,
            seed_r6 if split == "revealed_r6_test" else seed_r5,
            qw,
            20261100 + idx,
            cp_lo=cp_lo,
        )
        out.append(
            {
                "series": name,
                "split": split,
                "source": name.split("_")[0],
                "arm": arm,
                "feasible": bool(P.get("feasible", True)),
                "params": json.dumps(
                    {
                        k: (
                            float(v)
                            if isinstance(v, (int, float, np.floating, np.integer))
                            and not isinstance(v, bool)
                            else v
                        )
                        for k, v in P.items()
                    }
                ),
                **s,
            }
        )
    print(name, split, {o["arm"]: int(o["joint"]) for o in out}, flush=True)
    return out


def main():
    m6 = json.loads((R6 / "pool6_manifest.json").read_text())
    m5 = json.loads((HERE.parent / "fresh_cells" / "pool5_manifest.json").read_text())
    jobs = [
        (n, "revealed_r6_test", str(R6 / "r6_val"), str(R6 / "r6_test"), i)
        for i, n in enumerate(m6["monitor_traces_P2"])
    ]
    jobs += [
        (n, "revealed_r5_test", str(R5 / "r5_val"), str(R5 / "r5_test"), 100 + i)
        for i, n in enumerate(m5["monitor_traces"])
    ]
    with Pool(3) as pl:
        res = pl.map(one, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(HERE / f"val7{OUT_TAG}_rows.csv", index=False)
    summ = {
        "note": "design input on revealed test streams (public results); not evidence; rounds 5 and 6 are not re-scored"
    }
    for split, g in df.groupby("split"):
        summ[split] = {}
        for arm, ga in g.groupby("arm"):
            summ[split][arm] = {
                "n": int(len(ga)),
                "feasible": int(ga.feasible.sum()),
                "c1": int(ga.c1.sum()),
                "c2": int(ga.c2.sum()),
                "c3": int(ga.c3.sum()),
                "joint": int(ga.joint.sum()),
                "rand_ctrl_c3_share_mean": float(ga.rand_ctrl_c3only_share.mean()),
                "mean_det": float(ga.det.mean()),
                "mean_null_dense": float(ga.null_dense.mean()),
                "by_source_joint": {
                    s: [int(gs.joint.sum()), int(len(gs))]
                    for s, gs in ga.groupby("source")
                },
                "failed": ga[~ga.joint].series.tolist(),
            }
    (HERE / f"val7{OUT_TAG}_summary.json").write_text(json.dumps(summ, indent=1))
    print(
        json.dumps(
            {
                s: {
                    a: {k: v for k, v in d.items() if k != "failed"}
                    for a, d in x.items()
                }
                for s, x in summ.items()
                if s != "note"
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
