"""Round 6 library (frozen by REPAIR_PREREG_6.md): placebo-calibrated, onset-localised variants of the reference-free monitor.

Nothing here reads a test file. Statistic q and e are the round 3 to 5 streams (a2.streams on horizon-averaged h = 6 forecasts of E-Q and Chronos-Bolt).

Arms
  R5     the frozen round 3/4/5 RF detector (level by rf_level, no re-arm), scored by the corrected criterion (report only)
  V1     RF statistic, level re-calibrated so that the validation null detection rate (dense clocks, onset-localised, cap L) <= NULL_TARGET
  V2     V1 + CUSUM held at 0 through the burn-in (no release transient) + re-arm (S = 0 after every alarm event)
  V3     V2 + exponentially weighted reference warm-started on the WARM decisions that precede the scored stream
  STB    static CUSUM on q (validation mean and sd), K on the grid {0.5, 1, 2} and level by the same rules as V1 to V3, re-armed (fair baseline)
  ADB    ADWIN on the standardised delayed Bolt squared error, delta by the same null rule (fair baseline candidate, residual detector)
  RND    random alarm events at the no-fault event rate of the arm it controls (random control)

Event = the decision at which a run of consecutive hits reaches the persistence p (3 for CUSUM arms, 1 for ADWIN); a run that started earlier and is
still on does not create a new event (onset localisation). Detection of a change at window cp = an event at window in [cp + DELAY + 1, cp + L].
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for p in (
    HERE.parent / "round4",
    HERE.parent / "round3",
    HERE.parent / "paper_a",
    HERE.parent / "paper_a" / "round2",
    HERE.parent / "fresh_cells",
):
    sys.path.insert(0, str(p))
import stats_tools as T  # noqa: E402
import a2 as A2  # noqa: E402
import monitor_far as mf  # noqa: E402
import r3_monitor as M3  # noqa: E402

L_CAP = 150  # one lag cap for injected runs, matched control and null clocks
NULL_TARGET = 0.10  # validation null detection rate (q90 over block-bootstrap resamples and the stream itself)
FA_TARGET = 0.05
BURN = M3.BURN
WARM = 200
CLOCK_STEP = 5
LEVELS = np.arange(1.0, 60.01, 0.5)  # round 3-5 grid (1 to 30) extended to 60 for every round 6 CUSUM arm
ADWIN_DELTAS = [1e-5, 5e-5, 1e-4] + list(A2.ADWIN_DELTAS)  # round 3-5 grid extended downwards for the ADWIN baseline
MARGIN = 0.5  # corrected condition 3: detection rate minus null rate at least MARGIN
RF_GRID = M3.RF_GRID
B_BOOT = 60
OFF = A2.W + A2.DELAY
PLACEBO_CLOCKS = [190 + 30 * j for j in range(5)]


# ---------------------------------------------------------------- statistic
def rf_z_warm(q, G, floor, q_warm=None):
    if q_warm is None or len(q_warm) == 0:
        return M3.rf_z(q, G, floor)
    return M3.rf_z(np.concatenate([q_warm, q]), G, floor)[len(q_warm) :]


def cusum_events(Z, K, levels, rearm, burn_hold, persist=3, burn=BURN):
    """Z (B, n) standardised statistic. Returns events (B, L, n) bool and the round 3-5 FA onset share (B, L) (onsets_k on the hit stream)."""
    Z = np.atleast_2d(Z)
    B, n = Z.shape
    lv = np.asarray(levels, float)[None, :]
    S = np.zeros((B, lv.shape[1]))
    run = np.zeros(S.shape, int)
    runfa = np.zeros(S.shape, int)
    fa = np.zeros(S.shape)
    ev = np.zeros((B, lv.shape[1], n), bool)
    for k in range(n):
        if burn_hold and k < burn:
            S[:] = 0.0
        else:
            S = np.maximum(0.0, S + Z[:, k][:, None] - K)
        hit = S > lv
        if k < burn:
            hit[:] = False
        run = (run + 1) * hit
        e = run == persist
        ev[:, :, k] = e
        runfa = (runfa + 1) * hit
        on = runfa >= persist
        fa += on
        runfa[on] = 0
        if rearm:
            S[e] = 0.0
    return ev, fa / n


def events_from_hits(hit, persist):
    run, out = 0, np.zeros(hit.size, bool)
    for i, x in enumerate(hit):
        run = run + 1 if x else 0
        out[i] = run == persist
    return out


# ---------------------------------------------------------------- scoring
def detect(ev, t, cp, L=L_CAP):
    """ev (..., n) events aligned with window array t. True where an event falls in [cp + DELAY + 1, cp + L]."""
    lo, hi = np.searchsorted(t, cp + A2.DELAY + 1), np.searchsorted(
        t, cp + L, side="right"
    )
    if hi <= lo:
        return np.zeros(ev.shape[:-1], bool)
    return ev[..., lo:hi].any(-1)


SHORT_MIN = 30  # short-stream rule: minimum scorable decisions in the last (right-truncated) window


def dense_clocks(t, L=L_CAP, start=None):
    """Clocks every CLOCK_STEP windows from the end of the burn-in to n - L. Short-stream rule (GEANT validation, 281 windows, fewer than burn-in + L decisions):
    when fewer than 4 clocks fit, the clocks run to the last window minus SHORT_MIN and their lag windows are right-truncated at the end of the stream (optimistic null rate)."""
    start = int(t[0]) + BURN if start is None else start
    stop = int(t[-1]) - L + 1
    if stop < start + 3 * CLOCK_STEP:
        stop = int(t[-1]) - SHORT_MIN + 1
    return np.arange(start, stop, CLOCK_STEP)


def null_rate(ev, t, L=L_CAP, start=None):
    """Share of dense clocks of a NO-FAULT stream detected (onset-localised). ev (..., n)."""
    cl = dense_clocks(t, L, start)
    if cl.size == 0:
        return np.full(ev.shape[:-1], np.nan)
    cs = np.concatenate([np.zeros(ev.shape[:-1] + (1,)), np.cumsum(ev, -1)], -1)
    lo = np.searchsorted(t, cl + A2.DELAY + 1)
    hi = np.searchsorted(t, cl + L, side="right")
    return ((cs[..., hi] - cs[..., lo]) > 0).mean(-1)


def random_rate(n_events, m_decisions, w):
    """P(at least one of n_events random event decisions, placed uniformly without replacement among m_decisions, falls in a given window of w decisions)."""
    if n_events <= 0:
        return 0.0
    if n_events > m_decisions - w:
        return 1.0
    return 1.0 - comb(m_decisions - w, n_events) / comb(m_decisions, n_events)


CP_LO = {"long": 420, "short": 200}  # long: CESNET / Abilene test streams (>= 690 windows); short: GEANT test streams (428 windows)


def injection_cps(t, n, L=L_CAP, cp_lo=420):
    """Round 6 change points: cp_lo (420; 200 for the short GEANT design), the midpoint and n - L - 10, so that the whole lag window of every change point is inside the stream."""
    lo, hi = cp_lo, n - L - 10
    return [int(round(lo + f * (hi - lo))) for f in (0.0, 0.5, 1.0)]


# ---------------------------------------------------------------- calibration on validation
def boot_rows(x, G, B=B_BOOT, seed=51):
    rng = np.random.default_rng(seed)
    blk = int(min(2 * (G + 60), max(60, len(x) // 2)))
    return T.mbb_resample(x, blk, B, rng)


def calibrate_cusum(
    Zrows, K, t, rearm, burn_hold, null_target=NULL_TARGET, fa_target=FA_TARGET
):
    """Zrows[0] = the validation stream itself, Zrows[1:] = bootstrap resamples. Smallest feasible level, or None."""
    ev, fa = cusum_events(Zrows, K, LEVELS, rearm, burn_hold)
    nr = null_rate(ev, t)  # (B, L)
    ok = (
        (np.quantile(nr[1:], 0.90, axis=0) <= null_target)
        & (nr[0] <= null_target)
        & (np.quantile(fa[1:], 0.90, axis=0) <= fa_target)
    )
    idx = np.where(ok)[0]
    if idx.size == 0:
        return None
    i = idx[0]
    return {
        "level": float(LEVELS[i]),
        "val_null": float(nr[0, i]),
        "val_null_q90": float(np.quantile(nr[1:, i], 0.9)),
        "val_fa_q90": float(np.quantile(fa[1:, i], 0.9)),
    }


def arm_events(arm, P, q, e, q_warm=None):
    """Event array of one arm on one stream with frozen parameters P (dict)."""
    if arm == "R5":
        h = M3.rf_hits(q, P["G"], P["K"], P["level"], P["floor"])
        return events_from_hits(h, 3), float(A2.onsets_k(h, 3).sum() / len(q))
    if arm in ("V1", "V2", "V3"):
        z = rf_z_warm(q, P["G"], P["floor"], q_warm if arm == "V3" else None)
        ev, fa = cusum_events(
            z[None], P["K"], [P["level"]], rearm=arm != "V1", burn_hold=arm != "V1"
        )
        return ev[0, 0], float(fa[0, 0])
    if arm == "STB":
        z = (q - P["mu"]) / P["sd"]
        ev, fa = cusum_events(
            z[None], P.get("K", T.K_CUSUM), [P["level"]], rearm=True, burn_hold=True
        )
        return ev[0, 0], float(fa[0, 0])
    if arm == "ADB":
        al = A2.adwin_alarms((e - P["mu"]) / P["sd"], P["delta"])
        al[:BURN] = False
        return al, float(al.sum() / len(e))
    raise ValueError(arm)


def select_rf(arm, q, t, e, vruns, q_warm_val=None):
    """Validation selection of (G, K, level) for V1 / V2 / V3. vruns: validation injections (a2.inject_streams, positions 0.3 / 0.6)."""
    floor = max(0.5 * float(np.std(q)), 0.05)
    best = None
    for G, K in RF_GRID:
        rows = np.vstack([q[None], boot_rows(q, G)])
        if arm == "V3":
            Z = np.stack([rf_z_warm(r, G, floor, r[:WARM]) for r in rows])  # validation: warm start on the first WARM decisions (in-sample)
            tt = t
        else:
            Z = np.stack([M3.rf_z(r, G, floor) for r in rows])
            tt = t
        c = calibrate_cusum(Z, K, tt, rearm=arm != "V1", burn_hold=arm != "V1")
        if c is None:
            continue
        P = {"G": G, "K": K, "floor": floor, **c}
        det = 0
        for x in vruns:
            if arm == "V3":
                ev, _ = arm_events(arm, P, x["q"], None, q[:WARM])  # warm segment = clean validation prefix
                det += int(detect(ev, x["t"], x["cp"]))
            else:
                ev, _ = arm_events(arm, P, x["q"], None)
                det += int(detect(ev, x["t"], x["cp"]))
        key = (det / max(1, len(vruns)), -c["val_null"], G)
        if best is None or key > best[0]:
            best = (key, {**P, "val_det": key[0], "feasible": True})
    if best is None:
        return {
            "G": 60,
            "K": 2.0,
            "level": float(LEVELS[-1]),
            "floor": floor,
            "feasible": False,
            "val_det": float("nan"),
        }
    return best[1]


def select_stb(q, t, vruns=None):
    """Fair baseline: static CUSUM on q (validation mean and sd). With vruns (the round 6 default) K is chosen on the same grid {0.5, 1, 2} and by the same
    rule as V1 to V3 (validation detection, then lower validation null rate, then the first K), with the same fallback (K = 2, grid maximum, infeasible).
    vruns=None keeps the earlier fixed K = 0.5 construction (used only to reproduce val6.py)."""
    mu, sd = float(q.mean()), max(float(q.std()), 1e-3)
    z = (q - mu) / sd
    rows = np.vstack([z[None], boot_rows(z, 60)])
    if vruns is None:
        c = calibrate_cusum(rows, T.K_CUSUM, t, rearm=True, burn_hold=True)
        if c is None:
            return {"mu": mu, "sd": sd, "level": float(LEVELS[-1]), "feasible": False}
        return {"mu": mu, "sd": sd, **c, "feasible": True}
    best = None
    for K in sorted({k for _, k in RF_GRID}):
        c = calibrate_cusum(rows, K, t, rearm=True, burn_hold=True)
        if c is None:
            continue
        P = {"mu": mu, "sd": sd, "K": K, **c}
        det = sum(int(detect(arm_events("STB", P, x["q"], None)[0], x["t"], x["cp"])) for x in vruns)
        key = (det / max(1, len(vruns)), -c["val_null"])
        if best is None or key > best[0]:
            best = (key, {**P, "val_det": key[0], "feasible": True})
    if best is None:
        return {"mu": mu, "sd": sd, "K": 2.0, "level": float(LEVELS[-1]), "feasible": False, "val_det": float("nan")}
    return best[1]


def select_adb(e, t, B=40):
    me, se = float(e.mean()), max(float(e.std()), 1e-12)
    z = (e - me) / se
    rows = np.vstack([z[None], T.mbb_resample(z, 60, B, np.random.default_rng(53))])
    chosen = None
    for dl in ADWIN_DELTAS:  # increasing delta = more sensitive; keep the most sensitive feasible
        ev = np.stack([A2.adwin_alarms(r, dl) for r in rows])
        ev[:, :BURN] = False
        nr = null_rate(ev, t)
        fa = ev.mean(1)
        if (
            np.quantile(nr[1:], 0.9) <= NULL_TARGET
            and nr[0] <= NULL_TARGET
            and np.quantile(fa[1:], 0.9) <= FA_TARGET
        ):
            chosen = {
                "delta": dl,
                "val_null": float(nr[0]),
                "val_null_q90": float(np.quantile(nr[1:], 0.9)),
            }
    if chosen is None:
        return {"mu": me, "sd": se, "delta": ADWIN_DELTAS[0], "feasible": False}
    return {"mu": me, "sd": se, **chosen, "feasible": True}


# ---------------------------------------------------------------- per-trace evaluation (corrected criterion)
def binom_sf(x, n, p):
    """P(X >= x), X ~ Binomial(n, p)."""
    return float(sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(x, n + 1)))


def evaluate_trace(arm, P, y, fE, fB, inj, seed_fn, q_warm=None, rng_ctrl_seed=0, cp_lo=420):
    """Round 6 injection design on one stream. Returns per-trace statistics of the arm and of its random control."""
    t, a, r, q, e = A2.streams(y, fE, fB)
    n = len(y)
    ev0, fa = arm_events(arm, P, q, e, q_warm)
    cps = injection_cps(t, n, cp_lo=cp_lo)
    ctl = [bool(detect(ev0, t, cp)) for cp in cps]
    plc = [bool(detect(ev0, t, pj, L=100)) for pj in PLACEBO_CLOCKS]  # descriptive: the round 4/5 placebo clocks scored on events (cap 100)
    nul = float(null_rate(ev0, t))
    m = int((t >= t[0] + BURN).sum())
    n_ev = int(ev0[BURN:].sum())
    p_rand = random_rate(n_ev, m, L_CAP - A2.DELAY)
    sigma = float(np.std(fB - y)) + 1e-9
    det, det100 = [], []
    for pi, cp in enumerate(cps):
        for r_ in range(inj):
            ffs = mf.inject(fB, y, cp, np.random.default_rng(seed_fn(pi, r_)), sigma)
            t1, _, _, q1, e1 = A2.streams(y, fE, ffs)
            ev1, _ = arm_events(arm, P, q1, e1, q_warm)
            det.append(bool(detect(ev1, t1, cp)))
            det100.append(bool(detect(ev1, t1, cp, L=100)))
    x1, n1 = int(sum(det)), len(det)
    x2, n2 = int(sum(ctl)), len(ctl)
    plo5 = T.newcombe_diff(int(sum(det100)), n1, int(sum(plc)), len(plc))[1]
    ss, ss_lo, _ = T.newcombe_diff(x1, n1, x2, n2)
    p0 = max(p_rand, nul)
    p3 = binom_sf(x1, n1, p0)
    # random control: events at the same rate, uniformly placed after burn-in, scored by the same rule (200 draws)
    rr = np.random.default_rng(rng_ctrl_seed)
    rc_pass, rc_c3, rc_det, rc_bin = [], [], [], []
    for _ in range(200):
        evr = np.zeros(len(q), bool)
        if n_ev:
            evr[BURN + rr.choice(m, size=min(n_ev, m), replace=False)] = True
        xr = int(
            sum(bool(detect(evr, t, cp)) for cp in cps for _ in range(1)) * inj
        )  # random events do not react to the injection
        xc = int(sum(bool(detect(evr, t, cp)) for cp in cps))
        lo_r = T.newcombe_diff(xr, n1, xc, n2)[1]
        rc_pass.append((xr / n1 - p0 >= MARGIN) and (lo_r > 0))
        rc_c3.append(xr / n1 - p0 >= MARGIN)
        rc_bin.append(binom_sf(xr, n1, p0) < 0.05)
        rc_det.append(xr / n1)
    return {
        "fa": fa,
        "c1": fa <= FA_TARGET,
        "det": x1 / n1,
        "ctl": x2 / n2,
        "ss_lo": ss_lo,
        "c2": ss_lo > 0,
        "null_dense": nul,
        "p_rand": p_rand,
        "p0": p0,
        "p3": p3,
        "placebo_rate_events": float(np.mean(plc)),
        "det_cap100": float(np.mean(det100)),
        "c3_round5_rule_events": bool((plo5 > 0) and (ss_lo > 0)),
        "c3_binom": (p3 < 0.05) and (ss_lo > 0),
        "c3": (x1 / n1 - p0 >= MARGIN) and (ss_lo > 0),
        "joint": (fa <= FA_TARGET) and (ss_lo > 0) and (x1 / n1 - p0 >= MARGIN),
        "n_events_nofault": n_ev,
        "rand_ctrl_pass_share": float(np.mean(rc_pass)),
        "rand_ctrl_c3only_share": float(np.mean(rc_c3)),
        "rand_ctrl_binom_share": float(np.mean(rc_bin)),
        "rand_ctrl_det_mean": float(np.mean(rc_det)),
        "det_by_cp": [float(np.mean(det[i * inj:(i + 1) * inj])) for i in range(len(cps))],
    }
