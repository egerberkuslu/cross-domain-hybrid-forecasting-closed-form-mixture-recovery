"""Round 7 library (frozen by REPAIR_PREREG_7.md): scale-transfer-robust variants of the reference-free monitor. The functions are those of val7.py (the design-input script),
moved here unchanged except for: the removed two-stage arm W2, the event dispatch carrying the delayed Bolt error for the report-only ADB arm, the round 6 per-trace
evaluation reused with this dispatch, the degenerate-expert walk, and the Restricted test view.

Nothing here reads a test file unless the TestView is opened by the committed test stage.

Arms (REPAIR_PREREG_7.md section 3)
  W1    SRC      segment-residual calibration: floor 0.5 sd(q - pcfit(q)), level / K / G calibrated on the residual q - fit (block bootstrap of the residual)
  W3    RS-LMB   burst-robust floor 0.5 * 1.4826 MAD(q), Huber EW variance, level calibrated on raw q with the level-matched block bootstrap
  W4    RS-LMB-RF W3 with the W1 floor
  STB7  fair baseline: static CUSUM, reference = mean of the last 200 validation decisions, scale = sd of the validation residual, level and K calibrated like W1
  V3    round 6 V3 recipe re-selected on the round 7 validation (report only);  ADB  ADWIN comparator as in round 6 (report only)
Every arm: re-armed, burn-in held, onset-localised events, revised fallback (no feasible level -> smallest level meeting the validation-stream null and FA conditions,
flagged infeasible, otherwise the grid maximum) and selection key (feasible, validation detection, lower validation null, larger G).
"""
from __future__ import annotations

import builtins
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R6 = HERE.parent / "round6"
sys.path.insert(0, str(R6))
import r6_lib as L6  # noqa: E402  (puts rounds 3/4, paper_a and fresh_cells on sys.path)
import rcommon as R  # noqa: E402

A2, M3, mf, T = L6.A2, L6.M3, L6.mf, L6.T
VARIANTS = ("W1", "W3", "W4")
ARMS = ("W1", "W3", "W4", "STB7", "V3", "ADB")
DEGEN_LOG = float(
    np.log(100.0)
)  # degenerate-expert rule: |median validation q| > log 100
CP_LO = L6.CP_LO


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


# ------------------------------------------------------------------ events of one arm on one stream
def arm_events7(arm, P, q, e, q_warm=None):
    """Event array (n,) and the round 3-5 FA onset share of one arm with frozen parameters P. Same signature role as r6_lib.arm_events."""
    if arm in ("V3", "ADB"):
        return L6.arm_events(arm, P, q, e, q_warm)
    if arm == "STB7":
        z = (q - P["mu"]) / P["sd"]
        ev, fa = L6.cusum_events(z[None], P["K"], [P["level"]], True, True)
        return ev[0, 0], float(fa[0, 0])
    z = rf_z_rs(q, int(P["G"]), P["floor"], q_warm)
    ev, fa = L6.cusum_events(z[None], P["K"], [P["level"]], True, True)
    return ev[0, 0], float(fa[0, 0])


def det_val(arm, P, vruns, warm):
    return sum(
        int(L6.detect(arm_events7(arm, P, x["q"], None, warm)[0], x["t"], x["cp"]))
        for x in vruns
    ) / len(vruns)


# ------------------------------------------------------------------ selection on validation arrays only
def select7(arm, q, t, e, vruns):
    warm = q[: L6.WARM]
    if arm == "V3":
        return L6.select_rf("V3", q, t, e, vruns)
    if arm == "ADB":
        return L6.select_adb(e, t)
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
            det = det_val(
                "STB7", {**P, "mu": float(q.mean())}, vruns, None
            )  # validation injections are scored around the validation mean
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
        elif arm == "W1":
            floor = max(0.5 * float(res.std()), 0.05)
            rows = np.vstack([res[None], L6.boot_rows(res, G)])
        else:
            raise ValueError(arm)
        Z = np.stack([rf_z_rs(r, G, floor, r[: L6.WARM]) for r in rows])
        P = {"G": G, "K": K, "floor": floor, **calibrate(Z, K, t)}
        P["n_cp_val"] = ncp
        det = det_val(arm, P, vruns, warm)
        vn = P["val_null"] if np.isfinite(P["val_null"]) else 1.0
        key = (P["feasible"], det, -vn, G)
        if best is None or key > best[0]:
            best = (key, {**P, "val_det": det})
    return best[1]


# ------------------------------------------------------------------ per-trace evaluation: r6_lib.evaluate_trace with the round 7 event dispatch
def evaluate_trace7(
    arm, P, y, fE, fB, inj, seed_fn, q_warm=None, rng_ctrl_seed=0, cp_lo=420
):
    """Round 6 injection design on one stream (conditions c1, c2, c3', matched control, random control), events from arm_events7."""
    t, a, r, q, e = A2.streams(y, fE, fB)
    n = len(y)
    ev0, fa = arm_events7(arm, P, q, e, q_warm)
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
            t1, _, _, q1, e1 = A2.streams(y, fE, ffs)
            det.append(bool(L6.detect(arm_events7(arm, P, q1, e1, q_warm)[0], t1, cp)))
    x1, n1 = int(sum(det)), len(det)
    x2, n2 = int(sum(ctl)), len(ctl)
    ss, ss_lo, _ = T.newcombe_diff(x1, n1, x2, n2)
    p0 = max(p_rand, nul)
    p3 = L6.binom_sf(x1, n1, p0)
    rr = np.random.default_rng(rng_ctrl_seed)
    rc_pass, rc_c3, rc_det = [], [], []
    for _ in range(200):
        evr = np.zeros(len(q), bool)
        if n_ev:
            evr[L6.BURN + rr.choice(m, size=min(n_ev, m), replace=False)] = True
        xr = int(
            sum(bool(L6.detect(evr, t, cp)) for cp in cps) * inj
        )  # random events do not react to the injection
        xc = int(sum(bool(L6.detect(evr, t, cp)) for cp in cps))
        lo_r = T.newcombe_diff(xr, n1, xc, n2)[1]
        rc_pass.append((xr / n1 - p0 >= L6.MARGIN) and (lo_r > 0))
        rc_c3.append(xr / n1 - p0 >= L6.MARGIN)
        rc_det.append(xr / n1)
    return {
        "fa": fa,
        "c1": fa <= L6.FA_TARGET,
        "det": x1 / n1,
        "ctl": x2 / n2,
        "ss_lo": ss_lo,
        "c2": ss_lo > 0,
        "null_dense": nul,
        "p_rand": p_rand,
        "p0": p0,
        "p3": p3,
        "c3": (x1 / n1 - p0 >= L6.MARGIN) and (ss_lo > 0),
        "joint": (fa <= L6.FA_TARGET) and (ss_lo > 0) and (x1 / n1 - p0 >= L6.MARGIN),
        "n_events_nofault": n_ev,
        "rand_ctrl_pass_share": float(np.mean(rc_pass)),
        "rand_ctrl_c3only_share": float(np.mean(rc_c3)),
        "rand_ctrl_det_mean": float(np.mean(rc_det)),
        "det_by_cp": [
            float(np.mean(det[i * inj : (i + 1) * inj])) for i in range(len(cps))
        ],
    }


# ------------------------------------------------------------------ degenerate-expert rule (validation forecasts only)
def val_q_median(val_dir, name):
    """Median validation q of one stage-A cell, None if the cell is absent. Reads only r7_val."""
    p = Path(val_dir) / f"{name}__h6.npz"
    if not p.exists():
        return None
    y, fE, fB = M3.streams_from(M3.load_h6(Path(val_dir), name))
    q = A2.streams(y, fE, fB)[3]
    return float(np.median(q))


def degen_walk(manifest, val_dir, cap=3):
    """Deterministic final trace list of the primary pool. Per level (subnets first, then IPs) the ordered list of the manifest is walked: a trace whose |median validation q|
    exceeds log 100 is dropped and replaced by the next trace in the registered order; a trace that would break the family cap (at most `cap` per institution, subnets and IPs
    together, counting the final subnets) is skipped. Returns dict with selected, dropped, skipped_cap, needed (stage-A cells still missing).
    """
    inst = {e["name"]: e["institution"] for e in manifest["series"]}
    target = {lv: len(manifest["monitor_primary"][lv]) for lv in ("sub", "ip")}
    sel, dropped, skipped, needed, qmed = {"sub": [], "ip": []}, [], [], [], {}
    count = {}
    for lv in ("sub", "ip"):
        order = manifest["monitor_order"][lv]
        i = 0
        while len(sel[lv]) < target[lv] and i < len(order):
            slots = target[lv] - len(sel[lv])
            # examine the next `slots` candidates together; whatever has no validation forecast yet is requested in one batch
            window = []
            j = i
            while len(window) < slots and j < len(order):
                window.append(order[j])
                j += 1
            miss = [n for n in window if val_q_median(val_dir, n) is None]
            if miss:
                needed += miss
                break
            nm = order[i]
            i += 1
            qm = val_q_median(val_dir, nm)
            qmed[nm] = qm
            if abs(qm) > DEGEN_LOG:
                dropped.append({"name": nm, "median_q_val": qm})
                continue
            if count.get(inst[nm], 0) >= cap:
                skipped.append(nm)
                continue
            sel[lv].append(nm)
            count[inst[nm]] = count.get(inst[nm], 0) + 1
    return {
        "selected": sel,
        "dropped": dropped,
        "skipped_cap": skipped,
        "needed": needed,
        "median_q_val": qmed,
    }


# ------------------------------------------------------------------ Restricted test view
def guard_test_dir(test_dir):
    """Select / degen / dry-run: any read under r7_test raises (extends the rcommon guard), and the folder must not exist."""
    assert not Path(test_dir).exists(), "r7_test exists before the commit"
    R._GUARDED = tuple(R._GUARDED) + (str(test_dir),)
    try:
        np.load(Path(test_dir) / "any.npz")
        raise AssertionError("r7_test guard did not bite")
    except R.TestAccessError:
        pass


class TestView:
    """Restricted view of the round 7 test forecasts: opening it needs PREREG_7_COMMITTED, REPAIR_STAGE=test and rcommon.allow_test(); a read of one trace returns (y, fE, fB)."""

    __test__ = False

    def __init__(self, test_dir):
        self.dir = Path(test_dir)
        if (
            not (HERE / "PREREG_7_COMMITTED").exists()
            or os.environ.get("REPAIR_STAGE") != "test"
        ):
            raise PermissionError(
                "round 7 test windows are sealed (PREREG_7_COMMITTED and REPAIR_STAGE=test required)"
            )
        R.allow_test()
        self.accessed = []

    def load(self, name):
        self.accessed.append(name)
        return M3.streams_from(M3.load_h6(self.dir, name))
