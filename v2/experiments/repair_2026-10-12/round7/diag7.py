"""Round 7 design diagnosis (DIAGNOSIS7.md): why condition c2 of round 6 failed on cesnet_inst159, 202, 243, 254 (V1 to V3) and how abilene_od110 failed c3'.

Design only. Never re-scores round 6 (no success rule, no X count is recomputed). Two phases:
  phase V  validation streams of the 20 P2 traces (round6/r6_val) and the frozen round 6 parameters (frozen_r6.json, hash checked).
  phase T  the REVEALED round 6 test streams (round6/r6_test), whose result is public (r6_monitor_summary.json). Read for design, disclosed.
Measures, per trace and arm (V1, V3, STB):
  calibration: which of the three null conditions binds the level on validation (stream itself, bootstrap q90 null, bootstrap q90 FA share);
  natural level shifts of q (max standardised 150-window mean step), seasonality (ACF of y at 24 and 168 per split, weekly-profile correlation),
  heteroscedasticity (rolling 60-window sd of the Bolt error relative to the global sd that scales the injection),
  injected signal (q step, z of the RF reference and of a reference frozen at cp, CUSUM maximum against the level),
  adaptation loss (share of the injected z that the RF reference absorbs inside the lag window).
Writes diag7_rows.csv and diag7_summary.json. Memory well below 1 GB.
Run: PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python diag7.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R6 = HERE.parent / "round6"
sys.path.insert(0, str(R6))
import r6_lib as L6  # noqa: E402

A2, M3, mf = L6.A2, L6.M3, L6.mf
FOCUS = [
    "cesnet_inst159",
    "cesnet_inst202",
    "cesnet_inst243",
    "cesnet_inst254",
    "abilene_od110",
]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
test_seed6 = lambda pi, r_: A2.test_seed(pi, r_, 6)


def load(d, name):
    return M3.streams_from(M3.load_h6(d, name))


def max_step(q, w=150):
    """max over t of |mean(q[t:t+w]) - mean(q[t-w:t])| / pooled within-half sd (natural level shift of q)."""
    n = len(q)
    if n < 2 * w + 1:
        w = n // 3
    cs = np.concatenate([[0.0], np.cumsum(q)])
    cs2 = np.concatenate([[0.0], np.cumsum(q * q)])
    best = 0.0
    for t in range(w, n - w + 1):
        m1, m2 = (cs[t] - cs[t - w]) / w, (cs[t + w] - cs[t]) / w
        v1 = (cs2[t] - cs2[t - w]) / w - m1 * m1
        v2 = (cs2[t + w] - cs2[t]) / w - m2 * m2
        s = np.sqrt(max(0.5 * (v1 + v2), 1e-12))
        best = max(best, abs(m2 - m1) / s)
    return float(best)


def acf(x, lag):
    x = np.asarray(x, float)
    x = x - x.mean()
    if len(x) <= lag + 2:
        return float("nan")
    return float(np.dot(x[:-lag], x[lag:]) / np.dot(x, x))


def week_profile(x):
    x = (np.asarray(x, float) - np.mean(x)) / (np.std(x) + 1e-12)
    k = len(x) // 168
    if k < 1:
        return None
    return x[: k * 168].reshape(k, 168).mean(0)


def rf_z_frozen(q, G, floor, cp_idx, q_warm=None):
    """RF z, but the reference (m, v) is frozen at decision cp_idx (what the variant 'freeze on suspicion' would see)."""
    qq = q if q_warm is None else np.concatenate([q_warm, q])
    off = 0 if q_warm is None else len(q_warm)
    n = len(qq)
    z = np.zeros(n)
    m = float(np.mean(qq[:20]))
    v = float(np.var(qq[:20])) + 1e-12
    for t in range(n):
        if t - G >= 0 and t < cp_idx + off:
            x = qq[t - G]
            m = M3.DECAY * m + (1 - M3.DECAY) * x
            v = M3.DECAY * v + (1 - M3.DECAY) * (x - m) ** 2
        z[t] = (qq[t] - m) / max(np.sqrt(v), floor)
    return z[off:]


def z_of(arm, P, q, q_warm):
    if arm in ("V1", "V3"):
        return L6.rf_z_warm(q, P["G"], P["floor"], q_warm if arm == "V3" else None)
    return (q - P["mu"]) / P["sd"]


def cusum_max(z, K, lo, hi, burn_hold=True):
    S, mx = 0.0, 0.0
    for k in range(len(z)):
        S = 0.0 if (burn_hold and k < L6.BURN) else max(0.0, S + z[k] - K)
        if lo <= k < hi:
            mx = max(mx, S)
    return mx


def binding(arm, P, q, t):
    """Validation: smallest grid level meeting each null condition separately, with the frozen (G, K) of the arm."""
    if arm == "STB":
        z = (q - P["mu"]) / P["sd"]
        rows = np.vstack([z[None], L6.boot_rows(z, 60)])
        K, rearm, hold = P["K"], True, True
    else:
        G = P["G"]
        rows_q = np.vstack([q[None], L6.boot_rows(q, G)])
        if arm == "V3":
            rows = np.stack(
                [L6.rf_z_warm(r, G, P["floor"], r[: L6.WARM]) for r in rows_q]
            )
        else:
            rows = np.stack([M3.rf_z(r, G, P["floor"]) for r in rows_q])
        K, rearm, hold = P["K"], arm != "V1", arm != "V1"
    ev, fa = L6.cusum_events(rows, K, L6.LEVELS, rearm, hold)
    nr = L6.null_rate(ev, t)
    c_self = nr[0] <= L6.NULL_TARGET
    c_boot = np.quantile(nr[1:], 0.9, axis=0) <= L6.NULL_TARGET
    c_fa = np.quantile(fa[1:], 0.9, axis=0) <= L6.FA_TARGET
    first = lambda m: float(L6.LEVELS[np.argmax(m)]) if m.any() else float("inf")
    return {
        "lvl_self": first(c_self),
        "lvl_boot": first(c_boot),
        "lvl_fa": first(c_fa),
        "lvl_all": first(c_self & c_boot & c_fa),
        "boot_null_q90_at_self": float(np.quantile(nr[1:, np.argmax(c_self)], 0.9))
        if c_self.any()
        else float("nan"),
    }


def stream_feats(y, fE, fB, q):
    eB = fB - y
    sig = float(np.std(eB))
    rs = pd.Series(eB).rolling(60).std().to_numpy()[60:]
    return {
        "q_sd": float(np.std(q)),
        "q_step150": max_step(q),
        "acf24_y": acf(y, 24),
        "acf168_y": acf(y, 168),
        "het_ratio_p90_p10": float(
            np.nanpercentile(rs, 90) / max(np.nanpercentile(rs, 10), 1e-12)
        ),
        "sigma_inj": sig,
    }


def injected(arm, P, y, fE, fB, q_warm, cp_lo=420):
    """Revealed test only: injected runs regenerated with the round 6 test seeds; per change point the q step, z, frozen-reference z, CUSUM max."""
    t, a, r, q, e = A2.streams(y, fE, fB)
    n = len(y)
    cps = L6.injection_cps(t, n, cp_lo=cp_lo)
    sigma = float(np.std(fB - y)) + 1e-9
    z0 = z_of(arm, P, q, q_warm)
    K = P.get("K", 0.5)
    out = []
    for pi, cp in enumerate(cps):
        lo = int(np.searchsorted(t, cp + A2.DELAY + 1))
        hi = int(np.searchsorted(t, cp + L6.L_CAP, side="right"))
        ci = int(np.searchsorted(t, cp))
        loc = float(
            np.std((fB - y)[max(0, cp - 60) : cp])
        )  # local Bolt error sd before cp
        dq, zin, zfr, smx = [], [], [], []
        for r_ in range(A2.INJ_TEST):
            ffs = mf.inject(fB, y, cp, np.random.default_rng(test_seed6(pi, r_)), sigma)
            _, _, _, q1, _ = A2.streams(y, fE, ffs)
            z1 = z_of(arm, P, q1, q_warm)
            dq.append(float(np.mean(q1[lo:hi] - q[lo:hi])))
            zin.append(float(np.mean(z1[lo:hi])))
            if arm in ("V1", "V3"):
                zfr.append(
                    float(
                        np.mean(
                            rf_z_frozen(
                                q1,
                                P["G"],
                                P["floor"],
                                ci,
                                q_warm if arm == "V3" else None,
                            )[lo:hi]
                        )
                    )
                )
            smx.append(cusum_max(z1, K, lo, hi, burn_hold=arm != "V1"))
        out.append(
            {
                "cp": int(cp),
                "snr_local": float(3 * sigma / max(loc, 1e-12)),
                "dq_mean": float(np.mean(dq)),
                "dq_over_qsd": float(np.mean(dq) / (np.std(q) + 1e-12)),
                "z_inj_mean": float(np.mean(zin)),
                "z_frozen_mean": float(np.mean(zfr)) if zfr else float("nan"),
                "z_nofault_mean": float(np.mean(z0[lo:hi])),
                "S_max_inj_median": float(np.median(smx)),
                "level": float(P["level"]),
                "S_max_nofault": cusum_max(z0, K, lo, hi, burn_hold=arm != "V1"),
            }
        )
    return out


def main():
    fr_p = R6 / "frozen_r6.json"
    assert sha(fr_p) == (R6 / "frozen_r6.sha256").read_text().strip()
    fr = json.loads(fr_p.read_text())
    man = json.loads((R6 / "pool6_manifest.json").read_text())
    names = man["monitor_traces_P2"]
    rows6 = pd.read_csv(R6 / "r6_monitor_rows.csv")
    out, prof = [], {}
    for name in names:
        P = fr["params"][name]
        yv, fEv, fBv = load(R6 / "r6_val", name)
        tv, _, _, qv, _ = A2.streams(yv, fEv, fBv)
        yt, fEt, fBt = load(
            R6 / "r6_test", name
        )  # REVEALED round 6 test stream (design only, disclosed)
        tt, _, _, qt, _ = A2.streams(yt, fEt, fBt)
        fv, ft = stream_feats(yv, fEv, fBv, qv), stream_feats(yt, fEt, fBt, qt)
        pv, pt = week_profile(yv), week_profile(yt)
        prof_corr = (
            float(np.corrcoef(pv, pt)[0, 1])
            if pv is not None and pt is not None
            else float("nan")
        )
        for arm in ("V1", "V3", "STB"):
            Pa = P[arm]
            qw = np.array(P["q_warm_test"]) if arm == "V3" else None
            b = binding(arm, Pa, qv, tv)
            inj = injected(arm, Pa, yt, fEt, fBt, qw)
            r6 = rows6[(rows6.series == name) & (rows6.arm == arm)].iloc[0]
            out.append(
                {
                    "series": name,
                    "arm": arm,
                    "focus": name in FOCUS,
                    "feasible": bool(Pa.get("feasible", True)),
                    "G": Pa.get("G"),
                    "K": Pa.get("K"),
                    "level": Pa["level"],
                    "val_det": Pa.get("val_det"),
                    **{f"cal_{k}": v for k, v in b.items()},
                    **{f"val_{k}": v for k, v in fv.items()},
                    **{f"test_{k}": v for k, v in ft.items()},
                    "week_profile_corr_val_test": prof_corr,
                    "r6_det": r6.det,
                    "r6_ctl": r6.ctl,
                    "r6_c2": bool(r6.c2),
                    "r6_joint": bool(r6.joint),
                    "r6_n_events_nofault": int(r6.n_events_nofault),
                    "r6_det_by_cp": r6.det_by_cp,
                    "inj": json.dumps(inj),
                }
            )
        print("diag7", name, flush=True)
    df = pd.DataFrame(out)
    df.to_csv(HERE / "diag7_rows.csv", index=False)
    summ = {
        "note": "design diagnosis; validation streams plus the REVEALED round 6 test streams; round 6 is not re-scored",
        "frozen_r6_sha256": sha(fr_p),
    }
    for arm in ("V1", "V3", "STB"):
        g = df[df.arm == arm]
        summ[arm] = {}
        for grp, gg in (("c2_failed", g[~g.r6_c2]), ("c2_passed", g[g.r6_c2])):
            inj = [x for s in gg.inj for x in json.loads(s)]
            summ[arm][grp] = {
                "n": int(len(gg)),
                "traces": gg.series.tolist(),
                "feasible": int(gg.feasible.sum()),
                "median_level": float(gg.level.median()),
                "median_val_q_step150": float(gg.val_q_step150.median()),
                "median_test_q_step150": float(gg.test_q_step150.median()),
                "median_test_het_ratio": float(gg.test_het_ratio_p90_p10.median()),
                "median_week_profile_corr": float(
                    gg.week_profile_corr_val_test.median()
                ),
                "median_dq_over_qsd": float(np.median([x["dq_over_qsd"] for x in inj]))
                if inj
                else None,
                "median_z_inj": float(np.median([x["z_inj_mean"] for x in inj]))
                if inj
                else None,
                "median_z_frozen": float(
                    np.nanmedian([x["z_frozen_mean"] for x in inj])
                )
                if inj and arm != "STB"
                else None,
                "share_cp_Smax_over_level": float(
                    np.mean([x["S_max_inj_median"] > x["level"] for x in inj])
                )
                if inj
                else None,
                "binding_boot_gt_self": int((gg.cal_lvl_boot > gg.cal_lvl_self).sum()),
                "binding_fa_gt_self": int((gg.cal_lvl_fa > gg.cal_lvl_self).sum()),
            }
    (HERE / "diag7_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
