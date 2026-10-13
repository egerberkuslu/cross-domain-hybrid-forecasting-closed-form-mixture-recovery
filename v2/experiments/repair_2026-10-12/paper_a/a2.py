"""A-A2: repair of the alpha-monitor (detectors M0-M5b). Definitions frozen in REPAIR_PREREG.md section 4.

  python3 a2.py --stage select [--dry-run]     validation only (guard on), writes frozen_a2.json (+ sha256)
  REPAIR_STAGE=test python3 a2.py --stage test needs PREREG_COMMITTED
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

import rcommon as R
import stats_tools as T

sys.path.insert(0, str(R.FIX))
import monitor_far as mf  # noqa: E402

H = 6
DELAY = H
W = 60
POSITIONS_TEST = [0.2, 0.45, 0.7]
POSITIONS_VAL = [0.3, 0.6]
INJ_TEST, INJ_VAL = 10, 5
KS = [2, 3, 5, 8]
LAG_CAP = 150
ADWIN_DELTAS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.05]
TARGET = 0.05
CUSUM_DETS = ("M0", "M1", "M2", "M3", "M4", "M5a")
DETS = ("M0", "M1", "M2", "M3", "M4", "M5a", "M5b")
ALPHA_BASED = ("M1", "M2", "M3", "M4")


# ------------------------------------------------------------------ streams
def streams(y, fd, ff, delay=DELAY):
    """t_arr, a, r (as monitor_far), q (log MSE ratio Bolt/decomposition), e (delayed Bolt squared error)."""
    n = len(y)
    ef, ed = ff - y, fd - y
    diff = ed - ef
    cs = lambda x: np.concatenate([[0.0], np.cumsum(x)])
    c_ed, c_dd = cs(ef * diff), cs(diff * diff)
    c_ff, c_fd = cs(ef * ef), cs(ed * ed)
    t = np.arange(W + delay, n)
    hi, lo = t - delay, t - delay - W
    num, den = c_ed[hi] - c_ed[lo], c_dd[hi] - c_dd[lo]
    a = np.where(den < 1e-12, 0.0, np.clip(-num / np.maximum(den, 1e-300), 0, 1))
    mf_ = (c_ff[hi] - c_ff[lo]) / W
    md_ = (c_fd[hi] - c_fd[lo]) / W
    r = np.sqrt(mf_)
    q = np.log(np.maximum(mf_, 1e-12) / np.maximum(md_, 1e-12))
    e = ef[t - delay - 1] ** 2
    return t, a, r, q, e


def adwin_alarms(z, delta, min_len=5):
    n = len(z)
    cs, cs2 = np.concatenate([[0.0], np.cumsum(z)]), np.concatenate([[0.0], np.cumsum(z * z)])
    out = np.zeros(n, bool)
    start = 0
    for k in range(n):
        w = k + 1 - start
        if w < 2 * min_len:
            continue
        j = np.arange(min_len, w - min_len + 1)
        n0, n1 = j.astype(float), (w - j).astype(float)
        s0, s1 = cs[start + j] - cs[start], cs[k + 1] - cs[start + j]
        var = max((cs2[k + 1] - cs2[start]) / w - ((cs[k + 1] - cs[start]) / w) ** 2, 0.0)
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        dp = delta / np.log(max(w, 3.0))
        eps = np.sqrt(2.0 * var / m * np.log(2.0 / dp)) + 2.0 / (3.0 * m) * np.log(2.0 / dp)
        bad = np.abs(s0 / n0 - s1 / n1) > eps
        if bad.any():
            out[k] = True
            start += int(j[np.argmax(bad)])
    return out


def onsets_k(hit, k):
    out = np.zeros(hit.size, bool)
    run = 0
    for i, x in enumerate(hit):
        run = run + 1 if x else 0
        if run >= k:
            out[i] = True
            run = 0
    return out


def eval_run(hit, t_arr, cp, t_info, k):
    """Copy of monitor_far.eval_fault with persistence k. Returns (pre_fa, pre_info_alarm, lag)."""
    pre = t_arr < cp
    on = onsets_k(hit[pre], k)
    fa = float(on.mean()) if pre.any() else np.nan
    mid = (t_arr >= cp) & (t_arr < t_info)
    pia = bool(onsets_k(hit[mid], k).any()) if mid.any() else False
    post = t_arr >= t_info
    lag, run = None, 0
    for tk, x in zip(t_arr[post], hit[post]):
        run = run + 1 if x else 0
        if run >= k:
            lag = int(tk - cp)
            break
    return fa, pia, lag


# ---------------------------------------------------------------- detectors
def cusum_hits(x, mu, sd, h):
    return mf.cusum_hits(x, mu, sd, h, 0)


def build_params(a, r, q, e, dry=False):
    """All thresholds from ONE no-fault validation stream (validation only)."""
    sm = mf.smooth(a, "ema")
    P = {}
    p0 = mf.design_params(a, sm, r)
    P["M0"] = {k: (float(v) if not isinstance(v, bool) else v) for k, v in p0.items()}
    mu, sd = float(a.mean()), max(float(a.std()), mf.SD_FLOOR_ALPHA)
    tau1 = T.null_tau(a, seed=11)
    h1, f1 = T.null_level((a - mu) / sd, q=0.90, seed=12)
    P["M1"] = {"tau": tau1, "mu": mu, "sd": sd, "h": h1, "feasible": f1}
    mq, sq = float(q.mean()), max(float(q.std()), 1e-3)
    h2, f2 = T.null_level((q - mq) / sq, q=0.90, seed=13)
    P["M2"] = {"mu": mq, "sd": sq, "h": h2, "feasible": f2}
    P["M3"] = {"tau": tau1, "h1": h1, "h2": h2, "mu_a": mu, "sd_a": sd, "mu_q": mq, "sd_q": sq, "feasible": f1 and f2}
    me, se = float(e.mean()), max(float(e.std()), 1e-12)
    z = (e - me) / se
    h5, f5 = T.null_level(z, q=0.90, seed=14)
    P["M5a"] = {"mu": me, "sd": se, "h": h5, "feasible": f5}
    # ADWIN: largest delta whose bootstrap q90 onset share <= target (B = 100)
    rng = np.random.default_rng(15)
    Z = T.mbb_resample(z, 60, 30 if dry else 100, rng)
    chosen, feas = ADWIN_DELTAS[0], False
    for dl in ADWIN_DELTAS:
        rates = np.array([adwin_alarms(row, dl).mean() for row in Z])
        if np.quantile(rates, 0.90) <= TARGET:
            chosen, feas = dl, True
    P["M5b"] = {"mu": me, "sd": se, "delta": chosen, "feasible": feas}
    return P


def m4_choose(val_inj, a, P1):
    """Persistence k chosen on validation-only injections (highest detection among null-feasible (k, h) pairs)."""
    mu, sd = P1["mu"], P1["sd"]
    best = None
    for k in KS:
        hk, fk = T.null_level((a - mu) / sd, q=0.90, persist=k, seed=20 + k)
        if not fk:
            continue
        det = 0
        for (t_arr, a1, cp, t_info) in val_inj:
            sm = mf.smooth(a1, "ema")
            hit = (sm >= P1["tau"]) & cusum_hits(a1, mu, sd, hk)
            det += eval_run(hit, t_arr, cp, t_info, k)[2] is not None
        rate = det / max(1, len(val_inj))
        if best is None or rate > best[0] + 1e-12:
            best = (rate, k, hk)
    if best is None:
        return {"k": 3, "h": P1["h"], "val_det": float("nan"), "feasible": False}
    return {"k": best[1], "h": best[2], "val_det": best[0], "feasible": True}


def detector_hits(P, a, r, q, e):
    """Hit arrays (bool, decision grid) of every detector for one stream. M4 uses its own k (returned separately)."""
    sm = mf.smooth(a, "ema")
    out = {}
    c0 = mf.conditions(a, sm, r, {**P["M0"]}, start=0)
    out["M0"] = c0["comp"]
    p1 = P["M1"]
    cus1 = cusum_hits(a, p1["mu"], p1["sd"], p1["h"])
    out["M1"] = (sm >= p1["tau"]) & cus1
    p2 = P["M2"]
    cus2 = cusum_hits(q, p2["mu"], p2["sd"], p2["h"])
    out["M2"] = cus2
    out["M3"] = out["M1"] & cus2
    p4 = P["M4"]
    out["M4"] = (sm >= p1["tau"]) & cusum_hits(a, p1["mu"], p1["sd"], p4["h"])
    p5 = P["M5a"]
    out["M5a"] = cusum_hits(e, p5["mu"], p5["sd"], p5["h"])
    pb = P["M5b"]
    out["M5b"] = adwin_alarms((e - pb["mu"]) / pb["sd"], pb["delta"])
    return out


def block_perm(n, block, rng):
    nb = int(np.ceil(n / block))
    order = rng.permutation(nb)
    return np.concatenate([np.arange(b * block, min((b + 1) * block, n)) for b in order])


def persist_of(det, P):
    return 1 if det == "M5b" else (P["M4"]["k"] if det == "M4" else T.PERSIST_DEFAULT)


# --------------------------------------------------------------------- data
def val_stream(ds, s, pseudo_test=False):
    v = R._raw_val(ds, H, s)
    y, fd, ff = (v[k].mean(1) for k in ("y", "dec", "glob"))
    return y, fd, ff


def test_stream(ds, s, pseudo):
    if pseudo:
        return val_stream(ds, s)
    t = R.load_test(ds, H, s)
    return t["y"].mean(1), t["dec"].mean(1), t["glob"].mean(1)


def change_points(t_arr, n, positions):
    lo, hi = int(t_arr[0]) + 20, n - 110
    if hi <= lo:
        hi = lo + 1
    return [int(round(lo + p * (hi - lo))) for p in positions]


def inject_streams(y, fd, ff, positions, nseed, seed_fn, s):
    sigma = float(np.std(ff - y)) + 1e-9
    runs = []
    t_arr = None
    for pi, cp in enumerate(change_points(streams(y, fd, ff)[0], len(y), positions)):
        for r_ in range(nseed):
            rng = np.random.default_rng(seed_fn(pi, r_, s))
            ffs = mf.inject(ff, y, cp, rng, sigma)
            t_arr, a1, r1, q1, e1 = streams(y, fd, ffs)
            runs.append({"pos": positions[pi], "cp": cp, "t_info": cp + DELAY + 1, "inj": r_, "t": t_arr, "a": a1, "r": r1, "q": q1, "e": e1})
    return runs


val_seed = lambda pi, r_, s: 5_000_017 * (pi + 1) + 7919 * r_ + s
test_seed = lambda pi, r_, s: 1_000_003 * (pi + 1) + 7919 * r_ + s


# --------------------------------------------------------------------- select
def select(dry):
    log = R.install_guard()
    R.guard_selftest()
    traces = R.DATASETS[:2] if dry else R.DATASETS
    seeds = R.SEEDS[:1] if dry else R.SEEDS
    frozen, rows = {}, []
    for ds in traces:
        for s in seeds:
            y, fd, ff = val_stream(ds, s)
            t_arr, a, r, q, e = streams(y, fd, ff)
            if len(t_arr) < 30:
                continue
            # check the vectorised streams against monitor_far on this very stream
            t2, a2_, r2_ = mf.alpha_and_rmse_stream(y, fd, ff, DELAY)
            assert np.allclose(a, a2_, atol=1e-9) and np.allclose(r, r2_, atol=1e-9), "stream mismatch with monitor_far"
            P = build_params(a, r, q, e, dry)
            c0 = mf.conditions(a, mf.smooth(a, "ema"), r, P["M0"], start=0)["comp"]
            cpv = int(t_arr[len(t_arr) // 2])
            assert eval_run(c0, t_arr, cpv, cpv + DELAY + 1, 3) == mf.eval_fault(c0, t_arr, cpv, cpv + DELAY + 1, int(t_arr[0])), "eval_run differs from monitor_far.eval_fault"
            vinj = [(x["t"], x["a"], x["cp"], x["t_info"]) for x in inject_streams(y, fd, ff, POSITIONS_VAL, INJ_VAL, val_seed, s)]
            P["M4"] = m4_choose(vinj, a, P["M1"])
            # validation detection of every detector (selection of the headline alpha-based monitor)
            vruns = inject_streams(y, fd, ff, POSITIONS_VAL, INJ_VAL, val_seed, s)
            det_rate = {d: 0 for d in DETS}
            for x in vruns:
                hits = detector_hits(P, x["a"], x["r"], x["q"], x["e"])
                for d in DETS:
                    det_rate[d] += eval_run(hits[d], x["t"], x["cp"], x["t_info"], persist_of(d, P))[2] is not None
            nof = detector_hits(P, a, r, q, e)
            val_fa = {d: float(onsets_k(nof[d], persist_of(d, P)).mean()) for d in DETS}
            key = f"{ds}|{s}"
            frozen[key] = P
            rows.append({"dataset": ds, "seed": s, **{f"val_det_{d}": det_rate[d] / len(vruns) for d in DETS}, **{f"val_fa_{d}": val_fa[d] for d in DETS},
                         **{f"feasible_{d}": P[d].get("feasible", True) for d in ("M1", "M2", "M3", "M4", "M5a", "M5b")}, "n_val_decisions": len(t_arr)})
            R.progress(f"a2 select {ds} s{s}")
    df = pd.DataFrame(rows)
    assert not any("predictions" in p for p in log), "test data touched"
    ab = {d: float(df[f"val_det_{d}"].mean()) for d in ALPHA_BASED}
    headline = max(ALPHA_BASED, key=lambda d: (ab[d] - 1e-9 * ALPHA_BASED.index(d)))
    out = {"stage": "select", "dry_run": dry, "headline": headline, "val_detection_mean": {d: float(df[f"val_det_{d}"].mean()) for d in DETS},
           "val_fa_mean": {d: float(df[f"val_fa_{d}"].mean()) for d in DETS}, "params": frozen, "test_files_read": 0}
    print(json.dumps({k: v for k, v in out.items() if k != "params"}, indent=1))
    if dry:
        R.enable_pseudo()
        test_stage(frozen_in=out, pseudo=True, traces=traces, seeds=seeds)
        n_read = len([p for p in log if "predictions" in p])
        assert n_read == 0, "test data touched"
        print("dry-run ok; test files read:", n_read)
        return
    df.to_csv(R.HERE / "a2_select_rows.csv", index=False)
    p = R.HERE / "frozen_a2.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    (R.HERE / "frozen_a2.sha256").write_text(R.sha256_file(p) + "\n")


# ----------------------------------------------------------------------- test
def test_stage(frozen_in=None, pseudo=False, traces=None, seeds=None):
    if not pseudo:
        R.allow_test()
        p = R.HERE / "frozen_a2.json"
        assert R.sha256_file(p) == (R.HERE / "frozen_a2.sha256").read_text().strip(), "frozen_a2.json changed"
        frozen_in = json.loads(p.read_text())
    traces = traces or R.DATASETS
    seeds = seeds or R.SEEDS
    inj = 2 if pseudo else INJ_TEST
    headline = frozen_in["headline"]
    run_rows, nf_rows = [], []
    for ds in traces:
        for s in seeds:
            P = frozen_in["params"][f"{ds}|{s}"]
            y, fd, ff = test_stream(ds, s, pseudo)
            t_arr, a, r, q, e = streams(y, fd, ff)
            nof = detector_hits(P, a, r, q, e)
            for d in DETS:
                nf_rows.append({"dataset": ds, "seed": s, "detector": d, "fa": float(onsets_k(nof[d], persist_of(d, P)).mean()), "n": len(t_arr)})
            cps = change_points(t_arr, len(y), POSITIONS_TEST)
            for pi, cp in enumerate(cps):
                t_info = cp + DELAY + 1
                for d in DETS:  # matched no-injection control
                    _, pia, lag = eval_run(nof[d], t_arr, cp, t_info, persist_of(d, P))
                    run_rows.append({"dataset": ds, "seed": s, "detector": d, "pos": POSITIONS_TEST[pi], "inj": -1, "kind": "ctl", "detected": lag is not None, "lag": lag, "pre_info": pia, "m_after": int((t_arr >= t_info).sum())})
            sigma = float(np.std(ff - y)) + 1e-9
            for pi, cp in enumerate(cps):
                t_info = cp + DELAY + 1
                for r_ in range(inj):
                    rng = np.random.default_rng(test_seed(pi, r_, s))
                    ffs = mf.inject(ff, y, cp, rng, sigma)
                    t1, a1, r1, q1, e1 = streams(y, fd, ffs)
                    hits = detector_hits(P, a1, r1, q1, e1)
                    for d in DETS:
                        _, pia, lag = eval_run(hits[d], t1, cp, t_info, persist_of(d, P))
                        run_rows.append({"dataset": ds, "seed": s, "detector": d, "pos": POSITIONS_TEST[pi], "inj": r_, "kind": "inj", "detected": lag is not None, "lag": lag, "pre_info": pia, "m_after": int((t1 >= t_info).sum())})
                    # block-shuffled control: the same injected stream with its decisions permuted in blocks of 60
                    perm = block_perm(len(t1), 60, np.random.default_rng(test_seed(pi, r_, s) + 99))
                    hs = detector_hits(P, a1[perm], r1[perm], q1[perm], e1[perm])
                    for d in DETS:
                        _, pia, lag = eval_run(hs[d], t1, cp, t_info, persist_of(d, P))
                        run_rows.append({"dataset": ds, "seed": s, "detector": d, "pos": POSITIONS_TEST[pi], "inj": r_, "kind": "shuf", "detected": lag is not None, "lag": lag, "pre_info": pia, "m_after": int((t1 >= t_info).sum())})
        R.progress(f"a2 test {ds}")
    runs, nf = pd.DataFrame(run_rows), pd.DataFrame(nf_rows)
    out_dir = R.OUT
    runs.to_csv(out_dir / "a2_test_runs.csv", index=False)
    nf.to_csv(out_dir / "a2_test_nofault.csv", index=False)
    summ = summarize(runs, nf, headline, pseudo)
    (out_dir / "a2_test_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k != "table"}, indent=1, default=float))


def summarize(runs, nf, headline, pseudo):
    tab = []
    for d in DETS:
        for ds in sorted(runs.dataset.unique()):
            sel = lambda k: runs[(runs.detector == d) & (runs.dataset == ds) & (runs.kind == k)]
            ri, rc, rs = sel("inj"), sel("ctl"), sel("shuf")
            fa = nf[(nf.detector == d) & (nf.dataset == ds)].fa.mean()
            x1, n1, x2, n2, x3, n3 = int(ri.detected.sum()), len(ri), int(rc.detected.sum()), len(rc), int(rs.detected.sum()), len(rs)
            dd, lo, hi = T.newcombe_diff(x1, n1, x2, n2)
            cap = lambda g: int((g.detected & (g.lag.astype(float) <= LAG_CAP)).sum())  # prompt detection: lag <= 150 decisions
            cd, clo, chi = T.newcombe_diff(cap(ri), n1, cap(rs), n3)
            lags = ri[ri.detected].lag.astype(float)
            tab.append({"detector": d, "dataset": ds, "fa": float(fa), "det_rate": x1 / n1, "ctl_rate": x2 / n2, "shift_specific": dd, "ss_lo": lo, "ss_hi": hi,
                        "shuf_rate": x3 / n3, "above_shuffled": cd, "above_shuffled_lo": clo, "lag_median": float(lags.median()) if len(lags) else None,
                        "pre_info_share": float(ri.pre_info.mean())})
    t = pd.DataFrame(tab)
    h = t[t.detector == headline]
    c1 = int((h.fa <= 0.05).sum())
    c2 = int((h.ss_lo > 0).sum())
    # non-inferiority against the better of M5a / M5b (pooled detection on injected runs, paired by run)
    key = ["dataset", "seed", "pos", "inj"]
    inj = runs[runs.kind == "inj"]
    wide = inj.pivot_table(index=key, columns="detector", values="detected", aggfunc="first").astype(float)
    best5 = max(("M5a", "M5b"), key=lambda d: wide[d].mean())
    diff = (wide[headline] - wide[best5]).values
    rng = np.random.default_rng(3)
    bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000 if not pseudo else 200)])
    ni_lo = float(np.quantile(bs, 0.025))
    cond3 = bool(((h.above_shuffled_lo > 0) & (h.ss_lo > 0)).sum() >= min(c2, 4) and ni_lo > -0.10)
    # exact McNemar per (trace, placement): headline vs M0 and vs best5, Holm within comparison
    mc = {}
    for other in ("M0", best5):
        ps, lab = [], []
        for (ds, pos), g in inj.groupby(["dataset", "pos"]):
            w = g.pivot_table(index=["seed", "inj"], columns="detector", values="detected", aggfunc="first")
            b = int(((w[headline] == 1) & (w[other] == 0)).sum())
            c = int(((w[headline] == 0) & (w[other] == 1)).sum())
            ps.append(T.mcnemar_exact(b, c))
            lab.append((ds, pos, b, c))
        ph = R.holm(np.array(ps))
        mc[other] = [{"dataset": l[0], "pos": l[1], "b_head_only": l[2], "c_other_only": l[3], "p": float(p), "p_holm": float(pp)} for l, p, pp in zip(lab, ps, ph)]
    return {"headline": headline, "best_M5": best5, "cond1_fa_le5_traces": c1, "cond2_shift_specific_traces": c2, "noninferiority_lo95": ni_lo,
            "cond3": cond3, "success": bool(c1 >= 4 and c2 >= 4 and cond3), "mcnemar": mc, "table": tab, "pseudo": pseudo}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    select(a.dry_run) if a.stage == "select" else test_stage()
