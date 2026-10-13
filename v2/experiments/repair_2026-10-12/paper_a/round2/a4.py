"""Round 2, part 2: drift-adaptive alpha-monitor (D-A rolling-recalibrated CUSUM, D-B two-sample KS change detector).
Definitions frozen in REPAIR_PREREG_2.md section 5. Reuses the streams, injection and evaluation of round 1 (a2.py).

  python3 a4.py --stage select [--dry-run]        validation only (guard on), writes frozen_a4.json (+ sha256)
  REPAIR_STAGE=test python3 a4.py --stage test    needs PREREG_2_COMMITTED
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
sys.path.insert(0, str(P1))
import rcommon as R  # noqa: E402
import stats_tools as T  # noqa: E402
import a2 as A2  # noqa: E402
import monitor_far as mf  # noqa: E402

DA_GRID = [(Wc, G, K) for Wc in (30, 60, 120, 240) for G in (10, 30, 60) for K in (0.5, 1.0, 2.0)]
DB_GRID = [(30, 60, 10), (30, 90, 30), (45, 90, 30), (60, 120, 30), (60, 240, 60)]
KS_LEVELS = np.arange(0.30, 0.9751, 0.025)
DA_LEVELS = np.arange(1.0, 60.01, 1.0)
TARGET = 0.05
LAG_CAP = A2.LAG_CAP
DETS = ("M0", "M1", "M5b", "DA", "DB")
PSEUDO = False


def allow_test2():
    if not (HERE / "PREREG_2_COMMITTED").exists():
        raise R.TestAccessError("PREREG_2_COMMITTED file missing: test data stay sealed")
    R.allow_test()  # also needs REPAIR_STAGE=test and the round-1 file


# ------------------------------------------------------------------ D-A
def da_floor(sm_val):
    """sd floor of the trailing standardisation, fixed from the validation design stream."""
    return float(max(0.5 * np.std(sm_val), 0.01))


def roll_z(X, Wc, G, floor=0.01):
    """Trailing-reference standardisation of each row: z_j = (x_j - mean(ref)) / sd(ref), ref = x[j-G-Wc : j-G]; 0 before burn-in."""
    X = np.atleast_2d(X)
    B, n = X.shape
    c1 = np.concatenate([np.zeros((B, 1)), np.cumsum(X, 1)], 1)
    c2 = np.concatenate([np.zeros((B, 1)), np.cumsum(X * X, 1)], 1)
    Z = np.zeros_like(X)
    j = np.arange(G + Wc, n)
    if j.size == 0:
        return Z
    hi, lo = j - G, j - G - Wc
    m = (c1[:, hi] - c1[:, lo]) / Wc
    v = np.maximum((c2[:, hi] - c2[:, lo]) / Wc - m * m, 0.0)
    sd = np.maximum(np.sqrt(v), floor)
    Z[:, j] = (X[:, j] - m) / sd
    return Z


def da_level(sm_val, Wc, G, K, q=0.90, B=100, seed=31):
    rng = np.random.default_rng(seed)
    blk = int(min(2 * (Wc + G), max(60, len(sm_val) // 2)))
    Zs = roll_z(T.mbb_resample(sm_val, blk, B, rng), Wc, G, da_floor(sm_val))
    S = T.cusum_S(Zs, K)
    n = len(sm_val)
    rates = T.onset_rate_levels([S], DA_LEVELS, T.PERSIST_DEFAULT) * n / max(1, n - (G + Wc))
    qr = np.quantile(rates, q, axis=0)
    ok = np.where(qr <= TARGET)[0]
    if ok.size == 0:
        return float(DA_LEVELS[-1]), False
    return float(DA_LEVELS[ok[0]]), True


def da_hits(a, Wc, G, K, h, floor):
    sm = mf.smooth(a, "ema")
    z = roll_z(sm, Wc, G, floor)
    S = T.cusum_S(z, K)[0]
    return S > h


# ------------------------------------------------------------------ D-B
def ks_stream(x, Wr, Wc, G):
    n = len(x)
    out = np.zeros(n)
    for j in range(G + Wc + Wr - 1, n):
        r = np.sort(x[j - Wr + 1:j + 1])
        f = np.sort(x[j - Wr - G - Wc + 1:j - Wr - G + 1])
        allv = np.concatenate([r, f])
        out[j] = np.max(np.abs(np.searchsorted(r, allv, "right") / Wr - np.searchsorted(f, allv, "right") / Wc))
    return out


def db_level(sm_val, Wr, Wc, G, q=0.90, B=60, seed=41):
    rng = np.random.default_rng(seed)
    blk = int(min(2 * (Wc + G), max(60, len(sm_val) // 2)))
    Zs = T.mbb_resample(sm_val, blk, B, rng)
    D = np.stack([ks_stream(row, Wr, Wc, G) for row in Zs])
    n = len(sm_val)
    rates = T.onset_rate_levels([D], KS_LEVELS, T.PERSIST_DEFAULT) * n / max(1, n - (G + Wc + Wr - 1))
    qr = np.quantile(rates, q, axis=0)
    ok = np.where(qr <= TARGET)[0]
    if ok.size == 0:
        return float(KS_LEVELS[-1]), False
    return float(KS_LEVELS[ok[0]]), True


def db_hits(a, Wr, Wc, G, c):
    return ks_stream(mf.smooth(a, "ema"), Wr, Wc, G) > c


# ------------------------------------------------------------------ select
def det_val(hit_fn, vruns):
    n_det = 0
    for x in vruns:
        hit = hit_fn(x["a"])
        n_det += (A2.eval_run(hit, x["t"], x["cp"], x["t_info"], T.PERSIST_DEFAULT)[2] is not None and
                  A2.eval_run(hit, x["t"], x["cp"], x["t_info"], T.PERSIST_DEFAULT)[2] <= LAG_CAP)
    return n_det / max(1, len(vruns))


def select(dry):
    log = R.install_guard()
    R.guard_selftest()
    traces = R.DATASETS[:2] if dry else R.DATASETS
    seeds = R.SEEDS[:1] if dry else R.SEEDS
    da_grid = DA_GRID[:3] if dry else DA_GRID
    db_grid = DB_GRID[:2] if dry else DB_GRID
    if dry:
        traces = traces[:1]
    frozen, rows = {}, []
    for ds in traces:
        for s in seeds:
            y, fd, ff = A2.val_stream(ds, s)
            t_arr, a, r, q, e = A2.streams(y, fd, ff)
            sm = mf.smooth(a, "ema")
            vruns = A2.inject_streams(y, fd, ff, A2.POSITIONS_VAL, A2.INJ_VAL, A2.val_seed, s)
            best = {}
            for name, grid in (("DA", da_grid), ("DB", db_grid)):
                cand = []
                for cfg in grid:
                    if (cfg[1] + cfg[0] if name == "DA" else cfg[2] + cfg[1] + cfg[0] - 1) > len(sm) // 2:
                        continue  # burn-in may not exceed half of the validation stream
                    if name == "DA":
                        h, feas = da_level(sm, *cfg)
                        fn = lambda a_, cfg=cfg, h=h: da_hits(a_, *cfg, h, da_floor(sm))
                    else:
                        h, feas = db_level(sm, *cfg)
                        fn = lambda a_, cfg=cfg, h=h: db_hits(a_, *cfg, h)
                    if not feas:
                        continue
                    vd = det_val(fn, vruns)
                    burn = (cfg[1] + cfg[0]) if name == "DA" else (cfg[2] + cfg[1] + cfg[0] - 1)
                    vfa = float(A2.onsets_k(fn(a)[burn:], 3).mean())
                    cand.append((vd, -vfa, cfg[0], cfg, h, vfa))
                if cand:
                    cand.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
                    vd, _, _, cfg, h, vfa = cand[0]
                    best[name] = {"cfg": list(cfg), "burn": int(burn), "level": h, "floor": da_floor(sm), "feasible": True, "val_det": vd, "val_fa": vfa, "n_feasible_cfgs": len(cand)}
                else:
                    best[name] = {"cfg": list(grid[-1]), "burn": 0, "level": float(DA_LEVELS[-1] if name == "DA" else KS_LEVELS[-1]), "floor": da_floor(sm), "feasible": False, "val_det": float("nan"), "val_fa": float("nan"), "n_feasible_cfgs": 0}
            frozen[f"{ds}|{s}"] = best
            rows.append({"dataset": ds, "seed": s, **{f"{k}_{kk}": (str(v) if isinstance(v, list) else v) for k, b in best.items() for kk, v in b.items()}})
            R.progress(f"a4 select {ds} s{s}")
    df = pd.DataFrame(rows)
    assert not any("predictions" in p for p in log), "test data touched"
    md = {k: float(np.mean([np.nan_to_num(v[k]["val_det"]) for v in frozen.values()])) for k in ("DA", "DB")}
    mf_ = {k: float(np.nanmean([v[k]["val_fa"] for v in frozen.values()])) if any(np.isfinite(v[k]["val_fa"]) for v in frozen.values()) else float("nan") for k in ("DA", "DB")}
    headline = "DA" if md["DA"] >= md["DB"] else "DB"
    out = {"stage": "select", "dry_run": dry, "headline": headline, "val_detection_mean": md, "val_fa_mean": mf_, "params": frozen, "test_files_read": 0}
    print(json.dumps({k: v for k, v in out.items() if k != "params"}, indent=1))
    if dry:
        R.enable_pseudo()
        test_stage(frozen_in=out, pseudo=True, traces=traces, seeds=seeds)
        n_read = len([p for p in log if "predictions" in p])
        assert n_read == 0, "test data touched"
        print("dry-run ok; test files read:", n_read)
        return
    df.to_csv(HERE / "a4_select_rows.csv", index=False)
    p = HERE / "frozen_a4.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    (HERE / "frozen_a4.sha256").write_text(R.sha256_file(p) + "\n")


# ------------------------------------------------------------------ test
def all_hits(P1p, P4, a, r, q, e):
    """Hits of the five detectors for one stream. P1p: round-1 params (M0, M1, M5b); P4: round-2 params."""
    h = A2.detector_hits(P1p, a, r, q, e)
    out = {"M0": h["M0"], "M1": h["M1"], "M5b": h["M5b"]}
    p = P4["DA"]
    out["DA"] = da_hits(a, p["cfg"][0], p["cfg"][1], p["cfg"][2], p["level"], p["floor"])
    p = P4["DB"]
    out["DB"] = db_hits(a, p["cfg"][0], p["cfg"][1], p["cfg"][2], p["level"])
    return out


def test_stage(frozen_in=None, pseudo=False, traces=None, seeds=None):
    if not pseudo:
        allow_test2()
        p = HERE / "frozen_a4.json"
        assert R.sha256_file(p) == (HERE / "frozen_a4.sha256").read_text().strip(), "frozen_a4.json changed"
        frozen_in = json.loads(p.read_text())
        fr1 = json.loads((P1 / "frozen_a2.json").read_text())
        assert R.sha256_file(P1 / "frozen_a2.json") == (P1 / "frozen_a2.sha256").read_text().strip(), "frozen_a2.json changed"
    traces = traces or R.DATASETS
    seeds = seeds or R.SEEDS
    inj = 2 if pseudo else A2.INJ_TEST
    rows, nf_rows = [], []
    for ds in traces:
        for s in seeds:
            P4 = frozen_in["params"][f"{ds}|{s}"]
            if pseudo:
                yv, fdv, ffv = A2.val_stream(ds, s)
                _, av, rv, qv, ev = A2.streams(yv, fdv, ffv)
                P1p = A2.build_params(av, rv, qv, ev, True)
                P1p["M4"] = {"k": 3, "h": P1p["M1"]["h"]}
            else:
                P1p = fr1["params"][f"{ds}|{s}"]
            y, fd, ff = A2.test_stream(ds, s, pseudo)
            t_arr, a, r, q, e = A2.streams(y, fd, ff)
            nof = all_hits(P1p, P4, a, r, q, e)
            pers = {d: (1 if d == "M5b" else 3) for d in DETS}
            burns = {d: 0 for d in DETS}
            burns["DA"], burns["DB"] = P4["DA"]["burn"], P4["DB"]["burn"]
            for d in DETS:
                nf_rows.append({"dataset": ds, "seed": s, "detector": d, "fa": float(A2.onsets_k(nof[d][burns[d]:], pers[d]).mean()), "n": len(t_arr) - burns[d]})
            cps = A2.change_points(t_arr, len(y), A2.POSITIONS_TEST)
            for pi, cp in enumerate(cps):
                t_info = cp + A2.DELAY + 1
                for d in DETS:
                    _, pia, lag = A2.eval_run(nof[d], t_arr, cp, t_info, pers[d])
                    rows.append({"dataset": ds, "seed": s, "detector": d, "pos": A2.POSITIONS_TEST[pi], "inj": -1, "kind": "ctl", "detected": lag is not None and lag <= LAG_CAP, "lag": lag, "pre_info": pia})
            sigma = float(np.std(ff - y)) + 1e-9
            far = [max(range(3), key=lambda k: abs(cps[k] - cps[pi])) for pi in range(3)]
            for pi, cp in enumerate(cps):
                t_info = cp + A2.DELAY + 1
                wc = cps[far[pi]]
                for r_ in range(inj):
                    rng = np.random.default_rng(A2.test_seed(pi, r_, s))
                    ffs = mf.inject(ff, y, cp, rng, sigma)
                    t1, a1, r1, q1, e1 = A2.streams(y, fd, ffs)
                    hits = all_hits(P1p, P4, a1, r1, q1, e1)
                    for d in DETS:
                        _, pia, lag = A2.eval_run(hits[d], t1, cp, t_info, pers[d])
                        rows.append({"dataset": ds, "seed": s, "detector": d, "pos": A2.POSITIONS_TEST[pi], "inj": r_, "kind": "inj", "detected": lag is not None and lag <= LAG_CAP, "lag": lag, "pre_info": pia})
                        # permuted-injection-time control: the same alarms scored against the clock of the farthest other change point
                        _, _, lagw = A2.eval_run(hits[d], t1, wc, wc + A2.DELAY + 1, pers[d])
                        rows.append({"dataset": ds, "seed": s, "detector": d, "pos": A2.POSITIONS_TEST[pi], "inj": r_, "kind": "perm", "detected": lagw is not None and lagw <= LAG_CAP, "lag": lagw, "pre_info": False, "clock_separable": bool(abs(wc - cp) >= 30)})
        R.progress(f"a4 test {ds}")
    runs, nf = pd.DataFrame(rows), pd.DataFrame(nf_rows)
    outd = (HERE / "dryrun_out") if pseudo else HERE
    outd.mkdir(exist_ok=True)
    runs.to_csv(outd / "a4_test_runs.csv", index=False)
    nf.to_csv(outd / "a4_test_nofault.csv", index=False)
    summ = summarize(runs, nf, frozen_in["headline"], pseudo)
    (outd / "a4_test_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("table", "mcnemar")}, indent=1, default=float))


def summarize(runs, nf, headline, pseudo):
    tab = []
    for d in DETS:
        for ds in sorted(runs.dataset.unique()):
            g = lambda k: runs[(runs.detector == d) & (runs.dataset == ds) & (runs.kind == k)]
            ri, rc, rp = g("inj"), g("ctl"), g("perm")
            fa = float(nf[(nf.detector == d) & (nf.dataset == ds)].fa.mean())
            x1, n1, x2, n2, x3, n3 = int(ri.detected.sum()), len(ri), int(rc.detected.sum()), len(rc), int(rp.detected.sum()), len(rp)
            ss, lo, hi = T.newcombe_diff(x1, n1, x2, n2)
            pm, plo, phi = T.newcombe_diff(x1, n1, x3, n3)
            lags = ri[ri.detected].lag.astype(float)
            tab.append({"detector": d, "dataset": ds, "fa": fa, "det_rate": x1 / n1, "ctl_rate": x2 / n2, "shift_specific": ss, "ss_lo": lo, "ss_hi": hi,
                        "perm_rate": x3 / n3, "above_perm": pm, "above_perm_lo": plo, "clock_separable_share": float(rp.clock_separable.mean()),
                        "lag_median": float(lags.median()) if len(lags) else None})
    t = pd.DataFrame(tab)
    h = t[t.detector == headline]
    c1 = int((h.fa <= 0.05).sum())
    c2 = int((h.ss_lo > 0).sum())
    c3 = int(((h.above_perm_lo > 0) & (h.ss_lo > 0)).sum())
    ok = c1 >= 4 and c2 >= 4 and c3 >= min(c2, 4) and c2 >= 4
    return {"headline": headline, "cond1_fa_le5_traces": c1, "cond2_shift_specific_traces": c2, "cond3_above_permuted_clock_traces": c3,
            "success": bool(ok), "pseudo": pseudo, "table": tab}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    select(a.dry_run) if a.stage == "select" else test_stage()
