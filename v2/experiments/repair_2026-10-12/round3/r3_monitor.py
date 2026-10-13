"""Round 3 monitor: reference-free detector on the expert error ratio (REPAIR_PREREG_3.md section 5).

q_t = log(MSE_Bolt / MSE_E-Q) over the last 60 labelled windows (label delay h), horizon-averaged forecasts, as a2.streams with the E-Q forecast in the place of
the decomposition path. Detectors, all with the SAME validation budget:
  RF  reference-free CUSUM: z_t = (q_t - m_t) / max(sd_t, floor), (m_t, v_t) exponentially weighted (decay 0.99) from q_{t-G}, G in {30, 60}; CUSUM allowance K in {0.5, 1, 2};
      level from a block bootstrap of the validation q stream with the same rolling machinery (90th percentile of the onset share <= 5 %); burn-in of 100 decisions
      (alarms suppressed, decisions COUNTED in every false-alarm denominator)
  ST  static CUSUM on q with the validation mean and sd, level by the same bootstrap rule (the round 1 M2 construction)
  AD  ADWIN on the delayed Bolt squared error stream (round 1 M5b construction)
Traces: fresh series whose train-to-validation level shift is at least 0.3 train standard deviations (data-only rule), h = 6, test streams of the fresh pool.
  python3 r3_monitor.py --stage select [--dry-run]  |  REPAIR_STAGE=test python3 r3_monitor.py --stage test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / "paper_a"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(P1))
sys.path.insert(0, str(HERE.parent / "fresh_cells"))
sys.path.insert(0, str(P1 / "round2"))
import rcommon as R  # noqa: E402
import stats_tools as T  # noqa: E402
import a2 as A2  # noqa: E402
import monitor_far as mf  # noqa: E402
import r3_stages as S  # noqa: E402

H = 6
BURN = 100
DECAY = 0.99
RF_GRID = [(G, K) for G in (30, 60) for K in (0.5, 1.0, 2.0)]
LAG_CAP = A2.LAG_CAP
DRIFT_SD = 0.3
DETS = ("ST", "AD", "RF")


def drifting_traces():
    import fresh_cells as F
    m = json.loads((F.HERE / "pool_manifest.json").read_text())
    out = []
    for s in m["series"]:
        sp = json.loads((F.HERE / "splits" / s["name"] / "splits.json").read_text())
        ser = pd.read_parquet(F.HERE / "splits" / s["name"] / "series_native.parquet")["value"]
        tr = ser[(ser.index >= sp["train"]["start"]) & (ser.index <= sp["train"]["end"])]
        va = ser[(ser.index >= sp["val"]["start"]) & (ser.index <= sp["val"]["end"])]
        if abs(va.mean() - tr.mean()) / tr.std() >= DRIFT_SD:
            out.append(s["name"])
    return out


def rf_z(q, G, floor):
    """Exponentially weighted reference from q delayed by G decisions; returns z (zeros before the warm start)."""
    n = len(q)
    z = np.zeros(n)
    m = float(np.mean(q[:20]))
    v = float(np.var(q[:20])) + 1e-12
    for t in range(n):
        if t - G >= 0:
            x = q[t - G]
            m = DECAY * m + (1 - DECAY) * x
            v = DECAY * v + (1 - DECAY) * (x - m) ** 2
        z[t] = (q[t] - m) / max(np.sqrt(v), floor)
    return z


def rf_hits(q, G, K, level, floor):
    z = rf_z(q, G, floor)
    S_ = T.cusum_S(z[None], K)[0]
    h = S_ > level
    h[:BURN] = False
    return h


def rf_level(qv, G, K, floor, B=60, seed=51):
    rng = np.random.default_rng(seed)
    blk = int(min(2 * (G + 60), max(60, len(qv) // 2)))
    rows = T.mbb_resample(qv, blk, B, rng)
    Z = np.stack([rf_z(r, G, floor) for r in rows])
    Z[:, :BURN] = -10.0
    rates = T.onset_rate_levels([T.cusum_S(Z, K)], T.LEVEL_GRID, T.PERSIST_DEFAULT)  # denominators = all decisions (burn-in counted)
    qr = np.quantile(rates, 0.90, axis=0)
    ok = np.where(qr <= 0.05)[0]
    return (float(T.LEVEL_GRID[ok[0]]), True) if ok.size else (float(T.LEVEL_GRID[-1]), False)


def static_params(qv):
    mu, sd = float(qv.mean()), max(float(qv.std()), 1e-3)
    lev, feas = T.null_level((qv - mu) / sd, q=0.90, seed=52)
    return {"mu": mu, "sd": sd, "level": lev, "feasible": feas}


def adwin_params(ev):
    me, se = float(ev.mean()), max(float(ev.std()), 1e-12)
    z = (ev - me) / se
    rng = np.random.default_rng(53)
    Z = T.mbb_resample(z, 60, 40, rng)
    chosen, feas = A2.ADWIN_DELTAS[0], False
    for dl in A2.ADWIN_DELTAS:
        if np.quantile([A2.adwin_alarms(r, dl).mean() for r in Z], 0.90) <= 0.05:
            chosen, feas = dl, True
    return {"mu": me, "sd": se, "delta": chosen, "feasible": feas}


def all_hits(P, q, e):
    out = {"ST": mf.cusum_hits(q, P["ST"]["mu"], P["ST"]["sd"], P["ST"]["level"], 0),
           "AD": A2.adwin_alarms((e - P["AD"]["mu"]) / P["AD"]["sd"], P["AD"]["delta"])}
    r = P["RF"]
    out["RF"] = rf_hits(q, r["G"], r["K"], r["level"], r["floor"])
    return out


def streams_from(arr, h=H):
    """arr: dict with y, bolt, eq (n, h) -> y1, fE, fB horizon averaged."""
    return arr["y"][:, :h].mean(1), arr["eq"][:, :h].mean(1), arr["bolt"][:, :h].mean(1)


def load_h6(dirp, name):
    z = np.load(dirp / f"{name}__h{H}.npz")
    return {k: z[k].astype(np.float64) for k in z.files}


def select(names, val_dir, out_json, dry=False):
    params = {}
    for name in names:
        y, fE, fB = streams_from(load_h6(val_dir, name))
        t, a, r, q, e = A2.streams(y, fE, fB)
        vruns = A2.inject_streams(y, fE, fB, A2.POSITIONS_VAL, A2.INJ_VAL, A2.val_seed, 0)
        floor = max(0.5 * float(np.std(q)), 0.05)
        best = None
        for G, K in RF_GRID:
            lev, feas = rf_level(q, G, K, floor)
            if not feas:
                continue
            det = 0
            for x in vruns:
                hit = rf_hits(x["q"], G, K, lev, floor)
                lag = A2.eval_run(hit, x["t"], x["cp"], x["t_info"], 3)[2]
                det += int(lag is not None and lag <= LAG_CAP)
            vfa = float(A2.onsets_k(rf_hits(q, G, K, lev, floor), 3).sum() / len(q))
            key = (det / max(1, len(vruns)), -vfa, G)
            if best is None or key > best[0]:
                best = (key, G, K, lev)
        if best is None:
            rf = {"G": 60, "K": 2.0, "level": float(T.LEVEL_GRID[-1]), "floor": floor, "feasible": False, "val_det": float("nan")}
        else:
            rf = {"G": best[1], "K": best[2], "level": best[3], "floor": floor, "feasible": True, "val_det": best[0][0]}
        params[name] = {"RF": rf, "ST": static_params(q), "AD": adwin_params(e)}
    out = {"stage": "select", "dry_run": dry, "traces": names, "params": params}
    if not dry:
        Path(out_json).write_text(json.dumps(out, indent=1, default=float))
        (HERE / "frozen_r3_monitor.sha256").write_text(R.sha256_file(out_json) + "\n")
    return out


def run_traces(names, test_dir, frozen, inj, pseudo=False):
    rows, nf = [], []
    for name in names:
        P = frozen["params"][name]
        y, fE, fB = streams_from(load_h6(test_dir, name))
        t, a, r, q, e = A2.streams(y, fE, fB)
        nof = all_hits(P, q, e)
        pers = {"ST": 3, "AD": 1, "RF": 3}
        for d in DETS:  # FA over ALL decisions, burn-in included
            nf.append({"series": name, "detector": d, "fa": float(A2.onsets_k(nof[d], pers[d]).sum() / len(q)), "n": len(q)})
        cps = A2.change_points(t, len(y), A2.POSITIONS_TEST)
        for pi, cp in enumerate(cps):
            for d in DETS:
                lag = A2.eval_run(nof[d], t, cp, cp + A2.DELAY + 1, pers[d])[2]
                rows.append({"series": name, "detector": d, "pos": A2.POSITIONS_TEST[pi], "inj": -1, "kind": "ctl", "detected": lag is not None and lag <= LAG_CAP, "lag": lag})
        sigma = float(np.std(fB - y)) + 1e-9
        far = [max(range(3), key=lambda k: abs(cps[k] - cps[pi])) for pi in range(3)]
        for pi, cp in enumerate(cps):
            wc = cps[far[pi]]
            for r_ in range(inj):
                rng = np.random.default_rng(A2.test_seed(pi, r_, 0))
                ffs = mf.inject(fB, y, cp, rng, sigma)
                t1, a1, r1, q1, e1 = A2.streams(y, fE, ffs)
                hits = all_hits(P, q1, e1)
                for d in DETS:
                    lag = A2.eval_run(hits[d], t1, cp, cp + A2.DELAY + 1, pers[d])[2]
                    rows.append({"series": name, "detector": d, "pos": A2.POSITIONS_TEST[pi], "inj": r_, "kind": "inj", "detected": lag is not None and lag <= LAG_CAP, "lag": lag})
                    lw = A2.eval_run(hits[d], t1, wc, wc + A2.DELAY + 1, pers[d])[2]
                    rows.append({"series": name, "detector": d, "pos": A2.POSITIONS_TEST[pi], "inj": r_, "kind": "perm", "detected": lw is not None and lw <= LAG_CAP, "lag": lw, "clock_separable": bool(abs(wc - cp) >= 30)})
    return pd.DataFrame(rows), pd.DataFrame(nf)


def summarize(runs, nf, pseudo):
    tab = []
    for d in DETS:
        for s in sorted(runs.series.unique()):
            g = lambda k: runs[(runs.detector == d) & (runs.series == s) & (runs.kind == k)]
            ri, rc, rp = g("inj"), g("ctl"), g("perm")
            fa = float(nf[(nf.detector == d) & (nf.series == s)].fa.iloc[0])
            x1, n1, x2, n2, x3, n3 = int(ri.detected.sum()), len(ri), int(rc.detected.sum()), len(rc), int(rp.detected.sum()), len(rp)
            ss, lo, hi = T.newcombe_diff(x1, n1, x2, n2)
            pm, plo, phi = T.newcombe_diff(x1, n1, x3, n3)
            tab.append({"detector": d, "series": s, "fa": fa, "det_rate": x1 / n1, "ctl_rate": x2 / n2, "shift_specific": ss, "ss_lo": lo, "above_perm_lo": plo, "perm_rate": x3 / n3,
                        "clock_separable_share": float(rp.clock_separable.mean())})
    t = pd.DataFrame(tab)
    h = t[t.detector == "RF"]
    n = len(h)
    need = int(np.ceil(0.8 * n))
    c1, c2 = int((h.fa <= 0.05).sum()), int((h.ss_lo > 0).sum())
    c3 = int(((h.above_perm_lo > 0) & (h.ss_lo > 0)).sum())
    return {"traces": n, "need": need, "cond1_fa_le5": c1, "cond2_shift_specific": c2, "cond3_above_permuted_clock": c3,
            "success": bool(c1 >= need and c2 >= need and c3 >= min(c2, need)), "pseudo": pseudo, "table": tab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    log = R.install_guard()
    R.guard_selftest()
    if a.dry_run:
        import shutil, tempfile
        from src.preprocessing import load_preprocessed
        import fresh_cells as F
        tmp = Path(tempfile.mkdtemp(dir=str(HERE)))
        rv = F.Restricted(load_preprocessed("abilene"))
        o = S.stage_a("abilene", H, rv, out_dir=tmp, save_models=False, dry=True)
        np.savez(tmp / f"abilene__h{H}.npz", **o)
        fr = select(["abilene"], tmp, tmp / "f.json", dry=True)
        print("drifting traces of the fresh pool (rule only, no accuracy):", drifting_traces())
        runs, nf = run_traces(["abilene"], tmp, fr, inj=2, pseudo=True)
        s = summarize(runs, nf, True)
        print(json.dumps({k: v for k, v in s.items() if k != "table"}, indent=1))
        assert not any(sp == "test" for _, sp in rv.accessed)
        shutil.rmtree(tmp)
        print("dry-run ok; test windows accessed: 0")
        return
    names = drifting_traces()
    if a.stage == "select":
        out = select(names, S.VAL, HERE / "frozen_r3_monitor.json")
        print("frozen monitor parameters for", len(names), "traces")
        return
    if not (HERE / "PREREG_3_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test":
        raise SystemExit("test stage needs PREREG_3_COMMITTED and REPAIR_STAGE=test")
    p = HERE / "frozen_r3_monitor.json"
    assert R.sha256_file(p) == (HERE / "frozen_r3_monitor.sha256").read_text().strip(), "frozen_r3_monitor.json changed"
    runs, nf = run_traces(names, S.TEST, json.loads(p.read_text()), inj=A2.INJ_TEST)
    runs.to_csv(HERE / "r3_monitor_runs.csv", index=False)
    nf.to_csv(HERE / "r3_monitor_nofault.csv", index=False)
    s = summarize(runs, nf, False)
    (HERE / "r3_monitor_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps({k: v for k, v in s.items() if k != "table"}, indent=1))


if __name__ == "__main__":
    main()
