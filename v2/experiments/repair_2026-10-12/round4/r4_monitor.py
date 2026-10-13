"""Round 4 monitor: replication of the reference-free detector of round 3 on new drifting traces, with a clean-segment placebo control (REPAIR_PREREG_4.md section 5).

Detectors and calibration are those of round3/r3_monitor.py (RF, ST, AD; same grids, same validation budget, burn-in of 100 decisions counted in every false-alarm denominator).
New: the injection change points sit after a long clean segment and the random control is a PLACEBO clock, not a wrong-clock scoring.
  injection change points  cp in {420, 420 + 0.5 (n - 110 - 420), n - 110}          (window indices; needs n >= 690)
  matched no-injection control  the clean stream scored at the three cps (shift-specific detection, lag cap 150, as before)
  placebo clocks  P_j = 190 + 30 j, j = 0..4, in the clean prefix: the alarms of the clean stream scored as if a change point were at P_j with lag cap 100;
                  condition 3 compares detection of the injected runs within lag 100 with the placebo detection rate, Newcombe 95 % interval
Traces: A-half CESNET / Abilene series with train-to-validation shift >= 0.3 sd and >= 690 test windows (pool4_manifest.json monitor_traces), h = 6.
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
for p in (HERE, HERE.parent / "round3", HERE.parent / "paper_a", HERE.parent / "paper_a" / "round2", HERE.parent / "fresh_cells"):
    sys.path.insert(0, str(p))
import rcommon as R  # noqa: E402
import stats_tools as T  # noqa: E402
import a2 as A2  # noqa: E402
import monitor_far as mf  # noqa: E402
import r3_monitor as M3  # noqa: E402
import r4_stages as S4  # noqa: E402

CP_LO = 420
PLACEBO = [190 + 30 * j for j in range(5)]
CAP_INJ, CAP_PLC = 150, 100
DETS = M3.DETS


def injection_cps(n):
    hi = n - 110
    return [int(round(CP_LO + f * (hi - CP_LO))) for f in (0.0, 0.5, 1.0)]


def run_traces(names, test_dir, frozen, inj):
    rows, nf = [], []
    pers = {"ST": 3, "AD": 1, "RF": 3}
    for name in names:
        P = frozen["params"][name]
        y, fE, fB = M3.streams_from(M3.load_h6(test_dir, name))
        n = len(y)
        assert n >= 690, f"{name}: stream too short for the clean-segment design"
        t, a, r, q, e = A2.streams(y, fE, fB)
        nof = M3.all_hits(P, q, e)
        for d in DETS:
            nf.append({"series": name, "detector": d, "fa": float(A2.onsets_k(nof[d], pers[d]).sum() / len(q)), "n": len(q)})
        cps = injection_cps(n)
        for d in DETS:
            for pj in PLACEBO:
                lag = A2.eval_run(nof[d], t, pj, pj + A2.DELAY + 1, pers[d])[2]
                rows.append({"series": name, "detector": d, "kind": "plc", "pos": pj, "inj": -1, "lag": lag, "detected": lag is not None and lag <= CAP_PLC})
            for pi, cp in enumerate(cps):
                lag = A2.eval_run(nof[d], t, cp, cp + A2.DELAY + 1, pers[d])[2]
                rows.append({"series": name, "detector": d, "kind": "ctl", "pos": pi, "inj": -1, "lag": lag, "detected": lag is not None and lag <= CAP_INJ})
        sigma = float(np.std(fB - y)) + 1e-9
        for pi, cp in enumerate(cps):
            for r_ in range(inj):
                rng = np.random.default_rng(A2.test_seed(pi, r_, 0))
                ffs = mf.inject(fB, y, cp, rng, sigma)
                t1, a1, r1, q1, e1 = A2.streams(y, fE, ffs)
                hits = M3.all_hits(P, q1, e1)
                for d in DETS:
                    lag = A2.eval_run(hits[d], t1, cp, cp + A2.DELAY + 1, pers[d])[2]
                    rows.append({"series": name, "detector": d, "kind": "inj", "pos": pi, "inj": r_, "lag": lag, "detected": lag is not None and lag <= CAP_INJ, "detected_cap100": lag is not None and lag <= CAP_PLC})
    return pd.DataFrame(rows), pd.DataFrame(nf)


def summarize(runs, nf, pseudo):
    tab = []
    for d in DETS:
        for s in sorted(runs.series.unique()):
            g = lambda k: runs[(runs.detector == d) & (runs.series == s) & (runs.kind == k)]
            ri, rc, rp = g("inj"), g("ctl"), g("plc")
            fa = float(nf[(nf.detector == d) & (nf.series == s)].fa.iloc[0])
            x1, n1, x2, n2 = int(ri.detected.sum()), len(ri), int(rc.detected.sum()), len(rc)
            ss, lo, hi = T.newcombe_diff(x1, n1, x2, n2)
            x1c, x3, n3 = int(ri.detected_cap100.sum()), int(rp.detected.sum()), len(rp)
            pm, plo, phi = T.newcombe_diff(x1c, n1, x3, n3)
            tab.append({"detector": d, "series": s, "fa": fa, "det_rate": x1 / n1, "ctl_rate": x2 / n2, "shift_specific": ss, "ss_lo": lo, "det_cap100": x1c / n1, "placebo_rate": x3 / n3,
                        "above_placebo": pm, "above_placebo_lo": plo})
    t = pd.DataFrame(tab)
    h = t[t.detector == "RF"]
    n = len(h)
    need = int(np.ceil(0.8 * n))
    c1, c2 = int((h.fa <= 0.05).sum()), int((h.ss_lo > 0).sum())
    c3 = int(((h.above_placebo_lo > 0) & (h.ss_lo > 0)).sum())
    return {"traces": n, "need": need, "cond1_fa_le5": c1, "cond2_shift_specific": c2, "cond3_above_placebo": c3, "success": bool(c1 >= need and c2 >= need and c3 >= min(c2, need)), "pseudo": pseudo, "table": tab}


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
        import r3_stages as S3
        tmp = Path(tempfile.mkdtemp(dir=str(HERE)))
        rv = F.Restricted(load_preprocessed("abilene"))
        o = S3.stage_a("abilene", 6, rv, out_dir=tmp, save_models=False, dry=True)
        np.savez(tmp / "abilene__h6.npz", **o)
        fr = M3.select(["abilene"], tmp, None, dry=True)
        runs, nf = run_traces(["abilene"], tmp, fr, inj=2)
        s = summarize(runs, nf, True)
        print(json.dumps({k: v for k, v in s.items() if k != "table"}, indent=1))
        assert not any(sp == "test" for _, sp in rv.accessed)
        shutil.rmtree(tmp)
        print("dry-run ok; test windows accessed: 0")
        return
    names = json.loads((HERE.parent / "fresh_cells" / "pool4_manifest.json").read_text())["monitor_traces"]
    if a.stage == "select":
        out = M3.select(names, S4.VAL, None, dry=True)
        out["dry_run"] = False
        p = HERE / "frozen_r4_monitor.json"
        p.write_text(json.dumps(out, indent=1, default=float))
        (HERE / "frozen_r4_monitor.sha256").write_text(R.sha256_file(p) + "\n")
        print("frozen monitor parameters for", len(names), "traces")
        return
    if not (HERE / "PREREG_4_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test":
        raise SystemExit("test stage needs PREREG_4_COMMITTED and REPAIR_STAGE=test")
    p = HERE / "frozen_r4_monitor.json"
    assert R.sha256_file(p) == (HERE / "frozen_r4_monitor.sha256").read_text().strip(), "frozen_r4_monitor.json changed"
    runs, nf = run_traces(names, S4.TEST, json.loads(p.read_text()), inj=A2.INJ_TEST)
    runs.to_csv(HERE / "r4_monitor_runs.csv", index=False)
    nf.to_csv(HERE / "r4_monitor_nofault.csv", index=False)
    s = summarize(runs, nf, False)
    (HERE / "r4_monitor_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps({k: v for k, v in s.items() if k != "table"}, indent=1))


if __name__ == "__main__":
    main()
