"""Round 5 monitor: the round 4 design (reference-free detector, placebo clocks) on the stratum-F drifting traces (REPAIR_PREREG_5.md section 5).

Detectors, calibration, grids, validation budget, burn-in, injection, matched no-injection control, placebo clocks and the success rule are those of round4/r4_monitor.py, imported unchanged.
Traces: the CESNET and Abilene series of partition_F.json F_registered whose train-to-validation shift is at least 0.3 train sd and with at least 690 test windows at h = 6
(pool5_manifest.json monitor_traces). need = ceil(0.8 n) for each of the three conditions. Added, descriptive only: pass rates per condition with Wilson intervals, per source.
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
for p in (HERE, HERE.parent / "round4", HERE.parent / "round3", HERE.parent / "paper_a", HERE.parent / "paper_a" / "round2", HERE.parent / "fresh_cells"):
    sys.path.insert(0, str(p))
import rcommon as R  # noqa: E402
import r3_monitor as M3  # noqa: E402
import r4_monitor as M4  # noqa: E402
import r5_stages as S5  # noqa: E402


def wilson(x, n, z=1.96):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - w), float(c + w)]


def descriptive(s):
    h = pd.DataFrame(s["table"])
    h = h[h.detector == "RF"].copy()
    h["src"] = np.where(h.series.str.startswith("abilene"), "abilene", "cesnet")
    h["c1"], h["c2"] = h.fa <= 0.05, h.ss_lo > 0
    h["c3"] = (h.above_placebo_lo > 0) & (h.ss_lo > 0)
    out = {}
    for name, g in [("all", h)] + [(k, v) for k, v in h.groupby("src")]:
        out[name] = {c: {"pass": int(g[c].sum()), "n": int(len(g)), "wilson95": wilson(int(g[c].sum()), len(g))} for c in ("c1", "c2", "c3")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    R.install_guard()
    R.guard_selftest()
    if a.dry_run:
        import shutil
        import tempfile
        from src.preprocessing import load_preprocessed
        import fresh_cells as F
        import r3_stages as S3
        tmp = Path(tempfile.mkdtemp(dir=str(HERE)))
        rv = F.Restricted(load_preprocessed("abilene"))
        o = S3.stage_a("abilene", 6, rv, out_dir=tmp, save_models=False, dry=True)
        np.savez(tmp / "abilene__h6.npz", **o)
        fr = M3.select(["abilene"], tmp, None, dry=True)
        runs, nf = M4.run_traces(["abilene"], tmp, fr, inj=2)
        s = M4.summarize(runs, nf, True)
        d = descriptive(s)
        print(json.dumps({k: v for k, v in s.items() if k != "table"}, indent=1))
        print(json.dumps(d["all"]))
        assert not any(sp == "test" for _, sp in rv.accessed)
        shutil.rmtree(tmp)
        print("dry-run ok; test windows accessed: 0")
        return
    names = json.loads((HERE.parent / "fresh_cells" / "pool5_manifest.json").read_text())["monitor_traces"]
    if a.stage == "select":
        out = M3.select(names, S5.VAL, None, dry=True)
        out["dry_run"] = False
        p = HERE / "frozen_r5_monitor.json"
        p.write_text(json.dumps(out, indent=1, default=float))
        (HERE / "frozen_r5_monitor.sha256").write_text(R.sha256_file(p) + "\n")
        print("frozen monitor parameters for", len(names), "traces")
        return
    if not (HERE / "PREREG_5_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test":
        raise SystemExit("test stage needs PREREG_5_COMMITTED and REPAIR_STAGE=test")
    p = HERE / "frozen_r5_monitor.json"
    assert R.sha256_file(p) == (HERE / "frozen_r5_monitor.sha256").read_text().strip(), "frozen_r5_monitor.json changed"
    runs, nf = M4.run_traces(names, S5.TEST, json.loads(p.read_text()), inj=M4.A2.INJ_TEST)
    runs.to_csv(HERE / "r5_monitor_runs.csv", index=False)
    nf.to_csv(HERE / "r5_monitor_nofault.csv", index=False)
    s = M4.summarize(runs, nf, False)
    s["descriptive"] = descriptive(s)
    (HERE / "r5_monitor_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps({k: v for k, v in s.items() if k != "table"}, indent=1, default=float))


if __name__ == "__main__":
    main()
