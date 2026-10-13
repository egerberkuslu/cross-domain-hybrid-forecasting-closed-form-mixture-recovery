"""Round 6 stages A and B for the pool-6 monitor traces (REPAIR_PREREG_6.md). Same functions as round 3 (r3_stages), h = 6 only, other cells and folders.

Cells: the P2 traces (never allocated CESNET/Abilene series under the relaxed missingness rule, plus the drifting CESNET/Abilene series of the F+G half) and the P3 GEANT
descriptive stratum (pool6_manifest.json), all at h = 6.
  stage A  train-only E-Q / E-Q0, validation forecasts -> r6_val/, models -> r6_models/       (train and val windows only)
  stage B  needs PREREG_6_COMMITTED + REPAIR_STAGE=test: refit on train + validation, test forecasts -> r6_test/
  --dry-run  non-fresh Abilene aggregate stage A/B on a quarter of train (accuracy printed for that non-fresh cell only), then a shape-only check on a pool-6 series
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
sys.path.insert(0, str(HERE.parent / "fresh_cells"))
import r3_stages as S3  # noqa: E402
import fresh_cells as F  # noqa: E402
import eq_recipe as Q  # noqa: E402

VAL, TEST, MODELS = HERE / "r6_val", HERE / "r6_test", HERE / "r6_models"
S3.MODELS = MODELS


def monitor_names():
    m = json.loads((HERE / "pool6_manifest.json").read_text())
    return m["monitor_traces_P2"] + m["monitor_traces_P3_geant"]


def pool_cells():
    return [(n, 6) for n in monitor_names()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cells", nargs="*", default=None)
    a = ap.parse_args()
    if a.dry_run:
        from src.preprocessing import load_preprocessed
        rv = F.Restricted(load_preprocessed("abilene"))
        o = S3.stage_a("abilene", 6, rv, out_dir=HERE / "dry_tmp", save_models=False, dry=True)
        rm = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
        print("non-fresh abilene h6 stage A: rmse bolt %.4f, eq %.4f" % (rm(o["bolt"], o["y"]), rm(o["eq"], o["y"])))
        o2 = S3.stage_b("abilene", 6, rv, pseudo=True)
        print("non-fresh abilene h6 pseudo stage B: rmse eq %.4f" % rm(o2["eq"], o2["y"]))
        assert not any(s == "test" for _, s in rv.accessed)
        name = monitor_names()[0]
        rvf = F.Restricted(F.load_fresh(name))
        w = rvf.windows(6, "val")
        f = Q.X.features(w.X[:50].astype(np.float64), w.target_times[:50], 6)
        print("pool-6 structure check", name, w.X.shape, f.shape, "no model fit, no accuracy")
        try:
            rvf.windows(6, "test")
            raise AssertionError("Restricted view did not raise on a test window")
        except PermissionError:
            print("Restricted view raises on test windows: ok")
        assert not any(s == "test" for _, s in rvf.accessed)
        print("cells in the pool:", len(pool_cells()))
        print("dry-run ok; test windows accessed: 0")
        return
    if a.stage == "B" and (not (HERE / "PREREG_6_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test"):
        raise SystemExit("stage B needs PREREG_6_COMMITTED and REPAIR_STAGE=test")
    cells = pool_cells()
    if a.cells:
        cells = [c for c in cells if c[0] in a.cells]
    cache = {}
    for name, h in cells:
        if name not in cache:
            cache[name] = F.Restricted(F.load_fresh(name), allow_test=(a.stage == "B"))
        if a.stage == "A":
            S3.stage_a(name, h, cache[name], out_dir=VAL)
            assert not any(s == "test" for _, s in cache[name].accessed), "test windows accessed in stage A"
        else:
            S3.stage_b(name, h, cache[name], out_dir=TEST)
        print("done", a.stage, name, h, flush=True)


if __name__ == "__main__":
    main()
