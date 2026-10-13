"""Round 4 stages A and B for the pool-4 series (REPAIR_PREREG_4.md). Same functions as round 3 (r3_stages), other cells and folders.

Cells: the 60 accuracy series at all five horizons and the monitor-only drifting series at h = 6 (pool4_manifest.json).
  stage A  train-only E-Q / E-Q0, validation forecasts -> r4_val/, models -> r4_models/       (train and val windows only)
  stage B  needs PREREG_4_COMMITTED + REPAIR_STAGE=test: refit on train + validation, test forecasts -> r4_test/
  --dry-run  non-fresh Abilene aggregate stage A/B on a quarter of train (accuracy printed for that non-fresh cell only), then a shape-only check on a pool-4 series
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

VAL, TEST, MODELS = HERE / "r4_val", HERE / "r4_test", HERE / "r4_models"
S3.MODELS = MODELS


def pool_cells():
    m = json.loads((F.HERE / "pool4_manifest.json").read_text())
    cells = []
    for s in m["series"]:
        if s["accuracy"]:
            cells += [(s["name"], h) for h in F.HORIZONS]
        elif s["name"] in m["monitor_traces"]:
            cells.append((s["name"], 6))
    return cells


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
        name = json.loads((F.HERE / "pool4_manifest.json").read_text())["series"][0]["name"]
        rvf = F.Restricted(F.load_fresh(name))
        w = rvf.windows(6, "val")
        f = Q.X.features(w.X[:50].astype(np.float64), w.target_times[:50], 6)
        print("pool-4 structure check", name, w.X.shape, f.shape, "no model fit, no accuracy")
        assert not any(s == "test" for _, s in rvf.accessed)
        print("cells in the pool:", len(pool_cells()))
        print("dry-run ok; test windows accessed: 0")
        return
    if a.stage == "B" and (not (HERE / "PREREG_4_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test"):
        raise SystemExit("stage B needs PREREG_4_COMMITTED and REPAIR_STAGE=test")
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
