"""Round 7 stages A and B for the pool-7 monitor traces (REPAIR_PREREG_7.md). Same functions as rounds 3 and 6 (r3_stages), h = 6 only, other cells and folders.

Cells: the CESNET-TimeSeries24 subnet / IP-address traces (primary pool P7 plus the ordered reserve of the degenerate-expert rule) and the GEANT G7 descriptive stratum
(pool7_manifest.json), all at h = 6.
  stage A  train-only E-Q / E-Q0, validation forecasts -> r7_val/, models -> r7_models/       (train and val windows only)
  stage B  needs PREREG_7_COMMITTED + REPAIR_STAGE=test: refit on train + validation, test forecasts -> r7_test/
  --dry-run  non-fresh Abilene aggregate stage A/B on a quarter of train (accuracy printed for that non-fresh cell only), then a shape-only check on a pool-7 series
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

VAL, TEST, MODELS = HERE / "r7_val", HERE / "r7_test", HERE / "r7_models"
S3.MODELS = MODELS


def all_names():
    m = json.loads((HERE / "pool7_manifest.json").read_text())
    return m["monitor_order"]["sub"] + m["monitor_order"]["ip"] + m["G7_traces"]


def stage_a_queue():
    """Cells for stage A: written by `r7_monitor.py --stage degen` (primary cells first, replacements of dropped traces on demand)."""
    p = HERE / "stageA_queue.json"
    return json.loads(p.read_text())["needed"] if p.exists() else []


def stage_b_cells():
    """Cells for stage B: the frozen trace list (frozen_r7.json) only."""
    return json.loads((HERE / "frozen_r7.json").read_text())["traces"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cells", nargs="*", default=None)
    a = ap.parse_args()
    if a.dry_run:
        from src.preprocessing import load_preprocessed

        rv = F.Restricted(load_preprocessed("abilene"))
        o = S3.stage_a(
            "abilene", 6, rv, out_dir=HERE / "dry_tmp", save_models=False, dry=True
        )
        rm = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
        print(
            "non-fresh abilene h6 stage A: rmse bolt %.4f, eq %.4f"
            % (rm(o["bolt"], o["y"]), rm(o["eq"], o["y"]))
        )
        o2 = S3.stage_b("abilene", 6, rv, pseudo=True)
        print(
            "non-fresh abilene h6 pseudo stage B: rmse eq %.4f" % rm(o2["eq"], o2["y"])
        )
        assert not any(s == "test" for _, s in rv.accessed)
        name = all_names()[0]
        rvf = F.Restricted(F.load_fresh(name))
        w = rvf.windows(6, "val")
        f = Q.X.features(w.X[:50].astype(np.float64), w.target_times[:50], 6)
        print(
            "pool-7 structure check",
            name,
            w.X.shape,
            f.shape,
            "no model fit, no accuracy",
        )
        try:
            rvf.windows(6, "test")
            raise AssertionError("Restricted view did not raise on a test window")
        except PermissionError:
            print("Restricted view raises on test windows: ok")
        assert not any(s == "test" for _, s in rvf.accessed)
        print("cells known to the pool:", len(all_names()))
        print("dry-run ok; test windows accessed: 0")
        return
    if a.stage == "B" and (
        not (HERE / "PREREG_7_COMMITTED").exists()
        or os.environ.get("REPAIR_STAGE") != "test"
    ):
        raise SystemExit("stage B needs PREREG_7_COMMITTED and REPAIR_STAGE=test")
    cells = (
        a.cells if a.cells else (stage_a_queue() if a.stage == "A" else stage_b_cells())
    )
    known = set(all_names())
    assert all(c in known for c in cells), "unknown cell"
    cache = {}
    for name in cells:
        if name not in cache:
            cache[name] = F.Restricted(F.load_fresh(name), allow_test=(a.stage == "B"))
        if a.stage == "A":
            S3.stage_a(name, 6, cache[name], out_dir=VAL)
            assert not any(
                s == "test" for _, s in cache[name].accessed
            ), "test windows accessed in stage A"
        else:
            S3.stage_b(name, 6, cache[name], out_dir=TEST)
        print("done", a.stage, name, 6, flush=True)


if __name__ == "__main__":
    main()
