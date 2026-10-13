"""Round 3 stages A and B for fresh cells (REPAIR_PREREG_3.md sections 2 and 4).

  stage A (validation, phase 2 start): train-only E-Q and E-Q0 per cell, Bolt median and E-Q forecasts of the VALIDATION windows -> r3_val/<cell>__h<h>.npz,
                                       models -> r3_models/. Reads train and val windows only (Restricted view).
  stage B (test, after PREREG_3_COMMITTED + REPAIR_STAGE=test): refit E-Q and E-Q0 on train + validation windows, forecasts of the TEST windows -> r3_test/.
  --dry-run  runs A and B on a NON-fresh cell (the Abilene aggregate of the paper) with a pseudo test (its validation windows, refit on train only), prints
             accuracy for that non-fresh cell only, and asserts that no test windows were accessed. On a fresh cell it only checks shapes (no model fit).
No accuracy of any fresh series is computed in this file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "fresh_cells"))
import eq_recipe as Q  # noqa: E402
import fresh_cells as F  # noqa: E402

VAL, TEST, MODELS = HERE / "r3_val", HERE / "r3_test", HERE / "r3_models"


def pool_cells():
    m = json.loads((F.HERE / "pool_manifest.json").read_text())
    return [(s["name"], h) for s in m["series"] for h in F.HORIZONS]


def stage_a(name, h, rv, out_dir=VAL, save_models=True, dry=False):
    wtr, wva = rv.windows(h, "train"), rv.windows(h, "val")
    Xtr, ytr, ttr = wtr.X, wtr.y, wtr.target_times
    if dry:
        sel = np.arange(0, len(Xtr), 4)
        Xtr, ytr, ttr = Xtr[sel], ytr[sel], ttr[sel]
    eq = Q.fit_eq(Xtr, ytr, ttr, h)
    eq0 = Q.fit_eq(Xtr, ytr, ttr, h, null_seed=0)
    pe, p0 = Q.predict_eq(eq, wva.X, wva.target_times, h), Q.predict_eq(eq0, wva.X, wva.target_times, h)
    bolt = Q.bolt_median(wva.X, h)
    out = {"y": wva.y.astype(np.float32), "bolt": bolt.astype(np.float32), "eq": pe["q50"].astype(np.float32), "eq_q10": pe["q10"].astype(np.float32),
           "eq_q90": pe["q90"].astype(np.float32), "eq0": p0["q50"].astype(np.float32)}
    if save_models:
        out_dir.mkdir(exist_ok=True)
        MODELS.mkdir(exist_ok=True)
        np.savez_compressed(out_dir / f"{name}__h{h}.npz", **out)
        joblib.dump({"eq": eq, "eq0": eq0}, MODELS / f"{name}__h{h}.joblib")
    return out


def stage_b(name, h, rv, pseudo=False, out_dir=TEST):
    wtr, wva = rv.windows(h, "train"), rv.windows(h, "val")
    wte = wva if pseudo else rv.windows(h, "test")
    if pseudo:  # refit on train only so that nothing evaluated is in the training windows
        Xr, yr, tr = wtr.X, wtr.y, wtr.target_times
        sel = np.arange(0, len(Xr), 4)
        Xr, yr, tr = Xr[sel], yr[sel], tr[sel]
    else:
        import pandas as pd
        Xr, yr = np.vstack([wtr.X, wva.X]), np.vstack([wtr.y, wva.y])
        tr = wtr.target_times.append(wva.target_times)
    eq = Q.fit_eq(Xr, yr, tr, h)
    eq0 = Q.fit_eq(Xr, yr, tr, h, null_seed=0)
    pe, p0 = Q.predict_eq(eq, wte.X, wte.target_times, h), Q.predict_eq(eq0, wte.X, wte.target_times, h)
    bolt = Q.bolt_median(wte.X, h)
    out = {"y": wte.y.astype(np.float32), "bolt": bolt.astype(np.float32), "eq": pe["q50"].astype(np.float32), "eq_q10": pe["q10"].astype(np.float32),
           "eq_q90": pe["q90"].astype(np.float32), "eq0": p0["q50"].astype(np.float32)}
    if not pseudo:
        out_dir.mkdir(exist_ok=True)
        np.savez_compressed(out_dir / f"{name}__h{h}.npz", **out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cells", nargs="*", default=None, help="restrict to cell names like cesnet_inst0")
    a = ap.parse_args()
    if a.dry_run:
        from src.preprocessing import load_preprocessed
        rv = F.Restricted(load_preprocessed("abilene"))
        h = 6
        o = stage_a("abilene", h, rv, save_models=False, dry=True)
        rm = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
        print("non-fresh abilene h6 stage A (quarter of train): rmse bolt %.4f, eq %.4f, eq0 %.4f" % (rm(o["bolt"], o["y"]), rm(o["eq"], o["y"]), rm(o["eq0"], o["y"])))
        o2 = stage_b("abilene", h, rv, pseudo=True)
        print("non-fresh abilene h6 pseudo stage B: rmse eq %.4f" % rm(o2["eq"], o2["y"]))
        assert not any(s == "test" for _, s in rv.accessed), "test windows accessed"
        # fresh-cell structure check, no model fit and no accuracy
        pp = F.load_fresh("cesnet_inst0")
        rvf = F.Restricted(pp)
        w = rvf.windows(6, "val")
        feats = Q.X.features(w.X[:50].astype(np.float64), w.target_times[:50], 6)
        print("fresh cell structure check: val windows", w.X.shape, "feature rows", feats.shape, "no model fit, no accuracy")
        assert not any(s == "test" for _, s in rvf.accessed)
        print("dry-run ok; test windows accessed: 0")
        return
    from src.preprocessing import load_preprocessed
    cells = pool_cells()
    if a.cells:
        cells = [c for c in cells if c[0] in a.cells]
    if a.stage == "B":
        if not (HERE / "PREREG_3_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test":
            raise SystemExit("stage B needs PREREG_3_COMMITTED and REPAIR_STAGE=test")
    cache = {}
    for name, h in cells:
        if name not in cache:
            cache[name] = F.Restricted(F.load_fresh(name), allow_test=(a.stage == "B"))
        (stage_a if a.stage == "A" else stage_b)(name, h, cache[name])
        print("done", a.stage, name, h, flush=True)
        if a.stage == "A":
            assert not any(s == "test" for _, s in cache[name].accessed), "test windows accessed in stage A"


if __name__ == "__main__":
    main()
