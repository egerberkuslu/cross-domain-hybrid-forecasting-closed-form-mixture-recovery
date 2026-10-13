"""Round 2 phase 2: predict the TEST windows with the saved expert models (trained on TRAIN windows only in the val stage).
Needs PREREG_2_COMMITTED and REPAIR_STAGE=test. --dry-run reloads the saved models of one cell, predicts its VALIDATION windows
and checks that the forecasts equal the stored validation-stage forecasts (so that test forecasts come from the same models);
no test window is accessed in the dry run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import rcommon as R  # noqa: E402
import experts as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from src.preprocessing import load_preprocessed
    R.install_guard()
    R.guard_selftest()
    if a.dry_run:
        ds, h = "abilene", 6
        rv = X.Restricted(load_preprocessed(ds))
        w = rv.windows(h, "val")
        models = joblib.load(X.MODELS / f"{ds}__h{h}.joblib")
        bq = X.bolt_quantiles(w.X.astype(np.float32), h, a.device)
        out = X.predict_cell(models, w.X, w.target_times, h, bq)
        ref = np.load(X.VALX / f"{ds}__h{h}.npz")
        diffs = {k: float(np.max(np.abs(out[k] - ref[k]))) for k in out if k in ref.files}
        print("max abs diff vs validation-stage forecasts:", diffs)
        assert max(diffs.values()) < 1e-4, "saved models do not reproduce the validation forecasts"
        assert not any(s == "test" for _, s in rv.accessed)
        print("dry-run ok; test windows accessed: 0")
        return
    if not (HERE / "PREREG_2_COMMITTED").exists():
        raise R.TestAccessError("PREREG_2_COMMITTED missing")
    R.allow_test()
    (HERE / "test_experts").mkdir(exist_ok=True)
    cache = {}
    for ds, h in R.CELLS:
        if ds not in cache:
            cache[ds] = X.Restricted(load_preprocessed(ds), allow_test=True)
        rv = cache[ds]
        w = rv.windows(h, "test")
        models = joblib.load(X.MODELS / f"{ds}__h{h}.joblib")
        bq = X.bolt_quantiles(w.X.astype(np.float32), h, a.device)
        out = X.predict_cell(models, w.X, w.target_times, h, bq)
        st = R.load_test(ds, h, 42)
        n = min(len(st["y"]), len(w.y))
        d_y = float(np.max(np.abs(st["y"][:n] - w.y[:n])))
        db = np.abs(st["glob"][:n] - bq[:n, 4, :])
        d_b, d_b_med = float(db.max()), float(np.median(db))
        print(json.dumps({"cell": f"{ds}_h{h}", "n_test": int(n), "max_dy": d_y, "max_dbolt": d_b, "median_dbolt": d_b_med}), flush=True)
        assert d_y < 1e-5 and d_b < 0.05 * float(np.std(w.y)), "alignment with the stored test forecasts failed"
        # Deviation recorded in RUN_NOTES.md: the stored test Bolt forecast comes from an earlier GPU run and differs from the recomputed one by bf16 noise
        # (validation reproduces to 0.0). The residual experts therefore add their learned residual to the STORED Bolt forecast, which is exactly what
        # they do on validation, so that every arm of the comparison shares one Bolt forecast.
        for k in ("EB", "EB0"):
            if k in out:
                out[k] = st["glob"][:n].astype(np.float64) + (out[k][:n] - bq[:n, 4, :])
        np.savez_compressed(HERE / "test_experts" / f"{ds}__h{h}.npz", y_test=w.y.astype(np.float32), bolt_test=bq.astype(np.float32), **{k: v.astype(np.float32) for k, v in out.items()})
        R.progress(f"experts test {ds} h{h}")


if __name__ == "__main__":
    main()
