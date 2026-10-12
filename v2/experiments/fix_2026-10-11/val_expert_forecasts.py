"""Step 0 (optional but recommended): validation-split expert forecasts.

The CHA-S runs persisted TEST forecasts only. Three audit fixes need the two
experts' forecasts on the VALIDATION windows (degradation sweep fitted on
validation, stacked OLS fitted on validation, monitor thresholds designed on
validation streams). This script recomputes them WITHOUT retraining:

  * decomposition expert: STL + Theta + seasonal-naive are deterministic given
    the lookback; the residual LSTM weights are restored from the saved
    checkpoint (`lstm_residual_state_dict`). The darts BlockRNN module is
    materialised by a 1-epoch fit on a short train chunk and then OVERWRITTEN
    with the saved weights, so the forecasts come from the trained model.
  * global expert: Chronos-Bolt-small zero-shot (deterministic), loaded once
    per horizon from the local Hugging Face cache.

Self-checks written to val_forecasts/_manifest_<ts>.json:
  * val grid: RMSE of a*dec + (1-a)*glob on validation for the 21 grid points
    vs the checkpoint's alpha_search_diag (must match to ~1e-4 relative);
  * test decomposition forecast vs outputs/predictions/*decomp_only* (seed 42
    by default, --check-test all for every seed);
  * Chronos-Bolt test forecast vs outputs/predictions/*chronos_bolt_zs*.
Cells that fail the val-grid check are listed; downstream scripts then fall
back to the moment-based path (validation grid from the checkpoint JSON).

Run from the repo root on CPU, after the GPU sweep:
  python3 v2/experiments/fix_2026-10-11/val_expert_forecasts.py
Existing per-cell files are skipped (resume), never overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _common import (  # noqa: E402
    DATASETS,
    HORIZONS,
    SEEDS,
    ROOT,
    VAL_DIR,
    CKPT,
    load_pred,
    load_bolt,
    rmse,
)

sys.path.insert(0, str(ROOT))


def _short_chunk(arr: np.ndarray, need: int) -> np.ndarray:
    """First finite run of length `need` (only used to materialise darts)."""
    finite = np.isfinite(arr)
    run = 0
    for i, ok in enumerate(finite):
        run = run + 1 if ok else 0
        if run >= need:
            return arr[i - need + 1 : i + 1].astype(np.float32)
    raise RuntimeError("no finite chunk long enough to materialise the LSTM")


def build_restored_v3(ds, h, seed, pp, base, device, bolt_cache):
    import torch
    from src.models import build_model
    from src.models.deep_darts import LSTMForecaster

    ck_path = CKPT / f"{ds}__cha_hybrid_v3__h{h}__s{seed}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck.get("lstm_residual_state_dict")
    if sd is None:
        raise RuntimeError(f"{ck_path.name}: no lstm_residual_state_dict")

    m = build_model(
        "cha_hybrid_v3", horizon=h, hparams=dict(base), seed=seed, device=device
    )
    wins = pp.windows[h]
    train_arr = pp.split_scaled.train["value"].to_numpy(np.float32)

    # materialise the residual LSTM module (1 epoch on a short chunk), then
    # overwrite every weight with the trained checkpoint
    hp = dict(m.lstm_hparams)
    hp["n_epochs"] = 1
    lstm = LSTMForecaster(horizon=h, hparams=hp, seed=seed, device=device)
    L = int(hp.get("input_chunk_length", 168))
    chunk = _short_chunk(train_arr, L + h + 64)
    lstm.fit(wins["train"], wins["val"], train_series=chunk, val_series=chunk)
    lstm._darts_model.model.load_state_dict(sd)
    lstm._darts_model.model.eval()
    m.residual_model = lstm

    if h not in bolt_cache:
        g = m._build_global_expert()
        val_arr = pp.split_scaled.val["value"].to_numpy(np.float32)
        g.fit(wins["train"], wins["val"], train_series=train_arr, val_series=val_arr)
        bolt_cache[h] = g
    m.global_model = bolt_cache[h]
    m.alpha_h = float(ck.get("alpha_h", 0.0))
    m._fitted = True
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check-test", choices=["none", "42", "all"], default="42")
    args = ap.parse_args()

    import torch
    from src.preprocessing import load_preprocessed
    from src.training import MODEL_CONFIGS

    torch.set_num_threads(
        max(1, int(__import__("os").environ.get("OMP_NUM_THREADS", "4")))
    )
    base = next(b for r, v, b, s in MODEL_CONFIGS if v == "cha_hybrid_v3")
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"started": time.strftime("%F %T"), "cells": []}

    for ds in args.datasets:
        pp = load_preprocessed(ds)
        bolt_cache: dict = {}
        glob_val_cache: dict = {}
        for h in args.horizons:
            if h not in pp.windows:
                continue
            wins = pp.windows[h]
            for seed in args.seeds:
                out = VAL_DIR / f"{ds}__h{h}__s{seed}.npz"
                rec = {"dataset": ds, "horizon": h, "seed": seed, "file": out.name}
                if out.exists():
                    rec["status"] = "exists_skipped"
                    manifest["cells"].append(rec)
                    continue
                t0 = time.perf_counter()
                m = None
                try:
                    m, ck = build_restored_v3(
                        ds, h, seed, pp, base, args.device, bolt_cache
                    )
                    dec_val = m.predict_decomposition_only(wins["val"]).astype(
                        np.float64
                    )
                    if h not in glob_val_cache:
                        glob_val_cache[h] = m.predict_global_only(wins["val"]).astype(
                            np.float64
                        )
                    glob_val = glob_val_cache[h]
                    y_val = np.asarray(wins["val"].y, dtype=np.float64)

                    # check 1: validation grid reproduced
                    grid = ck.get("alpha_search_diag") or []
                    errs = []
                    for r in grid:
                        a = float(r["alpha"])
                        e = rmse(y_val, a * dec_val + (1 - a) * glob_val)
                        errs.append(
                            abs(e - float(r["val_rmse"]))
                            / max(float(r["val_rmse"]), 1e-12)
                        )
                    rec["val_grid_max_rel_err"] = float(max(errs)) if errs else None
                    rec["val_grid_ok"] = bool(errs) and max(errs) < 1e-3

                    # check 2/3: test reproduction
                    if args.check_test == "all" or (
                        args.check_test == "42" and seed == 42
                    ):
                        dec_test = m.predict_decomposition_only(wins["test"])
                        saved = load_pred(ds, "cha_hybrid_v3_decomp_only", h, seed)
                        if saved is not None:
                            n = min(saved[1].shape[0], dec_test.shape[0])
                            rec["test_dec_max_abs_diff"] = float(
                                np.max(np.abs(saved[1][:n] - dec_test[:n]))
                            )
                        if seed == args.seeds[0]:
                            g_test = m.predict_global_only(wins["test"])
                            sb = load_bolt(ds, h)
                            if sb is not None:
                                n = min(sb[1].shape[0], g_test.shape[0])
                                rec["test_bolt_max_abs_diff"] = float(
                                    np.max(np.abs(sb[1][:n] - g_test[:n]))
                                )
                    np.savez_compressed(
                        out,
                        y_val=y_val.astype(np.float32),
                        dec_val=dec_val.astype(np.float32),
                        glob_val=glob_val.astype(np.float32),
                        alpha_grid=np.float32(m.alpha_h),
                        horizon=np.int32(h),
                        seed=np.int32(seed),
                    )
                    rec["status"] = "ok"
                    rec["n_val"] = int(y_val.shape[0])
                except Exception as e:  # keep going, record the failure
                    rec["status"] = f"fail: {type(e).__name__}: {e}"
                rec["seconds"] = round(time.perf_counter() - t0, 1)
                print(json.dumps(rec))
                manifest["cells"].append(rec)
                m = None
    manifest["finished"] = time.strftime("%F %T")
    bad = [
        c
        for c in manifest["cells"]
        if c.get("status") == "ok" and not c.get("val_grid_ok")
    ]
    manifest["n_ok"] = sum(c.get("status") == "ok" for c in manifest["cells"])
    manifest["n_val_grid_fail"] = len(bad)
    ts = time.strftime("%Y%m%d-%H%M%S")
    (VAL_DIR / f"_manifest_{ts}.json").write_text(json.dumps(manifest, indent=2))
    print(f"ok={manifest['n_ok']}  val-grid failures={len(bad)}")


if __name__ == "__main__":
    main()
