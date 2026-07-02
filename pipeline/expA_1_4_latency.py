"""Stage A — Experiment 1.4: Inference latency on CPU vs GPU.

Reviewers Q1 routinely ask "what's the runtime cost?" — especially when a
model uses a foundation model.  This script benchmarks per-horizon inference
latency on the test split for every CHA-Hybrid v3 checkpoint AND its main
baselines (ARIMA, LSTM, Chronos-Bolt zero-shot) on both CPU and GPU, with
warm-up + median-of-N timing.

Output:
  outputs/eval_v3/tables/latency.csv

Runtime: ~10 min for a representative subset (we benchmark 1 seed per
dataset/horizon to keep it manageable).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.preprocessing import load_preprocessed
from src.models import build_model
from src.training import MODEL_CONFIGS
from src.utils.runlog import PhaseTimer

OUT_DIR = Path("outputs/eval_v3/tables")
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
HORIZONS = [1, 6, 24]
SEED = 42
N_REPEAT = 5  # repeats per cell, median reported
TARGET_BATCH = 64  # number of windows per timed inference call

# Variants to benchmark (must exist in MODEL_CONFIGS).
VARIANTS = [
    "arima",
    "lstm",
    "patchtst",
    "chronos_bolt_zs",
    "cha_hybrid_v3",
    "cha_hybrid_v4",
    "cha_hybrid_v4_fix",
]


def _build(variant: str, horizon: int, device: str):
    """`variant` is the public display name in MODEL_CONFIGS (e.g.
    ``chronos_bolt_zs``); the underlying REGISTRY key is the model class
    name (e.g. ``chronos_bolt``).  We look up both."""
    match = next(
        ((r, b) for r, v, b, s in MODEL_CONFIGS if v == variant),
        None,
    )
    if match is None:
        return None
    registry_name, base = match
    return build_model(
        registry_name, horizon=horizon, hparams=base, seed=SEED, device=device
    )


def _time_predict(model, windows, n_repeat: int) -> tuple[float, float]:
    # Warm up
    try:
        _ = model.predict(windows)
    except Exception as e:
        return float("nan"), float("nan")
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        try:
            _ = model.predict(windows)
        except Exception:
            return float("nan"), float("nan")
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), float(
        np.std(times, ddof=1) if len(times) > 1 else 0.0
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    for ds in DATASETS:
        try:
            pp = load_preprocessed(ds)
        except Exception as e:
            print(f"skip {ds}: {e}")
            continue
        for h in HORIZONS:
            wins = pp.windows.get(h)
            if wins is None:
                continue
            # Slice the test WindowSet to a fixed batch so timings are comparable
            from src.preprocessing.windowing import WindowSet

            full = wins["test"]
            test_subset = WindowSet(
                X=full.X[:TARGET_BATCH],
                y=full.y[:TARGET_BATCH],
                target_times=full.target_times[:TARGET_BATCH],
                split=full.split,
                horizon=full.horizon,
                lookback=full.lookback,
            )
            X_test = test_subset.X
            train_series = pp.split_scaled.train["value"].to_numpy("float32")
            val_series = pp.split_scaled.val["value"].to_numpy("float32")
            for variant in VARIANTS:
                for device in devices:
                    print(f"[1.4] {ds} h={h} {variant} on {device}")
                    try:
                        m = _build(variant, h, device)
                        if m is None:
                            continue
                        t0 = time.perf_counter()
                        m.fit(
                            wins["train"],
                            wins["val"],
                            train_series=train_series,
                            val_series=val_series,
                        )
                        fit_s = time.perf_counter() - t0
                    except Exception as e:
                        rows.append(
                            {
                                "dataset": ds,
                                "horizon": h,
                                "variant": variant,
                                "device": device,
                                "status": "fit_failed",
                                "error": f"{type(e).__name__}: {e}",
                            }
                        )
                        continue
                    median_s, std_s = _time_predict(m, test_subset, N_REPEAT)
                    per_window = median_s / max(len(X_test), 1)
                    rows.append(
                        {
                            "dataset": ds,
                            "horizon": h,
                            "variant": variant,
                            "device": device,
                            "n_test_batch": int(len(X_test)),
                            "fit_seconds": float(fit_s),
                            "predict_median_s": median_s,
                            "predict_std_s": std_s,
                            "predict_per_window_ms": float(per_window * 1000.0),
                            "status": "ok",
                        }
                    )
                    # free GPU memory
                    del m
                    if device == "cuda":
                        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    out = OUT_DIR / "latency.csv"
    df.to_csv(out, index=False)
    print(f"[1.4] wrote {out}  ({len(df)} rows)")
    if not df.empty:
        ok = df[df["status"] == "ok"]
        if not ok.empty:
            print(
                ok.groupby(["variant", "device"])["predict_per_window_ms"]
                .median()
                .unstack()
                .to_string()
            )


if __name__ == "__main__":
    with PhaseTimer(
        "expA_1.4_inference_latency",
        notes=f"variants={VARIANTS}, n_repeat={N_REPEAT}, batch={TARGET_BATCH}",
    ) as t:
        main()
        t.add_output("table", str(OUT_DIR / "latency.csv"))
