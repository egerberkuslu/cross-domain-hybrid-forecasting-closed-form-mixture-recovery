"""Stage A — Experiment 1.1: 95% bootstrap CI for all metrics.

Reviewers Q1 ask: "single-number metrics are not enough — give us uncertainty
bands."  This script reads every prediction ``.npz`` in
``outputs/predictions/`` and computes moving-block bootstrap (block length =
horizon) confidence intervals for RMSE / MAE / MAPE / sMAPE on the held-out
test split.

Output:
  outputs/eval_v3/tables/bootstrap_ci.csv  — one row per (ds, model, h, seed)
  outputs/eval_v3/tables/bootstrap_ci_agg.csv — averaged across seeds

The moving-block bootstrap is appropriate for serially dependent forecast
errors (Politis & Romano 1994); we use block length = horizon as a
conservative default that preserves at least one horizon-window of
dependence per block.

Runtime: ~5–10 min for the full grid.  No GPU required.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.runlog import PhaseTimer

PRED_DIR = Path("outputs/predictions")
OUT_DIR = Path("outputs/eval_v3/tables")
N_BOOT = 1000
RNG_SEED = 12345
ALPHA = 0.05  # 95 % CI


def _mape(y, p, eps=1e-8):
    return float(np.mean(np.abs((y - p) / np.maximum(np.abs(y), eps))) * 100.0)


def _smape(y, p, eps=1e-8):
    return float(
        np.mean(2.0 * np.abs(p - y) / np.maximum(np.abs(p) + np.abs(y), eps)) * 100.0
    )


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mae(y, p):
    return float(np.mean(np.abs(y - p)))


METRIC_FNS = {"rmse": _rmse, "mae": _mae, "mape": _mape, "smape": _smape}


def moving_block_bootstrap_indices(
    n: int, block_len: int, n_boot: int, rng
) -> np.ndarray:
    """Return (n_boot, n) array of resampled indices via overlapping moving blocks."""
    n_blocks = (n + block_len - 1) // block_len
    out = np.empty((n_boot, n_blocks * block_len), dtype=np.int64)
    max_start = max(n - block_len + 1, 1)
    starts = rng.integers(0, max_start, size=(n_boot, n_blocks))
    offsets = np.arange(block_len)
    idx = starts[:, :, None] + offsets[None, None, :]
    idx = idx.reshape(n_boot, -1)
    return np.minimum(idx[:, :n], n - 1)


def compute_one(path: Path, n_boot: int, rng) -> dict | None:
    d = np.load(path, allow_pickle=False)
    y_true = d["y_true_scaled"].ravel().astype(np.float64)
    y_pred = d["y_pred_scaled"].ravel().astype(np.float64)
    n = len(y_true)
    horizon = int(d["horizon"]) if "horizon" in d.files else 1
    if n < max(horizon + 5, 30):
        return None
    bl = max(horizon, 4)
    idxs = moving_block_bootstrap_indices(n, bl, n_boot, rng)
    row: dict = {}
    for name, fn in METRIC_FNS.items():
        boots = np.array([fn(y_true[i], y_pred[i]) for i in idxs])
        boots = boots[np.isfinite(boots)]
        if len(boots) < 50:
            row[name] = float("nan")
            row[f"{name}_lo"] = float("nan")
            row[f"{name}_hi"] = float("nan")
            row[f"{name}_se"] = float("nan")
            continue
        lo = float(np.percentile(boots, 100 * ALPHA / 2))
        hi = float(np.percentile(boots, 100 * (1 - ALPHA / 2)))
        row[name] = fn(y_true, y_pred)  # point estimate from full data
        row[f"{name}_lo"] = lo
        row[f"{name}_hi"] = hi
        row[f"{name}_se"] = float(np.std(boots, ddof=1))
    row["n_test"] = n
    row["block_len"] = bl
    return row


def parse_filename(p: Path) -> dict:
    parts = p.stem.split("__")
    if len(parts) != 4:
        return {}
    ds, variant, h, s = parts
    return {
        "dataset": ds,
        "variant": variant,
        "horizon": int(h.lstrip("h")),
        "seed": int(s.lstrip("s")),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PRED_DIR.glob("*.npz"))
    print(f"[1.1] {len(files)} prediction files")
    if not files:
        raise SystemExit("no prediction files; run pipeline/phase5_main.py first")

    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for i, p in enumerate(files, 1):
        meta = parse_filename(p)
        if not meta:
            continue
        try:
            r = compute_one(p, N_BOOT, rng)
        except Exception as e:
            print(f"  skip {p.name}: {type(e).__name__}: {e}")
            continue
        if r is None:
            continue
        rows.append({**meta, **r})
        if i % 100 == 0:
            print(f"  {i}/{len(files)}")
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "bootstrap_ci.csv"
    df.to_csv(out_csv, index=False)
    print(f"[1.1] wrote {out_csv}  ({len(df)} rows)")

    if df.empty:
        return

    # Aggregate over seeds: mean ± std of point + mean lo / mean hi
    agg_rows = []
    for (ds, var, h), sub in df.groupby(["dataset", "variant", "horizon"]):
        row = {"dataset": ds, "variant": var, "horizon": h, "n_seeds": len(sub)}
        for m in METRIC_FNS:
            row[f"{m}_mean"] = float(sub[m].mean())
            row[f"{m}_std"] = float(sub[m].std(ddof=1) if len(sub) > 1 else 0.0)
            row[f"{m}_lo_mean"] = float(sub[f"{m}_lo"].mean())
            row[f"{m}_hi_mean"] = float(sub[f"{m}_hi"].mean())
        agg_rows.append(row)
    agg = pd.DataFrame(agg_rows)
    out_agg = OUT_DIR / "bootstrap_ci_agg.csv"
    agg.to_csv(out_agg, index=False)
    print(f"[1.1] wrote {out_agg}  ({len(agg)} aggregated rows)")


if __name__ == "__main__":
    with PhaseTimer(
        "expA_1.1_bootstrap_ci",
        notes=f"N_BOOT={N_BOOT}, block=horizon, alpha={ALPHA}, seed={RNG_SEED}",
    ) as t:
        main()
        t.add_output("ci_csv", str(OUT_DIR / "bootstrap_ci.csv"))
        t.add_output("agg_csv", str(OUT_DIR / "bootstrap_ci_agg.csv"))
