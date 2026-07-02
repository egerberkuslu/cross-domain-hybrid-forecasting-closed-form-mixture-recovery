"""Phase-2 driver: preprocess all datasets + run the verification checklist."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import (
    SPLITS_DIR,
    load_preprocessed,
    preprocess_all,
)
from src.utils import load_config, setup_logging, set_global_seed
from src.utils.logging_setup import get_logger


def verify(cfg, processed: dict) -> dict:
    """Run the explicit Phase-2 verification checklist; return pass/fail map."""
    log = get_logger("phase2.verify")
    results: dict[str, list[tuple[str, bool, str]]] = {}

    for name, pp in processed.items():
        items: list[tuple[str, bool, str]] = []

        # ---- A) scaler fit on TRAIN ONLY ----
        train_mean = float(np.nanmean(pp.split_native.train["value"].values))
        train_std = float(np.nanstd(pp.split_native.train["value"].values))
        sc = pp.scaler
        if hasattr(sc, "mean_"):
            ok = abs(float(sc.mean_[0]) - train_mean) < 1e-3
            items.append(("scaler fit on TRAIN only (mean matches train)",
                          ok, f"scaler.mean_={float(sc.mean_[0]):.3e} vs train_mean={train_mean:.3e}"))
        if hasattr(sc, "scale_"):
            ok = abs(float(sc.scale_[0]) - train_std) < (1.0 + 1e-3) * train_std * 1e-3 + 1e-3
            items.append(("scaler scale ≈ train std",
                          ok, f"scaler.scale_={float(sc.scale_[0]):.3e} vs train_std={train_std:.3e}"))

        # additional rigorous check: scaling the train series gives mean ≈ 0, std ≈ 1
        z = pp.split_scaled.train["value"].dropna().to_numpy()
        items.append(("scaled train mean ≈ 0", abs(z.mean()) < 1e-6, f"{z.mean():.3e}"))
        items.append(("scaled train std ≈ 1", abs(z.std() - 1.0) < 1e-3, f"{z.std():.6f}"))

        # ---- B) chronological order ----
        try:
            pp.split_native.assert_chronological()
            ok = True; detail = "train_end < val_start < ... < test_start"
        except AssertionError as e:
            ok = False; detail = str(e)
        items.append(("splits chronological", ok, detail))

        # print sizes and ranges
        sizes = pp.split_native.sizes
        ranges = pp.split_native.ranges
        log.info("[%s] split sizes: %s", name, sizes)
        for s in ["train", "val", "test"]:
            log.info("[%s] %-5s: %s -> %s (n=%d)",
                     name, s, ranges[s][0], ranges[s][1], sizes[s])

        # ---- C) windowed shapes & no NaN / inf ----
        for h in pp.horizons:
            for split in ["train", "val", "test"]:
                ws = pp.windows[h][split]
                log.info("[%s] h=%2d %-5s X=%s y=%s",
                         name, h, split, ws.X.shape, ws.y.shape)
                items.append((
                    f"h={h} {split}: no NaN in X",
                    not np.isnan(ws.X).any(),
                    f"shape={ws.X.shape}",
                ))
                items.append((
                    f"h={h} {split}: no NaN in y",
                    not np.isnan(ws.y).any(),
                    f"shape={ws.y.shape}",
                ))
                items.append((
                    f"h={h} {split}: no inf",
                    not (np.isinf(ws.X).any() or np.isinf(ws.y).any()),
                    "ok",
                ))
                items.append((
                    f"h={h} {split}: lookback={pp.lookback}",
                    ws.X.shape[1] == pp.lookback,
                    str(ws.X.shape[1]),
                ))
                items.append((
                    f"h={h} {split}: y_dim={h}",
                    ws.y.shape[1] == h,
                    str(ws.y.shape[1]),
                ))
                if split != "train":
                    # confirm target timestamps lie inside the split's date range
                    lo, hi = ranges[split]
                    ok = (ws.target_times.min() >= lo) and (ws.target_times.max() <= hi)
                    items.append((
                        f"h={h} {split}: target timestamps in split range",
                        bool(ok),
                        f"{ws.target_times.min()} .. {ws.target_times.max()}",
                    ))

        # ---- D) identical splits will be reused for every model ----
        # round-trip: reload from disk and ensure equality
        reloaded = load_preprocessed(name, root=pp.out_dir.parent)
        ok = (
            reloaded.split_native.train.equals(pp.split_native.train)
            and reloaded.split_native.val.equals(pp.split_native.val)
            and reloaded.split_native.test.equals(pp.split_native.test)
        )
        items.append(("on-disk reload reproduces splits exactly", ok,
                      "ChronologicalSplit equality round-trip"))

        # same windows
        for h in pp.horizons:
            for split in ["train", "val", "test"]:
                a = pp.windows[h][split]
                b = reloaded.windows[h][split]
                ok = (
                    np.array_equal(a.X, b.X)
                    and np.array_equal(a.y, b.y)
                    and (a.target_times == b.target_times).all()
                )
                items.append((
                    f"on-disk reload reproduces windows h={h} {split}",
                    ok, "array_equal X & y & target_times",
                ))

        results[name] = items
    return results


def print_checklist(results: dict) -> None:
    print()
    print("=" * 82)
    print(f"{'STATUS':<6} {'CHECK':<60} {'DETAIL'}")
    print("=" * 82)
    total = passed = 0
    for name, items in results.items():
        print(f"-- {name} " + "-" * (78 - len(name)))
        for n, ok, detail in items:
            total += 1
            passed += int(ok)
            print(f"{'PASS' if ok else 'FAIL':<6} {n:<60} {detail}")
    print("=" * 82)
    print(f"TOTAL: {passed}/{total} passed")


def main() -> int:
    cfg = load_config()
    log_file = setup_logging(log_dir=cfg.resolve(cfg.paths.logs),
                             run_name="phase2_preprocess")
    log = get_logger("phase2")
    set_global_seed(cfg.random_seed)
    log.info("Phase 2 driver started — log: %s", log_file)

    processed = preprocess_all(cfg)
    log.info("Preprocessed %d datasets", len(processed))

    checklist = verify(cfg, processed)
    print_checklist(checklist)

    # exit non-zero on any failure
    all_ok = all(ok for items in checklist.values() for _, ok, _ in items)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
