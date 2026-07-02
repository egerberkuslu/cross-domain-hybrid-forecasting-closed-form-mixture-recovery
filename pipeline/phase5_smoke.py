"""Phase-5 smoke test: tiny subset to prove the grid runner executes end-to-end.

Runs ONE dataset (geant — smallest) × ONE horizon (h=1) × every model variant
(both deterministic and stochastic) × the FIRST seed only. Persists everything
through the same code path the full grid uses, so a green smoke test means
the full grid will execute without surprises.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.training import (
    MODEL_CONFIGS,
    HP_GRIDS,
    is_complete,
    metrics_path,
    run_single,
    scan_completed,
)
from src.training.hp_search import grid_search_one
from src.utils import detect_device, load_config, log_device_info, set_global_seed, setup_logging
from src.utils.logging_setup import get_logger


SMOKE_DATASETS = ["geant"]
SMOKE_HORIZONS = [1]


def main() -> int:
    cfg = load_config()
    log_file = setup_logging(log_dir=cfg.resolve(cfg.paths.logs), run_name="phase5_smoke")
    log = get_logger("phase5.smoke")
    set_global_seed(cfg.random_seed)
    info = log_device_info(detect_device())
    device = info.device
    log.info("Phase 5 smoke — datasets=%s horizons=%s device=%s",
             SMOKE_DATASETS, SMOKE_HORIZONS, device)

    rows = []
    t_start = time.perf_counter()
    for dataset in SMOKE_DATASETS:
        for h in SMOKE_HORIZONS:
            for registry_name, variant, base_hparams, stochastic in MODEL_CONFIGS:
                seed = int(cfg.seeds[0])
                log.info("=" * 90)
                log.info(">>> %s / %s / h=%d / seed=%d", dataset, variant, h, seed)
                log.info("=" * 90)

                # ---- HP search if the model has a tunable grid ----
                grid = HP_GRIDS.get(registry_name, {})
                if grid:
                    hp_dir = cfg.resolve(cfg.paths.hp_search)
                    hp_res = grid_search_one(
                        dataset_name=dataset, model_name=registry_name,
                        horizon=h, grid=grid, device=device,
                        seed=seed, out_dir=hp_dir,
                        base_hparams=base_hparams,
                    )
                    chosen = hp_res.chosen
                else:
                    chosen = dict(base_hparams)

                res = run_single(
                    dataset=dataset, model_name=registry_name, variant=variant,
                    horizon=h, seed=seed,
                    chosen_hparams=chosen, device=device, force=False,
                )
                rows.append(res)

    elapsed = time.perf_counter() - t_start

    # ---- summary ----
    n_ok = sum(1 for r in rows if r.status == "ok")
    n_fail = sum(1 for r in rows if r.status != "ok")
    print()
    print("=" * 100)
    print(f"PHASE-5 SMOKE — {len(rows)} runs in {elapsed:.1f}s ({n_ok} pass / {n_fail} fail)")
    print("=" * 100)
    print(f"{'DATASET':<8} {'VARIANT':<16} {'H':>3} {'SEED':>5} {'STATUS':<8} "
          f"{'RMSE_SCALED':>12} {'RMSE_NATIVE':>14} {'FIT_S':>7} {'PRED_S':>7}")
    print("-" * 100)
    for r in rows:
        rmse_s = r.metrics_scaled.get("rmse", float("nan"))
        rmse_n = r.metrics_native.get("rmse", float("nan"))
        print(f"{r.dataset:<8} {r.variant:<16} {r.horizon:>3} {r.seed:>5} "
              f"{r.status[:8]:<8} {rmse_s:>12.4f} {rmse_n:>14.4e} "
              f"{r.fit_seconds:>7.1f} {r.predict_seconds:>7.1f}")
    print("=" * 100)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
