"""Phase-5 main runner: 3 datasets × 16 model variants × 5 horizons × multi-seed.

Behaviour
---------
* HP search is performed ONCE per (dataset, model, horizon) with seed=42 and
  persisted to ``results/hyperparameters/<dataset>_<model>_h{h}.json``; later
  multi-seed runs reuse the same chosen hyperparameters.
* Per-run artifacts are persisted to ``results/metrics`` and
  ``results/predictions``; the runner is resumable — completed
  ``(dataset, variant, h, seed)`` tuples are skipped on restart.
* A final completion matrix is printed and saved as
  ``results/phase5_completion_matrix.json``.

CLI knobs
---------
``--datasets cesnet abilene geant``      restrict to a subset of datasets
``--horizons 1 24``                       restrict to a subset of horizons
``--models naive lstm timesfm_zs``        restrict to a subset of variants
``--max-seeds 3``                         cap multi-seed runs (default = all 5)
``--force``                               re-run even if artifacts exist
``--skip-hp-search``                      skip HP search entirely (use base_hparams)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import (
    HP_GRIDS,
    MODEL_CONFIGS,
    is_complete,
    metrics_path,
    run_single,
    scan_completed,
    seeds_for,
)
from src.training.hp_search import grid_search_one
from src.utils import (
    detect_device,
    load_config,
    log_device_info,
    set_global_seed,
    setup_logging,
)
from src.utils.io import write_json
from src.utils.logging_setup import get_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-5 full experiment runner.")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=None)
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="display-variant names to restrict the run to.",
    )
    p.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Cap on the number of seeds for stochastic models.",
    )
    p.add_argument(
        "--force", action="store_true", help="Re-run completed combinations."
    )
    p.add_argument("--skip-hp-search", action="store_true")
    p.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Persist trained model .pt files under results/checkpoints/. "
        "Recommended for the proposed CHA-Hybrid variants so the paper "
        "is fully reproducible.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()
    log_file = setup_logging(
        log_dir=cfg.resolve(cfg.paths.logs), run_name="phase5_main"
    )
    log = get_logger("phase5.main")
    set_global_seed(cfg.random_seed)
    info = log_device_info(detect_device())
    device = info.device

    datasets = args.datasets or list(cfg.datasets.keys())
    horizons = args.horizons or list(map(int, cfg.horizons))
    seeds_master = list(map(int, cfg.seeds))
    if args.max_seeds is not None and args.max_seeds < len(seeds_master):
        seeds_master = seeds_master[: args.max_seeds]
    model_filter = set(args.models) if args.models else None

    configs = [
        c for c in MODEL_CONFIGS if (model_filter is None or c[1] in model_filter)
    ]
    log.info(
        "Phase-5 main grid: datasets=%s horizons=%s models=%s seeds=%s",
        datasets,
        horizons,
        [c[1] for c in configs],
        seeds_master,
    )

    n_planned = 0
    for d in datasets:
        for h in horizons:
            for reg, variant, base, stoch in configs:
                n_planned += len(seeds_for(stoch, seeds_master))
    log.info("Planned runs: %d", n_planned)

    # ---- 1) HP search per (dataset, model, horizon) ----
    if not args.skip_hp_search:
        hp_dir = cfg.resolve(cfg.paths.hp_search)
        for d in datasets:
            for h in horizons:
                for reg, variant, base, stoch in configs:
                    grid = HP_GRIDS.get(reg, {})
                    if not grid:
                        continue
                    cache = hp_dir / f"{d}_{reg}_h{h}.json"
                    if cache.exists() and not args.force:
                        log.info("[hp] cached: %s", cache)
                        continue
                    log.info(
                        "[hp] running search %s/%s/h=%d (%d candidates)",
                        d,
                        reg,
                        h,
                        sum(len(v) for v in grid.values()),
                    )
                    grid_search_one(
                        dataset_name=d,
                        model_name=reg,
                        horizon=h,
                        grid=grid,
                        device=device,
                        seed=int(cfg.random_seed),
                        out_dir=hp_dir,
                        base_hparams=base,
                    )

    def _chosen_hparams(reg: str, base: dict, d: str, h: int) -> dict:
        hp_file = cfg.resolve(cfg.paths.hp_search) / f"{d}_{reg}_h{h}.json"
        if hp_file.exists():
            d_hp = json.loads(hp_file.read_text())
            return d_hp.get("chosen", dict(base))
        return dict(base)

    # ---- 2) main loop ----
    t_start = time.perf_counter()
    n_done = n_skip = n_fail = 0
    rows = []
    for d in datasets:
        for h in horizons:
            for reg, variant, base, stoch in configs:
                seeds = seeds_for(stoch, seeds_master)
                chosen = _chosen_hparams(reg, base, d, h)
                for seed in seeds:
                    if is_complete(d, variant, h, seed) and not args.force:
                        n_skip += 1
                        continue
                    log.info(">>> %s / %s / h=%d / seed=%d", d, variant, h, seed)
                    # Save checkpoints by default for the proposed CHA-Hybrid
                    # family, or whenever the user asks for it explicitly.
                    save_ck = bool(args.save_checkpoints) or variant.startswith(
                        "cha_hybrid"
                    )
                    res = run_single(
                        dataset=d,
                        model_name=reg,
                        variant=variant,
                        horizon=h,
                        seed=seed,
                        chosen_hparams=chosen,
                        device=device,
                        force=args.force,
                        save_checkpoint=save_ck,
                    )
                    rows.append(res)
                    n_done += 1
                    if res.status != "ok":
                        n_fail += 1

    elapsed = time.perf_counter() - t_start
    log.info(
        "Phase 5 main run finished — %d new runs, %d skipped (cached), %d fail "
        "in %.1f s",
        n_done,
        n_skip,
        n_fail,
        elapsed,
    )

    # ---- 3) build & save completion matrix ----
    completed = scan_completed()
    matrix: dict = {}
    for d in datasets:
        matrix[d] = {}
        for h in horizons:
            matrix[d][h] = {}
            for reg, variant, base, stoch in configs:
                planned_seeds = seeds_for(stoch, seeds_master)
                completed_seeds = sorted(completed.get((d, variant, h), {}).keys())
                fail_seeds = [
                    s
                    for s in completed_seeds
                    if completed.get((d, variant, h), {}).get(s, "ok") != "ok"
                ]
                matrix[d][h][variant] = {
                    "planned": planned_seeds,
                    "completed": completed_seeds,
                    "failed": fail_seeds,
                    "missing": sorted(set(planned_seeds) - set(completed_seeds)),
                }

    out_matrix = cfg.resolve(cfg.paths.results) / "phase5_completion_matrix.json"
    write_json(
        {
            "datasets": datasets,
            "horizons": horizons,
            "variants": [c[1] for c in configs],
            "seeds_master": seeds_master,
            "matrix": matrix,
            "elapsed_seconds": float(elapsed),
            "n_done": n_done,
            "n_skipped": n_skip,
            "n_failed": n_fail,
        },
        out_matrix,
    )
    log.info("Wrote completion matrix to %s", out_matrix)

    # ---- 4) pretty-print summary table ----
    print()
    print("=" * 110)
    print(f"PHASE-5 COMPLETION MATRIX  (datasets={datasets}  horizons={horizons})")
    print("=" * 110)
    header = f"{'VARIANT':<16} " + "  ".join(f"{d:^25s}" for d in datasets)
    print(header)
    print("-" * 110)
    for reg, variant, base, stoch in configs:
        line = f"{variant:<16}"
        for d in datasets:
            cells = []
            for h in horizons:
                cs = matrix[d][h][variant]["completed"]
                ps = matrix[d][h][variant]["planned"]
                ms = matrix[d][h][variant]["missing"]
                cells.append(f"h={h}:{len(cs)}/{len(ps)}")
            line += "  " + " ".join(cells)
        print(line)
    print("-" * 110)
    print(
        f"new={n_done}  cached_skipped={n_skip}  failed={n_fail}  elapsed={elapsed:.1f}s"
    )
    print(f"matrix saved to: {out_matrix}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
