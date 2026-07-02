"""Phase-5 status inspector: print the completion matrix at any time.

Reads ``results/metrics/*.json`` and prints a (variant × dataset × horizon)
grid showing how many seeds are completed vs planned. Also writes the
machine-readable matrix to ``results/phase5_completion_matrix.json``.

Useful both during a long-running grid (to peek at progress) and after,
as the Phase-5 verification artifact.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import MODEL_CONFIGS, scan_completed, seeds_for
from src.utils import load_config
from src.utils.io import write_json


def main() -> int:
    cfg = load_config()
    datasets = list(cfg.datasets.keys())
    horizons = list(map(int, cfg.horizons))
    seeds = list(map(int, cfg.seeds))

    completed = scan_completed()
    matrix = {}
    n_planned_total = 0
    n_done_total = 0
    n_fail_total = 0
    for d in datasets:
        matrix[d] = {}
        for h in horizons:
            matrix[d][h] = {}
            for reg, variant, base, stoch in MODEL_CONFIGS:
                planned = seeds_for(stoch, seeds)
                done = sorted(completed.get((d, variant, h), {}).keys())
                fails = [s for s in done
                         if completed.get((d, variant, h), {}).get(s, "ok") != "ok"]
                matrix[d][h][variant] = {
                    "planned": planned,
                    "completed": done,
                    "failed": fails,
                    "missing": sorted(set(planned) - set(done)),
                }
                n_planned_total += len(planned)
                n_done_total += len(done)
                n_fail_total += len(fails)

    out_path = cfg.resolve(cfg.paths.results) / "phase5_completion_matrix.json"
    write_json({
        "datasets": datasets, "horizons": horizons,
        "variants": [c[1] for c in MODEL_CONFIGS],
        "seeds_master": seeds,
        "matrix": matrix,
        "n_planned": n_planned_total,
        "n_done": n_done_total,
        "n_failed": n_fail_total,
        "pct_complete": round(100.0 * n_done_total / max(n_planned_total, 1), 2),
    }, out_path)

    print()
    print("=" * 110)
    print("PHASE-5 COMPLETION MATRIX")
    print(f"  datasets={datasets}, horizons={horizons}, seeds_master={seeds}")
    print(f"  total: {n_done_total}/{n_planned_total} done "
          f"({100.0*n_done_total/max(n_planned_total,1):.1f}%) "
          f"fails={n_fail_total}")
    print(f"  matrix file: {out_path}")
    print("=" * 110)

    # per-dataset table: variant × horizon cells showing "done/planned"
    for d in datasets:
        print(f"\nDataset = {d}")
        header = f"{'VARIANT':<16} " + " ".join(f"h={h:<3}" for h in horizons)
        print(header)
        print("-" * len(header))
        for reg, variant, base, stoch in MODEL_CONFIGS:
            cells = []
            for h in horizons:
                cell = matrix[d][h][variant]
                if len(cell["completed"]) == len(cell["planned"]):
                    sym = f" {len(cell['completed'])}/{len(cell['planned'])} "
                else:
                    sym = f"-{len(cell['completed'])}/{len(cell['planned'])}-"
                cells.append(f"{sym:<5}")
            print(f"{variant:<16} " + " ".join(cells))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
