"""Phase-6 evaluation driver.

Produces:
  * publication-ready metric tables (RMSE/MAE/MAPE/sMAPE/R^2)
  * pairwise Diebold-Mariano test (CHA-Hybrid vs every other)
  * 4-variant CHA-Hybrid ablation
  * computational-cost table
  * a per-(dataset, horizon) winner table

All under ``results/eval/`` so Phase 7 can pick them up for figures.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (
    cost_table,
    load_all_runs,
    pairwise_dm_against_proposed,
    proposed_rank,
    winners_table,
    write_cost_tables,
    write_dm_table,
    write_publication_tables,
    write_winners,
)
from src.evaluation.ablation import run_ablation_for_one
from src.training import MODEL_CONFIGS
from src.utils import (
    detect_device,
    load_config,
    log_device_info,
    set_global_seed,
    setup_logging,
)
from src.utils.logging_setup import get_logger


def get_cha_hybrid_base() -> dict:
    """Pull the cha_hybrid base_hparams from MODEL_CONFIGS."""
    for reg, variant, base, stoch in MODEL_CONFIGS:
        if variant == "cha_hybrid":
            return dict(base)
    raise RuntimeError("cha_hybrid not found in MODEL_CONFIGS")


def main() -> int:
    cfg = load_config()
    log_file = setup_logging(
        log_dir=cfg.resolve(cfg.paths.logs), run_name="phase6_eval"
    )
    log = get_logger("phase6")
    set_global_seed(cfg.random_seed)
    info = log_device_info(detect_device())
    device = info.device
    out_dir = cfg.resolve(cfg.paths.results) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # (A) Ablation: 4 CHA-Hybrid variants for every (dataset, horizon, seed)
    # --------------------------------------------------------------
    log.info("=" * 78)
    log.info("Phase 6 (A) — ABLATION")
    log.info("=" * 78)
    cha_base = get_cha_hybrid_base()
    datasets = list(cfg.datasets.keys())
    horizons = list(map(int, cfg.horizons))
    seeds = list(map(int, cfg.seeds))
    n_ablation_planned = len(datasets) * len(horizons) * len(seeds) * 4
    log.info(
        "Ablation planned: %d cells (%d datasets × %d horizons × %d seeds × 4 variants)",
        n_ablation_planned,
        len(datasets),
        len(horizons),
        len(seeds),
    )
    t0 = time.perf_counter()
    for ds in datasets:
        for h in horizons:
            for s in seeds:
                run_ablation_for_one(ds, h, s, cha_base, device, force=False)
    log.info("Ablation done in %.1f s", time.perf_counter() - t0)

    # --------------------------------------------------------------
    # (B) Aggregate metrics across all (main grid + ablation) runs
    # --------------------------------------------------------------
    log.info("=" * 78)
    log.info("Phase 6 (B) — METRIC TABLES")
    log.info("=" * 78)
    df_runs = load_all_runs()
    log.info("Loaded %d successful runs from results/metrics/", len(df_runs))
    written = write_publication_tables(df_runs, out_dir / "tables")
    for k, p in written.items():
        log.info("  table[%s] -> %s", k, p)

    # --------------------------------------------------------------
    # (C) Winners table + proposed-rank
    # --------------------------------------------------------------
    log.info("=" * 78)
    log.info("Phase 6 (C) — WINNERS & RANK")
    log.info("=" * 78)
    win_files = write_winners(df_runs, out_dir / "winners", proposed="cha_hybrid")
    for k, p in win_files.items():
        log.info("  winners[%s] -> %s", k, p)
    df_rank = proposed_rank(df_runs, "cha_hybrid", "rmse_native")
    log.info(
        "Proposed rank per (dataset, horizon):\n%s", df_rank.to_string(index=False)
    )

    # --------------------------------------------------------------
    # (D) Diebold-Mariano pairwise (proposed vs every other)
    # --------------------------------------------------------------
    log.info("=" * 78)
    log.info("Phase 6 (D) — DIEBOLD-MARIANO TEST")
    log.info("=" * 78)
    df_dm = pairwise_dm_against_proposed(proposed="cha_hybrid")
    p_dm = write_dm_table(df_dm, out_dir / "dm_test")
    log.info("Wrote DM test (%d comparisons) -> %s", len(df_dm), p_dm)
    # quick summary of how many comparisons the proposed model beats with p<0.05
    if len(df_dm):
        n_wins = int(((df_dm["statistic"] < 0) & (df_dm["significant_at_005"])).sum())
        n_loss = int(((df_dm["statistic"] > 0) & (df_dm["significant_at_005"])).sum())
        n_ns = int((~df_dm["significant_at_005"]).sum())
        log.info(
            "DM summary (CHA-Hybrid vs others, p<0.05): wins=%d losses=%d "
            "not_sig=%d / total=%d",
            n_wins,
            n_loss,
            n_ns,
            len(df_dm),
        )

    # --------------------------------------------------------------
    # (E) Cost table
    # --------------------------------------------------------------
    log.info("=" * 78)
    log.info("Phase 6 (E) — COST TABLE")
    log.info("=" * 78)
    cost_files = write_cost_tables(df_runs, out_dir / "cost")
    log.info("Cost files: %s", list(cost_files.values()))
    log.info("Overall cost table:\n%s", cost_table(df_runs).to_string())

    # --------------------------------------------------------------
    # final summary JSON
    # --------------------------------------------------------------
    summary = {
        "n_runs_aggregated": int(len(df_runs)),
        "datasets": datasets,
        "horizons": horizons,
        "seeds": seeds,
        "ablation_variants": [
            "cha_hybrid",
            "cha_hybrid_decomp_only",
            "cha_hybrid_global_only",
            "cha_hybrid_fixed_alpha_0.5",
            "cha_hybrid_altres_lstm",
        ],
        "out_dir": str(out_dir),
        "dm_total": int(len(df_dm)) if len(df_dm) else 0,
    }
    (out_dir / "phase6_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Wrote summary to %s", out_dir / "phase6_summary.json")
    log.info("Phase 6 done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
