"""Phase-6 evaluation v2: aggregates final results with CHA-Hybrid v2 as the proposed model.

Runs:
  * Ablation v2 (4 variants × 5 datasets × 5 horizons × 5 seeds)
  * Aggregated metric tables (mean ± std)
  * DM test: cha_hybrid_v2 vs every other variant
  * Cost table including v2
  * Winners + proposed (v2) rank table
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
from src.evaluation.ablation_v2 import run_ablation_v2_for_one
from src.training import MODEL_CONFIGS
from src.utils import (
    detect_device,
    load_config,
    log_device_info,
    set_global_seed,
    setup_logging,
)
from src.utils.logging_setup import get_logger


def get_cha_v2_base() -> dict:
    for reg, variant, base, stoch in MODEL_CONFIGS:
        if variant == "cha_hybrid_v2":
            return dict(base)
    raise RuntimeError("cha_hybrid_v2 not found in MODEL_CONFIGS")


def main() -> int:
    cfg = load_config()
    log_file = setup_logging(
        log_dir=cfg.resolve(cfg.paths.logs), run_name="phase6_eval_v2"
    )
    log = get_logger("phase6.v2")
    set_global_seed(cfg.random_seed)
    info = log_device_info(detect_device())
    device = info.device
    out_dir = cfg.resolve(cfg.paths.results) / "eval_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- (A) Ablation v2 ----
    log.info("=" * 78)
    log.info("Phase 6v2 (A) — ABLATION V2")
    log.info("=" * 78)
    cha_base = get_cha_v2_base()
    datasets = list(cfg.datasets.keys())
    horizons = list(map(int, cfg.horizons))
    seeds = list(map(int, cfg.seeds))
    n_planned = len(datasets) * len(horizons) * len(seeds) * 4
    log.info(
        "Ablation v2 planned: %d cells (%d datasets × %d horizons × %d seeds × 4 variants)",
        n_planned,
        len(datasets),
        len(horizons),
        len(seeds),
    )
    t0 = time.perf_counter()
    for ds in datasets:
        for h in horizons:
            for s in seeds:
                run_ablation_v2_for_one(ds, h, s, cha_base, device, force=False)
    log.info("Ablation v2 done in %.1fs", time.perf_counter() - t0)

    # ---- (B) aggregated tables ----
    log.info("=" * 78)
    log.info("Phase 6v2 (B) — METRIC TABLES")
    log.info("=" * 78)
    df_runs = load_all_runs()
    log.info("Loaded %d successful runs", len(df_runs))
    written = write_publication_tables(df_runs, out_dir / "tables")
    for k, p in written.items():
        log.info("  table[%s] -> %s", k, p)

    # ---- (C) winners + proposed rank ----
    log.info("=" * 78)
    log.info("Phase 6v2 (C) — WINNERS & RANK")
    log.info("=" * 78)
    write_winners(df_runs, out_dir / "winners", proposed="cha_hybrid_v2")
    df_rank = proposed_rank(df_runs, "cha_hybrid_v2", "rmse_native")
    log.info(
        "CHA-Hybrid v2 rank per (dataset, horizon):\n%s", df_rank.to_string(index=False)
    )

    # ---- (D) DM test: v2 vs everything ----
    log.info("=" * 78)
    log.info("Phase 6v2 (D) — DIEBOLD-MARIANO")
    log.info("=" * 78)
    df_dm = pairwise_dm_against_proposed(proposed="cha_hybrid_v2")
    p_dm = write_dm_table(df_dm, out_dir / "dm_test")
    if len(df_dm):
        n_wins = int(((df_dm["statistic"] < 0) & df_dm["significant_at_005"]).sum())
        n_loss = int(((df_dm["statistic"] > 0) & df_dm["significant_at_005"]).sum())
        n_ns = int((~df_dm["significant_at_005"]).sum())
        log.info(
            "DM summary (v2 vs others, p<0.05): wins=%d losses=%d not_sig=%d / total=%d",
            n_wins,
            n_loss,
            n_ns,
            len(df_dm),
        )

    # ---- (E) cost ----
    log.info("=" * 78)
    log.info("Phase 6v2 (E) — COST TABLE")
    log.info("=" * 78)
    write_cost_tables(df_runs, out_dir / "cost")
    log.info("Cost head:\n%s", cost_table(df_runs).head(20).to_string())

    summary = {
        "n_runs_aggregated": int(len(df_runs)),
        "datasets": datasets,
        "horizons": horizons,
        "seeds": seeds,
        "proposed": "cha_hybrid_v2",
        "ablation_variants": [
            "cha_hybrid_v2",
            "cha_hybrid_v2_decomp_only",
            "cha_hybrid_v2_global_only",
            "cha_hybrid_v2_fixed_alpha_0.5",
            "cha_hybrid_v2_altres_gru",
        ],
        "out_dir": str(out_dir),
        "dm_total": int(len(df_dm)) if len(df_dm) else 0,
    }
    (out_dir / "phase6_v2_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Phase 6 v2 done. Summary: %s", out_dir / "phase6_v2_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
