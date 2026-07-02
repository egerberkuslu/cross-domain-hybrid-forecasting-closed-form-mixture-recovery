"""Phase-6 evaluation v3: ablation + DM-test + cost for CHA-Hybrid v3.

Runs:
  * Ablation v3 (4 variants × 5 datasets × 5 horizons × 5 seeds = 500 cells,
    of which ~250 share the v3 fit and 125 alias Chronos-Bolt → only ~150
    new model fits to do).
  * Aggregated metric tables (mean ± std) over the full 2,400+ run grid.
  * DM test: cha_hybrid_v3 vs every other variant.
  * Cost table including v3.
  * Winners + proposed-v3 rank table.
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
    write_cost_tables,
    write_dm_table,
    write_publication_tables,
    write_winners,
)
from src.evaluation.ablation_v3 import run_ablation_v3_for_one
from src.training import MODEL_CONFIGS
from src.utils import (
    detect_device,
    load_config,
    log_device_info,
    set_global_seed,
    setup_logging,
)
from src.utils.logging_setup import get_logger


def get_cha_v3_base() -> dict:
    for reg, variant, base, stoch in MODEL_CONFIGS:
        if variant == "cha_hybrid_v3":
            return dict(base)
    raise RuntimeError("cha_hybrid_v3 not found in MODEL_CONFIGS")


def main() -> int:
    cfg = load_config()
    log_file = setup_logging(
        log_dir=cfg.resolve(cfg.paths.logs), run_name="phase6_eval_v3"
    )
    log = get_logger("phase6.v3")
    set_global_seed(cfg.random_seed)
    info = log_device_info(detect_device())
    device = info.device
    out_dir = cfg.resolve(cfg.paths.results) / "eval_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- (A) Ablation v3 ----
    log.info("=" * 78)
    log.info("Phase 6v3 (A) — ABLATION V3")
    log.info("=" * 78)
    cha_base = get_cha_v3_base()
    datasets = list(cfg.datasets.keys())
    horizons = list(map(int, cfg.horizons))
    seeds = list(map(int, cfg.seeds))
    t0 = time.perf_counter()
    for ds in datasets:
        for h in horizons:
            for s in seeds:
                run_ablation_v3_for_one(ds, h, s, cha_base, device, force=False)
    log.info("Ablation v3 done in %.1fs", time.perf_counter() - t0)

    # ---- (B) Aggregated tables ----
    log.info("=" * 78)
    log.info("Phase 6v3 (B) — METRIC TABLES")
    log.info("=" * 78)
    df_runs = load_all_runs()
    log.info("Loaded %d successful runs", len(df_runs))
    written = write_publication_tables(df_runs, out_dir / "tables")
    for k, p in written.items():
        log.info("  table[%s] -> %s", k, p)

    # ---- (C) winners + proposed rank ----
    log.info("=" * 78)
    log.info("Phase 6v3 (C) — WINNERS & RANK")
    log.info("=" * 78)
    write_winners(df_runs, out_dir / "winners", proposed="cha_hybrid_v3")
    df_rank = proposed_rank(df_runs, "cha_hybrid_v3", "rmse_native")
    log.info(
        "CHA-Hybrid v3 rank per (dataset, horizon):\n%s", df_rank.to_string(index=False)
    )

    # ---- (D) DM test ----
    log.info("=" * 78)
    log.info("Phase 6v3 (D) — DIEBOLD-MARIANO")
    log.info("=" * 78)
    df_dm = pairwise_dm_against_proposed(proposed="cha_hybrid_v3")
    p_dm = write_dm_table(df_dm, out_dir / "dm_test")
    if len(df_dm):
        n_wins = int(((df_dm["statistic"] < 0) & df_dm["significant_at_005"]).sum())
        n_loss = int(((df_dm["statistic"] > 0) & df_dm["significant_at_005"]).sum())
        n_ns = int((~df_dm["significant_at_005"]).sum())
        log.info(
            "DM summary (v3 vs others, p<0.05): wins=%d losses=%d not_sig=%d / total=%d",
            n_wins,
            n_loss,
            n_ns,
            len(df_dm),
        )

    # ---- (E) cost ----
    log.info("=" * 78)
    log.info("Phase 6v3 (E) — COST TABLE")
    log.info("=" * 78)
    write_cost_tables(df_runs, out_dir / "cost")
    log.info("Cost head:\n%s", cost_table(df_runs).head(25).to_string())

    summary = {
        "n_runs_aggregated": int(len(df_runs)),
        "datasets": datasets,
        "horizons": horizons,
        "seeds": seeds,
        "proposed": "cha_hybrid_v3",
        "ablation_variants": [
            "cha_hybrid_v3",
            "cha_hybrid_v3_decomp_only",
            "cha_hybrid_v3_global_only",
            "cha_hybrid_v3_fixed_alpha_0.5",
            "cha_hybrid_v3_altres_gru",
        ],
        "out_dir": str(out_dir),
        "dm_total": int(len(df_dm)) if len(df_dm) else 0,
    }
    (out_dir / "phase6_v3_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Phase 6 v3 done. Summary: %s", out_dir / "phase6_v3_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
