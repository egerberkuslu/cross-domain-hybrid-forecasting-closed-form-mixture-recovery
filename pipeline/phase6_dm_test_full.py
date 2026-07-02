"""Phase-6 DM-test (full coverage): CHA-S vs Chronos-Bolt + strong baselines on all 25 cells.

The original ``phase6_eval_v3.py`` produced
``outputs/eval_v3/dm_test/dm_test_pairwise.csv`` for only 3 datasets
(abilene, cesnet, geant) × 5 horizons = 15 cells per comparison. The two
NAB datasets (``nab_aws_cpu`` and ``nab_twitter``) were not in the dataset
list at the time of the run, even though their per-seed prediction NPZ
files were already on disk.

This phase re-runs the Diebold–Mariano (1995) test with the
Harvey-Leybourne-Newbold (1997) small-sample correction across the FULL
5-dataset × 5-horizon grid, comparing the proposed model ``cha_hybrid_v3``
(CHA-S) against:
  * ``chronos_bolt_zs`` — the strongest foundation-model baseline,
  * a curated list of other strong baselines (statistical, ML, deep
    learning, foundation models) so that the augmented CSV is broadly
    useful for the paper's significance discussion.

We DO NOT overwrite the original CSV. Results are written to
``outputs/eval_v3/dm_test/dm_test_pairwise_full.csv``. Provenance is
captured via ``PhaseTimer`` (``outputs/runlog.jsonl``).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.evaluation import pairwise_dm_against_proposed
from src.utils import (
    detect_device,
    load_config,
    log_device_info,
    set_global_seed,
    setup_logging,
)
from src.utils.logging_setup import get_logger
from src.utils.runlog import PhaseTimer

# Curated set of baselines to compare against the proposed CHA-S model.
# Chronos-Bolt is the headline baseline; the others give the reader a
# broad cross-section of statistical / deep / foundation models without
# bloating the CSV with the full ~30-variant matrix.
STRONG_BASELINES = [
    "chronos_bolt_zs",  # foundation (Amazon, T5-based)
    "chronos_zs",  # foundation (Amazon, original)
    "moirai_zs",  # foundation (Salesforce)
    "timesfm_zs",  # foundation (Google)
    "ttm_zs",  # foundation (IBM TTM)
    "patchtst",  # deep transformer baseline
    "nhits",  # deep MLP baseline
    "nbeats",  # deep MLP baseline
    "tft",  # deep attention baseline
    "dlinear",  # deep linear baseline
    "tide",  # deep MLP baseline
    "tsmixer",  # deep MLP baseline
    "xgboost",  # strong ML baseline
    "arima",  # classical baseline
    "farima",  # long-memory baseline
    "theta",  # classical baseline
    "seasonal_naive",  # naive baseline
]


def _summarize(df: pd.DataFrame, variant_b: str) -> tuple[int, int, int]:
    """Return (n_wins, n_ties, n_losses) for proposed vs ``variant_b``.

    Convention follows ``dm_test.py``: statistic < 0 ⇒ variant_a (proposed)
    has smaller loss ⇒ proposed wins.
    """
    sub = df[df["variant_b"] == variant_b]
    sig = sub["significant_at_005"]
    wins = int(((sub["statistic"] < 0) & sig).sum())
    losses = int(((sub["statistic"] > 0) & sig).sum())
    ties = int((~sig).sum())
    return wins, ties, losses


def main() -> int:
    cfg = load_config()
    setup_logging(log_dir=cfg.resolve(cfg.paths.logs), run_name="phase6_dm_test_full")
    log = get_logger("phase6.dm_full")
    set_global_seed(cfg.random_seed)
    log_device_info(detect_device())

    out_dir = cfg.resolve(cfg.paths.results) / "eval_v3" / "dm_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "dm_test_pairwise_full.csv"

    datasets = list(cfg.datasets.keys())  # all 5
    horizons = list(map(int, cfg.horizons))  # 1, 3, 6, 12, 24
    proposed = "cha_hybrid_v3"

    log.info("=" * 78)
    log.info("Phase 6 — DIEBOLD-MARIANO (FULL COVERAGE)")
    log.info("=" * 78)
    log.info(
        "Proposed: %s  |  datasets=%s  |  horizons=%s",
        proposed,
        datasets,
        horizons,
    )
    log.info("Baselines (%d): %s", len(STRONG_BASELINES), STRONG_BASELINES)

    with PhaseTimer(
        "phase6_dm_test_full",
        notes=f"CHA-S vs {len(STRONG_BASELINES)} baselines on {len(datasets)}x{len(horizons)} cells",
    ) as timer:
        df = pairwise_dm_against_proposed(
            proposed=proposed,
            other_variants=STRONG_BASELINES,
            datasets=datasets,
            horizons=horizons,
            loss="mse",
        )

        if df.empty:
            log.error("No DM rows produced — check predictions and y_true alignment.")
            return 1

        df = df.sort_values(["variant_b", "dataset", "horizon"]).reset_index(drop=True)
        df.to_csv(out_csv, index=False)
        timer.add_output("dm_csv", str(out_csv))
        timer.add_output("n_rows", int(len(df)))

        # ---- headline summary: CHA-S vs Chronos-Bolt ----
        bolt = df[df["variant_b"] == "chronos_bolt_zs"]
        n_bolt = int(len(bolt))
        wins, ties, losses = _summarize(df, "chronos_bolt_zs")
        timer.add_extra(
            "summary_vs_chronos_bolt",
            {"cells": n_bolt, "wins": wins, "ties": ties, "losses": losses},
        )

        # ---- broader per-baseline summary table ----
        summary_rows = []
        for v in STRONG_BASELINES:
            w, t, l = _summarize(df, v)
            sub = df[df["variant_b"] == v]
            summary_rows.append(
                {
                    "baseline": v,
                    "cells": int(len(sub)),
                    "cha_s_wins": w,
                    "ties": t,
                    "cha_s_losses": l,
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "dm_test_pairwise_full_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        timer.add_output("summary_csv", str(summary_csv))

        log.info("Wrote DM table: %s  (rows=%d)", out_csv, len(df))
        log.info("Wrote per-baseline summary: %s", summary_csv)

    # ---- print headline summary (will be relayed by the harness) ----
    print()
    print(f"DM-test CHA-S vs Chronos-Bolt (all {n_bolt} cells):")
    print(f"  CHA-S significantly better: {wins}")
    print(f"  Statistical tie: {ties}")
    print(f"  CHA-S significantly worse: {losses}")
    print()
    print("Per-baseline win/tie/loss for CHA-S:")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
