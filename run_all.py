"""End-to-end reproducibility entry point.

Runs the full pipeline:
    Phase 1 — data acquisition / processing
    Phase 2 — preprocessing (split + scale + window)
    Phase 3 — train + tune all baselines and foundation models
    Phase 4 — train the proposed CHA-Hybrid
    Phase 5 — full experiment grid (datasets x models x horizons x seeds)
    Phase 6 — evaluation (metrics, DM-test, ablation, cost)
    Phase 7 — figures, consolidated tables, README finalisation

NOTE: subsequent phases will progressively flesh out the bodies of the
``stage_*`` functions below. Phase 0 only wires the skeleton.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils import (
    detect_device,
    load_config,
    log_device_info,
    set_global_seed,
    setup_logging,
)
from src.utils.logging_setup import get_logger


# ----------------------------------------------------------------------
# Stages — each is implemented in its own phase. For Phase 0 they are
# all stubs that log "not yet implemented" so the runner is still wired.
# ----------------------------------------------------------------------


def stage_data_acquisition(cfg) -> None:
    """Phase 1: download + parse + cache + explore all three datasets."""
    from src.data_loaders import build_loader
    from src.data_loaders.explore import plot_overview, report_one, write_summaries

    log = get_logger("phase1")
    reports = []
    figs_dir = cfg.resolve(cfg.paths.figures)
    results_dir = cfg.resolve(cfg.paths.results)
    for name in cfg.datasets.keys():
        loader = build_loader(name, cfg)
        df = loader.run()
        rep = report_one(name, df, expected_freq=cfg.datasets[name].resample_to)
        plot_overview(name, df, figs_dir / f"{name}_overview.png")
        log.info(
            "[%s] %d rows | span_days=%.1f | missing_pct=%.3f%% | "
            "extreme_outliers=%d",
            name,
            rep.n_rows,
            rep.span_days,
            rep.missing_pct,
            rep.n_extreme_outliers_flagged,
        )
        reports.append(rep)
    write_summaries(
        reports,
        csv_path=results_dir / "data_summary.csv",
        json_path=results_dir / "data_summary.json",
    )
    log.info("Phase 1 complete. Summary: %s", results_dir / "data_summary.csv")


def stage_preprocessing(cfg) -> None:
    """Phase 2: interpolate → chronological split → train-fit scaler → window."""
    from src.preprocessing import preprocess_all

    log = get_logger("phase2")
    processed = preprocess_all(cfg)
    for name, pp in processed.items():
        sizes = pp.split_native.sizes
        log.info(
            "[%s] splits: train=%d val=%d test=%d | windows per h: %s",
            name,
            sizes["train"],
            sizes["val"],
            sizes["test"],
            {h: pp.windows[h]["train"].X.shape[0] for h in pp.horizons},
        )
    log.info("Phase 2 complete: %d datasets preprocessed.", len(processed))


def stage_baselines(cfg) -> None:
    """Phase 3: smoke-test every registered model on one dataset at h=1 and h=24."""
    # Delegate to the dedicated driver — runs every model in REGISTRY plus the
    # chronos fine-tune variant, persists HPs under results/hyperparameters/smoke/.
    import subprocess, sys

    log = get_logger("phase3")
    log.info("Launching phase3 smoke test (pipeline/phase3_smoke.py) ...")
    rc = subprocess.call([sys.executable, "pipeline/phase3_smoke.py"])
    if rc == 0:
        log.info("Phase 3 smoke test passed.")
    else:
        log.error("Phase 3 smoke test FAILED with exit code %d", rc)


def stage_proposed(cfg) -> None:
    """Phase 4: smoke + verification for the proposed CHA-Hybrid."""
    import subprocess, sys

    log = get_logger("phase4")
    log.info("Launching phase4 driver (pipeline/phase4_cha_hybrid.py) ...")
    rc = subprocess.call([sys.executable, "pipeline/phase4_cha_hybrid.py"])
    if rc == 0:
        log.info("Phase 4 verification passed.")
    else:
        log.error("Phase 4 verification FAILED with exit code %d", rc)


def stage_full_grid(cfg) -> None:
    """Phase 5: smoke + main runner for dataset × model × horizon × seed grid."""
    import subprocess, sys

    log = get_logger("phase5")
    log.info("Phase 5 smoke ...")
    rc = subprocess.call([sys.executable, "pipeline/phase5_smoke.py"])
    if rc != 0:
        log.error("Phase 5 smoke FAILED (rc=%d) — not launching main run.", rc)
        return
    log.info(
        "Phase 5 main full grid (resumable; --max-seeds 5 honours config seeds) ..."
    )
    subprocess.call([sys.executable, "pipeline/phase5_main.py"])


def stage_evaluation(cfg) -> None:
    """Phase 6: metrics aggregation + DM-test + ablation + cost."""
    import subprocess, sys

    log = get_logger("phase6")
    log.info("Launching phase6 driver (pipeline/phase6_eval.py) ...")
    rc = subprocess.call([sys.executable, "pipeline/phase6_eval.py"])
    if rc == 0:
        log.info("Phase 6 evaluation done.")
    else:
        log.error("Phase 6 evaluation FAILED with exit code %d", rc)


def stage_outputs(cfg) -> None:
    get_logger(__name__).info("Phase 7 (outputs / figures) — not yet implemented.")


STAGES = {
    "data": stage_data_acquisition,
    "preprocess": stage_preprocessing,
    "baselines": stage_baselines,
    "proposed": stage_proposed,
    "grid": stage_full_grid,
    "evaluate": stage_evaluation,
    "outputs": stage_outputs,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CHA-Hybrid end-to-end pipeline driver.")
    p.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the project YAML config (default: config/config.yaml).",
    )
    p.add_argument(
        "--stage",
        choices=list(STAGES.keys()) + ["all"],
        default="all",
        help="Run a single stage or the whole pipeline (default: all).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default="pipeline",
        help="Run name used in the log filename.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log_file = setup_logging(log_dir=Path("outputs/logs"), run_name=args.run_name)
    log = get_logger("run_all")
    log.info("=" * 72)
    log.info("CHA-Hybrid pipeline driver started")
    log.info("Config: %s", args.config)
    log.info("Stage : %s", args.stage)
    log.info("Log   : %s", log_file)
    log.info("=" * 72)

    cfg = load_config(args.config)
    set_global_seed(cfg.random_seed)
    log_device_info(detect_device())

    stages = list(STAGES.values()) if args.stage == "all" else [STAGES[args.stage]]
    for stage in stages:
        stage(cfg)

    log.info("Pipeline driver finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
