"""Phase-1 driver: download + parse + cache + explore all three datasets."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

from src.data_loaders import build_loader
from src.data_loaders.explore import plot_overview, report_one, write_summaries
from src.utils import load_config, setup_logging, set_global_seed
from src.utils.logging_setup import get_logger


def main() -> int:
    cfg = load_config()
    log_file = setup_logging(log_dir=cfg.resolve(cfg.paths.logs), run_name="phase1_data")
    log = get_logger("phase1")
    set_global_seed(cfg.random_seed)
    log.info("Phase 1 driver started. Log: %s", log_file)

    reports = []
    for name in cfg.datasets.keys():
        log.info("=" * 72)
        log.info("Dataset: %s", name)
        log.info("=" * 72)
        loader = build_loader(name, cfg)
        df = loader.run()

        # head / tail
        log.info("[%s] head:\n%s", name, df.head().to_string())
        log.info("[%s] tail:\n%s", name, df.tail().to_string())
        log.info("[%s] describe:\n%s", name, df["value"].describe().to_string())

        # report
        rep = report_one(name, df, expected_freq=cfg.datasets[name].resample_to)
        log.info("[%s] report: %s", name, rep)
        reports.append(rep)

        # plot
        figs_dir = cfg.resolve(cfg.paths.figures)
        out = plot_overview(name, df, figs_dir / f"{name}_overview.png")
        log.info("[%s] plot saved: %s", name, out)

    # consolidated summary
    results_dir = cfg.resolve(cfg.paths.results)
    write_summaries(
        reports,
        csv_path=results_dir / "data_summary.csv",
        json_path=results_dir / "data_summary.json",
    )
    log.info("Wrote consolidated summary to %s", results_dir / "data_summary.csv")
    log.info("Phase 1 done. %d datasets processed.", len(reports))
    return 0


if __name__ == "__main__":
    sys.exit(main())
