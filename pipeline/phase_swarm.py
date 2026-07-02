"""Phase: SwarmInfer FD-DSP simulator sweep.

Sweeps:
  fleet_size   ∈ {10, 25, 50}
  policy       ∈ {reactive, fd_dsp_seasonal_naive, fd_dsp_chronos_bolt,
                  fd_dsp_cha_hybrid_v3, oracle}
  workload     ∈ {burstgpt, azure_llm_2024, alibaba_pai}
  seed         ∈ {42, 123, 2024}

Total: 5 × 3 × 3 × 3 = 135 runs.

Each run replays 24–72 h of trace (configurable via --duration-hours).
Results are written to outputs/swarm_results/<label>.json and a combined
summary to outputs/swarm_results/summary.json.

Usage::

    # Quick smoke (synthetic workload, 1 hour):
    python pipeline/phase_swarm.py --smoke

    # Full sweep (real datasets, 24 h each):
    python pipeline/phase_swarm.py

    # Override duration:
    python pipeline/phase_swarm.py --duration-hours 48

    # Dry run (print plan, don't execute):
    python pipeline/phase_swarm.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure repo root is on sys.path when called as a script
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.swarm.cli import _json_default, run_single  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

FLEET_SIZES = [10, 25, 50]
SEEDS = [42, 123, 2024]
WORKLOADS = ["burstgpt", "azure_llm_2024", "alibaba_pai"]
DURATION_HOURS = 24  # default; overridable via --duration-hours

# Each policy entry: (cli_policy_name, cli_forecaster_name, display_label)
POLICIES: list[tuple[str, str, str]] = [
    ("reactive", "seasonal_naive", "reactive"),
    ("fd_dsp", "seasonal_naive", "fd_dsp_naive_forecaster"),
    ("fd_dsp", "chronos_bolt", "fd_dsp_chronos"),
    ("fd_dsp", "cha_hybrid_v3", "fd_dsp_cha_hybrid_v3"),
    ("fd_dsp_auto", "cha_hybrid_v3", "fd_dsp_auto_cha_hybrid_v3"),
    ("oracle", "seasonal_naive", "oracle"),
]

# Smoke sweep uses synthetic workload and only 1 run config
SMOKE_POLICIES: list[tuple[str, str, str]] = [
    ("reactive", "seasonal_naive", "reactive"),
    ("fd_dsp", "seasonal_naive", "fd_dsp_naive_forecaster"),
    ("oracle", "seasonal_naive", "oracle"),
]


# ---------------------------------------------------------------------------
# Run descriptor
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    fleet_size: int
    policy_name: str
    forecaster_name: str
    label: str
    dataset: str
    seed: int
    duration_hours: int

    @property
    def run_id(self) -> str:
        return f"{self.label}_n{self.fleet_size}_{self.dataset}_s{self.seed}"


def build_sweep(
    duration_hours: int,
    smoke: bool = False,
) -> list[RunSpec]:
    """Build the full list of RunSpec objects for the sweep."""
    policies = SMOKE_POLICIES if smoke else POLICIES
    fleet_sizes = [10] if smoke else FLEET_SIZES
    seeds = [42] if smoke else SEEDS
    workloads = ["synthetic"] if smoke else WORKLOADS
    dur = 1 if smoke else duration_hours

    specs: list[RunSpec] = []
    for fs in fleet_sizes:
        for pol_name, fc_name, label in policies:
            for ds in workloads:
                for sd in seeds:
                    specs.append(
                        RunSpec(
                            fleet_size=fs,
                            policy_name=pol_name,
                            forecaster_name=fc_name,
                            label=label,
                            dataset=ds,
                            seed=sd,
                            duration_hours=dur,
                        )
                    )
    return specs


# ---------------------------------------------------------------------------
# Sweep executor
# ---------------------------------------------------------------------------


def run_sweep(
    specs: list[RunSpec],
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
    skip_existing: bool = True,
) -> list[dict]:
    """Execute all RunSpecs, writing per-run JSON and a combined summary.

    Args:
        specs: List of RunSpec to execute.
        output_dir: Directory for per-run and summary JSON files.
        dry_run: If True, only print the plan without executing.
        verbose: Enable DEBUG logging inside run_single.
        skip_existing: Skip runs whose output file already exists.

    Returns:
        List of result dicts (one per completed run).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(specs)
    results: list[dict] = []

    logger.info("=== SwarmInfer FD-DSP sweep: %d runs ===", total)

    for idx, spec in enumerate(specs, start=1):
        out_path = output_dir / f"{spec.run_id}.json"

        if dry_run:
            logger.info(
                "[%d/%d] DRY-RUN  %s  (fleet=%d, dataset=%s, seed=%d, h=%d)",
                idx,
                total,
                spec.run_id,
                spec.fleet_size,
                spec.dataset,
                spec.seed,
                spec.duration_hours,
            )
            continue

        if skip_existing and out_path.exists():
            logger.info("[%d/%d] SKIP (exists): %s", idx, total, spec.run_id)
            try:
                with open(out_path) as fh:
                    results.append(json.load(fh))
            except Exception:
                pass
            continue

        logger.info(
            "[%d/%d] START  %s  fleet=%d  dataset=%s  seed=%d  h=%d",
            idx,
            total,
            spec.run_id,
            spec.fleet_size,
            spec.dataset,
            spec.seed,
            spec.duration_hours,
        )
        t0 = time.perf_counter()

        try:
            m = run_single(
                fleet_size=spec.fleet_size,
                policy_name=spec.policy_name,
                forecaster_name=spec.forecaster_name,
                dataset_name=spec.dataset,
                duration_hours=spec.duration_hours,
                seed=spec.seed,
                output_path=out_path,
                verbose=verbose,
            )
            elapsed = time.perf_counter() - t0
            results.append(m.to_dict())
            logger.info(
                "[%d/%d] DONE  %.1f s  p99=%.1f ms  rej=%.3f  slo=%.3f  "
                "claims=%d  releases=%d",
                idx,
                total,
                elapsed,
                m.p99_latency_ms,
                m.rejection_rate,
                m.slo_attainment,
                m.claim_count,
                m.release_count,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.error(
                "[%d/%d] FAILED  %.1f s  %s — %s",
                idx,
                total,
                elapsed,
                spec.run_id,
                exc,
            )
            # Write an error sentinel so we know what failed
            error_record = {
                "run_id": spec.run_id,
                "error": str(exc),
                "fleet_size": spec.fleet_size,
                "policy_name": spec.label,
                "dataset_name": spec.dataset,
                "seed": spec.seed,
            }
            results.append(error_record)
            with open(out_path.with_suffix(".error.json"), "w") as fh:
                json.dump(error_record, fh, indent=2)

    # Write combined summary
    if not dry_run and results:
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as fh:
            json.dump(results, fh, indent=2, default=_json_default)
        logger.info("Summary written to %s", summary_path)

        # Print a quick leaderboard to stdout
        _print_leaderboard(results)

    return results


def _print_leaderboard(results: list[dict]) -> None:
    """Print a compact leaderboard sorted by p99 latency."""
    valid = [r for r in results if "p99_latency_ms" in r and "error" not in r]
    if not valid:
        return

    print("\n=== SwarmInfer sweep leaderboard (sorted by p99 latency) ===")
    print(
        f"{'policy':<30} {'fleet':>5} {'dataset':<20} {'seed':>4} "
        f"{'p99 ms':>8} {'rej':>6} {'slo':>6} {'claims':>7}"
    )
    print("-" * 100)

    sorted_r = sorted(valid, key=lambda r: r.get("p99_latency_ms", float("inf")))
    for r in sorted_r[:20]:  # top-20
        print(
            f"{r.get('policy_name','?'):<30} "
            f"{r.get('fleet_size',0):>5} "
            f"{r.get('dataset_name','?'):<20} "
            f"{r.get('seed',0):>4} "
            f"{r.get('p99_latency_ms', float('nan')):>8.1f} "
            f"{r.get('rejection_rate', float('nan')):>6.3f} "
            f"{r.get('slo_attainment', float('nan')):>6.3f} "
            f"{r.get('claim_count', 0):>7}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python pipeline/phase_swarm.py",
        description="SwarmInfer FD-DSP parameter sweep",
    )
    p.add_argument(
        "--duration-hours",
        type=int,
        default=DURATION_HOURS,
        help=f"Hours of trace per run (default {DURATION_HOURS})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/swarm_results"),
        help="Output directory for per-run JSON files",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke: 3 policies × 1 fleet × synthetic × 1 seed = 3 runs",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sweep plan without executing",
    )
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-run even if output file already exists",
    )
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    specs = build_sweep(
        duration_hours=args.duration_hours,
        smoke=args.smoke,
    )

    run_sweep(
        specs=specs,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        verbose=args.verbose,
        skip_existing=not args.no_skip,
    )


if __name__ == "__main__":
    main()
