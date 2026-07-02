"""Command-line interface for the SwarmInfer discrete-event simulator.

Single run::

    python -m src.swarm.cli \\
        --fleet-size 25 \\
        --policy fd_dsp \\
        --forecaster seasonal_naive \\
        --dataset burstgpt \\
        --duration-hours 24 \\
        --seed 42 \\
        --output outputs/swarm/run.json

Sweep via YAML config::

    python -m src.swarm.cli --sweep config/swarm_sweep.yaml \\
        --output-dir outputs/swarm/

The sweep YAML format::

    fleet_sizes: [10, 25, 50]
    policies:
      - name: reactive
      - name: fd_dsp
        forecaster: seasonal_naive
      - name: oracle
    datasets: [burstgpt, azure_llm_2024]
    seeds: [42, 123, 2024]
    duration_hours: 24
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .fleet import make_fleet
from .forecasters import make_forecaster
from .metrics import RunMetrics
from .policy import PolicyHParams, make_policy
from .simulator import Simulator, SimulatorConfig
from .workload import Workload, WorkloadConfig

logger = logging.getLogger(__name__)

# Models exposed to every run: 3 concurrent models sharing the fleet.
_DEFAULT_MODEL_IDS = ["llm_small", "vision_yolo", "enc_dec"]


# ---------------------------------------------------------------------------
# Single-run driver
# ---------------------------------------------------------------------------


def run_single(
    fleet_size: int,
    policy_name: str,
    forecaster_name: str,
    dataset_name: str,
    duration_hours: int,
    seed: int,
    output_path: Path | None,
    gossip_period_ms: float = 5_000.0,
    policy_period_ms: float = 60_000.0,
    sigma_up: float = 0.80,
    sigma_down: float = 0.40,
    forecast_horizon: int = 6,
    verbose: bool = False,
) -> RunMetrics:
    """Execute one simulation run and optionally write results to JSON.

    Args:
        fleet_size: Number of nodes in the heterogeneous fleet.
        policy_name: One of 'reactive', 'fd_dsp', 'oracle'.
        forecaster_name: Forecaster key from FORECASTER_REGISTRY
            (ignored for reactive / oracle).
        dataset_name: Workload dataset ('burstgpt', 'azure_llm_2024',
            'alibaba_pai', or 'synthetic' for offline tests).
        duration_hours: How many hours of trace to replay.
        seed: RNG seed (applied to fleet build, workload sampling, sim routing).
        output_path: If provided, write the JSON result dict here.
        gossip_period_ms: Gossip fan-out period in ms.
        policy_period_ms: Policy evaluation period in ms.
        sigma_up: Claim saturation threshold.
        sigma_down: Release saturation threshold.
        forecast_horizon: h-step lookahead for FD-DSP.
        verbose: If True, set log level to DEBUG.

    Returns:
        RunMetrics instance.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Build workload
    wl_cfg = WorkloadConfig(
        dataset_name=dataset_name,
        duration_hours=duration_hours,
        model_ids=_DEFAULT_MODEL_IDS,
        seed=seed,
    )

    if dataset_name == "synthetic":
        workload = Workload.synthetic(
            n_hours=duration_hours,
            mean_rate=1000.0,
            seed=seed,
            model_ids=_DEFAULT_MODEL_IDS,
        )
    elif dataset_name == "synthetic_ramp":
        workload = Workload.synthetic_ramp(
            n_hours=duration_hours,
            baseline_rate=5_000.0,
            ramp_peak_rate=200_000.0,
            ramp_start_hour=duration_hours // 3,
            ramp_duration_hours=max(4, duration_hours // 3),
            seed=seed,
            model_ids=_DEFAULT_MODEL_IDS,
        )
    else:
        workload = Workload.from_dataset(dataset_name, wl_cfg)

    # Build fleet (uses default 3-model set matching _DEFAULT_MODEL_IDS)
    fleet = make_fleet(n_nodes=fleet_size, seed=seed)

    # Build policy
    hp = PolicyHParams(
        sigma_up=sigma_up,
        sigma_down=sigma_down,
        forecast_horizon=forecast_horizon,
    )

    if policy_name in ("reactive", "oracle"):
        forecaster = None
    else:
        forecaster = make_forecaster(forecaster_name)

    policy = make_policy(policy_name, forecaster=forecaster, hparams=hp)

    # Build sim config
    sim_cfg = SimulatorConfig(
        gossip_period_ms=gossip_period_ms,
        policy_period_ms=policy_period_ms,
        seed=seed,
    )

    sim = Simulator(
        fleet=fleet,
        workload=workload,
        policy=policy,
        config=sim_cfg,
        dataset_name=dataset_name,
    )

    metrics = sim.run()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(metrics.to_dict(), fh, indent=2, default=_json_default)
        logger.info("[cli] results written to %s", output_path)

    return metrics


def _json_default(obj: object) -> object:
    """JSON serialiser for numpy scalars."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    raise TypeError(f"Not serialisable: {type(obj)}")


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------


def run_sweep(sweep_yaml: Path, output_dir: Path, verbose: bool = False) -> None:
    """Execute a parameter sweep defined in a YAML file.

    Args:
        sweep_yaml: Path to sweep configuration YAML.
        output_dir: Directory where per-run JSON results are written.
        verbose: Enable DEBUG logging.
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        logger.error("PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)

    with open(sweep_yaml) as fh:
        cfg = yaml.safe_load(fh)

    fleet_sizes: list[int] = cfg.get("fleet_sizes", [25])
    policies: list[dict] = cfg.get(
        "policies",
        [{"name": "reactive"}, {"name": "fd_dsp", "forecaster": "seasonal_naive"}],
    )
    datasets: list[str] = cfg.get("datasets", ["burstgpt"])
    seeds: list[int] = cfg.get("seeds", [42])
    duration_hours: int = cfg.get("duration_hours", 24)

    total = len(fleet_sizes) * len(policies) * len(datasets) * len(seeds)
    logger.info("[sweep] %d runs planned", total)

    all_results: list[dict] = []
    run_idx = 0

    for fs in fleet_sizes:
        for pol in policies:
            pol_name = pol["name"]
            fc_name = pol.get("forecaster", "seasonal_naive")
            for ds in datasets:
                for sd in seeds:
                    run_idx += 1
                    label = f"{pol_name}_{fc_name}_n{fs}_{ds}_s{sd}"
                    out_path = output_dir / f"{label}.json"

                    if out_path.exists():
                        logger.info(
                            "[sweep] %d/%d SKIP (exists): %s", run_idx, total, label
                        )
                        with open(out_path) as fh:
                            all_results.append(json.load(fh))
                        continue

                    logger.info("[sweep] %d/%d START: %s", run_idx, total, label)
                    try:
                        m = run_single(
                            fleet_size=fs,
                            policy_name=pol_name,
                            forecaster_name=fc_name,
                            dataset_name=ds,
                            duration_hours=duration_hours,
                            seed=sd,
                            output_path=out_path,
                            verbose=verbose,
                        )
                        all_results.append(m.to_dict())
                        logger.info(
                            "[sweep] %d/%d DONE  p99=%.1f ms  rej=%.3f  slo=%.3f",
                            run_idx,
                            total,
                            m.p99_latency_ms,
                            m.rejection_rate,
                            m.slo_attainment,
                        )
                    except Exception as exc:
                        logger.error(
                            "[sweep] %d/%d FAILED: %s — %s", run_idx, total, label, exc
                        )

    # Write combined results table
    combined_path = output_dir / "sweep_results.json"
    with open(combined_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=_json_default)
    logger.info("[sweep] combined results -> %s", combined_path)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.swarm.cli",
        description="SwarmInfer FD-DSP discrete-event simulator",
    )
    p.add_argument("--fleet-size", type=int, default=25, help="Number of nodes")
    p.add_argument(
        "--policy",
        default="reactive",
        choices=["reactive", "fd_dsp", "fd_dsp_auto", "oracle"],
        help="Placement policy",
    )
    p.add_argument(
        "--forecaster",
        default="seasonal_naive",
        help="Forecaster name (used when --policy fd_dsp)",
    )
    p.add_argument(
        "--dataset",
        default="burstgpt",
        help="Workload dataset name (or 'synthetic' for offline tests)",
    )
    p.add_argument(
        "--duration-hours",
        type=int,
        default=24,
        help="Hours of trace to replay",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--output", type=Path, default=None, help="Output JSON path")
    p.add_argument(
        "--gossip-period-ms",
        type=float,
        default=5_000.0,
        help="Gossip period in ms",
    )
    p.add_argument(
        "--policy-period-ms",
        type=float,
        default=60_000.0,
        help="Policy evaluation period in ms",
    )
    p.add_argument("--sigma-up", type=float, default=0.80)
    p.add_argument("--sigma-down", type=float, default=0.40)
    p.add_argument("--forecast-horizon", type=int, default=6)
    p.add_argument("--verbose", action="store_true")

    # Sweep mode
    p.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="Path to sweep YAML config (enables sweep mode)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/swarm"),
        help="Output directory for sweep results",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for ``python -m src.swarm.cli``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.sweep is not None:
        run_sweep(args.sweep, args.output_dir, verbose=args.verbose)
        return

    metrics = run_single(
        fleet_size=args.fleet_size,
        policy_name=args.policy,
        forecaster_name=args.forecaster,
        dataset_name=args.dataset,
        duration_hours=args.duration_hours,
        seed=args.seed,
        output_path=args.output,
        gossip_period_ms=args.gossip_period_ms,
        policy_period_ms=args.policy_period_ms,
        sigma_up=args.sigma_up,
        sigma_down=args.sigma_down,
        forecast_horizon=args.forecast_horizon,
        verbose=args.verbose,
    )

    # Always print to stdout regardless of --output
    print(json.dumps(metrics.to_dict(), indent=2, default=_json_default))


if __name__ == "__main__":
    main()
