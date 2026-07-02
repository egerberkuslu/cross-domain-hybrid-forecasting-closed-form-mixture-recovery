"""Metric collector for the SwarmInfer discrete-event simulator.

Collects per-request latency samples and policy-action counts, then
computes summary statistics at the end of a run.

All latencies are in milliseconds (float).

SLO definition: a request meets its SLO if its latency <= slo_threshold_ms.
slo_threshold_ms defaults to 2× the median latency of the run.  When too
few requests complete to compute a meaningful median, the SLO threshold
falls back to a fixed 2000 ms.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Minimum number of completed requests required before percentiles are
# meaningful. Below this threshold we still compute but log a warning.
_MIN_SAMPLES_FOR_STATS = 10


@dataclass
class RunMetrics:
    """Summary statistics produced after one simulation run.

    All latency fields are in milliseconds.  Rates are fractions in [0, 1].
    """

    # Request-level
    n_requests_total: int = 0
    n_requests_completed: int = 0
    n_requests_rejected: int = 0

    p50_latency_ms: float = float("nan")
    p95_latency_ms: float = float("nan")
    p99_latency_ms: float = float("nan")
    mean_latency_ms: float = float("nan")

    rejection_rate: float = float("nan")

    # SLO
    slo_threshold_ms: float = float("nan")
    slo_attainment: float = float("nan")  # fraction of completed requests <= SLO

    # Policy churn
    claim_count: int = 0
    release_count: int = 0

    # Simulation metadata
    simulated_duration_ms: float = 0.0
    wall_clock_seconds: float = 0.0
    policy_name: str = ""
    dataset_name: str = ""
    fleet_size: int = 0
    seed: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict suitable for JSON output."""
        return {
            "n_requests_total": self.n_requests_total,
            "n_requests_completed": self.n_requests_completed,
            "n_requests_rejected": self.n_requests_rejected,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "mean_latency_ms": self.mean_latency_ms,
            "rejection_rate": self.rejection_rate,
            "slo_threshold_ms": self.slo_threshold_ms,
            "slo_attainment": self.slo_attainment,
            "claim_count": self.claim_count,
            "release_count": self.release_count,
            "simulated_duration_ms": self.simulated_duration_ms,
            "wall_clock_seconds": self.wall_clock_seconds,
            "policy_name": self.policy_name,
            "dataset_name": self.dataset_name,
            "fleet_size": self.fleet_size,
            "seed": self.seed,
        }


class MetricCollector:
    """Accumulates per-request observations during a simulation run.

    Args:
        slo_multiplier: SLO threshold = slo_multiplier × median latency.
        fixed_slo_ms: If provided, use this as the SLO threshold instead of
            computing it from the empirical median.  Useful for ablations.
    """

    def __init__(
        self,
        slo_multiplier: float = 2.0,
        fixed_slo_ms: float | None = None,
    ) -> None:
        self._slo_mult = slo_multiplier
        self._fixed_slo_ms = fixed_slo_ms

        self._latencies_ms: list[float] = []
        self._n_rejected: int = 0
        self._n_total: int = 0
        self._claim_count: int = 0
        self._release_count: int = 0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_request(
        self,
        latency_ms: float | None,
        rejected: bool,
    ) -> None:
        """Record one request outcome.

        Args:
            latency_ms: End-to-end latency in ms, or None if rejected.
            rejected: Whether the request was rejected (no host found).
        """
        self._n_total += 1
        if rejected:
            self._n_rejected += 1
        elif latency_ms is not None and np.isfinite(latency_ms):
            self._latencies_ms.append(float(latency_ms))

    def record_claim(self) -> None:
        """Record one successful claim action."""
        self._claim_count += 1

    def record_release(self) -> None:
        """Record one successful release action."""
        self._release_count += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarise(
        self,
        simulated_duration_ms: float = 0.0,
        wall_clock_seconds: float = 0.0,
        policy_name: str = "",
        dataset_name: str = "",
        fleet_size: int = 0,
        seed: int = 0,
    ) -> RunMetrics:
        """Compute and return RunMetrics from accumulated observations.

        Args:
            simulated_duration_ms: Total virtual time simulated.
            wall_clock_seconds: Real elapsed time of the simulation.
            policy_name: Name of the policy that was evaluated.
            dataset_name: Workload dataset name.
            fleet_size: Number of nodes in the fleet.
            seed: RNG seed used.

        Returns:
            RunMetrics instance.
        """
        m = RunMetrics(
            n_requests_total=self._n_total,
            n_requests_rejected=self._n_rejected,
            n_requests_completed=len(self._latencies_ms),
            claim_count=self._claim_count,
            release_count=self._release_count,
            simulated_duration_ms=simulated_duration_ms,
            wall_clock_seconds=wall_clock_seconds,
            policy_name=policy_name,
            dataset_name=dataset_name,
            fleet_size=fleet_size,
            seed=seed,
        )

        if self._n_total > 0:
            m.rejection_rate = self._n_rejected / self._n_total
        else:
            m.rejection_rate = 0.0

        lats = np.array(self._latencies_ms, dtype=float)

        if len(lats) >= _MIN_SAMPLES_FOR_STATS:
            m.p50_latency_ms = float(np.percentile(lats, 50))
            m.p95_latency_ms = float(np.percentile(lats, 95))
            m.p99_latency_ms = float(np.percentile(lats, 99))
            m.mean_latency_ms = float(np.mean(lats))

            if self._fixed_slo_ms is not None:
                m.slo_threshold_ms = self._fixed_slo_ms
            else:
                m.slo_threshold_ms = self._slo_mult * float(np.median(lats))

            if m.slo_threshold_ms > 0:
                m.slo_attainment = float(np.mean(lats <= m.slo_threshold_ms))
            else:
                m.slo_attainment = 1.0

        elif len(lats) > 0:
            logger.warning(
                "[metrics] only %d latency samples — stats may be unreliable",
                len(lats),
            )
            m.p50_latency_ms = float(np.percentile(lats, 50))
            m.p95_latency_ms = float(np.percentile(lats, 95))
            m.p99_latency_ms = float(np.percentile(lats, 99))
            m.mean_latency_ms = float(np.mean(lats))
            threshold = self._fixed_slo_ms or (self._slo_mult * float(np.median(lats)))
            m.slo_threshold_ms = threshold
            m.slo_attainment = (
                float(np.mean(lats <= threshold)) if threshold > 0 else 1.0
            )

        return m

    @property
    def n_completed(self) -> int:
        return len(self._latencies_ms)

    @property
    def n_rejected(self) -> int:
        return self._n_rejected
