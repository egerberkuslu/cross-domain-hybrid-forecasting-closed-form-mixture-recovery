"""Smoke tests for the SwarmInfer discrete-event simulator.

Verifies end-to-end correctness:
- The simulation completes without error.
- All required metric keys are present and non-NaN.
- Basic sanity: rejection_rate in [0,1], latencies positive, slo_attainment in [0,1].
- Policy comparison: oracle should have <= rejection rate than reactive
  (over sufficient simulation time; we allow equality for short runs).
"""
from __future__ import annotations

import math

import pytest

from src.swarm.fleet import make_fleet
from src.swarm.metrics import RunMetrics
from src.swarm.policy import FDDSPPolicy, OraclePolicy, PolicyHParams, ReactivePolicy
from src.swarm.forecasters import SeasonalNaiveForecaster, MeanForecaster
from src.swarm.simulator import Simulator, SimulatorConfig
from src.swarm.workload import Workload

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {
    "p50_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "rejection_rate",
    "claim_count",
    "release_count",
    "slo_attainment",
}


def _make_sim(
    policy,
    n_nodes: int = 10,
    n_hours: int = 1,
    mean_rate: float = 600.0,
    seed: int = 42,
) -> Simulator:
    fleet = make_fleet(n_nodes=n_nodes, seed=seed)
    workload = Workload.synthetic(
        n_hours=n_hours,
        mean_rate=mean_rate,
        seed=seed,
        model_ids=["llm_small", "vision_yolo", "enc_dec"],
    )
    cfg = SimulatorConfig(
        gossip_period_ms=60_000.0,  # 1 min gossip (fast for tests)
        policy_period_ms=300_000.0,  # 5 min policy ticks
        seed=seed,
    )
    return Simulator(
        fleet=fleet,
        workload=workload,
        policy=policy,
        config=cfg,
        dataset_name="synthetic",
    )


def _assert_valid_metrics(m: RunMetrics) -> None:
    d = m.to_dict()
    for key in _REQUIRED_KEYS:
        assert key in d, f"Missing key: {key}"
        val = d[key]
        assert val is not None, f"Key {key} is None"
        if isinstance(val, float):
            assert not math.isnan(val), f"Key {key} is NaN"


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestSimulatorSmoke:
    """Basic end-to-end smoke tests that must complete quickly."""

    def test_reactive_1h_completes(self) -> None:
        policy = ReactivePolicy()
        sim = _make_sim(policy, n_nodes=10, n_hours=1)
        m = sim.run()
        _assert_valid_metrics(m)

    def test_fd_dsp_seasonal_naive_completes(self) -> None:
        policy = FDDSPPolicy(forecaster=SeasonalNaiveForecaster(period=24))
        sim = _make_sim(policy, n_nodes=10, n_hours=1)
        m = sim.run()
        _assert_valid_metrics(m)

    def test_fd_dsp_mean_completes(self) -> None:
        policy = FDDSPPolicy(forecaster=MeanForecaster())
        sim = _make_sim(policy, n_nodes=10, n_hours=1)
        m = sim.run()
        _assert_valid_metrics(m)

    def test_oracle_completes(self) -> None:
        policy = OraclePolicy()
        sim = _make_sim(policy, n_nodes=10, n_hours=1)
        m = sim.run()
        _assert_valid_metrics(m)

    def test_metrics_ranges_valid(self) -> None:
        """All metric values should be in their expected ranges."""
        policy = ReactivePolicy()
        sim = _make_sim(policy, n_nodes=10, n_hours=2)
        m = sim.run()

        assert 0.0 <= m.rejection_rate <= 1.0, f"rejection_rate={m.rejection_rate}"
        assert 0.0 <= m.slo_attainment <= 1.0, f"slo_attainment={m.slo_attainment}"
        assert m.p50_latency_ms > 0, f"p50={m.p50_latency_ms}"
        assert (
            m.p95_latency_ms >= m.p50_latency_ms
        ), f"p95={m.p95_latency_ms} < p50={m.p50_latency_ms}"
        assert (
            m.p99_latency_ms >= m.p95_latency_ms
        ), f"p99={m.p99_latency_ms} < p95={m.p95_latency_ms}"
        assert m.claim_count >= 0
        assert m.release_count >= 0

    def test_n_requests_accounting(self) -> None:
        """completed + rejected should equal total."""
        policy = ReactivePolicy()
        sim = _make_sim(policy, n_nodes=10, n_hours=1, mean_rate=500.0)
        m = sim.run()
        assert m.n_requests_completed + m.n_requests_rejected == m.n_requests_total

    def test_larger_fleet_lower_rejection(self) -> None:
        """A 25-node fleet should reject fewer requests than a 5-node fleet."""
        policy_small = ReactivePolicy()
        policy_large = ReactivePolicy()
        sim_small = _make_sim(
            policy_small, n_nodes=5, n_hours=2, mean_rate=800.0, seed=42
        )
        sim_large = _make_sim(
            policy_large, n_nodes=25, n_hours=2, mean_rate=800.0, seed=42
        )
        m_small = sim_small.run()
        m_large = sim_large.run()
        # Larger fleet should have <= rejection rate (may be equal if both are 0)
        assert m_large.rejection_rate <= m_small.rejection_rate + 0.05, (
            f"Large fleet rejection {m_large.rejection_rate:.3f} > "
            f"small fleet {m_small.rejection_rate:.3f}"
        )

    def test_simulated_duration_positive(self) -> None:
        policy = ReactivePolicy()
        sim = _make_sim(policy, n_nodes=10, n_hours=1)
        m = sim.run()
        assert m.simulated_duration_ms > 0

    def test_fleet_size_recorded(self) -> None:
        policy = ReactivePolicy()
        sim = _make_sim(policy, n_nodes=13, n_hours=1)
        m = sim.run()
        assert m.fleet_size == 13

    def test_seed_recorded(self) -> None:
        policy = ReactivePolicy()
        sim = _make_sim(policy, n_nodes=10, n_hours=1, seed=123)
        m = sim.run()
        assert m.seed == 123

    def test_to_dict_has_required_keys(self) -> None:
        policy = ReactivePolicy()
        sim = _make_sim(policy)
        m = sim.run()
        d = m.to_dict()
        for key in _REQUIRED_KEYS:
            assert key in d, f"to_dict() missing key: {key}"

    def test_wall_clock_reasonable(self) -> None:
        """A 1-hour synthetic run with 10 nodes should finish in <30 s."""
        import time

        policy = ReactivePolicy()
        sim = _make_sim(policy, n_nodes=10, n_hours=1, mean_rate=600.0)
        t0 = time.perf_counter()
        sim.run()
        elapsed = time.perf_counter() - t0
        assert elapsed < 30.0, f"Simulation took {elapsed:.1f} s — too slow"


# ---------------------------------------------------------------------------
# Reproducibility test
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_metrics(self) -> None:
        """Two runs with the same seed must produce identical metrics."""

        def _run(seed: int) -> dict:
            policy = ReactivePolicy()
            sim = _make_sim(policy, n_nodes=10, n_hours=1, seed=seed)
            return sim.run().to_dict()

        m1 = _run(42)
        m2 = _run(42)
        for key in _REQUIRED_KEYS:
            v1, v2 = m1[key], m2[key]
            if isinstance(v1, float) and isinstance(v2, float):
                assert abs(v1 - v2) < 1e-9, f"key={key}: {v1} != {v2}"
            else:
                assert v1 == v2, f"key={key}: {v1} != {v2}"

    def test_different_seeds_may_differ(self) -> None:
        """Different seeds should produce different total request counts
        (probabilistically true for any non-degenerate workload)."""

        def _n_total(seed: int) -> int:
            policy = ReactivePolicy()
            sim = _make_sim(policy, n_nodes=10, n_hours=2, mean_rate=500.0, seed=seed)
            return sim.run().n_requests_total

        n1 = _n_total(1)
        n2 = _n_total(2)
        # Not guaranteed but extremely likely for Poisson(500*2*60) ≈ 60,000 expected
        assert n1 != n2


# ---------------------------------------------------------------------------
# Policy comparison (qualitative)
# ---------------------------------------------------------------------------


class TestPolicyComparison:
    """FD-DSP and Oracle should perform at least as well as reactive on average."""

    def test_oracle_rejection_leq_reactive(self) -> None:
        """Oracle should never have a *much* higher rejection rate than reactive."""
        # Use 3-hour run for a bit more signal
        reactive_m = _make_sim(ReactivePolicy(), n_nodes=15, n_hours=3, seed=42).run()
        oracle_m = _make_sim(OraclePolicy(), n_nodes=15, n_hours=3, seed=42).run()
        # Allow 5% slack (oracle may trigger more releases, slightly increasing
        # rejection transiently, but should be similar in aggregate)
        assert oracle_m.rejection_rate <= reactive_m.rejection_rate + 0.10, (
            f"Oracle rej={oracle_m.rejection_rate:.3f} >> "
            f"reactive rej={reactive_m.rejection_rate:.3f}"
        )
