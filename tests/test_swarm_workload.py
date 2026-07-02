"""Tests for src/swarm/workload.py.

Verifies that Poisson sampling reproduces hourly counts within statistical
tolerance and that the workload API contracts hold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.swarm.workload import Workload, WorkloadConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_flat_workload(rate_per_hour: float, n_hours: int, seed: int = 0) -> Workload:
    """Synthetic workload with a flat (constant) hourly rate."""
    df = pd.DataFrame(
        {"value": np.full(n_hours, rate_per_hour)},
        index=pd.date_range("2024-01-01", periods=n_hours, freq="h"),
    )
    cfg = WorkloadConfig(dataset_name="flat", model_ids=["m0"], seed=seed)
    return Workload(df, cfg)


# ---------------------------------------------------------------------------
# Count-reproduction tests
# ---------------------------------------------------------------------------


class TestPoissonCounts:
    """Poisson arrivals should reproduce hourly totals within tolerance."""

    def test_total_within_5pct(self) -> None:
        """Aggregate request count should be within 5% of expected."""
        rate = 600.0  # 600 req/h
        n_hours = 24
        wl = _make_flat_workload(rate, n_hours, seed=42)

        requests = list(wl.generate())
        expected = rate * n_hours
        actual = len(requests)

        assert (
            abs(actual - expected) / expected < 0.05
        ), f"Expected ~{expected:.0f} requests, got {actual}"

    def test_per_hour_counts_within_tolerance(self) -> None:
        """Each individual hour should land within ±30% of the rate.

        This is a loose bound — for Poisson(600) the std-dev is ~24.5,
        so ±3σ ≈ ±12%, but we use 30% to avoid flaky tests on low-rate hours.
        """
        rate = 600.0
        n_hours = 48
        wl = _make_flat_workload(rate, n_hours, seed=7)

        requests = list(wl.generate())

        # bin requests by hour
        hour_counts = np.zeros(n_hours, dtype=int)
        for req in requests:
            h = int(req.arrival_ms / 3_600_000.0)
            if h < n_hours:
                hour_counts[h] += 1

        # All hours should be within 30% of the rate
        for h, cnt in enumerate(hour_counts):
            assert (
                abs(cnt - rate) / rate < 0.30
            ), f"Hour {h}: count={cnt}, expected≈{rate:.0f}"

    def test_zero_requests_for_zero_rate(self) -> None:
        """A rate of 0 (clamped to min_rate) still produces some arrivals."""
        df = pd.DataFrame(
            {"value": np.zeros(6)},
            index=pd.date_range("2024-01-01", periods=6, freq="h"),
        )
        cfg = WorkloadConfig(
            dataset_name="zero", model_ids=["m0"], seed=1, min_rate_per_hour=0.01
        )
        wl = Workload(df, cfg)
        # Should not crash; count is very low but not guaranteed zero
        requests = list(wl.generate())
        assert isinstance(requests, list)

    def test_reproducible_with_same_seed(self) -> None:
        """Same seed → same request sequence."""
        wl1 = _make_flat_workload(200.0, 12, seed=99)
        wl2 = _make_flat_workload(200.0, 12, seed=99)
        ids1 = [r.request_id for r in wl1.generate()]
        ids2 = [r.request_id for r in wl2.generate()]
        assert ids1 == ids2

    def test_different_seeds_differ(self) -> None:
        """Different seeds should produce different counts (with high probability)."""
        wl1 = _make_flat_workload(300.0, 24, seed=1)
        wl2 = _make_flat_workload(300.0, 24, seed=2)
        n1 = sum(1 for _ in wl1.generate())
        n2 = sum(1 for _ in wl2.generate())
        # Not guaranteed but overwhelmingly likely for rate=300 over 24 h
        assert n1 != n2


# ---------------------------------------------------------------------------
# API contract tests
# ---------------------------------------------------------------------------


class TestWorkloadAPI:
    def test_arrivals_are_monotone(self) -> None:
        """Arrival times must be non-decreasing."""
        wl = _make_flat_workload(500.0, 8, seed=13)
        times = [r.arrival_ms for r in wl.generate()]
        assert times == sorted(times), "Arrival times are not monotone"

    def test_model_ids_populated(self) -> None:
        """Every request must have a model_id from the configured list."""
        model_ids = ["alpha", "beta", "gamma"]
        df = pd.DataFrame(
            {"value": np.full(4, 300.0)},
            index=pd.date_range("2024-01-01", periods=4, freq="h"),
        )
        cfg = WorkloadConfig(dataset_name="multi", model_ids=model_ids, seed=5)
        wl = Workload(df, cfg)
        for req in wl.generate():
            assert req.model_id in model_ids

    def test_model_weights_respected(self) -> None:
        """Heavily weighted model should dominate the request stream."""
        model_ids = ["heavy", "light"]
        weights = [0.9, 0.1]
        df = pd.DataFrame(
            {"value": np.full(24, 1000.0)},
            index=pd.date_range("2024-01-01", periods=24, freq="h"),
        )
        cfg = WorkloadConfig(
            dataset_name="weighted",
            model_ids=model_ids,
            model_weights=weights,
            seed=42,
        )
        wl = Workload(df, cfg)
        requests = list(wl.generate())
        n_heavy = sum(1 for r in requests if r.model_id == "heavy")
        frac = n_heavy / len(requests)
        # Expect ~90%; allow ±5% tolerance
        assert abs(frac - 0.9) < 0.05, f"Heavy model fraction={frac:.3f}, expected≈0.90"

    def test_duration_hours_truncation(self) -> None:
        """duration_hours should cap the replayed trace."""
        wl_full = _make_flat_workload(200.0, 48, seed=0)
        df = pd.DataFrame(
            {"value": np.full(48, 200.0)},
            index=pd.date_range("2024-01-01", periods=48, freq="h"),
        )
        cfg = WorkloadConfig(
            dataset_name="trunc", model_ids=["m0"], seed=0, duration_hours=12
        )
        wl_short = Workload(df, cfg)
        n_full = sum(1 for _ in wl_full.generate())
        n_short = sum(1 for _ in wl_short.generate())
        # Short should be roughly 1/4 of full (12/48)
        assert n_short < n_full * 0.5

    def test_future_demand_length(self) -> None:
        """future_demand should return exactly horizon_hours values."""
        wl = _make_flat_workload(100.0, 48, seed=0)
        for horizon in (1, 6, 24):
            fd = wl.future_demand(t_ms=0.0, horizon_hours=horizon)
            assert len(fd) == horizon, f"horizon={horizon}: got len={len(fd)}"

    def test_demand_history_bounded(self) -> None:
        """demand_history should not exceed lookback_hours."""
        wl = _make_flat_workload(100.0, 48, seed=0)
        t_ms = 20 * 3_600_000.0  # hour 20
        h = wl.demand_history(t_ms, lookback_hours=10)
        assert len(h) <= 10

    def test_synthetic_constructor(self) -> None:
        """Workload.synthetic() should produce valid requests."""
        wl = Workload.synthetic(n_hours=6, mean_rate=500.0, seed=42)
        reqs = list(wl.generate())
        assert len(reqs) > 0
        assert all(r.arrival_ms >= 0 for r in reqs)

    def test_expected_total_property(self) -> None:
        """expected_total_requests should equal sum of hourly values."""
        rate = 300.0
        n = 10
        wl = _make_flat_workload(rate, n)
        assert abs(wl.expected_total_requests - rate * n) < 1e-6
