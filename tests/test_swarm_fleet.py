"""Tests for src/swarm/fleet.py.

Verifies heterogeneity of the generated fleet, shard placement logic,
replica counts, and memory accounting.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.swarm.fleet import Fleet, NodeState, make_fleet, _default_models, TIER_NAMES
from src.swarm.types import ModelSpec, NodeSpec, StageSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_fleet() -> Fleet:
    return make_fleet(n_nodes=10, seed=42)


@pytest.fixture
def medium_fleet() -> Fleet:
    return make_fleet(n_nodes=25, seed=42)


# ---------------------------------------------------------------------------
# Heterogeneity tests
# ---------------------------------------------------------------------------


class TestHeterogeneity:
    def test_multiple_tiers_present(self, medium_fleet: Fleet) -> None:
        """A 25-node fleet should span at least 3 distinct tiers."""
        tiers = {ns.spec.tier for ns in medium_fleet.nodes.values()}
        assert len(tiers) >= 3, f"Expected >=3 tiers, got {tiers}"

    def test_compute_capacity_varies(self, medium_fleet: Fleet) -> None:
        """Compute capacities should not all be identical."""
        capacities = [ns.spec.compute_cu_per_s for ns in medium_fleet.nodes.values()]
        assert len(set(capacities)) > 1, "All nodes have same compute capacity"

    def test_memory_varies(self, medium_fleet: Fleet) -> None:
        """Memory budgets should not all be identical."""
        mems = [ns.spec.memory_mib for ns in medium_fleet.nodes.values()]
        assert len(set(mems)) > 1, "All nodes have same memory"

    def test_tier_names_valid(self, medium_fleet: Fleet) -> None:
        """Every node tier must be one of the known tier names."""
        for ns in medium_fleet.nodes.values():
            assert ns.spec.tier in TIER_NAMES, f"Unknown tier: {ns.spec.tier}"

    def test_node_ids_unique(self, medium_fleet: Fleet) -> None:
        ids = list(medium_fleet.nodes.keys())
        assert len(ids) == len(set(ids)), "Duplicate node IDs"

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different tier assignments."""
        f1 = make_fleet(n_nodes=20, seed=1)
        f2 = make_fleet(n_nodes=20, seed=2)
        tiers1 = [ns.spec.tier for ns in f1.nodes.values()]
        tiers2 = [ns.spec.tier for ns in f2.nodes.values()]
        assert tiers1 != tiers2, "Different seeds produced identical fleets"

    def test_same_seed_reproducible(self) -> None:
        f1 = make_fleet(n_nodes=15, seed=77)
        f2 = make_fleet(n_nodes=15, seed=77)
        tiers1 = [ns.spec.tier for ns in f1.nodes.values()]
        tiers2 = [ns.spec.tier for ns in f2.nodes.values()]
        assert tiers1 == tiers2


# ---------------------------------------------------------------------------
# Initial shard placement tests
# ---------------------------------------------------------------------------


class TestInitialPlacement:
    def test_all_stages_have_replicas(self, small_fleet: Fleet) -> None:
        """Every stage type should have at least 1 replica after build."""
        for key in small_fleet.stage_registry:
            count = small_fleet.replica_counts.get(key, 0)
            assert count >= 1, f"Stage {key} has 0 replicas"

    def test_replica_counts_consistent_with_shards(self, small_fleet: Fleet) -> None:
        """replica_counts must equal the actual count of nodes holding each shard."""
        for key, expected in small_fleet.replica_counts.items():
            actual = sum(1 for ns in small_fleet.nodes.values() if key in ns.shards)
            assert (
                actual == expected
            ), f"Stage {key}: replica_counts={expected} but {actual} nodes hold it"

    def test_memory_not_overcommitted(self, small_fleet: Fleet) -> None:
        """No node should have negative free memory."""
        for node_id, ns in small_fleet.nodes.items():
            free = ns.memory_free_mib(small_fleet.stage_registry)
            assert free >= 0, f"Node {node_id} has negative free memory: {free} MiB"


# ---------------------------------------------------------------------------
# Claim / release mechanics
# ---------------------------------------------------------------------------


class TestClaimRelease:
    def _tiny_fleet(self) -> Fleet:
        """One big node + one small node, one single-stage model."""
        stage = StageSpec(
            stage_id="s0", model_id="m0", compute_tau_cu_ms=100.0, memory_mib=128
        )
        model = ModelSpec(model_id="m0", stages=(stage,))
        big_spec = NodeSpec(
            node_id="big",
            tier="mini_pc",
            compute_cu_per_s=128.0,
            memory_mib=8192,
            net_latency_ms=2.0,
        )
        tiny_spec = NodeSpec(
            node_id="tiny",
            tier="esp32",
            compute_cu_per_s=0.5,
            memory_mib=64,
            net_latency_ms=20.0,
        )
        nodes = {
            "big": NodeState(spec=big_spec),
            "tiny": NodeState(spec=tiny_spec),
        }
        sr = {("m0", "s0"): stage}
        return Fleet(nodes=nodes, models={"m0": model}, stage_registry=sr)

    def test_claim_succeeds_when_memory_available(self) -> None:
        fleet = self._tiny_fleet()
        ok = fleet.apply_claim("big", "m0", "s0")
        assert ok
        assert ("m0", "s0") in fleet.nodes["big"].shards

    def test_claim_fails_when_memory_insufficient(self) -> None:
        fleet = self._tiny_fleet()
        # tiny has only 64 MiB, stage needs 128 MiB
        ok = fleet.apply_claim("tiny", "m0", "s0")
        assert not ok
        assert ("m0", "s0") not in fleet.nodes["tiny"].shards

    def test_replica_count_increments_on_claim(self) -> None:
        fleet = self._tiny_fleet()
        before = fleet.replica_counts.get(("m0", "s0"), 0)
        fleet.apply_claim("big", "m0", "s0")
        after = fleet.replica_counts.get(("m0", "s0"), 0)
        assert after == before + 1

    def test_release_respects_min_replicas(self) -> None:
        fleet = self._tiny_fleet()
        fleet.apply_claim("big", "m0", "s0")
        # Only 1 replica → release should fail (min_replicas=2)
        ok = fleet.apply_release("big", "m0", "s0", min_replicas=2)
        assert not ok
        assert ("m0", "s0") in fleet.nodes["big"].shards

    def test_release_succeeds_above_min_replicas(self) -> None:
        """Release works when replica count > min_replicas."""
        fleet = self._tiny_fleet()
        # Add a second big node manually
        import copy

        big2_spec = NodeSpec(
            node_id="big2",
            tier="mini_pc",
            compute_cu_per_s=128.0,
            memory_mib=8192,
            net_latency_ms=2.0,
        )
        fleet.nodes["big2"] = NodeState(spec=big2_spec)
        fleet.apply_claim("big", "m0", "s0")
        fleet.apply_claim("big2", "m0", "s0")
        assert fleet.replica_counts.get(("m0", "s0"), 0) == 2

        ok = fleet.apply_release("big", "m0", "s0", min_replicas=1)
        assert ok
        assert ("m0", "s0") not in fleet.nodes["big"].shards
        assert fleet.replica_counts[("m0", "s0")] == 1

    def test_hosts_for_stage_accurate(self) -> None:
        fleet = self._tiny_fleet()
        fleet.apply_claim("big", "m0", "s0")
        hosts = fleet.hosts_for_stage("m0", "s0")
        assert "big" in hosts
        assert "tiny" not in hosts

    def test_least_loaded_host_picks_lowest_queue(self) -> None:
        fleet = self._tiny_fleet()
        # Add second big node
        big2_spec = NodeSpec(
            node_id="big2",
            tier="mini_pc",
            compute_cu_per_s=128.0,
            memory_mib=8192,
            net_latency_ms=2.0,
        )
        fleet.nodes["big2"] = NodeState(spec=big2_spec)
        fleet.apply_claim("big", "m0", "s0")
        fleet.apply_claim("big2", "m0", "s0")
        fleet.nodes["big"].queue_depth = 5
        fleet.nodes["big2"].queue_depth = 1
        host = fleet.least_loaded_host("m0", "s0")
        assert host == "big2"


# ---------------------------------------------------------------------------
# Default models
# ---------------------------------------------------------------------------


class TestDefaultModels:
    def test_three_models_returned(self) -> None:
        models = _default_models()
        assert len(models) == 3

    def test_model_ids_unique(self) -> None:
        models = _default_models()
        ids = [m.model_id for m in models]
        assert len(ids) == len(set(ids))

    def test_all_stages_have_positive_memory(self) -> None:
        for m in _default_models():
            for s in m.stages:
                assert s.memory_mib > 0
                assert s.compute_tau_cu_ms > 0
