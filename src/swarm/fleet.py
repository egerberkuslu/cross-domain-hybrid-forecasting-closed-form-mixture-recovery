"""Fleet builder: creates N heterogeneous nodes across hardware tiers.

Device tiers are loosely inspired by SwarmInfer §3.3 device classes.
Each tier has a fixed compute capacity (CU/s), memory (MiB), and
network latency profile.  The fleet builder assigns nodes to tiers
with a configurable mix ratio, then seeds their initial shard
assignments with a simple round-robin across models and stages.

Typical usage::

    fleet = make_fleet(n_nodes=25, seed=42)
    node = fleet.nodes["node_04"]
    print(node.spec.tier, node.memory_free_mib)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .types import ModelSpec, NodeSpec, StageSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware tier definitions
# ---------------------------------------------------------------------------

# (tier_name, compute_cu_per_s, memory_mib, net_latency_ms, weight_in_fleet)
_TIER_DEFS: list[tuple[str, float, int, float, float]] = [
    # tier          CU/s   RAM MiB   net ms   fleet weight
    ("esp32", 0.5, 256, 20.0, 0.05),
    ("rpi4", 4.0, 3_072, 8.0, 0.30),
    ("jetson_nano", 16.0, 4_096, 5.0, 0.35),
    ("jetson_agx", 64.0, 16_384, 3.0, 0.20),
    ("mini_pc", 128.0, 32_768, 2.0, 0.10),
]

TIER_NAMES = [t[0] for t in _TIER_DEFS]
_TIER_MAP: dict[str, tuple[float, int, float]] = {
    name: (cu, mem, lat) for name, cu, mem, lat, _ in _TIER_DEFS
}
_TIER_WEIGHTS = np.array([t[4] for t in _TIER_DEFS], dtype=float)
_TIER_WEIGHTS /= _TIER_WEIGHTS.sum()


# ---------------------------------------------------------------------------
# Node runtime state
# ---------------------------------------------------------------------------


@dataclass
class NodeState:
    """Mutable runtime state of one node in the simulated fleet.

    Args:
        spec: Immutable hardware description.
        shards: Set of (model_id, stage_id) tuples currently loaded.
        queue_depth: Number of requests currently being processed or queued.
        demand_cache: Per-model EWMA of observed requests/s (for reactive
            policy and gossip).
        last_claim_release_ms: Per-(model_id, stage_id) last action time (ms),
            used for hysteresis enforcement.
    """

    spec: NodeSpec
    shards: set[tuple[str, str]] = field(default_factory=set)
    queue_depth: int = 0
    demand_cache: dict[str, float] = field(default_factory=dict)
    last_claim_release_ms: dict[tuple[str, str], float] = field(default_factory=dict)

    # Running totals for metrics
    total_requests_served: int = 0
    total_compute_ms: float = 0.0

    @property
    def memory_used_mib(self) -> int:
        """Sum of memory occupied by currently loaded shards."""
        return self._mem_used

    def set_memory_used(self, value: int) -> None:
        self._mem_used = value

    def __post_init__(self) -> None:
        self._mem_used: int = 0

    def memory_free_mib(self, all_stages: dict[tuple[str, str], StageSpec]) -> int:
        """Available memory given current shards and a stage registry."""
        used = sum(all_stages[k].memory_mib for k in self.shards if k in all_stages)
        return self.spec.memory_mib - used

    def has_shard(self, model_id: str, stage_id: str) -> bool:
        return (model_id, stage_id) in self.shards

    def load(
        self,
        model_id: str,
        stage_id: str,
        stage: StageSpec,
        all_stages: dict[tuple[str, str], StageSpec],
    ) -> bool:
        """Attempt to load a shard.  Returns True if successful."""
        free = self.memory_free_mib(all_stages)
        if stage.memory_mib > free:
            return False
        self.shards.add((model_id, stage_id))
        return True

    def unload(self, model_id: str, stage_id: str) -> bool:
        """Unload a shard.  Returns True if it was present."""
        key = (model_id, stage_id)
        if key in self.shards:
            self.shards.discard(key)
            return True
        return False

    def update_demand(self, model_id: str, new_obs: float, alpha: float = 0.1) -> None:
        """Update EWMA demand estimate for a model.

        Args:
            model_id: Model identifier.
            new_obs: New observed request rate (requests/s).
            alpha: EWMA smoothing factor (higher = more reactive).
        """
        prev = self.demand_cache.get(model_id, new_obs)
        self.demand_cache[model_id] = alpha * new_obs + (1.0 - alpha) * prev


# ---------------------------------------------------------------------------
# Fleet container
# ---------------------------------------------------------------------------


@dataclass
class Fleet:
    """Collection of nodes with shared model / stage registry.

    Args:
        nodes: Ordered dict of node_id -> NodeState.
        models: Dict of model_id -> ModelSpec.
        stage_registry: Flat dict of (model_id, stage_id) -> StageSpec.
        replica_counts: Current number of replicas per (model_id, stage_id).
    """

    nodes: dict[str, NodeState]
    models: dict[str, ModelSpec]
    stage_registry: dict[tuple[str, str], StageSpec]
    replica_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Initialise replica counts from node shards
        self._refresh_replica_counts()

    def _refresh_replica_counts(self) -> None:
        counts: dict[tuple[str, str], int] = {}
        for node in self.nodes.values():
            for key in node.shards:
                counts[key] = counts.get(key, 0) + 1
        self.replica_counts = counts

    def node_list(self) -> list[str]:
        return list(self.nodes.keys())

    def hosts_for_stage(self, model_id: str, stage_id: str) -> list[str]:
        """Return sorted list of node IDs that currently host a stage."""
        key = (model_id, stage_id)
        return [nid for nid, ns in self.nodes.items() if key in ns.shards]

    def least_loaded_host(self, model_id: str, stage_id: str) -> str | None:
        """Pick the host for a stage with the smallest queue depth.

        Returns None if no node hosts the stage.
        """
        hosts = self.hosts_for_stage(model_id, stage_id)
        if not hosts:
            return None
        return min(hosts, key=lambda nid: self.nodes[nid].queue_depth)

    def apply_claim(self, node_id: str, model_id: str, stage_id: str) -> bool:
        """Claim a stage on a node if memory allows.

        Returns True on success.
        """
        node = self.nodes[node_id]
        key = (model_id, stage_id)
        if key in node.shards:
            return True  # already held
        stage = self.stage_registry.get(key)
        if stage is None:
            return False
        ok = node.load(model_id, stage_id, stage, self.stage_registry)
        if ok:
            self.replica_counts[key] = self.replica_counts.get(key, 0) + 1
        return ok

    def apply_release(
        self, node_id: str, model_id: str, stage_id: str, min_replicas: int = 2
    ) -> bool:
        """Release a stage from a node if durability floor is met.

        Returns True on success.
        """
        key = (model_id, stage_id)
        current = self.replica_counts.get(key, 0)
        if current <= min_replicas:
            return False  # would drop below durability floor
        ok = self.nodes[node_id].unload(model_id, stage_id)
        if ok:
            self.replica_counts[key] = max(0, current - 1)
        return True


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def make_fleet(
    n_nodes: int = 25,
    models: Sequence[ModelSpec] | None = None,
    seed: int = 42,
    tier_weights: np.ndarray | None = None,
    initial_replicas_per_stage: int = 3,
) -> Fleet:
    """Build a heterogeneous fleet of N nodes with default model set.

    Args:
        n_nodes: Total number of nodes to create.
        models: List of ModelSpec to populate the fleet with.  If None,
            uses a default set of 3 toy models (suitable for smoke tests).
        seed: RNG seed for tier assignment and initial shard placement.
        tier_weights: Optional override for tier sampling probabilities
            (length-5 array matching _TIER_DEFS).  Defaults to the
            pre-defined fleet mix.
        initial_replicas_per_stage: How many nodes should initially host
            each stage (round-robin assignment during build).

    Returns:
        Fleet instance ready for simulation.
    """
    rng = np.random.default_rng(seed)

    if models is None:
        models = _default_models()

    weights = tier_weights if tier_weights is not None else _TIER_WEIGHTS
    tier_choices = rng.choice(len(_TIER_DEFS), size=n_nodes, p=weights)

    nodes: dict[str, NodeState] = {}
    for i, tier_idx in enumerate(tier_choices):
        tname, cu, mem, lat, _ = _TIER_DEFS[tier_idx]
        node_id = f"node_{i:03d}"
        spec = NodeSpec(
            node_id=node_id,
            tier=tname,
            compute_cu_per_s=cu,
            memory_mib=mem,
            net_latency_ms=lat,
        )
        nodes[node_id] = NodeState(spec=spec)

    model_dict = {m.model_id: m for m in models}
    stage_registry: dict[tuple[str, str], StageSpec] = {}
    for m in models:
        for s in m.stages:
            stage_registry[(m.model_id, s.stage_id)] = s

    fleet = Fleet(
        nodes=nodes,
        models=model_dict,
        stage_registry=stage_registry,
    )

    # Initial shard placement: round-robin replicas across nodes sorted by
    # descending memory so large nodes fill first.
    node_ids_sorted = sorted(
        nodes.keys(),
        key=lambda nid: nodes[nid].spec.memory_mib,
        reverse=True,
    )
    assignment_cursor = 0
    for m in models:
        for stage in m.stages:
            key = (m.model_id, stage.stage_id)
            placed = 0
            attempts = 0
            while placed < initial_replicas_per_stage and attempts < n_nodes:
                candidate = node_ids_sorted[assignment_cursor % len(node_ids_sorted)]
                assignment_cursor += 1
                attempts += 1
                ok = fleet.apply_claim(candidate, m.model_id, stage.stage_id)
                if ok:
                    placed += 1
            if placed < initial_replicas_per_stage:
                logger.warning(
                    "[fleet] stage %s/%s: only %d/%d replicas placed "
                    "(fleet may be memory-constrained)",
                    m.model_id,
                    stage.stage_id,
                    placed,
                    initial_replicas_per_stage,
                )

    logger.info(
        "[fleet] built %d nodes (%s), %d models, %d stage-types",
        n_nodes,
        _tier_summary(tier_choices),
        len(models),
        len(stage_registry),
    )
    return fleet


def _tier_summary(tier_choices: np.ndarray) -> str:
    names = [_TIER_DEFS[i][0] for i in tier_choices]
    counts = {n: names.count(n) for n in dict.fromkeys(names)}
    return " ".join(f"{k}×{v}" for k, v in counts.items())


def _default_models() -> list[ModelSpec]:
    """Three toy models used when no real model specs are provided."""
    # Small LLM: 4 stages, moderate memory per stage
    llm_stages = tuple(
        StageSpec(
            stage_id=f"llm_s{i}",
            model_id="llm_small",
            compute_tau_cu_ms=200.0 + i * 50.0,
            memory_mib=512,
        )
        for i in range(4)
    )
    llm = ModelSpec(model_id="llm_small", stages=llm_stages)

    # Vision model: 3 stages, light memory
    vision_stages = tuple(
        StageSpec(
            stage_id=f"vision_s{i}",
            model_id="vision_yolo",
            compute_tau_cu_ms=80.0 + i * 20.0,
            memory_mib=256,
        )
        for i in range(3)
    )
    vision = ModelSpec(model_id="vision_yolo", stages=vision_stages)

    # Encoder-decoder: 2 stages
    enc_stages = tuple(
        StageSpec(
            stage_id=f"enc_s{i}",
            model_id="enc_dec",
            compute_tau_cu_ms=120.0,
            memory_mib=384,
        )
        for i in range(2)
    )
    enc_dec = ModelSpec(model_id="enc_dec", stages=enc_stages)

    return [llm, vision, enc_dec]
