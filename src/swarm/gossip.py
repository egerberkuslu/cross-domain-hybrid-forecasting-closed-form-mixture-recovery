"""Gossip layer: periodic demand-state exchange between nodes.

The gossip model is deliberately simple:
- Every ``period_ms`` virtual milliseconds each node sends a GossipMsg
  to a random sample of ``fan_out`` peers.
- Messages arrive with a fixed propagation delay (``latency_ms``).
- On receipt each node merges the sender's demand snapshot into its own
  DemandCache via an EWMA.
- Replica counts from the message are merged as a max() (last-writer-wins
  monotone lattice — adequate for our simulator since we don't model
  deletions that race with gossip).

This module does not schedule events itself — the Simulator drives calls
to ``tick()`` at GOSSIP_TICK events and calls ``deliver()`` when a
previously scheduled GossipMsg arrives.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .types import GossipMsg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-node demand cache
# ---------------------------------------------------------------------------


@dataclass
class DemandCache:
    """Per-model exponential-moving-average of observed demand at one node.

    Args:
        alpha: EWMA smoothing coefficient.  Higher = more reactive to new
            observations, lower = smoother / less noisy.
        initial_demand: Seed value for all models before any observation
            arrives (avoids cold-start 0-demand causing spurious releases).
    """

    alpha: float = 0.1
    initial_demand: float = 0.0
    _cache: dict[str, float] = field(default_factory=dict)

    def update(self, model_id: str, observed_rate: float) -> None:
        """Incorporate a new demand observation.

        Args:
            model_id: Model whose demand was observed.
            observed_rate: Observed requests per second at the source node.
        """
        prev = self._cache.get(model_id, self.initial_demand)
        self._cache[model_id] = self.alpha * observed_rate + (1.0 - self.alpha) * prev

    def get(self, model_id: str) -> float:
        """Return current EWMA demand estimate (requests/s).

        Returns ``initial_demand`` for unknown models.
        """
        return self._cache.get(model_id, self.initial_demand)

    def snapshot(self) -> dict[str, float]:
        """Return a shallow copy of the current cache."""
        return dict(self._cache)

    def merge(self, remote_snapshot: dict[str, float]) -> None:
        """Merge a remote demand snapshot via EWMA.

        Each remote value is treated as a single new observation.
        """
        for model_id, rate in remote_snapshot.items():
            self.update(model_id, rate)


# ---------------------------------------------------------------------------
# Gossip layer
# ---------------------------------------------------------------------------


@dataclass
class GossipLayer:
    """Manages gossip state for the entire fleet.

    Args:
        node_ids: List of all node IDs in the fleet.
        period_ms: How often (virtual ms) each node fans out a gossip message.
        fan_out: Number of peers each node messages per period.
        propagation_latency_ms: Fixed one-way delivery delay in virtual ms.
        alpha: EWMA coefficient used by all DemandCaches.
        seed: RNG seed for peer selection.
    """

    node_ids: list[str]
    period_ms: float = 5_000.0  # 5 s default gossip period
    fan_out: int = 3
    propagation_latency_ms: float = 50.0
    alpha: float = 0.15
    seed: int = 42

    # per-node demand caches, initialised in __post_init__
    caches: dict[str, DemandCache] = field(default_factory=dict)
    # queue of (deliver_at_ms, GossipMsg) not yet processed
    _pending: list[tuple[float, GossipMsg]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        for nid in self.node_ids:
            self.caches[nid] = DemandCache(alpha=self.alpha)

    # ------------------------------------------------------------------
    # Outbound (called at GOSSIP_TICK for each node)
    # ------------------------------------------------------------------

    def make_messages(
        self,
        sender_id: str,
        t_ms: float,
        replica_counts: dict[tuple[str, str], int],
    ) -> list[GossipMsg]:
        """Construct fan_out gossip messages from a node at time t_ms.

        Args:
            sender_id: Node generating the messages.
            t_ms: Current virtual time (ms).
            replica_counts: Fleet-wide replica counts (from Fleet object).

        Returns:
            List of GossipMsg instances (one per peer, already tagged with
            their recv_time_ms = t_ms + propagation_latency_ms).
        """
        peers = self._select_peers(sender_id)
        cache = self.caches[sender_id]
        snapshot = cache.snapshot()
        recv_time = t_ms + self.propagation_latency_ms

        # Serialise replica_counts keys as strings for the payload
        # (dict keys must be hashable but GossipMsg uses typed dict)
        rc_copy = dict(replica_counts)

        msgs = []
        for peer_id in peers:
            msg = GossipMsg(
                sender=sender_id,
                recv_time_ms=recv_time,
                demand_snapshot=dict(snapshot),
                replica_counts=rc_copy,
                payload={"target": peer_id},
            )
            msgs.append(msg)
        return msgs

    def enqueue(self, msg: GossipMsg) -> None:
        """Add a message to the pending delivery queue."""
        self._pending.append((msg.recv_time_ms, msg))

    # ------------------------------------------------------------------
    # Inbound (called by Simulator when advancing virtual time)
    # ------------------------------------------------------------------

    def deliver_due(self, t_ms: float) -> list[tuple[str, GossipMsg]]:
        """Deliver all messages whose recv_time <= t_ms.

        Returns:
            List of (target_node_id, msg) pairs that were delivered.
            Updates each target's DemandCache in place.
        """
        delivered: list[tuple[str, GossipMsg]] = []
        remaining: list[tuple[float, GossipMsg]] = []

        for recv_time, msg in self._pending:
            if recv_time <= t_ms:
                target = msg.payload.get("target", msg.sender)
                if target in self.caches:
                    self.caches[target].merge(msg.demand_snapshot)
                delivered.append((target, msg))
            else:
                remaining.append((recv_time, msg))

        self._pending = remaining
        return delivered

    # ------------------------------------------------------------------
    # Demand observation
    # ------------------------------------------------------------------

    def observe_request(
        self, node_id: str, model_id: str, window_ms: float = 60_000.0
    ) -> None:
        """Record one request arrival at node_id for model_id.

        The observation is converted to an instantaneous rate and merged
        into the node's DemandCache.  We use a nominal window of
        ``window_ms`` ms to convert counts to rates.

        Args:
            node_id: Node that received the request.
            model_id: Model requested.
            window_ms: Observation window in ms (default 1 minute).
        """
        # 1 event in window_ms -> rate = 1 / (window_ms/1000) req/s
        rate = 1000.0 / window_ms
        if node_id in self.caches:
            self.caches[node_id].update(model_id, rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_peers(self, sender_id: str) -> list[str]:
        """Randomly select fan_out peers excluding the sender."""
        candidates = [nid for nid in self.node_ids if nid != sender_id]
        if not candidates:
            return []
        n = min(self.fan_out, len(candidates))
        chosen = self._rng.choice(candidates, size=n, replace=False)
        return list(chosen)

    def get_demand(self, node_id: str, model_id: str) -> float:
        """Return current EWMA demand estimate at a node for a model."""
        return self.caches[node_id].get(model_id)
