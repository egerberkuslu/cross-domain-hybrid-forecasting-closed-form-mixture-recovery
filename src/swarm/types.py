"""Core dataclasses for the SwarmInfer discrete-event simulator.

All physical capacities use simple scalar units:
  - compute: "capacity units / second" (CU/s). A Jetson has more than an RPi4.
  - memory:  MiB (integer).
  - latency: milliseconds (float).

Stage compute cost (tau) is in CU * ms per request (i.e., how many CU-ms
one request burns on that stage). Dividing by a node's CU/s capacity gives
the per-request service time in milliseconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EventKind(Enum):
    REQUEST_ARRIVAL = auto()
    REQUEST_COMPLETE = auto()
    GOSSIP_TICK = auto()
    POLICY_TICK = auto()
    FORECAST_TICK = auto()


class ActionKind(Enum):
    CLAIM = "claim"
    RELEASE = "release"


# ---------------------------------------------------------------------------
# Hardware / topology specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeSpec:
    """Static description of one node's hardware capabilities.

    Args:
        node_id: Unique string identifier.
        tier: Human-readable tier label (e.g. "jetson", "rpi4").
        compute_cu_per_s: Compute capacity in abstract "compute units per
            second".  Requests serviced by this node consume tau / capacity
            milliseconds.
        memory_mib: Total RAM available for model shards (MiB).
        net_latency_ms: One-way network latency from this node to a
            hypothetical neighbour (used for hop-by-hop stage chaining).
    """

    node_id: str
    tier: str
    compute_cu_per_s: float  # CU / s
    memory_mib: int
    net_latency_ms: float  # ms per hop


@dataclass(frozen=True)
class StageSpec:
    """One stage of a model (a "shard").

    Args:
        stage_id: Globally unique stage identifier string, e.g. "llm_s0".
        model_id: Which model this stage belongs to.
        compute_tau_cu_ms: Compute cost in CU * ms per request.  The actual
            service time at a given node is tau / node.compute_cu_per_s.
        memory_mib: How many MiB this stage occupies when loaded.
    """

    stage_id: str
    model_id: str
    compute_tau_cu_ms: float
    memory_mib: int


@dataclass(frozen=True)
class ModelSpec:
    """A model is an ordered list of stages (pipeline / chain topology).

    Args:
        model_id: Unique model identifier.
        stages: Ordered list of StageSpec; request traverses them in order.
    """

    model_id: str
    stages: tuple[StageSpec, ...]

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def total_memory_mib(self) -> int:
        return sum(s.memory_mib for s in self.stages)


# ---------------------------------------------------------------------------
# Request lifecycle
# ---------------------------------------------------------------------------


@dataclass
class Request:
    """One inference request travelling through the fleet.

    Args:
        request_id: Unique integer identifier.
        model_id: Which model to run.
        arrival_ms: Virtual time (ms) when the request arrived at the fleet.
        initiator_node: Node that first received / injected this request.
        deadline_ms: Optional SLO deadline (absolute virtual ms).
    """

    request_id: int
    model_id: str
    arrival_ms: float
    initiator_node: str
    deadline_ms: float | None = None

    # filled in as the request completes
    completion_ms: float | None = None
    rejected: bool = False
    rejection_reason: str = ""

    @property
    def latency_ms(self) -> float | None:
        if self.completion_ms is None:
            return None
        return self.completion_ms - self.arrival_ms


# ---------------------------------------------------------------------------
# Gossip
# ---------------------------------------------------------------------------


@dataclass
class GossipMsg:
    """A single gossip message exchanged between nodes.

    Args:
        sender: Node ID of the sender.
        recv_time_ms: Virtual time when this message is processed.
        demand_snapshot: Dict mapping model_id -> observed requests/s at sender.
        replica_counts: Dict mapping (model_id, stage_id) -> n_replicas known
            to sender.
        payload: Arbitrary extra data (for extensions).
    """

    sender: str
    recv_time_ms: float
    demand_snapshot: dict[str, float]
    replica_counts: dict[tuple[str, str], int]
    payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policy action
# ---------------------------------------------------------------------------


@dataclass
class PlacementAction:
    """A claim or release decision produced by a policy.

    Args:
        kind: CLAIM or RELEASE.
        model_id: Target model.
        stage_id: Target stage.
        node_id: Node issuing the action.
        t_ms: Virtual time of decision.
    """

    kind: ActionKind
    model_id: str
    stage_id: str
    node_id: str
    t_ms: float
