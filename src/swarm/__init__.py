"""SwarmInfer discrete-event simulator — public API.

Typical usage::

    from src.swarm import make_fleet, Workload, WorkloadConfig
    from src.swarm import Simulator, SimulatorConfig
    from src.swarm import ReactivePolicy, FDDSPPolicy, OraclePolicy
    from src.swarm import MetricCollector

    fleet = make_fleet(n_nodes=25, seed=42)
    workload = Workload.synthetic(n_hours=24, seed=42)
    policy = ReactivePolicy()
    sim = Simulator(fleet=fleet, workload=workload, policy=policy)
    metrics = sim.run()
    print(metrics.to_dict())
"""
from __future__ import annotations

from .fleet import Fleet, NodeState, make_fleet
from .forecasters import (
    ChronosBoltAdapter,
    ForecasterAdapter,
    LastValueForecaster,
    MeanForecaster,
    OracleForecaster,
    SeasonalNaiveForecaster,
    make_forecaster,
)
from .gossip import DemandCache, GossipLayer
from .metrics import MetricCollector, RunMetrics
from .policy import (
    FDDSPPolicy,
    OraclePolicy,
    Policy,
    PolicyHParams,
    ReactivePolicy,
    make_policy,
)
from .simulator import Simulator, SimulatorConfig
from .types import (
    ActionKind,
    EventKind,
    GossipMsg,
    ModelSpec,
    NodeSpec,
    PlacementAction,
    Request,
    StageSpec,
)
from .workload import Workload, WorkloadConfig

__all__ = [
    # fleet
    "Fleet",
    "NodeState",
    "make_fleet",
    # forecasters
    "ForecasterAdapter",
    "MeanForecaster",
    "LastValueForecaster",
    "SeasonalNaiveForecaster",
    "OracleForecaster",
    "ChronosBoltAdapter",
    "make_forecaster",
    # gossip
    "DemandCache",
    "GossipLayer",
    # metrics
    "MetricCollector",
    "RunMetrics",
    # policy
    "Policy",
    "PolicyHParams",
    "ReactivePolicy",
    "FDDSPPolicy",
    "OraclePolicy",
    "make_policy",
    # simulator
    "Simulator",
    "SimulatorConfig",
    # types
    "ActionKind",
    "EventKind",
    "GossipMsg",
    "ModelSpec",
    "NodeSpec",
    "PlacementAction",
    "Request",
    "StageSpec",
    # workload
    "Workload",
    "WorkloadConfig",
]
