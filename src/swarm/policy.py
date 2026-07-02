"""Shard-placement policies for the SwarmInfer discrete-event simulator.

Three concrete policies are provided:

ReactivePolicy
    Uses the current EWMA demand from the gossip cache — the SwarmInfer
    baseline.  No look-ahead.

FDDSPPolicy  (Forecast-Driven Decentralised Shard Placement)
    Wraps a ForecasterAdapter. Calls adapter.predict() on the local demand
    history to obtain a horizon-h demand forecast, then computes per-stage
    projected saturation σ_hat and issues claim/release actions accordingly.
    Implements the §7.3 algorithm from UNIFIED_PAPER_DESIGN.md.

OraclePolicy
    Cheating upper bound. Injects the true future demand from the Workload
    directly into an OracleForecaster, then delegates to the same §7.3
    decision rule as FDDSPPolicy.

All policies share the same ``decide()`` interface::

    actions = policy.decide(node_id, t_ms, fleet, gossip, workload)

which returns a list of PlacementAction objects.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .fleet import Fleet
from .forecasters import (
    ForecasterAdapter,
    LastValueForecaster,
    OracleForecaster,
    SeasonalNaiveForecaster,
)
from .gossip import GossipLayer
from .types import ActionKind, PlacementAction
from .workload import Workload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class PolicyHParams:
    """Shared hyperparameters for FD-DSP and Oracle policies.

    Args:
        sigma_up: Saturation threshold above which a claim is issued.
        sigma_down: Saturation threshold below which a release is issued.
        min_replicas: Durability floor — never release if replica count
            would fall below this.
        hysteresis_ms: Minimum ms between successive claim/release decisions
            for the same (model, stage) on the same node.
        forecast_horizon: Number of future hourly steps to forecast.
        lookback_hours: History window size fed to the forecaster.
    """

    sigma_up: float = 0.80
    sigma_down: float = 0.40
    min_replicas: int = 2
    hysteresis_ms: float = 300_000.0  # 5 minutes
    forecast_horizon: int = 6
    lookback_hours: int = 48


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Policy(ABC):
    """Abstract shard-placement policy.

    All concrete subclasses implement ``decide()`` which inspects fleet and
    gossip state at virtual time ``t_ms`` for a specific node and returns
    a (possibly empty) list of PlacementAction objects.
    """

    name: str = "base"

    @abstractmethod
    def decide(
        self,
        node_id: str,
        t_ms: float,
        fleet: Fleet,
        gossip: GossipLayer,
        workload: Workload,
    ) -> list[PlacementAction]:
        """Produce claim/release actions for ``node_id`` at time ``t_ms``.

        Args:
            node_id: The node making the decision.
            t_ms: Current virtual time (ms).
            fleet: Current fleet state (read-only from policy perspective).
            gossip: Gossip layer (provides demand cache).
            workload: Workload object (used by oracle; reactive ignores it).

        Returns:
            List of PlacementAction.  May be empty.
        """
        ...


# ---------------------------------------------------------------------------
# Reactive baseline
# ---------------------------------------------------------------------------


class ReactivePolicy(Policy):
    """SwarmInfer baseline: use current EWMA demand, no forecast.

    Decision rule:
    - Compute current saturation σ = D_ewma * τ / (ρ * C) for each
      stage the node could host.
    - If σ > sigma_up and node has memory: claim.
    - If σ < sigma_down and node holds the stage: release (if ρ > min_replicas).
    - Hysteresis prevents flapping.
    """

    name = "reactive"

    def __init__(self, hparams: PolicyHParams | None = None) -> None:
        self._hp = hparams or PolicyHParams()

    def decide(
        self,
        node_id: str,
        t_ms: float,
        fleet: Fleet,
        gossip: GossipLayer,
        workload: Workload,
    ) -> list[PlacementAction]:
        node = fleet.nodes[node_id]
        actions: list[PlacementAction] = []

        for model_id, model_spec in fleet.models.items():
            # Current demand EWMA at this node (req/s)
            d_ewma = gossip.get_demand(node_id, model_id)
            # Convert to req/hour for saturation calc (tau is CU*ms/req,
            # capacity is CU/s, so tau/capacity = ms; demand in req/s * ms = dimensionless)
            demand_rate = d_ewma  # req/s

            for stage in model_spec.stages:
                key = (model_id, stage.stage_id)
                rho = max(1, fleet.replica_counts.get(key, 1))

                # Saturation: fraction of node capacity consumed by this stage
                # σ = demand_rate [req/s] * tau [CU*ms/req] / (rho * capacity [CU/s]) / 1000
                # = demand_rate * tau / (rho * capacity * 1000)  [dimensionless]
                sigma = (demand_rate * stage.compute_tau_cu_ms) / (
                    rho * node.spec.compute_cu_per_s * 1000.0
                )

                actions.extend(
                    self._evaluate(
                        node_id, t_ms, fleet, model_id, stage.stage_id, sigma
                    )
                )

        return actions

    def _evaluate(
        self,
        node_id: str,
        t_ms: float,
        fleet: Fleet,
        model_id: str,
        stage_id: str,
        sigma: float,
    ) -> list[PlacementAction]:
        """Apply hysteresis and threshold logic; return actions."""
        node = fleet.nodes[node_id]
        key = (model_id, stage_id)
        hp = self._hp

        last_action = node.last_claim_release_ms.get(key, -hp.hysteresis_ms - 1.0)
        if t_ms - last_action < hp.hysteresis_ms:
            return []  # within hysteresis window

        stage = fleet.stage_registry.get(key)
        if stage is None:
            return []

        actions: list[PlacementAction] = []

        if sigma > hp.sigma_up and key not in node.shards:
            # Can we afford to load it?
            free_mib = node.memory_free_mib(fleet.stage_registry)
            if stage.memory_mib <= free_mib:
                actions.append(
                    PlacementAction(
                        kind=ActionKind.CLAIM,
                        model_id=model_id,
                        stage_id=stage_id,
                        node_id=node_id,
                        t_ms=t_ms,
                    )
                )
                node.last_claim_release_ms[key] = t_ms

        elif sigma < hp.sigma_down and key in node.shards:
            rho = fleet.replica_counts.get(key, 0)
            if rho > hp.min_replicas:
                actions.append(
                    PlacementAction(
                        kind=ActionKind.RELEASE,
                        model_id=model_id,
                        stage_id=stage_id,
                        node_id=node_id,
                        t_ms=t_ms,
                    )
                )
                node.last_claim_release_ms[key] = t_ms

        return actions


# ---------------------------------------------------------------------------
# FD-DSP: forecast-driven policy
# ---------------------------------------------------------------------------


class FDDSPPolicy(Policy):
    """Forecast-Driven Decentralised Shard Placement (Algorithm §7.3).

    Replaces the reactive EWMA demand estimate with a h-step-ahead forecast
    produced by a pluggable ForecasterAdapter.

    The forecaster receives the per-model demand history from the Workload
    (via demand_history()), computes projected saturation, and applies the
    same hysteresis-gated threshold rule as ReactivePolicy.
    """

    name = "fd_dsp"

    def __init__(
        self,
        forecaster: ForecasterAdapter | None = None,
        hparams: PolicyHParams | None = None,
    ) -> None:
        self._forecaster = forecaster or SeasonalNaiveForecaster(period=24)
        self._hp = hparams or PolicyHParams()

    def decide(
        self,
        node_id: str,
        t_ms: float,
        fleet: Fleet,
        gossip: GossipLayer,
        workload: Workload,
    ) -> list[PlacementAction]:
        node = fleet.nodes[node_id]
        hp = self._hp
        actions: list[PlacementAction] = []

        for model_id, model_spec in fleet.models.items():
            # Retrieve demand history from workload object
            history = workload.demand_history(
                t_ms,
                lookback_hours=hp.lookback_hours,
                model_id=model_id,
            )
            if len(history) == 0:
                # No history yet; fall back to gossip EWMA
                d_forecast = np.full(
                    hp.forecast_horizon, gossip.get_demand(node_id, model_id) * 3600.0
                )
            else:
                # history is in req/hour; forecast also in req/hour
                d_forecast = self._forecaster.predict(history, hp.forecast_horizon)

            # Use the step-h forecast (index h-1)
            h_idx = min(hp.forecast_horizon - 1, len(d_forecast) - 1)
            d_hat_per_hour = float(d_forecast[h_idx])
            d_hat_per_sec = d_hat_per_hour / 3600.0

            for stage in model_spec.stages:
                key = (model_id, stage.stage_id)
                rho = max(1, fleet.replica_counts.get(key, 1))

                sigma_hat = (d_hat_per_sec * stage.compute_tau_cu_ms) / (
                    rho * node.spec.compute_cu_per_s * 1000.0
                )

                actions.extend(
                    self._evaluate(
                        node_id, t_ms, fleet, model_id, stage.stage_id, sigma_hat
                    )
                )

        return actions

    def _evaluate(
        self,
        node_id: str,
        t_ms: float,
        fleet: Fleet,
        model_id: str,
        stage_id: str,
        sigma_hat: float,
    ) -> list[PlacementAction]:
        """Hysteresis + threshold evaluation (identical logic to reactive)."""
        node = fleet.nodes[node_id]
        key = (model_id, stage_id)
        hp = self._hp

        last_action = node.last_claim_release_ms.get(key, -hp.hysteresis_ms - 1.0)
        if t_ms - last_action < hp.hysteresis_ms:
            return []

        stage = fleet.stage_registry.get(key)
        if stage is None:
            return []

        actions: list[PlacementAction] = []

        if sigma_hat > hp.sigma_up and key not in node.shards:
            free_mib = node.memory_free_mib(fleet.stage_registry)
            if stage.memory_mib <= free_mib:
                actions.append(
                    PlacementAction(
                        kind=ActionKind.CLAIM,
                        model_id=model_id,
                        stage_id=stage_id,
                        node_id=node_id,
                        t_ms=t_ms,
                    )
                )
                node.last_claim_release_ms[key] = t_ms

        elif sigma_hat < hp.sigma_down and key in node.shards:
            rho = fleet.replica_counts.get(key, 0)
            if rho > hp.min_replicas:
                actions.append(
                    PlacementAction(
                        kind=ActionKind.RELEASE,
                        model_id=model_id,
                        stage_id=stage_id,
                        node_id=node_id,
                        t_ms=t_ms,
                    )
                )
                node.last_claim_release_ms[key] = t_ms

        return actions


# ---------------------------------------------------------------------------
# Oracle policy
# ---------------------------------------------------------------------------


class OraclePolicy(Policy):
    """Upper-bound oracle: uses true future demand from the workload.

    Delegates to FDDSPPolicy after injecting the true future demand into an
    OracleForecaster so the σ_hat computation is identical to FD-DSP.
    """

    name = "oracle"

    def __init__(self, hparams: PolicyHParams | None = None) -> None:
        self._hp = hparams or PolicyHParams()
        self._oracle_fc = OracleForecaster()
        self._inner = FDDSPPolicy(
            forecaster=self._oracle_fc,
            hparams=self._hp,
        )

    def decide(
        self,
        node_id: str,
        t_ms: float,
        fleet: Fleet,
        gossip: GossipLayer,
        workload: Workload,
    ) -> list[PlacementAction]:
        # For each model inject true future before delegating
        # (OracleForecaster.set_future is called fresh each tick)
        actions: list[PlacementAction] = []

        for model_id in fleet.models:
            future = workload.future_demand(
                t_ms,
                horizon_hours=self._hp.forecast_horizon,
                model_id=model_id,
            )
            # req/hour array — convert to req/s for consistency with FDDSPPolicy
            self._oracle_fc.set_future(future)

            # Temporarily restrict the inner policy to this model
            # by building a synthetic single-model workload view
            node = fleet.nodes[node_id]

            for stage in fleet.models[model_id].stages:
                key = (model_id, stage.stage_id)
                rho = max(1, fleet.replica_counts.get(key, 1))
                h_idx = min(self._hp.forecast_horizon - 1, len(future) - 1)
                d_hat_per_sec = float(future[h_idx]) / 3600.0

                sigma_hat = (d_hat_per_sec * stage.compute_tau_cu_ms) / (
                    rho * node.spec.compute_cu_per_s * 1000.0
                )

                actions.extend(
                    self._inner._evaluate(
                        node_id, t_ms, fleet, model_id, stage.stage_id, sigma_hat
                    )
                )

        return actions


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class FDDSPAutoTunePolicy(FDDSPPolicy):
    """FD-DSP with an online single-knob auto-tuner for sigma_up.

    Tracks the ratio of CLAIMs that get reverted (RELEASE on the same node-key)
    inside a short reversion window; if too many recent claims are reverted,
    the controller is over-issuing (false-positive saturation prediction) and
    sigma_up is raised. If too few claims are issued while utilisation is high,
    sigma_up is lowered. Both adjustments are exponentially smoothed and clipped
    to [0.40, 0.99].

    Hyperparameters (only ones not inherited):
        reversion_window_ms: time within which a CLAIM-then-RELEASE counts as
            a reversion (default 10 min).
        target_revert_rate: desired fraction of claims to be reverted; below it
            we lower sigma_up to be more aggressive, above it we raise it.
        adjust_step: per-decision smoothing rate of the sigma_up update.
    """

    name = "fd_dsp_auto"

    def __init__(
        self,
        forecaster: ForecasterAdapter | None = None,
        hparams: PolicyHParams | None = None,
        reversion_window_ms: float = 600_000.0,
        target_revert_rate: float = 0.15,
        adjust_step: float = 0.02,
        sigma_up_clip: tuple[float, float] = (0.40, 0.99),
        bootstrap_sigma_up: float = 0.80,
        free_memory_ratio_floor: float = 0.30,
    ) -> None:
        super().__init__(forecaster=forecaster, hparams=hparams)
        # bootstrap with a more exploratory sigma_up so the controller actually
        # issues claims and has signal to learn from; clip keeps it bounded.
        self._hp.sigma_up = bootstrap_sigma_up
        self._reversion_window_ms = reversion_window_ms
        self._target_revert_rate = target_revert_rate
        self._adjust_step = adjust_step
        self._sigma_up_clip = sigma_up_clip
        # require this fraction of node memory remain free after a claim, to
        # avoid memory thrashing that triggers shard reload cascades. This is
        # the second knob that the auto-tuner needs to deploy proactive claims
        # safely on already-loaded nodes.
        self._free_memory_ratio_floor = free_memory_ratio_floor
        # per-node memory of recent claims and reversions
        self._claim_log: dict[str, list[tuple[float, tuple[str, str]]]] = {}
        self._revert_count: dict[str, int] = {}
        self._claim_count: dict[str, int] = {}

    def _evaluate(
        self,
        node_id: str,
        t_ms: float,
        fleet: Fleet,
        model_id: str,
        stage_id: str,
        sigma_hat: float,
    ) -> list[PlacementAction]:
        # override to add memory-headroom guard before issuing claims.
        node = fleet.nodes[node_id]
        key = (model_id, stage_id)
        hp = self._hp
        last_action = node.last_claim_release_ms.get(key, -hp.hysteresis_ms - 1.0)
        if t_ms - last_action < hp.hysteresis_ms:
            return []
        stage = fleet.stage_registry.get(key)
        if stage is None:
            return []
        actions: list[PlacementAction] = []
        if sigma_hat > hp.sigma_up and key not in node.shards:
            free_mib = node.memory_free_mib(fleet.stage_registry)
            # require enough free memory AFTER the claim to leave headroom
            mem_total = float(node.spec.memory_mib)
            required_post_claim_free = self._free_memory_ratio_floor * mem_total
            if (
                stage.memory_mib <= free_mib
                and (free_mib - stage.memory_mib) >= required_post_claim_free
            ):
                actions.append(
                    PlacementAction(
                        kind=ActionKind.CLAIM,
                        model_id=model_id,
                        stage_id=stage_id,
                        node_id=node_id,
                        t_ms=t_ms,
                    )
                )
                node.last_claim_release_ms[key] = t_ms
        elif sigma_hat < hp.sigma_down and key in node.shards:
            rho = fleet.replica_counts.get(key, 0)
            if rho > hp.min_replicas:
                actions.append(
                    PlacementAction(
                        kind=ActionKind.RELEASE,
                        model_id=model_id,
                        stage_id=stage_id,
                        node_id=node_id,
                        t_ms=t_ms,
                    )
                )
                node.last_claim_release_ms[key] = t_ms
        return actions

    def decide(
        self,
        node_id: str,
        t_ms: float,
        fleet: Fleet,
        gossip: GossipLayer,
        workload: Workload,
    ) -> list[PlacementAction]:
        actions = super().decide(node_id, t_ms, fleet, gossip, workload)

        # update per-node claim/revert bookkeeping
        log = self._claim_log.setdefault(node_id, [])
        # purge older than reversion window
        cutoff = t_ms - self._reversion_window_ms
        log[:] = [(ts, key) for ts, key in log if ts >= cutoff]

        for action in actions:
            key = (action.model_id, action.stage_id)
            if action.kind == ActionKind.CLAIM:
                log.append((t_ms, key))
                self._claim_count[node_id] = self._claim_count.get(node_id, 0) + 1
            elif action.kind == ActionKind.RELEASE:
                # if we released a key we recently claimed, count as reversion
                if any(k == key for _, k in log):
                    self._revert_count[node_id] = self._revert_count.get(node_id, 0) + 1
                    # drop matching claim entries to avoid double-counting
                    log[:] = [(ts, k) for ts, k in log if k != key]

        # apply single-knob update only when we have a meaningful sample
        n_claims = self._claim_count.get(node_id, 0)
        if n_claims >= 5:
            revert_rate = self._revert_count.get(node_id, 0) / max(1, n_claims)
            err = revert_rate - self._target_revert_rate
            # if too many reverts, raise sigma_up; if too few, lower it.
            self._hp.sigma_up = float(
                np.clip(
                    self._hp.sigma_up + self._adjust_step * np.sign(err),
                    self._sigma_up_clip[0],
                    self._sigma_up_clip[1],
                )
            )

        return actions


_POLICY_REGISTRY: dict[str, type[Policy]] = {
    "reactive": ReactivePolicy,
    "fd_dsp": FDDSPPolicy,
    "fd_dsp_auto": FDDSPAutoTunePolicy,
    "oracle": OraclePolicy,
}


def make_policy(
    name: str,
    forecaster: ForecasterAdapter | None = None,
    hparams: PolicyHParams | None = None,
) -> Policy:
    """Instantiate a policy by name.

    Args:
        name: Policy name ('reactive', 'fd_dsp', 'fd_dsp_auto', 'oracle').
        forecaster: ForecasterAdapter for FD-DSP / oracle policies.
        hparams: PolicyHParams override.

    Returns:
        Policy instance.
    """
    if name not in _POLICY_REGISTRY:
        raise KeyError(f"Unknown policy '{name}'. Available: {list(_POLICY_REGISTRY)}")
    cls = _POLICY_REGISTRY[name]
    if cls is ReactivePolicy:
        return cls(hparams=hparams)
    if cls is FDDSPPolicy:
        return cls(forecaster=forecaster, hparams=hparams)
    if cls is FDDSPAutoTunePolicy:
        return cls(forecaster=forecaster, hparams=hparams)
    if cls is OraclePolicy:
        return cls(hparams=hparams)
    return cls()
