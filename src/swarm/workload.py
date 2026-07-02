"""Workload generator: replays hourly-aggregated traces as Poisson arrivals.

Design
------
The processed parquet files (``data/processed/<name>.parquet``) contain
one row per hour with a ``value`` column (float, request count for that hour).
This module:

1. Loads such a file (or accepts a DataFrame directly).
2. Interprets each hourly ``value`` as a Poisson rate (requests / hour).
3. Converts that rate to requests / minute by dividing by 60.
4. Samples arrival offsets within each minute using a Poisson process
   (inter-arrival times are Exponential(rate)).
5. Yields ``Request`` objects in chronological virtual-time order, where
   virtual time is milliseconds since trace start.

The simulator can consume the resulting sequence event by event without
buffering the whole trace — the Workload acts as a lazy generator.

Multi-model support
-------------------
When ``n_models > 1`` the request stream is split across models by randomly
assigning each request to a model according to ``model_weights`` (defaults
to uniform). This is a practical approximation to a real heterogeneous fleet
where multiple models share the infrastructure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .types import Request

logger = logging.getLogger(__name__)


@dataclass
class WorkloadConfig:
    """Configuration for the Workload generator.

    Args:
        dataset_name: Human-readable name (used for logging and file lookup).
        duration_hours: How many hours of trace to replay (from the start).
            None means replay the full available trace.
        model_ids: List of model IDs to emit requests for.
        model_weights: Probability of each model; must sum to 1.  Defaults
            to uniform if None.
        seed: RNG seed for reproducible Poisson sampling.
        min_rate_per_hour: Clamp zero / negative hourly counts to this floor
            (avoids degenerate hours with 0 arrivals in smoke tests).
    """

    dataset_name: str
    duration_hours: int | None = None
    model_ids: list[str] = field(default_factory=lambda: ["model_0"])
    model_weights: list[float] | None = None
    seed: int = 42
    min_rate_per_hour: float = 1.0


class Workload:
    """Replays a real hourly trace as per-minute Poisson request arrivals.

    Args:
        hourly_series: DataFrame with DatetimeIndex and 'value' column
            (requests per hour).
        cfg: WorkloadConfig.
    """

    def __init__(self, hourly_series: pd.DataFrame, cfg: WorkloadConfig) -> None:
        self._series = hourly_series.copy()
        self._cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

        n_models = len(cfg.model_ids)
        if cfg.model_weights is None:
            self._model_weights = np.ones(n_models) / n_models
        else:
            w = np.array(cfg.model_weights, dtype=float)
            self._model_weights = w / w.sum()

        # Trim to requested duration
        if cfg.duration_hours is not None:
            self._series = self._series.iloc[: cfg.duration_hours]

        # Floor the rates
        self._series["value"] = self._series["value"].clip(lower=cfg.min_rate_per_hour)

        # Total request count (deterministic expectation, useful for tests)
        self._expected_total: float = float(self._series["value"].sum())
        logger.info(
            "[workload:%s] %d hours loaded, expected total %.0f requests, " "models=%s",
            cfg.dataset_name,
            len(self._series),
            self._expected_total,
            cfg.model_ids,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_hours(self) -> int:
        return len(self._series)

    @property
    def expected_total_requests(self) -> float:
        return self._expected_total

    @property
    def hourly_counts(self) -> np.ndarray:
        """Return the (possibly clipped) hourly request counts as 1-D array."""
        return self._series["value"].to_numpy()

    # ------------------------------------------------------------------
    # Main generator
    # ------------------------------------------------------------------

    def generate(self) -> Iterator[Request]:
        """Yield Request objects in chronological virtual-time order.

        Virtual time starts at 0 ms (= trace start) and advances hour by hour.
        Within each hour the 60 minutes are sampled independently.  Within
        each minute inter-arrival times are drawn from Exp(rate/ms).
        """
        req_id = 0
        for hour_idx, row in enumerate(self._series.itertuples(index=False)):
            hour_start_ms = float(hour_idx) * 3_600_000.0  # ms
            rate_per_hour = float(row.value)
            rate_per_minute = rate_per_hour / 60.0
            rate_per_ms = rate_per_hour / 3_600_000.0  # for inter-arrival draws

            for minute in range(60):
                minute_start_ms = hour_start_ms + minute * 60_000.0
                # Poisson count for this minute
                n_arrivals = int(self._rng.poisson(lam=rate_per_minute))
                if n_arrivals == 0:
                    continue

                # Uniform offsets within [0, 60_000) ms — equivalent to
                # a Poisson process conditioned on n_arrivals events.
                offsets = np.sort(self._rng.uniform(0.0, 60_000.0, size=n_arrivals))

                # Assign models
                model_indices = self._rng.choice(
                    len(self._cfg.model_ids),
                    size=n_arrivals,
                    p=self._model_weights,
                )

                for k in range(n_arrivals):
                    arrival_ms = minute_start_ms + offsets[k]
                    model_id = self._cfg.model_ids[model_indices[k]]
                    yield Request(
                        request_id=req_id,
                        model_id=model_id,
                        arrival_ms=arrival_ms,
                        initiator_node="",  # filled by simulator
                    )
                    req_id += 1

    # ------------------------------------------------------------------
    # Oracle future demand
    # ------------------------------------------------------------------

    def future_demand(
        self,
        t_ms: float,
        horizon_hours: int,
        model_id: str | None = None,
    ) -> np.ndarray:
        """Return true future hourly demand (for oracle policy).

        Args:
            t_ms: Current virtual time in ms.
            horizon_hours: How many future hourly steps to return.
            model_id: If provided, scale by that model's weight; otherwise
                return total across all models.

        Returns:
            1-D float array of length ``horizon_hours``.
        """
        current_hour = int(t_ms / 3_600_000.0)
        start = current_hour + 1
        end = start + horizon_hours
        arr = self._series["value"].to_numpy()
        # Pad with last known value if we run off the end
        if start >= len(arr):
            return np.full(horizon_hours, arr[-1] if len(arr) > 0 else 0.0)
        window = arr[start:end]
        if len(window) < horizon_hours:
            pad = np.full(
                horizon_hours - len(window), window[-1] if len(window) > 0 else 0.0
            )
            window = np.concatenate([window, pad])

        if model_id is not None and model_id in self._cfg.model_ids:
            idx = self._cfg.model_ids.index(model_id)
            window = window * self._model_weights[idx]

        return window.astype(float)

    def demand_history(
        self,
        t_ms: float,
        lookback_hours: int,
        model_id: str | None = None,
    ) -> np.ndarray:
        """Return past hourly demand up to (not including) current hour.

        Args:
            t_ms: Current virtual time in ms.
            lookback_hours: Number of past hours to include.
            model_id: If provided, scale by model weight.

        Returns:
            1-D float array of length <= ``lookback_hours`` (may be shorter
            at trace start).
        """
        current_hour = int(t_ms / 3_600_000.0)
        start = max(0, current_hour - lookback_hours)
        end = current_hour
        arr = self._series["value"].to_numpy()
        window = arr[start:end]

        if model_id is not None and model_id in self._cfg.model_ids:
            idx = self._cfg.model_ids.index(model_id)
            window = window * self._model_weights[idx]

        return window.astype(float)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_parquet(cls, path: str | Path, cfg: WorkloadConfig) -> "Workload":
        """Load from a processed parquet file.

        Args:
            path: Path to the parquet file.
            cfg: WorkloadConfig.

        Returns:
            Workload instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Workload parquet not found: {path}")
        df = pd.read_parquet(path)
        if "value" not in df.columns:
            raise ValueError(
                f"Parquet must have 'value' column, got: {df.columns.tolist()}"
            )
        logger.info("[workload] loaded %s (%d rows)", path, len(df))
        return cls(df, cfg)

    @classmethod
    def from_dataset(cls, name: str, cfg: WorkloadConfig) -> "Workload":
        """Resolve dataset name to standard processed path and load.

        Args:
            name: Dataset name, e.g. 'burstgpt', 'azure_llm_2024'.
            cfg: WorkloadConfig (dataset_name is overridden to ``name``).

        Returns:
            Workload instance.
        """
        # Locate repo root relative to this file (src/swarm/workload.py)
        repo_root = Path(__file__).parent.parent.parent
        path = repo_root / "data" / "processed" / f"{name}.parquet"
        cfg = WorkloadConfig(
            dataset_name=name,
            duration_hours=cfg.duration_hours,
            model_ids=cfg.model_ids,
            model_weights=cfg.model_weights,
            seed=cfg.seed,
            min_rate_per_hour=cfg.min_rate_per_hour,
        )
        return cls.from_parquet(path, cfg)

    # ------------------------------------------------------------------
    # Synthetic fallback (for tests without real data)
    # ------------------------------------------------------------------

    @classmethod
    def synthetic(
        cls,
        n_hours: int = 24,
        mean_rate: float = 1000.0,
        seed: int = 42,
        model_ids: list[str] | None = None,
    ) -> "Workload":
        """Create a synthetic workload with sinusoidal diurnal pattern.

        Args:
            n_hours: Length of the synthetic trace.
            mean_rate: Mean requests per hour.
            seed: RNG seed for reproducibility.
            model_ids: Model IDs to use (default: ['model_0']).

        Returns:
            Workload instance.
        """
        if model_ids is None:
            model_ids = ["model_0"]
        rng = np.random.default_rng(seed)
        t = np.arange(n_hours)
        # Sinusoidal diurnal pattern + some noise
        pattern = mean_rate * (1.0 + 0.4 * np.sin(2 * np.pi * t / 24.0))
        noise = rng.normal(0, mean_rate * 0.05, size=n_hours)
        values = np.clip(pattern + noise, 1.0, None)
        df = pd.DataFrame(
            {"value": values},
            index=pd.date_range("2024-01-01", periods=n_hours, freq="h"),
        )
        cfg = WorkloadConfig(
            dataset_name="synthetic",
            model_ids=model_ids,
            seed=seed,
        )
        return cls(df, cfg)

    @classmethod
    def synthetic_ramp(
        cls,
        n_hours: int = 24,
        baseline_rate: float = 800.0,
        ramp_peak_rate: float = 3500.0,
        ramp_start_hour: int = 8,
        ramp_duration_hours: int = 6,
        seed: int = 42,
        model_ids: list[str] | None = None,
    ) -> "Workload":
        """Synthetic ramp workload: a sustained smooth ramp-up over several
        hours, ideal for demonstrating the value of horizon-h forecasting
        against a reactive baseline. The ramp is predictable from history,
        so a forecaster sees it h steps before the reactive EWMA does.

        Args:
            n_hours: Length of the synthetic trace.
            baseline_rate: Mean requests per hour outside the ramp window.
            ramp_peak_rate: Mean requests per hour at the ramp peak.
            ramp_start_hour: Hour at which the ramp begins.
            ramp_duration_hours: Length of the ramp window (rise + plateau).
            seed: RNG seed.
            model_ids: Model IDs.
        """
        if model_ids is None:
            model_ids = ["model_0"]
        rng = np.random.default_rng(seed)
        t = np.arange(n_hours)
        # Build a piecewise pattern: baseline, sigmoid rise, plateau, sigmoid fall.
        ramp_end = ramp_start_hour + ramp_duration_hours
        x = (t - ramp_start_hour - ramp_duration_hours / 2.0) / max(
            1.0, ramp_duration_hours / 6.0
        )
        sigmoid = 1.0 / (1.0 + np.exp(-x))
        # bell-like: rise then fall around mid-ramp
        rise = sigmoid * (1.0 - sigmoid) * 4.0
        in_window = ((t >= ramp_start_hour) & (t < ramp_end)).astype(float)
        pattern = baseline_rate + (ramp_peak_rate - baseline_rate) * rise * in_window
        noise = rng.normal(0, baseline_rate * 0.05, size=n_hours)
        values = np.clip(pattern + noise, 1.0, None)
        df = pd.DataFrame(
            {"value": values},
            index=pd.date_range("2024-01-01", periods=n_hours, freq="h"),
        )
        cfg = WorkloadConfig(
            dataset_name="synthetic_ramp",
            model_ids=model_ids,
            seed=seed,
        )
        return cls(df, cfg)
