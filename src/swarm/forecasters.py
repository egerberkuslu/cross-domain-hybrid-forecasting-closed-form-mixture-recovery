"""Forecaster adapters used by FD-DSP policy.

All adapters share the same interface::

    forecast = adapter.predict(history, horizon)

where ``history`` is a 1-D numpy float array of past demand observations
(requests/s or requests/hour — policy is consistent in its units) and
``horizon`` is the number of future steps to predict.

The return value is always a 1-D float array of length ``horizon`` with
non-negative values.

Adapters are designed to be fast: the default ``MeanForecaster`` and
``SeasonalNaiveForecaster`` are pure-numpy and take microseconds per call.
The ``ChaHybridV3Adapter`` wraps the real CHA-Hybrid v3 model but is only
instantiated when torch/chronos are available.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ForecasterAdapter(ABC):
    """Minimal interface for all forecasters used inside FD-DSP.

    Subclasses must be stateless (no mutable state that depends on calls to
    predict) so they can be safely shared across nodes without locks.
    """

    name: str = "base"

    @abstractmethod
    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Produce an h-step-ahead forecast.

        Args:
            history: 1-D float array of past observations (length >= 1).
            horizon: Number of future steps to forecast (>= 1).

        Returns:
            1-D float array of length ``horizon``, values >= 0.
        """
        ...

    def _safe_output(self, arr: np.ndarray, horizon: int) -> np.ndarray:
        """Clip, fill NaN, and ensure correct length."""
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 1 or len(arr) != horizon:
            arr = np.resize(arr, horizon)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        return np.clip(arr, 0.0, None)


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class MeanForecaster(ForecasterAdapter):
    """Forecast is constant at the mean of history. Fast sanity-check baseline."""

    name = "mean"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(horizon)
        val = float(np.mean(history))
        return self._safe_output(np.full(horizon, val), horizon)


class LastValueForecaster(ForecasterAdapter):
    """Naive: repeat the last observed value for all future steps."""

    name = "last_value"

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if len(history) == 0:
            return np.zeros(horizon)
        val = float(history[-1])
        return self._safe_output(np.full(horizon, val), horizon)


class SeasonalNaiveForecaster(ForecasterAdapter):
    """Seasonal naive: repeat the last full season (period steps).

    For hourly data with ``period=24`` this copies yesterday's pattern.
    Falls back to LastValue when history is shorter than one period.
    """

    name = "seasonal_naive"

    def __init__(self, period: int = 24) -> None:
        self.period = period

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if len(history) < self.period:
            return LastValueForecaster().predict(history, horizon)
        # Tile the last full season to cover horizon steps
        season = history[-self.period :]
        repeats = (horizon // self.period) + 2
        tiled = np.tile(season, repeats)
        return self._safe_output(tiled[:horizon], horizon)


class OracleForecaster(ForecasterAdapter):
    """Cheating oracle: returns the true future demand provided externally.

    The ``true_future`` array is set by the Simulator/Policy at each tick
    from the Workload.future_demand() call. This adapter is used only by
    OraclePolicy and should never be used in production.
    """

    name = "oracle"

    def __init__(self) -> None:
        self._true_future: np.ndarray = np.zeros(0)

    def set_future(self, future: np.ndarray) -> None:
        """Inject true future demand (called by OraclePolicy before predict)."""
        self._true_future = np.asarray(future, dtype=float)

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if len(self._true_future) == 0:
            return MeanForecaster().predict(history, horizon)
        return self._safe_output(self._true_future[:horizon], horizon)


class ChaHybridV3Adapter(ForecasterAdapter):
    """Adapter wrapping the real CHA-Hybrid v3 forecaster.

    This adapter is only instantiated if torch and the model weights are
    available.  It calls CHAHybridV3Forecaster.predict() which requires
    the model to have been previously fit.  For the simulator we use it in
    a simplified "zero-shot on history" mode: we pass the history as the
    context and let Chronos-Bolt produce the global expert forecast, then
    blend with the decomposition expert.

    Because the real model is slow (~100-500 ms per call on CPU), the
    simulator caches the most recent forecast per model and only refreshes
    every ``refresh_every_steps`` policy ticks.
    """

    name = "cha_hybrid_v3"

    def __init__(self, horizon: int = 6, device: str = "cpu") -> None:
        self._horizon = horizon
        self._device = device
        self._model = None
        self._last_forecast: dict[int, np.ndarray] = {}  # hash(history) -> forecast
        self._load_model(horizon, device)

    def _load_model(self, horizon: int, device: str) -> None:
        try:
            from src.models.cha_hybrid_v3 import CHAHybridV3Forecaster  # type: ignore

            self._model = CHAHybridV3Forecaster(horizon=horizon, device=device)
            logger.info("[forecaster] CHA-Hybrid v3 loaded (horizon=%d)", horizon)
        except Exception as exc:
            logger.warning(
                "[forecaster] CHA-Hybrid v3 unavailable (%s); "
                "falling back to SeasonalNaive",
                exc,
            )
            self._model = None

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self._model is None:
            return SeasonalNaiveForecaster(period=24).predict(history, horizon)
        try:
            # CHAHybridV3 expects scaled 1-D array; we pass raw and accept
            # the output directly (scale is consistent within the simulator).
            ctx = history.astype(np.float32)
            # predict signature: predict(X) where X is (N, lookback)
            X = ctx[np.newaxis, :]  # (1, lookback)
            out = self._model.predict(X)  # (1, horizon)
            return self._safe_output(out[0], horizon)
        except Exception as exc:
            logger.debug("[forecaster] CHA-Hybrid v3 predict failed: %s", exc)
            return SeasonalNaiveForecaster(period=24).predict(history, horizon)


class ChronosBoltAdapter(ForecasterAdapter):
    """Adapter wrapping Chronos-Bolt small via the existing project wrapper.

    Falls back to SeasonalNaive if the model is unavailable.
    """

    name = "chronos_bolt"

    def __init__(self, horizon: int = 6) -> None:
        self._horizon = horizon
        self._model = None
        self._load_model(horizon)

    def _load_model(self, horizon: int) -> None:
        try:
            from src.models.sota_2025 import ChronosBoltForecaster  # type: ignore

            self._model = ChronosBoltForecaster(horizon=horizon)
            logger.info("[forecaster] Chronos-Bolt loaded (horizon=%d)", horizon)
        except Exception as exc:
            logger.warning(
                "[forecaster] Chronos-Bolt unavailable (%s); "
                "falling back to SeasonalNaive",
                exc,
            )
            self._model = None

    def predict(self, history: np.ndarray, horizon: int) -> np.ndarray:
        if self._model is None:
            return SeasonalNaiveForecaster(period=24).predict(history, horizon)
        try:
            ctx = history.astype(np.float32)
            X = ctx[np.newaxis, :]
            out = self._model.predict(X)
            return self._safe_output(out[0], horizon)
        except Exception as exc:
            logger.debug("[forecaster] Chronos-Bolt predict failed: %s", exc)
            return SeasonalNaiveForecaster(period=24).predict(history, horizon)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FORECASTER_REGISTRY: dict[str, type[ForecasterAdapter]] = {
    "mean": MeanForecaster,
    "last_value": LastValueForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "oracle": OracleForecaster,
    "cha_hybrid_v3": ChaHybridV3Adapter,
    "chronos_bolt": ChronosBoltAdapter,
}


def make_forecaster(name: str, **kwargs: object) -> ForecasterAdapter:
    """Instantiate a forecaster by registry name.

    Args:
        name: Forecaster name from FORECASTER_REGISTRY.
        **kwargs: Passed to the forecaster's constructor.

    Returns:
        ForecasterAdapter instance.

    Raises:
        KeyError: If name is not in the registry.
    """
    if name not in FORECASTER_REGISTRY:
        raise KeyError(
            f"Unknown forecaster '{name}'. Available: {list(FORECASTER_REGISTRY)}"
        )
    return FORECASTER_REGISTRY[name](**kwargs)  # type: ignore[arg-type]
