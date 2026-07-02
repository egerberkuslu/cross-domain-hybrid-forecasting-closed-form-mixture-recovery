"""Factory that turns model names from ``config.yaml`` into instances.

Every model in Groups A/B/C and the proposed CHA-Hybrid registers here so
the Phase-5 experiment runner can iterate ``cfg.models`` and instantiate
forecasters uniformly.
"""
from __future__ import annotations

from typing import Type

from .base import BaseForecaster
from .naive import NaiveForecaster, SeasonalNaiveForecaster
from .statistical import ArimaForecaster, HoltWintersForecaster, ThetaForecaster
from .xgboost_model import XGBoostForecaster
from .deep_darts import (
    LSTMForecaster,
    GRUForecaster,
    TCNForecaster,
    NBEATSForecaster,
    DLinearForecaster,
    NHiTSForecaster,
    TFTForecaster,
    TiDEForecaster,
    TSMixerForecaster,
)
from .patchtst_hf import PatchTSTForecaster
from .chronos_model import ChronosForecaster
from .timesfm_model import TimesFMForecaster
from .sota_2025 import ChronosBoltForecaster, MoiraiForecaster, TTMForecaster
from .farima import FARIMAForecaster
from .cha_hybrid import CHAHybridForecaster
from .cha_hybrid_v2 import CHAHybridV2Forecaster
from .cha_hybrid_v3 import CHAHybridV3Forecaster
from .cha_hybrid_v4 import CHAHybridV4Forecaster
from .cha_hybrid_v4_fix import CHAHybridV4FixForecaster


REGISTRY: dict[str, Type[BaseForecaster]] = {
    # Group A
    "naive": NaiveForecaster,
    "seasonal_naive": SeasonalNaiveForecaster,
    "arima": ArimaForecaster,
    "holt_winters": HoltWintersForecaster,
    "theta": ThetaForecaster,
    "farima": FARIMAForecaster,
    # Group B
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,
    "gru": GRUForecaster,
    "tcn": TCNForecaster,
    "nbeats": NBEATSForecaster,
    "dlinear": DLinearForecaster,
    "patchtst": PatchTSTForecaster,
    # Group B' — modern 2023-2024 SOTA additions
    "nhits": NHiTSForecaster,
    "tft": TFTForecaster,
    "tide": TiDEForecaster,
    "tsmixer": TSMixerForecaster,
    # Group C — foundation models
    "chronos": ChronosForecaster,
    "timesfm": TimesFMForecaster,
    "chronos_bolt": ChronosBoltForecaster,
    "moirai": MoiraiForecaster,
    "ttm": TTMForecaster,
    # Proposed
    "cha_hybrid": CHAHybridForecaster,
    "cha_hybrid_v2": CHAHybridV2Forecaster,
    "cha_hybrid_v3": CHAHybridV3Forecaster,
    "cha_hybrid_v4": CHAHybridV4Forecaster,
    "cha_hybrid_v4_fix": CHAHybridV4FixForecaster,
}


def build_model(
    name: str, horizon: int, hparams=None, seed: int = 42, device: str = "cpu"
) -> BaseForecaster:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    cls = REGISTRY[name]
    return cls(horizon=horizon, hparams=hparams or {}, seed=seed, device=device)


def list_models() -> list[str]:
    return sorted(REGISTRY)
