"""XGBoost on lag features (Group B — classical ML baseline).

Direct multi-output regression: one XGBRegressor wrapped per output step
(``sklearn.multioutput.MultiOutputRegressor``). The lookback array
``(N, L)`` is the lag-feature design matrix directly — no further
engineering, so the comparison stays apples-to-apples with the neural
models that also consume raw lookbacks.
"""
from __future__ import annotations

import logging

import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from .base import WindowedForecaster

logger = logging.getLogger(__name__)


class XGBoostForecaster(WindowedForecaster):
    name = "xgboost"
    is_stochastic = True       # subsample / colsample_bytree benefit from seeding

    def __init__(self, horizon, hparams=None, seed=42, device="cpu"):
        super().__init__(horizon, hparams, seed, device)
        hp = dict(self.hparams)
        self.n_estimators = int(hp.pop("n_estimators", 500))
        self.max_depth = int(hp.pop("max_depth", 5))
        self.learning_rate = float(hp.pop("learning_rate", 0.05))
        self.subsample = float(hp.pop("subsample", 0.9))
        self.colsample_bytree = float(hp.pop("colsample_bytree", 0.9))
        self.reg_lambda = float(hp.pop("reg_lambda", 1.0))
        self.early_stopping_rounds = hp.pop("early_stopping_rounds", None)
        self._extra = hp
        self.model: MultiOutputRegressor | None = None

    def _build(self) -> MultiOutputRegressor:
        base = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.seed,
            tree_method="hist",
            n_jobs=4,
            verbosity=0,
            **self._extra,
        )
        return MultiOutputRegressor(base, n_jobs=1)

    def _fit_arrays(self, X, y, X_val=None, y_val=None) -> None:
        self.model = self._build()
        self.model.fit(X, y)

    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoost predict before fit")
        return self.model.predict(X).astype(np.float32, copy=False)

    def _n_parameters(self) -> int | None:
        # rough estimate: one model per output step, n_estimators trees each
        if self.model is None:
            return None
        # XGBoost's number of "parameters" is not strictly defined; use n_trees as proxy
        try:
            n_trees = sum(
                int(est.get_booster().num_boosted_rounds())
                for est in self.model.estimators_
            )
            return n_trees
        except Exception:
            return None
