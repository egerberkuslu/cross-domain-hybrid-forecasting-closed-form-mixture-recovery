"""Common forecaster interface used by every Phase-3 model.

Every model in this study — whether it's a one-line Naive baseline, a
gradient-boosted tree, a deep neural net, or a foundation model — exposes
exactly the same surface so the Phase-5 experiment runner can drive the
whole fleet with one loop. This makes the comparison structurally fair:
all models receive the same input arrays, are evaluated on identical
test windows, and report predictions in the same scaled domain.

Conventions
-----------
* All ``X`` and ``y`` arrays are in the **scaled** domain (StandardScaler
  fit on TRAIN only by Phase 2). The Phase-6 metric pipeline inverse-
  transforms predictions and ground-truth back to original units before
  computing RMSE / MAE / MAPE / sMAPE / R^2.
* ``X`` shape is ``(N, lookback)`` and ``y`` shape is ``(N, horizon)``.
* ``predict`` must return ``(N, horizon)`` finite floats (no NaN / inf).
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from src.preprocessing.windowing import WindowSet

logger = logging.getLogger(__name__)


@dataclass
class FitReport:
    """Bookkeeping returned by every fit() call (for cost analysis later)."""

    train_seconds: float = 0.0
    n_train_samples: int = 0
    n_parameters: int | None = None  # leave None when not meaningful
    extra: dict = field(default_factory=dict)


class BaseForecaster(ABC):
    """Abstract base class for every comparison + proposed model."""

    # subclasses set these
    name: str = "base"
    is_stochastic: bool = False  # True ⇒ benefits from multi-seed runs
    supports_multi_horizon: bool = True  # False ⇒ recursive 1-step rollout

    def __init__(
        self,
        horizon: int,
        hparams: dict | None = None,
        seed: int = 42,
        device: str = "cpu",
    ):
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.horizon = int(horizon)
        self.hparams = dict(hparams or {})
        self.seed = int(seed)
        self.device = device
        self.fit_report: FitReport | None = None
        self._fitted = False

    # --- subclass API ---

    @abstractmethod
    def fit(
        self,
        train: WindowSet,
        val: WindowSet | None = None,
        *,
        train_series: "np.ndarray | None" = None,
        val_series: "np.ndarray | None" = None,
    ) -> "BaseForecaster":
        """Fit using the train window set; may use val for early stopping.

        ``train_series`` / ``val_series`` are the corresponding 1-D scaled
        series (with NaNs in long-gap regions) — needed by models that
        operate on contiguous series rather than independent windows
        (e.g. darts deep models, foundation models)."""

    @abstractmethod
    def predict(self, windows: WindowSet) -> np.ndarray:
        """Return ``(N, horizon)`` predictions in the scaled domain."""

    # --- shared helpers ---

    def get_params(self) -> dict:
        return {
            "name": self.name,
            "horizon": self.horizon,
            "seed": self.seed,
            "hparams": dict(self.hparams),
            "is_stochastic": self.is_stochastic,
        }

    def save_checkpoint(self, path: str | Path) -> Path | None:
        """Persist whatever portion of the model is trainable/reproducible.

        Default behaviour:
          * For models with a ``self.model`` PyTorch nn.Module → torch.save the state_dict.
          * For models with ``self._darts_model`` → use the darts native ``.save()``.
          * For pickled stat / sklearn models → joblib.dump the model object.
          * Naive / SeasonalNaive / foundation-models-loaded-from-HF → nothing
            to save (just write a marker JSON with hparams + pretrained ID).

        Subclasses override when finer-grained control is needed (e.g.
        CHA-Hybrid which has two sub-models).

        Returns the path written (or None if nothing to save).
        """
        from pathlib import Path
        import json
        import torch as _torch

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # marker JSON written for every model (hparams, name, pretrained ids)
        marker = {
            "model_name": self.name,
            "horizon": self.horizon,
            "seed": self.seed,
            "hparams": dict(self.hparams),
            "is_stochastic": self.is_stochastic,
        }
        meta_path = p.with_suffix(p.suffix + ".json")
        with meta_path.open("w") as f:
            json.dump(marker, f, indent=2, default=str)
        # PyTorch state_dict if available
        for attr in ("model", "_model", "module", "_module"):
            obj = getattr(self, attr, None)
            if obj is not None and hasattr(obj, "state_dict"):
                _torch.save(obj.state_dict(), p)
                return p
        # darts native save
        darts_m = getattr(self, "_darts_model", None)
        if darts_m is not None and hasattr(darts_m, "save"):
            try:
                darts_m.save(str(p))
                return p
            except Exception as e:
                logger.warning("[%s] darts native save failed: %s", self.name, e)
        # sklearn / xgboost / statsmodels — joblib
        sk_m = getattr(self, "model", None)
        if sk_m is not None:
            try:
                import joblib

                joblib.dump(sk_m, p)
                return p
            except Exception as e:
                logger.warning("[%s] joblib save failed: %s", self.name, e)
        return meta_path

    def _check_pred(self, y_hat: np.ndarray, n_expected: int) -> np.ndarray:
        if not isinstance(y_hat, np.ndarray):
            y_hat = np.asarray(y_hat)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(-1, 1)
        if y_hat.shape != (n_expected, self.horizon):
            raise ValueError(
                f"{self.name}: predicted shape {y_hat.shape} != "
                f"expected ({n_expected}, {self.horizon})"
            )
        if not np.isfinite(y_hat).all():
            n_bad = int((~np.isfinite(y_hat)).sum())
            raise ValueError(
                f"{self.name}: predictions contain {n_bad} non-finite values"
            )
        return y_hat.astype(np.float32, copy=False)


class WindowedForecaster(BaseForecaster):
    """Convenience base for models that work directly on (X, y) windows.

    Concrete subclasses only need to implement :meth:`_fit_arrays` and
    :meth:`_predict_arrays`. Timing + shape checks are handled here.
    """

    @abstractmethod
    def _fit_arrays(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> None:
        ...

    @abstractmethod
    def _predict_arrays(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit(
        self,
        train: WindowSet,
        val: WindowSet | None = None,
        *,
        train_series=None,
        val_series=None,
    ) -> "WindowedForecaster":
        t0 = time.perf_counter()
        self._fit_arrays(
            train.X,
            train.y,
            X_val=None if val is None else val.X,
            y_val=None if val is None else val.y,
        )
        elapsed = time.perf_counter() - t0
        self.fit_report = FitReport(
            train_seconds=float(elapsed),
            n_train_samples=int(train.X.shape[0]),
            n_parameters=self._n_parameters(),
        )
        self._fitted = True
        return self

    def predict(self, windows: WindowSet) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(f"{self.name}: predict() before fit()")
        y_hat = self._predict_arrays(windows.X)
        return self._check_pred(y_hat, n_expected=windows.X.shape[0])

    def _n_parameters(self) -> int | None:  # subclasses override when known
        return None
