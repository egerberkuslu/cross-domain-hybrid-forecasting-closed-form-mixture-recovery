"""Diebold–Mariano (1995) test for equal forecast accuracy.

We use the standard implementation with the Bartlett HAC variance,
truncation lag ``h - 1`` (where ``h`` is the forecast horizon), and a
two-sided alternative.  A small-sample correction following
Harvey, Leybourne and Newbold (1997) is applied so the result is more
reliable on the small-test-set forecast vectors typical of our 1000-2000
window splits.

Convention: H0 = "the two models have equal expected loss".
* statistic positive  ⇒  model A has larger loss than model B
* statistic negative  ⇒  model A has smaller loss than model B (A better)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as student_t


@dataclass
class DMResult:
    statistic: float
    p_value: float
    mean_loss_diff: float
    horizon: int
    n_obs: int
    loss: str
    significant_at_005: bool

    def to_dict(self) -> dict:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "mean_loss_diff": self.mean_loss_diff,
            "horizon": self.horizon,
            "n_obs": self.n_obs,
            "loss": self.loss,
            "significant_at_005": bool(self.significant_at_005),
        }


def _autocov(x: np.ndarray, k: int) -> float:
    n = x.size
    if k >= n:
        return 0.0
    xm = x.mean()
    return float(np.sum((x[k:] - xm) * (x[: n - k] - xm)) / n)


def diebold_mariano(
    e_a: np.ndarray,
    e_b: np.ndarray,
    horizon: int = 1,
    loss: str = "mse",
    small_sample_correction: bool = True,
) -> DMResult:
    """Diebold–Mariano statistic comparing forecast errors of two models.

    ``e_a`` and ``e_b`` must be 1-D arrays of error values (truth - pred)
    aligned over the same forecast origins.  ``horizon`` is the multi-step
    forecast horizon used for the HAC truncation lag.
    """
    e_a = np.asarray(e_a, dtype=np.float64).ravel()
    e_b = np.asarray(e_b, dtype=np.float64).ravel()
    if e_a.shape != e_b.shape:
        raise ValueError(f"e_a {e_a.shape} != e_b {e_b.shape}")
    mask = np.isfinite(e_a) & np.isfinite(e_b)
    e_a = e_a[mask]
    e_b = e_b[mask]
    T = e_a.size
    if T < 8:
        return DMResult(np.nan, np.nan, np.nan, horizon, T, loss, False)

    if loss == "mse":
        loss_a = e_a**2
        loss_b = e_b**2
    elif loss == "mae":
        loss_a = np.abs(e_a)
        loss_b = np.abs(e_b)
    else:
        raise ValueError(f"unknown loss '{loss}'")

    d = loss_a - loss_b
    dbar = float(d.mean())

    # HAC long-run variance with Bartlett kernel, truncation lag = horizon - 1
    h = max(1, int(horizon))
    gamma0 = _autocov(d, 0)
    var = gamma0
    for k in range(1, h):
        var += 2.0 * (1.0 - k / h) * _autocov(d, k)
    if var <= 0:
        return DMResult(np.nan, np.nan, dbar, horizon, T, loss, False)

    stat = dbar / np.sqrt(var / T)

    if small_sample_correction and T > h:
        # Harvey, Leybourne & Newbold (1997) correction
        corr = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
        stat = stat * corr
        p_value = 2.0 * (1.0 - student_t.cdf(abs(stat), df=T - 1))
    else:
        p_value = 2.0 * (1.0 - norm.cdf(abs(stat)))

    return DMResult(
        statistic=float(stat),
        p_value=float(p_value),
        mean_loss_diff=float(dbar),
        horizon=int(horizon),
        n_obs=int(T),
        loss=loss,
        significant_at_005=bool(p_value < 0.05),
    )
