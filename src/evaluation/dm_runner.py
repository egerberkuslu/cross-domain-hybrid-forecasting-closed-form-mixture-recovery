"""Pairwise Diebold–Mariano tests: proposed CHA-Hybrid vs every other model.

Per the Phase-6 spec: "Run the Diebold-Mariano test pairwise (proposed
CHA-Hybrid vs every other model) per dataset and horizon."

For stochastic models we average the per-seed prediction matrices first
(this is the standard "mean-of-seeds" pre-aggregation used in most modern
forecasting benchmarks before the DM test), so each comparison reduces to
a single matched-pair error series.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .dm_test import diebold_mariano, DMResult

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("outputs/predictions")
METRICS_DIR = Path("outputs/metrics")


def load_pred_aggregated_over_seeds(
    dataset: str,
    variant: str,
    horizon: int,
    preds_dir: str | Path = PREDICTIONS_DIR,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (y_true_native, mean_y_pred_native_over_seeds).

    Native units; mean is taken across whatever seeds are available.
    Returns (None, None) if no prediction file matches.
    """
    pat = f"{dataset}__{variant}__h{horizon}__s*.npz"
    files = sorted(Path(preds_dir).glob(pat))
    if not files:
        return None, None
    preds = []
    y_true = None
    for f in files:
        d = np.load(f)
        if y_true is None:
            y_true = d["y_true_native"]
        preds.append(d["y_pred_native"])
    return y_true.astype(np.float64), np.mean(np.stack(preds, axis=0), axis=0).astype(
        np.float64
    )


def pairwise_dm_against_proposed(
    proposed: str = "cha_hybrid",
    other_variants: list[str] | None = None,
    datasets: list[str] | None = None,
    horizons: list[int] | None = None,
    loss: str = "mse",
) -> pd.DataFrame:
    """Run DM test for every (dataset, horizon) comparing proposed vs each other.

    Returns long-form: dataset, horizon, variant_a, variant_b, statistic,
    p_value, n_obs, significant_at_005.
    Convention: variant_a = proposed; sign of statistic indicates direction.
    """
    import re

    _STEM_RE = re.compile(
        r"^(?P<ds>.+?)__(?P<variant>.+)__h(?P<h>\d+)__s(?P<seed>-?\d+)$"
    )

    def _parse_stem(stem: str) -> dict | None:
        m = _STEM_RE.match(stem)
        return m.groupdict() if m else None

    if other_variants is None:
        seen = set()
        for f in PREDICTIONS_DIR.glob("*.npz"):
            p = _parse_stem(f.stem)
            if p:
                seen.add(p["variant"])
        other_variants = sorted(seen - {proposed})

    if datasets is None:
        datasets = sorted(
            {
                p["ds"]
                for f in PREDICTIONS_DIR.glob("*.npz")
                if (p := _parse_stem(f.stem))
            }
        )
    if horizons is None:
        horizons = sorted(
            {
                int(p["h"])
                for f in PREDICTIONS_DIR.glob("*.npz")
                if (p := _parse_stem(f.stem))
            }
        )

    rows = []
    for ds in datasets:
        for h in horizons:
            yt_a, yp_a = load_pred_aggregated_over_seeds(ds, proposed, h)
            if yt_a is None:
                continue
            e_a = (yt_a - yp_a).ravel()
            for variant_b in other_variants:
                yt_b, yp_b = load_pred_aggregated_over_seeds(ds, variant_b, h)
                if yt_b is None:
                    continue
                if not np.allclose(yt_a, yt_b):
                    logger.warning(
                        "[dm] y_true differs between %s and %s on %s h=%d — skipping",
                        proposed,
                        variant_b,
                        ds,
                        h,
                    )
                    continue
                e_b = (yt_b - yp_b).ravel()
                res = diebold_mariano(e_a, e_b, horizon=h, loss=loss)
                rows.append(
                    {
                        "dataset": ds,
                        "horizon": int(h),
                        "variant_a": proposed,
                        "variant_b": variant_b,
                        **res.to_dict(),
                    }
                )
    return pd.DataFrame(rows)


def write_dm_table(df: pd.DataFrame, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "dm_test_pairwise.csv"
    df.to_csv(p, index=False)
    return p
