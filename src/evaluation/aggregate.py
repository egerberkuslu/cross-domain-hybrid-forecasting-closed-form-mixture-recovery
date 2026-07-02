"""Aggregate per-run metrics into publication-ready tables.

Scans ``results/metrics/*.json`` and produces a tidy DataFrame indexed by
``(dataset, variant, horizon)`` with per-metric ``mean ± std`` across the
available seeds. Both scaled-domain and native-domain metrics are kept
since the paper will likely report native (bytes/h or kbps) but the
multi-seed variance is computed on the scaled domain for comparability
across datasets of very different magnitudes.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

METRICS_DIR = Path("outputs/metrics")


def load_all_runs(metrics_dir: str | Path = METRICS_DIR) -> pd.DataFrame:
    """Return a long-form DataFrame of every successful run.

    Columns: dataset, variant, horizon, seed, status, n_parameters,
             fit_seconds, predict_seconds,
             rmse_scaled, mae_scaled, mape_scaled, smape_scaled, r2_scaled,
             rmse_native, mae_native, mape_native, smape_native, r2_native,
             chosen_hparams.
    """
    rows = []
    for fn in sorted(Path(metrics_dir).glob("*.json")):
        try:
            d = json.loads(fn.read_text())
        except Exception as e:
            logger.warning("could not parse %s: %s", fn, e)
            continue
        if d.get("status") != "ok":
            continue
        ms = d.get("metrics_scaled", {})
        mn = d.get("metrics_native", {})
        rows.append(
            {
                "dataset": d["dataset"],
                "variant": d["variant"],
                "horizon": int(d["horizon"]),
                "seed": int(d["seed"]),
                "status": d["status"],
                "n_parameters": d.get("n_parameters"),
                "fit_seconds": float(d.get("fit_seconds", 0.0)),
                "predict_seconds": float(d.get("predict_seconds", 0.0)),
                "rmse_scaled": float(ms.get("rmse", np.nan)),
                "mae_scaled": float(ms.get("mae", np.nan)),
                "mape_scaled": float(ms.get("mape", np.nan)),
                "smape_scaled": float(ms.get("smape", np.nan)),
                "r2_scaled": float(ms.get("r2", np.nan)),
                "rmse_native": float(mn.get("rmse", np.nan)),
                "mae_native": float(mn.get("mae", np.nan)),
                "mape_native": float(mn.get("mape", np.nan)),
                "smape_native": float(mn.get("smape", np.nan)),
                "r2_native": float(mn.get("r2", np.nan)),
                "chosen_hparams": json.dumps(d.get("chosen_hparams", {}), default=str),
            }
        )
    return pd.DataFrame(rows)


METRIC_COLS_SCALED = [
    "rmse_scaled",
    "mae_scaled",
    "mape_scaled",
    "smape_scaled",
    "r2_scaled",
]
METRIC_COLS_NATIVE = [
    "rmse_native",
    "mae_native",
    "mape_native",
    "smape_native",
    "r2_native",
]


def aggregate(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate over seeds: mean ± std per (dataset, variant, horizon)."""
    grp = df_runs.groupby(["dataset", "variant", "horizon"])
    agg_funcs = {
        col: ["mean", "std", "count"] for col in METRIC_COLS_SCALED + METRIC_COLS_NATIVE
    }
    agg_funcs["fit_seconds"] = ["mean", "std"]
    agg_funcs["predict_seconds"] = ["mean", "std"]
    agg_funcs["n_parameters"] = ["first"]
    out = grp.agg(agg_funcs)
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    out = out.reset_index()
    return out


def per_metric_table(df_agg: pd.DataFrame, metric: str = "rmse_native") -> pd.DataFrame:
    """Pivot to a (variant × dataset × horizon) table of "mean ± std" strings.

    Returns a DataFrame whose index is variants and whose columns are
    (dataset, horizon) — formatted as scientific notation strings.
    """
    rows = {}
    for _, r in df_agg.iterrows():
        col = (r["dataset"], int(r["horizon"]))
        mean = r[f"{metric}_mean"]
        std = r[f"{metric}_std"]
        n = int(r[f"{metric}_count"])
        if pd.isna(std) or n <= 1:
            s = f"{mean:.3e}"
        else:
            s = f"{mean:.3e} ± {std:.2e}"
        rows.setdefault(r["variant"], {})[col] = s
    out = pd.DataFrame(rows).T
    # column order: (dataset, horizon)
    out = out.reindex(columns=sorted(out.columns, key=lambda c: (c[0], c[1])))
    out.index.name = "variant"
    return out


def write_publication_tables(
    df_runs: pd.DataFrame, out_dir: str | Path
) -> dict[str, Path]:
    """Write per-metric pivot tables (and the wide long-form) under out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # long form
    df_runs.to_csv(out_dir / "metrics_long_form.csv", index=False)
    written["long_form"] = out_dir / "metrics_long_form.csv"

    df_agg = aggregate(df_runs)
    df_agg.to_csv(out_dir / "metrics_aggregated.csv", index=False)
    written["aggregated"] = out_dir / "metrics_aggregated.csv"

    for metric in [
        "rmse_native",
        "mae_native",
        "mape_native",
        "smape_native",
        "r2_native",
        "rmse_scaled",
    ]:
        p = out_dir / f"table_{metric}.csv"
        per_metric_table(df_agg, metric).to_csv(p)
        written[metric] = p

    return written
