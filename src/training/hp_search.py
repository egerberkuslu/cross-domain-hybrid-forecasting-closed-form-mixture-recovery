"""Grid-search hyperparameter tuning on the validation split.

Every tunable Phase-3 model exposes its search grid in
``config.yaml::hp_search.<model>``. The runner here iterates the
Cartesian product of that grid, fits each candidate on the train set,
scores on the validation set (RMSE on the scaled domain), and persists
the winner under ``results/hyperparameters/<dataset>_<model>_h{h}.json``.

The selected hyperparameters are then re-used in Phase 5 for the multi-
seed final runs, so the fairness claim ("every baseline was tuned, not
run on defaults") is auditable from disk.
"""
from __future__ import annotations

import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.evaluation import rmse
from src.models import build_model
from src.preprocessing import load_preprocessed
from src.utils.io import write_json

logger = logging.getLogger(__name__)


@dataclass
class HPResult:
    dataset: str
    model: str
    horizon: int
    chosen: dict
    val_rmse: float
    search_space: list[dict]
    all_results: list[dict]   # one row per HP candidate
    elapsed_seconds: float


def _expand_grid(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    vals = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def grid_search_one(
    dataset_name: str,
    model_name: str,
    horizon: int,
    grid: dict,
    *,
    device: str,
    seed: int,
    out_dir: Path,
    base_hparams: dict | None = None,
) -> HPResult:
    """Tune a single (dataset, model, horizon) on val and persist the winner.

    ``base_hparams`` are fixed defaults merged into every candidate (e.g.
    foundation-model presets like 'pretrained=...').
    """
    pp = load_preprocessed(dataset_name)
    train_ws = pp.windows[horizon]["train"]
    val_ws = pp.windows[horizon]["val"]
    train_series = pp.split_scaled.train["value"].to_numpy(dtype=np.float32)
    val_series = pp.split_scaled.val["value"].to_numpy(dtype=np.float32)

    cands = _expand_grid(grid)
    base = dict(base_hparams or {})

    rows = []
    best_rmse = float("inf")
    best_hp: dict = {}
    t_all = time.perf_counter()
    for i, hp in enumerate(cands):
        full = {**base, **hp}
        try:
            m = build_model(model_name, horizon=horizon, hparams=full, seed=seed, device=device)
            t0 = time.perf_counter()
            m.fit(train_ws, val_ws,
                  train_series=train_series, val_series=val_series)
            fit_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            preds = m.predict(val_ws)
            pred_s = time.perf_counter() - t0
            val_r = float(rmse(val_ws.y, preds))
            rows.append({
                "candidate": full, "val_rmse": val_r,
                "fit_s": float(fit_s), "predict_s": float(pred_s),
                "n_params": int(m.fit_report.n_parameters) if m.fit_report and m.fit_report.n_parameters else None,
                "status": "ok",
            })
            logger.info("[hp] %s/%s/h=%d %d/%d %s -> val_rmse=%.4f",
                        dataset_name, model_name, horizon, i+1, len(cands), full, val_r)
            if val_r < best_rmse:
                best_rmse = val_r
                best_hp = full
        except Exception as e:
            rows.append({"candidate": full, "val_rmse": float("nan"),
                         "status": f"fail: {type(e).__name__}: {e}"})
            logger.warning("[hp] %s/%s/h=%d candidate %s failed: %s",
                           dataset_name, model_name, horizon, full, e)

    if not np.isfinite(best_rmse):
        # everything failed → record the first candidate as a placeholder
        best_hp = {**base, **(cands[0] if cands else {})}

    res = HPResult(
        dataset=dataset_name,
        model=model_name,
        horizon=int(horizon),
        chosen=best_hp,
        val_rmse=float(best_rmse) if np.isfinite(best_rmse) else float("nan"),
        search_space=cands,
        all_results=rows,
        elapsed_seconds=float(time.perf_counter() - t_all),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{dataset_name}_{model_name}_h{horizon}.json"
    write_json(asdict(res), out_file)
    logger.info("[hp] wrote %s (best val_rmse=%.4f)", out_file, res.val_rmse)
    return res
