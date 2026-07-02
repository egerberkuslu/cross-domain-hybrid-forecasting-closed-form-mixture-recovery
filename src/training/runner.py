"""Single-experiment runner used by both the smoke and the full-grid drivers.

Convention
----------
Every experiment is identified by a 4-tuple
``(dataset, model_variant, horizon, seed)``. The runner:

  1. Loads the preprocessed dataset (Phase-2 cache).
  2. Looks up the chosen hyperparameters (Phase-5 HP search, cached on disk).
  3. Instantiates the model from ``src.models.REGISTRY``.
  4. Fits, predicts on test, computes metrics in BOTH scaled and original units.
  5. Persists ``predictions/<id>.npz`` + ``metrics/<id>.json``.

The artifacts are self-contained: a downstream Phase-6/7 step can rebuild
the full results table by scanning ``results/metrics/`` alone.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.evaluation import compute_all
from src.models import build_model
from src.preprocessing import load_preprocessed
from src.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# directory conventions (relative to repo root)
# ---------------------------------------------------------------------------

PREDICTIONS_DIR = Path("outputs/predictions")
METRICS_DIR = Path("outputs/metrics")
HP_DIR = Path("outputs/hyperparameters")
CHECKPOINTS_DIR = Path("outputs/checkpoints")


# ---------------------------------------------------------------------------
# identifier helpers
# ---------------------------------------------------------------------------


def run_id(dataset: str, variant: str, horizon: int, seed: int) -> str:
    return f"{dataset}__{variant}__h{horizon}__s{seed}"


def metrics_path(dataset: str, variant: str, horizon: int, seed: int) -> Path:
    return METRICS_DIR / f"{run_id(dataset, variant, horizon, seed)}.json"


def predictions_path(dataset: str, variant: str, horizon: int, seed: int) -> Path:
    return PREDICTIONS_DIR / f"{run_id(dataset, variant, horizon, seed)}.npz"


def checkpoint_path(dataset: str, variant: str, horizon: int, seed: int) -> Path:
    return CHECKPOINTS_DIR / f"{run_id(dataset, variant, horizon, seed)}.pt"


def is_complete(dataset: str, variant: str, horizon: int, seed: int) -> bool:
    return (
        metrics_path(dataset, variant, horizon, seed).exists()
        and predictions_path(dataset, variant, horizon, seed).exists()
    )


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    dataset: str
    model: str  # registry key
    variant: str  # display name (e.g. "chronos_zs", "chronos_ft")
    horizon: int
    seed: int
    chosen_hparams: dict
    metrics_scaled: dict
    metrics_native: dict
    fit_seconds: float
    predict_seconds: float
    n_train_samples: int
    n_parameters: int | None
    status: str = "ok"
    error: str | None = None


def run_single(
    dataset: str,
    model_name: str,  # key in REGISTRY
    variant: str,  # display name persisted on disk (e.g. "chronos_zs")
    horizon: int,
    seed: int,
    chosen_hparams: dict,
    device: str,
    force: bool = False,
    save_checkpoint: bool = False,
) -> RunResult:
    """Run one (dataset, model, horizon, seed) experiment and persist artifacts.

    When ``save_checkpoint=True`` (typical for the proposed model so the
    paper is fully reproducible), the model's ``save_checkpoint`` hook
    writes a ``.pt`` file to ``results/checkpoints/<run_id>.pt``.
    """
    ensure_dir(PREDICTIONS_DIR)
    ensure_dir(METRICS_DIR)
    out_metrics = metrics_path(dataset, variant, horizon, seed)
    out_preds = predictions_path(dataset, variant, horizon, seed)

    if not force and is_complete(dataset, variant, horizon, seed):
        existing = json.loads(out_metrics.read_text())
        logger.info(
            "[run] %s — cached (skipping)", run_id(dataset, variant, horizon, seed)
        )
        return RunResult(
            dataset=dataset,
            model=model_name,
            variant=variant,
            horizon=horizon,
            seed=seed,
            chosen_hparams=existing.get("chosen_hparams", {}),
            metrics_scaled=existing.get("metrics_scaled", {}),
            metrics_native=existing.get("metrics_native", {}),
            fit_seconds=existing.get("fit_seconds", 0.0),
            predict_seconds=existing.get("predict_seconds", 0.0),
            n_train_samples=existing.get("n_train_samples", 0),
            n_parameters=existing.get("n_parameters"),
            status=existing.get("status", "ok"),
        )

    pp = load_preprocessed(dataset)
    train_ws = pp.windows[horizon]["train"]
    val_ws = pp.windows[horizon]["val"]
    test_ws = pp.windows[horizon]["test"]
    train_series = pp.split_scaled.train["value"].to_numpy(np.float32)
    val_series = pp.split_scaled.val["value"].to_numpy(np.float32)

    try:
        t0 = time.perf_counter()
        m = build_model(
            model_name,
            horizon=horizon,
            hparams=dict(chosen_hparams),
            seed=seed,
            device=device,
        )
        m.fit(train_ws, val_ws, train_series=train_series, val_series=val_series)
        fit_s = float(time.perf_counter() - t0)

        t0 = time.perf_counter()
        preds_scaled = m.predict(test_ws)  # (N, h) in scaled units
        pred_s = float(time.perf_counter() - t0)
    except Exception as e:
        logger.error(
            "[run] %s FAILED: %s: %s",
            run_id(dataset, variant, horizon, seed),
            type(e).__name__,
            e,
        )
        rec = {
            "dataset": dataset,
            "model": model_name,
            "variant": variant,
            "horizon": horizon,
            "seed": seed,
            "chosen_hparams": chosen_hparams,
            "status": f"fail: {type(e).__name__}: {e}",
        }
        write_json(rec, out_metrics)
        return RunResult(
            dataset=dataset,
            model=model_name,
            variant=variant,
            horizon=horizon,
            seed=seed,
            chosen_hparams=chosen_hparams,
            metrics_scaled={},
            metrics_native={},
            fit_seconds=0.0,
            predict_seconds=0.0,
            n_train_samples=int(train_ws.X.shape[0]),
            n_parameters=None,
            status=rec["status"],
            error=str(e),
        )

    # inverse-transform predictions and targets back to original units
    scaler = pp.scaler

    def _inverse(arr2d: np.ndarray) -> np.ndarray:
        flat = arr2d.reshape(-1, 1).astype(np.float64)
        return scaler.inverse_transform(flat).reshape(arr2d.shape).astype(np.float64)

    preds_native = _inverse(preds_scaled)
    y_true_native = _inverse(test_ws.y)

    metrics_scaled = compute_all(test_ws.y, preds_scaled)
    metrics_native = compute_all(y_true_native, preds_native)

    n_parameters = None
    if m.fit_report and m.fit_report.n_parameters is not None:
        n_parameters = int(m.fit_report.n_parameters)

    rec = {
        "dataset": dataset,
        "model": model_name,
        "variant": variant,
        "horizon": horizon,
        "seed": seed,
        "chosen_hparams": chosen_hparams,
        "metrics_scaled": metrics_scaled,
        "metrics_native": metrics_native,
        "fit_seconds": fit_s,
        "predict_seconds": pred_s,
        "n_train_samples": int(train_ws.X.shape[0]),
        "n_test_samples": int(test_ws.X.shape[0]),
        "n_parameters": n_parameters,
        "status": "ok",
    }
    write_json(rec, out_metrics)
    np.savez_compressed(
        out_preds,
        y_true_scaled=test_ws.y.astype(np.float32),
        y_pred_scaled=preds_scaled.astype(np.float32),
        y_true_native=y_true_native.astype(np.float64),
        y_pred_native=preds_native.astype(np.float64),
        target_times=np.array(test_ws.target_times.view("int64"), dtype=np.int64),
        horizon=np.int32(horizon),
        seed=np.int32(seed),
    )
    # optional checkpoint persistence (heavy — only for proposed models)
    if save_checkpoint:
        try:
            ensure_dir(CHECKPOINTS_DIR)
            ckpt = checkpoint_path(dataset, variant, horizon, seed)
            saved = m.save_checkpoint(ckpt)
            logger.info(
                "[run] %s checkpoint saved → %s",
                run_id(dataset, variant, horizon, seed),
                saved,
            )
        except Exception as e:
            logger.warning(
                "[run] %s checkpoint save failed: %s",
                run_id(dataset, variant, horizon, seed),
                e,
            )
    logger.info(
        "[run] %s OK  test_rmse_scaled=%.4f test_rmse_native=%.3e  fit=%.1fs pred=%.1fs",
        run_id(dataset, variant, horizon, seed),
        metrics_scaled["rmse"],
        metrics_native["rmse"],
        fit_s,
        pred_s,
    )
    return RunResult(
        dataset=dataset,
        model=model_name,
        variant=variant,
        horizon=horizon,
        seed=seed,
        chosen_hparams=chosen_hparams,
        metrics_scaled=metrics_scaled,
        metrics_native=metrics_native,
        fit_seconds=fit_s,
        predict_seconds=pred_s,
        n_train_samples=int(train_ws.X.shape[0]),
        n_parameters=n_parameters,
    )


# ---------------------------------------------------------------------------
# bulk completion bookkeeping (used by both smoke and main drivers)
# ---------------------------------------------------------------------------


def scan_completed() -> dict:
    """Return a dict of {(dataset, variant, h): {seed: status}} from disk."""
    out: dict = {}
    if not METRICS_DIR.exists():
        return out
    for p in METRICS_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            k = (d["dataset"], d["variant"], int(d["horizon"]))
            out.setdefault(k, {})[int(d["seed"])] = d.get("status", "ok")
        except Exception:
            continue
    return out
