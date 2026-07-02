"""CHA-Hybrid v3 ablation runner.

Variants (per the v3 architecture: Theta-trend + SeasonalNaive +
LSTM-residual + Chronos-Bolt global expert):

  * ``cha_hybrid_v3_decomp_only``      α = 1.0 (pure classical decomposition path)
  * ``cha_hybrid_v3_global_only``      α = 0.0 (pure Chronos-Bolt — aliased
                                       from existing ``chronos_bolt_zs`` runs)
  * ``cha_hybrid_v3_fixed_alpha_0.5``  α = 0.5 (no horizon-adaptive tuning)
  * ``cha_hybrid_v3_altres_gru``       residual sub-model = GRU instead of LSTM

The first three SHARE one v3 fit; the fourth is a separate fit.
"""
from __future__ import annotations

import logging
import shutil
import time
import json
from pathlib import Path

import numpy as np

from src.evaluation import compute_all
from src.models import build_model
from src.preprocessing import load_preprocessed
from src.training.runner import (
    metrics_path,
    predictions_path,
    is_complete,
)
from src.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


SHARED_FIT_VARIANTS = ("cha_hybrid_v3_decomp_only", "cha_hybrid_v3_fixed_alpha_0.5")
ALIAS_VARIANTS = ("cha_hybrid_v3_global_only",)
SEPARATE_FIT_VARIANTS = ("cha_hybrid_v3_altres_gru",)


def _save_run(
    dataset,
    variant,
    horizon,
    seed,
    y_true_scaled,
    y_pred_scaled,
    scaler,
    fit_s,
    predict_s,
    n_parameters,
    chosen_hparams,
    model_name="cha_hybrid_v3",
):
    out_metrics = metrics_path(dataset, variant, horizon, seed)
    out_preds = predictions_path(dataset, variant, horizon, seed)
    ensure_dir(out_metrics.parent)
    ensure_dir(out_preds.parent)

    def _inv(arr):
        flat = arr.reshape(-1, 1).astype(np.float64)
        return scaler.inverse_transform(flat).reshape(arr.shape).astype(np.float64)

    y_pred_native = _inv(y_pred_scaled)
    y_true_native = _inv(y_true_scaled)
    m_sc = compute_all(y_true_scaled, y_pred_scaled)
    m_na = compute_all(y_true_native, y_pred_native)
    rec = {
        "dataset": dataset,
        "model": model_name,
        "variant": variant,
        "horizon": int(horizon),
        "seed": int(seed),
        "chosen_hparams": chosen_hparams,
        "metrics_scaled": m_sc,
        "metrics_native": m_na,
        "fit_seconds": float(fit_s),
        "predict_seconds": float(predict_s),
        "n_train_samples": int(y_pred_scaled.shape[0]),
        "n_test_samples": int(y_pred_scaled.shape[0]),
        "n_parameters": int(n_parameters) if n_parameters else None,
        "status": "ok",
    }
    write_json(rec, out_metrics)
    np.savez_compressed(
        out_preds,
        y_true_scaled=y_true_scaled.astype(np.float32),
        y_pred_scaled=y_pred_scaled.astype(np.float32),
        y_true_native=y_true_native.astype(np.float64),
        y_pred_native=y_pred_native.astype(np.float64),
        horizon=np.int32(horizon),
        seed=np.int32(seed),
    )


def _alias_chronos_bolt_as_global_only(dataset: str, horizon: int, seed: int) -> bool:
    """Copy chronos_bolt_zs as cha_hybrid_v3_global_only (since α=0 ≡ pure Bolt)."""
    src_metrics = metrics_path(
        dataset, "chronos_bolt_zs", horizon, 42
    )  # bolt deterministic
    src_preds = predictions_path(dataset, "chronos_bolt_zs", horizon, 42)
    if not src_metrics.exists():
        return False
    dst_metrics = metrics_path(dataset, "cha_hybrid_v3_global_only", horizon, seed)
    dst_preds = predictions_path(dataset, "cha_hybrid_v3_global_only", horizon, seed)
    if is_complete(dataset, "cha_hybrid_v3_global_only", horizon, seed):
        return False
    d = json.loads(src_metrics.read_text())
    d["variant"] = "cha_hybrid_v3_global_only"
    d["seed"] = int(seed)
    d.setdefault("chosen_hparams", {})["alias"] = "chronos_bolt_zs"
    ensure_dir(dst_metrics.parent)
    ensure_dir(dst_preds.parent)
    write_json(d, dst_metrics)
    shutil.copy(src_preds, dst_preds)
    return True


def run_ablation_v3_for_one(
    dataset: str,
    horizon: int,
    seed: int,
    base_hparams: dict,
    device: str,
    force: bool = False,
) -> dict:
    pp = load_preprocessed(dataset)
    train_ws = pp.windows[horizon]["train"]
    val_ws = pp.windows[horizon]["val"]
    test_ws = pp.windows[horizon]["test"]
    train_series = pp.split_scaled.train["value"].to_numpy(np.float32)
    val_series = pp.split_scaled.val["value"].to_numpy(np.float32)

    results: dict = {}

    # ---- (1) shared-fit pass ----
    need_shared = any(
        not is_complete(dataset, v, horizon, seed) or force for v in SHARED_FIT_VARIANTS
    )
    if need_shared:
        t0 = time.perf_counter()
        m = build_model(
            "cha_hybrid_v3",
            horizon=horizon,
            hparams=dict(base_hparams),
            seed=seed,
            device=device,
        )
        m.fit(train_ws, val_ws, train_series=train_series, val_series=val_series)
        fit_s = time.perf_counter() - t0
        n_params = m.fit_report.n_parameters if m.fit_report else None
        chosen = dict(base_hparams)

        t0 = time.perf_counter()
        p_decomp = m.predict_decomposition_only(test_ws)
        p_global = m.predict_global_only(test_ws)
        p_alpha05 = 0.5 * p_decomp + 0.5 * p_global
        pred_s = time.perf_counter() - t0

        for variant, preds in [
            ("cha_hybrid_v3_decomp_only", p_decomp),
            ("cha_hybrid_v3_fixed_alpha_0.5", p_alpha05),
        ]:
            if is_complete(dataset, variant, horizon, seed) and not force:
                results[variant] = "cached"
                continue
            _save_run(
                dataset,
                variant,
                horizon,
                seed,
                test_ws.y,
                preds,
                pp.scaler,
                fit_s,
                pred_s,
                n_params,
                chosen,
            )
            results[variant] = "ok"
        logger.info(
            "[abl-v3] %s h=%d s=%d shared fit done in %.1fs",
            dataset,
            horizon,
            seed,
            fit_s,
        )
    else:
        for v in SHARED_FIT_VARIANTS:
            results[v] = "cached"

    # ---- (2) alias Chronos-Bolt as global_only ----
    if _alias_chronos_bolt_as_global_only(dataset, horizon, seed):
        results["cha_hybrid_v3_global_only"] = "aliased"
    else:
        results["cha_hybrid_v3_global_only"] = "cached_or_missing"

    # ---- (3) separate fit: residual model = GRU ----
    variant_alt = "cha_hybrid_v3_altres_gru"
    if is_complete(dataset, variant_alt, horizon, seed) and not force:
        results[variant_alt] = "cached"
    else:
        from src.models.cha_hybrid_v3 import CHAHybridV3Forecaster
        from src.models.deep_darts import GRUForecaster
        import src.models.cha_hybrid_v3 as v3mod
        import src.models.cha_hybrid_v2 as v2mod

        class _AltGRUResidual(CHAHybridV3Forecaster):
            name = "cha_hybrid_v3_altres_gru"

            def fit(self, train, val=None, *, train_series=None, val_series=None):
                # v3 inherits residual creation from v2; patch v2's LSTMForecaster
                # reference to GRUForecaster.
                _orig = v2mod.LSTMForecaster
                v2mod.LSTMForecaster = GRUForecaster
                try:
                    return super().fit(
                        train, val, train_series=train_series, val_series=val_series
                    )
                finally:
                    v2mod.LSTMForecaster = _orig

        t0 = time.perf_counter()
        m2 = _AltGRUResidual(
            horizon=horizon, hparams=dict(base_hparams), seed=seed, device=device
        )
        m2.fit(train_ws, val_ws, train_series=train_series, val_series=val_series)
        fit_s = time.perf_counter() - t0
        n_params = m2.fit_report.n_parameters if m2.fit_report else None
        t0 = time.perf_counter()
        preds = m2.predict(test_ws)
        pred_s = time.perf_counter() - t0
        chosen = dict(base_hparams)
        chosen["residual_model"] = "gru"
        _save_run(
            dataset,
            variant_alt,
            horizon,
            seed,
            test_ws.y,
            preds,
            pp.scaler,
            fit_s,
            pred_s,
            n_params,
            chosen,
        )
        results[variant_alt] = "ok"
        logger.info(
            "[abl-v3] %s h=%d s=%d altres_gru fit done in %.1fs",
            dataset,
            horizon,
            seed,
            fit_s,
        )

    return results
