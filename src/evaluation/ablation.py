"""CHA-Hybrid ablation runner — produces 4 variants for the ablation table.

Variants (per the Phase-6 spec):
  * cha_hybrid_decomp_only      α = 1.0 (decomposition forecast only)
  * cha_hybrid_global_only      α = 0.0 (global LSTM only)
  * cha_hybrid_fixed_alpha_0.5  α = 0.5 (fixed, no per-horizon tuning)
  * cha_hybrid_altres_lstm      residual model = LSTM (instead of GRU)

We re-fit a fresh CHA-Hybrid once per (dataset, horizon, seed) and call
the model's ``predict_decomposition_only`` / ``predict_global_only``
hooks plus a manual α=0.5 combination to avoid redundant retraining for
the first three variants.  The fourth (altres_lstm) is a separate fit
because its residual sub-model architecture differs.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from src.evaluation import compute_all
from src.models import build_model
from src.preprocessing import load_preprocessed
from src.training import metrics_path, predictions_path, run_id
from src.training.runner import is_complete
from src.utils.io import ensure_dir, write_json

logger = logging.getLogger(__name__)


# Variants we'll produce.  The first three share trained sub-models, the
# fourth needs its own fit.
SHARED_FIT_VARIANTS = (
    "cha_hybrid_decomp_only",
    "cha_hybrid_global_only",
    "cha_hybrid_fixed_alpha_0.5",
)
SEPARATE_FIT_VARIANTS = ("cha_hybrid_altres_lstm",)


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
):
    from src.training.runner import metrics_path, predictions_path

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
        "model": "cha_hybrid",
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


def run_ablation_for_one(
    dataset: str,
    horizon: int,
    seed: int,
    base_hparams: dict,
    device: str,
    force: bool = False,
) -> dict:
    """Run all four ablation variants for one (dataset, horizon, seed)."""
    pp = load_preprocessed(dataset)
    train_ws = pp.windows[horizon]["train"]
    val_ws = pp.windows[horizon]["val"]
    test_ws = pp.windows[horizon]["test"]
    train_series = pp.split_scaled.train["value"].to_numpy(np.float32)
    val_series = pp.split_scaled.val["value"].to_numpy(np.float32)

    results: dict = {}

    # ---- (1) shared-fit pass: build standard CHA-Hybrid once ----
    need_shared = any(
        not is_complete(dataset, v, horizon, seed) or force for v in SHARED_FIT_VARIANTS
    )
    if need_shared:
        t0 = time.perf_counter()
        m = build_model(
            "cha_hybrid",
            horizon=horizon,
            hparams=dict(base_hparams),
            seed=seed,
            device=device,
        )
        m.fit(train_ws, val_ws, train_series=train_series, val_series=val_series)
        fit_s = time.perf_counter() - t0
        n_params = m.fit_report.n_parameters if m.fit_report else None
        chosen = dict(base_hparams)
        # ablation predictions on test
        t0 = time.perf_counter()
        p_decomp = m.predict_decomposition_only(test_ws)
        p_global = m.predict_global_only(test_ws)
        p_alpha05 = 0.5 * p_decomp + 0.5 * p_global
        pred_s = time.perf_counter() - t0

        for variant, preds in [
            ("cha_hybrid_decomp_only", p_decomp),
            ("cha_hybrid_global_only", p_global),
            ("cha_hybrid_fixed_alpha_0.5", p_alpha05),
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
            "[ablation] %s h=%d s=%d shared fit done in %.1fs",
            dataset,
            horizon,
            seed,
            fit_s,
        )
    else:
        for v in SHARED_FIT_VARIANTS:
            results[v] = "cached"

    # ---- (2) separate fit: residual model = LSTM ----
    variant_altres = "cha_hybrid_altres_lstm"
    if is_complete(dataset, variant_altres, horizon, seed) and not force:
        results[variant_altres] = "cached"
    else:
        alt_hp = dict(base_hparams)
        alt_hp["residual_model"] = "lstm"
        # We swap the GRU sub-model class by replacing the gru hparam block
        # with an lstm block.  The cha_hybrid model still constructs a GRU
        # internally via deep_darts.GRUForecaster, so we monkey-trick by
        # passing extra hparams that route a BlockRNNModel(model='LSTM') as
        # the residual: the simplest is to swap the residual *block* in the
        # cha_hybrid module by name.  Since our CHAHybridForecaster
        # currently hard-codes GRUForecaster for the residual, we run a
        # variant where we re-use the same GRUForecaster but with the deeper
        # epoch budget already in base_hparams; for true LSTM-residual we
        # instantiate a custom variant via the registry override.
        from src.models.cha_hybrid import CHAHybridForecaster
        from src.models.deep_darts import LSTMForecaster

        class _CHA_LSTM_Residual(CHAHybridForecaster):
            name = "cha_hybrid_altres_lstm"

            def fit(self, train, val=None, *, train_series=None, val_series=None):
                # patch self.gru_model construction to use LSTM
                orig_gru_cls = LSTMForecaster
                # temporarily monkey-patch: GRUForecaster -> LSTMForecaster
                import src.models.cha_hybrid as cha_mod

                _orig = cha_mod.GRUForecaster
                cha_mod.GRUForecaster = LSTMForecaster
                try:
                    out = super().fit(
                        train, val, train_series=train_series, val_series=val_series
                    )
                finally:
                    cha_mod.GRUForecaster = _orig
                return out

        t0 = time.perf_counter()
        m2 = _CHA_LSTM_Residual(
            horizon=horizon, hparams=dict(base_hparams), seed=seed, device=device
        )
        m2.fit(train_ws, val_ws, train_series=train_series, val_series=val_series)
        fit_s = time.perf_counter() - t0
        n_params = m2.fit_report.n_parameters if m2.fit_report else None
        t0 = time.perf_counter()
        preds = m2.predict(test_ws)
        pred_s = time.perf_counter() - t0
        chosen = dict(base_hparams)
        chosen["residual_model"] = "lstm"
        _save_run(
            dataset,
            variant_altres,
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
        results[variant_altres] = "ok"
        logger.info(
            "[ablation] %s h=%d s=%d altres_lstm fit done in %.1fs",
            dataset,
            horizon,
            seed,
            fit_s,
        )

    return results
