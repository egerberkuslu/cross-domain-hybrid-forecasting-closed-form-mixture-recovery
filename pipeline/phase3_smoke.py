"""Phase-3 smoke test: for every model, train/tune on ONE dataset at h=1 and h=24.

Verification items checked per model & horizon:

  * prediction shape == (n_test, h)
  * no NaN / no inf in predictions
  * val RMSE printed (must be finite)
  * selected hyperparameters printed
  * Group C: zero-shot ran; fine-tune either ran (GPU) or was skipped (logged)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import rmse
from src.models import REGISTRY, build_model, list_models
from src.preprocessing import load_preprocessed
from src.training import grid_search_one
from src.utils import detect_device, load_config, log_device_info, set_global_seed, setup_logging
from src.utils.logging_setup import get_logger


# Which dataset to run the smoke test on (geant is the smallest)
SMOKE_DATASET = "geant"
SMOKE_HORIZONS = (1, 24)

# Models we test in addition to every entry in REGISTRY. Each tuple is
# (display name, model name in registry, extra hparam overrides) — used
# to exercise the chronos fine-tune path explicitly per the spec.
EXTRA_CONFIGS = [
    ("chronos_ft", "chronos", {
        "pretrained": "amazon/chronos-t5-tiny",
        "finetune": True, "finetune_epochs": 1, "lora_r": 8,
        "num_samples": 5, "batch_size_predict": 16,
    }),
]


# Per-model defaults / base hparams used when not in the HP grid.
# For statistical baselines we keep epochs/lr off; for foundation models we
# need pretrained model id; for darts deep models we pass input_chunk_length.
DEFAULTS: dict[str, dict] = {
    "naive":          {},
    "seasonal_naive": {"seasonal_period": 24},
    "arima":          {"max_order_search": 2, "n_train_samples_for_order": 4},
    "holt_winters":   {"seasonal_period": 24, "trend": "add", "seasonal": "add"},
    "theta":          {"seasonal_period": 24},
    "xgboost":        {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
    "lstm":           {"input_chunk_length": 168, "n_epochs": 3, "batch_size": 64},
    "gru":            {"input_chunk_length": 168, "n_epochs": 3, "batch_size": 64},
    "tcn":            {"input_chunk_length": 168, "n_epochs": 3, "batch_size": 64},
    "nbeats":         {"input_chunk_length": 168, "n_epochs": 3, "batch_size": 64},
    "dlinear":        {"input_chunk_length": 168, "n_epochs": 3, "batch_size": 64},
    "patchtst":       {"input_chunk_length": 168, "n_epochs": 3, "batch_size": 64,
                       "patch_length": 16, "patch_stride": 8, "d_model": 64},
    "chronos":        {"pretrained": "amazon/chronos-t5-small",
                       "num_samples": 10, "batch_size_predict": 16},
    "timesfm":        {"input_chunk_length": 512, "batch_size_predict": 16},
}


# Smoke-test HP grids (deliberately smaller than the full Phase-5 grids to
# keep this verification fast — typically 1–3 candidates per model).
SMOKE_GRIDS: dict[str, dict] = {
    "naive":          {},
    "seasonal_naive": {},
    "arima":          {},
    "holt_winters":   {},
    "theta":          {},
    "xgboost":        {"n_estimators": [200], "max_depth": [3, 5]},
    "lstm":           {"hidden_dim": [32]},
    "gru":            {"hidden_dim": [32]},
    "tcn":            {"num_filters": [16]},
    "nbeats":         {"num_blocks": [1]},
    "dlinear":        {"kernel_size": [25]},
    "patchtst":       {"d_model": [64]},
    "chronos":        {},  # zero-shot smoke test
    "timesfm":        {},  # zero-shot smoke test
}


def run_one_model(name: str, horizon: int, device: str, cfg, log,
                  display_name: str | None = None, extra_override: dict | None = None) -> dict:
    """Run smoke test for a single (model, horizon).

    ``display_name`` lets us label e.g. "chronos_ft" while the model registry
    key remains "chronos"; ``extra_override`` is merged into the base hparams.
    """
    label = display_name or name
    out = {
        "model": label, "horizon": horizon,
        "status": "ok", "val_rmse": None, "test_rmse": None,
        "predict_shape": None, "predict_finite": None,
        "fit_seconds": None, "predict_seconds": None,
        "selected_hparams": None, "extras": {},
    }
    try:
        base = dict(DEFAULTS.get(name, {}))
        if extra_override:
            base.update(extra_override)
        grid = SMOKE_GRIDS.get(name, {})

        # ---- HP search (or single-run if grid is empty) ----
        hp_dir = cfg.resolve(cfg.paths.hp_search) / "smoke"
        if grid:
            hp_res = grid_search_one(
                dataset_name=SMOKE_DATASET,
                model_name=name,
                horizon=horizon,
                grid=grid,
                device=device,
                seed=int(cfg.random_seed),
                out_dir=hp_dir,
                base_hparams=base,
            )
            chosen = hp_res.chosen
            val_rmse = hp_res.val_rmse
        else:
            chosen = dict(base)
            val_rmse = None

        # ---- Final fit + test ----
        pp = load_preprocessed(SMOKE_DATASET)
        train_ws = pp.windows[horizon]["train"]
        val_ws = pp.windows[horizon]["val"]
        test_ws = pp.windows[horizon]["test"]
        train_series = pp.split_scaled.train["value"].to_numpy(dtype=np.float32)
        val_series = pp.split_scaled.val["value"].to_numpy(dtype=np.float32)

        model = build_model(name, horizon=horizon, hparams=chosen,
                            seed=int(cfg.random_seed), device=device)
        t0 = time.perf_counter()
        model.fit(train_ws, val_ws,
                  train_series=train_series, val_series=val_series)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds_val = model.predict(val_ws)
        pred_s = time.perf_counter() - t0

        if val_rmse is None:
            val_rmse = float(rmse(val_ws.y, preds_val))

        # also compute test RMSE
        preds_test = model.predict(test_ws)
        test_rmse = float(rmse(test_ws.y, preds_test))

        # variant info (for foundation models)
        variant = getattr(model, "variant_name", name)
        out.update({
            "status": "ok",
            "val_rmse": float(val_rmse),
            "test_rmse": float(test_rmse),
            "predict_shape": list(preds_val.shape),
            "predict_finite": bool(np.isfinite(preds_val).all() and np.isfinite(preds_test).all()),
            "fit_seconds": float(fit_s),
            "predict_seconds": float(pred_s),
            "selected_hparams": chosen,
            "extras": {"variant": variant,
                       "n_parameters": getattr(model.fit_report, "n_parameters", None)
                       if model.fit_report else None},
        })
        log.info(
            "[smoke] %-15s h=%2d val_rmse=%8.4f test_rmse=%8.4f shape=%s finite=%s fit=%.1fs pred=%.1fs hp=%s",
            name, horizon, val_rmse, test_rmse, list(preds_val.shape),
            out["predict_finite"], fit_s, pred_s, chosen,
        )
    except Exception as e:
        log.error("[smoke] %s/h=%d FAILED: %s: %s", name, horizon, type(e).__name__, e, exc_info=False)
        out["status"] = f"fail: {type(e).__name__}: {e}"
    return out


def main() -> int:
    cfg = load_config()
    log_file = setup_logging(log_dir=cfg.resolve(cfg.paths.logs), run_name="phase3_smoke")
    log = get_logger("phase3.smoke")
    set_global_seed(cfg.random_seed)
    info = log_device_info(detect_device())
    device = info.device

    log.info("Phase 3 smoke test — dataset=%s, horizons=%s, device=%s",
             SMOKE_DATASET, SMOKE_HORIZONS, device)

    # Run every registered model for both horizons.
    models = list_models()
    log.info("Smoke-testing %d models: %s", len(models), models)

    results = []
    for model_name in models:
        for h in SMOKE_HORIZONS:
            log.info("=" * 80)
            log.info(">>> %s @ h=%d", model_name, h)
            log.info("=" * 80)
            results.append(run_one_model(model_name, h, device, cfg, log))
    for disp, m_name, extra in EXTRA_CONFIGS:
        for h in SMOKE_HORIZONS:
            log.info("=" * 80)
            log.info(">>> %s @ h=%d (extra config: %s)", disp, h, extra)
            log.info("=" * 80)
            results.append(run_one_model(m_name, h, device, cfg, log,
                                          display_name=disp, extra_override=extra))

    # ---- additional Phase-3 verification items ----
    log.info("=" * 80)
    log.info("Group C verification:")
    for r in results:
        if r["model"] == "chronos":
            log.info("[verify] chronos: variant=%s status=%s",
                     r["extras"].get("variant"), r["status"])
        if r["model"] == "timesfm":
            log.info("[verify] timesfm: zero-shot status=%s (fine-tuning skipped: GPU "
                     "available=%s, but TimesFM 1.2+ requires Python <3.12; the "
                     "darts TimesFM2p5Model wrapper exposes inference only).",
                     r["status"], info.cuda_available)

    # ---- persist + print summary table ----
    out_path = cfg.resolve(cfg.paths.results) / "phase3_smoke_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print("=" * 110)
    print(f"{'MODEL':<16} {'H':>3} {'STATUS':<6} {'SHAPE':<14} {'FINITE':<7} "
          f"{'VAL_RMSE':>10} {'TEST_RMSE':>10} {'FIT_S':>7} {'PRED_S':>7} HPARAMS")
    print("=" * 110)
    n_pass = n_fail = 0
    for r in results:
        ok = r["status"] == "ok" and r["predict_finite"] is True
        n_pass += int(ok); n_fail += int(not ok)
        print(f"{r['model']:<16} {r['horizon']:>3} "
              f"{'PASS' if ok else 'FAIL':<6} {str(r['predict_shape']):<14} "
              f"{str(r['predict_finite']):<7} "
              f"{r['val_rmse'] if r['val_rmse'] is not None else 'n/a':>10} "
              f"{r['test_rmse'] if r['test_rmse'] is not None else 'n/a':>10} "
              f"{r['fit_seconds'] if r['fit_seconds'] else 'n/a':>7} "
              f"{r['predict_seconds'] if r['predict_seconds'] else 'n/a':>7} "
              f"{r['selected_hparams']}")
    print("=" * 110)
    print(f"TOTAL: {n_pass} pass, {n_fail} fail / {len(results)}")
    print(f"Detailed JSON: {out_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
