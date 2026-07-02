"""Phase-4 driver — smoke + verification for the proposed CHA-Hybrid model.

Per the Phase-4 spec:
  * STL decomposition: assert trend + seasonal + residual reconstructs the
    original series within tolerance
  * print the tuned alpha_h per horizon
  * smoke-test predictions on one dataset (geant, the smallest)
  * assert no NaN in predictions
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from statsmodels.tsa.seasonal import STL

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import rmse, compute_all
from src.models import build_model
from src.preprocessing import load_preprocessed
from src.utils import detect_device, load_config, log_device_info, set_global_seed, setup_logging
from src.utils.logging_setup import get_logger
from src.utils.io import write_json


SMOKE_DATASET = "geant"
SMOKE_HORIZONS = (1, 3, 6, 12, 24)


def stl_reconstruction_check(series: np.ndarray, period: int, tol_rel: float = 1e-3) -> dict:
    """Independently re-derive STL trend+seasonal+residual and assert sum-recon."""
    finite = np.isfinite(series)
    if finite.sum() < 2 * period + 8:
        return {"ok": False, "reason": "series too short for STL"}
    seg = series[finite].astype(np.float64)
    res = STL(seg, period=period, robust=False).fit()
    recon = np.asarray(res.trend + res.seasonal + res.resid, dtype=np.float64)
    max_abs = float(np.max(np.abs(recon - seg)))
    sd = float(np.std(seg))
    rel = float(max_abs / (sd + 1e-12))
    return {
        "ok": rel < tol_rel,
        "max_abs_err": max_abs,
        "rel_err_vs_std": rel,
        "tol_rel": tol_rel,
        "n": int(seg.size),
    }


def main() -> int:
    cfg = load_config()
    log_file = setup_logging(log_dir=cfg.resolve(cfg.paths.logs), run_name="phase4_cha")
    log = get_logger("phase4")
    set_global_seed(cfg.random_seed)
    info = log_device_info(detect_device())
    device = info.device

    log.info("Phase 4 smoke — dataset=%s, horizons=%s, device=%s",
             SMOKE_DATASET, SMOKE_HORIZONS, device)

    pp = load_preprocessed(SMOKE_DATASET)
    train_series = pp.split_scaled.train["value"].to_numpy(np.float32)
    val_series = pp.split_scaled.val["value"].to_numpy(np.float32)

    # ---- A) Independent STL reconstruction check on the training series ----
    stl_period = int(cfg.cha_hybrid.stl_period)
    stl_check = stl_reconstruction_check(train_series, period=stl_period, tol_rel=1e-3)
    log.info("[verify] STL reconstruction on %s train: %s", SMOKE_DATASET, stl_check)

    # ---- B) For each horizon: fit CHA-Hybrid, tune alpha_h on val, predict ----
    cfg_cha = cfg.cha_hybrid
    base_hparams = {
        "stl_period":     stl_period,
        "trend_model":    cfg_cha.trend_model,
        "seasonal_model": cfg_cha.seasonal_model,
        "residual_model": cfg_cha.residual_model,
        "global_model":   cfg_cha.global_model,
        "alpha_search":   list(cfg_cha.alpha_search),
        "gru": {
            "hidden_dim":  int(cfg_cha.gru.hidden_size),
            "n_rnn_layers":int(cfg_cha.gru.num_layers),
            "dropout":     float(cfg_cha.gru.dropout),
            "n_epochs":    int(cfg_cha.gru.max_epochs),
            "batch_size":  int(cfg_cha.gru.batch_size),
            "lr":          float(cfg_cha.gru.lr),
        },
        "lstm": {
            "hidden_dim":  int(cfg_cha.lstm.hidden_size),
            "n_rnn_layers":int(cfg_cha.lstm.num_layers),
            "dropout":     float(cfg_cha.lstm.dropout),
            "n_epochs":    int(cfg_cha.lstm.max_epochs),
            "batch_size":  int(cfg_cha.lstm.batch_size),
            "lr":          float(cfg_cha.lstm.lr),
        },
    }
    # Keep epochs modest for the smoke test (Phase 5 will run full schedule).
    smoke_epochs = 5
    base_hparams["gru"]["n_epochs"] = smoke_epochs
    base_hparams["lstm"]["n_epochs"] = smoke_epochs

    results = []
    chosen_alphas: dict[int, float] = {}
    for h in SMOKE_HORIZONS:
        log.info("=" * 80)
        log.info(">>> CHA-Hybrid @ h=%d", h)
        log.info("=" * 80)
        tw = pp.windows[h]["train"]
        vw = pp.windows[h]["val"]
        tew = pp.windows[h]["test"]
        try:
            t0 = time.perf_counter()
            m = build_model("cha_hybrid", horizon=h, hparams=base_hparams,
                            seed=int(cfg.random_seed), device=device)
            m.fit(tw, vw, train_series=train_series, val_series=val_series)
            fit_s = time.perf_counter() - t0

            pv = m.predict(vw)
            pt = m.predict(tew)
            # ablation predictions: decomp-only, global-only
            pv_decomp = m.predict_decomposition_only(vw)
            pv_global = m.predict_global_only(vw)
            pt_decomp = m.predict_decomposition_only(tew)
            pt_global = m.predict_global_only(tew)

            row = {
                "horizon": h,
                "alpha_h": float(m.alpha_h),
                "val_rmse_combined": float(rmse(vw.y, pv)),
                "test_rmse_combined": float(rmse(tew.y, pt)),
                "val_rmse_decomp_only": float(rmse(vw.y, pv_decomp)),
                "val_rmse_global_only": float(rmse(vw.y, pv_global)),
                "test_rmse_decomp_only": float(rmse(tew.y, pt_decomp)),
                "test_rmse_global_only": float(rmse(tew.y, pt_global)),
                "predict_shape_val": list(pv.shape),
                "predict_shape_test": list(pt.shape),
                "finite_val": bool(np.isfinite(pv).all()),
                "finite_test": bool(np.isfinite(pt).all()),
                "fit_seconds": float(fit_s),
                "n_parameters": int(m.fit_report.n_parameters) if m.fit_report and m.fit_report.n_parameters else None,
                "stl_recon_diag": m._train_recon_diag,
                "alpha_search_diag": m._val_alpha_search_diag,
            }
            log.info("[smoke] h=%2d alpha=%.2f val_rmse=%.4f test_rmse=%.4f "
                     "(decomp_only test=%.4f, global_only test=%.4f) fit=%.1fs",
                     h, row["alpha_h"], row["val_rmse_combined"],
                     row["test_rmse_combined"], row["test_rmse_decomp_only"],
                     row["test_rmse_global_only"], fit_s)
            chosen_alphas[h] = float(m.alpha_h)
            results.append(row)
        except Exception as e:
            import traceback; traceback.print_exc()
            log.error("[smoke] h=%d FAILED: %s: %s", h, type(e).__name__, e)
            results.append({"horizon": h, "error": f"{type(e).__name__}: {e}"})

    # ---- save + print summary ----
    out_path = cfg.resolve(cfg.paths.results) / "phase4_cha_hybrid_summary.json"
    write_json({
        "dataset": SMOKE_DATASET,
        "horizons": list(SMOKE_HORIZONS),
        "stl_check": stl_check,
        "chosen_alpha_per_horizon": chosen_alphas,
        "per_horizon_results": results,
    }, out_path)

    # ---- final verification table ----
    print()
    print("=" * 100)
    print("PHASE-4 VERIFICATION TABLE")
    print("=" * 100)
    print(f"STL reconstruction (train series, period={stl_period}, tol_rel=1e-3): "
          f"{'PASS' if stl_check.get('ok') else 'FAIL'} "
          f"(max|y - (trend+seas+resid)|={stl_check.get('max_abs_err'):.3e}, "
          f"rel-to-std={stl_check.get('rel_err_vs_std'):.3e}, n={stl_check.get('n')})")
    print()
    print(f"{'H':>3} {'ALPHA_H':>8} {'VAL_RMSE':>10} {'TEST_RMSE':>10} "
          f"{'DECOMP_ONLY':>12} {'GLOBAL_ONLY':>12} {'NaN-in-test':>12} {'FIT_S':>7}")
    print("-" * 100)
    n_pass = 0
    n_runs = 0
    for r in results:
        if "error" in r:
            n_runs += 1
            print(f"{r['horizon']:>3}  FAIL: {r['error']}")
            continue
        n_runs += 1
        ok = (
            r["finite_val"] and r["finite_test"]
            and tuple(r["predict_shape_test"][1:]) == (r["horizon"],)
        )
        n_pass += int(ok)
        print(f"{r['horizon']:>3} {r['alpha_h']:>8.2f} "
              f"{r['val_rmse_combined']:>10.4f} {r['test_rmse_combined']:>10.4f} "
              f"{r['test_rmse_decomp_only']:>12.4f} {r['test_rmse_global_only']:>12.4f} "
              f"{'False' if r['finite_test'] else 'True':>12} "
              f"{r['fit_seconds']:>7.1f}")
    print("-" * 100)
    print(f"TOTAL: {n_pass} pass / {n_runs} runs")
    print(f"Detailed JSON: {out_path}")
    return 0 if n_pass == n_runs else 1


if __name__ == "__main__":
    sys.exit(main())
