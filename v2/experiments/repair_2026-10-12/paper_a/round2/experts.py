"""Round 2, part 1: candidate complementary experts for the CHA-S mixture. REPAIR_PREREG_2.md section 2.

Experts (all trained on TRAIN windows only, deterministic or fixed-seed, validation forecasts in the scaled domain):
  EQ  quantile gradient-boosted expert on lag and calendar features (median = point forecast, q90 - q10 = scale)
  ER  regime-aware seasonal-level ridge expert (regime = tercile of trailing volatility ratio, ridge per regime and step)
  EB  Bolt-conditioned residual expert (HistGB on features and Chronos-Bolt quantile summary, forecast = Bolt median + residual)
  EX  XGBoost lag-feature model as in the published pipeline (reference only, never a headline)
  null variants (EQ0, EB0): same family trained on feature rows shuffled across training windows (family-matched null expert)

Stages:  --stage val   writes val_experts/<expert>__<ds>__h<h>.npz and models/ (TRAIN and VAL windows only)
         --stage test  needs PREREG_2_COMMITTED + REPAIR_STAGE=test; loads the saved models and writes test_experts/ (phase 2)
A restricted view of the preprocessed dataset exposes only train and val windows in the val stage; the test windows object is
never touched. --dry-run runs one small cell and asserts that the test windows were not accessed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
ROOT = P1.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(P1))
import rcommon as R  # noqa: E402  (phase-1 data-role layer, guard)

DATASETS, HORIZONS = R.DATASETS, R.HORIZONS
LAGS = [1, 2, 3, 4, 6, 12, 24, 48, 72, 168]
QS = (0.1, 0.5, 0.9)
GB = dict(max_iter=150, learning_rate=0.06, max_depth=4, min_samples_leaf=40, l2_regularization=1.0, early_stopping=False, random_state=0)
GB_B = dict(max_iter=100, learning_rate=0.03, max_depth=3, min_samples_leaf=60, l2_regularization=5.0, early_stopping=False, random_state=0)
MODELS = HERE / "models"
VALX = HERE / "val_experts"
TESTX = HERE / "test_experts"


class Restricted:
    """View of a PreprocessedDataset that exposes train and val windows only (test access raises)."""

    def __init__(self, pp, allow_test=False):
        self._pp, self._allow = pp, allow_test
        self.accessed = []

    def windows(self, h, split):
        if split == "test" and not self._allow:
            raise PermissionError("test windows are sealed in the val stage")
        self.accessed.append((h, split))
        return self._pp.windows[h][split]


def features(X, tt, h, bq=None):
    """Row features for every (window, step k): returns F (n*h, d), index arrays (win, k)."""
    n = X.shape[0]
    last = X[:, -1]
    base = [last]
    for j in LAGS[1:]:
        base.append(X[:, -j] - last)
    base += [X[:, -6:].mean(1) - last, X[:, -24:].mean(1) - last, X[:, -168:].mean(1) - last, X[:, -24:].std(1), X[:, -168:].std(1) + 1e-6]
    i = np.arange(24) - 11.5
    base.append((X[:, -24:] * i[None]).sum(1) / (i * i).sum())
    B = np.stack(base, 1)
    first = tt - np.timedelta64(1, "h") * (h - 1)  # timestamp of the first target step
    hour0 = np.asarray(first.hour, float)
    dow0 = np.asarray(first.dayofweek, float)
    rows = []
    for k in range(h):
        hr = (hour0 + k) % 24
        dw = (dow0 + ((hour0 + k) // 24)) % 7
        cal = np.stack([np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24), np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7), np.full(n, k, float)], 1)
        cols = [B, cal]
        if bq is not None:  # bq: (n, 9, h) Chronos-Bolt quantiles
            med = bq[:, 4, k]
            cols.append(np.stack([med - last, bq[:, 8, k] - bq[:, 0, k], bq[:, 6, k] - bq[:, 2, k], bq[:, 4, :].mean(1) - last], 1))
        rows.append(np.concatenate(cols, 1))
    F = np.stack(rows, 1).reshape(n * h, -1)  # window-major
    return F


def targets(y, X):
    return (y - X[:, -1:]).reshape(-1)


# --------------------------------------------------------------- ER (ridge)
def er_design(X, h):
    last = X[:, -1]
    n = X.shape[0]
    out = np.zeros((n, h, 5))
    for k in range(h):
        sp = np.mean([X[:, 168 + k - 24 * j] for j in range(1, 8) if 168 + k - 24 * j >= 0], axis=0)
        out[:, k, 0] = sp - last
        out[:, k, 1] = X[:, -24:].mean(1) - last
        out[:, k, 2] = X[:, -168:].mean(1) - last
        out[:, k, 3] = last - X[:, -2]
        out[:, k, 4] = X[:, -6:].mean(1) - last
    return out


def er_regime(X):
    return X[:, -24:].std(1) / (X[:, -168:].std(1) + 1e-6)


def er_fit(Xtr, ytr, h):
    r = er_regime(Xtr)
    th = np.quantile(r, [1 / 3, 2 / 3])
    D = er_design(Xtr, h)
    T = ytr - Xtr[:, -1:]
    coefs = {}
    for g in range(3):
        m = np.digitize(r, th) == g
        if m.sum() < 30:
            m = np.ones(len(r), bool)
        for k in range(h):
            Z = np.concatenate([D[m, k, :], np.ones((m.sum(), 1))], 1)
            lam = 1.0 * m.sum()
            A = Z.T @ Z + lam * np.eye(Z.shape[1])
            A[-1, -1] -= lam  # intercept unpenalised
            coefs[(g, k)] = np.linalg.solve(A, Z.T @ T[m, k])
    return {"th": th, "coefs": coefs}


def er_predict(model, X, h):
    D = er_design(X, h)
    g = np.digitize(er_regime(X), model["th"])
    out = np.zeros((X.shape[0], h))
    for gi in range(3):
        m = g == gi
        if not m.any():
            continue
        for k in range(h):
            Z = np.concatenate([D[m, k, :], np.ones((m.sum(), 1))], 1)
            out[m, k] = Z @ model["coefs"][(gi, k)]
    return out + X[:, -1:]


# --------------------------------------------------------------- Bolt
_BOLT = {}


def bolt_quantiles(X, h, device="cuda"):
    """Chronos-Bolt-small quantiles (n, 9, h), batched, zero-shot (the pipeline of the shipped CHA-S)."""
    import torch
    from chronos import ChronosBoltPipeline
    if "p" not in _BOLT:
        _BOLT["p"] = ChronosBoltPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=device if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16 if (device == "cuda" and torch.cuda.is_available()) else torch.float32)
    p = _BOLT["p"]
    out = np.zeros((X.shape[0], 9, h), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 32):
            ctx = [torch.tensor(x, dtype=torch.float32) for x in X[s:s + 32]]
            q = p.predict(ctx, prediction_length=h, limit_prediction_length=False)
            out[s:s + len(ctx)] = q.detach().float().cpu().numpy()
    return out


# --------------------------------------------------------------- training / prediction
def hgb(**kw):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(**kw)


def train_cell(ds, h, rv, dry=False, device="cuda"):
    import joblib
    wtr, wva = rv.windows(h, "train"), rv.windows(h, "val")
    Xtr, ytr, ttr = wtr.X.astype(np.float64), wtr.y.astype(np.float64), wtr.target_times
    Xva, yva, tva = wva.X.astype(np.float64), wva.y.astype(np.float64), wva.target_times
    if dry:
        sel = np.arange(0, len(Xtr), 4)
        Xtr, ytr, ttr = Xtr[sel], ytr[sel], ttr[sel]
    res = {}
    t0 = time.perf_counter()
    # Bolt quantiles on train and val windows
    bq_tr = bolt_quantiles(Xtr.astype(np.float32), h, device)
    bq_va = bolt_quantiles(Xva.astype(np.float32), h, device)
    res["bolt_val_median"] = bq_va[:, 4, :]
    # EQ
    Ftr, Fva = features(Xtr, ttr, h), features(Xva, tva, h)
    T = targets(ytr, Xtr)
    q = {}
    for a in QS:
        m = hgb(loss="quantile", quantile=a, **GB).fit(Ftr, T)
        q[a] = m
        res[f"EQ_q{int(a * 100)}"] = (m.predict(Fva).reshape(len(Xva), h) + Xva[:, -1:])
    res["EQ"] = res["EQ_q50"]
    models = {"EQ": q}
    # EQ0 null expert: rows shuffled across training windows (labels intact, features decoupled)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Ftr))
    m0 = hgb(loss="quantile", quantile=0.5, **GB).fit(Ftr[perm], T)
    res["EQ0"] = m0.predict(Fva).reshape(len(Xva), h) + Xva[:, -1:]
    models["EQ0"] = m0
    # ER
    er = er_fit(Xtr, ytr, h)
    res["ER"] = er_predict(er, Xva, h)
    models["ER"] = er
    # EB: residual of Bolt median
    Gtr, Gva = features(Xtr, ttr, h, bq_tr), features(Xva, tva, h, bq_va)
    Tb = (ytr - bq_tr[:, 4, :]).reshape(-1)
    mb = hgb(loss="squared_error", **GB_B).fit(Gtr, Tb)
    res["EB"] = bq_va[:, 4, :] + mb.predict(Gva).reshape(len(Xva), h)
    models["EB"] = mb
    mb0 = hgb(loss="squared_error", **GB_B).fit(Gtr[perm], Tb)
    res["EB0"] = bq_va[:, 4, :] + mb0.predict(Gva).reshape(len(Xva), h)
    models["EB0"] = mb0
    # EX reference (published pipeline configuration)
    from src.models.xgboost_model import XGBoostForecaster
    xg = XGBoostForecaster(horizon=h, hparams={"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05}, seed=42)
    xg._fit_arrays(Xtr.astype(np.float32), ytr.astype(np.float32))
    res["EX"] = xg._predict_arrays(Xva.astype(np.float32)).astype(np.float64)
    models["EX"] = xg.model
    res["y_val"] = yva
    res["seconds"] = time.perf_counter() - t0
    if not dry:
        joblib.dump(models, MODELS / f"{ds}__h{h}.joblib")
    return res, bq_va


def predict_cell(models, X, tt, h, bq, which=("EQ", "EQ0", "ER", "EB", "EB0", "EX")):
    """Forecasts of the saved models for any windows (used by the phase-2 test stage and by the consistency check)."""
    X = X.astype(np.float64)
    F = features(X, tt, h)
    G = features(X, tt, h, bq)
    out = {}
    n = len(X)
    if "EQ" in which:
        out["EQ"] = models["EQ"][0.5].predict(F).reshape(n, h) + X[:, -1:]
        out["EQ_q10"] = models["EQ"][0.1].predict(F).reshape(n, h) + X[:, -1:]
        out["EQ_q90"] = models["EQ"][0.9].predict(F).reshape(n, h) + X[:, -1:]
    if "EQ0" in which:
        out["EQ0"] = models["EQ0"].predict(F).reshape(n, h) + X[:, -1:]
    if "ER" in which:
        out["ER"] = er_predict(models["ER"], X, h)
    if "EB" in which:
        out["EB"] = bq[:, 4, :] + models["EB"].predict(G).reshape(n, h)
    if "EB0" in which:
        out["EB0"] = bq[:, 4, :] + models["EB0"].predict(G).reshape(n, h)
    if "EX" in which:
        out["EX"] = models["EX"].predict(X.astype(np.float32)).astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["val", "test"], default="val")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--horizons", nargs="*", type=int, default=HORIZONS)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from src.preprocessing import load_preprocessed
    R.install_guard()
    R.guard_selftest()
    if a.stage == "test":
        raise SystemExit("test stage is implemented in experts_test.py (phase 2)")
    cells = [("abilene", 6)] if a.dry_run else [(d, h) for d in a.datasets for h in a.horizons]
    VALX.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    cache = {}
    for ds, h in cells:
        if ds not in cache:
            cache[ds] = Restricted(load_preprocessed(ds))
        rv = cache[ds]
        out = VALX / f"{ds}__h{h}.npz"
        if out.exists() and not a.dry_run:
            continue
        res, bq = train_cell(ds, h, rv, dry=a.dry_run, device=a.device)
        # consistency with the stored validation forecasts of the shipped CHA-S
        v = R._raw_val(ds, h, 42)
        yv = res["y_val"]
        n = min(len(yv), len(v["y"]))
        d_y = float(np.max(np.abs(yv[:n] - v["y"][:n])))
        d_b = float(np.max(np.abs(res["bolt_val_median"][:n] - v["glob"][:n])))
        keys = [k for k in res if k not in ("seconds", "bolt_val_median", "y_val")]
        print(json.dumps({"cell": f"{ds}_h{h}", "sec": round(res["seconds"], 1), "max_dy_vs_stored": d_y, "max_dbolt_vs_stored": d_b,
                          "rmse": {k: round(float(np.sqrt(np.mean((res[k][:n] - yv[:n]) ** 2))), 4) for k in keys}, "rmse_bolt_stored": round(float(np.sqrt(np.mean((v['glob'][:n] - yv[:n]) ** 2))), 4)}), flush=True)
        assert all(s != "test" for _, s in rv.accessed), "test windows accessed"
        if not a.dry_run:
            np.savez_compressed(out, **{k: np.asarray(res[k], np.float32) for k in keys + ["y_val"]}, bolt_val=bq.astype(np.float32), d_y=d_y, d_bolt=d_b)
        R.progress(f"experts val {ds} h{h}")
    if a.dry_run:
        n_acc = [x for c in cache.values() for x in c.accessed]
        assert not any(s == "test" for _, s in n_acc)
        print("dry-run ok; test windows accessed: 0; accesses:", n_acc)


if __name__ == "__main__":
    main()
