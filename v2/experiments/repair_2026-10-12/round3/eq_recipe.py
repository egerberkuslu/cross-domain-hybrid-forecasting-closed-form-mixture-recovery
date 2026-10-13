"""FROZEN E-Q training recipe for round 3 (REPAIR_PREREG_3.md section 2). Nothing here may change after the commit.

The recipe is the round 2 expert E-Q of round2/experts.py (features, quantile gradient boosting) with ONE pre-stated change: for the test forecasts the
model is refitted on train + validation windows (same hyper-parameters). Bolt is the CHA-S Chronos-Bolt-small zero-shot forecast, computed in the
shipped environment (.chas-venv, torch CPU float32, batch 32). `recipe_manifest()` records every hash that identifies the recipe.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R2 = HERE.parent / "paper_a" / "round2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "paper_a"))
sys.path.insert(0, str(R2))
import experts as X  # noqa: E402  (round 2 module, frozen: its sha256 is in the recipe manifest)

QS = X.QS
GB = dict(X.GB)
BOLT_ID = "amazon/chronos-bolt-small"


def fit_eq(Xw, yw, tt, h, null_seed=None):
    """Quantile models (0.1, 0.5, 0.9) of E-Q, or, if null_seed is given, the median model of the null expert E-Q0 trained on rows shuffled across windows."""
    F = X.features(Xw.astype(np.float64), tt, h)
    T = X.targets(yw.astype(np.float64), Xw.astype(np.float64))
    if null_seed is not None:
        perm = np.random.default_rng(null_seed).permutation(len(F))
        return X.hgb(loss="quantile", quantile=0.5, **GB).fit(F[perm], T)
    return {a: X.hgb(loss="quantile", quantile=a, **GB).fit(F, T) for a in QS}


def predict_eq(model, Xw, tt, h):
    Xw = Xw.astype(np.float64)
    F = X.features(Xw, tt, h)
    n = len(Xw)
    if isinstance(model, dict):
        return {k: model[a].predict(F).reshape(n, h) + Xw[:, -1:] for k, a in (("q50", 0.5), ("q10", 0.1), ("q90", 0.9))}
    return {"q50": model.predict(F).reshape(n, h) + Xw[:, -1:]}


def bolt_median(Xw, h):
    """Chronos-Bolt-small median forecast of every window (zero-shot, deterministic in the shipped environment)."""
    return X.bolt_quantiles(Xw.astype(np.float32), h, "cpu")[:, 4, :].astype(np.float64)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recipe_manifest():
    snap = next((Path.home() / ".cache/huggingface/hub/models--amazon--chronos-bolt-small/snapshots").iterdir())
    return {
        "expert": "E-Q (quantile gradient boosting on lag and calendar features, median point forecast)",
        "training_data": {"stage_A_validation_forecasts": "train windows only", "stage_B_test_forecasts": "train + validation windows (refit, same hyper-parameters)"},
        "null_expert": "E-Q0: same recipe, median model, feature rows shuffled across windows, permutation seed 0",
        "hyperparameters": {"quantiles": list(QS), **{k: v for k, v in GB.items()}},
        "features_source": {"file": "experiments/repair_2026-10-12/paper_a/round2/experts.py", "sha256": sha(R2 / "experts.py"), "function": "features(X, tt, h)"},
        "mixture": "M2: f = Bolt + alpha (E-Q - Bolt), alpha = Bates-Granger closed form on ALL validation windows (E-Q trained on train only), clipped to [0, 1] and snapped to the 0.05 grid",
        "bolt": {"model": BOLT_ID, "hf_snapshot": snap.name, "weights_sha256": sha(snap / "model.safetensors"), "device": "cpu", "dtype": "float32", "batch": 32},
        "environment": {"sklearn": md.version("scikit-learn"), "numpy": np.__version__, "chronos-forecasting": md.version("chronos-forecasting"), "python": sys.version.split()[0]},
        "round2_models_hash_file": "experiments/repair_2026-10-12/paper_a/round2/MODELS_SHA256.txt",
    }


if __name__ == "__main__":
    m = recipe_manifest()
    m["eq_recipe.py_sha256"] = sha(HERE / "eq_recipe.py")
    (HERE / "RECIPE_FROZEN.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))
