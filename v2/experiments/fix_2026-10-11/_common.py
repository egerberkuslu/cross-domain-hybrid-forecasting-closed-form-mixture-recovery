"""Shared helpers for the v2 recomputation scripts (Paper A).

Every script in this folder imports from here. Rules enforced here:
  * inputs are read from the EXISTING artefacts only (outputs/predictions,
    outputs/checkpoints/*.pt.json, outputs/eval_v3/...); nothing is retrained;
  * outputs go to this folder only, and the writer never overwrites: if the
    target name exists, a timestamped sibling is written instead;
  * no torch / darts import here, so every script except
    val_expert_forecasts.py runs on plain numpy/scipy/pandas.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CHA_ROOT", HERE.parents[2]))  # repository root (holds outputs/)
PRED = ROOT / "outputs" / "predictions"
CKPT = ROOT / "outputs" / "checkpoints"
EVAL = ROOT / "outputs" / "eval_v3"
VAL_DIR = HERE / "val_forecasts"  # written by val_expert_forecasts.py

DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
HORIZONS = [1, 3, 6, 12, 24]
SEEDS = [42, 123, 2024, 31337, 7]
DNAME = {
    "cesnet": "CESNET-TS24",
    "abilene": "Abilene",
    "geant": "GEANT",
    "nab_aws_cpu": "NAB AWS-CPU",
    "nab_twitter": "NAB Twitter",
}
BOLT = "chronos_bolt_zs"
CHAS = "cha_hybrid_v3"
DEC = "cha_hybrid_v3_decomp_only"
# the seven v3 variants that make up the 825-cell closed-form table
V3_VARIANTS = [
    "cha_hybrid_v3",
    "cha_hybrid_v3_tiny",
    "cha_hybrid_v3_mini",
    "cha_hybrid_v3_base",
    "cha_hybrid_v3_stl12",
    "cha_hybrid_v3_stl48",
    "cha_hybrid_v3_stl168",
]
# generation sweep (generation sweep): expert -> checkpoint variant
SWAP_VARIANTS = {
    "chronos_bolt_small": "cha_hybrid_v3",
    "chronos2": "cha_hybrid_swap_chronos2",
    "chronos2_small": "cha_hybrid_swap_chronos2_small",
    "timesfm3": "cha_hybrid_swap_timesfm3",
    "flowstate": "cha_hybrid_swap_flowstate",
    "tirex": "cha_hybrid_swap_tirex",
}
GRID_STEP = 0.05  # 21-point grid used by every v3 / swap checkpoint


# ----------------------------------------------------------------- loading
def npz_path(ds: str, variant: str, h: int, seed: int) -> Path:
    return PRED / f"{ds}__{variant}__h{h}__s{seed}.npz"


def load_pred(ds: str, variant: str, h: int, seed: int, native: bool = False):
    """Return (y_true, y_pred) as float64 arrays of shape (N, h), or None."""
    p = npz_path(ds, variant, h, seed)
    if not p.exists():
        return None
    d = np.load(p)
    k = "native" if native else "scaled"
    return (
        np.asarray(d[f"y_true_{k}"], dtype=np.float64),
        np.asarray(d[f"y_pred_{k}"], dtype=np.float64),
    )


def load_bolt(ds: str, h: int, native: bool = False, expert: str = BOLT):
    """Zero-shot experts are deterministic and stored for seed 42 only."""
    return load_pred(ds, expert, h, 42, native=native)


def align(*arrs):
    n = min(a.shape[0] for a in arrs)
    return [a[:n] for a in arrs]


def ckpt_json(ds: str, variant: str, h: int, seed: int) -> dict | None:
    p = CKPT / f"{ds}__{variant}__h{h}__s{seed}.pt.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def val_grid(ck: dict):
    """(alphas, val_mse) from the checkpoint's alpha_search_diag (RMSE^2)."""
    diag = ck.get("alpha_search_diag") or []
    if not diag:
        return None
    a = np.array([float(r["alpha"]) for r in diag])
    m = np.array([float(r["val_rmse"]) ** 2 for r in diag])
    o = np.argsort(a)
    return a[o], m[o]


def three_points(alphas: np.ndarray, mses: np.ndarray):
    def at(x):
        return float(mses[int(np.argmin(np.abs(alphas - x)))])

    return at(0.0), at(0.5), at(1.0)


def load_val_forecast(ds: str, h: int, seed: int):
    """Validation-split expert forecasts produced by val_expert_forecasts.py.

    Returns dict(y, dec, glob) with shapes (N_val, h), or None if step 0 has
    not been run for this cell.
    """
    p = VAL_DIR / f"{ds}__h{h}__s{seed}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {
        "y": np.asarray(d["y_val"], dtype=np.float64),
        "dec": np.asarray(d["dec_val"], dtype=np.float64),
        "glob": np.asarray(d["glob_val"], dtype=np.float64),
    }


# --------------------------------------------------------- closed form (BG)
def bg_from_three(m0: float, mh: float, m1: float):
    """Bates-Granger weight from M(0), M(1/2), M(1).

    Quadratic identity (exact for the mixture MSE):
        M(a) = (1-a)(1-2a) M(0) + 4a(1-a) M(1/2) + a(2a-1) M(1)
    so A = 2M(0)+2M(1)-4M(1/2) (leading coefficient = E[(e_dec-e_glob)^2])
    and the stationary point is
        a* = (3M(0) + M(1) - 4M(1/2)) / (4 (M(0) + M(1) - 2M(1/2))).
    Returns (a_unclipped, a_clipped, A).
    """
    A = 2.0 * m0 + 2.0 * m1 - 4.0 * mh
    if A <= 0:
        a_star = float("nan")
        a_clip = 0.0 if m0 <= m1 else 1.0
        return a_star, a_clip, A
    a_star = (3.0 * m0 + m1 - 4.0 * mh) / (2.0 * A)
    return float(a_star), float(np.clip(a_star, 0.0, 1.0)), float(A)


def quad_M(a, m0: float, mh: float, m1: float):
    a = np.asarray(a, dtype=float)
    return (1 - a) * (1 - 2 * a) * m0 + 4 * a * (1 - a) * mh + a * (2 * a - 1) * m1


def bg_from_errors(y, f_dec, f_glob):
    """Clipped BG optimum of mean((a f_dec + (1-a) f_glob - y)^2) on given data."""
    ed = np.ravel(f_dec - y)
    ef = np.ravel(f_glob - y)
    diff = ed - ef
    den = float(np.dot(diff, diff))
    if den < 1e-12:
        return 0.0
    return float(np.clip(-np.dot(ef, diff) / den, 0.0, 1.0))


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# ------------------------------------------------------ multiple testing
def holm(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    o = np.argsort(p)
    adj = np.empty(n)
    run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (n - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def bh(p):
    p = np.asarray(p, dtype=float)
    n = p.size
    o = np.argsort(p)[::-1]
    adj = np.empty(n)
    run = 1.0
    for rank_from_top, idx in enumerate(o):
        k = n - rank_from_top  # 1-based rank in ascending order
        run = min(run, p[idx] * n / k)
        adj[idx] = run
    return adj


# ---------------------------------------------------------------- output
def safe_path(name: str) -> Path:
    """Path in this folder that never overwrites an existing file."""
    p = HERE / name
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    ts = time.strftime("%Y%m%d-%H%M%S")
    q = HERE / f"{stem}__{ts}{suf}"
    i = 1
    while q.exists():
        q = HERE / f"{stem}__{ts}_{i}{suf}"
        i += 1
    print(f"[safe] {name} exists, writing {q.name} instead")
    return q


def write_csv(df, name: str) -> Path:
    p = safe_path(name)
    df.to_csv(p, index=False)
    print(f"[write] {p}")
    return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(obj, name: str) -> Path:
    p = safe_path(name)
    p.write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"[write] {p}")
    return p


def env_threads(n: int = 4) -> None:
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, str(n))
