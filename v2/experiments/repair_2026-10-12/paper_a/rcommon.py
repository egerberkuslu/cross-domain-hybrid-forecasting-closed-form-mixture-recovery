"""Shared data-role layer for the Paper A repair round (A-A1, A-A2, A-A3).

Roles (pre-registered in REPAIR_PREREG.md):
  VALIDATION  experiments/fix_2026-10-11/val_forecasts/*.npz   (read by every fit and selection stage)
  TEST        outputs/predictions/*.npz                         (read ONLY by the --stage test code paths)

Test access is physically gated:
  * install_guard() wraps numpy.load and builtins.open and raises TestAccessError for any path under
    outputs/predictions or outputs/checkpoints unless allow_test() was called;
  * allow_test() refuses unless the file PREREG_COMMITTED exists next to this module AND the
    environment variable REPAIR_STAGE=test is set;
  * the dry-run installs the guard and never calls allow_test().
"""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # /home/ege/Desktop/zaman-serisi-yayin
FIX = ROOT / "experiments" / "fix_2026-10-11"
sys.path.insert(0, str(FIX))
import _common as C  # noqa: E402

DATASETS, HORIZONS, SEEDS = C.DATASETS, C.HORIZONS, C.SEEDS
CELLS = [(d, h) for d in DATASETS for h in HORIZONS]  # 25 cells
GRID_STEP = 0.05

_TEST_ALLOWED = False
_GUARDED = (str(C.PRED), str(C.CKPT.parent / "predictions"))


class TestAccessError(RuntimeError):
    pass


def guard_selftest():
    """Prove the guard bites: a real test read must raise while sealed."""
    if _TEST_ALLOWED:
        return
    try:
        C.load_pred("abilene", C.DEC, 6, 42)
    except TestAccessError:
        return
    raise AssertionError("guard did not block a test read")


def allow_test():
    global _TEST_ALLOWED
    if not (HERE / "PREREG_COMMITTED").exists():
        raise TestAccessError("PREREG_COMMITTED file missing: test data stay sealed")
    if os.environ.get("REPAIR_STAGE") != "test":
        raise TestAccessError("REPAIR_STAGE=test not set")
    _TEST_ALLOWED = True


def _check(path):
    s = str(path)
    if not _TEST_ALLOWED and any(s.startswith(g) for g in _GUARDED):
        raise TestAccessError(f"test data read attempted: {s}")


_ORIG_LOAD, _ORIG_OPEN = np.load, builtins.open
_READ_LOG = []


def install_guard():
    def g_load(f, *a, **k):
        if isinstance(f, (str, os.PathLike)):
            _check(f)
            _READ_LOG.append(str(f))
        return _ORIG_LOAD(f, *a, **k)

    def g_open(f, *a, **k):
        if isinstance(f, (str, os.PathLike)):
            _check(f)
        return _ORIG_OPEN(f, *a, **k)

    np.load = g_load
    builtins.open = g_open
    Path_open = Path.open

    def p_open(self, *a, **k):
        _check(self)
        return Path_open(self, *a, **k)

    Path.open = p_open
    Path.read_text_orig = Path.read_text

    def p_read_text(self, *a, **k):
        _check(self)
        return Path.read_text_orig(self, *a, **k)

    Path.read_text = p_read_text
    return _READ_LOG


# ----------------------------------------------------------------- loaders
PSEUDO = False  # dry-run: CAL slice plays "validation", SEL slice plays "test"; nothing under outputs/predictions is read
OUT = HERE


def enable_pseudo():
    global PSEUDO, OUT
    PSEUDO = True
    OUT = HERE / "dryrun_out"
    OUT.mkdir(exist_ok=True)


def split_val(n, h):
    """CAL = [0, nc), purge gap h, SEL = [nc + h, n)."""
    nc = int(0.6 * n)
    return nc, nc + h


def _raw_val(ds, h, seed):
    v = C.load_val_forecast(ds, h, seed)
    if v is None:
        raise FileNotFoundError(f"validation forecast missing {ds} h{h} s{seed}")
    return v


def load_val(ds, h, seed):
    """Validation forecasts in the scaled domain: dict(y, dec, glob), shapes (n, h)."""
    v = _raw_val(ds, h, seed)
    if PSEUDO:
        nc, _ = split_val(len(v["y"]), h)
        return {k: x[:nc] for k, x in v.items()}
    return v


def deployed_alpha(ds, h, seed):
    """Alpha the shipped CHA-S uses (grid search on validation, stored in closed_form_cells.csv)."""
    import pandas as pd

    if PSEUDO:
        v = load_val(ds, h, seed)
        return snap(alpha_bg(*window_stats(v["y"], v["dec"], v["glob"])[1:]))
    global _AL
    try:
        _AL
    except NameError:
        _AL = pd.read_csv(FIX / "closed_form_cells.csv")
        _AL = _AL[_AL.variant == C.CHAS]
    r = _AL[(_AL.dataset == ds) & (_AL.horizon == h) & (_AL.seed == seed)]
    return float(r.alpha_grid.iloc[0])


def load_test(ds, h, seed, native=False):
    """TEST forecasts (scaled or native): dict(y, dec, glob, chas). Gated by allow_test()."""
    if PSEUDO:
        v = _raw_val(ds, h, seed)
        nc, s0 = split_val(len(v["y"]), h)
        a0 = deployed_alpha(ds, h, seed)
        y, dec, glob = (v[k][s0:] for k in ("y", "dec", "glob"))
        return {"y": y, "dec": dec, "glob": glob, "chas": mix(dec, glob, a0)}
    if not _TEST_ALLOWED:
        raise TestAccessError("load_test called while test is sealed")
    d = C.load_pred(ds, C.DEC, h, seed, native=native)
    b = C.load_bolt(ds, h, native=native)
    c = C.load_pred(ds, C.CHAS, h, seed, native=native)
    y, dec, glob, chas = C.align(d[0], d[1], b[1], c[1])
    return {"y": y, "dec": dec, "glob": glob, "chas": chas}


# ----------------------------------------------------------- closed forms
def window_stats(y, dec, glob):
    """Per-window sufficient statistics of the mixture error e = (glob-y) + a (dec-glob)."""
    eg, df = glob - y, dec - glob
    return (eg * eg).sum(1), (eg * df).sum(1), (df * df).sum(1)  # see, sed, sdd


def alpha_bg(sed, sdd):
    den = float(np.sum(sdd))
    return 0.0 if den < 1e-12 else float(np.clip(-np.sum(sed) / den, 0.0, 1.0))


def mse_from_stats(a, see, sed, sdd, h):
    n = len(see)
    return float((see.sum() + 2.0 * a * sed.sum() + a * a * sdd.sum()) / (n * h))


def mix(dec, glob, alpha):
    a = np.asarray(alpha, float)
    if a.ndim == 1:
        a = a[:, None]
    return glob + a * (dec - glob)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def snap(a):
    return float(np.clip(np.round(a / GRID_STEP) * GRID_STEP, 0.0, 1.0))


# ---------------------------------------------------------------- utility
def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


_LAST_PROG = [0.0]


def progress(msg, force=False):
    """Append to PROGRESS.md at most once per 5 minutes (or when forced)."""
    now = time.time()
    if not force and now - _LAST_PROG[0] < 300:
        return
    _LAST_PROG[0] = now
    with _ORIG_OPEN(HERE / "PROGRESS.md", "a") as f:
        f.write(f"- {stamp()} {msg}\n")


def holm(p):
    return C.holm(p)


def sign_flip_block_perm(d, block, B=20000, seed=0):
    """Two-sided sign-flip permutation on window-level differences d (n,), flips in contiguous blocks."""
    d = np.asarray(d, float)
    n = len(d)
    nb = int(np.ceil(n / block))
    bid = np.minimum(np.arange(n) // block, nb - 1)
    bs = np.bincount(bid, weights=d, minlength=nb)  # block sums
    t_obs = abs(bs.sum())
    rng = np.random.default_rng(seed)
    out = 0
    step = 2000
    for s in range(0, B, step):
        m = min(step, B - s)
        sg = rng.choice([-1.0, 1.0], size=(m, nb))
        out += int(np.sum(np.abs(sg @ bs) >= t_obs - 1e-15))
    return (out + 1) / (B + 1)
