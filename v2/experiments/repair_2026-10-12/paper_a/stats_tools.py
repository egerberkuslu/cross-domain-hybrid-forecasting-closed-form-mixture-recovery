"""Statistics shared by a1.py, a2.py, a3.py (numpy only)."""
from __future__ import annotations

import numpy as np

import rcommon as R


def block_boot_ratio(mse_a, mse_b, block, B=5000, seed=0):
    """Moving-block (circular) bootstrap of RMSE_a/RMSE_b - 1 from per-window MSE arrays. Returns (point, lo95, hi95, lo90, hi90, rmse_diff_lo90, rmse_diff_hi90)."""
    a, b = np.asarray(mse_a, float), np.asarray(mse_b, float)
    n = len(a)
    nb = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    st = rng.integers(0, n, size=(B, nb))
    idx = ((st[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, -1)[:, :n]
    ma, mb = a[idx].mean(1), b[idx].mean(1)
    ra, rb = np.sqrt(ma), np.sqrt(mb)
    ratio = ra / np.maximum(rb, 1e-300) - 1.0
    diff = ra - rb
    point = float(np.sqrt(a.mean()) / np.sqrt(b.mean()) - 1.0)
    q = lambda x, p: float(np.quantile(x, p))
    return point, q(ratio, 0.025), q(ratio, 0.975), q(diff, 0.05), q(diff, 0.95)


def tost_equivalent(diff_lo90, diff_hi90, rmse_ref, margin=0.05):
    return bool(diff_lo90 > -margin * rmse_ref and diff_hi90 < margin * rmse_ref)


def perm_p(mse_a, mse_b, block, B=20000, seed=0):
    return R.sign_flip_block_perm(np.asarray(mse_a) - np.asarray(mse_b), block, B=B, seed=seed)


def cell_cluster_boot_mean(vals, cells, B=10000, seed=0, two_stage=False):
    """Percentile bootstrap CI of the mean of per-cell values. two_stage: resample datasets then horizons within."""
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    if not two_stage:
        idx = rng.integers(0, len(vals), size=(B, len(vals)))
        m = vals[idx].mean(1)
    else:
        ds = np.array([c[0] for c in cells])
        uds = sorted(set(ds))
        groups = {d: vals[ds == d] for d in uds}
        m = np.empty(B)
        for i in range(B):
            pick = rng.choice(len(uds), size=len(uds))
            acc = []
            for p in pick:
                g = groups[uds[p]]
                acc.append(g[rng.integers(0, len(g), size=len(g))])
            m[i] = np.concatenate(acc).mean()
    return float(vals.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def newcombe_diff(x1, n1, x2, n2, z=1.96):
    """Newcombe (method 10 / hybrid score) CI for p1 - p2."""
    def wilson(x, n):
        if n == 0:
            return 0.0, 1.0
        p = x / n
        den = 1 + z * z / n
        c = (p + z * z / (2 * n)) / den
        w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return c - w, c + w
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    p1, p2 = x1 / max(n1, 1), x2 / max(n2, 1)
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return float(d), float(lo), float(hi)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar from discordant counts b (A only) and c (B only)."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(b, n, 0.5).pvalue)


def wilson(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - w), float(c + w)


# ------------------------------------------------ block-bootstrap null calibration (A-A1 V3, A-A2)
PERSIST_DEFAULT = 3
K_CUSUM = 0.5
LEVEL_GRID = np.arange(1.0, 30.01, 0.5)
EMA_LAM = 9.0 / 11.0


def mbb_resample(x, block, B, rng):
    """Moving-block bootstrap rows of a 1-d series: (B, len(x))."""
    x = np.asarray(x, float)
    n = len(x)
    nb = int(np.ceil(n / block))
    st = rng.integers(0, max(1, n - block + 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return x[idx]


def cusum_S(Z, K=K_CUSUM):
    """CUSUM path S_k = max(0, S_{k-1} + z_k - K) for each row of Z (B, n). Independent of the level."""
    S = np.zeros_like(Z)
    cur = np.zeros(Z.shape[0])
    for k in range(Z.shape[1]):
        cur = np.maximum(0.0, cur + Z[:, k] - K)
        S[:, k] = cur
    return S


def onset_rate_levels(S_list, levels, persist=PERSIST_DEFAULT):
    """Alarm-onset share per (row, level) for hits = OR_j (S_j > level); onsets as in monitor_far.alarm_onsets."""
    B, n = S_list[0].shape
    L = len(levels)
    run = np.zeros((B, L), int)
    cnt = np.zeros((B, L), int)
    lv = np.asarray(levels)[None, :]
    for k in range(n):
        hit = np.zeros((B, L), bool)
        for S in S_list:
            hit |= S[:, k][:, None] > lv
        run = (run + 1) * hit
        on = run >= persist
        cnt += on
        run[on] = 0
    return cnt / n


def null_level(x_std, q=0.90, target=0.05, block=60, B=300, persist=PERSIST_DEFAULT, two_sided=False, seed=0, levels=LEVEL_GRID):
    """Smallest CUSUM level whose q-quantile (over block-bootstrap resamples of the training stream) of the onset share is <= target.
    Returns (level, feasible)."""
    rng = np.random.default_rng(seed)
    Z = mbb_resample(x_std, block, B, rng)
    Ss = [cusum_S(Z)] + ([cusum_S(-Z)] if two_sided else [])
    rates = onset_rate_levels(Ss, levels, persist)
    qr = np.quantile(rates, q, axis=0)
    ok = np.where(qr <= target)[0]
    if ok.size == 0:
        return float(levels[-1]), False
    return float(levels[ok[0]]), True


def ema_rows(X, lam=EMA_LAM):
    out = np.empty_like(X)
    s = X[:, 0].copy()
    out[:, 0] = s
    for k in range(1, X.shape[1]):
        s = lam * s + (1 - lam) * X[:, k]
        out[:, k] = s
    return out


def null_tau(a_stream, q=0.90, level=0.95, floor=0.05, block=60, B=300, seed=1):
    """Absolute threshold: q-quantile over resamples of the `level`-quantile of the smoothed alpha stream, floored."""
    rng = np.random.default_rng(seed)
    X = ema_rows(mbb_resample(a_stream, block, B, rng))
    return float(max(np.quantile(np.quantile(X, level, axis=1), q) + 1e-9, floor))
