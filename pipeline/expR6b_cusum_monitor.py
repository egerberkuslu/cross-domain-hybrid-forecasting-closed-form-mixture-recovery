"""Robust alpha-monitor via CUSUM change detection on the alpha stream.

The absolute-threshold monitor false-alarms on datasets whose quiescent
alpha is naturally high (e.g. NAB-CPU). The fix is to detect a *change*
in alpha relative to its own running baseline rather than an absolute
level. We use a one-sided CUSUM on the standardised alpha stream:

  baseline mu, sigma estimated on a warm-up window;
  S_t = max(0, S_{t-1} + (z_t - k)),  z_t = (alpha_t - mu)/sigma;
  fire when S_t > H.

This is scale-free per dataset, so a high but stationary quiescent alpha
does not trigger; only a sustained upward shift does. We test on ALL FIVE
network datasets and report false-alarm rate (pre-changepoint) and
detection lag, to verify robustness before any paper claim.
"""

from __future__ import annotations

import numpy as np

PRED = "outputs/predictions"
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
H = 6
W = 60  # sliding window for alpha recovery
WARMUP = 40  # windows used to estimate the baseline (mu, sigma)
K = 0.5  # CUSUM slack (in sigma units)
H_CUSUM = 5.0  # CUSUM alarm threshold


def load(ds, model):
    d = np.load(f"{PRED}/{ds}__{model}__h{H}__s42.npz")
    return d["y_true_scaled"].mean(axis=1), d["y_pred_scaled"].mean(axis=1)


def recover_alpha(y, fd, ff):
    ed, ef = fd - y, ff - y
    diff = ed - ef
    den = float(np.dot(diff, diff))
    return 0.0 if den < 1e-12 else float(np.clip(-np.dot(ef, diff) / den, 0, 1))


def run(ds, seed=20260529):
    rng = np.random.default_rng(seed)
    y, fd = load(ds, "cha_hybrid_v3_decomp_only")
    _, ff = load(ds, "cha_hybrid_v3_global_only")
    n = min(len(y), len(fd), len(ff))
    y, fd, ff = y[:n], fd[:n], ff[:n]
    cp = max(W + WARMUP + 20, n // 2)
    sigma = float(np.std(ff - y)) + 1e-9
    ff_s = ff.copy()
    for t in range(cp, n):
        ramp = min(1.0, (t - cp) / 100.0)
        ff_s[t] = ff[t] + ramp * 3.0 * sigma * (1.0 + 0.5 * rng.standard_normal())
    a, idx = [], []
    for t in range(W, n):
        sl = slice(t - W, t)
        a.append(recover_alpha(y[sl], fd[sl], ff_s[sl]))
        idx.append(t)
    a = np.array(a)
    idx = np.array(idx)
    # baseline from the first WARMUP recovered values (all pre-changepoint)
    mu, sd = a[:WARMUP].mean(), a[:WARMUP].std() + 1e-6
    S = 0.0
    fired_at = None
    fa = 0
    pre_count = 0
    for k in range(WARMUP, len(a)):
        z = (a[k] - mu) / sd
        S = max(0.0, S + (z - K))
        if idx[k] < cp:
            pre_count += 1
            if S > H_CUSUM:
                fa += 1
                S = 0.0  # reset after a (false) alarm, as a real monitor would
        else:
            if S > H_CUSUM and fired_at is None:
                fired_at = int(idx[k] - cp)
    fa_rate = fa / max(1, pre_count)
    return {
        "dataset": ds,
        "n": n,
        "cp": cp,
        "baseline_alpha": round(float(mu), 3),
        "false_alarm_rate": round(fa_rate, 4),
        "detection_lag": fired_at,
    }


if __name__ == "__main__":
    print(f"{'dataset':14s} {'baseline-a':>10s} {'FA-rate':>8s} {'det-lag':>8s}")
    for ds in DATASETS:
        r = run(ds)
        lag = r["detection_lag"]
        print(
            f"{ds:14s} {r['baseline_alpha']:>10.3f} {r['false_alarm_rate']:>8.3f} "
            f"{(str(lag) if lag is not None else 'MISS'):>8s}"
        )
