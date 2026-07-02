"""TOST-style bootstrap equivalence test: CHA-S (v3) vs Chronos-Bolt.

Reviewer concern: the one-sided sign-flip permutation test can only detect
CHA-S being *better* and cannot establish equivalence; reporting its
non-rejections as 'statistically equivalent' is absence-of-evidence, not
evidence-of-absence. This script runs a proper equivalence test.

Method (bootstrap TOST on RMSE difference):
  For each (dataset, horizon): pool the per-window squared errors of CHA-S
  and Chronos-Bolt across seeds, bootstrap-resample windows (B=2000),
  compute Delta = RMSE_v3 - RMSE_bolt per resample, and form the 90% CI.
  Equivalence margin epsilon = 0.05 * RMSE_bolt (i.e. CHA-S is declared
  practically equivalent if its RMSE is within +/-5% of Bolt's). The cell
  is EQUIVALENT iff the entire 90% CI of Delta lies within [-eps, +eps]
  (TOST at alpha=0.05). Otherwise it is BETTER (CI below -eps),
  WORSE (CI above +eps), or INCONCLUSIVE (CI straddles a bound).

Output: outputs/eval_v3/tables/tost_equivalence.csv
"""

from __future__ import annotations

import glob
import os
import numpy as np
import pandas as pd

PRED = "outputs/predictions"
NET_DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
HORIZONS = [1, 3, 6, 12, 24]
SEEDS = [42, 123, 2024, 31337, 7]
B = 2000
EPS_FRAC = 0.05  # equivalence margin = 5% of Bolt RMSE
RNG = np.random.default_rng(12345)


def load_sq_errors(dataset, model, h, seed):
    f = f"{PRED}/{dataset}__{model}__h{h}__s{seed}.npz"
    if not os.path.exists(f):
        return None
    d = np.load(f)
    yt = d["y_true_scaled"]
    yp = d["y_pred_scaled"]
    # mean squared error per window (average over the horizon steps)
    return np.mean((yt - yp) ** 2, axis=1)


def main():
    rows = []
    for ds in NET_DATASETS:
        for h in HORIZONS:
            v3_parts, bolt_parts = [], []
            for s in SEEDS:
                e_v3 = load_sq_errors(ds, "cha_hybrid_v3", h, s)
                e_bolt = load_sq_errors(ds, "chronos_bolt_zs", h, s)
                if e_v3 is None or e_bolt is None:
                    continue
                n = min(len(e_v3), len(e_bolt))
                v3_parts.append(e_v3[:n])
                bolt_parts.append(e_bolt[:n])
            if not v3_parts:
                continue
            e_v3 = np.concatenate(v3_parts)
            e_bolt = np.concatenate(bolt_parts)
            n = len(e_v3)
            rmse_v3 = float(np.sqrt(e_v3.mean()))
            rmse_bolt = float(np.sqrt(e_bolt.mean()))
            eps = EPS_FRAC * rmse_bolt
            # paired bootstrap over windows
            deltas = np.empty(B)
            for b in range(B):
                idx = RNG.integers(0, n, n)
                deltas[b] = np.sqrt(e_v3[idx].mean()) - np.sqrt(e_bolt[idx].mean())
            lo, hi = np.percentile(deltas, [5, 95])  # 90% CI -> TOST alpha=0.05
            if hi < -eps:
                verdict = "BETTER"
            elif lo > eps:
                verdict = "WORSE"
            elif lo >= -eps and hi <= eps:
                verdict = "EQUIVALENT"
            else:
                verdict = "INCONCLUSIVE"
            rows.append(
                {
                    "dataset": ds,
                    "horizon": h,
                    "n_windows": n,
                    "rmse_v3": round(rmse_v3, 4),
                    "rmse_bolt": round(rmse_bolt, 4),
                    "delta": round(rmse_v3 - rmse_bolt, 5),
                    "ci90_lo": round(lo, 5),
                    "ci90_hi": round(hi, 5),
                    "eps": round(eps, 5),
                    "verdict": verdict,
                }
            )
            print(
                f"{ds:12s} h={h:2d} n={n:5d} rmse_v3={rmse_v3:.4f} "
                f"rmse_bolt={rmse_bolt:.4f} d={rmse_v3-rmse_bolt:+.5f} "
                f"CI90=[{lo:+.5f},{hi:+.5f}] eps={eps:.5f} -> {verdict}"
            )
    df = pd.DataFrame(rows)
    out = "outputs/eval_v3/tables/tost_equivalence.csv"
    df.to_csv(out, index=False)
    print("\n=== SUMMARY ===")
    print(df["verdict"].value_counts().to_dict())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
