"""Reviewer-R2 / C4: Permutation test for per-window v3-vs-Chronos-Bolt wins.

R2's concern: "18.5% wins" is per-window cherry-picking unless we test
against a null where both models are equivalent.

Permutation test setup (paired per-window):
  H_0: v3 and Chronos-Bolt have the same per-window squared-error
       distribution (their predictions are exchangeable conditional on y).
  Test statistic: T = fraction of windows where v3 squared-error is at
       least 5% below Chronos-Bolt's squared-error.
  Null distribution: for each permutation, randomly swap v3↔Bolt label
       on each window with probability 0.5 (sign-flip permutation).
  P-value: P(T_null >= T_observed | H_0) over 2000 permutations.

Output:
  outputs/eval_v3/tables/per_window_permutation.csv
  outputs/eval_v3/tables/per_window_permutation_summary.txt
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.runlog import PhaseTimer

PRED_DIR = Path("outputs/predictions")
OUT_TBL = Path("outputs/eval_v3/tables")
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
HORIZONS = [1, 3, 6, 12, 24]
SEEDS = [42, 123, 2024, 7, 31337]
TIE_BAND = 0.05
N_PERMS = 2000
RNG_SEED = 1234567


def _se(y, p):
    return np.mean((y - p) ** 2, axis=1)


def load_pair(ds, h, seed):
    p_v3 = PRED_DIR / f"{ds}__cha_hybrid_v3__h{h}__s{seed}.npz"
    p_b = PRED_DIR / f"{ds}__chronos_bolt_zs__h{h}__s42.npz"
    if not (p_v3.exists() and p_b.exists()):
        return None
    a = np.load(p_v3)
    b = np.load(p_b)
    return a["y_true_scaled"], a["y_pred_scaled"], b["y_pred_scaled"]


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    cell_rows = []
    all_obs_wins = []
    all_null_wins_first = []

    for ds in DATASETS:
        for h in HORIZONS:
            obs_se_v3_all = []
            obs_se_b_all = []
            for seed in SEEDS:
                data = load_pair(ds, h, seed)
                if data is None:
                    continue
                yt, yv3, yb = data
                n = min(yt.shape[0], yv3.shape[0], yb.shape[0])
                se_v3 = _se(yt[:n], yv3[:n])
                se_b = _se(yt[:n], yb[:n])
                obs_se_v3_all.append(se_v3)
                obs_se_b_all.append(se_b)
            if not obs_se_v3_all:
                continue
            obs_se_v3_all = np.concatenate(obs_se_v3_all)
            obs_se_b_all = np.concatenate(obs_se_b_all)
            n = len(obs_se_v3_all)
            rel = (obs_se_b_all - obs_se_v3_all) / np.maximum(obs_se_b_all, 1e-12)
            obs_win = float((rel > TIE_BAND).mean())

            # Permutation under H_0 (exchangeable per window via sign-flip)
            null_wins = np.empty(N_PERMS)
            stack = np.stack([obs_se_v3_all, obs_se_b_all], axis=1)
            for k in range(N_PERMS):
                flip = rng.random(n) > 0.5
                perm_v3 = np.where(flip, obs_se_b_all, obs_se_v3_all)
                perm_b = np.where(flip, obs_se_v3_all, obs_se_b_all)
                rel_perm = (perm_b - perm_v3) / np.maximum(perm_b, 1e-12)
                null_wins[k] = (rel_perm > TIE_BAND).mean()
            pval = float((null_wins >= obs_win).mean())
            cell_rows.append(
                {
                    "dataset": ds,
                    "horizon": h,
                    "n_windows": int(n),
                    "observed_win_rate": obs_win,
                    "null_mean_win_rate": float(null_wins.mean()),
                    "null_p975": float(np.percentile(null_wins, 97.5)),
                    "p_value": pval,
                    "significant_at_005": bool(pval < 0.05),
                }
            )
            print(
                f"  {ds:<14} h={h:>2d}  obs={obs_win:.3f}  null_mean={null_wins.mean():.3f}  p={pval:.4f}"
            )

    df = pd.DataFrame(cell_rows)
    df.to_csv(OUT_TBL / "per_window_permutation.csv", index=False)

    n_sig = int(df["significant_at_005"].sum())
    n_total = len(df)
    lines = [
        f"Per-window v3-vs-Chronos-Bolt permutation test",
        f"=" * 60,
        f"H_0: v3 and Chronos-Bolt have exchangeable per-window squared errors",
        f"Test statistic: fraction of windows where v3 strictly wins by ≥{TIE_BAND*100:.0f}%",
        f"Permutations: {N_PERMS} per cell, sign-flip exchange under H_0",
        f"",
        f"Cells with significant difference at α=0.05: {n_sig}/{n_total}",
        f"Mean observed win rate: {df['observed_win_rate'].mean():.3f}",
        f"Mean null win rate:     {df['null_mean_win_rate'].mean():.3f}",
        f"Interpretation: "
        + (
            f"v3 wins more windows than chance in {n_sig}/{n_total} cells "
            f"({100*n_sig/n_total:.0f}%)"
        ),
    ]
    (OUT_TBL / "per_window_permutation_summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    with PhaseTimer(
        "expR2_permutation_test",
        notes="C4: per-window permutation test for v3-vs-Bolt wins",
    ) as t:
        main()
        t.add_output("csv", str(OUT_TBL / "per_window_permutation.csv"))
        t.add_output("txt", str(OUT_TBL / "per_window_permutation_summary.txt"))
