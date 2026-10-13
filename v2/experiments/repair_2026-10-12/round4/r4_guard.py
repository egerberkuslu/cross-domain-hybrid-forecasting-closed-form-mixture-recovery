"""Round 4: weight-guard selection on ROUND-3 VALIDATION arrays only (round3/r3_val/*.npz: validation windows of the 100 round-3 fresh cells; no test window, no test forecast).

Disclosure: the round 3 TEST result (pooled gain 0.499 %, one Abilene series with a validation weight of 0.6 to 0.7 that lost 4.9 to 10.6 % on test) motivated the search for a guard.
The candidate guards were listed in REPAIR_PREREG_4.md before this script computed anything, and the choice below uses validation data only.

Nested split of the n validation windows of a cell: A = [0, 0.4 n), purge h, B = [.., 0.7 n), purge h, C = [.., n). alpha_A = Bates-Granger weight on A (snapped to 0.05).
  G0       alpha_A                                       (the round 3 rule)
  G1(tau)  alpha_A if the gain of alpha_A on B over Bolt is >= tau, else 0      tau = 0.5 %
  G2(l)    l * alpha_A                                   l in {0.25, 0.5, 0.75}
  G12(l)   G1 first, then l * alpha_A                    l in {0.25, 0.5, 0.75}
Score: mean over the 100 cells of log(RMSE_C(variant) / RMSE_C(Bolt)); the variant with the lowest score is chosen, ties within 0.0002 go to the simpler one
(order G0, G1, G2, G12). Writes guard_selection.csv and frozen_r4_guard.json (+ sha256).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R3 = HERE.parent / "round3"
sys.path.insert(0, str(HERE.parent / "paper_a"))
import rcommon as R  # noqa: E402

TAU = 0.005
LAMS = (0.25, 0.5, 0.75)


def alpha_bg(y, B, E):
    d = E - B
    return R.snap(R.alpha_bg(np.sum((B - y) * d, 1), np.sum(d * d, 1)))


def rm(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def split_nested(n, h):
    a_end = int(0.4 * n)
    b0, b_end = a_end + h, int(0.7 * n)
    return np.arange(a_end), np.arange(b0, b_end), np.arange(b_end + h, n)


def guard_alpha(y, B, E, h, a_idx, b_idx):
    a = alpha_bg(y[a_idx], B[a_idx], E[a_idx])
    gain_b = 1 - rm(y[b_idx], B[b_idx] + a * (E[b_idx] - B[b_idx])) / rm(y[b_idx], B[b_idx])
    return a, gain_b


def variants(a, gain_b):
    v = {"G0": a, "G1": a if gain_b >= TAU else 0.0}
    for l in LAMS:
        v[f"G2_{l}"] = l * a
        v[f"G12_{l}"] = l * (a if gain_b >= TAU else 0.0)
    return v


def main():
    log = R.install_guard()
    m = json.loads((R3.parent / "fresh_cells" / "pool_manifest.json").read_text())
    rows = []
    for s in m["series"]:
        for h in (1, 3, 6, 12, 24):
            z = np.load(R3 / "r3_val" / f"{s['name']}__h{h}.npz")
            y, B, E = z["y"].astype(float), z["bolt"].astype(float), z["eq"].astype(float)
            ai, bi, ci = split_nested(len(y), h)
            a, gb = guard_alpha(y, B, E, h, ai, bi)
            rc = rm(y[ci], B[ci])
            for name, al in variants(a, gb).items():
                rows.append({"series": s["name"], "source": s["source"], "horizon": h, "variant": name, "alpha": al, "gain_B": gb, "log_ratio_C": float(np.log(rm(y[ci], B[ci] + al * (E[ci] - B[ci])) / rc))})
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "guard_selection.csv", index=False)
    score = df.groupby("variant").log_ratio_C.mean()
    order = ["G0", "G1", "G2_0.25", "G2_0.5", "G2_0.75", "G12_0.25", "G12_0.5", "G12_0.75"]
    best = min(order, key=lambda k: (round(score[k] / 0.0002), order.index(k)))
    # ties: any variant within 0.0002 of the minimum goes to the simplest
    lo = score.min()
    best = next(k for k in order if score[k] <= lo + 0.0002)
    per_src = df[df.variant == best].groupby("source").log_ratio_C.mean().to_dict()
    fb = float((df[df.variant == "G1"].alpha == 0).mean())
    out = {"split": "A 0-40 %, B 40-70 %, C 70-100 % of validation windows, purge h", "tau": TAU, "lams": list(LAMS), "pooled_score_mean_log_ratio_C": score.to_dict(), "chosen": best, "chosen_per_source": per_src,
           "share_cells_G1_fallback": fb, "round3_validation_cells": 100, "deployment": "alpha_full = Bates-Granger on all validation windows; guard decision from alpha on the first 70 % evaluated on the last 30 % (purge h); chosen variant applied to alpha_full"}
    p = HERE / "frozen_r4_guard.json"
    p.write_text(json.dumps(out, indent=1))
    (HERE / "frozen_r4_guard.sha256").write_text(R.sha256_file(p) + "\n")
    assert not any("predictions" in x for x in log)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
