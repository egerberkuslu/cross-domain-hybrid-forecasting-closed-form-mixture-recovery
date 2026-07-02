"""Stage A — Experiment 4.4: ADF stationarity check on every dataset.

The Diebold–Mariano test and most asymptotic forecast-evaluation results
assume covariance-stationary forecast errors.  Reviewers commonly ask
"have you tested stationarity?"; this script answers with ADF (Augmented
Dickey–Fuller) and KPSS on each (dataset, split) combination, reporting
test statistic, p-value, and critical values at 1/5/10 %.

Output:
  outputs/eval_v3/tables/stationarity.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.preprocessing import load_preprocessed
from src.utils.runlog import PhaseTimer

OUT_DIR = Path("outputs/eval_v3/tables")
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]


def _adf_row(name: str, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    if x.size < 30 or np.allclose(x.std(), 0):
        return {
            "split": name,
            "n": int(x.size),
            "adf_stat": float("nan"),
            "adf_pvalue": float("nan"),
            "kpss_stat": float("nan"),
            "kpss_pvalue": float("nan"),
            "is_stationary_adf5pct": False,
            "is_stationary_kpss5pct": False,
        }
    try:
        adf = adfuller(x, autolag="AIC")
        adf_stat, adf_p = float(adf[0]), float(adf[1])
    except Exception:
        adf_stat = adf_p = float("nan")
    try:
        kp_stat, kp_p, _, _ = kpss(x, regression="c", nlags="auto")
        kp_stat, kp_p = float(kp_stat), float(kp_p)
    except Exception:
        kp_stat = kp_p = float("nan")
    return {
        "split": name,
        "n": int(x.size),
        "adf_stat": adf_stat,
        "adf_pvalue": adf_p,
        "kpss_stat": kp_stat,
        "kpss_pvalue": kp_p,
        "is_stationary_adf5pct": (adf_p < 0.05) if np.isfinite(adf_p) else False,
        "is_stationary_kpss5pct": (kp_p > 0.05) if np.isfinite(kp_p) else False,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in DATASETS:
        print(f"[4.4] {ds}")
        try:
            pp = load_preprocessed(ds)
        except Exception as e:
            print(f"  load failed: {e}")
            continue
        for sp in ("train", "val", "test"):
            x = getattr(pp.split_scaled, sp)["value"].to_numpy("float64")
            r = _adf_row(sp, x)
            r["dataset"] = ds
            rows.append(r)
    df = pd.DataFrame(rows)
    out = OUT_DIR / "stationarity.csv"
    df.to_csv(out, index=False)
    print(f"[4.4] wrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    with PhaseTimer(
        "expA_4.4_adf_stationarity",
        notes="ADF + KPSS on every (dataset, split)",
    ) as t:
        main()
        t.add_output("table", str(OUT_DIR / "stationarity.csv"))
