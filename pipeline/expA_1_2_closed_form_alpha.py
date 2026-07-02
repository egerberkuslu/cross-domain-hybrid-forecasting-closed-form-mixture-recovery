"""Stage A — Experiment 1.2: Bates-Granger closed-form α vs. grid-searched α.

For each CHA-Hybrid v3 checkpoint, recover the per-expert validation MSEs
(σ²_dec, σ²_global) and the cross-covariance σ_dec,global from the saved
``alpha_search_diag`` (grid trace).  Then compute the Bates–Granger optimal
weight in closed form

    α_BG = (σ²_global − σ_dec,global) / (σ²_dec + σ²_global − 2 σ_dec,global)

and compare against the grid-searched α we actually used.  The script
reports:

  * Δα = |α_BG − α_grid|
  * Δ-MSE on validation when we *swap* α_grid → α_BG (positive = grid wins)
  * α_BG_corr ignoring covariance (inverse-variance), and α_BG_cov full form

This experiment is the "methodological novelty" piece — it converts a flat
grid search into a *principled* combination rule.  Output goes into
``outputs/eval_v3/tables/alpha_bg.csv``.  Pure post-processing, ~30 s on the
existing checkpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.utils.runlog import PhaseTimer

CKPT_DIR = Path("outputs/checkpoints")
OUT_DIR = Path("outputs/eval_v3/tables")


def _quadratic_decomp_from_grid(grid: dict) -> dict | None:
    """Recover σ²_dec, σ²_global, σ_dec,global from the alpha_search grid trace.

    The mixture MSE on validation as a function of α has the form
        M(α) = α² σ²_dec + (1-α)² σ²_global + 2 α (1-α) σ_{dec,global}
    so {M(1), M(0), M(0.5)} are sufficient to solve for the three unknowns.
    """
    if not grid:
        return None

    def _to_mse(g: dict) -> float:
        for k in ("mse", "val_mse"):
            if k in g and g[k] is not None:
                return float(g[k])
        for k in ("rmse", "val_rmse"):
            if k in g and g[k] is not None:
                return float(g[k]) ** 2
        return float("nan")

    # grid is a dict-of-arrays or list-of-dicts depending on serialisation.
    if isinstance(grid, dict) and "alpha" in grid:
        alphas = np.asarray(grid["alpha"], dtype=float)
        if "mse" in grid:
            mses = np.asarray(grid["mse"], dtype=float)
        elif "val_rmse" in grid:
            mses = np.asarray(grid["val_rmse"], dtype=float) ** 2
        elif "rmse" in grid:
            mses = np.asarray(grid["rmse"], dtype=float) ** 2
        else:
            return None
    elif isinstance(grid, list) and grid and isinstance(grid[0], dict):
        alphas = np.asarray([g["alpha"] for g in grid], dtype=float)
        mses = np.asarray([_to_mse(g) for g in grid], dtype=float)
    else:
        return None
    if not np.all(np.isfinite(mses)):
        return None

    def at(target):
        i = int(np.argmin(np.abs(alphas - target)))
        return float(mses[i]), float(alphas[i])

    m1, _ = at(1.0)  # all weight on decomposition expert
    m0, _ = at(0.0)  # all weight on global expert
    mh, _ = at(0.5)  # equal weight
    sigma2_dec = m1
    sigma2_global = m0
    cov = 2.0 * mh - 0.5 * sigma2_dec - 0.5 * sigma2_global
    return {
        "sigma2_dec": sigma2_dec,
        "sigma2_global": sigma2_global,
        "cov_dec_global": cov,
        "mse_curve_argmin_alpha": float(alphas[int(np.argmin(mses))]),
        "mse_curve_argmin_value": float(np.min(mses)),
    }


def closed_form_alpha(sigma2_dec: float, sigma2_global: float, cov: float) -> dict:
    denom = sigma2_dec + sigma2_global - 2.0 * cov
    if denom <= 0:
        alpha_cov = float("nan")
    else:
        alpha_cov = float((sigma2_global - cov) / denom)
    alpha_iv = float(sigma2_global / (sigma2_dec + sigma2_global))

    # MSE of the closed-form mix under the same quadratic model:
    def mse_at(a):
        return a * a * sigma2_dec + (1 - a) ** 2 * sigma2_global + 2 * a * (1 - a) * cov

    return {
        "alpha_BG_cov": alpha_cov,
        "alpha_BG_iv": alpha_iv,
        "mse_BG_cov": mse_at(alpha_cov) if np.isfinite(alpha_cov) else float("nan"),
        "mse_BG_iv": mse_at(alpha_iv),
    }


def parse_ckpt_name(path: Path) -> dict:
    p = path.stem  # e.g. cesnet__cha_hybrid_v3__h6__s42
    parts = p.split("__")
    if len(parts) != 4:
        return {}
    return {
        "dataset": parts[0],
        "variant": parts[1],
        "horizon": int(parts[2].lstrip("h")),
        "seed": int(parts[3].lstrip("s")),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CKPT_DIR.glob("*cha_hybrid*.pt"))
    print(f"[1.2] {len(files)} checkpoints found")
    rows = []
    for p in files:
        meta = parse_ckpt_name(p)
        if not meta:
            continue
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  skip {p.name}: load failed ({e})")
            continue
        diag = ck.get("alpha_search_diag")
        sigma = _quadratic_decomp_from_grid(diag)
        if sigma is None:
            continue
        cf = closed_form_alpha(
            sigma["sigma2_dec"], sigma["sigma2_global"], sigma["cov_dec_global"]
        )
        a_grid = float(ck.get("alpha_h", float("nan")))
        # MSE of grid α evaluated under the quadratic model:
        m_grid = (
            a_grid * a_grid * sigma["sigma2_dec"]
            + (1 - a_grid) ** 2 * sigma["sigma2_global"]
            + 2 * a_grid * (1 - a_grid) * sigma["cov_dec_global"]
        )
        rows.append(
            {
                **meta,
                "alpha_grid": a_grid,
                "alpha_BG_cov": cf["alpha_BG_cov"],
                "alpha_BG_iv": cf["alpha_BG_iv"],
                "delta_alpha_cov": abs(a_grid - cf["alpha_BG_cov"])
                if np.isfinite(cf["alpha_BG_cov"])
                else float("nan"),
                "delta_alpha_iv": abs(a_grid - cf["alpha_BG_iv"]),
                "mse_grid": float(m_grid),
                "mse_BG_cov": cf["mse_BG_cov"],
                "mse_BG_iv": cf["mse_BG_iv"],
                **sigma,
            }
        )
    df = pd.DataFrame(rows)
    out = OUT_DIR / "alpha_bg.csv"
    df.to_csv(out, index=False)
    print(f"[1.2] wrote {out}  ({len(df)} rows)")
    if not df.empty:
        print(
            "summary:",
            f"mean Δα_cov={df['delta_alpha_cov'].mean():.4f}",
            f"mean Δα_iv ={df['delta_alpha_iv'].mean():.4f}",
            f"mean ΔMSE(grid−BG_cov)={(df['mse_grid']-df['mse_BG_cov']).mean():.6f}",
        )


if __name__ == "__main__":
    with PhaseTimer(
        "expA_1.2_bates_granger_alpha",
        notes="closed-form α (full BG and inverse-variance) vs. grid α",
    ) as t:
        main()
        t.add_output("table", str(OUT_DIR / "alpha_bg.csv"))
