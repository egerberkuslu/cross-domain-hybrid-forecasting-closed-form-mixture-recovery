"""audit item: stacked OLS fitted on validation (with intercept), scored on test.

pipeline/expC_5c_other_mixtures.py fitted y = b1 f_dec + b2 f_glob + b0 on the
TEST predictions and then used only alpha = b1 / (b1 + b2) as a convex weight,
dropping the intercept and the scale b1 + b2. That column is neither a
test-fit OLS (the "upper bound" reading) nor an out-of-sample stacker, so the
"ceiling" inference drawn from it is an artefact.

Here, per (dataset, horizon, seed), on the flattened (window x step) arrays:
  * ols_val_oos      : full OLS with intercept fitted on VALIDATION, applied
                       to TEST (the honest stacker). Needs step 0.
  * convex_val_oos   : the same fit's convex projection b1/(b1+b2), on TEST.
  * ols_test_insample: full OLS with intercept fitted on TEST (true upper
                       bound of linear stacking; label-leaking by design).
  * legacy           : expC_5c's alpha = b1/(b1+b2) from the test fit
                       (reproduces other_mixtures.csv rmse_stacked_lin).
  * fallback (only if step 0 is missing for the cell): fit on the first half
    of the test windows, score on the second half; CHA-S grid and closed form
    are rescored on the same second half. Rows are labelled fit_split.
CHA-S grid and closed-form (clipped, from the validation grid) are scored on
the same test windows for the comparison.

Outputs: stacked_ols_runs.csv, stacked_ols_cells.csv, stacked_ols_summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CHAS,
    DATASETS,
    DEC,
    EVAL,
    HORIZONS,
    SEEDS,
    align,
    bg_from_three,
    ckpt_json,
    load_bolt,
    load_pred,
    load_val_forecast,
    rmse,
    three_points,
    val_grid,
    write_csv,
    write_json,
)


def ols(y, fd, fg):
    X = np.column_stack([fd.ravel(), fg.ravel(), np.ones(fd.size)])
    coef, *_ = np.linalg.lstsq(X, y.ravel(), rcond=None)
    return coef  # b1, b2, b0


def apply(coef, fd, fg):
    b1, b2, b0 = coef
    return b1 * fd + b2 * fg + b0


def convex(coef):
    b1, b2, _ = coef
    s = b1 + b2
    return 0.5 if abs(s) < 1e-12 else float(b1 / s)


def main():
    rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            b = load_bolt(ds, h)
            if b is None:
                continue
            for s in SEEDS:
                d = load_pred(ds, DEC, h, s)
                ck = ckpt_json(ds, CHAS, h, s)
                if d is None or ck is None:
                    continue
                y, fd, fg = align(d[0], d[1], b[1])
                a_grid = float(ck["alpha_h"])
                a_cf = bg_from_three(*three_points(*val_grid(ck)))[1]
                vf = load_val_forecast(ds, h, s)
                coef_test = ols(y, fd, fg)
                base = {
                    "dataset": ds,
                    "horizon": h,
                    "seed": s,
                    "alpha_grid": a_grid,
                    "alpha_cf": a_cf,
                }
                if vf is not None:
                    yv, dv, gv = align(vf["y"], vf["dec"], vf["glob"])
                    cv = ols(yv, dv, gv)
                    a_cv = convex(cv)
                    rows.append(
                        {
                            **base,
                            "fit_split": "validation",
                            "b1": cv[0],
                            "b2": cv[1],
                            "b0": cv[2],
                            "rmse_ols_oos": rmse(y, apply(cv, fd, fg)),
                            "rmse_convex_oos": rmse(y, a_cv * fd + (1 - a_cv) * fg),
                            "rmse_ols_test_insample": rmse(y, apply(coef_test, fd, fg)),
                            "rmse_legacy_stacked": rmse(
                                y, convex(coef_test) * fd + (1 - convex(coef_test)) * fg
                            ),
                            "rmse_chas_grid": rmse(y, a_grid * fd + (1 - a_grid) * fg),
                            "rmse_chas_cf": rmse(y, a_cf * fd + (1 - a_cf) * fg),
                            "rmse_decomp": rmse(y, fd),
                            "rmse_bolt": rmse(y, fg),
                        }
                    )
                else:
                    k = y.shape[0] // 2
                    c1 = ols(y[:k], fd[:k], fg[:k])
                    a1 = convex(c1)
                    y2, d2, g2 = y[k:], fd[k:], fg[k:]
                    rows.append(
                        {
                            **base,
                            "fit_split": "test_first_half_FALLBACK",
                            "b1": c1[0],
                            "b2": c1[1],
                            "b0": c1[2],
                            "rmse_ols_oos": rmse(y2, apply(c1, d2, g2)),
                            "rmse_convex_oos": rmse(y2, a1 * d2 + (1 - a1) * g2),
                            "rmse_ols_test_insample": rmse(y, apply(coef_test, fd, fg)),
                            "rmse_legacy_stacked": rmse(
                                y, convex(coef_test) * fd + (1 - convex(coef_test)) * fg
                            ),
                            "rmse_chas_grid": rmse(y2, a_grid * d2 + (1 - a_grid) * g2),
                            "rmse_chas_cf": rmse(y2, a_cf * d2 + (1 - a_cf) * g2),
                            "rmse_decomp": rmse(y2, d2),
                            "rmse_bolt": rmse(y2, g2),
                        }
                    )
    runs = pd.DataFrame(rows)
    write_csv(runs, "stacked_ols_runs.csv")
    summary = {"fit_split_counts": runs.fit_split.value_counts().to_dict()}

    ref_p = EVAL / "tables" / "other_mixtures.csv"
    if ref_p.exists():
        ref = pd.read_csv(ref_p)[["dataset", "horizon", "seed", "rmse_stacked_lin"]]
        m = runs.merge(ref, on=["dataset", "horizon", "seed"], how="left")
        summary["legacy_stacked_reproduced_max_abs_diff"] = float(
            (m.rmse_legacy_stacked - m.rmse_stacked_lin).abs().max()
        )

    cells = (
        runs.groupby(["dataset", "horizon", "fit_split"])
        .mean(numeric_only=True)
        .reset_index()
    )
    cells["ols_oos_beats_chas_grid"] = cells.rmse_ols_oos < cells.rmse_chas_grid
    cells["ols_oos_vs_grid_pct"] = 100 * (cells.rmse_ols_oos / cells.rmse_chas_grid - 1)
    cells["insample_vs_grid_pct"] = 100 * (
        cells.rmse_ols_test_insample / cells.rmse_chas_grid - 1
    )
    write_csv(cells, "stacked_ols_cells.csv")
    summary["cells"] = int(len(cells))
    summary["ols_oos_beats_chas_grid"] = int(cells.ols_oos_beats_chas_grid.sum())
    summary["ols_oos_vs_grid_pct_range"] = [
        float(cells.ols_oos_vs_grid_pct.min()),
        float(cells.ols_oos_vs_grid_pct.max()),
    ]
    summary["test_insample_ols_vs_grid_pct_range"] = [
        float(cells.insample_vs_grid_pct.min()),
        float(cells.insample_vs_grid_pct.max()),
    ]
    nab = cells[(cells.dataset == "nab_aws_cpu") & (cells.horizon == 24)]
    if not nab.empty:
        summary["nab_aws_cpu_h24"] = nab.iloc[0][
            [
                "rmse_ols_oos",
                "rmse_ols_test_insample",
                "rmse_legacy_stacked",
                "rmse_chas_grid",
            ]
        ].to_dict()
    write_json(summary, "stacked_ols_summary.json")


if __name__ == "__main__":
    main()
