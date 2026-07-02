"""Per-(dataset, horizon) winner table — "which model wins where"."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def winners_table(df_runs: pd.DataFrame, metric: str = "rmse_native") -> pd.DataFrame:
    """For each (dataset, horizon) row, return the variant with the best (lowest)
    metric mean across seeds. For R^2 'higher is better' is auto-detected."""
    higher_is_better = metric.startswith("r2")
    rows = []
    for (ds, h), sub in df_runs.groupby(["dataset", "horizon"]):
        per_variant = sub.groupby("variant")[metric].mean()
        if higher_is_better:
            winner = per_variant.idxmax()
            val = per_variant.max()
        else:
            winner = per_variant.idxmin()
            val = per_variant.min()
        rows.append(
            {
                "dataset": ds,
                "horizon": int(h),
                "winner": winner,
                f"{metric}_winner": float(val),
                "n_variants": int(per_variant.size),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "horizon"]).reset_index(drop=True)


def proposed_rank(
    df_runs: pd.DataFrame, proposed: str = "cha_hybrid", metric: str = "rmse_native"
) -> pd.DataFrame:
    """Where does the proposed model rank in each (dataset, horizon) cell?"""
    higher_is_better = metric.startswith("r2")
    rows = []
    for (ds, h), sub in df_runs.groupby(["dataset", "horizon"]):
        per_variant = (
            sub.groupby("variant")[metric]
            .mean()
            .sort_values(ascending=not higher_is_better)
        )
        ranks = {v: r + 1 for r, v in enumerate(per_variant.index)}
        rows.append(
            {
                "dataset": ds,
                "horizon": int(h),
                "rank": ranks.get(proposed),
                "n": len(per_variant),
                "value": float(per_variant.get(proposed, float("nan"))),
                "winner": per_variant.index[0],
                "winner_value": float(per_variant.iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "horizon"]).reset_index(drop=True)


def write_winners(
    df_runs: pd.DataFrame, out_dir: str | Path, proposed: str = "cha_hybrid"
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for metric in ["rmse_native", "mae_native", "smape_native"]:
        p = out_dir / f"winners_{metric}.csv"
        winners_table(df_runs, metric).to_csv(p, index=False)
        written[metric] = p
    p = out_dir / f"proposed_rank_{proposed}.csv"
    proposed_rank(df_runs, proposed).to_csv(p, index=False)
    written[f"rank_{proposed}"] = p
    return written
