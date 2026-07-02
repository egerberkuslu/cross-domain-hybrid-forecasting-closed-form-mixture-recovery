"""Reviewer Round-3 N1: α=1 forced cross-dataset transfer at h ∈ {1, 6, 24}.

Generalises expR2_alpha1_transfer.py to multiple horizons so we can claim
horizon-robustness of the LSTM-residual transfer rather than a single-h
result.

Output:
  outputs/eval_v3/tables/cross_dataset_transfer_alpha1_multi_h.csv
  outputs/figures/cross_dataset_alpha1_multi_h.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.preprocessing import load_preprocessed
from src.models import build_model
from src.training import MODEL_CONFIGS
from src.utils.runlog import PhaseTimer

CKPT_DIR = Path("outputs/checkpoints")
OUT_TBL = Path("outputs/eval_v3/tables")
OUT_FIG = Path("outputs/figures")
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
HORIZONS = [1, 6, 24]
SEED = 42


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def run_horizon(h, device, base):
    target_data = {}
    for ds in DATASETS:
        try:
            pp = load_preprocessed(ds)
            wins = pp.windows[h]
            target_data[ds] = {
                "wins": wins,
                "ts": pp.split_scaled.train["value"].to_numpy("float32"),
                "vs": pp.split_scaled.val["value"].to_numpy("float32"),
            }
        except Exception as e:
            print(f"skip target {ds}: {e}")

    rmse_mat = np.full((len(DATASETS), len(DATASETS)), np.nan)
    rows = []
    for j, target in enumerate(DATASETS):
        td = target_data.get(target)
        if td is None:
            continue
        m_target = build_model(
            "cha_hybrid_v3", horizon=h, hparams=base, seed=SEED, device=device
        )
        m_target.fit(
            td["wins"]["train"],
            td["wins"]["val"],
            train_series=td["ts"],
            val_series=td["vs"],
        )
        decomp_target = m_target._predict_decomposition(td["wins"]["test"].X)
        y_true = td["wins"]["test"].y
        n = min(decomp_target.shape[0], y_true.shape[0])
        rmse_diag = _rmse(y_true[:n], decomp_target[:n])

        for i, source in enumerate(DATASETS):
            ck_path = CKPT_DIR / f"{source}__cha_hybrid_v3__h{h}__s{SEED}.pt"
            if not ck_path.exists():
                continue
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            lstm_sd = ck.get("lstm_residual_state_dict")
            try:
                if lstm_sd is not None and source != target:
                    m_target.residual_model._darts_model.model.load_state_dict(lstm_sd)
                    decomp = m_target._predict_decomposition(td["wins"]["test"].X)
                else:
                    decomp = decomp_target
                n = min(decomp.shape[0], y_true.shape[0])
                r = _rmse(y_true[:n], decomp[:n])
                rmse_mat[i, j] = r
                rows.append(
                    {
                        "horizon": h,
                        "source": source,
                        "target": target,
                        "rmse_alpha1": r,
                        "in_domain_rmse_alpha1": rmse_diag,
                        "transfer_ratio": r / rmse_diag
                        if rmse_diag > 0
                        else float("nan"),
                    }
                )
            except Exception as e:
                print(f"  fail h={h} {source}→{target}: {type(e).__name__}: {e}")
                rows.append(
                    {
                        "horizon": h,
                        "source": source,
                        "target": target,
                        "rmse_alpha1": float("nan"),
                        "in_domain_rmse_alpha1": rmse_diag,
                        "transfer_ratio": float("nan"),
                    }
                )
        del m_target
        if device == "cuda":
            torch.cuda.empty_cache()
    return rows, rmse_mat


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = next(b for r, v, b, s in MODEL_CONFIGS if v == "cha_hybrid_v3")

    all_rows = []
    mats = {}
    for h in HORIZONS:
        print(f"\n=== HORIZON h={h} ===")
        rows, mat = run_horizon(h, device, base)
        all_rows.extend(rows)
        mats[h] = mat

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_TBL / "cross_dataset_transfer_alpha1_multi_h.csv", index=False)
    print(
        f"\nwrote csv: {len(df)} rows ({len(HORIZONS)} horizons × {len(DATASETS)**2} pairs)"
    )

    # Summary
    for h in HORIZONS:
        sub = df[(df["horizon"] == h) & (df["source"] != df["target"])].dropna()
        if not sub.empty:
            rng = np.random.default_rng(42)
            boot = np.array(
                [
                    rng.choice(
                        sub["transfer_ratio"], size=len(sub), replace=True
                    ).mean()
                    for _ in range(2000)
                ]
            )
            lo, hi = np.percentile(boot, [2.5, 97.5])
            print(
                f"  h={h:>2d}  n={len(sub)}  mean={sub['transfer_ratio'].mean():.3f}  "
                f"95% CI [{lo:.3f}, {hi:.3f}]  worst={sub['transfer_ratio'].max():.3f}"
            )

    # 3-panel heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        mat = mats[h]
        diag = np.diag(mat)
        # Target-normalised: divide each cell by the target's (column's)
        # in-domain RMSE so cells are comparable across datasets and the
        # colour-bar maps cleanly onto "transfer degradation factor".
        norm = mat.copy()
        for j in range(len(DATASETS)):
            if np.isfinite(diag[j]) and diag[j] > 0:
                norm[:, j] = mat[:, j] / diag[j]
        im = ax.imshow(norm, cmap="RdYlGn_r", vmin=0.85, vmax=1.25)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS, rotation=30, ha="right")
        ax.set_yticks(range(len(DATASETS)))
        ax.set_yticklabels(DATASETS)
        ax.set_xlabel("target")
        ax.set_ylabel("source LSTM-residual from")
        ax.set_title(f"h = {h}")
        for i in range(len(DATASETS)):
            for j in range(len(DATASETS)):
                v = norm[i, j]
                if np.isfinite(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        color="black" if v < 1.5 else "white",
                        fontsize=8,
                    )
    fig.colorbar(im, ax=axes.ravel().tolist(), label="ratio to in-domain", shrink=0.7)
    fig.suptitle("α=1 forced cross-dataset transfer at three horizons", fontsize=12)
    fig.savefig(OUT_FIG / "cross_dataset_alpha1_multi_h.pdf", bbox_inches="tight")
    fig.savefig(
        OUT_FIG / "cross_dataset_alpha1_multi_h.png", bbox_inches="tight", dpi=150
    )
    print(f"wrote figure: cross_dataset_alpha1_multi_h.{{pdf,png}}")


if __name__ == "__main__":
    with PhaseTimer(
        "expR3_alpha1_transfer_multi_h",
        notes="N1 reviewer follow-up: α=1 transfer at h ∈ {1,6,24}",
    ) as t:
        main()
        t.add_output("csv", str(OUT_TBL / "cross_dataset_transfer_alpha1_multi_h.csv"))
        t.add_output("pdf", str(OUT_FIG / "cross_dataset_alpha1_multi_h.pdf"))
