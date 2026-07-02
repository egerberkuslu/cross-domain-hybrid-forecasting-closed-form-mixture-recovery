"""Reviewer-R2 / C2: Cross-dataset transfer with α=1 FORCED.

Original cross_dataset_transfer.csv showed transfer ratio ≈ 1.0, which is
an artefact: when validation-tuned α ≈ 0, the hybrid forecast equals the
target's Chronos-Bolt regardless of the source's LSTM weights, so the
transfer test does not exercise the learned residual head.

This script forces α=1.0 on every transfer, evaluating ONLY the
decomposition path (Theta-trend + SeasonalNaive + transferred LSTM
residual) on the target dataset. Now we are actually testing whether
the LSTM-residual generalises across operator domains.

Output:
  outputs/eval_v3/tables/cross_dataset_transfer_alpha1.csv
  outputs/figures/cross_dataset_alpha1_heatmap.pdf
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
HORIZONS = [1, 6, 24]  # span the horizon range — reviewer N1 follow-up
SEED = 42


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = next(b for r, v, b, s in MODEL_CONFIGS if v == "cha_hybrid_v3")

    target_data = {}
    for ds in DATASETS:
        try:
            pp = load_preprocessed(ds)
            wins = pp.windows[HORIZON]
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
        # Build a v3 on target so we have a valid runtime object
        m_target = build_model(
            "cha_hybrid_v3", horizon=HORIZON, hparams=base, seed=SEED, device=device
        )
        m_target.fit(
            td["wins"]["train"],
            td["wins"]["val"],
            train_series=td["ts"],
            val_series=td["vs"],
        )
        # In-domain α=1 RMSE (decomposition-path only on target)
        decomp_target = m_target._predict_decomposition(td["wins"]["test"].X)
        y_true = td["wins"]["test"].y
        n = min(decomp_target.shape[0], y_true.shape[0])
        rmse_diag = _rmse(y_true[:n], decomp_target[:n])

        for i, source in enumerate(DATASETS):
            ck_path = CKPT_DIR / f"{source}__cha_hybrid_v3__h{HORIZON}__s{SEED}.pt"
            if not ck_path.exists():
                print(f"  no ckpt for source={source}, target={target}")
                continue
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            lstm_sd = ck.get("lstm_residual_state_dict")
            print(f"[α=1] source={source} → target={target}")
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
                print(f"  fail {source}→{target}: {type(e).__name__}: {e}")
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "rmse_alpha1": float("nan"),
                        "in_domain_rmse_alpha1": rmse_diag,
                        "transfer_ratio": float("nan"),
                    }
                )
        # Restore source LSTM = target's own LSTM by re-fitting (lazy: keep last)
        del m_target
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TBL / "cross_dataset_transfer_alpha1.csv", index=False)
    print("wrote cross_dataset_transfer_alpha1.csv")

    off_diag = df[df["source"] != df["target"]].dropna(subset=["transfer_ratio"])
    if not off_diag.empty:
        rng = np.random.default_rng(42)
        boot = np.array(
            [
                rng.choice(
                    off_diag["transfer_ratio"], size=len(off_diag), replace=True
                ).mean()
                for _ in range(2000)
            ]
        )
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"  off-diagonal (n={len(off_diag)}):")
        print(f"    mean ratio: {off_diag['transfer_ratio'].mean():.3f}")
        print(f"    95% CI: [{lo:.3f}, {hi:.3f}]")
        print(f"    worst: {off_diag['transfer_ratio'].max():.3f}")

    # Heatmap — target-normalised so cells are comparable across
    # datasets with different raw RMSE scales (cell = source→target
    # RMSE divided by the *target's* own in-domain RMSE).
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    diag = np.diag(rmse_mat)
    norm = rmse_mat.copy()
    for j in range(len(DATASETS)):
        if np.isfinite(diag[j]) and diag[j] > 0:
            norm[:, j] = rmse_mat[:, j] / diag[j]
    im = ax.imshow(norm, cmap="RdYlGn_r", vmin=0.85, vmax=1.25)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, rotation=30, ha="right")
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_xlabel("target (evaluated on)")
    ax.set_ylabel("source LSTM-residual from")
    ax.set_title(
        f"Cross-dataset transfer with α=1 (decomp-only) at h={HORIZON}\n"
        "cell value = RMSE / target's own decomp-only RMSE"
    )
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
                    fontsize=9,
                )
    fig.colorbar(im, ax=ax, label="relative RMSE")
    fig.tight_layout()
    fig.savefig(OUT_FIG / "cross_dataset_alpha1_heatmap.pdf", bbox_inches="tight")
    fig.savefig(
        OUT_FIG / "cross_dataset_alpha1_heatmap.png", bbox_inches="tight", dpi=150
    )
    print(f"wrote cross_dataset_alpha1_heatmap.{{pdf,png}}")


if __name__ == "__main__":
    with PhaseTimer(
        "expR2_alpha1_transfer",
        notes="C2: α=1 forced cross-dataset transfer (exercise LSTM)",
    ) as t:
        main()
        t.add_output("csv", str(OUT_TBL / "cross_dataset_transfer_alpha1.csv"))
        t.add_output("pdf", str(OUT_FIG / "cross_dataset_alpha1_heatmap.pdf"))
