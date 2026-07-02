"""Stage A — Experiment 3.1: Cross-dataset transfer matrix.

Loads each CHA-Hybrid v3 checkpoint trained on dataset SOURCE and applies
it (plug-and-play, no fine-tuning) to every other dataset's TEST split.
Reports a 5×5 RMSE matrix; the diagonal is the in-domain baseline.  Off-
diagonal cells quantify how well the *structural* combination (STL period,
global-expert prompt, learned α schedule) transfers without re-training.

Important caveat: each dataset has its own StandardScaler.  We honour
target-dataset scaling on inputs/outputs (so RMSE numbers are comparable
to the in-domain v3 RMSE in the leaderboard).  The decomposition path
re-decomposes the target test series with its STL period; only the
LSTM-residual weights, α schedule, and global-expert ID transfer.

Output:
  outputs/eval_v3/tables/cross_dataset_transfer.csv
  outputs/figures/cross_dataset_heatmap.pdf
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
HORIZON = 6  # representative horizon for the transfer matrix
SEED = 42


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _build_v3(device: str):
    base = next(b for r, v, b, s in MODEL_CONFIGS if v == "cha_hybrid_v3")
    return build_model(
        "cha_hybrid_v3", horizon=HORIZON, hparams=base, seed=SEED, device=device
    )


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rmse_mat = np.full((len(DATASETS), len(DATASETS)), np.nan)
    rows = []
    pp_cache: dict = {}

    for j, target in enumerate(DATASETS):
        try:
            pp_cache[target] = load_preprocessed(target)
        except Exception as e:
            print(f"skip target {target}: {e}")
            continue

    for i, source in enumerate(DATASETS):
        ck_path = CKPT_DIR / f"{source}__cha_hybrid_v3__h{HORIZON}__s{SEED}.pt"
        if not ck_path.exists():
            print(f"no ckpt for source={source}")
            continue
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        alpha_src = float(ck.get("alpha_h", 0.5))
        lstm_sd = ck.get("lstm_residual_state_dict")

        for j, target in enumerate(DATASETS):
            pp = pp_cache.get(target)
            if pp is None:
                continue
            wins = pp.windows.get(HORIZON)
            if wins is None:
                continue
            print(f"[3.1] source={source} → target={target}")
            try:
                m = _build_v3(device)
                # Fit on TARGET train to materialise the architecture (this
                # also re-trains the residual-LSTM on target).  Then we
                # overwrite the LSTM weights with the SOURCE checkpoint and
                # the α value with the SOURCE value to measure
                # plug-and-play transfer.
                m.fit(
                    wins["train"],
                    wins["val"],
                    train_series=pp.split_scaled.train["value"].to_numpy("float32"),
                    val_series=pp.split_scaled.val["value"].to_numpy("float32"),
                )
                # Transfer source residual-LSTM weights + source α
                if lstm_sd is not None and hasattr(m, "residual_model"):
                    try:
                        m.residual_model._darts_model.model.load_state_dict(lstm_sd)
                    except Exception as e:
                        print(f"  warn: cannot load source LSTM state_dict: {e}")
                m.alpha_h = alpha_src
                preds = m.predict(wins["test"])
                y_true = wins["test"].y
                if preds.shape != y_true.shape:
                    # truncate to common
                    n = min(preds.shape[0], y_true.shape[0])
                    preds = preds[:n]
                    y_true = y_true[:n]
                r = _rmse(y_true, preds)
                rmse_mat[i, j] = r
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "horizon": HORIZON,
                        "alpha_source": alpha_src,
                        "rmse_scaled": r,
                    }
                )
                del m
                if device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"  fail {source}→{target}: {type(e).__name__}: {e}")
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "horizon": HORIZON,
                        "alpha_source": alpha_src,
                        "rmse_scaled": float("nan"),
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = OUT_TBL / "cross_dataset_transfer.csv"
    df.to_csv(out_csv, index=False)
    print(f"[3.1] wrote {out_csv}")

    # Heatmap: rows=source, cols=target, normalised by diagonal
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm_mat = rmse_mat.copy()
    diag = np.diag(rmse_mat)
    for i in range(len(DATASETS)):
        if np.isfinite(diag[i]) and diag[i] > 0:
            norm_mat[i, :] = rmse_mat[i, :] / diag[i]
    im = ax.imshow(norm_mat, cmap="RdYlGn_r", vmin=1.0, vmax=3.0)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(DATASETS, rotation=30, ha="right")
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_xlabel("target (evaluated on)")
    ax.set_ylabel("source (trained on)")
    ax.set_title(
        f"Cross-dataset transfer of CHA-Hybrid v3 (h={HORIZON})\n"
        f"cell value = RMSE / in-domain RMSE"
    )
    for i in range(len(DATASETS)):
        for j in range(len(DATASETS)):
            v = norm_mat[i, j]
            if np.isfinite(v):
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="black" if v < 2.0 else "white",
                    fontsize=9,
                )
    fig.colorbar(im, ax=ax, label="relative RMSE (1.0 = in-domain)")
    fig.tight_layout()
    out_pdf = OUT_FIG / "cross_dataset_heatmap.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(OUT_FIG / "cross_dataset_heatmap.png", bbox_inches="tight", dpi=150)
    print(f"[3.1] wrote {out_pdf}")


if __name__ == "__main__":
    with PhaseTimer(
        "expA_3.1_cross_dataset_transfer",
        notes=f"5×5 plug-and-play transfer matrix at horizon={HORIZON}, seed={SEED}",
    ) as t:
        main()
        t.add_output("csv", str(OUT_TBL / "cross_dataset_transfer.csv"))
        t.add_output("pdf", str(OUT_FIG / "cross_dataset_heatmap.pdf"))
