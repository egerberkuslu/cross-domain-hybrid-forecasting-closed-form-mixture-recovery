"""Stage C — Defense 6c: Cross-cell α(x) transfer.

To prove that v4's MLP learns a *policy*, not just a curve fit to one
val set, we transfer the trained α(x) from a SOURCE cell
(dataset, horizon, seed) to every OTHER target cell --- without
retraining the MLP --- and report the resulting test RMSE.  If the
transferred MLP still outperforms a no-alpha baseline (α≡0.5) on
foreign cells, we have evidence that α(x) generalises.

Output:
  outputs/eval_v3/tables/cross_cell_alpha_transfer.csv
  outputs/figures/cross_cell_alpha_heatmap.pdf
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
from src.models.cha_hybrid_v4 import _context_features, _AlphaMLP
from src.utils.runlog import PhaseTimer

CKPT_DIR = Path("outputs/checkpoints")
OUT_TBL = Path("outputs/eval_v3/tables")
OUT_FIG = Path("outputs/figures")
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
HORIZON = 6
SEED = 42


def _rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def load_v4_mlp(ds: str, hidden: int = 16):
    p = CKPT_DIR / f"{ds}__cha_hybrid_v4__h{HORIZON}__s{SEED}.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    mlp_state = ck.get("alpha_mlp_state_dict")
    if mlp_state is None:
        return None
    in_dim = mlp_state["net.0.weight"].shape[1]
    h = mlp_state["net.0.weight"].shape[0]
    mlp = _AlphaMLP(in_dim, h)
    mlp.load_state_dict(mlp_state)
    mlp.eval()
    return mlp


def build_target_v3(target: str, device: str):
    """Build and fit a v3 on the target dataset (so we have its two experts)."""
    base = next(b for r, v, b, s in MODEL_CONFIGS if v == "cha_hybrid_v3")
    m = build_model(
        "cha_hybrid_v3", horizon=HORIZON, hparams=base, seed=SEED, device=device
    )
    pp = load_preprocessed(target)
    wins = pp.windows[HORIZON]
    ts = pp.split_scaled.train["value"].to_numpy("float32")
    vs = pp.split_scaled.val["value"].to_numpy("float32")
    m.fit(wins["train"], wins["val"], train_series=ts, val_series=vs)
    return m, wins


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pre-fit v3 once per target (gives us decomp and global predictions)
    target_data = {}
    for ds in DATASETS:
        try:
            m, wins = build_target_v3(ds, device)
            decomp = m._predict_decomposition(wins["test"].X)
            globalp = m.global_model.predict(wins["test"])
            target_data[ds] = {
                "X_test": wins["test"].X,
                "y_test": wins["test"].y,
                "decomp": decomp,
                "global": globalp,
                "alpha_v3": float(m.alpha_h),
                "stl_period": int(m.stl_period),
            }
            del m
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"  skip target={ds}: {e}")

    # For each (source, target), apply source v4 MLP to target features
    rmse_mat = np.full((len(DATASETS), len(DATASETS)), np.nan)
    rows = []
    for i, source in enumerate(DATASETS):
        mlp = load_v4_mlp(source)
        if mlp is None:
            print(f"  no v4 MLP for source={source}")
            continue
        mlp = mlp.to(device)
        for j, target in enumerate(DATASETS):
            td = target_data.get(target)
            if td is None:
                continue
            feat = _context_features(td["X_test"], stl_period=td["stl_period"])
            feat_t = torch.tensor(feat, dtype=torch.float32, device=device)
            with torch.no_grad():
                a = mlp(feat_t).cpu().numpy().reshape(-1, 1)
            mix = a * td["decomp"] + (1 - a) * td["global"]
            r = _rmse(td["y_test"], mix)
            rmse_mat[i, j] = r
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "rmse_transferred": r,
                    "alpha_mean": float(a.mean()),
                    "alpha_std": float(a.std()),
                    "alpha_min": float(a.min()),
                    "alpha_max": float(a.max()),
                }
            )
        del mlp

    # Add "no transfer" baseline = α=0.5 constant
    baseline_05 = {}
    baseline_v3 = {}
    for j, target in enumerate(DATASETS):
        td = target_data.get(target)
        if td is None:
            continue
        mix_05 = 0.5 * td["decomp"] + 0.5 * td["global"]
        mix_v3 = td["alpha_v3"] * td["decomp"] + (1 - td["alpha_v3"]) * td["global"]
        baseline_05[target] = _rmse(td["y_test"], mix_05)
        baseline_v3[target] = _rmse(td["y_test"], mix_v3)

    df = pd.DataFrame(rows)
    df["target_rmse_alpha05"] = df["target"].map(baseline_05)
    df["target_rmse_v3"] = df["target"].map(baseline_v3)
    df["beats_alpha05"] = df["rmse_transferred"] < df["target_rmse_alpha05"]
    df["beats_v3"] = df["rmse_transferred"] < df["target_rmse_v3"]
    out_csv = OUT_TBL / "cross_cell_alpha_transfer.csv"
    df.to_csv(out_csv, index=False)
    print(f"[6c] wrote {out_csv}")

    # Headline
    if not df.empty:
        off_diag = df[df["source"] != df["target"]]
        n_off = len(off_diag)
        b05 = int(off_diag["beats_alpha05"].sum())
        bv3 = int(off_diag["beats_v3"].sum())
        print(
            f"  off-diagonal transfer: beats α=0.5 in {b05}/{n_off}, "
            f"beats v3-scalar in {bv3}/{n_off}"
        )

    # Heatmap
    pretty_ds = {
        "cesnet": "CESNET",
        "abilene": "Abilene",
        "geant": "GEANT",
        "nab_aws_cpu": "NAB-CPU",
        "nab_twitter": "NAB-Twitter",
    }
    pretty_labels = [pretty_ds.get(d, d) for d in DATASETS]
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    base_row = np.array([baseline_v3.get(d, np.nan) for d in DATASETS])
    norm = rmse_mat / base_row[None, :]
    im = ax.imshow(norm, cmap="RdYlGn_r", vmin=0.95, vmax=1.5)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels(pretty_labels, rotation=30, ha="right", fontsize=12)
    ax.set_yticks(range(len(DATASETS)))
    ax.set_yticklabels(pretty_labels, fontsize=12)
    ax.set_xlabel("target dataset (evaluated on)", fontsize=12)
    ax.set_ylabel(r"source CHA-L $\alpha(x)$ MLP from", fontsize=12)
    ax.set_title(
        rf"CHA-L $\alpha(x)$ cross-cell transfer ($h={HORIZON}$)"
        "\n"
        r"cell value = RMSE(transferred) / RMSE(target's own CHA-S)",
        fontsize=12,
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
                    color="black" if v < 1.2 else "white",
                    fontsize=11,
                    fontweight="bold",
                )
    cbar = fig.colorbar(im, ax=ax, label="RMSE ratio (1.0 = matches CHA-S)")
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label("RMSE ratio (1.0 = matches CHA-S)", fontsize=12)
    fig.tight_layout()
    out_pdf = OUT_FIG / "cross_cell_alpha_heatmap.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(OUT_FIG / "cross_cell_alpha_heatmap.png", bbox_inches="tight", dpi=150)
    print(f"[6c] wrote {out_pdf}")


if __name__ == "__main__":
    with PhaseTimer(
        "expC_6c_cross_cell_alpha",
        notes=f"5x5 cross-cell α(x) transfer at h={HORIZON}, seed={SEED}",
    ) as t:
        main()
        t.add_output("csv", str(OUT_TBL / "cross_cell_alpha_transfer.csv"))
        t.add_output("pdf", str(OUT_FIG / "cross_cell_alpha_heatmap.pdf"))
