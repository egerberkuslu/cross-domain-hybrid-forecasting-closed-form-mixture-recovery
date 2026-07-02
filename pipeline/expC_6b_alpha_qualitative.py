"""Stage C — Defense 6b: Per-window α(x) qualitative analysis.

For each dataset at a representative horizon, pick three windows with
*high* learned α(x) (decomp-dominant) and three with *low* α(x)
(foundation-dominant).  Plot the lookback + the three forecasts
(decomp, foundation, hybrid) side-by-side, and annotate each panel
with the structural feature vector that drove the MLP's decision.

This is the visual evidence that the learned α(x) is actually using
input structure, not just collapsing to the v3 scalar.

Output:
  outputs/figures/alpha_qualitative.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.preprocessing import load_preprocessed
from src.models import build_model
from src.training import MODEL_CONFIGS
from src.models.cha_hybrid_v4 import _context_features
from src.utils.runlog import PhaseTimer

OUT_FIG = Path("outputs/figures")
DATASETS = ["cesnet", "abilene", "geant"]  # 3 datasets, 6 windows each
HORIZON = 6
SEED = 42
N_HIGH = 3
N_LOW = 3


def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretty_ds = {
        "cesnet": "CESNET",
        "abilene": "Abilene",
        "geant": "GEANT",
        "nab_aws_cpu": "NAB-CPU",
        "nab_twitter": "NAB-Twitter",
    }
    fig, axes = plt.subplots(
        len(DATASETS),
        N_HIGH + N_LOW,
        figsize=(3.2 * (N_HIGH + N_LOW), 2.8 * len(DATASETS)),
        squeeze=False,
    )

    for i, ds in enumerate(DATASETS):
        ckpt_path = Path(
            f"outputs/checkpoints/{ds}__cha_hybrid_v4__h{HORIZON}__s{SEED}.pt"
        )
        if not ckpt_path.exists():
            print(f"  no v4 ckpt for {ds}, skipping")
            for ax in axes[i]:
                ax.text(
                    0.5,
                    0.5,
                    "no v4 ckpt",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            continue
        try:
            pp = load_preprocessed(ds)
        except Exception as e:
            print(f"  load_preprocessed({ds}) failed: {e}")
            continue
        wins = pp.windows.get(HORIZON)
        if wins is None:
            continue

        # Re-instantiate v4 and load its weights
        base = next(b for r, v, b, s in MODEL_CONFIGS if v == "cha_hybrid_v4")
        m = build_model(
            "cha_hybrid_v4", horizon=HORIZON, hparams=base, seed=SEED, device=device
        )
        ts = pp.split_scaled.train["value"].to_numpy("float32")
        vs = pp.split_scaled.val["value"].to_numpy("float32")
        m.fit(wins["train"], wins["val"], train_series=ts, val_series=vs)
        # Load v4 ckpt weights to ensure we're using the actual policy
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
            if ck.get("alpha_mlp_state_dict") and m.alpha_mlp is not None:
                m.alpha_mlp.load_state_dict(ck["alpha_mlp_state_dict"])
            if ck.get("lstm_residual_state_dict"):
                m.residual_model._darts_model.model.load_state_dict(
                    ck["lstm_residual_state_dict"]
                )
        except Exception as e:
            print(f"  warn: could not reload {ds} v4 weights: {e}")

        # Compute α(x) on test set
        X_test = wins["test"].X
        feat = _context_features(X_test, stl_period=m.stl_period)
        feat_t = torch.tensor(feat, dtype=torch.float32, device=device)
        with torch.no_grad():
            alpha_per_window = m.alpha_mlp(feat_t).cpu().numpy()
        # Pick N_HIGH highest, N_LOW lowest distinct windows
        order = np.argsort(alpha_per_window)
        low_idx = order[:N_LOW]
        high_idx = order[-N_HIGH:][::-1]
        picks = list(high_idx) + list(low_idx)

        # Get the three forecasts for each picked window
        decomp = m._predict_decomposition(X_test)
        global_pred = m.global_model.predict(wins["test"])
        hybrid = m.predict(wins["test"])
        y_true = wins["test"].y

        for j, idx in enumerate(picks):
            ax = axes[i][j]
            a = float(alpha_per_window[idx])
            lookback = X_test[idx]
            tt = np.arange(lookback.shape[-1])
            ax.plot(tt[-48:], lookback[-48:], "k-", lw=0.8, alpha=0.6, label="lookback")
            ax.axvline(tt[-1], color="grey", ls=":", lw=0.5)
            t_fc = np.arange(tt[-1] + 1, tt[-1] + 1 + HORIZON)
            ax.plot(t_fc, y_true[idx], "k-", lw=1.5, label="truth")
            ax.plot(
                t_fc,
                decomp[idx],
                "tab:blue",
                ls="--",
                lw=1.0,
                label="decomp" if (i == 0 and j == 0) else None,
            )
            ax.plot(
                t_fc,
                global_pred[idx],
                "tab:orange",
                ls=":",
                lw=1.0,
                label="global" if (i == 0 and j == 0) else None,
            )
            ax.plot(
                t_fc,
                hybrid[idx],
                "tab:green",
                lw=1.4,
                label="hybrid" if (i == 0 and j == 0) else None,
            )
            tag = r"high $\alpha$" if j < N_HIGH else r"low $\alpha$"
            ax.set_title(
                rf"{pretty_ds.get(ds, ds)} -- {tag}" + "\n" + rf"$\alpha(x)={a:.2f}$",
                fontsize=11,
            )
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.3)

        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    axes[0][0].legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.suptitle(
        rf"CHA-L learned $\alpha(x)$: high-$\alpha$ (decomp dominates) "
        rf"vs low-$\alpha$ (foundation dominates), $h={HORIZON}$",
        fontsize=14,
        fontweight="bold",
        y=1.005,
    )
    fig.tight_layout()
    out_pdf = OUT_FIG / "alpha_qualitative.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(OUT_FIG / "alpha_qualitative.png", bbox_inches="tight", dpi=150)
    print(f"[6b] wrote {out_pdf}")


if __name__ == "__main__":
    with PhaseTimer(
        "expC_6b_alpha_qualitative",
        notes=f"3 high-α + 3 low-α windows per dataset at h={HORIZON}",
    ) as t:
        main()
        t.add_output("pdf", str(OUT_FIG / "alpha_qualitative.pdf"))
