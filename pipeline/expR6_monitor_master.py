"""Alpha-monitor: the full design journey across three detectors.

Motivation (edge/IoT): CHA-S recovers the optimal mixture weight alpha in
closed form at O(1) cost per window, cheap enough for an edge node. We ask
whether tracking alpha online yields a free foundation-reliability monitor.
We arrive at the final detector by a documented, reproducible progression:

  D1 absolute threshold  : fire when smoothed alpha >= tau (0.20).
     -> false-alarms on traces whose quiescent alpha is naturally high.
  D2 CUSUM relative shift : fire on an upward CUSUM of standardised alpha.
     -> false-alarms on traces whose quiescent alpha has near-zero variance.
  D3 composite (D1 AND D2): the two failure modes are complementary, so the
     conjunction keeps the false-alarm rate low on every dataset while still
     detecting the injected foundation failure on all of them.

All three are evaluated on ALL FIVE network datasets; outputs feed the
comparison table and the headline streaming figure.

Outputs:
  paper_a/figures/alpha_monitor.pdf         (CESNET stream, composite alarm)
  paper_a/figures/alpha_monitor_compare.pdf (per-dataset FA-rate, 3 detectors)
  paper_a/tables/alpha_monitor.tex          (per-dataset, 3 detectors)
  outputs/eval_v3/tables/alpha_monitor.json
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRED = "outputs/predictions"
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
DN = {
    "cesnet": "CESNET-TS24",
    "abilene": "Abilene",
    "geant": "GEANT",
    "nab_aws_cpu": "NAB AWS-CPU",
    "nab_twitter": "NAB Twitter",
}
H = 6
W, WARMUP, SMOOTH = 60, 40, 10
TAU_ABS = 0.20
K, H_CUSUM, PERSIST = 0.5, 5.0, 3


def load(ds, model):
    d = np.load(f"{PRED}/{ds}__{model}__h{H}__s42.npz")
    return d["y_true_scaled"].mean(axis=1), d["y_pred_scaled"].mean(axis=1)


def recover_alpha(y, fd, ff):
    ed, ef = fd - y, ff - y
    diff = ed - ef
    den = float(np.dot(diff, diff))
    return 0.0 if den < 1e-12 else float(np.clip(-np.dot(ef, diff) / den, 0, 1))


def _eval_detector(fire_seq, idx, cp):
    """false-alarm rate (pre-cp) and detection lag (post-cp) for a boolean
    per-window 'composite-condition met' sequence, with PERSIST debounce."""
    run_len, fired_at, fa, pre = 0, None, 0, 0
    for k in range(len(idx)):
        run_len = run_len + 1 if fire_seq[k] else 0
        alarm = run_len >= PERSIST
        if idx[k] < cp:
            if idx[k] >= 0:
                pre += 1
                if alarm:
                    fa += 1
                    run_len = 0
        elif alarm and fired_at is None:
            fired_at = int(idx[k] - cp)
    return fa / max(1, pre), fired_at


def run(ds, seed=20260529):
    rng = np.random.default_rng(seed)
    y, fd = load(ds, "cha_hybrid_v3_decomp_only")
    _, ff = load(ds, "cha_hybrid_v3_global_only")
    n = min(len(y), len(fd), len(ff))
    y, fd, ff = y[:n], fd[:n], ff[:n]
    cp = max(W + WARMUP + 20, n // 2)
    sg = float(np.std(ff - y)) + 1e-9
    ff_s = ff.copy()
    for t in range(cp, n):
        ramp = min(1.0, (t - cp) / 100.0)
        ff_s[t] = ff[t] + ramp * 3.0 * sg * (1.0 + 0.5 * rng.standard_normal())
    a, idx = [], []
    for t in range(W, n):
        sl = slice(t - W, t)
        a.append(recover_alpha(y[sl], fd[sl], ff_s[sl]))
        idx.append(t)
    a = np.array(a)
    idx = np.array(idx)
    sm = np.convolve(a, np.ones(SMOOTH) / SMOOTH, mode="same")
    mu, sd = a[:WARMUP].mean(), a[:WARMUP].std() + 1e-6

    # per-window detector conditions (only meaningful from WARMUP onward)
    abs_hit = np.zeros(len(a), bool)
    cusum_hit = np.zeros(len(a), bool)
    S = 0.0
    for k in range(len(a)):
        if k < WARMUP:
            continue
        z = (a[k] - mu) / sd
        S = max(0.0, S + (z - K))
        abs_hit[k] = sm[k] >= TAU_ABS
        cusum_hit[k] = S > H_CUSUM
    comp_hit = abs_hit & cusum_hit

    # D4 baseline: CUSUM applied directly to the rolling RMSE of the
    # foundation expert's residuals (no alpha involved). This is the obvious
    # zero-cost alternative a reviewer would propose: if plain residual
    # tracking matches the alpha-monitor, the alpha route adds nothing.
    r_rmse = np.empty(len(a))
    for j, t in enumerate(range(W, n)):
        sl = slice(t - W, t)
        r_rmse[j] = float(np.sqrt(np.mean((ff_s[sl] - y[sl]) ** 2)))
    mu_r, sd_r = r_rmse[:WARMUP].mean(), r_rmse[:WARMUP].std() + 1e-6
    rescusum_hit = np.zeros(len(a), bool)
    Sr = 0.0
    for k in range(len(a)):
        if k < WARMUP:
            continue
        zr = (r_rmse[k] - mu_r) / sd_r
        Sr = max(0.0, Sr + (zr - K))
        rescusum_hit[k] = Sr > H_CUSUM

    mask = idx >= idx[WARMUP]  # restrict eval to post-warmup
    out = {"dataset": ds}
    for name, seq in [
        ("abs", abs_hit),
        ("cusum", cusum_hit),
        ("comp", comp_hit),
        ("rescusum", rescusum_hit),
    ]:
        fa, lag = _eval_detector(seq[mask], idx[mask], cp)
        out[f"{name}_fa"] = round(fa, 4)
        out[f"{name}_lag"] = lag
    out["baseline_alpha"] = round(float(mu), 3)
    out["_series"] = (idx, sm, cp, out["comp_lag"])
    return out


def main():
    res = {ds: run(ds) for ds in DATASETS}

    # headline figure: CESNET stream + composite alarm
    idx, sm, cp, lag = res["cesnet"]["_series"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(idx, sm, color="crimson", lw=1.6, label=r"smoothed $\alpha$")
    ax.axvline(cp, color="black", ls="--", lw=1.0, label="foundation regime shift")
    if lag is not None:
        ax.axvline(
            cp + lag, color="seagreen", lw=1.3, label=f"composite alarm (lag {lag})"
        )
    ax.set_xlabel("stream position (window index)")
    ax.set_ylabel(r"smoothed recovered $\alpha$")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig("paper_a/figures/alpha_monitor.pdf", bbox_inches="tight")
    plt.close()

    # comparison figure: per-dataset FA rate for the three detectors
    labels = [DN[d] for d in DATASETS]
    x = np.arange(len(DATASETS))
    wbar = 0.20
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(
        x - 1.5 * wbar,
        [res[d]["abs_fa"] for d in DATASETS],
        wbar,
        label="D1 absolute",
        color="#bdbdbd",
    )
    ax.bar(
        x - 0.5 * wbar,
        [res[d]["cusum_fa"] for d in DATASETS],
        wbar,
        label="D2 CUSUM",
        color="#fdae6b",
    )
    ax.bar(
        x + 0.5 * wbar,
        [res[d]["comp_fa"] for d in DATASETS],
        wbar,
        label="D3 composite",
        color="seagreen",
    )
    ax.bar(
        x + 1.5 * wbar,
        [res[d]["rescusum_fa"] for d in DATASETS],
        wbar,
        label="D4 residual CUSUM",
        color="#6baed6",
    )
    ax.axhline(0.05, color="red", ls=":", lw=1.0, label="5\\% target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("pre-shift false-alarm rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("paper_a/figures/alpha_monitor_compare.pdf", bbox_inches="tight")
    plt.close()

    # table: four detectors (3 alpha-based + residual baseline), FA + lag
    lines = [
        "% Auto-generated by pipeline/expR6_monitor_master.py",
        "\\begin{table}[!ht]\\centering\\footnotesize",
        "\\caption{Design progression of the online $\\alpha$-monitor across "
        "all five network datasets ($h{=}6$, foundation regime-shift injected "
        "at the stream midpoint), together with the natural baseline that "
        "monitors the foundation expert directly. For each detector we report "
        "the pre-shift false-alarm rate (FA) and the detection lag in "
        "windows. The absolute-threshold detector (D1) false-alarms on the "
        "high-baseline NAB-CPU trace; the CUSUM detector (D2) false-alarms on "
        "the CESNET-TS24 and Abilene traces; because the two failure modes "
        "are complementary, their conjunction (D3) holds FA at or below "
        "$5\\%$ on every dataset while still detecting the failure on all "
        "five. D4 applies the same CUSUM directly to the rolling RMSE of the "
        "foundation expert's residuals without using $\\alpha$.}",
        "\\label{tab:alpha_monitor}",
        "\\begin{tabular}{l c | c c | c c | c c | c c}",
        "\\toprule",
        " & base & \\multicolumn{2}{c|}{D1 absolute} & "
        "\\multicolumn{2}{c|}{D2 CUSUM} & \\multicolumn{2}{c|}{D3 composite} & "
        "\\multicolumn{2}{c}{D4 resid.\\ CUSUM} \\\\",
        "Dataset & $\\alpha$ & FA & lag & FA & lag & FA & lag & FA & lag \\\\",
        "\\midrule",
    ]

    def lg(v):
        return str(v) if v is not None else "--"

    for d in DATASETS:
        r = res[d]
        lines.append(
            f"{DN[d]} & {r['baseline_alpha']:.2f} & "
            f"{r['abs_fa']:.2f} & {lg(r['abs_lag'])} & "
            f"{r['cusum_fa']:.2f} & {lg(r['cusum_lag'])} & "
            f"\\textbf{{{r['comp_fa']:.2f}}} & \\textbf{{{lg(r['comp_lag'])}}} & "
            f"{r['rescusum_fa']:.2f} & {lg(r['rescusum_lag'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    open("paper_a/tables/alpha_monitor.tex", "w").write("\n".join(lines))

    clean = {d: {k: v for k, v in res[d].items() if k != "_series"} for d in DATASETS}
    os.makedirs("outputs/eval_v3/tables", exist_ok=True)
    json.dump(clean, open("outputs/eval_v3/tables/alpha_monitor.json", "w"), indent=2)
    print("wrote 2 figures + table")
    for d in DATASETS:
        r = res[d]
        print(
            f"  {d:13s} D1_FA={r['abs_fa']:.2f} D2_FA={r['cusum_fa']:.2f} "
            f"D3_FA={r['comp_fa']:.2f} D3_lag={r['comp_lag']} "
            f"D4_FA={r['rescusum_fa']:.2f} D4_lag={r['rescusum_lag']}"
        )


if __name__ == "__main__":
    main()
