"""Recompute the bootstrap SDs of tab:multimetric (h = 6) with per-seed pairing.

The published table (pipeline/expR4_multi_metric.py) concatenated the five CHA-S
seeds into one pool of 5N windows and bootstrapped that pool, while Chronos-Bolt
is one deterministic run of N windows. The pooled CHA-S SD is therefore about
sqrt(5) too small. Here, for every seed s, the SAME resampled window indices are
applied to CHA-S (seed s) and to Chronos-Bolt (seed 42, the only run that exists),
and the reported SD of a model is the mean over seeds of the per-seed bootstrap SD.
The point estimate is the mean over seeds. Scaled-domain arrays, i.i.d. windows,
B = 2000 (the published table used B = 1000).

Run:  python3 v2/peerj_v2/scripts/recompute_multimetric_sd.py  (from the repository root)
Writes: multimetric_h6_recomputed.csv and multimetric_h6_before_after.csv next to this file.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # repository root, holds outputs/
PRED = ROOT / "outputs" / "predictions"
HERE = Path(__file__).resolve().parent
DATASETS = ["cesnet", "abilene", "geant", "nab_aws_cpu", "nab_twitter"]
SEEDS = [42, 123, 2024, 31337, 7]
H, B = 6, 2000


def load(ds, model, seed):
    d = np.load(PRED / f"{ds}__{model}__h{H}__s{seed}.npz")
    return d["y_true_scaled"].astype(np.float64), d["y_pred_scaled"].astype(np.float64)


def smape(yt, yp):
    den = (np.abs(yt) + np.abs(yp)) / 2.0
    m = den > 1e-8
    return float(np.mean(np.abs(yt[m] - yp[m]) / den[m]) * 100.0)


def metrics(yt, yp):
    return np.array(
        [np.sqrt(np.mean((yt - yp) ** 2)), np.mean(np.abs(yt - yp)), smape(yt, yp)]
    )


def mbb(n, block, B, rng):
    """Moving-block bootstrap indices, block length = forecast horizon."""
    nb = int(np.ceil(n / block))
    st = rng.integers(0, max(n - block + 1, 1), size=(B, nb))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(B, -1)[:, :n]
    return np.minimum(idx, n - 1)


rng = np.random.default_rng(20261012)
rows = []
rows_blk = []
for ds in DATASETS:
    yb_t, yb = load(ds, "chronos_bolt_zs", 42)
    per_seed = []
    blk_seed = []
    for s in SEEDS:
        yt, yc = load(ds, "cha_hybrid_v3", s)
        n = min(len(yt), len(yb))
        assert np.allclose(yt[:n], yb_t[:n])
        yt, yc, ybs = yt[:n], yc[:n], yb[:n]
        idx = rng.integers(0, n, size=(B, n))
        mc = np.array([metrics(yt[i], yc[i]) for i in idx])
        mb = np.array([metrics(yt[i], ybs[i]) for i in idx])
        per_seed.append((metrics(yt, yc), mc.std(0), metrics(yt, ybs), mb.std(0)))
        idx = mbb(n, H, B, rng)
        mc = np.array([metrics(yt[i], yc[i]) for i in idx])
        mb = np.array([metrics(yt[i], ybs[i]) for i in idx])
        blk_seed.append((mc.std(0), mb.std(0)))
    pc = np.mean([p[0] for p in per_seed], 0)
    sc = np.mean([p[1] for p in per_seed], 0)
    pb = np.mean([p[2] for p in per_seed], 0)
    sb = np.mean([p[3] for p in per_seed], 0)
    for j, name in enumerate(("rmse", "mae", "smape")):
        rows.append(
            dict(dataset=ds, metric=name, chas=pc[j], chas_sd=sc[j], bolt=pb[j], bolt_sd=sb[j])
        )
    sbc = np.mean([p[0] for p in blk_seed], 0)
    sbb = np.mean([p[1] for p in blk_seed], 0)
    for j, name in enumerate(("rmse", "mae", "smape")):
        rows_blk.append(dict(dataset=ds, metric=name, chas=pc[j], chas_sd_block6=sbc[j],
                             bolt=pb[j], bolt_sd_block6=sbb[j], gap_in_bolt_sd_block6=(pc[j]-pb[j])/sbb[j]))
pd.DataFrame(rows_blk).to_csv(HERE / "multimetric_h6_block6.csv", index=False)
new = pd.DataFrame(rows)
new.to_csv(HERE / "multimetric_h6_recomputed.csv", index=False)

# before: the legacy pooled values that were published
# The legacy table lives in the unreleased outputs/eval_v3; without it the shipped
# multimetric_h6_before_after.csv is kept as is.
if (ROOT / "outputs" / "eval_v3" / "tables" / "multi_metric_all.csv").exists():
    legacy = pd.read_csv(ROOT / "outputs" / "eval_v3" / "tables" / "multi_metric_all.csv")
    legacy = legacy[legacy.horizon == H]
    ba = []
    for r in new.itertuples():
        for model, sd_new in (("CHA-S", r.chas_sd), ("Chronos-Bolt", r.bolt_sd)):
            old = legacy[(legacy.dataset == r.dataset) & (legacy.model == model)].iloc[0]
            ba.append(
                dict(
                    dataset=r.dataset, model=model, metric=r.metric,
                    sd_before=old[f"{r.metric}_sd"], sd_after=sd_new,
                    ratio_after_over_before=sd_new / old[f"{r.metric}_sd"],
                )
            )
    ba = pd.DataFrame(ba)
    ba.to_csv(HERE / "multimetric_h6_before_after.csv", index=False)
else:
    ba = pd.read_csv(HERE / "multimetric_h6_before_after.csv")
pd.set_option("display.width", 200)
print(new.to_string(index=False))
print(ba.to_string(index=False))
sub = ba[ba.model == "CHA-S"]
print("CHA-S SD ratio after/before: mean %.3f min %.3f max %.3f" % (
    sub.ratio_after_over_before.mean(), sub.ratio_after_over_before.min(), sub.ratio_after_over_before.max()))
tw = new[(new.dataset == "nab_twitter") & (new.metric == "smape")].iloc[0]
print("NAB Twitter sMAPE gap %.3f = %.2f Bolt SD (i.i.d.)" % (tw.chas - tw.bolt, (tw.chas - tw.bolt) / tw.bolt_sd))
blk = pd.DataFrame(rows_blk)
print(blk[(blk.dataset == "nab_twitter") & (blk.metric == "smape")].to_string(index=False))
new["gap_in_bolt_sd"] = (new.chas - new.bolt) / new.bolt_sd
print("cells with |gap| > 1 Bolt SD (i.i.d.):", new[new.gap_in_bolt_sd.abs() > 1][["dataset", "metric", "gap_in_bolt_sd"]].values.tolist())
print("cells with |gap| > 1 Bolt SD (block 6):", blk[(blk.chas - blk.bolt).abs() > blk.bolt_sd_block6][["dataset", "metric"]].values.tolist())
