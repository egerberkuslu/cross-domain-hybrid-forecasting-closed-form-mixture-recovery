"""Round 5 evaluation: guarded E-Q mixture on the stratum-A accuracy cells (REPAIR_PREREG_5.md section 4).

  python3 r5_eval.py --stage select [--dry-run]       validation arrays only (r5_val/): guarded weights -> frozen_r5.json (+ sha256)
  REPAIR_STAGE=test python3 r5_eval.py --stage test   needs PREREG_5_COMMITTED and r5_test/
Recipe and guard are those of round 4 (frozen_r4_guard.json, G2(0.25)), imported unchanged from r4_eval.guarded_alpha.
Changes against round 4: (i) condition 4 is one comparison per control (permuted-rows control a, null expert b), each by a series-clustered two-stage bootstrap, Holm over the two, and the pooled gain over each control must also be at least the minimum effect m;
(ii) the paired sign-flip permutation has B = 400,000 draws, for Monte Carlo precision near the Holm threshold (with 520 cells the threshold of the first ranks is about 1e-4, so B = 20,000 would put the Holm decision on a few permutation draws);
(iii) the random seed of the control permutations is new. Everything else (K wins, no loss, pooled gain, median) is round 4.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent / "round4", HERE.parent / "round3", HERE.parent / "paper_a", HERE.parent / "fresh_cells"):
    sys.path.insert(0, str(p))
import rcommon as R  # noqa: E402
import stats_tools as T  # noqa: E402
import r3_eval as E3  # noqa: E402
import r4_eval as E4  # noqa: E402
import r5_stages as S5  # noqa: E402

PERM_B = 400000
BOOT_B = 10000
MIN_PRIMARY = 0.005 * 0.4375
MIN_SECONDARY = 0.005
SEED_CTRL = 20261015


def select(cells, val_dir, out_json, dry=False):
    al = {}
    for name, h in cells:
        v = E3.load(val_dir, name, h)
        al[f"{name}|{h}"] = {"alpha_eq": E4.guarded_alpha(v["y"], v["bolt"], v["eq"], h), "alpha_eq0": E4.guarded_alpha(v["y"], v["bolt"], v["eq0"], h),
                             "alpha_eq_unguarded": E3.alpha_of(v["y"], v["bolt"], v["eq"]), "n_val": int(len(v["y"]))}
    out = {"stage": "select", "dry_run": dry, "guard": E4.guard_cfg()[0], "alphas": al, "n_cells": len(cells)}
    if not dry:
        Path(out_json).write_text(json.dumps(out, indent=1))
        (HERE / "frozen_r5.sha256").write_text(R.sha256_file(out_json) + "\n")
    return out


def test_cell(name, h, v, t, a, rng):
    """round3/r3_eval.test_cell with the permutation p at B = PERM_B; all other statistics unchanged."""
    y, B, E, E0 = t["y"], t["bolt"], t["eq"], t["eq0"]
    f = B + a["alpha_eq"] * (E - B)
    se_v, se_b = ((f - y) ** 2).mean(1), ((B - y) ** 2).mean(1)
    n = len(y)
    nblk = int(np.ceil(n / h))
    ca = []
    for _ in range(50):
        order = rng.permutation(nblk)
        idx = np.concatenate([np.arange(b * h, min((b + 1) * h, n)) for b in order])
        ca.append(((B + a["alpha_eq"] * (E[idx] - B) - y) ** 2).mean(1))
    mse_a = np.mean(ca, axis=0)
    mse_b = ((B + a["alpha_eq0"] * (E0 - B) - y) ** 2).mean(1)
    pt, lo, hi, dlo, dhi = T.block_boot_ratio(se_v, se_b, h)
    rb = float(np.sqrt(se_b.mean()))
    ident = bool(np.max(np.abs(f - B)) <= 1e-6 * np.std(y))
    return {"series": name, "horizon": h, "alpha_eq": a["alpha_eq"], "rmse_bolt": rb, "rmse_var": float(np.sqrt(se_v.mean())), "rel_vs_bolt": pt, "ci_lo": lo, "ci_hi": hi,
            "p_vs_bolt": T.perm_p(se_v, se_b, h, B=PERM_B), "identical_to_bolt": ident, "tost_equiv": T.tost_equivalent(dlo, dhi, rb) and not ident,
            "rmse_ctrl_a": float(np.sqrt(mse_a.mean())), "rmse_ctrl_b": float(np.sqrt(mse_b.mean()))}


def boot_means(vals, series, B=BOOT_B, seed=0, two_stage=True):
    """Bootstrap replicates of the mean of per-cell values. two_stage: series resampled, then the cells inside each drawn series; else cells resampled."""
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    if not two_stage:
        return vals[rng.integers(0, len(vals), size=(B, len(vals)))].mean(1)
    ser = np.asarray(series)
    u = sorted(set(ser))
    groups = [vals[ser == s] for s in u]
    sizes = {len(g) for g in groups}
    if len(sizes) == 1:
        M = np.vstack(groups)
        S, k = M.shape
        pick = rng.integers(0, S, size=(B, S))
        hid = rng.integers(0, k, size=(B, S, k))
        return M[pick[:, :, None], hid].mean((1, 2))
    m = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(u), size=len(u))
        m[i] = np.concatenate([groups[p][rng.integers(0, len(groups[p]), size=len(groups[p]))] for p in pick]).mean()
    return m


def ci_p(vals, series, two_stage=True, seed=0):
    m = boot_means(vals, series, two_stage=two_stage, seed=seed)
    p = (1 + int((m <= 0).sum())) / (len(m) + 1)
    return {"mean": float(np.mean(vals)), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)), "p_one_sided": float(p)}


def summarize(df):
    df = df.copy()
    df["p_holm"] = R.holm(df.p_vs_bolt.values)
    df["gain_vs_bolt"] = -df.rel_vs_bolt
    df["gain_vs_ctrl_a"] = 1 - df.rmse_var / df.rmse_ctrl_a
    df["gain_vs_ctrl_b"] = 1 - df.rmse_var / df.rmse_ctrl_b
    df["gain_vs_best_ctrl"] = 1 - df.rmse_var / np.minimum(df.rmse_ctrl_a, df.rmse_ctrl_b)
    K = int(np.ceil(0.08 * len(df)))
    ser = df.series.values
    loss = (df.p_holm < 0.05) & (df.rel_vs_bolt > 0)
    b_cell, b_ser = ci_p(df.gain_vs_bolt.values, ser, two_stage=False), ci_p(df.gain_vs_bolt.values, ser)
    ca, cb = ci_p(df.gain_vs_ctrl_a.values, ser, seed=1), ci_p(df.gain_vs_ctrl_b.values, ser, seed=2)
    hp = R.holm(np.array([ca["p_one_sided"], cb["p_one_sided"]]))
    ctrl_ok = bool(hp.max() < 0.05)
    best = ci_p(df.gain_vs_best_ctrl.values, ser, seed=3)  # round 4 comparison, reported only
    src = df.series.str.replace(r"\d.*$", "", regex=True).str.rstrip("_")
    med = float(df.gain_vs_bolt.median())

    def rule(m):
        win = (df.p_holm < 0.05) & (df.rel_vs_bolt <= -m)
        return win, bool(win.sum() >= K and loss.sum() == 0 and b_ser["mean"] >= m and b_cell["lo"] > 0 and b_ser["lo"] > 0 and ctrl_ok and ca["mean"] >= m and cb["mean"] >= m and med > 0)

    win_p, ok = rule(MIN_PRIMARY)
    win_s, ok_s = rule(MIN_SECONDARY)
    per_src = {s: {"cells": int((src == s).sum()), "wins": int(win_p[src == s].sum()), "mean_gain": float(df.gain_vs_bolt[src == s].mean()), "median_gain": float(df.gain_vs_bolt[src == s].median()),
                   "mean_gain_vs_ctrl_a": float(df.gain_vs_ctrl_a[src == s].mean()), "mean_gain_vs_ctrl_b": float(df.gain_vs_ctrl_b[src == s].mean())} for s in sorted(set(src))}
    return df, {"guard": E4.guard_cfg()[0], "min_gain_primary": MIN_PRIMARY, "cells": int(len(df)), "series": int(len(set(ser))), "K_wins_required": K,
                "cells_win_primary": int(win_p.sum()), "cells_win_at_0.5pct": int(win_s.sum()), "cells_loss": int(loss.sum()), "cells_identical": int(df.identical_to_bolt.sum()),
                "cells_tost_equiv": int(df.tost_equiv.sum()), "median_cell_gain": med,
                "pooled_gain_vs_bolt": {"mean": b_ser["mean"], "ci95_cell": [b_cell["lo"], b_cell["hi"]], "ci95_series_clustered": [b_ser["lo"], b_ser["hi"]]},
                "control_a_permuted_rows": ca, "control_b_null_expert": cb, "control_holm_adjusted_p": [float(x) for x in hp], "controls_ok": ctrl_ok,
                "round4_comparison_best_of_two_controls_reported_only": best, "per_source": per_src,
                "success": ok, "success_secondary_0.5pct": ok_s}


def test(cells, val_dir, test_dir, frozen, out_dir, pseudo=False):
    rng = np.random.default_rng(SEED_CTRL)
    rows = []
    for i, (name, h) in enumerate(cells):
        v, t = E3.load(val_dir, name, h), E3.load(test_dir, name, h)
        rows.append(test_cell(name, h, v, t, frozen["alphas"][f"{name}|{h}"], rng))
        R.progress(f"r5 test {i + 1}/{len(cells)} {name} h{h}")
    df, s = summarize(pd.DataFrame(rows))
    s["pseudo"] = bool(pseudo)
    df.to_csv(out_dir / "r5_test_cells.csv", index=False)
    (out_dir / "r5_test_summary.json").write_text(json.dumps(s, indent=1, default=float))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    R.install_guard()
    R.guard_selftest()
    if a.dry_run:
        global PERM_B
        PERM_B = 2000
        import shutil
        import tempfile
        from src.preprocessing import load_preprocessed
        import r3_stages as S3
        tmp = Path(tempfile.mkdtemp(dir=str(HERE)))
        rv = F_restricted = __import__("fresh_cells").Restricted(load_preprocessed("abilene"))
        cs = [("abilene", 6), ("abilene", 12)]
        S3.MODELS = tmp / "m"
        for name, h in cs:
            S3.stage_a(name, h, rv, out_dir=tmp, save_models=True, dry=True)
        fr = select(cs, tmp, tmp / "f.json", dry=True)
        tdir = tmp / "t"
        tdir.mkdir()
        for name, h in cs:
            o = S3.stage_b(name, h, rv, pseudo=True)
            np.savez(tdir / f"{name}__h{h}.npz", **o)
        s = test(cs, tmp, tdir, fr, tmp, pseudo=True)
        print(json.dumps({k: v for k, v in s.items() if k not in ("per_source",)}, indent=1, default=float))
        assert not any(sp == "test" for _, sp in rv.accessed)
        shutil.rmtree(tmp)
        print("dry-run ok; test windows accessed: 0")
        return
    m = json.loads((HERE.parent / "fresh_cells" / "pool5_manifest.json").read_text())
    accnames = {s["name"] for s in m["series"] if s["accuracy"]}
    acc = [c for c in S5.pool_cells() if c[0] in accnames]
    if a.stage == "select":
        out = select(acc, S5.VAL, HERE / "frozen_r5.json")
        print("frozen guarded alphas for", out["n_cells"], "cells; guard", out["guard"])
        return
    if not (HERE / "PREREG_5_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test":
        raise SystemExit("test stage needs PREREG_5_COMMITTED and REPAIR_STAGE=test")
    p = HERE / "frozen_r5.json"
    assert R.sha256_file(p) == (HERE / "frozen_r5.sha256").read_text().strip(), "frozen_r5.json changed"
    s = test(acc, S5.VAL, S5.TEST, json.loads(p.read_text()), HERE)
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
