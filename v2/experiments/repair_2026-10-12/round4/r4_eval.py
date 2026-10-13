"""Round 4 evaluation: guarded E-Q mixture on the pool-4 accuracy cells (REPAIR_PREREG_4.md section 4).

  python3 r4_eval.py --stage select [--dry-run]       validation arrays only (r4_val/): guarded weights -> frozen_r4.json (+ sha256)
  REPAIR_STAGE=test python3 r4_eval.py --stage test   needs PREREG_4_COMMITTED and r4_test/
Guard = the variant chosen on the round-3 validation arrays (frozen_r4_guard.json): G0 alpha | G1 fallback alpha=0 if the gain of alpha(first 70 %) on the last 30 % is < 0.5 %
| G2(l) l * alpha | G12(l). alpha = Bates-Granger on all validation windows, snapped to 0.05 before the guard.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "round3"))
sys.path.insert(0, str(HERE.parent / "paper_a"))
sys.path.insert(0, str(HERE.parent / "fresh_cells"))
import rcommon as R  # noqa: E402
import stats_tools as T  # noqa: E402
import r3_eval as E3  # noqa: E402
import r4_stages as S4  # noqa: E402
import r4_guard as G  # noqa: E402

RETENTION = None  # set from the chosen guard
MIN_GAIN_SCALED = None


def guard_cfg():
    g = json.loads((HERE / "frozen_r4_guard.json").read_text())
    ch = g["chosen"]
    lam = float(ch.split("_")[1]) if "_" in ch else 1.0
    return ch, lam, 2 * lam - lam ** 2 if ch.startswith("G2") else (1.0 if ch in ("G0", "G1") else 2 * lam - lam ** 2)


def guarded_alpha(y, B, E, h):
    """alpha_full on all validation windows, then the chosen guard (decision from alpha on the first 70 % evaluated on the last 30 %)."""
    ch, lam, _ = guard_cfg()
    a_full = E3.alpha_of(y, B, E)
    n = len(y)
    ai = np.arange(int(0.7 * n))
    bi = np.arange(int(0.7 * n) + h, n)
    a70 = E3.alpha_of(y[ai], B[ai], E[ai])
    gb = 1 - G.rm(y[bi], B[bi] + a70 * (E[bi] - B[bi])) / G.rm(y[bi], B[bi])
    keep = gb >= G.TAU
    if ch == "G0":
        return a_full
    if ch == "G1":
        return a_full if keep else 0.0
    if ch.startswith("G2"):
        return lam * a_full
    return lam * (a_full if keep else 0.0)


def select(cells, val_dir, out_json, dry=False):
    al = {}
    for name, h in cells:
        v = E3.load(val_dir, name, h)
        al[f"{name}|{h}"] = {"alpha_eq": guarded_alpha(v["y"], v["bolt"], v["eq"], h), "alpha_eq0": guarded_alpha(v["y"], v["bolt"], v["eq0"], h),
                             "alpha_eq_unguarded": E3.alpha_of(v["y"], v["bolt"], v["eq"]), "n_val": int(len(v["y"]))}
    out = {"stage": "select", "dry_run": dry, "guard": guard_cfg()[0], "alphas": al, "n_cells": len(cells)}
    if not dry:
        Path(out_json).write_text(json.dumps(out, indent=1))
        (HERE / "frozen_r4.sha256").write_text(R.sha256_file(out_json) + "\n")
    return out


def summarize(df):
    ch, lam, ret = guard_cfg()
    min_gain = 0.005 * ret
    df = df.copy()
    df["p_holm"] = R.holm(df.p_vs_bolt.values)
    df["gain_vs_bolt"] = -df.rel_vs_bolt
    df["gain_vs_best_ctrl"] = 1 - df.rmse_var / np.minimum(df.rmse_ctrl_a, df.rmse_ctrl_b)
    K = int(np.ceil(0.08 * len(df)))
    win = (df.p_holm < 0.05) & (df.rel_vs_bolt <= -min_gain)
    win05 = (df.p_holm < 0.05) & (df.rel_vs_bolt <= -0.005)
    loss = (df.p_holm < 0.05) & (df.rel_vs_bolt > 0)
    cl = list(zip(df.series, df.horizon))
    gb, gb2 = T.cell_cluster_boot_mean(df.gain_vs_bolt.values, cl), T.cell_cluster_boot_mean(df.gain_vs_bolt.values, cl, two_stage=True)
    gc, gc2 = T.cell_cluster_boot_mean(df.gain_vs_best_ctrl.values, cl), T.cell_cluster_boot_mean(df.gain_vs_best_ctrl.values, cl, two_stage=True)
    src = df.series.str.replace(r"\d.*$", "", regex=True).str.rstrip("_")
    per_src = {s: {"cells": int((src == s).sum()), "wins": int(win[src == s].sum()), "mean_gain": float(df.gain_vs_bolt[src == s].mean()), "median_gain": float(df.gain_vs_bolt[src == s].median())} for s in sorted(set(src))}
    med = float(df.gain_vs_bolt.median())
    ok = bool(win.sum() >= K and loss.sum() == 0 and gb[0] >= min_gain and gb[1] > 0 and gb2[1] > 0 and gc[1] > 0 and gc2[1] > 0 and med > 0)
    ok05 = bool(win05.sum() >= K and loss.sum() == 0 and gb[0] >= 0.005 and gb[1] > 0 and gb2[1] > 0 and gc[1] > 0 and gc2[1] > 0 and med > 0)
    return df, {"guard": ch, "retention": ret, "min_gain_primary": min_gain, "cells": int(len(df)), "K_wins_required": K, "cells_win": int(win.sum()), "cells_win_at_0.5pct": int(win05.sum()), "cells_loss": int(loss.sum()),
                "cells_identical": int(df.identical_to_bolt.sum()), "cells_tost_equiv": int(df.tost_equiv.sum()), "median_cell_gain": med,
                "pooled_gain_vs_bolt": {"mean": gb[0], "ci95_cell": [gb[1], gb[2]], "ci95_series_clustered": [gb2[1], gb2[2]]},
                "pooled_gain_vs_best_control": {"mean": gc[0], "ci95_cell": [gc[1], gc[2]], "ci95_series_clustered": [gc2[1], gc2[2]]}, "per_source": per_src,
                "success": ok, "success_secondary_unscaled_0.5pct": ok05}


def test(cells, val_dir, test_dir, frozen, out_dir, pseudo=False):
    rng = np.random.default_rng(20261014)
    rows = []
    for i, (name, h) in enumerate(cells):
        v, t = E3.load(val_dir, name, h), E3.load(test_dir, name, h)
        rows.append(E3.test_cell(name, h, v, t, frozen["alphas"][f"{name}|{h}"], rng))
        R.progress(f"r4 test {i + 1}/{len(cells)} {name} h{h}")
    df, s = summarize(pd.DataFrame(rows))
    s["pseudo"] = bool(pseudo)
    df.to_csv(out_dir / "r4_test_cells.csv", index=False)
    (out_dir / "r4_test_summary.json").write_text(json.dumps(s, indent=1, default=float))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    log = R.install_guard()
    R.guard_selftest()
    cells = [c for c in S4.pool_cells()]
    acc = [c for c in cells if c[1] != 6 or c[0] in {x[0] for x in cells if x[1] == 1}]
    if a.dry_run:
        import shutil
        import tempfile
        from src.preprocessing import load_preprocessed
        import fresh_cells as F
        import r3_stages as S3
        tmp = Path(tempfile.mkdtemp(dir=str(HERE)))
        rv = F.Restricted(load_preprocessed("abilene"))
        cs = [("abilene", 6), ("abilene", 12)]
        S3.MODELS = tmp / "m"
        for name, h in cs:
            S3.stage_a(name, h, rv, out_dir=tmp, save_models=True, dry=True)
        fr = select(cs, tmp, tmp / "f.json", dry=True)
        tdir = tmp / "t"; tdir.mkdir()
        for name, h in cs:
            o = S3.stage_b(name, h, rv, pseudo=True)
            np.savez(tdir / f"{name}__h{h}.npz", **o)
        s = test(cs, tmp, tdir, fr, tmp, pseudo=True)
        print(json.dumps({k: v for k, v in s.items() if k != "per_source"}, indent=1, default=float))
        assert not any(sp == "test" for _, sp in rv.accessed)
        shutil.rmtree(tmp)
        print("dry-run ok; test windows accessed: 0")
        return
    m = json.loads((HERE.parent / "fresh_cells" / "pool4_manifest.json").read_text())
    accnames = {s["name"] for s in m["series"] if s["accuracy"]}
    acc = [c for c in cells if c[0] in accnames]
    if a.stage == "select":
        out = select(acc, S4.VAL, HERE / "frozen_r4.json")
        print("frozen guarded alphas for", out["n_cells"], "cells; guard", out["guard"])
        return
    if not (HERE / "PREREG_4_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test":
        raise SystemExit("test stage needs PREREG_4_COMMITTED and REPAIR_STAGE=test")
    p = HERE / "frozen_r4.json"
    assert R.sha256_file(p) == (HERE / "frozen_r4.sha256").read_text().strip(), "frozen_r4.json changed"
    s = test(acc, S4.VAL, S4.TEST, json.loads(p.read_text()), HERE)
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
