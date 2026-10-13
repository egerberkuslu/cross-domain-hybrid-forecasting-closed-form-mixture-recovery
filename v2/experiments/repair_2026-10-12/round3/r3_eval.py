"""Round 3 evaluation of the E-Q mixture on the fresh pool (REPAIR_PREREG_3.md sections 3 and 4).

  python3 r3_eval.py --stage select [--dry-run]      validation arrays only (r3_val/), writes frozen_r3.json (+ sha256): alpha per cell
  REPAIR_STAGE=test python3 r3_eval.py --stage test  needs PREREG_3_COMMITTED and r3_test/ (r3_stages.py --stage B)
Mixture M2: f = Bolt + alpha (E-Q - Bolt), alpha = Bates-Granger closed form on all validation windows, clipped, snapped to the 0.05 grid.
Holm family = all cells. Success: see section 4 of the prereg.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "paper_a"))
sys.path.insert(0, str(HERE.parent / "fresh_cells"))
import rcommon as R  # noqa: E402
import stats_tools as T  # noqa: E402
import r3_stages as S  # noqa: E402

K_WINS = 8
MIN_GAIN = 0.005


def alpha_of(y, B, E):
    d = E - B
    return R.snap(R.alpha_bg(np.sum((B - y) * d, 1), np.sum(d * d, 1)))


def load(dirp, name, h):
    z = np.load(dirp / f"{name}__h{h}.npz")
    return {k: z[k].astype(np.float64) for k in z.files}


def select(cells, val_dir, out_json, dry=False):
    al = {}
    for name, h in cells:
        v = load(val_dir, name, h)
        al[f"{name}|{h}"] = {"alpha_eq": alpha_of(v["y"], v["bolt"], v["eq"]), "alpha_eq0": alpha_of(v["y"], v["bolt"], v["eq0"]), "n_val": int(len(v["y"]))}
    out = {"stage": "select", "dry_run": dry, "alphas": al, "n_cells": len(cells)}
    if not dry:
        Path(out_json).write_text(json.dumps(out, indent=1))
        (HERE / "frozen_r3.sha256").write_text(R.sha256_file(out_json) + "\n")
    return out


def test_cell(name, h, v, t, a, rng):
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
    # reference: the validation-chosen best single expert
    best_is_eq = np.mean((v["eq"] - v["y"]) ** 2) < np.mean((v["bolt"] - v["y"]) ** 2)
    se_best = ((E - y) ** 2).mean(1) if best_is_eq else se_b
    return {"series": name, "horizon": h, "alpha_eq": a["alpha_eq"], "rmse_bolt": rb, "rmse_var": float(np.sqrt(se_v.mean())), "rel_vs_bolt": pt, "ci_lo": lo, "ci_hi": hi,
            "p_vs_bolt": T.perm_p(se_v, se_b, h), "identical_to_bolt": ident, "tost_equiv": T.tost_equivalent(dlo, dhi, rb) and not ident,
            "rmse_ctrl_a": float(np.sqrt(mse_a.mean())), "rmse_ctrl_b": float(np.sqrt(mse_b.mean())), "rel_best_single_vs_bolt": float(np.sqrt(se_best.mean()) / rb - 1.0)}


def summarize(df):
    df = df.copy()
    df["p_holm"] = R.holm(df.p_vs_bolt.values)
    df["gain_vs_bolt"] = -df.rel_vs_bolt
    df["gain_vs_best_ctrl"] = 1 - df.rmse_var / np.minimum(df.rmse_ctrl_a, df.rmse_ctrl_b)
    win = (df.p_holm < 0.05) & (df.rel_vs_bolt <= -MIN_GAIN)
    loss = (df.p_holm < 0.05) & (df.rel_vs_bolt > 0)
    cl = list(zip(df.series, df.horizon))
    gb, gb2 = T.cell_cluster_boot_mean(df.gain_vs_bolt.values, cl), T.cell_cluster_boot_mean(df.gain_vs_bolt.values, cl, two_stage=True)
    gc, gc2 = T.cell_cluster_boot_mean(df.gain_vs_best_ctrl.values, cl), T.cell_cluster_boot_mean(df.gain_vs_best_ctrl.values, cl, two_stage=True)
    src = df.series.str.replace(r"\d.*$", "", regex=True).str.rstrip("_")
    per_src = {s: {"cells": int((src == s).sum()), "wins": int(win[src == s].sum()), "mean_gain": float(df.gain_vs_bolt[src == s].mean())} for s in sorted(set(src))}
    ok = bool(win.sum() >= K_WINS and loss.sum() == 0 and gb[0] >= MIN_GAIN and gb[1] > 0 and gb2[1] > 0 and gc[1] > 0 and gc2[1] > 0)
    return df, {"cells": int(len(df)), "holm_family": int(len(df)), "K_wins_required": K_WINS, "cells_win": int(win.sum()), "cells_loss": int(loss.sum()), "cells_identical": int(df.identical_to_bolt.sum()),
                "cells_tost_equiv": int(df.tost_equiv.sum()), "pooled_gain_vs_bolt": {"mean": gb[0], "ci95_cell": [gb[1], gb[2]], "ci95_series_clustered": [gb2[1], gb2[2]]},
                "pooled_gain_vs_best_control": {"mean": gc[0], "ci95_cell": [gc[1], gc[2]], "ci95_series_clustered": [gc2[1], gc2[2]]}, "per_source": per_src,
                "mean_rel_best_single_vs_bolt": float(df.rel_best_single_vs_bolt.mean()), "success": ok}


def test(cells, val_dir, test_dir, frozen, out_dir, pseudo=False):
    rng = np.random.default_rng(20261013)
    rows = []
    for i, (name, h) in enumerate(cells):
        v, t = load(val_dir, name, h), load(test_dir, name, h)
        rows.append(test_cell(name, h, v, t, frozen["alphas"][f"{name}|{h}"], rng))
        R.progress(f"r3 test {i + 1}/{len(cells)} {name} h{h}")
    df, s = summarize(pd.DataFrame(rows))
    s["pseudo"] = bool(pseudo)
    df.to_csv(out_dir / "r3_test_cells.csv", index=False)
    (out_dir / "r3_test_summary.json").write_text(json.dumps(s, indent=1, default=float))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    log = R.install_guard()
    R.guard_selftest()
    if a.dry_run:
        # non-fresh pseudo run: two Abilene aggregate cells, stage A on a quarter of train, pseudo stage B on the validation windows
        import tempfile
        from src.preprocessing import load_preprocessed
        import fresh_cells as F
        tmp = Path(tempfile.mkdtemp(dir=str(HERE)))
        rv = F.Restricted(load_preprocessed("abilene"))
        cells = [("abilene", 6), ("abilene", 12)]
        for name, h in cells:
            S.stage_a(name, h, rv, out_dir=tmp, save_models=True, dry=True)
        fr = select(cells, tmp, tmp / "frozen.json", dry=True)
        for name, h in cells:
            o = S.stage_b(name, h, rv, pseudo=True)
            np.savez(tmp / f"T_{name}__h{h}.npz", **o)
        for name, h in cells:
            (tmp / f"{name}__h{h}_t.npz").write_bytes((tmp / f"T_{name}__h{h}.npz").read_bytes())
        import shutil
        tdir = tmp / "t"; tdir.mkdir()
        for name, h in cells:
            shutil.copy(tmp / f"T_{name}__h{h}.npz", tdir / f"{name}__h{h}.npz")
        s = test(cells, tmp, tdir, fr, tmp, pseudo=True)
        print(json.dumps({k: v for k, v in s.items() if k != "per_source"}, indent=1, default=float))
        assert not any(sp == "test" for _, sp in rv.accessed), "test windows accessed"
        shutil.rmtree(tmp)
        for p in S.MODELS.glob("abilene__*"):
            p.unlink()  # dry-run artefacts of the non-fresh cell
        print("dry-run ok; test windows accessed: 0")
        return
    cells = S.pool_cells()
    if a.stage == "select":
        out = select(cells, S.VAL, HERE / "frozen_r3.json")
        print("frozen alphas for", out["n_cells"], "cells")
        return
    import os
    if not (HERE / "PREREG_3_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test":
        raise SystemExit("test stage needs PREREG_3_COMMITTED and REPAIR_STAGE=test")
    p = HERE / "frozen_r3.json"
    assert R.sha256_file(p) == (HERE / "frozen_r3.sha256").read_text().strip(), "frozen_r3.json changed"
    s = test(cells, S.VAL, S.TEST, json.loads(p.read_text()), HERE)
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
