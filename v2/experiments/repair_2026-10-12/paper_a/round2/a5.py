"""Round 2: test of the candidate experts in the CHA-S mixture against Chronos-Bolt. REPAIR_PREREG_2.md section 4.

  python3 a5.py --stage select [--dry-run] [--experts EB EQ]   validation only, writes frozen_a5.json (+ sha256)
  REPAIR_STAGE=test python3 a5.py --stage test                 needs PREREG_2_COMMITTED and test_experts/ (experts_test.py)
Mixture variants  M2 = Bolt + alpha (E - Bolt), alpha on the 0.05 grid from the Bates-Granger closed form on the fit slice
                  M3 = Bolt + w1 (dec - Bolt) + w2 (E - Bolt), NNLS on the fit slice, w >= 0, sum rescaled to <= 1 (per decomposition seed)
One Holm family over all candidate experts x 25 cells.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rcommon as R  # noqa: E402
import stats_tools as T  # noqa: E402

PSEUDO = False
NULLS = {"EQ": "EQ0", "EB": "EB0"}


def nnls(Xd, yd, iters=400):
    w = np.zeros(Xd.shape[1])
    Gm, c = Xd.T @ Xd, Xd.T @ yd
    lip = np.linalg.eigvalsh(Gm).max() + 1e-12
    for _ in range(iters):
        w = np.maximum(0.0, w - (Gm @ w - c) / lip)
    return w


def load_E(ds, h, ex, role):
    """Expert forecasts aligned to the arrays of rcommon: 'val' (fit slice) or 'test'."""
    n = len(R._raw_val(ds, h, R.SEEDS[0])["y"])
    if role == "val" or PSEUDO:
        E = np.load(HERE / "val_experts" / f"{ds}__h{h}.npz")[ex].astype(np.float64)[:n]
        if PSEUDO:
            nc, s0 = R.split_val(n, h)
            return E[:nc] if role == "val" else E[s0:n]
        return E
    return np.load(HERE / "test_experts" / f"{ds}__h{h}.npz")[ex].astype(np.float64)


def fit_variants(y, B, E, D, h, fit_idx):
    see, sed, sdd = R.window_stats(y[fit_idx], E[fit_idx], B[fit_idx])
    a2 = R.snap(R.alpha_bg(sed, sdd))
    X = np.stack([(D - B)[fit_idx].ravel(), (E - B)[fit_idx].ravel()], 1)
    w = nnls(X, (y - B)[fit_idx].ravel())
    if w.sum() > 1:
        w = w / w.sum()
    return a2, w


def forecasts(variant, B, E, D, a2, w):
    return B + a2 * (E - B) if variant == "M2" else B + w[0] * (D - B) + w[1] * (E - B)


def select(dry, experts):
    log = R.install_guard()
    R.guard_selftest()
    experts = experts or json.loads((HERE / "gate_e0_summary.json").read_text())["test_candidates"]
    if not experts:
        raise SystemExit("no expert candidates: nothing to select")
    cells = R.CELLS[:3] if dry else R.CELLS
    seeds = R.SEEDS[:2] if dry else R.SEEDS
    res = {}
    for expert in experts:
        rows = []
        for ds, h in cells:
            n = len(R._raw_val(ds, h, R.SEEDS[0])["y"])
            E = np.load(HERE / "val_experts" / f"{ds}__h{h}.npz")[expert].astype(np.float64)[:n]
            for s in seeds:
                v = R._raw_val(ds, h, s)
                y, B, D = (v[k].astype(np.float64) for k in ("y", "glob", "dec"))
                nc, s0 = R.split_val(n, h)
                cal, sel = np.arange(nc), np.arange(s0, n)
                a2, w = fit_variants(y, B, E, D, h, cal)
                rB = R.rmse(y[sel], B[sel])
                for var in ("M2", "M3"):
                    f = forecasts(var, B[sel], E[sel], D[sel], a2, w)
                    rows.append({"dataset": ds, "horizon": h, "seed": s, "variant": var, "log_ratio": float(np.log(R.rmse(y[sel], f) / rB))})
        pooled = pd.DataFrame(rows).groupby("variant").log_ratio.mean().to_dict()
        res[expert] = {"pooled_log_ratio_vs_bolt": pooled, "headline_variant": min(pooled, key=pooled.get)}
    assert not any("predictions" in p for p in log), "test data touched"
    out = {"stage": "select", "dry_run": dry, "experts": res, "test_files_read": 0}
    print(json.dumps(out, indent=1))
    if dry:
        R.enable_pseudo()
        globals()["PSEUDO"] = True
        test_stage(frozen=out, pseudo=True, cells=cells, seeds=seeds)
        assert not any("predictions" in p for p in log)
        print("dry-run ok; test files read: 0")
        return
    p = HERE / "frozen_a5.json"
    p.write_text(json.dumps(out, indent=1))
    (HERE / "frozen_a5.sha256").write_text(R.sha256_file(p) + "\n")


def test_cell(ds, h, ex, var, seeds, rng):
    null = NULLS.get(ex)
    m = {k: [] for k in ("bolt", "chas", "var", "ctrl_a", "ctrl_b")}
    ident = True
    for s in seeds:
        v = R.load_val(ds, h, s)
        t = R.load_test(ds, h, s)
        n = len(v["y"])
        Ev, Et = load_E(ds, h, ex, "val")[:n], load_E(ds, h, ex, "test")
        nt = min(len(Et), len(t["y"]))
        yv, Bv, Dv = v["y"], v["glob"], v["dec"]
        yt, Bt, Dt, Et = t["y"][:nt], t["glob"][:nt], t["dec"][:nt], Et[:nt]
        a2, w = fit_variants(yv, Bv, Ev, Dv, h, np.arange(n))
        f = forecasts(var, Bt, Et, Dt, a2, w)
        ident &= bool(np.max(np.abs(f - Bt)) <= 1e-6 * np.std(yt))
        m["bolt"].append(((Bt - yt) ** 2).mean(1))
        m["chas"].append(((t["chas"][:nt] - yt) ** 2).mean(1))
        m["var"].append(((f - yt) ** 2).mean(1))
        ca = []
        nblk = int(np.ceil(nt / h))
        for _ in range(50):
            order = rng.permutation(nblk)
            idx = np.concatenate([np.arange(b * h, min((b + 1) * h, nt)) for b in order])
            ca.append(((forecasts(var, Bt, Et[idx], Dt, a2, w) - yt) ** 2).mean(1))
        m["ctrl_a"].append(np.mean(ca, axis=0))
        E0v, E0t = load_E(ds, h, null, "val")[:n], load_E(ds, h, null, "test")[:nt]
        a20, w0 = fit_variants(yv, Bv, E0v, Dv, h, np.arange(n))
        m["ctrl_b"].append(((forecasts(var, Bt, E0t, Dt, a20, w0) - yt) ** 2).mean(1))
    mm = {k: np.mean(v, axis=0) for k, v in m.items()}
    pt, lo, hi, dlo, dhi = T.block_boot_ratio(mm["var"], mm["bolt"], h)
    rb = float(np.sqrt(mm["bolt"].mean()))
    return {"dataset": ds, "horizon": h, "expert": ex, "variant": var, "rmse_bolt": rb, "rmse_chas": float(np.sqrt(mm["chas"].mean())), "rmse_var": float(np.sqrt(mm["var"].mean())),
            "rel_vs_bolt": pt, "ci_lo": lo, "ci_hi": hi, "p_vs_bolt": T.perm_p(mm["var"], mm["bolt"], h), "p_vs_chas": T.perm_p(mm["var"], mm["chas"], h),
            "identical_to_bolt": ident, "tost_equiv": T.tost_equivalent(dlo, dhi, rb) and not ident,
            "rmse_ctrl_a": float(np.sqrt(mm["ctrl_a"].mean())), "rmse_ctrl_b": float(np.sqrt(mm["ctrl_b"].mean()))}


def test_stage(frozen=None, pseudo=False, cells=None, seeds=None):
    if not pseudo:
        if not (HERE / "PREREG_2_COMMITTED").exists():
            raise R.TestAccessError("PREREG_2_COMMITTED missing")
        R.allow_test()
        p = HERE / "frozen_a5.json"
        assert R.sha256_file(p) == (HERE / "frozen_a5.sha256").read_text().strip(), "frozen_a5.json changed"
        frozen = json.loads(p.read_text())
    cells = cells or R.CELLS
    seeds = seeds or R.SEEDS
    rng = np.random.default_rng(20261012)
    rows = []
    for ex, info in frozen["experts"].items():
        for ci, (ds, h) in enumerate(cells):
            rows.append(test_cell(ds, h, ex, info["headline_variant"], seeds, rng))
            R.progress(f"a5 test {ex} {ci + 1}/{len(cells)} {ds} h{h}")
    df = pd.DataFrame(rows)
    df["p_bolt_holm"] = R.holm(df.p_vs_bolt.values)  # one family over all experts x cells
    df["p_chas_holm"] = R.holm(df.p_vs_chas.values)
    df["gain_vs_best_ctrl"] = 1 - df.rmse_var / np.minimum(df.rmse_ctrl_a, df.rmse_ctrl_b)
    df["gain_vs_bolt"] = -df.rel_vs_bolt
    summ = {"holm_family_size": int(len(df)), "pseudo": bool(pseudo), "experts": {}}
    for ex in df.expert.unique():
        d = df[df.expert == ex]
        win = (d.p_bolt_holm < 0.05) & (d.rel_vs_bolt <= -0.005)
        loss = (d.p_bolt_holm < 0.05) & (d.rel_vs_bolt > 0)
        cl = list(zip(d.dataset, d.horizon))
        g_b = T.cell_cluster_boot_mean(d.gain_vs_bolt.values, cl)
        g_c = T.cell_cluster_boot_mean(d.gain_vs_best_ctrl.values, cl)
        g_c2 = T.cell_cluster_boot_mean(d.gain_vs_best_ctrl.values, cl, two_stage=True)
        summ["experts"][ex] = {"variant": d.variant.iloc[0], "cells_win_vs_bolt_holm_and_ge_0.5pct": int(win.sum()), "cells_loss_vs_bolt_holm": int(loss.sum()),
                               "cells_identical": int(d.identical_to_bolt.sum()), "cells_tost_equiv": int(d.tost_equiv.sum()),
                               "pooled_gain_vs_bolt": {"mean": g_b[0], "ci95_cell": [g_b[1], g_b[2]]},
                               "pooled_gain_vs_best_control": {"mean": g_c[0], "ci95_cell": [g_c[1], g_c[2]], "ci95_two_stage": [g_c2[1], g_c2[2]]},
                               "success": bool(win.sum() >= 5 and loss.sum() == 0 and g_b[0] >= 0.005 and g_b[1] > 0 and g_c[1] > 0)}
    outd = (HERE / "dryrun_out") if pseudo else HERE
    outd.mkdir(exist_ok=True)
    df.to_csv(outd / "a5_test_cells.csv", index=False)
    (outd / "a5_test_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--experts", nargs="*", default=None)
    a = ap.parse_args()
    select(a.dry_run, a.experts) if a.stage == "select" else test_stage()
