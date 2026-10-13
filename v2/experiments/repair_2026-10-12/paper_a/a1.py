"""A-A1: repair of the accuracy claim (variants V0-V5). Definitions frozen in REPAIR_PREREG.md section 3.

  python3 a1.py --stage select [--dry-run]     validation only, guard on, writes frozen_a1.json (+ sha256)
  REPAIR_STAGE=test python3 a1.py --stage test needs PREREG_COMMITTED, a passing Gate 0 (or --override-gate = exploratory)
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

import rcommon as R
import stats_tools as T

W_GRID = [24, 48, 96, 192, 384]
KAPPA_GRID = [0.0, 12.0]
LAM_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
S_GRID = [0.25, 0.5, 1.0]
Q_GRID = [0.90, 0.95, 0.99]
WQ = 60  # window of the log-ratio statistic of V3
R_RESET = 24
MIN_POST = 12


def win_stats(y, dec, glob):
    eg, ed = glob - y, dec - y
    df = dec - glob
    return {"see": (eg * eg).sum(1), "sed": (eg * df).sum(1), "sdd": (df * df).sum(1), "sdec": (ed * ed).sum(1)}


# ------------------------------------------------------------------- variants
def alpha_v1(st, n_hist, h, W, kappa, a0, lo_floor=None):
    """Recency-weighted closed form with label delay h; windows s <= g - h, at most W of them."""
    n = len(st["sed"])
    cs_e = np.concatenate([[0.0], np.cumsum(st["sed"])])
    cs_d = np.concatenate([[0.0], np.cumsum(st["sdd"])])
    g = np.arange(n_hist, n)
    hi = g - h
    lo = np.maximum(0, hi - W + 1)
    if lo_floor is not None:
        lo = np.maximum(lo, lo_floor)
    nwin = np.maximum(hi - lo + 1, 0)
    se = cs_e[np.maximum(hi + 1, 0)] - cs_e[lo]
    sd = cs_d[np.maximum(hi + 1, 0)] - cs_d[lo]
    sbar = np.where(nwin > 0, sd / np.maximum(nwin, 1), 0.0)
    num = -se + kappa * sbar * a0
    den = sd + kappa * sbar
    a = np.where((den > 1e-12) & (nwin > 0), num / np.maximum(den, 1e-300), a0)
    return np.clip(a, 0.0, 1.0)


def q_stream(st, h):
    """Rolling log MSE ratio Bolt/decomposition over WQ labelled windows, delay h; NaN where unavailable."""
    n = len(st["see"])
    ce = np.concatenate([[0.0], np.cumsum(st["see"])])
    cd = np.concatenate([[0.0], np.cumsum(st["sdec"])])
    q = np.full(n, np.nan)
    for g in range(h + WQ, n):
        hi = g - h
        lo = hi - WQ + 1
        q[g] = np.log(max(ce[hi + 1] - ce[lo], 1e-12) / max(cd[hi + 1] - cd[lo], 1e-12))
    return q


def alpha_v3(st, n_hist, h, W, qlevel, a0, seed=0):
    q = q_stream(st, h)
    tr = q[:n_hist][np.isfinite(q[:n_hist])]
    if tr.size < 30:
        return alpha_v1(st, n_hist, h, W, 0.0, a0)
    mu, sd = tr.mean(), max(tr.std(), 1e-6)
    lev, _ = T.null_level((tr - mu) / sd, q=qlevel, two_sided=True, seed=seed)
    n = len(st["sed"])
    out = np.empty(n - n_hist)
    Sp = Sm = 0.0
    fit_lo = 0
    run = 0
    cs_e = np.concatenate([[0.0], np.cumsum(st["sed"])])
    cs_d = np.concatenate([[0.0], np.cumsum(st["sdd"])])
    for j, g in enumerate(range(n_hist, n)):
        z = (q[g] - mu) / sd if np.isfinite(q[g]) else 0.0
        Sp = max(0.0, Sp + z - T.K_CUSUM)
        Sm = max(0.0, Sm - z - T.K_CUSUM)
        run = run + 1 if (Sp > lev or Sm > lev) else 0
        if run >= T.PERSIST_DEFAULT:
            fit_lo = max(fit_lo, g - h - R_RESET)
            Sp = Sm = 0.0
            run = 0
        hi = g - h
        lo = max(fit_lo, hi - W + 1, 0)
        if hi - lo + 1 < MIN_POST:
            out[j] = a0
            continue
        se, sdv = cs_e[hi + 1] - cs_e[lo], cs_d[hi + 1] - cs_d[lo]
        out[j] = a0 if sdv < 1e-12 else float(np.clip(-se / sdv, 0, 1))
    return out


def alpha_v4_vec(y, dec, glob, fit_idx, a0, s):
    eg, df = (glob - y)[fit_idx], (dec - glob)[fit_idx]
    den = (df * df).sum(0)
    ak = np.where(den > 1e-12, np.clip(-(eg * df).sum(0) / np.maximum(den, 1e-300), 0, 1), a0)
    return np.clip(a0 + s * (ak - a0), 0, 1)


def stack3(y, dec, glob, fit_idx):
    """V5: NNLS on [dec-glob, persist-glob], w>=0, sum rescaled to <=1. persistence of window t = y[t-1,0]."""
    per = np.repeat(np.vstack([y[:1, :1], y[:-1, :1]]), y.shape[1], axis=1)
    fi = np.asarray([i for i in fit_idx if i >= 1])
    X = np.stack([(dec - glob)[fi].ravel(), (per - glob)[fi].ravel()], axis=1)
    t = (y - glob)[fi].ravel()
    G, c = X.T @ X, X.T @ t
    lip = np.linalg.eigvalsh(G).max() + 1e-12
    w = np.zeros(2)
    for _ in range(500):
        w = np.maximum(0.0, w - (G @ w - c) / lip)
    if w.sum() > 1:
        w = w / w.sum()
    return w, per


def forecast(variant, cfg, y, dec, glob, n_hist, h, a0, fit_idx):
    """Forecasts for global windows g >= n_hist. fit_idx: windows used for the constant fits (V2, V4, V5)."""
    st = win_stats(y, dec, glob)
    d, gl = dec[n_hist:], glob[n_hist:]
    if variant == "V0":
        a = np.full(len(d), a0)
    elif variant == "V1":
        a = alpha_v1(st, n_hist, h, cfg["W"], cfg["kappa"], a0)
    elif variant == "V2":
        a = np.full(len(d), float(np.clip(cfg["lam"] * a0, 0, 1)))
    elif variant == "V3":
        a = alpha_v3(st, n_hist, h, cfg["W"], cfg["q"], a0)
    elif variant == "V4":
        a = np.tile(alpha_v4_vec(y, dec, glob, fit_idx, a0, cfg["s"])[None, :], (len(d), 1))
    elif variant == "V5":
        w, per = stack3(y, dec, glob, fit_idx)
        return gl + w[0] * (d - gl) + w[1] * (per[n_hist:] - gl), np.array([w[0], w[1]])
    else:
        raise ValueError(variant)
    return R.mix(d, gl, a), a


def configs():
    c = {"V0": [{}]}
    c["V1"] = [{"W": W, "kappa": k} for W, k in itertools.product(W_GRID, KAPPA_GRID)]
    c["V2"] = [{"lam": l} for l in LAM_GRID]
    c["V4"] = [{"s": s} for s in S_GRID]
    return c


split_val = R.split_val


# --------------------------------------------------------------------- select
def select(dry, criterion="acc"):
    log = R.install_guard()
    R.guard_selftest()
    cells = R.CELLS[:3] if dry else R.CELLS
    seeds = R.SEEDS[:2] if dry else R.SEEDS
    cf = configs()
    rows = []
    for ci, (ds, h) in enumerate(cells):
        for s in seeds:
            v = R.load_val(ds, h, s)
            y, dec, glob = v["y"], v["dec"], v["glob"]
            n = len(y)
            nc, s0 = split_val(n, h)
            see, sed, sdd = R.window_stats(y[:nc], dec[:nc], glob[:nc])
            a0 = R.snap(R.alpha_bg(sed, sdd))
            sel = np.arange(s0, n)
            ysel = y[sel]
            r0 = R.rmse(ysel, R.mix(dec[sel], glob[sel], a0))
            r_b, r_d = R.rmse(ysel, glob[sel]), R.rmse(ysel, dec[sel])
            floor_ref = min(r_b, r_d)
            fit_idx = np.arange(nc)
            # V3 grid is built after the V1 ranking inside the same cell/seed (top-3 windows by SEL here is a leak
            # into selection only, SEL being validation, so it is allowed; test never enters)
            res = {}
            for var in ("V1", "V2", "V4"):
                for cfg in cf[var]:
                    f, _ = forecast(var, cfg, y, dec, glob, nc, h, a0, fit_idx)
                    f_sel = f[s0 - nc:]
                    res[(var, json.dumps(cfg, sort_keys=True))] = R.rmse(ysel, f_sel)
            for W in (48, 96, 192):
                for q in Q_GRID:
                    cfg = {"W": W, "q": q}
                    f, _ = forecast("V3", cfg, y, dec, glob, nc, h, a0, fit_idx)
                    res[("V3", json.dumps(cfg, sort_keys=True))] = R.rmse(ysel, f[s0 - nc:])
            for (var, cfgs), r in res.items():
                rows.append({"dataset": ds, "horizon": h, "seed": s, "variant": var, "cfg": cfgs, "rmse_sel": r, "rmse_sel_v0": r0, "log_ratio": float(np.log(r / r0)), "excess": max(0.0, r / floor_ref - 1.0)})
        R.progress(f"a1 select {ci + 1}/{len(cells)} {ds} h{h}")
    df = pd.DataFrame(rows)
    assert not any("predictions" in p for p in log), "test data touched"
    pooled = df.groupby(["variant", "cfg"]).agg(pooled_log_ratio=("log_ratio", "mean"), pooled_excess=("excess", "mean")).reset_index()
    key1, key2 = ("pooled_log_ratio", "pooled_excess") if criterion == "acc" else ("pooled_excess", "pooled_log_ratio")
    best = {}
    for var, g in pooled.groupby("variant"):
        g = g.sort_values([key1, key2])
        best[var] = {"cfg": json.loads(g.iloc[0].cfg), "pooled_log_ratio": float(g.iloc[0].pooled_log_ratio), "pooled_excess": float(g.iloc[0].pooled_excess)}
    headline = min(best, key=lambda k: (best[k][key1], best[k][key2]))
    out = {"stage": "select", "criterion": criterion, "dry_run": dry, "best_per_variant": best, "headline_variant": headline, "n_cells": len(cells), "n_seeds": len(seeds), "test_files_read": 0}
    if dry:
        print(json.dumps(out, indent=1))
        R.enable_pseudo()
        test_stage(True, frozen=out, pseudo=True, criterion=criterion)
        n_read = len([p for p in log if "predictions" in p])
        assert n_read == 0, "test data touched"
        print("dry-run ok; test files read:", n_read)
        return
    tag = "" if criterion == "acc" else "b"
    pooled.to_csv(R.HERE / f"a1{tag}_select_pooled.csv", index=False)
    df.to_csv(R.HERE / f"a1{tag}_select_rows.csv", index=False)
    p = R.HERE / f"frozen_a1{tag}.json"
    p.write_text(json.dumps(out, indent=1))
    (R.HERE / f"frozen_a1{tag}.sha256").write_text(R.sha256_file(p) + "\n")
    print(json.dumps(out, indent=1))


def evaluate_arrays(y, dec, glob, n_hist, h, best, a0s, y_native):
    """Pure function used by the test stage and the dry run: RMSE of every variant on windows >= n_hist."""
    fit_idx = np.arange(n_hist)
    a0 = a0s["V0"]
    out = {}
    yt = y[n_hist:]
    for var in ("V0", "V1", "V2", "V3", "V4"):
        cfg = {} if var == "V0" else best[var]["cfg"]
        f, _ = forecast(var, cfg, y, dec, glob, n_hist, h, a0, fit_idx)
        out[var] = R.rmse(yt, f)
    return out


# ----------------------------------------------------------------------- test
def test_stage(override, frozen=None, pseudo=False, criterion="acc"):
    tag = "" if criterion == "acc" else "b"
    if not pseudo:
        gate = json.loads((R.HERE / "gate0_summary.json").read_text())
        if criterion == "acc" and not gate["gate0a_pass_run_V1_V4"] and not override:
            raise SystemExit("Gate 0a failed: A-A1 accuracy variants are not run on test (A-A1b floor repair uses --criterion floor)")
        R.allow_test()
        frozen = json.loads((R.HERE / f"frozen_a1{tag}.json").read_text())
        assert R.sha256_file(R.HERE / f"frozen_a1{tag}.json") == (R.HERE / f"frozen_a1{tag}.sha256").read_text().strip(), f"frozen_a1{tag}.json changed"
    cells = R.CELLS[:3] if pseudo else R.CELLS
    seeds = R.SEEDS[:2] if pseudo else R.SEEDS
    best = frozen["best_per_variant"]
    headline = frozen["headline_variant"]
    rng = np.random.default_rng(20261012)
    rows, perm_rows = [], []
    for ci, (ds, h) in enumerate(cells):
        per = {}
        mse = {k: [] for k in ("bolt", "dec", "chas", "V0", headline, "ctrl_a", "ctrl_b")}
        ident = True
        for s in seeds:
            v = R.load_val(ds, h, s)
            t = R.load_test(ds, h, s)
            y = np.vstack([v["y"], t["y"]])
            dec = np.vstack([v["dec"], t["dec"]])
            glob = np.vstack([v["glob"], t["glob"]])
            n_hist = len(v["y"])
            a0 = R.deployed_alpha(ds, h, s)
            f0, _ = forecast("V0", {}, y, dec, glob, n_hist, h, a0, np.arange(n_hist))
            assert np.allclose(f0, t["chas"], atol=1e-4), f"V0 does not reproduce the stored CHA-S forecast {ds} h{h} s{s}"
            fh, ah = forecast(headline, best[headline]["cfg"], y, dec, glob, n_hist, h, a0, np.arange(n_hist))
            yt = t["y"]
            mse["bolt"].append(((t["glob"] - yt) ** 2).mean(1))
            mse["chas"].append(((t["chas"] - yt) ** 2).mean(1))
            mse["dec"].append(((t["dec"] - yt) ** 2).mean(1))
            mse["V0"].append(((f0 - yt) ** 2).mean(1))
            mse[headline].append(((fh - yt) ** 2).mean(1))
            ident &= bool(np.max(np.abs(fh - t["glob"])) <= 1e-6 * np.std(yt))
            # controls: 50 draws each, RMSE averaged later
            a_series = np.asarray(ah if np.ndim(ah) == 1 else ah.mean(1))
            ca, cb = [], []
            nblk = int(np.ceil(len(a_series) / h))
            for _ in range(50):
                order = rng.permutation(nblk)
                idx = np.concatenate([np.arange(b * h, min((b + 1) * h, len(a_series))) for b in order])
                ca.append(((R.mix(t["dec"], t["glob"], a_series[idx]) - yt) ** 2).mean(1))
                cb.append(((R.mix(t["dec"], t["glob"], rng.choice(a_series, size=len(a_series))) - yt) ** 2).mean(1))
            mse["ctrl_a"].append(np.mean(ca, axis=0))
            mse["ctrl_b"].append(np.mean(cb, axis=0))
        m = {k: np.mean(v, axis=0) for k, v in mse.items()}  # seed-averaged per-window MSE
        pt, lo, hi, dlo, dhi = T.block_boot_ratio(m[headline], m["bolt"], h)
        p_bolt = T.perm_p(m[headline], m["bolt"], h)
        p_chas = T.perm_p(m[headline], m["chas"], h)
        rb = float(np.sqrt(m["bolt"].mean()))
        rows.append({"dataset": ds, "horizon": h, "headline": headline, "rmse_bolt": rb, "rmse_chas": float(np.sqrt(m["chas"].mean())), "rmse_var": float(np.sqrt(m[headline].mean())),
                     "rel_vs_bolt": pt, "ci_lo": lo, "ci_hi": hi, "p_vs_bolt": p_bolt, "p_vs_chas": p_chas, "identical_to_bolt": ident,
                     "tost_equiv": T.tost_equivalent(dlo, dhi, rb) and not ident, "rmse_ctrl_a": float(np.sqrt(m["ctrl_a"].mean())), "rmse_ctrl_b": float(np.sqrt(m["ctrl_b"].mean())), "rmse_dec": float(np.sqrt(m["dec"].mean()))})
        R.progress(f"a1 test {ci + 1}/{len(cells)} {ds} h{h}")
    df = pd.DataFrame(rows)
    df["p_vs_bolt_holm"] = R.holm(df.p_vs_bolt.values)
    df["p_vs_chas_holm"] = R.holm(df.p_vs_chas.values)
    df["gain_vs_best_ctrl"] = 1 - df.rmse_var / np.minimum(df.rmse_ctrl_a, df.rmse_ctrl_b)
    better = int(((df.p_vs_bolt_holm < 0.05) & (df.rel_vs_bolt < 0)).sum())
    worse = int(((df.p_vs_bolt_holm < 0.05) & (df.rel_vs_bolt > 0)).sum())
    cellsl = list(zip(df.dataset, df.horizon))
    g = T.cell_cluster_boot_mean(df.gain_vs_best_ctrl.values, cellsl)
    g2 = T.cell_cluster_boot_mean(df.gain_vs_best_ctrl.values, cellsl, two_stage=True)
    summ = {"criterion": criterion, "headline": headline, "cfg": best[headline]["cfg"], "cells_better_vs_bolt_holm": better, "cells_worse_vs_bolt_holm": worse,
            "cells_identical": int(df.identical_to_bolt.sum()), "cells_tost_equiv": int(df.tost_equiv.sum()),
            "pooled_gain_vs_best_control": {"mean": g[0], "ci95_cell": [g[1], g[2]], "ci95_two_stage": [g2[1], g2[2]]},
            "success": bool(better >= 5 and worse == 0 and g[1] > 0), "exploratory_override_gate": bool(override), "pseudo": bool(pseudo)}
    if criterion == "floor":
        ref = np.minimum(df.rmse_bolt, df.rmse_dec)
        tol = 1.001
        hold = lambda r: (r <= tol * ref).values
        h_var, h_v0 = hold(df.rmse_var), hold(df.rmse_chas)
        h_ca, h_cb = hold(df.rmse_ctrl_a), hold(df.rmse_ctrl_b)
        from scipy.stats import binomtest
        b_, c_ = int((h_var & ~h_v0).sum()), int((~h_var & h_v0).sum())
        p_sign = float(binomtest(b_, b_ + c_, 0.5, alternative="greater").pvalue) if b_ + c_ else 1.0
        summ["floor"] = {"holds_variant": int(h_var.sum()), "holds_V0": int(h_v0.sum()), "holds_ctrl_a": int(h_ca.sum()), "holds_ctrl_b": int(h_cb.sum()),
                         "gained_vs_V0": b_, "lost_vs_V0": c_, "p_sign_one_sided": p_sign,
                         "max_excess_pct_variant": float(100 * (df.rmse_var / ref - 1).max()), "max_excess_pct_V0": float(100 * (df.rmse_chas / ref - 1).max()),
                         "mean_rel_vs_bolt_variant": float(df.rel_vs_bolt.mean())}
        summ["success_floor"] = bool(h_var.sum() >= 22 and p_sign < 0.05 and h_var.sum() > max(h_ca.sum(), h_cb.sum()) and df.rel_vs_bolt.mean() <= float((df.rmse_chas / df.rmse_bolt - 1).mean()) + 1e-12)
    df.to_csv(R.OUT / f"a1{tag}_test_cells.csv", index=False)
    (R.OUT / f"a1{tag}_test_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--override-gate", action="store_true")
    ap.add_argument("--criterion", choices=["acc", "floor"], default="acc")
    a = ap.parse_args()
    select(a.dry_run, a.criterion) if a.stage == "select" else test_stage(a.override_gate, criterion=a.criterion)
