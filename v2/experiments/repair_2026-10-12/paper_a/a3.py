"""A-A3: repair of the learned weight alpha(x) (L1 ridge, L2 stumps, L3 pooled ridge, L4 distilled). REPAIR_PREREG.md section 5.

  python3 a3.py --stage select [--dry-run]     validation only (guard on), writes frozen_a3.json (+ sha256)
  REPAIR_STAGE=test python3 a3.py --stage test needs PREREG_COMMITTED
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

import rcommon as R
import stats_tools as T

LB = 24  # feature look-back
LAM_REL = [0.1, 1.0, 10.0, 100.0, 1000.0]
ROUNDS = [5, 10, 20, 40]
NU = 0.1
MU_REL = [1.0, 10.0, 100.0, 1000.0, np.inf]
NFEAT = 4
K_FOLDS = 5
MODELS = ("L1", "L2", "L3", "L4")


# ------------------------------------------------------------------ features
def win_stats(y, dec, glob):
    eg, ed, df = glob - y, dec - y, dec - glob
    return {"see": (eg * eg).sum(1), "sed": (eg * df).sum(1), "sdd": (df * df).sum(1), "sdec": (ed * ed).sum(1)}


def raw_features(y, st, h):
    """(n, 4) causal features: volatility, level gap, slope of the last LB values, label-delayed log error ratio."""
    n = len(y)
    x = y[:, 0]
    F = np.zeros((n, NFEAT))
    ok = np.zeros(n, bool)
    if n > LB:
        win = swv(x, LB)[: n - LB]  # win[i] = x[i:i+LB] -> belongs to window t = i + LB
        t = np.arange(LB, n)
        F[t, 0] = win.std(1)
        F[t, 1] = win[:, -6:].mean(1) - win.mean(1)
        i = np.arange(LB) - (LB - 1) / 2
        F[t, 2] = (win * i[None, :]).sum(1) / (i * i).sum()
        ok[t] = True
    ce = np.concatenate([[0.0], np.cumsum(st["see"])])
    cd = np.concatenate([[0.0], np.cumsum(st["sdec"])])
    for t in range(h + LB, n):
        hi = t - h
        lo = hi - LB + 1
        F[t, 3] = np.log(max(ce[hi + 1] - ce[lo], 1e-12) / max(cd[hi + 1] - cd[lo], 1e-12))
        ok[t] &= True
    ok[: h + LB] = False
    return F, ok


class Scaler:
    def __init__(self, F, ok, idx):
        m = ok[idx]
        self.mu = F[idx][m].mean(0) if m.any() else np.zeros(NFEAT)
        self.sd = np.maximum(F[idx][m].std(0), 1e-8) if m.any() else np.ones(NFEAT)

    def __call__(self, F, ok):
        X = (F - self.mu) / self.sd
        X[~ok] = 0.0
        return np.clip(X, -5.0, 5.0)


# -------------------------------------------------------------------- models
def suff(X, st, idx):
    sed, sdd = st["sed"][idx], st["sdd"][idx]
    Xi = X[idx]
    return (Xi * sdd[:, None]).T @ Xi, Xi.T @ sed, sdd, sed


def fit_l1(X, st, idx, a0, lam_rel):
    A, c1, _, _ = suff(X, st, idx)
    c = -(c1 + a0 * (X[idx] * st["sdd"][idx][:, None]).sum(0))
    lam = lam_rel * np.trace(A) / NFEAT + 1e-12
    return np.linalg.solve(A + lam * np.eye(NFEAT), c)


def alpha_lin(X, a0, b):
    return np.clip(a0 + X @ b, 0.0, 1.0)


def fit_l2(X, st, idx, a0, rounds_max):
    """Newton-boosted depth-1 stumps on alpha(x); returns list of (j, thr, vl, vr)."""
    a = np.full(len(idx), a0)
    Xi = X[idx]
    sed, sdd = st["sed"][idx], st["sdd"][idx]
    stumps = []
    for _ in range(rounds_max):
        g = sed + a * sdd
        hh = sdd + 1e-12
        best = None
        for j in range(NFEAT):
            o = np.argsort(Xi[:, j])
            cg, ch = np.cumsum(g[o]), np.cumsum(hh[o])
            G, Hs = cg[-1], ch[-1]
            pos = np.unique(np.clip((np.arange(1, 10) / 10 * len(o)).astype(int), 10, len(o) - 10)) if len(o) > 20 else []
            for p in pos:
                gl, hl, gr, hr = cg[p - 1], ch[p - 1], G - cg[p - 1], Hs - ch[p - 1]
                gain = gl * gl / hl + gr * gr / hr
                if best is None or gain > best[0]:
                    best = (gain, j, 0.5 * (Xi[o[p - 1], j] + Xi[o[p], j]), -gl / hl, -gr / hr)
        if best is None:
            break
        _, j, thr, vl, vr = best
        a = a + NU * np.where(Xi[:, j] <= thr, vl, vr)
        stumps.append((j, thr, vl, vr))
    return stumps


def alpha_stumps(X, a0, stumps, k):
    a = np.full(len(X), a0)
    for (j, thr, vl, vr) in stumps[:k]:
        a = a + NU * np.where(X[:, j] <= thr, vl, vr)
    return np.clip(a, 0.0, 1.0)


def centred_target(st, W):
    n = len(st["sed"])
    ce = np.concatenate([[0.0], np.cumsum(st["sed"])])
    cd = np.concatenate([[0.0], np.cumsum(st["sdd"])])
    t = np.arange(n)
    lo, hi = np.maximum(0, t - W // 2), np.minimum(n, t + W // 2 + 1)
    se, sd = ce[hi] - ce[lo], cd[hi] - cd[lo]
    return np.where(sd > 1e-12, np.clip(-se / np.maximum(sd, 1e-300), 0, 1), 0.0)


def fit_l4(X, st, idx, a0, lam_rel, tgt):
    w = st["sdd"][idx]
    Xi = X[idx]
    A = (Xi * w[:, None]).T @ Xi
    c = (Xi * (w * (tgt[idx] - a0))[:, None]).sum(0)
    lam = lam_rel * np.trace(A) / NFEAT + 1e-12
    return np.linalg.solve(A + lam * np.eye(NFEAT), c)


def fit_l3_cell(Ac, cc, b_pool, mu_rel):
    if np.isinf(mu_rel):
        return b_pool.copy()
    mu = mu_rel * np.trace(Ac) / NFEAT + 1e-12
    return np.linalg.solve(Ac + mu * np.eye(NFEAT), cc + mu * b_pool)


def mse_of(alpha, st, idx, h):
    a = alpha
    return float(np.mean((st["see"][idx] + 2 * a * st["sed"][idx] + a * a * st["sdd"][idx]) / h))


def folds(n, h):
    edges = np.linspace(0, n, K_FOLDS + 1).astype(int)
    out = []
    for k in range(K_FOLDS):
        te = np.arange(edges[k], edges[k + 1])
        tr = np.array([i for i in range(n) if i < te[0] - h or i > te[-1] + h])
        out.append((tr, te))
    return out


def a0_of(st, idx):
    return float(np.clip(-st["sed"][idx].sum() / max(st["sdd"][idx].sum(), 1e-12), 0, 1))


# ---------------------------------------------------------------- selection
def load_cells(cells, seeds):
    D = {}
    for (ds, h) in cells:
        for s in seeds:
            v = R.load_val(ds, h, s)
            st = win_stats(v["y"], v["dec"], v["glob"])
            F, ok = raw_features(v["y"], st, h)
            n = len(v["y"])
            sc = Scaler(F, ok, np.arange(int(0.6 * n)))
            D[(ds, h, s)] = {"v": v, "st": st, "F": F, "ok": ok, "n": n, "h": h}
    return D


def select(dry):
    log = R.install_guard()
    R.guard_selftest()
    cells = R.CELLS[:3] if dry else R.CELLS
    seeds = R.SEEDS[:2] if dry else R.SEEDS
    try:
        W_star = int(json.loads((R.HERE / "frozen_a1.json").read_text())["best_per_variant"]["V1"]["cfg"]["W"])
    except Exception:
        W_star = 96
    D = load_cells(cells, seeds)
    cv_rows, chosen = [], {}
    # scale per cell on the FINAL FIT slice (all validation windows)
    for key, d in D.items():
        d["sc"] = Scaler(d["F"], d["ok"], np.arange(d["n"]))
        d["X"] = d["sc"](d["F"], d["ok"])
        d["tgt"] = centred_target(d["st"], W_star)
    # L3 pooled normal equations per seed
    for s in seeds:
        keys = [k for k in D if k[2] == s]
        for k in keys:
            d = D[k]
            idx = np.arange(d["n"])
            a0 = a0_of(d["st"], idx)
            A, c1, _, _ = suff(d["X"], d["st"], idx)
            d["Anorm"] = A / d["st"]["see"].sum()
            d["cnorm"] = -(c1 + a0 * (d["X"] * d["st"]["sdd"][:, None]).sum(0)) / d["st"]["see"].sum()
    for key, d in D.items():
        ds, h, s = key
        st, X, n = d["st"], d["X"], d["n"]
        others = [k for k in D if k[2] == s and k != key]
        score = {}
        for fi, (tr, te) in enumerate(folds(n, h)):
            a0 = a0_of(st, tr)
            base = mse_of(np.full(len(te), a0), st, te, h)
            for lam in LAM_REL:
                b = fit_l1(X, st, tr, a0, lam)
                score.setdefault(("L1", lam), []).append(mse_of(alpha_lin(X[te], a0, b), st, te, h) / base)
                b4 = fit_l4(X, st, tr, a0, lam, d["tgt"])
                score.setdefault(("L4", lam), []).append(mse_of(alpha_lin(X[te], a0, b4), st, te, h) / base)
            stumps = fit_l2(X, st, tr, a0, max(ROUNDS))
            for Rr in ROUNDS:
                score.setdefault(("L2", Rr), []).append(mse_of(alpha_stumps(X[te], a0, stumps, Rr), st, te, h) / base)
            if others:
                Ap = sum(o_["Anorm"] for o_ in (D[k] for k in others))
                cp = sum(o_["cnorm"] for o_ in (D[k] for k in others))
                b_pool = np.linalg.solve(Ap + 10.0 * np.trace(Ap) / NFEAT * np.eye(NFEAT), cp)
                Atr, c1, _, _ = suff(X, st, tr)
                ctr = -(c1 + a0 * (X[tr] * st["sdd"][tr][:, None]).sum(0))
                for mu in MU_REL:
                    b = fit_l3_cell(Atr, ctr, b_pool, mu)
                    score.setdefault(("L3", mu), []).append(mse_of(alpha_lin(X[te], a0, b), st, te, h) / base)
        for (m, hp), v in score.items():
            cv_rows.append({"dataset": ds, "horizon": h, "seed": s, "model": m, "hp": hp, "cv_ratio": float(np.mean(v))})
        R.progress(f"a3 select {ds} h{h} s{s}")
    cv = pd.DataFrame(cv_rows)
    assert not any("predictions" in p for p in log), "test data touched"
    # per (cell, seed, model) best hyper-parameter; headline model by pooled mean log ratio of the CV ratio at that hp
    best = cv.loc[cv.groupby(["dataset", "horizon", "seed", "model"]).cv_ratio.idxmin()]
    pooled = best.groupby("model").cv_ratio.apply(lambda x: float(np.mean(np.log(x)))).to_dict()
    headline = min(pooled, key=pooled.get)
    hp = {f"{r.dataset}|{r.horizon}|{r.seed}|{r.model}": (None if np.isinf(r.hp) else float(r.hp)) for r in best.itertuples()}
    out = {"stage": "select", "dry_run": dry, "W_star": W_star, "headline_model": headline, "pooled_mean_log_cv_ratio": pooled, "hp": hp,
           "n_cells": len(cells), "n_seeds": len(seeds), "test_files_read": 0}
    print(json.dumps({k: v for k, v in out.items() if k != "hp"}, indent=1))
    if dry:
        R.enable_pseudo()
        test_stage(frozen=out, pseudo=True, cells=cells, seeds=seeds)
        n_read = len([p for p in log if "predictions" in p])
        assert n_read == 0, "test data touched"
        print("dry-run ok; test files read:", n_read)
        return
    cv.to_csv(R.HERE / "a3_select_cv.csv", index=False)
    p = R.HERE / "frozen_a3.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    (R.HERE / "frozen_a3.sha256").write_text(R.sha256_file(p) + "\n")


# ----------------------------------------------------------------------- test
def fit_final(models_needed, D, key, hp, W_star):
    """FINAL FIT on all validation windows of one cell with the frozen hyper-parameters; returns a predictor alpha(X)."""
    d = D[key]
    ds, h, s = key
    st, X, n = d["st"], d["X"], d["n"]
    idx = np.arange(n)
    a0 = R.deployed_alpha(ds, h, s)
    out = {}
    if "L1" in models_needed:
        b = fit_l1(X, st, idx, a0, hp[f"{ds}|{h}|{s}|L1"])
        out["L1"] = lambda Xt, b=b: alpha_lin(Xt, a0, b)
    if "L4" in models_needed:
        b = fit_l4(X, st, idx, a0, hp[f"{ds}|{h}|{s}|L4"], d["tgt"])
        out["L4"] = lambda Xt, b=b: alpha_lin(Xt, a0, b)
    if "L2" in models_needed:
        Rr = int(hp[f"{ds}|{h}|{s}|L2"])
        stumps = fit_l2(X, st, idx, a0, Rr)
        out["L2"] = lambda Xt, stumps=stumps, Rr=Rr: alpha_stumps(Xt, a0, stumps, Rr)
    if "L3" in models_needed:
        others = [k for k in D if k[2] == s and k != key]
        Ap = d["Anorm"] + sum(D[k]["Anorm"] for k in others)
        cp = d["cnorm"] + sum(D[k]["cnorm"] for k in others)
        b_pool = np.linalg.solve(Ap + 10.0 * np.trace(Ap) / NFEAT * np.eye(NFEAT), cp)
        Ac, c1, _, _ = suff(X, st, idx)
        cc = -(c1 + a0 * (X * st["sdd"][:, None]).sum(0))
        mu = hp[f"{ds}|{h}|{s}|L3"]
        b = fit_l3_cell(Ac, cc, b_pool, np.inf if mu is None else mu)
        out["L3"] = lambda Xt, b=b: alpha_lin(Xt, a0, b)
    return out, a0


def test_stage(frozen=None, pseudo=False, cells=None, seeds=None):
    if not pseudo:
        R.allow_test()
        p = R.HERE / "frozen_a3.json"
        assert R.sha256_file(p) == (R.HERE / "frozen_a3.sha256").read_text().strip(), "frozen_a3.json changed"
        frozen = json.loads(p.read_text())
    cells = cells or R.CELLS
    seeds = seeds or R.SEEDS
    headline, hp, W_star = frozen["headline_model"], frozen["hp"], frozen["W_star"]
    D = load_cells(cells, seeds)
    for key, d in D.items():
        d["sc"] = Scaler(d["F"], d["ok"], np.arange(d["n"]))
        d["X"] = d["sc"](d["F"], d["ok"])
        d["tgt"] = centred_target(d["st"], W_star)
    for s in seeds:
        for k in [k for k in D if k[2] == s]:
            d = D[k]
            idx = np.arange(d["n"])
            a0 = R.deployed_alpha(k[0], k[1], s)
            A, c1, _, _ = suff(d["X"], d["st"], idx)
            d["Anorm"] = A / d["st"]["see"].sum()
            d["cnorm"] = -(c1 + a0 * (d["X"] * d["st"]["sdd"][:, None]).sum(0)) / d["st"]["see"].sum()
    rng = np.random.default_rng(20261012)
    rows = []
    for ci, (ds, h) in enumerate(cells):
        m = {k: [] for k in ("bolt", "scalar", "model", "ctrl_a", "ctrl_b")}
        for s in seeds:
            key = (ds, h, s)
            d = D[key]
            t = R.load_test(ds, h, s)
            y = np.vstack([d["v"]["y"], t["y"]])
            dec = np.vstack([d["v"]["dec"], t["dec"]])
            glob = np.vstack([d["v"]["glob"], t["glob"]])
            st = win_stats(y, dec, glob)
            F, ok = raw_features(y, st, h)
            X = d["sc"](F, ok)[d["n"]:]
            pred, a0 = fit_final([headline], D, key, hp, W_star)
            al = pred[headline](X)
            yt = t["y"]
            mse = lambda a: ((R.mix(t["dec"], t["glob"], a) - yt) ** 2).mean(1)
            m["bolt"].append(((t["glob"] - yt) ** 2).mean(1))
            m["scalar"].append(mse(np.full(len(yt), a0)))
            assert np.allclose(R.mix(t["dec"], t["glob"], np.full(len(yt), a0)), t["chas"], atol=1e-4), "scalar CHA-S not reproduced"
            m["model"].append(mse(al))
            nblk = int(np.ceil(len(yt) / h))
            ca, cb = [], []
            for _ in range(50):
                order = rng.permutation(nblk)
                idx = np.concatenate([np.arange(b * h, min((b + 1) * h, len(yt))) for b in order])
                ca.append(mse(pred[headline](X[idx])))
                cb.append(mse(rng.choice(al, size=len(al))))
            m["ctrl_a"].append(np.mean(ca, axis=0))
            m["ctrl_b"].append(np.mean(cb, axis=0))
        mm = {k: np.mean(v, axis=0) for k, v in m.items()}
        pt, lo, hi, dlo, dhi = T.block_boot_ratio(mm["model"], mm["scalar"], h)
        rows.append({"dataset": ds, "horizon": h, "model": headline, "rmse_scalar": float(np.sqrt(mm["scalar"].mean())), "rmse_model": float(np.sqrt(mm["model"].mean())),
                     "rmse_bolt": float(np.sqrt(mm["bolt"].mean())), "rel_vs_scalar": pt, "ci_lo": lo, "ci_hi": hi, "p_vs_scalar": T.perm_p(mm["model"], mm["scalar"], h),
                     "rmse_ctrl_a": float(np.sqrt(mm["ctrl_a"].mean())), "rmse_ctrl_b": float(np.sqrt(mm["ctrl_b"].mean()))})
        R.progress(f"a3 test {ci + 1}/{len(cells)} {ds} h{h}")
    df = pd.DataFrame(rows)
    df["p_holm"] = R.holm(df.p_vs_scalar.values)
    df["gain_vs_best_ctrl"] = 1 - df.rmse_model / np.minimum(df.rmse_ctrl_a, df.rmse_ctrl_b)
    better = int(((df.p_holm < 0.05) & (df.rel_vs_scalar < 0)).sum())
    worse1 = int((df.rel_vs_scalar > 0.01).sum())
    cl = list(zip(df.dataset, df.horizon))
    g = T.cell_cluster_boot_mean(df.gain_vs_best_ctrl.values, cl)
    g2 = T.cell_cluster_boot_mean(df.gain_vs_best_ctrl.values, cl, two_stage=True)
    summ = {"headline_model": headline, "cells_better_than_scalar_holm": better, "cells_worse_than_scalar_by_gt_1pct": worse1,
            "pooled_gain_vs_best_control": {"mean": g[0], "ci95_cell": [g[1], g[2]], "ci95_two_stage": [g2[1], g2[2]]},
            "success": bool(better >= 8 and worse1 == 0 and g[1] > 0), "pseudo": bool(pseudo)}
    df.to_csv(R.OUT / "a3_test_cells.csv", index=False)
    (R.OUT / "a3_test_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    select(a.dry_run) if a.stage == "select" else test_stage()
