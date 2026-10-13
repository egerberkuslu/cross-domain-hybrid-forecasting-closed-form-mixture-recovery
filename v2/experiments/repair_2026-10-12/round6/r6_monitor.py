"""Round 6 monitor (REPAIR_PREREG_6.md): placebo-calibrated, onset-localised reference-free shift monitor on fresh drifting traces.

Arms (r6_lib): R5 (frozen round 5 RF recipe, report only), V1, V2, V3 (variants), STB (fair baseline, static CUSUM), ADB (ADWIN comparator, report only), RND (random control, inside evaluate_trace).
Strata: P2 (primary pool: P2_new + P2_FG, CESNET and Abilene, long design) and P3 (GEANT, descriptive, short design, never pooled).

  --stage select   validation arrays only (r6_val). Writes frozen_r6.json and frozen_r6.sha256. The r6_test folder is guarded and cannot be read.
  --stage test     needs PREREG_6_COMMITTED, REPAIR_STAGE=test, the frozen hash and the code hashes recorded at freeze time. Writes r6_monitor_rows.csv, r6_monitor_summary.json.
  --dry-run        non-fresh Abilene aggregate: stage A (quarter of train), select and test code on pseudo data; prints "dry-run ok; test windows accessed: 0".
Progress: per trace on stdout (the run script adds a 5-minute heartbeat to PROGRESS.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from math import ceil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_lib as L6  # noqa: E402  (puts rounds 3/4, paper_a and fresh_cells on sys.path)
import rcommon as R  # noqa: E402
import r6_stages as S6  # noqa: E402

A2, M3 = L6.A2, L6.M3
ARMS = ("R5", "V1", "V2", "V3", "STB", "ADB")
VARIANTS = ("V1", "V2", "V3")
RND_SEED = 20261016
test_seed6 = lambda pi, r_: A2.test_seed(pi, r_, 6)
CODE_FILES = ("r6_lib.py", "r6_monitor.py", "r6_stages.py")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ pool
def pool():
    m = json.loads((HERE / "pool6_manifest.json").read_text())
    strat = {e["name"]: e["stratum"] for e in m["series"]}
    src = {e["name"]: e["source"] for e in m["series"]}
    names = m["monitor_traces_P2"] + m["monitor_traces_P3_geant"]
    return names, strat, src


def design_of(stratum):
    return "short" if stratum == "P3" else "long"


# ------------------------------------------------------------------ selection (validation arrays only)
def select_one(job):
    name, val_dir = job
    t0 = time.time()
    y, fE, fB = M3.streams_from(M3.load_h6(Path(val_dir), name))
    t, a, r, q, e = A2.streams(y, fE, fB)
    vruns = A2.inject_streams(y, fE, fB, A2.POSITIONS_VAL, A2.INJ_VAL, A2.val_seed, 0)
    P = {"R5": M3.select([name], Path(val_dir), None, dry=True)["params"][name]["RF"]}
    for arm in VARIANTS:
        P[arm] = L6.select_rf(arm, q, t, e, vruns)
    P["STB"] = L6.select_stb(q, t, vruns)
    P["ADB"] = L6.select_adb(e, t)
    P["q_warm_test"] = [float(x) for x in q[-L6.WARM :]]  # V3 on test: the last WARM decisions of the validation q stream
    P = {k: ({kk: (float(vv) if isinstance(vv, np.generic) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v) for k, v in P.items()}
    print(f"select {name} n_val={len(y)} {time.time() - t0:.0f}s feasible " + " ".join(f"{a_}={int(P[a_].get('feasible', True))}" for a_ in ARMS), flush=True)
    return name, P


def select_all(names, val_dir, workers=2):
    jobs = [(n, str(val_dir)) for n in names]
    if workers > 1:
        with Pool(workers) as pl:
            res = pl.map(select_one, jobs, chunksize=1)
    else:
        res = [select_one(j) for j in jobs]
    return dict(res)


# ------------------------------------------------------------------ test rows
def test_rows(name, idx, P, stratum, source, test_dir, inj, cp_lo=None):
    y, fE, fB = M3.streams_from(M3.load_h6(Path(test_dir), name))
    lo = L6.CP_LO[design_of(stratum)] if cp_lo is None else cp_lo
    rows = []
    for arm in ARMS:
        qw = np.array(P["q_warm_test"]) if arm == "V3" else None
        ev = L6.evaluate_trace(arm, P[arm], y, fE, fB, inj, test_seed6, q_warm=qw, rng_ctrl_seed=RND_SEED + idx, cp_lo=lo)
        rows.append({"series": name, "stratum": stratum, "source": source, "arm": arm, "feasible": bool(P[arm].get("feasible", True)), "n_windows": int(len(y)),
                     **{k: v for k, v in ev.items() if k != "det_by_cp"}, "det_by_cp": json.dumps(ev["det_by_cp"])})
    return rows


def test_job(job):
    rows = test_rows(*job)
    print("test", job[0], "joint V1/V2/V3/STB:", {r["arm"]: int(r["joint"]) for r in rows if r["arm"] in ("V1", "V2", "V3", "STB")}, flush=True)
    return rows


# ------------------------------------------------------------------ statistics
def wilson(x, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - w), float(c + w)]


def holm(ps):
    """Holm step-down adjusted p-values (monotone), order preserved."""
    m = len(ps)
    order = np.argsort(ps)
    adj, run = np.zeros(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i])
        adj[i] = min(1.0, run)
    return [float(x) for x in adj]


def counts(g):
    return {c: {"pass": int(g[c].sum()), "n": int(len(g)), "wilson95": wilson(int(g[c].sum()), len(g))} for c in ("c1", "c2", "c3", "joint")}


def summarize(df, pseudo):
    out = {"pseudo": pseudo}
    p2 = df[df.stratum != "P3"]
    n = int(p2.series.nunique())
    need = ceil(0.8 * n)
    arms = {}
    for arm in ARMS:
        g = p2[p2.arm == arm]
        X = int(g.joint.sum())
        arms[arm] = {"X": X, "n": n, "need": need, "p_binom_vs_0.5": L6.binom_sf(X, n, 0.5) if n else float("nan"),
                     "rand_ctrl_c3_share_mean": float(g.rand_ctrl_c3only_share.mean()), "rand_ctrl_joint_share_mean": float(g.rand_ctrl_pass_share.mean()),
                     "feasible": int(g.feasible.sum()), "mean_det": float(g.det.mean()), "mean_null_dense": float(g.null_dense.mean()), "mean_p_rand": float(g.p_rand.mean()),
                     "conditions": counts(g),
                     "by_source": {s: counts(gg) for s, gg in g.groupby("source")}, "by_stratum": {s: counts(gg) for s, gg in g.groupby("stratum")},
                     "descriptive_round5_rule_events": {"c3_round5_rule": int((g.c3_round5_rule_events & g.c1).sum()), "n": int(len(g)), "note": "c1 and (c2 and placebo-clock c3) with events, cap 100, descriptive"}}
    padj = dict(zip(VARIANTS, holm([arms[v]["p_binom_vs_0.5"] for v in VARIANTS]))) if n else {}
    ok = {}
    for v in VARIANTS:
        arms[v]["p_holm"] = padj.get(v)
        ok[v] = bool(arms[v]["X"] >= need and padj.get(v, 1) < 0.05 and arms[v]["rand_ctrl_c3_share_mean"] <= 0.05)
        arms[v]["success"] = ok[v]
    winners = [v for v in VARIANTS if ok[v]]
    out["P2"] = {"traces": n, "need": need, "arms": arms, "success": bool(winners), "succeeding_variants": winners,
                 "headline": (max(winners, key=lambda v: (arms[v]["X"], -VARIANTS.index(v))) if winners else None)}
    # secondary: exact one-sided McNemar of each succeeding variant against STB (Holm across the succeeding variants), underpowered, not a success condition
    st = p2[p2.arm == "STB"].set_index("series").joint
    mc = {}
    for v in winners:
        j = p2[p2.arm == v].set_index("series").joint
        b, c = int((j & ~st[j.index]).sum()), int((~j & st[j.index]).sum())
        mc[v] = {"v_only": b, "stb_only": c, "p_one_sided": L6.binom_sf(b, b + c, 0.5) if b + c else 1.0}
    for v, a_ in zip(winners, holm([mc[v]["p_one_sided"] for v in winners]) if winners else []):
        mc[v]["p_holm"] = a_
        mc[v]["claim"] = bool(a_ < 0.05)
    out["P2"]["mcnemar_vs_STB"] = mc
    p3 = df[df.stratum == "P3"]
    out["P3_geant_descriptive"] = {"traces": int(p3.series.nunique()), "arms": {arm: {"X": int(g.joint.sum()), "n": int(len(g)), "conditions": counts(g), "rand_ctrl_c3_share_mean": float(g.rand_ctrl_c3only_share.mean())}
                                                                 for arm, g in p3.groupby("arm")}} if len(p3) else {}
    return out


# ------------------------------------------------------------------ main
def freeze_record(names, strat, params):
    return {"stage": "select", "dry_run": False, "traces": names, "strata": strat, "params": params,
            "constants": {"L_CAP": L6.L_CAP, "NULL_TARGET": L6.NULL_TARGET, "FA_TARGET": L6.FA_TARGET, "MARGIN": L6.MARGIN, "BURN": L6.BURN, "WARM": L6.WARM,
                          "CLOCK_STEP": L6.CLOCK_STEP, "LEVELS": [float(L6.LEVELS[0]), float(L6.LEVELS[-1]), 0.5], "CP_LO": L6.CP_LO, "RND_SEED": RND_SEED, "test_seed_s": 6, "inj": A2.INJ_TEST},
            "pool6_manifest_sha256": sha(HERE / "pool6_manifest.json"), "code_sha256": {f: sha(HERE / f) for f in CODE_FILES}}


def guard_test_dir():
    """Select / dry-run: any read under r6_test raises (extends the rcommon guard)."""
    R._GUARDED = tuple(R._GUARDED) + (str(S6.TEST),)
    try:
        np.load(S6.TEST / "any.npz")
        raise AssertionError("r6_test guard did not bite")
    except R.TestAccessError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "test"], required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    R.install_guard()
    R.guard_selftest()
    if a.dry_run:
        import shutil
        import tempfile
        from src.preprocessing import load_preprocessed
        import fresh_cells as F
        guard_test_dir()
        tmp = Path(tempfile.mkdtemp(dir=str(HERE)))
        rv = F.Restricted(load_preprocessed("abilene"))
        o = S6.S3.stage_a("abilene", 6, rv, out_dir=tmp, save_models=False, dry=True)
        np.savez(tmp / "abilene__h6.npz", **o)
        pars = select_all(["abilene"], tmp, workers=1)
        n_stream = len(M3.streams_from(M3.load_h6(tmp, "abilene"))[0])
        cp_lo = 420 if n_stream >= 690 else 200
        rows = test_rows("abilene", 0, pars["abilene"], "P2_new", "abilene", tmp, inj=2, cp_lo=cp_lo)
        df = pd.DataFrame(rows)
        s = summarize(df, True)
        print(json.dumps({"P2": {k: v for k, v in s["P2"].items() if k != "arms"}, "arm_X": {k: v["X"] for k, v in s["P2"]["arms"].items()}}, indent=1))
        print(df[["arm", "feasible", "c1", "c2", "c3", "joint", "det", "ctl", "null_dense", "p_rand", "rand_ctrl_c3only_share"]].to_string())
        try:
            rv.windows(6, "test")
            raise AssertionError("Restricted view did not raise on a test window")
        except PermissionError:
            print("Restricted view raises on test windows: ok")
        assert not any(sp == "test" for _, sp in rv.accessed)
        shutil.rmtree(tmp)
        print("dry-run ok; test windows accessed: 0")
        return
    names, strat, src = pool()
    if a.stage == "select":
        guard_test_dir()
        params = select_all(names, S6.VAL, a.workers)
        rec = freeze_record(names, strat, params)
        p = HERE / "frozen_r6.json"
        p.write_text(json.dumps(rec, indent=1, default=float))
        (HERE / "frozen_r6.sha256").write_text(sha(p) + "\n")
        print("frozen round 6 monitor parameters for", len(names), "traces; sha256", sha(p))
        return
    if not (HERE / "PREREG_6_COMMITTED").exists() or os.environ.get("REPAIR_STAGE") != "test":
        raise SystemExit("test stage needs PREREG_6_COMMITTED and REPAIR_STAGE=test")
    p = HERE / "frozen_r6.json"
    assert sha(p) == (HERE / "frozen_r6.sha256").read_text().strip(), "frozen_r6.json changed"
    fr = json.loads(p.read_text())
    assert fr["pool6_manifest_sha256"] == sha(HERE / "pool6_manifest.json"), "pool6_manifest.json changed"
    for f, h in fr["code_sha256"].items():
        assert sha(HERE / f) == h, f"{f} changed since the freeze"
    assert fr["traces"] == names
    R.allow_test()
    jobs = [(n, i, fr["params"][n], strat[n], src[n], str(S6.TEST), A2.INJ_TEST) for i, n in enumerate(names)]
    with Pool(a.workers) as pl:
        res = pl.map(test_job, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(HERE / "r6_monitor_rows.csv", index=False)
    s = summarize(df, False)
    s["frozen_r6_sha256"] = sha(p)
    (HERE / "r6_monitor_summary.json").write_text(json.dumps(s, indent=1, default=float))
    print(json.dumps({"P2": {k: v for k, v in s["P2"].items() if k != "arms"}, "arm_X": {k: v["X"] for k, v in s["P2"]["arms"].items()}}, indent=1, default=float))


if __name__ == "__main__":
    main()
