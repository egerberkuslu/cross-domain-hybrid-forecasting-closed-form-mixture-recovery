"""Round 7 monitor (REPAIR_PREREG_7.md): scale-transfer-robust reference-free shift monitor on fresh drifting CESNET-TimeSeries24 subnet and IP-address traces.

Arms (r7_lib): W1 (SRC), W3 (RS-LMB), W4 (RS-LMB-RF) = the variants; STB7 = the fair baseline (same calibration, static CUSUM); V3 (round 6 recipe) and ADB (ADWIN) = report only;
RND = random control (inside evaluate_trace7).
Strata: P7_sub and P7_ip (the primary pool P7, pooled), G7 (GEANT M7a leftovers, descriptive, short design, never pooled).

  --stage degen    validation arrays only (r7_val). Applies the degenerate-expert rule (|median validation q| > log 100 -> drop, replace by the next trace of the registered order),
                   writes degen7.json and stageA_queue.json (stage-A cells still missing).
  --stage select   validation arrays only. Needs a complete degen7.json. Writes frozen_r7.json and frozen_r7.sha256. The r7_test folder is guarded and must not exist.
  --stage test     needs PREREG_7_COMMITTED, REPAIR_STAGE=test, PREREG_7_SHA256 (REPAIR_PREREG_7.md) and every frozen file hash found in the registration. Writes r7_monitor_rows.csv,
                   r7_monitor_summary.json and r7_causes.json (classify7.py).
  --dry-run        non-fresh Abilene aggregate: stage A (quarter of train), select and test code on pseudo data; prints "dry-run ok; test windows accessed: 0".
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
import r7_lib as L7  # noqa: E402  (puts rounds 3/4/6, paper_a and fresh_cells on sys.path)
import r6_lib as L6  # noqa: E402
import rcommon as R  # noqa: E402
import r7_stages as S7  # noqa: E402

A2, M3 = L6.A2, L6.M3
ARMS, VARIANTS = L7.ARMS, L7.VARIANTS
RND_SEED = 20261017
test_seed7 = lambda pi, r_: A2.test_seed(pi, r_, 7)
CODE_FILES = ("r7_lib.py", "r7_monitor.py", "r7_stages.py", "classify7.py")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
STRATUM_OF = (
    lambda name: "G7"
    if name.startswith("geant")
    else ("P7_sub" if name.startswith("cesnet_sub") else "P7_ip")
)


# ------------------------------------------------------------------ pool
def manifest():
    return json.loads((HERE / "pool7_manifest.json").read_text())


def inst_map():
    return {e["name"]: e["institution"] for e in manifest()["series"]}


def final_names():
    d = json.loads((HERE / "degen7.json").read_text())
    assert d["complete"], "degen7.json is not complete"
    return d["selected"]["sub"] + d["selected"]["ip"] + d["g7"]


def design_of(stratum):
    return "short" if stratum == "G7" else "long"


# ------------------------------------------------------------------ degenerate-expert rule
def degen_stage():
    m = manifest()
    L7.guard_test_dir(S7.TEST)
    w = L7.degen_walk(m, S7.VAL)
    g7 = m["G7_traces"]
    g7_missing = [n for n in g7 if L7.val_q_median(S7.VAL, n) is None]
    needed = list(dict.fromkeys(w["needed"] + g7_missing))
    (HERE / "stageA_queue.json").write_text(json.dumps({"needed": needed}, indent=1))
    complete = not needed
    out = {
        "complete": complete,
        "rule": "drop a primary trace iff |median validation q| > log(100); replace by the next trace of the registered order (pool7_manifest.json monitor_order); skip a trace that would break "
        "the family cap (3 per institution, subnets and IPs together). Validation forecasts only.",
        "threshold": L7.DEGEN_LOG,
        "selected": w["selected"],
        "dropped": w["dropped"],
        "skipped_cap": w["skipped_cap"],
        "median_q_val": w["median_q_val"],
        "g7": g7,
        "g7_median_q_val": {
            n: L7.val_q_median(S7.VAL, n) for n in g7 if n not in g7_missing
        },
        "g7_flagged_degenerate_kept": [
            n
            for n in g7
            if n not in g7_missing and abs(L7.val_q_median(S7.VAL, n)) > L7.DEGEN_LOG
        ],
        "n_primary": len(w["selected"]["sub"]) + len(w["selected"]["ip"]),
        "n_sub": len(w["selected"]["sub"]),
        "n_ip": len(w["selected"]["ip"]),
    }
    (HERE / "degen7.json").write_text(json.dumps(out, indent=1))
    print(
        "degen: complete",
        complete,
        "needed",
        len(needed),
        "selected",
        out["n_sub"],
        "subnets +",
        out["n_ip"],
        "IPs; dropped",
        [d["name"] for d in w["dropped"]],
        flush=True,
    )


# ------------------------------------------------------------------ selection (validation arrays only)
def select_one(job):
    name, val_dir = job
    t0 = time.time()
    y, fE, fB = M3.streams_from(M3.load_h6(Path(val_dir), name))
    t, a, r, q, e = A2.streams(y, fE, fB)
    vruns = A2.inject_streams(y, fE, fB, A2.POSITIONS_VAL, A2.INJ_VAL, A2.val_seed, 0)
    P = {arm: L7.select7(arm, q, t, e, vruns) for arm in ARMS}
    P["q_warm_test"] = [
        float(x) for x in q[-L6.WARM :]
    ]  # warm start on test: the last WARM decisions of the validation q stream
    P = {
        k: (
            {
                kk: (float(vv) if isinstance(vv, np.generic) else vv)
                for kk, vv in v.items()
            }
            if isinstance(v, dict)
            else v
        )
        for k, v in P.items()
    }
    print(
        f"select {name} n_val={len(y)} {time.time() - t0:.0f}s feasible "
        + " ".join(f"{a_}={int(P[a_].get('feasible', True))}" for a_ in ARMS),
        flush=True,
    )
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
def test_rows(name, idx, P, stratum, source, institution, loader, inj, cp_lo=None):
    y, fE, fB = loader(name)
    lo = L7.CP_LO[design_of(stratum)] if cp_lo is None else cp_lo
    rows = []
    for arm in ARMS:
        qw = np.array(P["q_warm_test"]) if arm in ("W1", "W3", "W4", "V3") else None
        ev = L7.evaluate_trace7(
            arm,
            P[arm],
            y,
            fE,
            fB,
            inj,
            test_seed7,
            q_warm=qw,
            rng_ctrl_seed=RND_SEED + idx,
            cp_lo=lo,
        )
        rows.append(
            {
                "series": name,
                "stratum": stratum,
                "source": source,
                "institution": institution,
                "arm": arm,
                "feasible": bool(P[arm].get("feasible", True)),
                "n_windows": int(len(y)),
                **{k: v for k, v in ev.items() if k != "det_by_cp"},
                "det_by_cp": json.dumps(ev["det_by_cp"]),
            }
        )
    return rows


_VIEW = None


def _test_loader(name):
    global _VIEW
    if _VIEW is None:
        _VIEW = L7.TestView(S7.TEST)
    return _VIEW.load(name)


def test_job(job):
    rows = test_rows(*job[:6], _test_loader, *job[6:])
    print(
        "test",
        job[0],
        "joint W1/W3/W4/STB7:",
        {
            r["arm"]: int(r["joint"])
            for r in rows
            if r["arm"] in ("W1", "W3", "W4", "STB7")
        },
        flush=True,
    )
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
    return {
        c: {
            "pass": int(g[c].sum()),
            "n": int(len(g)),
            "wilson95": wilson(int(g[c].sum()), len(g)),
        }
        for c in ("c1", "c2", "c3", "joint")
    }


def summarize(df, pseudo):
    """Success rule of REPAIR_PREREG_7.md section 6 on the pooled primary pool P7 (strata P7_sub and P7_ip); G7 is descriptive and never pooled."""
    out = {"pseudo": pseudo}
    p7 = df[df.stratum != "G7"]
    n = int(p7.series.nunique())
    need = ceil(0.8 * n)
    arms = {}
    for arm in ARMS:
        g = p7[p7.arm == arm]
        X = int(g.joint.sum())
        arms[arm] = {
            "X": X,
            "n": n,
            "need": need,
            "p_binom_vs_0.5": L6.binom_sf(X, n, 0.5) if n else float("nan"),
            "rand_ctrl_c3_share_mean": float(g.rand_ctrl_c3only_share.mean()),
            "rand_ctrl_joint_share_mean": float(g.rand_ctrl_pass_share.mean()),
            "feasible": int(g.feasible.sum()),
            "mean_det": float(g.det.mean()),
            "mean_null_dense": float(g.null_dense.mean()),
            "mean_p_rand": float(g.p_rand.mean()),
            "conditions": counts(g),
            "by_level": {s: counts(gg) for s, gg in g.groupby("stratum")},
            "by_family": {
                str(k): {"traces": int(len(gg)), "joint": int(gg.joint.sum())}
                for k, gg in g.groupby("institution")
            }
            if "institution" in g
            else {},
        }
    padj = (
        dict(zip(VARIANTS, holm([arms[v]["p_binom_vs_0.5"] for v in VARIANTS])))
        if n
        else {}
    )
    ok = {}
    for v in VARIANTS:
        arms[v]["p_holm"] = padj.get(v)
        ok[v] = bool(
            arms[v]["X"] >= need
            and padj.get(v, 1) < 0.05
            and arms[v]["rand_ctrl_c3_share_mean"] <= 0.05
        )
        arms[v]["success"] = ok[v]
    winners = [v for v in VARIANTS if ok[v]]
    out["P7"] = {
        "traces": n,
        "need": need,
        "arms": arms,
        "success": bool(winners),
        "succeeding_variants": winners,
        "headline": (
            max(winners, key=lambda v: (arms[v]["X"], -VARIANTS.index(v)))
            if winners
            else None
        ),
    }
    # secondary: exact one-sided McNemar of each succeeding variant against STB7 (Holm across the succeeding variants), not a success condition
    st = p7[p7.arm == "STB7"].set_index("series").joint
    mc = {}
    for v in winners:
        j = p7[p7.arm == v].set_index("series").joint
        b, c = int((j & ~st[j.index]).sum()), int((~j & st[j.index]).sum())
        mc[v] = {
            "v_only": b,
            "stb7_only": c,
            "p_one_sided": L6.binom_sf(b, b + c, 0.5) if b + c else 1.0,
        }
    for v, a_ in zip(
        winners, holm([mc[v]["p_one_sided"] for v in winners]) if winners else []
    ):
        mc[v]["p_holm"] = a_
        mc[v]["claim"] = bool(a_ < 0.05)
    out["P7"]["mcnemar_vs_STB7"] = mc
    g7 = df[df.stratum == "G7"]
    out["G7_geant_descriptive"] = (
        {
            "traces": int(g7.series.nunique()),
            "pooled_into_success_rule": False,
            "arms": {
                arm: {
                    "X": int(g.joint.sum()),
                    "n": int(len(g)),
                    "conditions": counts(g),
                    "rand_ctrl_c3_share_mean": float(g.rand_ctrl_c3only_share.mean()),
                }
                for arm, g in g7.groupby("arm")
            },
        }
        if len(g7)
        else {}
    )
    return out


# ------------------------------------------------------------------ main
def freeze_record(names, params):
    m = manifest()
    return {
        "stage": "select",
        "dry_run": False,
        "traces": names,
        "strata": {n: STRATUM_OF(n) for n in names},
        "params": params,
        "constants": {
            "L_CAP": L6.L_CAP,
            "NULL_TARGET": L6.NULL_TARGET,
            "FA_TARGET": L6.FA_TARGET,
            "MARGIN": L6.MARGIN,
            "BURN": L6.BURN,
            "WARM": L6.WARM,
            "CLOCK_STEP": L6.CLOCK_STEP,
            "LEVELS": [float(L6.LEVELS[0]), float(L6.LEVELS[-1]), 0.5],
            "CP_LO": L6.CP_LO,
            "RND_SEED": RND_SEED,
            "test_seed_s": 7,
            "inj": A2.INJ_TEST,
            "DEGEN_LOG": L7.DEGEN_LOG,
        },
        "pool7_manifest_sha256": sha(HERE / "pool7_manifest.json"),
        "degen7_sha256": sha(HERE / "degen7.json"),
        "code_sha256": {f: sha(HERE / f) for f in CODE_FILES},
        "n_primary": sum(1 for n in names if STRATUM_OF(n) != "G7"),
    }


def verify_before_test():
    """Hash chain of the test stage: PREREG_7_SHA256 -> REPAIR_PREREG_7.md -> every frozen file hash that the registration lists."""
    if (
        not (HERE / "PREREG_7_COMMITTED").exists()
        or os.environ.get("REPAIR_STAGE") != "test"
    ):
        raise SystemExit("test stage needs PREREG_7_COMMITTED and REPAIR_STAGE=test")
    prereg = HERE / "REPAIR_PREREG_7.md"
    assert (
        sha(prereg) == (HERE / "PREREG_7_SHA256").read_text().split()[0]
    ), "REPAIR_PREREG_7.md changed"
    text = prereg.read_text()
    p = HERE / "frozen_r7.json"
    assert (
        sha(p) == (HERE / "frozen_r7.sha256").read_text().strip()
    ), "frozen_r7.json changed"
    fr = json.loads(p.read_text())
    assert fr["pool7_manifest_sha256"] == sha(
        HERE / "pool7_manifest.json"
    ), "pool7_manifest.json changed"
    assert fr["degen7_sha256"] == sha(HERE / "degen7.json"), "degen7.json changed"
    for f, h in fr["code_sha256"].items():
        assert sha(HERE / f) == h, f"{f} changed since the freeze"
    for f in (
        "frozen_r7.json",
        "pool7_manifest.json",
        "degen7.json",
        "RECIPE7_FROZEN.json",
        *CODE_FILES,
    ):
        assert sha(HERE / f) in text, f"{f}: sha256 not listed in REPAIR_PREREG_7.md"
    rec = json.loads((HERE / "RECIPE7_FROZEN.json").read_text())
    for f, h in {
        **rec["unchanged_from_earlier_rounds"],
        **{k: v for k, v in rec["round7_code"].items()},
    }.items():
        base = HERE.parent if f in rec["unchanged_from_earlier_rounds"] else HERE
        assert sha(base / f) == h, f"{f} differs from RECIPE7_FROZEN.json"
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["degen", "select", "test"], required=True)
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

        L7.guard_test_dir(S7.TEST)
        tmp = Path(tempfile.mkdtemp(dir=str(HERE)))
        rv = F.Restricted(load_preprocessed("abilene"))
        o = S7.S3.stage_a("abilene", 6, rv, out_dir=tmp, save_models=False, dry=True)
        np.savez(tmp / "abilene__h6.npz", **o)
        pars = select_all(["abilene"], tmp, workers=1)
        n_stream = len(M3.streams_from(M3.load_h6(tmp, "abilene"))[0])
        cp_lo = 420 if n_stream >= 690 else 200
        loader = lambda n: M3.streams_from(M3.load_h6(tmp, n))
        rows = test_rows(
            "abilene",
            0,
            pars["abilene"],
            "P7_sub",
            "abilene",
            0,
            loader,
            inj=2,
            cp_lo=cp_lo,
        )
        df = pd.DataFrame(rows)
        s = summarize(df, True)
        print(
            json.dumps(
                {
                    "P7": {k: v for k, v in s["P7"].items() if k != "arms"},
                    "arm_X": {k: v["X"] for k, v in s["P7"]["arms"].items()},
                },
                indent=1,
            )
        )
        print(
            df[
                [
                    "arm",
                    "feasible",
                    "c1",
                    "c2",
                    "c3",
                    "joint",
                    "det",
                    "ctl",
                    "null_dense",
                    "p_rand",
                    "rand_ctrl_c3only_share",
                ]
            ].to_string()
        )
        try:
            L7.TestView(S7.TEST)
            raise AssertionError("TestView opened without the commit")
        except PermissionError:
            print(
                "Restricted test view refuses without PREREG_7_COMMITTED and REPAIR_STAGE=test: ok"
            )
        try:
            rv.windows(6, "test")
            raise AssertionError("Restricted view did not raise on a test window")
        except PermissionError:
            print("Restricted view raises on test windows: ok")
        assert not any(sp == "test" for _, sp in rv.accessed)
        assert not S7.TEST.exists(), "r7_test exists"
        shutil.rmtree(tmp)
        print("dry-run ok; test windows accessed: 0")
        return
    if a.stage == "degen":
        degen_stage()
        return
    if a.stage == "select":
        L7.guard_test_dir(S7.TEST)
        names = final_names()
        params = select_all(names, S7.VAL, a.workers)
        rec = freeze_record(names, params)
        p = HERE / "frozen_r7.json"
        p.write_text(json.dumps(rec, indent=1, default=float))
        (HERE / "frozen_r7.sha256").write_text(sha(p) + "\n")
        print(
            "frozen round 7 monitor parameters for",
            len(names),
            "traces; sha256",
            sha(p),
        )
        return
    fr = verify_before_test()
    names, strat = fr["traces"], fr["strata"]
    inst = inst_map()
    R.allow_test()
    jobs = [
        (
            n,
            i,
            fr["params"][n],
            strat[n],
            "geant" if strat[n] == "G7" else "cesnet",
            inst.get(n, -1),
            A2.INJ_TEST,
        )
        for i, n in enumerate(names)
    ]
    with Pool(a.workers) as pl:
        res = pl.map(test_job, jobs, chunksize=1)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_csv(HERE / "r7_monitor_rows.csv", index=False)
    s = summarize(df, False)
    s["frozen_r7_sha256"] = sha(HERE / "frozen_r7.json")
    (HERE / "r7_monitor_summary.json").write_text(
        json.dumps(s, indent=1, default=float)
    )
    import classify7

    classify7.main(HERE / "r7_monitor_rows.csv")
    print(
        json.dumps(
            {
                "P7": {k: v for k, v in s["P7"].items() if k != "arms"},
                "arm_X": {k: v["X"] for k, v in s["P7"]["arms"].items()},
            },
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
