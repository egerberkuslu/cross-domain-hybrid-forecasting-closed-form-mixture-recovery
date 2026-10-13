"""Round 6 DRAFT feasibility on VALIDATION streams (and a labelled re-analysis of the revealed round 5 test streams).

Part 1 (design input, validation only): the 64 round 5 traces (round5/r5_val) and the 12 round 4 traces (round4/r4_val). Each arm is selected on the
validation stream (r6_lib.select_*), then evaluated IN-SAMPLE on the same validation stream with the round 6 injection design and seeds that differ from
the selection seeds. In-sample calibration makes the null rates optimistic; the pass probabilities are design inputs for the power calculation only.

Part 2 (revealed, report only, NOT used for any choice): the val-selected parameters scored on the round 5 TEST streams that r5_monitor.py already read
(r5_monitor_summary.json is public). The round 5 traces are not round 6 test traces. Its numbers are labelled 'revealed_r5_test' everywhere.

  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python val6.py
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r6_lib as L6  # noqa: E402

A2, M3 = L6.A2, L6.M3
R4, R5 = HERE.parent / "round4", HERE.parent / "round5"
ARMS = ("R5", "V1", "V2", "V3", "STB", "ADB")
seed_val6 = lambda pi, r_: 7_000_019 * (pi + 1) + 7919 * r_ + 3
seed_test = lambda pi, r_: A2.test_seed(pi, r_, 0)


def one(job):
    name, rnd = job
    rdir = R5 if rnd == 5 else R4
    vdir = rdir / ("r5_val" if rnd == 5 else "r4_val")
    frozen = json.loads(
        (
            rdir / ("frozen_r5_monitor.json" if rnd == 5 else "frozen_r4_monitor.json")
        ).read_text()
    )["params"][name]
    y, fE, fB = M3.streams_from(M3.load_h6(vdir, name))
    t, a, r, q, e = A2.streams(y, fE, fB)
    vruns = A2.inject_streams(y, fE, fB, A2.POSITIONS_VAL, A2.INJ_VAL, A2.val_seed, 0)
    P = {"R5": frozen["RF"]}
    for arm in ("V1", "V2", "V3"):
        P[arm] = L6.select_rf(arm, q, t, e, vruns)
    P["STB"] = L6.select_stb(q, t)
    P["ADB"] = L6.select_adb(e, t)
    rows = []
    for arm in ARMS:
        ev = L6.evaluate_trace(
            arm,
            P[arm],
            y,
            fE,
            fB,
            10,
            seed_val6,
            q_warm=q[: L6.WARM] if arm == "V3" else None,
            rng_ctrl_seed=11,
        )
        rows.append(
            {
                "series": name,
                "round": rnd,
                "split": "val_insample",
                "arm": arm,
                "feasible": bool(P[arm].get("feasible", True)),
                **{k: v for k, v in ev.items() if k != "det_by_cp"},
                "det_by_cp": json.dumps(ev["det_by_cp"]),
            }
        )
    if rnd == 5:
        yt, fEt, fBt = M3.streams_from(M3.load_h6(R5 / "r5_test", name))
        for arm in ARMS:
            ev = L6.evaluate_trace(
                arm,
                P[arm],
                yt,
                fEt,
                fBt,
                10,
                seed_test,
                q_warm=q[-L6.WARM :] if arm == "V3" else None,
                rng_ctrl_seed=12,
            )
            rows.append(
                {
                    "series": name,
                    "round": rnd,
                    "split": "revealed_r5_test",
                    "arm": arm,
                    "feasible": bool(P[arm].get("feasible", True)),
                    **{k: v for k, v in ev.items() if k != "det_by_cp"},
                    "det_by_cp": json.dumps(ev["det_by_cp"]),
                }
            )
    return rows, {
        name: {
            k: {
                kk: (vv if not isinstance(vv, np.generic) else float(vv))
                for kk, vv in v.items()
            }
            for k, v in P.items()
        }
    }


def main():
    n5 = json.loads((HERE.parent / "fresh_cells" / "pool5_manifest.json").read_text())[
        "monitor_traces"
    ]
    n4 = json.loads((HERE.parent / "fresh_cells" / "pool4_manifest.json").read_text())[
        "monitor_traces"
    ]
    jobs = [(n, 5) for n in n5] + [(n, 4) for n in n4]
    with Pool(4) as pool:
        res = pool.map(one, jobs, chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    params = {}
    for _, p in res:
        params.update(p)
    df = pd.DataFrame(rows)
    df.to_csv(HERE / "val6_rows.csv", index=False)
    (HERE / "val6_params.json").write_text(json.dumps(params, indent=1, default=float))
    summ = {}
    for (split, rnd, arm), g in df.groupby(["split", "round", "arm"]):
        summ[f"{split}|r{rnd}|{arm}"] = {
            "traces": int(len(g)),
            "feasible": int(g.feasible.sum()),
            "c1": int(g.c1.sum()),
            "c2": int(g.c2.sum()),
            "c3": int(g.c3.sum()), "c3_binom": int(g.c3_binom.sum()),
            "joint": int(g.joint.sum()),
            "mean_det": float(g.det.mean()),
            "mean_ctl": float(g.ctl.mean()),
            "mean_null_dense": float(g.null_dense.mean()),
            "mean_p_rand": float(g.p_rand.mean()),
            "traces_null_dense_gt_0.10": int((g.null_dense > 0.10).sum()),
            "mean_events_nofault": float(g.n_events_nofault.mean()),
            "rand_ctrl_c3only_share_mean": float(g.rand_ctrl_c3only_share.mean()),
            "rand_ctrl_det_mean": float(g.rand_ctrl_det_mean.mean()), "rand_ctrl_binom_share_mean": float(g.rand_ctrl_binom_share.mean()),
            "rand_ctrl_joint_share_mean": float(g.rand_ctrl_pass_share.mean()),
        }
    (HERE / "val6_summary.json").write_text(json.dumps(summ, indent=1))
    for k, v in summ.items():
        print(k, v)


if __name__ == "__main__":
    main()
