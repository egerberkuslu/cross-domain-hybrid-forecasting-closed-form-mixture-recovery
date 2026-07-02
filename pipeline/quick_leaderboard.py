"""Quick leaderboard: where does CHA-Hybrid-v2 rank vs baselines per (dataset, horizon)?

Reads ``results/metrics/*.json`` and prints:
  1. per (dataset, horizon) sorted leaderboard, marking cha_hybrid_v2 and cha_hybrid
  2. per (dataset, horizon) rank of cha_hybrid_v2
  3. CHA-v2 vs TimesFM head-to-head (since v2 uses TimesFM as the global expert)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_runs():
    rows = []
    for fn in sorted(Path("outputs/metrics").glob("*.json")):
        try:
            d = json.loads(fn.read_text())
        except Exception:
            continue
        if d.get("status") != "ok":
            continue
        ms = d.get("metrics_scaled", {})
        mn = d.get("metrics_native", {})
        rows.append(
            dict(
                dataset=d["dataset"],
                variant=d["variant"],
                horizon=int(d["horizon"]),
                seed=int(d["seed"]),
                rmse_scaled=float(ms.get("rmse", float("nan"))),
                rmse_native=float(mn.get("rmse", float("nan"))),
            )
        )
    return rows


def main():
    rows = load_runs()
    if not rows:
        print("no runs found")
        return

    # mean per (dataset, variant, horizon)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["dataset"], r["variant"], r["horizon"])].append(r["rmse_scaled"])
    means = {k: float(np.mean(v)) for k, v in grouped.items()}
    counts = {k: len(v) for k, v in grouped.items()}

    datasets = sorted({k[0] for k in means.keys()})
    horizons = sorted({k[2] for k in means.keys()})

    print("=" * 110)
    print(
        "PER (dataset, horizon) LEADERBOARD — rmse_scaled (mean over seeds), top 5 + CHA variants"
    )
    print("=" * 110)
    for ds in datasets:
        for h in horizons:
            scored = [
                (v, means[(ds, vname, h)], counts[(ds, vname, h)])
                for (d, vname, hh) in means.keys()
                if d == ds and hh == h
                for v in [vname]
            ]
            scored = sorted(
                set([(v, m, c) for (v, m, c) in scored]), key=lambda x: x[1]
            )
            print(f"\n--- {ds} h={h} ---")
            for rank, (v, m, c) in enumerate(scored, 1):
                mark = ""
                if v == "cha_hybrid_v2":
                    mark = "  ◀ proposed-v2"
                elif v == "cha_hybrid":
                    mark = "  ◀ proposed-v1"
                # show top 5 + CHA variants always
                if rank <= 5 or v.startswith("cha_hybrid"):
                    print(f"  {rank:2d}. {v:<28s} {m:.4f}   n_seeds={c}{mark}")

    # Rank summary for v2
    print()
    print("=" * 110)
    print("CHA-Hybrid v2 RANK SUMMARY")
    print("=" * 110)
    print(
        f"{'dataset':<14} {'h':>3} {'rank':>5} / total   rmse_v2     winner               winner_rmse"
    )
    for ds in datasets:
        for h in horizons:
            scored = sorted(
                set(
                    [
                        (vname, means[k])
                        for k in means.keys()
                        if k[0] == ds and k[2] == h
                        for vname in [k[1]]
                    ]
                ),
                key=lambda x: x[1],
            )
            for rank, (v, m) in enumerate(scored, 1):
                if v == "cha_hybrid_v2":
                    winner = scored[0]
                    print(
                        f"{ds:<14} {h:>3} {rank:>5} / {len(scored):<6} {m:.4f}     "
                        f"{winner[0]:<20} {winner[1]:.4f}"
                    )
                    break
            else:
                print(f"{ds:<14} {h:>3}  (cha_hybrid_v2 not found)")

    # v2 vs TimesFM head-to-head
    print()
    print("=" * 110)
    print("CHA-Hybrid v2 vs TimesFM zero-shot head-to-head")
    print("=" * 110)
    print(
        f"{'dataset':<14} {'h':>3}  {'cha_v2':>10} {'timesfm_zs':>12} {'Δ(v2-tfm)':>12}  winner"
    )
    for ds in datasets:
        for h in horizons:
            v2 = means.get((ds, "cha_hybrid_v2", h))
            tf = means.get((ds, "timesfm_zs", h))
            if v2 is None or tf is None:
                continue
            delta = v2 - tf
            winner = "v2" if v2 < tf else ("tfm" if tf < v2 else "tie")
            print(f"{ds:<14} {h:>3}  {v2:>10.4f} {tf:>12.4f} {delta:>+12.4f}  {winner}")


if __name__ == "__main__":
    main()
