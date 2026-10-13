"""Round 6 pool builder (REPAIR_PREREG_6.md section 1). Deterministic, data-only, no forecast, no accuracy, no test value read (test windows are only counted).

Candidate rules (fixed before any monitor score of these traces):
  P2_new   never allocated CESNET institutions (missing share <= 5 %) and Abilene OD pairs (zero share <= 5 %, missing share <= 16 %), median > 0
  P2_FG    the CESNET and Abilene series of reserved_series.json round5_FG FG_half (admissible under the original rule)
  P3       never allocated GEANT OD pairs, missing share <= 7 %, zero share <= 5 %, median > 0   (descriptive stratum, never pooled)
Every candidate is built with fresh_cells.build_series, the drift screen is recomputed with build_pool4.shift (train and validation only) and a series is a
monitor trace iff shift >= 0.3 train sd and n_test_windows_h6 >= 690 (CESNET, Abilene) or >= 420 (GEANT).
Writes round6/pool6_manifest.json and round6/reserved_round6_M.json (the additive block 'round6_M' for fresh_cells/reserved_series.json, merged by the coordinator).
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python build_pool6.py
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
FC = HERE.parent / "fresh_cells"
sys.path.insert(0, str(FC))
import fresh_cells as F  # noqa: E402
from build_pool4 import shift  # noqa: E402

CAST = {"cesnet": int, "abilene": int, "geant": str}
MISS_MAX = {"cesnet": 0.05, "abilene": 0.16, "geant": 0.07}
ZERO_MAX = 0.05
DRIFT_SD, MIN_WIN = 0.3, {"cesnet": 690, "abilene": 690, "geant": 420}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def allocated(src):
    R = json.loads((FC / "reserved_series.json").read_text())
    P = json.loads((FC / "partition_F.json").read_text())
    return {str(x) for blk in (R["reserved"][src], P["pilot"][src], P["F_registered"][src], R["round4_A"]["A_half"][src], R["round5_FG"]["FG_half"][src]) for x in blk}


def candidates():
    R = json.loads((FC / "reserved_series.json").read_text())
    out = []
    for src in ("cesnet", "abilene", "geant"):
        t = F.admissibility_table(src)
        t["sid"] = t.id.astype(str)
        al = allocated(src)
        for r in t.itertuples():
            if r.sid in al:
                continue
            if r.nan_share <= MISS_MAX[src] and r.zero_share <= ZERO_MAX and r.median > 0:
                out.append((src, r.sid, "P3" if src == "geant" else "P2_new", float(r.nan_share), float(r.zero_share)))
        if src != "geant":
            nz = {str(r.id): r for r in t.itertuples()}
            for x in R["round5_FG"]["FG_half"][src]:
                r = nz[str(x)]
                out.append((src, str(x), "P2_FG", float(r.nan_share), float(r.zero_share)))
    return out


def main():
    cand = candidates()
    print("candidates", len(cand), flush=True)
    series, excluded = [], []
    for src, sid, stratum, nan_share, zero_share in sorted(cand, key=lambda c: (c[0], c[2], int(c[1]) if c[0] != "geant" else 0, c[1])):
        sid_c = CAST[src](sid)
        name = F.series_name(src, sid_c)
        if not (F.SERIES_DIR / f"{name}.parquet").exists() or not (F.SPLITS_DIR / name / "splits.json").exists():
            pp = F.build_series(src, sid_c)
        else:
            pp = F.load_fresh(name)
        e = {
            "name": name, "source": src, "id": sid, "stratum": stratum, "nan_share": nan_share, "zero_share": zero_share,
            "train_val_shift_sd": shift(name), "n_test_windows_h6": int(pp.windows[6]["test"].X.shape[0]),
            "n_val_windows_h6": int(pp.windows[6]["val"].X.shape[0]), "n_train_windows_h6": int(pp.windows[6]["train"].X.shape[0]),
            "sha256": F.sha256_file(F.SERIES_DIR / f"{name}.parquet"),
            "splits_json_sha256": sha(F.SPLITS_DIR / name / "splits.json"),
        }
        e["monitor_trace"] = bool(e["train_val_shift_sd"] >= DRIFT_SD and e["n_test_windows_h6"] >= MIN_WIN[src])
        series.append(e)
        print(stratum, name, round(e["train_val_shift_sd"], 3), e["n_test_windows_h6"], e["monitor_trace"], flush=True)
    mon = lambda st: [e["name"] for e in series if e["stratum"] == st and e["monitor_trace"]]
    p2 = mon("P2_new") + mon("P2_FG")
    man = {
        "source": "round6 build_pool6.py (REPAIR_PREREG_6.md section 1)",
        "rules": {"missing_max": MISS_MAX, "zero_max": ZERO_MAX, "drift_sd": DRIFT_SD, "min_test_windows_h6": MIN_WIN},
        "series": series,
        "monitor_traces_P2": p2,
        "monitor_traces_P2_new": mon("P2_new"),
        "monitor_traces_P2_FG": mon("P2_FG"),
        "monitor_traces_P3_geant": mon("P3"),
        "counts": {
            "candidates": len(series),
            "P2_new_candidates": sum(e["stratum"] == "P2_new" for e in series),
            "P2_FG_candidates": sum(e["stratum"] == "P2_FG" for e in series),
            "P3_candidates": sum(e["stratum"] == "P3" for e in series),
            "P2_total": len(p2),
            "P2_by_source": {s: sum(1 for e in series if e["stratum"] != "P3" and e["monitor_trace"] and e["source"] == s) for s in ("cesnet", "abilene")},
            "P2_new": len(mon("P2_new")), "P2_FG": len(mon("P2_FG")), "P3_geant": len(mon("P3")),
        },
    }
    (HERE / "pool6_manifest.json").write_text(json.dumps(man, indent=1))
    ids = lambda st, src: [e["id"] for e in series if e["stratum"] == st and e["monitor_trace"] and e["source"] == src]
    blk = {
        "round6_M": {
            "purpose": "Round 6 monitor traces of Paper A (round6/REPAIR_PREREG_6.md). P2_new: never allocated CESNET/Abilene series admitted under the relaxed missingness rule; P2_FG: drifting CESNET/Abilene series of the round5_FG FG_half "
                       "(their test windows were scored by the F+G round 5 replication, never by E-Q, the guard, Chronos-Bolt mixtures or any monitor); P3: never allocated GEANT OD pairs (descriptive stratum, never pooled).",
            "rule": man["rules"],
            "P2_new": {s: ids("P2_new", s) for s in ("cesnet", "abilene")},
            "P2_FG": {s: ids("P2_FG", s) for s in ("cesnet", "abilene")},
            "P3": {"geant": ids("P3", "geant")},
            "pool6_manifest_sha256": sha(HERE / "pool6_manifest.json"),
            "note": "Additive block. The round5_FG block stays reserved for the F+G replication; the P2_FG ids are the only shared ids, F+G must not use E-Q or monitor outputs of them.",
        }
    }
    (HERE / "reserved_round6_M.json").write_text(json.dumps(blk, indent=1))
    print(json.dumps(man["counts"], indent=1))


if __name__ == "__main__":
    main()
