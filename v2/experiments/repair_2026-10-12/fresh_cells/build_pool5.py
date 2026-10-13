"""Build the round 5 pool of Paper A (REPAIR_PREREG_5.md) from reserved_series.json block round5_A.
Stratum A: the 104 unused A-half series (accuracy, five horizons). Stratum F: CESNET and Abilene series of partition_F.json, screened for the monitor (h = 6 only).
Writes pool5_manifest.json (hashes of every series file, window sizes, train-to-validation level shift, monitor traces). Computes no forecast and no accuracy;
the test windows are only counted (shape), never read as values by model code."""
import json
import sys

import pandas as pd

import fresh_cells as F
from build_pool4 import shift

R = json.loads((F.HERE / "reserved_series.json").read_text())["round5_A"]
CAST = {"cesnet": int, "abilene": int, "geant": str}


def entry(source, x, stratum, accuracy):
    sid = CAST[source](x)
    name = F.series_name(source, sid)
    if not (F.SERIES_DIR / f"{name}.parquet").exists() or not (F.SPLITS_DIR / name / "splits.json").exists():
        pp = F.build_series(source, sid)
    else:
        pp = F.load_fresh(name)
    return {
        "name": name, "source": source, "id": str(x), "stratum": stratum, "accuracy": accuracy,
        "train_val_shift_sd": shift(name), "n_test_windows_h6": int(pp.windows[6]["test"].X.shape[0]),
        "sha256": F.sha256_file(F.SERIES_DIR / f"{name}.parquet"),
        "window_sizes": {str(h): {k: int(pp.windows[h][k].X.shape[0]) for k in ("train", "val", "test")} for h in F.HORIZONS},
    }


def main():
    man = {"source_block": "reserved_series.json round5_A", "series": []}
    for s, ids in R["accuracy_A"].items():
        for x in ids:
            man["series"].append(entry(s, x, "A", True))
        print("A", s, len(ids), flush=True)
    for s, ids in R["monitor_F_candidates"].items():
        for x in ids:
            man["series"].append(entry(s, x, "F", False))
        print("F", s, len(ids), flush=True)
    drift = lambda t: t["source"] in ("cesnet", "abilene") and t["train_val_shift_sd"] >= 0.3 and t["n_test_windows_h6"] >= 690
    man["monitor_traces"] = [t["name"] for t in man["series"] if drift(t)]
    man["monitor_traces_by_stratum"] = {k: sum(1 for t in man["series"] if t["stratum"] == k and drift(t)) for k in ("A", "F")}
    (F.HERE / "pool5_manifest.json").write_text(json.dumps(man, indent=1))
    print("accuracy series", sum(t["accuracy"] for t in man["series"]), "monitor traces", len(man["monitor_traces"]), man["monitor_traces_by_stratum"])


if __name__ == "__main__":
    main()
