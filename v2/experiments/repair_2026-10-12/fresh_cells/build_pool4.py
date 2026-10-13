"""Build the round 4 pool of Paper A from the A half of the remaining admissible series (reserved_series.json, block round4_A).
Accuracy sample: the stride-selected 60 series (CESNET 30, Abilene 10, GEANT 20). Monitor candidates: all A-half CESNET and Abilene series (built for h = 6 use only).
Writes pool4_manifest.json (hashes of every series file, window sizes, train-to-validation level shift). Computes no forecast and no accuracy."""
import json

import pandas as pd

import fresh_cells as F

R = json.loads((F.HERE / "reserved_series.json").read_text())["round4_A"]
CAST = {"cesnet": int, "abilene": int, "geant": str}


def shift(name):
    sp = json.loads((F.SPLITS_DIR / name / "splits.json").read_text())
    ser = pd.read_parquet(F.SPLITS_DIR / name / "series_native.parquet")["value"]
    tr = ser[(ser.index >= sp["train"]["start"]) & (ser.index <= sp["train"]["end"])]
    va = ser[(ser.index >= sp["val"]["start"]) & (ser.index <= sp["val"]["end"])]
    return float(abs(va.mean() - tr.mean()) / tr.std())


def main():
    sel = R["accuracy_selected_by_stride"]["selected"]
    todo = {}
    for s in ("cesnet", "abilene", "geant"):
        ids = set(sel[s]) | (
            set(R["A_half"][s]) if s in ("cesnet", "abilene") else set()
        )
        todo[s] = [x for x in R["A_half"][s] if x in ids]
    man = {
        "source_block": "reserved_series.json round4_A",
        "accuracy_selected": sel,
        "series": [],
    }
    for s, ids in todo.items():
        for x in ids:
            sid = CAST[s](x)
            pp = F.build_series(s, sid)
            name = F.series_name(s, sid)
            man["series"].append(
                {
                    "name": name,
                    "source": s,
                    "id": str(x),
                    "accuracy": x in sel[s],
                    "train_val_shift_sd": shift(name),
                    "n_test_windows_h6": int(pp.windows[6]["test"].X.shape[0]),
                    "sha256": F.sha256_file(F.SERIES_DIR / f"{name}.parquet"),
                    "window_sizes": {
                        str(h): {
                            k: int(pp.windows[h][k].X.shape[0])
                            for k in ("train", "val", "test")
                        }
                        for h in F.HORIZONS
                    },
                }
            )
        print(s, len(ids), flush=True)
    mon = [
        t
        for t in man["series"]
        if t["source"] in ("cesnet", "abilene")
        and t["train_val_shift_sd"] >= 0.3
        and t["n_test_windows_h6"] >= 690
    ]
    man["monitor_traces"] = [t["name"] for t in mon]
    (F.HERE / "pool4_manifest.json").write_text(json.dumps(man, indent=1))
    print(
        "accuracy series",
        sum(t["accuracy"] for t in man["series"]),
        "monitor traces",
        len(mon),
    )


if __name__ == "__main__":
    main()
