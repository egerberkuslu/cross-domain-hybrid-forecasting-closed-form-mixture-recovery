"""Build the fresh pool of REPAIR_PREREG_3.md: 8 CESNET institutions, 6 Abilene OD pairs, 6 GEANT OD pairs (stride rule of fresh_cells.py).
Writes pool_manifest.json (selection, admissibility statistics, sha256 of every series file and split sizes). Computes NO forecast and NO accuracy."""
import json
import sys
from pathlib import Path

import fresh_cells as F

N = {"cesnet": 8, "abilene": 6, "geant": 6}


def main():
    manifest = {"rule": {"missing_slack": F.MISSING_SLACK, "zero_max": F.ZERO_MAX, "median_positive": True, "selection": "every k-th admissible id, k = floor(n_adm / N), from the first", "N": N},
                "series": []}
    for source, n in N.items():
        t = F.admissibility_table(source)
        adm = t[t.admissible].id.tolist()
        sel = F.select_series(source, n)
        print(source, "series", len(t), "admissible", len(adm), "selected", sel, flush=True)
        manifest.setdefault("admissibility", {})[source] = {"n_series": int(len(t)), "n_admissible": len(adm), "selected": [str(x) for x in sel],
                                                            "failures": {"missing": int((~t.ok_missing).sum()), "zero": int((~t.ok_zero).sum()), "median": int((~t.ok_median).sum())}}
        for sid in sel:
            pp = F.build_series(source, sid)
            name = F.series_name(source, sid)
            row = t[t.id == sid].iloc[0]
            manifest["series"].append({"name": name, "source": source, "id": str(sid), "n_hours": int(row.n_hours), "nan_share": float(row.nan_share), "zero_share": float(row.zero_share),
                                       "median": float(row["median"]), "mean": float(row["mean"]), "sha256": F.sha256_file(F.SERIES_DIR / f"{name}.parquet"),
                                       "window_sizes": {str(h): {s: int(pp.windows[h][s].X.shape[0]) for s in ("train", "val", "test")} for h in F.HORIZONS}})
    (F.HERE / "pool_manifest.json").write_text(json.dumps(manifest, indent=1))
    print("cells:", 5 * len(manifest["series"]))


if __name__ == "__main__":
    main()
