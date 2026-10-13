"""Round 7 pool builder (REPAIR_PREREG_7.md section 1). Deterministic, data-only, no forecast, no accuracy, no test value read (test windows are only counted;
the parent-correlation descriptive uses the first 85 % of the hours, that is train and validation, only).

Source: CESNET-TimeSeries24, Zenodo 10.5281/zenodo.13382427 (CC BY 4.0): institution_subnets.tar.gz (548 subnets), ip_addresses_sample.tar.gz (1,000 IP addresses),
ids_relationship.csv (IP -> subnet -> institution). Files under data/raw/cesnet_ts24_extra/ (md5 verified against the Zenodo record by download7.sh). The hourly
aggregation, the calendar (data/raw/cesnet/times.tar.gz, the file the shipped loaders use) and every later step are those of fresh_cells.py:
  hourly n_bytes -> outlier flag (100 x median -> NaN) -> time-interpolation of gaps <= 24 h -> split 0.70 / 0.15 / 0.15, scaler on train -> windows (lookback 168).

Candidate rule (the round 6 build_pool6 rule with the round 6 CESNET missing-data limit, fixed before any forecast):
  missing share <= 0.05, zero share <= 0.05, median > 0, parent institution not a monitor trace of rounds 3 to 6 (their test windows were revealed),
  then the drift screen build_pool4.shift >= 0.3 train sd (train and validation hours only) and n_test_windows_h6 >= 690.
Near-copy rule (coordinator decision after the pre-commit review, train and validation hours only): a drifting candidate with corr_parent_trainval >= 0.95 whose parent institution
was allocated to an earlier round (accuracy, pilot, F+G or monitor pool; flag parent_used_earlier_round) is dropped (monitor_trace stays True, corr_rule_excluded True).
Selection (D7): 30 subnets, then 30 IP addresses, by sorted id. Per level the screened list is cut by the family cap (at most 3 per institution, subnets and IPs
counted together: the IP cap is 3 minus the institution's selected subnets), then the stride rule of fresh_cells.select_series (every k-th, k = floor(n / 30), first
element first) gives the ORDER; the first 30 are primary, the rest are the ordered reserve used only by the validation-only degenerate-expert rule (r7_monitor.py --stage degen).
D4 stratum G7: the 16 GEANT leftovers admissible under rule M7a of DATA7.md (data7_leftovers_summary.json), descriptive, never pooled.
Writes pool7_manifest.json (sha256 of every series file and splits.json) and reserved_round7.json (additive blocks round7_C and round7_M for fresh_cells/reserved_series.json).
  PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python build_pool7.py
"""
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FC = HERE.parent / "fresh_cells"
sys.path.insert(0, str(FC))
import fresh_cells as F  # noqa: E402
from build_pool4 import shift  # noqa: E402

ROOT = F.ROOT
RAW7 = ROOT / "data" / "raw" / "cesnet_ts24_extra"
LEVELS = {
    "sub": {
        "tar": "institution_subnets.tar.gz",
        "prefix": "institution_subnets/agg_1_hour/",
        "cache": "cesnet_subnets_hourly.parquet",
        "name": "cesnet_sub",
    },
    "ip": {
        "tar": "ip_addresses_sample.tar.gz",
        "prefix": "ip_addresses_sample/agg_1_hour/",
        "cache": "cesnet_ips_hourly.parquet",
        "name": "cesnet_ip",
    },
}
ZENODO_MD5 = {
    "ids_relationship.csv": "f034e4e7f1844e36fa7d3a4a0cbfc600",
    "institution_subnets.tar.gz": "735b5ff436b6c67556cb67e2888b14ea",
    "ip_addresses_sample.tar.gz": "08451dab2d1eddb29a79467b03289232",
}
TIMES_MD5 = "a03813763e07646ca38f17ffd53e549e"
MISS_MAX, ZERO_MAX, DRIFT_SD, MIN_WIN = 0.05, 0.05, 0.3, 690
N_PER_LEVEL, FAMILY_CAP = 30, 3
CORR_MAX = 0.95  # near-copy rule (train and validation hours only): drop a candidate whose correlation with its parent institution series is >= 0.95 and whose parent was read in an earlier round
def _hash(algo, p):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


sha = lambda p: _hash("sha256", p)
md5 = lambda p: _hash("md5", p)


def times_map():
    p = F.RAW / "cesnet" / "times.tar.gz"
    assert md5(p) == TIMES_MD5, "times.tar.gz differs from the Zenodo record"
    with tarfile.open(p, "r:gz") as t:
        times = pd.read_csv(io.BytesIO(t.extractfile("times/times_1_hour.csv").read()))
    times["time"] = pd.to_datetime(times["time"], utc=True).dt.tz_convert(None)
    return times.set_index("id_time")["time"]


def hourly(level):
    """Hours x ids of one level, n_bytes, the fresh_cells._cesnet_hourly aggregation. Cached."""
    cfg = LEVELS[level]
    p = F.CACHE / cfg["cache"]
    if p.exists():
        return pd.read_parquet(p)
    for f, h in ZENODO_MD5.items():
        assert md5(RAW7 / f) == h, f"{f} md5 differs from the Zenodo record"
    tmap = times_map()
    cols = {}
    with tarfile.open(RAW7 / cfg["tar"], "r:gz") as t:
        for m in t:
            if (
                m.isfile()
                and m.name.startswith(cfg["prefix"])
                and m.name.endswith(".csv")
            ):
                sid = int(Path(m.name).stem)
                df = pd.read_csv(t.extractfile(m), usecols=["id_time", "n_bytes"])
                s = df.set_index("id_time")["n_bytes"].astype("float64")
                s.index = tmap.reindex(s.index).values
                cols[sid] = s[~pd.isna(s.index)]
    out = pd.DataFrame(cols).sort_index()
    out = out[sorted(out.columns)]
    out.index.name = "timestamp"
    out.to_parquet(p)
    return out


def relationship():
    """ip -> institution (unique), subnet -> primary institution (the smallest id when the record lists two), subnet -> all institutions, subnets per institution."""
    r = pd.read_csv(RAW7 / "ids_relationship.csv")
    assert r.id_ip.is_unique
    ip2inst = dict(zip(r.id_ip.astype(int), r.id_institution.astype(int)))
    sub = r.drop_duplicates(["id_institution_subnet", "id_institution"]).sort_values(["id_institution_subnet", "id_institution"])
    sub2insts = sub.groupby("id_institution_subnet").id_institution.apply(lambda x: sorted(int(i) for i in x)).to_dict()
    sub2insts = {int(k): v for k, v in sub2insts.items()}
    sub2inst = {k: v[0] for k, v in sub2insts.items()}
    n_sub_per_inst = sub.groupby("id_institution").size().to_dict()
    return ip2inst, sub2inst, sub2insts, n_sub_per_inst


def revealed_monitor_institutions():
    """Institutions whose institution-level series was a monitor trace of round 3, 4, 5 or 6 (test windows revealed)."""
    names = set()
    m4 = json.loads((FC / "pool4_manifest.json").read_text())
    m5 = json.loads((FC / "pool5_manifest.json").read_text())
    m6 = json.loads((HERE.parent / "round6" / "pool6_manifest.json").read_text())
    m3 = set(pd.read_csv(HERE.parent / "round3" / "r3_monitor_runs.csv", usecols=["series"]).series.unique())
    names |= (
        m3
        | set(m4["monitor_traces"])
        | set(m5["monitor_traces"])
        | set(m6["monitor_traces_P2"])
    )
    by_round = {
        "r3": m3,
        "r4": set(m4["monitor_traces"]),
        "r5": set(m5["monitor_traces"]),
        "r6": set(m6["monitor_traces_P2"]),
    }
    ids = lambda ns: {
        int(n[len("cesnet_inst") :]) for n in ns if n.startswith("cesnet_inst")
    }
    return ids(names), {k: sorted(ids(v)) for k, v in by_round.items()}


def all_previously_used_institutions():
    R = json.loads((FC / "reserved_series.json").read_text())
    P = json.loads((FC / "partition_F.json").read_text())
    u = (
        set(map(int, R["reserved"]["cesnet"]))
        | set(map(int, P["pilot"]["cesnet"]))
        | set(map(int, P["F_registered"]["cesnet"]))
    )
    u |= set(map(int, R["round4_A"]["A_half"]["cesnet"])) | set(
        map(int, R["round5_FG"]["FG_half"]["cesnet"])
    )
    m6 = json.loads((HERE.parent / "round6" / "pool6_manifest.json").read_text())
    u |= {int(e["id"]) for e in m6["series"] if e["source"] == "cesnet"}
    return u


def build_series7(level, sid):
    from src.preprocessing import preprocess_one

    cfg = LEVELS[level]
    name = f"{cfg['name']}{sid}"
    df = hourly(level)
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    s = F._flag_outliers(df[sid].reindex(full))
    out = pd.DataFrame({"value": s.astype("float64")})
    out.index.name = "timestamp"
    path = F.SERIES_DIR / f"{name}.parquet"
    out.to_parquet(path)
    return preprocess_one(
        name=name,
        series_path=path,
        freq="1h",
        lookback=F.LOOKBACK,
        horizons=F.HORIZONS,
        train_frac=F.FRACS[0],
        val_frac=F.FRACS[1],
        test_frac=F.FRACS[2],
        scaler_name="standard",
        out_root=F.SPLITS_DIR,
        interp_method="time",
    )


def level_table(level, parent_of):
    df = hourly(level)
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    inst_h = F._cesnet_hourly().reindex(full)
    cut = int(len(full) * 0.85)  # train and validation hours only (0.70 + 0.15)
    rows = []
    for c in df.columns:
        s = F._flag_outliers(df[c].reindex(full))
        parent = parent_of[int(c)]
        corr = float("nan")
        if parent in inst_h.columns:
            a, b = s.iloc[:cut], F._flag_outliers(inst_h[parent]).iloc[:cut]
            ok = a.notna() & b.notna()
            if ok.sum() > 100 and a[ok].std() > 0 and b[ok].std() > 0:
                corr = float(np.corrcoef(a[ok], b[ok])[0, 1])
        rows.append(
            {
                "level": level,
                "id": int(c),
                "institution": int(parent),
                "nan_share": float(s.isna().mean()),
                "zero_share": float((s == 0).mean()),
                "median": float(s.median()),
                "corr_parent_trainval": corr,
            }
        )
    return pd.DataFrame(rows)


def cut_cap(ids, inst_of, used):
    """Greedy family cap in id order: keep an id while its institution has fewer than FAMILY_CAP kept (counting `used` already)."""
    cnt, out = dict(used), []
    for i in ids:
        k = inst_of[i]
        if cnt.get(k, 0) < FAMILY_CAP:
            out.append(i)
            cnt[k] = cnt.get(k, 0) + 1
    return out


def stride_order(ids, n):
    k = max(1, len(ids) // n)
    first = ids[::k]
    return first + [i for i in ids if i not in set(first)]


def main():
    ip2inst, sub2inst, sub2insts, n_sub_per_inst = relationship()
    excl_inst, excl_by_round = revealed_monitor_institutions()
    used_inst = all_previously_used_institutions()
    print(
        "excluded parent institutions (monitor traces of rounds 3 to 6):",
        len(excl_inst),
        flush=True,
    )
    tabs, entries = {}, []
    for level, parent_of in (("sub", sub2inst), ("ip", ip2inst)):
        t = level_table(level, parent_of)
        t["parent_excluded"] = t.institution.isin(excl_inst)
        if level == "sub":  # a subnet listed under two institutions is excluded if either parent was a revealed monitor trace
            t["parent_excluded"] = t.id.map(lambda i: any(x in excl_inst for x in sub2insts[int(i)]))
        t["parent_used_earlier_round"] = t.institution.isin(used_inst)
        t["admissible"] = (
            (t.nan_share <= MISS_MAX) & (t.zero_share <= ZERO_MAX) & (t["median"] > 0)
        )
        tabs[level] = t
        print(
            level,
            "ids",
            len(t),
            "admissible",
            int(t.admissible.sum()),
            "parent excluded",
            int(t.parent_excluded.sum()),
            "candidates",
            int((t.admissible & ~t.parent_excluded).sum()),
            flush=True,
        )
    for level, t in tabs.items():
        cfg = LEVELS[level]
        for r in t[t.admissible & ~t.parent_excluded].sort_values("id").itertuples():
            name = f"{cfg['name']}{r.id}"
            if (
                not (F.SERIES_DIR / f"{name}.parquet").exists()
                or not (F.SPLITS_DIR / name / "splits.json").exists()
            ):
                pp = build_series7(level, r.id)
            else:
                pp = F.load_fresh(name)
            e = {
                "name": name,
                "level": level,
                "id": int(r.id),
                "institution": int(r.institution),
                "family_size_subnets": int(n_sub_per_inst.get(int(r.institution), 0)),
                "nan_share": r.nan_share,
                "zero_share": r.zero_share,
                "corr_parent_trainval": r.corr_parent_trainval,
                "parent_used_earlier_round": bool(r.parent_used_earlier_round),
                "train_val_shift_sd": shift(name),
                "n_test_windows_h6": int(pp.windows[6]["test"].X.shape[0]),
                "n_val_windows_h6": int(pp.windows[6]["val"].X.shape[0]),
                "n_train_windows_h6": int(pp.windows[6]["train"].X.shape[0]),
                "sha256": F.sha256_file(F.SERIES_DIR / f"{name}.parquet"),
                "splits_json_sha256": sha(F.SPLITS_DIR / name / "splits.json"),
            }
            e["corr_rule_excluded"] = bool(
                np.isfinite(e["corr_parent_trainval"])
                and e["corr_parent_trainval"] >= CORR_MAX
                and e["parent_used_earlier_round"]
            )
            e["monitor_trace"] = bool(
                e["train_val_shift_sd"] >= DRIFT_SD
                and e["n_test_windows_h6"] >= MIN_WIN
            )
            entries.append(e)
        print(
            level,
            "built",
            sum(1 for e in entries if e["level"] == level),
            "drifting monitor candidates",
            sum(1 for e in entries if e["level"] == level and e["monitor_trace"]),
            flush=True,
        )
    # ---- selection (D7)
    mon = {
        lv: sorted(
            [e for e in entries if e["level"] == lv and e["monitor_trace"] and not e["corr_rule_excluded"]],
            key=lambda e: e["id"],
        )
        for lv in LEVELS
    }
    inst_of = {lv: {e["id"]: e["institution"] for e in mon[lv]} for lv in LEVELS}
    sub_capped = cut_cap([e["id"] for e in mon["sub"]], inst_of["sub"], {})
    order_sub = stride_order(sub_capped, N_PER_LEVEL)
    prim_sub = order_sub[:N_PER_LEVEL]
    used = {}
    for i in prim_sub:
        used[inst_of["sub"][i]] = used.get(inst_of["sub"][i], 0) + 1
    ip_capped = cut_cap([e["id"] for e in mon["ip"]], inst_of["ip"], used)
    order_ip = stride_order(ip_capped, N_PER_LEVEL)
    prim_ip = order_ip[:N_PER_LEVEL]
    short = (N_PER_LEVEL - len(prim_sub)) + (N_PER_LEVEL - len(prim_ip))
    # ---- GEANT descriptive stratum G7 (D4): 16 leftovers admissible under M7a
    lo = json.loads((HERE / "data7_leftovers_summary.json").read_text())["geant"][
        "drifting_M7a_ids"
    ]
    g7 = []
    for gid in lo:
        name = F.series_name("geant", gid)
        pp = (
            F.build_series("geant", gid)
            if not (F.SERIES_DIR / f"{name}.parquet").exists()
            or not (F.SPLITS_DIR / name / "splits.json").exists()
            else F.load_fresh(name)
        )
        g7.append(
            {
                "name": name,
                "source": "geant",
                "id": gid,
                "train_val_shift_sd": shift(name),
                "n_test_windows_h6": int(pp.windows[6]["test"].X.shape[0]),
                "n_val_windows_h6": int(pp.windows[6]["val"].X.shape[0]),
                "sha256": F.sha256_file(F.SERIES_DIR / f"{name}.parquet"),
                "splits_json_sha256": sha(F.SPLITS_DIR / name / "splits.json"),
            }
        )
    g7_ok = [
        e
        for e in g7
        if e["train_val_shift_sd"] >= DRIFT_SD and e["n_test_windows_h6"] >= 420
    ]
    nm = lambda lv, ids: [f"{LEVELS[lv]['name']}{i}" for i in ids]
    man = {
        "source": "round7 build_pool7.py (REPAIR_PREREG_7.md section 1)",
        "dataset": {
            "zenodo": "10.5281/zenodo.13382427",
            "licence": "CC BY 4.0",
            "files_md5_zenodo": ZENODO_MD5,
            "times_tar_md5": TIMES_MD5,
            "files_sha256": {f: sha(RAW7 / f) for f in ZENODO_MD5},
        },
        "rules": {
            "missing_max": MISS_MAX,
            "zero_max": ZERO_MAX,
            "drift_sd": DRIFT_SD,
            "min_test_windows_h6": MIN_WIN,
            "per_level": N_PER_LEVEL,
            "family_cap": FAMILY_CAP,
            "corr_max_near_copy": CORR_MAX,
            "excluded_parent_institutions_rounds_3_to_6": sorted(excl_inst),
            "excluded_parents_by_round": excl_by_round,
        },
        "series": entries,
        "G7_series": g7,
        "monitor_order": {"sub": nm("sub", order_sub), "ip": nm("ip", order_ip)},
        "monitor_primary": {"sub": nm("sub", prim_sub), "ip": nm("ip", prim_ip)},
        "G7_traces": [e["name"] for e in g7_ok],
        "counts": {
            "ids_subnets": int(len(tabs["sub"])),
            "ids_ips": int(len(tabs["ip"])),
            "admissible_subnets": int(tabs["sub"].admissible.sum()),
            "admissible_ips": int(tabs["ip"].admissible.sum()),
            "parent_excluded_subnets": int(tabs["sub"].parent_excluded.sum()),
            "parent_excluded_ips": int(tabs["ip"].parent_excluded.sum()),
            "candidates_built_subnets": sum(e["level"] == "sub" for e in entries),
            "candidates_built_ips": sum(e["level"] == "ip" for e in entries),
            "drifting_len_ok_before_corr_rule_subnets": sum(e["level"] == "sub" and e["monitor_trace"] for e in entries),
            "drifting_len_ok_before_corr_rule_ips": sum(e["level"] == "ip" and e["monitor_trace"] for e in entries),
            "corr_rule_excluded": sorted(e["name"] for e in entries if e["monitor_trace"] and e["corr_rule_excluded"]),
            "drifting_len_ok_subnets": len(mon["sub"]),
            "drifting_len_ok_ips": len(mon["ip"]),
            "after_family_cap_subnets": len(sub_capped),
            "after_family_cap_ips": len(ip_capped),
            "primary_subnets": len(prim_sub),
            "primary_ips": len(prim_ip),
            "reserve_subnets": len(order_sub) - len(prim_sub),
            "reserve_ips": len(order_ip) - len(prim_ip),
            "primary_total": len(prim_sub) + len(prim_ip),
            "shortfall_vs_60": short,
            "institutions_in_primary": len(
                {inst_of["sub"][i] for i in prim_sub}
                | {inst_of["ip"][i] for i in prim_ip}
            ),
            "G7": len(g7_ok),
        },
    }
    (HERE / "pool7_manifest.json").write_text(json.dumps(man, indent=1))
    blk = {
        "round7_C": {
            "purpose": "Round 7 candidate pool of Paper A (round7/REPAIR_PREREG_7.md): every CESNET-TimeSeries24 subnet / IP-sample id that was built and screened (admissible under the round 6 CESNET limits, parent institution not a "
            "round 4, 5 or 6 monitor trace). Later draws from the subnet / IP level must exclude these ids.",
            "subnets": sorted(e["id"] for e in entries if e["level"] == "sub"),
            "ips": sorted(e["id"] for e in entries if e["level"] == "ip"),
            "excluded_parent_institutions": sorted(excl_inst),
        },
        "round7_M": {
            "purpose": "Round 7 monitor traces of Paper A. primary = the first 30 of each level in the registered order; reserve = the ordered replacements for the validation-only degenerate-expert rule "
            "(the final list is round7/degen7.json). G7: the GEANT M7a leftovers (descriptive stratum, never pooled).",
            "primary": man["monitor_primary"],
            "order": man["monitor_order"],
            "G7": {"geant": [e["id"] for e in g7_ok]},
            "pool7_manifest_sha256": sha(HERE / "pool7_manifest.json"),
            "note": "Additive blocks; merge into fresh_cells/reserved_series.json by the coordinator. Subnets and IPs are components of CESNET institutions whose series were forecast in earlier rounds.",
        },
    }
    (HERE / "reserved_round7.json").write_text(json.dumps(blk, indent=1))
    print(json.dumps(man["counts"], indent=1))


if __name__ == "__main__":
    main()
