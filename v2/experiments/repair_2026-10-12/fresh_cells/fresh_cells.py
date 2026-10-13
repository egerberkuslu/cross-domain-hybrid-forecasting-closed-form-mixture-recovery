"""Reusable builder of FRESH series cells from the raw network datasets already on disk.

Sources (all at 1 h after the same aggregation as the shipped loaders):
  cesnet  CESNET-TimeSeries24, one series per INSTITUTION (n_bytes, provider's hourly aggregation)       data/raw/cesnet
  abilene one series per OD PAIR of the 12x12 Abilene traffic matrix (5-min sums to hourly, x100 bytes)  data/raw/abilene
  geant   one series per OD PAIR of the 23x23 GEANT traffic matrix (15-min kbps sums to hourly)          data/raw/geant
The paper series are the aggregates (sum over institutions or OD pairs); the components below are different forecasting problems
but are NOT independent of the aggregate (same network, same calendar period). The README states this.

Selection (pre-stated, data-only, never by forecasts): a series is ADMISSIBLE iff
  (i)   its share of missing hours is at most the share of the corresponding aggregate series plus 0.5 percentage points,
  (ii)  at most 5 % of its hours are exactly zero, and
  (iii) its median is positive.
From the admissible series of a source, ordered by numeric id (OD index = 12 * src + dst for Abilene, (src, dst) lexicographic for GEANT),
take every k-th, k = floor(n_admissible / N), starting with the first, until N series are taken.

Preprocessing of an admissible series is the shipped pipeline: outlier clip (100 x median -> NaN), `src.preprocessing.preprocess_one` with
interpolation `time`, chronological split 0.70 / 0.15 / 0.15, standard scaler fitted on TRAIN only, lookback 168, horizons 1, 3, 6, 12, 24.

Public API
  FRESH_ROOT, CACHE, SERIES_DIR, SPLITS_DIR
  load_raw_hourly(source)            -> DataFrame (hours x series id), cached in cache/
  admissibility_table(source)        -> DataFrame with the three criteria per series
  select_series(source, n)           -> list of series ids by the stride rule
  build_series(source, sid, ...)     -> writes series/<name>.parquet and the preprocessing artefacts, returns PreprocessedDataset
  load_fresh(name)                   -> PreprocessedDataset (all windows on disk)
  Restricted(pp, allow_test=False)   -> view exposing train/val windows only (test access raises), same API as round2/experts.py
Run `python3 build_pool.py` for the pool of REPAIR_PREREG_3.md. No forecast or accuracy is computed anywhere in this module.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import sys
import tarfile
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
FRESH_ROOT = HERE
CACHE = HERE / "cache"
SERIES_DIR = HERE / "series"
SPLITS_DIR = HERE / "splits"
for d in (CACHE, SERIES_DIR, SPLITS_DIR):
    d.mkdir(exist_ok=True)

LOOKBACK, HORIZONS = 168, [1, 3, 6, 12, 24]
FRACS = (0.70, 0.15, 0.15)
MISSING_SLACK, ZERO_MAX = 0.005, 0.05
OUTLIER_FACTOR = 100.0


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ raw readers
def _flag_outliers(s: pd.Series) -> pd.Series:
    med = float(s.median())
    if np.isfinite(med) and med > 0:
        s = s.copy()
        s[s > OUTLIER_FACTOR * med] = np.nan
    return s


def _cesnet_hourly() -> pd.DataFrame:
    p = CACHE / "cesnet_institutions_hourly.parquet"
    if p.exists():
        return pd.read_parquet(p)
    with tarfile.open(RAW / "cesnet" / "times.tar.gz", "r:gz") as t:
        times = pd.read_csv(io.BytesIO(t.extractfile("times/times_1_hour.csv").read()))
    times["time"] = pd.to_datetime(times["time"], utc=True).dt.tz_convert(None)
    tmap = times.set_index("id_time")["time"]
    cols = {}
    with tarfile.open(RAW / "cesnet" / "institutions.tar.gz", "r:gz") as t:
        for m in t.getmembers():
            if m.name.startswith("institutions/agg_1_hour/") and m.name.endswith(".csv"):
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


_ABILENE_WEEKS = [("X%02d.gz" % i, d) for i, d in enumerate(
    ["2004-03-01", "2004-03-08", "2004-04-02", "2004-04-09", "2004-04-22", "2004-05-01", "2004-05-08", "2004-05-15", "2004-05-22", "2004-05-29", "2004-06-05",
     "2004-06-12", "2004-06-19", "2004-06-26", "2004-07-03", "2004-07-10", "2004-07-17", "2004-07-24", "2004-07-31", "2004-08-07", "2004-08-13", "2004-08-21",
     "2004-08-28", "2004-09-04"], 1)]


def _abilene_hourly() -> pd.DataFrame:
    p = CACHE / "abilene_od_hourly.parquet"
    if p.exists():
        return pd.read_parquet(p)
    frames = []
    for fname, start in _ABILENE_WEEKS:
        with gzip.open(RAW / "abilene" / fname, "rt") as f:
            arr = np.loadtxt(f, dtype=np.float64)
        od = arr[:, np.arange(144) * 5] * 100.0  # real OD column of each estimator block, bytes per 5 min
        idx = pd.date_range(start=start, periods=arr.shape[0], freq="5min")
        frames.append(pd.DataFrame(od, index=idx))
    df = pd.concat(frames).sort_index()
    df = df.groupby(level=0).mean() if df.index.has_duplicates else df
    out = df.resample("1h").sum(min_count=1)
    out.columns = list(range(144))
    out.index.name = "timestamp"
    out.to_parquet(p)
    return out


def _geant_hourly() -> pd.DataFrame:
    p = CACHE / "geant_od_hourly.parquet"
    if p.exists():
        return pd.read_parquet(p)
    rows, ts_list = [], []
    pairs = None
    with tarfile.open(RAW / "geant" / "traffic-matrices-anonymized-v2.tar.bz2", "r:bz2") as t:
        for m in t:
            if not (m.name.startswith("traffic-matrices/IntraTM-") and m.name.endswith(".xml")):
                continue
            root = ET.fromstring(t.extractfile(m).read())
            ts = None
            info = root.find("info")
            if info is not None and info.find("date") is not None:
                try:
                    ts = datetime.fromisoformat(info.find("date").text.strip())
                except ValueError:
                    ts = None
            if ts is None:
                y, mo, d, h, mi = map(int, re.search(r"IntraTM-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", m.name).groups())
                ts = datetime(y, mo, d, h, mi)
            vals = {}
            for src in root.iter("src"):
                for dst in src.findall("dst"):
                    try:
                        vals[(int(src.get("id")), int(dst.get("id")))] = float(dst.text)
                    except (TypeError, ValueError):
                        pass
            rows.append(vals)
            ts_list.append(ts)
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(ts_list)).sort_index()
    df = df[sorted(df.columns)]
    df = df.groupby(level=0).mean() if df.index.has_duplicates else df
    out = df.resample("1h").sum(min_count=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.index.name = "timestamp"
    out.to_parquet(p)
    return out


def load_raw_hourly(source: str) -> pd.DataFrame:
    """Hours x series, aggregated to hourly with the shipped rules (sum, min_count=1). Outliers are flagged later, per series."""
    return {"cesnet": _cesnet_hourly, "abilene": _abilene_hourly, "geant": _geant_hourly}[source]()


# ------------------------------------------------------------------ admissibility and selection
def aggregate_nan_share(source: str) -> float:
    s = pd.read_parquet(ROOT / "data" / "processed" / f"{source}.parquet")["value"]
    full = pd.date_range(s.index.min(), s.index.max(), freq="1h")
    return float(s.reindex(full).isna().mean())


def admissibility_table(source: str) -> pd.DataFrame:
    df = load_raw_hourly(source)
    agg_nan = aggregate_nan_share(source)
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    rows = []
    for c in df.columns:
        s = _flag_outliers(df[c].reindex(full))
        nan_share = float(s.isna().mean())
        zero_share = float((s == 0).mean())
        med = float(s.median())
        rows.append({"source": source, "id": c, "n_hours": len(s), "nan_share": nan_share, "zero_share": zero_share, "median": med, "mean": float(s.mean()),
                     "ok_missing": nan_share <= agg_nan + MISSING_SLACK, "ok_zero": zero_share <= ZERO_MAX, "ok_median": bool(med > 0)})
    t = pd.DataFrame(rows)
    t["admissible"] = t.ok_missing & t.ok_zero & t.ok_median
    return t


def reserved_ids(source: str) -> set:
    """Series ids reserved by other registrations (reserved_series.json); other draws must exclude them. Ids are returned in the dtype of the source."""
    p = HERE / "reserved_series.json"
    if not p.exists():
        return set()
    r = json.loads(p.read_text())["reserved"].get(source, [])
    return {int(x) if source in ("cesnet", "abilene") else x for x in r}


def select_series(source: str, n: int) -> list:
    t = admissibility_table(source)
    adm = [i for i in t[t.admissible].id.tolist()]
    k = max(1, len(adm) // n)
    return adm[::k][:n]


# ------------------------------------------------------------------ build / load
def series_name(source: str, sid) -> str:
    return {"cesnet": "cesnet_inst", "abilene": "abilene_od", "geant": "geant_od"}[source] + str(sid)


def build_series(source: str, sid, force=False):
    from src.preprocessing import preprocess_one
    name = series_name(source, sid)
    df = load_raw_hourly(source)
    full = pd.date_range(df.index.min(), df.index.max(), freq="1h")
    s = _flag_outliers(df[sid].reindex(full))
    out = pd.DataFrame({"value": s.astype("float64")})
    out.index.name = "timestamp"
    path = SERIES_DIR / f"{name}.parquet"
    out.to_parquet(path)
    return preprocess_one(name=name, series_path=path, freq="1h", lookback=LOOKBACK, horizons=HORIZONS, train_frac=FRACS[0], val_frac=FRACS[1],
                          test_frac=FRACS[2], scaler_name="standard", out_root=SPLITS_DIR, interp_method="time")


def load_fresh(name: str):
    from src.preprocessing import load_preprocessed
    return load_preprocessed(name, root=SPLITS_DIR)


class Restricted:
    """View of a PreprocessedDataset that exposes train and val windows only (test access raises unless allow_test)."""

    def __init__(self, pp, allow_test=False):
        self._pp, self._allow = pp, allow_test
        self.accessed = []

    def windows(self, h, split):
        if split == "test" and not self._allow:
            raise PermissionError("test windows are sealed")
        self.accessed.append((h, split))
        return self._pp.windows[h][split]
