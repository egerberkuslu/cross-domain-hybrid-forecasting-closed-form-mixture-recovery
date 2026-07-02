"""Phase-2 orchestrator.

Builds, caches, and (re)loads the canonical preprocessed dataset that
every model in Phases 3-5 must consume — guaranteeing identical splits
and identical scaling across the entire model fleet.

Outputs per dataset (under ``data/processed/splits/<name>/``):

    series_native.parquet      cleaned aggregate series (post-interpolation,
                               NaNs in long gaps preserved)
    series_scaled.parquet      same series after the train-fit scaler
    splits.json                {train,val,test}: {start, end, n}
    manifest.json              full preprocessing manifest (config snapshot)
    windows/h{h}_{split}.npz   X, y, target_times per horizon and split
    scaler.joblib              fitted scaler

These artifacts are deterministic given config + Phase-1 parquets.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .interpolation import handle_missing, InterpReport
from .splits import ChronologicalSplit, make_splits
from .scaling import fit_scaler_on_train, transform
from .windowing import WindowSet, make_sliding_windows

logger = logging.getLogger(__name__)


# Convention: outputs live here so every model can `load_preprocessed(name)`.
SPLITS_DIR = Path("data/processed/splits")


@dataclass
class PreprocessedDataset:
    name: str
    freq: str
    lookback: int
    horizons: list[int]
    series_native: pd.DataFrame      # cleaned, original units
    series_scaled: pd.DataFrame      # train-fit scaler applied
    split_native: ChronologicalSplit # in original units
    split_scaled: ChronologicalSplit # in scaled units
    scaler: object                   # fitted sklearn scaler
    windows: dict[int, dict[str, WindowSet]]  # {h: {split: WindowSet}}
    interp_report: InterpReport
    out_dir: Path


# ----------------------------------------------------------------------
# core
# ----------------------------------------------------------------------


def preprocess_one(
    name: str,
    series_path: str | Path,
    freq: str,
    lookback: int,
    horizons: list[int],
    train_frac: float,
    val_frac: float,
    test_frac: float,
    scaler_name: str,
    out_root: str | Path = SPLITS_DIR,
    interp_method: str = "time",
    interp_max_gap: int = 24,
) -> PreprocessedDataset:
    """Run the full Phase-2 pipeline for one dataset and cache to disk."""
    out_dir = Path(out_root) / name
    (out_dir / "windows").mkdir(parents=True, exist_ok=True)

    # 1) load loader output
    df = pd.read_parquet(series_path)
    logger.info("[%s] loaded %s rows=%d, nan=%d", name, series_path, len(df),
                int(df["value"].isna().sum()))

    # 2) handle missing values (reindex + interpolate small gaps)
    df_clean, interp = handle_missing(
        df, freq=freq, max_gap_steps=interp_max_gap, method=interp_method
    )

    # 3) chronological split BEFORE scaling
    split_native = make_splits(df_clean, train_frac, val_frac, test_frac)
    split_native.assert_chronological()

    # 4) fit scaler on TRAIN ONLY
    scaler_path = out_dir / "scaler.joblib"
    sc = fit_scaler_on_train(split_native.train, scaler_name=scaler_name,
                             save_to=scaler_path)
    train_only_mean = float(np.nanmean(split_native.train["value"].values))
    train_only_std = float(np.nanstd(split_native.train["value"].values))
    if hasattr(sc, "mean_"):
        assert abs(float(sc.mean_[0]) - train_only_mean) < 1e-3, (
            "scaler.mean_ does not match train-only mean — leakage detected"
        )
    # 5) apply scaler to entire cleaned series + each split
    series_scaled = transform(sc, df_clean)
    split_scaled = ChronologicalSplit(
        train=transform(sc, split_native.train),
        val=transform(sc, split_native.val),
        test=transform(sc, split_native.test),
    )
    split_scaled.assert_chronological()

    # 6) windowing per horizon, using last-target timestamps to assign split
    windows: dict[int, dict[str, WindowSet]] = {}
    for h in horizons:
        win_for_h = make_sliding_windows(
            series=series_scaled["value"],
            split_ranges=split_native.ranges,   # ranges identical in scaled domain
            lookback=lookback,
            horizon=int(h),
        )
        windows[int(h)] = win_for_h

        # cache per split as .npz
        for split_name, ws in win_for_h.items():
            np.savez_compressed(
                out_dir / "windows" / f"h{h}_{split_name}.npz",
                X=ws.X,
                y=ws.y,
                target_times=np.array(ws.target_times.view("int64"), dtype=np.int64),
                lookback=np.int32(ws.lookback),
                horizon=np.int32(ws.horizon),
            )

    # 7) write series + splits + manifest
    df_clean.to_parquet(out_dir / "series_native.parquet")
    series_scaled.to_parquet(out_dir / "series_scaled.parquet")

    split_meta = {
        s: {
            "start": str(getattr(split_native, s).index.min()),
            "end":   str(getattr(split_native, s).index.max()),
            "n":     int(len(getattr(split_native, s))),
        }
        for s in ["train", "val", "test"]
    }
    with (out_dir / "splits.json").open("w") as f:
        json.dump(split_meta, f, indent=2, default=str)

    manifest = {
        "name": name,
        "freq": freq,
        "lookback": int(lookback),
        "horizons": [int(h) for h in horizons],
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "scaler": scaler_name,
        "interp_method": interp_method,
        "interp_max_gap_steps": int(interp_max_gap),
        "n_total": int(len(df_clean)),
        "n_remaining_nan": int(df_clean["value"].isna().sum()),
        "interp_report": asdict(interp),
        "splits": split_meta,
        "window_sizes": {
            int(h): {s: int(windows[int(h)][s].X.shape[0]) for s in ["train", "val", "test"]}
            for h in horizons
        },
    }
    with (out_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("[%s] wrote preprocessing artifacts to %s", name, out_dir)
    return PreprocessedDataset(
        name=name,
        freq=freq,
        lookback=int(lookback),
        horizons=[int(h) for h in horizons],
        series_native=df_clean,
        series_scaled=series_scaled,
        split_native=split_native,
        split_scaled=split_scaled,
        scaler=sc,
        windows=windows,
        interp_report=interp,
        out_dir=out_dir,
    )


def preprocess_all(cfg) -> dict[str, PreprocessedDataset]:
    """Run preprocessing for every dataset in the config."""
    out_root = cfg.resolve(cfg.paths.data_processed) / "splits"
    out_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, PreprocessedDataset] = {}
    for name, dspec in cfg.datasets.items():
        results[name] = preprocess_one(
            name=name,
            series_path=cfg.resolve(dspec.processed_file),
            freq=dspec.resample_to,
            lookback=int(cfg.preprocessing.window_length),
            horizons=list(cfg.horizons),
            train_frac=float(cfg.preprocessing.split.train),
            val_frac=float(cfg.preprocessing.split.val),
            test_frac=float(cfg.preprocessing.split.test),
            scaler_name=str(cfg.preprocessing.scaler),
            out_root=out_root,
            interp_method=str(cfg.preprocessing.interpolation),
        )
    return results


def load_preprocessed(name: str, root: str | Path = SPLITS_DIR) -> PreprocessedDataset:
    """Reload a preprocessed dataset previously written by ``preprocess_one``.

    Used by every model in Phases 3-5 so that all models share the exact
    same train/val/test arrays.
    """
    out_dir = Path(root) / name
    manifest = json.loads((out_dir / "manifest.json").read_text())
    series_native = pd.read_parquet(out_dir / "series_native.parquet")
    series_scaled = pd.read_parquet(out_dir / "series_scaled.parquet")
    sc = joblib.load(out_dir / "scaler.joblib")

    # rebuild ChronologicalSplit objects from on-disk slices
    def _slice(df, m):
        return df.loc[(df.index >= pd.Timestamp(m["start"])) & (df.index <= pd.Timestamp(m["end"]))]
    split_native = ChronologicalSplit(
        train=_slice(series_native, manifest["splits"]["train"]),
        val=_slice(series_native, manifest["splits"]["val"]),
        test=_slice(series_native, manifest["splits"]["test"]),
    )
    split_scaled = ChronologicalSplit(
        train=_slice(series_scaled, manifest["splits"]["train"]),
        val=_slice(series_scaled, manifest["splits"]["val"]),
        test=_slice(series_scaled, manifest["splits"]["test"]),
    )

    # rebuild window sets per (h, split)
    windows: dict[int, dict[str, WindowSet]] = {}
    for h in manifest["horizons"]:
        windows[int(h)] = {}
        for split in ["train", "val", "test"]:
            npz = np.load(out_dir / "windows" / f"h{h}_{split}.npz")
            tt = pd.DatetimeIndex(pd.to_datetime(npz["target_times"]))
            windows[int(h)][split] = WindowSet(
                X=npz["X"],
                y=npz["y"],
                target_times=tt,
                split=split,
                horizon=int(npz["horizon"]),
                lookback=int(npz["lookback"]),
            )

    return PreprocessedDataset(
        name=name,
        freq=manifest["freq"],
        lookback=int(manifest["lookback"]),
        horizons=[int(h) for h in manifest["horizons"]],
        series_native=series_native,
        series_scaled=series_scaled,
        split_native=split_native,
        split_scaled=split_scaled,
        scaler=sc,
        windows=windows,
        interp_report=InterpReport(**manifest["interp_report"]),
        out_dir=out_dir,
    )
