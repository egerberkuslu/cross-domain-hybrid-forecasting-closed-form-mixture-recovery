"""Phase-2 preprocessing: missing-value handling, splits, scaling, windowing.

All Phase-2 outputs are deterministic given ``config/config.yaml`` and the
cached Phase-1 parquet files, and are written to ``data/processed/splits/``
so that every model in Phases 3-5 re-uses identical (train, val, test)
arrays — guaranteeing the comparison is fair.
"""
from .interpolation import handle_missing
from .splits import ChronologicalSplit, make_splits
from .scaling import fit_scaler_on_train, transform
from .windowing import make_sliding_windows, WindowSet
from .pipeline import (
    PreprocessedDataset,
    preprocess_one,
    preprocess_all,
    load_preprocessed,
    SPLITS_DIR,
)

__all__ = [
    "handle_missing",
    "ChronologicalSplit",
    "make_splits",
    "fit_scaler_on_train",
    "transform",
    "make_sliding_windows",
    "WindowSet",
    "PreprocessedDataset",
    "preprocess_one",
    "preprocess_all",
    "load_preprocessed",
    "SPLITS_DIR",
]
