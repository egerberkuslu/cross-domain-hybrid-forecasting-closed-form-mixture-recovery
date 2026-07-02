"""Strictly-chronological 70 / 15 / 15 train / val / test splitting.

No shuffle, no overlap, train end < val start, val end < test start.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    @property
    def ranges(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        return {
            "train": (self.train.index.min(), self.train.index.max()),
            "val":   (self.val.index.min(),   self.val.index.max()),
            "test":  (self.test.index.min(),  self.test.index.max()),
        }

    def assert_chronological(self) -> None:
        ranges = self.ranges
        assert ranges["train"][1] < ranges["val"][0], (
            f"train end {ranges['train'][1]} must precede val start {ranges['val'][0]}"
        )
        assert ranges["val"][1] < ranges["test"][0], (
            f"val end {ranges['val'][1]} must precede test start {ranges['test'][0]}"
        )


def make_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> ChronologicalSplit:
    """Split a time-indexed frame chronologically by row count.

    The series must already be re-indexed onto a complete uniform grid.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError("Frame must be sorted by time before splitting.")
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.")

    n = len(df)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_test = n - n_train - n_val

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    split = ChronologicalSplit(train=train, val=val, test=test)
    split.assert_chronological()
    logger.info(
        "[split] sizes: train=%d, val=%d, test=%d (frac=%.2f/%.2f/%.2f)",
        n_train, n_val, n_test, train_frac, val_frac, test_frac,
    )
    for name, (lo, hi) in split.ranges.items():
        logger.info("[split] %-5s range: %s -> %s (n=%d)", name, lo, hi, len(getattr(split, name)))
    return split
