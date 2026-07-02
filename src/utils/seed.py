"""Deterministic seeding across numpy / random / torch / cuda."""
from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every library that has a global RNG."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Some operations (e.g. RNN dropout) are non-deterministic by default;
            # this trades a small perf cost for reproducibility — exactly what we
            # want when comparing models fairly.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def derived_seeds(master: int, n: int) -> list[int]:
    """Deterministically derive *n* sub-seeds from a master seed."""
    rng = np.random.default_rng(master)
    return rng.integers(0, 2**31 - 1, size=n).tolist()


def iter_seeds(seeds: Iterable[int]) -> Iterable[int]:
    """Helper that yields seeds and sets them globally."""
    for s in seeds:
        set_global_seed(int(s))
        yield int(s)
