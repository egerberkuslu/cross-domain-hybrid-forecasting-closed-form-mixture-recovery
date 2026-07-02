"""Small IO helpers shared across the pipeline (atomic writes, dirs, JSON)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)
    tmp.replace(p)
    return p


def read_json(path: str | os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
