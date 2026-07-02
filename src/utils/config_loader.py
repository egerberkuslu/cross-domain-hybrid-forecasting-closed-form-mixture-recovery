"""Strongly-typed loader for `config/config.yaml`.

The loader returns a `ProjectConfig` namespace-style object so the rest of
the codebase can do `cfg.preprocessing.window_length` instead of nested
`dict.get(...)` calls (which silently swallow typos).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class _AttrDict(dict):
    """Dict that exposes keys as attributes (recursively)."""

    def __init__(self, mapping: dict | None = None) -> None:
        super().__init__()
        if mapping is None:
            mapping = {}
        for k, v in mapping.items():
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)


@dataclass
class ProjectConfig:
    """Light wrapper around the parsed yaml plus repo paths."""

    raw: _AttrDict
    config_path: Path
    repo_root: Path

    def __getattr__(self, item: str) -> Any:
        # Delegate attribute access to the underlying parsed yaml.
        return getattr(self.raw, item)

    def resolve(self, rel_path: str | os.PathLike) -> Path:
        """Resolve a path string from config (relative to repo root) to an absolute Path."""
        p = Path(rel_path)
        return p if p.is_absolute() else (self.repo_root / p).resolve()


def load_config(path: str | os.PathLike | None = None) -> ProjectConfig:
    """Load the project YAML config and return a ``ProjectConfig`` instance."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        parsed = yaml.safe_load(f) or {}
    return ProjectConfig(raw=_AttrDict(parsed), config_path=config_path, repo_root=REPO_ROOT)
