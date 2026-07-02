"""Phase-level timing & provenance tracker.

Every experiment phase (checkpoint regen, Phase 6, ablation, Stage A
experiments, ...) calls ``PhaseTimer`` as a context manager. It:

  * records start / end timestamps + wall-clock duration
  * snapshots the git commit hash (if in a git repo)
  * snapshots ``config/config.yaml`` content
  * collects user-supplied notes / outputs
  * appends one JSON record per phase to ``outputs/runlog.jsonl``

This file is the **audit trail** reviewers need: "show me when training
happened, with what config, and how long it took."  Combined with the
per-run ``outputs/metrics/*.json`` (which already records ``fit_seconds``
and ``predict_seconds`` per cell) it gives full provenance.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_PATH = Path("outputs/runlog.jsonl")


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2
        )
        return r.stdout.strip() or None
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=2
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def _snapshot_config(path: str = "config/config.yaml") -> dict:
    try:
        p = Path(path)
        if not p.exists():
            return {}
        import yaml

        return yaml.safe_load(p.read_text()) or {}
    except Exception:
        return {}


@dataclass
class PhaseTimer:
    """Context manager that times a pipeline phase and persists provenance.

    Usage:
        from src.utils.runlog import PhaseTimer
        with PhaseTimer("phase6_v3", notes="ablation + DM-test") as t:
            run_ablation_v3()
            t.add_output("ablation_csv", "outputs/eval_v3/tables/...")
    """

    phase: str
    notes: str = ""
    extra: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    log_path: Path = LOG_PATH

    # internal
    _t0: float = 0.0
    _start_iso: str = ""
    _end_iso: str = ""

    def __enter__(self) -> "PhaseTimer":
        self._t0 = time.perf_counter()
        self._start_iso = datetime.now(timezone.utc).isoformat()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        elapsed = time.perf_counter() - self._t0
        self._end_iso = datetime.now(timezone.utc).isoformat()
        record = {
            "phase": self.phase,
            "start_utc": self._start_iso,
            "end_utc": self._end_iso,
            "elapsed_seconds": float(elapsed),
            "elapsed_human": f"{elapsed/60:.1f} min"
            if elapsed >= 60
            else f"{elapsed:.1f} s",
            "notes": self.notes,
            "host": socket.gethostname(),
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "config_snapshot": _snapshot_config(),
            "outputs": self.outputs,
            "extra": self.extra,
            "status": "fail" if exc_type is not None else "ok",
        }
        if exc_type is not None:
            record["error_type"] = exc_type.__name__
            record["error_message"] = str(exc_val)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return False  # don't suppress exceptions

    def add_output(self, key: str, value: Any) -> None:
        self.outputs[key] = value

    def add_extra(self, key: str, value: Any) -> None:
        self.extra[key] = value


def list_phases(log_path: str | Path = LOG_PATH) -> list[dict]:
    """Read all phase records from runlog.jsonl."""
    p = Path(log_path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def summarize_runlog(log_path: str | Path = LOG_PATH) -> str:
    """Render a human-readable summary of all logged phases."""
    rows = list_phases(log_path)
    if not rows:
        return "(empty runlog)"
    lines = [f"{'PHASE':<32} {'STATUS':<6} {'ELAPSED':>10} {'WHEN':<26} NOTES"]
    lines.append("-" * 120)
    for r in rows:
        lines.append(
            f"{r['phase'][:32]:<32} {r['status']:<6} "
            f"{r['elapsed_human']:>10} {r['start_utc'][:19]:<26} {r['notes'][:40]}"
        )
    return "\n".join(lines)
