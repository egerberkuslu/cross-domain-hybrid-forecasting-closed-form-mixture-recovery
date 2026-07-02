"""Q1 Stage-C orchestrator — runs after Stage A (orchestrator + 3.1
wrapper) finishes.  Executes the reviewer-defense items that require
GPU and that depend on completed Stage-A training:

  1. v4_fix : re-train v4 with held-out early stopping (Defense 6a)
  2. 6b qualitative α(x) viz (needs trained v4 weights)
  3. 6c cross-cell α(x) transfer matrix (Defense 6c)
  4. v3-vs-v4 vs v4_fix comparison
  5. Rebuild paper tables

Phase-level timing recorded into outputs/runlog.jsonl as usual.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

from src.utils.runlog import PhaseTimer

LOG_DIR = Path("outputs/logs")


def _run(label: str, cmd: list[str] | str, notes: str = "") -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"q1_{label}.log"
    if isinstance(cmd, str):
        cmd_str = cmd
    else:
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n>>> {label}")
    print(f"    {cmd_str}")
    ok = True
    with PhaseTimer(label, notes=notes) as timer:
        timer.add_extra("cmd", cmd_str)
        timer.add_output("log", str(log_path))
        try:
            with open(log_path, "w") as fh:
                proc = subprocess.run(
                    cmd,
                    shell=isinstance(cmd, str),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            timer.add_extra("returncode", proc.returncode)
            if proc.returncode != 0:
                ok = False
        except Exception as e:
            timer.add_extra("error", f"{type(e).__name__}: {e}")
            ok = False
    print(f"    {'OK' if ok else 'FAILED'}")
    return ok


def wait_for_pgrep(pattern: str, label: str, poll_s: int = 60, max_s: int = 9000):
    print(f"\n>>> wait: {label}")
    t0 = time.perf_counter()
    while True:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if not r.stdout.strip():
            print(f"    none after {(time.perf_counter()-t0):.0f}s")
            return
        if time.perf_counter() - t0 > max_s:
            print(f"    timeout {max_s}s")
            return
        time.sleep(poll_s)


def main():
    # 0. Wait for Stage A orchestrator + 3.1 wrapper to fully finish
    wait_for_pgrep("run_q1_stage_a.py", "Stage A orchestrator", poll_s=60, max_s=14400)
    wait_for_pgrep(
        "expA_3_1_cross_dataset.py", "Stage A Exp 3.1", poll_s=60, max_s=7200
    )
    # And ensure no phase5 v3_* variants still running
    wait_for_pgrep(
        "phase5_main.py.*cha_hybrid_v3_",
        "any remaining v3 size/STL phase5",
        poll_s=60,
        max_s=7200,
    )

    # 1. v4_fix training: 125 cells, ~40 min GPU
    _run(
        "phase5_v4_fix",
        [
            sys.executable,
            "pipeline/phase5_main.py",
            "--skip-hp-search",
            "--max-seeds",
            "5",
            "--models",
            "cha_hybrid_v4_fix",
            "--force",
        ],
        notes="Defense 6a: v4 with held-out early stopping",
    )

    # 2. v3 vs v4 vs v4_fix head-to-head comparison
    _run(
        "expA_v3_v4_compare_after_fix",
        [sys.executable, "pipeline/expA_v3_v4_compare.py"],
        notes="v3 vs v4 vs v4_fix after all variants trained",
    )

    # 3. Qualitative α(x) viz (needs v4 ckpts)
    _run(
        "expC_6b_alpha_qualitative",
        [sys.executable, "pipeline/expC_6b_alpha_qualitative.py"],
        notes="Defense 6b: per-window α(x) qualitative",
    )

    # 4. Cross-cell α(x) transfer matrix
    _run(
        "expC_6c_cross_cell_alpha",
        [sys.executable, "pipeline/expC_6c_cross_cell_alpha.py"],
        notes="Defense 6c: cross-cell α(x) transfer",
    )

    # 5. Refresh per-window v3-vs-Bolt analysis with new predictions
    _run(
        "expC_5a_per_window_final",
        [sys.executable, "pipeline/expC_5a_per_window.py"],
        notes="Defense 5a: refresh per-window v3-vs-Bolt analysis",
    )

    _run(
        "expC_5c_other_mixtures_final",
        [sys.executable, "pipeline/expC_5c_other_mixtures.py"],
        notes="Defense 5c: refresh mixture function comparison",
    )

    # 6. Rebuild all paper tables
    _run(
        "build_paper_tables_final",
        [sys.executable, "pipeline/build_paper_tables.py"],
        notes="rebuild paper/tables/*.tex after Stage C",
    )

    print("\nStage C orchestrator done. Check outputs/runlog.jsonl for timings.")


if __name__ == "__main__":
    main()
