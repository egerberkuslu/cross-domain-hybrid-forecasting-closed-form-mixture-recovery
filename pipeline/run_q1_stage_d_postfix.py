"""Stage D — post-Stage-C fixup phase.

Re-runs the latency benchmark with the fixed WindowSet slicing,
runs build_appendix_runlog one final time so the appendix is fresh,
and refreshes all paper tables.  Fires after Stage C orchestrator
finishes.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

from src.utils.runlog import PhaseTimer

LOG_DIR = Path("outputs/logs")


def _run(label: str, cmd: list[str], notes: str = "") -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"q1_{label}.log"
    print(f">>> {label}")
    ok = True
    with PhaseTimer(label, notes=notes) as t:
        t.add_extra("cmd", " ".join(shlex.quote(c) for c in cmd))
        try:
            with open(log_path, "w") as fh:
                p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            t.add_extra("returncode", p.returncode)
            ok = p.returncode == 0
        except Exception as e:
            t.add_extra("error", f"{type(e).__name__}: {e}")
            ok = False
        t.add_output("log", str(log_path))
    print(f"    {'OK' if ok else 'FAILED'}")
    return ok


def wait_for_pgrep(pattern: str, label: str, poll_s: int = 60, max_s: int = 14400):
    print(f">>> wait: {label}")
    t0 = time.perf_counter()
    while True:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if not r.stdout.strip():
            return
        if time.perf_counter() - t0 > max_s:
            return
        time.sleep(poll_s)


def main():
    wait_for_pgrep("run_q1_stage_c.py", "Stage C orchestrator")
    wait_for_pgrep(
        "phase5_main.py.*cha_hybrid_v4_fix", "v4_fix phase5", poll_s=60, max_s=7200
    )

    _run(
        "expA_1_4_latency_retry",
        [sys.executable, "pipeline/expA_1_4_latency.py"],
        notes="re-run of latency benchmark with WindowSet slicing fix",
    )
    _run(
        "expC_v3_v4_v4fix",
        [sys.executable, "pipeline/expC_v3_v4_v4fix.py"],
        notes="3-way head-to-head v3 vs v4 vs v4_fix",
    )
    _run(
        "expA_2_x_sensitivity",
        [sys.executable, "pipeline/expA_2_x_sensitivity.py"],
        notes="refresh STL + foundation-size sensitivity tables",
    )
    _run(
        "phase7_figures_final",
        [sys.executable, "pipeline/phase7_figures.py"],
        notes="regenerate fig01-08 with full grid (v3/v4/v4_fix + size + stl variants)",
    )
    _run(
        "build_paper_tables_post",
        [sys.executable, "pipeline/build_paper_tables.py"],
        notes="refresh paper/tables/*.tex after Stage C+D",
    )
    _run(
        "build_appendix_runlog_post",
        [sys.executable, "pipeline/build_appendix_runlog.py"],
        notes="refresh paper/tables/appendix_runlog.tex",
    )

    print("Stage D done — paper auto-fill complete.")


if __name__ == "__main__":
    main()
