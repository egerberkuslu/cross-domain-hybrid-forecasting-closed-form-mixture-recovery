"""Q1 Stage-A orchestrator.

Executes the full Stage A roadmap end-to-end with phase-level timing and
reproducibility provenance recorded into ``outputs/runlog.jsonl``.

Order
-----
0. Wait for any in-flight v3 checkpoint regeneration to complete
1. Phase 6 v3  — ablation + DM-test + cost on baseline v3 cells
2. Phase 5 size-scaling variants  (cha_hybrid_v3_{tiny,mini,base})
3. Phase 5 STL-period variants    (cha_hybrid_v3_stl{12,48,168})
4. Exp 1.4   inference latency benchmark
5. Exp 1.3   per-sample interpretability visualisation
6. Phase 7   figures + tables (refreshed)

Each step writes a record to ``outputs/runlog.jsonl`` (start/end times,
git commit, config snapshot, status, outputs).  If any step fails, the
record is still written and orchestration continues with the next
independent step.
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
    """Run a command, tee output to logs/, capture timing via PhaseTimer."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"q1_{label}.log"
    if isinstance(cmd, str):
        cmd_str = cmd
    else:
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n>>> {label}")
    print(f"    {cmd_str}")
    print(f"    log → {log_path}")
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


def wait_for_pgrep(
    pattern: str, label: str, poll_s: int = 30, max_s: int = 9000
) -> None:
    """Block until no process matches the pattern."""
    print(f"\n>>> wait: {label} (pattern={pattern!r})")
    t0 = time.perf_counter()
    while True:
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if not r.stdout.strip():
            print(
                f"    no matching process found after {(time.perf_counter()-t0):.0f}s"
            )
            return
        if time.perf_counter() - t0 > max_s:
            print(f"    timeout after {max_s}s, continuing anyway")
            return
        print(f"    still running ({(time.perf_counter()-t0):.0f}s)…")
        time.sleep(poll_s)


def main():
    # 0. Wait for current v3 ckpt regen (if any)
    wait_for_pgrep(
        "phase5_main.py.*cha_hybrid_v3",
        "v3 checkpoint regeneration",
        poll_s=45,
        max_s=4500,
    )

    # 1. Phase 6 v3 evaluation (ablation + DM-test + cost)
    _run(
        "phase6_v3",
        [sys.executable, "pipeline/phase6_eval_v3.py"],
        notes="ablation v3 + DM-test small-sample + cost analysis",
    )

    # 2. Size-scaling variants of v3 (tiny / mini / base global expert)
    for tag in ("tiny", "mini", "base"):
        _run(
            f"phase5_v3_{tag}",
            [
                sys.executable,
                "pipeline/phase5_main.py",
                "--skip-hp-search",
                "--max-seeds",
                "5",
                "--models",
                f"cha_hybrid_v3_{tag}",
                "--force",
            ],
            notes=f"size-scaling: chronos-bolt-{tag} as global expert",
        )

    # 3. STL-period sensitivity (period = 12, 48, 168)
    for p in (12, 48, 168):
        _run(
            f"phase5_v3_stl{p}",
            [
                sys.executable,
                "pipeline/phase5_main.py",
                "--skip-hp-search",
                "--max-seeds",
                "5",
                "--models",
                f"cha_hybrid_v3_stl{p}",
                "--force",
            ],
            notes=f"STL period sensitivity: period={p}",
        )

    # 4. Inference-latency benchmark (CPU vs GPU)
    _run(
        "expA_1_4_latency",
        [sys.executable, "pipeline/expA_1_4_latency.py"],
        notes="CPU vs GPU inference latency",
    )

    # 5. Per-sample interpretability visualisation
    _run(
        "expA_1_3_per_sample",
        [sys.executable, "pipeline/expA_1_3_per_sample_viz.py"],
        notes="decomp-only vs global-only vs hybrid sample plots",
    )

    # 6. Final figures + tables (re-run with extended data)
    _run(
        "phase7_figures",
        [sys.executable, "pipeline/phase7_figures.py"],
        notes="final publication figures + tables",
    )

    print("\nStage A orchestrator done. Check outputs/runlog.jsonl for timings.")


if __name__ == "__main__":
    main()
