#!/bin/bash
# Round 6 monitor pipeline, crash-proof and resumable: stage A -> select (+ hashes into RUN_NOTES_R6.md) -> stage B -> r6_monitor test.
# Re-running skips finished work (valid per-cell files, select.done, summary). Environment:
#   UNTIL=select   stop after the select phase (default: all). Without PREREG_6_COMMITTED the script stops before stage B in any case.
# Launch / relaunch (production, after the coordinator wrote PREREG_6_COMMITTED and PREREG_6_SHA256):
#   systemd-run --user --unit=paper-a-r6 --collect --nice=19 -p CPUAffinity=8-11 --setenv=PATH=/home/ege/anaconda3/bin:/usr/bin:/bin \
#     --working-directory=/home/ege/Desktop/zaman-serisi-yayin/experiments/repair_2026-10-12/round6 bash run_r6.sh
# Same thing without systemd:  nice -n 19 taskset -c 8-11 bash run_r6.sh
# At most 2 workers; waits while MemAvailable < 10 GB; progress line every 5 minutes in round6/PROGRESS.md.
set -u
ROOT=/home/ege/Desktop/zaman-serisi-yayin
RD=$ROOT/experiments/repair_2026-10-12/round6
PY=$ROOT/.chas-venv/bin/python
NOTES=$RD/RUN_NOTES_R6.md
UNTIL=${UNTIL:-all}
MINGB=10
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$ROOT" || exit 1
exec 9>"$RD/run_r6.lock"
flock -n 9 || { echo "run_r6.sh already running"; exit 0; }

STAGEFILE=$RD/.stage
setstage() { echo "$1" > "$STAGEFILE"; }
setstage start
log() { echo "- $(date '+%F %T') $*" >> "$RD/PROGRESS.md"; }
memgb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
memwait() { while [ "$(memgb)" -lt "$MINGB" ]; do log "r6 MemAvailable $(memgb) GB < $MINGB GB, waiting"; sleep 60; done; }
nfiles() { ls "$1" 2>/dev/null | wc -l; }
NCELLS=$("$PY" -I -c "import json;m=json.load(open('$RD/pool6_manifest.json'));print(len(m['monitor_traces_P2'])+len(m['monitor_traces_P3_geant']))")

heartbeat() {
  while true; do
    log "r6 stage=$(cat "$STAGEFILE") r6_val=$(nfiles "$RD/r6_val") r6_models=$(nfiles "$RD/r6_models") r6_test=$(nfiles "$RD/r6_test") of $NCELLS cells, MemAvailable $(memgb) GB"
    sleep 300
  done
}
heartbeat & HB=$!
trap 'kill $HB 2>/dev/null' EXIT

# cell-level bookkeeping: integrity (code hashes), clean (drops invalid or half-written cell files, prints the number of missing series), todo (series with a missing cell, split over workers)
helper() { "$PY" -I - "$@" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
RD = Path("/home/ege/Desktop/zaman-serisi-yayin/experiments/repair_2026-10-12/round6")
cmd = sys.argv[1]
man = json.loads((RD / "pool6_manifest.json").read_text())
names = man["monitor_traces_P2"] + man["monitor_traces_P3_geant"]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz_ok(p):
    try:
        import numpy as np
        with np.load(p) as z:
            for k in z.files:
                z[k]
        return True
    except Exception:
        return False
def cell_ok(stage, n):
    npz = RD / ("r6_val" if stage == "A" else "r6_test") / f"{n}__h6.npz"
    if not npz.exists():
        return False
    ok = npz_ok(npz)
    if ok and stage == "A":
        j = RD / "r6_models" / f"{n}__h6.joblib"
        ok = j.exists() and j.stat().st_size > 0
    return ok
if cmd == "integrity":
    fz = json.loads((RD / "RECIPE6_FROZEN.json").read_text())
    for f, h in fz["round6_code"].items():
        assert sha(RD / f) == h, f"{f} differs from RECIPE6_FROZEN.json"
    for f, h in fz["unchanged_from_earlier_rounds"].items():
        assert sha(RD.parent / f) == h, f"{f} differs from RECIPE6_FROZEN.json"
    assert fz["pool6_manifest_sha256"] == sha(RD / "pool6_manifest.json"), "pool6_manifest.json changed"
    if len(sys.argv) > 2 and sys.argv[2] == "prereg":
        assert (RD / "PREREG_6_COMMITTED").exists(), "PREREG_6_COMMITTED missing"
        assert sha(RD / "REPAIR_PREREG_6.md") == (RD / "PREREG_6_SHA256").read_text().split()[0], "REPAIR_PREREG_6.md changed"
    print("integrity ok; cells", len(names))
elif cmd in ("todo", "clean"):
    stage = sys.argv[2]
    todo = []
    for n in names:
        if not cell_ok(stage, n):
            for p in [RD / ("r6_val" if stage == "A" else "r6_test") / f"{n}__h6.npz", RD / "r6_models" / f"{n}__h6.joblib"]:
                if p.exists() and (stage == "A" or p.suffix == ".npz"):
                    p.unlink()  # half-written
            todo.append(n)
    if cmd == "todo":
        w = int(sys.argv[3])
        print("\n".join(todo[w::2]))
    else:
        print(len(todo))
PYEOF
}

run_workers() {  # $1 = A or B
  local st=$1
  helper clean "$st" > /dev/null
  local pids=()
  for w in 0 1; do
    (
      helper todo "$st" $w | while read -r name; do
        [ -z "$name" ] && continue
        for try in 1 2 3; do
          memwait
          if "$PY" experiments/repair_2026-10-12/round6/r6_stages.py --stage "$st" --cells "$name" >> "$RD/log_stage${st}_$w.txt" 2>&1; then break; fi
          log "r6 stage $st worker $w: $name failed (try $try)"; sleep 20
        done
      done
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
}

setstage integrity
INT=$(helper integrity 2>&1) || { log "r6 INTEGRITY CHECK FAILED, abort: $INT"; exit 1; }
log "r6 start/resume: $INT"

# ---- stage A (validation forecasts, train windows only; REPAIR_STAGE must not be set)
setstage A
unset REPAIR_STAGE
if [ "$(helper clean A)" != "0" ]; then run_workers A; fi
if [ "$(helper clean A)" != "0" ]; then log "r6 stage A incomplete after workers, abort (rerun to resume)"; exit 2; fi
log "r6 stage A complete: r6_val=$(nfiles "$RD/r6_val") r6_models=$(nfiles "$RD/r6_models")"

# ---- select on validation arrays only, freeze, hash into RUN_NOTES_R6.md
setstage select
if [ ! -f "$RD/select.done" ]; then
  nice -n 19 "$PY" "$RD/r6_monitor.py" --stage select --workers 2 > "$RD/log_select_monitor.txt" 2>&1 || { log "r6 monitor select failed, abort"; exit 3; }
  lh() { (cd "$1" && sha256sum $(ls | sort)) | sha256sum | cut -d' ' -f1; }
  {
    echo
    echo "## Round 6 (REPAIR_PREREG_6), $(date '+%F %T')"
    echo "Stage A (train-only, validation forecasts) ran on all $NCELLS monitor cells at h = 6 (P2 and the P3 GEANT stratum). The select stage ran on validation arrays only (the r6_test folder is guarded); frozen before any stage B read:"
    echo "- frozen_r6.json sha256 $(sha256sum "$RD/frozen_r6.json" | cut -d' ' -f1) ($NCELLS traces, arms R5 V1 V2 V3 STB ADB)"
    echo "- pool6_manifest.json sha256 $(sha256sum "$RD/pool6_manifest.json" | cut -d' ' -f1); reserved_round6_M.json sha256 $(sha256sum "$RD/reserved_round6_M.json" | cut -d' ' -f1)"
    echo "- r6_val ($(nfiles "$RD/r6_val") files) sha256 of the sorted per-file sha256sum list: $(lh "$RD/r6_val")"
    echo "- r6_models ($(nfiles "$RD/r6_models") files) sha256 of the sorted per-file sha256sum list: $(lh "$RD/r6_models")"
    echo "- RECIPE6_FROZEN.json sha256 $(sha256sum "$RD/RECIPE6_FROZEN.json" | cut -d' ' -f1)"
    echo "- Commands: systemd-run --user --unit=paper-a-r6 (nice 19, CPUAffinity 8-11), run_r6.sh, OMP_NUM_THREADS=2, two workers, REPAIR_STAGE=test only for stage B and the test stage."
  } >> "$NOTES"
  (cd "$RD/r6_val" && sha256sum $(ls | sort)) > "$RD/R6_VAL_SHA256.txt"
  touch "$RD/select.done"
  log "r6 select done and hashes recorded in round6/RUN_NOTES_R6.md"
fi
if [ "$UNTIL" = "select" ]; then setstage select_done; log "r6 stopped after select (UNTIL=select)"; exit 0; fi

# ---- stage B (test forecasts, refit on train + validation): refuses without the commit marker
if [ ! -f "$RD/PREREG_6_COMMITTED" ]; then log "r6 PREREG_6_COMMITTED missing, stopping before stage B"; exit 7; fi
INT=$(helper integrity prereg 2>&1) || { log "r6 PREREG INTEGRITY CHECK FAILED, abort: $INT"; exit 1; }
setstage B
export REPAIR_STAGE=test
if [ "$(helper clean B)" != "0" ]; then run_workers B; fi
if [ "$(helper clean B)" != "0" ]; then log "r6 stage B incomplete after workers, abort (rerun to resume)"; exit 4; fi
log "r6 stage B complete: r6_test=$(nfiles "$RD/r6_test")"
(cd "$RD/r6_test" && sha256sum $(ls | sort)) > "$RD/R6_TEST_SHA256.txt"

# ---- test stage
setstage monitor_test
if [ ! -f "$RD/r6_monitor_summary.json" ]; then
  nice -n 19 "$PY" "$RD/r6_monitor.py" --stage test --workers 2 > "$RD/log_monitor_test.txt" 2>&1 || { log "r6 monitor test failed, abort"; exit 6; }
fi
setstage done
touch "$RD/ALL_DONE"
log "r6 ALL DONE: r6_monitor_summary.json written"
