#!/bin/bash
# Round 7 monitor pipeline, crash-proof and resumable: stage A (+ the validation-only degenerate-expert queue) -> select (+ hashes into RUN_NOTES_R7.md) -> stage B -> r7_monitor test.
# Re-running skips finished work (valid per-cell files, select.done, summary). Environment:
#   UNTIL=A        stop after stage A and the degenerate-expert queue
#   UNTIL=select   stop after the select phase (default: all). Without PREREG_7_COMMITTED the script stops before stage B in any case.
# Launch / relaunch:   nice -n 19 taskset -c 12-15 bash run_r7.sh
# (production: systemd-run --user --unit=paper-a-r7 --collect --nice=19 -p CPUAffinity=12-15 --setenv=PATH=/home/ege/anaconda3/bin:/usr/bin:/bin \
#     --working-directory=/home/ege/Desktop/zaman-serisi-yayin/experiments/repair_2026-10-12/round7 bash run_r7.sh)
# At most 2 workers; refuses to start (and waits while running) below MINGB GB MemAvailable; progress line every 5 minutes in round7/PROGRESS.md. No git.
set -u
ROOT=/home/ege/Desktop/zaman-serisi-yayin
RD=$ROOT/experiments/repair_2026-10-12/round7
PY=$ROOT/.chas-venv/bin/python
NOTES=$RD/RUN_NOTES_R7.md
UNTIL=${UNTIL:-all}
MINGB=8
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT" || exit 1
exec 9>"$RD/run_r7.lock"
flock -n 9 || { echo "run_r7.sh already running"; exit 0; }

STAGEFILE=$RD/.stage
setstage() { echo "$1" > "$STAGEFILE"; }
setstage start
log() { echo "- $(date '+%F %T') $*" >> "$RD/PROGRESS.md"; }
memgb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
memwait() { while [ "$(memgb)" -lt "$MINGB" ]; do log "r7 MemAvailable $(memgb) GB < $MINGB GB, waiting"; sleep 60; done; }
nfiles() { ls "$1" 2>/dev/null | wc -l; }
if [ "$(memgb)" -lt "$MINGB" ]; then log "r7 REFUSES to start: MemAvailable $(memgb) GB < $MINGB GB"; echo "MemAvailable below $MINGB GB, refusing to start"; exit 1; fi
ncells() { "$PY" -I -c "
import json
from pathlib import Path
rd=Path('$RD')
if (rd/'degen7.json').exists() and json.load(open(rd/'degen7.json'))['complete']:
    d=json.load(open(rd/'degen7.json')); print(len(d['selected']['sub'])+len(d['selected']['ip'])+len(d['g7']))
else:
    m=json.load(open(rd/'pool7_manifest.json')); print(len(m['monitor_primary']['sub'])+len(m['monitor_primary']['ip'])+len(m['G7_traces']))
"; }

heartbeat() {
  while true; do
    log "r7 stage=$(cat "$STAGEFILE") r7_val=$(nfiles "$RD/r7_val") r7_models=$(nfiles "$RD/r7_models") r7_test=$(nfiles "$RD/r7_test") of $(ncells) cells, MemAvailable $(memgb) GB"
    sleep 300
  done
}
( exec 9>&-; heartbeat ) & HB=$!
trap 'kill $HB 2>/dev/null' EXIT

# cell bookkeeping: integrity (code hashes, after the freeze), clean (drops invalid or half-written cell files, prints the number of cells still missing), todo (cells to run, split over workers)
helper() { "$PY" -I - "$@" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
RD = Path("/home/ege/Desktop/zaman-serisi-yayin/experiments/repair_2026-10-12/round7")
cmd = sys.argv[1]
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
    npz = RD / ("r7_val" if stage == "A" else "r7_test") / f"{n}__h6.npz"
    if not npz.exists():
        return False
    ok = npz_ok(npz)
    if ok and stage == "A":
        j = RD / "r7_models" / f"{n}__h6.joblib"
        ok = j.exists() and j.stat().st_size > 0
    return ok
def cells(stage):
    if stage == "A":
        p = RD / "stageA_queue.json"
        return json.loads(p.read_text())["needed"] if p.exists() else []
    return json.loads((RD / "frozen_r7.json").read_text())["traces"]
if cmd == "integrity":
    if not (RD / "RECIPE7_FROZEN.json").exists():
        print("pre-freeze run (RECIPE7_FROZEN.json not written yet)")
        sys.exit(0)
    fz = json.loads((RD / "RECIPE7_FROZEN.json").read_text())
    for f, h in fz["round7_code"].items():
        assert sha(RD / f) == h, f"{f} differs from RECIPE7_FROZEN.json"
    for f, h in fz["unchanged_from_earlier_rounds"].items():
        assert sha(RD.parent / f) == h, f"{f} differs from RECIPE7_FROZEN.json"
    assert fz["pool7_manifest_sha256"] == sha(RD / "pool7_manifest.json"), "pool7_manifest.json changed"
    if len(sys.argv) > 2 and sys.argv[2] == "prereg":
        assert sha(RD / "frozen_r7.json") == (RD / "frozen_r7.sha256").read_text().strip(), "frozen_r7.json changed"
        frz = json.loads((RD / "frozen_r7.json").read_text())
        assert frz["degen7_sha256"] == sha(RD / "degen7.json"), "degen7.json changed"
        assert frz["pool7_manifest_sha256"] == sha(RD / "pool7_manifest.json"), "pool7_manifest.json changed"
        for f, h in frz["code_sha256"].items():
            assert sha(RD / f) == h, f"{f} changed since the freeze"
        assert (RD / "PREREG_7_COMMITTED").exists(), "PREREG_7_COMMITTED missing"
        assert sha(RD / "REPAIR_PREREG_7.md") == (RD / "PREREG_7_SHA256").read_text().split()[0], "REPAIR_PREREG_7.md changed"
    print("integrity ok")
elif cmd in ("todo", "clean"):
    stage = sys.argv[2]
    todo = []
    for n in cells(stage):
        if not cell_ok(stage, n):
            for p in [RD / ("r7_val" if stage == "A" else "r7_test") / f"{n}__h6.npz", RD / "r7_models" / f"{n}__h6.joblib"]:
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
          if "$PY" experiments/repair_2026-10-12/round7/r7_stages.py --stage "$st" --cells "$name" >> "$RD/log_stage${st}_$w.txt" 2>&1; then break; fi
          log "r7 stage $st worker $w: $name failed (try $try)"; sleep 20
        done
      done
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
}

setstage integrity
INT=$(helper integrity 2>&1) || { log "r7 INTEGRITY CHECK FAILED, abort: $INT"; exit 1; }
log "r7 start/resume: $INT"

# ---- stage A (validation forecasts, train windows only; REPAIR_STAGE must not be set) with the validation-only degenerate-expert queue
setstage A
unset REPAIR_STAGE
for round in 1 2 3 4 5 6 7 8; do
  "$PY" "$RD/r7_monitor.py" --stage degen >> "$RD/log_degen.txt" 2>&1 || { log "r7 degen step failed, abort"; exit 3; }
  if [ "$(helper clean A)" = "0" ]; then break; fi
  run_workers A
done
if [ "$(helper clean A)" != "0" ]; then log "r7 stage A incomplete after workers, abort (rerun to resume)"; exit 2; fi
"$PY" "$RD/r7_monitor.py" --stage degen >> "$RD/log_degen.txt" 2>&1
log "r7 stage A complete: r7_val=$(nfiles "$RD/r7_val") r7_models=$(nfiles "$RD/r7_models") $(grep -c . "$RD/degen7.json" >/dev/null && echo "degen7.json written")"

if [ "$UNTIL" = "A" ]; then setstage A_done; log "r7 stopped after stage A (UNTIL=A)"; exit 0; fi

# ---- select on validation arrays only, freeze, hash into RUN_NOTES_R7.md
setstage select
if [ ! -f "$RD/select.done" ]; then
  nice -n 19 "$PY" "$RD/r7_monitor.py" --stage select --workers 2 > "$RD/log_select_monitor.txt" 2>&1 || { log "r7 monitor select failed, abort"; exit 3; }
  lh() { (cd "$1" && sha256sum $(ls | sort)) | sha256sum | cut -d' ' -f1; }
  {
    echo
    echo "## Round 7 (REPAIR_PREREG_7), $(date '+%F %T')"
    echo "Stage A (train-only, validation forecasts) ran on all monitor cells at h = 6 (P7 and the G7 GEANT stratum); the degenerate-expert rule (validation forecasts only) fixed the final list in degen7.json; the select stage ran on validation arrays only (the r7_test folder is guarded and absent); frozen before any stage B read:"
    echo "- frozen_r7.json sha256 $(sha256sum "$RD/frozen_r7.json" | cut -d' ' -f1)"
    echo "- degen7.json sha256 $(sha256sum "$RD/degen7.json" | cut -d' ' -f1); pool7_manifest.json sha256 $(sha256sum "$RD/pool7_manifest.json" | cut -d' ' -f1); reserved_round7.json sha256 $(sha256sum "$RD/reserved_round7.json" | cut -d' ' -f1)"
    echo "- r7_val ($(nfiles "$RD/r7_val") files) sha256 of the sorted per-file sha256sum list: $(lh "$RD/r7_val")"
    echo "- r7_models ($(nfiles "$RD/r7_models") files) sha256 of the sorted per-file sha256sum list: $(lh "$RD/r7_models")"
    echo "- Commands: run_r7.sh (nice 19, taskset 12-15, OMP_NUM_THREADS=1, two workers, REPAIR_STAGE=test only for stage B and the test stage)."
  } >> "$NOTES"
  (cd "$RD/r7_val" && sha256sum $(ls | sort)) > "$RD/R7_VAL_SHA256.txt"
  touch "$RD/select.done"
  log "r7 select done and hashes recorded in round7/RUN_NOTES_R7.md"
fi
if [ "$UNTIL" = "select" ]; then setstage select_done; log "r7 stopped after select (UNTIL=select)"; exit 0; fi

# ---- stage B (test forecasts, refit on train + validation): refuses without the commit marker
if [ ! -f "$RD/PREREG_7_COMMITTED" ]; then log "r7 PREREG_7_COMMITTED missing, stopping before stage B"; exit 7; fi
INT=$(helper integrity prereg 2>&1) || { log "r7 PREREG INTEGRITY CHECK FAILED, abort: $INT"; exit 1; }
setstage B
export REPAIR_STAGE=test
if [ "$(helper clean B)" != "0" ]; then run_workers B; fi
if [ "$(helper clean B)" != "0" ]; then log "r7 stage B incomplete after workers, abort (rerun to resume)"; exit 4; fi
log "r7 stage B complete: r7_test=$(nfiles "$RD/r7_test")"
(cd "$RD/r7_test" && sha256sum $(ls | sort)) > "$RD/R7_TEST_SHA256.txt"

# ---- test stage
setstage monitor_test
if [ ! -f "$RD/r7_monitor_summary.json" ]; then
  nice -n 19 "$PY" "$RD/r7_monitor.py" --stage test --workers 2 > "$RD/log_monitor_test.txt" 2>&1 || { log "r7 monitor test failed, abort"; exit 6; }
fi
setstage done
touch "$RD/ALL_DONE"
log "r7 ALL DONE: r7_monitor_summary.json written"
