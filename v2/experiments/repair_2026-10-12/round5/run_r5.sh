#!/bin/bash
# Round 5 phase 2, crash-proof and resumable: stage A -> select (+ hashes into RUN_NOTES) -> stage B -> r5_eval test -> r5_monitor test.
# Re-running skips finished work (valid per-cell files, select.done, summaries). Launch / relaunch (see RESUME_NOW.md):
#   systemd-run --user --unit=paper-a-r5 --collect --nice=19 -p CPUAffinity=8-11 --setenv=PATH=/home/ege/anaconda3/bin:/usr/bin:/bin --setenv=REPAIR_STAGE=test \
#     --working-directory=/home/ege/Desktop/zaman-serisi-yayin/experiments/repair_2026-10-12/round5 bash run_r5.sh
# At most 2 model workers; waits while MemAvailable < 10 GB; progress line every 5 minutes in round5/PROGRESS.md.
set -u
ROOT=/home/ege/Desktop/zaman-serisi-yayin
RD=$ROOT/experiments/repair_2026-10-12/round5
PY=$ROOT/.chas-venv/bin/python
NOTES=$ROOT/experiments/repair_2026-10-12/paper_a/RUN_NOTES.md
PREREG_SHA=2856d807b7402150022391741601c0f6d06d7b0c8e730530dacf6b74e14a39e3
MINGB=10
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "$ROOT" || exit 1
exec 9>"$RD/run_r5.lock"
flock -n 9 || { echo "run_r5.sh already running"; exit 0; }

STAGE=start
STAGEFILE=$RD/.stage
setstage() { STAGE=$1; echo "$1" > "$STAGEFILE"; }
setstage start
log() { echo "- $(date '+%F %T') $*" >> "$RD/PROGRESS.md"; }
memgb() { awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
memwait() { while [ "$(memgb)" -lt "$MINGB" ]; do log "r5 MemAvailable $(memgb) GB < $MINGB GB, waiting"; sleep 60; done; }
nfiles() { ls "$1" 2>/dev/null | wc -l; }

heartbeat() {
  while true; do
    log "r5 stage=$(cat "$STAGEFILE") r5_val=$(nfiles "$RD/r5_val") r5_models=$(nfiles "$RD/r5_models") r5_test=$(nfiles "$RD/r5_test") of 584 cells, MemAvailable $(memgb) GB"
    sleep 300
  done
}
heartbeat & HB=$!
trap 'kill $HB 2>/dev/null' EXIT

# cell-level bookkeeping: verify (drops invalid or half-written cell files), todo (series with a missing cell, alternating split over workers)
helper() { "$PY" -I - "$@" <<'PYEOF'
import hashlib, json, sys, zipfile
from pathlib import Path
RD = Path("/home/ege/Desktop/zaman-serisi-yayin/experiments/repair_2026-10-12/round5")
FC = RD.parent / "fresh_cells"
cmd = sys.argv[1]
man = json.loads((FC / "pool5_manifest.json").read_text())
H = [1, 3, 6, 12, 24]
cells = []
for s in man["series"]:
    if s["accuracy"]:
        cells += [(s["name"], h) for h in H]
    elif s["name"] in man["monitor_traces"]:
        cells.append((s["name"], 6))
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
def cell_ok(stage, n, h):
    npz = RD / ("r5_val" if stage == "A" else "r5_test") / f"{n}__h{h}.npz"
    if not npz.exists():
        return False
    ok = npz_ok(npz)
    if ok and stage == "A":
        j = RD / "r5_models" / f"{n}__h{h}.joblib"
        ok = j.exists() and j.stat().st_size > 0
    return ok
if cmd == "integrity":
    assert sha(RD / "REPAIR_PREREG_5.md") == sys.argv[2], "REPAIR_PREREG_5.md changed"
    assert (RD / "PREREG_5_COMMITTED").exists(), "PREREG_5_COMMITTED missing"
    fz = json.loads((RD / "RECIPE5_FROZEN.json").read_text())
    for f, h in fz["round5_code"].items():
        assert sha(RD / f) == h, f"{f} differs from RECIPE5_FROZEN.json"
    print("integrity ok; cells", len(cells))
elif cmd in ("todo", "clean"):
    stage = sys.argv[2]
    names = []
    for n, h in cells:
        if not cell_ok(stage, n, h):
            for p in [RD / ("r5_val" if stage == "A" else "r5_test") / f"{n}__h{h}.npz", RD / "r5_models" / f"{n}__h{h}.joblib"]:
                if p.exists() and (stage == "A" or p.suffix == ".npz"):
                    p.unlink()  # half-written
            if n not in names:
                names.append(n)
    if cmd == "todo":
        w = int(sys.argv[3])
        print("\n".join(names[w::2]))
    else:
        print(len(names))
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
          if "$PY" experiments/repair_2026-10-12/round5/r5_stages.py --stage "$st" --cells "$name" >> "$RD/log_stage${st}_$w.txt" 2>&1; then break; fi
          log "r5 stage $st worker $w: $name failed (try $try)"; sleep 20
        done
      done
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
}

setstage integrity
INT=$(helper integrity "$PREREG_SHA" 2>&1) || { log "r5 INTEGRITY CHECK FAILED, abort: $INT"; exit 1; }
log "r5 phase 2 start/resume: $INT"

# ---- stage A (validation forecasts, train windows only; REPAIR_STAGE must not be set)
setstage A
if [ "$(helper clean A)" != "0" ]; then
  unset REPAIR_STAGE
  run_workers A
fi
if [ "$(helper clean A)" != "0" ]; then log "r5 stage A incomplete after workers, abort (rerun the unit to resume)"; exit 2; fi
log "r5 stage A complete: r5_val=$(nfiles "$RD/r5_val") r5_models=$(nfiles "$RD/r5_models")"

# ---- select on validation arrays only, freeze, hash into RUN_NOTES
setstage select
if [ ! -f "$RD/select.done" ]; then
  unset REPAIR_STAGE
  nice -n 19 "$PY" "$RD/r5_eval.py" --stage select > "$RD/log_select_eval.txt" 2>&1 || { log "r5 eval select failed, abort"; exit 3; }
  nice -n 19 "$PY" "$RD/r5_monitor.py" --stage select > "$RD/log_select_monitor.txt" 2>&1 || { log "r5 monitor select failed, abort"; exit 3; }
  lh() { (cd "$1" && sha256sum $(ls | sort)) | sha256sum | cut -d' ' -f1; }
  {
    echo
    echo "## Round 5 (PREREG_5 commit 6f617a4), $(date '+%F %T')"
    echo "Stage A (train-only, validation forecasts) ran on all 584 cells (520 accuracy, 64 monitor at h = 6). Select stages ran on validation arrays only; frozen before any stage B read:"
    echo "- frozen_r5.json sha256 $(sha256sum "$RD/frozen_r5.json" | cut -d' ' -f1) (guard G2_0.25, 520 cells)"
    echo "- frozen_r5_monitor.json sha256 $(sha256sum "$RD/frozen_r5_monitor.json" | cut -d' ' -f1) (64 traces)"
    echo "- frozen_r4_guard.json sha256 $(sha256sum "$ROOT/experiments/repair_2026-10-12/round4/frozen_r4_guard.json" | cut -d' ' -f1)"
    echo "- r5_val ($(nfiles "$RD/r5_val") files) sha256 of the sorted per-file sha256sum list: $(lh "$RD/r5_val")"
    echo "- r5_models ($(nfiles "$RD/r5_models") files) sha256 of the sorted per-file sha256sum list: $(lh "$RD/r5_models")"
    echo "- REPAIR_PREREG_5.md sha256 $PREREG_SHA; RECIPE5_FROZEN.json sha256 $(sha256sum "$RD/RECIPE5_FROZEN.json" | cut -d' ' -f1)"
    echo "- Commands: systemd-run --user --unit=paper-a-r5 (nice 19, CPUAffinity 8-11), run_r5.sh, OMP_NUM_THREADS=2, two workers, REPAIR_STAGE=test only for stage B and the test stages."
  } >> "$NOTES"
  touch "$RD/select.done"
  log "r5 select done and hashes recorded in paper_a/RUN_NOTES.md"
fi

# ---- stage B (test forecasts, refit on train + validation)
setstage B
export REPAIR_STAGE=test
if [ "$(helper clean B)" != "0" ]; then run_workers B; fi
if [ "$(helper clean B)" != "0" ]; then log "r5 stage B incomplete after workers, abort (rerun the unit to resume)"; exit 4; fi
log "r5 stage B complete: r5_test=$(nfiles "$RD/r5_test")"

# ---- test stages
setstage eval_test
if [ ! -f "$RD/r5_test_summary.json" ]; then
  nice -n 19 "$PY" "$RD/r5_eval.py" --stage test > "$RD/log_eval_test.txt" 2>&1 || { log "r5 eval test failed, abort"; exit 5; }
fi
setstage monitor_test
if [ ! -f "$RD/r5_monitor_summary.json" ]; then
  nice -n 19 "$PY" "$RD/r5_monitor.py" --stage test > "$RD/log_monitor_test.txt" 2>&1 || { log "r5 monitor test failed, abort"; exit 6; }
fi
setstage done
touch "$RD/ALL_DONE"
log "r5 ALL DONE: r5_test_summary.json and r5_monitor_summary.json written"
