#!/usr/bin/env bash
# v2: recompute the Paper A numbers on the EXISTING predictions.
# CPU only, no retraining, no .tex edits. Outputs land in this folder and are
# never overwritten (a timestamped sibling is written if a name exists).
#
# Usage (from anywhere):
#   bash v2/experiments/fix_2026-10-11/run_all.sh            # full run incl. step 0
#   SKIP_VAL=1 bash v2/experiments/fix_2026-10-11/run_all.sh # moments/fallback paths only
# Env: PY (default python3), THREADS (default 4).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"  # repository root (holds outputs/)
cd "$ROOT"
PY="${PY:-python3}"
THREADS="${THREADS:-4}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES=""   # CPU only, keeps the GPU free
export PYTHONHASHSEED=0

mkdir -p "$HERE/logs"
TS=$(date +%Y%m%d-%H%M%S)
LOG="$HERE/logs/run_all_$TS.log"
echo "[run_all] $TS  PY=$PY  THREADS=$THREADS" | tee -a "$LOG"
"$PY" -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'pandas', pandas.__version__)" 2>&1 | tee -a "$LOG"

declare -a FAILED=()
run() {
  local name="$1"; shift
  local t0=$(date +%s)
  echo "=== [$name] start $(date +%T)" | tee -a "$LOG"
  if "$PY" "$HERE/$name" "$@" >>"$LOG" 2>&1; then
    echo "=== [$name] ok ($(( $(date +%s) - t0 )) s)" | tee -a "$LOG"
  else
    echo "=== [$name] FAILED ($(( $(date +%s) - t0 )) s), see $LOG" | tee -a "$LOG"
    FAILED+=("$name")
  fi
}

# 0. validation-split expert forecasts (torch/darts/chronos, CPU; ~30-60 min)
if [[ -z "${SKIP_VAL:-}" ]]; then
  run val_expert_forecasts.py --device cpu --check-test 42
else
  echo "=== [val_expert_forecasts.py] skipped (SKIP_VAL=1): steps 5-7 use moment/fallback paths" | tee -a "$LOG"
fi
# 1-4: test-split only, independent of step 0
run closed_form_stats.py                      # items 1 (floor count), 4, Abilene ratio, NAB estimators
run dm_tost_identity.py --B 2000 --block h --generation   # DM identities, BH/Holm, TOST identities, generation-sweep tables
run permutation_two_sided.py --n-perm 10000 --block h --generation
run bootstrap_pairing.py --B 1000
# 5-7: use val_forecasts/ when present, otherwise documented fallbacks
run stacked_ols.py
run degradation_oos.py --horizons 6 --replicates 20
run monitor_far.py --horizon 6 --inj-seeds 10

echo "[run_all] done $(date +%T); failed: ${FAILED[*]:-none}" | tee -a "$LOG"
[[ ${#FAILED[@]} -eq 0 ]]
