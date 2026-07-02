#!/usr/bin/env bash
# Run the 7 newly added SOTA baselines on the full 5-seed grid.
# Resumable — skips already-completed cells.
set -e
cd "$(dirname "$0")/.."

# Wait for the existing 5-seed grid to finish (if still running)
while pgrep -f "phase5_main.py --skip-hp-search --max-seeds 5" >/dev/null 2>&1; do
  echo "[wait] existing phase5_main still running, sleeping 30s..."
  sleep 30
done

echo "[start] launching new-SOTA grid for nhits, tft, tide, tsmixer, chronos_bolt_zs, moirai_zs, ttm_zs"
exec python3 experiments/phase5_main.py --skip-hp-search --max-seeds 5 \
  --models nhits tft tide tsmixer chronos_bolt_zs moirai_zs ttm_zs \
  2>&1 | tee logs/phase5_new_sota.log
