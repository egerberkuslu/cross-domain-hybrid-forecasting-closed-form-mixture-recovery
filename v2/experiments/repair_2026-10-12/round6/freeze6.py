"""Writes RECIPE6_FROZEN.json: the E-Q recipe manifest, the hashes of every earlier-round module that round 6 imports unchanged, the round 6 code hashes and the pool-6 manifest hash. Fits and computes nothing."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "round3"))
sys.path.insert(0, str(HERE.parent / "paper_a"))
sys.path.insert(0, str(HERE.parent / "paper_a" / "round2"))
import eq_recipe as Q  # noqa: E402

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
REPAIR = HERE.parent
m = Q.recipe_manifest()
UNCHANGED = ["paper_a/a2.py", "paper_a/stats_tools.py", "paper_a/rcommon.py", "round3/eq_recipe.py", "round3/RECIPE_FROZEN.json", "round3/r3_stages.py", "round3/r3_monitor.py",
             "round4/r4_monitor.py", "round4/frozen_r4_guard.json", "round5/r5_stages.py", "round5/RECIPE5_FROZEN.json", "fresh_cells/fresh_cells.py", "fresh_cells/build_pool4.py",
             "../fix_2026-10-11/monitor_far.py"]
m["unchanged_from_earlier_rounds"] = {f: sha(REPAIR / f) for f in UNCHANGED}
m["round6_code"] = {f: sha(HERE / f) for f in ("r6_lib.py", "r6_monitor.py", "r6_stages.py", "build_pool6.py", "power6.py", "diag6.py", "classify6.py", "val6.py", "inv6.py", "classify6_r6.py")}
m["round6_outputs"] = {f: sha(HERE / f) for f in ("power6.json", "power6_combined.json", "power6_scenarios.json", "pool6_manifest.json", "reserved_round6_M.json", "dryrun_log.txt")}
m["pool6_manifest_sha256"] = sha(HERE / "pool6_manifest.json")
m["commands"] = {"build_pool6": "cd experiments/repair_2026-10-12/round6 && PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python build_pool6.py",
                 "dry_run": "PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 .chas-venv/bin/python experiments/repair_2026-10-12/round6/r6_monitor.py --stage select --dry-run   (from the repository root)",
                 "pipeline": "bash experiments/repair_2026-10-12/round6/run_r6.sh   (UNTIL=select stops after the select phase)"}
(HERE / "RECIPE6_FROZEN.json").write_text(json.dumps(m, indent=1))
print(json.dumps({k: m[k] for k in ("round6_code", "round6_outputs", "pool6_manifest_sha256")}, indent=1))
