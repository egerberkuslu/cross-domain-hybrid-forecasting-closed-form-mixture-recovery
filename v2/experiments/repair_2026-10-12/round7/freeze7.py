"""Writes RECIPE7_FROZEN.json: the E-Q recipe manifest, the hashes of every earlier-round module that round 7 imports unchanged, the round 7 code hashes, the round 7 output hashes
and the pool-7 manifest hash. Fits and computes nothing. Run after the select stage and the dry run (the registration lists its sha256)."""
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
             "round6/r6_lib.py", "round6/RECIPE6_FROZEN.json", "../fix_2026-10-11/monitor_far.py"]
m["unchanged_from_earlier_rounds"] = {f: sha(REPAIR / f) for f in UNCHANGED}
CODE = ("r7_lib.py", "r7_monitor.py", "r7_stages.py", "classify7.py", "build_pool7.py", "freeze7.py", "power7.py", "run_r7.sh", "download7.sh", "alt7_counts.py",
        "diag7.py", "diag7_extra.py", "val7.py", "data7_inventory.py")
m["round7_code"] = {f: sha(HERE / f) for f in CODE}
OUT = ("power7.json", "pool7_manifest.json", "reserved_round7.json", "degen7.json", "frozen_r7.json", "dryrun_log.txt", "alt7_counts.json", "DIAGNOSIS7.md", "DATA7.md",
       "val7_summary.json", "val7_w4_summary.json", "R7_VAL_SHA256.txt")
m["round7_outputs"] = {f: sha(HERE / f) for f in OUT}
m["pool7_manifest_sha256"] = sha(HERE / "pool7_manifest.json")
m["zenodo_files_sha256"] = sha(HERE.parents[2] / "data" / "raw" / "cesnet_ts24_extra" / "SHA256SUMS.txt")
m["commands"] = {"download": "bash experiments/repair_2026-10-12/round7/download7.sh   (curl, md5 checked against the Zenodo record API)",
                 "build_pool7": "cd experiments/repair_2026-10-12/round7 && PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python build_pool7.py",
                 "dry_run": "PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 .chas-venv/bin/python experiments/repair_2026-10-12/round7/r7_monitor.py --stage select --dry-run   (from the repository root)",
                 "pipeline": "UNTIL=select nice -n 19 taskset -c 12-15 bash experiments/repair_2026-10-12/round7/run_r7.sh   (without UNTIL: all stages, stage B only after PREREG_7_COMMITTED)"}
(HERE / "RECIPE7_FROZEN.json").write_text(json.dumps(m, indent=1))
print(json.dumps({k: m[k] for k in ("round7_code", "round7_outputs", "pool7_manifest_sha256")}, indent=1))
