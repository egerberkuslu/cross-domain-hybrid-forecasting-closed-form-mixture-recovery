"""Writes RECIPE5_FROZEN.json: the round 3 E-Q recipe and the round 4 guard (both unchanged), the pool-5 manifest and the code hashes. Fits and computes nothing."""
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
R3, R4, FC = HERE.parent / "round3", HERE.parent / "round4", HERE.parent / "fresh_cells"
m = Q.recipe_manifest()
m["unchanged_from_round3"] = {f: sha(R3 / f) for f in ("eq_recipe.py", "RECIPE_FROZEN.json", "r3_stages.py", "r3_eval.py", "r3_monitor.py")}
m["unchanged_from_round4"] = {f: sha(R4 / f) for f in ("frozen_r4_guard.json", "r4_guard.py", "r4_stages.py", "r4_eval.py", "r4_monitor.py", "RECIPE4_FROZEN.json")}
m["guard"] = {"file": "round4/frozen_r4_guard.json", "sha256": sha(R4 / "frozen_r4_guard.json"), "chosen": json.loads((R4 / "frozen_r4_guard.json").read_text())["chosen"]}
m["pool5"] = {"manifest": "fresh_cells/pool5_manifest.json", "sha256": sha(FC / "pool5_manifest.json"), "reserved_series.json": sha(FC / "reserved_series.json"),
              "partition_F.json": sha(FC / "partition_F.json"), "build_pool5.py": sha(FC / "build_pool5.py"), "fresh_cells.py": sha(FC / "fresh_cells.py")}
m["round5_code"] = {f: sha(HERE / f) for f in ("r5_stages.py", "r5_eval.py", "r5_monitor.py", "power_calc5.py", "power_monitor5.py", "r4_data_check.py", "freeze5.py")}
m["round5_power"] = {f: sha(HERE / f) for f in ("power_calc5.json", "power_monitor5.json", "r4_data_check.txt")}
m["commands"] = {"r4_data_check.txt": "cd experiments/repair_2026-10-12/round5 && PYTHONDONTWRITEBYTECODE=1 ../../../.chas-venv/bin/python r4_data_check.py > r4_data_check.txt 2>/dev/null",
             "power_calc5.json": "PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 8-11 .chas-venv/bin/python experiments/repair_2026-10-12/round5/power_calc5.py", "power_monitor5.json": "PYTHONDONTWRITEBYTECODE=1 .chas-venv/bin/python experiments/repair_2026-10-12/round5/power_monitor5.py"}
(HERE / "RECIPE5_FROZEN.json").write_text(json.dumps(m, indent=1))
print(json.dumps({k: m[k] for k in ("guard", "pool5", "round5_code", "round5_power")}, indent=1))
