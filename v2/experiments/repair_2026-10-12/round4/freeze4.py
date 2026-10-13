"""Writes RECIPE4_FROZEN.json: the round 3 E-Q recipe (unchanged), the chosen guard, the pool-4 manifest and the code hashes. Fits and computes nothing."""
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
m = Q.recipe_manifest()
m["unchanged_from_round3"] = {"eq_recipe.py": sha(HERE.parent / "round3" / "eq_recipe.py"), "RECIPE_FROZEN.json": sha(HERE.parent / "round3" / "RECIPE_FROZEN.json"), "r3_stages.py": sha(HERE.parent / "round3" / "r3_stages.py"),
                              "r3_eval.py": sha(HERE.parent / "round3" / "r3_eval.py"), "r3_monitor.py": sha(HERE.parent / "round3" / "r3_monitor.py")}
m["guard"] = {"file": "round4/frozen_r4_guard.json", "sha256": sha(HERE / "frozen_r4_guard.json"), "chosen": json.loads((HERE / "frozen_r4_guard.json").read_text())["chosen"]}
m["pool4"] = {"manifest": "fresh_cells/pool4_manifest.json", "sha256": sha(HERE.parent / "fresh_cells" / "pool4_manifest.json"), "reserved_series.json": sha(HERE.parent / "fresh_cells" / "reserved_series.json")}
m["round4_code"] = {f: sha(HERE / f) for f in ("r4_guard.py", "r4_stages.py", "r4_eval.py", "r4_monitor.py", "power_calc4.py", "power_guard4.py", "power_monitor4.py")}
(HERE / "RECIPE4_FROZEN.json").write_text(json.dumps(m, indent=1))
print(json.dumps({k: m[k] for k in ("guard", "pool4", "round4_code")}, indent=1))
