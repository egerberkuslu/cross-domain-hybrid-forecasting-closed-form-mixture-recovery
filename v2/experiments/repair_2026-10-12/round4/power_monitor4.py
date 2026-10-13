"""Round 4 monitor power from the round-3 per-trace results (RF on 12 drifting fresh traces: false-alarm <= 5 % on 11, shift-specific on 12, above the permuted clock on 9).
Per-trace success probabilities p1 (false alarm), p2 (shift-specific), p3 (above the control). Rule: each condition on at least ceil(0.8 n) of n traces. Conditions are treated as independent
across conditions (the product is a lower-bound-style approximation). p1 and p2 use the round 3 proportions (Jeffreys-smoothed); p3 is unknown for the NEW placebo control,
so a grid is shown. Output power_monitor4.json."""
import json
import numpy as np
from scipy.stats import binom

n = 12
need = int(np.ceil(0.8 * n))
p1, p2 = (11 + 0.5) / (12 + 1), (12 + 0.5) / (12 + 1)
out = {"n_traces": n, "need": need, "p1": p1, "p2": p2, "grid": {}}
for p3 in (0.75, 0.85, 0.90, 0.95, 1.0):
    pw = [1 - binom.cdf(need - 1, n, p) for p in (p1, p2, p3)]
    out["grid"][str(p3)] = {"p_C1": float(pw[0]), "p_C2": float(pw[1]), "p_C3": float(pw[2]), "power": float(np.prod(pw))}
    print(p3, {k: round(v, 3) for k, v in out["grid"][str(p3)].items()})
json.dump(out, open("power_monitor4.json", "w"), indent=1)
