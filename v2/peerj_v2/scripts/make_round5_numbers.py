#!/usr/bin/env python3
"""Writes numbers_round5.tex (LaTeX macros) and numbers_round5.json from the repair round 5 result files.

Inputs (experiments/repair_2026-10-12/round5): r5_test_summary.json, r5_monitor_summary.json, r5_test_cells.csv,
power_calc5.json, power_monitor5.json; round 4 reference values from round4/r4_test_summary.json and r4_monitor_summary.json.
Nothing is typed by hand: every number in the round 5 text of main.tex, highlights.tex and cover_letter.tex is a macro from this file.
Macro names carry letters only (LaTeX forbids digits in control sequences), prefix rv.
Run: python3 scripts/make_round5_numbers.py   (from peerj_v2)"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # v2/peerj_v2 in this repository
V2 = ROOT.parent  # v2/, which holds experiments/repair_2026-10-12
R5 = V2 / "experiments" / "repair_2026-10-12" / "round5"
R4 = V2 / "experiments" / "repair_2026-10-12" / "round4"
e5 = json.loads((R5 / "r5_test_summary.json").read_text())
m5 = json.loads((R5 / "r5_monitor_summary.json").read_text())
pw = json.loads((R5 / "power_calc5.json").read_text())
pm = json.loads((R5 / "power_monitor5.json").read_text())
e4 = json.loads((R4 / "r4_test_summary.json").read_text())
cells = pd.read_csv(R5 / "r5_test_cells.csv")
cells["src"] = cells.series.str.replace(r"\d.*$", "", regex=True).str.rstrip("_")

M = {}  # macro name -> value
RAW = {}


def put(name, value, text):
    M[name] = text
    RAW[name] = value


def pct(x, d=3):
    """Fraction to percent, d decimals, hyphen minus (macros are used inside math mode)."""
    return f"{100 * x:.{d}f}"


def num(x, d=3):
    return f"{x:.{d}f}"


# structure of the pool
put("rvCells", e5["cells"], str(e5["cells"]))
put("rvSeries", e5["series"], str(e5["series"]))
put("rvK", e5["K_wins_required"], str(e5["K_wins_required"]))
put("rvWins", e5["cells_win_primary"], str(e5["cells_win_primary"]))
put("rvWinsHalf", e5["cells_win_at_0.5pct"], str(e5["cells_win_at_0.5pct"]))
put("rvLoss", e5["cells_loss"], str(e5["cells_loss"]))
put("rvIdent", e5["cells_identical"], str(e5["cells_identical"]))
nid = ~cells.identical_to_bolt
put(
    "rvBetter",
    int((cells.gain_vs_bolt[nid] > 0).sum()),
    str(int((cells.gain_vs_bolt[nid] > 0).sum())),
)
put(
    "rvWorse",
    int((cells.gain_vs_bolt[nid] < 0).sum()),
    str(int((cells.gain_vs_bolt[nid] < 0).sum())),
)
put("rvMinPrimary", e5["min_gain_primary"], pct(e5["min_gain_primary"], 3))
put(
    "rvMinSecondary", 0.005, "0.5"
)  # secondary minimum of the registration (REPAIR_PREREG_5.md section 4)
put("rvMedian", e5["median_cell_gain"], pct(e5["median_cell_gain"], 2))

# pooled gain over Chronos-Bolt
pb = e5["pooled_gain_vs_bolt"]
put("rvPooled", pb["mean"], pct(pb["mean"]))
put("rvPooledCellLo", pb["ci95_cell"][0], pct(pb["ci95_cell"][0]))
put("rvPooledCellHi", pb["ci95_cell"][1], pct(pb["ci95_cell"][1]))
put(
    "rvPooledSerLo", pb["ci95_series_clustered"][0], pct(pb["ci95_series_clustered"][0])
)
put(
    "rvPooledSerHi", pb["ci95_series_clustered"][1], pct(pb["ci95_series_clustered"][1])
)

# controls
for tag, key in (
    ("A", "control_a_permuted_rows"),
    ("B", "control_b_null_expert"),
    ("Best", "round4_comparison_best_of_two_controls_reported_only"),
):
    c = e5[key]
    put(f"rvCtrl{tag}Mean", c["mean"], pct(c["mean"]))
    put(f"rvCtrl{tag}Lo", c["lo"], pct(c["lo"]))
    put(f"rvCtrl{tag}Hi", c["hi"], pct(c["hi"]))
put(
    "rvCtrlHolm",
    e5["control_holm_adjusted_p"][0],
    f"{e5['control_holm_adjusted_p'][0]:.4f}",
)
put(
    "rvCtrlHolmB",
    e5["control_holm_adjusted_p"][1],
    f"{e5['control_holm_adjusted_p'][1]:.4f}",
)
assert e5["controls_ok"] and e5["success"] and e5["success_secondary_0.5pct"]
g = cells.groupby("series")[
    ["gain_vs_ctrl_a", "gain_vs_ctrl_b", "gain_vs_best_ctrl"]
].mean()
put(
    "rvNegSerA",
    int((g.gain_vs_ctrl_a < 0).sum()),
    str(int((g.gain_vs_ctrl_a < 0).sum())),
)
put(
    "rvNegSerB",
    int((g.gain_vs_ctrl_b < 0).sum()),
    str(int((g.gain_vs_ctrl_b < 0).sum())),
)
# Gain of each random control over Chronos-Bolt itself (mean over cells of 1 - RMSE(control) / RMSE(Bolt)),
# on the same cell-mean definition as the pooled gain of the mixture. A negative value means the control is worse than Bolt.
cva = float((1 - cells.rmse_ctrl_a / cells.rmse_bolt).mean())
cvb = float((1 - cells.rmse_ctrl_b / cells.rmse_bolt).mean())
put("rvCtrlAvsBolt", cva, pct(abs(cva), 2))  # absolute value, control a is worse than Bolt
put("rvCtrlBvsBolt", cvb, pct(abs(cvb), 2))  # absolute value, control b is better than Bolt
assert cva < 0 < cvb

# per source
for tag, key in (("Cesnet", "cesnet_inst"), ("Geant", "geant_od")):
    s = e5["per_source"][key]
    put(f"rv{tag}Cells", s["cells"], str(s["cells"]))
    put(f"rv{tag}Series", s["cells"] // 5, str(s["cells"] // 5))
    put(f"rv{tag}Wins", s["wins"], str(s["wins"]))
    put(f"rv{tag}Mean", s["mean_gain"], pct(s["mean_gain"], 2))
    put(f"rv{tag}Median", s["median_gain"], pct(s["median_gain"], 2))
    put(f"rv{tag}CtrlA", s["mean_gain_vs_ctrl_a"], pct(s["mean_gain_vs_ctrl_a"], 2))
    put(f"rv{tag}CtrlB", s["mean_gain_vs_ctrl_b"], pct(s["mean_gain_vs_ctrl_b"], 2))
    assert (cells.src == key).sum() == s["cells"]

# power statements of the registration
a104 = pw["res"]["A104"]
put("rvPowEmpPrim", a104["EMP_primary"]["power"], f"{a104['EMP_primary']['power']:.2f}")
put(
    "rvPowHalfPrim",
    a104["HALF_primary"]["power"],
    f"{a104['HALF_primary']['power']:.2f}",
)
put(
    "rvPowEmpSec",
    a104["EMP_secondary"]["power"],
    f"{a104['EMP_secondary']['power']:.2f}",
)

# monitor (descriptive)
put("rvMonTraces", m5["traces"], str(m5["traces"]))
put("rvMonNeed", m5["need"], str(m5["need"]))
for c, k in (("One", "c1"), ("Two", "c2"), ("Three", "c3")):
    for grp, gname in (("All", "all"), ("Aby", "abilene"), ("Ces", "cesnet")):
        d = m5["descriptive"][gname][k]
        put(f"rvMon{c}{grp}N", d["n"], str(d["n"]))
        put(f"rvMon{c}{grp}Pass", d["pass"], str(d["pass"]))
        put(f"rvMon{c}{grp}Rate", d["pass"] / d["n"], f"{100 * d['pass'] / d['n']:.1f}")
        put(f"rvMon{c}{grp}Lo", d["wilson95"][0], f"{100 * d['wilson95'][0]:.1f}")
        put(f"rvMon{c}{grp}Hi", d["wilson95"][1], f"{100 * d['wilson95'][1]:.1f}")
assert (
    m5["cond1_fa_le5"] == m5["descriptive"]["all"]["c1"]["pass"] and not m5["success"]
)
r = pm["res"]
put(
    "rvPowMonTwelve",
    r["12_round4_design"]["POST"]["power"],
    f"{r['12_round4_design']['POST']['power']:.2f}",
)
put(
    "rvPowMonAll",
    r["64_all_F"]["POST"]["power"],
    f"{r['64_all_F']['POST']['power']:.2f}",
)
put(
    "rvPowMonNineAll",
    r["64_all_F"]["P90"]["power"],
    f"{r['64_all_F']['P90']['power']:.2f}",
)

# constants fixed by the registration text (REPAIR_PREREG_5.md sections 2 and 5)
put("rvPowMonPNine", 0.90, "0.90")
put("rvMonFrac", 0.8, "80")
put("rvMonMinDec", 690, "690")

# round 4 reference values, each under its own rule, none re-scored
put("rvFourCells", e4["cells"], str(e4["cells"]))
put("rvFourWins", e4["cells_win"], str(e4["cells_win"]))
put("rvFourK", e4["K_wins_required"], str(e4["K_wins_required"]))
put(
    "rvFourPooled",
    e4["pooled_gain_vs_bolt"]["mean"],
    pct(e4["pooled_gain_vs_bolt"]["mean"]),
)
put(
    "rvFourBestSerLo",
    e4["pooled_gain_vs_best_control"]["ci95_series_clustered"][0],
    pct(e4["pooled_gain_vs_best_control"]["ci95_series_clustered"][0]),
)
put(
    "rvFourBestMean",
    e4["pooled_gain_vs_best_control"]["mean"],
    pct(e4["pooled_gain_vs_best_control"]["mean"], 2),
)

lines = [
    "% Written by peerj_v2/scripts/make_round5_numbers.py from experiments/repair_2026-10-12/round5 (r5_test_summary.json, r5_monitor_summary.json, r5_test_cells.csv, power_*5.json) and round4/r4_test_summary.json. Do not edit by hand.",
]
for k in sorted(M):
    lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
(ROOT / "numbers_round5.tex").write_text("\n".join(lines) + "\n")
(ROOT / "numbers_round5.json").write_text(json.dumps(RAW, indent=1, sort_keys=True))
print(f"{len(M)} macros written to numbers_round5.tex")
