#!/usr/bin/env python3
"""Writes numbers_round6.tex (LaTeX macros), numbers_round6.json and tables/monitor_round6.tex from the repair round 6 result files.

Inputs (experiments/repair_2026-10-12/round6): r6_monitor_summary.json, r6_causes.json, power6_scenarios.json, val6_summary.json,
pool6_manifest.json, inv6_summary.json, inv6_unallocated_{cesnet,abilene,geant}.csv; round 4 reference values from round4/r4_monitor_summary.json.
Nothing is typed by hand in the round 6 text of main.tex, highlights.tex and cover_letter.tex: every number is a macro from this file.
Constants fixed by the registration text (REPAIR_PREREG_6.md, DIAGNOSIS.md) are put() below with their source.
Macro names carry letters only (LaTeX forbids digits in control sequences), prefix rsm.
Run: python3 scripts/make_round6_numbers.py   (from peerj_v2)"""
import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # v2/peerj_v2 in this repository
V2 = ROOT.parent  # v2/, which holds experiments/repair_2026-10-12
R6 = V2 / "experiments" / "repair_2026-10-12" / "round6"
R4 = V2 / "experiments" / "repair_2026-10-12" / "round4"
ms = json.loads((R6 / "r6_monitor_summary.json").read_text())
ca = json.loads((R6 / "r6_causes.json").read_text())
pw = json.loads((R6 / "power6_scenarios.json").read_text())
vs = json.loads((R6 / "val6_summary.json").read_text())
pm = json.loads((R6 / "pool6_manifest.json").read_text())
inv = json.loads((R6 / "inv6_summary.json").read_text())
m4 = json.loads((R4 / "r4_monitor_summary.json").read_text())

P2 = ms["P2"]
P3 = ms["P3_geant_descriptive"]
assert not ms["pseudo"] and P2["success"] is False and P2["succeeding_variants"] == []
assert P2["traces"] == 20 and P2["need"] == 16 and P3["traces"] == 18

M = {}
RAW = {}


def put(name, value, text):
    M[name] = text
    RAW[name] = value


def hu(x, d=2):
    """Round half up on the shortest decimal form of x (0.605 -> 0.61), not on its binary value."""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def pct(x, d=1):
    return hu(100 * x, d)


ARM = {
    "R5": "Rfive",
    "V1": "Vone",
    "V2": "Vtwo",
    "V3": "Vthree",
    "STB": "Stb",
    "ADB": "Adb",
}
COND = {"c1": "COne", "c2": "CTwo", "c3": "CThree", "joint": "J"}
GRP = {"all": "All", "abilene": "Aby", "cesnet": "Ces", "P2_new": "New", "P2_FG": "Fg"}

# pool structure
c = pm["counts"]
put("rsmTraces", P2["traces"], str(P2["traces"]))
put("rsmNeed", P2["need"], str(P2["need"]))
put("rsmCes", c["P2_by_source"]["cesnet"], str(c["P2_by_source"]["cesnet"]))
put("rsmAby", c["P2_by_source"]["abilene"], str(c["P2_by_source"]["abilene"]))
put("rsmNew", c["P2_new"], str(c["P2_new"]))
put("rsmFg", c["P2_FG"], str(c["P2_FG"]))
put("rsmGeant", c["P3_geant"], str(c["P3_geant"]))
put("rsmCand", c["candidates"], str(c["candidates"]))
assert c["P2_new"] + c["P2_FG"] == P2["traces"] == c["P2_total"]
rules = pm["rules"]
put(
    "rsmMissCes",
    rules["missing_max"]["cesnet"],
    f"{100 * rules['missing_max']['cesnet']:.0f}",
)
put(
    "rsmMissAby",
    rules["missing_max"]["abilene"],
    f"{100 * rules['missing_max']['abilene']:.0f}",
)
put(
    "rsmMissGeant",
    rules["missing_max"]["geant"],
    f"{100 * rules['missing_max']['geant']:.0f}",
)
put("rsmZeroMax", rules["zero_max"], f"{100 * rules['zero_max']:.0f}")
put("rsmDriftSd", rules["drift_sd"], f"{rules['drift_sd']:.1f}")
put(
    "rsmMinWinLong",
    rules["min_test_windows_h6"]["cesnet"],
    str(rules["min_test_windows_h6"]["cesnet"]),
)

# arms: conditions by group, Wilson intervals
for a, tag in ARM.items():
    arm = P2["arms"][a]
    put(f"rsm{tag}PBinom", arm["p_binom_vs_0.5"], f"{arm['p_binom_vs_0.5']:.4f}")
    put(f"rsm{tag}Feas", arm["feasible"], str(arm["feasible"]))
    put(f"rsm{tag}GridMax", arm["n"] - arm["feasible"], str(arm["n"] - arm["feasible"]))
    put(
        f"rsm{tag}Rnd",
        arm["rand_ctrl_c3_share_mean"],
        pct(arm["rand_ctrl_c3_share_mean"], 1),
    )
    put(f"rsm{tag}Det", arm["mean_det"], pct(arm["mean_det"], 1))
    for cname, ctag in COND.items():
        for gname, gtag in GRP.items():
            if gname == "all":
                d = arm["conditions"][cname]
            elif gname in ("abilene", "cesnet"):
                d = arm["by_source"][gname][cname]
            else:
                d = arm["by_stratum"][gname][cname]
            put(f"rsm{tag}{ctag}{gtag}N", d["n"], str(d["n"]))
            put(f"rsm{tag}{ctag}{gtag}Pass", d["pass"], str(d["pass"]))
            put(
                f"rsm{tag}{ctag}{gtag}Rate",
                d["pass"] / d["n"],
                pct(d["pass"] / d["n"], 1),
            )
            put(f"rsm{tag}{ctag}{gtag}Lo", d["wilson95"][0], pct(d["wilson95"][0], 1))
            put(f"rsm{tag}{ctag}{gtag}Hi", d["wilson95"][1], pct(d["wilson95"][1], 1))
    # descriptive round 4/5 placebo-clock rule on events
    put(
        f"rsm{tag}Old",
        arm["descriptive_round5_rule_events"]["c3_round5_rule"],
        str(arm["descriptive_round5_rule_events"]["c3_round5_rule"]),
    )
holm = {P2["arms"][v]["p_holm"] for v in ("V1", "V2", "V3")}
assert len(holm) == 1
hv = holm.pop()
put("rsmHolm", hv, f"{hv:.3f}")
put(
    "rsmRndMax",
    max(P2["arms"][v]["rand_ctrl_c3_share_mean"] for v in ("V1", "V2", "V3")),
    pct(max(P2["arms"][v]["rand_ctrl_c3_share_mean"] for v in ("V1", "V2", "V3")), 1),
)
put("rsmRndLimit", 0.05, "5")
put("rsmAlpha", 0.05, "0.05")
put("rsmHalf", 0.5, "0.5")
assert all(P2["arms"][v]["X"] == 15 for v in ("V1", "V2", "V3"))
assert P2["arms"]["V1"]["X"] == P2["need"] - 1

# GEANT descriptive stratum
for a, tag in ARM.items():
    d = P3["arms"][a]["conditions"]["joint"]
    put(f"rsm{tag}GeantPass", d["pass"], str(d["pass"]))
    put(f"rsm{tag}GeantN", d["n"], str(d["n"]))

# causes (registered classes, P2 only)
CL = {
    "c1_false_alarm_excess": "COne",
    "c2_not_shift_specific": "CTwo",
    "c3_null_exceeded": "CThree",
    "c3_slow_detection": "Slow",
}
for a, tag in ARM.items():
    e = ca[f"{a}_P2"]
    put(f"rsm{tag}Failed", e["failed"], str(e["failed"]))
    for k, t in CL.items():
        put(f"rsm{tag}Cause{t}", e[k], str(e[k]))
failed = {
    v: {n for k in CL for n in ca[f"{v}_P2"]["names"][k]} for v in ("V1", "V2", "V3")
}
shared = failed["V1"] & failed["V2"] & failed["V3"]
put("rsmShared", len(shared), str(len(shared)))
put(
    "rsmSharedCes",
    sum(n.startswith("cesnet") for n in shared),
    str(sum(n.startswith("cesnet") for n in shared)),
)
put(
    "rsmSharedAby",
    sum(n.startswith("abilene") for n in shared),
    str(sum(n.startswith("abilene") for n in shared)),
)
cs = [set(ca[f"{v}_P2"]["names"]["c2_not_shift_specific"]) for v in ("V1", "V2", "V3")]
put("rsmCTwoShared", len(cs[0] & cs[1] & cs[2]), str(len(cs[0] & cs[1] & cs[2])))
assert all(n.startswith("cesnet") for n in cs[0] & cs[1] & cs[2])
# names of the traces that fail for all three variants (registered cause rule, r6_causes.json)
ids = sorted((n.split("_")[1] for n in shared if n.startswith("cesnet")), key=lambda t: int(t[4:]))
ids = [t[4:] for t in ids]
put("rsmSharedCesIds", ids, ", ".join(ids[:-1]) + " and " + ids[-1])
aby_ids = sorted(n.split("_")[1][2:] for n in shared if n.startswith("abilene"))
assert len(aby_ids) == 1
put("rsmSharedAbyId", aby_ids[0], aby_ids[0])

# power scenarios of the registration
for a, tag in (("V1", "Vone"), ("V2", "Vtwo"), ("V3", "Vthree")):
    s = pw[a]
    put(
        f"rsmPow{tag}Hom",
        s["power_homogeneous_revealed_rate"],
        hu(s["power_homogeneous_revealed_rate"], 2),
    )
    put(
        f"rsmPow{tag}Split",
        s["power_feasible_infeasible_split"],
        hu(s["power_feasible_infeasible_split"], 2),
    )
    put(
        f"rsmPow{tag}Down",
        s["power_split_and_two_traces_fail"],
        hu(s["power_split_and_two_traces_fail"], 2),
    )
    put(
        f"rsmPow{tag}Drop",
        s["power_infeasible_traces_dropped"],
        hu(s["power_infeasible_traces_dropped"], 2),
    )
    put(
        f"rsmRate{tag}",
        s["revealed_pass_rate_all64"],
        hu(s["revealed_pass_rate_all64"], 3),
    )
# range of the probability of a miss over the registered power scenarios, from the printed powers (1 - 0.91 = 0.09)
printed = [
    float(hu(pw[v][k], 2))
    for v in ("V1", "V2", "V3")
    for k in (
        "power_homogeneous_revealed_rate",
        "power_feasible_infeasible_split",
        "power_split_and_two_traces_fail",
    )
]
put("rsmMissLo", 1 - max(printed), hu(1 - max(printed), 2))
put("rsmMissHi", 1 - min(printed), hu(1 - min(printed), 2))
down = pw["note"]["fail_traces"]
nd = sum(1 for n in down if all(n in failed[v] for v in ("V1", "V2", "V3")))
put("rsmDownNamed", len(down), str(len(down)))
put("rsmDownFailed", nd, str(nd))
assert (
    nd == 1
    and "cesnet_inst159" in failed["V1"]
    and "cesnet_inst217" not in (failed["V1"] | failed["V2"] | failed["V3"])
)

# revealed round 5 re-analysis (motivation only)
for a, tag in ARM.items():
    r = vs[f"revealed_r5_test|r5|{a}"]
    put(f"rsmRev{tag}Pass", r["joint"], str(r["joint"]))
put(
    "rsmRevN",
    vs["revealed_r5_test|r5|V1"]["traces"],
    str(vs["revealed_r5_test|r5|V1"]["traces"]),
)

# round 4 monitor reference, under its own rule
put("rsmFourN", m4["traces"], str(m4["traces"]))
put("rsmFourNeed", m4["need"], str(m4["need"]))
put("rsmFourCOne", m4["cond1_fa_le5"], str(m4["cond1_fa_le5"]))
put("rsmFourCTwo", m4["cond2_shift_specific"], str(m4["cond2_shift_specific"]))
put("rsmFourCThree", m4["cond3_above_placebo"], str(m4["cond3_above_placebo"]))

# inventory: what remains unallocated after round 6
cand = {(s["source"], str(s["id"])) for s in pm["series"]}
for src, tag in (("cesnet", "Ces"), ("abilene", "Aby"), ("geant", "Geant")):
    rows = list(csv.DictReader(open(R6 / f"inv6_unallocated_{src}.csv")))
    s = inv[src]
    assert (
        s["unallocated_total"] == len(rows)
        and s["unallocated_admissible"] == 0
        and not any(r["admissible"] == "True" for r in rows)
    )
    drawn = sum(1 for r in rows if (src, r["id"]) in cand)
    put(f"rsmInv{tag}Unalloc", len(rows), str(len(rows)))
    put(
        f"rsmInv{tag}Adm", s["unallocated_admissible"], str(s["unallocated_admissible"])
    )
    put(f"rsmInv{tag}Drawn", drawn, str(drawn))
    put(f"rsmInv{tag}Left", len(rows) - drawn, str(len(rows) - drawn))

# constants fixed by the registration text (REPAIR_PREREG_6.md sections 0 to 4, DIAGNOSIS.md section 2 and 3)
put("rsmCapOld", 100, "100")  # cap of the round 4 and 5 condition 3
put("rsmCapNew", 150, "150")  # one cap of the corrected condition 3'
put("rsmClocks", 5, "5")  # placebo clocks of rounds 4 and 5
put(
    "rsmEffClocks", 2.3, "2.3"
)  # effective number of independent placebo observations, DIAGNOSIS.md 2b
put("rsmFaBudget", 5, "5")  # condition 1 false-alarm budget in percent
put("rsmMargin", 0.5, "0.5")  # margin of c3'
put("rsmNullTarget", 10, "10")  # null target in percent
put("rsmGridMax", 60, "60")  # level grid maximum
put("rsmWarm", 200, "200")  # warm-start decisions of V3
put("rsmKGrid", None, r"\{0.5,\,1,\,2\}")  # allowance grid of V1 to V3 and the baseline
put(
    "rsmFgH", None, r"1, 6 and 24"
)  # horizons of the Chronos-Bolt test forecasts scored by the companion replication
put("rsmGeantWin", 428, "428")  # GEANT test windows at h = 6
put("rsmAbyWin", 699, "699")  # Abilene test windows at h = 6
put("rsmGeantVal", 281, "281")  # GEANT validation windows at h = 6
put("rsmBurn", 100, "100")  # burn-in decisions

lines = [
    "% Written by peerj_v2/scripts/make_round6_numbers.py from experiments/repair_2026-10-12/round6 (r6_monitor_summary.json, r6_causes.json, power6_scenarios.json, val6_summary.json, pool6_manifest.json, inv6_summary.json) and round4/r4_monitor_summary.json. Do not edit by hand.",
]
for k in sorted(M):
    lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
(ROOT / "numbers_round6.tex").write_text("\n".join(lines) + "\n")
(ROOT / "numbers_round6.json").write_text(json.dumps(RAW, indent=1, sort_keys=True))
print(f"{len(M)} macros written to numbers_round6.tex")

# per-source and per-stratum table
names = {
    "R5": "Frozen fifth-round detector",
    "V1": "Variant 1, null-calibrated level",
    "V2": "Variant 2, with re-arm",
    "V3": "Variant 3, with warm start",
    "STB": "Static CUSUM baseline",
    "ADB": "ADWIN comparator",
}


def cell(d):
    return f"{d['pass']}"


body = []
for a in ("V1", "V2", "V3", "STB", "R5", "ADB"):
    arm = P2["arms"][a]
    cc = arm["conditions"]
    j = cc["joint"]
    row = [
        names[a],
        cell(cc["c1"]),
        cell(cc["c2"]),
        cell(cc["c3"]),
        f"{j['pass']}",
        f"{100 * j['wilson95'][0]:.1f} to {100 * j['wilson95'][1]:.1f}",
        cell(arm["by_source"]["cesnet"]["joint"]),
        cell(arm["by_source"]["abilene"]["joint"]),
        cell(arm["by_stratum"]["P2_new"]["joint"]),
        cell(arm["by_stratum"]["P2_FG"]["joint"]),
        f"{P3['arms'][a]['conditions']['joint']['pass']}",
    ]
    body.append(" & ".join(row) + r" \\")
n = P2["arms"]["V1"]["conditions"]["joint"]["n"]
hdr = (
    f"Arm & $c_1$ & $c_2$ & $c_3'$ & Joint & Wilson $95\\,\\%$ (\\%) & CESNET & Abilene & New & Companion & GEANT \\\\\n"
    f" & of {n} & of {n} & of {n} & of {n} & & of {P2['arms']['V1']['by_source']['cesnet']['joint']['n']} & of {P2['arms']['V1']['by_source']['abilene']['joint']['n']} & of {P2['arms']['V1']['by_stratum']['P2_new']['joint']['n']} & of {P2['arms']['V1']['by_stratum']['P2_FG']['joint']['n']} & of {P3['traces']} \\\\"
)
tex = (
    "% Written by peerj_v2/scripts/make_round6_numbers.py from experiments/repair_2026-10-12/round6/r6_monitor_summary.json\n"
    "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}p{3.3cm}rrrrcrrrrr@{}}\n\\toprule\n"
    + hdr
    + "\n\\midrule\n"
    + "\n".join(body)
    + "\n\\bottomrule\n\\end{tabular}\n}\n"
)
(ROOT / "tables" / "monitor_round6.tex").write_text(tex)
print(tex)
