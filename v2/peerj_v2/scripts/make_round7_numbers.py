#!/usr/bin/env python3
"""Writes numbers_round7.tex (LaTeX macros), numbers_round7.json and tables/monitor_round7.tex from the repair round 7 result files.

Inputs (experiments/repair_2026-10-12/round7): r7_monitor_summary.json, r7_monitor_rows.csv, r7_causes.json, mcnemar7.json,
power7.json, pool7_manifest.json, degen7.json, val7_summary.json, val7_w4_summary.json; earlier monitor rounds from
round3/r3_monitor_summary.json, round4/r4_monitor_summary.json, round5/r5_monitor_summary.json, round6/r6_monitor_summary.json.
Nothing is typed by hand in the round 7 text of main.tex, highlights.tex and cover_letter.tex: every number is a macro from this file.
Constants fixed by the registration text (REPAIR_PREREG_7.md) are put() below with their source.
Macro names carry letters only (LaTeX forbids digits in control sequences), prefix rsv.
Arm tags: W1 Wone, W3 Wthree, W4 Wfour (variants), STB7 Stb (static baseline), V3 Vsix (sixth-round recipe re-selected), ADB Adb (ADWIN).
Run: python3 scripts/make_round7_numbers.py   (from peerj_v2)"""
import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # v2/peerj_v2 in this repository
V2 = ROOT.parent  # v2/, which holds experiments/repair_2026-10-12
X = V2 / "experiments" / "repair_2026-10-12"
R7 = X / "round7"
ms = json.loads((R7 / "r7_monitor_summary.json").read_text())
ca = json.loads((R7 / "r7_causes.json").read_text())
mc = json.loads((R7 / "mcnemar7.json").read_text())
pw = json.loads((R7 / "power7.json").read_text())
pm = json.loads((R7 / "pool7_manifest.json").read_text())
dg = json.loads((R7 / "degen7.json").read_text())
v5 = json.loads((R7 / "val7_summary.json").read_text())["revealed_r6_test"]
v5.update(json.loads((R7 / "val7_w4_summary.json").read_text())["revealed_r6_test"])
rows = list(csv.DictReader(open(R7 / "r7_monitor_rows.csv")))
m3 = json.loads((X / "round3" / "r3_monitor_summary.json").read_text())
m4 = json.loads((X / "round4" / "r4_monitor_summary.json").read_text())
m5 = json.loads((X / "round5" / "r5_monitor_summary.json").read_text())
m6 = json.loads((X / "round6" / "r6_monitor_summary.json").read_text())["P2"]

P = ms["P7"]
G = ms["G7_geant_descriptive"]
assert not ms["pseudo"] and P["success"] is True
assert P["succeeding_variants"] == ["W1", "W4"] and P["headline"] == "W1"
assert P["traces"] == 40 and P["need"] == 32 and G["traces"] == 15
assert mc["mismatch"] == [] and mc["claim_holds_for_all_succeeding"] is True
assert mc["success_rule"]["succeeding"] == ["W1", "W4"] and mc["headline"] == "W1"

M = {}
RAW = {}


def put(name, value, text):
    M[name] = text
    RAW[name] = value


def hu(x, d=2):
    """Round half up on the shortest decimal form of x (0.605 -> 0.61), not on its binary value."""
    return str(
        Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    )


def pct(x, d=1):
    return hu(100 * x, d)


def pfmt(p):
    """p in plain decimals above 1e-3, otherwise two significant digits in scientific form."""
    if p >= 1e-3:
        return hu(p, 4)
    m, e = f"{p:.1e}".split("e")
    return rf"\ensuremath{{{m}\times 10^{{{int(e)}}}}}"


def wilson(k, n, z=1.96):
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


# the Wilson routine must reproduce the intervals of the summary
for arm in P["arms"].values():
    for d in arm["conditions"].values():
        lo, hi = wilson(d["pass"], d["n"])
        assert abs(lo - d["wilson95"][0]) < 1e-9 and abs(hi - d["wilson95"][1]) < 1e-9

ARM = {
    "W1": "Wone",
    "W3": "Wthree",
    "W4": "Wfour",
    "STB7": "Stb",
    "V3": "Vsix",
    "ADB": "Adb",
}
COND = {"c1": "COne", "c2": "CTwo", "c3": "CThree", "joint": "J"}
LEVEL = {"P7_sub": "Sub", "P7_ip": "Ip"}

# pool structure
c = pm["counts"]
n_sub = len(dg["selected"]["sub"])
n_ip = len(dg["selected"]["ip"])
assert n_sub + n_ip == P["traces"] == dg["n_primary"] == 40
put("rsvTraces", P["traces"], str(P["traces"]))
put("rsvNeed", P["need"], str(P["need"]))
put("rsvSub", n_sub, str(n_sub))
put("rsvIp", n_ip, str(n_ip))
put("rsvGeant", G["traces"], str(G["traces"]))
put("rsvPlanned", 60, "60")  # REPAIR_PREREG_7.md R7-7 and decision D7
put("rsvInst", c["institutions_in_primary"] - 0, str(c["institutions_in_primary"]))
sel = set(dg["selected"]["sub"]) | set(dg["selected"]["ip"])
ser = {s["name"]: s for s in pm["series"]}
inst_all = {ser[n]["institution"] for n in sel}
inst_sub = {ser[n]["institution"] for n in dg["selected"]["sub"]}
inst_ip = {ser[n]["institution"] for n in dg["selected"]["ip"]}
rows_inst = {r["institution"] for r in rows if r["stratum"].startswith("P7")}
assert len(inst_all) == len(rows_inst) == 26 == c["institutions_in_primary"] - 0
put("rsvInstSub", len(inst_sub), str(len(inst_sub)))
put("rsvInstIp", len(inst_ip), str(len(inst_ip)))
put(
    "rsvParentEarlier",
    sum(bool(ser[n]["parent_used_earlier_round"]) for n in sel),
    str(sum(bool(ser[n]["parent_used_earlier_round"]) for n in sel)),
)
put(
    "rsvExclParents",
    len(pm["rules"]["excluded_parent_institutions_rounds_3_to_6"]),
    str(len(pm["rules"]["excluded_parent_institutions_rounds_3_to_6"])),
)
put(
    "rsvCorrMax",
    pm["rules"]["corr_max_near_copy"],
    f"{pm['rules']['corr_max_near_copy']:.2f}",
)
put("rsvFamilyCap", pm["rules"]["family_cap"], str(pm["rules"]["family_cap"]))
put("rsvDriftSd", pm["rules"]["drift_sd"], f"{pm['rules']['drift_sd']:.1f}")
put("rsvMissMax", pm["rules"]["missing_max"], f"{100 * pm['rules']['missing_max']:.0f}")
put("rsvZeroMax", pm["rules"]["zero_max"], f"{100 * pm['rules']['zero_max']:.0f}")
put(
    "rsvMinWin",
    pm["rules"]["min_test_windows_h6"],
    str(pm["rules"]["min_test_windows_h6"]),
)
put("rsvIdsSub", c["ids_subnets"], str(c["ids_subnets"]))
put("rsvIdsIp", c["ids_ips"], str(c["ids_ips"]))
put("rsvAdmSub", c["admissible_subnets"], str(c["admissible_subnets"]))
put("rsvAdmIp", c["admissible_ips"], str(c["admissible_ips"]))
put("rsvDriftSubN", c["drifting_len_ok_subnets"], str(c["drifting_len_ok_subnets"]))
put("rsvDriftIpN", c["drifting_len_ok_ips"], str(c["drifting_len_ok_ips"]))
put("rsvCapSub", c["after_family_cap_subnets"], str(c["after_family_cap_subnets"]))
put("rsvCapIp", c["after_family_cap_ips"], str(c["after_family_cap_ips"]))
assert c["corr_rule_excluded"] == []
dropped = dg["dropped"]
put("rsvDegenDropped", len(dropped), str(len(dropped)))
put(
    "rsvDegenRatio", 100, "100"
)  # degenerate-expert rule: validation MSE ratio beyond 100 (REPAIR_PREREG_7.md section 1)
put(
    "rsvNearCopyCorr", 1.0, "1.000"
)  # R7-5: the two near-copies left by the first freeze correlate 1.000 with their parents
put("rsvNearCopyN", 2, "2")  # cesnet_sub437 and cesnet_sub480, R7-5

# arms
for a, tag in ARM.items():
    arm = P["arms"][a]
    put(f"rsv{tag}X", arm["X"], str(arm["X"]))
    put(f"rsv{tag}PBinom", arm["p_binom_vs_0.5"], pfmt(arm["p_binom_vs_0.5"]))
    put(f"rsv{tag}Feas", arm["feasible"], str(arm["feasible"]))
    put(f"rsv{tag}Fall", arm["n"] - arm["feasible"], str(arm["n"] - arm["feasible"]))
    put(
        f"rsv{tag}Rnd",
        arm["rand_ctrl_c3_share_mean"],
        pct(arm["rand_ctrl_c3_share_mean"], 1),
    )
    put(
        f"rsv{tag}RndJ",
        arm["rand_ctrl_joint_share_mean"],
        pct(arm["rand_ctrl_joint_share_mean"], 1),
    )
    put(f"rsv{tag}Det", arm["mean_det"], pct(arm["mean_det"], 1))
    for cname, ctag in COND.items():
        d = arm["conditions"][cname]
        put(f"rsv{tag}{ctag}Pass", d["pass"], str(d["pass"]))
        put(f"rsv{tag}{ctag}N", d["n"], str(d["n"]))
        put(f"rsv{tag}{ctag}Rate", d["pass"] / d["n"], pct(d["pass"] / d["n"], 1))
        put(f"rsv{tag}{ctag}Lo", d["wilson95"][0], pct(d["wilson95"][0], 1))
        put(f"rsv{tag}{ctag}Hi", d["wilson95"][1], pct(d["wilson95"][1], 1))
    for lv, ltag in LEVEL.items():
        d = arm["by_level"][lv]["joint"]
        put(f"rsv{tag}J{ltag}Pass", d["pass"], str(d["pass"]))
        put(f"rsv{tag}J{ltag}N", d["n"], str(d["n"]))
        put(f"rsv{tag}J{ltag}Rate", d["pass"] / d["n"], pct(d["pass"] / d["n"], 1))
        put(f"rsv{tag}J{ltag}Lo", d["wilson95"][0], pct(d["wilson95"][0], 1))
        put(f"rsv{tag}J{ltag}Hi", d["wilson95"][1], pct(d["wilson95"][1], 1))
    # pass counts among traces with and without a feasible level, from the rows
    for fl, ftag in ((True, "OnFeas"), (False, "OnFall")):
        rr = [
            r
            for r in rows
            if r["arm"] == a
            and r["stratum"].startswith("P7")
            and (r["feasible"] == "True") == fl
        ]
        k = sum(r["joint"] == "True" for r in rr)
        put(f"rsv{tag}{ftag}Pass", k, str(k))
        put(f"rsv{tag}{ftag}N", len(rr), str(len(rr)))
    # per institution: share of institutions whose every trace passes, and of those with at least one passing trace
    byinst = {}
    for r in rows:
        if r["arm"] == a and r["stratum"].startswith("P7"):
            byinst.setdefault(r["institution"], []).append(r["joint"] == "True")
    ni = len(byinst)
    allp = sum(all(v) for v in byinst.values())
    anyp = sum(any(v) for v in byinst.values())
    assert ni == 26 and sum(len(v) for v in byinst.values()) == 40
    assert sum(sum(v) for v in byinst.values()) == arm["X"]
    fam = arm["by_family"]
    assert sum(f["joint"] == f["traces"] for f in fam.values()) == allp
    put(f"rsv{tag}InstN", ni, str(ni))
    put(f"rsv{tag}InstAll", allp, str(allp))
    put(f"rsv{tag}InstAllRate", allp / ni, pct(allp / ni, 1))
    lo, hi = wilson(allp, ni)
    put(f"rsv{tag}InstAllLo", lo, pct(lo, 1))
    put(f"rsv{tag}InstAllHi", hi, pct(hi, 1))
    put(f"rsv{tag}InstAny", anyp, str(anyp))
    lo, hi = wilson(anyp, ni)
    put(f"rsv{tag}InstAnyLo", lo, pct(lo, 1))
    put(f"rsv{tag}InstAnyHi", hi, pct(hi, 1))
    # G7 descriptive
    d = G["arms"][a]["conditions"]["joint"]
    put(f"rsv{tag}GeantPass", d["pass"], str(d["pass"]))
    put(f"rsv{tag}GeantN", d["n"], str(d["n"]))
    put(f"rsv{tag}GeantRate", d["pass"] / d["n"], pct(d["pass"] / d["n"], 1))
    put(f"rsv{tag}GeantLo", d["wilson95"][0], pct(d["wilson95"][0], 1))
    put(f"rsv{tag}GeantHi", d["wilson95"][1], pct(d["wilson95"][1], 1))
    put(
        f"rsv{tag}GeantRnd",
        G["arms"][a]["rand_ctrl_c3_share_mean"],
        pct(G["arms"][a]["rand_ctrl_c3_share_mean"], 1),
    )
    # failure classes (registered, classify7.py)
    e = ca[f"{a}_P7"]
    put(f"rsv{tag}Failed", e["failed"], str(e["failed"]))
    for k, t in (
        ("c1_false_alarm_excess", "CauseCOne"),
        ("c2_insensitive", "CauseInsens"),
        ("c2_control_alarms", "CauseCtl"),
        ("c3_null_exceeded", "CauseNull"),
        ("c3_slow_detection", "CauseSlow"),
    ):
        put(f"rsv{tag}{t}", e[k], str(e[k]))

for v in ("W1", "W3", "W4"):
    assert ca[f"{v}_P7"]["c3_slow_detection"] == 0  # the text states that no miss is a slow detection
put("rsvWoneOverVsix", P["arms"]["W1"]["X"] - P["arms"]["V3"]["X"], str(P["arms"]["W1"]["X"] - P["arms"]["V3"]["X"]))
# Holm-adjusted binomial p of the three variants (REPAIR_PREREG_7.md section 6)
for v, tag in (("W1", "Wone"), ("W3", "Wthree"), ("W4", "Wfour")):
    ph = P["arms"][v]["p_holm"]
    assert abs(ph - mc["success_rule"]["p_holm3"][v]) < 1e-12
    put(f"rsv{tag}PHolm", ph, pfmt(ph))
put("rsvAlpha", 0.05, "0.05")
put("rsvRndLimit", 0.05, "5")
rm = max(P["arms"][v]["rand_ctrl_c3_share_mean"] for v in ("W1", "W3", "W4"))
put("rsvRndMax", rm, pct(rm, 1))
assert P["arms"]["W3"]["X"] == P["need"] - 1 and P["arms"]["W4"]["X"] == P["need"]

# secondary claim: exact McNemar against the static baseline, Holm over the succeeding variants
for v, tag in (("W1", "Wone"), ("W4", "Wfour")):
    r = mc["mcnemar_vs_STB7"][v]
    put(f"rsv{tag}McVOnly", r["v_only"], str(r["v_only"]))
    put(f"rsv{tag}McStbOnly", r["stb7_only"], str(r["stb7_only"]))
    put(f"rsv{tag}McBoth", r["both"], str(r["both"]))
    put(f"rsv{tag}McNeither", r["neither"], str(r["neither"]))
    put(f"rsv{tag}McP", r["p_one_sided"], pfmt(r["p_one_sided"]))
    put(f"rsv{tag}McPHolm", r["p_holm"], pfmt(r["p_holm"]))
    assert r["claim"] is True

# power of the registration (power7.json), feasibility adjusted and at fixed pass rates
fa = pw["feasibility_adjusted"]
for v, tag in (("W1", "Wone"), ("W3", "Wthree"), ("W4", "Wfour")):
    put(f"rsvPow{tag}", fa[v]["power_R80"], hu(fa[v]["power_R80"], 2))
    put(
        f"rsvPow{tag}Eff",
        fa[v]["power_R80_design_effect_1.5"],
        hu(fa[v]["power_R80_design_effect_1.5"], 2),
    )
    put(f"rsvPi{tag}", fa[v]["pi_adjusted"], hu(fa[v]["pi_adjusted"], 3))
    put(f"rsvPiRev{tag}", fa[v]["design_pass_all"], hu(fa[v]["design_pass_all"], 3))
    put(
        f"rsvFeasShare{tag}",
        fa[v]["round7_feasible_share"],
        hu(fa[v]["round7_feasible_share"], 3),
    )
ap = pw["actual_pool"]["rules"]["R80"]["power"]
put("rsvPowAtEighty", ap["pi=0.8"], hu(ap["pi=0.8"], 2))
put("rsvPowAtSixAbout", ap["pi=0.75"], hu(ap["pi=0.75"], 2))
put(
    "rsvPowNeff",
    pw["actual_pool"]["n_eff_design_effect_1.5"],
    str(pw["actual_pool"]["n_eff_design_effect_1.5"]),
)

# revealed round 6 design inputs (not evidence): REPAIR_PREREG_7.md R7-2, val7_summary.json
for v, tag in (
    ("W1", "Wone"),
    ("W3", "Wthree"),
    ("W4", "Wfour"),
    ("STB7", "Stb"),
    ("V3", "Vsix"),
):
    put(f"rsvRev{tag}Pass", v5[v]["joint"], str(v5[v]["joint"]))
put("rsvRevN", v5["W1"]["n"], str(v5["W1"]["n"]))

# earlier monitor rounds, each under its own rule
put("rsvThreeN", m3["traces"], str(m3["traces"]))
put("rsvThreeNeed", m3["need"], str(m3["need"]))
put("rsvThreeCOne", m3["cond1_fa_le5"], str(m3["cond1_fa_le5"]))
put("rsvThreeCTwo", m3["cond2_shift_specific"], str(m3["cond2_shift_specific"]))
put(
    "rsvThreeCThree",
    m3["cond3_above_permuted_clock"],
    str(m3["cond3_above_permuted_clock"]),
)
put("rsvFourN", m4["traces"], str(m4["traces"]))
put("rsvFourNeed", m4["need"], str(m4["need"]))
put("rsvFourCOne", m4["cond1_fa_le5"], str(m4["cond1_fa_le5"]))
put("rsvFourCTwo", m4["cond2_shift_specific"], str(m4["cond2_shift_specific"]))
put("rsvFourCThree", m4["cond3_above_placebo"], str(m4["cond3_above_placebo"]))
put("rsvFiveN", m5["traces"], str(m5["traces"]))
put("rsvFiveNeed", m5["need"], str(m5["need"]))
put("rsvFiveCOne", m5["cond1_fa_le5"], str(m5["cond1_fa_le5"]))
put("rsvFiveCTwo", m5["cond2_shift_specific"], str(m5["cond2_shift_specific"]))
put("rsvFiveCThree", m5["cond3_above_placebo"], str(m5["cond3_above_placebo"]))
put("rsvSixN", m6["traces"], str(m6["traces"]))
put("rsvSixNeed", m6["need"], str(m6["need"]))
six = {m6["arms"][v]["conditions"]["joint"]["pass"] for v in ("V1", "V2", "V3")}
assert six == {15}
put("rsvSixJ", 15, "15")
put("rsvSixHolm", m6["arms"]["V1"]["p_holm"], hu(m6["arms"]["V1"]["p_holm"], 3))
assert not (m3["success"] or m4["success"] or m5["success"] or m6["success"])

# constants fixed by the registration text (REPAIR_PREREG_7.md sections 2, 3 and 6)
put("rsvBurn", 100, "100")  # burn-in decisions
put("rsvWarm", 200, "200")  # warm-start decisions
put("rsvSegMax", 4, "4")  # at most four change points in the binary segmentation of W1
put("rsvSegMin", 120, "120")  # minimum segment length
put(
    "rsvFloorFrac", 0.5, "0.5"
)  # scale floor of W1 as a share of the residual standard deviation
put(
    "rsvHuber", 3, "3"
)  # Huber clip of the exponentially weighted variance, in standard deviations
put("rsvCapNew", 150, "150")  # one cap of the corrected condition 3'
put("rsvMargin", 0.5, "0.5")  # margin of c3'
put("rsvNullTarget", 10, "10")  # null target in percent
put("rsvGridMax", 60, "60")
put("rsvDesignEffect", 1.5, "1.5")
put("rsvPiLow", 0.80, "0.80")  # pass rate of the second power scenario, REPAIR_PREREG_7.md section 6
put("rsvPiLower", 0.75, "0.75")  # pass rate of the sixth round, same table

lines = [
    "% Written by peerj_v2/scripts/make_round7_numbers.py from experiments/repair_2026-10-12/round7 (r7_monitor_summary.json, r7_monitor_rows.csv, r7_causes.json, mcnemar7.json, power7.json, pool7_manifest.json, degen7.json, val7_summary.json) and the monitor summaries of rounds 3 to 6. Do not edit by hand.",
]
for k in sorted(M):
    lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
(ROOT / "numbers_round7.tex").write_text("\n".join(lines) + "\n")
(ROOT / "numbers_round7.json").write_text(json.dumps(RAW, indent=1, sort_keys=True))
print(f"{len(lines) - 1} macros written to numbers_round7.tex")

# per-arm and per-level table
names = {
    "W1": "W1, segment-residual calibration",
    "W3": "W3, robust scale with level-matched bootstrap",
    "W4": "W4, W3 with the floor of W1",
    "STB7": "Static CUSUM baseline",
    "V3": "Sixth-round recipe, re-selected",
    "ADB": "ADWIN comparator",
}
body = []
for a in ("W1", "W3", "W4", "STB7", "V3", "ADB"):
    arm = P["arms"][a]
    cc = arm["conditions"]
    j = cc["joint"]
    row = [
        names[a],
        str(cc["c1"]["pass"]),
        str(cc["c2"]["pass"]),
        str(cc["c3"]["pass"]),
        str(j["pass"]),
        f"{pct(j['wilson95'][0])} to {pct(j['wilson95'][1])}",
        str(arm["by_level"]["P7_sub"]["joint"]["pass"]),
        str(arm["by_level"]["P7_ip"]["joint"]["pass"]),
        str(G["arms"][a]["conditions"]["joint"]["pass"]),
    ]
    body.append(" & ".join(row) + r" \\")
n = P["traces"]
ns = P["arms"]["W1"]["by_level"]["P7_sub"]["joint"]["n"]
ni_ = P["arms"]["W1"]["by_level"]["P7_ip"]["joint"]["n"]
hdr = (
    "Arm & $c_1$ & $c_2$ & $c_3'$ & Joint & Wilson $95\\,\\%$ (\\%) & Subnets & IP addresses & GEANT \\\\\n"
    f" & of {n} & of {n} & of {n} & of {n} & & of {ns} & of {ni_} & of {G['traces']} \\\\"
)
tex = (
    "% Written by peerj_v2/scripts/make_round7_numbers.py from experiments/repair_2026-10-12/round7/r7_monitor_summary.json\n"
    "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}p{4.4cm}rrrrcrrr@{}}\n\\toprule\n"
    + hdr
    + "\n\\midrule\n"
    + "\n".join(body)
    + "\n\\bottomrule\n\\end{tabular}\n}\n"
)
(ROOT / "tables" / "monitor_round7.tex").write_text(tex)
print(tex)
