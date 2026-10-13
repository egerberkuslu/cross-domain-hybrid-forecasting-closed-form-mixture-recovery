#!/usr/bin/env python3
"""Writes tables/repair_round.tex from the repair-round result files.
Inputs (experiments/repair_2026-10-12/paper_a): gate0_summary.json, gate0c_summary.json, a1b_test_summary.json, a2_test_summary.json,
a3_test_summary.json. Run: python3 scripts/make_repair_table.py"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # v2/peerj_v2 in this repository
V2 = ROOT.parent  # v2/, which holds experiments/repair_2026-10-12
RX = V2 / "experiments" / "repair_2026-10-12" / "paper_a"
g0 = json.loads((RX / "gate0_summary.json").read_text())
g0c = json.loads((RX / "gate0c_summary.json").read_text())
a1b = json.loads((RX / "a1b_test_summary.json").read_text())
a2 = json.loads((RX / "a2_test_summary.json").read_text())
a3 = json.loads((RX / "a3_test_summary.json").read_text())
t2 = pd.DataFrame(a2["table"])
h2 = t2[t2.detector == a2["headline"]]
a3c = pd.read_csv(RX / "a3_test_cells.csv")
pc = lambda x, d=2: f"{100 * x:.{d}f}"
sg = lambda x, d=2: ("$+$" if x >= 0 else "$-$") + f"{abs(100 * x):.{d}f}"
fl = a1b["floor"]
rows = [
    ("Accuracy headroom",
     "Weight estimated without its block improves the constant weight by at least $0.5\\,\\%$ in at least $6$ of $25$ cells.",
     f"{g0['cells_with_xfit_headroom_ge_0.5pct']} of $25$ cells, mean {sg(g0['mean_xfit_nb_L48'])}\\,\\%, best cell {sg(g0['max_xfit_nb_L48'])}\\,\\%. "
     f"Test-oracle constant weight beats Chronos-Bolt by at most {pc(g0c['max_test_oracle_constant_gain_vs_bolt'])}\\,\\%."),
    ("Endpoint bound",
     "Bound holds in at least $22$ of $25$ cells, in more cells than $\\alphaH$ fixed at the shipped value, and in more cells than both random controls.",
     f"Selected rule $\\alphaH{{=}}0$ holds in {fl['holds_variant']} cells because it equals Chronos-Bolt in all {a1b['cells_identical']}, and ties both controls at {fl['holds_ctrl_a']}. Shipped weight holds in {fl['holds_V0']}."),
    ("Shift monitor",
     "No-fault FA at most $5\\,\\%$ and shift-specific detection on at least $4$ of $5$ traces, and no worse than Page-Hinkley and ADWIN.",
     f"FA at most $5\\,\\%$ on {a2['cond1_fa_le5_traces']} of $5$ traces, shift-specific detection on {a2['cond2_shift_specific_traces']} of $5$, namely Abilene, CESNET and GEANT."),
    ("Learned $\\alpha(\\mathbf{x})$",
     "Better than the scalar in at least $8$ of $25$ cells, none worse by more than $1\\,\\%$, gain over the best random control with a confidence interval above $0$.",
     f"{a3['cells_better_than_scalar_holm']} cell better, {a3['cells_worse_than_scalar_by_gt_1pct']} cells worse by more than $1\\,\\%$. Gain over best control {sg(a3['pooled_gain_vs_best_control']['mean'])}\\,\\%, "
     f"interval {sg(a3['pooled_gain_vs_best_control']['ci95_cell'][0])} to {sg(a3['pooled_gain_vs_best_control']['ci95_cell'][1])}\\,\\%."),
]
R2 = V2 / "experiments" / "repair_2026-10-12" / "paper_a" / "round2"
a5 = json.loads((R2 / "a5_test_summary.json").read_text())
a4 = json.loads((R2 / "a4_test_summary.json").read_text())
eq, eb = a5["experts"]["EQ"], a5["experts"]["EB"]
rows += [
    ("Complementary expert",
     "A second expert mixed with Chronos-Bolt wins in at least $5$ of $25$ cells after Holm over $50$ comparisons with a gain of at least $0.5\\,\\%$, none significantly worse, pooled gain with an interval above $0$, also over the random controls.",
     f"Quantile expert wins {eq['cells_win_vs_bolt_holm_and_ge_0.5pct']} cells, none worse, pooled gain {sg(eq['pooled_gain_vs_bolt']['mean'])}\\,\\%, interval {sg(eq['pooled_gain_vs_bolt']['ci95_cell'][0])} to {sg(eq['pooled_gain_vs_bolt']['ci95_cell'][1])}\\,\\%. Residual expert wins {eb['cells_win_vs_bolt_holm_and_ge_0.5pct']} cell, pooled gain {sg(eb['pooled_gain_vs_bolt']['mean'])}\\,\\%."),
    ("Drift-adaptive monitor",
     "No-fault FA at most $5\\,\\%$, shift-specific detection and detection above a permuted-clock control, each on at least $4$ of $5$ traces.",
     f"FA at most $5\\,\\%$ on {a4['cond1_fa_le5_traces']} of $5$ traces, shift-specific detection on {a4['cond2_shift_specific_traces']} and above the permuted clock on {a4['cond3_above_permuted_clock_traces']}."),
]
R3 = V2 / "experiments" / "repair_2026-10-12" / "round3"
e3 = json.loads((R3 / "r3_test_summary.json").read_text())
m3 = json.loads((R3 / "r3_monitor_summary.json").read_text())
rows += [
    ("Quantile expert on fresh series",
     "At least $8$ of $100$ cells better than Chronos-Bolt after Holm with a gain of at least $0.5\\,\\%$, none significantly worse, pooled gain of at least $0.5\\,\\%$ with intervals above $0$ over cells and over series, also over the random controls.",
     f"{e3['cells_win']} cells better, none worse, pooled gain {pc(e3['pooled_gain_vs_bolt']['mean'])}\\,\\%, interval {sg(e3['pooled_gain_vs_bolt']['ci95_cell'][0])} to {sg(e3['pooled_gain_vs_bolt']['ci95_cell'][1])}\\,\\% over cells and {sg(e3['pooled_gain_vs_bolt']['ci95_series_clustered'][0])} to {sg(e3['pooled_gain_vs_bolt']['ci95_series_clustered'][1])}\\,\\% over series."),
    ("Reference-free monitor",
     f"No-fault FA at most $5\\,\\%$, shift-specific detection and detection above a permuted-clock control, each on at least {m3['need']} of {m3['traces']} fresh traces.",
     f"{m3['cond1_fa_le5']}, {m3['cond2_shift_specific']} and {m3['cond3_above_permuted_clock']} of {m3['traces']} traces."),
]
R4 = V2 / "experiments" / "repair_2026-10-12" / "round4"
e4 = json.loads((R4 / "r4_test_summary.json").read_text())
m4 = json.loads((R4 / "r4_monitor_summary.json").read_text())
yn = lambda b: "Yes" if b else "No"
sec4 = ("The secondary $0.5\\,\\%$ rule is met as well." if e4["success_secondary_unscaled_0.5pct"] else "The secondary $0.5\\,\\%$ rule is not met either.")
rows += [
    ("Guarded quantile expert on a second fresh pool",
     f"At least {e4['K_wins_required']} of {e4['cells']} cells better than Chronos-Bolt after Holm with a gain of at least ${pc(e4['min_gain_primary'], 3)}\\,\\%$ as primary minimum, which is $0.5\\,\\%$ scaled by the guard retention $0.4375$, with $0.5\\,\\%$ as secondary minimum, none significantly worse, pooled gain of at least the minimum with intervals above $0$ over cells and over series, also over the random controls, and a positive median cell gain.",
     f"{e4['cells_win']} cells better at the primary minimum and {e4['cells_win_at_0.5pct']} at $0.5\\,\\%$, {e4['cells_loss']} worse, pooled gain {pc(e4['pooled_gain_vs_bolt']['mean'])}\\,\\%, interval {sg(e4['pooled_gain_vs_bolt']['ci95_cell'][0])} to {sg(e4['pooled_gain_vs_bolt']['ci95_cell'][1])}\\,\\% over cells and {sg(e4['pooled_gain_vs_bolt']['ci95_series_clustered'][0])} to {sg(e4['pooled_gain_vs_bolt']['ci95_series_clustered'][1])}\\,\\% over series, median cell gain {pc(e4['median_cell_gain'])}\\,\\%. Gain over the better control {pc(e4['pooled_gain_vs_best_control']['mean'])}\\,\\%, interval {sg(e4['pooled_gain_vs_best_control']['ci95_series_clustered'][0], 3)} to {sg(e4['pooled_gain_vs_best_control']['ci95_series_clustered'][1])}\\,\\% over series. {sec4}",
     yn(e4["success"])),
    ("Reference-free monitor with placebo clocks",
     f"No-fault FA at most $5\\,\\%$, shift-specific detection and detection above a placebo-clock control inside the clean segment, each on at least {m4['need']} of {m4['traces']} fresh traces, the third among the traces that are shift-specific.",
     f"{m4['cond1_fa_le5']}, {m4['cond2_shift_specific']} and {m4['cond3_above_placebo']} of {m4['traces']} traces.",
     yn(m4["success"])),
]
R5 = V2 / "experiments" / "repair_2026-10-12" / "round5"
e5 = json.loads((R5 / "r5_test_summary.json").read_text())
m5 = json.loads((R5 / "r5_monitor_summary.json").read_text())
sec5 = (
    "The secondary $0.5\\,\\%$ rule is met as well."
    if e5["success_secondary_0.5pct"]
    else "The secondary $0.5\\,\\%$ rule is not met."
)
rows += [
    (
        "Guarded quantile expert on a third fresh pool",
        f"At least {e5['K_wins_required']} of {e5['cells']} cells better than Chronos-Bolt after Holm with a gain of at least ${pc(e5['min_gain_primary'], 3)}\\,\\%$ as primary minimum and $0.5\\,\\%$ as secondary minimum, none significantly worse, pooled gain of at least the minimum with intervals above $0$ over cells and over series, each of the two random controls cleared alone with Holm over the two and a pooled gain over each of at least the minimum, and a positive median cell gain.",
        f"{e5['cells_win_primary']} cells better at the primary minimum and {e5['cells_win_at_0.5pct']} at $0.5\\,\\%$, {e5['cells_loss']} worse, pooled gain {pc(e5['pooled_gain_vs_bolt']['mean'])}\\,\\%, interval {sg(e5['pooled_gain_vs_bolt']['ci95_cell'][0])} to {sg(e5['pooled_gain_vs_bolt']['ci95_cell'][1])}\\,\\% over cells and {sg(e5['pooled_gain_vs_bolt']['ci95_series_clustered'][0])} to {sg(e5['pooled_gain_vs_bolt']['ci95_series_clustered'][1])}\\,\\% over series, median cell gain {pc(e5['median_cell_gain'])}\\,\\%. Gain over the permuted-rows control {pc(e5['control_a_permuted_rows']['mean'])}\\,\\%, over the null expert {pc(e5['control_b_null_expert']['mean'])}\\,\\%, Holm-adjusted $p$ {e5['control_holm_adjusted_p'][0]:.4f} for both. {sec5}",
        yn(e5["success"]),
    ),
    (
        "Placebo-clock monitor on the F stratum",
        f"Same three conditions as in the fourth registration, each on at least {m5['need']} of {m5['traces']} drifting traces, declared uninformative in advance, so that pass rates with Wilson intervals are the reported result.",
        f"{m5['cond1_fa_le5']}, {m5['cond2_shift_specific']} and {m5['cond3_above_placebo']} of {m5['traces']} traces.",
        yn(m5["success"]),
    ),
]
R6 = V2 / "experiments" / "repair_2026-10-12" / "round6"
m6 = json.loads((R6 / "r6_monitor_summary.json").read_text())["P2"]
v6 = [m6["arms"][v] for v in ("V1", "V2", "V3")]
assert m6["traces"] == 20 and json.loads((R6 / "pool6_manifest.json").read_text())["counts"]["P2_FG"] == 9  # caption words "nine of twenty"
hp6 = {a["p_holm"] for a in v6}
assert len(hp6) == 1
rows += [
    (
        "Corrected-criterion monitor on drifting traces",
        f"Three variants. At least one passes the three conditions, the third corrected to onset-localised events with a margin of $0.5$ over the larger of the dense-clock and the random-alarm null rate, on at least {m6['need']} of {m6['traces']} drifting traces, with a Holm-adjusted one-sided binomial $p$ below $0.05$ against one half and a random-control pass share of at most $5\\,\\%$.",
        f"{', '.join(str(a['conditions']['joint']['pass']) for a in v6)} of {m6['traces']} traces for the three variants, Holm-adjusted $p$ {hp6.pop():.3f}. Static baseline {m6['arms']['STB']['X']}, frozen fifth-round detector {m6['arms']['R5']['X']}.",
        yn(m6["success"]),
    )
]
R7 = V2 / "experiments" / "repair_2026-10-12" / "round7"
m7 = json.loads((R7 / "r7_monitor_summary.json").read_text())["P7"]
mc7 = json.loads((R7 / "mcnemar7.json").read_text())
w7 = [m7["arms"][v] for v in ("W1", "W3", "W4")]
assert m7["traces"] == 40 and mc7["mismatch"] == [] and m7["headline"] == "W1"
p7 = f"{w7[0]['p_holm']:.1e}".split("e")
rows += [
    (
        "Scale-robust monitor on CESNET subnets and IP addresses",
        f"Three variants. At least one passes the three conditions of the sixth registration on at least {m7['need']} of {m7['traces']} drifting CESNET subnet and IP-address traces, with a Holm-adjusted one-sided binomial $p$ below $0.05$ against one half over the three variants and a random-control pass share of at most $5\\,\\%$.",
        f"{', '.join(str(a['conditions']['joint']['pass']) for a in w7)} of {m7['traces']} traces for the three variants, Holm-adjusted $p$ ${p7[0]}\\times 10^{{{int(p7[1])}}}$ for the first. Static baseline {m7['arms']['STB7']['X']}, sixth-round recipe re-selected {m7['arms']['V3']['X']}.",
        yn(m7["success"]),
    )
]
rows = [r if len(r) == 4 else r + ("No",) for r in rows]
body = "\n".join(f"{a} & {b} & {c} & {d} \\\\" + ("\n\\midrule" if i < len(rows) - 1 else "") for i, (a, b, c, d) in enumerate(rows))
tex = r"""% Written by peerj_v2/scripts/make_repair_table.py from experiments/repair_2026-10-12/paper_a
{\footnotesize
\begin{longtable}{p{1.9cm} p{4.4cm} p{5.2cm} c}
\caption{Outcome of the fourteen pre-registered repair tests on the held-out test split, four from the first registration, two from each of the next four and one from each of the sixth and the seventh, the last eight on fresh series, the fifth monitor on a stratum, the sixth monitor on a pool in which nine of twenty traces belong to a half whose Chronos-Bolt forecasts another study had scored and the seventh monitor on subnet and IP-address series that are components of the same network as earlier institution series, as the text discloses. Every rule was committed before the test split was read, and parameters were selected on validation windows only. The fifth registration is the fifth repair attempt, and its changed control condition was motivated by the test result of the fourth. The sixth registration re-tested the monitor alone, and its corrected third condition was motivated by the test result of the fifth. The seventh registration is the seventh monitor rule and the fifth test of a reference-free monitor, after four such tests that missed their rules, and it was designed after the test streams of rounds 4 to 6 had been revealed.}
\label{tab:repair}\\
\toprule
Claim repaired & Pre-registered rule & Measured & Rule met \\
\midrule
\endfirsthead
\toprule
Claim repaired & Pre-registered rule & Measured & Rule met \\
\midrule
\endhead
\bottomrule
\endfoot
""" + body + r"""
\end{longtable}
}
"""
(ROOT / "tables" / "repair_round.tex").write_text(tex)
print(tex)
