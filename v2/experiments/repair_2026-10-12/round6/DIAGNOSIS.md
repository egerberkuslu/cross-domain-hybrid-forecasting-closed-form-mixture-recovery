# Round 6 diagnosis: why condition 3 of the round 5 monitor failed

Written 2026-10-12 by a round 6 design worker. No test file was read except the round 5 TEST streams of the 64 stratum-F traces (`round5/r5_test/*__h6.npz`). `r5_monitor.py --stage test` had already read and scored those streams, and the result is public (`round5/r5_monitor_summary.json`, commit 809833f). They are re-analysed here only to explain the miss. No round 6 parameter is chosen from them, and they will not be round 6 test traces. Every number below comes from a script in this folder:

| file | what it computes |
|---|---|
| `diag6.py` -> `diag6_rows.csv`, `diag6_summary.json` | alarm timelines of ST, AD and RF (frozen round 5 parameters, hash re-checked) on the round 5 validation and the revealed test streams, plus the injected and placebo lags from `r5_monitor_runs.csv` |
| `classify6.py` -> `diag6_classes.csv`, `diag6_classes.json` | the cause of each RF condition-3 miss |
| `val6.py` -> `val6_rows.csv`, `val6_summary.json` | the round 6 arms selected on validation, then scored in-sample on validation and, labelled `revealed_r5_test`, on the revealed round 5 test streams |

All runs used `PYTHONDONTWRITEBYTECODE=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python <script>`.

## 1. The registered condition, restated

Condition 3 of `round4/r4_monitor.py`, used unchanged in round 5, works like this. The no-fault test stream is scored at five placebo clocks P_j = 190 + 30 j (windows 190 to 310). A clock counts as detected if a run of 3 hits starts or continues anywhere in [P_j + 7, P_j + 100]. Each of the 30 injected runs (3 change points x 10 seeds) is scored with the same cap of 100. A trace passes if the Newcombe 95 % interval of the difference between det_cap100 (n = 30) and placebo_rate (n = 5) lies above 0, and condition 2 also holds. RF passed on 44 of 64 traces against the 52 needed.

## 2. Measured causes of the 20 RF misses (`diag6_classes.json`)

| cause | traces | definition |
|---|---|---|
| condition 2 failed | 2 | abilene_od131, abilene_od78 |
| placebo clocks fire | 10 | placebo_rate >= 0.6 (3 or more of the 5 clocks) |
| slow detection under the cap of 100 | 9 | placebo_rate < 0.6 and det_cap100 <= 0.7 (od131 also counted under condition 2) |
| resolution only | 0 | |

Together these cover all 20 misses.

### 2a. Slow detection (9 traces): the cap of 100 is too tight, not too loose

All 9 traces have a no-fault false-alarm share of 0 and a placebo rate of 0, except od5 and od11 at 0.2. They detect the injection within the cap of 150 that condition 2 uses (det_rate 0.43 to 1.00), but rarely within 100 (det_cap100 0.03 to 0.63). Over all 1920 RF injected runs, the median lag is 71 windows and the 90th percentile is 105. 222 runs (11.6 %) land between 100 and 150, and 129 (6.7 %) never alarm. The mean det_cap100 is 0.814 against 0.929 at the cap of 150. The cause is the design itself. The injected fault ramps in over 100 windows (`monitor_far.inject`), and q is a 60-window MSE ratio read with a delay of 6, so a conservative detector, with level selected for a false-alarm share of at most 5 %, reaches the fault after about 70 to 110 windows. The cap of 100 for condition 3 also differs from the cap of 150 that condition 2 and the round 3 selection use. With 5 placebo clocks, the Newcombe interval is coarse. With 0 of 5 placebo clocks firing, a trace needs at least 20 of 30 injected runs inside the cap of 100 (`max_placebo_hits_of5_passing_given_inj_hits_of30_cap100`). These 9 detectors have no false alarm at all, yet they fail.

### 2b. Placebo clocks fire (10 traces): warm-up and burn-in release, not stream-wide false alarms

- The alarms sit where the clocks are. On the revealed test streams, RF alarm onsets per 100 decisions run at 3.34 in the post-burn-in prefix (windows 166 to 419) and at 1.85 in the rest of the stream, a ratio of 1.8. On validation the figures are 5.19 against 2.31, a ratio of 2.2 (`diag6_summary.json`, `onsets_prefix_per100` and `onsets_rest_per100`). For 7 of the 10 placebo-failed traces the no-fault stream has essentially no alarm after window 420 (rest rate at most 0.36 per 100 decisions): od100, od102, od17, od20, od61, od62 and inst192. The remaining 3 (od130, od142, od55) alarm throughout, with false-alarm shares of 6.0 %, 4.6 % and 6.2 %. The placebo clocks therefore sample the noisiest part of the stream, the part where RF's exponentially weighted reference is still converging. That reference has decay 0.99, a time constant of about 100 decisions, starts from the mean of q[:20], and is fed with a delay of G = 30 or 60.
- Burn-in release. The CUSUM keeps accumulating during the 100 burn-in decisions while alarms are only suppressed. On 17 of 64 test traces S is already above the level when the burn-in ends (15 of 64 on validation), and 18 traces are in alarm at that instant. 4 of the 10 placebo-failed traces have their first alarm at window 168, two windows after the burn-in ends (od130, od20, od61, od17). Three more traces show the same release (od11, od5, od78).
- Sustained alarms count as detections. In 29 of the 80 RF placebo detections, the alarm run had started before the placebo clock (`placebo_pre_active`). Since `eval_run` accepts any run reaching 3 hits after P_j + 7, an alarm that is already on detects every later clock.
- The clocks overlap. Lag windows [P_j + 7, P_j + 100] are 94 windows long and 30 apart, so one alarm event covers up to 4 of the 5 clocks. The effective number of independent placebo observations is about (410 - 197) / 94 = 2.3, while the Newcombe interval treats them as 5.

### 2c. The criterion conflicts with the false-alarm budget of condition 1

AD shows this most cleanly. It has a tiny false-alarm share (mean onset share 1.3 % on test) and no warm-up, yet its mean placebo rate is 0.716 and 48 of 64 traces reach 0.6 or more. For a random alarm process at rate lambda = 0.013 per decision, the chance of at least one alarm in a 94-decision window is 1 - (1 - 0.013)^94 = 0.71, which matches. At the 5 % budget that condition 1 allows, the same chance is 1 - 0.95^94 = 0.99. A detector that passes condition 1 at its budget therefore fails condition 3 by construction. In practice condition 3 demands an alarm rate several times below the condition 1 budget, which no detector was ever calibrated for: RF, ST and AD all set their level against the 5 % false-alarm share.

### 2d. Natural drift: the driver for ST, not for RF

ST (static CUSUM on q with the validation mean and sd) fires at placebo clocks on 36 of 64 test traces (rate >= 0.6). Its hit share in the prefix rises from 18.5 % on validation to 36.3 % on test, it passes condition 1 on only 23 of 64 traces, and 92 of its 170 placebo detections come from alarm runs that started before the clock. A static reference cannot follow the validation-to-test level change of a drifting trace, so ST fires on any level change. For RF the measured relation between placebo rate and natural drift of the clean prefix is weak, and its sign flips between splits: the correlation with |q shift| is +0.28 on validation and -0.24 on test, and with |y shift| it is -0.005 and -0.05. The hypothesis that "the placebo clocks fall into natural drift" is not supported as the cause of the RF misses. The RF misses come from the cap, the warm-up and burn-in transient, and the conflict with the budget.

## 3. Is condition 3 ill-posed? Yes, as implemented; the idea behind it is sound

Asking that a monitor detects a change more often than it alarms by chance is right, and condition 2 alone (n = 3 matched clocks) is too weak to carry the claim. The implementation is ill-posed for four measured reasons:

1. It contradicts condition 1. At the condition 1 budget, the probability of a chance alarm in 94 decisions is 0.99 (section 2c).
2. Its null sample is biased and too small. The 5 clocks lie in the warm-up and burn-in-release part of the stream (section 2b), and they overlap so that the effective n is about 2.3 while the test assumes 5.
3. Its cap is inconsistent. It uses a cap of 100 while condition 2 and the selection use 150, although the injected ramp takes 100 windows to reach full size (section 2a).
4. It counts sustained alarms as detections. An alarm that was already on before the clock counts.

Natural drift inside the placebo window is a real problem only for the static baseline ST.

A corrected condition 3 is proposed in `REPAIR_PREREG_6_DRAFT.md` section 4: onset-localised events, one cap of 150 everywhere, and a null rate taken from dense clocks over the whole post-burn-in no-fault stream and from a rate-matched random-alarm process, with a margin. **Disclosure: this correction is motivated by the round 5 TEST result.** The paper must say so, must report the round 5 miss under its registered rule (44 of 64), and must not re-score round 5 under the new rule as a success.

## 4. What the validation-only feasibility shows (`val6_summary.json`)

The full table is in section 5 of `REPAIR_PREREG_6_DRAFT.md`. Joint pass counts under the corrected criterion, of 64 round 5 traces:

| arm | validation, in-sample | revealed r5 test |
|---|---|---|
| R5 (frozen round 5 RF) | 43 | 51 |
| V1 | 58 | 54 |
| V2 | 62 | 55 |
| V3 | 63 | 56 |
| STB baseline | 61 | 48 |
| ADB comparator | 26 | 26 |

These are design inputs, not test estimates. The validation numbers are in-sample, because each level is calibrated and scored on the same stream. The revealed test numbers come from data whose round 5 result is already public; they are disclosed and were used only for the power scenario. Two facts matter for honesty:

1. The corrected criterion alone moves the frozen round 5 RF from 44 to 51 of 64 on the revealed test, still one short of 52. Round 5 remains a failure under its registered rule, and the paper must not re-score it.
2. Under the corrected criterion the random control (rate-matched random alarm events) passes c3' on only 0.9 to 4.3 % of draws, so the corrected criterion has about the right size. The exact binomial alternative does not (5 to 24 %), because the injected runs come in clusters of 10.
