# Round 7 diagnosis: why condition c2 failed on cesnet_inst159, 202, 243 and 254, and why abilene_od110 exceeded the c3' null

Written 2026-10-12 by a round 7 design worker. Design only: no round 7 test exists and no success rule of any round is recomputed. Data read: the round 6 validation streams (`round6/r6_val`, 20 P2 traces), the frozen round 6 parameters (`round6/frozen_r6.json`, sha256 re-checked: f170954f...) and the **revealed** round 6 test streams (`round6/r6_test`) and rows (`round6/r6_monitor_rows.csv`), whose result is public (`round6/r6_monitor_summary.json`: V1 to V3 at 15 of 20 against 16, Holm p 0.062). They are read here to explain the miss and to design round 7. **Round 6 is not re-scored**: it remains a failure under its registered rule, and nothing below is evidence for the monitor claim. The paper must say that the round 7 design was informed by the revealed round 6 test streams (Disclosure R7-1 of the draft registration).

Scripts (all in `round7/`, run as `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 taskset -c 12-15 ../../../.chas-venv/bin/python <script>`, peak memory well under 1 GB):

| script | output | what it measures |
|---|---|---|
| `diag7.py` | `diag7_rows.csv`, `diag7_summary.json` | per trace and arm (V1, V3, STB): which null condition binds the validation level; natural level shifts, seasonality and heteroscedasticity on validation and revealed test; the injected signal regenerated with the round 6 test seeds (q step, z with the RF reference and with a reference frozen at the change point, CUSUM maximum against the level) |
| `diag7_extra.py` | `diag7_extra.csv`, `diag7_extra_summary.json` | injected z minus K per change point, the validation-to-test change of the scale of q, whether the floor binds on test, degenerate-q check |
| `val7.py` | `val7_rows.csv`, `val7_summary.json` (`val7_w4_*` for the later W4 arm) | candidate repairs selected on validation and scored on the revealed round 5 and round 6 test streams (design input for the registration, section 5 of the draft) |

## 1. The c2 failures are not control alarms; they are blind detectors

The registered classifier (`classify6_r6.py`) put cesnet_inst159, 202, 243 and 254 into class (2) "c2 not shift specific" for V1, V2 and V3. The label is correct in the registered sense (the Newcombe lower bound of det - ctl is at or below 0), but the matched no-injection control is silent on every one of them:

| trace (V3) | det | ctl | no-fault test events | det by change point (420, mid, n - 160) |
|---|---|---|---|---|
| cesnet_inst159 | 0.067 | 0 | 0 | 0.2, 0.0, 0.0 |
| cesnet_inst202 | 0.333 | 0 | 0 | 0.0, 0.0, 1.0 |
| cesnet_inst243 | 0.267 | 0 | 0 | 0.0, 0.0, 0.8 |
| cesnet_inst254 | 0.233 | 0 | 0 | 0.0, 0.0, 0.7 |

(`round6/r6_monitor_rows.csv`.) With ctl = 0 out of 3, the Newcombe interval needs det of roughly one half before its lower bound clears 0 (det 0.367 gives -0.21, det 0.667 gives +0.08, same file, column `ss_lo`). So c2 failed because the monitor did not see the injected shift at two of the three change points, not because it alarmed without one. The class name should read "insensitive (low detection, silent control)" in the paper's cause sentence. That is a wording fix and not a re-classification.

## 2. Why the detector was blind: a validation scale that does not transfer to the test stream

The coordinator's three candidate causes, measured (`diag7_summary.json`, `diag7_extra_summary.json`, V3 arm, c2-failed traces n = 4 against c2-passed n = 16, medians):

| quantity | c2 failed | c2 passed | reading |
|---|---|---|---|
| sd of q, validation | 1.65 | 0.52 | validation q is three times as variable |
| sd of q, revealed test | 0.55 | 0.48 | the test streams are ordinary |
| ratio test / validation sd of q | 0.27 | 1.00 | the scale shrinks by a factor of about 4 from validation to test |
| heteroscedasticity of the Bolt error, validation (p90 / p10 of the rolling 60-window sd) | 12.5 | 3.2 | bursty validation period |
| same, revealed test | 3.5 | 5.5 | the failed traces are calmer than the rest on test |
| natural level shift of q, validation (largest standardised 150-window mean step) | 3.40 | 2.58 | larger natural regime shifts on validation |
| same, revealed test | 2.59 | 2.73 | no difference on test |
| week-profile correlation of y, validation vs test | 0.22 | 0.31 | weak; inst159 has 0.81 and fails |
| ACF of y at lag 24, test | 0.24 | 0.51 | some loss of daily structure on test |
| traces where the floor exceeds the test sd of q | 3 of 4 | 0 of 16 | the validation floor dominates the test standardisation |
| traces whose mean injected z is below K at all three change points | 4 of 4 | 1 of 16 | the CUSUM drift under the fault is negative |
| bootstrap null condition binds the level (needs a higher level than the stream itself) | 3 of 4 | 10 of 16 | common, not specific |

The mechanism follows from these numbers. On cesnet_inst159, 202 and 254 the validation window contains bursts of Bolt error and natural regime shifts of q (binary segmentation `val7.pcfit` finds 4, 2 and 1 change points). These inflate the validation scale, which round 6 uses in two places. The floor of the RF standardisation is 0.5 sd(q_val), at 1.04, 0.95 and 0.70, while the test sd of q is 0.55, 0.54 and 0.27, so on test the floor rather than the local EW variance sets the divisor. The null-calibrated level and the K grid are also set on the turbulent stream: every (G, K) with a feasible level had zero validation detection on inst159 (V3 picked K = 2 and level 58.5 with val_det 0.0), and no level up to 60 met the bootstrap null on inst202, which then took the registered fallback of K = 2 at level 60. With that scale the injected q step (0.4 to 1.9 in log-MSE units, `diag7_rows.csv` column `inj`) becomes a z of 0.1 to 1.6, below K = 2 at every change point (`diag7_extra.csv`, V3_zinj_minus_K from -0.9 to -1.6). The CUSUM increment z - K is then negative in expectation, and the detector finds the fault only by noise, mostly at the late change point, where the local Bolt error is small and the injected step is largest (inst202: local signal-to-noise 3 sigma / local sd = 20.2 at n - 160 against 1.5 at 420).

The fourth trace has a different cause. On cesnet_inst243 the median q is -16.7 on validation and -13.7 on test: the E-Q MSE is about e^16.7, near 1.8e7 times the Bolt MSE, so E-Q is catastrophically wrong on this series and q mostly tracks E-Q's own error. No level up to 60 meets even the stream-itself null condition on validation (`cal_lvl_self = inf`), so V1 to V3 run at the fallback of K = 2 and level 60. This is a degenerate-expert trace, and a validation-only admissibility rule for it is proposed in the draft (|median q_val| <= log 100).

Answers to the three candidate causes:

1. **Detector alarms in the matched no-injection control**: no. The control is silent on all four traces (0 no-fault events on test for V3).
2. **Natural level shifts**: yes, but on the validation stream, not in the test windows. They inflate the scale and the level, which then do not transfer to the calmer test stream.
3. **Seasonality break**: weak. The validation-to-test week-profile correlation is lower on the failed traces (0.22 against 0.31) and the daily ACF on test is lower (0.24 against 0.51), but inst159 fails with a profile correlation of 0.81. It is not the driver.
4. **Residual heteroscedasticity**: yes, as the source of the validation turbulence (p90/p10 of the rolling Bolt-error sd 12.5 against 3.2). It also matters on test at single change points: a burst at the change point makes the injected step small (inst202 at 420, inst254 at the midpoint, q step 0.14).

The reference-adaptation hypothesis (that the EW reference absorbs the injected ramp) is not the cause on these traces. With the reference frozen at the change point, the mean injected z inside the lag window is 0.71 against 0.88 with the adaptive reference (failed traces, V3), while on the passing traces freezing raises it from 2.49 to 2.85. Freezing the reference would not have rescued them.

## 3. abilene_od110 (V2, V3: c3' null exceeded; V1: c1 false-alarm excess)

V3 detected every injection (det 1.0, ctl 0) but produced 4 no-fault events on the revealed test stream, so p_dense = 0.60 and p_rand = 0.72. The margin det - p0 >= 0.5 is unreachable once p0 > 0.5. The test stream of od110 carries a natural level shift that validation did not show: the largest standardised 150-window step of q is 3.22 on test against 1.06 on validation, the Bolt-error heteroscedasticity ratio is 21.1 on test (16.0 on validation), and the week-profile correlation between validation and test is 0.12. This is a genuine regime change in the no-fault test stream, and a reference-free monitor should alarm on it. The corrected criterion counts such alarms as chance alarms, which is the price of a null that is measured on the same stream. A one-alarm version (W3 in `val7.py`, 1 no-fault event, p0 = 0.38) passes; nothing in the criterion should be changed for this trace.

## 4. What the candidate repairs do on the revealed streams (design input, not evidence)

`val7.py` selects each candidate on validation and scores it with the round 6 corrected criterion on the revealed round 6 test streams (round 6 seeds, s = 6) and on the revealed round 5 test streams (round 5 seeds, s = 0). V3 is re-selected with the frozen round 6 code as a check and reproduces the published counts (15 of 20 on round 6, 56 of 64 on round 5, as `round6/val6_summary.json`). Joint pass counts:

| arm | what changes | revealed r6 (of 20) | revealed r5 (of 64) | pooled (of 84) | random-control c3' share (r6 / r5) |
|---|---|---|---|---|---|
| V3 (round 6) | none | 15 | 56 | 71 (0.845) | 0.000 / 0.009 |
| W1 SRC | level, K, G and floor calibrated on the segment residual of validation q | 15 | 54 | 69 (0.821) | 0.000 / 0.016 |
| W2 SRC-TC | W1 plus a two-stage confirmation (stage 1 at null 0.30, a 30-decision confirmation window, threshold c calibrated on validation) | 13 | 53 | 66 (0.786) | 0.006 / 0.021 |
| W3 RS-LMB | burst-robust floor (MAD), Huber-clipped EW variance, level-matched block bootstrap, revised fallback and selection key | 18 | 54 | 72 (0.857) | 0.004 / 0.009 |
| W4 RS-LMB-RF | W3 with the W1 floor (0.5 sd of the validation residual) | 18 | 56 | 74 (0.881) | 0.005 / 0.009 |
| STB7 (fair baseline for the round 7 calibration) | static CUSUM, reference = mean of the last 200 validation decisions, scale and level from the validation residual | 9 | 38 | 47 (0.560) | 0.018 / 0.021 |

Per-trace effects on the five round 6 misses (`val7_rows.csv`): W1 repairs inst159 (det 0.97) and inst254 (0.93); W3 repairs inst202 (0.67), inst254 (1.0) and od110 (one no-fault event); none repairs inst243 (degenerate E-Q, section 2). W3 does not repair inst159, because its MAD floor (1.14) is as large as the round 6 floor: the validation q of inst159 is regime-shaped, not bursty. W1 loses abilene_od0 and cesnet_inst152 on the round 6 streams. On the round 5 streams every W arm trades traces against V3: W1 loses 7 and gains 5, W2 loses 9 and gains 6, W3 loses 5 and gains 3, W4 loses 4 and gains 4, and every lost trace is an Abilene OD pair. The two-stage confirmation costs detection (mean det 0.76 against 0.82 for V3 on round 6, 0.88 against 0.93 on round 5). That is consistent with the extra 30-decision delay before an event, but the delay was not measured separately. W2 is not carried forward as a variant.

Every W arm was designed after the round 6 test result, and W4 also after the first `val7.py` run, so the round 6 column is optimistic for them in particular. The round 5 column is the less contaminated design input. Even there, round 6 was itself designed on the round 5 streams (round 6 Disclosure 1).

### 4a. W4

W4 (`ARMS7=W4 OUT7=_w4`, `val7_w4_rows.csv`, `val7_w4_summary.json`) was added after the first run, because W1 and W3 failed on complementary traces: W3 kept the large floor on inst159, and W1 lost Abilene traces to its residual calibration. W4 keeps the W3 calibration and takes the W1 floor. Against V3 it loses no trace on the round 6 streams and gains od110, inst159 and inst254, so it fails only on inst202 and inst243. On the round 5 streams it loses 4 traces and gains 4, which leaves the V3 count of 56 unchanged. Validation feasibility is low (10 of 20 and 37 of 64), so more traces run on the revised fallback. The null stays controlled on the revealed streams (mean p_dense 0.069 on both splits, random-control c3' share 0.5 % and 0.9 %). Because W4 was designed after seeing both W1 and W3 on these same streams, its round 6 count of 18 is the most optimistic number in this table.

## 5. Consequences for the round 7 design

1. The repair target is the scale transfer from validation to test, not control alarms. Variants that only change alarm confirmation (two-stage) do not address it and lose detection under the cap of 150.
2. Two components carry design evidence: the segment-residual floor (from W1) and the level-matched bootstrap with the revised fallback (from W3). Their failures are complementary on the round 6 streams, and W4 combines them. The draft registers W1, W3 and W4, nested so that each component can be attributed.
3. The round 6 fallback (K = 2 at level 60 when no level is feasible) is the least sensitive configuration on the grid and turns infeasibility into blindness. The revised fallback (the smallest level meeting the stream-itself null and FA conditions, flagged infeasible; selection key: feasibility, then validation detection) is part of every round 7 arm, including the baseline.
4. Degenerate-expert traces (inst243) cannot be monitored by a log-MSE ratio. A validation-only admissibility rule for them is proposed and disclosed as motivated by the revealed round 6 test.
5. Natural regime shifts inside the no-fault test stream (od110) are real shifts. They stay in the null, and the criterion is unchanged.
