# REPAIR PRE-REGISTRATION 6, Paper A: placebo-calibrated, onset-localised reference-free shift monitor on fresh drifting traces

Status: ready to commit. It becomes registered when the coordinator writes `round6/PREREG_6_SHA256` (sha256 of this file) and `round6/PREREG_6_COMMITTED` and the commit exists. State at the time of writing (2026-10-12): the 148 candidate series are built (counts only, no forecast), stage A (train-only models, validation forecasts) has run for the 38 monitor cells, the validation selection has run (twice, see Disclosure 6) and its parameters are frozen (section 8). No stage B (test forecast) exists and no test window of any round 6 trace has been read. Rounds 1 to 5 (`REPAIR_PREREG.md`, `REPAIR_PREREG_2.md`, `round3/REPAIR_PREREG_3.md`, `round4/REPAIR_PREREG_4.md`, `round5/REPAIR_PREREG_5.md`) are frozen and untouched. Folder: `experiments/repair_2026-10-12/round6/`.

## 0. Why round 6, and disclosures

Round 6 is the third registered attempt at the monitor claim of Paper A (rounds 4 and 5 preceded it) and the sixth repair round of the paper. The paper reports it as such whatever the outcome. In round 4 the monitor passed conditions 1 to 3 on 9, 11 and 8 of 12 traces against 10 needed. In round 5 the frozen RF detector passed condition 1 on 58 of 64 traces, condition 2 on 62 and condition 3 on 44, against the 52 needed. The registered 80 % rule failed, and the round 5 registration had declared that rule uninformative in advance (power 0.02).

Disclosure 1, the criterion change was motivated by the round 5 test result. The corrected condition 3' of section 4 (onset-localised events, one cap of 150 for injected runs, matched control and null clocks, a null rate equal to the larger of the whole-stream dense-clock rate and the rate of a random-control alarm process, a margin of 0.5), the null target of 0.10 and the extended level grid were all designed after the round 5 test result was public, and the cause analysis of `DIAGNOSIS.md` was done on the revealed round 5 test streams of the 64 stratum-F traces. The paper states this in the same paragraph that reports round 6. The revealed re-analysis (`val6_summary.json`, rows `revealed_r5_test`: the validation-selected variants V1, V2 and V3 pass the corrected criterion on 54, 55 and 56 of 64 round 5 traces, the frozen round 5 detector on 51 of 64) is reported as motivation only and never as evidence for the claim. Round 5 remains a failure under its own registered rule (44 of 64 against 52) and is never re-scored as a success.

Disclosure 2, revealed data informed the power scenario. The per-trace pass probabilities of the power calculation (section 6) come from that revealed re-analysis. It was not used to choose between variants (all three go to the test) and not to tune any parameter, since every parameter is selected on validation by the code frozen in section 8.

Disclosure 3, earlier lessons constrained the variant set. The user's notes record that a CUSUM on residuals lost to the alpha-monitor and should not be proposed again ("D4 dersi", `[brain: Claude Bilgi Tabani/Kaynaklar/zaman-serisi-tahmin.md]`, `[brain: wiki/concepts/cha-s.md]` ^d4-baseline). A residual detector therefore enters only as a reported comparator (ADB, section 3), not as a repair variant.

Disclosure 4, reuse of the F+G half. Pool P2 contains the drifting CESNET and Abilene series of the F+G half (`reserved_series.json` block `round5_FG`, reserved for the F+G round 5 replication, `submission/paper_fg_trust/experiments/exp_f_repair5.py`, `contract/REPAIR_PREREG_5.md`). This is a reuse and it is disclosed in the same way as round 5 disclosed its stratum F: the F+G replication (`submission/paper_fg_trust/experiments/exp_f_repair5.py`) computed and scored Chronos-Bolt test forecasts at h = 1, 6 and 24 for the 9 P2_FG series, as its base model (`results/raw_f/repair5/pred/*__chronos_bolt__h6.npz` and `rows.csv`), together with Chronos-2 and FlowState arms. No E-Q, guard or monitor ever saw those forecasts, and the monitor parameters of round 6 are chosen on validation only. No E-Q forecast of these series exists. No accuracy of any round 6 trace is computed or reported. The F+G replication is finished, and its consumers must not use the h = 6 E-Q forecasts or the monitor outputs of these nine series. The P2_FG stratum is reported separately from the P2_new stratum in the per-stratum table, and the two are pooled for the success rule only.

Disclosure 5, relaxed missingness. The admissibility rule of `fresh_cells.py` (missing share at most the aggregate's plus 0.5 points) left no unused series. The relaxed rules of section 1 (CESNET missing at most 5 %, Abilene zero share at most 5 % and missing at most 16 %, GEANT missing at most 7 %) are registered here and were fixed before any monitor score of these traces. They were chosen after `inv6.py` had counted the missingness shares of the unallocated series, with no forecast and no test value read. The relaxed series have more interpolated hours (Abilene up to 16 %), so their q streams are smoother than those of earlier rounds.

Disclosure 6, order of work. Stage A and the validation selection were run before the commit, at the coordinator's instruction, because the frozen hashes make the selection reproducible and checkable. Both read only train and validation windows (`Restricted` view, `r6_test` folder guard), and the registration fixes the hashes of the validation forecasts and the frozen parameters (section 8). The selection was run twice. The first run gave the static baseline STB a fixed K of 0.5, and the coordinator then decided that the baseline must get the same K grid and fallback rule as V1 to V3, so that it differs from V2 only in the standardisation. The code was changed, the select stage and the dry run were repeated, and every hash of section 8 is that of the second run (the superseded hashes are in `RUN_NOTES_R6.md`). The change touched only STB. The frozen parameters of R5, V1, V2, V3 and ADB, the validation forecasts and the stage A models are unchanged, and no test window had been read at either run.

## 1. Traces

Inventory (`inv6.py` -> `inv6_summary.json`): every admissible series of every source is allocated to the round 3 reserved list, the pilot, `partition_F.json` F_registered, the round 4/5 A half, or the round 5 F+G half. The unallocated series are inadmissible only by the missing-share rule (CESNET 83, GEANT 195) or by the zero-share, missing-share or median rule (Abilene 16). `build_pool6.py` (deterministic, counts only) builds every candidate with `fresh_cells.build_series`, recomputes the screen with `build_pool4.shift` (train and validation hours only) and keeps a series as a monitor trace iff the train-to-validation shift is at least 0.3 train sd and the h = 6 test stream has at least 690 windows (CESNET, Abilene) or 420 windows (GEANT).

| stratum | candidates | monitor traces (shift >= 0.3) | note |
|---|---|---|---|
| P2_new, CESNET institutions | 50 | 9 | never allocated, missing share <= 5 % |
| P2_new, Abilene OD pairs | 2 | 2 | never allocated, zero share <= 5 %, missing <= 16 % (od0 15.7 %, od112 14.6 %) |
| P2_FG, CESNET institutions | 45 | 4 | F+G half, reused (Disclosure 4) |
| P2_FG, Abilene OD pairs | 10 | 5 | F+G half, reused (Disclosure 4) |
| **P2 total (primary pool)** | **107** | **20** (13 CESNET, 7 Abilene) | |
| P3, GEANT OD pairs | 41 | 18 | descriptive stratum, never pooled; never allocated, missing <= 7 % |

`pool6_manifest.json` (sha256 in section 8) lists every candidate with its shift, window counts and the sha256 of its series file and of its `splits.json`. The traces are: P2_new CESNET 51, 152, 159, 177, 202, 217, 243, 246, 271, Abilene OD 0, 112; P2_FG CESNET 26, 47, 56, 254, Abilene OD 42, 56, 82, 97, 110; P3 GEANT od10_12, od10_14, od10_16, od10_2, od10_21, od10_3, od10_9, od15_11, od15_8, od18_11, od1_4, od22_23, od3_15, od4_7, od5_11, od8_11, od8_21, od9_21. The additive block `round6_M` for `fresh_cells/reserved_series.json` is `round6/reserved_round6_M.json` (merged by the coordinator, not by the worker). The exact screen can differ from the approximate screen of `inv6.py` by up to 0.28 sd; the exact one decides, and the series with an exact shift between 0.2 and 0.3 are not traces.

Non-independence. The Abilene OD components come from one network over the same 699 test windows, the GEANT OD components likewise over the same 428 windows, and the CESNET institutions share one network and calendar. The per-trace counting rule treats traces as independent. The Wilson and binomial intervals of the Abilene stratum are read as optimistic, as in round 5, and per-source rates are reported.

GEANT design (P3). The GEANT validation stream has 281 windows, fewer than the burn-in (100 decisions) plus the lag window (up to 150), so the null-calibrated level of section 3 is undefined with the long-stream clocks. Short-stream rule, registered for streams where fewer than four dense clocks fit: the clocks run from the end of the burn-in to the last window minus 30, their lag windows are right-truncated at the end of the stream, and the resulting null rate is optimistic (`r6_lib.dense_clocks`). The GEANT test stream (428 windows) uses the full-window clocks. The GEANT change points are 200, 234 and 268 (`injection_cps` with `cp_lo = 200`, n - 160 for the last), the burn-in is 100 decisions (window 166), and the dense null clocks run from 166 to 278 (about 23 clocks), so the lag windows of the three change points overlap and the null sample is small. One no-fault event on a GEANT test stream already gives p_rand = 144/262 = 0.55, so that det - p0 cannot reach the margin of 0.5: c3' passes on GEANT only with zero no-fault events. GEANT is reported per stratum with its own pass rates and never enters the success rule. It is expected to be weakly informative and is reported as it comes out.

New series need h = 6 forecasts. Stage A (train-only models giving validation forecasts) and stage B (refit on train plus validation giving test forecasts) run with the frozen round 3 E-Q recipe and the Chronos-Bolt-small snapshot through `round3/r3_stages.py` functions (`round6/r6_stages.py`), restricted to h = 6. No accuracy of any round 6 trace is computed or reported.

## 2. What stays frozen

The statistic is unchanged: q is the log MSE ratio of Chronos-Bolt to E-Q over the last 60 labelled windows with label delay 6, built from horizon-averaged h = 6 forecasts, and e is the delayed Bolt squared error (`a2.streams`). Also unchanged: the burn-in of 100 decisions (counted in every false-alarm denominator), persistence 3 for the CUSUM arms and 1 for ADWIN, the injected fault (`monitor_far.inject`, a 3 sigma ramp over 100 windows), the RF reference (exponentially weighted with decay 0.99, delayed by G), the RF grid G in {30, 60}, K in {0.5, 1, 2}, and condition 1 (no-fault false-alarm onset share at most 5 %, the round 3 to 5 definition `onsets_k`).

## 3. Arms

All arms are selected on the validation stream of each trace. The selection injections are `a2.inject_streams` at positions 0.3 and 0.6 with 5 seeds (`a2.val_seed`), as in round 3. The implementation is `round6/r6_lib.py` and `round6/r6_monitor.py` (hashes in section 8).

Common calibration rule, the "null-calibrated level". The level is the smallest value on the grid 1.0 to 60.0 in steps of 0.5 (the round 3 to 5 grid of 1 to 30, extended) that meets three conditions. First, the null detection rate of the validation stream itself is at most NULL_TARGET = 0.10. That rate is the share of dense clocks, every 5 windows from the end of the burn-in to n - 150, that see an alarm event in [c + 7, c + 150]. Second, the 90th percentile of the same rate over 60 moving-block bootstrap resamples (block min(2 (G + 60), max(60, n/2))) is at most 0.10. Third, the 90th percentile of the false-alarm onset share is at most 0.05. If no level qualifies, the arm uses the grid maximum and the trace is flagged infeasible. Infeasible traces stay in the denominator and are reported.

An event is the decision at which a run of consecutive hits reaches the persistence. A run that began earlier and is still on does not create a new event, which is what onset localisation means.

- **V1, RF-NC.** The RF statistic with the null-calibrated level. (G, K) is chosen by validation detection (an event within [cp + 7, cp + 150]), then by the lower validation null rate, then by G.
- **V2, RF-NC-RA.** V1, plus the CUSUM held at 0 throughout the burn-in, which removes the release transient of `DIAGNOSIS.md` 2b, plus a re-arm that sets S to 0 after every event so that each alarm is a localised event.
- **V3, RF-NC-RA-WS.** V2 with the exponentially weighted reference warm-started on the 200 decisions that precede the scored stream, which removes the warm-up excess of `DIAGNOSIS.md` 2b. On test, these are the last 200 decisions of the trace's validation q stream (stored in `frozen_r6.json` as `q_warm_test`), which ends about 66 windows before the first test decision. On validation they are the first 200 decisions of the same stream (in-sample).
- **Fair baseline, STB.** A static CUSUM on q with the validation mean and sd (the round 1 M2 and round 3 ST construction), with K chosen on the same grid {0.5, 1, 2} and by the same rule as V1 to V3 (validation detection, then the lower validation null rate, then the smaller K), the same null-calibrated level, the same fallback (K = 2, grid maximum, infeasible), re-arm, burn-in hold and onset-localised scoring (same calibration and same scoring as V1 to V3). It differs from V2 only in having no reference-free standardisation, so it isolates what the reference-free part adds. It is the single registered baseline. At the freeze it found a feasible level on 20 of 20 P2 traces and 18 of 18 GEANT traces, with K = 0.5, 1 and 2 on 2, 15 and 3 of the P2 traces, and only one P2 trace (cesnet_inst152) sits at the level-grid maximum of 60. The earlier fixed K = 0.5 construction (feasible on 3 of 20, 17 of 20 at the grid maximum) was replaced before the commit (Disclosure 6).
- **Reported comparator, ADB.** ADWIN on the standardised delayed Bolt squared error, with delta the most sensitive value on {1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2} that meets the null rule. It is a residual detector, reported and not tested (Disclosure 3). It enters no success rule.
- **Report only, R5.** The frozen round 5 RF recipe (`rf_level` calibration on the round 6 validation stream, no re-arm), scored by the corrected criterion, to separate the effect of the criterion from the effect of the repair. A descriptive column also scores every arm by the round 4/5 placebo-clock rule on events (cap 100); the round 5 official figure stays 44 of 64.
- **Random control, RND.** For each trace and each arm, the arm's number of no-fault test events after the burn-in is placed uniformly at random among the post-burn-in decisions, 200 draws with seed 20261016 + trace index (position in the frozen trace list). Each draw is scored by the same per-trace rule as the arm. The random control does not react to the injection, so its injected and matched-control detections coincide. The control runs at each arm's own no-fault rate.

## 4. Per-trace conditions (corrected)

Change points: cp in {420, midpoint, n - 160} for the long design (CESNET, Abilene), so that the whole lag window [cp + 7, cp + 150] lies inside the stream; {200, 234, 268} for the GEANT short design. Each change point gets 10 test seeds, `a2.test_seed` with s = 6 (new seeds; the round 5 seeds used s = 0). One cap, L = 150, applies to injected runs, the matched control and the null clocks.

- **c1 (unchanged).** The no-fault false-alarm onset share is at most 5 %.
- **c2 (localised).** Shift specificity: the Newcombe 95 % interval of det - ctl lies above 0. Here det is the share of the 30 injected runs with an event in [cp + 7, cp + 150], and ctl is the share of the 3 matched no-injection clocks with an event in the same windows of the no-fault stream.
- **c3' (corrected, replaces the placebo-clock condition).** det - p0 >= MARGIN = 0.5, where p0 = max(p_dense, p_rand). p_dense is the dense-clock null rate of the no-fault test stream (clocks every 5 windows from the end of the burn-in to n - 150, events in [c + 7, c + 150]). p_rand is the probability that the trace's own number of no-fault events, placed uniformly at random among the M post-burn-in decisions, puts at least one event in a window of 144 decisions, computed as 1 - C(M - 144, N) / C(M, N). c2 must also hold.
  - Why a margin and not a p-value: the 30 injected runs share the pre-change stream within each change point, and the random control's detections come in clusters of 10. An exact binomial test of det against p0 therefore has the wrong size. On the random control it passes on 5 to 24 % of draws on the 64 round 5 traces (`val6_summary.json`, `rand_ctrl_binom_share_mean`, across arms and splits). The margin rule passes on 0.9 to 4.3 % (`rand_ctrl_c3only_share_mean`).
  - Why p0 is a maximum: the dense-clock rate uses the whole post-burn-in stream (not the warm-up prefix) and shows the clustering of real alarms. The random rate is the analytic rate-matched null and is the more conservative of the two when alarms cluster.
- **Joint pass J = c1 and c2 and c3'.**

## 5. Validation feasibility (design input; in-sample except the labelled revealed column)

`val6.py`. Selection and scoring both on validation: the 64 round 5 traces (`round5/r5_val`) and the 12 round 4 traces (`round4/r4_val`). The `revealed_r5_test` column applies the validation-selected parameters to the round 5 test streams that were already read (Disclosure 1). This table is the design input of the power calculation and is not a result of round 6. `val6.py` ran with the draft of `r6_lib.py`; the later edits (the change-point origin parameter `cp_lo`, the descriptive placebo-clock columns and the short-stream clocks of section 1) leave every quantity of this table unchanged, since the short-stream rule applies only when fewer than four dense clocks fit (never on the 64 round 5 and 12 round 4 validation streams).

| arm | feasible (r5 val, of 64) | J, r5 val in-sample | J, r4 val in-sample (of 12) | J, revealed r5 test (of 64) | mean null p_dense, revealed test | random-control c3' pass share, revealed |
|---|---|---|---|---|---|---|
| R5 (frozen round 5 RF) | 63 | 43 | 10 | 51 | 0.195 | 0.036 |
| V1 RF-NC | 49 | 58 | 11 | 54 | 0.067 | 0.014 |
| V2 RF-NC-RA | 48 | 62 | 12 | 55 | 0.072 | 0.013 |
| V3 RF-NC-RA-WS | 56 | 63 | 12 | 56 | 0.075 | 0.009 |
| STB (baseline) | 10 | 61 | 12 | 48 | 0.268 | 0.026 |
| ADB (comparator) | 1 | 26 | 6 | 26 | 0.488 | 0.017 |

Per source on the revealed test (J): V3 Abilene 46/53, CESNET 10/11; STB 37/53 and 11/11; R5 41/53 and 10/11. The V3 misses on the revealed test are, by cause: c2 fails with slow detection even at cap 150 (inst118, od104, od131), a high null with many no-fault events (od43, p_rand 0.94; od74, p_dense 0.71), and a matched-control alarm (od46). On the revealed test, McNemar on J: V3 against STB 12 against 4 discordant traces (one-sided p 0.038), V2 against STB 11 against 4 (p 0.059), V1 against STB 11 against 5 (p 0.105). The STB row of this table is the earlier fixed K = 0.5 construction (`val6.py` calls `select_stb` without validation injections, which keeps that construction and its recorded outputs); the registered STB of section 3 chooses K on the grid and is feasible on far more traces, so the row is a design input for the earlier version only. The strict null target of 0.10 is unreachable on the grid for 8 to 16 of 64 traces, which then run at the grid maximum; the static baseline is competitive in-sample but loses null control out of sample because its reference does not follow the validation-to-test level change.

## 6. Success rule, Holm and power

For each variant v in {V1, V2, V3}, with X_v the number of P2 traces with J, on the primary pool of n = 20 traces (the P2 count of section 1; GEANT is not in the pool):

1. **Count.** X_v >= ceil(0.8 n) = 16, the round 4/5 fraction, unchanged.
2. **Holm across variants.** p_v = P(Binomial(n, 0.5) >= X_v), a one-sided exact test against "the monitor works on at most half of the traces". The three p_v are Holm-adjusted and the adjusted p_v must be below 0.05.
3. **Random-control validity.** The mean over P2 traces of the random control's c3' pass share for arm v (the share of the 200 draws with det - p0 >= 0.5) is at most 0.05. If it is above 0.05, c3' is declared invalid for that arm and the arm cannot succeed.

The monitor claim succeeds if at least one variant meets 1 to 3. When several succeed, the paper names all of them, and the headline is the one with the largest X_v (ties broken by order of simplicity, V1 before V2 before V3). This choice is fixed before the test.

**Secondary comparative claim (not a condition of success).** "The reference-free part matters": exact one-sided McNemar on the paired J of v against STB, Holm across the succeeding variants, adjusted p < 0.05. STB has the same K grid, null rule and fallback as the variants, so the comparison isolates the reference-free standardisation. STB found a feasible level on validation for 20 of 20 P2 traces (against 13, 13 and 14 for V1 to V3), so it is a harder baseline than the fixed K = 0.5 version of the revealed re-analysis. The comparison is declared underpowered in advance for n = 20 (the revealed test needed 64 traces to reach p 0.038). Beating the baseline is therefore secondary: a failure of this comparison does not turn a success of the main rule into a failure, and a success of it is stated only when it holds.

**Power.** Computed on the actual P2 count by `power6.py` (`power6.json`, `power6_combined.json`, `power6_scenarios.json`), with traces treated as independent (the Abilene ones are not). The binding count is k = max(ceil(0.8 n), Holm count against 0.5); for n = 20 both are 16.

| n | k | per-trace pass probability for power 0.8 | power at pi = 0.84 | at 0.86 | at 0.875 | at 0.90 | at 0.95 |
|---|---|---|---|---|---|---|---|
| 20 (P2, primary) | 16 | 0.842 | 0.794 | 0.863 | 0.905 | 0.957 | 0.997 |
| 18 (P3 GEANT, descriptive only) | 15 | 0.870 | 0.677 | 0.762 | 0.820 | 0.902 | 0.989 |

The best available design input for the per-trace pass probability is the revealed round 5 re-analysis, at 0.844 (V1), 0.859 (V2) and 0.875 (V3). The in-sample validation values (0.91 to 0.98) are optimistic by construction. The relaxed series have more missing data and may be harder, and the frozen selection found a level meeting the null conditions for only 13 (V1), 13 (V2) and 14 (V3) of the 20 P2 traces and for 20 (STB) (section 8). `power6.py` therefore computes the power of V1, V2 and V3 on the 20 P2 traces (k = 16) for four scenarios, with the revealed pass rate by validation feasibility (V1: 0.816 feasible and 0.933 infeasible; V2: 0.833 and 0.938; V3: 0.875 and 0.875):

| scenario | V1 | V2 | V3 |
|---|---|---|---|
| every trace passes with the overall revealed rate (0.844, 0.859, 0.875) | 0.808 | 0.861 | 0.905 |
| the revealed feasible and infeasible rates applied to the frozen feasibility split | 0.857 | 0.894 | 0.905 |
| the split, and cesnet_inst159 and cesnet_inst217 fail (both pass with probability 0) | 0.504 | 0.569 | 0.605 |
| infeasible traces dropped (n = 13, 13, 14; k = 11, 11, 12), for comparison only | 0.563 | 0.628 | 0.749 |

The design keeps the infeasible traces in the denominator, because dropping them after the selection would let the selection decide which traces count. The third row is a downside scenario with two traces named in advance (inst159 is feasible and inst217 infeasible for all three variants). The V1 value of the first row is 0.81, not the 0.79 of the grid value at 0.84 in the table above. The stated power is therefore a range, from about 0.5 to about 0.9, and the lower end applies if those two traces fail. If the rule is missed, the miss is reported with these numbers and not reinterpreted.

## 7. Guards and conduct

- The `Restricted` view of `fresh_cells.py` raises on any test window during stage A, selection and the dry run. Stages A and B are guarded by that view and by `run_r6.sh` (stage B needs the commit marker and `REPAIR_STAGE=test`). `rcommon.install_guard()` and `guard_selftest()` run at the start of every monitor stage (select, dry run and test), and the select stage and the dry run additionally extend the guard to the `round6/r6_test` folder and check that a read of it raises. The folder `round6/r6_test` does not exist at the time of writing.
- The test stage refuses to run without `round6/PREREG_6_COMMITTED` and `REPAIR_STAGE=test`. `frozen_r6.json` (the per-trace parameters of every arm) is hashed into `frozen_r6.sha256` at the end of selection, and the test stage rechecks that hash, the hash of `pool6_manifest.json` recorded in the frozen file, and the hashes of `r6_lib.py`, `r6_monitor.py` and `r6_stages.py` recorded at freeze time. `run_r6.sh` also verifies `RECIPE6_FROZEN.json` and, before stage B, the hash of this file against `round6/PREREG_6_SHA256`.
- Dry run: `r6_monitor.py --stage select --dry-run` on the non-fresh Abilene aggregate (as in rounds 4 and 5) runs the full select and test code on pseudo data and prints `dry-run ok; test windows accessed: 0`. The log is `round6/dryrun_log.txt`.
- Order of the remaining work: coordinator commit with `PREREG_6_SHA256` and `PREREG_6_COMMITTED`; then `run_r6.sh` (stage B with `REPAIR_STAGE=test`, then the test stage); then the paper and thesis update in the same round.
- Resources: `run_r6.sh` is resumable and systemd friendly (`nice -n 19`, `taskset -c 8-11`, at most 2 workers, waits while MemAvailable is below 10 GB, a progress line every 5 minutes in `round6/PROGRESS.md`). No git by the worker.

## 8. Code manifest and frozen state (sha256)

| file | role | sha256 |
|---|---|---|
| `round6/r6_lib.py` | arms (STB with the K grid), null-calibrated selection, corrected per-trace conditions, short-stream rule | `32a0a541ccabff7ec8b66550b75eb3f472cfb8157a1a66ddce38f7725f8fb0d7` |
| `round6/r6_monitor.py` | stage select / test, guards, summary, success rule | `581a2f4b1c070cd2a2c302968c17a5a5b7eaf59be63e11508ba5b1ec049a910f` |
| `round6/r6_stages.py` | stage A / B for the pool-6 cells | `37c11cdd42c141b962b2f0bf5aabea424ff4ffe3b06e89c076daeb0cba42d1ce` |
| `round6/run_r6.sh` | resumable pipeline (stage A, select, stage B, test) | `656092ba2559d1b04d40872603e11abb4149b2eac010b409b4193ed3c5bdba85` |
| `round6/build_pool6.py` | trace builder (counts only) | `e517c97a9db878f7800976a6f01cf50ec39e2c0c2f5f689bd5f57f7a4ac8df1a` |
| `round6/freeze6.py` | code and data hashes | `b582b0b9e8629ec1e7c978556768890784e47115d899905f8ecd939a97792616` |
| `round6/power6.py` | power | `c54cc451bc9b8e7a6b82da0d4a3484a187753cebed82b19f3f41aa2ba2a9aa0f` |
| `round6/diag6.py` | diagnosis (revealed round 5 test streams, Disclosure 1) | `68b7ff0a2b2e1b497f97efc6579269f084bbd2017c988b830151f0dc725f4a28` |
| `round6/classify6.py` | cause classification | `1e747f6064a8dc1d41c4eb1f221a04f9ed2e4ff32f067d47ccf4690118b309d1` |
| `round6/val6.py` | validation feasibility and revealed re-analysis (design input) | `78412c0bf4588166ba79fe8c493e3da5f9f2f9874454d51bc0f2239d1cd5ed27` |
| `round6/inv6.py` | inventory (no round 6 trace) | `9d3beb54cad0ec23787626a12f8e380037718f3fb6b2fbb35005dd3214c33ace` |
| `round6/classify6_r6.py` | registered cause classes of the failed traces (section 9), run after the test stage | `243fb0643d46178ce00df0f3b11292bb30d0b06a5a8f95ff4d1d3491466b81bd` |
| `round6/DIAGNOSIS.md` | diagnosis text | `d8e7ed0a8c519971471d6ceb2343c0d4314805c837b6e0d45808a2773c6b19b3` |

Data and frozen state:

| file | role | sha256 |
|---|---|---|
| `round6/pool6_manifest.json` | pool manifest, 148 candidates, 20 + 18 monitor traces | `46757dcba4b7e748972f3adc5b22840436cb721e64ecb1c262e6763f3ac1db86` |
| `round6/reserved_round6_M.json` | additive block round6_M for reserved_series.json (merged by the coordinator) | `11d6a77079252b5cf7bb4a6f821afbeb43ca7ce60fc6e070db4f45e65d19a314` |
| `round6/power6.json` | power, rules A and B | `55f52160c1241a26eacaddcd8f093b39e99f1fa73e12645963af484044c86e5b` |
| `round6/power6_combined.json` | power, combined rule | `1c1c79114c2b946692ecf678ed4c890bd5aa518e672f3a41000e4bd7439d0161` |
| `round6/power6_scenarios.json` | power scenarios of section 6 (frozen feasibility, revealed pass rates) | `72cb2a4c8c17e5b7ef2748f17ab43eb8e60bfceb113f0b6ccd5a276e539ba5a9` |
| `round6/dryrun_log.txt` | dry-run log | `6813693df0360c45a17f7a9a7906826c1fbc784ac9f0c29f692f76d64f8c3d05` |
| `round6/RECIPE6_FROZEN.json` | E-Q recipe manifest, hashes of the reused modules | `453c6748671209f3975a9cd4c8921aa4f5487c4510161655b6984030d2dc4a16` |
| `round6/frozen_r6.json` | frozen per-trace parameters of every arm (38 traces) | `f170954fdaa3400460798ed094e846702ffacfd5e82b40ecfda60421b0e114f5` |
| `round6/frozen_r6.sha256` | sha256 of frozen_r6.json | `48ff92ec505b43c61198cf915286fc746d40140340892bea012cf3695be47ecf` |
| `round6/R6_VAL_SHA256.txt` | per-file sha256 of the 38 validation forecast files | `98e999867997c414114fc3f9fd6f68be9eb8476ff9504cfce9fffc841f2d39ee` |
| `round6/val6_rows.csv` | validation feasibility rows | `a31598a3830f95aef904a0072f011c28c81259cbb04255fbd74dc9c8c1619a9e` |
| `round6/val6_summary.json` | validation feasibility summary | `05c1d825030ef79992864dd1931ff64b275f76f1b6dd24392bfaed3bbb6229b3` |
| `round6/val6_params.json` | validation selections of the feasibility run | `8a744f9e21c89dba8d8fbd16a2089728ec8fb9baf2e522f122f2ec6955af0da4` |
| `round6/diag6_summary.json` | diagnosis summary | `8a5b93f50788d6454b9a902c80ee57bb9cc8c8cc1487dfdc8369048d068c39a3` |
| `round6/diag6_classes.json` | cause classes | `5116ff6068d965eaa4bb87c0f67253fa899273a7cba6275836650123a7a1f164` |
| `round6/inv6_summary.json` | inventory summary | `909e39e6cb8623efe8629828ea64100b415b3dd217f6f4d5780c646b514c5673` |

- `round6/r6_val` (38 files, h = 6 validation forecasts of E-Q and Chronos-Bolt): sha256 of the sorted per-file sha256sum list `98e999867997c414114fc3f9fd6f68be9eb8476ff9504cfce9fffc841f2d39ee`. `round6/r6_models` (38 joblib files): `3c8e3c0656e90c51db97ffa05bc17dcf034cdc66604e8f6b254007e01853ec11`.
- Modules of earlier rounds imported unchanged (`a2.py`, `stats_tools.py`, `rcommon.py`, `eq_recipe.py`, `r3_stages.py`, `r3_monitor.py`, `r4_monitor.py`, `r5_stages.py`, `fresh_cells.py`, `build_pool4.py`, `monitor_far.py`, the round 3 and round 5 recipe files and `frozen_r4_guard.json`): their sha256 are in `RECIPE6_FROZEN.json` (`unchanged_from_earlier_rounds`) and are checked by `run_r6.sh` at every start.
- `reserved_series.json` is not edited by round 6. The block `round6_M` in `reserved_round6_M.json` is merged by the coordinator; its hash is recorded here before the merge.
- Commands: `build_pool6.py`, the dry run and the pipeline are listed in `RECIPE6_FROZEN.json` (`commands`); the select stage ran (second run, Disclosure 6) as `UNTIL=select nice -n 19 taskset -c 12-15 bash run_r6.sh` on 2026-10-12 and wrote the hashes to `round6/RUN_NOTES_R6.md` (to be copied to `paper_a/RUN_NOTES.md` by the coordinator).
- `PREREG_6_SHA256` and `PREREG_6_COMMITTED` are written by the coordinator; this file's own hash is not listed in itself.

Frozen validation state (`frozen_r6.json`, 38 traces). Arms that found a level meeting the three null conditions on the validation stream (feasible / traces); the others run at the grid maximum and stay in the denominator:

| arm | P2 (20 traces) | P3 GEANT (18 traces) |
|---|---|---|
| R5 | 19/20 | 17/18 |
| V1 | 13/20 | 16/18 |
| V2 | 13/20 | 17/18 |
| V3 | 14/20 | 18/18 |
| STB | 20/20 | 18/18 |
| ADB | 0/20 | 7/18 |

The P2 feasibility (V1 13, V2 13, V3 14 of 20) is lower than on the 64 round 5 traces (49, 48 and 56 of 64). The infeasible traces keep a level of 60 and are scored on the test like any other, so they can still pass; the revealed re-analysis passed 54 to 56 of 64 with 8 to 16 infeasible traces. The power of section 6 is given for the scenarios there. STB is feasible everywhere because its K grid and its static standardisation let the null rule be met at a high level (levels up to 60 on validation), and its test behaviour can differ because its reference does not follow the validation-to-test level change.

## 9. Outcome text (fixed now)

**Success** (at least one variant meets section 6). The paper says: "A reference-free CUSUM on the expert error ratio, with its level calibrated on validation so that chance alarms in a 150-window horizon stay at or below 10 % on the traces where that target is reachable, and with re-armed, onset-localised alarms, detected injected shifts above its own no-fault alarm rate, above a rate-matched random-alarm control and specifically against a matched no-injection control on X of 20 drifting traces (CESNET institutions and Abilene OD pairs), against the 16 required, with a Holm-adjusted one-sided binomial p of P against the hypothesis that it works on at most half of the traces." The paragraph states the calibration counts (the level met the 10 % target on validation on 13 of 20 traces for V1 and V2 and on 14 of 20 for V3, and sat at the grid maximum of 60 on the other 7 and 6, which stay in the count), and states that the 20 traces are 11 never-used series and 9 reused F+G series (Disclosure 4), reported separately, not as 20 fresh traces. The paragraph names the succeeding variants and the headline variant, gives the per-source and per-stratum rates with Wilson intervals, the STB and ADB rates, the R5 rate under both rules, the GEANT descriptive stratum and the random-control size. It repeats Disclosures 1, 2, 4 and 5 in substance, states that round 6 is the third monitor attempt and that rounds 4 and 5 failed under their own rules (round 4: 9, 11 and 8 of 12 against 10; round 5: 58, 62 and 44 of 64 against 52), and says that the corrected criterion was motivated by the round 5 test result. The comparative sentence about STB appears only if the McNemar claim holds. On success the monitor result may enter the contributions list, the results section, the highlights and thesis chapter 6, each with this disclosure. It does not enter the abstract.

**Failure** (no variant meets section 6). The paper reports X_v for each variant and the baselines with Wilson intervals, the per-source and per-stratum table, the power statement of section 6, and the measured cause. The cause is assigned by the rule registered here and implemented in `classify6_r6.py`, applied to the round 6 rows of the 20 P2 traces: every failed pair of variant and trace takes the first matching class of (1) c1 false-alarm excess, c1 false; (2) c2 not shift-specific, c1 true and c2 false; (3) c3' null exceeded, c1 and c2 true, c3' false and p0 = max(p_dense, p_rand) above 0.5, so that the margin is unreachable even at full detection; (4) c3' slow detection beyond the cap, c1 and c2 true, c3' false and p0 at most 0.5, so that the detection share within 150 windows stays below p0 + 0.5. The counts per class, variant and stratum, with the trace names, go to `r6_causes.json` and are reported. Because the rule is registered before any stage B forecast, the classification is not post hoc. It describes causes and is not a test. The paper states that three registered attempts at the monitor claim failed, and no monitor claim enters the abstract, contributions, highlights or cover letter. The thesis chapter carries the same negative result. No round 7 threshold is introduced from the round 6 result without a new registration that discloses it.
