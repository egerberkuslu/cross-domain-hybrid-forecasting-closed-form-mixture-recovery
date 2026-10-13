# REPAIR PRE-REGISTRATION 4, Paper A: guarded quantile-expert mixture on a second fresh pool, and the reference-free monitor with a placebo control

Written 2026-10-12 (round 4, phase 1). Rounds 1 to 3 (`REPAIR_PREREG.md`, `REPAIR_PREREG_2.md`, `round3/REPAIR_PREREG_3.md`) are frozen and untouched. This file is FROZEN when the coordinator replies "PREREG 4 COMMITTED". Until then no accuracy of any pool-4 series has been computed (not on validation windows either) and no test window of the pool has been read by model code. Folders: `experiments/repair_2026-10-12/round4/` and `experiments/repair_2026-10-12/fresh_cells/`.

## 0. Why round 4, and a disclosure

Round 3 (100 fresh component cells) missed its rule narrowly: 11 Holm-significant wins against 8 required and none worse, but a pooled gain of 0.499 % against 0.5 % with intervals that include 0, a median cell gain of 1.0 %, and one Abilene series (validation weight 0.6 to 0.7, test RMSE 4.9 to 10.6 % higher) that pulled the mean down. The reference-free monitor met two of its three conditions (9 of 12 traces on the third, 10 required).

Disclosure. The round 3 TEST results motivated both changes below: the weight guard of section 1 (the one-series loss tail) and the placebo control of section 5 (the three traces whose clean stream alarms). The guard was chosen on round-3 VALIDATION arrays only, by a rule listed before it was evaluated; the placebo control was designed from the structure of the round 3 control, not tuned on any pool-4 result.

## 1. The weight guard (chosen on round-3 validation arrays, no test data)

Candidates (listed before the selection was computed), applied to the Bates-Granger weight alpha of E-Q against Chronos-Bolt: G0 alpha (the round 3 rule); G1 fall back to alpha = 0 when the gain of the weight fitted on an earlier part of validation, measured on a later part, is below 0.5 %; G2(l) l x alpha for l in {0.25, 0.5, 0.75}; G12(l) G1 then G2.

Selection (`round4/r4_guard.py`, output `guard_selection.csv`, `frozen_r4_guard.json`, sha256 `566443299bcb80eed5df57d7f9a7afa9cb96b3d07c1185b9ef4a59a9c161a4d8`). The 100 round-3 validation arrays (validation windows only, E-Q trained on train only) are split per cell into A (first 40 %), B (to 70 %) and C (rest) with purges of h windows; alpha_A is fitted on A, the guard decision uses B, the score is the mean over the 100 cells of log(RMSE_C(variant) / RMSE_C(Bolt)); the lowest score is chosen, ties within 0.0002 to the simpler. Scores: G0 +0.0192, G1 -0.0022, G2(0.25) -0.0059, G2(0.5) -0.0023, G2(0.75) +0.0070, G12(0.25) -0.0039, G12(0.5) -0.0049, G12(0.75) -0.0041. Chosen: G2(0.25), alpha_final = 0.25 x alpha_full (alpha_full on all validation windows, snapped to 0.05, the product not snapped). Per source the chosen variant scores -0.0083 (Abilene), -0.0043 (CESNET), -0.0055 (GEANT); G0 scores -0.0097, +0.0371 and +0.0244. (G0 has a median of -0.97 % and a mean of +1.92 %: its validation failures are a tail, as in round 3.) The null expert uses the same transformation.

Consequence stated before the test: the shrink 0.25 retains a fraction 2 l - l^2 = 0.4375 of the attainable mean-squared-error gain of the unshrunk weight (quadratic mixing loss). A guard that is right about validation drift is therefore expected to cut a 1.0 % cell gain to about 0.44 %.

## 2. Frozen recipe (hashed) and the new pool

Recipe: the round 3 E-Q recipe unchanged (`round3/eq_recipe.py` sha256 `d45d3ae7ea731d580d3199ce27f5c6f4cd48cee5bc52e2857fa4597bc9cfcf24`, `round3/RECIPE_FROZEN.json` `5a96181ebe5633f76daf34646a301c6d5453e2b60e8187b99a57f258c590cde8`, features and hyper-parameters of `round2/experts.py` `12dd347a...`, Bolt-small snapshot `772f3d25...` on CPU float32), train-only models for the validation forecasts, a refit on train plus validation for the test forecasts, plus the guard of section 1. All hashes, including the round 4 code, are in `round4/RECIPE4_FROZEN.json` (sha256 `1c06e8ca852bfea6460f84be5dc0b62c0144124f0ba1609d228d6cc4624d0f71`).

Pool. The remaining admissible series (admissible rule of `fresh_cells.py`, unchanged: CESNET 200, Abilene 128, GEANT 329) minus `partition_F.json` (registered and pilot) minus the round 3 list leave CESNET 90, Abilene 20 and GEANT 222. They were split deterministically between this registration and the F+G round 5: sorted by id (GEANT by numeric (src, dst)), even positions to Paper A, odd positions to F+G; both lists are in `fresh_cells/reserved_series.json` (sha256 `e21c458f1c6c0932866c9189f5f27cab131cbab13ea6fbe2994d76feaf016a34`, blocks `round4_A` and `round5_FG`). The A half holds CESNET 45, Abilene 10 and GEANT 111.

Accuracy sample, N = 60 series (CESNET 30, Abilene 10, GEANT 20) drawn from the A half by the stride rule (every k-th, k = floor(n_A / N), from the first; Abilene takes all 10), at the five horizons: 300 cells. N was chosen from the power calculation of section 3 and compute (about 75 s of worker time per cell: 300 cells need roughly 3 h on two workers; N = 75 would add about 40 minutes and raise the empirical-scenario power from 0.53 to 0.60). Manifest `fresh_cells/pool4_manifest.json` (sha256 `7988ca52a7c3639558168b3aa1df22c5ad6fcb457761c2f036814aaf4d1aa2e8`, series file hashes, window counts, train-to-validation shift of each series).

As in round 3 the series are components of the networks of the paper aggregates, so they are different forecasting problems but not independent of the aggregates, and the pool-4 series belong to the same networks and periods as the round 3 pool.

## 3. Power calculation (round-3 effect distribution, no pool-4 data)

`round4/power_calc4.py`, `power_guard4.py` (outputs `power_calc4.json`, `power_guard4.json`). A pool-4 series copies the five (gain, se) cells of a round 3 series of the same source, drawn with replacement, so that series-to-series heterogeneity is the empirical one; se = bootstrap interval width / 3.92. Scenarios: MEDIAN, true gain = 1.02 % (the round 3 median cell gain) x a lognormal(0, 0.5) series factor; EMP, the round 3 estimates themselves (mean 0.50 %, heavy negative tail); TAIL, the round 3 estimates with negative values set to 0. 400 simulated pools per setting; the complete rule of section 4 with K = ceil(0.08 x cells).

| series (cells) | K | MEDIAN | EMP | TAIL |
|---|---|---|---|---|
| 40 (200) | 16 | 0.96 | 0.37 | 0.94 |
| 60 (300) | 24 | 1.00 | 0.53 | 0.96 |
| 75 (375) | 30 | 1.00 | 0.60 | 0.95 |

N = 60 satisfies the criterion (power at least 0.8 at the round 3 median effect). The EMP power is limited by the empirical mean sitting at the 0.5 % minimum and does not grow quickly with N; the rule cannot confirm an effect that is not above its minimum.

Effect of the guard on the effect-size thresholds. The simulation is scale-free: if the guard multiplies every gain and its se by the retention 0.4375, the same power holds when the minimum effects are multiplied by 0.4375 as well. With the minimum kept at 0.5 % and effects scaled, power at N = 60 is 0.57 (MEDIAN), 0.01 (EMP) and 0.47 (TAIL) (`power_guard4.json`): a registration that keeps the unscaled 0.5 % minimum under a guard chosen to shrink the gain would be built to fail. The primary rule therefore scales the minimum effect by the retention, 0.5 % x 0.4375 = 0.219 %; the unscaled 0.5 % version of the identical rule is reported as a secondary criterion. This is a departure from "same thresholds as round 3" and is flagged for the coordinator before the commit.

## 4. The accuracy rule (part 1)

Comparison, controls and statistics are those of round 3 (section 4 of `round3/REPAIR_PREREG_3.md`): per cell, the guarded M2 mixture Bolt + alpha_final (E-Q - Bolt) against Chronos-Bolt, seed-free, paired two-sided sign-flip block permutation (block h, B = 20,000), Holm over all 300 cells, relative RMSE change with a moving-block bootstrap interval, TOST and identical cells reported; random controls (a) E-Q rows permuted in blocks of h with the same weight and (b) the null expert E-Q0 with its own guarded weight; better control = the lower cell RMSE. Per source reporting (CESNET, Abilene, GEANT: cells, wins, mean and median gain) is part of the registration.

Success (all required; m = 0.219 % is the primary minimum effect, 0.5 % in the secondary version):

1. at least ceil(0.08 x 300) = 24 cells better than Chronos-Bolt after Holm with a relative RMSE gain of at least m;
2. no cell significantly worse after Holm;
3. the pooled gain over Chronos-Bolt (mean over cells) is at least m, and its lower 95 % bound is above 0 under the cell-level bootstrap and under the series-clustered two-stage bootstrap (series resampled, then horizons within a series, B = 10,000);
4. the pooled gain over the better of the two controls has a lower 95 % bound above 0 under both bootstraps;
5. the median cell gain over Chronos-Bolt is above 0.

Per outcome. Success: the paper states that a quantile expert on lag and calendar features with a validation-chosen shrinkage of its mixture weight improves on Chronos-Bolt on fresh component series of the same networks, with the cell counts, per-source results, pooled gain and both intervals, discloses the round 3 motivation of the guard and the retention-scaled minimum, and reports the secondary 0.5 % version; the abstract, contributions, highlights, cover letter and thesis chapter 6 are updated consistently, stating the confirmed claim as "E-Q complements Chronos-Bolt". Failure: reported with the per-source table and the power statement of section 3; if the secondary rule is met and the primary is not (impossible by construction for the thresholds, since 0.5 % exceeds m) nothing further is claimed.

## 5. The monitor (part 2): replication with a placebo control

Detectors, calibration, grids, validation budget, burn-in (100 decisions counted in every false-alarm denominator), injection and the shift-specific test are those of round 3 (`round3/r3_monitor.py`; RF, ST, AD). Traces: the A-half CESNET and Abilene series (GEANT test streams of 428 windows are too short for the design below) whose train-to-validation level shift is at least 0.3 train standard deviations and whose test stream has at least 690 windows, at h = 6: cesnet_inst3, 37, 49, 111, 262, 275 and abilene_od6, 35, 49, 63, 76, 89, 12 traces (`pool4_manifest.json`, `monitor_traces`); two of them (cesnet_inst262, 275) are outside the accuracy sample and get h = 6 cells only. The monitor uses the unguarded forecasts of E-Q and Bolt (the guard acts on the mixture, not on the error ratio).

New control. The round 3 permuted-clock control scored the alarms of an injected run against another placement's clock; on traces whose clean stream already alarms it detected as often as the true clock. Round 4 places the injections after a long clean segment and uses placebo clocks inside it.

- Injection change points: cp in {420, 420 + 0.5 (n - 110 - 420), n - 110} (window indices of the test stream, n at least 690), ten injection seeds: 30 injected runs per trace and detector.
- Matched no-injection control: the clean stream scored at the same three cps, lag cap 150, as before.
- Placebo clocks: P_j = 190 + 30 j for j = 0..4 in the clean prefix (after the 100-decision burn-in). The clean stream's alarms are scored as if a change point were at P_j, lag cap 100; the placebo window ends at most 410, before the first injection at 420. Condition 3 compares the detection of the injected runs within lag 100 with the placebo detection rate (5 trials per trace and detector), Newcombe 95 % interval.

Success for RF (all required, n = 12 traces, 10 required):

1. no-fault false-alarm rate (all decisions in the denominator) at most 5 % on at least 10 traces;
2. shift-specific detection against the matched no-injection control (Newcombe interval above 0) on at least 10 traces;
3. detection above the placebo rate (Newcombe interval above 0) on at least min(10, number of traces satisfying condition 2) traces.

Power (`round4/power_monitor4.py`, `power_monitor4.json`). From the round 3 per-trace results (false alarm met on 11 of 12, shift-specific on 12 of 12; Jeffreys-smoothed 0.85 and 0.99 for the probability that a rule with 10 of 12 is met), the unknown per-trace probability p3 of condition 3 under the new placebo control gives a rule power of 0.33 (p3 = 0.75, the round 3 value under the old control), 0.62 (0.85), 0.75 (0.90), 0.82 (0.95) and 0.84 (1.0). The number of traces is fixed by the pool (12 drifting A-half traces of adequate length), so power cannot be raised by sampling more; the rule has power 0.8 only if the placebo control is passed by about 95 % of traces. The registration accepts that, and a miss will be reported with the per-trace table and the power statement.

## 6. Code manifest, dry runs and conduct

| file | role | sha256 |
|---|---|---|
| `round4/r4_guard.py` | guard selection on round-3 validation arrays | `84983f9a2c348531958683fefe70549731150024c06cd24ee91ab2ab56175983` |
| `round4/r4_stages.py` | stage A and B for pool-4 cells (302 cells: 300 accuracy and 2 monitor-only at h = 6) | `32aadadc9f220f18347ab272eee666f959d69a0f927b2f0dbcb7242aad30ff23` |
| `round4/r4_eval.py` | guarded weights, test statistics, success rule | `fca4b390687521c63cb9825ab27db2c6bb9c0c74e397567313168c3bb87cce50` |
| `round4/r4_monitor.py` | RF, ST, AD with clean-segment placebo | `86f9deccbd028214da37a6cc656d5fdbed4806c01ab1519f4bf81c6fb2f432b1` |
| `round4/power_calc4.py`, `power_guard4.py`, `power_monitor4.py` | power calculations | `3e766cd6...`, `63c42402...`, `246741aa...` |
| `fresh_cells/build_pool4.py`, `fresh_cells.py` | pool-4 builder, reusable module | `44a5dab666886d1381476e9457ff4f7fe6d835058ec7db7e1118b0bd87ed9060`, `f4f5a4da...` |
| round 3 modules reused unchanged | `eq_recipe.py`, `r3_stages.py`, `r3_eval.py`, `r3_monitor.py` | in `RECIPE4_FROZEN.json` |

Dry runs, 2026-10-12, `round4/dryrun_log.txt`: `r4_stages.py --stage A --dry-run` (the non-fresh Abilene aggregate, a quarter of the training windows, RMSE Bolt 0.1973 and E-Q 0.1984, a pseudo stage B, a shape-only check of a pool-4 series with no model fit and no accuracy, 302 cells), `r4_eval.py --stage select --dry-run` and `r4_monitor.py --stage select --dry-run` (full select and test code on non-fresh pseudo data), each printing `dry-run ok; test windows accessed: 0`. Accuracy is computed only on non-fresh cells in the dry runs. The `Restricted` view raises on test windows; stage B and the test stages need `PREREG_4_COMMITTED` and `REPAIR_STAGE=test`; frozen JSON files are hashed before the test read and re-checked.

Resources (est.): stage A about 302 cells x 35 s, stage B about 302 cells x 45 s, monitor select about 15 minutes; `nice -n 19 taskset -c 8-11`, at most two workers, Bolt on CPU float32, 5-minute progress lines in `paper_a/PROGRESS.md`. No git.

Phase 2 order after the commit: stage A for all cells, `r4_eval.py --stage select` and `r4_monitor.py --stage select`, record the hashes of `frozen_r4.json`, `frozen_r4_monitor.json` and the stage-A files in `paper_a/RUN_NOTES.md`, `touch PREREG_4_COMMITTED`, stage B, `r4_eval.py --stage test`, `r4_monitor.py --stage test`, then the paper and thesis updates in the same round.

## Coordinator decision (2026-10-12, before any pool-4 accuracy is computed)
The retention-scaled minimum effect is CONFIRMED as primary: 0.5 % × 0.4375 = 0.219 % (pooled gain and per-cell win threshold), with the unscaled 0.5 % rule reported as secondary. Rationale: the guard G2(0.25) was chosen on round-3 validation arrays only and mechanically retains 0.4375 of the attainable MSE gain; keeping 0.5 % would make the design unable to pass by construction (power 0.01 under round-3 effects). The paper will state both thresholds and which one the claim rests on. N stays 60 series (power ≥ 0.8 at the round-3 median effect).
