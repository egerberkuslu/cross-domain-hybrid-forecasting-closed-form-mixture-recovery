# REPAIR PRE-REGISTRATION 2, Paper A (CHA-S): complementary expert and drift-adaptive monitor

Written 2026-10-12 (round 2, phase 1). The frozen file `REPAIR_PREREG.md` (commit 5e142c0) is not edited. This file is FROZEN once the coordinator replies "PREREG 2 COMMITTED"; until then no test split is read by any round 2 code. Sections 1 to 6 are fixed before any test read. Section 8 (Gate E0 result and expert manifest) is appended after a validation-only run whose definition is in section 3 and whose code hashes are listed in section 7.

Folder: `experiments/repair_2026-10-12/paper_a/round2/` (code, validation forecasts of the experts, saved models); it imports the round 1 modules in `experiments/repair_2026-10-12/paper_a/` (`rcommon.py`, `stats_tools.py`, `a2.py`, `gate0.py`).

## 0. What round 1 left and what round 2 attacks

Round 1 (REPAIR_PREREG.md, results in `RUN_NOTES.md`) failed all four rules on the held-out test: accuracy headroom of the CHA-S mixture (Gate 0a 0 of 25 cells, test-oracle constant mixture beats Chronos-Bolt by at most 0.57 %), the endpoint bound, the shift monitor (no-fault false alarms at most 5 % on 3 of 5 traces and shift-specific detection on 3 of 5) and the learned weight.

Two measured causes remain. First, the ceiling over Chronos-Bolt is set by the two experts: the decomposition path has a mean validation error variance ratio of 2.41 and a mean error correlation of 0.66 against Bolt (`ceiling_r_rho_cells.csv`), which caps an oracle constant mixture at 0.39 % on average. A mixture can beat Bolt only if a second expert has Bolt-level accuracy and errors that are not almost the same as Bolt's. Second, the monitor fails on NAB because the alpha stream moves between validation and test (`a2_diag.csv`: NAB AWS-CPU mean 0.21 to 0.32, share above the validation-designed threshold 0.04 to 0.53), so a threshold designed on a validation stream cannot hold 5 %.

Round 2 therefore tests (part 1) a stronger complementary expert in the mixture and (part 2) a drift-adaptive monitor whose reference is a trailing window and not the validation stream.

## 1. Data roles and enforcement (unchanged from section 1 of the frozen file, plus the expert training rule)

- Roles per cell: CAL = first 60 % of the validation windows, SEL = last 40 % after a purge gap of h windows, FINAL FIT = all validation windows, TEST sealed. Validation windows are the windows of the preprocessed validation split; the stored validation forecasts of Bolt and of the decomposition path (five seeds) are the ones of round 1.
- Expert training uses TRAIN windows only (the windows of the preprocessed train split, last target timestamp before the validation start). The expert code reads the preprocessed dataset through `Restricted`, a view that exposes the train and val windows and raises on a request for test windows; the val stage asserts afterwards that no test access was recorded. Models are fitted once and saved (`models/<dataset>__h<h>.joblib`); the phase 2 test stage predicts the test windows with these saved models and does not refit anything.
- Chronos-Bolt quantiles on train and validation windows are recomputed on the GPU and checked against the stored validation forecast (`max_dbolt_vs_stored`, `max_dy_vs_stored`, both 0.0 in the logs), so the experts see exactly the Bolt forecast of CHA-S.
- Test guard: `rcommon.install_guard()` (round 1) blocks `np.load` and `open` on `outputs/predictions`; `allow_test()` needs `PREREG_COMMITTED`, `REPAIR_STAGE=test` and, for round 2 code, the file `round2/PREREG_2_COMMITTED`. Every fit and selection script calls `guard_selftest()`, which proves that a real test read raises.
- Statistics and decision rules of section 2 of the frozen file apply unchanged: paired two-sided sign-flip block permutation on seed-averaged per-window squared error (block h, B = 20,000), Holm across the 25 cells, moving-block bootstrap CI, TOST with a 5 % margin for equivalence, identical forecasts reported as identical, pooled gains with cell-level bootstrap (B = 10,000), dataset-clustered two-stage bootstrap as robustness.

## 2. Part 1: candidate complementary experts

All experts forecast the scaled windows. Features of a window (lookback 168 hourly values): the last value, the lags 2, 3, 4, 6, 12, 24, 48, 72 and 168 as differences to the last value, the means of the last 6, 24 and 168 values as differences to the last value, the standard deviations of the last 24 and 168 values, the OLS slope of the last 24 values, sine and cosine of the target hour and weekday and the horizon step k. Targets are y_k minus the last value, so that the models do not extrapolate a level.

- E-Q, probabilistic expert. Scikit-learn `HistGradientBoostingRegressor` with the quantile loss at 0.1, 0.5 and 0.9 on the stacked (window, step) rows, 150 iterations, learning rate 0.06, depth 4, minimum leaf 40, L2 1.0, no early stopping, `random_state` 0. The point forecast is the median; the interval width q90 minus q10 is the expert's own scale.
- E-R, regime-aware seasonal-level expert. A regime variable r = std(last 24) / std(last 168) cut at the train terciles, and per (regime, step) a ridge regression (penalty equal to the number of regime windows, intercept free) of y_k minus the last value on five terms: the 7-day same-hour profile minus the last value, the 24-value and the 168-value means minus the last value, the last difference and the 6-value mean minus the last value. Deterministic.
- E-B, Bolt-conditioned residual expert. The same features plus Bolt's median minus the last value, the 0.1 to 0.9 and 0.3 to 0.7 quantile widths and the mean Bolt median minus the last value; `HistGradientBoostingRegressor` with the squared loss (100 iterations, learning rate 0.03, depth 3, minimum leaf 60, L2 5.0) fitted on the train residual y_k minus Bolt's median at step k; the forecast is Bolt's median plus the predicted residual. This expert is the strongest candidate and also the one that changes the nature of the claim: if it is selected the paper's contribution becomes "foundation forecaster plus a learned residual correction", and the text will say so.
- E-X, reference only: XGBoost on the raw lookback with the published pipeline configuration (300 trees, depth 5, learning rate 0.05, seed 42). It is evaluated by Gate E0 for comparison and is never eligible as the headline expert.
- Null experts of the same family: E-Q0 and E-B0 are trained exactly as E-Q and E-B on feature rows shuffled across the training windows (labels intact, features decoupled, permutation seed 0). They are the family-matched random control of part 1.

Hyper-parameters above are fixed and were not tuned on validation or test.

## 3. Gate E0 (validation only, phase 1)

Definition (code `round2/gate_e0.py`). For each expert and each of the 25 cells, from the validation forecast of the expert and the stored validation forecasts of Bolt:

- r = MSE_E / MSE_Bolt on CAL, rho = correlation of the two pooled error series on CAL;
- g_in = in-sample gain (RMSE) of the best [0, 1] mixture of Bolt and E over Bolt on CAL (the oracle ceiling, the Bates-Granger closed form);
- g_oos = gain on SEL of the mixture whose weight (grid 0.05) is fitted on CAL, over Bolt on SEL;
- g_3 = the same for NNLS stacking of [Bolt, decomposition, E] fitted on CAL (weights at least 0, sum rescaled to at most 1), averaged over the five decomposition seeds;
- xfit = cross-fitted block headroom (round 1 Gate 0a, L = 48, adjacent blocks) of the [Bolt, E] pair on the whole validation slice, relative to the constant weight.

Rule, fixed here. Gate E0 passes for an expert iff (a) r is at most 1.2 in at least 15 of the 25 cells, (b) g_in is at least 1.0 % in at least 8 of the 25 cells, and (c) g_oos is at least 0.5 % in at least 6 of the 25 cells. The coordinator's two numbers are reported next to this rule and are not additional pass conditions: the median r and rho, and the number of cells with r at most 1.2 and rho at most 0.8. Condition (b) uses the oracle gain, which already contains r and rho, so an expert with rho above 0.8 can pass through a low r (an expert better than Bolt on its own) and is then described as a residual or replacement expert, not as a complementary one. The numbers of cells (15, 8, 6) were chosen before the run. g_3 and xfit are reported, and a time-varying variant is added to the test only if xfit is at least 0.5 % in at least 6 cells (it is not used for the pass decision).

E-X and the null experts are reported and are not eligible. The experts that enter the part 1 test are listed in section 4 (they were fixed after the Gate E0 result of section 8 was seen, which is disclosed there).

If no expert passes, the accuracy claim keeps "equivalent to the foundation expert and readable", the part 1 test is not run, the measured r, rho and gains of each candidate are the reported reason in the paper and the thesis, and the next candidate family is proposed in the report.

## 4. Part 1 test (phase 2)

Experts entering the test. Gate E0 as defined in section 3 was written before the run and was run once on the validation forecasts of all candidates (section 8). E-B passes it (17 cells with r at most 1.2, 10 cells with g_in at least 1 %, 6 cells with g_oos at least 0.5 %, the last at the threshold exactly) but its mean out-of-sample gain over Bolt is -0.06 %. E-Q misses condition (a) by two cells (13 of 25 cells with r at most 1.2, against 15 required) and passes conditions (b) and (c) clearly (20 cells with g_in at least 1 %, 17 cells with g_oos at least 0.5 %, mean g_oos +2.13 %); its family-matched null expert E-Q0 gives mean g_oos +0.11 %. The decision, taken after seeing these validation numbers and before any test read, is to test both experts, E-B because it passes the gate and E-Q as a secondary candidate that misses one condition of the gate narrowly. To keep that decision from inflating the evidence, the Holm family is the 50 comparisons of the two experts over the 25 cells, and a claim based on E-Q is described in the paper as a candidate that did not meet the pre-test gate. The coordinator may strike E-Q from this list before committing; the code then runs E-B alone with a Holm family of 25. E-R (5 cells with r at most 1.2) and E-X (reference) are not tested.

Variants for each tested expert E (code `round2/a5.py`, select stage on validation only, test stage after the commit):

- M2: f = Bolt + alpha (E - Bolt), alpha from the Bates-Granger closed form on the fit slice, snapped to the 0.05 grid.
- M3: f = Bolt + w1 (dec - Bolt) + w2 (E - Bolt), NNLS on the fit slice per decomposition seed, w at least 0, sum rescaled to at most 1.

The headline variant of each expert is the one with the lower pooled mean log RMSE ratio to Bolt on SEL (fit on CAL), frozen in `frozen_a5.json` with its sha256 before any test read. The expert test forecasts come from the saved models (`experts_test.py`, which asserts alignment with the stored test windows and the stored Bolt test forecast).

Comparison set per cell: Chronos-Bolt (primary), the shipped CHA-S (secondary), the best of the two experts chosen on CAL. Controls: (a) timing control, the expert forecast rows permuted in blocks of h windows inside the test stream with the same fitted weights, 50 draws; (b) family-matched null expert, the mixture with E-Q0 or E-B0 respectively with its weights fitted by the same rule on validation. The better control is the one with the lower cell RMSE.

Success per expert (all required):

1. at least 5 of 25 cells are better than Chronos-Bolt after Holm (family of 50, or 25 if E-Q is struck) with a relative RMSE gain of at least 0.5 % (minimum effect size);
2. no cell is significantly worse than Chronos-Bolt after Holm;
3. the pooled relative RMSE gain over Chronos-Bolt, the mean over the 25 cells, is at least 0.5 % and its cell-level bootstrap CI lower bound is above 0;
4. the pooled gain over the better of the two controls has a cell-level bootstrap CI above 0.

Per outcome. If E-Q succeeds and E-B does not, E-Q is reported as the headline with the disclosure above. If both succeed the one with the larger pooled gain over the better control is the headline and the other is reported. Success: the paper states the expert, the mixture and the per-cell gains (Holm, effect sizes, TOST for the cells claimed equivalent), the abstract and contributions are rewritten to the measured gain (and, for E-B, to the residual-correction framing), and thesis chapter 6 adds the expert to the method and the result tables. Failure with a passing gate: the negative test is written with its numbers, the validation gate numbers are reported next to it, and the cause is the validation-to-test shift measured by the cell-level gap between g_oos and the test gain. Neither tested expert succeeds although the gate numbers were favourable: the negative test is written and the cause is the validation-to-test shift. No expert passes the gate and none is added: as stated in section 3.

## 5. Part 2: drift-adaptive alpha-monitor

Setting as in round 1 (frozen file section 4.1): five traces at h = 6, five decomposition seeds, the v2 injection (ramp of +3 sigma over 100 windows, 50 % multiplicative noise) into the foundation forecast of the test stream, change points at 0.2, 0.45 and 0.7 of `[t_arr[0] + 20, n - 110]`, ten injection seeds, label delay h, persistence 3, causal EMA smoothing (lambda 9/11) of the alpha stream. New detectors (code `round2/a4.py`):

- D-A, rolling-recalibrated CUSUM. The smoothed alpha is standardised at every decision j against a trailing fault-free-assumed reference window of the stream itself, ref = the Wc decisions that end G decisions before j, z_j = (x_j - mean(ref)) / max(sd(ref), floor) with the floor fixed from the validation design stream (0.5 x the standard deviation of the smoothed validation alpha, at least 0.01). A one-sided CUSUM with allowance K runs on z; an alarm is S above the level. Grid: Wc in {30, 60, 120, 240}, G in {10, 30, 60}, K in {0.5, 1.0, 2.0}, configurations whose burn-in G + Wc exceeds half of the validation stream are dropped.
- D-B, two-sample change detector on the distribution of alpha. The two-sample Kolmogorov-Smirnov statistic between the last Wr decisions and a reference of Wc decisions that ends G decisions earlier; an alarm is a statistic above the level. Grid (Wr, Wc, G) in {(30, 60, 10), (30, 90, 30), (45, 90, 30), (60, 120, 30), (60, 240, 60)} with the same burn-in rule.
- Reference methods under the same validation budget: the round 1 detectors M0 (v2 D3), M1 (round 1 headline, block-bootstrap threshold) and M5b (ADWIN), with their frozen round 1 parameters (`frozen_a2.json`, sha256 checked).

Calibration (validation only, per trace and decomposition seed). The level of D-A and D-B is the smallest on its grid whose 90th percentile over B = 100 (D-A) or B = 60 (D-B) moving-block-bootstrap resamples of the smoothed validation alpha stream (block length 2 (Wc + G), at least 60, at most half of the stream) of the alarm-onset share is at most 5 %, the share being counted over the decisions after the burn-in. Selection among the feasible configurations: the highest detection rate on validation-only synthetic injections (2 placements x 5 validation injection seeds, disjoint from test, lag at most 150 decisions), ties to the lower onset share on the actual validation stream, then to the larger Wc. An infeasible detector on a trace is flagged and counts as failing the false-alarm condition there. The headline detector is the one of D-A and D-B with the higher mean validation detection (infeasible traces count as 0), ties to D-A. Selection is frozen in `frozen_a4.json` with its sha256 before the test read.

False-alarm accounting. The no-fault false-alarm rate of D-A and D-B is the alarm-onset share over the decisions after their burn-in; the round 1 detectors have no burn-in. Detections require a lag of at most 150 decisions after the first informative decision.

Controls and statistics. (i) Matched no-injection control: same change points on the clean stream; shift-specific detection = detection on injected runs minus the control rate, Newcombe 95 % interval. (ii) Random control with permuted injection times: the alarms of every injected run are scored against the change-point clock of the farthest other placement of the same trace, so a detector that alarms because of the shift at the true time is not credited at a wrong time; the Newcombe interval of the detection rate minus the permuted-clock rate must exclude 0, and a trace whose change points are less than 30 decisions apart is flagged as not clock-separable (short NAB streams). (iii) Exact McNemar per (trace, placement) of the headline against M0 and against M1, reported (Holm across the 5 x 3 comparisons).

Success (all required):

1. no-fault false-alarm rate at most 5 % on at least 4 of 5 traces on the test streams;
2. shift-specific detection against the matched no-injection control, Newcombe interval above 0, on at least 4 of 5 traces;
3. on the traces of condition 2, the detection rate exceeds the permuted-clock control with a Newcombe interval above 0 on at least min(4, number of traces of condition 2) traces.

Per outcome. All three: the monitor claim returns in the paper (abstract, contribution 5, monitor subsection and table) with the per-trace numbers of the headline detector, the thesis chapter 6 monitor section and the "exploratory" threat are rewritten, and the two monitor figures are regenerated. Condition 1 or 2 fails: the v2 reframing "exploratory" stays and the measured cause (burn-in, clean-stream natural change points that fire the control, short streams) is stated with the per-trace table; the failing variants are named.

## 6. Resources and conduct

CPU `nice -n 19 taskset -c 8-11`, `OMP_NUM_THREADS=2`; the expert training uses the GPU only for Chronos-Bolt inference (nvidia-smi below 10 GB: 0.3 GB used before the run); MemAvailable above 10 GB checked before each stage; 5-minute progress lines in `PROGRESS.md`. No git. Round 2 phase 2 (after "PREREG 2 COMMITTED"): `experts_test.py`, `a5.py` (select then test, for the experts of section 4), `a4.py` (select then test), then the paper and thesis updates in the same round.

Run commands for phase 2, in this order: `python3 a4.py --stage select` and `python3 a5.py --stage select --experts EB EQ` (validation only); `touch round2/PREREG_2_COMMITTED` (exists once the coordinator commits); `REPAIR_STAGE=test .chas-venv/bin/python round2/experts_test.py` (GPU), `REPAIR_STAGE=test python3 round2/a5.py --stage test`, `REPAIR_STAGE=test python3 round2/a4.py --stage test`.

## 7. Code manifest (end of phase 1, `experiments/repair_2026-10-12/paper_a/` and `round2/`)

| file | role | sha256 |
|---|---|---|
| `round2/experts.py` | candidate experts E-Q, E-R, E-B, E-X and null experts, validation stage, `Restricted` view, `predict_cell` (added after the validation run started; the training code is unchanged) | `12dd347abade46ed64113215791ab74dfc81f55f3877e253559cce2ee64a3ada` |
| `round2/experts_test.py` | phase 2 test forecasts of the saved models, with the dry-run reproduction check | `26df880c82a01900e30def7aac1c2ffc331dbdd31d07c7191500d7b475f1c0dd` |
| `round2/gate_e0.py` | Gate E0 (validation only) | `900075e0a7625984bc8284fccfb8a8b54651b13713b7380131326a1c012c6310` |
| `round2/a5.py` | expert mixture variants M2, M3, select and test stages | `85ad8669e6ea83d480319057b81e9e1d3c38a89439afcbfad6262581adef771c` |
| `round2/a4.py` | drift-adaptive monitor D-A, D-B, select and test stages | `91950a0220997773d6f443914400208b5cddfc41519f453632b3567d2ed3fdfc` |
| `rcommon.py`, `stats_tools.py`, `a2.py`, `gate0.py` (round 1, unchanged) | guard, statistics, round 1 streams and evaluation | `747cddd2...`, `3e8b97b1...`, `d915197b...`, `7888876e...` (full values in `RUN_NOTES.md` and section 8 of the frozen file) |

Saved expert models: 25 files `round2/models/<dataset>__h<h>.joblib` (174 MB), sha256 of the sorted per-file sha256 list `c92bc94bab1076678bb2df4f4e4a66fb2365801e47d2595149f74e5b0a804901`. Validation forecasts: 25 files `round2/val_experts/*.npz`; for every cell the logs report `max_dy_vs_stored = 0.0` and `max_dbolt_vs_stored = 0.0` against the stored round 1 validation forecasts (`round2/log_experts_val.txt`, `log_experts_val2.txt`).

Dry runs, all on 2026-10-12, each printed the guard self-test result and `test files read: 0` or `test windows accessed: 0`: `experts.py --dry-run` (one cell, quarter of the training windows; accesses recorded: train and val only), `gate_e0.py --dry-run`, `a4.py --stage select --dry-run` (full select code on one trace, then the full test-stage code in pseudo mode with the validation stream as the pseudo test stream), `a5.py --stage select --dry-run --experts EB EQ` (select on 3 cells, then the full test-stage code in pseudo mode on the SEL slice), `experts_test.py --dry-run` (reloads the saved models of abilene h6, predicts its validation windows through the same `predict_cell` the test stage uses, maximum absolute difference to the stored validation forecasts 1.2e-07, no test window accessed).

## 8. Gate E0 result (validation only, appended after the run of `round2/gate_e0.py`; definitions of section 3 unchanged)

Cells with each condition, per expert, 25 cells (outputs `round2/gate_e0_cells.csv`, `gate_e0_summary.json`). Columns: cells with r at most 1.2 (a, needs 15), cells with g_in at least 1 % (b, needs 8), cells with g_oos at least 0.5 % (c, needs 6), gate verdict, median r, median rho, cells with r at most 1.2 and rho at most 0.8, mean g_in in percent, mean g_oos in percent, cells with g_3 at least 0.5 %, mean g_3 in percent, cells with xfit at least 0.5 %.

| expert | a | b | c | gate | median r | median rho | r<=1.2 and rho<=0.8 | mean g_in | mean g_oos | g_3 cells | mean g_3 | xfit cells |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EQ | 13 | 20 | 17 | fail | 1.175 | 0.826 | 2 | 2.49 | 2.13 | 17 | 2.27 | 1 |
| ER | 5 | 11 | 8 | fail | 1.612 | 0.693 | 0 | 0.81 | 0.46 | 10 | 0.60 | 0 |
| EB | 17 | 10 | 6 | pass | 1.063 | 0.955 | 0 | 1.15 | -0.06 | 7 | 0.23 | 0 |
| EX | 10 | 12 | 6 | fail | 1.234 | 0.788 | 3 | 2.12 | -4.87 | 6 | -4.71 | 6 |
| EQ0 | 0 | 1 | 4 | fail | 2.283 | 0.626 | 0 | 0.33 | 0.11 | 8 | 0.29 | 0 |
| EB0 | 19 | 2 | 2 | fail | 1.058 | 0.994 | 0 | 0.25 | -0.08 | 7 | 0.14 | 0 |

Reading. E-Q is a strong complementary expert on validation: its error correlation with Bolt is high (median 0.83) but its accuracy is close (median MSE ratio 1.18), which gives a mean out-of-sample mixture gain of 2.13 % over Bolt on SEL (17 of 25 cells at least 0.5 %, maximum 7.7 %), against 0.11 % for its null expert, and the gain survives in the three-expert form (mean 2.27 %). It misses the pre-stated condition (a) by two cells (13 of 25). E-B passes the gate with the minimum on (c) and has a negative mean out-of-sample gain (-0.06 %); its in-sample ceiling (mean 1.15 %, 10 cells at least 1 %) is about that of its null expert plus 0.9 points, so E-B is a Bolt copy plus a small correction. E-R is too inaccurate (5 cells with r at most 1.2). E-X, the reference, fails the gate and loses 4.9 % on SEL on average. The time-varying headroom xfit stays at about -2 % on average for every expert (E-X is the only one with 6 cells at least 0.5 %), so no time-varying variant is added to the test.

Decision for the test (section 4): E-B (gate pass) and E-Q (secondary candidate, misses (a) by two cells) are tested, Holm family of 50.

## 9. Part 2 information check on validation (seed 42, no test read)

For each trace the number of D-A configurations (burn-in at most half of the validation stream) whose block-bootstrap null level is feasible, and the number of D-B configurations feasible: CESNET-TS24 12 and 2, Abilene 8 and 1, GEANT 7 and 0, NAB AWS-CPU 8 and 0, NAB Twitter 6 and 0. D-A has a feasible configuration on all five traces, D-B only on the two longest traces, so D-B is expected to be the weaker challenger and D-A the headline. Validation-only injections are detected at rate 1.0 by the best D-A configuration on every trace, so the selection among feasible configurations falls through to the tie-breaks of section 5.
