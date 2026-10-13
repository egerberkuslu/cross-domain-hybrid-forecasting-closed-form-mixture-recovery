# REPAIR PRE-REGISTRATION 3, Paper A: confirmation of the quantile expert E-Q on fresh series, and a reference-free monitor

Written 2026-10-12 (round 3, phase 1). `REPAIR_PREREG.md` (round 1) and `REPAIR_PREREG_2.md` (round 2) are frozen and untouched. This file is FROZEN when the coordinator replies "PREREG 3 COMMITTED"; until then no fresh-series accuracy of any kind has been computed (not on the validation windows either), and no test window of the pool has been read by any model code. Folders: `experiments/repair_2026-10-12/round3/` (recipe, stages, evaluation, monitor, power calculation) and `experiments/repair_2026-10-12/fresh_cells/` (the reusable builder, shared with the F+G worker).

## 0. Question, and what changes against round 2

Round 2 (test split of the five paper series) gave the quantile expert E-Q a pooled RMSE gain over Chronos-Bolt of +3.03 % (95 % cell-level interval 1.65 to 4.53 %, dataset-clustered interval -0.27 to 6.46 %), no cell significantly worse, and three Holm-significant cells (GEANT h = 1, 3, 6), against five required. The pre-registered rule was not met and the pooled gain stood as an unconfirmed observation. Round 3 asks whether that observation holds on series that were never used in any paper, with a rule fixed from the round 2 effect sizes and a power calculation.

Changes against round 2, all stated here before any fresh-series result exists. (1) The mixture is the two-expert mixture M2, Chronos-Bolt plus E-Q, without the decomposition path: the decomposition path is a per-cell, per-seed LSTM model that does not exist for fresh series, and in round 2 the validation-selected variants M2 and M3 differed by 0.001 in the pooled SEL log ratio (-0.0219 against -0.0233). The claim tested is therefore "E-Q complements Chronos-Bolt", not "E-Q complements CHA-S". (2) E-Q is refitted on train plus validation windows for the test forecasts (the coordinator's request), with the mixture weight taken from validation forecasts of the train-only model. (3) The success rule is stated for N = 100 cells with a series-clustered interval.

## 1. Frozen E-Q recipe (hashed)

`round3/eq_recipe.py` and `round3/RECIPE_FROZEN.json` define the recipe; the features and hyper-parameters are those of `paper_a/round2/experts.py` (sha256 `12dd347abade46ed64113215791ab74dfc81f55f3877e253559cce2ee64a3ada`, unchanged since the round 2 models were hashed).

- Features (lookback 168 hourly values, per (window, step k) row): the last value, the lags 2, 3, 4, 6, 12, 24, 48, 72 and 168 as differences to the last value, the means of the last 6, 24 and 168 values minus the last value, the standard deviations of the last 24 and 168 values, the OLS slope of the last 24 values, sine and cosine of the target hour and weekday, and k. Target y_k minus the last value.
- Model: scikit-learn 1.9.0 `HistGradientBoostingRegressor`, quantile loss at 0.1, 0.5 and 0.9, 150 iterations, learning rate 0.06, depth 4, minimum leaf 40, L2 1.0, no early stopping, `random_state` 0. Point forecast: the median. Null expert E-Q0: same model at the median on feature rows shuffled across the training windows (permutation seed 0).
- Bolt: Chronos-Bolt-small zero-shot, HF snapshot `772f3d25d38aec6d914c8949dab4462e2d46f5d8`, weights sha256 `06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd`, computed in the shipped environment (`.chas-venv`, torch 2.10 CPU float32, chronos-forecasting 2.2.2, batch 32) for the validation and test windows of every cell. (In round 2 the stored test Bolt forecast came from an earlier bf16 run; in round 3 every arm uses one freshly computed Bolt forecast, so no stored forecast is involved.)
- Mixture M2: f = Bolt + alpha (E-Q - Bolt), alpha the clipped Bates-Granger closed form on all validation windows of the cell (E-Q trained on train only), snapped to the 0.05 grid; the null mixture uses E-Q0 with its own alpha by the same rule.
- Stage A (validation forecasts, train-only models) is run after the commit and writes `r3_val/` and `r3_models/`; stage B (refit on train plus validation, test forecasts) writes `r3_test/` and needs `PREREG_3_COMMITTED` and `REPAIR_STAGE=test`. The weights are frozen in `frozen_r3.json` (sha256 in `frozen_r3.sha256`, to be recorded in `RUN_NOTES.md`) before stage B.

## 2. The fresh pool

Reusable module `fresh_cells/fresh_cells.py` (README in its docstring). Sources, all at one-hour resolution with the shipped aggregation: CESNET-TimeSeries24 institutions (283 series, provider's hourly aggregation, `n_bytes`), Abilene OD pairs (144 series from the 12 x 12 traffic matrix, 5-minute sums to hourly), GEANT OD pairs (524 series from the 23 x 23 matrix, 15-minute sums to hourly). The paper series are the aggregates of these components, so the fresh series are different forecasting problems from the same networks and the same calendar periods, not independent of the aggregates; this is a limit of what "fresh" means here and is stated in the paper.

Selection rule, data-only and fixed before any forecast of a fresh series existed. A series is admissible iff its share of missing hours is at most the aggregate's share plus 0.5 points, at most 5 % of its hours are exactly zero, and its median is positive (after the shipped outlier flag at 100 x median). From the admissible series of a source, ordered by numeric id (OD index 12 src + dst for Abilene, lexicographic (src, dst) for GEANT), every k-th is taken with k = floor(n_admissible / N), starting from the first. Admissible: CESNET 200 of 283, Abilene 128 of 144, GEANT 329 of 524. N = 8, 6 and 6.

Selected (manifest `fresh_cells/pool_manifest.json`, sha256 `f938864f0c688c16803aa87bda69f4f3a22eddb5a08e41765be811fc29aaa03b`, with the sha256 of every series file): CESNET institutions 0, 33, 64, 97, 125, 164, 193, 225; Abilene OD indices 1, 27, 50, 73, 95, 119; GEANT OD pairs 1_1, 4_9, 8_4, 12_16, 16_5, 19_21. That is 20 series and 100 cells (horizons 1, 3, 6, 12, 24), with the same splits as the paper (0.70, 0.15, 0.15, chronological, scaler on train only, window 168; window counts per cell in the manifest).

Disjointness from the F+G scale test. `fresh_cells/reserved_series.json` (sha256 `b4c9cc37534c8d2294279e504b08e64ba79a5431a9c59f9bbe0732806acf2128`) reserves these 20 series; the F+G partition `fresh_cells/partition_F.json` (sha256 `f39e8c16789e733bcb85dc1c3b1e16586ceed6f936529f050a2a6f8e80646df6`) registers 100 series per source plus a 5-series pilot and states that it excludes the Paper A manifest. Checked: the intersection of the F+G registered set and pilot with the 20 series is empty for all three sources. Nothing else may be drawn by this registration.

## 3. Power calculation (round 2 effect sizes, no fresh series involved)

Script `round3/power_calc.py`, output `power_calc.json`. Template: the 25 E-Q cells of round 2 (rel RMSE change and its bootstrap interval, se = width / 3.92). A fresh series takes the round 2 cells of its source at the same horizon; its true effect is shrink x round 2 effect x u, u lognormal(0, 0.5) per series; the estimate is normal around the true effect with the round 2 se; two-sided cell p values, Holm over 100 cells, win = Holm p below 0.05 and gain at least 0.5 %. 1,500 simulated pools per setting.

Under no effect, P(wins >= 5) = P(wins >= 8) = 0 (Holm controls the family error). Mean number of wins: 20 at the full round 2 effect, 12.5 at 75 % of it, 5.3 at half of it, 0.7 at a quarter. Power of the complete rule (wins at least K, no significant loss, pooled gain at least 0.5 %, cell-level and series-clustered lower bounds above 0):

| K | full effect | 75 % | 50 % | 25 % |
|---|---|---|---|---|
| 5 | 1.00 | 0.97 | 0.54 | 0.01 |
| 8 | 0.99 | 0.86 | 0.23 | 0.00 |
| 10 | 0.98 | 0.72 | 0.11 | 0.00 |

Choice, made before any fresh result: K = 8 of 100 cells. It has power 0.86 if three quarters of the round 2 effect is real (round 2 effects of the largest cells are subject to winner's-curse inflation, which is why 75 % and not 100 % is the planning value) and a false-confirmation probability of essentially 0. If the true effect is half of the round 2 estimate the rule has power 0.23, i.e. this design cannot confirm an effect that small, and an unconfirmed outcome would not refute it; the paper will say so.

## 4. The accuracy rule (part 1)

Comparison per cell: M2 against Chronos-Bolt, seed-free (E-Q and Bolt are deterministic), paired two-sided sign-flip block permutation on the per-window squared-error difference (block h, B = 20,000), Holm over all 100 cells; relative RMSE change with a moving-block bootstrap interval; TOST (5 % margin) and bit-identical cells reported as in round 2. Reference rows (reported, not in the rule): E-Q alone, the better single model on validation, and the null mixture.

Random controls (per cell): (a) E-Q forecast rows permuted in blocks of h windows in the test stream with the same alpha, 50 draws; (b) the null expert E-Q0 mixed by its own validation alpha. The better control is the one with the lower cell RMSE.

Success (all required):

1. at least 8 of the 100 cells are better than Chronos-Bolt after Holm with a relative RMSE gain of at least 0.5 %;
2. no cell is significantly worse after Holm;
3. the pooled gain over Chronos-Bolt (mean over the 100 cells) is at least 0.5 %, and its lower 95 % bound is above 0 under the cell-level bootstrap AND under the series-clustered two-stage bootstrap (series resampled, then horizons within a series, B = 10,000);
4. the pooled gain over the better of the two controls has a lower 95 % bound above 0 under both bootstraps.

Per-source results (CESNET, Abilene, GEANT) and the per-cell table are reported whatever the outcome. Per outcome. Success: the paper states that a quantile expert on lag and calendar features, mixed with Chronos-Bolt, improves on the foundation model on fresh series of the same networks, with the cell counts, the pooled gain and both intervals, discloses that the confirmation used components of the networks of the discovery series, and rewrites abstract, contributions, highlights and cover letter (and thesis chapter 6 and its statement of the CHA-S contribution) consistently; the decomposition path is then described as one of two possible partners, with the round 2 three-expert result next to it. Failure with a positive pooled gain: reported as an unconfirmed observation with the power statement of section 3. Failure with a non-positive pooled gain: the round 2 gain is reported as not reproduced on fresh series and E-Q is dropped from the paper's contributions.

## 5. The monitor (part 2): reference-free detector on the expert error ratio

Streams. For a trace at h = 6, horizon-averaged forecasts of Bolt and E-Q, and the same stream construction as round 1 (`a2.streams` with the E-Q forecast in the position of the decomposition path): q_t = log(MSE_Bolt / MSE_E-Q) over the last 60 labelled windows, label delay h, one q value per decision.

Traces: the fresh series with a train-to-validation level shift of at least 0.3 train standard deviations (data-only rule on the train and validation values, no test value and no forecast): cesnet_inst193, cesnet_inst225, abilene_od1, abilene_od27, abilene_od50, abilene_od95, abilene_od119, geant_od1_1, geant_od4_9, geant_od8_4, geant_od16_5, geant_od19_21, 12 traces; their test streams have 1,007 (CESNET), 699 (Abilene) and 428 (GEANT) decisions, longer than the NAB streams of rounds 1 and 2. The injection is the v2 injection (ramp +3 sigma over 100 windows, 50 % multiplicative noise) into the Bolt forecast of the test stream, change points at 0.2, 0.45 and 0.7 of `[t_arr[0] + 20, n - 110]`, 10 injection seeds (30 injected runs and 3 matched no-injection control runs per trace and detector).

Detectors (code `round3/r3_monitor.py`), all with the same validation budget:

- RF, reference-free CUSUM. z_t = (q_t - m_t) / max(sd_t, floor); (m_t, v_t) are exponentially weighted (decay 0.99, warm start on the first 20 q values) from q delayed by G decisions, G in {30, 60}; floor = max(0.5 sd of the validation q stream, 0.05); one-sided CUSUM with allowance K in {0.5, 1, 2}; alarm when above the level; persistence 3. No validation mean or threshold-on-the-level of q is used: only the CUSUM level and the floor come from validation. The level is the smallest on the grid 1..30 (step 0.5) whose 90th percentile over 60 moving-block-bootstrap resamples of the validation q stream (block 2 (G + 60), at least 60, at most half of the stream, the same rolling machinery applied to every resample) of the alarm-onset share is at most 5 %. Burn-in: alarms are suppressed for the first 100 decisions, and those decisions COUNT in every false-alarm denominator, as in the calibration. Selection among feasible (G, K): the highest detection on validation-only injections (2 placements x 5 seeds, disjoint from test, lag at most 150), ties to the lower validation onset share, then to the larger G.
- ST, static CUSUM on q standardised with the validation mean and standard deviation, level by the same bootstrap rule, no burn-in (the round 1 M2 construction).
- AD, ADWIN on the delayed Bolt squared-error stream, delta selected by the round 1 rule (largest delta with the bootstrap 90th percentile of the onset share at most 5 %).

Controls and statistics as in round 2: matched no-injection control, shift-specific detection with a Newcombe 95 % interval, random control with permuted injection times (the alarms of each injected run scored against the clock of the farthest other placement of the same trace; traces whose change points are less than 30 decisions apart are flagged), lag cap 150. False-alarm rate = alarm onsets over ALL decisions of the test stream.

Success for RF (all required, with n = 12 traces and 10 required, i.e. 80 %):

1. no-fault false-alarm rate at most 5 % on at least 10 of the 12 traces;
2. shift-specific detection (Newcombe interval above 0) on at least 10 of the 12 traces;
3. detection above the permuted-clock control (Newcombe interval above 0) on at least min(10, number of traces of condition 2) traces.

No power calculation is attached to the monitor rule (there is no round 2 effect size for this detector); a failure on fewer than 10 traces is reported with the per-trace table and the cause by trace (clean-stream natural changes, shift-detection lag, calibratability). Outcome statements as in section 5.4 of round 2.

## 6. Code manifest, dry runs and conduct

| file | role | sha256 |
|---|---|---|
| `round3/eq_recipe.py` | frozen recipe, `recipe_manifest()` | `d45d3ae7ea731d580d3199ce27f5c6f4cd48cee5bc52e2857fa4597bc9cfcf24` |
| `round3/RECIPE_FROZEN.json` | recipe manifest with all hashes | `5a96181ebe5633f76daf34646a301c6d5453e2b60e8187b99a57f258c590cde8` |
| `round3/r3_stages.py` | stage A (train-only, validation forecasts) and stage B (refit, test forecasts) | `6b3f2e79169e729e73bd17da3f7012a7b2133e5f15623756b957b2c1cb327521` |
| `round3/r3_eval.py` | alpha selection, test statistics, success rule | `e8cb9393f24b3f2ea4efdda71c49fb9a24317b4abc7678cd8b252042f412ba8d` |
| `round3/r3_monitor.py` | RF, ST, AD detectors, select and test stages | `f99a3f690eef95a95bf5c9f7032299f1d95206273e2404b665a7cb884c8b1e15` |
| `round3/power_calc.py`, `power_calc.json` | power calculation | `45350fd6...`, `ffa47f66...` |
| `fresh_cells/fresh_cells.py`, `build_pool.py` | reusable builder | `f4f5a4da72b34d3938865a36f8b66d970d270f8f76387a20d1b6c715f3a50660`, `ab4519af...` |
| round 1 and 2 modules reused unchanged | `rcommon.py`, `stats_tools.py`, `a2.py`, `round2/experts.py` | `747cddd2...`, `3e8b97b1...`, `d915197b...`, `12dd347a...` |

Dry runs, 2026-10-12, `round3/dryrun_log.txt`: `r3_stages.py --stage A --dry-run` (the Abilene aggregate of the paper, a non-fresh cell, quarter of the training windows: RMSE bolt 0.1973, E-Q 0.1984, null 0.3798, i.e. the round 2 values; a pseudo stage B; a fresh-cell structure check with no model fit and no accuracy; zero test windows accessed through the `Restricted` view), `r3_eval.py --stage select --dry-run` and `r3_monitor.py --stage select --dry-run` (full select and test code on non-fresh pseudo data), each printing `dry-run ok; test windows accessed: 0`. Accuracy is computed only on non-fresh cells in the dry runs. `eq_recipe.py` was run once to write `RECIPE_FROZEN.json`; it fits nothing.

Guards. The `Restricted` view raises on a request for test windows; stage B and the test stages additionally need `PREREG_3_COMMITTED` and `REPAIR_STAGE=test`; the frozen JSON files are hashed before the test read and re-checked.

Resources (est.): stage A about 100 cells x 40 to 60 s (quantile models on up to 4,500 x 24 rows, Bolt on about 1,000 validation windows on CPU), stage B about 100 cells x 60 to 90 s, the monitor select stage about 20 minutes, all under `nice -n 19 taskset -c 8-11`, at most two workers, `OMP_NUM_THREADS=2`; MemAvailable above 10 GB checked per stage; 5-minute progress lines in `paper_a/PROGRESS.md`. No git.

Phase 2 order after the commit: `python3 r3_stages.py --stage A` (validation, guard on) then `python3 r3_eval.py --stage select` and `python3 r3_monitor.py --stage select`, record the hashes of `frozen_r3.json` and `frozen_r3_monitor.json` and of the 100 stage-A model files in `RUN_NOTES.md`, then `touch PREREG_3_COMMITTED` (exists once committed), `REPAIR_STAGE=test python3 r3_stages.py --stage B`, `r3_eval.py --stage test`, `r3_monitor.py --stage test`, then paper and thesis updates in the same round.
