# REPAIR PRE-REGISTRATION, Paper A (CHA-S) items A-A1, A-A2, A-A3

Written 2026-10-12 (phase 1). Source of the items: `doktora-tez-denemeler/tez_turkce/docs/plan/ONARIM_ENVANTERI.md` entries A-A1, A-A2, A-A3 and its section 1 (common protocol). Paper: `submission/paper_a/peerj_v2` (never `../peerj`). Code and outputs of this repair: `experiments/repair_2026-10-12/paper_a/`.

Status of this file: FROZEN once the coordinator replies "PREREG COMMITTED". Everything in sections 1 to 7 is fixed before any test split is read. Section 9 (Gate 0 result) is appended after a validation-only run whose definition is in section 3.1 and whose code hash is recorded below. Anything decided after test data are seen is labelled exploratory and never enters a headline.

Hashes of the Gate 0 code at the time the Gate 0 run started (see section 8 for the full code manifest):

- `gate0.py` sha256 `7888876ea37e1a702e3c03f539faa9242dd5d27d00c8eb644dac1393309cd98f`
- `rcommon.py` sha256 `33b7a1ccf834662fcd5b28aeb416775207650e87f46b36677bd393786066c13c` (extended afterwards by the pseudo-mode and the guard self-test; Gate 0 was rerun with the final file and its outputs are byte-identical in content, see section 9)

## 0. What is repaired and why (one paragraph per item)

A-A1. CHA-S almost never beats Chronos-Bolt (0 better, 14 tied, 3 worse, 8 identical cells in the seed-paired block permutation). Hypotheses for the cause: a constant alpha cannot follow regime shifts, the decomposition path leaves little room (alpha in [0, 0.10] on all 25 network cells), and the weight ignores time-varying error correlation. The repair candidates are recency-weighted, shrunk, change-point-reset and horizon-wise closed forms. Because those candidates only matter if the weight has exploitable structure, a validation-only headroom Gate 0 runs first.

A-A2. The alpha-monitor keeps the no-fault false-alarm rate (FA) at or below 5 % on 3 of 5 traces and separates the injected shift from the no-injection control on 2 of 5 (v2 results). The repair candidates are a block-bootstrap (conformal-type) threshold rule, an error-ratio statistic, a fusion rule, a validation-chosen persistence rule and the two standard reference detectors Page-Hinkley and ADWIN as fair baselines.

A-A3. The learned weight alpha(x) (CHA-L, CHA-LR) gives median +0.04 % and +1.17 % relative RMSE against the scalar weight. The repair candidates are ridge regression to the scalar, depth-1 boosted stumps, partial pooling across the 25 cells and distillation from the rolling closed-form target.

## 1. Data roles (disjoint) and what exists on disk

The persisted artefacts are forecasts of the two experts (decomposition path `dec`, five seeds 42, 123, 2024, 31337, 7; foundation path `glob`, Chronos-Bolt-small, deterministic, stored for seed 42), the true values `y`, all in the scaled domain, windows with stride 1 (verified: `y[t+1, :-1] == y[t, 1:]`), `h` steps per window. No train-split forecasts exist and none are regenerated.

| role | data | use |
|---|---|---|
| CALIBRATION (CAL) | first 60 % of the validation windows of the cell (chronological) | fit every parameter that is fitted (alpha, ridge coefficients, stump ensembles, per-step weights, thresholds) |
| SELECTION (SEL) | last 40 % of the validation windows, after a purge gap of h windows | choose variant hyper-parameters, one pooled setting over all 25 cells x 5 seeds; this is the inner rolling origin (history = CAL, then growing) |
| FINAL FIT | all validation windows | the refit that is carried to the test stage with the hyper-parameters frozen from SEL |
| TEST | the existing test split, 199 to 1007 windows per cell, untouched | read once by the `--stage test` code path, after `PREREG_COMMITTED` exists |

Online variants (recency-weighted, change-point-reset) update on test windows whose targets are fully observed at the forecast origin (label delay h: window s is usable at origin t iff s <= t - h) plus the whole validation stream as history. That is causal use of the observed stream and not a fit on test data; the hyper-parameters (window W, shrinkage, thresholds) are frozen from SEL before the first test window is read.

A-A2 uses the five network traces at h = 6 (the horizon of the v2 monitor). Thresholds and every selected parameter come from the no-fault validation stream of the same trace and decomposition seed plus validation-only synthetic injections whose injection seeds are disjoint from the test injection seeds (validation seeds use `5_000_017 * (pi + 1) + 7919 * r + s`, test seeds use the v2 formula `1_000_003 * (pi + 1) + 7919 * r + s`).

Disclosure of test-file opens before this registration. While exploring the data layout I opened the stored test forecast files (CHA-S, Bolt, decomposition-only, seed 42, all 25 cells) once to read the number of windows per cell, and the test `y` of Abilene and NAB Twitter at h = 6 to check stride-1 contiguity. No error, RMSE or weight was computed from them; no design decision above depends on a test value. Separately, Gate 0c and the A-A1b motivation use only tables that are already published in the paper (`closed_form_test_accuracy.csv`, `floor_bound_test.csv`, `outputs/eval_v3/tables/metrics_aggregated.csv`).

Enforcement. `rcommon.py` installs a guard (`install_guard`) that wraps `numpy.load`, `builtins.open`, `Path.open` and `Path.read_text` and raises `TestAccessError` for any path under `outputs/predictions` or `outputs/checkpoints/../predictions` unless `allow_test()` was called. `allow_test()` refuses unless the file `experiments/repair_2026-10-12/paper_a/PREREG_COMMITTED` exists and `REPAIR_STAGE=test` is set. Every fitting and selection script calls `install_guard()` first and never calls `allow_test()`. Each script has `--dry-run`, which runs the whole select stage on a few cells under the guard and a pseudo-test pass on the SEL slice with the real evaluator, asserts that the guard log contains no test path, and prints the number of test files read (must be 0).

## 2. Common statistics and decision rules

1. Primary accuracy metric: RMSE in the scaled domain, pooled over the h steps. Secondary: MAE, sMAPE and MASE on the native scale (MASE with the lag-1 naive in-stream denominator). Per cell the five decomposition seeds are averaged to window-level paired differences, never pooled as independent replicates.
2. Paired test: seed-paired two-sided sign-flip block permutation on the window-level squared-error difference (block length h, B = 20,000), as in `fix_2026-10-11/permutation_two_sided.py`. Holm across the 25 cells inside each family stated below. Effect size: relative RMSE difference with a moving-block bootstrap 95 % CI (block h, B = 5,000).
3. Equivalence: TOST on RMSE_variant - RMSE_Bolt with margin 5 % of RMSE_Bolt, 90 % moving-block-bootstrap CI (block h, B = 5,000), as in `dm_tost_identity.py`. Forecast pairs that are bit-identical (max abs difference <= 1e-6 std(y)) are reported as "identical" and are never counted as equivalence evidence. Non-significance is never written as parity.
4. Pooled gain over a control: per cell g_c = 1 - RMSE_variant / RMSE_(better control), where the better control is the control with the lower cell RMSE averaged over its 50 draws. CI = percentile bootstrap of the mean of g_c over cells (B = 10,000, cells resampled with replacement). Robustness: the same bootstrap resampling datasets first and horizons within (two-stage, 5 clusters), reported next to it. The headline condition uses the cell-level bootstrap named in the inventory; if the two-stage CI includes 0 the text says "pooled over cells" and states the two-stage interval.
5. Outcome rule: a success rule met means the repaired method and its numbers go into paper and thesis. Not met means the negative result is written with the numbers, the failed variants named and the failure explained by a measured quantity.

## 3. A-A1 repair of the accuracy claim

### 3.1 Gate 0 (headroom), validation only, run in phase 1

Question: is there any exploitable structure in a time-varying or step-varying mixture weight at all? Definition (code `gate0.py`, frozen by the hashes above). Per cell and decomposition seed, on VALIDATION windows only:

- deployed constant weight: the grid weight CHA-S ships (`closed_form_cells.csv`, alpha_grid), RMSE_const = RMSE of that constant on all validation windows;
- in-sample block oracle (upper bound, never a decision quantity because it is optimistic by construction): non-overlapping blocks of L windows, the exact clipped Bates-Granger weight inside each block, RMSE pooled over blocks;
- cross-fitted block headroom (the decision quantity): each block is scored with the weight estimated WITHOUT that block, from the two adjacent blocks with h windows purged next to the block ("nb", primary), and from the previous block only ("prev", causal, reported); headroom = 1 - RMSE_xfit / RMSE_const; mean over the 5 seeds; L = 48 primary (L = 24 and 96 reported as sensitivity);
- drift transfer loss (reported, not a rule): weight fitted on one validation half scored on the other half against that half's own optimum, averaged over both directions;
- Gate 0b (third expert, reported and used for V5 only): NNLS stacking of [Bolt, decomposition, persistence of the last observed value] fitted on CAL and scored on SEL against the CAL-fitted scalar CHA-S weight, as relative RMSE gain, mean over seeds.

Rule, fixed in advance and taken from the inventory with the noise correction replaced by cross-fitting (an in-sample oracle always shows headroom from fitting noise, so it cannot gate anything): a cell "has headroom" iff the primary cross-fitted headroom is >= 0.5 %. Gate 0a passes (variants V1 to V4 are run) iff at least 6 of the 25 cells have headroom; equivalently V1 to V4 are declared infeasible if the headroom is below 0.5 % in at least 20 of 25 cells. Gate 0b passes (V5 is run) iff at least 6 of 25 cells show a 3-expert gain >= 0.5 %.

Consequence if Gate 0a fails: V1 to V4 are not run on test for the accuracy claim (no test read for the accuracy variants). Their select stage on validation still runs, because A-A3 L4 needs the V1 window W* and because A-A1b below uses the same variants. The paper keeps the sentence "equivalent to the foundation expert and readable" and states the measured headroom as the reason, with the cross-fitted and in-sample numbers; the thesis does the same. The negative result is then written together with whatever V5 shows. The v2 upload then goes as it is for A-A1.

### 3.2 Variants (run only if Gate 0a passes; V5 only if Gate 0b passes)

All variants mix the same two stored forecasts, `f = glob + alpha (dec - glob)`, alpha clipped to [0, 1]. No expert is retrained.

- V0 (current CHA-S): constant alpha on the 0.05 grid fitted on the FINAL FIT slice (on SEL scoring: fitted on CAL).
- V1 recency-weighted closed form: alpha_t = clip((-sum sed + kappa sdd_bar alpha_0) / (sum sdd + kappa sdd_bar), 0, 1) over the last W labelled windows (label delay h), alpha_0 = deployed constant, sdd_bar the mean of sdd in the fit window. Grid W in {24, 48, 96, 192, 384} x kappa in {0, 12} (10 configs).
- V2 shrinkage: alpha = lambda alpha_dep, lambda in {0, 0.25, 0.5, 0.75, 1.0, 1.25}.
- V3 change-point reset: V1 whose fit window restarts at the CUSUM alarm of the log-ratio statistic of the A-A2 error-ratio monitor (threshold from the A-A2 block-bootstrap rule at null level q in {0.90, 0.95, 0.99}); fewer than 12 labelled post-reset windows fall back to V2 with lambda = 1. Grid W in the best three V1 windows x q (9 configs).
- V4 per-horizon-step weight: alpha_k = alpha_dep + s (alpha_k^BG - alpha_dep), alpha_k^BG the closed form per step k on the fit slice, s in {0.25, 0.5, 1.0}. For h = 1 it equals V0 (reported as identical, not as tie).
- V5 (conditional on Gate 0b) three-expert NNLS stacking on [Bolt, decomposition, persistence], weights >= 0 with sum rescaled to <= 1, fitted on the FINAL FIT slice; ridge-free.

Reference baselines on the same test cells: Chronos-Bolt; current CHA-S (stored forecast, reproduced); best-of-two chosen on validation (the expert with the lower FINAL FIT RMSE); fixed alpha = 0.5; XGBoost lag features (stored forecasts `*_xgboost_*`, reference only, not re-tuned, because it is a different model family and it is not part of the mixture claim). Fairness: every variant gets the same CAL/SEL budget and the grids above; no variant sees test windows except through the causal label-delayed stream.

Selection (SEL only, pooled): for every config the score is the mean over the 125 (cell, seed) pairs of log(RMSE_SEL(config) / RMSE_SEL(V0 fitted on CAL)). Per variant the config with the lowest pooled score is kept (ties to the larger kappa and the longer W, i.e. the simpler one). The headline variant is the variant with the lowest pooled score among V1 to V4 (V5 only if it passed Gate 0b); if no variant has a negative pooled score the headline variant is still the lowest one and the outcome rule below applies to it. Selection is written to `frozen_a1.json` with its sha256 before any test read.

### 3.3 Random controls

(a) Timing control: the alpha series of the variant permuted in contiguous blocks of h windows within the test stream (marginal kept, timing destroyed); 50 draws. (b) Wrong-cell control: alpha_t drawn i.i.d. from the pooled SEL alpha distribution of the same variant in another random cell of the same seed; 50 draws. For V4 the controls shuffle the step index of alpha_k within the window. Both controls use the same Bolt and decomposition forecasts, only alpha differs.

### 3.4 Test, statistics, success rule

Test comparison per cell, headline variant against Chronos-Bolt: paired permutation family of 25 (Holm). Secondary families: headline variant against current CHA-S (25, Holm) and all non-headline variants against Chronos-Bolt (exploratory, Holm over 25 x number of variants).

Success (all three required):

1. the headline variant is significantly better than Chronos-Bolt after Holm in at least 5 of 25 cells;
2. it is significantly worse in 0 cells;
3. its pooled gain over the better of the two controls has a cell-level bootstrap CI excluding 0 (section 2, item 4).

TOST is reported for the cells where equivalence is claimed; "identical" cells (forecasts bit-identical because alpha = 0) are reported separately.

### 3.5 What the paper and the thesis say per outcome

- Gate 0a fails: contribution 3 and 6 keep the wording "equivalent to the foundation expert with a readable decomposition path"; the abstract does not claim superiority; the discussion states the measured cross-fitted headroom (and the in-sample oracle as an upper bound) as the reason no time-varying or step-varying weight can help; thesis chapter 6 (`09_durust-olumsuz-sonuclar`, `08_degerlendirme`) says the same with the numbers. No test read for A-A1.
- Gate 0a passes and the success rule is met: the repaired weight rule becomes the CHA-S weight (named in the paper as the recency-weighted / shrunk / horizon-wise variant selected on validation), the abstract and contribution 3 report the number of cells with a significant gain and the pooled gain over the control, and the equivalence statements are rewritten per cell with TOST. The thesis chapter 6 method section and results are updated and the old constant-weight numbers move to a comparison table.
- Gate 0a passes and the rule is not met: the negative result is written (the variants named, the cells where each won or lost, the control gap); the paper keeps "equivalent and readable"; the thesis states that the measured headroom existed but the validation-selected rule did not turn it into held-out gain, with the control numbers.

### 3.6 A-A1b: do-no-harm (floor) repair, run only if the coordinator approves it with the commit

Why. Gate 0 closes the superiority route (section 9), but a second, different property of the weight is still repairable and is already measured in v2: the "floor" property, RMSE(CHA-S) <= min(RMSE(Bolt), RMSE(decomposition)), which holds out of sample in only 16 of 25 cells (excess up to 3.3 %, five Abilene cells where the test-oracle constant weight is 0, i.e. a validation-to-test shift that the validation stream cannot show; existing numbers, `floor_bound_test.csv`, `gate0c_cells.csv`). Online variants (V1, V3), shrinkage (V2) and per-step weights (V4) can act on that shift, which no validation-only headroom measure can see. The claim being repaired is "CHA-S never loses to its better endpoint", not "CHA-S beats Bolt".

Same variants, grids, controls and CAL/SEL data roles as section 3.2, with these differences. (1) Selection criterion (SEL only, pooled): the mean over 125 (cell, seed) pairs of the SEL excess max(0, RMSE_config / min(RMSE_Bolt, RMSE_dec) - 1); ties to the lower pooled log-ratio of section 3.2. Frozen in `frozen_a1b.json`. (2) Gate 0 does not gate A-A1b; its Gate-0c motivation is the 9 of 25 published floor failures. (3) Test endpoint per cell: floor holds iff RMSE_variant <= 1.001 x min(RMSE_Bolt, RMSE_dec) (a fixed 0.1 % tolerance, applied identically to V0 and to the controls). (4) Success (all required): the floor holds in at least 22 of 25 cells; it holds in more cells than V0 with a one-sided exact sign test over the cells where the two differ, p < 0.05; it holds in more cells than either control; the mean relative RMSE against Bolt over the 25 cells is not worse than that of V0. (5) Outcome: success means the paper states a do-no-harm guarantee for the selected weight rule with its measured out-of-sample count (and the 16/25 of v2 as the before value); failure means the paper keeps the v2 sentence that the floor bound holds in 16/25 cells and the thesis reports the variant counts and the controls.

## 4. A-A2 repair of the alpha-monitor

### 4.1 Setting (identical to v2 apart from the listed changes)

Traces CESNET-TS24, Abilene, GEANT, NAB AWS-CPU, NAB Twitter at h = 6, five decomposition seeds. Streams: rolling clipped Bates-Granger alpha `a_t` and rolling foundation RMSE `r_t` on the last W = 60 labelled windows (label delay h), horizon-averaged, as `monitor_far.alpha_and_rmse_stream`; causal EMA smoother with lambda = 9/11; persistence rule of 3 consecutive hits unless M4 overrides; shift injection exactly the v2 ramp (+3 sigma over 100 windows with 50 % multiplicative noise) into the foundation forecast of the test stream; the matched no-injection control stream has the same change points. Change points at 0.2, 0.45, 0.7 of the post-warm-up range `[t_arr[0] + 20, n - 110]` (the placement-robust set of `mon_placement_variant.py`), 10 injection seeds, 5 decomposition seeds: 150 injected runs and 15 control runs per trace and detector.

### 4.2 Detectors

- M0: current v2 D3 (absolute threshold on the smoothed alpha AND CUSUM of alpha, thresholds from the validation stream with the "smallest h with onset rate <= 5 %" rule). Reference.
- M1: same two conditions but thresholds from a block bootstrap of the validation stream: B = 300 moving-block resamples (block length 60, the width of the rolling window, so dependence created by the window is kept) of the validation `a_t` and `r_t` streams; CUSUM level h* = smallest value of the grid 1.0..30.0 step 0.5 such that the 90th percentile over resamples of the onset rate is <= 5 %; absolute threshold tau* = the 90th percentile over resamples of the 95th percentile of the smoothed alpha, floored at 0.05. (An a-priori rule with an upper confidence bound instead of the point-estimate rule.)
- M2: error-ratio statistic `q_t = log(rolling MSE(Bolt) / rolling MSE(decomposition))` on the same W and delay, CUSUM on q_t standardized on the validation stream, level by the M1 rule.
- M3: fusion, alarm only if the M1 alpha condition AND the M2 condition both hold on the same decision.
- M4: persistence length k chosen from {2, 3, 5, 8} jointly with the CUSUM level on validation: among the (k, h) pairs whose bootstrap q90 onset rate is <= 5 %, take the one with the highest detection rate on validation-only synthetic injections (2 placements x 5 validation-only injection seeds, disjoint from test), ties to the smaller k. Applied to M1.
- M5a: Page-Hinkley on the standardized Bolt squared-error stream (reference detector, one-sided increase; the Page-Hinkley statistic max(0, PH_{t-1} + z_t - delta) equals a one-sided CUSUM, allowance delta = 0.5 sd as for the alpha CUSUM), threshold lambda chosen by the same q90 null rule. The stream is e_t = (foundation error of the newest labelled window)^2, standardized with the validation mean and sd.
- M5b: ADWIN (variance-based bound with delta' = delta / ln(window), minimum sub-window 5, delta in {0.0005, 0.001, 0.002, 0.005, 0.01, 0.05}) on the same standardized Bolt error stream; every ADWIN change signal is one alarm onset (no persistence); the largest delta (most sensitive) whose bootstrap q90 onset share is <= 5 % over B = 100 block-bootstrap resamples (B = 300 for the CUSUM-type detectors); if no delta is feasible the smallest delta is used and the trace is flagged infeasible.

Selection of the headline alpha-based monitor among M1 to M4 (SEL role = validation-only injections, nothing from test): highest mean validation detection rate over the five traces subject to the validation null onset rate q90 <= 5 % on every trace; ties to the simpler (M1 < M2 < M3 < M4). Frozen in `frozen_a2.json` with its sha256 before the test read.

### 4.3 Metrics, controls, statistics, success rule

Metrics on TEST: no-fault FA (onset share on the clean stream, mean over the 5 seeds), detection rate and median lag over the 150 injected runs, pre-information alarm share, no-injection control detection rate (15 runs), shift-specific detection = detection rate minus control rate with a Newcombe 95 % interval. Random control: the block-shuffled control, i.e. every injected test stream with its decisions permuted in contiguous blocks of 60 under a common permutation of the four stream arrays (marginals and block structure kept, the timing of the shift against the change point destroyed), same change point; for this comparison only, a detection counts when its lag is at most 150 decisions (so prompt detection is compared; the rate without the cap is reported next to it and is what conditions 1 and 2 use); shift-specific timing = detection rate on injected minus on shuffled streams with a Newcombe 95 % interval. (A Bernoulli chance floor at the detector's own alarm rate was considered and dropped before any test read, because over the hundreds of decisions after the change point it is close to 1 for any detector with a non-zero false-alarm rate and therefore carries no information.) Statistics: exact McNemar per (trace, placement) between the headline monitor and M0, and between the headline monitor and the better of M5a/M5b, Holm across the 5 traces x 3 placements within each comparison.

Success (all required):

1. no-fault FA <= 5 % on at least 4 of 5 traces on TEST;
2. shift-specific detection (Newcombe CI excluding 0) on at least 4 of 5 traces;
3. the headline alpha-based monitor beats the block-shuffled control (Newcombe CI excluding 0) on at least min(condition-2 traces, 4) traces and is non-inferior to the better of M5a/M5b at matched validation FA: pooled over traces, detection-rate difference with a paired bootstrap lower bound above -0.10.

If condition 3 fails but 1 and 2 hold, the paper states that the alpha-monitor is a free by-product of the mixture that is not a better detector than the reference detectors, with the numbers.

### 4.4 What the paper and the thesis say per outcome

- All three conditions: the monitor claim returns in the abstract and contribution 5 with the per-trace FA, detection and shift-specific detection of the headline monitor; the thesis chapter 6 monitor section and the "exploratory" threat in chapter 6 section 10 are updated; Table `tab:alpha_monitor` and the two monitor figures are regenerated from the new outputs by `make_outputs.py` extensions.
- Conditions 1 and 2 hold, 3 fails: monitor stays in the paper as a by-product with the comparison against Page-Hinkley/ADWIN stated.
- Condition 1 or 2 fails: the v2 reframing "exploratory" is kept and the measured cause is stated (validation versus test non-stationarity of the alpha stream, quantified by the share of test alpha-stream variance not explained by the validation null; plus the fraction of detections also present in the control).

## 5. A-A3 repair of the learned weight alpha(x)

### 5.1 Features and models

Causal features of window t from the observed series `series[s] = y[s, 0]` before the origin and from the experts' past errors: (f1) volatility `std(last 24 values) / std over CAL`, (f2) level gap `(mean(last 6) - mean(last 24)) / std over CAL`, (f3) slope of the last 24 values (OLS, per CAL std), (f4) label-delayed log error ratio `log(rolling MSE(Bolt) / rolling MSE(decomposition))` over the last 24 labelled windows (delay h). Standardized with CAL statistics. Windows with fewer than 24 past values use x = 0 (alpha = scalar). No feature uses the target of window t.

All models predict `alpha_t = clip(alpha_0 + b' x_t, 0, 1)` with alpha_0 the deployed scalar weight, trained by minimizing the actual mixture squared error (not a noisy local-weight regression target) except L4.

- L1 ridge to the scalar: closed-form weighted least squares on the regressors `diff_t (x)` with penalty lambda ||b||^2, lambda in {0.1, 1, 10, 100, 1000} x mean(sdd) chosen by blocked 5-fold CV (contiguous folds, h-window purge) on the validation slice.
- L2 depth-1 boosted stumps: Newton boosting of alpha(x) with stumps on one feature, shrinkage 0.1, rounds R in {5, 10, 20, 40} chosen by the same blocked CV. (The inventory's monotone constraints are dropped: no sign of the effect of these features can be stated a priori; depth 1, shrinkage and CV-chosen rounds are the regularizers. This is the only change against the inventory text.)
- L3 hierarchical pooling over the 25 cells: pooled coefficient vector b_pool from the stacked, cell-normalized ridge problem of all cells of the same seed plus per-cell deviations delta_c with ridge strength mu in {1, 10, 100, 1000, infinity (fully pooled)}; chosen by leave-one-cell-out blocked CV (the cell's own validation slice, others' coefficients frozen).
- L4 distillation: target `alpha*_t` = clipped closed form of V1's centred window of width W* = the V1 window chosen at SEL, student = ridge on x_t (as L1), student predicts alpha at test from causal features only.

Baselines: scalar CHA-S (V0) and Chronos-Bolt. Fair budget: the same CAL/validation windows, the same blocked CV grids, no test peeking.

Selection (validation only): per model the CV mean squared error pooled over cells and seeds; the headline model is the one with the lowest pooled CV MSE ratio to V0. Frozen in `frozen_a3.json` with its sha256 before the test read.

### 5.2 Controls, statistics, success rule

Controls: (a) feature-permutation control: the frozen model applied to the test features permuted across windows within the cell (contiguous blocks of h), 50 draws; (b) constant-alpha control: a constant alpha drawn from the model's own SEL alpha distribution, 50 draws. Per cell paired block permutation of the headline model against scalar CHA-S with Holm over 25 cells.

Success (all required): better than scalar CHA-S in at least 8 of 25 cells after Holm AND no cell worse than the scalar by more than 1 % relative RMSE AND a pooled gain over the better control whose cell-level bootstrap CI excludes 0.

### 5.3 What the paper and the thesis say per outcome

- Success: v4 (CHA-L) is reported with its regularized form as a contribution that improves the scalar baseline, with the cells and the control gap; the old CHA-L/CHA-LR table is kept as the unregularized comparison; thesis chapter 6 adds the method and the result.
- Failure: the paper and thesis say "the scalar weight is the sufficient statistic at this data size" with the measured control result and the per-model pooled CV ratios; the old untraceable numbers ("0.18 vs 0.06", "four configurations on a held-out CESNET subset") are removed or replaced by the traceable table.

## 6. Cross-item consistency and ordering

Order of execution in phase 2: A-A1 select (criterion acc and, if approved, criterion floor), A-A2 select, A-A3 select (all validation only, guard on), freeze the JSON files with hashes, then the test stages (A-A1 accuracy test only if Gate 0a passed; A-A1b test if approved). A-A1 V3 uses the A-A2 M1 threshold function; A-A3 L4 uses the A-A1 V1 window. Seeds: numpy `default_rng` seeds are written in each script and logged in `manifest_<item>.json` with the code hashes, the frozen-config hash and the machine state (load, free memory, taskset).

## 7. Resources

CPU only (`nice -n 19 taskset -c 8-11`, `OMP_NUM_THREADS=2`), memory under 4 GB per process, MemAvailable checked above 10 GB before each stage, GPU not needed. Progress line to `PROGRESS.md` at least every 5 minutes. No git.


## 8. Code manifest (end of phase 1, all in `experiments/repair_2026-10-12/paper_a/`)

| file | role | sha256 |
|---|---|---|
| `rcommon.py` | data roles, test guard (`install_guard`, `guard_selftest`, `allow_test`), pseudo-mode for dry runs, closed forms, permutation test | `747cddd2ac9c76b8ca61dccd76521c09a9beffeec59e80f214716d711785ac33` |
| `stats_tools.py` | block bootstrap, TOST, Holm helper, Newcombe, McNemar, block-bootstrap null levels | `3e8b97b101f1d8d73364d26876ee7b5e695708560628a6ba52a0e62079160fe1` |
| `gate0.py` | Gate 0a/0b (validation only) | `7888876ea37e1a702e3c03f539faa9242dd5d27d00c8eb644dac1393309cd98f` |
| `gate0c_existing.py` | test-side constant-weight oracle from existing published tables (no forecast file opened) | `2cc6ad946ff5b7095e94629807d2a054cbe6d105ccce78795e5a04c81bf46b37` |
| `a1.py` | A-A1 V0 to V5 and A-A1b (`--criterion acc|floor`), select and test stages | `d522b528b7dbba7533796a1ee8cbc762d60d8cc07afbed15a54a29d3ea98fb72` |
| `a2.py` | A-A2 detectors M0 to M5b, select and test stages | `d915197bacddf57efa687c819a41f7be10f2c2b7420443a45d6db05879234dc0` |
| `a3.py` | A-A3 models L1 to L4, select and test stages | `01dc4ffb12a5cffde3c0a2bf10818cec03cbe598afdea1de9b6c97368383a399` |

Dry runs (`--stage select --dry-run`, run under `nice -n 19 taskset -c 8-11`): each installs the guard, proves the guard blocks a real test read (`guard_selftest`), runs the select stage on 2 to 3 cells and 1 to 2 seeds, then runs the full test-stage code in pseudo-mode (CAL slice plays validation, SEL slice plays test, or the validation stream itself for the monitor), and asserts that the guard log contains no path under `outputs/predictions`. Result of all runs on 2026-10-12: `gate0.py --dry-run`, `a1.py --criterion acc`, `a1.py --criterion floor`, `a2.py`, `a3.py` each printed `dry-run ok; test files read: 0`. Checks inside the dry runs: the vectorised monitor streams equal `monitor_far.alpha_and_rmse_stream` (atol 1e-9), `eval_run` equals `monitor_far.eval_fault` for persistence 3, and the test stage asserts that V0 / scalar CHA-S reproduces the stored CHA-S forecast (atol 1e-4) on the real test run.

Run commands for phase 2 (in this order, all under `nice -n 19 taskset -c 8-11`, `OMP_NUM_THREADS=2`):

1. `python3 a1.py --stage select` and `python3 a1.py --stage select --criterion floor`; `python3 a2.py --stage select`; `python3 a3.py --stage select` (validation only).
2. `touch PREREG_COMMITTED` (the coordinator or this worker after the reply), then `REPAIR_STAGE=test python3 a2.py --stage test`, `REPAIR_STAGE=test python3 a3.py --stage test`, `REPAIR_STAGE=test python3 a1.py --stage test --criterion floor` (if approved), and `a1.py --stage test` only if the coordinator overrides Gate 0a (then labelled exploratory).

## 9. Gate 0 result (appended after the validation-only run; definitions in section 3.1 unchanged)

Run 2026-10-12 11:5x, 25 cells x 5 decomposition seeds, validation windows only, `gate0_cells.csv`, `gate0_summary.json`.

Gate 0a: FAIL. Cross-fitted block headroom (L = 48, adjacent blocks) is at least 0.5 % in 0 of 25 cells (L = 24: 1 of 25, L = 96: 0 of 25, previous-block causal variant: 0 of 25). Mean -2.56 %, median -1.74 %, maximum +0.14 %; per dataset Abilene -6.9 %, CESNET -1.1 %, GEANT -3.5 %, NAB AWS-CPU -0.1 %, NAB Twitter -1.1 %. The in-sample block oracle (upper bound, optimistic by construction) averages +1.71 % (Abilene 4.7 %, CESNET 1.8 %, GEANT 1.7 %, NAB Twitter 0.3 %, NAB AWS-CPU 0.0 %), and all of it disappears when the block weight is estimated without the block, i.e. it is fitting noise. The mean RMSE gain of the shipped constant weight over Bolt on validation is only +0.31 % (cell maximum 1.3 %), so even a perfect time-varying mixture would start from a ceiling of a few percent in-sample. The drift transfer loss (one-parameter weight fitted on one validation half scored on the other) averages 0.35 % and reaches 0.5 % in 4 cells (CESNET h12 1.2 %, CESNET h24 3.4 %, Abilene h3 0.7 %, Abilene h24 0.8 %).

Gate 0b: FAIL. A third expert (persistence) added by NNLS and fitted on CAL gains at least 0.5 % over the CAL-fitted scalar CHA-S on SEL in 3 of 25 cells (CESNET h1 1.3 %, CESNET h3 0.6 %, GEANT h6 0.7 %), mean +0.11 %.

Gate 0c (informational, existing published numbers only, `gate0c_existing.py`, no forecast file opened): the mixture MSE is exactly quadratic in alpha, so the stored test RMSEs of Bolt, decomposition and CHA-S fix the test-oracle constant weight. The test-oracle constant weight would improve on the shipped one by at least 0.5 % in 4 of 25 cells (Abilene h1 3.2 %, h3 2.9 %, h6 1.6 %, h12 0.6 %); in those cells the test-oracle weight is 0, i.e. pure Chronos-Bolt. Against Bolt the best constant mixture gains at most 0.57 % in any cell (mean 0.08 %, at least 0.5 % in 1 of 25 cells). So with a constant weight the ceiling above Bolt is below 1 % everywhere, and the 9 floor failures of v2 (excess up to 3.3 %, mean 1.03 % over the failing cells) are a validation-to-test shift in the weight, not a missing mixture gain.

Decision recorded: Gate 0a fails, so the accuracy repair V1 to V4 is infeasible by the pre-registered rule; V5 is infeasible by Gate 0b. A-A1 therefore keeps "equivalent to the foundation expert with a readable decomposition path". The measured reason for the paper and the thesis is the ceiling: the constant mixture can gain at most 0.57 % over Bolt even with the test-oracle weight, the validation-only time-varying headroom is pure noise after cross-fitting, and the five Abilene floor failures come from a weight that is optimal at 0 on test while the validation optimum is 0.04 to 0.10.

## 10. Addendum (written after the test stages, design only, nothing below has been run on test): a stronger complementary expert for a later accuracy round

Status: this is a proposal for a later round. It is written after the phase 2 test stages of this round, so it is not part of the frozen pre-registration of sections 1 to 8, and no test split is read for it until it is committed in its own file (`contract/REPAIR_PREREG_EXPERT.md` or a new section here with a new commit).

Why the ceiling is the experts, not the weight rule. The mixture MSE of Bolt and a second expert E with error variance ratio r = MSE_E / MSE_Bolt and error correlation rho has the closed-form optimum MSE_mix / MSE_Bolt = r (1 - rho^2) / (1 + r - 2 rho sqrt(r)). On the validation split the decomposition path has a measured mean r = 2.41 (range 1.42 to 8.40) and a mean rho = 0.66 (range 0.30 to 0.83), which gives an oracle constant-mixture gain of 0.4 % in RMSE on average and 1.3 % at most (`closed_form_cells.csv`, M0, M1 and the leading coefficient A, via rho sigma_B sigma_E = (M0 + M1 - A) / 2). A second expert changes the ceiling only if r and rho both move: r = 1.0 with rho = 0.8 gives 5.1 %, r = 1.0 with rho = 0.9 gives 2.5 %, r = 1.2 with rho = 0.8 gives 1.7 %, r = 1.2 with rho = 0.9 or r = 1.5 with rho = 0.8 gives 0.0 %. The requirement for a useful new expert is therefore Bolt-level accuracy (r <= 1.2) and an error correlation with Bolt of at most 0.8. The Gate 0c numbers (best constant mixture beats Bolt by at most 0.57 %) and the published XGBoost row (it beats Bolt in only 2 of 25 cells) are consistent with this.

Candidates.

- E-Q, probabilistic second expert. A quantile model on lag and calendar features (LightGBM or a quantile regression forest, quantiles 0.1, 0.5, 0.9) whose median is the point forecast and whose interval width is the confidence. The mixture then weighs the two experts by calibrated predictive variance (the inverse-variance form is the one that failed for point errors in v2, so the variance must be calibrated on CAL by the empirical coverage of the intervals, with Bolt's own quantile outputs as the second variance source). Motivation: Bolt's quantile spread is unused in CHA-S, and an expert whose errors are heteroscedastic with a predictable scale gives a weight that can vary with a causal signal other than the past error ratio.
- E-R, regime-aware decomposition expert. The decomposition path with a regime variable (a two-state threshold on the label-delayed level and volatility features of section 5.1, or a Markov-switching residual model) and a separate residual LSTM or Theta parameter set per regime, so that the part of the decomposition that failed at the Abilene regime shift is the part that is modelled. Motivation: the five Abilene floor failures are a regime shift between validation and test, and the present decomposition expert has r = 2.2 to 2.9 there.
- E-X, retained as a reference only: XGBoost lag features, already published (2 of 25 cells better than Bolt), no new run.

Gate E0 (validation only, same CAL, SEL and FINAL FIT roles as section 1, guard installed, one number per cell averaged over the five seeds). For a candidate expert E whose validation forecasts are produced by training on the train split only:

1. Competence. RMSE_E / RMSE_Bolt <= 1.15 on CAL in at least 15 of 25 cells.
2. Decorrelation and ceiling. The oracle constant mixture of Bolt and E, computed from r and rho on CAL (formula above), improves on Bolt by at least 1.0 % in RMSE in at least 8 of 25 cells; and the same quantity measured on SEL with the weight fitted on CAL (out-of-sample) improves by at least 0.5 % in at least 6 of 25 cells.
3. Three-expert headroom. NNLS stacking of [Bolt, decomposition, E] fitted on CAL and scored on SEL improves on the CAL-fitted scalar CHA-S by at least 0.5 % in at least 6 of 25 cells.
4. Cross-fitted time-varying headroom. The gate 0a measurement of section 3.1 (adjacent-block weight, L = 48) on the pair [Bolt, E] reaches at least 0.5 % in at least 6 of 25 cells.

Rule. If conditions 1 and 2 hold, the candidate is run through the variant machinery of section 3.2 (V0 constant, V1 recency, V2 shrinkage, V4 per-step, and the three-expert V5) with the test protocol of section 3.4 and the success rule of section 3.4 (at least 5 of 25 cells significantly better than Bolt after Holm, none significantly worse, pooled gain over the better random control with a cell-level bootstrap CI excluding 0), with condition 3 or 4 selecting between the two-expert and the three-expert form. If condition 1 or 2 fails the candidate is closed with its measured r and rho as the reason. Fair baseline and control: Bolt, current CHA-S, the best of the two experts on CAL, and an expert of the same family with permuted features (a model trained on feature rows shuffled across windows within the training split), because a weaker or noisier expert can lower the correlation without adding information.

Cost estimate (not a commitment): E-Q is CPU, a few minutes per cell per seed with LightGBM, about 25 x 5 fits for validation and test forecasts; E-R reuses the stored STL and Theta components and retrains the residual LSTM per regime on GPU or CPU, about 1 to 2 GPU hours; both need the validation forecasts regenerated without touching the test files, as `val_expert_forecasts.py` did.
