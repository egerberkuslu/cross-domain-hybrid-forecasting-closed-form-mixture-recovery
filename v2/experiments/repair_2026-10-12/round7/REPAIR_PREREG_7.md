# REPAIR PRE-REGISTRATION 7, Paper A: scale-transfer-robust reference-free shift monitor on fresh CESNET subnet and IP-address traces

Status: **final text, not yet registered.** It becomes registered only when the coordinator writes `round7/PREREG_7_SHA256` (the sha256 of this file) and `round7/PREREG_7_COMMITTED` and the commit exists. This file was prepared from `REPAIR_PREREG_7_DRAFT.md` after the coordinator decisions D1 to D7 (section 10), the counts-only pool build, stage A (train-only, validation forecasts), the validation-only degenerate-expert rule, the validation-only selection and the freeze. An earlier freeze with 41 traces was superseded before any registration by the coordinator decisions after the pre-commit review (round 3 monitor parents excluded, near-copy rule; `RUN_NOTES_R7.md`). No round 7 test window has been read: `round7/r7_test` does not exist and the dry run printed `dry-run ok; test windows accessed: 0` (`round7/dryrun_log.txt`). Rounds 1 to 6 are frozen and untouched. Folder: `experiments/repair_2026-10-12/round7/`.

## 0. Why round 7, and disclosures

Round 7 is the **seventh monitor rule** of Paper A and the **fifth pre-registered test of a reference-free monitor** (registrations 3, 4, 5, 6 and 7; registration 1 tested a shift monitor and registration 2 a drift-adaptive monitor), following the erratum in `round6/RUN_NOTES_R6.md`. The paper says so whatever the outcome. Earlier outcomes under their own rules: round 4, 9, 11 and 8 of 12 against 10; round 5, 58, 62 and 44 of 64 against 52; round 6, V1 to V3 at 15 of 20 against 16 (Holm p 0.062), static baseline 14 of 20.

- **R7-1, designed after seeing rounds 4 to 6.** The diagnosis (`DIAGNOSIS7.md`) used the round 6 validation streams and the **revealed round 6 test streams and rows**. Its finding: the c2 failures are blind detectors whose validation scale does not transfer to the test stream, and one trace has a degenerate E-Q. The candidate repairs were selected on validation and scored on the revealed round 5 and round 6 test streams (`val7.py`). These numbers are design inputs and are never reported as evidence. Rounds 5 and 6 are not re-scored, and both remain failures under their registered rules.
- **R7-2, the variant set was chosen after seeing those revealed scores.** The two-stage confirmation variant (W2 in `val7.py`) was dropped because it lost detection there. W4 was built after the first `val7.py` run as the combination of W1 and W3. The variants' revealed round 6 counts (15, 18 and 18 of 20) are therefore optimistic.
- **R7-3, the degenerate-expert rule** (section 1) was motivated by cesnet_inst243 on the revealed round 6 test (decision D6). It uses only validation forecasts.
- **R7-4, power inputs** (section 6) come from the revealed re-analysis. No parameter is tuned on them: every per-trace parameter is selected on validation by the frozen code.
- **R7-5, new data of an already used network (decision D1).** The primary traces are CESNET-TimeSeries24 subnets and IP addresses. They have never been forecast, but they are components of the same network and the same calendar as the CESNET institution series whose test windows rounds 3 to 6 and the F+G arms have read: of the 40 final traces, 39 have a parent institution that was allocated to an earlier round (accuracy, pilot, F+G or monitor pool). Subnets and IPs whose parent institution was a **revealed monitor trace of any earlier round (rounds 3, 4, 5 and 6)** are excluded (32 institutions; the round 3 monitor traces are cesnet_inst193 and cesnet_inst225, and the first freeze of this round had wrongly left them out, which let cesnet_sub437 and cesnet_sub480 in; both have an hourly series identical to their parent's, correlation 1.000 on train and validation hours). **Near-copy rule, registered now and using train and validation hours only:** a drifting candidate whose correlation with its parent institution series on those hours is at least 0.95 and whose parent institution was allocated to an earlier round is dropped (`corr_parent_trainval`, `parent_used_earlier_round`, `corr_rule_excluded` in `pool7_manifest.json`). After the round 3 exclusion the rule drops no further trace (`corr_rule_excluded` is empty); the two near-copies above leave through the parent exclusion, and either rule alone removes them. The highest remaining correlations are 0.87 (subnet 249, parent 61) and 0.84 (subnet 164, parent 23). The orphan validation forecast of cesnet_sub437 stays in `r7_val` (it is listed in `R7_VAL_SHA256.txt` and in the `r7_val` list hash) but is in no manifest, degen or frozen list and is never scored.
- **R7-6, Disclosure 3 of round 6 still holds.** A residual CUSUM lost earlier ("D4 dersi", `[brain: wiki/concepts/cha-s.md]`) and is not a variant.
- **R7-7, the pool is smaller than planned.** The draft planned 60 traces. The registered rules (decision D7) give 40: the IP-address sample has only 65 of 1,000 series with at most 5 % missing hours, and the family cap of three per institution, counted for subnets and IPs together, removes 29 of 59 drifting subnets and part of the IPs (most come from institution 2, which has 105 subnets). The rules were not tuned to reach 60. `alt7_counts.json` lists, as counts only, what relaxing the missing limit (DATA7.md rule M7a) or the cap would add (at most 54 under M7a with the cap applied per level; its shift is the inventory `tv_shift`, which differs marginally from `build_pool4.shift`, 58 against 59 drifting subnets under the registered rule); none of these is adopted. The pool rule is kept because under R80 the power does not improve with a larger pool at the plausible pass rates: at pi = 0.80 it is 0.57 to 0.63 for every n from 20 to 80 (`power7.json`). The success threshold and the power of section 6 are those of n = 40.
- **R7-8, the degenerate-expert rule dropped one trace.** `cesnet_sub5` has a median validation q of -8.53 (E-Q's validation MSE about 5047 times Bolt's, more than 100 times). No reserve subnet was left after the round 3 exclusion, so it was not replaced and the pool has 29 subnets. `degen7.json` lists every median q. The 15 G7 traces are not subject to the rule (no replacement exists); `g7_flagged_degenerate_kept` in `degen7.json` is empty.
- **R7-9, the GEANT stratum has 15 traces, not 16.** `geant_od5_2` has a train-to-validation shift of 0.2996 train sd under `build_pool4.shift`, the screen applied to every trace of every round, against 0.30 in the inventory screen of `data7_inventory.py`; it is below the 0.3 rule and is not a monitor trace. G7 is descriptive and never pooled (decision D4).
- **R7-10, subnets and IPs overlap in institutions.** The 40 final traces come from 26 institutions (20 for the 29 subnets, 9 for the 11 IPs). Per-institution results are reported.

## 1. Traces

- **P7 primary pool, n = 40 (29 subnets, 11 IP addresses), long design.** CESNET-TimeSeries24 `institution_subnets` and `ip_addresses_sample` (Zenodo 10.5281/zenodo.13382427, CC BY 4.0), downloaded to `data/raw/cesnet_ts24_extra/` with md5 checked against the Zenodo record by `download7.sh`. Built by `build_pool7.py` (counts only) with the `fresh_cells` pipeline: hourly `n_bytes`, outlier flag at 100 x median, gaps of at most 24 h interpolated in time, longer gaps leave windows out, split 0.70 / 0.15 / 0.15, scaler on train, lookback 168, h = 6. The calendar is `data/raw/cesnet/times.tar.gz`, the file the shipped loaders use (md5 `a03813763e07646ca38f17ffd53e549e`, equal to the Zenodo record).
- **Candidate rule (decision D7, fixed before any forecast, the round 6 `build_pool6.py` rule with the round 6 CESNET missing-data limit).** Missing share at most 0.05, zero share at most 0.05, median positive, parent institution not a monitor trace of rounds 3 to 6 and not a near-copy of an earlier-used parent (R7-5); then the drift screen `build_pool4.shift` >= 0.3 train sd (train and validation hours only) and at least 690 complete test windows at h = 6 (all candidates have 1,007).
- **Counts.** Hourly files in the archives: 546 subnets (the relationship file lists 548) and 1,000 IP addresses. Admissible: 451 subnets and 65 IPs. Parent excluded (before the admissibility screen, all ids): 77 subnets and 130 IPs. Built and screened candidates: 389 subnets and 44 IPs. Drifting with enough windows: 59 subnets and 19 IPs; after the near-copy rule 59 and 19. After the family cap (at most 3 per institution, subnets and IPs together, the IP cap being 3 minus the institution's selected subnets): 30 subnets and 11 IPs. A subnet that the relationship file lists under two institutions counts under the smaller institution id for the cap and is excluded if either parent was a revealed monitor trace.
- **Selection (decision D7).** Per level the list is cut by the family cap in id order and then ordered by the stride rule of `fresh_cells.select_series` (every k-th element, k = floor(n / 30), first element first; the remaining elements follow in id order). The first 30 of the subnet order are primary (after the family cap exactly 30 subnets remain, so the reserve is empty; the IP order has 11 elements, all primary). Subnets are taken before IPs. The reserve of the degenerate-expert rule is 0 subnets. The order is in `pool7_manifest.json` (`monitor_order`, `monitor_primary`).
- **Degenerate-expert rule (decision D6, validation only, applied after stage A and before selection).** A trace is dropped iff |median q_val| > log(100), that is, one expert's validation MSE is more than 100 times the other's. A dropped trace is replaced by the next trace in the registered order, skipping any trace that would break the family cap. This is the only step that uses forecasts before the freeze, and it uses validation forecasts only (`r7_monitor.py --stage degen`, `degen7.json`). Result: 1 trace dropped (`cesnet_sub5`), not replaced (no reserve); final list 29 subnets and 11 IPs in 26 institutions.
- **G7 descriptive stratum (decision D4):** the 15 GEANT leftovers admissible under rule M7a of `DATA7.md` (b) that pass the shift screen, short design (about 428 test windows; change points 200, midpoint, n - 160), as the round 6 P3 stratum. It is reported separately, never pooled, and has no success rule.
- **Not in this round (decision D2):** Telecom Italia Milan.
- **Additive blocks** `round7_C` (every built subnet and IP id plus the excluded parents) and `round7_M` (the primary order, the reserve, the G7 ids) for `fresh_cells/reserved_series.json` are in `reserved_round7.json`; the coordinator merges them.

## 2. What stays frozen

Unchanged from round 6: the statistic q (log MSE ratio of Chronos-Bolt to E-Q over the last 60 labelled windows, label delay 6, horizon-averaged h = 6), e, the burn-in of 100, persistence 3, the injected fault (`monitor_far.inject`, 3 sigma ramp over 100 windows), the RF reference (EW decay 0.99, delayed by G), G in {30, 60}, K in {0.5, 1, 2}, the level grid 1 to 60 in steps of 0.5, NULL_TARGET 0.10, FA target 0.05, the corrected per-trace conditions c1, c2 and c3' with L = 150 and MARGIN 0.5, the change points {420, midpoint, n - 160} (G7: {200, midpoint, n - 160}), 10 seeds per change point (new seeds: `a2.test_seed` with s = 7), the validation injections (positions 0.3 and 0.6, 5 seeds), the E-Q recipe and the Chronos-Bolt snapshot (`RECIPE7_FROZEN.json`: amazon/chronos-bolt-small snapshot `772f3d25d38aec6d914c8949dab4462e2d46f5d8`, weights sha256 `06a6a19bbe74bc10a9cd193bd4bf2bf638ae07f7e0d51653ae7ab8ea968a21dd`).

## 3. Arms

Every arm uses the **revised fallback and selection key**. Selection prefers a (G, K) with a feasible level, then higher validation detection, then lower validation null, then larger G. If no level meets the three null conditions (stream itself, bootstrap q90 of the null rate, bootstrap q90 of the false-alarm share), the level is the smallest one meeting the validation-stream null and FA conditions (flagged infeasible, `fallback: self`), and otherwise 60 (`fallback: max`). This replaces round 6's K = 2 at level 60, the least sensitive point of the grid (`DIAGNOSIS7.md` section 5). Implementation: `round7/r7_lib.py`, from the `val7.py` functions (`rf_z_rs`, `pcfit`, `lmb_rows`, `calibrate`, `select`), checked before the freeze against `val7.py`: identical selected parameters of W1, W3, W4 and STB7 on two revealed validation streams (cesnet_inst51, abilene_od0) and identical det, ctl, p0 and J on one revealed round 6 test stream (cesnet_inst51, four arms).

- **W1, SRC (segment-residual calibration).** The validation q is segmented by binary segmentation (at most 4 change points, segments of at least 120, accepted when the standardised mean difference is at least 1.0). Level, K and G are calibrated on the residual q - fit (moving-block bootstrap of the residual). Floor = max(0.5 sd(residual), 0.05). The EW variance is Huber-clipped at 3 sd. On test the RF statistic runs on raw q, warm-started on the last 200 validation decisions. Design input: revealed r6 15/20, r5 54/64.
- **W3, RS-LMB.** Floor = max(0.5 x 1.4826 MAD(q_val), 0.05), Huber EW variance, level calibrated on raw q with the level-matched block bootstrap (each block shifted so its first 20 values match the mean of the last 20 values of the previous block). Same warm start. Design input: r6 18/20, r5 54/64.
- **W4, RS-LMB-RF.** W3 with the floor of W1. Design input: r6 18/20, r5 56/64.
- **Fair baseline, STB7.** Static CUSUM on q: reference = mean of the last 200 validation decisions (the warm segment of the variants), scale = sd of the validation residual. K grid, null-calibrated level, residual calibration, fallback, re-arm, burn-in hold and onset-localised scoring are identical to W1. It differs from W1 only in having no reference-free standardisation. It is the single registered baseline. Design input: r6 9/20, r5 38/64.
- **Report only:** V3 (the round 6 V3 recipe, re-selected on round 7 validation) to separate the effect of the repair from the effect of new data; and ADB (ADWIN) as in round 6.
- **Random control, RND.** As in round 6 section 3: each arm's own number of no-fault test events placed uniformly after the burn-in, 200 draws, seed 20261017 + trace index, scored by the same rule.
- **Not registered.** The two-stage confirmation (it lost detection on the revealed streams, `DIAGNOSIS7.md` section 4). The E-Q/Bolt forecast-disagreement statistic (decision D5): the injected fault perturbs only the Bolt forecast, so a disagreement statistic reacts to the injection mechanism itself and would pass for that reason, not because it detects real shifts.

Validation-only facts of the frozen selection (no test value involved): feasible level found on 35 of 40 P7 traces for W1, 18 for W3, 17 for W4, 40 for STB7 and 36 for V3 (G7: 15, 15, 15, 15, 15 of 15). The remaining W1, W3 and W4 traces run on the revised fallback; the number is reported per trace in the paper. The design pass rates of `val7.py` were higher on feasible traces than on infeasible ones, so the feasibility counts matter for power: section 6 gives the feasibility-adjusted rates. STB7 is selected with its validation detection scored around the validation mean `q.mean()` (as in `val7.py`), while on test its reference is the mean of the last 200 validation decisions; this is the frozen construction and is disclosed.

## 4. Per-trace conditions

Unchanged from round 6 section 4: c1 (no-fault onset share <= 5 %), c2 (Newcombe 95 % lower bound of det - ctl > 0, 30 injected runs against 3 matched clocks), c3' (det - max(p_dense, p_rand) >= 0.5, with c2). J = c1 and c2 and c3'. The cause classes for failures are those of `classify6_r6.py` with class (2) split by the control (`classify7.py`): c1 false-alarm excess; c2 insensitive (ctl = 0); c2 control alarms (ctl > 0); c3 null exceeded; c3 slow detection. Det and ctl are reported separately.

## 5. Design input (revealed, not evidence)

`DIAGNOSIS7.md` section 4, `val7_summary.json`, `val7_w4_summary.json`. Pooled revealed pass rates over 84 traces: W1 0.821, W3 0.857, W4 0.881, V3 0.845, STB7 0.560. The random-control c3' share is at most 2.1 % for every arm. These streams are not part of round 7.

## 6. Success rule, Holm and power

For each variant v in {W1, W3, W4}, X_v is the number of P7 traces with J, out of n = 40 (subnets and IP addresses pooled):

1. **Count.** X_v >= ceil(0.8 n) = 32 (rule R80, unchanged from rounds 4 to 6; decision D3).
2. **Holm.** p_v = P(Bin(n, 0.5) >= X_v), Holm-adjusted across the three variants, adjusted p < 0.05.
3. **Random-control validity.** The mean random-control c3' pass share is at most 0.05.

The claim succeeds if at least one variant meets 1 to 3. The headline is the variant with the largest X_v; ties go to W1, then W3, then W4. This choice is fixed now. Secondary claim, not a success condition: one-sided exact McNemar of each succeeding variant against STB7, Holm-adjusted across the succeeding variants. The G7 stratum has no success rule.

**Power** (`power7.json`, block `actual_pool`; independent traces, Holm first step alpha / 3, exact binomial): n = 40, k = 32.

| pi | 0.75 (round 6 actual) | 0.80 | 0.821 (W1 pooled) | 0.845 (W1, W3 r5) | 0.857 (W3 pooled) | 0.875 (W4 r5) | 0.881 (W4 pooled) |
|---|---|---|---|---|---|---|---|
| power, R80 | 0.3 | 0.593 | 0.72 | 0.843 | 0.892 | 0.945 | 0.958 |
| power, R80, design effect 1.5 (n_eff 27, k 22) | 0.299 | 0.539 | 0.649 | 0.768 | 0.82 | 0.888 | 0.906 |

Stated power: about 0.843 to 0.958 if the repaired variants keep their revealed round 5 and pooled rates (0.845 to 0.881) on the new traces, about 0.593 if pi falls to 0.80, and about 0.3 at the round 6 rate; the smallest pi with power 0.8 is 0.836. With n = 40 the power at the design rates is slightly below the 60-trace plan of the draft (0.87 at 0.845); the rule is not changed (decision D3). Feasibility-adjusted design rates (`power7.json`, block `feasibility_adjusted`): re-weighting the revealed pass rates of feasible and infeasible traces (84 revealed traces) to the feasibility shares of this pool gives pi = 0.8326 for W1 (feasible share 0.875), 0.8314 for W3 (0.45) and 0.8641 for W4 (0.425), hence power 0.784, 0.777 and 0.915 at n = 40 (with the design effect 0.708, 0.702 and 0.849). This is the power statement to quote; the unadjusted rows above use the pooled design rates.

The alternative rule R60 (binomial against 0.6) would give 0.732 at pi = 0.80 and is **not** adopted. If R80 is missed, the miss is reported with these numbers and not reinterpreted.

## 7. Guards and conduct

- `Restricted` view for every stage before the commit: `fresh_cells.Restricted` hides test windows in stage A, and `r7_lib.guard_test_dir` extends the `rcommon` guard to `round7/r7_test` in the degen, select and dry-run stages, asserts that the folder does not exist and that a read of it raises. The test forecasts are read only through `r7_lib.TestView`, which refuses without `PREREG_7_COMMITTED` and `REPAIR_STAGE=test` and calls `rcommon.allow_test()`.
- The test stage (`r7_monitor.py --stage test`) needs `round7/PREREG_7_COMMITTED` and `REPAIR_STAGE=test`. It rechecks that the sha256 of this file equals the first field of `PREREG_7_SHA256`, that the sha256 of `frozen_r7.json`, `pool7_manifest.json`, `degen7.json`, `RECIPE7_FROZEN.json`, `r7_lib.py`, `r7_monitor.py`, `r7_stages.py` and `classify7.py` appear in section 8 of this file and equal the current files, that `frozen_r7.json` equals `frozen_r7.sha256` and its recorded code hashes, and that every hash of `RECIPE7_FROZEN.json` (earlier-round modules and round 7 code) equals the current file. `run_r7.sh` repeats the integrity check before stage B.
- Dry run: `r7_monitor.py --stage select --dry-run` on the non-fresh Abilene aggregate prints `dry-run ok; test windows accessed: 0` to `round7/dryrun_log.txt` (done before the freeze; sha256 in section 8).
- Order: download and counts-only `build_pool7.py` (done); stage A on 57 cells (the 40 final P7 traces, the 15 G7 traces, the dropped `cesnet_sub5` and the orphan `cesnet_sub437`; done); the degenerate-expert rule (done); power on the actual n (done, section 6); select and freeze (done); dry run (done); this registration and its commit; stage B; test; then the paper and thesis update in the same round.
- Resources: `nice -n 19`, `taskset -c 12-15`, at most 2 workers, the run script refuses to start below 8 GB MemAvailable and waits while it is below 8 GB, no single process over 4 GB, archives streamed one at a time, resumable script with a heartbeat line (with MemAvailable) every 5 minutes in `round7/PROGRESS.md`. No git by workers.

## 8. Code and data manifest (sha256)

`RECIPE7_FROZEN.json` (sha256 `e198ac466ab78ce46bac59fb5a2e05c2852ad2438073377868d7420a7be60ce2`) lists the E-Q recipe, the Chronos-Bolt snapshot and every hash below; `frozen_r7.json` is the frozen parameter file.

| file | sha256 |
|---|---|
| `frozen_r7.json` | `41b8e1a608fb10877ac658034861870c3e1d43c3950a33c31010cd172e2de7d3` |
| `RECIPE7_FROZEN.json` | `e198ac466ab78ce46bac59fb5a2e05c2852ad2438073377868d7420a7be60ce2` |
| `power7.json` | `ae6a9702c1045a0c3fdfaa2d2cdce303ea418f108f4e6ccf632341c21050ccdb` |
| `pool7_manifest.json` | `f89b0ef26fa86759f41e7bf51b70392a741ab5ecb37201cac6a69323ed4d05d0` |
| `reserved_round7.json` | `ec415961f1248df2b445815be9d80621497bf2f36b4c6840517603795ce34857` |
| `degen7.json` | `d168834b3460c068eb41c9b7af941ea34ee788e71698955fb60277ad5cf64e4d` |
| `dryrun_log.txt` | `05a61de7c1fdff8cc875bd5c8e61a20274c747e80718871b7a468facb90d29bc` |
| `alt7_counts.json` | `66abcee9f4c6a8466374e59ff77c90ae41a4ec96a94f39c70856852f9fc147d0` |
| `DIAGNOSIS7.md` | `672b44a0e9ca8ec767f9133851562f22e68f8a6695c7a1576a13b95c5e1ddf45` |
| `DATA7.md` | `5982f2c0585b8649fdea6d6fdb6f3965687aff1027f8f89bf3d720ad8d15dc8f` |
| `val7_summary.json` | `5391a7c9238ade9428c5bd07e18f63ee22cb49847891b1302ab8ade142fc94df` |
| `val7_w4_summary.json` | `deae22a89418df5d26de5656c654a4f188291aafeecf3037cdfc28d29401ee92` |
| `R7_VAL_SHA256.txt` | `823c6d3ff6239bdcf140c7dc7152b0f50c0863a4435e8981c24da44a8b49475b` |

Round 7 code:

| file | sha256 |
|---|---|
| `r7_lib.py` | `1da05e630f1cddebb0fe76154f3aaac4ba395c42f2225f3e03d99b243d6f8e97` |
| `r7_monitor.py` | `7dfbfe129aa60223cb702e172eed9480ab745ef7c150bae7100e9a31611bfbc6` |
| `r7_stages.py` | `b6d78493e9d27d9ce0e18b33e5dfe46f46dc428300a1fbd9dec11038ea88f03f` |
| `classify7.py` | `671c360192b5b6877cffaf90c1a23d2be534221aac424f5c276507c3f0f8219f` |
| `build_pool7.py` | `e2e2815f1e992fb18d1cbb974f2abccb2de6e85198d030d484fd614e9b9433a8` |
| `freeze7.py` | `204e1f32d01e971904ca70594285fe4de6b1c77c1be54f7e2fb5feb0b9fd8f86` |
| `power7.py` | `3be3a5f8e1816847a0dfcc8d1abcba12951172b242b68363e50b276b6b3976dc` |
| `run_r7.sh` | `ff0ff3679a4bd70925d240d65d78e231ba123dc9afb3cb3b5d15a36910bff03c` |
| `download7.sh` | `0152c747aead9cdcb303d4f6df6d68882eae828543dacb974b2cedff3e01e18b` |
| `alt7_counts.py` | `a003d1cddf8ee390348d02f2ae6fc37bdde5fc2ab41819e47fd1fa013c628466` |
| `diag7.py` | `7593423c3985cd3233ad9da87a3ad6e243d48ad7818ffcc292089432e49102ee` |
| `diag7_extra.py` | `c56fd319d0c963408e04ef05c8c4dce9127c90a80f78f67bf854dd288a534041` |
| `val7.py` | `746583228ea04c1e021eac8c0df1746f29308afa46588ea8937aa55e6af8af1e` |
| `data7_inventory.py` | `30cc3fe283692939de539e8d5e2b604693d541286f34b025cb5f9efcb7150028` |

Earlier-round modules imported unchanged:

| file (relative to `experiments/repair_2026-10-12/`) | sha256 |
|---|---|
| `paper_a/a2.py` | `d915197bacddf57efa687c819a41f7be10f2c2b7420443a45d6db05879234dc0` |
| `paper_a/stats_tools.py` | `3e8b97b101f1d8d73364d26876ee7b5e695708560628a6ba52a0e62079160fe1` |
| `paper_a/rcommon.py` | `747cddd2ac9c76b8ca61dccd76521c09a9beffeec59e80f214716d711785ac33` |
| `round3/eq_recipe.py` | `d45d3ae7ea731d580d3199ce27f5c6f4cd48cee5bc52e2857fa4597bc9cfcf24` |
| `round3/RECIPE_FROZEN.json` | `5a96181ebe5633f76daf34646a301c6d5453e2b60e8187b99a57f258c590cde8` |
| `round3/r3_stages.py` | `6b3f2e79169e729e73bd17da3f7012a7b2133e5f15623756b957b2c1cb327521` |
| `round3/r3_monitor.py` | `f99a3f690eef95a95bf5c9f7032299f1d95206273e2404b665a7cb884c8b1e15` |
| `round4/r4_monitor.py` | `86f9deccbd028214da37a6cc656d5fdbed4806c01ab1519f4bf81c6fb2f432b1` |
| `round4/frozen_r4_guard.json` | `566443299bcb80eed5df57d7f9a7afa9cb96b3d07c1185b9ef4a59a9c161a4d8` |
| `round5/r5_stages.py` | `1d840c9cd1a73d8194ebd148083b6bd50ef90438a89716fff21d3bdd80ee083b` |
| `round5/RECIPE5_FROZEN.json` | `b78bbedfa3e2dac67735b34220db08c5a6c74bd9b0eaacb57f34d974110c2f5c` |
| `fresh_cells/fresh_cells.py` | `f4f5a4da72b34d3938865a36f8b66d970d270f8f76387a20d1b6c715f3a50660` |
| `fresh_cells/build_pool4.py` | `44a5dab666886d1381476e9457ff4f7fe6d835058ec7db7e1118b0bd87ed9060` |
| `round6/r6_lib.py` | `32a0a541ccabff7ec8b66550b75eb3f472cfb8157a1a66ddce38f7725f8fb0d7` |
| `round6/RECIPE6_FROZEN.json` | `453c6748671209f3975a9cd4c8921aa4f5487c4510161655b6984030d2dc4a16` |
| `../fix_2026-10-11/monitor_far.py` | `c281d994bcf2bcedb35de47eaa979ce861dd0d024e353b73fc421e539e9a782c` |

Stage A (train-only, validation forecasts; 57 cells, including the dropped `cesnet_sub5` and the orphan `cesnet_sub437`): sha256 of the sorted per-file `sha256sum` list of `r7_val` `823c6d3ff6239bdcf140c7dc7152b0f50c0863a4435e8981c24da44a8b49475b`, of `r7_models` `64613b60786e6b8cb8459286e934578e49095d1f4d436bd21faa35b52e03e450`, per-file list in `R7_VAL_SHA256.txt`. Every series file and `splits.json` has its sha256 in `pool7_manifest.json`.

Data: Zenodo 10.5281/zenodo.13382427, md5 from the record and sha256 of the downloaded files.

| file | md5 (Zenodo record) | sha256 |
|---|---|---|
| `ids_relationship.csv` | `f034e4e7f1844e36fa7d3a4a0cbfc600` | `559f99f3cbe0eedf24ba3926aeae0fd88172c2ea62ef7b2327a1888ecb1516bd` |
| `institution_subnets.tar.gz` | `735b5ff436b6c67556cb67e2888b14ea` | `f765dec60691af8a21f55a650fe3d1aafed59d7eee8a165edff569147ccb9127` |
| `ip_addresses_sample.tar.gz` | `08451dab2d1eddb29a79467b03289232` | `aeb1e31a149684a99f951ce41b4ad9d8ee79c52682ec14524c8591f867d717dd` |

## 9. Outcome text (fixed now)

**Success.** "A reference-free CUSUM on the expert error ratio, calibrated on validation with a burst- and regime-robust scale [variant wording], detected injected shifts above its own no-fault alarm rate, above a rate-matched random-alarm control and specifically against a matched no-injection control on X of 40 fresh drifting CESNET subnet and IP-address traces, against the 32 required, Holm-adjusted one-sided binomial p = P." The same paragraph:

- states that this is the seventh monitor rule and the fifth pre-registered test of a reference-free monitor, and that the four earlier reference-free tests missed their rules (with their counts);
- states that the design was made after seeing rounds 4 to 6, including the revealed round 6 test streams (R7-1 to R7-4);
- states that the traces are components of the same network as earlier CESNET institution series, that the pool has 40 instead of the planned 60 traces and excludes near-copies of earlier-used institution series (R7-5, R7-7);
- gives the per-level (subnet, IP) and per-institution rates with Wilson intervals, the STB7, V3 and ADB rates, the random-control size, the feasibility counts and the G7 descriptive rates.

The comparative sentence against STB7 appears only if McNemar holds. The result may enter the contributions, the results, the highlights and thesis chapter 6, each with this disclosure. It does not enter the abstract.

**Failure.** The paper reports X_v with Wilson intervals, the per-level table, the stated power range and the registered cause classes (`classify7.py`: class (2) split into "insensitive: ctl = 0" and "control alarms: ctl > 0"). It states that five pre-registered tests of a reference-free monitor, among seven monitor rules, missed their rules. No monitor claim enters the abstract, contributions, highlights or cover letter, and the thesis chapter carries the same negative result. Under Ege's rule a further repair would need a new registration that discloses round 7.

## 10. Decisions of the coordinator (settled)

- **D1 primary pool:** never-forecast CESNET-TimeSeries24 subnets and the IP-address sample (Zenodo 13382427, CC BY 4.0; `institution_subnets`, `ip_addresses_sample`, `ids_relationship.csv`, 945 MB, checksums verified); subnets and IPs whose parent institution was a revealed monitor trace of any earlier round (3 to 6) are excluded, plus the near-copy rule of R7-5; disclosed as components of the same network as earlier institution series (R7-5).
- **D2 Milan:** not in this round.
- **D3 success rule:** unchanged (at least 80 % of the primary pool, one-sided binomial against 50 % with Holm over the variants, random control at most 5 %).
- **D4 GEANT M7a:** a separate descriptive stratum G7, never pooled (15 traces, R7-9).
- **D5 disagreement statistic:** none.
- **D6 degenerate-expert rule:** drop a trace if its median validation q is beyond +-log 100, validation only, disclosed as motivated by inst243 in the revealed round 6 test (R7-3, R7-8).
- **D7 pool:** 60 traces planned (30 subnets, 30 IPs), at most 3 per institution counting subnets and IPs together, deterministic selection by sorted id after the drift screen, counts only; the rules yield 40 (R7-7).
- **Review of the first freeze:** after the pre-commit review the round 3 monitor parents and the near-copy rule were added, the feasibility-adjusted power row and the STB7 note were added, and the first 41-trace freeze was superseded.
