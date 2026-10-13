# REPAIR PRE-REGISTRATION 7 (DRAFT), Paper A: scale-transfer-robust reference-free shift monitor on fresh CESNET subnet and IP traces

Status: **draft, not registered.** It becomes registered only when the coordinator has settled the decisions in section 10, the frozen code and hashes of section 8 exist, `round7/PREREG_7_SHA256` and `round7/PREREG_7_COMMITTED` are written and the commit exists. No round 7 trace has been built, forecast or read. Rounds 1 to 6 are frozen and untouched. Folder: `experiments/repair_2026-10-12/round7/`.

## 0. Why round 7, and disclosures

Round 7 is the **seventh monitor rule** of Paper A and the **fifth pre-registered test of a reference-free monitor** (registrations 3, 4, 5, 6 and 7; registration 1 tested a shift monitor and registration 2 a drift-adaptive monitor), following the erratum in `round6/RUN_NOTES_R6.md`. The paper says so whatever the outcome. Earlier outcomes under their own rules: round 4, 9, 11 and 8 of 12 against 10; round 5, 58, 62 and 44 of 64 against 52; round 6, V1 to V3 at 15 of 20 against 16 (Holm p 0.062), static baseline 14 of 20.

- **R7-1, designed after seeing rounds 4 to 6.** The diagnosis (`DIAGNOSIS7.md`) used the round 6 validation streams and the **revealed round 6 test streams and rows**. Its finding: the c2 failures are blind detectors whose validation scale does not transfer to the test stream, and one trace has a degenerate E-Q. The candidate repairs were selected on validation and scored on the revealed round 5 and round 6 test streams (`val7.py`). These numbers are design inputs and are never reported as evidence. Rounds 5 and 6 are not re-scored, and both remain failures under their registered rules.
- **R7-2, the variant set was chosen after seeing those revealed scores.** W2 (two-stage confirmation) was dropped because it lost detection there. W4 was built after the first `val7.py` run as the combination of W1 and W3. The variants' revealed round 6 counts (15, 18 and 18 of 20) are therefore optimistic.
- **R7-3, the degenerate-expert rule** (section 1) was motivated by cesnet_inst243 on the revealed round 6 test. It uses only validation forecasts.
- **R7-4, power inputs** (section 6) come from the revealed re-analysis. No parameter is tuned on them: every per-trace parameter is selected on validation by the frozen code.
- **R7-5, new data of an already used network.** The primary traces are CESNET subnets and IP addresses. They have never been forecast, but they come from the same network and calendar as the CESNET institutions whose test windows rounds 5 and 6 and the F+G arms have read. Families whose parent institution was a round 5 or 6 monitor trace are excluded, and family caps apply.
- **R7-6, Disclosure 3 of round 6 still holds.** A residual CUSUM lost earlier ("D4 dersi", `[brain: wiki/concepts/cha-s.md]`) and is not a variant.

## 1. Traces

- **P7 primary pool, n = 60, long design.** CESNET-TimeSeries24 `institution_subnets` (30) and `ip_addresses_sample` (30), Zenodo 10.5281/zenodo.13382427, CC BY 4.0. They are built by `build_pool7.py` (counts only) with `fresh_cells.build_series` adapted to the archive path. Screen: `build_pool4.shift` >= 0.3 train sd (train and validation hours only). Length: at least 690 h = 6 test windows. Admissibility: the original `fresh_cells` rule, plus missing-data rule M7a (`DATA7.md` (b)), plus the degenerate-expert rule. Selection: stride in id order over the admissible drifting list, at most 3 per institution family. Excluded: single-subnet institutions, and families whose parent was a round 5 or 6 monitor trace. If a level has fewer than 30 traces the other fills in. If both together give fewer than 60, n is the total, and the power of section 6 is recomputed **before** stage A and recorded here.
- **Degenerate-expert rule (validation only, applied after stage A and before selection).** A trace is dropped iff |median q_val| > log(100), that is, one expert's validation MSE is more than 100 times the other's. Dropped traces are listed and replaced by the next trace in stride order. This is the only step that uses forecasts before the commit, and it uses validation forecasts only.
- **P7-M second stratum (optional, decision D2):** Telecom Italia Milan internet activity, 5 x 5-square macro-cells, 10-minute resolution, up to 30 traces, with the same screen and design. It is reported separately with its own Holm family and never pooled.
- **P7-G descriptive stratum (optional, decision D4):** the 16 GEANT leftovers admissible under M7a, short design, as in round 6 P3.
- An additive block `round7_C` (and `round7_M`) for `fresh_cells/reserved_series.json` is written as a separate file and merged by the coordinator.

## 2. What stays frozen

These are unchanged from round 6: the statistic q (log MSE ratio of Chronos-Bolt to E-Q over the last 60 labelled windows, label delay 6, horizon-averaged h = 6), e, the burn-in of 100, persistence 3, the injected fault (`monitor_far.inject`, 3 sigma ramp over 100 windows), the RF reference (EW decay 0.99, delayed by G), G in {30, 60}, K in {0.5, 1, 2}, the level grid 1 to 60 in steps of 0.5, NULL_TARGET 0.10, FA target 0.05, the corrected per-trace conditions c1, c2 and c3' with L = 150 and MARGIN 0.5, the change points {420, midpoint, n - 160}, 10 seeds per change point (new seeds: `a2.test_seed` with s = 7), the validation injections (positions 0.3 and 0.6, 5 seeds), and the E-Q recipe and Bolt snapshot (`RECIPE6_FROZEN.json` modules), for P7.

## 3. Arms

Every arm uses the **revised fallback and selection key**. Selection prefers a (G, K) with a feasible level, then higher validation detection, then lower validation null, then larger G. If no level meets the three null conditions, the level is the smallest one meeting the validation-stream null and FA conditions (flagged infeasible), and otherwise 60. This replaces round 6's K = 2 at level 60, the least sensitive point of the grid (`DIAGNOSIS7.md` section 5). Implementation: `round7/r7_lib.py`, frozen from `val7.py` functions (`rf_z_rs`, `pcfit`, `lmb_rows`, `calibrate`).

- **V7-1, SRC (segment-residual calibration).** The validation q is segmented by binary segmentation (at most 4 change points, segments of at least 120, accepted when the standardised mean difference is at least 1.0). Level, K and G are calibrated on the residual q - fit (moving-block bootstrap of the residual). Floor = max(0.5 sd(residual), 0.05). The EW variance is Huber-clipped at 3 sd. On test the RF statistic runs on raw q, warm-started on the last 200 validation decisions. Design input: revealed r6 15/20, r5 54/64.
- **V7-2, RS-LMB.** Floor = max(0.5 x 1.4826 MAD(q_val), 0.05), Huber EW variance, level calibrated on raw q with the level-matched block bootstrap (each block shifted so its first 20 values match the mean of the last 20 values of the previous block). Same warm start. Design input: r6 18/20, r5 54/64.
- **V7-3, RS-LMB-RF.** V7-2 with the floor of V7-1. Design input: r6 18/20, r5 56/64.
- **Fair baseline, STB7.** Static CUSUM on q: reference = mean of the last 200 validation decisions (the warm segment of the variants), scale = sd of the validation residual. K grid, null-calibrated level, residual calibration, fallback, re-arm, burn-in hold and onset-localised scoring are identical to V7-1. It differs from V7-1 only in having no reference-free standardisation. It is the single registered baseline. Design input: r6 9/20, r5 38/64.
- **Report only:** V3 (the round 6 V3 recipe, re-selected on round 7 validation) to separate the effect of the repair from the effect of new data; and ADB as in round 6.
- **Random control, RND.** As in round 6 section 3: each arm's own number of no-fault test events placed uniformly after the burn-in, 200 draws, seed 20261017 + trace index, scored by the same rule.
- **Not registered.** The two-stage confirmation (it lost detection on the revealed streams, `DIAGNOSIS7.md` section 4). The E-Q/Bolt forecast-disagreement statistic: the injected fault perturbs only the Bolt forecast, so a disagreement statistic reacts to the injection mechanism itself and would pass for that reason, not because it detects real shifts (decision D5).

## 4. Per-trace conditions

Unchanged from round 6 section 4: c1 (no-fault onset share <= 5 %), c2 (Newcombe 95 % lower bound of det - ctl > 0, 30 injected runs against 3 matched clocks), c3' (det - max(p_dense, p_rand) >= 0.5, with c2). J = c1 and c2 and c3'. The cause classes for failures are those of `classify6_r6.py`, with class (2) renamed in the text to "insensitive or not shift specific", with det and ctl reported separately.

## 5. Design input (revealed, not evidence)

`DIAGNOSIS7.md` section 4, `val7_summary.json`, `val7_w4_summary.json`. Pooled revealed pass rates over 84 traces: V7-1 0.821, V7-2 0.857, V7-3 0.881, V3 0.845, STB7 0.560. The random-control c3' share is at most 2.1 % for every arm. Validation feasibility of V7-3 is low (10 of 20 and 37 of 64), so many traces run on the revised fallback. It is reported per trace.

## 6. Success rule, Holm and power

For each variant v in {V7-1, V7-2, V7-3}, X_v is the number of P7 traces with J, out of n = 60:

1. **Count.** X_v >= ceil(0.8 n) = 48 (rule R80, unchanged from rounds 4 to 6).
2. **Holm.** p_v = P(Bin(n, 0.5) >= X_v), Holm-adjusted across the three variants, adjusted p < 0.05.
3. **Random-control validity.** The mean random-control c3' pass share is at most 0.05.

The claim succeeds if at least one variant meets 1 to 3. The headline is the variant with the largest X_v; ties go to V7-1, then V7-2, then V7-3. This choice is fixed now. Secondary claim, not a success condition: one-sided exact McNemar of each succeeding variant against STB7, Holm-adjusted.

**Power** (`power7.json`, independent traces, Holm first step): n = 60, k = 48.

| π | 0.75 (round 6 actual) | 0.80 | 0.821 (V7-1 pooled) | 0.845 (V7-1/V7-2 r5) | 0.857 (V7-2 pooled) | 0.875 (V7-3 r5) | 0.881 (V7-3 pooled) |
|---|---|---|---|---|---|---|---|
| power, R80 | 0.23 | 0.58 | 0.73 | 0.87 | 0.92 | 0.97 | 0.98 |
| power, R80, design effect 1.5 (n_eff 40) | 0.30 | 0.59 | 0.72 | 0.84 | 0.89 | 0.95 | 0.96 |

Stated power: about 0.85 to 0.97 if the repaired variants keep their revealed round 5 rates on the new traces, about 0.58 if π falls to 0.80, and about 0.23 at the round 6 rate. The alternative rule R60 (binomial against 0.6) would give 0.87 at π = 0.80. It is **not** adopted in this draft (decision D3). If R80 is missed, the miss is reported with these numbers and not reinterpreted.

## 7. Guards and conduct

- `Restricted` view for every stage before the commit. The select stage and the dry run extend the `rcommon` guard to `round7/r7_test`, and a read of it must raise.
- The test stage needs `round7/PREREG_7_COMMITTED` and `REPAIR_STAGE=test`. It rechecks the hashes of `frozen_r7.json`, `pool7_manifest.json`, `r7_lib.py`, `r7_monitor.py`, `r7_stages.py`, the recipe manifest and this file against `PREREG_7_SHA256`.
- Dry run: `r7_monitor.py --stage select --dry-run` on the non-fresh Abilene aggregate prints `dry-run ok; test windows accessed: 0` to `round7/dryrun_log.txt`.
- Order: download and counts-only `build_pool7.py`; stage A (validation forecasts); the degenerate-expert rule; select and freeze; dry run; power recomputed on the actual n if n < 60; registration and commit; stage B; test; then the paper and thesis update in the same round.
- Resources: `nice -n 19`, `taskset -c 12-15`, at most 2 workers, wait while MemAvailable < 10 GB, no single process over 4 GB, archives streamed one at a time, resumable script with a heartbeat in `round7/PROGRESS.md`. No git by workers.

## 8. Code manifest (to be filled at freeze)

`r7_lib.py`, `r7_monitor.py`, `r7_stages.py`, `build_pool7.py`, `freeze7.py`, `power7.py` (sha256 of the current version to be recorded), `diag7.py`, `diag7_extra.py`, `val7.py`, `data7_inventory.py`, `DIAGNOSIS7.md`, `DATA7.md`, `pool7_manifest.json`, `frozen_r7.json`, `R7_VAL_SHA256.txt`, `RECIPE7_FROZEN.json`, `dryrun_log.txt`: `[TODO: hashes at freeze]`.

## 9. Outcome text (fixed now)

**Success.** "A reference-free CUSUM on the expert error ratio, calibrated on validation with a burst- and regime-robust scale [variant wording], detected injected shifts above its own no-fault alarm rate, above a rate-matched random-alarm control and specifically against a matched no-injection control on X of 60 fresh drifting CESNET subnet and IP-address traces, against the 48 required, Holm-adjusted one-sided binomial p = P." The same paragraph:

- states that this is the seventh monitor rule and the fifth pre-registered test of a reference-free monitor, and that the four earlier reference-free tests missed their rules (with their counts);
- states that the design was made after seeing rounds 4 to 6, including the revealed round 6 test streams (R7-1 to R7-4);
- states that the traces come from the same network as earlier CESNET traces (R7-5);
- gives the per-level (subnet, IP) and per-family rates with Wilson intervals, the STB7, V3 and ADB rates, the random-control size and the feasibility counts.

The comparative sentence against STB7 appears only if McNemar holds. The result may enter the contributions, the results, the highlights and thesis chapter 6, each with this disclosure. It does not enter the abstract.

**Failure.** The paper reports X_v with Wilson intervals, the per-level table, the stated power range and the registered cause classes (`classify7.py`, the same rule as `classify6_r6.py`, with class (2) split into "insensitive: ctl = 0" and "control alarms: ctl > 0"). It states that five pre-registered tests of a reference-free monitor, among seven monitor rules, missed their rules. No monitor claim enters the abstract, contributions, highlights or cover letter, and the thesis chapter carries the same negative result. Under Ege's rule a further repair would need a new registration that discloses round 7.

## 10. Open decisions (see the report)

D1 primary source, D2 Milan stratum, D3 success rule, D4 GEANT stratum, D5 disagreement statistic, D6 degenerate-expert rule, D7 pool size and family caps.
