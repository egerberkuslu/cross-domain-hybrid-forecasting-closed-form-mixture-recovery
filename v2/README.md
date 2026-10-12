# v2: corrected analysis and reproducibility scripts

This folder holds the corrected statistical analysis that accompanies the revised version of the manuscript. Release **v1.0.0** is the version that was submitted and is left unchanged. Release **v2.0.0** adds this folder and does not alter any v1 file, apart from a pointer to this folder in the top-level README. The forecasts, checkpoints and training code are the v1 ones, so v2 reruns no model and trains nothing. Every number below is recomputed from the per-window prediction tensors of the v1.0.0 release.

## What changed against v1

A re-examination of the reported numbers found claims that the original analysis did not support, and the revised manuscript replaces them with the following.

1. **Permutation test.** The one-sided win-rate test with per-window sign flips is replaced by a seed-paired, two-sided block permutation test (block length equal to the horizon h, Holm correction over the 17 cells whose forecasts differ). No cell favours CHA-S at any block length. The three CESNET wins of v1 are gone, and the sensitivity to the block length (1, h, 2h, 24) is reported.
2. **Diebold-Mariano test.** The DM statistic now uses one loss differential per forecast origin (the squared error averaged over the h steps) with a Newey-West variance, instead of treating the overlapping steps as independent. Benjamini-Hochberg and Holm corrections are reported, and bit-identical forecasts count as ties.
3. **Equivalence test (TOST).** Seed 42 only, because the zero-shot Chronos-Bolt run exists for one seed: 14 equivalent cells, 10 identical cells, 1 inconclusive cell (v1 stated 24 of 25).
4. **Closed-form weight.** The clipped estimator that is actually used is compared with the 21-point grid: mean absolute gap 0.009 and maximum 0.025 over 825 cells (v1 reported a mean gap of 0.029 and a worst case of 0.18).
5. **Alpha monitor.** The thresholds are designed on the validation stream only, the smoother is causal, the labels are delayed by h, and a no-fault control is added. False-alarm rate, detection rate and lag are reported per trace (no-fault false alarms at or below 5 % on three of five traces).
6. **Degradation sweep, stacked OLS, multi-metric SDs.** The degradation weights are fitted on validation and scored on test. The stacked least-squares comparison is fitted on validation with an intercept. The bootstrap standard deviations of the CHA-S rows were about sqrt(5) too small and are recomputed with per-seed pairing.
7. **Wording.** Statements such as "costs no accuracy" are replaced by the measured quantities.

## Layout

```
v2/
  README.md
  requirements-analysis.txt
  peerj_v2/
    scripts/            analysis scripts of the revised manuscript
    results_v2/         their outputs (CSV, JSON, logs, the two monitor figures)
  experiments/fix_2026-10-11/
    *.py, run_all.sh    recomputation scripts (closed form, TOST, degradation, stacked OLS, ...)
    *.csv, *.json       their outputs
    val_forecasts/      validation-split expert forecasts (126 files, 4.6 MB)
    independent_check/  independent re-derivation of the DM kernels
```

## Environment

The analyses ran on CPU with Python 3.12.4, numpy 1.26.4, scipy 1.11.4 and pandas 3.0.3 (`requirements-analysis.txt`). Only `val_expert_forecasts.py` needs torch, darts and chronos-forecasting (the environment of the main `requirements-lock.txt`); its output is shipped in `val_forecasts/`, so it does not have to be rerun. Timings were measured on an AMD Ryzen 7 5800X.

## Getting the inputs

The scripts read per-window predictions and checkpoint summaries, which are large and live in the v1.0.0 release assets, not in git. From the repository root:

```bash
gh release download v1.0.0 -p 'cha-*.tar.gz'      # or download from the Releases page
mkdir -p outputs
for t in predictions checkpoints metrics; do tar xzf cha-$t.tar.gz -C outputs; done
```

This produces `outputs/predictions/`, `outputs/checkpoints/` and `outputs/metrics/` (all three are git-ignored). The scripts locate the repository root from their own path, or from the environment variable `CHA_ROOT` when it is set. Two kinds of file are not in the release and are therefore not needed to reproduce a number. The legacy tables `outputs/eval_v3/` are used only for "before" comparison columns, and the scripts skip those comparisons when the folder is missing. The predictions of the five newer foundation models and their hybrids (generation sweep) are used only by the `--generation` option of `dm_tost_identity.py` and `permutation_two_sided.py`, which the paper does not cite.

## Reproducibility table

Commands run from `v2/peerj_v2/scripts/` for the first group and from the repository root for `experiments/` scripts. All scripts accept no arguments unless shown.

| Result in the paper | Script | Output |
|---|---|---|
| Seed-paired block permutation test against Chronos-Bolt, block length h, Holm over 17 cells: 0 better, 14 ties, 3 worse (Abilene h=3 and h=6, NAB Twitter h=3), 8 identical | `peerj_v2/scripts/perm_analysis.py` | `results_v2/perm_summary.json`, `results_v2/perm_cells_blockh.csv` |
| Block-length sensitivity: block 1 gives 0 / 9 / 8, block 2h gives 0 / 16 / 1, block 24 gives 0 / 17 / 0 | `peerj_v2/scripts/perm_analysis.py` | `results_v2/perm_cells_block1.csv`, `perm_cells_block2h.csv`, `perm_cells_block24.csv`, `perm_summary.json` |
| Per-origin DM test with Bartlett weights, lag h-1 and the HLN correction; against Chronos-Bolt 0 wins, 22 ties, 3 losses (Abilene h=1, 3, 6); over 17 baselines 226 / 196 / 3 uncorrected | `peerj_v2/scripts/dm_analysis.py` | `results_v2/dm_summary.json`, `dm_cells.csv`, `dm_per_baseline.csv` |
| Benjamini-Hochberg count (215 / 207 / 3) and Holm count (151 / 274 / 0), corrected over the 417 comparisons that differ | `peerj_v2/scripts/dm_analysis.py` | `results_v2/dm_summary.json` (`origin_bh`, `origin_holm`), `dm_cells.csv` (`p_origin_bh`, `p_origin_holm`) |
| Rectangular variance kernel in place of Bartlett: BH leaves 2 losses (Abilene h=1, 3), Holm totals 127 / 298 / 0 | `experiments/fix_2026-10-11/independent_check/dm_kernel_check.py` | `independent_check/dm_kernel_check.csv` (columns `p2` and `s2`) |
| Flattened DM (overlapping steps treated as independent): 276 / 135 / 14, NAB Twitter h=24 win with p=0.0007 against 0.36 per origin | `peerj_v2/scripts/dm_analysis.py` | `results_v2/dm_cells.csv` (columns `p_flat`, `p_origin`, `verdict_flat_raw`) |
| TOST equivalence on seed 42: 14 equivalent, 10 identical, 1 inconclusive (Abilene h=1) | `experiments/fix_2026-10-11/dm_tost_identity.py --B 2000 --block h` | `tost_cells.csv` (mode `legacy_iid`), `dm_tost_summary.json` |
| Alpha monitor design thresholds (validation only, EMA lambda = 9/11, label delay h) | `peerj_v2/scripts/mon_analysis.py` | `results_v2/mon_thresholds.csv`, `mon_summary.json` |
| Monitor no-fault false-alarm rate per trace (at or below 5 % on three of five) | `peerj_v2/scripts/mon_analysis.py` | `results_v2/mon_nofault.csv`, `mon_cells.csv` (`nofault_fa_mean`) |
| Monitor detection rate and median lag per trace, and the no-injection control | `peerj_v2/scripts/mon_analysis.py` | `results_v2/mon_runs.csv`, `mon_control.csv`, `mon_cells.csv` (`det_rate_inj`, `lag_median_inj`, `det_rate_ctl`) |
| Monitor stream figure and comparison figure | stream data from `peerj_v2/scripts/mon_analysis.py` (the plotting script is not included) | `results_v2/mon_example_streams.csv`, `alpha_monitor.png`, `alpha_monitor_compare.png` |
| NAB placement variant: change points moved to 0.2, 0.45, 0.7 of the post-warm-up stream | `peerj_v2/scripts/mon_placement_variant.py` | `results_v2/mon_placement_variant.json` |
| Closed-form fit time (median 42 microseconds) against the 21-point grid (308 microseconds), 200 repetitions on one pinned core | `peerj_v2/scripts/time_closed_form_monitor.py` | `results_v2/timing_closed_form_monitor.json` |
| Closed form against grid over 825 cells (mean 0.009, maximum 0.025); validation-floor count 16 of 25; clip counts per dataset | `experiments/fix_2026-10-11/closed_form_stats.py` | `closed_form_summary.json`, `closed_form_cells.csv`, `floor_bound_test.csv` |
| Degradation sweep with weights fitted on validation and scored on test | `experiments/fix_2026-10-11/degradation_oos.py --horizons 6 --replicates 20` | `degradation_oos_cells.csv`, `degradation_oos_summary.json`, `degradation_oos_runs.csv` |
| Stacked least squares with intercept against the grid weight (16 of 25 cells) | `experiments/fix_2026-10-11/stacked_ols.py` | `stacked_ols_cells.csv`, `stacked_ols_summary.json` |
| Multi-metric bootstrap standard deviations with per-seed pairing (h=6) | `peerj_v2/scripts/recompute_multimetric_sd.py` (run from the repository root) | `peerj_v2/scripts/multimetric_h6_recomputed.csv`, `multimetric_h6_before_after.csv`, `multimetric_h6_block6.csv` |
| Cross-check of the same bootstrap standard deviations (fix-wave implementation) | `experiments/fix_2026-10-11/bootstrap_pairing.py --B 1000` | `bootstrap_paired.csv`, `bootstrap_paired_h6_table.csv` |

`monitor_injection_runs.csv` (the per-run injection results of the fix-wave monitor script `monitor_far.py`) is shipped gzip-compressed because the plain file is 12 MB.

`experiments/fix_2026-10-11/run_all.sh` runs the `experiments/` scripts in order (set `SKIP_VAL=1` to use the shipped `val_forecasts/`). Those scripts never overwrite an existing output file and write a timestamped sibling instead.

## Reproduction checks

The permutation, DM, monitor, placement-variant and multi-metric scripts were rerun from a fresh clone with the v1.0.0 assets extracted into `outputs/`.

| Script | Result of the rerun |
|---|---|
| `perm_analysis.py` | all five output files byte-identical to the shipped ones |
| `mon_analysis.py` | all output files byte-identical, including the figures' source CSVs |
| `mon_placement_variant.py` | `mon_placement_variant.json` byte-identical |
| `recompute_multimetric_sd.py` | three CSV files byte-identical |
| `dm_analysis.py` | identical verdicts and counts; the p-values differ from the shipped file by at most 1e-13 (floating-point summation order), and one JSON entry differs in its last digit |
| `independent_check/dm_kernel_check.py` | CSV byte-identical; counts 226/196/3, 215/207/3, 151/274/0 (Bartlett), 200/223/2 and 127/298/0 (rectangular) |

The closed-form timing depends on the machine and is not expected to match to the microsecond.

## Not included

The manuscript sources (text, tables, figures as typeset), the scripts that typeset tables and figures for the manuscript and apply text edits to it, internal verification notes and run logs, the raw datasets, and the large artifacts of v1.0.0 (checkpoints, predictions, metrics) are not part of this folder.
