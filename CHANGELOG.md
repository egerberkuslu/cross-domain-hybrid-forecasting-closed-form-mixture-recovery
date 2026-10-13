# Changelog

## v2.1.0

v2.1.0: pre-registered repair rounds 3-7 (accuracy round 5, monitor round 7), K6/K7 fixes

- Adds `v2/experiments/repair_2026-10-12/`: the registrations (`REPAIR_PREREG*.md`, `PREREG_n_COMMITTED`, `PREREG_n_SHA256`), frozen parameter files and digests (`RECIPE*_FROZEN.json`, `frozen_*.json`, `frozen_*.sha256`), stage scripts, power calculations, run scripts, summary JSON and the small CSV result tables of repair rounds 1 and 2 (`paper_a/`) and rounds 3 to 7, with the pool builders and series manifests of `fresh_cells/`. Fitted models, per-cell forecast arrays and derived series are not included; `v2/experiments/repair_2026-10-12/README.md` lists them with sizes.
- Adds `v2/experiments/repair_2026-10-12/verify_frozen.py`, which recomputes the frozen SHA-256 digests (125 in this release).
- Adds `v2/peerj_v2/scripts/make_round5_numbers.py`, `make_round6_numbers.py`, `make_round7_numbers.py` and `make_repair_table.py` with their outputs (`numbers_round{5,6,7}.{json,tex}`, `tables/`). The scripts read the repair-round folders relative to their own location; a rerun reproduces the shipped outputs byte for byte.
- Documents per-round commands and compute estimates in the top-level README and in the repair folder README (main benchmark about 3 h; repair round 5 about 2.5 h for stage A and about 5 h for stage B).
- K6 and K7 are the sixth and seventh review gates (independent verifier and red-team) that the revised manuscript passed through after the repair rounds. Their fixes (2026-10-14) scope the manuscript claims to what the registrations test: the round 7 monitor result stays out of the abstract, the detector-family sentence, the equivalence and E-Q claims, the controls against Chronos-Bolt and the reproducibility statement. They change the manuscript text; no file of v2.0.0 is modified.
- `.gitignore` ignores the local links `v2/data`, `v2/src`, `v2/outputs` and `v2/.chas-venv` that the repair run scripts expect.

## v2.0.0

Corrected analysis and reproducibility scripts (2026-10-12): seed-paired block permutation test, per-origin Diebold-Mariano test, equivalence test on seed 42, clipped closed-form weight against the 21-point grid, alpha monitor with validation-designed thresholds and a no-fault control, out-of-sample degradation sweep, stacked least squares, multi-metric bootstrap standard deviations. See `v2/README.md`.

## v1.0.0

Submitted version: code, configurations, provenance log, and the release assets (checkpoints, prediction tensors, metrics).
