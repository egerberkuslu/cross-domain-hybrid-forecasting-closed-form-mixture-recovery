
## Round 7 (REPAIR_PREREG_7), 2026-10-12 23:34:02
Stage A (train-only, validation forecasts) ran on all monitor cells at h = 6 (P7 and the G7 GEANT stratum); the degenerate-expert rule (validation forecasts only) fixed the final list in degen7.json; the select stage ran on validation arrays only (the r7_test folder is guarded and absent); frozen before any stage B read:
- frozen_r7.json sha256 2bec42f96111bbd4f328c24b92d111b5d2dee1ce0658a39174f89650ddd5e16c
- degen7.json sha256 978edaed2ed61d1e656132094e423ba5a7e7c2494970d959513b4fa74821507d; pool7_manifest.json sha256 963430afa0c8e0c1b06177fdbfbd785460889fd5694d038f33876a948a9759f2; reserved_round7.json sha256 58785533e467e7668ecfd5dd4b35ff2b9cf800892a68ef0787c0b69b0433b5ed
- r7_val (57 files) sha256 of the sorted per-file sha256sum list: 823c6d3ff6239bdcf140c7dc7152b0f50c0863a4435e8981c24da44a8b49475b
- r7_models (57 files) sha256 of the sorted per-file sha256sum list: 64613b60786e6b8cb8459286e934578e49095d1f4d436bd21faa35b52e03e450
- Commands: run_r7.sh (nice 19, taskset 12-15, OMP_NUM_THREADS=1, two workers, REPAIR_STAGE=test only for stage B and the test stage).
- RECIPE7_FROZEN.json sha256 df53c6228ad5bc45d4232c075f8b6730acd03128cd5c704858da78004aa75ba8 (freeze7.py, after the select run and the dry run); dryrun_log.txt sha256 46568797bde765e31fe8c86f89cc0bf5ffa281e6a401dcef488a9ab7f9b39a1d ends with "dry-run ok; test windows accessed: 0"; r7_test is absent.
- Degenerate-expert rule (validation forecasts only): cesnet_sub5 dropped (median q_val -8.53), replaced by cesnet_sub459; final pool 30 subnets + 11 IPs = 41 traces (+ 15 GEANT G7). Power on n = 41: power7.json block actual_pool.
- Pool shortfall versus the planned 60: see REPAIR_PREREG_7.md R7-7 and alt7_counts.json.
- Operational note: run_r7.sh closes the lock descriptor in the heartbeat subshell (an orphaned sleep 300 held run_r7.lock after the UNTIL=A run until killed); stage A, select and the degen loop were run with UNTIL=A and UNTIL=select.

## Round 7, first select run SUPERSEDED 2026-10-12 (before any commit)
The first select run (41 traces: 30 subnets + 11 IPs, frozen_r7.json 2bec42f96111bbd4f328c24b92d111b5d2dee1ce0658a39174f89650ddd5e16c, RECIPE7_FROZEN.json df53c6228ad5bc45d4232c075f8b6730acd03128cd5c704858da78004aa75ba8, REPAIR_PREREG_7.md f7deffb095f42df93f96fca60cda58f1d239a9b2e84a70bbc4ff89471616874a) is superseded by the coordinator decision after the pre-commit review: the monitor traces of round 3 (cesnet_inst193, cesnet_inst225) join the excluded parents, and the near-copy rule (corr_parent_trainval >= 0.95 and parent read in an earlier round, train and validation hours only) is registered. cesnet_sub437 (parent 193) and cesnet_sub480 (parent 225) leave the pool; the validation forecast of cesnet_sub437 stays in r7_val (an orphan cell: it is listed in R7_VAL_SHA256.txt and in the r7_val list hash, but it is not in pool7_manifest.json monitor_order, degen7.json or frozen_r7.json and is never scored). No test window existed at any time.

## Round 7 (REPAIR_PREREG_7), 2026-10-12 23:52:36
Stage A (train-only, validation forecasts) ran on all monitor cells at h = 6 (P7 and the G7 GEANT stratum); the degenerate-expert rule (validation forecasts only) fixed the final list in degen7.json; the select stage ran on validation arrays only (the r7_test folder is guarded and absent); frozen before any stage B read:
- frozen_r7.json sha256 41b8e1a608fb10877ac658034861870c3e1d43c3950a33c31010cd172e2de7d3
- degen7.json sha256 d168834b3460c068eb41c9b7af941ea34ee788e71698955fb60277ad5cf64e4d; pool7_manifest.json sha256 f89b0ef26fa86759f41e7bf51b70392a741ab5ecb37201cac6a69323ed4d05d0; reserved_round7.json sha256 ec415961f1248df2b445815be9d80621497bf2f36b4c6840517603795ce34857
- r7_val (57 files) sha256 of the sorted per-file sha256sum list: 823c6d3ff6239bdcf140c7dc7152b0f50c0863a4435e8981c24da44a8b49475b
- r7_models (57 files) sha256 of the sorted per-file sha256sum list: 64613b60786e6b8cb8459286e934578e49095d1f4d436bd21faa35b52e03e450
- Commands: run_r7.sh (nice 19, taskset 12-15, OMP_NUM_THREADS=1, two workers, REPAIR_STAGE=test only for stage B and the test stage).
- Second freeze (final, 40 traces = 29 subnets + 11 IPs, 15 G7): RECIPE7_FROZEN.json sha256 e198ac466ab78ce46bac59fb5a2e05c2852ad2438073377868d7420a7be60ce2; dryrun_log.txt sha256 05a61de7c1fdff8cc875bd5c8e61a20274c747e80718871b7a468facb90d29bc ends with "dry-run ok; test windows accessed: 0"; r7_test absent; power7.json block actual_pool (n = 40, k = 32) and feasibility_adjusted.
