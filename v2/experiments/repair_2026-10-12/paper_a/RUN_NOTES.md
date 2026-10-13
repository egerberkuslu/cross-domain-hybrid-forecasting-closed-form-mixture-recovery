# RUN_NOTES (phase 2, 2026-10-12 11:53:58)

Frozen before any test read. Prereg commit 5e142c0; REPAIR_PREREG.md sha256 bbb07525e2b6a9b144531966548d199e9793ebda1421aba99c26309893134cc5.

| file | sha256 |
|---|---|
| frozen_a1.json | 73ebe607972790cf1842aa26dffba0d56a08431043826c43afc20577ec392b82 |
| frozen_a1b.json | b4d4c3e57a2b0f34dc2a887024547be055807fbaa8557d09febcb5bc2f97d257 |
| frozen_a2.json | f6f71d5f5bd8d5cdbd86919a8fe5a2724c03206422b05fb8b152147a6571b180 |
| frozen_a3.json | 9982c4e3db341717e81b68a2baa495b1bef66de62113dad2a5bf31733d57d29a |
| a1.py | d522b528b7dbba7533796a1ee8cbc762d60d8cc07afbed15a54a29d3ea98fb72 |
| a2.py | d915197bacddf57efa687c819a41f7be10f2c2b7420443a45d6db05879234dc0 |
| a3.py | 01dc4ffb12a5cffde3c0a2bf10818cec03cbe598afdea1de9b6c97368383a399 |
| rcommon.py | 747cddd2ac9c76b8ca61dccd76521c09a9beffeec59e80f214716d711785ac33 |
| stats_tools.py | 3e8b97b101f1d8d73364d26876ee7b5e695708560628a6ba52a0e62079160fe1 |

Headlines selected on validation: A-A1 acc V2 (lam=0.75; test NOT run, Gate 0a failed); A-A1b floor V2 lam=0.0 (alpha=0, i.e. pure Chronos-Bolt); A-A2 M1; A-A3 L1 (pooled mean log CV ratio -0.0043).
Commands: nice -n 19 taskset -c 8-11, OMP_NUM_THREADS=2, REPAIR_STAGE=test for the test stages. Machine load before test: load average: 14.33, 11.74, 10.51

## Outcomes (test stages run 2026-10-12, REPAIR_STAGE=test)

- A-A1 accuracy (V1-V4): not run on test, Gate 0a failed (0/25), Gate 0b failed (3/25), Gate 0c bound 0.57 %.
- A-A1b floor (approved): headline V2 lam=0 (alpha=0) holds 25/25 as identical copy of Bolt, controls 25/25, V0 18/25 at 0.1 % tolerance, sign test p=0.0078 vs V0 but control condition fails, success_floor=false. Exploratory (a1b_explore.py): V1 8, V3 1, V4 18 cells.
- A-A2: headline M1; FA <= 5 % on 3/5 (0.004, 0.000, 0.031, 0.164, 0.128 for CESNET, Abilene, GEANT, AWS, Twitter), shift-specific detection 3/5, M1 vs M0 no discordant run, ADWIN FA ok on 5/5 but control alarms in all pairs; success=false. Exploratory diagnosis a2_diag.csv: NAB test alpha stream moved relative to validation.
- A-A3: headline L1; 1 cell better, 4 worse > 1 % (Abilene h1 +9.9 %, h3 +8.4 %), gain over best control -0.48 % (CI -1.17, +0.04); success=false.
- Addendum for a stronger complementary expert: REPAIR_PREREG.md section 10 (design only, nothing run).
- Paper A v2: Repair round subsection + tables/repair_round.tex + CHANGES_V2.md, built 32 pp, 0 errors. Thesis: sections/06_tahmin/09b_onarim-turu.tex, built 365 pp, 0 errors, 0 undefined.

## Round 2 (PREREG_2 commit 77e1e3a), 2026-10-12 12:39:32
Model hashes verified against round2/MODELS_SHA256.txt (match). Select stage a5 (EB, EQ) frozen before any round-2 test read:
- frozen_a5.json sha256 5cbadb669818c241d703f3f5a9b3517dffdd9dad11e0fc6cec1134f0337d73cc
- REPAIR_PREREG_2.md sha256 64ccaba1c17dedafd308004aa859a3fda360178e3c203bbdebe8f803f81b5cef
- round2/experts.py 12dd347abade46ed64113215791ab74dfc81f55f3877e253559cce2ee64a3ada
- round2/experts_test.py 26df880c82a01900e30def7aac1c2ffc331dbdd31d07c7191500d7b475f1c0dd
- round2/a4.py 91950a0220997773d6f443914400208b5cddfc41519f453632b3567d2ed3fdfc
- round2/a5.py 85ad8669e6ea83d480319057b81e9e1d3c38a89439afcbfad6262581adef771c
- round2/gate_e0.py 900075e0a7625984bc8284fccfb8a8b54651b13713b7380131326a1c012c6310
- headline variants: EB M3, EQ M3 (pooled SEL log ratio EB -0.0024, EQ -0.0233)
- frozen_a4.json sha256 2ddd57e59760cdb1d1716616a16d30c299fe48d568f9061ac231b6bf916d1c15 (headline DA)
- DEVIATION (decided before any test accuracy was computed): experts_test.py asserts alignment of fresh Chronos-Bolt test forecasts with the stored test Bolt forecast. They differ by bf16 noise (median about 0.001 to 0.003 in the scaled domain, max 0.0107 on CESNET h1, 0.061 on NAB AWS-CPU h1; validation reproduces to 0.0). Tolerance changed from 1e-3 absolute to 5 % of std(y); residual experts EB and EB0 add their learned residual to the STORED test Bolt forecast, as on validation. experts_test.py sha256 now 6a3a680f0eca5d4e6b4d1b744557168986f75f06c9114dd1142b158ae12b29c5.

## Round 2 outcomes (test stages 2026-10-12)
- a5 E-Q (M3): 3 cells win after Holm(50) with >=0.5 % (GEANT h1,h3,h6), 0 losses, pooled gain vs Bolt +3.03 % CI [1.65, 4.53], gain over best control +2.94 % CI [1.41, 4.58] (two-stage [-0.27, 6.46]); rule needs 5 cells -> NOT MET. E-B (M3): 1 win, pooled +0.44 % CI [-0.17, 1.00] -> NOT MET.
- a4 D-A (headline): FA<=5 % on 3/5, shift-specific on 3/5, above permuted clock 3/5 -> NOT MET (needs 4). D-B feasible on 14/25 validation streams, detects nothing on NAB.

## Round 3 (PREREG_3 commit c4f0a4d), 2026-10-12 14:00:04
Series hashes verified against pool_manifest.json (no mismatch); code hashes match the prereg table. Stage A (train-only, validation forecasts) ran on all 100 cells; frozen before any stage B read:
- frozen_r3.json sha256 66ca55e4a7e15ac5ead96370f4f85da8041d75b2c9a35c569c2996a96015541b
- REPAIR_PREREG_3.md sha256 97fe25ccc956dfd7833a17f2a7c5d35ebb77bc8d23c02123bf0e3ce9304fd61c
- r3_models (100 files) sha256 of sorted per-file list: 83026cc4dc9e4f83444c505d86485ccb5183884f2975caa7bb5fa8dfe3d3ecb2
- r3_val (100 files) sha256 of sorted per-file list: 8dac4fe5c66cba41fb4ea66106d5622fe9264a1ee468afff3cbece880aef803b
- frozen_r3_monitor.json sha256 73c9b851a851af36663dc8c2823ef632f1a9e845a2eec56070bb8d0d5bc08ba9 (12 traces)

## Round 3 outcomes (test stages 2026-10-12)
- r3_eval (E-Q mixture, 100 fresh cells): 11 wins after Holm(100) (K = 8 required), 0 significant losses, pooled gain +0.499 % (cell CI -0.06, +1.01; series-clustered -0.70, +1.37), over best control +0.62 % (cell CI +0.01, +1.21; series-clustered -0.71, +1.79) -> NOT MET (conditions 3 and 4). Per source: CESNET 8 wins mean +1.33 %, Abilene 3 wins mean -0.14 %, GEANT 0 wins mean +0.03 %.
- r3_monitor (RF, 12 traces, 10 required): FA <= 5 % on 11, shift-specific 12, above permuted clock 9 -> NOT MET (condition 3). ST 4/8/4, AD 12/4/7.

## Round 4 (PREREG_4 commit 2cf0a6d), 2026-10-12 16:10:27
Stage A (train-only, validation forecasts) ran on all 302 cells (300 accuracy, 2 monitor-only at h = 6; no missing file). Select stages ran on validation arrays only; frozen before any stage B read:
- frozen_r4.json sha256 cbd12263b2952591b31c2a64629100154e30203e8c3ba6944d5606bc03442f38 (guard G2_0.25, 300 cells)
- frozen_r4_monitor.json sha256 893e0c542d3ff61a9f461f044de3e25cbf5e3fcfb90b0f754aef85d44f7aa221 (12 traces)
- frozen_r4_guard.json sha256 566443299bcb80eed5df57d7f9a7afa9cb96b3d07c1185b9ef4a59a9c161a4d8
- r4_val (302 files) sha256 of sorted per-file list: c17b79fec10fe01401ef351bb1354d2fc60fe00870fb348de6b5c25e0ec50eec
- r4_models (302 files) sha256 of sorted per-file list: 026734f35c6c1f0a0aa75ea6df478a3f689545f836febe18c5e8898f59ac58ad
- REPAIR_PREREG_4.md sha256 5708f9dec599af44d726d29594e478a19a2af156d24ba5653ec36f2328d001ff
- Commands: nice -n 19 taskset -c 8-11, OMP_NUM_THREADS=2, REPAIR_STAGE=test for the test stages.

## Round 4 outcomes (test stages 2026-10-12, REPAIR_STAGE=test)
- r4_eval (guarded E-Q mixture G2_0.25, 300 fresh cells, primary minimum 0.219 %, K = 24): 66 wins after Holm(300) at the primary minimum (50 at 0.5 %), 0 significant losses, pooled gain +0.604 % (cell CI 0.52, 0.69; series-clustered 0.46, 0.77), median cell gain 0.35 %, gain over best control +0.142 % (cell CI 0.06, 0.22; series-clustered -0.0024, 0.288) -> condition 4 fails under the series-clustered bootstrap, primary NOT MET, secondary 0.5 % NOT MET (same condition). Per source: Abilene 21/50 mean +0.99 %, CESNET 22/150 +0.28 %, GEANT 23/100 +0.90 %. Better 265, worse 15, identical 20 cells.
- r4_monitor (RF, 12 traces, 10 required): FA <= 5 % on 9, shift-specific 11, above placebo and shift-specific 8 (above placebo alone 9) -> NOT MET. ST 3/5/5, AD 12/6/3.
- Exploratory after scoring (labelled as such in paper and thesis): mixture vs each control alone +0.635 % (series CI 0.37, 0.95) and +0.429 % (0.30, 0.57); per-cell better control gains +0.46 % on Bolt; 24 of 60 series have negative net gain; series sd 0.54 %; N about 130 series for power 0.85.
- Paper A v2: sec:repair + tables/repair_round.tex + CHANGES_V2.md, 35 pp, 0 errors, 0 undefined. Thesis: 09b_onarim-turu.tex (+ tab:onarim-dort) and 11_bolum-ozeti.tex, 385 pp, 0 errors, 0 undefined.

## Round 5 (PREREG_5 commit 6f617a4), 2026-10-12 19:07:10
Stage A (train-only, validation forecasts) ran on all 584 cells (520 accuracy, 64 monitor at h = 6). Select stages ran on validation arrays only; frozen before any stage B read:
- frozen_r5.json sha256 c83f263d4464b9aa0099e2f8f658bdc49b4f67801e81c04e3de7718bb6a1b877 (guard G2_0.25, 520 cells)
- frozen_r5_monitor.json sha256 610c1059bfdadad0b35defc72c0c9e03fb5d21e81e811c07a55967b4958eac2e (64 traces)
- frozen_r4_guard.json sha256 566443299bcb80eed5df57d7f9a7afa9cb96b3d07c1185b9ef4a59a9c161a4d8
- r5_val (584 files) sha256 of the sorted per-file sha256sum list: c2efb168da2a0598c832c38c7ef94d8eb23067fcad72094c410a3ee1e8ac7e27
- r5_models (584 files) sha256 of the sorted per-file sha256sum list: eb17e9f21b237394c3f6f16d243cff0c5db1bc42225fbd2f722aa78836c8deaa
- REPAIR_PREREG_5.md sha256 2856d807b7402150022391741601c0f6d06d7b0c8e730530dacf6b74e14a39e3; RECIPE5_FROZEN.json sha256 b78bbedfa3e2dac67735b34220db08c5a6c74bd9b0eaacb57f34d974110c2f5c
- Commands: systemd-run --user --unit=paper-a-r5 (nice 19, CPUAffinity 8-11), run_r5.sh, OMP_NUM_THREADS=2, two workers, REPAIR_STAGE=test only for stage B and the test stages.

## Round 6 (REPAIR_PREREG_6), first select run, SUPERSEDED 2026-10-12 20:40
The first select run (STB with a fixed K = 0.5) is superseded by the coordinator decision to give STB the same K grid {0.5, 1, 2} and fallback rule as V1 to V3. It was frozen but never committed and no stage B existed. Superseded hashes: frozen_r6.json 96e778c4d3c7db445a0bbe737c4c7ed433b7ee52b7ae6372d1f1710167726635, RECIPE6_FROZEN.json abd53bac23eb1013aaea4c31622da9dfaa414055b015e1ceb6588638b3db1b72, r6_lib.py 2915d61bf7ee64fcd225b05fdb1bd8f26d74dee951b113c34dda163d2a73e6d3. Stage A (r6_val, r6_models) is unchanged by the edit.

## Round 6 (REPAIR_PREREG_6), 2026-10-12 20:56:38 (second select run: STB with K grid {0.5, 1, 2})
Stage A (train-only, validation forecasts) ran on all 38 monitor cells at h = 6 (P2 and the P3 GEANT stratum). The select stage ran on validation arrays only (the r6_test folder is guarded); frozen before any stage B read:
- frozen_r6.json sha256 f170954fdaa3400460798ed094e846702ffacfd5e82b40ecfda60421b0e114f5 (38 traces, arms R5 V1 V2 V3 STB ADB)
- pool6_manifest.json sha256 46757dcba4b7e748972f3adc5b22840436cb721e64ecb1c262e6763f3ac1db86; reserved_round6_M.json sha256 11d6a77079252b5cf7bb4a6f821afbeb43ca7ce60fc6e070db4f45e65d19a314
- r6_val (38 files) sha256 of the sorted per-file sha256sum list: 98e999867997c414114fc3f9fd6f68be9eb8476ff9504cfce9fffc841f2d39ee
- r6_models (38 files) sha256 of the sorted per-file sha256sum list: 3c8e3c0656e90c51db97ffa05bc17dcf034cdc66604e8f6b254007e01853ec11
- RECIPE6_FROZEN.json sha256 453c6748671209f3975a9cd4c8921aa4f5487c4510161655b6984030d2dc4a16 (final, regenerated by freeze6.py after the select run, the power scenarios, the dry run and the addition of classify6_r6.py; the select-time value was 60ddfd0f65ecc71ebd42e9b7260cb4acec4e69db6a7e040148c4e87936ee431e)
- Commands: systemd-run --user --unit=paper-a-r6 (nice 19, CPUAffinity 8-11), run_r6.sh, OMP_NUM_THREADS=2, two workers, REPAIR_STAGE=test only for stage B and the test stage.
- r6_lib.py sha256 32a0a541ccabff7ec8b66550b75eb3f472cfb8157a1a66ddce38f7725f8fb0d7 (select_stb chooses K on the V1 to V3 grid by validation detection, then lower null rate; fallback K = 2, grid maximum, infeasible), r6_monitor.py sha256 581a2f4b1c070cd2a2c302968c17a5a5b7eaf59be63e11508ba5b1ec049a910f (select_stb called with the validation injections).
- Feasibility after the rerun, P2 (20) / GEANT (18): R5 19 / 17, V1 13 / 16, V2 13 / 17, V3 14 / 18, STB 20 / 18 (was 3 / 18), ADB 0 / 7. STB level: 1 of 20 P2 traces at the grid maximum of 60 (inst152), K = 0.5 on 2, 1.0 on 15, 2.0 on 3 of the 20.
- Dry run repeated after the edit: round6/dryrun_log.txt ends with "dry-run ok; test windows accessed: 0" (sha256 6813693df0360c45a17f7a9a7906826c1fbc784ac9f0c29f692f76d64f8c3d05). r6_test is absent.
- Select command: UNTIL=select nice -n 19 taskset -c 12-15 bash run_r6.sh. Power scenarios: power6.py wrote power6_scenarios.json (sha256 72cb2a4c8c17e5b7ef2748f17ab43eb8e60bfceb113f0b6ccd5a276e539ba5a9).

## Erratum (2026-10-12, after round 6): count of monitor attempts
REPAIR_PREREG_6.md section 0 (line 7: "the third registered attempt at the monitor claim") and section 9 (lines 186 and 188: "round 6 is the third monitor attempt", "three registered attempts") counted the attempts only from round 4. The paper's own Table 1 (tables/repair_round.tex) lists four pre-registered tests of a reference-free monitor (registrations 3, 4, 5 and 6) and six monitor rules in all (registration 1 shift monitor, registration 2 drift-adaptive monitor, registrations 3 to 6). The paper and the thesis use the full count: "four pre-registered tests of a reference-free monitor, among six monitor rules in all, missed their rules". The frozen REPAIR_PREREG_6.md is not edited; this line is the correction of record.

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
