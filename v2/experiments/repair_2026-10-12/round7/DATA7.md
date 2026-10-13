# Round 7 data plan: fresh drifting traces for the Paper A monitor

Written 2026-10-12 by a round 7 design worker. No forecast was computed and no test value of any new series was read. Probes are under `round7/data_probe/` (9.4 MB in total). The existing pool is exhausted (round 6 used every admissible unallocated CESNET and Abilene series). Following Ege's rule, that is treated as a data constraint to solve.

## (a) A never-used source

### a1. CESNET-TimeSeries24, finer aggregation levels (recommended primary)

- Source: Zenodo record 10.5281/zenodo.13382427, licence CC BY 4.0, no access gate `[web: https://zenodo.org/api/records/13382427]`. Files: `institution_subnets.tar.gz` 774,004,934 B (548 subnets), `ip_addresses_sample.tar.gz` 170,864,480 B (1,000 IP addresses), `ids_relationship.csv` 3,824,315 B (IP to subnet to institution). `ip_addresses_full.tar.gz` (40 GB) is not needed.
- Format, checked on 3 MB ranged heads (`data_probe/cesnet/*.head3MB.tar.gz`): it is the same as the institutions archive already in use, `<level>/agg_1_hour/<id>.csv` with `id_time, n_flows, n_packets, n_bytes, ...`. The calendar is the same (6,717 hourly ids). `fresh_cells._cesnet_hourly` therefore needs only a different archive path and folder prefix. The E-Q recipe, Bolt, the 0.70 / 0.15 / 0.15 split, h = 6 and the long design (test about 1,009 hours, about 1,004 windows) all stay unchanged.
- Never used: no round forecast any subnet or IP series. The caveat is that this is the same network and calendar as the CESNET institutions, and subnets and IPs are components of institutions. From `ids_relationship.csv`: 283 institutions and 548 subnets, of which 71 institutions have 2 or more subnets (340 subnets). Family sizes: 40 families of 2, 15 of 3, 5 of 4, ..., and one of 105 (institution 2). Five multi-subnet families (31 subnets) have a parent that was a round 5 or round 6 monitor trace whose test stream was revealed. The proposal excludes them. The remaining 309 subnets come from 66 families, or 161 with a cap of 3 per family. The other 208 subnets belong to single-subnet institutions and are probably near-copies of an already used institution series. They are excluded `[TODO: confirm by comparing series on download]`.
- Probe screen. This is the complete hourly files inside the 3 MB heads: raw n_bytes, approximate split by index, no outlier flag. It is tiny, so it is a planning number only.

| level | complete files in head | missing <= 5 % | of which train-to-validation shift >= 0.3 sd | admissible and drifting, Wilson 95 % |
|---|---|---|---|---|
| subnets | 11 | 10 | 5 (ids 101, 102, 103, 105, 106) | 5/11, [0.21, 0.72] |
| IP sample | 21 | 9 | 6 (ids 100610, 10125, 103, 10396, 10729, 11) | 6/21, [0.14, 0.50] |

- Planning estimate of admissible drifting traces, to be replaced by the registered counts-only screen: subnets (309 eligible) 65 to 222, or 34 to 116 with the family cap of 3. IP sample (1,000) 140 to 500. The low ends already exceed the n = 60 recommended in (c).
- Drift screen and split (identical in spirit to round 4 to 6): `fresh_cells.build_series` (outlier flag at 100 x median, gaps <= 24 h interpolated in time, longer gaps leave windows out, split 0.70 / 0.15 / 0.15, scaler on train), then `build_pool4.shift` (|mean val - mean train| / sd train >= 0.3, train and validation hours only), with at least 690 h = 6 test windows. Admissibility is the original `fresh_cells` rule (missing at most the aggregate share + 0.5 points, zero share <= 5 %, median > 0), plus the missing-data rule M7a of (b) and the degenerate-expert rule of the draft registration (validation only). Selection from the admissible drifting list is by the stride rule in id order, with at most 3 traces per institution family for subnets and at most 3 IPs per institution.
- Download: 945 MB in total, streamed one archive at a time; parsing one hourly CSV at a time keeps memory under 1 GB.

### a2. Telecom Italia Milan (recommended as an independent second stratum, needs two decisions)

- Source: Harvard Dataverse doi:10.7910/DVN/EGZHFV, "Telecommunications - SMS, Call, Internet - MI", 62 daily files `sms-call-internet-mi-2013-11-01.txt` to `...-2014-01-01.txt`, 20,804,803,507 B in total (300 to 373 MB per file) `[web: https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/EGZHFV]`.
- Licence: ODbL 1.0, with a mandatory attribution hyperlink "from BigDataChallenge contest" (terms of use in the dataset record). Derived databases must stay ODbL.
- Access: downloads are gated by a Dataverse guestbook (guestbookID 96). The anonymous API refused the download (`data_probe/milan/dataverse_guestbook_refusal.json`), so **no format sample could be taken**. Someone has to fill in the guestbook form with Ege's details. A worker must not do this.
- Format, from the data descriptor (Barlacchi et al. 2015, Scientific Data; `[web: https://pmc.ncbi.nlm.nih.gov/articles/PMC4622222/]`): square id, time interval start (ms, 10-minute slots), country code, SMS-in, SMS-out, call-in, call-out, internet activity (number of CDRs). All values are scaled by an undisclosed constant k. The fetched summary gave the Milan grid as "1,000 squares"; the 3.17 MB `milano-grid.geojson` (doi:10.7910/DVN/QJWLFU) suggests about 10,000 polygons `[TODO: verify the grid size on download]`.
- Split problem: at 1 h there are only 1,488 hours, a test split of about 223 windows, which is shorter than GEANT's 428. The long design is impossible at 1 h. At the native 10 minutes there are 8,928 steps: train to about 13 Dec, validation 14 to 23 Dec (about 1,339 steps), test 23 Dec to 1 Jan (about 1,334 windows at h = 6, one hour ahead). The long design fits.
- Recipe problem: E-Q is hard-wired to hourly data (`paper_a/round2/experts.py::features`: `np.timedelta64(1, "h")`, hour-of-day and day-of-week from an hourly step, 24 and 168 used as hours). A 10-minute port changes the frozen E-Q recipe, so it needs its own recipe hash and a disclosure. Bolt is unchanged.
- Content risk: the test split is the Christmas and New Year week. Natural regime shifts inside the no-fault test stream would raise p_dense and the control rate (the abilene_od110 pattern of DIAGNOSIS7 section 3). The per-trace pass rate is expected to be lower than on CESNET.
- Unit and screen: internet activity summed over country codes per macro-cell of 5 x 5 squares, which reduces zeros and spatial dependence (400 macro-cells if the grid is 100 x 100), with the same admissibility, missing-data and shift rules. Expected count: **not measurable without data** `[TODO: counts-only screen after the guestbook download]`. As a planning bracket only, 10 to 60 % of 400, that is 40 to 240.
- Download: about 20.8 GB, streamed one file at a time. Aggregate each file to macro-cells and delete it, so disk and memory stay small (chunked reading, under 2 GB).

### a3. MAWI / WIDE: rejected

The archive holds daily 15-minute packet traces of the WIDE transit link. The longest continuous traces run 48 to 96 h, which is far too short for a 0.70 / 0.15 / 0.15 hourly split with 690 test windows. The data are tcpdump traces with a research-only use clause, and the archive publishes no aggregated time series `[web: https://mawi.wide.ad.jp/mawi/]`.

## (b) Unallocated, inadmissible drifting leftovers under a registered missing-data rule

`data7_inventory.py` -> `data7_leftovers.csv`, `data7_leftovers_summary.json`. It reads the raw hourly caches. Values are used for train and validation hours only (the shift screen). For the test split only the positions of missing hours are counted.

Proposed rule **M7a**, fixed before any forecast. Gap filling stays the shipped rule: linear interpolation in time for gaps of at most 24 h, and longer gaps remove the affected windows. A series is admissible iff all of the following hold:

- the train missing share is at most 20 % and the longest train gap is at most 168 h (both decided on training data only);
- the zero share is at most 5 % and the median is positive;
- the test split has at least 690 (CESNET, Abilene) or 420 (GEANT) complete h = 6 windows, with at most one break, counted from positions only.

M7b relaxes this to a train missing share of at most 30 % and at most 3 breaks. M7c is the length rule alone.

| source | leftovers (never allocated, including round 6 candidates) | drifting (shift >= 0.3) | M7a | M7b | M7c |
|---|---|---|---|---|---|
| CESNET | 33 | 3 (ids 140, 264, 283) | 0 | 0 | 1 (264) |
| Abilene | 14 | 2 (OD 48, 84) | 0 | 0 | 0 |
| GEANT | 154 | 97 | 16 | 38 | 49 |

Which series become admissible, exactly:

- CESNET: none under M7a or M7b. Ids 140 and 283 have no test windows, because validation and test are 100 % missing. Id 264 has 695 windows, but 70 % of its test hours would be interpolated and its train missing share is 35 %. It passes only the length rule M7c and should not be admitted.
- Abilene OD 48 and 84: they are inadmissible because of the **zero** rule (zero share 11.3 % and 11.4 %), not the missing rule. They also have a 432-hour train gap, so M7a excludes them. Admitting them needs a relaxed zero rule, which is not recommended.
- GEANT, M7a (16): 1_3, 1_9, 3_3, 4_23, 5_2, 8_2, 9_2, 10_7, 15_21, 16_21, 17_8, 19_3, 19_11, 19_16, 20_11, 23_16. M7b (38) adds 1_10, 3_10, 3_11, 3_14, 3_18, 4_3, 5_10, 6_8, 6_11, 8_10, 8_23, 9_10, 10_11, 11_10, 12_10, 15_10, 17_10, 18_10, 19_10, 20_3, 21_10, 23_3 (lists in `data7_leftovers_summary.json`). Every GEANT trace runs the short design: about 423 test windows, and a validation stream (281 windows) shorter than burn-in plus cap. On that design V1 to V3 passed only 6, 8 and 9 of 18 in round 6, so GEANT cannot carry the success rule. **Recommendation:** at most a descriptive stratum under M7a (16 traces), never pooled.

Conclusion: the leftovers add no usable trace to the primary pool. A new source is necessary.

## (c) Pool and power

Per-trace pass probabilities, design inputs from `val7_*_summary.json` (Wilson 95 %). Round 6 actual V1 to V3: 15/20 = 0.75 [0.53, 0.89]. Revealed round 5 (the less contaminated input): W1 and W3 54/64 = 0.844 [0.74, 0.91], W4 56/64 = 0.875 [0.77, 0.94]. Pooled revealed r5 + r6: W1 0.821, W3 0.857, W4 0.881. On a new source, π is expected to sit lower than on the streams the variants were designed on.

Power (`power7.py` -> `power7.json`, exact binomial, Holm first step alpha / 3, traces independent):

| rule | n | k | π = 0.75 | 0.80 | 0.821 | 0.845 | 0.857 | 0.875 |
|---|---|---|---|---|---|---|---|---|
| R80 (X >= 0.8 n and binomial vs 0.5) | 20 | 16 | 0.415 | 0.630 | | | | |
| R80 | 40 | 32 | 0.300 | 0.593 | 0.720 | 0.843 | 0.892 | 0.945 |
| R80 | 60 | 48 | 0.232 | 0.576 | 0.731 | 0.871 | 0.921 | 0.968 |
| R80 | 80 | 64 | 0.184 | 0.566 | 0.744 | 0.894 | 0.941 | 0.981 |
| R60 (binomial vs 0.6) | 60 | 45 | 0.569 | 0.869 | 0.940 | 0.982 | 0.991 | 0.998 |
| R50 (binomial vs 0.5 alone) | 60 | 39 | 0.970 | 0.998 | 1.000 | 1.000 | 1.000 | 1.000 |

Two facts decide the pool. First, under R80 power at π = 0.80 is about 0.57 to 0.63 for every n, and at π = 0.75 (the round 6 rate) it falls as n grows. No pool reaches power 0.8 at the round 6 rate. A larger pool helps R80 only when π >= 0.84. Second, at the repaired-variant design input (π 0.845 to 0.875), n = 60 gives R80 power 0.87 to 0.97, while n = 20 gives about 0.83 at 0.85 and much less below that.

**Recommended pool:** n = 60 primary traces from the CESNET finer levels (a1). Take 30 subnets (stride over the admissible drifting list, at most 3 per family, parents not round 5 or 6 monitor traces) and 30 IP-sample series (stride, at most 3 per institution, parent institution not a round 5 or 6 monitor trace). If one level has fewer than 30, fill from the other. Milan (a2) runs as a separately reported second stratum of up to 30 macro-cells: never pooled into the primary rule, its own Holm family, and only if Ege approves both the guestbook download and the 10-minute E-Q port. GEANT M7a goes in as a descriptive stratum or is dropped.

Non-independence: subnets and IPs share one network and calendar, and IPs sit inside institutions. With a design effect of 1.5 (n_eff = 40), R80 power falls to 0.84 at π = 0.845 and 0.59 at π = 0.80 (`power7.json`, `design_effect_1.5`). The family caps limit this, and per-family results are reported.

Rule choice (coordinator decision, section 6 of the draft): keeping R80 gives power 0.87 to 0.97 at the repaired design input, but only 0.58 at π = 0.80. R60 (H0: works on at most 60 % of traces) gives 0.87 at π = 0.80, but it is a weaker claim and would be a rule change made after six failures. The draft keeps R80 and reports the power range honestly.
