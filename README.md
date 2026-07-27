# CHA-S — Cross-Domain Hybrid Forecasting with Closed-Form Mixture Recovery for Networked AI Workloads

Reference implementation, experiment drivers, configuration, and execution
provenance for the manuscript

> **Cross-Domain Hybrid Forecasting with Closed-Form Mixture Recovery for Networked AI Workloads**
> Ege Erberk Uslu (ORCID 0000-0001-9119-8574), Orhan Dağdeviren (ORCID 0000-0001-8789-5086)
> Department of Computer Engineering, Ege University, İzmir, Türkiye
> Submitted to *PeerJ Computer Science*.

## Description

CHA-S (Chronos-Hybrid-Adaptive Scalar) is a two-expert forecaster for
non-stationary network and AI-inference workloads. A decomposition expert
factors the series with STL, extrapolates the trend with the Theta model,
replicates the seasonal component with seasonal-naive, and models the residual
with a single-layer LSTM. A frozen Chronos-Bolt foundation model, used
zero-shot and never fine-tuned, serves as the global expert. The two forecasts
are fused by a convex combination with one scalar weight per horizon,
`alpha_h`, so the forecast at horizon `h` is
`alpha_h * decomposition + (1 - alpha_h) * foundation`.

The point of the method is how that weight is obtained. Validation mean
squared error is quadratic in `alpha`, so three evaluations at `alpha = 0`,
`0.5`, and `1` determine all three coefficients of the quadratic and yield the
Bates-Granger optimum in closed form at O(1) cost, with no gating network and
no grid search. Because the weight is one interpretable number recomputed
cheaply, it doubles as a zero-cost reliability monitor on the frozen
foundation expert: it rises when that expert degrades, and thresholding it
online flags injected distribution shifts.

This repository contains everything needed to reproduce that pipeline:
dataset loaders, preprocessing, the model zoo used for baselines, the
proposed CHA-Hybrid family, the evaluation and statistical-testing code
(Diebold-Mariano with the Harvey-Leybourne-Newbold correction, sign-flip
permutation tests, bootstrap TOST equivalence), the degradation and
alpha-monitor experiments, and the table and figure builders. It does **not**
contain the raw datasets, the trained checkpoints, or the per-window
prediction tensors; see *Dataset Information* and *Large artifacts* below.
`provenance_runlog.jsonl` records every experiment phase that produced a
reported number.

## Dataset Information

**No dataset files are distributed with this repository.** All nine traces are
public and are fetched by the loaders in `src/data_loaders/` on first use into
`data/raw/<dataset>/`, then cached as Parquet under `data/processed/`. The
`data/` tree is created at run time and is not tracked by git.

| # | Trace | Domain | Native Δt | Analysis Δt | n_train / n_val / n_test | Source |
|---|---|---|---|---|---|---|
| 1 | CESNET-TimeSeries24 | NREN backbone link | 10 min | 1 h | 4703 / 1008 / 1007 | Zenodo record 13382427, DOI 10.5281/zenodo.13382427 — <https://zenodo.org/records/13382427> |
| 2 | Abilene | backbone traffic matrix | 5 min | 1 h | 3259 / 698 / 699 | TOTEM project <https://totem.run.montefiore.uliege.be/>; the loader downloads the weekly `X??.gz` matrices from the Internet2/Abilene archive at <http://www.cs.utexas.edu/~yzhang/research/AbileneTM> |
| 3 | GEANT | backbone traffic matrix | 15 min | 1 h | 1994 / 427 / 428 | TOTEM anonymized traffic matrices, `traffic-matrices-anonymized-v2.tar.bz2` — <https://totem.run.montefiore.uliege.be/> |
| 4 | NAB AWS-CPU | host metric | 5 min | 1 h | 1054 / 226 / 225 | Numenta Anomaly Benchmark, `realKnownCause/cpu_utilization_asg_misconfiguration.csv` — <https://github.com/numenta/NAB> |
| 5 | NAB Twitter (@AAPL) | application event rate | 5 min | 1 h | 928 / 199 / 199 | NAB, `realTweets/Twitter_volume_AAPL.csv` — <https://github.com/numenta/NAB> |
| 6 | BurstGPT | LLM serving | request-level | 1 h | 2033 / 435 / 436 | <https://github.com/HPMLL/BurstGPT>; the loader pulls the two public CSVs from the mirror <https://huggingface.co/datasets/lzzmm/BurstGPT> |
| 7 | AzureLLMInferenceTrace-2024 | LLM serving | request-level | 1 h | 151 / 32 / 33 | Azure Public Dataset — <https://github.com/Azure/AzurePublicDataset> |
| 8 | AzureLLM-2024-5m | LLM serving | request-level | 5 min | 1814 / 389 / 389 | Same raw files as row 7, aggregated at 5-minute granularity |
| 9 | Alibaba PAI (MLaaS-in-the-Wild) | GPU-cluster job arrivals | event-level | 1 h | 1158 / 248 / 249 | <https://github.com/alibaba/clusterdata> (`cluster-trace-gpu-v2020`); the loader pulls `pai_job_table.tar.gz` from the Alibaba OSS mirror |

Rows 7 and 8 are two aggregations of the same raw Azure trace and are treated
as separate evaluation cells: the hourly aggregate falls below the training
sample floor of the LSTM residual head, and the paper reports that failure
explicitly, while the 5-minute aggregate fits cleanly.

Splits are chronological (70 / 15 / 15, no shuffling and no leakage), the
lookback window is 168 steps, and horizons are h ∈ {1, 3, 6, 12, 24}. All of
these are set in `config/config.yaml`.

If a download host is unreachable, place the expected raw file into
`data/raw/<dataset>/` by hand and re-run; each loader checks the local path
before downloading and raises a `FileNotFoundError` naming the missing file.

## Code Information

| Path | Contents |
|---|---|
| `run_all.py` | Top-level driver. Seven stages: `data`, `preprocess`, `baselines`, `proposed`, `grid`, `evaluate`, `outputs`. |
| `config/config.yaml` | Single source of truth: dataset specs, paths, seeds, split ratios, window length, horizons, model groups, hyperparameter grids, CHA-Hybrid settings, evaluation metrics. |
| `src/data_loaders/` | One loader per trace (`cesnet`, `abilene`, `geant`, `nab`, `burstgpt`, `azure_llm_2024`, `azure_llm_2024_5m`, `alibaba_pai`) over a shared `BaseLoader` that does download → parse → resample → outlier flagging → Parquet cache. `build_loader(name, cfg)` is the factory. |
| `src/preprocessing/` | Interpolation, chronological splitting, train-fit scaling, windowing (`pipeline.py` exposes `preprocess_all`). |
| `src/models/` | Model zoo behind one `registry.py` interface: naive, seasonal-naive, ARIMA, Holt-Winters, Theta, FARIMA; XGBoost, LSTM, GRU, TCN, N-BEATS, DLinear, PatchTST, NHiTS, TFT, TiDE, TSMixer; Chronos-T5, Chronos-Bolt, TimesFM, MOIRAI, TTM; and the proposed family `cha_hybrid`, `cha_hybrid_v2`, `cha_hybrid_v3` (CHA-S, Chronos-Bolt global expert), `cha_hybrid_v4` and `cha_hybrid_v4_fix` (learned `alpha(x)` MLP head). |
| `src/training/` | Training runner, hyperparameter search, grid execution. |
| `src/evaluation/` | Metrics (RMSE, MAE, MAPE, sMAPE, R²), Diebold-Mariano test with the HLN small-sample correction, ablations, aggregation, cost accounting, winner tables. |
| `src/utils/` | Config loader, logging, global seeding, device detection, IO, and `runlog.py` (the `PhaseTimer` provenance recorder). |
| `src/swarm/` | Auxiliary discrete-event edge-fleet simulator (fleet, policy, gossip, workload, metrics, CLI). It supports the edge-deployment discussion and companion work; **no number reported in this manuscript comes from it**. |
| `pipeline/` | 54 experiment drivers and builders. `phase1_data.py` … `phase7_figures.py` are the staged pipeline; `expA_*` are the Stage-A experiments (bootstrap CI, closed-form α, per-sample visualization, latency, sensitivity, cross-dataset transfer, stationarity); `expC_*` are the mixture and per-window analyses; `expR*` are the reviewer-driven experiments (paired and permutation tests, TOST equivalence, α-transfer, LSTM variance share, multi-metric, α-stretch degradation sweep, α-monitor detectors D1–D4, edge footprint); `build_*.py` emit LaTeX tables and figures. |
| `scripts/build_flat_submission.py` | Packages a flat, publisher-compliant LaTeX source bundle. |
| `tests/` | 48 pytest unit tests covering the `src/swarm` simulator (fleet, workload, simulator). |
| `provenance_runlog.jsonl` | 95 phase records. Each line carries the phase name, UTC start/end, wall-clock duration, hostname, git commit, dirty flag, and a full snapshot of `config/config.yaml` at execution time. |
| `requirements.txt`, `requirements-lock.txt` | Loose-bounded and exact environments (see *Requirements*). |

## Requirements

Python 3.10-3.12; the reported results were produced on Python 3.12.4 with
CUDA 12.8 on a single NVIDIA RTX 3060 (12 GB). A CUDA GPU is strongly
recommended for the foundation-model and deep baselines; the statistical
baselines and all post-processing run on CPU.

Main dependencies: NumPy, pandas, SciPy, scikit-learn, statsmodels,
matplotlib, seaborn; darts, statsforecast, pmdarima for the forecasting
baselines; XGBoost, PyTorch, torchmetrics, PyTorch Lightning, einops for the
ML/DL models; chronos-forecasting, transformers, accelerate, peft and
huggingface_hub for the foundation experts; PyYAML, tqdm, joblib, rich and
pyarrow for configuration, logging and Parquet IO.

Two dependency files are provided and they serve different purposes.
`requirements.txt` carries deliberately loose lower bounds and is the
convenient install. `requirements-lock.txt` pins the exact versions of the
environment that produced every number in the manuscript, verified
2026-07-21; **use the lock file to reproduce the paper.**

`pytest` is needed only to run the test suite and is not listed in either file.

## Usage Instructions

### Install

```bash
git clone https://github.com/egerberkuslu/cross-domain-hybrid-forecasting-closed-form-mixture-recovery.git
cd cross-domain-hybrid-forecasting-closed-form-mixture-recovery

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# exact paper environment (recommended)
pip install -r requirements-lock.txt
# or, for a convenience install with loose bounds
# pip install -r requirements.txt
```

### Run the pipeline

All commands are run from the repository root.

```bash
# whole pipeline, all seven stages
python run_all.py

# a single stage
python run_all.py --stage data        # download, parse, cache, profile all 9 traces
python run_all.py --stage preprocess  # interpolate, split 70/15/15, scale, window
python run_all.py --stage baselines   # smoke-test every registered model
python run_all.py --stage proposed    # CHA-Hybrid smoke test and verification
python run_all.py --stage grid        # dataset x model x horizon x seed grid (resumable)
python run_all.py --stage evaluate    # metrics, DM test, ablations, cost

# alternative config or a named log file
python run_all.py --config config/config.yaml --run-name my_run
```

Seeds are `{42, 123, 2024, 7, 31337}` as set in `config/config.yaml`; the
manuscript uses five seeds for the network cells and three for the
AI-inference traces.

A full grid run takes about two hours on a single RTX 3060 (the main grid
phases sum to 1.7 h in the shipped provenance log). Stages 3-6 delegate to
`pipeline/phase{3,4,5,6}_*.py`, and the phase-5 main runner is resumable, so
an interrupted grid can be restarted with the same command. It also accepts
filters directly:

```bash
python pipeline/phase5_main.py --datasets cesnet abilene --horizons 1 6 24
```

### Figures, tables, and the individual experiments

The `outputs` stage of `run_all.py` is a stub; figures and paper tables are
produced by their own drivers. Several of them write LaTeX snippets into
`paper_a/tables/` and `paper_a/figures/` (the manuscript directories), so
create those first:

```bash
mkdir -p paper_a/tables paper_a/figures

python pipeline/phase7_figures.py            # all paper figures + summary CSVs
python pipeline/build_paper_tables.py        # LaTeX tables from outputs/eval_v3/tables/*.csv

python pipeline/expA_1_2_closed_form_alpha.py  # closed-form Bates-Granger alpha vs. 21-point grid
python pipeline/expA_1_1_bootstrap_ci.py       # bootstrap confidence intervals
python pipeline/expR3_tost_equivalence.py      # bootstrap TOST equivalence vs. Chronos-Bolt
python pipeline/expR2_permutation_test.py      # paired sign-flip permutation test
python pipeline/phase6_dm_test_full.py         # Diebold-Mariano grid with HLN correction
python pipeline/expR5_alpha_stretch.py         # controlled foundation-degradation sweep
python pipeline/expR6_monitor_master.py        # alpha-monitor detectors D1-D4
python pipeline/expR7_edge_footprint.py        # single-core edge latency and memory footprint
python pipeline/expA_2_x_sensitivity.py        # STL period and model-size sensitivity
python pipeline/expA_3_1_cross_dataset.py      # cross-dataset transfer matrix
```

Results land under `outputs/` following the `paths` block of
`config/config.yaml`: `outputs/metrics/`, `outputs/predictions/`,
`outputs/figures/`, `outputs/hyperparameters/`, `outputs/logs/`, and
`outputs/eval_v3/tables/`.

### Tests

```bash
pip install pytest
python -m pytest tests/ -q
```

48 tests covering the auxiliary edge-fleet simulator in `src/swarm/`
(fleet management, workload generation, and the discrete-event simulator).

## Methodology

The pipeline mirrors the manuscript section by section.

**1. Data preparation** (`run_all.py --stage data`, then `--stage preprocess`).
Each loader downloads its raw trace, parses it to a single aggregate `value`
series at native frequency, resamples to the analysis frequency, flags
physically implausible outliers against a robust median-times-factor criterion
(recorded, not silently dropped), and caches Parquet. Preprocessing then
interpolates gaps in time, splits chronologically 70/15/15, fits the scaler on
the training split only, and builds 168-step lookback windows for every
horizon.

**2. Baselines** (`--stage baselines`). Every model in `src/models/registry.py`
is exercised under one interface so that statistical, machine-learning, deep,
and foundation baselines all see identical windows, splits, and scaling.

**3. The two experts** (`--stage proposed`). The decomposition expert applies
STL with period P = 24 by default, extrapolates the trend with Theta,
extrapolates the seasonal component by seasonal-naive replication of the last
observed period, and predicts the residual with a single-layer LSTM (64 hidden
units, dropout 0.1, early stopping on validation residual MSE). The global
expert is Chronos-Bolt (`amazon/chronos-bolt-small`) run zero-shot and frozen,
so the same checkpoint is used unchanged on every trace.

**4. Closed-form mixture weight** (`pipeline/expA_1_2_closed_form_alpha.py`).
Validation MSE of the convex mixture is quadratic in `alpha`, so evaluations
at `alpha = 0`, `1/2`, and `1` recover the decomposition variance, the global
variance, and their cross-covariance, and the Bates-Granger optimum follows in
closed form. The script computes this weight for every cell and compares it
against the exhaustive 21-point grid search that the models otherwise use,
which is the experiment behind the reported mean absolute deviation between
the two.

**5. Full grid** (`--stage grid`). Dataset x model x horizon x seed, resumable,
with per-run metrics and timings written to `outputs/metrics/`.

**6. Evaluation and inference** (`--stage evaluate`, plus the `expR*` drivers).
RMSE, MAE, MAPE, sMAPE and R² per cell; Diebold-Mariano with the HLN
small-sample correction; a paired sign-flip permutation test on per-window
outcomes; and a bootstrap TOST equivalence test with a 5 % margin against the
Chronos-Bolt foundation expert, which is what supports the equivalence claim
rather than a non-rejected superiority test.

**7. Degradation and monitoring** (`expR5_alpha_stretch.py`,
`expR6_monitor_master.py`). The foundation forecast is degraded with
calibrated additive noise across a gamma sweep, and the recovered weight is
tracked as it climbs toward the decomposition path, showing the mixture is
adaptive rather than a fixed wrapper. Online, the same weight feeds three
detectors, an absolute threshold, a CUSUM on the standardized weight, and
their conjunction, evaluated against a residual-CUSUM control on all five
network traces for detection lag and false-alarm rate.

**8. Cost** (`expA_1_4_latency.py`, `expR7_edge_footprint.py`). Train and
predict times per model, plus a batch-of-one streaming measurement pinned to a
single CPU core with peak resident memory, characterizing the edge-class
deployment envelope.

**9. Outputs** (`phase7_figures.py`, `build_paper_tables.py`). Figures and
LaTeX tables are regenerated from the CSVs written by the stages above.

## Reproducibility and provenance

Every stage that produced a reported number is wrapped in the `PhaseTimer`
context manager of `src/utils/runlog.py`, which appends one JSON record per
phase. `provenance_runlog.jsonl` is the accumulated log shipped with this
repository: 95 records, each carrying the phase name, UTC start and end,
elapsed wall-clock time, hostname, git commit and dirty flag, and a complete
snapshot of the configuration in force. Any table or figure in the manuscript
can therefore be traced back to the seed, configuration digest, and machine
that produced it.

## Large artifacts

Trained checkpoints (`cha-checkpoints.tar.gz`, ~76 MB), per-cell per-window
prediction tensors (`cha-predictions.tar.gz`, ~228 MB), and the aggregated
metric CSVs (`cha-metrics.tar.gz`, ~1 MB) exceed the git budget and are
published as assets of release **v1.0.0** on the Releases page of this
repository. The pipeline regenerates all of them from scratch if they are
absent.

## Citation

A machine-readable [`CITATION.cff`](CITATION.cff) is included, so GitHub's
"Cite this repository" button gives the same reference. If you use this
code, please cite the paper:

```bibtex
@article{uslu2026chas,
  author  = {Uslu, Ege Erberk and Da{\u{g}}deviren, Orhan},
  title   = {Cross-Domain Hybrid Forecasting with Closed-Form Mixture
             Recovery for Networked {AI} Workloads},
  year    = {2026},
  note    = {Submitted to PeerJ Computer Science},
  url     = {https://github.com/egerberkuslu/cross-domain-hybrid-forecasting-closed-form-mixture-recovery}
}
```

Please also cite the datasets you use:

- **CESNET-TimeSeries24** — Koumar J., Hynek K., Čejka T., Šiška P. *CESNET-TimeSeries24: Time Series Dataset for Network Traffic Anomaly Detection and Forecasting.* Scientific Data 12:338, 2025. Data: DOI 10.5281/zenodo.13382427.
- **Abilene / GEANT (TOTEM)** — Uhlig S., Quoitin B., Lepropre J., Balon S. *Providing Public Intradomain Traffic Matrices to the Research Community.* ACM SIGCOMM Computer Communication Review, 2006.
- **NAB (AWS-CPU, Twitter)** — Lavin A., Ahmad S. *Evaluating Real-Time Anomaly Detection Algorithms: The Numenta Anomaly Benchmark.* IEEE ICMLA, 2015.
- **BurstGPT** — Wang Y. et al. *BurstGPT: A Real-World Workload Dataset to Optimize LLM Serving Systems.* 2024.
- **AzureLLMInferenceTrace-2024** — Patel P. et al. *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA, 2024.
- **Alibaba PAI** — Weng Q. et al. *MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters.* USENIX NSDI, 2022.

The frozen global expert is Chronos-Bolt (Amazon, 2024-2025); please cite it
as well when reporting CHA-S results.

## License

MIT. See [`LICENSE`](LICENSE). Copyright (c) 2026 Ege Erberk Uslu.

The datasets are distributed by their respective owners under their own
licenses and terms of use; check each source before redistribution.

## Getting help, contributing, and contact

For questions, reproduction problems, or bug reports, please open an issue on
this repository; issues are the fastest channel and keep the answer public for
the next reader.

Issues and pull requests are welcome, particularly additional baselines: the
harness exposes a drop-in model interface, so a new forecaster only has to
implement the `src/models/base.py` contract and register itself in
`src/models/registry.py` to be evaluated under conditions identical to every
result in the paper. When reporting a reproduction problem, please include the
relevant record from `provenance_runlog.jsonl`, the output of
`pip freeze`, and the failing command.

Correspondence: Ege Erberk Uslu, Department of Computer Engineering, Ege
University, İzmir, Türkiye.
