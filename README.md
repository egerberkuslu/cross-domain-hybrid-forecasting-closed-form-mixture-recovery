# CHA-Hybrid: Cross-Domain Hybrid Forecasting with Closed-Form Mixture Recovery

Code, configurations, and provenance for the paper:

> **Cross-Domain Hybrid Forecasting with Closed-Form Mixture Recovery for Networked AI Workloads**
> Ege Erberk Uslu, Orhan Dağdeviren — Ege University, İzmir, Türkiye

CHA-S fuses a classical decomposition expert (STL + Theta trend + LSTM residual)
with a frozen foundation expert (Chronos-Bolt) through a single per-horizon
scalar mixture weight α_h, recovered in **closed form** from three
Bates–Granger validation evaluations at O(1) cost. The same weight doubles as a
zero-cost foundation-reliability monitor.

## Repository layout

```
src/               modeling code (data loaders, models, training, evaluation)
pipeline/          experiment drivers and table/figure builders (expA_*.py, build_*.py)
scripts/           utility scripts
config/            experiment configurations
tests/             unit tests
run_all.py         top-level entry point
requirements.txt   pinned Python environment (Python 3.12, PyTorch 2.x, CUDA 12.x)
provenance_runlog.jsonl   phase-level execution log: every reported number traces
                          back to the seed, config digest, and hardware that produced it
```

## Datasets (all public)

| Dataset | Domain | Source |
|---|---|---|
| CESNET-TimeSeries24 | backbone link | <https://zenodo.org/records/13382427> |
| Abilene, GEANT | backbone link | TOTEM project <https://totem.run.montefiore.uliege.be/> |
| NAB AWS-CPU, NAB Twitter | host / application | <https://github.com/numenta/NAB> |
| BurstGPT | LLM serving | <https://github.com/HPMLL/BurstGPT> |
| AzureLLMInferenceTrace-2024 | LLM serving | <https://github.com/Azure/AzurePublicDataset> |
| Alibaba PAI (MLaaS-in-the-Wild) | GPU-cluster jobs | <https://github.com/alibaba/clusterdata> |

Loaders in `src/data_loaders/` reproduce the exact chronological
70/15/15 train/validation/test splits reported in the paper.

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py            # full grid; ~3 h on a single RTX 3060 (12 GB)
```

Seeds: `{7, 42, 123, 2024, 31337}` for the five-seed network cells,
`{42, 123, 2024}` for the three-seed AI-inference traces.

## Large artifacts

Trained checkpoints (~89 MB) and per-cell, per-window prediction tensors
(~229 MB, `.npz`) exceed the git budget and are distributed as release
assets — see the **Releases** page of this repository.

## Citation

The paper is under review; a BibTeX entry will be added on acceptance.

## License

MIT — see `LICENSE`.
