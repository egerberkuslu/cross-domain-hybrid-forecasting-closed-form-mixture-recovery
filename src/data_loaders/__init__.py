"""Per-dataset loaders that return a common pandas DataFrame format.

All loaders inherit :class:`BaseLoader` and yield a frame indexed by a
strictly monotonic-increasing tz-naive ``DatetimeIndex`` with a single
column ``value`` containing the aggregate total network traffic per bin.
"""

from .base import BaseLoader, DatasetSpec
from .cesnet import CesnetLoader
from .abilene import AbileneLoader
from .geant import GeantLoader
from .nab import NABAWSCpuLoader, NABTwitterAAPLLoader
from .burstgpt import BurstGPTLoader
from .azure_llm_2024 import AzureLLM2024Loader
from .azure_llm_2024_5m import AzureLLM2024_5mLoader
from .alibaba_pai import AlibabaPAILoader


def build_loader(name: str, cfg) -> BaseLoader:
    """Build the loader for `name` from the project config."""
    spec = cfg.datasets[name]
    raw_dir = cfg.resolve(spec.raw_dir)
    processed_path = cfg.resolve(spec.processed_file)
    resample = spec.resample_to
    if name == "cesnet":
        return CesnetLoader(raw_dir, processed_path, resample_to=resample)
    if name == "abilene":
        return AbileneLoader(raw_dir, processed_path, resample_to=resample)
    if name == "geant":
        return GeantLoader(raw_dir, processed_path, resample_to=resample)
    if name == "nab_aws_cpu":
        return NABAWSCpuLoader(raw_dir, processed_path, resample_to=resample)
    if name == "nab_twitter":
        return NABTwitterAAPLLoader(raw_dir, processed_path, resample_to=resample)
    if name == "burstgpt":
        return BurstGPTLoader(raw_dir, processed_path, resample_to=resample)
    if name == "azure_llm_2024":
        return AzureLLM2024Loader(raw_dir, processed_path, resample_to=resample)
    if name == "azure_llm_2024_5m":
        return AzureLLM2024_5mLoader(raw_dir, processed_path, resample_to=resample)
    if name == "alibaba_pai":
        return AlibabaPAILoader(raw_dir, processed_path, resample_to=resample)
    raise KeyError(f"Unknown dataset: {name}")


__all__ = [
    "BaseLoader",
    "DatasetSpec",
    "CesnetLoader",
    "AbileneLoader",
    "GeantLoader",
    "NABAWSCpuLoader",
    "NABTwitterAAPLLoader",
    "BurstGPTLoader",
    "AzureLLM2024Loader",
    "AzureLLM2024_5mLoader",
    "AlibabaPAILoader",
    "build_loader",
]
