"""Detect compute device (CUDA / CPU) and log details once.

Phase 3 conditionally fine-tunes Chronos / TimesFM on the GPU; if no GPU
is detected we skip fine-tuning and log a clear reason. This module
centralises that decision so the rest of the codebase just calls
:func:`detect_device`.
"""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    device: str               # "cuda" | "cpu"
    cuda_available: bool
    n_gpus: int
    gpu_name: str | None
    gpu_mem_gb: float | None
    cuda_version: str | None
    torch_version: str | None
    platform: str

    def can_finetune_foundation_models(self) -> bool:
        """Whether we have enough GPU to fine-tune small foundation models."""
        return self.cuda_available and (self.gpu_mem_gb or 0) >= 6.0


def detect_device() -> DeviceInfo:
    if torch is None:
        return DeviceInfo("cpu", False, 0, None, None, None, None, platform.platform())
    cuda_avail = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_avail else 0
    gpu_name: str | None = None
    gpu_mem_gb: float | None = None
    if cuda_avail and n_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return DeviceInfo(
        device="cuda" if cuda_avail else "cpu",
        cuda_available=cuda_avail,
        n_gpus=n_gpus,
        gpu_name=gpu_name,
        gpu_mem_gb=gpu_mem_gb,
        cuda_version=torch.version.cuda,
        torch_version=torch.__version__,
        platform=platform.platform(),
    )


def log_device_info(info: DeviceInfo | None = None) -> DeviceInfo:
    info = info or detect_device()
    logger.info("=" * 72)
    logger.info("Compute environment:")
    logger.info("  platform        : %s", info.platform)
    logger.info("  torch version   : %s", info.torch_version)
    logger.info("  cuda version    : %s", info.cuda_version)
    logger.info("  cuda available  : %s", info.cuda_available)
    logger.info("  n_gpus          : %s", info.n_gpus)
    logger.info("  gpu_name        : %s", info.gpu_name)
    logger.info("  gpu_mem_gb      : %s", info.gpu_mem_gb)
    logger.info("  selected device : %s", info.device)
    logger.info("  finetune ok     : %s", info.can_finetune_foundation_models())
    logger.info("=" * 72)
    return info
