"""Small runtime helpers for optional PyTorch distributed training."""

from __future__ import annotations

import os
from typing import TypeVar

import torch
import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel


# Type variable for a PyTorch model type, used for type hinting in unwrap_model.
ModelT = TypeVar("ModelT", bound=torch.nn.Module)


# Helper to read integer environment variables with a default.
def _env_int(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(
                    f"Expected environment variable {name} to be an integer, got {value!r}"
                ) from exc
    return default


def _copy_slurm_environment() -> None:
    slurm_mapping = {
        "RANK": "SLURM_PROCID",
        "LOCAL_RANK": "SLURM_LOCALID",
        "WORLD_SIZE": "SLURM_NTASKS",
    }
    for target, source in slurm_mapping.items():
        if target not in os.environ and source in os.environ:
            os.environ[target] = os.environ[source]


# Distributed training helper, global rank
def rank() -> int:
    """Return the global process rank, defaulting to zero."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return _env_int("RANK", "SLURM_PROCID", default=0)


# Distributed training helper, local rank
def local_rank() -> int:
    """Return the node-local process rank, defaulting to zero."""
    return _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)


# Distributed training helper, world size
def world_size() -> int:
    """Return the number of participating processes, defaulting to one."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1)

# Distributed training helper, check if distributed
def is_distributed() -> bool:
    """Return whether the runtime represents a multi-process job."""
    return world_size() > 1


# Distributed training helper, check if main process
def is_main_process() -> bool:
    """Return whether this is the process responsible for shared side effects."""
    return rank() == 0


# Distributed training helper, initialize process group
def initialize(
    *,
    enabled: str | bool = "auto",
    backend: str | None = None,
) -> None:
    """Initialize a process group when a multi-process launcher is detected."""
    _copy_slurm_environment()

    if isinstance(enabled, str):
        enabled = enabled.lower()
        if enabled not in {"auto", "true", "false"}:
            raise ValueError("enabled must be 'auto', true, or false")
    elif not isinstance(enabled, bool):
        raise TypeError("enabled must be 'auto', true, or false")

    launched = world_size() > 1
    explicitly_disabled = enabled is False or enabled == "false"
    explicitly_enabled = enabled is True or enabled == "true"

    if explicitly_disabled:
        if launched:
            raise RuntimeError(
                "A multi-process launcher was detected, but distributed training is disabled"
            )
        return

    if not launched:
        if explicitly_enabled:
            raise RuntimeError(
                "Distributed training is enabled, but no multi-process launcher was detected"
            )
        return

    if dist.is_available() and dist.is_initialized():
        return
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    selected_backend = (
        backend or ("nccl" if torch.cuda.is_available() else "gloo")
    ).lower()
    if selected_backend == "nccl":
        if not dist.is_nccl_available():
            raise RuntimeError("The nccl distributed backend is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("The nccl distributed backend requires CUDA/ROCm devices")
        torch.cuda.set_device(local_rank())

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=selected_backend, init_method="env://")


# Distributed training helper, synchronize processes with a barrier
def barrier() -> None:
    """Synchronize initialized distributed processes; otherwise do nothing."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


# Distributed training helper, cleanup process group
def cleanup() -> None:
    """Destroy the current process group when one has been initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# Distributed training helper, unwrap model, to get model from DDP or DP wrapper
def unwrap_model(model: ModelT) -> ModelT:
    """Return the underlying model from a PyTorch parallel wrapper."""
    while isinstance(model, (DistributedDataParallel, DataParallel)):
        model = model.module  # type: ignore[assignment]
    return model  # type: ignore[return-value]


# Collect all public symbols for the module, for use with `from distributed import *`.
__all__ = [
    "barrier",
    "cleanup",
    "initialize",
    "is_distributed",
    "is_main_process",
    "local_rank",
    "rank",
    "unwrap_model",
    "world_size",
]
