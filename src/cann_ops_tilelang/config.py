"""Runtime configuration for Ascend operators.

This module intentionally performs no device initialization at import time.
"""

from __future__ import annotations

import os
from functools import cache, lru_cache

import torch

_ASCEND_DEVICE_NODE = '/dev/davinci_manager'
_NUM_VECTOR_CORES_ENV = 'OPS_TILELANG_NUM_VECTOR_CORES'
_PRINT_KERNEL_SOURCE_ENV = 'OPS_TILELANG_PRINT_KERNEL_SOURCE'


@lru_cache(maxsize=1)
def is_ascend_available() -> bool:
    """Return whether the process appears to run in an Ascend environment."""

    return os.path.exists(_ASCEND_DEVICE_NODE) and hasattr(torch, 'npu')


def require_ascend() -> None:
    """Raise a clear error when an Ascend runtime is unavailable."""

    if not is_ascend_available():
        raise RuntimeError(
            'ops-tilelang requires an Ascend NPU runtime. Install a CANN-compatible '
            'PyTorch NPU build and run on a host exposing /dev/davinci_manager.'
        )


def get_device() -> torch.device:
    """Return the current Ascend device without silently selecting another backend."""

    require_ascend()
    return torch.device('npu', torch.npu.current_device())


@cache
def get_num_ai_cores(device_index: int | None = None) -> int:
    """Return the number of Ascend AI/cube cores on the selected device."""

    require_ascend()
    index = torch.npu.current_device() if device_index is None else device_index
    properties = torch.npu.get_device_properties(index)
    cube_cores = getattr(properties, 'cube_core_num', None)
    if cube_cores is None:
        raise RuntimeError('The current torch_npu build does not expose cube_core_num.')
    return int(cube_cores)


@cache
def get_num_vector_cores(device_index: int | None = None) -> int:
    """Return the number of vector cores available to a TileLang kernel.

    ``OPS_TILELANG_NUM_VECTOR_CORES`` can cap the value for debugging or
    reproducible performance experiments.
    """

    detected = get_num_ai_cores(device_index) * 2

    override = os.getenv(_NUM_VECTOR_CORES_ENV)
    if override is None:
        return detected
    requested = int(override)
    if requested <= 0 or requested > detected:
        raise ValueError(f'{_NUM_VECTOR_CORES_ENV} must be in [1, {detected}], got {requested}.')
    return requested


def should_print_kernel_source() -> bool:
    """Return whether generated TileLang source should be printed."""

    return os.getenv(_PRINT_KERNEL_SOURCE_ENV, '0').lower() in {'1', 'true', 'yes', 'on'}


def clear_runtime_caches() -> None:
    """Clear lazily cached hardware queries, mainly for tests."""

    is_ascend_available.cache_clear()
    get_num_ai_cores.cache_clear()
    get_num_vector_cores.cache_clear()
