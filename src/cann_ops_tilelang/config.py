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
_DEFAULT_TOKEN_ALIGNMENT = 128
_DEFAULT_QUANT_CLAMP_MIN_VALUES = {
    'e4m3': 1e-4,
    'e2m1': 6.0 * (2.0**-126),
}

_num_ai_cores_override: int | None = None
_token_alignment = _DEFAULT_TOKEN_ALIGNMENT
_quant_clamp_min_values = _DEFAULT_QUANT_CLAMP_MIN_VALUES.copy()


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
    detected = int(cube_cores)
    if device_index is None and _num_ai_cores_override is not None:
        return _num_ai_cores_override
    return detected


def set_num_ai_cores(num_ai_cores: int) -> None:
    """Override the AI core count used for kernel tuning in tests and experiments."""

    if isinstance(num_ai_cores, bool) or not isinstance(num_ai_cores, int):
        raise TypeError('num_ai_cores must be an integer.')

    global _num_ai_cores_override
    previous = _num_ai_cores_override
    _num_ai_cores_override = None
    get_num_ai_cores.cache_clear()
    detected = get_num_ai_cores()
    if not 0 < num_ai_cores <= detected:
        _num_ai_cores_override = previous
        get_num_ai_cores.cache_clear()
        raise ValueError(f'num_ai_cores must be in [1, {detected}], got {num_ai_cores}.')
    _num_ai_cores_override = num_ai_cores
    get_num_ai_cores.cache_clear()
    get_num_vector_cores.cache_clear()


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


def get_token_alignment() -> int:
    """Return the token alignment used by fused quantization kernels."""

    return _token_alignment


def set_token_alignment(token_alignment: int) -> None:
    """Set the token alignment used by fused quantization kernels."""

    if isinstance(token_alignment, bool) or not isinstance(token_alignment, int):
        raise TypeError('token_alignment must be an integer.')
    if token_alignment <= 0:
        raise ValueError(f'token_alignment must be positive, got {token_alignment}.')
    global _token_alignment
    _token_alignment = token_alignment


def _validate_quant_format(fmt: str) -> None:
    if fmt not in _quant_clamp_min_values:
        raise ValueError(f'Unsupported quantization format: {fmt}.')


def get_amax_clamp_for_quant(fmt: str) -> float:
    """Return the minimum magnitude used when computing quantization scales."""

    _validate_quant_format(fmt)
    return _quant_clamp_min_values[fmt]


def set_amax_clamp_for_quant(fmt: str, clamp_min_value: float) -> None:
    """Set the minimum magnitude used when computing quantization scales."""

    _validate_quant_format(fmt)
    value = float(clamp_min_value)
    if value <= 0:
        raise ValueError(f'clamp_min_value must be positive, got {clamp_min_value}.')
    _quant_clamp_min_values[fmt] = value


def reset_runtime_config() -> None:
    """Reset mutable quantization tuning settings to their defaults."""

    global _num_ai_cores_override, _token_alignment, _quant_clamp_min_values
    _num_ai_cores_override = None
    _token_alignment = _DEFAULT_TOKEN_ALIGNMENT
    _quant_clamp_min_values = _DEFAULT_QUANT_CLAMP_MIN_VALUES.copy()
    clear_runtime_caches()


def clear_runtime_caches() -> None:
    """Clear lazily cached hardware queries, mainly for tests."""

    is_ascend_available.cache_clear()
    get_num_ai_cores.cache_clear()
    get_num_vector_cores.cache_clear()
