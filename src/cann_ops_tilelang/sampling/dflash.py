"""Public Ascend interfaces for DFlash helper operators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Integral, Real

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .dflash_kernel import _confidence_head_fused_tl_ascend, _dflash_input_store_tl_ascend

_SAMPLING_EPS = 1e-5


def _check_npu_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be an NPU tensor, got {tensor.device}.')
    if device is not None and tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}.')
    if tensor.dtype != dtype:
        raise TypeError(f'{name} must use {dtype}, got {tensor.dtype}.')


def _resolve_temperature(temperature: Real | Sequence[Real], step_i: int) -> float:
    if isinstance(step_i, bool) or not isinstance(step_i, Integral) or step_i < 0:
        raise ValueError('step_i must be a non-negative integer.')
    value: Real
    if isinstance(temperature, Sequence) and not isinstance(temperature, (str, bytes)):
        if step_i >= len(temperature):
            raise ValueError(f'step_i {step_i} is outside the temperature sequence.')
        value = temperature[step_i]
    else:
        value = temperature  # type: ignore[assignment]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError('temperature must be a real number or a sequence of real numbers.')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('temperature must be finite.')
    return max(value, _SAMPLING_EPS)


def confidence_head_fused(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    markov_emb: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    step_i: int = 0,
    temperature: Real | Sequence[Real] = 1.0,
) -> torch.Tensor:
    """Apply the DFlash confidence head on Ascend.

    ``temperature`` replaces the framework environment setting used by the
    original integration and may be a scalar or a per-step sequence.
    """

    require_ascend()
    _check_npu_tensor('hidden', hidden, dtype=torch.bfloat16)
    _check_npu_tensor('weight', weight, dtype=torch.bfloat16, device=hidden.device)
    if hidden.ndim != 2:
        raise ValueError(f'hidden must be two-dimensional, got shape {tuple(hidden.shape)}.')
    if hidden.shape[1] <= 0:
        raise ValueError('hidden last dimension must be positive.')
    if hidden.stride(1) != 1:
        raise ValueError('hidden must have unit inner stride.')
    if not weight.is_contiguous():
        raise ValueError('weight must be contiguous.')

    num_rows, n_embed = hidden.shape
    has_markov = markov_emb is not None
    markov_rank = 0
    if markov_emb is not None:
        _check_npu_tensor('markov_emb', markov_emb, dtype=torch.bfloat16, device=hidden.device)
        if markov_emb.ndim != 2 or markov_emb.shape[0] != num_rows:
            raise ValueError(f'markov_emb must have shape ({num_rows}, markov_rank).')
        markov_rank = markov_emb.shape[1]
        if markov_rank <= 0:
            raise ValueError('markov_emb last dimension must be positive when provided.')
        if not markov_emb.is_contiguous():
            raise ValueError('markov_emb must be contiguous.')
    total_dim = n_embed + markov_rank
    if weight.numel() != total_dim:
        raise ValueError(f'weight must contain {total_dim} elements, got {weight.numel()}.')
    if total_dim % 64 or total_dim > 11000:
        raise ValueError('n_embed + markov_rank must be divisible by 64 and at most 11000.')

    if out is None:
        out = torch.empty(num_rows, dtype=torch.float32, device=hidden.device)
    else:
        _check_npu_tensor('out', out, dtype=torch.float32, device=hidden.device)
        if out.shape != (num_rows,):
            raise ValueError(f'out must have shape ({num_rows},), got {tuple(out.shape)}.')
        if out.stride(0) <= 0:
            raise ValueError('out must have a positive stride.')

    b8_fits_ub = 2 * (8 * 2 * total_dim + 64) + 2 * total_dim <= 210000
    b8_enough_work = num_rows >= 4 * get_num_vector_cores()
    block_rows = 8 if b8_enough_work and b8_fits_ub else 4
    kernel = _confidence_head_fused_tl_ascend(
        n_embed=n_embed,
        markov_rank=markov_rank,
        has_markov_emb=has_markov,
        block_rows=block_rows,
    )
    kernel(hidden, markov_emb, weight.view(-1), out, _resolve_temperature(temperature, step_i))
    return out


def dflash_input_store(
    hidden_states: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-reduce the optional HC dimension and store BF16 DFlash input."""

    require_ascend()
    _check_npu_tensor('hidden_states', hidden_states, dtype=torch.bfloat16)
    if hidden_states.ndim < 2 or not hidden_states.is_contiguous():
        raise ValueError('hidden_states must be a contiguous tensor with at least two dimensions.')
    n_embed = hidden_states.shape[-1]
    if n_embed <= 0 or n_embed > 8192 or n_embed % 128:
        raise ValueError('hidden_states last dimension must be positive, divisible by 128, and at most 8192.')
    hc_mult = hidden_states.shape[-2] if hidden_states.ndim > 2 else 1
    if hc_mult <= 0:
        raise ValueError('hidden_states HC dimension must be positive.')
    num_rows = hidden_states.numel() // (hc_mult * n_embed)
    hidden_states = hidden_states.view(num_rows, hc_mult, n_embed)
    rows_per_task = max(1, min(4, 32768 // n_embed)) if hc_mult == 1 else 1
    stage_bytes = 2 * rows_per_task * hc_mult * n_embed
    if stage_bytes > 240000:
        raise ValueError(
            f'dflash_input_store requires {stage_bytes} bytes per UB stage, exceeding the 240000-byte limit.'
        )

    if out is None:
        out = torch.empty((num_rows, n_embed), dtype=torch.bfloat16, device=hidden_states.device)
    else:
        _check_npu_tensor('out', out, dtype=torch.bfloat16, device=hidden_states.device)
        if out.shape != (num_rows, n_embed):
            raise ValueError(f'out must have shape {(num_rows, n_embed)}, got {tuple(out.shape)}.')
        if out.stride(1) != 1 or out.stride(0) % 4:
            raise ValueError('out must have unit inner stride and a row stride divisible by 4.')

    if num_rows == 0:
        return out

    kernel = _dflash_input_store_tl_ascend(n_embed, hc_mult, rows_per_task)
    kernel(hidden_states, out)
    return out
