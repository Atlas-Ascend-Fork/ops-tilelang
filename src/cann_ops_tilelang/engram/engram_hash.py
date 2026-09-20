"""Public interface for the Engram n-gram hash operator."""

from __future__ import annotations

import math

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .engram_hash_kernel import get_engram_hash_kernel


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
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """Compute Engram n-gram hash embedding indices on Ascend.

    ``ngram_token_ids`` has shape ``(num_tokens, max_ngram_size)`` and uses
    INT32. ``multipliers`` uses INT64, while ``vocab_sizes`` and ``offsets``
    use INT32. The caller must provide strictly positive vocabulary sizes.
    """

    if not all(isinstance(tensor, torch.Tensor) for tensor in (ngram_token_ids, multipliers, vocab_sizes, offsets)):
        raise TypeError('ngram_token_ids, multipliers, vocab_sizes and offsets must be torch.Tensor instances.')

    require_ascend()
    _check_npu_tensor('ngram_token_ids', ngram_token_ids, dtype=torch.int32)
    _check_npu_tensor('multipliers', multipliers, dtype=torch.int64, device=ngram_token_ids.device)
    _check_npu_tensor('vocab_sizes', vocab_sizes, dtype=torch.int32, device=ngram_token_ids.device)
    _check_npu_tensor('offsets', offsets, dtype=torch.int32, device=ngram_token_ids.device)

    if ngram_token_ids.ndim != 2:
        raise ValueError('ngram_token_ids must have shape (num_tokens, max_ngram_size).')
    num_tokens, max_ngram_size = ngram_token_ids.shape
    if max_ngram_size < 2:
        raise ValueError(f'max_ngram_size must be at least 2, got {max_ngram_size}.')
    if vocab_sizes.ndim != 3:
        raise ValueError(
            'vocab_sizes must have shape (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram).'
        )
    num_ngram_layers, vocab_ngram_size, num_embed_table_per_ngram = vocab_sizes.shape
    if num_ngram_layers == 0:
        raise ValueError('num_ngram_layers must be positive.')
    if vocab_ngram_size != max_ngram_size - 1:
        raise ValueError(f'vocab_sizes dimension 1 must be {max_ngram_size - 1}, got {vocab_ngram_size}.')
    if num_embed_table_per_ngram == 0:
        raise ValueError('num_embed_table_per_ngram must be positive.')
    if multipliers.shape != (num_ngram_layers, max_ngram_size):
        raise ValueError(
            f'multipliers must have shape {(num_ngram_layers, max_ngram_size)}, got {tuple(multipliers.shape)}.'
        )

    num_out_cols = (max_ngram_size - 1) * num_embed_table_per_ngram
    if offsets.shape != (num_ngram_layers, num_out_cols):
        raise ValueError(f'offsets must have shape {(num_ngram_layers, num_out_cols)}, got {tuple(offsets.shape)}.')
    token_alignment = math.lcm(
        8 // math.gcd(max_ngram_size, 8),
        8 // math.gcd(num_out_cols, 8),
    )
    if 1024 // num_out_cols < token_alignment:
        raise ValueError(
            'the n-gram configuration exceeds the 1024-thread Ascend SimtVF limit: '
            f'max_ngram_size={max_ngram_size}, num_out_cols={num_out_cols}.'
        )

    output = torch.empty(
        (num_ngram_layers, num_tokens, num_out_cols),
        dtype=torch.int32,
        device=ngram_token_ids.device,
    )
    if num_tokens == 0:
        return output

    available_cores = get_num_vector_cores()
    if num_ngram_layers > available_cores:
        raise ValueError(
            f'num_ngram_layers must not exceed the Ascend vector core count {available_cores}, got {num_ngram_layers}.'
        )
    kernel = get_engram_hash_kernel(
        max_ngram_size,
        num_ngram_layers,
        num_embed_table_per_ngram,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(ngram_token_ids, multipliers, vocab_sizes, offsets, output)
    return output
