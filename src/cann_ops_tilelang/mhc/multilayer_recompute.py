"""Ascend MHC multilayer recompute operator."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend
from cann_ops_tilelang.utils import align_up

from .multilayer_recompute_kernel import mhc_multilayer_recompute_asc

_HIDDEN_ALIGNMENT = 64
_MAX_HIDDEN_BLOCK = 8192


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be an NPU tensor, got {tensor.device}.')
    if tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}.')
    if tensor.dtype != dtype:
        raise TypeError(f'{name} must use {dtype}, got {tensor.dtype}.')
    if tensor.shape != shape:
        raise ValueError(f'{name} must have shape {shape}, got {tuple(tensor.shape)}.')
    if not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def _check_tensor_list(
    name: str,
    tensors: Sequence[torch.Tensor],
    *,
    length: int,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensors, list):
        raise TypeError(f'{name} must be a list of torch.Tensor instances.')
    if len(tensors) != length:
        raise ValueError(f'{name} must contain {length} tensors, got {len(tensors)}.')
    for index, tensor in enumerate(tensors):
        _check_tensor(
            f'{name}[{index}]',
            tensor,
            dtype=dtype,
            shape=shape,
            device=device,
        )


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _check_output_aliases(
    outputs: Sequence[tuple[str, torch.Tensor]],
    inputs: Sequence[tuple[str, torch.Tensor]],
) -> None:
    for output_index, (output_name, output) in enumerate(outputs):
        for input_name, input_tensor in inputs:
            if _shares_storage(output, input_tensor):
                raise ValueError(f'{output_name} must not share storage with {input_name}.')
        for other_name, other_output in outputs[:output_index]:
            if _shares_storage(output, other_output):
                raise ValueError(f'{output_name} must not share storage with {other_name}.')


def _make_pointer_tables(
    tensor_lists: Sequence[Sequence[torch.Tensor]],
    device: torch.device,
) -> list[torch.Tensor]:
    sizes = [len(tensors) for tensors in tensor_lists]
    pointer_buffer = torch.empty(sum(sizes), dtype=torch.int64, device='cpu', pin_memory=True)
    offset = 0
    for tensors in tensor_lists:
        for index, tensor in enumerate(tensors):
            pointer_buffer[offset + index] = tensor.data_ptr()
        offset += len(tensors)

    device_buffer = pointer_buffer.to(device=device, non_blocking=True)
    pointer_tables: list[torch.Tensor] = []
    offset = 0
    for size in sizes:
        pointer_tables.append(device_buffer[offset : offset + size])
        offset += size
    return pointer_tables


def mhc_multilayer_recompute(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
    layer_input_list: list[torch.Tensor],
    residual_list: list[torch.Tensor],
) -> None:
    """Recompute MHC layer inputs and residuals into caller-provided buffers."""

    if not isinstance(initial_residual, torch.Tensor):
        raise TypeError('initial_residual must be a torch.Tensor.')
    if initial_residual.ndim < 3:
        raise ValueError(
            f'initial_residual must have shape (..., mhc_mult, hidden_size), got {tuple(initial_residual.shape)}.'
        )

    require_ascend()
    if initial_residual.device.type != 'npu':
        raise ValueError(f'initial_residual must be an NPU tensor, got {initial_residual.device}.')
    if initial_residual.dtype != torch.bfloat16:
        raise TypeError(f'initial_residual must use {torch.bfloat16}, got {initial_residual.dtype}.')
    if not initial_residual.is_contiguous():
        raise ValueError('initial_residual must be contiguous.')

    for name, tensors in (
        ('pre_mix_list', pre_mix_list),
        ('layer_output_list', layer_output_list),
        ('post_mix_list', post_mix_list),
        ('comb_mix_list', comb_mix_list),
        ('layer_input_list', layer_input_list),
        ('residual_list', residual_list),
    ):
        if not isinstance(tensors, list):
            raise TypeError(f'{name} must be a list of torch.Tensor instances.')

    num_layers = len(pre_mix_list)
    num_post_layers = len(layer_output_list)
    if num_layers <= 0:
        raise ValueError(f'pre_mix_list must contain at least one tensor, got {num_layers}.')
    if num_post_layers not in (num_layers - 1, num_layers):
        raise ValueError(
            f'layer_output_list must contain {num_layers - 1} or {num_layers} tensors, got {num_post_layers}.'
        )

    leading_shape = tuple(initial_residual.shape[:-2])
    mhc_mult, hidden_size = initial_residual.shape[-2:]
    if mhc_mult <= 0:
        raise ValueError(f'mhc_mult must be positive, got {mhc_mult}.')
    if hidden_size <= 0 or hidden_size % _HIDDEN_ALIGNMENT:
        raise ValueError(f'hidden_size must be a positive multiple of {_HIDDEN_ALIGNMENT}, got {hidden_size}.')

    device = initial_residual.device
    _check_tensor_list(
        'pre_mix_list',
        pre_mix_list,
        length=num_layers,
        dtype=torch.float32,
        shape=(*leading_shape, mhc_mult, 1),
        device=device,
    )
    _check_tensor_list(
        'layer_output_list',
        layer_output_list,
        length=num_post_layers,
        dtype=torch.bfloat16,
        shape=(*leading_shape, hidden_size),
        device=device,
    )
    _check_tensor_list(
        'post_mix_list',
        post_mix_list,
        length=num_post_layers,
        dtype=torch.float32,
        shape=(*leading_shape, mhc_mult, 1),
        device=device,
    )
    _check_tensor_list(
        'comb_mix_list',
        comb_mix_list,
        length=num_post_layers,
        dtype=torch.float32,
        shape=(*leading_shape, mhc_mult, mhc_mult),
        device=device,
    )
    _check_tensor_list(
        'layer_input_list',
        layer_input_list,
        length=num_layers,
        dtype=torch.bfloat16,
        shape=(*leading_shape, hidden_size),
        device=device,
    )
    _check_tensor_list(
        'residual_list',
        residual_list,
        length=num_post_layers,
        dtype=torch.bfloat16,
        shape=(*leading_shape, mhc_mult, hidden_size),
        device=device,
    )

    input_tensors = [('initial_residual', initial_residual)]
    for list_name, tensors in (
        ('pre_mix_list', pre_mix_list),
        ('layer_output_list', layer_output_list),
        ('post_mix_list', post_mix_list),
        ('comb_mix_list', comb_mix_list),
    ):
        input_tensors.extend((f'{list_name}[{index}]', tensor) for index, tensor in enumerate(tensors))
    output_tensors = [
        *((f'layer_input_list[{index}]', tensor) for index, tensor in enumerate(layer_input_list)),
        *((f'residual_list[{index}]', tensor) for index, tensor in enumerate(residual_list)),
    ]
    _check_output_aliases(output_tensors, input_tensors)

    num_tokens = initial_residual.numel() // (mhc_mult * hidden_size)
    if num_tokens == 0:
        return

    (
        pre_mix_ptrs,
        layer_input_ptrs,
        residual_ptrs,
        layer_output_ptrs,
        post_mix_ptrs,
        comb_mix_ptrs,
    ) = _make_pointer_tables(
        [
            pre_mix_list,
            layer_input_list,
            residual_list,
            layer_output_list,
            post_mix_list,
            comb_mix_list,
        ],
        device,
    )

    hidden_block = min(align_up(hidden_size, _HIDDEN_ALIGNMENT), _MAX_HIDDEN_BLOCK)
    mhc_multilayer_recompute_asc(
        initial_residual.view(num_tokens, mhc_mult, hidden_size),
        pre_mix_ptrs,
        layer_output_ptrs,
        post_mix_ptrs,
        comb_mix_ptrs,
        layer_input_ptrs,
        residual_ptrs,
        h_blk=hidden_block,
        num_vec_cores=get_num_vector_cores(),
    )
