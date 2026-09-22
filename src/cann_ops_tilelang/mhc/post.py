"""Public interfaces for MHC post forward and backward."""

from __future__ import annotations

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source
from cann_ops_tilelang.utils import align_up, ceil_div

from .post_kernel import get_mhc_post_backward_kernel, get_mhc_post_forward_kernel

_HIDDEN_ALIGNMENT = 64


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


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def _validate_common_inputs(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> tuple[int, int, int, int]:
    tensors = (x, residual, post_layer_mix, comb_res_mix)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('x, residual, post_layer_mix and comb_res_mix must be torch.Tensor instances.')

    require_ascend()
    _check_npu_tensor('x', x, dtype=torch.bfloat16)
    _check_npu_tensor('residual', residual, dtype=torch.bfloat16, device=x.device)
    _check_npu_tensor('post_layer_mix', post_layer_mix, dtype=torch.float32, device=x.device)
    _check_npu_tensor('comb_res_mix', comb_res_mix, dtype=torch.float32, device=x.device)

    if residual.ndim != 4:
        raise ValueError('residual must have shape (num_seqs, num_tokens, mhc_mult, hidden_size).')
    num_seqs, num_tokens, mhc_mult, hidden_size = residual.shape
    if mhc_mult <= 0:
        raise ValueError(f'mhc_mult must be positive, got {mhc_mult}.')
    if hidden_size <= 0 or hidden_size % _HIDDEN_ALIGNMENT:
        raise ValueError(f'hidden_size must be a positive multiple of {_HIDDEN_ALIGNMENT}, got {hidden_size}.')

    expected_x_shape = (num_seqs, num_tokens, hidden_size)
    if x.shape != expected_x_shape:
        raise ValueError(f'x must have shape {expected_x_shape}, got {tuple(x.shape)}.')
    expected_post_shape = (num_seqs, num_tokens, mhc_mult, 1)
    if post_layer_mix.shape != expected_post_shape:
        raise ValueError(f'post_layer_mix must have shape {expected_post_shape}, got {tuple(post_layer_mix.shape)}.')
    expected_comb_shape = (num_seqs, num_tokens, mhc_mult, mhc_mult)
    if comb_res_mix.shape != expected_comb_shape:
        raise ValueError(f'comb_res_mix must have shape {expected_comb_shape}, got {tuple(comb_res_mix.shape)}.')
    return num_seqs, num_tokens, mhc_mult, hidden_size


def _choose_hidden_block(hidden_size: int) -> int:
    hidden_block = hidden_size if hidden_size <= 4096 else min(ceil_div(hidden_size, 2), 4096)
    return align_up(hidden_block, _HIDDEN_ALIGNMENT)


def mhc_post_forward(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the MHC post residual mix on Ascend.

    ``x`` has shape ``(num_seqs, num_tokens, hidden_size)`` and ``residual``
    has shape ``(num_seqs, num_tokens, mhc_mult, hidden_size)``. Activations
    use BF16, while both mix tensors use FP32.
    """
    num_seqs, num_tokens, mhc_mult, hidden_size = _validate_common_inputs(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    if out is not None:
        _check_npu_tensor('out', out, dtype=torch.bfloat16, device=x.device)
        if out.shape != residual.shape:
            raise ValueError(f'out must have shape {tuple(residual.shape)}, got {tuple(out.shape)}.')
        for name, tensor in (
            ('x', x),
            ('residual', residual),
            ('post_layer_mix', post_layer_mix),
            ('comb_res_mix', comb_res_mix),
        ):
            if _shares_storage(out, tensor):
                raise ValueError(f'out and {name} must not share storage.')

    result = torch.empty_like(residual) if out is None else out
    if num_seqs == 0 or num_tokens == 0:
        return result

    hidden_block = _choose_hidden_block(hidden_size)
    kernel = get_mhc_post_forward_kernel(
        mhc_mult,
        hidden_size,
        hidden_block,
        get_num_vector_cores(),
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    flattened_tokens = num_seqs * num_tokens
    kernel(
        comb_res_mix.view(flattened_tokens, mhc_mult, mhc_mult),
        residual.view(flattened_tokens, mhc_mult, hidden_size),
        post_layer_mix.view(flattened_tokens, mhc_mult),
        x.view(flattened_tokens, hidden_size),
        result.view(flattened_tokens, mhc_mult, hidden_size),
    )
    return result


def mhc_post_backward(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    d_o: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gradients for the four inputs of :func:`mhc_post_forward`."""
    num_seqs, num_tokens, mhc_mult, hidden_size = _validate_common_inputs(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    _check_npu_tensor('d_o', d_o, dtype=torch.bfloat16, device=x.device)
    if d_o.shape != residual.shape:
        raise ValueError(f'd_o must have shape {tuple(residual.shape)}, got {tuple(d_o.shape)}.')

    grad_x = torch.empty_like(x)
    grad_residual = torch.empty_like(residual)
    grad_post_layer_mix = torch.empty_like(post_layer_mix)
    grad_comb_res_mix = torch.empty_like(comb_res_mix)
    if num_seqs == 0 or num_tokens == 0:
        return grad_x, grad_residual, grad_post_layer_mix, grad_comb_res_mix

    hidden_block = _choose_hidden_block(hidden_size)
    num_stages = 3 if hidden_block <= 2560 else 2
    kernel = get_mhc_post_backward_kernel(
        mhc_mult,
        hidden_size,
        hidden_block,
        num_stages,
        get_num_vector_cores(),
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    flattened_tokens = num_seqs * num_tokens
    kernel(
        d_o.view(flattened_tokens, mhc_mult, hidden_size),
        comb_res_mix.view(flattened_tokens, mhc_mult, mhc_mult),
        residual.view(flattened_tokens, mhc_mult, hidden_size),
        post_layer_mix.view(flattened_tokens, mhc_mult),
        x.view(flattened_tokens, hidden_size),
        grad_comb_res_mix.view(flattened_tokens, mhc_mult, mhc_mult),
        grad_residual.view(flattened_tokens, mhc_mult, hidden_size),
        grad_post_layer_mix.view(flattened_tokens, mhc_mult),
        grad_x.view(flattened_tokens, hidden_size),
    )
    return grad_x, grad_residual, grad_post_layer_mix, grad_comb_res_mix
