"""Public interfaces for the Engram gate forward and backward operators."""

from __future__ import annotations

import math
import os
from numbers import Real

import torch

from cann_ops_tilelang import config
from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .engram_gate_kernel import get_engram_gate_bwd_kernel, get_engram_gate_fwd_kernel

_SUPPORTED_HC_MULT = 4
_NUM_GRAD_V_CORES = 12

# The Ascend runtime is fixed for a compiled graph; only this check is constant.
# Real implementations retain the check, and mutable core counts stay traceable.
torch.compiler.assume_constant_result(require_ascend)


def _check_real(name: str, value: float, *, positive: bool) -> None:
    if not isinstance(value, Real):
        raise TypeError(f'{name} must be a real number.')
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite.')
    if positive and value <= 0:
        raise ValueError(f'{name} must be greater than zero.')
    if not positive and value < 0:
        raise ValueError(f'{name} must be non-negative.')


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


def _validate_common_inputs(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
) -> tuple[int, int, int]:
    _check_npu_tensor('hidden_states', hidden_states, dtype=torch.bfloat16)
    _check_npu_tensor('kv', kv, dtype=torch.bfloat16, device=hidden_states.device)
    _check_npu_tensor('weight_fused', weight_fused, dtype=torch.float32, device=hidden_states.device)

    if hidden_states.ndim != 3:
        raise ValueError('hidden_states must have shape (num_tokens, hc_mult, hidden_size).')
    num_tokens, hc_mult, hidden_size = hidden_states.shape
    if hc_mult != _SUPPORTED_HC_MULT:
        raise ValueError(f'Engram gate currently requires hc_mult == {_SUPPORTED_HC_MULT}, got {hc_mult}.')
    if hidden_size == 0 or hidden_size % 64:
        raise ValueError(f'hidden_size must be a positive multiple of 64, got {hidden_size}.')
    if kv.shape != (num_tokens, (hc_mult + 1) * hidden_size):
        raise ValueError(f'kv must have shape {(num_tokens, (hc_mult + 1) * hidden_size)}, got {tuple(kv.shape)}.')
    if weight_fused.shape != (hc_mult, hidden_size):
        raise ValueError(f'weight_fused must have shape {(hc_mult, hidden_size)}, got {tuple(weight_fused.shape)}.')
    return num_tokens, hc_mult, hidden_size


def _split_kv(kv: torch.Tensor, hc_mult: int, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = kv.shape[0]
    k, v = kv.split([hc_mult * hidden_size, hidden_size], dim=-1)
    return k.view(num_tokens, hc_mult, hidden_size), v


def _validate_fwd_inputs(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    eps: float,
    clamp_value: float,
    save_for_backward: bool,
) -> tuple[int, int, int]:
    shape = _validate_common_inputs(hidden_states, kv, weight_fused)
    _check_real('eps', eps, positive=False)
    _check_real('clamp_value', clamp_value, positive=True)
    if not isinstance(save_for_backward, bool):
        raise TypeError('save_for_backward must be bool.')
    return shape


def _engram_gate_fwd_impl(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    eps: float,
    clamp_value: float,
    save_for_backward: bool = True,
    num_cores: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    num_tokens, hc_mult, hidden_size = _validate_fwd_inputs(
        hidden_states,
        kv,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward,
    )
    require_ascend()

    output = torch.empty_like(hidden_states)
    stats_shape = (num_tokens, hc_mult)
    if save_for_backward:
        dot = torch.empty(stats_shape, dtype=torch.float32, device=hidden_states.device)
        gate_score = torch.empty_like(dot)
        rstd_x = torch.empty_like(dot)
        rstd_k = torch.empty_like(dot)
    else:
        # The no-save Ascend kernel keeps a signature-compatible placeholder
        # but does not read or write any of these tensors.
        dot = torch.empty(stats_shape, dtype=torch.float32, device=hidden_states.device)
        gate_score = rstd_x = rstd_k = dot

    if num_tokens == 0:
        if save_for_backward:
            return output, dot, gate_score, rstd_x, rstd_k
        return output, None, None, None, None

    k, v = _split_kv(kv, hc_mult, hidden_size)
    k_stride_s, k_stride_h, _ = k.stride()
    v_stride_s, _ = v.stride()
    num_cores = get_num_vector_cores() if num_cores is None else num_cores
    if num_cores <= 0 or num_cores % hc_mult:
        raise RuntimeError(f'Ascend vector core count {num_cores} must be divisible by hc_mult={hc_mult}.')
    kernel = get_engram_gate_fwd_kernel(
        hidden_size,
        float(eps),
        hidden_size**-0.5,
        k_stride_s,
        k_stride_h,
        v_stride_s,
        float(clamp_value),
        hc_mult,
        num_cores,
        save_for_backward,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(hidden_states, k, v, weight_fused, output, dot, gate_score, rstd_x, rstd_k)

    if save_for_backward:
        return output, dot, gate_score, rstd_x, rstd_k
    return output, None, None, None, None


# PyTorch 2.10 cannot infer optional Tensor returns, so declare the schema.
@torch.library.custom_op(
    'cann_ops_tilelang::engram_gate_fwd',
    mutates_args=(),
    schema='(Tensor hidden_states, Tensor kv, Tensor weight_fused, float eps, float clamp_value, '
    'int num_cores, bool save_for_backward=True) -> (Tensor, Tensor?, Tensor?, Tensor?, Tensor?)',
)
def _compiled_gate_fwd(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    eps: float,
    clamp_value: float,
    num_cores: int,
    save_for_backward: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    return _engram_gate_fwd_impl(hidden_states, kv, weight_fused, eps, clamp_value, save_for_backward, num_cores)


@_compiled_gate_fwd.register_fake
def _gate_fwd_fake(hidden_states, kv, weight_fused, eps, clamp_value, num_cores, save_for_backward=True):
    num_tokens, hc_mult, _ = _validate_fwd_inputs(
        hidden_states,
        kv,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward,
    )
    if num_tokens and (num_cores <= 0 or num_cores % hc_mult):
        raise RuntimeError(f'Ascend vector core count {num_cores} must be divisible by hc_mult={hc_mult}.')
    output = torch.empty_like(hidden_states)
    if not save_for_backward:
        return output, None, None, None, None
    dot = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=hidden_states.device)
    return output, dot, torch.empty_like(dot), torch.empty_like(dot), torch.empty_like(dot)


def engram_gate_fwd(
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    eps: float,
    clamp_value: float,
    save_for_backward: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Apply the Engram gate and optionally return backward statistics.

    ``hidden_states`` has shape ``(num_tokens, 4, hidden_size)``. ``kv`` packs
    four key heads followed by one shared value head and therefore has shape
    ``(num_tokens, 5 * hidden_size)``. All activation tensors use BF16;
    ``weight_fused`` and the saved statistics use FP32.

    With a compatible torch_npu, supports fixed-shape inference with
    ``torch.compile(..., backend='npugraph_ex', fullgraph=True, dynamic=False)``.
    The compiled path uses an opaque custom operator with a fake implementation;
    tracing does not compile or launch the TileLang kernel. No autograd formula
    is registered, including when ``save_for_backward=True``.
    """
    if torch.compiler.is_compiling():
        # Empty inputs return before core configuration is read in the eager path.
        num_cores = _get_gate_num_cores() if hidden_states.numel() else 0
        return _compiled_gate_fwd(hidden_states, kv, weight_fused, eps, clamp_value, num_cores, save_for_backward)
    return _engram_gate_fwd_impl(hidden_states, kv, weight_fused, eps, clamp_value, save_for_backward)


def _validate_bwd_inputs(
    grad_out: torch.Tensor,
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    dot: torch.Tensor,
    gate_score: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_k: torch.Tensor,
    clamp_value: float,
) -> tuple[int, int, int]:
    num_tokens, hc_mult, hidden_size = _validate_common_inputs(hidden_states, kv, weight_fused)
    _check_npu_tensor('grad_out', grad_out, dtype=torch.bfloat16, device=hidden_states.device)
    if grad_out.shape != hidden_states.shape:
        raise ValueError(f'grad_out must have shape {tuple(hidden_states.shape)}, got {tuple(grad_out.shape)}.')
    stats_shape = (num_tokens, hc_mult)
    for name, tensor in (
        ('dot', dot),
        ('gate_score', gate_score),
        ('rstd_x', rstd_x),
        ('rstd_k', rstd_k),
    ):
        _check_npu_tensor(name, tensor, dtype=torch.float32, device=hidden_states.device)
        if tensor.shape != stats_shape:
            raise ValueError(f'{name} must have shape {stats_shape}, got {tuple(tensor.shape)}.')
    _check_real('clamp_value', clamp_value, positive=True)
    return num_tokens, hc_mult, hidden_size


def _get_gate_num_cores() -> int:
    # Compiled wrappers need this metadata without tracing require_ascend.
    # Keep mutable tuning settings visible to Dynamo so they remain guarded.
    detected = config.get_num_ai_cores() * 2
    # Guard the key set as well, including an unset-to-set environment change.
    key = 'OPS_TILELANG_NUM_VECTOR_CORES'
    override = os.environ[key] if key in tuple(os.environ) else None
    if override is None:
        return detected
    requested = int(override)
    if requested <= 0 or requested > detected:
        raise ValueError(f'OPS_TILELANG_NUM_VECTOR_CORES must be in [1, {detected}], got {requested}.')
    return requested


def _get_bwd_num_partials(hc_mult: int, num_cores: int) -> int:
    num_main_cores = num_cores - _NUM_GRAD_V_CORES
    num_head_groups = hc_mult // 2
    if num_main_cores <= 0 or num_main_cores % num_head_groups:
        raise RuntimeError(
            f'Ascend vector core count {num_cores} is incompatible with '
            f'{_NUM_GRAD_V_CORES} grad-v cores and hc_mult={hc_mult}.'
        )
    return num_main_cores // num_head_groups


def _engram_gate_bwd_impl(
    grad_out: torch.Tensor,
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    dot: torch.Tensor,
    gate_score: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_k: torch.Tensor,
    clamp_value: float,
    num_cores: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_ascend()
    num_tokens, hc_mult, hidden_size = _validate_bwd_inputs(
        grad_out, hidden_states, kv, weight_fused, dot, gate_score, rstd_x, rstd_k, clamp_value
    )
    num_cores = get_num_vector_cores() if num_cores is None else num_cores
    num_partials = _get_bwd_num_partials(hc_mult, num_cores)

    grad_hidden_states = torch.empty_like(hidden_states)
    grad_kv = torch.empty_like(kv)
    grad_w_partial = torch.empty(
        (num_partials, hc_mult, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    if num_tokens == 0:
        grad_w_partial.zero_()
        return grad_hidden_states, grad_kv, grad_w_partial

    k, v = _split_kv(kv, hc_mult, hidden_size)
    grad_k_flat, grad_v = grad_kv.split([hc_mult * hidden_size, hidden_size], dim=-1)
    grad_k = grad_k_flat.view(num_tokens, hc_mult, hidden_size)
    k_stride_s, k_stride_h, _ = k.stride()
    v_stride_s, _ = v.stride()
    kernel = get_engram_gate_bwd_kernel(
        hidden_size,
        hidden_size**-0.5,
        k_stride_s,
        k_stride_h,
        v_stride_s,
        num_cores,
        _NUM_GRAD_V_CORES,
        float(clamp_value),
        hc_mult,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(
        grad_out,
        hidden_states,
        k,
        v,
        weight_fused,
        dot,
        gate_score,
        rstd_x,
        rstd_k,
        grad_hidden_states,
        grad_k,
        grad_v,
        grad_w_partial,
    )
    return grad_hidden_states, grad_kv, grad_w_partial


@torch.library.custom_op('cann_ops_tilelang::engram_gate_bwd', mutates_args=())
def _compiled_gate_bwd(
    grad_out: torch.Tensor,
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    dot: torch.Tensor,
    gate_score: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_k: torch.Tensor,
    clamp_value: float,
    num_cores: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _engram_gate_bwd_impl(
        grad_out, hidden_states, kv, weight_fused, dot, gate_score, rstd_x, rstd_k, clamp_value, num_cores
    )


@_compiled_gate_bwd.register_fake
def _gate_bwd_fake(grad_out, hidden_states, kv, weight_fused, dot, gate_score, rstd_x, rstd_k, clamp_value, num_cores):
    _, hc_mult, hidden_size = _validate_bwd_inputs(
        grad_out, hidden_states, kv, weight_fused, dot, gate_score, rstd_x, rstd_k, clamp_value
    )
    num_partials = _get_bwd_num_partials(hc_mult, num_cores)
    grad_w_partial = torch.empty(
        (num_partials, hc_mult, hidden_size),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return torch.empty_like(hidden_states), torch.empty_like(kv), grad_w_partial


def engram_gate_bwd(
    grad_out: torch.Tensor,
    hidden_states: torch.Tensor,
    kv: torch.Tensor,
    weight_fused: torch.Tensor,
    dot: torch.Tensor,
    gate_score: torch.Tensor,
    rstd_x: torch.Tensor,
    rstd_k: torch.Tensor,
    clamp_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the Engram gate backward kernel.

    Returns gradients for ``hidden_states`` and packed ``kv``, plus partial
    FP32 gradients for ``weight_fused``. Pass the latter to
    :func:`engram_grad_w_reduce` together with the unfused weights.

    The explicit backward call supports fixed-shape
    ``torch.compile(..., backend='npugraph_ex', fullgraph=True, dynamic=False)``
    Tested with torch/torch_npu 2.10 using ``torch.inference_mode()`` for
    both graph calls; this does not add automatic differentiation support.
    Vector core count is an internal scalar argument so fake execution can infer
    the partial-gradient shape without querying the device. No autograd formula
    or higher-order gradient support is registered.
    """
    if torch.compiler.is_compiling():
        return _compiled_gate_bwd(
            grad_out,
            hidden_states,
            kv,
            weight_fused,
            dot,
            gate_score,
            rstd_x,
            rstd_k,
            clamp_value,
            _get_gate_num_cores(),
        )
    return _engram_gate_bwd_impl(
        grad_out, hidden_states, kv, weight_fused, dot, gate_score, rstd_x, rstd_k, clamp_value
    )
