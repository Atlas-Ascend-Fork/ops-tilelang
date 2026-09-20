"""Public interface for fused Mixture-of-Experts routing."""

from __future__ import annotations

import math

import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .moe_topk_gate_kernel import get_moe_topk_gate_kernel
from .scoring import ScoringFunc


def _check_npu_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    ndim: int,
    contiguous: bool = True,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f'{name} must be a torch.Tensor.')
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be an NPU tensor, got {tensor.device}.')
    if tensor.device != device:
        raise ValueError(f'{name} must be on {device}, got {tensor.device}.')
    if tensor.dtype != dtype:
        raise TypeError(f'{name} must use {dtype}, got {tensor.dtype}.')
    if tensor.ndim != ndim:
        raise ValueError(f'{name} must have {ndim} dimensions, got {tensor.ndim}.')
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f'{name} must be contiguous.')


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.numel() > 0
        and right.numel() > 0
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
    )


def moe_topk_gate(
    logits: torch.Tensor,
    num_topk: int,
    use_shared_as_routed: bool,
    num_shared_experts: int,
    routed_scaling_factor: float,
    ep_rank: int,
    scoring_func: str = 'identity',
    mask: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    image_bias: torch.Tensor | None = None,
    image_token_mask: torch.Tensor | None = None,
    fix_routing_mask: torch.Tensor | None = None,
    to_physical_map: torch.Tensor | None = None,
    logical_count: torch.Tensor | None = None,
    unmapped_topk_idx: torch.Tensor | None = None,
    force_random: torch.Tensor | None = None,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute stable top-k expert routing on Ascend."""

    if not isinstance(logits, torch.Tensor):
        raise TypeError('logits must be a torch.Tensor.')
    if not isinstance(num_topk, int) or isinstance(num_topk, bool):
        raise TypeError(f'num_topk must be an int, got {type(num_topk).__name__}.')
    if not isinstance(use_shared_as_routed, bool):
        raise TypeError('use_shared_as_routed must be a bool.')
    if not isinstance(num_shared_experts, int) or isinstance(num_shared_experts, bool):
        raise TypeError(f'num_shared_experts must be an int, got {type(num_shared_experts).__name__}.')
    if not isinstance(routed_scaling_factor, (int, float)) or isinstance(routed_scaling_factor, bool):
        raise TypeError('routed_scaling_factor must be a real number.')
    if not math.isfinite(routed_scaling_factor):
        raise ValueError('routed_scaling_factor must be finite.')
    if not isinstance(ep_rank, int) or isinstance(ep_rank, bool):
        raise TypeError(f'ep_rank must be an int, got {type(ep_rank).__name__}.')
    if ep_rank < 0:
        raise ValueError(f'ep_rank must be non-negative, got {ep_rank}.')
    if not isinstance(scoring_func, str):
        raise TypeError('scoring_func must be a string.')

    require_ascend()
    device = logits.device
    _check_npu_tensor('logits', logits, dtype=torch.float32, device=device, ndim=2)
    num_tokens, num_routed_experts = logits.shape
    if num_topk <= 0 or num_topk > min(32, num_routed_experts):
        raise ValueError(f'num_topk must be in [1, min(32, {num_routed_experts})], got {num_topk}.')

    if use_shared_as_routed:
        if num_shared_experts not in (1, 2):
            raise ValueError(f'num_shared_experts must be 1 or 2, got {num_shared_experts}.')
        if num_topk % num_shared_experts != 0:
            raise ValueError('num_topk must be divisible by num_shared_experts.')
        routed_group_size = num_topk // num_shared_experts
        if num_routed_experts % routed_group_size != 0:
            raise ValueError('num_routed_experts must be divisible by num_topk // num_shared_experts.')
    else:
        num_shared_experts = 0

    has_bias = bias is not None
    if bias is not None:
        _check_npu_tensor('bias', bias, dtype=torch.float32, device=device, ndim=1)
        if bias.shape != (num_routed_experts,):
            raise ValueError(f'bias must have shape {(num_routed_experts,)}, got {tuple(bias.shape)}.')

    has_image_token_mask = image_token_mask is not None
    if has_image_token_mask != (image_bias is not None):
        raise ValueError('image_bias and image_token_mask must be both provided or both None.')
    if image_bias is not None and image_token_mask is not None:
        _check_npu_tensor('image_bias', image_bias, dtype=torch.float32, device=device, ndim=1)
        _check_npu_tensor('image_token_mask', image_token_mask, dtype=torch.bool, device=device, ndim=1)
        if image_bias.shape != (num_routed_experts,):
            raise ValueError(f'image_bias must have shape {(num_routed_experts,)}, got {tuple(image_bias.shape)}.')
        if image_token_mask.shape != (num_tokens,):
            raise ValueError(f'image_token_mask must have shape {(num_tokens,)}, got {tuple(image_token_mask.shape)}.')

    if mask is not None:
        _check_npu_tensor('mask', mask, dtype=torch.bool, device=device, ndim=1)
        if mask.shape != (num_tokens,):
            raise ValueError(f'mask must have shape {(num_tokens,)}, got {tuple(mask.shape)}.')
    if force_random is not None:
        _check_npu_tensor('force_random', force_random, dtype=torch.bool, device=device, ndim=1)
        if force_random.shape != (num_tokens,):
            raise ValueError(f'force_random must have shape {(num_tokens,)}, got {tuple(force_random.shape)}.')

    if (to_physical_map is None) != (logical_count is None):
        raise ValueError('to_physical_map and logical_count must be both provided or both None.')
    num_logical_experts = num_routed_experts + num_shared_experts
    if to_physical_map is not None and logical_count is not None:
        _check_npu_tensor('to_physical_map', to_physical_map, dtype=torch.int32, device=device, ndim=2)
        _check_npu_tensor('logical_count', logical_count, dtype=torch.int32, device=device, ndim=1)
        if to_physical_map.shape[0] != num_logical_experts or to_physical_map.shape[1] <= 0:
            raise ValueError(
                'to_physical_map must have shape '
                f'({num_logical_experts}, num_duplicate_experts), got {tuple(to_physical_map.shape)}.'
            )
        if logical_count.shape != (num_logical_experts,):
            raise ValueError(
                f'logical_count must have shape {(num_logical_experts,)}, got {tuple(logical_count.shape)}.'
            )

    if unmapped_topk_idx is not None:
        _check_npu_tensor(
            'unmapped_topk_idx',
            unmapped_topk_idx,
            dtype=torch.int64,
            device=device,
            ndim=2,
            contiguous=False,
        )
        if unmapped_topk_idx.shape != (num_tokens, num_topk):
            raise ValueError(
                f'unmapped_topk_idx must have shape {(num_tokens, num_topk)}, got {tuple(unmapped_topk_idx.shape)}.'
            )
        if unmapped_topk_idx.stride(1) != 1:
            raise ValueError('unmapped_topk_idx must have a last-dimension stride of 1.')
    if fix_routing_mask is not None:
        if unmapped_topk_idx is None:
            raise ValueError('fix_routing_mask requires unmapped_topk_idx.')
        _check_npu_tensor('fix_routing_mask', fix_routing_mask, dtype=torch.bool, device=device, ndim=1)
        if fix_routing_mask.shape != (num_tokens,):
            raise ValueError(f'fix_routing_mask must have shape {(num_tokens,)}, got {tuple(fix_routing_mask.shape)}.')

    scoring_type = ScoringFunc.from_str(scoring_func).value

    output_shape = (num_tokens, num_topk + num_shared_experts)
    if out is None:
        topk_idx = torch.empty(output_shape, dtype=torch.int64, device=device)
        topk_weights = torch.empty(output_shape, dtype=torch.float32, device=device)
    else:
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError('out must be a tuple of (topk_idx, topk_weights).')
        topk_idx, topk_weights = out
        _check_npu_tensor('out[0]', topk_idx, dtype=torch.int64, device=device, ndim=2)
        _check_npu_tensor('out[1]', topk_weights, dtype=torch.float32, device=device, ndim=2)
        if topk_idx.shape != output_shape:
            raise ValueError(f'out[0] must have shape {output_shape}, got {tuple(topk_idx.shape)}.')
        if topk_weights.shape != output_shape:
            raise ValueError(f'out[1] must have shape {output_shape}, got {tuple(topk_weights.shape)}.')

        if _shares_storage(topk_idx, topk_weights):
            raise ValueError('out[0] and out[1] must not share storage.')
        inputs = (
            ('logits', logits),
            ('mask', mask),
            ('bias', bias),
            ('image_bias', image_bias),
            ('image_token_mask', image_token_mask),
            ('fix_routing_mask', fix_routing_mask),
            ('to_physical_map', to_physical_map),
            ('logical_count', logical_count),
            ('unmapped_topk_idx', unmapped_topk_idx),
            ('force_random', force_random),
        )
        for output_name, output in (('out[0]', topk_idx), ('out[1]', topk_weights)):
            for input_name, input_tensor in inputs:
                if input_tensor is not None and _shares_storage(output, input_tensor):
                    raise ValueError(f'{output_name} and {input_name} must not share storage.')

    if num_tokens == 0:
        return topk_idx, topk_weights

    kernel = get_moe_topk_gate_kernel(
        scoring_type,
        num_topk,
        num_routed_experts,
        mask is not None,
        fix_routing_mask is not None,
        unmapped_topk_idx is not None,
        to_physical_map is not None,
        has_bias,
        has_image_token_mask,
        force_random is not None,
    )
    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    kernel(
        logits,
        bias,
        image_bias,
        image_token_mask,
        mask,
        fix_routing_mask,
        to_physical_map,
        logical_count,
        topk_idx,
        unmapped_topk_idx,
        topk_weights,
        force_random,
        num_shared_experts,
        float(routed_scaling_factor),
        ep_rank,
    )
    return topk_idx, topk_weights
