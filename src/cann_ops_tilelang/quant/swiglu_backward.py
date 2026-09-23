import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import (
    get_token_alignment,
    require_ascend,
    should_print_kernel_source,
)

from .swiglu_backward_kernel import get_swiglu_backward_kernel


def _swiglu_backward_impl(
    x: torch.Tensor,
    act_x_grad: torch.Tensor,
    fmt: str,
    psum_num_tokens_per_expert: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    routed_scaling_factor: float | None = None,
    clamp_value: float | None = None,
    do_recompute: bool = False,
) -> tuple[torch.Tensor, ...]:
    require_ascend()
    if x.device.type != 'npu':
        raise ValueError(f'x must be an NPU tensor, got {x.device}.')
    if act_x_grad.device != x.device:
        raise ValueError('act_x_grad must be on the same NPU device as x.')

    assert fmt in ('bf16', 'fp32')
    assert x.dtype in (torch.bfloat16, torch.float32)
    assert x.dtype == act_x_grad.dtype
    assert (x.dim() == 2 or x.dim() == 3) and x.is_contiguous()
    assert x.size(-1) % 2 == 0

    hidden = x.size(-1) // 2
    x = x.view(-1, hidden * 2)
    act_x_grad = act_x_grad.view(-1, hidden)
    num_expanded_tokens = x.size(0)
    assert act_x_grad.shape == (num_expanded_tokens, hidden)

    if topk_weights is not None:
        assert topk_weights.dim() == 1 and topk_weights.is_contiguous()
        assert topk_weights.dtype == torch.float32
        assert topk_weights.shape == (num_expanded_tokens,)
        assert topk_weights.device == x.device

    if psum_num_tokens_per_expert is not None:
        assert psum_num_tokens_per_expert.dim() == 1
        assert psum_num_tokens_per_expert.dtype == torch.int32
        assert psum_num_tokens_per_expert.is_contiguous()
        assert psum_num_tokens_per_expert.device == x.device
    num_experts = psum_num_tokens_per_expert.shape[0] if psum_num_tokens_per_expert is not None else 0

    if routed_scaling_factor is not None:
        assert topk_weights is not None, 'routed_scaling_factor requires topk_weights'

    out_dtype = torch.bfloat16 if fmt == 'bf16' else torch.float32
    kernel = get_swiglu_backward_kernel(
        hidden,
        num_experts,
        get_token_alignment(),
        topk_weights is not None,
        routed_scaling_factor is not None,
        clamp_value is not None,
        x_dtype=T.dtype(x.dtype),
        act_x_grad_dtype=T.dtype(act_x_grad.dtype),
        out_dtype=T.dtype(out_dtype),
        do_recompute=do_recompute,
    )

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    out = torch.empty((num_expanded_tokens, hidden), dtype=out_dtype, device=x.device) if do_recompute else None
    x_grad = torch.empty((num_expanded_tokens, hidden * 2), dtype=out_dtype, device=x.device)
    weight_grad = (
        torch.empty((num_expanded_tokens,), dtype=torch.float32, device=x.device) if topk_weights is not None else None
    )

    kernel_clamp_value = 0 if clamp_value is None else clamp_value
    kernel_routed_scaling_factor = 1.0 if routed_scaling_factor is None else routed_scaling_factor
    if num_expanded_tokens > 0:
        kernel(
            x,
            act_x_grad,
            topk_weights,
            kernel_routed_scaling_factor,
            psum_num_tokens_per_expert,
            x_grad,
            weight_grad,
            out,
            kernel_clamp_value,
        )

    result = [x_grad]
    if topk_weights is not None:
        result.append(weight_grad)
    if do_recompute:
        result.append(out)
    return tuple(result)


def swiglu_backward(
    x: torch.Tensor,
    act_x_grad: torch.Tensor,
    fmt: str,
    psum_num_tokens_per_expert: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
    clamp_value: float | None = None,
    do_recompute: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Apply the Ascend SwiGLU backward kernel without quantization.

    The return order is ``x_grad``, followed by ``weight_grad`` when routing
    weights are provided, and ``act_out`` when recomputation is requested.
    """

    return _swiglu_backward_impl(
        x,
        act_x_grad,
        fmt,
        psum_num_tokens_per_expert,
        topk_weights,
        routed_scaling_factor,
        clamp_value,
        do_recompute,
    )
