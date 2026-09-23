import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import (
    get_num_vector_cores,
    get_token_alignment,
    require_ascend,
    should_print_kernel_source,
)

from .common import alloc_scaling_factors, cast_epilogue, get_cast_output_config, get_physical_hidden
from .swiglu_forward_kernel import (
    get_swiglu_forward_and_per_token_cast_kernel,
    get_swiglu_forward_kernel,
)
from .types import QuantTensor


def swiglu_forward_and_per_token_cast_impl(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int | None,
    psum_num_tokens_per_expert: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    clamp_value: float | None = None,
    clamped_count: torch.Tensor | None = None,
) -> QuantTensor | torch.Tensor:
    require_ascend()
    if x.device.type != 'npu':
        raise ValueError(f'x must be an NPU tensor, got {x.device}.')
    assert x.dim() == 2 and x.is_contiguous()
    num_expanded_tokens, hidden = x.shape
    assert hidden % 2 == 0
    hidden = hidden // 2
    if psum_num_tokens_per_expert is not None:
        assert psum_num_tokens_per_expert.dim() == 1
    num_experts = psum_num_tokens_per_expert.shape[0] if psum_num_tokens_per_expert is not None else 0

    if clamped_count is not None:
        assert clamp_value is not None
        assert clamped_count.dim() == 1
        assert clamped_count.shape[0] == 4

    if routed_scaling_factor is not None:
        assert topk_weights is not None, 'routed_scaling_factor requires topk_weights'

    # Swiglu forward : hidden -> hidden // 2
    assert (fmt, num_per_channels) in [
        ('e4m3', 128),
        ('e4m3', 32),
        ('e2m1', 128),
        ('e2m1', 32),
        ('bf16', None),
        ('fp32', None),
    ]

    out_config = get_cast_output_config(
        fmt,
        (1, num_per_channels if num_per_channels is not None else 1),
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
    )

    get_kernel = get_swiglu_forward_and_per_token_cast_kernel if out_config.with_sf else get_swiglu_forward_kernel
    kernel = get_kernel(
        hidden,
        num_experts,
        get_token_alignment(),
        topk_weights is not None,
        routed_scaling_factor is not None,
        clamp_value,
        clamped_count is not None,
        in_dtype=T.dtype(x.dtype),
        out_config=out_config,
        num_cores=get_num_vector_cores(),
    )

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    # Allocate output and launch
    out = torch.empty(
        (num_expanded_tokens, get_physical_hidden(hidden, out_config.torch_dtype)),
        dtype=out_config.torch_dtype,
        device=x.device,
    )
    out_sf = (
        alloc_scaling_factors((num_expanded_tokens, hidden), out_config, device=x.device)
        if out_config.with_sf
        else None
    )

    if num_expanded_tokens > 0:
        routed_scaling_factor = routed_scaling_factor if routed_scaling_factor is not None else 1.0
        kernel(x, out, out_sf, psum_num_tokens_per_expert, topk_weights, routed_scaling_factor, clamped_count)

    if out_config.with_sf:
        out_sf = cast_epilogue(out_sf, out_config)
        return out, out_sf
    else:
        return out


def swiglu_forward_and_per_token_cast(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    psum_num_tokens_per_expert: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    clamp_value: float | None = None,
    clamped_count: torch.Tensor | None = None,
) -> QuantTensor:
    """Fuse SwiGLU forward"""
    return swiglu_forward_and_per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        psum_num_tokens_per_expert,
        topk_weights,
        routed_scaling_factor,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        clamp_value,
        clamped_count,
    )


def swiglu_forward(
    x: torch.Tensor,
    fmt: str,
    psum_num_tokens_per_expert: torch.Tensor | None = None,
    topk_weights: torch.Tensor | None = None,
    routed_scaling_factor: float | None = None,
    clamp_value: float | None = None,
    clamped_count: torch.Tensor | None = None,
) -> torch.Tensor:
    assert fmt in ('fp32', 'bf16'), 'swiglu_forward only supports fp32 or bf16 output'
    return swiglu_forward_and_per_token_cast_impl(
        x,
        fmt,
        None,
        psum_num_tokens_per_expert,
        topk_weights,
        routed_scaling_factor=routed_scaling_factor,
        clamp_value=clamp_value,
        clamped_count=clamped_count,
    )
