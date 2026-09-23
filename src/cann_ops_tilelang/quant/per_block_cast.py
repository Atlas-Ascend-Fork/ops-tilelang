import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source
from cann_ops_tilelang.quant.common import (
    QuantTensor,
    alloc_scaling_factors,
    cast_epilogue,
    get_cast_input_and_config,
    get_cast_output_config,
)

from .per_block_cast_kernel import get_per_block_cast_kernel_asc


def per_block_cast_impl(
    x: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    sf_only: bool = False,
    sf: torch.Tensor | None = None,
) -> torch.Tensor | QuantTensor:
    """
        sf_only: If True, only compute and return sf factors.
        sf: Pre-computed sf factors; if provided, cast-only mode is used.

    Returns:
        Scale-factor tensor when ``sf_only=True``, casted tensor when ``sf``
        is provided, or a tuple ``(out, out_sf)`` with quantized output and sf factors.
    """
    require_ascend()
    assert x.device.type == 'npu'
    assert x.is_contiguous() and x.dim() == 2
    num_tokens, hidden = x.shape

    assert num_tokens % 128 == 0
    assert hidden % 128 == 0
    if sf is not None:
        assert not sf_only and not use_tma_aligned_col_major_sf and not use_packed_ue8m0

    x, _, in_config = get_cast_input_and_config(x, None)
    out_config = get_cast_output_config(fmt, block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)

    kernel = get_per_block_cast_kernel_asc(
        hidden=hidden,
        in_config=in_config,
        out_config=out_config,
        sf_only=sf_only,
        cast_only=(sf is not None),
    )

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    out = torch.empty(
        (num_tokens, hidden if fmt == 'e4m3' else hidden // 2), dtype=out_config.torch_dtype, device=x.device
    )
    out_sf = (
        sf
        if sf is not None
        else alloc_scaling_factors(
            (num_tokens, hidden),
            out_config=out_config,
            device=x.device,
        )
    )

    if num_tokens > 0:
        kernel(x, out, out_sf)

    out_sf = cast_epilogue(out_sf, out_config)

    if sf_only:
        return out_sf

    if sf is not None:
        return out

    return out, out_sf


def per_block_cast(
    x: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> QuantTensor:
    """Cast a matrix to FP8/FP4 with per-block scaling factors.

    Args:
        x: Input 2D contiguous tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        block_size: Scaling block size as (num_per_tokens, num_per_channels).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
       Tuple of quantized output and sf factors.
    """
    return per_block_cast_impl(x, fmt, block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0)


def per_block_cast_with_sf_only(
    x: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4, only output the scaling factors.

    Args:
        x: Input 2D contiguous tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        block_size: Scaling block size as (num_per_tokens, num_per_channels).
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        Scaling factors only
    """
    return per_block_cast_impl(
        x, fmt, block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, sf_only=True
    )


def per_block_cast_with_precomputed_sf(
    x: torch.Tensor,
    fmt: str,
    block_size: tuple[int, int],
    sf: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor:
    """Cast a matrix to FP8/FP4 with per-block scaling using precomputed sf factors.

    Args:
        x: Input 2D contiguous tensor of shape (num_tokens, hidden).
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        block_size: Scaling block size as (num_per_tokens, num_per_channels).
        sf: Pre-computed sf factors.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.

    Returns:
        Casted tensor using the precomputed sf factors.
    """
    return per_block_cast_impl(x, fmt, block_size, use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, sf=sf)
