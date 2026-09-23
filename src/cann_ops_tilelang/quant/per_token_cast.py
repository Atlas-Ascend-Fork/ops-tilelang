import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .common import (
    alloc_scaling_factors,
    cast_epilogue,
    get_cast_input_and_config,
    get_cast_output_config,
    get_logical_hidden,
    get_physical_hidden,
    get_sf_shape,
)
from .per_token_cast_kernel import get_per_token_cast_kernel_asc, tall_prequant_tile_pays
from .types import QuantTensor


def per_token_cast_impl(
    x: torch.Tensor | QuantTensor,
    fmt: str,
    num_per_channels: int,
    x_block_size: tuple[int, int] | None = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    sf_only: bool = False,
    stochastic_cast: bool = False,
    sf: torch.Tensor | None = None,
    out_x: torch.Tensor | None = None,
    out_sf: torch.Tensor | None = None,
) -> torch.Tensor | QuantTensor:
    require_ascend()
    assert fmt in ('e4m3', 'e2m1')

    x, x_sf, in_config = get_cast_input_and_config(x, x_block_size)
    if x.device.type != 'npu':
        raise ValueError(f'x must be an NPU tensor, got {x.device}.')
    out_config = get_cast_output_config(
        fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
    )

    assert x.dim() == 2
    assert x.stride(1) == 1
    num_tokens, hidden = x.shape
    hidden = get_logical_hidden(hidden, x.dtype)
    assert num_per_channels in (16, 32, 64, 128) or (num_per_channels == hidden and hidden % 64 == 0)
    assert sf is None or out_sf is None

    # The kernel is built for a hidden size but not for a token count, so
    # the tile-shape question it cannot answer is answered here.
    kernel = get_per_token_cast_kernel_asc(
        hidden=hidden,
        token_stride=get_logical_hidden(x.stride(0), x.dtype),
        in_config=in_config,
        out_config=out_config,
        sf_only=sf_only,
        cast_only=sf is not None,
        stochastic_cast=stochastic_cast,
        small_batch=not tall_prequant_tile_pays(num_tokens, hidden),
    )

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    if out_x is None:
        out_x = torch.empty(
            (num_tokens, get_physical_hidden(hidden, out_config.torch_dtype)),
            dtype=out_config.torch_dtype,
            device=x.device,
        )
    if out_sf is None:
        out_sf = (
            sf
            if sf is not None
            else alloc_scaling_factors(
                (num_tokens, hidden),
                out_config=out_config,
                device=x.device,
            )
        )
    else:
        sf_shape = get_sf_shape((num_tokens, hidden), out_config)
        assert not use_tma_aligned_col_major_sf, 'Not supported yet'
        if use_packed_ue8m0:
            out_sf = out_sf.view(dtype=torch.uint8)
        assert out_sf.shape == sf_shape

    if num_tokens > 0:
        kernel(x, x_sf, out_x, out_sf)

    out_sf = cast_epilogue(out_sf, out_config)

    if sf_only:
        return out_sf
    if sf is not None:
        return out_x
    return out_x, out_sf


def per_token_cast(
    x: torch.Tensor | QuantTensor,
    fmt: str,
    num_per_channels: int,
    x_block_size: tuple[int, int] | None = None,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    stochastic_cast: bool = False,
    out: QuantTensor | None = None,
) -> QuantTensor:
    """Cast a matrix to FP8/FP4 with per-token (row-wise) scaling factors.

    Args:
        x: Input 2D tensor or a tuple of 2D tensor and 2D scaling factors.
        fmt: Target quantized format (``'e4m3'`` or ``'e2m1'``).
        num_per_channels: Number of channels in each scaling block.
        x_block_size: Input scaling block size for pre-quantized inputs.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf factors.
        round_sf: Whether to round scaling factors to powers of two.
        use_packed_ue8m0: Whether to use packed UE8M0 format for sf factors.
        stochastic_cast: Whether to use stochastic rounding for the narrowing cast.
        out: Optional pre-allocated QuantTensor (data, sf).

    Returns:
        A tuple ``(out, out_sf)`` with quantized output and sf factors.
    """
    out_x, out_sf = (None, None) if out is None else out
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        x_block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        stochastic_cast=stochastic_cast,
        out_x=out_x,
        out_sf=out_sf,
    )


def per_token_cast_with_sf_only(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    out_sf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return only the per-token scaling factors for an FP8/FP4 cast."""
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        None,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        sf_only=True,
        out_sf=out_sf,
    )


def per_token_cast_with_precomputed_sf(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    sf: torch.Tensor,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cast with caller-provided per-token scaling factors."""
    return per_token_cast_impl(
        x,
        fmt,
        num_per_channels,
        None,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        sf=sf,
        out_x=out,
    )
