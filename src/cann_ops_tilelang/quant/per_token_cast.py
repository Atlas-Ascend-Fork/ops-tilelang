from typing import NamedTuple

import torch

from cann_ops_tilelang.config import get_amax_clamp_for_quant, require_ascend, should_print_kernel_source

from .common import (
    CastInputConfig,
    CastOutputConfig,
    alloc_scaling_factors,
    cast_epilogue,
    get_cast_input_and_config,
    get_cast_output_config,
    get_cast_output_config_impl,
    get_logical_hidden,
    get_packed_ue8m0_torch_dtype,
    get_physical_hidden,
    get_sf_shape,
)
from .per_token_cast_kernel import get_per_token_cast_kernel_asc, tall_prequant_tile_pays
from .types import QuantTensor


class _PreparedCast(NamedTuple):
    x: torch.Tensor
    x_sf: torch.Tensor | None
    in_config: CastInputConfig
    out_config: CastOutputConfig
    num_tokens: int
    hidden: int


def _validate_per_token_cast(
    x: torch.Tensor | QuantTensor,
    fmt: str,
    num_per_channels: int,
    x_block_size: tuple[int, int] | None,
    use_tma_aligned_col_major_sf: bool,
    round_sf: bool,
    use_packed_ue8m0: bool,
    clamp_min_value: float | None = None,
) -> _PreparedCast:
    """Validate metadata without environment probes, JIT or device execution."""
    assert fmt in ('e4m3', 'e2m1')

    x, x_sf, in_config = get_cast_input_and_config(x, x_block_size)
    if x.device.type != 'npu':
        raise ValueError(f'x must be an NPU tensor, got {x.device}.')
    if clamp_min_value is None:
        out_config = get_cast_output_config(
            fmt, (1, num_per_channels), use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0
        )
    else:
        out_config = get_cast_output_config_impl(
            fmt,
            (1, num_per_channels),
            use_tma_aligned_col_major_sf,
            round_sf,
            use_packed_ue8m0,
            clamp_min_value,
        )

    assert x.dim() == 2
    assert x.stride(1) == 1
    num_tokens, hidden = x.shape
    hidden = get_logical_hidden(hidden, x.dtype)
    assert num_per_channels in (16, 32, 64, 128) or (num_per_channels == hidden and hidden % 64 == 0)
    return _PreparedCast(x, x_sf, in_config, out_config, num_tokens, hidden)


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
    clamp_min_value: float | None = None,
) -> torch.Tensor | QuantTensor:
    require_ascend()
    x, x_sf, in_config, out_config, num_tokens, hidden = _validate_per_token_cast(
        x,
        fmt,
        num_per_channels,
        x_block_size,
        use_tma_aligned_col_major_sf,
        round_sf,
        use_packed_ue8m0,
        clamp_min_value,
    )
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

    # Cast-only does not return SF. Avoid unnecessary dtype views of the
    # read-only caller buffer, especially empty inputs rebuilt by graph capture.
    if sf is not None and not sf_only:
        return out_x
    out_sf = cast_epilogue(out_sf, out_config)

    if sf_only:
        return out_sf
    return out_x, out_sf


def _registered_cast_kwargs(x, fmt, num_per_channels, options, clamp_min_value, x_sf=None, x_block_size=None):
    """Decode graph-schema flags into the existing eager implementation's keywords.

    The bool list groups related static options, in order: TMA layout, scale
    rounding, UE8M0 packing, stochastic casting. Clamp stays an explicit input.
    """
    tma, rounded, packed, stochastic = options
    return dict(
        x=x if x_sf is None else (x, x_sf),
        fmt=fmt,
        num_per_channels=num_per_channels,
        x_block_size=None if x_block_size is None else tuple(x_block_size),
        use_tma_aligned_col_major_sf=tma,
        round_sf=rounded,
        use_packed_ue8m0=packed,
        stochastic_cast=stochastic,
        clamp_min_value=clamp_min_value,
    )


def _prepare_registered_cast(x, fmt, num_per_channels, options, clamp_min_value, x_sf=None, x_block_size=None):
    kwargs = _registered_cast_kwargs(x, fmt, num_per_channels, options, clamp_min_value, x_sf, x_block_size)
    del kwargs['stochastic_cast']
    return _validate_per_token_cast(**kwargs)


def _allocate_cast_data(prepared):
    return torch.empty(
        (prepared.num_tokens, get_physical_hidden(prepared.hidden, prepared.out_config.torch_dtype)),
        dtype=prepared.out_config.torch_dtype,
        device=prepared.x.device,
    )


def _allocate_cast_scale(prepared):
    scale = alloc_scaling_factors((prepared.num_tokens, prepared.hidden), prepared.out_config, prepared.x.device)
    return cast_epilogue(scale, prepared.out_config)


def _validate_scale_output(prepared, out_sf):
    config = prepared.out_config
    assert not config.use_tma_aligned_col_major_sf, 'Not supported yet'
    raw_sf = out_sf.view(torch.uint8) if config.use_packed_ue8m0 else out_sf
    assert raw_sf.shape == get_sf_shape((prepared.num_tokens, prepared.hidden), config)


@torch.library.custom_op('cann_ops_tilelang::per_token_cast', mutates_args=())
def _compiled_per_token_cast(
    x: torch.Tensor,
    x_sf: torch.Tensor | None,
    fmt: str,
    num_per_channels: int,
    x_block_size: list[int] | None,
    options: list[bool],
    clamp_min_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    kwargs = _registered_cast_kwargs(x, fmt, num_per_channels, options, clamp_min_value, x_sf, x_block_size)
    return per_token_cast_impl(**kwargs)


@_compiled_per_token_cast.register_fake
def _per_token_cast_fake(x, x_sf, fmt, num_per_channels, x_block_size, options, clamp_min_value):
    prepared = _prepare_registered_cast(x, fmt, num_per_channels, options, clamp_min_value, x_sf, x_block_size)
    return _allocate_cast_data(prepared), _allocate_cast_scale(prepared)


@torch.library.custom_op('cann_ops_tilelang::per_token_cast_out', mutates_args=('out_x', 'out_sf'))
def _compiled_per_token_cast_out(
    x: torch.Tensor,
    x_sf: torch.Tensor | None,
    fmt: str,
    num_per_channels: int,
    x_block_size: list[int] | None,
    options: list[bool],
    clamp_min_value: float,
    out_x: torch.Tensor,
    out_sf: torch.Tensor,
) -> None:
    kwargs = _registered_cast_kwargs(x, fmt, num_per_channels, options, clamp_min_value, x_sf, x_block_size)
    per_token_cast_impl(**kwargs, out_x=out_x, out_sf=out_sf)


@_compiled_per_token_cast_out.register_fake
def _per_token_cast_out_fake(x, x_sf, fmt, num_per_channels, x_block_size, options, clamp_min_value, out_x, out_sf):
    prepared = _prepare_registered_cast(x, fmt, num_per_channels, options, clamp_min_value, x_sf, x_block_size)
    _validate_scale_output(prepared, out_sf)


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

    The compile path uses custom operators with metadata-only fake implementations.
    The allocation and out-buffer paths have separate mutation contracts. Kernel
    math and stochastic rounding are unchanged. Use fixed-shape npugraph_ex
    execution in inference_mode; no autograd formula is registered.
    """
    if torch.compiler.is_compiling():
        # Flatten QuantTensor only at the internal registration boundary.
        x_data, x_sf = x if isinstance(x, tuple) else (x, None)
        block_size = None if x_block_size is None else list(x_block_size)
        # Explicit configuration input: changing the clamp must not be hidden
        # behind the opaque custom op or a cached configuration object.
        clamp_min_value = get_amax_clamp_for_quant(fmt)
        args = (
            x_data,
            x_sf,
            fmt,
            num_per_channels,
            block_size,
            [use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, stochastic_cast],
            clamp_min_value,
        )
        if out is None:
            return _compiled_per_token_cast(*args)
        out_x, out_sf = out
        _compiled_per_token_cast_out(*args, out_x, out_sf)
        # custom_op cannot return an input alias. Reconstruct the original
        # public return outside that boundary, preserving the packed-SF view.
        returned_sf = out_sf.view(torch.uint8) if use_packed_ue8m0 else out_sf
        if use_packed_ue8m0:
            returned_sf = returned_sf.view(get_packed_ue8m0_torch_dtype())
        return out_x, returned_sf

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


@torch.library.custom_op('cann_ops_tilelang::per_token_cast_with_sf_only', mutates_args=())
def _compiled_per_token_cast_with_sf_only(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    options: list[bool],
    clamp_min_value: float,
) -> torch.Tensor:
    kwargs = _registered_cast_kwargs(x, fmt, num_per_channels, options, clamp_min_value)
    return per_token_cast_impl(**kwargs, sf_only=True)


@_compiled_per_token_cast_with_sf_only.register_fake
def _per_token_cast_with_sf_only_fake(x, fmt, num_per_channels, options, clamp_min_value):
    prepared = _prepare_registered_cast(x, fmt, num_per_channels, options, clamp_min_value)
    return _allocate_cast_scale(prepared)


@torch.library.custom_op('cann_ops_tilelang::per_token_cast_with_sf_only_out', mutates_args=('out_sf',))
def _compiled_per_token_cast_with_sf_only_out(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    options: list[bool],
    clamp_min_value: float,
    out_sf: torch.Tensor,
) -> None:
    kwargs = _registered_cast_kwargs(x, fmt, num_per_channels, options, clamp_min_value)
    per_token_cast_impl(**kwargs, sf_only=True, out_sf=out_sf)


@_compiled_per_token_cast_with_sf_only_out.register_fake
def _per_token_cast_with_sf_only_out_fake(x, fmt, num_per_channels, options, clamp_min_value, out_sf):
    prepared = _prepare_registered_cast(x, fmt, num_per_channels, options, clamp_min_value)
    _validate_scale_output(prepared, out_sf)


@torch.library.custom_op('cann_ops_tilelang::per_token_cast_with_precomputed_sf', mutates_args=())
def _compiled_per_token_cast_with_precomputed_sf(
    x: torch.Tensor,
    sf: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    options: list[bool],
    clamp_min_value: float,
) -> torch.Tensor:
    kwargs = _registered_cast_kwargs(x, fmt, num_per_channels, options, clamp_min_value)
    return per_token_cast_impl(**kwargs, sf=sf)


@_compiled_per_token_cast_with_precomputed_sf.register_fake
def _per_token_cast_with_precomputed_sf_fake(x, sf, fmt, num_per_channels, options, clamp_min_value):
    prepared = _prepare_registered_cast(x, fmt, num_per_channels, options, clamp_min_value)
    return _allocate_cast_data(prepared)


@torch.library.custom_op('cann_ops_tilelang::per_token_cast_with_precomputed_sf_out', mutates_args=('out_x',))
def _compiled_per_token_cast_with_precomputed_sf_out(
    x: torch.Tensor,
    sf: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    options: list[bool],
    clamp_min_value: float,
    out_x: torch.Tensor,
) -> None:
    kwargs = _registered_cast_kwargs(x, fmt, num_per_channels, options, clamp_min_value)
    per_token_cast_impl(**kwargs, sf=sf, out_x=out_x)


@_compiled_per_token_cast_with_precomputed_sf_out.register_fake
def _per_token_cast_with_precomputed_sf_out_fake(x, sf, fmt, num_per_channels, options, clamp_min_value, out_x):
    # Caller owns output storage; only validate input metadata here.
    _prepare_registered_cast(x, fmt, num_per_channels, options, clamp_min_value)


def per_token_cast_with_sf_only(
    x: torch.Tensor,
    fmt: str,
    num_per_channels: int,
    use_tma_aligned_col_major_sf: bool = False,
    round_sf: bool = False,
    use_packed_ue8m0: bool = False,
    out_sf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return only the per-token scaling factors for an FP8/FP4 cast.

    Compile registration preserves allocation and caller-owned output modes.
    Fixed-shape inference graph execution is supported with npugraph_ex.
    """
    if torch.compiler.is_compiling():
        args = (
            x,
            fmt,
            num_per_channels,
            [use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, False],
            get_amax_clamp_for_quant(fmt),
        )
        if out_sf is None:
            return _compiled_per_token_cast_with_sf_only(*args)
        _compiled_per_token_cast_with_sf_only_out(*args, out_sf)
        # Return the same storage/view as the eager SF epilogue, outside the op.
        if use_packed_ue8m0:
            return out_sf.view(torch.uint8).view(get_packed_ue8m0_torch_dtype())
        return out_sf
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
    """Cast with caller-provided per-token scaling factors.

    The supplied sf remains read-only; only a caller-provided out is mutated.
    sf uses the raw kernel layout: transpose a TMA-aligned public scale tensor
    first, then view packed UE8M0 storage as uint8 when packing is enabled.
    Fixed-shape inference graph execution is supported with npugraph_ex.
    """
    if torch.compiler.is_compiling():
        args = (
            x,
            sf,
            fmt,
            num_per_channels,
            [use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0, False],
            get_amax_clamp_for_quant(fmt),
        )
        if out is None:
            return _compiled_per_token_cast_with_precomputed_sf(*args)
        _compiled_per_token_cast_with_precomputed_sf_out(*args, out)
        return out
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
