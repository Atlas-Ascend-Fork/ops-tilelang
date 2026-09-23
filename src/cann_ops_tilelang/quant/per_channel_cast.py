import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source
from cann_ops_tilelang.quant.common import (
    QuantTensor,
    alloc_scaling_factors,
    cast_epilogue,
    get_cast_input_and_config,
    get_cast_output_config,
    get_sf_shape,
)

from .per_channel_cast_kernel import get_per_channel_cast_kernel_asc


def per_channel_cast(
    x: torch.Tensor | QuantTensor,
    fmt: str,
    num_per_tokens: int,
    round_sf: bool = False,
    num_per_channels: int | None = None,
    use_packed_ue8m0: bool = False,
) -> QuantTensor:
    """Per-channel cast to FP8 (row-major SF).

    Args:
        x: Input 2D tensor of shape ``(num_tokens, hidden)``, or a ``QuantTensor``
            ``(data, sf_invs)`` for rescaling pre-quantized FP8 inputs.
        fmt: Target FP8 format (must be ``'e4m3'``).
        num_per_tokens: Number of tokens sharing each scaling factor block (``32`` or ``128``).
        round_sf: Whether to round scaling factors to powers of two.
        num_per_channels: Number of channels in each input scaling block. Must be
            ``32`` or ``128`` when ``x`` is a ``QuantTensor``, ``None`` otherwise.
        use_packed_ue8m0: Whether to store sf factors as packed UE8M0 bytes.

    Returns:
        A tuple ``(out, out_sf)`` with FP8 output and row-major sf-factor tensor.
    """
    require_ascend()
    x, x_sf_invs, in_config = get_cast_input_and_config(x, None if num_per_channels is None else (1, num_per_channels))

    assert fmt == 'e4m3'
    assert x.device.type == 'npu'
    assert x.dim() == 2 and x.is_contiguous()
    assert num_per_tokens in (32, 128)
    num_tokens, hidden = x.shape
    assert hidden % 128 == 0, f'Per channel cast requires hidden % 128 == 0, got {hidden}'
    if x_sf_invs is not None:
        assert num_per_channels in (32, 128)
        assert x.dtype == torch.float8_e4m3fn
        assert x_sf_invs.dim() == 2 and x_sf_invs.is_contiguous()
        assert tuple(x_sf_invs.shape) == get_sf_shape((num_tokens, hidden), in_config)
        if in_config.use_tma_aligned_col_major_sf and in_config.use_packed_ue8m0:
            x_sf_invs = x_sf_invs.view(torch.int16)

    out_config = get_cast_output_config(fmt, (num_per_tokens, 1), False, round_sf, use_packed_ue8m0)
    assert num_tokens % 128 == 0
    out = torch.empty((num_tokens, hidden), dtype=out_config.torch_dtype, device=x.device)
    out_sf = alloc_scaling_factors((num_tokens, hidden), out_config, device=x.device)
    kernel = get_per_channel_cast_kernel_asc(hidden, in_config=in_config, out_config=out_config)

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    if num_tokens > 0:
        kernel(x, out, out_sf, x_sf_invs)
    out_sf = cast_epilogue(out_sf, out_config)
    return out, out_sf
