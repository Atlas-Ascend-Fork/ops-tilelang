import torch
import torch.nn.functional as F

from cann_ops_tilelang.config import get_amax_clamp_for_quant
from cann_ops_tilelang.quant.common import (
    BaseCastConfig,
    get_packed_ue8m0_pack_factor,
    get_packed_ue8m0_torch_dtype,
    unpack_from_e2m1fn_x2,
)
from cann_ops_tilelang.quant.types import QuantTensor
from cann_ops_tilelang.utils import align_up, ceil_div

# Bound temporary tensor usage when several NPU workers evaluate large FP4 references concurrently.
_E2M1_REF_CHUNK_NUMEL = 8 * 1024 * 1024


def right_shift_unsigned(x, shift: int):
    # Mask the shifted value to preserve unsigned semantics.
    return (x >> shift) & (0xFFFFFFFF >> shift)


def get_max_quant_val(dtype: torch.dtype):
    max_quant_by_dtype = {torch.int8: 6.0, torch.float8_e4m3fn: 448.0}
    assert dtype in max_quant_by_dtype
    return max_quant_by_dtype[dtype]


def transform_sf(sf: torch.Tensor, input_config: BaseCastConfig | None = None) -> torch.Tensor:
    if sf.dtype == torch.float32:
        return sf
    assert sf.dtype == get_packed_ue8m0_torch_dtype()
    num_rows, num_cols = sf.shape
    sf = sf.contiguous()
    if sf.stride(-1) != 1:
        sf = sf.as_strided(size=sf.shape, stride=(sf.shape[-1], 1))
    sf = sf.view(torch.uint8)
    sf = sf.to(torch.int32)
    sf = (sf * 0x00800000).view(torch.float32)
    if input_config is not None and input_config.sf_col_pack:
        pack_factor = get_packed_ue8m0_pack_factor()
        sf = sf.view(num_rows, num_cols, pack_factor).permute(0, 2, 1).reshape(num_rows * pack_factor, num_cols)
    return sf


def cast_back(x: QuantTensor, fmt: str, block_size: tuple[int, int] = (32, 32)) -> torch.Tensor:
    """Dequantize an FP8/FP4 tensor back to BF16 or FP32 (PyTorch reference).

    Args:
        x: Quantized tensor pair ``(data, sf_factors)``.
        fmt: Target output format, either ``'bf16'`` or ``'fp32'``.
        block_size: Scaling block size as ``(block_h, block_w)``.

    Returns:
        Dequantized tensor in the requested format.
    """
    input_tensor, input_sf = x
    assert input_tensor.dtype in (torch.float8_e4m3fn, torch.int8)
    # Expand x_sf and do elementwise multiply with x
    input_config = BaseCastConfig(
        input_tensor.dtype, block_size, use_packed_ue8m0=input_sf.dtype == get_packed_ue8m0_torch_dtype()
    )
    input_sf = transform_sf(input_sf, input_config)
    input_sf = input_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1)
    input_tensor = (
        unpack_from_e2m1fn_x2(input_tensor) if input_tensor.dtype == torch.int8 else input_tensor.to(torch.float32)
    )
    input_sf = input_sf[: input_tensor.shape[0], : input_tensor.shape[1]]
    x = input_tensor * input_sf
    out_dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[fmt]
    return x.to(dtype=out_dtype)


def cast(
    x: torch.Tensor | QuantTensor,
    fmt: str,
    block_size: tuple[int, int] = (32, 32),
    sf: torch.Tensor | None = None,
    x_block_size: tuple[int, int] | None = None,
    round_sf: bool = False,
    use_tma_aligned_col_major_sf: bool = False,
    use_packed_ue8m0: bool = False,
) -> torch.Tensor | QuantTensor:
    """Cast a 2D tensor to MX format with 2D block-shared sf.

    Args:
        x: Input 2D tensor (H, W), optionally with scaling factors.
        fmt: Target quantization format, e.g. "e4m3".
        block_size: Block shape (block_h, block_w), e.g. (32, 32).
        sf: Optional precomputed dequant sf.
        x_block_size: Input block shape when `x_packed` carries sf.
        round_sf: Whether to round sf.
        use_tma_aligned_col_major_sf: Whether to use TMA-aligned column-major sf.
        use_packed_ue8m0: Whether to store sf in packed UE8M0.

    Returns:
        Quantized tensor (packed on the last dim for mxfp4) and optional sf.
    """
    has_input_sf = isinstance(x, tuple)
    if has_input_sf:
        input_tensor, input_sf = x
        assert input_tensor.dtype in (torch.float8_e4m3fn, torch.int8)
        assert input_sf is not None and x_block_size is not None
        # Expand x_sf and do elementwise multiply with x
        input_config = BaseCastConfig(
            input_tensor.dtype, x_block_size, use_packed_ue8m0=input_sf.dtype == get_packed_ue8m0_torch_dtype()
        )
        input_sf = transform_sf(input_sf, input_config)
        input_sf = input_sf.repeat_interleave(x_block_size[0], dim=0).repeat_interleave(x_block_size[1], dim=1)
        input_tensor = (
            unpack_from_e2m1fn_x2(input_tensor) if input_tensor.dtype == torch.int8 else input_tensor.to(torch.float32)
        )
        input_sf = input_sf[: input_tensor.shape[0], : input_tensor.shape[1]]
        x = input_tensor * input_sf
    else:
        assert x.dtype in (torch.bfloat16, torch.float16, torch.float32)

    assert x.ndim == 2, 'Only 2D tensors are supported for 2D blocking.'
    h, w = x.shape
    bh, bw = block_size
    pack_sf_along_m = use_packed_ue8m0 and bw == 1

    out_dtype = {'e2m1': torch.int8, 'e4m3': torch.float8_e4m3fn}[fmt]
    max_quant_val = get_max_quant_val(out_dtype)
    is_fp4 = out_dtype == torch.int8
    device = x.device

    if h == 0:
        out_w = w // 2 if is_fp4 else w
        out_weight = torch.empty((0, out_w), dtype=out_dtype, device=device)
        if sf is not None:
            return out_weight
        sf_h = 0
        dq_sf_dtype = torch.float32
        sf_w = (w + bw - 1) // bw
        if use_packed_ue8m0:
            dq_sf_dtype = get_packed_ue8m0_torch_dtype()
            if pack_sf_along_m:
                sf_h = ceil_div(sf_h, get_packed_ue8m0_pack_factor())
            else:
                sf_w = ceil_div(sf_w, get_packed_ue8m0_pack_factor())
        if use_tma_aligned_col_major_sf:
            dq_sf = torch.empty((sf_w, sf_h), dtype=dq_sf_dtype, device=device).T
        else:
            dq_sf = torch.empty((sf_h, sf_w), dtype=dq_sf_dtype, device=device)
        return out_weight, dq_sf

    if is_fp4:
        assert w % 2 == 0, 'For mxfp4, the width must be even for packing.'

    # 1) Padding
    pad_h = (bh - h % bh) % bh
    pad_w = (bw - w % bw) % bw

    # F.pad args are (left, right, top, bottom).
    padded_src = F.pad(x.to(torch.float32), (0, pad_w, 0, pad_h))
    valid_mask = F.pad(torch.ones_like(x, dtype=torch.bool), (0, pad_w, 0, pad_h))

    ph, pw = padded_src.shape

    if sf is None:
        # 2) Block-wise max: (Hb, bh, Wb, bw) -> (Hb, Wb, bh*bw)
        reshaped_for_max = (
            padded_src.view(ph // bh, bh, pw // bw, bw).permute(0, 2, 1, 3).reshape(ph // bh, pw // bw, -1)
        )
        reshaped_mask = valid_mask.view(ph // bh, bh, pw // bw, bw).permute(0, 2, 1, 3).reshape(ph // bh, pw // bw, -1)

        abs_f = torch.abs(reshaped_for_max)
        # Exclude padded positions from max.
        abs_f = torch.where(reshaped_mask, abs_f, torch.tensor(-1.0, device=device, dtype=abs_f.dtype))
        max_val, _ = abs_f.max(dim=-1, keepdim=True)  # shape: (ph/bh, pw/bw, 1)
        max_val = torch.clamp(max_val, min=get_amax_clamp_for_quant(fmt))

        # 3) Compute sf.
        assert max_val.dtype == torch.float32
        # Fill explicitly so the reference follows the kernel's reciprocal order.
        max_quant_val_expanded = torch.full_like(max_val, max_quant_val, dtype=torch.float32)
        dequant_sf = max_val / max_quant_val_expanded

        ds_int = dequant_sf.view(torch.int32)
        # Upcast to ue8m0
        if round_sf:
            ds_int_rounded = (ds_int + 0x007FFFFF) & 0x7F800000
            dequant_sf_rounded = ds_int_rounded.view(torch.float32)
            quant_sf = torch.where(dequant_sf_rounded == 0, torch.tensor(0.0, device=device), 1.0 / dequant_sf_rounded)
        else:
            ds_int_rounded = ds_int
            quant_sf = torch.where(
                ds_int_rounded == 0, torch.tensor(0.0, device=device), max_quant_val_expanded / max_val
            )
    else:
        assert not use_packed_ue8m0 and not use_tma_aligned_col_major_sf
        expected_sf_shape = (ph // bh, pw // bw)
        assert sf.ndim == 2, f'sf must be 2D, got {sf.ndim}D'
        assert tuple(sf.shape) == expected_sf_shape, (
            f'sf shape mismatch: expected {expected_sf_shape}, got {tuple(sf.shape)}'
        )
        quant_sf = sf.reciprocal().unsqueeze(-1)

    # 4) Quantize data.
    if has_input_sf:
        quant_sf_extended = (
            quant_sf.repeat_interleave(block_size[0], dim=0).repeat_interleave(block_size[1], dim=1).squeeze(-1)
        )
        quant_sf_extended = quant_sf_extended[:h, :w]
        quant_tensor = x * quant_sf_extended
    else:
        # Map to block layout: (Hb, bh, Wb, bw) * (Hb, 1, Wb, 1)
        padded_src_view = padded_src.view(ph // bh, bh, pw // bw, bw)
        quant_sf_view = quant_sf.view(ph // bh, 1, pw // bw, 1)

        assert padded_src_view.dtype == torch.float32 and quant_sf_view.dtype == torch.float32
        quant_tensor = (padded_src_view * quant_sf_view).reshape(ph, pw)
        # Crop back to original shape.
        quant_tensor = quant_tensor[:h, :w]

    # 5) Cast type and pack FP4.
    if not is_fp4:
        quant_tensor = torch.clamp(quant_tensor, -max_quant_val, max_quant_val)
        out_weight = quant_tensor.to(out_dtype)
    else:
        e2m1_value = convert_to_e2m1_bits(quant_tensor, max_quant_val, device)
        # Pack (H, W) -> (H, W/2)
        e2m1_value = e2m1_value.view(h, w // 2, 2)
        out_weight = e2m1_value[..., 0] | (e2m1_value[..., 1] * 16)
        out_weight = out_weight.view(torch.int8)

    if sf is not None:
        return out_weight

    # 6) Format sf output.
    ds_int_rounded = ds_int_rounded.squeeze(-1)
    tma_alignment = 1
    pack_factor = get_packed_ue8m0_pack_factor()
    pad_h = (
        align_up(ds_int_rounded.shape[0], pack_factor if pack_sf_along_m else tma_alignment) - ds_int_rounded.shape[0]
    )
    pad_w = (
        align_up(ds_int_rounded.shape[1], pack_factor if use_packed_ue8m0 and not pack_sf_along_m else 1)
        - ds_int_rounded.shape[1]
    )
    ds_int_rounded_padded = F.pad(ds_int_rounded, (0, pad_w, 0, pad_h))
    if use_packed_ue8m0:
        if pack_sf_along_m:
            assert not use_tma_aligned_col_major_sf
            ue8 = (ds_int_rounded_padded >> 23).to(torch.uint8)
            dq_sf = (
                ue8.view(ue8.shape[0] // pack_factor, pack_factor, ue8.shape[1])
                .permute(0, 2, 1)
                .contiguous()
                .view(get_packed_ue8m0_torch_dtype())
                .view(ue8.shape[0] // pack_factor, ue8.shape[1])
            )
        else:
            dq_sf = (ds_int_rounded_padded >> 23).to(torch.int8).view(get_packed_ue8m0_torch_dtype())  # (H/bh, W/bw)
    else:
        dq_sf = ds_int_rounded_padded.view(torch.float32)
    if use_tma_aligned_col_major_sf:
        dq_sf = dq_sf.T.contiguous()
        # [1,512].T is already contiguous, so .contiguous() won't set stride[1] to 1; force it
        dq_sf = dq_sf.as_strided(dq_sf.shape, (dq_sf.shape[1], 1))
        dq_sf = dq_sf.T[: ds_int_rounded.shape[0], :]

    return out_weight, dq_sf


def _convert_to_e2m1_bits_chunk(quant_tensor, device):
    q_int = quant_tensor.contiguous().view(torch.int32)
    signs = q_int & 0x80000000
    exponents = (q_int >> 23) & 0xFF
    mantissas_orig = q_int & 0x7FFFFF

    e8_bias, e2_bias = 127, 1
    # Adjust mantissas for subnormals.
    is_subnormal = exponents < e8_bias
    shift = (e8_bias - exponents - 1).clamp(min=0, max=31)
    mantissas_pre = 0x400000 | right_shift_unsigned(mantissas_orig, 1)
    bit0_dropped = (mantissas_orig & 0x1) != 0
    # Avoid the unsupported dynamic left shift on Ascend; mantissas_pre uses 23 bits.
    mask = torch.full_like(shift, 0x7FFFFF) >> (23 - shift.clamp(max=23))
    dropped_post = (mantissas_pre & mask) != 0
    sticky = is_subnormal & (bit0_dropped | dropped_post)
    mantissas = torch.where(is_subnormal, mantissas_pre >> shift, mantissas_orig)
    exponents = torch.maximum(exponents, torch.tensor(e8_bias - e2_bias, device=device)) - (e8_bias - e2_bias)
    # Round to nearest, ties to even (RTNE)
    m2bits = right_shift_unsigned(mantissas, 21) & 0x3
    lsb_keep = right_shift_unsigned(m2bits, 1) & 0x1
    guard = m2bits & 0x1
    sticky |= (mantissas & 0x1FFFFF) != 0
    round_inc = guard & (sticky.to(torch.int32) | lsb_keep)
    e2m1_tmp = right_shift_unsigned(((exponents * 4) | m2bits) + round_inc, 1)
    e2m1_tmp = torch.minimum(e2m1_tmp, torch.tensor(0x7, device=device))
    e2m1_value = (right_shift_unsigned(signs, 28) | e2m1_tmp).to(torch.uint8)  # shape: (..., even_axis_shape)

    return e2m1_value


def convert_to_e2m1_bits(quant_tensor, max_quant_val, device):
    """FP4 cast with bounded reference-memory usage."""
    del max_quant_val

    if quant_tensor.numel() == 0:
        return torch.empty(quant_tensor.shape, dtype=torch.uint8, device=quant_tensor.device)
    if quant_tensor.numel() <= _E2M1_REF_CHUNK_NUMEL:
        return _convert_to_e2m1_bits_chunk(quant_tensor, device)

    e2m1_value = torch.empty(quant_tensor.shape, dtype=torch.uint8, device=quant_tensor.device)
    rows_per_chunk = max(1, _E2M1_REF_CHUNK_NUMEL // quant_tensor.shape[1])
    for row_start in range(0, quant_tensor.shape[0], rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, quant_tensor.shape[0])
        chunk_value = _convert_to_e2m1_bits_chunk(quant_tensor[row_start:row_end], device)
        e2m1_value[row_start:row_end].copy_(chunk_value)

    return e2m1_value
