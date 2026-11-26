import torch

from cann_ops_tilelang.attention.cache_kernel import (
    _load_index_kv_cache_quant_tl_ascend,
    _load_kv_cache_llama_tl_ascend,
    _load_kv_cache_per32_tl_ascend,
    _load_kv_cache_tl_ascend,
    _store_index_kv_cache_quant_tl_ascend,
    _store_kv_cache_llama_tl_ascend,
    _store_kv_cache_per32_tl_ascend,
    _store_kv_cache_tl_ascend,
)
from cann_ops_tilelang.config import get_num_vector_cores, require_ascend
from cann_ops_tilelang.quant import per_token_cast
from cann_ops_tilelang.quant.common import CastOutputConfig, alloc_scaling_factors, cast_epilogue


def _validate_npu_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != 'npu':
        raise ValueError(f'{name} must be on an NPU, got {tensor.device}.')


def _validate_slot_mapping(slot_mapping: torch.Tensor, expected_tokens: int | None = None) -> None:
    _validate_npu_tensor(slot_mapping, 'slot_mapping')
    if slot_mapping.ndim != 1 or slot_mapping.dtype != torch.int32:
        raise ValueError('slot_mapping must be a 1D torch.int32 tensor.')
    if expected_tokens is not None and slot_mapping.numel() != expected_tokens:
        raise ValueError(f'slot_mapping must contain {expected_tokens} entries, got {slot_mapping.numel()}.')


def _validate_cache(cache: torch.Tensor, slot_mapping: torch.Tensor, *, dtype: torch.dtype) -> None:
    _validate_npu_tensor(cache, 'kv_cache')
    if cache.device != slot_mapping.device:
        raise ValueError('kv_cache and slot_mapping must be on the same device.')
    if cache.ndim != 4 or cache.dtype != dtype:
        raise ValueError(f'kv_cache must be a 4D {dtype} tensor.')


def _validate_store_input(kv, slot_mapping, hidden_size):
    _validate_npu_tensor(kv, 'kv')
    _validate_slot_mapping(slot_mapping)
    if kv.ndim != 3 or kv.shape[-1] != hidden_size or kv.dtype != torch.bfloat16:
        raise ValueError(f'kv must be a 3D BF16 tensor with last dimension {hidden_size}.')
    if kv.shape[0] != slot_mapping.numel() or kv.device != slot_mapping.device:
        raise ValueError('kv and slot_mapping must have matching token counts and devices.')


def _validate_quant_cache(kv_cache, slot_mapping, bytes_per_token):
    _validate_cache(kv_cache, slot_mapping, dtype=torch.uint8)
    if kv_cache.shape[2:] != (1, bytes_per_token):
        raise ValueError(f'kv_cache must have trailing shape (1, {bytes_per_token}).')
    if kv_cache.shape[0] <= 0 or kv_cache.shape[1] <= 0:
        raise ValueError('kv_cache block dimensions must be positive.')


def _validate_kv_metadata(kv_cache, slot_mapping, use_fp8_rope=False, num_per_channels=64):
    _validate_slot_mapping(slot_mapping)
    if kv_cache is None:
        return
    if not use_fp8_rope and num_per_channels == 64:
        size = 584
    elif use_fp8_rope and num_per_channels == 32:
        size = 528
    else:
        raise ValueError('Supported (use_fp8_rope, num_per_channels) pairs are (False, 64) and (True, 32).')
    _validate_quant_cache(kv_cache, slot_mapping, size)


def _validate_index_metadata(kv_cache, slot_mapping, is_fp4=False, is_mxfp8=False):
    _validate_slot_mapping(slot_mapping)
    if kv_cache is not None:
        _validate_quant_cache(kv_cache, slot_mapping, 68 if is_fp4 else 132)


def _validate_llama_metadata(kv_cache, n_head_kv, slot_mapping):
    _validate_slot_mapping(slot_mapping)
    if n_head_kv <= 0:
        raise ValueError('n_head_kv must be positive.')
    if kv_cache is not None:
        _validate_cache(kv_cache, slot_mapping, dtype=torch.bfloat16)
        if kv_cache.shape[2:] != (n_head_kv, 128):
            raise ValueError('kv_cache heads must match n_head_kv, with head size 128.')
        if kv_cache.shape[0] <= 0 or kv_cache.shape[1] <= 0:
            raise ValueError('kv_cache block dimensions must be positive.')


def _allocate_index_result(slot_mapping, is_fp4, is_mxfp8, *, none_cache=False):
    n = slot_mapping.numel()
    data = torch.empty(
        (n, 64 if is_fp4 else 128), dtype=torch.int8 if is_fp4 else torch.float8_e4m3fn, device=slot_mapping.device
    )
    if none_cache:
        # The random fallback preserves per_token_cast's packed int16 layout,
        # including the FP4 column-major stride. It differs from cache records.
        output_config = CastOutputConfig(
            torch_dtype=data.dtype,
            sf_block=(1, 32 if is_fp4 or is_mxfp8 else 128),
            use_tma_aligned_col_major_sf=is_fp4,
            use_packed_ue8m0=is_fp4 or is_mxfp8,
            round_sf=True,
        )
        scale = alloc_scaling_factors((n, 128), output_config, slot_mapping.device)
        scale = cast_epilogue(scale, output_config).squeeze(-1)
    else:
        scale = torch.empty(
            (n,), dtype=torch.int32 if is_fp4 or is_mxfp8 else torch.float32, device=slot_mapping.device
        )
    return data, scale


def _store_kv_cache_64(kv, kv_cache, slot_mapping) -> None:
    num_blocks, block_size, constant_1, bytes_per_token = kv_cache.shape
    assert constant_1 == 1
    assert bytes_per_token == 448 + 64 * 2 + 8
    kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
    kv_cache_value = kv_cache[:, : block_size * (448 + 64 * 2)].view(num_blocks, block_size, 448 + 64 * 2)
    kv_cache_sf = kv_cache[:, slice(block_size * (448 + 64 * 2), None)]
    kv_cache_fp8_raw = kv_cache_value[:, :, :448]
    kv_cache_fp8 = kv_cache_fp8_raw.view(torch.float8_e4m3fn)
    kv_cache_bf16 = kv_cache_value[:, :, 448:]
    kv_cache_bf16 = kv_cache_bf16.view(*kv_cache_bf16.shape[:-1], -1).view(torch.bfloat16)
    kv_cache_sf = kv_cache_sf.view(num_blocks, block_size, 8)
    assert kv_cache_fp8.stride(0) == kv_cache_bf16.stride(0) * 2
    assert kv_cache_fp8.stride(0) == kv_cache_sf.stride(0)
    num_stages = 2
    n_tokens = kv.shape[0]
    if n_tokens % 64 == 0:
        store_block_m = 64
    elif n_tokens % 32 == 0:
        store_block_m = 32
    else:
        store_block_m = 16
    full_tiles = n_tokens % store_block_m == 0
    store_kernel = _store_kv_cache_tl_ascend(
        block_size,
        num_stages=num_stages,
        block_m=store_block_m,
        cache_block_stride=kv_cache.stride(0),
        static_n=n_tokens if n_tokens == 65536 else 0,
        static_num_blocks=num_blocks if block_size in (64, 129) else 0,
        full_tiles=full_tiles,
        num_cores=get_num_vector_cores(),
    )
    store_kernel(kv[:, 0, :], kv_cache, slot_mapping)


def _store_kv_cache_impl(
    kv: torch.Tensor,
    kv_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    use_fp8_rope: bool = False,
    num_per_channels: int = 64,
) -> None:
    _validate_store_input(kv, slot_mapping, 512)
    if kv.shape[1] != 1:
        raise ValueError('kv must have one head.')
    _validate_kv_metadata(kv_cache, slot_mapping, use_fp8_rope=use_fp8_rope, num_per_channels=num_per_channels)
    require_ascend()
    if kv_cache is None or not slot_mapping.numel():
        return
    if not use_fp8_rope and num_per_channels == 64:
        _store_kv_cache_64(kv, kv_cache, slot_mapping)
    elif use_fp8_rope and num_per_channels == 32:
        num_blocks, block_size, constant_1, bytes_per_token = kv_cache.shape
        assert constant_1 == 1
        assert bytes_per_token == 512 + 16
        kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
        kv_cache_fp8 = kv_cache[:, : block_size * 512].view(num_blocks, block_size, 512).view(torch.float8_e4m3fn)
        kv_cache_sf = kv_cache[:, block_size * 512 :].view(num_blocks, block_size, 16)
        assert kv_cache_fp8.stride(0) == kv_cache_sf.stride(0)
        store_block_m = 64
        store_kernel = _store_kv_cache_per32_tl_ascend(
            block_size,
            full_tiles=kv.shape[0] % store_block_m == 0,
            block_m=store_block_m,
            num_cores=get_num_vector_cores(),
        )
        store_kernel(kv[:, 0, :], kv_cache_fp8, kv_cache_sf, slot_mapping)
    else:
        raise ValueError(
            f'Unsupported combination: use_fp8_rope={use_fp8_rope}, '
            f'num_per_channels={num_per_channels}. Supported: (False, 64) and (True, 32).'
        )


def _load_kv_cache_64(out, kv_cache, slot_mapping, n) -> None:
    num_blocks, block_size, constant_1, bytes_per_token = kv_cache.shape
    assert constant_1 == 1
    assert bytes_per_token == 448 + 64 * 2 + 8
    kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
    kv_cache_value = kv_cache[:, : block_size * (448 + 64 * 2)].view(num_blocks, block_size, 448 + 64 * 2)
    kv_cache_sf = kv_cache[:, slice(block_size * (448 + 64 * 2), None)]
    kv_cache_fp8 = kv_cache_value[:, :, :448].view(torch.float8_e4m3fn)
    kv_cache_bf16 = kv_cache_value[:, :, 448:]
    kv_cache_bf16 = kv_cache_bf16.view(*kv_cache_bf16.shape[:-1], -1).view(torch.bfloat16)
    kv_cache_sf = kv_cache_sf.view(num_blocks, block_size, 8)
    assert kv_cache_fp8.stride(0) == kv_cache_bf16.stride(0) * 2
    assert kv_cache_fp8.stride(0) == kv_cache_sf.stride(0)
    if n % 64 == 0:
        load_kernel = _load_kv_cache_tl_ascend(
            block_size, num_stages=2, full_tiles=True, block_m=64, num_cores=get_num_vector_cores()
        )
    else:
        load_kernel = _load_kv_cache_tl_ascend(
            block_size, num_stages=4, full_tiles=False, block_m=32, num_cores=get_num_vector_cores()
        )
    load_kernel(out[:, 0, :], kv_cache_value, kv_cache_sf, slot_mapping)


def _load_kv_cache_impl(
    kv_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    use_fp8_rope: bool = False,
    num_per_channels: int = 64,
) -> torch.Tensor:
    _validate_kv_metadata(kv_cache, slot_mapping, use_fp8_rope, num_per_channels)
    require_ascend()
    if not slot_mapping.numel():
        return torch.empty((0, 1, 512), dtype=torch.bfloat16, device=slot_mapping.device)
    if kv_cache is None:
        return torch.randn(slot_mapping.shape[0], 1, 512, dtype=torch.bfloat16, device=slot_mapping.device)
    _validate_cache(kv_cache, slot_mapping, dtype=torch.uint8)
    n = slot_mapping.shape[0]
    out = torch.empty((n, 1, 512), dtype=torch.bfloat16, device=slot_mapping.device)
    if not use_fp8_rope and num_per_channels == 64:
        _load_kv_cache_64(out, kv_cache, slot_mapping, n)
    elif use_fp8_rope and num_per_channels == 32:
        num_blocks, block_size, constant_1, bytes_per_token = kv_cache.shape
        assert constant_1 == 1
        assert bytes_per_token == 512 + 16
        kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
        kv_cache_fp8 = kv_cache[:, : block_size * 512].view(num_blocks, block_size, 512).view(torch.float8_e4m3fn)
        kv_cache_sf = kv_cache[:, block_size * 512 :].view(num_blocks, block_size, 16)
        cache_block_stride = kv_cache_fp8.stride(0)
        assert cache_block_stride == kv_cache_sf.stride(0)
        assert cache_block_stride >= block_size * (512 + 16)
        assert cache_block_stride % 32 == 0
        assert kv_cache_fp8.stride(1) == 512
        assert kv_cache_sf.stride(1) == 16
        load_block_m = 64 if n % 64 == 0 else 32
        load_kernel = _load_kv_cache_per32_tl_ascend(
            block_size,
            full_tiles=n % load_block_m == 0,
            block_m=load_block_m,
            cache_block_stride=cache_block_stride,
            prefetch_slot_waves=n == 65536 and load_block_m == 64,
            num_cores=get_num_vector_cores(),
        )
        load_kernel(out[:, 0, :], kv_cache_fp8, kv_cache_sf, slot_mapping)
    else:
        raise ValueError(
            f'Unsupported combination: use_fp8_rope={use_fp8_rope}, '
            f'num_per_channels={num_per_channels}. Supported: (False, 64) and (True, 32).'
        )
    return out


def _store_index_kv_cache_impl(
    kv: torch.Tensor,
    kv_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    is_fp4: bool = False,
    is_mxfp8: bool = False,
) -> None:
    _validate_store_input(kv, slot_mapping, 128)
    if kv.shape[1] != 1:
        raise ValueError('kv must have one head.')
    _validate_index_metadata(kv_cache, slot_mapping, is_fp4, is_mxfp8)
    require_ascend()
    if kv_cache is None or not slot_mapping.numel():
        return
    num_blocks, block_size, constant_1, bytes_per_token = kv_cache.shape
    assert constant_1 == 1
    if is_fp4:
        assert bytes_per_token == 64 + 4
        kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
        kv_cache_fp4 = kv_cache[:, : block_size * 64].view(num_blocks, block_size, 64).view(torch.int8)
        kv_cache_sf = kv_cache[:, block_size * 64 :].view(num_blocks, block_size, 4).view(torch.uint8)
        assert kv_cache_fp4.stride(0) == kv_cache_sf.stride(0)
        store_kernel = _store_index_kv_cache_quant_tl_ascend(
            block_size, 'fp4', kv_cache_fp4.stride(1) * 2, kv_cache_sf.stride(1), num_cores=get_num_vector_cores()
        )
        store_kernel(kv[:, 0, :], kv_cache_fp4, kv_cache_sf, slot_mapping)
    elif is_mxfp8:
        assert bytes_per_token == 128 + 4
        kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
        kv_cache_fp8 = kv_cache[:, : block_size * 128].view(num_blocks, block_size, 128).view(torch.float8_e4m3fn)
        kv_cache_sf = kv_cache[:, block_size * 128 :].view(num_blocks, block_size, 4).view(torch.uint8)
        assert kv_cache_fp8.stride(0) == kv_cache_sf.stride(0)
        store_kernel = _store_index_kv_cache_quant_tl_ascend(
            block_size, 'mxfp8', kv_cache_fp8.stride(1), kv_cache_sf.stride(1), num_cores=get_num_vector_cores()
        )
        store_kernel(kv[:, 0, :], kv_cache_fp8, kv_cache_sf, slot_mapping)
    else:
        assert bytes_per_token == 128 + 4
        kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
        kv_cache_fp8 = kv_cache[:, : block_size * 128].view(num_blocks, block_size, 128).view(torch.float8_e4m3fn)
        kv_cache_sf = kv_cache[:, block_size * 128 :].view(num_blocks, block_size, 4)
        kv_cache_sf = kv_cache_sf.view(*kv_cache_sf.shape[:-1], -1).view(torch.float)
        assert kv_cache_fp8.stride(0) == kv_cache_sf.stride(0) * 4
        store_kernel = _store_index_kv_cache_quant_tl_ascend(
            block_size, 'fp8', kv_cache_fp8.stride(1), kv_cache_sf.stride(1), num_cores=get_num_vector_cores()
        )
        store_kernel(kv[:, 0, :], kv_cache_fp8, kv_cache_sf, slot_mapping)


def _load_index_without_cache(slot_mapping, is_fp4, is_mxfp8) -> tuple[torch.Tensor, torch.Tensor]:
    if is_fp4:
        k_bf16 = torch.randn(slot_mapping.shape[0], 128, dtype=torch.bfloat16, device=slot_mapping.device)
        k_fp4, k_sf = per_token_cast(
            k_bf16, 'e2m1', 32, round_sf=True, use_tma_aligned_col_major_sf=True, use_packed_ue8m0=True
        )
        return (k_fp4, k_sf.squeeze(-1))
    elif is_mxfp8:
        k_bf16 = torch.randn(slot_mapping.shape[0], 128, dtype=torch.bfloat16, device=slot_mapping.device)
        k_fp8, k_sf = per_token_cast(
            k_bf16, 'e4m3', 32, round_sf=True, use_tma_aligned_col_major_sf=False, use_packed_ue8m0=True
        )
        return (k_fp8, k_sf.squeeze(-1))
    else:
        k_bf16 = torch.randn(slot_mapping.shape[0], 128, dtype=torch.bfloat16, device=slot_mapping.device)
        k_fp8, k_sf = per_token_cast(k_bf16, 'e4m3', 128, round_sf=True)
        return (k_fp8, k_sf.squeeze(-1))


def _load_index_fp4(
    kv_cache, slot_mapping, num_blocks, block_size, bytes_per_token, n
) -> tuple[torch.Tensor, torch.Tensor]:
    assert bytes_per_token == 64 + 4
    kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
    kv_cache_fp4 = kv_cache[:, : block_size * 64].view(num_blocks, block_size, 64).view(torch.int8)
    kv_cache_sf = kv_cache[:, slice(block_size * 64, None)].view(num_blocks, block_size, 4)
    kv_cache_sf = kv_cache_sf.view(*kv_cache_sf.shape[:-1], -1).view(torch.int32)
    assert kv_cache_fp4.stride(0) == kv_cache_sf.stride(0) * 4
    kv_fp4 = torch.empty((n, 64), dtype=torch.int8, device=slot_mapping.device)
    kv_sf = torch.empty((n,), dtype=torch.int32, device=slot_mapping.device)
    load_kernel = _load_index_kv_cache_quant_tl_ascend(
        block_size, 64, kv_cache_fp4.stride(1), kv_cache_sf.stride(1), num_cores=get_num_vector_cores()
    )
    load_kernel(kv_fp4.view(torch.uint8), kv_sf, kv_cache_fp4.view(torch.uint8), kv_cache_sf, slot_mapping)
    return (kv_fp4, kv_sf)


def _load_index_kv_cache_impl(
    kv_cache: torch.Tensor | None, slot_mapping: torch.Tensor, is_fp4: bool = False, is_mxfp8: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_index_metadata(kv_cache, slot_mapping, is_fp4, is_mxfp8)
    require_ascend()
    if not slot_mapping.numel():
        return _allocate_index_result(slot_mapping, is_fp4, is_mxfp8, none_cache=kv_cache is None)
    if kv_cache is None:
        return _load_index_without_cache(slot_mapping, is_fp4, is_mxfp8)
    _validate_cache(kv_cache, slot_mapping, dtype=torch.uint8)
    num_blocks, block_size, constant_1, bytes_per_token = kv_cache.shape
    n = slot_mapping.shape[0]
    assert constant_1 == 1
    if is_fp4:
        return _load_index_fp4(kv_cache, slot_mapping, num_blocks, block_size, bytes_per_token, n)
    elif is_mxfp8:
        assert bytes_per_token == 128 + 4
        kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
        kv_cache_fp8 = kv_cache[:, : block_size * 128].view(num_blocks, block_size, 128).view(torch.float8_e4m3fn)
        kv_cache_sf = kv_cache[:, block_size * 128 :].view(num_blocks, block_size, 4)
        kv_cache_sf = kv_cache_sf.view(*kv_cache_sf.shape[:-1], -1).view(torch.int32)
        assert kv_cache_fp8.stride(0) == kv_cache_sf.stride(0) * 4
        kv_fp8 = torch.empty((n, 128), dtype=torch.float8_e4m3fn, device=slot_mapping.device)
        kv_scale = torch.empty((n,), dtype=torch.int32, device=slot_mapping.device)
        load_kernel = _load_index_kv_cache_quant_tl_ascend(
            block_size, 128, kv_cache_fp8.stride(1), kv_cache_sf.stride(1), num_cores=get_num_vector_cores()
        )
        load_kernel(kv_fp8.view(torch.uint8), kv_scale, kv_cache_fp8.view(torch.uint8), kv_cache_sf, slot_mapping)
        return (kv_fp8, kv_scale)
    else:
        assert bytes_per_token == 128 + 4
        kv_cache = kv_cache.view(num_blocks, block_size * bytes_per_token)
        kv_cache_fp8 = kv_cache[:, : block_size * 128].view(num_blocks, block_size, 128).view(torch.float8_e4m3fn)
        kv_cache_sf = kv_cache[:, block_size * 128 :].view(num_blocks, block_size, 4)
        kv_cache_sf = kv_cache_sf.view(*kv_cache_sf.shape[:-1], -1).view(torch.float)
        assert kv_cache_fp8.stride(0) == kv_cache_sf.stride(0) * 4
        kv_fp8 = torch.empty((n, 128), dtype=torch.float8_e4m3fn, device=slot_mapping.device)
        kv_sf = torch.empty((n,), dtype=torch.float, device=slot_mapping.device)
        load_kernel = _load_index_kv_cache_quant_tl_ascend(
            block_size, 128, kv_cache_fp8.stride(1), kv_cache_sf.stride(1), num_cores=get_num_vector_cores()
        )
        load_kernel(
            kv_fp8.view(torch.uint8),
            kv_sf.view(torch.int32),
            kv_cache_fp8.view(torch.uint8),
            kv_cache_sf.view(torch.int32),
            slot_mapping,
        )
        return (kv_fp8, kv_sf)


def _store_kv_cache_llama_impl(
    store_kv: torch.Tensor, kv_cache: torch.Tensor | None, slot_mapping: torch.Tensor
) -> None:
    _validate_store_input(store_kv, slot_mapping, 128)
    _validate_llama_metadata(kv_cache, store_kv.shape[1], slot_mapping)
    require_ascend()
    if kv_cache is None or not slot_mapping.numel():
        return
    _, block_size, n_head_kv, head_size = kv_cache.shape
    assert head_size == 128
    store_kv = store_kv.view(-1, n_head_kv * head_size)
    kv_cache = kv_cache.view(-1, block_size, n_head_kv * head_size)
    store_kernel = _store_kv_cache_llama_tl_ascend(block_size, n_head_kv * head_size, num_cores=get_num_vector_cores())
    store_kernel(store_kv, kv_cache, slot_mapping)


def _load_kv_cache_llama_impl(
    kv_cache: torch.Tensor | None, n_head_kv: int, slot_mapping: torch.Tensor
) -> torch.Tensor:
    _validate_llama_metadata(kv_cache, n_head_kv, slot_mapping)
    require_ascend()
    if not slot_mapping.numel():
        return torch.empty((0, n_head_kv, 128), dtype=torch.bfloat16, device=slot_mapping.device)
    if kv_cache is None:
        return torch.randn((slot_mapping.shape[0], n_head_kv, 128), dtype=torch.bfloat16, device=slot_mapping.device)
    _validate_cache(kv_cache, slot_mapping, dtype=torch.bfloat16)
    _, block_size, num_head_kv, head_size = kv_cache.shape
    assert head_size == 128
    assert num_head_kv == n_head_kv
    n = slot_mapping.shape[0]
    kv_cache = kv_cache.view(-1, block_size, n_head_kv * head_size)
    kv = torch.empty((n, n_head_kv * head_size), dtype=torch.bfloat16, device=slot_mapping.device)
    load_kernel = _load_kv_cache_llama_tl_ascend(block_size, n_head_kv * head_size, num_cores=get_num_vector_cores())
    load_kernel(kv, kv_cache, slot_mapping)
    return kv.view(n, n_head_kv, head_size)


@torch.library.custom_op('cann_ops_tilelang::store_kv_cache', mutates_args=('kv_cache',))
def store_kv_cache(
    kv: torch.Tensor,
    kv_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    use_fp8_rope: bool = False,
    num_per_channels: int = 64,
) -> None:
    """Store KV to cache with fp8 quantization.

    Supported configurations:
      - (use_fp8_rope=False, num_per_channels=64): nope(448) per-64 fp8 + rope(64) bf16.
        bytes_per_token = 448 + 64*2 + 8 = 584.
      - (use_fp8_rope=True, num_per_channels=32): full 512 per-32 fp8.
        bytes_per_token = 512 + 16 = 528.

    Args:
        kv: [n, 1, 512], bf16
        kv_cache: [num_blocks, block_size, 1, bytes_per_token], uint8
        slot_mapping: [n], int32, -1 means padding
        use_fp8_rope: if True, rope is also stored as fp8 (per32 mode)
        num_per_channels: quantization group size (64 or 32)

    Supports fixed-shape npugraph_ex inference; no automatic autograd formula is registered.
    """
    return _store_kv_cache_impl(kv, kv_cache, slot_mapping, use_fp8_rope, num_per_channels)


@store_kv_cache.register_fake
def _store_kv_cache_fake(kv, kv_cache, slot_mapping, use_fp8_rope=False, num_per_channels=64):
    _validate_store_input(kv, slot_mapping, 512)
    if kv.shape[1] != 1:
        raise ValueError('kv must have one head.')
    _validate_kv_metadata(kv_cache, slot_mapping, use_fp8_rope=use_fp8_rope, num_per_channels=num_per_channels)


@torch.library.custom_op('cann_ops_tilelang::load_kv_cache', mutates_args=())
def load_kv_cache(
    kv_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    use_fp8_rope: bool = False,
    num_per_channels: int = 64,
) -> torch.Tensor:
    """Load KV from cache with fp8 dequantization.

    Supported configurations:
      - (use_fp8_rope=False, num_per_channels=64): nope(448) per-64 fp8 + rope(64) bf16.
      - (use_fp8_rope=True, num_per_channels=32): full 512 per-32 fp8.

    Args:
        kv_cache: [num_blocks, block_size, 1, bytes_per_token], uint8
        slot_mapping: [n], int32, -1 means padding
        use_fp8_rope: if True, rope is also stored as fp8 (per32 mode)
        num_per_channels: quantization group size (64 or 32)

    Returns:
        A new BF16 tensor with shape [n, 1, 512].

    The former optional out= argument has been removed. Use the returned tensor
    directly, or copy it into a caller-owned buffer outside this operator.

    Supports fixed-shape npugraph_ex inference; no automatic autograd formula is registered.
    """
    return _load_kv_cache_impl(kv_cache, slot_mapping, use_fp8_rope, num_per_channels)


@load_kv_cache.register_fake
def _load_kv_cache_fake(kv_cache, slot_mapping, use_fp8_rope=False, num_per_channels=64):
    _validate_kv_metadata(kv_cache, slot_mapping, use_fp8_rope, num_per_channels)
    return torch.empty((slot_mapping.numel(), 1, 512), dtype=torch.bfloat16, device=slot_mapping.device)


@torch.library.custom_op('cann_ops_tilelang::store_index_kv_cache', mutates_args=('kv_cache',))
def store_index_kv_cache(
    kv: torch.Tensor,
    kv_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor,
    is_fp4: bool = False,
    is_mxfp8: bool = False,
) -> None:
    """Args:
        kv: [n, 1, dim], bf16
        kv_cache: [num_blocks, block_size, 1, bytes_per_token], uint8
        slot_mapping: [n], int32, -1 means padding

    Supports fixed-shape npugraph_ex inference; no automatic autograd formula is registered.
    """
    return _store_index_kv_cache_impl(kv, kv_cache, slot_mapping, is_fp4, is_mxfp8)


@store_index_kv_cache.register_fake
def _store_index_kv_cache_fake(kv, kv_cache, slot_mapping, is_fp4=False, is_mxfp8=False):
    _validate_store_input(kv, slot_mapping, 128)
    if kv.shape[1] != 1:
        raise ValueError('kv must have one head.')
    _validate_index_metadata(kv_cache, slot_mapping, is_fp4, is_mxfp8)


@torch.library.custom_op('cann_ops_tilelang::load_index_kv_cache', mutates_args=())
def load_index_kv_cache(
    kv_cache: torch.Tensor | None, slot_mapping: torch.Tensor, is_fp4: bool = False, is_mxfp8: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Args:
        kv_cache: [num_blocks, block_size, 1, bytes_per_token], uint8
        slot_mapping: [n], int32, -1 means padding
        is_fp4: bool, whether to return fp4 or fp8

    Returns:
        if is_fp4:
            kv_fp4: [n, dim // 2], int8
            kv_sf: [n], int32
        if is_mxfp8 (fp8 per-32 ue8m0):
            kv_fp8: [n, dim], float8_e4m3
            kv_sf: [n], int32 (packed 4 ue8m0)
        else (fp8 per-token float32):
            kv_fp8: [n, dim], float8_e4m3
            kv_sf: [n], float32

    A None cache retains the random per_token_cast fallback. Its FP4/MXFP8
    scales use that quantizer's packed int16 matrix layout, including FP4's
    column-major stride, rather than the int32 records stored in a real cache.

    Supports fixed-shape npugraph_ex inference; no automatic autograd formula is registered.
    """
    return _load_index_kv_cache_impl(kv_cache, slot_mapping, is_fp4, is_mxfp8)


@load_index_kv_cache.register_fake
def _load_index_kv_cache_fake(kv_cache, slot_mapping, is_fp4=False, is_mxfp8=False):
    _validate_index_metadata(kv_cache, slot_mapping, is_fp4, is_mxfp8)
    return _allocate_index_result(slot_mapping, is_fp4, is_mxfp8, none_cache=kv_cache is None)


@torch.library.custom_op('cann_ops_tilelang::store_kv_cache_llama', mutates_args=('kv_cache',))
def store_kv_cache_llama(store_kv: torch.Tensor, kv_cache: torch.Tensor | None, slot_mapping: torch.Tensor) -> None:
    """Scatter BF16 LLaMA KV in place, preserving skipped slots and page padding."""
    return _store_kv_cache_llama_impl(store_kv, kv_cache, slot_mapping)


@store_kv_cache_llama.register_fake
def _store_kv_cache_llama_fake(store_kv, kv_cache, slot_mapping):
    _validate_store_input(store_kv, slot_mapping, 128)
    _validate_llama_metadata(kv_cache, store_kv.shape[1], slot_mapping)


@torch.library.custom_op('cann_ops_tilelang::load_kv_cache_llama', mutates_args=())
def load_kv_cache_llama(kv_cache: torch.Tensor | None, n_head_kv: int, slot_mapping: torch.Tensor) -> torch.Tensor:
    """Gather BF16 LLaMA KV, returning zeros for padding slots and a new tensor."""
    return _load_kv_cache_llama_impl(kv_cache, n_head_kv, slot_mapping)


@load_kv_cache_llama.register_fake
def _load_kv_cache_llama_fake(kv_cache, n_head_kv, slot_mapping):
    _validate_llama_metadata(kv_cache, n_head_kv, slot_mapping)
    return torch.empty((slot_mapping.numel(), n_head_kv, 128), dtype=torch.bfloat16, device=slot_mapping.device)
