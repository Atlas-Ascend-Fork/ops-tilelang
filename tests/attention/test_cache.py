from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.attention.cache import (
    load_index_kv_cache,
    load_kv_cache,
    load_kv_cache_llama,
    store_index_kv_cache,
    store_kv_cache,
    store_kv_cache_llama,
)
from cann_ops_tilelang.config import get_device, get_num_vector_cores
from cann_ops_tilelang.quant import per_token_cast
from tests.quant.cast_ref import cast as torch_cast

pytestmark = pytest.mark.npu


@pytest.fixture(autouse=True)
def _restore_default_device():
    """Keep cache tests from changing tensor placement in other test modules."""
    yield
    torch.set_default_device('cpu')


@pytest.mark.parametrize('use_fp8_rope,num_per_channels', [(False, 64), (True, 32)])
@pytest.mark.parametrize('block_size,num_blocks,n', [(64, 2333, 65536), (32, 4666, 65536), (129, 2333, 65536)])
def test_kvcache(block_size: int, num_blocks: int, n: int, use_fp8_rope: bool, num_per_channels: int) -> None:
    assert num_blocks * block_size > n
    torch.set_default_device('npu')
    if not use_fp8_rope:
        bytes_per_token = 448 + 64 * 2 + 8
        alignment_size = 576
    else:
        bytes_per_token = 512 + 16
        alignment_size = 512
    kv_cache_byte_per_block = (block_size * bytes_per_token + alignment_size - 1) // alignment_size * alignment_size
    kv_cache = torch.empty((num_blocks, kv_cache_byte_per_block), dtype=torch.uint8)
    kv_cache = kv_cache[:, : block_size * bytes_per_token].view(num_blocks, block_size, 1, bytes_per_token)
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32)[:n]
    store_kv = torch.randn((n, 1, 512), dtype=torch.bfloat16)
    store_kv_cache(store_kv, kv_cache, slot_mapping, use_fp8_rope=use_fp8_rope, num_per_channels=num_per_channels)
    cache_flat = kv_cache.view(num_blocks, block_size * bytes_per_token)
    cache_block = slot_mapping // block_size
    cache_offset = slot_mapping % block_size
    ref_quant = per_token_cast(store_kv[:, 0, :], 'e4m3', num_per_channels, round_sf=True, use_packed_ue8m0=True)
    ref_fp8, ref_sf = ref_quant
    ref_sf_u8 = ref_sf.view(torch.uint8)
    if not use_fp8_rope:
        cache_value = cache_flat[:, : block_size * (448 + 64 * 2)].view(num_blocks, block_size, 448 + 64 * 2)
        cache_fp8_u8 = cache_value[:, :, :448]
        cache_bf16_u8 = cache_value[:, :, 448:]
        cache_bf16 = cache_bf16_u8.view(*cache_bf16_u8.shape[:-1], -1).view(torch.bfloat16)
        cache_sf = cache_flat[:, block_size * (448 + 64 * 2) :].view(num_blocks, block_size, 8)
        assert torch.equal(cache_fp8_u8[cache_block, cache_offset], ref_fp8[:, :448].view(torch.uint8))
        assert torch.equal(cache_bf16[cache_block, cache_offset], store_kv[:, 0, 448:])
        assert torch.equal(cache_sf[cache_block, cache_offset], ref_sf_u8)
    else:
        cache_fp8_u8 = cache_flat[:, : block_size * 512].view(num_blocks, block_size, 512)
        cache_sf = cache_flat[:, block_size * 512 :].view(num_blocks, block_size, 16)
        assert torch.equal(cache_fp8_u8[cache_block, cache_offset], ref_fp8.view(torch.uint8))
        assert torch.equal(cache_sf[cache_block, cache_offset], ref_sf_u8)
    load_kv = load_kv_cache(kv_cache, slot_mapping, use_fp8_rope=use_fp8_rope, num_per_channels=num_per_channels)
    ref_fp8_cpu = ref_fp8.cpu().float().view(n, 512 // num_per_channels, num_per_channels)
    ref_scale_bits_cpu = ref_sf_u8.cpu().to(torch.int32) << 23
    ref_scale_cpu = ref_scale_bits_cpu.view(torch.float32)
    ref_load_kv = (ref_fp8_cpu * ref_scale_cpu.unsqueeze(-1)).to(torch.bfloat16).view(n, 1, 512)
    load_kv_cpu = load_kv.cpu()
    if not use_fp8_rope:
        assert torch.equal(ref_load_kv[:, :, :448], load_kv_cpu[:, :, :448])
        assert torch.equal(store_kv[:, :, 448:].cpu(), load_kv_cpu[:, :, 448:])
    else:
        assert torch.equal(ref_load_kv, load_kv_cpu)


@pytest.mark.parametrize('mode', ['fp8', 'mxfp8', 'fp4'])
@pytest.mark.parametrize('block_size,num_blocks,n', [(64, 2333, 65536), (32, 4666, 65536)])
def test_index_kvcache(block_size: int, num_blocks: int, n: int, mode: str) -> None:
    assert num_blocks * block_size > n
    torch.set_default_device('npu')
    is_fp4 = mode == 'fp4'
    is_mxfp8 = mode == 'mxfp8'
    bytes_per_token = 64 + 4 if is_fp4 else 128 + 4
    kv_cache = torch.empty((num_blocks, block_size, 1, bytes_per_token), dtype=torch.uint8)
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32)[:n]
    store_kv = torch.randn((n, 1, 128), dtype=torch.bfloat16)
    store_index_kv_cache(store_kv, kv_cache, slot_mapping, is_fp4=is_fp4, is_mxfp8=is_mxfp8)
    load_kv = load_index_kv_cache(kv_cache, slot_mapping, is_fp4=is_fp4, is_mxfp8=is_mxfp8)
    if mode == 'fp8':
        ref = per_token_cast(store_kv.contiguous().view(n, 128), 'e4m3', 128, round_sf=True)
        ref[1].squeeze_(-1)
        assert torch.equal(ref[0].view(torch.uint8).cpu(), load_kv[0].view(torch.uint8).cpu())
        assert torch.equal(ref[1].cpu(), load_kv[1].cpu())
    elif mode == 'mxfp8':
        ref = per_token_cast(
            store_kv[:, 0, :], 'e4m3', 32, round_sf=True, use_tma_aligned_col_major_sf=False, use_packed_ue8m0=True
        )
        assert torch.equal(ref[0].view(torch.uint8).cpu(), load_kv[0].view(torch.uint8).cpu())
        assert torch.equal(ref[1].contiguous().view(torch.uint8).flatten().cpu(), load_kv[1].view(torch.uint8).cpu())
    else:
        ref = per_token_cast(
            store_kv[:, 0, :], 'e2m1', 32, round_sf=True, use_tma_aligned_col_major_sf=True, use_packed_ue8m0=True
        )
        assert torch.equal(ref[0].view(torch.uint8).cpu(), load_kv[0].view(torch.uint8).cpu())
        assert torch.equal(ref[1].contiguous().view(torch.uint8).flatten().cpu(), load_kv[1].view(torch.uint8).cpu())


@pytest.mark.parametrize('mode', ['fp8', 'mxfp8', 'fp4'])
@pytest.mark.parametrize(
    'block_size,num_blocks,n',
    [
        pytest.param(8, 4, 19, id='small-block-tail'),
        pytest.param(32, 2, 33, id='block-cross-boundary'),
        pytest.param(32, 9, 257, id='persistent-multi-wave'),
    ],
)
def test_store_index_kvcache_ascend(block_size: int, num_blocks: int, n: int, mode: str) -> None:
    """Validate quantization, cache layout, slot scatter, and padding writes."""
    capacity = num_blocks * block_size
    assert n <= capacity
    is_fp4 = mode == 'fp4'
    is_mxfp8 = mode == 'mxfp8'
    data_bytes = 64 if is_fp4 else 128
    bytes_per_token = data_bytes + 4
    fmt = 'e2m1' if is_fp4 else 'e4m3'
    group_size = 32 if is_fp4 or is_mxfp8 else 128
    cpu_generator = torch.Generator().manual_seed(20240728)
    store_kv_cpu = torch.randn((n, 1, 128), dtype=torch.bfloat16, generator=cpu_generator, device='cpu')
    store_kv_cpu[0].zero_()
    store_kv_cpu[1, 0] = torch.linspace(-512, 512, 128, dtype=torch.bfloat16, device='cpu')
    store_kv = store_kv_cpu.to('npu')
    slot_mapping_cpu = torch.randperm(capacity, generator=cpu_generator, dtype=torch.int32, device='cpu')[:n]
    slot_mapping_cpu[2::7] = -1
    slot_mapping = slot_mapping_cpu.to('npu')
    sentinel = 165
    kv_cache = torch.full((num_blocks, block_size, 1, bytes_per_token), sentinel, dtype=torch.uint8, device='npu')
    store_index_kv_cache(store_kv, kv_cache, slot_mapping, is_fp4=is_fp4, is_mxfp8=is_mxfp8)
    ref_data, ref_sf = torch_cast(
        store_kv[:, 0, :],
        fmt,
        block_size=(1, group_size),
        round_sf=True,
        use_packed_ue8m0=is_fp4 or is_mxfp8,
    )
    torch.npu.synchronize()
    ref_data_bytes = ref_data.contiguous().view(torch.uint8).cpu().view(n, data_bytes)
    ref_sf_bytes = ref_sf.contiguous().view(torch.uint8).cpu().view(n, 4)
    expected = torch.full((num_blocks, block_size * bytes_per_token), sentinel, dtype=torch.uint8, device='cpu')
    for token_idx, slot_idx in enumerate(slot_mapping_cpu.tolist()):
        if slot_idx == -1:
            continue
        cache_block = slot_idx // block_size
        cache_offset = slot_idx % block_size
        data_begin = cache_offset * data_bytes
        sf_begin = block_size * data_bytes + cache_offset * 4
        expected[cache_block, data_begin : data_begin + data_bytes] = ref_data_bytes[token_idx]
        expected[cache_block, sf_begin : sf_begin + 4] = ref_sf_bytes[token_idx]
    actual = kv_cache.cpu().view(num_blocks, block_size * bytes_per_token)
    mismatch = torch.nonzero(actual != expected)
    assert mismatch.numel() == 0, (
        f'{mode} cache bytes differ at {mismatch[:8].tolist()}; '
        'this includes untouched slots and slot_mapping=-1 validation'
    )


@pytest.mark.parametrize('mode', ['fp8', 'mxfp8', 'fp4'])
@pytest.mark.parametrize(
    'block_size,num_blocks,n,padded_block_stride',
    [
        pytest.param(8, 4, 19, False, id='small-contiguous-tail'),
        pytest.param(32, 2, 33, True, id='block-cross-boundary'),
        pytest.param(32, 9, 257, True, id='persistent-multi-wave'),
        pytest.param(64, 2, 65, False, id='block-size-64'),
    ],
)
def test_load_index_kvcache_ascend(
    block_size: int, num_blocks: int, n: int, mode: str, padded_block_stride: bool
) -> None:
    """Validate byte-exact gathers, padded block strides, duplicate slots, and -1."""
    capacity = num_blocks * block_size
    assert n <= capacity
    is_fp4 = mode == 'fp4'
    is_mxfp8 = mode == 'mxfp8'
    data_bytes = 64 if is_fp4 else 128
    bytes_per_token = data_bytes + 4
    logical_block_bytes = block_size * bytes_per_token
    block_stride = (logical_block_bytes + 575) // 576 * 576 if padded_block_stride else logical_block_bytes
    cache_bytes = torch.arange(num_blocks * block_stride, dtype=torch.int64, device='cpu')
    cache_bytes = ((cache_bytes * 73 + 19) % 256).to(torch.uint8).view(num_blocks, block_stride)
    cache_storage = cache_bytes.to('npu')
    kv_cache = cache_storage[:, :logical_block_bytes].view(num_blocks, block_size, 1, bytes_per_token)
    assert kv_cache.stride(0) == block_stride
    assert (block_stride > logical_block_bytes) == padded_block_stride
    slot_mapping_cpu = ((torch.arange(n, dtype=torch.int32, device='cpu') * 17 + 3) % capacity).to(torch.int32)
    slot_mapping_cpu[0] = -1
    slot_mapping_cpu[2] = capacity - 1
    slot_mapping_cpu[3] = slot_mapping_cpu[1]
    slot_mapping_cpu[5::7] = -1
    slot_mapping_cpu[-1] = -1
    slot_mapping = slot_mapping_cpu.to('npu')
    loaded_data, loaded_sf = load_index_kv_cache(kv_cache, slot_mapping, is_fp4=is_fp4, is_mxfp8=is_mxfp8)
    torch.npu.synchronize()
    expected_data_dtype = torch.int8 if is_fp4 else torch.float8_e4m3fn
    expected_sf_dtype = torch.int32 if is_fp4 or is_mxfp8 else torch.float32
    assert loaded_data.shape == (n, data_bytes)
    assert loaded_data.dtype == expected_data_dtype
    assert loaded_sf.shape == (n,)
    assert loaded_sf.dtype == expected_sf_dtype
    expected_data = torch.zeros((n, data_bytes), dtype=torch.uint8, device='cpu')
    expected_sf = torch.zeros((n, 4), dtype=torch.uint8, device='cpu')
    logical_cache = cache_bytes[:, :logical_block_bytes]
    for token_idx, slot_idx in enumerate(slot_mapping_cpu.tolist()):
        if slot_idx == -1:
            continue
        cache_block = slot_idx // block_size
        cache_offset = slot_idx % block_size
        data_begin = cache_offset * data_bytes
        sf_begin = block_size * data_bytes + cache_offset * 4
        expected_data[token_idx] = logical_cache[cache_block, data_begin : data_begin + data_bytes]
        expected_sf[token_idx] = logical_cache[cache_block, sf_begin : sf_begin + 4]
    actual_data = loaded_data.view(torch.uint8).cpu().view(n, data_bytes)
    actual_sf = loaded_sf.view(torch.uint8).cpu().view(n, 4)
    assert torch.equal(actual_data, expected_data)
    assert torch.equal(actual_sf, expected_sf)


@pytest.mark.parametrize(
    'block_size,num_blocks,n_spec,n_head_kv,input_padding,cache_padding',
    [
        pytest.param(128, 2, 0, 2, 16, 32, id='empty-input'),
        pytest.param(8, 4, 19, 1, 0, 0, id='single-head-small-tail'),
        pytest.param(32, 2, 33, 4, 16, 32, id='multi-head-cross-block-padded-strides'),
        pytest.param(128, 4, 'all-aiv-cores', 2, 16, 32, id='all-aiv-cores-one-token'),
        pytest.param(128, 4, 'front-aiv-core-extra-token', 2, 16, 32, id='front-aiv-core-extra-token'),
        pytest.param(128, 16, 'simt-second-wave', 8, 32, 64, id='simt-second-wave'),
    ],
)
def test_store_kv_cache_llama_ascend(
    block_size: int, num_blocks: int, n_spec: int | str, n_head_kv: int, input_padding: int, cache_padding: int
) -> None:
    """Validate exact scatter, -1 skips, dynamic strides, and SIMT tails."""
    if isinstance(n_spec, str):
        num_vec_cores = get_num_vector_cores()
        n = {
            'all-aiv-cores': num_vec_cores,
            'front-aiv-core-extra-token': num_vec_cores + 1,
            'simt-second-wave': num_vec_cores * 8 + 1,
        }[n_spec]
    else:
        n = n_spec
    capacity = num_blocks * block_size
    element_per_token = n_head_kv * 128
    logical_block_elements = block_size * element_per_token
    assert n <= capacity
    cpu_generator = torch.Generator().manual_seed(20240731)
    store_storage_cpu = torch.full((n, element_per_token + input_padding), -123.0, dtype=torch.bfloat16, device='cpu')
    store_values_cpu = torch.randn((n, element_per_token), dtype=torch.bfloat16, generator=cpu_generator, device='cpu')
    store_storage_cpu[:, :element_per_token] = store_values_cpu
    store_storage = store_storage_cpu.to('npu')
    store_kv = store_storage[:, :element_per_token].view(n, n_head_kv, 128)
    sentinel = -77.0
    cache_storage = torch.full(
        (num_blocks, logical_block_elements + cache_padding), sentinel, dtype=torch.bfloat16, device='npu'
    )
    kv_cache = cache_storage[:, :logical_block_elements].view(num_blocks, block_size, n_head_kv, 128)
    if n > 0:
        assert store_kv.stride(0) == element_per_token + input_padding
    assert kv_cache.stride(0) == logical_block_elements + cache_padding
    slot_mapping_cpu = torch.randperm(capacity, generator=cpu_generator, dtype=torch.int32, device='cpu')[:n]
    if n > 0:
        slot_mapping_cpu[1::6] = -1
        slot_mapping_cpu[-1] = -1
    slot_mapping = slot_mapping_cpu.to('npu')
    store_kv_cache_llama(store_kv, kv_cache, slot_mapping)
    torch.npu.synchronize()
    expected = torch.full(
        (num_blocks, logical_block_elements + cache_padding), sentinel, dtype=torch.bfloat16, device='cpu'
    )
    expected_cache = expected[:, :logical_block_elements].view(num_blocks, block_size, n_head_kv, 128)
    store_values = store_values_cpu.view(n, n_head_kv, 128)
    for token_idx, slot_idx in enumerate(slot_mapping_cpu.tolist()):
        if slot_idx == -1:
            continue
        expected_cache[slot_idx // block_size, slot_idx % block_size] = store_values[token_idx]
    actual = cache_storage.cpu()
    mismatch = torch.nonzero(actual != expected)
    assert mismatch.numel() == 0, (
        f'LLaMA cache differs at {mismatch[:8].tolist()}; '
        'this includes untouched slots, -1 entries, and block-stride padding'
    )
    assert torch.equal(store_storage.cpu(), store_storage_cpu), 'store_kv input was modified'


@pytest.mark.parametrize('block_size,num_blocks,n', [(128, 2333, 65536), (64, 2333, 32768), (32, 4666, 32768)])
def test_llama_kv_cache(block_size: int, num_blocks: int, n: int) -> None:
    assert num_blocks * block_size > n
    torch.set_default_device('npu')
    head_n = 4
    head_size = 128
    kv_cache = torch.empty((num_blocks, block_size, head_n, head_size), dtype=torch.bfloat16)
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32)[:n]
    positions = torch.randperm(n, dtype=torch.int32)[:4]
    slot_mapping[positions] = -1
    store_kv = torch.randn((n, head_n, head_size), dtype=torch.bfloat16)
    store_kv_cache_llama(store_kv, kv_cache, slot_mapping)
    load_kv = load_kv_cache_llama(kv_cache, head_n, slot_mapping)
    store_kv[positions] = torch.zeros((head_n, head_size), dtype=torch.bfloat16)
    assert torch.equal(load_kv, store_kv)


@pytest.mark.parametrize(
    'block_size,num_blocks,n,head_n',
    [
        pytest.param(128, 2, 5, 1, id='single-head-boundaries'),
        pytest.param(64, 3, 67, 4, id='multi-head-cross-block'),
        pytest.param(32, 7, 193, 8, id='persistent-multi-wave'),
        pytest.param(128, 2, 17, 32, id='full-mha-wide-token'),
    ],
)
def test_load_kv_cache_llama_ascend(block_size: int, num_blocks: int, n: int, head_n: int) -> None:
    """Validate NPU gather, page boundaries, repeated slots, and -1 padding."""
    capacity = num_blocks * block_size
    assert n <= capacity
    cpu_generator = torch.Generator().manual_seed(20240731)
    kv_cache_cpu = torch.randn(
        (num_blocks, block_size, head_n, 128), dtype=torch.bfloat16, device='cpu', generator=cpu_generator
    )
    slot_mapping_cpu = ((torch.arange(n, dtype=torch.int64, device='cpu') * 37 + 11) % capacity).to(torch.int32)
    slot_mapping_cpu[0] = 0
    slot_mapping_cpu[1] = block_size - 1
    slot_mapping_cpu[2] = block_size
    slot_mapping_cpu[3] = capacity - 1
    slot_mapping_cpu[4] = -1
    slot_mapping_cpu[5::7] = -1
    slot_mapping_cpu[6::11] = 0
    expected = torch.zeros((n, head_n, 128), dtype=torch.bfloat16, device='cpu')
    valid = slot_mapping_cpu != -1
    expected[valid] = kv_cache_cpu.view(capacity, head_n, 128)[slot_mapping_cpu[valid].long()]
    actual = load_kv_cache_llama(kv_cache_cpu.to('npu'), head_n, slot_mapping_cpu.to('npu'))
    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert actual.device.type == 'npu'
    assert torch.equal(actual.cpu(), expected)


def _benchmark_device() -> torch.device:
    return get_device()


def _make_kv_cache_4d(
    num_blocks: int, block_size: int, bytes_per_token: int, alignment_size: int = 576
) -> torch.Tensor:
    """Helper to create properly aligned 4D kv_cache for benchmarks."""
    kv_cache_byte_per_block = (block_size * bytes_per_token + alignment_size - 1) // alignment_size * alignment_size
    kv_cache = torch.zeros((num_blocks, kv_cache_byte_per_block), dtype=torch.uint8, device=_benchmark_device())
    return kv_cache[:, : block_size * bytes_per_token].view(num_blocks, block_size, 1, bytes_per_token)


@pytest.mark.benchmark
@pytest.mark.parametrize('use_fp8_rope,num_per_channels', [(False, 64), (True, 32)])
@pytest.mark.parametrize('block_size,num_blocks,n', [(64, 2333, 65536), (32, 4666, 65536), (129, 2333, 65536)])
def test_store_kv_cache_benchmark(
    block_size: int,
    num_blocks: int,
    n: int,
    use_fp8_rope: bool,
    num_per_channels: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    if not use_fp8_rope:
        bytes_per_token = 448 + 64 * 2 + 8
        alignment_size = 576
    else:
        bytes_per_token = 512 + 16
        alignment_size = 512
    kv_cache = _make_kv_cache_4d(num_blocks, block_size, bytes_per_token, alignment_size=alignment_size)
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32, device=_benchmark_device())[:n]
    store_kv = torch.randn(n, 1, 512, dtype=torch.bfloat16, device=_benchmark_device())
    t_us = benchmark_timer(
        lambda: store_kv_cache(
            store_kv, kv_cache, slot_mapping, use_fp8_rope=use_fp8_rope, num_per_channels=num_per_channels
        )
    )
    total_bytes = store_kv.numel() * store_kv.element_size() + n * bytes_per_token
    benchmark_record(
        kernel='cache',
        operation='store_kv_cache',
        params={'block_size': block_size, 'n': n, 'num_per_channels': num_per_channels},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1000000000.0 / (t_us / 1000000.0)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('use_fp8_rope,num_per_channels', [(False, 64), (True, 32)])
@pytest.mark.parametrize('block_size,num_blocks,n', [(64, 2333, 65536), (32, 4666, 65536), (129, 2333, 65536)])
def test_load_kv_cache_benchmark(
    block_size: int,
    num_blocks: int,
    n: int,
    use_fp8_rope: bool,
    num_per_channels: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    if not use_fp8_rope:
        bytes_per_token = 448 + 64 * 2 + 8
        alignment_size = 576
    else:
        bytes_per_token = 512 + 16
        alignment_size = 512
    kv_cache = _make_kv_cache_4d(num_blocks, block_size, bytes_per_token, alignment_size=alignment_size)
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32, device=_benchmark_device())[:n]
    store_kv = torch.randn(n, 1, 512, dtype=torch.bfloat16, device=_benchmark_device())
    store_kv_cache(store_kv, kv_cache, slot_mapping, use_fp8_rope=use_fp8_rope, num_per_channels=num_per_channels)
    t_us = benchmark_timer(
        lambda: load_kv_cache(kv_cache, slot_mapping, use_fp8_rope=use_fp8_rope, num_per_channels=num_per_channels)
    )
    total_bytes = n * bytes_per_token + n * 512 * 2
    benchmark_record(
        kernel='cache',
        operation='load_kv_cache',
        params={'block_size': block_size, 'n': n, 'num_per_channels': num_per_channels},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1000000000.0 / (t_us / 1000000.0)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('mode', ['fp8', 'mxfp8', 'fp4'])
@pytest.mark.parametrize('block_size,num_blocks,n', [(64, 2333, 65536), (32, 4666, 65536)])
def test_store_index_kv_cache_benchmark(
    block_size: int,
    num_blocks: int,
    n: int,
    mode: str,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    is_fp4 = mode == 'fp4'
    is_mxfp8 = mode == 'mxfp8'
    bytes_per_token = 64 + 4 if is_fp4 else 128 + 4
    kv_cache = torch.empty((num_blocks, block_size, 1, bytes_per_token), dtype=torch.uint8, device=_benchmark_device())
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32, device=_benchmark_device())[:n]
    store_kv = torch.randn(n, 1, 128, dtype=torch.bfloat16, device=_benchmark_device())
    t_us = benchmark_timer(
        lambda: store_index_kv_cache(store_kv, kv_cache, slot_mapping, is_fp4=is_fp4, is_mxfp8=is_mxfp8)
    )
    total_bytes = n * 128 * 2 + n * bytes_per_token
    benchmark_record(
        kernel='cache',
        operation=f'store_index_kv_cache_{mode}',
        params={'block_size': block_size, 'n': n},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1000000000.0 / (t_us / 1000000.0)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('mode', ['fp8', 'mxfp8', 'fp4'])
@pytest.mark.parametrize('block_size,num_blocks,n', [(64, 2333, 65536), (32, 4666, 65536)])
def test_load_index_kv_cache_benchmark(
    block_size: int,
    num_blocks: int,
    n: int,
    mode: str,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> Any:
    is_fp4 = mode == 'fp4'
    is_mxfp8 = mode == 'mxfp8'
    bytes_per_token = 64 + 4 if is_fp4 else 128 + 4
    kv_cache = torch.empty((num_blocks, block_size, 1, bytes_per_token), dtype=torch.uint8, device=_benchmark_device())
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32, device=_benchmark_device())[:n]
    store_kv = torch.randn(n, 1, 128, dtype=torch.bfloat16, device=_benchmark_device())
    store_index_kv_cache(store_kv, kv_cache, slot_mapping, is_fp4=is_fp4, is_mxfp8=is_mxfp8)
    t_us = benchmark_timer(lambda: load_index_kv_cache(kv_cache, slot_mapping, is_fp4=is_fp4, is_mxfp8=is_mxfp8))
    total_bytes = n * bytes_per_token + n * 128 * 2
    benchmark_record(
        kernel='cache',
        operation=f'load_index_kv_cache_{mode}',
        params={'block_size': block_size, 'n': n},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1000000000.0 / (t_us / 1000000.0)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('block_size,num_blocks,n', [(128, 2333, 65536)])
def test_store_kv_cache_llama_benchmark(
    block_size: int, num_blocks: int, n: int, benchmark_timer: Callable[..., Any], benchmark_record: Callable[..., Any]
) -> Any:
    head_n, head_size = (4, 128)
    kv_cache = torch.zeros(
        (num_blocks, block_size, head_n, head_size), dtype=torch.bfloat16, device=_benchmark_device()
    )
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32, device=_benchmark_device())[:n]
    store_kv = torch.randn(n, head_n, head_size, dtype=torch.bfloat16, device=_benchmark_device())
    t_us = benchmark_timer(lambda: store_kv_cache_llama(store_kv, kv_cache, slot_mapping))
    total_bytes = store_kv.numel() * 2 * 2
    benchmark_record(
        kernel='cache',
        operation='store_kv_cache_LLaMA',
        params={'block_size': block_size, 'n': n},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1000000000.0 / (t_us / 1000000.0)},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('block_size,num_blocks,n', [(128, 2333, 65536)])
def test_load_kv_cache_llama_benchmark(
    block_size: int, num_blocks: int, n: int, benchmark_timer: Callable[..., Any], benchmark_record: Callable[..., Any]
) -> Any:
    head_n, head_size = (4, 128)
    kv_cache = torch.zeros(
        (num_blocks, block_size, head_n, head_size), dtype=torch.bfloat16, device=_benchmark_device()
    )
    slot_mapping = torch.randperm(num_blocks * block_size, dtype=torch.int32, device=_benchmark_device())[:n]
    store_kv = torch.randn(n, head_n, head_size, dtype=torch.bfloat16, device=_benchmark_device())
    store_kv_cache_llama(store_kv, kv_cache, slot_mapping)
    t_us = benchmark_timer(lambda: load_kv_cache_llama(kv_cache, head_n, slot_mapping))
    total_bytes = store_kv.numel() * 2 * 2
    benchmark_record(
        kernel='cache',
        operation='load_kv_cache_LLaMA',
        params={'block_size': block_size, 'n': n},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1000000000.0 / (t_us / 1000000.0)},
    )
