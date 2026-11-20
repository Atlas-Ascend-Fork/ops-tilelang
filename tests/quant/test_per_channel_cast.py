import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.utils import str_to_dtype
from tests.quant.cast_ref import cast, cast_back
from tests.testing.bench import make_param_id
from tests.testing.generator import (
    generate_cast_config,
    generate_hidden_sizes,
    generate_num_tokens,
    generate_rand_float,
    get_test_level,
)
from tests.testing.numeric import assert_equal, check_bias, count_bytes
from tests.testing.quant import clear_unused_sf_col_pack

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    num_per_channels = params['num_per_channels']
    in_use_tma_aligned_col_major_sf = params['in_use_tma_aligned_col_major_sf']
    in_round_sf = params['in_round_sf']
    in_use_packed_ue8m0 = params['in_use_packed_ue8m0']

    in_with_sf_factor = in_dtype in ('e4m3', 'e2m1')
    source_dtype = 'bf16' if in_with_sf_factor else in_dtype
    x = generate_rand_float((num_tokens, hidden)).to(str_to_dtype(source_dtype))
    original_x = x

    if in_with_sf_factor:
        x = cast(
            x,
            in_dtype,
            (1, num_per_channels),
            use_tma_aligned_col_major_sf=in_use_tma_aligned_col_major_sf,
            round_sf=in_round_sf,
            use_packed_ue8m0=in_use_packed_ue8m0,
        )

    return x, original_x


def generate_test_params(level: int) -> list[dict]:
    params = [
        {
            'num_per_tokens': num_per_tokens,
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'round_sf': out_round_sf,
            'in_dtype': in_dtype,
            'num_per_channels': num_per_channels,
            'use_packed_ue8m0': out_use_packed_ue8m0,
            'in_use_tma_aligned_col_major_sf': in_use_tma_aligned_col_major_sf,
            'in_round_sf': in_round_sf,
            'in_use_packed_ue8m0': in_use_packed_ue8m0,
        }
        for num_per_tokens in (32, 128)
        for num_tokens in generate_num_tokens(level, 128, backward_only=True)
        for hidden_size in generate_hidden_sizes(128)
        for in_dtype in ('bf16', 'e4m3')
        for in_use_tma_aligned_col_major_sf, in_round_sf, in_use_packed_ue8m0 in generate_cast_config(level, in_dtype)
        for out_round_sf, out_use_packed_ue8m0 in (((False, False), (True, True)) if level >= 1 else ((True, True),))
        for num_per_channels in ((num_per_tokens,) if in_dtype in ('e4m3',) else (None,))
    ]
    # Ascend 过滤规则：
    # ① 数学必要条件：num_tokens % num_per_tokens == 0。
    # 注：原 tma packed input 过滤（hidden % (num_per_channels×2) == 0）已移除，
    #   kernel 改用 ceil_div 分配 sf_in_ub，hidden=128+num_per_channels=128 时有 1 行（非 0）。
    # 注：packed 配对过滤（num_tokens % (2×num_per_tokens) == 0）已移除——non-psum kernel
    #   对尾部打包行用 valid_rows 截断（见 per_channel_cast_asc.py），
    #   num_tokens=8064、num_per_tokens=128 的 packed 组合（8064%256=128≠0）实测通过。
    params = [p for p in params if p['num_tokens'] % p['num_per_tokens'] == 0]
    return params


def _per_channel_cast_args(x, params):
    data, scale = x if isinstance(x, tuple) else (x, None)
    args = (data, scale, 'e4m3', params['num_per_tokens'])
    kwargs = dict(
        round_sf=params['round_sf'],
        num_per_channels=params['num_per_channels'],
        use_packed_ue8m0=params['use_packed_ue8m0'],
    )
    return args, kwargs


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_channel_cast(params):
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    x, original_x = generate_test_data(params)
    args, kwargs = _per_channel_cast_args(x, params)
    x_fp8, per_channel_sf_inv = cann_ops_tilelang.per_channel_cast(*args, **kwargs)
    x_fp8_ref, per_channel_sf_inv_ref = cast(
        x,
        'e4m3',
        block_size=(num_per_tokens, 1),
        x_block_size=(1, num_per_channels) if num_per_channels is not None else None,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    assert_equal(x_fp8, x_fp8_ref)

    # Check bias
    x_casted_back = cast_back((x_fp8_ref, per_channel_sf_inv_ref), 'fp32', (num_per_tokens, 1))
    check_bias(x_casted_back, original_x)

    if use_packed_ue8m0:
        per_channel_sf_inv = clear_unused_sf_col_pack(per_channel_sf_inv, params['num_tokens'], num_per_tokens)
        per_channel_sf_inv_ref = clear_unused_sf_col_pack(per_channel_sf_inv_ref, params['num_tokens'], num_per_tokens)
    assert_equal(per_channel_sf_inv, per_channel_sf_inv_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_per_channel_cast_benchmark(benchmark_timer, benchmark_record, params):
    x, _ = generate_test_data(params)
    args, kwargs = _per_channel_cast_args(x, params)
    x_fp8, per_channel_sf_inv = cann_ops_tilelang.per_channel_cast(*args, **kwargs)
    t_us = benchmark_timer(lambda: cann_ops_tilelang.per_channel_cast(*args, **kwargs))
    num_bytes = count_bytes(x, x_fp8, per_channel_sf_inv)

    benchmark_record(
        kernel='per_channel_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


def _refresh_per_channel_cast_input(source, raw, step, prequantized):
    refreshed = cast(raw, 'e4m3', (1, 128), round_sf=True) if prequantized else raw
    if step != 1:
        return refreshed
    if prequantized:
        source[0].view(torch.uint8).copy_(refreshed[0].view(torch.uint8))
        source[1].copy_(refreshed[1])
    else:
        source.copy_(refreshed)
    return source


def _check_per_channel_cast_registration(source, token_group, kw):
    data, scale = source if isinstance(source, tuple) else (source, None)

    torch.library.opcheck(
        torch.ops.cann_ops_tilelang.per_channel_cast.default,
        (
            data,
            scale,
            'e4m3',
            token_group,
            kw['round_sf'],
            kw['num_per_channels'],
            kw['use_packed_ue8m0'],
        ),
        # FP8 allclose is unavailable in TorchNPU's standard schema checker.
        # The BF16 case covers schema; both cases verify input bytes in the caller.
        test_utils=('test_faketensor',) if scale is not None else ('test_schema', 'test_faketensor'),
    )


@pytest.mark.parametrize('prequantized', [False, True])
@torch.inference_mode()
def test_per_channel_cast_compile(prequantized):
    from cann_ops_tilelang.config import get_device
    from tests.testing.quant import call_quant_graph, check_quant_graph_replay, compile_quant_graph

    packed = prequantized
    token_group = 128 if prequantized else 32
    # Two token groups keep every packed scale byte meaningful in this core case.
    tokens = 256 if prequantized else 128
    channel_group = 128 if prequantized else None
    kw = dict(round_sf=packed, num_per_channels=channel_group, use_packed_ue8m0=packed)
    compiled = compile_quant_graph(cann_ops_tilelang.per_channel_cast)
    source = None
    records = []
    for step in range(3):
        raw = torch.randn((tokens, 256), dtype=torch.bfloat16, device=get_device())
        source = _refresh_per_channel_cast_input(source, raw, step, prequantized)
        inputs = source if isinstance(source, tuple) else (source,)
        snapshots = tuple(value.clone() for value in inputs)
        expected = cast(
            source,
            'e4m3',
            (token_group, 1),
            x_block_size=(1, 128) if prequantized else None,
            round_sf=packed,
            use_packed_ue8m0=packed,
        )
        data, scale = source if isinstance(source, tuple) else (source, None)
        result, events = call_quant_graph(compiled, data, scale, 'e4m3', token_group, **kw)
        records.append(events)
        for actual, reference in zip(result, expected, strict=True):
            assert_equal(actual, reference)
        for value, before in zip(inputs, snapshots, strict=True):
            assert_equal(value, before)
        del result, actual
    check_quant_graph_replay(records)
    _check_per_channel_cast_registration(source, token_group, kw)


@torch.inference_mode()
def test_per_channel_cast_compile_special_values():
    from cann_ops_tilelang.config import get_device
    from tests.testing.quant import call_quant_graph, check_quant_graph_replay, compile_quant_graph

    compiled = compile_quant_graph(cann_ops_tilelang.per_channel_cast)
    x = torch.zeros((128, 256), dtype=torch.bfloat16, device=get_device())
    finite_columns = torch.ones(256, dtype=torch.bool, device=x.device)
    finite_columns[0] = finite_columns[32] = finite_columns[64] = False
    records = []
    for step in range(3):
        if step == 1:
            x.fill_(0.5)
        elif step == 2:
            x = torch.full_like(x, -1.25)
        if step:
            x[:, 96:128].zero_()
            row = 0 if step == 1 else 32
            x[row, 0] = float('nan')
            x[row, 32] = float('inf')
            x[row, 64] = -float('inf')
        before = x.clone()
        reference = cast(x, 'e4m3', (32, 1), round_sf=True)
        eager = cann_ops_tilelang.per_channel_cast(x, None, 'e4m3', 32, round_sf=True)
        result, events = call_quant_graph(compiled, x, None, 'e4m3', 32, round_sf=True)
        records.append(events)
        for actual, expected in zip(result, eager, strict=True):
            assert_equal(actual, expected)
        # Existing non-finite quantization differs from cast_ref. Check graph/eager above;
        # independently check every column for zero input, and finite-only columns otherwise.
        if step == 0:
            for actual, expected in zip(result, reference, strict=True):
                assert_equal(actual, expected)
        else:
            assert_equal(
                result[0].view(torch.uint8)[:, finite_columns], reference[0].view(torch.uint8)[:, finite_columns]
            )
            assert_equal(result[1][:, finite_columns], reference[1][:, finite_columns])
        assert_equal(x, before)
        del result, actual, eager, reference
    check_quant_graph_replay(records)
