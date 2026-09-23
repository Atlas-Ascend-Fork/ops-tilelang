import os

import pytest

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


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_channel_cast(params):
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    x, original_x = generate_test_data(params)
    x_fp8, per_channel_sf_inv = cann_ops_tilelang.per_channel_cast(
        x,
        'e4m3',
        num_per_tokens,
        round_sf=round_sf,
        num_per_channels=num_per_channels,
        use_packed_ue8m0=use_packed_ue8m0,
    )
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
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    x, _ = generate_test_data(params)
    x_fp8, per_channel_sf_inv = cann_ops_tilelang.per_channel_cast(
        x,
        'e4m3',
        num_per_tokens,
        round_sf=round_sf,
        num_per_channels=num_per_channels,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    t_us = benchmark_timer(
        lambda: cann_ops_tilelang.per_channel_cast(
            x,
            'e4m3',
            num_per_tokens,
            round_sf=round_sf,
            num_per_channels=num_per_channels,
            use_packed_ue8m0=use_packed_ue8m0,
        )
    )
    num_bytes = count_bytes(x, x_fp8, per_channel_sf_inv)

    benchmark_record(
        kernel='per_channel_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
