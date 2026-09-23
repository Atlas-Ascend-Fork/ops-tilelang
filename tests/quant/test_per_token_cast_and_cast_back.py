import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.utils import str_to_dtype
from tests.quant.cast_ref import cast, cast_back
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tests.testing.numeric import assert_equal, check_bias, count_bytes

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']

    return torch.randn((num_tokens, hidden), dtype=str_to_dtype(in_dtype), device=get_device())


def generate_test_params(level: int) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'in_dtype': in_dtype,
            'num_per_channels': num_per_channels,
            'fmt': fmt,
            'round_sf': round_sf,
        }
        for num_tokens in generate_num_tokens(level)
        for hidden_size in generate_hidden_sizes(64)
        for in_dtype in ('bf16', 'fp16', 'fp32')
        for num_per_channels in (32, 64, 128)
        for fmt in ('e4m3', 'e2m1')
        for round_sf in ((True,) if level == 0 else (False, True))
        if hidden_size % num_per_channels == 0
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_token_cast_and_cast_back(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    num_per_channels = params['num_per_channels']
    fmt = params['fmt']
    round_sf = params['round_sf']

    original_x = generate_test_data(params)

    def func(x):
        return cann_ops_tilelang.per_token_cast_and_cast_back(
            x, fmt, num_per_channels=num_per_channels, round_sf=round_sf
        )

    x = original_x.clone()
    func(x)

    x_quant, x_sf = cast(original_x, fmt, (1, num_per_channels), round_sf=round_sf)
    x_ref = cast_back((x_quant, x_sf), in_dtype, (1, num_per_channels))

    assert_equal(x, x_ref)

    check_bias(x, original_x)

    if num_tokens > 0:
        x_non_contiguous = torch.randn((num_tokens, hidden * 2), dtype=original_x.dtype, device=original_x.device)[
            :, :hidden
        ]
        x_non_contiguous.copy_(original_x)
        func(x_non_contiguous)
        assert_equal(x_non_contiguous, x, check_stride=False)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_per_token_cast_and_cast_back_benchmark(benchmark_timer, benchmark_record, params):
    num_per_channels = params['num_per_channels']
    fmt = params['fmt']
    round_sf = params['round_sf']

    x = generate_test_data(params)

    def func():
        return cann_ops_tilelang.per_token_cast_and_cast_back(
            x, fmt, num_per_channels=num_per_channels, round_sf=round_sf
        )

    t_us = benchmark_timer(func)
    num_bytes = 2 * count_bytes(x)

    benchmark_record(
        kernel='per_token_cast_and_cast_back',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
