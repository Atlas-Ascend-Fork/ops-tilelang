import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_hidden_sizes, get_test_level
from tests.testing.numeric import assert_equal, count_bytes

os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_data(params):
    hc_mult = params['hc']
    hidden_size = params['hidden']
    wh_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=get_device())
    we_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=get_device())
    return wh_data, we_data


def generate_test_params(level: int) -> list[dict]:
    del level
    return [{'hc': hc, 'hidden': hidden_size} for hc in (4,) for hidden_size in generate_hidden_sizes(256)]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_fused_weight(params):
    wh_data, we_data = generate_test_data(params)

    ref = wh_data.float() * we_data.float()
    out = cann_ops_tilelang.engram_fused_weight(wh_data, we_data)

    assert out.dtype == torch.float32
    assert_equal(out, ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_engram_fused_weight_benchmark(benchmark_timer, benchmark_record, params):
    wh_data, we_data = generate_test_data(params)
    out = cann_ops_tilelang.engram_fused_weight(wh_data, we_data)

    t_us = benchmark_timer(lambda: cann_ops_tilelang.engram_fused_weight(wh_data, we_data))

    num_bytes = count_bytes(wh_data, we_data, out)
    benchmark_record(
        kernel='engram_fused_weight',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
