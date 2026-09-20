import os

import pytest
import torch

from cann_ops_tilelang import (
    engram_sinkhorn_finalize,
    engram_sinkhorn_momentum_update,
    engram_sinkhorn_step,
)
from cann_ops_tilelang.config import get_device
from tests.engram.engram_sinkhorn_ref import (
    engram_sinkhorn_finalize_ref,
    engram_sinkhorn_momentum_update_ref,
    engram_sinkhorn_step_ref,
)
from tests.testing.bench import make_param_id
from tests.testing.generator import get_test_level
from tests.testing.numeric import calc_diff, count_bytes

os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu

_DTYPES = {
    'bf16': torch.bfloat16,
    'fp32': torch.float32,
}


def generate_momentum_test_data(params):
    numel = params['numel']
    grad_dtype = _DTYPES[params['grad_dtype']]
    momentum_dtype = _DTYPES[params['momentum_dtype']]
    nesterov = params['nesterov']
    beta1 = params['beta1']

    grad = torch.randn(numel, dtype=grad_dtype, device=get_device())
    momentum_buffer = torch.randn(numel, dtype=momentum_dtype, device=get_device())
    return grad, momentum_buffer, beta1, nesterov


def generate_momentum_test_params(level: int) -> list[dict]:
    sizes = [1001 * 256, 10001 * 256, 100001 * 512, 1000001 * 512]
    grad_dtypes = ('fp32',) if level == 0 else ('bf16', 'fp32')
    return [
        {
            'numel': numel,
            'grad_dtype': grad_dtype,
            'momentum_dtype': momentum_dtype,
            'nesterov': nesterov,
            'beta1': beta1,
        }
        for numel in sizes
        for grad_dtype in grad_dtypes
        for momentum_dtype in ('bf16', 'fp32')
        for nesterov in (False, True)
        for beta1 in (0.95,)
    ]


@pytest.mark.parametrize('params', generate_momentum_test_params(get_test_level()), ids=make_param_id)
def test_engram_sinkhorn_momentum_update(params):
    grad, momentum_buffer, beta1, nesterov = generate_momentum_test_data(params)
    momentum_buffer_ref = momentum_buffer.clone()

    sinkhorn_input_ref = engram_sinkhorn_momentum_update_ref(grad, momentum_buffer_ref, beta1, nesterov)
    sinkhorn_input = engram_sinkhorn_momentum_update(grad, momentum_buffer, beta1, nesterov)

    diff_momentum = calc_diff(momentum_buffer, momentum_buffer_ref)
    diff_sinkhorn = calc_diff(sinkhorn_input, sinkhorn_input_ref)
    assert diff_momentum < 5e-8, f'momentum_buffer mismatch: {diff_momentum:.6e}'
    assert diff_sinkhorn < 5e-8, f'sinkhorn_input mismatch: {diff_sinkhorn:.6e}'


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_momentum_test_params(0), ids=make_param_id)
def test_engram_sinkhorn_momentum_update_benchmark(benchmark_timer, benchmark_record, params):
    grad, momentum_buffer, beta1, nesterov = generate_momentum_test_data(params)
    sinkhorn_input = engram_sinkhorn_momentum_update(grad, momentum_buffer, beta1, nesterov)

    time_us = benchmark_timer(lambda: engram_sinkhorn_momentum_update(grad, momentum_buffer, beta1, nesterov))

    num_bytes = count_bytes(grad, momentum_buffer, momentum_buffer, sinkhorn_input)
    bandwidth_gbs = num_bytes / time_us / 1e3
    benchmark_record(
        kernel='engram_sinkhorn',
        operation='momentum_update',
        params=params,
        time_us=time_us,
        bandwidth_gbs=bandwidth_gbs,
    )


def generate_row_step_test_data(params):
    num_rows = params['num_rows']
    head_dim = params['head_dim']
    matrix = torch.randn(num_rows, head_dim, dtype=torch.float32, device=get_device())
    row_scale = torch.rand(num_rows, dtype=torch.float32, device=get_device())
    col_scale = torch.rand(head_dim, dtype=torch.float32, device=get_device())
    return matrix, row_scale, col_scale, params['eps']


def generate_row_step_test_params(level: int) -> list[dict]:
    return [
        {'num_rows': num_rows, 'head_dim': head_dim, 'eps': eps}
        for num_rows in [1001, 10001, 100001]
        for head_dim in [256, 5120]
        for eps in (1e-20,)
    ]


@pytest.mark.parametrize('params', generate_row_step_test_params(get_test_level()), ids=make_param_id)
def test_engram_sinkhorn_step(params):
    matrix, row_scale, col_scale, eps = generate_row_step_test_data(params)
    row_scale_ref = row_scale.clone()

    row_norm_ref, col_sumsq_ref = engram_sinkhorn_step_ref(matrix, row_scale_ref, col_scale, eps)
    row_norm, col_sumsq = engram_sinkhorn_step(matrix, row_scale, col_scale, eps)

    assert calc_diff(row_norm, row_norm_ref) < 1e-10
    assert calc_diff(row_scale, row_scale_ref) < 1e-10
    assert calc_diff(col_sumsq, col_sumsq_ref) < 1e-10


@pytest.mark.parametrize('params', generate_row_step_test_params(get_test_level()), ids=make_param_id)
def test_engram_sinkhorn_finalize(params):
    matrix, row_scale, col_scale, eps = generate_row_step_test_data(params)
    alpha = params['head_dim'] ** 0.5

    out_ref = engram_sinkhorn_finalize_ref(matrix, row_scale, col_scale, eps, alpha)
    out = engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha)

    assert calc_diff(out, out_ref) < 1e-10


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_row_step_test_params(0), ids=make_param_id)
def test_engram_sinkhorn_step_benchmark(benchmark_timer, benchmark_record, params):
    matrix, row_scale, col_scale, eps = generate_row_step_test_data(params)

    row_norm, col_sumsq = engram_sinkhorn_step(matrix, row_scale, col_scale, eps)

    time_us = benchmark_timer(lambda: engram_sinkhorn_step(matrix, row_scale, col_scale, eps))

    num_bytes = count_bytes(matrix, row_scale, col_scale, row_scale, row_norm, col_sumsq)
    bandwidth_gbs = num_bytes / time_us / 1e3
    benchmark_record(
        kernel='engram_sinkhorn',
        operation='step',
        params=params,
        time_us=time_us,
        bandwidth_gbs=bandwidth_gbs,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_row_step_test_params(0), ids=make_param_id)
def test_engram_sinkhorn_finalize_benchmark(benchmark_timer, benchmark_record, params):
    matrix, row_scale, col_scale, eps = generate_row_step_test_data(params)
    alpha = params['head_dim'] ** 0.5
    out = torch.empty_like(matrix)
    engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha, out)

    time_us = benchmark_timer(lambda: engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha, out))

    num_bytes = count_bytes(matrix, row_scale, col_scale, out)
    bandwidth_gbs = num_bytes / time_us / 1e3
    benchmark_record(
        kernel='engram_sinkhorn',
        operation='finalize',
        params=params,
        time_us=time_us,
        bandwidth_gbs=bandwidth_gbs,
    )
