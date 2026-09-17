import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device
from tests.engram.engram_sinkhorn_step_ref import engram_sinkhorn_step_ref
from tests.testing.bench import make_param_id
from tests.testing.generator import get_test_level
from tests.testing.numeric import calc_diff, count_bytes

os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_params(level: int) -> list[dict]:
    source_cases = [
        {'num_rows': num_rows, 'head_dim': head_dim, 'eps': 1e-20}
        for num_rows in (1001, 10001, 100001)
        for head_dim in (256, 5120)
    ]
    if level == 0:
        return [source_cases[0], source_cases[3]]
    if level == 1:
        return source_cases
    return [
        {'num_rows': 0, 'head_dim': 256, 'eps': 1e-20},
        {'num_rows': 1, 'head_dim': 256, 'eps': 0.0},
        {'num_rows': 17, 'head_dim': 512, 'eps': 1e-6},
        *source_cases,
    ]


def generate_test_data(params):
    device = get_device()
    matrix = torch.randn(
        params['num_rows'],
        params['head_dim'],
        dtype=torch.float32,
        device=device,
    )
    row_scale = torch.rand(params['num_rows'], dtype=torch.float32, device=device)
    col_scale = torch.rand(params['head_dim'], dtype=torch.float32, device=device)
    return matrix, row_scale, col_scale


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_sinkhorn_step(params):
    matrix, row_scale, col_scale = generate_test_data(params)
    row_scale_ref = row_scale.clone()

    row_norm_ref, col_sumsq_ref = engram_sinkhorn_step_ref(
        matrix,
        row_scale_ref,
        col_scale,
        params['eps'],
    )
    row_norm, col_sumsq = cann_ops_tilelang.engram_sinkhorn_step(
        matrix,
        row_scale,
        col_scale,
        params['eps'],
    )

    assert row_norm.shape == row_scale.shape
    assert col_sumsq.shape == col_scale.shape
    assert row_norm.dtype == col_sumsq.dtype == torch.float32
    assert row_norm.device == col_sumsq.device == matrix.device
    assert calc_diff(row_norm, row_norm_ref) < 1e-10
    assert calc_diff(row_scale, row_scale_ref) < 1e-10
    assert calc_diff(col_sumsq, col_sumsq_ref) < 1e-10


def test_engram_sinkhorn_step_empty_rows():
    params = {'num_rows': 0, 'head_dim': 256, 'eps': 1e-20}
    matrix, row_scale, col_scale = generate_test_data(params)

    row_norm, col_sumsq = cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale, col_scale, params['eps'])

    assert row_norm.shape == (0,)
    assert torch.count_nonzero(col_sumsq).item() == 0


def test_engram_sinkhorn_step_rejects_invalid_inputs():
    device = get_device()
    matrix = torch.randn((17, 256), dtype=torch.float32, device=device)
    row_scale = torch.rand((17,), dtype=torch.float32, device=device)
    col_scale = torch.rand((256,), dtype=torch.float32, device=device)

    with pytest.raises(TypeError, match='torch.Tensor'):
        cann_ops_tilelang.engram_sinkhorn_step(None, row_scale, col_scale, 1e-20)
    with pytest.raises(ValueError, match='NPU tensor'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix.cpu(), row_scale, col_scale, 1e-20)
    with pytest.raises(ValueError, match='NPU tensor'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale.cpu(), col_scale, 1e-20)
    with pytest.raises(TypeError, match='torch.float32'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix.bfloat16(), row_scale, col_scale, 1e-20)
    with pytest.raises(TypeError, match='torch.float32'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale, col_scale.bfloat16(), 1e-20)
    with pytest.raises(ValueError, match='shape'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix.view(-1), row_scale, col_scale, 1e-20)
    with pytest.raises(ValueError, match='row_scale'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale[:-1], col_scale, 1e-20)
    with pytest.raises(ValueError, match='col_scale'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale, col_scale[:-1], 1e-20)
    with pytest.raises(ValueError, match='multiple of 256'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix[:, :-1].contiguous(), row_scale, col_scale[:-1], 1e-20)
    wide_matrix = torch.randn((1, 6144), dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match='shared UB capacity'):
        cann_ops_tilelang.engram_sinkhorn_step(
            wide_matrix,
            torch.rand((1,), device=device),
            torch.rand((6144,), device=device),
            1e-20,
        )
    with pytest.raises(ValueError, match='contiguous'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix.T, torch.rand(256, device=device), row_scale, 1e-20)
    with pytest.raises(ValueError, match='contiguous'):
        non_contiguous_row_scale = torch.rand((17, 2), device=device)[:, 0]
        cann_ops_tilelang.engram_sinkhorn_step(matrix, non_contiguous_row_scale, col_scale, 1e-20)
    with pytest.raises(ValueError, match='contiguous'):
        non_contiguous_col_scale = torch.rand((256, 2), device=device)[:, 0]
        cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale, non_contiguous_col_scale, 1e-20)
    with pytest.raises(TypeError, match='real number'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale, col_scale, None)
    with pytest.raises(ValueError, match='finite and non-negative'):
        cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale, col_scale, -1.0)
    for invalid_eps in (float('inf'), float('nan')):
        with pytest.raises(ValueError, match='finite and non-negative'):
            cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale, col_scale, invalid_eps)


def test_engram_sinkhorn_step_rejects_input_aliases():
    device = get_device()
    col_scale = torch.rand((256,), dtype=torch.float32, device=device)
    storage = torch.empty((17 * 256,), dtype=torch.float32, device=device)
    aliased_matrix = storage.view(17, 256)
    aliased_row_scale = storage[:17]
    with pytest.raises(ValueError, match='must not share storage'):
        cann_ops_tilelang.engram_sinkhorn_step(aliased_matrix, aliased_row_scale, col_scale, 1e-20)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_engram_sinkhorn_step_benchmark(benchmark_timer, benchmark_record, params):
    matrix, row_scale, col_scale = generate_test_data(params)
    row_norm, col_sumsq = cann_ops_tilelang.engram_sinkhorn_step(
        matrix,
        row_scale,
        col_scale,
        params['eps'],
    )

    time_us = benchmark_timer(
        lambda: cann_ops_tilelang.engram_sinkhorn_step(matrix, row_scale, col_scale, params['eps'])
    )
    num_bytes = count_bytes(matrix, row_scale, col_scale, row_scale, row_norm, col_sumsq)
    benchmark_record(
        kernel='engram_sinkhorn_step',
        operation='step',
        params=params,
        time_us=time_us,
        bandwidth_gbs=num_bytes / time_us / 1e3,
    )
