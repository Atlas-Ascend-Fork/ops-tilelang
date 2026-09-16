import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device, get_num_vector_cores
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_hidden_sizes, get_test_level
from tests.testing.numeric import calc_diff, count_bytes

os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def engram_grad_w_reduce_ref(
    grad_w_partial,
    weight_hidden,
    weight_embed,
    grad_weight_hidden,
    grad_weight_embed,
):
    grad_w_sum = grad_w_partial.sum(0)
    grad_weight_hidden.add_(grad_w_sum * weight_embed.float())
    grad_weight_embed.add_(grad_w_sum * weight_hidden.float())


def generate_num_partials(level: int) -> list[int]:
    gate_num_partials = (get_num_vector_cores() - 12) // 2
    values = [gate_num_partials]
    if level >= 1:
        values.insert(0, max(gate_num_partials - 9, 1))
    if level >= 2:
        values.insert(0, 1)
    return list(dict.fromkeys(values))


def generate_test_params(level: int) -> list[dict]:
    return [
        {'hidden': hidden_size, 'num_partials': num_partials}
        for hidden_size in generate_hidden_sizes(256)
        for num_partials in generate_num_partials(level)
    ]


def generate_test_data(params):
    torch.manual_seed(0)
    hidden_size = params['hidden']
    num_partials = params['num_partials']
    hc_mult = 4
    device = get_device()
    grad_w_partial = torch.randn(num_partials, hc_mult, hidden_size, dtype=torch.float32, device=device)
    weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    return grad_w_partial, weight_hidden, weight_embed


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_grad_w_reduce(params):
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    output_shape = weight_hidden.shape
    device = weight_hidden.device
    grad_weight_hidden_ref = torch.randn(output_shape, dtype=torch.float32, device=device)
    grad_weight_embed_ref = torch.randn(output_shape, dtype=torch.float32, device=device)
    grad_weight_hidden = grad_weight_hidden_ref.clone()
    grad_weight_embed = grad_weight_embed_ref.clone()

    engram_grad_w_reduce_ref(
        grad_w_partial,
        weight_hidden,
        weight_embed,
        grad_weight_hidden_ref,
        grad_weight_embed_ref,
    )
    cann_ops_tilelang.engram_grad_w_reduce(
        grad_w_partial,
        weight_hidden,
        weight_embed,
        grad_weight_hidden,
        grad_weight_embed,
    )
    for name, actual, expected in (
        ('grad_weight_hidden', grad_weight_hidden, grad_weight_hidden_ref),
        ('grad_weight_embed', grad_weight_embed, grad_weight_embed_ref),
    ):
        diff = calc_diff(actual, expected)
        assert diff < 1e-10, f'{name} mismatch: {diff:.6e}'


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_engram_grad_w_reduce_benchmark(benchmark_timer, benchmark_record, params):
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    output_shape = weight_hidden.shape
    device = weight_hidden.device
    grad_weight_hidden = torch.randn(output_shape, dtype=torch.float32, device=device)
    grad_weight_embed = torch.randn(output_shape, dtype=torch.float32, device=device)
    time_us = benchmark_timer(
        lambda: cann_ops_tilelang.engram_grad_w_reduce(
            grad_w_partial,
            weight_hidden,
            weight_embed,
            grad_weight_hidden,
            grad_weight_embed,
        )
    )
    num_bytes = count_bytes(
        grad_w_partial,
        weight_hidden,
        weight_embed,
        grad_weight_hidden,
        grad_weight_embed,
    )
    benchmark_record(
        kernel='engram_grad_w_reduce',
        operation='fwd',
        params=params,
        time_us=time_us,
        bandwidth_gbs=num_bytes / time_us / 1e3,
    )
