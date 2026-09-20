import os

import pytest
import torch

from cann_ops_tilelang import engram_grad_w_reduce
from cann_ops_tilelang.config import clear_runtime_caches, get_device
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_hidden_sizes, generate_num_sms, get_test_level
from tests.testing.numeric import calc_diff, count_bytes

os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def grad_w_reduce_ref(
    grad_w_partial,
    weight_hidden,
    weight_embed,
    grad_weight_hidden,
    grad_weight_embed,
):
    grad_w_sum = grad_w_partial.sum(0)
    grad_weight_hidden += grad_w_sum * weight_embed.float()
    grad_weight_embed += grad_w_sum * weight_hidden.float()


def generate_test_data(params):
    hidden_size = params['hidden']
    hc_mult = 4
    num_persistent_blocks = params['num_sms']
    device = get_device()
    grad_w_partial = torch.randn(
        num_persistent_blocks,
        hc_mult,
        hidden_size,
        dtype=torch.float32,
        device=device,
    )
    weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=device)
    return grad_w_partial, weight_hidden, weight_embed


def generate_test_params(level: int) -> list[dict]:
    return [
        {
            'hidden': hidden_size,
            'num_sms': num_sms,
        }
        for hidden_size in generate_hidden_sizes(256)
        for num_sms in generate_num_sms(level)
    ]


def _set_num_sms(monkeypatch, num_sms: int) -> None:
    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', str(num_sms * 2))
    clear_runtime_caches()


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_grad_w_reduce(params, monkeypatch):
    hidden_size = params['hidden']
    device = get_device()

    _set_num_sms(monkeypatch, params['num_sms'])
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    hc_mult = grad_w_partial.shape[1]

    grad_wh_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_we_ref = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_weight_hidden = grad_wh_ref.clone()
    grad_weight_embed = grad_we_ref.clone()
    grad_w_reduce_ref(grad_w_partial, weight_hidden, weight_embed, grad_wh_ref, grad_we_ref)
    engram_grad_w_reduce(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
    diff_wh = calc_diff(grad_weight_hidden.cpu(), grad_wh_ref.cpu())
    assert diff_wh < 1e-10, f'grad_wh mismatch: {diff_wh:.6e}'
    diff_we = calc_diff(grad_weight_embed.cpu(), grad_we_ref.cpu())
    assert diff_we < 1e-10, f'grad_we mismatch: {diff_we:.6e}'


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_engram_grad_w_reduce_benchmark(benchmark_timer, benchmark_record, params, monkeypatch):
    hidden_size = params['hidden']
    device = get_device()

    _set_num_sms(monkeypatch, params['num_sms'])
    grad_w_partial, weight_hidden, weight_embed = generate_test_data(params)
    hc_mult = grad_w_partial.shape[1]
    grad_weight_hidden = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)
    grad_weight_embed = torch.randn(hc_mult, hidden_size, dtype=torch.float32, device=device)

    t_us = benchmark_timer(
        lambda: engram_grad_w_reduce(
            grad_w_partial,
            weight_hidden,
            weight_embed,
            grad_weight_hidden,
            grad_weight_embed,
        )
    )

    num_bytes = count_bytes(grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='engram_grad_w_reduce',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )
