import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device
from tests.engram.engram_gate_ref import engram_gate_ref
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tests.testing.numeric import calc_diff, count_bytes

os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_data(params):
    if get_device().type == 'npu':
        torch.manual_seed(0)
    num_tokens = params['num_tokens']
    hc_mult = params['hc']
    hidden_size = params['hidden']
    eps = 1e-20
    clamp_value = 1e-6
    x_data = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=get_device())
    kv_data = torch.randn(num_tokens, (hc_mult + 1) * hidden_size, dtype=torch.bfloat16, device=get_device())
    wh_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=get_device())
    we_data = torch.randn(hc_mult, hidden_size, dtype=torch.bfloat16, device=get_device())
    weight_fused = wh_data.float() * we_data.float()
    grad_out = torch.randn(num_tokens, hc_mult, hidden_size, dtype=torch.bfloat16, device=get_device())
    return x_data, kv_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value


def generate_test_params(level: int) -> list[dict]:
    return [
        {'num_tokens': t, 'hc': hc, 'hidden': hidden_size}
        for t in generate_num_tokens(level)
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(256)
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_gate_bwd(params):
    x_data, kv_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value = generate_test_data(params)

    x_ref = x_data.clone().requires_grad_(True)
    kv_ref = kv_data.clone().requires_grad_(True)
    wh_ref = wh_data.float().requires_grad_(True)
    we_ref = we_data.float().requires_grad_(True)
    output_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_ref,
        kv_ref,
        wh_ref,
        we_ref,
        clamp_value,
        eps,
        save_for_backward=True,
    )
    output_ref.backward(grad_out)

    grad_x, grad_kv, grad_w_partial = cann_ops_tilelang.engram_gate_bwd(
        grad_out,
        x_data,
        kv_data,
        weight_fused,
        dot_ref,
        gate_score_ref,
        rstd_x_ref,
        rstd_k_ref,
        clamp_value,
    )
    grad_w_fused = grad_w_partial.sum(0)
    grad_wh = grad_w_fused * we_data.float()
    grad_we = grad_w_fused * wh_data.float()

    diff_x = calc_diff(grad_x, x_ref.grad)
    assert diff_x < 1e-8, f'grad_x mismatch: {diff_x:.6e}'
    diff_kv = calc_diff(grad_kv, kv_ref.grad)
    assert diff_kv < 1e-8, f'grad_kv mismatch: {diff_kv:.6e}'
    diff_wh = calc_diff(grad_wh, wh_ref.grad)
    assert diff_wh < 1e-8, f'grad_wh mismatch: {diff_wh:.6e}'
    diff_we = calc_diff(grad_we, we_ref.grad)
    assert diff_we < 1e-8, f'grad_we mismatch: {diff_we:.6e}'


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_engram_gate_bwd_benchmark(benchmark_timer, benchmark_record, params):
    x_data, kv_data, wh_data, we_data, weight_fused, grad_out, eps, clamp_value = generate_test_data(params)

    _, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_data,
        kv_data,
        wh_data,
        we_data,
        clamp_value,
        eps,
        save_for_backward=True,
    )
    grad_x, grad_kv, grad_w_partial = cann_ops_tilelang.engram_gate_bwd(
        grad_out,
        x_data,
        kv_data,
        weight_fused,
        dot_ref,
        gate_score_ref,
        rstd_x_ref,
        rstd_k_ref,
        clamp_value,
    )
    grad_w_fused = grad_w_partial.sum(0)
    grad_wh = grad_w_fused * we_data.float()
    grad_we = grad_w_fused * wh_data.float()

    def run_backward():
        return cann_ops_tilelang.engram_gate_bwd(
            grad_out,
            x_data,
            kv_data,
            weight_fused,
            dot_ref,
            gate_score_ref,
            rstd_x_ref,
            rstd_k_ref,
            clamp_value,
        )

    t_us = benchmark_timer(run_backward)
    num_bytes = count_bytes(
        grad_out,
        x_data,
        kv_data,
        weight_fused,
        dot_ref,
        gate_score_ref,
        rstd_x_ref,
        rstd_k_ref,
        grad_x,
        grad_kv,
        grad_wh,
        grad_we,
    )
    benchmark_record(
        kernel='engram_gate_bwd',
        operation='bwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
