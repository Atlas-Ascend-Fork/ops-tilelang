import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang import config
from cann_ops_tilelang.config import get_device
from tests.engram.engram_gate_ref import compile_gate_for_test, engram_gate_ref, update_gate_inputs
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tests.testing.numeric import assert_equal, calc_diff, count_bytes

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
    return x_data, kv_data, wh_data, we_data, weight_fused, eps, clamp_value


def generate_test_params(level: int) -> list[dict]:
    return [
        {'num_tokens': t, 'hc': hc, 'hidden': hidden_size}
        for t in generate_num_tokens(level)
        for hc in (4,)
        for hidden_size in generate_hidden_sizes(256)
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_gate_fwd(params):
    x_data, kv_data, wh_data, we_data, weight_fused, eps, clamp_value = generate_test_data(params)

    out_ref, dot_ref, gate_score_ref, rstd_x_ref, rstd_k_ref = engram_gate_ref(
        x_data,
        kv_data,
        wh_data,
        we_data,
        clamp_value,
        eps,
        save_for_backward=True,
    )

    out_save, dot, gate_score, rstd_x, rstd_k = cann_ops_tilelang.engram_gate_fwd(
        x_data,
        kv_data,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward=True,
    )
    assert dot is not None and gate_score is not None and rstd_x is not None and rstd_k is not None
    diff_out = calc_diff(out_save, out_ref)
    assert diff_out < 2e-10, f'out_save mismatch: {diff_out:.6e}'
    diff_dot = calc_diff(dot, dot_ref)
    assert diff_dot < 2e-10, f'dot mismatch: {diff_dot:.6e}'
    diff_gate = calc_diff(gate_score, gate_score_ref)
    assert diff_gate < 2e-10, f'gate_score mismatch: {diff_gate:.6e}'
    diff_rstd_x = calc_diff(rstd_x, rstd_x_ref)
    assert diff_rstd_x < 2e-10, f'rstd_x mismatch: {diff_rstd_x:.6e}'
    diff_rstd_k = calc_diff(rstd_k, rstd_k_ref)
    assert diff_rstd_k < 2e-10, f'rstd_k mismatch: {diff_rstd_k:.6e}'

    out_no_save, dot_n, gate_score_n, rstd_x_n, rstd_k_n = cann_ops_tilelang.engram_gate_fwd(
        x_data,
        kv_data,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward=False,
    )
    assert dot_n is None and gate_score_n is None and rstd_x_n is None and rstd_k_n is None
    diff_out = calc_diff(out_no_save, out_ref)
    assert diff_out < 2e-10, f'out_no_save mismatch: {diff_out:.6e}'
    assert_equal(out_no_save, out_save)


@pytest.mark.parametrize('save_for_backward', [False, True], ids=['no_save', 'save'])
@torch.inference_mode()
def test_engram_gate_fwd_compile(save_for_backward):
    x_data, kv_data, wh_data, we_data, weight_fused, eps, clamp_value = generate_test_data(
        {'num_tokens': 128, 'hc': 4, 'hidden': 4096}
    )
    compiled = compile_gate_for_test(cann_ops_tilelang.engram_gate_fwd)
    for step in range(3):
        x_data, kv_data, wh_data, we_data, weight_fused = update_gate_inputs(
            step, x_data, kv_data, wh_data, we_data, weight_fused
        )

        references = engram_gate_ref(
            x_data,
            kv_data,
            wh_data,
            we_data,
            clamp_value,
            eps,
            save_for_backward=True,
        )
        if not save_for_backward:
            references = (references[0], None, None, None, None)
        outputs = compiled(x_data, kv_data, weight_fused, eps, clamp_value, save_for_backward)
        for name, actual, reference in zip(
            ('output', 'dot', 'gate_score', 'rstd_x', 'rstd_k'),
            outputs,
            references,
            strict=True,
        ):
            if reference is None:
                assert actual is None, f'{name} must be None when save_for_backward=False'
                continue
            assert actual is not None
            assert actual.shape == reference.shape
            assert actual.dtype == reference.dtype
            assert actual.device == reference.device
            diff = calc_diff(actual, reference)
            assert diff < 2e-10, f'{name} mismatch at step {step}: {diff:.6e}'
        # Release every graph output reference before the next call reuses storage.
        del outputs, actual


def _make_fwd_core_inputs(num_tokens):
    data = generate_test_data({'num_tokens': num_tokens, 'hc': 4, 'hidden': 4096})
    x_data, kv_data, wh_data, we_data, _, eps, clamp_value = data
    references = engram_gate_ref(
        hidden_states=x_data,
        kv=kv_data,
        weight_hidden=wh_data,
        weight_embed=we_data,
        clamp_value=clamp_value,
        eps=eps,
        save_for_backward=True,
    )
    return data, references


def _assert_fwd_core_outputs(outputs, references, save_for_backward):
    assert len(outputs) == len(references) == 5
    for index, (actual, reference) in enumerate(zip(outputs, references, strict=True)):
        if index and not save_for_backward:
            assert actual is None
            continue
        assert actual is not None
        assert actual.shape == reference.shape
        assert actual.dtype == reference.dtype
        assert actual.device == reference.device
        if reference.numel():
            diff = calc_diff(actual, reference)
            assert diff < 2e-10, f'output {index} mismatch after a core-count change: {diff:.6e}'


def _assert_fwd_core_rejected(compiled, data, save_for_backward, num_cores):
    x_data, kv_data, _, _, weight_fused, eps, clamp_value = data
    for operator in (cann_ops_tilelang.engram_gate_fwd, compiled):
        with pytest.raises(RuntimeError, match='divisible') as error:
            operator(x_data, kv_data, weight_fused, eps, clamp_value, save_for_backward)
        assert str(num_cores) in str(error.value)


@pytest.mark.parametrize('save_for_backward', [False, True], ids=['no_save', 'save'])
@torch.inference_mode()
def test_engram_gate_fwd_compile_core_config(save_for_backward, monkeypatch):
    torch.compiler.reset()
    try:
        monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        data, references = _make_fwd_core_inputs(128)
        x_data, kv_data, _, _, weight_fused, eps, clamp_value = data
        compiled = compile_gate_for_test(cann_ops_tilelang.engram_gate_fwd)
        default_ai_cores = config.get_num_ai_cores()
        assert default_ai_cores >= 4 and default_ai_cores % 2 == 0
        invalid_ai_cores = 13 if default_ai_cores >= 13 else default_ai_cores - 1
        for phase in ('default', 'reduced', 'invalid', 'reset'):
            if phase == 'reduced':
                config.set_num_ai_cores(default_ai_cores - 2)
                assert config.get_num_vector_cores() == 2 * default_ai_cores - 4
            if phase == 'invalid':
                config.set_num_ai_cores(invalid_ai_cores)
            if phase == 'reset':
                config.reset_runtime_config()
            if phase == 'invalid':
                _assert_fwd_core_rejected(compiled, data, save_for_backward, 2 * invalid_ai_cores)
            else:
                outputs = compiled(x_data, kv_data, weight_fused, eps, clamp_value, save_for_backward)
                _assert_fwd_core_outputs(outputs, references, save_for_backward)
                del outputs
    finally:
        config.reset_runtime_config()
        torch.compiler.reset()


@pytest.mark.parametrize('save_for_backward', [False, True], ids=['no_save', 'save'])
@torch.inference_mode()
def test_engram_gate_fwd_compile_empty_core_config(save_for_backward, monkeypatch):
    torch.compiler.reset()
    try:
        monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        data, references = _make_fwd_core_inputs(0)
        x_data, kv_data, _, _, weight_fused, eps, clamp_value = data
        invalid_ai_cores = 13 if config.get_num_ai_cores() >= 13 else 1
        config.set_num_ai_cores(invalid_ai_cores)
        compiled = compile_gate_for_test(cann_ops_tilelang.engram_gate_fwd)
        for vector_cap in (None, 'abc'):
            if vector_cap is not None:
                monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', vector_cap)
            config.clear_runtime_caches()
            for operator in (cann_ops_tilelang.engram_gate_fwd, compiled):
                outputs = operator(x_data, kv_data, weight_fused, eps, clamp_value, save_for_backward)
                _assert_fwd_core_outputs(outputs, references, save_for_backward)
                del outputs
    finally:
        config.reset_runtime_config()
        torch.compiler.reset()


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_engram_gate_fwd_benchmark(benchmark_timer, benchmark_record, params):
    x_data, kv_data, _, _, weight_fused, eps, clamp_value = generate_test_data(params)

    out_save, dot, gate_score, rstd_x, rstd_k = cann_ops_tilelang.engram_gate_fwd(
        x_data,
        kv_data,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward=True,
    )
    t_save_us = benchmark_timer(
        lambda: cann_ops_tilelang.engram_gate_fwd(
            x_data,
            kv_data,
            weight_fused,
            eps,
            clamp_value,
            save_for_backward=True,
        )
    )
    num_bytes_save = count_bytes(x_data, kv_data, weight_fused, out_save, dot, gate_score, rstd_x, rstd_k)
    benchmark_record(
        kernel='engram_gate_fwd',
        operation='fwd',
        params={**params, 'save': True},
        time_us=t_save_us,
        bandwidth_gbs=num_bytes_save / t_save_us / 1e3,
    )

    out_no_save = cann_ops_tilelang.engram_gate_fwd(
        x_data,
        kv_data,
        weight_fused,
        eps,
        clamp_value,
        save_for_backward=False,
    )[0]
    t_no_save_us = benchmark_timer(
        lambda: cann_ops_tilelang.engram_gate_fwd(
            x_data,
            kv_data,
            weight_fused,
            eps,
            clamp_value,
            save_for_backward=False,
        )
    )
    num_bytes_no_save = count_bytes(x_data, kv_data, weight_fused, out_no_save)
    benchmark_record(
        kernel='engram_gate_fwd',
        operation='fwd',
        params={**params, 'save': False},
        time_us=t_no_save_us,
        bandwidth_gbs=num_bytes_no_save / t_no_save_us / 1e3,
    )
