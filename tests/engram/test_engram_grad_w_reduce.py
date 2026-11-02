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


def _compile_inputs_and_storage(partials=7, hidden=256):
    inputs = list(generate_test_data({'hidden': hidden, 'num_sms': partials}))
    storages = []
    for _ in range(2):
        # Contiguous offset views also check that surrounding storage is untouched.
        storage = torch.randn(4 * hidden + 2, device=inputs[0].device)
        inputs.append(storage[1:-1].view(4, hidden))
        storages.append(storage)
    return inputs, storages


def _compile_inputs(partials=7, hidden=256):
    inputs, _ = _compile_inputs_and_storage(partials, hidden)
    return inputs


def _assert_accumulation(call, inputs, expected, storages):
    readonly = [tensor.clone() for tensor in inputs[:3]]
    surrounding = [(storage, storage[[0, -1]].clone()) for storage in storages]
    pointers = [tensor.data_ptr() for tensor in inputs[3:]]
    grad_w_reduce_ref(*inputs[:3], *expected)
    assert call(*inputs) is None
    for actual, reference, pointer in zip(inputs[3:], expected, pointers, strict=True):
        assert actual.data_ptr() == pointer
        assert actual.dtype == torch.float32 and actual.is_contiguous()
        assert calc_diff(actual.cpu(), reference.cpu()) < 1e-10
    for actual, before in zip(inputs[:3], readonly, strict=True):
        assert torch.equal(actual, before)
    for storage, before in surrounding:
        assert torch.equal(storage[[0, -1]], before)


@pytest.mark.parametrize('partials,hidden', [(1, 256), (7, 256), (37, 2560), (38, 3072), (75, 6144), (7, 4096)])
def test_engram_grad_w_reduce_compile(partials, hidden):
    inputs, storages = _compile_inputs_and_storage(partials, hidden)
    expected = [tensor.clone() for tensor in inputs[3:]]
    compiled = torch.compile(engram_grad_w_reduce, backend='npugraph_ex', fullgraph=True, dynamic=False)
    for step in range(3):
        if step == 1:
            for tensor in inputs[:3]:
                tensor.copy_(torch.randn_like(tensor))
        elif step == 2:
            replacement = _compile_inputs(partials, hidden)
            inputs[:3] = replacement[:3]
        with torch.inference_mode():
            _assert_accumulation(compiled, inputs, expected, storages)
    # Switching persistent buffers must update the new caller-owned storage.
    replacement, storages = _compile_inputs_and_storage(partials, hidden)
    expected = [tensor.clone() for tensor in replacement[3:]]
    with torch.inference_mode():
        _assert_accumulation(compiled, replacement, expected, storages)


def test_engram_grad_w_reduce_opcheck():
    from cann_ops_tilelang.engram.engram_grad_w_reduce import _compiled_engram_grad_w_reduce

    torch.library.opcheck(
        _compiled_engram_grad_w_reduce,
        tuple(_compile_inputs()),
        test_utils=('test_schema', 'test_faketensor'),
    )


def test_engram_grad_w_reduce_compile_composition():
    def model(partial, wh, we, gwh, gwe):
        engram_grad_w_reduce(partial * 0.5, wh, we, gwh, gwe)
        return gwh + gwe

    inputs = _compile_inputs()
    expected = [tensor.clone() for tensor in inputs[3:]]
    compiled = torch.compile(model, backend='npugraph_ex', fullgraph=True, dynamic=False)
    for _ in range(3):
        inputs[0].copy_(torch.randn_like(inputs[0]))
        grad_w_reduce_ref(inputs[0] * 0.5, *inputs[1:3], *expected)
        with torch.inference_mode():
            actual = compiled(*inputs)
            torch.testing.assert_close(actual, expected[0] + expected[1], rtol=1e-5, atol=1e-5)
            for tensor, reference in zip(inputs[3:], expected, strict=True):
                assert calc_diff(tensor.cpu(), reference.cpu()) < 1e-10
            del actual


@pytest.mark.parametrize('case', ['empty', 'hidden', 'hc', 'rank', 'dtype', 'shape', 'layout'])
def test_engram_grad_w_reduce_invalid_contract(case):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from cann_ops_tilelang.engram.engram_grad_w_reduce import _engram_grad_w_reduce_fake

    inputs = _compile_inputs()
    device = inputs[0].device
    if case == 'empty':
        inputs[0] = torch.empty(0, 4, 256, device=device)
    elif case == 'hidden':
        inputs[0] = torch.empty(7, 4, 128, device=device)
    elif case == 'hc':
        inputs[0] = torch.empty(7, 2, 256, device=device)
    elif case == 'rank':
        inputs[0] = inputs[0].flatten()
    elif case == 'dtype':
        inputs[1] = inputs[1].float()
    elif case == 'shape':
        inputs[4] = inputs[4][:2]
    else:
        inputs[0] = torch.empty(7, 4, 512, device=device)[..., ::2]
    with pytest.raises((ValueError, TypeError)):
        engram_grad_w_reduce(*inputs)
    with FakeTensorMode() as mode:
        fake = [mode.from_tensor(tensor) for tensor in inputs]
        with pytest.raises((ValueError, TypeError)):
            _engram_grad_w_reduce_fake(*fake)
