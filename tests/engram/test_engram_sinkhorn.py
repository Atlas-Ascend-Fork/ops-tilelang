# MIT License
#
# Copyright (c) 2026 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from cann_ops_tilelang import (
    engram_sinkhorn_finalize,
    engram_sinkhorn_momentum_update,
    engram_sinkhorn_step,
)
from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.engram.engram_sinkhorn import (
    engram_sinkhorn_finalize as registered_finalize,
)
from cann_ops_tilelang.engram.engram_sinkhorn import (
    engram_sinkhorn_momentum_update as registered_momentum_update,
)
from cann_ops_tilelang.engram.engram_sinkhorn import (
    engram_sinkhorn_step as registered_step,
)
from tests.engram.engram_sinkhorn_ref import (
    engram_sinkhorn_finalize_ref,
    engram_sinkhorn_momentum_update_ref,
    engram_sinkhorn_step_ref,
)
from tests.testing.bench import make_param_id
from tests.testing.generator import get_test_level
from tests.testing.numeric import calc_diff, count_bytes

MOMENTUM_CASES = [
    f'engram_sinkhorn_momentum_update:{grad}_{momentum}_{nesterov}'
    for grad, momentum, nesterov in product(('fp32', 'bf16'), ('fp32', 'bf16'), (False, True))
]
SINKHORN_CASES = MOMENTUM_CASES + ['engram_sinkhorn_step', 'engram_sinkhorn_finalize']


_PUBLIC_OPERATORS = {
    'engram_sinkhorn_momentum_update': engram_sinkhorn_momentum_update,
    'engram_sinkhorn_step': engram_sinkhorn_step,
    'engram_sinkhorn_finalize': engram_sinkhorn_finalize,
}

_REGISTERED_OPERATORS = {
    'engram_sinkhorn_momentum_update': registered_momentum_update,
    'engram_sinkhorn_step': registered_step,
    'engram_sinkhorn_finalize': registered_finalize,
}


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [clone(x) for x in value]
    return value


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for x in value:
            yield from _tensors(x)


@dataclass
class Case:
    name: str
    args: list
    mutated: tuple[int, ...]
    reference: object
    metric: str = 'diff'
    atol: float = 1e-5
    rtol: float = 1e-5
    tolerance: float = 1e-10
    aggregate: tuple[int, ...] = ()
    guards: list = field(default_factory=list)


def _make_engram_sinkhorn_momentum_update_case(name, variant, n, h, device, rand, buffer):
    grad_dtype, momentum_dtype, nesterov = variant.split('_')
    args = [rand(h, bf=grad_dtype == 'bf16'), rand(h, bf=momentum_dtype == 'bf16'), 0.9, nesterov == 'True']

    def reference(a):
        return engram_sinkhorn_momentum_update_ref(*a)

    return Case(name, args, (1,), reference, tolerance=5e-8)


def _make_engram_sinkhorn_step_case(name, variant, n, h, device, rand, buffer):
    args = [rand(n, h), rand(n, positive=True), rand(h, positive=True), 1e-06]

    def reference(a):
        row_norm = torch.linalg.vector_norm(a[1].unsqueeze(1) * a[0] * a[2], dim=1)
        a[1].div_(row_norm + a[3])
        col = (a[1].unsqueeze(1) * a[0] * a[2]).square().sum(0)
        return (row_norm, col)

    case = Case(name, args, (1,), reference)
    return case


def _make_engram_sinkhorn_finalize_case(name, variant, n, h, device, rand, buffer):
    args = [
        rand(n, h),
        rand(n, positive=True),
        rand(h, positive=True),
        1e-06,
        0.7,
    ]

    def reference(a):
        x = a[1].unsqueeze(1) * a[0] * a[2]
        result = a[4] * x / (torch.linalg.vector_norm(x, dim=1, keepdim=True) + a[3])
        return result

    case = Case(name, args, (), reference)
    return case


def _guard_case(case, device):
    for index in case.mutated:
        tensors = case.args[index] if isinstance(case.args[index], list) else [case.args[index]]
        for j, tensor in enumerate(tensors):
            storage = torch.randn(tensor.numel() + 128, dtype=tensor.dtype, device=device)
            view = storage[64:-64].view_as(tensor)
            view.copy_(tensor)
            case.guards.append((storage, torch.cat((storage[:64], storage[-64:])).clone()))
            if isinstance(case.args[index], list):
                case.args[index][j] = view
            else:
                case.args[index] = view
    return case


_CASE_BUILDERS = {
    'engram_sinkhorn_momentum_update': _make_engram_sinkhorn_momentum_update_case,
    'engram_sinkhorn_step': _make_engram_sinkhorn_step_case,
    'engram_sinkhorn_finalize': _make_engram_sinkhorn_finalize_case,
}


def make_case(key, n=7, h=256):
    name, _, variant = key.partition(':')
    device = get_device()

    def rand(*shape, bf=False, positive=False):
        x = torch.randn(shape, device=device) * 0.1
        if positive:
            x = x.abs() + 0.5
        return x.bfloat16() if bf else x

    def buffer(*shape, bf=False):
        return rand(*shape, bf=bf)

    case = _CASE_BUILDERS[name](name, variant, n, h, device, rand, buffer)
    return _guard_case(case, device)


def compare(case, actual, expected):
    assert (actual is None) == (expected is None)
    left, right = list(_tensors(actual)), list(_tensors(expected))
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
        assert a.is_contiguous()
        if not a.numel():
            continue
        if case.metric == 'exact':
            assert torch.equal(a, b)
        elif case.metric == 'close':
            torch.testing.assert_close(a, b, atol=case.atol, rtol=case.rtol)
        else:
            assert calc_diff(a.float(), b.float()) < case.tolerance


def profile_call(compiled, args):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with torch.inference_mode():
            actual = compiled(*args)
        torch.npu.synchronize()
    events = {event.key: event.count for event in profile.key_averages()}
    return actual, {name: events.get(name, 0) for name in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')}


def _copy_values(destination, source):
    for old, new in zip(_tensors(destination), _tensors(source), strict=True):
        old.copy_(new)


def _refresh_case_inputs(case, expected, key, step, n, h):
    fresh = make_case(key, n, h)
    indices = range(len(case.args)) if step == 3 else [i for i in range(len(case.args)) if i not in case.mutated]
    for index in indices:
        if step == 1 and isinstance(case.args[index], (torch.Tensor, list)):
            _copy_values(case.args[index], fresh.args[index])
            _copy_values(expected[index], fresh.args[index])
        else:
            case.args[index] = fresh.args[index]
            expected[index] = clone(fresh.args[index])
    if step == 3:
        case.guards = fresh.guards


def _check_state(case, expected, readonly, pointers):
    for index in case.mutated:
        actual, reference = case.args[index], expected[index]
        if index in case.aggregate:
            actual, reference = actual.sum(0), reference.sum(0)
        compare(case, actual, reference)
    assert pointers == [x.data_ptr() for i in case.mutated for x in _tensors(case.args[i])]
    for index, before in readonly.items():
        for actual, reference in zip(_tensors(case.args[index]), _tensors(before), strict=True):
            assert torch.equal(actual, reference)
    for storage, before in case.guards:
        assert torch.equal(torch.cat((storage[:64], storage[-64:])), before)


def check_compile(key, n=7, h=256):
    torch.manual_seed(20261102)
    case = make_case(key, n, h)
    expected = [clone(x) for x in case.args]
    compiled = torch.compile(_PUBLIC_OPERATORS[case.name], backend='npugraph_ex', fullgraph=True, dynamic=False)
    evidence = []
    for step in range(4):
        if step:
            _refresh_case_inputs(case, expected, key, step, n, h)
        readonly = {i: clone(value) for i, value in enumerate(case.args) if i not in case.mutated}
        expected_result = case.reference(expected)
        pointers = [x.data_ptr() for i in case.mutated for x in _tensors(case.args[i])]
        actual, event_counts = profile_call(compiled, case.args)
        compare(case, actual, expected_result)
        _check_state(case, expected, readonly, pointers)
        if case.name == 'engram_sinkhorn_finalize':
            _check_finalize_output(actual, case.args)
        evidence.append(event_counts)
        del actual, expected_result
    assert evidence[0]['acl_graph_capture'] > 0
    assert all(e['acl_graph_replay'] > 0 and e['fx_run_eagerly'] == 0 for e in evidence)
    assert evidence[1]['acl_graph_capture'] == 0
    folder = os.environ.get('ENGRAM_COMPILE_EVIDENCE_DIR')
    if folder:
        Path(folder).mkdir(exist_ok=True, parents=True)
        Path(folder, f'{key.replace(":", "_")}-{n}x{h}.json').write_text(json.dumps(evidence, indent=2))


def _registered_operator(key):
    name = key.partition(':')[0] if key in MOMENTUM_CASES else key
    return _REGISTERED_OPERATORS[name]


def check_opcheck(key):
    case = make_case(key)
    registered = _registered_operator(key)
    args = case.args
    torch.library.opcheck(
        registered,
        tuple(args),
        test_utils=('test_schema', 'test_faketensor'),
    )


def check_once(key, n=7, h=256):
    case = make_case(key, n, h)
    expected = [clone(v) for v in case.args]
    result = case.reference(expected)
    compiled = torch.compile(_PUBLIC_OPERATORS[case.name], backend='npugraph_ex', fullgraph=True, dynamic=False)
    with torch.inference_mode():
        actual = compiled(*case.args)
        compare(case, actual, result)
        for i in case.mutated:
            compare(case, case.args[i], expected[i])
        del actual


def check_invalid_metadata(key, kind):
    case = make_case(key)
    if kind == 'dtype':
        case.args[0] = case.args[0].to(torch.int16)
    elif kind == 'rank':
        case.args[0] = case.args[0].unsqueeze(0)
    else:
        case.args[0] = case.args[0][..., ::2]
    with pytest.raises((TypeError, ValueError)):
        _PUBLIC_OPERATORS[case.name](*case.args)
    args = case.args
    registered = _registered_operator(key)
    with FakeTensorMode() as mode:
        args = [mode.from_tensor(v) if isinstance(v, torch.Tensor) else v for v in args]
        with pytest.raises((TypeError, ValueError)):
            registered(*args)


os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


@pytest.fixture(autouse=True)
def _isolate_compile_cache():
    # Each fixed-shape case owns its Dynamo cache; replay rounds share one object.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


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
    out = engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha)

    time_us = benchmark_timer(lambda: engram_sinkhorn_finalize(matrix, row_scale, col_scale, eps, alpha))

    num_bytes = count_bytes(matrix, row_scale, col_scale, out)
    bandwidth_gbs = num_bytes / time_us / 1e3
    benchmark_record(
        kernel='engram_sinkhorn',
        operation='finalize',
        params=params,
        time_us=time_us,
        bandwidth_gbs=bandwidth_gbs,
    )


@pytest.mark.parametrize('key', SINKHORN_CASES)
def test_engram_sinkhorn_compile(key):
    check_compile(key)


@pytest.mark.parametrize('key', SINKHORN_CASES)
def test_engram_sinkhorn_opcheck(key):
    check_opcheck(key)


@pytest.mark.parametrize('key', SINKHORN_CASES)
@pytest.mark.parametrize('kind', ['dtype', 'rank', 'stride'])
def test_engram_sinkhorn_compile_invalid_metadata(key, kind):
    check_invalid_metadata(key, kind)


@pytest.mark.parametrize('key', SINKHORN_CASES)
def test_engram_sinkhorn_compile_empty(key):
    check_once(key, n=0, h=0 if 'momentum_update' in key else 256)


@pytest.mark.parametrize('key', ['engram_sinkhorn_step', 'engram_sinkhorn_finalize'])
@pytest.mark.parametrize('rows,hidden', [(1, 256), (129, 5120)])
def test_engram_sinkhorn_compile_shapes(key, rows, hidden):
    check_once(key, n=rows, h=hidden)


def test_engram_sinkhorn_compile_composition():
    device = get_device()
    grad = torch.randn(7 * 256, device=device) * 0.1
    momentum = torch.randn_like(grad) * 0.1
    rows = torch.rand(7, device=device) + 0.5
    cols = torch.rand(256, device=device) + 0.5
    args = [grad, momentum, rows, cols]

    def model(grad, momentum, rows, cols):
        values = engram_sinkhorn_momentum_update(grad, momentum, 0.9, True).view(7, 256)
        norms, squares = engram_sinkhorn_step(values, rows, cols, 1e-6)
        result = engram_sinkhorn_finalize(values, rows, cols, 1e-6, 0.7)
        return result + 0.25, norms, squares, rows + 1, momentum + 1

    compiled = torch.compile(model, backend='npugraph_ex', fullgraph=True, dynamic=False)
    expected_args = [clone(v) for v in args]
    for step in range(3):
        if step:
            grad.copy_(torch.randn_like(grad) * 0.1)
            expected_args[0].copy_(grad)
        values = engram_sinkhorn_momentum_update_ref(expected_args[0], expected_args[1], 0.9, True).view(7, 256)
        norms, squares = engram_sinkhorn_step_ref(values, expected_args[2], expected_args[3], 1e-6)
        result = engram_sinkhorn_finalize_ref(values, expected_args[2], expected_args[3], 1e-6, 0.7)
        expected = (result + 0.25, norms, squares, expected_args[2] + 1, expected_args[1] + 1)
        actual, events = profile_call(compiled, args)
        for a, b in zip(actual, expected, strict=True):
            assert calc_diff(a, b) < 1e-10
        assert events['acl_graph_replay'] > 0 and events['fx_run_eagerly'] == 0
        if step:
            assert events['acl_graph_capture'] == 0
        else:
            assert events['acl_graph_capture'] > 0
        del actual


@pytest.mark.parametrize('key', [MOMENTUM_CASES[0], 'engram_sinkhorn_step', 'engram_sinkhorn_finalize'])
@pytest.mark.parametrize('value', ['0', '-1', 'invalid'])
def test_engram_sinkhorn_fake_ignores_core_configuration(key, value, monkeypatch):
    case = make_case(key)
    module = sys.modules['cann_ops_tilelang.engram.engram_sinkhorn']
    op = _registered_operator(key)
    args = case.args

    def unexpected_query(*args, **kwargs):
        raise AssertionError('Fake must not query runtime or hardware')

    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', value)
    monkeypatch.setattr(module, 'get_num_vector_cores', unexpected_query)
    monkeypatch.setattr(module, 'require_ascend', unexpected_query)
    with FakeTensorMode() as mode:
        tensors = [mode.from_tensor(v) if isinstance(v, torch.Tensor) else v for v in args]
        op(*tensors)


@pytest.mark.parametrize('key', [MOMENTUM_CASES[0], 'engram_sinkhorn_step', 'engram_sinkhorn_finalize'])
def test_engram_sinkhorn_empty_ignores_core_configuration(key, monkeypatch):
    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', 'invalid')
    check_once(key, n=0, h=0 if 'momentum_update' in key else 256)


_INVALID_NESTEROV = [0, 1, 2, -1, 1.0, None, 'true']


def _assert_nesterov_rejected(fn, args, kwargs, error_type):
    before = [tensor.clone() for tensor in args[:2]]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with pytest.raises(error_type, match='nesterov must be a bool'), torch.inference_mode():
            fn(*args, **kwargs)
        torch.npu.synchronize()
    events = {event.key: event.count for event in profile.key_averages()}
    for name in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly'):
        assert events.get(name, 0) == 0
    for actual, expected in zip(args[:2], before, strict=True):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize('nesterov', _INVALID_NESTEROV)
@pytest.mark.parametrize('compiled_mode', [False, True])
@pytest.mark.parametrize('as_keyword', [False, True])
@pytest.mark.parametrize('numel', [0, 256])
def test_engram_sinkhorn_nesterov_invalid_scalar(nesterov, compiled_mode, as_keyword, numel):
    grad = torch.randn(numel, device=get_device(), dtype=torch.float32)
    storage = torch.randn(numel + 128, device=grad.device)
    end = 64 + numel
    momentum = storage[64:end]
    original_storage = storage.clone()
    fn = (
        torch.compile(engram_sinkhorn_momentum_update, backend='npugraph_ex', fullgraph=True, dynamic=False)
        if compiled_mode
        else engram_sinkhorn_momentum_update
    )
    args = (grad, momentum, 0.9) if as_keyword else (grad, momentum, 0.9, nesterov)
    kwargs = {'nesterov': nesterov} if as_keyword else {}
    error_type = (TypeError, RuntimeError) if compiled_mode else TypeError
    _assert_nesterov_rejected(fn, args, kwargs, error_type)
    assert torch.equal(storage, original_storage)


@pytest.mark.parametrize('nesterov', [None, False, True], ids=['default', 'false', 'true'])
def test_engram_sinkhorn_nesterov_boolean_compile(nesterov):
    flag = True if nesterov is None else nesterov
    case = make_case(f'engram_sinkhorn_momentum_update:bf16_fp32_{flag}', n=1, h=256)
    expected = clone(case.args)
    compiled = torch.compile(engram_sinkhorn_momentum_update, backend='npugraph_ex', fullgraph=True, dynamic=False)
    evidence = []
    for _ in range(3):
        case.args[0].normal_()
        expected[0].copy_(case.args[0])
        result = case.reference(expected)
        grad_before = case.args[0].clone()
        args = case.args[:3] if nesterov is None else case.args
        actual, events = profile_call(compiled, args)
        compare(case, actual, result)
        compare(case, case.args[1], expected[1])
        assert torch.equal(case.args[0], grad_before)
        assert events['acl_graph_replay'] > 0 and events['fx_run_eagerly'] == 0
        evidence.append(events)
        del actual, result
    assert evidence[0]['acl_graph_capture'] > 0 and evidence[1]['acl_graph_capture'] == 0
    folder = os.environ.get('ENGRAM_COMPILE_EVIDENCE_DIR')
    if folder:
        name = 'default' if nesterov is None else str(nesterov).lower()
        Path(folder, f'nesterov-{name}.json').write_text(json.dumps(evidence, indent=2))


def test_engram_sinkhorn_nesterov_type_guard_after_replay():
    case = make_case('engram_sinkhorn_momentum_update:fp32_fp32_True', n=1, h=256)
    expected = clone(case.args)
    compiled = torch.compile(engram_sinkhorn_momentum_update, backend='npugraph_ex', fullgraph=True, dynamic=False)
    for flag, invalid in ((False, (0,)), (True, (1, 2))):
        case.args[3] = expected[3] = flag
        for _ in range(3):
            reference = case.reference(expected)
            actual, events = profile_call(compiled, case.args)
            compare(case, actual, reference)
            compare(case, case.args[1], expected[1])
            assert events['acl_graph_replay'] > 0 and events['fx_run_eagerly'] == 0
            del actual, reference
        for value in invalid:
            _assert_nesterov_rejected(compiled, tuple(case.args[:3]), {'nesterov': value}, (TypeError, RuntimeError))


_FINALIZE_SHAPES = [(1, 64), (2, 128), (8, 256), (17, 512), (129, 1024), (3, 5120)]


def _check_finalize_output(result, args):
    matrix = args[0]
    assert result.shape == matrix.shape
    assert result.dtype == matrix.dtype and result.device == matrix.device
    assert result.is_contiguous() and result.storage_offset() == 0
    if result.numel():
        assert all(result.untyped_storage().data_ptr() != value.untyped_storage().data_ptr() for value in args[:3])


@pytest.mark.parametrize('rows,hidden', _FINALIZE_SHAPES)
def test_engram_sinkhorn_finalize_allocates_result(rows, hidden):
    case = make_case('engram_sinkhorn_finalize', n=rows, h=hidden)
    before = clone(case.args)
    expected = case.reference(clone(case.args))
    result = engram_sinkhorn_finalize(*case.args)
    second = engram_sinkhorn_finalize(*case.args)
    compare(case, result, expected)
    compare(case, second, expected)
    _check_finalize_output(result, case.args)
    assert result.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    for actual, original in zip(case.args[:3], before[:3], strict=True):
        assert torch.equal(actual, original)


@pytest.mark.parametrize('rows,hidden', _FINALIZE_SHAPES)
def test_engram_sinkhorn_finalize_compile_replay_shapes(rows, hidden):
    check_compile('engram_sinkhorn_finalize', n=rows, h=hidden)
