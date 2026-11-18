# MIT License
#
# Copyright (c) 2026 ops-tilelang contributors
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

"""Head mix graph contracts and strict scalar validation through direct registration."""

import pytest
import torch

import cann_ops_tilelang as ops
from tests.mhc import test_mhc_graph_utils as graph_utils

pytestmark = pytest.mark.npu
NAMES = ('mhc_head_compute_mix_forward', 'mhc_head_compute_mix_backward')


isolate_graphs = graph_utils.isolate_graphs


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('n', [7, 513])
def test_head_compile_replay(name, n):
    graph_utils.check_replay(name, n=n)


@pytest.mark.parametrize('name', NAMES)
def test_head_compile_opcheck(name):
    case = graph_utils.make_case(name)
    op, args, kwargs = graph_utils.registered_case(case)
    torch.library.opcheck(op, args, kwargs=kwargs, test_utils=('test_schema', 'test_faketensor'))


@pytest.mark.parametrize('name', NAMES)
def test_head_compile_empty(name):
    case = graph_utils.make_case(name, n=0)
    graph_utils.check_round(case, graph_utils.compile_case(case))


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('compiled', [False, True])
def test_head_compile_invalid_dtype(name, compiled):
    case = graph_utils.make_case(name)
    case.args[0] = case.args[0].bfloat16()
    before = graph_utils.clone_args(case.args)
    fn = graph_utils.compile_case(case) if compiled else graph_utils.public_op(name)
    with pytest.raises((TypeError, RuntimeError), match='float32'):
        fn(*case.args)
    for a, b in zip(case.args, before, strict=True):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('field,value', [('tile', True), ('tile', 1.5), ('tile', 3), ('tile', 128), ('scalar', True)])
def test_head_compile_raw_scalar(name, compiled, field, value):
    case = graph_utils.make_case(name)
    index = -1 if field == 'tile' else -2
    case.args[index] = value
    before = graph_utils.clone_args(case.args)
    fn = graph_utils.compile_case(case) if compiled else graph_utils.public_op(name)
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
        pytest.raises((TypeError, ValueError, RuntimeError)),
    ):
        fn(*case.args)
    counts = {e.key: e.count for e in profile.key_averages()}
    assert all(counts.get(key, 0) == 0 for key in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly'))
    for a, b in zip(case.args, before, strict=True):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)


@pytest.mark.parametrize('eps,tile', [(0.0, 1), (0.02, 8), (0.0001, 64)])
def test_head_compile_forward_parameters(eps, tile):
    case = graph_utils.make_case(NAMES[0])
    case.args[-2:] = [eps, tile]
    call = graph_utils.compile_case(case)
    records = [graph_utils.check_round(case, call) for _ in range(3)]
    assert records[0]['acl_graph_capture'] > 0
    assert all(e['acl_graph_replay'] > 0 for e in records)
    assert records[1]['acl_graph_capture'] == 0
    graph_utils.save_evidence(f'head-parameters-{eps}-{tile}', records)


def test_head_compile_forward_backward_composition():
    case = graph_utils.make_case(NAMES[1])
    grad, x, scale, base, xgrad, sgrad, bgrad, cores, tile = case.args
    out = torch.empty_like(x)

    def model(grad, x, scale, base, out, xgrad, sgrad, bgrad):
        ops.mhc_head_compute_mix_forward(x, scale, base, out, 1e-06, tile)
        ops.mhc_head_compute_mix_backward(grad, x, scale, base, xgrad, sgrad, bgrad, cores, tile)
        return out + 0.25

    compiled = torch.compile(model, backend='npugraph_ex', fullgraph=True, dynamic=False)
    records = []
    for _ in range(3):
        x.copy_(torch.randn_like(x) * 0.1)
        grad.copy_(torch.randn_like(grad) * 0.1)
        expected = graph_utils.clone_args(case.args)
        case.reference(expected)
        actual, event = graph_utils.inspect_call(compiled, grad, x, scale, base, out, xgrad, sgrad, bgrad)
        torch.testing.assert_close(actual, (x * scale + base).sigmoid() + 1e-06 + 0.25, atol=1e-05, rtol=0.0001)
        torch.testing.assert_close(xgrad, expected[4], atol=1e-05, rtol=0.0001)
        torch.testing.assert_close(sgrad.sum(0), expected[5].sum(0), atol=1e-05, rtol=0.0001)
        torch.testing.assert_close(bgrad.sum(0), expected[6].sum(0), atol=1e-05, rtol=0.0001)
        records.append(event)
        del actual
    assert records[0]['acl_graph_capture'] > 0
    assert all(e['acl_graph_replay'] > 0 and e['fx_run_eagerly'] == 0 for e in records)
    assert records[1]['acl_graph_capture'] == 0
    graph_utils.save_evidence('head-composition', records)


@pytest.mark.parametrize('name', NAMES)
def test_head_compile_scalar_guard_after_replay(name):
    case = graph_utils.make_case(name)
    if name.endswith('forward'):
        case.args[-1] = 1
    call = graph_utils.compile_case(case)
    records = [graph_utils.check_round(case, call) for _ in range(3)]
    assert records[0]['acl_graph_capture'] > 0
    assert records[1]['acl_graph_capture'] == 0
    assert all(e['acl_graph_replay'] > 0 for e in records)
    index, value = (-1, True) if name.endswith('forward') else (-2, 4.0)
    case.args[index] = value
    before = graph_utils.clone_args(case.args)
    with (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
        pytest.raises((TypeError, RuntimeError)),
        torch.inference_mode(),
    ):
        call(*case.args)
    counts = {e.key: e.count for e in profile.key_averages()}
    assert all(counts.get(key, 0) == 0 for key in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly'))
    for a, b in zip(case.args, before, strict=True):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('compiled', [False, True])
def test_head_direct_scalar_types_after_replay(name, compiled):
    case = graph_utils.make_case(name)
    fn = graph_utils.compile_case(case) if compiled else graph_utils.public_op(name)
    graph_utils.check_round(case, fn)
    invalid = [(-1, value) for value in (None, '0', True, 1.5, 3, 128)]
    if name.endswith('forward'):
        invalid += [(-2, value) for value in (None, '1e-6', True, -1, float('inf'), float('nan'))]
    else:
        invalid += [(-2, value) for value in (None, '4', True, 4.0, 0, -1)]
    for index, value in invalid:
        args = list(case.args)
        args[index] = value
        before = graph_utils.clone_args(case.args)
        error = (TypeError, ValueError, RuntimeError) if compiled else (TypeError, ValueError)
        with (
            torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile,
            pytest.raises(error),
            torch.inference_mode(),
        ):
            fn(*args)
        counts = {e.key: e.count for e in profile.key_averages()}
        assert all(counts.get(key, 0) == 0 for key in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly'))
        for actual, expected in zip(case.args, before, strict=True):
            if isinstance(actual, torch.Tensor):
                assert torch.equal(actual, expected)
