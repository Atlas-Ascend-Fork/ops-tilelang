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

"""Expand forward/backward graph contracts, specialization and compositions."""

import pytest
import torch

import cann_ops_tilelang as ops
from tests.mhc import test_mhc_graph_utils as graph_utils

pytestmark = pytest.mark.npu
NAMES = ('expand_to_mhc_forward', 'expand_to_mhc_backward')


isolate_graphs = graph_utils.isolate_graphs


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('n,h,streams', [(7, 256, 4), (513, 4096, 2), (8, 7168, 8)])
def test_expand_compile_replay(name, n, h, streams):
    graph_utils.check_replay(name, n=n, h=h, streams=streams)


@pytest.mark.parametrize('name', NAMES)
def test_expand_compile_opcheck(name):
    case = graph_utils.make_case(name)
    op, args, kwargs = graph_utils.registered_case(case)
    torch.library.opcheck(op, args, kwargs=kwargs, test_utils=('test_schema', 'test_faketensor'))


@pytest.mark.parametrize('name', NAMES)
def test_expand_compile_empty(name):
    case = graph_utils.make_case(name, n=0)
    events = graph_utils.check_round(case, graph_utils.compile_case(case))
    assert events['fx_run_eagerly'] == 0


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('compiled', [False, True])
def test_expand_compile_invalid_dtype(name, compiled):
    case = graph_utils.make_case(name)
    case.args[0] = case.args[0].float()
    before = graph_utils.clone_args(case.args)
    fn = graph_utils.compile_case(case) if compiled else graph_utils.public_op(name)
    with pytest.raises((TypeError, RuntimeError), match='bfloat16'):
        fn(*case.args)
    for a, b in zip(case.args, before, strict=True):
        assert torch.equal(a, b)


def test_expand_compile_forward_backward_composition():
    case = graph_utils.make_case(NAMES[0])
    x, out = case.args
    grad = torch.randn_like(out)
    xgrad = torch.empty_like(x)

    def model(x, out, grad, xgrad):
        ops.expand_to_mhc_forward(x, out)
        ops.expand_to_mhc_backward(grad, xgrad)
        return out.float().sum(1) + xgrad.float()

    compiled = torch.compile(model, backend='npugraph_ex', fullgraph=True, dynamic=False)
    records = []
    for _ in range(3):
        x.copy_(torch.randn_like(x))
        grad.copy_(torch.randn_like(grad))
        expected = x.float() * 4 + grad.float().sum(1).bfloat16().float()
        actual, event = graph_utils.inspect_call(compiled, x, out, grad, xgrad)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        records.append(event)
        del actual
    assert records[0]['acl_graph_capture'] > 0
    assert all(e['acl_graph_replay'] > 0 and e['fx_run_eagerly'] == 0 for e in records)
    assert records[1]['acl_graph_capture'] == 0
    graph_utils.save_evidence('expand-composition', records)
