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

"""Graph-test helpers limited to MHC expand and head mix."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch

from cann_ops_tilelang import (
    expand_to_mhc_backward,
    expand_to_mhc_forward,
    mhc_head_compute_mix_backward,
    mhc_head_compute_mix_forward,
)
from cann_ops_tilelang.config import get_device

PUBLIC_OPS = {
    'expand_to_mhc_forward': expand_to_mhc_forward,
    'expand_to_mhc_backward': expand_to_mhc_backward,
    'mhc_head_compute_mix_forward': mhc_head_compute_mix_forward,
    'mhc_head_compute_mix_backward': mhc_head_compute_mix_backward,
}


def public_op(name):
    result = PUBLIC_OPS.get(name)
    if result is None:
        raise ValueError(f'Unsupported MHC graph case: {name}.')
    return result


@dataclass
class GraphCase:
    name: str
    args: list
    mutated: tuple[int, ...]
    reference: object
    exact: bool = False
    aggregate: tuple[int, ...] = ()
    guards: list = field(default_factory=list)


def clone_args(args):
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in args]


def make_case(name, n=7, h=256, streams=4, cores=4):
    device = get_device()

    def rand(*shape, bf=False):
        value = torch.randn(shape, device=device) * 0.1
        return value.bfloat16() if bf else value

    if name == 'expand_to_mhc_forward':
        args = [rand(n, h, bf=True), rand(n, streams, h, bf=True)]

        def reference(a):
            a[1].copy_(a[0].unsqueeze(1).expand(-1, streams, -1))

        case = GraphCase(name, args, (1,), reference, exact=True)
    elif name == 'expand_to_mhc_backward':
        args = [rand(n, streams, h, bf=True), rand(n, h, bf=True)]

        def reference(a):
            a[1].copy_(a[0].float().sum(1).bfloat16())

        case = GraphCase(name, args, (1,), reference, exact=True)
    elif name == 'mhc_head_compute_mix_forward':
        args = [rand(n, 4), rand(1), rand(4), rand(n, 4), 1e-6, 0]

        def reference(a):
            a[3].copy_((a[0] * a[1] + a[2]).sigmoid() + a[4])

        case = GraphCase(name, args, (3,), reference)
    elif name == 'mhc_head_compute_mix_backward':
        args = [rand(n, 4), rand(n, 4), rand(1), rand(4), rand(n, 4), rand(cores, 1), rand(cores, 4), cores, 0]

        def reference(a):
            # Autograd stays outside inference mode and does not use the adapter.
            x, scale, base = [v.detach().clone().requires_grad_(True) for v in a[1:4]]
            grads = torch.autograd.grad((x * scale + base).sigmoid(), (x, scale, base), a[0])
            a[4].copy_(grads[0])
            a[5].zero_()
            a[6].zero_()
            a[5][0].copy_(grads[1])
            a[6][0].copy_(grads[2])

        case = GraphCase(name, args, (4, 5, 6), reference, aggregate=(5, 6))
    else:
        raise ValueError(f'Unsupported MHC graph case: {name}.')
    for index in case.mutated:
        tensor = case.args[index]
        storage = torch.randn(tensor.numel() + 128, dtype=tensor.dtype, device=device)
        view = storage[64:-64].view_as(tensor)
        view.copy_(tensor)
        case.args[index] = view
        case.guards.append((storage, torch.cat((storage[:64], storage[-64:])).clone()))
    return case


def assert_outputs(case, actual, expected):
    assert actual is None
    for index, (value, ref) in enumerate(zip(case.args, expected, strict=True)):
        if not isinstance(value, torch.Tensor):
            continue
        if index not in case.mutated:
            assert torch.equal(value, ref)
            continue
        assert value.shape == ref.shape and value.dtype == ref.dtype and value.device == ref.device
        if index in case.aggregate:
            value, ref = value.sum(0), ref.sum(0)
        if case.exact:
            assert torch.equal(value, ref)
        else:
            torch.testing.assert_close(value, ref, atol=1e-5, rtol=1e-4)
    assert_guards(case)


def assert_guards(case):
    for storage, before in case.guards:
        assert torch.equal(torch.cat((storage[:64], storage[-64:])), before)


def inspect_call(fn, *args, **kwargs):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile, torch.inference_mode():
        actual = fn(*args, **kwargs)
        torch.npu.synchronize()
    counts = {e.key: e.count for e in profile.key_averages()}
    return actual, {name: counts.get(name, 0) for name in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')}


def save_evidence(name, records):
    folder = os.environ.get('MHC_COMPILE_EVIDENCE_DIR')
    if folder:
        path = Path(folder)
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + '.json')).write_text(json.dumps(records, indent=2))


def check_round(case, call, *, changes=None, kwargs=None):
    if changes:
        for index, value in changes.items():
            case.args[index] = value
    expected = clone_args(case.args)
    case.reference(expected)
    actual, events = inspect_call(call, *case.args, **(kwargs or {}))
    assert_outputs(case, actual, expected)
    assert events['fx_run_eagerly'] == 0
    return events


def compile_case(case):
    return torch.compile(public_op(case.name), backend='npugraph_ex', fullgraph=True, dynamic=False)


def registered_case(case):
    assert callable(public_op(case.name))  # Load the public lazy operator before inspecting its registry.
    op = getattr(torch.ops.cann_ops_tilelang, case.name).default
    return op, tuple(case.args), {}


def check_replay(name, *, n=7, h=256, streams=4):
    case = make_case(name, n=n, h=h, streams=streams)
    compiled = compile_case(case)
    events = [check_round(case, compiled)]
    # Same address, new data; then readonly buffers and finally all buffers change address.
    for step in (1, 2, 3):
        fresh = make_case(name, n=n, h=h, streams=streams)
        for index, value in enumerate(fresh.args):
            if index in case.mutated and step != 3:
                continue
            if isinstance(value, torch.Tensor) and step == 1:
                case.args[index].copy_(value)
            else:
                case.args[index] = value
        if step == 3:
            case.guards = fresh.guards
        events.append(check_round(case, compiled))
    assert events[0]['acl_graph_capture'] > 0
    assert all(e['acl_graph_replay'] > 0 for e in events)
    assert events[1]['acl_graph_capture'] == 0
    save_evidence(f'{name}-{n}-{h}-{streams}', events)


def reset_graph_test_state():
    torch.manual_seed(20261105)
    torch.compiler.reset()


@pytest.fixture(autouse=True)
def isolate_graphs():
    reset_graph_test_state()
    yield
    torch.compiler.reset()
