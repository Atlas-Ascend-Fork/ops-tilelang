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

"""Shared fixed-shape graph checks used by gather tests."""

import logging

import pytest
import torch

from cann_ops_tilelang import config


@pytest.fixture
def reset_compile_cache():
    """Give each fixed-shape test its own compiler cache without affecting other modules."""
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _check(condition, message):
    """Fail a test explicitly, including when Python runs with optimization enabled."""
    if not condition:
        pytest.fail(str(message))


def replace_tensor_inputs(inputs, replacement, *, inplace):
    """Update existing storage on the second call, or return fresh tensors."""
    if not inplace:
        return replacement
    for tensor, updated in zip(inputs, replacement, strict=True):
        tensor.copy_(updated)
    return inputs


def make_zos_sequence_layout(n_seq, prev_n_seq, max_seqlen_q, mode, offset, zero_queries):
    """Build lengths and remapping shared by the two ZOS gather input factories."""
    rows = torch.arange(n_seq, dtype=torch.int32)
    pattern = torch.tensor([0, max_seqlen_q, max(1, max_seqlen_q // 2), 1], dtype=torch.int32)
    seqlens = pattern[(rows % 4).long()].roll(offset)
    if zero_queries:
        seqlens.zero_()
    cumulative = seqlens.cumsum(0, dtype=torch.int32)
    remap = (rows + offset) % prev_n_seq
    if mode == 'new':
        remap.fill_(-1)
    elif mode == 'mixed':
        remap[(rows + offset) % 2 == 0] = -1
    return seqlens, cumulative, remap


def assert_inplace_call(implementation, inputs, expected, **kwargs):
    """Check None return, output storage, values, and all read-only inputs."""
    before = tuple(tensor.clone() for tensor in inputs[:-1])
    output_pointer = inputs[-1].data_ptr()
    _check(implementation(*inputs, **kwargs) is None, 'Check failed: implementation(*inputs, **kwargs) is None')
    _check(inputs[-1].data_ptr() == output_pointer, 'Check failed: inputs[-1].data_ptr() == output_pointer')
    _check(torch.equal(inputs[-1], expected), 'Check failed: torch.equal(inputs[-1], expected)')
    for tensor, original in zip(inputs[:-1], before, strict=True):
        _check(torch.equal(tensor, original), 'read-only input changed')


def profile_npu_call(function, *inputs, **kwargs):
    """Run one call and return its result and ACL Graph CPU profiler events."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = function(*inputs, **kwargs)
        torch.npu.synchronize()
    counts = {event.key: event.count for event in profile.key_averages()}
    events = {
        'acl_graph_capture': counts.get('acl_graph_capture', 0),
        'acl_graph_replay': counts.get('acl_graph_replay', 0),
        'fx_run_eagerly': counts.get('fx_run_eagerly', 0),
    }
    return result, events


def profile_inplace_replay(compiled, inputs, expected):
    """Profile repeated ZOS writes and a following read of the output buffer."""
    events_by_call = []
    for _ in range(3):
        inputs[-1].fill_(-99)
        actual, events = profile_npu_call(compiled, *inputs)
        events_by_call.append(events)
        _check(torch.equal(inputs[-1], expected), 'Check failed: torch.equal(inputs[-1], expected)')
        _check(torch.equal(actual, expected + 1), 'Check failed: torch.equal(actual, expected + 1)')
        del actual
    return events_by_call


def assert_acl_graph_replay(events_by_call):
    """Require capture, later replay, and no explicit eager fallback."""
    _check(any(events['acl_graph_capture'] for events in events_by_call), events_by_call)
    _check(any(events['acl_graph_replay'] for events in events_by_call[1:]), events_by_call)
    _check(not any(events['fx_run_eagerly'] for events in events_by_call), events_by_call)


def check_core_config(op, inputs, expected, monkeypatch, caplog, *, mutates_output=False, **kwargs):
    """Check one compiled gather call across core configuration changes."""
    caplog.set_level(logging.INFO, logger=__name__)
    env_name = 'OPS_TILELANG_NUM_VECTOR_CORES'
    monkeypatch.delenv(env_name, raising=False)
    config.clear_runtime_caches()
    default_cores = config.get_num_vector_cores()
    output = inputs[-1] if mutates_output else None
    initial_output = output.clone() if output is not None else None
    read_only = inputs[:-1] if mutates_output else inputs
    originals = tuple(tensor.clone() for tensor in read_only)
    compiled = torch.compile(op, backend='npugraph_ex', fullgraph=True, dynamic=False)

    def check_call(label, expected_cores, *, capture):
        _check(config.get_num_vector_cores() == expected_cores, label)
        if output is not None:
            output.copy_(initial_output)
        result, events = profile_npu_call(compiled, *inputs, **kwargs)
        logging.getLogger(__name__).info('%s: ACL Graph events=%s', label, events)
        _check(not events['fx_run_eagerly'], label)
        _check(bool(events['acl_graph_capture']) == capture, label)
        _check(bool(events['acl_graph_replay']), label)
        if output is not None:
            _check(result is None, label)
            result = output
        _check(torch.equal(result, expected), label)

    try:
        check_call('default', default_cores, capture=True)
        check_call('default replay', default_cores, capture=False)
        config.set_num_ai_cores(1)
        check_call('AI core override', 2, capture=True)
        config.set_num_ai_cores(1)
        check_call('same AI core override', 2, capture=False)
        with pytest.raises(ValueError, match='num_ai_cores'):
            config.set_num_ai_cores(0)
        check_call('invalid AI core override', 2, capture=False)
        config.reset_runtime_config()
        check_call('AI core override removed', default_cores, capture=True)

        with monkeypatch.context() as env:
            for label, value, capture in (
                ('environment override added', '1', True),
                ('environment override changed', '2', True),
                ('same environment override', '2', False),
            ):
                env.setenv(env_name, value)
                config.clear_runtime_caches()
                check_call(label, int(value), capture=capture)
        config.clear_runtime_caches()
        check_call('environment override removed', default_cores, capture=True)
        for tensor, original in zip(read_only, originals, strict=True):
            _check(torch.equal(tensor, original), 'read-only input changed')
    finally:
        config.reset_runtime_config()
        config.clear_runtime_caches()
