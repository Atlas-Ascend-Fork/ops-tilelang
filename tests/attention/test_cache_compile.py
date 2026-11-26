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

"""Independent state, metadata and ACL Graph checks for paged cache APIs."""

import logging
import re

import pytest
import torch

from cann_ops_tilelang import config
from cann_ops_tilelang.attention import cache as cache_api
from tests.quant.cast_ref import cast as torch_cast

pytestmark = pytest.mark.npu
_LOGGER = logging.getLogger(__name__)
_MODES = ('kv64', 'kv32', 'fp8', 'mxfp8', 'fp4', 'llama')


@pytest.fixture(autouse=True)
def _isolate_compiler_between_tests():
    # CustomOpDef callables share Dynamo's external_utils.inner code object.
    # Keep independent test specializations from exhausting the shared cache.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _case(mode, n=17, offset=0):
    llama = mode == 'llama'
    index = mode in ('fp8', 'mxfp8', 'fp4')
    heads, hidden = (2, 128) if llama else (1, 128 if index else 512)
    values = torch.arange(n * heads * hidden, device='cpu').float().sin() * (3 + offset)
    values = values.to(torch.bfloat16).view(n, heads, hidden).to('npu')
    slots = ((torch.arange(n, device='cpu') * 3 + offset) % 64).to(torch.int32)
    slots[1::7] = -1
    slots = slots.to('npu')
    if llama:
        storage = torch.full((2, 32 * heads * hidden + 32), -77, dtype=torch.bfloat16, device='npu')
        cache = storage[:, : 32 * heads * hidden].view(2, 32, heads, hidden)
        return values, storage, cache, slots
    size = {'kv64': 584, 'kv32': 528, 'fp8': 132, 'mxfp8': 132, 'fp4': 68}[mode]
    stride = (32 * size + 575) // 576 * 576 + 576
    storage = torch.full((2, stride), 165, dtype=torch.uint8, device='npu')
    return values, storage, storage[:, : 32 * size].view(2, 32, 1, size), slots


def _operations(mode):
    if mode in ('kv64', 'kv32'):
        return (
            cache_api.store_kv_cache,
            cache_api.load_kv_cache,
            {
                'use_fp8_rope': mode == 'kv32',
                'num_per_channels': 32 if mode == 'kv32' else 64,
            },
        )
    if mode == 'llama':
        return cache_api.store_kv_cache_llama, cache_api.load_kv_cache_llama, {}
    return (
        cache_api.store_index_kv_cache,
        cache_api.load_index_kv_cache,
        {
            'is_fp4': mode == 'fp4',
            'is_mxfp8': mode == 'mxfp8',
        },
    )


def _store_reference(mode, values, storage, slots):
    """Quantize independently, then scatter raw bytes into the documented layout."""
    expected = storage.cpu().clone()
    values = values.cpu()
    if mode == 'llama':
        view = expected[:, : 32 * 256].view(2, 32, 2, 128)
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot >= 0:
                view[slot // 32, slot % 32] = values[row]
        return expected
    n = values.shape[0]
    width = values.shape[-1]
    group = {'kv64': 64, 'kv32': 32, 'fp8': 128, 'mxfp8': 32, 'fp4': 32}[mode]
    data, scale = torch_cast(
        values[:, 0],
        'e2m1' if mode == 'fp4' else 'e4m3',
        block_size=(1, group),
        round_sf=True,
        use_packed_ue8m0=mode != 'fp8',
    )
    data = data.contiguous().view(torch.uint8).view(n, width // 2 if mode == 'fp4' else width)
    scale = scale.contiguous().view(torch.uint8).view(n, -1)
    if mode == 'kv64':
        data = torch.cat((data[:, :448], values[:, 0, 448:].contiguous().view(torch.uint8)), dim=1)
    for row, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            block, position = divmod(slot, 32)
            begin = position * data.shape[1]
            scale_begin = 32 * data.shape[1] + position * scale.shape[1]
            expected[block, slice(begin, begin + data.shape[1])] = data[row]
            expected[block, slice(scale_begin, scale_begin + scale.shape[1])] = scale[row]
    return expected


def _load_reference(mode, storage, slots):
    """Decode CPU cache bytes without invoking another cache or TileLang API."""
    storage = storage.cpu()
    indices = slots.cpu().tolist()
    if mode == 'llama':
        output = torch.zeros((len(indices), 2, 128), dtype=torch.bfloat16, device='cpu')
        cache = storage[:, : 32 * 256].view(2, 32, 2, 128)
        for row, slot in enumerate(indices):
            if slot >= 0:
                output[row] = cache[slot // 32, slot % 32]
        return output
    width = {'kv64': 576, 'kv32': 512, 'fp8': 128, 'mxfp8': 128, 'fp4': 64}[mode]
    scale_width = {'kv64': 8, 'kv32': 16}.get(mode, 4)
    data = torch.zeros((len(indices), width), dtype=torch.uint8, device='cpu')
    scale = torch.zeros((len(indices), scale_width), dtype=torch.uint8, device='cpu')
    for row, slot in enumerate(indices):
        if slot >= 0:
            block, position = divmod(slot, 32)
            data[row] = storage[block, slice(position * width, (position + 1) * width)]
            start = 32 * width + position * scale_width
            scale[row] = storage[block, slice(start, start + scale_width)]
    if mode in ('fp8', 'mxfp8', 'fp4'):
        return data, scale
    group = 64 if mode == 'kv64' else 32
    quant_width = 448 if mode == 'kv64' else 512
    quant = data[:, :quant_width].contiguous().view(torch.float8_e4m3fn).float()
    factors = (scale.to(torch.int32) << 23).view(torch.float32)
    expanded = factors.repeat_interleave(group, dim=1)[:, :quant_width]
    output = (quant * expanded).to(torch.bfloat16)
    if mode == 'kv64':
        rope = data[:, 448:].contiguous().view(torch.bfloat16)
        output = torch.cat((output, rope), dim=1)
    return output.view(len(indices), 1, 512)


def _check_loaded(mode, output, expected):
    if isinstance(output, tuple):
        for actual, reference in zip(output, expected, strict=True):
            assert torch.equal(actual.contiguous().view(torch.uint8).cpu().view(reference.shape), reference)
        assert output[0].dtype == (torch.int8 if mode == 'fp4' else torch.float8_e4m3fn)
        assert output[1].dtype == (torch.float32 if mode == 'fp8' else torch.int32)
    else:
        assert output.dtype == torch.bfloat16
        assert output.shape == expected.shape
        assert torch.equal(output.cpu(), expected)
    for actual in output if isinstance(output, tuple) else (output,):
        assert actual.device.type == 'npu'
        assert actual.is_contiguous() and actual.storage_offset() == 0


def _profile_call(compiled, *args, **kwargs):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        output = compiled(*args, **kwargs)
        torch.npu.synchronize()
    counts = {event.key: event.count for event in profile.key_averages()}
    events = {key: counts.get(key, 0) for key in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')}
    assert events['fx_run_eagerly'] == 0
    return output, events


def _compile(op):
    return torch.compile(op, backend='npugraph_ex', fullgraph=True, dynamic=False)


def _load_args(mode, cache, slots):
    return (cache, 2, slots) if mode == 'llama' else (cache, slots)


@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize('operation', ('store', 'load'))
@torch.inference_mode()
def test_cache_compile(mode, operation):
    store, load, options = _operations(mode)
    compiled = _compile(store if operation == 'store' else load)
    values, storage, cache, slots = _case(mode)
    observations = []
    for step in range(3):
        if step == 1:
            next_values, _, _, next_slots = _case(mode, offset=step)
            values.copy_(next_values)
            slots.copy_(next_slots)
        elif step == 2:
            values, storage, cache, slots = _case(mode, offset=step)
        before_values, before_slots = values.clone(), slots.clone()
        expected_cache = _store_reference(mode, values, storage, slots)
        if operation == 'store':
            output, events = _profile_call(compiled, values, cache, slots, **options)
            assert output is None
            assert torch.equal(storage.cpu(), expected_cache)
        else:
            storage.copy_(expected_cache.to(storage.device))
            before_cache = storage.clone()
            expected = _load_reference(mode, expected_cache, slots)
            output, events = _profile_call(compiled, *_load_args(mode, cache, slots), **options)
            _check_loaded(mode, output, expected)
            assert torch.equal(storage, before_cache)
        assert torch.equal(values, before_values) and torch.equal(slots, before_slots)
        observations.append(events)
        del output
    assert any(item['acl_graph_capture'] for item in observations)
    assert observations[1]['acl_graph_replay'] > 0
    assert observations[1]['acl_graph_capture'] == 0
    _LOGGER.info('%s %s graph events: %s', mode, operation, observations)


@pytest.mark.parametrize('mode', ('kv64', 'kv32'))
@torch.inference_mode()
def test_cache_compile_model(mode):
    store, load, options = _operations(mode)
    values, storage, cache, slots = _case(mode)

    def model(values, cache, slots):
        store(values, cache, slots, **options)
        returned = load(cache, slots, **options)
        return returned, returned.float() + 1

    compiled = _compile(model)
    for _step in range(3):
        values.add_(0.5)
        expected_cache = _store_reference(mode, values, storage, slots)
        expected = _load_reference(mode, expected_cache, slots)
        returned, following = compiled(values, cache, slots)
        _check_loaded(mode, returned, expected)
        assert returned.untyped_storage().data_ptr() != storage.untyped_storage().data_ptr()
        assert torch.equal(storage.cpu(), expected_cache)
        assert torch.equal(following.cpu(), expected.float() + 1)
        del returned, following


@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize('empty', (False, True))
@torch.inference_mode()
def test_cache_registration_and_empty(mode, empty):
    values, storage, cache, slots = _case(mode, n=0 if empty else 17)
    store, load, options = _operations(mode)
    checks = ('test_schema', 'test_faketensor')
    torch.library.opcheck(store, (values, cache, slots), kwargs=options, test_utils=checks)
    torch.library.opcheck(load, _load_args(mode, cache, slots), kwargs=options, test_utils=checks)
    if empty:
        torch.library.opcheck(store, (values, None, slots), kwargs=options, test_utils=checks)
        torch.library.opcheck(load, _load_args(mode, None, slots), kwargs=options, test_utils=checks)
        before = storage.clone()
        assert _compile(store)(values, cache, slots, **options) is None
        output = _compile(load)(*_load_args(mode, cache, slots), **options)
        for tensor in output if isinstance(output, tuple) else (output,):
            assert tensor.shape[0] == 0
        assert torch.equal(storage, before)


@pytest.mark.parametrize('mode', _MODES)
@torch.inference_mode()
def test_cache_none_compile(mode):
    values, _, _, slots = _case(mode)
    store, load, options = _operations(mode)
    assert _compile(store)(values, None, slots, **options) is None
    torch.library.opcheck(
        load, _load_args(mode, None, slots), kwargs=options, test_utils=('test_schema', 'test_faketensor')
    )
    compiled = _compile(load)
    previous = None
    observations = []
    for _ in range(3):
        output, events = _profile_call(compiled, *_load_args(mode, None, slots), **options)
        observations.append(events)
        data = output[0] if isinstance(output, tuple) else output
        snapshot = data.view(torch.uint8).clone()
        assert snapshot.numel() > 0
        if previous is not None:
            assert not torch.equal(snapshot, previous), 'random fallback repeated old graph output'
        previous = snapshot
        del data, output
    assert any(item['acl_graph_capture'] for item in observations)
    assert any(item['acl_graph_replay'] for item in observations[1:])
    _LOGGER.info('%s None-cache graph events: %s', mode, observations)


def test_cache_invalid_metadata():
    values, _, cache, slots = _case('kv64')
    with pytest.raises(ValueError, match='Supported'):
        cache_api.load_kv_cache(cache, slots, use_fp8_rope=True, num_per_channels=64)
    with pytest.raises(ValueError, match='4D'):
        cache_api.load_kv_cache(cache[0], slots)
    with pytest.raises(ValueError, match='int32'):
        cache_api.store_kv_cache(values, cache, slots.long())


@pytest.mark.parametrize('mode', ('fp8', 'mxfp8', 'fp4', 'fp4-both'))
@torch.inference_mode()
def test_index_cache_compile_model(mode):
    reference_mode = 'fp4' if mode == 'fp4-both' else mode
    values, storage, cache, slots = _case(reference_mode)
    store, load, options = _operations(reference_mode)
    if mode == 'fp4-both':
        options['is_mxfp8'] = True

    def model(values, cache, slots):
        store(values, cache, slots, **options)
        data, scale = load(cache, slots, **options)
        return data.view(torch.uint8).clone(), scale.view(torch.uint8).clone()

    expected_cache = _store_reference(reference_mode, values, storage, slots)
    expected = _load_reference(reference_mode, expected_cache, slots)
    output = _compile(model)(values, cache, slots)
    for actual, reference in zip(output, expected, strict=True):
        assert torch.equal(actual.cpu().view(reference.shape), reference)
    assert torch.equal(storage.cpu(), expected_cache)


def _record_cache_factory(factory_name, factory, observed):
    def record(*args, **kwargs):
        kernel = factory(*args, **kwargs)
        extents = re.findall(r'T\.thread_binding\((\d+), thread="blockIdx\.x"\)', kernel.prim_func.script())
        assert extents == [str(config.get_num_vector_cores())], kernel.prim_func.script()
        signature = (factory_name, args, tuple(sorted(kwargs.items())))
        observed.append((signature, kernel))
        return kernel

    return record


def _record_cache_kernels(monkeypatch):
    observed = []
    for name, factory in tuple(vars(cache_api).items()):
        if name.endswith('_tl_ascend'):
            monkeypatch.setattr(cache_api, name, _record_cache_factory(name, factory, observed))
    return observed


def _change_core_configuration(phase, environment):
    if phase in ('reduced', 'same'):
        config.set_num_ai_cores(1)
    elif phase == 'reset':
        config.reset_runtime_config()
    elif phase in ('cap', 'changed_cap'):
        cap = min(4 if phase == 'cap' else 2, 2 * config.get_num_ai_cores())
        environment.setenv('OPS_TILELANG_NUM_VECTOR_CORES', str(cap))
        config.clear_runtime_caches()
    elif phase == 'unset':
        environment.delenv('OPS_TILELANG_NUM_VECTOR_CORES')
        config.clear_runtime_caches()
    elif phase == 'invalid_ai':
        with pytest.raises(ValueError):
            config.set_num_ai_cores(0)


def _check_core_configuration_call(mode, operation, call, inputs, options):
    values, storage, cache, slots = inputs
    before_values, before_slots = values.clone(), slots.clone()
    expected_cache = _store_reference(mode, values, storage, slots)
    if operation == 'store':
        output, events = _profile_call(call, values, cache, slots, **options)
        assert output is None
        assert torch.equal(storage.cpu(), expected_cache)
    else:
        storage.copy_(expected_cache.to(storage.device))
        output, events = _profile_call(call, *_load_args(mode, cache, slots), **options)
        _check_loaded(mode, output, _load_reference(mode, expected_cache, slots))
        assert torch.equal(storage.cpu(), expected_cache)
    assert torch.equal(values, before_values) and torch.equal(slots, before_slots)
    del output
    return events


@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize('operation', ('store', 'load'))
@torch.inference_mode()
def test_cache_compile_core_configuration(mode, operation, monkeypatch):
    try:
        with monkeypatch.context() as environment:
            _run_core_configuration_cases(mode, operation, environment)
    finally:
        config.reset_runtime_config()


def _run_core_configuration_cases(mode, operation, environment):
    environment.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
    config.reset_runtime_config()
    kernels = _record_cache_kernels(environment)
    store, load, options = _operations(mode)
    op = store if operation == 'store' else load
    compiled = _compile(op)
    inputs = _case(mode)
    initial_kernels = {}
    observations = []
    for phase in ('default', 'reduced', 'same', 'reset', 'cap', 'changed_cap', 'unset', 'invalid_ai'):
        _change_core_configuration(phase, environment)
        inputs[0].add_(0.125)
        kernels.clear()
        _check_core_configuration_call(mode, operation, op, inputs, options)
        first = _check_core_configuration_call(mode, operation, compiled, inputs, options)
        second = _check_core_configuration_call(mode, operation, compiled, inputs, options)
        assert kernels, 'the public API did not request a real cache kernel'
        if phase == 'default':
            initial_kernels.update(kernels)
        if phase in ('reset', 'unset'):
            assert all(kernel is initial_kernels.get(key) for key, kernel in kernels), phase
        assert first['acl_graph_replay'] > 0
        assert bool(first['acl_graph_capture']) == (phase not in ('same', 'invalid_ai'))
        assert second['acl_graph_capture'] == 0 and second['acl_graph_replay'] > 0
        observations.append((phase, config.get_num_vector_cores(), first, second))
    _check_invalid_vector_caps(mode, operation, op, compiled, inputs, options, environment)
    _LOGGER.info('%s %s core configuration events: %s', mode, operation, observations)


def _check_invalid_vector_caps(mode, operation, op, compiled, inputs, options, environment):
    maximum = 2 * config.get_num_ai_cores()
    for cap in ('0', 'invalid', str(maximum + 1)):
        environment.setenv('OPS_TILELANG_NUM_VECTOR_CORES', cap)
        config.clear_runtime_caches()
        for call in (op, compiled):
            with pytest.raises(ValueError):
                _check_core_configuration_call(mode, operation, call, inputs, options)
        environment.delenv('OPS_TILELANG_NUM_VECTOR_CORES')
        config.clear_runtime_caches()
        _check_core_configuration_call(mode, operation, compiled, inputs, options)
        replay = _check_core_configuration_call(mode, operation, compiled, inputs, options)
        assert replay['acl_graph_capture'] == 0 and replay['acl_graph_replay'] > 0


@pytest.mark.parametrize('mode', _MODES)
@pytest.mark.parametrize('operation', ('store', 'load'))
@torch.inference_mode()
def test_cache_compile_empty_core_configuration(mode, operation, monkeypatch):
    try:
        with monkeypatch.context() as environment:
            _run_empty_core_configuration_cases(mode, operation, environment)
    finally:
        config.reset_runtime_config()


def _check_empty_core_output(operation, output):
    if operation == 'store':
        assert output is None
        return
    for tensor in output if isinstance(output, tuple) else (output,):
        assert tensor.numel() == 0


def _run_empty_core_configuration_cases(mode, operation, environment):
    environment.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
    config.reset_runtime_config()
    kernels = _record_cache_kernels(environment)
    values, storage, cache, slots = _case(mode, n=0)
    store, load, options = _operations(mode)
    op = store if operation == 'store' else load
    args = (values, cache, slots) if operation == 'store' else _load_args(mode, cache, slots)
    compiled = _compile(op)
    before = storage.clone()
    for cap in (None, '0', '2', None):
        if cap is None:
            environment.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        else:
            environment.setenv('OPS_TILELANG_NUM_VECTOR_CORES', cap)
        config.clear_runtime_caches()
        for call in (op, compiled):
            output = call(*args, **options)
            _check_empty_core_output(operation, output)
            del output
        assert torch.equal(storage, before)
        assert not kernels, 'empty inputs must not query or compile a cache kernel'
