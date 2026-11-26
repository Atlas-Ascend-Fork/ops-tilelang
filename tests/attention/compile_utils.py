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

"""Profiler assertions shared by attention torch.compile tests."""

import importlib
import re
from typing import Any

import torch

from cann_ops_tilelang import config


def profiled_call(compiled, *args) -> tuple[Any, dict[str, int]]:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        result = compiled(*args)
        torch.npu.synchronize()
    counts = {event.key: event.count for event in profile.key_averages()}
    names = ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')
    return result, {name: counts.get(name, 0) for name in names}


def assert_capture_and_replay(events: list[dict[str, int]]) -> None:
    _check(events[0]['acl_graph_capture'] > 0, events)
    _check(all(event['acl_graph_replay'] > 0 for event in events[1:]), events)
    _check(not any(event['fx_run_eagerly'] for event in events), events)


def _check(condition: bool, details: Any) -> None:
    if not condition:
        raise AssertionError(details)


def assert_tensor_outputs(actual, expected) -> None:
    if isinstance(actual, torch.Tensor):
        _check(isinstance(expected, torch.Tensor), (actual, expected))
        _check(actual.shape == expected.shape, (actual.shape, expected.shape))
        _check(actual.dtype == expected.dtype, (actual.dtype, expected.dtype))
        _check(actual.device == expected.device, (actual.device, expected.device))
        _check(torch.equal(actual, expected), (actual, expected))
        return
    _check(len(actual) == len(expected), (len(actual), len(expected)))
    for got, want in zip(actual, expected, strict=True):
        if got is None or want is None:
            _check(got is want, (got, want))
        else:
            assert_tensor_outputs(got, want)


def record_jit_core_counts(monkeypatch, module_name: str, factory_name: str) -> list[tuple[int, int]]:
    """Record the core argument and generated TIR from actual JIT factory calls."""
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    seen = []

    def recorded_factory(*args, **kwargs):
        requested = kwargs.get('num_cores', args[-1] if args else None)
        tir = factory.get_tir(*args, **kwargs)
        match = re.search(r'thread_binding\((\d+), thread="blockIdx.x"\)', tir.script())
        assert match is not None
        seen.append((requested, int(match.group(1))))
        return factory(*args, **kwargs)

    monkeypatch.setattr(module, factory_name, recorded_factory)
    return seen


def _assert_new_jit_core_counts(observations, previous_lengths, expected: int) -> None:
    for seen, previous in zip(observations, previous_lengths, strict=True):
        assert len(seen) > previous
        assert seen[-1] == (expected, expected)


def assert_core_config_recompiles(call, monkeypatch, observations: tuple[list[tuple[int, int]], ...]) -> None:
    """Check graph recapture and actual JIT cores after configuration changes."""
    monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
    config.clear_runtime_caches()
    default_ai_cores = config.get_num_ai_cores()
    assert default_ai_cores > 2
    try:
        config.set_num_ai_cores(default_ai_cores - 1)
        config.set_num_ai_cores(default_ai_cores)
        previous_lengths = [len(seen) for seen in observations]
        _, initial = profiled_call(call)
        assert initial['acl_graph_capture'] > 0 and not initial['fx_run_eagerly']
        _assert_new_jit_core_counts(observations, previous_lengths, 2 * default_ai_cores)

        config.set_num_ai_cores(default_ai_cores - 1)
        assert config.get_num_vector_cores() == 2 * (default_ai_cores - 1)
        previous_lengths = [len(seen) for seen in observations]
        _, changed = profiled_call(call)
        assert changed['acl_graph_capture'] > 0 and not changed['fx_run_eagerly']
        _assert_new_jit_core_counts(observations, previous_lengths, 2 * (default_ai_cores - 1))

        config.set_num_ai_cores(default_ai_cores - 1)
        previous_lengths = [len(seen) for seen in observations]
        _, repeated = profiled_call(call)
        assert repeated['acl_graph_replay'] > 0 and not repeated['fx_run_eagerly']
        assert [len(seen) for seen in observations] == previous_lengths

        monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', str(2 * default_ai_cores - 4))
        config.clear_runtime_caches()
        assert config.get_num_vector_cores() == 2 * default_ai_cores - 4
        previous_lengths = [len(seen) for seen in observations]
        _, capped = profiled_call(call)
        assert capped['acl_graph_capture'] > 0 and not capped['fx_run_eagerly']
        _assert_new_jit_core_counts(observations, previous_lengths, 2 * default_ai_cores - 4)

        monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES')
        config.clear_runtime_caches()
        previous_lengths = [len(seen) for seen in observations]
        _, restored = profiled_call(call)
        assert restored['acl_graph_capture'] > 0 and not restored['fx_run_eagerly']
        _assert_new_jit_core_counts(observations, previous_lengths, 2 * (default_ai_cores - 1))

        config.reset_runtime_config()
        previous_lengths = [len(seen) for seen in observations]
        _, default_restored = profiled_call(call)
        assert default_restored['acl_graph_capture'] > 0 and not default_restored['fx_run_eagerly']
        _assert_new_jit_core_counts(observations, previous_lengths, 2 * default_ai_cores)
        previous_lengths = [len(seen) for seen in observations]
        _, replayed = profiled_call(call)
        assert replayed['acl_graph_replay'] > 0 and not replayed['fx_run_eagerly']
        assert [len(seen) for seen in observations] == previous_lengths
    finally:
        monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        config.reset_runtime_config()
