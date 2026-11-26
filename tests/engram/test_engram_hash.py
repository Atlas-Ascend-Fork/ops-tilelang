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
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from cann_ops_tilelang import engram_hash
from cann_ops_tilelang.config import get_device
from tests.engram.engram_hash_ref import engram_hash_ref, make_offsets
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_num_tokens, get_test_level
from tests.testing.numeric import assert_equal, count_bytes

os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


@pytest.fixture(autouse=True)
def _isolate_compile_cache():
    # Each fixed-shape case owns its Dynamo cache; replay rounds share one object.
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def generate_test_data(params):
    num_tokens = params['num_tokens']
    max_ngram_size = params['ngram']
    num_ngram_layers = params['layers']
    num_embed_table_per_ngram = params['tables']
    device = get_device()
    ngram_token_ids = torch.randint(
        0,
        100000,
        (num_tokens, max_ngram_size),
        dtype=torch.int32,
        device=device,
    )
    multipliers = torch.randint(
        0,
        100000,
        (num_ngram_layers, max_ngram_size),
        dtype=torch.int64,
        device=device,
    )
    vocab_sizes = torch.randint(
        100000,
        1000000,
        (num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram),
        dtype=torch.int32,
        device=device,
    )
    offsets = make_offsets(vocab_sizes)
    return ngram_token_ids, multipliers, vocab_sizes, offsets


def generate_test_params(level: int) -> list[dict]:
    return [{'num_tokens': t} for t in generate_num_tokens(level)]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_engram_hash(params):
    num_tokens = params['num_tokens']
    max_ngram_size = 3
    num_ngram_layers = 2
    num_embed_table_per_ngram = 8

    ngram_token_ids, multipliers, vocab_sizes, offsets = generate_test_data(
        {
            'num_tokens': num_tokens,
            'ngram': max_ngram_size,
            'layers': num_ngram_layers,
            'tables': num_embed_table_per_ngram,
        }
    )

    output = engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)
    output_ref = engram_hash_ref(ngram_token_ids, multipliers, vocab_sizes, offsets)
    assert_equal(output, output_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_engram_hash_benchmark(benchmark_timer, benchmark_record, params):
    max_ngram_size = 3
    num_ngram_layers = 2
    num_embed_table_per_ngram = 8

    ngram_token_ids, multipliers, vocab_sizes, offsets = generate_test_data(
        {
            **params,
            'ngram': max_ngram_size,
            'layers': num_ngram_layers,
            'tables': num_embed_table_per_ngram,
        }
    )
    output = engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)

    t_us = benchmark_timer(lambda: engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets))

    num_bytes = count_bytes(ngram_token_ids, multipliers, vocab_sizes, offsets, output)
    bandwidth_gbs = num_bytes / t_us / 1e3
    benchmark_record(
        kernel='engram_hash',
        operation='fwd',
        params={
            **params,
            'ngram': max_ngram_size,
            'layers': num_ngram_layers,
            'tables': num_embed_table_per_ngram,
        },
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )


def _hash_inputs(tokens=7):
    return generate_test_data({'num_tokens': tokens, 'ngram': 3, 'layers': 2, 'tables': 2})


def _profile_hash_call(compiled, inputs):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profiler:
        with torch.inference_mode():
            output = compiled(*inputs)
        torch.npu.synchronize()
    counts = {item.key: item.count for item in profiler.key_averages()}
    return output, {key: counts.get(key, 0) for key in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')}


def _save_hash_evidence(name, counts):
    folder = os.environ.get('ENGRAM_COMPILE_EVIDENCE_DIR')
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        Path(folder, name + '.json').write_text(json.dumps(counts, indent=2))


@pytest.mark.parametrize('kind', ['dtype', 'rank', 'stride'])
def test_engram_hash_compile_invalid_metadata(kind):
    from cann_ops_tilelang.engram.engram_hash import engram_hash as registered_hash

    args = list(_hash_inputs())
    if kind == 'dtype':
        args[0] = args[0].to(torch.int16)
    elif kind == 'rank':
        args[0] = args[0].unsqueeze(0)
    else:
        args[0] = args[0][..., ::2]
    with pytest.raises((TypeError, ValueError)):
        engram_hash(*args)
    with FakeTensorMode() as mode:
        fake_args = [mode.from_tensor(value) for value in args]
        with pytest.raises((TypeError, ValueError)):
            registered_hash(*fake_args)


def _refresh_hash_inputs(args, round_id):
    fresh = _hash_inputs()
    if round_id == 1:
        for current, new in zip(args, fresh, strict=True):
            current.copy_(new)
        return args
    return fresh


def test_engram_hash_compile():
    args = _hash_inputs()
    compiled = torch.compile(engram_hash, backend='npugraph_ex', fullgraph=True, dynamic=False)
    evidence = []
    for round_id in range(4):
        if round_id:
            args = _refresh_hash_inputs(args, round_id)
        snapshots = [value.clone() for value in args]
        expected = engram_hash_ref(*args)
        actual, events = _profile_hash_call(compiled, args)
        assert_equal(actual, expected)
        assert actual.shape == expected.shape and actual.dtype == torch.int32
        assert actual.device == args[0].device and actual.is_contiguous()
        for current, snapshot in zip(args, snapshots, strict=True):
            assert torch.equal(current, snapshot)
        evidence.append(events)
        del actual
    assert evidence[0]['acl_graph_capture'] > 0
    assert evidence[1]['acl_graph_capture'] == 0
    assert all(e['acl_graph_replay'] > 0 and e['fx_run_eagerly'] == 0 for e in evidence)
    _save_hash_evidence('engram_hash', evidence)


def test_engram_hash_opcheck():
    from cann_ops_tilelang.engram.engram_hash import engram_hash as registered_hash

    torch.library.opcheck(
        registered_hash,
        _hash_inputs(),
        test_utils=('test_schema', 'test_faketensor'),
    )


@pytest.mark.parametrize('tokens', [0, 1, 129, 8001])
def test_engram_hash_compile_shapes(tokens):
    args = _hash_inputs(tokens)
    expected = engram_hash_ref(*args)
    compiled = torch.compile(engram_hash, backend='npugraph_ex', fullgraph=True, dynamic=False)
    with torch.inference_mode():
        actual = compiled(*args)
        assert_equal(actual, expected)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.device == expected.device and actual.is_contiguous()
        del actual


@pytest.mark.parametrize('value', ['0', '-1', 'invalid', '16'])
def test_engram_hash_fake_ignores_core_configuration(value, monkeypatch):
    module = sys.modules['cann_ops_tilelang.engram.engram_hash']

    def unexpected_query(*args, **kwargs):
        raise AssertionError('Fake must not query runtime or hardware')

    args = _hash_inputs()
    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', value)
    monkeypatch.setattr(module, 'get_num_vector_cores', unexpected_query)
    monkeypatch.setattr(module, 'require_ascend', unexpected_query)
    with FakeTensorMode() as mode:
        tensors = [mode.from_tensor(t) for t in args]
        result = engram_hash(*tensors)
        assert result.shape == (2, 7, 4) and result.dtype == torch.int32


def test_engram_hash_empty_ignores_core_configuration(monkeypatch):
    monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', 'invalid')
    test_engram_hash_compile_shapes(0)
