import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.utils import str_to_dtype
from tests.quant.cast_ref import cast, cast_back
from tests.testing.bench import make_param_id
from tests.testing.generator import (
    generate_cast_config,
    generate_hidden_sizes,
    generate_num_tokens,
    generate_rand_float,
    get_test_level,
)
from tests.testing.numeric import assert_equal, check_bias, count_bytes
from tests.testing.quant import clear_unused_sf

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    fmt = params['fmt']
    block_size = params['block_size']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']

    x = generate_rand_float((num_tokens, hidden)).to(str_to_dtype(in_dtype))
    base_args = dict(
        x=x,
        fmt=fmt,
        block_size=block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    return (x, base_args)


def generate_test_params(level: int) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'in_dtype': in_dtype,
            'fmt': fmt,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'block_size': block_size,
        }
        for num_tokens in generate_num_tokens(level, alignment=128)
        for hidden_size in generate_hidden_sizes(128)
        for in_dtype in ('bf16', 'fp32')
        for fmt in ('e4m3', 'e2m1')
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in (
            [(False, True, False)] if level == 0 else generate_cast_config(level, fmt)
        )
        for block_size in ((128, 128), (32, 32))
    ]


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_block_cast(params):
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    block_size = params['block_size']

    x, base_args = generate_test_data(params)

    # Test cast
    x_casted, per_block_sf_inv = cann_ops_tilelang.per_block_cast(**base_args)
    x_casted_ref, per_block_sf_inv_ref = cast(**base_args)
    x_casted_back = cast_back((x_casted, per_block_sf_inv), 'fp32', block_size)
    if use_packed_ue8m0:
        per_block_sf_inv = clear_unused_sf(per_block_sf_inv, hidden, block_size[1])
        per_block_sf_inv_ref = clear_unused_sf(per_block_sf_inv_ref, hidden, block_size[1])
    assert_equal(per_block_sf_inv, per_block_sf_inv_ref)
    assert_equal(x_casted, x_casted_ref)

    # Check bias
    check_bias(x_casted_back, x)

    # Test cast only mode
    if not use_tma_aligned_col_major_sf:
        # TMA aligned or packed ue8m0 sf is used for FP8/FP4 GEMM, not for other cast
        x_casted = cann_ops_tilelang.per_block_cast_with_precomputed_sf(**base_args, sf=per_block_sf_inv)
        x_casted_ref = cast(**base_args, sf=per_block_sf_inv)
        assert_equal(x_casted, x_casted_ref)

    # Test sf only mode
    twice_per_block_sf_inv = cann_ops_tilelang.per_block_cast_with_sf_only(**base_args)
    if use_packed_ue8m0:
        twice_per_block_sf_inv = clear_unused_sf(twice_per_block_sf_inv, hidden, block_size[1])
    assert_equal(twice_per_block_sf_inv, per_block_sf_inv_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_per_block_cast_benchmark(benchmark_timer, benchmark_record, params):
    x, args = generate_test_data(params)

    x_casted, per_block_sf_inv = cann_ops_tilelang.per_block_cast(**args)

    t_us = benchmark_timer(lambda: cann_ops_tilelang.per_block_cast(**args))
    num_bytes = count_bytes(x, x_casted, per_block_sf_inv)

    benchmark_record(
        kernel='per_block_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


@pytest.mark.parametrize('entry', ['cast', 'sf_only'])
@pytest.mark.parametrize(
    ('fmt', 'dtype', 'block_size', 'tma', 'packed'),
    [
        ('e4m3', torch.bfloat16, (32, 32), False, False),
        ('e2m1', torch.float32, (128, 128), True, True),
    ],
)
@torch.inference_mode()
def test_per_block_cast_compile(entry, fmt, dtype, block_size, tma, packed):
    from tests.testing.quant import call_quant_graph, check_quant_graph_replay, compile_quant_graph

    op = cann_ops_tilelang.per_block_cast if entry == 'cast' else cann_ops_tilelang.per_block_cast_with_sf_only
    compiled = compile_quant_graph(op)
    kw = dict(use_tma_aligned_col_major_sf=tma, round_sf=True, use_packed_ue8m0=packed)
    x = torch.zeros((128, 256), dtype=dtype, device='npu')
    records = []
    for step in range(3):
        if step == 1:
            x.normal_()
        elif step == 2:
            x = torch.randn_like(x)
        before = x.clone()
        expected = cast(x, fmt, block_size, **kw)
        result, events = call_quant_graph(compiled, x, fmt, block_size, **kw)
        records.append(events)
        outputs = result if entry == 'cast' else (result,)
        references = expected if entry == 'cast' else (expected[1],)
        for actual, reference in zip(outputs, references, strict=True):
            assert_equal(actual, reference)
        assert_equal(x, before)
        del result, outputs, actual
    check_quant_graph_replay(records)
    registered = getattr(
        torch.ops.cann_ops_tilelang, 'per_block_cast' if entry == 'cast' else 'per_block_cast_with_sf_only'
    )

    torch.library.opcheck(
        registered.default,
        (x, fmt, list(block_size), tma, True, packed),
        test_utils=('test_schema', 'test_faketensor'),
    )


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@torch.inference_mode()
def test_per_block_cast_precomputed_compile(fmt):
    from tests.testing.quant import call_quant_graph, check_quant_graph_replay, compile_quant_graph

    compiled = compile_quant_graph(cann_ops_tilelang.per_block_cast_with_precomputed_sf)
    block_size = (32, 32)
    x = torch.randn((128, 256), dtype=torch.bfloat16, device='npu')
    sf = cast(x, fmt, block_size, round_sf=True)[1]
    records = []
    for step in range(3):
        if step == 1:
            x.neg_()
            sf.mul_(2)
        elif step == 2:
            x = torch.randn_like(x)
            sf = cast(x, fmt, block_size, round_sf=True)[1]
        before = (x.clone(), sf.clone())
        expected = cast(x, fmt, block_size, sf=sf)
        result, events = call_quant_graph(compiled, x, fmt, block_size, sf)
        records.append(events)
        assert_equal(result.view(torch.uint8), expected.view(torch.uint8))
        assert result.shape == expected.shape and result.dtype == expected.dtype
        assert_equal(x, before[0])
        assert_equal(sf, before[1])
        del result
    check_quant_graph_replay(records)

    torch.library.opcheck(
        torch.ops.cann_ops_tilelang.per_block_cast_with_precomputed_sf.default,
        (x, fmt, list(block_size), sf, False, False, False),
        test_utils=('test_schema', 'test_faketensor'),
    )


@pytest.mark.parametrize('entry', ['cast', 'sf_only', 'precomputed'])
@torch.inference_mode()
def test_per_block_cast_compile_config_changes(entry, monkeypatch):
    from tests.testing.quant import (
        check_quant_config_result,
        check_quant_config_updates,
        compile_quant_graph,
        quant_config_test_context,
        record_quant_config_factory,
    )

    op = {
        'cast': cann_ops_tilelang.per_block_cast,
        'sf_only': cann_ops_tilelang.per_block_cast_with_sf_only,
        'precomputed': cann_ops_tilelang.per_block_cast_with_precomputed_sf,
    }[entry]
    with quant_config_test_context(monkeypatch) as patch:
        calls = record_quant_config_factory('per_block_cast.get_per_block_cast_kernel_asc', patch)
        # Small values make clamp changes observable when the operator computes SF.
        x = torch.linspace(-5e-5, 5e-5, 128 * 128, device='npu').reshape(128, 128).to(torch.bfloat16)
        block = (32, 32)
        sf = cast(x, 'e4m3', block, round_sf=True)[1]
        args = (
            (x, 'e4m3', block, sf, False, True, False)
            if entry == 'precomputed'
            else (x, 'e4m3', block, False, True, False)
        )
        compiled = compile_quant_graph(op)

        def check(recapture):
            if entry == 'precomputed':
                # External SF fixes the scale; changing clamp must not alter it.
                expected = cast(x, 'e4m3', block, sf=sf, round_sf=True)
            else:
                data, scale = cast(x, 'e4m3', block, round_sf=True)
                expected = (data, scale) if entry == 'cast' else scale
            return check_quant_config_result(compiled, args, expected, calls, recapture)

        check_quant_config_updates(check, patch)


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
def test_per_block_fake_without_runtime_queries(fmt, monkeypatch):
    from importlib import import_module

    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    from torch.utils._pytree import tree_leaves

    def reject_query(*args, **kwargs):
        raise AssertionError('Fake execution must not query runtime configuration or build kernels.')

    for name in ('per_block_cast', 'per_block_cast_lossless'):
        module = import_module('cann_ops_tilelang.quant.' + name)
        target = module.config if name == 'per_block_cast_lossless' else module
        for attribute in ('get_amax_clamp_for_quant', 'get_num_vector_cores', 'require_ascend'):
            monkeypatch.setattr(target, attribute, reject_query)
        factory = (
            'get_per_block_cast_kernel_asc' if name == 'per_block_cast' else 'get_per_block_cast_lossless_kernel_asc'
        )
        monkeypatch.setattr(module, factory, reject_query)
    with FakeTensorMode():
        x = torch.empty((128, 256), dtype=torch.bfloat16, device='npu')
        sf = torch.empty((4, 8), dtype=torch.float32, device='npu')
        data = torch.empty((128, 128), dtype=torch.int8, device='npu')
        input_sf = torch.empty((128, 8), dtype=torch.float32, device='npu')
        results = (
            cann_ops_tilelang.per_block_cast(x, fmt, (32, 32)),
            cann_ops_tilelang.per_block_cast_with_sf_only(x, fmt, (32, 32)),
            cann_ops_tilelang.per_block_cast_with_precomputed_sf(x, fmt, (32, 32), sf),
            cann_ops_tilelang.per_block_cast_lossless((data, input_sf), 'e4m3', (1, 32), (32, 32)),
        )
        for value in tree_leaves(results):
            if not isinstance(value, FakeTensor):
                raise AssertionError('Expected metadata-only FakeTensor output.')
