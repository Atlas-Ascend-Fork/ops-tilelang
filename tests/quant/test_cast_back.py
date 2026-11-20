import importlib
import math
import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.utils import str_to_dtype
from tests.quant.cast_ref import cast, cast_back
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_cast_config, generate_hidden_sizes, generate_num_tokens, get_test_level
from tests.testing.numeric import assert_equal, calc_diff, count_bytes

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_data_per_token(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    fmt = params['fmt']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    num_per_channels = params['num_per_channels']
    out_dtype = params['out_dtype']

    x = torch.randn((num_tokens, hidden), dtype=str_to_dtype(out_dtype), device=get_device())
    x_fp8, x_sf = cann_ops_tilelang.per_token_cast(
        x,
        fmt,
        num_per_channels=num_per_channels,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    def func():
        return cann_ops_tilelang.per_token_cast_back((x_fp8, x_sf), out_dtype, num_per_channels=num_per_channels)

    return (x, x_fp8, x_sf, out_dtype, func)


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    round_sf = params['round_sf']
    fmt = params['fmt']
    out_dtype = params['out_dtype']
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']

    x = torch.randn((num_tokens, hidden), dtype=str_to_dtype(out_dtype), device=get_device())
    x_casted, x_sf = cast(x, fmt, (num_per_tokens, num_per_channels), round_sf=round_sf)

    def func():
        return cann_ops_tilelang.cast_back((x_casted, x_sf), out_dtype, (num_per_tokens, num_per_channels))

    return (x, x_casted, x_sf, out_dtype, func)


def generate_test_params_per_token(level: int) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'fmt': fmt,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'num_per_channels': num_per_channels,
            'out_dtype': out_dtype,
        }
        for num_tokens in generate_num_tokens(level)
        for hidden_size in generate_hidden_sizes()
        for fmt in ('e2m1', 'e4m3')
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in generate_cast_config(level, fmt)
        for num_per_channels in (32, 128)
        for out_dtype in ('fp32', 'bf16')
    ]


def generate_test_params(level: int) -> list[dict]:
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'round_sf': round_sf,
            'fmt': fmt,
            'out_dtype': out_dtype,
            'num_per_tokens': num_per_tokens,
            'num_per_channels': num_per_channels,
        }
        for num_tokens in generate_num_tokens(level)
        for hidden_size in generate_hidden_sizes()
        for round_sf in ((True,) if level == 0 else (False, True))
        for fmt in ('e4m3',)
        for out_dtype in ('bf16', 'fp32')
        for num_per_tokens, num_per_channels in ((32, 1), (32, 32), (128, 1), (128, 128))
    ]


@pytest.mark.parametrize('params', generate_test_params_per_token(get_test_level()), ids=make_param_id)
def test_cast_back_per_token(params):
    hidden = params['hidden']
    fmt = params['fmt']
    num_per_channels = params['num_per_channels']

    # Test correctness
    x, x_fp8, x_sf, out_dtype_str, func = generate_test_data_per_token(params)
    x_fp8_bf16 = func()
    x_fp8_bf16_ref = cast_back((x_fp8, x_sf), out_dtype_str, (1, num_per_channels))

    diff = calc_diff(x, x_fp8_bf16)
    assert diff < (2e-2 if fmt == 'e2m1' else 1e-3), (
        f'{x}, {x_fp8_bf16}, {fmt=}, {hidden=}, {num_per_channels=}, {diff=}'
    )

    assert_equal(x_fp8_bf16, x_fp8_bf16_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params_per_token(0), ids=make_param_id)
def test_cast_back_per_token_benchmark(benchmark_timer, benchmark_record, params):
    x, x_fp8, x_sf, out_dtype_str, func = generate_test_data_per_token(params)

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_fp8, x_sf)

    benchmark_record(
        kernel='cast_back_per_token',
        operation='fwd',
        params={**params, 'out_dtype': out_dtype_str},
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_cast_back(params):
    num_per_tokens = params['num_per_tokens']
    num_per_channels = params['num_per_channels']

    _, x_casted, x_sf, out_dtype_str, func = generate_test_data(params)
    x_casted_back = func()
    x_casted_back_ref = cast_back((x_casted, x_sf), out_dtype_str, (num_per_tokens, num_per_channels))

    assert_equal(x_casted_back, x_casted_back_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_cast_back_benchmark(benchmark_timer, benchmark_record, params):
    x, x_casted, x_sf, out_dtype_str, func = generate_test_data(params)

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='cast_back',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@torch.inference_mode()
def test_per_token_cast_compile_then_cast_back(fmt):
    x = torch.randn((31, 512), dtype=torch.bfloat16, device=get_device())

    def model(value):
        sf = cann_ops_tilelang.per_token_cast_with_sf_only(value, fmt, 32, round_sf=True)
        data = cann_ops_tilelang.per_token_cast_with_precomputed_sf(value, fmt, 32, sf, round_sf=True)
        return data, sf

    compiled = torch.compile(model, backend='npugraph_ex', fullgraph=True, dynamic=False)
    for step in range(3):
        if step == 1:
            x.neg_().add_(0.125)
        elif step == 2:
            x = torch.randn_like(x)
        reference = cast(x, fmt, (1, 32), round_sf=True)
        data, sf = compiled(x)
        assert_equal(data, reference[0])
        assert_equal(sf, reference[1])
        restored = cann_ops_tilelang.per_token_cast_back((data, sf), 'fp32', num_per_channels=32)
        assert_equal(restored, cast_back(reference, 'fp32', (1, 32)))
        del data, sf


@pytest.mark.parametrize(
    ('entry', 'fmt', 'out_dtype', 'block_size', 'packed'),
    [('block', 'e4m3', 'fp32', (32, 32), False), ('token', 'e2m1', 'bf16', (1, 32), True)],
)
@torch.inference_mode()
def test_cast_back_compile(entry, fmt, out_dtype, block_size, packed):
    from tests.testing.quant import call_quant_graph, check_quant_graph_replay, compile_quant_graph

    compiled = compile_quant_graph(
        cann_ops_tilelang.cast_back if entry == 'block' else cann_ops_tilelang.per_token_cast_back
    )
    source = None
    records = []
    for step in range(3):
        raw = torch.randn((33, 256), dtype=torch.bfloat16, device=get_device())
        refreshed = cast(raw, fmt, block_size, round_sf=True, use_packed_ue8m0=packed)
        if step == 1:
            source[0].view(torch.uint8).copy_(refreshed[0].view(torch.uint8))
            source[1].copy_(refreshed[1])
        else:
            source = refreshed
        snapshots = tuple(value.clone() for value in source)
        expected = cast_back(source, out_dtype, block_size)
        scale_arg = block_size if entry == 'block' else block_size[1]
        result, events = call_quant_graph(compiled, source, out_dtype, scale_arg)
        records.append(events)
        assert_equal(result, expected)
        for value, before in zip(source, snapshots, strict=True):
            assert_equal(value, before)
        del result
    check_quant_graph_replay(records)

    torch.library.opcheck(
        torch.ops.cann_ops_tilelang.cast_back.default,
        (source, out_dtype, block_size, None),
        # TorchNPU schema_check uses allclose, which does not support FP8 inputs.
        # FP4 runs the same schema check; both cases check read-only inputs above.
        test_utils=('test_faketensor',) if fmt == 'e4m3' else ('test_schema', 'test_faketensor'),
    )


@torch.inference_mode()
def test_unpack_fp4_compile():
    from tests.testing.quant import call_quant_graph, check_quant_graph_replay, compile_quant_graph

    # Independent table for all sixteen E2M1 codes, including signed zero.
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=get_device())
    compiled = compile_quant_graph(cann_ops_tilelang.unpack_from_e2m1fn_x2)
    values = torch.arange(256, device=get_device(), dtype=torch.int32)
    x = values.to(torch.uint8).view(torch.int8).reshape(16, 16)
    records = []
    for step in range(3):
        if step == 1:
            x.copy_(((values + 17) % 256).to(torch.uint8).view(torch.int8).reshape_as(x))
        elif step == 2:
            x = torch.flip(x, dims=[0]).clone()
        before = x.clone()
        codes = x.to(torch.int32)
        expected = torch.stack((levels[codes & 15], levels[(codes >> 4) & 15]), dim=-1).reshape(16, 32)
        if step == 1:
            # Reuse temporary host scalar buffers before replaying the captured graph.
            host_buffers = [torch.full((), 7, dtype=torch.int32, pin_memory=True) for _ in range(128)]
        result, events = call_quant_graph(compiled, x)
        if step == 1:
            del host_buffers
        records.append(events)
        assert_equal(result, expected)
        assert_equal(x, before)
        del result
    check_quant_graph_replay(records)


@pytest.mark.parametrize('entry', ['block', 'token'])
@torch.inference_mode()
def test_cast_back_compile_special_values(entry):
    from tests.testing.quant import call_quant_graph, check_quant_graph_replay, compile_quant_graph

    block = (32, 32) if entry == 'block' else (1, 32)
    compiled = compile_quant_graph(
        cann_ops_tilelang.cast_back if entry == 'block' else cann_ops_tilelang.per_token_cast_back
    )
    source = None
    records = []
    for step in range(3):
        if entry == 'block':
            data = torch.full((33, 256), float(step), dtype=torch.bfloat16, device=get_device())
            data = data.to(torch.float8_e4m3fn)
        else:
            data = torch.full((33, 128), 0x22 * step, dtype=torch.int8, device=get_device())
        scale = torch.full(((33 + block[0] - 1) // block[0], 8), 1.0, dtype=torch.float32, device=get_device())
        if step:
            scale.fill_(2.0 if step == 1 else 0.5)
            row = 0 if step == 1 else scale.shape[0] - 1
            scale[row, 0] = float('nan')
            scale[row, 1] = float('inf')
            scale[row, 2] = -float('inf')
        if step == 1:
            source[0].view(torch.uint8).copy_(data.view(torch.uint8))
            source[1].copy_(scale)
        else:
            source = (data, scale)
        snapshots = tuple(value.clone() for value in source)
        expected = cast_back(source, 'fp32', block)
        scale_arg = block if entry == 'block' else block[1]
        result, events = call_quant_graph(compiled, source, 'fp32', scale_arg)
        records.append(events)
        # NaN payloads are unspecified; finite values, signed zeros and infinities remain bit-exact.
        nan_mask = torch.isnan(expected)
        assert_equal(torch.isnan(result), nan_mask)
        assert_equal(result[~nan_mask], expected[~nan_mask])
        for value, before in zip(source, snapshots, strict=True):
            assert_equal(value, before)
        del result
    check_quant_graph_replay(records)


def _check_quant_config_call(compiled, args, expected, calls, cores, clamp, recapture, mutated=None):
    from tests.testing.quant import call_quant_graph

    before = len(calls)
    tensors = args[0] if isinstance(args[0], tuple) else (args[0],)
    snapshots = tuple(value.clone() for value in tensors)
    result, events = (
        call_quant_graph(compiled, *args, round_sf=True) if clamp is not None else call_quant_graph(compiled, *args)
    )
    assert events['fx_run_eagerly'] == 0
    assert events['acl_graph_replay'] > 0
    assert bool(events['acl_graph_capture']) is recapture
    assert (len(calls) > before) is recapture
    assert calls[-1][0] == cores
    if clamp is not None:
        assert math.isclose(calls[-1][1], clamp, rel_tol=0.0, abs_tol=0.0)
    if mutated is not None:
        assert result is None
        assert_equal(mutated, expected)
    else:
        outputs = result if isinstance(result, tuple) else (result,)
        references = expected if isinstance(expected, tuple) else (expected,)
        for actual, reference in zip(outputs, references, strict=True):
            assert_equal(actual, reference)
        for value, snapshot in zip(tensors, snapshots, strict=True):
            assert_equal(value, snapshot)
    return calls[-1][2]


def _quant_config_test_case(entry, raw):
    if entry in ('cast_back', 'per_token_cast_back'):
        source = cast(raw, 'e4m3', (1, 32), round_sf=True)
        scale_arg = (1, 32) if entry == 'cast_back' else 32
        return (source, 'bf16', scale_arg), cast_back(source, 'bf16', (1, 32))
    if entry == 'per_channel_cast':
        return (raw, None, 'e4m3', 32), cast(raw, 'e4m3', (32, 1), round_sf=True)
    return (raw, 'e4m3', 32), cast_back(cast(raw, 'e4m3', (1, 32), round_sf=True), 'bf16', (1, 32))


def _record_quant_config_factory(entry, patch):
    module_name = 'cast_back' if entry == 'per_token_cast_back' else entry
    factory_name, core_key = {
        'cast_back': ('get_cast_back_kernel_ascend', 'n_cores'),
        'per_channel_cast': ('get_per_channel_cast_kernel_asc', 'num_cores'),
        'per_token_cast_and_cast_back': ('get_per_token_cast_and_cast_back_kernel_asc', 'n_cores'),
    }[module_name]
    module = importlib.import_module('cann_ops_tilelang.quant.' + module_name)
    factory = getattr(module, factory_name)
    calls = []

    def record_factory(*args, **kwargs):
        kernel = factory(*args, **kwargs)
        output_config = kwargs.get('out_config')
        clamp = None if output_config is None else output_config.clamp_min_value
        calls.append((kwargs[core_key], clamp, kernel))
        return kernel

    patch.setattr(module, factory_name, record_factory)
    return calls, module_name


def _check_quant_core_config_changes(check, patch):
    from cann_ops_tilelang import config

    ai_cores = config.get_num_ai_cores()
    default_cores = config.get_num_vector_cores()
    baseline_kernel = check(True)
    # Override, repeat, restore; then add, change and remove the environment cap.
    for step in range(1, 7):
        if step in (1, 2):
            config.set_num_ai_cores(max(1, ai_cores // 2))
        elif step == 3:
            config.reset_runtime_config()
        elif step in (4, 5):
            divisor = 2 if step == 4 else 3
            patch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', str(max(1, default_cores // divisor)))
            config.clear_runtime_caches()
        else:
            patch.delenv('OPS_TILELANG_NUM_VECTOR_CORES')
            config.clear_runtime_caches()
        kernel = check(step != 2)
        if step in (3, 6):
            assert kernel is baseline_kernel
    with pytest.raises(ValueError):
        config.set_num_ai_cores(0)
    check(False)
    return baseline_kernel


def _check_quant_clamp_config_changes(check, clamp, baseline_kernel):
    from cann_ops_tilelang import config

    for value, recapture in ((clamp * 100, True), (clamp * 100, False), (clamp, True)):
        config.set_amax_clamp_for_quant('e4m3', value)
        kernel = check(recapture)
    assert kernel is baseline_kernel
    with pytest.raises(ValueError):
        config.set_amax_clamp_for_quant('e4m3', float('nan'))
    check(False)


@pytest.mark.parametrize(
    'entry', ['cast_back', 'per_token_cast_back', 'per_channel_cast', 'per_token_cast_and_cast_back']
)
@torch.inference_mode()
def test_quant_compile_config_changes(entry, monkeypatch):
    from cann_ops_tilelang import config
    from tests.testing.quant import compile_quant_graph

    with monkeypatch.context() as patch:
        patch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        config.reset_runtime_config()
        try:
            calls, module_name = _record_quant_config_factory(entry, patch)
            raw = torch.linspace(-5e-5, 5e-5, 128 * 128, device=get_device()).reshape(128, 128).to(torch.bfloat16)
            original = raw.clone()
            args, _ = _quant_config_test_case(entry, raw)
            compiled = compile_quant_graph(getattr(cann_ops_tilelang, entry))
            mutated = raw if entry == 'per_token_cast_and_cast_back' else None

            def check(recapture):
                # Keep the callable, tensor metadata and input storage unchanged.
                raw.copy_(original)
                _, expected = _quant_config_test_case(entry, raw)
                clamp = config.get_amax_clamp_for_quant('e4m3') if module_name != 'cast_back' else None
                return _check_quant_config_call(
                    compiled, args, expected, calls, config.get_num_vector_cores(), clamp, recapture, mutated
                )

            baseline_kernel = _check_quant_core_config_changes(check, patch)
            if module_name != 'cast_back':
                _check_quant_clamp_config_changes(check, config.get_amax_clamp_for_quant('e4m3'), baseline_kernel)
        finally:
            config.reset_runtime_config()
            patch.undo()
            config.clear_runtime_caches()
