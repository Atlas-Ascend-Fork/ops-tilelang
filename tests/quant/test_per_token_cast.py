import logging
import math
import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_amax_clamp_for_quant, get_device, set_amax_clamp_for_quant
from cann_ops_tilelang.utils import str_to_dtype
from tests.quant.cast_ref import cast, cast_back
from tests.testing.bench import make_param_id
from tests.testing.generator import generate_hidden_sizes, generate_num_tokens, get_test_level
from tests.testing.numeric import assert_equal, check_bias, count_bytes
from tests.testing.quant import clear_unused_sf, quant_level_rank

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_data(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    in_dtype = params['in_dtype']
    fmt = params['fmt']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    x_block_size = params.get('x_block_size')

    in_with_sf_factor = in_dtype in ('e4m3', 'e2m1')

    if in_with_sf_factor:
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=get_device())
        original_x = x
        x = cast(
            x,
            in_dtype,
            x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
        )
    else:
        x = torch.randn((num_tokens, hidden), dtype=str_to_dtype(in_dtype), device=get_device())
        original_x = x

    base_args = dict(
        x=x,
        fmt=fmt,
        x_block_size=x_block_size,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=round_sf,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    return (x, original_x, base_args, in_with_sf_factor)


_SAMPLED_CAST_CASES = [
    # Raw inputs: cover every SF group, both output formats, both SF layouts,
    # packed/unpacked SF, and the hidden-tail path.
    dict(
        in_dtype='fp32',
        num_per_channels=32,
        x_block_size=None,
        fmt='e4m3',
        use_tma_aligned_col_major_sf=False,
        use_packed_ue8m0=False,
    ),
    dict(
        in_dtype='bf16',
        num_per_channels=64,
        x_block_size=None,
        fmt='e2m1',
        use_tma_aligned_col_major_sf=True,
        use_packed_ue8m0=True,
    ),
    dict(
        in_dtype='fp32',
        num_per_channels=128,
        x_block_size=None,
        fmt='e2m1',
        use_tma_aligned_col_major_sf=False,
        use_packed_ue8m0=True,
    ),
    dict(
        in_dtype='bf16',
        num_per_channels=16,
        x_block_size=None,
        fmt='e4m3',
        use_tma_aligned_col_major_sf=True,
        use_packed_ue8m0=False,
    ),
    # Pre-quantized inputs: cover FP8/FP4 sources and both supported source
    # block layouts. The dedicated packed row-major regression below keeps the
    # formerly failing (32, 32) interaction explicit.
    dict(
        in_dtype='e4m3',
        num_per_channels=32,
        x_block_size=(128, 128),
        fmt='e4m3',
        use_tma_aligned_col_major_sf=False,
        use_packed_ue8m0=False,
    ),
    dict(
        in_dtype='e4m3',
        num_per_channels=128,
        x_block_size=(32, 32),
        fmt='e2m1',
        use_tma_aligned_col_major_sf=True,
        use_packed_ue8m0=True,
    ),
    dict(
        in_dtype='e2m1',
        num_per_channels=32,
        x_block_size=(32, 32),
        fmt='e4m3',
        use_tma_aligned_col_major_sf=False,
        use_packed_ue8m0=True,
    ),
    dict(
        in_dtype='e2m1',
        num_per_channels=128,
        x_block_size=(128, 128),
        fmt='e2m1',
        use_tma_aligned_col_major_sf=True,
        use_packed_ue8m0=False,
    ),
]


def _generate_exhaustive_test_params(level: int) -> list[dict]:
    """Return the full interaction matrix used by Level-2 and benchmarks."""
    return [
        {
            'num_tokens': num_tokens,
            'hidden': hidden_size,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': True,
            'use_packed_ue8m0': use_packed_ue8m0,
            'in_dtype': in_dtype,
            'num_per_channels': num_per_channels,
            'x_block_size': x_block_size,
            'fmt': fmt,
        }
        for num_tokens in generate_num_tokens(level)
        for hidden_size in ([3072, 16384, 65536] if level == 0 else generate_hidden_sizes())
        for use_tma_aligned_col_major_sf in ([True] if level == 0 else [False, True])
        for use_packed_ue8m0 in ([True] if level == 0 else [False, True])
        for in_dtype in ['fp32', 'bf16', 'e4m3', 'e2m1']
        for num_per_channels in ((32, 128) if in_dtype in ('e4m3', 'e2m1') else (32, 64, 128))
        for x_block_size in (((128, 128), (32, 32)) if in_dtype in ('e4m3', 'e2m1') else (None,))
        for fmt in ('e4m3', 'e2m1')
    ]


def _generate_sampled_test_params(level: int) -> list[dict]:
    """Cover each important value once without multiplying independent axes."""
    hidden_sizes = [3072, 16384, 65536] if level == 0 else generate_hidden_sizes()
    cases = _SAMPLED_CAST_CASES[:3] if level == 0 else _SAMPLED_CAST_CASES
    num_tokens = generate_num_tokens(level)
    sample_count = max(len(hidden_sizes), len(cases))

    return [
        {
            'num_tokens': num_tokens[index % len(num_tokens)],
            'hidden': hidden_sizes[index % len(hidden_sizes)],
            'round_sf': True,
            **cases[index % len(cases)],
        }
        for index in range(sample_count)
    ]


def generate_test_params(level: int) -> list[dict]:
    """Generate correctness cases: sampled by default, exhaustive at Level-2."""
    if level >= 2:
        return _generate_exhaustive_test_params(level)
    return _generate_sampled_test_params(level)


def generate_benchmark_params() -> list[dict]:
    """Keep the established performance matrix independent of test level."""
    return _generate_exhaustive_test_params(0)


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_token_cast(params):
    num_tokens = params['num_tokens']
    hidden = params['hidden']
    use_tma_aligned_col_major_sf = params['use_tma_aligned_col_major_sf']
    round_sf = params['round_sf']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    in_dtype = params['in_dtype']
    num_per_channels = params['num_per_channels']
    x_block_size = params.get('x_block_size')
    fmt = params['fmt']

    in_with_sf_factor = in_dtype in ('e4m3', 'e2m1')
    # Test correctness
    x, original_x, base_args, in_with_sf_factor = generate_test_data(params)

    def func():
        return cann_ops_tilelang.per_token_cast(
            **base_args,
            num_per_channels=num_per_channels,
        )

    def func_ref():
        return cast(
            **base_args,
            block_size=(1, num_per_channels),
        )

    x_casted, x_sf = func()
    x_casted_ref, x_sf_ref = func_ref()
    x_casted_back = cast_back((x_casted, x_sf), 'fp32', (1, num_per_channels))

    if use_packed_ue8m0:
        x_sf = clear_unused_sf(x_sf, hidden, num_per_channels)
        x_sf_ref = clear_unused_sf(x_sf_ref, hidden, num_per_channels)

    assert_equal(x_casted, x_casted_ref)
    assert_equal(x_sf, x_sf_ref)

    # Check bias
    check_bias(x_casted_back, original_x)

    # Test non-contiguous input (stride(0) != hidden)
    if not in_with_sf_factor and fmt == 'e4m3' and not use_packed_ue8m0 and num_tokens > 0:
        x_non_contiguous = torch.randn((num_tokens, hidden * 2), dtype=str_to_dtype(in_dtype), device=get_device())[
            :, :hidden
        ]
        x_non_contiguous.copy_(original_x)
        x_non_contiguous_casted, x_non_contiguous_sf = cann_ops_tilelang.per_token_cast(
            x=x_non_contiguous,
            fmt=fmt,
            x_block_size=x_block_size,
            use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
            round_sf=round_sf,
            use_packed_ue8m0=use_packed_ue8m0,
            num_per_channels=num_per_channels,
        )
        assert_equal(x_non_contiguous_casted, x_casted)
        assert_equal(x_non_contiguous_sf, x_sf)

    if not in_with_sf_factor and num_per_channels != hidden:
        base_args.pop('x_block_size')
        # Test cast only mode
        if not use_tma_aligned_col_major_sf and not use_packed_ue8m0:
            # TMA aligned or packed ue8m0 sf is used for FP8/FP4 GEMM, not for other cast
            x_casted = cann_ops_tilelang.per_token_cast_with_precomputed_sf(
                **base_args, num_per_channels=num_per_channels, sf=x_sf
            )
            x_casted_ref = cast(**base_args, block_size=(1, num_per_channels), sf=x_sf)
            assert_equal(x_casted, x_casted_ref)

        # Test sf only mode
        twice_sf_inv = cann_ops_tilelang.per_token_cast_with_sf_only(**base_args, num_per_channels=num_per_channels)
        if use_packed_ue8m0:
            twice_sf_inv = clear_unused_sf(twice_sf_inv, hidden, num_per_channels)
        assert_equal(twice_sf_inv, x_sf)


@pytest.mark.parametrize(
    ('fmt', 'use_tma_aligned_col_major_sf', 'use_packed_ue8m0'),
    [
        pytest.param('e4m3', False, False, id='fp8-row-major-fp32-sf'),
        pytest.param('e2m1', True, True, id='fp4-col-major-packed-sf'),
    ],
)
def test_per_token_cast_full_row(fmt, use_tma_aligned_col_major_sf, use_packed_ue8m0):
    num_tokens = 31
    hidden = 576
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=get_device())
    kwargs = dict(
        fmt=fmt,
        num_per_channels=hidden,
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=True,
        use_packed_ue8m0=use_packed_ue8m0,
    )

    x_casted, x_sf = cann_ops_tilelang.per_token_cast(x, **kwargs)
    x_casted_ref, x_sf_ref = cast(
        x,
        fmt,
        block_size=(1, hidden),
        use_tma_aligned_col_major_sf=use_tma_aligned_col_major_sf,
        round_sf=True,
        use_packed_ue8m0=use_packed_ue8m0,
    )
    if use_packed_ue8m0:
        x_sf = clear_unused_sf(x_sf, hidden, hidden)
        x_sf_ref = clear_unused_sf(x_sf_ref, hidden, hidden)

    assert_equal(x_casted, x_casted_ref)
    assert_equal(x_sf, x_sf_ref)


def test_per_token_cast_group16_hidden_tail():
    num_tokens = 31
    hidden = 176
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=get_device())
    x_casted, x_sf = cann_ops_tilelang.per_token_cast(x, 'e4m3', num_per_channels=16, round_sf=True)
    x_casted_ref, x_sf_ref = cast(x, 'e4m3', block_size=(1, 16), round_sf=True)

    assert_equal(x_casted, x_casted_ref)
    assert_equal(x_sf, x_sf_ref)


@pytest.mark.parametrize(
    ('in_dtype', 'fmt'),
    [
        pytest.param('e4m3', 'e2m1', id='fp8-to-fp4'),
        pytest.param('e2m1', 'e4m3', id='fp4-to-fp8'),
    ],
)
def test_per_token_cast_prequantized_packed_row_major(in_dtype, fmt):
    num_tokens = 65
    hidden = 576
    input_block_size = (32, 32)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=get_device())
    x_quantized = cast(
        x,
        in_dtype,
        block_size=input_block_size,
        round_sf=True,
        use_packed_ue8m0=True,
    )
    cast_kwargs = dict(
        fmt=fmt,
        x_block_size=input_block_size,
        round_sf=True,
        use_packed_ue8m0=True,
    )

    x_casted, x_sf = cann_ops_tilelang.per_token_cast(x_quantized, num_per_channels=32, **cast_kwargs)
    x_casted_ref, x_sf_ref = cast(x_quantized, block_size=(1, 32), **cast_kwargs)

    assert_equal(x_casted, x_casted_ref)
    assert_equal(
        clear_unused_sf(x_sf, hidden, 32),
        clear_unused_sf(x_sf_ref, hidden, 32),
    )


def test_per_token_cast_full_row_stochastic():
    num_tokens = 31
    hidden = 512
    x = torch.randn((num_tokens, hidden), dtype=torch.float32, device=get_device())
    kwargs = dict(fmt='e4m3', num_per_channels=hidden, round_sf=True)

    x_stochastic, sf_stochastic = cann_ops_tilelang.per_token_cast(x, **kwargs, stochastic_cast=True)
    x_stochastic_again, sf_stochastic_again = cann_ops_tilelang.per_token_cast(x, **kwargs, stochastic_cast=True)
    x_nearest, sf_nearest = cann_ops_tilelang.per_token_cast(x, **kwargs)

    assert_equal(x_stochastic, x_stochastic_again)
    assert_equal(sf_stochastic, sf_stochastic_again)
    assert_equal(sf_stochastic, sf_nearest)
    assert not torch.equal(x_stochastic.view(torch.uint8), x_nearest.view(torch.uint8))
    check_bias(cast_back((x_stochastic, sf_stochastic), 'fp32', (1, hidden)), x)
    stochastic_rank = quant_level_rank(x_stochastic.cpu(), 'e4m3')
    nearest_rank = quant_level_rank(x_nearest.cpu(), 'e4m3')
    assert ((stochastic_rank - nearest_rank).abs() <= 1).all()


def test_set_amax_clamp_for_quant():
    num_tokens = 32
    hidden = 32
    default_clamp = get_amax_clamp_for_quant('e4m3')
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=get_device())
    _, sf_default = cann_ops_tilelang.per_token_cast(x, 'e4m3', num_per_channels=hidden, round_sf=False)
    set_amax_clamp_for_quant('e4m3', 1e6)
    try:
        x_kernel, sf_kernel = cann_ops_tilelang.per_token_cast(x, 'e4m3', num_per_channels=hidden, round_sf=False)
        x_ref, sf_ref = cast(x, 'e4m3', block_size=(1, hidden), round_sf=False)
        assert_equal(x_kernel, x_ref)
        assert_equal(sf_kernel, sf_ref)
        min_expected_sf = 1e6 / 448.0
        assert (sf_kernel >= min_expected_sf * 0.99).all()
        assert (sf_kernel > sf_default).all()
    finally:
        set_amax_clamp_for_quant('e4m3', default_clamp)


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_per_token_cast_stochastic(params):
    hidden = params['hidden']
    num_per_channels = params['num_per_channels']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    fmt = params['fmt']

    x, original_x, base_args, in_with_sf = generate_test_data(params)
    ref = cast_back(x, 'fp32', params['x_block_size']) if in_with_sf else original_x.float()
    scale = ref.abs().mean()

    def func_stochastic():
        return cann_ops_tilelang.per_token_cast(**base_args, num_per_channels=num_per_channels, stochastic_cast=True)

    def func_nearest():
        return cann_ops_tilelang.per_token_cast(**base_args, num_per_channels=num_per_channels, stochastic_cast=False)

    if ref.shape[0] == 0:
        func_stochastic()
        return

    # Check determinism
    x1_casted, sf1 = func_stochastic()
    x2_casted, sf2 = func_stochastic()
    sf1_masked, sf2_masked = sf1, sf2
    if use_packed_ue8m0:
        sf1_masked = clear_unused_sf(sf1, hidden, num_per_channels)
        sf2_masked = clear_unused_sf(sf2, hidden, num_per_channels)
    assert_equal(x1_casted, x2_casted)
    assert_equal(sf1_masked, sf2_masked)

    # Check SR differs from RNE (skipped if input is exactly representable in the output format)
    x_casted_nearest, sf_nearest = func_nearest()
    in_dtype = params['in_dtype']
    exactly_representable = in_dtype == 'e2m1' or (in_dtype == 'e4m3' and fmt == 'e4m3')
    if not exactly_representable:
        assert not torch.equal(x1_casted.view(torch.uint8), x_casted_nearest.view(torch.uint8)), (
            'SR output is identical to RNE'
        )

    # Check rounding direction bias
    dequant_stochastic = cast_back((x1_casted, sf1), 'fp32', (1, num_per_channels))
    check_bias(dequant_stochastic, ref)

    # Check rounding magnitude bias: mean signed error ~ N(0, step^2 / n), step ~ 2^(-mantissa_bits)
    mantissa_bits = 3 if fmt == 'e4m3' else 1
    step = 2.0 ** (-mantissa_bits)
    mean_error = (dequant_stochastic - ref).mean().abs() / scale
    allowed_mean_error = 10 * step / math.sqrt(ref.numel())
    assert mean_error < allowed_mean_error, (
        f'Mean error not close to 0 (size = {ref.numel()}): {mean_error=:.2e} {allowed_mean_error=:.2e}'
    )

    # Check SR and RNE pick numerically adjacent quant levels, and use identical sf.
    sf_nearest_masked = clear_unused_sf(sf_nearest, hidden, num_per_channels) if use_packed_ue8m0 else sf_nearest
    assert_equal(sf1_masked, sf_nearest_masked)
    rank_stochastic = quant_level_rank(x1_casted.cpu(), fmt)
    rank_nearest = quant_level_rank(x_casted_nearest.cpu(), fmt)
    assert ((rank_stochastic - rank_nearest).abs() <= 1).all(), (
        f'SR output is not adjacent to RNE: max rank gap={(rank_stochastic - rank_nearest).abs().max().item()}'
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_benchmark_params(), ids=make_param_id)
def test_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    num_per_channels = params['num_per_channels']

    x, _, base_args, _ = generate_test_data(params)

    def func():
        return cann_ops_tilelang.per_token_cast(
            **base_args,
            num_per_channels=num_per_channels,
        )

    x_casted, x_sf = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='per_token_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_benchmark_params(), ids=make_param_id)
def test_per_token_cast_stochastic_benchmark(benchmark_timer, benchmark_record, params):
    num_per_channels = params['num_per_channels']

    x, _, base_args, _ = generate_test_data(params)

    def func():
        return cann_ops_tilelang.per_token_cast(
            **base_args,
            num_per_channels=num_per_channels,
            stochastic_cast=True,
        )

    x_casted, x_sf = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='per_token_cast_stochastic',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


def generate_vllm_benchmark_params():
    cases = [
        ('1.6T out_proj 1536', 128, True, 1536),
        ('1.6T out_proj 16384', 128, True, 16384),
        ('1.6T megamoe 7168', 32, False, 7168),
        ('1.6T attn 65536', 128, True, 65536),
        ('284B out_proj 1024', 128, True, 1024),
        ('284B out_proj 8192', 128, True, 8192),
        ('284B megamoe 4096', 32, False, 4096),
        ('284B attn 32768', 128, True, 32768),
    ]
    num_tokens_list = [256, 512, 32768]
    params = []
    for num_tokens in num_tokens_list:
        for _name, num_per_channels, use_col_major, hidden_size in cases:
            params.append(
                {
                    'num_tokens': num_tokens,
                    'hidden': hidden_size,
                    'in_dtype': 'bf16',
                    'num_per_channels': num_per_channels,
                    'fmt': 'e4m3',
                    'use_tma_aligned_col_major_sf': use_col_major,
                    'round_sf': True,
                    'use_packed_ue8m0': True,
                    'x_block_size': None,
                }
            )
    return params


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_vllm_benchmark_params(), ids=make_param_id)
def test_vllm_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    num_per_channels = params['num_per_channels']

    x, _, base_args, _ = generate_test_data(params)

    def func():
        return cann_ops_tilelang.per_token_cast(
            **base_args,
            num_per_channels=num_per_channels,
        )

    x_casted, x_sf = func()

    t_us = benchmark_timer(func)
    num_bytes = count_bytes(x, x_casted, x_sf)

    benchmark_record(
        kernel='per_token_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )


logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _isolate_compile_cache(request):
    # Each parameter case builds an independent graph; retain disk JIT caches.
    if 'compile' in request.node.name:
        torch.compiler.reset()
    yield
    if 'compile' in request.node.name:
        torch.compiler.reset()


def _compiled():
    return torch.compile(cann_ops_tilelang.per_token_cast, backend='npugraph_ex', fullgraph=True, dynamic=False)


def _reference(x, kw):
    return cast(
        x,
        kw['fmt'],
        block_size=(1, kw['num_per_channels']),
        x_block_size=kw.get('x_block_size'),
        round_sf=kw.get('round_sf', False),
        use_tma_aligned_col_major_sf=kw.get('use_tma_aligned_col_major_sf', False),
        use_packed_ue8m0=kw.get('use_packed_ue8m0', False),
    )


def _check(actual, expected, hidden, group, packed):
    assert isinstance(actual, tuple) and len(actual) == 2
    for value, reference in zip(actual, expected, strict=True):
        assert value.shape == reference.shape
        assert value.dtype == reference.dtype
        assert value.device == reference.device
        if value.numel():
            assert value.stride() == reference.stride()
        assert value.storage_offset() == 0
    assert_equal(actual[0], expected[0])
    sf, ref_sf = actual[1], expected[1]
    if packed:
        groups = (hidden + group - 1) // group

        def valid_bytes(scale):
            if scale.numel() == 0:
                return scale.contiguous().flatten().view(torch.uint8)
            # Both layouts return packed channel-group pairs per token.
            # Make a snapshot before dropping the unused padding byte.
            return scale.contiguous().view(torch.uint8)[:, :groups]

        sf = valid_bytes(sf)
        ref_sf = valid_bytes(ref_sf)
    assert_equal(sf, ref_sf)


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@pytest.mark.parametrize('tma', [False, True], ids=['row', 'column'])
@pytest.mark.parametrize('packed', [False, True], ids=['fp32-sf', 'packed-sf'])
@pytest.mark.parametrize('tokens', [0, 31])
@torch.inference_mode()
def test_per_token_cast_compile_layout(fmt, tma, packed, tokens):
    hidden, group = 176, 16
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device=get_device())
    kw = dict(fmt=fmt, num_per_channels=group, round_sf=True, use_tma_aligned_col_major_sf=tma, use_packed_ue8m0=packed)
    compiled = _compiled()
    groups = (hidden + group - 1) // group
    for step in range(3):
        if step == 1:
            x.mul_(-1.25).add_(0.125)
        elif step == 2:
            x = torch.randn_like(x)
        before = x.clone()
        expected = _reference(x, kw)
        result = compiled(x, **kw)
        _check(result, expected, hidden, group, packed)
        packed_shape = (tokens, (groups + 1) // 2)
        shape = packed_shape if packed else (tokens, groups)
        assert result[1].shape == shape
        assert result[1].stride() == ((1, shape[0]) if tma else (shape[1], 1))
        assert_equal(x, before)
        del result


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_per_token_cast_compile_strided(dtype):
    hidden, group = 576, 32
    backing = torch.randn((31, hidden * 2), dtype=dtype, device=get_device())
    kw = dict(fmt='e4m3', num_per_channels=group, round_sf=False)
    compiled = _compiled()
    for step in range(3):
        if step == 2:
            backing = torch.randn_like(backing)
        x = backing[:, :hidden]
        if step == 1:
            x.neg_().add_(0.25)
        before = backing.clone()
        expected = _reference(x, kw)
        result = compiled(x, **kw)
        _check(result, expected, hidden, group, False)
        assert x.stride() == (hidden * 2, 1)
        assert_equal(backing, before)
        del result


def _copy_quantized_source(destination, source):
    for target, value in zip(destination, source, strict=True):
        if target.dtype == torch.float8_e4m3fn:
            target.view(torch.uint8).copy_(value.view(torch.uint8))
        else:
            target.copy_(value)


@pytest.mark.parametrize(('source_fmt', 'fmt'), [('e4m3', 'e2m1'), ('e2m1', 'e4m3')])
@pytest.mark.parametrize('source_tma', [False, True])
@torch.inference_mode()
def test_per_token_cast_compile_prequantized(source_fmt, fmt, source_tma):
    hidden, group = 576, 32
    kw = dict(fmt=fmt, num_per_channels=group, x_block_size=(32, 32), round_sf=True)
    compiled = _compiled()
    source = None
    for step in range(3):
        original = torch.randn((65, hidden), dtype=torch.bfloat16, device=get_device())
        refreshed = cast(
            original,
            source_fmt,
            (32, 32),
            round_sf=True,
            use_tma_aligned_col_major_sf=source_tma,
            use_packed_ue8m0=True,
        )
        if step == 1:
            _copy_quantized_source(source, refreshed)
        else:
            source = refreshed
        before = tuple(t.clone() for t in source)
        before_layout = tuple((t.stride(), t.storage_offset()) for t in source)
        expected = _reference(source, kw)
        result = compiled(source, **kw)
        _check(result, expected, hidden, group, False)
        for value, snapshot, layout in zip(source, before, before_layout, strict=True):
            # clone may compact a strided view; compare its values separately.
            assert_equal(value, snapshot, check_stride=False)
            assert (value.stride(), value.storage_offset()) == layout
        del result


@pytest.mark.parametrize(
    ('fmt', 'packed', 'raw_sf'),
    [
        ('e4m3', False, False),
        ('e2m1', True, False),
        ('e4m3', True, True),
    ],
)
@torch.inference_mode()
def test_per_token_cast_compile_out(fmt, packed, raw_sf):
    hidden, group = 512, 32
    x = torch.randn((31, hidden), dtype=torch.bfloat16, device=get_device())
    kw = dict(fmt=fmt, num_per_channels=group, round_sf=True, use_packed_ue8m0=packed)
    expected = _reference(x, kw)
    out = tuple(torch.empty_like(t) for t in expected)
    if raw_sf:
        out = (out[0], out[1].view(torch.uint8))
    compiled = _compiled()
    for step in range(3):
        if step == 1:
            x.neg_().add_(0.125)
        elif step == 2:
            x = torch.randn_like(x)
            out = tuple(torch.empty_like(t) for t in out)
        for buffer in out:
            buffer.view(torch.uint8).fill_(0x5A)
        before = x.clone()
        expected = _reference(x, kw)
        result = compiled(x, **kw, out=out)
        _check(result, expected, hidden, group, packed)
        assert result[0] is out[0]
        assert result[0].data_ptr() == out[0].data_ptr()
        assert result[1].data_ptr() == out[1].data_ptr()
        sf_view = out[1].view(torch.int16) if packed else out[1]
        _check((out[0], sf_view), expected, hidden, group, packed)
        assert_equal(x, before)
        del result


@torch.inference_mode()
def test_per_token_cast_compile_stochastic():
    # Kernel randomness is derived from inputs/indexes, not a new RNG seed per call.
    x = torch.randn((31, 512), dtype=torch.float32, device=get_device())
    kw = dict(fmt='e4m3', num_per_channels=512, round_sf=True, stochastic_cast=True)
    compiled = _compiled()
    for step in range(3):
        if step == 1:
            x.neg_().add_(0.125)
        elif step == 2:
            x = torch.randn_like(x)
        expected = cann_ops_tilelang.per_token_cast(x, **kw)
        for _ in range(2):
            result = compiled(x, **kw)
            _check(result, expected, 512, 512, False)
            del result


@torch.inference_mode()
def test_per_token_cast_compile_config_change():
    x = torch.zeros((31, 512), dtype=torch.bfloat16, device=get_device())
    kw = dict(fmt='e4m3', num_per_channels=128, round_sf=False)
    default = get_amax_clamp_for_quant('e4m3')
    compiled = _compiled()
    try:
        for clamp in (default, 1e6, default):
            set_amax_clamp_for_quant('e4m3', clamp)
            expected = _reference(x, kw)
            result = compiled(x, **kw)
            _check(result, expected, 512, 128, False)
            del result
    finally:
        set_amax_clamp_for_quant('e4m3', default)


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@pytest.mark.parametrize('with_out', [False, True], ids=['allocate', 'out'])
def test_per_token_cast_compile_opcheck(with_out, fmt):
    x = torch.randn((31, 512), dtype=torch.bfloat16, device=get_device())
    args = (x, None, fmt, 32, None, [False, True, False, False], get_amax_clamp_for_quant(fmt))
    op = torch.ops.cann_ops_tilelang.per_token_cast.default
    if with_out:
        dtype = torch.float8_e4m3fn if fmt == 'e4m3' else torch.int8
        width = 512 if fmt == 'e4m3' else 256
        args += (
            torch.empty((31, width), dtype=dtype, device=x.device),
            torch.empty((31, 16), dtype=torch.float32, device=x.device),
        )
        op = torch.ops.cann_ops_tilelang.per_token_cast_out.default
    # TorchNPU isclose cannot compare FP8 inputs inside PyTorch's schema checker.
    # FP4 covers the same mutation schema; FP8 writes/aliases are byte-checked above.
    checks = ('test_faketensor',) if with_out and fmt == 'e4m3' else ('test_schema', 'test_faketensor')
    torch.library.opcheck(op, args, test_utils=checks)


@pytest.mark.parametrize('with_out', [False, True], ids=['allocate', 'out'])
@torch.inference_mode()
def test_per_token_cast_compile_replay(with_out):
    x = torch.randn((31, 512), dtype=torch.bfloat16, device=get_device())
    kw = dict(fmt='e4m3', num_per_channels=32, round_sf=True)
    if with_out:
        kw['out'] = (
            torch.empty_like(x, dtype=torch.float8_e4m3fn),
            torch.empty((31, 16), dtype=torch.float32, device=x.device),
        )
    compiled = _compiled()
    records = []
    for step in range(3):
        if step:
            x.neg_().add_(0.125)
        expected = _reference(x, kw)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            result = compiled(x, **kw)
            torch.npu.synchronize()
        events = {event.key: event.count for event in profile.key_averages()}
        counts = {name: events.get(name, 0) for name in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')}
        records.append(counts)
        logger.info('graph call %s: %s', step, counts)
        _check(result, expected, 512, 32, False)
        del result
    assert sum(e['acl_graph_capture'] for e in records) > 0, records
    assert records[-1]['acl_graph_replay'] > 0, records
    assert records[-1]['acl_graph_capture'] == 0, records
    assert records[-1]['fx_run_eagerly'] == 0, records


@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@pytest.mark.parametrize('tma', [False, True])
@pytest.mark.parametrize('packed', [False, True])
def test_per_token_cast_compile_fake_layout(fmt, tma, packed):
    x = torch.randn((31, 176), dtype=torch.bfloat16, device=get_device())
    args = (x, None, fmt, 16, None, [tma, True, packed, False], get_amax_clamp_for_quant(fmt))
    torch.library.opcheck(torch.ops.cann_ops_tilelang.per_token_cast.default, args, test_utils=('test_faketensor',))


@torch.inference_mode()
def test_per_token_cast_compile_composition():
    def model(x):
        data, scale = cann_ops_tilelang.per_token_cast(x + 0.25, 'e4m3', 32, round_sf=True)
        return (data.float() * scale.repeat_interleave(32, dim=1)).sum(dim=1)

    x = torch.randn((31, 512), dtype=torch.bfloat16, device=get_device())
    compiled = torch.compile(model, backend='npugraph_ex', fullgraph=True, dynamic=False)
    for step in range(3):
        if step == 1:
            x.neg_().add_(0.125)
        elif step == 2:
            x = torch.randn_like(x)
        data, scale = cast(x + 0.25, 'e4m3', (1, 32), round_sf=True)
        expected = (data.float() * scale.repeat_interleave(32, dim=1)).sum(dim=1)
        result = compiled(x)
        assert_equal(result, expected)
        del result


def _raw_precomputed_scale(scale, tma, packed):
    # The existing cast-only implementation passes sf directly to the kernel.
    raw = scale.T if tma else scale
    return raw.view(torch.uint8) if packed else raw


@pytest.mark.parametrize('entry', ['sf_only', 'precomputed'])
@pytest.mark.parametrize('fmt', ['e4m3', 'e2m1'])
@pytest.mark.parametrize('tma', [False, True])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('tokens', [0, 31])
@torch.inference_mode()
def test_per_token_cast_variants_compile_layout(entry, fmt, tma, packed, tokens):
    x = torch.randn((tokens, 176), dtype=torch.bfloat16, device=get_device())
    kw = dict(fmt=fmt, num_per_channels=16, round_sf=True, use_tma_aligned_col_major_sf=tma, use_packed_ue8m0=packed)
    op = (
        cann_ops_tilelang.per_token_cast_with_sf_only
        if entry == 'sf_only'
        else cann_ops_tilelang.per_token_cast_with_precomputed_sf
    )
    compiled = torch.compile(op, backend='npugraph_ex', fullgraph=True, dynamic=False)
    sf_buffer = None
    for step in range(3):
        if step == 1:
            x.neg_().add_(0.125)
        elif step == 2:
            x = torch.randn_like(x)
        before = x.clone()
        reference = _reference(x, kw)
        if entry == 'sf_only':
            result = compiled(x, **kw)
            _check((reference[0], result), reference, 176, 16, packed)
        else:
            refreshed = _raw_precomputed_scale(reference[1], tma, packed)
            if step == 1:
                sf_buffer.copy_(refreshed)
            else:
                sf_buffer = refreshed
            sf_before = sf_buffer.clone()
            sf_meta = (sf_buffer.shape, sf_buffer.stride(), sf_buffer.storage_offset())
            result = compiled(x, sf=sf_buffer, **kw)
            _check((result, reference[1]), reference, 176, 16, packed)
            assert_equal(sf_buffer, sf_before, check_stride=False)
            assert (sf_buffer.shape, sf_buffer.stride(), sf_buffer.storage_offset()) == sf_meta
        assert_equal(x, before)
        del result


@pytest.mark.parametrize('entry', ['sf_only', 'precomputed'])
@pytest.mark.parametrize(
    ('fmt', 'packed', 'raw_out'),
    [
        ('e4m3', False, False),
        ('e2m1', True, False),
        ('e4m3', True, True),
    ],
)
@torch.inference_mode()
def test_per_token_cast_variants_compile_out(entry, fmt, packed, raw_out):
    x = torch.randn((31, 512), dtype=torch.bfloat16, device=get_device())
    kw = dict(fmt=fmt, num_per_channels=32, round_sf=True, use_packed_ue8m0=packed)
    op = (
        cann_ops_tilelang.per_token_cast_with_sf_only
        if entry == 'sf_only'
        else cann_ops_tilelang.per_token_cast_with_precomputed_sf
    )
    compiled = torch.compile(op, backend='npugraph_ex', fullgraph=True, dynamic=False)
    reference = _reference(x, kw)
    out = torch.empty_like(reference[1 if entry == 'sf_only' else 0])
    if entry == 'sf_only' and raw_out:
        out = out.view(torch.uint8)
    for step in range(3):
        if step == 1:
            x.neg_().add_(0.125)
        elif step == 2:
            x = torch.randn_like(x)
            out = torch.empty_like(out)
        out.view(torch.uint8).fill_(0x5A)
        before = x.clone()
        reference = _reference(x, kw)
        if entry == 'sf_only':
            result = compiled(x, out_sf=out, **kw)
            _check((reference[0], result), reference, 512, 32, packed)
            assert result.data_ptr() == out.data_ptr()
        else:
            sf = _raw_precomputed_scale(reference[1], False, packed)
            sf_before = sf.clone()
            result = compiled(x, sf=sf, out=out, **kw)
            _check((result, reference[1]), reference, 512, 32, packed)
            assert result is out
            assert_equal(sf, sf_before)
        assert_equal(x, before)
        del result


@pytest.mark.parametrize('entry', ['sf_only', 'precomputed'])
@pytest.mark.parametrize('with_out', [False, True])
def test_per_token_cast_variants_compile_opcheck(entry, with_out):
    x = torch.randn((31, 512), dtype=torch.bfloat16, device=get_device())
    kw = dict(fmt='e2m1', num_per_channels=32, round_sf=True)
    reference = _reference(x, kw)
    prefix = 'per_token_cast_with_' + ('sf_only' if entry == 'sf_only' else 'precomputed_sf')
    args = (x,)
    if entry == 'precomputed':
        args += (reference[1],)
    args += ('e2m1', 32, [False, True, False, False], get_amax_clamp_for_quant('e2m1'))
    if with_out:
        args += (torch.empty_like(reference[1 if entry == 'sf_only' else 0]),)
        prefix += '_out'
    torch.library.opcheck(
        getattr(torch.ops.cann_ops_tilelang, prefix).default, args, test_utils=('test_schema', 'test_faketensor')
    )


@pytest.mark.parametrize('entry', ['sf_only', 'precomputed'])
@pytest.mark.parametrize('with_out', [False, True])
@torch.inference_mode()
def test_per_token_cast_variants_compile_replay(entry, with_out):
    x = torch.randn((31, 512), dtype=torch.bfloat16, device=get_device())
    kw = dict(fmt='e4m3', num_per_channels=32, round_sf=True)
    op = (
        cann_ops_tilelang.per_token_cast_with_sf_only
        if entry == 'sf_only'
        else cann_ops_tilelang.per_token_cast_with_precomputed_sf
    )
    compiled = torch.compile(op, backend='npugraph_ex', fullgraph=True, dynamic=False)
    reference = _reference(x, kw)
    sf = reference[1].clone()
    out = torch.empty_like(reference[1 if entry == 'sf_only' else 0])
    records = []
    for step in range(3):
        if step:
            x.neg_().add_(0.125)
        reference = _reference(x, kw)
        sf.copy_(reference[1])
        call_kw = dict(kw)
        if entry == 'precomputed':
            call_kw['sf'] = sf
        if with_out:
            call_kw['out_sf' if entry == 'sf_only' else 'out'] = out
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
            result = compiled(x, **call_kw)
            torch.npu.synchronize()
        events = {e.key: e.count for e in profile.key_averages()}
        counts = {name: events.get(name, 0) for name in ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')}
        logger.info('%s out=%s call %s: %s', entry, with_out, step, counts)
        records.append(counts)
        if entry == 'sf_only':
            _check((reference[0], result), reference, 512, 32, False)
        else:
            _check((result, reference[1]), reference, 512, 32, False)
            assert_equal(sf, reference[1])
        del result
    assert sum(e['acl_graph_capture'] for e in records) > 0, records
    assert records[-1]['acl_graph_replay'] > 0, records
    assert records[-1]['acl_graph_capture'] == 0, records
    assert records[-1]['fx_run_eagerly'] == 0, records


@torch.inference_mode()
def test_per_token_cast_sf_only_compile_config_change():
    x = torch.zeros((31, 512), dtype=torch.bfloat16, device=get_device())
    op = cann_ops_tilelang.per_token_cast_with_sf_only
    compiled = torch.compile(op, backend='npugraph_ex', fullgraph=True, dynamic=False)
    default = get_amax_clamp_for_quant('e4m3')
    try:
        for clamp in (default, 1e6, default):
            set_amax_clamp_for_quant('e4m3', clamp)
            reference = cast(x, 'e4m3', (1, 32))
            result = compiled(x, 'e4m3', 32)
            assert_equal(result, reference[1])
            del result
    finally:
        set_amax_clamp_for_quant('e4m3', default)


@torch.inference_mode()
def test_per_token_cast_precomputed_compile_scale_updates():
    x = torch.randn((31, 512), dtype=torch.bfloat16, device=get_device())
    scale = cast(x, 'e4m3', (1, 32))[1]
    compiled = torch.compile(
        cann_ops_tilelang.per_token_cast_with_precomputed_sf, backend='npugraph_ex', fullgraph=True, dynamic=False
    )
    for step in range(3):
        if step == 1:
            scale.mul_(2)
        elif step == 2:
            # Keep provided scales in range: reducing below amax/448 can
            # overflow FP8 and exercises kernel saturation rather than replay.
            scale = scale.clone() * 2
        before = scale.clone()
        expected = cast(x, 'e4m3', (1, 32), sf=scale)
        result = compiled(x, 'e4m3', 32, scale)
        assert_equal(result, expected)
        assert_equal(scale, before)
        del result
