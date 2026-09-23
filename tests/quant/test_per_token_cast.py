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
