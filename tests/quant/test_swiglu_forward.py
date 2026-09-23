import os

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device, set_num_ai_cores, set_token_alignment
from tests.quant.cast_ref import cast
from tests.quant.swiglu_ref import swiglu_forward
from tests.testing.bench import get_cast_params, make_param_id
from tests.testing.generator import (
    generate_cast_config,
    generate_hidden_sizes,
    generate_moe_params,
    generate_num_sms,
    generate_psum_layout,
    generate_samples,
    get_test_level,
)
from tests.testing.numeric import assert_equal, count_bytes
from tests.testing.quant import clear_unused_sf

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'
pytestmark = pytest.mark.npu


def generate_test_data(params):
    hidden = params['hidden']
    mask, psum_num_tokens_per_expert = generate_psum_layout(params)
    num_expanded_tokens = mask.shape[0]
    topk_weights = torch.randn((num_expanded_tokens,), dtype=torch.float32, device=get_device())
    x = torch.randn((num_expanded_tokens, hidden * 2), dtype=torch.bfloat16, device=get_device())
    _clamped_count = torch.zeros(4, dtype=torch.int64, device=get_device())

    return (x, mask, psum_num_tokens_per_expert, topk_weights, _clamped_count)


def generate_test_params(level: int) -> list[dict]:
    params = [
        {
            **moe,
            **args,
            'fmt': fmt,
            'hidden': hidden_size,
            'with_psum_layout': with_psum_layout,
            'num_per_channels': num_per_channels,
            'use_tma_aligned_col_major_sf': use_tma_aligned_col_major_sf,
            'round_sf': round_sf,
            'use_packed_ue8m0': use_packed_ue8m0,
            'routed_scaling_factor': routed_scaling_factor,
        }
        for moe in generate_moe_params(level)
        for fmt in ('bf16', 'fp32', 'e4m3', 'e2m1')
        for num_per_channels in ((128, 32) if fmt in ('e4m3', 'e2m1') else (None,))
        for hidden_size in [h // 2 for h in ([4096, 7168] if level == 0 else generate_hidden_sizes(64 * 2))]
        for use_tma_aligned_col_major_sf, round_sf, use_packed_ue8m0 in generate_cast_config(level, fmt)
        for with_psum_layout in ((True,) if level == 0 else (True, False))
        for args in generate_samples(
            level,
            clamp_value=(None, 0.5),
            num_sms=generate_num_sms(level),
            with_weights=(True, False),
            alignment=(256,),
        )
        for routed_scaling_factor in (((2.5,) if level == 0 else (2.5, None)) if args['with_weights'] else (None,))
    ]
    return params


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_swiglu_forward_and_per_token_cast(params):
    hidden = params['hidden']
    fmt = params['fmt']
    with_weights = params['with_weights']
    num_per_channels = params['num_per_channels']
    with_psum_layout = params['with_psum_layout']
    use_packed_ue8m0 = params['use_packed_ue8m0']
    clamp_value = params['clamp_value']
    routed_scaling_factor = params['routed_scaling_factor']

    set_token_alignment(params['alignment'])
    set_num_ai_cores(params['num_sms'])
    x, mask, psum_num_tokens_per_expert, post_weights, _clamped_count = generate_test_data(params)

    do_clamp_count = clamp_value is not None
    clamped_count_ref = _clamped_count.clone() if do_clamp_count else None
    clamped_count = _clamped_count.clone() if do_clamp_count else None
    args = dict(
        x=x,
        fmt=fmt,
        num_per_channels=num_per_channels,
        psum_num_tokens_per_expert=psum_num_tokens_per_expert if with_psum_layout else None,
        clamp_value=clamp_value,
        topk_weights=post_weights if with_weights else None,
        routed_scaling_factor=routed_scaling_factor,
        **get_cast_params(params),
    )

    def func_ref():
        out = swiglu_forward(
            x,
            psum_num_tokens_per_expert if with_psum_layout else None,
            post_weights if with_weights else None,
            routed_scaling_factor,
            clamp_value,
            clamped_count_ref,
        )
        if fmt in ('e4m3', 'e2m1'):
            out = cast(out, fmt, (1, num_per_channels), **get_cast_params(params))
        elif fmt == 'bf16':
            out = out.bfloat16()
        return out

    out_ref = func_ref()

    if fmt in ('e4m3', 'e2m1'):
        out = cann_ops_tilelang.swiglu_forward_and_per_token_cast(**args, clamped_count=clamped_count)
        x_quant_ref, x_sf_ref = out_ref
        x_quant, x_sf = out
        if use_packed_ue8m0:
            x_sf_ref = clear_unused_sf(x_sf_ref, hidden, num_per_channels).to(torch.int32)
            x_sf = clear_unused_sf(x_sf, hidden, num_per_channels).to(torch.int32)
        if with_psum_layout:
            x_quant_ref = x_quant_ref.float().masked_fill(~mask.unsqueeze(1), 0)
            x_sf_ref = x_sf_ref.masked_fill(~mask.unsqueeze(1), 0)
            x_quant = x_quant.float().masked_fill(~mask.unsqueeze(1), 0)
            x_sf = x_sf.masked_fill(~mask.unsqueeze(1), 0)
        assert_equal(x_sf, x_sf_ref)
        assert_equal(x_quant, x_quant_ref)
    else:
        out = cann_ops_tilelang.swiglu_forward(
            x=x,
            fmt=fmt,
            psum_num_tokens_per_expert=psum_num_tokens_per_expert if with_psum_layout else None,
            topk_weights=post_weights if with_weights else None,
            routed_scaling_factor=routed_scaling_factor,
            clamp_value=clamp_value,
            clamped_count=clamped_count,
        )
        if with_psum_layout:
            out_ref = out_ref.masked_fill(~mask.unsqueeze(1), 0)
            out = out.masked_fill(~mask.unsqueeze(1), 0)
        assert_equal(out, out_ref)

    if do_clamp_count:
        assert_equal(clamped_count, clamped_count_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_swiglu_forward_and_per_token_cast_benchmark(benchmark_timer, benchmark_record, params):
    with_psum_layout = params['with_psum_layout']
    with_weights = params['with_weights']
    num_per_channels = params['num_per_channels']
    clamp_value = params['clamp_value']
    fmt = params['fmt']
    routed_scaling_factor = params['routed_scaling_factor']

    set_token_alignment(params['alignment'])
    x, mask, psum_num_tokens_per_expert, post_weights, _clamped_count = generate_test_data(params)

    do_clamp_count = clamp_value is not None
    clamped_count = _clamped_count.clone() if do_clamp_count else None
    if fmt in ('e4m3', 'e2m1'):
        args = dict(
            x=x,
            fmt=fmt,
            num_per_channels=num_per_channels,
            psum_num_tokens_per_expert=psum_num_tokens_per_expert if with_psum_layout else None,
            clamp_value=clamp_value,
            topk_weights=post_weights if with_weights else None,
            routed_scaling_factor=routed_scaling_factor,
            **get_cast_params(params),
        )

        def func():
            return cann_ops_tilelang.swiglu_forward_and_per_token_cast(**args, clamped_count=clamped_count)

    else:

        def func():
            return cann_ops_tilelang.swiglu_forward(
                x=x,
                fmt=fmt,
                psum_num_tokens_per_expert=psum_num_tokens_per_expert if with_psum_layout else None,
                topk_weights=post_weights if with_weights else None,
                routed_scaling_factor=routed_scaling_factor,
                clamp_value=clamp_value,
                clamped_count=clamped_count,
            )

    out = func()

    t_us = benchmark_timer(func)
    if isinstance(out, torch.Tensor):
        num_bytes = count_bytes(x[mask], out[mask])
    elif isinstance(out, tuple):
        assert len(out) == 2
        num_bytes = count_bytes(x[mask], (out[0][mask], out[1][mask]))
    else:
        raise NotImplementedError('Unsupported out type')

    benchmark_record(
        kernel='swiglu_forward_and_per_token_cast',
        operation='fwd',
        params=params,
        time_us=t_us,
        bandwidth_gbs=num_bytes / t_us / 1e3,
    )
