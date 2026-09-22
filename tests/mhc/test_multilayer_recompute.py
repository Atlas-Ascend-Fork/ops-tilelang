import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device

pytestmark = pytest.mark.npu

_CORRECTNESS_CASES = [
    (1, 1, 2560),
    (3, 2, 2560),
    (3, 3, 2560),
    (10, 9, 2560),
    (10, 10, 2560),
    (10, 9, 4096),
    (10, 10, 4096),
    (10, 9, 5120),
    (10, 10, 5120),
    (10, 9, 7168),
    (10, 10, 7168),
    (10, 9, 8192),
    (10, 10, 8192),
]

_BENCH_CASES = [
    (10, 9, 1, 8192, 4, 2560),
    (10, 10, 1, 8192, 4, 2560),
    (10, 9, 1, 8192, 4, 4096),
    (10, 10, 1, 8192, 4, 4096),
    (10, 9, 1, 8192, 4, 5120),
    (10, 10, 1, 8192, 4, 5120),
    (10, 9, 1, 8192, 4, 7168),
    (10, 10, 1, 8192, 4, 7168),
    (10, 9, 1, 8192, 4, 8192),
    (10, 10, 1, 8192, 4, 8192),
]


def generate_multilayer_recompute_test_data(
    bs: int,
    seq: int,
    mhc_mult: int,
    hidden: int,
    num_layers: int,
    num_post: int,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    device = get_device()
    initial_residual = torch.randn((bs, seq, mhc_mult, hidden), device=device, dtype=torch.bfloat16)
    pre_mix_list = [torch.randn((bs, seq, mhc_mult, 1), device=device, dtype=torch.float32) for _ in range(num_layers)]
    layer_output_list = [torch.randn((bs, seq, hidden), device=device, dtype=torch.bfloat16) for _ in range(num_post)]
    post_mix_list = [torch.randn((bs, seq, mhc_mult, 1), device=device, dtype=torch.float32) for _ in range(num_post)]
    comb_mix_list = [
        torch.randn((bs, seq, mhc_mult, mhc_mult), device=device, dtype=torch.float32) for _ in range(num_post)
    ]
    layer_input_list = [torch.empty((bs, seq, hidden), device=device, dtype=torch.bfloat16) for _ in range(num_layers)]
    residual_list = [
        torch.empty((bs, seq, mhc_mult, hidden), device=device, dtype=torch.bfloat16) for _ in range(num_post)
    ]
    return (
        initial_residual,
        pre_mix_list,
        layer_output_list,
        post_mix_list,
        comb_mix_list,
        layer_input_list,
        residual_list,
    )


def _mhc_multilayer_recompute_ref(
    initial_residual: torch.Tensor,
    pre_mix_list: list[torch.Tensor],
    layer_output_list: list[torch.Tensor],
    post_mix_list: list[torch.Tensor],
    comb_mix_list: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    layer_input_refs: list[torch.Tensor] = []
    residual_refs: list[torch.Tensor] = []
    residual = initial_residual
    mhc_mult = initial_residual.shape[-2]
    hidden = initial_residual.shape[-1]
    num_tokens = initial_residual.numel() // (mhc_mult * hidden)
    for index in range(len(pre_mix_list)):
        layer_input = torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=residual.device)
        cann_ops_tilelang.mhc_pre_apply_mix_forward(
            residual.view(num_tokens, mhc_mult, hidden),
            pre_mix_list[index].view(num_tokens, mhc_mult),
            layer_input,
        )
        layer_input = layer_input.view(*residual.shape[:-2], hidden)
        layer_input_refs.append(layer_input)
        if index < len(layer_output_list):
            residual = cann_ops_tilelang.mhc_post_forward(
                layer_output_list[index],
                residual,
                post_mix_list[index],
                comb_mix_list[index],
            )
            residual_refs.append(residual)
    return layer_input_refs, residual_refs


def _compute_io_bytes(
    n: int,
    mhc_mult: int,
    hidden: int,
    num_layers: int,
    num_post: int,
) -> tuple[int, int]:
    io_pre = n * mhc_mult * hidden * 2 + n * mhc_mult * 4 + n * hidden * 2
    io_post = (
        n * hidden * 2
        + n * mhc_mult * hidden * 2
        + n * mhc_mult * 4
        + n * mhc_mult * mhc_mult * 4
        + n * mhc_mult * hidden * 2
    )
    io_ref = num_layers * io_pre + num_post * io_post
    io_fused = (
        n * mhc_mult * hidden * 2
        + num_layers * (n * mhc_mult * 4 + n * hidden * 2)
        + num_post * (n * hidden * 2 + n * mhc_mult * 4 + n * mhc_mult * mhc_mult * 4 + n * mhc_mult * hidden * 2)
    )
    return io_ref, io_fused


@pytest.mark.parametrize('num_layers,num_post,hidden', _CORRECTNESS_CASES)
def test_mhc_multilayer_recompute_correctness(num_layers: int, num_post: int, hidden: int) -> None:
    torch.manual_seed(0)
    (
        initial_residual,
        pre_mix_list,
        layer_output_list,
        post_mix_list,
        comb_mix_list,
        layer_input_list,
        residual_list,
    ) = generate_multilayer_recompute_test_data(1, 8192, 4, hidden, num_layers, num_post)
    layer_input_ref, residual_ref = _mhc_multilayer_recompute_ref(
        initial_residual,
        pre_mix_list,
        layer_output_list,
        post_mix_list,
        comb_mix_list,
    )
    cann_ops_tilelang.mhc_multilayer_recompute(
        initial_residual,
        pre_mix_list,
        layer_output_list,
        post_mix_list,
        comb_mix_list,
        layer_input_list,
        residual_list,
    )

    for index in range(num_layers):
        assert torch.equal(layer_input_list[index], layer_input_ref[index]), (
            f'layer_input[{index}] mismatch! max diff = '
            f'{(layer_input_list[index].float() - layer_input_ref[index].float()).abs().max().item()}'
        )
    for index in range(num_post):
        assert torch.equal(residual_list[index], residual_ref[index]), (
            f'residual[{index}] mismatch! max diff = '
            f'{(residual_list[index].float() - residual_ref[index].float()).abs().max().item()}'
        )


@pytest.mark.benchmark
@pytest.mark.parametrize('num_layers,num_post,bs,seq,mhc_mult,hidden', _BENCH_CASES)
def test_mhc_multilayer_recompute_benchmark(
    num_layers: int,
    num_post: int,
    bs: int,
    seq: int,
    mhc_mult: int,
    hidden: int,
    benchmark_record,
    benchmark_timer,
) -> None:
    torch.manual_seed(0)
    n = bs * seq
    io_ref, io_fused = _compute_io_bytes(n, mhc_mult, hidden, num_layers, num_post)
    theory = io_ref / io_fused
    (
        initial_residual,
        pre_mix_list,
        layer_output_list,
        post_mix_list,
        comb_mix_list,
        layer_input_list,
        residual_list,
    ) = generate_multilayer_recompute_test_data(bs, seq, mhc_mult, hidden, num_layers, num_post)

    def fn_ref() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return _mhc_multilayer_recompute_ref(
            initial_residual,
            pre_mix_list,
            layer_output_list,
            post_mix_list,
            comb_mix_list,
        )

    def fn_fused() -> None:
        cann_ops_tilelang.mhc_multilayer_recompute(
            initial_residual,
            pre_mix_list,
            layer_output_list,
            post_mix_list,
            comb_mix_list,
            layer_input_list,
            residual_list,
        )

    fn_ref()
    fn_fused()
    t_ref_us = benchmark_timer(fn_ref)
    t_fused_us = benchmark_timer(fn_fused)
    speedup = t_ref_us / t_fused_us
    bw_ref_gbs = io_ref / t_ref_us / 1e3
    bw_fused_gbs = io_fused / t_fused_us / 1e3

    benchmark_record(
        kernel='mhc_multilayer_recompute',
        operation='recompute',
        params={
            'num_layers': num_layers,
            'num_post': num_post,
            'bs': bs,
            'seq': seq,
            'mhc_mult': mhc_mult,
            'hidden': hidden,
        },
        time_us=t_fused_us,
        bandwidth_gbs=bw_fused_gbs,
        extras={
            'ref_time_us': t_ref_us,
            'speedup': speedup,
            'bw_ref_gbs': bw_ref_gbs,
            'theory': theory,
            'efficiency': speedup / theory,
        },
    )
