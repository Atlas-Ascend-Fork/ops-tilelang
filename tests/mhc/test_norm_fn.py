from contextlib import contextmanager

import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device
from tests.mhc.mhc_ref import mhc_pre_norm_fn_partials_ref, mhc_pre_norm_fn_ref
from tests.testing.numeric import count_bytes

pytestmark = pytest.mark.npu


@contextmanager
def _reference_matmul_precision():
    old_allow_hf32 = torch.npu.matmul.allow_hf32
    torch.npu.matmul.allow_hf32 = True
    try:
        yield
    finally:
        torch.npu.matmul.allow_hf32 = old_allow_hf32


def generate_norm_fn_test_data(
    n1: int,
    mhc_mult: int,
    hidden_size: int,
    generate_normw: bool,
) -> dict[str, torch.Tensor]:
    n0 = 1
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    device = get_device()

    residual = (
        torch.randn((n0, n1, mhc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, 1, -1, 1))
        .bfloat16()
    )

    fn = (
        torch.randn((mhc_mult3, mhc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(mhc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)

    normw = torch.randn((mhc_hidden_size,), dtype=torch.float, device=device) * 0.1 + 1.0 if generate_normw else None

    out_grad = torch.randn((n0, n1, mhc_mult3), dtype=torch.float, device=device)

    return {
        'residual': residual,
        'fn': fn,
        'normw': normw,
        'out_grad': out_grad,
        'mhc_norm_eps': 1e-6,
    }


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    # Keep the reduction on the accelerator. This form is algebraically
    # equivalent to 1 - 2 * <x, y> / (||x||^2 + ||y||^2), but avoids the
    # cancellation in '1 - sim' and is therefore accurate enough in fp32.
    x, y = x.float(), y.float()
    delta = x - y
    numerator = (delta * delta).sum()
    denominator = (x * x + y * y).sum()
    diff = torch.where(denominator == 0, 0.0, numerator / denominator)
    return float(diff)


def _assert_calc_diff_close(x: torch.Tensor, y: torch.Tensor, tol: float = 1e-8) -> None:
    diff = _calc_diff(x, y)
    assert diff < tol, f'calc_diff={diff:.10g} exceeds tolerance {tol}'


def _ref_reduce_rmsnorm(
    out_mul: torch.Tensor,
    sqrsum: torch.Tensor,
    mhc_hidden_size: int,
    eps: float,
) -> torch.Tensor:
    """Torch tail of mhc_pre_norm_fn_ref: rms-normalize and reduce over the RMS group."""
    rms = (sqrsum / mhc_hidden_size + eps).rsqrt()
    return (out_mul * rms.unsqueeze(-1)).sum(-2)


@pytest.mark.parametrize('n1', [4096, 8192])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 5120, 7168])
@pytest.mark.parametrize('generate_normw', [False, True])
def test_correctness(
    n1: int,
    hidden_size: int,
    generate_normw: bool,
) -> None:
    """Cover the fn x normw merge and the reduce/rmsnorm operators (fwd + bwd).

    The GEMM partials are built from the torch reference so this test isolates
    the merge and reduce/RMSNorm operators.
    """
    mhc_mult = 4
    mhc_hidden_size = mhc_mult * hidden_size
    mhc_mult3 = mhc_mult * (2 + mhc_mult)

    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=generate_normw,
    )
    residual = test_data['residual']
    fn = test_data['fn']
    normw = test_data['normw']
    out_grad = test_data['out_grad']
    mhc_norm_eps = test_data['mhc_norm_eps']
    num_tokens = residual.shape[0] * residual.shape[1]

    actual_n_splits = 1
    merged_fn = fn
    if normw is not None:
        # fn x normw merge, compared against the merge that mhc_pre_norm_fn_ref
        # performs in torch before its GEMM.
        merged_fn = torch.empty_like(fn)
        cann_ops_tilelang.mhc_fn_normw_merge_forward(fn, normw, merged_fn)
        _assert_calc_diff_close(merged_fn, fn * normw)

        merged_fn_grad = torch.randn_like(fn)
        fn_grad = torch.zeros_like(fn)
        normw_grad = torch.zeros_like(normw)
        cann_ops_tilelang.mhc_fn_normw_merge_backward(fn, normw, merged_fn_grad, fn_grad, normw_grad)
        _assert_calc_diff_close(fn_grad, merged_fn_grad * normw)
        _assert_calc_diff_close(normw_grad, (merged_fn_grad * fn).sum(0))

    with _reference_matmul_precision():
        mixes_ref = mhc_pre_norm_fn_ref(residual, fn, normw, mhc_norm_eps)
        gemm_out_mul, gemm_out_sqrsum = mhc_pre_norm_fn_partials_ref(residual, merged_fn, actual_n_splits)

    out_mul = torch.empty(num_tokens, 1, mhc_mult3, dtype=torch.float32, device=residual.device)
    sqrsum = torch.empty(num_tokens, 1, dtype=torch.float32, device=residual.device)
    mixes_tl = torch.empty(num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device)
    cann_ops_tilelang.mhc_reduce_partials_and_rmsnorm_forward(
        gemm_out_mul,
        gemm_out_sqrsum,
        out_mul,
        sqrsum,
        mixes_tl,
        rms_group_size=mhc_hidden_size,
        rms_eps=mhc_norm_eps,
        n_splits=actual_n_splits,
    )
    torch.testing.assert_close(mixes_tl.view_as(mixes_ref), mixes_ref, atol=1e-3, rtol=1e-3)

    # The reduce/rmsnorm backward only sees the fwd outputs: feed it the kernel
    # outputs and compare against autograd through the torch reference tail.
    out_mul_grad = torch.empty_like(out_mul)
    sqrsum_grad = torch.empty_like(sqrsum)
    cann_ops_tilelang.mhc_reduce_partials_and_rmsnorm_backward(
        out_grad.view(-1, mhc_mult3),
        out_mul,
        sqrsum,
        out_mul_grad,
        sqrsum_grad,
        rms_group_size=mhc_hidden_size,
        rms_eps=mhc_norm_eps,
    )

    out_mul_ref = out_mul.clone().requires_grad_()
    sqrsum_ref = sqrsum.clone().requires_grad_()
    mixes_ref_tail = _ref_reduce_rmsnorm(out_mul_ref, sqrsum_ref, mhc_hidden_size, mhc_norm_eps)
    torch.autograd.backward([mixes_ref_tail], [out_grad.view(-1, mhc_mult3)])
    assert out_mul_ref.grad is not None
    assert sqrsum_ref.grad is not None
    _assert_calc_diff_close(out_mul_grad, out_mul_ref.grad, tol=1e-8)
    _assert_calc_diff_close(sqrsum_grad, sqrsum_ref.grad, tol=1e-8)


def test_reduce_rmsnorm_token_tail() -> None:
    test_correctness(n1=129, hidden_size=1280, generate_normw=True)


def _ref_fwd_matmul(
    x: torch.Tensor,
    fn: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mhc_hidden_size = fn.shape[1]
    mhc_mult3 = fn.shape[0]
    x_flat = x.float().flatten(2, 3)
    fn_float = fn.float()
    out_mul = torch.einsum(
        'mbk,nbk->mbn',
        x_flat.view(-1, 1, mhc_hidden_size),
        fn_float.view(mhc_mult3, 1, mhc_hidden_size),
    )
    sqrsum = x_flat.view(-1, 1, mhc_hidden_size).square().sum(-1)
    return out_mul, sqrsum


@pytest.mark.parametrize('n1', [512, 4096])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096, 5120, 7168])
def test_gemm_forward(n1: int, hidden_size: int) -> None:
    mhc_mult = 4
    mhc_hidden_size = mhc_mult * hidden_size
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    device = get_device()

    residual = torch.randn(1, n1, mhc_mult, hidden_size, dtype=torch.float, device=device).bfloat16()
    fn = torch.randn(mhc_mult3, mhc_mult, hidden_size, dtype=torch.float, device=device).mul(1e-4).flatten(1, 2)
    out_mul_splitted = torch.empty(1, n1, 1, mhc_mult3, dtype=torch.float32, device=device)
    sqrsum_splitted = torch.empty(1, n1, 1, dtype=torch.float32, device=device)

    cann_ops_tilelang.mhc_gemm_with_sqrsum_forward(
        residual.view(n1, mhc_hidden_size),
        fn,
        out_mul_splitted,
        sqrsum_splitted,
        n_splits=16,
    )
    with _reference_matmul_precision():
        out_mul_ref, sqrsum_ref = _ref_fwd_matmul(residual, fn)

    _assert_calc_diff_close(out_mul_splitted.sum(0), out_mul_ref, tol=1e-8)
    _assert_calc_diff_close(sqrsum_splitted.sum(0), sqrsum_ref, tol=1e-8)


@pytest.mark.parametrize('n1', [13, 48])
def test_gemm_forward_short_tokens(n1: int) -> None:
    """Cover the split-K forward path below one 128-token block."""

    test_gemm_forward(n1=n1, hidden_size=1280)


@pytest.mark.parametrize('n1', [512, 4096])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096, 5120, 7168])
def test_gemm_and_reduce_forward(n1: int, hidden_size: int) -> None:
    mhc_mult = 4
    mhc_hidden_size = mhc_mult * hidden_size
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    eps = 1e-6
    device = get_device()

    residual = torch.randn(1, n1, mhc_mult, hidden_size, dtype=torch.float, device=device).bfloat16()
    fn = torch.randn(mhc_mult3, mhc_mult, hidden_size, dtype=torch.float, device=device).mul(1e-4).flatten(1, 2)
    out_mul_splitted = torch.empty(1, n1, 1, mhc_mult3, dtype=torch.float32, device=device)
    sqrsum_splitted = torch.empty(1, n1, 1, dtype=torch.float32, device=device)
    out_mul = torch.empty(n1, 1, mhc_mult3, dtype=torch.float32, device=device)
    sqrsum = torch.empty(n1, 1, dtype=torch.float32, device=device)
    out = torch.empty(n1, mhc_mult3, dtype=torch.float32, device=device)

    cann_ops_tilelang.mhc_gemm_with_sqrsum_forward(
        residual.view(n1, mhc_hidden_size),
        fn,
        out_mul_splitted,
        sqrsum_splitted,
        n_splits=16,
    )
    cann_ops_tilelang.mhc_reduce_partials_and_rmsnorm_forward(
        out_mul_splitted,
        sqrsum_splitted,
        out_mul,
        sqrsum,
        out,
        rms_group_size=mhc_hidden_size,
        rms_eps=eps,
        n_splits=1,
    )

    with _reference_matmul_precision():
        out_mul_ref, sqrsum_ref = _ref_fwd_matmul(residual, fn)
    out_ref = _ref_reduce_rmsnorm(out_mul_ref, sqrsum_ref, mhc_hidden_size, eps)
    torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-3)


def test_gemm_deterministic_reproducibility() -> None:
    """The deterministic split-K variant must be bitwise reproducible."""
    mhc_mult = 4
    hidden_size = 1280
    mhc_hidden_size = mhc_mult * hidden_size
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    device = get_device()

    residual = torch.randn(1, 512, mhc_mult, hidden_size, dtype=torch.float, device=device).bfloat16()
    x = residual.view(512, mhc_hidden_size)
    fn = torch.randn(mhc_mult3, mhc_mult, hidden_size, dtype=torch.float, device=device).mul(1e-4).flatten(1, 2)

    def run() -> tuple[torch.Tensor, torch.Tensor]:
        out_mul = torch.empty(1, 512, 1, mhc_mult3, dtype=torch.float32, device=device)
        sqrsum = torch.empty(1, 512, 1, dtype=torch.float32, device=device)
        cann_ops_tilelang.mhc_gemm_with_sqrsum_forward(x, fn, out_mul, sqrsum, n_splits=16)
        return out_mul, sqrsum

    old = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        out1_mul, out1_sq = run()
        out2_mul, out2_sq = run()
    finally:
        torch.use_deterministic_algorithms(old)

    assert torch.equal(out1_mul, out2_mul)
    assert torch.equal(out1_sq, out2_sq)


@pytest.mark.parametrize('n1', [512, 4096])
@pytest.mark.parametrize('hidden_size', [1280, 2560, 4096, 5120, 7168])
def test_gemm_backward(n1: int, hidden_size: int) -> None:
    mhc_mult = 4
    mhc_hidden_size = mhc_mult * hidden_size
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    device = get_device()

    residual = torch.randn(1, n1, mhc_mult, hidden_size, dtype=torch.float, device=device).bfloat16()
    fn = torch.randn(mhc_mult3, mhc_mult, hidden_size, dtype=torch.float, device=device).mul(1e-4).flatten(1, 2)
    out_mul_grad = torch.randn(n1, 1, mhc_mult3, dtype=torch.float32, device=device)
    sqrsum_grad = torch.randn(n1, 1, dtype=torch.float32, device=device).mul_(1e-4)
    x_grad = torch.zeros(n1, mhc_hidden_size, dtype=torch.bfloat16, device=device)
    fn_grad = torch.empty_like(fn)

    cann_ops_tilelang.mhc_gemm_with_sqrsum_backward(
        out_mul_grad,
        sqrsum_grad,
        residual.view(n1, mhc_hidden_size),
        fn,
        x_grad,
        fn_grad,
    )

    residual_ref = residual.clone().requires_grad_()
    fn_ref = fn.clone().requires_grad_()
    with _reference_matmul_precision():
        out_mul_ref, sqrsum_ref = _ref_fwd_matmul(residual_ref, fn_ref)
        torch.autograd.backward([out_mul_ref, sqrsum_ref], [out_mul_grad, sqrsum_grad])
    assert residual_ref.grad is not None
    assert fn_ref.grad is not None
    _assert_calc_diff_close(x_grad.view_as(residual_ref), residual_ref.grad, tol=2e-9)
    _assert_calc_diff_close(fn_grad, fn_ref.grad, tol=1e-9)


def test_gemm_backward_token_tail() -> None:
    test_gemm_backward(n1=129, hidden_size=1280)


def test_gemm_backward_accumulates_existing_x_grad() -> None:
    """The backward kernel must treat x_grad as an in/out buffer."""
    torch.manual_seed(20260728)
    device = get_device()
    num_tokens, mhc_mult, hidden_size = 128, 4, 128
    mhc_hidden_size = mhc_mult * hidden_size
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    x = torch.randn(num_tokens, mhc_hidden_size, dtype=torch.float32, device=device).bfloat16()
    fn = torch.randn(mhc_mult3, mhc_hidden_size, dtype=torch.float32, device=device).mul_(1e-3)
    out_mul_grad = torch.randn(num_tokens, 1, mhc_mult3, dtype=torch.float32, device=device)
    sqrsum_grad = torch.randn(num_tokens, 1, dtype=torch.float32, device=device).mul_(1e-4)

    def run(seed: float) -> tuple[torch.Tensor, torch.Tensor]:
        x_grad = torch.full_like(x, seed)
        fn_grad = torch.empty_like(fn)
        cann_ops_tilelang.mhc_gemm_with_sqrsum_backward(out_mul_grad, sqrsum_grad, x, fn, x_grad, fn_grad)
        return x_grad, fn_grad

    zero_grad, zero_fn_grad = run(0.0)
    plus_grad, plus_fn_grad = run(2.0)
    minus_grad, minus_fn_grad = run(-2.0)
    plus_expected = (zero_grad.float() + 2.0).bfloat16()
    minus_expected = (zero_grad.float() - 2.0).bfloat16()

    assert not torch.equal(plus_grad, zero_grad), 'existing x_grad was overwritten'
    torch.testing.assert_close(plus_grad, plus_expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(minus_grad, minus_expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(plus_fn_grad, zero_fn_grad, atol=0, rtol=0)
    torch.testing.assert_close(minus_fn_grad, zero_fn_grad, atol=0, rtol=0)


_NORMW_MERGE_BENCH_CASES = [5120, 10240, 16384, 20480, 28672]


@pytest.mark.benchmark
@pytest.mark.parametrize('hidden', _NORMW_MERGE_BENCH_CASES)
def test_fn_normw_merge_fwd_benchmark(
    hidden: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    device = get_device()
    mhc_mult3 = 24
    fn = torch.randn(mhc_mult3, hidden, dtype=torch.float32, device=device)
    normw = torch.randn(hidden, dtype=torch.float32, device=device)
    out_fn = torch.empty_like(fn)

    def _run():
        cann_ops_tilelang.mhc_fn_normw_merge_forward(fn, normw, out_fn)

    _run()
    t_us = benchmark_timer(_run)
    io_gb = count_bytes(fn, normw, out_fn) / 1e9
    benchmark_record(
        kernel='mhc_fn_normw_merge_forward',
        operation='normw_merge_forward',
        params={'hidden': hidden},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'mhc_mult3': mhc_mult3},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('hidden', _NORMW_MERGE_BENCH_CASES)
def test_fn_normw_merge_bwd_benchmark(
    hidden: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    device = get_device()
    mhc_mult3 = 24
    fn = torch.randn(mhc_mult3, hidden, dtype=torch.float32, device=device)
    normw = torch.randn(hidden, dtype=torch.float32, device=device)
    out_fn_grad = torch.randn_like(fn)
    fn_grad = torch.randn_like(fn)
    normw_grad = torch.randn_like(normw)

    def _run():
        cann_ops_tilelang.mhc_fn_normw_merge_backward(
            fn,
            normw,
            out_fn_grad,
            fn_grad,
            normw_grad,
        )

    _run()
    t_us = benchmark_timer(_run)
    io_gb = (
        count_bytes(
            fn,
            normw,
            out_fn_grad,
            fn_grad,
            fn_grad,
            normw_grad,
            normw_grad,
        )
        / 1e9
    )
    benchmark_record(
        kernel='mhc_fn_normw_merge_backward',
        operation='normw_merge_backward',
        params={'hidden': hidden},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'mhc_mult3': mhc_mult3, 'grad_buffers_inout': True},
    )


_NORM_FN_BENCH_CASES = [
    (512, 7168),
    (512, 5120),
    (512, 4096),
    (8192, 7168),
    (8192, 5120),
    (8192, 4096),
]

_NORM_FN_FWD_MNK_BENCH_CASES = [
    (4096, 24, 28672),
    (4096, 24, 20480),
    (8192, 24, 28672),
    (8192, 24, 20480),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('n1,hidden_size', _NORM_FN_BENCH_CASES)
def test_gemm_forward_benchmark(
    n1: int,
    hidden_size: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    mhc_mult = 4
    n_splits = 16
    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=True,
    )
    residual = test_data['residual']
    fn = test_data['fn']
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    num_tokens = residual.shape[0] * residual.shape[1]
    x = residual.view(-1, mhc_hidden_size)
    out_mul_splitted = torch.empty(1, num_tokens, 1, mhc_mult3, dtype=torch.float32, device=x.device)
    sqrsum_splitted = torch.empty(1, num_tokens, 1, dtype=torch.float32, device=x.device)
    io_gb = count_bytes(x, fn, out_mul_splitted, sqrsum_splitted) / 1e9

    def _run():
        cann_ops_tilelang.mhc_gemm_with_sqrsum_forward(
            x,
            fn,
            out_mul_splitted,
            sqrsum_splitted,
            n_splits=n_splits,
        )

    _run()
    t_us = benchmark_timer(_run)
    benchmark_record(
        kernel='mhc_gemm_with_sqrsum_forward',
        operation='norm_fn_forward',
        params={
            'n1': n1,
            'hidden_size': hidden_size,
            'mhc_mult': mhc_mult,
            'mhc_hidden_size': mhc_hidden_size,
            'mhc_mult3': mhc_mult3,
        },
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'num_tokens': num_tokens, 'n_splits': n_splits},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('m,n,k', _NORM_FN_FWD_MNK_BENCH_CASES)
def test_gemm_forward_mnk_benchmark(
    m: int,
    n: int,
    k: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    n_splits = 16
    device = get_device()
    x = torch.randn(m, k, dtype=torch.float32, device=device).bfloat16()
    fn = torch.randn(n, k, dtype=torch.float32, device=device).mul_(1e-4)
    out_mul_splitted = torch.empty(1, m, 1, n, dtype=torch.float32, device=device)
    sqrsum_splitted = torch.empty(1, m, 1, dtype=torch.float32, device=device)
    io_gb = count_bytes(x, fn, out_mul_splitted, sqrsum_splitted) / 1e9

    def _run():
        cann_ops_tilelang.mhc_gemm_with_sqrsum_forward(
            x,
            fn,
            out_mul_splitted,
            sqrsum_splitted,
            n_splits=n_splits,
        )

    _run()
    t_us = benchmark_timer(_run)
    benchmark_record(
        kernel='mhc_gemm_with_sqrsum_forward',
        operation='norm_fn_forward_mnk',
        params={'m': m, 'n': n, 'k': k},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'n_splits': n_splits, 'actual_n_splits': 1},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n1,hidden_size', _NORM_FN_BENCH_CASES)
def test_gemm_backward_benchmark(
    n1: int,
    hidden_size: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    mhc_mult = 4
    test_data = generate_norm_fn_test_data(
        n1=n1,
        mhc_mult=mhc_mult,
        hidden_size=hidden_size,
        generate_normw=True,
    )
    residual = test_data['residual']
    fn = test_data['fn']
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    mhc_hidden_size = mhc_mult * hidden_size
    num_tokens = residual.shape[0] * residual.shape[1]
    out_mul_grad = test_data['out_grad'].view(-1, mhc_mult3)
    sqrsum_grad = torch.randn(num_tokens, dtype=torch.float32, device=residual.device)
    x = residual.view(-1, mhc_hidden_size)
    x_grad = torch.randn_like(x)
    fn_grad = torch.empty_like(fn)
    io_gb = (
        count_bytes(
            out_mul_grad,
            sqrsum_grad,
            x,
            fn,
            x_grad,
            x_grad,
            fn_grad,
        )
        / 1e9
    )

    def _run():
        cann_ops_tilelang.mhc_gemm_with_sqrsum_backward(
            out_mul_grad.view(-1, 1, mhc_mult3),
            sqrsum_grad.view(-1, 1),
            x,
            fn,
            x_grad,
            fn_grad,
        )

    _run()
    t_us = benchmark_timer(_run)
    benchmark_record(
        kernel='mhc_gemm_with_sqrsum_backward',
        operation='norm_fn_backward',
        params={
            'n1': n1,
            'hidden_size': hidden_size,
            'mhc_mult': mhc_mult,
            'mhc_hidden_size': mhc_hidden_size,
            'mhc_mult3': mhc_mult3,
        },
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb, 'num_tokens': num_tokens, 'x_grad_inout': True},
    )
