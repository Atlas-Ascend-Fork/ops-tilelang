import pytest
import torch

from cann_ops_tilelang import expand_to_mhc_backward as expand_to_mhc_bwd
from cann_ops_tilelang import expand_to_mhc_forward as expand_to_mhc_fwd
from cann_ops_tilelang.config import get_device
from tests.mhc.mhc_ref import expand_to_mhc_ref
from tests.testing.numeric import count_bytes

pytestmark = pytest.mark.npu


def generate_expand_test_data(
    n0: int, n1: int, mhc_mult: int, h: int, device: str | None = None
) -> dict[str, torch.Tensor]:
    device = get_device() if device is None else device
    x = torch.randn(n0, n1, h, dtype=torch.bfloat16, device=device)
    o_grad = torch.randn(n0, n1, mhc_mult, h, dtype=torch.bfloat16, device=device)

    return {'x': x, 'o_grad': o_grad, 'mhc_mult': mhc_mult}


def _run_expand_fwd(x: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    out = x.new_empty(*x.shape[:-1], mhc_mult, x.shape[-1])
    assert x.is_contiguous()
    expand_to_mhc_fwd(x.flatten(0, -2), out.flatten(0, -3))
    return out


def _run_expand_bwd(o_grad: torch.Tensor) -> torch.Tensor:
    x_grad = o_grad.new_empty(*o_grad.shape[:-2], o_grad.shape[-1])
    expand_to_mhc_bwd(o_grad.flatten(0, -3), x_grad.flatten(0, -2))
    return x_grad


@pytest.mark.parametrize('n0', [1, 2])
@pytest.mark.parametrize('n1', [1024, 4096])
@pytest.mark.parametrize('mhc_mult', [2, 4, 8])
@pytest.mark.parametrize('h', [1280, 2560, 5120, 7168])
def test_expand_comprehensive(n0: int, n1: int, mhc_mult: int, h: int) -> None:
    test_data = generate_expand_test_data(n0=n0, n1=n1, mhc_mult=mhc_mult, h=h)

    out_tl = _run_expand_fwd(test_data['x'], test_data['mhc_mult'])
    out_ref = expand_to_mhc_ref(test_data['x'], test_data['mhc_mult'])
    torch.testing.assert_close(out_tl, out_ref)

    x_grad_tl = _run_expand_bwd(test_data['o_grad'])

    x_ref = test_data['x'].clone().requires_grad_()
    out_ref = expand_to_mhc_ref(x_ref, test_data['mhc_mult'])
    torch.autograd.backward([out_ref], [test_data['o_grad']])
    assert x_ref.grad is not None

    torch.testing.assert_close(x_grad_tl, x_ref.grad)


def test_expand_short_ragged_tokens() -> None:
    test_expand_comprehensive(n0=1, n1=129, mhc_mult=4, h=7168)


_EXPAND_BENCH_CASES = [
    (1, 512, 4, 7168),
    (1, 512, 4, 5120),
    (1, 512, 4, 4096),
    (1, 8192, 4, 7168),
    (1, 8192, 4, 5120),
    (1, 8192, 4, 4096),
]


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc_mult,h', _EXPAND_BENCH_CASES)
def test_expand_fwd_benchmark(
    n0: int,
    n1: int,
    mhc_mult: int,
    h: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    num_tokens = n0 * n1
    device = get_device()
    x = torch.randn(num_tokens, h, dtype=torch.bfloat16, device=device)
    o = torch.empty(num_tokens, mhc_mult, h, dtype=torch.bfloat16, device=device)

    io_gb = count_bytes(x, o) / 1e9

    t_us = benchmark_timer(lambda: expand_to_mhc_fwd(x, o))

    benchmark_record(
        kernel='expand_to_mhc_forward',
        operation='expand_forward',
        params={'n0': n0, 'n1': n1, 'mhc_mult': mhc_mult, 'h': h},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb},
    )


@pytest.mark.benchmark
@pytest.mark.parametrize('n0,n1,mhc_mult,h', _EXPAND_BENCH_CASES)
def test_expand_bwd_benchmark(
    n0: int,
    n1: int,
    mhc_mult: int,
    h: int,
    benchmark_timer,
    benchmark_record,
) -> None:
    num_tokens = n0 * n1
    device = get_device()
    o_grad = torch.randn(num_tokens, mhc_mult, h, dtype=torch.bfloat16, device=device)
    x_grad = torch.empty(num_tokens, h, dtype=torch.bfloat16, device=device)

    io_gb = count_bytes(o_grad, x_grad) / 1e9

    t_us = benchmark_timer(lambda: expand_to_mhc_bwd(o_grad, x_grad))

    benchmark_record(
        kernel='expand_to_mhc_backward',
        operation='expand_backward',
        params={'n0': n0, 'n1': n1, 'mhc_mult': mhc_mult, 'h': h},
        time_us=t_us,
        bandwidth_gbs=io_gb / t_us * 1e6,
        extras={'io_gb': io_gb},
    )
