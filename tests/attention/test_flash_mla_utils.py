from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.attention.flash_mla_utils import convert_kv_index
from cann_ops_tilelang.config import get_device

pytestmark = pytest.mark.npu


def convert_kv_index_ref(
    topk_idxs: torch.Tensor, block_size: int, context_lens: torch.Tensor, block_tables: torch.Tensor
) -> None:
    """PyTorch reference for the in-place logical-to-physical index conversion."""
    _, s, k = topk_idxs.shape
    valid = torch.arange(k, device=topk_idxs.device, dtype=torch.int32).view(1, 1, k) < context_lens.unsqueeze(-1)
    safe_topk_idxs = torch.where(valid, topk_idxs, 0)
    logical_blocks = torch.div(safe_topk_idxs, block_size, rounding_mode='floor').long()
    physical_blocks = torch.gather(block_tables[:, None, :].expand(-1, s, -1), 2, logical_blocks)
    converted = physical_blocks * block_size + torch.remainder(safe_topk_idxs, block_size)
    topk_idxs.copy_(torch.where(valid, converted, -1))


def _make_convert_kv_index_inputs(
    b: int, s: int, k: int, n: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = get_device()
    topk_idxs = torch.randint(0, n * block_size, (b, s, k), dtype=torch.int32, device=device)
    context_lens = torch.randint(0, k + 1, (b, s), dtype=torch.int32, device=device)
    if context_lens.numel() == 1:
        context_lens.fill_(k + 17)
    else:
        context_lens.view(-1)[0] = 0
        context_lens.view(-1)[-1] = k + 17
    valid = torch.arange(k, device=device, dtype=torch.int32).view(1, 1, k) < context_lens.unsqueeze(-1)
    topk_idxs.masked_fill_(~valid, n * block_size + 7)
    block_tables = (
        torch.stack([torch.randperm(n * 2, device=device)[:n] for _ in range(b)]).to(torch.int32).contiguous()
    )
    return (topk_idxs, context_lens, block_tables)


def test_convert_kv_index_known_values() -> None:
    device = get_device()
    topk_idxs = torch.tensor([[[0, 3, 4, 6, 999]]], dtype=torch.int32, device=device)
    context_lens = torch.tensor([[4]], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[7, 2, 9]], dtype=torch.int32, device=device)
    expected = torch.tensor([[[28, 31, 8, 10, -1]]], dtype=torch.int32, device=device)
    convert_kv_index(topk_idxs, 4, context_lens, block_tables)
    assert torch.equal(topk_idxs, expected), f'topk_idxs={topk_idxs!r} vs expected={expected!r}'


@pytest.mark.parametrize(
    'b,s,k,n,block_size', [(1, 1, 1, 4, 1), (2, 3, 127, 17, 16), (4, 1, 512, 128, 32), (3, 2, 513, 65, 64)]
)
def test_convert_kv_index(b: int, s: int, k: int, n: int, block_size: int) -> None:
    topk_idxs, context_lens, block_tables = _make_convert_kv_index_inputs(b, s, k, n, block_size)
    expected = topk_idxs.clone()
    actual = topk_idxs.clone()
    actual_again = topk_idxs.clone()
    convert_kv_index_ref(expected, block_size, context_lens, block_tables)
    convert_kv_index(actual, block_size, context_lens, block_tables)
    convert_kv_index(actual_again, block_size, context_lens, block_tables)
    assert torch.equal(actual, expected), f'FAIL b={b!r}, s={s!r}, k={k!r}, n={n!r}, block_size={block_size!r}'
    assert torch.equal(actual, actual_again), 'FAIL: non-deterministic output'


@pytest.mark.benchmark
@pytest.mark.parametrize('b', [1, 128, 512])
def test_convert_kv_index_benchmark(
    b: int, benchmark_timer: Callable[..., Any], benchmark_record: Callable[..., Any]
) -> Any:
    s = 1
    k = 512
    n = 4096
    block_size = 32
    device = get_device()
    topk_idxs = torch.randint(0, n * block_size, (b, s, k), dtype=torch.int32, device=device)
    context_lens = torch.full((b, s), k, dtype=torch.int32, device=device)
    block_tables = torch.arange(n, dtype=torch.int32, device=device).expand(b, n).contiguous()

    def fn() -> None:
        convert_kv_index(topk_idxs, block_size, context_lens, block_tables)

    t_us = benchmark_timer(fn)
    index_bytes = topk_idxs.numel() * topk_idxs.element_size()
    data_bytes = index_bytes * 3 + context_lens.numel() * context_lens.element_size()
    benchmark_record(
        kernel='flash_mla_utils',
        operation='convert_kv_index',
        params={'b': b, 's': s, 'k': k, 'n': n, 'block_size': block_size},
        time_us=t_us,
        extras={'bandwidth_gbs': data_bytes / 1000000000.0 / (t_us / 1000000.0)},
    )
