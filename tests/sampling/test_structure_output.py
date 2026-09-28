from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.sampling.structure_output import apply_token_bitmask_inplace

pytestmark = pytest.mark.npu

BITS_PER_BLOCK = 32


def apply_token_bitmask_inplace_ref(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    vocab_size = min(logits.shape[1], bitmask.shape[1] * BITS_PER_BLOCK)
    rows = indices.cpu()
    cols = torch.arange(vocab_size, device=logits.device)
    word_ids = cols // BITS_PER_BLOCK
    bit_ids = cols % BITS_PER_BLOCK
    for row in rows:
        row_idx = int(row)
        allowed = ((bitmask[row_idx, word_ids] >> bit_ids) & 1).bool()
        logits[row_idx, :vocab_size].masked_fill_(~allowed, -torch.inf)


@pytest.mark.parametrize('dtype', [torch.float32])
def test_apply_token_bitmask_inplace(dtype: torch.dtype) -> None:
    batch_size = 17
    vocab_size = 8193
    bitmask_size = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK
    device = get_device()
    logits = torch.randn((batch_size, vocab_size), dtype=dtype, device=device)
    bitmask = torch.randint(-(2**31), 2**31 - 1, (batch_size, bitmask_size), dtype=torch.int32, device=device)
    indices = torch.tensor([0, 2, 5, 8, 13], dtype=torch.int32, device=device)

    logits_ref = logits.clone()
    logits_tl = logits.clone()
    apply_token_bitmask_inplace_ref(logits_ref, bitmask, indices)
    apply_token_bitmask_inplace(logits_tl, bitmask, indices)
    assert torch.equal(logits_ref, logits_tl)


@pytest.mark.benchmark
@pytest.mark.parametrize('batch_size', [1, 256, 512])
def test_apply_token_bitmask_inplace_benchmark(
    batch_size: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    vocab_size = 129280
    bitmask_size = (vocab_size + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK
    device = get_device()
    logits = torch.randn((batch_size, vocab_size), dtype=torch.float32, device=device)
    bitmask = torch.randint(-(2**31), 2**31 - 1, (batch_size, bitmask_size), dtype=torch.int32, device=device)
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)

    io_bytes = batch_size * vocab_size * logits.element_size()
    io_bytes += batch_size * bitmask_size * bitmask.element_size()
    io_bytes += batch_size * indices.element_size()
    t_us = benchmark_timer(lambda: apply_token_bitmask_inplace(logits, bitmask, indices))
    benchmark_record(
        kernel='structure_output',
        operation='apply_token_bitmask_inplace',
        params={'batch_size': batch_size, 'vocab_size': vocab_size},
        time_us=t_us,
        extras={'bandwidth_gbs': io_bytes / 1e9 / (t_us / 1e6)},
    )
