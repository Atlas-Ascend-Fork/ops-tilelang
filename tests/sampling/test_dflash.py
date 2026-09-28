from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.sampling.dflash import confidence_head_fused, dflash_input_store

pytestmark = pytest.mark.npu


def confidence_head_fused_ref(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    markov_emb: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    features = torch.cat([hidden, markov_emb], dim=-1) if markov_emb is not None else hidden
    logits = (features.float() @ weight.float()).squeeze(-1)
    scaled = (logits / max(temperature, 1e-5)).clamp(-60.0, 60.0)
    result = 1.0 / (1.0 + (-scaled).exp())
    if out is None:
        return result
    out.copy_(result)
    return out


def dflash_input_store_ref(hidden_states: torch.Tensor, out: torch.Tensor) -> None:
    hc_mult = hidden_states.shape[-2] if hidden_states.ndim > 2 else 1
    if hc_mult > 1:
        hidden_states = hidden_states.mean(dim=-2).to(hidden_states.dtype)
    out.copy_(hidden_states)


@pytest.mark.parametrize('num_rows', [4097])
@pytest.mark.parametrize('n_embed', [4096, 7168])
@pytest.mark.parametrize('markov_rank', [None, 128, 256, 512])
def test_confidence_head(num_rows: int, n_embed: int, markov_rank: int | None) -> None:
    device = get_device()
    total_dim = n_embed + (markov_rank or 0)
    hidden = torch.randn(num_rows, n_embed, dtype=torch.bfloat16, device=device) / 10
    weight = torch.randn(total_dim, 1, dtype=torch.bfloat16, device=device) / 10
    markov_emb = torch.randn(num_rows, markov_rank, dtype=torch.bfloat16, device=device) if markov_rank else None

    def run_kernel() -> torch.Tensor:
        out = torch.empty(num_rows * 7, dtype=torch.float32, device=device)[::7]
        confidence_head_fused(hidden, weight, markov_emb, out=out, temperature=1.0)
        return out.contiguous()

    expected = confidence_head_fused_ref(hidden, weight, markov_emb)
    actual = run_kernel()
    max_diff = (actual - expected).abs().max().item()
    assert max_diff < 1e-4, f'{num_rows=}, {markov_rank=}, {max_diff=}'

    results = [run_kernel() for _ in range(10)]
    assert all(torch.equal(results[0], result) for result in results[1:])


@pytest.mark.benchmark
@pytest.mark.parametrize('num_rows', [1, 128, 4097])
@pytest.mark.parametrize('n_embed', [4096, 7168])
@pytest.mark.parametrize('markov_rank', [None, 256])
def test_confidence_head_benchmark(
    num_rows: int,
    n_embed: int,
    markov_rank: int | None,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    device = get_device()
    total_dim = n_embed + (markov_rank or 0)
    hidden = torch.randn(num_rows, n_embed, dtype=torch.bfloat16, device=device) / 10
    weight = torch.randn(total_dim, 1, dtype=torch.bfloat16, device=device) / 10
    markov_emb = torch.randn(num_rows, markov_rank, dtype=torch.bfloat16, device=device) if markov_rank else None

    time_us = benchmark_timer(lambda: confidence_head_fused(hidden, weight, markov_emb, temperature=1.0))
    tensors = [hidden, weight] + ([markov_emb] if markov_emb is not None else [])
    data_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    benchmark_record(
        kernel='dflash',
        operation='confidence_head',
        params={'num_rows': num_rows, 'markov_rank': markov_rank or 0, 'n_embed': n_embed},
        time_us=time_us,
        extras={'bandwidth_gbs': data_bytes / time_us / 1e3},
    )


@pytest.mark.parametrize('num_rows', [3, 7, 233, 4097, 8198])
@pytest.mark.parametrize('n_embed', [4096, 7168])
@pytest.mark.parametrize('hc_mult', [1, 4])
def test_dflash_input_store(num_rows: int, n_embed: int, hc_mult: int) -> None:
    device = get_device()
    if hc_mult > 1:
        hidden_states = torch.randn(num_rows, hc_mult, n_embed, dtype=torch.bfloat16, device=device)
    else:
        hidden_states = torch.randn(num_rows, n_embed, dtype=torch.bfloat16, device=device)

    num_layers = 3
    buffer_ref = torch.randn(num_rows, n_embed * num_layers, dtype=torch.bfloat16, device=device)
    buffer_actual = buffer_ref.clone()
    buffer_repeat = buffer_ref.clone()
    out_ref = buffer_ref[:, n_embed : 2 * n_embed]
    out_actual = buffer_actual[:, n_embed : 2 * n_embed]

    dflash_input_store_ref(hidden_states, out_ref)
    dflash_input_store(hidden_states, out_actual)

    max_diff = (buffer_actual.float() - buffer_ref.float()).abs().max().item()
    assert max_diff < 2e-3, f'{num_rows=}, {n_embed=}, {hc_mult=}, {max_diff=}'

    dflash_input_store(hidden_states, buffer_repeat[:, n_embed : 2 * n_embed])
    assert torch.equal(buffer_actual, buffer_repeat)


@pytest.mark.benchmark
@pytest.mark.parametrize('num_rows', [1, 128, 4097])
@pytest.mark.parametrize('n_embed', [4096, 7168])
@pytest.mark.parametrize('hc_mult', [1, 4])
def test_dflash_input_store_benchmark(
    num_rows: int,
    n_embed: int,
    hc_mult: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    device = get_device()
    if hc_mult > 1:
        hidden_states = torch.randn(num_rows, hc_mult, n_embed, dtype=torch.bfloat16, device=device)
    else:
        hidden_states = torch.randn(num_rows, n_embed, dtype=torch.bfloat16, device=device)
    buffer_actual = torch.randn(num_rows, n_embed * 3, dtype=torch.bfloat16, device=device)
    out = buffer_actual[:, n_embed : 2 * n_embed]

    time_us = benchmark_timer(lambda: dflash_input_store(hidden_states, out))
    data_bytes = sum(tensor.numel() * tensor.element_size() for tensor in (hidden_states, out))
    benchmark_record(
        kernel='dflash',
        operation='dflash_input_store',
        params={'hc_mult': hc_mult, 'num_rows': num_rows, 'n_embed': n_embed},
        time_us=time_us,
        extras={'bandwidth_gbs': data_bytes / time_us / 1e3},
    )
