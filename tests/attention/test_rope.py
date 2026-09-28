from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.attention.rope import rotary_embedding
from cann_ops_tilelang.config import get_device

pytestmark = pytest.mark.npu


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    name: str = '',
) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    for start in range(0, actual.numel(), 8 * 1024 * 1024):
        end = min(start + 8 * 1024 * 1024, actual.numel())
        actual_chunk, expected_chunk = actual_flat[start:end].cpu(), expected_flat[start:end].cpu()
        try:
            tolerances = {'atol': 3e-2, 'rtol': 1e-2} if actual.dtype == torch.bfloat16 else {}
            torch.testing.assert_close(actual_chunk, expected_chunk, **tolerances)
        except AssertionError as error:
            raise AssertionError(f'{name or "tensor"}[{start}:{end}]: {error}') from error


def rotary_embedding_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
    seqlen_offsets: int = 0,
    interleaved: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Pure-PyTorch reference for rotary embedding."""
    if positions is not None:
        cos_full = cos[positions + seqlen_offsets].unsqueeze(1)
        sin_full = sin[positions + seqlen_offsets].unsqueeze(1)
    else:
        cos_full = cos[seqlen_offsets].unsqueeze(0).unsqueeze(0)
        sin_full = sin[seqlen_offsets].unsqueeze(0).unsqueeze(0)
    rot_dim = cos.shape[-1] * 2
    out = torch.empty_like(x)

    if conjugate:
        sin_full = -sin_full

    out[:, :, rot_dim:] = x[:, :, rot_dim:]
    x_rot = x[:, :, :rot_dim].clone()

    if interleaved:
        x0 = x_rot[:, :, 0::2]
        x1 = x_rot[:, :, 1::2]
    else:
        x0 = x_rot[:, :, : rot_dim // 2]
        x1 = x_rot[:, :, rot_dim // 2 :]

    x0_res = x0 * cos_full - x1 * sin_full
    x1_res = x0 * sin_full + x1 * cos_full

    if interleaved:
        x_rot[:, :, 0::2] = x0_res
        x_rot[:, :, 1::2] = x1_res
    else:
        x_rot[:, :, : rot_dim // 2] = x0_res
        x_rot[:, :, rot_dim // 2 :] = x1_res

    out[:, :, :rot_dim] = x_rot
    return out


@pytest.mark.parametrize('nheads,nkvheads', [(1, 1), (32, 32), (64, 64), (128, 128)])
@pytest.mark.parametrize('dim', [16, 32, 64, 128])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float])
@pytest.mark.parametrize('interleaved', [True, False])
@pytest.mark.parametrize('conjugate', [True, False])
def test_apply_rotary(
    nheads: int,
    nkvheads: int,
    dim: int,
    interleaved: bool,
    conjugate: bool,
    dtype: torch.dtype,
) -> None:
    seqlen = 6144
    seqlen_total = 24 * 1024
    device = get_device()

    query = torch.randn(seqlen, nheads, dim + 128, dtype=dtype, device=device)[..., -dim:]
    key = torch.randn(seqlen, nkvheads, dim + 128, dtype=dtype, device=device)[..., -dim:]
    cos_sin = torch.randn(seqlen_total, dim, dtype=torch.float, device=device)
    cos = cos_sin[:, : dim // 2]
    sin = cos_sin[:, dim // 2 :]
    positions = torch.randint(0, seqlen_total, (seqlen,), dtype=torch.int32, device=device)

    ref_query = rotary_embedding_ref(query, cos, sin, positions, interleaved=interleaved, conjugate=conjugate)
    ref_key = rotary_embedding_ref(key, cos, sin, positions, interleaved=interleaved, conjugate=conjugate)

    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    rotary_embedding(
        positions,
        query,
        key,
        query_out,
        key_out,
        head_size=dim,
        cos_sin_cache=cos_sin,
        is_neox=(not interleaved),
        conjugate=conjugate,
    )

    assert_close(ref_query, query_out, 'query')
    assert_close(ref_key, key_out, 'key')


def _run_rotary_case(
    seqlen: int,
    nheads: int,
    nkvheads: int,
    head_size: int,
    rotary_dim: int,
    dtype: torch.dtype,
    interleaved: bool,
    conjugate: bool = False,
    positions_dtype: torch.dtype = torch.int32,
    flattened: bool = False,
    use_key: bool = True,
    out_of_place: bool = True,
    head_padding: int = 0,
) -> None:
    device = get_device()

    def make_tensor(heads: int) -> torch.Tensor:
        x = torch.randn(seqlen, heads, head_size + head_padding, dtype=dtype, device=device)[..., :head_size]
        return x.reshape(seqlen, -1) if flattened else x

    query = make_tensor(nheads)
    key = make_tensor(nkvheads) if use_key else None
    query_before, key_before = query.clone(), None if key is None else key.clone()
    cos_sin = torch.randn(257, rotary_dim, dtype=torch.float32, device=device)
    cos, sin = cos_sin.chunk(2, dim=-1)
    positions = torch.randint(0, 257, (seqlen,), dtype=positions_dtype, device=device)
    query_out = torch.empty_like(query, memory_format=torch.contiguous_format) if out_of_place else None
    key_out = torch.empty_like(key, memory_format=torch.contiguous_format) if out_of_place and key is not None else None
    rotary_embedding(positions, query, key, query_out, key_out, head_size, cos_sin, not interleaved, conjugate)

    for name, source, output, before, heads in (
        ('query', query, query_out, query_before, nheads),
        ('key', key, key_out, key_before, nkvheads),
    ):
        if source is None or before is None:
            continue
        before = before.view(seqlen, heads, head_size)
        actual = (source if output is None else output).view(seqlen, heads, head_size)
        expected = rotary_embedding_ref(before, cos, sin, positions, interleaved=interleaved, conjugate=conjugate)
        assert_close(actual, expected, name=name)
        assert torch.equal(actual[..., rotary_dim:], before[..., rotary_dim:])
        if out_of_place:
            assert torch.equal(source, before.view_as(source))


@pytest.mark.parametrize(
    'flattened,use_key,out_of_place,positions_dtype,interleaved,conjugate,dtype',
    [
        (False, True, True, torch.int64, False, False, torch.bfloat16),
        (True, False, False, torch.int32, True, True, torch.float32),
        (True, True, True, torch.int64, True, False, torch.float32),
        (False, False, False, torch.int32, False, True, torch.bfloat16),
    ],
)
def test_apply_rotary_interface_variants(
    flattened: bool,
    use_key: bool,
    out_of_place: bool,
    positions_dtype: torch.dtype,
    interleaved: bool,
    conjugate: bool,
    dtype: torch.dtype,
) -> None:
    _run_rotary_case(
        seqlen=17,
        nheads=7,
        nkvheads=3,
        head_size=96,
        rotary_dim=64,
        dtype=dtype,
        interleaved=interleaved,
        conjugate=conjugate,
        positions_dtype=positions_dtype,
        flattened=flattened,
        use_key=use_key,
        out_of_place=out_of_place,
        head_padding=0 if flattened else 16,
    )


def test_rotary_embedding_input_contract() -> None:
    seqlen = 4
    head_size = 64
    query = torch.randn(seqlen, 2, head_size, dtype=torch.bfloat16)
    positions = torch.arange(seqlen, dtype=torch.int32)
    cos_sin = torch.randn(32, 32, dtype=torch.float32)

    def apply(
        pos: torch.Tensor = positions,
        cache: torch.Tensor = cos_sin,
        out: torch.Tensor | None = None,
    ) -> None:
        rotary_embedding(pos, query, None, out, None, head_size, cache, True)

    with pytest.raises(AssertionError, match='cos_sin_cache must use float32'):
        apply(cache=cos_sin.to(torch.bfloat16))

    positions_storage = torch.arange(seqlen * 2, dtype=torch.int32)
    with pytest.raises(AssertionError, match='positions must be contiguous'):
        apply(pos=positions_storage[::2])

    cos_sin_storage = torch.randn(32, 64, dtype=torch.float32)
    with pytest.raises(AssertionError, match='cos_sin_cache must be contiguous'):
        apply(cache=cos_sin_storage[:, ::2])

    with pytest.raises(AssertionError, match='query_out must have the same dtype'):
        apply(out=torch.empty_like(query, dtype=torch.float32))


@pytest.mark.parametrize(
    'dtype,interleaved,out_of_place',
    [
        (torch.bfloat16, False, True),
        (torch.bfloat16, True, False),
        (torch.float32, False, False),
        (torch.float32, True, True),
    ],
)
def test_apply_rotary_ascend_decode_partial(
    dtype: torch.dtype,
    interleaved: bool,
    out_of_place: bool,
) -> None:
    _run_rotary_case(
        seqlen=1,
        nheads=128,
        nkvheads=8,
        head_size=108,
        rotary_dim=32,
        dtype=dtype,
        interleaved=interleaved,
        out_of_place=out_of_place,
        head_padding=64,
    )


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'nheads,nkvheads,dim',
    [
        (128, 128, 128),
    ],
)
def test_apply_rotary_benchmark(
    nheads: int,
    nkvheads: int,
    dim: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    seqlen = 6144
    seqlen_total = 24 * 1024
    dtype = torch.bfloat16
    device = get_device()

    query = torch.randn(seqlen, nheads, dim + 128, dtype=dtype, device=device)[..., -dim:]
    key = torch.randn(seqlen, nkvheads, dim + 128, dtype=dtype, device=device)[..., -dim:]
    cos_sin = torch.randn(seqlen_total, dim, dtype=torch.float, device=device)
    positions = torch.randint(0, seqlen_total, (seqlen,), dtype=torch.int32, device=device)
    query_out = torch.empty_like(query)
    key_out = torch.empty_like(key)

    def bench_fn() -> None:
        rotary_embedding(
            positions,
            query,
            key,
            query_out,
            key_out,
            head_size=dim,
            cos_sin_cache=cos_sin,
            is_neox=True,
            conjugate=False,
        )

    t_us = benchmark_timer(bench_fn)

    # I/O: read (query + key + cos_sin_cache[positions]) + write (query_out + key_out)
    total_bytes = (query.numel() + key.numel()) * 2 * dtype.itemsize

    benchmark_record(
        kernel='rope',
        operation='rotary_embedding',
        params={
            'dim': dim,
            'nheads': nheads,
            'nkvheads': nkvheads,
            'seqlen': seqlen,
        },
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )
