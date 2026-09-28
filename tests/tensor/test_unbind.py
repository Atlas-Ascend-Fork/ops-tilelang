from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.tensor.unbind import unbind

pytestmark = pytest.mark.npu


def unbind_ref(from_: torch.Tensor) -> list[torch.Tensor]:
    return [t.contiguous() for t in torch.unbind(from_, dim=0)]


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32, torch.int64])
@pytest.mark.parametrize('shape', [(20, 20), (20, 7, 13), (20,)])
def test_unbind(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    if dtype.is_floating_point:
        x = torch.randn(shape, dtype=dtype, device=get_device())
    else:
        x = torch.randint(0, 1000, shape, dtype=dtype, device=get_device())

    out = unbind(x)
    ref = unbind_ref(x)
    for out_i, ref_i in zip(out, ref, strict=True):
        assert out_i.data_ptr() != ref_i.data_ptr()
        assert out_i.is_contiguous()
        assert torch.equal(out_i, ref_i)


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32, torch.int64])
def test_unbind_strided_dim0(dtype: torch.dtype) -> None:
    if dtype.is_floating_point:
        x = torch.randn((40, 2000), dtype=dtype, device=get_device())[::2]
    else:
        x = torch.randint(0, 1000, (40, 2000), dtype=dtype, device=get_device())[::2]

    out = unbind(x)
    ref = unbind_ref(x)

    for out_i, ref_i in zip(out, ref, strict=True):
        assert out_i.is_contiguous()
        assert torch.equal(out_i, ref_i)


def test_unbind_non_contiguous_inner() -> None:
    x = torch.randn((40, 20, 30), dtype=torch.bfloat16, device=get_device())
    x = x[::2].transpose(-1, -2)

    out = unbind(x)
    ref = unbind_ref(x)
    for out_i, ref_i in zip(out, ref, strict=True):
        assert out_i.is_contiguous()
        assert torch.equal(out_i, ref_i)


@pytest.mark.parametrize('req_cnt', [64, 256, 512])
def test_unbind_small_specialized(req_cnt: int) -> None:
    """Exercise every static 4 KiB kernel, including grouped requests."""
    x = torch.arange(
        req_cnt * 1024,
        dtype=torch.float32,
        device=get_device(),
    ).reshape(req_cnt, 1024)

    out = unbind(x)
    ref = unbind_ref(x)

    assert len(out) == req_cnt
    for out_i, ref_i in zip(out, ref, strict=True):
        assert out_i.is_contiguous()
        assert torch.equal(out_i, ref_i)


GENERALIZED_STATIC_PROFILES = [
    ('scalar-fp32', torch.float32, ()),
    ('vector1-bf16', torch.bfloat16, (1,)),
    ('vector3-fp16', torch.float16, (3,)),
    ('vector5-int64', torch.int64, (5,)),
    ('vector17-int32', torch.int32, (17,)),
    ('vector31-uint8', torch.uint8, (31,)),
    ('vector33-bool', torch.bool, (33,)),
    ('matrix-prime-fp32', torch.float32, (3, 5)),
    ('rank3-bf16', torch.bfloat16, (2, 3, 7)),
    ('rank3-int64', torch.int64, (2, 5, 3)),
]


GENERALIZED_RUNTIME_PROFILES = [
    ('single-contiguous', 1, 'contiguous'),
    ('two-strided', 2, 'strided_dim0'),
    ('tiny-inner-noncontiguous', 3, 'non_contiguous_inner'),
    ('seven-contiguous', 7, 'contiguous'),
    ('prime-strided', 17, 'strided_dim0'),
    ('below-core-count', 31, 'non_contiguous_inner'),
    ('core-tail-minus-one', 47, 'contiguous'),
    ('core-count', 48, 'strided_dim0'),
    ('core-tail-plus-one', 49, 'non_contiguous_inner'),
    ('multi-core-rounds', 97, 'contiguous'),
    ('large-prime', 129, 'strided_dim0'),
    ('persistent-tail', 257, 'non_contiguous_inner'),
]


GENERALIZED_CASES = [
    pytest.param(dtype, inner_shape, req_cnt, layout, id=f'{static_name}-{runtime_name}')
    for static_name, dtype, inner_shape in GENERALIZED_STATIC_PROFILES
    for runtime_name, req_cnt, layout in GENERALIZED_RUNTIME_PROFILES
]


def _make_generalized_unbind_input(
    dtype: torch.dtype,
    inner_shape: tuple[int, ...],
    req_cnt: int,
    layout: str,
) -> torch.Tensor:
    def make_values(shape: tuple[int, ...]) -> torch.Tensor:
        numel = 1
        for extent in shape:
            numel *= extent
        values = torch.arange(numel, dtype=torch.float32, device=get_device()).reshape(shape)
        if dtype == torch.bool:
            return (values.to(torch.int32) % 3) == 0
        if dtype.is_floating_point:
            values = values / 17.0 - 5.0
        return values.to(dtype)

    logical_shape = (req_cnt, *inner_shape)
    if layout == 'contiguous':
        return make_values(logical_shape)
    if layout == 'strided_dim0':
        return make_values((req_cnt * 2, *inner_shape))[::2]
    if len(inner_shape) >= 2:
        storage_shape = (req_cnt, *inner_shape[:-2], inner_shape[-1], inner_shape[-2])
        return make_values(storage_shape).transpose(-1, -2)
    if inner_shape:
        return make_values((req_cnt, inner_shape[0] * 2))[:, ::2]
    return make_values((req_cnt * 2,))[::2]


@pytest.mark.parametrize('dtype,inner_shape,req_cnt,layout', GENERALIZED_CASES)
def test_unbind_generalized(
    dtype: torch.dtype,
    inner_shape: tuple[int, ...],
    req_cnt: int,
    layout: str,
) -> None:
    x = _make_generalized_unbind_input(dtype, inner_shape, req_cnt, layout)
    out = unbind(x)
    ref = unbind_ref(x)

    assert len(out) == req_cnt
    assert len({tensor.data_ptr() for tensor in out}) == req_cnt
    for out_i, ref_i in zip(out, ref, strict=True):
        assert out_i.is_contiguous()
        assert out_i.shape == inner_shape
        assert out_i.dtype == dtype
        assert torch.equal(out_i, ref_i)


@pytest.mark.benchmark
@pytest.mark.parametrize('req_cnt', [64, 256, 512])
@pytest.mark.parametrize('nbytes_per_req', [4096, 4096 * 1024])
def test_unbind_benchmark(
    req_cnt: int,
    nbytes_per_req: int,
    benchmark_timer: Callable[..., Any],
    benchmark_record: Callable[..., Any],
) -> None:
    x = torch.randn((req_cnt, nbytes_per_req // 4), dtype=torch.float32, device=get_device())

    t_us = benchmark_timer(lambda: unbind(x))
    total_bytes = req_cnt * nbytes_per_req * 2
    benchmark_record(
        kernel='unbind',
        operation='unbind',
        params={'nbytes_per_req': nbytes_per_req, 'req_cnt': req_cnt},
        time_us=t_us,
        extras={'bandwidth_gbs': total_bytes / 1e9 / (t_us / 1e6)},
    )
