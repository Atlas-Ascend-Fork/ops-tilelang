# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

import os
from dataclasses import dataclass, replace
from itertools import product

import torch

VOCAB_BLOCK = 8192
ROW_BLOCK = 256
GRID_CORES = 64
DTYPES = ('float32', 'bfloat16', 'float16')
REDUCTIONS = ('none', 'sum', 'mean')


@dataclass(frozen=True)
class Case:
    name: str
    n: int
    v: int
    dtype: str = 'float32'
    reduction: str = 'none'
    pattern: str = 'random'
    ignore: str = 'none'
    ignore_index: int = -100
    padding: int = 0
    offset: int = 0
    seed: int = 20261026

    @property
    def id(self):
        return f'{self.name}-N{self.n}-V{self.v}-{self.dtype}-{self.reduction}'


def get_test_level():
    level = int(os.environ.get('OPS_TILELANG_TEST_LEVEL', '1'))
    if level not in (0, 1, 2):
        raise ValueError('OPS_TILELANG_TEST_LEVEL must be 0, 1, or 2')
    return level


def generate_test_params(level=1, *, block_v=VOCAB_BLOCK, block_rows=ROW_BLOCK, grid_cores=GRID_CORES):
    if level not in (0, 1, 2):
        raise ValueError('invalid test level or planning tile size')
    if block_v < 2 or block_rows < 2 or grid_cores < 1:
        raise ValueError('invalid test level or planning tile size')
    geometries = _test_geometries(block_v, block_rows, grid_cores)
    if level == 0:
        return [
            geometries[2],
            replace(geometries[11], dtype='bfloat16', reduction='mean'),
            replace(geometries[18], dtype='float16', reduction='sum'),
        ]
    cases = [
        replace(case, dtype=dtype, reduction=reduction)
        for (case, dtype, reduction) in product(geometries, DTYPES, REDUCTIONS)
    ]
    cases.extend(Case('fp32-overflow', 3, 2, reduction=r, pattern='overflow') for r in REDUCTIONS)
    cases.extend(Case('fp32-sum-overflow', 3, 2, reduction=r, pattern='sum_overflow') for r in REDUCTIONS)
    for dtype, reduction in product(('float32', 'bfloat16'), REDUCTIONS):
        cases.append(
            Case('row-overflow-finite-mean', 3, 2, dtype=dtype, reduction=reduction, pattern='row_overflow_mixed')
        )
        cases.append(
            Case(
                'sum-overflow-with-ignore',
                7,
                2,
                dtype=dtype,
                reduction=reduction,
                pattern='sum_overflow',
                ignore='partial',
            )
        )
    if level == 2:
        _add_interaction_cases(cases, block_v, block_rows)
    return cases


def make_inputs(case, *, block_v=VOCAB_BLOCK, return_storage=False):
    rng = torch.Generator().manual_seed(case.seed)
    x = torch.randn(case.n, case.v, generator=rng)
    positions = [
        0,
        case.v - 1,
        case.v // 2,
        min(block_v - 1, case.v - 1),
        min(block_v, case.v - 1),
        min(2 * block_v - 1, case.v - 1),
        (case.v - 1) // block_v * block_v,
    ]
    y = torch.tensor([positions[i % len(positions)] for i in range(case.n)], dtype=torch.int64)
    _apply_pattern(case, x, y, positions)
    if case.ignore == 'partial':
        y[::2] = case.ignore_index
    elif case.ignore == 'all':
        y.fill_(case.ignore_index)
    elif case.ignore == 'one_valid':
        y.fill_(case.ignore_index)
        y[-1] = case.v - 1
    elif case.ignore != 'none':
        raise ValueError(f'unknown ignore mode: {case.ignore}')
    dtype = getattr(torch, case.dtype)
    stride = case.v + case.padding
    storage = torch.full((case.offset + case.n * stride,), 777, dtype=dtype)
    logits = storage.as_strided((case.n, case.v), (stride, 1), case.offset)
    logits.copy_(x.to(dtype))
    target_storage = torch.full((case.n + 1,), 999999, dtype=torch.int64)
    targets = target_storage[1:]
    targets.copy_(y)
    if return_storage:
        return logits, targets, storage, target_storage
    return (logits, targets)


def coverage_tags(case, logits, targets, *, block_v=VOCAB_BLOCK, block_rows=ROW_BLOCK, grid_cores=GRID_CORES):
    valid = targets != case.ignore_index
    tags = {f'dtype:{case.dtype}', f'reduction:{case.reduction}', f'pattern:{case.pattern}'}
    tags.add('rows:empty' if case.n == 0 else 'rows:nonempty')
    tags.add('vocab:one-class' if case.v == 1 else 'vocab:many-classes')
    tags.add('chunks:single' if case.v <= block_v else 'chunks:multiple')
    tags.add('vocab:full-block' if case.v % block_v == 0 else 'vocab:tail')
    tags.add('rows:reduction-tail' if case.n % block_rows else 'rows:reduction-aligned')
    if case.n > block_rows:
        tags.add('rows:multi-reduction-block')
    if case.n > grid_cores:
        tags.add('rows:persistent-multi-wave')
    tags.add('layout:row-strided' if case.padding else 'layout:compact')
    if logits.storage_offset():
        tags.add('layout:offset')
    if case.n:
        _add_target_tags(tags, case, targets, valid, block_v, grid_cores)
    if case.ignore_index != -100:
        tags.add('ignore:custom-in-range' if 0 <= case.ignore_index < case.v else 'ignore:custom-out-of-range')
    return tags


REQUIRED_TAGS = {
    *(f'dtype:{d}' for d in DTYPES),
    *(f'reduction:{r}' for r in REDUCTIONS),
    'rows:empty',
    'rows:nonempty',
    'vocab:one-class',
    'vocab:many-classes',
    'chunks:single',
    'chunks:multiple',
    'vocab:full-block',
    'vocab:tail',
    'rows:reduction-tail',
    'rows:reduction-aligned',
    'rows:multi-reduction-block',
    'rows:persistent-multi-wave',
    'layout:row-strided',
    'layout:compact',
    'layout:offset',
    'tasks:one-wave',
    'tasks:multi-wave',
    'tasks:full-wave',
    'tasks:tail-wave',
    'ignore:none',
    'ignore:partial',
    'ignore:all',
    'ignore:one-valid',
    'ignore:custom-in-range',
    'ignore:custom-out-of-range',
    'target:first',
    'target:last',
    'target:before-boundary',
    'target:after-boundary',
    'pattern:example',
    'pattern:boundary',
    'pattern:zeros',
    'pattern:negative',
    'pattern:large_constant',
    'pattern:large_range',
    'pattern:dominant',
    'pattern:wide_random',
    'pattern:overflow',
    'pattern:sum_overflow',
    'pattern:row_overflow_mixed',
    'pattern:random',
}


def _test_geometries(block_v, block_rows, grid_cores):
    geometries = [
        Case('empty', 0, 7),
        Case('one-class', 3, 1, ignore='partial'),
        Case('cat-dog-bird', 3, 3, pattern='example'),
        Case('tiny-tail', 7, 31),
        Case('vector-full', 7, 32),
        Case('vector-tail', 7, 33),
        Case('block-minus-one', 7, block_v - 1, pattern='boundary'),
        Case('block-exact', 7, block_v, pattern='boundary'),
        Case('block-plus-one', 7, block_v + 1, pattern='boundary'),
        Case('two-full-blocks', 7, 2 * block_v, pattern='boundary'),
        Case('two-blocks-plus-one', 7, 2 * block_v + 1, pattern='boundary'),
        Case('model-vocab', 7, 129280, pattern='boundary', ignore='partial'),
        Case('model-minus-one', 2, 129279),
        Case('model-plus-one', 2, 129281),
        Case('row-minus-one', block_rows - 1, 33),
        Case('row-exact', block_rows, 33),
        Case('row-plus-one', block_rows + 1, 33),
        Case('persistent-tail', 2 * grid_cores + 1, 127),
        Case('strided-offset', 7, block_v + 1, padding=7, offset=3, ignore='partial'),
        Case('label-first-last', 7, 17, pattern='boundary'),
        Case('all-ignore', 7, block_v + 1, ignore='all'),
        Case('only-one-valid', 7, 33, ignore='one_valid'),
        Case('custom-ignore-positive', 7, 33, ignore='partial', ignore_index=40),
        Case('custom-ignore-class', 7, 33, ignore='partial', ignore_index=0),
        Case('custom-ignore-negative', 7, 33, ignore='partial', ignore_index=-7),
        Case('zeros', 3, 33, pattern='zeros'),
        Case('negative-padding', 3, block_v + 1, pattern='negative'),
        Case('large-common-offset', 3, 33, pattern='large_constant'),
        Case('large-range', 3, 33, pattern='large_range'),
        Case('confident-right-wrong', 7, 33, pattern='dominant'),
        Case('wide-random', 7, block_v + 1, pattern='wide_random'),
        Case('single-row', 1, 7),
        Case('single-row-model', 1, 129280),
        Case('many-row-blocks', 16 * block_rows + 1, 33, ignore='partial'),
    ]
    return geometries


def _add_interaction_cases(cases, block_v, block_rows):
    interaction_shapes = [
        (1, 1),
        (7, block_v - 1),
        (7, block_v),
        (7, block_v + 1),
        (7, 129280),
        (block_rows + 1, 33),
    ]
    for (n, v), dtype, reduction, ignore, padding in product(
        interaction_shapes, DTYPES, REDUCTIONS, ('none', 'partial', 'all'), (0, 7)
    ):
        cases.append(
            Case(
                f'interaction-{ignore}-pad{padding}',
                n,
                v,
                dtype,
                reduction,
                ignore=ignore,
                padding=padding,
                offset=1,
                seed=20261027,
            )
        )


def _apply_pattern(case, x, y, positions):
    if case.pattern == 'example':
        x[:] = torch.tensor([2.0, 1.0, 0.0])
        y[:] = torch.tensor([0, 1, 2])
    elif case.pattern == 'boundary':
        x.fill_(-7)
        for i in range(case.n):
            x[i, positions[(i + 1) % len(positions)]] = 9
    elif case.pattern == 'zeros':
        x.zero_()
    elif case.pattern == 'negative':
        x.fill_(-10000)
    elif case.pattern == 'large_constant':
        x.fill_(10000 if case.dtype == 'float16' else 100000000.0)
    elif case.pattern == 'large_range':
        x.fill_(-10000)
        x[:, :2] = torch.tensor([10000.0, 9999.0])
    elif case.pattern == 'dominant':
        x.fill_(-100)
        x[:, 0] = 100
    elif case.pattern == 'wide_random':
        x *= 20
    elif case.pattern in ('overflow', 'sum_overflow', 'row_overflow_mixed'):
        magnitude = 1e38 if case.pattern == 'sum_overflow' else 2.5e38
        x[:] = torch.tensor([magnitude, -magnitude])
        y.fill_(1)
        if case.pattern == 'row_overflow_mixed':
            y[1:] = 0
    elif case.pattern != 'random':
        raise ValueError(f'unknown pattern: {case.pattern}')


def _add_target_tags(tags, case, targets, valid, block_v, grid_cores):
    tasks = case.n * ((case.v + block_v - 1) // block_v)
    tags.add('tasks:one-wave' if tasks <= grid_cores else 'tasks:multi-wave')
    tags.add('tasks:full-wave' if tasks % grid_cores == 0 else 'tasks:tail-wave')
    count = int(valid.sum())
    tags.add('ignore:all' if count == 0 else 'ignore:none' if count == case.n else 'ignore:partial')
    if count == 1:
        tags.add('ignore:one-valid')
    for label, value in (
        ('first', 0),
        ('last', case.v - 1),
        ('before-boundary', block_v - 1),
        ('after-boundary', block_v),
    ):
        if bool(((targets == value) & valid).any()):
            tags.add(f'target:{label}')
