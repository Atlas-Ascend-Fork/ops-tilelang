# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

import torch

INPUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
REDUCTIONS = ('none', 'sum', 'mean')


def validate_reference_inputs(logits, targets, *, ignore_index=-100, reduction='none'):
    if not isinstance(logits, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError('logits and targets must be tensors')
    if logits.ndim != 2 or targets.ndim != 1:
        raise ValueError('expected logits [N, V] and targets [N]')
    if logits.shape[0] != targets.shape[0] or logits.shape[1] < 1:
        raise ValueError('matching N and V >= 1 are required')
    if logits.dtype not in INPUT_DTYPES or targets.dtype != torch.int64:
        raise TypeError('expected FP32/BF16/FP16 logits and int64 targets')
    if logits.device.type != 'cpu' or targets.device.type != 'cpu':
        raise ValueError('references require CPU inputs')
    if logits.stride(1) != 1 or (logits.shape[0] > 1 and logits.stride(0) < logits.shape[1]):
        raise ValueError('logits require contiguous columns and nonoverlapping rows')
    if not targets.is_contiguous():
        raise ValueError('targets must be contiguous')
    if type(ignore_index) is not int:
        raise TypeError('ignore_index must be a Python int, not bool')
    if not -(2**63) <= ignore_index < 2**63:
        raise ValueError('ignore_index must fit int64')
    if reduction not in REDUCTIONS:
        raise ValueError('reduction must be none, sum, or mean')
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('input logits must be finite')
    invalid = (targets != ignore_index) & ((targets < 0) | (targets >= logits.shape[1]))
    if bool(invalid.any()):
        raise ValueError('target must be in [0, V) or equal ignore_index')


@torch.no_grad()
def row_losses_fp64(logits, targets, *, ignore_index=-100):
    validate_reference_inputs(logits, targets, ignore_index=ignore_index)
    valid = targets != ignore_index
    x = logits.double()[valid]
    shifted = x - x.max(dim=1, keepdim=True).values
    selected = shifted.gather(1, targets[valid, None]).squeeze(1)
    rows = torch.zeros(logits.shape[0], dtype=torch.float64)
    rows[valid] = shifted.exp().sum(dim=1).log() - selected
    return rows


@torch.no_grad()
def cross_entropy_golden(logits, targets, *, ignore_index=-100, reduction='none'):
    if reduction not in REDUCTIONS:
        raise ValueError('reduction must be none, sum, or mean')
    rows = row_losses_fp64(logits, targets, ignore_index=ignore_index)
    valid = targets != ignore_index
    if reduction == 'none':
        return rows.float()
    if reduction == 'sum':
        return rows.sum().float()
    return (rows.sum() / valid.sum()).float()
