# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.

import torch

from .cases import make_inputs
from .reference import cross_entropy_golden

RTOL = 1e-05
ATOL = 1e-05


def assert_forward_matches(
    actual, logits_cpu, targets_cpu, *, ignore_index=-100, reduction='none', expected_device='cpu'
):
    expected = cross_entropy_golden(logits_cpu, targets_cpu, ignore_index=ignore_index, reduction=reduction)
    assert_expected_matches(
        actual, expected, targets_cpu, ignore_index=ignore_index, reduction=reduction, expected_device=expected_device
    )


def assert_expected_matches(
    actual, expected, targets_cpu, *, ignore_index=-100, reduction='none', expected_device='cpu'
):
    if not (isinstance(actual, torch.Tensor)):
        raise AssertionError('output must be a tensor')
    if not (actual.dtype == torch.float32):
        raise AssertionError('loss must be FP32 even for BF16/FP16 input')
    if not (actual.shape == expected.shape):
        raise AssertionError('loss shape does not match reduction')
    if not (actual.device == torch.device(expected_device)):
        raise AssertionError('loss is on the wrong device')
    if not (actual.is_contiguous()):
        raise AssertionError('loss must be contiguous')
    actual = actual.detach().cpu()
    valid_count = int((targets_cpu != ignore_index).sum())
    allowed_nan = reduction == 'mean' and valid_count == 0
    if not (bool(torch.isnan(expected).any()) == allowed_nan):
        raise AssertionError('precision case invariant failed')
    if not (torch.equal(torch.isnan(actual), torch.isnan(expected))):
        raise AssertionError('NaN mask differs')
    if not (torch.equal(torch.isposinf(actual), torch.isposinf(expected))):
        raise AssertionError('+Inf mask differs')
    if not (torch.equal(torch.isneginf(actual), torch.isneginf(expected))):
        raise AssertionError('-Inf mask differs')
    finite = torch.isfinite(expected)
    atol = ATOL * max(valid_count, 1) if reduction == 'sum' else ATOL
    torch.testing.assert_close(actual[finite], expected[finite], rtol=RTOL, atol=atol)
    if reduction == 'none':
        ignored = targets_cpu == ignore_index
        if not (torch.equal(actual[ignored], torch.zeros_like(actual[ignored]))):
            raise AssertionError('ignored loss must be exactly zero')
    elif valid_count == 0 and reduction == 'sum':
        if not (actual.item() == 0.0):
            raise AssertionError('empty/all-ignored sum must be exactly zero')


def run_precision_case(implementation, case, *, device='cpu'):
    (x_cpu, y_cpu, x_base, y_base) = make_inputs(case, return_storage=True)
    x_storage = x_base.to(device, copy=True)
    y_storage = y_base.to(device, copy=True)
    x = x_storage.as_strided(x_cpu.shape, x_cpu.stride(), x_cpu.storage_offset())
    y = y_storage.as_strided(y_cpu.shape, y_cpu.stride(), y_cpu.storage_offset())
    if not (x.stride() == x_cpu.stride() and x.storage_offset() == x_cpu.storage_offset()):
        raise AssertionError('precision case invariant failed')
    if not (y.stride() == y_cpu.stride() and y.storage_offset() == y_cpu.storage_offset()):
        raise AssertionError('precision case invariant failed')
    result = implementation(x, y, ignore_index=case.ignore_index, reduction=case.reduction)
    assert_forward_matches(
        result, x_cpu, y_cpu, ignore_index=case.ignore_index, reduction=case.reduction, expected_device=x.device
    )
    if not (torch.equal(x_storage.cpu(), x_base)):
        raise AssertionError('logits or padding modified')
    if not (torch.equal(y_storage.cpu(), y_base)):
        raise AssertionError('targets or guard storage modified')
    if result.numel():
        pointer = result.untyped_storage().data_ptr()
        if not (pointer not in (x_storage.untyped_storage().data_ptr(), y_storage.untyped_storage().data_ptr())):
            raise AssertionError('output aliases an input')
    return result
