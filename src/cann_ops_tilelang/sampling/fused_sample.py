from __future__ import annotations

import torch

from cann_ops_tilelang.config import (
    get_num_vector_cores,
    require_ascend,
    should_print_kernel_source,
)

from .fused_sample_kernel import (
    _fused_sample_tl_ascend,
    _fused_sample_tl_ascend_inrow,
)

MTP_SEED_OFFSET = 1000000007


def _validate_fused_sample_inputs(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_p: torch.Tensor | None,
    ftp: torch.Tensor | None,
    sample_positions: torch.Tensor | None,
    seq_seeds: torch.Tensor | None,
    logprobs_out: torch.Tensor | None,
    last_nextn_token_logprobs: torch.Tensor | None,
    last_next_n_token_ids: torch.Tensor | None,
) -> None:
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError('logits must be a two-dimensional float32 tensor')
    if logits.stride(1) != 1:
        raise ValueError('the last dimension of logits must be contiguous')
    if logits.device.type != 'npu':
        raise ValueError('fused_sample only supports Ascend NPU tensors')

    req_cnt = ftp.shape[0] if ftp is not None else logits.shape[0]
    if temperatures.shape != (req_cnt,) or temperatures.dtype != torch.float32:
        raise ValueError('temperatures must be a float32 tensor with one value per request')
    if temperatures.stride(0) <= 0:
        raise ValueError('temperatures must have a positive stride')

    if top_p is not None:
        if top_p.shape != temperatures.shape or top_p.dtype != torch.float32:
            raise ValueError('top_p must match temperatures and use float32')
        if top_p.stride(0) <= 0:
            raise ValueError('top_p must have a positive stride')
    if ftp is not None and (ftp.shape != temperatures.shape or ftp.dtype != torch.int32 or not ftp.is_contiguous()):
        raise ValueError('ftp must be a contiguous int32 tensor with one value per request')
    if sample_positions is not None:
        if (
            sample_positions.shape != temperatures.shape
            or sample_positions.dtype != torch.int32
            or not sample_positions.is_contiguous()
        ):
            raise ValueError('sample_positions must be a contiguous int32 tensor with one value per request')
        if seq_seeds is None or (
            seq_seeds.shape != temperatures.shape or seq_seeds.dtype != torch.uint64 or not seq_seeds.is_contiguous()
        ):
            raise ValueError('seq_seeds must be a contiguous uint64 tensor with one value per request')
    if logprobs_out is not None and (
        logprobs_out.ndim != 2 or logprobs_out.dtype != torch.float32 or logprobs_out.stride(1) != 1
    ):
        raise ValueError('logprobs_out must be a two-dimensional float32 tensor')
    if last_nextn_token_logprobs is not None:
        if last_nextn_token_logprobs.dtype != torch.float32 or last_nextn_token_logprobs.stride(-1) != 1:
            raise ValueError('last_nextn_token_logprobs must use float32 with a contiguous last dimension')
        if last_next_n_token_ids is None or (
            last_next_n_token_ids.dtype != torch.int32 or not last_next_n_token_ids.is_contiguous()
        ):
            raise ValueError('last_next_n_token_ids must be a contiguous int32 tensor')

    tensors = (
        temperatures,
        top_p,
        ftp,
        sample_positions,
        seq_seeds,
        logprobs_out,
        last_nextn_token_logprobs,
        last_next_n_token_ids,
    )
    if any(tensor is not None and tensor.device != logits.device for tensor in tensors):
        raise ValueError('all inputs must be on the same NPU device')


def fused_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_p: torch.Tensor | None = None,
    *,
    tag: str = 'default',
    ftp: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    sample_positions: torch.Tensor | None = None,
    seq_seeds: torch.Tensor | None = None,
    logprobs_out: torch.Tensor | None = None,
    last_nextn_token_logprobs: torch.Tensor | None = None,
    last_next_n_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Apply temperature, optional top-p, log-softmax, and optional sampling."""
    require_ascend()
    _validate_fused_sample_inputs(
        logits,
        temperatures,
        top_p,
        ftp,
        sample_positions,
        seq_seeds,
        logprobs_out,
        last_nextn_token_logprobs,
        last_next_n_token_ids,
    )

    do_top_p = top_p is not None
    return_batch_format = ftp is not None
    do_sample = sample_positions is not None
    do_reject_sampling = last_nextn_token_logprobs is not None
    vocab_size = logits.shape[1]
    if vocab_size <= 0:
        raise ValueError('vocab_size must be positive')

    if return_batch_format:
        if isinstance(max_seqlen_q, bool) or not isinstance(max_seqlen_q, int) or max_seqlen_q <= 0:
            raise ValueError('max_seqlen_q must be a positive integer when ftp is provided')
        req_cnt = ftp.shape[0]
        if req_cnt <= 0:
            raise ValueError('ftp must contain at least one request')
        num_blocks = max_seqlen_q * req_cnt
        if logits.shape[0] > num_blocks:
            raise ValueError('the packed logits rows exceed max_seqlen_q * req_cnt')
    else:
        req_cnt = logits.shape[0]
        if req_cnt <= 0:
            raise ValueError('logits must contain at least one row')
        num_blocks = req_cnt

    reject_steps = 0
    if do_reject_sampling:
        if not return_batch_format:
            raise ValueError('reject sampling requires ftp')
        if not do_sample:
            raise ValueError('reject sampling requires sample_positions')
        if last_next_n_token_ids is None:
            raise ValueError('last_next_n_token_ids is required for reject sampling')
        reject_steps = max_seqlen_q - 1
        if last_nextn_token_logprobs.shape != (reject_steps, req_cnt, vocab_size):
            raise ValueError('last_nextn_token_logprobs has an invalid shape')
        if last_next_n_token_ids.shape != (reject_steps, req_cnt):
            raise ValueError('last_next_n_token_ids has an invalid shape')

    if do_sample and seq_seeds is None:
        raise ValueError('seq_seeds is required when sample_positions is provided')

    vec_cores = get_num_vector_cores()
    use_inrow = (
        vocab_size >= 65536 and do_top_p and return_batch_format and do_sample and 0 < logits.shape[0] <= vec_cores
    )
    max_vocab_size = 262144 if do_reject_sampling else 524288
    if vocab_size > max_vocab_size:
        raise ValueError(f'vocab_size exceeds the supported limit {max_vocab_size} for this fused_sample path')

    if logprobs_out is None:
        logprobs = torch.empty((num_blocks, vocab_size), dtype=logits.dtype, device=logits.device)
    else:
        if logprobs_out.shape != (num_blocks, vocab_size):
            raise ValueError('logprobs_out has an invalid shape')
        logprobs = logprobs_out
    sample_result = torch.empty((num_blocks,), dtype=torch.int32, device=logits.device) if do_sample else None
    accept_result_verify = (
        torch.empty((reject_steps, req_cnt), dtype=torch.bool, device=logits.device) if do_reject_sampling else None
    )
    sample_reject_cur = (
        torch.empty((reject_steps, req_cnt), dtype=torch.int32, device=logits.device) if do_reject_sampling else None
    )

    seed_offset = MTP_SEED_OFFSET * (0 if tag == 'default' else 1)

    if use_inrow:
        groups_per_row = vec_cores // logits.shape[0]
        inrow_cores = logits.shape[0] * groups_per_row
        kernel = _fused_sample_tl_ascend_inrow(
            vocab_size,
            do_reject_sampling,
            groups_per_row,
            inrow_cores,
        )
        partial_f = torch.empty((logits.shape[0], groups_per_row, 8), dtype=torch.float32, device=logits.device)
        search_f = torch.empty((logits.shape[0], 2, groups_per_row * 3), dtype=torch.float32, device=logits.device)
        row_f = torch.empty((logits.shape[0], 8), dtype=torch.float32, device=logits.device)
        row_i = torch.empty((logits.shape[0], 8), dtype=torch.int32, device=logits.device)
        kernel(
            logits,
            temperatures,
            top_p,
            ftp,
            sample_positions,
            seq_seeds,
            logprobs,
            sample_result,
            last_nextn_token_logprobs,
            last_next_n_token_ids,
            accept_result_verify,
            sample_reject_cur,
            partial_f,
            search_f,
            row_f,
            row_i,
            seed_offset,
        )
    else:
        kernel = _fused_sample_tl_ascend(
            vocab_size,
            do_top_p,
            return_batch_format,
            do_sample or do_reject_sampling,
            do_reject_sampling,
            min(vec_cores, max(num_blocks, 1)),
        )
        kernel(
            logits,
            temperatures,
            top_p,
            ftp,
            sample_positions,
            seq_seeds,
            logprobs,
            sample_result,
            last_nextn_token_logprobs,
            last_next_n_token_ids,
            accept_result_verify,
            sample_reject_cur,
            seed_offset,
        )

    if should_print_kernel_source():
        print(kernel.get_kernel_source())

    if return_batch_format:
        logprobs = logprobs.view(max_seqlen_q, req_cnt, vocab_size)
        if sample_result is not None:
            sample_result = sample_result.view(max_seqlen_q, req_cnt)
    return logprobs, sample_result, accept_result_verify, sample_reject_cur
