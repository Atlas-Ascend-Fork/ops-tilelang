"""Public Ascend interface for prompt token log probabilities."""

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .prompt_logprobs_kernel import _get_prompt_logprobs_tensor_tl_ascend


@torch.library.custom_op('cann_ops_tilelang::get_prompt_logprobs_tensor', mutates_args=())
def get_prompt_logprobs_tensor(
    prompt_tokens: torch.Tensor,
    logprobs: torch.Tensor,
    token_index: torch.Tensor,
) -> torch.Tensor:
    """Find each prompt token's log probability among retained token ids."""
    return _get_prompt_logprobs_tensor_impl(prompt_tokens, logprobs, token_index)


def _validate_get_prompt_logprobs_tensor(prompt_tokens, logprobs, token_index):
    for name, tensor in (('prompt_tokens', prompt_tokens), ('logprobs', logprobs), ('token_index', token_index)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f'{name} must be a torch.Tensor.')
    if (
        prompt_tokens.device.type != 'npu'
        or logprobs.device != prompt_tokens.device
        or token_index.device != prompt_tokens.device
    ):
        raise ValueError('all inputs must be on the same NPU device.')
    if prompt_tokens.ndim != 1 or logprobs.ndim != 2 or token_index.ndim != 2:
        raise ValueError('prompt_tokens must be 1D; logprobs and token_index must be 2D.')
    if logprobs.shape != token_index.shape or logprobs.shape[0] != prompt_tokens.shape[0]:
        raise ValueError('input shapes are inconsistent.')
    if prompt_tokens.dtype != torch.int32 or token_index.dtype != torch.int32:
        raise TypeError('prompt_tokens and token_index must use torch.int32.')
    if logprobs.dtype != torch.float32:
        raise TypeError('logprobs must use torch.float32.')
    if logprobs.shape[1] == 0:
        raise ValueError('topk count must be positive.')
    if logprobs.shape[1] > 47016:
        raise ValueError('topk count must be at most 47016.')
    if not prompt_tokens.is_contiguous():
        raise ValueError('prompt_tokens must be contiguous.')
    if logprobs.stride(1) != 1 or token_index.stride(1) != 1:
        raise ValueError('logprobs and token_index must have unit inner stride.')


def _get_prompt_logprobs_tensor_impl(
    prompt_tokens: torch.Tensor, logprobs: torch.Tensor, token_index: torch.Tensor
) -> torch.Tensor:
    require_ascend()
    _validate_get_prompt_logprobs_tensor(prompt_tokens, logprobs, token_index)
    output = torch.empty((logprobs.shape[0],), dtype=logprobs.dtype, device=logprobs.device)
    if prompt_tokens.numel() == 0:
        return output
    num_cores = get_num_vector_cores()
    kernel = _get_prompt_logprobs_tensor_tl_ascend(logprobs.shape[1], num_cores)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(prompt_tokens, logprobs, token_index, output)
    return output


@get_prompt_logprobs_tensor.register_fake
def _get_prompt_logprobs_tensor_fake(prompt_tokens, logprobs, token_index):
    _validate_get_prompt_logprobs_tensor(prompt_tokens, logprobs, token_index)
    return torch.empty((logprobs.shape[0],), dtype=logprobs.dtype, device=logprobs.device)
