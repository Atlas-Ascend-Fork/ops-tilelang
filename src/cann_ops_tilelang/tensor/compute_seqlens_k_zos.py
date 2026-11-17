"""Public interface for the ZOS sequence-length update."""

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .compute_seqlens_k_zos_kernel import get_compute_seqlens_k_zos_kernel


def _validate_nextn(nextn: int) -> None:
    if not isinstance(nextn, int) or isinstance(nextn, bool):
        raise TypeError('nextn must be an integer.')
    if nextn < 0:
        raise ValueError(f'nextn must be non-negative, got {nextn}.')


def _validate_inputs(
    seqlens_k: torch.Tensor,
    seqlens_q: torch.Tensor,
    remap: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    *,
    nextn: int,
) -> None:
    """Validate metadata without inspecting the runtime or tensor data."""
    tensors = (seqlens_k, seqlens_q, remap, prev_num_accepted_tokens)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError('all inputs must be torch.Tensor instances.')
    _validate_nextn(nextn)

    if seqlens_k.ndim != 1:
        raise ValueError('seqlens_k must be a 1D tensor.')
    if seqlens_q.ndim != 1:
        raise ValueError('seqlens_q must be a 1D tensor.')
    if remap.ndim != 1:
        raise ValueError('remap must be a 1D tensor.')
    if prev_num_accepted_tokens.ndim != 1:
        raise ValueError('prev_num_accepted_tokens must be a 1D tensor.')
    if seqlens_k.numel() == 0:
        raise ValueError('n_seq must be positive.')
    if prev_num_accepted_tokens.numel() == 0:
        raise ValueError('n_prev_seq must be positive.')
    if seqlens_q.shape != seqlens_k.shape:
        raise ValueError('seqlens_q must match seqlens_k.')
    if remap.shape != seqlens_k.shape:
        raise ValueError('remap must match seqlens_k.')
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError('all inputs must use torch.int32.')
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError('all inputs must be contiguous.')
    if any(tensor.device != seqlens_k.device for tensor in tensors):
        raise ValueError('all inputs must be on the same device.')
    if seqlens_k.device.type != 'npu':
        raise ValueError(f'all inputs must be NPU tensors, got {seqlens_k.device}.')


def _compute_seqlens_k_zos_impl(
    seqlens_k: torch.Tensor,
    seqlens_q: torch.Tensor,
    remap: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    *,
    nextn: int,
) -> None:
    _validate_inputs(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens, nextn=nextn)
    require_ascend()
    num_blocks = (seqlens_k.numel() + 127) // 128
    num_cores = min(1 << (num_blocks - 1).bit_length(), get_num_vector_cores())
    kernel = get_compute_seqlens_k_zos_kernel(nextn, num_cores)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens)


@torch.library.custom_op('cann_ops_tilelang::compute_seqlens_k_zos', mutates_args=('seqlens_k',))
def compute_seqlens_k_zos(
    seqlens_k: torch.Tensor,
    seqlens_q: torch.Tensor,
    remap: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    *,
    nextn: int,
) -> None:
    """Update ``seqlens_k`` in place for a ZOS decode step on Ascend.

    Only ``seqlens_k`` is modified; the other tensors are read-only. The return
    value is always ``None``. All tensors must be non-empty, contiguous 1D int32
    tensors on the same NPU, with the first three tensors having equal shapes.

    Supports fixed-shape ``torch.compile(..., backend='npugraph_ex',
    fullgraph=True, dynamic=False)`` with a compatible torch_npu. The custom op
    declares the mutation and its fake implementation only validates metadata.
    ``nextn`` must be a non-negative integer.
    """
    _compute_seqlens_k_zos_impl(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens, nextn=nextn)


@compute_seqlens_k_zos.register_fake
def _compute_seqlens_k_zos_fake(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens, *, nextn):
    _validate_inputs(seqlens_k, seqlens_q, remap, prev_num_accepted_tokens, nextn=nextn)
    return None


def compute_seqlens_k_zos_ref(
    seqlens_k: torch.Tensor,
    seqlens_q: torch.Tensor,
    remap: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    *,
    nextn: int,
) -> None:
    """Reference implementation of the in-place ZOS sequence-length update."""
    seqlens_k += seqlens_q
    mask = remap != -1
    remap_clamped = remap.clamp(min=0)
    accepted = prev_num_accepted_tokens[remap_clamped.long()]
    adjustment = torch.where(mask, nextn - accepted, torch.zeros_like(seqlens_k))
    seqlens_k -= adjustment
