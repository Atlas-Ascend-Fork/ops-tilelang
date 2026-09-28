"""Public Ascend interface for copying tensor slices into independent tensors."""

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend, should_print_kernel_source

from .unbind_kernel import _unbind_tl_ascend, _unbind_tl_ascend_small


def unbind(from_: torch.Tensor) -> list[torch.Tensor]:
    """Copy ``from_.unbind(0)`` into independent NPU tensors."""
    if not isinstance(from_, torch.Tensor):
        raise TypeError('from_ must be a torch.Tensor.')
    require_ascend()
    if from_.device.type != 'npu':
        raise ValueError('from_ must be an NPU tensor.')
    if from_.ndim == 0 or from_.shape[0] == 0:
        raise ValueError('from_ must have a non-empty leading dimension.')

    req_cnt = from_.shape[0]
    target = from_[0]
    if not target.is_contiguous() or from_.stride(-1) != 1 or (from_.dim() == 1 and not from_.is_contiguous()):
        from_ = from_.contiguous()
        target = from_[0]

    out = [torch.empty(target.shape, dtype=from_.dtype, device=from_.device) for _ in range(req_cnt)]
    nbytes = target.numel() * target.element_size()
    from_bytes = from_.view(torch.int8).reshape(req_cnt, nbytes)
    out_ptrs = torch.tensor([tensor.data_ptr() for tensor in out], dtype=torch.int64, device='cpu').to(
        from_.device, non_blocking=True
    )

    if nbytes == 4096 and req_cnt in (64, 256, 512):
        num_stages = 3 if req_cnt == 64 else 4
        requests_per_task = {64: 1, 256: 2, 512: 4}[req_cnt]
        kernel = _unbind_tl_ascend_small(
            from_bytes.stride(0),
            num_stages,
            get_num_vector_cores(),
            req_cnt,
            requests_per_task,
        )
    else:
        kernel = _unbind_tl_ascend(from_bytes.stride(0))
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(from_bytes, out_ptrs)
    return out
