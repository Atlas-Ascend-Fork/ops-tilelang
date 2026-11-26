"""Public Ascend interfaces for image embedding operators."""

from __future__ import annotations

import torch

from cann_ops_tilelang.config import get_num_vector_cores, require_ascend

from .image_embed_ops_kernel import (
    _bmp_to_patches_tl_ascend,
    _prepare_image_attn_pos_id_tl_ascend,
    _replace_input_embeds_tl_ascend,
    _shuffle_image_tokens_gather_tl_ascend,
    _shuffle_image_tokens_scatter_tl_ascend,
)


def _validate_replace_input_embeds_inputs(
    input_embeds: torch.Tensor,  # [num_tokens, hidden]
    input_ids: torch.Tensor,  # [num_tokens]
    images_embeds: torch.Tensor,  # [img_tokens, hidden]
    image_token_id: int,
    image_pad_token_ids: torch.Tensor,  # [num_pad_tokens]
    image_pad_embeds: torch.Tensor,  # [num_pad_tokens, hidden]
) -> None:
    tensors = (input_embeds, input_ids, images_embeds, image_pad_token_ids, image_pad_embeds)
    if any(tensor.device.type != 'npu' for tensor in tensors):
        raise ValueError('all inputs must be NPU tensors.')
    if any(tensor.device != input_embeds.device for tensor in tensors[1:]):
        raise ValueError('all inputs must be on the same NPU device.')
    if input_embeds.ndim != 2 or images_embeds.ndim != 2 or image_pad_embeds.ndim != 2:
        raise ValueError('embedding tensors must be two-dimensional.')
    if input_ids.shape != (input_embeds.shape[0],):
        raise ValueError('input_ids must have one entry per input embedding.')
    if images_embeds.shape[1] != input_embeds.shape[1] or image_pad_embeds.shape[1] != input_embeds.shape[1]:
        raise ValueError('all embedding tensors must have the same hidden dimension.')
    if image_pad_token_ids.ndim != 1 or image_pad_embeds.shape[0] != image_pad_token_ids.shape[0]:
        raise ValueError('image pad ids and embeddings must contain the same number of entries.')
    if image_pad_token_ids.shape[0] > 8:
        raise ValueError('at most eight image pad token types are supported.')
    if any(tensor.dtype != torch.bfloat16 for tensor in (input_embeds, images_embeds, image_pad_embeds)):
        raise TypeError('embedding tensors must use torch.bfloat16.')
    if input_ids.dtype != torch.int32 or image_pad_token_ids.dtype != torch.int32:
        raise TypeError('token id tensors must use torch.int32.')


def _replace_input_embeds_impl(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    images_embeds: torch.Tensor,
    image_token_id: int,
    image_pad_token_ids: torch.Tensor,
    image_pad_embeds: torch.Tensor,
) -> None:
    _validate_replace_input_embeds_inputs(
        input_embeds, input_ids, images_embeds, image_token_id, image_pad_token_ids, image_pad_embeds
    )
    require_ascend()
    num_cores = get_num_vector_cores()

    # inclusive cumsum == src_row + 1 for image tokens (kernel-side encoding);
    # one eq + one cumsum instead of four host kernels
    image_token_offset = torch.cumsum(input_ids == image_token_id, dim=0, dtype=torch.int32)
    num_tokens_val = input_embeds.shape[0]
    rows_per_core = (num_tokens_val + num_cores - 1) // num_cores
    kernel = _replace_input_embeds_tl_ascend
    kernel(input_embeds.shape[-1], num_cores=num_cores, rows_per_core=rows_per_core)(
        input_embeds,
        input_ids,
        images_embeds,
        image_token_id,
        image_token_offset,
        image_pad_token_ids,
        image_pad_embeds,
    )


def _validate_shuffle_image_tokens_inputs(
    input_embeds: torch.Tensor,  # [num_tokens, hidden]
    image_real_offsets: torch.Tensor,  # [num_imgs], int32
    image_real_sizes: torch.Tensor,  # [num_imgs], int32
    image_ids: torch.Tensor,  # [img_tokens], int32
) -> None:
    tensors = (input_embeds, image_real_offsets, image_real_sizes, image_ids)
    if any(tensor.device.type != 'npu' for tensor in tensors):
        raise ValueError('all inputs must be NPU tensors.')
    if any(tensor.device != input_embeds.device for tensor in tensors[1:]):
        raise ValueError('all inputs must be on the same NPU device.')
    if input_embeds.ndim != 2 or input_embeds.dtype != torch.bfloat16:
        raise TypeError('input_embeds must be a two-dimensional bfloat16 tensor.')
    if any(tensor.ndim != 1 or tensor.dtype != torch.int32 for tensor in tensors[1:]):
        raise TypeError('image metadata must be one-dimensional int32 tensors.')
    if image_real_offsets.shape != image_real_sizes.shape:
        raise ValueError('image_real_offsets and image_real_sizes must have the same shape.')


def _shuffle_image_tokens_impl(
    input_embeds: torch.Tensor,
    image_real_offsets: torch.Tensor,
    image_real_sizes: torch.Tensor,
    image_ids: torch.Tensor,
) -> None:
    _validate_shuffle_image_tokens_inputs(input_embeds, image_real_offsets, image_real_sizes, image_ids)
    require_ascend()
    img_tokens = image_ids.shape[0]
    hidden = input_embeds.shape[-1]
    num_cores = get_num_vector_cores()

    # Precompute prefix sum of image_real_sizes as starting indices in the flat image token array
    img_token_starts = torch.zeros_like(image_real_sizes)
    if image_real_sizes.shape[0] > 1:
        img_token_starts[1:] = torch.cumsum(image_real_sizes[:-1], dim=0, dtype=torch.int32)

    # Allocate temporary buffer for the gather-scatter pattern
    temp_buffer = torch.empty((img_tokens, hidden), dtype=input_embeds.dtype, device=input_embeds.device)

    # Phase 1: Gather from shuffled source positions into temp_buffer
    _shuffle_image_tokens_gather_tl_ascend(hidden, num_cores)(
        input_embeds,
        image_real_offsets,
        img_token_starts,
        temp_buffer,
    )
    _shuffle_image_tokens_scatter_tl_ascend(hidden, num_cores)(
        input_embeds,
        image_real_offsets,
        image_real_sizes,
        img_token_starts,
        temp_buffer,
    )


def _validate_prepare_image_attn_pos_id_inputs(
    images_spatial_crop: torch.Tensor,  # [num_imgs, 2], int32 or int64
    cu_seqlens_vit: torch.Tensor,  # [num_imgs + 1], int32
    total_seqlens_vit: int,
    win_size: int,
) -> None:
    if images_spatial_crop.device.type != 'npu' or cu_seqlens_vit.device.type != 'npu':
        raise ValueError('image metadata must be NPU tensors.')
    if images_spatial_crop.device != cu_seqlens_vit.device:
        raise ValueError('image metadata must be on the same NPU device.')
    if images_spatial_crop.ndim != 2 or images_spatial_crop.shape[1] != 2:
        raise ValueError('images_spatial_crop must have shape (num_images, 2).')
    if cu_seqlens_vit.shape != (images_spatial_crop.shape[0] + 1,):
        raise ValueError('cu_seqlens_vit must have shape (num_images + 1,).')
    if images_spatial_crop.dtype not in (torch.int32, torch.int64) or cu_seqlens_vit.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise TypeError('image metadata must use torch.int32 or torch.int64.')
    if total_seqlens_vit < 0:
        raise ValueError('total_seqlens_vit must be non-negative.')
    if win_size < 1:
        raise ValueError(f'win_size must be >= 1, got {win_size}')


def _prepare_image_attn_pos_id_impl(
    images_spatial_crop: torch.Tensor,
    cu_seqlens_vit: torch.Tensor,
    total_seqlens_vit: int,
    win_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_prepare_image_attn_pos_id_inputs(images_spatial_crop, cu_seqlens_vit, total_seqlens_vit, win_size)
    require_ascend()
    images_spatial_crop_i32 = images_spatial_crop.to(dtype=torch.int32)
    cu_seqlens_vit_i32 = cu_seqlens_vit.to(dtype=torch.int32)

    device = images_spatial_crop.device
    images_global_attn_pos_id = torch.empty((total_seqlens_vit, 2), dtype=torch.int32, device=device)
    images_window_attn_pos_id = torch.empty((total_seqlens_vit, 2), dtype=torch.int32, device=device)

    num_imgs = int(images_spatial_crop.shape[0])
    if num_imgs == 1:
        pattern_id = 'single'
    elif total_seqlens_vit <= num_imgs * 4096 or num_imgs >= 224:
        pattern_id = 'owner'
    else:
        pattern_id = 'flat'
    owner_mode = ('win1' if win_size == 1 else 'window') if pattern_id == 'owner' else 'unused'
    kernel = _prepare_image_attn_pos_id_tl_ascend(pattern_id, owner_mode, get_num_vector_cores())
    kernel(
        images_spatial_crop_i32,
        cu_seqlens_vit_i32,
        images_global_attn_pos_id,
        images_window_attn_pos_id,
        win_size,
    )
    return images_global_attn_pos_id, images_window_attn_pos_id


def _validate_bmp_to_patches_inputs(
    image_raw_concat: torch.Tensor,
    metadata: tuple[torch.Tensor, ...],
    total_patches: int,
    patch_size: int,
    mean: float,
    std: float,
) -> None:
    cu_raw_bytes, images_spatial_crop, cu_seqlens_vit, row_strides, flip_flags = metadata
    tensors = (image_raw_concat, *metadata)
    if any(tensor.device.type != 'npu' for tensor in tensors):
        raise ValueError('all inputs must be NPU tensors.')
    if any(tensor.device != image_raw_concat.device for tensor in tensors[1:]):
        raise ValueError('all inputs must be on the same NPU device.')
    if image_raw_concat.ndim != 1 or image_raw_concat.dtype != torch.uint8:
        raise TypeError('image_raw_concat must be a one-dimensional uint8 tensor.')
    if any(tensor.dtype != torch.int32 for tensor in tensors[1:]):
        raise TypeError('BMP metadata must use torch.int32.')
    num_imgs = images_spatial_crop.shape[0]
    if images_spatial_crop.shape != (num_imgs, 2):
        raise ValueError('images_spatial_crop must have shape (num_images, 2).')
    if cu_raw_bytes.shape != (num_imgs + 1,) or cu_seqlens_vit.shape != (num_imgs + 1,):
        raise ValueError('prefix sum tensors must have shape (num_images + 1,).')
    if row_strides.shape != (num_imgs,) or flip_flags.shape != (num_imgs,):
        raise ValueError('row_strides and flip_flags must have shape (num_images,).')
    if total_patches < 0 or patch_size <= 0 or std <= 0:
        raise ValueError('total_patches must be non-negative, patch_size and std must be positive.')


def _bmp_to_patches_impl(
    image_raw_concat: torch.Tensor,
    cu_raw_bytes: torch.Tensor,
    images_spatial_crop: torch.Tensor,
    cu_seqlens_vit: torch.Tensor,
    row_strides: torch.Tensor,
    flip_flags: torch.Tensor,
    *,
    total_patches: int,
    patch_size: int,
    mean: float,
    std: float,
) -> torch.Tensor:
    _validate_bmp_to_patches_inputs(
        image_raw_concat,
        (cu_raw_bytes, images_spatial_crop, cu_seqlens_vit, row_strides, flip_flags),
        total_patches,
        patch_size,
        mean,
        std,
    )
    require_ascend()
    output = torch.empty(
        (total_patches, 3, patch_size, patch_size), dtype=torch.bfloat16, device=image_raw_concat.device
    )
    kernel = _bmp_to_patches_tl_ascend(patch_size, get_num_vector_cores())
    kernel_output = output.view(total_patches, 3 * patch_size * patch_size)
    kernel_args = (
        image_raw_concat,
        cu_raw_bytes,
        images_spatial_crop.to(dtype=torch.int32),
        cu_seqlens_vit.to(dtype=torch.int32),
        row_strides,
        flip_flags,
        mean,
        std,
    )
    # The FP16 grouped path is safe while all uint8 inputs normalize into
    # this range. The 1/16 headroom below 2.0 bounds rounding to one BF16 ULP.
    norm_bound = 1.9375 * std
    fp16_norm_safe = int(std > 0.0 and -norm_bound <= mean <= norm_bound and -norm_bound <= 1.0 - mean <= norm_bound)
    kernel(*kernel_args, fp16_norm_safe, kernel_output)
    return output


@torch.library.custom_op('cann_ops_tilelang::replace_input_embeds', mutates_args=('input_embeds',))
def replace_input_embeds(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    images_embeds: torch.Tensor,
    image_token_id: int,
    image_pad_token_ids: torch.Tensor,
    image_pad_embeds: torch.Tensor,
) -> None:
    """Replace image and pad token embeddings in ``input_embeds`` in place."""
    return _replace_input_embeds_impl(
        input_embeds, input_ids, images_embeds, image_token_id, image_pad_token_ids, image_pad_embeds
    )


@replace_input_embeds.register_fake
def _replace_input_embeds_fake(
    input_embeds, input_ids, images_embeds, image_token_id, image_pad_token_ids, image_pad_embeds
):
    _validate_replace_input_embeds_inputs(
        input_embeds, input_ids, images_embeds, image_token_id, image_pad_token_ids, image_pad_embeds
    )


@torch.library.custom_op('cann_ops_tilelang::shuffle_image_tokens', mutates_args=('input_embeds',))
def shuffle_image_tokens(
    input_embeds: torch.Tensor,
    image_real_offsets: torch.Tensor,
    image_real_sizes: torch.Tensor,
    image_ids: torch.Tensor,
) -> None:
    """Interleave each image's embedding rows in ``input_embeds`` in place."""
    return _shuffle_image_tokens_impl(input_embeds, image_real_offsets, image_real_sizes, image_ids)


@shuffle_image_tokens.register_fake
def _shuffle_image_tokens_fake(input_embeds, image_real_offsets, image_real_sizes, image_ids):
    _validate_shuffle_image_tokens_inputs(input_embeds, image_real_offsets, image_real_sizes, image_ids)


@torch.library.custom_op('cann_ops_tilelang::prepare_image_attn_pos_id', mutates_args=())
def prepare_image_attn_pos_id(
    images_spatial_crop: torch.Tensor,
    cu_seqlens_vit: torch.Tensor,
    total_seqlens_vit: int,
    win_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global and window position IDs for image attention."""
    return _prepare_image_attn_pos_id_impl(images_spatial_crop, cu_seqlens_vit, total_seqlens_vit, win_size)


@prepare_image_attn_pos_id.register_fake
def _prepare_image_attn_pos_id_fake(images_spatial_crop, cu_seqlens_vit, total_seqlens_vit, win_size):
    _validate_prepare_image_attn_pos_id_inputs(images_spatial_crop, cu_seqlens_vit, total_seqlens_vit, win_size)
    shape = (total_seqlens_vit, 2)
    return (
        torch.empty(shape, dtype=torch.int32, device=images_spatial_crop.device),
        torch.empty(shape, dtype=torch.int32, device=images_spatial_crop.device),
    )


@torch.library.custom_op('cann_ops_tilelang::bmp_to_patches', mutates_args=())
def bmp_to_patches(
    image_raw_concat: torch.Tensor,
    cu_raw_bytes: torch.Tensor,
    images_spatial_crop: torch.Tensor,
    cu_seqlens_vit: torch.Tensor,
    row_strides: torch.Tensor,
    flip_flags: torch.Tensor,
    total_patches: int,
    patch_size: int,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Convert concatenated BMP pixels to normalized BF16 image patches."""
    return _bmp_to_patches_impl(
        image_raw_concat,
        cu_raw_bytes,
        images_spatial_crop,
        cu_seqlens_vit,
        row_strides,
        flip_flags,
        total_patches=total_patches,
        patch_size=patch_size,
        mean=mean,
        std=std,
    )


@bmp_to_patches.register_fake
def _bmp_to_patches_fake(
    image_raw_concat,
    cu_raw_bytes,
    images_spatial_crop,
    cu_seqlens_vit,
    row_strides,
    flip_flags,
    total_patches,
    patch_size,
    mean,
    std,
):
    _validate_bmp_to_patches_inputs(
        image_raw_concat,
        (cu_raw_bytes, images_spatial_crop, cu_seqlens_vit, row_strides, flip_flags),
        total_patches,
        patch_size,
        mean,
        std,
    )
    return torch.empty((total_patches, 3, patch_size, patch_size), dtype=torch.bfloat16, device=image_raw_concat.device)
