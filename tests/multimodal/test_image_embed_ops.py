from collections.abc import Callable
from typing import Any

import pytest
import torch

from cann_ops_tilelang.config import get_device
from cann_ops_tilelang.multimodal.image_embed_ops import (
    bmp_to_patches,
    prepare_image_attn_pos_id,
    replace_input_embeds,
    shuffle_image_tokens,
)

pytestmark = pytest.mark.npu


def replace_input_embeds_ref(
    input_embeds: torch.Tensor,  # [num_tokens, hidden]
    input_ids: torch.Tensor,  # [num_tokens]
    images_embeds: torch.Tensor,  # [img_tokens, hidden]
    image_token_id: int,
    image_pad_token_ids: torch.Tensor,  # [num_pad_tokens]
    image_pad_embeds: torch.Tensor,  # [num_pad_tokens, hidden]
) -> None:
    """Pure PyTorch reference implementation for replace_input_embeds."""
    # Replace image tokens
    image_mask = input_ids == image_token_id
    if image_mask.any():
        img_count = image_mask.sum().item()
        input_embeds[image_mask] = images_embeds[:img_count]

    # Replace pad tokens
    for i in range(image_pad_token_ids.shape[0]):
        pad_mask = input_ids == image_pad_token_ids[i]
        if pad_mask.any():
            input_embeds[pad_mask] = image_pad_embeds[i]


def shuffle_image_tokens_ref(
    input_embeds: torch.Tensor,  # [num_tokens, hidden]
    image_real_offsets: torch.Tensor,  # [num_imgs]
    image_real_sizes: torch.Tensor,  # [num_imgs]
) -> None:
    """Pure PyTorch reference implementation for shuffle_image_tokens."""
    num_imgs = image_real_offsets.shape[0]
    for i in range(num_imgs):
        offset = image_real_offsets[i].item()
        size = image_real_sizes[i].item()
        range_tensor = torch.arange(size, device=input_embeds.device) + offset
        range_tensor = range_tensor.reshape(2, -1).transpose(0, 1).reshape(-1)
        input_embeds[offset : offset + size] = input_embeds[range_tensor]


def build_image_data(
    num_tokens: int,
    hidden: int,
    num_imgs: int,
    img_size_per_image: int,
    num_pad_tokens: int = 3,
    vocab_size: int = 129280,
) -> dict:
    """Build test data for image embed ops.

    Args:
        num_tokens: total number of tokens in input_embeds
        hidden: hidden dimension
        num_imgs: number of images
        img_size_per_image: number of image tokens per image (must be even for shuffle)
        num_pad_tokens: number of pad token types
        vocab_size: vocabulary size of the model, used to derive special image token IDs
    Returns:
        dict with all tensors needed for testing
    """
    total_img_tokens = num_imgs * img_size_per_image

    # Ensure we have enough tokens for images and pad tokens
    assert num_tokens >= total_img_tokens + num_pad_tokens * num_imgs, (
        'num_tokens must be large enough to hold all image tokens and pad tokens'
    )

    device = get_device()
    input_embeds = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)

    # Build input_ids: place image tokens and pad tokens at specific positions
    image_token_id = vocab_size + 1
    pad_token_ids = [
        vocab_size + 2,
        vocab_size + 3,
        vocab_size + 4,
    ][:num_pad_tokens]  # newline, start, end

    input_ids = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    # Place image tokens and pad tokens contiguously for each image
    # Layout per image: [pad_token_0, img_token * img_size_per_image, pad_token_1, pad_token_2]
    # But to keep it simple: place image tokens first, then pad tokens
    pos = 0
    image_real_offsets = []
    image_real_sizes = []
    for _ in range(num_imgs):
        # One pad token before image
        input_ids[pos] = pad_token_ids[0]
        pos += 1
        # Image tokens
        image_real_offsets.append(pos)
        image_real_sizes.append(img_size_per_image)
        for _ in range(img_size_per_image):
            input_ids[pos] = image_token_id
            pos += 1
        # Remaining pad tokens after image
        for p in range(1, num_pad_tokens):
            input_ids[pos] = pad_token_ids[p]
            pos += 1

    images_embeds = torch.randn(total_img_tokens, hidden, dtype=torch.bfloat16, device=device)

    image_pad_token_ids = torch.tensor(pad_token_ids, dtype=torch.int32, device=device)
    image_pad_embeds = torch.randn(num_pad_tokens, hidden, dtype=torch.bfloat16, device=device)

    image_real_offsets_t = torch.tensor(image_real_offsets, dtype=torch.int32, device=device)
    image_real_sizes_t = torch.tensor(image_real_sizes, dtype=torch.int32, device=device)

    # Build image_ids: for each image, image_real_sizes[i] entries of value i
    image_ids_list = []
    for i in range(num_imgs):
        image_ids_list.append(torch.full((image_real_sizes[i],), i, dtype=torch.int32, device=device))
    image_ids = torch.cat(image_ids_list)

    return {
        'input_embeds': input_embeds,
        'input_ids': input_ids,
        'images_embeds': images_embeds,
        'image_token_id': image_token_id,
        'image_pad_token_ids': image_pad_token_ids,
        'image_pad_embeds': image_pad_embeds,
        'image_real_offsets': image_real_offsets_t,
        'image_real_sizes': image_real_sizes_t,
        'image_ids': image_ids,
    }


@pytest.mark.parametrize('num_tokens', [1024, 4096, 16384])
def test_replace_input_embeds(num_tokens: int) -> None:
    hidden = 4096
    num_pad_tokens = 3
    num_imgs = 2
    img_size_per_image = (num_tokens - num_imgs * num_pad_tokens) // num_imgs
    img_size_per_image = max(img_size_per_image, 2)
    img_size_per_image = img_size_per_image - (img_size_per_image % 2)

    data = build_image_data(num_tokens, hidden, num_imgs, img_size_per_image, num_pad_tokens)

    embeds_ref = data['input_embeds'].clone()
    embeds_kernel = data['input_embeds'].clone()

    replace_input_embeds_ref(
        embeds_ref,
        data['input_ids'],
        data['images_embeds'],
        data['image_token_id'],
        data['image_pad_token_ids'],
        data['image_pad_embeds'],
    )
    replace_input_embeds(
        embeds_kernel,
        data['input_ids'],
        data['images_embeds'],
        data['image_token_id'],
        data['image_pad_token_ids'],
        data['image_pad_embeds'],
    )

    assert torch.equal(embeds_kernel, embeds_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('num_tokens', [1024, 4096, 16384])
def test_replace_input_embeds_benchmark(
    num_tokens: int, benchmark_record: Callable[..., Any], benchmark_timer: Callable[..., Any]
) -> None:
    hidden = 4096
    num_pad_tokens = 3
    num_imgs = 2
    img_size_per_image = (num_tokens - num_imgs * num_pad_tokens) // num_imgs
    img_size_per_image = max(img_size_per_image, 2)
    img_size_per_image = img_size_per_image - (img_size_per_image % 2)

    data = build_image_data(num_tokens, hidden, num_imgs, img_size_per_image, num_pad_tokens)

    def fn():
        replace_input_embeds(
            data['input_embeds'].clone(),
            data['input_ids'],
            data['images_embeds'],
            data['image_token_id'],
            data['image_pad_token_ids'],
            data['image_pad_embeds'],
        )

    t_us = benchmark_timer(fn)
    benchmark_record(
        kernel='replace_input_embeds',
        operation='fwd',
        params={'num_tokens': num_tokens, 'hidden': hidden},
        time_us=t_us,
    )


# Functional branches: None (data-driven). Cover small/med/large shapes.
@pytest.mark.parametrize('num_tokens', [1024, 4096, 16384])
def test_shuffle_image_tokens(num_tokens: int) -> None:
    hidden = 4096
    num_imgs = 2
    num_pad_tokens = 3
    img_size_per_image = (num_tokens - num_imgs * num_pad_tokens) // num_imgs
    img_size_per_image = max(img_size_per_image, 2)
    img_size_per_image = img_size_per_image - (img_size_per_image % 2)

    data = build_image_data(num_tokens, hidden, num_imgs, img_size_per_image, num_pad_tokens)

    embeds_ref = data['input_embeds'].clone()
    embeds_kernel = data['input_embeds'].clone()

    shuffle_image_tokens_ref(embeds_ref, data['image_real_offsets'], data['image_real_sizes'])
    shuffle_image_tokens(embeds_kernel, data['image_real_offsets'], data['image_real_sizes'], data['image_ids'])

    assert torch.equal(embeds_kernel, embeds_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('num_tokens', [1024, 4096, 16384])
def test_shuffle_image_tokens_benchmark(
    num_tokens: int, benchmark_record: Callable[..., Any], benchmark_timer: Callable[..., Any]
) -> None:
    hidden = 4096
    num_imgs = 2
    num_pad_tokens = 3
    img_size_per_image = (num_tokens - num_imgs * num_pad_tokens) // num_imgs
    img_size_per_image = max(img_size_per_image, 2)
    img_size_per_image = img_size_per_image - (img_size_per_image % 2)

    data = build_image_data(num_tokens, hidden, num_imgs, img_size_per_image, num_pad_tokens)

    def fn():
        shuffle_image_tokens(
            data['input_embeds'].clone(),
            data['image_real_offsets'],
            data['image_real_sizes'],
            data['image_ids'],
        )

    t_us = benchmark_timer(fn)
    benchmark_record(
        kernel='shuffle_image_tokens',
        operation='fwd',
        params={'num_tokens': num_tokens, 'hidden': hidden},
        time_us=t_us,
    )


def prepare_image_attn_pos_id_ref(
    images_spatial_crop: torch.Tensor,  # [num_imgs, 2], [n_vit_w, n_vit_h]
    cu_seqlens_vit: torch.Tensor,  # [num_imgs + 1]
    total_seqlens_vit: int,
    win_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference implementation for prepare_image_attn_pos_id.

    Mirrors the original image-processing logic.
    """
    import math

    num_imgs = images_spatial_crop.shape[0]
    global_list = []
    window_list = []

    for i in range(num_imgs):
        n_vit_w = images_spatial_crop[i, 0].item()
        n_vit_h = images_spatial_crop[i, 1].item()

        # Global attention position IDs
        hpos_id = torch.arange(n_vit_h).unsqueeze(1).expand(-1, n_vit_w)
        wpos_id = torch.arange(n_vit_w).unsqueeze(0).expand(n_vit_h, -1)
        global_pos = torch.stack([hpos_id, wpos_id], dim=-1).reshape(-1, 2)

        # Window attention position IDs
        n_window_h = math.ceil(n_vit_h / win_size)
        n_window_w = math.ceil(n_vit_w / win_size)
        pad_h = (n_window_h * win_size - n_vit_h) // 2
        pad_w = (n_window_w * win_size - n_vit_w) // 2
        dev_h = win_size - pad_h
        dev_w = win_size - pad_w

        hpos_id = torch.arange(n_vit_h)
        hpos_id[hpos_id >= dev_h] = (hpos_id[hpos_id >= dev_h] - dev_h) % win_size
        hpos_id = hpos_id.unsqueeze(1).expand(-1, n_vit_w)
        wpos_id = torch.arange(n_vit_w)
        wpos_id[wpos_id >= dev_w] = (wpos_id[wpos_id >= dev_w] - dev_w) % win_size
        wpos_id = wpos_id.unsqueeze(0).expand(n_vit_h, -1)
        window_pos = torch.stack([hpos_id, wpos_id], dim=-1).reshape(-1, 2)

        global_list.append(global_pos)
        window_list.append(window_pos)

    images_global_attn_pos_id = torch.cat(global_list, dim=0).to(dtype=torch.int32)
    images_window_attn_pos_id = torch.cat(window_list, dim=0).to(dtype=torch.int32)
    assert images_global_attn_pos_id.shape[0] == total_seqlens_vit
    return images_global_attn_pos_id, images_window_attn_pos_id


# Functional branches: win_size=1 (prod config) and dynamic win_size>1.
# Keep several non-production values here to protect the public API contract.
@pytest.mark.parametrize('win_size', [1, 2, 3, 5, 7, 8, 14, 16])
def test_prepare_image_attn_pos_id(win_size: int) -> None:
    device = get_device()
    num_imgs = 50
    spatial_crops = [(54, 54)] * num_imgs

    images_spatial_crop = torch.tensor(spatial_crops, dtype=torch.int32, device=device)
    seqlens = images_spatial_crop[:, 0] * images_spatial_crop[:, 1]
    cu_seqlens_vit = torch.zeros(num_imgs + 1, dtype=torch.int32, device=device)
    cu_seqlens_vit[1:] = seqlens.cumsum(dim=0)
    total_seqlens_vit = cu_seqlens_vit[-1].item()

    ref_global, ref_window = prepare_image_attn_pos_id_ref(
        images_spatial_crop, cu_seqlens_vit, total_seqlens_vit, win_size
    )
    kernel_global, kernel_window = prepare_image_attn_pos_id(
        images_spatial_crop, cu_seqlens_vit, total_seqlens_vit, win_size
    )

    assert torch.equal(kernel_global, ref_global.to(device))
    assert torch.equal(kernel_window, ref_window.to(device))


@pytest.mark.parametrize(
    ('patch_size', 'spatial_crops', 'flip_flags_list'),
    [
        (1, [(1, 1), (2, 3)], [0, 1]),
        (2, [(1, 1), (3, 2), (2, 1)], [0, 1, 0]),
        (2, [(1, 1)] * 50, [image_idx % 2 for image_idx in range(50)]),
        (14, [(9, 5), (4, 3), (7, 2)], [1, 0, 1]),
        (17, [(2, 2), (1, 3), (3, 1)], [1, 0, 1]),
        (32, [(1, 1)], [1]),
        (33, [(1, 1)], [0]),
        (256, [(1, 1)], [1]),
        (14, [(54, 54)], [0]),
    ],
)
def test_bmp_to_patches_ascend(
    patch_size: int, spatial_crops: list[tuple[int, int]], flip_flags_list: list[int]
) -> None:
    """Exercise padding, flips, persistent-core tails, and multi-round SIMT."""
    torch.manual_seed(0)
    mean = 0.48145466
    std = 0.26862954
    num_imgs = len(spatial_crops)

    images_spatial_crop_cpu = torch.tensor(spatial_crops, dtype=torch.int32)
    seqlens = images_spatial_crop_cpu[:, 0] * images_spatial_crop_cpu[:, 1]
    cu_seqlens_vit_cpu = torch.zeros(num_imgs + 1, dtype=torch.int32)
    cu_seqlens_vit_cpu[1:] = seqlens.cumsum(dim=0)
    total_patches = int(cu_seqlens_vit_cpu[-1])

    raw_chunks = []
    cu_raw_bytes_list = [0]
    row_strides_list = []
    for (n_vit_w, n_vit_h), flip_flag in zip(spatial_crops, flip_flags_list, strict=True):
        width = n_vit_w * patch_size
        height = n_vit_h * patch_size
        raw_bytes, row_stride = _build_fake_bmp(height, width, bottom_up=bool(flip_flag))
        pixel_data = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)[54:]
        raw_chunks.append(pixel_data)
        cu_raw_bytes_list.append(cu_raw_bytes_list[-1] + pixel_data.numel())
        row_strides_list.append(row_stride)

    image_raw_concat_cpu = torch.cat(raw_chunks)
    cu_raw_bytes_cpu = torch.tensor(cu_raw_bytes_list, dtype=torch.int32)
    row_strides_cpu = torch.tensor(row_strides_list, dtype=torch.int32)
    flip_flags_cpu = torch.tensor(flip_flags_list, dtype=torch.int32)

    ref_output = bmp_to_patches_ref(
        image_raw_concat_cpu,
        cu_raw_bytes_cpu,
        images_spatial_crop_cpu,
        cu_seqlens_vit_cpu,
        row_strides_cpu,
        flip_flags_cpu,
        total_patches,
        patch_size,
        mean,
        std,
    )

    npu_inputs = [
        tensor.to(get_device())
        for tensor in (
            image_raw_concat_cpu,
            cu_raw_bytes_cpu,
            images_spatial_crop_cpu,
            cu_seqlens_vit_cpu,
            row_strides_cpu,
            flip_flags_cpu,
        )
    ]
    kernel_output = bmp_to_patches(*npu_inputs, total_patches, patch_size, mean, std)

    assert kernel_output.device.type == 'npu'
    torch.testing.assert_close(kernel_output.cpu().float(), ref_output.float(), rtol=0, atol=0.01)


def bmp_to_patches_ref(
    image_raw_concat: torch.Tensor,  # [total_raw_bytes], uint8, on the source device
    cu_raw_bytes: torch.Tensor,  # [num_imgs + 1], int32
    images_spatial_crop: torch.Tensor,  # [num_imgs, 2], [n_vit_w, n_vit_h]
    cu_seqlens_vit: torch.Tensor,  # [num_imgs + 1], int32
    row_strides: torch.Tensor,  # [num_imgs], int32
    flip_flags: torch.Tensor,  # [num_imgs], int32
    total_patches: int,
    patch_size: int,
    mean: float,
    std: float,
) -> torch.Tensor:
    """Pure PyTorch reference implementation for bmp_to_patches.

    Mirrors the original load_image_from_bmp loop (ve_server.py lines 109-156).
    """
    num_imgs = images_spatial_crop.shape[0]
    result_tensors = []

    for i in range(num_imgs):
        n_vit_w = images_spatial_crop[i, 0].item()
        n_vit_h = images_spatial_crop[i, 1].item()
        best_width = n_vit_w * patch_size
        best_height = n_vit_h * patch_size
        row_stride = row_strides[i].item()
        do_flip = flip_flags[i].item()

        raw_start = cu_raw_bytes[i].item()
        raw_end = cu_raw_bytes[i + 1].item()
        image_raw = image_raw_concat[raw_start:raw_end]

        # Reshape to (best_height, row_stride), strip padding
        image = image_raw.reshape(best_height, row_stride)
        image = image[:, : best_width * 3].reshape(best_height, best_width, 3)

        # BMP bottom-up flip
        if do_flip:
            image = torch.flip(image, dims=[0])

        # BGR -> RGB
        image = torch.flip(image, dims=[-1])

        # HWC -> CHW, uint8 -> float32, /255.0
        image = image.permute(2, 0, 1).to(torch.float32).div_(255.0)

        # Normalize: (x - mean) / std
        image.sub_(mean).div_(std)

        # Cast to bfloat16
        image = image.to(torch.bfloat16)

        # Reshape to patches: (3, H, W) -> (n_patches, 3, patch_size, patch_size)
        image = (
            image.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
            .permute(1, 3, 0, 2, 4)
            .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
        )
        result_tensors.append(image)

    return torch.cat(result_tensors, dim=0)


def _build_fake_bmp(height: int, width: int, bottom_up: bool = True):
    """Build a minimal 24-bit BMP byte buffer with random pixel data.

    Returns:
        raw_bytes: bytes, valid BMP file
        pixel_data: (height, row_stride) uint8 tensor of raw pixel rows (for verification)
    """
    import struct as st

    row_stride = (width * 3 + 3) & ~3
    pixel_data_size = row_stride * height
    header_size = 54
    file_size = header_size + pixel_data_size

    # BMP file header (14 bytes) + DIB header (40 bytes)
    bmp_height = height if bottom_up else -height
    header = st.pack('<2sIHHI', b'BM', file_size, 0, 0, header_size)
    dib = st.pack('<IiiHHIIiiII', 40, width, bmp_height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0)

    # Random pixel data
    pixel_bytes = torch.randint(0, 256, (pixel_data_size,), dtype=torch.uint8)
    raw_bytes = header + dib + pixel_bytes.cpu().numpy().tobytes()

    return raw_bytes, row_stride
