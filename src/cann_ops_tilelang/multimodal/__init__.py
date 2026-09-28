"""Multimodal operators for Ascend."""

__all__ = [
    'attn_rel_h_rel_w',
    'bmp_to_patches',
    'prepare_image_attn_pos_id',
    'replace_input_embeds',
    'shuffle_image_tokens',
]

from .dsocr import attn_rel_h_rel_w
from .image_embed_ops import (
    bmp_to_patches,
    prepare_image_attn_pos_id,
    replace_input_embeds,
    shuffle_image_tokens,
)
