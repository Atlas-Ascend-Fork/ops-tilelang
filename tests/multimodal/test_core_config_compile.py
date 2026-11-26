# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT

"""Configuration change tests for multimodal graph operators."""

import re

import pytest
import torch

from cann_ops_tilelang import config
from cann_ops_tilelang.multimodal import dsocr_kernel, image_embed_ops_kernel
from cann_ops_tilelang.multimodal.dsocr import attn_rel_h_rel_w
from cann_ops_tilelang.multimodal.image_embed_ops import (
    bmp_to_patches,
    prepare_image_attn_pos_id,
    replace_input_embeds,
    shuffle_image_tokens,
)

pytestmark = pytest.mark.npu


def _events(call):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        call()
        torch.npu.synchronize()
    return {event.key: event.count for event in profile.key_averages()}


def _attention_call():
    device = config.get_device()
    qkv = torch.randn((1, 196, 3, 80, 64), dtype=torch.bfloat16, device=device)
    q, k, v = qkv.unbind(2)
    rel_h = torch.randn((1, 196, 80, 16), dtype=torch.bfloat16, device=device)
    rel_w = torch.randn_like(rel_h)
    args = (q, k, v, rel_h, rel_w)
    expected = attn_rel_h_rel_w(*args)
    compiled = torch.compile(attn_rel_h_rel_w, backend='npugraph_ex', fullgraph=True, dynamic=False)

    def call():
        torch.testing.assert_close(compiled(*args), expected, rtol=0.01, atol=0.005)

    return call


def _replace_call():
    device = config.get_device()
    embeds = torch.randn((32, 4096), dtype=torch.bfloat16, device=device)
    initial = embeds.clone()
    ids = torch.zeros(32, dtype=torch.int32, device=device)
    ids[:8] = 99
    images = torch.randn((8, 4096), dtype=torch.bfloat16, device=device)
    pad_ids = torch.tensor([100], dtype=torch.int32, device=device)
    pads = torch.randn((1, 4096), dtype=torch.bfloat16, device=device)
    args = (embeds, ids, images, 99, pad_ids, pads)
    expected = initial.clone()
    replace_input_embeds(expected, *args[1:])
    compiled = torch.compile(replace_input_embeds, backend='npugraph_ex', fullgraph=True, dynamic=False)

    def call():
        embeds.copy_(initial)
        assert compiled(*args) is None
        torch.testing.assert_close(embeds, expected)

    return call


def _shuffle_call():
    device = config.get_device()
    embeds = torch.randn((8, 4096), dtype=torch.bfloat16, device=device)
    initial = embeds.clone()
    offsets = torch.tensor([0], dtype=torch.int32, device=device)
    sizes = torch.tensor([8], dtype=torch.int32, device=device)
    ids = torch.zeros(8, dtype=torch.int32, device=device)
    args = (embeds, offsets, sizes, ids)
    expected = initial.clone()
    shuffle_image_tokens(expected, offsets, sizes, ids)
    compiled = torch.compile(shuffle_image_tokens, backend='npugraph_ex', fullgraph=True, dynamic=False)

    def call():
        embeds.copy_(initial)
        assert compiled(*args) is None
        torch.testing.assert_close(embeds, expected)

    return call


def _positions_call():
    device = config.get_device()
    crops = torch.tensor([[2, 2]], dtype=torch.int32, device=device)
    cu = torch.tensor([0, 4], dtype=torch.int32, device=device)
    args = (crops, cu, 4, 2)
    expected = prepare_image_attn_pos_id(*args)
    compiled = torch.compile(prepare_image_attn_pos_id, backend='npugraph_ex', fullgraph=True, dynamic=False)

    def call():
        actual = compiled(*args)
        for output, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(output, reference)

    return call


def _bmp_call():
    device = config.get_device()
    pixels = torch.randint(0, 256, (16,), dtype=torch.uint8, device=device)
    cu_raw = torch.tensor([0, 16], dtype=torch.int32, device=device)
    crops = torch.tensor([[2, 2]], dtype=torch.int32, device=device)
    cu_vit = torch.tensor([0, 4], dtype=torch.int32, device=device)
    strides = torch.tensor([8], dtype=torch.int32, device=device)
    flips = torch.tensor([0], dtype=torch.int32, device=device)
    args = (pixels, cu_raw, crops, cu_vit, strides, flips, 4, 1, 0.5, 0.25)
    expected = bmp_to_patches(*args)
    compiled = torch.compile(bmp_to_patches, backend='npugraph_ex', fullgraph=True, dynamic=False)

    def call():
        torch.testing.assert_close(compiled(*args), expected)

    return call


_KERNEL_NAMES = {
    'attention': ('_attn_rel_h_rel_w_tl_ascend',),
    'replace': ('_replace_input_embeds_tl_ascend',),
    'shuffle': ('_shuffle_image_tokens_gather_tl_ascend', '_shuffle_image_tokens_scatter_tl_ascend'),
    'positions': ('_prepare_image_attn_pos_id_tl_ascend',),
    'bmp': ('_bmp_to_patches_tl_ascend',),
}


def _record_tir_factory(original, counts):
    def record(*args, **kwargs):
        tir = original(*args, **kwargs)
        binding = re.search(r'thread_binding\((\d+)', tir.script())
        assert binding is not None
        counts.append(int(binding.group(1)))
        return tir

    return record


def _record_kernel_cores(operator, monkeypatch):
    """Record core counts from generated TIR, including both shuffle phases."""
    module = dsocr_kernel if operator == 'attention' else image_embed_ops_kernel
    observed = []
    for name in _KERNEL_NAMES.get(operator, ()):
        factory = getattr(module, name)
        counts = []
        observed.append(counts)
        monkeypatch.setattr(factory, 'get_tir', _record_tir_factory(factory.get_tir, counts))
    return observed


def _assert_kernel_cores(observed, expected):
    assert all(counts and counts[-1] == expected for counts in observed), observed


@pytest.mark.parametrize('operator', ['attention', 'replace', 'shuffle', 'positions', 'bmp'])
@torch.inference_mode()
def test_multimodal_core_config_recompiles(operator, monkeypatch):
    """Reuse one compiled callable across setter and environment changes."""
    import torch_npu  # noqa: F401

    builders = {
        'attention': _attention_call,
        'replace': _replace_call,
        'shuffle': _shuffle_call,
        'positions': _positions_call,
        'bmp': _bmp_call,
    }
    monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
    config.clear_runtime_caches()
    try:
        call = builders.get(operator)()
        observed = _record_kernel_cores(operator, monkeypatch)
        default_ai = config.get_num_ai_cores()
        assert default_ai > 2
        first_events = _events(call)
        assert first_events.get('acl_graph_capture', 0) or first_events.get('acl_graph_replay', 0)

        config.set_num_ai_cores(default_ai - 1)
        assert config.get_num_ai_cores() == default_ai - 1
        assert _events(call).get('acl_graph_capture', 0) > 0
        _assert_kernel_cores(observed, default_ai - 1 if operator == 'attention' else 2 * default_ai - 2)

        config.set_num_ai_cores(default_ai - 1)
        assert _events(call).get('acl_graph_replay', 0) > 0

        if operator != 'attention':
            monkeypatch.setenv('OPS_TILELANG_NUM_VECTOR_CORES', str(2 * default_ai - 4))
            config.clear_runtime_caches()
            assert config.get_num_vector_cores() == 2 * default_ai - 4
            assert _events(call).get('acl_graph_capture', 0) > 0
            _assert_kernel_cores(observed, 2 * default_ai - 4)

        monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        config.reset_runtime_config()
        assert config.get_num_ai_cores() == default_ai
        assert _events(call).get('acl_graph_capture', 0) > 0
    finally:
        monkeypatch.delenv('OPS_TILELANG_NUM_VECTOR_CORES', raising=False)
        config.reset_runtime_config()
