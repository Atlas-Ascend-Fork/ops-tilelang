"""High-performance Ascend operators implemented with TileLang."""

__all__ = [
    'engram_fused_weight',
    'engram_gate_bwd',
    'engram_gate_fwd',
    'engram_grad_w_reduce',
    'engram_sinkhorn_step',
    'moe_topk_gate',
    'normalize_weight',
    'topk_gate',
]

from .engram import engram_fused_weight, engram_gate_bwd, engram_gate_fwd, engram_grad_w_reduce, engram_sinkhorn_step
from .moe import moe_topk_gate, normalize_weight, topk_gate
