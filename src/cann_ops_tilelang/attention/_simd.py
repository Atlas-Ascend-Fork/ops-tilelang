"""Small Ascend SIMD helpers shared by migrated cache kernels."""

from typing import Any

import tilelang
from tilelang.ascend import language as T
from tilelang.backend.target import determine_target

ASCEND_VSTSX2_BF16_PRELUDE = r"""
__simd_callee__ inline void tl_vstsx2_bf16(
    __ubuf__ bfloat16_t *dst, vector_bf16 even, vector_bf16 odd,
    vector_bool mask) {
  ::vsts(even, odd, dst, 0, INTLV_B16, mask);
}
"""


def is_pto_target() -> bool:
    target = determine_target(tilelang.env.get_default_target(), return_object=True)
    keys = target.keys
    if isinstance(keys, str):
        keys = (keys,)
    return any(str(key) == 'pto' for key in keys)


def vstsx2_bf16(dst: Any, even: Any, odd: Any, mask: Any) -> None:
    dst_ptr = T.access_ptr(dst, 'w', 256)
    if is_pto_target():
        T.evaluate(T.call_extern('void', 'pto.vstsx2', even, odd, dst_ptr, T.int32(0), 'INTLV_B16', mask))
    else:
        T.evaluate(T.call_extern('void', 'tl_vstsx2_bf16', dst_ptr, even, odd, mask))
