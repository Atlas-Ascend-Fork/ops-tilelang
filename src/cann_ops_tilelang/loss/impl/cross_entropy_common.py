# Copyright (c) 2026 ops-tilelang contributors
# SPDX-License-Identifier: MIT
# See LICENSE in the repository root for the full license text.


import tilelang

VOCAB_BLOCK = 24576
VOCAB_BYTES = 98304
ROW_BLOCK = 256
GRID_CORES = 64
SMALL_MAX_LANE_WORK = 16
SIMD_ROW_MAX_WAVES = 2
BALANCED_FUSED_BLOCK = True
FUSED_UB_BUDGET = 208 * 1024
COMPILED_KERNELS = {}


def _compile(name, program):
    compiled = tilelang.compile(
        program,
        target='ascend',
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: name.startswith(('chunk', 'row_'))},
    )
    COMPILED_KERNELS[name] = compiled
    return compiled
