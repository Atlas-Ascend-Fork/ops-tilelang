import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

from cann_ops_tilelang.config import get_num_vector_cores
from cann_ops_tilelang.utils import align_up, ceil_div


@tilelang.jit
def _get_topk_gate_kernel_impl(num_experts: int, num_topk: int, num_cores: int):
    """Top-k expert selection per token for Ascend NPU.

    Returns indices of the largest num_topk scores per row (ties broken by
    smaller index). Three compile-time specialized select_rows variants are
    selected based on num_experts: merged two-row (<=192 experts), single-row
    (<=384), or multi-group (>384). Output is int64 indices.
    """
    # ---------- Compile-time constants and hardware adaptation ----------
    num_stages = 6
    vector_length = 64

    # Grouping: split experts into groups that fit integer vector registers.
    max_vregs_per_group = 6
    num_vregs_per_group = ceil_div(min(num_experts, max_vregs_per_group * vector_length), vector_length)
    experts_per_group = num_vregs_per_group * vector_length
    num_groups = ceil_div(num_experts, experts_per_group)
    if num_groups == 2:
        # 2-group layouts break the in-UB masking cycle: every pick after the
        # first repeats the initial argmax. Widening to >=3 groups of 3 vregs
        # runs the proven multi-group structure (verified at e=769/1025).
        num_vregs_per_group = ceil_div(num_vregs_per_group, 2)
        experts_per_group = num_vregs_per_group * vector_length
        num_groups = ceil_div(num_experts, experts_per_group)
    num_aligned_experts = num_groups * experts_per_group

    # Tiling: single group (<=384 experts) loads 8 rows at once. Multi-group
    # tile_rows widens just enough that each row burst is >=32B and 32B-aligned
    # (num_topk < 4 rows are 8/16/24B; sub-32B MTE writes must be 32B aligned).
    if num_groups == 1:
        tile_rows = 8
    elif num_topk >= 4:
        tile_rows = 1
    else:
        tile_rows = 4 if num_topk % 2 else 2
    # acc_ub stores top‑k indices as int32; output is int64, so each index uses 2 int32 slots.
    acc_size = align_up(tile_rows * num_topk * 2, 32)
    acc_valid = tile_rows * num_topk * 2
    inner_unroll = min(tile_rows, 2)
    outer_iters = tile_rows // inner_unroll

    num_tokens = T.dynamic('num_tokens')

    # ---------- Utility macros ----------
    def _tree_reduce(count, mask, op, value_fn):
        """Balanced reduction tree over `count` values built by value_fn.
        Called from traced kernel body; no TIR loop constructs inside.
        """
        current = [value_fn(i) for i in range(count)]
        while len(current) > 1:
            nxt = []
            for j in range(0, len(current) - 1, 2):
                nxt.append(op(current[j], current[j + 1], mask))
            if len(current) % 2:
                nxt.append(current[-1])
            current = nxt
        return current[0]

    @T.macro
    def _simd_prologue():
        """Initialize vector constants: all‑true mask, lane‑1 mask, -inf, int32 max, expert count."""
        mask_32b = S.pset(32, 'PAT_ALL')
        one = S.pset(32, 'PAT_VL1')
        neg_inf = S.vdup(-T.infinity(T.float32), T.float32, mask_32b)
        int_max = S.vdup(T.max_value(T.int32), T.int32, mask_32b)
        num_exp = S.vdup(T.int32(num_experts), T.int32, mask_32b)
        return mask_32b, one, neg_inf, int_max, num_exp

    @T.macro
    def _select_one_k(scores_vec, index_vec, acc_ub, slot, mask_32b, one, neg_inf, int_max):
        """Single pick: find max (tie‑break by smallest index), store to acc_ub,
        then mask the chosen expert out with -inf.
        """
        acc_max = _tree_reduce(
            num_vregs_per_group, mask_32b, lambda a, b, m: S.vmax(a, b, m), lambda vi: scores_vec[vi]
        )
        mx = S.vdupv(S.vcmax(acc_max, mask_32b), mask_32b)
        acc_min_idx = _tree_reduce(
            num_vregs_per_group,
            mask_32b,
            lambda a, b, m: S.vmin(a, b, m),
            lambda vi: S.vsel(index_vec[vi], int_max, S.vcmp(scores_vec[vi], mx, mask_32b, 'eq')),
        )
        idx = S.vdupv(S.vcmin(acc_min_idx, mask_32b), mask_32b)
        S.vsts(acc_ub[slot], idx, one, 'ONEPT_B32')
        for i in T.unroll(num_vregs_per_group, explicit=True):
            scores_vec[i] = S.vsel(neg_inf, scores_vec[i], S.vcmp(index_vec[i], idx, mask_32b, 'eq'))

    @T.macro
    def _load_group(scores_vec, index_vec, scores_ub, row, group_id, neg_inf, num_exp, mask_32b):
        """Load one expert group from UB into registers, masking out invalid padding."""
        base = group_id * experts_per_group
        for i in T.unroll(num_vregs_per_group, explicit=True):
            offset = base + i * vector_length
            index_vec[i] = S.vci(T.int32(offset), T.int32)
            scores_vec[i] = S.vsel(
                S.vld(scores_ub[row, offset]), neg_inf, S.vcmp(index_vec[i], num_exp, mask_32b, 'lt')
            )

    # ---------- Three compile‑time specialized select_rows variants ----------
    if num_groups == 1:
        if num_vregs_per_group <= 3:
            # Variant 1: merged two‑row (≤192 experts). Process two rows per iteration.
            @T.macro
            def select_rows(base_row, scores_ub, acc_ub):
                for tr2 in T.serial(outer_iters):
                    with T.SimdVF():
                        mask_32b, one, neg_inf, int_max, num_exp = _simd_prologue()

                        sv0 = S.alloc_local((num_vregs_per_group,), T.float32)
                        iv0 = S.alloc_local((num_vregs_per_group,), T.int32)
                        sv1 = S.alloc_local((num_vregs_per_group,), T.float32)
                        iv1 = S.alloc_local((num_vregs_per_group,), T.int32)

                        # Load two rows and mask invalid lanes.
                        for i in T.unroll(num_vregs_per_group, explicit=True):
                            sv0[i] = S.vld(scores_ub[tr2 * 2, i * vector_length])
                            iv0[i] = S.vci(T.int32(i * vector_length), T.int32)
                            sv1[i] = S.vld(scores_ub[tr2 * 2 + 1, i * vector_length])
                            iv1[i] = S.vci(T.int32(i * vector_length), T.int32)
                            sv0[i] = S.vsel(sv0[i], neg_inf, S.vcmp(iv0[i], num_exp, mask_32b, 'lt'))
                            sv1[i] = S.vsel(sv1[i], neg_inf, S.vcmp(iv1[i], num_exp, mask_32b, 'lt'))

                        # Interleave top‑k selection for both rows.
                        for k in T.serial(num_topk):
                            _select_one_k(
                                sv0, iv0, acc_ub, (tr2 * 2) * num_topk * 2 + 2 * k, mask_32b, one, neg_inf, int_max
                            )
                            _select_one_k(
                                sv1, iv1, acc_ub, (tr2 * 2 + 1) * num_topk * 2 + 2 * k, mask_32b, one, neg_inf, int_max
                            )
        else:
            # Variant 2: single‑row vectorized (192 < experts ≤384). Process one row at a time.
            @T.macro
            def select_rows(base_row, scores_ub, acc_ub):
                for tr in T.serial(tile_rows):
                    row = base_row + tr
                    if row < num_tokens:
                        with T.SimdVF():
                            mask_32b, one, neg_inf, int_max, num_exp = _simd_prologue()
                            scores_vec = S.alloc_local((num_vregs_per_group,), T.float32)
                            index_vec = S.alloc_local((num_vregs_per_group,), T.int32)

                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                scores_vec[i] = S.vld(scores_ub[tr, i * vector_length])
                                index_vec[i] = S.vci(T.int32(i * vector_length), T.int32)

                            for i in T.unroll(num_vregs_per_group, explicit=True):
                                scores_vec[i] = S.vsel(
                                    scores_vec[i], neg_inf, S.vcmp(index_vec[i], num_exp, mask_32b, 'lt')
                                )

                            for k in T.serial(num_topk):
                                _select_one_k(
                                    scores_vec,
                                    index_vec,
                                    acc_ub,
                                    tr * num_topk * 2 + 2 * k,
                                    mask_32b,
                                    one,
                                    neg_inf,
                                    int_max,
                                )
    else:
        # Variant 3: multi‑group (>384 experts). Experts span multiple groups; process one row at a time.
        @T.macro
        def select_rows(base_row, scores_ub, acc_ub):
            for tr in T.serial(tile_rows):
                row = base_row + tr
                if row < num_tokens:
                    with T.SimdVF():
                        mask_32b, one, neg_inf, int_max, num_exp = _simd_prologue()
                        scores_vec = S.alloc_local((num_vregs_per_group,), T.float32)
                        index_vec = S.alloc_local((num_vregs_per_group,), T.int32)
                        tmp_idx_vec = S.alloc_local((num_vregs_per_group,), T.int32)
                        acc = S.alloc_var(T.float32)
                        acc_idx = S.alloc_var(T.int32)

                        for k in T.serial(num_topk):
                            # ---- Step 1: scan all groups to find global maximum ----
                            acc = neg_inf
                            for group_id in T.serial(num_groups):
                                _load_group(scores_vec, index_vec, scores_ub, tr, group_id, neg_inf, num_exp, mask_32b)
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    acc = S.vmax(acc, scores_vec[i], mask_32b)
                            mx = S.vdupv(S.vcmax(acc, mask_32b), mask_32b)

                            # ---- Step 2: scan all groups to find smallest index among maxima ----
                            acc_idx = int_max
                            for group_id in T.serial(num_groups):
                                _load_group(scores_vec, index_vec, scores_ub, tr, group_id, neg_inf, num_exp, mask_32b)
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    tmp_idx_vec[i] = S.vsel(
                                        index_vec[i], int_max, S.vcmp(scores_vec[i], mx, mask_32b, 'eq')
                                    )
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    acc_idx = S.vmin(acc_idx, tmp_idx_vec[i], mask_32b)
                            idx = S.vdupv(S.vcmin(acc_idx, mask_32b), mask_32b)

                            # ---- Store result ----
                            S.vsts(acc_ub[tr * num_topk * 2 + 2 * k], idx, one, 'ONEPT_B32')

                            # ---- Step 3: mask out the selected expert (write back to scores_ub) ----
                            for group_id in T.serial(num_groups):
                                base = group_id * experts_per_group
                                for i in T.unroll(num_vregs_per_group, explicit=True):
                                    offset = base + i * vector_length
                                    index_vec[i] = S.vci(T.int32(offset), T.int32)
                                    scores_vec[i] = S.vld(scores_ub[tr, offset])
                                    scores_vec[i] = S.vsel(
                                        neg_inf, scores_vec[i], S.vcmp(index_vec[i], idx, mask_32b, 'eq')
                                    )
                                    S.vsts(scores_ub[tr, offset], scores_vec[i], mask_32b)

                            S.mem_bar('VST_VLD')

    # ---------- Main kernel ----------
    @T.prim_func
    def topk_gate_kernel(
        scores: T.Tensor[(num_tokens, num_experts), T.float32],
        topk_idx: T.Tensor[(num_tokens, num_topk), T.int64],
    ):
        with T.Kernel(num_cores) as core_id:
            # Allocate shared memory (UB) buffers
            scores_ub = T.alloc_shared((tile_rows, num_aligned_experts), T.float32)
            acc_ub = T.alloc_shared((acc_size,), T.int32)
            topk_idx_flat = T.view(topk_idx, (num_tokens * num_topk * 2,), T.int32)
            T.annotate_buffer_versions({scores_ub: num_stages, acc_ub: num_stages})

            with T.SimdVF():
                T.clear(acc_ub)  # zero out all versions; ensures high 32 bits are 0

            num_outer_iters = T.ceildiv(num_tokens, tile_rows * num_cores)
            # Software-pipelined main loop: overlap load, compute, store
            for w in T.Pipelined(
                num_outer_iters,
                num_stages=num_stages,
                annotations={'multi_buffer_eligible': [scores_ub, acc_ub]},
            ):
                base_row = w * tile_rows * num_cores + core_id * tile_rows

                # ---- Load stage: GM -> UB ----
                for tr in T.unroll(tile_rows, explicit=True):
                    row = base_row + tr
                    if row < num_tokens:
                        T.copy(scores[row, :], scores_ub[tr, :num_experts])

                # ---- Compute stage: top‑k selection within UB ----
                select_rows(base_row, scores_ub, acc_ub)

                # ---- Store stage: UB -> GM ----
                # Whole-tile bursts only: sub-32B GM writes (num_topk < 4 row
                # tails) hit "MTE non-atomic address not aligned" and fault the
                # core. The wrapper pads the output so every started tile is
                # whole; pad rows hold garbage and are dropped by the view.
                if base_row < num_tokens:
                    T.copy(
                        acc_ub[:acc_valid],
                        topk_idx_flat[base_row * num_topk * 2 : base_row * num_topk * 2 + acc_valid],
                    )

    return topk_gate_kernel


def get_topk_gate_kernel(num_experts: int, num_topk: int, num_cores: int = 0):
    """Build a kernel with num_cores resolved from the live runtime config.

    The @tilelang.jit impl is keyed on the concrete num_cores value, so the
    resolution must happen here per call: resolving the sentinel inside the
    traced body would be frozen into the first compiled specialization.
    """
    if num_cores <= 0:
        num_cores = get_num_vector_cores()
    return _get_topk_gate_kernel_impl(num_experts, num_topk, num_cores)
