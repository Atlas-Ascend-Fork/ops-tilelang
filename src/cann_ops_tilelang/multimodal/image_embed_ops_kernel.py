"""TileLang Ascend kernels for image embedding operations."""

import tilelang
import torch
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S
from tilelang.ascend.target import target_is_pto
from tvm.target import Target

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(out_idx=None)
def _replace_input_embeds_tl_ascend(
    hidden: int,
    rows_per_core: int = 16384,
    batch_rows: int = 8,
) -> T.prim_func:
    num_tokens = T.dynamic('num_tokens')
    num_pad_tokens = T.dynamic('num_pad_tokens')
    img_tokens = T.dynamic('img_tokens')

    token_id_dtype = torch.int32
    embed_dtype = torch.bfloat16

    num_cores = get_num_vector_cores()
    vec_size = 64
    max_groups = (rows_per_core + vec_size - 1) // vec_size
    max_pad_tokens = 8

    @T.prim_func
    def _replace_input_embeds_ascend(
        input_embeds: T.Tensor([num_tokens, hidden], embed_dtype),  # type: ignore
        input_ids: T.Tensor([num_tokens], token_id_dtype),  # type: ignore
        images_embeds: T.Tensor([img_tokens, hidden], embed_dtype),  # type: ignore
        image_token_id: T.int32,
        image_token_offset: T.Tensor([num_tokens], token_id_dtype),  # type: ignore
        image_pad_token_ids: T.Tensor([num_pad_tokens], token_id_dtype),  # type: ignore
        image_pad_embeds: T.Tensor([num_pad_tokens, hidden], embed_dtype),  # type: ignore
    ):
        with T.Kernel(num_cores) as core_id:
            row_base = num_tokens // num_cores
            row_rem = num_tokens % num_cores
            row_begin = core_id * row_base + T.min(core_id, row_rem)
            row_end = row_begin + row_base + T.cast(core_id < row_rem, 'int32')
            row_count = row_end - row_begin

            ub_rows = max_groups * vec_size
            input_ids_ub = T.alloc_shared((ub_rows,), token_id_dtype)
            offset_ub = T.alloc_shared((ub_rows,), token_id_dtype)
            pad_ids_ub = T.alloc_shared((max_pad_tokens,), token_id_dtype)
            embeds_ub = T.alloc_shared((batch_rows, hidden), embed_dtype)
            img_dst_ub = T.alloc_shared((max_groups * vec_size,), token_id_dtype)
            img_src_ub = T.alloc_shared((max_groups * vec_size,), token_id_dtype)
            # +1 group of slack: Phase 2 reads rstart[r+1] past the last run
            img_rstart_ub = T.alloc_shared(((max_groups + 1) * vec_size,), token_id_dtype)
            img_nrun_ub = T.alloc_shared((max_groups,), token_id_dtype)
            img_cnt_ub = T.alloc_shared((max_groups,), token_id_dtype)
            pad_dst_ub = T.alloc_shared((max_groups * max_pad_tokens * vec_size,), token_id_dtype)
            pad_type_ub = T.alloc_shared((max_groups * max_pad_tokens * vec_size,), token_id_dtype)

            # Zero UB buffers inside SimdVF (like engram_sinkhorn_asc.py)
            with T.SimdVF():
                T.clear(input_ids_ub)
                T.clear(offset_ub)
                T.clear(img_dst_ub)
                T.clear(img_src_ub)
                T.clear(pad_dst_ub)
                T.clear(pad_type_ub)

            T.copy(input_ids[row_begin:row_end], input_ids_ub[:row_count])
            T.copy(image_token_offset[row_begin:row_end], offset_ub[:row_count])
            T.copy(image_pad_token_ids[:num_pad_tokens], pad_ids_ub[:num_pad_tokens])

            num_groups = T.ceildiv(row_count, vec_size)

            # === Phase 1: scan — per-group SimdVF (g as parameter) ===
            for g in T.serial(num_groups):
                base = g * vec_size
                write_off = g * vec_size

                # Phase 1a: image scan
                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    zero = S.vdup(T.int32(0), T.int32, full)
                    target_img = S.vdup(image_token_id, T.int32, full)

                    ids = S.vld(input_ids_ub[base])
                    # inclusive cumsum: for an image token the value is already
                    # src_row + 1, matching the packed +1 encoding
                    offsets = S.vld(offset_ub[base])
                    lane = S.vci(T.int32(0), T.int32, 'INC_ORDER')
                    indices_p1 = S.vadds(S.vadds(lane, T.int32(base), full), T.int32(1), full)

                    img_mask = S.vcmp(ids, target_img, full, op='eq')
                    packed_dst = S.vsqz(S.vsel(indices_p1, zero, img_mask), img_mask)
                    packed_src = S.vsqz(S.vsel(offsets, zero, img_mask), img_mask)
                    S.vsts(img_dst_ub[write_off], packed_dst, full)
                    S.vsts(img_src_ub[write_off], packed_src, full)

                    # run starts in PACKED-slot space, computed in-register:
                    # ext[j] = packed[j] continues packed[j-1] (dst AND src +1);
                    # start = !ext, restricted to valid lanes (< cnt) since
                    # unfilled packed lanes hold 0 and would fake runs.
                    # `lane` (0..63) is the packed-slot coordinate.
                    one_i = S.vdup(T.int32(1), T.int32, full)
                    cnt_v = S.vcadd(S.vsel(one_i, zero, img_mask), full)
                    lane_m1 = S.vmaxs(S.vadds(lane, T.int32(-1), full), T.int32(0), full)
                    ext_d = S.vcmp(packed_dst, S.vadds(S.vselr(packed_dst, lane_m1), T.int32(1), full), full, op='eq')
                    ext_s = S.vcmp(packed_src, S.vadds(S.vselr(packed_src, lane_m1), T.int32(1), full), full, op='eq')
                    ext_v = S.vand(S.vsel(one_i, zero, ext_d), S.vsel(one_i, zero, ext_s), full)
                    valid = S.vcmp(lane, S.vdupv(cnt_v, full), full, op='lt')
                    sm = S.vcmp(
                        S.vand(
                            S.vsel(one_i, zero, S.vcmp(ext_v, zero, full, op='eq')), S.vsel(one_i, zero, valid), full
                        ),
                        one_i,
                        full,
                        op='eq',
                    )
                    S.vsts(img_rstart_ub[write_off], S.vsqz(S.vsel(lane, zero, sm), sm), full)
                    one_lane = S.pset(32, 'PAT_VL1')
                    S.vsts(img_cnt_ub[g], cnt_v, one_lane, dist='ONEPT_B32')
                    S.vsts(img_nrun_ub[g], S.vcadd(S.vsel(one_i, zero, sm), full), one_lane, dist='ONEPT_B32')

                # Phase 1b: pad scan — per pad type, separate SimdVF
                for p in T.serial(num_pad_tokens):
                    with T.SimdVF():
                        full2 = S.pset(32, 'PAT_ALL')
                        zero2 = S.vdup(T.int32(0), T.int32, full2)
                        pad_id = S.vdup(pad_ids_ub[p], T.int32, full2)

                        ids2 = S.vld(input_ids_ub[base])
                        indices2 = S.vci(T.int32(0), T.int32, 'INC_ORDER')
                        indices2 = S.vadds(indices2, T.int32(base), full2)
                        indices2_p1 = S.vadds(indices2, T.int32(1), full2)

                        pad_mask = S.vcmp(ids2, pad_id, full2, op='eq')
                        sel_pad = S.vsel(indices2_p1, zero2, pad_mask)
                        packed_pad = S.vsqz(sel_pad, pad_mask)
                        pad_type_p1 = S.vadds(S.vdup(T.int32(p), T.int32, full2), T.int32(1), full2)
                        packed_type = S.vsqz(pad_type_p1, pad_mask)
                        pad_off = (g * max_pad_tokens + p) * vec_size
                        S.vsts(pad_dst_ub[pad_off], packed_pad, full2)
                        S.vsts(pad_type_ub[pad_off], packed_type, full2)

            # === Phase 2: image copies, walking the SIMD-compressed runs ===
            for g in T.serial(num_groups):
                grp_base = g * vec_size
                nrun = img_nrun_ub[g]
                cnt = img_cnt_ub[g]
                for r in T.serial(nrun):
                    rs = img_rstart_ub[grp_base + r]
                    # tail run ends at cnt; the rstart[r+1] lookahead may hit
                    # stale lanes — killed by the multiply
                    has_nxt = T.cast(r + 1 < nrun, token_id_dtype)
                    run_len = img_rstart_ub[grp_base + r + 1] * has_nxt + cnt * (1 - has_nxt) - rs
                    run_src = img_src_ub[grp_base + rs] - 1
                    run_dst_g = row_begin + img_dst_ub[grp_base + rs] - 1
                    num_chunks = T.ceildiv(run_len, batch_rows)
                    for chunk in T.serial(num_chunks):
                        chunk_start = chunk * batch_rows
                        chunk_len = T.min(batch_rows, run_len - chunk_start)
                        T.copy(
                            images_embeds[run_src + chunk_start : run_src + chunk_start + chunk_len, :hidden],
                            embeds_ub[:chunk_len, :hidden],
                        )
                        T.copy(
                            embeds_ub[:chunk_len, :hidden],
                            input_embeds[run_dst_g + chunk_start : run_dst_g + chunk_start + chunk_len, :hidden],
                        )

            # === Phase 3: pad T.copy ===
            for g in T.serial(num_groups):
                for p in T.serial(num_pad_tokens):
                    pad_off = (g * max_pad_tokens + p) * vec_size
                    slot2 = T.alloc_var(token_id_dtype)
                    slot2 = 0
                    while slot2 < vec_size and pad_dst_ub[pad_off + slot2] != 0:
                        dst = pad_dst_ub[pad_off + slot2] - 1
                        p_type = pad_type_ub[pad_off + slot2] - 1
                        out_row = row_begin + dst
                        T.copy(image_pad_embeds[p_type, :hidden], embeds_ub[0, :hidden])
                        T.copy(embeds_ub[0, :hidden], input_embeds[out_row, :hidden])
                        slot2 = slot2 + 1

    return _replace_input_embeds_ascend


@tilelang.jit(
    out_idx=None,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
)
def _shuffle_image_tokens_gather_tl_ascend(
    hidden: int,
) -> T.prim_func:
    """Snapshot image tokens in their original order using tiled DMA."""
    num_tokens = T.dynamic('num_tokens')
    img_tokens = T.dynamic('img_tokens')
    num_imgs = T.dynamic('num_imgs')

    offset_dtype = torch.int32
    embed_dtype = torch.bfloat16

    num_cores = get_num_vector_cores()
    rows_per_tile = 11

    @T.macro
    def load_tile(row_ub, src, src_row, valid_rows, event_id):
        T.ascend_wait_flag('MTE3_MTE2', event_id)
        T.copy(src[src_row : src_row + valid_rows, :], row_ub[0:valid_rows, :], l2_cache_ctrl=0)
        T.ascend_set_flag('MTE2_MTE3', event_id)

    @T.macro
    def store_tile(row_ub, dst, dst_row, valid_rows, event_id):
        T.ascend_wait_flag('MTE2_MTE3', event_id)
        T.copy(row_ub[0:valid_rows, :], dst[dst_row : dst_row + valid_rows, :], l2_cache_ctrl=4)
        T.ascend_set_flag('MTE3_MTE2', event_id)

    @T.prim_func
    def _shuffle_image_tokens_gather_ascend(
        input_embeds: T.Tensor([num_tokens, hidden], embed_dtype),  # type: ignore
        image_real_offsets: T.Tensor([num_imgs], offset_dtype),  # type: ignore
        img_token_starts: T.Tensor([num_imgs], offset_dtype),  # type: ignore
        temp_buffer: T.Tensor([img_tokens, hidden], embed_dtype),  # type: ignore
    ):
        with T.Kernel(num_cores) as core_id:
            row_ub0 = T.alloc_shared((rows_per_tile, hidden), embed_dtype)
            row_ub1 = T.alloc_shared((rows_per_tile, hidden), embed_dtype)
            rows_per_core = T.ceildiv(img_tokens, num_cores)
            core_begin = core_id * rows_per_core
            core_end = T.min(core_begin + rows_per_core, img_tokens)
            T.ascend_set_flag('MTE3_MTE2', 0)
            T.ascend_set_flag('MTE3_MTE2', 1)

            for img_id in T.serial(num_imgs):
                img_begin = img_token_starts[img_id]
                img_end = T.alloc_var(offset_dtype)
                img_end = img_tokens
                if img_id + 1 < num_imgs:
                    img_end = img_token_starts[img_id + 1]

                copy_begin = T.alloc_var(offset_dtype)
                copy_end = T.alloc_var(offset_dtype)
                copy_begin = core_begin
                copy_end = core_end
                if copy_begin < img_begin:
                    copy_begin = img_begin
                if copy_end > img_end:
                    copy_end = img_end

                if copy_begin < copy_end:
                    copy_rows = copy_end - copy_begin
                    num_tiles = T.ceildiv(copy_rows, rows_per_tile)

                    real_offset = image_real_offsets[img_id]
                    first_rows = T.min(rows_per_tile, copy_rows)
                    first_src_row = real_offset + copy_begin - img_begin
                    load_tile(row_ub0, input_embeds, first_src_row, first_rows, 0)

                    for tile_id in T.serial(num_tiles):
                        flat_row = copy_begin + tile_id * rows_per_tile
                        valid_rows = T.min(rows_per_tile, copy_end - flat_row)

                        if tile_id + 1 < num_tiles:
                            next_flat_row = flat_row + rows_per_tile
                            next_rows = T.min(rows_per_tile, copy_end - next_flat_row)
                            next_src_row = real_offset + next_flat_row - img_begin
                            if tile_id % 2 == 0:
                                load_tile(row_ub1, input_embeds, next_src_row, next_rows, 1)
                            else:
                                load_tile(row_ub0, input_embeds, next_src_row, next_rows, 0)

                        if tile_id % 2 == 0:
                            store_tile(row_ub0, temp_buffer, flat_row, valid_rows, 0)
                        else:
                            store_tile(row_ub1, temp_buffer, flat_row, valid_rows, 1)

            T.ascend_wait_flag('MTE3_MTE2', 0)
            T.ascend_wait_flag('MTE3_MTE2', 1)

    return _shuffle_image_tokens_gather_ascend


@tilelang.jit(
    out_idx=None,
    pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
)
def _shuffle_image_tokens_scatter_tl_ascend(
    hidden: int,
) -> T.prim_func:
    """Interleave the snapshot with contiguous reads and stride-2 DMA writes."""
    num_tokens = T.dynamic('num_tokens')
    img_tokens = T.dynamic('img_tokens')
    num_imgs = T.dynamic('num_imgs')

    offset_dtype = torch.int32
    embed_dtype = torch.bfloat16

    num_cores = get_num_vector_cores()
    rows_per_tile = 11

    @T.macro
    def load_tile(row_ub, src, src_row, valid_rows, event_id):
        T.ascend_wait_flag('MTE3_MTE2', event_id)
        T.copy(src[src_row : src_row + valid_rows, :], row_ub[0:valid_rows, :], l2_cache_ctrl=0)
        T.ascend_set_flag('MTE2_MTE3', event_id)

    @T.macro
    def store_tile(row_ub, dst, dst_row, valid_rows, event_id):
        T.ascend_wait_flag('MTE2_MTE3', event_id)
        dst_strided = T.make_tensor(
            T.access_ptr(dst[dst_row, 0], 'w', extent=(2 * valid_rows - 1) * hidden),
            shape=(valid_rows, hidden),
            dtype=embed_dtype,
            strides=(2 * hidden, 1),
        )
        T.copy(row_ub[0:valid_rows, :], dst_strided, l2_cache_ctrl=4)
        T.ascend_set_flag('MTE3_MTE2', event_id)

    @T.prim_func
    def _shuffle_image_tokens_scatter_ascend(
        input_embeds: T.Tensor([num_tokens, hidden], embed_dtype),  # type: ignore
        image_real_offsets: T.Tensor([num_imgs], offset_dtype),  # type: ignore
        image_real_sizes: T.Tensor([num_imgs], offset_dtype),  # type: ignore
        img_token_starts: T.Tensor([num_imgs], offset_dtype),  # type: ignore
        temp_buffer: T.Tensor([img_tokens, hidden], embed_dtype),  # type: ignore
    ):
        with T.Kernel(num_cores) as core_id:
            row_ub0 = T.alloc_shared((rows_per_tile, hidden), embed_dtype)
            row_ub1 = T.alloc_shared((rows_per_tile, hidden), embed_dtype)
            rows_per_core = T.ceildiv(img_tokens, num_cores)
            core_begin = core_id * rows_per_core
            core_end = T.min(core_begin + rows_per_core, img_tokens)
            T.ascend_set_flag('MTE3_MTE2', 0)
            T.ascend_set_flag('MTE3_MTE2', 1)

            for img_id in T.serial(num_imgs):
                img_begin = img_token_starts[img_id]
                img_size = image_real_sizes[img_id]
                img_end = img_begin + img_size
                copy_begin = T.alloc_var(offset_dtype)
                copy_end = T.alloc_var(offset_dtype)
                copy_begin = core_begin
                copy_end = core_end
                if copy_begin < img_begin:
                    copy_begin = img_begin
                if copy_end > img_end:
                    copy_end = img_end

                if copy_begin < copy_end:
                    local_begin = copy_begin - img_begin
                    local_end = copy_end - img_begin
                    half_size = img_size // 2
                    real_offset = image_real_offsets[img_id]

                    even_begin = (local_begin + 1) // 2
                    even_end = (local_end + 1) // 2
                    even_rows = even_end - even_begin
                    if even_rows > 0:
                        even_tiles = T.ceildiv(even_rows, rows_per_tile)
                        first_rows = T.min(rows_per_tile, even_rows)
                        load_tile(row_ub0, temp_buffer, img_begin + even_begin, first_rows, 0)

                        for tile_id in T.serial(even_tiles):
                            row_offset = tile_id * rows_per_tile
                            valid_rows = T.min(rows_per_tile, even_rows - row_offset)
                            dst_row = real_offset + 2 * (even_begin + row_offset)

                            if tile_id + 1 < even_tiles:
                                next_row_offset = row_offset + rows_per_tile
                                next_rows = T.min(rows_per_tile, even_rows - next_row_offset)
                                next_src_row = img_begin + even_begin + next_row_offset
                                if tile_id % 2 == 0:
                                    load_tile(row_ub1, temp_buffer, next_src_row, next_rows, 1)
                                else:
                                    load_tile(row_ub0, temp_buffer, next_src_row, next_rows, 0)

                            if tile_id % 2 == 0:
                                store_tile(row_ub0, input_embeds, dst_row, valid_rows, 0)
                            else:
                                store_tile(row_ub1, input_embeds, dst_row, valid_rows, 1)

                    odd_begin = local_begin // 2
                    odd_end = local_end // 2
                    odd_rows = odd_end - odd_begin
                    if odd_rows > 0:
                        odd_tiles = T.ceildiv(odd_rows, rows_per_tile)
                        first_rows = T.min(rows_per_tile, odd_rows)
                        first_src_row = img_begin + half_size + odd_begin
                        load_tile(row_ub0, temp_buffer, first_src_row, first_rows, 0)

                        for tile_id in T.serial(odd_tiles):
                            row_offset = tile_id * rows_per_tile
                            valid_rows = T.min(rows_per_tile, odd_rows - row_offset)
                            dst_row = real_offset + 2 * (odd_begin + row_offset) + 1

                            if tile_id + 1 < odd_tiles:
                                next_row_offset = row_offset + rows_per_tile
                                next_rows = T.min(rows_per_tile, odd_rows - next_row_offset)
                                next_src_row = img_begin + half_size + odd_begin + next_row_offset
                                if tile_id % 2 == 0:
                                    load_tile(row_ub1, temp_buffer, next_src_row, next_rows, 1)
                                else:
                                    load_tile(row_ub0, temp_buffer, next_src_row, next_rows, 0)

                            if tile_id % 2 == 0:
                                store_tile(row_ub0, input_embeds, dst_row, valid_rows, 0)
                            else:
                                store_tile(row_ub1, input_embeds, dst_row, valid_rows, 1)

            T.ascend_wait_flag('MTE3_MTE2', 0)
            T.ascend_wait_flag('MTE3_MTE2', 1)

    return _shuffle_image_tokens_scatter_ascend


@tilelang.jit(out_idx=None, pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: True})
def _prepare_image_attn_pos_id_tl_ascend(pattern_id: str, owner_mode: str) -> T.prim_func:
    if pattern_id not in ('single', 'owner', 'flat'):
        raise ValueError(f'unsupported prepare_image_attn_pos_id pattern: {pattern_id}')
    if pattern_id == 'owner' and owner_mode not in ('win1', 'window'):
        raise ValueError(f'unsupported prepare_image_attn_pos_id owner mode: {owner_mode}')

    num_imgs = T.dynamic('num_imgs')
    total_seqlens_vit = T.dynamic('total_seqlens_vit')

    dtype = torch.int32
    num_cores = get_num_vector_cores()
    vector_lanes = 64
    # The two position outputs are int32 [row, col] records.  Each output
    # buffer is explicitly forced to two versions below, so one pipeline
    # element consumes 2 outputs * 2 stages * 2 coordinates * 4 bytes = 32B.
    # Keep a small fixed reserve for compiler/runtime UB bookkeeping instead
    # of sizing the logical tiles to the physical limit with no headroom.
    ub_bytes = 248 * 1024
    ub_reserve_bytes = 8 * 1024
    ub_budget_bytes = ub_bytes - ub_reserve_bytes
    pipeline_bytes_per_patch = 2 * 2 * 2 * 4
    pipeline_tile_cap = ((ub_budget_bytes // pipeline_bytes_per_patch) // vector_lanes) * vector_lanes
    single_bytes_per_column = pipeline_bytes_per_patch + 4  # window_col_ub
    single_row_tile_cap = ((ub_budget_bytes // single_bytes_per_column) // vector_lanes) * vector_lanes
    row_tile_cap = single_row_tile_cap if pattern_id == 'single' else 2048
    row_vector_blocks = row_tile_cap // vector_lanes
    owner_tile_cap = 6912

    @T.macro
    def write_tile_win1(global_ub, window_ub, tile_start, tile_count, n_vit_w):
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            n_vit_w_f = S.vcvt(S.vdup(n_vit_w, T.int32, full), T.float32)
            inv_n_vit_w = S.vdiv(S.vdup(1.0, T.float32, full), n_vit_w_f, full)
            for vec_idx in T.serial(T.ceildiv(tile_count, vector_lanes)):
                vec_start = vec_idx * vector_lanes
                patch = S.vci(T.int32(tile_start + vec_start), T.int32)
                patch_f = S.vcvt(patch, T.float32)
                row = S.vcvt(S.vmul(patch_f, inv_n_vit_w, full), T.int32, round='ROUND_C')
                prod = S.vmuls(row, n_vit_w, full)
                col_raw = S.vsub(patch, prod, full)
                ge_w = S.vcmps(col_raw, n_vit_w, full, 'ge')
                col_hi = S.vadds(col_raw, -n_vit_w, full)
                row_hi = S.vadds(row, 1, full)
                col_adj = S.vsel(col_hi, col_raw, ge_w)
                row_adj = S.vsel(row_hi, row, ge_w)
                lt_zero = S.vcmps(col_adj, 0, full, 'lt')
                col = S.vsel(S.vadds(col_adj, n_vit_w, full), col_adj, lt_zero)
                row_final = S.vsel(S.vadds(row_adj, -1, full), row_adj, lt_zero)
                zero = S.vdup(0, T.int32, full)

                # Form [row, col] pairs in registers, then write the pair
                # vector contiguously. The final partial vector may overwrite
                # inactive UB elements, but the following DMA copies only
                # tile_count records.
                global_lo, global_hi = S.vintlv(row_final, col)
                S.vsts(global_ub[0, vec_start, 0], global_lo, full)
                S.vsts(global_ub[0, vec_start + 32, 0], global_hi, full)
                window_lo, window_hi = S.vintlv(zero, zero)
                S.vsts(window_ub[0, vec_start, 0], window_lo, full)
                S.vsts(window_ub[0, vec_start + 32, 0], window_hi, full)

    @T.macro
    def write_tile_window(
        global_ub,
        window_ub,
        tile_start,
        tile_count,
        n_vit_w,
        win_size,
        dev_h,
        dev_w,
    ):
        with T.SimdVF():
            full = S.pset(32, 'PAT_ALL')
            n_vit_w_f = S.vcvt(S.vdup(n_vit_w, T.int32, full), T.float32)
            inv_n_vit_w = S.vdiv(S.vdup(1.0, T.float32, full), n_vit_w_f, full)
            win_size_f = S.vcvt(S.vdup(win_size, T.int32, full), T.float32)
            inv_win_size = S.vdiv(S.vdup(1.0, T.float32, full), win_size_f, full)
            for vec_idx in T.serial(T.ceildiv(tile_count, vector_lanes)):
                vec_start = vec_idx * vector_lanes
                patch = S.vci(T.int32(tile_start + vec_start), T.int32)
                patch_f = S.vcvt(patch, T.float32)
                row = S.vcvt(S.vmul(patch_f, inv_n_vit_w, full), T.int32, round='ROUND_C')
                prod = S.vmuls(row, n_vit_w, full)
                col_raw = S.vsub(patch, prod, full)
                ge_w = S.vcmps(col_raw, n_vit_w, full, 'ge')
                col_hi = S.vadds(col_raw, -n_vit_w, full)
                row_hi = S.vadds(row, 1, full)
                col_adj = S.vsel(col_hi, col_raw, ge_w)
                row_adj = S.vsel(row_hi, row, ge_w)
                lt_zero = S.vcmps(col_adj, 0, full, 'lt')
                col = S.vsel(S.vadds(col_adj, n_vit_w, full), col_adj, lt_zero)
                row_final = S.vsel(S.vadds(row_adj, -1, full), row_adj, lt_zero)

                d_row = S.vadds(row_final, -dev_h, full)
                q_row = S.vcvt(S.vmul(S.vcvt(d_row, T.float32), inv_win_size, full), T.int32, round='ROUND_C')
                mod_row_raw = S.vsub(d_row, S.vmuls(q_row, win_size, full), full)
                mod_row_ge = S.vcmps(mod_row_raw, win_size, full, 'ge')
                mod_row_hi = S.vsel(S.vadds(mod_row_raw, -win_size, full), mod_row_raw, mod_row_ge)
                mod_row_lt = S.vcmps(mod_row_hi, 0, full, 'lt')
                mod_row = S.vsel(S.vadds(mod_row_hi, win_size, full), mod_row_hi, mod_row_lt)
                win_row = S.vsel(row_final, mod_row, S.vcmps(row_final, dev_h, full, 'lt'))

                d_col = S.vadds(col, -dev_w, full)
                q_col = S.vcvt(S.vmul(S.vcvt(d_col, T.float32), inv_win_size, full), T.int32, round='ROUND_C')
                mod_col_raw = S.vsub(d_col, S.vmuls(q_col, win_size, full), full)
                mod_col_ge = S.vcmps(mod_col_raw, win_size, full, 'ge')
                mod_col_hi = S.vsel(S.vadds(mod_col_raw, -win_size, full), mod_col_raw, mod_col_ge)
                mod_col_lt = S.vcmps(mod_col_hi, 0, full, 'lt')
                mod_col = S.vsel(S.vadds(mod_col_hi, win_size, full), mod_col_hi, mod_col_lt)
                win_col = S.vsel(col, mod_col, S.vcmps(col, dev_w, full, 'lt'))

                global_lo, global_hi = S.vintlv(row_final, col)
                S.vsts(global_ub[0, vec_start, 0], global_lo, full)
                S.vsts(global_ub[0, vec_start + 32, 0], global_hi, full)
                window_lo, window_hi = S.vintlv(win_row, win_col)
                S.vsts(window_ub[0, vec_start, 0], window_lo, full)
                S.vsts(window_ub[0, vec_start + 32, 0], window_hi, full)

    @T.macro
    def write_tile(
        global_ub,
        window_ub,
        tile_start,
        tile_count,
        n_vit_w,
        win_size,
        dev_h,
        dev_w,
    ):
        if win_size == 1:
            write_tile_win1(global_ub, window_ub, tile_start, tile_count, n_vit_w)
        else:
            write_tile_window(
                global_ub,
                window_ub,
                tile_start,
                tile_count,
                n_vit_w,
                win_size,
                dev_h,
                dev_w,
            )

    @T.macro
    def process_single_image(
        images_spatial_crop,
        images_global_attn_pos_id,
        images_window_attn_pos_id,
        global_ub,
        window_ub,
        window_col_ub,
        core_id,
        win_size,
    ):
        n_vit_w = T.alloc_var(dtype)
        n_vit_h = T.alloc_var(dtype)
        rows_per_core = T.alloc_var(dtype)
        row_begin = T.alloc_var(dtype)
        n_window_h = T.alloc_var(dtype)
        n_window_w = T.alloc_var(dtype)
        pad_h = T.alloc_var(dtype)
        pad_w = T.alloc_var(dtype)
        dev_h = T.alloc_var(dtype)
        dev_w = T.alloc_var(dtype)
        n_vit_w = images_spatial_crop[0, 0]
        n_vit_h = images_spatial_crop[0, 1]
        rows_per_core = T.ceildiv(n_vit_h, num_cores)
        row_begin = core_id * rows_per_core
        n_window_h = T.ceildiv(n_vit_h, win_size)
        n_window_w = T.ceildiv(n_vit_w, win_size)
        pad_h = (n_window_h * win_size - n_vit_h) // 2
        pad_w = (n_window_w * win_size - n_vit_w) // 2
        dev_h = win_size - pad_h
        dev_w = win_size - pad_w

        for tile_idx in T.serial(T.ceildiv(n_vit_w, row_tile_cap)):
            tile_start = T.alloc_var(dtype)
            tile_count = T.alloc_var(dtype)
            tile_start = tile_idx * row_tile_cap
            tile_count = T.min(row_tile_cap, n_vit_w - tile_start)

            if win_size != 1:
                with T.SimdVF():
                    full = S.pset(32, 'PAT_ALL')
                    win_size_f = S.vcvt(S.vdup(win_size, T.int32, full), T.float32)
                    inv_win_size = S.vdiv(S.vdup(1.0, T.float32, full), win_size_f, full)
                    for vec_idx in T.serial(row_vector_blocks):
                        vec_start = vec_idx * vector_lanes
                        col = S.vci(T.int32(tile_start + vec_start), T.int32)
                        d_col = S.vadds(col, -dev_w, full)
                        q_col = S.vcvt(
                            S.vmul(S.vcvt(d_col, T.float32), inv_win_size, full),
                            T.int32,
                            round='ROUND_C',
                        )
                        mod_col_raw = S.vsub(d_col, S.vmuls(q_col, win_size, full), full)
                        mod_col_ge = S.vcmps(mod_col_raw, win_size, full, 'ge')
                        mod_col_hi = S.vsel(S.vadds(mod_col_raw, -win_size, full), mod_col_raw, mod_col_ge)
                        mod_col_lt = S.vcmps(mod_col_hi, 0, full, 'lt')
                        mod_col = S.vsel(S.vadds(mod_col_hi, win_size, full), mod_col_hi, mod_col_lt)
                        win_col = S.vsel(col, mod_col, S.vcmps(col, dev_w, full, 'lt'))
                        S.vsts(window_col_ub[vec_start], win_col, full)

            for row_offset in T.Pipelined(
                rows_per_core,
                num_stages=2,
                annotations={'multi_buffer_eligible': [global_ub, window_ub]},
            ):
                row = T.alloc_var(dtype)
                row = row_begin + row_offset
                if row < n_vit_h:
                    window_row = T.alloc_var(dtype)
                    if win_size != 1:
                        if row < dev_h:  # noqa: SIM108
                            window_row = row
                        else:
                            window_row = (row - dev_h) % win_size

                    if win_size == 1:
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            zero = S.vdup(0, T.int32, full)
                            for vec_idx in T.serial(row_vector_blocks):
                                vec_start = vec_idx * vector_lanes
                                col = S.vci(T.int32(tile_start + vec_start), T.int32)
                                row_vector = S.vdup(T.int32(row), T.int32, full)
                                global_lo, global_hi = S.vintlv(row_vector, col)
                                S.vsts(global_ub[0, vec_start, 0], global_lo, full)
                                S.vsts(global_ub[0, vec_start + 32, 0], global_hi, full)
                                window_lo, window_hi = S.vintlv(zero, zero)
                                S.vsts(window_ub[0, vec_start, 0], window_lo, full)
                                S.vsts(window_ub[0, vec_start + 32, 0], window_hi, full)
                    else:
                        with T.SimdVF():
                            full = S.pset(32, 'PAT_ALL')
                            for vec_idx in T.serial(row_vector_blocks):
                                vec_start = vec_idx * vector_lanes
                                col = S.vci(T.int32(tile_start + vec_start), T.int32)
                                row_vector = S.vdup(T.int32(row), T.int32, full)
                                global_lo, global_hi = S.vintlv(row_vector, col)
                                S.vsts(global_ub[0, vec_start, 0], global_lo, full)
                                S.vsts(global_ub[0, vec_start + 32, 0], global_hi, full)
                                win_col = S.vld(window_col_ub[vec_start])
                                window_row_vector = S.vdup(T.int32(window_row), T.int32, full)
                                window_lo, window_hi = S.vintlv(window_row_vector, win_col)
                                S.vsts(window_ub[0, vec_start, 0], window_lo, full)
                                S.vsts(window_ub[0, vec_start + 32, 0], window_hi, full)

                    output_offset = T.alloc_var(dtype)
                    output_offset = row * n_vit_w + tile_start
                    T.copy(
                        global_ub[0, 0:tile_count, 0:2],
                        images_global_attn_pos_id[output_offset : output_offset + tile_count, 0:2],
                    )
                    T.copy(
                        window_ub[0, 0:tile_count, 0:2],
                        images_window_attn_pos_id[output_offset : output_offset + tile_count, 0:2],
                    )

    @T.macro
    def process_flattened_partition(
        images_spatial_crop,
        cu_seqlens_vit,
        images_global_attn_pos_id,
        images_window_attn_pos_id,
        global_ub,
        window_ub,
        core_id,
        win_size,
    ):
        active_cores = T.min(num_cores, T.max(1, T.ceildiv(total_seqlens_vit, 2048)))
        core_start = (total_seqlens_vit * core_id) // active_cores
        core_end = (total_seqlens_vit * (core_id + 1)) // active_cores

        if core_id < active_cores:
            first_img = T.alloc_var(dtype)
            first_img = 0
            for image_scan in T.serial(num_imgs):
                if core_start >= cu_seqlens_vit[image_scan + 1]:
                    first_img = image_scan + 1

            for image_offset in T.serial(num_imgs):
                img_id = first_img + image_offset
                if img_id < num_imgs:
                    image_begin = cu_seqlens_vit[img_id]
                    image_end = cu_seqlens_vit[img_id + 1]
                    segment_begin = T.max(core_start, image_begin)
                    segment_end = T.min(core_end, image_end)
                    if segment_begin < segment_end:
                        n_vit_w = T.alloc_var(dtype)
                        n_vit_h = T.alloc_var(dtype)
                        segment_patches = T.alloc_var(dtype)
                        n_window_h = T.alloc_var(dtype)
                        n_window_w = T.alloc_var(dtype)
                        pad_h = T.alloc_var(dtype)
                        pad_w = T.alloc_var(dtype)
                        dev_h = T.alloc_var(dtype)
                        dev_w = T.alloc_var(dtype)
                        n_vit_w = images_spatial_crop[img_id, 0]
                        n_vit_h = images_spatial_crop[img_id, 1]
                        segment_patches = segment_end - segment_begin
                        n_window_h = T.ceildiv(n_vit_h, win_size)
                        n_window_w = T.ceildiv(n_vit_w, win_size)
                        pad_h = (n_window_h * win_size - n_vit_h) // 2
                        pad_w = (n_window_w * win_size - n_vit_w) // 2
                        dev_h = win_size - pad_h
                        dev_w = win_size - pad_w
                        for tile_idx in T.Pipelined(
                            T.ceildiv(segment_patches, pipeline_tile_cap),
                            num_stages=2,
                            annotations={'multi_buffer_eligible': [global_ub, window_ub]},
                        ):
                            tile_start = T.alloc_var(dtype)
                            tile_count = T.alloc_var(dtype)
                            tile_start = segment_begin - image_begin + tile_idx * pipeline_tile_cap
                            tile_count = T.min(pipeline_tile_cap, segment_patches - tile_idx * pipeline_tile_cap)

                            write_tile(
                                global_ub,
                                window_ub,
                                tile_start,
                                tile_count,
                                n_vit_w,
                                win_size,
                                dev_h,
                                dev_w,
                            )

                            output_offset = T.alloc_var(dtype)
                            output_offset = segment_begin + tile_idx * pipeline_tile_cap
                            T.copy(
                                global_ub[0, 0:tile_count, 0:2],
                                images_global_attn_pos_id[output_offset : output_offset + tile_count, 0:2],
                            )
                            T.copy(
                                window_ub[0, 0:tile_count, 0:2],
                                images_window_attn_pos_id[output_offset : output_offset + tile_count, 0:2],
                            )

    @T.macro
    def process_image_owner(
        images_spatial_crop,
        cu_seqlens_vit,
        images_global_attn_pos_id,
        images_window_attn_pos_id,
        global_ub,
        window_ub,
        crop_ub,
        cu_ub,
        window_i32_ub,
        core_id,
        win_size,
    ):
        imgs_per_core = T.ceildiv(num_imgs, num_cores)
        img_begin = core_id * imgs_per_core
        for img_offset in T.serial(imgs_per_core):
            img_id = img_begin + img_offset
            if img_id < num_imgs:
                T.copy(images_spatial_crop[img_id, 0:2], crop_ub[0:2])
                T.copy(cu_seqlens_vit[img_id : img_id + 1], cu_ub[0:1])
                n_vit_w = crop_ub[0]
                n_vit_h = crop_ub[1]
                image_patches = n_vit_w * n_vit_h
                image_begin = cu_ub[0]

                if owner_mode == 'window':
                    n_window_h = T.ceildiv(n_vit_h, win_size)
                    n_window_w = T.ceildiv(n_vit_w, win_size)
                    window_i32_ub[0] = win_size - (n_window_h * win_size - n_vit_h) // 2
                    window_i32_ub[1] = win_size - (n_window_w * win_size - n_vit_w) // 2

                num_tiles = T.ceildiv(image_patches, owner_tile_cap)
                for tile_group in T.serial(T.ceildiv(num_tiles, 2)):
                    group_tiles = T.min(2, num_tiles - tile_group * 2)
                    for tile_side in T.Pipelined(
                        group_tiles,
                        num_stages=2,
                        annotations={'multi_buffer_eligible': [global_ub, window_ub]},
                    ):
                        tile_idx = tile_group * 2 + tile_side
                        tile_start = tile_idx * owner_tile_cap
                        tile_count = T.min(owner_tile_cap, image_patches - tile_start)
                        if owner_mode == 'win1':
                            write_tile_win1(global_ub, window_ub, tile_start, tile_count, n_vit_w)
                        else:
                            write_tile_window(
                                global_ub,
                                window_ub,
                                tile_start,
                                tile_count,
                                n_vit_w,
                                win_size,
                                window_i32_ub[0],
                                window_i32_ub[1],
                            )
                        output_offset = image_begin + tile_start
                        T.copy(
                            global_ub[0, 0:tile_count, 0:2],
                            images_global_attn_pos_id[output_offset : output_offset + tile_count, 0:2],
                        )
                        T.copy(
                            window_ub[0, 0:tile_count, 0:2],
                            images_window_attn_pos_id[output_offset : output_offset + tile_count, 0:2],
                        )

    @T.prim_func
    def _prepare_image_attn_pos_id_ascend(
        images_spatial_crop: T.Tensor([num_imgs, 2], dtype),
        cu_seqlens_vit: T.Tensor([num_imgs + 1], dtype),
        images_global_attn_pos_id: T.Tensor([total_seqlens_vit, 2], dtype),
        images_window_attn_pos_id: T.Tensor([total_seqlens_vit, 2], dtype),
        win_size: T.int32,
    ):
        with T.Kernel(num_cores) as core_id:
            if pattern_id == 'single':
                global_ub = T.alloc_shared((1, row_tile_cap, 2), dtype)
                window_ub = T.alloc_shared((1, row_tile_cap, 2), dtype)
                window_col_ub = T.alloc_shared((row_tile_cap,), dtype)
                T.annotate_buffer_versions({global_ub: 2, window_ub: 2})
                process_single_image(
                    images_spatial_crop,
                    images_global_attn_pos_id,
                    images_window_attn_pos_id,
                    global_ub,
                    window_ub,
                    window_col_ub,
                    core_id,
                    win_size,
                )
            elif pattern_id == 'flat':
                global_ub = T.alloc_shared((1, pipeline_tile_cap, 2), dtype)
                window_ub = T.alloc_shared((1, pipeline_tile_cap, 2), dtype)
                T.annotate_buffer_versions({global_ub: 2, window_ub: 2})
                process_flattened_partition(
                    images_spatial_crop,
                    cu_seqlens_vit,
                    images_global_attn_pos_id,
                    images_window_attn_pos_id,
                    global_ub,
                    window_ub,
                    core_id,
                    win_size,
                )
            else:
                global_ub = T.alloc_shared((1, owner_tile_cap, 2), dtype)
                window_ub = T.alloc_shared((1, owner_tile_cap, 2), dtype)
                crop_ub = T.alloc_shared((64,), dtype)
                cu_ub = T.alloc_shared((64,), dtype)
                window_i32_ub = T.alloc_shared((64,), dtype)
                T.annotate_buffer_versions({global_ub: 2, window_ub: 2})
                process_image_owner(
                    images_spatial_crop,
                    cu_seqlens_vit,
                    images_global_attn_pos_id,
                    images_window_attn_pos_id,
                    global_ub,
                    window_ub,
                    crop_ub,
                    cu_ub,
                    window_i32_ub,
                    core_id,
                    win_size,
                )

    return _prepare_image_attn_pos_id_ascend


@tilelang.jit(out_idx=None, target='ascend')
def _bmp_to_patches_tl_ascend(patch_size: int) -> T.prim_func:
    """Ascend AIV implementation of ``bmp_to_patches``.

    Small patches use one padded, strided GM-to-UB copy; large patches retain
    an alignment-independent scalar-gather fallback.
    """
    num_imgs = T.dynamic('num_imgs')
    total_raw_bytes = T.dynamic('total_raw_bytes')
    total_patches = T.dynamic('total_patches')

    pixels_per_patch = patch_size * patch_size
    elems_per_patch = 3 * pixels_per_patch
    num_cores = get_num_vector_cores()
    use_row_dma = patch_size <= 32
    threads = 256
    num_pixel_iters = (pixels_per_patch + threads - 1) // threads
    # Keep one full thread-width slot per channel. Tail lanes repeat the final
    # pixel in UB and are excluded from the GM copy, so every lane reaches the
    # shared-memory synchronization region.
    ub_channel_stride = threads
    raw_ub_row_stride = ((patch_size * 3 + 31) // 32) * 32
    simd_pixels = 64
    max_group_chunks = 12
    max_group_elems = 32768
    patches_per_simd = max(1, simd_pixels // patch_size)
    row_group_patches = max(
        1,
        min(
            max_group_chunks * simd_pixels // patch_size,
            max_group_elems // elems_per_patch,
        ),
    )
    row_group_pixels = row_group_patches * patch_size
    row_group_chunks = (row_group_pixels + simd_pixels - 1) // simd_pixels
    row_group_elems = row_group_patches * elems_per_patch
    row_group_raw_cols = row_group_pixels * 3
    row_group_raw_stride = ((row_group_raw_cols + 256 + 31) // 32) * 32
    # SIMD grouping pays off when one vector spans a moderate number of
    # patches.  This is a tiling regime, not an exact patch-size dispatch.
    use_grouped_row = use_row_dma and 4 <= patches_per_simd <= 8
    # Bisheng 15.0.5 can stack-smash while compiling the grouped SIMD path for
    # PTO. AUTO keeps the optimized path; PTO uses the safe row fallback.
    use_grouped_row = False if target_is_pto(Target.current(allow_none=False)) else use_grouped_row
    min_group_tasks = max(1, num_cores // 8)

    @T.macro
    def find_image(cu_seqlens_vit, patch_idx) -> T.int32:
        img_begin = T.alloc_var(T.int32, 0)
        img_end = T.alloc_var(T.int32, num_imgs)
        while img_begin < img_end:
            img_mid = T.alloc_var(T.int32, (img_begin + img_end) // 2)
            if patch_idx >= cu_seqlens_vit[img_mid + 1]:
                img_begin = img_mid + 1
            else:
                img_end = img_mid
        return img_begin

    @T.macro
    def normalize_bf16(raw_value, mean, inv_std) -> T.bfloat16:
        return T.cast(
            (T.cast(raw_value, T.float32) / 255.0 - mean) * inv_std,
            T.bfloat16,
        )

    @T.macro
    def process_row_patch(
        image_raw_concat,
        cu_raw_bytes,
        images_spatial_crop,
        cu_seqlens_vit,
        row_strides,
        flip_flags,
        mean,
        output,
        output_ub,
        raw_ub,
        inv_std,
        patch_idx,
    ):
        # Map the flattened patch index to an image in O(log num_imgs).
        img_id = T.int32(find_image(cu_seqlens_vit, patch_idx))

        n_vit_w = T.int32(images_spatial_crop[img_id, 0])
        n_vit_h = T.int32(images_spatial_crop[img_id, 1])
        local_patch = T.int32(patch_idx - cu_seqlens_vit[img_id])
        tile_row = T.int32(local_patch // n_vit_w)
        tile_col = T.int32(local_patch % n_vit_w)
        raw_base = T.int32(cu_raw_bytes[img_id])
        row_stride = T.alloc_var(T.int32)
        row_stride = row_strides[img_id]
        best_height = T.int32(n_vit_h * patch_size)
        do_flip = T.int32(flip_flags[img_id])

        # Load the whole patch rectangle once, then consume all 256-pixel
        # rounds from the same UB tile.
        dma_first_row = T.int32(
            T.if_then_else(
                do_flip == 1,
                best_height - (tile_row + 1) * patch_size,
                tile_row * patch_size,
            )
        )
        dma_src_offset = T.int32(raw_base + dma_first_row * row_stride + tile_col * patch_size * 3)
        raw_patch = T.make_tensor(
            T.access_ptr(
                image_raw_concat[dma_src_offset],
                'r',
                extent=(patch_size - 1) * row_stride + patch_size * 3,
            ),
            (patch_size, patch_size * 3),
            dtype=T.uint8,
            strides=(row_stride, 1),
        )
        T.copy(raw_patch, raw_ub[:, : patch_size * 3], data_select=True)

        for pixel_iter in T.serial(num_pixel_iters):
            pixel_offset = T.int32(pixel_iter * threads)
            valid_pixels = T.int32(T.min(threads, pixels_per_patch - pixel_offset))

            with T.SimtVF(threads=threads):
                thread_idx = T.get_thread_binding()
                pixel_idx = T.int32(pixel_offset + thread_idx)
                safe_pixel_idx = T.int32(T.min(pixel_idx, pixels_per_patch - 1))
                patch_pixel_row = T.int32(safe_pixel_idx // patch_size)
                ub_pixel_row = T.int32(
                    T.if_then_else(
                        do_flip == 1,
                        patch_size - 1 - patch_pixel_row,
                        patch_pixel_row,
                    )
                )
                ub_pixel_col = T.int32((safe_pixel_idx % patch_size) * 3)
                raw_r = raw_ub[ub_pixel_row, ub_pixel_col + 2]
                raw_g = raw_ub[ub_pixel_row, ub_pixel_col + 1]
                raw_b = raw_ub[ub_pixel_row, ub_pixel_col]

                output_ub[thread_idx] = normalize_bf16(raw_r, mean, inv_std)
                output_ub[ub_channel_stride + thread_idx] = normalize_bf16(raw_g, mean, inv_std)
                output_ub[2 * ub_channel_stride + thread_idx] = normalize_bf16(raw_b, mean, inv_std)

            for channel in T.serial(3):
                ub_offset = T.int32(channel * ub_channel_stride)
                channel_offset = T.int32(channel * pixels_per_patch + pixel_offset)
                T.copy(
                    output_ub[ub_offset : ub_offset + valid_pixels],
                    output[patch_idx, channel_offset : channel_offset + valid_pixels],
                )

    @T.macro
    def process_direct_round(
        image_raw_concat,
        cu_raw_bytes,
        images_spatial_crop,
        cu_seqlens_vit,
        row_strides,
        flip_flags,
        mean,
        output,
        output_ub,
        inv_std,
        patch_idx,
        pixel_iter,
    ):
        img_id = T.int32(find_image(cu_seqlens_vit, patch_idx))

        n_vit_w = T.int32(images_spatial_crop[img_id, 0])
        n_vit_h = T.int32(images_spatial_crop[img_id, 1])
        local_patch = T.int32(patch_idx - cu_seqlens_vit[img_id])
        tile_row = T.int32(local_patch // n_vit_w)
        tile_col = T.int32(local_patch % n_vit_w)
        raw_base = T.int32(cu_raw_bytes[img_id])
        row_stride = T.alloc_var(T.int32)
        row_stride = row_strides[img_id]
        best_height = T.int32(n_vit_h * patch_size)
        do_flip = T.int32(flip_flags[img_id])

        pixel_offset = T.int32(pixel_iter * threads)
        valid_pixels = T.int32(T.min(threads, pixels_per_patch - pixel_offset))

        with T.SimtVF(threads=threads):
            thread_idx = T.get_thread_binding()
            pixel_idx = T.int32(pixel_offset + thread_idx)
            safe_pixel_idx = T.int32(T.min(pixel_idx, pixels_per_patch - 1))
            source_row = T.int32(tile_row * patch_size + safe_pixel_idx // patch_size)
            pixel_row = T.if_then_else(
                do_flip == 1,
                best_height - 1 - source_row,
                source_row,
            )
            pixel_col = T.int32(tile_col * patch_size + safe_pixel_idx % patch_size)
            src_offset = T.int32(raw_base + pixel_row * row_stride + pixel_col * 3)
            raw_r = image_raw_concat[src_offset + 2]
            raw_g = image_raw_concat[src_offset + 1]
            raw_b = image_raw_concat[src_offset]

            output_ub[thread_idx] = normalize_bf16(raw_r, mean, inv_std)
            output_ub[ub_channel_stride + thread_idx] = normalize_bf16(raw_g, mean, inv_std)
            output_ub[2 * ub_channel_stride + thread_idx] = normalize_bf16(raw_b, mean, inv_std)

        for channel in T.serial(3):
            ub_offset = T.int32(channel * ub_channel_stride)
            channel_offset = T.int32(channel * pixels_per_patch + pixel_offset)
            T.copy(
                output_ub[ub_offset : ub_offset + valid_pixels],
                output[patch_idx, channel_offset : channel_offset + valid_pixels],
            )

    @T.macro
    def transform_simd_group(
        group_output_ub,
        group_raw_ub,
        raw_stage,
        channel_indices_ub,
        output_indices_ub,
        do_flip,
        norm_scale_f16,
        norm_bias_f16,
    ):
        with T.SimdVF():
            active_pixels = T.simd.pset(16, 'PAT_VL64')
            norm_scale_vec = T.simd.vdup(norm_scale_f16, 'float16', active_pixels)
            norm_bias_vec = T.simd.vdup(norm_bias_f16, 'float16', active_pixels)
            channel_indices = T.simd.alloc_local((3,), 'uint8')
            channel_indices[0] = T.simd.vld(channel_indices_ub[0, 0])
            channel_indices[1] = T.simd.vld(channel_indices_ub[1, 0])
            channel_indices[2] = T.simd.vld(channel_indices_ub[2, 0])
            destination_bases = T.simd.alloc_local((row_group_chunks,), 'uint16')
            for raw_chunk in T.unroll(row_group_chunks, explicit=True):
                destination_bases[raw_chunk] = T.simd.vld(output_indices_ub[raw_chunk, 0])

            for patch_pixel_row in T.serial(patch_size):
                ub_pixel_row = T.int32(
                    T.if_then_else(
                        do_flip == 1,
                        patch_size - 1 - patch_pixel_row,
                        patch_pixel_row,
                    )
                )
                for raw_chunk in T.unroll(row_group_chunks, explicit=True):
                    raw_bytes = T.simd.vld(group_raw_ub[raw_stage, ub_pixel_row, raw_chunk * simd_pixels * 3])
                    for channel in T.unroll(3, explicit=True):
                        output_delta = T.uint16(channel * pixels_per_patch + patch_pixel_row * patch_size)
                        channel_bytes = T.simd.vselr(raw_bytes, channel_indices[channel])
                        raw_u16 = T.simd.vcvt(channel_bytes, T.uint16, part=0)
                        raw_i16 = T.reinterpret(raw_u16, 'int16x128')
                        normalized_f16 = T.simd.alloc_var('float16')
                        normalized_f16 = T.simd.vcvt(raw_i16, T.float16)
                        T.simd.vmadd(
                            normalized_f16,
                            norm_scale_vec,
                            norm_bias_vec,
                            active_pixels,
                        )
                        normalized_bf16 = T.simd.vcvt(
                            normalized_f16,
                            T.bfloat16,
                            active_pixels,
                        )
                        destination_indices = T.simd.vadds(
                            destination_bases[raw_chunk],
                            output_delta,
                            active_pixels,
                        )
                        T.simd.vscatter(
                            normalized_bf16,
                            group_output_ub[raw_stage, 0],
                            destination_indices,
                            active_pixels,
                        )

    @T.macro
    def process_simd_group(
        image_raw_concat,
        cu_raw_bytes,
        images_spatial_crop,
        cu_seqlens_vit,
        row_strides,
        flip_flags,
        output,
        group_output_ub,
        group_raw_ub,
        channel_indices_ub,
        output_indices_ub,
        norm_scale_f16,
        norm_bias_f16,
        group_idx,
        raw_stage,
    ):
        group_start = T.int32(group_idx * row_group_patches)
        img_id = T.int32(find_image(cu_seqlens_vit, group_start))

        n_vit_w = T.int32(images_spatial_crop[img_id, 0])
        n_vit_h = T.int32(images_spatial_crop[img_id, 1])
        local_patch = T.int32(group_start - cu_seqlens_vit[img_id])
        tile_row = T.int32(local_patch // n_vit_w)
        tile_col = T.int32(local_patch % n_vit_w)

        raw_base = T.int32(cu_raw_bytes[img_id])
        row_stride = T.alloc_var(T.int32)
        row_stride = row_strides[img_id]
        best_height = T.int32(n_vit_h * patch_size)
        do_flip = T.int32(flip_flags[img_id])
        dma_first_row = T.int32(
            T.if_then_else(
                do_flip == 1,
                best_height - (tile_row + 1) * patch_size,
                tile_row * patch_size,
            )
        )
        dma_src_offset = T.int32(raw_base + dma_first_row * row_stride + tile_col * patch_size * 3)
        raw_group = T.make_tensor(
            T.access_ptr(
                image_raw_concat[dma_src_offset],
                'r',
                extent=(patch_size - 1) * row_stride + row_group_raw_cols,
            ),
            (patch_size, row_group_raw_cols),
            dtype=T.uint8,
            strides=(row_stride, 1),
        )
        T.copy(
            raw_group,
            group_raw_ub[raw_stage, :, :row_group_raw_cols],
            data_select=True,
        )

        output_group = T.make_tensor(
            T.access_ptr(output[group_start, 0], 'w', extent=row_group_elems),
            (row_group_elems,),
            dtype=T.bfloat16,
        )

        # Bisheng's vector selector requires a static UB stage in SIMD;
        # keep the matching MTE3 source static as well.
        if raw_stage == 0:
            transform_simd_group(
                group_output_ub,
                group_raw_ub,
                0,
                channel_indices_ub,
                output_indices_ub,
                do_flip,
                norm_scale_f16,
                norm_bias_f16,
            )
            T.copy(group_output_ub[0, :row_group_elems], output_group)
        else:
            transform_simd_group(
                group_output_ub,
                group_raw_ub,
                1,
                channel_indices_ub,
                output_indices_ub,
                do_flip,
                norm_scale_f16,
                norm_bias_f16,
            )
            T.copy(group_output_ub[1, :row_group_elems], output_group)

    @T.prim_func
    def _bmp_to_patches_ascend(
        image_raw_concat: T.Tensor([total_raw_bytes], T.uint8),  # type: ignore
        cu_raw_bytes: T.Tensor([num_imgs + 1], T.int32),  # type: ignore
        images_spatial_crop: T.Tensor([num_imgs, 2], T.int32),  # type: ignore
        cu_seqlens_vit: T.Tensor([num_imgs + 1], T.int32),  # type: ignore
        row_strides: T.Tensor([num_imgs], T.int32),  # type: ignore
        flip_flags: T.Tensor([num_imgs], T.int32),  # type: ignore
        mean: T.float32,
        std: T.float32,
        fp16_norm_safe: T.int32,
        output: T.Tensor([total_patches, elems_per_patch], T.bfloat16),  # type: ignore
    ) -> None:
        with T.Kernel(num_cores) as core_id:
            # For large patches, keep output UB bounded to one 256-pixel round.
            output_ub = T.alloc_shared((3 * ub_channel_stride,), T.bfloat16)
            inv_std = T.float32(1.0 / std)
            if use_row_dma:
                raw_ub = T.alloc_shared((patch_size, raw_ub_row_stride), T.uint8)
                run_row_fallback = T.alloc_var(T.int32, 1)
                T.ascend_set_copy_pad_value(0, dtype=T.uint8)
                if use_grouped_row:
                    group_output_ub = T.alloc_shared((2, row_group_elems + 640), T.bfloat16)
                    group_raw_ub = T.alloc_shared((2, patch_size, row_group_raw_stride), T.uint8)
                    channel_indices_ub = T.alloc_shared((3, 256), T.uint8)
                    output_indices_ub = T.alloc_shared((row_group_chunks, 128), T.uint16)
                    T.annotate_manual_multi_buffer(group_raw_ub, group_output_ub)
                    group_tasks = T.ceildiv(total_patches, row_group_patches)
                    if group_tasks >= min_group_tasks:
                        with T.SimtVF(threads=threads):
                            index_thread = T.get_thread_binding()
                            for lut_channel in T.serial(3):
                                if index_thread < 128 and index_thread % 2 == 0:
                                    channel_indices_ub[lut_channel, index_thread] = T.cast(
                                        (index_thread // 2) * 3 + (2 - lut_channel),
                                        T.uint8,
                                    )
                                else:
                                    channel_indices_ub[lut_channel, index_thread] = T.uint8(0)
                            if index_thread < 128:
                                for raw_chunk in T.serial(row_group_chunks):
                                    group_pixel = T.int32(raw_chunk * simd_pixels + index_thread)
                                    if index_thread < simd_pixels and group_pixel < row_group_pixels:
                                        output_indices_ub[raw_chunk, index_thread] = T.cast(
                                            (group_pixel // patch_size) * elems_per_patch + group_pixel % patch_size,
                                            T.uint16,
                                        )
                                    else:
                                        output_indices_ub[raw_chunk, index_thread] = T.cast(
                                            row_group_elems + index_thread,
                                            T.uint16,
                                        )
                        all_groups_full = T.alloc_var(T.int32)
                        all_groups_full = fp16_norm_safe
                        for scan_img in T.serial(num_imgs):
                            if images_spatial_crop[scan_img, 0] % row_group_patches != 0:
                                all_groups_full = 0
                        if all_groups_full == 1:
                            norm_scale_f16 = T.float16(1.0 / (255.0 * std))
                            norm_bias_f16 = T.float16(-mean / std)
                            run_row_fallback = 0
                            full_waves = group_tasks // num_cores
                            for group_iter in T.Pipelined(
                                full_waves,
                                num_stages=2,
                                annotations={'enable_offset': True},
                            ):
                                group_idx = T.int32(group_iter * num_cores + core_id)
                                raw_stage = group_iter % 2
                                process_simd_group(
                                    image_raw_concat,
                                    cu_raw_bytes,
                                    images_spatial_crop,
                                    cu_seqlens_vit,
                                    row_strides,
                                    flip_flags,
                                    output,
                                    group_output_ub,
                                    group_raw_ub,
                                    channel_indices_ub,
                                    output_indices_ub,
                                    norm_scale_f16,
                                    norm_bias_f16,
                                    group_idx,
                                    raw_stage,
                                )

                            tail_group_idx = T.int32(full_waves * num_cores + core_id)
                            if tail_group_idx < group_tasks:
                                tail_stage = full_waves % 2
                                process_simd_group(
                                    image_raw_concat,
                                    cu_raw_bytes,
                                    images_spatial_crop,
                                    cu_seqlens_vit,
                                    row_strides,
                                    flip_flags,
                                    output,
                                    group_output_ub,
                                    group_raw_ub,
                                    channel_indices_ub,
                                    output_indices_ub,
                                    norm_scale_f16,
                                    norm_bias_f16,
                                    tail_group_idx,
                                    tail_stage,
                                )
                if run_row_fallback == 1:
                    for patch_idx in T.Persistent([total_patches], num_cores, core_id, group_size=1, num_stages=1):
                        process_row_patch(
                            image_raw_concat,
                            cu_raw_bytes,
                            images_spatial_crop,
                            cu_seqlens_vit,
                            row_strides,
                            flip_flags,
                            mean,
                            output,
                            output_ub,
                            raw_ub,
                            inv_std,
                            patch_idx,
                        )
            else:
                # Direct-GM patches avoid that duplicated UB load and benefit
                # from distributing their 256-pixel rounds across the cores.
                for task_idx in T.Persistent(
                    [total_patches * num_pixel_iters], num_cores, core_id, group_size=1, num_stages=1
                ):
                    process_direct_round(
                        image_raw_concat,
                        cu_raw_bytes,
                        images_spatial_crop,
                        cu_seqlens_vit,
                        row_strides,
                        flip_flags,
                        mean,
                        output,
                        output_ub,
                        inv_std,
                        T.int32(task_idx // num_pixel_iters),
                        T.int32(task_idx % num_pixel_iters),
                    )

    return _bmp_to_patches_ascend
