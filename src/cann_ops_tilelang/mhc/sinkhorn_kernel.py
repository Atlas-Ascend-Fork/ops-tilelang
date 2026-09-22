import tilelang
from tilelang.ascend import language as T
from tilelang.ascend.language import simd as S

_TOKEN_TILE = 64
_PASS_CONFIGS = {'tl.enable_fast_math': True}


@tilelang.jit(target='pto', pass_configs=_PASS_CONFIGS)
def mhc_sinkhorn_forward_kernel(
    x,
    out,
    repeat: int,
    eps: float,
    tile_num: int,
    num_cores: int,
    threads: int,
):
    """PTO Sinkhorn forward using an element-major 64-token SIMD tile."""

    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')

    x: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    out: T.Tensor[(num_tokens, mhc, mhc), T.float32]

    with T.Kernel(num_cores) as core_id:
        matrix_ub = T.alloc_shared((mhc, mhc, _TOKEN_TILE), T.float32)

        for tile_id in T.Persistent(
            [tile_num],
            num_cores,
            core_id,
            group_size=1,
            num_stages=0,
        ):
            token_start = tile_id * _TOKEN_TILE
            valid_tokens = T.min(_TOKEN_TILE, num_tokens - token_start)

            with T.SimtVF(threads=threads):
                load_thread_idx = T.get_thread_binding()
                load_token = load_thread_idx // 16
                load_element = load_thread_idx % 16
                row = load_element // mhc
                col = load_element % mhc
                if load_token < valid_tokens:
                    matrix_ub[row, col, load_token] = x[token_start + load_token, row, col]
                else:
                    matrix_ub[row, col, load_token] = 0.0

            with T.SimdVF():
                softmax_zero = S.vdup(0.0, T.float32)
                one = S.vdup(1.0, T.float32)
                initial_eps = S.vdup(eps, T.float32)
                values = S.alloc_local((mhc, mhc), T.float32)
                row_max = S.alloc_var(T.float32)
                reduction = S.alloc_var(T.float32)
                recip = S.alloc_var(T.float32)

                for row in T.unroll(mhc, explicit=True):
                    for col in T.unroll(mhc, explicit=True):
                        values[row, col] = S.vld(matrix_ub[row, col, 0])

                for row in T.unroll(mhc, explicit=True):
                    row_max = values[row, 0]
                    for col in T.unroll(mhc - 1, explicit=True):
                        row_max = S.vmax(row_max, values[row, col + 1])
                    reduction = softmax_zero
                    for col in T.unroll(mhc, explicit=True):
                        values[row, col] = S.vexpdif(values[row, col], row_max)
                        reduction = S.vadd(reduction, values[row, col])
                    recip = S.vdiv(one, reduction)
                    for col in T.unroll(mhc, explicit=True):
                        values[row, col] = S.vadds(S.vmul(values[row, col], recip), eps)

                for col in T.unroll(mhc, explicit=True):
                    reduction = initial_eps
                    for row in T.unroll(mhc, explicit=True):
                        reduction = S.vadd(reduction, values[row, col])
                    recip = S.vdiv(one, reduction)
                    for row in T.unroll(mhc, explicit=True):
                        values[row, col] = S.vmul(values[row, col], recip)

                for _ in T.serial(repeat - 1):
                    for row in T.unroll(mhc, explicit=True):
                        reduction = initial_eps
                        for col in T.unroll(mhc, explicit=True):
                            reduction = S.vadd(reduction, values[row, col])
                        recip = S.vdiv(one, reduction)
                        for col in T.unroll(mhc, explicit=True):
                            values[row, col] = S.vmul(values[row, col], recip)
                    for col in T.unroll(mhc, explicit=True):
                        reduction = initial_eps
                        for row in T.unroll(mhc, explicit=True):
                            reduction = S.vadd(reduction, values[row, col])
                        recip = S.vdiv(one, reduction)
                        for row in T.unroll(mhc, explicit=True):
                            values[row, col] = S.vmul(values[row, col], recip)

                for row in T.unroll(mhc, explicit=True):
                    for col in T.unroll(mhc, explicit=True):
                        S.vsts(matrix_ub[row, col, 0], values[row, col])

            with T.SimtVF(threads=threads):
                store_thread_idx = T.get_thread_binding()
                store_token = store_thread_idx // 16
                store_element = store_thread_idx % 16
                row = store_element // mhc
                col = store_element % mhc
                if store_token < valid_tokens:
                    out[token_start + store_token, row, col] = matrix_ub[row, col, store_token]


@tilelang.jit(target='pto', pass_configs=_PASS_CONFIGS)
def mhc_sinkhorn_backward_kernel(
    grad_output,
    x,
    grad_input,
    repeat: int,
    eps: float,
    tile_num: int,
    num_cores: int,
    threads: int,
):
    """PTO Sinkhorn backward using reciprocal-multiply normalization."""

    num_tokens = T.dynamic('num_tokens')
    mhc = T.const('mhc')
    num_states = repeat * 2

    grad_output: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    x: T.Tensor[(num_tokens, mhc, mhc), T.float32]
    grad_input: T.Tensor[(num_tokens, mhc, mhc), T.float32]

    with T.Kernel(num_cores) as core_id:
        matrix_ub = T.alloc_shared((mhc, mhc, _TOKEN_TILE), T.float32)
        grad_ub = T.alloc_shared((mhc, mhc, _TOKEN_TILE), T.float32)
        states_ub = T.alloc_shared((num_states, mhc, mhc, _TOKEN_TILE), T.float32)
        sums_ub = T.alloc_shared((num_states, mhc, _TOKEN_TILE), T.float32)

        for tile_id in T.Persistent(
            [tile_num],
            num_cores,
            core_id,
            group_size=1,
            num_stages=0,
        ):
            token_start = tile_id * _TOKEN_TILE
            valid_tokens = T.min(_TOKEN_TILE, num_tokens - token_start)

            with T.SimtVF(threads=threads):
                load_thread_idx = T.get_thread_binding()
                load_token = load_thread_idx // 16
                load_element = load_thread_idx % 16
                row = load_element // mhc
                col = load_element % mhc
                if load_token < valid_tokens:
                    matrix_ub[row, col, load_token] = x[token_start + load_token, row, col]
                    grad_ub[row, col, load_token] = grad_output[token_start + load_token, row, col]
                else:
                    matrix_ub[row, col, load_token] = 0.0
                    grad_ub[row, col, load_token] = 0.0

            with T.SimdVF():
                zero = S.vdup(0.0, T.float32)
                one = S.vdup(1.0, T.float32)
                initial_eps = S.vdup(eps, T.float32)
                values = S.alloc_local((mhc, mhc), T.float32)
                row_max = S.alloc_var(T.float32)
                reduction = S.alloc_var(T.float32)
                recip = S.alloc_var(T.float32)

                for row in T.unroll(mhc, explicit=True):
                    for col in T.unroll(mhc, explicit=True):
                        values[row, col] = S.vld(matrix_ub[row, col, 0])

                for row in T.unroll(mhc, explicit=True):
                    row_max = values[row, 0]
                    for col in T.unroll(mhc - 1, explicit=True):
                        row_max = S.vmax(row_max, values[row, col + 1])
                    reduction = zero
                    for col in T.unroll(mhc, explicit=True):
                        values[row, col] = S.vexpdif(values[row, col], row_max)
                        reduction = S.vadd(reduction, values[row, col])
                    recip = S.vdiv(one, reduction)
                    for col in T.unroll(mhc, explicit=True):
                        values[row, col] = S.vmul(values[row, col], recip)
                        S.vsts(states_ub[0, row, col, 0], values[row, col])
                        values[row, col] = S.vadds(values[row, col], eps)
                        S.vsts(states_ub[1, row, col, 0], values[row, col])

                for col in T.unroll(mhc, explicit=True):
                    reduction = initial_eps
                    for row in T.unroll(mhc, explicit=True):
                        reduction = S.vadd(reduction, values[row, col])
                    S.vsts(sums_ub[1, col, 0], reduction)
                    recip = S.vdiv(one, reduction)
                    for row in T.unroll(mhc, explicit=True):
                        values[row, col] = S.vmul(values[row, col], recip)
                        if repeat > 1:
                            S.vsts(states_ub[2, row, col, 0], values[row, col])

                for step in T.serial(max(repeat - 2, 0)):
                    row_state = step * 2 + 2
                    col_state = row_state + 1

                    for row in T.unroll(mhc, explicit=True):
                        reduction = initial_eps
                        for col in T.unroll(mhc, explicit=True):
                            reduction = S.vadd(reduction, values[row, col])
                        S.vsts(sums_ub[row_state, row, 0], reduction)
                        recip = S.vdiv(one, reduction)
                        for col in T.unroll(mhc, explicit=True):
                            values[row, col] = S.vmul(values[row, col], recip)
                            S.vsts(states_ub[col_state, row, col, 0], values[row, col])

                    for col in T.unroll(mhc, explicit=True):
                        reduction = initial_eps
                        for row in T.unroll(mhc, explicit=True):
                            reduction = S.vadd(reduction, values[row, col])
                        S.vsts(sums_ub[col_state, col, 0], reduction)
                        recip = S.vdiv(one, reduction)
                        for row in T.unroll(mhc, explicit=True):
                            values[row, col] = S.vmul(values[row, col], recip)
                            S.vsts(states_ub[row_state + 2, row, col, 0], values[row, col])

                if repeat > 1:
                    for row in T.unroll(mhc, explicit=True):
                        reduction = initial_eps
                        for col in T.unroll(mhc, explicit=True):
                            reduction = S.vadd(reduction, values[row, col])
                        S.vsts(sums_ub[num_states - 2, row, 0], reduction)
                        recip = S.vdiv(one, reduction)
                        for col in T.unroll(mhc, explicit=True):
                            values[row, col] = S.vmul(values[row, col], recip)
                            S.vsts(states_ub[num_states - 1, row, col, 0], values[row, col])

                    for col in T.unroll(mhc, explicit=True):
                        reduction = initial_eps
                        for row in T.unroll(mhc, explicit=True):
                            reduction = S.vadd(reduction, values[row, col])
                        S.vsts(sums_ub[num_states - 1, col, 0], reduction)
                S.mem_bar('VST_VLD')

            with T.SimdVF():
                reverse_zero = S.vdup(0.0, T.float32)
                grad_zero = S.vdup(0.0, T.float32)
                one_r = S.vdup(1.0, T.float32)
                grads = S.alloc_local((mhc, mhc), T.float32)
                states = S.alloc_local((mhc,), T.float32)
                dot = S.alloc_var(T.float32)
                denom = S.alloc_var(T.float32)
                correction = S.alloc_var(T.float32)
                recip = S.alloc_var(T.float32)

                for row in T.unroll(mhc, explicit=True):
                    for col in T.unroll(mhc, explicit=True):
                        grads[row, col] = S.vld(grad_ub[row, col, 0])

                for reverse_pair in T.serial(repeat - 1):
                    col_state = num_states - 1 - reverse_pair * 2
                    row_state = col_state - 1

                    for col in T.unroll(mhc, explicit=True):
                        dot = reverse_zero
                        for row in T.unroll(mhc, explicit=True):
                            states[row] = S.vld(states_ub[col_state, row, col, 0])
                            dot = S.vadd(dot, S.vmul(grads[row, col], states[row]))
                        denom = S.vld(sums_ub[col_state, col, 0])
                        recip = S.vdiv(one_r, denom)
                        correction = S.vmul(dot, recip)
                        for row in T.unroll(mhc, explicit=True):
                            grads[row, col] = S.vmul(S.vsub(grads[row, col], correction), recip)

                    for row in T.unroll(mhc, explicit=True):
                        dot = reverse_zero
                        for col in T.unroll(mhc, explicit=True):
                            states[col] = S.vld(states_ub[row_state, row, col, 0])
                            dot = S.vadd(dot, S.vmul(grads[row, col], states[col]))
                        denom = S.vld(sums_ub[row_state, row, 0])
                        recip = S.vdiv(one_r, denom)
                        correction = S.vmul(dot, recip)
                        for col in T.unroll(mhc, explicit=True):
                            grads[row, col] = S.vmul(S.vsub(grads[row, col], correction), recip)

                for col in T.unroll(mhc, explicit=True):
                    dot = grad_zero
                    for row in T.unroll(mhc, explicit=True):
                        states[row] = S.vld(states_ub[1, row, col, 0])
                        dot = S.vadd(dot, S.vmul(grads[row, col], states[row]))
                    denom = S.vld(sums_ub[1, col, 0])
                    recip = S.vdiv(one_r, denom)
                    correction = S.vmul(dot, recip)
                    for row in T.unroll(mhc, explicit=True):
                        grads[row, col] = S.vmul(S.vsub(grads[row, col], correction), recip)

                for row in T.unroll(mhc, explicit=True):
                    dot = grad_zero
                    for col in T.unroll(mhc, explicit=True):
                        states[col] = S.vld(states_ub[0, row, col, 0])
                        dot = S.vadd(dot, S.vmul(grads[row, col], states[col]))
                    for col in T.unroll(mhc, explicit=True):
                        grads[row, col] = S.vmul(S.vsub(grads[row, col], dot), states[col])

                for row in T.unroll(mhc, explicit=True):
                    for col in T.unroll(mhc, explicit=True):
                        S.vsts(grad_ub[row, col, 0], grads[row, col])

            with T.SimtVF(threads=threads):
                store_thread_idx = T.get_thread_binding()
                store_token = store_thread_idx // 16
                store_element = store_thread_idx % 16
                row = store_element // mhc
                col = store_element % mhc
                if store_token < valid_tokens:
                    grad_input[token_start + store_token, row, col] = grad_ub[row, col, store_token]
