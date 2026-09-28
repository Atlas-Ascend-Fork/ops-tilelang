import tilelang
import torch
from tilelang.ascend import language as T

from cann_ops_tilelang.config import get_num_vector_cores


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    }
)
def get_gather_kernel(dim: int, dtype: torch.dtype, index_dtype: torch.dtype):
    """Build the Ascend gather kernel."""
    num_rows = T.dynamic('num_rows')
    num_indices = T.dynamic('num_indices')

    if dim <= 4:
        tile_elements = 1024
    elif dim <= 128:
        tile_elements = 2048
    else:
        tile_elements = 16384
    block_dim = min(tilelang.next_power_of_2(dim), tile_elements)
    block_rows = tile_elements // block_dim
    num_cores = get_num_vector_cores()
    num_dim_blocks = T.ceildiv(dim, block_dim)

    if block_dim <= 8192:
        use_simtvf = block_dim <= 16
        num_row_blocks = T.ceildiv(num_indices, block_rows)
        num_stages = 1 if use_simtvf else 4

        if use_simtvf:
            num_tile_elements = block_rows * block_dim
            threads = min(2048, num_tile_elements)
            serial_iters = T.ceildiv(num_tile_elements, threads)

        @T.prim_func
        def gather_kernel(
            x: T.Tensor([num_rows, dim], dtype),
            index: T.Tensor([num_indices], index_dtype),
            output: T.Tensor([num_indices, dim], dtype),
        ):
            with T.Kernel(num_cores) as core_id:
                values_ub = T.alloc_shared([block_rows, block_dim], dtype)
                index_ub = T.alloc_shared([block_rows], index_dtype)

                for block in T.Persistent(
                    [num_row_blocks * num_dim_blocks], num_cores, core_id, group_size=1, num_stages=num_stages
                ):
                    row_block = block // num_dim_blocks
                    dim_block = block % num_dim_blocks
                    row_start = row_block * block_rows
                    dim_start = dim_block * block_dim
                    copy_rows = T.min(block_rows, num_indices - row_start)
                    copy_dim = T.min(block_dim, dim - dim_start)

                    T.copy(index[row_start : row_start + copy_rows], index_ub[:copy_rows])

                    if use_simtvf:
                        with T.SimtVF(threads=threads):
                            thread_id = T.get_thread_binding()
                            for iteration in T.serial(serial_iters):
                                element_idx = iteration * threads + thread_id
                                if element_idx < num_tile_elements:
                                    row_idx = element_idx // block_dim
                                    dim_idx = element_idx % block_dim
                                    if row_idx < copy_rows and dim_idx < copy_dim:
                                        source_idx = index_ub[row_idx]
                                        if source_idx >= 0 and source_idx < num_rows:
                                            values_ub[row_idx, dim_idx] = x[source_idx, dim_start + dim_idx]
                                        else:
                                            values_ub[row_idx, dim_idx] = 0
                    else:
                        for row_idx in T.serial(block_rows):
                            if row_idx < copy_rows:
                                source_idx = index_ub[row_idx]
                                if source_idx >= 0 and source_idx < num_rows:
                                    T.copy(
                                        x[source_idx, dim_start : dim_start + copy_dim],
                                        values_ub[row_idx, :copy_dim],
                                    )
                                else:
                                    T.fill(values_ub[row_idx, :copy_dim], 0)

                    T.copy(
                        values_ub[:copy_rows, :copy_dim],
                        output[
                            row_start : row_start + copy_rows,
                            dim_start : dim_start + copy_dim,
                        ],
                    )

    else:

        @T.prim_func
        def gather_kernel(
            x: T.Tensor([num_rows, dim], dtype),
            index: T.Tensor([num_indices], index_dtype),
            output: T.Tensor([num_indices, dim], dtype),
        ):
            with T.Kernel(num_cores) as core_id:
                values_ub = T.alloc_shared([block_dim], dtype)
                for row_idx, dim_block in T.Persistent(
                    [num_indices, num_dim_blocks],
                    num_cores,
                    core_id,
                    group_size=1,
                    num_stages=4,
                ):
                    dim_start = dim_block * block_dim
                    copy_length = T.min(block_dim, dim - dim_start)
                    source_idx = index[row_idx]
                    if source_idx >= 0 and source_idx < num_rows:
                        T.copy(x[source_idx, dim_start : dim_start + copy_length], values_ub[:copy_length])
                    else:
                        T.fill(values_ub, 0)
                    T.copy(values_ub[:copy_length], output[row_idx, dim_start : dim_start + copy_length])

    return gather_kernel
