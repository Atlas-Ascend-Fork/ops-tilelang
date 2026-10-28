# Architecture

## 分层

```text
框架或用户代码
        │
        ▼
cann_ops_tilelang 公共 API
  - 参数语义
  - device/dtype/shape/stride 校验
  - 输出内存分配
        │
        ▼
Ascend kernel generator
  - TileLang JIT 特化参数
  - block/core/pipeline 配置
  - GM/UB 数据搬运
  - SimdVF/SimtVF/Cube 计算
        │
        ▼
TileLang Ascend lowering / CANN toolchain
        │
        ▼
Ascend NPU
```

## 模块边界

### 公共 API

公共 API 必须只接受和返回 PyTorch Tensor、标量或简单配置对象。API 层负责：

- 确认运行设备为 Ascend NPU；
- 校验 shape、dtype、stride、layout 和可选输出；
- 计算不会依赖设备 IR 的 launch 参数；
- 调用 kernel generator 并执行 kernel；
- 对空 Tensor 给出定义良好的行为。

### Kernel generator

Kernel generator 使用 `@tilelang.jit`，参数分成两类：

- **编译期参数**：dtype、block size、是否启用某个融合分支；
- **运行期动态参数**：token 数、序列长度等可通过 `T.dynamic` 表达的维度。

不要把高基数、频繁变化的运行期值错误地放进编译期参数，否则会导致编译缓存膨胀。

### 配置层

`config.py` 只处理 Ascend 环境探测、核心数和调试开关。配置查询使用惰性调用，包导入阶段不得访问设备。

### 测试层

测试辅助工具和 PyTorch reference 全部位于 `tests/`，不属于运行时包，也不会进入 wheel。

测试框架由测试目录根部的 `tests/conftest.py` 统一装载插件，每个算子测试分为：

1. CPU 可执行的导入和参数校验；
2. NPU 正确性测试；
3. 显式启用的 benchmark；
4. NPU 峰值显存画像和 `large_npu_mem` 自动分组；
5. 按测试文件分片、按芯片集群选择的性能基线。

xdist worker 在导入 TileLang 前通过 `ASCEND_RT_VISIBLE_DEVICES` 绑定 NPU。普通测试和大/未知显存测试分两阶段执行；benchmark 独占每张 NPU。随机种子由 `--seed` 与 node ID 共同生成，保证并行调度变化时每个用例仍可复现。

### PyTorch reference 层

每个算子域的设备无关 PyTorch reference 与对应测试放在一起，例如 `tests/engram/engram_gate_ref.py`。reference 只作为数值语义、梯度和测试 oracle，不参与生产 kernel dispatch，也不构成 `cann_ops_tilelang` 的公开接口。Engram gate 的前向和反向均以该实现为基准。

## 示例调用链

`engram_fused_weight(weight_hidden, weight_embed)` 的调用链：

```text
cann_ops_tilelang.engram_fused_weight
  -> validate BF16/NPU/2-D/contiguous inputs
  -> choose block size and persistent scheduling from total elements
  -> get_engram_fused_weight_kernel(hidden_size, hc_mult)
  -> allocate an FP32 output
  -> launch flattened tensors
  -> compare with the FP32 PyTorch reference
```

Engram gate 训练链路：

```text
engram_fused_weight(weight_hidden, weight_embed)
  -> engram_gate_fwd(..., save_for_backward=True)
  -> output + (dot, gate_score, rstd_x, rstd_k)
  -> engram_gate_bwd(...)
  -> grad_hidden_states + grad_kv + grad_w_partial
  -> engram_grad_w_reduce(...)
  -> update grad_weight_hidden and grad_weight_embed in place
```

MHC 训练链路由显式前向和反向原语组成：

```text
expand_to_mhc_forward(...)
  -> mhc_fn_normw_merge_forward(...)
  -> mhc_gemm_with_sqrsum_forward(...)
  -> mhc_reduce_partials_and_rmsnorm_forward(...)
  -> mhc_pre_split_mixes_forward(...)
  -> mhc_sinkhorn_forward(...)
  -> mhc_pre_apply_mix_forward(...)
  -> upper-layer computation
  -> mhc_post_forward(...)

mhc_post_backward(...)
  -> upper-layer backward
  -> mhc_pre_apply_mix_backward(...)
  -> mhc_sinkhorn_backward(...)
  -> mhc_pre_split_mixes_backward(...)
  -> mhc_reduce_partials_and_rmsnorm_backward(...)
  -> mhc_gemm_with_sqrsum_backward(...)
  -> mhc_fn_normw_merge_backward(...)
  -> expand_to_mhc_backward(...)
```

`mhc_head_compute_mix_forward` 和 `mhc_head_compute_mix_backward` 处理 head mixing 系数，`mhc_multilayer_recompute` 融合多层 pre/post mix 重计算。`mhc_pre_big_fuse` 融合 reduce/RMSNorm、split mixes、Sinkhorn 和 pre apply mix 的前向路径。

## Engram fused weight 的 torch.compile 接入

`engram_fused_weight` 支持固定形状推理的 `npugraph_ex` 编译路径：

```python
import torch
import torch_npu

from cann_ops_tilelang import engram_fused_weight


def model(weight_hidden, weight_embed):
    return engram_fused_weight(-weight_hidden, weight_embed) + 1


compiled_model = torch.compile(
    model, backend='npugraph_ex', fullgraph=True, dynamic=False,
)
with torch.inference_mode():
    a = torch.randn(4, 4096, device='npu', dtype=torch.bfloat16)
    b = torch.randn_like(a)
    result = compiled_model(a, b)
```

编译时，公共接口调用 `cann_ops_tilelang::engram_fused_weight` 自定义算子；其 fake 实现只校验
shape、dtype、device 和布局，并推导同形状的 FP32 输出。实际执行仍使用原有 TileLang 内核。
公共接口只接收两个输入张量，直接返回 FP32 结果，不修改输入。普通 eager 调用保持原有路径。

本仓库要求 PyTorch >= 2.10；此接入使用 `torch.library.custom_op`，并需要提供
`npugraph_ex` 的配套 `torch_npu`。实机验证环境为 PyTorch / torch_npu 2.10。
输入仍须为连续的二维 BF16 NPU 张量，非空元素数须能被 256 整除；空输入也受支持。
这里按形状编译，形状变化可能触发重新编译，未注册 autograd 公式。
首次执行包含 TileLang JIT 和图准备开销。

以下测试包含原有 eager 用例和一个固定形状的 compile 基本功能用例，验证
`npugraph_ex` 编译执行的结果与 FP32 参考结果一致：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPS_TILELANG_TEST_LEVEL=0 \
    python -m pytest tests/engram/test_engram_fused_weight.py -m 'not benchmark' -v
```

## 扩展方向

- `src/cann_ops_tilelang/quant/`：公共 scaling-factor layout 和 FP8/FP4 类型系统；
- `src/cann_ops_tilelang/moe/`：在现有 TopK、routing 和 expert mapping 基础上扩展 expert dispatch/combine；
- `src/cann_ops_tilelang/norm/`：RMSNorm、LayerNorm 和融合残差；
- `src/cann_ops_tilelang/attention/`：RoPE、KV Cache、prefill/decode；
- `tests/<operator>/benchmark/`：在算子测试目录内按测试文件存储 JSONL 性能基线。
- `tests/<operator>/npu_mem_profiles/`：在算子测试目录内按测试文件存储 JSON 峰值显存画像。
