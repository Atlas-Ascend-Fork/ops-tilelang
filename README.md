# ops-tilelang

`ops-tilelang` 是面向 Ascend 平台、基于 TileLang DSL 开发的高性能算子仓库。项目目标是为 CANN 社区提供结构清晰、容易验证、可持续调优和便于上层集成的算子实现。

> **硬件支持范围**：本代码仓仅支持 Ascend A5 及以上架构，不保证在更早的 Ascend 架构上可编译或正确运行。

当前仓库提供基础工程框架：Engram 算子族覆盖前向、反向、分片梯度归并及 Sinkhorn 行归一化；MoE 算子族提供 `topk_gate`、`moe_topk_gate` 和 `normalize_weight`。算子测试接入分级参数生成、确定性随机种子、NPU 显存画像、A5 性能回归框架、正则筛选、xdist 多卡绑定以及分阶段测试调度。

## 设计原则

- **接口与调度分离**：Python wrapper 负责语义、校验和内存；`*_kernel.py` 负责硬件调度。
- **按业务域聚合**：同一算子族的实现、导出和后续调优配置保持局部内聚。
- **正确性先于性能**：每个算子必须具备 PyTorch reference、边界测试和 NPU 正确性测试。
- **性能可回归**：benchmark 与普通测试分离，基线按测试文件和芯片集群分片保存。
- **避免隐式全局行为**：包导入不初始化 NPU，也不编译 kernel；编译只发生在算子首次调用时。

## 目录结构

```text
ops-tilelang/
├── src/cann_ops_tilelang/
│   ├── config.py                    # Ascend 环境与运行配置
│   ├── utils.py                     # 通用纯 Python 工具
│   ├── engram/                       # Engram 前向、反向和梯度归并算子
│   └── moe/                          # MoE 路由和权重归一化算子
├── tests/
│   ├── testing/                      # 参数生成、数值断言和 benchmark 工具
│   ├── pytest_*_plugin.py            # benchmark/显存/seed/xdist/warning 插件
│   ├── engram/                       # Engram 测试与 PyTorch reference
│   └── moe/                          # MoE 测试与 PyTorch reference
├── docs/                            # 架构和新增算子指南
├── scripts/                         # 本地/CI 执行入口
└── pyproject.toml                   # 构建、依赖和工具配置
```

## 环境要求

- Python 3.10+
- Ascend A5 及以上架构的 NPU
- 可用的 CANN 与 Ascend 驱动环境
- 与 CANN 版本匹配的 PyTorch NPU
- 支持 Ascend target 的 TileLang

仓库只声明通用的 `torch` 和 `tilelang` Python 依赖。`torch_npu`、CANN Toolkit 与驱动具有严格的版本配套关系，应由部署环境安装，不在 PyPI 依赖中强行固定。

## 安装

### 安装本地开发版本

```bash
pip install -e ".[dev]"
```

### 安装发布版本

```bash
pip install ops-tilelang
```

发行包名保持为 `ops-tilelang`，Python 导入命名空间为 `cann_ops_tilelang`。

### 构建并安装 Wheel 包

在已安装开发依赖的源码目录中构建 wheel：

```bash
./scripts/build_wheel.sh
```

构建产物位于 `dist/`，可以直接安装：

```bash
pip install dist/ops_tilelang-*.whl
```

wheel 只包含运行时算子代码，不包含 `tests/` 下的测试工具和 PyTorch reference；`torch_npu`、CANN Toolkit、驱动和与环境匹配的 TileLang 仍需由目标 Ascend 环境提供。

## 使用示例

`engram_fused_weight` 将两份二维 BF16 权重在 FP32 中相乘：

```python
from cann_ops_tilelang import engram_fused_weight

weight_hidden = torch.randn((4, 4096), dtype=torch.bfloat16, device='npu')
weight_embed = torch.randn_like(weight_hidden)
weight_fused = engram_fused_weight(weight_hidden, weight_embed)
assert weight_fused.dtype == torch.float32
```

Engram gate 前向可选择保存反向所需的统计量，反向产生输入梯度和权重分片梯度，随后由 `engram_grad_w_reduce` 原地归并到两份 FP32 权重梯度：

```python
from cann_ops_tilelang import engram_gate_bwd, engram_gate_fwd, engram_grad_w_reduce

output, dot, gate, rstd_x, rstd_k = engram_gate_fwd(
    hidden_states, kv, weight_fused, eps=1e-20, clamp_value=1e-6
)
grad_x, grad_kv, grad_w_partial = engram_gate_bwd(
    grad_out, hidden_states, kv, weight_fused, dot, gate, rstd_x, rstd_k, clamp_value=1e-6
)
engram_grad_w_reduce(
    grad_w_partial, weight_hidden, weight_embed, grad_weight_hidden, grad_weight_embed
)
```

`engram_sinkhorn_step` 对二维 FP32 矩阵执行一次 Sinkhorn 行归一化，并原地更新行缩放系数：

```python
from cann_ops_tilelang import engram_sinkhorn_step

matrix = torch.randn((1001, 256), dtype=torch.float32, device='npu')
row_scale = torch.rand((1001,), dtype=torch.float32, device='npu')
col_scale = torch.rand((256,), dtype=torch.float32, device='npu')
row_norm, col_sumsq = engram_sinkhorn_step(matrix, row_scale, col_scale, eps=1e-20)
```

三个输入必须位于同一 NPU、连续且为 FP32，并且不能共享存储。`matrix` 的形状为 `(num_rows, head_dim)`，`head_dim` 必须是正的 256 倍数；调用方需保证 `row_scale` 包含非负的 Sinkhorn 缩放系数。已验证的典型维度为 256 和 5120。由于 `SimtVF` 需预留线程上下文，wrapper 会拒绝共享 UB 估算超过 216 KiB 的 shape。`num_rows=0` 时返回空的 `row_norm` 和全零的 `col_sumsq`。

MoE 路由算子接收连续的 NPU FP32 分数。`topk_gate` 稳定选择每个 token 的专家索引，分数相同时选择较小索引；`moe_topk_gate` 融合打分、偏置、掩码、固定路由、随机路由和逻辑到物理专家映射；`normalize_weight` 返回每行分母及归一化权重：

```python
from cann_ops_tilelang import moe_topk_gate, normalize_weight, topk_gate

topk_idx = topk_gate(scores, num_topk=6)
denominator, normalized_weights = normalize_weight(topk_weights)
topk_idx, topk_weights = moe_topk_gate(
    logits,
    num_topk=6,
    use_shared_as_routed=False,
    num_shared_experts=1,
    routed_scaling_factor=1.0,
    ep_rank=0,
    scoring_func='sigmoid',
)
```

## 测试

```bash
# CPU 可执行的导入与接口测试
pytest -m 'not npu and not benchmark'

# Ascend 正确性测试
pytest -m 'npu and not benchmark'

# 显式运行性能回归；-K 可按参数表达式筛选
pytest -m benchmark --run-benchmark -K 'hidden=4096'

# 刷新 A5 性能基线和 NPU 峰值显存画像
pytest tests/engram/test_engram_fused_weight.py -m benchmark --run-benchmark --refresh-baseline
pytest tests/engram/test_engram_fused_weight.py -m 'not benchmark' --refresh-mem

# 刷新整个 Engram 算子族
pytest tests/engram -m benchmark --run-benchmark --refresh-baseline
pytest tests/engram -m 'not benchmark' --refresh-mem

# 运行或刷新 MoE 算子族
pytest tests/moe -m 'npu and not benchmark'
pytest tests/moe -m benchmark --run-benchmark --refresh-baseline
pytest tests/moe -m 'not benchmark' --refresh-mem

# 标准入口
./scripts/run_test.sh
```

`OPS_TILELANG_TEST_LEVEL=0|1|2` 控制测试参数规模；`--seed` 提供逐用例可复现随机数。完整的插件参数与基线维护流程见 [tests/README.md](tests/README.md)。

设置 `OPS_TILELANG_PRINT_KERNEL_SOURCE=1` 可以在首次构建 kernel 时打印生成源码。

### CI 执行入口

`scripts/run_ci.sh` 通过 `OPS_TILELANG_CI_TASK` 选择执行阶段：

```bash
OPS_TILELANG_CI_TASK=lint ./scripts/run_ci.sh
OPS_TILELANG_CI_TASK=cpu ./scripts/run_ci.sh
OPS_TILELANG_CI_TASK=wheel ./scripts/run_ci.sh
OPS_TILELANG_CI_TASK=npu ./scripts/run_ci.sh
OPS_TILELANG_CI_TASK=benchmark ./scripts/run_ci.sh tests/engram

# 在 Ascend A5 及以上 CI runner 上执行全部阶段
OPS_TILELANG_CI_TASK=all ./scripts/run_ci.sh
```

NPU 正确性测试默认每卡 2 个 worker，大/未知显存测试默认每卡 1 个 worker。可通过 `TILELANG_TEST_WORKERS_PER_NPU` 和 `TILELANG_TEST_WORKERS_PER_NPU_LARGE` 调整。

## 新增算子

请阅读 [docs/adding_an_operator.md](docs/adding_an_operator.md)。新增算子至少需要：

1. 明确、类型化的公共 API；
2. Ascend TileLang kernel；
3. PyTorch reference 和正确性测试；
4. 空 Tensor、非法 dtype/shape 等边界测试；
5. 代表性 shape 的 benchmark；
6. 文档中写明支持的 dtype、shape、layout 和已知限制。

## License

MIT License，详见 [LICENSE](LICENSE)。
