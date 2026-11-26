# 交叉熵前向

`cross_entropy_forward` 根据每行的 logits 和正确类别编号计算 loss。仅支持硬标签，即每行指定一个正确类别；返回结果不参与自动求导。

## 接口

```python
cross_entropy_forward(logits, targets, *, ignore_index=-100, reduction='none')
```

| 参数 | 要求 |
|---|---|
| `logits` | Ascend NPU 上的二维张量 `[N, V]`，支持 FP32、FP16、BF16；`N >= 0`，`V >= 1` |
| `targets` | 与 `logits` 在同一设备上的连续 int64 张量 `[N]`，每个值表示该行的正确类别编号 |
| `ignore_index` | 要忽略的标签值，默认 `-100`，需为 int64 范围内的整数 |
| `reduction` | `'none'`、`'sum'` 或 `'mean'`，默认 `'none'` |

`logits` 的每一行必须连续。两行之间可以有未使用的存储空间，但不能重叠。例如，连续的 `[N, V + 8]` 张量取前 `V` 列后仍可作为输入。

调用者需保证 `logits` 中的值有限，且 `targets` 除 `ignore_index` 外均在 `[0, V)` 内；接口不做逐元素数值和标签范围检查。

## 输出

输出为与输入同一设备上的 FP32 张量。

| `reduction` | 计算结果 | 输出形状 |
|---|---|---|
| `'none'` | 每行一个 loss；被忽略的行输出 0 | `[N]` |
| `'sum'` | 所有未被忽略的行的 loss 之和 | 标量 `[]` |
| `'mean'` | 所有未被忽略的行的 loss 之和，除以未被忽略的行数 | 标量 `[]` |

`N = 0` 时，`'none'` 返回空张量，`'sum'` 返回 0，`'mean'` 返回 NaN。所有行都被忽略时，`'none'` 返回全 0，`'sum'` 返回 0，`'mean'` 返回 NaN。

## 调用示例

```python
import torch
import torch_npu

from cann_ops_tilelang import cross_entropy_forward

logits = torch.randn((3, 5), device='npu', dtype=torch.float16)
targets = torch.tensor([0, 2, -100], device='npu', dtype=torch.int64)
loss = cross_entropy_forward(logits, targets, reduction='mean')
```

第三行标签为 `-100`，因此只对前两行的 loss 求平均。

需要配合 `torch.compile` 使用时（固定输入形状，在 `torch.no_grad()` 下执行）：

```python
compiled = torch.compile(cross_entropy_forward, backend='npugraph_ex', fullgraph=True, dynamic=False)
with torch.no_grad():
    loss = compiled(logits, targets, reduction='mean')
```

## 测试

在配置好 Ascend 环境并安装本仓库后，运行完整精度和接口测试：

```bash
OPS_TILELANG_TEST_LEVEL=2 pytest tests/loss -m 'not benchmark'
```

测试以 FP64 为参考，覆盖 FP32、FP16、BF16、三种 `reduction`、空输入、忽略标签、非连续行步长和 `torch.compile` 调用。

运行性能测试：

```bash
pytest tests/loss -m benchmark --run-benchmark
```

性能用例固定 `V = 129280`，包含 8 种行数、3 种输入类型和 3 种 `reduction`，共 72 档。计时方式和基线生成方法见[测试说明](../../tests/README.md)。最大档 `[65536, 129280]` 仅 FP32 `logits` 就约占 31.6 GiB，运行前需留出足够显存。
