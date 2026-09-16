# 新增算子开发流程

本文描述一个 Ascend TileLang 算子从接口设计到性能基线入库的完整流程。单输出算子可参考 `engram_fused_weight`；包含动态 token、前反向中间量和分片梯度归并的算子族可参考 `engram_gate_fwd`、`engram_gate_bwd` 与 `engram_grad_w_reduce`。

## 1. 明确算子契约与命名

编码前先确定以下内容：

- 输入、输出及可选 `out=` 参数的语义；
- 支持的 dtype、shape、layout、stride 和对齐要求；
- 空 Tensor、非连续 Tensor、别名和溢出的行为；
- PyTorch reference；
- 典型、小尺寸、边界和高负载 shape；
- 性能指标是时延、带宽、吞吐量还是组合指标。

算子、文件和公开函数统一使用 `snake_case`。名称应包含必要的业务域，避免过于通用的名称。例如使用 `engram_fused_weight`，不要仅使用 `fused_weight`。

建议的内部命名约定：

```text
公开函数                 <operator>
接口文件                 <operator>.py
Kernel 实现文件          <operator>_kernel.py
kernel generator         get_<operator>_kernel
TileLang prim_func       <operator>_kernel
测试文件                 test_<operator>.py
```

## 2. 创建目录

如果算子属于已有业务域，放入该域目录；否则先创建业务域。以下以 `foo_bar` 为例：

```text
src/cann_ops_tilelang/foo/
├── __init__.py
├── foo_bar.py
└── foo_bar_kernel.py

tests/foo/
├── test_foo_bar.py
├── benchmark/
│   └── test_foo_bar.jsonl
└── npu_mem_profiles/
    └── test_foo_bar.json
```

`benchmark/` 和 `npu_mem_profiles/` 中的文件由测试运行生成或更新。性能基线和显存画像必须通过 pytest 刷新命令生成，不要手工拼接记录。

## 3. 编写公共接口

`foo_bar.py` 只处理用户可见语义和 kernel launch，通常需要完成：

1. 检查参数是否为 `torch.Tensor`；
2. 调用 `require_ascend()`，禁止静默回退到 CPU 或 CUDA；
3. 检查 NPU device、dtype、shape、layout、stride 和 contiguous；
4. 检查硬件或 kernel 要求的长度、分块和对齐约束；
5. 校验可选 `out=` 的 shape、dtype、device、连续性和别名关系；
6. 为 `out is None` 分配输出；
7. 在调用 kernel 前处理空 Tensor；
8. 选择编译期参数并获取 kernel；
9. 根据 `OPS_TILELANG_PRINT_KERNEL_SOURCE` 打印生成源码；
10. launch kernel 并返回结果。

校验失败应抛出 `TypeError`、`ValueError` 或 `RuntimeError`。公共接口不要使用可被 `python -O` 删除的 `assert`。

简化结构如下：

```python
import torch

from cann_ops_tilelang.config import require_ascend, should_print_kernel_source

from .foo_bar_kernel import get_foo_bar_kernel


def foo_bar(x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    require_ascend()
    # 完成 device、dtype、shape、layout、alignment 和 out 校验。
    result = torch.empty_like(x) if out is None else out
    if x.numel() == 0:
        return result

    kernel = get_foo_bar_kernel(...)
    if should_print_kernel_source():
        print(kernel.get_kernel_source())
    kernel(x, result)
    return result
```

不要在模块导入阶段查询 NPU 属性或触发 kernel 编译；这些行为应推迟到第一次算子调用。

## 4. 编写 Ascend TileLang kernel

`foo_bar_kernel.py` 只放置 kernel generator 和私有调度辅助函数。设备与核心数查询统一使用 `cann_ops_tilelang.config`。

编译期参数与运行期参数需要明确区分：

- dtype、block size、pipeline stage 和融合开关通常是编译期参数；
- token 数、有效长度等高基数维度优先使用 `T.dynamic`；
- 不要把频繁变化的运行期值全部放入 generator 参数，否则会扩大 JIT 缓存。

推荐按以下顺序演进：

1. 实现保守、容易与 reference 对齐的版本；
2. 确认尾块、空输入和非整除 shape 行为；
3. 引入 GM/UB 分块和 `T.SimdVF` 或 `T.SimtVF`；
4. 使用 `T.Persistent` 做核间任务分配；
5. 使用 `T.annotate_buffer_versions` 和多 stage pipeline；
6. 有明确收益时再引入 Cube/Vector MixedKernel。

kernel 内部可使用断言验证编译期不变量，但用户输入错误必须在公共接口中转成清晰异常。

## 5. 导出公开 API

稳定算子需要逐级显式导出：

```python
# src/cann_ops_tilelang/foo/__init__.py
from .foo_bar import foo_bar

__all__ = ['foo_bar']
```

然后更新：

- `src/cann_ops_tilelang/__init__.py`；

不要使用 `import *`。实验性接口可以暂时只通过完整模块路径访问，不应提前加入根包。

## 6. 编写正确性与契约测试

测试文件应复用 `tests/testing/` 中的仓内测试工具：

- `get_test_level()`：读取 `OPS_TILELANG_TEST_LEVEL=0|1|2`；
- `generate_samples()` 及各类 shape generator：生成分级参数；
- `make_param_id()`：生成稳定、可筛选的 pytest ID；
- `assert_equal()` 或其他数值断言：比较 TileLang 输出与 reference；
- 根 pytest 插件：根据 `--seed` 为每个 node ID 设置稳定随机种子。

Ascend-only 测试需要添加 `pytest.mark.npu`：

```python
import pytest
import torch

import cann_ops_tilelang
from cann_ops_tilelang.config import get_device
from tests.testing.bench import make_param_id
from tests.testing.generator import get_test_level
from tests.testing.numeric import assert_equal


pytestmark = pytest.mark.npu


def generate_test_params(level: int) -> list[dict]:
    # level 0 为核心集合，1 为默认集合，2 为完整/边界集合。
    ...


@pytest.mark.parametrize('params', generate_test_params(get_test_level()), ids=make_param_id)
def test_foo_bar(params):
    x = ...
    ref = ...
    out = cann_ops_tilelang.foo_bar(x)
    assert_equal(out, ref)
```

至少覆盖：

- 所有支持的 dtype；
- 核心、典型、最小、最大和边界 shape；
- 整除与尾块场景；
- 空 Tensor；
- 非法 device、dtype、shape、layout、stride 和对齐；
- 非连续输入；
- `out=` 缓冲区及不允许的别名；
- 输出 shape、dtype、device 和数值结果。

## 7. 添加 benchmark

benchmark 与正确性测试放在同一测试文件中，使用统一 fixture：

```python
from tests.testing.numeric import count_bytes


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(0), ids=make_param_id)
def test_foo_bar_benchmark(benchmark_timer, benchmark_record, params):
    x = ...
    out = cann_ops_tilelang.foo_bar(x)

    time_us = benchmark_timer(lambda: cann_ops_tilelang.foo_bar(x))
    num_bytes = count_bytes(x, out)
    benchmark_record(
        kernel='foo_bar',
        operation='fwd',
        params=params,
        time_us=time_us,
        bandwidth_gbs=num_bytes / time_us / 1e3,
    )
```

Ascend benchmark 固定使用 `msprof` 后端并归入 `a5` 集群。`kernel` 字段必须与公开算子名一致，改名时需要重新生成基线。

## 8. 运行快速检查与正确性测试

先运行静态检查和 CPU 可执行测试：

```bash
./scripts/check.sh
```

再从核心参数集开始验证目标算子：

```bash
OPS_TILELANG_TEST_LEVEL=0 \
  pytest tests/foo/test_foo_bar.py -m 'not benchmark'
```

核心集合通过后运行默认和完整集合：

```bash
OPS_TILELANG_TEST_LEVEL=1 pytest tests/foo/test_foo_bar.py -m 'not benchmark'
OPS_TILELANG_TEST_LEVEL=2 pytest tests/foo/test_foo_bar.py -m 'not benchmark'
```

最后验证标准两阶段 xdist 调度：

```bash
./scripts/run_test.sh tests/foo/test_foo_bar.py
```

第一阶段并行执行普通显存用例；第二阶段以更少 worker 执行大显存或尚无画像的用例。

## 9. 生成 NPU 显存画像

新增用例在没有画像时会被保守标记为 `large_npu_mem`。正确性稳定后刷新画像：

```bash
pytest tests/foo/test_foo_bar.py -m 'not benchmark' --refresh-mem
```

只补充全仓缺失项时使用：

```bash
pytest --refresh-mem --refresh-mem-missing-only
```

检查生成的 `tests/foo/npu_mem_profiles/test_foo_bar.json`，确认用例数量、node ID 和峰值显存合理。默认 15000 MB 以上的用例进入第二阶段，可用 `--npu-mem-threshold` 临时调整阈值。

## 10. 生成并验证 A5 性能基线

第一次生成基线：

```bash
pytest tests/foo/test_foo_bar.py \
  -m benchmark \
  --run-benchmark \
  --refresh-baseline
```

刷新后必须立即再运行一次回归比较，确认全部记录命中：

```bash
pytest tests/foo/test_foo_bar.py \
  -m benchmark \
  --run-benchmark \
  --benchmark-regression-threshold=0.10 \
  --benchmark-min-delta-us=1.0
```

可使用 `-K` 按参数表达式缩小范围：

```bash
pytest tests/foo/test_foo_bar.py -m benchmark --run-benchmark \
  -K 'hidden=4096 or hidden=7168'
```

提交前确认报告中没有 missing baseline 和非预期 regression，并检查 `tests/foo/benchmark/test_foo_bar.jsonl` 只包含当前公开算子名和 `a5` 集群记录。

## 11. 构建与提交前检查

当前仓库暂不提供模型组合接口。多 kernel 串联、自定义 backward、`main_grad` 累加和训练/推理编排应由上层框架完成；`src/cann_ops_tilelang/` 只保留稳定的底层算子 API。

纯 PyTorch reference 与算子测试放在同一业务域，例如 `tests/engram/engram_gate_ref.py`，仅作为正确性和梯度 oracle，不得由运行时代码导入。

在当前开发容器中构建源码包和 wheel：

```bash
python -m build --no-isolation
```

提交前检查清单：

- [ ] 文件、公开函数、generator、prim_func 和 benchmark `kernel` 命名一致；
- [ ] 没有 CPU/CUDA 静默回退或导入时设备初始化；
- [ ] 公共接口完成输入、输出和边界校验；
- [ ] PyTorch reference 与分级正确性测试通过；
- [ ] `out=`、空 Tensor 和非法输入有明确行为；
- [ ] Ruff check 与 format 通过；
- [ ] 单进程和两阶段 xdist 测试通过；
- [ ] NPU 显存画像已生成并提交；
- [ ] A5 benchmark 基线已生成，二次回归无缺失；
- [ ] sdist 和 wheel 构建成功；
- [ ] README 或算子文档写明支持范围和已知限制。

性能相关提交还应记录 SoC、CANN、PyTorch NPU、TileLang 版本，以及 shape、dtype、layout、计时方法、优化前后时延和正确性误差。

更多 pytest 插件参数与维护命令见 [`tests/README.md`](../tests/README.md)。
