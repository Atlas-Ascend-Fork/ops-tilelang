---
name: ops-tilelang-torch-compile
description: 为本仓库 Ascend TileLang 算子接入 torch.compile 的 npugraph_ex 后端，并验证正确性及 ACL Graph 捕获、重放行为。
---

# Ops TileLang torch.compile 接入

为 `src/cann_ops_tilelang/` 中的算子实现固定形状 NPU 图执行，以独立数值校验、正确的副作用和真实捕获/重放作为完成标准。

## 1. 确认环境与算子契约

- 在目标仓库和分支操作，读取适用的 `AGENTS.md`，保留已有修改。检查 Python、torch、torch_npu、TileLang、CANN 和设备版本，在执行测试的 shell 中加载实际安装的 Ascend 环境脚本。
- 阅读目标公共接口、wrapper、kernel、reference、测试及配置；基础封装参考 `src/cann_ops_tilelang/engram/engram_fused_weight.py` 和 `tests/engram/test_engram_fused_weight.py`。可选统计量、硬件核数等复杂契约再参考 `src/cann_ops_tilelang/engram/engram_gate.py`、`src/cann_ops_tilelang/config.py` 和对应测试。
- 明确输入限制、输出结构与元信息、可选返回、`out=`、所有写入参数、存储别名、空输入及配置依赖。保留公共签名、默认值、返回语义和原有数值标准；优先调整封装，保留 kernel 数学逻辑及调度。

## 2. 封装 custom_op 与 fake

将元信息校验拆为 `_validate_*`，将运行环境检查、分配、TileLang JIT 和 launch 保留在 `_impl`。公共函数直接使用 `@torch.library.custom_op` 注册，eager 和 compile 统一经该 op 调用 `_impl`。

确需补充内部硬件标量、处理 `out=` 分支、返回输入别名或在 schema 转换前严格校验参数时，保留薄 wrapper；wrapper 在 eager 和 compile 下均调用内部注册 op，不按执行模式分流到不同实现。

以下模板以单输入、单输出且真实实现用 `empty_like(x)` 分配输出的函数式算子为例；按实际接口实现 `_target_impl`、校验和 fake 分配：

```python
import torch


@torch.library.custom_op('cann_ops_tilelang::target', mutates_args=())
def target(x: torch.Tensor) -> torch.Tensor:
    return _target_impl(x)


@target.register_fake
def _target_fake(x):
    _validate_target_metadata(x)
    return torch.empty_like(x)
```

先定义被 `custom_op` 装饰的函数，再执行其 `register_fake` 装饰器；后者在模块导入时立即求值，提前注册会因算子名尚未定义而触发 `NameError`，导致整个包无法导入。

fake 只使用元信息，返回与真实执行一致的 shape、dtype、device、布局、stride、offset 和 Tensor/None 结构；按契约分配独立存储。fake 中不读取 Tensor 数值，不执行 `.item()`、设备查询、环境文件检查、JIT 或真实 kernel。按真实输出分配规则选用 `empty_like`、`empty` 等构造方式。

`custom_op` 隔离 Dynamo 追踪，但真实实现仍须满足 NPU 捕获约束。设备到主机拷贝、同步和值依赖 Python 控制流等问题应在真实执行路径定位并解决，不能仅靠封装消除。

### 特殊契约

| 场景 | 实现要求 |
| --- | --- |
| 可选 Tensor 返回 | 使用准确的 schema，例如 `(Tensor x, bool save_stats=True) -> (Tensor, Tensor?)`。PyTorch 2.10 对部分可选 Tensor 元组不能自动推导；核对参数、默认值及每个返回分支。 |
| 严格标量类型 | schema 可能将 `0/1` 转换为 bool。需要保留原始 Python 类型限制时，在公共 wrapper 调用 op 前校验；仅在 fake 或真实实现中校验转换后的值不足以保持原有契约。 |
| 原地更新 | `mutates_args` 列出所有被写入参数，包括 cache、out、scale 和辅助状态；仅不写输入时使用空元组。 |
| 更新后返回输入别名 | 参考 PyTorch 2.10 的高层 custom_op 不允许返回输入别名。内部 op 声明 mutation、返回 `None`，fake 也返回 `None`；公共 wrapper 在 op 调用后返回原缓冲区。验证 functionalization/copy-back 后的写入与存储别名。 |
| `out=None` 与 `out=tensor` | 分别注册分配输出、写入缓冲区的稳定契约，由公共接口分流并保留 `out=`。复杂别名按当前注册机制实现。 |
| 输出、计算或调度依赖可变配置 | 以内部有类型参数传递实际配置，按下节处理；公共接口不新增参数。 |
| 输出长度依赖数据 | 使用后端支持的表示；fake 不读设备数据或虚构固定 shape。容量化、填充或接口调整必须满足原有契约。 |

编译路径中的 `posix.stat` 等环境探测应移出追踪，真实执行仍保留 `require_ascend()`；优先在对应算子内适配，避免改变通用配置行为。

显式 `*_bwd` 入图不等于注册自动 autograd，也不代表 `.backward()` 捕获或高阶梯度支持；相关能力须单独实现和验证。随机数、多流、通信和特殊设备格式须按实际后端确认捕获支持。

### 显式传递配置

用有明确类型和含义的内部参数表达实际配置依赖，贯穿 **wrapper → Custom Op → _impl → JIT 工厂及 launch**。下游使用传入值，避免再次读取全局配置；fake 按依赖使用同一值。

- 仅处理目标算子实际依赖的配置，复用 helper 做解析和校验，保留设备选择、覆盖规则及空输入短路。优先传递实际参与执行的参数；确需额外的图失效标记时说明必要性。
- 每层缓存键覆盖其静态依赖；外层缓存失效不代表内层 JIT 失效。不用缓存装饰器或常量化声明冻结可变配置。
- 环境变量依赖同时涵盖存在性和值，通过运行时切换验证守卫生效。

## 3. 选择图执行上下文

```python
import torch_npu  # 注册 NPU 设备及后端

compiled = torch.compile(
    public_op, backend='npugraph_ex', fullgraph=True, dynamic=False,
)
outputs = compiled(*args, **kwargs)  # 在调用方需要的梯度或 inference 上下文中执行
```

- 保留调用方的梯度记录语义；需要 autograd 的任务不能默认切换到 `inference_mode()`。
- Engram gate 在已验证的 torch 2.10.0+cpu / torch_npu 2.10.0、CANN 9.3.0、Ascend 950DT_9574 环境中，先在 inference 模式捕获 fwd、再在模式外捕获 bwd 会于 `capture_begin` 报 inference Tensor 原地更新错误。该环境下，无需 autograd 记录的 fwd 和显式 bwd 测试使用相同的外层 `torch.inference_mode()` 作为兼容策略；其它算子、版本和训练场景应分别验证。
- autograd reference 的正常梯度叶子、前向和 `.backward()` 在 inference 上下文外执行；不要对需要 reference 求导的整个 bwd 测试加 inference 装饰器。
- 一次复用验证保持输入的 shape、dtype、device、stride、storage offset、`requires_grad`、`is_inference()` 及特化标量一致。inference 上下文中新建 Tensor 可能改变属性；替换输入时保持创建方式一致。
- 捕获异常后启动新 Python/pytest 进程复测。`Offset increment outside graph capture encountered unexpectedly` 可能是此前捕获失败的次生错误，先定位首次失败。

以下最小复现从仓库根目录在新进程运行，仅在内存中令 bwd 的 compiled 调用退出 inference 模式，不修改测试文件：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python - <<'PY'
import torch
from tests.engram import test_engram_gate_fwd as fwd
from tests.engram import test_engram_gate_bwd as bwd

fwd.test_engram_gate_fwd_compile(False)
compile_gate = bwd.compile_gate_for_test

def compile_without_inference(op):
    compiled = compile_gate(op)

    def call(*args):
        with torch.inference_mode(False):
            return compiled(*args)
    return call

bwd.compile_gate_for_test = compile_without_inference
bwd.test_engram_gate_bwd_compile(128)
PY
```

## 4. 编写正确性与契约测试

在目标测试文件新增 compile 用例，选择代表性非空形状、支持的空输入、dtype、可选输出和关键分支。每个模式连续验证：

1. 首次 compiled 调用。
2. 在原存储上更新有效数据，刷新相应权重、统计量、索引及状态。
3. 换为同元信息的新存储，并明确改变有效数据。

每轮使用当前数据重新计算独立 reference；索引、scale 和其它受约束输入保持合法。检查返回结构、Tensor/None、shape、dtype、device、必要的布局、stride 和 offset，再比较数值。正确处理零元素和零归约，避免 NaN 导致无效比较。

- **显式反向：** 每轮新建 reference 梯度叶子，避免梯度累积；使用前向统计量时，将当前输入对应的 detached 统计量写入测试缓冲区。检查梯度输出及分片元信息，按 API 语义归并分片并比较最终梯度。
- **原地状态：** reference 与被测状态相互独立，同时保留各自应有的 view、offset 和别名；比较返回值、实际写入、未修改区域和只读输入，按相同序列演进状态。
- **配置变化：** 保持 Tensor 元信息不变，在同一 compiled 对象上验证默认、覆盖、恢复和非法值；环境变量包含新增、改值、删除。序列内不重建图或清空编译/JIT 缓存来强制生效，仅按配置 API 契约清理查询缓存。
- **配置取证：** 校验数值、副作用及图事件，并从实际工厂参数、内核对象或生成源码确认配置生效和恢复后的复用；不能仅检查 Dynamo 重编译计数。
- **输出生命周期：** 在下次调用前完成校验，释放 tuple/list、解包变量和循环变量持有的临时图输出；需保存时在图外复制快照。保留调用方持久状态，避免在 inference 上下文外原地修改 inference 输出。
- **组合模型：** 验证目标算子与合理的上下游 Tensor 运算；有配套前后向时验证统计量在同图传递到显式 backward。

对实际注册的 `CustomOpDef` 或 `OpOverload` 检查 schema 和 fake，补齐内部标量参数：

```python
torch.library.opcheck(
    registered_op, args, kwargs=kwargs,
    test_utils=('test_schema', 'test_faketensor'),
)
```

声明动态 shape、autograd 等额外能力时补对应检查。opcheck 不替代数值和副作用验证。

## 5. 证明捕获、重放与复用

按算子和关键分支分别取证，使用确有设备工作的非空用例；空输入或纯别名可能不产生设备图。检查当前后端实现和配置，排除禁用编译、force-eager、捕获上限等回退。

TorchNPU 2.10 可通过 CPU profiler 的 `acl_graph_capture`、`acl_graph_replay`、`fx_run_eagerly` 事件诊断。可逐次调用采集：

```python
def inspect_call(compiled, *args, **kwargs):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
    ) as profile:
        result = compiled(*args, **kwargs)
        torch.npu.synchronize()
    events = {event.key: event.count for event in profile.key_averages()}
    names = ('acl_graph_capture', 'acl_graph_replay', 'fx_run_eagerly')
    return result, {name: events.get(name, 0) for name in names}
```

由调用方在所需执行上下文中调用 `inspect_call`；逐轮在图外准备输入和 reference，调用后校验并释放临时输出。确认真实捕获、固定元信息的后续调用进入重放路径，且无 eager 回退；组合或批量汇总事件不能代替单算子证据。事件名按安装的 torch_npu 实现核实，缺失时使用对应后端日志或计数器，缺证据不能判定通过。

`fullgraph=True` 只排除 Dynamo 图中断。首次调用内部可能同时捕获和重放，不预设固定 warmup/capture 顺序。区分 Dynamo 重新编译和 NPU 重捕获；函数式算子验证复用，原地状态换地址可能合法重捕获。不要删除 mutation 声明来减少捕获次数，也不要把内部计数器固定数量作为所有数值测试的断言。

## 6. 回归与验收

先在新进程执行包导入检查，确认 op 定义和 fake 注册可正常完成，再运行新增 compile 用例及受影响的非 benchmark 测试；配套前后向须在同一新进程按 fwd→bwd 顺序执行。共享 helper 变更覆盖直接使用方。命令以 Engram gate 为例，替换为目标算子的测试入口：

```bash
PYTHONPATH=src python -c "import cann_ops_tilelang"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPS_TILELANG_TEST_LEVEL=0 python -m pytest \
  tests/engram/test_engram_gate_fwd.py::test_engram_gate_fwd_compile \
  tests/engram/test_engram_gate_bwd.py::test_engram_gate_bwd_compile -x -v --tb=short

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPS_TILELANG_TEST_LEVEL=0 python -m pytest \
  tests/engram/test_engram_gate_fwd.py tests/engram/test_engram_gate_bwd.py \
  -m 'not benchmark' -x -v --tb=short
```

从仓库根目录执行，保存实际命令、版本、退出码及日志；使用 `tee` 时以 `PIPESTATUS[0]` 判断 pytest 结果。运行项目要求的 lint/格式检查和 `git diff --check`。常规改造不刷新 benchmark 基线、显存画像，也不删除磁盘 JIT 缓存；测试间进程内编译缓存的隔离按第 4 节处理，新进程不代表磁盘 JIT 缓存为空。

| 失败现象 | 排查方向 |
| --- | --- |
| `capture_begin` 报 inference Tensor 原地更新错误 | 检查版本和同进程前序捕获；在上述 2.10 复现环境按第 3 节处理，保留调用方的梯度语义。 |
| Dynamo 遇到文件探测或设备查询 | 分离环境检查，保留真实执行校验，确认元信息查询可追踪。 |
| 导入时报算子名未定义 | 将 `register_fake` 放在对应 custom op 定义之后；语法检查通过不代表装饰器能正确执行。 |
| schema、fake 或别名错误 | 核对可选返回、分支元信息、mutation、存储别名及 copy-back。 |
| 数值正确但缺少重放证据 | 检查有效设备工作、事件语义和后端回退；按算子逐次取证。 |
| 反复编译或捕获 | 检查 Tensor 属性、标量、可变配置、状态地址和后端捕获限制。 |

验收须同时满足：公共契约保留、独立数值与副作用校验通过、注册契约正确、真实 NPU 捕获及后续重放成立、相关回归通过。不得通过换后端、关闭 fullgraph、放宽精度、调整测试顺序或无条件 skip 掩盖失败。

批量改造逐算子记录“已实机打通 / 已改造待实机 / 具体阻塞”，列明支持边界、测试入口和图证据。静态检查、CPU `aot_eager`、fake/schema 或单独数值通过均不能代替实机入图验收。
