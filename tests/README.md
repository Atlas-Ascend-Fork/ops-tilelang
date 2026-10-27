# 测试与性能回归框架

测试框架沿用 `TileKernels-Nightly` 的完整链路，并收敛为 Ascend/NPU 语义。测试目录根部的 `tests/conftest.py` 负责在 TileLang 导入前绑定 xdist worker，随后装载以下独立 pytest 插件：

- `pytest_random_plugin.py`：`--seed` 与 node ID 组合，得到逐用例稳定随机种子；
- `pytest_npu_mem_plugin.py`：记录峰值 NPU 显存，自动标记 `large_npu_mem`；
- `pytest_benchmark_plugin.py`：计时、JSONL 输出、性能基线比较与刷新；
- `pytest_improvement_plugin.py`：通过 `--legacy-table` 汇总几何平均性能；
- `pytest_xdist_failfast_plugin.py`：并行 `-x` 时停止全部 worker；
- `pytest_tilelang_warning_plugin.py`：在终端汇总 TileLang 编译警告。

## CI 增量测试

CI 根据 `pr_filelist.txt` 中的变更，通过 `.gitcode/scripts/select_tests.py` 静态分析源码和测试的导入关系，自动选择直接或间接受影响的测试。新增算子及测试无需登记映射；测试应通过常规 Python 导入引用被测实现和参考实现。

共享配置、包入口、删除文件或无法确定依赖关系的代码变更保守执行全仓 L0。全局触发策略保留在 `.gitcode/config/full_test_triggers.conf`，其中不维护算子与测试的对应关系。选择器的 CPU 回归测试可用 `python -m unittest discover -s .gitcode/scripts/tests` 运行。

`.gitcode/` 下的 CI 文件默认不触发算子测试，便于管理员在机器故障等场景调整执行逻辑。仅修改 CI 文件或文档时跳过算子用例；同时修改源码或测试时，仍正常选择受影响的测试。显式配置的全量触发规则优先，管理员可按需添加 CI 路径规则；选择器的 CPU 回归测试仍会执行。

## 测试分级

`OPS_TILELANG_TEST_LEVEL` 取值为 `0`、`1`、`2`，分别表示核心、默认和完整参数集合。参数生成器位于 `tests/testing/generator.py`。

```bash
OPS_TILELANG_TEST_LEVEL=0 pytest tests/engram -m 'not benchmark'
OPS_TILELANG_TEST_LEVEL=2 pytest tests/engram -m 'not benchmark'
```

## 性能基线

benchmark 默认跳过，必须显式传入 `--run-benchmark`。Ascend 使用 `msprof` 后端，并将结果归入 `a5` 集群。基线存储在 `tests/<operator>/benchmark/<测试文件>.jsonl`。

```bash
# 全部 benchmark
pytest -m benchmark --run-benchmark

# 用参数表达式筛选；支持 and/or/not 和正则
pytest -m benchmark --run-benchmark -K 'hidden=4096 or hidden=7168'

# 指定输出、阈值和最小绝对变化
pytest -m benchmark --run-benchmark \
  --benchmark-output=/tmp/ops-tilelang.jsonl \
  --benchmark-regression-threshold=0.10 \
  --benchmark-min-delta-us=1.0

# 确认性能变化是预期后刷新分片基线
pytest tests/engram/test_engram_fused_weight.py \
  -m benchmark --run-benchmark --refresh-baseline

# Engram gate 前向/反向及梯度归并
pytest tests/engram -m benchmark --run-benchmark --refresh-baseline
```

`--refresh-baseline` 在 `pytest_sessionfinish` 收到 `exitstatus=0` 时更新基线；收到测试失败、fixture 错误或中断状态时不更新，原始结果输出仍可保留。后续终端报告或其他插件的错误不属于此守卫范围。

`-K` 的布尔运算优先级为 `not`、`and`、`or`（相邻正则也按 `and` 处理）。
解析器不支持用括号对布尔表达式分组；括号属于单个正则模式的语法。
例如，用 `-K 'hidden=(4096|7168) and not round=True'` 筛选两种 hidden 值并排除 round=True，
不要写成 `-K '(hidden=4096 or hidden=7168) and not round=True'`。

`--benchmark-verbose` 展示扩展列，`--legacy-table` 展示算子级几何平均汇总。

## NPU 显存画像

显存画像存储在 `tests/<operator>/npu_mem_profiles/<测试文件>.json`。无画像的测试会先归入 `large_npu_mem`，避免以高并发运行未知用例。

```bash
# 运行并刷新选中用例的显存画像
pytest tests/engram/test_engram_fused_weight.py -m 'not benchmark' --refresh-mem

# 刷新整个 Engram 算子族的画像
pytest tests/engram -m 'not benchmark' --refresh-mem

# 只运行尚无画像的测试
pytest --refresh-mem --refresh-mem-missing-only

# 调整大显存阈值或保留原始 JSONL
pytest --refresh-mem --npu-mem-threshold=15000 \
  --npu-mem-output=/tmp/ops-tilelang-mem.jsonl
```

`--npu-mem-flush-interval` 控制长任务周期写回间隔。

## 标准执行入口

`scripts/run_test.sh` 默认分两阶段运行：普通测试每张 NPU 两个 worker，大/未知显存测试每张 NPU 一个 worker。可通过 `TILELANG_TEST_WORKERS_PER_NPU` 和 `TILELANG_TEST_WORKERS_PER_NPU_LARGE` 调整。

```bash
./scripts/run_test.sh
OPS_TILELANG_RUN_BENCHMARK=1 ./scripts/run_test.sh tests/engram
```
