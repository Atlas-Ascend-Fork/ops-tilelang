# 测试与性能回归框架

测试框架沿用 `TileKernels-Nightly` 的完整链路，并收敛为 Ascend/NPU 语义。测试目录根部的 `tests/conftest.py` 负责在 TileLang 导入前绑定 xdist worker，随后装载以下独立 pytest 插件：

- `pytest_random_plugin.py`：`--seed` 与 node ID 组合，得到逐用例稳定随机种子；
- `pytest_npu_mem_plugin.py`：记录峰值 NPU 显存，自动标记 `large_npu_mem`；
- `pytest_benchmark_plugin.py`：计时、JSONL 输出、性能基线比较与刷新；
- `pytest_improvement_plugin.py`：通过 `--legacy-table` 汇总几何平均性能；
- `pytest_xdist_failfast_plugin.py`：并行 `-x` 时停止全部 worker；
- `pytest_tilelang_warning_plugin.py`：在终端汇总 TileLang 编译警告。

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

# 用参数表达式筛选；支持 and/or/not、括号和正则
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
