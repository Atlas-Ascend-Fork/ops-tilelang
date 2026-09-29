# ops-tilelang

`ops-tilelang` is a high-performance operator repository for the Ascend platform, built with the TileLang DSL. The project aims to provide the CANN community with operator implementations that have a clear structure, are easy to validate, support continuous performance tuning, and integrate cleanly with upper-layer frameworks.

> **Hardware support**: This repository supports only Ascend A5 and later architectures. Compilation and correct execution on earlier Ascend architectures are not guaranteed.

The repository provides a complete operator development and validation framework, including operator implementation and exports, runtime configuration, tiered test parameter generation, deterministic random seeds, correctness validation, NPU memory profiling, A5 performance regression testing, xdist-based multi-device binding, staged test scheduling, wheel packaging, and CI execution.

## Design Principles

- **Separate APIs from scheduling**: Python wrappers handle semantics, validation, and memory management, while `*_kernel.py` files implement hardware scheduling.
- **Organize by functional domain**: Implementations, exports, and future tuning configurations for the same operator family remain locally cohesive.
- **Correctness before performance**: Every operator must include a PyTorch reference implementation, boundary tests, and NPU correctness tests.
- **Track performance regressions**: Benchmarks are separated from regular tests, and baselines are stored by test file and chip cluster.
- **Avoid implicit global behavior**: Importing the package neither initializes the NPU nor compiles kernels. A kernel is compiled only when its operator is called for the first time.

## Repository Structure

```text
ops-tilelang/
├── src/cann_ops_tilelang/
│   ├── config.py                    # Ascend environment and runtime configuration
│   ├── utils.py                     # General-purpose pure Python utilities
│   ├── engram/                      # Engram operator implementations
│   └── xxx/                         # Other operator implementations
├── tests/
│   ├── testing/                     # Parameter generation, numerical assertions, and benchmark utilities
│   ├── pytest_*_plugin.py           # Benchmark, memory, seed, xdist, and warning plugins
│   ├── engram/                      # Engram tests and PyTorch references
│   └── xxx/                         # Tests and PyTorch references for other operators
├── docs/                            # Architecture and operator development guides
├── scripts/                         # Local and CI entry points
└── pyproject.toml                   # Build, dependency, and tooling configuration
```

## Requirements

- Python 3.10+
- An NPU based on the Ascend A5 or a later architecture
- A working CANN and Ascend driver environment
- PyTorch NPU compatible with the installed CANN version
- TileLang with Ascend target support

### Version Compatibility

CANN **9.3.0 or later** is required. Use a **CANN community weekly build**, available from the [CANN download page](https://www.hiascend.com/cann/download?versionId=799&ids=d806%2Ch0501%2Ch0601%2Ch0703&currentTab=1).

The repository declares only the general-purpose `torch` and `tilelang` Python dependencies. `torch_npu`, the CANN Toolkit, and the driver have strict compatibility requirements and must be installed by the deployment environment rather than pinned as PyPI dependencies.

## Installation

### Install for Local Development

```bash
pip install -e ".[dev]"
```

### Install a Released Version

```bash
pip install ops-tilelang
```

The distribution package is named `ops-tilelang`, while the Python import namespace is `cann_ops_tilelang`.

### Build and Install a Wheel

Build the wheel from a source tree with the development dependencies installed:

```bash
./scripts/build_wheel.sh
```

The build artifact is written to `dist/` and can be installed directly:

```bash
pip install dist/ops_tilelang-*.whl
```

The wheel contains only the runtime operator code. It does not include the test utilities or PyTorch references under `tests/`. `torch_npu`, the CANN Toolkit, the driver, and a TileLang version compatible with the target environment must still be provided by the target Ascend environment.

## Usage Examples

`engram_fused_weight` multiplies two two-dimensional BF16 weights in FP32:

```python
import torch

from cann_ops_tilelang import engram_fused_weight

weight_hidden = torch.randn((4, 4096), dtype=torch.bfloat16, device='npu')
weight_embed = torch.randn_like(weight_hidden)
weight_fused = engram_fused_weight(weight_hidden, weight_embed)
assert weight_fused.dtype == torch.float32
```

## Testing

```bash
# Run the engram_fused_weight correctness tests
pytest tests/engram/test_engram_fused_weight.py -m 'npu and not benchmark'

# Run the engram_fused_weight performance regression tests
pytest tests/engram/test_engram_fused_weight.py -m benchmark --run-benchmark

# Refresh the A5 performance baseline and peak NPU memory profile for engram_fused_weight
pytest tests/engram/test_engram_fused_weight.py -m benchmark --run-benchmark --refresh-baseline
pytest tests/engram/test_engram_fused_weight.py -m 'not benchmark' --refresh-mem

# Standard entry point
./scripts/run_test.sh
```

`OPS_TILELANG_TEST_LEVEL=0|1|2` controls the size of the test parameter set. `--seed` provides reproducible random data for each test case. See [tests/README.md](tests/README.md) for all plugin options and the baseline maintenance workflow.

Set `OPS_TILELANG_PRINT_KERNEL_SOURCE=1` to print the generated source when a kernel is built for the first time.

### CI Entry Point

`scripts/run_ci.sh` selects the execution stage through `OPS_TILELANG_CI_TASK`:

```bash
OPS_TILELANG_CI_TASK=lint ./scripts/run_ci.sh
OPS_TILELANG_CI_TASK=cpu ./scripts/run_ci.sh
OPS_TILELANG_CI_TASK=wheel ./scripts/run_ci.sh
OPS_TILELANG_CI_TASK=npu ./scripts/run_ci.sh
OPS_TILELANG_CI_TASK=benchmark ./scripts/run_ci.sh tests/engram/test_engram_fused_weight.py

# Run all stages on an Ascend A5 or later CI runner
OPS_TILELANG_CI_TASK=all ./scripts/run_ci.sh
```

NPU correctness tests use two workers per device by default, while tests with large or unknown memory requirements use one worker per device by default. Adjust these values with `TILELANG_TEST_WORKERS_PER_NPU` and `TILELANG_TEST_WORKERS_PER_NPU_LARGE`.

## Adding an Operator

See [docs/adding_an_operator.md](docs/adding_an_operator.md). A new operator must include at least:

1. A clear, typed public API;
2. An Ascend TileLang kernel;
3. A PyTorch reference implementation and correctness tests;
4. Boundary tests covering empty tensors and invalid dtypes or shapes;
5. Benchmarks for representative shapes;
6. Documentation of supported dtypes, shapes, layouts, and known limitations.

## License

MIT License. See [LICENSE](LICENSE) for details.
