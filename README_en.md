# ops-tilelang

`ops-tilelang` is a high-performance operator repository for the Ascend platform, built with the TileLang DSL. The project aims to provide the CANN community with operator implementations that have a clear structure, are easy to validate, support continuous performance tuning, and integrate cleanly with upper-layer frameworks.

> **Hardware support**: This repository supports only Ascend A5 and later architectures. Compilation and correct execution on earlier Ascend architectures are not guaranteed.

The repository currently provides a foundational project framework. The Engram operator family covers forward propagation, backward propagation, sharded gradient reduction, and Sinkhorn row normalization. The MoE operator family provides `topk_gate`, `moe_topk_gate`, and `normalize_weight`. Operator tests support tiered parameter generation, deterministic random seeds, NPU memory profiling, A5 performance regression testing, regular-expression filtering, xdist-based multi-device binding, and staged test scheduling.

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
│   ├── engram/                      # Engram forward, backward, and gradient reduction operators
│   └── moe/                         # MoE routing and weight normalization operators
├── tests/
│   ├── testing/                     # Parameter generation, numerical assertions, and benchmark utilities
│   ├── pytest_*_plugin.py           # Benchmark, memory, seed, xdist, and warning plugins
│   ├── engram/                      # Engram tests and PyTorch references
│   └── moe/                         # MoE tests and PyTorch references
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

The Engram gate forward operator can optionally preserve the statistics required by the backward pass. The backward operator produces input gradients and sharded weight gradients, which `engram_grad_w_reduce` then reduces in place into two FP32 weight gradients:

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

`engram_sinkhorn_step` performs one Sinkhorn row-normalization step on a two-dimensional FP32 matrix and updates the row scaling factors in place:

```python
import torch

from cann_ops_tilelang import engram_sinkhorn_step

matrix = torch.randn((1001, 256), dtype=torch.float32, device='npu')
row_scale = torch.rand((1001,), dtype=torch.float32, device='npu')
col_scale = torch.rand((256,), dtype=torch.float32, device='npu')
row_norm, col_sumsq = engram_sinkhorn_step(matrix, row_scale, col_scale, eps=1e-20)
```

All three inputs must reside on the same NPU, be contiguous FP32 tensors, and must not share storage. `matrix` must have shape `(num_rows, head_dim)`, where `head_dim` is a positive multiple of 256. The caller must ensure that `row_scale` contains non-negative Sinkhorn scaling factors. Typical validated dimensions are 256 and 5120. Because `SimtVF` reserves thread context, the wrapper rejects shapes whose estimated shared UB usage exceeds 216 KiB. When `num_rows=0`, the operator returns an empty `row_norm` and an all-zero `col_sumsq`.

The MoE routing operators accept contiguous FP32 scores on an NPU. `topk_gate` performs a stable expert selection for each token and selects the smaller index when scores are equal. `moe_topk_gate` fuses scoring, bias, masking, fixed routing, random routing, and logical-to-physical expert mapping. `normalize_weight` returns the denominator and normalized weights for each row:

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

## Testing

```bash
# Import and API tests that can run on CPU
pytest -m 'not npu and not benchmark'

# Ascend correctness tests
pytest -m 'npu and not benchmark'

# Run performance regression tests explicitly; use -K to filter parameter expressions
pytest -m benchmark --run-benchmark -K 'hidden=4096'

# Refresh A5 performance baselines and peak NPU memory profiles
pytest tests/engram/test_engram_fused_weight.py -m benchmark --run-benchmark --refresh-baseline
pytest tests/engram/test_engram_fused_weight.py -m 'not benchmark' --refresh-mem

# Refresh the complete Engram operator family
pytest tests/engram -m benchmark --run-benchmark --refresh-baseline
pytest tests/engram -m 'not benchmark' --refresh-mem

# Run or refresh the MoE operator family
pytest tests/moe -m 'npu and not benchmark'
pytest tests/moe -m benchmark --run-benchmark --refresh-baseline
pytest tests/moe -m 'not benchmark' --refresh-mem

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
OPS_TILELANG_CI_TASK=benchmark ./scripts/run_ci.sh tests/engram

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
