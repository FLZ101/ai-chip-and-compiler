# ResNet18 Full Execution Plan

This document defines the remaining work to run a real pretrained ResNet18
in the simulator and compare its output with PyTorch.

## 1. Goal

Run this exact path:

```text
image (1 x 3 x 224 x 224)
  -> ResNet18 stem
  -> 4 residual stages
  -> global average pool
  -> flatten
  -> final Linear
  -> 1000-class logits
```

The first milestone uses a static `1x3x224x224` input and FP32.  The
acceptance test compares the complete 1000-way output with PyTorch.

## 2. Current state

Already available:

- ResNet18 pretrained checkpoint download/caching.
- torch-mlir import and canonicalization of full ResNet18 to
  `linalg-on-tensors`.
- simulator execution of a pretrained ResNet18 stem
  (`conv7x7 -> BatchNorm -> ReLU`).
- general `Linear` lowering.
- scalar conv loop lowering with bounded ISA.
- `BatchNorm2d` frontend lowering.
- wave block scheduler, typed buffers, memory planner, and launch queue.

Missing for full ResNet18:

- full imported-module lowering from linalg-on-tensors to `mgpu.isa`,
- general 4D conv/pool/reduction kernels with grid/wave scheduling,
- adaptive/global average pooling and flatten,
- model-wide memory plan for all activations and weights,
- output comparison at `1x3x224x224`.

## 3. Target sizes

ResNet18:

- input: `1x3x224x224`
- first conv: `3 -> 64`, `7x7`, stride 2
- stages: `64`, `128`, `256`, `512` channels
- output classes: `1000`
- parameters: ~11.7M

FP32:

- weights: ~47 MB
- activations: roughly 8-16 MB with sequential allocation
- final logits: `1x1000`

## 4. Phases

### Phase A: Full-module pipeline

Implement:

- import full ResNet18 and canonicalize,
- bufferize function boundaries,
- map bufferized `linalg.conv_2d_nchw_fchw`, `linalg.generic`, and
  pooling ops to `mgpu.launch`,
- extend `run_model_pipeline` to handle multiple ISA-producing launches,
- execute launches in dependency order through the runtime queue.

Files:

- `compiler/mlir/mgpu_runner.py`
- `mlir/mgpu/lib/mgpu/MapToMGPUPass.cpp`
- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`

Exit criteria:

- full canonicalized ResNet18 lowers without `torch.aten`,
- pipeline emits one or more `mgpu.isa` ops,
- unsupported ops fail with op/shape/dtype diagnostics.

### Phase B: Scalable conv/pool kernels

Generalize the current scalar conv loop to:

- use the wave scheduler for large grids,
- support large channel/spatial dimensions without unrolled reduction code,
- handle `pad/stride/dilation` generically,
- emit loop-based max/average pooling,
- add global average pooling over the last two dimensions.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `tests/operator_e2e/test_cnn_kernels_e2e.py`

Exit criteria:

- large conv and pooling tensors produce bounded ISA,
- numerical results match NumPy/PyTorch references,
- `224x224` conv kernels terminate in test time.

### Phase C: Full model memory plan

Use the lifetime-aware memory planner for:

- weights,
- stage activations,
- pooled feature map,
- final logits,
- scratch buffers.

Files:

- `runtime/memory_plan.py`
- `compiler/mlir/mgpu_runner.py`
- `runtime/driver.py`

Exit criteria:

- peak memory fits the configured global memory,
- plan is deterministic,
- allocator reports total and peak bytes.

### Phase D: End-to-end ResNet18 forward

Implement:

- static `1x3x224x224` export,
- full op sequence through the pipeline,
- final `Linear` and output extraction,
- numerical comparison against PyTorch.

Files:

- `tests/model_e2e/test_real_cnn_execution.py`
- `tests/model_e2e/test_real_pretrained_cnn.py`

Exit criteria:

- full ResNet18 logits match PyTorch within documented tolerance,
- deterministic repeated runs,
- runtime/cycle profiling is captured.

### Phase E: Numerical and performance gates

Implement:

- cropped smoke test,
- opt-in full model test,
- cycle/profile assertions,
- tolerance documentation.

Files:

- `tests/model_e2e/`
- `docs/tolerances.md`

Exit criteria:

- full static `224x224` forward matches PyTorch,
- pre-commit, unittest, and lit are green.

## 5. Risks

- **Large conv grids.**  The first ResNet18 conv produces tens of thousands
  of blocks.  Wave scheduling is required and correctness must precede
  speed.
- **Pooling/adaptive pooling.**  Global average pool and 4D pooling need
  explicit lowering.
- **Python simulator speed.**  A full 224x224 run will be slow.  Keep a
  cropped test in CI and gate the full test behind
  `MGPU_DOWNLOAD_MODELS=1`.
- **Function-boundary bufferization.**  Full modules require correct
  one-shot-bufferize options before map-to-mgpu.

## 6. Deliverable

`tests/model_e2e/test_real_cnn_execution.py` is extended to run a full
pretrained ResNet18 forward in the simulator and compare the 1000-class
output with PyTorch.
