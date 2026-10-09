# Hardware Scaling Plan

This document describes how to scale the minimal GPU simulator from
operator-sized tests to real CNN and small-LLM workloads.  It is the
implementation plan for the "Hardware scaling" section in
[next-stage-steps.md](next-stage-steps.md).

## 1. Goal

Run the already-imported real models end to end:

- ResNet18 or MobileNetV3-Small at `1x3x224x224`.
- GPT-2-small or Pythia-70M in a small autoregressive decode loop.

Today those models import and canonicalize correctly, but the functional
simulator cannot schedule or hold their workloads.  This plan closes that gap.

## 2. Current constraints

The relevant implementation is in [hw/core.py](../hw/core.py),
[runtime/driver.py](../runtime/driver.py), and
[runtime/buffer.py](../runtime/buffer.py).

### Device resources

- Default global memory is `1 MiB`.
- Default shared memory is `64 KiB` per SM.
- Default device has `4` SMs.
- Every lane has a fixed 64-entry register file.
- There is no configurable register-file size or occupancy limit.

### Block scheduling

- `launch` requires `total_blocks <= sm_count`.
- Each SM runs at most one block.
- There is no wave scheduler, resident-block replacement, or block-context
  save/restore.
- `BAR.SYNC` is block-scoped, which is correct, but it is only tested for one
  resident block per SM.

### Memory model

- `runtime.Buffer` supports only FP32 and assumes four bytes per element.
- `Driver.allocate` is append-only; there is no free, reuse, or lifetime.
- Weight, activation, and KV-cache memory planning is manual.
- There is no typed host/device copy API for INT8/INT4/FP8.

### Codegen scalability

- Several `mgpu-to-isa` emitters unroll reductions or kernel loops directly.
- Elementwise kernels use `grid = ceil(n / 32)` but the simulator cannot run
  those grids yet.
- Conv/attention emitters often assume small tiles or one output block.
- The full linalg-on-tensors pipeline is not wired into the Python driver.

## 3. Scaling targets

| Workload | Approx. weights | Peak activation / cache | Required device shape |
| --- | --- | --- | --- |
| ResNet18 | ~47 MB FP32 | ~8-16 MB FP32 activations | hundreds of SMs or wave scheduling |
| GPT-2-small | ~500 MB FP32 | weights + KV cache | many resident blocks and typed buffers |
| Pythia-70M | ~300 MB FP32 | weights + KV cache | same as above, smaller than GPT-2 |

The simulator does not need to be fast.  It needs to be deterministic,
correct, and able to finish within reasonable test time.

## 4. Workstreams

### 4.1 Configurable device and occupancy

Implement:

- configurable `global_size`, `shared_size`, `sm_count`,
- configurable per-lane register count,
- max resident warps/blocks per SM,
- resource-occupancy validation before launch.

Files:

- `hw/core.py`
- `hw/device.py`

Exit criteria:

- a test creates devices with non-default resource sizes,
- an invalid launch fails with a resource-occupancy diagnostic.

### 4.2 Block scheduler

Replace one-block-per-SM with a small wave scheduler.

Implement:

- `BlockContext` holding `block_id`, warp states, and special-register values,
- a FIFO of ready blocks,
- scheduling as many blocks per SM as occupancy allows,
- running waves until all blocks have exited,
- preserving deterministic round-robin issue order within each SM,
- reusing SMs across waves without corrupting global/shared memory.

Files:

- `hw/core.py`
- `hw/device.py`

Exit criteria:

- a grid with `total_blocks > sm_count` runs correctly,
- repeated launches produce identical profiles,
- `ctaid.x/y/z` remains correct across waves.

### 4.3 Typed memory allocator and runtime buffers

Implement:

- element-size-aware `Buffer` (`f32/f16/bf16/f8/i8/i4/u4/i32`),
- typed `write`/`read` using existing `ByteMemory` methods,
- free/reuse lists and alignment,
- allocator reports peak bytes and fragmentation.

Files:

- `runtime/buffer.py`
- `runtime/driver.py`
- `hw/memory.py` if a small allocator helper belongs there.

Exit criteria:

- typed round-trip through the runtime works,
- freeing and reallocating does not corrupt live buffers,
- an allocation that would exceed global memory fails clearly.

### 4.4 Model memory planning

Implement a planner that assigns:

- model weights,
- intermediate activations,
- KV cache,
- scratch buffers,

with known lifetimes and a compaction pass when possible.

Files:

- `compiler/mlir/memory_plan.py` or `runtime/memory_plan.py`.

Exit criteria:

- ResNet18 and GPT-2 block memory maps are generated deterministically,
- the planner reports total bytes and peak live bytes.

Status: **implemented (generic first-fit/lifetime planner)**

### 4.5 MLIR pipeline integration

Wire the real linalg-on-tensors path into the Python driver.

Implement:

- function-boundary bufferization,
- `linalg` -> tiled loops -> `mgpu` -> ISA pipeline,
- lowering `scf.for` and `memref.subview` produced by bufferization,
- a single `run_model_pipeline` entry point,
- clear diagnostics for unsupported ops/dtypes/shapes.

Files:

- `compiler/mlir/pipeline.py`
- `compiler/mlir/mgpu_runner.py`
- `mlir/mgpu/lib/mgpu/*Pass.cpp`

Exit criteria:

- imported ResNet18 and GPT-2 block lower to `mgpu.isa`,
- pipeline failure messages include op, pass, shape, and dtype.

Status: **implemented for single-linalg-op model fragments**

### 4.6 ISA/codegen scalability

Remove hard-coded small-workload assumptions.

Implement:

- loop-based conv/reduction kernels instead of huge unrolled bodies,
- grid-aware elementwise/conv/matmul emitters,
- real virtual-register liveness and spill slots,
- optional basic instruction scheduling,
- stable launch dimensions computed from shapes.

Files:

- `mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp`
- `compiler/mlir/mgpu_runner.py`

Exit criteria:

- `n > 32` elementwise tests run in waves,
- conv/reduction emit bounded-size ISA for large tensors,
- generated ISA still matches NumPy references.

Status: **implemented (grid-aware elementwise/matmul, wave execution, and
loop-based scalar conv/matmul)**

### 4.7 Runtime launch queue and synchronization

Implement:

- a command queue of kernels,
- explicit device synchronization,
- multi-launch profiling.

Files:

- `runtime/driver.py`
- `runtime/profiler.py`

Exit criteria:

- queued launches execute deterministically,
- synchronization does not corrupt state,
- profiler reports per-launch and aggregate cycles.

Status: **implemented**

## 5. Milestones

### M1: Configurable resources

- `global_size`, `shared_size`, `sm_count`, register count configurable.
- Tests cover non-default resource sizes and occupancy errors.

Status: **implemented**

### M2: Wave block scheduler

- Grids larger than SM count run correctly.
- Elementwise and conv tests with `grid > sm_count` pass.

Status: **implemented**

### M3: Typed runtime memory

- Typed buffers and allocator pass round-trip tests.
- Free/reuse and alignment pass.

Status: **implemented**

### M4: Real CNN execution

- ResNet18 or MobileNetV3-Small runs end to end.
- Output matches PyTorch/ONNX Runtime.

Status: **cropped pretrained stem implemented; full 224x224 model pending
codegen scalability**

### M5: Real LLM decode

- GPT-2-small or Pythia-70M runs a short greedy decode loop.
- Generated token IDs match PyTorch.

Status: **component execution and masked softmax implemented; full greedy
decode pending multi-head attention**

## 6. Risks and decisions

- **Python performance.**  Large unrolled conv kernels will be slow.
  Prefer loop-based ISA and keep tests on cropped/static shapes first.
- **Function-boundary bufferization.**  This is the main MLIR blocker.
  Validate the exact pass options before wiring the full pipeline.
- **Occupancy model.**  Keep it simple: resident blocks are bounded by
  register count, shared memory, and warp slots, not a detailed scoreboard.
- **Determinism.**  The scheduler must keep round-robin selection and stable
  wave ordering; profile snapshots are compared in tests.

## 7. Verification

For every milestone:

```bash
source .venv/bin/activate
pre-commit run --all-files
python -m unittest discover -v
lit -q tests/mlir
```

Add model-level tests only after the corresponding device-level scaling test
passes.
