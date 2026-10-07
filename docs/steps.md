# Implementation Steps

This document breaks the project into ordered implementation steps. Each step has:

- a clear goal,
- the files/modules to add or change,
- required behavior,
- exit criteria,
- the tests that prove it works.

Step 001 is complete. Later steps are ordered so the simulator and compiler grow together without a domain-specific fork.

---

## Step 001: ISA, memory, and first simulator slice

Status: **done**

Goal:

- Define the initial ISA.
- Model byte-addressable memory and per-thread registers.
- Execute a simple SIMT warp.
- Run vector add end to end.

Implement:

- `hw/isa.py`
- `hw/memory.py`
- `hw/core.py`
- `hw/device.py`
- `tests/unit/test_isa.py`
- `tests/operator_e2e/test_vector_add_e2e.py`

Exit criteria:

- ISA parses and formats deterministically.
- Vector-add ISA program matches expected output.
- Identical launches produce identical cycle counts.

Reference:

- [step-001.md](step-001.md)

---

## Step 002: Branches, divergence, and reconvergence

Status: **done**

Goal:

- Make the simulator handle SIMT control flow.

Implement:

- branch target resolution using labels,
- per-warp active mask updates,
- a simple reconvergence stack,
- `BRA`, `CMP`, `SEL`, and predicated execution.

Behavior:

- A divergent warp executes both paths serially.
- Inactive lanes do not update registers or memory.
- The warp reconverges with the original active mask.

Tests:

- if/else divergence,
- nested branches,
- predicated writes,
- divergent reduction.

Exit criteria:

- Divergent ISA programs match scalar reference execution.

---

## Step 003: Block barriers and multi-warp scheduling

Status: **done**

Goal:

- Add real block synchronization and allow more than one warp per SM.

Implement:

- block-scoped `BAR.SYNC`,
- per-warp barrier arrival and release,
- multiple `WarpState` objects per `SM`,
- a simple round-robin warp scheduler.

Behavior:

- Warps block at a barrier until all warps in the block arrive.
- Shared-memory writes are visible after the barrier.
- Multiple warps on one SM execute independently.

Tests:

- shared-memory reduction with two warps,
- barrier deadlock prevention,
- independent warp progress.

Exit criteria:

- Multi-warp reduction matches reference and terminates deterministically.

---

## Step 004: Timing model and performance counters

Status: **done**

Goal:

- Move from "one instruction per cycle" to a simple stall-aware timing model.

Implement:

- per-instruction latency,
- memory latency and bandwidth,
- stall reasons: fetch, ALU, memory, barrier, scoreboard,
- counters for cycles, issued instructions, stalls, and memory traffic,
- optional shared-memory bank-conflict detection.

Tests:

- cycle stability,
- tiled vs naive matmul IPC comparison later,
- memory-bound vs ALU-bound kernel differences.

Exit criteria:

- Profile mode reports stable counters and plausible stall behavior.

Reference:

- [step-004.md](step-004.md)

---

## Step 005: Unified Model IR and manual frontend

Status: **done**

Goal:

- Define a language-neutral model graph for CNN and Transformer ops.

Implement:

- `compiler/model_ir.py`
- typed `Value` and `Node` classes,
- CNN op nodes,
- Transformer/LLM op nodes,
- shape and dtype metadata,
- `compiler/frontends/manual_frontend.py`.

Op nodes must include:

- elementwise, reduction, matmul,
- conv2d, pooling, normalization,
- softmax, attention, RoPE, KV-cache,
- reshape/transpose/concat/slice/pad/resize.

Exit criteria:

- A manual graph can be built, printed, and validated.

Reference:

- [step-005.md](step-005.md)

---

## Step 006: MLIR builder and `mgpu` dialect

Status: **done**

Goal:

- Emit real MLIR text for the model graph.

Implement:

- `compiler/mlir/mlir_builder.py`
- `compiler/mlir/passes.py`
- `compiler/mlir/mgpu_dialect.td`

Use:

- `func`, `arith`, `math`, `memref`, `scf`, `linalg`/`tosa`,
- `mgpu.launch`,
- `mgpu.thread_id`, `mgpu.block_id`, `mgpu.grid_dim`, `mgpu.block_dim`,
- `mgpu.shared_alloc`, `mgpu.barrier`, `mgpu.async_copy`,
- `mgpu.mma`.

Exit criteria:

- Vector-add, matmul, and conv graphs emit deterministic MLIR text.

Reference:

- [step-006.md](step-006.md)

---

## Step 007: Shared operator E2E harness

Status: **done**

Goal:

- Create one reusable end-to-end test pipeline.

Implement:

- `tests/operator_e2e/helpers.py`,
- reference runners for NumPy comparison,
- a helper that runs:

```text
model graph -> MLIR -> ISA -> simulator -> output
```

Exit criteria:

- Every new operator test uses the same harness.

Reference:

- [step-007.md](step-007.md)

---

## Step 008: Common kernels

Status: **done**

Goal:

- Implement the shared compute backbone.

Kernels:

- elementwise add/sub/mul/div,
- reductions,
- tiled matmul.

Lowering work:

- thread/block mapping,
- shared-memory tiling for matmul,
- register tiling and `FMA`.

Tests:

- elementwise E2E,
- reduction E2E,
- matmul E2E.

Exit criteria:

- Common kernels match NumPy.
- Tiled matmul is correct.

Reference:

- [step-008.md](step-008.md)

---

## Step 009: Activation and shape/layout kernels

Goal:

- Add ops needed by both CNN and Transformer graphs.

Kernels:

- `ReLU`, `LeakyReLU`, `SiLU`, `GELU`, `Sigmoid`,
- `Reshape`, `Transpose`, `Concat`, `Slice`, `Pad`, `Resize/Upsample`.

Exit criteria:

- Each operator has a NumPy golden E2E test.

---

## Step 010: CNN kernels

Goal:

- Run ordinary CNN layers.

Kernels:

- `Conv2d`,
- `DepthwiseConv2d`,
- `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool`.

Initial strategy:

- conv via im2col + tiled matmul,
- add direct convolution later if needed.

Exit criteria:

- Conv and pooling E2E tests pass.

---

## Step 011: CNN normalization and detection helpers

Goal:

- Support full CNN/object-detection graphs.

Kernels:

- inference-mode `BatchNorm`,
- `GroupNorm`,
- `BoxDecode`,
- simple `NMS`.

Exit criteria:

- A small detector-head graph compiles and runs.

---

## Step 012: Transformer normalization and softmax

Goal:

- Add the stable numeric primitives used by attention.

Kernels:

- online `Softmax`,
- `LayerNorm`,
- `RMSNorm`.

Exit criteria:

- Each op matches NumPy within tolerance and is numerically stable.

---

## Step 013: Attention, RoPE, and KV-cache

Goal:

- Run small Transformer attention blocks.

Kernels:

- `RoPE`,
- causal mask generation,
- scaled dot-product attention,
- masked attention,
- `KVCacheAppend`,
- `KVCacheGather`.

Initial strategy:

- attention as tiled QK^T, online softmax, PV matmul.

Exit criteria:

- Attention and KV-cache E2E tests pass.

---

## Step 014: ONNX and PyTorch FX frontends

Goal:

- Import real exported models.

Implement:

- `compiler/frontends/onnx_frontend.py`,
- `compiler/frontends/torch_fx_frontend.py`.

Exit criteria:

- An exported CNN graph lowers through the same pipeline as the manual frontend.
- A small PyTorch Transformer block exports and lowers.

---

## Step 015: Runtime and driver

Goal:

- Provide a stable host/device API.

Implement:

- `runtime/buffer.py`,
- `runtime/driver.py`,
- `runtime/profiler.py`.

Features:

- allocation,
- host-to-device and device-to-host copy,
- kernel launch,
- synchronization,
- profiling.

Exit criteria:

- Multiple launches are deterministic and do not corrupt state.

---

## Step 016: CNN/object-detection model examples

Goal:

- Validate the complete CNN path.

Examples:

- ResNet block,
- FPN-style neck,
- YOLO/SSD-style detection head,
- small full detector on synthetic input.

Exit criteria:

- Full CNN/detector graphs compile and match reference.

---

## Step 017: Transformer/LLM model examples

Goal:

- Validate the complete Transformer path.

Examples:

- Transformer encoder block,
- GPT/decoder-style block with causal mask,
- KV-cache decode step,
- small autoregressive loop.

Exit criteria:

- Transformer examples compile and match reference.

---

## Step 018: Hardening and documentation

Goal:

- Make the project reliable and maintainable.

Tasks:

- consistent diagnostics,
- boundary checks,
- tolerance documentation,
- docstrings and comments,
- full test suite green,
- update architecture and type docs.

Exit criteria:

- `pre-commit`, unit tests, operator E2E tests, and model E2E tests all pass.

---

## Future optional steps

After the main path works:

- FP16/BF16 compute,
- INT8 quantization,
- INT4/UINT4 unpack-and-dequantize,
- FP8 E4M3/E5M2 kernels,
- tensor-core `MMA` execution,
- shared-memory bank-conflict-aware padding,
- kernel autotuning.
