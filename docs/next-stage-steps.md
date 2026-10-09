# Detailed Steps for the Production MLIR Stage

This document turns the next-stage plan into concrete, ordered engineering
steps.

The steps are grouped by phase.  Steps inside a phase are ordered; later phases
depend on earlier phases.  Each step lists the files/modules involved, the
required implementation, tests, and a completion condition.

---

## Phase 0: Real MLIR Toolchain

### 0.1 Choose the MLIR integration model

Decide between:

- a C++ MLIR/LLVM build with `mgpu-opt` and `mgpu-translate`, or
- Python bindings provided by `torch-mlir`/`mlir-python`.

Recommendation:

- start with Python bindings for fast iteration,
- add a C++ tool later for production passes.

Files:

- `pyproject.toml`
- `compiler/mlir/toolchain.md` (new)

Tests:

- document the choice and runtime requirements.

Done when:

- the toolchain can import `torch_mlir` and parse standard MLIR.

### 0.2 Build a real `mgpu` dialect

Convert `compiler/mlir/mgpu_dialect.td` into a compiled dialect.

Implement:

- dialect registration,
- op declarations/definitions,
- verifiers,
- parsers/printers,
- canonicalization hooks.

Files:

- `compiler/mlir/mgpu_dialect.td`
- C++ or Python dialect sources

Tests:

- parse `mgpu.launch`
- parse `mgpu.thread_id`
- parse `mgpu.barrier`
- reject malformed ops

Done when:

- `mgpu.*` ops round-trip through MLIR parse/print.

### 0.3 Add a real MLIR test harness

Replace text-only Python checks with actual MLIR parse/verify tests.

Implement:

- `mgpu-opt` or Python equivalent,
- `lit` config for MLIR passes,
- FileCheck tests for pass output.

Files:

- `tests/mlir/`
- `tools/mgpu_opt.py` or CMake target

Done when:

- `lit` parses and verifies every existing `.mlir` test.

---

## Phase 1: Type and Memory Legalization

### 1.1 Define the complete MLIR type map

Implement mappings for:

- `f32`
- `f16`
- `bf16`
- `f8E4M3FN`
- `f8E5M2`
- `i8`
- `i4`
- `ui4`
- `i1`

Files:

- `compiler/model_ir.py`
- `compiler/mlir/mlir_builder.py`

Tests:

- parse/print all types
- invalid dtype rejection

Done when:

- every supported dtype has a stable MLIR type.

### 1.2 Add typed simulator memory accesses

Extend the ISA/memory model so loads/stores carry a type.

Suggested ISA forms:

```asm
LDG.F16 R0, [R1]
LDG.BF16 R0, [R1]
LDG.I8 R0, [R1]
LDG.F8E4M3 R0, [R1]
```

Files:

- `hw/isa.py`
- `hw/memory.py`
- `hw/core.py`

Tests:

- typed round-trip loads/stores
- out-of-bounds checks

Done when:

- non-FP32 typed accesses work in the simulator.

### 1.3 Add INT4/UINT4 packed layout semantics

Implement:

- two elements per byte,
- signed/unsigned unpack,
- zero-point handling later.

Files:

- `hw/memory.py`
- `docs/types.md`

Tests:

- pack/unpack round-trip
- signed/unsigned edge cases

Done when:

- INT4/UINT4 packing matches NumPy bit-level references.

### 1.4 Add quantize/dequantize ops

Define:

- `mgpu.quant`
- `mgpu.dequant`
- `mgpu.requant`

Attributes:

- scale
- zero point
- storage type
- compute type

Files:

- `compiler/mlir/mgpu_dialect.td`
- `compiler/mlir/mlir_builder.py`

Tests:

- parse/print quant ops
- semantic unit tests

Done when:

- quant/dequant metadata is represented explicitly.

---

## Phase 2: Real Model Import

### 2.1 Import PyTorch FX with torch-mlir

Status: **done**

Reference: [step-2.1.md](step-2.1.md)

Implement:

- replace manual FX converter with `torch_mlir` import,
- produce stable MLIR module,
- preserve parameter and buffer names.

Files:

- `compiler/frontends/torch_fx_frontend.py`
- `compiler/mlir/importer.py` (new)

Tests:

- small CNN,
- small Transformer block,
- deterministic output.

Done when:

- real FX modules produce parsed MLIR.

### 2.2 Import ONNX with torch-mlir/onnx-mlir

Status: **done**

Reference: [step-2.2.md](step-2.2.md)

Implement:

- ONNX to MLIR import,
- initializer handling,
- dynamic dimension placeholder support.

Files:

- `compiler/frontends/onnx_frontend.py`
- `compiler/mlir/importer.py`

Tests:

- exported CNN,
- exported Transformer block.

Done when:

- real ONNX models import without manual op mapping.

### 2.3 Canonicalize imported MLIR

Status: **done**

Reference: [step-2.3.md](step-2.3.md)

Run:

- canonicalization,
- shape inference,
- decomposition.

Target dialects:

- `linalg`
- `tosa`
- `arith`
- `math`
- `memref`

Tests:

- op decomposition is deterministic.

Done when:

- imported models lower to a common high-level dialect set.

---

## Phase 3: Core MLIR Passes

### 3.1 Implement `model-to-linalg`

Status: **done**

Reference: [step-3.1.md](step-3.1.md)

Lower high-level model ops:

- attention,
- layer norm,
- RMS norm,
- softmax,
- convolution,
- pooling.

Tests:

- one lit test per op.

### 3.2 Implement tiling

Status: **done**

Reference: [step-3.2.md](step-3.2.md)

Tile:

- matmul,
- conv,
- reductions,
- attention.

Configurable tile sizes:

- `16x16x16` for matmul,
- `8x8` conv tiles,
- sequence tiles for attention.

Tests:

- tiled IR has expected `scf.for` nests.

### 3.3 Implement bufferization

Status: **done**

Reference: [step-3.3.md](step-3.3.md)

Convert tensors to `memref`.

Tests:

- no tensor types remain after the pass.

### 3.4 Implement grid/block/warp mapping

Status: **done**

Reference: [step-3.4.md](step-3.4.md)

Lower loops to:

- `mgpu.launch`,
- `mgpu.block_id`,
- `mgpu.thread_id`,
- `mgpu.grid_dim`,
- `mgpu.block_dim`.

Tests:

- mapping is deterministic for fixed launch sizes.

### 3.5 Implement `lower-to-mgpu`

Status: **done**

Reference: [step-3.5.md](step-3.5.md)

Insert:

- `mgpu.shared_alloc`,
- `mgpu.async_copy`,
- `mgpu.barrier`,
- `mgpu.mma`.

Tests:

- shared-memory tiles and barriers are present.

### 3.6 Implement `mgpu-to-isa`

Status: **done**

Reference: [step-3.6.md](step-3.6.md)

Emit ISA:

- register allocation,
- instruction scheduling,
- labels,
- memory addresses,
- launch dimensions.

Tests:

- emitted ISA runs in the simulator and matches NumPy.

---

## Phase 4: Tensor-Core MMA

### 4.1 Execute `MMA` in the simulator

Status: **done**

Reference: [step-4.1.md](step-4.1.md)

Implement functional matrix multiply-accumulate.

Supported:

- `f32`,
- `f16`,
- `bf16`.

Files:

- `hw/core.py`
- `hw/isa.py`

Tests:

- one `MMA` tile matches NumPy.

### 4.2 Add fragment layout and shared-memory swizzle

Status: **done**

Reference: [step-4.2.md](step-4.2.md)

Implement:

- A/B/C fragment layout,
- swizzle pattern,
- bank-conflict reporting.

Tests:

- conflict counters are stable.

### 4.3 Lower tiled matmul to MMA

Status: **done**

Reference: [step-4.3.md](step-4.3.md)

Replace scalar FMA inner loops with `mgpu.mma`.

Tests:

- matmul E2E.

### 4.4 Lower conv and attention to MMA

Status: **done**

Reference: [step-4.4.md](step-4.4.md)

Use im2col-style tiling for conv and attention.

Tests:

- conv E2E,
- attention E2E.

---

## Phase 5: Quantization

### 5.1 Import QDQ patterns

Status: **done**

Recognize:

- `QuantizeLinear`,
- `DequantizeLinear`,
- QDQ-wrapped weights/activations.

Tests:

- QDQ graph lowers to quant/dequant ops.

The ONNX importer plus the canonicalizer already lower QDQ to
`arith.fptosi`/`arith.sitofp` inside `linalg.generic`; the explicit
`mgpu.quant`/`mgpu.dequant`/`mgpu.requant` ops are the target dialect for
manually constructed quantized graphs.  See
`tests/unit/test_mlir_canonicalizer.py`.

### 5.2 Lower INT8 matmul/conv

Status: **done** for scalar INT8 matmul and conv.

Implement:

- integer SIMT matmul,
- tensor-core INT8 matmul,
- accumulator rescale.

Tests:

- INT8 matmul/conv match PTQ reference.

Implemented as typed SIMT scalar matmul/conv fallbacks in
`MGpuToIsaPass.cpp` with INT32 accumulation, plus a `MMA.I8`
16x16x16 tensor-core path for integer matmul.  See
`tests/operator_e2e/test_quantization_e2e.py`.

### 5.3 Add INT4/UINT4 unpack/dequant

Status: **done**

Implement:

- packed loads,
- signed/unsigned unpack,
- scale/zero-point dequant.

Tests:

- packed matmul matches reference.

Nibble-address semantics are implemented for `mgpu.dequant` and covered by
the packed-memory and quantization E2E tests.

### 5.4 Add FP8 E4M3/E5M2

Status: **done**

Implement:

- conversion,
- FP8 matmul,
- MMA FP8 path.

Tests:

- FP8 matmul within documented tolerance.

The simulator now has `f8E4M3FN` and `f8E5M2` typed memory access, and the
scalar matmul fallback can load/store FP8 operands and accumulate in FP32.
See `tests/operator_e2e/test_typed_memory_e2e.py` and
`tests/operator_e2e/test_quantization_e2e.py`.

---

## Phase 6: Real Model Examples

### 6.1 ResNet-style block

Status: **done**

- export from PyTorch/ONNX,
- lower through full pass pipeline,
- compare with framework output.

The block is exported through both the PyTorch FX frontend and the ONNX
frontend, lowered with the current C++ ``mgpu-opt`` backend, and compared
against PyTorch and the ONNX reference evaluator.

Tests: `tests/model_e2e/test_real_cnn.py`

### 6.2 YOLO/SSD-style detection head

- box decode,
- NMS,
- export and run.

### 6.3 Small GPT decoder block

- causal mask,
- attention,
- residual.

### 6.4 KV-cache decode

- append,
- gather,
- decode step.

### 6.5 Autoregressive loop

- run several decode steps,
- compare outputs.

### 6.6 Real CNN target

Target:

- `MobileNetV3-Small` or `ResNet18`

Steps:

1. Export from PyTorch and ONNX.
2. Freeze to static `1x3x224x224`.
3. Import with `torch-mlir`.
4. Add missing op/pass coverage discovered by the importer.
5. Run `mgpu-opt` lowering.
6. Execute in simulator.
7. Compare output against PyTorch/ONNX Runtime.

Tests:

- `tests/model_e2e/test_real_cnn.py`

### 6.7 Real small LLM target

Target order:

1. `Pythia-70M`
2. `GPT-2-small`
3. `TinyLlama` as stretch

Steps:

1. Export/trace a small decoder model.
2. Staticize sequence length.
3. Import with `torch-mlir`.
4. Implement or complete general `Linear`, `LayerNorm`, `Softmax`,
   attention, RoPE, and KV-cache lowering.
5. Implement memory planning for model weights and KV-cache.
6. Run a greedy decode loop.
7. Compare generated token IDs with PyTorch.

Tests:

- `tests/model_e2e/test_real_llm_decode.py`

---

## Hardware scaling

Before real models can run, extend the simulator configuration:

- larger global memory,
- more SMs,
- multiple resident blocks per SM,
- shared-memory size control,
- register-file size control,
- memory allocator for model weights, activations, and KV-cache.

---

## Phase 7: Runtime and Hardening

### 7.1 Add launch queue/command buffer

Files:

- `runtime/driver.py`
- `runtime/profiler.py`

Tests:

- queued launches execute deterministically.

### 7.2 Add explicit synchronization

Tests:

- synchronization does not corrupt state.

### 7.3 Add source-location diagnostics

Include:

- op,
- shape,
- dtype,
- pass,
- source location.

Tests:

- invalid graph errors contain all fields.

### 7.4 Add autotuning hooks

Expose:

- tile sizes,
- MMA layouts,
- shared-memory padding.

Tests:

- autotuner runs without changing numerical results.

---

## Final verification

Run:

```bash
pre-commit run --all-files
python -m unittest discover -v
lit -q tests/mlir
```

The stage is complete when all real-model examples and quantization tests pass.
