# Next Stage Plan: Real MLIR Lowering for Production PyTorch/ONNX Models

This document defines the next major stage after the Python simulator MVP.

Detailed ordered engineering steps are in
[next-stage-steps.md](next-stage-steps.md).

The goal is to replace the handwritten `Model IR -> ISA` lowerings with a real
MLIR pass pipeline and to make the hardware/compiler stack capable of lowering
real exported PyTorch and ONNX models to simulator ISA.

The plan explicitly includes:

- real MLIR parsing/verification and pass execution,
- missing FP16/BF16/FP8/INT8/INT4/quantized type support,
- tensor-core `mgpu.mma` execution,
- quantization/dequantization lowering,
- realistic model-level tests.

---

## 1. Guiding principles

1. **One canonical lowering path.**

   ```text
   PyTorch FX / ONNX
          |
          v
   MLIR: torch/tosa/linalg + mgpu
          |
          v
   tiling, bufferization, vectorization
          |
          v
   mgpu device dialect
          |
          v
   simulator ISA
   ```

2. **Functional correctness before performance.**
   Every pass must keep NumPy/PyTorch golden parity.

3. **Explicit types.**
   Storage type and compute type are represented separately.

4. **No silent fallback.**
   Unsupported ops, dtypes, or dynamic shapes fail with a clear diagnostic.

5. **Test each lowering layer.**
   Use `lit`/FileCheck for MLIR text, plus E2E golden tests.

---

## 2. Current-state assumptions

- The simulator is functional and deterministic for FP32.
- The Model IR, manual frontend, MLIR text builder, runtime, and per-op ISA
  lowerings already exist.
- Phases 2.1 through 2.3 are complete: real PyTorch FX and ONNX models import
  with `torch-mlir`, preserve parameter/initializer/dynamic-shape metadata, and
  canonicalize to `linalg-on-tensors` or `tosa`.
- The current MLIR builder emits text but does not parse or run passes.
- The `mgpu` dialect exists only as TableGen/ODS text.
- Simulator `MMA` is reserved but not executed.
- Only FP32 execution is implemented.

---

## 3. Work tracks

### Track A: Real MLIR toolchain

Build and use actual MLIR/LLVM libraries or Python bindings.

### Track B: Hardware type/tensor-core/quantization support

Extend the simulator and ISA so MLIR has a complete target.

### Track C: Model import and lowering

Import real PyTorch and ONNX models into a canonical MLIR dialect.

### Track D: Runtime and model validation

Run real exported models end to end and compare against framework output.

---

## 4. Phase plan

### Phase 0: Toolchain and dialect registration

Deliverables:

- Integration decision is documented in
  [mlir-integration-decision.md](mlir-integration-decision.md).
- Build a minimal `mgpu-opt` / `mgpu-translate` executable.
- Register the `mgpu` dialect from `mgpu_dialect.td`.
- Add CMake/build integration.
- Add `lit` tests that parse and verify the emitted MLIR.

Exit criteria:

- `mgpu.launch`, `mgpu.thread_id`, `mgpu.barrier`, and `mgpu.mma` parse and
  verify.
- Existing Python-generated MLIR text passes a basic parser smoke test.

### Phase 1: Type and memory legalization

Deliverables:

- Complete the `DType` and MLIR type mapping for:
  - `f32`
  - `f16`
  - `bf16`
  - `f8E4M3FN`
  - `f8E5M2`
  - `i8`
  - `i4`
  - `ui4`
- Add typed load/store opcodes or type attributes to the simulator memory
  model.
- Define packed layout semantics for INT4/UINT4.
- Add `mgpu.quant`, `mgpu.dequant`, and `mgpu.requant` ops with scale and
  zero-point metadata.

Exit criteria:

- Typed memory accesses pass unit tests.
- INT4/UINT4 packing/unpacking matches NumPy references.

### Phase 2: Real model import

Deliverables:

- PyTorch FX import via `torch-mlir`.
- ONNX import via `torch-mlir-import-onnx` or `onnx-mlir`.
- Canonicalization to `linalg`/`tosa`.
- Static-shape path first.
- Dynamic-shape support for sequence lengths later.

Exit criteria:

- A small exported ResNet-style block imports and lowers.
- A small Transformer block imports and lowers.
- A small exported GPT/decoder block imports and lowers.

### Phase 3: Core MLIR passes

Deliverables:

- `model-to-linalg`
- `linalg-tiling`
- `bufferization`
- `convert-linalg-to-loops`
- `map-to-mgpu`
- `lower-to-mgpu`
- `mgpu-to-isa`

Pass responsibilities:

```text
model-to-linalg:
    decompose high-level ops to linalg/tosa.

linalg-tiling:
    tile matmul/conv/reductions.

bufferization:
    convert tensors to memrefs.

map-to-mgpu:
    map loops to grid/block/warp/thread IDs.

lower-to-mgpu:
    insert shared_alloc, async_copy, barrier, and mma.

mgpu-to-isa:
    emit ISA text, register allocation, and scheduling metadata.
```

Exit criteria:

- Matmul, conv, elementwise, reduction, softmax, layer norm, and attention
  lower through the full pass pipeline.
- All pass outputs are deterministic.

### Phase 4: Tensor-core MMA

Deliverables:

- Execute `MMA` in the simulator.
- Support at least:
  - `mma.16x16x16.f32`
  - `mma.16x16x16.f16`
  - `mma.16x16x16.bf16`
- Lower tiled matmul/conv/attention to shared-memory + `mgpu.mma`.
- Implement fragment layout and shared-memory swizzling.

Exit criteria:

- Tiled matmul using `MMA` matches NumPy.
- Conv/attention using `MMA` match reference.
- Profile mode reports MMA utilization.

### Phase 5: Quantization

Deliverables:

- Import ONNX `QuantizeLinear`/`DequantizeLinear` QDQ patterns.
- Lower quantized matmul/conv to integer SIMT or tensor-core kernels.
- INT8 matmul and conv.
- INT4/UINT4 unpack-and-dequantize.
- FP8 E4M3/E5M2 matmul.
- Scale/zero-point propagation.

Exit criteria:

- INT8 matmul and conv match a PTQ reference within documented tolerance.
- INT4 unpack/dequant matches reference.
- FP8 matmul matches reference within FP8 tolerance.

### Phase 6: Real model examples

Deliverables:

- ResNet block imported from PyTorch/ONNX.
- YOLO/SSD-style detection head.
- Small GPT-style decoder block with causal mask.
- KV-cache decode step.
- Small autoregressive loop.

Exit criteria:

- Real exported models run end to end through the simulator.
- Outputs match PyTorch/ONNX Runtime references.

### Phase 7: Runtime and hardening

Deliverables:

- Command buffer or launch queue.
- Explicit synchronization.
- Profiling integration for MLIR-generated kernels.
- Better diagnostics with source location and op/shape/dtype.
- Autotuning hooks for tile sizes and MMA layouts.

Exit criteria:

- Multiple launches are deterministic.
- Errors include the failing op, shape, dtype, and pass.

---

## 5. Missing-type roadmap

| Type | Storage | Priority | Required work |
| --- | --- | --- | --- |
| FP32 | 4 bytes | done | current baseline |
| FP16 | 2 bytes | high | typed loads/stores + SIMT arithmetic + MMA |
| BF16 | 2 bytes | high | typed loads/stores + conversion + MMA |
| FP8 E4M3FN | 1 byte | medium | conversion, scale, MMA |
| FP8 E5M2 | 1 byte | medium | conversion, scale, MMA |
| INT8 | 1 byte | high | QDQ, SIMT matmul, tensor-core path |
| INT4 | packed | medium | unpack/dequant |
| UINT4 | packed | medium | unpack/dequant |
| BOOL | 1 byte | support | masks and predicates |

---

## 6. Tensor-core execution model

Proposed `MMA` target shape:

```text
MMA m16n16k16.f32
MMA m16n16k16.f16
MMA m16n16k16.bf16
MMA m16n16k16.f8e4m3
```

Lowering flow:

```text
linalg.matmul
   -> tile to 16x16x16
   -> bufferize to memref
   -> copy tiles to shared memory with swizzle
   -> barrier
   -> mgpu.mma
   -> epilogue add bias/requant
```

Simulator work:

- add MMA functional execution,
- add matrix fragment layout,
- add shared-memory bank/conflict tracking for MMA layouts,
- add profile counters for MMA issues and utilization.

---

## 7. Testing strategy

### MLIR layer

- `lit` + FileCheck tests for every pass.
- Round-trip parse/print tests.
- Negative tests for unsupported ops/dtypes.

### Operator layer

- One E2E test per operator.
- Golden comparison with NumPy/PyTorch.
- Documented tolerances in `docs/tolerances.md`.

### Model layer

- Small exported models.
- ONNX Runtime or PyTorch as reference.
- Determinism and profiler checks.

---

## 8. Real CNN and LLM targets

The plan is not complete without explicit real-model acceptance gates.

### CNN target

Primary target:

- `MobileNetV3-Small` or `ResNet18`

Input:

```text
1x3x224x224
```

Required path:

```text
PyTorch or ONNX export
  -> torch-mlir import
  -> mgpu-opt lowering
  -> ISA
  -> simulator
  -> PyTorch/ONNX Runtime comparison
```

Acceptance:

- predicted class or output tensor matches reference within documented
  tolerance,
- deterministic across repeated runs,
- profiler reports plausible memory traffic and cycles.

### LLM target

Primary targets, in order:

1. `Pythia-70M`
2. `GPT-2-small`
3. `TinyLlama` as stretch

Input:

```text
tokenized prompt with static sequence length
```

Required path:

```text
tokenizer input
  -> embedding/attention/FFN
  -> greedy decode
  -> KV-cache append/gather
  -> generated token IDs
```

Acceptance:

- generated token IDs match a PyTorch reference decode loop,
- KV-cache decode is deterministic,
- memory planner handles the model's KV-cache size.

### Hardware sizing

The simulator must be scaled from the current 1 MB / 4 SMs to support the
chosen models. Add explicit knobs for:

- global memory size,
- SM count,
- blocks per SM,
- shared-memory size,
- register file size.

---

## 9. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| C++ MLIR build complexity | Use prebuilt wheels/containers or Python bindings first |
| Dynamic shapes | Implement static path first, add shape functions incrementally |
| Quantization accuracy | Use PTQ references and documented tolerances |
| Shared-memory bank conflicts | Add conflict-aware padding and profile counters |
| Pass ordering bugs | Keep passes small and lit-tested individually |
| Large model memory | Start with tiny real models, grow device memory |

---

## 10. Definition of done

The stage is complete when:

- real MLIR passes parse and lower exported PyTorch/ONNX models,
- FP16/BF16/FP8/INT8/INT4 types are executable,
- tensor-core MMA is functional,
- quantized matmul/conv match references,
- all MLIR/operator/model tests pass,
- runtime diagnostics are clear and deterministic.
