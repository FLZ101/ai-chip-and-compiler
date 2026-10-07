# Plan: Minimal GPU Simulator and MLIR Compiler for CNN and Transformer/LLM Inference

## Final goal

Design and simulate a minimal GPU that captures the essential behavior of modern GPUs, and build an MLIR-based compiler for it that runs:

- ordinary CNN inference, including object-detection-style models;
- small Transformer/LLM inference workloads, including encoder blocks, decoder/GPT-style blocks, and KV-cache decoding.

CNN/object detection and Transformer/LLM are treated as **joint first-class targets**, not sequential goals. They share one hardware model, one ISA, one MLIR lowering pipeline, one runtime, one kernel library, and one end-to-end test harness.

Every operator gets an end-to-end test through the full path:

```text
op/model input
   -> frontend / Model IR
   -> MLIR lowering
   -> mgpu + ISA codegen
   -> GPU simulator
   -> numeric output compared against NumPy/reference
```

An "operator test" is a full compiler + simulator integration test, not only a unit test of one Python kernel function.

---

## 1. Design principles

These principles apply from the first sprint:

- **One execution model:** SIMT warps, active masks, barriers, shared memory, and tensor-core-like MMA are shared by convolution and attention.
- **One compiler pipeline:** frontend -> Model IR -> MLIR -> tiling/mapping -> `mgpu` -> ISA.
- **One runtime:** grid/block launch, buffers, synchronization, and profiling are domain-neutral.
- **One test framework:** CNN and Transformer/LLM operators use the same E2E harness and tolerances.
- **Dataflow first:** matmul, reductions, elementwise, softmax, and normalization are common building blocks. Conv2d and attention are domain-specific compositions over those blocks.
- **No domain-specific fork:** do not create a separate "CNN compiler" and "Transformer compiler".

---

## 2. Scope and non-goals

### In scope

- Minimal modern-GPU-like architecture: SMs, warps, lanes, global/L2/shared/register memory, barriers, MMA unit.
- Cycle-approximate simulator.
- MLIR-based compiler using standard MLIR plus a small `mgpu` device dialect.
- Operator set covering both normal CNNs/object detectors and small Transformer/LLM blocks.
- End-to-end tests for every supported operator.
- Model-level tests for both a CNN/object-detection network and a Transformer block.

### Out of scope for MVP

- Real silicon, FPGA synthesis, or cycle-exact hardware verification.
- Production runtime, dynamic-shape autotuning, and large-scale deployment.
- Quantized INT8/INT4 execution unless added later.
- Full LLM training and very large model serving.
- Literally every model or framework feature.

### Fixed MVP choices

- **Simulator and compiler:** Python.
- **Data type:** FP32 first. Reserve FP16/BF16 in the ISA and memory typing.
- **ISA:** small scalar/vector SIMT ISA with explicit memory, barrier, and MMA instructions.
- **Memory:** explicit global/shared/register address spaces.
- **Execution:** 32-lane warps, active masks, per-SM warp scheduler.
- **Model IR:** CNN and Transformer/LLM ops are both present from the first Model IR definition.
- **MLIR:** standard `func`, `arith`, `math`, `memref`, `scf`, `linalg`/`tosa`, plus a small `mgpu` dialect.

---

## 3. Target workloads

### CNN/object-detection track

1. ResNet-style block with residual connection.
2. FPN-style neck with upsample and concatenation.
3. YOLO/SSD-style detection head.
4. Small full detector on a tiny synthetic image.

### Transformer/LLM track

1. Transformer encoder block.
2. GPT/decoder-style block with causal attention.
3. KV-cache append and gather.
4. Small autoregressive decode loop with changing sequence length.

Both tracks are validated against NumPy/reference implementations.

---

## 4. Unified operator inventory

| Category | Operators | Used by |
| --- | --- | --- |
| Matmul/linear | `MatMul`, `Linear`, `BatchMatMul` | CNN FC layers; Transformer QKV/attention/projection |
| Convolution | `Conv2d`, `DepthwiseConv2d` | CNN backbone and detection head |
| Pooling | `MaxPool2d`, `AvgPool2d`, `GlobalAvgPool` | CNN |
| Activation | `ReLU`, `LeakyReLU`, `SiLU/Swish`, `GELU`, `Sigmoid`, `HardSigmoid` | CNN and Transformer |
| Elementwise | `Add`, `Mul`, `Sub`, `Div`, `FMA` | Residuals and fused blocks |
| Reduction | `ReduceSum`, `ReduceMean`, `ReduceMax` | CNN pooling and Transformer softmax/norm |
| Normalization | `BatchNorm`, `GroupNorm`, `LayerNorm`, `RMSNorm` | CNN norm; Transformer norm |
| Softmax/attention | `Softmax`, `MaskedSoftmax`, `ScaledDotProductAttention`, `MaskedAttention` | Transformer and detection score paths |
| Position/sequence | `RoPE`, `KVCacheAppend`, `KVCacheGather`, `CausalMask` | Transformer/LLM |
| Shape/layout | `Reshape`, `Transpose`, `Concat`, `Slice`, `Pad`, `Resize/Upsample`, `Expand`, `Squeeze`, `Unsqueeze` | Both tracks |
| Detection helpers | `BoxDecode`, `NMS` | Object detection |

Every operator gets:

- deterministic small inputs,
- a NumPy/reference expected result,
- an end-to-end compiler/simulator test,
- a documented tolerance.

### Generic fallback

Unknown ops are decomposed when possible. If decomposition is not available, the compiler fails with op name, shape, dtype, and reason. Silent wrong-code fallback is forbidden.

---

## 5. Repository structure

```text
ai-chip-and-compiler/
├── PLAN.md
├── README.md
├── hw/
│   ├── isa.py
│   ├── memory.py
│   ├── core.py
│   ├── device.py
│   └── trace.py
├── compiler/
│   ├── model_ir.py
│   ├── frontends/
│   │   ├── manual_frontend.py
│   │   ├── onnx_frontend.py
│   │   └── torch_fx_frontend.py
│   ├── mlir/
│   │   ├── mgpu_dialect.td
│   │   ├── mlir_builder.py
│   │   └── passes.py
│   ├── codegen/
│   │   ├── scheduler.py
│   │   ├── kernels.py
│   │   └── isa_emitter.py
│   └── pipeline.py
├── runtime/
│   ├── buffer.py
│   ├── driver.py
│   └── profiler.py
├── kernels/
│   └── library.py
├── tests/
│   ├── unit/
│   ├── operator_e2e/
│   └── model_e2e/
├── examples/
│   ├── cnn/
│   │   ├── resnet_block.py
│   │   ├── fpn.py
│   │   └── yolo_style_head.py
│   └── transformer/
│       ├── encoder_block.py
│       ├── gpt_block.py
│       └── kvcache_decode.py
└── scripts/
    ├── compile_model.py
    └── simulate.py
```

---

## 6. Hardware specification

### Device

- Configurable number of SMs; default 4.
- Global memory controller with a simple L2 cache.
- Host command processor.

### Streaming multiprocessor

- 32 lanes, one warp.
- 64 KiB register file.
- 64 KiB shared memory.
- One warp scheduler with scoreboard.
- FP32 vector ALU: add, mul, fma, compare, select.
- Simplified 16x16x16 FP32 MMA unit.

### Memory hierarchy

- Global memory: byte-addressable array, large enough for CNN activations and KV-cache tensors.
- L2 cache: simple configurable cache.
- Shared memory: per-SM scratchpad with explicit barriers.
- Registers: per-thread virtual registers.

### Execution model

- Kernel launches as a grid of thread blocks.
- Blocks map to SMs.
- Threads execute in 32-thread warps.
- Divergent branches use active masks and reconvergence.
- Explicit block barriers.

The same shared-memory tiling and reduction model supports both conv reuse and attention reuse.

---

## 7. Instruction set architecture

Instruction groups:

- ALU: `ADD`, `SUB`, `MUL`, `FMA`, `DIV`, `MOV`, `NEG`, `ABS`, `MAX`, `MIN`.
- Predicate: `CMP`, `SEL`, `PRED`.
- Special math: `RCP`, `RSQRT`, `EXP2`, `LOG2`.
- Memory: `LDG`, `STG`, `LDS`, `STS`, `LDG.128`, `STG.128`.
- Control: `BRA`, `RET`, `EXIT`, `BAR.SYNC`.
- Tensor: `MMA.16x16x16.f32`.
- Address spaces: global, shared, local, constant.
- Special registers: `tid`, `ntid`, `ctaid`, `nctaid`, `laneid`, `warpid`, `clock`.

These instructions cover:

- CNN conv/matmul/elementwise/pooling paths.
- Transformer softmax, LayerNorm/RMSNorm, RoPE, attention, and KV-cache paths.

Deliverable: `hw/isa.py`, ISA reference in `docs/isa.md`, and a vector-add ISA smoke test.

---

## 8. Functional simulator build order

1. Memory spaces with bounds checks.
2. Lane register files and thread state.
3. Warp scheduler with active masks.
4. ALU semantics.
5. Global/shared load/store semantics.
6. Branch, return, exit, barrier.
7. Kernel launch and grid/block/warp ID calculation.
8. Instruction trace/debugger.

Acceptance:

- Handwritten vector-add ISA program matches NumPy.
- Reduction and divergent-branch ISA programs match scalar execution.
- Shared-memory barrier behavior is deterministic.

---

## 9. Cycle-approximate timing model

After functional correctness:

- Per-instruction latency.
- Memory latency and bandwidth.
- Stall reasons: fetch, ALU, memory, barrier, scoreboard.
- Shared-memory bank-conflict detection.
- L2 cache hit/miss counters.
- Profile mode: cycles, IPC, stall counts, cache stats, bandwidth.

Acceptance:

- Stable cycle counts across runs.
- Tiled matmul is faster than naive matmul.
- Conv2d and attention kernels report expected memory traffic and reuse patterns.

---

## 10. MLIR compiler

### Pipeline

```text
PyTorch FX / ONNX / manual graph
            │
            ▼
Model IR
            │
            ▼
MLIR textual IR: func + arith + math + memref + scf + linalg/tosa + mgpu
            │
            ▼
Lowering: bufferize -> tile -> map to grid/block/warp -> vectorize -> codegen
            │
            ▼
mgpu device ops and explicit memory ops
            │
            ▼
ISA assembly
            │
            ▼
GPU simulator
```

### Custom `mgpu` dialect

Common device ops:

- `mgpu.launch`
- `mgpu.thread_id`
- `mgpu.block_id`
- `mgpu.block_dim`
- `mgpu.grid_dim`
- `mgpu.shared_alloc`
- `mgpu.shared_dealloc`
- `mgpu.barrier`
- `mgpu.async_copy`
- `mgpu.mma`
- `mgpu.alloc_global`
- `mgpu.load_global`
- `mgpu.store_global`

Higher-level ops for the shared backend:

- `mgpu.softmax`
- `mgpu.layernorm`
- `mgpu.rmsnorm`
- `mgpu.rope`
- `mgpu.attention`
- `mgpu.kvcache_update`

### Standard MLIR use

- `linalg.matmul`
- `linalg.conv_2d_nhwc_hwcf`
- `linalg.generic`
- `scf.for`, `scf.if`
- `memref.alloc`, `memref.load`, `memref.store`
- `arith` and `math`

### Shared codegen strategy

1. Elementwise: thread-per-element.
2. Reduction: shared-memory tree.
3. Matmul: tiled global->shared->register.
4. Softmax: online max/sum reduction.
5. LayerNorm/RMSNorm: row-wise reduction + affine.
6. Attention: tiled QK^T, online softmax, PV matmul.
7. Conv2d: im2col + matmul first, then optional direct tiled convolution.
8. Pooling: sliding-window reduction.
9. RoPE/KV-cache: layout and index transforms over the same memory model.
10. Detection helpers: box decoding and NMS.

The lowering path is shared. CNN-specific and Transformer-specific logic is isolated in op lowering/kernel templates, not in the scheduler or runtime.

---

## 11. Frontends and Model IR

Define `compiler/model_ir.py` with:

- typed values and graph nodes;
- the full unified operator inventory from section 4;
- shape, dtype, strides, padding, groups, mask, and cache metadata;
- first-class CNN and Transformer/LLM nodes from the first definition.

Frontends:

- Manual frontend for deterministic tests.
- ONNX frontend when `onnx` is available.
- PyTorch FX frontend when `torch` is available.

Frontend behavior:

- Emit clear diagnostics for unsupported ops.
- Preserve metadata needed for shape inference and lowering.
- Do not silently ignore attributes.

---

## 12. Per-operator end-to-end test strategy

One shared harness for all operators:

1. Generate deterministic small input tensors.
2. Build a one-op model graph.
3. Compile to MLIR.
4. Lower to ISA.
5. Execute on the simulator.
6. Compute reference with NumPy/reference implementation.
7. Compare with `np.allclose` and documented tolerance.

The harness is domain-neutral. Adding an operator requires:

- a reference implementation,
- a Model IR builder,
- a lowering/kernel mapping entry.

Example CNN E2E cases:

- `Conv2d`: `[1,3,32,32] -> [1,8,30,30]`, 3x3 kernel, stride 1.
- `MaxPool2d`: `[1,4,16,16] -> [1,4,8,8]`.
- `Resize/Upsample`: `[1,8,8,8] -> [1,8,16,16]`.
- `Concat` and residual `Add` on feature maps.

Example Transformer/LLM E2E cases:

- `MatMul`: `[2,3,4] x [4,5] -> [2,3,5]`.
- `Softmax`: `[2,8,16]` over the last dim.
- `LayerNorm`/`RMSNorm`: `[2,16,64]`.
- `RoPE`: `[1,8,64]`.
- `ScaledDotProductAttention`: small `Q/K/V` tensors.
- `KVCacheAppend`/`KVCacheGather`: fixed cache size and changing sequence length.

Tests live in `tests/operator_e2e/`.

---

## 13. Model-level end-to-end tests

Both domains are tested as whole models.

CNN/object detection:

1. ResNet block with residual add.
2. FPN-style upsample + concat.
3. YOLO/SSD-style detection head.
4. Full small detector on a tiny synthetic image.

Transformer/LLM:

1. Transformer encoder block.
2. GPT/decoder-style block with causal mask.
3. KV-cache decode step with changing sequence length.
4. Small multi-step autoregressive loop.

Acceptance:

- Full CNN/detector and Transformer models compile without unsupported-op errors.
- Simulator output matches reference within tolerance.
- CNN model contains conv, activation, elementwise, shape, and detection ops.
- Transformer model contains matmul, softmax, layernorm/RMSNorm, RoPE, attention, and KV-cache ops.

---

## 14. Runtime and driver

- Device handle.
- Buffer allocation/deallocation.
- Host-to-device and device-to-host copies.
- Kernel launch with grid/block dimensions and scalar args.
- Device synchronization.
- Profiling counters.

Acceptance:

- Multiple launches do not corrupt state.
- Device memory is reused deterministically.
- Sync semantics match a CUDA-like model.

The runtime treats CNN and Transformer kernels identically.

---

## 15. Build order and milestones

The order builds a common foundation first, then interleaves CNN-specific and Transformer-specific work. Both domain tracks are required for completion.

| Step | Work | Exit condition |
| --- | --- | --- |
| 1 | Write ISA and hardware spec for both domains | ISA covers conv/matmul/elementwise/reduction/softmax/norm/attention/rope/kvcache |
| 2 | Implement memory and ALU | Vector-add ISA program matches NumPy |
| 3 | Implement warps, divergence, barriers | Reduction/divergence tests pass |
| 4 | Add kernel launch and device model | Multi-block grid executes correctly |
| 5 | Add cycle counters | Profile mode reports stable stats |
| 6 | Define unified Model IR and manual frontend | CNN + Transformer op nodes parse and lower to MLIR skeleton |
| 7 | Define `mgpu` dialect and MLIR builder | Emitted MLIR is deterministic |
| 8 | Build shared E2E operator harness | First operator E2E test passes |
| 9 | Add common kernels: elementwise/reduction/matmul | Shared matmul/reduce/elementwise E2E tests pass |
| 10 | Add activation and shape/layout kernels | Activation/reshape/transpose/concat/slice/pad/resize E2E tests pass |
| 11 | Add CNN kernels: conv2d/depthwise conv/pooling | Conv and pooling E2E tests pass |
| 12 | Add CNN normalization/detection helpers | BN/GroupNorm/BoxDecode/NMS E2E tests pass |
| 13 | Add Transformer kernels: softmax/layernorm/rmsnorm | Transformer norm/softmax E2E tests pass |
| 14 | Add Transformer kernels: rope/attention/kv-cache | Attention and KV-cache E2E tests pass |
| 15 | Add ONNX/PyTorch FX frontends | Exported CNN and Transformer paths run end to end |
| 16 | Add runtime/driver | Multi-launch/sync passes |
| 17 | Add CNN model examples | ResNet/FPN/detector model tests pass |
| 18 | Add Transformer model examples | Encoder/GPT/KV-cache model tests pass |
| 19 | Polish docs/tests/errors | Full shared suite green |

Steps 11-14 can be worked in either order, but both must pass before model-level integration.

---

## 16. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Conv2d and attention both need high-performance tiling | Share the same tiled matmul/reduction scheduler; implement im2col and attention tiles over it |
| Transformer requirements add early complexity | Keep Model IR/ISA interfaces stable and small; implement one common kernel at a time |
| Numerical mismatch in reductions | Use fixed reduction order and documented tolerances |
| Object-detection postprocessing is large | Implement a small reference path first; move NMS to device later |
| ONNX/PyTorch export is environment-dependent | Manual frontend is source of truth; framework frontends are optional adapters |
| Simulator is slow for full models | Use tiny synthetic models and limit default image/sequence sizes |
| Dynamic sequence length complicates codegen | Treat grid/block dims and scalar args as runtime values; avoid baking shapes into kernel code |

---

## 17. First implementation sprint

The first sprint proves the shared foundation and does not fork the two domains:

1. `hw/isa.py`
2. `hw/memory.py`
3. `hw/core.py`
4. `hw/device.py`
5. `compiler/model_ir.py` skeleton with the unified CNN + Transformer operator inventory
6. `tests/unit/test_isa.py`
7. `tests/operator_e2e/test_vector_add_e2e.py`

After this slice works, matmul/reduction/elementwise become the shared kernel layer, and CNN and Transformer kernels are added on top of the same scheduler and test harness.
