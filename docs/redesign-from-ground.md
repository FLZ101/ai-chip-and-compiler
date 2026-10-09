# Redesign from Ground

This document proposes a clean-slate architecture for the compiler and GPU
simulator. It is not an immediate rewrite plan; it is the reference design to
guide incremental migration away from the current MVP structure.

## 1. Why redesign

The current codebase evolved from:

```text
Python Model IR
  -> hand-written MLIR text
  -> custom mgpu dialect
  -> Python/C++ ISA simulator
```

That path proved the product ideas, but it also created coupling:

- `Model IR`, MLIR text emission, target ops, and simulator ISA evolved
  together,
- software fallback kernels live beside compiler tests,
- runtime memory planning is partly duplicated between Python and C++,
- simulator timing, functional execution, and profiling are mixed,
- diagnostics are useful but not consistently structured.

A ground-up design separates those concerns and makes each layer independently
specifiable and testable.

## 2. Design goals

1. **Explicit contracts.**
   Each layer has a textual or binary contract and a verifier.

2. **One source of truth.**
   ISA semantics are normative and shared by Python and C++ simulators.

3. **No hidden state.**
   Allocation, scheduling, caching, and lowering decisions are explicit and
   reproducible.

4. **Layered fallback.**
   Software reference execution is a first-class backend, not test-only glue.

5. **Observability.**
   Every lowering, launch, memory transaction, and profiling event can be
   traced.

## 3. Proposed architecture

```text
PyTorch FX          ONNX
      |               |
      +-------+-------+
              v
      Frontend / Importer
              |
              v
      High-level MLIR
      (torch/tosa/linalg)
              |
              v
      Core compiler IR
      (typed tensors + explicit memory)
              |
              v
      mgpu device IR
      (launch, thread, shared memory, mma)
              |
              v
      ISA
              |
      +-------+-------+
      |               |
      v               v
  Functional      Cycle-approximate
  Simulator       Simulator
      |               |
      +-------+-------+
              v
      Golden outputs, traces, profiles
```

## 4. Layer contracts

### 4.1 Frontend

Responsibilities:

- import PyTorch FX and ONNX,
- preserve names and source locations,
- lower framework control flow,
- report unsupported ops with source context,
- emit verifiable high-level MLIR.

Contract:

- input: exported model,
- output: valid MLIR module,
- no target-specific decisions.

### 4.2 High-level optimizer

Responsibilities:

- shape inference,
- canonicalization,
- fusion,
- decomposition,
- dtype and layout propagation,
- liveness and memory planning.

Contract:

- input and output are verifiable MLIR,
- all transformations preserve a recorded golden trace.

### 4.3 Device lowering

Responsibilities:

- tiling,
- bufferization,
- address lowering,
- grid/block/warp mapping,
- shared-memory allocation,
- synchronization insertion,
- tensor-core legalization.

Contract:

- output contains only device ops and explicit synchronization,
- every shared-memory access has a known offset and bank behavior.

### 4.4 ISA backend

Responsibilities:

- instruction selection,
- register allocation,
- spilling,
- instruction scheduling,
- encoding/printing,
- assembler/disassembler,
- relocation/linking.

Contract:

- output parses with the normative ISA parser,
- semantics are exactly those in the ISA specification.

### 4.5 Simulator

Responsibilities:

- functional execution,
- cycle-approximate timing,
- memory hierarchy modeling,
- warp scheduling,
- profiling and tracing.

Contract:

- same ISA parser as the compiler backend,
- Python and C++ implementations agree on all valid programs,
- profiles are deterministic.

## 5. Core IR design

The core IR should be:

- MLIR-based,
- statically typed,
- explicit about memory space and layout,
- able to represent:

  - tensors,
  - memrefs,
  - shared memory,
  - local memory,
  - scalar control flow,
  - reductions,
  - tensor-core operations,
  - quantized dtypes.

Suggested dialect split:

```text
high-level compute: linalg/tosa
device mapping:     mgpu
execution:          isa
runtime metadata:   runtime
```

## 6. ISA design

Start from a normative ISA document before writing backend or simulator code.

The ISA must define:

- register file,
- special registers,
- data types,
- instruction encodings,
- operand rules,
- memory spaces,
- synchronization,
- control flow,
- tensor-core semantics,
- deterministic edge cases.

## 7. Simulator design

Split into:

- **functional core**: executes one warp/block without timing,
- **timing core**: adds scoreboard, pipeline, memory, and scheduler models,
- **memory system**: global, L2, shared, local, and async-copy models,
- **trace/profiler**: records events without affecting timing.

Use the same `Device`, `SM`, `Warp`, and `Lane` abstractions in both Python
and C++, with C++ as the production implementation and Python as the
reference.

## 8. Testing pyramid

```text
unit tests
  -> IR round-trip and FileCheck
  -> pass differential tests
  -> operator golden tests
  -> random program fuzzing
  -> model-zoo regression tests
  -> performance regression tests
```

Every layer has:

- deterministic golden output,
- failure reproducer,
- source-location diagnostics,
- performance counters where relevant.

## 9. Migration strategy

Do not delete the current repo. Build the redesign alongside it.

Recommended migration order:

1. write the ISA specification,
2. build a standalone ISA parser and functional simulator,
3. define the runtime device/command model,
4. define `mgpu` dialect verifier,
5. replace text emission with real MLIR passes,
6. move software reference kernels behind a stable backend API,
7. add C++ cycle-approximate simulator,
8. migrate real-model tests,
9. retire legacy text-generation paths.

## 10. Milestones

```text
R0: ISA spec + parser + functional simulator
R1: stable MLIR import/verify/reproducer
R2: memory model and runtime contracts
R3: full compiler pipeline for core ops
R4: cycle-approximate C++ simulator
R5: real CNN and GPT-2 execution
R6: quantization and tensor-core coverage
R7: fuzzing, CI, packaging, and release gate
```

## 11. Non-goals

- bit-exact NVIDIA/CUDA compatibility,
- full GPU virtualization,
- dynamic-shape support in the first redesign,
- automatic kernel synthesis from Python loops,
- exhaustive power/thermal modeling.
