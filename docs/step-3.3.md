# Step 3.3: Bufferization

This document describes the Phase 3.3 implementation.

The compiler now uses MLIR's built-in `one-shot-bufferize` pass to convert
tensor-form `linalg` IR to `memref` form before target-specific lowering.

---

## 1. What was implemented

`mgpu-opt` now registers all MLIR passes in addition to all dialects. This
makes the upstream pass:

```text
one-shot-bufferize
```

available directly from the command line:

```bash
mgpu-opt input.mlir \
  --pass-pipeline='builtin.module(one-shot-bufferize)' \
  -o bufferized.mlir
```

For this project, bufferization intentionally reuses the upstream pass rather
than adding a custom `bufferize` pass, because one-shot bufferization already
handles allocation insertion, destination-style conversion, and function-boundary
policy.

---

## 2. Pipeline position

```text
model-to-linalg
      |
tile-linalg
      |
one-shot-bufferize
      |
map-to-mgpu
      |
lower-to-mgpu
      |
mgpu-to-isa
```

---

## 3. What is bufferization and why one-shot-bufferize

Bufferization converts value-semantics tensor IR into buffer/memory IR:

```text
tensor<2x4xf32>          -> memref<2x4xf32>
tensor.empty()           -> memref.alloc()
tensor.insert_slice      -> memref.subview + memref.copy
```

It matters here because `linalg` on tensors is functional and easy to analyze,
while the simulator target needs explicit memory operations and eventually ISA
loads/stores.

`one-shot-bufferize` is used because it is the modern upstream MLIR
bufferization pass:

- destination-style aware, matching `linalg`,
- inserts allocations and copies automatically,
- handles liveness analysis and `memref` type conversion,
- supports function-boundary bufferization when needed,
- is maintained upstream rather than duplicating the older deprecated
  `std-bufferize` flow.

For this project, reusing the built-in pass avoids reimplementing allocation,
copy, and liveness logic. The compiler only places it in the pipeline:

```text
linalg-on-tensors -> one-shot-bufferize -> memref/linalg -> mgpu lowering
```

---

## 4. Tests

Added [../tests/mlir/bufferize.mlir](../tests/mlir/bufferize.mlir). The test
bufferizes a small `tensor.empty` + `linalg.fill` function and verifies:

- a `memref.alloc` appears,
- no `tensor<` type remains,
- the `linalg.fill` now operates on a `memref`.

Run:

```bash
source .venv/bin/activate
lit -q tests/mlir
```

---

## 5. Files changed

- [../mlir/mgpu/tools/mgpu-opt/mgpu-opt.cpp](../mlir/mgpu/tools/mgpu-opt/mgpu-opt.cpp)
- [../tests/mlir/bufferize.mlir](../tests/mlir/bufferize.mlir)
