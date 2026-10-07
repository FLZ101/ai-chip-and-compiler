# Step 3.5: Lower to mgpu

This document describes the Phase 3.5 implementation.

The compiler now inserts the target-specific `mgpu` memory and synchronization
operations into mapped launch bodies.

---

## 1. What was implemented

Added `lower-to-mgpu`, a `func::FuncOp` pass in
[../mlir/mgpu/lib/mgpu/LowerToMGPUPass.cpp](../mlir/mgpu/lib/mgpu/LowerToMGPUPass.cpp).

The pass:

- inserts `mgpu.barrier` at the start and end of every `mgpu.launch`,
- inserts an `mgpu.shared_alloc` plus `mgpu.async_copy` after each
  `memref.subview` tile,
- inserts a reserved `mgpu.mma` before each `linalg.matmul` /
  `linalg.matmul_transpose_b`.

The `mgpu.mma` op remains a placeholder until Phase 4 executes tensor-core MMA
in the simulator.

---

## 2. Usage

```bash
mgpu-opt input.mlir \
  --pass-pipeline='builtin.module(func.func(lower-to-mgpu))' \
  -o lowered.mlir
```

---

## 3. What the pass does

`lower-to-mgpu` is the target-specific lowering pass that inserts `mgpu`
device operations into mapped kernels.

For every `mgpu.launch`, it inserts synchronization:

```text
mgpu.barrier
...
mgpu.barrier
```

`mgpu.launch` represents one thread block. Its operations may be executed by
multiple warps that share data through shared memory. The leading barrier
ensures all threads have arrived before shared-memory tiles are read. The
trailing barrier ensures all shared/global writes finish before the block exits
and another launch reads them. This is the `mgpu` equivalent of CUDA's
`__syncthreads()`:

```text
mgpu.launch {
  mgpu.barrier
  copy global -> shared
  compute on shared
  mgpu.barrier
  mgpu.return
}
```

For every `memref.subview` tile, it inserts a shared-memory staging sequence:

```text
mgpu.shared_alloc
mgpu.async_copy
```

After tiling, a `memref.subview` is a tile of a larger global-memory buffer.
Shared memory is much closer to the compute units, so the pass allocates a
shared-memory staging buffer with `mgpu.shared_alloc`, then issues an
`mgpu.async_copy` to move the global tile into shared memory before compute:

```text
global tile
      |
      v
mgpu.shared_alloc
      |
      v
mgpu.async_copy
      |
      v
compute reads shared tile
```

This is the `mgpu` analogue of CUDA shared-memory staging:

```cpp
__shared__ float tile[TILE_SIZE];
copy_global_to_shared(tile, global_tile);
__syncthreads();
```

For every matmul-style linalg operation, it inserts a reserved tensor-core op:

```text
mgpu.mma
```

The target has a future tensor-core matrix-multiply-accumulate unit. At this
stage, `mgpu.mma` is inserted as a reserved target op marking where tensor-core
execution will happen. The original `linalg.matmul` /
`linalg.matmul_transpose_b` remains until Phase 4 implements functional MMA in
the simulator.

```text
linalg.matmul
      |
      v
mgpu.mma (reserved)
      |
      v
Phase 4 simulator execution
```

The `mgpu.mma` remains a placeholder until Phase 4 executes tensor-core MMA
in the simulator.

---

## 4. Pipeline placement

`lower-to-mgpu` belongs after loops have been mapped to the `mgpu` execution
model and before final ISA emission:

```text
torch/ONNX model
      |
      v
torch-mlir import + canonicalization
      |
      v
linalg-on-tensors
      |
      v
tile-linalg
      |
      v
one-shot-bufferize
      |
      v
map-to-mgpu
      |
      v
lower-to-mgpu      <-- this pass
      |
      v
mgpu-to-isa
```

It must run after tiling and bufferization because it operates on
`memref.subview` tiles. It must run after `map-to-mgpu` because it inserts into
`mgpu.launch` bodies. It must run before `mgpu-to-isa`, whose output is
simulator ISA rather than `mgpu.*` operations.

---

## 5. Tests

Added [../tests/mlir/lower_to_mgpu.mlir](../tests/mlir/lower_to_mgpu.mlir). The
test verifies that a launch containing a subview and matmul receives:

- barriers,
- a shared allocation,
- an async copy,
- a reserved `mgpu.mma`.

Run:

```bash
source .venv/bin/activate
lit -q tests/mlir
```

---

## 6. Files changed

- [../mlir/mgpu/include/mgpu/MGPUPasses.h](../mlir/mgpu/include/mgpu/MGPUPasses.h)
- [../mlir/mgpu/lib/mgpu/LowerToMGPUPass.cpp](../mlir/mgpu/lib/mgpu/LowerToMGPUPass.cpp)
- [../mlir/mgpu/lib/mgpu/MGPUPasses.cpp](../mlir/mgpu/lib/mgpu/MGPUPasses.cpp)
- [../mlir/mgpu/lib/mgpu/CMakeLists.txt](../mlir/mgpu/lib/mgpu/CMakeLists.txt)
- [../tests/mlir/lower_to_mgpu.mlir](../tests/mlir/lower_to_mgpu.mlir)
