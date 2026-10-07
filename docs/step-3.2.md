# Step 3.2: Linalg Tiling in C++

This document describes the Phase 3.2 implementation.

The first target-specific MLIR pass now lives in the out-of-tree `mgpu` C++
dialect and runs inside `mgpu-opt`.

---

## 1. What was implemented

Added `tile-linalg`, a `func::FuncOp` pass in
[../mlir/mgpu/lib/mgpu/TileLinalgPass.cpp](../mlir/mgpu/lib/mgpu/TileLinalgPass.cpp).

The pass:

- walks buffer-form `linalg` operations,
- chooses target tile sizes:
  - `16x16x16` for matmul/attention matmuls,
  - `1x1x8x8` for convolution and pooling,
  - `16x16` for rank-2 reductions and elementwise ops,
  - `16x16x16` otherwise,
- invokes MLIR's `linalg::tileLinalgOp`,
- leaves tensor-form operations until after Phase 3.3 bufferization.

The pass is registered in `mgpu-opt` and is exposed as:

```bash
mgpu-opt input.mlir \
  --pass-pipeline='builtin.module(func.func(tile-linalg))' \
  -o tiled.mlir
```

---

## 2. Why C++

`linalg::tileLinalgOp` already handles:

- `scf.for` nest construction,
- boundary `affine.min` computations,
- `memref.subview` creation,
- destination-style operand updates.

Reimplementing this in Python would duplicate a large amount of MLIR
infrastructure. The C++ pass uses the real MLIR tiling implementation directly.

---

## 3. Tests

New lit tests run generated Model IR through `mgpu-opt` and FileCheck for
`scf.for`, `memref.subview`, and the tiled linalg op:

- [../tests/mlir/tile_matmul.mlir](../tests/mlir/tile_matmul.mlir)
- [../tests/mlir/tile_conv.mlir](../tests/mlir/tile_conv.mlir)
- [../tests/mlir/tile_reduction.mlir](../tests/mlir/tile_reduction.mlir)
- [../tests/mlir/tile_attention.mlir](../tests/mlir/tile_attention.mlir)

Run:

```bash
source .venv/bin/activate
lit -q tests/mlir
```

---

## 4. Files changed

- [../mlir/mgpu/include/mgpu/MGPUPasses.h](../mlir/mgpu/include/mgpu/MGPUPasses.h)
- [../mlir/mgpu/lib/mgpu/TileLinalgPass.cpp](../mlir/mgpu/lib/mgpu/TileLinalgPass.cpp)
- [../mlir/mgpu/lib/mgpu/CMakeLists.txt](../mlir/mgpu/lib/mgpu/CMakeLists.txt)
- [../mlir/mgpu/tools/mgpu-opt/mgpu-opt.cpp](../mlir/mgpu/tools/mgpu-opt/mgpu-opt.cpp)
- [../tests/mlir/tile_matmul.mlir](../tests/mlir/tile_matmul.mlir)
- [../tests/mlir/tile_conv.mlir](../tests/mlir/tile_conv.mlir)
- [../tests/mlir/tile_reduction.mlir](../tests/mlir/tile_reduction.mlir)
- [../tests/mlir/tile_attention.mlir](../tests/mlir/tile_attention.mlir)
