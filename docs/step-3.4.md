# Step 3.4: Grid/Block/Warp Mapping

This document describes the Phase 3.4 implementation.

The C++ compiler now maps top-level `scf.for` loop nests into the `mgpu`
execution model by wrapping them in `mgpu.launch` and exposing the device index
operations.

---

## 1. What was implemented

Added `map-to-mgpu`, a `func::FuncOp` pass in
[../mlir/mgpu/lib/mgpu/MapToMGPUPass.cpp](../mlir/mgpu/lib/mgpu/MapToMGPUPass.cpp).

For each top-level `scf.for` loop:

- creates an `mgpu.launch` with:
  - grid `1x1x1`,
  - thread block `32x1x1`,
- moves the loop into the launch body,
- inserts:
  - `mgpu.thread_id`
  - `mgpu.block_id`
  - `mgpu.grid_dim`
  - `mgpu.block_dim`
- terminates the body with `mgpu.return`.

The launch dimensions are constants, so the output is deterministic for a
given input.

---

## 2. Usage

```bash
mgpu-opt input.mlir \
  --pass-pipeline='builtin.module(func.func(map-to-mgpu))' \
  -o mapped.mlir
```

---

## 3. Tests

Added [../tests/mlir/map_to_mgpu.mlir](../tests/mlir/map_to_mgpu.mlir). The test
maps a single top-level `scf.for` loop and verifies:

- `mgpu.launch`
- all four device index ops
- the original `scf.for` remains in the launch body
- `mgpu.return`

Run:

```bash
source .venv/bin/activate
lit -q tests/mlir
```

---

## 4. Why launch dimensions are currently fixed

The current pass uses fixed launch dimensions:

```text
grid    1x1x1
threads 32x1x1
```

This is the first deterministic mapping stage. It proves that top-level loops
can be wrapped in `mgpu.launch` and that the device index operations are placed
correctly, without yet implementing launch-geometry analysis.

The dimensions can be determined by the compiler later. The natural source is
the tiled `scf.for` structure:

```text
scf.for %gx = 0 to grid_x step 1
  scf.for %gy = 0 to grid_y step 1
    scf.for %tx = 0 to block_x step 1
      ...
```

The compiler can:

1. identify the parallel outer loops,
2. compute their trip counts,
3. partition those loops into grid and block dimensions,
4. emit the corresponding `mgpu.launch blocks(...) threads(...)`.

For example, tiled loops with ranges:

```text
outer = 4
next  = 1
inner = 32
```

would map to:

```text
grid 4x1x1
threads 32x1x1
```

Once the mapping pass consumes the tiled loop structure, launch dimensions will
be compiler-derived instead of hardcoded.

---

## 5. Files changed

- [../mlir/mgpu/include/mgpu/MGPUPasses.h](../mlir/mgpu/include/mgpu/MGPUPasses.h)
- [../mlir/mgpu/lib/mgpu/MapToMGPUPass.cpp](../mlir/mgpu/lib/mgpu/MapToMGPUPass.cpp)
- [../mlir/mgpu/lib/mgpu/MGPUPasses.cpp](../mlir/mgpu/lib/mgpu/MGPUPasses.cpp)
- [../mlir/mgpu/lib/mgpu/CMakeLists.txt](../mlir/mgpu/lib/mgpu/CMakeLists.txt)
- [../tests/mlir/map_to_mgpu.mlir](../tests/mlir/map_to_mgpu.mlir)
