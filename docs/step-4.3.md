# Step 4.3: Lower Tiled Matmul to Warp-Wide MMA

This document describes the Phase 4.3 implementation and the follow-up work
that fully implements the `MMA` instruction.

The matmul backend no longer uses a scalar `FMA` inner loop.  For a square
`16x16x16` matmul, it loads A and B fragments into the 32-lane warp and issues
one warp-wide `MMA` instruction:

```text
MMA R16, R0, R8, R16
```

The simulator expands that instruction to:

```text
D = A @ B + C
```

where A, B, C, and D are 16x16 fragments distributed over the warp according
to the fragment layout from Phase 4.2.  This is a real tensor-core execution
model, not a per-lane scalar fallback.

## 1. What changed

- Python reference matmul lowering in
  [../compiler/codegen/common_kernels.py](../compiler/codegen/common_kernels.py)
  emits one fragment-wide `MMA` instruction for a 16x16x16 tile.
- C++ `mgpu-to-isa` matmul path in
  [../mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp](../mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp)
  emits the same fragment-wide `MMA` code for a 16x16x16 `linalg.matmul`.
- The simulator recognizes `MMA` as a warp-wide opcode in
  [../hw/core.py](../hw/core.py) and implements `A @ B + C` over 32 lanes.
- The scoreboard and source-register analysis treat the A/B/C/D operands as
  eight-register fragment blocks rather than single scalar registers.

The `mgpu.mma` operation inserted by `lower-to-mgpu` remains the device-level
representation for tensor-core work.  The current executable matmul path is
already fragment-aware; later phases will use the same instruction for conv
and attention.

## 2. Tests

- Python matmul E2E tests in
  [../tests/operator_e2e/test_common_kernels_e2e.py](../tests/operator_e2e/test_common_kernels_e2e.py)
  and
  [../tests/unit/test_mgpu_to_isa.py](../tests/unit/test_mgpu_to_isa.py)
  verify the 16x16x16 matmul matches NumPy.
- [../tests/operator_e2e/test_mma_e2e.py](../tests/operator_e2e/test_mma_e2e.py)
  covers the full MMA instruction and rejects partial warps.
- The C++ matmul lit test in
  [../tests/mlir/mgpu_to_isa_matmul.mlir](../tests/mlir/mgpu_to_isa_matmul.mlir)
  checks for the fragment-wide `MMA R16, R0, R8, R16` instruction and the
  32-thread launch shape.

## 3. Files changed

- [../compiler/codegen/common_kernels.py](../compiler/codegen/common_kernels.py)
- [../hw/core.py](../hw/core.py)
- [../hw/isa.py](../hw/isa.py)
- [../mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp](../mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp)
- [../tests/operator_e2e/test_common_kernels_e2e.py](../tests/operator_e2e/test_common_kernels_e2e.py)
- [../tests/operator_e2e/test_mma_e2e.py](../tests/operator_e2e/test_mma_e2e.py)
- [../tests/unit/test_mgpu_to_isa.py](../tests/unit/test_mgpu_to_isa.py)
- [../tests/mlir/mgpu_to_isa_matmul.mlir](../tests/mlir/mgpu_to_isa_matmul.mlir)
- [../docs/step-008.md](../docs/step-008.md)
- [../docs/steps.md](../docs/steps.md)
