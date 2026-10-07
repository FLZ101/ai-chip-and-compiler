# Step 3.6: mgpu-to-isa

Phase 3.6 provides two complementary ISA paths:

- a Python reference lowering that runs validated Model IR in the simulator,
- a C++ `mgpu-to-isa` pass in `mgpu-opt`.

---

## 1. Implemented

Python:

- [../compiler/mlir/mgpu_to_isa.py](../compiler/mlir/mgpu_to_isa.py)
- verifies `mgpu.launch` is present
- dispatches to the Model IR ISA generators
- returns ISA text plus launch dimensions

C++:

- [../mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp](../mlir/mgpu/lib/mgpu/MGpuToIsaPass.cpp)
- lowers vector-add and square-tiled matmul launches
- handles all `mgpu.*` ops with mappings or placeholders
- emits an `mgpu.isa` op

Allocator support:

- virtual register allocation
- linear-scan-style physical register reuse
- spill-store emission

---

## 2. Tests

- [../tests/unit/test_mgpu_to_isa.py](../tests/unit/test_mgpu_to_isa.py)
  - vector add and matmul match NumPy
- MLIR lit tests:
  - [../tests/mlir/mgpu_to_isa_cpp.mlir](../tests/mlir/mgpu_to_isa_cpp.mlir)
  - [../tests/mlir/mgpu_to_isa_matmul.mlir](../tests/mlir/mgpu_to_isa_matmul.mlir)
  - [../tests/mlir/mgpu_to_isa_all_ops.mlir](../tests/mlir/mgpu_to_isa_all_ops.mlir)
  - [../tests/mlir/mgpu_to_isa_spill.mlir](../tests/mlir/mgpu_to_isa_spill.mlir)

---

## 3. Relationship to target backends

MLIR has mature RISC-V/x86/CUDA paths:

```text
MLIR -> LLVM Dialect -> LLVM IR -> llc -> target assembly
MLIR -> NVVM Dialect -> LLVM/NVPTX -> PTX -> ptxas
```

Those paths reuse LLVM regalloc, scheduling, spilling, and ABI handling.

Our custom simulator ISA is not LLVM/NVPTX/x86/RISC-V, so LLVM's allocator is
not directly reusable. We instead implement the same backend concepts for the
custom textual ISA.

---

## 4. Future enhancements

- liveness intervals for general SSA code
- instruction scheduling for cycle-accurate performance
- binary encoding when an assembler/hardware decoder is added

Priority:

```text
1. ISA printing        -> implemented
2. virtual registers   -> implemented
3. spilling            -> scaffold implemented
4. liveness intervals  -> future
5. instruction scheduling -> future
6. binary encoding     -> future
```
