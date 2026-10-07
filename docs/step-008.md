# Step 008: Common Kernels

This document describes the first production ISA kernels.

Step 008 adds a small shared compute backbone under
[../compiler/codegen](../compiler/codegen):

- rank-1 elementwise add/sub/mul/div,
- rank-1 sum/mean/max reductions,
- square tiled matrix multiplication.

All kernels use the shared operator E2E harness from Step 007.

---

## 1. Lowering API

File: [../compiler/codegen/common_kernels.py](../compiler/codegen/common_kernels.py)

`lower_common_kernel` dispatches on the single Model IR node:

```python
from compiler.codegen import lower_common_kernel

isa_text, (grid_dim, block_dim) = lower_common_kernel(graph, mlir_text, layout)
```

The lowering functions are intentionally small and deterministic.  They return
the ISA text and launch dimensions expected by the shared harness.

---

## 2. Elementwise kernels

The elementwise path uses one 32-thread block and maps:

```text
global_tid = ctaid.x * ntid.x + tid.x
```

Out-of-range lanes are skipped with a predicated branch.  The same code
generator handles `add`, `sub`, `mul`, and `div`.

---

## 3. Reduction kernels

Reductions use one block whose size is the next power of two after the input
length.  Each thread loads one value into shared memory, then performs a
shared-memory tree reduction with `BAR.SYNC` between levels:

- `REDUCE_SUM` uses `ADD`,
- `REDUCE_MEAN` uses `ADD` followed by a final `DIV`,
- `REDUCE_MAX` uses `MAX` with a very negative identity.

Only thread 0 writes the scalar result.

---

## 4. Tiled matmul

The initial matmul kernel supports square matrices where:

- `M == N`,
- `K` is divisible by `N`.

It launches one block with shape `(N, M)`:

- `tid.x` selects a column,
- `tid.y` selects a row,
- `K` is traversed in shared-memory tiles of size `N`.

For each tile, every thread cooperatively loads one `A` element and one `B`
element into shared memory, synchronizes, and accumulates its output element
with `FMA`.  This demonstrates shared-memory tiling and register accumulation;
rectangular and multi-block tiling can be added later.

---

## 5. MLIR builder update

[../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py) now emits
`linalg.generic` reduction text for the reduction ops, so the Step 007 harness
can run reductions through the full Model IR to MLIR to ISA pipeline.

---

## 6. Files changed

- [../compiler/codegen/__init__.py](../compiler/codegen/__init__.py)
- [../compiler/codegen/common_kernels.py](../compiler/codegen/common_kernels.py)
- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../tests/operator_e2e/test_common_kernels_e2e.py](../tests/operator_e2e/test_common_kernels_e2e.py)

---

## 7. Tests

New tests cover:

- elementwise add/sub/mul/div,
- reduction sum/mean/max,
- tiled matmul with `FMA` and shared memory.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.
