# Step 006: MLIR Builder and `mgpu` Dialect

This document describes the second compiler milestone.

Step 006 lowers a small, explicit subset of the unified Model IR into
deterministic MLIR text.  The output combines standard MLIR dialects with the
custom `mgpu` device dialect.

---

## 1. MLIR builder

File: [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)

`MLIRBuilder` converts a validated `Graph` into a textual MLIR module:

```python
from compiler.frontends.manual_frontend import ManualFrontend
from compiler.mlir import MLIRBuilder
from compiler.model_ir import DType

frontend = ManualFrontend("vector_add")
a = frontend.add_input("a", DType.F32, (8,))
b = frontend.add_input("b", DType.F32, (8,))
frontend.output(frontend.add(a, b))

print(MLIRBuilder().build(frontend.build()))
```

The builder currently emits:

- `func.func` with `memref` arguments and return values,
- `arith.constant` index constants,
- `memref.alloc` for each intermediate result,
- `linalg.generic` for elementwise add/sub/mul/div,
- `linalg.matmul` for matrix multiplication,
- `linalg.conv_2d_nchw_fchw` for convolution with no padding.

Each compute operation is wrapped in an `mgpu.launch` region containing:

- `mgpu.thread_id` for x/y/z,
- `mgpu.block_id` for x/y/z,
- `mgpu.grid_dim` for x/y/z,
- `mgpu.block_dim` for x/y/z,
- `mgpu.barrier`,
- `mgpu.return`.

Unsupported Model IR ops fail with `MLIRBuilderError` rather than emitting
silently wrong text.

---

## 2. `mgpu` dialect

File: [../compiler/mlir/mgpu_dialect.td](../compiler/mlir/mgpu_dialect.td)

The TableGen file defines the custom device dialect:

- `mgpu.launch`
- `mgpu.thread_id`
- `mgpu.block_id`
- `mgpu.grid_dim`
- `mgpu.block_dim`
- `mgpu.shared_alloc`
- `mgpu.barrier`
- `mgpu.async_copy`
- `mgpu.mma`
- `mgpu.return`

`mgpu.mma` is reserved for the future tensor-core path.  Shared-memory tiling,
async copy, and `mgpu.mma` lowering are introduced by later passes.

---

## 3. Pass pipeline

File: [../compiler/mlir/passes.py](../compiler/mlir/passes.py)

The initial named pipeline is:

```text
model-to-linalg -> tile-linalg -> linalg-to-mgpu -> mgpu-to-isa
```

The transformations are implemented in later steps.  This step establishes the
names and order.

---

## 4. Files changed

- [../compiler/mlir/__init__.py](../compiler/mlir/__init__.py)
- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../compiler/mlir/passes.py](../compiler/mlir/passes.py)
- [../compiler/mlir/mgpu_dialect.td](../compiler/mlir/mgpu_dialect.td)
- [../tests/unit/test_mlir_builder.py](../tests/unit/test_mlir_builder.py)
- [../tests/unit/test_mlir_passes.py](../tests/unit/test_mlir_passes.py)

---

## 5. Tests

New tests cover:

- deterministic vector-add MLIR,
- matmul `linalg.matmul`,
- conv `linalg.conv_2d_nchw_fchw`,
- `mgpu` launch/thread/block/grid/block-dim/barrier emission,
- unsupported op rejection,
- `mgpu_dialect.td` device-op definitions,
- default pass pipeline order and descriptions.

Run:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

All existing tests continue to pass.

---

## 6. Lit tests

The MLIR text output is also checked with LLVM-style lit tests in
[../tests/mlir](../tests/mlir):

- [vector_add.mlir](../tests/mlir/vector_add.mlir)
- [matmul.mlir](../tests/mlir/matmul.mlir)
- [conv2d.mlir](../tests/mlir/conv2d.mlir)

Each test runs `emit_graph.py` through the `filecheck` clone and verifies the
expected `func`, `linalg`, and `mgpu` lines.

Run:

```bash
source .venv/bin/activate
lit -v tests/mlir
```
