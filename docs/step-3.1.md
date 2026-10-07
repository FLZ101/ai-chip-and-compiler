# Step 3.1: Model-to-Linalg

This document describes the implementation of Phase 3.1 from
[next-stage-steps.md](next-stage-steps.md).

The `MLIRBuilder` now emits real `linalg` contracts for the high-level Model IR
ops required by CNN and Transformer/LLM workloads.

---

## 1. What was implemented

The following Model IR ops are now lowered to `linalg` instead of identity
placeholders:

- convolution:
  `linalg.conv_2d_nchw_fchw`
- max pooling:
  `linalg.pooling_nchw_max`
- average pooling:
  `linalg.pooling_nchw_sum`
- global average pooling:
  `linalg.generic` reduction followed by scalar division
- softmax:
  max reduction, subtract, `math.exp`, sum reduction, divide
- layer norm:
  mean/variance reductions and broadcast arithmetic
- RMS norm:
  mean-square reduction and broadcast arithmetic
- attention:
  `memref.collapse_shape`, `linalg.matmul_transpose_b`, softmax, and
  `linalg.matmul`

Non-zero convolution padding still uses the legacy identity placeholder because
explicit padding insertion belongs to the Phase 3.2 tiling pass.

---

## 2. Scope of the initial lowering

The initial lowering is intentionally static and explicit:

- softmax/layer norm/RMS norm support rank-2 inputs,
- attention supports rank-4 `Q/K/V` with batch and head dimensions equal to
  one,
- pooling requires zero padding.

These restrictions keep the first model-to-linalg pass verifiable. Dynamic
ranks and general batch/head layouts are deferred until the tiling pass has
canonical loops.

---

## 3. Tests

One lit test was added per op:

- [../tests/mlir/softmax.mlir](../tests/mlir/softmax.mlir)
- [../tests/mlir/layernorm.mlir](../tests/mlir/layernorm.mlir)
- [../tests/mlir/rmsnorm.mlir](../tests/mlir/rmsnorm.mlir)
- [../tests/mlir/attention.mlir](../tests/mlir/attention.mlir)
- [../tests/mlir/pooling.mlir](../tests/mlir/pooling.mlir)
- convolution is covered by the updated
  [../tests/mlir/conv2d.mlir](../tests/mlir/conv2d.mlir)

Each test runs:

```text
emit_graph.py -> FileCheck -> mgpu-opt parse/verify
```

Run:

```bash
source .venv/bin/activate
lit -q tests/mlir
```

---

## 4. Files changed

- [../compiler/mlir/mlir_builder.py](../compiler/mlir/mlir_builder.py)
- [../tests/mlir/emit_graph.py](../tests/mlir/emit_graph.py)
- [../tests/mlir/softmax.mlir](../tests/mlir/softmax.mlir)
- [../tests/mlir/layernorm.mlir](../tests/mlir/layernorm.mlir)
- [../tests/mlir/rmsnorm.mlir](../tests/mlir/rmsnorm.mlir)
- [../tests/mlir/attention.mlir](../tests/mlir/attention.mlir)
- [../tests/mlir/pooling.mlir](../tests/mlir/pooling.mlir)
- [../tests/mlir/conv2d.mlir](../tests/mlir/conv2d.mlir)
- [../tests/unit/test_mlir_builder.py](../tests/unit/test_mlir_builder.py)
